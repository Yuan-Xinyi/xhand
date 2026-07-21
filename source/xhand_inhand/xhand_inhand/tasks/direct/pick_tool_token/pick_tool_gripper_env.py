"""Physical xArm7 parallel-gripper ablation for the PickTool sequence."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from ..pick_cube.pick_cube_env import PickCubeEnv
from .grasp_signals import rigid_hold_quality, update_grasp_latch
from .pick_tool_gripper_env_cfg import PickToolGripperEnvCfg


class PickToolGripperEnv(PickCubeEnv):
    cfg: PickToolGripperEnvCfg

    def __init__(self, cfg: PickToolGripperEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev, n = self.device, self.num_envs
        self._arm_ids_t = torch.as_tensor(self._arm_joint_ids, dtype=torch.long, device=dev)
        self._finger_ids_t = torch.as_tensor(self._hand_joint_ids, dtype=torch.long, device=dev)

        from scipy.spatial import ConvexHull

        from .tool_asset import TOOL_OBJ, TOOL_SCALE

        vertices = []
        with open(TOOL_OBJ) as stream:
            for line in stream:
                if line.startswith("v "):
                    values = line.split()
                    vertices.append((float(values[1]), float(values[2]), float(values[3])))
        vertices_t = torch.tensor(vertices, dtype=torch.float, device=dev) * torch.tensor(
            TOOL_SCALE, device=dev
        )
        hull = ConvexHull(vertices_t.cpu().numpy()).vertices
        self._obj_hull_local = vertices_t[torch.as_tensor(hull, device=dev)]
        rest_quat = torch.tensor(cfg.object_cfg.init_state.rot, dtype=torch.float, device=dev)
        rest_z = float(cfg.object_cfg.init_state.pos[2])
        rest_world_z = quat_apply(
            rest_quat.expand(self._obj_hull_local.shape[0], 4), self._obj_hull_local
        )[:, 2]
        self._table_surface_z = (rest_world_z + rest_z).min()
        self._handle_center_local = torch.tensor(cfg.handle_center, dtype=torch.float, device=dev)

        self._contact_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._lost_contact_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._is_grasped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._grasp_bonus_given = torch.zeros(n, dtype=torch.bool, device=dev)
        self._success_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._success_paid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_close_quality = torch.zeros(n, device=dev)
        self._prev_contact_quality = torch.zeros(n, device=dev)
        self._prev_lift_potential = torch.zeros(n, device=dev)
        self._potential_initialized = torch.zeros(n, dtype=torch.bool, device=dev)
        self._unlatched_lift_failure = torch.zeros(n, dtype=torch.bool, device=dev)
        self._policy_obs_cache = None
        self._compute_intermediate_values()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._jaw_contact_sensors: dict[str, ContactSensor] = {}
        for name in self.cfg.ee_body_names:
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{name}",
                    filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                )
            )
            self._jaw_contact_sensors[name] = sensor
            self.scene.sensors[f"object_contact_{name}"] = sensor

        table_spawn = sim_utils.UsdFileCfg(usd_path=self.cfg.table_usd)
        table_spawn.func(
            "/World/envs/env_.*/Table",
            table_spawn,
            translation=self.cfg.table_pos,
            orientation=self.cfg.table_rot,
        )
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05)
        )
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _jaw_forces(self) -> torch.Tensor:
        forces = []
        for name in self.ee_names:
            matrix = self._jaw_contact_sensors[name].data.force_matrix_w
            if matrix is None:
                raise RuntimeError(f"filtered jaw contact unavailable for {name}")
            forces.append(matrix.norm(dim=-1).sum(dim=(1, 2)))
        return torch.stack(forces, dim=1)

    def _object_true_min_z(self) -> torch.Tensor:
        n, m = self.num_envs, self._obj_hull_local.shape[0]
        world = quat_apply(
            self.object_quat_w.unsqueeze(1).expand(-1, m, -1).reshape(-1, 4),
            self._obj_hull_local.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3),
        ).reshape(n, m, 3)
        return (world[..., 2] + self.object_pos_w[:, 2:3]).min(dim=1).values

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        if hasattr(self, "_handle_center_local"):
            self.handle_center_w = self.object_pos_w + quat_apply(
                self.object_quat_w, self._handle_center_local.expand(self.num_envs, -1)
            )

    def _gripper_signals(self) -> dict[str, torch.Tensor]:
        forces = self._jaw_forces()
        strength = torch.tanh(forces / self.cfg.gripper_force_saturation)
        contact = forces >= self.cfg.gripper_contact_force_thr
        hold, slip_lin, slip_ang = rigid_hold_quality(
            self.robot.data.body_com_pos_w[:, self.palm_idx],
            self.robot.data.body_com_lin_vel_w[:, self.palm_idx],
            self.robot.data.body_com_ang_vel_w[:, self.palm_idx],
            self.object.data.root_com_pos_w,
            self.object.data.root_com_lin_vel_w,
            self.object.data.root_com_ang_vel_w,
            self.cfg.hold_v_scale,
            self.cfg.hold_w_scale,
        )
        midpoint = self.finger_pad_w.mean(dim=1)
        distance = torch.norm(midpoint - self.handle_center_w, dim=-1)
        proximity = torch.exp(-distance / self.cfg.gripper_proximity_scale)
        max_force = forces.max(dim=1).values
        safe_contact = torch.clamp(
            (self.cfg.gripper_terminate_force - max_force)
            / (self.cfg.gripper_terminate_force - self.cfg.gripper_safe_force),
            0.0,
            1.0,
        )
        contact_quality = strength.min(dim=1).values * safe_contact
        quality = torch.minimum(contact_quality, hold) * contact.all(dim=1).float()
        close_quality = proximity
        return {
            "forces": forces,
            "contact": contact,
            "hold": hold,
            "slip_lin": slip_lin,
            "slip_ang": slip_ang,
            "proximity": proximity,
            "contact_quality": contact_quality,
            "quality": quality,
            "close_quality": close_quality,
        }

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        action = actions.clone().clamp(-1.0, 1.0)
        self.actions = action
        raw = self.dof_targets.clone()
        raw[:, self._arm_ids_t] = self.dof_targets[:, self._arm_ids_t] + (
            self.cfg.action_scale * action[:, :7]
        )
        # +1 closes, -1 opens. Both physical finger joints use positive q outward.
        width = self.cfg.gripper_closed_width + (
            self.cfg.gripper_open_width - self.cfg.gripper_closed_width
        ) * 0.5 * (1.0 - action[:, 7:8])
        raw[:, self._finger_ids_t] = width.expand(-1, len(self._hand_joint_ids))
        raw = torch.clamp(raw, self.dof_lower, self.dof_upper)
        self.dof_targets = self.cfg.act_moving_average * raw + (
            1.0 - self.cfg.act_moving_average
        ) * self.dof_targets
        self.dof_targets = torch.clamp(self.dof_targets, self.dof_lower, self.dof_upper)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        signals = self._gripper_signals()
        clearance = self._object_true_min_z() - self._table_surface_z
        obs = torch.cat(
            (
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                self.ee_pos_b,
                self.palm_center_b,
                self.object_pos_b,
                self.object_quat_w,
                self.target_pos,
                self.target_quat,
                self.actions,
                torch.tanh(signals["forces"] / self.cfg.gripper_force_saturation),
                signals["quality"].unsqueeze(-1),
                self._is_grasped.float().unsqueeze(-1),
                torch.clamp(clearance / self.cfg.lift_success_height, 0.0, 1.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        if obs.shape[1] != self.cfg.observation_space:
            raise RuntimeError(
                f"built {obs.shape[1]} gripper observations, expected {self.cfg.observation_space}"
            )
        self._policy_obs_cache = obs
        return {"policy": obs, "critic": obs}

    def _get_states(self) -> torch.Tensor:
        return self._policy_obs_cache

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        signals = self._gripper_signals()
        quality = signals["quality"]
        self._is_grasped, self._contact_steps, self._lost_contact_steps, _, _ = update_grasp_latch(
            quality,
            self._is_grasped,
            self._contact_steps,
            self._lost_contact_steps,
            high_threshold=cfg.gripper_quality_high,
            low_threshold=cfg.gripper_quality_low,
            confirm_steps=cfg.gripper_confirm_steps,
            release_steps=cfg.gripper_release_steps,
        )
        max_force = signals["forces"].max(dim=1).values
        first_grasp = self._is_grasped & (~self._grasp_bonus_given) & (
            max_force <= cfg.gripper_safe_force
        )
        self._grasp_bonus_given |= first_grasp

        clearance = self._object_true_min_z() - self._table_surface_z
        lift_fraction = torch.clamp(clearance / cfg.lift_success_height, 0.0, 1.0)
        lift_potential = lift_fraction * self._is_grasped.float() * quality
        ready = self._potential_initialized.float()
        r_close = cfg.gripper_close_progress_scale * (
            cfg.shaping_discount * signals["close_quality"] - self._prev_close_quality
        ) * ready
        r_contact = cfg.gripper_contact_progress_scale * (
            cfg.shaping_discount * signals["contact_quality"] - self._prev_contact_quality
        ) * ready
        r_grasp = cfg.grasp_bonus * first_grasp.float()
        r_lift = cfg.lift_progress_scale * (
            cfg.shaping_discount * lift_potential - self._prev_lift_potential
        ) * ready
        newly_successful = self._is_success & (~self._success_paid)
        r_success = cfg.success_bonus * newly_successful.float()
        self._success_paid |= self._is_success
        force_excess = torch.clamp(
            (max_force - cfg.gripper_safe_force)
            / max(cfg.gripper_terminate_force - cfg.gripper_safe_force, 1.0e-6),
            0.0,
            1.0,
        )
        r_force = -cfg.force_excess_penalty_scale * force_excess.square()

        self._prev_close_quality.copy_(signals["close_quality"])
        self._prev_contact_quality.copy_(signals["contact_quality"])
        self._prev_lift_potential.copy_(lift_potential)
        self._potential_initialized.fill_(True)
        log = self.extras.setdefault("log", {})
        log["gripper_proximity_mean"] = signals["proximity"].mean()
        log["gripper_two_contact_frac"] = signals["contact"].all(dim=1).float().mean()
        log["gripper_force_max"] = max_force.max()
        log["grasp_quality_mean"] = quality.mean()
        log["is_grasped_phase_frac"] = self._is_grasped.float().mean()
        log["clearance_max"] = clearance.max()
        log["lift_ge_5cm_frac"] = (clearance >= 0.05).float().mean()
        log["lift_ge_20cm_frac"] = (clearance >= 0.20).float().mean()
        log["success_frac"] = self._is_success.float().mean()
        return r_close + r_contact + r_grasp + r_lift + r_success + r_force

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        cfg = self.cfg
        signals = self._gripper_signals()
        clearance = self._object_true_min_z() - self._table_surface_z
        max_force = signals["forces"].max(dim=1).values
        slow = (self.object.data.root_com_lin_vel_w.norm(dim=-1) < cfg.success_max_obj_lin_speed) & (
            self.object.data.root_com_ang_vel_w.norm(dim=-1) < cfg.success_max_obj_ang_speed
        )
        success_inst = (
            (clearance >= cfg.lift_success_height)
            & self._is_grasped
            & (signals["quality"] >= cfg.gripper_quality_high)
            & slow
            & (max_force <= cfg.gripper_safe_force)
        )
        self._success_steps = torch.where(
            success_inst, self._success_steps + 1, torch.zeros_like(self._success_steps)
        )
        self._is_success |= self._success_steps >= cfg.success_hold_steps
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - cfg.drop_height)
        unsafe = max_force > cfg.gripper_terminate_force
        self._unlatched_lift_failure.copy_(
            (clearance >= cfg.unlatched_lift_failure_height) & (~self._is_grasped)
        )
        failure = (dropped | unsafe | self._unlatched_lift_failure) & (~self._is_success)
        terminated = failure | self._is_success
        time_out = (self.episode_length_buf >= self.max_episode_length - 1) & (~terminated)
        self.extras["pick_tool_terminal"] = {
            "success": self._is_success.clone(),
            "failure": failure.clone(),
            "time_out": time_out.clone(),
            "dropped": dropped.clone(),
            "unsafe_force": unsafe.clone(),
            "unlatched_clearance_ge_5cm": self._unlatched_lift_failure.clone(),
            "true_clearance": clearance.clone(),
            "is_grasped": self._is_grasped.clone(),
            "grasp_quality": signals["quality"].clone(),
            "hold_quality": signals["hold"].clone(),
            "max_force": max_force.clone(),
            "success_steps": self._success_steps.clone(),
        }
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._contact_steps[ids] = 0
        self._lost_contact_steps[ids] = 0
        self._is_grasped[ids] = False
        self._grasp_bonus_given[ids] = False
        self._success_steps[ids] = 0
        self._success_paid[ids] = False
        self._prev_close_quality[ids] = 0.0
        self._prev_contact_quality[ids] = 0.0
        self._prev_lift_potential[ids] = 0.0
        self._potential_initialized[ids] = False
        self._unlatched_lift_failure[ids] = False
