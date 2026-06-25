# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

from xhand_inhand.utils import apply_fingertip_friction

from .pick_cube_env_cfg import PickCubeEnvCfg


class PickCubeEnv(DirectRLEnv):
    cfg: PickCubeEnvCfg

    def __init__(self, cfg: PickCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # reach assembly: the 5 fingertips + the palm-center point
        self.ee_ids, _ = self.robot.find_bodies(self.cfg.ee_body_names)
        self.palm_idx = self.robot.body_names.index(self.cfg.palm_body_name)
        self.palm_center_offset = torch.tensor(
            self.cfg.palm_center_offset, dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        # per-finger pad offset, aligned to the ACTUAL resolved ee_ids order (find_bodies may
        # reorder), so the green markers land on the pad surfaces, not the distal tips or joints.
        self.ee_names = [self.robot.body_names[i] for i in self.ee_ids]
        self.finger_pad_offset = torch.tensor(
            [self.cfg.finger_pad_offsets[n] for n in self.ee_names], dtype=torch.float, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)  # (N, 5, 3)

        # joint limits / defaults
        limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower = limits[..., 0]
        self.dof_upper = limits[..., 1]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        # per-shape friction (SimToolReal): all robot shapes low, the 5 fingertip distal
        # links high, so non-fingerpad grasps (knuckle / finger-gap / palm scoop) are
        # physically unprofitable and the policy converges to fingerpad grasping.
        apply_fingertip_friction(
            self.robot, self.cfg.ee_body_names,
            robot_friction=self.cfg.robot_friction,
            fingertip_friction=self.cfg.fingertip_friction,
        )

        # action / target buffers (full relative joint control)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.dof_targets = self.default_joint_pos.clone()

        # cube rest height (env-local) -- used by lift and drop checks
        self.object_default_z = self.object.data.default_root_state[:, 2].clone()

        # arm / hand joint groups for action regularization
        self._arm_joint_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self._hand_joint_ids, _ = self.robot.find_joints(self.cfg.hand_joint_names)

        self._thumb_ee_idx = self.ee_names.index("thumb_rota_link2")
        self._other_ee_idx = [i for i in range(len(self.ee_names)) if i != self._thumb_ee_idx]

        # per-episode progress ratchets. -1 means "not armed yet"; first geometry update fills it.
        n_ft = len(self.ee_ids)
        self._lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._highest_lift = torch.zeros(self.num_envs, device=self.device)
        self._closest_fingertip_dist = torch.full((self.num_envs, n_ft), -1.0, device=self.device)
        self._curr_fingertip_distances = torch.zeros((self.num_envs, n_ft), device=self.device)
        self._is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # fixed goal point (env-local) + per-episode target orientation
        self.target_pos = torch.tensor(self.cfg.target_pos, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat[:, 0] = 1.0
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self._resample_goal(self.robot._ALL_INDICES)

        # Debug markers are authored as children of the robot links, not as world-space
        # point-instancer entries. This keeps their local transform fixed to the link in the
        # USD hierarchy, so the visible pad cannot lag behind the rendered hand.
        self.dbg_markers = None
        self._debug_pad_prim_paths = {}
        self._last_printed_debug_pad_offsets = None
        self._enable_interactive_pad_calibration = bool(self.cfg.debug_markers and self.sim.has_gui())
        if self._enable_interactive_pad_calibration:
            self._create_link_attached_debug_markers()

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        # static table (per-env prop, spawned before cloning)
        table_spawn = sim_utils.UsdFileCfg(usd_path=self.cfg.table_usd)
        table_spawn.func(
            "/World/envs/env_.*/Table", table_spawn, translation=self.cfg.table_pos, orientation=self.cfg.table_rot
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # full RELATIVE joint position control: target += scale * action (all 19 joints)
        self.actions = actions.clone().clamp(-1.0, 1.0)
        raw_targets = torch.clamp(
            self.dof_targets + self.cfg.action_scale * self.actions,
            self.dof_lower,
            self.dof_upper,
        )
        self.dof_targets = (
            self.cfg.act_moving_average * raw_targets + (1.0 - self.cfg.act_moving_average) * self.dof_targets
        )
        self.dof_targets = torch.clamp(self.dof_targets, self.dof_lower, self.dof_upper)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.dof_targets)

    # ------------------------------------------------------------------ goal
    def _resample_goal(self, env_ids):
        n = len(env_ids)
        # target orientation from roll/pitch/yaw ranges (like the reference UniformPoseCommand)
        rr, rp, ry = (
            self.cfg.target_rot_range_roll,
            self.cfg.target_rot_range_pitch,
            self.cfg.target_rot_range_yaw,
        )
        roll = sample_uniform(rr[0], rr[1], (n,), device=self.device)
        pitch = sample_uniform(rp[0], rp[1], (n,), device=self.device)
        yaw = sample_uniform(ry[0], ry[1], (n,), device=self.device)
        self.target_quat[env_ids] = quat_from_euler_xyz(roll, pitch, yaw)
        self._update_goal_marker()

    def _update_goal_marker(self):
        pos = self.target_pos + self.scene.env_origins
        self.goal_markers.visualize(pos, self.target_quat)

    # ------------------------------------------------------------------ mdp
    def _compute_intermediate_values(self):
        self._sync_finger_pad_offsets_from_debug_markers()
        root = self.robot.data.root_pos_w
        self.object_pos_w = self.object.data.root_pos_w
        self.object_quat_w = self.object.data.root_quat_w
        self.object_pos_b = self.object_pos_w - root
        # corrected finger pads: the link2 body ORIGIN sits at the proximal (mid) joint, while
        # the geometric tip is too far distal. Add the per-finger pad offset (rotated into world)
        # so the "end effector" is the pad surface used for grasping.
        ee_body_w = self.robot.data.body_pos_w[:, self.ee_ids]  # (N, 5, 3) proximal joints
        ft_quat = self.robot.data.body_quat_w[:, self.ee_ids]  # (N, 5, 4)
        ft_off = quat_apply(
            ft_quat.reshape(-1, 4), self.finger_pad_offset.reshape(-1, 3)
        ).reshape(self.num_envs, -1, 3)
        self.finger_pad_w = ee_body_w + ft_off  # (N, 5, 3) finger pads
        self.ee_pos_w = self.finger_pad_w  # grasp assembly = finger pads
        self.ee_pos_b = (self.ee_pos_w - root.unsqueeze(1)).reshape(self.num_envs, -1)  # (N, 15)
        # grasp-center point = palm body pos + offset (in palm frame) toward the fingers
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        self.palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.palm_center_w = palm_pos + quat_apply(self.palm_quat, self.palm_center_offset)
        self.palm_center_b = self.palm_center_w - root
        palm_local_normal = torch.tensor([0.0, -1.0, 0.0], device=self.device).expand(self.num_envs, 3)
        self.palm_normal_w = quat_apply(self.palm_quat, palm_local_normal)

        # fingertip -> object-center distances for pre-lift progress shaping
        self._curr_fingertip_distances = torch.norm(
            self.finger_pad_w - self.object_pos_w.unsqueeze(1), dim=-1
        )

        ft_sent = self._closest_fingertip_dist < 0.0
        self._closest_fingertip_dist = torch.where(
            ft_sent, self._curr_fingertip_distances, self._closest_fingertip_dist
        )

    def _update_dbg_markers(self):
        # Link-attached debug markers are static local prims under each link, so they need no
        # per-frame world-space updates.
        return

    def _create_link_attached_debug_markers(self):
        stage = sim_utils.get_current_stage()

        def _spawn_sphere(prim_path, color, radius, translation):
            if stage.GetPrimAtPath(prim_path).IsValid():
                return
            cfg = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            cfg.func(prim_path, cfg, translation=translation)

        env_id = 0
        robot_path = f"/World/envs/env_{env_id}/Robot"

        for body_name, offset in self.cfg.finger_pad_offsets.items():
            parent_path = f"{robot_path}/{body_name}"
            if not stage.GetPrimAtPath(parent_path).IsValid():
                continue
            marker_path = f"{parent_path}/dbg_finger_pad"
            _spawn_sphere(
                marker_path,
                (0.0, 1.0, 0.0),
                0.007,
                tuple(float(v) for v in offset),
            )
            self._debug_pad_prim_paths[body_name] = marker_path

        palm_path = f"{robot_path}/{self.cfg.palm_body_name}"
        if not stage.GetPrimAtPath(palm_path).IsValid():
            return

        _spawn_sphere(
            f"{palm_path}/dbg_palm_center",
            (1.0, 0.0, 0.0),
            0.02,
            tuple(float(v) for v in self.cfg.palm_center_offset),
        )
        for i in range(8):
            k = float(i + 1) * 0.012
            pos = (
                float(self.cfg.palm_center_offset[0]),
                float(self.cfg.palm_center_offset[1]) - k,
                float(self.cfg.palm_center_offset[2]),
            )
            _spawn_sphere(
                f"{palm_path}/dbg_palm_normal_{i}",
                (0.0, 0.3, 1.0),
                0.008,
                pos,
            )

    def _sync_finger_pad_offsets_from_debug_markers(self):
        if not self._enable_interactive_pad_calibration:
            return
        if not self._debug_pad_prim_paths:
            return

        stage = sim_utils.get_current_stage()
        offsets = []
        for body_name in self.ee_names:
            prim = stage.GetPrimAtPath(self._debug_pad_prim_paths.get(body_name, ""))
            if not prim.IsValid():
                return
            value = prim.GetAttribute("xformOp:translate").Get()
            if value is None:
                return
            offsets.append((float(value[0]), float(value[1]), float(value[2])))

        offset_tensor = torch.tensor(offsets, dtype=torch.float, device=self.device).unsqueeze(0).repeat(
            self.num_envs, 1, 1
        )
        if torch.max(torch.abs(offset_tensor - self.finger_pad_offset)) <= 1e-6:
            return
        self.finger_pad_offset.copy_(offset_tensor)

        if self._last_printed_debug_pad_offsets is None:
            should_print = True
        else:
            should_print = torch.max(torch.abs(offset_tensor[0] - self._last_printed_debug_pad_offsets)) > 1e-3
        if not should_print:
            return

        self._last_printed_debug_pad_offsets = offset_tensor[0].clone()
        offset_by_name = {name: offsets[i] for i, name in enumerate(self.ee_names)}
        print("[finger-pad-calib] Updated finger_pad_offsets:")
        print("    finger_pad_offsets = {")
        for name in self.cfg.finger_pad_offsets:
            x, y, z = offset_by_name[name]
            print(f'        "{name}": ({x:.6f}, {y:.6f}, {z:.6f}),')
        print("    }")

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        if self.dbg_markers is not None:
            self._update_dbg_markers()
        obs = torch.cat(
            (
                self.robot.data.joint_pos,  # 19
                self.robot.data.joint_vel,  # 19
                self.ee_pos_b,  # 15
                self.palm_center_b,  # 3
                self.object_pos_b,  # 3
                self.object_quat_w,  # 4
                self.target_pos,  # 3 (fixed, env-local)
                self.target_quat,  # 4
                self.actions,  # 19
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg

        # ---- lifting: sparse one-shot bonus when the object crosses the lift threshold ----
        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        z_lift = cfg.lift_z_offset + object_z - self.object_default_z
        actual_lift = object_z - self.object_default_z
        lifted = (z_lift > cfg.lifting_bonus_threshold) | self._lifted_object
        just_crossed = lifted & ~self._lifted_object
        lift_bonus = just_crossed.float() * cfg.lifting_bonus
        self._lifted_object = lifted
        self._is_success = actual_lift >= cfg.lift_success_height

        # Dense lift progress is a per-episode ratchet: it only pays for new max lift height
        # between the lift-bonus threshold and success height, so lowering and raising again
        # cannot re-collect the same reward.
        dense_lift_floor = torch.full_like(actual_lift, cfg.lifting_bonus_threshold)
        dense_lift_ceiling = torch.full_like(actual_lift, cfg.lift_success_height)
        prev_dense_lift = torch.maximum(self._highest_lift, dense_lift_floor)
        curr_dense_lift = torch.minimum(actual_lift, dense_lift_ceiling)
        dense_lift_delta = torch.clamp(curr_dense_lift - prev_dense_lift, min=0.0)
        dense_lift_rew = dense_lift_delta * cfg.dense_lift_rew_scale
        self._highest_lift = torch.maximum(self._highest_lift, actual_lift)

        # ---- pre-lift fingertip approach: progress in all 5 fingertip-to-object distances ----
        ft_deltas = torch.clamp(self._closest_fingertip_dist - self._curr_fingertip_distances, 0.0, 10.0)
        self._closest_fingertip_dist = torch.minimum(self._closest_fingertip_dist, self._curr_fingertip_distances)
        pre_lift = (~lifted).float()
        thumb_ft_delta = ft_deltas[:, self._thumb_ee_idx]
        other_ft_deltas = ft_deltas[:, self._other_ee_idx]
        thumb_ft_rew = thumb_ft_delta * pre_lift * cfg.distance_delta_rew_scale
        other_ft_rew = other_ft_deltas.sum(dim=-1) * pre_lift * cfg.distance_delta_rew_scale
        ft_rew = thumb_ft_rew + other_ft_rew

        # ---- action penalty: L1 joint velocity, arm heavier than hand ----
        jv = self.robot.data.joint_vel
        kuka_pen = -cfg.kuka_actions_penalty_scale * jv[:, self._arm_joint_ids].abs().sum(dim=-1)
        hand_pen = -cfg.hand_actions_penalty_scale * jv[:, self._hand_joint_ids].abs().sum(dim=-1)
        self.prev_actions = self.actions.clone()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["lifted_frac"] = lifted.float().mean()
        self.extras["log"]["z_lift_mean"] = z_lift.mean()
        self.extras["log"]["object_actual_lift_mean"] = (object_z - self.object_default_z).mean()
        self.extras["log"]["highest_lift_mean"] = self._highest_lift.mean()
        self.extras["log"]["success_frac"] = self._is_success.float().mean()
        self.extras["log"]["fingertip_dist_mean"] = self._curr_fingertip_distances.mean()
        self.extras["log"]["thumb_dist_mean"] = self._curr_fingertip_distances[:, self._thumb_ee_idx].mean()
        self.extras["log"]["other_fingers_dist_mean"] = self._curr_fingertip_distances[:, self._other_ee_idx].mean()
        self.extras["log"]["thumb_ft_delta_mean"] = thumb_ft_delta.mean()
        self.extras["log"]["other_fingers_ft_delta_mean"] = other_ft_deltas.mean()
        self.extras["log"]["ft_rew_mean"] = ft_rew.mean()
        self.extras["log"]["thumb_ft_rew_mean"] = thumb_ft_rew.mean()
        self.extras["log"]["other_fingers_ft_rew_mean"] = other_ft_rew.mean()
        self.extras["log"]["lift_bonus_mean"] = lift_bonus.mean()
        self.extras["log"]["dense_lift_delta_mean"] = dense_lift_delta.mean()
        self.extras["log"]["dense_lift_rew_mean"] = dense_lift_rew.mean()
        self.extras["log"]["kuka_pen_mean"] = kuka_pen.mean()
        self.extras["log"]["hand_pen_mean"] = hand_pen.mean()

        return (
            ft_rew
            + lift_bonus
            + dense_lift_rew
            + kuka_pen
            + hand_pen
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        actual_lift = object_z - self.object_default_z
        self._is_success = actual_lift >= self.cfg.lift_success_height

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - self.cfg.drop_height)
        if not self.cfg.terminate_on_drop:
            dropped = torch.zeros_like(dropped)
        return dropped | self._is_success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._lifted_object[env_ids] = False
        self._highest_lift[env_ids] = 0.0
        self._closest_fingertip_dist[env_ids] = -1.0
        self._is_success[env_ids] = False

        # robot to (optionally randomized) home pose: arm joints get a uniform offset around
        # their home angle, clamped to the joint limits so randomization never leaves the URDF
        # range. The bound (cfg.reset_arm_joint_noise) is kept small enough to keep the hand off
        # the table -- no physics-based collision check is done.
        joint_pos = self.default_joint_pos[env_ids].clone()
        if self.cfg.reset_arm_joint_noise > 0.0:
            arm_noise = sample_uniform(
                -self.cfg.reset_arm_joint_noise,
                self.cfg.reset_arm_joint_noise,
                (len(env_ids), len(self._arm_joint_ids)),
                device=self.device,
            )
            joint_pos[:, self._arm_joint_ids] += arm_noise
            limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.dof_targets[env_ids] = joint_pos
        self._compute_intermediate_values()
        self._closest_fingertip_dist[env_ids] = -1.0

        # cube pose: default + xy noise + random yaw
        object_state = self.object.data.default_root_state[env_ids].clone()
        object_state[:, 0:2] = self._sample_non_overlapping_object_xy(env_ids, object_state[:, 0:2])
        object_state[:, 0:3] += self.scene.env_origins[env_ids]
        lo, hi = self.cfg.reset_object_yaw_range
        yaw = sample_uniform(lo, hi, (len(env_ids),), device=self.device)
        z_axis = torch.zeros((len(env_ids), 3), device=self.device)
        z_axis[:, 2] = 1.0
        yaw_quat = quat_from_angle_axis(yaw, z_axis)
        object_state[:, 3:7] = quat_mul(yaw_quat, object_state[:, 3:7])
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)

        self._resample_goal(env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._compute_intermediate_values()

    def _sample_non_overlapping_object_xy(self, env_ids: Sequence[int] | torch.Tensor, default_xy: torch.Tensor):
        nx, ny = self.cfg.reset_object_pos_noise
        n = len(env_ids)
        xy = default_xy.clone()
        candidate = xy
        hand_points = torch.cat(
            [self.finger_pad_w[env_ids], self.palm_center_w[env_ids].unsqueeze(1)],
            dim=1,
        )

        remaining = torch.ones(n, dtype=torch.bool, device=self.device)
        for _ in range(32):
            noise = sample_uniform(-1.0, 1.0, (n, 2), device=self.device)
            candidate = default_xy.clone()
            candidate[:, 0] += nx * noise[:, 0]
            candidate[:, 1] += ny * noise[:, 1]
            candidate_w = torch.cat(
                [
                    candidate + self.scene.env_origins[env_ids, 0:2],
                    self.object.data.default_root_state[env_ids, 2:3] + self.scene.env_origins[env_ids, 2:3],
                ],
                dim=-1,
            )
            min_dist = torch.norm(hand_points - candidate_w.unsqueeze(1), dim=-1).min(dim=-1).values
            accept = remaining & (min_dist > self.cfg.reset_min_hand_object_dist)
            xy[accept] = candidate[accept]
            remaining = remaining & ~accept
            if not remaining.any():
                break
        xy[remaining] = candidate[remaining]
        return xy
