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
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

from xhand_inhand.utils import apply_fingertip_friction

from .pick_repose_cube_env_cfg import PickReposeCubeEnvCfg


class PickReposeCubeEnv(DirectRLEnv):
    cfg: PickReposeCubeEnvCfg

    def __init__(self, cfg: PickReposeCubeEnvCfg, render_mode: str | None = None, **kwargs):
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

        # cube rest height (env-local) -- used by the lift shaping and the drop test
        self.object_default_z = self.object.data.default_root_state[:, 2].clone()

        # arm / hand joint groups for the (arm-heavy, hand-light) L1 velocity penalty
        self._arm_joint_ids, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self._hand_joint_ids, _ = self.robot.find_joints(self.cfg.hand_joint_names)

        # 4 cube-corner keypoints (a tetrahedral subset of the 8 corners; enough to pin a 6D pose),
        # scaled to the cube half-edge. Distances between object corners and goal corners unify the
        # position AND orientation error into a single scalar (SimToolReal keypoint reward).
        corners = torch.tensor(
            [(1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1)], dtype=torch.float, device=self.device
        )
        self._keypoint_offsets = corners * self.cfg.keypoint_half_extent  # (4, 3)

        # thumb vs the other four fingertips, for the thumb-opposition grasp-quality metric
        self._thumb_ee_idx = self.ee_names.index("thumb_rota_link2")
        self._other_ee_idx = [i for i in range(len(self.ee_names)) if i != self._thumb_ee_idx]

        # fingertip ids within the contact sensor + thumb position (for the contact grasp gate)
        self.tip_ids, tip_names = self._contact_sensor.find_bodies(self.cfg.ee_body_names)
        self.thumb_local = tip_names.index(self.cfg.thumb_tip_name)

        # ---- per-episode reward trackers (SimToolReal): progress ratchets + lift/success state.
        # sentinel -1 in the "closest" buffers means "not yet armed" (first obs fills it). ----
        n_ft = len(self.ee_ids)
        self._lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._closest_fingertip_dist = torch.full((self.num_envs, n_ft), -1.0, device=self.device)
        self._closest_keypoint_max_dist = torch.full((self.num_envs,), -1.0, device=self.device)
        self._curr_fingertip_distances = torch.zeros((self.num_envs, n_ft), device=self.device)
        self._keypoints_max_dist = torch.zeros(self.num_envs, device=self.device)
        self._near_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._near_goal_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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
        # fingertip contact sensing (real XHand has fingertip tactile)
        self._contact_sensor = ContactSensor(
            ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=0)
        )

        # static table (per-env prop, spawned before cloning)
        table_spawn = sim_utils.UsdFileCfg(usd_path=self.cfg.table_usd)
        table_spawn.func(
            "/World/envs/env_.*/Table", table_spawn, translation=self.cfg.table_pos, orientation=self.cfg.table_rot
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["contact"] = self._contact_sensor

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # full RELATIVE joint position control: target += scale * action (all 19 joints)
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.dof_targets + self.cfg.action_scale * self.actions
        self.dof_targets = torch.clamp(targets, self.dof_lower, self.dof_upper)

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

        # ---- SimToolReal reward geometry (kept idempotent; this runs >once per step) ----
        # fingertip -> object-center distances (N, 5)
        self._curr_fingertip_distances = torch.norm(
            self.finger_pad_w - self.object_pos_w.unsqueeze(1), dim=-1
        )
        # keypoint max-corner distance: object corners vs goal corners -> unifies pos + orient error
        kp = self._keypoint_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)  # (N, 4, 3)
        obj_kp = self.object_pos_w.unsqueeze(1) + quat_apply(
            self.object_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4), kp.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)
        goal_pos_w = self.target_pos + self.scene.env_origins
        goal_kp = goal_pos_w.unsqueeze(1) + quat_apply(
            self.target_quat.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4), kp.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)
        self._keypoints_max_dist = torch.norm(obj_kp - goal_kp, dim=-1).max(dim=-1).values
        # arm the progress ratchets on first observation (sentinel -1 -> current)
        ft_sent = self._closest_fingertip_dist < 0.0
        self._closest_fingertip_dist = torch.where(
            ft_sent, self._curr_fingertip_distances, self._closest_fingertip_dist
        )
        kp_sent = self._closest_keypoint_max_dist < 0.0
        self._closest_keypoint_max_dist = torch.where(
            kp_sent, self._keypoints_max_dist, self._closest_keypoint_max_dist
        )

        # ---- fingertip contact -> force-closure grasp gate (real tactile signal) ----
        # net contact force magnitude per fingertip; a real grasp = thumb in contact AND >=1 other
        # finger in contact. This needs the fingers to actually CLOSE and press (caging / posing
        # without contact -> no force -> not grasped), which geometry alone could not enforce.
        net = self._contact_sensor.data.net_forces_w
        self.tip_contact_mag = torch.norm(net[:, self.tip_ids, :], dim=-1)  # (N, 5)
        in_contact = self.tip_contact_mag > self.cfg.contact_force_threshold
        thumb_c = in_contact[:, self.thumb_local]
        others = in_contact.clone()
        others[:, self.thumb_local] = False
        self.grasped = thumb_c & others.any(dim=1)  # (N,) bool

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
                self.tip_contact_mag.clamp(max=20.0),  # 5 (fingertip contact force)
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
        # _get_dones() runs first each step: it refreshes the geometry (via
        # _compute_intermediate_values) and updates _near_goal / _is_success. Here we only sum
        # the SimToolReal reward terms and advance the progress ratchets.
        cfg = self.cfg

        # ---- LIFTING: dense height reward (pre-lift only) + one-shot bonus on first crossing ----
        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        z_lift = cfg.lift_z_offset + object_z - self.object_default_z
        lifted = (z_lift > cfg.lifting_bonus_threshold) | self._lifted_object
        just_crossed = lifted & ~self._lifted_object
        lift_bonus = just_crossed.float() * cfg.lifting_bonus
        lift_rew = torch.clamp(z_lift, 0.0, 0.5) * (~lifted).float() * cfg.lifting_rew_scale
        self._lifted_object = lifted

        # ---- FINGERTIP approach (progress, pre-lift): sum of per-tip closest-distance gains ----
        ft_deltas = torch.clamp(self._closest_fingertip_dist - self._curr_fingertip_distances, 0.0, 10.0)
        self._closest_fingertip_dist = torch.minimum(self._closest_fingertip_dist, self._curr_fingertip_distances)
        pre_lift = (~lifted).float()
        thumb_ft_delta = ft_deltas[:, self._thumb_ee_idx]
        other_ft_deltas = ft_deltas[:, self._other_ee_idx]
        thumb_ft_rew = thumb_ft_delta * pre_lift * cfg.distance_delta_rew_scale
        other_ft_rew = other_ft_deltas.sum(dim=-1) * pre_lift * cfg.distance_delta_rew_scale
        ft_rew = thumb_ft_rew + other_ft_rew

        # ---- GRASP QUALITY (geometric, no contact sensor): palm-closeness x thumb-opposition.
        # Caging leaves the palm far and the thumb un-opposed -> ~0; a force-closure palm grasp -> ~1.
        # Used as a dense post-lift guide AND to GATE the keypoint reward (a cage can't reorient). ----
        obj = self.object_pos_w
        palm_close = 1.0 - torch.tanh(torch.norm(self.palm_center_w - obj, dim=-1) / cfg.palm_std)
        tips = self.finger_pad_w  # (N, 5, 3)
        thumb_v = tips[:, self._thumb_ee_idx] - obj
        others_v = tips[:, self._other_ee_idx].mean(dim=1) - obj
        thumb_n = thumb_v / (thumb_v.norm(dim=-1, keepdim=True) + 1e-6)
        others_n = others_v / (others_v.norm(dim=-1, keepdim=True) + 1e-6)
        oppose = torch.clamp(-(thumb_n * others_n).sum(dim=-1), 0.0, 1.0)  # 1 => thumb opposes fingers
        grasp_quality = palm_close * oppose  # (N,) in [0, 1]
        # dense guide, gated on `lifted` so it can't be farmed by holding a grasp on the table
        # (the failure mode when it was always-on). The keypoint floor below keeps lift valuable
        # even before the grasp is good, so this no longer deadlocks.
        grasp_rew = grasp_quality * lifted.float() * cfg.w_grasp

        # ---- KEYPOINT tracking (progress, post-lift), gated by the CONTACT grasp with a FLOOR so a
        # cage still keeps "lift" valuable (no deadlock), while a real force-closure grasp (thumb +
        # finger actually pressing, from the contact sensor) pays ~4x -> drives caging/posing toward
        # a true closed grasp. Geometry alone (grasp_quality) could not enforce finger CLOSURE. ----
        kp_gate = cfg.grasp_gate_floor + (1.0 - cfg.grasp_gate_floor) * self.grasped.float()
        kp_delta = torch.clamp(self._closest_keypoint_max_dist - self._keypoints_max_dist, 0.0, 100.0)
        self._closest_keypoint_max_dist = torch.minimum(self._closest_keypoint_max_dist, self._keypoints_max_dist)
        kp_rew = kp_delta * lifted.float() * kp_gate * cfg.keypoint_rew_scale

        # ---- ACTION penalty: L1 joint velocity, arm penalized ~10x the hand ----
        jv = self.robot.data.joint_vel
        kuka_pen = -cfg.kuka_actions_penalty_scale * jv[:, self._arm_joint_ids].abs().sum(dim=-1)
        hand_pen = -cfg.hand_actions_penalty_scale * jv[:, self._hand_joint_ids].abs().sum(dim=-1)

        # ---- SUCCESS bonus: amortized over the near-goal hold (set in _get_dones) ----
        goal_bonus = self._near_goal.float() * (cfg.reach_goal_bonus / cfg.success_steps)

        self.prev_actions = self.actions.clone()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["lifted_frac"] = lifted.float().mean()
        self.extras["log"]["near_goal_frac"] = self._near_goal.float().mean()
        self.extras["log"]["successes_mean"] = self.successes.float().mean()
        self.extras["log"]["keypoint_dist_mean"] = self._keypoints_max_dist.mean()
        self.extras["log"]["fingertip_dist_mean"] = self._curr_fingertip_distances.mean()
        self.extras["log"]["thumb_dist_mean"] = self._curr_fingertip_distances[:, self._thumb_ee_idx].mean()
        self.extras["log"]["other_fingers_dist_mean"] = self._curr_fingertip_distances[:, self._other_ee_idx].mean()
        self.extras["log"]["thumb_ft_delta_mean"] = thumb_ft_delta.mean()
        self.extras["log"]["other_fingers_ft_delta_mean"] = other_ft_deltas.mean()
        self.extras["log"]["ft_rew_mean"] = ft_rew.mean()
        self.extras["log"]["thumb_ft_rew_mean"] = thumb_ft_rew.mean()
        self.extras["log"]["other_fingers_ft_rew_mean"] = other_ft_rew.mean()
        self.extras["log"]["lift_rew_mean"] = lift_rew.mean()
        self.extras["log"]["lift_bonus_mean"] = lift_bonus.mean()
        self.extras["log"]["grasp_rew_mean"] = grasp_rew.mean()
        self.extras["log"]["kp_rew_mean"] = kp_rew.mean()
        self.extras["log"]["goal_bonus_mean"] = goal_bonus.mean()
        self.extras["log"]["kuka_pen_mean"] = kuka_pen.mean()
        self.extras["log"]["hand_pen_mean"] = hand_pen.mean()
        self.extras["log"]["grasp_quality_mean"] = grasp_quality.mean()
        self.extras["log"]["palm_close_mean"] = palm_close.mean()
        self.extras["log"]["oppose_mean"] = oppose.mean()
        self.extras["log"]["grasp_contact_frac"] = self.grasped.float().mean()
        self.extras["log"]["tip_contact_mag_mean"] = self.tip_contact_mag.mean()

        return lift_rew + lift_bonus + ft_rew + grasp_rew + kp_rew + kuka_pen + hand_pen + goal_bonus

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()  # refreshes geometry + keypoint / fingertip distances

        # ---- near-goal / success bookkeeping (once per step, before rewards read it) ----
        self._near_goal = self._keypoints_max_dist <= self.cfg.success_tolerance
        self._near_goal_steps = self._near_goal_steps + self._near_goal.long()  # non-consecutive count
        self._is_success = self._near_goal_steps >= self.cfg.success_steps

        # on success: bank it and RESAMPLE a new goal pose (continuous reorientation, SimToolReal-style)
        self.successes = self.successes + self._is_success.long()
        succ_ids = self._is_success.nonzero(as_tuple=False).squeeze(-1)
        if succ_ids.numel() > 0:
            self._resample_goal(succ_ids)
            self._closest_keypoint_max_dist[succ_ids] = -1.0  # re-arm keypoint ratchet for the new goal
            self._near_goal_steps[succ_ids] = 0
            self._is_success[succ_ids] = False  # consumed; don't let it re-fire / re-terminate
            self.episode_length_buf[succ_ids] = 0  # treat as a soft boundary so timeout doesn't fire

        # ---- terminations ----
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - self.cfg.drop_height)
        if not self.cfg.terminate_on_drop:
            dropped = torch.zeros_like(dropped)
        hand_far = self._curr_fingertip_distances.max(dim=-1).values > self.cfg.hand_far_dist
        if self.cfg.max_consecutive_successes > 0:
            max_succ = self.successes >= self.cfg.max_consecutive_successes
        else:
            max_succ = torch.zeros_like(dropped)
        terminated = dropped | hand_far | max_succ
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset per-episode reward trackers (sentinel -1 in the "closest" ratchets = not yet armed,
        # re-filled by _compute_intermediate_values at the end of this reset)
        self._lifted_object[env_ids] = False
        self._closest_fingertip_dist[env_ids] = -1.0
        self._closest_keypoint_max_dist[env_ids] = -1.0
        self._near_goal[env_ids] = False
        self._near_goal_steps[env_ids] = 0
        self._is_success[env_ids] = False
        self.successes[env_ids] = 0

        # robot to default pose
        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.dof_targets[env_ids] = joint_pos

        # cube pose: default + xy noise + random yaw
        object_state = self.object.data.default_root_state[env_ids].clone()
        nx, ny = self.cfg.reset_object_pos_noise
        noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_state[:, 0] += nx * noise[:, 0]
        object_state[:, 1] += ny * noise[:, 1]
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
