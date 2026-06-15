# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab DirectRLEnv port of the SimToolReal dexterous tool-manipulation task.

Faithful port of ``simtoolreal-main/isaacgymenvs/tasks/simtoolreal/env.py`` with the
robot swapped to xArm7 + XHand (19 DOF). All task logic (reward gating + progress
ratchets, asymmetric obs, obs/action/object-state delays + noise, random force/torque
domain randomization, delta/absolute goal sampling, tolerance curriculum) is preserved;
robot-specific constants are adapted to ours.

NOTE on conventions: Isaac Lab uses WXYZ quaternions throughout (the reference used
XYZW). All rotation math here is WXYZ via ``isaaclab.utils.math``.
"""

from __future__ import annotations

import json
import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import (
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    random_orientation,
    sample_uniform,
)

from .simtoolreal_env_cfg import SimToolRealEnvCfg


class SimToolRealEnv(DirectRLEnv):
    cfg: SimToolRealEnvCfg

    def __init__(self, cfg: SimToolRealEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        N, dev = self.num_envs, self.device

        # ---- joint index maps (Isaac Lab reorders joints -> address by name) --------
        self.arm_ids, _ = self.robot.find_joints(self.cfg.arm_joint_expr, preserve_order=True)
        self.hand_ids, _ = self.robot.find_joints(self.cfg.hand_joint_expr, preserve_order=True)
        self.arm_ids = torch.tensor(self.arm_ids, dtype=torch.long, device=dev)
        self.hand_ids = torch.tensor(self.hand_ids, dtype=torch.long, device=dev)
        self.num_arm_dofs = len(self.arm_ids)
        self.num_hand_dofs = len(self.hand_ids)
        self.num_dofs = self.num_arm_dofs + self.num_hand_dofs
        # action layout: [0:7] -> arm_ids, [7:19] -> hand_ids
        self.act_to_joint = torch.cat([self.arm_ids, self.hand_ids])

        # body indices
        self.palm_idx, _ = self.robot.find_bodies(self.cfg.palm_body_name)
        self.palm_idx = self.palm_idx[0]
        self.fingertip_ids, _ = self.robot.find_bodies(
            list(self.cfg.fingertip_body_names), preserve_order=True
        )
        self.fingertip_ids = torch.tensor(self.fingertip_ids, dtype=torch.long, device=dev)
        self.num_fingertips = len(self.fingertip_ids)

        # ---- joint limits (per robot joint order) ----------------------------------
        limits = self.robot.root_physx_view.get_dof_limits().to(dev)  # (N, ndof, 2)
        self.dof_lower = limits[0, :, 0].clone()  # (ndof,)
        self.dof_upper = limits[0, :, 1].clone()

        # ---- home pose (xArm7): set arm joints to a reasonable ready pose ------------
        self.default_dof_pos = self.robot.data.default_joint_pos.clone()  # (N, ndof)

        # ---- action / target buffers (robot joint order) ----------------------------
        self.num_actions = int(self.cfg.action_space)
        self.num_obs = int(self.cfg.observation_space)
        self.actions = torch.zeros((N, self.num_actions), device=dev)
        self.cur_targets = self.default_dof_pos.clone()
        self.prev_targets = self.default_dof_pos.clone()

        # ---- offsets ----------------------------------------------------------------
        self.palm_offset = torch.tensor(self.cfg.palm_offset, device=dev).repeat(N, 1)
        self.fingertip_offset = (
            torch.tensor(self.cfg.fingertip_offset, device=dev).repeat(N, self.num_fingertips, 1)
        )

        # ---- per-env object dims from the pool manifest -----------------------------
        self._load_object_dims()

        # ---- keypoint offsets (object-sized for obs, fixed-size for reward) ----------
        corners = torch.tensor(
            [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=torch.float, device=dev
        )  # (4,3)
        self.num_keypoints = self.cfg.num_keypoints
        ks = self.cfg.keypoint_scale
        # object-sized: corner * aabb_extent[axis] * keypoint_scale / 2   (per env)
        self.object_keypoint_offsets = (
            corners.unsqueeze(0) * self.object_aabb[:, None, :] * ks / 2.0
        )  # (N,4,3)
        fixed = torch.tensor(self.cfg.fixed_size, device=dev)
        self.object_keypoint_offsets_fixed = (
            corners.unsqueeze(0) * fixed[None, None, :] * ks / 2.0
        ).repeat(N, 1, 1)  # (N,4,3)
        self.object_scale_noise_multiplier = torch.ones((N, 3), device=dev)

        # ---- goal state -------------------------------------------------------------
        self.goal_pos = torch.zeros((N, 3), device=dev)
        self.goal_rot = torch.zeros((N, 4), device=dev)
        self.goal_rot[:, 0] = 1.0
        self._setup_target_volume()

        # ---- reward / ratchet / success tracking ------------------------------------
        self.lifted_object = torch.zeros(N, dtype=torch.bool, device=dev)
        self.closest_fingertip_dist = -torch.ones((N, self.num_fingertips), device=dev)
        self.furthest_hand_dist = -torch.ones(N, device=dev)
        self.closest_keypoint_max_dist = -torch.ones(N, device=dev)
        self.closest_keypoint_max_dist_fixed = -torch.ones(N, device=dev)
        self.finger_rew_coeffs = torch.ones((N, self.num_fingertips), device=dev)
        self.near_goal_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.successes = torch.zeros(N, device=dev)
        self.prev_episode_successes = torch.zeros(N, device=dev)
        self.reset_goal_buf = torch.zeros(N, dtype=torch.bool, device=dev)
        self.object_init_z = torch.full((N,), self.cfg.object_rest_z, device=dev)

        # ---- tolerance curriculum ---------------------------------------------------
        self.success_tolerance = float(self.cfg.success_tolerance)
        self.last_curriculum_update = 0
        self.total_env_steps = 0

        # ---- force / torque DR buffers ----------------------------------------------
        # NOTE: all pool objects MUST share one rigid-body link name ("handle_head") or
        # the GPU rigid-body view drops the odd ones out (CUDA assert) -> see gen_pool.
        self.object_mass = self.object.root_physx_view.get_masses().to(dev).view(N)  # (N,)
        self.ext_forces = torch.zeros((N, 1, 3), device=dev)
        self.ext_torques = torch.zeros((N, 1, 3), device=dev)
        self.random_force_prob = self._sample_log_uniform(*self.cfg.force_prob_range, N)
        self.random_torque_prob = self._sample_log_uniform(*self.cfg.torque_prob_range, N)

        # ---- delay queues -----------------------------------------------------------
        self.obs_queue = torch.zeros((N, self.cfg.obs_delay_max, self.num_obs), device=dev)
        self.action_queue = torch.zeros((N, self.cfg.action_delay_max, self.num_actions), device=dev)
        self.object_state_queue = torch.zeros((N, self.cfg.object_state_delay_max, 13), device=dev)

        # intermediate buffers filled each step
        self.palm_center_pos = torch.zeros((N, 3), device=dev)
        self.fingertip_pos = torch.zeros((N, self.num_fingertips, 3), device=dev)
        self.last_reward = torch.zeros(N, device=dev)
        # guard so _compute_intermediate_values runs exactly once per step (set in
        # _get_dones during step(); _get_observations recomputes on the reset() path).
        self._inter_ready = False

        self.extras["log"] = {}

    # ============================================================ scene
    def _setup_scene(self):
        # NOTE: scene.replicate_physics=False -> InteractiveScene has ALREADY cloned the
        # (empty) per-env xforms before this runs. Each spawner's @clone decorator then
        # spawns into env_0 and clones to every env; MultiAssetSpawnerCfg spawns a DIFFERENT
        # object per env. We must therefore use env_.* paths and must NOT call
        # clone_environments() -- doing so re-copies env_0's object into every env, which
        # was the "extra box" (env_0's hammer duplicated on top of each env's own tool).
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        # ground (global) + a static table box per env (env_.* so @clone fills all envs)
        spawn_ground = sim_utils.GroundPlaneCfg()
        spawn_ground.func("/World/ground", spawn_ground)
        table_cfg = sim_utils.CuboidCfg(
            size=self.cfg.table_size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
            rigid_props=None,
        )
        table_cfg.func("/World/envs/env_.*/Table", table_cfg, translation=self.cfg.table_center)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        # replicate_physics=False does NOT auto-filter cross-env collisions (clone_environments
        # would have). Without filtering, the GPU rigid-body view can fail to register all
        # objects. Set up per-env collision filtering explicitly (ground is global).
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

        # default GUI camera looking at env_0's robot/table workspace (so a non-headless
        # run actually shows the scene instead of an empty default view).
        try:
            self.sim.set_camera_view(eye=(1.8, 1.8, 1.3), target=(0.3, 0.0, 0.5))
        except Exception:
            pass

        # goal marker DISABLED for now: the VisualizationMarkers PointInstancer renders
        # broken under Fabric ("mismatched prototypes" -> some instances show a fallback
        # cube, the mysterious "extra box"). It's purely cosmetic (not used by training),
        # so we drop it. A/B: with this None, any remaining box is the eraser tool itself.
        self.goal_markers = None

    # ============================================================ helpers: object dims
    def _load_object_dims(self):
        """Build per-env object AABB extents from the manifest (env i -> object i%P)."""
        N, dev = self.num_envs, self.device
        with open(self.cfg.object_manifest_path) as f:
            pool = json.load(f)["objects"]
        aabb = torch.tensor([o["aabb_extents"] for o in pool], dtype=torch.float, device=dev)  # (P,3)
        P = aabb.shape[0]
        idx = torch.arange(N, device=dev) % P
        self.object_pool_idx = idx
        self.object_aabb = aabb[idx]  # (N,3)

    def _setup_target_volume(self):
        dev = self.device
        mins = torch.tensor(self.cfg.target_volume_mins, device=dev)
        maxs = torch.tensor(self.cfg.target_volume_maxs, device=dev)
        origin = (mins + maxs) / 2.0
        half = (maxs - mins) / 2.0 * self.cfg.target_volume_region_scale
        self.tv_min = origin - half
        self.tv_max = origin + half

    # ============================================================ pre-physics / action
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # mid-episode goal resampling for envs that just succeeded (set last step)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._resample_goal(goal_env_ids, is_first_goal=False)
            self.reset_goal_buf[goal_env_ids] = False

        actions = actions.clone().to(self.device)
        # ---- action delay queue ----
        if self.cfg.use_action_delay:
            self._update_queue(self.action_queue, actions)
            didx = torch.randint(0, self.cfg.action_delay_max, (self.num_envs,), device=self.device)
            actions = self.action_queue[torch.arange(self.num_envs), didx].clone()
        self.actions = actions

        # ---- arm: relative to previous target (+1.5*dt*a), then EMA blend ----
        a_arm = actions[:, : self.num_arm_dofs]
        a_hand = actions[:, self.num_arm_dofs :]
        prev_arm = self.prev_targets[:, self.arm_ids]
        prev_hand = self.prev_targets[:, self.hand_ids]
        lo_arm, hi_arm = self.dof_lower[self.arm_ids], self.dof_upper[self.arm_ids]
        lo_hand, hi_hand = self.dof_lower[self.hand_ids], self.dof_upper[self.hand_ids]

        if self.cfg.use_relative_control:
            base_arm = self.robot.data.joint_pos[:, self.arm_ids]
        else:
            base_arm = prev_arm
        tgt_arm = base_arm + self.cfg.dof_speed_scale * self.step_dt * a_arm
        tgt_arm = torch.clamp(tgt_arm, lo_arm, hi_arm)
        tgt_arm = self.cfg.arm_moving_average * tgt_arm + (1.0 - self.cfg.arm_moving_average) * prev_arm

        # ---- hand: scale [-1,1] -> limits, then EMA blend, then clamp ----
        tgt_hand = 0.5 * (a_hand + 1.0) * (hi_hand - lo_hand) + lo_hand
        tgt_hand = self.cfg.hand_moving_average * tgt_hand + (1.0 - self.cfg.hand_moving_average) * prev_hand
        tgt_hand = torch.clamp(tgt_hand, lo_hand, hi_hand)

        self.cur_targets[:, self.arm_ids] = tgt_arm
        self.cur_targets[:, self.hand_ids] = tgt_hand
        self.prev_targets[:] = self.cur_targets

        # ---- random object forces/torques (domain randomization) ----
        self._apply_random_forces()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.cur_targets)
        if self.cfg.force_scale > 0.0 or self.cfg.torque_scale > 0.0:
            self.object.set_external_force_and_torque(self.ext_forces, self.ext_torques)

    def _apply_random_forces(self):
        N, dev = self.num_envs, self.device
        self.ext_forces.zero_()
        self.ext_torques.zero_()
        lifted = self.lifted_object.float().view(N, 1)
        if self.cfg.force_scale > 0.0:
            hit = (torch.rand(N, device=dev) < self.random_force_prob).view(N, 1)
            f = torch.randn((N, 3), device=dev) * self.object_mass.view(N, 1) * self.cfg.force_scale
            f = f * hit
            if self.cfg.force_only_when_lifted:
                f = f * lifted
            self.ext_forces[:, 0, :] = f
        if self.cfg.torque_scale > 0.0:
            hit = (torch.rand(N, device=dev) < self.random_torque_prob).view(N, 1)
            t = torch.randn((N, 3), device=dev) * self.object_mass.view(N, 1) * self.cfg.torque_scale
            t = t * hit
            if self.cfg.torque_only_when_lifted:
                t = t * lifted
            self.ext_torques[:, 0, :] = t

    # ============================================================ intermediate values
    def _compute_intermediate_values(self):
        N, dev = self.num_envs, self.device
        # palm center
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx] - self.scene.env_origins
        palm_rot = self.robot.data.body_quat_w[:, self.palm_idx]  # wxyz
        self.palm_pos = palm_pos
        self.palm_rot = palm_rot
        self.palm_vel = self.robot.data.body_vel_w[:, self.palm_idx]  # (N,6) lin+ang
        self.palm_center_pos = palm_pos + quat_apply(palm_rot, self.palm_offset)

        # fingertips (+ offset along tip)
        ft_pos = self.robot.data.body_pos_w[:, self.fingertip_ids] - self.scene.env_origins.unsqueeze(1)
        ft_rot = self.robot.data.body_quat_w[:, self.fingertip_ids]
        ft_off = quat_apply(ft_rot.reshape(-1, 4), self.fingertip_offset.reshape(-1, 3)).reshape(
            N, self.num_fingertips, 3
        )
        self.fingertip_pos = ft_pos + ft_off
        self.fingertip_pos_rel_palm = self.fingertip_pos - self.palm_center_pos.unsqueeze(1)

        # object clean state (env-local frame)
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w  # wxyz
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w
        object_state = torch.cat(
            [self.object_pos, self.object_rot, self.object_linvel, self.object_angvel], dim=-1
        )  # (N,13)

        # ---- observed (delayed + noisy) object state ----
        self._update_queue(self.object_state_queue, object_state)
        observed = object_state.clone()
        if self.cfg.use_object_state_delay_noise:
            didx = torch.randint(0, self.cfg.object_state_delay_max, (N,), device=dev)
            observed = self.object_state_queue[torch.arange(N), didx].clone()
            observed[:, 0:3] += torch.randn((N, 3), device=dev) * self.cfg.object_state_xyz_noise_std
            observed[:, 3:7] = self._sample_delta_quat(
                observed[:, 3:7], self.cfg.object_state_rotation_noise_degrees
            )
        self.observed_object_pos = observed[:, 0:3]
        self.observed_object_rot = observed[:, 3:7]

        # ---- fingertip->object distances (for approach reward) ----
        d = torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1)  # (N,F)
        self.curr_fingertip_distances = d

        # ---- keypoints ----
        scale_noise = self.object_scale_noise_multiplier  # (N,3)
        off = self.object_keypoint_offsets * scale_noise.unsqueeze(1)  # (N,4,3)
        self.obj_keypoint_pos = self.object_pos.unsqueeze(1) + quat_apply(
            self.object_rot.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4),
            off.reshape(-1, 3),
        ).reshape(N, self.num_keypoints, 3)
        self.goal_keypoint_pos = self.goal_pos.unsqueeze(1) + quat_apply(
            self.goal_rot.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4),
            off.reshape(-1, 3),
        ).reshape(N, self.num_keypoints, 3)
        self.observed_obj_keypoint_pos = self.observed_object_pos.unsqueeze(1) + quat_apply(
            self.observed_object_rot.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4),
            off.reshape(-1, 3),
        ).reshape(N, self.num_keypoints, 3)

        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos
        self.observed_keypoints_rel_goal = self.observed_obj_keypoint_pos - self.goal_keypoint_pos
        self.keypoints_rel_palm = self.obj_keypoint_pos - self.palm_center_pos.unsqueeze(1)
        self.observed_keypoints_rel_palm = (
            self.observed_obj_keypoint_pos - self.palm_center_pos.unsqueeze(1)
        )
        self.keypoints_max_dist = torch.norm(self.keypoints_rel_goal, dim=-1).max(dim=-1).values

        # fixed-size keypoints (for reward distance)
        offf = self.object_keypoint_offsets_fixed  # (N,4,3)
        objk = self.object_pos.unsqueeze(1) + quat_apply(
            self.object_rot.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4),
            offf.reshape(-1, 3),
        ).reshape(N, self.num_keypoints, 3)
        goalk = self.goal_pos.unsqueeze(1) + quat_apply(
            self.goal_rot.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4),
            offf.reshape(-1, 3),
        ).reshape(N, self.num_keypoints, 3)
        self.keypoints_max_dist_fixed = torch.norm(objk - goalk, dim=-1).max(dim=-1).values

    # ============================================================ dones
    def _get_dones(self):
        self._compute_intermediate_values()
        self._inter_ready = True
        N, dev = self.num_envs, self.device
        ones = torch.ones(N, dtype=torch.bool, device=dev)
        zeros = torch.zeros(N, dtype=torch.bool, device=dev)

        object_z_low = self.object_pos[:, 2] < self.cfg.fall_z
        max_consec = self.successes >= self.cfg.max_consecutive_successes
        hand_far = self.curr_fingertip_distances.max(dim=-1).values > self.cfg.hand_far_dist
        if self.cfg.reset_when_dropped:
            dropped = (self.object_pos[:, 2] < self.object_init_z) & self.lifted_object
        else:
            dropped = zeros

        terminated = object_z_low | max_consec | hand_far | dropped
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ============================================================ rewards
    def _get_rewards(self):
        N = self.num_envs
        cfg = self.cfg

        # ---- lifting reward ----
        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_z
        lifting_rew = torch.clamp(z_lift, 0.0, 0.5)
        lifted = (z_lift > cfg.lifting_bonus_threshold) | self.lifted_object
        just_lifted = lifted & ~self.lifted_object
        lift_bonus_rew = cfg.lifting_bonus * just_lifted.float()
        lifting_rew = lifting_rew * (~lifted).float()
        self.lifted_object = lifted

        # ---- fingertip approach (progress ratchet), gated before lift ----
        ft_delta = self.closest_fingertip_dist - self.curr_fingertip_distances
        self.closest_fingertip_dist = torch.minimum(
            self.closest_fingertip_dist, self.curr_fingertip_distances
        )
        # initialize ratchet on first valid frame (was -1)
        first = self.closest_fingertip_dist < 0
        self.closest_fingertip_dist = torch.where(
            first, self.curr_fingertip_distances, self.closest_fingertip_dist
        )
        ft_delta = torch.clamp(ft_delta, 0.0, 10.0) * self.finger_rew_coeffs
        fingertip_delta_rew = ft_delta.sum(dim=-1) * (~lifted).float()

        # ---- keypoint reward (progress ratchet), gated after lift ----
        if cfg.fixed_size_keypoint_reward:
            kp_now = self.keypoints_max_dist_fixed
            closest = self.closest_keypoint_max_dist_fixed
        else:
            kp_now = self.keypoints_max_dist
            closest = self.closest_keypoint_max_dist
        kp_delta = closest - kp_now
        new_closest = torch.minimum(closest, kp_now)
        firstk = closest < 0
        new_closest = torch.where(firstk, kp_now, new_closest)
        if cfg.fixed_size_keypoint_reward:
            self.closest_keypoint_max_dist_fixed = new_closest
        else:
            self.closest_keypoint_max_dist = new_closest
        kp_delta = torch.where(firstk, torch.zeros_like(kp_delta), kp_delta)
        keypoint_rew = torch.clamp(kp_delta, 0.0, 100.0) * lifted.float()

        # ---- success / near-goal ----
        keypoint_success_tol = self.success_tolerance * cfg.keypoint_scale
        near_goal = kp_now <= keypoint_success_tol
        if cfg.force_consecutive_near_goal_steps:
            self.near_goal_steps = (self.near_goal_steps + near_goal.long()) * near_goal.long()
        else:
            self.near_goal_steps += near_goal.long()
        is_success = self.near_goal_steps >= cfg.success_steps
        self.successes += is_success.float()
        self.reset_goal_buf = is_success.clone()
        self.near_goal_steps = torch.where(
            is_success, torch.zeros_like(self.near_goal_steps), self.near_goal_steps
        )

        if cfg.force_consecutive_near_goal_steps:
            bonus_rew = is_success.float() * cfg.reach_goal_bonus
        else:
            bonus_rew = near_goal.float() * (cfg.reach_goal_bonus / cfg.success_steps)

        # ---- action penalties (joint velocity L1, arm vs hand) ----
        jv = self.robot.data.joint_vel
        arm_pen = -jv[:, self.arm_ids].abs().sum(dim=-1) * cfg.kuka_actions_penalty_scale
        hand_pen = -jv[:, self.hand_ids].abs().sum(dim=-1) * cfg.hand_actions_penalty_scale

        # ---- object velocity penalties (disabled by default) ----
        obj_lin_pen = -(self.object_linvel**2).sum(dim=-1) * cfg.object_lin_vel_penalty_scale
        obj_ang_pen = -(self.object_angvel**2).sum(dim=-1) * cfg.object_ang_vel_penalty_scale

        reward = (
            fingertip_delta_rew * cfg.distance_delta_rew_scale
            + lifting_rew * cfg.lifting_rew_scale
            + lift_bonus_rew
            + keypoint_rew * cfg.keypoint_rew_scale
            + arm_pen
            + hand_pen
            + bonus_rew
            + obj_lin_pen
            + obj_ang_pen
        )
        self.last_reward = reward.detach()

        # ---- tolerance curriculum tick ----
        self.total_env_steps += self.num_envs
        self._tolerance_curriculum()

        self.extras["log"]["success_tolerance"] = self.success_tolerance
        self.extras["log"]["lifted_frac"] = self.lifted_object.float().mean()
        self.extras["log"]["mean_successes"] = self.successes.mean()
        return reward

    # ============================================================ observations
    def _get_observations(self):
        # reset() calls this without _get_dones having run -> compute intermediates here.
        if not self._inter_ready:
            self._compute_intermediate_values()
        self._inter_ready = False

        N = self.num_envs
        jpos = self.robot.data.joint_pos[:, self.act_to_joint]
        jvel = self.robot.data.joint_vel[:, self.act_to_joint]
        prev_targ = self.prev_targets[:, self.act_to_joint]
        object_scales = self.object_aabb * self.object_scale_noise_multiplier

        # ---- policy obs (uses observed/noisy keypoints, noisy joint vel) ----
        jvel_noisy = jvel + torch.randn_like(jvel) * self.cfg.joint_velocity_obs_noise_std
        obs = torch.cat(
            [
                jpos,
                jvel_noisy,
                prev_targ,
                self.palm_center_pos,
                self.palm_rot,
                self.observed_object_rot,
                self.fingertip_pos_rel_palm.reshape(N, -1),
                self.observed_keypoints_rel_palm.reshape(N, -1),
                self.observed_keypoints_rel_goal.reshape(N, -1),
                object_scales,
            ],
            dim=-1,
        )
        # ---- obs delay queue ----
        if self.cfg.use_obs_delay:
            self._update_queue(self.obs_queue, obs)
            didx = torch.randint(0, self.cfg.obs_delay_max, (N,), device=self.device)
            obs = self.obs_queue[torch.arange(N), didx].clone()

        # ---- critic state (clean, asymmetric) ----
        state = torch.cat(
            [
                jpos,
                jvel,
                prev_targ,
                self.palm_center_pos,
                self.palm_rot,
                self.palm_vel,
                self.object_rot,
                torch.cat([self.object_linvel, self.object_angvel], dim=-1),
                self.fingertip_pos_rel_palm.reshape(N, -1),
                self.keypoints_rel_palm.reshape(N, -1),
                self.keypoints_rel_goal.reshape(N, -1),
                object_scales,
                self.closest_keypoint_max_dist.unsqueeze(-1),
                self.closest_fingertip_dist,
                self.lifted_object.float().unsqueeze(-1),
                torch.log(self.episode_length_buf.float() / 10 + 1).unsqueeze(-1),
                torch.log(self.successes + 1).unsqueeze(-1),
                (0.01 * self.last_reward).unsqueeze(-1),
            ],
            dim=-1,
        )

        c = self.cfg.clamp_abs_observations
        if c > 0:
            obs = obs.clamp(-c, c)
            state = state.clamp(-c, c)

        # update goal marker
        if self.goal_markers is not None:
            self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot)

        return {"policy": obs, "critic": state}

    # ============================================================ reset
    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        N = len(env_ids)
        dev = self.device

        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0.0

        # ---- robot DOF reset: home pose + per-joint noise * (limit-relative span) ----
        default = self.default_dof_pos[env_ids]
        delta_max = self.dof_upper - default
        delta_min = self.dof_lower - default
        rand = sample_uniform(0.0, 1.0, (N, self.robot.num_joints), dev)
        rand_delta = delta_min + (delta_max - delta_min) * rand
        noise_coeff = torch.zeros(self.robot.num_joints, device=dev)
        noise_coeff[self.arm_ids] = self.cfg.reset_dof_pos_noise_arm
        noise_coeff[self.hand_ids] = self.cfg.reset_dof_pos_noise_fingers
        dof_pos = torch.clamp(default + noise_coeff * rand_delta, self.dof_lower, self.dof_upper)
        dof_vel = self.cfg.reset_dof_vel_noise * sample_uniform(
            -1.0, 1.0, (N, self.robot.num_joints), dev
        )
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.cur_targets[env_ids] = dof_pos
        self.prev_targets[env_ids] = dof_pos

        # ---- object pose reset on the table ----
        rest_z = self.cfg.object_rest_z + sample_uniform(
            -self.cfg.table_reset_z_range, self.cfg.table_reset_z_range, (N,), dev
        )
        self.object_init_z[env_ids] = rest_z
        rp = sample_uniform(-1.0, 1.0, (N, 3), dev)
        pos = torch.zeros((N, 3), device=dev)
        pos[:, 0] = self.cfg.object_spawn_x + self.cfg.reset_position_noise_x * rp[:, 0]
        pos[:, 1] = self.cfg.object_spawn_y + self.cfg.reset_position_noise_y * rp[:, 1]
        pos[:, 2] = rest_z + self.cfg.reset_position_noise_z * rp[:, 2]
        if self.cfg.randomize_object_rotation:
            rot = random_orientation(N, dev)
        else:
            rot = torch.zeros((N, 4), device=dev)
            rot[:, 0] = 1.0
        root_pose = torch.cat([pos + self.scene.env_origins[env_ids], rot], dim=-1)
        self.object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros((N, 6), device=dev), env_ids=env_ids)

        self.object_scale_noise_multiplier[env_ids] = sample_uniform(
            *self.cfg.object_scale_noise_multiplier_range, (N, 3), dev
        )

        # ---- ratchet / tracking resets ----
        self.lifted_object[env_ids] = False
        self.closest_fingertip_dist[env_ids] = -1
        self.furthest_hand_dist[env_ids] = -1
        self.closest_keypoint_max_dist[env_ids] = -1
        self.closest_keypoint_max_dist_fixed[env_ids] = -1
        self.near_goal_steps[env_ids] = 0
        self.reset_goal_buf[env_ids] = False

        # ---- force/torque probabilities ----
        self.random_force_prob[env_ids] = self._sample_log_uniform(
            *self.cfg.force_prob_range, N
        )
        self.random_torque_prob[env_ids] = self._sample_log_uniform(
            *self.cfg.torque_prob_range, N
        )

        # ---- first goal ----
        self._resample_goal(env_ids, is_first_goal=True)

    # ============================================================ goal sampling
    def _resample_goal(self, env_ids, is_first_goal: bool):
        dev = self.device
        N = len(env_ids)
        gtype = self.cfg.goal_sampling_type
        if is_first_goal or gtype == "absolute":
            pos = self.tv_min + sample_uniform(0.0, 1.0, (N, 3), dev) * (self.tv_max - self.tv_min)
            rot = random_orientation(N, dev)
        elif gtype == "delta":
            pos = self.goal_pos[env_ids] + sample_uniform(
                -self.cfg.delta_goal_distance, self.cfg.delta_goal_distance, (N, 3), dev
            )
            pos = torch.clamp(pos, self.tv_min, self.tv_max)
            rot = self._sample_delta_quat(self.goal_rot[env_ids], self.cfg.delta_rotation_degrees)
        elif gtype == "coin_flip":
            coin = sample_uniform(0.0, 1.0, (N, 1), dev)
            pos_t = torch.clamp(
                self.goal_pos[env_ids]
                + sample_uniform(-self.cfg.delta_goal_distance, self.cfg.delta_goal_distance, (N, 3), dev),
                self.tv_min,
                self.tv_max,
            )
            rot_r = self._sample_delta_quat(self.goal_rot[env_ids], self.cfg.delta_rotation_degrees)
            pos = torch.where(coin < 0.5, pos_t, self.goal_pos[env_ids])
            rot = torch.where(coin < 0.5, self.goal_rot[env_ids], rot_r)
        else:
            raise ValueError(f"unknown goalSamplingType: {gtype}")
        self.goal_pos[env_ids] = pos
        self.goal_rot[env_ids] = rot
        # reset keypoint ratchet on new goal
        self.closest_keypoint_max_dist[env_ids] = -1
        self.closest_keypoint_max_dist_fixed[env_ids] = -1
        self.near_goal_steps[env_ids] = 0

    # ============================================================ misc helpers
    def _sample_delta_quat(self, quat_wxyz, delta_degrees):
        """Post-multiply quat by a random rotation of magnitude in [-delta, delta] (body frame)."""
        N = quat_wxyz.shape[0]
        dev = quat_wxyz.device
        rad = delta_degrees * math.pi / 180.0
        axis = torch.randn((N, 3), device=dev)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        angle = sample_uniform(-rad, rad, (N,), dev)
        delta = quat_from_angle_axis(angle, axis)
        return quat_mul(quat_wxyz, delta)

    def _sample_log_uniform(self, min_value, max_value, n):
        dev = self.device
        lo, hi = math.log(min_value), math.log(max_value)
        return torch.exp(lo + (hi - lo) * torch.rand(n, device=dev))

    def _update_queue(self, queue, current):
        """Shift history down by 1 and insert current at index 0; fill on episode start."""
        T = queue.shape[1]
        # fill the whole queue with the current value for the first step(s) after a reset
        # (episode_length_buf is 0 during pre-physics, 1 by the time obs is computed).
        is_start = (self.episode_length_buf <= 1).view(-1, 1, 1)
        queue[:] = torch.where(is_start, current.unsqueeze(1).expand(-1, T, -1), queue)
        queue[:, 1:] = queue[:, :-1].clone()
        queue[:, 0] = current

    def _tolerance_curriculum(self):
        if self.total_env_steps - self.last_curriculum_update < self.cfg.tolerance_curriculum_interval:
            return
        if self.prev_episode_successes.mean().item() < 3.0:
            return
        self.success_tolerance *= self.cfg.tolerance_curriculum_increment
        self.success_tolerance = min(self.success_tolerance, float(self.cfg.success_tolerance))
        self.success_tolerance = max(self.success_tolerance, float(self.cfg.target_success_tolerance))
        self.last_curriculum_update = self.total_env_steps
