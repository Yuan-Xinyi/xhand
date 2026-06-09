# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform

from ..scene_lighting import add_ceiling_fluorescent_lights
from .pick_repose_pen_env_cfg import PickReposePenEnvCfg


class PickReposePenEnv(DirectRLEnv):
    cfg: PickReposePenEnvCfg

    def __init__(self, cfg: PickReposePenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # reach assembly: the 5 fingertips + the palm-center point
        self.ee_ids, _ = self.robot.find_bodies(self.cfg.ee_body_names)
        self.palm_idx = self.robot.body_names.index(self.cfg.palm_body_name)
        self.palm_center_offset = torch.tensor(
            self.cfg.palm_center_offset, dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        # directed big-end axis (local) + the table normal it is sampled around (world +Z)
        self.pen_big_end_axis = torch.tensor(
            self.cfg.pen_big_end_axis, dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.table_normal = torch.tensor(self.cfg.table_normal, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # 4 object-frame keypoints at (+-sx/2, +-sy/2, +-sz/2) (eq. 9 of the reference)
        sx, sy, sz = (s / 2.0 for s in self.cfg.keypoint_scales)
        self.keypoint_offsets = torch.tensor(
            [[sx, sy, sz], [sx, sy, -sz], [-sx, -sy, sz], [-sx, -sy, -sz]],
            dtype=torch.float,
            device=self.device,
        )  # (4, 3)
        self.num_kp = self.keypoint_offsets.shape[0]

        # arm vs hand joint masks (for the split-L1 smoothness penalty)
        finger_keys = ("index", "mid", "ring", "pinky", "thumb")
        hand = [any(k in n for k in finger_keys) for n in self.robot.joint_names]
        self.hand_mask = torch.tensor(hand, dtype=torch.bool, device=self.device)
        self.arm_mask = ~self.hand_mask

        # joint limits / defaults
        limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower = limits[..., 0]
        self.dof_upper = limits[..., 1]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        # action / target buffers (full relative joint control)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.dof_targets = self.default_joint_pos.clone()

        # lift threshold + per-episode stateful reward variables
        self.object_default_z = self.object.data.default_root_state[:, 2].clone()
        self.lift_height = self.object_default_z + self.cfg.lift_margin
        self.object_init_z = self.object_default_z.clone()
        self.grasped_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.min_ft_dist = torch.full((self.num_envs,), float("inf"), device=self.device)  # d_ft*
        self.min_goal_dist = torch.full((self.num_envs,), float("inf"), device=self.device)  # d*

        # fixed goal point (env-local) + per-episode RANDOMIZED orientation (goal_axis / quat)
        self.target_pos = torch.tensor(self.cfg.target_pos, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.goal_axis = self.table_normal.clone()
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_quat[:, 0] = 1.0
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self._resample_goal(self.robot._ALL_INDICES)

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

        # lab fluorescent ceiling tubes + faint ambient fill
        add_ceiling_fluorescent_lights()

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
        # sample a big-end target direction UNIFORMLY within goal_cone_angle of +Z
        cos_max = math.cos(self.cfg.goal_cone_angle)
        cos_theta = 1.0 - torch.rand(n, device=self.device) * (1.0 - cos_max)
        theta = torch.acos(cos_theta.clamp(-1.0, 1.0))
        phi = 2.0 * math.pi * torch.rand(n, device=self.device)
        sin_theta = torch.sin(theta)
        axis = torch.stack([sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)], dim=-1)
        self.goal_axis[env_ids] = axis
        # full target orientation: rotate the pen's big-end local axis onto the sampled goal axis
        self.goal_quat[env_ids] = self._quat_from_two_vectors(self.pen_big_end_axis[env_ids], axis)
        self._update_goal_marker()

    def _quat_from_two_vectors(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cross = torch.cross(a, b, dim=-1)
        cross_norm = torch.norm(cross, dim=-1, keepdim=True)
        default = torch.zeros_like(cross)
        default[:, 0] = 1.0
        rot_axis = torch.where(cross_norm > 1e-6, cross / cross_norm.clamp(min=1e-6), default)
        angle = torch.acos(torch.clamp(torch.sum(a * b, dim=-1), -1.0, 1.0))
        return quat_from_angle_axis(angle, rot_axis)

    def _update_goal_marker(self):
        pos = self.target_pos + self.scene.env_origins
        self.goal_markers.visualize(pos, self.goal_quat)

    # ------------------------------------------------------------------ mdp
    def _keypoint_distance(self) -> torch.Tensor:
        """d(o, g) = max_i ||o_i - g_i|| over the 4 object-frame keypoints (world frame)."""
        kp = self.keypoint_offsets.unsqueeze(0).expand(self.num_envs, self.num_kp, 3)  # (N, K, 3)
        obj_q = self.object_quat_w.unsqueeze(1).expand(self.num_envs, self.num_kp, 4)
        goal_q = self.goal_quat.unsqueeze(1).expand(self.num_envs, self.num_kp, 4)
        o_kp = self.object_pos_w.unsqueeze(1) + quat_apply(obj_q, kp)  # (N, K, 3)
        g_kp = (self.target_pos + self.scene.env_origins).unsqueeze(1) + quat_apply(goal_q, kp)
        return torch.norm(o_kp - g_kp, dim=-1).max(dim=1).values  # (N,)

    def _compute_intermediate_values(self):
        root = self.robot.data.root_pos_w
        self.ee_pos_w = self.robot.data.body_pos_w[:, self.ee_ids]  # (N, 5, 3)
        self.object_pos_w = self.object.data.root_pos_w
        self.object_quat_w = self.object.data.root_quat_w
        self.object_pos_b = self.object_pos_w - root
        self.ee_pos_b = (self.ee_pos_w - root.unsqueeze(1)).reshape(self.num_envs, -1)  # (N, 15)
        # pen's DIRECTED big-end axis (world) -- for observation
        self.pen_big_axis_w = quat_apply(self.object_quat_w, self.pen_big_end_axis)
        # grasp-center point = palm body pos + offset (in palm frame) toward the fingers
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.palm_center_w = palm_pos + quat_apply(palm_quat, self.palm_center_offset)
        self.palm_center_b = self.palm_center_w - root
        # mean fingertip -> pen distance (for the approach ratchet) and keypoint pose distance
        self.mean_ft_dist = torch.norm(self.ee_pos_w - self.object_pos_w.unsqueeze(1), dim=-1).mean(dim=1)
        self.keypoint_dist = self._keypoint_distance()

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        obs = torch.cat(
            (
                self.robot.data.joint_pos,  # 19
                self.robot.data.joint_vel,  # 19
                self.ee_pos_b,  # 15
                self.palm_center_b,  # 3
                self.object_pos_b,  # 3
                self.object_quat_w,  # 4
                self.pen_big_axis_w,  # 3 (current big-end axis in world)
                self.goal_axis,  # 3 (randomized target big-end direction)
                self.target_pos,  # 3 (fixed, env-local)
                self.actions,  # 19
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        z = self.object_pos_w[:, 2]
        d = self.keypoint_dist
        dft = self.mean_ft_dist

        # ---- r_smooth: -lambda * L1(joint_vel), split arm / hand ----
        jv = self.robot.data.joint_vel.abs()
        r_smooth = (
            -self.cfg.lambda_arm_smooth * jv[:, self.arm_mask].sum(dim=1)
            - self.cfg.lambda_hand_smooth * jv[:, self.hand_mask].sum(dim=1)
        )

        # ---- r_approach: progress ratchet on the mean fingertip->pen distance ----
        r_approach = self.cfg.lambda_approach * torch.clamp(self.min_ft_dist - dft, min=0.0)
        self.min_ft_dist = torch.minimum(self.min_ft_dist, dft)

        # ---- r_lift: dense height ramp + one-time bonus, ONLY before grasped ----
        not_grasped = ~self.grasped_latch
        r_lift = not_grasped.float() * (self.cfg.lambda_lift * torch.clamp(z - self.object_init_z, min=0.0))
        newly_lifted = not_grasped & (z >= self.lift_height)
        r_lift = r_lift + newly_lifted.float() * self.cfg.bonus_lifted
        # initialize the goal ratchet at the moment of grasp, then latch I_grasped
        self.min_goal_dist[newly_lifted] = d[newly_lifted]
        self.grasped_latch = self.grasped_latch | (z >= self.lift_height)

        # ---- r_goal: keypoint progress ratchet + sparse success, ONLY after grasped ----
        # (before grasp min_goal_dist=inf, so guard with where() to avoid 0*inf = nan)
        grasped = self.grasped_latch
        progress = torch.clamp(self.min_goal_dist - d, min=0.0)
        r_goal = torch.where(grasped, self.cfg.lambda_goal * progress, torch.zeros_like(progress))
        self.min_goal_dist = torch.where(grasped, torch.minimum(self.min_goal_dist, d), self.min_goal_dist)
        success = grasped & (d < self.cfg.success_eps)
        r_goal = r_goal + success.float() * self.cfg.bonus_success

        # on success, sample a fresh orientation goal and restart the ratchet from the new d
        if self.cfg.resample_goal_on_success:
            succ_ids = torch.nonzero(success, as_tuple=False).flatten()
            if len(succ_ids) > 0:
                self._resample_goal(succ_ids)
                self.min_goal_dist[succ_ids] = self._keypoint_distance()[succ_ids]

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["grasped_frac"] = grasped.float().mean()
        self.extras["log"]["keypoint_dist"] = d.mean()
        self.extras["log"]["success_frac"] = success.float().mean()
        self.extras["log"]["lifted_frac"] = (z > self.lift_height).float().mean()

        return r_smooth + r_approach + r_lift + r_goal

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - self.cfg.drop_height)
        return dropped, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # robot to default pose
        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.dof_targets[env_ids] = joint_pos

        # pen pose: default + xy noise + random yaw
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

        # reset per-episode stateful reward variables
        self.grasped_latch[env_ids] = False
        self.min_goal_dist[env_ids] = float("inf")
        self.object_init_z[env_ids] = object_state[:, 2]

        self._resample_goal(env_ids)
        self.actions[env_ids] = 0.0
        self._compute_intermediate_values()
        # initialize the approach ratchet from the post-reset fingertip distance
        self.min_ft_dist[env_ids] = self.mean_ft_dist[env_ids]
