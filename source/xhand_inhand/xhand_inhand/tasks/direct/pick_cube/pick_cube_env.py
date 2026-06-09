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
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

from ..scene_lighting import add_ceiling_fluorescent_lights
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

        # joint limits / defaults
        limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower = limits[..., 0]
        self.dof_upper = limits[..., 1]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        # action / target buffers (full relative joint control)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.dof_targets = self.default_joint_pos.clone()

        # cube rest height -> "lifted" threshold (mirror of franka `object_is_lifted`)
        self.object_default_z = self.object.data.default_root_state[:, 2].clone()
        self.lift_height = self.object_default_z + self.cfg.lift_margin

        # fixed goal point (env-local) + per-episode target orientation
        self.target_pos = torch.tensor(self.cfg.target_pos, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat[:, 0] = 1.0
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
        root = self.robot.data.root_pos_w
        self.ee_pos_w = self.robot.data.body_pos_w[:, self.ee_ids]  # (N, 5, 3)
        self.object_pos_w = self.object.data.root_pos_w
        self.object_quat_w = self.object.data.root_quat_w
        self.object_pos_b = self.object_pos_w - root
        self.ee_pos_b = (self.ee_pos_w - root.unsqueeze(1)).reshape(self.num_envs, -1)  # (N, 15)
        # grasp-center point = palm body pos + offset (in palm frame) toward the fingers
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.palm_center_w = palm_pos + quat_apply(palm_quat, self.palm_center_offset)
        self.palm_center_b = self.palm_center_w - root

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
                self.target_pos,  # 3 (fixed, env-local)
                self.target_quat,  # 4
                self.actions,  # 19
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # ---- reaching: grasp assembly (palm-center + fingertips) -> cube center ----
        reach_points = torch.cat([self.palm_center_w.unsqueeze(1), self.ee_pos_w], dim=1)  # (N, 6, 3)
        reach_dist = torch.norm(reach_points - self.object_pos_w.unsqueeze(1), dim=-1)  # (N, 6)
        reach = (1.0 - torch.tanh(reach_dist / self.cfg.reach_std)).mean(dim=1) * self.cfg.w_reach

        # ---- lifting: height-gated (mirror of franka `object_is_lifted`) ----
        lifted = (self.object_pos_w[:, 2] > self.lift_height).float()
        lift_reward = lifted * self.cfg.w_lift

        # ---- position tracking: cube -> fixed target point, GATED by lift ----
        target_w = self.target_pos + self.scene.env_origins
        pos_dist = torch.norm(self.object_pos_w - target_w, p=2, dim=-1)
        goal_track = lifted * (1.0 - torch.tanh(pos_dist / self.cfg.goal_track_std)) * self.cfg.w_goal_track
        goal_track_fine = (
            lifted * (1.0 - torch.tanh(pos_dist / self.cfg.goal_track_fine_std)) * self.cfg.w_goal_track_fine
        )

        # ---- orientation tracking: cube -> target orientation, GATED by lift ----
        rot_err = quat_error_magnitude(self.object_quat_w, self.target_quat)
        orient_track = lifted * (1.0 - torch.tanh(rot_err / self.cfg.orient_track_std)) * self.cfg.w_orient_track

        # ---- success: combined position + orientation error, GATED by lift ----
        success_val = (
            (1.0 - torch.tanh(pos_dist / self.cfg.success_pos_std))
            * (1.0 - torch.tanh(rot_err / self.cfg.success_rot_std))
            * lifted
        )
        success_reward = success_val * self.cfg.w_success

        # ---- regularization (franka weights: tiny) ----
        action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1) * self.cfg.w_action_rate
        joint_vel_pen = torch.sum(self.robot.data.joint_vel**2, dim=-1) * self.cfg.w_joint_vel
        self.prev_actions = self.actions.clone()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["lifted_frac"] = lifted.mean()
        self.extras["log"]["reach_mean"] = reach.mean()
        self.extras["log"]["success_frac"] = (
            (pos_dist < 0.05) & (rot_err < 0.2) & (lifted > 0.5)
        ).float().mean()

        return reach + lift_reward + goal_track + goal_track_fine + orient_track + success_reward + action_rate + joint_vel_pen

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
