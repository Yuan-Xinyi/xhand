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
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_from_angle_axis, quat_mul, sample_uniform

from ..scene_lighting import add_ceiling_fluorescent_lights
from .fr3_reorient_env_cfg import Fr3ReorientEnvCfg


class Fr3ReorientEnv(DirectRLEnv):
    cfg: Fr3ReorientEnvCfg

    def __init__(self, cfg: Fr3ReorientEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # reach bodies: palm-center point + fingertips
        self.ee_ids, _ = self.robot.find_bodies(self.cfg.fingertip_body_names)
        self.palm_idx = self.robot.body_names.index(self.cfg.palm_body_name)
        self.palm_center_offset = torch.tensor(
            self.cfg.palm_center_offset, dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        # fingertip ids in the contact sensor + thumb position
        self.tip_ids, tip_names = self._contact_sensor.find_bodies(self.cfg.fingertip_body_names)
        self.thumb_local = tip_names.index(self.cfg.thumb_tip_name)

        limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower = limits[..., 0]
        self.dof_upper = limits[..., 1]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.dof_targets = self.default_joint_pos.clone()
        self.object_default_z = self.object.data.default_root_state[:, 2].clone()

        # goal: target position + full target orientation
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)  # env-local
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat[:, 0] = 1.0
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self._resample_goal(self.robot._ALL_INDICES)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._contact_sensor = ContactSensor(
            ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=0)
        )

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

        # lab fluorescent ceiling tubes + faint ambient fill
        add_ceiling_fluorescent_lights()

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.dof_targets + self.cfg.action_scale * self.actions
        self.dof_targets = torch.clamp(targets, self.dof_lower, self.dof_upper)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.dof_targets)

    # ------------------------------------------------------------------ goal
    def _resample_goal(self, env_ids):
        n = len(env_ids)
        rx, ry, rz = self.cfg.target_pos_range_x, self.cfg.target_pos_range_y, self.cfg.target_pos_range_z
        tp = torch.empty((n, 3), device=self.device)
        tp[:, 0] = sample_uniform(rx[0], rx[1], (n,), device=self.device)
        tp[:, 1] = sample_uniform(ry[0], ry[1], (n,), device=self.device)
        tp[:, 2] = sample_uniform(rz[0], rz[1], (n,), device=self.device)
        self.target_pos[env_ids] = tp
        # uniform random orientation (normalized gaussian quaternion)
        q = torch.randn((n, 4), device=self.device)
        self.target_quat[env_ids] = q / q.norm(dim=-1, keepdim=True)
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
        self.target_pos_b = self.target_pos
        # palm-center point (grasp center)
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.palm_center_w = palm_pos + quat_apply(palm_quat, self.palm_center_offset)

        # fingertip net contact force -> grasp gate
        net = self._contact_sensor.data.net_forces_w
        self.tip_contact_mag = torch.norm(net[:, self.tip_ids, :], dim=-1)  # (N, 5)
        in_contact = self.tip_contact_mag > self.cfg.contact_force_threshold
        thumb_c = in_contact[:, self.thumb_local]
        others = in_contact.clone()
        others[:, self.thumb_local] = False
        self.grasped = thumb_c & others.any(dim=1)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        obs = torch.cat(
            (
                self.robot.data.joint_pos,  # 19
                self.robot.data.joint_vel,  # 19
                self.ee_pos_b,  # 15
                self.tip_contact_mag.clamp(max=20.0),  # 5
                self.object_pos_b,  # 3
                self.object_quat_w,  # 4
                self.target_pos_b,  # 3
                self.target_quat,  # 4
                self.actions,  # 19
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        grasp = self.grasped.float()

        # reach: max distance over (palm-center + fingertips) to the cube center
        reach_points = torch.cat([self.palm_center_w.unsqueeze(1), self.ee_pos_w], dim=1)  # (N, 6, 3)
        ee_dist = torch.norm(reach_points - self.object_pos_w.unsqueeze(1), dim=-1)  # (N, 6)
        max_dist = ee_dist.max(dim=1).values
        reach = (1.0 - torch.tanh(max_dist / self.cfg.reach_std)) * self.cfg.w_reach

        # position tracking to the target, GATED by grasp
        pos_dist = torch.norm(self.object_pos_w - (self.target_pos + self.scene.env_origins), p=2, dim=-1)
        pos_track = (1.0 - torch.tanh(pos_dist / self.cfg.pos_track_std)) * grasp * self.cfg.w_pos_track

        # orientation tracking (full quaternion error), GATED by grasp
        rot_err = quat_error_magnitude(self.object_quat_w, self.target_quat)
        orient_track = (1.0 - torch.tanh(rot_err / self.cfg.rot_track_std)) * grasp * self.cfg.w_orient_track

        # success: combined pose error, gated by grasp
        success_val = (
            (1.0 - torch.tanh(pos_dist / self.cfg.success_pos_std))
            * (1.0 - torch.tanh(rot_err / self.cfg.success_rot_std))
            * grasp
        )
        success_reward = success_val * self.cfg.w_success

        action_l2 = torch.sum(self.actions**2, dim=-1) * self.cfg.w_action_l2
        action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1) * self.cfg.w_action_rate_l2
        self.prev_actions = self.actions.clone()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["grasp_frac"] = grasp.mean()
        self.extras["log"]["lifted_frac"] = (self.object_pos_w[:, 2] > (self.object_default_z + 0.06)).float().mean()
        self.extras["log"]["success_frac"] = ((pos_dist < 0.05) & (rot_err < 0.2) & self.grasped).float().mean()

        return reach + pos_track + orient_track + success_reward + action_l2 + action_rate

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - self.cfg.drop_height)
        return dropped, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.dof_targets[env_ids] = joint_pos

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
