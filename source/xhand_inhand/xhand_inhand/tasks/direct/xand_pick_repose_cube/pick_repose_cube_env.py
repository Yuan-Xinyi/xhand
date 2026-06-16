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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

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
        # per-finger TIP offset, aligned to the ACTUAL resolved ee_ids order (find_bodies may
        # reorder), so the green markers land on the real fingertips, not the proximal joints.
        ee_names = [self.robot.body_names[i] for i in self.ee_ids]
        self.fingertip_tip_offset = torch.tensor(
            [self.cfg.fingertip_tip_offsets[n] for n in ee_names], dtype=torch.float, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)  # (N, 5, 3)

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

        # ---- per-episode buffers ----
        # PICK/CARRY are anti-camping: lift is a one-shot latched bonus; carry is a ratchet that
        # pays only for beating the closest distance reached so far (sentinel 1e3 = not yet armed).
        self.lifted_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.best_pos_dist = torch.full((self.num_envs,), 1.0e3, device=self.device)
        # REPOSE is the xhand_repose-style continuous reorientation: count of goals solved this
        # episode (each success resamples a new goal orientation).
        self.successes = torch.zeros(self.num_envs, device=self.device)

        # fixed goal point (env-local) + per-episode target orientation
        self.target_pos = torch.tensor(self.cfg.target_pos, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat[:, 0] = 1.0
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self._resample_goal(self.robot._ALL_INDICES)

        # debug markers: palm-center (red), fingertips (green), palm-normal ray (blue).
        # only built with a GUI present -- headless training then pays nothing.
        self.dbg_markers = None
        if self.cfg.debug_markers and self.sim.has_gui():
            def _sphere(name, color, r):
                return VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path=f"/Visuals/dbg_{name}",
                        markers={name: sim_utils.SphereCfg(
                            radius=r,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                        )},
                    )
                )
            self.dbg_markers = {
                "palm": _sphere("palm", (1.0, 0.0, 0.0), 0.02),
                "ft": _sphere("ft", (0.0, 1.0, 0.0), 0.012),
                "normal": _sphere("normal", (0.0, 0.3, 1.0), 0.008),
            }

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
        self.object_pos_w = self.object.data.root_pos_w
        self.object_quat_w = self.object.data.root_quat_w
        self.object_pos_b = self.object_pos_w - root
        # corrected fingertip TIPS: the link2 body ORIGIN sits at the proximal (mid) joint, so
        # using it makes the hand grasp with its KNUCKLES. Add the per-finger tip offset (rotated
        # into world) so the "end effector" is the real fingertips -- used in BOTH obs and reward.
        ee_body_w = self.robot.data.body_pos_w[:, self.ee_ids]  # (N, 5, 3) proximal joints
        ft_quat = self.robot.data.body_quat_w[:, self.ee_ids]  # (N, 5, 4)
        ft_off = quat_apply(
            ft_quat.reshape(-1, 4), self.fingertip_tip_offset.reshape(-1, 3)
        ).reshape(self.num_envs, -1, 3)
        self.fingertip_tip_w = ee_body_w + ft_off  # (N, 5, 3) real tips
        self.ee_pos_w = self.fingertip_tip_w  # grasp assembly = real fingertips
        self.ee_pos_b = (self.ee_pos_w - root.unsqueeze(1)).reshape(self.num_envs, -1)  # (N, 15)
        # grasp-center point = palm body pos + offset (in palm frame) toward the fingers
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        self.palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.palm_center_w = palm_pos + quat_apply(self.palm_quat, self.palm_center_offset)
        self.palm_center_b = self.palm_center_w - root

    def _update_dbg_markers(self):
        org = self.scene.env_origins
        # palm-center (red)
        self.dbg_markers["palm"].visualize(self.palm_center_w)
        # fingertips (green): corrected TIPS (joint-origin + tip offset), (N,5,3) -> (N*5,3)
        self.dbg_markers["ft"].visualize(self.fingertip_tip_w.reshape(-1, 3))
        # palm normal (blue ray): palm-local -Y (the grasp side the palm faces),
        # drawn as a string of small spheres from the palm center outward.
        n_dir = torch.tensor([0.0, -1.0, 0.0], device=self.device).expand(self.num_envs, 3)
        palm_normal_w = quat_apply(self.palm_quat, n_dir)  # (N,3)
        ks = torch.arange(1, 9, device=self.device).float() * 0.012  # 8 pts, ~0.1 m ray
        ray = (
            self.palm_center_w.unsqueeze(1) + palm_normal_w.unsqueeze(1) * ks.view(1, -1, 1)
        ).reshape(-1, 3)
        self.dbg_markers["normal"].visualize(ray)

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
        # _get_dones() runs first each step and refreshes self.object_pos_w / object_quat_w
        # (via _compute_intermediate_values), which we use here.
        lifted = self.object_pos_w[:, 2] > self.lift_height
        liftedf = lifted.float()

        target_w = self.target_pos + self.scene.env_origins
        pos_dist = torch.norm(self.object_pos_w - target_w, p=2, dim=-1)
        rot_err = quat_error_magnitude(self.object_quat_w, self.target_quat)
        at_center = pos_dist < self.cfg.inhand_pos_thresh

        # ---- grasp approach (dense, DECAYS to zero after lift): palm-center + fingertips -> cube.
        # Bootstraps the grasp; gating by (1 - lifted) closes the "hold the grasp and farm reach"
        # loophole (dropping the cube terminates the episode, so the grasp is still enforced).
        reach_points = torch.cat([self.palm_center_w.unsqueeze(1), self.ee_pos_w], dim=1)  # (N, 6, 3)
        reach_dist = torch.norm(reach_points - self.object_pos_w.unsqueeze(1), dim=-1)  # (N, 6)
        reach = (1.0 - torch.tanh(reach_dist / self.cfg.reach_std)).mean(dim=1) * self.cfg.w_reach * (1.0 - liftedf)

        # ---- PICK: ONE-SHOT bonus the first time the cube clears the table (latched) ----
        newly_lifted = lifted & ~self.lifted_once
        lift_reward = newly_lifted.float() * self.cfg.w_lift
        self.lifted_once |= lifted

        # ---- CARRY (ratchet): pay ONLY for beating the closest distance reached so far.
        # holding still -> no new record -> 0; drifting back -> 0 (clamped, never negative).
        uninit_p = lifted & (self.best_pos_dist > 1.0e2)  # arm the ratchet at the lift moment
        self.best_pos_dist = torch.where(uninit_p, pos_dist, self.best_pos_dist)
        carry_reward = torch.clamp(self.best_pos_dist - pos_dist, min=0.0) * liftedf * self.cfg.w_carry
        self.best_pos_dist = torch.where(lifted, torch.minimum(self.best_pos_dist, pos_dist), self.best_pos_dist)

        # ---- REPOSE: xhand_repose / InHandManipulationEnv reward, VERBATIM but STAGED ----
        # active only once the cube is lifted AND carried to the goal point (the in-hand phase).
        repose_active = (lifted & at_center).float()
        # dense hyperbolic orientation reward: 1/(|rot_err| + rot_eps) * rot_reward_scale
        rot_reward = (1.0 / (torch.abs(rot_err) + self.cfg.rot_eps)) * self.cfg.rot_reward_scale * repose_active

        # success: orientation within tolerance, while at the goal point -> one-shot bonus, then
        # RESAMPLE a new target orientation (continuous in-hand reorientation; the moving goalpost
        # is what prevents camping here, exactly as in the original xhand_repose).
        solved_now = (torch.abs(rot_err) <= self.cfg.success_tolerance) & lifted & at_center
        success_bonus = solved_now.float() * self.cfg.reach_goal_bonus
        if solved_now.any():
            ids = solved_now.nonzero(as_tuple=False).squeeze(-1)
            self.successes[ids] += 1.0
            self._resample_goal(ids)

        # ---- regularization (tiny) ----
        action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1) * self.cfg.w_action_rate
        joint_vel_pen = torch.sum(self.robot.data.joint_vel**2, dim=-1) * self.cfg.w_joint_vel
        self.prev_actions = self.actions.clone()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["lifted_frac"] = liftedf.mean()
        self.extras["log"]["at_center_frac"] = (liftedf * at_center.float()).mean()
        self.extras["log"]["solved_frac"] = solved_now.float().mean()
        self.extras["log"]["successes_mean"] = self.successes.mean()
        self.extras["log"]["reach_mean"] = reach.mean()

        return reach + lift_reward + carry_reward + rot_reward + success_bonus + action_rate + joint_vel_pen

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # dropping the cube terminates (success no longer terminates -- it resamples the goal)
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - self.cfg.drop_height)
        return dropped, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset per-episode buffers (carry-ratchet sentinel 1e3 = not yet armed)
        self.lifted_once[env_ids] = False
        self.best_pos_dist[env_ids] = 1.0e3
        self.successes[env_ids] = 0.0
        # guard against a vectorized-indexing bug silently locking the carry ratchet across
        # episodes: after reset these envs MUST read the sentinel. Costs nothing when disabled.
        if self.cfg.debug_ratchet_asserts:
            assert torch.all(self.best_pos_dist[env_ids] == 1.0e3), "carry ratchet not reset"
            assert torch.all(self.successes[env_ids] == 0.0), "successes not reset"

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
