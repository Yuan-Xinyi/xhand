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

        self.ee_ids, _ = self.robot.find_bodies(self.cfg.ee_body_names)
        self.palm_idx = self.robot.body_names.index(self.cfg.palm_body_name)
        self.palm_center_offset = torch.tensor(
            self.cfg.palm_center_offset, dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.pen_big_end_axis = torch.tensor(
            self.cfg.pen_big_end_axis, dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.table_normal = torch.tensor(self.cfg.table_normal, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # 4 object-frame keypoints at (+-sx/2, +-sy/2, +-sz/2); large scale on the pen long axis
        sx, sy, sz = (s / 2.0 for s in self.cfg.keypoint_scales)
        self.keypoint_offsets = torch.tensor(
            [[sx, sy, sz], [sx, sy, -sz], [-sx, -sy, sz], [-sx, -sy, -sz]], dtype=torch.float, device=self.device
        )
        self.num_kp = self.keypoint_offsets.shape[0]

        finger_keys = ("index", "mid", "ring", "pinky", "thumb")
        hand = [any(k in n for k in finger_keys) for n in self.robot.joint_names]
        self.hand_mask = torch.tensor(hand, dtype=torch.bool, device=self.device)
        self.arm_mask = ~self.hand_mask

        limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower = limits[..., 0]
        self.dof_upper = limits[..., 1]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.ema_action = torch.zeros_like(self.actions)
        self.dof_targets = self.default_joint_pos.clone()
        scale = torch.full((self.cfg.action_space,), self.cfg.action_scale, device=self.device)
        scale[self.arm_mask] = self.cfg.arm_action_scale
        self.joint_action_scale = scale

        self.object_default_z = self.object.data.default_root_state[:, 2].clone()
        self.lift_height = self.object_default_z + self.cfg.lift_margin
        self.object_init_z = self.object_default_z.clone()
        self.grasped_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.min_ft_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.min_goal_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        # dmin_ep tracks the per-episode BEST distance to the FINAL goal (NOT the active
        # waypoint) -- it must NOT reset on a waypoint advance, or it would read ~waypoint_step
        # forever and hide whether the pen actually approaches the destination.
        self.dmin_ep = torch.full((self.num_envs,), float("inf"), device=self.device)
        # how many waypoints into the CURRENT journey the active goal is (reset to 0 on a fresh
        # final). Climbing 0->~N = real forward progress; stuck near 1 = jitter at wp_1.
        self.wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # in-hand bootstrap bookkeeping
        self.inhand_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fromtable_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.closed_qpos = self.default_joint_pos.clone()
        self.closed_qpos[:, self.hand_mask] = self.default_joint_pos[:, self.hand_mask] + self.cfg.inhand_close_frac * (
            self.dof_upper[:, self.hand_mask] - self.default_joint_pos[:, self.hand_mask]
        )
        self.default_palm_center_b = None
        self.default_palm_quat = None

        # ---- goal: a FINAL target pose + a moving WAYPOINT that advances toward it ----
        # final_pos/final_quat: episode goal (env-local pos + full orientation).
        # target_pos (the ACTIVE waypoint, reused by keypoint dist / marker / obs) advances
        # toward final_pos by cfg.waypoint_step each time it is reached. goal_quat (active
        # orientation) == final_quat (the orientation goal does not need waypointing -- the
        # in-hand pen is born already aligned to it). goal_axis = final big-end direction.
        self.final_pos = torch.tensor(self.cfg.target_pos, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.target_pos = self.final_pos.clone()
        self.goal_axis = self.table_normal.clone()
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_quat[:, 0] = 1.0
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self._resample_goal(self.robot._ALL_INDICES)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
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
        add_ceiling_fluorescent_lights()

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.ema_action = self.cfg.action_ema * self.actions + (1.0 - self.cfg.action_ema) * self.ema_action
        targets = self.dof_targets + self.joint_action_scale * self.ema_action
        self.dof_targets = torch.clamp(targets, self.dof_lower, self.dof_upper)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.dof_targets)

    # ------------------------------------------------------------------ goal
    def _resample_goal(self, env_ids, near_previous: bool = False):
        """Sample the FINAL target pose (position + cone orientation). Does NOT set the
        waypoint (target_pos) -- that is initialized from the pen pose by _set_waypoint."""
        n = len(env_ids)
        rx, ry, rz = self.cfg.target_pos_range_x, self.cfg.target_pos_range_y, self.cfg.target_pos_range_z
        if near_previous:
            direction = torch.randn((n, 3), device=self.device)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            radius = self.cfg.goal_seq_pos_delta * torch.rand((n, 1), device=self.device) ** (1.0 / 3.0)
            fp = self.final_pos[env_ids] + direction * radius
        else:
            fp = torch.empty((n, 3), device=self.device)
            fp[:, 0] = sample_uniform(rx[0], rx[1], (n,), device=self.device)
            fp[:, 1] = sample_uniform(ry[0], ry[1], (n,), device=self.device)
            fp[:, 2] = sample_uniform(rz[0], rz[1], (n,), device=self.device)
        fp[:, 0] = fp[:, 0].clamp(rx[0], rx[1])
        fp[:, 1] = fp[:, 1].clamp(ry[0], ry[1])
        fp[:, 2] = fp[:, 2].clamp(rz[0], rz[1])
        self.final_pos[env_ids] = fp
        # final big-end direction, sampled uniformly within the cone of +Z
        cos_max = math.cos(self.cfg.goal_cone_angle)
        cos_theta = 1.0 - torch.rand(n, device=self.device) * (1.0 - cos_max)
        theta = torch.acos(cos_theta.clamp(-1.0, 1.0))
        phi = 2.0 * math.pi * torch.rand(n, device=self.device)
        sin_theta = torch.sin(theta)
        axis = torch.stack([sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)], dim=-1)
        self.goal_axis[env_ids] = axis
        self.goal_quat[env_ids] = self._quat_from_two_vectors(self.pen_big_end_axis[env_ids], axis)

    def _set_waypoint(self, env_ids):
        """Place the active waypoint cfg.waypoint_step ahead of the pen toward the final goal."""
        pen_local = self.object_pos_w[env_ids] - self.scene.env_origins[env_ids]
        to_final = self.final_pos[env_ids] - pen_local
        dist = to_final.norm(dim=-1, keepdim=True)
        step = torch.minimum(dist, torch.full_like(dist, self.cfg.waypoint_step))
        self.target_pos[env_ids] = pen_local + to_final / dist.clamp(min=1e-6) * step
        self._update_goal_marker()

    def _advance_waypoints(self, reached_ids):
        """A reached waypoint advances one step toward the final goal; if it IS the final goal,
        resample a fresh (nearby) final and restart the waypoint from the current pen pose."""
        to_final = self.final_pos[reached_ids] - self.target_pos[reached_ids]
        dist = to_final.norm(dim=-1)
        at_final = dist <= (self.cfg.waypoint_step + 1e-6)
        # advance toward final by one step (clamped so it never overshoots)
        step = torch.minimum(dist, torch.full_like(dist, self.cfg.waypoint_step))
        self.target_pos[reached_ids] = (
            self.target_pos[reached_ids] + to_final / dist.clamp(min=1e-6).unsqueeze(-1) * step.unsqueeze(-1)
        )
        # terminal reach (waypoint == final): new final near the old one, waypoint from the pen
        self.wp_idx[reached_ids] += 1  # advanced one waypoint along the journey
        term = reached_ids[at_final]
        if len(term) > 0:
            self._resample_goal(term, near_previous=True)
            self._set_waypoint(term)
            self.wp_idx[term] = 0  # new final -> new journey starts at waypoint 0
        self._update_goal_marker()
        # PITFALL A: recompute keypoint dist against the NEW waypoint and reseed the ratchet
        # there, else max(min_goal_dist - d, 0) = 0 over the first half of every new segment.
        self._compute_intermediate_values()
        self.min_goal_dist[reached_ids] = self.keypoint_dist[reached_ids]
        return at_final

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
        kp = self.keypoint_offsets.unsqueeze(0).expand(self.num_envs, self.num_kp, 3)
        obj_q = self.object_quat_w.unsqueeze(1).expand(self.num_envs, self.num_kp, 4)
        goal_q = self.goal_quat.unsqueeze(1).expand(self.num_envs, self.num_kp, 4)
        o_kp = self.object_pos_w.unsqueeze(1) + quat_apply(obj_q, kp)
        g_kp = (self.target_pos + self.scene.env_origins).unsqueeze(1) + quat_apply(goal_q, kp)
        return torch.norm(o_kp - g_kp, dim=-1).max(dim=1).values

    def _compute_intermediate_values(self):
        root = self.robot.data.root_pos_w
        self.ee_pos_w = self.robot.data.body_pos_w[:, self.ee_ids]
        self.object_pos_w = self.object.data.root_pos_w
        self.object_quat_w = self.object.data.root_quat_w
        self.object_pos_b = self.object_pos_w - root
        self.ee_pos_b = (self.ee_pos_w - root.unsqueeze(1)).reshape(self.num_envs, -1)
        self.pen_big_axis_w = quat_apply(self.object_quat_w, self.pen_big_end_axis)
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.palm_center_w = palm_pos + quat_apply(palm_quat, self.palm_center_offset)
        self.palm_center_b = self.palm_center_w - root
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
                self.pen_big_axis_w,  # 3
                self.goal_axis,  # 3 (final big-end direction)
                self.target_pos,  # 3 (ACTIVE waypoint, env-local)
                self.actions,  # 19
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        z = self.object_pos_w[:, 2]
        d = self.keypoint_dist
        dft = self.mean_ft_dist

        jv = self.robot.data.joint_vel.abs()
        r_smooth = (
            -self.cfg.lambda_arm_smooth * jv[:, self.arm_mask].sum(dim=1)
            - self.cfg.lambda_hand_smooth * jv[:, self.hand_mask].sum(dim=1)
        )
        obj_vel = self.object.data.root_lin_vel_w.norm(dim=-1) + self.object.data.root_ang_vel_w.norm(dim=-1)
        r_smooth = r_smooth - self.cfg.lambda_obj_vel * obj_vel

        not_grasped = ~self.grasped_latch
        r_approach = not_grasped.float() * self.cfg.lambda_approach * torch.clamp(self.min_ft_dist - dft, min=0.0)
        self.min_ft_dist = torch.minimum(self.min_ft_dist, dft)

        r_lift = not_grasped.float() * (self.cfg.lambda_lift * torch.clamp(z - self.object_init_z, min=0.0))
        newly_lifted = not_grasped & (z >= self.lift_height)
        r_lift = r_lift + newly_lifted.float() * self.cfg.bonus_lifted
        self.min_goal_dist[newly_lifted] = d[newly_lifted]
        self.grasped_latch = self.grasped_latch | (z >= self.lift_height)

        grasped = self.grasped_latch
        progress = torch.clamp(self.min_goal_dist - d, min=0.0)
        r_goal = torch.where(grasped, self.cfg.lambda_goal * progress, torch.zeros_like(progress))
        self.min_goal_dist = torch.where(grasped, torch.minimum(self.min_goal_dist, d), self.min_goal_dist)

        # PITFALL B: track the per-episode BEST distance to the FINAL goal (not the moving
        # waypoint), measured BEFORE the advance (which may resample the final on a terminal).
        final_dist = torch.norm(self.object_pos_w - (self.final_pos + self.scene.env_origins), dim=-1)
        self.dmin_ep = torch.minimum(self.dmin_ep, final_dist)

        # reaching the ACTIVE waypoint: pay the success bonus and advance the waypoint
        reached = grasped & (d < self.cfg.success_eps)
        r_goal = r_goal + reached.float() * self.cfg.bonus_success
        terminal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reached_ids = torch.nonzero(reached, as_tuple=False).flatten()
        if len(reached_ids) > 0:
            at_final = self._advance_waypoints(reached_ids)
            terminal[reached_ids] = at_final

        if "log" not in self.extras:
            self.extras["log"] = dict()
        zero = torch.zeros((), device=self.device)
        ft, ih = self.fromtable_mask, self.inhand_mask
        self.extras["log"]["grasped_frac"] = grasped.float().mean()
        self.extras["log"]["grasped_frac_fromtable"] = grasped[ft].float().mean() if ft.any() else zero
        self.extras["log"]["lifted_frac"] = (z > self.lift_height).float().mean()
        self.extras["log"]["waypoint_reached_frac"] = reached.float().mean()
        self.extras["log"]["terminal_success_frac"] = terminal.float().mean()
        self.extras["log"]["active_wp_idx_inhand"] = self.wp_idx[ih].float().mean() if ih.any() else zero
        self.extras["log"]["keypoint_dist_inhand"] = d[ih].mean() if ih.any() else zero  # to active waypoint
        self.extras["log"]["dmin_ep_inhand"] = self.dmin_ep[ih].mean() if ih.any() else zero  # BEST to final
        self.extras["log"]["final_dist_inhand"] = final_dist[ih].mean() if ih.any() else zero  # now to final
        self.extras["log"]["rgoal_mean_inhand"] = r_goal[ih].mean() if ih.any() else zero

        return r_smooth + r_approach + r_lift + r_goal

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        z = self.object_pos_w[:, 2]
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fell_off = z < (self.object_default_z - self.cfg.drop_height)
        # from-table envs that grasp then drop end early; in-hand envs are NOT killed on drop
        dropped_after_lift = (
            self.grasped_latch & self.fromtable_mask & (z < self.object_init_z + 0.5 * self.cfg.lift_margin)
        )
        return (fell_off | dropped_after_lift), time_out

    def _inhand_pen_pose(self, ids: torch.Tensor) -> torch.Tensor:
        """In-hand pen: held at the (default-arm) palm center, oriented to the FINAL orientation
        (goal_quat), so the goal phase starts orientation-aligned. grasp_probe --pen_orient up
        confirms the closed hand holds it near-vertical."""
        pos = self.scene.env_origins[ids] + self.default_palm_center_b.unsqueeze(0)
        return torch.cat([pos, self.goal_quat[ids]], dim=-1)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # write DEFAULT pose first, then capture the constant default-arm palm transform once
        default_pos = self.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), env_ids=env_ids)
        if self.default_palm_center_b is None:
            self._compute_intermediate_values()
            self.default_palm_center_b = (self.palm_center_w - self.scene.env_origins).mean(dim=0)
            self.default_palm_quat = self.robot.data.body_quat_w[:, self.palm_idx][0].clone()

        inhand = torch.rand(n, device=self.device) < self.cfg.inhand_reset_frac
        self.inhand_mask[env_ids] = inhand
        self.fromtable_mask[env_ids] = ~inhand

        joint_pos = default_pos.clone()
        joint_pos[inhand] = self.closed_qpos[env_ids][inhand]
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.dof_targets[env_ids] = joint_pos

        # all envs first get a table placement (fixes object_init_z reference)
        object_state = self.object.data.default_root_state[env_ids].clone()
        nx, ny = self.cfg.reset_object_pos_noise
        noise = sample_uniform(-1.0, 1.0, (n, 2), device=self.device)
        object_state[:, 0] += nx * noise[:, 0]
        object_state[:, 1] += ny * noise[:, 1]
        object_state[:, 0:3] += self.scene.env_origins[env_ids]
        lo, hi = self.cfg.reset_object_yaw_range
        yaw = sample_uniform(lo, hi, (n,), device=self.device)
        z_axis = torch.zeros((n, 3), device=self.device)
        z_axis[:, 2] = 1.0
        object_state[:, 3:7] = quat_mul(quat_from_angle_axis(yaw, z_axis), object_state[:, 3:7])
        self.object_init_z[env_ids] = object_state[:, 2]  # table rest z for ALL (pitfall 2)
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)

        # sample the FINAL goal first (in-hand pen is oriented to it), then override in-hand pens
        self._resample_goal(env_ids)
        if inhand.any():
            ih = env_ids[inhand]
            self.object.write_root_pose_to_sim(self._inhand_pen_pose(ih), ih)
            self.object.write_root_velocity_to_sim(torch.zeros((len(ih), 6), device=self.device), ih)

        self.grasped_latch[env_ids] = False
        self.grasped_latch[env_ids[inhand]] = True
        self.min_goal_dist[env_ids] = float("inf")
        self.dmin_ep[env_ids] = float("inf")
        self.wp_idx[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.ema_action[env_ids] = 0.0

        self._compute_intermediate_values()  # fresh pen pose
        self._set_waypoint(env_ids)  # waypoint = pen + one step toward final
        self._compute_intermediate_values()  # keypoint dist against the waypoint
        self.min_ft_dist[env_ids] = self.mean_ft_dist[env_ids]
        # PITFALL 1: in-hand envs are born grasped -> seed d* with the (near) waypoint distance
        if inhand.any():
            self.min_goal_dist[env_ids[inhand]] = self.keypoint_dist[env_ids[inhand]]
