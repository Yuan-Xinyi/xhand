# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 + XHand claw-hammer nail task (Direct workflow, CrossDex tokenized actions).

Implements the task of ``claw_hammer_experiment_plan.md`` on top of the pick_cube_token
action pipeline (7 arm joint deltas + 9-dim eigengrasp token -> 12 xhand joint targets):

  PHASE 0 (PULL):  the hammer starts in the hand, the nail fully inserted in the guide
                   fixture. Align the CLAW slot with the nail head base (claw pull dir
                   ~ world +Z, i.e. handle up) and pull the nail up ``pull_target`` (3 cm).
  PHASE 1 (PRESS): (only if ``enable_press_phase``) reorient the hammer IN HAND so the
                   FACE normal points down at the nail, then press the nail back to
                   within ``press_success_depth`` of the inserted state.

REWARD (phase-wise, plan sec. 8):

  always:  - slip penalty (hammer grip point drifting from the palm center)
           - constrained-arm penalty (palm translation / wrist rotation beyond the
             episode-start pose limits, plan sec. 6)
           - drop penalty + termination when the hammer leaves the hand
           - L1 joint-velocity action regularization (arm heavier than hand)

  PULL:    - w_claw_dist * ||claw_slot - nail_head_base||          [dense approach]
           + w_claw_align * relu(cos(pull_dir, +Z))                [gated: claw near]
           + w_pull * ratcheted nail-UP progress                   [gated: claw ENGAGED]
           - w_wrong_pull * nail-UP progress while NOT engaged     [wrong-contact]
           + pull_bonus (once, at nail_q >= pull_target while engaged)

  PRESS:   - w_face_dist * ||face_center - nail_head_top||         [dense approach]
           + w_face_align * relu(cos(-face_normal, +Z))            [gated: face near]
           + w_press * ratcheted nail-DOWN progress                [gated: face ENGAGED]
           - w_wrong_press * nail-DOWN progress while NOT engaged  [wrong-contact]
           + success_bonus (held ``success_hold_steps``)

"Engaged" is a geometric proxy (no contact sensors): the functional point within an
engage radius of the nail head AND the functional axis aligned with the nail axis. Nail
progress only pays while the correct functional region is engaged, per the plan:
"success should require the correct functional region to be active".

SUCCESS:  MVP-1: nail_q >= pull_target while claw engaged and tool in hand, held.
          MVP-3: additionally press back to nail_q <= press_success_depth while face
          engaged and tool in hand, held.
"""
from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul, sample_uniform

from ..pick_cube_token.pick_cube_token_env import PickCubeTokenEnv
from .hammer_nail_env_cfg import HammerNailTokenEnvCfg


class HammerNailTokenEnv(PickCubeTokenEnv):
    cfg: HammerNailTokenEnvCfg

    def __init__(self, cfg: HammerNailTokenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        N = self.num_envs

        # ---- hammer functional geometry (local frame), broadcast per-env ----
        def _vec(v):
            return torch.tensor(v, dtype=torch.float, device=self.device).repeat((N, 1))

        self.claw_slot_local = _vec(cfg.claw_slot_local)
        self.claw_pull_dir_local = _vec(cfg.claw_pull_dir_local)
        self.face_center_local = _vec(cfg.face_center_local)
        self.face_normal_local = _vec(cfg.face_normal_local)
        self.grip_center_local = _vec(cfg.grip_center_local)

        # ---- in-hand reset placement (palm frame) ----
        self._in_palm_pos = _vec(cfg.hammer_in_palm_pos)
        self._in_palm_quat = _vec(cfg.hammer_in_palm_quat)

        # pre-shaped grasp joint targets, ordered like self._hand_joint_ids
        _, hand_joint_names = self.robot.find_joints(self.cfg.hand_joint_names)
        grasp_vals = [float(cfg.reset_hand_grasp_pos.get(n, 0.0)) for n in hand_joint_names]
        self._grasp_joint_pos = torch.tensor(grasp_vals, dtype=torch.float, device=self.device)

        # ---- nail articulation indices ----
        self._nail_body_ids, _ = self.nail.find_bodies("nail")
        self._nail_body_idx = self._nail_body_ids[0]
        self._nail_joint_ids, _ = self.nail.find_joints("nail_joint")
        self._nail_joint_idx = self._nail_joint_ids[0]

        # ---- per-episode task state ----
        self._phase = torch.zeros(N, dtype=torch.long, device=self.device)       # 0 pull, 1 press
        # episode-start nail depth; pull/press targets are DISPLACEMENTS relative to it
        self._nail_q0 = torch.zeros(N, device=self.device)
        self._nail_highest = torch.zeros(N, device=self.device)                  # up-ratchet (disp)
        self._nail_lowest = torch.full((N,), float(cfg.nail_travel), device=self.device)  # down-ratchet (disp)
        self._prev_nail_q = torch.zeros(N, device=self.device)
        self._pull_done = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._success_steps = torch.zeros(N, dtype=torch.long, device=self.device)
        # constrained-arm reference pose (episode-start palm pose, world frame)
        self._palm_start_pos = torch.zeros((N, 3), device=self.device)
        self._palm_start_quat = torch.zeros((N, 4), device=self.device)
        self._palm_start_quat[:, 0] = 1.0

        self._compute_intermediate_values()

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)          # the hammer
        self.nail = Articulation(self.cfg.nail_cfg)             # guide block + nail

        table_spawn = sim_utils.UsdFileCfg(usd_path=self.cfg.table_usd)
        table_spawn.func(
            "/World/envs/env_.*/Table", table_spawn, translation=self.cfg.table_pos, orientation=self.cfg.table_rot
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["nail"] = self.nail
        self.scene.rigid_objects["object"] = self.object

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ goal
    def _resample_goal(self, env_ids):
        # fixed goal: a marker at the pulled nail-head height (fixture + block top + 3 cm)
        self.target_quat[env_ids] = 0.0
        self.target_quat[env_ids, 0] = 1.0
        self._update_goal_marker()

    # ------------------------------------------------------------------ mdp
    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        cfg = self.cfg
        root = self.robot.data.root_pos_w

        # hammer functional frames in world
        self.claw_slot_w = self.object_pos_w + quat_apply(self.object_quat_w, self.claw_slot_local)
        self.claw_pull_dir_w = quat_apply(self.object_quat_w, self.claw_pull_dir_local)
        self.face_center_w = self.object_pos_w + quat_apply(self.object_quat_w, self.face_center_local)
        self.face_normal_w = quat_apply(self.object_quat_w, self.face_normal_local)
        self.grip_point_w = self.object_pos_w + quat_apply(self.object_quat_w, self.grip_center_local)
        self.claw_slot_b = self.claw_slot_w - root
        self.face_center_b = self.face_center_w - root

        # nail state: q (0 = flush with the block, + up), engagement point = nail body origin
        # (head base). Task progress is the DISPLACEMENT from the episode-start depth q0.
        self.nail_q = self.nail.data.joint_pos[:, self._nail_joint_idx]
        self.nail_qd = self.nail.data.joint_vel[:, self._nail_joint_idx]
        self.nail_disp = self.nail_q - self._nail_q0
        self.nail_engage_w = self.nail.data.body_pos_w[:, self._nail_body_idx]
        self.nail_top_w = self.nail_engage_w + torch.tensor(
            [0.0, 0.0, cfg.nail_head_thickness], device=self.device
        )
        self.nail_engage_b = self.nail_engage_w - root

        # functional-region engagement proxies. The claw target is the HOOK point: on the
        # shaft, claw_hook_offset BELOW the head underside -- resting on top of the head
        # (the v2 cheat) is ~2 cm away vertically and does not count.
        self.nail_hook_w = self.nail_engage_w.clone()
        self.nail_hook_w[:, 2] -= cfg.claw_hook_offset
        hook_delta = self.claw_slot_w - self.nail_hook_w
        self.claw_dist = torch.norm(hook_delta, dim=-1)
        self.claw_dist_xy = torch.norm(hook_delta[:, :2], dim=-1)
        self.claw_dz = hook_delta[:, 2]
        self.claw_align = self.claw_pull_dir_w[:, 2]            # cos vs world +Z
        self.claw_engaged = (
            (self.claw_dist_xy < cfg.claw_engage_radius_xy)
            & (self.claw_dz.abs() < cfg.claw_engage_dz)
            & (self.claw_align > cfg.claw_align_cos)
        )
        self.face_dist = torch.norm(self.face_center_w - self.nail_top_w, dim=-1)
        self.face_align = -self.face_normal_w[:, 2]             # cos of -normal vs +Z (face down)
        self.face_engaged = (self.face_dist < cfg.face_engage_radius) & (self.face_align > cfg.face_align_cos)

        # tool-in-hand: grip point vs palm center (slip / drop)
        self.grip_to_palm = torch.norm(self.grip_point_w - self.palm_center_w, dim=-1)
        self.tool_dropped = self.grip_to_palm > cfg.tool_drop_dist
        self.tool_in_hand = ~self.tool_dropped

        # constrained arm: palm drift from the episode-start pose
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        self.palm_disp = torch.norm(palm_pos - self._palm_start_pos, dim=-1)
        self.palm_rot_err = quat_error_magnitude(self.palm_quat, self._palm_start_quat)

    def _compute_success_inst(self) -> torch.Tensor:
        """Instantaneous success state (before the hold requirement)."""
        pulled = (self.nail_disp >= self.cfg.pull_target) & self.claw_engaged
        if not self.cfg.enable_press_phase:
            return pulled & self.tool_in_hand
        pressed = (
            (self._phase == 1)
            & (self.nail_disp <= self.cfg.press_success_depth)
            & self.face_engaged
        )
        return pressed & self.tool_in_hand

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        if self.dbg_markers is not None:
            self._update_dbg_markers()
        obs = torch.cat(
            (
                self.robot.data.joint_pos,          # 19
                self.robot.data.joint_vel,          # 19
                self.ee_pos_b,                      # 15
                self.palm_center_b,                 # 3
                self.object_pos_b,                  # 3  (hammer)
                self.object_quat_w,                 # 4
                self.claw_slot_b,                   # 3  (claw slot in base frame)
                self.face_center_b,                 # 3  (hammer face in base frame)
                self.claw_pull_dir_w,               # 3
                self.face_normal_w,                 # 3
                self.nail_engage_b,                 # 3  (nail head base in base frame)
                self.nail_disp.unsqueeze(-1),       # 1  (displacement from the start depth)
                self.nail_qd.unsqueeze(-1),         # 1
                self._phase.float().unsqueeze(-1),  # 1
                self.actions,                       # 16
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        p_pull = (self._phase == 0).float()
        p_press = (self._phase == 1).float()

        # ---- nail progress bookkeeping (displacement from q0, shared by both phases) ----
        # up-ratchet: only NEW max displacement inside [0, pull_target] can pay
        capped_d = torch.clamp(self.nail_disp, max=cfg.pull_target)
        up_delta = torch.clamp(capped_d - torch.clamp(self._nail_highest, max=cfg.pull_target), min=0.0)
        self._nail_highest = torch.maximum(self._nail_highest, self.nail_disp)
        # down-ratchet (press phase): only NEW min displacement inside [press_success_depth, ...) pays
        floored_d = torch.clamp(self.nail_disp, min=cfg.press_success_depth)
        down_delta = torch.clamp(
            torch.clamp(self._nail_lowest, min=cfg.press_success_depth) - floored_d, min=0.0
        )
        self._nail_lowest = torch.minimum(self._nail_lowest, self.nail_disp)
        # raw per-step motion for the wrong-contact penalties
        raw_delta = self.nail_q - self._prev_nail_q
        self._prev_nail_q = self.nail_q.clone()

        claw_eng = self.claw_engaged.float()
        face_eng = self.face_engaged.float()

        # ---- PULL phase ----
        r_claw_dist = -cfg.w_claw_dist * self.claw_dist
        claw_near = (self.claw_dist < cfg.claw_near_dist).float()
        r_claw_align = cfg.w_claw_align * claw_near * torch.clamp(self.claw_align, min=0.0)
        r_pull = cfg.w_pull * claw_eng * up_delta / cfg.pull_target
        wrong_pull = -cfg.w_wrong_pull * (1.0 - claw_eng) * torch.clamp(raw_delta, min=0.0) / cfg.pull_target
        pull_reached = (self.nail_disp >= cfg.pull_target) & self.claw_engaged & ~self._pull_done
        r_pull_bonus = pull_reached.float() * cfg.pull_bonus
        # phase transition (MVP-3): pull done -> press phase, arm the down-ratchet
        if cfg.enable_press_phase:
            entering = pull_reached & (self._phase == 0)
            self._phase = torch.where(entering, torch.ones_like(self._phase), self._phase)
            self._nail_lowest = torch.where(entering, self.nail_disp, self._nail_lowest)
        self._pull_done = self._pull_done | pull_reached

        r_phase_pull = p_pull * (r_claw_dist + r_claw_align) + r_pull + wrong_pull + r_pull_bonus

        # ---- PRESS phase (zero everywhere in MVP-1: phase never becomes 1) ----
        r_face_dist = -cfg.w_face_dist * self.face_dist
        face_near = (self.face_dist < cfg.face_near_dist).float()
        r_face_align = cfg.w_face_align * face_near * torch.clamp(self.face_align, min=0.0)
        r_press = cfg.w_press * face_eng * down_delta / cfg.pull_target
        wrong_press = (
            -cfg.w_wrong_press * p_press * (1.0 - face_eng) * torch.clamp(-raw_delta, min=0.0) / cfg.pull_target
        )
        r_phase_press = p_press * (r_face_dist + r_face_align + r_press) + wrong_press

        # ---- shared penalties: slip, constrained arm, drop, action regularization ----
        r_slip = -cfg.w_slip * torch.clamp(self.grip_to_palm - cfg.slip_free_dist, min=0.0)
        r_arm = -cfg.w_arm_translation * torch.clamp(self.palm_disp - cfg.arm_translation_limit, min=0.0)
        r_wrist = -cfg.w_wrist_rotation * torch.clamp(self.palm_rot_err - cfg.wrist_rotation_limit, min=0.0)
        r_drop = -cfg.drop_penalty * self.tool_dropped.float()
        jv = self.robot.data.joint_vel
        r_act = (
            -cfg.kuka_actions_penalty_scale * jv[:, self._arm_joint_ids].abs().sum(dim=-1)
            - cfg.hand_actions_penalty_scale * jv[:, self._hand_joint_ids].abs().sum(dim=-1)
        )
        self.prev_actions = self.actions.clone()

        # ---- success: full task state HELD success_hold_steps ----
        succ_inst = self._compute_success_inst()
        # per-step hold stream: makes STAYING in the success state the best-paying policy
        # (must beat the claw-align hover income, see cfg.w_success_hold)
        r_hold = cfg.w_success_hold * succ_inst.float()
        self._success_steps = torch.where(
            succ_inst, self._success_steps + 1, torch.zeros_like(self._success_steps)
        )
        held = self._success_steps >= cfg.success_hold_steps
        just_held = held & ~self._is_success
        r_success = just_held.float() * cfg.success_bonus
        self._is_success = self._is_success | held

        if "log" not in self.extras:
            self.extras["log"] = dict()
        log = self.extras["log"]
        log["nail_q_mean"] = self.nail_q.mean()
        log["nail_disp_mean"] = self.nail_disp.mean()
        log["nail_highest_mean"] = self._nail_highest.mean()
        log["claw_dist_mean"] = self.claw_dist.mean()
        log["claw_dist_xy_mean"] = self.claw_dist_xy.mean()
        log["claw_dz_mean"] = self.claw_dz.mean()
        log["claw_align_mean"] = self.claw_align.mean()
        log["claw_engaged_frac"] = claw_eng.mean()
        log["face_dist_mean"] = self.face_dist.mean()
        log["face_align_mean"] = self.face_align.mean()
        log["face_engaged_frac"] = face_eng.mean()
        log["pull_done_frac"] = self._pull_done.float().mean()
        log["press_phase_frac"] = p_press.mean()
        log["grip_to_palm_mean"] = self.grip_to_palm.mean()
        log["tool_dropped_frac"] = self.tool_dropped.float().mean()
        log["palm_disp_mean"] = self.palm_disp.mean()
        log["palm_rot_err_mean"] = self.palm_rot_err.mean()
        log["succ_inst_frac"] = succ_inst.float().mean()
        log["success_frac"] = self._is_success.float().mean()
        log["r_claw_dist_mean"] = (p_pull * r_claw_dist).mean()
        log["r_claw_align_mean"] = (p_pull * r_claw_align).mean()
        log["r_pull_mean"] = r_pull.mean()
        log["r_pull_bonus_mean"] = r_pull_bonus.mean()
        log["r_wrong_pull_mean"] = wrong_pull.mean()
        log["r_face_dist_mean"] = (p_press * r_face_dist).mean()
        log["r_face_align_mean"] = (p_press * r_face_align).mean()
        log["r_press_mean"] = (p_press * r_press).mean()
        log["r_wrong_press_mean"] = wrong_press.mean()
        log["r_slip_mean"] = r_slip.mean()
        log["r_arm_mean"] = (r_arm + r_wrist).mean()
        log["r_drop_mean"] = r_drop.mean()
        log["r_hold_mean"] = r_hold.mean()
        log["r_success_mean"] = r_success.mean()

        return (
            r_phase_pull
            + r_phase_press
            + r_slip
            + r_arm
            + r_wrist
            + r_drop
            + r_act
            + r_hold
            + r_success
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # keep the success latch consistent for same-step termination
        succ_inst = self._compute_success_inst()
        held = torch.where(
            succ_inst, self._success_steps + 1, torch.zeros_like(self._success_steps)
        ) >= self.cfg.success_hold_steps
        self._is_success = self._is_success | held

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.tool_dropped
        if self.cfg.terminate_on_success:
            terminated = terminated | self._is_success
        return terminated, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # skip PickCubeEnv._reset_idx (cube-on-table logic); DirectRLEnv resets the buffers
        DirectRLEnv._reset_idx(self, env_ids)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = len(env_ids)

        # robot: arm to the (deterministic) home pose, hand pre-shaped around the handle.
        # NOTE the placement below reads the palm pose from the CURRENT sim state; with
        # reset_arm_joint_noise = 0 the arm never leaves home, so this is exact.
        joint_pos = self.default_joint_pos[env_ids].clone()
        joint_pos[:, self._hand_ids_t] = self._grasp_joint_pos.unsqueeze(0)
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.dof_targets[env_ids] = joint_pos

        # hammer IN the hand: cfg pose relative to the palm body frame
        palm_pos = self.robot.data.body_pos_w[env_ids, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[env_ids, self.palm_idx]
        obj_state = torch.zeros((n, 13), device=self.device)
        obj_state[:, 0:3] = palm_pos + quat_apply(palm_quat, self._in_palm_pos[env_ids])
        obj_state[:, 3:7] = quat_mul(palm_quat, self._in_palm_quat[env_ids])
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)

        # nail: PROTRUDING so the claw fits under the head, + small depth noise (plan sec. 11)
        nail_q = self.cfg.nail_init_protrusion + sample_uniform(
            0.0, self.cfg.reset_nail_depth_noise, (n, self.nail.num_joints), device=self.device
        )
        self.nail.write_joint_state_to_sim(nail_q, torch.zeros_like(nail_q), env_ids=env_ids)

        # per-episode task state (pull/press progress is measured relative to q0)
        q0 = nail_q[:, self._nail_joint_idx]
        self._nail_q0[env_ids] = q0
        self._phase[env_ids] = 0
        self._nail_highest[env_ids] = 0.0
        self._nail_lowest[env_ids] = float(self.cfg.nail_travel)
        self._prev_nail_q[env_ids] = q0
        self._pull_done[env_ids] = False
        self._success_steps[env_ids] = 0
        self._is_success[env_ids] = False
        self._palm_start_pos[env_ids] = palm_pos
        self._palm_start_quat[env_ids] = palm_quat

        self._resample_goal(env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._compute_intermediate_values()
