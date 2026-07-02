# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 + XHand pick-a-pen with CrossDex action tokenization (Direct workflow).

Subclasses :class:`PickCubeTokenEnv`, so the ACTION pipeline is inherited verbatim
(7 arm joint deltas + a 9-dim eigengrasp token -> 12 absolute xhand joint targets).
Only the object, the goal, the observation layout and the reward/termination change.

REWARD (CrossDex / UniDexGrasp++ style -- DENSE, not ratchet):

    r = r_dis + r_height + r_xy + r_orient + r_tip + r_success

  * r_dis     = -(2*avg_fingertip_obj_dist + palm_obj_dist)       [dense, always on]
                continuously pulls the whole hand onto the pen every step -- no cap to run
                out, no "hover and farm" local optimum.
  * r_height  = close_gate * w_height * (lift / lift_target)      [gated on hand-close]
                only pays once the hand is actually AT the pen, so the policy can't farm
                height by knocking / shoving the pen up.
  * r_xy      = -w_xy * ||obj_xy - obj_init_xy||                  [dense]
                penalises sliding the pen across the table (anti "sweep it away").
  * r_orient  = lifted_gate * w_orient * relu(cos_tilt)           [our tip-down goal]
  * r_tip     = lifted_gate * w_tip * clamp(tip_clear/margin,0,1) [our tip-unoccluded goal]
  * r_success = success_bonus, once, when the full success state is HELD success_hold_steps
                (CrossDex requires the success pose to persist, not just flash for one frame).

SUCCESS: lift >= 10 cm AND pen upright within 20 deg (tip down) AND tip >= 2 cm below the
hand AND the hand is on the pen -- held for success_hold_steps consecutive steps.
"""
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.utils.math import quat_apply

from ..pick_cube_token.pick_cube_token_env import PickCubeTokenEnv
from .pick_pen_token_env_cfg import PickPenTokenEnvCfg


class PickPenTokenEnv(PickCubeTokenEnv):
    cfg: PickPenTokenEnvCfg

    def __init__(self, cfg: PickPenTokenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # pen-frame axes / offsets, broadcast per-env
        self.pen_up_axis = torch.tensor(cfg.pen_up_axis, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.pen_tip_offset = torch.tensor(cfg.pen_tip_offset, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.upright_cos = math.cos(cfg.upright_success_angle)

        # success must be HELD for success_hold_steps consecutive steps (CrossDex robustness)
        self._success_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # per-episode initial pen xy (world) for the horizontal-displacement penalty
        self._object_init_xy = self.object.data.default_root_state[:, :2].clone()

        # fresh intermediate values so obs/reward have pen fields before the first step
        self._compute_intermediate_values()

    # ------------------------------------------------------------------ goal
    def _resample_goal(self, env_ids):
        # fixed goal: pen standing TIP DOWN. Identity orientation puts the pen's local +Z
        # (big end) along world +Z, so the small end (tip) points straight down.
        self.target_quat[env_ids] = 0.0
        self.target_quat[env_ids, 0] = 1.0
        self._update_goal_marker()

    # ------------------------------------------------------------------ mdp
    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()

        # pen long-axis (big-end) direction in world, and its tilt vs the table normal (+Z)
        self.pen_up_axis_w = quat_apply(self.object_quat_w, self.pen_up_axis)  # (N, 3), unit
        self.cos_tilt = self.pen_up_axis_w[:, 2]  # == dot with world +Z

        # pen tip point (small end) in world / base frame
        self.tip_pos_w = self.object_pos_w + quat_apply(self.object_quat_w, self.pen_tip_offset)
        self.tip_pos_b = self.tip_pos_w - self.robot.data.root_pos_w

        # tip clearance = how far the tip sits BELOW the lowest hand point (finger pads + palm)
        hand_z = torch.cat([self.finger_pad_w[:, :, 2], self.palm_center_w[:, 2:3]], dim=1)  # (N, 6)
        self.hand_min_z = hand_z.min(dim=1).values
        self.tip_clear = self.hand_min_z - self.tip_pos_w[:, 2]

        # lift above the table rest (env origins sit at z=0, so world z == env-local z)
        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.actual_lift = object_z - self.object_default_z

        # hand-to-object distances (CrossDex): mean fingertip dist (weighted) + palm dist
        self.avg_ft_dist = self._curr_fingertip_distances.mean(dim=1)
        self.palm_dist = torch.norm(self.palm_center_w - self.object_pos_w, dim=-1)

        # horizontal pen displacement from its episode start (anti-slide)
        self.xy_disp = torch.norm(self.object_pos_w[:, :2] - self._object_init_xy, dim=-1)

    def _hand_close_gate(self) -> torch.Tensor:
        return (self.avg_ft_dist < self.cfg.ft_close_thr) & (self.palm_dist < self.cfg.palm_close_thr)

    def _compute_success_inst(self) -> torch.Tensor:
        """Instantaneous success state (before the hold requirement)."""
        return (
            (self.actual_lift >= self.cfg.lift_success_height - self.cfg.success_lift_band)
            & (self.cos_tilt >= self.upright_cos)
            & (self.tip_clear >= self.cfg.tip_clearance_margin)
            & self._hand_close_gate()
        )

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
                self.pen_up_axis_w,  # 3 (pen big-end direction in world)
                self.tip_pos_b,  # 3 (pen tip in base frame)
                self.target_pos,  # 3 (fixed, env-local)
                self.target_quat,  # 4 (fixed tip-down goal)
                self.actions,  # 16
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        close_gate = self._hand_close_gate()

        # 1) r_dis: dense pull of the whole hand onto the pen (fingertips weighted 2x)
        r_dis = -(cfg.w_ft_dist * self.avg_ft_dist + cfg.w_palm_dist * self.palm_dist)

        # 2) r_height: gated lift ramp (0 -> w_height as the pen rises to the target lift). Gated
        #    on the hand being ON the pen so it can't be farmed by knocking the pen upward.
        lift = torch.clamp(self.actual_lift, min=0.0)
        lift_ramp = torch.clamp(lift / cfg.lift_success_height, max=1.0)
        r_height = close_gate.float() * cfg.w_height * lift_ramp

        # 3) r_xy: penalise horizontal sliding of the pen away from its start
        r_xy = -cfg.w_xy * self.xy_disp

        # 4) r_orient + r_tip: our tip-down + tip-unoccluded goals, gated on grasped-and-lifted
        lifted_gate = close_gate & (lift > cfg.orient_gate_lift)
        r_orient = lifted_gate.float() * cfg.w_orient * torch.clamp(self.cos_tilt, min=0.0)
        tip_ramp = torch.clamp(self.tip_clear / cfg.tip_clearance_margin, min=0.0, max=1.0)
        r_tip = lifted_gate.float() * cfg.w_tip * tip_ramp

        # 5) r_success: sparse bonus once the full success state has been HELD long enough
        succ_inst = self._compute_success_inst()
        self._success_steps = torch.where(
            succ_inst, self._success_steps + 1, torch.zeros_like(self._success_steps)
        )
        held = self._success_steps >= cfg.success_hold_steps
        just_held = held & ~self._is_success
        r_success = just_held.float() * cfg.success_bonus
        self._is_success = self._is_success | held
        self.prev_actions = self.actions.clone()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        log = self.extras["log"]
        log["close_gate_frac"] = close_gate.float().mean()
        log["avg_ft_dist_mean"] = self.avg_ft_dist.mean()
        log["palm_dist_mean"] = self.palm_dist.mean()
        log["actual_lift_mean"] = self.actual_lift.mean()
        log["lifted_gate_frac"] = lifted_gate.float().mean()
        log["cos_tilt_mean"] = self.cos_tilt.mean()
        log["upright_frac"] = (self.cos_tilt >= self.upright_cos).float().mean()
        log["tip_clear_mean"] = self.tip_clear.mean()
        log["tip_clear_frac"] = (self.tip_clear >= cfg.tip_clearance_margin).float().mean()
        log["xy_disp_mean"] = self.xy_disp.mean()
        log["succ_inst_frac"] = succ_inst.float().mean()
        log["success_frac"] = self._is_success.float().mean()
        log["r_dis_mean"] = r_dis.mean()
        log["r_height_mean"] = r_height.mean()
        log["r_xy_mean"] = r_xy.mean()
        log["r_orient_mean"] = r_orient.mean()
        log["r_tip_mean"] = r_tip.mean()
        log["r_success_mean"] = r_success.mean()

        return r_dis + r_height + r_xy + r_orient + r_tip + r_success

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # keep the success latch consistent for same-step termination
        succ_inst = self._compute_success_inst()
        held = torch.where(
            succ_inst, self._success_steps + 1, torch.zeros_like(self._success_steps)
        ) >= self.cfg.success_hold_steps
        self._is_success = self._is_success | held

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - self.cfg.drop_height)
        if not self.cfg.terminate_on_drop:
            dropped = torch.zeros_like(dropped)
        return dropped | self._is_success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._success_steps[env_ids] = 0
        # capture the pen's start xy (world) for the horizontal-displacement penalty
        self._object_init_xy[env_ids] = self.object_pos_w[env_ids, :2]
