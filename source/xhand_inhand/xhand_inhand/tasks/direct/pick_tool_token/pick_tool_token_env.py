# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 + XHand pick-a-tool with CrossDex action tokenization (Direct workflow).

Subclass of :class:`PickCubeTokenEnv`: inherits the entire CrossDex token action pipeline
(7 arm joint deltas + a 9-dim eigengrasp token retargeted to 12 absolute xhand joint
targets) AND the pick_cube staged lift reward / observations / resets. The object is a
hammer, and the ONE task-specific change is retargeting the fingertip-approach reward from
the mesh ROOT (the hammer's ungraspable bottom) to a set of HANDLE KEYPOINTS: each fingertip
is drawn to its NEAREST handle keypoint, so the fingers wrap the handle centerline instead
of poking one point / pressing with the back of the hand. See ``cfg.grasp_keypoints``.

Each keypoint gets a draggable yellow sphere (GUI only, child of env_0's Object prim). Drag
them in the viewport to reposition the handle keypoints; the new object-local coordinates are
read back and printed. Gated on the GUI, so headless training pays nothing.
"""
from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply

from ..pick_cube_token.pick_cube_token_env import PickCubeTokenEnv
from .pick_tool_token_env_cfg import PickToolTokenEnvCfg


class PickToolTokenEnv(PickCubeTokenEnv):
    cfg: PickToolTokenEnvCfg

    def __init__(self, cfg: PickToolTokenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # handle grasp keypoints in the object's local frame -- (K, 3)
        self.grasp_keypoints_local = torch.tensor(cfg.grasp_keypoints, dtype=torch.float, device=self.device)
        self._n_kp = self.grasp_keypoints_local.shape[0]

        # interactive keypoint calibration markers (GUI only): one draggable yellow sphere per
        # keypoint, authored as a child of env_0's Object prim. Drag them to move the keypoints;
        # the new object-local coords are read back and printed.
        self._grasp_marker_paths = []
        self._last_printed_keypoints = None
        self._enable_grasp_calib = bool(getattr(cfg, "debug_grasp_marker", True) and self.sim.has_gui())
        if self._enable_grasp_calib:
            self._create_object_grasp_markers()

        # refresh so the handle-based distances exist before the first step
        self._compute_intermediate_values()

    # ------------------------------------------------------------------ keypoint markers
    def _create_object_grasp_markers(self):
        stage = sim_utils.get_current_stage()
        obj_path = "/World/envs/env_0/Object"
        if not stage.GetPrimAtPath(obj_path).IsValid():
            self._enable_grasp_calib = False
            return
        for k, kp in enumerate(self.cfg.grasp_keypoints):
            marker_path = f"{obj_path}/dbg_kp_{k}"
            if not stage.GetPrimAtPath(marker_path).IsValid():
                cfg = sim_utils.SphereCfg(
                    radius=0.010,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
                )
                cfg.func(marker_path, cfg, translation=tuple(float(v) for v in kp))
            self._grasp_marker_paths.append(marker_path)

    def _sync_keypoints_from_markers(self):
        if not getattr(self, "_enable_grasp_calib", False) or not self._grasp_marker_paths:
            return
        stage = sim_utils.get_current_stage()
        new = []
        for path in self._grasp_marker_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return
            val = prim.GetAttribute("xformOp:translate").Get()
            if val is None:
                return
            new.append([float(val[0]), float(val[1]), float(val[2])])
        off = torch.tensor(new, dtype=torch.float, device=self.device)  # (K, 3)
        if torch.max(torch.abs(off - self.grasp_keypoints_local)) <= 1e-6:
            return
        self.grasp_keypoints_local = off
        if self._last_printed_keypoints is None or torch.max(torch.abs(off - self._last_printed_keypoints)) > 1e-3:
            self._last_printed_keypoints = off.clone()
            pts = ", ".join(f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})" for p in off.tolist())
            print(f"[grasp-calib] grasp_keypoints = ({pts},)")

    # ------------------------------------------------------------------ mdp
    def _compute_intermediate_values(self):
        # capture the reset sentinel BEFORE super() consumes it, so we can re-seed the
        # progress ratchet with HANDLE distances (super seeds it with ROOT distances).
        ft_sentinel = self._closest_fingertip_dist < 0.0
        super()._compute_intermediate_values()

        if getattr(self, "_enable_grasp_calib", False):
            self._sync_keypoints_from_markers()

        # handle keypoints in world: (N, K, 3)
        kp_w = torch.stack(
            [
                self.object_pos_w + quat_apply(self.object_quat_w, self.grasp_keypoints_local[k].expand(self.num_envs, 3))
                for k in range(self._n_kp)
            ],
            dim=1,
        )
        self.grasp_keypoints_w = kp_w
        self.grasp_center_w = kp_w.mean(dim=1)  # handle-region center (for logging/markers)

        # each fingertip pad -> its NEAREST handle keypoint (fingers wrap the handle centerline)
        # finger_pad_w (N, 5, 3), kp_w (N, K, 3) -> dists (N, 5, K) -> min over K -> (N, 5)
        dists = torch.norm(self.finger_pad_w.unsqueeze(2) - kp_w.unsqueeze(1), dim=-1)
        self._curr_fingertip_distances = dists.min(dim=2).values

        # re-seed the just-reset envs' ratchet with the handle-based distance
        self._closest_fingertip_dist = torch.where(
            ft_sentinel, self._curr_fingertip_distances, self._closest_fingertip_dist
        )

    # ------------------------------------------------------------------ grasp gate
    def _grasp_close_gate(self) -> torch.Tensor:
        """True where the hand is genuinely CLOSED on the handle keypoints: the thumb AND at
        least ``grasp_min_other_fingers`` other fingertips are within ``grasp_close_thr`` of a
        handle keypoint. The lift reward is gated on this, so the hammer cannot be knocked /
        scooped up for reward without an actual handle grasp. (Diagnostic on the ungated policy:
        it crossed the 10 cm lift with fingers still ~9 cm from the keypoints -- a knock.)"""
        d = self._curr_fingertip_distances  # (N, 5) fingertip -> nearest handle keypoint
        thr = self.cfg.grasp_close_thr
        thumb_close = d[:, self._thumb_ee_idx] < thr
        others_close = (d[:, self._other_ee_idx] < thr).sum(dim=1) >= self.cfg.grasp_min_other_fingers
        return thumb_close & others_close

    # ------------------------------------------------------------------ reward (grasp-gated lift)
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        actual_lift = object_z - self.object_default_z
        close_gate = self._grasp_close_gate()

        # ---- fingertip approach: ratchet progress toward each finger's nearest handle keypoint,
        #      active until the hand is closed on the handle. Guides the fingers ONTO the
        #      keypoints; the ratchet makes it non-farmable (only new closest distances pay). ----
        ft_deltas = torch.clamp(self._closest_fingertip_dist - self._curr_fingertip_distances, 0.0, 10.0)
        self._closest_fingertip_dist = torch.minimum(self._closest_fingertip_dist, self._curr_fingertip_distances)
        approach_active = (~close_gate).float()
        thumb_ft_rew = ft_deltas[:, self._thumb_ee_idx] * approach_active * cfg.distance_delta_rew_scale
        other_ft_rew = ft_deltas[:, self._other_ee_idx].sum(dim=-1) * approach_active * cfg.distance_delta_rew_scale
        ft_rew = thumb_ft_rew + other_ft_rew

        # ---- GATED lift: the sparse +bonus and the dense lift progress pay ONLY while the hand
        #      is closed on the handle, so a knock/scoop (fingers off the handle) earns nothing. ----
        grasped_lift = close_gate & (actual_lift > cfg.lifting_bonus_threshold)
        just_crossed = grasped_lift & ~self._lifted_object
        lift_bonus = just_crossed.float() * cfg.lifting_bonus
        self._lifted_object = self._lifted_object | grasped_lift

        floor = torch.full_like(actual_lift, cfg.lifting_bonus_threshold)
        ceil = torch.full_like(actual_lift, cfg.lift_success_height)
        prev = torch.maximum(self._highest_lift, floor)
        curr = torch.minimum(actual_lift, ceil)
        dense_delta = torch.clamp(curr - prev, min=0.0)
        dense_lift_rew = close_gate.float() * dense_delta * cfg.dense_lift_rew_scale
        # only advance the dense-lift ratchet while grasped, so a knock cannot consume the range
        self._highest_lift = torch.where(close_gate, torch.maximum(self._highest_lift, actual_lift), self._highest_lift)

        self._is_success = grasped_lift & (actual_lift >= cfg.lift_success_height)

        jv = self.robot.data.joint_vel
        kuka_pen = -cfg.kuka_actions_penalty_scale * jv[:, self._arm_joint_ids].abs().sum(dim=-1)
        hand_pen = -cfg.hand_actions_penalty_scale * jv[:, self._hand_joint_ids].abs().sum(dim=-1)
        self.prev_actions = self.actions.clone()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        log = self.extras["log"]
        log["close_gate_frac"] = close_gate.float().mean()
        log["avg_ft_to_kp_mean"] = self._curr_fingertip_distances.mean()
        log["thumb_to_kp_mean"] = self._curr_fingertip_distances[:, self._thumb_ee_idx].mean()
        log["grasped_lift_frac"] = grasped_lift.float().mean()
        log["actual_lift_mean"] = actual_lift.mean()
        log["highest_lift_mean"] = self._highest_lift.mean()
        log["success_frac"] = self._is_success.float().mean()
        log["ft_rew_mean"] = ft_rew.mean()
        log["lift_bonus_mean"] = lift_bonus.mean()
        log["dense_lift_rew_mean"] = dense_lift_rew.mean()

        return ft_rew + lift_bonus + dense_lift_rew + kuka_pen + hand_pen

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        object_z = self.object_pos_w[:, 2] - self.scene.env_origins[:, 2]
        actual_lift = object_z - self.object_default_z
        # success requires a real grasp (gate), not just the object being high
        self._is_success = self._grasp_close_gate() & (actual_lift >= self.cfg.lift_success_height)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped = self.object_pos_w[:, 2] < (self.object_default_z - self.cfg.drop_height)
        if not self.cfg.terminate_on_drop:
            dropped = torch.zeros_like(dropped)
        return dropped | self._is_success, time_out
