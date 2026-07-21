# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 + XHand pick-a-cube with CrossDex action tokenization (Direct workflow).

Subclasses :class:`PickCubeEnv` and changes ONLY the action pipeline:

  * arm  (action[:7])   -> relative joint position control (unchanged from pick_cube)
  * hand (action[7:16]) -> 9-dim eigengrasp token, decoded + retargeted to 12 absolute
                           xhand joint targets via the offline-trained RetargetingNN.

Everything else (scene, observations layout, reward, resets) is inherited.
"""
from __future__ import annotations

import torch

from ..pick_cube.pick_cube_env import PickCubeEnv
from .pick_cube_token_env_cfg import PickCubeTokenEnvCfg
from .retarget_infer import EigenRetarget


class PickCubeTokenEnv(PickCubeEnv):
    cfg: PickCubeTokenEnvCfg

    def __init__(self, cfg: PickCubeTokenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- load the CrossDex eigengrasp -> xhand-joint retargeting network ----
        self.retarget = EigenRetarget(
            weights_path=cfg.retarget_weights_path,
            meta_path=cfg.retarget_meta_path,
            device=str(self.device),
        )
        assert self.retarget.n_eigengrasps == cfg.n_hand_tokens, (
            f"token dim mismatch: cfg.n_hand_tokens={cfg.n_hand_tokens} "
            f"vs network n_eigengrasps={self.retarget.n_eigengrasps}"
        )

        # hand joint ids/names in articulation order, and the permutation that reorders the
        # network output (its own joint order) into the articulation hand-joint order.
        self._hand_ids_t = torch.as_tensor(self._hand_joint_ids, dtype=torch.long, device=self.device)
        self._arm_ids_t = torch.as_tensor(self._arm_joint_ids, dtype=torch.long, device=self.device)
        _, hand_joint_names = self.robot.find_joints(self.cfg.hand_joint_names)
        assert self.retarget.robot_dim == len(hand_joint_names), (
            f"hand dof mismatch: robot has {len(hand_joint_names)} hand joints, "
            f"network outputs {self.retarget.robot_dim}"
        )
        # nn_out[:, perm] is ordered like hand_joint_names (== order of self._hand_joint_ids)
        self._retarget2isaac = self.retarget.permutation_to(hand_joint_names)

        # split the index range of the action vector
        self._n_arm = int(getattr(cfg, "arm_action_dim", len(self._arm_joint_ids)))
        self._n_tokens = cfg.n_hand_tokens

    # ------------------------------------------------------------------ step
    def _decode_hand_action(self, hand_action: torch.Tensor) -> torch.Tensor:
        """Decode a hand action into absolute targets in articulation hand-joint order.

        Subclasses may extend the hand action while keeping the arm update, limit clamp and
        moving-average path below identical.  The base cube task remains exactly token-only.
        """
        if hand_action.shape[1] != self._n_tokens:
            raise ValueError(
                f"Expected {self._n_tokens} hand-token actions, got {hand_action.shape[1]}."
            )
        hand_nn = self.retarget.retarget_from_unit_action(hand_action)
        return hand_nn[:, self._retarget2isaac]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        a = actions.clone().clamp(-1.0, 1.0)
        self.actions = a

        arm_a = a[:, : self._n_arm]            # (N, 7) relative arm deltas
        hand_action = a[:, self._n_arm :]       # token, optionally extended by a subclass

        # arm: relative joint position control (same scheme as pick_cube)
        raw_targets = self.dof_targets.clone()
        if self._arm_ids_t.numel() > 0:
            raw_targets[:, self._arm_ids_t] = (
                self.dof_targets[:, self._arm_ids_t] + self.cfg.action_scale * arm_a
            )
        elif self._n_arm > 0:
            self._preprocess_virtual_arm_action(arm_a)

        # hand: subclass hook -> absolute targets in articulation hand-joint order
        hand_abs = self._decode_hand_action(hand_action)
        raw_targets[:, self._hand_ids_t] = hand_abs

        # clamp, then smooth with the moving average (CrossDex smooths the hand the same way)
        raw_targets = torch.clamp(raw_targets, self.dof_lower, self.dof_upper)
        self.dof_targets = (
            self.cfg.act_moving_average * raw_targets
            + (1.0 - self.cfg.act_moving_average) * self.dof_targets
        )
        self.dof_targets = torch.clamp(self.dof_targets, self.dof_lower, self.dof_upper)

    def _preprocess_virtual_arm_action(self, arm_action: torch.Tensor) -> None:
        """Hook for an arm-free environment that supplies virtual Cartesian actions."""

        raise RuntimeError("virtual arm actions require an environment override")
