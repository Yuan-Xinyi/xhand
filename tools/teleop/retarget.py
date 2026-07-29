# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MANO 21-keypoint -> xhand 12-joint retargeting for live teleoperation.

Thin wrapper around the SAME `dex_retargeting` DexPilot optimizer that
`tools/crossdex_retarget/build_retarget_nn.py` uses to generate its training set —
so the mapping a human hand drives at teleop time is identical to the one the RL
token policy was distilled from. Runs in the `wilor` conda env (has
`dex_retargeting`), NOT inside Isaac Lab.

Input : (21, 3) MANO joint positions in METERS (WiLoR `pred_keypoints_3d` already
        is; manopth returns mm, hence the /1000 in build_retarget_nn).
Output: (12,) xhand joint angles (rad) in `protocol.HAND_JOINT_NAMES` order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from protocol import HAND_JOINT_NAMES

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_URDF_DIR = _REPO / "source/xhand_inhand/xhand_inhand/assets/xhand2R32"
_CFG_FN = _REPO / "tools/crossdex_retarget/configs/xhand_right_dexpilot.yml"


# DexPilot only matches fingertip POSITIONS, so these five joints get no
# gradient and sit at their limit midpoints forever (index_joint2 &
# middle/ring/pinky_joint1 at 0.96 rad, thumb_joint2 at 0.83) — the hand looks
# half-curled no matter what the human does. We drive them directly from the
# human finger CURL instead: summed angles between consecutive phalanx segments
# of the matching MANO finger, mapped linearly onto the joint range.
#   wire joint index -> (4 MANO keypoint indices of that finger, mcp->tip)
_CURL_DRIVEN = {
    2: (1, 2, 3, 4),       # thumb_joint2  <- thumb curl
    5: (5, 6, 7, 8),       # index_joint2  <- index curl
    7: (9, 10, 11, 12),    # middle_joint1 <- middle curl
    9: (13, 14, 15, 16),   # ring_joint1   <- ring curl
    11: (17, 18, 19, 20),  # pinky_joint1  <- pinky curl
}
_CURL_STRAIGHT = 0.25   # rad of summed curl treated as "fully straight" (dead zone)
_CURL_FULL = 2.40       # rad of summed curl treated as "fully curled"


def _chain_curl(kp: np.ndarray, chain) -> float:
    """Summed bend angle (rad) along one finger's 3 phalanx segments."""
    a, b, c, d = chain
    total = 0.0
    prev = kp[b] - kp[a]
    for p, q in ((b, c), (c, d)):
        cur = kp[q] - kp[p]
        cosang = np.dot(prev, cur) / (np.linalg.norm(prev) * np.linalg.norm(cur) + 1e-9)
        total += float(np.arccos(np.clip(cosang, -1.0, 1.0)))
        prev = cur
    return total


class HandRetargeter:
    """DexPilot fingertip retargeting: MANO keypoints -> xhand joint targets."""

    def __init__(self, low_pass_alpha: float | None = None, curl_distal: bool = True):
        from dex_retargeting.retargeting_config import RetargetingConfig

        RetargetingConfig.set_default_urdf_dir(str(_URDF_DIR))
        override = dict(add_dummy_free_joint=True)
        if low_pass_alpha is not None:
            override["low_pass_alpha"] = low_pass_alpha
        cfg = RetargetingConfig.load_from_file(str(_CFG_FN), override=override)
        self._rt = cfg.build()

        # output order of rt.retarget(...)[6:] (drop the 6 dummy free-joint DOF)
        self._out_names = list(self._rt.joint_names[6:])
        # DexPilot builds fingertip vectors as jp[task_idx] - jp[origin_idx]
        idx = np.asarray(self._rt.optimizer.target_link_human_indices)
        self._origin_idx, self._task_idx = idx[0], idx[1]
        # permutation: retargeter output order -> canonical wire order
        self._perm = [self._out_names.index(n) for n in HAND_JOINT_NAMES]
        # curl-driven distal joints: EMA state matching the optimizer's low-pass
        self._curl_distal = curl_distal
        self._curl_alpha = 1.0 if low_pass_alpha is None else float(low_pass_alpha)
        self._curl_state: np.ndarray | None = None

    @property
    def output_joint_names(self):
        """xhand joint names in wire order (== protocol.HAND_JOINT_NAMES)."""
        return list(HAND_JOINT_NAMES)

    @property
    def joint_limits(self) -> np.ndarray:
        """(12, 2) [lower, upper] rad per joint, in wire order.

        NB: SeqRetargeting.joint_limits is in TARGET order (target_joint_names),
        unlike retarget()'s return which is scattered back to pinocchio dof
        order (joint_names) — the two need different permutations.
        """
        tgt = list(self._rt.optimizer.target_joint_names)[6:]
        lim = np.asarray(self._rt.joint_limits, dtype=np.float32)[6:]
        return lim[[tgt.index(n) for n in HAND_JOINT_NAMES]]

    def reset(self):
        """Clear the optimizer's warm-start / temporal low-pass state."""
        self._rt.reset()
        self._curl_state = None

    def retarget(self, keypoints_m: np.ndarray) -> np.ndarray:
        """(21,3) MANO keypoints in meters -> (12,) xhand joints in wire order."""
        jp = np.asarray(keypoints_m, dtype=np.float64).reshape(21, 3)
        ref_value = jp[self._task_idx, :] - jp[self._origin_idx, :]
        qpos = self._rt.retarget(ref_value)[6:]
        q = qpos[self._perm].astype(np.float32)
        if self._curl_distal:
            q = self._apply_curl(jp, q)
        return q

    def _apply_curl(self, jp: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Overwrite the DexPilot-dead distal joints with human-curl targets."""
        lim = self.joint_limits
        raw = np.empty(len(_CURL_DRIVEN), dtype=np.float32)
        for k, (j, chain) in enumerate(_CURL_DRIVEN.items()):
            t = (_chain_curl(jp, chain) - _CURL_STRAIGHT) / (_CURL_FULL - _CURL_STRAIGHT)
            # straight human finger -> 0 (open), full curl -> joint upper limit
            raw[k] = np.clip(t, 0.0, 1.0) * lim[j, 1]
        if self._curl_state is None:
            self._curl_state = raw
        else:
            a = self._curl_alpha
            self._curl_state = (1 - a) * self._curl_state + a * raw
        for k, j in enumerate(_CURL_DRIVEN):
            q[j] = self._curl_state[k]
        return q
