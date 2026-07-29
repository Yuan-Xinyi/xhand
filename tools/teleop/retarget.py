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


class HandRetargeter:
    """DexPilot fingertip retargeting: MANO keypoints -> xhand joint targets."""

    def __init__(self, low_pass_alpha: float | None = None):
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

    @property
    def output_joint_names(self):
        """xhand joint names in wire order (== protocol.HAND_JOINT_NAMES)."""
        return list(HAND_JOINT_NAMES)

    def reset(self):
        """Clear the optimizer's warm-start / temporal low-pass state."""
        self._rt.reset()

    def retarget(self, keypoints_m: np.ndarray) -> np.ndarray:
        """(21,3) MANO keypoints in meters -> (12,) xhand joints in wire order."""
        jp = np.asarray(keypoints_m, dtype=np.float64).reshape(21, 3)
        ref_value = jp[self._task_idx, :] - jp[self._origin_idx, :]
        qpos = self._rt.retarget(ref_value)[6:]
        return qpos[self._perm].astype(np.float32)
