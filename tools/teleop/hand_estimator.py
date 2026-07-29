# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera image -> MANO hand keypoints. Pluggable front-end for teleoperation.

`HandEstimator` is the interface the perception node depends on; `WiLoREstimator`
is the concrete WiLoR-mini backend. To swap in a newer model later (Hamba,
Fast-HaMeR, ...) implement `estimate()` returning the same `HandResult` and the
rest of the pipeline is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HandResult:
    """One detected hand, in the estimator's canonical MANO layout."""

    keypoints_3d: np.ndarray  # (21, 3) MANO joints, meters, wrist-relative
    keypoints_2d: np.ndarray  # (21, 2) image-pixel joints (for overlay)
    is_right: bool            # handedness
    bbox: np.ndarray          # (4,) x1,y1,x2,y2
    score: float              # detector confidence


class HandEstimator:
    """Interface: RGB uint8 image -> list[HandResult]."""

    def estimate(self, rgb: np.ndarray) -> list[HandResult]:  # pragma: no cover
        raise NotImplementedError


def _bbox_from_kp2d(kp2d: np.ndarray, w: int, h: int, pad: float = 0.4) -> list[float]:
    """Tight (x1,y1,x2,y2) box around 2D keypoints, padded and image-clamped."""
    x1, y1 = kp2d.min(0)
    x2, y2 = kp2d.max(0)
    bw, bh = (x2 - x1), (y2 - y1)
    x1 -= pad * bw; x2 += pad * bw
    y1 -= pad * bh; y2 += pad * bh
    return [float(max(0, x1)), float(max(0, y1)), float(min(w - 1, x2)), float(min(h - 1, y2))]


class WiLoREstimator(HandEstimator):
    """WiLoR-mini backend with detection-skip + ROI tracking.

    YOLO detection is the slow, drop-prone stage. So we run it only every
    ``redetect_interval`` frames (or after a tracking loss); in between we crop
    the ViT to the last frame's keypoint box via ``predict_with_bboxes`` — faster,
    and it bridges the detector's misses so tracking is far less jumpy.
    """

    def __init__(self, device: str = "cuda", fp16: bool = True, redetect_interval: int = 12):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        self._dev = torch.device(device)
        dtype = torch.float16 if fp16 else torch.float32
        self._pipe = WiLorHandPose3dEstimationPipeline(device=self._dev, dtype=dtype, verbose=False)
        self._redetect_interval = int(redetect_interval)
        self._bboxes = None          # np (N,4) cached ROI, or None -> force YOLO
        self._is_rights = None       # np (N,) handedness for the cached ROIs
        self._since_detect = 0

    def _to_results(self, out, w: int, h: int) -> list[HandResult]:
        results, new_boxes, new_rights = [], [], []
        for d in out:
            preds = d.get("wilor_preds")
            if preds is None:
                continue
            kp3d = np.asarray(preds["pred_keypoints_3d"]).reshape(-1, 21, 3)[0]
            kp2d = np.asarray(preds["pred_keypoints_2d"]).reshape(-1, 21, 2)[0]
            results.append(
                HandResult(
                    keypoints_3d=kp3d.astype(np.float32),
                    keypoints_2d=kp2d.astype(np.float32),
                    is_right=bool(round(float(d.get("is_right", 1)))),
                    bbox=np.asarray(d.get("hand_bbox", [0, 0, 0, 0]), dtype=np.float32),
                    score=float(d.get("hand_bbox_score", d.get("score", 1.0))),
                )
            )
            new_boxes.append(_bbox_from_kp2d(kp2d, w, h))       # ROI follows the hand
            new_rights.append(float(d.get("is_right", 1)))
        if new_boxes:
            self._bboxes = np.asarray(new_boxes, dtype=np.float32)
            self._is_rights = np.asarray(new_rights, dtype=np.float32)
        else:
            self._bboxes = None       # lost -> re-run YOLO next frame
            self._is_rights = None
        return results

    def estimate(self, rgb: np.ndarray) -> list[HandResult]:
        """rgb: (H,W,3) uint8 RGB. Returns all tracked hands."""
        h, w = rgb.shape[:2]
        redetect = self._bboxes is None or self._since_detect >= self._redetect_interval
        if redetect:
            out = self._pipe.predict(rgb, hand_conf=0.25)   # lower conf -> re-acquire easier
            self._since_detect = 0
        else:
            out = self._pipe.predict_with_bboxes(rgb, self._bboxes, self._is_rights)
            self._since_detect += 1
        return self._to_results(out, w, h)

    def reset(self):
        self._bboxes = None
        self._is_rights = None
        self._since_detect = 0


def mirror_to_right(kp3d: np.ndarray) -> np.ndarray:
    """Mirror a LEFT-hand MANO keypoint set into the RIGHT-hand frame.

    xhand is a right hand and the DexPilot config maps a right MANO hand. If the
    user prefers to teleop with their left hand, negate X (reflect across the
    sagittal plane) so the retargeter sees a valid right-hand pose.
    """
    out = np.asarray(kp3d, dtype=np.float32).copy()
    out[:, 0] *= -1.0
    return out
