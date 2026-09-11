# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop dashboard — the perception node's `--show` window.

One canvas, three panes, refreshed every camera frame:

    +----------------+-----------+----------------------+
    |  camera +      | MANO 3D   |  xhand joint targets |
    |  2D skeleton   | skeleton  |  (the mapping output)|
    |                | front/side|  12 bars w/ limits   |
    +----------------+-----------+----------------------+

Left  = what the camera sees.  Middle = what WiLoR reconstructed (the mapping
INPUT).  Right = what DexPilot mapped it to (the mapping OUTPUT — exactly the
12 values on the UDP wire, i.e. what the sim or the REAL xhand will execute).
Made for eyeballing the mapping before driving real hardware.
"""
from __future__ import annotations

import cv2
import numpy as np

from protocol import HAND_JOINT_NAMES

# MANO connectivity: wrist 0, then 4 joints per finger.
_FINGERS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]
# BGR, one color per finger — same palette in every pane so you can match
# a MANO finger to its xhand joints at a glance.
_FINGER_COLORS = [(80, 80, 255), (0, 165, 255), (0, 220, 220), (80, 220, 80), (255, 160, 60)]
_FINGER_LABELS = ["thumb", "index", "middle", "ring", "pinky"]

# wire joint -> finger index (for bar colors)
_JOINT_FINGER = [next(i for i, f in enumerate(_FINGER_LABELS) if n.startswith(f))
                 for n in HAND_JOINT_NAMES]

_BG = (28, 24, 20)          # panel background
_GRID = (60, 54, 48)
_TEXT = (235, 235, 235)
_DIM = (150, 150, 150)
_OK = (80, 220, 80)
_BAD = (60, 60, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(w: int, h: int) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = _BG
    return img


def _draw_skeleton_proj(img, kp, ax_u: int, ax_v: int, flip_u: bool, box, label: str):
    """Draw one orthographic projection of the (21,3) keypoints into `box`."""
    x0, y0, w, h = box
    cv2.putText(img, label, (x0 + 8, y0 + 18), _FONT, 0.45, _DIM, 1, cv2.LINE_AA)
    if kp is None:
        return
    p = kp - kp[0]                      # wrist-centered, meters
    u = p[:, ax_u] * (-1.0 if flip_u else 1.0)
    v = p[:, ax_v]
    # fixed scale so the hand doesn't "breathe" as tracking wobbles
    scale = min(w, h) * 3.4             # px per meter (hand span ~0.2 m)
    cu = x0 + w * 0.5 + u * scale
    cv = y0 + h * 0.62 + v * scale
    pts = np.stack([cu, cv], axis=1).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], x0 + 2, x0 + w - 3)
    pts[:, 1] = np.clip(pts[:, 1], y0 + 2, y0 + h - 3)
    for finger, col in zip(_FINGERS, _FINGER_COLORS):
        for a, b in zip(finger[:-1], finger[1:]):
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[finger[-1]]), 4, col, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), 5, _TEXT, -1, cv2.LINE_AA)


class TeleopDashboard:
    """Composes the 3-pane teleop window. cv2-only (no extra deps)."""

    def __init__(self, joint_limits: np.ndarray, cam_size=(640, 480),
                 pose_w: int = 300, bars_w: int = 400):
        self._lim = np.asarray(joint_limits, dtype=np.float32)   # (12,2) wire order
        self._cam_w, self._cam_h = cam_size
        self._pose_w = pose_w
        self._bars_w = bars_w
        self._h = self._cam_h
        self._w = self._cam_w + pose_w + bars_w

    # ---------------------------------------------------------------- panes
    def _draw_camera(self, canvas, bgr, hand, fps, valid):
        cam = cv2.resize(bgr, (self._cam_w, self._cam_h)) \
            if bgr.shape[:2] != (self._cam_h, self._cam_w) else bgr.copy()
        if hand is not None:
            kp2d = hand.keypoints_2d
            sx = self._cam_w / bgr.shape[1]
            sy = self._cam_h / bgr.shape[0]
            pts = (kp2d * np.array([sx, sy])).astype(int)
            for finger, col in zip(_FINGERS, _FINGER_COLORS):
                for a, b in zip(finger[:-1], finger[1:]):
                    cv2.line(cam, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(cam, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
        status = "TRACKING" if valid else "NO HAND"
        cv2.putText(cam, f"{status}  {fps:4.1f} FPS", (10, 30), _FONT, 0.8,
                    _OK if valid else _BAD, 2, cv2.LINE_AA)
        canvas[:, :self._cam_w] = cam

    def _draw_pose(self, canvas, kp3d):
        x0 = self._cam_w
        pane = _panel(self._pose_w, self._h)
        canvas[:, x0:x0 + self._pose_w] = pane
        cv2.line(canvas, (x0, 0), (x0, self._h), _GRID, 1)
        cv2.putText(canvas, "MANO 3D (input)", (x0 + 8, 24), _FONT, 0.55, _TEXT, 1, cv2.LINE_AA)
        half = (self._h - 30) // 2
        # camera frame: x right, y down, z forward
        _draw_skeleton_proj(canvas, kp3d, 0, 1, False, (x0, 30, self._pose_w, half), "front (x-y)")
        cv2.line(canvas, (x0 + 10, 30 + half), (x0 + self._pose_w - 10, 30 + half), _GRID, 1)
        _draw_skeleton_proj(canvas, kp3d, 2, 1, True, (x0, 30 + half, self._pose_w, half), "side (z-y)")
        if kp3d is None:
            cv2.putText(canvas, "NO HAND", (x0 + self._pose_w // 2 - 45, self._h // 2),
                        _FONT, 0.6, _BAD, 2, cv2.LINE_AA)

    def _draw_bars(self, canvas, q, valid):
        x0 = self._cam_w + self._pose_w
        w = self._bars_w
        canvas[:, x0:x0 + w] = _panel(w, self._h)
        cv2.line(canvas, (x0, 0), (x0, self._h), _GRID, 1)
        cv2.putText(canvas, "xhand joints rad (output = wire = robot)",
                    (x0 + 8, 24), _FONT, 0.5, _TEXT, 1, cv2.LINE_AA)

        n = len(HAND_JOINT_NAMES)
        row_h = (self._h - 40) // n
        name_w = 118                       # left column for names
        val_w = 62                         # right column for numbers
        bx0 = x0 + name_w
        bw = w - name_w - val_w - 16
        for j, name in enumerate(HAND_JOINT_NAMES):
            y = 40 + j * row_h
            col = _FINGER_COLORS[_JOINT_FINGER[j]]
            lo, hi = float(self._lim[j, 0]), float(self._lim[j, 1])
            frac = 0.0 if hi <= lo else (float(q[j]) - lo) / (hi - lo)
            frac = min(max(frac, 0.0), 1.0)
            cv2.putText(canvas, name.replace("_joint", " j"), (x0 + 8, y + row_h - 10),
                        _FONT, 0.45, col, 1, cv2.LINE_AA)
            # track + fill + zero tick
            ty = y + row_h // 2 - 4
            cv2.rectangle(canvas, (bx0, ty), (bx0 + bw, ty + 9), _GRID, -1)
            fill = col if valid else tuple(int(c * 0.45) for c in col)
            cv2.rectangle(canvas, (bx0, ty), (bx0 + int(bw * frac), ty + 9), fill, -1)
            if lo < 0.0 < hi:
                zx = bx0 + int(bw * (-lo) / (hi - lo))
                cv2.line(canvas, (zx, ty - 2), (zx, ty + 11), _TEXT, 1)
            cv2.putText(canvas, f"{float(q[j]):+.2f}", (bx0 + bw + 8, y + row_h - 10),
                        _FONT, 0.45, _TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"range e.g. [{self._lim[3,0]:+.2f}, {self._lim[3,1]:+.2f}]  "
                            f"| = 0 rad", (x0 + 8, self._h - 8), _FONT, 0.4, _DIM, 1, cv2.LINE_AA)

    # ---------------------------------------------------------------- public
    def render(self, bgr, hand, kp3d, q, fps: float, valid: bool) -> np.ndarray:
        """Compose one dashboard frame. Returns the BGR canvas for imshow."""
        canvas = _panel(self._w, self._h)
        self._draw_camera(canvas, bgr, hand, fps, valid)
        self._draw_pose(canvas, kp3d if valid else None)
        self._draw_bars(canvas, q, valid)
        return canvas
