# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-Euro filter for smoothing noisy hand keypoints during teleoperation.

Monocular hand estimators (WiLoR) jitter frame-to-frame. A plain low-pass trades
jitter for lag; the One-Euro filter (Casiez et al., CHI 2012) is adaptive — it
smooths hard when the hand is still (kills jitter) and loosens when the hand moves
fast (kills lag). Applied to the (21, 3) MANO keypoints before retargeting.

Tuning:
    min_cutoff  lower  -> smoother/steadier when still (more lag)
    beta        higher -> more responsive during fast motion (less lag)
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroArray:
    """One-Euro filter over an arbitrary-shaped float array (elementwise)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.7, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None or dt <= 0.0:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            return x.astype(np.float32)

        # derivative, low-pass filtered
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # adaptive cutoff -> more smoothing when slow, less when fast
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha_vec(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat.astype(np.float32)


def _alpha_vec(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)
