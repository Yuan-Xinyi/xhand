#!/usr/bin/env python3
"""Pure-Torch truth contract for the coupled POWER-grasp lift audit.

This module intentionally knows nothing about Isaac Sim.  The physical audit feeds it
the task's mesh-minimum clearance, contact/latch state, relative-motion hold quality,
and object velocities.  In particular, neither object-root height nor a rotated AABB
is accepted as a lift signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class LiftAuditThresholds:
    """Conservative thresholds shared by the 5 cm and 20 cm verdicts."""

    micro_clearance_m: float = 0.05
    success_clearance_m: float = 0.20
    min_grasp_quality: float = 0.35
    min_hold_quality: float = 0.50
    max_force_n: float = 30.0
    max_object_linear_speed_m_s: float = 0.20
    max_object_angular_speed_rad_s: float = 3.00

    def __post_init__(self) -> None:
        numeric = {
            "micro_clearance_m": self.micro_clearance_m,
            "success_clearance_m": self.success_clearance_m,
            "min_grasp_quality": self.min_grasp_quality,
            "min_hold_quality": self.min_hold_quality,
            "max_force_n": self.max_force_n,
            "max_object_linear_speed_m_s": self.max_object_linear_speed_m_s,
            "max_object_angular_speed_rad_s": self.max_object_angular_speed_rad_s,
        }
        if any(not math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("lift audit thresholds must all be finite")
        if not 0.0 < self.micro_clearance_m < self.success_clearance_m:
            raise ValueError("clearance thresholds require 0 < micro < success")
        for name in ("min_grasp_quality", "min_hold_quality"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in (
            "max_force_n",
            "max_object_linear_speed_m_s",
            "max_object_angular_speed_rad_s",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


def _validate_vector(name: str, value: torch.Tensor, *, size: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {tuple(value.shape)}")


def classify_lift_truth(
    *,
    true_clearance_m: torch.Tensor,
    is_grasped: torch.Tensor,
    power_latched: torch.Tensor,
    grasp_quality: torch.Tensor,
    power_grasp_quality: torch.Tensor,
    hold_quality: torch.Tensor,
    max_finger_force_n: torch.Tensor,
    object_linear_speed_m_s: torch.Tensor,
    object_angular_speed_rad_s: torch.Tensor,
    thresholds: LiftAuditThresholds,
    require_power_contract: bool = True,
) -> dict[str, torch.Tensor]:
    """Classify instantaneous physical truth without any height proxy.

    ``micro_stable`` and ``success_stable`` are instantaneous masks.  The caller
    must require consecutive frames and must additionally reject any trajectory
    that has ever exceeded the force limit, lost its grasp while airborne, or
    saturated a commanded arm target.
    """

    if not isinstance(thresholds, LiftAuditThresholds):
        raise TypeError("thresholds must be LiftAuditThresholds")
    if not isinstance(require_power_contract, bool):
        raise TypeError("require_power_contract must be boolean")
    size = int(true_clearance_m.shape[0]) if true_clearance_m.ndim == 1 else -1
    for name, value in (
        ("true_clearance_m", true_clearance_m),
        ("is_grasped", is_grasped),
        ("power_latched", power_latched),
        ("grasp_quality", grasp_quality),
        ("power_grasp_quality", power_grasp_quality),
        ("hold_quality", hold_quality),
        ("max_finger_force_n", max_finger_force_n),
        ("object_linear_speed_m_s", object_linear_speed_m_s),
        ("object_angular_speed_rad_s", object_angular_speed_rad_s),
    ):
        _validate_vector(name, value, size=size)
        if value.device != true_clearance_m.device:
            raise ValueError(f"{name} is on {value.device}, expected {true_clearance_m.device}")
    if is_grasped.dtype is not torch.bool or power_latched.dtype is not torch.bool:
        raise TypeError("is_grasped and power_latched must be boolean tensors")
    for name, value in (
        ("true_clearance_m", true_clearance_m),
        ("grasp_quality", grasp_quality),
        ("power_grasp_quality", power_grasp_quality),
        ("hold_quality", hold_quality),
        ("max_finger_force_n", max_finger_force_n),
        ("object_linear_speed_m_s", object_linear_speed_m_s),
        ("object_angular_speed_rad_s", object_angular_speed_rad_s),
    ):
        if not value.dtype.is_floating_point:
            raise TypeError(f"{name} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains NaN or infinity")

    safe_force = max_finger_force_n <= thresholds.max_force_n
    legacy_load_bearing = (
        is_grasped
        & (grasp_quality >= thresholds.min_grasp_quality)
        & (hold_quality >= thresholds.min_hold_quality)
        & safe_force
    )
    power_load_bearing = power_latched & (
        power_grasp_quality >= thresholds.min_grasp_quality
    )
    load_bearing_grasp = legacy_load_bearing & (
        power_load_bearing if require_power_contract else torch.ones_like(power_latched)
    )
    slow = (
        (object_linear_speed_m_s < thresholds.max_object_linear_speed_m_s)
        & (object_angular_speed_rad_s < thresholds.max_object_angular_speed_rad_s)
    )
    airborne = true_clearance_m >= thresholds.micro_clearance_m
    return {
        "safe_force": safe_force,
        "load_bearing_grasp": load_bearing_grasp,
        "slow": slow,
        "airborne_unlatched": airborne
        & (
            (~is_grasped | ~power_latched)
            if require_power_contract
            else ~is_grasped
        ),
        "micro_stable": airborne & load_bearing_grasp & slow,
        "success_stable": (
            (true_clearance_m >= thresholds.success_clearance_m)
            & load_bearing_grasp
            & slow
        ),
    }


__all__ = ["LiftAuditThresholds", "classify_lift_truth"]
