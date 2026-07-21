#!/usr/bin/env python3
"""Sim-free tests for the strict coupled POWER lift truth contract."""

from __future__ import annotations

import torch

from coupled_power_lift_contract import LiftAuditThresholds, classify_lift_truth


def _truth(**overrides: object) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {
        "true_clearance_m": torch.tensor([0.20, 0.049, 0.22]),
        "is_grasped": torch.tensor([True, True, False]),
        "power_latched": torch.tensor([True, True, True]),
        "grasp_quality": torch.tensor([0.35, 0.8, 0.9]),
        "power_grasp_quality": torch.tensor([0.35, 0.8, 0.9]),
        "hold_quality": torch.tensor([0.50, 0.9, 0.9]),
        "max_finger_force_n": torch.tensor([30.0, 10.0, 10.0]),
        "object_linear_speed_m_s": torch.tensor([0.10, 0.0, 0.0]),
        "object_angular_speed_rad_s": torch.tensor([1.0, 0.0, 0.0]),
    }
    values.update(overrides)  # type: ignore[arg-type]
    return classify_lift_truth(thresholds=LiftAuditThresholds(), **values)


def main() -> None:
    truth = _truth()
    assert truth["success_stable"].tolist() == [True, False, False]
    assert truth["micro_stable"].tolist() == [True, False, False]
    assert truth["airborne_unlatched"].tolist() == [False, False, True]

    # A root-height or rotated-AABB peak cannot enter this API.  Sub-threshold
    # mesh-minimum clearance stays a failure even with every grasp signal ideal.
    proxy_hack = _truth(true_clearance_m=torch.tensor([0.0, 0.0, 0.0]))
    assert not bool(proxy_hack["micro_stable"].any())
    assert not bool(proxy_hack["success_stable"].any())

    overforce = _truth(max_finger_force_n=torch.tensor([30.001, 10.0, 10.0]))
    assert not bool(overforce["success_stable"][0])

    stale_power_latch = _truth(
        power_grasp_quality=torch.tensor([0.349, 0.8, 0.9])
    )
    assert not bool(stale_power_latch["success_stable"][0])
    native_contract = _truth(
        power_latched=torch.tensor([False, True, True]),
        power_grasp_quality=torch.tensor([0.0, 0.8, 0.9]),
        require_power_contract=False,
    )
    assert bool(native_contract["success_stable"][0])

    fling = _truth(
        is_grasped=torch.tensor([False, True, False]),
        power_latched=torch.tensor([False, True, False]),
    )
    assert not bool(fling["success_stable"][0])
    assert bool(fling["airborne_unlatched"][0])

    moving = _truth(object_linear_speed_m_s=torch.tensor([0.20, 0.0, 0.0]))
    assert not bool(moving["success_stable"][0])

    try:
        LiftAuditThresholds(max_force_n=float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite thresholds must be rejected")

    try:
        _truth(hold_quality=torch.tensor([float("nan"), 0.9, 0.9]))
    except FloatingPointError:
        pass
    else:
        raise AssertionError("non-finite truth must be rejected")

    print("coupled_power_lift_contract_test: PASS")


if __name__ == "__main__":
    main()
