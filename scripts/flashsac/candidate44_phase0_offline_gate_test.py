#!/usr/bin/env python3
"""Simulation-free tests for Candidate44's offline release calculations."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidate44_phase0_offline_gate import (  # noqa: E402
    CHECKPOINT_TENSOR_FILENAMES,
    _semantic_equal,
    action_error_metrics,
    apply_registered_gates,
    audit_checkpoint_tensor_files,
    build_panel_report,
)


def test_action_panel_metrics_cover_every_action_group() -> None:
    target = torch.zeros(8, 21)
    baseline = torch.full((8, 21), 0.2)
    candidate = torch.full((8, 21), 0.1)
    report = build_panel_report(baseline, candidate, target)
    assert report["rows"] == 8
    assert abs(report["baseline"]["action_rmse"] - 0.2) < 1.0e-6
    assert abs(report["candidate"]["action_rmse"] - 0.1) < 1.0e-6
    for name in ("action", "arm", "token", "residual"):
        assert abs(
            report["candidate_to_baseline_rmse_ratio"][f"{name}_rmse"] - 0.5
        ) < 1.0e-6
    drift = action_error_metrics(candidate, baseline)
    assert abs(drift["action_rmse"] - 0.1) < 1.0e-6


def test_registered_gate_is_all_must_pass_and_has_no_nan_escape() -> None:
    passing_phase0 = {
        "candidate_to_baseline_rmse_ratio": {
            "action_rmse": 0.85,
            "arm_rmse": 0.85,
            "token_rmse": 0.95,
            "residual_rmse": 0.85,
        }
    }
    retention = {
        "candidate_minus_baseline_target_rmse": {"action_rmse": 0.03}
    }
    passing = apply_registered_gates(
        passing_phase0,
        retention,
        retention,
        {"all_pass": True},
    )
    assert passing["all_pass"]
    assert passing["decision"] == "release_to_paired_physical_gate"

    for phase0_key in ("action_rmse", "arm_rmse", "token_rmse", "residual_rmse"):
        rejected_phase0 = {
            "candidate_to_baseline_rmse_ratio": dict(
                passing_phase0["candidate_to_baseline_rmse_ratio"]
            )
        }
        rejected_phase0["candidate_to_baseline_rmse_ratio"][phase0_key] = float("nan")
        rejected = apply_registered_gates(
            rejected_phase0,
            retention,
            retention,
            {"all_pass": True},
        )
        assert not rejected["all_pass"]
    assert not apply_registered_gates(
        passing_phase0,
        retention,
        retention,
        {"all_pass": False},
    )["all_pass"]


def test_semantic_equality_is_tensor_dtype_shape_and_value_strict() -> None:
    payload = {
        "state": {0: {"step": torch.tensor(4), "value": torch.tensor([1.0])}},
        "group": [1, 2.0, None],
    }
    cloned = {
        "state": {0: {"step": torch.tensor(4), "value": torch.tensor([1.0])}},
        "group": [1, 2.0, None],
    }
    assert _semantic_equal(payload, cloned)
    cloned["state"][0]["value"] = torch.tensor([1.0], dtype=torch.float64)
    assert not _semantic_equal(payload, cloned)


def test_checkpoint_audit_covers_every_pt_tensor_and_nonfinite_value() -> None:
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory)
        for index, filename in enumerate(sorted(CHECKPOINT_TENSOR_FILENAMES)):
            torch.save(
                {
                    "nested": [
                        torch.tensor([float(index)], dtype=torch.float32),
                        torch.tensor([index], dtype=torch.int64),
                    ]
                },
                checkpoint / filename,
            )
        passing = audit_checkpoint_tensor_files(checkpoint)
        assert passing["all_finite"]
        assert set(passing["files"]) == CHECKPOINT_TENSOR_FILENAMES
        assert all(
            entry["tensor_count"] == 2 for entry in passing["files"].values()
        )

        torch.save(
            {"nested": torch.tensor([float("nan")])}, checkpoint / "critic.pt"
        )
        rejected = audit_checkpoint_tensor_files(checkpoint)
        assert not rejected["all_finite"]
        assert not rejected["files"]["critic.pt"]["all_finite"]

        torch.save({"extra": torch.tensor(1.0)}, checkpoint / "unregistered.pt")
        try:
            audit_checkpoint_tensor_files(checkpoint)
        except ValueError:
            pass
        else:
            raise AssertionError("unexpected checkpoint tensor file was accepted")


def main() -> None:
    test_action_panel_metrics_cover_every_action_group()
    print("[PASS] action-panel group metrics")
    test_registered_gate_is_all_must_pass_and_has_no_nan_escape()
    print("[PASS] registered all-must-pass thresholds")
    test_semantic_equality_is_tensor_dtype_shape_and_value_strict()
    print("[PASS] optimizer semantic equality")
    test_checkpoint_audit_covers_every_pt_tensor_and_nonfinite_value()
    print("[PASS] complete checkpoint tensor-file finiteness audit")


if __name__ == "__main__":
    main()
