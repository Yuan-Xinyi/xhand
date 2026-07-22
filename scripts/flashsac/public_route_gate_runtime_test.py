#!/usr/bin/env python3
"""Simulation-free tests for the development-only frozen gate runtime."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from public_route_gate_dataset import load_analysis_plan, sha256_file
from public_route_gate_runtime import load_frozen_public_route_gate
from train_public_route_gate import (
    HEADS,
    HIDDEN_DIM,
    MODEL_FORMAT_VERSION,
    MODEL_KIND,
    REPORT_KIND,
    semantic_model_sha256,
    validate_gate_model,
)


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _model(plan: dict[str, Any]) -> dict[str, Any]:
    member_count = 5
    tensors = {
        "feature_mean": torch.zeros(165, dtype=torch.float64),
        "feature_scale": torch.ones(165, dtype=torch.float64),
        "layer0_weight": torch.zeros(
            (member_count, HIDDEN_DIM, 165), dtype=torch.float32
        ),
        "layer0_bias": torch.zeros(
            (member_count, HIDDEN_DIM), dtype=torch.float32
        ),
        "layer1_weight": torch.zeros(
            (member_count, HIDDEN_DIM, HIDDEN_DIM), dtype=torch.float32
        ),
        "layer1_bias": torch.zeros(
            (member_count, HIDDEN_DIM), dtype=torch.float32
        ),
        "output_weight": torch.zeros(
            (member_count, len(HEADS), HIDDEN_DIM), dtype=torch.float32
        ),
        "output_bias": torch.tensor(
            [[2.0, -2.0, -2.0, -2.0, -2.0, -2.0]] * member_count,
            dtype=torch.float32,
        ),
    }
    metadata = {
        "analysis_plan_sha256": plan["_sha256"],
        "analysis_seal_tag": plan["preregistration"]["seal_tag"],
        "collection_manifest_sha256": plan["preregistration"][
            "trial_manifest_sha256"
        ],
        "collection_seal_tag": plan["preregistration"]["trial_seal_tag"],
        "dataset_semantic_sha256": "d" * 64,
        "excluded_cohorts": ["pilot", "development", "blind"],
        "feature_contract": "pick_tool_public_route_gate_feature165_v1",
        "head_order": list(HEADS),
        "input_ledger_sha256": "e" * 64,
        "input_ledger_tag": plan["preregistration"]["train_ledger_tag"],
        "member_seeds": [0, 1, 2, 3, 4],
        "normalization_contract": plan["normalization"],
        "optimizer_contract": plan["optimization"],
        "decision_contract": plan["inference"],
        "run_receipts": [{"cohort": "train", "artifact_sha256": "a" * 64}],
        "trained_cohort": "train",
        "training_rows": 32,
        "trainer_source_sha256": "f" * 64,
    }
    return validate_gate_model(
        {
            "kind": MODEL_KIND,
            "format_version": MODEL_FORMAT_VERSION,
            "metadata": metadata,
            "tensors": tensors,
        }
    )


def _report(model: dict[str, Any], model_sha: str, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "analysis_git": {
            "branch": plan["preregistration"]["branch"],
            "current_commit": "c" * 40,
            "analysis_seal_tag": plan["preregistration"]["seal_tag"],
            "analysis_seal_commit": "a" * 40,
            "implementation_commit": plan["preregistration"]["implementation_commit"],
            "trial_seal_tag": plan["preregistration"]["trial_seal_tag"],
            "trial_seal_commit": "b" * 40,
            "source_sha256": {
                "scripts/flashsac/train_public_route_gate.py": model["metadata"][
                    "trainer_source_sha256"
                ]
            },
        },
        "analysis_plan_sha256": plan["_sha256"],
        "collection_acceptance": {"cohort": "train", "accepted": True},
        "dataset_semantic_sha256": model["metadata"]["dataset_semantic_sha256"],
        "factual_arm_rows": {"continue": 16, "route": 16},
        "factual_event_counts": {"success": 2, "dropped": 0, "unsafe_force": 0},
        "format_version": 1,
        "input_ledger_sha256": model["metadata"]["input_ledger_sha256"],
        "input_ledger_tag": model["metadata"]["input_ledger_tag"],
        "kind": REPORT_KIND,
        "member_diagnostics": [
            {
                "member_seed": seed,
                "optimizer_steps": 10,
                "final_factual_bce": 0.1,
                "maximum_preclip_gradient_norm": 1.0,
            }
            for seed in range(5)
        ],
        "model_semantic_sha256": semantic_model_sha256(model),
        "model_sha256": model_sha,
        "status": "complete",
        "train_continue_decisions": 32,
        "train_continue_rate": 1.0,
        "training_rows": 32,
    }


def _publish(
    directory: Path,
    *,
    mutate_model: Callable[[dict[str, Any]], None] | None = None,
    mutate_report: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, Path, Path]:
    source_plan = Path(__file__).with_name("public_route_gate_analysis_plan.json")
    plan_path = directory / "analysis_plan.json"
    plan_path.write_bytes(source_plan.read_bytes())
    plan = load_analysis_plan(plan_path, require_preregistered=False)
    model = copy.deepcopy(_model(plan))
    if mutate_model is not None:
        mutate_model(model)
    model_path = directory / "gate.pt"
    report_path = directory / "gate.json"
    torch.save(model, model_path)
    report = _report(model, sha256_file(model_path), plan)
    if mutate_report is not None:
        mutate_report(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return model_path, report_path, plan_path


def _load(model_path: Path, report_path: Path, plan_path: Path):
    return load_frozen_public_route_gate(
        model_path,
        report_path,
        analysis_plan_path=plan_path,
        cohort="development",
        require_preregistered_plan=False,
        enforce_canonical_paths=False,
        enforce_frozen_model_tag=False,
    )


def test_weights_only_model_receipt_and_development_decision() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        model_path, report_path, plan_path = _publish(Path(directory_name))
        runtime = _load(model_path, report_path, plan_path)
        decision = runtime.decide_development(torch.zeros((3, 165)))
        assert torch.equal(decision.continue_search, torch.ones(3, dtype=torch.bool))
        assert torch.equal(decision.route_v6, torch.zeros(3, dtype=torch.bool))
        assert decision.probabilities.shape == (5, 3, 6)
        metadata = runtime.audit_metadata()
        assert metadata["allowed_cohort"] == "development"
        assert metadata["blind_collection_opened"] is False
        assert metadata["default_on_error"] == "fixed_route_v6"
        runtime.verify_unchanged()
        _expect_error(ValueError, runtime.validate_cohort, "blind")

        with model_path.open("ab") as stream:
            stream.write(b"changed")
        _expect_error(RuntimeError, runtime.verify_unchanged)


def test_loader_rejects_report_model_and_cohort_drift() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        model_path, report_path, plan_path = _publish(
            directory,
            mutate_report=lambda report: report.__setitem__("model_sha256", "0" * 64),
        )
        _expect_error(ValueError, _load, model_path, report_path, plan_path)

    with tempfile.TemporaryDirectory() as directory_name:
        model_path, report_path, plan_path = _publish(
            Path(directory_name),
            mutate_model=lambda model: model["metadata"].__setitem__(
                "input_ledger_tag", "wrong-tag"
            ),
        )
        _expect_error(ValueError, _load, model_path, report_path, plan_path)

    with tempfile.TemporaryDirectory() as directory_name:
        model_path, report_path, plan_path = _publish(Path(directory_name))
        _expect_error(
            ValueError,
            load_frozen_public_route_gate,
            model_path,
            report_path,
            analysis_plan_path=plan_path,
            cohort="blind",
            require_preregistered_plan=False,
            enforce_canonical_paths=False,
            enforce_frozen_model_tag=False,
        )


def test_runtime_detects_in_memory_model_report_and_receipt_mutation() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        model_path, report_path, plan_path = _publish(Path(directory_name))
        runtime = _load(model_path, report_path, plan_path)
        runtime.model["tensors"]["output_bias"].zero_()
        _expect_error(RuntimeError, runtime.verify_unchanged)

    with tempfile.TemporaryDirectory() as directory_name:
        model_path, report_path, plan_path = _publish(Path(directory_name))
        runtime = _load(model_path, report_path, plan_path)
        runtime.training_report["status"] = "tampered"
        _expect_error(RuntimeError, runtime.verify_unchanged)

    with tempfile.TemporaryDirectory() as directory_name:
        model_path, report_path, plan_path = _publish(Path(directory_name))
        runtime = _load(model_path, report_path, plan_path)
        runtime.frozen_receipt["frozen_model_tag"] = "tampered"
        _expect_error(RuntimeError, runtime.verify_unchanged)
