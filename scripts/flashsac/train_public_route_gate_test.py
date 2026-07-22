#!/usr/bin/env python3
"""Simulation-free tests for the frozen public-route gate trainer/runtime."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch
import torch.nn.functional as F

import train_public_route_gate as trainer_module
from public_route_gate_dataset import load_analysis_plan
from public_route_gate_dataset_test import _complete_synthetic, _load
from train_public_route_gate import (
    FEATURE_DIM,
    HEADS,
    MODEL_FORMAT_VERSION,
    MODEL_KIND,
    decision_from_probabilities,
    ensemble_probabilities,
    factual_logits_and_labels,
    fit_feature_normalizer,
    publish_model_and_report_no_clobber,
    semantic_model_sha256,
    train_gate_ensemble,
    validate_gate_model,
)


def _raises(
    error: type[BaseException], function: Callable[..., Any], *args: Any, **kwargs: Any
) -> None:
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _fake_model() -> dict[str, Any]:
    tensors = {
        "feature_mean": torch.zeros(FEATURE_DIM, dtype=torch.float64),
        "feature_scale": torch.ones(FEATURE_DIM, dtype=torch.float64),
        "layer0_weight": torch.zeros((5, 128, FEATURE_DIM), dtype=torch.float32),
        "layer0_bias": torch.zeros((5, 128), dtype=torch.float32),
        "layer1_weight": torch.zeros((5, 128, 128), dtype=torch.float32),
        "layer1_bias": torch.zeros((5, 128), dtype=torch.float32),
        "output_weight": torch.zeros((5, 6, 128), dtype=torch.float32),
        "output_bias": torch.zeros((5, 6), dtype=torch.float32),
    }
    return {
        "kind": MODEL_KIND,
        "format_version": MODEL_FORMAT_VERSION,
        "metadata": {
            "analysis_plan_sha256": "a" * 64,
            "analysis_seal_tag": "analysis-tag",
            "collection_manifest_sha256": "b" * 64,
            "collection_seal_tag": "collection-tag",
            "dataset_semantic_sha256": "c" * 64,
            "excluded_cohorts": ["pilot", "development", "blind"],
            "feature_contract": "pick_tool_public_route_gate_feature165_v1",
            "head_order": list(HEADS),
            "input_ledger_sha256": "e" * 64,
            "input_ledger_tag": "train-ledger-tag",
            "member_seeds": [0, 1, 2, 3, 4],
            "normalization_contract": {"std_floor": 1.0e-6},
            "optimizer_contract": {"optimizer": "AdamW"},
            "decision_contract": {"success_margin": 0.05},
            "run_receipts": [],
            "trained_cohort": "train",
            "training_rows": 8,
            "trainer_source_sha256": "d" * 64,
        },
        "tensors": tensors,
    }


def test_population_normalizer_and_floor_are_exact() -> None:
    feature = torch.zeros((2, FEATURE_DIM), dtype=torch.float32)
    feature[:, 0] = torch.tensor((0.0, 2.0))
    mean, scale = fit_feature_normalizer(feature)
    assert mean.dtype == torch.float64 and scale.dtype == torch.float64
    assert mean[0].item() == 1.0 and scale[0].item() == 1.0
    assert bool((scale[1:] == 1.0e-6).all())


def test_factual_gather_gives_no_gradient_to_unassigned_output_heads() -> None:
    logits = torch.zeros((4, 6), dtype=torch.float32, requires_grad=True)
    treatment = torch.tensor((False, True, False, True), dtype=torch.bool)
    success = torch.tensor((True, False, False, True), dtype=torch.bool)
    dropped = torch.tensor((False, True, False, False), dtype=torch.bool)
    unsafe = torch.tensor((False, False, True, False), dtype=torch.bool)
    factual, labels = factual_logits_and_labels(
        logits,
        treatment_route=treatment,
        success=success,
        dropped=dropped,
        unsafe_force=unsafe,
    )
    F.binary_cross_entropy_with_logits(factual, labels).backward()
    assert factual.shape == labels.shape == (4, 3)
    assert bool((logits.grad[~treatment, 3:] == 0.0).all())
    assert bool((logits.grad[treatment, :3] == 0.0).all())
    assert bool((logits.grad[~treatment, :3] != 0.0).all())
    assert bool((logits.grad[treatment, 3:] != 0.0).all())


def test_population_bounds_equality_and_nonfinite_default_route() -> None:
    probabilities = torch.zeros((5, 3, 6), dtype=torch.float64)
    probabilities[:, :, 0] = 0.75
    probabilities[:, :, 3] = torch.tensor((0.70, 0.71, 0.70))
    probabilities[:, :, 1] = 0.10
    probabilities[:, :, 4] = 0.10
    probabilities[:, :, 2] = 0.20
    probabilities[:, :, 5] = 0.20
    probabilities[0, 2, 0] = float("nan")
    decision = decision_from_probabilities(probabilities)
    assert decision.tolist() == [True, False, False]


def test_model_schema_inference_semantic_hash_and_fail_closed() -> None:
    model = _fake_model()
    validated = validate_gate_model(model)
    probabilities = ensemble_probabilities(
        validated, torch.zeros((3, FEATURE_DIM), dtype=torch.float32)
    )
    assert probabilities.shape == (5, 3, 6)
    assert bool((probabilities == 0.5).all())
    assert len(semantic_model_sha256(validated)) == 64

    short_scale = copy.deepcopy(model)
    short_scale["tensors"]["feature_scale"][0] = 0.0
    _raises(ValueError, validate_gate_model, short_scale)
    wrong_heads = copy.deepcopy(model)
    wrong_heads["metadata"]["head_order"] = list(reversed(HEADS))
    _raises(ValueError, validate_gate_model, wrong_heads)
    dev_fit = copy.deepcopy(model)
    dev_fit["metadata"]["trained_cohort"] = "development"
    _raises(ValueError, validate_gate_model, dev_fit)


def test_weights_only_atomic_pair_and_no_clobber() -> None:
    model = validate_gate_model(_fake_model())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model_path, report_path = root / "gate.pt", root / "gate.json"
        digest = publish_model_and_report_no_clobber(
            model,
            {"kind": "test", "status": "complete"},
            model_output=model_path,
            report_output=report_path,
            repository_root=root,
        )
        assert digest == hashlib.sha256(model_path.read_bytes()).hexdigest()
        validate_gate_model(torch.load(model_path, map_location="cpu", weights_only=True))
        _raises(
            FileExistsError,
            publish_model_and_report_no_clobber,
            model,
            {"kind": "replacement"},
            model_output=model_path,
            report_output=report_path,
            repository_root=root,
        )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        real_parent = root / "real"
        real_parent.mkdir()
        linked_parent = root / "linked"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        _raises(
            ValueError,
            publish_model_and_report_no_clobber,
            model,
            {"kind": "test", "status": "complete"},
            model_output=linked_parent / "gate.pt",
            report_output=linked_parent / "gate.json",
            repository_root=root,
        )


def test_fixed_factual_fit_is_deterministic_and_rejects_development() -> None:
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _manifest_path, plan_path, _manifest = _complete_synthetic(root, "train")
        dataset = _load(root, "train")
        plan = load_analysis_plan(plan_path, require_preregistered=True)
        accepted = {
            "accepted": True,
            "errors": [],
            "cohort": "train",
            "row_count": int(dataset["tensors"]["feature"].shape[0]),
        }
        original_acceptance = trainer_module.validate_collection_acceptance
        trainer_module.validate_collection_acceptance = lambda *_args, **_kwargs: accepted
        try:
            kwargs = {
                "analysis_plan_sha256": plan["_sha256"],
                "input_ledger_sha256": "e" * 64,
                "input_ledger_tag": plan["preregistration"]["train_ledger_tag"],
                "trainer_source_sha256": "d" * 64,
            }
            first, first_report = train_gate_ensemble(dataset, plan, **kwargs)
            second, second_report = train_gate_ensemble(dataset, plan, **kwargs)
        finally:
            trainer_module.validate_collection_acceptance = original_acceptance
        assert semantic_model_sha256(first) == semantic_model_sha256(second)
        assert first_report["model_semantic_sha256"] == second_report[
            "model_semantic_sha256"
        ]
        assert all(
            item["optimizer_steps"] == 200
            for item in first_report["member_diagnostics"]
        )
        assert not torch.equal(
            first["tensors"]["layer0_weight"][0],
            first["tensors"]["layer0_weight"][1],
        )

        _complete_synthetic(root, "development")
        development = _load(root, "development")
        _raises(
            ValueError,
            train_gate_ensemble,
            development,
            plan,
            **kwargs,
        )


if __name__ == "__main__":
    test_population_normalizer_and_floor_are_exact()
    test_factual_gather_gives_no_gradient_to_unassigned_output_heads()
    test_population_bounds_equality_and_nonfinite_default_route()
    test_model_schema_inference_semantic_hash_and_fail_closed()
    test_weights_only_atomic_pair_and_no_clobber()
    test_fixed_factual_fit_is_deterministic_and_rejects_development()
    print("public route gate trainer tests passed")
