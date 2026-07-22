#!/usr/bin/env python3
"""Simulation-free contract tests for the recoverability selector runtime."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

import train_recoverability_selector as trainer
from recoverability_selector_runtime import (
    BLIND_MANIFEST_KIND,
    EXPECTED_DECISION_SEMANTICS,
    EXPECTED_HEAD_ORDER,
    MODEL_KIND,
    TRAINER_SOURCE_KEY,
    load_blind_runtime_manifest,
    load_recoverability_selector,
    sha256_file,
)
from evaluate import validate_recoverability_selector_evaluation_config


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _base_model() -> dict[str, Any]:
    feature_set = "raw_public_v1"
    dimension = trainer.FEATURE_DIMS[feature_set]
    trainer_sha = sha256_file(Path(trainer.__file__).resolve())
    metadata = {
        "dataset_kind": "pick_tool_recoverability_pairs_v1",
        "dataset_sha256": "a" * 64,
        "data_semantic_sha256": "b" * 64,
        "chosen_feature_semantic_sha256": "c" * 64,
        "source_sha256": {TRAINER_SOURCE_KEY: trainer_sha},
        "torch_version": str(torch.__version__),
        "max_iter": 200,
        "training_configuration": {"feature_sets": [feature_set]},
        "feature_contract": trainer.feature_contract(feature_set),
        "head_order": list(EXPECTED_HEAD_ORDER),
        "decision_semantics": EXPECTED_DECISION_SEMANTICS,
        "p_continue_min": 0.5,
        "continue_probability_margin": 0.1,
        "l2": 0.1,
        "training_seeds": [257, 258, 259],
        "development_seed": 260,
        "training_rows": 10,
        "development_rows": 4,
        "normalization": "training_seed_rows_only_population_std_constant_to_one",
        "dtype": "float64",
        "pairing_semantics": "independent_gpu_rollout_diagnostic_v1",
        "causal_counterfactual_claim_allowed": False,
        "collection_canonical_provenance": {
            "task_mode": "full_task",
            "observation_contract": "pick_tool_markov115_v1",
            "observation_dim": 115,
            "action_dim": 21,
            "requested_episodes": 512,
            "num_envs": 512,
            "deterministic_policy_actions": True,
            "episode_length_s": 20.0,
            "max_episode_steps": 1000,
            "use_compile": False,
            "approach_checkpoint_sha256": "1" * 64,
            "flashsac_actor_sha256": "2" * 64,
            "flashsac_task_contract_sha256": "3" * 64,
            "flashsac_fork_commit": "4" * 40,
            "flashsac_upstream_commit": "5" * 40,
            "source_sha256": {"synthetic_collection.py": "d" * 64},
            "supervisor": {
                "minimum_zero_based_episode_step": 400,
                "pregrasp_score_threshold": 0.32,
                "minimum_proximity_quality": 0.02,
                "hold_steps": 4,
                "safe_force_limit_n": 30.0,
                "requires_unlatched": True,
                "requires_abs_true_clearance_le_m": 0.005,
            },
            "selector_eligible_fields": list(trainer.EXPECTED_INPUT_FIELDS),
            "audit_only_private_fields": ["handoff_step"],
        },
    }
    weight = torch.zeros((2, dimension), dtype=torch.float64)
    # raw_public_v1 places SEARCH action[0] at feature 108.  Opposite heads
    # make the decision direction unambiguous without depending on rounding.
    weight[0, 108] = 2.0
    weight[1, 108] = -2.0
    return {
        "kind": MODEL_KIND,
        "format_version": 1,
        "metadata": metadata,
        "feature_mean": torch.zeros(dimension, dtype=torch.float64),
        "feature_scale": torch.ones(dimension, dtype=torch.float64),
        "head_weight": weight,
        "head_bias": torch.zeros(2, dtype=torch.float64),
    }


def _report(model: dict[str, Any], model_path: Path, model_sha: str) -> dict[str, Any]:
    metadata = model["metadata"]
    return {
        "status": "complete",
        "kind": MODEL_KIND,
        "format_version": 1,
        "dataset_sha256": metadata["dataset_sha256"],
        "data_semantic_sha256": metadata["data_semantic_sha256"],
        "trainer_source_sha256": metadata["source_sha256"][TRAINER_SOURCE_KEY],
        "torch_version": metadata["torch_version"],
        "max_iter": metadata["max_iter"],
        "training_configuration": metadata["training_configuration"],
        "training_seeds": metadata["training_seeds"],
        "development_seed": metadata["development_seed"],
        "accepted_for_blind_simulation_evaluation": True,
        "cross_validated_train_seed_report": {"acceptance": {"accepted": True}},
        "held_out_development_report": {"acceptance": {"accepted": True}},
        "model": str(model_path),
        "model_sha256": model_sha,
        "model_metadata": metadata,
        "selected": {
            "feature_set": metadata["feature_contract"]["name"],
            "feature_dimension": metadata["feature_contract"]["dimension"],
            "l2": metadata["l2"],
            "p_continue_min": metadata["p_continue_min"],
            "continue_probability_margin": metadata[
                "continue_probability_margin"
            ],
        },
    }


def _publish(
    directory: Path,
    *,
    mutate_model: Callable[[dict[str, Any]], None] | None = None,
    mutate_report: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, Path]:
    model = copy.deepcopy(_base_model())
    if mutate_model is not None:
        mutate_model(model)
    model_path = directory / "selector.pt"
    report_path = directory / "selector.json"
    torch.save(model, model_path)
    report = _report(model, model_path, sha256_file(model_path))
    if mutate_report is not None:
        mutate_report(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return model_path, report_path


def test_serialization_probabilities_and_default_route_direction() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        model_path, report_path = _publish(Path(directory_name))
        runtime = load_recoverability_selector(
            model_path, report_path, device=torch.device("cpu")
        )
        observation = torch.zeros((2, 115), dtype=torch.float32)
        observation[:, 59] = 1.0  # valid wxyz identity quaternion
        search_action = torch.zeros((2, 21), dtype=torch.float32)
        search_action[:, 0] = torch.tensor([1.0, -1.0])
        flashsac_action = torch.zeros((2, 21), dtype=torch.float32)
        decision = runtime.decide(observation, search_action, flashsac_action)
        high = 1.0 / (1.0 + math.exp(-2.0))
        low = 1.0 / (1.0 + math.exp(2.0))
        expected = torch.tensor(
            [
                [high, low],
                [low, high],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(decision.probabilities, expected, atol=1.0e-12, rtol=0.0)
        assert torch.equal(decision.continue_search, torch.tensor([True, False]))
        assert torch.equal(decision.route_flashsac, torch.tensor([False, True]))
        assert runtime.audit_metadata()["default_decision"] == "route_flashsac"
        provenance = runtime.metadata["collection_canonical_provenance"]
        runtime.validate_live_evaluation_contract(
            approach_checkpoint_sha256="1" * 64,
            flashsac_actor_sha256="2" * 64,
            flashsac_task_contract_sha256="3" * 64,
            flashsac_fork_commit="4" * 40,
            flashsac_upstream_commit="5" * 40,
            use_compile=False,
            supervisor=provenance["supervisor"],
            collection_source_sha256=provenance["source_sha256"],
        )
        _expect_error(
            ValueError,
            runtime.validate_live_evaluation_contract,
            approach_checkpoint_sha256="1" * 64,
            flashsac_actor_sha256="2" * 64,
            flashsac_task_contract_sha256="3" * 64,
            flashsac_fork_commit="4" * 40,
            flashsac_upstream_commit="5" * 40,
            use_compile=False,
            supervisor={**provenance["supervisor"], "pregrasp_score_threshold": 0.3},
            collection_source_sha256=provenance["source_sha256"],
        )
        smoke_audit = runtime.validate_blind_configuration(
            seed=257,
            episodes=32,
            num_envs=32,
            strict_blind=False,
        )
        assert smoke_audit["seen_seed"] is True
        assert smoke_audit["scale_matches_collection"] is False
        assert smoke_audit["blind_claim_allowed"] is False
        _expect_error(
            ValueError,
            runtime.validate_blind_configuration,
            seed=257,
            episodes=512,
            num_envs=512,
            strict_blind=True,
        )
        _expect_error(
            ValueError,
            runtime.validate_blind_configuration,
            seed=261,
            episodes=64,
            num_envs=64,
            strict_blind=True,
        )
        blind_audit = runtime.validate_blind_configuration(
            seed=261,
            episodes=512,
            num_envs=512,
            strict_blind=True,
        )
        assert blind_audit["seen_seed"] is False
        assert blind_audit["scale_matches_collection"] is True
        assert blind_audit["blind_claim_allowed"] is True
        runtime.verify_unchanged()


def test_loader_rejects_contract_and_integrity_corruption() -> None:
    corruptions: list[
        tuple[
            Callable[[dict[str, Any]], None] | None,
            Callable[[dict[str, Any]], None] | None,
        ]
    ] = [
        (
            lambda model: model["feature_scale"].__setitem__(0, -1.0),
            None,
        ),
        (
            lambda model: model.__setitem__(
                "head_bias", model["head_bias"].to(torch.float32)
            ),
            None,
        ),
        (
            lambda model: model["metadata"].__setitem__(
                "head_order", list(reversed(EXPECTED_HEAD_ORDER))
            ),
            None,
        ),
        (
            lambda model: model["metadata"]["feature_contract"].__setitem__(
                "dimension", 149
            ),
            None,
        ),
        (
            lambda model: model["metadata"].__setitem__(
                "source_sha256", {TRAINER_SOURCE_KEY: "0" * 64}
            ),
            None,
        ),
        (
            lambda model: model["metadata"].__setitem__("p_continue_min", math.nan),
            None,
        ),
        (
            None,
            lambda report: report.__setitem__(
                "accepted_for_blind_simulation_evaluation", False
            ),
        ),
        (None, lambda report: report.__setitem__("model_sha256", "f" * 64)),
    ]
    for index, (mutate_model, mutate_report) in enumerate(corruptions):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name) / str(index)
            directory.mkdir()
            # NaN metadata cannot be written with strict JSON, which itself is
            # a useful fail-closed publication check.  Serialize that one with
            # Python's permissive encoder so the loader gets to reject it.
            if index == 5:
                model = _base_model()
                mutate_model(model)  # type: ignore[misc]
                model_path = directory / "selector.pt"
                report_path = directory / "selector.json"
                torch.save(model, model_path)
                report_path.write_text(
                    json.dumps(_report(model, model_path, sha256_file(model_path))),
                    encoding="utf-8",
                )
            else:
                model_path, report_path = _publish(
                    directory,
                    mutate_model=mutate_model,
                    mutate_report=mutate_report,
                )
            _expect_error(
                (ValueError if index != 1 else TypeError),
                load_recoverability_selector,
                model_path,
                report_path,
                device="cpu",
            )


def test_runtime_rejects_non_public_input_contracts() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        model_path, report_path = _publish(Path(directory_name))
        runtime = load_recoverability_selector(model_path, report_path, device="cpu")
        observation = torch.zeros((1, 115), dtype=torch.float32)
        observation[:, 59] = 1.0
        action = torch.zeros((1, 21), dtype=torch.float32)
        _expect_error(
            TypeError,
            runtime.decide,
            observation.to(torch.float64),
            action,
            action,
        )
        bad = observation.clone()
        bad[0, 0] = float("nan")
        _expect_error(ValueError, runtime.decide, bad, action, action)
        _expect_error(
            ValueError,
            runtime.decide,
            observation,
            action[:, :20],
            action,
        )


def test_probability_clamp_matches_trainer() -> None:
    def saturate(model: dict[str, Any]) -> None:
        model["head_weight"].zero_()
        model["head_bias"] = torch.tensor(
            [1000.0, -1000.0], dtype=torch.float64
        )

    with tempfile.TemporaryDirectory() as directory_name:
        model_path, report_path = _publish(
            Path(directory_name), mutate_model=saturate
        )
        runtime = load_recoverability_selector(model_path, report_path, device="cpu")
        observation = torch.zeros((1, 115), dtype=torch.float32)
        observation[:, 59] = 1.0
        action = torch.zeros((1, 21), dtype=torch.float32)
        probabilities = runtime.decide(observation, action, action).probabilities
        epsilon = torch.finfo(torch.float64).eps
        assert probabilities[0, 0].item() == 1.0 - epsilon
        assert probabilities[0, 1].item() == epsilon


def test_independent_blind_manifest_pins_evaluator_and_runtime() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        evaluator = directory / "evaluate.py"
        runtime_source = directory / "runtime.py"
        manifest = directory / "manifest.json"
        evaluator.write_text("reviewed evaluator\n", encoding="utf-8")
        runtime_source.write_text("reviewed runtime\n", encoding="utf-8")
        manifest.write_text(
            json.dumps(
                {
                    "kind": BLIND_MANIFEST_KIND,
                    "format_version": 1,
                    "source_sha256": {
                        "scripts/flashsac/evaluate.py": sha256_file(evaluator),
                        "scripts/flashsac/recoverability_selector_runtime.py": (
                            sha256_file(runtime_source)
                        ),
                    },
                },
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        loaded = load_blind_runtime_manifest(
            manifest,
            evaluator_path=evaluator,
            runtime_path=runtime_source,
        )
        loaded.verify_unchanged()
        evaluator.write_text("unreviewed evaluator\n", encoding="utf-8")
        _expect_error(RuntimeError, loaded.verify_unchanged)
        _expect_error(
            ValueError,
            load_blind_runtime_manifest,
            manifest,
            evaluator_path=evaluator,
            runtime_path=runtime_source,
        )


def test_evaluator_cli_contract_is_fail_closed() -> None:
    selector = Path("selector.pt")
    report = Path("selector.json")
    approach = Path("search.pth")
    validate_recoverability_selector_evaluation_config(
        selector_checkpoint=selector,
        selector_report=report,
        approach_checkpoint=approach,
        approach_base_only=False,
        approach_handoff_output=None,
        selector_blind=True,
    )
    _expect_error(
        ValueError,
        validate_recoverability_selector_evaluation_config,
        selector_checkpoint=None,
        selector_report=None,
        approach_checkpoint=None,
        approach_base_only=False,
        approach_handoff_output=None,
        selector_blind=True,
    )
    for kwargs in (
        {
            "selector_checkpoint": selector,
            "selector_report": None,
            "approach_checkpoint": approach,
            "approach_base_only": False,
            "approach_handoff_output": None,
        },
        {
            "selector_checkpoint": selector,
            "selector_report": report,
            "approach_checkpoint": None,
            "approach_base_only": False,
            "approach_handoff_output": None,
        },
        {
            "selector_checkpoint": selector,
            "selector_report": report,
            "approach_checkpoint": approach,
            "approach_base_only": True,
            "approach_handoff_output": None,
        },
        {
            "selector_checkpoint": selector,
            "selector_report": report,
            "approach_checkpoint": approach,
            "approach_base_only": False,
            "approach_handoff_output": Path("evidence.pt"),
        },
    ):
        _expect_error(
            ValueError,
            validate_recoverability_selector_evaluation_config,
            **kwargs,
        )


def main() -> None:
    test_serialization_probabilities_and_default_route_direction()
    test_loader_rejects_contract_and_integrity_corruption()
    test_runtime_rejects_non_public_input_contracts()
    test_probability_clamp_matches_trainer()
    test_independent_blind_manifest_pins_evaluator_and_runtime()
    test_evaluator_cli_contract_is_fail_closed()
    print("recoverability_selector_runtime tests passed")


if __name__ == "__main__":
    main()
