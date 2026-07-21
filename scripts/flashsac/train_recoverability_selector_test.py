#!/usr/bin/env python3
"""Deterministic, simulation-free tests for recoverability selector training."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Any
from unittest import mock

import torch

import train_recoverability_selector as selector_module
from train_recoverability_selector import (
    DATASET_KIND,
    EXPECTED_FEATURE_POLICY,
    FEATURE_DIMS,
    MODEL_KIND,
    _publish_model_and_report,
    build_features,
    evaluate_threshold,
    feature_contract,
    predict_probabilities,
    sha256_file,
    train_selector,
    validate_dataset,
)


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _metadata(rows: int, seeds: list[int]) -> dict[str, Any]:
    return {
        "pairing_semantics": "independent_gpu_rollout_diagnostic_v1",
        "causal_counterfactual_claim_allowed": False,
        "strong_pair_semantics": "synthetic exact strong pairs",
        "max_observation_abs_error": 1.0e-5,
        "max_action_abs_error": 1.0e-5,
        "max_score_abs_error": 1.0e-5,
        "max_proximity_abs_error": 1.0e-5,
        "max_force_abs_error_n": 1.0e-2,
        "max_clearance_abs_error_m": 1.0e-5,
        "seeds": seeds,
        "rows": rows,
        "strong_rows": rows,
        "rejected_common_rows": 0,
        "strong_discordant_rows": rows,
        "strong_route_only_gain": rows // 2,
        "strong_route_regression": rows // 2,
        "feature_policy": EXPECTED_FEATURE_POLICY,
        "strict_success_semantics": "synthetic strict truth",
        "source_pairs": [],
        "canonical_provenance": {
            "task_mode": "full_task",
            "observation_contract": "pick_tool_markov115_v1",
            "observation_dim": 115,
            "action_dim": 21,
            "requested_episodes": 12,
            "num_envs": 12,
            "episode_length_s": 20.0,
            "max_episode_steps": 1000,
            "deterministic_policy_actions": True,
            "use_compile": False,
            "approach_checkpoint_sha256": "a" * 64,
            "flashsac_actor_sha256": "b" * 64,
            "flashsac_task_contract_sha256": "c" * 64,
            "flashsac_fork_commit": "synthetic-fork",
            "flashsac_upstream_commit": "synthetic-upstream",
            "source_sha256": {"synthetic.py": "d" * 64},
            "supervisor": {},
            "selector_eligible_fields": [
                "observation",
                "search_action",
                "flashsac_action",
            ],
            "audit_only_private_fields": [],
            "outcome_semantics": "synthetic",
        },
        "builder_source_sha256": "e" * 64,
        "selector_input_fields": [
            "observation",
            "search_action",
            "flashsac_action",
        ],
        "preference_label_contract": {
            "target": "route_label",
            "valid_mask": "preference_label_valid",
            "sample_weight": "preference_sample_weight",
            "tie_semantics": "invalid_for_direct_preference",
        },
        "dual_outcome_label_contract": {
            "targets": ["continue_success", "route_success"],
            "valid_mask": "paired_outcome_label_valid",
            "tie_semantics": "valid_for_two_independent_outcome_heads",
        },
        "forbidden_selector_feature_patterns": [
            "*_success",
            "*_failure",
            "*_time_out",
            "*_dropped",
            "*_unsafe_force",
            "*clearance*",
            "*pregrasp_score*",
            "*proximity_quality*",
            "*max_force*",
            "*handoff_step*",
            "seed",
            "env_slot",
            "slot_episode_index",
        ],
    }


def _synthetic_dataset() -> dict[str, Any]:
    seeds = [1, 2, 3, 4]
    per_seed = 12
    rows = len(seeds) * per_seed
    observation = torch.zeros((rows, 115), dtype=torch.float32)
    search_action = torch.zeros((rows, 21), dtype=torch.float32)
    flashsac_action = torch.zeros((rows, 21), dtype=torch.float32)
    seed_tensor = torch.empty(rows, dtype=torch.long)
    continue_success = torch.empty(rows, dtype=torch.bool)
    route_success = torch.empty(rows, dtype=torch.bool)
    for group, seed in enumerate(seeds):
        start = group * per_seed
        stop = start + per_seed
        latent = torch.linspace(-2.0, 2.0, per_seed)
        seed_tensor[start:stop] = seed
        observation[start:stop, 0] = latent
        observation[start:stop, 1] = float(seed) * 0.01
        observation[start:stop, 38:53] = latent[:, None] * 0.02
        observation[start:stop, 53:56] = latent[:, None] * 0.01
        observation[start:stop, 56:59] = torch.tensor([0.1, -0.2, 0.3])
        observation[start:stop, 59] = 1.0
        observation[start:stop, 70] = latent * 0.1
        observation[start:stop, 92] = latent * 0.05
        search_action[start:stop, 0] = latent * 0.2
        flashsac_action[start:stop, 0] = -latent * 0.2
        continue_success[start:stop] = latent < 0.4
        route_success[start:stop] = latent > -0.4
    valid = torch.ones(rows, dtype=torch.bool)
    zeros = torch.zeros(rows, dtype=torch.bool)
    return {
        "kind": DATASET_KIND,
        "format_version": 1,
        "metadata": _metadata(rows, seeds),
        "observation": observation,
        "continue_observation": observation.clone(),
        "search_action": search_action,
        "continue_search_action": search_action.clone(),
        "flashsac_action": flashsac_action,
        "continue_flashsac_action": flashsac_action.clone(),
        "seed": seed_tensor,
        "strong_pair": valid.clone(),
        "paired_outcome_label_valid": valid.clone(),
        "continue_success": continue_success,
        "route_success": route_success,
        "continue_dropped": zeros.clone(),
        "route_dropped": zeros.clone(),
        "continue_unsafe_force": zeros.clone(),
        "route_unsafe_force": zeros.clone(),
    }


def _clone_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.clone()
            if isinstance(value, torch.Tensor)
            else dict(value)
            if key == "metadata"
            else value
        )
        for key, value in payload.items()
    }


def test_feature_contracts_and_object_frame_geometry() -> None:
    observation = torch.zeros((2, 115), dtype=torch.float64)
    search = torch.arange(21, dtype=torch.float64).repeat(2, 1)
    flash = -search
    observation[:, 0:38] = torch.arange(38, dtype=torch.float64)

    object_position = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    pad_delta = torch.arange(15, dtype=torch.float64).reshape(5, 3) * 0.1
    palm_delta = torch.tensor([0.4, -0.3, 0.2], dtype=torch.float64)
    observation[0, 56:59] = object_position
    observation[0, 59] = 1.0
    observation[0, 38:53] = (object_position + pad_delta).reshape(15)
    observation[0, 53:56] = object_position + palm_delta

    # +90 degrees about world z: object x maps to world y.
    root_half = math.sqrt(0.5)
    observation[1, 59:63] = torch.tensor(
        [root_half, 0.0, 0.0, root_half], dtype=torch.float64
    )
    world_y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    observation[1, 38:53] = world_y.repeat(5)
    observation[1, 53:56] = world_y
    observation[:, 70:86] = 0.25
    observation[:, 86] = 0.125
    observation[:, 87:92] = -0.5
    observation[:, 92:115] = 0.75

    features = build_features(observation, search, flash, "public_gate_v1")
    assert features.shape == (2, FEATURE_DIMS["public_gate_v1"])
    assert torch.allclose(features[0, 38:53], pad_delta.reshape(15), atol=1.0e-12)
    assert torch.allclose(features[0, 53:56], palm_delta, atol=1.0e-12)
    assert torch.equal(features[0, 56:59], object_position)
    assert torch.allclose(
        features[0, 59:65],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64),
        atol=1.0e-12,
    )
    assert torch.allclose(
        features[1, 38:53].reshape(5, 3),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).repeat(5, 1),
        atol=1.0e-12,
    )
    assert torch.allclose(
        features[1, 59:65],
        torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=torch.float64),
        atol=1.0e-12,
    )
    assert torch.equal(features[:, 65:81], observation[:, 70:86])
    assert torch.equal(features[:, 81:86], observation[:, 87:92])
    assert torch.equal(features[:, 86:87], observation[:, 86:87])
    assert torch.equal(features[:, 87:110], observation[:, 92:115])
    assert torch.equal(features[:, 110:131], search)
    assert torch.equal(features[:, 131:152], flash)
    assert feature_contract("public_gate_v1")["pad_input_order"] == [
        "middle",
        "pinky",
        "ring",
        "index",
        "thumb",
    ]

    raw = build_features(observation, search, flash, "raw_public_v1")
    assert raw.shape == (2, FEATURE_DIMS["raw_public_v1"])
    changed_target = observation.clone()
    changed_target[:, 63:70] = 999.0
    assert torch.equal(raw, build_features(changed_target, search, flash, "raw_public_v1"))
    assert torch.equal(
        features, build_features(changed_target, search, flash, "public_gate_v1")
    )

    local = build_features(observation, search, flash, "local84_v1")
    assert local.shape == (2, FEATURE_DIMS["local84_v1"])
    assert torch.allclose(local[0, 0:15], pad_delta.reshape(15), atol=1.0e-12)
    assert torch.allclose(local[0, 15:18], palm_delta, atol=1.0e-12)
    assert torch.equal(local[:, 18:19], observation[:, 86:87])
    assert torch.equal(local[:, 19:42], observation[:, 92:115])
    assert torch.equal(local[:, 42:63], search)
    assert torch.equal(local[:, 63:84], flash)
    contract = feature_contract("local84_v1")
    assert contract["dimension"] == 84
    assert "q_0_19" in contract["explicitly_excluded"]
    assert "qd_19_38" in contract["explicitly_excluded"]
    assert "object_world_position_56_59" in contract["explicitly_excluded"]
    assert "previous_action_70_86_plus_87_92" in contract["explicitly_excluded"]

    excluded_changes = observation.clone()
    excluded_changes[:, 0:38] += 1000.0
    excluded_changes[:, 63:86] -= 2000.0
    excluded_changes[:, 87:92] += 3000.0
    assert torch.equal(
        local, build_features(excluded_changes, search, flash, "local84_v1")
    )
    translated = observation.clone()
    translation = torch.tensor([7.0, -5.0, 3.0], dtype=torch.float64)
    translated[:, 38:53] = (
        translated[:, 38:53].reshape(2, 5, 3) + translation
    ).reshape(2, 15)
    translated[:, 53:56] += translation
    translated[:, 56:59] += translation
    assert torch.allclose(
        local,
        build_features(translated, search, flash, "local84_v1"),
        atol=1.0e-12,
    )


def test_dataset_validation_fails_closed() -> None:
    dataset = _synthetic_dataset()
    validated = validate_dataset(dataset)
    assert validated["metadata"]["seeds"] == [1, 2, 3, 4]

    weakened = _clone_dataset(dataset)
    weakened["metadata"] = dict(weakened["metadata"])
    weakened["metadata"]["feature_policy"] = "all fields allowed"
    _expect_error(ValueError, validate_dataset, weakened)

    noncanonical = _clone_dataset(dataset)
    noncanonical["observation"][0, 4] = 1.0
    _expect_error(ValueError, validate_dataset, noncanonical)

    bad_mask = _clone_dataset(dataset)
    bad_mask["paired_outcome_label_valid"][0] = False
    _expect_error(ValueError, validate_dataset, bad_mask)

    latched = _clone_dataset(dataset)
    latched["observation"][0, 106] = 1.0
    latched["continue_observation"][0, 106] = 1.0
    _expect_error(ValueError, validate_dataset, latched)


def _run_synthetic_training(dataset: dict[str, Any]):
    return train_selector(
        dataset,
        dataset_sha256="f" * 64,
        dev_seed=4,
        feature_sets=["local84_v1", "public_gate_v1", "raw_public_v1"],
        l2_values=[0.01, 0.1],
        p_min_values=[0.25, 0.4, 0.6, 1.0],
        margin_values=[0.0, 0.05, 0.2],
        max_iter=100,
        minimum_cv_net_gain=1,
        minimum_gain_retention=0.75,
        minimum_regression_avoidance=0.25,
    )


def test_grouped_selection_is_deterministic_and_dev_is_isolated() -> None:
    dataset = _synthetic_dataset()
    model_a, report_a = _run_synthetic_training(dataset)
    model_b, report_b = _run_synthetic_training(dataset)
    assert model_a["kind"] == MODEL_KIND
    for name in ("feature_mean", "feature_scale", "head_weight", "head_bias"):
        assert torch.equal(model_a[name], model_b[name])
    assert report_a["selected"] == report_b["selected"]
    assert report_a["cross_validated_train_seed_report"] == report_b[
        "cross_validated_train_seed_report"
    ]
    assert model_a["metadata"]["training_seeds"] == [1, 2, 3]
    assert model_a["metadata"]["development_seed"] == 4
    assert model_a["metadata"]["max_iter"] == 100
    assert model_a["metadata"]["torch_version"] == str(torch.__version__)
    assert model_a["metadata"]["training_configuration"]["feature_sets"] == [
        "local84_v1",
        "public_gate_v1",
        "raw_public_v1",
    ]
    assert model_a["metadata"]["head_order"] == [
        "continue_search_strict_success",
        "route_flashsac_strict_success",
    ]

    # Changing only held-out dev features and labels may alter its report and
    # semantic data hash, but never selection or final parameters.
    changed_dev = _clone_dataset(dataset)
    dev = changed_dev["seed"] == 4
    changed_dev["observation"][dev, 0] *= -1.0
    changed_dev["continue_observation"][dev, 0] *= -1.0
    changed_dev["continue_success"][dev] = ~changed_dev["continue_success"][dev]
    model_c, report_c = _run_synthetic_training(changed_dev)
    for name in ("feature_mean", "feature_scale", "head_weight", "head_bias"):
        assert torch.equal(model_a[name], model_c[name])
    for name in (
        "feature_set",
        "feature_dimension",
        "l2",
        "p_continue_min",
        "continue_probability_margin",
        "loo_threshold_metrics",
    ):
        assert report_a["selected"][name] == report_c["selected"][name]
    assert report_a["cross_validated_train_seed_report"] == report_c[
        "cross_validated_train_seed_report"
    ]


def test_known_probability_head_order_and_default_route_veto() -> None:
    requested_probabilities = torch.tensor(
        [
            [0.90, 0.20],  # high-confidence continue veto
            [0.60, 0.80],  # route head is stronger: keep default route
            [0.40, 0.10],  # continue is stronger but below absolute floor
            [0.80, 0.75],  # continue margin is too small
        ],
        dtype=torch.float64,
    )
    feature_logits = torch.logit(requested_probabilities)
    probabilities = predict_probabilities(
        feature_logits,
        torch.zeros(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    assert torch.allclose(probabilities, requested_probabilities, atol=1.0e-12)

    false = torch.zeros(4, dtype=torch.bool)
    outcomes = {
        # Only row zero benefits from continuing.  If the head columns or veto
        # direction are reversed, the selected-success assertion fails.
        "continue_success": torch.tensor([True, False, False, False]),
        "route_success": false.clone(),
        "continue_dropped": false.clone(),
        "route_dropped": false.clone(),
        "continue_unsafe_force": false.clone(),
        "route_unsafe_force": false.clone(),
    }
    metrics = evaluate_threshold(
        probabilities,
        outcomes,
        torch.ones(4, dtype=torch.long),
        p_min=0.50,
        margin=0.10,
        minimum_gain_retention=0.75,
    )
    assert metrics["aggregate"]["continue_decisions"] == 1
    assert metrics["aggregate"]["route_decisions"] == 3
    assert metrics["aggregate"]["selected_successes"] == 1
    assert metrics["aggregate"]["net_success_delta_vs_route"] == 1


def test_transactional_no_clobber_publication() -> None:
    model, report = _run_synthetic_training(_synthetic_dataset())
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "selector.pt"
        report_path = root / "selector.json"
        final_report = _publish_model_and_report(
            model, report, output=output, report_path=report_path
        )
        assert output.is_file() and report_path.is_file()
        assert final_report["model_sha256"] == sha256_file(output)
        with report_path.open("r", encoding="utf-8") as stream:
            assert json.load(stream)["model_sha256"] == sha256_file(output)
        loaded = torch.load(output, map_location="cpu", weights_only=True)
        assert loaded["kind"] == MODEL_KIND
        _expect_error(
            FileExistsError,
            _publish_model_and_report,
            model,
            report,
            output=output,
            report_path=report_path,
        )

    # The first hard link publishes the model.  Inject a failure in the second
    # link and verify that publication rolls the model back and leaves neither
    # a final report nor a staging file behind.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "rollback_selector.pt"
        report_path = root / "rollback_selector.json"
        real_link = selector_module.os.link
        link_calls = 0

        def fail_second_link(source, destination):
            nonlocal link_calls
            link_calls += 1
            if link_calls == 2:
                raise OSError("injected report hard-link failure")
            return real_link(source, destination)

        with mock.patch.object(
            selector_module.os, "link", side_effect=fail_second_link
        ):
            _expect_error(
                OSError,
                _publish_model_and_report,
                model,
                report,
                output=output,
                report_path=report_path,
            )
        assert link_calls == 2
        assert not output.exists() and not output.is_symlink()
        assert not report_path.exists() and not report_path.is_symlink()
        assert not list(root.glob(".*.tmp-*"))


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    test_feature_contracts_and_object_frame_geometry()
    test_dataset_validation_fails_closed()
    test_grouped_selection_is_deterministic_and_dev_is_isolated()
    test_known_probability_head_order_and_default_route_veto()
    test_transactional_no_clobber_publication()
    print("train_recoverability_selector tests passed")


if __name__ == "__main__":
    main()
