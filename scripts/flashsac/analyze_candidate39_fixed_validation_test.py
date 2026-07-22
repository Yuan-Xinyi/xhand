#!/usr/bin/env python3
"""CPU-only tests for Candidate 39 fixed-direction validation analysis."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import torch

import candidate39_fixed_episode_residual as contract
from analyze_candidate39_fixed_validation import (
    ASSIGNMENT_SALT,
    DISCOVERY_ASSIGNMENT_SALT,
    EVENT_FIELDS,
    GATE_THRESHOLDS,
    NUM_ENVS,
    REPLICATES,
    SEEDS,
    _build_gate_results,
    _independent_rank,
    _independent_treatment_mask,
    _registered_validation_gates,
    _validate_sealed_fixed_direction,
    compute_validation,
    publish_json_no_clobber,
    validate_validation_artifacts,
)


SHA = "1" * 64
ALT_SHA = "3" * 64
GIT_SHA = "2" * 40


def _expect(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _plan() -> dict[str, object]:
    path = Path(__file__).resolve().parents[2] / contract.VALIDATION_PLAN
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata(seed: int, replicate: str) -> dict[str, object]:
    plan = _plan()
    policies = plan["immutable_policies"]
    source = {
        "scripts/flashsac/source.py": SHA,
        **plan["implementation_seal"]["source_sha256"],
    }
    assets = {"asset.usd": SHA}
    return {
        **contract.REQUIRED_METADATA,
        "assignment_mask_sha256": contract.assignment_mask_sha256(
            contract.exact_balanced_treatment_mask(
                seed=seed, num_envs=NUM_ENVS, replicate=replicate
            )
        ),
        "seed": seed,
        "replicate": replicate,
        "num_envs": NUM_ENVS,
        "v6_checkpoint": policies["v6_checkpoint"],
        "search_checkpoint": policies["search_checkpoint"],
        "validation_plan_sha256": contract.sha256_file(
            Path(__file__).resolve().parents[2] / contract.VALIDATION_PLAN
        ),
        "v6_actor_sha256": policies["v6_actor_sha256"],
        "v6_task_contract_sha256": policies["v6_task_contract_sha256"],
        "v6_bridge_state_sha256": policies["v6_bridge_state_sha256"],
        "frozen_lift_actor_sha256": policies["frozen_lift_actor_sha256"],
        "frozen_lift_semantic_sha256": policies["frozen_lift_semantic_sha256"],
        "frozen_lift_source_actor_sha256": policies[
            "frozen_lift_source_actor_sha256"
        ],
        "search_checkpoint_sha256": policies["search_checkpoint_sha256"],
        "source_manifest_sha256": contract.manifest_sha256(source),
        "runtime_asset_manifest_sha256": contract.manifest_sha256(assets),
        "source_sha256": source,
        "runtime_asset_sha256": assets,
        "flashsac_upstream_commit": policies["flashsac_fork_commit"],
        "flashsac_fork_commit": policies["flashsac_fork_commit"],
        "git": {
            "commit": GIT_SHA,
            "branch": contract.REQUIRED_BRANCH,
            "source_files_dirty": False,
            "flashsac_commit": policies["flashsac_fork_commit"],
            "flashsac_dirty": False,
        },
        "runtime": {
            "python": "test",
            "torch": str(torch.__version__),
            "cuda": "test",
            "cudnn": 0,
            "cuda_device_index": 0,
            "cuda_device_name": "test",
            "cuda_device_capability": [0, 0],
            "isaac_sim": "test",
            "packages": {
                "isaaclab": "test",
                "isaaclab_tasks": "test",
                "isaaclab_assets": "test",
                "numpy": "test",
                "gymnasium": "test",
            },
            "nvidia_smi_inventory": "test",
            "platform": "test",
            "seed": seed,
        },
    }


def _artifact(
    *,
    seed: int,
    replicate: str,
    favored_slots: tuple[int, ...] = (),
    harmed_slots: tuple[int, ...] = (),
    untriggered_slots: tuple[int, ...] = (),
) -> dict[str, object]:
    treatment = contract.exact_balanced_treatment_mask(
        seed=seed, num_envs=NUM_ENVS, replicate=replicate
    )
    favored = torch.zeros(NUM_ENVS, dtype=torch.bool)
    harmed = torch.zeros(NUM_ENVS, dtype=torch.bool)
    favored[list(favored_slots)] = True
    harmed[list(harmed_slots)] = True
    latched = (favored & treatment) | (harmed & ~treatment)
    triggered = torch.ones(NUM_ENVS, dtype=torch.bool)
    triggered[list(untriggered_slots)] = False
    latched &= triggered
    trigger_step = torch.where(
        triggered,
        torch.full((NUM_ENVS,), 3, dtype=torch.int64),
        torch.full((NUM_ENVS,), -1, dtype=torch.int64),
    )
    trigger_score = torch.where(
        triggered,
        torch.full((NUM_ENVS,), 0.4, dtype=torch.float32),
        torch.zeros(NUM_ENVS, dtype=torch.float32),
    )
    first_latch = torch.where(
        latched,
        trigger_step,
        torch.full((NUM_ENVS,), -1, dtype=torch.int64),
    )

    row_env: list[int] = []
    row_age: list[int] = []
    for age in range(contract.WINDOW_STEPS):
        for slot in range(NUM_ENVS):
            if not bool(triggered[slot]):
                continue
            if bool(latched[slot]) and age > 0:
                continue
            row_env.append(slot)
            row_age.append(age)
    env = torch.tensor(row_env, dtype=torch.int64)
    age = torch.tensor(row_age, dtype=torch.int64)
    rows = int(env.numel())
    active = treatment[env]
    observation = torch.zeros((rows, contract.OBSERVATION_DIM), dtype=torch.float32)
    observation[:, 92:97] = 0.4
    base_mean = torch.zeros((rows, contract.HAND_ACTION_DIM), dtype=torch.float32)
    baseline = torch.zeros((rows, contract.ACTION_DIM), dtype=torch.float32)
    applied = torch.where(
        active[:, None],
        contract.expected_applied_delta().expand(rows, -1),
        torch.zeros((rows, contract.HAND_ACTION_DIM), dtype=torch.float32),
    )
    candidate = baseline.clone()
    overlaid = torch.tanh(base_mean + applied)
    candidate[:, contract.ARM_ACTION_DIM :] = torch.where(
        applied != 0, overlaid, baseline[:, contract.ARM_ACTION_DIM :]
    )
    transition_latch = latched[env] & (age == 0)
    steps = {
        "row_env_slot": env,
        "row_episode_step": trigger_step[env] + age,
        "row_close_age": age,
        "row_public_latch_before": torch.zeros(rows, dtype=torch.bool),
        "row_residual_active": active,
        "row_observation": observation,
        "row_base_mean_hand": base_mean,
        "row_applied_delta": applied,
        "row_baseline_action": baseline,
        "row_candidate_action": candidate,
        "row_executed_action": candidate.clone(),
        "row_transition_public_latch": transition_latch,
        "row_transition_grasped": transition_latch.clone(),
        "row_transition_true_clearance_m": torch.zeros(rows),
        "row_grasp_quality": transition_latch.to(torch.float32),
        "row_hold_quality": transition_latch.to(torch.float32),
        "row_max_force_n": torch.ones(rows),
    }
    intervention_steps = torch.bincount(env[active], minlength=NUM_ENVS)
    episodes = {
        "env_slot": torch.arange(NUM_ENVS, dtype=torch.int64),
        "treatment": treatment,
        "assignment_rank": contract.assignment_rank(
            seed=seed, num_envs=NUM_ENVS
        ),
        "fixed_z": contract.expected_fixed_z().expand(NUM_ENVS, -1).clone(),
        "triggered": triggered,
        "trigger_step": trigger_step,
        "trigger_score": trigger_score,
        "first_latch_step": first_latch,
        "latch_released_after_first": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "intervention_steps": intervention_steps,
        "episode_length": torch.full(
            (NUM_ENVS,), contract.MAX_EPISODE_ACTIONS, dtype=torch.int64
        ),
        "trajectory_max_force_n": torch.ones(NUM_ENVS),
        "max_true_clearance_m": torch.zeros(NUM_ENVS),
        "success": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "failure": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "time_out": torch.ones(NUM_ENVS, dtype=torch.bool),
        "dropped": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "unsafe_force": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "unlatched_clearance_ge_5cm": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "ever_grasped": latched.clone(),
        "ever_clearance_ge_20cm": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "latched_within_window": latched,
    }
    return contract.build_artifact(
        metadata=_metadata(seed, replicate), episodes=episodes, steps=steps
    )


def _artifacts(
    *,
    favored_by_seed: dict[int, tuple[int, ...]] | None = None,
    harmed_by_seed: dict[int, tuple[int, ...]] | None = None,
) -> dict[tuple[int, str], dict[str, object]]:
    favored_by_seed = favored_by_seed or {seed: tuple(range(8)) for seed in SEEDS}
    harmed_by_seed = harmed_by_seed or {seed: () for seed in SEEDS}
    return {
        (seed, replicate): _artifact(
            seed=seed,
            replicate=replicate,
            favored_slots=favored_by_seed[seed],
            harmed_slots=harmed_by_seed[seed],
        )
        for seed in SEEDS
        for replicate in REPLICATES
    }


def test_positive_validation_reports_raw_ht_primary_and_never_writes_direction() -> None:
    report = compute_validation(_artifacts())
    assert report["all_gates_pass"] is True
    assert report["decision"] == "advance_to_formal_screen"
    assert report["direction_generated_or_modified"] is False
    assert "candidate_direction" not in report
    assert "fixed_z" not in report
    assert report["primary"]["per_seed_delta"] == {"329": 0.125, "330": 0.125}
    assert report["primary"]["equal_seed_delta"] == 0.125
    assert report["raw_funnel"]["treatment"][
        "latched_within_32_actions"
    ]["count"] == 16
    assert report["raw_funnel"]["control"][
        "latched_within_32_actions"
    ]["count"] == 0
    assert report["per_seed_horvitz_thompson"]["329"][
        "latched_within_32_actions"
    ]["known_propensity"] == 0.5
    assert set(report["raw_funnel"]["treatment"]) >= {
        "latched_within_32_actions",
        "ever_grasped",
        "success",
        "dropped",
        "unsafe_force",
        "unlatched_clearance_ge_5cm",
    }
    assert report["paired_common_exposure_sensitivity"][
        "common_exposure_opportunity_slots"
    ] == 128
    assert report["step_audit"]["aggregate"][
        "component_and_l2_budget_violations"
    ] == 0
    assert set(report["gates"]) == set(GATE_THRESHOLDS)


def test_zero_primary_rejects_without_generating_or_changing_direction() -> None:
    empty = {seed: () for seed in SEEDS}
    report = compute_validation(
        _artifacts(favored_by_seed=empty, harmed_by_seed=empty)
    )
    assert report["all_gates_pass"] is False
    assert report["decision"] == "reject_fixed_direction"
    assert report["gates"][
        "primary_conditional_latched_within_32_actions_delta_min_exclusive"
    ]["pass"] is False
    assert report["direction_generated_or_modified"] is False
    assert "candidate_direction" not in report


def test_independent_gpu_trigger_differences_are_valid_and_use_actual_domains() -> None:
    artifacts = _artifacts()
    artifacts[(329, "b")] = _artifact(
        seed=329,
        replicate="b",
        favored_slots=tuple(range(8)),
        untriggered_slots=(63,),
    )
    validate_validation_artifacts(artifacts)
    report = compute_validation(artifacts)
    pair = report["paired_common_exposure_sensitivity"]["per_seed"]["329"]
    assert pair["trigger_mask_discordant_slots"] == 1
    assert pair["common_exposure_opportunity_slots"] == 63
    ht = report["per_seed_horvitz_thompson"]["329"][
        "latched_within_32_actions"
    ]
    assert ht["domain_rows"] == 127
    assert report["paired_common_exposure_sensitivity"][
        "trigger_masks_and_clocks_required_bit_exact"
    ] is False


def test_assignment_fixed_z_semantic_source_and_runtime_corruption_fail_closed() -> None:
    artifacts = _artifacts()

    corrupted = copy.deepcopy(artifacts)
    corrupted[(329, "a")]["episodes"]["assignment_rank"][0] ^= 1
    _expect(ValueError, validate_validation_artifacts, corrupted)

    corrupted = copy.deepcopy(artifacts)
    corrupted[(329, "a")]["episodes"]["fixed_z"][0, 0] += 1.0
    _expect(ValueError, validate_validation_artifacts, corrupted)

    corrupted = copy.deepcopy(artifacts)
    corrupted[(329, "a")]["metadata"]["fixed_direction_semantic_sha256"] = ALT_SHA
    _expect(ValueError, validate_validation_artifacts, corrupted)

    # Preserve each seed's A/B identity while changing only seed 330's source
    # lineage.  The analyzer's global identity check must reject this.
    corrupted = copy.deepcopy(artifacts)
    for replicate in REPLICATES:
        metadata = corrupted[(330, replicate)]["metadata"]
        metadata["source_sha256"] = {"scripts/flashsac/source.py": ALT_SHA}
        metadata["source_manifest_sha256"] = contract.manifest_sha256(
            metadata["source_sha256"]
        )
    _expect(ValueError, validate_validation_artifacts, corrupted)

    corrupted = copy.deepcopy(artifacts)
    for replicate in REPLICATES:
        corrupted[(330, replicate)]["metadata"]["runtime"][
            "cuda_device_name"
        ] = "different"
    _expect(ValueError, validate_validation_artifacts, corrupted)


def test_assignment_is_exact_balanced_complement_and_independent_salt() -> None:
    assert ASSIGNMENT_SALT != DISCOVERY_ASSIGNMENT_SALT
    for seed in SEEDS:
        a = _independent_treatment_mask(
            seed=seed, num_envs=NUM_ENVS, replicate="a"
        )
        b = _independent_treatment_mask(
            seed=seed, num_envs=NUM_ENVS, replicate="b"
        )
        assert int(a.sum()) == int(b.sum()) == NUM_ENVS // 2
        assert torch.equal(a, ~b)
        assert torch.equal(
            _independent_rank(seed=seed, num_envs=NUM_ENVS),
            contract.assignment_rank(seed=seed, num_envs=NUM_ENVS),
        )
        assert torch.equal(
            a,
            contract.exact_balanced_treatment_mask(
                seed=seed, num_envs=NUM_ENVS, replicate="a"
            ),
        )


def test_registered_plan_gates_and_every_boundary_are_exact() -> None:
    assert _registered_validation_gates() == GATE_THRESHOLDS
    passing = {
        "primary_delta": 1e-12,
        "positive_primary_seeds": 1,
        "minimum_per_seed_primary_delta": -0.1,
        "ever_grasped_delta": 0.0,
        "success_delta": -0.03,
        "common_exposure_total": 32,
        "common_exposure_min_per_seed": 12,
        "treatment_triggered_dropped": 0,
        "treatment_triggered_unsafe_force": 0,
        "unlatched_clearance_ge_5cm_delta": 0.02,
        "new_abs_action_ge_098_element_fraction": 0.005,
        "max_episodes_with_new_abs_action_ge_098_per_run": 1,
        "any_new_abs_action_ge_0999": False,
        "component_and_l2_budget_violations": 0,
        "action_algebra_exact": True,
        "identity_exact": True,
    }
    assert all(row["pass"] for row in _build_gate_results(passing).values())
    failures = {
        "primary_conditional_latched_within_32_actions_delta_min_exclusive": (
            "primary_delta",
            0.0,
        ),
        "positive_primary_delta_seeds_min": ("positive_primary_seeds", 0),
        "per_seed_primary_delta_floor": ("minimum_per_seed_primary_delta", -0.100001),
        "conditional_ever_grasped_delta_min": ("ever_grasped_delta", -1e-12),
        "conditional_success_delta_floor": ("success_delta", -0.030001),
        "common_exposure_opportunity_slots_min_across_validation": (
            "common_exposure_total",
            31,
        ),
        "common_exposure_opportunity_slots_min_per_seed": (
            "common_exposure_min_per_seed",
            11,
        ),
        "treatment_triggered_dropped_max": ("treatment_triggered_dropped", 1),
        "treatment_triggered_unsafe_force_max": (
            "treatment_triggered_unsafe_force",
            1,
        ),
        "conditional_unlatched_clearance_ge_5cm_delta_max": (
            "unlatched_clearance_ge_5cm_delta",
            0.020001,
        ),
        "new_abs_action_ge_098_element_fraction_max": (
            "new_abs_action_ge_098_element_fraction",
            0.005001,
        ),
        "episodes_with_new_abs_action_ge_098_max_per_run": (
            "max_episodes_with_new_abs_action_ge_098_per_run",
            2,
        ),
        "any_new_abs_action_ge_0999": ("any_new_abs_action_ge_0999", True),
        "component_and_l2_budget_violations": (
            "component_and_l2_budget_violations",
            1,
        ),
        "action_algebra_arm_latch_control_inactive_and_zero_exact": (
            "action_algebra_exact",
            False,
        ),
        "checkpoint_manifest_assignment_and_fixed_z_identity_exact": (
            "identity_exact",
            False,
        ),
    }
    assert set(failures) == set(GATE_THRESHOLDS)
    for gate_name, (metric_name, bad_value) in failures.items():
        metrics = dict(passing)
        metrics[metric_name] = bad_value
        results = _build_gate_results(metrics)
        assert results[gate_name]["pass"] is False, gate_name


def test_fixed_receipt_and_atomic_no_clobber_json() -> None:
    receipt = _validate_sealed_fixed_direction()
    assert receipt["fixed_direction_payload_sha256"] == contract.FIXED_DIRECTION_SHA256
    assert receipt["fixed_direction_semantic_sha256"] == (
        contract.FIXED_DIRECTION_SEMANTIC_SHA256
    )
    assert receipt["direction_was_recomputed_or_modified"] is False
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "validation.json"
        payload = {"decision": "advance_to_formal_screen", "fixed_z_modified": False}
        publish_json_no_clobber(payload, path)
        assert json.loads(path.read_text(encoding="utf-8")) == payload
        _expect(FileExistsError, publish_json_no_clobber, payload, path)


if __name__ == "__main__":
    test_positive_validation_reports_raw_ht_primary_and_never_writes_direction()
    test_zero_primary_rejects_without_generating_or_changing_direction()
    test_independent_gpu_trigger_differences_are_valid_and_use_actual_domains()
    test_assignment_fixed_z_semantic_source_and_runtime_corruption_fail_closed()
    test_assignment_is_exact_balanced_complement_and_independent_salt()
    test_registered_plan_gates_and_every_boundary_are_exact()
    test_fixed_receipt_and_atomic_no_clobber_json()
    print("analyze_candidate39_fixed_validation_test: PASS")
