#!/usr/bin/env python3
"""CPU-only tests for Candidate41's fail-closed development analyzer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

import pytest
import torch

import candidate41_public_arm_ramp as ramp
import candidate41_public_arm_ramp_episode as contract
from candidate41_public_arm_ramp_episode_test import _artifact as _base_artifact
from analyze_candidate41_public_arm_ramp_validation import (
    ATTEMPT_RECEIPT_FIELDS,
    ATTEMPT_RECEIPT_KIND,
    GATE_NAMES,
    RUN_ORDER,
    SEEDS,
    _action_clock_audit,
    _authority_by_age_count,
    _build_gate_results,
    _canonical_invocation_stem,
    _ht_outcome,
    _independent_rank,
    _independent_treatment_mask,
    _paired_common_eligibility,
    _strict_json_bytes,
    compute_validation,
    publish_json_no_clobber,
    registered_gate_thresholds,
    stable_transport_outcomes,
    validate_attempt_receipts,
    validate_validation_artifacts,
)


def _control_u5_artifact(*, seed: int, replicate: str) -> dict:
    """Make a valid treatment-success/control-U5 contrast from the fixture."""

    artifact = copy.deepcopy(
        _base_artifact(seed=seed, replicate=replicate, num_envs=64)
    )
    episodes = artifact["episodes"]
    steps = artifact["steps"]
    control = ~episodes["treatment"]
    episodes["success"][control] = False
    episodes["failure"][control] = True
    episodes["unlatched_clearance_ge_5cm"][control] = True

    terminal_control = (
        (~steps["row_treatment"])
        & steps["row_transition_done"]
    )
    steps["row_transition_observation"][
        terminal_control, ramp.PUBLIC_LATCH_INDEX
    ] = 0.0
    steps["row_transition_public_latch"][terminal_control] = False
    steps["row_transition_grasped"][terminal_control] = False
    steps["row_transition_grasp_quality"][terminal_control] = 0.0
    steps["row_transition_hold_quality"][terminal_control] = 0.0
    steps["row_transition_max_force_n"][terminal_control] = 0.0
    return contract.validate_artifact(
        artifact, require_sealed_plan=False
    )


def _artifacts() -> dict[tuple[int, str], dict]:
    return {
        (seed, replicate): _control_u5_artifact(
            seed=seed, replicate=replicate
        )
        for seed, replicate in RUN_ORDER
    }


def _passing_metrics() -> dict[str, object]:
    """Return values exactly on every inclusive registered boundary."""

    return {
        "primary_delta": 0.1000000000001,
        "positive_primary_seeds": 1,
        "minimum_primary_seed_delta": -0.05,
        "transport_delta": -0.08,
        "minimum_transport_seed_delta": -0.15,
        "unlatched5_delta": -0.08,
        "paired_unlatched5_delta": -0.08,
        "success_delta": -0.03,
        "true20_delta": -0.03,
        "stable5_delta": -0.05,
        "ever_grasped_delta": -0.03,
        "eligible_total": 32,
        "eligible_min_seed": 12,
        "eligible_min_seed_arm": 8,
        "paired_common_total": 16,
        "paired_common_min_seed": 6,
        "positive16_total": 16,
        "positive16_min_seed": 4,
        "full32_total": 12,
        "full32_min_seed": 3,
        "partial_rows": 32,
        "treatment_unlatched5": 3,
        "pooled_preeligible_unlatched5": 0,
        "treatment_dropped": 0,
        "treatment_unsafe": 0,
        "fixed_budget_violations": 0,
        "any_new_0999": False,
        "preeligibility_exact": True,
        "hand_exact": True,
        "ramp_exact": True,
        "control_exact": True,
        "identity_exact": True,
    }


def test_all_32_registered_gate_boundaries_are_executable_and_exact() -> None:
    thresholds = registered_gate_thresholds()
    assert len(GATE_NAMES) == 32
    assert set(thresholds) == set(GATE_NAMES)
    passing = _passing_metrics()
    gates = _build_gate_results(passing, thresholds)
    assert set(gates) == set(GATE_NAMES)
    assert all(row["pass"] for row in gates.values())

    failures = {
        "primary_conditional_safe_terminal_utility_delta_min_exclusive": (
            "primary_delta", 0.1
        ),
        "positive_primary_delta_seeds_min": ("positive_primary_seeds", 0),
        "per_seed_primary_delta_floor": ("minimum_primary_seed_delta", -0.050001),
        "conditional_stable_transport_restricted_mean_delta_floor": (
            "transport_delta", -0.080001
        ),
        "per_seed_stable_transport_delta_floor": (
            "minimum_transport_seed_delta", -0.150001
        ),
        "conditional_unlatched_clearance_ge_5cm_delta_max": (
            "unlatched5_delta", -0.079999
        ),
        "paired_common_eligibility_unlatched_clearance_ge_5cm_delta_max": (
            "paired_unlatched5_delta", -0.079999
        ),
        "conditional_success_delta_floor": ("success_delta", -0.030001),
        "conditional_true_clearance_ge_20cm_delta_floor": (
            "true20_delta", -0.030001
        ),
        "conditional_stable_held_5cm_by_action96_delta_floor": (
            "stable5_delta", -0.050001
        ),
        "conditional_ever_grasped_delta_floor": (
            "ever_grasped_delta", -0.030001
        ),
        "eligible_opportunity_slots_min_across_validation": ("eligible_total", 31),
        "eligible_opportunity_slots_min_per_seed": ("eligible_min_seed", 11),
        "eligible_opportunity_slots_min_per_seed_arm": ("eligible_min_seed_arm", 7),
        "paired_common_eligible_slots_min_across_validation": (
            "paired_common_total", 15
        ),
        "paired_common_eligible_slots_min_per_seed": (
            "paired_common_min_seed", 5
        ),
        "treatment_eligible_positive_authority_by_action16_min_across_validation": (
            "positive16_total", 15
        ),
        "treatment_eligible_positive_authority_by_action16_min_per_seed": (
            "positive16_min_seed", 3
        ),
        "treatment_eligible_full_authority_by_action32_min_across_validation": (
            "full32_total", 11
        ),
        "treatment_eligible_full_authority_by_action32_min_per_seed": (
            "full32_min_seed", 2
        ),
        "treatment_partial_authority_rows_min_across_validation": (
            "partial_rows", 31
        ),
        "treatment_eligible_unlatched_clearance_ge_5cm_max": (
            "treatment_unlatched5", 4
        ),
        "pooled_option_active_pre_eligible_unlatched_clearance_ge_5cm_max": (
            "pooled_preeligible_unlatched5", 1
        ),
        "treatment_eligible_dropped_max": ("treatment_dropped", 1),
        "treatment_eligible_unsafe_force_max": ("treatment_unsafe", 1),
        "fixed_residual_component_and_l2_budget_violations": (
            "fixed_budget_violations", 1
        ),
        "any_new_abs_action_ge_0999": ("any_new_0999", True),
        "pre_eligibility_action_and_state_machine_parity_exact": (
            "preeligibility_exact", False
        ),
        "hand14_invariance_exact": ("hand_exact", False),
        "linear_authority_clock_and_action_algebra_exact": ("ramp_exact", False),
        "control_candidate39_route_exact": ("control_exact", False),
        "checkpoint_manifest_assignment_and_source_identity_exact": (
            "identity_exact", False
        ),
    }
    assert set(failures) == set(GATE_NAMES)
    for gate, (metric, bad_value) in failures.items():
        values = dict(passing)
        values[metric] = bad_value
        assert not _build_gate_results(values, thresholds)[gate]["pass"], gate


def test_ht_primary_is_success_minus_u5_on_pre_treatment_domain() -> None:
    treatment = torch.tensor([[True, False], [False, True]])
    eligible = torch.ones_like(treatment)
    success = torch.tensor([[True, False], [False, True]])
    u5 = torch.tensor([[False, True], [True, False]])
    utility = success.to(torch.float64) - u5.to(torch.float64)
    result = _ht_outcome(treatment, eligible, utility)
    assert result["domain_rows"] == 4
    assert result["treatment_rows"] == result["control_rows"] == 2
    assert result["treatment"] == 1.0
    assert result["control"] == -1.0
    assert result["delta"] == 2.0

    # A post-treatment authority mask must never enter this API/domain.
    authority_observed = treatment.clone()
    assert int(authority_observed.sum()) == 2
    assert result["domain_rows"] == int(eligible.sum())


def test_assignment_implementation_is_independent_balanced_and_complementary() -> None:
    for seed in SEEDS:
        a = _independent_treatment_mask(
            seed=seed, num_envs=64, replicate="a"
        )
        b = _independent_treatment_mask(
            seed=seed, num_envs=64, replicate="b"
        )
        assert int(a.sum()) == 32
        assert torch.equal(a, ~b)
        assert torch.equal(
            _independent_rank(seed=seed, num_envs=64),
            ramp.assignment_rank(seed=seed, num_envs=64),
        )


def test_full_compute_uses_ht_primary_pooled_paired_u5_and_all_32_gates() -> None:
    artifacts = _artifacts()
    validate_validation_artifacts(artifacts)
    report = compute_validation(artifacts)
    assert report["all_gates_pass"] is True
    assert report["decision"] == "advance_to_formal_screen"
    assert set(report["gates"]) == set(GATE_NAMES)
    assert len(report["gates"]) == 32
    assert report["authority_onset_or_full_used_as_causal_domain"] is False
    assert report["primary"]["per_seed_delta"] == {
        "341": 2.0,
        "342": 2.0,
    }
    assert report["primary"]["equal_seed_delta"] == 2.0
    equal_ht = report["equal_seed_horvitz_thompson"]
    assert equal_ht["success"]["delta"] == 1.0
    assert equal_ht["unlatched_clearance_ge_5cm"]["delta"] == -1.0
    paired = report["paired_common_eligibility_sensitivity"]
    assert paired["common_eligible_slots"] == 128
    assert paired["minimum_common_eligible_slots_per_seed"] == 64
    assert paired["aggregate_events"][
        "unlatched_clearance_ge_5cm"
    ]["paired_delta"] == -1.0
    assert report["treatment_counts"]["positive_authority_by_action16"] == {
        "total": 128,
        "per_seed": {"341": 64, "342": 64},
    }
    assert report["treatment_counts"]["full_authority_by_action32"] == {
        "total": 128,
        "per_seed": {"341": 64, "342": 64},
    }
    assert report["treatment_counts"]["partial_authority_rows"] == 3584
    assert _strict_json_bytes(report) == _strict_json_bytes(
        compute_validation(artifacts)
    )


def test_stable_transport_requires_15_rows_and_success_suffix_starts_after_terminal() -> None:
    base = _base_artifact(seed=341, replicate="a", num_envs=64)
    treatment_slot = int(base["episodes"]["treatment"].nonzero()[0])
    selected = base["steps"]["row_env_slot"] == treatment_slot
    ages = base["steps"]["row_eligible_age"][selected]

    fourteen = copy.deepcopy(base)
    fourteen["episodes"]["success"][treatment_slot] = False
    fourteen["episodes"]["failure"][treatment_slot] = True
    fourteen["episodes"]["dropped"][treatment_slot] = True
    fourteen_rows = fourteen["steps"]["row_env_slot"] == treatment_slot
    fourteen["steps"]["row_transition_true_clearance_m"][fourteen_rows] = 0.0
    fourteen["steps"]["row_transition_true_clearance_m"][
        fourteen_rows & (fourteen["steps"]["row_eligible_age"] < 14)
    ] = 0.20
    values = stable_transport_outcomes(fourteen)["stable_transition_value"]
    assert not bool((values[treatment_slot] >= 0.25).any())

    fifteen = copy.deepcopy(base)
    fifteen["episodes"]["success"][treatment_slot] = False
    fifteen["episodes"]["failure"][treatment_slot] = True
    fifteen["episodes"]["dropped"][treatment_slot] = True
    fifteen_rows = fifteen["steps"]["row_env_slot"] == treatment_slot
    fifteen["steps"]["row_transition_true_clearance_m"][fifteen_rows] = 0.0
    fifteen["steps"]["row_transition_true_clearance_m"][
        fifteen_rows & (fifteen["steps"]["row_eligible_age"] < 15)
    ] = 0.20
    stable = stable_transport_outcomes(fifteen)
    assert stable["stable_transition_value"][treatment_slot, 14] == 1.0
    assert bool(stable["stable_held_5cm_by_action96"][treatment_slot])

    suffix = copy.deepcopy(base)
    suffix_rows = suffix["steps"]["row_env_slot"] == treatment_slot
    suffix["steps"]["row_transition_true_clearance_m"][suffix_rows] = 0.0
    suffix["steps"]["row_transition_true_clearance_m"][
        suffix_rows & (suffix["steps"]["row_transition_done"])
    ] = 0.20
    suffix_values = stable_transport_outcomes(suffix)[
        "stable_transition_value"
    ][treatment_slot]
    terminal_age = int(ages[-1])
    assert terminal_age == 32
    assert not bool((suffix_values[: terminal_age + 1] != 0.0).any())
    assert bool((suffix_values[terminal_age + 1 :] == 1.0).all())


def test_authority_deadlines_are_inclusive_ages_0_to_15_and_0_to_31() -> None:
    episode = {
        "first_eligible_step": torch.tensor([10, 10, 10, 10, 10]),
        "first_positive_authority_step": torch.tensor([9, 10, 25, 26, -1]),
        "first_full_authority_step": torch.tensor([9, 10, 41, 42, -1]),
    }
    domain = torch.ones(5, dtype=torch.bool)
    assert _authority_by_age_count(
        episode,
        domain,
        clock_field="first_positive_authority_step",
        maximum_inclusive_age=15,
    ) == 2
    assert _authority_by_age_count(
        episode,
        domain,
        clock_field="first_full_authority_step",
        maximum_inclusive_age=31,
    ) == 2
    with pytest.raises(ValueError):
        _authority_by_age_count(
            episode,
            domain.to(torch.float32),
            clock_field="first_positive_authority_step",
            maximum_inclusive_age=15,
        )


def test_linear_float32_action_audit_reconstructs_scale_arm_and_hand_exactly() -> None:
    artifacts = _artifacts()
    audit = _action_clock_audit(artifacts)["aggregate"]
    assert audit["linear_authority_clock_and_action_algebra_exact"] is True
    assert audit["hand14_invariance_exact"] is True
    assert audit["control_candidate39_route_exact"] is True
    assert audit["treatment_partial_authority_rows"] == 3584

    mismatched = copy.deepcopy(artifacts)
    artifact = mismatched[(341, "a")]
    row = int(
        (
            artifact["steps"]["row_treatment"]
            & (artifact["steps"]["row_authority_scale"] > 0.0)
            & (artifact["steps"]["row_authority_scale"] < 1.0)
        ).nonzero()[0]
    )
    slot = int(artifact["steps"]["row_env_slot"][row])
    current = artifact["steps"]["row_requested_action"][row, 0]
    artifact["steps"]["row_requested_action"][row, 0] = torch.nextafter(
        current, torch.tensor(1.0, dtype=torch.float32)
    )
    artifact["episodes"]["treatment_arm_ramp_violations"][slot] = 1
    changed = _action_clock_audit(mismatched)["aggregate"]
    assert changed["treatment_arm_ramp_violations"] == 1
    assert changed["linear_authority_clock_and_action_algebra_exact"] is False

    unacknowledged = copy.deepcopy(artifacts)
    unacknowledged[(341, "a")]["steps"]["row_authority_scale"][row] += 0.01
    with pytest.raises(ValueError, match="online and trace"):
        _action_clock_audit(unacknowledged)


def test_atomic_no_clobber_publication() -> None:
    payload = {"decision": "advance_to_formal_screen", "network_updates": 0}
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "candidate41.json"
        publish_json_no_clobber(payload, output)
        assert json.loads(output.read_text(encoding="utf-8")) == payload
        with pytest.raises(FileExistsError):
            publish_json_no_clobber(payload, output)


def _attempt_payload(
    stem: Path,
    *,
    seed: int,
    replicate: str,
    num_envs: int,
    attempt: int,
    first_step: bool,
    collection_commit: str,
) -> dict:
    treatment = ramp.exact_balanced_treatment_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    payload = {
        "kind": ATTEMPT_RECEIPT_KIND,
        "status": "failed",
        "attempt": attempt,
        "first_env_step_invoked": first_step,
        "retry_permitted": not first_step,
        "seed": seed,
        "replicate": replicate,
        "num_envs": num_envs,
        "canonical_artifact_output": str(Path(f"{stem}.pt")),
        "canonical_report_output": str(Path(f"{stem}.json")),
        "collection_commit": collection_commit,
        "assignment_mask_sha256": ramp.assignment_mask_sha256(treatment),
        "source_manifest_sha256": "a" * 64,
        "checkpoint_manifest_sha256": "b" * 64,
        "argument_receipt": {
            "seed": seed,
            "replicate": replicate,
            "num_envs": num_envs,
            "output_stem": str(stem),
        },
        "error_type": "RuntimeError",
        "error": "synthetic failure",
        "traceback": "synthetic traceback",
    }
    assert set(payload) == ATTEMPT_RECEIPT_FIELDS
    return payload


def test_attempt_receipts_allow_bound_prestep_retry_and_reject_final_poststep() -> None:
    collection_commit = "c" * 40
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        smoke = _canonical_invocation_stem(root, 340, "b", 8)
        first = _attempt_payload(
            smoke,
            seed=340,
            replicate="b",
            num_envs=8,
            attempt=1,
            first_step=False,
            collection_commit=collection_commit,
        )
        smoke.parent.mkdir(parents=True, exist_ok=True)
        first_path = Path(f"{smoke}.failed_attempt_001.json")
        first_path.write_text(
            json.dumps(first), encoding="utf-8"
        )
        second = copy.deepcopy(first)
        second["attempt"] = 2
        Path(f"{smoke}.failed_attempt_002.json").write_text(
            json.dumps(second), encoding="utf-8"
        )
        summary = validate_attempt_receipts(
            root, collection_commit=collection_commit
        )
        assert len(summary) == 2
        assert all(row["retry_permitted"] for row in summary)
        first_sha = hashlib.sha256(first_path.read_bytes()).hexdigest()
        assert summary[0]["receipt_sha256"] == first_sha

        first["error"] = "synthetic pre-step failure with amended detail"
        first_path.write_text(json.dumps(first), encoding="utf-8")
        amended = validate_attempt_receipts(
            root, collection_commit=collection_commit
        )
        amended_sha = hashlib.sha256(first_path.read_bytes()).hexdigest()
        assert amended[0]["receipt_sha256"] == amended_sha
        assert amended_sha != first_sha

        final = _canonical_invocation_stem(root, 342, "a", 64)
        final.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{final}.pt").write_bytes(b"durable canonical artifact")
        Path(f"{final}.json").write_bytes(b"durable canonical report")
        rejected = _attempt_payload(
            final,
            seed=342,
            replicate="a",
            num_envs=64,
            attempt=1,
            first_step=True,
            collection_commit=collection_commit,
        )
        Path(f"{final}.failed_attempt_001.json").write_text(
            json.dumps(rejected), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="post-env.step"):
            validate_attempt_receipts(root, collection_commit=collection_commit)


def test_attempt_receipts_reject_gaps_and_schema_changes() -> None:
    collection_commit = "c" * 40
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        smoke = _canonical_invocation_stem(root, 340, "b", 8)
        value = _attempt_payload(
            smoke,
            seed=340,
            replicate="b",
            num_envs=8,
            attempt=2,
            first_step=False,
            collection_commit=collection_commit,
        )
        smoke.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{smoke}.failed_attempt_002.json").write_text(
            json.dumps(value), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="not contiguous"):
            validate_attempt_receipts(root, collection_commit=collection_commit)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        smoke = _canonical_invocation_stem(root, 340, "b", 8)
        value = _attempt_payload(
            smoke,
            seed=340,
            replicate="b",
            num_envs=8,
            attempt=1,
            first_step=False,
            collection_commit=collection_commit,
        )
        value.pop("traceback")
        smoke.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{smoke}.failed_attempt_001.json").write_text(
            json.dumps(value), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="schema"):
            validate_attempt_receipts(root, collection_commit=collection_commit)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
