#!/usr/bin/env python3
"""CPU-only tests for Candidate40's fail-closed development analyzer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import torch

import candidate40_verified_arm_episode as contract
import candidate40_verified_arm_handoff as verifier
from collect_online_close_ab import RUNTIME_ASSET_EXPECTED_SHA256
from analyze_candidate40_verified_arm_validation import (
    GATE_THRESHOLDS,
    NUM_ENVS,
    REPLICATES,
    RUN_ORDER,
    SEEDS,
    _action_clock_audit,
    _build_gate_results,
    _independent_rank,
    _independent_treatment_mask,
    _registered_validation_gates,
    _strict_json_bytes,
    _ht_outcome,
    _validate_collection_commit_provenance,
    _validate_registered_smoke,
    _validate_smoke_bindings,
    compute_validation,
    publish_json_no_clobber,
    stable_transport_outcomes,
    validate_validation_artifacts,
)


SHA = "1" * 64
ALT_SHA = "3" * 64
GIT_SHA = "2" * 40
TRACE_ROWS = 30
TRIGGER_STEP = contract.HANDOFF_HOLD_STEPS - 1
FIRST_ELIGIBLE_STEP = TRIGGER_STEP + 2


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
    sources = {"scripts/flashsac/mock_candidate40_source.py": SHA}
    assets = {"assets/mock_hammer.usd": SHA}
    runtime = {
        "python": "test",
        "torch": str(torch.__version__),
        "cuda": "test",
        "cudnn": 0,
        "cuda_device_index": 0,
        "cuda_device_name": "test",
        "cuda_device_capability": [0, 0],
        "isaac_sim": "test",
        "packages": {name: "test" for name in contract.RUNTIME_PACKAGE_FIELDS},
        "nvidia_smi_inventory": "test",
        "platform": "test",
        "seed": seed,
    }
    metadata = {
        **contract.REQUIRED_METADATA,
        "assignment_mask_sha256": verifier.assignment_mask_sha256(
            verifier.exact_balanced_treatment_mask(
                seed=seed, num_envs=NUM_ENVS, replicate=replicate
            )
        ),
        "seed": seed,
        "replicate": replicate,
        "num_envs": NUM_ENVS,
        "smoke_artifact_sha256": SHA,
        "smoke_report_sha256": ALT_SHA,
        "validation_plan_sha256": contract.sha256_file(
            Path(__file__).resolve().parents[2] / contract.VALIDATION_PLAN
        ),
        "source_manifest_sha256": contract.manifest_sha256(sources),
        "runtime_asset_manifest_sha256": contract.manifest_sha256(assets),
        "source_sha256": sources,
        "runtime_asset_sha256": assets,
        "flashsac_upstream_commit": "87edc9061150ae9e962dd84e6544e27a1554b3ab",
        "git": {
            "commit": GIT_SHA,
            "branch": contract.REQUIRED_BRANCH,
            "source_files_dirty": False,
            "flashsac_commit": contract.FLASHSAC_FORK_COMMIT,
            "flashsac_dirty": False,
        },
        "runtime": runtime,
    }
    assert set(metadata) == set(contract.METADATA_FIELDS)
    return metadata


def _encode_action(observation: torch.Tensor, action: torch.Tensor) -> None:
    observation[:, contract.TRANSITION_ARM_TOKEN_SLICE] = action[:, :16]
    observation[:, contract.TRANSITION_DISTAL_SLICE] = action[:, 16:]


def _base_observation(rows: int, *, latch: bool) -> torch.Tensor:
    observation = torch.zeros((rows, contract.OBSERVATION_DIM), dtype=torch.float32)
    observation[:, 92:97] = 0.4
    observation[:, 97:102] = 0.1
    observation[:, 103] = 0.8
    observation[:, 104] = 0.8
    observation[:, contract.PUBLIC_LATCH_INDEX] = float(latch)
    return observation


def _artifact(
    *,
    seed: int,
    replicate: str,
    eligible_mask: torch.Tensor | None = None,
    treatment_lift: bool = True,
    high_start_age: int = 15,
    high_frames: int = 15,
    successful: bool = False,
    trace_rows: int = TRACE_ROWS,
    first_eligible_step: int = FIRST_ELIGIBLE_STEP,
) -> dict[str, object]:
    treatment = verifier.exact_balanced_treatment_mask(
        seed=seed, num_envs=NUM_ENVS, replicate=replicate
    )
    if eligible_mask is None:
        eligible = torch.ones(NUM_ENVS, dtype=torch.bool)
    else:
        eligible = eligible_mask.clone().to(dtype=torch.bool)
        assert eligible.shape == (NUM_ENVS,)

    # Candidate39 fixed-residual audit: every option-active action from the
    # q.30/h4 trigger through eligibility is retained.  Ineligible slots never
    # trigger and reach the authored timeout horizon without audit rows.
    assert TRIGGER_STEP < first_eligible_step
    assert first_eligible_step - TRIGGER_STEP <= contract.WINDOW_STEPS
    eligible_slots = torch.arange(NUM_ENVS)[eligible]
    pre_env_parts = [
        eligible_slots.clone()
        for _ in range(TRIGGER_STEP, first_eligible_step)
    ]
    pre_env = torch.cat(pre_env_parts).to(torch.int64)
    pre_step = torch.cat(
        (
            *(
                torch.full(
                    (int(eligible_slots.numel()),), step, dtype=torch.int64
                )
                for step in range(TRIGGER_STEP, first_eligible_step)
            ),
        )
    )
    pre_age = pre_step - TRIGGER_STEP
    pre_rows = int(pre_env.numel())
    pre_observation = _base_observation(pre_rows, latch=False)
    baseline = torch.zeros((pre_rows, contract.ACTION_DIM), dtype=torch.float32)
    common = contract.expected_fixed_residual_action(baseline)
    pre_transition = pre_observation.clone()
    _encode_action(pre_transition, common)
    final_pre = pre_step == first_eligible_step - 1
    witness = final_pre
    pre_transition[witness, contract.PUBLIC_LATCH_INDEX] = 1.0
    # Preserve reset-before transition continuity throughout the pre-table.
    for slot in range(NUM_ENVS):
        rows = (pre_env == slot).nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() > 1:
            pre_observation[rows[1:]] = pre_transition[rows[:-1]]
    # The first trace observation must continue the final pre transition.
    first_trace_observation = _base_observation(NUM_ENVS, latch=True)
    for slot in range(NUM_ENVS):
        if bool(eligible[slot]):
            rows = (pre_env == slot).nonzero(as_tuple=False).squeeze(-1)
            first_trace_observation[slot] = pre_transition[rows[-1]]
            first_trace_observation[slot, 97:102] = 0.1
            first_trace_observation[slot, 103] = 0.8
            first_trace_observation[slot, 104] = 0.8
            first_trace_observation[slot, contract.PUBLIC_LATCH_INDEX] = 1.0

    pre_steps = {
        "pre_env_slot": pre_env,
        "pre_episode_step": pre_step,
        "pre_close_age": pre_age,
        "pre_option_active": torch.ones(pre_rows, dtype=torch.bool),
        "pre_public_latch_before": torch.zeros(pre_rows, dtype=torch.bool),
        "pre_residual_active": torch.ones(pre_rows, dtype=torch.bool),
        "pre_first_latch_witness": witness,
        "pre_observation": pre_observation,
        "pre_applied_delta": contract.expected_applied_delta()
        .expand(pre_rows, -1)
        .clone(),
        "pre_baseline_action": baseline,
        "pre_common_action": common,
        "pre_requested_action": common.clone(),
        "pre_transition_observation": pre_transition,
        "pre_task_action": common.clone(),
        "pre_transition_public_latch": witness,
        "pre_transition_grasped": witness.clone(),
        "pre_transition_true_clearance_m": torch.zeros(pre_rows),
        "pre_grasp_quality": witness.to(torch.float32) * 0.8,
        "pre_hold_quality": witness.to(torch.float32) * 0.8,
        "pre_max_force_n": torch.ones(pre_rows),
        "pre_transition_done": torch.zeros(pre_rows, dtype=torch.bool),
    }

    # Build a complete, step-major trace from first eligibility to terminal.
    trace_env_parts: list[torch.Tensor] = []
    trace_age_parts: list[torch.Tensor] = []
    observation_parts: list[torch.Tensor] = []
    transition_parts: list[torch.Tensor] = []
    common_parts: list[torch.Tensor] = []
    requested_parts: list[torch.Tensor] = []
    task_parts: list[torch.Tensor] = []
    before_parts: list[torch.Tensor] = []
    after_parts: list[torch.Tensor] = []
    enabled_parts: list[torch.Tensor] = []
    complete_parts: list[torch.Tensor] = []
    pre_enable_parts: list[torch.Tensor] = []
    clearance_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    current = first_trace_observation.clone()
    for age in range(trace_rows):
        rows = int(eligible_slots.numel())
        env = eligible_slots.clone()
        before = torch.full(
            (rows,), min(age, verifier.VERIFY_STEPS), dtype=torch.int64
        )
        after = torch.clamp(before + 1, max=verifier.VERIFY_STEPS)
        enabled = before == verifier.VERIFY_STEPS
        complete = before == verifier.VERIFY_STEPS - 1
        pre_enable = age < verifier.VERIFY_STEPS
        common_action = torch.zeros((rows, contract.ACTION_DIM))
        common_action[:, : contract.ARM_ACTION_DIM] = 0.1
        requested = common_action.clone()
        treated = treatment[env]
        requested[treated & (~enabled), : contract.ARM_ACTION_DIM] = 0.0
        transition = current[env].clone()
        _encode_action(transition, requested)
        transition[:, contract.PUBLIC_LATCH_INDEX] = 1.0
        transition[:, 97:102] = 0.1
        transition[:, 103] = 0.8
        transition[:, 104] = 0.8
        high = (
            treated
            & treatment_lift
            & (age >= high_start_age)
            & (age < high_start_age + high_frames)
        )
        clearance = high.to(torch.float32) * 0.20
        if successful and age == trace_rows - 1:
            # A strict-success mock includes an observed, held true-20cm
            # terminal transition; suffix utility still begins afterward.
            clearance.fill_(0.20)

        trace_env_parts.append(env)
        trace_age_parts.append(torch.full((rows,), age, dtype=torch.int64))
        observation_parts.append(current[env].clone())
        transition_parts.append(transition)
        common_parts.append(common_action)
        requested_parts.append(requested)
        task_parts.append(requested.clone())
        before_parts.append(before)
        after_parts.append(after)
        enabled_parts.append(enabled)
        complete_parts.append(complete)
        pre_enable_parts.append(torch.full((rows,), pre_enable, dtype=torch.bool))
        clearance_parts.append(clearance)
        done_parts.append(
            torch.full((rows,), age == trace_rows - 1, dtype=torch.bool)
        )
        current[env] = transition

    def joined(parts: list[torch.Tensor]) -> torch.Tensor:
        if not parts:
            raise AssertionError("mock trace unexpectedly empty")
        return torch.cat(parts)

    row_env = joined(trace_env_parts)
    row_age = joined(trace_age_parts)
    rows = int(row_env.numel())
    row_observation = joined(observation_parts)
    row_transition = joined(transition_parts)
    row_common = joined(common_parts)
    row_requested = joined(requested_parts)
    before = joined(before_parts)
    after = joined(after_parts)
    enabled = joined(enabled_parts)
    verification_complete = joined(complete_parts)
    pre_enable = joined(pre_enable_parts)
    clearance = joined(clearance_parts)
    done = joined(done_parts)
    row_treatment = treatment[row_env]
    previous_enabled = torch.zeros(rows, dtype=torch.bool)
    relock = torch.zeros(rows, dtype=torch.bool)
    reenable = torch.zeros(rows, dtype=torch.bool)
    for slot in eligible_slots.tolist():
        selected = row_env == slot
        slot_enabled = enabled[selected]
        previous = torch.cat((torch.zeros(1, dtype=torch.bool), slot_enabled[:-1]))
        previous_enabled[selected] = previous
        relock[selected] = previous & ~slot_enabled
        new = slot_enabled & ~previous
        ever_before = torch.cat(
            (torch.zeros(1, dtype=torch.bool), new.cumsum(0)[:-1] > 0)
        )
        reenable[selected] = new & ever_before

    steps = {
        "row_env_slot": row_env,
        "row_episode_step": first_eligible_step + row_age,
        "row_eligible_age": row_age,
        "row_observation": row_observation,
        "row_transition_observation": row_transition,
        "row_public_force_counters": torch.zeros((rows, 2)),
        "row_option_active": torch.ones(rows, dtype=torch.bool),
        "row_public_latch_before": torch.ones(rows, dtype=torch.bool),
        "row_stable_current": torch.ones(rows, dtype=torch.bool),
        "row_public_grasp_quality": torch.full((rows,), 0.8),
        "row_public_hold_quality": torch.full((rows,), 0.8),
        "row_public_max_force_strength": torch.full((rows,), 0.1),
        "row_stable_count_before": before,
        "row_stable_count_after": after,
        "row_common_action": row_common,
        "row_requested_action": row_requested,
        "row_task_action": row_requested.clone(),
        "row_arm_target_error": torch.zeros((rows, contract.ARM_ACTION_DIM)),
        "row_arm_joint_velocity": torch.zeros((rows, contract.ARM_ACTION_DIM)),
        "row_arm_target_error_abs_max": torch.zeros(rows),
        "row_arm_joint_speed_abs_max": torch.zeros(rows),
        "row_pre_action_true_clearance_m": torch.zeros(rows),
        "row_treatment": row_treatment,
        "row_activated": torch.ones(rows, dtype=torch.bool),
        "row_first_eligible": row_age == 0,
        "row_arm_enabled": enabled,
        "row_verification_complete": verification_complete,
        "row_relock": relock,
        "row_reenable": reenable,
        "row_pre_enable": pre_enable,
        "row_transition_public_latch": torch.ones(rows, dtype=torch.bool),
        "row_transition_grasped": torch.ones(rows, dtype=torch.bool),
        "row_transition_grasp_quality": torch.full((rows,), 0.8),
        "row_transition_hold_quality": torch.full((rows,), 0.8),
        "row_transition_max_force_n": torch.full((rows,), 10.0),
        "row_transition_object_lin_speed": torch.zeros(rows),
        "row_transition_object_ang_speed": torch.zeros(rows),
        "row_transition_true_clearance_m": clearance,
        "row_transition_done": done,
        "row_pre_release_launch": torch.zeros(rows, dtype=torch.bool),
    }

    episode_length = torch.where(
        eligible,
        torch.full((NUM_ENVS,), first_eligible_step + trace_rows, dtype=torch.int64),
        torch.full(
            (NUM_ENVS,), contract.MAX_EPISODE_ACTIONS, dtype=torch.int64
        ),
    )
    first_eligible = torch.where(
        eligible,
        torch.full((NUM_ENVS,), first_eligible_step, dtype=torch.int64),
        torch.full((NUM_ENVS,), -1, dtype=torch.int64),
    )
    first_latch = torch.where(
        eligible,
        torch.full(
            (NUM_ENVS,), first_eligible_step - 1, dtype=torch.int64
        ),
        torch.full((NUM_ENVS,), -1, dtype=torch.int64),
    )
    first_enabled = torch.where(
        eligible,
        torch.full(
            (NUM_ENVS,), first_eligible_step + verifier.VERIFY_STEPS, dtype=torch.int64
        ),
        torch.full((NUM_ENVS,), -1, dtype=torch.int64),
    )
    verification_step = torch.where(
        eligible,
        torch.full(
            (NUM_ENVS,),
            first_eligible_step + verifier.VERIFY_STEPS - 1,
            dtype=torch.int64,
        ),
        torch.full((NUM_ENVS,), -1, dtype=torch.int64),
    )
    success = eligible & torch.full((NUM_ENVS,), successful, dtype=torch.bool)
    lifted = success | (
        eligible & treatment
        if treatment_lift and high_frames >= 1 and high_start_age < trace_rows
        else torch.zeros(NUM_ENVS, dtype=torch.bool)
    )
    dropped = eligible & (~success)
    time_out = ~eligible
    max_clearance = lifted.to(torch.float32) * 0.20
    episodes = {
        "env_slot": torch.arange(NUM_ENVS, dtype=torch.int64),
        "treatment": treatment,
        "assignment_rank": verifier.assignment_rank(seed=seed, num_envs=NUM_ENVS),
        "fixed_z": contract.expected_fixed_z().expand(NUM_ENVS, -1).clone(),
        "triggered": eligible.clone(),
        "trigger_step": torch.where(
            eligible,
            torch.full((NUM_ENVS,), TRIGGER_STEP, dtype=torch.int64),
            torch.full((NUM_ENVS,), -1, dtype=torch.int64),
        ),
        "trigger_score": eligible.to(torch.float32) * 0.4,
        "first_latch_step": first_latch,
        "first_eligible_step": first_eligible,
        "verification_complete_step": verification_step,
        "first_arm_enabled_step": first_enabled,
        "first_relock_step": torch.full((NUM_ENVS,), -1, dtype=torch.int64),
        "first_reenable_step": torch.full((NUM_ENVS,), -1, dtype=torch.int64),
        "terminal_step": episode_length - 1,
        "intervention_steps": torch.full(
            (NUM_ENVS,), first_eligible_step - TRIGGER_STEP, dtype=torch.int64
        )
        * eligible.to(torch.int64),
        "trace_rows": eligible.to(torch.int64) * trace_rows,
        "stable_count_max": eligible.to(torch.int64) * verifier.VERIFY_STEPS,
        "arm_enable_count": eligible.to(torch.int64),
        "relock_count": torch.zeros(NUM_ENVS, dtype=torch.int64),
        "reenable_count": torch.zeros(NUM_ENVS, dtype=torch.int64),
        "episode_length": episode_length,
        "trajectory_max_force_n": torch.where(
            eligible, torch.full((NUM_ENVS,), 10.0), torch.ones(NUM_ENVS)
        ),
        "max_true_clearance_m": max_clearance,
        "first_eligible_clearance_m": torch.zeros(NUM_ENVS),
        "pre_release_max_clearance_m": torch.zeros(NUM_ENVS),
        "eligible": eligible,
        "pre_release_launch": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "latched_within_window": eligible.clone(),
        "success": success,
        "failure": dropped.clone(),
        "time_out": time_out,
        "dropped": dropped,
        "unsafe_force": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "unlatched_clearance_ge_5cm": torch.zeros(NUM_ENVS, dtype=torch.bool),
        "ever_grasped": eligible.clone(),
        "ever_clearance_ge_20cm": lifted,
        "option_active_pre_eligible_unlatched_clearance_ge_5cm": torch.zeros(
            NUM_ENVS, dtype=torch.bool
        ),
        "pre_eligibility_action_violations": torch.zeros(NUM_ENVS, dtype=torch.int64),
        "hand_invariance_violations": torch.zeros(NUM_ENVS, dtype=torch.int64),
        "treatment_arm_gate_violations": torch.zeros(NUM_ENVS, dtype=torch.int64),
        "control_route_violations": torch.zeros(NUM_ENVS, dtype=torch.int64),
        "task_action_reconstruction_violations": torch.zeros(
            NUM_ENVS, dtype=torch.int64
        ),
        "fixed_residual_budget_violations": torch.zeros(NUM_ENVS, dtype=torch.int64),
        "action_bound_violations": torch.zeros(NUM_ENVS, dtype=torch.int64),
        "treatment_pre_enable_nonzero_arm_rows": torch.zeros(
            NUM_ENVS, dtype=torch.int64
        ),
        "new_abs_action_ge_0999": torch.zeros(NUM_ENVS, dtype=torch.bool),
    }
    return contract.build_artifact(
        _metadata(seed, replicate),
        episodes,
        pre_steps,
        steps,
        require_sealed_plan=False,
    )


def _artifacts(
    *,
    eligibility: dict[tuple[int, str], torch.Tensor] | None = None,
) -> dict[tuple[int, str], dict[str, object]]:
    return {
        (seed, replicate): _artifact(
            seed=seed,
            replicate=replicate,
            eligible_mask=None if eligibility is None else eligibility[(seed, replicate)],
            successful=True,
        )
        for seed in SEEDS
        for replicate in REPLICATES
    }


def test_positive_report_uses_pre_treatment_ht_and_registered_utility() -> None:
    artifacts = _artifacts()
    report = compute_validation(artifacts)
    assert report["all_gates_pass"] is True
    assert report["decision"] == "advance_to_formal_screen"
    assert report["arm_enabled_was_used_as_causal_domain"] is False
    assert report["primary"]["per_seed_delta"] == {
        "335": 1.0 / 128.0,
        "336": 1.0 / 128.0,
    }
    assert report["primary"]["equal_seed_delta"] == 1.0 / 128.0
    assert report["per_seed_horvitz_thompson"]["335"][
        "stable_transport_restricted_mean"
    ]["domain_rows"] == 128
    assert report["treatment_counts"]["first_arm_releases_by_action32"][
        "total"
    ] == 128
    assert set(report["gates"]) == set(GATE_THRESHOLDS)
    assert _strict_json_bytes(report) == _strict_json_bytes(
        compute_validation(artifacts)
    )


def test_stable_window_spikes_and_terminal_suffix_semantics() -> None:
    one = _artifact(
        seed=335,
        replicate="a",
        high_start_age=15,
        high_frames=1,
        trace_rows=16,
    )
    fourteen = _artifact(
        seed=335,
        replicate="a",
        high_start_age=15,
        high_frames=14,
        trace_rows=29,
    )
    fifteen = _artifact(
        seed=335,
        replicate="a",
        high_start_age=15,
        high_frames=15,
        trace_rows=30,
    )
    treatment = one["episodes"]["treatment"]
    assert not bool(
        (stable_transport_outcomes(one)["stable_transport_restricted_mean"][treatment] > 0).any()
    )
    assert not bool(
        (stable_transport_outcomes(fourteen)["stable_transport_restricted_mean"][treatment] > 0).any()
    )
    assert torch.equal(
        stable_transport_outcomes(fifteen)["stable_transport_restricted_mean"][treatment],
        torch.full((int(treatment.sum()),), 1.0 / 128.0, dtype=torch.float64),
    )

    success = _artifact(
        seed=335,
        replicate="a",
        treatment_lift=False,
        high_frames=0,
        successful=True,
        trace_rows=20,
    )
    success_values = stable_transport_outcomes(success)["stable_transition_value"]
    # terminal age is 19: observed positions 0..19 retain zero; only the
    # unobserved suffix beginning at age 20 receives absorbing success credit.
    assert not bool((success_values[:, :20] != 0).any())
    assert bool((success_values[:, 20:] == 1).all())
    timeout = _artifact(
        seed=335,
        replicate="a",
        treatment_lift=False,
        high_frames=0,
        successful=False,
        trace_rows=20,
    )
    assert not bool(
        (stable_transport_outcomes(timeout)["stable_transition_value"] != 0).any()
    )


def test_ht_uses_actual_eligibility_not_arm_enablement_mediator() -> None:
    treatment = torch.tensor([[True, False], [False, True]])
    eligibility = torch.ones_like(treatment)
    outcome = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    result = _ht_outcome(treatment, eligibility, outcome)
    assert result["domain_rows"] == 4
    assert result["treatment_rows"] == result["control_rows"] == 2
    assert result["delta"] == -1.0
    # A hypothetical post-treatment release mask could select only treatment,
    # but it is deliberately absent from the estimator API/domain.
    post_treatment_release = treatment.clone()
    assert int(post_treatment_release.sum()) == 2
    assert result["domain_rows"] == int(eligibility.sum())


def test_missing_paired_support_rejects_deterministically_not_typeerror() -> None:
    eligibility: dict[tuple[int, str], torch.Tensor] = {}
    first = torch.zeros(NUM_ENVS, dtype=torch.bool)
    first[: NUM_ENVS // 2] = True
    for seed in SEEDS:
        eligibility[(seed, "a")] = first
        eligibility[(seed, "b")] = ~first
    report = compute_validation(_artifacts(eligibility=eligibility))
    assert report["all_gates_pass"] is False
    assert report["decision"] == "reject_candidate40"
    assert report["paired_common_eligibility_sensitivity"][
        "common_eligible_slots"
    ] == 0
    role = report["paired_common_eligibility_sensitivity"]["role"]
    assert "paired unlatched-5cm acceptance gate" in role
    assert "never a replacement for the HT primary" in role
    assert report["gates"][
        "paired_common_eligibility_unlatched_clearance_ge_5cm_delta_max"
    ] == {"value": 1.0, "threshold": 0.0, "comparison": "<=", "pass": False}


def test_discordant_gpu_eligibility_and_clocks_use_actual_ht_and_common_intersection() -> None:
    a_mask = torch.zeros(NUM_ENVS, dtype=torch.bool)
    b_mask = torch.zeros(NUM_ENVS, dtype=torch.bool)
    a_mask[:48] = True
    b_mask[16:] = True
    artifacts: dict[tuple[int, str], dict[str, object]] = {}
    for seed in SEEDS:
        artifacts[(seed, "a")] = _artifact(
            seed=seed,
            replicate="a",
            eligible_mask=a_mask,
            first_eligible_step=TRIGGER_STEP + 2,
            successful=True,
        )
        artifacts[(seed, "b")] = _artifact(
            seed=seed,
            replicate="b",
            eligible_mask=b_mask,
            first_eligible_step=TRIGGER_STEP + 3,
            successful=True,
        )
    validate_validation_artifacts(artifacts)
    report = compute_validation(artifacts)
    assert report["per_seed_horvitz_thompson"]["335"][
        "stable_transport_restricted_mean"
    ]["domain_rows"] == 96
    paired = report["paired_common_eligibility_sensitivity"]
    assert paired["common_eligible_slots"] == 64
    for seed in SEEDS:
        row = paired["per_seed"][str(seed)]
        assert row["eligible_in_both_replicates"] == 32
        assert row["eligibility_mask_discordant_slots"] == 32
    assert paired["eligibility_masks_and_clocks_required_bit_exact"] is False


def test_assignment_complement_provenance_and_corruption_fail_closed() -> None:
    artifacts = _artifacts()
    validate_validation_artifacts(artifacts)
    for seed in SEEDS:
        a = _independent_treatment_mask(seed=seed, num_envs=NUM_ENVS, replicate="a")
        b = _independent_treatment_mask(seed=seed, num_envs=NUM_ENVS, replicate="b")
        assert int(a.sum()) == NUM_ENVS // 2
        assert torch.equal(a, ~b)
        assert torch.equal(
            _independent_rank(seed=seed, num_envs=NUM_ENVS),
            verifier.assignment_rank(seed=seed, num_envs=NUM_ENVS),
        )

    bad = copy.deepcopy(artifacts)
    bad[(335, "a")]["episodes"]["assignment_rank"][[0, 1]] = bad[(335, "a")][
        "episodes"
    ]["assignment_rank"][[1, 0]]
    _expect(ValueError, validate_validation_artifacts, bad)

    bad = copy.deepcopy(artifacts)
    bad[(335, "a")]["episodes"]["fixed_z"][0, 0] += 1.0
    _expect(ValueError, validate_validation_artifacts, bad)

    bad = copy.deepcopy(artifacts)
    bad[(335, "a")]["steps"]["row_stable_count_before"][0] = 1
    _expect(ValueError, validate_validation_artifacts, bad)

    # Preserve each seed's A/B local identity but alter seed 336 lineage.  The
    # joint analyzer identity check must still reject it.
    bad = copy.deepcopy(artifacts)
    for replicate in REPLICATES:
        meta = bad[(336, replicate)]["metadata"]
        meta["source_sha256"] = {
            "scripts/flashsac/mock_candidate40_source.py": ALT_SHA
        }
        meta["source_manifest_sha256"] = contract.manifest_sha256(
            meta["source_sha256"]
        )
    _expect(ValueError, validate_validation_artifacts, bad)


def test_smoke_receipts_are_canonical_and_bind_every_development_run() -> None:
    artifacts = _artifacts()
    receipt = {"artifact_sha256": SHA, "report_sha256": ALT_SHA}
    _validate_smoke_bindings(artifacts, receipt)

    bad_receipt = {"artifact_sha256": ALT_SHA, "report_sha256": ALT_SHA}
    _expect(ValueError, _validate_smoke_bindings, artifacts, bad_receipt)
    _expect(
        ValueError,
        _validate_smoke_bindings,
        artifacts,
        {"artifact_sha256": "not-a-sha", "report_sha256": ALT_SHA},
    )

    plan = {
        "execution": {
            "non_evidence_smoke": {
                "seed": 334,
                "replicate": "b",
                "num_envs": 8,
                "output": "logs/flashsac/pick_tool/not_the_registered_smoke/trial",
            }
        }
    }
    with tempfile.TemporaryDirectory() as directory:
        _expect(
            ValueError,
            _validate_registered_smoke,
            plan,
            repository_root=Path(directory),
        )


def test_collection_commit_must_descend_from_seal_with_unchanged_sources() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repository = Path(directory)

        def git(*arguments: str) -> str:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return result.stdout.decode("utf-8").strip()

        git("init", "--quiet")
        git("config", "user.name", "Candidate40 Test")
        git("config", "user.email", "candidate40@example.invalid")
        source = repository / "sealed_source.py"
        source.write_text("SEALED = True\n", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        runtime_source = repository / "runtime_source.py"
        runtime_source.write_text("RUNTIME = 'sealed'\n", encoding="utf-8")
        runtime_digest = hashlib.sha256(runtime_source.read_bytes()).hexdigest()
        git("add", "sealed_source.py", "runtime_source.py")
        git("commit", "--quiet", "-m", "implementation")
        implementation_commit = git("rev-parse", "HEAD")

        (repository / "seal_receipt.txt").write_text("sealed\n", encoding="utf-8")
        git("add", "seal_receipt.txt")
        git("commit", "--quiet", "-m", "seal receipt")
        collection_commit = git("rev-parse", "HEAD")
        artifacts = {
            key: {
                "metadata": {
                    "git": {"commit": collection_commit},
                    "source_sha256": {
                        "sealed_source.py": digest,
                        "runtime_source.py": runtime_digest,
                    },
                    "runtime_asset_sha256": dict(
                        RUNTIME_ASSET_EXPECTED_SHA256
                    ),
                }
            }
            for key in RUN_ORDER
        }
        implementation = {
            "implementation_commit": implementation_commit,
            "source_sha256": {"sealed_source.py": digest},
        }
        result = _validate_collection_commit_provenance(
            artifacts, implementation, repository_root=repository
        )
        assert result["implementation_commit_is_ancestor"] is True
        assert result["collection_commit"] == collection_commit

        source.write_text("SEALED = False\n", encoding="utf-8")
        git("add", "sealed_source.py")
        git("commit", "--quiet", "-m", "changed sealed source")
        changed_commit = git("rev-parse", "HEAD")
        changed = copy.deepcopy(artifacts)
        for artifact in changed.values():
            artifact["metadata"]["git"]["commit"] = changed_commit
        _expect(
            ValueError,
            _validate_collection_commit_provenance,
            changed,
            implementation,
            repository_root=repository,
        )

        git("checkout", "--quiet", collection_commit)
        runtime_source.write_text("RUNTIME = 'changed'\n", encoding="utf-8")
        changed_runtime_digest = hashlib.sha256(
            runtime_source.read_bytes()
        ).hexdigest()
        git("add", "runtime_source.py")
        git("commit", "--quiet", "-m", "changed unsealed runtime source")
        changed_runtime_commit = git("rev-parse", "HEAD")
        changed_runtime = copy.deepcopy(artifacts)
        for artifact in changed_runtime.values():
            artifact["metadata"]["git"]["commit"] = changed_runtime_commit
            artifact["metadata"]["source_sha256"][
                "runtime_source.py"
            ] = changed_runtime_digest
        _expect(
            ValueError,
            _validate_collection_commit_provenance,
            changed_runtime,
            implementation,
            repository_root=repository,
        )

        empty_tree = git("hash-object", "-t", "tree", "--stdin")
        unrelated_commit = git("commit-tree", empty_tree, "-m", "unrelated")
        unrelated = copy.deepcopy(artifacts)
        for artifact in unrelated.values():
            artifact["metadata"]["git"]["commit"] = unrelated_commit
        _expect(
            ValueError,
            _validate_collection_commit_provenance,
            unrelated,
            implementation,
            repository_root=repository,
        )


def test_action_audit_includes_pre_eligibility_saturation() -> None:
    artifacts = _artifacts()
    artifact = artifacts[(335, "a")]
    artifact["pre_steps"]["pre_baseline_action"][0, 0] = 0.998
    artifact["pre_steps"]["pre_requested_action"][0, 0] = 0.999
    artifact["episodes"]["new_abs_action_ge_0999"][0] = True
    audit = _action_clock_audit(artifacts)
    assert audit["aggregate"]["any_new_abs_action_ge_0999"] is True


def _passing_metrics() -> dict[str, object]:
    return {
        "any_new_0999": False,
        "identity_exact": True,
        "ever_grasped_delta": -0.03,
        "never_lift_delta": 0.0,
        "success_delta": -0.03,
        "true20_delta": -0.03,
        "unlatched5_delta": 0.0,
        "control_exact": True,
        "eligible_total": 32,
        "eligible_min_seed": 12,
        "eligible_min_seed_arm": 8,
        "fixed_budget_violations": 0,
        "hand_exact": True,
        "paired_unlatched5_delta": 0.0,
        "paired_common_total": 16,
        "paired_common_min_seed": 6,
        "minimum_primary_seed_delta": -0.1,
        "pooled_preeligible_unlatched5": 0,
        "positive_primary_seeds": 1,
        "preeligibility_exact": True,
        "primary_delta": 1e-12,
        "treatment_dropped": 0,
        "treatment_release32_total": 16,
        "treatment_release32_min_seed": 4,
        "treatment_unlatched5": 0,
        "treatment_pre_release_launch": 0,
        "treatment_stable5_total": 8,
        "treatment_stable5_min_seed": 2,
        "treatment_unsafe": 0,
        "preenable_nonzero_arm_rows": 0,
        "clock_exact": True,
    }


def test_registered_gates_and_every_boundary_are_executable_and_exact() -> None:
    assert _registered_validation_gates() == GATE_THRESHOLDS
    passing = _passing_metrics()
    assert all(row["pass"] for row in _build_gate_results(passing).values())
    failures = {
        "any_new_abs_action_ge_0999": ("any_new_0999", True),
        "checkpoint_manifest_assignment_and_source_identity_exact": ("identity_exact", False),
        "conditional_ever_grasped_delta_floor": ("ever_grasped_delta", -0.030001),
        "conditional_never_stable_held_5cm_by_action96_delta_max": ("never_lift_delta", 1e-12),
        "conditional_success_delta_floor": ("success_delta", -0.030001),
        "conditional_true_clearance_ge_20cm_delta_floor": ("true20_delta", -0.030001),
        "conditional_unlatched_clearance_ge_5cm_delta_max": ("unlatched5_delta", 1e-12),
        "control_candidate39_route_exact": ("control_exact", False),
        "eligible_opportunity_slots_min_across_validation": ("eligible_total", 31),
        "eligible_opportunity_slots_min_per_seed": ("eligible_min_seed", 11),
        "eligible_opportunity_slots_min_per_seed_arm": ("eligible_min_seed_arm", 7),
        "fixed_residual_component_and_l2_budget_violations": ("fixed_budget_violations", 1),
        "hand14_invariance_exact": ("hand_exact", False),
        "paired_common_eligibility_unlatched_clearance_ge_5cm_delta_max": ("paired_unlatched5_delta", 1e-12),
        "paired_common_eligible_slots_min_across_validation": ("paired_common_total", 15),
        "paired_common_eligible_slots_min_per_seed": ("paired_common_min_seed", 5),
        "per_seed_primary_delta_floor": ("minimum_primary_seed_delta", -0.100001),
        "pooled_option_active_pre_eligible_unlatched_clearance_ge_5cm_max": ("pooled_preeligible_unlatched5", 1),
        "positive_primary_delta_seeds_min": ("positive_primary_seeds", 0),
        "pre_eligibility_action_and_state_machine_parity_exact": ("preeligibility_exact", False),
        "primary_conditional_stable_transport_restricted_mean_delta_min_exclusive": ("primary_delta", 0.0),
        "treatment_eligible_dropped_max": ("treatment_dropped", 1),
        "treatment_eligible_first_arm_releases_by_action32_min_across_validation": ("treatment_release32_total", 15),
        "treatment_eligible_first_arm_releases_by_action32_min_per_seed": ("treatment_release32_min_seed", 3),
        "treatment_eligible_post_latch_unlatched_clearance_ge_5cm_max": ("treatment_unlatched5", 1),
        "treatment_eligible_pre_release_launch_max": ("treatment_pre_release_launch", 1),
        "treatment_eligible_stable_held_5cm_by_action96_min_across_validation": ("treatment_stable5_total", 7),
        "treatment_eligible_stable_held_5cm_by_action96_min_per_seed": ("treatment_stable5_min_seed", 1),
        "treatment_eligible_unsafe_force_max": ("treatment_unsafe", 1),
        "treatment_pre_enable_nonzero_arm_rows_max": ("preenable_nonzero_arm_rows", 1),
        "verified_arm_clock_gate_and_live_relock_exact": ("clock_exact", False),
    }
    assert set(failures) == set(GATE_THRESHOLDS)
    for gate, (metric, bad_value) in failures.items():
        row = dict(passing)
        row[metric] = bad_value
        assert _build_gate_results(row)[gate]["pass"] is False, gate


def test_atomic_no_clobber_publication() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "candidate40.json"
        payload = {"decision": "advance_to_formal_screen", "network_updates": 0}
        publish_json_no_clobber(payload, output)
        assert json.loads(output.read_text(encoding="utf-8")) == payload
        _expect(FileExistsError, publish_json_no_clobber, payload, output)


if __name__ == "__main__":
    test_positive_report_uses_pre_treatment_ht_and_registered_utility()
    test_stable_window_spikes_and_terminal_suffix_semantics()
    test_ht_uses_actual_eligibility_not_arm_enablement_mediator()
    test_missing_paired_support_rejects_deterministically_not_typeerror()
    test_assignment_complement_provenance_and_corruption_fail_closed()
    test_registered_gates_and_every_boundary_are_executable_and_exact()
    test_atomic_no_clobber_publication()
    print("analyze_candidate40_verified_arm_validation_test: PASS")
