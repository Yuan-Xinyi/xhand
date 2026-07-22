#!/usr/bin/env python3
"""CPU corruption tests and synthetic fixtures for Candidate43 artifacts."""

from __future__ import annotations

import copy
import ast
import inspect
import json
from pathlib import Path
import tempfile
from unittest import mock

import torch

import candidate42_public_arm_ramp as c42_ramp
import candidate42_public_arm_ramp_episode as c42_contract
import candidate43_short_arm_ramp as ramp
import candidate43_short_arm_ramp_episode as contract
import analyze_candidate43_short_arm_ramp_validation as analyzer


def _raises(error: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _metadata(*, seed: int, replicate: str, num_envs: int) -> dict:
    sources = {"scripts/flashsac/fake_candidate43.py": "a" * 64}
    assets = {"/tmp/fake_candidate43.usd": "b" * 64}
    plan = contract.validate_sealed_plan(require_sealed=False)
    return {
        **contract.REQUIRED_METADATA,
        "seed": seed,
        "replicate": replicate,
        "num_envs": num_envs,
        "assignment_mask_sha256": ramp.assignment_mask_sha256(
            ramp.exact_balanced_treatment_mask(
                seed=seed, num_envs=num_envs, replicate=replicate
            )
        ),
        "implementation_commit": "1" * 40,
        "preregistration_plan_sha256": plan["preregistration_receipt"].get(
            "sha256"
        ),
        "validation_plan_sha256": contract.sha256_file(
            Path(__file__).resolve().parents[2] / contract.VALIDATION_PLAN
        ),
        "smoke_artifact_sha256": None if num_envs == 8 else "d" * 64,
        "smoke_report_sha256": None if num_envs == 8 else "e" * 64,
        "source_manifest_sha256": contract.manifest_sha256(sources),
        "runtime_asset_manifest_sha256": contract.manifest_sha256(assets),
        "source_sha256": sources,
        "runtime_asset_sha256": assets,
        "flashsac_upstream_commit": (
            "87edc9061150ae9e962dd84e6544e27a1554b3ab"
        ),
        "git": {
            "commit": "c" * 40,
            "branch": contract.REQUIRED_BRANCH,
            "source_files_dirty": False,
            "flashsac_commit": contract.FLASHSAC_FORK_COMMIT,
            "flashsac_dirty": False,
        },
        "runtime": {
            "python": "3.11",
            "torch": "2",
            "cuda": "12",
            "cudnn": 9000,
            "cuda_device_index": 0,
            "cuda_device_name": "fake",
            "cuda_device_capability": [8, 9],
            "isaac_sim": "5",
            "packages": {
                name: "1" for name in contract.RUNTIME_PACKAGE_FIELDS
            },
            "nvidia_smi_inventory": "fake",
            "platform": "linux",
            "seed": seed,
        },
    }


def _sealed_plan_fixture() -> dict:
    plan = copy.deepcopy(
        contract.validate_sealed_plan(require_sealed=False)
    )
    plan["status"] = contract.SEALED_PLAN_STATUS
    plan["preregistration_receipt"]["commit"] = (
        contract.PREREGISTRATION_COMMIT
    )
    plan["preregistration_receipt"]["sha256"] = (
        contract.PRISTINE_PLAN_SHA256
    )
    required = plan["implementation_seal_requirements"][
        "required_simulation_free_tests"
    ]
    plan["implementation_seal"] = {
        "status": "complete_without_simulator_evidence",
        "implementation_commit": "3" * 40,
        "source_sha256": {
            relative: f"{index + 1:064x}"
            for index, relative in enumerate(contract.IMPLEMENTATION_SOURCE_FILES)
        },
        "simulator_evidence_before_source_seal": False,
        "simulation_free_tests": {
            "status": "passed",
            "passed": 1,
            "failed": 0,
            "required_contracts_covered": copy.deepcopy(required),
        },
        "static_audit": {
            "status": "passed",
            "blockers": 0,
            "simulator_invocations": 0,
            "runtime_assets": 8,
            "runtime_source_files": 1,
        },
    }
    plan["collection_seal"] = {"tag": contract.COLLECTION_SEAL_TAG}
    return plan


def _observation(
    rows: int,
    *,
    stable: bool,
    task_action: torch.Tensor | None = None,
) -> torch.Tensor:
    observation = torch.zeros(
        (rows, contract.OBSERVATION_DIM), dtype=torch.float32
    )
    if stable:
        observation[:, ramp.PUBLIC_LATCH_INDEX] = 1.0
        observation[
            :, ramp.FORCE_STRENGTH_START : ramp.FORCE_STRENGTH_STOP
        ] = 0.2
        observation[:, ramp.STRICT_WRAP_QUALITY_INDEX] = 0.6
        observation[:, ramp.HOLD_QUALITY_INDEX] = 0.7
    if task_action is not None:
        observation[:, contract.TRANSITION_ARM_TOKEN_SLICE] = task_action[:, :16]
        observation[:, contract.TRANSITION_DISTAL_SLICE] = task_action[:, 16:]
    return observation


def _cat(storage: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.cat(chunks, dim=0) for name, chunks in storage.items()}


def _payload(
    *,
    seed: int = 352,
    replicate: str = "b",
    num_envs: int = 8,
    include_relock: bool = True,
) -> tuple[
    dict,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    treatment = ramp.exact_balanced_treatment_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    assignment_rank = ramp.assignment_rank(seed=seed, num_envs=num_envs)
    env_slot = torch.arange(num_envs, dtype=torch.long)
    baseline = torch.zeros((num_envs, contract.ACTION_DIM), dtype=torch.float32)
    common_pre = contract.expected_fixed_residual_action(baseline)
    fixed_delta = contract.expected_applied_delta().expand(num_envs, -1).clone()

    pre_storage = {name: [] for name in contract.PRE_STEP_FIELDS}
    observation = _observation(num_envs, stable=False)
    observation[:, contract.PROXIMITY_SLICE] = 0.4
    observation[:, contract.THUMB_PROXIMITY_INDEX] = 0.4
    trace_initial: torch.Tensor | None = None
    for index in range(2):
        episode_step = index + contract.HANDOFF_HOLD_STEPS - 1
        task_action = common_pre.clone()
        transition = _observation(
            num_envs, stable=index == 1, task_action=task_action
        )
        values = {
            "pre_env_slot": env_slot,
            "pre_episode_step": torch.full(
                (num_envs,), episode_step, dtype=torch.long
            ),
            "pre_close_age": torch.full((num_envs,), index, dtype=torch.long),
            "pre_option_active": torch.ones(num_envs, dtype=torch.bool),
            "pre_public_latch_before": torch.zeros(num_envs, dtype=torch.bool),
            "pre_residual_active": torch.ones(num_envs, dtype=torch.bool),
            "pre_trigger_witness": torch.full(
                (num_envs,), index == 0, dtype=torch.bool
            ),
            "pre_first_latch_witness": torch.full(
                (num_envs,), index == 1, dtype=torch.bool
            ),
            "pre_observation": observation,
            "pre_applied_delta": fixed_delta,
            "pre_baseline_action": baseline,
            "pre_common_action": common_pre,
            "pre_requested_action": common_pre,
            "pre_transition_observation": transition,
            "pre_task_action": task_action,
            "pre_transition_public_latch": torch.full(
                (num_envs,), index == 1, dtype=torch.bool
            ),
            "pre_transition_grasped": torch.full(
                (num_envs,), index == 1, dtype=torch.bool
            ),
            "pre_transition_true_clearance_m": torch.zeros(num_envs),
            "pre_grasp_quality": torch.full(
                (num_envs,), 0.6 if index == 1 else 0.0
            ),
            "pre_hold_quality": torch.full(
                (num_envs,), 0.7 if index == 1 else 0.0
            ),
            "pre_max_force_n": torch.full(
                (num_envs,), 10.0 if index == 1 else 0.0
            ),
            "pre_transition_done": torch.zeros(num_envs, dtype=torch.bool),
        }
        for name in contract.PRE_STEP_FIELDS:
            pre_storage[name].append(values[name].clone())
        observation = transition
        trace_initial = transition
    assert trace_initial is not None

    # Nine stable actions ramp 0..1, one same-action relock, then another
    # nine stable actions reramp 0..1.  This exercises both D8 segments.
    stable_schedule = [True] * 19
    if include_relock:
        stable_schedule[9] = False
    trace_storage = {name: [] for name in contract.STEP_FIELDS}
    state = ramp.initial_public_arm_ramp_state(num_envs, device="cpu")
    observation = trace_initial
    current_clearance = torch.zeros(num_envs, dtype=torch.float32)
    for age, stable in enumerate(stable_schedule):
        assert bool(
            (observation[:, ramp.PUBLIC_LATCH_INDEX] == 1).all()
        ) == stable
        common = torch.full(
            (num_envs, contract.ACTION_DIM), 0.1, dtype=torch.float32
        )
        common[:, : contract.ARM_ACTION_DIM] = 0.2
        counters = torch.zeros((num_envs, 2), dtype=torch.float32)
        routed = ramp.apply_public_arm_ramp(
            common,
            observation,
            counters,
            torch.ones(num_envs, dtype=torch.bool),
            torch.ones(num_envs, dtype=torch.bool),
            treatment,
            state,
            reset_mask=torch.zeros(num_envs, dtype=torch.bool),
        )
        state = routed.next_state
        next_stable = (
            stable_schedule[age + 1] if age + 1 < len(stable_schedule) else True
        )
        task_action = routed.action.clone()
        transition = _observation(
            num_envs, stable=next_stable, task_action=task_action
        )
        transition_latch = transition[:, ramp.PUBLIC_LATCH_INDEX] == 1
        transition_clearance = torch.full(
            (num_envs,), 0.2 if age == len(stable_schedule) - 1 else 0.001
        )
        target_error = torch.full((num_envs, contract.ARM_ACTION_DIM), 0.1)
        velocity = torch.full((num_envs, contract.ARM_ACTION_DIM), -0.02)
        episode_step = age + contract.HANDOFF_HOLD_STEPS + 1
        values = {
            "row_env_slot": env_slot,
            "row_episode_step": torch.full(
                (num_envs,), episode_step, dtype=torch.long
            ),
            "row_eligible_age": torch.full((num_envs,), age, dtype=torch.long),
            "row_observation": observation,
            "row_transition_observation": transition,
            "row_public_force_counters": counters,
            "row_option_active": torch.ones(num_envs, dtype=torch.bool),
            "row_public_latch_before": observation[:, ramp.PUBLIC_LATCH_INDEX] == 1,
            "row_stable_current": routed.stable_current,
            "row_public_grasp_quality": routed.public_grasp_quality,
            "row_public_hold_quality": routed.public_hold_quality,
            "row_public_max_force_strength": routed.public_max_force_strength,
            "row_stable_count_before": routed.stable_count_before,
            "row_stable_count_after": routed.stable_count_after,
            "row_authority_scale": routed.authority_scale,
            "row_common_action": common,
            "row_requested_action": routed.action,
            "row_task_action": task_action,
            "row_arm_target_error": target_error,
            "row_arm_joint_velocity": velocity,
            "row_arm_target_error_abs_max": target_error.abs().max(dim=-1).values,
            "row_arm_joint_speed_abs_max": velocity.abs().max(dim=-1).values,
            "row_pre_action_true_clearance_m": current_clearance,
            "row_treatment": treatment,
            "row_activated": routed.activated_this_action,
            "row_first_eligible": routed.first_eligible,
            "row_relock": routed.relock,
            "row_reramp": routed.reramp,
            "row_transition_public_latch": transition_latch,
            "row_transition_grasped": transition_latch,
            "row_transition_grasp_quality": torch.where(
                transition_latch,
                torch.full((num_envs,), 0.6),
                torch.zeros(num_envs),
            ),
            "row_transition_hold_quality": torch.where(
                transition_latch,
                torch.full((num_envs,), 0.7),
                torch.zeros(num_envs),
            ),
            "row_transition_max_force_n": torch.where(
                transition_latch,
                torch.full((num_envs,), 10.0),
                torch.zeros(num_envs),
            ),
            "row_transition_object_lin_speed": torch.full((num_envs,), 0.01),
            "row_transition_object_ang_speed": torch.full((num_envs,), 0.02),
            "row_transition_true_clearance_m": transition_clearance,
            "row_transition_done": torch.full(
                (num_envs,), age == len(stable_schedule) - 1, dtype=torch.bool
            ),
        }
        for name in contract.STEP_FIELDS:
            trace_storage[name].append(values[name].clone())
        observation = transition
        current_clearance = transition_clearance

    treatment_long = treatment.to(torch.long)
    control_long = (~treatment).to(torch.long)
    first_eligible = contract.HANDOFF_HOLD_STEPS + 1
    terminal = first_eligible + len(stable_schedule) - 1
    minus_one = torch.full((num_envs,), -1, dtype=torch.long)
    episodes = {
        "env_slot": env_slot,
        "treatment": treatment,
        "assignment_rank": assignment_rank,
        "fixed_z": contract.expected_fixed_z().expand(num_envs, -1).clone(),
        "triggered": torch.ones(num_envs, dtype=torch.bool),
        "trigger_step": torch.full(
            (num_envs,), contract.HANDOFF_HOLD_STEPS - 1, dtype=torch.long
        ),
        "trigger_score": torch.full((num_envs,), 0.4),
        "first_latch_step": torch.full(
            (num_envs,), contract.HANDOFF_HOLD_STEPS, dtype=torch.long
        ),
        "first_eligible_step": torch.full(
            (num_envs,), first_eligible, dtype=torch.long
        ),
        "first_positive_authority_step": torch.where(
            treatment,
            torch.full((num_envs,), first_eligible + 1, dtype=torch.long),
            torch.full((num_envs,), first_eligible, dtype=torch.long),
        ),
        "first_full_authority_step": torch.where(
            treatment,
            torch.full((num_envs,), first_eligible + 8, dtype=torch.long),
            torch.full((num_envs,), first_eligible, dtype=torch.long),
        ),
        "first_relock_step": torch.where(
            treatment & include_relock,
            torch.full((num_envs,), first_eligible + 9, dtype=torch.long),
            minus_one,
        ),
        "first_reramp_step": torch.where(
            treatment & include_relock,
            torch.full((num_envs,), first_eligible + 11, dtype=torch.long),
            minus_one,
        ),
        "terminal_step": torch.full((num_envs,), terminal, dtype=torch.long),
        "intervention_steps": torch.full((num_envs,), 2, dtype=torch.long),
        "trace_rows": torch.full(
            (num_envs,), len(stable_schedule), dtype=torch.long
        ),
        "stable_count_max": torch.full(
            (num_envs,), ramp.VERIFY_STEPS, dtype=torch.long
        ),
        "positive_authority_rows": treatment_long
        * (16 if include_relock else 18)
        + control_long * 19,
        "partial_authority_rows": treatment_long
        * (14 if include_relock else 7),
        "full_authority_rows": treatment_long
        * (2 if include_relock else 11)
        + control_long * 19,
        "authority_sum": treatment.float()
        * (9.0 if include_relock else 14.5)
        + (~treatment).float() * 19.0,
        "relock_count": treatment_long.clone()
        if include_relock
        else torch.zeros_like(treatment_long),
        "reramp_count": treatment_long.clone()
        if include_relock
        else torch.zeros_like(treatment_long),
        "episode_length": torch.full(
            (num_envs,), terminal + 1, dtype=torch.long
        ),
        "trajectory_max_force_n": torch.full((num_envs,), 10.0),
        "max_true_clearance_m": torch.full((num_envs,), 0.2),
        "first_eligible_clearance_m": torch.zeros(num_envs),
        "eligible": torch.ones(num_envs, dtype=torch.bool),
        "latched_within_window": torch.ones(num_envs, dtype=torch.bool),
        "success": torch.ones(num_envs, dtype=torch.bool),
        "failure": torch.zeros(num_envs, dtype=torch.bool),
        "time_out": torch.zeros(num_envs, dtype=torch.bool),
        "dropped": torch.zeros(num_envs, dtype=torch.bool),
        "unsafe_force": torch.zeros(num_envs, dtype=torch.bool),
        "unlatched_clearance_ge_5cm": torch.zeros(num_envs, dtype=torch.bool),
        "ever_grasped": torch.ones(num_envs, dtype=torch.bool),
        "ever_clearance_ge_20cm": torch.ones(num_envs, dtype=torch.bool),
        "option_active_pre_eligible_unlatched_clearance_ge_5cm": torch.zeros(
            num_envs, dtype=torch.bool
        ),
        "pre_eligibility_action_violations": torch.zeros(
            num_envs, dtype=torch.long
        ),
        "hand_invariance_violations": torch.zeros(num_envs, dtype=torch.long),
        "treatment_arm_ramp_violations": torch.zeros(
            num_envs, dtype=torch.long
        ),
        "authority_clock_violations": torch.zeros(num_envs, dtype=torch.long),
        "control_route_violations": torch.zeros(num_envs, dtype=torch.long),
        "task_action_reconstruction_violations": torch.zeros(
            num_envs, dtype=torch.long
        ),
        "fixed_residual_budget_violations": torch.zeros(
            num_envs, dtype=torch.long
        ),
        "action_bound_violations": torch.zeros(num_envs, dtype=torch.long),
        "new_abs_action_ge_0999": torch.zeros(num_envs, dtype=torch.bool),
    }
    return _metadata(seed=seed, replicate=replicate, num_envs=num_envs), episodes, _cat(
        pre_storage
    ), _cat(trace_storage)


def _artifact(
    *,
    seed: int = 352,
    replicate: str = "b",
    num_envs: int = 8,
    include_relock: bool = True,
) -> dict:
    metadata, episodes, pre_steps, steps = _payload(
        seed=seed,
        replicate=replicate,
        num_envs=num_envs,
        include_relock=include_relock,
    )
    return contract.build_artifact(
        metadata,
        episodes,
        pre_steps,
        steps,
        require_sealed_plan=False,
    )


def _report(artifact: dict) -> dict:
    metadata = artifact["metadata"]
    return {
        "kind": contract.REPORT_KIND,
        "status": "complete",
        "collector": contract.COLLECTOR,
        "seed": metadata["seed"],
        "replicate": metadata["replicate"],
        "num_envs": metadata["num_envs"],
        "vector_steps": int(artifact["episodes"]["episode_length"].max()),
        "v6_actor_sha256": metadata["v6_actor_sha256"],
        "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
        "fixed_direction_sha256": metadata["fixed_direction_sha256"],
        "summary": contract.summarize_artifact(
            artifact, require_sealed_plan=False
        ),
    }


def _stable_public_observation(rows: int) -> torch.Tensor:
    observation = torch.zeros((rows, ramp.OBSERVATION_DIM), dtype=torch.float32)
    observation[:, ramp.PUBLIC_LATCH_INDEX] = 1.0
    observation[:, ramp.STRICT_WRAP_QUALITY_INDEX] = 0.6
    observation[:, ramp.HOLD_QUALITY_INDEX] = 0.7
    observation[:, ramp.FORCE_STRENGTH_START : ramp.FORCE_STRENGTH_STOP] = 0.2
    return observation


def test_candidate43_constants_assignment_and_inherited_public_contract() -> None:
    assert ramp.VERIFY_STEPS == 8
    assert ramp.RAMP_DENOMINATOR == 8.0
    assert ramp.SUPERVISOR_CONTRACT == (
        "public_short_linear_arm_authority_ramp_d8_v1"
    )
    assert ramp.PublicStableState is c42_ramp.PublicStableState
    assert ramp.PublicArmRampState is c42_ramp.PublicArmRampState
    assert ramp.PublicArmRampStep is c42_ramp.PublicArmRampStep
    assert ramp.public_stable_state is c42_ramp.public_stable_state
    for name in (
        "OBSERVATION_DIM",
        "ACTION_DIM",
        "ARM_ACTION_DIM",
        "HAND_ACTION_DIM",
        "FORCE_STRENGTH_START",
        "FORCE_STRENGTH_STOP",
        "STRICT_WRAP_QUALITY_INDEX",
        "HOLD_QUALITY_INDEX",
        "PUBLIC_LATCH_INDEX",
        "GRASP_QUALITY_MIN",
        "HOLD_QUALITY_MIN",
        "PUBLIC_FORCE_STRENGTH_MAX",
        "SAFE_FORCE_STRENGTH_MAX",
        "SAFE_FORCE_LIMIT_N",
        "FORCE_SATURATION_N",
        "PUBLIC_FORCE_COUNTER_CONTRACT",
        "PUBLIC_FORCE_COUNTER_FEATURES",
    ):
        assert getattr(ramp, name) == getattr(c42_ramp, name), name

    for seed, count in ((352, 8), (353, 64), (354, 64)):
        a = ramp.exact_balanced_treatment_mask(
            seed=seed, num_envs=count, replicate="a"
        )
        b = ramp.exact_balanced_treatment_mask(
            seed=seed, num_envs=count, replicate="b"
        )
        rank = ramp.assignment_rank(seed=seed, num_envs=count)
        assert int(a.sum()) == count // 2
        assert torch.equal(b, ~a)
        assert torch.equal(a, rank < count // 2)
        assert torch.equal(torch.sort(rank).values, torch.arange(count))
        assert ramp.assignment_mask_sha256(a) == ramp.assignment_mask_sha256(
            a.clone()
        )
    for count in (2, 16, 63, 128, True):
        _raises(
            ValueError,
            ramp.exact_balanced_treatment_mask,
            seed=352,
            num_envs=count,
            replicate="a",
        )


def test_d8_sequence_hand_control_search_relock_reramp_and_reset() -> None:
    rows = 2
    treatment = torch.tensor([True, False])
    active = torch.ones(rows, dtype=torch.bool)
    option = torch.ones(rows, dtype=torch.bool)
    reset = torch.zeros(rows, dtype=torch.bool)
    counters = torch.zeros((rows, 2), dtype=torch.float32)
    stable = _stable_public_observation(rows)
    baseline = torch.linspace(
        -0.9, 0.9, rows * ramp.ACTION_DIM, dtype=torch.float32
    ).reshape(rows, ramp.ACTION_DIM)
    state = ramp.initial_public_arm_ramp_state(rows)
    expected = torch.arange(9, dtype=torch.float32) / torch.tensor(
        8.0, dtype=torch.float32
    )
    for ordinal, authority in enumerate(expected, start=1):
        step = ramp.apply_public_arm_ramp(
            baseline,
            stable,
            counters,
            option,
            active,
            treatment,
            state,
            reset_mask=reset,
        )
        assert torch.equal(step.stable_count_before, torch.full((rows,), ordinal - 1))
        assert torch.equal(
            step.stable_count_after,
            torch.full((rows,), min(ordinal, 8)),
        )
        assert torch.equal(step.ramp_scale, authority.repeat(rows))
        assert float(step.authority_scale[0]) == float(authority)
        assert float(step.authority_scale[1]) == 1.0
        assert torch.equal(
            step.action[:, ramp.ARM_ACTION_DIM :],
            baseline[:, ramp.ARM_ACTION_DIM :],
        )
        assert torch.equal(step.action[1], baseline[1])
        assert bool(step.full_authority_this_action[0]) == (ordinal == 9)
        if ordinal < 9:
            assert not bool(step.full_authority_this_action[0])
        state = step.next_state

    unstable = stable.clone()
    unstable[:, ramp.HOLD_QUALITY_INDEX] = 0.0
    relocked = ramp.apply_public_arm_ramp(
        baseline,
        unstable,
        counters,
        option,
        active,
        treatment,
        state,
        reset_mask=reset,
    )
    assert float(relocked.authority_scale[0]) == 0.0
    assert int(relocked.stable_count_after[0]) == 0
    assert bool(relocked.relock[0])
    assert torch.equal(
        relocked.action[0, : ramp.ARM_ACTION_DIM],
        torch.zeros(ramp.ARM_ACTION_DIM),
    )
    assert torch.equal(relocked.action[1], baseline[1])

    zero_after_relock = ramp.apply_public_arm_ramp(
        baseline,
        stable,
        counters,
        option,
        active,
        treatment,
        relocked.next_state,
        reset_mask=reset,
    )
    positive_after_relock = ramp.apply_public_arm_ramp(
        baseline,
        stable,
        counters,
        option,
        active,
        treatment,
        zero_after_relock.next_state,
        reset_mask=reset,
    )
    assert float(zero_after_relock.authority_scale[0]) == 0.0
    assert float(positive_after_relock.authority_scale[0]) == 0.125
    assert bool(positive_after_relock.reramp[0])

    search = ramp.apply_public_arm_ramp(
        baseline,
        stable,
        counters,
        torch.zeros(rows, dtype=torch.bool),
        active,
        treatment,
        positive_after_relock.next_state,
        reset_mask=reset,
    )
    assert torch.equal(search.action, baseline)
    reset_step = ramp.apply_public_arm_ramp(
        baseline,
        stable,
        counters,
        option,
        active,
        treatment,
        positive_after_relock.next_state,
        reset_mask=torch.tensor([True, False]),
    )
    assert int(reset_step.stable_count_before[0]) == 0
    assert float(reset_step.authority_scale[0]) == 0.0


def test_artifact_independently_reconstructs_d8_and_rejects_corruption() -> None:
    artifact = _artifact()
    checked = contract.validate_artifact(artifact, require_sealed_plan=False)
    contract.validate_report(_report(checked), checked, require_sealed_plan=False)
    steps = checked["steps"]
    episodes = checked["episodes"]
    treatment_slot = int(episodes["treatment"].nonzero()[0])
    control_slot = int((~episodes["treatment"]).nonzero()[0])
    treated = steps["row_env_slot"] == treatment_slot
    control = steps["row_env_slot"] == control_slot
    expected_segment = torch.arange(9, dtype=torch.float32) / torch.tensor(
        8.0, dtype=torch.float32
    )
    scales = steps["row_authority_scale"][treated]
    assert torch.equal(scales[:9], expected_segment)
    assert float(scales[9]) == 0.0
    assert torch.equal(scales[10:], expected_segment)
    assert bool((scales[:8] < 1).all()) and float(scales[8]) == 1.0
    assert torch.equal(
        steps["row_requested_action"][treated, contract.ARM_ACTION_DIM :],
        steps["row_common_action"][treated, contract.ARM_ACTION_DIM :],
    )
    assert torch.equal(
        steps["row_requested_action"][control],
        steps["row_common_action"][control],
    )
    assert int(episodes["first_full_authority_step"][treatment_slot]) == (
        int(episodes["first_eligible_step"][treatment_slot]) + 8
    )

    treatment_row = int((treated & (steps["row_eligible_age"] == 4)).nonzero()[0])
    corruptions = []
    for table, field, index, delta in (
        ("steps", "row_authority_scale", treatment_row, 0.01),
        ("steps", "row_stable_count_before", treatment_row, 1),
        ("steps", "row_stable_count_after", treatment_row, 1),
        ("steps", "row_requested_action", (treatment_row, 0), 0.01),
        ("steps", "row_requested_action", (treatment_row, 7), 0.01),
        ("episodes", "first_full_authority_step", treatment_slot, 1),
        ("episodes", "authority_sum", treatment_slot, 0.01),
    ):
        bad = copy.deepcopy(artifact)
        bad[table][field][index] += delta
        corruptions.append(bad)
    unstable_row = int((treated & ~steps["row_stable_current"]).nonzero()[0])
    bad = copy.deepcopy(artifact)
    bad["steps"]["row_authority_scale"][unstable_row] = 0.125
    corruptions.append(bad)
    for bad in corruptions:
        _raises(
            ValueError,
            contract.validate_artifact,
            bad,
            require_sealed_plan=False,
        )


def test_complementary_64_env_artifacts_and_plan_protection() -> None:
    a = _artifact(seed=353, replicate="a", num_envs=64)
    b = _artifact(seed=353, replicate="b", num_envs=64)
    contract.validate_complementary_artifacts(
        a, b, require_sealed_plan=False
    )
    assert contract.FINAL_REPORT_KIND == (
        "pick_tool_candidate42_transactional_public_arm_ramp_final_report_v1"
    )
    assert contract.TRANSACTION_CONTRACT == (
        "candidate42_parent_child_fsync_no_clobber_commit_v1"
    )
    plan = contract.validate_sealed_plan(require_sealed=False)
    root = Path(__file__).resolve().parents[2]
    assert contract.sha256_file(root / contract.CANDIDATE42_PLAN) == (
        contract.CANDIDATE42_PLAN_SHA256
    )
    assert contract.canonical_json_sha256(plan["transaction_protocol"]) == (
        contract.SOURCE_SECTION_SHA256["transaction_protocol"]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "plan.json"
        changed = copy.deepcopy(plan)
        changed["candidate43_intervention_contract"]["clock_state_machine"][
            "authority_scale"
        ] = "float32 count_before divided by 7"
        path.write_text(json.dumps(changed), encoding="utf-8")
        _raises(
            ValueError,
            contract.validate_sealed_plan,
            path,
            require_sealed=False,
        )


def _control_u5_artifact(*, seed: int, replicate: str) -> dict:
    """Construct a valid treatment-success/control-U5 analyzer contrast."""

    artifact = copy.deepcopy(
        _artifact(seed=seed, replicate=replicate, num_envs=64)
    )
    episodes = artifact["episodes"]
    steps = artifact["steps"]
    control = ~episodes["treatment"]
    episodes["success"][control] = False
    episodes["failure"][control] = True
    episodes["unlatched_clearance_ge_5cm"][control] = True
    terminal_control = (~steps["row_treatment"]) & steps[
        "row_transition_done"
    ]
    steps["row_transition_observation"][
        terminal_control, ramp.PUBLIC_LATCH_INDEX
    ] = 0.0
    steps["row_transition_public_latch"][terminal_control] = False
    steps["row_transition_grasped"][terminal_control] = False
    steps["row_transition_grasp_quality"][terminal_control] = 0.0
    steps["row_transition_hold_quality"][terminal_control] = 0.0
    steps["row_transition_max_force_n"][terminal_control] = 0.0
    return contract.validate_artifact(artifact, require_sealed_plan=False)


def _validation_artifacts() -> dict[tuple[int, str], dict]:
    return {
        (seed, replicate): _control_u5_artifact(
            seed=seed, replicate=replicate
        )
        for seed, replicate in analyzer.RUN_ORDER
    }


def _passing_gate_metrics() -> dict[str, object]:
    """Place every inclusive gate exactly on its boundary."""

    return {
        "secondary_terminal_utility_delta": 0.1000000000001,
        "positive_secondary_terminal_utility_seeds": 1,
        "minimum_secondary_terminal_utility_seed_delta": -0.05,
        "primary_stable_transport_delta": -0.08,
        "minimum_primary_stable_transport_seed_delta": -0.15,
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


def test_analyzer_all_32_gate_boundaries_and_exclusive_utility() -> None:
    thresholds = analyzer.registered_gate_thresholds()
    assert len(analyzer.GATE_NAMES) == 32
    assert set(thresholds) == set(analyzer.GATE_NAMES)
    passing = _passing_gate_metrics()
    gates = analyzer._build_gate_results(passing, thresholds)
    assert set(gates) == set(analyzer.GATE_NAMES)
    assert all(row["pass"] for row in gates.values())
    assert gates[
        "secondary_conditional_safe_terminal_utility_delta_min_exclusive"
    ]["comparison"] == ">"
    assert gates[
        "primary_conditional_stable_transport_restricted_mean_delta_floor"
    ]["comparison"] == ">="

    failures = {
        "secondary_conditional_safe_terminal_utility_delta_min_exclusive": (
            "secondary_terminal_utility_delta", 0.1
        ),
        "positive_secondary_terminal_utility_delta_seeds_min": (
            "positive_secondary_terminal_utility_seeds", 0
        ),
        "per_seed_secondary_terminal_utility_delta_floor": (
            "minimum_secondary_terminal_utility_seed_delta", -0.050001
        ),
        "primary_conditional_stable_transport_restricted_mean_delta_floor": (
            "primary_stable_transport_delta", -0.080001
        ),
        "per_seed_primary_stable_transport_delta_floor": (
            "minimum_primary_stable_transport_seed_delta", -0.150001
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
        "eligible_opportunity_slots_min_across_validation": (
            "eligible_total", 31
        ),
        "eligible_opportunity_slots_min_per_seed": ("eligible_min_seed", 11),
        "eligible_opportunity_slots_min_per_seed_arm": (
            "eligible_min_seed_arm", 7
        ),
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
        "short_linear_authority_clock_and_action_algebra_exact": (
            "ramp_exact", False
        ),
        "control_candidate39_route_exact": ("control_exact", False),
        "checkpoint_manifest_assignment_and_source_identity_exact": (
            "identity_exact", False
        ),
    }
    assert set(failures) == set(analyzer.GATE_NAMES)
    for gate, (metric, bad_value) in failures.items():
        changed = dict(passing)
        changed[metric] = bad_value
        assert not analyzer._build_gate_results(
            changed, thresholds
        )[gate]["pass"], gate


def test_analyzer_ht_propensity_denominator_and_equal_seed_audit() -> None:
    treatment = torch.tensor([[True, False], [False, True]])
    eligible = torch.ones_like(treatment)
    utility = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    result = analyzer._ht_outcome(treatment, eligible, utility)
    assert result == {
        "known_propensity": 0.5,
        "domain_rows": 4,
        "treatment_rows": 2,
        "control_rows": 2,
        "treatment_sum": 2.0,
        "control_sum": -2.0,
        "treatment": 1.0,
        "control": -1.0,
        "delta": 2.0,
    }
    # Authority observation is deliberately different and never enters the
    # denominator; the pre-treatment eligibility mask remains all four rows.
    authority_observed = treatment.clone()
    assert int(authority_observed.sum()) == 2
    assert result["domain_rows"] == int(eligible.sum())

    report = analyzer.compute_validation(_validation_artifacts())
    audit = report["horvitz_thompson_denominator_audit"]
    assert audit["known_propensity"] == 0.5
    assert audit["authority_clock_used_in_denominator"] is False
    assert audit["all_event_denominators_and_ht_arithmetic_exact"] is True
    assert audit["equal_seed_recomputation_exact"] is True
    for seed in ("353", "354"):
        assert audit["per_seed"][seed][
            "pre_treatment_eligible_denominator"
        ] == 128
        assert audit["per_seed"][seed]["treatment_eligible_rows"] == 64
        assert audit["per_seed"][seed]["control_eligible_rows"] == 64
        for event in analyzer.EVENT_FIELDS:
            assert audit["per_seed"][seed]["events"][event]["exact"] is True


def test_analyzer_d8_action_audit_and_inclusive_authority_ages() -> None:
    artifacts = _validation_artifacts()
    aggregate = analyzer._action_clock_audit(artifacts)["aggregate"]
    assert aggregate["sealed_short_ramp_denominator"] == 8
    assert aggregate["first_full_authority_stable_ordinal"] == 9
    assert aggregate[
        "short_linear_authority_clock_and_action_algebra_exact"
    ] is True
    assert aggregate["hand14_invariance_exact"] is True
    assert aggregate["control_candidate39_route_exact"] is True
    for name in (
        "d8_stable_count_transition_violations",
        "d8_stable_count_continuity_violations",
        "d8_full_authority_too_early_violations",
        "d8_same_action_hard_relock_violations",
    ):
        assert aggregate[name] == 0
    assert aggregate["treatment_partial_authority_rows"] == 1792

    episode = {
        "first_eligible_step": torch.tensor([10, 10, 10, 10, 10]),
        "first_positive_authority_step": torch.tensor([9, 10, 25, 26, -1]),
        "first_full_authority_step": torch.tensor([9, 10, 41, 42, -1]),
    }
    domain = torch.ones(5, dtype=torch.bool)
    assert analyzer._authority_by_age_count(
        episode,
        domain,
        clock_field="first_positive_authority_step",
        maximum_inclusive_age=15,
    ) == 2
    assert analyzer._authority_by_age_count(
        episode,
        domain,
        clock_field="first_full_authority_step",
        maximum_inclusive_age=31,
    ) == 2

    changed = copy.deepcopy(artifacts)
    artifact = changed[(353, "a")]
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
    bad = analyzer._action_clock_audit(changed)["aggregate"]
    assert bad["treatment_arm_ramp_violations"] == 1
    assert bad[
        "short_linear_authority_clock_and_action_algebra_exact"
    ] is False


def test_analyzer_fixed_order_formal_seeds_and_deterministic_report_bytes() -> None:
    assert analyzer.RUN_ORDER == (
        (353, "a"),
        (353, "b"),
        (354, "b"),
        (354, "a"),
    )
    assert analyzer.INVOCATION_SEQUENCE == (
        (352, "b", 8),
        (353, "a", 64),
        (353, "b", 64),
        (354, "b", 64),
        (354, "a", 64),
    )
    plan = contract.validate_sealed_plan(require_sealed=False)
    assert plan["decision_rule"]["formal_seeds"] == [355, 356, 357]
    assert plan["decision_rule"]["formal_status"] == "reserved and untouched"
    assert not ({355, 356, 357} & set(analyzer.SEEDS))
    artifacts = _validation_artifacts()
    checked = analyzer.validate_validation_artifacts(artifacts)
    assert tuple(checked) == analyzer.RUN_ORDER
    missing = dict(artifacts)
    missing.pop((354, "a"))
    _raises(ValueError, analyzer.validate_validation_artifacts, missing)
    extra = dict(artifacts)
    extra[(355, "a")] = extra[(353, "a")]
    _raises(ValueError, analyzer.validate_validation_artifacts, extra)

    first = analyzer.compute_validation(artifacts)
    second = analyzer.compute_validation(artifacts)
    assert first["all_gates_pass"] is True
    assert first["decision"] == "advance_to_formal_screen"
    assert first["primary"]["name"] == (
        "conditional_stable_transport_restricted_mean_delta"
    )
    assert first["primary"]["registered_role"] == (
        "primary_noninferiority_all_must_pass"
    )
    assert first["secondary_terminal_utility"]["registered_role"] == (
        "secondary_but_all_must_pass"
    )
    assert first["secondary_terminal_utility"]["equal_seed_delta"] == 2.0
    assert len(first["gates"]) == 32
    assert analyzer._strict_json_bytes(first) == analyzer._strict_json_bytes(
        second
    )


def test_analyzer_transport_window_remains_exactly_15_rows() -> None:
    base = _artifact(
        seed=353,
        replicate="a",
        num_envs=64,
        include_relock=False,
    )
    slot = int(base["episodes"]["treatment"].nonzero()[0])

    def unsuccessful_with_prefix(prefix: int) -> dict:
        artifact = copy.deepcopy(base)
        episodes = artifact["episodes"]
        steps = artifact["steps"]
        episodes["success"][slot] = False
        episodes["failure"][slot] = True
        episodes["dropped"][slot] = True
        selected = steps["row_env_slot"] == slot
        steps["row_transition_true_clearance_m"][selected] = 0.0
        steps["row_transition_true_clearance_m"][
            selected & (steps["row_eligible_age"] < prefix)
        ] = 0.20
        return artifact

    fourteen = analyzer.stable_transport_outcomes(
        unsuccessful_with_prefix(14)
    )["stable_transition_value"][slot]
    assert not bool((fourteen >= 0.25).any())
    fifteen_result = analyzer.stable_transport_outcomes(
        unsuccessful_with_prefix(15)
    )
    assert fifteen_result["stable_transition_value"][slot, 14] == 1.0
    assert bool(fifteen_result["stable_held_5cm_by_action96"][slot])


def test_candidate43_and_candidate42_section_receipts_are_unambiguous() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(
        (root / contract.VALIDATION_PLAN).read_text(encoding="utf-8")
    )
    live = contract.normalized_scientific_section_sha256(plan)
    assert live == contract.CANDIDATE43_SECTION_SHA256
    assert live == contract.SCIENTIFIC_SECTION_SHA256
    assert live != contract.SOURCE_SECTION_SHA256
