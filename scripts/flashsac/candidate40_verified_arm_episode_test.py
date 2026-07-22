#!/usr/bin/env python3
"""CPU corruption tests for the Candidate40 verified-arm artifact contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import torch

import candidate40_verified_arm_episode as contract
import candidate40_verified_arm_handoff as verifier


def _raises(error: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _metadata(*, seed: int, replicate: str, num_envs: int) -> dict:
    sources = {"scripts/flashsac/fake_candidate40.py": "a" * 64}
    assets = {"/tmp/fake_candidate40.usd": "b" * 64}
    return {
        **contract.REQUIRED_METADATA,
        "seed": seed,
        "replicate": replicate,
        "num_envs": num_envs,
        "assignment_mask_sha256": verifier.assignment_mask_sha256(
            verifier.exact_balanced_treatment_mask(
                seed=seed, num_envs=num_envs, replicate=replicate
            )
        ),
        "smoke_artifact_sha256": None if num_envs == 8 else "d" * 64,
        "smoke_report_sha256": None if num_envs == 8 else "e" * 64,
        "validation_plan_sha256": contract.sha256_file(
            Path(__file__).resolve().parents[2] / contract.VALIDATION_PLAN
        ),
        "source_manifest_sha256": contract.manifest_sha256(sources),
        "runtime_asset_manifest_sha256": contract.manifest_sha256(assets),
        "source_sha256": sources,
        "runtime_asset_sha256": assets,
        "flashsac_upstream_commit": "87edc9061150ae9e962dd84e6544e27a1554b3ab",
        "git": {
            "commit": "c" * 40,
            "branch": "flashsac-pick-tool-curriculum",
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


def _observation(
    num_envs: int,
    *,
    stable: bool,
    task_action: torch.Tensor | None = None,
) -> torch.Tensor:
    obs = torch.zeros((num_envs, contract.OBSERVATION_DIM), dtype=torch.float32)
    if stable:
        obs[:, verifier.PUBLIC_LATCH_INDEX] = 1.0
        obs[:, verifier.FORCE_STRENGTH_START : verifier.FORCE_STRENGTH_STOP] = 0.2
        obs[:, verifier.STRICT_WRAP_QUALITY_INDEX] = 0.6
        obs[:, verifier.HOLD_QUALITY_INDEX] = 0.7
    if task_action is not None:
        obs[:, contract.TRANSITION_ARM_TOKEN_SLICE] = task_action[:, :16]
        obs[:, contract.TRANSITION_DISTAL_SLICE] = task_action[:, 16:]
    return obs


def _cat(storage: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.cat(chunks, dim=0) for name, chunks in storage.items()}


def _payload(
    *, seed: int = 334, replicate: str = "b", num_envs: int = 8
) -> tuple[dict, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    treatment = verifier.exact_balanced_treatment_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    rank = verifier.assignment_rank(seed=seed, num_envs=num_envs)
    env_slot = torch.arange(num_envs, dtype=torch.long)
    baseline = torch.zeros((num_envs, contract.ACTION_DIM), dtype=torch.float32)
    common_pre = contract.expected_fixed_residual_action(baseline)
    delta = contract.expected_applied_delta().expand(num_envs, -1).clone()

    # Two common fixed-residual actions: the second creates the first public
    # latch transition.  Candidate40 assignment is deliberately absent here.
    pre_storage = {name: [] for name in contract.PRE_STEP_FIELDS}
    pre_obs = _observation(num_envs, stable=False)
    pre_obs[:, contract.PROXIMITY_SLICE] = 0.4
    pre_obs[:, contract.THUMB_PROXIMITY_INDEX] = 0.4
    trace_initial: torch.Tensor | None = None
    for step_index in range(2):
        episode_step = step_index + contract.HANDOFF_HOLD_STEPS - 1
        task_action = common_pre.clone()
        transition = _observation(
            num_envs,
            stable=step_index == 1,
            task_action=task_action,
        )
        values = {
            "pre_env_slot": env_slot,
            "pre_episode_step": torch.full((num_envs,), episode_step, dtype=torch.long),
            "pre_close_age": torch.full((num_envs,), step_index, dtype=torch.long),
            "pre_option_active": torch.ones(num_envs, dtype=torch.bool),
            "pre_public_latch_before": torch.zeros(num_envs, dtype=torch.bool),
            "pre_residual_active": torch.ones(num_envs, dtype=torch.bool),
            "pre_first_latch_witness": torch.full(
                (num_envs,), step_index == 1, dtype=torch.bool
            ),
            "pre_observation": pre_obs,
            "pre_applied_delta": delta,
            "pre_baseline_action": baseline,
            "pre_common_action": common_pre,
            "pre_requested_action": common_pre,
            "pre_transition_observation": transition,
            "pre_task_action": task_action,
            "pre_transition_public_latch": torch.full(
                (num_envs,), step_index == 1, dtype=torch.bool
            ),
            "pre_transition_grasped": torch.full(
                (num_envs,), step_index == 1, dtype=torch.bool
            ),
            "pre_transition_true_clearance_m": torch.zeros(num_envs),
            "pre_grasp_quality": torch.full(
                (num_envs,), 0.6 if step_index == 1 else 0.0
            ),
            "pre_hold_quality": torch.full(
                (num_envs,), 0.7 if step_index == 1 else 0.0
            ),
            "pre_max_force_n": torch.full(
                (num_envs,), 10.0 if step_index == 1 else 0.0
            ),
            "pre_transition_done": torch.zeros(num_envs, dtype=torch.bool),
        }
        for name in contract.PRE_STEP_FIELDS:
            pre_storage[name].append(values[name].clone())
        pre_obs = transition
        trace_initial = transition
    assert trace_initial is not None

    # Stable rows 2..17 verify and enable; row 18 loses latch and relocks on
    # that same action; rows 19..34 complete a fresh verification and re-enable.
    trace_storage = {name: [] for name in contract.STEP_FIELDS}
    trace_length = 33
    stable_schedule = [True] * trace_length
    stable_schedule[16] = False
    state = verifier.initial_verified_arm_state(num_envs, device="cpu")
    observation = trace_initial
    current_clearance = torch.zeros(num_envs, dtype=torch.float32)
    for age, stable in enumerate(stable_schedule):
        episode_step = age + contract.HANDOFF_HOLD_STEPS + 1
        if bool((observation[:, verifier.PUBLIC_LATCH_INDEX] == 1.0).all()) != stable:
            raise AssertionError("fixture observation schedule is inconsistent")
        common = torch.full(
            (num_envs, contract.ACTION_DIM), 0.1, dtype=torch.float32
        )
        common[:, : contract.ARM_ACTION_DIM] = 0.2 if stable else 0.0
        counters = torch.zeros((num_envs, 2), dtype=torch.float32)
        state_before = state
        verified = verifier.apply_verified_arm_handoff(
            common,
            observation,
            counters,
            torch.ones(num_envs, dtype=torch.bool),
            torch.ones(num_envs, dtype=torch.bool),
            treatment,
            state,
            reset_mask=torch.zeros(num_envs, dtype=torch.bool),
        )
        state = verified.next_state
        task_action = verified.action.clone()
        next_stable = stable_schedule[age + 1] if age + 1 < trace_length else True
        transition = _observation(
            num_envs, stable=next_stable, task_action=task_action
        )
        transition_clearance = torch.full(
            (num_envs,), 0.2 if age == trace_length - 1 else 0.001
        )
        reenable = verified.newly_enabled & state_before.ever_enabled
        pre_enable = (~state_before.ever_enabled) & (~verified.arm_enabled_this_action)
        target_error = torch.full((num_envs, contract.ARM_ACTION_DIM), 0.1)
        velocity = torch.full((num_envs, contract.ARM_ACTION_DIM), -0.02)
        values = {
            "row_env_slot": env_slot,
            "row_episode_step": torch.full((num_envs,), episode_step, dtype=torch.long),
            "row_eligible_age": torch.full((num_envs,), age, dtype=torch.long),
            "row_observation": observation,
            "row_transition_observation": transition,
            "row_public_force_counters": counters,
            "row_option_active": torch.ones(num_envs, dtype=torch.bool),
            "row_public_latch_before": observation[:, verifier.PUBLIC_LATCH_INDEX] == 1.0,
            "row_stable_current": verified.stable_current,
            "row_public_grasp_quality": verified.public_grasp_quality,
            "row_public_hold_quality": verified.public_hold_quality,
            "row_public_max_force_strength": verified.public_max_force_strength,
            "row_stable_count_before": verified.stable_count_before,
            "row_stable_count_after": verified.stable_count_after,
            "row_common_action": common,
            "row_requested_action": verified.action,
            "row_task_action": task_action,
            "row_arm_target_error": target_error,
            "row_arm_joint_velocity": velocity,
            "row_arm_target_error_abs_max": target_error.abs().max(dim=-1).values,
            "row_arm_joint_speed_abs_max": velocity.abs().max(dim=-1).values,
            "row_pre_action_true_clearance_m": current_clearance,
            "row_treatment": treatment,
            "row_activated": verified.activated_this_action,
            "row_first_eligible": verified.first_eligible,
            "row_arm_enabled": verified.arm_enabled_this_action,
            "row_verification_complete": verified.verification_complete,
            "row_relock": verified.relock,
            "row_reenable": reenable,
            "row_pre_enable": pre_enable,
            "row_transition_public_latch": transition[:, verifier.PUBLIC_LATCH_INDEX] == 1.0,
            "row_transition_grasped": transition[:, verifier.PUBLIC_LATCH_INDEX] == 1.0,
            "row_transition_grasp_quality": torch.where(
                transition[:, verifier.PUBLIC_LATCH_INDEX] == 1.0,
                torch.full((num_envs,), 0.6),
                torch.zeros(num_envs),
            ),
            "row_transition_hold_quality": torch.where(
                transition[:, verifier.PUBLIC_LATCH_INDEX] == 1.0,
                torch.full((num_envs,), 0.7),
                torch.zeros(num_envs),
            ),
            "row_transition_max_force_n": torch.where(
                transition[:, verifier.PUBLIC_LATCH_INDEX] == 1.0,
                torch.full((num_envs,), 10.0),
                torch.zeros(num_envs),
            ),
            "row_transition_object_lin_speed": torch.full((num_envs,), 0.01),
            "row_transition_object_ang_speed": torch.full((num_envs,), 0.02),
            "row_transition_true_clearance_m": transition_clearance,
            "row_transition_done": torch.full(
                (num_envs,), age == trace_length - 1, dtype=torch.bool
            ),
            "row_pre_release_launch": torch.zeros(num_envs, dtype=torch.bool),
        }
        for name in contract.STEP_FIELDS:
            trace_storage[name].append(values[name].clone())
        observation = transition
        current_clearance = transition_clearance

    episodes = {
        "env_slot": env_slot,
        "treatment": treatment,
        "assignment_rank": rank,
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
            (num_envs,), contract.HANDOFF_HOLD_STEPS + 1, dtype=torch.long
        ),
        "verification_complete_step": torch.full((num_envs,), 19, dtype=torch.long),
        "first_arm_enabled_step": torch.full((num_envs,), 20, dtype=torch.long),
        "first_relock_step": torch.full((num_envs,), 21, dtype=torch.long),
        "first_reenable_step": torch.full((num_envs,), 37, dtype=torch.long),
        "terminal_step": torch.full((num_envs,), 37, dtype=torch.long),
        "intervention_steps": torch.full((num_envs,), 2, dtype=torch.long),
        "trace_rows": torch.full((num_envs,), trace_length, dtype=torch.long),
        "stable_count_max": torch.full((num_envs,), verifier.VERIFY_STEPS, dtype=torch.long),
        "arm_enable_count": torch.full((num_envs,), 2, dtype=torch.long),
        "relock_count": torch.ones(num_envs, dtype=torch.long),
        "reenable_count": torch.ones(num_envs, dtype=torch.long),
        "episode_length": torch.full((num_envs,), 38, dtype=torch.long),
        "trajectory_max_force_n": torch.full((num_envs,), 10.0),
        "max_true_clearance_m": torch.full((num_envs,), 0.2),
        "first_eligible_clearance_m": torch.zeros(num_envs),
        "pre_release_max_clearance_m": torch.full((num_envs,), 0.001),
        "eligible": torch.ones(num_envs, dtype=torch.bool),
        "pre_release_launch": torch.zeros(num_envs, dtype=torch.bool),
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
        "pre_eligibility_action_violations": torch.zeros(num_envs, dtype=torch.long),
        "hand_invariance_violations": torch.zeros(num_envs, dtype=torch.long),
        "treatment_arm_gate_violations": torch.zeros(num_envs, dtype=torch.long),
        "control_route_violations": torch.zeros(num_envs, dtype=torch.long),
        "task_action_reconstruction_violations": torch.zeros(num_envs, dtype=torch.long),
        "fixed_residual_budget_violations": torch.zeros(num_envs, dtype=torch.long),
        "action_bound_violations": torch.zeros(num_envs, dtype=torch.long),
        "treatment_pre_enable_nonzero_arm_rows": torch.zeros(num_envs, dtype=torch.long),
        "new_abs_action_ge_0999": torch.zeros(num_envs, dtype=torch.bool),
    }
    return (
        _metadata(seed=seed, replicate=replicate, num_envs=num_envs),
        episodes,
        _cat(pre_storage),
        _cat(trace_storage),
    )


def _artifact(
    *, seed: int = 334, replicate: str = "b", num_envs: int = 8
) -> dict:
    metadata, episodes, pre_steps, steps = _payload(
        seed=seed, replicate=replicate, num_envs=num_envs
    )
    return contract.build_artifact(
        metadata,
        episodes,
        pre_steps,
        steps,
        require_sealed_plan=False,
    )


def _post_window_payload() -> tuple[dict, dict, dict, dict]:
    """Ineligible terminal U5 on CLOSE age 32, after residual retirement."""

    seed, replicate, num_envs = 334, "b", 8
    treatment = verifier.exact_balanced_treatment_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    env_slot = torch.arange(num_envs, dtype=torch.long)
    baseline = torch.zeros((num_envs, contract.ACTION_DIM), dtype=torch.float32)
    overlaid = contract.expected_fixed_residual_action(baseline)
    fixed_delta = contract.expected_applied_delta().expand(num_envs, -1).clone()
    storage = {name: [] for name in contract.PRE_STEP_FIELDS}
    observation = _observation(num_envs, stable=False)
    observation[:, contract.PROXIMITY_SLICE] = 0.4
    observation[:, contract.THUMB_PROXIMITY_INDEX] = 0.4
    for age in range(33):
        episode_step = age + contract.HANDOFF_HOLD_STEPS - 1
        residual_active = age < contract.WINDOW_STEPS
        common = overlaid if residual_active else baseline
        delta = fixed_delta if residual_active else torch.zeros_like(fixed_delta)
        task_action = common.clone()
        transition = _observation(
            num_envs, stable=False, task_action=task_action
        )
        values = {
            "pre_env_slot": env_slot,
            "pre_episode_step": torch.full((num_envs,), episode_step, dtype=torch.long),
            "pre_close_age": torch.full((num_envs,), age, dtype=torch.long),
            "pre_option_active": torch.ones(num_envs, dtype=torch.bool),
            "pre_public_latch_before": torch.zeros(num_envs, dtype=torch.bool),
            "pre_residual_active": torch.full(
                (num_envs,), residual_active, dtype=torch.bool
            ),
            "pre_first_latch_witness": torch.zeros(num_envs, dtype=torch.bool),
            "pre_observation": observation,
            "pre_applied_delta": delta,
            "pre_baseline_action": baseline,
            "pre_common_action": common,
            "pre_requested_action": common,
            "pre_transition_observation": transition,
            "pre_task_action": task_action,
            "pre_transition_public_latch": torch.zeros(num_envs, dtype=torch.bool),
            "pre_transition_grasped": torch.zeros(num_envs, dtype=torch.bool),
            "pre_transition_true_clearance_m": torch.full(
                (num_envs,), 0.05 if age == 32 else 0.0
            ),
            "pre_grasp_quality": torch.zeros(num_envs),
            "pre_hold_quality": torch.zeros(num_envs),
            "pre_max_force_n": torch.zeros(num_envs),
            "pre_transition_done": torch.full(
                (num_envs,), age == 32, dtype=torch.bool
            ),
        }
        for name in contract.PRE_STEP_FIELDS:
            storage[name].append(values[name].clone())
        observation = transition

    _, _, _, sample_steps = _payload()
    empty_steps = {name: value[:0].clone() for name, value in sample_steps.items()}
    zeros_long = torch.zeros(num_envs, dtype=torch.long)
    minus_one = torch.full((num_envs,), -1, dtype=torch.long)
    episodes = {
        "env_slot": env_slot,
        "treatment": treatment,
        "assignment_rank": verifier.assignment_rank(seed=seed, num_envs=num_envs),
        "fixed_z": contract.expected_fixed_z().expand(num_envs, -1).clone(),
        "triggered": torch.ones(num_envs, dtype=torch.bool),
        "trigger_step": torch.full(
            (num_envs,), contract.HANDOFF_HOLD_STEPS - 1, dtype=torch.long
        ),
        "trigger_score": torch.full((num_envs,), 0.4),
        "first_latch_step": minus_one.clone(),
        "first_eligible_step": minus_one.clone(),
        "verification_complete_step": minus_one.clone(),
        "first_arm_enabled_step": minus_one.clone(),
        "first_relock_step": minus_one.clone(),
        "first_reenable_step": minus_one.clone(),
        "terminal_step": torch.full((num_envs,), 35, dtype=torch.long),
        "intervention_steps": torch.full(
            (num_envs,), contract.WINDOW_STEPS, dtype=torch.long
        ),
        "trace_rows": zeros_long.clone(),
        "stable_count_max": zeros_long.clone(),
        "arm_enable_count": zeros_long.clone(),
        "relock_count": zeros_long.clone(),
        "reenable_count": zeros_long.clone(),
        "episode_length": torch.full((num_envs,), 36, dtype=torch.long),
        "trajectory_max_force_n": torch.zeros(num_envs),
        "max_true_clearance_m": torch.full((num_envs,), 0.05),
        "first_eligible_clearance_m": torch.zeros(num_envs),
        "pre_release_max_clearance_m": torch.zeros(num_envs),
        "eligible": torch.zeros(num_envs, dtype=torch.bool),
        "pre_release_launch": torch.zeros(num_envs, dtype=torch.bool),
        "latched_within_window": torch.zeros(num_envs, dtype=torch.bool),
        "success": torch.zeros(num_envs, dtype=torch.bool),
        "failure": torch.ones(num_envs, dtype=torch.bool),
        "time_out": torch.zeros(num_envs, dtype=torch.bool),
        "dropped": torch.zeros(num_envs, dtype=torch.bool),
        "unsafe_force": torch.zeros(num_envs, dtype=torch.bool),
        "unlatched_clearance_ge_5cm": torch.ones(num_envs, dtype=torch.bool),
        "ever_grasped": torch.zeros(num_envs, dtype=torch.bool),
        "ever_clearance_ge_20cm": torch.zeros(num_envs, dtype=torch.bool),
        "option_active_pre_eligible_unlatched_clearance_ge_5cm": torch.ones(
            num_envs, dtype=torch.bool
        ),
        "pre_eligibility_action_violations": zeros_long.clone(),
        "hand_invariance_violations": zeros_long.clone(),
        "treatment_arm_gate_violations": zeros_long.clone(),
        "control_route_violations": zeros_long.clone(),
        "task_action_reconstruction_violations": zeros_long.clone(),
        "fixed_residual_budget_violations": zeros_long.clone(),
        "action_bound_violations": zeros_long.clone(),
        "treatment_pre_enable_nonzero_arm_rows": zeros_long.clone(),
        "new_abs_action_ge_0999": torch.zeros(num_envs, dtype=torch.bool),
    }
    return _metadata(seed=seed, replicate=replicate, num_envs=num_envs), episodes, _cat(storage), empty_steps


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


def test_valid_clock_relock_and_summary() -> None:
    artifact = _artifact()
    contract.validate_artifact(artifact, require_sealed_plan=False)
    contract.validate_report(
        _report(artifact), artifact, require_sealed_plan=False
    )
    episodes = artifact["episodes"]
    assert bool((episodes["first_arm_enabled_step"] == 20).all())
    assert bool((episodes["first_relock_step"] == 21).all())
    assert bool((episodes["first_reenable_step"] == 37).all())
    assert contract.summarize_artifact(
        artifact, require_sealed_plan=False
    )["trace_rows"] == 8 * 33


def test_corruptions_fail_closed() -> None:
    base = _artifact()
    mutations: list[dict] = []
    bad = copy.deepcopy(base)
    bad["pre_steps"]["pre_transition_observation"][0, 70] += 0.01
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["pre_steps"]["pre_common_action"][0, 7] += 0.01
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["steps"]["row_stable_count_before"][8] += 1
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["steps"]["row_requested_action"][0, 7] += 0.01
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["steps"]["row_arm_target_error_abs_max"][0] += 0.01
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["first_relock_step"][0] += 1
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["treatment"][0] = ~bad["episodes"]["treatment"][0]
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["metadata"]["source_manifest_sha256"] = "f" * 64
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["dropped"][0] = True
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["unsafe_force"][0] = True
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["unlatched_clearance_ge_5cm"][0] = True
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["ever_clearance_ge_20cm"][0] = False
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["success"][0] = False
    bad["episodes"]["time_out"][0] = True
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["trigger_step"][0] = contract.HANDOFF_HOLD_STEPS - 2
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["trigger_score"][0] += 0.01
    mutations.append(bad)
    for artifact in mutations:
        _raises(
            ValueError,
            contract.validate_artifact,
            artifact,
            require_sealed_plan=False,
        )
    bad = copy.deepcopy(base)
    bad["steps"]["row_pre_action_true_clearance_m"][0] = float("nan")
    _raises(
        FloatingPointError,
        contract.validate_artifact,
        bad,
        require_sealed_plan=False,
    )


def test_post_window_option_trace_and_u5_are_reconstructed() -> None:
    metadata, episodes, pre_steps, steps = _post_window_payload()
    artifact = contract.build_artifact(
        metadata,
        episodes,
        pre_steps,
        steps,
        require_sealed_plan=False,
    )
    assert not bool(artifact["pre_steps"]["pre_residual_active"][-1])
    assert torch.equal(
        artifact["pre_steps"]["pre_applied_delta"][-1],
        torch.zeros(contract.HAND_ACTION_DIM),
    )
    assert torch.equal(
        artifact["pre_steps"]["pre_common_action"][-1],
        artifact["pre_steps"]["pre_baseline_action"][-1],
    )
    assert bool(
        artifact["episodes"][
            "option_active_pre_eligible_unlatched_clearance_ge_5cm"
        ].all()
    )
    for field, mutate in (
        ("pre_applied_delta", lambda value: value.__setitem__((-1, 0), 0.01)),
        ("pre_common_action", lambda value: value.__setitem__((-1, 0), 0.01)),
    ):
        bad = copy.deepcopy(artifact)
        mutate(bad["pre_steps"][field])
        _raises(
            ValueError,
            contract.validate_artifact,
            bad,
            require_sealed_plan=False,
        )
    bad = copy.deepcopy(artifact)
    bad["episodes"][
        "option_active_pre_eligible_unlatched_clearance_ge_5cm"
    ][0] = False
    _raises(
        ValueError,
        contract.validate_artifact,
        bad,
        require_sealed_plan=False,
    )
    bad = copy.deepcopy(artifact)
    bad["episodes"]["max_true_clearance_m"][0] = 0.049
    _raises(
        ValueError,
        contract.validate_artifact,
        bad,
        require_sealed_plan=False,
    )


def test_complement_and_no_clobber_publication() -> None:
    left = _artifact(seed=335, replicate="a", num_envs=64)
    right = _artifact(seed=335, replicate="b", num_envs=64)
    contract.validate_complementary_artifacts(
        left, right, require_sealed_plan=False
    )
    corrupt = copy.deepcopy(right)
    corrupt["episodes"]["fixed_z"][0, 0] += 0.01
    _raises(
        ValueError,
        contract.validate_complementary_artifacts,
        left,
        corrupt,
        require_sealed_plan=False,
    )

    artifact = _artifact()
    report = _report(artifact)
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        artifact_path = root / "trial.pt"
        report_path = root / "trial.json"
        digest = contract.publish_artifact_and_report_no_clobber(
            artifact,
            report,
            artifact_path,
            report_path,
            require_sealed_plan=False,
        )
        assert digest == contract.sha256_file(artifact_path)
        loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
        contract.validate_artifact(loaded, require_sealed_plan=False)
        published = json.loads(report_path.read_text(encoding="utf-8"))
        contract.validate_report(
            published,
            loaded,
            published=True,
            require_sealed_plan=False,
        )
        _raises(
            FileExistsError,
            contract.publish_artifact_and_report_no_clobber,
            artifact,
            report,
            artifact_path,
            report_path,
            require_sealed_plan=False,
        )


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    test_valid_clock_relock_and_summary()
    test_corruptions_fail_closed()
    test_post_window_option_trace_and_u5_are_reconstructed()
    test_complement_and_no_clobber_publication()
    print("candidate40_verified_arm_episode tests passed")


if __name__ == "__main__":
    main()
