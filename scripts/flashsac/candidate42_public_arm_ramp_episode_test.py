#!/usr/bin/env python3
"""CPU corruption tests and synthetic fixtures for Candidate42 artifacts."""

from __future__ import annotations

import copy
import ast
import inspect
import json
from pathlib import Path
import tempfile
from unittest import mock

import torch

import candidate41_public_arm_ramp_episode as c41_contract
import candidate42_public_arm_ramp as ramp
import candidate42_public_arm_ramp_episode as contract


def _raises(error: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _metadata(*, seed: int, replicate: str, num_envs: int) -> dict:
    sources = {"scripts/flashsac/fake_candidate42.py": "a" * 64}
    assets = {"/tmp/fake_candidate42.usd": "b" * 64}
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
    plan["preregistration_receipt"]["commit"] = "1" * 40
    plan["preregistration_receipt"]["sha256"] = "2" * 64
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
    *, seed: int = 346, replicate: str = "b", num_envs: int = 8
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

    # 16 stable actions ramp 0..1, one same-action relock, then another
    # 16 stable actions that reramp 0..1.
    stable_schedule = [True] * 33
    stable_schedule[16] = False
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
            torch.full((num_envs,), first_eligible + 15, dtype=torch.long),
            torch.full((num_envs,), first_eligible, dtype=torch.long),
        ),
        "first_relock_step": torch.where(
            treatment,
            torch.full((num_envs,), first_eligible + 16, dtype=torch.long),
            minus_one,
        ),
        "first_reramp_step": torch.where(
            treatment,
            torch.full((num_envs,), first_eligible + 18, dtype=torch.long),
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
        "positive_authority_rows": treatment_long * 30 + control_long * 33,
        "partial_authority_rows": treatment_long * 28,
        "full_authority_rows": treatment_long * 2 + control_long * 33,
        "authority_sum": treatment.float() * 16.0 + (~treatment).float() * 33.0,
        "relock_count": treatment_long.clone(),
        "reramp_count": treatment_long.clone(),
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
    *, seed: int = 346, replicate: str = "b", num_envs: int = 8
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


def test_valid_ramp_relock_reramp_and_summary() -> None:
    assert contract.IMPLEMENTATION_SOURCE_FILES == (
        "scripts/flashsac/candidate42_public_arm_ramp.py",
        "scripts/flashsac/candidate42_public_arm_ramp_test.py",
        "scripts/flashsac/candidate42_public_arm_ramp_episode.py",
        "scripts/flashsac/candidate42_public_arm_ramp_episode_test.py",
        "scripts/flashsac/candidate42_attempt_transaction.py",
        "scripts/flashsac/candidate42_attempt_transaction_test.py",
        "scripts/flashsac/candidate42_public_arm_ramp_collection.py",
        "scripts/flashsac/candidate42_public_arm_ramp_collection_test.py",
        "scripts/flashsac/collect_candidate42_public_arm_ramp_child.py",
        "scripts/flashsac/collect_candidate42_public_arm_ramp_child_test.py",
        "scripts/flashsac/collect_candidate42_public_arm_ramp_ab.py",
        "scripts/flashsac/collect_candidate42_public_arm_ramp_ab_test.py",
        "scripts/flashsac/analyze_candidate42_public_arm_ramp_validation.py",
        "scripts/flashsac/analyze_candidate42_public_arm_ramp_validation_test.py",
    )
    artifact = _artifact()
    contract.validate_artifact(artifact, require_sealed_plan=False)
    contract.validate_report(
        _report(artifact), artifact, require_sealed_plan=False
    )
    ep = artifact["episodes"]
    treatment = ep["treatment"]
    assert bool((ep["first_positive_authority_step"][treatment] == 6).all())
    assert bool((ep["first_full_authority_step"][treatment] == 20).all())
    assert bool((ep["first_relock_step"][treatment] == 21).all())
    assert bool((ep["first_reramp_step"][treatment] == 23).all())
    assert contract.summarize_artifact(
        artifact, require_sealed_plan=False
    )["trace_rows"] == 8 * 33


def test_label_normalized_science_is_mechanically_equal_to_candidate41() -> None:
    assert contract.EPISODE_FIELDS == c41_contract.EPISODE_FIELDS
    assert contract.PRE_STEP_FIELDS == c41_contract.PRE_STEP_FIELDS
    assert contract.STEP_FIELDS == c41_contract.STEP_FIELDS
    functions = (
        "_tensor",
        "_table",
        "_episode_table",
        "_pre_table",
        "_step_table",
        "_validate_episode_headers",
        "_validate_pre_trace",
        "_linear_ramp_scale",
        "_first_clock",
        "_validate_step_trace",
        "_validate_semantics",
        "validate_artifact",
        "build_artifact",
        "summarize_artifact",
        "validate_complementary_artifacts",
    )
    for name in functions:
        c42_source = inspect.getsource(getattr(contract, name)).replace(
            "Candidate42", "Candidate41"
        )
        c41_source = inspect.getsource(getattr(c41_contract, name))
        assert ast.dump(
            ast.parse(c42_source), include_attributes=False
        ) == ast.dump(ast.parse(c41_source), include_attributes=False), name

    generator = torch.Generator().manual_seed(4242)
    stable = torch.rand(4096, generator=generator) > 0.35
    count = torch.randint(
        0, ramp.VERIFY_STEPS + 1, (4096,), generator=generator
    )
    assert torch.equal(
        contract._linear_ramp_scale(stable, count),
        c41_contract._linear_ramp_scale(stable, count),
    )

    root = Path(__file__).resolve().parents[2]
    c41_plan = json.loads((root / contract.CANDIDATE41_PLAN).read_text())
    c42_plan = json.loads((root / contract.VALIDATION_PLAN).read_text())
    assert contract.sha256_file(root / contract.CANDIDATE41_PLAN) == (
        contract.CANDIDATE41_PLAN_SHA256
    )
    expected = {
        name: contract.canonical_json_sha256(c41_plan[name])
        for name in contract.SCIENTIFIC_SECTION_SHA256
    }
    assert expected == contract.SCIENTIFIC_SECTION_SHA256
    assert contract.normalized_scientific_section_sha256(c42_plan) == expected


def test_exact_float32_authority_sequence_and_control_identity() -> None:
    artifact = _artifact()
    steps = artifact["steps"]
    treatment_slot = int(artifact["episodes"]["treatment"].nonzero()[0])
    control_slot = int((~artifact["episodes"]["treatment"]).nonzero()[0])
    treatment_rows = steps["row_env_slot"] == treatment_slot
    control_rows = steps["row_env_slot"] == control_slot
    expected = torch.arange(16, dtype=torch.float32) / torch.tensor(
        15.0, dtype=torch.float32
    )
    assert torch.equal(steps["row_authority_scale"][treatment_rows][:16], expected)
    assert float(steps["row_authority_scale"][treatment_rows][16]) == 0.0
    assert bool((steps["row_authority_scale"][control_rows] == 1).all())
    assert torch.equal(
        steps["row_requested_action"][control_rows],
        steps["row_common_action"][control_rows],
    )


def test_corruptions_fail_closed() -> None:
    base = _artifact()
    mutations: list[dict] = []

    def changed(path: tuple[str, str], index, delta=1) -> None:
        bad = copy.deepcopy(base)
        bad[path[0]][path[1]][index] += delta
        mutations.append(bad)

    changed(("steps", "row_authority_scale"), 8, 0.01)
    changed(("steps", "row_requested_action"), (8, 0), 0.01)
    changed(("steps", "row_requested_action"), (8, 7), 0.01)
    changed(("steps", "row_stable_count_before"), 8, 1)
    changed(("steps", "row_arm_target_error_abs_max"), 0, 0.01)
    changed(("episodes", "first_positive_authority_step"), 0, 1)
    changed(("episodes", "positive_authority_rows"), 0, 1)
    changed(("episodes", "authority_sum"), 0, 0.01)
    bad = copy.deepcopy(base)
    bad["pre_steps"]["pre_trigger_witness"][0] = False
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["pre_steps"]["pre_first_latch_witness"][8] = False
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["treatment"][0] = ~bad["episodes"]["treatment"][0]
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["metadata"]["source_manifest_sha256"] = "f" * 64
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["metadata"]["implementation_commit"] = "not-a-git-sha"
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["authority_clock_violations"][0] = 1
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["unsafe_force"][0] = True
    mutations.append(bad)
    bad = copy.deepcopy(base)
    bad["episodes"]["ever_clearance_ge_20cm"][0] = False
    mutations.append(bad)
    bad = copy.deepcopy(base)
    keep = torch.ones(base["pre_steps"]["pre_env_slot"].numel(), dtype=torch.bool)
    keep[0] = False
    for name in contract.PRE_STEP_FIELDS:
        bad["pre_steps"][name] = bad["pre_steps"][name][keep]
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


def test_plan_identity_fail_closed() -> None:
    path = Path(__file__).resolve().parents[2] / contract.VALIDATION_PLAN
    plan = json.loads(path.read_text(encoding="utf-8"))
    mutations = []
    bad = copy.deepcopy(plan)
    bad["candidate42_intervention_contract"]["name"] = "changed"
    mutations.append(bad)
    bad = copy.deepcopy(plan)
    bad["assignment"]["salt"] = "changed"
    mutations.append(bad)
    bad = copy.deepcopy(plan)
    bad["execution"]["run_order"] = list(reversed(bad["execution"]["run_order"]))
    mutations.append(bad)
    bad = copy.deepcopy(plan)
    bad["causal_basis"]["candidate40_combined_report"]["sha256"] = "f" * 64
    mutations.append(bad)
    bad = copy.deepcopy(plan)
    bad["causal_basis"]["candidate41_rejection_manifest"]["sha256"] = "f" * 64
    mutations.append(bad)
    bad = copy.deepcopy(plan)
    bad["candidate42_intervention_contract"]["eligibility_domain"] = (
        bad["candidate42_intervention_contract"]["eligibility_domain"].replace(
            "Candidate42", "Candidate43"
        )
    )
    mutations.append(bad)
    bad = copy.deepcopy(plan)
    bad["development_validation_gates"][
        "conditional_success_delta_floor"
    ] = -0.04
    mutations.append(bad)
    bad = copy.deepcopy(plan)
    bad["status"] = "sealed_before_smoke_and_collection"
    mutations.append(bad)
    with tempfile.TemporaryDirectory() as name:
        target = Path(name) / "plan.json"
        for mutation in mutations:
            target.write_text(json.dumps(mutation), encoding="utf-8")
            _raises(
                ValueError,
                contract.validate_sealed_plan,
                target,
                require_sealed=False,
            )


def test_sealed_plan_audit_schema_fails_closed() -> None:
    valid = _sealed_plan_fixture()
    mutations: list[dict] = []

    bad = copy.deepcopy(valid)
    bad["implementation_seal"].pop(
        "simulator_evidence_before_source_seal"
    )
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["simulator_evidence_before_source_seal"] = True
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["simulation_free_tests"]["passed"] = 0
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["simulation_free_tests"]["failed"] = 1
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["simulation_free_tests"][
        "required_contracts_covered"
    ] = bad["implementation_seal"]["simulation_free_tests"][
        "required_contracts_covered"
    ][:-1]
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["simulation_free_tests"]["unexpected"] = True
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["static_audit"]["blockers"] = 1
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["static_audit"]["simulator_invocations"] = 1
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["static_audit"]["runtime_assets"] = 7
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["static_audit"]["runtime_source_files"] = 0
    mutations.append(bad)
    bad = copy.deepcopy(valid)
    bad["implementation_seal"]["static_audit"]["unexpected"] = True
    mutations.append(bad)

    with tempfile.TemporaryDirectory() as name:
        target = Path(name) / "sealed-plan.json"
        source_receipts = valid["implementation_seal"]["source_sha256"]
        actual_sha256_file = contract.sha256_file

        def sealed_sha256_file(path) -> str:
            candidate = Path(path)
            try:
                relative = candidate.resolve().relative_to(
                    Path(__file__).resolve().parents[2]
                ).as_posix()
            except ValueError:
                return actual_sha256_file(candidate)
            if relative in source_receipts:
                return source_receipts[relative]
            return actual_sha256_file(candidate)

        with mock.patch.object(
            contract, "sha256_file", side_effect=sealed_sha256_file
        ):
            target.write_text(json.dumps(valid), encoding="utf-8")
            contract.validate_sealed_plan(target, require_sealed=True)
            for mutation in mutations:
                target.write_text(json.dumps(mutation), encoding="utf-8")
                _raises(
                    ValueError,
                    contract.validate_sealed_plan,
                    target,
                    require_sealed=True,
                )


def test_metadata_implementation_commit_is_bound_when_sealed() -> None:
    metadata = _metadata(seed=346, replicate="b", num_envs=8)
    plan = contract.validate_sealed_plan(require_sealed=False)
    fake_sealed = {
        **plan,
        "implementation_seal": {
            "implementation_commit": "2" * 40,
            "source_sha256": {},
        },
    }
    with mock.patch.object(
        contract, "validate_sealed_plan", return_value=fake_sealed
    ):
        _raises(
            ValueError,
            contract._metadata,
            metadata,
            require_sealed_plan=True,
        )


def _post_exit_authority(artifact: dict) -> dict:
    metadata = artifact["metadata"]
    plan = contract.validate_sealed_plan(require_sealed=False)
    preregistration_commit = plan["preregistration_receipt"].get("commit")
    return {
        "status": "passed",
        "collection_commit": metadata["git"]["commit"],
        "implementation_commit": metadata["implementation_commit"],
        "preregistration_tag_commit": (
            preregistration_commit
            if preregistration_commit is not None
            else "2" * 40
        ),
        "collection_tag_commit": metadata["git"]["commit"],
        "superproject_clean": True,
        "submodules_clean": True,
        "git": copy.deepcopy(metadata["git"]),
        "source_sha256": copy.deepcopy(metadata["source_sha256"]),
        "checkpoint_sha256": {
            "v6": {
                "actor.pt": contract.V6_ACTOR_SHA256,
                "task_contract.json": contract.V6_TASK_CONTRACT_SHA256,
                "torch_bridge_state.pt": contract.V6_BRIDGE_STATE_SHA256,
                "frozen_lift_actor.pt": contract.FROZEN_LIFT_ACTOR_SHA256,
            },
            "search": contract.SEARCH_CHECKPOINT_SHA256,
            "fixed_direction": contract.FIXED_DIRECTION_SHA256,
            contract.CANDIDATE40_RESULT: contract.CANDIDATE40_RESULT_SHA256,
            contract.CANDIDATE40_REPORT: contract.CANDIDATE40_REPORT_SHA256,
            contract.CANDIDATE41_PLAN: contract.CANDIDATE41_PLAN_SHA256,
            contract.CANDIDATE41_RESULT: contract.CANDIDATE41_RESULT_SHA256,
            contract.FIXED_DIRECTION_MANIFEST: (
                contract.FIXED_DIRECTION_MANIFEST_SHA256
            ),
        },
        "runtime_asset_sha256": copy.deepcopy(
            metadata["runtime_asset_sha256"]
        ),
        "submodule_commit_sha256": {
            "FlashSAC": contract.FLASHSAC_FORK_COMMIT
        },
    }


def test_complements_and_transactional_final_report_envelope() -> None:
    left = _artifact(seed=347, replicate="a", num_envs=64)
    right = _artifact(seed=347, replicate="b", num_envs=64)
    contract.validate_complementary_artifacts(
        left, right, require_sealed_plan=False
    )
    corrupt = copy.deepcopy(right)
    corrupt["episodes"]["assignment_rank"][0] += 1
    _raises(
        ValueError,
        contract.validate_complementary_artifacts,
        left,
        corrupt,
        require_sealed_plan=False,
    )

    artifact = _artifact()
    core = _report(artifact)
    core_bytes = contract.report_core_bytes(
        core, artifact, require_sealed_plan=False
    )
    assert core_bytes.endswith(b"\n")
    artifact_output = (
        Path(__file__).resolve().parents[2] / contract.SMOKE_ARTIFACT_PATH
    ).resolve()
    final = contract.build_final_report(
        core,
        artifact,
        run_id="8" * 64,
        attempt_id="9" * 64,
        attempt_number=1,
        child_exit_sha256="7" * 64,
        child_exit_size=123,
        post_exit_authority=_post_exit_authority(artifact),
        artifact_sha256="6" * 64,
        artifact_size=456,
        artifact_output=artifact_output,
        require_sealed_plan=False,
    )
    checked = contract.validate_final_report(
        final, artifact, require_sealed_plan=False
    )
    assert contract.final_report_bytes(
        final, artifact, require_sealed_plan=False
    ).endswith(b"\n")
    assert checked["report_core"] == core
    assert checked["identity"] == {
        "run_id": "8" * 64,
        "attempt_id": "9" * 64,
        "attempt_number": 1,
    }
    assert checked["predecessors"] == {
        "50_child_exit.json": {"sha256": "7" * 64, "size": 123}
    }

    mutations: list[dict] = []
    bad = copy.deepcopy(final)
    bad["predecessors"]["50_child_exit.json"]["sha256"] = "0" * 64
    mutations.append(bad)
    bad = copy.deepcopy(final)
    bad["identity"]["attempt_number"] = 2
    mutations.append(bad)
    bad = copy.deepcopy(final)
    bad["report_core"]["vector_steps"] += 1
    mutations.append(bad)
    bad = copy.deepcopy(final)
    bad["post_exit_authority"]["source_sha256"][
        "scripts/flashsac/fake_candidate42.py"
    ] = "0" * 64
    mutations.append(bad)
    bad = copy.deepcopy(final)
    bad["artifact_output"] = str(artifact_output.with_name("foreign.pt"))
    mutations.append(bad)
    bad = copy.deepcopy(final)
    bad["unexpected"] = True
    mutations.append(bad)
    for mutation in mutations:
        _raises(
            ValueError,
            contract.validate_final_report,
            mutation,
            artifact,
            require_sealed_plan=False,
        )
    _raises(
        ValueError,
        contract.validate_report,
        core,
        artifact,
        published=True,
        require_sealed_plan=False,
    )


def test_post_exit_preregistration_tag_must_match_registered_receipt() -> None:
    artifact = _artifact()
    authority = _post_exit_authority(artifact)
    authority["preregistration_tag_commit"] = "2" * 40
    plan = copy.deepcopy(contract.validate_sealed_plan(require_sealed=False))
    plan["preregistration_receipt"]["commit"] = "1" * 40
    assert authority["preregistration_tag_commit"] == "2" * 40
    with mock.patch.object(contract, "validate_sealed_plan", return_value=plan):
        _raises(
            ValueError,
            contract._post_exit_authority,
            authority,
            artifact["metadata"],
        )


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    test_valid_ramp_relock_reramp_and_summary()
    test_exact_float32_authority_sequence_and_control_identity()
    test_corruptions_fail_closed()
    test_plan_identity_fail_closed()
    test_sealed_plan_audit_schema_fails_closed()
    test_metadata_implementation_commit_is_bound_when_sealed()
    test_complements_and_transactional_final_report_envelope()
    test_post_exit_preregistration_tag_must_match_registered_receipt()
    print("candidate42_public_arm_ramp_episode tests passed")


if __name__ == "__main__":
    main()
