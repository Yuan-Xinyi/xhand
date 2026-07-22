#!/usr/bin/env python3
"""Fail-closed Candidate41 public-arm-ramp artifact contract.

Candidate41 preserves Candidate39's complete route and Candidate40's public
stable predicate/count clock.  The only randomized change is the effective
arm7 multiplier after pre-treatment eligibility.  This module records every
option-active pre-eligibility row (plus the unique first-latch witness) and a
contiguous eligibility-to-terminal trace, then independently reconstructs the
linear float32 authority clock and every requested action.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

import candidate40_verified_arm_episode as c40
import candidate41_public_arm_ramp as ramp


ARTIFACT_KIND = "pick_tool_candidate41_public_arm_ramp_episode_v1"
REPORT_KIND = "pick_tool_candidate41_public_arm_ramp_report_v1"
FORMAT_VERSION = 1
COLLECTOR = "collect_candidate41_public_arm_ramp_ab.py"
COLLECTION_CONTRACT = "candidate39_fixed_route_then_public_linear_arm_authority_ramp_v1"
AUTHORITY_CONTRACT = ramp.SUPERVISOR_CONTRACT
DELTA_CONTRACT = c40.DELTA_CONTRACT
VALIDATION_PLAN = "scripts/flashsac/candidate41_public_arm_ramp_development_plan.json"
PLAN_KIND = "pick_tool_candidate41_public_arm_ramp_development_plan_v1"
SEALED_PLAN_STATUS = "sealed_before_smoke_and_collection"
COLLECTION_SEAL_TAG = "pick-tool-candidate41-public-arm-ramp-validation-v1-20260722"
IMPLEMENTATION_SEAL_FIELDS = (
    "status",
    "implementation_commit",
    "source_sha256",
    "simulator_evidence_before_source_seal",
    "simulation_free_tests",
    "static_audit",
)
SIMULATION_FREE_TEST_FIELDS = (
    "status",
    "passed",
    "failed",
    "required_contracts_covered",
)
STATIC_AUDIT_FIELDS = (
    "status",
    "blockers",
    "simulator_invocations",
    "runtime_assets",
    "runtime_source_files",
)

CANDIDATE40_RESULT = "scripts/flashsac/candidate40_verified_arm_validation_result.json"
CANDIDATE40_RESULT_SHA256 = "5d8dde68fdef084ab065ed906047aaa0b0ec0cedcfe90ff9799c296b3180d676"
CANDIDATE40_REPORT = "logs/flashsac/pick_tool/52_c40_verified_arm_development_validation.json"
CANDIDATE40_REPORT_SHA256 = "8c03f040794b390afe4f086093bd08d2aba5ab91b5cc795ca6a4d0416b26aa50"

FIXED_DIRECTION_MANIFEST = c40.FIXED_DIRECTION_MANIFEST
FIXED_DIRECTION_MANIFEST_SHA256 = c40.FIXED_DIRECTION_MANIFEST_SHA256
FIXED_DIRECTION_PATH = c40.FIXED_DIRECTION_PATH
FIXED_DIRECTION_SHA256 = c40.FIXED_DIRECTION_SHA256
FIXED_DIRECTION_PAYLOAD_KIND = c40.FIXED_DIRECTION_PAYLOAD_KIND
FIXED_DIRECTION_SEMANTIC_SHA256 = c40.FIXED_DIRECTION_SEMANTIC_SHA256
V6_CHECKPOINT_PATH = c40.V6_CHECKPOINT_PATH
SEARCH_CHECKPOINT_PATH = c40.SEARCH_CHECKPOINT_PATH
V6_ACTOR_SHA256 = c40.V6_ACTOR_SHA256
V6_TASK_CONTRACT_SHA256 = c40.V6_TASK_CONTRACT_SHA256
V6_BRIDGE_STATE_SHA256 = c40.V6_BRIDGE_STATE_SHA256
FROZEN_LIFT_ACTOR_SHA256 = c40.FROZEN_LIFT_ACTOR_SHA256
FROZEN_LIFT_SEMANTIC_SHA256 = c40.FROZEN_LIFT_SEMANTIC_SHA256
FROZEN_LIFT_SOURCE_ACTOR_SHA256 = c40.FROZEN_LIFT_SOURCE_ACTOR_SHA256
SEARCH_CHECKPOINT_SHA256 = c40.SEARCH_CHECKPOINT_SHA256
FLASHSAC_FORK_COMMIT = c40.FLASHSAC_FORK_COMMIT
REQUIRED_BRANCH = c40.REQUIRED_BRANCH
KIT_ARGS = c40.KIT_ARGS
SMOKE_ARTIFACT_PATH = (
    "logs/flashsac/pick_tool/53_c41_public_arm_ramp_smoke_s340_b/trial.pt"
)
SMOKE_REPORT_PATH = (
    "logs/flashsac/pick_tool/53_c41_public_arm_ramp_smoke_s340_b/trial.json"
)

IMPLEMENTATION_SOURCE_FILES = (
    "scripts/flashsac/candidate41_public_arm_ramp.py",
    "scripts/flashsac/candidate41_public_arm_ramp_test.py",
    "scripts/flashsac/candidate41_public_arm_ramp_episode.py",
    "scripts/flashsac/candidate41_public_arm_ramp_episode_test.py",
    "scripts/flashsac/collect_candidate41_public_arm_ramp_ab.py",
    "scripts/flashsac/collect_candidate41_public_arm_ramp_ab_test.py",
    "scripts/flashsac/analyze_candidate41_public_arm_ramp_validation.py",
    "scripts/flashsac/analyze_candidate41_public_arm_ramp_validation_test.py",
)

OBSERVATION_DIM = c40.OBSERVATION_DIM
ACTION_DIM = c40.ACTION_DIM
ARM_ACTION_DIM = c40.ARM_ACTION_DIM
HAND_ACTION_DIM = c40.HAND_ACTION_DIM
WINDOW_STEPS = c40.WINDOW_STEPS
HANDOFF_MIN_SCORE = c40.HANDOFF_MIN_SCORE
HANDOFF_HOLD_STEPS = c40.HANDOFF_HOLD_STEPS
RAW_Z_ABS_CAP = c40.RAW_Z_ABS_CAP
TOKEN_SCALE = c40.TOKEN_SCALE
DISTAL_SCALE = c40.DISTAL_SCALE
TOKEN_COMPONENT_CAP = c40.TOKEN_COMPONENT_CAP
DISTAL_COMPONENT_CAP = c40.DISTAL_COMPONENT_CAP
PRE_TANH_L2_CAP = c40.PRE_TANH_L2_CAP
MAX_EPISODE_ACTIONS = c40.MAX_EPISODE_ACTIONS
TRACE_METRIC_HORIZON = c40.TRACE_METRIC_HORIZON
PUBLIC_GATE_STATE_CONTRACT = c40.PUBLIC_GATE_STATE_CONTRACT
PUBLIC_LATCH_INDEX = c40.PUBLIC_LATCH_INDEX
TRANSITION_ARM_TOKEN_SLICE = c40.TRANSITION_ARM_TOKEN_SLICE
TRANSITION_DISTAL_SLICE = c40.TRANSITION_DISTAL_SLICE
PROXIMITY_SLICE = c40.PROXIMITY_SLICE
THUMB_PROXIMITY_INDEX = c40.THUMB_PROXIMITY_INDEX

GIT_FIELDS = c40.GIT_FIELDS
RUNTIME_FIELDS = c40.RUNTIME_FIELDS
RUNTIME_PACKAGE_FIELDS = c40.RUNTIME_PACKAGE_FIELDS


EPISODE_FIELDS = (
    "env_slot", "treatment", "assignment_rank", "fixed_z",
    "triggered", "trigger_step", "trigger_score", "first_latch_step",
    "first_eligible_step", "first_positive_authority_step",
    "first_full_authority_step", "first_relock_step", "first_reramp_step",
    "terminal_step", "intervention_steps", "trace_rows",
    "stable_count_max", "positive_authority_rows", "partial_authority_rows",
    "full_authority_rows", "authority_sum", "relock_count", "reramp_count",
    "episode_length", "trajectory_max_force_n", "max_true_clearance_m",
    "first_eligible_clearance_m", "eligible", "latched_within_window",
    "success", "failure", "time_out", "dropped", "unsafe_force",
    "unlatched_clearance_ge_5cm", "ever_grasped", "ever_clearance_ge_20cm",
    "option_active_pre_eligible_unlatched_clearance_ge_5cm",
    "pre_eligibility_action_violations", "hand_invariance_violations",
    "treatment_arm_ramp_violations", "authority_clock_violations",
    "control_route_violations", "task_action_reconstruction_violations",
    "fixed_residual_budget_violations", "action_bound_violations",
    "new_abs_action_ge_0999",
)

PRE_STEP_FIELDS = (
    "pre_env_slot", "pre_episode_step", "pre_close_age",
    "pre_option_active", "pre_public_latch_before", "pre_residual_active",
    "pre_trigger_witness", "pre_first_latch_witness", "pre_observation",
    "pre_applied_delta", "pre_baseline_action", "pre_common_action",
    "pre_requested_action", "pre_transition_observation", "pre_task_action",
    "pre_transition_public_latch", "pre_transition_grasped",
    "pre_transition_true_clearance_m", "pre_grasp_quality",
    "pre_hold_quality", "pre_max_force_n", "pre_transition_done",
)

STEP_FIELDS = (
    "row_env_slot", "row_episode_step", "row_eligible_age",
    "row_observation", "row_transition_observation",
    "row_public_force_counters", "row_option_active",
    "row_public_latch_before", "row_stable_current",
    "row_public_grasp_quality", "row_public_hold_quality",
    "row_public_max_force_strength", "row_stable_count_before",
    "row_stable_count_after", "row_authority_scale",
    "row_common_action", "row_requested_action", "row_task_action",
    "row_arm_target_error", "row_arm_joint_velocity",
    "row_arm_target_error_abs_max", "row_arm_joint_speed_abs_max",
    "row_pre_action_true_clearance_m", "row_treatment", "row_activated",
    "row_first_eligible", "row_relock", "row_reramp",
    "row_transition_public_latch", "row_transition_grasped",
    "row_transition_grasp_quality", "row_transition_hold_quality",
    "row_transition_max_force_n", "row_transition_object_lin_speed",
    "row_transition_object_ang_speed", "row_transition_true_clearance_m",
    "row_transition_done",
)

_EPISODE_BOOL = {
    "treatment", "triggered", "eligible", "latched_within_window",
    "success", "failure", "time_out", "dropped", "unsafe_force",
    "unlatched_clearance_ge_5cm", "ever_grasped",
    "ever_clearance_ge_20cm",
    "option_active_pre_eligible_unlatched_clearance_ge_5cm",
    "new_abs_action_ge_0999",
}
_EPISODE_LONG = {
    "env_slot", "assignment_rank", "trigger_step", "first_latch_step",
    "first_eligible_step", "first_positive_authority_step",
    "first_full_authority_step", "first_relock_step", "first_reramp_step",
    "terminal_step", "intervention_steps", "trace_rows",
    "stable_count_max", "positive_authority_rows", "partial_authority_rows",
    "full_authority_rows", "relock_count", "reramp_count", "episode_length",
    "pre_eligibility_action_violations", "hand_invariance_violations",
    "treatment_arm_ramp_violations", "authority_clock_violations",
    "control_route_violations", "task_action_reconstruction_violations",
    "fixed_residual_budget_violations", "action_bound_violations",
}
_PRE_BOOL = {
    "pre_option_active", "pre_public_latch_before", "pre_residual_active",
    "pre_trigger_witness", "pre_first_latch_witness",
    "pre_transition_public_latch", "pre_transition_grasped",
    "pre_transition_done",
}
_PRE_LONG = c40._PRE_LONG
_STEP_BOOL = {
    "row_option_active", "row_public_latch_before", "row_stable_current",
    "row_treatment", "row_activated", "row_first_eligible", "row_relock",
    "row_reramp", "row_transition_public_latch", "row_transition_grasped",
    "row_transition_done",
}
_STEP_LONG = {
    "row_env_slot", "row_episode_step", "row_eligible_age",
    "row_stable_count_before", "row_stable_count_after",
}

METADATA_FIELDS = (
    "kind", "version", "validation_only", "first_episode_only",
    "network_updates", "collector", "collection_contract",
    "authority_contract", "delta_contract", "assignment_contract",
    "assignment_salt", "assignment_mask_sha256", "handoff_min_score",
    "handoff_hold_steps", "window_steps", "verify_steps", "token_scale",
    "distal_scale", "raw_z_abs_cap", "token_component_cap",
    "distal_component_cap", "pre_tanh_l2_cap", "trace_metric_horizon",
    "public_gate_state_contract", "public_safe_force_strength_max",
    "observation_dim", "action_dim", "max_episode_actions", "kit_args",
    "seed", "replicate", "num_envs", "v6_checkpoint", "search_checkpoint",
    "validation_plan", "implementation_commit", "preregistration_plan_sha256",
    "validation_plan_sha256", "candidate40_result",
    "candidate40_result_sha256", "candidate40_report",
    "candidate40_report_sha256", "fixed_direction_manifest",
    "fixed_direction_manifest_sha256", "fixed_direction_path",
    "fixed_direction_sha256", "fixed_direction_payload_kind",
    "fixed_direction_semantic_sha256", "smoke_artifact_path",
    "smoke_report_path", "smoke_artifact_sha256", "smoke_report_sha256",
    "v6_actor_sha256", "v6_task_contract_sha256", "v6_bridge_state_sha256",
    "frozen_lift_actor_sha256", "frozen_lift_semantic_sha256",
    "frozen_lift_source_actor_sha256", "search_checkpoint_sha256",
    "common_fixed_residual_all_slots", "source_manifest_sha256",
    "runtime_asset_manifest_sha256", "source_sha256", "runtime_asset_sha256",
    "flashsac_upstream_commit", "flashsac_fork_commit", "git", "runtime",
)

REPORT_FIELDS = (
    "kind", "status", "collector", "seed", "replicate", "num_envs",
    "vector_steps", "v6_actor_sha256", "search_checkpoint_sha256",
    "fixed_direction_sha256", "summary",
)
PUBLISHED_REPORT_FIELDS = (*REPORT_FIELDS, "artifact_sha256", "artifact_output")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


sha256_file = c40.sha256_file
manifest_sha256 = c40.manifest_sha256
expected_fixed_z = c40.expected_fixed_z
expected_applied_delta = c40.expected_applied_delta
expected_fixed_residual_action = c40.expected_fixed_residual_action
validate_fixed_direction_payload = c40.validate_fixed_direction_payload
load_fixed_direction = c40.load_fixed_direction
reconstruct_task_action = c40.reconstruct_task_action


def assignment_mask_sha256(mask: torch.Tensor) -> str:
    return ramp.assignment_mask_sha256(mask)


def _sha(value: Any, name: str) -> str:
    return c40._sha(value, name)


def _git_sha(value: Any, name: str) -> str:
    return c40._git_sha(value, name)


def _sha_manifest(value: Any, name: str) -> dict[str, str]:
    return c40._sha_manifest(value, name)


def _strict_json(value: Any, name: str = "value") -> Any:
    return c40._strict_json(value, name)


def validate_sealed_plan(
    path: str | os.PathLike[str] | None = None,
    *,
    require_sealed: bool = True,
) -> dict[str, Any]:
    plan_path = _root() / VALIDATION_PLAN if path is None else Path(path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Candidate41 plan: {plan_path}") from error
    if not isinstance(plan, dict) or plan.get("kind") != PLAN_KIND:
        raise ValueError("Candidate41 plan identity changed")
    status = plan.get("status")
    allowed = {
        "preregistered_before_implementation_or_simulator_evidence",
        SEALED_PLAN_STATUS,
    }
    if status not in allowed or (require_sealed and status != SEALED_PLAN_STATUS):
        raise ValueError("Candidate41 plan is not in an allowed sealed state")
    assignment = plan.get("assignment")
    intervention = plan.get("candidate41_intervention_contract")
    execution = plan.get("execution")
    immutable = plan.get("immutable_policies_and_payloads")
    causal = plan.get("causal_basis")
    if not all(
        isinstance(section, dict)
        for section in (assignment, intervention, execution, immutable, causal)
    ):
        raise ValueError("Candidate41 plan omitted a contract section")
    if (
        assignment.get("contract") != ramp.ASSIGNMENT_CONTRACT
        or assignment.get("salt") != ramp.ASSIGNMENT_SALT
        or assignment.get("num_envs") != 64
    ):
        raise ValueError("Candidate41 assignment contract changed")
    if intervention.get("name") != AUTHORITY_CONTRACT or intervention.get(
        "primary_metric_horizon_actions"
    ) != TRACE_METRIC_HORIZON:
        raise ValueError("Candidate41 intervention identity changed")
    if execution.get("development_seeds") != [341, 342] or execution.get(
        "run_order"
    ) != ["341a", "341b", "342b", "342a"] or execution.get(
        "num_envs_per_run"
    ) != 64 or execution.get("kit_args") != KIT_ARGS:
        raise ValueError("Candidate41 development run identity changed")
    smoke = execution.get("non_evidence_smoke", {})
    if (smoke.get("seed"), smoke.get("replicate"), smoke.get("num_envs")) != (
        340,
        "b",
        8,
    ):
        raise ValueError("Candidate41 smoke identity changed")
    expected_immutable = {
        "v6_checkpoint": V6_CHECKPOINT_PATH,
        "v6_actor_sha256": V6_ACTOR_SHA256,
        "v6_task_contract_sha256": V6_TASK_CONTRACT_SHA256,
        "v6_bridge_state_sha256": V6_BRIDGE_STATE_SHA256,
        "frozen_lift_actor_sha256": FROZEN_LIFT_ACTOR_SHA256,
        "frozen_lift_semantic_sha256": FROZEN_LIFT_SEMANTIC_SHA256,
        "frozen_lift_source_actor_sha256": FROZEN_LIFT_SOURCE_ACTOR_SHA256,
        "search_checkpoint": SEARCH_CHECKPOINT_PATH,
        "search_checkpoint_sha256": SEARCH_CHECKPOINT_SHA256,
        "candidate39_fixed_direction_manifest": FIXED_DIRECTION_MANIFEST,
        "candidate39_fixed_direction_manifest_sha256": FIXED_DIRECTION_MANIFEST_SHA256,
        "candidate39_fixed_z_payload": FIXED_DIRECTION_PATH,
        "candidate39_fixed_z_payload_sha256": FIXED_DIRECTION_SHA256,
        "flashsac_fork_commit": FLASHSAC_FORK_COMMIT,
        "network_updates": 0,
    }
    for name, expected in expected_immutable.items():
        if immutable.get(name) != expected:
            raise ValueError(f"Candidate41 immutable receipt changed: {name}")
    result = causal.get("candidate40_result_manifest", {})
    report = causal.get("candidate40_combined_report", {})
    if (result.get("path"), result.get("sha256")) != (
        CANDIDATE40_RESULT,
        CANDIDATE40_RESULT_SHA256,
    ) or (report.get("path"), report.get("sha256")) != (
        CANDIDATE40_REPORT,
        CANDIDATE40_REPORT_SHA256,
    ):
        raise ValueError("Candidate40 causal receipts changed")
    receipt = plan.get("preregistration_receipt")
    if not isinstance(receipt, dict) or receipt.get("tag") != (
        "pick-tool-candidate41-public-arm-ramp-plan-v1-20260722"
    ):
        raise ValueError("Candidate41 preregistration receipt changed")
    if receipt.get("sha256") is not None:
        _sha(receipt["sha256"], "preregistration_receipt.sha256")
    seal_requirements = plan.get("implementation_seal_requirements")
    if not isinstance(seal_requirements, dict):
        raise ValueError("Candidate41 plan omitted implementation seal requirements")
    required_tests = seal_requirements.get("required_simulation_free_tests")
    if (
        not isinstance(required_tests, list)
        or not required_tests
        or any(not isinstance(item, str) or not item for item in required_tests)
        or len(set(required_tests)) != len(required_tests)
    ):
        raise ValueError("Candidate41 required simulation-free tests changed")
    if require_sealed:
        _git_sha(receipt.get("commit"), "preregistration_receipt.commit")
        _sha(receipt.get("sha256"), "preregistration_receipt.sha256")
        seal = plan.get("implementation_seal")
        if (
            not isinstance(seal, dict)
            or set(seal) != set(IMPLEMENTATION_SEAL_FIELDS)
            or seal.get("status") != "complete_without_simulator_evidence"
        ):
            raise ValueError("Candidate41 implementation is not sealed")
        _git_sha(seal.get("implementation_commit"), "implementation_commit")
        sources = _sha_manifest(seal.get("source_sha256"), "implementation sources")
        if set(sources) != set(IMPLEMENTATION_SOURCE_FILES):
            raise ValueError("Candidate41 implementation source set changed")
        for relative, digest in sources.items():
            if sha256_file(_root() / relative) != digest:
                raise ValueError(f"Candidate41 sealed source changed: {relative}")
        if seal.get("simulator_evidence_before_source_seal") is not False:
            raise ValueError("Candidate41 source seal followed simulator evidence")
        tests = seal.get("simulation_free_tests")
        if (
            not isinstance(tests, dict)
            or set(tests) != set(SIMULATION_FREE_TEST_FIELDS)
            or tests.get("status") != "passed"
            or not isinstance(tests.get("passed"), int)
            or isinstance(tests.get("passed"), bool)
            or tests["passed"] <= 0
            or not isinstance(tests.get("failed"), int)
            or isinstance(tests.get("failed"), bool)
            or tests["failed"] != 0
            or tests.get("required_contracts_covered") != required_tests
        ):
            raise ValueError("Candidate41 simulation-free test seal changed")
        static_audit = seal.get("static_audit")
        if (
            not isinstance(static_audit, dict)
            or set(static_audit) != set(STATIC_AUDIT_FIELDS)
            or static_audit.get("status") != "passed"
            or type(static_audit.get("blockers")) is not int
            or static_audit["blockers"] != 0
            or type(static_audit.get("simulator_invocations")) is not int
            or static_audit["simulator_invocations"] != 0
            or type(static_audit.get("runtime_assets")) is not int
            or static_audit["runtime_assets"] != 8
            or type(static_audit.get("runtime_source_files")) is not int
            or static_audit["runtime_source_files"] <= 0
        ):
            raise ValueError("Candidate41 static-audit seal changed")
        collection_seal = plan.get("collection_seal")
        if not isinstance(collection_seal, dict) or collection_seal.get("tag") != (
            COLLECTION_SEAL_TAG
        ):
            raise ValueError("Candidate41 collection seal changed")
    return plan


REQUIRED_METADATA = {
    "kind": ARTIFACT_KIND,
    "version": FORMAT_VERSION,
    "validation_only": True,
    "first_episode_only": True,
    "network_updates": 0,
    "collector": COLLECTOR,
    "collection_contract": COLLECTION_CONTRACT,
    "authority_contract": AUTHORITY_CONTRACT,
    "delta_contract": DELTA_CONTRACT,
    "assignment_contract": ramp.ASSIGNMENT_CONTRACT,
    "assignment_salt": ramp.ASSIGNMENT_SALT,
    "handoff_min_score": HANDOFF_MIN_SCORE,
    "handoff_hold_steps": HANDOFF_HOLD_STEPS,
    "window_steps": WINDOW_STEPS,
    "verify_steps": ramp.VERIFY_STEPS,
    "token_scale": TOKEN_SCALE,
    "distal_scale": DISTAL_SCALE,
    "raw_z_abs_cap": RAW_Z_ABS_CAP,
    "token_component_cap": TOKEN_COMPONENT_CAP,
    "distal_component_cap": DISTAL_COMPONENT_CAP,
    "pre_tanh_l2_cap": PRE_TANH_L2_CAP,
    "trace_metric_horizon": TRACE_METRIC_HORIZON,
    "public_gate_state_contract": PUBLIC_GATE_STATE_CONTRACT,
    "public_safe_force_strength_max": ramp.PUBLIC_FORCE_STRENGTH_MAX,
    "observation_dim": OBSERVATION_DIM,
    "action_dim": ACTION_DIM,
    "max_episode_actions": MAX_EPISODE_ACTIONS,
    "kit_args": KIT_ARGS,
    "v6_checkpoint": V6_CHECKPOINT_PATH,
    "search_checkpoint": SEARCH_CHECKPOINT_PATH,
    "validation_plan": VALIDATION_PLAN,
    "candidate40_result": CANDIDATE40_RESULT,
    "candidate40_result_sha256": CANDIDATE40_RESULT_SHA256,
    "candidate40_report": CANDIDATE40_REPORT,
    "candidate40_report_sha256": CANDIDATE40_REPORT_SHA256,
    "fixed_direction_manifest": FIXED_DIRECTION_MANIFEST,
    "fixed_direction_manifest_sha256": FIXED_DIRECTION_MANIFEST_SHA256,
    "fixed_direction_path": FIXED_DIRECTION_PATH,
    "fixed_direction_sha256": FIXED_DIRECTION_SHA256,
    "fixed_direction_payload_kind": FIXED_DIRECTION_PAYLOAD_KIND,
    "fixed_direction_semantic_sha256": FIXED_DIRECTION_SEMANTIC_SHA256,
    "smoke_artifact_path": SMOKE_ARTIFACT_PATH,
    "smoke_report_path": SMOKE_REPORT_PATH,
    "v6_actor_sha256": V6_ACTOR_SHA256,
    "v6_task_contract_sha256": V6_TASK_CONTRACT_SHA256,
    "v6_bridge_state_sha256": V6_BRIDGE_STATE_SHA256,
    "frozen_lift_actor_sha256": FROZEN_LIFT_ACTOR_SHA256,
    "frozen_lift_semantic_sha256": FROZEN_LIFT_SEMANTIC_SHA256,
    "frozen_lift_source_actor_sha256": FROZEN_LIFT_SOURCE_ACTOR_SHA256,
    "search_checkpoint_sha256": SEARCH_CHECKPOINT_SHA256,
    "common_fixed_residual_all_slots": True,
    "flashsac_fork_commit": FLASHSAC_FORK_COMMIT,
}


def _metadata(value: Any, *, require_sealed_plan: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(METADATA_FIELDS):
        raise ValueError(f"metadata fields must be exactly {METADATA_FIELDS}")
    result = _strict_json(dict(value), "metadata")
    plan = validate_sealed_plan(require_sealed=require_sealed_plan)
    for name, expected in REQUIRED_METADATA.items():
        if result[name] != expected or type(result[name]) is not type(expected):
            raise ValueError(f"metadata.{name} differs from Candidate41 contract")
    final_digest = sha256_file(_root() / VALIDATION_PLAN)
    if result["validation_plan_sha256"] != final_digest:
        raise ValueError("metadata final plan digest changed")
    prereg = plan["preregistration_receipt"].get("sha256")
    if result["preregistration_plan_sha256"] != prereg:
        raise ValueError("metadata pristine plan digest changed")
    implementation_commit = _git_sha(
        result["implementation_commit"], "metadata.implementation_commit"
    )
    if require_sealed_plan and implementation_commit != plan[
        "implementation_seal"
    ]["implementation_commit"]:
        raise ValueError("metadata implementation commit changed")
    seed, replicate, count = result["seed"], result["replicate"], result["num_envs"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Candidate41 seed is invalid")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("Candidate41 num_envs is invalid")
    if count == 8 and (seed, replicate) != (340, "b"):
        raise ValueError("8-env smoke must be exact Candidate41 run 340b")
    if count == 64 and seed not in {341, 342}:
        raise ValueError("64-env Candidate41 development uses seeds 341/342")
    if count not in {8, 64} or replicate not in {"a", "b"}:
        raise ValueError("invalid Candidate41 run identity")
    if count == 8:
        if result["smoke_artifact_sha256"] is not None or result[
            "smoke_report_sha256"
        ] is not None:
            raise ValueError("smoke cannot depend on its own receipts")
    else:
        _sha(result["smoke_artifact_sha256"], "smoke_artifact_sha256")
        _sha(result["smoke_report_sha256"], "smoke_report_sha256")
    expected_mask = ramp.exact_balanced_treatment_mask(
        seed=seed, num_envs=count, replicate=replicate
    )
    if result["assignment_mask_sha256"] != ramp.assignment_mask_sha256(
        expected_mask
    ):
        raise ValueError("Candidate41 assignment receipt changed")
    checked_manifests: dict[str, dict[str, str]] = {}
    for name in ("source_sha256", "runtime_asset_sha256"):
        manifest = _sha_manifest(result[name], f"metadata.{name}")
        checked_manifests[name] = manifest
        receipt_name = name.removesuffix("sha256") + "manifest_sha256"
        if result[receipt_name] != manifest_sha256(manifest):
            raise ValueError(f"metadata {name} manifest receipt changed")
    git = result["git"]
    if not isinstance(git, dict) or set(git) != set(GIT_FIELDS):
        raise ValueError("Candidate41 git provenance schema changed")
    if (
        git["branch"] != REQUIRED_BRANCH
        or git["source_files_dirty"]
        or git["flashsac_dirty"]
    ):
        raise ValueError("Candidate41 collection provenance is dirty/wrong branch")
    if type(git["source_files_dirty"]) is not bool or type(
        git["flashsac_dirty"]
    ) is not bool:
        raise TypeError("Candidate41 dirty flags must be bool")
    _git_sha(git["commit"], "metadata.git.commit")
    _git_sha(git["flashsac_commit"], "metadata.git.flashsac_commit")
    if git["flashsac_commit"] != FLASHSAC_FORK_COMMIT:
        raise ValueError("FlashSAC fork commit changed")
    if require_sealed_plan:
        sealed = _sha_manifest(
            plan["implementation_seal"]["source_sha256"], "implementation sources"
        )
        for relative, digest in sealed.items():
            if checked_manifests["source_sha256"].get(relative) != digest:
                raise ValueError(f"metadata omitted sealed source {relative}")
    if result["flashsac_upstream_commit"] != (
        "87edc9061150ae9e962dd84e6544e27a1554b3ab"
    ):
        raise ValueError("FlashSAC upstream commit changed")
    runtime = result["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_FIELDS):
        raise ValueError("Candidate41 runtime provenance schema changed")
    if runtime["seed"] != seed:
        raise ValueError("runtime seed changed")
    if not isinstance(runtime["packages"], dict) or set(runtime["packages"]) != set(
        RUNTIME_PACKAGE_FIELDS
    ):
        raise ValueError("Candidate41 runtime package provenance changed")
    return result


def _tensor(
    name: str, value: Any, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    return c40._tensor(name, value, shape, dtype)


def _table(
    value: Any,
    fields: tuple[str, ...],
    bools: set[str],
    longs: set[str],
    suffix: Mapping[str, tuple[int, ...]],
    *,
    name: str,
    rows: int | None = None,
) -> dict[str, torch.Tensor]:
    return c40._table(value, fields, bools, longs, suffix, name=name, rows=rows)


def _episode_table(value: Any, count: int) -> dict[str, torch.Tensor]:
    return _table(
        value,
        EPISODE_FIELDS,
        _EPISODE_BOOL,
        _EPISODE_LONG,
        {"fixed_z": (HAND_ACTION_DIM,)},
        name="episodes",
        rows=count,
    )


def _pre_table(value: Any) -> dict[str, torch.Tensor]:
    suffix = {
        "pre_observation": (OBSERVATION_DIM,),
        "pre_transition_observation": (OBSERVATION_DIM,),
        "pre_applied_delta": (HAND_ACTION_DIM,),
        "pre_baseline_action": (ACTION_DIM,),
        "pre_common_action": (ACTION_DIM,),
        "pre_requested_action": (ACTION_DIM,),
        "pre_task_action": (ACTION_DIM,),
    }
    return _table(
        value, PRE_STEP_FIELDS, _PRE_BOOL, _PRE_LONG, suffix, name="pre_steps"
    )


def _step_table(value: Any) -> dict[str, torch.Tensor]:
    suffix = {
        "row_observation": (OBSERVATION_DIM,),
        "row_transition_observation": (OBSERVATION_DIM,),
        "row_public_force_counters": (2,),
        "row_common_action": (ACTION_DIM,),
        "row_requested_action": (ACTION_DIM,),
        "row_task_action": (ACTION_DIM,),
        "row_arm_target_error": (ARM_ACTION_DIM,),
        "row_arm_joint_velocity": (ARM_ACTION_DIM,),
    }
    return _table(value, STEP_FIELDS, _STEP_BOOL, _STEP_LONG, suffix, name="steps")


def _validate_episode_headers(
    meta: Mapping[str, Any], ep: Mapping[str, torch.Tensor]
) -> None:
    count = int(meta["num_envs"])
    slots = torch.arange(count)
    if not torch.equal(ep["env_slot"], slots):
        raise ValueError("Candidate41 episode slots are not canonical")
    expected_treatment = ramp.exact_balanced_treatment_mask(
        seed=int(meta["seed"]),
        num_envs=count,
        replicate=str(meta["replicate"]),
    )
    expected_rank = ramp.assignment_rank(seed=int(meta["seed"]), num_envs=count)
    if not torch.equal(ep["treatment"], expected_treatment) or not torch.equal(
        ep["assignment_rank"], expected_rank
    ):
        raise ValueError("Candidate41 assignment differs from sealed design")
    if not torch.equal(ep["fixed_z"], expected_fixed_z().expand(count, -1)):
        raise ValueError("fixed_z must be identical in every Candidate41 slot")

    lengths = ep["episode_length"]
    if bool(((lengths < 1) | (lengths > MAX_EPISODE_ACTIONS)).any()) or not torch.equal(
        ep["terminal_step"], lengths - 1
    ):
        raise ValueError("Candidate41 terminal clocks changed")
    outcomes = ep["success"].long() + ep["failure"].long() + ep["time_out"].long()
    if not torch.equal(outcomes, torch.ones_like(outcomes)):
        raise ValueError("Candidate41 outcomes are not exclusive/exhaustive")
    if bool((ep["time_out"] & (lengths != MAX_EPISODE_ACTIONS)).any()):
        raise ValueError("Candidate41 timeout is outside the authored horizon")
    authored_failure = (
        ep["dropped"] | ep["unsafe_force"] | ep["unlatched_clearance_ge_5cm"]
    )
    if not torch.equal(ep["failure"], authored_failure):
        raise ValueError("Candidate41 failure sources changed")
    if bool(
        (
            ep["success"]
            & (
                (~ep["ever_grasped"])
                | (~ep["ever_clearance_ge_20cm"])
                | authored_failure
            )
        ).any()
    ):
        raise ValueError("Candidate41 strict success truth changed")
    if not torch.equal(
        ep["ever_clearance_ge_20cm"], ep["max_true_clearance_m"] >= 0.20
    ):
        raise ValueError("Candidate41 true20 aggregate changed")
    if bool(
        (
            ep["unlatched_clearance_ge_5cm"]
            & (ep["max_true_clearance_m"] < 0.05)
        ).any()
    ):
        raise ValueError("Candidate41 U5 is below 5 cm maximum clearance")
    if bool((ep["trajectory_max_force_n"] < 0).any()):
        raise ValueError("Candidate41 episode force maximum is negative")

    triggered = ep["triggered"]
    if not torch.equal(triggered, ep["trigger_step"] >= 0):
        raise ValueError("Candidate41 trigger flag/clock disagree")
    if bool(((ep["trigger_step"] < -1) | (ep["trigger_step"] >= lengths)).any()):
        raise ValueError("Candidate41 trigger clock is outside the episode")
    if bool(
        (triggered & (ep["trigger_step"] < HANDOFF_HOLD_STEPS - 1)).any()
    ):
        raise ValueError("Candidate41 trigger violates q.30/h4")
    if bool(((ep["trigger_score"] < 0) | (ep["trigger_score"] > 1)).any()):
        raise ValueError("Candidate41 trigger score escaped [0,1]")
    if bool((ep["trigger_score"][~triggered] != 0).any()) or bool(
        (ep["trigger_score"][triggered] < HANDOFF_MIN_SCORE).any()
    ):
        raise ValueError("Candidate41 trigger score changed")

    eligible = ep["eligible"]
    clock_fields = (
        "first_latch_step",
        "first_eligible_step",
        "first_positive_authority_step",
        "first_full_authority_step",
        "first_relock_step",
        "first_reramp_step",
    )
    for field in clock_fields:
        value = ep[field]
        if bool(((value < -1) | (value >= lengths)).any()):
            raise ValueError(f"Candidate41 {field} is outside the episode")
    if not torch.equal(eligible, ep["first_eligible_step"] >= 0):
        raise ValueError("Candidate41 eligibility flag/clock disagree")
    if bool((eligible & ~triggered).any()) or bool(
        (ep["first_eligible_step"][eligible] < ep["trigger_step"][eligible]).any()
    ):
        raise ValueError("Candidate41 eligibility precedes common handoff")
    if bool((eligible & (ep["first_latch_step"] < 0)).any()) or bool(
        (
            ep["first_latch_step"][eligible]
            >= ep["first_eligible_step"][eligible]
        ).any()
    ):
        raise ValueError("Candidate41 first latch/eligibility clocks changed")
    if bool((ep["trace_rows"][~eligible] != 0).any()) or not torch.equal(
        ep["trace_rows"][eligible],
        lengths[eligible] - ep["first_eligible_step"][eligible],
    ):
        raise ValueError("Candidate41 trace does not span eligibility to terminal")
    if bool((ep["intervention_steps"] < 0).any()) or bool(
        (ep["intervention_steps"] > WINDOW_STEPS).any()
    ):
        raise ValueError("Candidate41 residual intervention count changed")
    if bool((ep["stable_count_max"] < 0).any()) or bool(
        (ep["stable_count_max"] > ramp.VERIFY_STEPS).any()
    ):
        raise ValueError("Candidate41 stable count escaped bounds")
    for field in (
        "positive_authority_rows",
        "partial_authority_rows",
        "full_authority_rows",
        "relock_count",
        "reramp_count",
    ):
        if bool((ep[field] < 0).any()) or bool((ep[field] > ep["trace_rows"]).any()):
            raise ValueError(f"Candidate41 {field} is outside trace bounds")
    if bool((ep["authority_sum"] < 0).any()) or bool(
        (ep["authority_sum"] > ep["trace_rows"].float()).any()
    ):
        raise ValueError("Candidate41 authority sum escaped trace bounds")
    if bool((ep["first_eligible_clearance_m"][~eligible] != 0).any()):
        raise ValueError("ineligible Candidate41 episode has eligibility clearance")
    expected_latched = (
        triggered
        & (ep["first_latch_step"] >= ep["trigger_step"])
        & ((ep["first_latch_step"] - ep["trigger_step"]) < WINDOW_STEPS)
    )
    if not torch.equal(ep["latched_within_window"], expected_latched):
        raise ValueError("Candidate41 latched-window label changed")
    if bool(((ep["first_latch_step"] >= 0) & ~ep["ever_grasped"]).any()) or bool(
        (eligible & ~ep["ever_grasped"]).any()
    ):
        raise ValueError("Candidate41 public latch lacks physical grasp support")


def _validate_pre_trace(
    ep: Mapping[str, torch.Tensor], pre: Mapping[str, torch.Tensor]
) -> None:
    count = int(ep["env_slot"].numel())
    lengths = ep["episode_length"]
    eligible = ep["eligible"]
    pre_env = pre["pre_env_slot"]
    if bool(((pre_env < 0) | (pre_env >= count)).any()):
        raise ValueError("Candidate41 pre trace contains invalid slot")
    if bool(pre["pre_public_latch_before"].any()):
        raise ValueError("Candidate41 pre-treatment row is publicly latched")
    if pre_env.numel():
        step = pre["pre_episode_step"]
        option = pre["pre_option_active"]
        close_age = pre["pre_close_age"]
        first_latch = ep["first_latch_step"][pre_env]
        expected_residual = (
            option
            & (close_age >= 0)
            & (close_age < WINDOW_STEPS)
            & ((first_latch < 0) | (step <= first_latch))
        )
        if not torch.equal(pre["pre_residual_active"], expected_residual):
            raise ValueError("Candidate41 residual window/retirement changed")
        expected_delta = torch.where(
            expected_residual.unsqueeze(-1),
            expected_applied_delta().expand(pre_env.numel(), -1),
            torch.zeros((pre_env.numel(), HAND_ACTION_DIM), dtype=torch.float32),
        )
        if not torch.equal(pre["pre_applied_delta"], expected_delta):
            raise ValueError("Candidate41 fixed residual delta changed")
        order = step * count + pre_env
        if order.numel() > 1 and not bool((order[1:] > order[:-1]).all()):
            raise ValueError("Candidate41 pre trace is not canonical")
        if bool(((step < 0) | (step >= lengths[pre_env])).any()):
            raise ValueError("Candidate41 pre trace clock is outside episode")
        expected_age = torch.where(
            option,
            step - ep["trigger_step"][pre_env],
            torch.full_like(step, -1),
        )
        if not torch.equal(close_age, expected_age) or bool((close_age < -1).any()):
            raise ValueError("Candidate41 CLOSE age changed")
        if bool((pre["pre_observation"][:, PUBLIC_LATCH_INDEX] != 0).any()):
            raise ValueError("Candidate41 pre observation is publicly latched")
        for name in (
            "pre_baseline_action",
            "pre_common_action",
            "pre_requested_action",
            "pre_task_action",
        ):
            if bool((pre[name].abs() > 1).any()):
                raise ValueError(f"Candidate41 {name} escaped action bounds")
        overlaid = expected_fixed_residual_action(pre["pre_baseline_action"])
        expected_common = torch.where(
            expected_residual.unsqueeze(-1), overlaid, pre["pre_baseline_action"]
        )
        if not torch.allclose(
            pre["pre_common_action"], expected_common, rtol=0, atol=2e-6
        ):
            raise ValueError("Candidate41 pre common residual algebra changed")
        if not torch.equal(pre["pre_requested_action"], pre["pre_common_action"]):
            raise ValueError("Candidate41 changed a pre-eligibility action")
        if not torch.equal(
            pre["pre_task_action"],
            reconstruct_task_action(pre["pre_transition_observation"]),
        ):
            raise ValueError("Candidate41 pre task action reconstruction changed")
        transition_latch = (
            pre["pre_transition_observation"][:, PUBLIC_LATCH_INDEX] == 1
        )
        if not torch.equal(pre["pre_transition_public_latch"], transition_latch):
            raise ValueError("Candidate41 pre transition latch changed")
        expected_witness = (
            (ep["first_latch_step"][pre_env] >= 0)
            & (step == ep["first_latch_step"][pre_env])
            & transition_latch
        )
        if not torch.equal(pre["pre_first_latch_witness"], expected_witness):
            raise ValueError("Candidate41 unique first-latch witness changed")
        expected_trigger_witness = (
            ep["triggered"][pre_env]
            & (step == ep["trigger_step"][pre_env])
        )
        if not torch.equal(pre["pre_trigger_witness"], expected_trigger_witness):
            raise ValueError("Candidate41 unique trigger witness changed")
        if not bool(
            (
                option
                | pre["pre_trigger_witness"]
                | pre["pre_first_latch_witness"]
            ).all()
        ):
            raise ValueError("Candidate41 pre trace contains out-of-domain row")
        if not torch.equal(
            pre["pre_transition_done"], step == ep["terminal_step"][pre_env]
        ):
            raise ValueError("Candidate41 pre terminal clock changed")
        for field in ("pre_grasp_quality", "pre_hold_quality"):
            if bool(((pre[field] < 0) | (pre[field] > 1)).any()):
                raise ValueError(f"Candidate41 {field} escaped [0,1]")
        if bool((pre["pre_max_force_n"] < 0).any()):
            raise ValueError("Candidate41 pre force is negative")
        if bool(
            (
                pre["pre_max_force_n"]
                > ep["trajectory_max_force_n"][pre_env] + 1e-5
            ).any()
        ) or bool(
            (
                pre["pre_transition_true_clearance_m"]
                > ep["max_true_clearance_m"][pre_env] + 1e-6
            ).any()
        ):
            raise ValueError("Candidate41 pre truth exceeds episode aggregate")
        if bool(
            (
                pre["pre_transition_grasped"]
                & ~ep["ever_grasped"][pre_env]
            ).any()
        ):
            raise ValueError("Candidate41 pre grasp exceeds episode aggregate")
    if not torch.equal(
        torch.bincount(
            pre_env[pre["pre_residual_active"]], minlength=count
        ),
        ep["intervention_steps"],
    ):
        raise ValueError("Candidate41 pre rows disagree with intervention counts")

    for slot in range(count):
        selected = pre_env == slot
        option_selected = selected & pre["pre_option_active"]
        option_steps = pre["pre_episode_step"][option_selected]
        stop = (
            int(ep["first_eligible_step"][slot])
            if bool(eligible[slot])
            else int(lengths[slot])
        )
        expected_steps = (
            torch.arange(int(ep["trigger_step"][slot]), stop)
            if bool(ep["triggered"][slot])
            else torch.empty(0, dtype=torch.long)
        )
        if not torch.equal(option_steps, expected_steps):
            raise ValueError(
                f"Candidate41 pre-eligibility trace incomplete for env {slot}"
            )
        witness = pre["pre_first_latch_witness"][selected]
        expected_count = int(ep["first_latch_step"][slot] >= 0)
        if int(witness.sum()) != expected_count:
            raise ValueError(f"Candidate41 latch witness count changed for env {slot}")
        if expected_count:
            witness_step = pre["pre_episode_step"][selected][witness][0]
            if int(witness_step) != int(ep["first_latch_step"][slot]):
                raise ValueError(f"Candidate41 latch witness clock changed for env {slot}")
        trigger_witness = pre["pre_trigger_witness"][selected]
        # A trigger that is itself the first eligible decision is already the
        # first post-eligibility trace row.  All earlier triggers (including
        # every triggered ineligible episode) require the explicit pre marker.
        trigger_is_trace_first = bool(eligible[slot]) and (
            int(ep["trigger_step"][slot])
            == int(ep["first_eligible_step"][slot])
        )
        expected_trigger_count = int(
            bool(ep["triggered"][slot]) and not trigger_is_trace_first
        )
        if int(trigger_witness.sum()) != expected_trigger_count:
            raise ValueError(f"Candidate41 trigger witness count changed for env {slot}")
        if expected_trigger_count:
            trigger_witness_step = pre["pre_episode_step"][selected][
                trigger_witness
            ][0]
            if int(trigger_witness_step) != int(ep["trigger_step"][slot]):
                raise ValueError(
                    f"Candidate41 trigger witness clock changed for env {slot}"
                )
        obs = pre["pre_observation"][selected]
        transition = pre["pre_transition_observation"][selected]
        steps = pre["pre_episode_step"][selected]
        if steps.numel() > 1:
            consecutive = steps[1:] == steps[:-1] + 1
            if bool(consecutive.any()) and not torch.equal(
                transition[:-1][consecutive], obs[1:][consecutive]
            ):
                raise ValueError(f"Candidate41 pre continuity changed for env {slot}")
        pre_u5 = (
            pre["pre_option_active"][selected]
            & (~pre["pre_transition_grasped"][selected])
            & (pre["pre_transition_true_clearance_m"][selected] >= 0.05)
        )
        if bool(pre_u5.any()) != bool(
            ep["option_active_pre_eligible_unlatched_clearance_ge_5cm"][slot]
        ):
            raise ValueError(f"Candidate41 pre-eligibility U5 changed for env {slot}")


def _linear_ramp_scale(
    stable: torch.Tensor, count_before: torch.Tensor
) -> torch.Tensor:
    """Reconstruct the sealed float32 count-before/15 authority mapping."""

    denominator = torch.tensor(
        float(ramp.VERIFY_STEPS), dtype=torch.float32, device=count_before.device
    )
    fraction = count_before.to(dtype=torch.float32) / denominator
    return torch.where(stable, fraction, torch.zeros_like(fraction))


def _first_clock(steps: torch.Tensor, mask: torch.Tensor) -> int:
    indices = mask.nonzero(as_tuple=False).squeeze(-1)
    return -1 if indices.numel() == 0 else int(steps[indices[0]])


def _validate_step_trace(
    ep: Mapping[str, torch.Tensor],
    pre: Mapping[str, torch.Tensor],
    st: Mapping[str, torch.Tensor],
) -> None:
    count = int(ep["env_slot"].numel())
    lengths = ep["episode_length"]
    eligible = ep["eligible"]
    env = st["row_env_slot"]
    pre_env = pre["pre_env_slot"]
    if bool(((env < 0) | (env >= count)).any()) or not torch.equal(
        torch.bincount(env, minlength=count), ep["trace_rows"]
    ):
        raise ValueError("Candidate41 trace row counts disagree with episodes")
    if env.numel() == 0 and bool(eligible.any()):
        raise ValueError("eligible Candidate41 artifact has no trace")

    for slot in ep["triggered"].nonzero(as_tuple=False).squeeze(-1).tolist():
        trigger_step = int(ep["trigger_step"][slot])
        pre_trigger = (pre_env == slot) & (
            pre["pre_episode_step"] == trigger_step
        )
        trace_trigger = (env == slot) & (st["row_episode_step"] == trigger_step)
        if int(pre_trigger.sum()) + int(trace_trigger.sum()) != 1:
            raise ValueError(f"Candidate41 trigger witness is not unique for env {slot}")
        observation = (
            pre["pre_observation"][pre_trigger][0]
            if bool(pre_trigger.any())
            else st["row_observation"][trace_trigger][0]
        )
        second_nonthumb = torch.topk(observation[PROXIMITY_SLICE], k=2).values[1]
        expected_score = torch.minimum(
            observation[THUMB_PROXIMITY_INDEX], second_nonthumb
        )
        if not torch.isclose(
            ep["trigger_score"][slot], expected_score, rtol=0, atol=1e-6
        ):
            raise ValueError(f"Candidate41 trigger score changed for env {slot}")

    order = st["row_episode_step"] * count + env
    if order.numel() > 1 and not bool((order[1:] > order[:-1]).all()):
        raise ValueError("Candidate41 trace is not canonical step-major/env-major")
    if not bool(st["row_option_active"].all()) or not bool(
        st["row_activated"].all()
    ):
        raise ValueError("Candidate41 trace contains inactive rows")
    if not torch.equal(st["row_treatment"], ep["treatment"][env]):
        raise ValueError("Candidate41 trace assignment changed")
    if not torch.equal(
        st["row_episode_step"],
        ep["first_eligible_step"][env] + st["row_eligible_age"],
    ):
        raise ValueError("Candidate41 eligible ages changed")
    first = st["row_eligible_age"] == 0
    if not torch.equal(st["row_first_eligible"], first) or not bool(
        st["row_public_latch_before"][first].all()
    ):
        raise ValueError("Candidate41 first-eligibility edge changed")
    if not torch.equal(
        st["row_public_latch_before"],
        st["row_observation"][:, PUBLIC_LATCH_INDEX] == 1,
    ):
        raise ValueError("Candidate41 pre-action latch reconstruction changed")
    if not torch.equal(
        st["row_task_action"],
        reconstruct_task_action(st["row_transition_observation"]),
    ):
        raise ValueError("Candidate41 task action reconstruction changed")
    if not torch.equal(
        st["row_transition_public_latch"],
        st["row_transition_observation"][:, PUBLIC_LATCH_INDEX] == 1,
    ):
        raise ValueError("Candidate41 transition latch reconstruction changed")

    public = ramp.public_stable_state(
        st["row_observation"],
        st["row_public_force_counters"],
        st["row_option_active"],
        torch.ones_like(st["row_option_active"]),
    )
    if not torch.equal(st["row_stable_current"], public.stable):
        raise ValueError("Candidate41 public stable predicate changed")
    if not torch.equal(
        st["row_public_grasp_quality"], public.grasp_quality
    ) or not torch.equal(st["row_public_hold_quality"], public.hold_quality) or not torch.equal(
        st["row_public_max_force_strength"], public.max_force_strength
    ):
        raise ValueError("Candidate41 public verifier telemetry changed")

    before = st["row_stable_count_before"]
    after = st["row_stable_count_after"]
    expected_after = torch.where(
        st["row_stable_current"],
        torch.clamp(before + 1, max=ramp.VERIFY_STEPS),
        torch.zeros_like(before),
    )
    if not torch.equal(after, expected_after) or bool(
        ((before < 0) | (before > ramp.VERIFY_STEPS)).any()
    ):
        raise ValueError("Candidate41 stable-count transition changed")
    raw_scale = _linear_ramp_scale(st["row_stable_current"], before)
    effective_scale = torch.where(
        st["row_treatment"], raw_scale, torch.ones_like(raw_scale)
    )
    if not torch.equal(st["row_authority_scale"], effective_scale):
        raise ValueError("Candidate41 effective authority scale changed")
    if bool(((effective_scale < 0) | (effective_scale > 1)).any()):
        raise ValueError("Candidate41 authority escaped [0,1]")

    common = st["row_common_action"]
    requested = st["row_requested_action"]
    for name, action in (
        ("common", common),
        ("requested", requested),
        ("task", st["row_task_action"]),
    ):
        if bool((action.abs() > 1).any()):
            raise ValueError(f"Candidate41 trace {name} action escaped bounds")
    if not torch.equal(common[:, ARM_ACTION_DIM:], requested[:, ARM_ACTION_DIM:]):
        raise ValueError("Candidate41 changed hand14")
    control = ~st["row_treatment"]
    if bool(control.any()) and not torch.equal(requested[control], common[control]):
        raise ValueError("Candidate41 changed control Candidate39 route")
    treated = st["row_treatment"]
    expected_arm = effective_scale.unsqueeze(-1) * common[:, :ARM_ACTION_DIM]
    if bool(treated.any()) and not torch.equal(
        requested[treated, :ARM_ACTION_DIM], expected_arm[treated]
    ):
        raise ValueError("Candidate41 treatment arm ramp algebra changed")

    expected_error = st["row_arm_target_error"].abs().max(dim=-1).values
    expected_speed = st["row_arm_joint_velocity"].abs().max(dim=-1).values
    if not torch.equal(
        st["row_arm_target_error_abs_max"], expected_error
    ) or not torch.equal(st["row_arm_joint_speed_abs_max"], expected_speed):
        raise ValueError("Candidate41 arm-state summary changed")
    if bool((expected_error < 0).any()) or bool((expected_speed < 0).any()):
        raise ValueError("Candidate41 arm-state telemetry is negative")
    for field in (
        "row_public_grasp_quality",
        "row_public_hold_quality",
        "row_public_max_force_strength",
        "row_transition_grasp_quality",
        "row_transition_hold_quality",
    ):
        if bool(((st[field] < 0) | (st[field] > 1)).any()):
            raise ValueError(f"Candidate41 {field} escaped [0,1]")
    for field in (
        "row_transition_max_force_n",
        "row_transition_object_lin_speed",
        "row_transition_object_ang_speed",
    ):
        if bool((st[field] < 0).any()):
            raise ValueError(f"Candidate41 {field} is negative")
    if bool(
        (
            st["row_transition_max_force_n"]
            > ep["trajectory_max_force_n"][env] + 1e-5
        ).any()
    ):
        raise ValueError("Candidate41 trace force exceeds episode maximum")
    if bool(
        (
            st["row_transition_true_clearance_m"]
            > ep["max_true_clearance_m"][env] + 1e-6
        ).any()
    ) or bool(
        (
            st["row_pre_action_true_clearance_m"]
            > ep["max_true_clearance_m"][env] + 1e-6
        ).any()
    ):
        raise ValueError("Candidate41 trace clearance exceeds episode maximum")
    if bool(
        (st["row_transition_grasped"] & ~ep["ever_grasped"][env]).any()
    ) or bool(
        (
            (st["row_transition_true_clearance_m"] >= 0.20)
            & ~ep["ever_clearance_ge_20cm"][env]
        ).any()
    ):
        raise ValueError("Candidate41 trace truth exceeds episode aggregate")

    violation_fields = (
        "pre_eligibility_action_violations",
        "hand_invariance_violations",
        "treatment_arm_ramp_violations",
        "authority_clock_violations",
        "control_route_violations",
        "task_action_reconstruction_violations",
        "fixed_residual_budget_violations",
        "action_bound_violations",
    )
    for name in violation_fields:
        if bool((ep[name] != 0).any()):
            raise ValueError(f"Candidate41 online audit failed: {name}")

    for slot in range(count):
        selected = env == slot
        slot_steps = st["row_episode_step"][selected]
        if not bool(eligible[slot]):
            if slot_steps.numel():
                raise ValueError(f"ineligible Candidate41 env {slot} has trace")
            continue
        expected_steps = torch.arange(
            int(ep["first_eligible_step"][slot]), int(lengths[slot])
        )
        if not torch.equal(slot_steps, expected_steps):
            raise ValueError(f"Candidate41 trace is incomplete for env {slot}")
        slot_observation = st["row_observation"][selected]
        slot_transition = st["row_transition_observation"][selected]
        if slot_steps.numel() > 1 and not torch.equal(
            slot_transition[:-1], slot_observation[1:]
        ):
            raise ValueError(f"Candidate41 trace continuity changed for env {slot}")
        slot_pre = pre_env == slot
        if bool(slot_pre.any()):
            last_pre = pre["pre_episode_step"][slot_pre][-1]
            if int(last_pre) + 1 == int(slot_steps[0]) and not torch.equal(
                pre["pre_transition_observation"][slot_pre][-1], slot_observation[0]
            ):
                raise ValueError(f"Candidate41 pre/trace continuity changed for env {slot}")

        slot_before = before[selected]
        slot_after = after[selected]
        if int(slot_before[0]) != 0 or (
            slot_before.numel() > 1
            and not torch.equal(slot_before[1:], slot_after[:-1])
        ):
            raise ValueError(f"Candidate41 authority clock discontinuous for env {slot}")
        slot_scale = effective_scale[selected]
        slot_treatment = bool(ep["treatment"][slot])
        positive = slot_scale > 0
        full = slot_scale == 1
        previous_positive = torch.cat(
            (torch.zeros(1, dtype=torch.bool), positive[:-1])
        )
        ever_positive_before = torch.cat(
            (
                torch.zeros(1, dtype=torch.bool),
                positive.to(torch.long).cumsum(0)[:-1] > 0,
            )
        )
        expected_relock = (
            previous_positive & (~positive)
            if slot_treatment
            else torch.zeros_like(positive)
        )
        expected_reramp = (
            positive & (~previous_positive) & ever_positive_before
            if slot_treatment
            else torch.zeros_like(positive)
        )
        if not torch.equal(st["row_relock"][selected], expected_relock):
            raise ValueError(f"Candidate41 relock edge changed for env {slot}")
        if not torch.equal(st["row_reramp"][selected], expected_reramp):
            raise ValueError(f"Candidate41 reramp edge changed for env {slot}")
        done = st["row_transition_done"][selected]
        expected_done = slot_steps == ep["terminal_step"][slot]
        if not torch.equal(done, expected_done) or int(done.sum()) != 1:
            raise ValueError(f"Candidate41 terminal edge changed for env {slot}")

        derived_clocks = {
            "first_positive_authority_step": _first_clock(slot_steps, positive),
            "first_full_authority_step": _first_clock(slot_steps, full),
            "first_relock_step": _first_clock(slot_steps, expected_relock),
            "first_reramp_step": _first_clock(slot_steps, expected_reramp),
        }
        for field, expected in derived_clocks.items():
            if int(ep[field][slot]) != expected:
                raise ValueError(f"Candidate41 episode {field} changed for env {slot}")
        partial = positive & (~full)
        expected_counts = {
            "stable_count_max": int(slot_after.max()),
            "positive_authority_rows": int(positive.sum()),
            "partial_authority_rows": int(partial.sum()),
            "full_authority_rows": int(full.sum()),
            "relock_count": int(expected_relock.sum()),
            "reramp_count": int(expected_reramp.sum()),
        }
        for field, expected in expected_counts.items():
            if int(ep[field][slot]) != expected:
                raise ValueError(f"Candidate41 episode {field} changed for env {slot}")
        # Match the collector's per-action float32 ``+=`` order.  A reduction
        # kernel may use a different summation tree and create a false receipt
        # mismatch on long timeout traces.
        expected_sum = torch.zeros((), dtype=torch.float32)
        for scale in slot_scale:
            expected_sum += scale
        if not torch.equal(ep["authority_sum"][slot], expected_sum):
            raise ValueError(f"Candidate41 authority sum changed for env {slot}")
        first_clearance = st["row_pre_action_true_clearance_m"][selected][0]
        if not torch.equal(ep["first_eligible_clearance_m"][slot], first_clearance):
            raise ValueError(f"Candidate41 first clearance changed for env {slot}")

    no_trace = ~eligible
    for field in (
        "first_positive_authority_step",
        "first_full_authority_step",
        "first_relock_step",
        "first_reramp_step",
    ):
        if bool((ep[field][no_trace] != -1).any()):
            raise ValueError(f"ineligible Candidate41 episode has {field}")
    for field in (
        "stable_count_max",
        "positive_authority_rows",
        "partial_authority_rows",
        "full_authority_rows",
        "relock_count",
        "reramp_count",
        "authority_sum",
    ):
        if bool((ep[field][no_trace] != 0).any()):
            raise ValueError(f"ineligible Candidate41 episode has {field}")

    pre_new = torch.zeros(count, dtype=torch.bool)
    if pre_env.numel():
        pre_elements = (pre["pre_requested_action"].abs() >= 0.999) & (
            pre["pre_baseline_action"].abs() < 0.999
        )
        pre_new.scatter_reduce_(
            0, pre_env, pre_elements.any(dim=-1), reduce="amax"
        )
    trace_new = torch.zeros(count, dtype=torch.bool)
    trace_elements = (requested.abs() >= 0.999) & (common.abs() < 0.999)
    if env.numel():
        trace_new.scatter_reduce_(
            0, env, trace_elements.any(dim=-1), reduce="amax"
        )
    if not torch.equal(ep["new_abs_action_ge_0999"], pre_new | trace_new):
        raise ValueError("Candidate41 new action saturation aggregate changed")


def _validate_semantics(
    meta: Mapping[str, Any],
    ep: Mapping[str, torch.Tensor],
    pre: Mapping[str, torch.Tensor],
    st: Mapping[str, torch.Tensor],
) -> None:
    _validate_episode_headers(meta, ep)
    _validate_pre_trace(ep, pre)
    _validate_step_trace(ep, pre, st)


def validate_artifact(
    artifact: Mapping[str, Any], *, require_sealed_plan: bool = True
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "metadata",
        "episodes",
        "pre_steps",
        "steps",
    }:
        raise ValueError(
            "Candidate41 artifact must contain metadata/episodes/pre_steps/steps"
        )
    meta = _metadata(artifact["metadata"], require_sealed_plan=require_sealed_plan)
    episodes = _episode_table(artifact["episodes"], int(meta["num_envs"]))
    pre = _pre_table(artifact["pre_steps"])
    steps = _step_table(artifact["steps"])
    _validate_semantics(meta, episodes, pre, steps)
    return {
        "metadata": meta,
        "episodes": episodes,
        "pre_steps": pre,
        "steps": steps,
    }


def build_artifact(
    metadata: Mapping[str, Any],
    episodes: Mapping[str, torch.Tensor],
    pre_steps: Mapping[str, torch.Tensor],
    steps: Mapping[str, torch.Tensor],
    *,
    require_sealed_plan: bool = True,
) -> dict[str, Any]:
    payload = {
        "metadata": _strict_json(dict(metadata), "metadata"),
        "episodes": {
            name: episodes[name].detach().cpu().clone() for name in EPISODE_FIELDS
        },
        "pre_steps": {
            name: pre_steps[name].detach().cpu().clone() for name in PRE_STEP_FIELDS
        },
        "steps": {
            name: steps[name].detach().cpu().clone() for name in STEP_FIELDS
        },
    }
    return validate_artifact(payload, require_sealed_plan=require_sealed_plan)


def summarize_artifact(
    artifact: Mapping[str, Any], *, require_sealed_plan: bool = True
) -> dict[str, Any]:
    value = validate_artifact(artifact, require_sealed_plan=require_sealed_plan)
    ep, st = value["episodes"], value["steps"]
    treatment = ep["treatment"]

    def arm(mask: torch.Tensor) -> dict[str, int | float]:
        bool_names = (
            "eligible",
            "success",
            "ever_clearance_ge_20cm",
            "unlatched_clearance_ge_5cm",
            "dropped",
            "unsafe_force",
        )
        return {
            "episodes": int(mask.sum()),
            **{name: int((mask & ep[name]).sum()) for name in bool_names},
            "positive_authority_rows": int(ep["positive_authority_rows"][mask].sum()),
            "partial_authority_rows": int(ep["partial_authority_rows"][mask].sum()),
            "full_authority_rows": int(ep["full_authority_rows"][mask].sum()),
            "authority_sum": float(ep["authority_sum"][mask].sum()),
            "relocks": int(ep["relock_count"][mask].sum()),
            "reramps": int(ep["reramp_count"][mask].sum()),
        }

    return {
        "episodes": int(ep["env_slot"].numel()),
        "pre_step_rows": int(value["pre_steps"]["pre_env_slot"].numel()),
        "trace_rows": int(st["row_env_slot"].numel()),
        "treatment": arm(treatment),
        "control": arm(~treatment),
        "action_audit": {
            name: int(ep[name].sum())
            for name in (
                "pre_eligibility_action_violations",
                "hand_invariance_violations",
                "treatment_arm_ramp_violations",
                "authority_clock_violations",
                "control_route_violations",
                "task_action_reconstruction_violations",
                "fixed_residual_budget_violations",
                "action_bound_violations",
            )
        }
        | {
            "any_new_abs_action_ge_0999": bool(
                ep["new_abs_action_ge_0999"].any()
            )
        },
    }


def validate_report(
    report: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    published: bool = False,
    require_sealed_plan: bool = True,
) -> dict[str, Any]:
    value = validate_artifact(artifact, require_sealed_plan=require_sealed_plan)
    fields = PUBLISHED_REPORT_FIELDS if published else REPORT_FIELDS
    if not isinstance(report, dict) or set(report) != set(fields):
        raise ValueError(f"Candidate41 report fields must be exactly {fields}")
    checked = _strict_json(report, "report")
    meta, ep = value["metadata"], value["episodes"]
    expected = {
        "kind": REPORT_KIND,
        "status": "complete",
        "collector": COLLECTOR,
        "seed": meta["seed"],
        "replicate": meta["replicate"],
        "num_envs": meta["num_envs"],
        "vector_steps": int(ep["episode_length"].max()),
        "v6_actor_sha256": meta["v6_actor_sha256"],
        "search_checkpoint_sha256": meta["search_checkpoint_sha256"],
        "fixed_direction_sha256": meta["fixed_direction_sha256"],
        "summary": summarize_artifact(
            value, require_sealed_plan=require_sealed_plan
        ),
    }
    for name, expected_value in expected.items():
        if checked[name] != expected_value:
            raise ValueError(f"Candidate41 report.{name} differs from artifact")
    if published:
        _sha(checked["artifact_sha256"], "report.artifact_sha256")
        if not isinstance(checked["artifact_output"], str) or not checked[
            "artifact_output"
        ]:
            raise ValueError("Candidate41 artifact_output must be non-empty")
    return checked


def validate_complementary_artifacts(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    require_sealed_plan: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    left = validate_artifact(a, require_sealed_plan=require_sealed_plan)
    right = validate_artifact(b, require_sealed_plan=require_sealed_plan)
    ma, mb = left["metadata"], right["metadata"]
    if {ma["replicate"], mb["replicate"]} != {"a", "b"}:
        raise ValueError("Candidate41 complements must contain a and b")
    for name in METADATA_FIELDS:
        if name not in {"replicate", "assignment_mask_sha256"} and ma[name] != mb[name]:
            raise ValueError(f"Candidate41 complementary metadata differs at {name}")
    ea, eb = left["episodes"], right["episodes"]
    if not torch.equal(ea["treatment"], ~eb["treatment"]):
        raise ValueError("Candidate41 assignments are not exact complements")
    for name in ("env_slot", "assignment_rank", "fixed_z"):
        if not torch.equal(ea[name], eb[name]):
            raise ValueError(f"Candidate41 complementary {name} changed")
    return left, right


def _owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _unlink_if_ours(output: Path, temporary: Path) -> None:
    try:
        if _owned(output) and _owned(temporary) and os.path.samefile(output, temporary):
            output.unlink()
    except FileNotFoundError:
        pass


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    checked = _strict_json(dict(payload), "json payload")
    return (
        json.dumps(checked, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def publish_artifact_and_report_no_clobber(
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
    artifact_output: str | os.PathLike[str],
    report_output: str | os.PathLike[str],
    *,
    require_sealed_plan: bool = True,
) -> str:
    """Durably publish a validated pair without replacing either name."""

    value = validate_artifact(artifact, require_sealed_plan=require_sealed_plan)
    checked_report = validate_report(
        report, value, require_sealed_plan=require_sealed_plan
    )
    artifact_path = Path(os.path.abspath(os.fspath(artifact_output)))
    report_path = Path(os.path.abspath(os.fspath(report_output)))
    if artifact_path == report_path:
        raise ValueError("Candidate41 artifact/report paths must differ")
    for output in (artifact_path, report_path):
        if _owned(output):
            raise FileExistsError(f"Candidate41 output exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[Path] = []
    links: dict[Path, Path] = {}
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{artifact_path.name}.tmp-", dir=artifact_path.parent
        )
        os.close(descriptor)
        artifact_tmp = Path(name)
        temporaries.append(artifact_tmp)
        links[artifact_path] = artifact_tmp
        with artifact_tmp.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(artifact_tmp)
        final_report = {
            **checked_report,
            "artifact_sha256": digest,
            "artifact_output": str(artifact_path),
        }
        validate_report(
            final_report,
            value,
            published=True,
            require_sealed_plan=require_sealed_plan,
        )
        descriptor, name = tempfile.mkstemp(
            prefix=f".{report_path.name}.tmp-", dir=report_path.parent
        )
        report_tmp = Path(name)
        temporaries.append(report_tmp)
        links[report_path] = report_tmp
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(final_report))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(artifact_tmp, artifact_path)
        os.link(report_tmp, report_path)
        for directory in {artifact_path.parent, report_path.parent}:
            directory_descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        return digest
    except BaseException:
        for output, temporary in reversed(tuple(links.items())):
            _unlink_if_ours(output, temporary)
        raise
    finally:
        for temporary in temporaries:
            if _owned(temporary):
                temporary.unlink()


def publish_json_no_clobber(
    payload: Mapping[str, Any], output: str | os.PathLike[str]
) -> None:
    serialized = _json_bytes(payload)
    path = Path(os.path.abspath(os.fspath(output)))
    if _owned(path):
        raise FileExistsError(f"Candidate41 output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        _unlink_if_ours(path, temporary)
        raise
    finally:
        if _owned(temporary):
            temporary.unlink()


def atomic_torch_save(
    payload: Mapping[str, Any], output: str | os.PathLike[str]
) -> None:
    path = Path(os.path.abspath(os.fspath(output)))
    if _owned(path):
        raise FileExistsError(f"Candidate41 output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except BaseException:
        _unlink_if_ours(path, temporary)
        raise
    finally:
        if _owned(temporary):
            temporary.unlink()


def atomic_json_dump(
    payload: Mapping[str, Any], output: str | os.PathLike[str]
) -> None:
    publish_json_no_clobber(payload, output)


__all__ = [
    "ACTION_DIM",
    "ARM_ACTION_DIM",
    "ARTIFACT_KIND",
    "AUTHORITY_CONTRACT",
    "COLLECTOR",
    "EPISODE_FIELDS",
    "FIXED_DIRECTION_PATH",
    "HAND_ACTION_DIM",
    "METADATA_FIELDS",
    "PRE_STEP_FIELDS",
    "REQUIRED_METADATA",
    "REPORT_FIELDS",
    "RUNTIME_FIELDS",
    "RUNTIME_PACKAGE_FIELDS",
    "STEP_FIELDS",
    "TRACE_METRIC_HORIZON",
    "VALIDATION_PLAN",
    "assignment_mask_sha256",
    "atomic_json_dump",
    "atomic_torch_save",
    "build_artifact",
    "expected_applied_delta",
    "expected_fixed_residual_action",
    "expected_fixed_z",
    "load_fixed_direction",
    "manifest_sha256",
    "publish_artifact_and_report_no_clobber",
    "publish_json_no_clobber",
    "reconstruct_task_action",
    "sha256_file",
    "summarize_artifact",
    "validate_artifact",
    "validate_complementary_artifacts",
    "validate_report",
    "validate_sealed_plan",
]
