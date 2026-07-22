#!/usr/bin/env python3
"""Fail-closed Candidate43 D8 public-arm-ramp artifact contract.

Candidate43 preserves Candidate42's route, eligibility, public stability,
same-action hard relock, trace, truth, and transaction schemas.  The only
scientific change is the treatment clock clamp/denominator 15 -> 8.  This
module independently reconstructs the D8 float32 authority clock and every
requested action from the recorded public trace.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

import candidate40_verified_arm_episode as c40
import candidate43_short_arm_ramp as ramp


ARTIFACT_KIND = "pick_tool_candidate43_short_public_arm_ramp_episode_v1"
REPORT_KIND = "pick_tool_candidate43_short_public_arm_ramp_report_core_v1"
FINAL_REPORT_KIND = "pick_tool_candidate42_transactional_public_arm_ramp_final_report_v1"
FORMAT_VERSION = 1
COLLECTOR = "collect_candidate43_short_arm_ramp_ab.py"
COLLECTION_CONTRACT = "candidate39_fixed_route_then_public_short_linear_arm_authority_ramp_d8_v1"
AUTHORITY_CONTRACT = ramp.SUPERVISOR_CONTRACT
DELTA_CONTRACT = c40.DELTA_CONTRACT
VALIDATION_PLAN = (
    "scripts/flashsac/candidate43_short_arm_ramp_development_plan.json"
)
PLAN_KIND = "pick_tool_candidate43_short_public_arm_ramp_development_plan_v1"
# The pristine plan permits filling receipts and adding seals, but does not
# permit rewriting its top-level status.  A complete seal is determined below
# from those added objects, not by mutating this registered value.
SEALED_PLAN_STATUS = "preregistered_before_implementation_or_simulator_evidence"
PREREGISTRATION_TAG = (
    "pick-tool-candidate43-short-arm-ramp-plan-v1-20260723"
)
PREREGISTRATION_COMMIT = "87566ab4cd4af9137387c36c43135d13eb48c323"
PRISTINE_PLAN_SHA256 = (
    "ab2533a6a96d133b1330c1fc7803d781fb515ff392ed68ac99f1a841beea26e8"
)
COLLECTION_SEAL_TAG = (
    "pick-tool-candidate43-short-arm-ramp-validation-v1-20260723"
)
TRANSACTION_CONTRACT = "candidate42_parent_child_fsync_no_clobber_commit_v1"
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

CANDIDATE42_PLAN = (
    "scripts/flashsac/candidate42_transactional_arm_ramp_development_plan.json"
)
CANDIDATE42_PLAN_SHA256 = (
    "242f40d844e6536d8cf030e612a2537e3785ad97a5c9c3d5737ab612e4be60fa"
)
CANDIDATE42_PREREGISTRATION_COMMIT = (
    "6e46f567f7730a1ef296fd794ebaf10ea85dde13"
)
CANDIDATE42_PREREGISTRATION_TAG = (
    "pick-tool-candidate42-transactional-arm-ramp-plan-v1-20260722"
)
CANDIDATE42_COLLECTION_SEAL_COMMIT = (
    "5587ec7f352526b248d8616a63d9160824551806"
)
CANDIDATE42_COLLECTION_SEAL_TAG = (
    "pick-tool-candidate42-transactional-arm-ramp-validation-v1-20260722"
)
CANDIDATE42_RESULT = (
    "scripts/flashsac/candidate42_transactional_arm_ramp_validation_result.json"
)
CANDIDATE42_RESULT_SHA256 = (
    "8552a12c505d3e404ccfdf5bd8b609fba9c88a3c10a3a024b1979ad9d49154de"
)
CANDIDATE42_RESULT_COMMIT = "1afc375ab5546a55971e4c545423262b4a2eed32"
CANDIDATE42_RESULT_TAG = (
    "pick-tool-candidate42-transactional-arm-ramp-validation-rejected-v1-20260723"
)
CANDIDATE42_RESULT_TAG_OBJECT_OID = (
    "516091723b8f44488e22fe64014006fc6dbdcde3"
)
CANDIDATE42_REPORT = (
    "logs/flashsac/pick_tool/54_c42_txn_public_arm_ramp_development_validation.json"
)
CANDIDATE42_REPORT_SHA256 = (
    "1e353c4e1694fe6d5fab032d3466b5b63b9e17f88ec798197802b7238802b8e9"
)

# Exact Candidate42 source-section receipts inherited by the Candidate43 plan.
SOURCE_SECTION_SHA256 = {
    "common_candidate39_route": (
        "3d5d43bea745676c3c62d354d696d32af5009dc65517f1dd09e1f7b7e7f4f17b"
    ),
    "candidate42_intervention_contract": (
        "358a6177fdae453a5882a95fccc4c946c318845f69beaaf553018dbe07a250b6"
    ),
    "estimands": (
        "cf6d4459e4cec94919295965feb22082f3b63e22312e8a02e3c7f65e838e9966"
    ),
    "truth_and_estimator_contract": (
        "f1184e58043619294c9e953ab9585c2ce2c377332b21fa39cdf688d860806863"
    ),
    "development_validation_gates": (
        "9382d341d7c0084b6d0d4194122547d8e89fdac47efb6f9a1e50fed36a6190bc"
    ),
    "immutable_policies_and_payloads": (
        "2ab74509901fff758d38ab474ee6d62c5b19d64719d215ffcf990cc7cec2fed7"
    ),
    "transaction_protocol": (
        "8b6a3fca3ab44eb24e8e5ce8ad648ac5fe9742b585a6c5d2cbda97f6eddb54a4"
    ),
}
# Candidate43's own preregistered scientific sections.  Keep these distinct
# from ``SOURCE_SECTION_SHA256``: the latter deliberately names the sealed
# Candidate42 causal source, while this object names the live D8 contract.
CANDIDATE43_SECTION_SHA256 = {
    "common_candidate39_route": (
        "3d5d43bea745676c3c62d354d696d32af5009dc65517f1dd09e1f7b7e7f4f17b"
    ),
    "candidate43_intervention_contract": (
        "b4ea60bd27c8caca610c5142eab1588b266b80a6f32d1e24d4b59ad6f8ce79ed"
    ),
    "estimands": (
        "1c50159cf489dc24a6427e2ccb484d9a3811fa4b8edc9b42ce36592bd8a6cd78"
    ),
    "truth_and_estimator_contract": (
        "3f2f6791d939b831cb9bac2ba1a53c0fb33d8c6825ad4aa373d7102e275fa50f"
    ),
    "development_validation_gates": (
        "2153d3d2957b9eee0b0002422d8302bd59425763335a546fab321ab9897a862c"
    ),
    "immutable_policies_and_payloads": (
        "2ab74509901fff758d38ab474ee6d62c5b19d64719d215ffcf990cc7cec2fed7"
    ),
    "transaction_protocol": (
        "8b6a3fca3ab44eb24e8e5ce8ad648ac5fe9742b585a6c5d2cbda97f6eddb54a4"
    ),
}
SCIENTIFIC_SECTION_SHA256 = CANDIDATE43_SECTION_SHA256

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
    "logs/flashsac/pick_tool/55_c43_short_arm_ramp_smoke_s352_b/trial.pt"
)
SMOKE_REPORT_PATH = (
    "logs/flashsac/pick_tool/55_c43_short_arm_ramp_smoke_s352_b/trial.json"
)

IMPLEMENTATION_SOURCE_FILES = (
    "scripts/flashsac/candidate43_short_arm_ramp.py",
    "scripts/flashsac/candidate43_short_arm_ramp_episode.py",
    "scripts/flashsac/candidate43_short_arm_ramp_collection.py",
    "scripts/flashsac/collect_candidate43_short_arm_ramp_child.py",
    "scripts/flashsac/collect_candidate43_short_arm_ramp_ab.py",
    "scripts/flashsac/analyze_candidate43_short_arm_ramp_validation.py",
    "scripts/flashsac/candidate43_short_arm_ramp_science_test.py",
    "scripts/flashsac/candidate43_short_arm_ramp_transaction_test.py",
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
    "validation_plan_sha256", "candidate42_plan",
    "candidate42_plan_sha256", "candidate42_preregistration_commit",
    "candidate42_preregistration_tag", "candidate42_collection_seal_commit",
    "candidate42_collection_seal_tag", "candidate42_result",
    "candidate42_result_sha256", "candidate42_result_commit",
    "candidate42_result_tag", "candidate42_result_tag_object_oid",
    "candidate42_report", "candidate42_report_sha256",
    "fixed_direction_manifest",
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
FINAL_REPORT_FIELDS = (
    "transaction_contract", "receipt", "identity", "predecessors",
    "kind", "version", "status", "run_id", "attempt_id", "attempt_number",
    "child_exit_sha256", "post_exit_authority",
    "source_manifest_sha256", "checkpoint_manifest_sha256",
    "runtime_asset_manifest_sha256", "artifact_sha256", "artifact_size",
    "artifact_output", "report_core", "report_core_sha256",
)
TRANSACTION_IDENTITY_FIELDS = ("run_id", "attempt_id", "attempt_number")
PREDECESSOR_FIELDS = ("sha256", "size")
POST_EXIT_AUTHORITY_FIELDS = (
    "status", "collection_commit", "implementation_commit",
    "preregistration_tag_commit", "collection_tag_commit",
    "superproject_clean", "submodules_clean", "git", "source_sha256",
    "checkpoint_sha256", "runtime_asset_sha256", "submodule_commit_sha256",
)


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


def canonical_json_sha256(value: Any) -> str:
    """Hash strict JSON using the preregistered sorted compact encoding."""

    checked = _strict_json(value, "canonical JSON")
    encoded = (
        json.dumps(
            checked,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replace_labels(value: Any, substitutions: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_labels(item, substitutions)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_labels(item, substitutions) for item in value]
    if isinstance(value, str):
        result = value
        for current, source in substitutions:
            result = result.replace(current, source)
        return result
    return value


PROTECTED_PLAN_CANONICAL_SHA256 = (
    "938ecea0d4ba34c57f5b41effc80beebb80bf679d2c27424dabb1aeb7600bc93"
)


def _protected_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the three mutations explicitly allowed after preregistration."""

    protected = _strict_json(dict(plan), "Candidate43 plan")
    receipt = protected.get("preregistration_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("Candidate43 plan omitted preregistration_receipt")
    receipt["commit"] = None
    receipt["sha256"] = None
    protected.pop("implementation_seal", None)
    protected.pop("collection_seal", None)
    return protected


def normalized_scientific_section_sha256(
    plan: Mapping[str, Any],
) -> dict[str, str]:
    """Hash C43's scientific sections without delegating D8 to Candidate42."""

    required = (
        "common_candidate39_route",
        "candidate43_intervention_contract",
        "estimands",
        "truth_and_estimator_contract",
        "development_validation_gates",
        "immutable_policies_and_payloads",
        "transaction_protocol",
    )
    if not isinstance(plan, Mapping) or any(name not in plan for name in required):
        raise ValueError("Candidate43 plan omitted a scientific contract section")
    return {name: canonical_json_sha256(plan[name]) for name in required}


def _validate_candidate42_causal_receipts(
    plan: Mapping[str, Any], source_plan: Mapping[str, Any]
) -> None:
    causal = plan.get("causal_basis")
    if not isinstance(causal, dict):
        raise ValueError("Candidate43 plan omitted causal_basis")
    plan_receipt = causal.get("candidate42_validation_plan")
    result_receipt = causal.get("candidate42_rejection_result")
    report_receipt = causal.get("candidate42_combined_report")
    expected_plan = {
        "path": CANDIDATE42_PLAN,
        "sha256": CANDIDATE42_PLAN_SHA256,
        "pristine_preregistration_sha256": (
            "7b9e98d06aae1c992fb4ae7fe9c283f9b746228ab7d012f8cafcd8b3c20d3727"
        ),
        "preregistration_commit": CANDIDATE42_PREREGISTRATION_COMMIT,
        "preregistration_tag": CANDIDATE42_PREREGISTRATION_TAG,
        "collection_seal_commit": CANDIDATE42_COLLECTION_SEAL_COMMIT,
        "collection_seal_tag": CANDIDATE42_COLLECTION_SEAL_TAG,
    }
    expected_result = {
        "path": CANDIDATE42_RESULT,
        "sha256": CANDIDATE42_RESULT_SHA256,
        "decision": "reject_candidate42",
        "formal_seeds_349_351_touched": False,
        "result_commit": CANDIDATE42_RESULT_COMMIT,
        "result_tag": CANDIDATE42_RESULT_TAG,
        "result_tag_object_oid": CANDIDATE42_RESULT_TAG_OBJECT_OID,
    }
    expected_report = {
        "path": CANDIDATE42_REPORT,
        "sha256": CANDIDATE42_REPORT_SHA256,
        "deterministic_recompute_exact": True,
    }
    if plan_receipt != expected_plan:
        raise ValueError("Candidate43 Candidate42-plan receipt changed")
    if result_receipt != expected_result:
        raise ValueError("Candidate43 Candidate42-result receipt changed")
    if report_receipt != expected_report:
        raise ValueError("Candidate43 Candidate42-report receipt changed")

    inheritance = plan.get("scientific_contract_inheritance")
    if (
        not isinstance(inheritance, dict)
        or inheritance.get("source_candidate") != "Candidate42"
        or inheritance.get("source_plan_path") != CANDIDATE42_PLAN
        or inheritance.get("source_plan_sha256") != CANDIDATE42_PLAN_SHA256
        or inheritance.get("source_section_sha256") != SOURCE_SECTION_SHA256
        or inheritance.get("network_updates") != 0
        or inheritance.get(
            "any_other_numeric_or_algorithmic_substitution_permitted"
        )
        is not False
    ):
        raise ValueError("Candidate43 scientific inheritance receipt changed")
    for name in (
        "immutable_policies_and_payloads",
        "common_candidate39_route",
        "transaction_protocol",
    ):
        if plan.get(name) != source_plan.get(name):
            raise ValueError(f"Candidate43 changed inherited exact object {name}")
        if canonical_json_sha256(source_plan[name]) != SOURCE_SECTION_SHA256[name]:
            raise ValueError(f"sealed Candidate42 source section changed: {name}")


def validate_sealed_plan(
    path: str | os.PathLike[str] | None = None,
    *,
    require_sealed: bool = True,
) -> dict[str, Any]:
    """Validate the exact preregistered plan plus its allowed later receipts."""

    plan_path = _root() / VALIDATION_PLAN if path is None else Path(path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Candidate43 plan: {plan_path}") from error
    if not isinstance(plan, dict) or plan.get("kind") != PLAN_KIND:
        raise ValueError("Candidate43 plan identity changed")
    if plan.get("status") != SEALED_PLAN_STATUS:
        raise ValueError("Candidate43 plan status changed")
    if canonical_json_sha256(_protected_plan(plan)) != (
        PROTECTED_PLAN_CANONICAL_SHA256
    ):
        raise ValueError("Candidate43 protected preregistration content changed")

    source_path = _root() / CANDIDATE42_PLAN
    if sha256_file(source_path) != CANDIDATE42_PLAN_SHA256:
        raise ValueError("sealed Candidate42 plan bytes changed")
    try:
        source_plan = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read sealed Candidate42 plan") from error
    _validate_candidate42_causal_receipts(plan, source_plan)
    if normalized_scientific_section_sha256(plan) != CANDIDATE43_SECTION_SHA256:
        raise ValueError("Candidate43 scientific section receipts changed")

    assignment = plan.get("assignment")
    intervention = plan.get("candidate43_intervention_contract")
    execution = plan.get("execution")
    gates = plan.get("development_validation_gates")
    decision = plan.get("decision_rule")
    if (
        not isinstance(assignment, dict)
        or assignment.get("contract") != ramp.ASSIGNMENT_CONTRACT
        or assignment.get("salt") != ramp.ASSIGNMENT_SALT
        or assignment.get("allowed_run_num_envs") != [8, 64]
    ):
        raise ValueError("Candidate43 assignment contract changed")
    clock = intervention.get("clock_state_machine") if isinstance(
        intervention, dict
    ) else None
    if (
        not isinstance(intervention, dict)
        or intervention.get("name") != AUTHORITY_CONTRACT
        or intervention.get("source_candidate") != "Candidate42"
        or intervention.get("network_updates") != 0
        or intervention.get("primary_metric_horizon_actions")
        != TRACE_METRIC_HORIZON
        or not isinstance(clock, dict)
        or "min(8, stable_count_before+1)" not in clock.get(
            "stable_count_after", ""
        )
        or "float32 8.0" not in clock.get("authority_scale", "")
    ):
        raise ValueError("Candidate43 D8 intervention identity changed")
    if (
        not isinstance(execution, dict)
        or execution.get("development_seeds") != [353, 354]
        or execution.get("run_order") != ["353a", "353b", "354b", "354a"]
        or execution.get("num_envs_per_run") != 64
        or execution.get("kit_args") != KIT_ARGS
    ):
        raise ValueError("Candidate43 development run identity changed")
    smoke = execution.get("non_evidence_smoke", {})
    if (smoke.get("seed"), smoke.get("replicate"), smoke.get("num_envs")) != (
        352,
        "b",
        8,
    ):
        raise ValueError("Candidate43 smoke identity changed")
    if (
        not isinstance(gates, dict)
        or gates.get("all_must_pass") is not True
        or len(gates) - 1 != 32
        or not isinstance(decision, dict)
        or decision.get("scientific_gate_count") != 32
        or decision.get("all_scientific_gates_must_pass") is not True
        or decision.get("formal_seeds") != [355, 356, 357]
        or decision.get("candidate42_formal_seeds_forbidden")
        != [349, 350, 351]
    ):
        raise ValueError("Candidate43 scientific decision contract changed")
    if canonical_json_sha256(plan["transaction_protocol"]) != (
        SOURCE_SECTION_SHA256["transaction_protocol"]
    ):
        raise ValueError("Candidate43 transaction protocol changed")

    receipt = plan.get("preregistration_receipt")
    if not isinstance(receipt, dict) or receipt.get("tag") != PREREGISTRATION_TAG:
        raise ValueError("Candidate43 preregistration receipt changed")
    commit, digest = receipt.get("commit"), receipt.get("sha256")
    if (commit is None) != (digest is None):
        raise ValueError("Candidate43 preregistration receipt is partial")
    if commit is not None and (
        _git_sha(commit, "preregistration_receipt.commit")
        != PREREGISTRATION_COMMIT
        or _sha(digest, "preregistration_receipt.sha256")
        != PRISTINE_PLAN_SHA256
    ):
        raise ValueError("Candidate43 preregistration receipt differs from tag")

    seal_requirements = plan.get("implementation_seal_requirements")
    required_tests = (
        seal_requirements.get("required_simulation_free_tests")
        if isinstance(seal_requirements, dict)
        else None
    )
    if (
        not isinstance(required_tests, list)
        or not required_tests
        or any(not isinstance(item, str) or not item for item in required_tests)
        or len(set(required_tests)) != len(required_tests)
    ):
        raise ValueError("Candidate43 simulation-free requirements changed")
    if require_sealed:
        if commit is None:
            raise ValueError("Candidate43 preregistration receipt is not filled")
        seal = plan.get("implementation_seal")
        if (
            not isinstance(seal, dict)
            or set(seal) != set(IMPLEMENTATION_SEAL_FIELDS)
            or seal.get("status") != "complete_without_simulator_evidence"
        ):
            raise ValueError("Candidate43 implementation is not sealed")
        _git_sha(seal.get("implementation_commit"), "implementation_commit")
        sources = _sha_manifest(seal.get("source_sha256"), "implementation sources")
        if set(sources) != set(IMPLEMENTATION_SOURCE_FILES):
            raise ValueError("Candidate43 implementation source set changed")
        for relative, expected in sources.items():
            if sha256_file(_root() / relative) != expected:
                raise ValueError(f"Candidate43 sealed source changed: {relative}")
        if seal.get("simulator_evidence_before_source_seal") is not False:
            raise ValueError("Candidate43 source seal followed simulator evidence")
        tests = seal.get("simulation_free_tests")
        if (
            not isinstance(tests, dict)
            or set(tests) != set(SIMULATION_FREE_TEST_FIELDS)
            or tests.get("status") != "passed"
            or type(tests.get("passed")) is not int
            or tests["passed"] <= 0
            or type(tests.get("failed")) is not int
            or tests["failed"] != 0
            or tests.get("required_contracts_covered") != required_tests
        ):
            raise ValueError("Candidate43 simulation-free test seal changed")
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
            raise ValueError("Candidate43 static-audit seal changed")
        collection_seal = plan.get("collection_seal")
        if (
            not isinstance(collection_seal, dict)
            or collection_seal.get("tag") != COLLECTION_SEAL_TAG
        ):
            raise ValueError("Candidate43 collection seal changed")
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
    "candidate42_plan": CANDIDATE42_PLAN,
    "candidate42_plan_sha256": CANDIDATE42_PLAN_SHA256,
    "candidate42_preregistration_commit": CANDIDATE42_PREREGISTRATION_COMMIT,
    "candidate42_preregistration_tag": CANDIDATE42_PREREGISTRATION_TAG,
    "candidate42_collection_seal_commit": CANDIDATE42_COLLECTION_SEAL_COMMIT,
    "candidate42_collection_seal_tag": CANDIDATE42_COLLECTION_SEAL_TAG,
    "candidate42_result": CANDIDATE42_RESULT,
    "candidate42_result_sha256": CANDIDATE42_RESULT_SHA256,
    "candidate42_result_commit": CANDIDATE42_RESULT_COMMIT,
    "candidate42_result_tag": CANDIDATE42_RESULT_TAG,
    "candidate42_result_tag_object_oid": CANDIDATE42_RESULT_TAG_OBJECT_OID,
    "candidate42_report": CANDIDATE42_REPORT,
    "candidate42_report_sha256": CANDIDATE42_REPORT_SHA256,
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
            raise ValueError(f"metadata.{name} differs from Candidate43 contract")
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
        raise ValueError("Candidate43 seed is invalid")
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("Candidate43 num_envs is invalid")
    if count == 8 and (seed, replicate) != (352, "b"):
        raise ValueError("8-env smoke must be exact Candidate43 run 352b")
    if count == 64 and seed not in {353, 354}:
        raise ValueError("64-env Candidate43 development uses seeds 353/354")
    if count not in {8, 64} or replicate not in {"a", "b"}:
        raise ValueError("invalid Candidate43 run identity")
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
        raise ValueError("Candidate43 assignment receipt changed")
    checked_manifests: dict[str, dict[str, str]] = {}
    for name in ("source_sha256", "runtime_asset_sha256"):
        manifest = _sha_manifest(result[name], f"metadata.{name}")
        checked_manifests[name] = manifest
        receipt_name = name.removesuffix("sha256") + "manifest_sha256"
        if result[receipt_name] != manifest_sha256(manifest):
            raise ValueError(f"metadata {name} manifest receipt changed")
    git = result["git"]
    if not isinstance(git, dict) or set(git) != set(GIT_FIELDS):
        raise ValueError("Candidate43 git provenance schema changed")
    if (
        git["branch"] != REQUIRED_BRANCH
        or git["source_files_dirty"]
        or git["flashsac_dirty"]
    ):
        raise ValueError("Candidate43 collection provenance is dirty/wrong branch")
    if type(git["source_files_dirty"]) is not bool or type(
        git["flashsac_dirty"]
    ) is not bool:
        raise TypeError("Candidate43 dirty flags must be bool")
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
        raise ValueError("Candidate43 runtime provenance schema changed")
    if runtime["seed"] != seed:
        raise ValueError("runtime seed changed")
    if not isinstance(runtime["packages"], dict) or set(runtime["packages"]) != set(
        RUNTIME_PACKAGE_FIELDS
    ):
        raise ValueError("Candidate43 runtime package provenance changed")
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
        raise ValueError("Candidate43 episode slots are not canonical")
    expected_treatment = ramp.exact_balanced_treatment_mask(
        seed=int(meta["seed"]),
        num_envs=count,
        replicate=str(meta["replicate"]),
    )
    expected_rank = ramp.assignment_rank(seed=int(meta["seed"]), num_envs=count)
    if not torch.equal(ep["treatment"], expected_treatment) or not torch.equal(
        ep["assignment_rank"], expected_rank
    ):
        raise ValueError("Candidate43 assignment differs from sealed design")
    if not torch.equal(ep["fixed_z"], expected_fixed_z().expand(count, -1)):
        raise ValueError("fixed_z must be identical in every Candidate43 slot")

    lengths = ep["episode_length"]
    if bool(((lengths < 1) | (lengths > MAX_EPISODE_ACTIONS)).any()) or not torch.equal(
        ep["terminal_step"], lengths - 1
    ):
        raise ValueError("Candidate43 terminal clocks changed")
    outcomes = ep["success"].long() + ep["failure"].long() + ep["time_out"].long()
    if not torch.equal(outcomes, torch.ones_like(outcomes)):
        raise ValueError("Candidate43 outcomes are not exclusive/exhaustive")
    if bool((ep["time_out"] & (lengths != MAX_EPISODE_ACTIONS)).any()):
        raise ValueError("Candidate43 timeout is outside the authored horizon")
    authored_failure = (
        ep["dropped"] | ep["unsafe_force"] | ep["unlatched_clearance_ge_5cm"]
    )
    if not torch.equal(ep["failure"], authored_failure):
        raise ValueError("Candidate43 failure sources changed")
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
        raise ValueError("Candidate43 strict success truth changed")
    if not torch.equal(
        ep["ever_clearance_ge_20cm"], ep["max_true_clearance_m"] >= 0.20
    ):
        raise ValueError("Candidate43 true20 aggregate changed")
    if bool(
        (
            ep["unlatched_clearance_ge_5cm"]
            & (ep["max_true_clearance_m"] < 0.05)
        ).any()
    ):
        raise ValueError("Candidate43 U5 is below 5 cm maximum clearance")
    if bool((ep["trajectory_max_force_n"] < 0).any()):
        raise ValueError("Candidate43 episode force maximum is negative")

    triggered = ep["triggered"]
    if not torch.equal(triggered, ep["trigger_step"] >= 0):
        raise ValueError("Candidate43 trigger flag/clock disagree")
    if bool(((ep["trigger_step"] < -1) | (ep["trigger_step"] >= lengths)).any()):
        raise ValueError("Candidate43 trigger clock is outside the episode")
    if bool(
        (triggered & (ep["trigger_step"] < HANDOFF_HOLD_STEPS - 1)).any()
    ):
        raise ValueError("Candidate43 trigger violates q.30/h4")
    if bool(((ep["trigger_score"] < 0) | (ep["trigger_score"] > 1)).any()):
        raise ValueError("Candidate43 trigger score escaped [0,1]")
    if bool((ep["trigger_score"][~triggered] != 0).any()) or bool(
        (ep["trigger_score"][triggered] < HANDOFF_MIN_SCORE).any()
    ):
        raise ValueError("Candidate43 trigger score changed")

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
            raise ValueError(f"Candidate43 {field} is outside the episode")
    if not torch.equal(eligible, ep["first_eligible_step"] >= 0):
        raise ValueError("Candidate43 eligibility flag/clock disagree")
    if bool((eligible & ~triggered).any()) or bool(
        (ep["first_eligible_step"][eligible] < ep["trigger_step"][eligible]).any()
    ):
        raise ValueError("Candidate43 eligibility precedes common handoff")
    if bool((eligible & (ep["first_latch_step"] < 0)).any()) or bool(
        (
            ep["first_latch_step"][eligible]
            >= ep["first_eligible_step"][eligible]
        ).any()
    ):
        raise ValueError("Candidate43 first latch/eligibility clocks changed")
    if bool((ep["trace_rows"][~eligible] != 0).any()) or not torch.equal(
        ep["trace_rows"][eligible],
        lengths[eligible] - ep["first_eligible_step"][eligible],
    ):
        raise ValueError("Candidate43 trace does not span eligibility to terminal")
    if bool((ep["intervention_steps"] < 0).any()) or bool(
        (ep["intervention_steps"] > WINDOW_STEPS).any()
    ):
        raise ValueError("Candidate43 residual intervention count changed")
    if bool((ep["stable_count_max"] < 0).any()) or bool(
        (ep["stable_count_max"] > ramp.VERIFY_STEPS).any()
    ):
        raise ValueError("Candidate43 stable count escaped bounds")
    for field in (
        "positive_authority_rows",
        "partial_authority_rows",
        "full_authority_rows",
        "relock_count",
        "reramp_count",
    ):
        if bool((ep[field] < 0).any()) or bool((ep[field] > ep["trace_rows"]).any()):
            raise ValueError(f"Candidate43 {field} is outside trace bounds")
    if bool((ep["authority_sum"] < 0).any()) or bool(
        (ep["authority_sum"] > ep["trace_rows"].float()).any()
    ):
        raise ValueError("Candidate43 authority sum escaped trace bounds")
    if bool((ep["first_eligible_clearance_m"][~eligible] != 0).any()):
        raise ValueError("ineligible Candidate43 episode has eligibility clearance")
    expected_latched = (
        triggered
        & (ep["first_latch_step"] >= ep["trigger_step"])
        & ((ep["first_latch_step"] - ep["trigger_step"]) < WINDOW_STEPS)
    )
    if not torch.equal(ep["latched_within_window"], expected_latched):
        raise ValueError("Candidate43 latched-window label changed")
    if bool(((ep["first_latch_step"] >= 0) & ~ep["ever_grasped"]).any()) or bool(
        (eligible & ~ep["ever_grasped"]).any()
    ):
        raise ValueError("Candidate43 public latch lacks physical grasp support")


def _validate_pre_trace(
    ep: Mapping[str, torch.Tensor], pre: Mapping[str, torch.Tensor]
) -> None:
    count = int(ep["env_slot"].numel())
    lengths = ep["episode_length"]
    eligible = ep["eligible"]
    pre_env = pre["pre_env_slot"]
    if bool(((pre_env < 0) | (pre_env >= count)).any()):
        raise ValueError("Candidate43 pre trace contains invalid slot")
    if bool(pre["pre_public_latch_before"].any()):
        raise ValueError("Candidate43 pre-treatment row is publicly latched")
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
            raise ValueError("Candidate43 residual window/retirement changed")
        expected_delta = torch.where(
            expected_residual.unsqueeze(-1),
            expected_applied_delta().expand(pre_env.numel(), -1),
            torch.zeros((pre_env.numel(), HAND_ACTION_DIM), dtype=torch.float32),
        )
        if not torch.equal(pre["pre_applied_delta"], expected_delta):
            raise ValueError("Candidate43 fixed residual delta changed")
        order = step * count + pre_env
        if order.numel() > 1 and not bool((order[1:] > order[:-1]).all()):
            raise ValueError("Candidate43 pre trace is not canonical")
        if bool(((step < 0) | (step >= lengths[pre_env])).any()):
            raise ValueError("Candidate43 pre trace clock is outside episode")
        expected_age = torch.where(
            option,
            step - ep["trigger_step"][pre_env],
            torch.full_like(step, -1),
        )
        if not torch.equal(close_age, expected_age) or bool((close_age < -1).any()):
            raise ValueError("Candidate43 CLOSE age changed")
        if bool((pre["pre_observation"][:, PUBLIC_LATCH_INDEX] != 0).any()):
            raise ValueError("Candidate43 pre observation is publicly latched")
        for name in (
            "pre_baseline_action",
            "pre_common_action",
            "pre_requested_action",
            "pre_task_action",
        ):
            if bool((pre[name].abs() > 1).any()):
                raise ValueError(f"Candidate43 {name} escaped action bounds")
        overlaid = expected_fixed_residual_action(pre["pre_baseline_action"])
        expected_common = torch.where(
            expected_residual.unsqueeze(-1), overlaid, pre["pre_baseline_action"]
        )
        if not torch.allclose(
            pre["pre_common_action"], expected_common, rtol=0, atol=2e-6
        ):
            raise ValueError("Candidate43 pre common residual algebra changed")
        if not torch.equal(pre["pre_requested_action"], pre["pre_common_action"]):
            raise ValueError("Candidate43 changed a pre-eligibility action")
        if not torch.equal(
            pre["pre_task_action"],
            reconstruct_task_action(pre["pre_transition_observation"]),
        ):
            raise ValueError("Candidate43 pre task action reconstruction changed")
        transition_latch = (
            pre["pre_transition_observation"][:, PUBLIC_LATCH_INDEX] == 1
        )
        if not torch.equal(pre["pre_transition_public_latch"], transition_latch):
            raise ValueError("Candidate43 pre transition latch changed")
        expected_witness = (
            (ep["first_latch_step"][pre_env] >= 0)
            & (step == ep["first_latch_step"][pre_env])
            & transition_latch
        )
        if not torch.equal(pre["pre_first_latch_witness"], expected_witness):
            raise ValueError("Candidate43 unique first-latch witness changed")
        expected_trigger_witness = (
            ep["triggered"][pre_env]
            & (step == ep["trigger_step"][pre_env])
        )
        if not torch.equal(pre["pre_trigger_witness"], expected_trigger_witness):
            raise ValueError("Candidate43 unique trigger witness changed")
        if not bool(
            (
                option
                | pre["pre_trigger_witness"]
                | pre["pre_first_latch_witness"]
            ).all()
        ):
            raise ValueError("Candidate43 pre trace contains out-of-domain row")
        if not torch.equal(
            pre["pre_transition_done"], step == ep["terminal_step"][pre_env]
        ):
            raise ValueError("Candidate43 pre terminal clock changed")
        for field in ("pre_grasp_quality", "pre_hold_quality"):
            if bool(((pre[field] < 0) | (pre[field] > 1)).any()):
                raise ValueError(f"Candidate43 {field} escaped [0,1]")
        if bool((pre["pre_max_force_n"] < 0).any()):
            raise ValueError("Candidate43 pre force is negative")
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
            raise ValueError("Candidate43 pre truth exceeds episode aggregate")
        if bool(
            (
                pre["pre_transition_grasped"]
                & ~ep["ever_grasped"][pre_env]
            ).any()
        ):
            raise ValueError("Candidate43 pre grasp exceeds episode aggregate")
    if not torch.equal(
        torch.bincount(
            pre_env[pre["pre_residual_active"]], minlength=count
        ),
        ep["intervention_steps"],
    ):
        raise ValueError("Candidate43 pre rows disagree with intervention counts")

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
                f"Candidate43 pre-eligibility trace incomplete for env {slot}"
            )
        witness = pre["pre_first_latch_witness"][selected]
        expected_count = int(ep["first_latch_step"][slot] >= 0)
        if int(witness.sum()) != expected_count:
            raise ValueError(f"Candidate43 latch witness count changed for env {slot}")
        if expected_count:
            witness_step = pre["pre_episode_step"][selected][witness][0]
            if int(witness_step) != int(ep["first_latch_step"][slot]):
                raise ValueError(f"Candidate43 latch witness clock changed for env {slot}")
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
            raise ValueError(f"Candidate43 trigger witness count changed for env {slot}")
        if expected_trigger_count:
            trigger_witness_step = pre["pre_episode_step"][selected][
                trigger_witness
            ][0]
            if int(trigger_witness_step) != int(ep["trigger_step"][slot]):
                raise ValueError(
                    f"Candidate43 trigger witness clock changed for env {slot}"
                )
        obs = pre["pre_observation"][selected]
        transition = pre["pre_transition_observation"][selected]
        steps = pre["pre_episode_step"][selected]
        if steps.numel() > 1:
            consecutive = steps[1:] == steps[:-1] + 1
            if bool(consecutive.any()) and not torch.equal(
                transition[:-1][consecutive], obs[1:][consecutive]
            ):
                raise ValueError(f"Candidate43 pre continuity changed for env {slot}")
        pre_u5 = (
            pre["pre_option_active"][selected]
            & (~pre["pre_transition_grasped"][selected])
            & (pre["pre_transition_true_clearance_m"][selected] >= 0.05)
        )
        if bool(pre_u5.any()) != bool(
            ep["option_active_pre_eligible_unlatched_clearance_ge_5cm"][slot]
        ):
            raise ValueError(f"Candidate43 pre-eligibility U5 changed for env {slot}")


def _linear_ramp_scale(
    stable: torch.Tensor, count_before: torch.Tensor
) -> torch.Tensor:
    """Independently reconstruct float32 ``min(count_before, 8) / 8``."""

    denominator = torch.tensor(
        8.0, dtype=torch.float32, device=count_before.device
    )
    fraction = torch.clamp(count_before, max=8).to(dtype=torch.float32) / denominator
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
        raise ValueError("Candidate43 trace row counts disagree with episodes")
    if env.numel() == 0 and bool(eligible.any()):
        raise ValueError("eligible Candidate43 artifact has no trace")

    for slot in ep["triggered"].nonzero(as_tuple=False).squeeze(-1).tolist():
        trigger_step = int(ep["trigger_step"][slot])
        pre_trigger = (pre_env == slot) & (
            pre["pre_episode_step"] == trigger_step
        )
        trace_trigger = (env == slot) & (st["row_episode_step"] == trigger_step)
        if int(pre_trigger.sum()) + int(trace_trigger.sum()) != 1:
            raise ValueError(f"Candidate43 trigger witness is not unique for env {slot}")
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
            raise ValueError(f"Candidate43 trigger score changed for env {slot}")

    order = st["row_episode_step"] * count + env
    if order.numel() > 1 and not bool((order[1:] > order[:-1]).all()):
        raise ValueError("Candidate43 trace is not canonical step-major/env-major")
    if not bool(st["row_option_active"].all()) or not bool(
        st["row_activated"].all()
    ):
        raise ValueError("Candidate43 trace contains inactive rows")
    if not torch.equal(st["row_treatment"], ep["treatment"][env]):
        raise ValueError("Candidate43 trace assignment changed")
    if not torch.equal(
        st["row_episode_step"],
        ep["first_eligible_step"][env] + st["row_eligible_age"],
    ):
        raise ValueError("Candidate43 eligible ages changed")
    first = st["row_eligible_age"] == 0
    if not torch.equal(st["row_first_eligible"], first) or not bool(
        st["row_public_latch_before"][first].all()
    ):
        raise ValueError("Candidate43 first-eligibility edge changed")
    if not torch.equal(
        st["row_public_latch_before"],
        st["row_observation"][:, PUBLIC_LATCH_INDEX] == 1,
    ):
        raise ValueError("Candidate43 pre-action latch reconstruction changed")
    if not torch.equal(
        st["row_task_action"],
        reconstruct_task_action(st["row_transition_observation"]),
    ):
        raise ValueError("Candidate43 task action reconstruction changed")
    if not torch.equal(
        st["row_transition_public_latch"],
        st["row_transition_observation"][:, PUBLIC_LATCH_INDEX] == 1,
    ):
        raise ValueError("Candidate43 transition latch reconstruction changed")

    public = ramp.public_stable_state(
        st["row_observation"],
        st["row_public_force_counters"],
        st["row_option_active"],
        torch.ones_like(st["row_option_active"]),
    )
    if not torch.equal(st["row_stable_current"], public.stable):
        raise ValueError("Candidate43 public stable predicate changed")
    if not torch.equal(
        st["row_public_grasp_quality"], public.grasp_quality
    ) or not torch.equal(st["row_public_hold_quality"], public.hold_quality) or not torch.equal(
        st["row_public_max_force_strength"], public.max_force_strength
    ):
        raise ValueError("Candidate43 public verifier telemetry changed")

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
        raise ValueError("Candidate43 stable-count transition changed")
    raw_scale = _linear_ramp_scale(st["row_stable_current"], before)
    effective_scale = torch.where(
        st["row_treatment"], raw_scale, torch.ones_like(raw_scale)
    )
    if not torch.equal(st["row_authority_scale"], effective_scale):
        raise ValueError("Candidate43 effective authority scale changed")
    if bool(((effective_scale < 0) | (effective_scale > 1)).any()):
        raise ValueError("Candidate43 authority escaped [0,1]")

    common = st["row_common_action"]
    requested = st["row_requested_action"]
    for name, action in (
        ("common", common),
        ("requested", requested),
        ("task", st["row_task_action"]),
    ):
        if bool((action.abs() > 1).any()):
            raise ValueError(f"Candidate43 trace {name} action escaped bounds")
    if not torch.equal(common[:, ARM_ACTION_DIM:], requested[:, ARM_ACTION_DIM:]):
        raise ValueError("Candidate43 changed hand14")
    control = ~st["row_treatment"]
    if bool(control.any()) and not torch.equal(requested[control], common[control]):
        raise ValueError("Candidate43 changed control Candidate39 route")
    treated = st["row_treatment"]
    expected_arm = effective_scale.unsqueeze(-1) * common[:, :ARM_ACTION_DIM]
    if bool(treated.any()) and not torch.equal(
        requested[treated, :ARM_ACTION_DIM], expected_arm[treated]
    ):
        raise ValueError("Candidate43 treatment arm ramp algebra changed")

    expected_error = st["row_arm_target_error"].abs().max(dim=-1).values
    expected_speed = st["row_arm_joint_velocity"].abs().max(dim=-1).values
    if not torch.equal(
        st["row_arm_target_error_abs_max"], expected_error
    ) or not torch.equal(st["row_arm_joint_speed_abs_max"], expected_speed):
        raise ValueError("Candidate43 arm-state summary changed")
    if bool((expected_error < 0).any()) or bool((expected_speed < 0).any()):
        raise ValueError("Candidate43 arm-state telemetry is negative")
    for field in (
        "row_public_grasp_quality",
        "row_public_hold_quality",
        "row_public_max_force_strength",
        "row_transition_grasp_quality",
        "row_transition_hold_quality",
    ):
        if bool(((st[field] < 0) | (st[field] > 1)).any()):
            raise ValueError(f"Candidate43 {field} escaped [0,1]")
    for field in (
        "row_transition_max_force_n",
        "row_transition_object_lin_speed",
        "row_transition_object_ang_speed",
    ):
        if bool((st[field] < 0).any()):
            raise ValueError(f"Candidate43 {field} is negative")
    if bool(
        (
            st["row_transition_max_force_n"]
            > ep["trajectory_max_force_n"][env] + 1e-5
        ).any()
    ):
        raise ValueError("Candidate43 trace force exceeds episode maximum")
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
        raise ValueError("Candidate43 trace clearance exceeds episode maximum")
    if bool(
        (st["row_transition_grasped"] & ~ep["ever_grasped"][env]).any()
    ) or bool(
        (
            (st["row_transition_true_clearance_m"] >= 0.20)
            & ~ep["ever_clearance_ge_20cm"][env]
        ).any()
    ):
        raise ValueError("Candidate43 trace truth exceeds episode aggregate")

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
            raise ValueError(f"Candidate43 online audit failed: {name}")

    for slot in range(count):
        selected = env == slot
        slot_steps = st["row_episode_step"][selected]
        if not bool(eligible[slot]):
            if slot_steps.numel():
                raise ValueError(f"ineligible Candidate43 env {slot} has trace")
            continue
        expected_steps = torch.arange(
            int(ep["first_eligible_step"][slot]), int(lengths[slot])
        )
        if not torch.equal(slot_steps, expected_steps):
            raise ValueError(f"Candidate43 trace is incomplete for env {slot}")
        slot_observation = st["row_observation"][selected]
        slot_transition = st["row_transition_observation"][selected]
        if slot_steps.numel() > 1 and not torch.equal(
            slot_transition[:-1], slot_observation[1:]
        ):
            raise ValueError(f"Candidate43 trace continuity changed for env {slot}")
        slot_pre = pre_env == slot
        if bool(slot_pre.any()):
            last_pre = pre["pre_episode_step"][slot_pre][-1]
            if int(last_pre) + 1 == int(slot_steps[0]) and not torch.equal(
                pre["pre_transition_observation"][slot_pre][-1], slot_observation[0]
            ):
                raise ValueError(f"Candidate43 pre/trace continuity changed for env {slot}")

        slot_before = before[selected]
        slot_after = after[selected]
        if int(slot_before[0]) != 0 or (
            slot_before.numel() > 1
            and not torch.equal(slot_before[1:], slot_after[:-1])
        ):
            raise ValueError(f"Candidate43 authority clock discontinuous for env {slot}")
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
            raise ValueError(f"Candidate43 relock edge changed for env {slot}")
        if not torch.equal(st["row_reramp"][selected], expected_reramp):
            raise ValueError(f"Candidate43 reramp edge changed for env {slot}")
        done = st["row_transition_done"][selected]
        expected_done = slot_steps == ep["terminal_step"][slot]
        if not torch.equal(done, expected_done) or int(done.sum()) != 1:
            raise ValueError(f"Candidate43 terminal edge changed for env {slot}")

        derived_clocks = {
            "first_positive_authority_step": _first_clock(slot_steps, positive),
            "first_full_authority_step": _first_clock(slot_steps, full),
            "first_relock_step": _first_clock(slot_steps, expected_relock),
            "first_reramp_step": _first_clock(slot_steps, expected_reramp),
        }
        for field, expected in derived_clocks.items():
            if int(ep[field][slot]) != expected:
                raise ValueError(f"Candidate43 episode {field} changed for env {slot}")
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
                raise ValueError(f"Candidate43 episode {field} changed for env {slot}")
        # Match the collector's per-action float32 ``+=`` order.  A reduction
        # kernel may use a different summation tree and create a false receipt
        # mismatch on long timeout traces.
        expected_sum = torch.zeros((), dtype=torch.float32)
        for scale in slot_scale:
            expected_sum += scale
        if not torch.equal(ep["authority_sum"][slot], expected_sum):
            raise ValueError(f"Candidate43 authority sum changed for env {slot}")
        first_clearance = st["row_pre_action_true_clearance_m"][selected][0]
        if not torch.equal(ep["first_eligible_clearance_m"][slot], first_clearance):
            raise ValueError(f"Candidate43 first clearance changed for env {slot}")

    no_trace = ~eligible
    for field in (
        "first_positive_authority_step",
        "first_full_authority_step",
        "first_relock_step",
        "first_reramp_step",
    ):
        if bool((ep[field][no_trace] != -1).any()):
            raise ValueError(f"ineligible Candidate43 episode has {field}")
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
            raise ValueError(f"ineligible Candidate43 episode has {field}")

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
        raise ValueError("Candidate43 new action saturation aggregate changed")


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
            "Candidate43 artifact must contain metadata/episodes/pre_steps/steps"
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
    if published:
        raise ValueError(
            "Candidate43 canonical reports require validate_final_report; "
            "31_report_core is never itself published"
        )
    value = validate_artifact(artifact, require_sealed_plan=require_sealed_plan)
    if not isinstance(report, dict) or set(report) != set(REPORT_FIELDS):
        raise ValueError(
            f"Candidate43 report fields must be exactly {REPORT_FIELDS}"
        )
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
            raise ValueError(f"Candidate43 report.{name} differs from artifact")
    return checked


def report_core_bytes(
    report: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    require_sealed_plan: bool = True,
) -> bytes:
    """Return the one canonical byte encoding permitted for 31_report_core."""

    return _json_bytes(
        validate_report(
            report,
            artifact,
            require_sealed_plan=require_sealed_plan,
        )
    )


def _canonical_artifact_output(metadata: Mapping[str, Any]) -> Path:
    root = _root()
    seed = int(metadata["seed"])
    replicate = str(metadata["replicate"])
    if int(metadata["num_envs"]) == 8:
        return (root / SMOKE_ARTIFACT_PATH).resolve()
    return (
        root
        / "logs/flashsac/pick_tool"
        / f"55_c43_short_arm_ramp_dev_s{seed}_{replicate}/trial.pt"
    ).resolve()


def _post_exit_authority(
    value: Any, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        POST_EXIT_AUTHORITY_FIELDS
    ):
        raise ValueError(
            "post_exit_authority fields must be exactly "
            f"{POST_EXIT_AUTHORITY_FIELDS}"
        )
    result = _strict_json(dict(value), "post_exit_authority")
    if (
        result["status"] != "passed"
        or result["superproject_clean"] is not True
        or result["submodules_clean"] is not True
    ):
        raise ValueError("post-exit authority did not pass cleanly")
    collection_commit = _git_sha(
        result["collection_commit"], "post_exit_authority.collection_commit"
    )
    implementation_commit = _git_sha(
        result["implementation_commit"],
        "post_exit_authority.implementation_commit",
    )
    preregistration_tag_commit = _git_sha(
        result["preregistration_tag_commit"],
        "post_exit_authority.preregistration_tag_commit",
    )
    sealed_plan = validate_sealed_plan(require_sealed=False)
    registered_preregistration_commit = sealed_plan[
        "preregistration_receipt"
    ].get("commit")
    if registered_preregistration_commit is not None and (
        preregistration_tag_commit
        != _git_sha(
            registered_preregistration_commit,
            "preregistration_receipt.commit",
        )
    ):
        raise ValueError(
            "post-exit preregistration tag differs from the sealed receipt"
        )
    if _git_sha(
        result["collection_tag_commit"],
        "post_exit_authority.collection_tag_commit",
    ) != collection_commit:
        raise ValueError("post-exit collection tag does not resolve to HEAD")
    if collection_commit != metadata["git"]["commit"] or (
        implementation_commit != metadata["implementation_commit"]
    ):
        raise ValueError("post-exit Git commits differ from artifact authority")
    if result["git"] != metadata["git"]:
        raise ValueError("post-exit Git provenance differs from artifact authority")

    source = _sha_manifest(
        result["source_sha256"], "post_exit_authority.source_sha256"
    )
    runtime_assets = _sha_manifest(
        result["runtime_asset_sha256"],
        "post_exit_authority.runtime_asset_sha256",
    )
    if source != metadata["source_sha256"] or runtime_assets != metadata[
        "runtime_asset_sha256"
    ]:
        raise ValueError("post-exit source/runtime assets differ from artifact")

    checkpoints = result["checkpoint_sha256"]
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError("post-exit checkpoint authority must be a non-empty mapping")
    expected_v6 = {
        "actor.pt": V6_ACTOR_SHA256,
        "task_contract.json": V6_TASK_CONTRACT_SHA256,
        "torch_bridge_state.pt": V6_BRIDGE_STATE_SHA256,
        "frozen_lift_actor.pt": FROZEN_LIFT_ACTOR_SHA256,
    }
    exact_checkpoints = {
        "v6": expected_v6,
        "search": SEARCH_CHECKPOINT_SHA256,
        "fixed_direction": FIXED_DIRECTION_SHA256,
        CANDIDATE42_PLAN: CANDIDATE42_PLAN_SHA256,
        CANDIDATE42_RESULT: CANDIDATE42_RESULT_SHA256,
        CANDIDATE42_REPORT: CANDIDATE42_REPORT_SHA256,
        FIXED_DIRECTION_MANIFEST: FIXED_DIRECTION_MANIFEST_SHA256,
    }
    for name, expected in exact_checkpoints.items():
        if checkpoints.get(name) != expected:
            raise ValueError(f"post-exit checkpoint authority changed: {name}")
    submodules = result["submodule_commit_sha256"]
    if not isinstance(submodules, dict):
        raise ValueError("submodule_commit_sha256 must be a mapping")
    for name, commit in submodules.items():
        if not isinstance(name, str) or not name:
            raise ValueError("submodule path must be a non-empty string")
        _git_sha(commit, f"post_exit_authority.submodule[{name!r}]")
    return result


def build_final_report(
    report_core: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    run_id: str,
    attempt_id: str,
    attempt_number: int,
    child_exit_sha256: str,
    child_exit_size: int,
    post_exit_authority: Mapping[str, Any],
    artifact_sha256: str,
    artifact_size: int,
    artifact_output: str | os.PathLike[str],
    require_sealed_plan: bool = True,
) -> dict[str, Any]:
    """Build 55_final_report, the exact canonical ``trial.json`` payload."""

    checked_artifact = validate_artifact(
        artifact, require_sealed_plan=require_sealed_plan
    )
    checked_core = validate_report(
        report_core,
        checked_artifact,
        require_sealed_plan=require_sealed_plan,
    )
    authority = _post_exit_authority(
        post_exit_authority, checked_artifact["metadata"]
    )
    payload = {
        "transaction_contract": TRANSACTION_CONTRACT,
        "receipt": "55_final_report.json",
        "identity": {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
        },
        "predecessors": {
            "50_child_exit.json": {
                "sha256": child_exit_sha256,
                "size": child_exit_size,
            }
        },
        "kind": FINAL_REPORT_KIND,
        "version": FORMAT_VERSION,
        "status": "complete",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "child_exit_sha256": child_exit_sha256,
        "post_exit_authority": authority,
        "source_manifest_sha256": manifest_sha256(authority["source_sha256"]),
        "checkpoint_manifest_sha256": canonical_json_sha256(
            authority["checkpoint_sha256"]
        ),
        "runtime_asset_manifest_sha256": manifest_sha256(
            authority["runtime_asset_sha256"]
        ),
        "artifact_sha256": artifact_sha256,
        "artifact_size": artifact_size,
        "artifact_output": os.fspath(artifact_output),
        "report_core": checked_core,
        "report_core_sha256": hashlib.sha256(
            report_core_bytes(
                checked_core,
                checked_artifact,
                require_sealed_plan=require_sealed_plan,
            )
        ).hexdigest(),
    }
    return validate_final_report(
        payload,
        checked_artifact,
        require_sealed_plan=require_sealed_plan,
    )


def validate_final_report(
    report: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    require_sealed_plan: bool = True,
) -> dict[str, Any]:
    """Validate the 55/canonical envelope and its embedded 31 report core."""

    checked_artifact = validate_artifact(
        artifact, require_sealed_plan=require_sealed_plan
    )
    if not isinstance(report, Mapping) or set(report) != set(FINAL_REPORT_FIELDS):
        raise ValueError(
            f"Candidate43 final report fields must be exactly {FINAL_REPORT_FIELDS}"
        )
    result = _strict_json(dict(report), "final_report")
    if (
        result["transaction_contract"] != TRANSACTION_CONTRACT
        or result["receipt"] != "55_final_report.json"
        or result["kind"] != FINAL_REPORT_KIND
        or result["version"] != FORMAT_VERSION
        or result["status"] != "complete"
    ):
        raise ValueError("Candidate43 final report identity changed")
    _sha(result["run_id"], "final_report.run_id")
    _sha(result["attempt_id"], "final_report.attempt_id")
    _sha(result["child_exit_sha256"], "final_report.child_exit_sha256")
    _sha(result["artifact_sha256"], "final_report.artifact_sha256")
    identity = result["identity"]
    if (
        not isinstance(identity, dict)
        or set(identity) != set(TRANSACTION_IDENTITY_FIELDS)
        or identity
        != {
            "run_id": result["run_id"],
            "attempt_id": result["attempt_id"],
            "attempt_number": result["attempt_number"],
        }
    ):
        raise ValueError("Candidate43 final report transaction identity changed")
    predecessors = result["predecessors"]
    if (
        not isinstance(predecessors, dict)
        or set(predecessors) != {"50_child_exit.json"}
        or not isinstance(predecessors["50_child_exit.json"], dict)
        or set(predecessors["50_child_exit.json"])
        != set(PREDECESSOR_FIELDS)
        or predecessors["50_child_exit.json"]["sha256"]
        != result["child_exit_sha256"]
        or type(predecessors["50_child_exit.json"]["size"]) is not int
        or predecessors["50_child_exit.json"]["size"] < 1
    ):
        raise ValueError("Candidate43 final report predecessor chain changed")
    if (
        type(result["attempt_number"]) is not int
        or result["attempt_number"] < 1
        or type(result["artifact_size"]) is not int
        or result["artifact_size"] < 1
    ):
        raise ValueError("Candidate43 final report counters are invalid")
    expected_output = _canonical_artifact_output(checked_artifact["metadata"])
    output = result["artifact_output"]
    if (
        not isinstance(output, str)
        or not output
        or not Path(output).is_absolute()
        or Path(output).resolve() != expected_output
    ):
        raise ValueError("Candidate43 final report artifact path is not canonical")
    authority = _post_exit_authority(
        result["post_exit_authority"], checked_artifact["metadata"]
    )
    expected_receipts = {
        "source_manifest_sha256": manifest_sha256(authority["source_sha256"]),
        "checkpoint_manifest_sha256": canonical_json_sha256(
            authority["checkpoint_sha256"]
        ),
        "runtime_asset_manifest_sha256": manifest_sha256(
            authority["runtime_asset_sha256"]
        ),
    }
    for name, expected in expected_receipts.items():
        if result[name] != expected:
            raise ValueError(f"Candidate43 final report {name} changed")
    core = validate_report(
        result["report_core"],
        checked_artifact,
        require_sealed_plan=require_sealed_plan,
    )
    expected_core_sha = hashlib.sha256(
        report_core_bytes(
            core,
            checked_artifact,
            require_sealed_plan=require_sealed_plan,
        )
    ).hexdigest()
    if result["report_core_sha256"] != expected_core_sha:
        raise ValueError("Candidate43 final report core hash changed")
    result["post_exit_authority"] = authority
    result["report_core"] = core
    return result


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
        raise ValueError("Candidate43 complements must contain a and b")
    for name in METADATA_FIELDS:
        if name not in {"replicate", "assignment_mask_sha256"} and ma[name] != mb[name]:
            raise ValueError(f"Candidate43 complementary metadata differs at {name}")
    ea, eb = left["episodes"], right["episodes"]
    if not torch.equal(ea["treatment"], ~eb["treatment"]):
        raise ValueError("Candidate43 assignments are not exact complements")
    for name in ("env_slot", "assignment_rank", "fixed_z"):
        if not torch.equal(ea[name], eb[name]):
            raise ValueError(f"Candidate43 complementary {name} changed")
    return left, right


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    checked = _strict_json(dict(payload), "json payload")
    return (
        json.dumps(checked, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def final_report_bytes(
    report: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    require_sealed_plan: bool = True,
) -> bytes:
    """Return the exact bytes staged at 55 and hard-linked as ``trial.json``."""

    return _json_bytes(
        validate_final_report(
            report,
            artifact,
            require_sealed_plan=require_sealed_plan,
        )
    )


__all__ = [
    "ACTION_DIM",
    "ARM_ACTION_DIM",
    "ARTIFACT_KIND",
    "AUTHORITY_CONTRACT",
    "CANDIDATE42_COLLECTION_SEAL_COMMIT",
    "CANDIDATE42_COLLECTION_SEAL_TAG",
    "CANDIDATE43_SECTION_SHA256",
    "CANDIDATE42_PLAN",
    "CANDIDATE42_PLAN_SHA256",
    "CANDIDATE42_PREREGISTRATION_COMMIT",
    "CANDIDATE42_PREREGISTRATION_TAG",
    "CANDIDATE42_REPORT",
    "CANDIDATE42_REPORT_SHA256",
    "CANDIDATE42_RESULT",
    "CANDIDATE42_RESULT_COMMIT",
    "CANDIDATE42_RESULT_SHA256",
    "CANDIDATE42_RESULT_TAG",
    "CANDIDATE42_RESULT_TAG_OBJECT_OID",
    "COLLECTION_SEAL_TAG",
    "COLLECTOR",
    "EPISODE_FIELDS",
    "FINAL_REPORT_FIELDS",
    "FINAL_REPORT_KIND",
    "FIXED_DIRECTION_PATH",
    "HAND_ACTION_DIM",
    "IMPLEMENTATION_SOURCE_FILES",
    "METADATA_FIELDS",
    "POST_EXIT_AUTHORITY_FIELDS",
    "PREREGISTRATION_COMMIT",
    "PREREGISTRATION_TAG",
    "PRISTINE_PLAN_SHA256",
    "PRE_STEP_FIELDS",
    "REQUIRED_METADATA",
    "REPORT_FIELDS",
    "REPORT_KIND",
    "RUNTIME_FIELDS",
    "RUNTIME_PACKAGE_FIELDS",
    "SCIENTIFIC_SECTION_SHA256",
    "SOURCE_SECTION_SHA256",
    "SMOKE_ARTIFACT_PATH",
    "SMOKE_REPORT_PATH",
    "STEP_FIELDS",
    "TRACE_METRIC_HORIZON",
    "TRANSACTION_CONTRACT",
    "VALIDATION_PLAN",
    "assignment_mask_sha256",
    "build_artifact",
    "build_final_report",
    "canonical_json_sha256",
    "expected_applied_delta",
    "expected_fixed_residual_action",
    "expected_fixed_z",
    "final_report_bytes",
    "load_fixed_direction",
    "manifest_sha256",
    "normalized_scientific_section_sha256",
    "reconstruct_task_action",
    "report_core_bytes",
    "sha256_file",
    "summarize_artifact",
    "validate_artifact",
    "validate_complementary_artifacts",
    "validate_final_report",
    "validate_report",
    "validate_sealed_plan",
]
