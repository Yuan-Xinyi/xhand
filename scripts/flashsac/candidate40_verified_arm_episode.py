#!/usr/bin/env python3
"""Fail-closed artifact contract for Candidate40 verified-arm development.

Candidate40 keeps the complete Candidate39 route common to both randomized
arms, including the sealed fixed hand residual.  Assignment controls only a
post-eligibility arm7 overlay.  Artifacts retain a C39-style residual table and
a separate, complete first-eligible-to-terminal trace.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

import candidate40_verified_arm_handoff as verifier


ARTIFACT_KIND = "pick_tool_candidate40_verified_arm_episode_v1"
REPORT_KIND = "pick_tool_candidate40_verified_arm_report_v1"
FORMAT_VERSION = 1
COLLECTOR = "collect_candidate40_verified_arm_ab.py"
COLLECTION_CONTRACT = "candidate39_fixed_route_then_public_verified_live_arm_handoff_v1"
DELTA_CONTRACT = "candidate39_fixed_z_common_all_slots_pre_tanh_v1"
VALIDATION_PLAN = "scripts/flashsac/candidate40_verified_arm_handoff_development_plan.json"
CANDIDATE39_RESULT = "scripts/flashsac/candidate39_fixed_validation_result.json"
CANDIDATE39_RESULT_SHA256 = "911f263fce089759fa3d306038458cdefcb1ed6d0e602289df29720f73662e44"
CANDIDATE39_REPORT = "logs/flashsac/pick_tool/51_c39_fixed_direction_development_validation.json"
CANDIDATE39_REPORT_SHA256 = "fd4b72ebcb5711c241b2ec0f5fb6dc85105095d1accfa5c26cd25b2bf0fccaef"
FIXED_DIRECTION_MANIFEST = "scripts/flashsac/candidate39_fixed_direction_manifest.json"
FIXED_DIRECTION_MANIFEST_SHA256 = "b4140aef4e9aae105028dcfd2ea416ba58d734cc8f3ddb951790e49ca94d8772"
FIXED_DIRECTION_PATH = "logs/flashsac/pick_tool/50_c39_residual_fixed_z.pt"
FIXED_DIRECTION_SHA256 = "ca60397c7414e9710b3089f64855548b79ce9d92d0687c6eb014a4ae8934a304"
FIXED_DIRECTION_PAYLOAD_KIND = "pick_tool_candidate39_fixed_residual_direction_v1"
FIXED_DIRECTION_SEMANTIC_SHA256 = "417eae962a0d071bd3b966d253e777e4460cda3bcb0343eb335b6495bd68a9f1"
FIXED_DIRECTION_SOURCE_DESIGN_CONTRACT = "candidate39_sha256_rank_paired_antithetic_v1"
FIXED_DIRECTION_SOURCE_DESIGN_SALT = "pick_tool_candidate39_option_residual_20260722_v1"
FIXED_DIRECTION_DISCOVERY_SEEDS = (327, 328)
V6_CHECKPOINT_PATH = "logs/flashsac/pick_tool/16_v6_router_zero_actor_s254/checkpoint_final"
SEARCH_CHECKPOINT_PATH = (
    "logs/rl_games/pick_tool_token/0_bootstrap_handoff_20260720/nn/"
    "pick_tool_stage7_dagger_full_iter3_bc.pth"
)
SMOKE_ARTIFACT_PATH = (
    "logs/flashsac/pick_tool/52_c40_verified_arm_smoke_s334_b/trial.pt"
)
SMOKE_REPORT_PATH = (
    "logs/flashsac/pick_tool/52_c40_verified_arm_smoke_s334_b/trial.json"
)
V6_ACTOR_SHA256 = "d9aacbd48891c192c0d1491514137262b82990fa71440787076403de06606288"
V6_TASK_CONTRACT_SHA256 = "8a2a135fa1cd5bb01965fc7dc6d6a73978dffc1bf55941ae392146de1d17f61d"
V6_BRIDGE_STATE_SHA256 = "78176bb99eb5dac80a2789b231d54db6cf2b86132bc7835a129dd7f5a7df978c"
FROZEN_LIFT_ACTOR_SHA256 = "117f1b0ae3641bd24b6b9f3d585576a79b5aff968139c0dde86e6346f60ebaa0"
FROZEN_LIFT_SEMANTIC_SHA256 = "8af0fdbc2572346b7519fbb4b29354fb5bc882a0c0020c8c3ace8376696ab324"
FROZEN_LIFT_SOURCE_ACTOR_SHA256 = "7757869eaa1df02f5f52c2dcd1353fb4a486b7512019650a8234d76702b9a1fb"
SEARCH_CHECKPOINT_SHA256 = "b91555d8227cf4e87e41ae858d7f8b1ee42ae920a9d829f2f5b1453b529f1f02"
FLASHSAC_FORK_COMMIT = "5ecf331fa11cd457dd39018b3d68af571b257666"
REQUIRED_BRANCH = "flashsac-pick-tool-curriculum"
KIT_ARGS = "--/app/extensions/fsWatcherEnabled=false"
SEALED_PLAN_STATUS = "sealed_before_smoke_and_collection"

IMPLEMENTATION_SOURCE_FILES = (
    "scripts/flashsac/collect_candidate39_episode_residual_ab.py",
    "scripts/flashsac/candidate40_verified_arm_handoff.py",
    "scripts/flashsac/candidate40_verified_arm_handoff_test.py",
    "scripts/flashsac/candidate40_verified_arm_episode.py",
    "scripts/flashsac/candidate40_verified_arm_episode_test.py",
    "scripts/flashsac/collect_candidate40_verified_arm_ab.py",
    "scripts/flashsac/collect_candidate40_verified_arm_ab_test.py",
    "scripts/flashsac/analyze_candidate40_verified_arm_validation.py",
    "scripts/flashsac/analyze_candidate40_verified_arm_validation_test.py",
)

OBSERVATION_DIM = 115
ACTION_DIM = 21
ARM_ACTION_DIM = 7
HAND_ACTION_DIM = 14
TOKEN_ACTION_DIM = 9
WINDOW_STEPS = 32
HANDOFF_MIN_SCORE = 0.30
HANDOFF_HOLD_STEPS = 4
RAW_Z_ABS_CAP = 2.0
TOKEN_SCALE = 0.05
DISTAL_SCALE = 0.025
TOKEN_COMPONENT_CAP = 0.10
DISTAL_COMPONENT_CAP = 0.05
PRE_TANH_L2_CAP = 0.20
MAX_EPISODE_ACTIONS = 999
TRACE_METRIC_HORIZON = 128
PRE_RELEASE_CLEARANCE_LIMIT_M = 0.015
PUBLIC_GATE_STATE_CONTRACT = "pick_tool_public_gate_state_v1"
PUBLIC_LATCH_INDEX = 106
TRANSITION_ARM_TOKEN_SLICE = slice(70, 86)
TRANSITION_DISTAL_SLICE = slice(87, 92)
PROXIMITY_SLICE = slice(92, 96)
THUMB_PROXIMITY_INDEX = 96

FIXED_Z_VALUES = (
    -0.3442349135875702, -0.662828803062439, -0.17566385865211487,
    -0.953892171382904, -0.3761844038963318, -1.9863512516021729,
    -1.886328935623169, 0.834323525428772, -1.2883188724517822,
    -0.14828261733055115, 0.12304867058992386, 1.2580629587173462,
    0.5757084488868713, 0.7408758401870728,
)

EPISODE_FIELDS = (
    "env_slot", "treatment", "assignment_rank", "fixed_z",
    "triggered", "trigger_step", "trigger_score", "first_latch_step",
    "first_eligible_step", "verification_complete_step",
    "first_arm_enabled_step", "first_relock_step", "first_reenable_step",
    "terminal_step", "intervention_steps", "trace_rows",
    "stable_count_max", "arm_enable_count", "relock_count", "reenable_count",
    "episode_length", "trajectory_max_force_n", "max_true_clearance_m",
    "first_eligible_clearance_m", "pre_release_max_clearance_m",
    "eligible", "pre_release_launch", "latched_within_window",
    "success", "failure", "time_out", "dropped", "unsafe_force",
    "unlatched_clearance_ge_5cm", "ever_grasped", "ever_clearance_ge_20cm",
    "option_active_pre_eligible_unlatched_clearance_ge_5cm",
    "pre_eligibility_action_violations", "hand_invariance_violations",
    "treatment_arm_gate_violations", "control_route_violations",
    "task_action_reconstruction_violations",
    "fixed_residual_budget_violations", "action_bound_violations",
    "treatment_pre_enable_nonzero_arm_rows", "new_abs_action_ge_0999",
)

PRE_STEP_FIELDS = (
    "pre_env_slot", "pre_episode_step", "pre_close_age",
    "pre_option_active", "pre_public_latch_before", "pre_residual_active",
    "pre_first_latch_witness", "pre_observation",
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
    "row_public_max_force_strength",
    "row_stable_count_before", "row_stable_count_after",
    "row_common_action", "row_requested_action", "row_task_action",
    "row_arm_target_error", "row_arm_joint_velocity",
    "row_arm_target_error_abs_max", "row_arm_joint_speed_abs_max",
    "row_pre_action_true_clearance_m",
    "row_treatment", "row_activated", "row_first_eligible",
    "row_arm_enabled", "row_verification_complete", "row_relock",
    "row_reenable", "row_pre_enable", "row_transition_public_latch",
    "row_transition_grasped", "row_transition_grasp_quality",
    "row_transition_hold_quality", "row_transition_max_force_n",
    "row_transition_object_lin_speed", "row_transition_object_ang_speed",
    "row_transition_true_clearance_m", "row_transition_done",
    "row_pre_release_launch",
)

_EPISODE_BOOL = {
    "treatment", "triggered", "eligible", "pre_release_launch",
    "latched_within_window", "success", "failure", "time_out", "dropped",
    "unsafe_force", "unlatched_clearance_ge_5cm", "ever_grasped",
    "ever_clearance_ge_20cm",
    "option_active_pre_eligible_unlatched_clearance_ge_5cm",
    "new_abs_action_ge_0999",
}
_EPISODE_LONG = {
    "env_slot", "assignment_rank", "trigger_step", "first_latch_step",
    "first_eligible_step", "verification_complete_step",
    "first_arm_enabled_step", "first_relock_step", "first_reenable_step",
    "terminal_step", "intervention_steps", "trace_rows", "stable_count_max",
    "arm_enable_count", "relock_count", "reenable_count", "episode_length",
    "pre_eligibility_action_violations", "hand_invariance_violations",
    "treatment_arm_gate_violations", "control_route_violations",
    "task_action_reconstruction_violations", "fixed_residual_budget_violations",
    "action_bound_violations", "treatment_pre_enable_nonzero_arm_rows",
}
_PRE_BOOL = {
    "pre_option_active", "pre_public_latch_before", "pre_residual_active",
    "pre_first_latch_witness",
    "pre_transition_public_latch", "pre_transition_grasped",
    "pre_transition_done",
}
_PRE_LONG = {"pre_env_slot", "pre_episode_step", "pre_close_age"}
_STEP_BOOL = {
    "row_option_active", "row_public_latch_before", "row_stable_current",
    "row_treatment", "row_activated", "row_first_eligible", "row_arm_enabled",
    "row_verification_complete", "row_relock", "row_reenable", "row_pre_enable",
    "row_transition_public_latch", "row_transition_grasped",
    "row_transition_done", "row_pre_release_launch",
}
_STEP_LONG = {
    "row_env_slot", "row_episode_step", "row_eligible_age",
    "row_stable_count_before", "row_stable_count_after",
}

METADATA_FIELDS = (
    "kind", "version", "validation_only", "first_episode_only",
    "network_updates", "collector", "collection_contract", "delta_contract",
    "assignment_contract", "assignment_salt", "assignment_mask_sha256",
    "handoff_min_score", "handoff_hold_steps", "window_steps", "verify_steps",
    "token_scale", "distal_scale", "raw_z_abs_cap", "token_component_cap",
    "distal_component_cap", "pre_tanh_l2_cap", "trace_metric_horizon",
    "pre_release_clearance_limit_m", "public_gate_state_contract",
    "public_safe_force_strength_max", "observation_dim", "action_dim",
    "max_episode_actions", "kit_args", "seed", "replicate", "num_envs",
    "v6_checkpoint", "search_checkpoint", "validation_plan",
    "validation_plan_sha256", "candidate39_result", "candidate39_result_sha256",
    "candidate39_report", "candidate39_report_sha256",
    "fixed_direction_manifest", "fixed_direction_manifest_sha256",
    "fixed_direction_path", "fixed_direction_sha256",
    "fixed_direction_payload_kind", "fixed_direction_semantic_sha256",
    "smoke_artifact_path", "smoke_report_path", "smoke_artifact_sha256",
    "smoke_report_sha256",
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
GIT_FIELDS = (
    "commit", "branch", "source_files_dirty", "flashsac_commit", "flashsac_dirty",
)
RUNTIME_FIELDS = (
    "python", "torch", "cuda", "cudnn", "cuda_device_index",
    "cuda_device_name", "cuda_device_capability", "isaac_sim", "packages",
    "nvidia_smi_inventory", "platform", "seed",
)
RUNTIME_PACKAGE_FIELDS = (
    "isaaclab", "isaaclab_tasks", "isaaclab_assets", "numpy", "gymnasium",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_fixed_z() -> torch.Tensor:
    return torch.tensor(FIXED_Z_VALUES, dtype=torch.float32)


def expected_applied_delta() -> torch.Tensor:
    fixed = expected_fixed_z().clamp(-RAW_Z_ABS_CAP, RAW_Z_ABS_CAP)
    scale = fixed.new_tensor([TOKEN_SCALE] * TOKEN_ACTION_DIM + [DISTAL_SCALE] * 5)
    cap = fixed.new_tensor([TOKEN_COMPONENT_CAP] * TOKEN_ACTION_DIM + [DISTAL_COMPONENT_CAP] * 5)
    delta = torch.maximum(torch.minimum(fixed * scale, cap), -cap)
    norm = torch.linalg.vector_norm(delta)
    return delta if bool(norm <= PRE_TANH_L2_CAP) else delta * (PRE_TANH_L2_CAP / norm)


def expected_fixed_residual_action(baseline_action: torch.Tensor) -> torch.Tensor:
    """Apply the sealed Candidate39 hand residual in pre-tanh space."""

    if (
        not isinstance(baseline_action, torch.Tensor)
        or baseline_action.ndim != 2
        or baseline_action.shape[1] != ACTION_DIM
        or baseline_action.dtype != torch.float32
    ):
        raise ValueError("baseline_action must be float32 [N,21]")
    if not bool(torch.isfinite(baseline_action).all()) or bool(
        (baseline_action.abs() > 1.0).any()
    ):
        raise ValueError("baseline_action must be finite and bounded by one")
    action = baseline_action.clone()
    hand = baseline_action[:, ARM_ACTION_DIM:]
    delta = expected_applied_delta().to(device=hand.device).expand_as(hand)
    margin = max(1.0e-6, 4 * torch.finfo(torch.float32).eps)
    overlaid = torch.tanh(
        torch.atanh(hand.clamp(-1.0 + margin, 1.0 - margin)) + delta
    )
    action[:, ARM_ACTION_DIM:] = torch.where(delta != 0.0, overlaid, hand)
    return action


def assignment_mask_sha256(mask: torch.Tensor) -> str:
    return verifier.assignment_mask_sha256(mask)


def manifest_sha256(manifest: Mapping[str, str]) -> str:
    checked = _sha_manifest(manifest, "manifest")
    encoded = json.dumps(checked, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase Git SHA")
    return value


def _sha_manifest(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty plain dictionary")
    if any(not isinstance(path, str) or not path for path in value):
        raise TypeError(f"{name} keys must be non-empty strings")
    return {path: _sha(digest, f"{name}[{path!r}]") for path, digest in value.items()}


def _strict_json(value: Any, name: str = "value") -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains NaN or infinity")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} contains a non-string key")
        return {key: _strict_json(item, f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item, f"{name}[]") for item in value]
    raise TypeError(f"{name} contains unsupported {type(value).__name__}")


def validate_fixed_direction_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"kind", "format_version", "design_contract", "design_salt", "discovery_seeds", "fixed_z", "discovery_report_semantic_sha256"}
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("fixed-direction payload schema changed")
    expected = {
        "kind": FIXED_DIRECTION_PAYLOAD_KIND,
        "format_version": 1,
        "design_contract": FIXED_DIRECTION_SOURCE_DESIGN_CONTRACT,
        "design_salt": FIXED_DIRECTION_SOURCE_DESIGN_SALT,
        "discovery_report_semantic_sha256": FIXED_DIRECTION_SEMANTIC_SHA256,
    }
    for name, value in expected.items():
        if payload[name] != value or type(payload[name]) is not type(value):
            raise ValueError(f"fixed-direction payload {name} changed")
    seeds = payload["discovery_seeds"]
    fixed = payload["fixed_z"]
    if not isinstance(seeds, torch.Tensor) or seeds.dtype != torch.int64 or seeds.device.type != "cpu" or not torch.equal(seeds, torch.tensor(FIXED_DIRECTION_DISCOVERY_SEEDS)):
        raise ValueError("fixed-direction discovery seeds changed")
    if not isinstance(fixed, torch.Tensor) or fixed.dtype != torch.float32 or fixed.device.type != "cpu" or fixed.shape != (HAND_ACTION_DIM,) or not torch.equal(fixed, expected_fixed_z()):
        raise ValueError("fixed-direction vector changed")
    return {**dict(payload), "discovery_seeds": seeds.clone(), "fixed_z": fixed.clone()}


def load_fixed_direction(path: str | os.PathLike[str]) -> dict[str, Any]:
    if sha256_file(path) != FIXED_DIRECTION_SHA256:
        raise ValueError("fixed-direction SHA256 differs from sealed receipt")
    return validate_fixed_direction_payload(torch.load(path, map_location="cpu", weights_only=True))


def validate_sealed_plan(path: str | os.PathLike[str] | None = None, *, require_sealed: bool = True) -> dict[str, Any]:
    plan_path = _root() / VALIDATION_PLAN if path is None else Path(path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Candidate40 plan: {plan_path}") from error
    if not isinstance(plan, dict) or plan.get("kind") != "pick_tool_candidate40_verified_arm_handoff_development_plan_v1":
        raise ValueError("Candidate40 plan identity changed")
    status = plan.get("status")
    if require_sealed and status != SEALED_PLAN_STATUS:
        raise ValueError("Candidate40 plan is not sealed for collection")
    if not require_sealed and status not in {"preregistered_before_implementation_or_simulator_evidence", SEALED_PLAN_STATUS}:
        raise ValueError("Candidate40 plan status changed")
    assignment = plan.get("assignment")
    contract = plan.get("candidate40_intervention_contract")
    execution = plan.get("execution")
    immutable = plan.get("immutable_policies_and_payloads")
    causal = plan.get("causal_basis")
    if not all(isinstance(x, dict) for x in (assignment, contract, execution, immutable, causal)):
        raise ValueError("Candidate40 plan omitted a contract section")
    if assignment.get("contract") != verifier.ASSIGNMENT_CONTRACT or assignment.get("salt") != verifier.ASSIGNMENT_SALT or assignment.get("num_envs") != 64:
        raise ValueError("Candidate40 assignment contract changed")
    expected_contract = {
        "name": verifier.VERIFIER_CONTRACT,
        "primary_metric_horizon_actions": TRACE_METRIC_HORIZON,
    }
    for name, value in expected_contract.items():
        if contract.get(name) != value:
            raise ValueError(f"Candidate40 intervention changed: {name}")
    if contract.get("actor_visible_features", {}).get("float32_safe_force_strength_max") != verifier.PUBLIC_FORCE_STRENGTH_MAX:
        raise ValueError("Candidate40 public force threshold changed")
    smoke = execution.get("non_evidence_smoke", {})
    if (smoke.get("seed"), smoke.get("replicate"), smoke.get("num_envs")) != (334, "b", 8):
        raise ValueError("Candidate40 smoke identity changed")
    if execution.get("development_seeds") != [335, 336] or execution.get("run_order") != ["335a", "335b", "336b", "336a"] or execution.get("num_envs_per_run") != 64 or execution.get("kit_args") != KIT_ARGS:
        raise ValueError("Candidate40 development run identity changed")
    expected_immutable = {
        "v6_checkpoint": V6_CHECKPOINT_PATH, "v6_actor_sha256": V6_ACTOR_SHA256,
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
        "flashsac_fork_commit": FLASHSAC_FORK_COMMIT, "network_updates": 0,
    }
    for name, value in expected_immutable.items():
        if immutable.get(name) != value:
            raise ValueError(f"Candidate40 immutable receipt changed: {name}")
    result_manifest = causal.get("candidate39_result_manifest", {})
    report = causal.get("candidate39_combined_report", {})
    if (result_manifest.get("path"), result_manifest.get("sha256")) != (CANDIDATE39_RESULT, CANDIDATE39_RESULT_SHA256) or (report.get("path"), report.get("sha256")) != (CANDIDATE39_REPORT, CANDIDATE39_REPORT_SHA256):
        raise ValueError("Candidate39 causal receipts changed")
    if require_sealed:
        seal = plan.get("implementation_seal")
        if not isinstance(seal, dict) or seal.get("status") != "complete_without_simulator_evidence":
            raise ValueError("Candidate40 implementation is not sealed")
        _git_sha(seal.get("implementation_commit"), "implementation_commit")
        sources = _sha_manifest(seal.get("source_sha256"), "implementation source_sha256")
        if set(sources) != set(IMPLEMENTATION_SOURCE_FILES):
            raise ValueError("Candidate40 implementation source set changed")
        for relative, digest in sources.items():
            if sha256_file(_root() / relative) != digest:
                raise ValueError(f"Candidate40 sealed source changed: {relative}")
        _strict_json(seal.get("simulation_free_tests"), "simulation_free_tests")
    return plan


REQUIRED_METADATA = {
    "kind": ARTIFACT_KIND, "version": FORMAT_VERSION, "validation_only": True,
    "first_episode_only": True, "network_updates": 0, "collector": COLLECTOR,
    "collection_contract": COLLECTION_CONTRACT, "delta_contract": DELTA_CONTRACT,
    "assignment_contract": verifier.ASSIGNMENT_CONTRACT,
    "assignment_salt": verifier.ASSIGNMENT_SALT,
    "handoff_min_score": HANDOFF_MIN_SCORE, "handoff_hold_steps": HANDOFF_HOLD_STEPS,
    "window_steps": WINDOW_STEPS, "verify_steps": verifier.VERIFY_STEPS,
    "token_scale": TOKEN_SCALE, "distal_scale": DISTAL_SCALE,
    "raw_z_abs_cap": RAW_Z_ABS_CAP, "token_component_cap": TOKEN_COMPONENT_CAP,
    "distal_component_cap": DISTAL_COMPONENT_CAP, "pre_tanh_l2_cap": PRE_TANH_L2_CAP,
    "trace_metric_horizon": TRACE_METRIC_HORIZON,
    "pre_release_clearance_limit_m": PRE_RELEASE_CLEARANCE_LIMIT_M,
    "public_gate_state_contract": PUBLIC_GATE_STATE_CONTRACT,
    "public_safe_force_strength_max": verifier.PUBLIC_FORCE_STRENGTH_MAX,
    "observation_dim": OBSERVATION_DIM, "action_dim": ACTION_DIM,
    "max_episode_actions": MAX_EPISODE_ACTIONS, "kit_args": KIT_ARGS,
    "v6_checkpoint": V6_CHECKPOINT_PATH, "search_checkpoint": SEARCH_CHECKPOINT_PATH,
    "validation_plan": VALIDATION_PLAN, "candidate39_result": CANDIDATE39_RESULT,
    "candidate39_result_sha256": CANDIDATE39_RESULT_SHA256,
    "candidate39_report": CANDIDATE39_REPORT,
    "candidate39_report_sha256": CANDIDATE39_REPORT_SHA256,
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
            raise ValueError(f"metadata.{name} differs from Candidate40 contract")
    if result["validation_plan_sha256"] != sha256_file(_root() / VALIDATION_PLAN):
        raise ValueError("metadata plan digest changed")
    seed, replicate, num_envs = result["seed"], result["replicate"], result["num_envs"]
    if not _is_int(seed) or seed < 0 or not _is_int(num_envs):
        raise ValueError("Candidate40 seed/num_envs have invalid types or ranges")
    if num_envs == 8 and (seed, replicate) != (334, "b"):
        raise ValueError("8-env smoke must be exact Candidate40 run 334b")
    if num_envs == 64 and seed not in {335, 336}:
        raise ValueError("64-env Candidate40 development uses only seeds 335/336")
    if num_envs not in {8, 64} or replicate not in {"a", "b"}:
        raise ValueError("invalid Candidate40 run identity")
    if num_envs == 8:
        if result["smoke_artifact_sha256"] is not None or result["smoke_report_sha256"] is not None:
            raise ValueError("smoke artifact cannot depend on its own prerequisite receipts")
    else:
        _sha(result["smoke_artifact_sha256"], "metadata.smoke_artifact_sha256")
        _sha(result["smoke_report_sha256"], "metadata.smoke_report_sha256")
    mask = verifier.exact_balanced_treatment_mask(seed=seed, num_envs=num_envs, replicate=replicate)
    if result["assignment_mask_sha256"] != verifier.assignment_mask_sha256(mask):
        raise ValueError("assignment receipt changed")
    checked_manifests: dict[str, dict[str, str]] = {}
    for name in ("source_sha256", "runtime_asset_sha256"):
        manifest = _sha_manifest(result[name], f"metadata.{name}")
        checked_manifests[name] = manifest
        if result[name.removesuffix("sha256") + "manifest_sha256"] != manifest_sha256(manifest):
            raise ValueError(f"metadata {name} manifest receipt changed")
    git = result["git"]
    if not isinstance(git, dict) or set(git) != set(GIT_FIELDS):
        raise ValueError("Candidate40 git provenance schema changed")
    if git["branch"] != REQUIRED_BRANCH or git["source_files_dirty"] or git["flashsac_dirty"]:
        raise ValueError("Candidate40 collection git provenance is dirty/wrong branch")
    if type(git["source_files_dirty"]) is not bool or type(git["flashsac_dirty"]) is not bool:
        raise TypeError("Candidate40 git dirty flags must be bool")
    _git_sha(git["commit"], "metadata.git.commit")
    _git_sha(git["flashsac_commit"], "metadata.git.flashsac_commit")
    if git["flashsac_commit"] != FLASHSAC_FORK_COMMIT:
        raise ValueError("FlashSAC fork commit changed")
    if require_sealed_plan:
        seal = plan["implementation_seal"]
        sealed_sources = _sha_manifest(seal["source_sha256"], "implementation source_sha256")
        for relative, digest in sealed_sources.items():
            if checked_manifests["source_sha256"].get(relative) != digest:
                raise ValueError(f"metadata source manifest omitted sealed source {relative}")
    if result["flashsac_upstream_commit"] != "87edc9061150ae9e962dd84e6544e27a1554b3ab":
        raise ValueError("FlashSAC upstream commit changed")
    runtime = result["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_FIELDS):
        raise ValueError("Candidate40 runtime provenance schema changed")
    if runtime["seed"] != seed:
        raise ValueError("runtime provenance changed")
    if not isinstance(runtime["packages"], dict) or set(runtime["packages"]) != set(RUNTIME_PACKAGE_FIELDS):
        raise ValueError("Candidate40 runtime package provenance changed")
    return result


def _tensor(name: str, value: Any, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU {dtype} with shape {shape}")
    if dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return value.contiguous()


def _table(value: Any, fields: tuple[str, ...], bools: set[str], longs: set[str], suffix: Mapping[str, tuple[int, ...]], *, name: str, rows: int | None = None) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{name} fields must be exactly {fields}")
    first = value[fields[0]]
    if rows is None:
        if not isinstance(first, torch.Tensor) or first.ndim != 1:
            raise ValueError(f"{name}.{fields[0]} must be rank one")
        rows = int(first.numel())
    result = {}
    for field in fields:
        dtype = torch.bool if field in bools else torch.long if field in longs else torch.float32
        result[field] = _tensor(f"{name}.{field}", value[field], (rows, *suffix.get(field, ())), dtype)
    return result


def _episode_table(value: Any, count: int) -> dict[str, torch.Tensor]:
    return _table(value, EPISODE_FIELDS, _EPISODE_BOOL, _EPISODE_LONG, {"fixed_z": (HAND_ACTION_DIM,)}, name="episodes", rows=count)


def _pre_table(value: Any) -> dict[str, torch.Tensor]:
    suffix = {
        "pre_observation": (OBSERVATION_DIM,),
        "pre_transition_observation": (OBSERVATION_DIM,),
        "pre_applied_delta": (HAND_ACTION_DIM,),
        "pre_baseline_action": (ACTION_DIM,), "pre_common_action": (ACTION_DIM,),
        "pre_requested_action": (ACTION_DIM,), "pre_task_action": (ACTION_DIM,),
    }
    return _table(value, PRE_STEP_FIELDS, _PRE_BOOL, _PRE_LONG, suffix, name="pre_steps")


def _step_table(value: Any) -> dict[str, torch.Tensor]:
    suffix = {
        "row_observation": (OBSERVATION_DIM,),
        "row_transition_observation": (OBSERVATION_DIM,),
        "row_public_force_counters": (2,),
        "row_common_action": (ACTION_DIM,), "row_requested_action": (ACTION_DIM,),
        "row_task_action": (ACTION_DIM,),
        "row_arm_target_error": (ARM_ACTION_DIM,),
        "row_arm_joint_velocity": (ARM_ACTION_DIM,),
    }
    return _table(value, STEP_FIELDS, _STEP_BOOL, _STEP_LONG, suffix, name="steps")


def reconstruct_task_action(transition_observation: torch.Tensor) -> torch.Tensor:
    if transition_observation.ndim != 2 or transition_observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("transition observation must be [N,115]")
    return torch.cat((transition_observation[:, TRANSITION_ARM_TOKEN_SLICE], transition_observation[:, TRANSITION_DISTAL_SLICE]), dim=-1)


def _validate_semantics(meta: Mapping[str, Any], ep: Mapping[str, torch.Tensor], pre: Mapping[str, torch.Tensor], st: Mapping[str, torch.Tensor]) -> None:
    count = int(meta["num_envs"])
    slots = torch.arange(count)
    if not torch.equal(ep["env_slot"], slots):
        raise ValueError("episode slots are not canonical")
    expected_treatment = verifier.exact_balanced_treatment_mask(seed=int(meta["seed"]), num_envs=count, replicate=str(meta["replicate"]))
    expected_rank = verifier.assignment_rank(seed=int(meta["seed"]), num_envs=count)
    if not torch.equal(ep["treatment"], expected_treatment) or not torch.equal(ep["assignment_rank"], expected_rank):
        raise ValueError("episode assignment differs from sealed design")
    if not torch.equal(ep["fixed_z"], expected_fixed_z().expand(count, -1)):
        raise ValueError("fixed_z must be identical in every slot")
    lengths = ep["episode_length"]
    if bool(((lengths < 1) | (lengths > MAX_EPISODE_ACTIONS)).any()) or not torch.equal(ep["terminal_step"], lengths - 1):
        raise ValueError("episode terminal clocks changed")
    outcome_count = ep["success"].long() + ep["failure"].long() + ep["time_out"].long()
    if not torch.equal(outcome_count, torch.ones_like(outcome_count)):
        raise ValueError("episode outcomes are not exclusive/exhaustive")
    if bool((ep["time_out"] & (lengths != MAX_EPISODE_ACTIONS)).any()):
        raise ValueError("time_out must occur at the authored horizon")
    authored_failure = ep["dropped"] | ep["unsafe_force"] | ep["unlatched_clearance_ge_5cm"]
    if not torch.equal(ep["failure"], authored_failure):
        raise ValueError("failure sources changed")
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
        raise ValueError("strict success disagrees with physical/safety outcomes")
    if not torch.equal(
        ep["ever_clearance_ge_20cm"], ep["max_true_clearance_m"] >= 0.20
    ):
        raise ValueError("true-20cm aggregate disagrees with maximum clearance")
    if bool(
        (
            ep["unlatched_clearance_ge_5cm"]
            & (ep["max_true_clearance_m"] < 0.05)
        ).any()
    ):
        raise ValueError("unlatched-5cm outcome is below 5 cm maximum clearance")
    triggered = ep["triggered"]
    if not torch.equal(triggered, ep["trigger_step"] >= 0):
        raise ValueError("trigger flag disagrees with trigger clock")
    if bool(((ep["trigger_step"] < -1) | (ep["trigger_step"] >= lengths)).any()):
        raise ValueError("trigger clock is outside the episode")
    if bool(
        (
            triggered
            & (ep["trigger_step"] < HANDOFF_HOLD_STEPS - 1)
        ).any()
    ):
        raise ValueError("trigger step violates the q.30/h4 hold clock")
    if bool(((ep["trigger_score"] < 0.0) | (ep["trigger_score"] > 1.0)).any()):
        raise ValueError("trigger score escaped [0,1]")
    if bool((ep["trigger_score"][~triggered] != 0.0).any()) or bool(
        (ep["trigger_score"][triggered] < HANDOFF_MIN_SCORE).any()
    ):
        raise ValueError("trigger score disagrees with the handoff contract")
    eligible = ep["eligible"]
    for field in (
        "first_latch_step", "first_eligible_step", "first_arm_enabled_step",
        "verification_complete_step", "first_relock_step", "first_reenable_step",
    ):
        clock = ep[field]
        if bool(((clock < -1) | (clock >= lengths)).any()):
            raise ValueError(f"{field} is outside episode")
    if not torch.equal(eligible, ep["first_eligible_step"] >= 0):
        raise ValueError("eligible flag disagrees with clock")
    if bool((eligible & (~triggered)).any()) or bool(
        (ep["first_eligible_step"][eligible] < ep["trigger_step"][eligible]).any()
    ):
        raise ValueError("eligibility precedes the common option handoff")
    if bool((eligible & (ep["first_latch_step"] < 0)).any()) or bool(
        (ep["first_latch_step"][eligible] >= ep["first_eligible_step"][eligible]).any()
    ):
        raise ValueError("first latch and first eligible clocks changed")
    if bool((ep["trace_rows"][~eligible] != 0).any()) or not torch.equal(ep["trace_rows"][eligible], lengths[eligible] - ep["first_eligible_step"][eligible]):
        raise ValueError("trace rows do not span eligibility through terminal")
    if bool((ep["intervention_steps"] < 0).any()) or bool((ep["intervention_steps"] > WINDOW_STEPS).any()):
        raise ValueError("fixed residual intervention count invalid")
    if bool((ep["stable_count_max"] < 0).any()) or bool((ep["stable_count_max"] > verifier.VERIFY_STEPS).any()):
        raise ValueError("stable count escaped verifier bounds")
    if bool((ep["first_eligible_clearance_m"][~eligible] != 0).any()) or bool((ep["pre_release_max_clearance_m"][~eligible] != 0).any()):
        raise ValueError("ineligible episodes have eligibility clearance")
    if bool(ep["pre_release_launch"][~eligible].any()):
        raise ValueError("ineligible episodes have a pre-release launch label")
    if bool((ep["max_true_clearance_m"] + 1e-6 < ep["pre_release_max_clearance_m"]).any()) or bool((ep["trajectory_max_force_n"] < 0).any()):
        raise ValueError("episode physical maxima changed")
    expected_latched_within = (
        triggered
        & (ep["first_latch_step"] >= ep["trigger_step"])
        & ((ep["first_latch_step"] - ep["trigger_step"]) < WINDOW_STEPS)
    )
    if not torch.equal(ep["latched_within_window"], expected_latched_within):
        raise ValueError("latched-within-window label changed")

    if bool(((ep["first_latch_step"] >= 0) & ~ep["ever_grasped"]).any()) or bool(
        (eligible & ~ep["ever_grasped"]).any()
    ):
        raise ValueError("public latch/eligibility lacks an eventual physical grasp")

    pre_env = pre["pre_env_slot"]
    if bool(((pre_env < 0) | (pre_env >= count)).any()):
        raise ValueError("pre-step rows contain an invalid environment slot")
    if bool(pre["pre_public_latch_before"].any()):
        raise ValueError("pre-treatment audit rows must be publicly unlatched")
    if pre_env.numel():
        pre_step = pre["pre_episode_step"]
        pre_option = pre["pre_option_active"]
        pre_age = pre["pre_close_age"]
        first_latch = ep["first_latch_step"][pre_env]
        expected_residual = (
            pre_option
            & (pre_age >= 0)
            & (pre_age < WINDOW_STEPS)
            & ((first_latch < 0) | (pre_step <= first_latch))
        )
        if not torch.equal(pre["pre_residual_active"], expected_residual):
            raise ValueError("pre-step residual window/retirement changed")
        expected_delta = torch.where(
            expected_residual.unsqueeze(-1),
            expected_applied_delta().expand(pre_env.numel(), -1),
            torch.zeros((pre_env.numel(), HAND_ACTION_DIM), dtype=torch.float32),
        )
        if not torch.equal(pre["pre_applied_delta"], expected_delta):
            raise ValueError("pre-step fixed residual/zero delta changed")
        pre_order = pre["pre_episode_step"] * count + pre_env
        if pre_order.numel() > 1 and not bool((pre_order[1:] > pre_order[:-1]).all()):
            raise ValueError("pre-step rows are not canonical step-major/env-major")
        if bool(
            ((pre["pre_episode_step"] < 0) | (pre["pre_episode_step"] >= lengths[pre_env])).any()
        ):
            raise ValueError("pre-step clock is outside the first episode")
        expected_age = torch.where(
            pre_option,
            pre_step - ep["trigger_step"][pre_env],
            torch.full_like(pre_step, -1),
        )
        if not torch.equal(pre_age, expected_age) or bool((pre_age < -1).any()):
            raise ValueError("pre-step CLOSE age changed")
        if bool((pre["pre_observation"][:, PUBLIC_LATCH_INDEX] != 0.0).any()):
            raise ValueError("pre-step observation is not publicly unlatched")
        for name in ("pre_baseline_action", "pre_common_action", "pre_requested_action", "pre_task_action"):
            if bool((pre[name].abs() > 1.0).any()):
                raise ValueError(f"{name} escaped action bounds")
        overlaid = expected_fixed_residual_action(pre["pre_baseline_action"])
        expected_common = torch.where(
            expected_residual.unsqueeze(-1), overlaid, pre["pre_baseline_action"]
        )
        if not torch.allclose(pre["pre_common_action"], expected_common, rtol=0.0, atol=2.0e-6):
            raise ValueError("pre-step common action disagrees with fixed pre-tanh algebra")
        if not torch.equal(pre["pre_requested_action"], pre["pre_common_action"]):
            raise ValueError("Candidate40 changed a pre-eligible residual action")
        if not torch.equal(
            pre["pre_task_action"], reconstruct_task_action(pre["pre_transition_observation"])
        ):
            raise ValueError("pre-step task action is not reset-before reconstruction")
        transition_latch = pre["pre_transition_observation"][:, PUBLIC_LATCH_INDEX] == 1.0
        if not torch.equal(pre["pre_transition_public_latch"], transition_latch):
            raise ValueError("pre-step transition latch disagrees with transition observation")
        expected_witness = (
            (ep["first_latch_step"][pre_env] >= 0)
            & (pre_step == ep["first_latch_step"][pre_env])
            & transition_latch
        )
        if not torch.equal(pre["pre_first_latch_witness"], expected_witness):
            raise ValueError("first-latch witness changed")
        if not bool((pre_option | pre["pre_first_latch_witness"]).all()):
            raise ValueError("pre table contains a row outside its audit domain")
        if not torch.equal(
            pre["pre_transition_done"], pre_step == ep["terminal_step"][pre_env]
        ):
            raise ValueError("pre-step terminal clock changed")
        for field in ("pre_grasp_quality", "pre_hold_quality"):
            if bool(((pre[field] < 0.0) | (pre[field] > 1.0)).any()):
                raise ValueError(f"{field} escaped [0,1]")
        if bool((pre["pre_max_force_n"] < 0.0).any()):
            raise ValueError("pre-step force is negative")
        if bool((pre["pre_max_force_n"] > ep["trajectory_max_force_n"][pre_env] + 1.0e-5).any()) or bool(
            (pre["pre_transition_true_clearance_m"] > ep["max_true_clearance_m"][pre_env] + 1.0e-6).any()
        ):
            raise ValueError("pre-step physical telemetry exceeds episode maximum")
        if bool((pre["pre_transition_grasped"] & ~ep["ever_grasped"][pre_env]).any()):
            raise ValueError("pre-step grasp truth exceeds episode aggregate")
    if not torch.equal(
        torch.bincount(pre_env[pre["pre_residual_active"]], minlength=count),
        ep["intervention_steps"],
    ):
        raise ValueError("pre-step row counts disagree with interventions")
    for slot in range(count):
        selected = pre_env == slot
        option_selected = selected & pre["pre_option_active"]
        option_steps = pre["pre_episode_step"][option_selected]
        option_stop = (
            int(ep["first_eligible_step"][slot])
            if bool(eligible[slot])
            else int(lengths[slot])
        )
        expected_option_steps = (
            torch.arange(int(ep["trigger_step"][slot]), option_stop)
            if bool(triggered[slot])
            else torch.empty(0, dtype=torch.long)
        )
        if not torch.equal(option_steps, expected_option_steps):
            raise ValueError(f"option-active pre-eligibility audit is incomplete for env {slot}")
        witness = pre["pre_first_latch_witness"][selected]
        expected_witness_count = int(ep["first_latch_step"][slot] >= 0)
        if int(witness.sum()) != expected_witness_count:
            raise ValueError(f"first-latch witness count changed for env {slot}")
        if expected_witness_count:
            witness_step = pre["pre_episode_step"][selected][witness][0]
            if int(witness_step) != int(ep["first_latch_step"][slot]):
                raise ValueError(f"first-latch witness clock changed for env {slot}")
        slot_pre_observation = pre["pre_observation"][selected]
        slot_pre_transition = pre["pre_transition_observation"][selected]
        slot_pre_steps = pre["pre_episode_step"][selected]
        if slot_pre_steps.numel() > 1:
            consecutive = slot_pre_steps[1:] == slot_pre_steps[:-1] + 1
            if bool(consecutive.any()) and not torch.equal(
                slot_pre_transition[:-1][consecutive],
                slot_pre_observation[1:][consecutive],
            ):
                raise ValueError(f"pre-step transition continuity changed for env {slot}")
        pre_u5 = (
            pre["pre_option_active"][selected]
            & (~pre["pre_transition_grasped"][selected])
            & (pre["pre_transition_true_clearance_m"][selected] >= 0.05)
        )
        if bool(pre_u5.any()) != bool(
            ep["option_active_pre_eligible_unlatched_clearance_ge_5cm"][slot]
        ):
            raise ValueError(f"pre-eligible option U5 aggregate changed for env {slot}")

    env = st["row_env_slot"]
    if bool(((env < 0) | (env >= count)).any()) or not torch.equal(torch.bincount(env, minlength=count), ep["trace_rows"]):
        raise ValueError("trace row counts disagree with episodes")
    if env.numel() == 0 and bool(eligible.any()):
        raise ValueError("eligible artifact has no trace")
    for slot in triggered.nonzero(as_tuple=False).squeeze(-1).tolist():
        trigger_clock = int(ep["trigger_step"][slot])
        pre_trigger = (pre_env == slot) & (
            pre["pre_episode_step"] == trigger_clock
        )
        trace_trigger = (env == slot) & (
            st["row_episode_step"] == trigger_clock
        )
        if int(pre_trigger.sum()) + int(trace_trigger.sum()) != 1:
            raise ValueError(
                f"trigger pre-action observation is not unique for env {slot}"
            )
        trigger_observation = (
            pre["pre_observation"][pre_trigger][0]
            if bool(pre_trigger.any())
            else st["row_observation"][trace_trigger][0]
        )
        second_nonthumb = torch.topk(
            trigger_observation[PROXIMITY_SLICE], k=2
        ).values[1]
        expected_score = torch.minimum(
            trigger_observation[THUMB_PROXIMITY_INDEX], second_nonthumb
        )
        if not torch.isclose(
            ep["trigger_score"][slot], expected_score, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(
                f"trigger score disagrees with trigger observation for env {slot}"
            )
    order_key = st["row_episode_step"] * count + env
    if order_key.numel() > 1 and not bool((order_key[1:] > order_key[:-1]).all()):
        raise ValueError("trace rows are not canonical step-major/env-major")
    if not bool(st["row_option_active"].all()) or not bool(st["row_activated"].all()):
        raise ValueError("trace contains inactive option/verifier rows")
    if not torch.equal(st["row_treatment"], ep["treatment"][env]):
        raise ValueError("trace treatment changed")
    if not torch.equal(st["row_episode_step"], ep["first_eligible_step"][env] + st["row_eligible_age"]):
        raise ValueError("trace eligibility ages changed")
    first = st["row_eligible_age"] == 0
    if not torch.equal(st["row_first_eligible"], first) or not bool(st["row_public_latch_before"][first].all()):
        raise ValueError("first eligible clock changed")
    if not torch.equal(st["row_public_latch_before"], st["row_observation"][:, PUBLIC_LATCH_INDEX] == 1.0):
        raise ValueError("pre-action latch disagrees with row observation")
    if not torch.equal(st["row_task_action"], reconstruct_task_action(st["row_transition_observation"])):
        raise ValueError("task action is not reset-before reconstruction")
    if not torch.equal(st["row_transition_public_latch"], st["row_transition_observation"][:, PUBLIC_LATCH_INDEX] == 1.0):
        raise ValueError("transition latch disagrees with transition observation")
    public = verifier.public_stable_state(st["row_observation"], st["row_public_force_counters"], st["row_option_active"], torch.ones_like(st["row_option_active"]))
    if not torch.equal(st["row_stable_current"], public.stable):
        raise ValueError("public stable predicate changed")
    if not torch.equal(st["row_public_grasp_quality"], public.grasp_quality) or not torch.equal(
        st["row_public_hold_quality"], public.hold_quality
    ) or not torch.equal(st["row_public_max_force_strength"], public.max_force_strength):
        raise ValueError("public verifier telemetry changed")
    before, after = st["row_stable_count_before"], st["row_stable_count_after"]
    expected_after = torch.where(st["row_stable_current"], torch.clamp(before + 1, max=verifier.VERIFY_STEPS), torch.zeros_like(before))
    if not torch.equal(after, expected_after) or bool(((before < 0) | (before > verifier.VERIFY_STEPS)).any()):
        raise ValueError("stable-count transition changed")
    expected_enabled = st["row_stable_current"] & (before == verifier.VERIFY_STEPS)
    if not torch.equal(st["row_arm_enabled"], expected_enabled):
        raise ValueError("arm enable clock changed")
    common, requested = st["row_common_action"], st["row_requested_action"]
    for name, action in (("common", common), ("requested", requested), ("task", st["row_task_action"])):
        if bool((action.abs() > 1.0).any()):
            raise ValueError(f"trace {name} action escaped bounds")
    if not torch.equal(common[:, ARM_ACTION_DIM:], requested[:, ARM_ACTION_DIM:]):
        raise ValueError("Candidate40 changed hand14")
    control = ~st["row_treatment"]
    if bool(control.any()) and not torch.equal(requested[control], common[control]):
        raise ValueError("Candidate40 changed control route")
    treated = st["row_treatment"]
    expected_arm = torch.where(st["row_arm_enabled"].unsqueeze(-1), common[:, :ARM_ACTION_DIM], torch.zeros_like(common[:, :ARM_ACTION_DIM]))
    if bool(treated.any()) and not torch.equal(requested[treated, :ARM_ACTION_DIM], expected_arm[treated]):
        raise ValueError("Candidate40 treatment arm gate changed")
    expected_target_error = st["row_arm_target_error"].abs().max(dim=-1).values
    expected_joint_speed = st["row_arm_joint_velocity"].abs().max(dim=-1).values
    if not torch.equal(st["row_arm_target_error_abs_max"], expected_target_error) or not torch.equal(
        st["row_arm_joint_speed_abs_max"], expected_joint_speed
    ):
        raise ValueError("arm target-error/speed summaries changed")
    if bool((st["row_arm_target_error_abs_max"] < 0).any()) or bool((st["row_arm_joint_speed_abs_max"] < 0).any()):
        raise ValueError("arm target/speed telemetry must be nonnegative")
    for field in ("row_public_grasp_quality", "row_public_hold_quality", "row_public_max_force_strength", "row_transition_grasp_quality", "row_transition_hold_quality"):
        if bool(((st[field] < 0) | (st[field] > 1)).any()):
            raise ValueError(f"{field} escaped [0,1]")
    if bool((st["row_transition_max_force_n"] < 0).any()) or bool((st["row_transition_object_lin_speed"] < 0).any()) or bool((st["row_transition_object_ang_speed"] < 0).any()):
        raise ValueError("transition physical telemetry is negative")
    if bool((st["row_transition_max_force_n"] > ep["trajectory_max_force_n"][env] + 1.0e-5).any()):
        raise ValueError("trace force exceeds episode maximum")
    if bool((st["row_transition_true_clearance_m"] > ep["max_true_clearance_m"][env] + 1.0e-6).any()) or bool(
        (st["row_pre_action_true_clearance_m"] > ep["max_true_clearance_m"][env] + 1.0e-6).any()
    ):
        raise ValueError("trace clearance exceeds episode maximum")
    if bool((st["row_transition_grasped"] & ~ep["ever_grasped"][env]).any()) or bool(
        ((st["row_transition_true_clearance_m"] >= 0.20) & ~ep["ever_clearance_ge_20cm"][env]).any()
    ):
        raise ValueError("trace physical truth exceeds episode aggregates")

    violation_names = (
        "pre_eligibility_action_violations", "hand_invariance_violations",
        "treatment_arm_gate_violations", "control_route_violations",
        "task_action_reconstruction_violations", "fixed_residual_budget_violations",
        "action_bound_violations",
    )
    for name in violation_names:
        if bool((ep[name] != 0).any()):
            raise ValueError(f"Candidate40 action-algebra audit failed: {name}")

    for slot in range(count):
        selected = env == slot
        slot_steps = st["row_episode_step"][selected]
        if not bool(eligible[slot]):
            if slot_steps.numel():
                raise ValueError(f"ineligible env {slot} has trace rows")
            continue
        expected_steps = torch.arange(
            int(ep["first_eligible_step"][slot]), int(lengths[slot])
        )
        if not torch.equal(slot_steps, expected_steps):
            raise ValueError(f"trace is not complete and contiguous for env {slot}")
        slot_observation = st["row_observation"][selected]
        slot_transition = st["row_transition_observation"][selected]
        if slot_steps.numel() > 1 and not torch.equal(
            slot_transition[:-1], slot_observation[1:]
        ):
            raise ValueError(f"trace transition continuity changed for env {slot}")
        slot_pre = pre_env == slot
        if bool(slot_pre.any()):
            last_pre_step = pre["pre_episode_step"][slot_pre][-1]
            if int(last_pre_step) + 1 == int(slot_steps[0]) and not torch.equal(
                pre["pre_transition_observation"][slot_pre][-1], slot_observation[0]
            ):
                raise ValueError(f"pre/eligible transition continuity changed for env {slot}")
        slot_before = before[selected]
        slot_after = after[selected]
        if int(slot_before[0]) != 0 or (
            slot_before.numel() > 1 and not torch.equal(slot_before[1:], slot_after[:-1])
        ):
            raise ValueError(f"stable clock is discontinuous for env {slot}")
        stable_slot = st["row_stable_current"][selected]
        enabled_slot = st["row_arm_enabled"][selected]
        complete_slot = stable_slot & (slot_before == verifier.VERIFY_STEPS - 1)
        if not torch.equal(st["row_verification_complete"][selected], complete_slot):
            raise ValueError(f"verification-complete edge changed for env {slot}")
        previous_enabled = torch.cat((torch.zeros(1, dtype=torch.bool), enabled_slot[:-1]))
        newly_enabled = enabled_slot & ~previous_enabled
        ever_enabled_before = torch.cat(
            (torch.zeros(1, dtype=torch.bool), newly_enabled.cumsum(0)[:-1] > 0)
        )
        expected_relock = previous_enabled & ~enabled_slot
        expected_reenable = newly_enabled & ever_enabled_before
        expected_pre_enable = (~ever_enabled_before) & (~enabled_slot)
        if not torch.equal(st["row_relock"][selected], expected_relock):
            raise ValueError(f"live relock edge changed for env {slot}")
        if not torch.equal(st["row_reenable"][selected], expected_reenable):
            raise ValueError(f"re-enable edge changed for env {slot}")
        if not torch.equal(st["row_pre_enable"][selected], expected_pre_enable):
            raise ValueError(f"pre-enable state changed for env {slot}")
        done = st["row_transition_done"][selected]
        expected_done = slot_steps == ep["terminal_step"][slot]
        if not torch.equal(done, expected_done) or int(done.sum()) != 1:
            raise ValueError(f"trace terminal edge changed for env {slot}")

        def first_clock(mask: torch.Tensor) -> int:
            ids = mask.nonzero(as_tuple=False).squeeze(-1)
            return -1 if ids.numel() == 0 else int(slot_steps[ids[0]])

        derived = {
            "verification_complete_step": first_clock(complete_slot),
            "first_arm_enabled_step": first_clock(enabled_slot),
            "first_relock_step": first_clock(expected_relock),
            "first_reenable_step": first_clock(expected_reenable),
        }
        for field, expected in derived.items():
            if int(ep[field][slot]) != expected:
                raise ValueError(f"episode {field} disagrees with trace for env {slot}")
        enable_count = int(newly_enabled.sum())
        relock_count = int(expected_relock.sum())
        reenable_count = int(expected_reenable.sum())
        if int(ep["arm_enable_count"][slot]) != enable_count or int(
            ep["relock_count"][slot]
        ) != relock_count or int(ep["reenable_count"][slot]) != reenable_count:
            raise ValueError(f"episode verifier edge counts changed for env {slot}")
        if int(ep["stable_count_max"][slot]) != int(slot_after.max()):
            raise ValueError(f"episode stable-count maximum changed for env {slot}")

        slot_first_clearance = st["row_pre_action_true_clearance_m"][selected][0]
        if not torch.equal(ep["first_eligible_clearance_m"][slot], slot_first_clearance):
            raise ValueError(f"first-eligible clearance changed for env {slot}")
        pre_enable_clearance = st["row_transition_true_clearance_m"][selected][expected_pre_enable]
        values = torch.cat((slot_first_clearance.reshape(1), pre_enable_clearance))
        pre_release_max = values.max()
        if not torch.equal(ep["pre_release_max_clearance_m"][slot], pre_release_max):
            raise ValueError(f"pre-release clearance maximum changed for env {slot}")
        launch_condition = (
            (pre_release_max > PRE_RELEASE_CLEARANCE_LIMIT_M)
            | ((pre_release_max - slot_first_clearance) >= PRE_RELEASE_CLEARANCE_LIMIT_M)
        )
        expected_episode_launch = bool(ep["treatment"][slot]) and bool(launch_condition)
        if bool(ep["pre_release_launch"][slot]) != expected_episode_launch:
            raise ValueError(f"pre-release launch label changed for env {slot}")
        row_clearance = torch.maximum(
            st["row_transition_true_clearance_m"][selected], slot_first_clearance
        )
        expected_row_launch = (
            ep["treatment"][slot]
            & expected_pre_enable
            & (
                (row_clearance > PRE_RELEASE_CLEARANCE_LIMIT_M)
                | ((row_clearance - slot_first_clearance) >= PRE_RELEASE_CLEARANCE_LIMIT_M)
            )
        )
        if not torch.equal(st["row_pre_release_launch"][selected], expected_row_launch):
            raise ValueError(f"row pre-release launch label changed for env {slot}")

    no_trace = ~eligible
    for field in (
        "verification_complete_step", "first_arm_enabled_step", "first_relock_step",
        "first_reenable_step",
    ):
        if bool((ep[field][no_trace] != -1).any()):
            raise ValueError(f"ineligible episodes have {field}")
    for field in ("stable_count_max", "arm_enable_count", "relock_count", "reenable_count"):
        if bool((ep[field][no_trace] != 0).any()):
            raise ValueError(f"ineligible episodes have {field}")

    pre_new = torch.zeros(count, dtype=torch.bool)
    if pre_env.numel():
        pre_elements = (pre["pre_requested_action"].abs() >= 0.999) & (
            pre["pre_baseline_action"].abs() < 0.999
        )
        pre_new.scatter_reduce_(0, pre_env, pre_elements.any(dim=-1), reduce="amax")
    trace_new = torch.zeros(count, dtype=torch.bool)
    trace_elements = (requested.abs() >= 0.999) & (common.abs() < 0.999)
    trace_new.scatter_reduce_(0, env, trace_elements.any(dim=-1), reduce="amax")
    if not torch.equal(ep["new_abs_action_ge_0999"], pre_new | trace_new):
        raise ValueError("new |action|>=0.999 aggregate changed")
    treatment_pre_enable = (
        st["row_treatment"]
        & st["row_pre_enable"]
        & (st["row_requested_action"][:, :ARM_ACTION_DIM] != 0.0).any(dim=-1)
    )
    derived_nonzero = torch.bincount(env[treatment_pre_enable], minlength=count)
    if not torch.equal(ep["treatment_pre_enable_nonzero_arm_rows"], derived_nonzero):
        raise ValueError("treatment pre-enable nonzero-arm aggregate changed")


def validate_artifact(artifact: Mapping[str, Any], *, require_sealed_plan: bool = True) -> dict[str, Any]:
    if not isinstance(artifact, Mapping) or set(artifact) != {"metadata", "episodes", "pre_steps", "steps"}:
        raise ValueError("Candidate40 artifact must contain metadata/episodes/pre_steps/steps")
    meta = _metadata(artifact["metadata"], require_sealed_plan=require_sealed_plan)
    episodes = _episode_table(artifact["episodes"], int(meta["num_envs"]))
    pre = _pre_table(artifact["pre_steps"])
    steps = _step_table(artifact["steps"])
    _validate_semantics(meta, episodes, pre, steps)
    return {"metadata": meta, "episodes": episodes, "pre_steps": pre, "steps": steps}


def build_artifact(metadata: Mapping[str, Any], episodes: Mapping[str, torch.Tensor], pre_steps: Mapping[str, torch.Tensor], steps: Mapping[str, torch.Tensor], *, require_sealed_plan: bool = True) -> dict[str, Any]:
    payload = {
        "metadata": _strict_json(dict(metadata), "metadata"),
        "episodes": {name: episodes[name].detach().cpu().clone() for name in EPISODE_FIELDS},
        "pre_steps": {name: pre_steps[name].detach().cpu().clone() for name in PRE_STEP_FIELDS},
        "steps": {name: steps[name].detach().cpu().clone() for name in STEP_FIELDS},
    }
    return validate_artifact(payload, require_sealed_plan=require_sealed_plan)


def summarize_artifact(artifact: Mapping[str, Any], *, require_sealed_plan: bool = True) -> dict[str, Any]:
    value = validate_artifact(artifact, require_sealed_plan=require_sealed_plan)
    ep, st = value["episodes"], value["steps"]
    treatment = ep["treatment"]
    def arm(mask: torch.Tensor) -> dict[str, int]:
        names = ("eligible", "success", "ever_clearance_ge_20cm", "unlatched_clearance_ge_5cm", "dropped", "unsafe_force", "pre_release_launch")
        return {"episodes": int(mask.sum()), **{name: int((mask & ep[name]).sum()) for name in names}}
    return {
        "episodes": int(ep["env_slot"].numel()), "pre_step_rows": int(value["pre_steps"]["pre_env_slot"].numel()),
        "trace_rows": int(st["row_env_slot"].numel()), "treatment": arm(treatment), "control": arm(~treatment),
        "action_audit": {
            "pre_eligibility_action_violations": int(ep["pre_eligibility_action_violations"].sum()),
            "hand_invariance_violations": int(ep["hand_invariance_violations"].sum()),
            "treatment_arm_gate_violations": int(ep["treatment_arm_gate_violations"].sum()),
            "control_route_violations": int(ep["control_route_violations"].sum()),
            "task_action_reconstruction_violations": int(ep["task_action_reconstruction_violations"].sum()),
            "fixed_residual_budget_violations": int(ep["fixed_residual_budget_violations"].sum()),
            "action_bound_violations": int(ep["action_bound_violations"].sum()),
            "any_new_abs_action_ge_0999": bool(ep["new_abs_action_ge_0999"].any()),
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
    expected_fields = PUBLISHED_REPORT_FIELDS if published else REPORT_FIELDS
    if not isinstance(report, dict) or set(report) != set(expected_fields):
        raise ValueError(f"report fields must be exactly {expected_fields}")
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
        "summary": summarize_artifact(value, require_sealed_plan=require_sealed_plan),
    }
    for name, expected_value in expected.items():
        if checked[name] != expected_value:
            raise ValueError(f"report.{name} differs from the artifact")
    if published:
        _sha(checked["artifact_sha256"], "report.artifact_sha256")
        if not isinstance(checked["artifact_output"], str) or not checked["artifact_output"]:
            raise ValueError("report.artifact_output must be a non-empty string")
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
        raise ValueError("complementary artifacts must contain replicates a and b")
    for name in METADATA_FIELDS:
        if name not in {"replicate", "assignment_mask_sha256"} and ma[name] != mb[name]:
            raise ValueError(f"complementary metadata differs at {name}")
    ea, eb = left["episodes"], right["episodes"]
    if not torch.equal(ea["treatment"], ~eb["treatment"]):
        raise ValueError("Candidate40 assignments are not exact complements")
    for name in ("env_slot", "assignment_rank", "fixed_z"):
        if not torch.equal(ea[name], eb[name]):
            raise ValueError(f"complementary episode {name} changed")
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
    """Durably publish the artifact/report pair or roll back both names."""

    value = validate_artifact(artifact, require_sealed_plan=require_sealed_plan)
    checked_report = validate_report(
        report, value, require_sealed_plan=require_sealed_plan
    )
    artifact_path = Path(os.path.abspath(os.fspath(artifact_output)))
    report_path = Path(os.path.abspath(os.fspath(report_output)))
    if artifact_path == report_path:
        raise ValueError("artifact and report output paths must differ")
    for output in (artifact_path, report_path):
        if _owned(output):
            raise FileExistsError(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[Path] = []
    links: dict[Path, Path] = {}
    try:
        fd, name = tempfile.mkstemp(
            prefix=f".{artifact_path.name}.tmp-", dir=artifact_path.parent
        )
        os.close(fd)
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
        fd, name = tempfile.mkstemp(
            prefix=f".{report_path.name}.tmp-", dir=report_path.parent
        )
        report_tmp = Path(name)
        temporaries.append(report_tmp)
        links[report_path] = report_tmp
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(final_report))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(artifact_tmp, artifact_path)
        os.link(report_tmp, report_path)
        for directory in {artifact_path.parent, report_path.parent}:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
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
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        _unlink_if_ours(path, temporary)
        raise
    finally:
        if _owned(temporary):
            temporary.unlink()


def atomic_torch_save(payload: Mapping[str, Any], output: str | os.PathLike[str]) -> None:
    """Compatibility wrapper with no-clobber semantics."""

    path = Path(os.path.abspath(os.fspath(output)))
    if _owned(path):
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(fd)
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


def atomic_json_dump(payload: Mapping[str, Any], output: str | os.PathLike[str]) -> None:
    """Compatibility wrapper with no-clobber semantics."""

    publish_json_no_clobber(payload, output)


__all__ = [
    "ACTION_DIM", "ARM_ACTION_DIM", "ARTIFACT_KIND", "COLLECTOR",
    "EPISODE_FIELDS", "FIXED_DIRECTION_PATH", "HAND_ACTION_DIM",
    "METADATA_FIELDS", "PRE_STEP_FIELDS", "REQUIRED_METADATA", "REPORT_FIELDS",
    "STEP_FIELDS", "TRACE_METRIC_HORIZON", "VALIDATION_PLAN",
    "RUNTIME_FIELDS", "RUNTIME_PACKAGE_FIELDS",
    "assignment_mask_sha256", "atomic_json_dump", "atomic_torch_save",
    "build_artifact", "expected_applied_delta", "expected_fixed_z",
    "expected_fixed_residual_action", "load_fixed_direction", "manifest_sha256",
    "publish_artifact_and_report_no_clobber", "publish_json_no_clobber",
    "reconstruct_task_action",
    "sha256_file", "summarize_artifact", "validate_artifact",
    "validate_complementary_artifacts", "validate_report", "validate_sealed_plan",
]
