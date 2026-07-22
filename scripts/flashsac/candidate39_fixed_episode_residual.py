#!/usr/bin/env python3
"""Fail-closed evidence contract for Candidate 39 fixed-direction validation.

This contract is intentionally distinct from the randomized direction-discovery
artifact.  There is no per-episode ``raw_z``: every treated slot receives the
same sealed hand14 direction, while a separate SHA256-ranked assignment decides
which slots are treated.  The artifact retains every slot's first episode and
the complete ragged pre-latch action/observation audit window.
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


ARTIFACT_KIND = "pick_tool_candidate39_fixed_episode_residual_validation_v1"
REPORT_KIND = "pick_tool_candidate39_fixed_episode_residual_report_v1"
FORMAT_VERSION = 1
COLLECTOR = "collect_candidate39_fixed_residual_ab.py"
COLLECTION_CONTRACT = (
    "search_q030_h4_then_fixed_direction_prelatch_hand_residual32_v1"
)
DELTA_CONTRACT = "fixed_z_clip_scale_component_clip_l2_clip_pre_tanh_v1"
ASSIGNMENT_CONTRACT = (
    "candidate39_fixed_direction_sha256_rank_exact_balanced_complement_v1"
)
ASSIGNMENT_SALT = "pick_tool_candidate39_fixed_direction_validation_20260722_v1"
VALIDATION_PLAN = (
    "scripts/flashsac/candidate39_fixed_direction_validation_plan.json"
)
FIXED_DIRECTION_MANIFEST = (
    "scripts/flashsac/candidate39_fixed_direction_manifest.json"
)
FIXED_DIRECTION_PATH = "logs/flashsac/pick_tool/50_c39_residual_fixed_z.pt"
KIT_ARGS = "--/app/extensions/fsWatcherEnabled=false"
REQUIRED_BRANCH = "flashsac-pick-tool-curriculum"
SEALED_PLAN_STATUS = "sealed_before_smoke_and_collection"
IMPLEMENTATION_SOURCE_FILES = (
    "scripts/flashsac/candidate39_fixed_episode_residual.py",
    "scripts/flashsac/candidate39_fixed_episode_residual_test.py",
    "scripts/flashsac/collect_candidate39_fixed_residual_ab.py",
    "scripts/flashsac/analyze_candidate39_fixed_validation.py",
    "scripts/flashsac/analyze_candidate39_fixed_validation_test.py",
)

FIXED_DIRECTION_PAYLOAD_KIND = "pick_tool_candidate39_fixed_residual_direction_v1"
FIXED_DIRECTION_SOURCE_DESIGN_CONTRACT = (
    "candidate39_sha256_rank_paired_antithetic_v1"
)
FIXED_DIRECTION_SOURCE_DESIGN_SALT = (
    "pick_tool_candidate39_option_residual_20260722_v1"
)
FIXED_DIRECTION_DISCOVERY_SEEDS = (327, 328)
FIXED_DIRECTION_SHA256 = (
    "ca60397c7414e9710b3089f64855548b79ce9d92d0687c6eb014a4ae8934a304"
)
FIXED_DIRECTION_SEMANTIC_SHA256 = (
    "417eae962a0d071bd3b966d253e777e4460cda3bcb0343eb335b6495bd68a9f1"
)
FIXED_DIRECTION_MANIFEST_SHA256 = (
    "b4140aef4e9aae105028dcfd2ea416ba58d734cc8f3ddb951790e49ca94d8772"
)
FIXED_Z_VALUES = (
    -0.3442349135875702,
    -0.662828803062439,
    -0.17566385865211487,
    -0.953892171382904,
    -0.3761844038963318,
    -1.9863512516021729,
    -1.886328935623169,
    0.834323525428772,
    -1.2883188724517822,
    -0.14828261733055115,
    0.12304867058992386,
    1.2580629587173462,
    0.5757084488868713,
    0.7408758401870728,
)
V6_ACTOR_SHA256 = (
    "d9aacbd48891c192c0d1491514137262b82990fa71440787076403de06606288"
)
V6_TASK_CONTRACT_SHA256 = (
    "8a2a135fa1cd5bb01965fc7dc6d6a73978dffc1bf55941ae392146de1d17f61d"
)
V6_BRIDGE_STATE_SHA256 = (
    "78176bb99eb5dac80a2789b231d54db6cf2b86132bc7835a129dd7f5a7df978c"
)
FROZEN_LIFT_ACTOR_SHA256 = (
    "117f1b0ae3641bd24b6b9f3d585576a79b5aff968139c0dde86e6346f60ebaa0"
)
FROZEN_LIFT_SEMANTIC_SHA256 = (
    "8af0fdbc2572346b7519fbb4b29354fb5bc882a0c0020c8c3ace8376696ab324"
)
FROZEN_LIFT_SOURCE_ACTOR_SHA256 = (
    "7757869eaa1df02f5f52c2dcd1353fb4a486b7512019650a8234d76702b9a1fb"
)
SEARCH_CHECKPOINT_SHA256 = (
    "b91555d8227cf4e87e41ae858d7f8b1ee42ae920a9d829f2f5b1453b529f1f02"
)
FLASHSAC_FORK_COMMIT = "5ecf331fa11cd457dd39018b3d68af571b257666"

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
PUBLIC_LATCH_INDEX = 106
PROXIMITY_SLICE = slice(92, 96)
THUMB_PROXIMITY_INDEX = 96

EPISODE_FIELDS = (
    "env_slot",
    "treatment",
    "assignment_rank",
    "fixed_z",
    "triggered",
    "trigger_step",
    "trigger_score",
    "first_latch_step",
    "latch_released_after_first",
    "intervention_steps",
    "episode_length",
    "trajectory_max_force_n",
    "max_true_clearance_m",
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
    "ever_grasped",
    "ever_clearance_ge_20cm",
    "latched_within_window",
)
STEP_FIELDS = (
    "row_env_slot",
    "row_episode_step",
    "row_close_age",
    "row_public_latch_before",
    "row_residual_active",
    "row_observation",
    "row_base_mean_hand",
    "row_applied_delta",
    "row_baseline_action",
    "row_candidate_action",
    "row_executed_action",
    "row_transition_public_latch",
    "row_transition_grasped",
    "row_transition_true_clearance_m",
    "row_grasp_quality",
    "row_hold_quality",
    "row_max_force_n",
)

_EPISODE_BOOL = {
    "treatment",
    "triggered",
    "latch_released_after_first",
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
    "ever_grasped",
    "ever_clearance_ge_20cm",
    "latched_within_window",
}
_EPISODE_LONG = {
    "env_slot",
    "assignment_rank",
    "trigger_step",
    "first_latch_step",
    "intervention_steps",
    "episode_length",
}
_STEP_BOOL = {
    "row_public_latch_before",
    "row_residual_active",
    "row_transition_public_latch",
    "row_transition_grasped",
}
_STEP_LONG = {"row_env_slot", "row_episode_step", "row_close_age"}

GIT_FIELDS = (
    "commit",
    "branch",
    "source_files_dirty",
    "flashsac_commit",
    "flashsac_dirty",
)
RUNTIME_PACKAGE_FIELDS = (
    "isaaclab",
    "isaaclab_tasks",
    "isaaclab_assets",
    "numpy",
    "gymnasium",
)
RUNTIME_FIELDS = (
    "python",
    "torch",
    "cuda",
    "cudnn",
    "cuda_device_index",
    "cuda_device_name",
    "cuda_device_capability",
    "isaac_sim",
    "packages",
    "nvidia_smi_inventory",
    "platform",
    "seed",
)

METADATA_FIELDS = (
    "kind",
    "version",
    "validation_only",
    "first_episode_only",
    "network_updates",
    "collector",
    "collection_contract",
    "delta_contract",
    "assignment_contract",
    "assignment_salt",
    "assignment_mask_sha256",
    "handoff_min_score",
    "handoff_hold_steps",
    "window_steps",
    "token_scale",
    "distal_scale",
    "raw_z_abs_cap",
    "token_component_cap",
    "distal_component_cap",
    "pre_tanh_l2_cap",
    "observation_dim",
    "action_dim",
    "max_episode_actions",
    "kit_args",
    "seed",
    "replicate",
    "num_envs",
    "v6_checkpoint",
    "search_checkpoint",
    "validation_plan",
    "validation_plan_sha256",
    "fixed_direction_manifest",
    "fixed_direction_manifest_sha256",
    "fixed_direction_path",
    "fixed_direction_sha256",
    "fixed_direction_payload_kind",
    "fixed_direction_source_design_contract",
    "fixed_direction_source_design_salt",
    "fixed_direction_discovery_seeds",
    "fixed_direction_semantic_sha256",
    "v6_actor_sha256",
    "v6_task_contract_sha256",
    "v6_bridge_state_sha256",
    "frozen_lift_actor_sha256",
    "frozen_lift_semantic_sha256",
    "frozen_lift_source_actor_sha256",
    "search_checkpoint_sha256",
    "source_manifest_sha256",
    "runtime_asset_manifest_sha256",
    "source_sha256",
    "runtime_asset_sha256",
    "flashsac_upstream_commit",
    "flashsac_fork_commit",
    "git",
    "runtime",
)

REPORT_FIELDS = (
    "kind",
    "status",
    "collector",
    "seed",
    "replicate",
    "num_envs",
    "vector_steps",
    "v6_actor_sha256",
    "search_checkpoint_sha256",
    "fixed_direction_sha256",
    "summary",
)
PUBLISHED_REPORT_FIELDS = (*REPORT_FIELDS, "artifact_sha256", "artifact_output")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_plan_digest() -> str:
    path = _root() / VALIDATION_PLAN
    try:
        return sha256_file(path)
    except OSError as error:
        raise ValueError(f"cannot read sealed validation plan: {path}") from error


def expected_fixed_z() -> torch.Tensor:
    return torch.tensor(FIXED_Z_VALUES, dtype=torch.float32)


def validate_fixed_direction_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "kind",
        "format_version",
        "design_contract",
        "design_salt",
        "discovery_seeds",
        "fixed_z",
        "discovery_report_semantic_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("fixed-direction payload schema changed")
    fixed = {
        "kind": FIXED_DIRECTION_PAYLOAD_KIND,
        "format_version": 1,
        "design_contract": FIXED_DIRECTION_SOURCE_DESIGN_CONTRACT,
        "design_salt": FIXED_DIRECTION_SOURCE_DESIGN_SALT,
        "discovery_report_semantic_sha256": FIXED_DIRECTION_SEMANTIC_SHA256,
    }
    for name, expected in fixed.items():
        if payload[name] != expected or type(payload[name]) is not type(expected):
            raise ValueError(f"fixed-direction payload {name} changed")
    seeds = payload["discovery_seeds"]
    if (
        not isinstance(seeds, torch.Tensor)
        or seeds.device.type != "cpu"
        or seeds.dtype != torch.int64
        or not torch.equal(
            seeds, torch.tensor(FIXED_DIRECTION_DISCOVERY_SEEDS, dtype=torch.int64)
        )
    ):
        raise ValueError("fixed-direction discovery seeds changed")
    fixed_z = payload["fixed_z"]
    if (
        not isinstance(fixed_z, torch.Tensor)
        or fixed_z.device.type != "cpu"
        or fixed_z.dtype != torch.float32
        or fixed_z.shape != (HAND_ACTION_DIM,)
        or not bool(torch.isfinite(fixed_z).all())
        or not torch.equal(fixed_z, expected_fixed_z())
    ):
        raise ValueError("fixed-direction vector differs from the sealed float32 hand14")
    return {
        **dict(payload),
        "discovery_seeds": seeds.detach().clone(),
        "fixed_z": fixed_z.detach().clone(),
    }


def load_fixed_direction(path: str | os.PathLike[str]) -> dict[str, Any]:
    value = Path(path)
    if sha256_file(value) != FIXED_DIRECTION_SHA256:
        raise ValueError("fixed-direction file SHA256 differs from the sealed receipt")
    try:
        payload = torch.load(value, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("fixed-direction file is not weights-only loadable") from error
    return validate_fixed_direction_payload(payload)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ranked_slots(*, seed: int, num_envs: int) -> list[int]:
    if not _is_int(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not _is_int(num_envs) or num_envs < 2 or num_envs % 2:
        raise ValueError("num_envs must be an even integer of at least two")
    return sorted(
        range(num_envs),
        key=lambda slot: (
            hashlib.sha256(
                f"{ASSIGNMENT_SALT}\0rank\0{seed}\0{slot}".encode("utf-8")
            ).digest(),
            slot,
        ),
    )


def exact_balanced_treatment_mask(
    *, seed: int, num_envs: int, replicate: str
) -> torch.Tensor:
    """Return an independent fixed-direction assignment; B complements A."""

    if replicate not in {"a", "b"}:
        raise ValueError("replicate must be 'a' or 'b'")
    ranked = _ranked_slots(seed=seed, num_envs=num_envs)
    arm_a = torch.zeros(num_envs, dtype=torch.bool)
    arm_a[torch.tensor(ranked[: num_envs // 2], dtype=torch.long)] = True
    return arm_a if replicate == "a" else ~arm_a


def assignment_rank(*, seed: int, num_envs: int) -> torch.Tensor:
    """Return each stable slot's zero-based rank under the sealed assignment."""

    ranked = _ranked_slots(seed=seed, num_envs=num_envs)
    result = torch.empty(num_envs, dtype=torch.long)
    result[torch.tensor(ranked, dtype=torch.long)] = torch.arange(
        num_envs, dtype=torch.long
    )
    return result


def assignment_mask_sha256(mask: torch.Tensor) -> str:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 1
        or mask.dtype != torch.bool
        or mask.device.type != "cpu"
    ):
        raise ValueError("assignment mask receipt requires a rank-one CPU bool tensor")
    encoded = bytes(int(value) for value in mask.tolist())
    return hashlib.sha256(
        ASSIGNMENT_SALT.encode("utf-8") + b"\0mask-v1\0" + encoded
    ).hexdigest()


def expected_applied_delta() -> torch.Tensor:
    fixed_z = expected_fixed_z()
    scale = fixed_z.new_tensor(
        [TOKEN_SCALE] * TOKEN_ACTION_DIM
        + [DISTAL_SCALE] * (HAND_ACTION_DIM - TOKEN_ACTION_DIM)
    )
    cap = fixed_z.new_tensor(
        [TOKEN_COMPONENT_CAP] * TOKEN_ACTION_DIM
        + [DISTAL_COMPONENT_CAP] * (HAND_ACTION_DIM - TOKEN_ACTION_DIM)
    )
    delta = fixed_z.clamp(-RAW_Z_ABS_CAP, RAW_Z_ABS_CAP) * scale
    delta = torch.maximum(torch.minimum(delta, cap), -cap)
    norm = torch.linalg.vector_norm(delta)
    if bool(norm > PRE_TANH_L2_CAP):
        delta = delta * (PRE_TANH_L2_CAP / norm)
    return delta


def _strict_json(value: Any, name: str = "value") -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} has a non-string key")
            result[key] = _strict_json(item, f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_strict_json(item, f"{name}[]") for item in value]
    raise TypeError(f"{name} contains unsupported {type(value).__name__}")


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _git_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return value


def _sha_manifest(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty plain dictionary")
    result: dict[str, str] = {}
    for path, digest in value.items():
        if not isinstance(path, str) or not path:
            raise ValueError(f"{name} has an empty/non-string path")
        result[path] = _sha(digest, f"{name}[{path!r}]")
    return result


def manifest_sha256(manifest: Mapping[str, str]) -> str:
    checked = _sha_manifest(manifest, "manifest")
    encoded = json.dumps(
        checked, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_sealed_validation_plan(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    plan_path = _root() / VALIDATION_PLAN if path is None else Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read sealed validation plan: {plan_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("sealed validation plan must be a JSON object")
    if payload.get("kind") != (
        "pick_tool_candidate39_fixed_direction_development_validation_plan_v1"
    ) or payload.get("status") != SEALED_PLAN_STATUS:
        raise ValueError("Candidate 39 fixed-direction plan is not sealed for collection")
    seal = payload.get("implementation_seal")
    expected_seal_fields = {
        "status",
        "implementation_commit",
        "source_sha256",
        "simulation_free_tests",
    }
    if not isinstance(seal, dict) or set(seal) != expected_seal_fields:
        raise ValueError("validation plan implementation_seal schema changed")
    if seal["status"] != "complete_without_simulator_evidence":
        raise ValueError("validation implementation was not sealed simulation-free")
    _git_sha(seal["implementation_commit"], "plan implementation_commit")
    source = _sha_manifest(seal["source_sha256"], "plan implementation source_sha256")
    if set(source) != set(IMPLEMENTATION_SOURCE_FILES):
        raise ValueError("plan implementation source set differs from fixed contract")
    for relative in IMPLEMENTATION_SOURCE_FILES:
        actual = sha256_file(_root() / relative)
        if source[relative] != actual:
            raise ValueError(f"plan implementation source hash changed: {relative}")
    tests = seal["simulation_free_tests"]
    if not isinstance(tests, dict) or not tests:
        raise ValueError("plan simulation_free_tests must be a non-empty mapping")
    _strict_json(tests, "plan implementation_seal.simulation_free_tests")

    assignment = payload.get("assignment")
    intervention = payload.get("fixed_intervention_contract")
    execution = payload.get("execution")
    evidence = payload.get("immutable_fixed_direction_evidence")
    policies = payload.get("immutable_policies")
    if not all(
        isinstance(item, dict)
        for item in (assignment, intervention, execution, evidence, policies)
    ):
        raise ValueError("sealed validation plan omitted an immutable contract section")
    if (
        assignment.get("contract") != ASSIGNMENT_CONTRACT
        or assignment.get("salt") != ASSIGNMENT_SALT
        or assignment.get("num_envs") != 64
    ):
        raise ValueError("sealed plan assignment differs from executable contract")
    registered_intervention = {
        "window_actions": WINDOW_STEPS,
        "token9_pre_tanh_scale": TOKEN_SCALE,
        "distal5_pre_tanh_scale": DISTAL_SCALE,
        "fixed_z_component_clip": RAW_Z_ABS_CAP,
        "token9_component_abs_cap": TOKEN_COMPONENT_CAP,
        "distal5_component_abs_cap": DISTAL_COMPONENT_CAP,
        "delta14_l2_cap": PRE_TANH_L2_CAP,
        "atanh_epsilon": 1.0e-6,
        "network_updates": 0,
    }
    for name, expected in registered_intervention.items():
        if intervention.get(name) != expected or type(intervention.get(name)) is not type(
            expected
        ):
            raise ValueError(f"sealed plan intervention changed: {name}")
    manifest = evidence.get("manifest")
    fixed_payload = evidence.get("payload")
    if not isinstance(manifest, dict) or not isinstance(fixed_payload, dict):
        raise ValueError("sealed plan omitted fixed-direction receipts")
    if (
        manifest.get("path") != FIXED_DIRECTION_MANIFEST
        or manifest.get("sha256") != FIXED_DIRECTION_MANIFEST_SHA256
        or fixed_payload.get("path") != FIXED_DIRECTION_PATH
        or fixed_payload.get("sha256") != FIXED_DIRECTION_SHA256
        or fixed_payload.get("kind") != FIXED_DIRECTION_PAYLOAD_KIND
    ):
        raise ValueError("sealed plan fixed-direction receipt changed")
    registered_policies = {
        "v6_actor_sha256": V6_ACTOR_SHA256,
        "v6_task_contract_sha256": V6_TASK_CONTRACT_SHA256,
        "v6_bridge_state_sha256": V6_BRIDGE_STATE_SHA256,
        "frozen_lift_actor_sha256": FROZEN_LIFT_ACTOR_SHA256,
        "frozen_lift_semantic_sha256": FROZEN_LIFT_SEMANTIC_SHA256,
        "frozen_lift_source_actor_sha256": FROZEN_LIFT_SOURCE_ACTOR_SHA256,
        "search_checkpoint_sha256": SEARCH_CHECKPOINT_SHA256,
        "flashsac_fork_commit": FLASHSAC_FORK_COMMIT,
    }
    for name, expected in registered_policies.items():
        if policies.get(name) != expected:
            raise ValueError(f"sealed plan policy receipt changed: {name}")
    smoke = execution.get("non_evidence_smoke")
    if not isinstance(smoke, dict) or (
        smoke.get("seed"), smoke.get("replicate"), smoke.get("num_envs")
    ) != (326, "b", 8):
        raise ValueError("sealed plan smoke identity changed")
    if (
        execution.get("seeds") != [329, 330]
        or execution.get("replicates") != ["a", "b"]
        or execution.get("num_envs_per_run") != 64
        or execution.get("run_order") != ["329a", "329b", "330b", "330a"]
        or execution.get("kit_args") != KIT_ARGS
    ):
        raise ValueError("sealed plan evidence run identity changed")
    return payload


REQUIRED_METADATA = {
    "kind": ARTIFACT_KIND,
    "version": FORMAT_VERSION,
    "validation_only": True,
    "first_episode_only": True,
    "network_updates": 0,
    "collector": COLLECTOR,
    "collection_contract": COLLECTION_CONTRACT,
    "delta_contract": DELTA_CONTRACT,
    "assignment_contract": ASSIGNMENT_CONTRACT,
    "assignment_salt": ASSIGNMENT_SALT,
    "handoff_min_score": HANDOFF_MIN_SCORE,
    "handoff_hold_steps": HANDOFF_HOLD_STEPS,
    "window_steps": WINDOW_STEPS,
    "token_scale": TOKEN_SCALE,
    "distal_scale": DISTAL_SCALE,
    "raw_z_abs_cap": RAW_Z_ABS_CAP,
    "token_component_cap": TOKEN_COMPONENT_CAP,
    "distal_component_cap": DISTAL_COMPONENT_CAP,
    "pre_tanh_l2_cap": PRE_TANH_L2_CAP,
    "observation_dim": OBSERVATION_DIM,
    "action_dim": ACTION_DIM,
    "max_episode_actions": MAX_EPISODE_ACTIONS,
    "kit_args": KIT_ARGS,
    "validation_plan": VALIDATION_PLAN,
    "fixed_direction_manifest": FIXED_DIRECTION_MANIFEST,
    "fixed_direction_manifest_sha256": FIXED_DIRECTION_MANIFEST_SHA256,
    "fixed_direction_path": FIXED_DIRECTION_PATH,
    "fixed_direction_sha256": FIXED_DIRECTION_SHA256,
    "fixed_direction_payload_kind": FIXED_DIRECTION_PAYLOAD_KIND,
    "fixed_direction_source_design_contract": (
        FIXED_DIRECTION_SOURCE_DESIGN_CONTRACT
    ),
    "fixed_direction_source_design_salt": FIXED_DIRECTION_SOURCE_DESIGN_SALT,
    "fixed_direction_discovery_seeds": list(FIXED_DIRECTION_DISCOVERY_SEEDS),
    "fixed_direction_semantic_sha256": FIXED_DIRECTION_SEMANTIC_SHA256,
    "v6_actor_sha256": V6_ACTOR_SHA256,
    "v6_task_contract_sha256": V6_TASK_CONTRACT_SHA256,
    "v6_bridge_state_sha256": V6_BRIDGE_STATE_SHA256,
    "frozen_lift_actor_sha256": FROZEN_LIFT_ACTOR_SHA256,
    "frozen_lift_semantic_sha256": FROZEN_LIFT_SEMANTIC_SHA256,
    "frozen_lift_source_actor_sha256": FROZEN_LIFT_SOURCE_ACTOR_SHA256,
    "search_checkpoint_sha256": SEARCH_CHECKPOINT_SHA256,
    "flashsac_fork_commit": FLASHSAC_FORK_COMMIT,
}


def _metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(METADATA_FIELDS):
        raise ValueError(f"metadata fields must be exactly {METADATA_FIELDS}")
    result = _strict_json(dict(value), "metadata")
    validate_sealed_validation_plan()
    for name, expected in REQUIRED_METADATA.items():
        if result[name] != expected or type(result[name]) is not type(expected):
            raise ValueError(f"metadata.{name} differs from the fixed contract")
    if result["validation_plan_sha256"] != _validation_plan_digest():
        raise ValueError("metadata validation-plan digest differs from the sealed file")
    seed = result["seed"]
    num_envs = result["num_envs"]
    replicate = result["replicate"]
    if not _is_int(seed) or seed < 0:
        raise ValueError("metadata.seed must be a non-negative integer")
    if not _is_int(num_envs) or num_envs not in {8, 64}:
        raise ValueError("metadata.num_envs must be exactly 8 (smoke) or 64 (evidence)")
    if num_envs == 8 and (seed, replicate) != (326, "b"):
        raise ValueError("8-env non-evidence smoke must be exact run 326b")
    if num_envs == 64 and seed not in {329, 330}:
        raise ValueError("64-env fixed-direction evidence is reserved to seeds 329/330")
    if replicate not in {"a", "b"}:
        raise ValueError("metadata.replicate must be a or b")
    expected_mask = exact_balanced_treatment_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    if result["assignment_mask_sha256"] != assignment_mask_sha256(expected_mask):
        raise ValueError("metadata assignment-mask receipt differs from run identity")
    for name in ("v6_checkpoint", "search_checkpoint"):
        if not isinstance(result[name], str) or not result[name]:
            raise ValueError(f"metadata.{name} must be non-empty")
    for name in (
        "assignment_mask_sha256",
        "validation_plan_sha256",
        "fixed_direction_manifest_sha256",
        "fixed_direction_sha256",
        "fixed_direction_semantic_sha256",
        "v6_actor_sha256",
        "v6_task_contract_sha256",
        "v6_bridge_state_sha256",
        "frozen_lift_actor_sha256",
        "frozen_lift_semantic_sha256",
        "frozen_lift_source_actor_sha256",
        "search_checkpoint_sha256",
        "source_manifest_sha256",
        "runtime_asset_manifest_sha256",
    ):
        _sha(result[name], f"metadata.{name}")
    source = _sha_manifest(result["source_sha256"], "metadata.source_sha256")
    assets = _sha_manifest(
        result["runtime_asset_sha256"], "metadata.runtime_asset_sha256"
    )
    if manifest_sha256(source) != result["source_manifest_sha256"]:
        raise ValueError("source manifest digest disagrees with source_sha256")
    if manifest_sha256(assets) != result["runtime_asset_manifest_sha256"]:
        raise ValueError("runtime asset manifest digest disagrees with runtime_asset_sha256")
    _git_sha(result["flashsac_upstream_commit"], "metadata.flashsac_upstream_commit")
    _git_sha(result["flashsac_fork_commit"], "metadata.flashsac_fork_commit")
    git = result["git"]
    if not isinstance(git, dict) or set(git) != set(GIT_FIELDS):
        raise ValueError("metadata.git has an unexpected schema")
    _git_sha(git["commit"], "metadata.git.commit")
    _git_sha(git["flashsac_commit"], "metadata.git.flashsac_commit")
    if git["branch"] != REQUIRED_BRANCH:
        raise ValueError(f"metadata.git.branch must be {REQUIRED_BRANCH!r}")
    if git["flashsac_commit"] != result["flashsac_fork_commit"]:
        raise ValueError("metadata.git.flashsac_commit disagrees with FlashSAC fork")
    if type(git["source_files_dirty"]) is not bool or type(git["flashsac_dirty"]) is not bool:
        raise TypeError("metadata git dirty flags must be bool")
    if git["source_files_dirty"] or git["flashsac_dirty"]:
        raise ValueError("fixed-direction collection requires clean committed sources")
    runtime = result["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_FIELDS):
        raise ValueError("metadata.runtime has an unexpected schema")
    if runtime["seed"] != seed:
        raise ValueError("metadata.runtime.seed disagrees with seed")
    packages = runtime["packages"]
    if not isinstance(packages, dict) or set(packages) != set(RUNTIME_PACKAGE_FIELDS):
        raise ValueError("metadata.runtime.packages has an unexpected schema")
    return result


def _tensor(
    name: str, value: Any, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype or value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU {dtype}")
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return value.contiguous()


def _episode_table(value: Any, count: int) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(EPISODE_FIELDS):
        raise ValueError(f"episode fields must be exactly {EPISODE_FIELDS}")
    result: dict[str, torch.Tensor] = {}
    for name in EPISODE_FIELDS:
        shape = (count, HAND_ACTION_DIM) if name == "fixed_z" else (count,)
        dtype = (
            torch.bool
            if name in _EPISODE_BOOL
            else torch.long
            if name in _EPISODE_LONG
            else torch.float32
        )
        result[name] = _tensor(f"episodes.{name}", value[name], shape, dtype)
    return result


def _step_table(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(STEP_FIELDS):
        raise ValueError(f"step fields must be exactly {STEP_FIELDS}")
    first = value.get("row_env_slot")
    if not isinstance(first, torch.Tensor) or first.ndim != 1:
        raise ValueError("steps.row_env_slot must be rank one")
    rows = int(first.shape[0])
    suffix = {
        "row_observation": (OBSERVATION_DIM,),
        "row_base_mean_hand": (HAND_ACTION_DIM,),
        "row_applied_delta": (HAND_ACTION_DIM,),
        "row_baseline_action": (ACTION_DIM,),
        "row_candidate_action": (ACTION_DIM,),
        "row_executed_action": (ACTION_DIM,),
    }
    result: dict[str, torch.Tensor] = {}
    for name in STEP_FIELDS:
        dtype = (
            torch.bool
            if name in _STEP_BOOL
            else torch.long
            if name in _STEP_LONG
            else torch.float32
        )
        result[name] = _tensor(
            f"steps.{name}", value[name], (rows, *suffix.get(name, ())), dtype
        )
    return result


def _validate_semantics(
    metadata: Mapping[str, Any],
    episodes: Mapping[str, torch.Tensor],
    steps: Mapping[str, torch.Tensor],
) -> None:
    count = metadata["num_envs"]
    if not torch.equal(episodes["env_slot"], torch.arange(count)):
        raise ValueError("episode env_slot must be canonical arange")
    treatment = exact_balanced_treatment_mask(
        seed=metadata["seed"], num_envs=count, replicate=metadata["replicate"]
    )
    if not torch.equal(episodes["treatment"], treatment):
        raise ValueError("episode treatment differs from fixed-direction assignment")
    expected_rank = assignment_rank(seed=metadata["seed"], num_envs=count)
    if not torch.equal(episodes["assignment_rank"], expected_rank):
        raise ValueError("episode assignment_rank differs from SHA256 ranking")
    expected_z = expected_fixed_z().expand(count, -1)
    if not torch.equal(episodes["fixed_z"], expected_z):
        raise ValueError("episode fixed_z differs from the sealed float32 hand14")

    lengths = episodes["episode_length"]
    if bool(((lengths < 1) | (lengths > MAX_EPISODE_ACTIONS)).any()):
        raise ValueError("episode_length is outside [1,999]")
    success = episodes["success"]
    failure = episodes["failure"]
    timeout = episodes["time_out"]
    if not bool(((success.long() + failure.long() + timeout.long()) == 1).all()):
        raise ValueError("success/failure/time_out must partition episodes")
    authored_failure = (
        episodes["dropped"]
        | episodes["unsafe_force"]
        | episodes["unlatched_clearance_ge_5cm"]
    )
    if not torch.equal(failure, authored_failure):
        raise ValueError("failure differs from authored failure sources")
    if bool((timeout & (lengths != MAX_EPISODE_ACTIONS)).any()):
        raise ValueError("time_out must occur at the authored horizon")
    clearance = episodes["max_true_clearance_m"]
    if not torch.equal(episodes["ever_clearance_ge_20cm"], clearance >= 0.20):
        raise ValueError("20 cm truth disagrees with maximum true clearance")
    bad_success = (
        success
        & (
            ~episodes["ever_grasped"]
            | ~episodes["ever_clearance_ge_20cm"]
            | episodes["dropped"]
            | episodes["unsafe_force"]
            | episodes["unlatched_clearance_ge_5cm"]
        )
    )
    if bool(bad_success.any()):
        raise ValueError("success violates physical/safety truth")
    if bool(
        (episodes["unlatched_clearance_ge_5cm"] & (clearance < 0.05)).any()
    ):
        raise ValueError("unlatched 5 cm truth exceeds maximum clearance")
    if bool((episodes["trajectory_max_force_n"] < 0).any()):
        raise ValueError("trajectory force cannot be negative")

    triggered = episodes["triggered"]
    trigger_step = episodes["trigger_step"]
    trigger_score = episodes["trigger_score"]
    if bool((trigger_step[~triggered] != -1).any()) or bool(
        (trigger_score[~triggered] != 0).any()
    ):
        raise ValueError("untriggered episode has trigger data")
    if bool(
        (
            (trigger_step[triggered] < HANDOFF_HOLD_STEPS - 1)
            | (trigger_step[triggered] >= lengths[triggered])
        ).any()
    ):
        raise ValueError("trigger step violates q.30/h4 or episode bounds")
    if bool(
        (
            (trigger_score[triggered] < HANDOFF_MIN_SCORE)
            | (trigger_score[triggered] > 1)
        ).any()
    ):
        raise ValueError("trigger score violates q.30")
    first_latch = episodes["first_latch_step"]
    if bool(((first_latch < -1) | (first_latch >= lengths)).any()):
        raise ValueError("first_latch_step is outside the episode")
    if bool((episodes["latch_released_after_first"] & (first_latch < 0)).any()):
        raise ValueError("latch cannot release before a first latch")
    expected_window_latch = (
        triggered
        & (first_latch >= trigger_step)
        & ((first_latch - trigger_step) < WINDOW_STEPS)
    )
    if not torch.equal(
        episodes["latched_within_window"], expected_window_latch
    ):
        raise ValueError("latched_within_window disagrees with trigger/latch clocks")

    env_slot = steps["row_env_slot"]
    episode_step = steps["row_episode_step"]
    close_age = steps["row_close_age"]
    rows = int(env_slot.numel())
    if bool(((env_slot < 0) | (env_slot >= count)).any()):
        raise ValueError("ragged step table has invalid env slots")
    if bool(((episode_step < 0) | (episode_step >= lengths[env_slot])).any()):
        raise ValueError("row episode step is outside its episode")
    order_key = episode_step * count + env_slot
    if rows > 1 and not bool((order_key[1:] > order_key[:-1]).all()):
        raise ValueError("rows must be unique canonical step-major/env-major order")
    row_counts = torch.bincount(env_slot, minlength=count)
    pretrigger_latch = triggered & (first_latch >= 0) & (
        first_latch < trigger_step
    )
    should_have_rows = triggered & ~pretrigger_latch
    if (
        bool((row_counts[~triggered] != 0).any())
        or bool((row_counts[pretrigger_latch] != 0).any())
        or bool((row_counts[should_have_rows] < 1).any())
        or bool((row_counts > WINDOW_STEPS).any())
    ):
        raise ValueError("ragged row counts disagree with triggered/window contract")
    active_counts = torch.bincount(
        env_slot[steps["row_residual_active"]], minlength=count
    )
    if not torch.equal(episodes["intervention_steps"], active_counts):
        raise ValueError("intervention_steps disagrees with active rows")
    for slot in range(count):
        selected = env_slot == slot
        selected_count = int(selected.sum())
        if not selected_count:
            continue
        expected_age = torch.arange(selected_count)
        if not torch.equal(close_age[selected], expected_age) or not torch.equal(
            episode_step[selected], trigger_step[slot] + expected_age
        ):
            raise ValueError("per-env CLOSE ages are not contiguous from trigger")
        stop = min(WINDOW_STEPS, int(lengths[slot] - trigger_step[slot]))
        latch = int(first_latch[slot])
        if (
            latch >= int(trigger_step[slot])
            and latch < int(trigger_step[slot]) + WINDOW_STEPS
        ):
            stop = min(stop, latch - int(trigger_step[slot]) + 1)
        if selected_count != stop:
            raise ValueError("ragged row count disagrees with window/latch/terminal stop")
        transition_latch = steps["row_transition_public_latch"][selected]
        if latch >= int(trigger_step[slot]) and latch < int(trigger_step[slot]) + selected_count:
            expected_transition_latch = episode_step[selected] == latch
        else:
            expected_transition_latch = torch.zeros_like(transition_latch)
        if not torch.equal(transition_latch, expected_transition_latch):
            raise ValueError(
                "transition public latch disagrees with the authoritative first edge"
            )
    if bool(steps["row_public_latch_before"].any()):
        raise ValueError("residual-window rows must be pre-latch")
    expected_active = treatment[env_slot]
    if not torch.equal(steps["row_residual_active"], expected_active):
        raise ValueError("residual_active must exactly select treated window rows")

    observation = steps["row_observation"]
    if not torch.equal(
        observation[:, PUBLIC_LATCH_INDEX], torch.zeros(rows, dtype=torch.float32)
    ):
        raise ValueError("observation latch disagrees with pre-latch rows")
    nonthumb_second = torch.topk(
        observation[:, PROXIMITY_SLICE], 2, dim=-1
    ).values[:, 1]
    score = torch.minimum(
        observation[:, THUMB_PROXIMITY_INDEX], nonthumb_second
    )
    first_rows = close_age == 0
    if not torch.allclose(
        score[first_rows], trigger_score[env_slot[first_rows]], rtol=0, atol=1e-6
    ):
        raise ValueError("trigger score disagrees with trigger observation")

    baseline = steps["row_baseline_action"]
    candidate = steps["row_candidate_action"]
    executed = steps["row_executed_action"]
    if bool((baseline.abs() > 1).any()) or bool((candidate.abs() > 1).any()) or bool(
        (executed.abs() > 1).any()
    ):
        raise ValueError("action escaped [-1,1]")
    if not torch.equal(candidate[:, :ARM_ACTION_DIM], baseline[:, :ARM_ACTION_DIM]):
        raise ValueError("fixed direction changed arm7")
    if not torch.equal(
        baseline[:, :ARM_ACTION_DIM], torch.zeros_like(baseline[:, :ARM_ACTION_DIM])
    ):
        raise ValueError("pre-latch CLOSE baseline arm must be exact zero")
    if not torch.allclose(
        torch.tanh(steps["row_base_mean_hand"]),
        baseline[:, ARM_ACTION_DIM:],
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("raw actor mean does not reconstruct baseline hand")
    applied = torch.where(
        expected_active[:, None],
        expected_applied_delta().expand(rows, -1),
        torch.zeros((rows, HAND_ACTION_DIM), dtype=torch.float32),
    )
    if not torch.allclose(
        steps["row_applied_delta"], applied, rtol=0, atol=1e-6
    ):
        raise ValueError("applied delta differs from the one sealed fixed direction")
    inactive = ~expected_active
    if bool(inactive.any()) and (
        not torch.equal(candidate[inactive], baseline[inactive])
        or not torch.equal(
            steps["row_applied_delta"][inactive],
            torch.zeros_like(steps["row_applied_delta"][inactive]),
        )
    ):
        raise ValueError("control rows are not bit-exact baseline")
    margin = max(1e-6, 4 * torch.finfo(torch.float32).eps)
    baseline_hand = baseline[:, ARM_ACTION_DIM:]
    overlaid = torch.tanh(
        torch.atanh(baseline_hand.clamp(-1 + margin, 1 - margin)) + applied
    )
    expected_hand = torch.where(applied != 0, overlaid, baseline_hand)
    if not torch.allclose(
        candidate[:, ARM_ACTION_DIM:], expected_hand, rtol=0, atol=2e-6
    ):
        raise ValueError("candidate hand disagrees with fixed pre-tanh overlay")
    if not torch.equal(executed, candidate):
        raise ValueError("executed window action differs from audited candidate")
    if bool(
        (
            steps["row_applied_delta"][:, :TOKEN_ACTION_DIM].abs()
            > TOKEN_COMPONENT_CAP + 1e-6
        ).any()
    ) or bool(
        (
            steps["row_applied_delta"][:, TOKEN_ACTION_DIM:].abs()
            > DISTAL_COMPONENT_CAP + 1e-6
        ).any()
    ) or bool(
        (
            torch.linalg.vector_norm(steps["row_applied_delta"], dim=-1)
            > PRE_TANH_L2_CAP + 1e-6
        ).any()
    ):
        raise ValueError("applied fixed residual exceeds its registered budget")
    for name in ("row_grasp_quality", "row_hold_quality"):
        if bool(((steps[name] < 0) | (steps[name] > 1)).any()):
            raise ValueError(f"{name} escaped [0,1]")
    if bool((steps["row_max_force_n"] < 0).any()):
        raise ValueError("row force cannot be negative")
    if bool(
        (
            steps["row_max_force_n"]
            > episodes["trajectory_max_force_n"][env_slot] + 1e-5
        ).any()
    ) or bool(
        (
            steps["row_transition_true_clearance_m"]
            > clearance[env_slot] + 1e-6
        ).any()
    ):
        raise ValueError("row physical telemetry exceeds episode maximum")
    if bool(
        (
            steps["row_transition_grasped"]
            & ~episodes["ever_grasped"][env_slot]
        ).any()
    ):
        raise ValueError("row grasp truth disagrees with episode ever_grasped")


def validate_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "metadata",
        "episodes",
        "steps",
    }:
        raise ValueError("artifact must contain exactly metadata/episodes/steps")
    metadata = _metadata(artifact["metadata"])
    episodes = _episode_table(artifact["episodes"], metadata["num_envs"])
    steps = _step_table(artifact["steps"])
    value = {"metadata": metadata, "episodes": episodes, "steps": steps}
    _validate_semantics(metadata, episodes, steps)
    return value


def build_artifact(
    metadata: Mapping[str, Any],
    episodes: Mapping[str, torch.Tensor],
    steps: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    payload = {
        "metadata": dict(metadata),
        "episodes": {
            name: episodes[name].detach().cpu().clone() for name in EPISODE_FIELDS
        },
        "steps": {
            name: steps[name].detach().cpu().clone() for name in STEP_FIELDS
        },
    }
    return validate_artifact(payload)


def _arm_summary(
    episodes: Mapping[str, torch.Tensor], mask: torch.Tensor
) -> dict[str, int]:
    names = (
        "triggered",
        "latched_within_window",
        "ever_grasped",
        "ever_clearance_ge_20cm",
        "success",
        "failure",
        "time_out",
        "dropped",
        "unsafe_force",
        "unlatched_clearance_ge_5cm",
    )
    return {
        "episodes": int(mask.sum()),
        **{name: int((mask & episodes[name]).sum()) for name in names},
    }


def summarize_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_artifact(artifact)
    episodes = value["episodes"]
    steps = value["steps"]
    treatment = episodes["treatment"]
    has_rows = bool(steps["row_env_slot"].numel())
    delta_norm = torch.linalg.vector_norm(steps["row_applied_delta"], dim=-1)
    new_098 = (steps["row_candidate_action"].abs() >= 0.98) & (
        steps["row_baseline_action"].abs() < 0.98
    )
    return {
        "episodes": int(treatment.numel()),
        "step_rows": int(steps["row_env_slot"].numel()),
        "control": _arm_summary(episodes, ~treatment),
        "treatment": _arm_summary(episodes, treatment),
        "action_audit": {
            "applied_delta_abs_max": (
                float(steps["row_applied_delta"].abs().max()) if has_rows else 0.0
            ),
            "applied_delta_l2_max": float(delta_norm.max()) if has_rows else 0.0,
            "candidate_action_abs_max": (
                float(steps["row_candidate_action"].abs().max())
                if has_rows
                else 0.0
            ),
            "new_abs_ge_098_elements": int(new_098.sum()),
            "episodes_with_new_abs_ge_098": int(
                torch.unique(
                    steps["row_env_slot"][new_098.any(dim=-1)]
                ).numel()
            ),
            "any_new_abs_ge_0999": bool(
                (
                    (steps["row_candidate_action"].abs() >= 0.999)
                    & (steps["row_baseline_action"].abs() < 0.999)
                ).any()
            ),
            "budget_violations": 0,
        },
    }


def validate_report(
    report: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    published: bool = False,
) -> dict[str, Any]:
    value = validate_artifact(artifact)
    expected_fields = PUBLISHED_REPORT_FIELDS if published else REPORT_FIELDS
    if not isinstance(report, dict) or set(report) != set(expected_fields):
        raise ValueError(f"report fields must be exactly {expected_fields}")
    checked = _strict_json(report, "report")
    metadata = value["metadata"]
    episodes = value["episodes"]
    fixed = {
        "kind": REPORT_KIND,
        "status": "complete",
        "collector": COLLECTOR,
        "seed": metadata["seed"],
        "replicate": metadata["replicate"],
        "num_envs": metadata["num_envs"],
        "vector_steps": int(episodes["episode_length"].max()),
        "v6_actor_sha256": metadata["v6_actor_sha256"],
        "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
        "fixed_direction_sha256": FIXED_DIRECTION_SHA256,
        "summary": summarize_artifact(value),
    }
    for name, expected in fixed.items():
        if checked[name] != expected:
            raise ValueError(f"report.{name} differs from artifact")
    if published:
        _sha(checked["artifact_sha256"], "report.artifact_sha256")
        if not isinstance(checked["artifact_output"], str) or not checked[
            "artifact_output"
        ]:
            raise ValueError("report.artifact_output must be non-empty")
    return checked


def validate_complementary_artifacts(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    left = validate_artifact(a)
    right = validate_artifact(b)
    left_meta = left["metadata"]
    right_meta = right["metadata"]
    if {left_meta["replicate"], right_meta["replicate"]} != {"a", "b"}:
        raise ValueError("complementary artifacts must contain replicates a and b")
    for name in METADATA_FIELDS:
        if name not in {"replicate", "assignment_mask_sha256"} and left_meta[
            name
        ] != right_meta[name]:
            raise ValueError(f"cross-artifact metadata differs at {name}")
    if not torch.equal(
        left["episodes"]["treatment"], ~right["episodes"]["treatment"]
    ):
        raise ValueError("fixed-direction A/B assignments are not exact complements")
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
    return (
        json.dumps(
            _strict_json(payload), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode()


def publish_artifact_and_report_no_clobber(
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
    artifact_output: Path,
    report_output: Path,
) -> str:
    value = validate_artifact(artifact)
    checked_report = validate_report(report, value)
    artifact_output = Path(os.path.abspath(os.fspath(artifact_output)))
    report_output = Path(os.path.abspath(os.fspath(report_output)))
    if artifact_output == report_output:
        raise ValueError("artifact and report paths must differ")
    for output in (artifact_output, report_output):
        if _owned(output):
            raise FileExistsError(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[Path] = []
    links: dict[Path, Path] = {}
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{artifact_output.name}.tmp-", dir=artifact_output.parent
        )
        os.close(descriptor)
        artifact_temporary = Path(name)
        temporaries.append(artifact_temporary)
        links[artifact_output] = artifact_temporary
        with artifact_temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(artifact_temporary)
        final_report = {
            **checked_report,
            "artifact_sha256": digest,
            "artifact_output": str(artifact_output),
        }
        validate_report(final_report, value, published=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{report_output.name}.tmp-", dir=report_output.parent
        )
        report_temporary = Path(name)
        temporaries.append(report_temporary)
        links[report_output] = report_temporary
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(final_report))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(artifact_temporary, artifact_output)
        os.link(report_temporary, report_output)
        for directory in {artifact_output.parent, report_output.parent}:
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


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    serialized = _json_bytes(payload)
    output = Path(os.path.abspath(os.fspath(output)))
    if _owned(output):
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        _unlink_if_ours(output, temporary)
        raise
    finally:
        if _owned(temporary):
            temporary.unlink()


__all__ = [
    "ARTIFACT_KIND",
    "REPORT_KIND",
    "FORMAT_VERSION",
    "COLLECTOR",
    "COLLECTION_CONTRACT",
    "ASSIGNMENT_CONTRACT",
    "ASSIGNMENT_SALT",
    "VALIDATION_PLAN",
    "FIXED_DIRECTION_MANIFEST",
    "FIXED_DIRECTION_PATH",
    "FIXED_DIRECTION_SHA256",
    "FIXED_DIRECTION_SEMANTIC_SHA256",
    "REQUIRED_METADATA",
    "METADATA_FIELDS",
    "EPISODE_FIELDS",
    "STEP_FIELDS",
    "exact_balanced_treatment_mask",
    "assignment_rank",
    "assignment_mask_sha256",
    "expected_fixed_z",
    "expected_applied_delta",
    "load_fixed_direction",
    "validate_fixed_direction_payload",
    "build_artifact",
    "validate_artifact",
    "summarize_artifact",
    "validate_report",
    "validate_complementary_artifacts",
    "publish_artifact_and_report_no_clobber",
    "publish_json_no_clobber",
    "sha256_file",
    "manifest_sha256",
]
