#!/usr/bin/env python3
"""Simulation-free contract for randomized online CLOSE A/B trials."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch


ARTIFACT_KIND = "pick_tool_online_close_ab_trial_v2"
REPORT_KIND = "pick_tool_online_close_ab_collection_report_v2"
FORMAT_VERSION = 2
COLLECTION_CONTRACT = (
    "search_q030_h4_then_randomized_close_then_common_lift_fswatchoff_v2"
)
ASSIGNMENT_CONTRACT = "sha256_rank_exact_balanced_complement_v1"
ASSIGNMENT_SALT = "pick_tool_online_close_ab_20260722_v2"
KIT_ARGS = "--/app/extensions/fsWatcherEnabled=false"
HANDOFF_MIN_SCORE = 0.30
HANDOFF_HOLD_STEPS = 4
OBSERVATION_DIM = 115
ACTION_DIM = 21
MAX_EPISODE_ACTIONS = 999
SOURCE_PATH_COUNT = 93
SOURCE_PATH_SET_SHA256 = (
    "c8b297dd8a73f1924f1badd7a9c22b1e691ba15817f21111f590b8433a3f5082"
)

OUTCOME_NAMES = (
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
    "ever_grasped",
    "ever_clearance_ge_20cm",
)
TENSOR_NAMES = (
    "env_slot",
    "assignment_candidate",
    "triggered",
    "trigger_step",
    "trigger_score",
    "episode_length",
    "max_true_clearance_m",
    *OUTCOME_NAMES,
)
SHA_METADATA_NAMES = (
    "baseline_actor_sha256",
    "candidate_actor_sha256",
    "baseline_task_contract_sha256",
    "candidate_task_contract_sha256",
    "common_frozen_lift_actor_sha256",
    "common_frozen_lift_semantic_sha256",
    "common_frozen_lift_source_actor_sha256",
    "search_checkpoint_sha256",
    "baseline_bridge_state_sha256",
    "candidate_bridge_state_sha256",
)
METADATA_NAMES = (
    "kind",
    "format_version",
    "collection_contract",
    "assignment_contract",
    "assignment_salt",
    "handoff_min_score",
    "handoff_hold_steps",
    "observation_dim",
    "action_dim",
    "max_episode_actions",
    "kit_args",
    "seed",
    "replicate",
    "num_envs",
    "baseline_checkpoint",
    "candidate_checkpoint",
    "search_checkpoint",
    *SHA_METADATA_NAMES,
    "flashsac_upstream_commit",
    "flashsac_fork_commit",
    "source_sha256",
    "runtime_asset_sha256",
    "git",
    "runtime",
)
GIT_METADATA_NAMES = (
    "commit",
    "branch",
    "source_files_dirty",
    "flashsac_commit",
    "flashsac_dirty",
)
RUNTIME_PACKAGE_NAMES = (
    "isaaclab",
    "isaaclab_tasks",
    "isaaclab_assets",
    "numpy",
    "gymnasium",
)
RUNTIME_ASSET_NAMES = (
    "/tmp/xhand_inhand/pick_tool_token/.asset_hash",
    "/tmp/xhand_inhand/pick_tool_token/Props/instanceable_meshes.usd",
    "/tmp/xhand_inhand/pick_tool_token/tool_hammer.usd",
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_base.usd",
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_physics.usd",
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_robot.usd",
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/configuration/xarm7_xhand_sensor.usd",
    "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/xarm7_xhand.usd",
)
RUNTIME_METADATA_NAMES = (
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
REPORT_NAMES = (
    "kind",
    "status",
    "seed",
    "replicate",
    "num_envs",
    "vector_steps",
    "collection_contract",
    "baseline_actor_sha256",
    "candidate_actor_sha256",
    "common_frozen_lift_actor_sha256",
    "search_checkpoint_sha256",
    "summary",
)
PUBLISHED_REPORT_NAMES = (*REPORT_NAMES, "artifact_sha256", "artifact_output")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_git_sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return value


def _require_json_safe(value: Any, *, name: str) -> None:
    """Reject pickle-only metadata and non-finite values recursively."""

    if value is None or isinstance(value, (str, bool)) or _is_int(value):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_safe(item, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} contains a non-string mapping key")
            _require_json_safe(item, name=f"{name}.{key}")
        return
    raise TypeError(f"{name} contains non-JSON-safe value {type(value).__name__}")


def _require_sha_mapping(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty plain dictionary")
    result: dict[str, str] = {}
    for path, digest in value.items():
        if not isinstance(path, str) or not path:
            raise ValueError(f"{name} keys must be non-empty paths")
        result[path] = _require_sha256(digest, name=f"{name}[{path!r}]")
    return result


def assignment_candidate_mask(
    *, seed: int, num_envs: int, replicate: str
) -> torch.Tensor:
    """Return a deterministic balanced assignment; b exactly complements a."""

    if not _is_int(seed):
        raise TypeError("seed must be an integer")
    if not _is_int(num_envs) or num_envs < 2 or num_envs % 2:
        raise ValueError("num_envs must be a positive even integer")
    if replicate not in {"a", "b"}:
        raise ValueError("replicate must be 'a' or 'b'")
    ranked = sorted(
        range(num_envs),
        key=lambda env_id: (
            hashlib.sha256(
                f"{ASSIGNMENT_SALT}\0{seed}\0{env_id}".encode("utf-8")
            ).digest(),
            env_id,
        ),
    )
    assignment_a = torch.zeros(num_envs, dtype=torch.bool)
    assignment_a[torch.tensor(ranked[: num_envs // 2], dtype=torch.long)] = True
    return assignment_a if replicate == "a" else ~assignment_a


def select_executed_action(
    *,
    search_action: torch.Tensor,
    baseline_action: torch.Tensor,
    candidate_action: torch.Tensor,
    option_active: torch.Tensor,
    assignment_candidate: torch.Tensor,
    public_latch: torch.Tensor,
) -> torch.Tensor:
    """Select SEARCH before handoff and the preregistered CLOSE arm afterwards."""

    if not isinstance(search_action, torch.Tensor) or search_action.ndim != 2:
        raise ValueError("search_action must be a rank-two tensor")
    batch, action_dim = search_action.shape
    if action_dim != ACTION_DIM or not search_action.dtype.is_floating_point:
        raise ValueError(f"search_action must be floating [{batch},{ACTION_DIM}]")
    for name, value in (
        ("baseline_action", baseline_action),
        ("candidate_action", candidate_action),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != search_action.shape
            or value.dtype != search_action.dtype
            or value.device != search_action.device
        ):
            raise ValueError(f"{name} must match search_action")
    for name, value in (
        ("option_active", option_active),
        ("assignment_candidate", assignment_candidate),
        ("public_latch", public_latch),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (batch,)
            or value.dtype != torch.bool
            or value.device != search_action.device
        ):
            raise ValueError(f"{name} must be a co-located bool[{batch}] tensor")
    if not bool(
        torch.isfinite(search_action).all()
        and torch.isfinite(baseline_action).all()
        and torch.isfinite(candidate_action).all()
    ):
        raise FloatingPointError("an A/B action contains NaN or infinity")
    if bool(public_latch.any()) and not torch.equal(
        baseline_action[public_latch], candidate_action[public_latch]
    ):
        raise RuntimeError(
            "baseline and candidate disagree after latch; frozen LIFT is not common"
        )
    selected_close = torch.where(
        assignment_candidate.unsqueeze(-1), candidate_action, baseline_action
    )
    return torch.where(option_active.unsqueeze(-1), selected_close, search_action)


def _require_metadata(metadata: Mapping[str, Any]) -> tuple[int, int, str]:
    if not isinstance(metadata, dict):
        raise TypeError("artifact metadata must be a plain dictionary")
    if set(metadata) != set(METADATA_NAMES):
        raise ValueError(f"artifact metadata must contain exactly {METADATA_NAMES}")
    _require_json_safe(metadata, name="metadata")
    fixed = {
        "kind": ARTIFACT_KIND,
        "format_version": FORMAT_VERSION,
        "collection_contract": COLLECTION_CONTRACT,
        "assignment_contract": ASSIGNMENT_CONTRACT,
        "assignment_salt": ASSIGNMENT_SALT,
        "handoff_min_score": HANDOFF_MIN_SCORE,
        "handoff_hold_steps": HANDOFF_HOLD_STEPS,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "max_episode_actions": MAX_EPISODE_ACTIONS,
        "kit_args": KIT_ARGS,
    }
    for key, expected in fixed.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata field {key!r} changed")
    seed = metadata.get("seed")
    num_envs = metadata.get("num_envs")
    replicate = metadata.get("replicate")
    if not _is_int(seed):
        raise TypeError("metadata seed must be an integer")
    if not _is_int(num_envs) or num_envs < 2 or num_envs % 2:
        raise ValueError("metadata num_envs must be a positive even integer")
    if replicate not in {"a", "b"}:
        raise ValueError("metadata replicate must be 'a' or 'b'")
    for key in ("baseline_checkpoint", "candidate_checkpoint", "search_checkpoint"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"metadata {key!r} must be a non-empty path")
    for key in SHA_METADATA_NAMES:
        _require_sha256(metadata.get(key), name=f"metadata.{key}")
    if metadata["baseline_actor_sha256"] == metadata["candidate_actor_sha256"]:
        raise ValueError("A/B actor checkpoints are byte-identical")
    if metadata["baseline_task_contract_sha256"] != metadata[
        "candidate_task_contract_sha256"
    ]:
        raise ValueError("A/B task contracts differ")
    source_sha = _require_sha_mapping(
        metadata["source_sha256"], name="metadata.source_sha256"
    )
    source_path_contract = hashlib.sha256(
        "\0".join(sorted(source_sha)).encode("utf-8")
    ).hexdigest()
    if (
        len(source_sha) != SOURCE_PATH_COUNT
        or source_path_contract != SOURCE_PATH_SET_SHA256
    ):
        raise ValueError("metadata.source_sha256 path contract changed")
    runtime_assets = _require_sha_mapping(
        metadata["runtime_asset_sha256"], name="metadata.runtime_asset_sha256"
    )
    if set(runtime_assets) != set(RUNTIME_ASSET_NAMES):
        raise ValueError(
            f"metadata.runtime_asset_sha256 must contain exactly {RUNTIME_ASSET_NAMES}"
        )
    _require_git_sha(
        metadata["flashsac_upstream_commit"],
        name="metadata.flashsac_upstream_commit",
    )
    fork_commit = _require_git_sha(
        metadata["flashsac_fork_commit"], name="metadata.flashsac_fork_commit"
    )
    git = metadata["git"]
    if not isinstance(git, dict) or set(git) != set(GIT_METADATA_NAMES):
        raise ValueError(f"metadata.git must contain exactly {GIT_METADATA_NAMES}")
    _require_git_sha(git["commit"], name="metadata.git.commit")
    if not isinstance(git["branch"], str) or not git["branch"]:
        raise ValueError("metadata.git.branch must be non-empty")
    if git["source_files_dirty"] is not False or git["flashsac_dirty"] is not False:
        raise ValueError("A/B evidence cannot authenticate a dirty source tree")
    if _require_git_sha(git["flashsac_commit"], name="metadata.git.flashsac_commit") != fork_commit:
        raise ValueError("FlashSAC submodule commit differs from the loaded fork")
    runtime = metadata["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_METADATA_NAMES):
        raise ValueError(
            f"metadata.runtime must contain exactly {RUNTIME_METADATA_NAMES}"
        )
    if runtime["seed"] != seed:
        raise ValueError("metadata.runtime.seed differs from the trial seed")
    if not _is_int(runtime["cudnn"]) or not _is_int(runtime["cuda_device_index"]):
        raise TypeError("runtime CUDA integer fields must be integers")
    capability = runtime["cuda_device_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or not all(_is_int(item) for item in capability)
    ):
        raise ValueError("runtime CUDA capability must be a two-integer list")
    packages = runtime["packages"]
    if not isinstance(packages, dict) or set(packages) != set(RUNTIME_PACKAGE_NAMES):
        raise ValueError(
            f"metadata.runtime.packages must contain exactly {RUNTIME_PACKAGE_NAMES}"
        )
    if not all(isinstance(value, str) and value for value in packages.values()):
        raise ValueError("runtime package versions must be non-empty strings")
    for key in (
        "python",
        "torch",
        "cuda",
        "cuda_device_name",
        "isaac_sim",
        "nvidia_smi_inventory",
        "platform",
    ):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise ValueError(f"metadata.runtime.{key} must be non-empty")
    return seed, num_envs, replicate


def validate_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a CPU, weights-only first-episode trial artifact."""

    if not isinstance(artifact, Mapping) or set(artifact) != {"metadata", "tensors"}:
        raise ValueError("artifact must contain exactly metadata and tensors")
    metadata = artifact["metadata"]
    seed, num_envs, replicate = _require_metadata(metadata)
    tensors = artifact["tensors"]
    if not isinstance(tensors, Mapping) or set(tensors) != set(TENSOR_NAMES):
        raise ValueError(f"artifact tensors must contain exactly {TENSOR_NAMES}")
    bool_names = {"assignment_candidate", "triggered", *OUTCOME_NAMES}
    long_names = {"env_slot", "trigger_step", "episode_length"}
    checked: dict[str, torch.Tensor] = {}
    for name in TENSOR_NAMES:
        value = tensors[name]
        expected_dtype = (
            torch.bool
            if name in bool_names
            else torch.long
            if name in long_names
            else torch.float32
        )
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (num_envs,)
            or value.dtype != expected_dtype
            or value.device.type != "cpu"
        ):
            raise ValueError(
                f"artifact tensor {name!r} must be CPU {expected_dtype}[{num_envs}]"
            )
        checked[name] = value

    expected_assignment = assignment_candidate_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    if not torch.equal(checked["env_slot"], torch.arange(num_envs)):
        raise ValueError("artifact env_slot must be canonical arange(num_envs)")
    if not torch.equal(checked["assignment_candidate"], expected_assignment):
        raise ValueError("artifact assignment differs from its preregistered mask")
    triggered = checked["triggered"]
    trigger_step = checked["trigger_step"]
    trigger_score = checked["trigger_score"]
    if bool((trigger_step[triggered] < HANDOFF_HOLD_STEPS - 1).any()) or bool(
        (trigger_step[~triggered] != -1).any()
    ):
        raise ValueError("trigger_step violates the debounced triggered mask")
    if bool((trigger_step[triggered] >= MAX_EPISODE_ACTIONS).any()):
        raise ValueError("trigger_step exceeds the first-episode horizon")
    if not bool(torch.isfinite(trigger_score).all()):
        raise FloatingPointError("trigger_score is not finite")
    if bool(
        (
            (trigger_score[triggered] < HANDOFF_MIN_SCORE)
            | (trigger_score[triggered] > 1.0)
        ).any()
    ) or bool((trigger_score[~triggered] != 0.0).any()):
        raise ValueError("trigger_score disagrees with the public handoff")
    episode_length = checked["episode_length"]
    if bool(
        ((episode_length < 1) | (episode_length > MAX_EPISODE_ACTIONS)).any()
    ):
        raise ValueError("episode_length is outside the authored horizon")
    if bool((trigger_step[triggered] >= episode_length[triggered]).any()):
        raise ValueError("trigger_step must precede the terminal episode length")
    if not bool(torch.isfinite(checked["max_true_clearance_m"]).all()):
        raise FloatingPointError("true clearance is not finite")

    success = checked["success"]
    failure = checked["failure"]
    timeout = checked["time_out"]
    if bool(((success & failure) | (success & timeout) | (failure & timeout)).any()):
        raise ValueError("primary terminal outcomes overlap")
    if not bool((success | failure | timeout).all()):
        raise ValueError("primary terminal outcomes do not partition first episodes")
    failure_sources = (
        checked["dropped"]
        | checked["unsafe_force"]
        | checked["unlatched_clearance_ge_5cm"]
    )
    if not torch.equal(failure, failure_sources):
        raise ValueError("failure differs from its task-authored sources")
    if bool((timeout & (episode_length != MAX_EPISODE_ACTIONS)).any()):
        raise ValueError("time_out must occur at the exact authored horizon")
    maximum = checked["max_true_clearance_m"]
    ever_20cm = maximum >= 0.20
    if not torch.equal(checked["ever_clearance_ge_20cm"], ever_20cm):
        raise ValueError("20 cm event disagrees with true-clearance maxima")
    if bool(
        (
            success
            & (
                ~checked["ever_grasped"]
                | ~checked["ever_clearance_ge_20cm"]
                | checked["dropped"]
                | checked["unsafe_force"]
                | checked["unlatched_clearance_ge_5cm"]
            )
        ).any()
    ):
        raise ValueError("success violates grasp, clearance, or safety truth")
    if bool(
        (checked["unlatched_clearance_ge_5cm"] & (maximum < 0.05)).any()
    ):
        raise ValueError("unlatched 5 cm event disagrees with true-clearance maxima")
    return {"metadata": dict(metadata), "tensors": checked}


def build_artifact(
    *, metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    normalized = {
        "metadata": dict(metadata),
        "tensors": {name: tensors[name].detach().cpu().clone() for name in TENSOR_NAMES},
    }
    return validate_artifact(normalized)


def _event_summary(tensors: Mapping[str, torch.Tensor], mask: torch.Tensor) -> dict[str, Any]:
    env_ids = mask.nonzero(as_tuple=False).flatten().tolist()
    result: dict[str, Any] = {"episodes": len(env_ids), "env_ids": env_ids}
    for name in OUTCOME_NAMES:
        selected = mask & tensors[name]
        ids = selected.nonzero(as_tuple=False).flatten().tolist()
        result[name] = len(ids)
        result[f"{name}_env_ids"] = ids
    if env_ids:
        result["success_rate"] = result["success"] / len(env_ids)
    else:
        result["success_rate"] = None
    return result


def summarize_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_artifact(artifact)
    tensors = validated["tensors"]
    candidate = tensors["assignment_candidate"]
    baseline = ~candidate
    triggered = tensors["triggered"]
    return {
        "all_assigned": {
            "baseline": _event_summary(tensors, baseline),
            "candidate": _event_summary(tensors, candidate),
        },
        "triggered": {
            "baseline": _event_summary(tensors, baseline & triggered),
            "candidate": _event_summary(tensors, candidate & triggered),
        },
        "untriggered": {
            "baseline": int((baseline & ~triggered).sum()),
            "candidate": int((candidate & ~triggered).sum()),
        },
    }


def validate_report(
    report: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    published: bool = False,
) -> dict[str, Any]:
    """Bind the human-readable report exactly to the tensor artifact."""

    validated = validate_artifact(artifact)
    if not isinstance(report, dict):
        raise TypeError("A/B report must be a plain dictionary")
    expected_names = PUBLISHED_REPORT_NAMES if published else REPORT_NAMES
    if set(report) != set(expected_names):
        raise ValueError(f"A/B report must contain exactly {expected_names}")
    _require_json_safe(report, name="report")
    metadata = validated["metadata"]
    fixed = {
        "kind": REPORT_KIND,
        "status": "complete",
        "seed": metadata["seed"],
        "replicate": metadata["replicate"],
        "num_envs": metadata["num_envs"],
        "collection_contract": COLLECTION_CONTRACT,
        "baseline_actor_sha256": metadata["baseline_actor_sha256"],
        "candidate_actor_sha256": metadata["candidate_actor_sha256"],
        "common_frozen_lift_actor_sha256": metadata[
            "common_frozen_lift_actor_sha256"
        ],
        "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
    }
    for key, expected in fixed.items():
        if report[key] != expected:
            raise ValueError(f"report field {key!r} differs from the artifact")
    vector_steps = report["vector_steps"]
    if not _is_int(vector_steps) or vector_steps != int(
        validated["tensors"]["episode_length"].max().item()
    ):
        raise ValueError("report vector_steps differs from first-episode lengths")
    if report["summary"] != summarize_artifact(validated):
        raise ValueError("report summary differs from the tensor artifact")
    if published:
        _require_sha256(report["artifact_sha256"], name="report.artifact_sha256")
        if not isinstance(report["artifact_output"], str) or not report["artifact_output"]:
            raise ValueError("report artifact_output must be a non-empty path")
    return dict(report)


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode(
        "utf-8"
    )


def _owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _unlink_if_same_file(path: Path, temporary: Path) -> None:
    """Remove only a final link that still names our own temporary inode."""

    try:
        if _owned(path) and _owned(temporary) and os.path.samefile(path, temporary):
            path.unlink()
    except FileNotFoundError:
        pass


def publish_artifact_and_report_no_clobber(
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    artifact_output: Path,
    report_output: Path,
) -> str:
    """Transactionally publish immutable PT+JSON evidence."""

    validated = validate_artifact(artifact)
    validated_report = validate_report(report, validated)
    artifact_output = Path(os.path.abspath(os.fspath(artifact_output)))
    report_output = Path(os.path.abspath(os.fspath(report_output)))
    if artifact_output == report_output:
        raise ValueError("artifact and report outputs must differ")
    for path in (artifact_output, report_output):
        if _owned(path):
            raise FileExistsError(f"A/B evidence output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    intended_links: dict[Path, Path] = {}
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{artifact_output.name}.tmp-", dir=artifact_output.parent
        )
        os.close(descriptor)
        artifact_temp = Path(name)
        temporary_paths.append(artifact_temp)
        intended_links[artifact_output] = artifact_temp
        with artifact_temp.open("wb") as stream:
            torch.save(validated, stream)
            stream.flush()
            os.fsync(stream.fileno())
        artifact_sha = sha256_file(artifact_temp)
        final_report = dict(validated_report)
        final_report["artifact_sha256"] = artifact_sha
        final_report["artifact_output"] = str(artifact_output)
        validate_report(final_report, validated, published=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{report_output.name}.tmp-", dir=report_output.parent
        )
        report_temp = Path(name)
        temporary_paths.append(report_temp)
        intended_links[report_output] = report_temp
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_strict_json_bytes(final_report))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(artifact_temp, artifact_output)
        os.link(report_temp, report_output)
        for directory in {artifact_output.parent, report_output.parent}:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return artifact_sha
    except BaseException:
        for path, temporary in reversed(tuple(intended_links.items())):
            _unlink_if_same_file(path, temporary)
        raise
    finally:
        for path in temporary_paths:
            if _owned(path):
                path.unlink()


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    _require_json_safe(dict(payload), name="JSON payload")
    serialized = _strict_json_bytes(payload)
    output = Path(os.path.abspath(os.fspath(output)))
    if _owned(output):
        raise FileExistsError(f"JSON output already exists: {output}")
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
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        _unlink_if_same_file(output, temporary)
        raise
    finally:
        if _owned(temporary):
            temporary.unlink()


__all__ = [
    "ACTION_DIM",
    "ARTIFACT_KIND",
    "ASSIGNMENT_CONTRACT",
    "ASSIGNMENT_SALT",
    "COLLECTION_CONTRACT",
    "FORMAT_VERSION",
    "HANDOFF_HOLD_STEPS",
    "HANDOFF_MIN_SCORE",
    "MAX_EPISODE_ACTIONS",
    "KIT_ARGS",
    "OBSERVATION_DIM",
    "OUTCOME_NAMES",
    "REPORT_KIND",
    "RUNTIME_ASSET_NAMES",
    "SOURCE_PATH_COUNT",
    "SOURCE_PATH_SET_SHA256",
    "TENSOR_NAMES",
    "assignment_candidate_mask",
    "build_artifact",
    "publish_artifact_and_report_no_clobber",
    "publish_json_no_clobber",
    "select_executed_action",
    "sha256_file",
    "summarize_artifact",
    "validate_artifact",
    "validate_report",
]
