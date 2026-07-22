#!/usr/bin/env python3
"""Collect Candidate40's verified-arm development A/B artifacts.

Both assignments execute one exact common Candidate39 route.  Candidate40
assignment is consumed only by the pure post-eligibility arm7 supervisor;
the sealed fixed hand residual is applied to every slot.  The collector keeps
all option-active pre-eligibility rows, one SEARCH-side first-latch witness,
and every row from first eligibility through the authoritative terminal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

import torch

import candidate40_verified_arm_episode as artifact_contract
import candidate40_verified_arm_handoff as verifier
from collect_candidate39_episode_residual_ab import _append_rows, _cat_rows
from collect_online_close_ab import (
    RUNTIME_ASSET_EXPECTED_SHA256,
    _checkpoint_directory,
    _checkpoint_hashes,
    _load_routed_agent,
    _regular_file,
    _validate_live_task,
    git_provenance,
    runtime_asset_fingerprints,
    runtime_provenance,
    source_fingerprints as ab_source_fingerprints,
)
from online_handoff import update_online_handoff
import option_residual_screen


NUM_ENVS = 64
ALLOWED_NUM_ENVS = (8, NUM_ENVS)
EXPECTED_V6_SHA256 = {
    "actor.pt": artifact_contract.V6_ACTOR_SHA256,
    "task_contract.json": artifact_contract.V6_TASK_CONTRACT_SHA256,
    "torch_bridge_state.pt": artifact_contract.V6_BRIDGE_STATE_SHA256,
    "frozen_lift_actor.pt": artifact_contract.FROZEN_LIFT_ACTOR_SHA256,
}
EXTRA_SOURCE_FILES = tuple(
    sorted(
        set(artifact_contract.IMPLEMENTATION_SOURCE_FILES)
        | {
            "scripts/flashsac/option_residual_screen.py",
            artifact_contract.VALIDATION_PLAN,
            "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/public_gate_state.py",
        }
    )
)
CANDIDATE40_RUN_ORDER = ((335, "a"), (335, "b"), (336, "b"), (336, "a"))


@dataclass(frozen=True)
class CollectionSpec:
    repository_root: Path
    v6_checkpoint: Path
    search_checkpoint: Path
    fixed_direction: Path
    seed: int
    replicate: str
    num_envs: int
    kit_args: str
    output_stem: Path
    artifact_output: Path
    report_output: Path
    checkpoint_sha256: Mapping[str, Mapping[str, str] | str]
    source_sha256: Mapping[str, str]
    collection_commit: str
    treatment: torch.Tensor
    assignment_rank: torch.Tensor
    assignment_mask_sha256: str
    fixed_z: torch.Tensor
    smoke_artifact_sha256: str | None
    smoke_report_sha256: str | None


def _owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _require_exact(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        valid = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
        )
    else:
        valid = actual == expected and type(actual) is type(expected)
    if not valid:
        raise ValueError(f"Candidate40 requires exact {name}={expected!r}")


def require_git_ancestor(root: Path, ancestor: str, descendant: str = "HEAD") -> None:
    """Fail unless ``ancestor`` is an ancestor of the clean collection HEAD."""

    if not isinstance(ancestor, str) or len(ancestor) != 40:
        raise ValueError("implementation ancestor must be a full Git SHA")
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"collection commit {descendant!r} does not descend from {ancestor}"
        )


def _git_blob_sha256(root: Path, *, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ("git", "show", f"{commit}:{relative}"),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(f"source is absent from authority commit: {relative}")
    raw = completed.stdout
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError:
        lines = []
    if lines and lines[0] == "version https://git-lfs.github.com/spec/v1":
        if (
            len(lines) != 3
            or not lines[1].startswith("oid sha256:")
            or not lines[2].startswith("size ")
        ):
            raise ValueError(f"malformed Git-LFS pointer: {relative}")
        digest = lines[1].removeprefix("oid sha256:")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"malformed Git-LFS oid: {relative}")
        try:
            expected_size = int(lines[2].removeprefix("size "))
        except ValueError as error:
            raise ValueError(f"malformed Git-LFS size: {relative}") from error
        current = _regular_file(root / relative, label=f"Git-LFS source {relative}")
        if expected_size < 0 or current.stat().st_size != expected_size:
            raise ValueError(f"Git-LFS source size changed: {relative}")
        if artifact_contract.sha256_file(current) != digest:
            raise ValueError(f"Git-LFS working-tree content changed: {relative}")
        return digest
    return hashlib.sha256(raw).hexdigest()


def validate_implementation_source_authority(
    root: Path,
    source_sha256: Mapping[str, str],
    *,
    implementation_commit: str,
) -> None:
    """Bind every runtime source to the pre-simulator implementation seal."""

    flash_prefix = "third_party/FlashSAC/"
    generated = {
        path: digest
        for path, digest in RUNTIME_ASSET_EXPECTED_SHA256.items()
        if not Path(path).is_absolute()
    }
    for relative, digest in source_sha256.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"source manifest contains an unsafe path: {relative}")
        if relative == artifact_contract.VALIDATION_PLAN:
            # The final seal is necessarily committed after the implementation
            # commit.  Its exact current bytes are validated independently.
            continue
        if relative.startswith(flash_prefix):
            flash_relative = relative.removeprefix(flash_prefix)
            authority = _git_blob_sha256(
                root / "third_party/FlashSAC",
                commit=artifact_contract.FLASHSAC_FORK_COMMIT,
                relative=flash_relative,
            )
        elif relative in generated:
            authority = generated[relative]
        else:
            authority = _git_blob_sha256(
                root, commit=implementation_commit, relative=relative
            )
        if authority != digest:
            raise ValueError(
                f"runtime source differs from implementation authority: {relative}"
            )


def _validate_smoke_prerequisite(root: Path) -> tuple[str, str]:
    artifact_path = _regular_file(
        root / artifact_contract.SMOKE_ARTIFACT_PATH,
        label="canonical Candidate40 smoke artifact",
    )
    report_path = _regular_file(
        root / artifact_contract.SMOKE_REPORT_PATH,
        label="canonical Candidate40 smoke report",
    )
    artifact_sha = artifact_contract.sha256_file(artifact_path)
    report_sha = artifact_contract.sha256_file(report_path)
    try:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        raise ValueError("cannot load the canonical Candidate40 smoke pair") from error
    checked = artifact_contract.validate_artifact(artifact)
    artifact_contract.validate_report(report, checked, published=True)
    if report["artifact_sha256"] != artifact_sha:
        raise ValueError("Candidate40 smoke report does not bind its artifact bytes")
    if Path(report["artifact_output"]).resolve() != artifact_path.resolve():
        raise ValueError("Candidate40 smoke report points to a non-canonical artifact")
    meta, episodes = checked["metadata"], checked["episodes"]
    if (meta["seed"], meta["replicate"], meta["num_envs"]) != (334, "b", 8):
        raise ValueError("Candidate40 smoke run identity changed")
    if meta["runtime_asset_sha256"] != RUNTIME_ASSET_EXPECTED_SHA256:
        raise ValueError("Candidate40 smoke runtime assets changed")
    treatment_eligible = episodes["treatment"] & episodes["eligible"]
    if not bool(treatment_eligible.any()):
        raise ValueError("Candidate40 smoke did not exercise a treatment overlay")
    forbidden = (
        episodes["dropped"]
        | episodes["unsafe_force"]
        | episodes["pre_release_launch"]
    )
    if bool((treatment_eligible & forbidden).any()):
        raise ValueError("Candidate40 smoke failed a treatment safety prerequisite")
    if bool(episodes["new_abs_action_ge_0999"].any()) or bool(
        (episodes["treatment_pre_enable_nonzero_arm_rows"] != 0).any()
    ):
        raise ValueError("Candidate40 smoke failed an action-algebra prerequisite")
    return artifact_sha, report_sha


def _validate_causal_receipts(root: Path) -> dict[str, str]:
    receipts = (
        (artifact_contract.CANDIDATE39_RESULT, artifact_contract.CANDIDATE39_RESULT_SHA256),
        (artifact_contract.CANDIDATE39_REPORT, artifact_contract.CANDIDATE39_REPORT_SHA256),
        (
            artifact_contract.FIXED_DIRECTION_MANIFEST,
            artifact_contract.FIXED_DIRECTION_MANIFEST_SHA256,
        ),
    )
    loaded: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for relative, expected_sha in receipts:
        path = _regular_file(root / relative, label=f"registered causal receipt {relative}")
        actual_sha = artifact_contract.sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"registered causal receipt changed: {relative}")
        hashes[relative] = actual_sha
        try:
            loaded[relative] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"registered causal receipt is not JSON: {relative}") from error
    result = loaded[artifact_contract.CANDIDATE39_RESULT]
    report = loaded[artifact_contract.CANDIDATE39_REPORT]
    manifest = loaded[artifact_contract.FIXED_DIRECTION_MANIFEST]
    if (
        result.get("kind")
        != "pick_tool_candidate39_fixed_direction_development_validation_result_v1"
        or result.get("status") != "sealed_rejected_before_formal_screen"
        or not isinstance(result.get("decision"), dict)
        or result["decision"].get("analyzer_decision") != "reject_fixed_direction"
        or result["decision"].get("all_gates_pass") is not False
    ):
        raise ValueError("Candidate39 result semantic identity changed")
    if (
        report.get("kind")
        != "pick_tool_candidate39_fixed_direction_development_validation_v1"
        or report.get("status") != "complete"
        or report.get("decision") != "reject_fixed_direction"
    ):
        raise ValueError("Candidate39 combined report semantic identity changed")
    fixed = manifest.get("fixed_direction", {})
    if (
        manifest.get("kind") != "pick_tool_candidate39_fixed_direction_manifest_v1"
        or manifest.get("status") != "sealed_before_fixed_direction_validation"
        or fixed.get("path") != artifact_contract.FIXED_DIRECTION_PATH
        or fixed.get("sha256") != artifact_contract.FIXED_DIRECTION_SHA256
        or fixed.get("payload_kind") != artifact_contract.FIXED_DIRECTION_PAYLOAD_KIND
    ):
        raise ValueError("Candidate39 fixed-direction manifest semantics changed")
    return hashes


def _canonical_dev_stem(root: Path, seed: int, replicate: str) -> Path:
    return (
        root
        / f"logs/flashsac/pick_tool/52_c40_verified_arm_dev_s{seed}_{replicate}/trial"
    ).resolve()


def _validate_prior_authority(
    metadata: Mapping[str, Any],
    *,
    source_sha256: Mapping[str, str],
    smoke_artifact_sha256: str,
    smoke_report_sha256: str,
    collection_commit: str,
) -> None:
    if metadata.get("source_sha256") != dict(source_sha256):
        raise ValueError("prior Candidate40 artifact used a different source authority")
    if (
        metadata.get("smoke_artifact_sha256") != smoke_artifact_sha256
        or metadata.get("smoke_report_sha256") != smoke_report_sha256
    ):
        raise ValueError("prior Candidate40 artifact used different smoke authority")
    git = metadata.get("git")
    if not isinstance(git, Mapping) or git.get("commit") != collection_commit:
        raise ValueError("prior Candidate40 artifact used a different collection commit")
    if metadata.get("runtime_asset_sha256") != RUNTIME_ASSET_EXPECTED_SHA256:
        raise ValueError("prior Candidate40 artifact used different runtime assets")


def _validate_published_pair(
    artifact_path: Path,
    report_path: Path,
    *,
    seed: int,
    replicate: str,
    source_sha256: Mapping[str, str],
    smoke_artifact_sha256: str,
    smoke_report_sha256: str,
    collection_commit: str,
) -> None:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checked = artifact_contract.validate_artifact(artifact)
    artifact_contract.validate_report(report, checked, published=True)
    digest = artifact_contract.sha256_file(artifact_path)
    if report["artifact_sha256"] != digest or Path(
        report["artifact_output"]
    ).resolve() != artifact_path.resolve():
        raise ValueError("prior Candidate40 report does not bind canonical artifact bytes")
    metadata = checked["metadata"]
    if (metadata["seed"], metadata["replicate"], metadata["num_envs"]) != (
        seed,
        replicate,
        64,
    ):
        raise ValueError("prior Candidate40 development run identity changed")
    _validate_prior_authority(
        metadata,
        source_sha256=source_sha256,
        smoke_artifact_sha256=smoke_artifact_sha256,
        smoke_report_sha256=smoke_report_sha256,
        collection_commit=collection_commit,
    )


def _validate_run_order(
    root: Path,
    *,
    seed: int,
    replicate: str,
    output_stem: Path,
    source_sha256: Mapping[str, str],
    smoke_artifact_sha256: str,
    smoke_report_sha256: str,
    collection_commit: str,
) -> None:
    identity = (seed, replicate)
    if identity not in CANDIDATE40_RUN_ORDER:
        raise ValueError("run is outside the sealed Candidate40 development order")
    index = CANDIDATE40_RUN_ORDER.index(identity)
    expected_current = _canonical_dev_stem(root, seed, replicate)
    if output_stem.resolve() != expected_current:
        raise ValueError(f"development output_stem must be exactly {expected_current}")
    for prior_seed, prior_replicate in CANDIDATE40_RUN_ORDER[:index]:
        prior_stem = _canonical_dev_stem(root, prior_seed, prior_replicate)
        try:
            artifact_path = _regular_file(
                Path(f"{prior_stem}.pt"),
                label=f"prior Candidate40 artifact {prior_seed}{prior_replicate}",
            )
            report_path = _regular_file(
                Path(f"{prior_stem}.json"),
                label=f"prior Candidate40 report {prior_seed}{prior_replicate}",
            )
        except FileNotFoundError as error:
            raise ValueError(
                f"Candidate40 run order prerequisite is missing: {prior_seed}{prior_replicate}"
            ) from error
        _validate_published_pair(
            artifact_path,
            report_path,
            seed=prior_seed,
            replicate=prior_replicate,
            source_sha256=source_sha256,
            smoke_artifact_sha256=smoke_artifact_sha256,
            smoke_report_sha256=smoke_report_sha256,
            collection_commit=collection_commit,
        )
    for later_seed, later_replicate in CANDIDATE40_RUN_ORDER[index + 1 :]:
        later_stem = _canonical_dev_stem(root, later_seed, later_replicate)
        if _owned(Path(f"{later_stem}.pt")) or _owned(Path(f"{later_stem}.json")):
            raise ValueError(
                f"later Candidate40 output already exists out of order: {later_seed}{later_replicate}"
            )


def build_spec(args: argparse.Namespace) -> CollectionSpec:
    """Freeze assignment, payloads, receipts, and smoke gate before AppLauncher."""

    root = Path(__file__).resolve().parents[2]
    sealed_plan = artifact_contract.validate_sealed_plan()
    require_git_ancestor(
        root, sealed_plan["implementation_seal"]["implementation_commit"]
    )
    if not isinstance(args.seed, int) or isinstance(args.seed, bool):
        raise TypeError("seed must be an integer")
    if type(args.num_envs) is not int or args.num_envs not in ALLOWED_NUM_ENVS:
        raise ValueError(f"num_envs must be one of {ALLOWED_NUM_ENVS}")
    if args.replicate not in {"a", "b"}:
        raise ValueError("replicate must be a or b")
    if args.num_envs == 8 and (args.seed, args.replicate) != (334, "b"):
        raise ValueError("the only Candidate40 smoke is exact run 334b/8-env")
    if args.num_envs == 64 and args.seed not in {335, 336}:
        raise ValueError("Candidate40 development uses only seeds 335 and 336")
    for name, expected in (
        ("window_steps", artifact_contract.WINDOW_STEPS),
        ("token_scale", artifact_contract.TOKEN_SCALE),
        ("distal_scale", artifact_contract.DISTAL_SCALE),
        ("raw_z_abs_cap", artifact_contract.RAW_Z_ABS_CAP),
        ("token_component_cap", artifact_contract.TOKEN_COMPONENT_CAP),
        ("distal_component_cap", artifact_contract.DISTAL_COMPONENT_CAP),
        ("pre_tanh_l2_cap", artifact_contract.PRE_TANH_L2_CAP),
        ("verify_steps", verifier.VERIFY_STEPS),
    ):
        _require_exact(name, getattr(args, name), expected)
    if getattr(args, "kit_args", None) != artifact_contract.KIT_ARGS:
        raise ValueError(
            f"Candidate40 requires exact --kit_args={artifact_contract.KIT_ARGS!r}"
        )

    v6 = _checkpoint_directory(args.v6_checkpoint, label="exact V6 checkpoint")
    search = _regular_file(args.search_checkpoint, label="exact SEARCH checkpoint")
    fixed = _regular_file(args.fixed_direction, label="sealed Candidate39 fixed_z")
    canonical_inputs = {
        "V6 checkpoint": (v6, root / artifact_contract.V6_CHECKPOINT_PATH),
        "SEARCH checkpoint": (search, root / artifact_contract.SEARCH_CHECKPOINT_PATH),
        "fixed_z payload": (fixed, root / artifact_contract.FIXED_DIRECTION_PATH),
    }
    for label, (actual, expected) in canonical_inputs.items():
        if actual.resolve() != expected.resolve():
            raise ValueError(f"Candidate40 {label} is not the registered canonical path")
    causal_hashes = _validate_causal_receipts(root)
    checkpoint_sha256: dict[str, Mapping[str, str] | str] = {
        "v6": _checkpoint_hashes(v6),
        "search": artifact_contract.sha256_file(search),
        "fixed_direction": artifact_contract.sha256_file(fixed),
        **causal_hashes,
    }
    if checkpoint_sha256["v6"] != EXPECTED_V6_SHA256:
        raise ValueError("Candidate40 requires the exact sealed V6 checkpoint")
    if checkpoint_sha256["search"] != artifact_contract.SEARCH_CHECKPOINT_SHA256:
        raise ValueError("Candidate40 requires the exact sealed SEARCH checkpoint")
    if checkpoint_sha256["fixed_direction"] != artifact_contract.FIXED_DIRECTION_SHA256:
        raise ValueError("Candidate40 requires the exact sealed fixed_z payload")
    fixed_payload = artifact_contract.load_fixed_direction(fixed)
    frozen_source_sha256 = source_fingerprints(root)
    implementation_commit = sealed_plan["implementation_seal"][
        "implementation_commit"
    ]
    validate_implementation_source_authority(
        root,
        frozen_source_sha256,
        implementation_commit=implementation_commit,
    )
    # These bytes are required before AppLauncher as well as rechecked inside
    # and after collection, so a stale generated USD cannot consume a seed.
    runtime_asset_fingerprints(root)
    prelaunch_git = git_provenance(
        root, tuple(sorted(frozen_source_sha256))
    )
    if (
        prelaunch_git["commit"] == ""
        or prelaunch_git["branch"] != artifact_contract.REQUIRED_BRANCH
        or prelaunch_git["source_files_dirty"]
        or prelaunch_git["flashsac_dirty"]
        or prelaunch_git["flashsac_commit"] != artifact_contract.FLASHSAC_FORK_COMMIT
    ):
        raise ValueError(
            "Candidate40 pre-launch Git authority is dirty, detached, or changed"
        )
    collection_commit = str(prelaunch_git["commit"])

    output_stem = Path(os.path.abspath(os.fspath(args.output_stem)))
    if args.num_envs == 8:
        expected_smoke_stem = (root / artifact_contract.SMOKE_ARTIFACT_PATH).with_suffix("")
        if output_stem.resolve() != expected_smoke_stem.resolve():
            raise ValueError(f"smoke output_stem must be exactly {expected_smoke_stem}")
    smoke_artifact_sha: str | None = None
    smoke_report_sha: str | None = None
    if args.num_envs == 64:
        smoke_artifact_sha, smoke_report_sha = _validate_smoke_prerequisite(root)
        checkpoint_sha256[artifact_contract.SMOKE_ARTIFACT_PATH] = smoke_artifact_sha
        checkpoint_sha256[artifact_contract.SMOKE_REPORT_PATH] = smoke_report_sha
        _validate_run_order(
            root,
            seed=int(args.seed),
            replicate=str(args.replicate),
            output_stem=output_stem,
            source_sha256=frozen_source_sha256,
            smoke_artifact_sha256=smoke_artifact_sha,
            smoke_report_sha256=smoke_report_sha,
            collection_commit=collection_commit,
        )
    artifact_output = Path(f"{output_stem}.pt")
    report_output = Path(f"{output_stem}.json")
    for output in (artifact_output, report_output):
        if _owned(output):
            raise FileExistsError(f"Candidate40 output already exists: {output}")

    treatment = verifier.exact_balanced_treatment_mask(
        seed=args.seed, num_envs=args.num_envs, replicate=args.replicate
    )
    rank = verifier.assignment_rank(seed=args.seed, num_envs=args.num_envs)
    assignment_a = rank < (args.num_envs // 2)
    expected_from_rank = assignment_a if args.replicate == "a" else ~assignment_a
    if not torch.equal(treatment, expected_from_rank):
        raise RuntimeError("Candidate40 assignment mask disagrees with assignment rank")
    if int(treatment.sum()) != args.num_envs // 2:
        raise RuntimeError("Candidate40 assignment is not exactly balanced")
    assignment_sha = verifier.assignment_mask_sha256(treatment)

    return CollectionSpec(
        repository_root=root,
        v6_checkpoint=v6,
        search_checkpoint=search,
        fixed_direction=fixed,
        seed=int(args.seed),
        replicate=str(args.replicate),
        num_envs=int(args.num_envs),
        kit_args=artifact_contract.KIT_ARGS,
        output_stem=output_stem,
        artifact_output=artifact_output,
        report_output=report_output,
        checkpoint_sha256=checkpoint_sha256,
        source_sha256=frozen_source_sha256,
        collection_commit=collection_commit,
        treatment=treatment.clone(),
        assignment_rank=rank.clone(),
        assignment_mask_sha256=assignment_sha,
        fixed_z=fixed_payload["fixed_z"].clone(),
        smoke_artifact_sha256=smoke_artifact_sha,
        smoke_report_sha256=smoke_report_sha,
    )


def source_fingerprints(root: Path) -> dict[str, str]:
    result = ab_source_fingerprints(root)
    for relative in EXTRA_SOURCE_FILES:
        result[relative] = artifact_contract.sha256_file(
            _regular_file(root / relative, label=f"source {relative}")
        )
    return dict(sorted(result.items()))


def _current_hashes(spec: CollectionSpec) -> dict[str, Mapping[str, str] | str]:
    current: dict[str, Mapping[str, str] | str] = {
        "v6": _checkpoint_hashes(spec.v6_checkpoint),
        "search": artifact_contract.sha256_file(spec.search_checkpoint),
        "fixed_direction": artifact_contract.sha256_file(spec.fixed_direction),
    }
    for relative in (
        artifact_contract.CANDIDATE39_RESULT,
        artifact_contract.CANDIDATE39_REPORT,
        artifact_contract.FIXED_DIRECTION_MANIFEST,
    ):
        current[relative] = artifact_contract.sha256_file(
            _regular_file(
                spec.repository_root / relative,
                label=f"registered causal receipt {relative}",
            )
        )
    if spec.smoke_artifact_sha256 is not None:
        current[artifact_contract.SMOKE_ARTIFACT_PATH] = artifact_contract.sha256_file(
            _regular_file(
                spec.repository_root / artifact_contract.SMOKE_ARTIFACT_PATH,
                label="canonical Candidate40 smoke artifact",
            )
        )
        current[artifact_contract.SMOKE_REPORT_PATH] = artifact_contract.sha256_file(
            _regular_file(
                spec.repository_root / artifact_contract.SMOKE_REPORT_PATH,
                label="canonical Candidate40 smoke report",
            )
        )
    return current


def _read_transition_telemetry(
    info: Mapping[str, Any], *, num_envs: int, device: torch.device
) -> dict[str, torch.Tensor]:
    from evaluate import _require_vector, _terminal_mapping

    raw = _terminal_mapping(info)
    names_and_dtypes = {
        "is_grasped": torch.bool,
        "true_clearance": torch.float32,
        "grasp_quality": torch.float32,
        "hold_quality": torch.float32,
        "max_force": torch.float32,
        "object_lin_speed": torch.float32,
        "object_ang_speed": torch.float32,
    }
    result = {
        name: _require_vector(
            f"pick_tool_terminal[{name!r}]",
            raw.get(name),
            num_envs=num_envs,
            device=device,
            dtype=dtype,
        )
        for name, dtype in names_and_dtypes.items()
    }
    for name, value in result.items():
        if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"reset-before {name} is non-finite")
    for name in ("grasp_quality", "hold_quality"):
        if bool(((result[name] < 0.0) | (result[name] > 1.0)).any()):
            raise RuntimeError(f"reset-before {name} escaped [0,1]")
    for name in ("max_force", "object_lin_speed", "object_ang_speed"):
        if bool((result[name] < 0.0).any()):
            raise RuntimeError(f"reset-before {name} is negative")
    return result


def _arm_state(task: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arm_ids = task._arm_ids_t
    target_error = task.dof_targets[:, arm_ids] - task.robot.data.joint_pos[:, arm_ids]
    velocity = task.robot.data.joint_vel[:, arm_ids]
    if target_error.shape[1] != artifact_contract.ARM_ACTION_DIM:
        raise RuntimeError("live task arm telemetry is not seven-dimensional")
    return (
        target_error,
        velocity,
        target_error.abs().max(dim=-1).values,
        velocity.abs().max(dim=-1).values,
    )


def _increment(counter: torch.Tensor, mask: torch.Tensor) -> None:
    counter.add_(mask.long())


@torch.inference_mode()
def run_collection(
    spec: CollectionSpec, *, device_string: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Candidate40 collection requires CUDA, got {device}")

    # Recompute every CPU authority before importing the task.
    if not torch.equal(
        spec.treatment,
        verifier.exact_balanced_treatment_mask(
            seed=spec.seed, num_envs=spec.num_envs, replicate=spec.replicate
        ),
    ) or not torch.equal(
        spec.assignment_rank,
        verifier.assignment_rank(seed=spec.seed, num_envs=spec.num_envs),
    ):
        raise RuntimeError("pre-launch Candidate40 assignment changed")
    if verifier.assignment_mask_sha256(spec.treatment) != spec.assignment_mask_sha256:
        raise RuntimeError("pre-launch Candidate40 assignment receipt changed")
    fixed_payload = artifact_contract.load_fixed_direction(spec.fixed_direction)
    if not torch.equal(fixed_payload["fixed_z"], spec.fixed_z):
        raise RuntimeError("pre-launch Candidate40 fixed_z changed")

    from adapter import make_pick_tool_env
    from evaluate import (
        FULL_TASK_MODE,
        StrictEpisodeTracker,
        _load_diagnostic_approach_actor,
        _read_physical_truth,
        _seed_everything,
        physical_truth_from_terminal_info,
        validate_terminal_events,
    )
    from xhand_inhand.tasks.direct.pick_tool_token.public_gate_state import (
        PUBLIC_GATE_STATE_CONTRACT,
        PUBLIC_GATE_STATE_EXTRAS_KEY,
        public_gate_feature_tensor,
        validate_public_gate_feature_values,
    )

    _seed_everything(spec.seed)
    importlib.import_module("xhand_inhand.tasks")
    source_before = source_fingerprints(spec.repository_root)
    if source_before != dict(spec.source_sha256):
        raise RuntimeError("Candidate40 source authority changed after pre-launch freeze")
    runtime_assets_before = runtime_asset_fingerprints(spec.repository_root)
    for path, digest in runtime_assets_before.items():
        if not Path(path).is_absolute() and source_before.get(path) != digest:
            raise RuntimeError("source/runtime asset fingerprint disagreement")
    git_paths = tuple(
        sorted(
            set(source_before)
            | {path for path in runtime_assets_before if not Path(path).is_absolute()}
        )
    )
    git_before = git_provenance(spec.repository_root, git_paths)
    if git_before["source_files_dirty"] or git_before["flashsac_dirty"]:
        raise RuntimeError("Candidate40 collection requires committed clean sources")
    if (
        git_before["commit"] != spec.collection_commit
        or git_before["branch"] != artifact_contract.REQUIRED_BRANCH
        or git_before["flashsac_commit"] != artifact_contract.FLASHSAC_FORK_COMMIT
    ):
        raise RuntimeError("Candidate40 Git authority changed after pre-launch freeze")
    if PUBLIC_GATE_STATE_CONTRACT != artifact_contract.PUBLIC_GATE_STATE_CONTRACT:
        raise RuntimeError("public force-counter sidecar contract changed")

    env = make_pick_tool_env(
        num_envs=spec.num_envs,
        device=device_string,
        seed=spec.seed,
        strict=True,
        validate_finite=True,
    )
    try:
        _validate_live_task(env)
        agent, load = _load_routed_agent(env, spec.v6_checkpoint, seed=spec.seed)
        if (
            git_before["flashsac_commit"] != load["fork_commit"]
            or load["fork_commit"] != artifact_contract.FLASHSAC_FORK_COMMIT
            or load["frozen_semantic_sha256"]
            != artifact_contract.FROZEN_LIFT_SEMANTIC_SHA256
            or load["frozen_source_actor_sha256"]
            != artifact_contract.FROZEN_LIFT_SOURCE_ACTOR_SHA256
        ):
            raise RuntimeError("loaded FlashSAC fork differs from submodule HEAD")
        search_actor = _load_diagnostic_approach_actor(
            spec.search_checkpoint, device=env.device
        ).eval()
        if _current_hashes(spec) != spec.checkpoint_sha256:
            raise RuntimeError("a sealed Candidate40 input changed while loading")

        observation, reset_info = env.reset(
            seed=spec.seed, randomize_episode_lengths=False
        )
        if observation.shape != (spec.num_envs, artifact_contract.OBSERVATION_DIM):
            raise RuntimeError("Candidate40 reset violated obs115")
        public_counters = public_gate_feature_tensor(
            reset_info[PUBLIC_GATE_STATE_EXTRAS_KEY],
            num_envs=spec.num_envs,
            device=env.device,
        )
        validate_public_gate_feature_values(public_counters)
        initial_truth = _read_physical_truth(env.unwrapped, task_mode=FULL_TASK_MODE)
        if bool(initial_truth.grasped.any()):
            raise RuntimeError("Candidate40 reset unexpectedly begins grasped")
        current_clearance = initial_truth.clearance.clone()
        tracker = StrictEpisodeTracker(
            episodes=spec.num_envs,
            num_envs=spec.num_envs,
            device=env.device,
            initial_truth=initial_truth,
            task_mode=FULL_TASK_MODE,
        )

        treatment = spec.treatment.to(device=env.device)
        all_fixed = torch.ones(spec.num_envs, dtype=torch.bool, device=env.device)
        raw_z = spec.fixed_z.to(device=env.device).expand(spec.num_envs, -1).clone()
        scale = option_residual_screen.grouped_pre_tanh_scale(
            token_scale=artifact_contract.TOKEN_SCALE,
            distal_scale=artifact_contract.DISTAL_SCALE,
            dtype=torch.float32,
            device=env.device,
        )
        component_cap = torch.tensor(
            [artifact_contract.TOKEN_COMPONENT_CAP] * artifact_contract.TOKEN_ACTION_DIM
            + [artifact_contract.DISTAL_COMPONENT_CAP] * 5,
            dtype=torch.float32,
            device=env.device,
        )
        expected_delta = artifact_contract.expected_applied_delta().to(env.device)

        zeros_long = torch.zeros(spec.num_envs, dtype=torch.long, device=env.device)
        minus_one = torch.full_like(zeros_long, -1)
        ready_count = zeros_long.clone()
        option_active = torch.zeros(spec.num_envs, dtype=torch.bool, device=env.device)
        triggered = torch.zeros_like(option_active)
        trigger_step = minus_one.clone()
        trigger_score = torch.zeros(spec.num_envs, dtype=torch.float32, device=env.device)
        ever_latched = torch.zeros_like(option_active)
        first_latch_step = minus_one.clone()
        episode_step = zeros_long.clone()
        verifier_state = verifier.initial_verified_arm_state(
            spec.num_envs, device=env.device
        )
        authoritative_reset_mask = torch.zeros_like(option_active)
        first_eligible_step = minus_one.clone()
        verification_complete_step = minus_one.clone()
        first_arm_enabled_step = minus_one.clone()
        first_relock_step = minus_one.clone()
        first_reenable_step = minus_one.clone()
        intervention_steps = zeros_long.clone()
        stable_count_max = zeros_long.clone()
        arm_enable_count = zeros_long.clone()
        relock_count = zeros_long.clone()
        reenable_count = zeros_long.clone()
        trajectory_max_force = torch.zeros(
            spec.num_envs, dtype=torch.float32, device=env.device
        )
        first_eligible_clearance = torch.zeros_like(trajectory_max_force)
        pre_release_max_clearance = torch.zeros_like(trajectory_max_force)
        pre_release_launch = torch.zeros_like(option_active)
        option_preeligible_u5 = torch.zeros_like(option_active)

        violation_names = (
            "pre_eligibility_action_violations",
            "hand_invariance_violations",
            "treatment_arm_gate_violations",
            "control_route_violations",
            "task_action_reconstruction_violations",
            "fixed_residual_budget_violations",
            "action_bound_violations",
            "treatment_pre_enable_nonzero_arm_rows",
        )
        violations = {name: zeros_long.clone() for name in violation_names}
        new_abs_action_ge_0999 = torch.zeros_like(option_active)

        pre_shapes = {
            name: (
                (artifact_contract.OBSERVATION_DIM,)
                if name in {"pre_observation", "pre_transition_observation"}
                else (artifact_contract.HAND_ACTION_DIM,)
                if name == "pre_applied_delta"
                else (artifact_contract.ACTION_DIM,)
                if name in {
                    "pre_baseline_action", "pre_common_action",
                    "pre_requested_action", "pre_task_action",
                }
                else ()
            )
            for name in artifact_contract.PRE_STEP_FIELDS
        }
        pre_bool = {
            "pre_option_active", "pre_public_latch_before", "pre_residual_active",
            "pre_first_latch_witness", "pre_transition_public_latch",
            "pre_transition_grasped", "pre_transition_done",
        }
        pre_long = {"pre_env_slot", "pre_episode_step", "pre_close_age"}
        pre_dtypes = {
            name: torch.bool if name in pre_bool else torch.long if name in pre_long else torch.float32
            for name in artifact_contract.PRE_STEP_FIELDS
        }
        pre_rows: dict[str, list[torch.Tensor]] = {
            name: [] for name in artifact_contract.PRE_STEP_FIELDS
        }

        step_shapes = {
            name: (
                (artifact_contract.OBSERVATION_DIM,)
                if name in {"row_observation", "row_transition_observation"}
                else (2,)
                if name == "row_public_force_counters"
                else (artifact_contract.ACTION_DIM,)
                if name in {"row_common_action", "row_requested_action", "row_task_action"}
                else (artifact_contract.ARM_ACTION_DIM,)
                if name in {"row_arm_target_error", "row_arm_joint_velocity"}
                else ()
            )
            for name in artifact_contract.STEP_FIELDS
        }
        step_bool = {
            "row_option_active", "row_public_latch_before", "row_stable_current",
            "row_treatment", "row_activated", "row_first_eligible",
            "row_arm_enabled", "row_verification_complete", "row_relock",
            "row_reenable", "row_pre_enable", "row_transition_public_latch",
            "row_transition_grasped", "row_transition_done",
            "row_pre_release_launch",
        }
        step_long = {
            "row_env_slot", "row_episode_step", "row_eligible_age",
            "row_stable_count_before", "row_stable_count_after",
        }
        step_dtypes = {
            name: torch.bool if name in step_bool else torch.long if name in step_long else torch.float32
            for name in artifact_contract.STEP_FIELDS
        }
        step_rows: dict[str, list[torch.Tensor]] = {
            name: [] for name in artifact_contract.STEP_FIELDS
        }
        env_slot = torch.arange(spec.num_envs, dtype=torch.long, device=env.device)
        cfg = env.unwrapped.cfg
        vector_steps = 0

        while not tracker.complete and vector_steps < artifact_contract.MAX_EPISODE_ACTIONS:
            active_before = tracker.active.clone()
            handoff = update_online_handoff(
                observation,
                ready_count_before=ready_count,
                option_active_before=option_active,
                min_score=artifact_contract.HANDOFF_MIN_SCORE,
                hold_steps=artifact_contract.HANDOFF_HOLD_STEPS,
            )
            trigger = active_before & handoff["trigger"]
            if bool((trigger & triggered).any()):
                raise RuntimeError("a Candidate40 slot triggered twice")
            triggered |= trigger
            trigger_step = torch.where(trigger, episode_step, trigger_step)
            trigger_score = torch.where(
                trigger, handoff["score"].to(torch.float32), trigger_score
            )
            ready_count = torch.where(
                active_before,
                handoff["ready_count_after"],
                torch.zeros_like(ready_count),
            )
            option_active = active_before & handoff["option_active_after"]

            latch_float = observation[:, artifact_contract.PUBLIC_LATCH_INDEX]
            if not bool(((latch_float == 0.0) | (latch_float == 1.0)).all()):
                raise RuntimeError("public latch is not binary")
            public_latch = latch_float == 1.0
            if bool((active_before & (~ever_latched) & public_latch).any()):
                raise RuntimeError("first latch edge was not recorded on its transition")
            window = option_residual_screen.option_residual_window(
                episode_active=active_before,
                option_active=option_active,
                public_latch=public_latch,
                ever_latched=ever_latched,
                episode_step=episode_step,
                trigger_step=trigger_step,
                window_steps=artifact_contract.WINDOW_STEPS,
            )

            actor_mean = agent.deterministic_actor_mean(observation)
            raw_close_action = agent.apply_action_authority(
                torch.tanh(actor_mean), observation
            )
            baseline_action = agent.sample_actions(
                vector_steps + 1, {"next_observation": observation}, training=False
            )
            if baseline_action.shape != (spec.num_envs, artifact_contract.ACTION_DIM):
                raise RuntimeError("V6 baseline violated action21")
            close_rows = active_before & option_active & (~public_latch)
            if bool(close_rows.any()) and not torch.equal(
                raw_close_action[close_rows], baseline_action[close_rows]
            ):
                raise RuntimeError("V6 CLOSE differs from raw deterministic authority")
            if bool(close_rows.any()) and not torch.equal(
                baseline_action[close_rows, : artifact_contract.ARM_ACTION_DIM],
                torch.zeros_like(
                    baseline_action[close_rows, : artifact_contract.ARM_ACTION_DIM]
                ),
            ):
                raise RuntimeError("V6 CLOSE emitted nonzero arm authority")
            lift_rows = active_before & option_active & public_latch
            if bool(lift_rows.any()):
                expected_lift = agent.frozen_lift_actions(observation)
                if not torch.equal(baseline_action[lift_rows], expected_lift[lift_rows]):
                    raise RuntimeError("common post-latch route differs from frozen LIFT")

            residual = option_residual_screen.apply_option_residual(
                baseline_action=baseline_action,
                option_active=window.active,
                public_latch=public_latch,
                treatment=all_fixed,
                raw_z=raw_z,
                pre_tanh_scale=scale,
                raw_z_abs_cap=artifact_contract.RAW_Z_ABS_CAP,
                pre_tanh_abs_cap=component_cap,
                pre_tanh_l2_cap=artifact_contract.PRE_TANH_L2_CAP,
            )
            if not torch.equal(residual.eligible, window.active):
                raise RuntimeError("fixed residual assignment incorrectly used Candidate40 arm")
            expected_step_delta = torch.where(
                window.active.unsqueeze(-1),
                expected_delta.expand(spec.num_envs, -1),
                torch.zeros_like(residual.pre_tanh_residual),
            )
            residual_bad = ~torch.isclose(
                residual.pre_tanh_residual,
                expected_step_delta,
                rtol=0.0,
                atol=1.0e-6,
            ).all(dim=-1)
            _increment(violations["fixed_residual_budget_violations"], active_before & residual_bad)
            search_action = search_actor(observation).clamp(-1.0, 1.0)
            if search_action.shape != baseline_action.shape or not bool(
                torch.isfinite(search_action).all()
            ):
                raise RuntimeError("SEARCH emitted an invalid action")
            common_baseline = torch.where(
                option_active.unsqueeze(-1), baseline_action, search_action
            )
            common_action = torch.where(
                option_active.unsqueeze(-1), residual.action, search_action
            )
            common_baseline = torch.where(
                active_before.unsqueeze(-1), common_baseline, torch.zeros_like(common_baseline)
            )
            common_action = torch.where(
                active_before.unsqueeze(-1), common_action, torch.zeros_like(common_action)
            )

            state_before = verifier.reset_verified_arm_state(
                verifier_state, authoritative_reset_mask
            )
            if bool(authoritative_reset_mask.any()):
                reset_ids = authoritative_reset_mask
                if bool(state_before.activated[reset_ids].any()) or bool(
                    (state_before.stable_count[reset_ids] != 0).any()
                ) or bool(state_before.arm_enabled_previous[reset_ids].any()) or bool(
                    state_before.ever_enabled[reset_ids].any()
                ):
                    raise RuntimeError("authoritative reset receipt retained verifier state")
            verified = verifier.apply_verified_arm_handoff(
                common_action,
                observation,
                public_counters,
                option_active,
                active_before,
                treatment,
                verifier_state,
                reset_mask=authoritative_reset_mask,
            )
            if bool(authoritative_reset_mask.any()):
                reset_ids = authoritative_reset_mask
                if not torch.equal(
                    verified.activated_before[reset_ids],
                    state_before.activated[reset_ids],
                ) or not torch.equal(
                    verified.stable_count_before[reset_ids],
                    state_before.stable_count[reset_ids],
                ):
                    raise RuntimeError("authoritative reset did not clear verifier state")
            requested_action = verified.action
            first_eligible_step = torch.where(
                verified.first_eligible, episode_step, first_eligible_step
            )
            first_eligible_clearance = torch.where(
                verified.first_eligible, current_clearance, first_eligible_clearance
            )
            pre_release_max_clearance = torch.where(
                verified.first_eligible, current_clearance, pre_release_max_clearance
            )
            verification_complete_step = torch.where(
                verified.verification_complete & (verification_complete_step < 0),
                episode_step,
                verification_complete_step,
            )
            first_arm_enabled_step = torch.where(
                verified.newly_enabled & (first_arm_enabled_step < 0),
                episode_step,
                first_arm_enabled_step,
            )
            reenable = verified.newly_enabled & state_before.ever_enabled
            first_relock_step = torch.where(
                verified.relock & (first_relock_step < 0),
                episode_step,
                first_relock_step,
            )
            first_reenable_step = torch.where(
                reenable & (first_reenable_step < 0),
                episode_step,
                first_reenable_step,
            )
            _increment(arm_enable_count, verified.newly_enabled & active_before)
            _increment(relock_count, verified.relock & active_before)
            _increment(reenable_count, reenable & active_before)
            stable_count_max = torch.where(
                active_before,
                torch.maximum(stable_count_max, verified.stable_count_after),
                stable_count_max,
            )
            pre_enable = (
                verified.activated_this_action
                & (~state_before.ever_enabled)
                & (~verified.arm_enabled_this_action)
            )

            expected_treatment_arm = torch.where(
                verified.arm_enabled_this_action.unsqueeze(-1),
                common_action[:, : artifact_contract.ARM_ACTION_DIM],
                torch.zeros_like(common_action[:, : artifact_contract.ARM_ACTION_DIM]),
            )
            preeligible_bad = (
                active_before
                & (~verified.activated_this_action)
                & (requested_action != common_action).any(dim=-1)
            )
            hand_bad = active_before & (
                requested_action[:, artifact_contract.ARM_ACTION_DIM :]
                != common_action[:, artifact_contract.ARM_ACTION_DIM :]
            ).any(dim=-1)
            treatment_gate_bad = (
                active_before
                & treatment
                & verified.activated_this_action
                & (
                    requested_action[:, : artifact_contract.ARM_ACTION_DIM]
                    != expected_treatment_arm
                ).any(dim=-1)
            )
            control_bad = active_before & (~treatment) & (
                requested_action != common_action
            ).any(dim=-1)
            bounds_bad = active_before & (
                (~torch.isfinite(requested_action)).any(dim=-1)
                | (requested_action.abs() > 1.0).any(dim=-1)
            )
            treatment_pre_nonzero = (
                active_before
                & treatment
                & pre_enable
                & (requested_action[:, : artifact_contract.ARM_ACTION_DIM] != 0.0).any(dim=-1)
            )
            for name, mask in (
                ("pre_eligibility_action_violations", preeligible_bad),
                ("hand_invariance_violations", hand_bad),
                ("treatment_arm_gate_violations", treatment_gate_bad),
                ("control_route_violations", control_bad),
                ("action_bound_violations", bounds_bad),
                ("treatment_pre_enable_nonzero_arm_rows", treatment_pre_nonzero),
            ):
                _increment(violations[name], mask)
            pre_reference = torch.where(
                verified.activated_this_action.unsqueeze(-1),
                common_action,
                common_baseline,
            )
            new_abs_action_ge_0999 |= active_before & (
                (requested_action.abs() >= 0.999)
                & (pre_reference.abs() < 0.999)
            ).any(dim=-1)

            target_error, arm_velocity, target_error_max, arm_speed_max = _arm_state(
                env.unwrapped
            )
            eligible_age = torch.where(
                verified.activated_this_action,
                episode_step - first_eligible_step,
                torch.full_like(episode_step, -1),
            )
            pending_pre = {
                "pre_env_slot": env_slot,
                "pre_episode_step": episode_step,
                "pre_close_age": torch.where(
                    option_active, window.close_age, torch.full_like(window.close_age, -1)
                ),
                "pre_option_active": option_active,
                "pre_public_latch_before": public_latch,
                "pre_residual_active": residual.eligible,
                "pre_observation": observation,
                "pre_applied_delta": residual.pre_tanh_residual,
                "pre_baseline_action": common_baseline,
                "pre_common_action": common_action,
                "pre_requested_action": requested_action,
            }
            pending_step = {
                "row_env_slot": env_slot,
                "row_episode_step": episode_step,
                "row_eligible_age": eligible_age,
                "row_observation": observation,
                "row_public_force_counters": verified.force_counter_features,
                "row_option_active": option_active,
                "row_public_latch_before": public_latch,
                "row_stable_current": verified.stable_current,
                "row_public_grasp_quality": verified.public_grasp_quality,
                "row_public_hold_quality": verified.public_hold_quality,
                "row_public_max_force_strength": verified.public_max_force_strength,
                "row_stable_count_before": verified.stable_count_before,
                "row_stable_count_after": verified.stable_count_after,
                "row_common_action": common_action,
                "row_requested_action": requested_action,
                "row_arm_target_error": target_error,
                "row_arm_joint_velocity": arm_velocity,
                "row_arm_target_error_abs_max": target_error_max,
                "row_arm_joint_speed_abs_max": arm_speed_max,
                "row_pre_action_true_clearance_m": current_clearance,
                "row_treatment": treatment,
                "row_activated": verified.activated_this_action,
                "row_first_eligible": verified.first_eligible,
                "row_arm_enabled": verified.arm_enabled_this_action,
                "row_verification_complete": verified.verification_complete,
                "row_relock": verified.relock,
                "row_reenable": reenable,
                "row_pre_enable": pre_enable,
            }

            next_observation, reward, terminated, truncated, info = env.step(
                requested_action
            )
            vector_steps += 1
            executed = env.last_executed_policy_action
            if executed is None or not torch.equal(executed, requested_action):
                raise RuntimeError("adapter executed a different Candidate40 request")
            done = terminated | truncated
            telemetry = _read_transition_telemetry(
                info, num_envs=spec.num_envs, device=env.device
            )
            transition_observation = info.get("transition_next_observation")
            if (
                not isinstance(transition_observation, torch.Tensor)
                or transition_observation.shape != observation.shape
                or transition_observation.dtype != torch.float32
                or transition_observation.device != env.device
                or not bool(torch.isfinite(transition_observation).all())
            ):
                raise RuntimeError("adapter omitted reset-before transition obs115")
            task_action = artifact_contract.reconstruct_task_action(
                transition_observation
            )
            independent_task_action = torch.cat(
                (
                    transition_observation[:, artifact_contract.TRANSITION_ARM_TOKEN_SLICE],
                    transition_observation[:, artifact_contract.TRANSITION_DISTAL_SLICE],
                ),
                dim=-1,
            )
            reconstruction_bad = active_before & (
                task_action != independent_task_action
            ).any(dim=-1)
            _increment(
                violations["task_action_reconstruction_violations"], reconstruction_bad
            )
            task_bounds_bad = active_before & (
                (~torch.isfinite(task_action)).any(dim=-1)
                | (task_action.abs() > 1.0).any(dim=-1)
            )
            _increment(violations["action_bound_violations"], task_bounds_bad)
            transition_latch_float = transition_observation[
                :, artifact_contract.PUBLIC_LATCH_INDEX
            ]
            if not bool(
                ((transition_latch_float == 0.0) | (transition_latch_float == 1.0)).all()
            ):
                raise RuntimeError("transition public latch is not binary")
            transition_latch = transition_latch_float == 1.0
            newly_latched = active_before & (~ever_latched) & transition_latch
            first_latch_step = torch.where(
                newly_latched, episode_step, first_latch_step
            )

            row_prelaunch = (
                active_before
                & treatment
                & pre_enable
                & (
                    (torch.maximum(telemetry["true_clearance"], first_eligible_clearance)
                     > artifact_contract.PRE_RELEASE_CLEARANCE_LIMIT_M)
                    | (
                        telemetry["true_clearance"] - first_eligible_clearance
                        >= artifact_contract.PRE_RELEASE_CLEARANCE_LIMIT_M
                    )
                )
            )
            pre_release_max_clearance = torch.where(
                active_before & pre_enable,
                torch.maximum(pre_release_max_clearance, telemetry["true_clearance"]),
                pre_release_max_clearance,
            )
            pre_release_launch |= row_prelaunch
            preeligible = active_before & (~verified.activated_this_action)
            option_preeligible_u5 |= (
                preeligible
                & option_active
                & (~telemetry["is_grasped"])
                & (telemetry["true_clearance"] >= 0.05)
            )
            intervention_steps += residual.eligible.long()
            trajectory_max_force = torch.where(
                active_before,
                torch.maximum(trajectory_max_force, telemetry["max_force"]),
                trajectory_max_force,
            )

            pending_pre.update(
                {
                    "pre_first_latch_witness": newly_latched,
                    "pre_transition_observation": transition_observation,
                    "pre_task_action": task_action,
                    "pre_transition_public_latch": transition_latch,
                    "pre_transition_grasped": telemetry["is_grasped"],
                    "pre_transition_true_clearance_m": telemetry["true_clearance"],
                    "pre_grasp_quality": telemetry["grasp_quality"],
                    "pre_hold_quality": telemetry["hold_quality"],
                    "pre_max_force_n": telemetry["max_force"],
                    "pre_transition_done": done,
                }
            )
            pre_record = preeligible & (option_active | newly_latched)
            _append_rows(pre_rows, mask=pre_record, values=pending_pre)
            pending_step.update(
                {
                    "row_transition_observation": transition_observation,
                    "row_task_action": task_action,
                    "row_transition_public_latch": transition_latch,
                    "row_transition_grasped": telemetry["is_grasped"],
                    "row_transition_grasp_quality": telemetry["grasp_quality"],
                    "row_transition_hold_quality": telemetry["hold_quality"],
                    "row_transition_max_force_n": telemetry["max_force"],
                    "row_transition_object_lin_speed": telemetry["object_lin_speed"],
                    "row_transition_object_ang_speed": telemetry["object_ang_speed"],
                    "row_transition_true_clearance_m": telemetry["true_clearance"],
                    "row_transition_done": done,
                    "row_pre_release_launch": row_prelaunch,
                }
            )
            trace_record = active_before & verified.activated_this_action
            _append_rows(step_rows, mask=trace_record, values=pending_step)

            events = validate_terminal_events(
                info,
                terminated,
                truncated,
                task_mode=FULL_TASK_MODE,
                close_option_confirm_steps=int(cfg.close_option_confirm_steps),
                power_required_other_contacts=int(cfg.power_grasp_required_other_contacts),
                power_grasp_quality_threshold=float(cfg.power_grasp_quality_high),
                close_option_min_hold_quality=float(cfg.close_option_min_hold_quality),
                close_option_safe_force_limit=float(cfg.grasp_bonus_max_force),
            )
            transition_truth = physical_truth_from_terminal_info(
                info,
                num_envs=spec.num_envs,
                device=env.device,
                task_mode=FULL_TASK_MODE,
                close_option_confirm_steps=int(cfg.close_option_confirm_steps),
                close_option_min_hold_quality=float(cfg.close_option_min_hold_quality),
                grasp_quality_threshold=float(cfg.grasp_quality_high),
                safe_force_limit=float(cfg.grasp_bonus_max_force),
            )
            post_reset_truth = _read_physical_truth(
                env.unwrapped, task_mode=FULL_TASK_MODE
            )
            tracker.step(
                reward=reward.to(torch.float32),
                terminated=terminated,
                truncated=truncated,
                events=events,
                transition_truth=transition_truth,
                post_reset_truth=post_reset_truth,
            )
            ever_latched |= newly_latched
            verifier_state = verified.next_state
            authoritative_reset_mask = done.clone()
            next_step = episode_step + active_before.long()
            episode_step = torch.where(done, torch.zeros_like(next_step), next_step)
            ready_count.masked_fill_(done, 0)
            option_active.masked_fill_(done, False)
            ever_latched.masked_fill_(done, False)
            public_counters = public_gate_feature_tensor(
                info[PUBLIC_GATE_STATE_EXTRAS_KEY],
                num_envs=spec.num_envs,
                device=env.device,
            )
            current_clearance = torch.where(
                done, post_reset_truth.clearance, transition_truth.clearance
            )
            observation = next_observation

        if not tracker.complete or len(tracker.records) != spec.num_envs:
            raise RuntimeError(
                f"Candidate40 completed {len(tracker.records)}/{spec.num_envs} first episodes"
            )
        cleared_final = verifier.reset_verified_arm_state(
            verifier_state, authoritative_reset_mask
        )
        if bool(cleared_final.activated[authoritative_reset_mask].any()) or bool(
            (cleared_final.stable_count[authoritative_reset_mask] != 0).any()
        ) or bool(cleared_final.arm_enabled_previous[authoritative_reset_mask].any()) or bool(
            cleared_final.ever_enabled[authoritative_reset_mask].any()
        ):
            raise RuntimeError("terminal authoritative reset did not clear verifier state")
        validate_public_gate_feature_values(public_counters)
        records_by_slot = {
            int(record["env_slot"]): record for record in tracker.records
        }
        if set(records_by_slot) != set(range(spec.num_envs)):
            raise RuntimeError("Candidate40 tracker omitted an environment slot")
        ordered = [records_by_slot[index] for index in range(spec.num_envs)]
        if any(int(record["slot_episode_index"]) != 0 for record in ordered):
            raise RuntimeError("Candidate40 artifact contains a post-reset episode")

        pre_tensors = _cat_rows(
            pre_rows, shapes=pre_shapes, dtypes=pre_dtypes
        )
        step_tensors = _cat_rows(
            step_rows, shapes=step_shapes, dtypes=step_dtypes
        )
        trace_count = torch.bincount(
            step_tensors["row_env_slot"], minlength=spec.num_envs
        )
        episode_lengths = torch.tensor(
            [int(record["length"]) for record in ordered], dtype=torch.long
        )
        episode_tensors = {
            "env_slot": torch.arange(spec.num_envs, dtype=torch.long),
            "treatment": spec.treatment.to(torch.bool),
            "assignment_rank": spec.assignment_rank.to(torch.long),
            "fixed_z": spec.fixed_z.expand(spec.num_envs, -1).clone(),
            "triggered": triggered.cpu(),
            "trigger_step": trigger_step.cpu(),
            "trigger_score": trigger_score.cpu(),
            "first_latch_step": first_latch_step.cpu(),
            "first_eligible_step": first_eligible_step.cpu(),
            "verification_complete_step": verification_complete_step.cpu(),
            "first_arm_enabled_step": first_arm_enabled_step.cpu(),
            "first_relock_step": first_relock_step.cpu(),
            "first_reenable_step": first_reenable_step.cpu(),
            "terminal_step": episode_lengths - 1,
            "intervention_steps": intervention_steps.cpu(),
            "trace_rows": trace_count,
            "stable_count_max": stable_count_max.cpu(),
            "arm_enable_count": arm_enable_count.cpu(),
            "relock_count": relock_count.cpu(),
            "reenable_count": reenable_count.cpu(),
            "episode_length": episode_lengths,
            "trajectory_max_force_n": trajectory_max_force.cpu(),
            "max_true_clearance_m": torch.tensor(
                [float(record["max_true_clearance_m"]) for record in ordered],
                dtype=torch.float32,
            ),
            "first_eligible_clearance_m": first_eligible_clearance.cpu(),
            "pre_release_max_clearance_m": pre_release_max_clearance.cpu(),
            "eligible": (first_eligible_step >= 0).cpu(),
            "pre_release_launch": pre_release_launch.cpu(),
            "latched_within_window": (
                triggered
                & (first_latch_step >= trigger_step)
                & ((first_latch_step - trigger_step) < artifact_contract.WINDOW_STEPS)
            ).cpu(),
            "success": torch.tensor(
                [bool(record["success"]) for record in ordered], dtype=torch.bool
            ),
            "failure": torch.tensor(
                [bool(record["failure"]) for record in ordered], dtype=torch.bool
            ),
            "time_out": torch.tensor(
                [bool(record["time_out"]) for record in ordered], dtype=torch.bool
            ),
            "dropped": torch.tensor(
                [bool(record["dropped"]) for record in ordered], dtype=torch.bool
            ),
            "unsafe_force": torch.tensor(
                [bool(record["unsafe_force"]) for record in ordered], dtype=torch.bool
            ),
            "unlatched_clearance_ge_5cm": torch.tensor(
                [bool(record["ever_unlatched_clearance_ge_5cm"]) for record in ordered],
                dtype=torch.bool,
            ),
            "ever_grasped": torch.tensor(
                [bool(record["ever_grasped"]) for record in ordered], dtype=torch.bool
            ),
            "ever_clearance_ge_20cm": torch.tensor(
                [bool(record["ever_clearance_ge_20cm"]) for record in ordered],
                dtype=torch.bool,
            ),
            "option_active_pre_eligible_unlatched_clearance_ge_5cm": option_preeligible_u5.cpu(),
            **{name: value.cpu() for name, value in violations.items()},
            "new_abs_action_ge_0999": new_abs_action_ge_0999.cpu(),
        }

        source_after = source_fingerprints(spec.repository_root)
        runtime_assets_after = runtime_asset_fingerprints(spec.repository_root)
        git_after = git_provenance(spec.repository_root, git_paths)
        if (
            source_after != source_before
            or runtime_assets_after != runtime_assets_before
            or git_after != git_before
            or _current_hashes(spec) != spec.checkpoint_sha256
        ):
            raise RuntimeError("Candidate40 source/runtime/checkpoint changed during collection")
        v6_hashes = spec.checkpoint_sha256["v6"]
        if not isinstance(v6_hashes, Mapping):
            raise TypeError("V6 checkpoint receipt is not a mapping")
        metadata = {
            **artifact_contract.REQUIRED_METADATA,
            "seed": spec.seed,
            "replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "assignment_mask_sha256": spec.assignment_mask_sha256,
            "validation_plan_sha256": source_before[artifact_contract.VALIDATION_PLAN],
            "smoke_artifact_sha256": spec.smoke_artifact_sha256,
            "smoke_report_sha256": spec.smoke_report_sha256,
            "source_manifest_sha256": artifact_contract.manifest_sha256(source_before),
            "runtime_asset_manifest_sha256": artifact_contract.manifest_sha256(
                runtime_assets_before
            ),
            "source_sha256": source_before,
            "runtime_asset_sha256": runtime_assets_before,
            "flashsac_upstream_commit": load["upstream_commit"],
            "git": git_before,
            "runtime": runtime_provenance(seed=spec.seed, device=env.device),
        }
        artifact = artifact_contract.build_artifact(
            metadata=metadata,
            episodes=episode_tensors,
            pre_steps=pre_tensors,
            steps=step_tensors,
        )
        report = {
            "kind": artifact_contract.REPORT_KIND,
            "status": "complete",
            "collector": artifact_contract.COLLECTOR,
            "seed": spec.seed,
            "replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "vector_steps": int(episode_lengths.max()),
            "v6_actor_sha256": metadata["v6_actor_sha256"],
            "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
            "fixed_direction_sha256": metadata["fixed_direction_sha256"],
            "summary": artifact_contract.summarize_artifact(artifact),
        }
        artifact_contract.validate_report(report, artifact)
        return artifact, report
    finally:
        env.close()


def parse_args() -> tuple[argparse.Namespace, CollectionSpec, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--v6_checkpoint", type=Path,
        default=Path(artifact_contract.V6_CHECKPOINT_PATH),
    )
    parser.add_argument(
        "--search_checkpoint", type=Path,
        default=Path(artifact_contract.SEARCH_CHECKPOINT_PATH),
    )
    parser.add_argument(
        "--fixed_direction", type=Path,
        default=Path(artifact_contract.FIXED_DIRECTION_PATH),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicate", choices=("a", "b"), required=True)
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS)
    parser.add_argument("--window_steps", type=int, default=artifact_contract.WINDOW_STEPS)
    parser.add_argument("--token_scale", type=float, default=artifact_contract.TOKEN_SCALE)
    parser.add_argument("--distal_scale", type=float, default=artifact_contract.DISTAL_SCALE)
    parser.add_argument("--raw_z_abs_cap", type=float, default=artifact_contract.RAW_Z_ABS_CAP)
    parser.add_argument(
        "--token_component_cap", type=float,
        default=artifact_contract.TOKEN_COMPONENT_CAP,
    )
    parser.add_argument(
        "--distal_component_cap", type=float,
        default=artifact_contract.DISTAL_COMPONENT_CAP,
    )
    parser.add_argument(
        "--pre_tanh_l2_cap", type=float,
        default=artifact_contract.PRE_TANH_L2_CAP,
    )
    parser.add_argument("--verify_steps", type=int, default=verifier.VERIFY_STEPS)
    parser.add_argument("--output_stem", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        spec = build_spec(args)
    except (TypeError, ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    launcher = AppLauncher(args)
    return args, spec, launcher.app


def publish_failure_attempt(spec: CollectionSpec, error: BaseException) -> Path:
    payload = {
        "kind": artifact_contract.REPORT_KIND,
        "status": "failed",
        "seed": spec.seed,
        "replicate": spec.replicate,
        "num_envs": spec.num_envs,
        "canonical_artifact_output": str(spec.artifact_output),
        "canonical_report_output": str(spec.report_output),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    for attempt in range(1, 10_000):
        output = Path(f"{spec.output_stem}.failed_attempt_{attempt:03d}.json")
        try:
            artifact_contract.publish_json_no_clobber(payload, output)
        except FileExistsError:
            continue
        return output
    raise RuntimeError("Candidate40 failure-attempt namespace is exhausted")


def main() -> None:
    args, spec, simulation_app = parse_args()
    try:
        try:
            artifact, report = run_collection(
                spec, device_string=str(args.device or "cuda:0")
            )
            digest = artifact_contract.publish_artifact_and_report_no_clobber(
                artifact,
                report,
                artifact_output=spec.artifact_output,
                report_output=spec.report_output,
            )
            print(
                "[candidate40-verified-arm-ab] "
                f"seed={spec.seed} replicate={spec.replicate} "
                f"envs={spec.num_envs} trace_rows={report['summary']['trace_rows']} "
                f"sha256={digest}",
                flush=True,
            )
        except BaseException as error:
            failure = publish_failure_attempt(spec, error)
            print(f"[candidate40-verified-arm-ab] failure recorded at {failure}", flush=True)
            raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
