#!/usr/bin/env python3
"""Collect Candidate43's public-arm-ramp development A/B artifacts.

Both assignments execute one exact common Candidate39 route.  Candidate43
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
from typing import Any, Callable, Mapping

import torch

import candidate42_attempt_transaction as transaction
import candidate43_short_arm_ramp_episode as artifact_contract
import candidate43_short_arm_ramp as verifier
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
TOKEN_ACTION_DIM = 9
PUBLIC_ARM_RAMP_DENOMINATOR = 8.0
EXPECTED_V6_SHA256 = {
    "actor.pt": artifact_contract.V6_ACTOR_SHA256,
    "task_contract.json": artifact_contract.V6_TASK_CONTRACT_SHA256,
    "torch_bridge_state.pt": artifact_contract.V6_BRIDGE_STATE_SHA256,
    "frozen_lift_actor.pt": artifact_contract.FROZEN_LIFT_ACTOR_SHA256,
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_XHAND_PACKAGE_ROOT = _REPOSITORY_ROOT / "source/xhand_inhand/xhand_inhand"


def _dynamic_import_package_sources() -> tuple[str, ...]:
    """Conservatively close IsaacLab's recursive task-package importer."""

    discovered: set[str] = set()
    for path in _XHAND_PACKAGE_ROOT.rglob("*.py"):
        relative_parts = path.relative_to(_XHAND_PACKAGE_ROOT).parts
        if ".claude" in relative_parts or "__pycache__" in relative_parts:
            continue
        discovered.add(path.relative_to(_REPOSITORY_ROOT).as_posix())
    tasks = _XHAND_PACKAGE_ROOT / "tasks"
    for pattern in ("*.yaml", "*.yml"):
        for path in tasks.rglob(pattern):
            relative_parts = path.relative_to(_XHAND_PACKAGE_ROOT).parts
            if ".claude" in relative_parts or "__pycache__" in relative_parts:
                continue
            discovered.add(path.relative_to(_REPOSITORY_ROOT).as_posix())
    return tuple(sorted(discovered))


DYNAMIC_IMPORT_PACKAGE_SOURCES = _dynamic_import_package_sources()
EXTRA_SOURCE_FILES = tuple(
    sorted(
        set(artifact_contract.IMPLEMENTATION_SOURCE_FILES)
        | set(DYNAMIC_IMPORT_PACKAGE_SOURCES)
        | {
            # Imported directly for row materialization.  Candidate40 omitted
            # this edge from its source closure; Candidate43 preregisters it.
            "scripts/flashsac/collect_candidate39_episode_residual_ab.py",
            "scripts/flashsac/candidate39_episode_residual.py",
            # Candidate43's supervisor and artifact contract reuse these
            # sealed Candidate40 modules directly; neither is in the generic
            # online-collector source manifest.
            "scripts/flashsac/candidate40_verified_arm_handoff.py",
            "scripts/flashsac/candidate40_verified_arm_episode.py",
            # Candidate43 aliases this exact sealed scientific supervisor by
            # function identity; it is therefore a direct runtime edge.
            "scripts/flashsac/candidate41_public_arm_ramp.py",
            # Candidate43's D8 supervisor reuses Candidate42's immutable
            # public predicate/record types and every process imports the
            # exact Candidate42 transaction implementation.  Keep both in
            # the collection-time runtime closure rather than trusting the
            # implementation-source list to imply transitive imports.
            "scripts/flashsac/candidate42_public_arm_ramp.py",
            "scripts/flashsac/candidate42_attempt_transaction.py",
            "scripts/flashsac/option_residual_screen.py",
            artifact_contract.VALIDATION_PLAN,
            "source/xhand_inhand/xhand_inhand/tasks/direct/pick_tool_token/public_gate_state.py",
        }
    )
)
CANDIDATE43_INVOCATION_SEQUENCE = (
    (352, "b", 8),
    (353, "a", 64),
    (353, "b", 64),
    (354, "b", 64),
    (354, "a", 64),
)
CANDIDATE43_RUN_ORDER = tuple(
    (seed, replicate) for seed, replicate, num_envs in CANDIDATE43_INVOCATION_SEQUENCE
    if num_envs == 64
)
USER_VISIBLE_ARGUMENT_FIELDS = (
    "device",
    "enable_cameras",
    "experience",
    "fixed_direction",
    "headless",
    "kit_args",
    "livestream",
    "num_envs",
    "output_stem",
    "pre_tanh_l2_cap",
    "raw_z_abs_cap",
    "replicate",
    "search_checkpoint",
    "seed",
    "token_component_cap",
    "token_scale",
    "distal_component_cap",
    "distal_scale",
    "v6_checkpoint",
    "verify_steps",
    "window_steps",
)
PREREGISTRATION_TAG = "pick-tool-candidate43-short-arm-ramp-plan-v1-20260723"
COLLECTION_SEAL_TAG = "pick-tool-candidate43-short-arm-ramp-validation-v1-20260723"


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
    checkpoint_sha256: Mapping[str, Mapping[str, str] | str]
    source_sha256: Mapping[str, str]
    collection_commit: str
    implementation_commit: str
    preregistration_tag_commit: str
    preregistration_plan_sha256: str
    argument_receipt: Mapping[str, Any]
    treatment: torch.Tensor
    assignment_rank: torch.Tensor
    assignment_mask_sha256: str
    fixed_z: torch.Tensor
    smoke_artifact_sha256: str | None
    smoke_report_sha256: str | None


@dataclass
class AttemptProgress:
    """Whether this invocation crossed the authoritative first-step boundary."""

    first_env_step_invoked: bool = False


@dataclass(frozen=True)
class ReconstructedPublicArmRamp:
    """Independent float32 reconstruction used by the online action audit."""

    ramp_scale: torch.Tensor
    authority_scale: torch.Tensor
    stable_count_after: torch.Tensor
    requested_arm: torch.Tensor


def authoritative_env_step(
    env: Any,
    action: torch.Tensor,
    *,
    progress: AttemptProgress,
    durable_boundary: Callable[[], None],
) -> tuple[Any, Any, Any, Any, Any]:
    """Durably arm the unique first-step boundary before entering env.step."""

    if not progress.first_env_step_invoked:
        durable_boundary()
        progress.first_env_step_invoked = True
    return env.step(action)


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
        raise ValueError(f"Candidate43 requires exact {name}={expected!r}")


def reconstruct_public_arm_ramp(
    *,
    common_arm: torch.Tensor,
    stable_current: torch.Tensor,
    stable_count_before: torch.Tensor,
    treatment: torch.Tensor,
    activated_this_action: torch.Tensor,
    option_active: torch.Tensor,
    episode_active: torch.Tensor,
) -> ReconstructedPublicArmRamp:
    """Rebuild Candidate43's public clock and effective arm multiplier.

    This deliberately does not call the supervisor.  Collection compares its
    result with both the supervisor telemetry and the requested action, so a
    shared bad clock or scale cannot authenticate itself.
    """

    if (
        not isinstance(common_arm, torch.Tensor)
        or common_arm.ndim != 2
        or common_arm.shape[1] != artifact_contract.ARM_ACTION_DIM
        or common_arm.dtype != torch.float32
    ):
        raise ValueError("common_arm must be a [batch, 7] float32 tensor")
    rows = int(common_arm.shape[0])
    device = common_arm.device
    for name, value in (
        ("stable_current", stable_current),
        ("treatment", treatment),
        ("activated_this_action", activated_this_action),
        ("option_active", option_active),
        ("episode_active", episode_active),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (rows,)
            or value.dtype != torch.bool
            or value.device != device
        ):
            raise ValueError(f"{name} must be a [{rows}] bool tensor on {device}")
    if (
        not isinstance(stable_count_before, torch.Tensor)
        or stable_count_before.shape != (rows,)
        or stable_count_before.dtype != torch.long
        or stable_count_before.device != device
    ):
        raise ValueError(
            f"stable_count_before must be a [{rows}] long tensor on {device}"
        )
    if not bool(torch.isfinite(common_arm).all()) or bool(
        (common_arm.abs() > 1.0).any()
    ):
        raise ValueError("common_arm must be finite and bounded by one")
    if bool(
        (
            (stable_count_before < 0)
            | (stable_count_before > verifier.VERIFY_STEPS)
        ).any()
    ):
        raise ValueError("stable_count_before escaped the sealed public clock")

    denominator = torch.tensor(
        PUBLIC_ARM_RAMP_DENOMINATOR, dtype=torch.float32, device=device
    )
    fraction = torch.clamp(
        stable_count_before, max=verifier.VERIFY_STEPS
    ).to(dtype=torch.float32) / denominator
    ramp_scale = torch.where(
        stable_current, fraction, torch.zeros_like(fraction)
    )
    overlay = (
        episode_active & treatment & activated_this_action & option_active
    )
    authority_scale = torch.where(
        overlay, ramp_scale, torch.ones_like(ramp_scale)
    )
    candidate_count = torch.where(
        stable_current,
        torch.clamp(stable_count_before + 1, max=verifier.VERIFY_STEPS),
        torch.zeros_like(stable_count_before),
    )
    stable_count_after = torch.where(
        episode_active, candidate_count, stable_count_before
    )
    return ReconstructedPublicArmRamp(
        ramp_scale=ramp_scale,
        authority_scale=authority_scale,
        stable_count_after=stable_count_after,
        requested_arm=common_arm * authority_scale.unsqueeze(-1),
    )


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


def validate_collection_seal_name_only_diff(
    root: Path, *, implementation_commit: str, collection_commit: str
) -> None:
    """Require the post-implementation seal commit to change only the plan."""

    for name, value in (
        ("implementation_commit", implementation_commit),
        ("collection_commit", collection_commit),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a full lowercase Git SHA")
    completed = subprocess.run(
        (
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            implementation_commit,
            collection_commit,
            "--",
        ),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("cannot audit Candidate43 implementation-to-seal diff")
    names = tuple(line for line in completed.stdout.splitlines() if line)
    if names != (artifact_contract.VALIDATION_PLAN,):
        raise ValueError(
            "Candidate43 collection seal may differ from implementation only at the plan"
        )


def _git_resolve(root: Path, revision: str) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", f"{revision}^{{}}"),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or len(value) != 40:
        raise ValueError(f"cannot resolve registered Git authority {revision!r}")
    return value


def _git_blob_bytes(root: Path, *, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ("git", "show", f"{commit}:{relative}"),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(f"source is absent from authority commit: {relative}")
    return completed.stdout


def _protected_plan_contract(plan: Mapping[str, Any], *, pristine: bool) -> dict[str, Any]:
    """Return all preregistered fields that the final seal may not change."""

    try:
        strict = json.loads(
            json.dumps(
                dict(plan), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Candidate43 plan is not strict JSON") from error
    # Candidate43's preregistered mutation whitelist does not permit changing
    # the top-level status.  Final authority is expressed only by the filled
    # preregistration receipt plus implementation/collection seals.
    expected_status = "preregistered_before_implementation_or_simulator_evidence"
    if strict.get("status") != expected_status:
        raise ValueError("Candidate43 plan status is inconsistent with its seal phase")
    receipt = strict.get("preregistration_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("Candidate43 plan omitted its preregistration receipt")
    if pristine and (
        receipt.get("commit") is not None or receipt.get("sha256") is not None
    ):
        raise ValueError("pristine Candidate43 plan already contained post-hoc receipts")
    if pristine:
        if strict.get("implementation_seal") is not None or strict.get(
            "collection_seal"
        ) is not None:
            raise ValueError("pristine Candidate43 plan already contained a later seal")
    else:
        if not isinstance(strict.get("implementation_seal"), dict) or not isinstance(
            strict.get("collection_seal"), dict
        ):
            raise ValueError("final Candidate43 plan omitted a required seal")

    strict.pop("status")
    receipt.pop("commit", None)
    receipt.pop("sha256", None)
    strict.pop("implementation_seal", None)
    strict.pop("collection_seal", None)
    return strict


def validate_preregistration_protected_fields(
    pristine_plan: Mapping[str, Any], final_plan: Mapping[str, Any]
) -> None:
    """Reject every final-plan change not explicitly allowed by preregistration."""

    pristine = _protected_plan_contract(pristine_plan, pristine=True)
    final = _protected_plan_contract(final_plan, pristine=False)
    if final != pristine:
        raise ValueError("Candidate43 final plan changed a preregistered protected field")


def require_exact_collection_head(root: Path, plan: Mapping[str, Any]) -> str:
    """Require the attached HEAD named by the separately committed seal tag."""

    collection = plan.get("collection_seal")
    preregistration = plan.get("preregistration_receipt")
    if not isinstance(collection, Mapping) or not isinstance(
        preregistration, Mapping
    ):
        raise ValueError("Candidate43 plan omitted collection/preregistration authority")
    if collection.get("tag") != COLLECTION_SEAL_TAG:
        raise ValueError("Candidate43 collection-seal tag changed")
    if preregistration.get("tag") != PREREGISTRATION_TAG:
        raise ValueError("Candidate43 preregistration tag changed")
    head = _git_resolve(root, "HEAD")
    collection_commit = _git_resolve(root, COLLECTION_SEAL_TAG)
    preregistration_commit = _git_resolve(root, PREREGISTRATION_TAG)
    if head != collection_commit:
        raise ValueError("HEAD must equal the separately sealed Candidate43 collection commit")
    declared_preregistration = preregistration.get("commit")
    if declared_preregistration != preregistration_commit:
        raise ValueError("Candidate43 preregistration receipt disagrees with its tag")
    require_git_ancestor(root, preregistration_commit, collection_commit)
    pristine_sha = preregistration.get("sha256")
    if not isinstance(pristine_sha, str) or len(pristine_sha) != 64:
        raise ValueError("Candidate43 pristine plan SHA256 is absent")
    if _git_blob_sha256(
        root, commit=preregistration_commit, relative=artifact_contract.VALIDATION_PLAN
    ) != pristine_sha:
        raise ValueError("Candidate43 pristine preregistration plan changed")
    try:
        pristine_plan = json.loads(
            _git_blob_bytes(
                root,
                commit=preregistration_commit,
                relative=artifact_contract.VALIDATION_PLAN,
            ).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Candidate43 pristine preregistration plan is not JSON") from error
    if not isinstance(pristine_plan, dict):
        raise ValueError("Candidate43 pristine preregistration plan is not an object")
    validate_preregistration_protected_fields(pristine_plan, plan)
    return collection_commit


def _git_blob_sha256(root: Path, *, commit: str, relative: str) -> str:
    raw = _git_blob_bytes(root, commit=commit, relative=relative)
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


def validate_all_submodules_clean(root: Path) -> None:
    """Reject a dirty superproject or any divergent/dirty recursive submodule."""

    superproject = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if superproject.returncode != 0 or superproject.stdout.strip():
        raise ValueError("Candidate43 requires a clean superproject worktree")

    status = subprocess.run(
        ("git", "submodule", "status", "--recursive"),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if status.returncode != 0:
        raise ValueError("cannot audit Candidate43 recursive submodules")
    for line in status.stdout.splitlines():
        if line and line[0] != " ":
            raise ValueError("Candidate43 submodule is missing or at an unregistered commit")
    dirty = subprocess.run(
        (
            "git", "submodule", "foreach", "--quiet", "--recursive",
            "test -z \"$(git status --porcelain)\"",
        ),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if dirty.returncode != 0:
        raise ValueError("Candidate43 requires every recursive submodule to be clean")


def _validate_smoke_authority_coverage(
    episodes: Mapping[str, torch.Tensor],
) -> None:
    """Require the non-evidence smoke to traverse both D8 authority regimes."""

    treatment_eligible = episodes["treatment"] & episodes["eligible"]
    if not bool(treatment_eligible.any()):
        raise ValueError("Candidate43 smoke did not exercise a treatment overlay")
    if not bool((episodes["partial_authority_rows"][treatment_eligible] > 0).any()):
        raise ValueError("Candidate43 smoke did not exercise partial arm authority")
    if not bool((episodes["full_authority_rows"][treatment_eligible] > 0).any()):
        raise ValueError("Candidate43 smoke did not exercise full arm authority")


def _validate_smoke_prerequisite(root: Path) -> tuple[str, str]:
    smoke_stem = (root / artifact_contract.SMOKE_ARTIFACT_PATH).with_suffix("")
    transaction.validate_committed_namespace(
        transaction.TransactionPaths.from_output_stem(smoke_stem)
    )
    artifact_path = _regular_file(
        root / artifact_contract.SMOKE_ARTIFACT_PATH,
        label="canonical Candidate43 smoke artifact",
    )
    report_path = _regular_file(
        root / artifact_contract.SMOKE_REPORT_PATH,
        label="canonical Candidate43 smoke report",
    )
    artifact_sha = artifact_contract.sha256_file(artifact_path)
    report_sha = artifact_contract.sha256_file(report_path)
    try:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        raise ValueError("cannot load the canonical Candidate43 smoke pair") from error
    checked = artifact_contract.validate_artifact(artifact)
    checked_report = artifact_contract.validate_final_report(report, checked)
    if checked_report["artifact_sha256"] != artifact_sha:
        raise ValueError("Candidate43 smoke report does not bind its artifact bytes")
    if Path(checked_report["artifact_output"]).resolve() != artifact_path.resolve():
        raise ValueError("Candidate43 smoke report points to a non-canonical artifact")
    meta, episodes = checked["metadata"], checked["episodes"]
    if (meta["seed"], meta["replicate"], meta["num_envs"]) != (352, "b", 8):
        raise ValueError("Candidate43 smoke run identity changed")
    if meta["runtime_asset_sha256"] != RUNTIME_ASSET_EXPECTED_SHA256:
        raise ValueError("Candidate43 smoke runtime assets changed")
    _validate_smoke_authority_coverage(episodes)
    if bool(episodes["new_abs_action_ge_0999"].any()):
        raise ValueError("Candidate43 smoke failed an action-algebra prerequisite")
    return artifact_sha, report_sha


def _validate_causal_receipts(root: Path) -> dict[str, str]:
    receipts = (
        (artifact_contract.CANDIDATE42_PLAN, artifact_contract.CANDIDATE42_PLAN_SHA256),
        (
            artifact_contract.CANDIDATE42_RESULT,
            artifact_contract.CANDIDATE42_RESULT_SHA256,
        ),
        (
            artifact_contract.CANDIDATE42_REPORT,
            artifact_contract.CANDIDATE42_REPORT_SHA256,
        ),
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
    candidate42_plan = loaded[artifact_contract.CANDIDATE42_PLAN]
    candidate42_result = loaded[artifact_contract.CANDIDATE42_RESULT]
    candidate42_report = loaded[artifact_contract.CANDIDATE42_REPORT]
    manifest = loaded[artifact_contract.FIXED_DIRECTION_MANIFEST]
    candidate42_implementation = candidate42_plan.get("implementation_seal")
    candidate42_collection = candidate42_plan.get("collection_seal")
    if (
        candidate42_plan.get("kind")
        != "pick_tool_candidate42_transactional_public_arm_ramp_development_plan_v1"
        or candidate42_plan.get("status")
        != "preregistered_before_implementation_or_simulator_evidence"
        or not isinstance(candidate42_implementation, dict)
        or not isinstance(candidate42_collection, dict)
        or candidate42_collection.get("tag")
        != artifact_contract.CANDIDATE42_COLLECTION_SEAL_TAG
    ):
        raise ValueError("Candidate42 sealed-plan semantic identity changed")
    if (
        candidate42_result.get("kind")
        != "pick_tool_candidate42_transactional_public_arm_ramp_development_validation_result_v1"
        or candidate42_result.get("status") != "sealed_rejected_before_formal_screen"
        or candidate42_result.get("decision", {}).get("analyzer_decision")
        != "reject_candidate42"
        or candidate42_result.get("decision", {}).get("all_gates_pass") is not False
    ):
        raise ValueError("Candidate42 rejection-result semantic identity changed")
    if (
        candidate42_report.get("kind")
        != "pick_tool_candidate42_transactional_public_arm_ramp_development_validation_v1"
        or candidate42_report.get("status") != "complete"
        or candidate42_report.get("decision") != "reject_candidate42"
    ):
        raise ValueError("Candidate42 combined-report semantic identity changed")
    tag_object = subprocess.run(
        ("git", "rev-parse", f"refs/tags/{artifact_contract.CANDIDATE42_RESULT_TAG}"),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if (
        tag_object.returncode != 0
        or tag_object.stdout.strip()
        != artifact_contract.CANDIDATE42_RESULT_TAG_OBJECT_OID
        or _git_resolve(root, artifact_contract.CANDIDATE42_RESULT_TAG)
        != artifact_contract.CANDIDATE42_RESULT_COMMIT
        or _git_resolve(root, artifact_contract.CANDIDATE42_PREREGISTRATION_TAG)
        != artifact_contract.CANDIDATE42_PREREGISTRATION_COMMIT
        or _git_resolve(root, artifact_contract.CANDIDATE42_COLLECTION_SEAL_TAG)
        != artifact_contract.CANDIDATE42_COLLECTION_SEAL_COMMIT
    ):
        raise ValueError("Candidate42 causal Git tag authority changed")
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
        / f"logs/flashsac/pick_tool/55_c43_short_arm_ramp_dev_s{seed}_{replicate}/trial"
    ).resolve()


def _canonical_invocation_stem(
    root: Path, seed: int, replicate: str, num_envs: int
) -> Path:
    if (seed, replicate, num_envs) == CANDIDATE43_INVOCATION_SEQUENCE[0]:
        return (root / artifact_contract.SMOKE_ARTIFACT_PATH).with_suffix("").resolve()
    return _canonical_dev_stem(root, seed, replicate)


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_argument_value(name: str, value: Any) -> Any:
    if isinstance(value, os.PathLike):
        return str(Path(value).resolve())
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"argument {name} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _canonical_argument_value(f"{name}[{index}]", item)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"argument mapping {name} must have string keys")
        return {
            key: _canonical_argument_value(f"{name}.{key}", item)
            for key, item in sorted(value.items())
        }
    raise TypeError(f"argument {name} is not strict-JSON compatible")


def _argument_receipt(args: argparse.Namespace, output_stem: Path) -> dict[str, Any]:
    if not isinstance(args, argparse.Namespace):
        raise TypeError("args must be an argparse.Namespace")
    values = vars(args)
    missing = [name for name in USER_VISIBLE_ARGUMENT_FIELDS if name not in values]
    if missing:
        raise ValueError(f"missing user-visible arguments: {missing}")
    receipt = {
        name: _canonical_argument_value(name, values[name])
        for name in USER_VISIBLE_ARGUMENT_FIELDS
    }
    receipt["output_stem"] = str(output_stem.resolve())
    return receipt


def _argument_receipt_for_identity(
    receipt: Mapping[str, Any],
    *,
    seed: int,
    replicate: str,
    num_envs: int,
    output_stem: Path,
) -> dict[str, Any]:
    result = dict(receipt)
    result.update(
        {
            "seed": seed,
            "replicate": replicate,
            "num_envs": num_envs,
            "output_stem": str(output_stem.resolve()),
        }
    )
    return result


def _validate_exact_invocation_attempt(
    root: Path,
    *,
    seed: int,
    replicate: str,
    num_envs: int,
    collection_commit: str,
    assignment_mask_sha256: str,
    source_manifest_sha256: str,
    checkpoint_sha256: Mapping[str, Mapping[str, str] | str],
    argument_receipt: Mapping[str, Any],
    allow_current_canonical_evidence: bool = False,
) -> None:
    """Check only the registered canonical sequence before child rollout.

    Attempt allocation, retry receipts, boundary classification and recovery
    belong exclusively to the non-Isaac transaction parent.  The scientific
    collector merely refuses a missing prerequisite, an already committed
    current position, or any later canonical evidence.
    """

    identity = (seed, replicate, num_envs)
    if identity not in CANDIDATE43_INVOCATION_SEQUENCE:
        raise ValueError("invocation is outside Candidate43's exact five-entry sequence")
    current_index = CANDIDATE43_INVOCATION_SEQUENCE.index(identity)
    for index, (item_seed, item_replicate, item_envs) in enumerate(
        CANDIDATE43_INVOCATION_SEQUENCE
    ):
        stem = _canonical_invocation_stem(
            root, item_seed, item_replicate, item_envs
        )
        canonical = (
            Path(f"{stem}.pt"),
            Path(f"{stem}.json"),
            Path(f"{stem}.commit.json"),
        )
        owned = tuple(_owned(path) for path in canonical)
        if index < current_index:
            if owned != (True, True, True):
                raise ValueError(
                    f"Candidate43 invocation prerequisite is incomplete: "
                    f"{item_seed}{item_replicate}"
                )
        elif index == current_index:
            if any(owned) and not allow_current_canonical_evidence:
                raise FileExistsError(f"Candidate43 canonical output already exists: {stem}")
        elif any(owned):
            raise ValueError(
                f"later Candidate43 invocation already exists out of order: "
                f"{item_seed}{item_replicate}"
            )


def _validate_prior_authority(
    metadata: Mapping[str, Any],
    *,
    source_sha256: Mapping[str, str],
    smoke_artifact_sha256: str,
    smoke_report_sha256: str,
    collection_commit: str,
) -> None:
    if metadata.get("source_sha256") != dict(source_sha256):
        raise ValueError("prior Candidate43 artifact used a different source authority")
    if (
        metadata.get("smoke_artifact_sha256") != smoke_artifact_sha256
        or metadata.get("smoke_report_sha256") != smoke_report_sha256
    ):
        raise ValueError("prior Candidate43 artifact used different smoke authority")
    git = metadata.get("git")
    if not isinstance(git, Mapping) or git.get("commit") != collection_commit:
        raise ValueError("prior Candidate43 artifact used a different collection commit")
    if metadata.get("runtime_asset_sha256") != RUNTIME_ASSET_EXPECTED_SHA256:
        raise ValueError("prior Candidate43 artifact used different runtime assets")


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
    transaction.validate_committed_namespace(
        transaction.TransactionPaths.from_output_stem(
            artifact_path.with_suffix("")
        )
    )
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checked = artifact_contract.validate_artifact(artifact)
    checked_report = artifact_contract.validate_final_report(report, checked)
    digest = artifact_contract.sha256_file(artifact_path)
    if checked_report["artifact_sha256"] != digest or Path(
        checked_report["artifact_output"]
    ).resolve() != artifact_path.resolve():
        raise ValueError("prior Candidate43 report does not bind canonical artifact bytes")
    metadata = checked["metadata"]
    if (metadata["seed"], metadata["replicate"], metadata["num_envs"]) != (
        seed,
        replicate,
        64,
    ):
        raise ValueError("prior Candidate43 development run identity changed")
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
    if identity not in CANDIDATE43_RUN_ORDER:
        raise ValueError("run is outside the sealed Candidate43 development order")
    index = CANDIDATE43_RUN_ORDER.index(identity)
    expected_current = _canonical_dev_stem(root, seed, replicate)
    if output_stem.resolve() != expected_current:
        raise ValueError(f"development output_stem must be exactly {expected_current}")
    for prior_seed, prior_replicate in CANDIDATE43_RUN_ORDER[:index]:
        prior_stem = _canonical_dev_stem(root, prior_seed, prior_replicate)
        try:
            artifact_path = _regular_file(
                Path(f"{prior_stem}.pt"),
                label=f"prior Candidate43 artifact {prior_seed}{prior_replicate}",
            )
            report_path = _regular_file(
                Path(f"{prior_stem}.json"),
                label=f"prior Candidate43 report {prior_seed}{prior_replicate}",
            )
        except FileNotFoundError as error:
            raise ValueError(
                f"Candidate43 run order prerequisite is missing: {prior_seed}{prior_replicate}"
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
    for later_seed, later_replicate in CANDIDATE43_RUN_ORDER[index + 1 :]:
        later_stem = _canonical_dev_stem(root, later_seed, later_replicate)
        if any(
            _owned(Path(f"{later_stem}{suffix}"))
            for suffix in (".pt", ".json", ".commit.json")
        ):
            raise ValueError(
                f"later Candidate43 output already exists out of order: {later_seed}{later_replicate}"
            )


def validate_runtime_source_count(
    sealed_plan: Mapping[str, Any], source_sha256: Mapping[str, str]
) -> None:
    """Bind the static-audit source count to the actual pre-launch closure."""

    try:
        declared = sealed_plan["implementation_seal"]["static_audit"][
            "runtime_source_files"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("Candidate43 seal omitted the runtime source count") from error
    if type(declared) is not int or declared != len(source_sha256):
        raise ValueError(
            "Candidate43 static-audit runtime source count does not match "
            "the pre-launch source closure"
        )


def build_spec(
    args: argparse.Namespace, *, allow_current_canonical_evidence: bool = False
) -> CollectionSpec:
    """Freeze assignment, payloads, receipts, and smoke gate before AppLauncher."""

    root = Path(__file__).resolve().parents[2]
    if type(allow_current_canonical_evidence) is not bool:
        raise TypeError("allow_current_canonical_evidence must be bool")
    sealed_plan = artifact_contract.validate_sealed_plan()
    collection_commit = require_exact_collection_head(root, sealed_plan)
    require_git_ancestor(
        root, sealed_plan["implementation_seal"]["implementation_commit"]
    )
    if not isinstance(args.seed, int) or isinstance(args.seed, bool):
        raise TypeError("seed must be an integer")
    if type(args.num_envs) is not int or args.num_envs not in ALLOWED_NUM_ENVS:
        raise ValueError(f"num_envs must be one of {ALLOWED_NUM_ENVS}")
    if args.replicate not in {"a", "b"}:
        raise ValueError("replicate must be a or b")
    if args.num_envs == 8 and (args.seed, args.replicate) != (352, "b"):
        raise ValueError("the only Candidate43 smoke is exact run 352b/8-env")
    if args.num_envs == 64 and args.seed not in {353, 354}:
        raise ValueError("Candidate43 development uses only seeds 353 and 354")
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
            f"Candidate43 requires exact --kit_args={artifact_contract.KIT_ARGS!r}"
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
            raise ValueError(f"Candidate43 {label} is not the registered canonical path")
    causal_hashes = _validate_causal_receipts(root)
    checkpoint_sha256: dict[str, Mapping[str, str] | str] = {
        "v6": _checkpoint_hashes(v6),
        "search": artifact_contract.sha256_file(search),
        "fixed_direction": artifact_contract.sha256_file(fixed),
        **causal_hashes,
    }
    if checkpoint_sha256["v6"] != EXPECTED_V6_SHA256:
        raise ValueError("Candidate43 requires the exact sealed V6 checkpoint")
    if checkpoint_sha256["search"] != artifact_contract.SEARCH_CHECKPOINT_SHA256:
        raise ValueError("Candidate43 requires the exact sealed SEARCH checkpoint")
    if checkpoint_sha256["fixed_direction"] != artifact_contract.FIXED_DIRECTION_SHA256:
        raise ValueError("Candidate43 requires the exact sealed fixed_z payload")
    fixed_payload = artifact_contract.load_fixed_direction(fixed)
    frozen_source_sha256 = source_fingerprints(root)
    validate_runtime_source_count(sealed_plan, frozen_source_sha256)
    implementation_commit = sealed_plan["implementation_seal"][
        "implementation_commit"
    ]
    validate_collection_seal_name_only_diff(
        root,
        implementation_commit=implementation_commit,
        collection_commit=collection_commit,
    )
    validate_implementation_source_authority(
        root,
        frozen_source_sha256,
        implementation_commit=implementation_commit,
    )
    # These bytes are required before AppLauncher as well as rechecked inside
    # and after collection, so a stale generated USD cannot consume a seed.
    runtime_asset_fingerprints(root)
    validate_all_submodules_clean(root)
    prelaunch_git = git_provenance(
        root, tuple(sorted(frozen_source_sha256))
    )
    if (
        prelaunch_git["commit"] != collection_commit
        or prelaunch_git["branch"] != artifact_contract.REQUIRED_BRANCH
        or prelaunch_git["source_files_dirty"]
        or prelaunch_git["flashsac_dirty"]
        or prelaunch_git["flashsac_commit"] != artifact_contract.FLASHSAC_FORK_COMMIT
    ):
        raise ValueError(
            "Candidate43 pre-launch Git authority is dirty, detached, or changed"
        )

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
    treatment = verifier.exact_balanced_treatment_mask(
        seed=args.seed, num_envs=args.num_envs, replicate=args.replicate
    )
    rank = verifier.assignment_rank(seed=args.seed, num_envs=args.num_envs)
    assignment_a = rank < (args.num_envs // 2)
    expected_from_rank = assignment_a if args.replicate == "a" else ~assignment_a
    if not torch.equal(treatment, expected_from_rank):
        raise RuntimeError("Candidate43 assignment mask disagrees with assignment rank")
    if int(treatment.sum()) != args.num_envs // 2:
        raise RuntimeError("Candidate43 assignment is not exactly balanced")
    assignment_sha = verifier.assignment_mask_sha256(treatment)
    argument_receipt = _argument_receipt(args, output_stem)
    _validate_exact_invocation_attempt(
        root,
        seed=int(args.seed),
        replicate=str(args.replicate),
        num_envs=int(args.num_envs),
        collection_commit=collection_commit,
        assignment_mask_sha256=assignment_sha,
        source_manifest_sha256=artifact_contract.manifest_sha256(
            frozen_source_sha256
        ),
        checkpoint_sha256=checkpoint_sha256,
        argument_receipt=argument_receipt,
        allow_current_canonical_evidence=allow_current_canonical_evidence,
    )

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
        checkpoint_sha256=checkpoint_sha256,
        source_sha256=frozen_source_sha256,
        collection_commit=collection_commit,
        implementation_commit=implementation_commit,
        preregistration_tag_commit=str(
            sealed_plan["preregistration_receipt"]["commit"]
        ),
        preregistration_plan_sha256=str(
            sealed_plan["preregistration_receipt"]["sha256"]
        ),
        argument_receipt=argument_receipt,
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


def add_scientific_arguments(parser: argparse.ArgumentParser) -> None:
    """Register only campaign arguments, without importing AppLauncher."""

    parser.add_argument(
        "--v6_checkpoint",
        type=Path,
        default=Path(artifact_contract.V6_CHECKPOINT_PATH),
    )
    parser.add_argument(
        "--search_checkpoint",
        type=Path,
        default=Path(artifact_contract.SEARCH_CHECKPOINT_PATH),
    )
    parser.add_argument(
        "--fixed_direction",
        type=Path,
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
        "--token_component_cap",
        type=float,
        default=artifact_contract.TOKEN_COMPONENT_CAP,
    )
    parser.add_argument(
        "--distal_component_cap",
        type=float,
        default=artifact_contract.DISTAL_COMPONENT_CAP,
    )
    parser.add_argument(
        "--pre_tanh_l2_cap",
        type=float,
        default=artifact_contract.PRE_TANH_L2_CAP,
    )
    parser.add_argument("--verify_steps", type=int, default=verifier.VERIFY_STEPS)
    parser.add_argument("--output_stem", type=Path, required=True)


def _current_hashes(spec: CollectionSpec) -> dict[str, Mapping[str, str] | str]:
    current: dict[str, Mapping[str, str] | str] = {
        "v6": _checkpoint_hashes(spec.v6_checkpoint),
        "search": artifact_contract.sha256_file(spec.search_checkpoint),
        "fixed_direction": artifact_contract.sha256_file(spec.fixed_direction),
    }
    for relative in (
        artifact_contract.CANDIDATE42_PLAN,
        artifact_contract.CANDIDATE42_RESULT,
        artifact_contract.CANDIDATE42_REPORT,
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
                label="canonical Candidate43 smoke artifact",
            )
        )
        current[artifact_contract.SMOKE_REPORT_PATH] = artifact_contract.sha256_file(
            _regular_file(
                spec.repository_root / artifact_contract.SMOKE_REPORT_PATH,
                label="canonical Candidate43 smoke report",
            )
        )
    return current


def validate_post_application_authority(spec: CollectionSpec) -> None:
    """Recheck every mutable authority after environment/application teardown."""

    source = source_fingerprints(spec.repository_root)
    causal_hashes = _validate_causal_receipts(spec.repository_root)
    runtime_assets = runtime_asset_fingerprints(spec.repository_root)
    if source != dict(spec.source_sha256) or _current_hashes(spec) != dict(
        spec.checkpoint_sha256
    ):
        raise RuntimeError("Candidate43 source or checkpoint changed after collection")
    if any(spec.checkpoint_sha256.get(path) != digest for path, digest in causal_hashes.items()):
        raise RuntimeError("Candidate43 causal receipt changed after collection")
    if runtime_assets != RUNTIME_ASSET_EXPECTED_SHA256:
        raise RuntimeError("Candidate43 runtime asset changed after collection")
    git_paths = tuple(
        sorted(
            set(source)
            | {path for path in runtime_assets if not Path(path).is_absolute()}
        )
    )
    provenance = git_provenance(spec.repository_root, git_paths)
    if (
        provenance["commit"] != spec.collection_commit
        or provenance["branch"] != artifact_contract.REQUIRED_BRANCH
        or provenance["source_files_dirty"]
        or provenance["flashsac_commit"] != artifact_contract.FLASHSAC_FORK_COMMIT
        or provenance["flashsac_dirty"]
        or _git_resolve(spec.repository_root, COLLECTION_SEAL_TAG)
        != spec.collection_commit
    ):
        raise RuntimeError("Candidate43 Git authority changed after collection")
    sealed_plan = artifact_contract.validate_sealed_plan()
    preregistration_receipt = sealed_plan.get("preregistration_receipt")
    if not isinstance(preregistration_receipt, Mapping):
        raise RuntimeError("Candidate43 sealed plan lost preregistration authority")
    declared_preregistration_commit = preregistration_receipt.get("commit")
    actual_preregistration_commit = _git_resolve(
        spec.repository_root, PREREGISTRATION_TAG
    )
    if (
        declared_preregistration_commit != spec.preregistration_tag_commit
        or actual_preregistration_commit != spec.preregistration_tag_commit
    ):
        raise RuntimeError(
            "Candidate43 preregistration tag or sealed receipt changed after collection"
        )
    validate_all_submodules_clean(spec.repository_root)


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


def pre_trace_record_mask(
    *,
    preeligible: torch.Tensor,
    option_active: torch.Tensor,
    trigger: torch.Tensor,
    newly_latched: torch.Tensor,
) -> torch.Tensor:
    """Select pre rows, leaving trigger==eligibility to the post trace."""

    if not isinstance(preeligible, torch.Tensor) or preeligible.ndim != 1:
        raise ValueError("preeligible must be a rank-one tensor")
    rows = int(preeligible.numel())
    device = preeligible.device
    for name, value in (
        ("preeligible", preeligible),
        ("option_active", option_active),
        ("trigger", trigger),
        ("newly_latched", newly_latched),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (rows,)
            or value.dtype != torch.bool
            or value.device != device
        ):
            raise ValueError(f"{name} must be a [{rows}] bool tensor on {device}")
    return preeligible & (option_active | trigger | newly_latched)


@torch.inference_mode()
def run_collection(
    spec: CollectionSpec,
    *,
    device_string: str,
    durable_boundary: Callable[[], None],
    progress: AttemptProgress | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if progress is None:
        progress = AttemptProgress()
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Candidate43 collection requires CUDA, got {device}")

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
        raise RuntimeError("pre-launch Candidate43 assignment changed")
    if verifier.assignment_mask_sha256(spec.treatment) != spec.assignment_mask_sha256:
        raise RuntimeError("pre-launch Candidate43 assignment receipt changed")
    fixed_payload = artifact_contract.load_fixed_direction(spec.fixed_direction)
    if not torch.equal(fixed_payload["fixed_z"], spec.fixed_z):
        raise RuntimeError("pre-launch Candidate43 fixed_z changed")

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
        raise RuntimeError("Candidate43 source authority changed after pre-launch freeze")
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
        raise RuntimeError("Candidate43 collection requires committed clean sources")
    if (
        git_before["commit"] != spec.collection_commit
        or git_before["branch"] != artifact_contract.REQUIRED_BRANCH
        or git_before["flashsac_commit"] != artifact_contract.FLASHSAC_FORK_COMMIT
    ):
        raise RuntimeError("Candidate43 Git authority changed after pre-launch freeze")
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
            raise RuntimeError("a sealed Candidate43 input changed while loading")

        observation, reset_info = env.reset(
            seed=spec.seed, randomize_episode_lengths=False
        )
        if observation.shape != (spec.num_envs, artifact_contract.OBSERVATION_DIM):
            raise RuntimeError("Candidate43 reset violated obs115")
        public_counters = public_gate_feature_tensor(
            reset_info[PUBLIC_GATE_STATE_EXTRAS_KEY],
            num_envs=spec.num_envs,
            device=env.device,
        )
        validate_public_gate_feature_values(public_counters)
        initial_truth = _read_physical_truth(env.unwrapped, task_mode=FULL_TASK_MODE)
        if bool(initial_truth.grasped.any()):
            raise RuntimeError("Candidate43 reset unexpectedly begins grasped")
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
            [artifact_contract.TOKEN_COMPONENT_CAP] * TOKEN_ACTION_DIM
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
        verifier_state = verifier.initial_public_arm_ramp_state(
            spec.num_envs, device=env.device
        )
        authoritative_reset_mask = torch.zeros_like(option_active)
        first_eligible_step = minus_one.clone()
        first_positive_authority_step = minus_one.clone()
        first_full_authority_step = minus_one.clone()
        first_relock_step = minus_one.clone()
        first_reramp_step = minus_one.clone()
        intervention_steps = zeros_long.clone()
        stable_count_max = zeros_long.clone()
        authority_scale_sum = torch.zeros(
            spec.num_envs, dtype=torch.float32, device=env.device
        )
        positive_authority_rows = zeros_long.clone()
        partial_authority_rows = zeros_long.clone()
        full_authority_rows = zeros_long.clone()
        relock_count = zeros_long.clone()
        reramp_count = zeros_long.clone()
        trajectory_max_force = torch.zeros(
            spec.num_envs, dtype=torch.float32, device=env.device
        )
        first_eligible_clearance = torch.zeros_like(trajectory_max_force)
        option_preeligible_u5 = torch.zeros_like(option_active)

        violation_names = (
            "pre_eligibility_action_violations",
            "hand_invariance_violations",
            "treatment_arm_ramp_violations",
            "authority_clock_violations",
            "control_route_violations",
            "task_action_reconstruction_violations",
            "fixed_residual_budget_violations",
            "action_bound_violations",
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
            "pre_trigger_witness", "pre_first_latch_witness",
            "pre_transition_public_latch",
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
            "row_relock", "row_reramp", "row_transition_public_latch",
            "row_transition_grasped", "row_transition_done",
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
                raise RuntimeError("a Candidate43 slot triggered twice")
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
                raise RuntimeError("fixed residual assignment incorrectly used Candidate43 arm")
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

            state_before = verifier.reset_public_arm_ramp_state(
                verifier_state, authoritative_reset_mask
            )
            if bool(authoritative_reset_mask.any()):
                reset_ids = authoritative_reset_mask
                if bool(state_before.activated[reset_ids].any()) or bool(
                    (state_before.stable_count[reset_ids] != 0).any()
                ) or bool((state_before.ramp_scale_previous[reset_ids] != 0.0).any()) or bool(
                    state_before.ever_positive_authority[reset_ids].any()
                ) or bool(
                    state_before.ever_full_authority[reset_ids].any()
                ):
                    raise RuntimeError("authoritative reset receipt retained verifier state")
            ramped = verifier.apply_public_arm_ramp(
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
                    ramped.activated_before[reset_ids],
                    state_before.activated[reset_ids],
                ) or not torch.equal(
                    ramped.stable_count_before[reset_ids],
                    state_before.stable_count[reset_ids],
                ):
                    raise RuntimeError("authoritative reset did not clear verifier state")
            requested_action = ramped.action
            first_eligible_step = torch.where(
                ramped.first_eligible, episode_step, first_eligible_step
            )
            first_eligible_clearance = torch.where(
                ramped.first_eligible, current_clearance, first_eligible_clearance
            )
            positive_now = ramped.activated_this_action & (
                ramped.authority_scale > 0.0
            )
            full_now = ramped.activated_this_action & (
                ramped.authority_scale == 1.0
            )
            first_positive_authority_step = torch.where(
                positive_now & (first_positive_authority_step < 0),
                episode_step,
                first_positive_authority_step,
            )
            first_full_authority_step = torch.where(
                full_now & (first_full_authority_step < 0),
                episode_step,
                first_full_authority_step,
            )
            first_relock_step = torch.where(
                ramped.relock & (first_relock_step < 0),
                episode_step,
                first_relock_step,
            )
            first_reramp_step = torch.where(
                ramped.reramp & (first_reramp_step < 0),
                episode_step,
                first_reramp_step,
            )
            _increment(relock_count, ramped.relock & active_before)
            _increment(reramp_count, ramped.reramp & active_before)
            stable_count_max = torch.where(
                active_before,
                torch.maximum(stable_count_max, ramped.stable_count_after),
                stable_count_max,
            )
            trace_active = active_before & ramped.activated_this_action
            authority_scale_sum += torch.where(
                trace_active, ramped.authority_scale, torch.zeros_like(ramped.authority_scale)
            )
            positive_authority = trace_active & (ramped.authority_scale > 0.0)
            partial_authority = positive_authority & (ramped.authority_scale < 1.0)
            full_authority = trace_active & (ramped.authority_scale == 1.0)
            _increment(positive_authority_rows, positive_authority)
            _increment(partial_authority_rows, partial_authority)
            _increment(full_authority_rows, full_authority)

            audit = reconstruct_public_arm_ramp(
                common_arm=common_action[:, : artifact_contract.ARM_ACTION_DIM],
                stable_current=ramped.stable_current,
                stable_count_before=ramped.stable_count_before,
                treatment=treatment,
                activated_this_action=ramped.activated_this_action,
                option_active=option_active,
                episode_active=active_before,
            )
            authority_clock_bad = active_before & (
                (ramped.ramp_scale != audit.ramp_scale)
                | (ramped.authority_scale != audit.authority_scale)
                | (ramped.stable_count_after != audit.stable_count_after)
            )
            preeligible_bad = (
                active_before
                & (~ramped.activated_this_action)
                & (requested_action != common_action).any(dim=-1)
            )
            hand_bad = active_before & (
                requested_action[:, artifact_contract.ARM_ACTION_DIM :]
                != common_action[:, artifact_contract.ARM_ACTION_DIM :]
            ).any(dim=-1)
            treatment_gate_bad = (
                active_before
                & treatment
                & (
                    requested_action[:, : artifact_contract.ARM_ACTION_DIM]
                    != audit.requested_arm
                ).any(dim=-1)
            )
            control_bad = active_before & (~treatment) & (
                requested_action != common_action
            ).any(dim=-1)
            bounds_bad = active_before & (
                (~torch.isfinite(requested_action)).any(dim=-1)
                | (requested_action.abs() > 1.0).any(dim=-1)
            )
            for name, mask in (
                ("pre_eligibility_action_violations", preeligible_bad),
                ("hand_invariance_violations", hand_bad),
                ("treatment_arm_ramp_violations", treatment_gate_bad),
                ("authority_clock_violations", authority_clock_bad),
                ("control_route_violations", control_bad),
                ("action_bound_violations", bounds_bad),
            ):
                _increment(violations[name], mask)
            pre_reference = torch.where(
                ramped.activated_this_action.unsqueeze(-1),
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
                ramped.activated_this_action,
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
                "pre_trigger_witness": trigger,
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
                "row_public_force_counters": ramped.force_counter_features,
                "row_option_active": option_active,
                "row_public_latch_before": public_latch,
                "row_stable_current": ramped.stable_current,
                "row_public_grasp_quality": ramped.public_grasp_quality,
                "row_public_hold_quality": ramped.public_hold_quality,
                "row_public_max_force_strength": ramped.public_max_force_strength,
                "row_stable_count_before": ramped.stable_count_before,
                "row_stable_count_after": ramped.stable_count_after,
                "row_authority_scale": ramped.authority_scale,
                "row_common_action": common_action,
                "row_requested_action": requested_action,
                "row_arm_target_error": target_error,
                "row_arm_joint_velocity": arm_velocity,
                "row_arm_target_error_abs_max": target_error_max,
                "row_arm_joint_speed_abs_max": arm_speed_max,
                "row_pre_action_true_clearance_m": current_clearance,
                "row_treatment": treatment,
                "row_activated": ramped.activated_this_action,
                "row_first_eligible": ramped.first_eligible,
                "row_relock": ramped.relock,
                "row_reramp": ramped.reramp,
            }

            next_observation, reward, terminated, truncated, info = authoritative_env_step(
                env,
                requested_action,
                progress=progress,
                durable_boundary=durable_boundary,
            )
            vector_steps += 1
            executed = env.last_executed_policy_action
            if executed is None or not torch.equal(executed, requested_action):
                raise RuntimeError("adapter executed a different Candidate43 request")
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

            preeligible = active_before & (~ramped.activated_this_action)
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
            pre_record = pre_trace_record_mask(
                preeligible=preeligible,
                option_active=option_active,
                trigger=trigger,
                newly_latched=newly_latched,
            )
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
                }
            )
            trace_record = active_before & ramped.activated_this_action
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
            verifier_state = ramped.next_state
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
                f"Candidate43 completed {len(tracker.records)}/{spec.num_envs} first episodes"
            )
        cleared_final = verifier.reset_public_arm_ramp_state(
            verifier_state, authoritative_reset_mask
        )
        if bool(cleared_final.activated.any()) or bool(
            (cleared_final.stable_count != 0).any()
        ) or bool(
            (cleared_final.ramp_scale_previous != 0.0).any()
        ) or bool(
            cleared_final.ever_positive_authority.any()
        ) or bool(
            cleared_final.ever_full_authority.any()
        ):
            raise RuntimeError("terminal authoritative reset did not clear verifier state")
        validate_public_gate_feature_values(public_counters)
        records_by_slot = {
            int(record["env_slot"]): record for record in tracker.records
        }
        if set(records_by_slot) != set(range(spec.num_envs)):
            raise RuntimeError("Candidate43 tracker omitted an environment slot")
        ordered = [records_by_slot[index] for index in range(spec.num_envs)]
        if any(int(record["slot_episode_index"]) != 0 for record in ordered):
            raise RuntimeError("Candidate43 artifact contains a post-reset episode")

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
            "first_positive_authority_step": first_positive_authority_step.cpu(),
            "first_full_authority_step": first_full_authority_step.cpu(),
            "first_relock_step": first_relock_step.cpu(),
            "first_reramp_step": first_reramp_step.cpu(),
            "terminal_step": episode_lengths - 1,
            "intervention_steps": intervention_steps.cpu(),
            "trace_rows": trace_count,
            "stable_count_max": stable_count_max.cpu(),
            "positive_authority_rows": positive_authority_rows.cpu(),
            "partial_authority_rows": partial_authority_rows.cpu(),
            "full_authority_rows": full_authority_rows.cpu(),
            "authority_sum": authority_scale_sum.cpu(),
            "relock_count": relock_count.cpu(),
            "reramp_count": reramp_count.cpu(),
            "episode_length": episode_lengths,
            "trajectory_max_force_n": trajectory_max_force.cpu(),
            "max_true_clearance_m": torch.tensor(
                [float(record["max_true_clearance_m"]) for record in ordered],
                dtype=torch.float32,
            ),
            "first_eligible_clearance_m": first_eligible_clearance.cpu(),
            "eligible": (first_eligible_step >= 0).cpu(),
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
            raise RuntimeError("Candidate43 source/runtime/checkpoint changed during collection")
        v6_hashes = spec.checkpoint_sha256["v6"]
        if not isinstance(v6_hashes, Mapping):
            raise TypeError("V6 checkpoint receipt is not a mapping")
        metadata = {
            **artifact_contract.REQUIRED_METADATA,
            "seed": spec.seed,
            "replicate": spec.replicate,
            "num_envs": spec.num_envs,
            "assignment_mask_sha256": spec.assignment_mask_sha256,
            "implementation_commit": spec.implementation_commit,
            "preregistration_plan_sha256": spec.preregistration_plan_sha256,
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
