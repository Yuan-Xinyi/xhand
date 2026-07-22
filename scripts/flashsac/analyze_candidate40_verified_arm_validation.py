#!/usr/bin/env python3
"""Fail-closed development analysis for Candidate40 verified-arm handoff.

The randomized difference begins only on the first option-active, latch-visible
pre-action decision.  Consequently every causal estimand is evaluated on that
pre-treatment eligibility domain.  Verification completion and arm enablement
are post-treatment mediators and are reported only as outcomes/funnels; they
never select the Horvitz--Thompson population.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import torch

import candidate40_verified_arm_episode as artifact_contract
import candidate40_verified_arm_handoff as verifier
from collect_online_close_ab import RUNTIME_ASSET_EXPECTED_SHA256


REPORT_KIND = "pick_tool_candidate40_verified_arm_development_validation_v1"
FORMAT_VERSION = 1
SEEDS = (335, 336)
REPLICATES = ("a", "b")
RUN_ORDER = ((335, "a"), (335, "b"), (336, "b"), (336, "a"))
NUM_ENVS = 64
KNOWN_PROPENSITY = 0.5
UTILITY_HORIZON = 128
STABLE_WINDOW = 15
STABLE_5CM_FRACTION = 0.25
STABLE_5CM_ACTION_HORIZON = 96

INPUT_PATTERN = (
    "logs/flashsac/pick_tool/52_c40_verified_arm_dev_s{seed}_{replicate}/trial.pt"
)
REPORT_OUTPUT = (
    "logs/flashsac/pick_tool/52_c40_verified_arm_development_validation.json"
)

EVENT_FIELDS = {
    "stable_transport_restricted_mean": None,
    "terminal_utility": None,
    "success": "success",
    "ever_true_clearance_ge_20cm": "ever_clearance_ge_20cm",
    "stable_held_5cm_by_action96": None,
    "never_stable_held_5cm_by_action96": None,
    "ever_grasped": "ever_grasped",
    "unlatched_clearance_ge_5cm": "unlatched_clearance_ge_5cm",
}

# Executable names, scalar types, and thresholds must match the sealed plan
# exactly.  A report is eligible to advance only when every row passes.
GATE_THRESHOLDS: dict[str, Any] = {
    "any_new_abs_action_ge_0999": False,
    "checkpoint_manifest_assignment_and_source_identity_exact": True,
    "conditional_ever_grasped_delta_floor": -0.03,
    "conditional_never_stable_held_5cm_by_action96_delta_max": 0.0,
    "conditional_success_delta_floor": -0.03,
    "conditional_true_clearance_ge_20cm_delta_floor": -0.03,
    "conditional_unlatched_clearance_ge_5cm_delta_max": 0.0,
    "control_candidate39_route_exact": True,
    "eligible_opportunity_slots_min_across_validation": 32,
    "eligible_opportunity_slots_min_per_seed": 12,
    "eligible_opportunity_slots_min_per_seed_arm": 8,
    "fixed_residual_component_and_l2_budget_violations": 0,
    "hand14_invariance_exact": True,
    "paired_common_eligibility_unlatched_clearance_ge_5cm_delta_max": 0.0,
    "paired_common_eligible_slots_min_across_validation": 16,
    "paired_common_eligible_slots_min_per_seed": 6,
    "per_seed_primary_delta_floor": -0.1,
    "pooled_option_active_pre_eligible_unlatched_clearance_ge_5cm_max": 0,
    "positive_primary_delta_seeds_min": 1,
    "pre_eligibility_action_and_state_machine_parity_exact": True,
    "primary_conditional_stable_transport_restricted_mean_delta_min_exclusive": 0.0,
    "treatment_eligible_dropped_max": 0,
    "treatment_eligible_first_arm_releases_by_action32_min_across_validation": 16,
    "treatment_eligible_first_arm_releases_by_action32_min_per_seed": 4,
    "treatment_eligible_post_latch_unlatched_clearance_ge_5cm_max": 0,
    "treatment_eligible_pre_release_launch_max": 0,
    "treatment_eligible_stable_held_5cm_by_action96_min_across_validation": 8,
    "treatment_eligible_stable_held_5cm_by_action96_min_per_seed": 2,
    "treatment_eligible_unsafe_force_max": 0,
    "treatment_pre_enable_nonzero_arm_rows_max": 0,
    "verified_arm_clock_gate_and_live_relock_exact": True,
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _regular_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular non-symlink file: {path}")
    return path


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _torch_load_weights_only(raw: bytes) -> Any:
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("weights_only PyTorch loading is required") from error


def _same_typed_scalar(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _read_plan(repository_root: Path | None = None) -> tuple[dict[str, Any], str]:
    root = _root() if repository_root is None else Path(repository_root)
    path = _regular_file(
        root / artifact_contract.VALIDATION_PLAN, "Candidate40 validation plan"
    )
    raw = path.read_bytes()
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("Candidate40 plan is invalid JSON") from error
    if not isinstance(plan, dict):
        raise RuntimeError("Candidate40 plan must be a JSON object")
    return plan, _sha256_bytes(raw)


def _registered_validation_gates(
    repository_root: Path | None = None,
) -> dict[str, Any]:
    plan, _ = _read_plan(repository_root)
    gates = plan.get("development_validation_gates")
    if not isinstance(gates, dict):
        raise RuntimeError("Candidate40 plan has no development gate mapping")
    gates = dict(gates)
    if gates.pop("all_must_pass", None) is not True:
        raise RuntimeError("Candidate40 gates are not fail-closed")
    if set(gates) != set(GATE_THRESHOLDS):
        raise RuntimeError("Candidate40 analyzer gate keys differ from the plan")
    for name, expected in GATE_THRESHOLDS.items():
        if not _same_typed_scalar(gates[name], expected):
            raise RuntimeError(f"Candidate40 threshold {name!r} differs from the plan")
    return gates


def _validate_implementation_seal(
    plan: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    seal = plan.get("implementation_seal")
    expected_fields = {
        "status",
        "implementation_commit",
        "source_sha256",
        "simulation_free_tests",
    }
    if not isinstance(seal, Mapping) or set(seal) != expected_fields:
        raise ValueError("Candidate40 implementation seal schema changed")
    if seal["status"] != "complete_without_simulator_evidence":
        raise ValueError("Candidate40 implementation is not complete and simulator-free")
    commit = seal["implementation_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("Candidate40 implementation commit must be a full Git SHA")
    source = seal["source_sha256"]
    expected_sources = tuple(artifact_contract.IMPLEMENTATION_SOURCE_FILES)
    if not isinstance(source, dict) or set(source) != set(expected_sources):
        raise ValueError("Candidate40 implementation source receipt set changed")
    for relative in expected_sources:
        digest = source[relative]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid implementation digest: {relative}")
        current = _sha256_bytes(
            _regular_file(repository_root / relative, relative).read_bytes()
        )
        if current != digest:
            raise ValueError(f"Candidate40 source changed after seal: {relative}")
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repository_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if committed.returncode != 0 or _sha256_bytes(committed.stdout) != digest:
            raise ValueError(
                f"implementation commit does not contain sealed source: {relative}"
            )
    tests = seal["simulation_free_tests"]
    if not isinstance(tests, dict) or not tests:
        raise ValueError("Candidate40 simulation-free test receipt is empty")
    _strict_json_bytes({"simulation_free_tests": tests})
    return {
        "status": seal["status"],
        "implementation_commit": commit,
        "source_sha256": dict(source),
        "simulation_free_tests": dict(tests),
    }


def _validate_collection_commit_provenance(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    implementation: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Prove collection followed the seal without changing sealed sources."""

    implementation_commit = implementation.get("implementation_commit")
    sealed_sources = implementation.get("source_sha256")
    if not isinstance(implementation_commit, str) or not isinstance(
        sealed_sources, Mapping
    ):
        raise ValueError("Candidate40 implementation receipt is incomplete")
    collection_commits = {
        artifacts[key]["metadata"]["git"]["commit"] for key in RUN_ORDER
    }
    if len(collection_commits) != 1:
        raise ValueError("Candidate40 development runs used different Git commits")
    collection_commit = next(iter(collection_commits))
    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            implementation_commit,
            collection_commit,
        ],
        cwd=repository_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ancestry.returncode == 1:
        raise ValueError(
            "Candidate40 collection commit is not descended from the "
            "implementation seal"
        )
    if ancestry.returncode != 0:
        raise RuntimeError("could not verify Candidate40 collection ancestry")

    metadata_manifest = artifacts[RUN_ORDER[0]]["metadata"]["source_sha256"]
    verified_sources: dict[str, str] = {}
    for relative, expected_digest in sealed_sources.items():
        if metadata_manifest.get(relative) != expected_digest:
            raise ValueError(
                f"Candidate40 collection manifest omitted sealed source: {relative}"
            )
        committed = subprocess.run(
            ["git", "show", f"{collection_commit}:{relative}"],
            cwd=repository_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if (
            committed.returncode != 0
            or _sha256_bytes(committed.stdout) != expected_digest
        ):
            raise ValueError(
                f"Candidate40 sealed source changed by collection commit: {relative}"
            )
        verified_sources[str(relative)] = str(expected_digest)

    flash_prefix = "third_party/FlashSAC/"
    generated = {
        path: digest
        for path, digest in RUNTIME_ASSET_EXPECTED_SHA256.items()
        if not Path(path).is_absolute()
    }

    def git_blob(root: Path, commit: str, relative: str) -> str:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"Candidate40 authority commit omitted runtime source: {relative}"
            )
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
                raise ValueError(
                    f"Candidate40 malformed Git-LFS pointer: {relative}"
                )
            digest = lines[1].removeprefix("oid sha256:")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    f"Candidate40 malformed Git-LFS oid: {relative}"
                )
            try:
                expected_size = int(lines[2].removeprefix("size "))
            except ValueError as error:
                raise ValueError(
                    f"Candidate40 malformed Git-LFS size: {relative}"
                ) from error
            current = _regular_file(
                root / relative, f"Candidate40 Git-LFS source {relative}"
            )
            if expected_size < 0 or current.stat().st_size != expected_size:
                raise ValueError(
                    f"Candidate40 Git-LFS source size changed: {relative}"
                )
            if _sha256_bytes(current.read_bytes()) != digest:
                raise ValueError(
                    f"Candidate40 Git-LFS content changed: {relative}"
                )
            return digest
        return _sha256_bytes(raw)

    verified_runtime: dict[str, str] = {}
    for relative, expected_digest in metadata_manifest.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Candidate40 source manifest contains an unsafe path: {relative}"
            )
        if relative == artifact_contract.VALIDATION_PLAN:
            authority = git_blob(repository_root, collection_commit, relative)
        elif relative.startswith(flash_prefix):
            authority = git_blob(
                repository_root / "third_party/FlashSAC",
                artifact_contract.FLASHSAC_FORK_COMMIT,
                relative.removeprefix(flash_prefix),
            )
        elif relative in generated:
            authority = generated[relative]
        else:
            implementation_authority = git_blob(
                repository_root, implementation_commit, relative
            )
            collection_authority = git_blob(
                repository_root, collection_commit, relative
            )
            if implementation_authority != collection_authority:
                raise ValueError(
                    "Candidate40 runtime source changed after implementation "
                    f"seal: {relative}"
                )
            authority = implementation_authority
        if authority != expected_digest:
            raise ValueError(
                f"Candidate40 runtime source receipt changed: {relative}"
            )
        verified_runtime[str(relative)] = str(expected_digest)
    expected_runtime_assets = dict(RUNTIME_ASSET_EXPECTED_SHA256)
    for key in RUN_ORDER:
        if (
            artifacts[key]["metadata"]["runtime_asset_sha256"]
            != expected_runtime_assets
        ):
            raise ValueError(
                f"Candidate40 {key[0]}{key[1]} runtime asset receipt changed"
            )
    return {
        "implementation_commit": implementation_commit,
        "collection_commit": collection_commit,
        "implementation_commit_is_ancestor": True,
        "sealed_source_sha256_at_collection": verified_sources,
        "full_runtime_source_sha256_at_collection": verified_runtime,
        "runtime_asset_sha256_at_collection": expected_runtime_assets,
    }


def _validate_repository_receipts(
    repository_root: Path | None = None,
) -> dict[str, Any]:
    root = _root() if repository_root is None else Path(repository_root)
    plan, plan_sha = _read_plan(root)
    artifact_contract.validate_sealed_plan(
        root / artifact_contract.VALIDATION_PLAN, require_sealed=True
    )
    _registered_validation_gates(root)
    implementation = _validate_implementation_seal(plan, repository_root=root)
    receipts = {
        "candidate39_result": (
            artifact_contract.CANDIDATE39_RESULT,
            artifact_contract.CANDIDATE39_RESULT_SHA256,
        ),
        "candidate39_report": (
            artifact_contract.CANDIDATE39_REPORT,
            artifact_contract.CANDIDATE39_REPORT_SHA256,
        ),
        "fixed_direction_manifest": (
            artifact_contract.FIXED_DIRECTION_MANIFEST,
            artifact_contract.FIXED_DIRECTION_MANIFEST_SHA256,
        ),
        "fixed_direction_payload": (
            artifact_contract.FIXED_DIRECTION_PATH,
            artifact_contract.FIXED_DIRECTION_SHA256,
        ),
    }
    result: dict[str, Any] = {
        "validation_plan": artifact_contract.VALIDATION_PLAN,
        "validation_plan_sha256": plan_sha,
        "implementation_seal": implementation,
    }
    for label, (relative, expected) in receipts.items():
        digest = _sha256_bytes(_regular_file(root / relative, label).read_bytes())
        if digest != expected:
            raise ValueError(f"Candidate40 {label} receipt changed")
        result[label] = relative
        result[f"{label}_sha256"] = digest
    payload = artifact_contract.load_fixed_direction(
        root / artifact_contract.FIXED_DIRECTION_PATH
    )
    if not torch.equal(payload["fixed_z"], artifact_contract.expected_fixed_z()):
        raise ValueError("Candidate40 fixed direction changed")
    result["fixed_direction_was_recomputed_or_modified"] = False
    result["non_evidence_smoke"] = _validate_registered_smoke(
        plan, repository_root=root
    )
    return result


def _validate_registered_smoke(
    plan: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Require the exact preregistered 334b smoke and all of its safety checks."""

    execution = plan.get("execution")
    smoke = execution.get("non_evidence_smoke") if isinstance(execution, Mapping) else None
    if not isinstance(smoke, Mapping) or (
        smoke.get("seed"), smoke.get("replicate"), smoke.get("num_envs")
    ) != (334, "b", 8):
        raise ValueError("Candidate40 smoke registration changed")
    stem = smoke.get("output")
    if not isinstance(stem, str) or not stem:
        raise ValueError("Candidate40 smoke output registration is missing")
    if (
        f"{stem}.pt" != artifact_contract.SMOKE_ARTIFACT_PATH
        or f"{stem}.json" != artifact_contract.SMOKE_REPORT_PATH
    ):
        raise ValueError("Candidate40 smoke paths differ from the canonical contract")
    artifact_path = _regular_file(repository_root / f"{stem}.pt", "Candidate40 smoke")
    report_path = _regular_file(
        repository_root / f"{stem}.json", "Candidate40 smoke report"
    )
    artifact_raw = artifact_path.read_bytes()
    artifact = artifact_contract.validate_artifact(
        _torch_load_weights_only(artifact_raw), require_sealed_plan=True
    )
    try:
        report = json.loads(report_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError("Candidate40 smoke report is invalid JSON") from error
    checked = artifact_contract.validate_report(
        report, artifact, published=True, require_sealed_plan=True
    )
    artifact_digest = _sha256_bytes(artifact_raw)
    if checked["artifact_sha256"] != artifact_digest:
        raise ValueError("Candidate40 smoke report artifact receipt changed")
    if Path(checked["artifact_output"]).resolve() != artifact_path.resolve():
        raise ValueError("Candidate40 smoke report points at a different artifact")
    metadata = artifact["metadata"]
    if (
        metadata["seed"], metadata["replicate"], metadata["num_envs"]
    ) != (334, "b", 8):
        raise ValueError("Candidate40 smoke artifact identity changed")
    if metadata["runtime_asset_sha256"] != RUNTIME_ASSET_EXPECTED_SHA256:
        raise ValueError("Candidate40 smoke runtime asset receipt changed")
    episodes = artifact["episodes"]
    domain = episodes["treatment"] & episodes["eligible"]
    traced = torch.bincount(
        artifact["steps"]["row_env_slot"], minlength=int(metadata["num_envs"])
    ) > 0
    if not bool((domain & traced).any()):
        raise ValueError("Candidate40 smoke has no treatment-eligible trace")
    if bool(
        (
            domain
            & (
                episodes["dropped"]
                | episodes["unsafe_force"]
                | episodes["pre_release_launch"]
            )
        ).any()
    ):
        raise ValueError("Candidate40 smoke failed treatment safety checks")
    summary = checked["summary"]["action_audit"]
    zero_names = (
        "pre_eligibility_action_violations",
        "hand_invariance_violations",
        "treatment_arm_gate_violations",
        "control_route_violations",
        "task_action_reconstruction_violations",
        "fixed_residual_budget_violations",
        "action_bound_violations",
    )
    if any(summary[name] != 0 for name in zero_names) or summary[
        "any_new_abs_action_ge_0999"
    ] is not False:
        raise ValueError("Candidate40 smoke failed action/clock/budget checks")
    if int(episodes["treatment_pre_enable_nonzero_arm_rows"].sum()) != 0:
        raise ValueError("Candidate40 smoke moved a treatment arm before enablement")
    return {
        "run": "334b",
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": artifact_digest,
        "report": str(report_path.resolve()),
        "report_sha256": _sha256_bytes(report_path.read_bytes()),
        "registered_checks_passed": True,
        "outcomes_used_to_change_plan": False,
    }


def _independent_rank(*, seed: int, num_envs: int) -> torch.Tensor:
    ranked = sorted(
        range(num_envs),
        key=lambda slot: (
            hashlib.sha256(
                f"{verifier.ASSIGNMENT_SALT}\0rank\0{seed}\0{slot}".encode(
                    "utf-8"
                )
            ).digest(),
            slot,
        ),
    )
    result = torch.empty(num_envs, dtype=torch.int64)
    result[torch.tensor(ranked, dtype=torch.int64)] = torch.arange(
        num_envs, dtype=torch.int64
    )
    return result


def _independent_treatment_mask(
    *, seed: int, num_envs: int, replicate: str
) -> torch.Tensor:
    if replicate not in REPLICATES:
        raise ValueError("replicate must be a or b")
    rank = _independent_rank(seed=seed, num_envs=num_envs)
    arm_a = rank < (num_envs // 2)
    return arm_a if replicate == "a" else ~arm_a


def _without_run_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    for name in ("seed", "replicate", "assignment_mask_sha256"):
        normalized.pop(name, None)
    runtime = normalized.get("runtime")
    if isinstance(runtime, Mapping):
        runtime = dict(runtime)
        runtime.pop("seed", None)
        normalized["runtime"] = runtime
    return normalized


def validate_validation_artifacts(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    require_sealed_plan: bool = False,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Validate the exact four runs, provenance identity, and complements."""

    expected_keys = {(seed, replicate) for seed in SEEDS for replicate in REPLICATES}
    if set(artifacts) != expected_keys:
        raise ValueError("Candidate40 evidence must be exactly seeds 335/336 x a/b")
    validated: dict[tuple[int, str], dict[str, Any]] = {}
    reference_identity: dict[str, Any] | None = None
    for seed, replicate in RUN_ORDER:
        artifact = artifact_contract.validate_artifact(
            artifacts[(seed, replicate)], require_sealed_plan=require_sealed_plan
        )
        metadata = artifact["metadata"]
        episodes = artifact["episodes"]
        if (
            metadata["seed"] != seed
            or metadata["replicate"] != replicate
            or metadata["num_envs"] != NUM_ENVS
        ):
            raise ValueError(f"Candidate40 artifact {seed}{replicate} identity changed")
        expected_treatment = _independent_treatment_mask(
            seed=seed, num_envs=NUM_ENVS, replicate=replicate
        )
        expected_rank = _independent_rank(seed=seed, num_envs=NUM_ENVS)
        if not torch.equal(episodes["env_slot"], torch.arange(NUM_ENVS)):
            raise ValueError(f"Candidate40 {seed}{replicate} slots changed")
        if not torch.equal(episodes["treatment"], expected_treatment):
            raise ValueError(f"Candidate40 {seed}{replicate} assignment changed")
        if not torch.equal(episodes["assignment_rank"], expected_rank):
            raise ValueError(f"Candidate40 {seed}{replicate} rank changed")
        if not torch.equal(
            episodes["fixed_z"],
            artifact_contract.expected_fixed_z().expand(NUM_ENVS, -1),
        ):
            raise ValueError(f"Candidate40 {seed}{replicate} fixed_z changed")
        if int(expected_treatment.sum()) != NUM_ENVS // 2:
            raise RuntimeError("Candidate40 assignment is not exactly balanced")
        identity = _without_run_identity(metadata)
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise ValueError(
                "Candidate40 checkpoint, source, runtime, plan, or route identity differs"
            )
        validated[(seed, replicate)] = artifact
    for seed in SEEDS:
        artifact_contract.validate_complementary_artifacts(
            validated[(seed, "a")],
            validated[(seed, "b")],
            require_sealed_plan=require_sealed_plan,
        )
        a = validated[(seed, "a")]["episodes"]
        b = validated[(seed, "b")]["episodes"]
        if not torch.equal(a["treatment"], ~b["treatment"]):
            raise ValueError(f"Candidate40 seed {seed} assignments are not complements")
        for field in ("env_slot", "assignment_rank", "fixed_z"):
            if not torch.equal(a[field], b[field]):
                raise ValueError(f"Candidate40 seed {seed} cross-replicate {field} changed")
        # Eligibility masks/clocks and physical trajectories are intentionally
        # not required to match across independent GPU simulations.
    return validated


def _validate_smoke_bindings(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    smoke_receipt: Mapping[str, Any],
) -> None:
    """Bind every development run to the exact preregistered smoke pair."""

    artifact_sha = smoke_receipt.get("artifact_sha256")
    report_sha = smoke_receipt.get("report_sha256")
    for label, digest in (
        ("smoke artifact", artifact_sha),
        ("smoke report", report_sha),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid Candidate40 {label} receipt")
    for seed, replicate in RUN_ORDER:
        metadata = artifacts[(seed, replicate)]["metadata"]
        if (
            metadata["smoke_artifact_sha256"] != artifact_sha
            or metadata["smoke_report_sha256"] != report_sha
        ):
            raise ValueError(
                f"Candidate40 {seed}{replicate} was not collected after the "
                "registered smoke pair"
            )


def _read_artifact(path: Path, *, require_sealed_plan: bool) -> tuple[dict[str, Any], str]:
    path = _regular_file(path, "Candidate40 artifact")
    raw = path.read_bytes()
    payload = _torch_load_weights_only(raw)
    return (
        artifact_contract.validate_artifact(
            payload, require_sealed_plan=require_sealed_plan
        ),
        _sha256_bytes(raw),
    )


def load_fixed_evidence(
    repository_root: Path | None = None,
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    root = _root() if repository_root is None else Path(repository_root)
    fixed_receipt = _validate_repository_receipts(root)
    artifacts: dict[tuple[int, str], dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for seed, replicate in RUN_ORDER:
        path = root / INPUT_PATTERN.format(seed=seed, replicate=replicate)
        artifact, digest = _read_artifact(path, require_sealed_plan=True)
        artifacts[(seed, replicate)] = artifact
        receipts.append(
            {
                "run": f"{seed}{replicate}",
                "artifact": str(path.resolve()),
                "artifact_sha256": digest,
            }
        )
    validated = validate_validation_artifacts(artifacts, require_sealed_plan=True)
    _validate_smoke_bindings(validated, fixed_receipt["non_evidence_smoke"])
    _validate_collection_commit_provenance(
        validated,
        fixed_receipt["implementation_seal"],
        repository_root=root,
    )
    return validated, receipts


def stable_transport_outcomes(
    artifact: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Recompute registered utilities from reset-before transition truth."""

    value = artifact_contract.validate_artifact(
        artifact, require_sealed_plan=False
    )
    episodes = value["episodes"]
    steps = value["steps"]
    count = int(episodes["env_slot"].numel())
    utility = torch.zeros(count, dtype=torch.float64)
    stable5 = torch.zeros(count, dtype=torch.bool)
    observed_values = torch.zeros((count, UTILITY_HORIZON), dtype=torch.float64)
    stable_values = torch.zeros_like(observed_values)
    observed_mask = torch.zeros((count, UTILITY_HORIZON), dtype=torch.bool)

    for slot in range(count):
        if not bool(episodes["eligible"][slot]):
            continue
        mask = steps["row_env_slot"] == slot
        ages = steps["row_eligible_age"][mask]
        if ages.numel() == 0:
            raise ValueError(f"eligible Candidate40 slot {slot} has no trace")
        within = ages < UTILITY_HORIZON
        ages = ages[within]
        if ages.numel() and not torch.equal(
            ages, torch.arange(int(ages.numel()), dtype=torch.int64)
        ):
            raise ValueError(f"Candidate40 slot {slot} utility trace has a gap")
        transition_latch = steps["row_transition_public_latch"][mask][within]
        grasp_quality = steps["row_transition_grasp_quality"][mask][within]
        hold_quality = steps["row_transition_hold_quality"][mask][within]
        force = steps["row_transition_max_force_n"][mask][within]
        clearance = steps["row_transition_true_clearance_m"][mask][within]
        valid = (
            transition_latch
            & (grasp_quality >= 0.35)
            & (hold_quality >= 0.50)
            & (force <= 30.0)
        )
        x = valid.to(torch.float64) * torch.clamp(
            clearance.to(torch.float64) / 0.20, 0.0, 1.0
        )
        n = int(ages.numel())
        if n:
            observed_values[slot, :n] = x
            observed_mask[slot, :n] = True
        for end in range(STABLE_WINDOW - 1, n):
            stable_values[slot, end] = x[
                end - STABLE_WINDOW + 1 : end + 1
            ].min()
        if bool(episodes["success"][slot]):
            terminal_age = int(
                episodes["terminal_step"][slot]
                - episodes["first_eligible_step"][slot]
            )
            if terminal_age < 0:
                raise ValueError("Candidate40 success precedes eligibility")
            suffix_start = min(UTILITY_HORIZON, terminal_age + 1)
            stable_values[slot, suffix_start:] = 1.0
        utility[slot] = stable_values[slot].mean()
        stable5[slot] = bool(
            (stable_values[slot, :STABLE_5CM_ACTION_HORIZON] >= STABLE_5CM_FRACTION).any()
        )
    return {
        "stable_transport_restricted_mean": utility,
        "stable_held_5cm_by_action96": stable5,
        "never_stable_held_5cm_by_action96": episodes["eligible"] & (~stable5),
        "observed_transition_value": observed_values,
        "stable_transition_value": stable_values,
        "observed_mask": observed_mask,
    }


def _outcome_table(artifact: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    episodes = artifact["episodes"]
    stable = stable_transport_outcomes(artifact)
    terminal_utility = (
        episodes["success"].to(torch.float64)
        - episodes["unlatched_clearance_ge_5cm"].to(torch.float64)
    )
    return {
        "stable_transport_restricted_mean": stable[
            "stable_transport_restricted_mean"
        ],
        "terminal_utility": terminal_utility,
        "success": episodes["success"],
        "ever_true_clearance_ge_20cm": episodes["ever_clearance_ge_20cm"],
        "stable_held_5cm_by_action96": stable["stable_held_5cm_by_action96"],
        "never_stable_held_5cm_by_action96": stable[
            "never_stable_held_5cm_by_action96"
        ],
        "ever_grasped": episodes["ever_grasped"],
        "unlatched_clearance_ge_5cm": episodes[
            "unlatched_clearance_ge_5cm"
        ],
    }


def _ht_outcome(
    treatment: torch.Tensor, domain: torch.Tensor, outcome: torch.Tensor
) -> dict[str, float | int]:
    if outcome.dtype == torch.bool:
        numeric = outcome.to(torch.float64)
    elif outcome.dtype.is_floating_point:
        numeric = outcome.to(torch.float64)
    else:
        raise TypeError("HT outcome must be bool or floating point")
    if numeric.shape != domain.shape or treatment.shape != domain.shape:
        raise ValueError("HT treatment/domain/outcome shapes differ")
    if not bool(torch.isfinite(numeric).all()):
        raise FloatingPointError("HT outcome contains NaN or infinity")
    denominator = int(domain.sum())
    treatment_rows = int((treatment & domain).sum())
    control_rows = int(((~treatment) & domain).sum())
    if denominator <= 0 or treatment_rows <= 0 or control_rows <= 0:
        raise ValueError("a Candidate40 seed lacks eligible support in one arm")
    treatment_sum = float(numeric[treatment & domain].sum())
    control_sum = float(numeric[(~treatment) & domain].sum())
    treatment_estimate = treatment_sum / (KNOWN_PROPENSITY * denominator)
    control_estimate = control_sum / ((1.0 - KNOWN_PROPENSITY) * denominator)
    return {
        "known_propensity": KNOWN_PROPENSITY,
        "domain_rows": denominator,
        "treatment_rows": treatment_rows,
        "control_rows": control_rows,
        "treatment_sum": treatment_sum,
        "control_sum": control_sum,
        "treatment": treatment_estimate,
        "control": control_estimate,
        "delta": treatment_estimate - control_estimate,
    }


def _raw_arm_funnel(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]], *, treatment_arm: bool
) -> dict[str, Any]:
    assigned = 0
    eligible = 0
    sums = {name: 0.0 for name in EVENT_FIELDS}
    for seed in SEEDS:
        for replicate in REPLICATES:
            artifact = artifacts[(seed, replicate)]
            episodes = artifact["episodes"]
            outcomes = _outcome_table(artifact)
            arm = episodes["treatment"] if treatment_arm else ~episodes["treatment"]
            domain = arm & episodes["eligible"]
            assigned += int(arm.sum())
            eligible += int(domain.sum())
            for name, outcome in outcomes.items():
                sums[name] += float(outcome.to(torch.float64)[domain].sum())
    return {
        "assigned": assigned,
        "eligible": eligible,
        "eligibility_rate_observed": eligible / assigned if assigned else None,
        **{
            name: {
                "sum": value,
                "mean_given_eligibility_observed": value / eligible
                if eligible
                else None,
            }
            for name, value in sums.items()
        },
    }


def _paired_common_eligibility(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    total_common = 0
    aggregate = {
        name: {"treatment_sum": 0.0, "control_sum": 0.0}
        for name in EVENT_FIELDS
    }
    for seed in SEEDS:
        a = artifacts[(seed, "a")]
        b = artifacts[(seed, "b")]
        ea, eb = a["episodes"], b["episodes"]
        oa, ob = _outcome_table(a), _outcome_table(b)
        common = ea["eligible"] & eb["eligible"]
        treatment_is_a = ea["treatment"]
        count = int(common.sum())
        events: dict[str, Any] = {}
        for name in EVENT_FIELDS:
            treatment_outcome = torch.where(treatment_is_a, oa[name], ob[name])
            control_outcome = torch.where(treatment_is_a, ob[name], oa[name])
            treatment_sum = float(
                treatment_outcome.to(torch.float64)[common].sum()
            )
            control_sum = float(control_outcome.to(torch.float64)[common].sum())
            aggregate[name]["treatment_sum"] += treatment_sum
            aggregate[name]["control_sum"] += control_sum
            events[name] = {
                "treatment_sum": treatment_sum,
                "control_sum": control_sum,
                "treatment_mean": treatment_sum / count if count else None,
                "control_mean": control_sum / count if count else None,
                "paired_delta": (treatment_sum - control_sum) / count
                if count
                else None,
            }
        per_seed[str(seed)] = {
            "eligible_in_both_replicates": count,
            "eligibility_mask_discordant_slots": int(
                (ea["eligible"] ^ eb["eligible"]).sum()
            ),
            "events": events,
        }
        total_common += count
    aggregate_events = {
        name: {
            **sums,
            "treatment_mean": sums["treatment_sum"] / total_common
            if total_common
            else None,
            "control_mean": sums["control_sum"] / total_common
            if total_common
            else None,
            "paired_delta": (
                sums["treatment_sum"] - sums["control_sum"]
            )
            / total_common
            if total_common
            else None,
        }
        for name, sums in aggregate.items()
    }
    return {
        "role": (
            "consistency diagnostic for every outcome except the explicitly "
            "registered paired unlatched-5cm acceptance gate; never a "
            "replacement for the HT primary"
        ),
        "domain": "same seed/env_slot eligible before Candidate40 action in both replicates",
        "eligibility_masks_and_clocks_required_bit_exact": False,
        "common_eligible_slots": total_common,
        "minimum_common_eligible_slots_per_seed": min(
            row["eligible_in_both_replicates"] for row in per_seed.values()
        ),
        "per_seed": per_seed,
        "aggregate_events": aggregate_events,
    }


def _action_clock_audit(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]]
) -> dict[str, Any]:
    sums = {
        "pre_eligibility_action_violations": 0,
        "hand_invariance_violations": 0,
        "treatment_arm_gate_violations": 0,
        "control_route_violations": 0,
        "task_action_reconstruction_violations": 0,
        "fixed_residual_budget_violations": 0,
        "action_bound_violations": 0,
        "treatment_pre_enable_nonzero_arm_rows": 0,
    }
    any_new_0999 = False
    per_run: list[dict[str, Any]] = []
    for seed, replicate in RUN_ORDER:
        artifact = artifacts[(seed, replicate)]
        episodes = artifact["episodes"]
        steps = artifact["steps"]
        row = {
            name: int(episodes[name].sum())
            for name in sums
        }
        row["any_new_abs_action_ge_0999"] = bool(
            episodes["new_abs_action_ge_0999"].any()
        )
        row["run"] = f"{seed}{replicate}"
        pre_enable = (
            steps["row_treatment"]
            & steps["row_pre_enable"]
            & (steps["row_requested_action"][:, : artifact_contract.ARM_ACTION_DIM] != 0).any(dim=-1)
        )
        if int(pre_enable.sum()) != row["treatment_pre_enable_nonzero_arm_rows"]:
            raise ValueError("online and trace pre-enable arm audits disagree")
        pre = artifact["pre_steps"]
        pre_newly_saturated = (
            (pre["pre_requested_action"].abs() >= 0.999)
            & (pre["pre_baseline_action"].abs() < 0.999)
        )
        common = steps["row_common_action"]
        requested = steps["row_requested_action"]
        newly_saturated = (requested.abs() >= 0.999) & (common.abs() < 0.999)
        derived_new_saturation = bool(pre_newly_saturated.any()) or bool(
            newly_saturated.any()
        )
        if derived_new_saturation != row["any_new_abs_action_ge_0999"]:
            raise ValueError("online and trace action saturation audits disagree")
        for name in sums:
            sums[name] += int(row[name])
        any_new_0999 |= bool(row["any_new_abs_action_ge_0999"])
        per_run.append(row)
    exact = {
        "pre_eligibility_action_and_state_machine_parity_exact": sums[
            "pre_eligibility_action_violations"
        ]
        == 0,
        "hand14_invariance_exact": sums["hand_invariance_violations"] == 0,
        "control_candidate39_route_exact": sums["control_route_violations"] == 0,
        "verified_arm_clock_gate_and_live_relock_exact": (
            sums["treatment_arm_gate_violations"] == 0
            and sums["task_action_reconstruction_violations"] == 0
            and sums["action_bound_violations"] == 0
        ),
    }
    return {
        "per_run": per_run,
        "aggregate": {**sums, "any_new_abs_action_ge_0999": any_new_0999, **exact},
    }


def _gate(value: Any, threshold: Any, comparison: str, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "pass": bool(passed),
    }


def _build_gate_results(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    t = GATE_THRESHOLDS
    specs = {
        "any_new_abs_action_ge_0999": ("any_new_0999", "==", lambda v, x: v is x),
        "checkpoint_manifest_assignment_and_source_identity_exact": ("identity_exact", "==", lambda v, x: v is x),
        "conditional_ever_grasped_delta_floor": ("ever_grasped_delta", ">=", lambda v, x: v >= x),
        "conditional_never_stable_held_5cm_by_action96_delta_max": ("never_lift_delta", "<=", lambda v, x: v <= x),
        "conditional_success_delta_floor": ("success_delta", ">=", lambda v, x: v >= x),
        "conditional_true_clearance_ge_20cm_delta_floor": ("true20_delta", ">=", lambda v, x: v >= x),
        "conditional_unlatched_clearance_ge_5cm_delta_max": ("unlatched5_delta", "<=", lambda v, x: v <= x),
        "control_candidate39_route_exact": ("control_exact", "==", lambda v, x: v is x),
        "eligible_opportunity_slots_min_across_validation": ("eligible_total", ">=", lambda v, x: v >= x),
        "eligible_opportunity_slots_min_per_seed": ("eligible_min_seed", ">=", lambda v, x: v >= x),
        "eligible_opportunity_slots_min_per_seed_arm": ("eligible_min_seed_arm", ">=", lambda v, x: v >= x),
        "fixed_residual_component_and_l2_budget_violations": ("fixed_budget_violations", "==", lambda v, x: v == x),
        "hand14_invariance_exact": ("hand_exact", "==", lambda v, x: v is x),
        "paired_common_eligibility_unlatched_clearance_ge_5cm_delta_max": ("paired_unlatched5_delta", "<=", lambda v, x: v <= x),
        "paired_common_eligible_slots_min_across_validation": ("paired_common_total", ">=", lambda v, x: v >= x),
        "paired_common_eligible_slots_min_per_seed": ("paired_common_min_seed", ">=", lambda v, x: v >= x),
        "per_seed_primary_delta_floor": ("minimum_primary_seed_delta", ">=", lambda v, x: v >= x),
        "pooled_option_active_pre_eligible_unlatched_clearance_ge_5cm_max": ("pooled_preeligible_unlatched5", "<=", lambda v, x: v <= x),
        "positive_primary_delta_seeds_min": ("positive_primary_seeds", ">=", lambda v, x: v >= x),
        "pre_eligibility_action_and_state_machine_parity_exact": ("preeligibility_exact", "==", lambda v, x: v is x),
        "primary_conditional_stable_transport_restricted_mean_delta_min_exclusive": ("primary_delta", ">", lambda v, x: v > x),
        "treatment_eligible_dropped_max": ("treatment_dropped", "<=", lambda v, x: v <= x),
        "treatment_eligible_first_arm_releases_by_action32_min_across_validation": ("treatment_release32_total", ">=", lambda v, x: v >= x),
        "treatment_eligible_first_arm_releases_by_action32_min_per_seed": ("treatment_release32_min_seed", ">=", lambda v, x: v >= x),
        "treatment_eligible_post_latch_unlatched_clearance_ge_5cm_max": ("treatment_unlatched5", "<=", lambda v, x: v <= x),
        "treatment_eligible_pre_release_launch_max": ("treatment_pre_release_launch", "<=", lambda v, x: v <= x),
        "treatment_eligible_stable_held_5cm_by_action96_min_across_validation": ("treatment_stable5_total", ">=", lambda v, x: v >= x),
        "treatment_eligible_stable_held_5cm_by_action96_min_per_seed": ("treatment_stable5_min_seed", ">=", lambda v, x: v >= x),
        "treatment_eligible_unsafe_force_max": ("treatment_unsafe", "<=", lambda v, x: v <= x),
        "treatment_pre_enable_nonzero_arm_rows_max": ("preenable_nonzero_arm_rows", "<=", lambda v, x: v <= x),
        "verified_arm_clock_gate_and_live_relock_exact": ("clock_exact", "==", lambda v, x: v is x),
    }
    if set(specs) != set(GATE_THRESHOLDS):
        raise RuntimeError("not every Candidate40 gate has executable semantics")
    result = {}
    for name, (metric, comparison, predicate) in specs.items():
        value = metrics[metric]
        threshold = t[name]
        result[name] = _gate(value, threshold, comparison, predicate(value, threshold))
    return result


def compute_validation(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    require_sealed_plan: bool = False,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    artifacts = validate_validation_artifacts(
        artifacts, require_sealed_plan=require_sealed_plan
    )
    fixed_receipt = (
        _validate_repository_receipts(repository_root)
        if require_sealed_plan
        else {
            "validation_plan": artifact_contract.VALIDATION_PLAN,
            "validation_plan_sha256": _read_plan(repository_root)[1],
            "fixed_direction_was_recomputed_or_modified": False,
        }
    )
    if require_sealed_plan:
        _validate_smoke_bindings(
            artifacts, fixed_receipt["non_evidence_smoke"]
        )
        fixed_receipt["collection_provenance"] = (
            _validate_collection_commit_provenance(
                artifacts,
                fixed_receipt["implementation_seal"],
                repository_root=(
                    _root() if repository_root is None else Path(repository_root)
                ),
            )
        )

    outcomes = {key: _outcome_table(value) for key, value in artifacts.items()}
    per_seed_ht: dict[str, Any] = {}
    per_seed_eligibility: dict[str, Any] = {}
    for seed in SEEDS:
        treatment = torch.stack(
            [artifacts[(seed, replicate)]["episodes"]["treatment"] for replicate in REPLICATES]
        )
        eligible = torch.stack(
            [artifacts[(seed, replicate)]["episodes"]["eligible"] for replicate in REPLICATES]
        )
        per_seed_ht[str(seed)] = {
            name: _ht_outcome(
                treatment,
                eligible,
                torch.stack([outcomes[(seed, replicate)][name] for replicate in REPLICATES]),
            )
            for name in EVENT_FIELDS
        }
        per_seed_eligibility[str(seed)] = {
            "eligible_rows": int(eligible.sum()),
            "treatment_eligible_rows": int((treatment & eligible).sum()),
            "control_eligible_rows": int(((~treatment) & eligible).sum()),
        }
    equal_seed_ht = {
        name: {
            quantity: sum(
                per_seed_ht[str(seed)][name][quantity] for seed in SEEDS
            )
            / len(SEEDS)
            for quantity in ("treatment", "control", "delta")
        }
        for name in EVENT_FIELDS
    }
    primary_by_seed = {
        str(seed): per_seed_ht[str(seed)]["stable_transport_restricted_mean"]["delta"]
        for seed in SEEDS
    }
    paired = _paired_common_eligibility(artifacts)
    action_audit = _action_clock_audit(artifacts)
    audit = action_audit["aggregate"]

    treatment_dropped = 0
    treatment_unsafe = 0
    treatment_unlatched5 = 0
    treatment_pre_release_launch = 0
    pooled_preeligible_unlatched5 = 0
    release32_by_seed: dict[str, int] = {}
    stable5_by_seed: dict[str, int] = {}
    for seed in SEEDS:
        releases = 0
        stable5_count = 0
        for replicate in REPLICATES:
            artifact = artifacts[(seed, replicate)]
            episode = artifact["episodes"]
            stable = outcomes[(seed, replicate)]["stable_held_5cm_by_action96"]
            domain = episode["treatment"] & episode["eligible"]
            treatment_dropped += int((domain & episode["dropped"]).sum())
            treatment_unsafe += int((domain & episode["unsafe_force"]).sum())
            treatment_unlatched5 += int(
                (domain & episode["unlatched_clearance_ge_5cm"]).sum()
            )
            treatment_pre_release_launch += int(
                (domain & episode["pre_release_launch"]).sum()
            )
            pooled_preeligible_unlatched5 += int(
                episode[
                    "option_active_pre_eligible_unlatched_clearance_ge_5cm"
                ].sum()
            )
            release_age = episode["first_arm_enabled_step"] - episode["first_eligible_step"]
            releases += int(
                (domain & (release_age >= 0) & (release_age < 32)).sum()
            )
            stable5_count += int((domain & stable).sum())
        release32_by_seed[str(seed)] = releases
        stable5_by_seed[str(seed)] = stable5_count

    eligible_counts = [row["eligible_rows"] for row in per_seed_eligibility.values()]
    eligible_arm_counts = [
        row[name]
        for row in per_seed_eligibility.values()
        for name in ("treatment_eligible_rows", "control_eligible_rows")
    ]
    paired_unlatched5 = paired["aggregate_events"][
        "unlatched_clearance_ge_5cm"
    ]["paired_delta"]
    # Missing common support must produce a deterministic failed report, not a
    # TypeError in the numeric comparator.  One is the worst possible binary
    # treatment-minus-control risk contrast and therefore also fails the
    # registered <= 0 gate independently of the common-slot power gates.
    paired_unlatched5_for_gate = (
        1.0 if paired_unlatched5 is None else paired_unlatched5
    )
    metrics = {
        "primary_delta": equal_seed_ht["stable_transport_restricted_mean"]["delta"],
        "positive_primary_seeds": sum(value > 0.0 for value in primary_by_seed.values()),
        "minimum_primary_seed_delta": min(primary_by_seed.values()),
        "ever_grasped_delta": equal_seed_ht["ever_grasped"]["delta"],
        "never_lift_delta": equal_seed_ht[
            "never_stable_held_5cm_by_action96"
        ]["delta"],
        "success_delta": equal_seed_ht["success"]["delta"],
        "true20_delta": equal_seed_ht["ever_true_clearance_ge_20cm"]["delta"],
        "unlatched5_delta": equal_seed_ht["unlatched_clearance_ge_5cm"]["delta"],
        "eligible_total": sum(eligible_counts),
        "eligible_min_seed": min(eligible_counts),
        "eligible_min_seed_arm": min(eligible_arm_counts),
        "paired_common_total": paired["common_eligible_slots"],
        "paired_common_min_seed": paired["minimum_common_eligible_slots_per_seed"],
        "paired_unlatched5_delta": paired_unlatched5_for_gate,
        "treatment_dropped": treatment_dropped,
        "treatment_unsafe": treatment_unsafe,
        "treatment_unlatched5": treatment_unlatched5,
        "treatment_pre_release_launch": treatment_pre_release_launch,
        "pooled_preeligible_unlatched5": pooled_preeligible_unlatched5,
        "treatment_release32_total": sum(release32_by_seed.values()),
        "treatment_release32_min_seed": min(release32_by_seed.values()),
        "treatment_stable5_total": sum(stable5_by_seed.values()),
        "treatment_stable5_min_seed": min(stable5_by_seed.values()),
        "preenable_nonzero_arm_rows": audit[
            "treatment_pre_enable_nonzero_arm_rows"
        ],
        "fixed_budget_violations": audit["fixed_residual_budget_violations"],
        "any_new_0999": audit["any_new_abs_action_ge_0999"],
        "preeligibility_exact": audit[
            "pre_eligibility_action_and_state_machine_parity_exact"
        ],
        "hand_exact": audit["hand14_invariance_exact"],
        "control_exact": audit["control_candidate39_route_exact"],
        "clock_exact": audit["verified_arm_clock_gate_and_live_relock_exact"],
        "identity_exact": True,
    }
    gates = _build_gate_results(metrics)
    all_pass = all(row["pass"] for row in gates.values())
    report = {
        "kind": REPORT_KIND,
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "development_only": True,
        "fixed_evidence_receipt": fixed_receipt,
        "registered_development_validation_gates": {
            "all_must_pass": True,
            **GATE_THRESHOLDS,
        },
        "causal_estimand_note": (
            "All HT effects use the first option-active latch-visible pre-action "
            "eligibility domain with known propensity 0.5. Verification and arm "
            "enablement are post-treatment mediators and never select this domain."
        ),
        "utility_contract": {
            "horizon_actions": UTILITY_HORIZON,
            "stable_window_actions": STABLE_WINDOW,
            "success_suffix_value": 1.0,
            "other_terminal_suffix_value": 0.0,
            "true_clearance_normalizer_m": 0.20,
            "stable_held_5cm_action_horizon": STABLE_5CM_ACTION_HORIZON,
        },
        "raw_funnel": {
            "treatment": _raw_arm_funnel(artifacts, treatment_arm=True),
            "control": _raw_arm_funnel(artifacts, treatment_arm=False),
        },
        "eligibility_support": per_seed_eligibility,
        "per_seed_horvitz_thompson": per_seed_ht,
        "equal_seed_horvitz_thompson": equal_seed_ht,
        "primary": {
            "name": "conditional_stable_transport_restricted_mean_delta",
            "known_propensity": KNOWN_PROPENSITY,
            "per_seed_delta": primary_by_seed,
            "equal_seed_delta": metrics["primary_delta"],
            "positive_delta_seeds": metrics["positive_primary_seeds"],
        },
        "paired_common_eligibility_sensitivity": paired,
        "action_and_clock_audit": action_audit,
        "treatment_counts": {
            "eligible_dropped": treatment_dropped,
            "eligible_unsafe_force": treatment_unsafe,
            "eligible_post_latch_unlatched_clearance_ge_5cm": treatment_unlatched5,
            "eligible_pre_release_launch": treatment_pre_release_launch,
            "first_arm_releases_by_action32": {
                "total": sum(release32_by_seed.values()),
                "per_seed": release32_by_seed,
            },
            "stable_held_5cm_by_action96": {
                "total": sum(stable5_by_seed.values()),
                "per_seed": stable5_by_seed,
            },
        },
        "pooled_option_active_pre_eligible_unlatched_clearance_ge_5cm": pooled_preeligible_unlatched5,
        "gates": gates,
        "all_gates_pass": all_pass,
        "decision": "advance_to_formal_screen" if all_pass else "reject_candidate40",
        "arm_enabled_was_used_as_causal_domain": False,
        "network_updates": 0,
    }
    _strict_json_bytes(report)
    return report


def analyze_fixed_evidence(
    repository_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifacts, receipts = load_fixed_evidence(repository_root)
    report = compute_validation(
        artifacts, require_sealed_plan=True, repository_root=repository_root
    )
    report["input_artifact_receipts"] = receipts
    _strict_json_bytes(report)
    return report, receipts


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    """Atomically publish strict JSON while refusing any existing destination."""

    output = Path(os.path.abspath(os.fspath(output)))
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"refusing to overwrite Candidate40 report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = _strict_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report-output", type=Path, default=Path(REPORT_OUTPUT))
    args = parser.parse_args()
    report, _ = analyze_fixed_evidence()
    output = args.report_output
    if not output.is_absolute():
        output = _root() / output
    publish_json_no_clobber(report, output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
