#!/usr/bin/env python3
"""Strict, simulation-free loader for public-route factual gate evidence.

The collector deliberately writes one immutable artifact/report pair per
cohort, seed and complementary replicate.  This module is the only supported
way to turn those files into rows used by the public-route gate analysis.  It
fails closed on missing pairs, non-canonical paths, receipt drift, row loss,
assignment drift and collection-underpowering.

The returned dataset contains only tensors and plain Python values supported
by ``torch.load(..., weights_only=True)``.  Pilot evidence is intentionally not
loadable through the cohort API.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

import torch

from public_route_trial_contract import (
    FEATURE_DIM,
    FEATURE_CONTRACT,
    READINESS_MIN_SCORE,
    STRATUM_CLOSE_THRESHOLD,
    STRATUM_G_THRESHOLD,
    assignment_for,
    validate_trial_artifact,
)


DEFAULT_MANIFEST = Path(__file__).with_name("public_route_trial_manifest.json")
DEFAULT_ANALYSIS_PLAN = Path(__file__).with_name("public_route_gate_analysis_plan.json")

DATASET_KIND = "pick_tool_public_route_factual_dataset_v1"
DATASET_FORMAT_VERSION = 1
LEDGER_KIND = "pick_tool_public_route_evidence_ledger_v1"
LEDGER_FORMAT_VERSION = 1
COLLECTOR_REPORT_KIND = "pick_tool_public_route_trial_collection_report_v2"
TRIAL_MANIFEST_KIND = "pick_tool_public_route_trial_manifest_v2"
ANALYSIS_PLAN_KIND = "pick_tool_public_route_gate_analysis_plan_v1"

REPLICATES = ("a", "b")
REPLICATE_INDEX = {"a": 0, "b": 1}
ALLOWED_COHORTS = ("train", "development")
TENSOR_FIELDS = (
    "feature",
    "treatment_route",
    "success",
    "dropped",
    "unsafe_force",
    "seed",
    "env_slot",
    "replicate_index",
    "candidate_step",
    "stratum",
)
BOOL_TENSOR_FIELDS = ("treatment_route", "success", "dropped", "unsafe_force")
LONG_TENSOR_FIELDS = ("seed", "env_slot", "replicate_index", "candidate_step", "stratum")

_REPORT_FIELDS = {
    "all_assigned_continue_outcomes",
    "all_assigned_route_outcomes",
    "artifact_sha256",
    "assignment_continue_slots",
    "assignment_route_slots",
    "candidate_continue_outcomes",
    "candidate_continue_rows",
    "candidate_route_outcomes",
    "candidate_route_rows",
    "candidate_rows",
    "claim_boundary",
    "cohort",
    "git",
    "kind",
    "manifest_sha256",
    "num_envs",
    "replicate",
    "runtime_asset_sha256",
    "seed",
    "source_sha256",
    "status",
    "untriggered_episodes",
    "vector_steps",
}
_SUMMARY_FIELDS = {
    "episodes",
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "ever_grasped",
    "ever_clearance_ge_5cm",
    "max_true_clearance_m",
}
_RECEIPT_FIELDS = {
    "cohort",
    "seed",
    "replicate",
    "replicate_index",
    "registered_cohort_run_index",
    "num_envs",
    "candidate_rows",
    "candidate_route_rows",
    "candidate_continue_rows",
    "artifact_path",
    "artifact_sha256",
    "artifact_semantic_sha256",
    "assignment_route_sha256",
    "report_path",
    "report_sha256",
}
_DATASET_METADATA_FIELDS = {
    "cohort",
    "manifest_sha256",
    "analysis_plan_sha256",
    "assignment_salt",
    "feature_contract",
    "feature_dim",
    "seeds",
    "replicates",
    "canonical_row_order",
    "factual_events",
    "factual_rows_only",
    "common_collection_receipt",
}
_LEDGER_FIELDS = {
    "kind",
    "format_version",
    "status",
    "cohort",
    "manifest_sha256",
    "analysis_plan_sha256",
    "dataset_kind",
    "dataset_format_version",
    "dataset_semantic_sha256",
    "input_bundle_sha256",
    "row_count",
    "expected_runs",
    "canonical_row_order",
    "collection_seal_tag",
    "analysis_seal_tag",
    "ledger_tag",
    "acceptance",
    "receipts",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_regular(path: Path, label: str) -> tuple[dict[str, Any], str]:
    source = Path(os.path.abspath(os.fspath(path)))
    raw = _snapshot_regular_bytes(source, label)
    digest_before = hashlib.sha256(raw).hexdigest()
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{label} root must be a JSON object")
    return payload, digest_before


def _snapshot_regular_bytes(path: Path, label: str) -> bytes:
    """Read one inode through one no-follow fd and reject a path race."""

    path = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError) as error:
        raise FileNotFoundError(
            f"{label} must be a regular non-symlink file: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FileNotFoundError(f"{label} must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"{label} changed while its snapshot was read")
    finally:
        os.close(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} path disappeared after its snapshot") from error
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        raise RuntimeError(f"{label} path changed during its snapshot")
    return b"".join(chunks)


def _plain_json(value: Any, *, label: str) -> Any:
    """Return a detached strict-JSON value and reject exotic subclasses."""

    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    decoded = json.loads(
        encoded,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(decoded, (dict, list, str, int, float, bool, type(None))):
        raise TypeError(f"{label} is not plain JSON")
    return decoded


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    normalized = _plain_json(payload, label="evidence ledger")
    if not isinstance(normalized, dict):
        raise TypeError("evidence ledger must be a mapping")
    return (
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _relative_output(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    root = root.resolve()
    candidate = Path(os.path.abspath(os.fspath(root / value)))
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return candidate


def _require_nonsymlink_chain(path: Path, root: Path, label: str) -> None:
    """Reject a symlink at the leaf or any existing component below root."""

    root = Path(os.path.abspath(os.fspath(root)))
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} is outside the repository") from error
    current = root
    if current.is_symlink():
        raise ValueError(f"{label} repository root must not be a symlink")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {current}")


def _require_regular_canonical(path: Path, root: Path, label: str) -> Path:
    _require_nonsymlink_chain(path, root, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    return path


def load_analysis_plan(
    path: Path = DEFAULT_ANALYSIS_PLAN,
    *,
    require_preregistered: bool = True,
) -> dict[str, Any]:
    """Load and validate the immutable analysis choices.

    ``require_preregistered=False`` exists only so the simulation-free tests
    can exercise the draft-to-seal lifecycle.  Dataset publication always
    calls this function with the default, fail-closed setting.
    """

    plan, digest = _load_json_regular(path, "analysis plan")
    expected_top = {
        "data",
        "development",
        "format_version",
        "inference",
        "kind",
        "model",
        "normalization",
        "optimization",
        "outputs",
        "preregistration",
        "randomization_inference",
        "status",
    }
    if set(plan) != expected_top:
        raise ValueError("analysis plan top-level schema is not exact")
    if plan.get("kind") != ANALYSIS_PLAN_KIND or plan.get("format_version") != 1:
        raise ValueError("unsupported analysis plan")
    allowed_status = {"preregistered"} if require_preregistered else {
        "draft_until_seal_commit",
        "preregistered",
    }
    if plan.get("status") not in allowed_status:
        raise ValueError("analysis plan is not preregistered")

    data = plan.get("data")
    expected_data_fields = {
        "allowed_fit_cohort",
        "development_use",
        "factual_events",
        "factual_rows_only",
        "loader_checks",
        "pilot_use",
        "replicates",
        "semantic_hash_fields",
        "train_seeds",
    }
    if not isinstance(data, dict) or set(data) != expected_data_fields:
        raise ValueError("analysis data contract is not exact")
    if (
        data.get("allowed_fit_cohort") != "train"
        or data.get("factual_events") != ["success", "dropped", "unsafe_force"]
        or data.get("factual_rows_only") is not True
        or data.get("replicates") != ["a", "b"]
        or data.get("semantic_hash_fields")
        != [
            "feature",
            "factual_treatment_route",
            "outcome_success",
            "outcome_dropped",
            "outcome_unsafe_force",
            "seed",
            "env_slot",
            "replicate_index",
            "candidate_step",
            "stratum",
            "artifact_sha256_receipts",
            "report_sha256_receipts",
        ]
    ):
        raise ValueError("analysis factual-data contract drifted")
    train_seeds = data.get("train_seeds")
    if (
        not isinstance(train_seeds, list)
        or not train_seeds
        or any(type(seed) is not int for seed in train_seeds)
        or len(set(train_seeds)) != len(train_seeds)
    ):
        raise ValueError("analysis train seeds are invalid")

    development = plan.get("development")
    if not isinstance(development, dict) or set(development) != {
        "acceptance",
        "blind_rule",
        "doubly_robust_nuisance",
        "estimators",
        "seeds",
    }:
        raise ValueError("analysis development contract is not exact")
    development_seeds = development.get("seeds")
    if (
        not isinstance(development_seeds, list)
        or not development_seeds
        or any(type(seed) is not int for seed in development_seeds)
        or len(set(development_seeds)) != len(development_seeds)
        or set(development_seeds) & set(train_seeds)
    ):
        raise ValueError("analysis development seeds are invalid")
    acceptance = development.get("acceptance")
    if acceptance != {
        "added_drop_95pct_upper_max": 0.0,
        "added_unsafe_force_95pct_upper_max": 0.0,
        "continue_rate_max": 0.25,
        "continue_rate_min": 0.02,
        "success_delta_95pct_lower_min": 0.0,
    }:
        raise ValueError("development policy-acceptance thresholds drifted")
    if (
        development.get("blind_rule")
        != "blind collection remains closed unless the immutable development audit passes every acceptance condition"
        or development.get("doubly_robust_nuisance")
        != "frozen five-member ensemble probability mean for each event and arm, clipped to [1e-6,1-1e-6]"
        or development.get("estimators")
        != {
            "fixed_route": "mean(2 * I(factual_route) * outcome)",
            "frozen_gate": "mean(2 * I(factual_arm equals frozen_gate_decision) * outcome)",
            "known_propensity": 0.5,
        }
    ):
        raise ValueError("analysis development estimator contract drifted")

    expected_inference = {
        "bound": "population mean plus_or_minus one population standard deviation across five members, clipped to [0,1]",
        "decision": "continue iff continue_success_LCB - route_success_UCB >= 0.05 AND continue_drop_UCB <= route_drop_LCB AND continue_unsafe_UCB <= route_unsafe_LCB; otherwise route",
        "equality_is_continue": True,
        "failure_default": "fixed route for non-finite values, invalid model, or underpowered data",
        "success_margin": 0.05,
    }
    if plan.get("inference") != expected_inference:
        raise ValueError("analysis inference contract drifted")
    expected_normalization = {
        "fit_rows": "all and only train-cohort factual candidate rows",
        "mean_dtype": "float64",
        "population_std_correction": 0,
        "std_floor": 1.0e-6,
        "training_cast_after_normalization": "float32",
    }
    if plan.get("normalization") != expected_normalization:
        raise ValueError("analysis normalization contract drifted")
    expected_model = {
        "architecture": [
            "Linear(165,128)",
            "SiLU",
            "Linear(128,128)",
            "SiLU",
            "Linear(128,6)",
        ],
        "ensemble_members": 5,
        "heads": [
            "continue_success",
            "continue_drop",
            "continue_unsafe_force",
            "route_success",
            "route_drop",
            "route_unsafe_force",
        ],
        "member_seeds": [0, 1, 2, 3, 4],
    }
    if plan.get("model") != expected_model:
        raise ValueError("analysis model contract drifted")
    expected_optimization = {
        "adamw_betas": [0.9, 0.999],
        "adamw_amsgrad": False,
        "adamw_capturable": False,
        "adamw_eps": 1.0e-8,
        "adamw_foreach": False,
        "adamw_fused": False,
        "adamw_maximize": False,
        "batch_size": 128,
        "epochs": 200,
        "gradient_clip_norm": 1.0,
        "initialization": "PyTorch Linear default initialization after torch.manual_seed(member_seed)",
        "last_batch": "included without padding or dropping",
        "learning_rate": 0.0003,
        "loss": "unweighted BCEWithLogits over exactly the assigned arm's three factual heads; the other three masks are zero",
        "optimizer": "AdamW",
        "runtime": "CPU float32 model, one intra-op thread, one inter-op thread, torch deterministic algorithms enabled",
        "shuffle": "one CPU torch.Generator seeded by member_seed; one randperm over all rows at each epoch",
        "weight_decay": 0.0001,
    }
    if plan.get("optimization") != expected_optimization:
        raise ValueError("analysis optimization contract drifted")
    expected_randomization = {
        "bootstrap": {
            "ci": "sort 20000 estimates ascending and use fixed one-based order-statistic ranks 500 and 19500",
            "replicates": 20000,
            "resampling_unit": "within each development seed, resample its observed seed_x_env_slot clusters with replacement to the same observed-cluster count; every available a/b row stays together and multiplicity is preserved",
            "seed": "unsigned big-endian integer from the first 8 bytes of SHA256(trial_manifest_sha256 + NUL + development_cluster_bootstrap_v1)",
        },
        "cluster_key": ["seed", "env_slot"],
        "events_reported_separately": ["success", "dropped", "unsafe_force"],
        "permutation": {
            "p_value": "two-sided plus-one Monte Carlo: (1 + count(abs(permuted)>=abs(observed))) / (replicates + 1)",
            "replicates": 20000,
            "seed": "unsigned big-endian integer from the first 8 bytes of SHA256(trial_manifest_sha256 + NUL + development_cluster_permutation_v1)",
            "swap": "independently for each development seed and Monte Carlo replicate, choose exactly 256 of 512 env slots as replicate-a route; replicate-b is the exact complement; apply those arm labels to every available row without changing features or outcomes",
        },
        "safety_events_do_not_cancel": True,
    }
    if plan.get("randomization_inference") != expected_randomization:
        raise ValueError("analysis randomization-inference contract drifted")
    outputs = plan.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "development_audit",
        "development_ledger",
        "frozen_model",
        "train_ledger",
        "training_report",
    }:
        raise ValueError("analysis output schema is not exact")
    expected_outputs = {
        "development_audit": "logs/flashsac/pick_tool/29_public_route_trial_v2_20260722/analysis/development_acceptance_v1.json",
        "development_ledger": "logs/flashsac/pick_tool/29_public_route_trial_v2_20260722/analysis/development_evidence_ledger_v1.json",
        "frozen_model": "logs/flashsac/pick_tool/29_public_route_trial_v2_20260722/analysis/frozen_public_route_gate_train_v1.pt",
        "train_ledger": "logs/flashsac/pick_tool/29_public_route_trial_v2_20260722/analysis/train_evidence_ledger_v1.json",
        "training_report": "logs/flashsac/pick_tool/29_public_route_trial_v2_20260722/analysis/frozen_public_route_gate_train_v1.json",
    }
    if outputs != expected_outputs:
        raise ValueError("analysis output paths drifted")
    for key, value in outputs.items():
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"analysis output {key} is not repository-relative")

    preregistration = plan.get("preregistration")
    if not isinstance(preregistration, dict) or set(preregistration) != {
        "branch",
        "collection_training_contract_sha256",
        "development_audit_tag",
        "development_ledger_tag",
        "frozen_model_tag",
        "implementation_base_commit",
        "implementation_commit",
        "seal_tag",
        "train_ledger_tag",
        "trial_manifest_sha256",
        "trial_seal_tag",
    }:
        raise ValueError("analysis preregistration schema is not exact")
    for name in (
        "collection_training_contract_sha256",
        "trial_manifest_sha256",
    ):
        if not _is_sha256(preregistration.get(name)):
            raise ValueError(f"analysis preregistration {name} is invalid")
    if (
        preregistration.get("branch") != "flashsac-pick-tool-curriculum"
        or preregistration.get("collection_training_contract_sha256")
        != "f76cffea274651547f0c47efcd6226a8b0a91d9716fe1b10b8a9f8cc8aca7f37"
        or preregistration.get("implementation_base_commit")
        != "c0da68cf6fb05cb57e9f53ddc65385cd2e1c2f3b"
        or preregistration.get("seal_tag")
        != "pick-tool-public-route-gate-analysis-v1-20260722"
        or preregistration.get("train_ledger_tag")
        != "pick-tool-public-route-train-ledger-v1-20260722"
        or preregistration.get("development_audit_tag")
        != "pick-tool-public-route-development-audit-v1-20260722"
        or preregistration.get("development_ledger_tag")
        != "pick-tool-public-route-development-ledger-v1-20260722"
        or preregistration.get("frozen_model_tag")
        != "pick-tool-public-route-frozen-gate-v1-20260722"
        or preregistration.get("trial_seal_tag")
        != "pick-tool-public-route-trial-v2-20260722"
    ):
        raise ValueError("analysis preregistration identity drifted")
    base = preregistration.get("implementation_base_commit")
    if not (
        isinstance(base, str)
        and len(base) == 40
        and all(character in "0123456789abcdef" for character in base)
    ):
        raise ValueError("analysis implementation base commit is invalid")
    implementation = preregistration.get("implementation_commit")
    if plan["status"] == "preregistered":
        if not (
            isinstance(implementation, str)
            and len(implementation) == 40
            and all(character in "0123456789abcdef" for character in implementation)
        ):
            raise ValueError("analysis implementation commit is not sealed")
    elif implementation != "SET_AFTER_DRAFT_IMPLEMENTATION_COMMIT":
        raise ValueError("draft analysis implementation placeholder is invalid")
    return {**plan, "_sha256": digest, "_path": str(Path(path).resolve())}


def _load_trial_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest, digest = _load_json_regular(path, "trial manifest")
    if (
        manifest.get("kind") != TRIAL_MANIFEST_KIND
        or manifest.get("format_version") != 2
        or manifest.get("status") != "preregistered"
    ):
        raise ValueError("unsupported or unsealed trial manifest")
    required = {
        "assignment",
        "checkpoints",
        "claim_boundary",
        "cohorts",
        "collection_acceptance",
        "feature_contract",
        "flashsac",
        "format_version",
        "gate",
        "kind",
        "outputs",
        "preregistration",
        "runtime_assets",
        "status",
        "task",
        "training",
        "trial_population",
    }
    if set(manifest) != required:
        raise ValueError("trial manifest top-level schema is not exact")
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, dict) or set(cohorts) != {
        "pilot",
        "train",
        "development",
        "blind",
    }:
        raise ValueError("trial manifest cohorts are invalid")
    all_seeds: set[int] = set()
    for cohort, entry in cohorts.items():
        if not isinstance(entry, dict):
            raise TypeError(f"manifest cohort {cohort} must be a mapping")
        seeds = entry.get("seeds")
        if (
            not isinstance(seeds, list)
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or all_seeds.intersection(seeds)
        ):
            raise ValueError("manifest cohort seeds are invalid or overlap")
        all_seeds.update(seeds)
        num_envs = entry.get("num_envs")
        if type(num_envs) is not int or num_envs <= 0 or num_envs % 2:
            raise ValueError(f"manifest cohort {cohort} num_envs must be positive and even")
        if cohort in ALLOWED_COHORTS and entry.get("replicates") != ["a", "b"]:
            raise ValueError(f"manifest cohort {cohort} lacks complementary replicates")
    assignment = manifest.get("assignment")
    if (
        not isinstance(assignment, dict)
        or assignment.get("algorithm") != "sha256_rank_balanced_v1"
        or assignment.get("replicates") != ["a", "b"]
        or assignment.get("replicate_b_semantics")
        != "exact_boolean_complement_of_replicate_a"
        or assignment.get("known_route_propensity") != 0.5
        or not isinstance(assignment.get("salt"), str)
        or not assignment["salt"]
    ):
        raise ValueError("trial assignment contract is invalid")
    feature = manifest.get("feature_contract")
    if (
        not isinstance(feature, dict)
        or feature.get("kind") != FEATURE_CONTRACT
        or feature.get("feature_dim") != FEATURE_DIM
    ):
        raise ValueError("trial feature contract is invalid")
    outputs = manifest.get("outputs")
    if (
        not isinstance(outputs, dict)
        or outputs.get("artifact_template") != "{cohort}_s{seed}_{replicate}.pt"
        or outputs.get("report_template") != "{cohort}_s{seed}_{replicate}.json"
        or outputs.get("run_order_enforced") is not True
    ):
        raise ValueError("trial output contract is invalid")
    _validate_threshold_schema(manifest)
    return manifest, digest


def _validate_threshold_schema(manifest: Mapping[str, Any]) -> None:
    values = manifest.get("collection_acceptance")
    required = {
        "development_min_factual_rows_per_arm",
        "development_min_factual_rows_per_seed_arm",
        "development_min_factual_rows_total",
        "minimum_rows_per_stratum_arm",
        "pilot_changes_allowed",
        "train_min_factual_rows_per_arm",
        "train_min_factual_rows_per_seed_arm",
        "train_min_factual_rows_total",
    }
    if not isinstance(values, dict) or set(values) != required:
        raise ValueError("collection acceptance schema is not exact")
    for name in required - {"pilot_changes_allowed"}:
        if type(values[name]) is not int or values[name] < 0:
            raise ValueError(f"collection acceptance {name} must be a non-negative integer")


def _canonical_run_paths(
    manifest: Mapping[str, Any],
    repository_root: Path,
    cohort: str,
    seed: int,
    replicate: str,
) -> tuple[Path, Path]:
    outputs = manifest["outputs"]
    output_root = _relative_output(
        repository_root, outputs["repository_relative_root"], "trial output root"
    )
    values = {"cohort": cohort, "seed": seed, "replicate": replicate}
    artifact = output_root / outputs["artifact_template"].format(**values)
    report = output_root / outputs["report_template"].format(**values)
    if artifact.parent != output_root or report.parent != output_root or artifact == report:
        raise ValueError("trial templates do not produce one safe canonical pair")
    return artifact, report


def _canonical_tensor_bytes(value: torch.Tensor) -> bytes:
    value = value.detach().cpu().contiguous()
    dtype_name = str(value.dtype).encode("ascii")
    shape = json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    # Every admitted dtype has a stable NumPy representation on this platform.
    return (
        len(dtype_name).to_bytes(4, "big")
        + dtype_name
        + len(shape).to_bytes(4, "big")
        + shape
        + value.numpy().tobytes(order="C")
    )


def _hash_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(_canonical_tensor_bytes(value)).hexdigest()


def semantic_trial_artifact_sha256(payload: Mapping[str, Any]) -> str:
    normalized = validate_trial_artifact(payload)
    digest = hashlib.sha256()
    digest.update(b"pick_tool_public_route_trial_semantic_v1\0")
    digest.update(
        json.dumps(
            normalized["metadata"], sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    )
    for name in sorted(normalized["tensors"]):
        digest.update(b"\0" + name.encode("utf-8") + b"\0")
        digest.update(_canonical_tensor_bytes(normalized["tensors"][name]))
    return digest.hexdigest()


def _load_weights_only_artifact(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    _require_regular_canonical(path, root, "trial artifact")
    raw = _snapshot_regular_bytes(path, "trial artifact")
    digest = hashlib.sha256(raw).hexdigest()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("trial artifact root must be a mapping")
    return validate_trial_artifact(payload), digest


def _event_summary(tensors: Mapping[str, torch.Tensor], mask: torch.Tensor) -> dict[str, Any]:
    count = int(mask.sum().item())
    clearance = tensors["outcome_max_true_clearance_m"][mask]
    return {
        "episodes": count,
        "success": int(tensors["outcome_success"][mask].sum().item()),
        "failure": int(tensors["outcome_failure"][mask].sum().item()),
        "time_out": int(tensors["outcome_time_out"][mask].sum().item()),
        "dropped": int(tensors["outcome_dropped"][mask].sum().item()),
        "unsafe_force": int(tensors["outcome_unsafe_force"][mask].sum().item()),
        "ever_grasped": int(tensors["outcome_ever_grasped"][mask].sum().item()),
        "ever_clearance_ge_5cm": int((clearance >= 0.05).sum().item()),
        "max_true_clearance_m": float(clearance.max().item()) if count else 0.0,
    }


def _validate_report_summary(actual: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(actual, dict) or set(actual) != _SUMMARY_FIELDS:
        raise ValueError(f"{label} summary schema is not exact")
    for name in _SUMMARY_FIELDS - {"max_true_clearance_m"}:
        if type(actual[name]) is not int or actual[name] != expected[name]:
            raise ValueError(f"{label} summary {name} disagrees with artifact")
    maximum = actual["max_true_clearance_m"]
    if type(maximum) not in (int, float) or not math.isfinite(float(maximum)):
        raise ValueError(f"{label} summary maximum is not finite")
    if not math.isclose(float(maximum), float(expected["max_true_clearance_m"]), rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"{label} summary maximum disagrees with artifact")


def _validate_all_assigned_summary(actual: Any, num_envs: int, label: str) -> None:
    if not isinstance(actual, dict) or set(actual) != _SUMMARY_FIELDS:
        raise ValueError(f"{label} all-assigned summary schema is not exact")
    if actual.get("episodes") != num_envs // 2:
        raise ValueError(f"{label} all-assigned episode count is invalid")
    for name in _SUMMARY_FIELDS - {"episodes", "max_true_clearance_m"}:
        if type(actual[name]) is not int or not 0 <= actual[name] <= num_envs // 2:
            raise ValueError(f"{label} all-assigned {name} count is invalid")
    maximum = actual["max_true_clearance_m"]
    if type(maximum) not in (int, float) or not math.isfinite(float(maximum)):
        raise ValueError(f"{label} all-assigned maximum is invalid")


def _checkpoint_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    search = manifest["checkpoints"]["search"]
    route = manifest["checkpoints"]["route_v6"]
    return {
        "search_checkpoint_sha256": search["sha256"],
        "route_actor_sha256": route["actor_sha256"],
        "route_task_contract_sha256": route["task_contract_sha256"],
        "route_frozen_actor_sha256": route["frozen_lift_actor_sha256"],
        "route_bridge_state_sha256": route["torch_bridge_state_sha256"],
    }


def _load_one_registered_run(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    repository_root: Path,
    cohort: str,
    seed: int,
    replicate: str,
    registered_cohort_run_index: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load one canonical pair and return artifact, report and compact receipt."""

    artifact_path, report_path = _canonical_run_paths(
        manifest, repository_root, cohort, seed, replicate
    )
    artifact, artifact_sha = _load_weights_only_artifact(artifact_path, repository_root)
    _require_regular_canonical(report_path, repository_root, "trial report")
    report, report_sha = _load_json_regular(report_path, "trial report")
    if set(report) != _REPORT_FIELDS:
        raise ValueError("trial report schema is not exact")
    num_envs = int(manifest["cohorts"][cohort]["num_envs"])
    if (
        report.get("kind") != COLLECTOR_REPORT_KIND
        or report.get("status") != "complete"
        or report.get("cohort") != cohort
        or report.get("seed") != seed
        or report.get("replicate") != replicate
        or report.get("num_envs") != num_envs
        or report.get("manifest_sha256") != manifest_sha256
        or report.get("artifact_sha256") != artifact_sha
    ):
        raise ValueError("trial report identity or immutable receipt is invalid")
    vector_steps = report.get("vector_steps")
    if type(vector_steps) is not int or not 1 <= vector_steps <= 999:
        raise ValueError("trial report vector_steps is invalid")

    metadata = artifact["metadata"]
    provenance = metadata["provenance"]
    if (
        metadata["seed"] != seed
        or metadata["num_envs"] != num_envs
        or metadata["replicate"] != replicate
        or metadata["assignment_salt"] != manifest["assignment"]["salt"]
        or provenance.get("cohort") != cohort
        or provenance.get("seed") != seed
        or provenance.get("replicate") != replicate
        or provenance.get("num_envs") != num_envs
        or provenance.get("episodes") != num_envs
        or provenance.get("manifest_sha256") != manifest_sha256
        or provenance.get("feature_contract") != manifest["feature_contract"]
        or provenance.get("gate_contract") != manifest["gate"]
        or provenance.get("task_contract") != manifest["task"]
        or provenance.get("assignment_salt") != manifest["assignment"]["salt"]
        or provenance.get("known_route_propensity") != 0.5
        or provenance.get("causal_counterfactual_claim_allowed") is not False
    ):
        raise ValueError("artifact provenance disagrees with manifest/run identity")
    for name, expected in _checkpoint_receipt(manifest).items():
        if provenance.get(name) != expected:
            raise ValueError(f"artifact checkpoint receipt {name} disagrees with manifest")
    for name in ("source_sha256", "runtime_asset_sha256", "git"):
        value = provenance.get(name)
        if report.get(name) != value or not isinstance(value, dict) or not value:
            raise ValueError(f"artifact/report {name} receipts disagree or are empty")
    for mapping_name in ("source_sha256", "runtime_asset_sha256"):
        if any(not isinstance(key, str) or not _is_sha256(value) for key, value in provenance[mapping_name].items()):
            raise ValueError(f"artifact {mapping_name} contains an invalid digest")
    if provenance["runtime_asset_sha256"] != manifest["runtime_assets"]["files"]:
        raise ValueError("artifact runtime-asset receipts differ from the sealed manifest")
    if (
        provenance["source_sha256"].get(
            "scripts/flashsac/public_route_trial_manifest.json"
        )
        != manifest_sha256
    ):
        raise ValueError("sealed source receipt does not authenticate the trial manifest")
    git = provenance["git"]
    if (
        git.get("branch") != manifest["preregistration"]["branch"]
        or not isinstance(git.get("commit"), str)
        or len(git["commit"]) != 40
        or any(character not in "0123456789abcdef" for character in git["commit"])
        or not isinstance(git.get("seal_commit"), str)
        or len(git["seal_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in git["seal_commit"]
        )
        or git.get("source_files_dirty") is not False
        or git.get("flashsac_dirty") is not False
        or git.get("flashsac_commit") != manifest["flashsac"]["fork_commit"]
        or git.get("seal_tag") != manifest["preregistration"]["seal_tag"]
        or git.get("implementation_commit")
        != manifest["preregistration"]["implementation_commit"]
    ):
        raise ValueError("artifact Git receipt is dirty or differs from the trial seal")
    if report.get("claim_boundary") != manifest["claim_boundary"]:
        raise ValueError("report claim boundary differs from manifest")

    tensors = artifact["tensors"]
    count = int(tensors["feature"].shape[0])
    route_mask = tensors["factual_treatment_route"]
    route_count = int(route_mask.sum().item())
    continue_count = count - route_count
    integer_expectations = {
        "candidate_rows": count,
        "untriggered_episodes": num_envs - count,
        "candidate_route_rows": route_count,
        "candidate_continue_rows": continue_count,
        "assignment_route_slots": num_envs // 2,
        "assignment_continue_slots": num_envs // 2,
    }
    for name, expected in integer_expectations.items():
        if type(report.get(name)) is not int or report[name] != expected:
            raise ValueError(f"report {name} disagrees with complete artifact")
    _validate_report_summary(
        report["candidate_route_outcomes"], _event_summary(tensors, route_mask), "route"
    )
    _validate_report_summary(
        report["candidate_continue_outcomes"],
        _event_summary(tensors, ~route_mask),
        "continue",
    )
    _validate_all_assigned_summary(
        report["all_assigned_route_outcomes"], num_envs, "route"
    )
    _validate_all_assigned_summary(
        report["all_assigned_continue_outcomes"], num_envs, "continue"
    )

    relative_artifact = artifact_path.relative_to(repository_root.resolve()).as_posix()
    relative_report = report_path.relative_to(repository_root.resolve()).as_posix()
    receipt = {
        "cohort": cohort,
        "seed": seed,
        "replicate": replicate,
        "replicate_index": REPLICATE_INDEX[replicate],
        "registered_cohort_run_index": registered_cohort_run_index,
        "num_envs": num_envs,
        "candidate_rows": count,
        "candidate_route_rows": route_count,
        "candidate_continue_rows": continue_count,
        "artifact_path": relative_artifact,
        "artifact_sha256": artifact_sha,
        "artifact_semantic_sha256": semantic_trial_artifact_sha256(artifact),
        "assignment_route_sha256": _hash_tensor(tensors["assignment_route"]),
        "report_path": relative_report,
        "report_sha256": report_sha,
    }
    common = {
        "source_sha256": provenance["source_sha256"],
        "runtime_asset_sha256": provenance["runtime_asset_sha256"],
        "git": provenance["git"],
        "checkpoint_sha256": _checkpoint_receipt(manifest),
        "claim_boundary": manifest["claim_boundary"],
    }
    return artifact, receipt, common


def _empty_or_cat(parts: Sequence[torch.Tensor], *, width: int | None, dtype: torch.dtype) -> torch.Tensor:
    if parts:
        return torch.cat(tuple(parts), dim=0).contiguous()
    shape = (0, width) if width is not None else (0,)
    return torch.empty(shape, dtype=dtype)


def load_registered_factual_cohort(
    cohort: str,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    analysis_plan_path: Path = DEFAULT_ANALYSIS_PLAN,
    repository_root: Path | None = None,
    require_preregistered_plan: bool = True,
) -> dict[str, Any]:
    """Load every canonical complementary pair for ``train`` or ``development``."""

    if cohort not in ALLOWED_COHORTS:
        raise ValueError("factual gate loader accepts only train or development")
    manifest, manifest_sha = _load_trial_manifest(manifest_path)
    plan = load_analysis_plan(
        analysis_plan_path, require_preregistered=require_preregistered_plan
    )
    if plan["preregistration"]["trial_manifest_sha256"] != manifest_sha:
        raise ValueError("analysis plan is bound to a different trial manifest")
    expected_plan_seeds = (
        plan["data"]["train_seeds"]
        if cohort == "train"
        else plan["development"]["seeds"]
    )
    seeds = manifest["cohorts"][cohort]["seeds"]
    if seeds != expected_plan_seeds:
        raise ValueError(f"analysis plan {cohort} seeds differ from trial manifest")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else DEFAULT_MANIFEST.resolve().parents[2]
    )

    columns: dict[str, list[torch.Tensor]] = {name: [] for name in TENSOR_FIELDS}
    receipts: list[dict[str, Any]] = []
    common_receipt: dict[str, Any] | None = None
    per_pair_assignments: dict[tuple[int, str], torch.Tensor] = {}
    for seed_index, seed in enumerate(seeds):
        for replicate in REPLICATES:
            registered_replicates = ("a", "b") if seed % 2 == 0 else ("b", "a")
            registered_index = seed_index * 2 + registered_replicates.index(replicate)
            artifact, receipt, common = _load_one_registered_run(
                manifest=manifest,
                manifest_sha256=manifest_sha,
                repository_root=root,
                cohort=cohort,
                seed=seed,
                replicate=replicate,
                registered_cohort_run_index=registered_index,
            )
            if common_receipt is None:
                common_receipt = common
            elif common != common_receipt:
                raise ValueError("sealed source/runtime/checkpoint/Git receipts differ across runs")
            tensors = artifact["tensors"]
            assignment = tensors["assignment_route"]
            per_pair_assignments[(seed, replicate)] = assignment
            order = torch.argsort(tensors["env_slot"], stable=True)
            count = int(order.numel())
            columns["feature"].append(tensors["feature"].index_select(0, order))
            columns["treatment_route"].append(
                tensors["factual_treatment_route"].index_select(0, order)
            )
            columns["success"].append(tensors["outcome_success"].index_select(0, order))
            columns["dropped"].append(tensors["outcome_dropped"].index_select(0, order))
            columns["unsafe_force"].append(
                tensors["outcome_unsafe_force"].index_select(0, order)
            )
            columns["seed"].append(torch.full((count,), seed, dtype=torch.long))
            columns["env_slot"].append(tensors["env_slot"].index_select(0, order))
            columns["replicate_index"].append(
                torch.full((count,), REPLICATE_INDEX[replicate], dtype=torch.long)
            )
            columns["candidate_step"].append(
                tensors["candidate_step"].index_select(0, order)
            )
            columns["stratum"].append(tensors["stratum"].index_select(0, order))
            receipts.append(receipt)
        if not torch.equal(
            per_pair_assignments[(seed, "b")], ~per_pair_assignments[(seed, "a")]
        ):
            raise ValueError(f"seed {seed} replicate assignments are not exact complements")

    dataset = {
        "kind": DATASET_KIND,
        "format_version": DATASET_FORMAT_VERSION,
        "metadata": {
            "cohort": cohort,
            "manifest_sha256": manifest_sha,
            "analysis_plan_sha256": plan["_sha256"],
            "assignment_salt": manifest["assignment"]["salt"],
            "feature_contract": FEATURE_CONTRACT,
            "feature_dim": FEATURE_DIM,
            "seeds": list(seeds),
            "replicates": list(REPLICATES),
            "canonical_row_order": "manifest_seed_then_replicate_a_b_then_env_slot",
            "factual_events": ["success", "dropped", "unsafe_force"],
            "factual_rows_only": True,
            "common_collection_receipt": common_receipt or {},
        },
        "tensors": {
            "feature": _empty_or_cat(columns["feature"], width=FEATURE_DIM, dtype=torch.float32),
            **{
                name: _empty_or_cat(
                    columns[name],
                    width=None,
                    dtype=torch.bool if name in BOOL_TENSOR_FIELDS else torch.long,
                )
                for name in TENSOR_FIELDS
                if name != "feature"
            },
        },
        "receipts": receipts,
    }
    return validate_factual_dataset(dataset, expected_cohort=cohort)


def _require_dataset_tensor(
    tensors: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = tensors.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"dataset tensor {name} must have shape {shape}")
    if value.dtype != dtype or value.device.type != "cpu":
        raise TypeError(f"dataset tensor {name} must be a CPU {dtype} tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"dataset tensor {name} contains NaN or infinity")
    return value.contiguous()


def validate_factual_dataset(
    payload: Mapping[str, Any], *, expected_cohort: str | None = None
) -> dict[str, Any]:
    """Validate row completeness, canonical order and treatment assignments."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "kind",
        "format_version",
        "metadata",
        "tensors",
        "receipts",
    }:
        raise ValueError("factual dataset top-level schema is not exact")
    if payload.get("kind") != DATASET_KIND or payload.get("format_version") != 1:
        raise ValueError("unsupported factual dataset")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != _DATASET_METADATA_FIELDS:
        raise ValueError("factual dataset metadata schema is not exact")
    metadata = _plain_json(dict(metadata), label="dataset metadata")
    cohort = metadata["cohort"]
    if cohort not in ALLOWED_COHORTS or (
        expected_cohort is not None and cohort != expected_cohort
    ):
        raise ValueError("factual dataset cohort is invalid")
    if not _is_sha256(metadata["manifest_sha256"]) or not _is_sha256(
        metadata["analysis_plan_sha256"]
    ):
        raise ValueError("factual dataset manifest/plan receipt is invalid")
    if (
        not isinstance(metadata["assignment_salt"], str)
        or not metadata["assignment_salt"]
        or metadata["feature_contract"] != FEATURE_CONTRACT
        or metadata["feature_dim"] != FEATURE_DIM
        or metadata["replicates"] != ["a", "b"]
        or metadata["canonical_row_order"]
        != "manifest_seed_then_replicate_a_b_then_env_slot"
        or metadata["factual_events"] != ["success", "dropped", "unsafe_force"]
        or metadata["factual_rows_only"] is not True
    ):
        raise ValueError("factual dataset fixed metadata contract drifted")
    seeds = metadata["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("factual dataset seeds are invalid")
    common = metadata["common_collection_receipt"]
    if not isinstance(common, dict) or set(common) != {
        "source_sha256",
        "runtime_asset_sha256",
        "git",
        "checkpoint_sha256",
        "claim_boundary",
    }:
        raise ValueError("factual dataset common collection receipt is invalid")
    for name in ("source_sha256", "runtime_asset_sha256", "checkpoint_sha256"):
        values = common[name]
        if not isinstance(values, dict) or not values or any(
            not isinstance(key, str) or not _is_sha256(value)
            for key, value in values.items()
        ):
            raise ValueError(f"factual dataset common {name} receipt is invalid")

    tensors_value = payload.get("tensors")
    if not isinstance(tensors_value, Mapping) or set(tensors_value) != set(TENSOR_FIELDS):
        raise ValueError("factual dataset tensor schema is not exact")
    feature_value = tensors_value.get("feature")
    if not isinstance(feature_value, torch.Tensor) or feature_value.ndim != 2:
        raise ValueError("dataset feature must be rank two")
    count = int(feature_value.shape[0])
    tensors: dict[str, torch.Tensor] = {
        "feature": _require_dataset_tensor(
            tensors_value, "feature", (count, FEATURE_DIM), torch.float32
        )
    }
    for name in BOOL_TENSOR_FIELDS:
        tensors[name] = _require_dataset_tensor(
            tensors_value, name, (count,), torch.bool
        )
    for name in LONG_TENSOR_FIELDS:
        tensors[name] = _require_dataset_tensor(
            tensors_value, name, (count,), torch.long
        )
    if bool((tensors["success"] & (tensors["dropped"] | tensors["unsafe_force"])).any()):
        raise ValueError("dataset success overlaps a safety failure")
    if bool(((tensors["stratum"] < 0) | (tensors["stratum"] > 3)).any()):
        raise ValueError("dataset stratum is outside [0,3]")
    if bool(((tensors["replicate_index"] < 0) | (tensors["replicate_index"] > 1)).any()):
        raise ValueError("dataset replicate_index is outside {0,1}")
    feature = tensors["feature"]
    expected_onehot = torch.nn.functional.one_hot(
        torch.full((count,), 4), num_classes=5
    ).float()
    if not torch.equal(feature[:, 114:119], expected_onehot):
        raise ValueError("dataset contains a non-trigger feature row")
    if not bool(((feature[:, 104] == 1.0) & (feature[:, 105] == 0.0)).all()):
        raise ValueError("dataset contains a latched or non-public trigger row")
    second = torch.topk(feature[:, 91:95], k=2, dim=-1).values[:, 1]
    score = torch.minimum(feature[:, 95], second)
    if bool((score < READINESS_MIN_SCORE).any()):
        raise ValueError("dataset contains a below-threshold readiness row")
    expected_stratum = (
        (score >= STRATUM_G_THRESHOLD).long() * 2
        + (feature[:, 101] >= STRATUM_CLOSE_THRESHOLD).long()
    )
    if not torch.equal(expected_stratum, tensors["stratum"]):
        raise ValueError("dataset stratum disagrees with public feature")

    receipts_value = payload.get("receipts")
    if not isinstance(receipts_value, (list, tuple)):
        raise TypeError("factual dataset receipts must be a sequence")
    receipts = [_plain_json(receipt, label="dataset receipt") for receipt in receipts_value]
    expected_pair_order = [
        (seed, replicate) for seed in seeds for replicate in REPLICATES
    ]
    if len(receipts) != len(expected_pair_order):
        raise ValueError("factual dataset is missing a complementary replicate receipt")
    seed_rank = {seed: index for index, seed in enumerate(seeds)}
    expected_row_keys: list[tuple[int, int, int]] = []
    for receipt, (seed, replicate) in zip(receipts, expected_pair_order, strict=True):
        if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
            raise ValueError("factual dataset receipt schema is not exact")
        replicate_index = REPLICATE_INDEX[replicate]
        seed_index = seed_rank[seed]
        registered_replicates = ("a", "b") if seed % 2 == 0 else ("b", "a")
        registered_index = seed_index * 2 + registered_replicates.index(replicate)
        if (
            receipt["cohort"] != cohort
            or receipt["seed"] != seed
            or receipt["replicate"] != replicate
            or receipt["replicate_index"] != replicate_index
            or receipt["registered_cohort_run_index"] != registered_index
            or type(receipt["num_envs"]) is not int
            or receipt["num_envs"] <= 0
            or receipt["num_envs"] % 2
        ):
            raise ValueError("factual dataset receipt identity is invalid")
        for name in (
            "artifact_sha256",
            "artifact_semantic_sha256",
            "assignment_route_sha256",
            "report_sha256",
        ):
            if not _is_sha256(receipt[name]):
                raise ValueError(f"factual dataset receipt {name} is invalid")
        for name, suffix in (("artifact_path", ".pt"), ("report_path", ".json")):
            path = receipt[name]
            if (
                not isinstance(path, str)
                or Path(path).is_absolute()
                or ".." in Path(path).parts
                or not path.endswith(suffix)
            ):
                raise ValueError(f"factual dataset receipt {name} is non-canonical")
        expected_assignment = assignment_for(
            cohort,
            seed,
            receipt["num_envs"],
            metadata["assignment_salt"],
            replicate,
        )
        if receipt["assignment_route_sha256"] != _hash_tensor(expected_assignment):
            raise ValueError("factual dataset assignment receipt is not registered")
        group = (tensors["seed"] == seed) & (
            tensors["replicate_index"] == replicate_index
        )
        group_count = int(group.sum().item())
        route_count = int(tensors["treatment_route"][group].sum().item())
        if (
            type(receipt["candidate_rows"]) is not int
            or type(receipt["candidate_route_rows"]) is not int
            or type(receipt["candidate_continue_rows"]) is not int
            or receipt["candidate_rows"] != group_count
            or receipt["candidate_route_rows"] != route_count
            or receipt["candidate_continue_rows"] != group_count - route_count
        ):
            raise ValueError("factual dataset row counts disagree with immutable receipts")
        slots = tensors["env_slot"][group]
        if bool(((slots < 0) | (slots >= receipt["num_envs"])).any()) or len(
            set(slots.tolist())
        ) != group_count:
            raise ValueError("factual dataset has duplicate or invalid env slots")
        if group_count and not torch.equal(
            tensors["treatment_route"][group], expected_assignment[slots]
        ):
            raise ValueError("factual dataset treatment differs from registered assignment")
        expected_row_keys.extend(
            (seed_rank[seed], replicate_index, int(slot)) for slot in slots.tolist()
        )
    actual_row_keys = [
        (seed_rank.get(int(seed), -1), int(replicate), int(slot))
        for seed, replicate, slot in zip(
            tensors["seed"].tolist(),
            tensors["replicate_index"].tolist(),
            tensors["env_slot"].tolist(),
            strict=True,
        )
    ]
    if any(key[0] < 0 for key in actual_row_keys) or actual_row_keys != sorted(actual_row_keys):
        raise ValueError("factual dataset rows are not in canonical order")
    if len(expected_row_keys) != count:
        raise ValueError("factual dataset contains rows outside registered receipts")

    return {
        "kind": DATASET_KIND,
        "format_version": DATASET_FORMAT_VERSION,
        "metadata": metadata,
        "tensors": tensors,
        "receipts": receipts,
    }


def semantic_dataset_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical tensor content plus raw artifact/report receipts."""

    dataset = validate_factual_dataset(payload)
    digest = hashlib.sha256()
    digest.update(b"pick_tool_public_route_factual_dataset_semantic_v1\0")
    digest.update(
        json.dumps(
            dataset["metadata"], sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    )
    for name in TENSOR_FIELDS:
        digest.update(b"\0" + name.encode("utf-8") + b"\0")
        digest.update(_canonical_tensor_bytes(dataset["tensors"][name]))
    digest.update(b"\0receipts\0")
    digest.update(
        json.dumps(
            dataset["receipts"], sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _acceptance_manifest(plan_or_manifest: Mapping[str, Any] | Path) -> dict[str, Any]:
    if isinstance(plan_or_manifest, (str, os.PathLike, Path)):
        payload, _ = _load_trial_manifest(Path(plan_or_manifest))
        return payload
    if not isinstance(plan_or_manifest, Mapping):
        raise TypeError("collection acceptance source must be a manifest mapping or path")
    if plan_or_manifest.get("kind") == TRIAL_MANIFEST_KIND:
        payload = dict(plan_or_manifest)
        _validate_threshold_schema(payload)
        return payload
    if "collection_acceptance" in plan_or_manifest:
        payload = {"collection_acceptance": plan_or_manifest["collection_acceptance"]}
        _validate_threshold_schema(payload)
        return payload
    if plan_or_manifest.get("kind") == ANALYSIS_PLAN_KIND:
        payload, manifest_sha256 = _load_trial_manifest(DEFAULT_MANIFEST)
        expected = plan_or_manifest.get("preregistration", {}).get("trial_manifest_sha256")
        if expected != manifest_sha256:
            raise ValueError("analysis plan is not bound to the default trial manifest")
        return payload
    raise ValueError("collection acceptance source is unsupported")


def validate_collection_acceptance(
    payload: Mapping[str, Any],
    plan_or_manifest: Mapping[str, Any] | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Return exact preregistered row-count diagnostics without hiding failure."""

    dataset = validate_factual_dataset(payload)
    manifest = _acceptance_manifest(plan_or_manifest)
    cohort = dataset["metadata"]["cohort"]
    thresholds = manifest["collection_acceptance"]
    prefix = "train" if cohort == "train" else "development"
    total_min = thresholds[f"{prefix}_min_factual_rows_total"]
    arm_min = thresholds[f"{prefix}_min_factual_rows_per_arm"]
    seed_arm_min = thresholds[f"{prefix}_min_factual_rows_per_seed_arm"]
    stratum_arm_min = thresholds["minimum_rows_per_stratum_arm"]
    treatment = dataset["tensors"]["treatment_route"]
    total = int(treatment.numel())
    arm_rows = {
        "continue": int((~treatment).sum().item()),
        "route": int(treatment.sum().item()),
    }
    seed_arm_rows: dict[str, dict[str, int]] = {}
    for seed in dataset["metadata"]["seeds"]:
        seed_mask = dataset["tensors"]["seed"] == seed
        seed_arm_rows[str(seed)] = {
            "continue": int((seed_mask & ~treatment).sum().item()),
            "route": int((seed_mask & treatment).sum().item()),
        }
    stratum_arm_rows: dict[str, dict[str, int]] = {}
    for stratum in range(4):
        stratum_mask = dataset["tensors"]["stratum"] == stratum
        stratum_arm_rows[str(stratum)] = {
            "continue": int((stratum_mask & ~treatment).sum().item()),
            "route": int((stratum_mask & treatment).sum().item()),
        }
    checks = {
        "total_rows": total >= total_min,
        "rows_per_arm": min(arm_rows.values()) >= arm_min,
        "rows_per_seed_arm": min(
            count for values in seed_arm_rows.values() for count in values.values()
        )
        >= seed_arm_min,
        "rows_per_stratum_arm": min(
            count for values in stratum_arm_rows.values() for count in values.values()
        )
        >= stratum_arm_min,
    }
    return {
        "cohort": cohort,
        "row_count": total,
        "thresholds": {
            "minimum_rows_total": total_min,
            "minimum_rows_per_arm": arm_min,
            "minimum_rows_per_seed_arm": seed_arm_min,
            "minimum_rows_per_stratum_arm": stratum_arm_min,
        },
        "arm_rows": arm_rows,
        "seed_arm_rows": seed_arm_rows,
        "stratum_arm_rows": stratum_arm_rows,
        "checks": checks,
        "accepted": all(checks.values()),
    }


def build_evidence_ledger(
    payload: Mapping[str, Any],
    plan_or_manifest: Mapping[str, Any] | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    dataset = validate_factual_dataset(payload)
    acceptance = validate_collection_acceptance(dataset, plan_or_manifest)
    semantic_sha256 = semantic_dataset_sha256(dataset)
    ledger_tag = (
        "pick-tool-public-route-train-ledger-v1-20260722"
        if dataset["metadata"]["cohort"] == "train"
        else "pick-tool-public-route-development-ledger-v1-20260722"
    )
    ledger = {
        "kind": LEDGER_KIND,
        "format_version": LEDGER_FORMAT_VERSION,
        "status": "complete",
        "cohort": dataset["metadata"]["cohort"],
        "manifest_sha256": dataset["metadata"]["manifest_sha256"],
        "analysis_plan_sha256": dataset["metadata"]["analysis_plan_sha256"],
        "dataset_kind": DATASET_KIND,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_semantic_sha256": semantic_sha256,
        "input_bundle_sha256": semantic_sha256,
        "row_count": int(dataset["tensors"]["feature"].shape[0]),
        "expected_runs": len(dataset["receipts"]),
        "canonical_row_order": dataset["metadata"]["canonical_row_order"],
        "collection_seal_tag": "pick-tool-public-route-trial-v2-20260722",
        "analysis_seal_tag": "pick-tool-public-route-gate-analysis-v1-20260722",
        "ledger_tag": ledger_tag,
        "acceptance": acceptance,
        "receipts": dataset["receipts"],
    }
    return validate_evidence_ledger(
        ledger, dataset=dataset, plan_or_manifest=plan_or_manifest
    )


def validate_evidence_ledger(
    payload: Mapping[str, Any],
    *,
    dataset: Mapping[str, Any],
    plan_or_manifest: Mapping[str, Any] | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    dataset = validate_factual_dataset(dataset)
    if not isinstance(payload, Mapping) or set(payload) != _LEDGER_FIELDS:
        raise ValueError("evidence ledger schema is not exact")
    ledger = _plain_json(dict(payload), label="evidence ledger")
    expected_acceptance = validate_collection_acceptance(dataset, plan_or_manifest)
    semantic_sha256 = semantic_dataset_sha256(dataset)
    ledger_tag = (
        "pick-tool-public-route-train-ledger-v1-20260722"
        if dataset["metadata"]["cohort"] == "train"
        else "pick-tool-public-route-development-ledger-v1-20260722"
    )
    fixed = {
        "kind": LEDGER_KIND,
        "format_version": LEDGER_FORMAT_VERSION,
        "status": "complete",
        "cohort": dataset["metadata"]["cohort"],
        "manifest_sha256": dataset["metadata"]["manifest_sha256"],
        "analysis_plan_sha256": dataset["metadata"]["analysis_plan_sha256"],
        "dataset_kind": DATASET_KIND,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_semantic_sha256": semantic_sha256,
        "input_bundle_sha256": semantic_sha256,
        "row_count": int(dataset["tensors"]["feature"].shape[0]),
        "expected_runs": len(dataset["receipts"]),
        "canonical_row_order": dataset["metadata"]["canonical_row_order"],
        "collection_seal_tag": "pick-tool-public-route-trial-v2-20260722",
        "analysis_seal_tag": "pick-tool-public-route-gate-analysis-v1-20260722",
        "ledger_tag": ledger_tag,
        "acceptance": expected_acceptance,
        "receipts": dataset["receipts"],
    }
    if ledger != fixed:
        raise ValueError("evidence ledger disagrees with canonical factual dataset")
    return ledger


def _publish_json_no_clobber(payload: Mapping[str, Any], output: Path, root: Path) -> None:
    output = Path(os.path.abspath(os.fspath(output)))
    _require_nonsymlink_chain(output, root, "evidence ledger output")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"evidence ledger already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require_nonsymlink_chain(output.parent, root, "evidence ledger parent")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_strict_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def publish_evidence_ledger_no_clobber(
    cohort: str,
    *,
    dataset: Mapping[str, Any] | None = None,
    manifest_path: Path = DEFAULT_MANIFEST,
    analysis_plan_path: Path = DEFAULT_ANALYSIS_PLAN,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Build, validate and publish the cohort's sole canonical JSON ledger."""

    if cohort not in ALLOWED_COHORTS:
        raise ValueError("evidence ledger cohort must be train or development")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else DEFAULT_MANIFEST.resolve().parents[2]
    )
    manifest, manifest_sha = _load_trial_manifest(manifest_path)
    plan = load_analysis_plan(analysis_plan_path, require_preregistered=True)
    if plan["preregistration"]["trial_manifest_sha256"] != manifest_sha:
        raise ValueError("analysis plan is bound to a different trial manifest")
    if dataset is None:
        dataset = load_registered_factual_cohort(
            cohort,
            manifest_path=manifest_path,
            analysis_plan_path=analysis_plan_path,
            repository_root=root,
            require_preregistered_plan=True,
        )
    dataset = validate_factual_dataset(dataset, expected_cohort=cohort)
    if (
        dataset["metadata"]["manifest_sha256"] != manifest_sha
        or dataset["metadata"]["analysis_plan_sha256"] != plan["_sha256"]
    ):
        raise ValueError("dataset is not bound to the loaded manifest and analysis plan")
    ledger = build_evidence_ledger(dataset, manifest)
    output_name = "train_ledger" if cohort == "train" else "development_ledger"
    output = _relative_output(root, plan["outputs"][output_name], "evidence ledger output")
    _publish_json_no_clobber(ledger, output, root)
    return ledger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cohort", choices=ALLOWED_COHORTS, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ledger = publish_evidence_ledger_no_clobber(args.cohort)
    print(
        json.dumps(
            {
                "cohort": ledger["cohort"],
                "row_count": ledger["row_count"],
                "accepted": ledger["acceptance"]["accepted"],
                "dataset_semantic_sha256": ledger["dataset_semantic_sha256"],
                "ledger_tag": ledger["ledger_tag"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "DEFAULT_MANIFEST",
    "DEFAULT_ANALYSIS_PLAN",
    "DATASET_KIND",
    "LEDGER_KIND",
    "sha256_file",
    "load_analysis_plan",
    "load_registered_factual_cohort",
    "validate_factual_dataset",
    "validate_collection_acceptance",
    "semantic_trial_artifact_sha256",
    "semantic_dataset_sha256",
    "build_evidence_ledger",
    "validate_evidence_ledger",
    "publish_evidence_ledger_no_clobber",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
