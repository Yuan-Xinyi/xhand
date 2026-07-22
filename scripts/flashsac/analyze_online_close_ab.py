#!/usr/bin/env python3
"""Preregistered, simulation-free analysis of online CLOSE A/B trials.

The command line deliberately accepts only an analysis plan.  The plan pins
the six evidence files, model identities, Monte Carlo choices, thresholds,
and the output path before outcomes are inspected.
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
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import torch

from online_close_ab import (
    ARTIFACT_KIND,
    ASSIGNMENT_CONTRACT,
    ASSIGNMENT_SALT,
    COLLECTION_CONTRACT,
    FORMAT_VERSION as ARTIFACT_FORMAT_VERSION,
    HANDOFF_HOLD_STEPS,
    HANDOFF_MIN_SCORE,
    KIT_ARGS,
    REPORT_KIND as COLLECTION_REPORT_KIND,
    SOURCE_PATH_COUNT,
    SOURCE_PATH_SET_SHA256,
    assignment_candidate_mask,
    sha256_file,
    validate_artifact,
    validate_report,
)


PLAN_KIND = "pick_tool_online_close_ab_analysis_plan_v1"
REPORT_KIND = "pick_tool_online_close_ab_analysis_v1"
ANALYSIS_FORMAT_VERSION = 1
PRODUCTION_SEEDS = (290, 291, 292)
PRODUCTION_REPLICATES = ("a", "b")
PRODUCTION_RUN_ORDER = (
    (290, "a"),
    (290, "b"),
    (291, "b"),
    (291, "a"),
    (292, "a"),
    (292, "b"),
)
PRODUCTION_NUM_ENVS = 1024
MONTE_CARLO_REPLICATES = 20_000
EVENTS = (
    "success",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
)
ADVERSE_EVENTS = EVENTS[1:]
DOMAINS = ("conditional_triggered", "itt_all_assigned")
ESTIMATORS = ("horvitz_thompson", "hajek")
_MC_BATCH_SIZE = 128


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_load(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    raw = _read_regular_bytes(path, label=label)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {token}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _require_no_symlink_chain(path: Path, *, root: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {current}")


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    root = _repository_root()
    _require_no_symlink_chain(path, root=root, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FileNotFoundError(f"{label} is not a readable regular file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_exact(value: Any, names: Sequence[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{label} must contain exactly {tuple(names)}")
    return value


def _require_finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _resolve_registered_path(path_value: Any, *, root: Path, label: str) -> Path:
    """Normalize a preregistered path without requiring future evidence."""

    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    resolved = Path(os.path.abspath(path))
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside repository root {root}") from exc
    return resolved


def _resolve_output(path_value: Any, *, root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("output.report must be a non-empty path")
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    return _resolve_registered_path(path_value, root=root, label="output.report")


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_text(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=_repository_root(), text=True
    ).strip()


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(("git", *arguments), cwd=_repository_root())


def _git_paths_match(commit: str, paths: Sequence[str]) -> bool:
    return (
        subprocess.run(
            ("git", "diff", "--quiet", commit, "--", *paths),
            cwd=_repository_root(),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", ancestor, descendant),
            cwd=_repository_root(),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _validate_preregistration(
    preregistration: Mapping[str, Any], *, plan_path: Path, plan_sha256: str
) -> str:
    head = _git_text("rev-parse", "HEAD")
    branch = _git_text("branch", "--show-current")
    tag_commit = _git_text(
        "rev-parse", f"refs/tags/{preregistration['seal_tag']}^{{commit}}"
    )
    parent_line = _git_text("rev-list", "--parents", "-n", "1", tag_commit).split()
    expected_parent = preregistration["implementation_commit"]
    if head != tag_commit:
        raise ValueError("current HEAD must equal the analysis seal tag target")
    if len(parent_line) != 2 or parent_line != [tag_commit, expected_parent]:
        raise ValueError("analysis seal must be a single-parent child of implementation_commit")
    implementation_line = _git_text(
        "rev-list", "--parents", "-n", "1", expected_parent
    ).split()
    collector_base = preregistration["collector_base_commit"]
    if implementation_line != [expected_parent, collector_base] or not _git_is_ancestor(
        collector_base, expected_parent
    ):
        raise ValueError("analysis implementation must be a direct child of collector base")
    if branch != preregistration["branch"]:
        raise ValueError("current Git branch differs from the preregistered branch")
    if sha256_file(Path(__file__).resolve()) != preregistration[
        "analysis_source_sha256"
    ]:
        raise ValueError("analysis implementation bytes differ from the preregistered source")
    relative_plan = plan_path.relative_to(_repository_root()).as_posix()
    changed_at_seal = _git_text(
        "diff", "--name-only", expected_parent, tag_commit
    ).splitlines()
    if changed_at_seal != [relative_plan]:
        raise ValueError("analysis seal commit must add or change only its registered plan")
    sealed_plan = _git_bytes("show", f"{tag_commit}:{relative_plan}")
    if hashlib.sha256(sealed_plan).hexdigest() != plan_sha256:
        raise ValueError("working-tree analysis plan differs from the sealed Git blob")
    protected_sources = (
        "scripts/flashsac/analyze_online_close_ab.py",
        "scripts/flashsac/online_close_ab.py",
    )
    if not _git_paths_match(tag_commit, protected_sources):
        raise ValueError("analysis or evidence-contract source differs from the seal")
    return tag_commit


def load_analysis_plan(path: Path) -> dict[str, Any]:
    """Load the exact production analysis plan without reading outcomes."""

    repository_root = _repository_root()
    authored_path = Path(path)
    if authored_path.is_symlink():
        raise ValueError("analysis plan path must not be a symlink")
    plan_path = authored_path.resolve(strict=True)
    try:
        plan_path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("analysis plan must reside inside the repository") from exc
    plan, digest = _strict_json_load(plan_path, label="analysis plan")
    _require_exact(
        plan,
        (
            "kind",
            "format_version",
            "status",
            "data",
            "estimands",
            "inference",
            "acceptance",
            "output",
            "preregistration",
        ),
        label="analysis plan",
    )
    if (
        plan["kind"] != PLAN_KIND
        or plan["format_version"] != ANALYSIS_FORMAT_VERSION
        or plan["status"] != "preregistered"
    ):
        raise ValueError("unsupported or unregistered analysis plan")

    preregistration = _require_exact(
        plan["preregistration"],
        (
            "branch",
            "collector_base_commit",
            "implementation_commit",
            "seal_tag",
            "analysis_source_sha256",
        ),
        label="analysis preregistration",
    )
    for name in ("branch", "seal_tag"):
        if not isinstance(preregistration[name], str) or not preregistration[name]:
            raise ValueError(f"analysis preregistration {name} must be non-empty")
    for name in ("collector_base_commit", "implementation_commit"):
        commit = preregistration[name]
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError(f"analysis {name} must be a 40-character Git SHA")
    _require_sha256(
        preregistration["analysis_source_sha256"],
        name="analysis preregistration analysis_source_sha256",
    )
    seal_commit = _validate_preregistration(
        preregistration, plan_path=plan_path, plan_sha256=digest
    )

    data = _require_exact(
        plan["data"],
        ("seeds", "replicates", "num_envs", "events", "identity", "runs"),
        label="analysis plan data",
    )
    if data["seeds"] != list(PRODUCTION_SEEDS):
        raise ValueError("analysis plan must use production seeds 290--292")
    if data["replicates"] != list(PRODUCTION_REPLICATES):
        raise ValueError("analysis plan must use complementary replicates a,b")
    if data["num_envs"] != PRODUCTION_NUM_ENVS:
        raise ValueError("analysis plan must use 1024 environments per run")
    if data["events"] != list(EVENTS):
        raise ValueError("analysis events must keep success and safety separate")
    identity = _require_exact(
        data["identity"],
        (
            "artifact_kind",
            "artifact_format_version",
            "report_kind",
            "collection_contract",
            "assignment_contract",
            "assignment_salt",
            "kit_args",
            "handoff_min_score",
            "handoff_hold_steps",
            "baseline_actor_sha256",
            "candidate_actor_sha256",
            "common_frozen_lift_actor_sha256",
            "common_frozen_lift_semantic_sha256",
            "common_frozen_lift_source_actor_sha256",
            "search_checkpoint_sha256",
            "baseline_task_contract_sha256",
            "candidate_task_contract_sha256",
            "baseline_bridge_state_sha256",
            "candidate_bridge_state_sha256",
            "flashsac_upstream_commit",
            "flashsac_fork_commit",
            "source_sha256_digest",
            "source_path_count",
            "source_path_set_sha256",
            "runtime_asset_sha256_digest",
            "runtime_static_sha256",
            "collector_git_branch",
        ),
        label="analysis identity",
    )
    if (
        identity["artifact_kind"] != ARTIFACT_KIND
        or identity["artifact_format_version"] != ARTIFACT_FORMAT_VERSION
        or identity["report_kind"] != COLLECTION_REPORT_KIND
        or identity["collection_contract"] != COLLECTION_CONTRACT
        or identity["assignment_contract"] != ASSIGNMENT_CONTRACT
        or identity["assignment_salt"] != ASSIGNMENT_SALT
        or identity["kit_args"] != KIT_ARGS
        or identity["handoff_min_score"] != HANDOFF_MIN_SCORE
        or identity["handoff_hold_steps"] != HANDOFF_HOLD_STEPS
    ):
        raise ValueError("analysis identity contract changed")
    for name in (
        "baseline_actor_sha256",
        "candidate_actor_sha256",
        "common_frozen_lift_actor_sha256",
        "common_frozen_lift_semantic_sha256",
        "common_frozen_lift_source_actor_sha256",
        "search_checkpoint_sha256",
        "baseline_task_contract_sha256",
        "candidate_task_contract_sha256",
        "baseline_bridge_state_sha256",
        "candidate_bridge_state_sha256",
        "source_sha256_digest",
        "source_path_set_sha256",
        "runtime_asset_sha256_digest",
        "runtime_static_sha256",
    ):
        _require_sha256(identity[name], name=f"analysis identity {name}")
    if (
        identity["source_path_count"] != SOURCE_PATH_COUNT
        or identity["source_path_set_sha256"] != SOURCE_PATH_SET_SHA256
    ):
        raise ValueError("analysis source-path contract changed")
    for name in (
        "flashsac_upstream_commit",
        "flashsac_fork_commit",
    ):
        value = identity[name]
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"analysis identity {name} must be a 40-character Git SHA")
    if not isinstance(identity["collector_git_branch"], str) or not identity[
        "collector_git_branch"
    ]:
        raise ValueError("analysis identity collector_git_branch must be non-empty")
    if identity["baseline_actor_sha256"] == identity["candidate_actor_sha256"]:
        raise ValueError("analysis plan does not identify distinct A/B actors")

    runs = data["runs"]
    expected_keys = list(PRODUCTION_RUN_ORDER)
    if not isinstance(runs, list) or len(runs) != len(expected_keys):
        raise ValueError("analysis plan must register exactly six runs")
    normalized_runs: list[dict[str, Any]] = []
    for index, (run, expected_key) in enumerate(zip(runs, expected_keys)):
        run = _require_exact(
            run,
            ("seed", "replicate", "artifact", "report"),
            label=f"analysis run {index}",
        )
        if (run["seed"], run["replicate"]) != expected_key:
            raise ValueError("analysis runs must be ordered seed then a,b")
        normalized_runs.append(
            {
                **run,
                "artifact": _resolve_registered_path(
                    run["artifact"], root=repository_root, label=f"run {expected_key} artifact"
                ),
                "report": _resolve_registered_path(
                    run["report"], root=repository_root, label=f"run {expected_key} report"
                ),
            }
        )
    all_paths = [item[key] for item in normalized_runs for key in ("artifact", "report")]
    if len(set(all_paths)) != 12:
        raise ValueError("analysis run evidence paths must be unique")

    estimands = _require_exact(
        plan["estimands"],
        (
            "domains",
            "primary_estimator",
            "diagnostic_estimator",
            "known_propensity",
            "seed_weighting",
            "contrast",
            "conditional_population",
            "itt_population",
        ),
        label="analysis estimands",
    )
    expected_estimands = {
        "domains": list(DOMAINS),
        "primary_estimator": "horvitz_thompson",
        "diagnostic_estimator": "hajek",
        "known_propensity": 0.5,
        "seed_weighting": "equal arithmetic mean of the three per-seed estimates",
        "contrast": "candidate minus baseline",
        "conditional_population": "all first episodes whose public SEARCH handoff triggered",
        "itt_population": "all assigned first episodes",
    }
    if estimands != expected_estimands:
        raise ValueError("analysis estimand contract changed")

    inference = _require_exact(
        plan["inference"], ("bootstrap", "fisher"), label="analysis inference"
    )
    bootstrap = _require_exact(
        inference["bootstrap"],
        ("replicates", "seed", "resampling_unit", "lower_rank", "upper_rank"),
        label="analysis bootstrap",
    )
    if bootstrap["replicates"] != MONTE_CARLO_REPLICATES:
        raise ValueError("cluster bootstrap must use 20000 replicates")
    if bootstrap["resampling_unit"] != (
        "within each seed resample 1024 env_slot clusters with replacement; "
        "replicates a,b and their complementary assignments stay together"
    ):
        raise ValueError("cluster bootstrap resampling contract changed")
    for name in ("seed", "lower_rank", "upper_rank"):
        if not _is_int(bootstrap[name]):
            raise TypeError(f"bootstrap {name} must be an integer")
    if not (0 <= bootstrap["seed"] < 2**63):
        raise ValueError("bootstrap seed is outside the CPU generator range")
    if not (
        1 <= bootstrap["lower_rank"] <= bootstrap["upper_rank"] <= MONTE_CARLO_REPLICATES
    ):
        raise ValueError("bootstrap fixed ranks are invalid")
    if bootstrap["lower_rank"] != 1000 or bootstrap["upper_rank"] != 19001:
        raise ValueError("one-sided 95% bounds require fixed ranks 1000 and 19001")

    fisher = _require_exact(
        inference["fisher"],
        ("replicates", "seed", "assignment", "p_value", "primary_statistic"),
        label="analysis Fisher inference",
    )
    if fisher["replicates"] != MONTE_CARLO_REPLICATES:
        raise ValueError("Fisher randomization must use 20000 replicates")
    if not _is_int(fisher["seed"]) or not (0 <= fisher["seed"] < 2**63):
        raise ValueError("Fisher seed is outside the CPU generator range")
    if fisher["assignment"] != (
        "independently per seed choose exactly 512 candidate env slots in replicate a; "
        "replicate b is the exact complement"
    ) or fisher["p_value"] != (
        "plus-one Monte Carlo; report greater, less, and two-sided absolute-tail probabilities"
    ) or fisher["primary_statistic"] != (
        "equal-seed-weight conditional success Horvitz-Thompson candidate-minus-baseline delta"
    ):
        raise ValueError("Fisher sharp-null contract changed")

    acceptance = _require_exact(
        plan["acceptance"],
        (
            "minimum_triggered_per_arm_total",
            "minimum_triggered_per_seed_arm",
            "minimum_triggered_per_run_arm",
            "conditional_success_point_min",
            "conditional_success_ht_lower_strict_min",
            "itt_success_point_min",
            "itt_success_ht_lower_strict_min",
            "conditional_success_fisher_greater_p_max",
            "positive_seed_count_min",
            "seed_meets_both_success_margins_count_min",
            "conditional_adverse_ht_upper_max",
            "minimum_baseline_events_for_safety_inference",
            "maximum_candidate_events_when_safety_underpowered",
            "require_every_condition",
        ),
        label="analysis acceptance",
    )
    for name in (
        "minimum_triggered_per_arm_total",
        "minimum_triggered_per_seed_arm",
        "minimum_triggered_per_run_arm",
        "positive_seed_count_min",
        "seed_meets_both_success_margins_count_min",
        "minimum_baseline_events_for_safety_inference",
        "maximum_candidate_events_when_safety_underpowered",
    ):
        if not _is_int(acceptance[name]) or acceptance[name] < 0:
            raise ValueError(f"acceptance {name} must be a non-negative integer")
    if acceptance["positive_seed_count_min"] > len(PRODUCTION_SEEDS) or acceptance[
        "seed_meets_both_success_margins_count_min"
    ] > len(PRODUCTION_SEEDS):
        raise ValueError("acceptance seed-count threshold exceeds three seeds")
    for name in (
        "conditional_success_point_min",
        "conditional_success_ht_lower_strict_min",
        "itt_success_point_min",
        "itt_success_ht_lower_strict_min",
        "conditional_success_fisher_greater_p_max",
    ):
        number = _require_finite_number(acceptance[name], name=f"acceptance {name}")
        if not -1.0 <= number <= 1.0:
            raise ValueError(f"acceptance {name} is outside [-1,1]")
    safety_max = _require_exact(
        acceptance["conditional_adverse_ht_upper_max"],
        ADVERSE_EVENTS,
        label="adverse-event thresholds",
    )
    for event in ADVERSE_EVENTS:
        number = _require_finite_number(safety_max[event], name=f"safety {event}")
        if not -1.0 <= number <= 1.0:
            raise ValueError(f"safety threshold {event} is outside [-1,1]")
    if acceptance["require_every_condition"] is not True:
        raise ValueError("production acceptance must fail closed on every condition")

    output = _require_exact(plan["output"], ("report",), label="analysis output")
    normalized = dict(plan)
    normalized["data"] = {**data, "runs": normalized_runs}
    normalized["output"] = {"report": _resolve_output(output["report"], root=repository_root)}
    normalized["_path"] = plan_path
    normalized["_sha256"] = digest
    normalized["_seal_commit"] = seal_commit
    return normalized


def _torch_load_weights(raw: bytes) -> Any:
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch weights_only loading is required") from exc


def _identity_projection(metadata: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(metadata)
    projected.pop("seed")
    projected.pop("replicate")
    runtime = dict(projected["runtime"])
    runtime.pop("seed")
    projected["runtime"] = runtime
    return projected


def load_evidence(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate and assemble the exact six-run tensor cube."""

    expected_identity = plan["data"]["identity"]
    loaded: dict[tuple[int, str], dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    reference_identity: dict[str, Any] | None = None
    for run in plan["data"]["runs"]:
        artifact_path = run["artifact"]
        report_path = run["report"]
        artifact_raw = _read_regular_bytes(artifact_path, label="trial artifact")
        artifact_sha = hashlib.sha256(artifact_raw).hexdigest()
        artifact = validate_artifact(_torch_load_weights(artifact_raw))
        report, report_sha = _strict_json_load(report_path, label="collection report")
        validate_report(report, artifact, published=True)
        if report["artifact_sha256"] != artifact_sha:
            raise ValueError("collection report artifact receipt is stale")
        reported_path = Path(report["artifact_output"])
        if not reported_path.is_absolute():
            reported_path = Path.cwd() / reported_path
        if Path(os.path.abspath(reported_path)) != artifact_path:
            raise ValueError("collection report points at a different artifact")

        metadata = artifact["metadata"]
        if (
            metadata["seed"] != run["seed"]
            or metadata["replicate"] != run["replicate"]
            or metadata["num_envs"] != PRODUCTION_NUM_ENVS
        ):
            raise ValueError("registered run identity differs from artifact metadata")
        actual_identity = {
            "artifact_kind": metadata["kind"],
            "artifact_format_version": metadata["format_version"],
            "report_kind": report["kind"],
            "collection_contract": metadata["collection_contract"],
            "assignment_contract": metadata["assignment_contract"],
            "assignment_salt": metadata["assignment_salt"],
            "kit_args": metadata["kit_args"],
            "handoff_min_score": metadata["handoff_min_score"],
            "handoff_hold_steps": metadata["handoff_hold_steps"],
            "baseline_actor_sha256": metadata["baseline_actor_sha256"],
            "candidate_actor_sha256": metadata["candidate_actor_sha256"],
            "common_frozen_lift_actor_sha256": metadata[
                "common_frozen_lift_actor_sha256"
            ],
            "common_frozen_lift_semantic_sha256": metadata[
                "common_frozen_lift_semantic_sha256"
            ],
            "common_frozen_lift_source_actor_sha256": metadata[
                "common_frozen_lift_source_actor_sha256"
            ],
            "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
            "baseline_task_contract_sha256": metadata[
                "baseline_task_contract_sha256"
            ],
            "candidate_task_contract_sha256": metadata[
                "candidate_task_contract_sha256"
            ],
            "baseline_bridge_state_sha256": metadata[
                "baseline_bridge_state_sha256"
            ],
            "candidate_bridge_state_sha256": metadata[
                "candidate_bridge_state_sha256"
            ],
            "flashsac_upstream_commit": metadata["flashsac_upstream_commit"],
            "flashsac_fork_commit": metadata["flashsac_fork_commit"],
            "source_sha256_digest": _semantic_sha256(metadata["source_sha256"]),
            "source_path_count": len(metadata["source_sha256"]),
            "source_path_set_sha256": hashlib.sha256(
                "\0".join(sorted(metadata["source_sha256"])).encode("utf-8")
            ).hexdigest(),
            "runtime_asset_sha256_digest": _semantic_sha256(
                metadata["runtime_asset_sha256"]
            ),
            "runtime_static_sha256": _semantic_sha256(
                {key: value for key, value in metadata["runtime"].items() if key != "seed"}
            ),
            "collector_git_branch": metadata["git"]["branch"],
        }
        if actual_identity != expected_identity:
            raise ValueError("evidence identity differs from the preregistered identity")
        if metadata["git"]["commit"] != plan["_seal_commit"]:
            raise ValueError("artifact collector commit differs from the analysis seal target")
        projection = _identity_projection(metadata)
        if reference_identity is None:
            reference_identity = projection
        elif projection != reference_identity:
            raise ValueError("source, runtime, checkpoint, or task identity differs across runs")

        key = (run["seed"], run["replicate"])
        if key in loaded:
            raise ValueError("duplicate evidence run")
        loaded[key] = artifact
        receipts.append(
            {
                "seed": key[0],
                "replicate": key[1],
                "artifact": str(artifact_path),
                "artifact_sha256": artifact_sha,
                "report": str(report_path),
                "report_sha256": report_sha,
            }
        )

    for seed in PRODUCTION_SEEDS:
        tensors_a = loaded[(seed, "a")]["tensors"]
        tensors_b = loaded[(seed, "b")]["tensors"]
        if not torch.equal(tensors_a["env_slot"], tensors_b["env_slot"]):
            raise ValueError(f"seed {seed} replicate slots differ")
        if not torch.equal(
            tensors_a["assignment_candidate"], ~tensors_b["assignment_candidate"]
        ):
            raise ValueError(f"seed {seed} assignments are not exact complements")
        for replicate, tensors in (("a", tensors_a), ("b", tensors_b)):
            expected = assignment_candidate_mask(
                seed=seed, num_envs=PRODUCTION_NUM_ENVS, replicate=replicate
            )
            if not torch.equal(tensors["assignment_candidate"], expected):
                raise ValueError("artifact assignment is not the registered deterministic mask")

    cube: dict[str, torch.Tensor] = {}
    for name in ("assignment_candidate", "triggered", *EVENTS):
        cube[name] = torch.stack(
            [
                torch.stack(
                    [loaded[(seed, replicate)]["tensors"][name] for replicate in PRODUCTION_REPLICATES]
                )
                for seed in PRODUCTION_SEEDS
            ]
        )
    return {"runs": loaded, "cube": cube, "receipts": receipts}


def _domain_mask(cube: Mapping[str, torch.Tensor], domain: str) -> torch.Tensor:
    if domain == "conditional_triggered":
        return cube["triggered"]
    if domain == "itt_all_assigned":
        return torch.ones_like(cube["triggered"])
    raise ValueError(f"unknown analysis domain {domain!r}")


def _event_cube(cube: Mapping[str, torch.Tensor]) -> torch.Tensor:
    result = torch.stack([cube[name] for name in EVENTS], dim=-1)
    if result.shape != (3, 2, PRODUCTION_NUM_ENVS, len(EVENTS)):
        raise ValueError("evidence event cube has the wrong shape")
    return result.to(torch.float64)


def point_estimates(cube: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    assignment = cube["assignment_candidate"]
    outcomes = _event_cube(cube)
    result: dict[str, Any] = {estimator: {} for estimator in ESTIMATORS}
    for domain in DOMAINS:
        selected = _domain_mask(cube, domain)
        per_seed: dict[str, dict[str, list[float]]] = {
            estimator: {} for estimator in ESTIMATORS
        }
        aggregate: dict[str, dict[str, dict[str, float]]] = {
            estimator: {} for estimator in ESTIMATORS
        }
        for seed_index, seed in enumerate(PRODUCTION_SEEDS):
            d = selected[seed_index]
            z = assignment[seed_index]
            y = outcomes[seed_index]
            denominator = float(d.sum())
            if denominator <= 0.0:
                raise ValueError(f"seed {seed} has no rows in {domain}")
            ht_candidate = (2.0 * (z & d)[..., None] * y).sum(dim=(0, 1)) / denominator
            ht_baseline = (2.0 * ((~z) & d)[..., None] * y).sum(dim=(0, 1)) / denominator
            candidate_rows = int((z & d).sum())
            baseline_rows = int(((~z) & d).sum())
            if candidate_rows <= 0 or baseline_rows <= 0:
                raise ValueError(f"seed {seed} lacks one randomized arm in {domain}")
            hajek_candidate = ((z & d)[..., None] * y).sum(dim=(0, 1)) / candidate_rows
            hajek_baseline = (((~z) & d)[..., None] * y).sum(dim=(0, 1)) / baseline_rows
            for estimator, candidate, baseline in (
                ("horvitz_thompson", ht_candidate, ht_baseline),
                ("hajek", hajek_candidate, hajek_baseline),
            ):
                delta = candidate - baseline
                per_seed[estimator][str(seed)] = {
                    "candidate": candidate.tolist(),
                    "baseline": baseline.tolist(),
                    "delta": delta.tolist(),
                    "domain_rows": int(denominator),
                    "candidate_rows": candidate_rows,
                    "baseline_rows": baseline_rows,
                }
        for estimator in ESTIMATORS:
            seed_values = list(per_seed[estimator].values())
            for event_index, event in enumerate(EVENTS):
                candidate = sum(value["candidate"][event_index] for value in seed_values) / 3.0
                baseline = sum(value["baseline"][event_index] for value in seed_values) / 3.0
                aggregate[estimator][event] = {
                    "candidate": candidate,
                    "baseline": baseline,
                    "delta": candidate - baseline,
                }
            result[estimator][domain] = {
                "events": aggregate[estimator],
                "per_seed": per_seed[estimator],
            }
    return result


def cluster_bootstrap(
    cube: Mapping[str, torch.Tensor], *, replicates: int, random_seed: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Bootstrap env-slot clusters while keeping complementary a/b rows together."""

    assignment = cube["assignment_candidate"]
    outcomes = _event_cube(cube)
    result = {
        estimator: {
            domain: torch.zeros((replicates, len(EVENTS)), dtype=torch.float64)
            for domain in DOMAINS
        }
        for estimator in ESTIMATORS
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    for seed_index, seed in enumerate(PRODUCTION_SEEDS):
        z = assignment[seed_index]
        y = outcomes[seed_index]
        components: dict[str, tuple[torch.Tensor, ...]] = {}
        for domain in DOMAINS:
            d = _domain_mask(cube, domain)[seed_index]
            sign = torch.where(z, 1.0, -1.0)
            ht_numerator = (2.0 * sign[..., None] * d[..., None] * y).sum(dim=0)
            ht_denominator = d.to(torch.float64).sum(dim=0)
            candidate_numerator = (z[..., None] * d[..., None] * y).sum(dim=0)
            candidate_denominator = (z & d).to(torch.float64).sum(dim=0)
            baseline_numerator = ((~z)[..., None] * d[..., None] * y).sum(dim=0)
            baseline_denominator = ((~z) & d).to(torch.float64).sum(dim=0)
            components[domain] = (
                ht_numerator,
                ht_denominator,
                candidate_numerator,
                candidate_denominator,
                baseline_numerator,
                baseline_denominator,
            )
        for start in range(0, replicates, _MC_BATCH_SIZE):
            stop = min(start + _MC_BATCH_SIZE, replicates)
            sample = torch.randint(
                PRODUCTION_NUM_ENVS,
                (stop - start, PRODUCTION_NUM_ENVS),
                generator=generator,
            )
            for domain in DOMAINS:
                (
                    ht_numerator,
                    ht_denominator,
                    candidate_numerator,
                    candidate_denominator,
                    baseline_numerator,
                    baseline_denominator,
                ) = components[domain]
                ht_denom = ht_denominator[sample].sum(dim=1)
                cand_denom = candidate_denominator[sample].sum(dim=1)
                base_denom = baseline_denominator[sample].sum(dim=1)
                if bool(((ht_denom <= 0) | (cand_denom <= 0) | (base_denom <= 0)).any()):
                    raise RuntimeError(f"bootstrap produced an empty {domain} arm for seed {seed}")
                result["horvitz_thompson"][domain][start:stop] += (
                    ht_numerator[sample].sum(dim=1) / ht_denom[:, None] / 3.0
                )
                result["hajek"][domain][start:stop] += (
                    candidate_numerator[sample].sum(dim=1) / cand_denom[:, None]
                    - baseline_numerator[sample].sum(dim=1) / base_denom[:, None]
                ) / 3.0
    for estimator in ESTIMATORS:
        for domain in DOMAINS:
            if not bool(torch.isfinite(result[estimator][domain]).all()):
                raise FloatingPointError("cluster bootstrap produced a non-finite estimate")
    return result


def fisher_randomization(
    cube: Mapping[str, torch.Tensor], *, replicates: int, random_seed: int
) -> dict[str, torch.Tensor]:
    """Monte Carlo sharp-null distribution under balanced complementary labels."""

    outcomes = _event_cube(cube)
    result = {
        domain: torch.zeros((replicates, len(EVENTS)), dtype=torch.float64)
        for domain in DOMAINS
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    for start in range(0, replicates, _MC_BATCH_SIZE):
        stop = min(start + _MC_BATCH_SIZE, replicates)
        batch = stop - start
        for seed_index in range(len(PRODUCTION_SEEDS)):
            scores = torch.rand(
                (batch, PRODUCTION_NUM_ENVS), generator=generator, dtype=torch.float64
            )
            selected_slots = torch.topk(
                scores,
                k=PRODUCTION_NUM_ENVS // 2,
                dim=1,
                largest=False,
                sorted=False,
            ).indices
            assignment_a = torch.zeros(
                (batch, PRODUCTION_NUM_ENVS), dtype=torch.bool
            )
            assignment_a.scatter_(1, selected_slots, True)
            assignment = torch.stack((assignment_a, ~assignment_a), dim=1)
            sign = torch.where(assignment, 1.0, -1.0)
            for domain in DOMAINS:
                d = _domain_mask(cube, domain)[seed_index]
                denominator = d.to(torch.float64).sum()
                if denominator <= 0:
                    raise ValueError("Fisher conditional population is empty")
                result[domain][start:stop] += (
                    2.0
                    * sign[..., None]
                    * d[None, ..., None]
                    * outcomes[seed_index][None, ...]
                ).sum(dim=(1, 2)) / denominator / 3.0
    return result


def _intervals(
    samples: Mapping[str, Mapping[str, torch.Tensor]], *, lower_rank: int, upper_rank: int
) -> dict[str, Any]:
    result: dict[str, Any] = {estimator: {} for estimator in ESTIMATORS}
    for estimator in ESTIMATORS:
        for domain in DOMAINS:
            ordered = torch.sort(samples[estimator][domain], dim=0).values
            result[estimator][domain] = {
                event: {
                    "lower": float(ordered[lower_rank - 1, event_index]),
                    "upper": float(ordered[upper_rank - 1, event_index]),
                }
                for event_index, event in enumerate(EVENTS)
            }
    return result


def _fisher_p_values(
    null_samples: Mapping[str, torch.Tensor], points: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain in DOMAINS:
        observed = torch.tensor(
            [
                points["horvitz_thompson"][domain]["events"][event]["delta"]
                for event in EVENTS
            ],
            dtype=torch.float64,
        )
        samples = null_samples[domain]
        denominator = int(samples.shape[0]) + 1
        result[domain] = {}
        for index, event in enumerate(EVENTS):
            value = observed[index]
            result[domain][event] = {
                "observed": float(value),
                "greater": (1 + int((samples[:, index] >= value).sum())) / denominator,
                "less": (1 + int((samples[:, index] <= value).sum())) / denominator,
                "two_sided": (
                    1 + int((samples[:, index].abs() >= value.abs()).sum())
                ) / denominator,
            }
    return result


def _counts(evidence: Mapping[str, Any]) -> dict[str, Any]:
    runs = evidence["runs"]
    per_run: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    total = {"baseline": 0, "candidate": 0}
    raw_events = {
        domain: {
            event: {"baseline": 0, "candidate": 0} for event in EVENTS
        }
        for domain in DOMAINS
    }
    for seed in PRODUCTION_SEEDS:
        seed_counts = {"baseline": 0, "candidate": 0}
        for replicate in PRODUCTION_REPLICATES:
            tensors = runs[(seed, replicate)]["tensors"]
            z = tensors["assignment_candidate"]
            triggered = tensors["triggered"]
            key = f"{seed}_{replicate}"
            per_run[key] = {
                "baseline": int(((~z) & triggered).sum()),
                "candidate": int((z & triggered).sum()),
            }
            for arm in ("baseline", "candidate"):
                seed_counts[arm] += per_run[key][arm]
                total[arm] += per_run[key][arm]
            for domain in DOMAINS:
                domain_mask = (
                    triggered
                    if domain == "conditional_triggered"
                    else torch.ones_like(triggered)
                )
                for event in EVENTS:
                    raw_events[domain][event]["baseline"] += int(
                        ((~z) & domain_mask & tensors[event]).sum()
                    )
                    raw_events[domain][event]["candidate"] += int(
                        (z & domain_mask & tensors[event]).sum()
                    )
        per_seed[str(seed)] = seed_counts
    return {
        "triggered_per_run": per_run,
        "triggered_per_seed": per_seed,
        "triggered_total": total,
        "raw_event_counts": raw_events,
    }


def decide(
    *,
    plan: Mapping[str, Any],
    points: Mapping[str, Any],
    intervals: Mapping[str, Any],
    fisher: Mapping[str, Any],
    counts: Mapping[str, Any],
) -> dict[str, Any]:
    threshold = plan["acceptance"]
    checks: dict[str, bool] = {}
    checks["minimum_triggered_per_arm_total"] = all(
        value >= threshold["minimum_triggered_per_arm_total"]
        for value in counts["triggered_total"].values()
    )
    checks["minimum_triggered_per_seed_arm"] = all(
        value >= threshold["minimum_triggered_per_seed_arm"]
        for seed_counts in counts["triggered_per_seed"].values()
        for value in seed_counts.values()
    )
    checks["minimum_triggered_per_run_arm"] = all(
        value >= threshold["minimum_triggered_per_run_arm"]
        for run_counts in counts["triggered_per_run"].values()
        for value in run_counts.values()
    )
    conditional_point = points["horvitz_thompson"]["conditional_triggered"][
        "events"
    ]["success"]["delta"]
    itt_point = points["horvitz_thompson"]["itt_all_assigned"]["events"][
        "success"
    ]["delta"]
    conditional_lower = intervals["horvitz_thompson"]["conditional_triggered"][
        "success"
    ]["lower"]
    itt_lower = intervals["horvitz_thompson"]["itt_all_assigned"]["success"][
        "lower"
    ]
    checks["conditional_success_point"] = (
        conditional_point >= threshold["conditional_success_point_min"]
    )
    checks["conditional_success_ht_lower"] = (
        conditional_lower > threshold["conditional_success_ht_lower_strict_min"]
    )
    checks["itt_success_point"] = itt_point >= threshold["itt_success_point_min"]
    checks["itt_success_ht_lower"] = (
        itt_lower > threshold["itt_success_ht_lower_strict_min"]
    )
    checks["conditional_success_fisher"] = (
        fisher["conditional_triggered"]["success"]["greater"]
        <= threshold["conditional_success_fisher_greater_p_max"]
    )
    per_seed = points["horvitz_thompson"]["conditional_triggered"]["per_seed"]
    itt_per_seed = points["horvitz_thompson"]["itt_all_assigned"]["per_seed"]
    positive = sum(
        per_seed[str(seed)]["delta"][0] > 0.0
        and itt_per_seed[str(seed)]["delta"][0] > 0.0
        for seed in PRODUCTION_SEEDS
    )
    meets_both = sum(
        per_seed[str(seed)]["delta"][0]
        >= threshold["conditional_success_point_min"]
        and itt_per_seed[str(seed)]["delta"][0] >= threshold["itt_success_point_min"]
        for seed in PRODUCTION_SEEDS
    )
    checks["positive_seed_count"] = positive >= threshold["positive_seed_count_min"]
    checks["seed_meets_both_success_margins_count"] = (
        meets_both >= threshold["seed_meets_both_success_margins_count_min"]
    )

    safety: dict[str, Any] = {}
    for event in ADVERSE_EVENTS:
        baseline_events = counts["raw_event_counts"]["conditional_triggered"][event][
            "baseline"
        ]
        candidate_events = counts["raw_event_counts"]["conditional_triggered"][event][
            "candidate"
        ]
        underpowered = (
            baseline_events < threshold["minimum_baseline_events_for_safety_inference"]
        )
        upper = intervals["horvitz_thompson"]["conditional_triggered"][event][
            "upper"
        ]
        if underpowered:
            passed = candidate_events <= threshold[
                "maximum_candidate_events_when_safety_underpowered"
            ]
            rule = "candidate_count_fallback"
        else:
            passed = upper <= threshold["conditional_adverse_ht_upper_max"][event]
            rule = "conditional_ht_upper_bound"
        checks[f"safety_{event}"] = passed
        safety[event] = {
            "rule": rule,
            "underpowered": underpowered,
            "baseline_events": baseline_events,
            "candidate_events": candidate_events,
            "conditional_ht_delta_upper": upper,
            "itt_ht_delta": points["horvitz_thompson"]["itt_all_assigned"][
                "events"
            ][event]["delta"],
            "itt_ht_interval": intervals["horvitz_thompson"][
                "itt_all_assigned"
            ][event],
            "threshold": (
                threshold["maximum_candidate_events_when_safety_underpowered"]
                if underpowered
                else threshold["conditional_adverse_ht_upper_max"][event]
            ),
            "passed": passed,
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_seed_count": positive,
        "seed_meets_both_success_margins_count": meets_both,
        "safety": safety,
    }


def analyze(plan: Mapping[str, Any]) -> dict[str, Any]:
    evidence = load_evidence(plan)
    points = point_estimates(evidence["cube"])
    bootstrap_spec = plan["inference"]["bootstrap"]
    samples = cluster_bootstrap(
        evidence["cube"],
        replicates=bootstrap_spec["replicates"],
        random_seed=bootstrap_spec["seed"],
    )
    intervals = _intervals(
        samples,
        lower_rank=bootstrap_spec["lower_rank"],
        upper_rank=bootstrap_spec["upper_rank"],
    )
    fisher_spec = plan["inference"]["fisher"]
    fisher_null = fisher_randomization(
        evidence["cube"],
        replicates=fisher_spec["replicates"],
        random_seed=fisher_spec["seed"],
    )
    fisher = _fisher_p_values(fisher_null, points)
    counts = _counts(evidence)
    decision = decide(
        plan=plan, points=points, intervals=intervals, fisher=fisher, counts=counts
    )
    return {
        "kind": REPORT_KIND,
        "format_version": ANALYSIS_FORMAT_VERSION,
        "status": "complete",
        "analysis_plan": str(plan["_path"]),
        "analysis_plan_sha256": plan["_sha256"],
        "preregistration": {
            **plan["preregistration"],
            "seal_commit": plan["_seal_commit"],
        },
        "collection_contract": COLLECTION_CONTRACT,
        "identity": plan["data"]["identity"],
        "evidence_receipts": evidence["receipts"],
        "counts": counts,
        "point_estimates": points,
        "bootstrap": {
            "replicates": bootstrap_spec["replicates"],
            "seed": bootstrap_spec["seed"],
            "lower_rank": bootstrap_spec["lower_rank"],
            "upper_rank": bootstrap_spec["upper_rank"],
            "intervals": intervals,
        },
        "fisher_randomization": {
            "replicates": fisher_spec["replicates"],
            "seed": fisher_spec["seed"],
            "p_values": fisher,
        },
        "acceptance_thresholds": plan["acceptance"],
        "decision": decision,
    }


def _publish(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    temporary: Path | None = None
    linked = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        linked = True
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if linked and temporary is not None:
            try:
                if path.stat().st_ino == temporary.stat().st_ino:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis_plan", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = load_analysis_plan(args.analysis_plan)
    report = analyze(plan)
    _publish(plan["output"]["report"], report)
    print(json.dumps(report["decision"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
