#!/usr/bin/env python3
"""Fail-closed reserved-seed validation of Candidate 39's fixed direction.

The fixed hand14 direction is an immutable input to this analysis.  This file
only estimates its treatment effect on seeds 329/330 and decides whether it is
eligible for a separately preregistered formal screen.  It never derives,
rewrites, or publishes a direction.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

import candidate39_fixed_episode_residual as artifact_contract


REPORT_KIND = "pick_tool_candidate39_fixed_direction_development_validation_v1"
FORMAT_VERSION = 1
SEEDS = (329, 330)
REPLICATES = ("a", "b")
RUN_ORDER = ((329, "a"), (329, "b"), (330, "b"), (330, "a"))
NUM_ENVS = 64
KNOWN_PROPENSITY = 0.5
INPUT_PATTERN = (
    "logs/flashsac/pick_tool/51_c39_fixed_direction_dev_s{seed}_{replicate}/trial.pt"
)
REPORT_OUTPUT = (
    "logs/flashsac/pick_tool/51_c39_fixed_direction_development_validation.json"
)

# Kept independently in the analyzer so a change in the collector/contract
# cannot silently redefine the randomized assignment checked here.
ASSIGNMENT_CONTRACT = (
    "candidate39_fixed_direction_sha256_rank_exact_balanced_complement_v1"
)
ASSIGNMENT_SALT = "pick_tool_candidate39_fixed_direction_validation_20260722_v1"
DISCOVERY_ASSIGNMENT_SALT = "pick_tool_candidate39_option_residual_20260722_v1"

EVENT_FIELDS = {
    "latched_within_32_actions": "latched_within_window",
    "ever_grasped": "ever_grasped",
    "ever_true_clearance_ge_20cm": "ever_clearance_ge_20cm",
    "success": "success",
    "dropped": "dropped",
    "unsafe_force": "unsafe_force",
    "unlatched_clearance_ge_5cm": "unlatched_clearance_ge_5cm",
}

# These keys and JSON scalar values must exactly equal the sealed plan after
# its all_must_pass control flag is removed.
GATE_THRESHOLDS: dict[str, Any] = {
    "primary_conditional_latched_within_32_actions_delta_min_exclusive": 0.0,
    "positive_primary_delta_seeds_min": 1,
    "per_seed_primary_delta_floor": -0.1,
    "conditional_ever_grasped_delta_min": 0.0,
    "conditional_success_delta_floor": -0.03,
    "common_exposure_opportunity_slots_min_across_validation": 32,
    "common_exposure_opportunity_slots_min_per_seed": 12,
    "treatment_triggered_dropped_max": 0,
    "treatment_triggered_unsafe_force_max": 0,
    "conditional_unlatched_clearance_ge_5cm_delta_max": 0.02,
    "new_abs_action_ge_098_element_fraction_max": 0.005,
    "episodes_with_new_abs_action_ge_098_max_per_run": 1,
    "any_new_abs_action_ge_0999": False,
    "component_and_l2_budget_violations": 0,
    "action_algebra_arm_latch_control_inactive_and_zero_exact": True,
    "checkpoint_manifest_assignment_and_fixed_z_identity_exact": True,
}

TOKEN_COMPONENT_CAP = 0.10
DISTAL_COMPONENT_CAP = 0.05
DELTA_L2_CAP = 0.20
_BUDGET_ATOL = 2.0e-7


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
    path = _regular_file(root / artifact_contract.VALIDATION_PLAN, "validation plan")
    raw = path.read_bytes()
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("Candidate 39 fixed-direction plan is invalid JSON") from error
    if not isinstance(plan, dict):
        raise RuntimeError("Candidate 39 fixed-direction plan must be a JSON object")
    return plan, _sha256_bytes(raw)


def _registered_validation_gates(
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Require executable gate names, types, and values to match the plan."""

    plan, _ = _read_plan(repository_root)
    gates = plan.get("development_validation_gates")
    if not isinstance(gates, dict):
        raise RuntimeError("fixed-direction plan has no development gate mapping")
    gates = dict(gates)
    if gates.pop("all_must_pass", None) is not True:
        raise RuntimeError("fixed-direction development gates are not fail-closed")
    if set(gates) != set(GATE_THRESHOLDS):
        raise RuntimeError("fixed-direction analyzer gate keys differ from the plan")
    for name, expected in GATE_THRESHOLDS.items():
        if not _same_typed_scalar(gates[name], expected):
            raise RuntimeError(
                f"fixed-direction analyzer threshold {name!r} differs from the plan"
            )
    return gates


def _independent_rank(*, seed: int, num_envs: int) -> torch.Tensor:
    ranked = sorted(
        range(num_envs),
        key=lambda slot: (
            hashlib.sha256(
                f"{ASSIGNMENT_SALT}\0rank\0{seed}\0{slot}".encode("utf-8")
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


def _validate_plan_identity(plan: Mapping[str, Any]) -> None:
    if plan.get("kind") != (
        "pick_tool_candidate39_fixed_direction_development_validation_plan_v1"
    ):
        raise ValueError("fixed-direction validation-plan kind changed")
    if plan.get("status") != artifact_contract.SEALED_PLAN_STATUS:
        raise ValueError("fixed-direction validation plan is not sealed")
    if plan.get("branch") != artifact_contract.REQUIRED_BRANCH:
        raise ValueError("fixed-direction validation-plan branch changed")
    assignment = plan.get("assignment")
    if not isinstance(assignment, Mapping):
        raise ValueError("fixed-direction plan assignment is missing")
    exact_assignment = {
        "contract": ASSIGNMENT_CONTRACT,
        "salt": ASSIGNMENT_SALT,
        "salt_is_independent_of_discovery": True,
        "num_envs": NUM_ENVS,
    }
    for name, expected in exact_assignment.items():
        if not _same_typed_scalar(assignment.get(name), expected):
            raise ValueError(f"fixed-direction plan assignment {name!r} changed")
    if ASSIGNMENT_SALT == DISCOVERY_ASSIGNMENT_SALT:
        raise RuntimeError("fixed-direction assignment reused the discovery salt")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping) or (
        execution.get("seeds") != list(SEEDS)
        or execution.get("replicates") != list(REPLICATES)
        or execution.get("num_envs_per_run") != NUM_ENVS
        or execution.get("run_order") != [f"{seed}{rep}" for seed, rep in RUN_ORDER]
    ):
        raise ValueError("fixed-direction plan execution set/order changed")
    estimands = plan.get("estimands")
    if not isinstance(estimands, Mapping) or not _same_typed_scalar(
        estimands.get("known_propensity"), KNOWN_PROPENSITY
    ):
        raise ValueError("fixed-direction plan propensity changed")


def _validate_implementation_seal(
    plan: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Bind the sealed source hashes to the named implementation commit."""

    seal = plan.get("implementation_seal")
    expected_fields = {
        "status",
        "implementation_commit",
        "source_sha256",
        "simulation_free_tests",
    }
    if not isinstance(seal, Mapping) or set(seal) != expected_fields:
        raise ValueError("fixed-direction implementation seal schema changed")
    if seal["status"] != "complete_without_simulator_evidence":
        raise ValueError("fixed-direction implementation seal status changed")
    commit = seal["implementation_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("implementation_commit must be a full lowercase Git SHA")
    source = seal["source_sha256"]
    expected_sources = tuple(artifact_contract.IMPLEMENTATION_SOURCE_FILES)
    if not isinstance(source, dict) or set(source) != set(expected_sources):
        raise ValueError("implementation source receipt set changed")
    for relative in expected_sources:
        digest = source[relative]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"implementation source digest is invalid: {relative}")
        current = _sha256_bytes(
            _regular_file(repository_root / relative, f"implementation source {relative}")
            .read_bytes()
        )
        if current != digest:
            raise ValueError(f"implementation source changed after seal: {relative}")
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
        raise ValueError("implementation simulation-free test receipt is empty")
    # Contract validation below checks strict-JSON serializability.  Preserve
    # the exact registered receipt without interpreting outcomes post hoc.
    return {
        "status": seal["status"],
        "implementation_commit": commit,
        "source_sha256": dict(source),
        "simulation_free_tests": dict(tests),
    }


def _validate_sealed_fixed_direction(
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Validate plan -> manifest -> payload path/hash/semantic receipts."""

    root = _root() if repository_root is None else Path(repository_root)
    plan, plan_sha = _read_plan(root)
    sealed_plan = artifact_contract.validate_sealed_validation_plan(
        root / artifact_contract.VALIDATION_PLAN
    )
    if sealed_plan != plan:
        raise ValueError("contract and analyzer read different validation plans")
    _validate_plan_identity(plan)
    _registered_validation_gates(root)
    implementation_receipt = _validate_implementation_seal(
        plan, repository_root=root
    )
    evidence = plan.get("immutable_fixed_direction_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("plan has no immutable fixed-direction evidence")
    plan_manifest = evidence.get("manifest")
    plan_payload = evidence.get("payload")
    if not isinstance(plan_manifest, Mapping) or not isinstance(plan_payload, Mapping):
        raise ValueError("plan fixed-direction receipt is incomplete")
    if (
        plan_manifest.get("path") != artifact_contract.FIXED_DIRECTION_MANIFEST
        or plan_manifest.get("sha256")
        != artifact_contract.FIXED_DIRECTION_MANIFEST_SHA256
        or plan_payload.get("path") != artifact_contract.FIXED_DIRECTION_PATH
        or plan_payload.get("sha256") != artifact_contract.FIXED_DIRECTION_SHA256
        or plan_payload.get("kind")
        != artifact_contract.FIXED_DIRECTION_PAYLOAD_KIND
    ):
        raise ValueError("plan fixed-direction path/hash/kind receipt changed")

    manifest_path = _regular_file(
        root / artifact_contract.FIXED_DIRECTION_MANIFEST,
        "fixed-direction manifest",
    )
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = _sha256_bytes(manifest_raw)
    if manifest_sha != artifact_contract.FIXED_DIRECTION_MANIFEST_SHA256:
        raise ValueError("fixed-direction manifest SHA256 differs from the receipt")
    try:
        manifest = json.loads(manifest_raw)
    except json.JSONDecodeError as error:
        raise ValueError("fixed-direction manifest is invalid JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("fixed-direction manifest must be an object")
    manifest_direction = manifest.get("fixed_direction")
    manifest_report = manifest.get("discovery_report")
    if not isinstance(manifest_direction, Mapping) or not isinstance(
        manifest_report, Mapping
    ):
        raise ValueError("fixed-direction manifest lacks direction/report receipts")
    if (
        manifest.get("kind") != "pick_tool_candidate39_fixed_direction_manifest_v1"
        or manifest.get("status") != "sealed_before_fixed_direction_validation"
        or manifest_direction.get("path") != artifact_contract.FIXED_DIRECTION_PATH
        or manifest_direction.get("sha256")
        != artifact_contract.FIXED_DIRECTION_SHA256
        or manifest_direction.get("payload_kind")
        != artifact_contract.FIXED_DIRECTION_PAYLOAD_KIND
        or manifest_report.get("semantic_sha256")
        != artifact_contract.FIXED_DIRECTION_SEMANTIC_SHA256
    ):
        raise ValueError("fixed-direction manifest semantic receipt changed")

    payload_path = _regular_file(
        root / artifact_contract.FIXED_DIRECTION_PATH, "fixed-direction payload"
    )
    payload_raw = payload_path.read_bytes()
    if _sha256_bytes(payload_raw) != artifact_contract.FIXED_DIRECTION_SHA256:
        raise ValueError("fixed-direction payload SHA256 differs from the receipt")
    payload = artifact_contract.validate_fixed_direction_payload(
        _torch_load_weights_only(payload_raw)
    )
    if payload["discovery_report_semantic_sha256"] != (
        artifact_contract.FIXED_DIRECTION_SEMANTIC_SHA256
    ):
        raise ValueError("fixed-direction payload semantic receipt changed")
    if not torch.equal(payload["fixed_z"], artifact_contract.expected_fixed_z()):
        raise ValueError("fixed-direction payload vector changed")
    manifest_z = torch.tensor(manifest_direction.get("fixed_z"), dtype=torch.float32)
    if not torch.equal(manifest_z, payload["fixed_z"]):
        raise ValueError("manifest and payload fixed_z differ")
    return {
        "validation_plan": artifact_contract.VALIDATION_PLAN,
        "validation_plan_sha256": plan_sha,
        "implementation_seal": implementation_receipt,
        "fixed_direction_manifest": artifact_contract.FIXED_DIRECTION_MANIFEST,
        "fixed_direction_manifest_sha256": manifest_sha,
        "fixed_direction_payload": artifact_contract.FIXED_DIRECTION_PATH,
        "fixed_direction_payload_sha256": artifact_contract.FIXED_DIRECTION_SHA256,
        "fixed_direction_payload_kind": artifact_contract.FIXED_DIRECTION_PAYLOAD_KIND,
        "fixed_direction_semantic_sha256": (
            artifact_contract.FIXED_DIRECTION_SEMANTIC_SHA256
        ),
        "fixed_direction_dtype": "float32",
        "fixed_direction_shape": [artifact_contract.HAND_ACTION_DIM],
        "direction_was_recomputed_or_modified": False,
    }


def _read_artifact(path: Path) -> tuple[dict[str, Any], str]:
    path = _regular_file(path, "fixed-direction artifact")
    raw = path.read_bytes()
    return artifact_contract.validate_artifact(_torch_load_weights_only(raw)), _sha256_bytes(
        raw
    )


def _without_run_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    normalized.pop("seed", None)
    normalized.pop("replicate", None)
    # This receipt is deliberately run-varying because replicate B hashes the
    # exact complement of replicate A (and each seed has an independent rank).
    normalized.pop("assignment_mask_sha256", None)
    runtime = normalized.get("runtime")
    if isinstance(runtime, Mapping):
        runtime = dict(runtime)
        runtime.pop("seed", None)
        normalized["runtime"] = runtime
    return normalized


def _require_tensor_equal(
    actual: torch.Tensor, expected: torch.Tensor, label: str
) -> None:
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise ValueError(f"{label} is not bit-exact")


def validate_validation_artifacts(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    repository_root: Path | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Validate four runs, independent assignment, lineage, and complements."""

    fixed_receipt = _validate_sealed_fixed_direction(repository_root)
    if (
        artifact_contract.ASSIGNMENT_CONTRACT != ASSIGNMENT_CONTRACT
        or artifact_contract.ASSIGNMENT_SALT != ASSIGNMENT_SALT
    ):
        raise RuntimeError("artifact and analyzer fixed-direction assignments differ")
    expected_keys = {(seed, replicate) for seed in SEEDS for replicate in REPLICATES}
    if set(artifacts) != expected_keys:
        raise ValueError("validation evidence must be exactly seeds 329/330 x a/b")

    validated: dict[tuple[int, str], dict[str, Any]] = {}
    reference_identity: dict[str, Any] | None = None
    plan, _ = _read_plan(repository_root)
    policies = plan.get("immutable_policies")
    if not isinstance(policies, Mapping):
        raise ValueError("plan immutable policy receipt is missing")
    policy_fields = (
        "v6_checkpoint",
        "v6_actor_sha256",
        "v6_task_contract_sha256",
        "v6_bridge_state_sha256",
        "frozen_lift_actor_sha256",
        "frozen_lift_semantic_sha256",
        "frozen_lift_source_actor_sha256",
        "search_checkpoint",
        "search_checkpoint_sha256",
        "flashsac_fork_commit",
    )
    for seed, replicate in RUN_ORDER:
        artifact = artifact_contract.validate_artifact(artifacts[(seed, replicate)])
        metadata = artifact["metadata"]
        episodes = artifact["episodes"]
        if metadata["seed"] != seed or metadata["replicate"] != replicate:
            raise ValueError(f"artifact {seed}{replicate} run identity changed")
        if metadata["num_envs"] != NUM_ENVS:
            raise ValueError(f"artifact {seed}{replicate} is not a 64-env run")
        if metadata["validation_plan_sha256"] != fixed_receipt[
            "validation_plan_sha256"
        ]:
            raise ValueError(f"artifact {seed}{replicate} plan receipt changed")
        for name in policy_fields:
            if metadata.get(name) != policies.get(name):
                raise ValueError(f"artifact {seed}{replicate} policy {name!r} changed")
        implementation_sources = plan["implementation_seal"]["source_sha256"]
        for relative, digest in implementation_sources.items():
            if metadata["source_sha256"].get(relative) != digest:
                raise ValueError(
                    f"artifact {seed}{replicate} source receipt changed: {relative}"
                )
        expected_treatment = _independent_treatment_mask(
            seed=seed, num_envs=NUM_ENVS, replicate=replicate
        )
        expected_rank = _independent_rank(seed=seed, num_envs=NUM_ENVS)
        _require_tensor_equal(
            episodes["env_slot"], torch.arange(NUM_ENVS), f"{seed}{replicate} slots"
        )
        _require_tensor_equal(
            episodes["treatment"], expected_treatment, f"{seed}{replicate} assignment"
        )
        _require_tensor_equal(
            episodes["assignment_rank"], expected_rank, f"{seed}{replicate} rank"
        )
        _require_tensor_equal(
            episodes["fixed_z"],
            artifact_contract.expected_fixed_z().expand(NUM_ENVS, -1),
            f"{seed}{replicate} fixed_z",
        )
        if int(expected_treatment.sum()) != NUM_ENVS // 2:
            raise RuntimeError("independent assignment is not exactly balanced")
        identity = _without_run_identity(metadata)
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise ValueError(
                "checkpoint, manifest, fixed-z, source, asset, or runtime identity differs"
            )
        validated[(seed, replicate)] = artifact

    for seed in SEEDS:
        a, b = artifact_contract.validate_complementary_artifacts(
            validated[(seed, "a")], validated[(seed, "b")]
        )
        _require_tensor_equal(
            a["episodes"]["treatment"],
            ~b["episodes"]["treatment"],
            f"seed {seed} exact complement",
        )
        for field in ("env_slot", "assignment_rank", "fixed_z"):
            _require_tensor_equal(
                a["episodes"][field],
                b["episodes"][field],
                f"seed {seed} cross-replicate {field}",
            )
        # Trigger masks/clocks may differ across independent GPU runs.  They
        # are intentionally not an identity requirement; HT uses each run's
        # actually observed pre-treatment trigger domain.
    return validated


def load_fixed_evidence(
    repository_root: Path | None = None,
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    root = _root() if repository_root is None else Path(repository_root)
    artifacts: dict[tuple[int, str], dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for seed, replicate in RUN_ORDER:
        path = root / INPUT_PATTERN.format(seed=seed, replicate=replicate)
        artifact, digest = _read_artifact(path)
        artifacts[(seed, replicate)] = artifact
        receipts.append(
            {
                "run": f"{seed}{replicate}",
                "artifact": str(path.resolve()),
                "artifact_sha256": digest,
            }
        )
    return validate_validation_artifacts(
        artifacts, repository_root=root
    ), receipts


def _ht_event(
    treatment: torch.Tensor, domain: torch.Tensor, event: torch.Tensor
) -> dict[str, float | int]:
    denominator = int(domain.sum())
    treatment_rows = int((treatment & domain).sum())
    control_rows = int(((~treatment) & domain).sum())
    if denominator <= 0 or treatment_rows <= 0 or control_rows <= 0:
        raise ValueError("a reserved seed lacks triggered support in one arm")
    treatment_count = int((treatment & domain & event).sum())
    control_count = int(((~treatment) & domain & event).sum())
    treatment_estimate = treatment_count / (KNOWN_PROPENSITY * denominator)
    control_estimate = control_count / ((1.0 - KNOWN_PROPENSITY) * denominator)
    return {
        "known_propensity": KNOWN_PROPENSITY,
        "domain_rows": denominator,
        "treatment_rows": treatment_rows,
        "control_rows": control_rows,
        "treatment_count": treatment_count,
        "control_count": control_count,
        "treatment": treatment_estimate,
        "control": control_estimate,
        "delta": treatment_estimate - control_estimate,
    }


def _raw_arm_funnel(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    treatment_arm: bool,
) -> dict[str, Any]:
    assigned = 0
    triggered = 0
    counts = {name: 0 for name in EVENT_FIELDS}
    for seed in SEEDS:
        for replicate in REPLICATES:
            episodes = artifacts[(seed, replicate)]["episodes"]
            arm = episodes["treatment"] if treatment_arm else ~episodes["treatment"]
            domain = arm & episodes["triggered"]
            assigned += int(arm.sum())
            triggered += int(domain.sum())
            for public_name, field in EVENT_FIELDS.items():
                counts[public_name] += int((domain & episodes[field]).sum())
    return {
        "assigned": assigned,
        "triggered": triggered,
        "trigger_rate_observed": triggered / assigned if assigned else None,
        **{
            name: {
                "count": count,
                "rate_given_trigger_observed": count / triggered if triggered else None,
            }
            for name, count in counts.items()
        },
    }


def _paired_common_exposure(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    total_common = 0
    per_seed: dict[str, Any] = {}
    aggregate_counts = {
        name: {"treatment": 0, "control": 0} for name in EVENT_FIELDS
    }
    for seed in SEEDS:
        artifact_a = artifacts[(seed, "a")]
        artifact_b = artifacts[(seed, "b")]
        a = artifact_a["episodes"]
        b = artifact_b["episodes"]
        rows_a = torch.bincount(
            artifact_a["steps"]["row_env_slot"], minlength=NUM_ENVS
        )
        rows_b = torch.bincount(
            artifact_b["steps"]["row_env_slot"], minlength=NUM_ENVS
        )
        both_triggered = a["triggered"] & b["triggered"]
        common = both_triggered & (rows_a > 0) & (rows_b > 0)
        treatment_is_a = a["treatment"]
        events: dict[str, Any] = {}
        for public_name, field in EVENT_FIELDS.items():
            treatment_outcome = torch.where(treatment_is_a, a[field], b[field])
            control_outcome = torch.where(treatment_is_a, b[field], a[field])
            treatment_count = int((common & treatment_outcome).sum())
            control_count = int((common & control_outcome).sum())
            denominator = int(common.sum())
            aggregate_counts[public_name]["treatment"] += treatment_count
            aggregate_counts[public_name]["control"] += control_count
            events[public_name] = {
                "treatment_count": treatment_count,
                "control_count": control_count,
                "treatment_rate": treatment_count / denominator if denominator else None,
                "control_rate": control_count / denominator if denominator else None,
                "paired_delta": (
                    (treatment_count - control_count) / denominator
                    if denominator
                    else None
                ),
            }
        common_count = int(common.sum())
        total_common += common_count
        trigger_clock_comparable = both_triggered
        per_seed[str(seed)] = {
            "triggered_in_both_replicates": int(both_triggered.sum()),
            "trigger_mask_discordant_slots": int(
                (a["triggered"] ^ b["triggered"]).sum()
            ),
            "trigger_clock_discordant_given_both_triggered": int(
                (
                    trigger_clock_comparable
                    & (a["trigger_step"] != b["trigger_step"])
                ).sum()
            ),
            "common_exposure_opportunity_slots": common_count,
            "excluded_without_common_intervention_opportunity": int(
                (both_triggered & ~common).sum()
            ),
            "events": events,
        }
    aggregate_events = {
        name: {
            **counts,
            "treatment_rate": counts["treatment"] / total_common
            if total_common
            else None,
            "control_rate": counts["control"] / total_common
            if total_common
            else None,
            "paired_delta": (counts["treatment"] - counts["control"])
            / total_common
            if total_common
            else None,
        }
        for name, counts in aggregate_counts.items()
    }
    return {
        "role": "power_and_consistency_diagnostic_only",
        "domain": (
            "same seed/env_slot triggered and possessing at least one eligible "
            "window row in both complementary replicates"
        ),
        "trigger_masks_and_clocks_required_bit_exact": False,
        "common_exposure_opportunity_slots": total_common,
        "minimum_common_exposure_opportunity_slots_per_seed": min(
            row["common_exposure_opportunity_slots"] for row in per_seed.values()
        ),
        "per_seed": per_seed,
        "aggregate_events": aggregate_events,
    }


def _per_run_step_audit(
    artifact: Mapping[str, Any], *, run: str
) -> dict[str, Any]:
    episodes = artifact["episodes"]
    steps = artifact["steps"]
    active = steps["row_residual_active"]
    baseline = steps["row_baseline_action"]
    candidate = steps["row_candidate_action"]
    baseline_hand = baseline[:, artifact_contract.ARM_ACTION_DIM :]
    candidate_hand = candidate[:, artifact_contract.ARM_ACTION_DIM :]
    new_098 = active[:, None] & (candidate_hand.abs() >= 0.98) & (
        baseline_hand.abs() < 0.98
    )
    new_0999 = active[:, None] & (candidate_hand.abs() >= 0.999) & (
        baseline_hand.abs() < 0.999
    )
    active_elements = int(active.sum()) * artifact_contract.HAND_ACTION_DIM
    new_elements = int(new_098.sum())
    saturated_slots = torch.unique(steps["row_env_slot"][new_098.any(dim=-1)])

    delta = steps["row_applied_delta"]
    token_bad = (
        delta[:, : artifact_contract.TOKEN_ACTION_DIM].abs()
        > TOKEN_COMPONENT_CAP + _BUDGET_ATOL
    ).any(dim=-1)
    distal_bad = (
        delta[:, artifact_contract.TOKEN_ACTION_DIM :].abs()
        > DISTAL_COMPONENT_CAP + _BUDGET_ATOL
    ).any(dim=-1)
    l2_bad = torch.linalg.vector_norm(delta.to(torch.float64), dim=-1) > (
        DELTA_L2_CAP + _BUDGET_ATOL
    )
    inactive = ~active
    budget_bad = token_bad | distal_bad | l2_bad | (
        inactive & (delta != 0).any(dim=-1)
    )

    row_treatment = episodes["treatment"][steps["row_env_slot"]]
    expected_delta = artifact_contract.expected_applied_delta().expand(
        delta.shape[0], -1
    )
    expected_delta = torch.where(
        active[:, None], expected_delta, torch.zeros_like(expected_delta)
    )
    algebra_bad = (
        (active != row_treatment)
        | (candidate[:, : artifact_contract.ARM_ACTION_DIM]
           != baseline[:, : artifact_contract.ARM_ACTION_DIM]).any(dim=-1)
        | (steps["row_executed_action"] != candidate).any(dim=-1)
        | (~torch.isclose(delta, expected_delta, rtol=0, atol=1e-6)).any(dim=-1)
        | (inactive & (candidate != baseline).any(dim=-1))
        | steps["row_public_latch_before"]
    )
    return {
        "run": run,
        "residual_active_rows": int(active.sum()),
        "residual_active_hand_elements": active_elements,
        "new_abs_action_ge_098_elements": new_elements,
        "new_abs_action_ge_098_element_fraction": (
            new_elements / active_elements if active_elements else 0.0
        ),
        "episodes_with_new_abs_action_ge_098": int(saturated_slots.numel()),
        "any_new_abs_action_ge_0999": bool(new_0999.any()),
        "component_and_l2_budget_violations": int(budget_bad.sum()),
        "action_algebra_violating_rows": int(algebra_bad.sum()),
        "arm_latch_control_inactive_and_zero_exact": not bool(algebra_bad.any()),
    }


def _gate(
    value: Any, threshold: Any, comparison: str, passed: bool
) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "pass": bool(passed),
    }


def _build_gate_results(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Apply every sealed boundary; useful for exhaustive CPU boundary tests."""

    t = GATE_THRESHOLDS
    results = {
        "primary_conditional_latched_within_32_actions_delta_min_exclusive": _gate(
            metrics["primary_delta"],
            t["primary_conditional_latched_within_32_actions_delta_min_exclusive"],
            ">",
            metrics["primary_delta"]
            > t["primary_conditional_latched_within_32_actions_delta_min_exclusive"],
        ),
        "positive_primary_delta_seeds_min": _gate(
            metrics["positive_primary_seeds"],
            t["positive_primary_delta_seeds_min"],
            ">=",
            metrics["positive_primary_seeds"]
            >= t["positive_primary_delta_seeds_min"],
        ),
        "per_seed_primary_delta_floor": _gate(
            metrics["minimum_per_seed_primary_delta"],
            t["per_seed_primary_delta_floor"],
            ">=",
            metrics["minimum_per_seed_primary_delta"]
            >= t["per_seed_primary_delta_floor"],
        ),
        "conditional_ever_grasped_delta_min": _gate(
            metrics["ever_grasped_delta"],
            t["conditional_ever_grasped_delta_min"],
            ">=",
            metrics["ever_grasped_delta"]
            >= t["conditional_ever_grasped_delta_min"],
        ),
        "conditional_success_delta_floor": _gate(
            metrics["success_delta"],
            t["conditional_success_delta_floor"],
            ">=",
            metrics["success_delta"] >= t["conditional_success_delta_floor"],
        ),
        "common_exposure_opportunity_slots_min_across_validation": _gate(
            metrics["common_exposure_total"],
            t["common_exposure_opportunity_slots_min_across_validation"],
            ">=",
            metrics["common_exposure_total"]
            >= t["common_exposure_opportunity_slots_min_across_validation"],
        ),
        "common_exposure_opportunity_slots_min_per_seed": _gate(
            metrics["common_exposure_min_per_seed"],
            t["common_exposure_opportunity_slots_min_per_seed"],
            ">=",
            metrics["common_exposure_min_per_seed"]
            >= t["common_exposure_opportunity_slots_min_per_seed"],
        ),
        "treatment_triggered_dropped_max": _gate(
            metrics["treatment_triggered_dropped"],
            t["treatment_triggered_dropped_max"],
            "<=",
            metrics["treatment_triggered_dropped"]
            <= t["treatment_triggered_dropped_max"],
        ),
        "treatment_triggered_unsafe_force_max": _gate(
            metrics["treatment_triggered_unsafe_force"],
            t["treatment_triggered_unsafe_force_max"],
            "<=",
            metrics["treatment_triggered_unsafe_force"]
            <= t["treatment_triggered_unsafe_force_max"],
        ),
        "conditional_unlatched_clearance_ge_5cm_delta_max": _gate(
            metrics["unlatched_clearance_ge_5cm_delta"],
            t["conditional_unlatched_clearance_ge_5cm_delta_max"],
            "<=",
            metrics["unlatched_clearance_ge_5cm_delta"]
            <= t["conditional_unlatched_clearance_ge_5cm_delta_max"],
        ),
        "new_abs_action_ge_098_element_fraction_max": _gate(
            metrics["new_abs_action_ge_098_element_fraction"],
            t["new_abs_action_ge_098_element_fraction_max"],
            "<=",
            metrics["new_abs_action_ge_098_element_fraction"]
            <= t["new_abs_action_ge_098_element_fraction_max"],
        ),
        "episodes_with_new_abs_action_ge_098_max_per_run": _gate(
            metrics["max_episodes_with_new_abs_action_ge_098_per_run"],
            t["episodes_with_new_abs_action_ge_098_max_per_run"],
            "<=",
            metrics["max_episodes_with_new_abs_action_ge_098_per_run"]
            <= t["episodes_with_new_abs_action_ge_098_max_per_run"],
        ),
        "any_new_abs_action_ge_0999": _gate(
            metrics["any_new_abs_action_ge_0999"],
            t["any_new_abs_action_ge_0999"],
            "==",
            metrics["any_new_abs_action_ge_0999"]
            is t["any_new_abs_action_ge_0999"],
        ),
        "component_and_l2_budget_violations": _gate(
            metrics["component_and_l2_budget_violations"],
            t["component_and_l2_budget_violations"],
            "==",
            metrics["component_and_l2_budget_violations"]
            == t["component_and_l2_budget_violations"],
        ),
        "action_algebra_arm_latch_control_inactive_and_zero_exact": _gate(
            metrics["action_algebra_exact"],
            t["action_algebra_arm_latch_control_inactive_and_zero_exact"],
            "==",
            metrics["action_algebra_exact"]
            is t["action_algebra_arm_latch_control_inactive_and_zero_exact"],
        ),
        "checkpoint_manifest_assignment_and_fixed_z_identity_exact": _gate(
            metrics["identity_exact"],
            t["checkpoint_manifest_assignment_and_fixed_z_identity_exact"],
            "==",
            metrics["identity_exact"]
            is t["checkpoint_manifest_assignment_and_fixed_z_identity_exact"],
        ),
    }
    if set(results) != set(GATE_THRESHOLDS):
        raise RuntimeError("not every registered fixed-direction gate was evaluated")
    return results


def compute_validation(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    artifacts = validate_validation_artifacts(
        artifacts, repository_root=repository_root
    )
    fixed_receipt = _validate_sealed_fixed_direction(repository_root)

    per_seed_ht: dict[str, Any] = {}
    for seed in SEEDS:
        treatment = torch.stack(
            [artifacts[(seed, replicate)]["episodes"]["treatment"] for replicate in REPLICATES]
        )
        triggered = torch.stack(
            [artifacts[(seed, replicate)]["episodes"]["triggered"] for replicate in REPLICATES]
        )
        per_seed_ht[str(seed)] = {
            public_name: _ht_event(
                treatment,
                triggered,
                torch.stack(
                    [
                        artifacts[(seed, replicate)]["episodes"][field]
                        for replicate in REPLICATES
                    ]
                ),
            )
            for public_name, field in EVENT_FIELDS.items()
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

    paired = _paired_common_exposure(artifacts)
    step_runs = [
        _per_run_step_audit(artifacts[(seed, replicate)], run=f"{seed}{replicate}")
        for seed, replicate in RUN_ORDER
    ]
    active_elements = sum(row["residual_active_hand_elements"] for row in step_runs)
    new_098 = sum(row["new_abs_action_ge_098_elements"] for row in step_runs)
    aggregate_step = {
        "residual_active_hand_elements": active_elements,
        "new_abs_action_ge_098_elements": new_098,
        "new_abs_action_ge_098_element_fraction": (
            new_098 / active_elements if active_elements else 0.0
        ),
        "max_episodes_with_new_abs_action_ge_098_per_run": max(
            row["episodes_with_new_abs_action_ge_098"] for row in step_runs
        ),
        "any_new_abs_action_ge_0999": any(
            row["any_new_abs_action_ge_0999"] for row in step_runs
        ),
        "component_and_l2_budget_violations": sum(
            row["component_and_l2_budget_violations"] for row in step_runs
        ),
        "action_algebra_violating_rows": sum(
            row["action_algebra_violating_rows"] for row in step_runs
        ),
        "arm_latch_control_inactive_and_zero_exact": all(
            row["arm_latch_control_inactive_and_zero_exact"] for row in step_runs
        ),
    }
    treatment_dropped = 0
    treatment_unsafe = 0
    for artifact in artifacts.values():
        episode = artifact["episodes"]
        domain = episode["treatment"] & episode["triggered"]
        treatment_dropped += int((domain & episode["dropped"]).sum())
        treatment_unsafe += int((domain & episode["unsafe_force"]).sum())

    primary_by_seed = {
        str(seed): per_seed_ht[str(seed)]["latched_within_32_actions"]["delta"]
        for seed in SEEDS
    }
    metrics = {
        "primary_delta": equal_seed_ht["latched_within_32_actions"]["delta"],
        "positive_primary_seeds": sum(value > 0.0 for value in primary_by_seed.values()),
        "minimum_per_seed_primary_delta": min(primary_by_seed.values()),
        "ever_grasped_delta": equal_seed_ht["ever_grasped"]["delta"],
        "success_delta": equal_seed_ht["success"]["delta"],
        "common_exposure_total": paired["common_exposure_opportunity_slots"],
        "common_exposure_min_per_seed": paired[
            "minimum_common_exposure_opportunity_slots_per_seed"
        ],
        "treatment_triggered_dropped": treatment_dropped,
        "treatment_triggered_unsafe_force": treatment_unsafe,
        "unlatched_clearance_ge_5cm_delta": equal_seed_ht[
            "unlatched_clearance_ge_5cm"
        ]["delta"],
        "new_abs_action_ge_098_element_fraction": aggregate_step[
            "new_abs_action_ge_098_element_fraction"
        ],
        "max_episodes_with_new_abs_action_ge_098_per_run": aggregate_step[
            "max_episodes_with_new_abs_action_ge_098_per_run"
        ],
        "any_new_abs_action_ge_0999": aggregate_step[
            "any_new_abs_action_ge_0999"
        ],
        "component_and_l2_budget_violations": aggregate_step[
            "component_and_l2_budget_violations"
        ],
        "action_algebra_exact": aggregate_step[
            "arm_latch_control_inactive_and_zero_exact"
        ],
        "identity_exact": True,
    }
    gates = _build_gate_results(metrics)
    all_pass = all(row["pass"] for row in gates.values())
    report = {
        "kind": REPORT_KIND,
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "development_only": True,
        "fixed_direction_receipt": fixed_receipt,
        "registered_development_validation_gates": {
            "all_must_pass": True,
            **GATE_THRESHOLDS,
        },
        "estimand_note": (
            "The primary and secondary conditional effects are per-seed "
            "Horvitz-Thompson treatment-minus-control estimates with known "
            "propensity 0.5 over each run's actually observed pre-treatment "
            "trigger domain, then equally averaged across seeds. They are "
            "distinct from the raw observed arm proportions below."
        ),
        "raw_funnel": {
            "treatment": _raw_arm_funnel(artifacts, treatment_arm=True),
            "control": _raw_arm_funnel(artifacts, treatment_arm=False),
        },
        "per_seed_horvitz_thompson": per_seed_ht,
        "equal_seed_horvitz_thompson": equal_seed_ht,
        "primary": {
            "name": "conditional_latched_within_32_actions_delta",
            "known_propensity": KNOWN_PROPENSITY,
            "per_seed_delta": primary_by_seed,
            "equal_seed_delta": metrics["primary_delta"],
            "positive_delta_seeds": metrics["positive_primary_seeds"],
        },
        "paired_common_exposure_sensitivity": paired,
        "step_audit": {"per_run": step_runs, "aggregate": aggregate_step},
        "safety_counts": {
            "treatment_triggered_dropped": treatment_dropped,
            "treatment_triggered_unsafe_force": treatment_unsafe,
        },
        "gates": gates,
        "all_gates_pass": all_pass,
        "decision": "advance_to_formal_screen" if all_pass else "reject_fixed_direction",
        "direction_generated_or_modified": False,
    }
    _strict_json_bytes(report)
    return report


def analyze_fixed_evidence(
    repository_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifacts, receipts = load_fixed_evidence(repository_root)
    report = compute_validation(artifacts, repository_root=repository_root)
    report["input_artifact_receipts"] = receipts
    _strict_json_bytes(report)
    return report, receipts


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    # The artifact contract's implementation is fsync'd, hard-link atomic, and
    # refuses both pre-existing files and symlink destinations.
    artifact_contract.publish_json_no_clobber(payload, output)


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
