#!/usr/bin/env python3
"""Analyze the sealed Candidate42 public-arm-ramp development A/B.

The causal domain is the first option-active latch-visible pre-action row.
Authority clocks are post-treatment mediators and never select that domain.
"""

from __future__ import annotations

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

import candidate42_public_arm_ramp as ramp
import candidate42_public_arm_ramp_episode as artifact_contract
import candidate42_attempt_transaction as transaction


REPORT_KIND = "pick_tool_candidate42_transactional_public_arm_ramp_development_validation_v1"
FORMAT_VERSION = 1
SEEDS = (347, 348)
REPLICATES = ("a", "b")
RUN_ORDER = ((347, "a"), (347, "b"), (348, "b"), (348, "a"))
INVOCATION_SEQUENCE = (
    (346, "b", 8),
    (347, "a", 64),
    (347, "b", 64),
    (348, "b", 64),
    (348, "a", 64),
)
COMMITTED_NAMESPACE_FIELDS = {
    "status",
    "identity",
    "attempt_number",
    "run_id",
    "canonical_artifact_output",
    "canonical_report_output",
    "canonical_commit_output",
    "artifact_sha256",
    "report_sha256",
    "commit_sha256",
    "retryable_prior_attempts",
}
KNOWN_PROPENSITY = 0.5
UTILITY_HORIZON = 128
STABLE_WINDOW = 15
STABLE_5CM_FRACTION = 0.25
STABLE_5CM_ACTION_HORIZON = 96
INPUT_PATTERN = (
    "logs/flashsac/pick_tool/54_c42_txn_public_arm_ramp_dev_s{seed}_{replicate}/trial.pt"
)
REPORT_OUTPUT = (
    "logs/flashsac/pick_tool/54_c42_txn_public_arm_ramp_development_validation.json"
)

EVENT_FIELDS = (
    "stable_transport_restricted_mean",
    "terminal_utility",
    "success",
    "ever_true_clearance_ge_20cm",
    "stable_held_5cm_by_action96",
    "never_stable_held_5cm_by_action96",
    "ever_grasped",
    "unlatched_clearance_ge_5cm",
)

GATE_NAMES = (
    "primary_conditional_safe_terminal_utility_delta_min_exclusive",
    "positive_primary_delta_seeds_min",
    "per_seed_primary_delta_floor",
    "conditional_stable_transport_restricted_mean_delta_floor",
    "per_seed_stable_transport_delta_floor",
    "conditional_unlatched_clearance_ge_5cm_delta_max",
    "paired_common_eligibility_unlatched_clearance_ge_5cm_delta_max",
    "conditional_success_delta_floor",
    "conditional_true_clearance_ge_20cm_delta_floor",
    "conditional_stable_held_5cm_by_action96_delta_floor",
    "conditional_ever_grasped_delta_floor",
    "eligible_opportunity_slots_min_across_validation",
    "eligible_opportunity_slots_min_per_seed",
    "eligible_opportunity_slots_min_per_seed_arm",
    "paired_common_eligible_slots_min_across_validation",
    "paired_common_eligible_slots_min_per_seed",
    "treatment_eligible_positive_authority_by_action16_min_across_validation",
    "treatment_eligible_positive_authority_by_action16_min_per_seed",
    "treatment_eligible_full_authority_by_action32_min_across_validation",
    "treatment_eligible_full_authority_by_action32_min_per_seed",
    "treatment_partial_authority_rows_min_across_validation",
    "treatment_eligible_unlatched_clearance_ge_5cm_max",
    "pooled_option_active_pre_eligible_unlatched_clearance_ge_5cm_max",
    "treatment_eligible_dropped_max",
    "treatment_eligible_unsafe_force_max",
    "fixed_residual_component_and_l2_budget_violations",
    "any_new_abs_action_ge_0999",
    "pre_eligibility_action_and_state_machine_parity_exact",
    "hand14_invariance_exact",
    "linear_authority_clock_and_action_algebra_exact",
    "control_candidate39_route_exact",
    "checkpoint_manifest_assignment_and_source_identity_exact",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    checked = _strict_json(payload)
    return (json.dumps(checked, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    return path


def _canonical_invocation_stem(
    root: Path, seed: int, replicate: str, num_envs: int
) -> Path:
    if (seed, replicate, num_envs) == INVOCATION_SEQUENCE[0]:
        return (root / artifact_contract.SMOKE_ARTIFACT_PATH).with_suffix("").resolve()
    return (
        root
        / f"logs/flashsac/pick_tool/54_c42_txn_public_arm_ramp_dev_s{seed}_{replicate}/trial"
    ).resolve()


def _sha_receipt(value: Any, name: str, *, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Candidate42 {name} is not a lowercase hex receipt")
    return value


def _transaction_run_metadata(
    stem: Path, committed: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Load the already-validated 00 identity for an accepted commit."""

    context = transaction.open_attempt(stem, int(committed["attempt_number"]))
    identity = context.identity.as_dict()
    if identity != committed["identity"]:
        raise ValueError("Candidate42 committed and intent identities differ")
    metadata = getattr(context, "run_metadata", None)
    if metadata is None:
        metadata = getattr(context, "run_identity", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("Candidate42 intent omitted immutable run metadata")
    return metadata


def validate_committed_invocation(
    root: Path,
    *,
    seed: int,
    replicate: str,
    num_envs: int,
    collection_commit: str,
    preregistration_commit: str | None = None,
) -> dict[str, Any]:
    """Validate a whole 00--70 namespace before exposing canonical bytes."""

    _sha_receipt(collection_commit, "collection commit", length=40)
    stem = _canonical_invocation_stem(root, seed, replicate, num_envs)
    paths = transaction.TransactionPaths.from_output_stem(stem)
    committed = transaction.validate_committed_namespace(paths)
    if not isinstance(committed, Mapping) or set(committed) != COMMITTED_NAMESPACE_FIELDS:
        raise ValueError("Candidate42 committed namespace schema changed")
    value = dict(committed)
    if value["status"] != "committed":
        raise ValueError("Candidate42 namespace is not durably committed")
    identity = value["identity"]
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"run_id", "attempt_id", "attempt_number"}
        or identity["run_id"] != value["run_id"]
        or identity["attempt_number"] != value["attempt_number"]
        or identity["attempt_id"]
        != transaction.compute_attempt_id(value["run_id"], value["attempt_number"])
    ):
        raise ValueError("Candidate42 committed transaction identity changed")
    expected_paths = {
        "canonical_artifact_output": str(Path(f"{stem}.pt")),
        "canonical_report_output": str(Path(f"{stem}.json")),
        "canonical_commit_output": str(Path(f"{stem}.commit.json")),
    }
    for name, expected in expected_paths.items():
        path = Path(value[name])
        if value[name] != expected or path.is_symlink() or not path.is_file():
            raise ValueError(f"Candidate42 {name} is not the registered regular file")
    run_metadata = _transaction_run_metadata(stem, value)
    preregistration_tag_commit = _sha_receipt(
        run_metadata.get("preregistration_tag_commit"),
        "preregistration tag commit",
        length=40,
    )
    if preregistration_commit is not None and preregistration_tag_commit != (
        _sha_receipt(
            preregistration_commit,
            "registered preregistration commit",
            length=40,
        )
    ):
        raise ValueError(
            "Candidate42 intent preregistration tag differs from the sealed receipt"
        )
    expected_assignment = ramp.assignment_mask_sha256(
        ramp.exact_balanced_treatment_mask(
            seed=seed, num_envs=num_envs, replicate=replicate
        )
    )
    expected_metadata = {
        "collection_commit": collection_commit,
        "assignment_mask_sha256": expected_assignment,
        "seed": seed,
        "replicate": replicate,
        "num_envs": num_envs,
        **expected_paths,
    }
    for name, expected in expected_metadata.items():
        if run_metadata.get(name) != expected:
            raise ValueError(f"Candidate42 intent run metadata changed at {name}")
    arguments = run_metadata.get("argument_receipt")
    if not isinstance(arguments, Mapping) or (
        arguments.get("seed"),
        arguments.get("replicate"),
        arguments.get("num_envs"),
        arguments.get("output_stem"),
    ) != (seed, replicate, num_envs, str(stem)):
        raise ValueError("Candidate42 intent argument receipt changed")
    for name, path_name in (
        ("artifact_sha256", "canonical_artifact_output"),
        ("report_sha256", "canonical_report_output"),
        ("commit_sha256", "canonical_commit_output"),
    ):
        _sha_receipt(value[name], name, length=64)
        if artifact_contract.sha256_file(Path(value[path_name])) != value[name]:
            raise ValueError(f"Candidate42 committed {name} differs from canonical bytes")
    prior = value["retryable_prior_attempts"]
    if (
        not isinstance(prior, list)
        or len(prior) != value["attempt_number"] - 1
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"run_id", "attempt_id", "attempt_number"}
            or item["run_id"] != value["run_id"]
            or item["attempt_number"] != index
            or item["attempt_id"]
            != transaction.compute_attempt_id(value["run_id"], index)
            for index, item in enumerate(prior, start=1)
        )
    ):
        raise ValueError("Candidate42 retryable prior-attempt summary changed")
    value["run_metadata"] = dict(run_metadata)
    return value


def validate_committed_invocations(
    root: Path, *, collection_commit: str, preregistration_commit: str
) -> dict[tuple[int, str], dict[str, Any]]:
    """Validate all five registered transaction commits before analysis."""

    return {
        (seed, replicate): validate_committed_invocation(
            root,
            seed=seed,
            replicate=replicate,
            num_envs=num_envs,
            collection_commit=collection_commit,
            preregistration_commit=preregistration_commit,
        )
        for seed, replicate, num_envs in INVOCATION_SEQUENCE
    }


def _read_plan(
    repository_root: Path | None = None, *, require_sealed: bool
) -> tuple[dict[str, Any], str]:
    root = _root() if repository_root is None else Path(repository_root)
    path = _regular_file(root / artifact_contract.VALIDATION_PLAN, "Candidate42 plan")
    raw = path.read_bytes()
    plan = artifact_contract.validate_sealed_plan(path, require_sealed=require_sealed)
    return plan, _sha256_bytes(raw)


def registered_gate_thresholds(
    repository_root: Path | None = None, *, require_sealed: bool = False
) -> dict[str, Any]:
    plan, _ = _read_plan(repository_root, require_sealed=require_sealed)
    gates = plan.get("development_validation_gates")
    if not isinstance(gates, dict) or gates.get("all_must_pass") is not True:
        raise ValueError("Candidate42 plan omitted all-must-pass gates")
    values = {key: value for key, value in gates.items() if key != "all_must_pass"}
    if set(values) != set(GATE_NAMES):
        raise ValueError("Candidate42 registered gate set changed")
    return values


def _independent_rank(*, seed: int, num_envs: int) -> torch.Tensor:
    ranked = sorted(
        range(num_envs),
        key=lambda slot: (
            hashlib.sha256(
                f"{ramp.ASSIGNMENT_SALT}\0rank\0{seed}\0{slot}".encode("utf-8")
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
    rank = _independent_rank(seed=seed, num_envs=num_envs)
    mask_a = rank < (num_envs // 2)
    return mask_a if replicate == "a" else ~mask_a


def _without_run_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "seed", "replicate", "assignment_mask_sha256", "runtime",
        "source_manifest_sha256", "runtime_asset_manifest_sha256",
    }
    result = {key: value for key, value in metadata.items() if key not in excluded}
    runtime = metadata.get("runtime")
    if isinstance(runtime, Mapping):
        result["runtime_without_seed"] = {
            key: value for key, value in runtime.items() if key != "seed"
        }
    return result


def validate_validation_artifacts(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    require_sealed_plan: bool = False,
) -> dict[tuple[int, str], dict[str, Any]]:
    if set(artifacts) != set(RUN_ORDER):
        raise ValueError("Candidate42 fixed run set changed")
    checked: dict[tuple[int, str], dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for seed, replicate in RUN_ORDER:
        value = artifact_contract.validate_artifact(
            artifacts[(seed, replicate)], require_sealed_plan=require_sealed_plan
        )
        metadata = value["metadata"]
        episodes = value["episodes"]
        if (metadata["seed"], metadata["replicate"], metadata["num_envs"]) != (
            seed, replicate, 64,
        ):
            raise ValueError("Candidate42 artifact run identity changed")
        expected_rank = _independent_rank(seed=seed, num_envs=64)
        expected_mask = _independent_treatment_mask(
            seed=seed, num_envs=64, replicate=replicate
        )
        if not torch.equal(episodes["assignment_rank"], expected_rank):
            raise ValueError("Candidate42 independent assignment rank changed")
        if not torch.equal(episodes["treatment"], expected_mask):
            raise ValueError("Candidate42 independent treatment mask changed")
        common = _without_run_identity(metadata)
        if reference is None:
            reference = common
        elif common != reference:
            raise ValueError("Candidate42 artifacts do not share one authority")
        checked[(seed, replicate)] = value
    for seed in SEEDS:
        a = checked[(seed, "a")]["episodes"]["treatment"]
        b = checked[(seed, "b")]["episodes"]["treatment"]
        if not torch.equal(a, ~b):
            raise ValueError("Candidate42 assignments are not exact complements")
    return checked


def stable_transport_outcomes(
    artifact: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
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
        within = ages < UTILITY_HORIZON
        ages = ages[within]
        if ages.numel() == 0 or not torch.equal(
            ages, torch.arange(int(ages.numel()), dtype=torch.int64)
        ):
            raise ValueError(f"eligible Candidate42 slot {slot} utility trace has a gap")
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
        observed_values[slot, :n] = x
        observed_mask[slot, :n] = True
        for end in range(STABLE_WINDOW - 1, n):
            stable_values[slot, end] = x[end - STABLE_WINDOW + 1 : end + 1].min()
        if bool(episodes["success"][slot]):
            terminal_age = int(
                episodes["terminal_step"][slot] - episodes["first_eligible_step"][slot]
            )
            if terminal_age < 0:
                raise ValueError("Candidate42 success precedes eligibility")
            stable_values[slot, min(UTILITY_HORIZON, terminal_age + 1) :] = 1.0
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
    return {
        "stable_transport_restricted_mean": stable["stable_transport_restricted_mean"],
        "terminal_utility": (
            episodes["success"].to(torch.float64)
            - episodes["unlatched_clearance_ge_5cm"].to(torch.float64)
        ),
        "success": episodes["success"],
        "ever_true_clearance_ge_20cm": episodes["ever_clearance_ge_20cm"],
        "stable_held_5cm_by_action96": stable["stable_held_5cm_by_action96"],
        "never_stable_held_5cm_by_action96": stable[
            "never_stable_held_5cm_by_action96"
        ],
        "ever_grasped": episodes["ever_grasped"],
        "unlatched_clearance_ge_5cm": episodes["unlatched_clearance_ge_5cm"],
    }


def _ht_outcome(
    treatment: torch.Tensor, domain: torch.Tensor, outcome: torch.Tensor
) -> dict[str, float | int]:
    numeric = outcome.to(torch.float64)
    if treatment.shape != domain.shape or numeric.shape != domain.shape:
        raise ValueError("Candidate42 HT shapes differ")
    if not bool(torch.isfinite(numeric).all()):
        raise FloatingPointError("Candidate42 HT outcome is non-finite")
    denominator = int(domain.sum())
    treatment_rows = int((treatment & domain).sum())
    control_rows = int(((~treatment) & domain).sum())
    if denominator <= 0 or treatment_rows <= 0 or control_rows <= 0:
        raise ValueError("a Candidate42 seed lacks eligible support in one arm")
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
                "mean_given_eligibility_observed": value / eligible if eligible else None,
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
            treatment_sum = float(treatment_outcome.to(torch.float64)[common].sum())
            control_sum = float(control_outcome.to(torch.float64)[common].sum())
            aggregate[name]["treatment_sum"] += treatment_sum
            aggregate[name]["control_sum"] += control_sum
            events[name] = {
                "treatment_sum": treatment_sum,
                "control_sum": control_sum,
                "treatment_mean": treatment_sum / count if count else None,
                "control_mean": control_sum / count if count else None,
                "paired_delta": (treatment_sum - control_sum) / count if count else None,
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
            if total_common else None,
            "control_mean": sums["control_sum"] / total_common
            if total_common else None,
            "paired_delta": (
                (sums["treatment_sum"] - sums["control_sum"]) / total_common
                if total_common else None
            ),
        }
        for name, sums in aggregate.items()
    }
    return {
        "role": (
            "pooled same-seed same-env-slot sensitivity; only the registered "
            "paired U5 row is an acceptance gate"
        ),
        "domain": "same seed/env_slot eligible before Candidate42 action in both replicates",
        "common_eligible_slots": total_common,
        "minimum_common_eligible_slots_per_seed": min(
            row["eligible_in_both_replicates"] for row in per_seed.values()
        ),
        "per_seed": per_seed,
        "aggregate_events": aggregate_events,
    }


_VIOLATION_FIELDS = (
    "pre_eligibility_action_violations",
    "hand_invariance_violations",
    "treatment_arm_ramp_violations",
    "authority_clock_violations",
    "control_route_violations",
    "task_action_reconstruction_violations",
    "fixed_residual_budget_violations",
    "action_bound_violations",
)


def _action_clock_audit(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    sums = {name: 0 for name in _VIOLATION_FIELDS}
    any_new_0999 = False
    partial_rows = 0
    per_run: list[dict[str, Any]] = []
    for seed, replicate in RUN_ORDER:
        artifact = artifacts[(seed, replicate)]
        episodes = artifact["episodes"]
        steps = artifact["steps"]
        row = {name: int(episodes[name].sum()) for name in _VIOLATION_FIELDS}
        row["any_new_abs_action_ge_0999"] = bool(
            episodes["new_abs_action_ge_0999"].any()
        )
        row["run"] = f"{seed}{replicate}"
        stable = steps["row_stable_current"]
        before = steps["row_stable_count_before"]
        ramp_scale = torch.where(
            stable,
            before.to(torch.float32) / torch.tensor(
                15.0, dtype=torch.float32, device=before.device
            ),
            torch.zeros_like(before, dtype=torch.float32),
        )
        overlay = (
            steps["row_treatment"]
            & steps["row_activated"]
            & steps["row_option_active"]
        )
        expected_scale = torch.where(
            overlay, ramp_scale, torch.ones_like(ramp_scale)
        )
        scale_bad = steps["row_authority_scale"] != expected_scale
        common = steps["row_common_action"]
        requested = steps["row_requested_action"]
        expected_arm = torch.where(
            overlay.unsqueeze(-1),
            ramp_scale.unsqueeze(-1) * common[:, : artifact_contract.ARM_ACTION_DIM],
            common[:, : artifact_contract.ARM_ACTION_DIM],
        )
        arm_bad = (requested[:, : artifact_contract.ARM_ACTION_DIM] != expected_arm).any(
            dim=-1
        )
        hand_bad = (
            requested[:, artifact_contract.ARM_ACTION_DIM :]
            != common[:, artifact_contract.ARM_ACTION_DIM :]
        ).any(dim=-1)
        if int((scale_bad | arm_bad).sum()) != row[
            "treatment_arm_ramp_violations"
        ]:
            raise ValueError("online and trace Candidate42 ramp audits disagree")
        if int(hand_bad.sum()) != row["hand_invariance_violations"]:
            raise ValueError("online and trace Candidate42 hand audits disagree")
        derived_partial = int((overlay & (ramp_scale > 0.0) & (ramp_scale < 1.0)).sum())
        if derived_partial != int(episodes["partial_authority_rows"].sum()):
            raise ValueError("online and trace Candidate42 partial-row counts disagree")
        pre = artifact["pre_steps"]
        pre_new_saturation = (
            (pre["pre_requested_action"].abs() >= 0.999)
            & (pre["pre_baseline_action"].abs() < 0.999)
        )
        new_saturation = (requested.abs() >= 0.999) & (common.abs() < 0.999)
        derived_new = bool(pre_new_saturation.any()) or bool(new_saturation.any())
        if derived_new != row["any_new_abs_action_ge_0999"]:
            raise ValueError("online and trace Candidate42 saturation audits disagree")
        for name in _VIOLATION_FIELDS:
            sums[name] += row[name]
        partial_rows += derived_partial
        any_new_0999 |= row["any_new_abs_action_ge_0999"]
        per_run.append(row)
    exact = {
        "pre_eligibility_action_and_state_machine_parity_exact": (
            sums["pre_eligibility_action_violations"] == 0
        ),
        "hand14_invariance_exact": sums["hand_invariance_violations"] == 0,
        "control_candidate39_route_exact": sums["control_route_violations"] == 0,
        "linear_authority_clock_and_action_algebra_exact": (
            sums["treatment_arm_ramp_violations"] == 0
            and sums["authority_clock_violations"] == 0
            and sums["task_action_reconstruction_violations"] == 0
            and sums["action_bound_violations"] == 0
        ),
    }
    return {
        "per_run": per_run,
        "aggregate": {
            **sums,
            "treatment_partial_authority_rows": partial_rows,
            "any_new_abs_action_ge_0999": any_new_0999,
            **exact,
        },
    }


def _gate(value: Any, threshold: Any, comparison: str, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "pass": bool(passed),
    }


def _build_gate_results(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    specs = {
        "primary_conditional_safe_terminal_utility_delta_min_exclusive": ("primary_delta", ">", lambda v, x: v > x),
        "positive_primary_delta_seeds_min": ("positive_primary_seeds", ">=", lambda v, x: v >= x),
        "per_seed_primary_delta_floor": ("minimum_primary_seed_delta", ">=", lambda v, x: v >= x),
        "conditional_stable_transport_restricted_mean_delta_floor": ("transport_delta", ">=", lambda v, x: v >= x),
        "per_seed_stable_transport_delta_floor": ("minimum_transport_seed_delta", ">=", lambda v, x: v >= x),
        "conditional_unlatched_clearance_ge_5cm_delta_max": ("unlatched5_delta", "<=", lambda v, x: v <= x),
        "paired_common_eligibility_unlatched_clearance_ge_5cm_delta_max": ("paired_unlatched5_delta", "<=", lambda v, x: v <= x),
        "conditional_success_delta_floor": ("success_delta", ">=", lambda v, x: v >= x),
        "conditional_true_clearance_ge_20cm_delta_floor": ("true20_delta", ">=", lambda v, x: v >= x),
        "conditional_stable_held_5cm_by_action96_delta_floor": ("stable5_delta", ">=", lambda v, x: v >= x),
        "conditional_ever_grasped_delta_floor": ("ever_grasped_delta", ">=", lambda v, x: v >= x),
        "eligible_opportunity_slots_min_across_validation": ("eligible_total", ">=", lambda v, x: v >= x),
        "eligible_opportunity_slots_min_per_seed": ("eligible_min_seed", ">=", lambda v, x: v >= x),
        "eligible_opportunity_slots_min_per_seed_arm": ("eligible_min_seed_arm", ">=", lambda v, x: v >= x),
        "paired_common_eligible_slots_min_across_validation": ("paired_common_total", ">=", lambda v, x: v >= x),
        "paired_common_eligible_slots_min_per_seed": ("paired_common_min_seed", ">=", lambda v, x: v >= x),
        "treatment_eligible_positive_authority_by_action16_min_across_validation": ("positive16_total", ">=", lambda v, x: v >= x),
        "treatment_eligible_positive_authority_by_action16_min_per_seed": ("positive16_min_seed", ">=", lambda v, x: v >= x),
        "treatment_eligible_full_authority_by_action32_min_across_validation": ("full32_total", ">=", lambda v, x: v >= x),
        "treatment_eligible_full_authority_by_action32_min_per_seed": ("full32_min_seed", ">=", lambda v, x: v >= x),
        "treatment_partial_authority_rows_min_across_validation": ("partial_rows", ">=", lambda v, x: v >= x),
        "treatment_eligible_unlatched_clearance_ge_5cm_max": ("treatment_unlatched5", "<=", lambda v, x: v <= x),
        "pooled_option_active_pre_eligible_unlatched_clearance_ge_5cm_max": ("pooled_preeligible_unlatched5", "<=", lambda v, x: v <= x),
        "treatment_eligible_dropped_max": ("treatment_dropped", "<=", lambda v, x: v <= x),
        "treatment_eligible_unsafe_force_max": ("treatment_unsafe", "<=", lambda v, x: v <= x),
        "fixed_residual_component_and_l2_budget_violations": ("fixed_budget_violations", "==", lambda v, x: v == x),
        "any_new_abs_action_ge_0999": ("any_new_0999", "==", lambda v, x: v is x),
        "pre_eligibility_action_and_state_machine_parity_exact": ("preeligibility_exact", "==", lambda v, x: v is x),
        "hand14_invariance_exact": ("hand_exact", "==", lambda v, x: v is x),
        "linear_authority_clock_and_action_algebra_exact": ("ramp_exact", "==", lambda v, x: v is x),
        "control_candidate39_route_exact": ("control_exact", "==", lambda v, x: v is x),
        "checkpoint_manifest_assignment_and_source_identity_exact": ("identity_exact", "==", lambda v, x: v is x),
    }
    if set(specs) != set(GATE_NAMES) or set(thresholds) != set(GATE_NAMES):
        raise RuntimeError("not every Candidate42 gate has executable semantics")
    return {
        name: _gate(metrics[metric], thresholds[name], comparison, predicate(metrics[metric], thresholds[name]))
        for name, (metric, comparison, predicate) in specs.items()
    }


def _authority_by_age_count(
    episode: Mapping[str, torch.Tensor],
    domain: torch.Tensor,
    *,
    clock_field: str,
    maximum_inclusive_age: int,
) -> int:
    """Count domain rows whose first authority event meets an inclusive age bound."""

    first_eligible = episode["first_eligible_step"]
    first_event = episode[clock_field]
    if (
        not isinstance(domain, torch.Tensor)
        or domain.dtype != torch.bool
        or domain.shape != first_eligible.shape
        or first_event.shape != first_eligible.shape
    ):
        raise ValueError("Candidate42 authority deadline tensors differ")
    if not isinstance(maximum_inclusive_age, int) or isinstance(
        maximum_inclusive_age, bool
    ) or maximum_inclusive_age < 0:
        raise ValueError("Candidate42 authority deadline age is invalid")
    age = first_event - first_eligible
    return int(
        (
            domain
            & (age >= 0)
            & (age <= maximum_inclusive_age)
        ).sum()
    )


def compute_validation(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    require_sealed_plan: bool = False,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    checked = validate_validation_artifacts(
        artifacts, require_sealed_plan=require_sealed_plan
    )
    thresholds = registered_gate_thresholds(
        repository_root, require_sealed=require_sealed_plan
    )
    outcomes = {key: _outcome_table(value) for key, value in checked.items()}
    per_seed_ht: dict[str, Any] = {}
    per_seed_eligibility: dict[str, Any] = {}
    for seed in SEEDS:
        treatment = torch.stack(
            [checked[(seed, replicate)]["episodes"]["treatment"] for replicate in REPLICATES]
        )
        eligible = torch.stack(
            [checked[(seed, replicate)]["episodes"]["eligible"] for replicate in REPLICATES]
        )
        per_seed_ht[str(seed)] = {
            name: _ht_outcome(
                treatment,
                eligible,
                torch.stack(
                    [outcomes[(seed, replicate)][name] for replicate in REPLICATES]
                ),
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
            ) / len(SEEDS)
            for quantity in ("treatment", "control", "delta")
        }
        for name in EVENT_FIELDS
    }
    primary_by_seed = {
        str(seed): per_seed_ht[str(seed)]["terminal_utility"]["delta"]
        for seed in SEEDS
    }
    transport_by_seed = {
        str(seed): per_seed_ht[str(seed)][
            "stable_transport_restricted_mean"
        ]["delta"]
        for seed in SEEDS
    }
    paired = _paired_common_eligibility(checked)
    action_audit = _action_clock_audit(checked)
    audit = action_audit["aggregate"]

    treatment_dropped = 0
    treatment_unsafe = 0
    treatment_unlatched5 = 0
    pooled_preeligible_unlatched5 = 0
    positive16_by_seed: dict[str, int] = {}
    full32_by_seed: dict[str, int] = {}
    for seed in SEEDS:
        positive16 = 0
        full32 = 0
        for replicate in REPLICATES:
            episode = checked[(seed, replicate)]["episodes"]
            domain = episode["treatment"] & episode["eligible"]
            treatment_dropped += int((domain & episode["dropped"]).sum())
            treatment_unsafe += int((domain & episode["unsafe_force"]).sum())
            treatment_unlatched5 += int(
                (domain & episode["unlatched_clearance_ge_5cm"]).sum()
            )
            pooled_preeligible_unlatched5 += int(
                episode[
                    "option_active_pre_eligible_unlatched_clearance_ge_5cm"
                ].sum()
            )
            positive16 += _authority_by_age_count(
                episode,
                domain,
                clock_field="first_positive_authority_step",
                maximum_inclusive_age=15,
            )
            full32 += _authority_by_age_count(
                episode,
                domain,
                clock_field="first_full_authority_step",
                maximum_inclusive_age=31,
            )
        positive16_by_seed[str(seed)] = positive16
        full32_by_seed[str(seed)] = full32

    eligible_counts = [row["eligible_rows"] for row in per_seed_eligibility.values()]
    eligible_arm_counts = [
        row[name]
        for row in per_seed_eligibility.values()
        for name in ("treatment_eligible_rows", "control_eligible_rows")
    ]
    paired_unlatched5 = paired["aggregate_events"][
        "unlatched_clearance_ge_5cm"
    ]["paired_delta"]
    paired_unlatched5_for_gate = (
        1.0 if paired_unlatched5 is None else paired_unlatched5
    )
    metrics = {
        "primary_delta": equal_seed_ht["terminal_utility"]["delta"],
        "positive_primary_seeds": sum(value > 0.0 for value in primary_by_seed.values()),
        "minimum_primary_seed_delta": min(primary_by_seed.values()),
        "transport_delta": equal_seed_ht["stable_transport_restricted_mean"]["delta"],
        "minimum_transport_seed_delta": min(transport_by_seed.values()),
        "unlatched5_delta": equal_seed_ht["unlatched_clearance_ge_5cm"]["delta"],
        "paired_unlatched5_delta": paired_unlatched5_for_gate,
        "success_delta": equal_seed_ht["success"]["delta"],
        "true20_delta": equal_seed_ht["ever_true_clearance_ge_20cm"]["delta"],
        "stable5_delta": equal_seed_ht["stable_held_5cm_by_action96"]["delta"],
        "ever_grasped_delta": equal_seed_ht["ever_grasped"]["delta"],
        "eligible_total": sum(eligible_counts),
        "eligible_min_seed": min(eligible_counts),
        "eligible_min_seed_arm": min(eligible_arm_counts),
        "paired_common_total": paired["common_eligible_slots"],
        "paired_common_min_seed": paired["minimum_common_eligible_slots_per_seed"],
        "positive16_total": sum(positive16_by_seed.values()),
        "positive16_min_seed": min(positive16_by_seed.values()),
        "full32_total": sum(full32_by_seed.values()),
        "full32_min_seed": min(full32_by_seed.values()),
        "partial_rows": audit["treatment_partial_authority_rows"],
        "treatment_unlatched5": treatment_unlatched5,
        "pooled_preeligible_unlatched5": pooled_preeligible_unlatched5,
        "treatment_dropped": treatment_dropped,
        "treatment_unsafe": treatment_unsafe,
        "fixed_budget_violations": audit["fixed_residual_budget_violations"],
        "any_new_0999": audit["any_new_abs_action_ge_0999"],
        "preeligibility_exact": audit[
            "pre_eligibility_action_and_state_machine_parity_exact"
        ],
        "hand_exact": audit["hand14_invariance_exact"],
        "ramp_exact": audit["linear_authority_clock_and_action_algebra_exact"],
        "control_exact": audit["control_candidate39_route_exact"],
        "identity_exact": True,
    }
    gates = _build_gate_results(metrics, thresholds)
    all_pass = all(row["pass"] for row in gates.values())
    plan, plan_sha = _read_plan(
        repository_root, require_sealed=require_sealed_plan
    )
    report = {
        "kind": REPORT_KIND,
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "development_only": True,
        "validation_plan": artifact_contract.VALIDATION_PLAN,
        "validation_plan_sha256": plan_sha,
        "preregistration_receipt": plan["preregistration_receipt"],
        "registered_development_validation_gates": {
            "all_must_pass": True,
            **thresholds,
        },
        "causal_estimand_note": (
            "All HT effects use first option-active latch-visible pre-action "
            "eligibility with known propensity 0.5. Authority onset/full clocks "
            "are post-treatment mediators and never select the domain."
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
            "treatment": _raw_arm_funnel(checked, treatment_arm=True),
            "control": _raw_arm_funnel(checked, treatment_arm=False),
        },
        "eligibility_support": per_seed_eligibility,
        "per_seed_horvitz_thompson": per_seed_ht,
        "equal_seed_horvitz_thompson": equal_seed_ht,
        "primary": {
            "name": "conditional_safe_terminal_utility_delta",
            "known_propensity": KNOWN_PROPENSITY,
            "per_seed_delta": primary_by_seed,
            "equal_seed_delta": metrics["primary_delta"],
            "positive_delta_seeds": metrics["positive_primary_seeds"],
        },
        "stable_transport_noninferiority": {
            "per_seed_delta": transport_by_seed,
            "equal_seed_delta": metrics["transport_delta"],
        },
        "paired_common_eligibility_sensitivity": paired,
        "action_and_clock_audit": action_audit,
        "treatment_counts": {
            "eligible_dropped": treatment_dropped,
            "eligible_unsafe_force": treatment_unsafe,
            "eligible_unlatched_clearance_ge_5cm": treatment_unlatched5,
            "positive_authority_by_action16": {
                "total": sum(positive16_by_seed.values()),
                "per_seed": positive16_by_seed,
            },
            "full_authority_by_action32": {
                "total": sum(full32_by_seed.values()),
                "per_seed": full32_by_seed,
            },
            "partial_authority_rows": audit["treatment_partial_authority_rows"],
        },
        "pooled_option_active_pre_eligible_unlatched_clearance_ge_5cm": (
            pooled_preeligible_unlatched5
        ),
        "gates": gates,
        "all_gates_pass": all_pass,
        "decision": "advance_to_formal_screen" if all_pass else "reject_candidate42",
        "authority_onset_or_full_used_as_causal_domain": False,
        "network_updates": 0,
    }
    _strict_json_bytes(report)
    return report


def _torch_load_weights_only(raw: bytes) -> Any:
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def _read_artifact(
    path: Path, *, require_sealed_plan: bool
) -> tuple[dict[str, Any], str]:
    raw = _regular_file(path, "Candidate42 artifact").read_bytes()
    value = _torch_load_weights_only(raw)
    checked = artifact_contract.validate_artifact(
        value, require_sealed_plan=require_sealed_plan
    )
    return checked, _sha256_bytes(raw)


def _validate_registered_smoke(
    root: Path,
    committed: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_path = _regular_file(
        Path(committed["canonical_artifact_output"]),
        "Candidate42 smoke artifact",
    )
    report_path = _regular_file(
        Path(committed["canonical_report_output"]),
        "Candidate42 smoke report",
    )
    raw = artifact_path.read_bytes()
    artifact = artifact_contract.validate_artifact(
        _torch_load_weights_only(raw), require_sealed_plan=True
    )
    report = artifact_contract.validate_final_report(
        json.loads(report_path.read_text(encoding="utf-8")),
        artifact,
        require_sealed_plan=True,
    )
    digest = _sha256_bytes(raw)
    if (
        digest != committed["artifact_sha256"]
        or artifact_contract.sha256_file(report_path) != committed["report_sha256"]
        or report["artifact_sha256"] != digest
        or report["artifact_size"] != len(raw)
        or report["run_id"] != committed["run_id"]
        or report["attempt_id"] != committed["identity"]["attempt_id"]
        or report["attempt_number"] != committed["attempt_number"]
    ):
        raise ValueError("Candidate42 smoke report does not bind artifact bytes")
    metadata = artifact["metadata"]
    if (metadata["seed"], metadata["replicate"], metadata["num_envs"]) != (
        346, "b", 8,
    ):
        raise ValueError("Candidate42 smoke identity changed")
    violations = sum(int(artifact["episodes"][name].sum()) for name in _VIOLATION_FIELDS)
    if violations or bool(artifact["episodes"]["new_abs_action_ge_0999"].any()):
        raise ValueError("Candidate42 smoke failed action/trace algebra")
    run_metadata = committed["run_metadata"]
    for name, actual in (
        ("source_manifest_sha256", metadata["source_manifest_sha256"]),
        ("runtime_asset_manifest_sha256", metadata["runtime_asset_manifest_sha256"]),
        ("checkpoint_manifest_sha256", report["checkpoint_manifest_sha256"]),
    ):
        if run_metadata.get(name) != actual:
            raise ValueError(f"Candidate42 smoke intent differs at {name}")
    return {
        "run": "346b",
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": digest,
        "report": str(report_path.resolve()),
        "report_sha256": artifact_contract.sha256_file(report_path),
        "commit": committed["canonical_commit_output"],
        "commit_sha256": committed["commit_sha256"],
        "run_id": committed["run_id"],
        "attempt_id": committed["identity"]["attempt_id"],
        "attempt_number": committed["attempt_number"],
        "retryable_prior_attempts": committed["retryable_prior_attempts"],
        "registered_checks_passed": True,
        "outcomes_used_to_change_plan": False,
    }


def _validate_collection_head(root: Path, plan: Mapping[str, Any]) -> str:
    seal = plan.get("collection_seal")
    if not isinstance(seal, Mapping) or seal.get("tag") != (
        "pick-tool-candidate42-transactional-arm-ramp-validation-v1-20260722"
    ):
        raise ValueError("Candidate42 collection seal tag changed")
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    tagged = subprocess.run(
        ("git", "rev-parse", f"{seal['tag']}^{{}}"), cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if head != tagged:
        raise ValueError("HEAD is not the exact Candidate42 collection seal")
    prereg = plan.get("preregistration_receipt")
    if not isinstance(prereg, Mapping) or prereg.get("tag") != (
        "pick-tool-candidate42-transactional-arm-ramp-plan-v1-20260722"
    ):
        raise ValueError("Candidate42 preregistration receipt changed")
    prereg_commit = subprocess.run(
        ("git", "rev-parse", f"{prereg['tag']}^{{}}"), cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if prereg.get("commit") != prereg_commit:
        raise ValueError("Candidate42 preregistration tag/commit disagree")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", prereg_commit, head), cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if ancestor.returncode != 0:
        raise ValueError("Candidate42 collection does not descend from preregistration")
    pristine_raw = subprocess.run(
        ("git", "show", f"{prereg_commit}:{artifact_contract.VALIDATION_PLAN}"),
        cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if _sha256_bytes(pristine_raw) != prereg.get("sha256"):
        raise ValueError("Candidate42 pristine plan receipt changed")
    pristine = json.loads(pristine_raw.decode("utf-8"))
    normalized = json.loads(json.dumps(plan))
    normalized["status"] = pristine["status"]
    normalized["preregistration_receipt"] = pristine["preregistration_receipt"]
    normalized.pop("implementation_seal", None)
    normalized.pop("collection_seal", None)
    if normalized != pristine:
        raise ValueError("Candidate42 protected preregistration content changed")
    dirty = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"), cwd=root,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout
    if dirty:
        raise ValueError("Candidate42 collection/analyzer worktree is dirty")
    return head


def load_fixed_evidence(
    repository_root: Path | None = None,
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = _root() if repository_root is None else Path(repository_root)
    plan, plan_sha = _read_plan(root, require_sealed=True)
    collection_commit = _validate_collection_head(root, plan)
    committed = validate_committed_invocations(
        root,
        collection_commit=collection_commit,
        preregistration_commit=plan["preregistration_receipt"]["commit"],
    )
    smoke = _validate_registered_smoke(root, committed[(346, "b")])
    # These are the exact Candidate42 closure helpers used by build_spec and
    # the post-exit parent validator.  The non-Isaac parent intentionally does
    # not re-export them, so importing from that module would fail only on the
    # real sealed-evidence path (the synthetic estimator tests do not reach
    # this lazy import).
    from candidate42_public_arm_ramp_collection import (
        runtime_asset_fingerprints,
        source_fingerprints,
    )

    current_source = source_fingerprints(root)
    current_runtime_assets = runtime_asset_fingerprints(root)
    artifacts: dict[tuple[int, str], dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for seed, replicate in RUN_ORDER:
        namespace = committed[(seed, replicate)]
        path = Path(namespace["canonical_artifact_output"])
        artifact, digest = _read_artifact(path, require_sealed_plan=True)
        report_path = Path(namespace["canonical_report_output"])
        collector_report = artifact_contract.validate_final_report(
            json.loads(
                _regular_file(report_path, "Candidate42 collector report").read_text(
                    encoding="utf-8"
                )
            ),
            artifact,
            require_sealed_plan=True,
        )
        if (
            digest != namespace["artifact_sha256"]
            or artifact_contract.sha256_file(report_path) != namespace["report_sha256"]
            or collector_report["artifact_sha256"] != digest
            or collector_report["artifact_size"] != path.stat().st_size
            or Path(collector_report["artifact_output"]).resolve() != path.resolve()
            or collector_report["run_id"] != namespace["run_id"]
            or collector_report["attempt_id"] != namespace["identity"]["attempt_id"]
            or collector_report["attempt_number"] != namespace["attempt_number"]
        ):
            raise ValueError("Candidate42 collector report does not bind artifact")
        metadata = artifact["metadata"]
        run_metadata = namespace["run_metadata"]
        for name, actual in (
            ("source_manifest_sha256", metadata["source_manifest_sha256"]),
            (
                "runtime_asset_manifest_sha256",
                metadata["runtime_asset_manifest_sha256"],
            ),
            (
                "checkpoint_manifest_sha256",
                collector_report["checkpoint_manifest_sha256"],
            ),
        ):
            if run_metadata.get(name) != actual:
                raise ValueError(f"Candidate42 intent differs from artifact at {name}")
        if metadata["source_sha256"] != current_source:
            raise ValueError("Candidate42 current runtime source closure changed")
        if metadata["runtime_asset_sha256"] != current_runtime_assets:
            raise ValueError("Candidate42 current runtime assets changed")
        if metadata["validation_plan_sha256"] != plan_sha:
            raise ValueError("Candidate42 artifact used another final plan")
        if metadata["smoke_artifact_sha256"] != smoke["artifact_sha256"] or (
            metadata["smoke_report_sha256"] != smoke["report_sha256"]
        ):
            raise ValueError("Candidate42 artifact used another smoke authority")
        git = metadata.get("git")
        if not isinstance(git, Mapping) or git.get("commit") != collection_commit:
            raise ValueError("Candidate42 artifact used another collection commit")
        artifacts[(seed, replicate)] = artifact
        receipts.append({
            "run": f"{seed}{replicate}",
            "artifact": str(path.resolve()),
            "artifact_sha256": digest,
            "collector_report": str(report_path.resolve()),
            "collector_report_sha256": artifact_contract.sha256_file(report_path),
            "canonical_commit": namespace["canonical_commit_output"],
            "canonical_commit_sha256": namespace["commit_sha256"],
            "run_id": namespace["run_id"],
            "attempt_id": namespace["identity"]["attempt_id"],
            "attempt_number": namespace["attempt_number"],
            "retryable_prior_attempts": namespace["retryable_prior_attempts"],
        })
    checked = validate_validation_artifacts(artifacts, require_sealed_plan=True)
    fixed = {
        "validation_plan": artifact_contract.VALIDATION_PLAN,
        "validation_plan_sha256": plan_sha,
        "preregistration_receipt": plan["preregistration_receipt"],
        "implementation_seal": plan["implementation_seal"],
        "collection_seal": plan["collection_seal"],
        "collection_commit": collection_commit,
        "transaction_namespaces": {
            f"{seed}{replicate}": {
                "run_id": namespace["run_id"],
                "attempt_id": namespace["identity"]["attempt_id"],
                "attempt_number": namespace["attempt_number"],
                "canonical_commit": namespace["canonical_commit_output"],
                "canonical_commit_sha256": namespace["commit_sha256"],
                "retryable_prior_attempts": namespace["retryable_prior_attempts"],
            }
            for (seed, replicate), namespace in committed.items()
        },
        "non_evidence_smoke": smoke,
    }
    return checked, receipts, fixed


def analyze_fixed_evidence(
    repository_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifacts, receipts, fixed = load_fixed_evidence(repository_root)
    report = compute_validation(
        artifacts, require_sealed_plan=True, repository_root=repository_root
    )
    report["fixed_evidence_receipt"] = fixed
    report["input_artifact_receipts"] = receipts
    _strict_json_bytes(report)
    return report, receipts


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    raw = _strict_json_bytes(payload)
    output = Path(os.path.abspath(os.fspath(output)))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Candidate42 report: {output}")
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        _fsync_directory(output.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> None:
    root = _root()
    report, receipts = analyze_fixed_evidence(root)
    output = root / REPORT_OUTPUT
    publish_json_no_clobber(report, output)
    digest = artifact_contract.sha256_file(output)
    passed = sum(row["pass"] for row in report["gates"].values())
    print(
        "[candidate42-public-arm-ramp-validation] "
        f"decision={report['decision']} gates={passed}/{len(report['gates'])} "
        f"artifacts={len(receipts)} sha256={digest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
