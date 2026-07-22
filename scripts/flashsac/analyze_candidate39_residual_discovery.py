#!/usr/bin/env python3
"""Fail-closed development analysis for Candidate 39 coherent residuals.

This is deliberately a *direction-discovery* analysis, not an online policy
acceptance test.  The two complementary replicates let each stable simulator
slot supply one treatment outcome and one control outcome.  A fixed residual
direction is exposed only if every preregistered development and safety gate
passes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

import candidate39_episode_residual as artifact_contract
from option_residual_screen import (
    DESIGN_CONTRACT,
    DESIGN_SALT,
    HAND_ACTION_DIM,
    build_option_design,
)


REPORT_KIND = "pick_tool_candidate39_residual_discovery_v1"
FIXED_Z_KIND = "pick_tool_candidate39_fixed_residual_direction_v1"
FORMAT_VERSION = 1
SEEDS = (327, 328)
REPLICATES = ("a", "b")
RUN_ORDER = ((327, "a"), (327, "b"), (328, "b"), (328, "a"))
NUM_ENVS = 64
INPUT_PATTERN = (
    "logs/flashsac/pick_tool/50_c39_residual_dev_s{seed}_{replicate}/trial.pt"
)
REPORT_OUTPUT = (
    "logs/flashsac/pick_tool/50_c39_residual_development_discovery.json"
)
FIXED_Z_OUTPUT = "logs/flashsac/pick_tool/50_c39_residual_fixed_z.pt"

EPISODE_FIELDS = (
    "env_slot",
    "treatment",
    "raw_z",
    "pair_slot",
    "antithetic_sign",
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
    "row_transition_grasped",
    "row_transition_true_clearance_m",
    "row_grasp_quality",
    "row_hold_quality",
    "row_max_force_n",
)
EVENTS = (
    "latched_within_window",
    "ever_grasped",
    "success",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
)
GATE_THRESHOLDS = {
    "common_exposure_opportunity_slots_min_across_discovery": 32,
    "discordant_latched_within_window_slots_min": 8,
    "treatment_triggered_dropped_max": 0,
    "treatment_triggered_unsafe_force_max": 0,
    "conditional_unlatched_clearance_ge_5cm_delta_max": 0.02,
    "conditional_ever_grasped_delta_floor": -0.1,
    "new_abs_action_ge_098_element_fraction_max": 0.005,
    "episodes_with_new_abs_action_ge_098_max_per_run": 1,
    "any_new_abs_action_ge_0999": False,
    "component_and_l2_budget_violations": 0,
}
TOKEN_COMPONENT_CAP = 0.1
DISTAL_COMPONENT_CAP = 0.05
DELTA_L2_CAP = 0.2
_BUDGET_ATOL = 2.0e-7


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _registered_development_gates() -> dict[str, Any]:
    """Require the executable thresholds to match the sealed plan exactly."""

    path = _root() / artifact_contract.DEVELOPMENT_PLAN
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read Candidate 39 development plan: {path}") from error
    gates = payload.get("development_gates")
    if not isinstance(gates, dict):
        raise RuntimeError("Candidate 39 development plan has no gate mapping")
    gates = dict(gates)
    if gates.pop("all_must_pass_before_direction_extraction", None) is not True:
        raise RuntimeError("Candidate 39 development gates are not fail-closed")
    if gates != GATE_THRESHOLDS:
        raise RuntimeError(
            "Candidate 39 analyzer thresholds differ from the development plan"
        )
    return gates


def _owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _semantic_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _torch_load_weights_only(raw: bytes) -> Any:
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("weights_only PyTorch loading is required") from error


def _read_artifact(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"artifact is not a regular non-symlink file: {path}")
    raw = path.read_bytes()
    artifact = artifact_contract.validate_artifact(_torch_load_weights_only(raw))
    return artifact, _sha256_bytes(raw)


def _without_run_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only explicitly run-varying fields before identity comparison."""

    normalized = dict(metadata)
    normalized.pop("seed", None)
    normalized.pop("replicate", None)
    runtime = normalized.get("runtime")
    if isinstance(runtime, Mapping):
        runtime = dict(runtime)
        runtime.pop("seed", None)
        normalized["runtime"] = runtime
    return normalized


def _require_tensor_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise ValueError(f"{label} is not bit-exact")


def validate_discovery_artifacts(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Validate all artifacts, registered randomization, and shared lineage."""

    _registered_development_gates()

    if tuple(artifact_contract.EPISODE_FIELDS) != EPISODE_FIELDS or tuple(
        artifact_contract.STEP_FIELDS
    ) != STEP_FIELDS:
        raise RuntimeError("Candidate 39 analyzer/artifact schemas no longer agree")
    fixed_numeric_contract = {
        "WINDOW_STEPS": 32,
        "RAW_Z_ABS_CAP": 2.0,
        "TOKEN_COMPONENT_CAP": TOKEN_COMPONENT_CAP,
        "DISTAL_COMPONENT_CAP": DISTAL_COMPONENT_CAP,
        "PRE_TANH_L2_CAP": DELTA_L2_CAP,
    }
    for name, expected in fixed_numeric_contract.items():
        if getattr(artifact_contract, name) != expected:
            raise RuntimeError(f"Candidate 39 artifact {name} changed")
    expected_keys = {(seed, replicate) for seed in SEEDS for replicate in REPLICATES}
    if set(artifacts) != expected_keys:
        raise ValueError("discovery evidence must contain exactly seeds 327/328 x a/b")
    validated: dict[tuple[int, str], dict[str, Any]] = {}
    reference_identity: dict[str, Any] | None = None
    for seed, replicate in RUN_ORDER:
        artifact = artifact_contract.validate_artifact(artifacts[(seed, replicate)])
        metadata = artifact["metadata"]
        episodes = artifact["episodes"]
        steps = artifact["steps"]
        if set(episodes) != set(EPISODE_FIELDS):
            raise ValueError(f"seed {seed}{replicate} episode schema changed")
        if set(steps) != set(STEP_FIELDS):
            raise ValueError(f"seed {seed}{replicate} step schema changed")
        for key, expected in (
            ("seed", seed),
            ("replicate", replicate),
            ("num_envs", NUM_ENVS),
        ):
            if metadata.get(key) != expected:
                raise ValueError(f"seed {seed}{replicate} metadata {key!r} changed")

        expected_design = build_option_design(
            seed=seed,
            num_envs=NUM_ENVS,
            replicate=replicate,
            dtype=episodes["raw_z"].dtype,
            device="cpu",
        )
        _require_tensor_equal(
            episodes["env_slot"], torch.arange(NUM_ENVS), f"{seed}{replicate} env_slot"
        )
        _require_tensor_equal(
            episodes["treatment"],
            expected_design.treatment,
            f"{seed}{replicate} treatment",
        )
        _require_tensor_equal(
            episodes["raw_z"], expected_design.raw_z, f"{seed}{replicate} raw_z"
        )
        _require_tensor_equal(
            episodes["pair_slot"],
            expected_design.pair_slot,
            f"{seed}{replicate} pair_slot",
        )
        _require_tensor_equal(
            episodes["antithetic_sign"],
            expected_design.antithetic_sign,
            f"{seed}{replicate} antithetic_sign",
        )

        identity = _without_run_identity(metadata)
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise ValueError(
                "policy, source, hash, asset, or runtime identity differs across runs"
            )
        validated[(seed, replicate)] = artifact

    for seed in SEEDS:
        artifact_a = validated[(seed, "a")]
        artifact_b = validated[(seed, "b")]
        artifact_contract.validate_complementary_artifacts(artifact_a, artifact_b)
        episode_a = artifact_a["episodes"]
        episode_b = artifact_b["episodes"]
        _require_tensor_equal(
            episode_a["treatment"], ~episode_b["treatment"], f"seed {seed} complement"
        )
        for name in ("env_slot", "raw_z", "pair_slot", "antithetic_sign"):
            _require_tensor_equal(
                episode_a[name], episode_b[name], f"seed {seed} cross-replicate {name}"
            )
    return validated


def load_fixed_evidence(
    repository_root: Path | None = None,
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    """Load only the four preregistered development artifacts."""

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
    return validate_discovery_artifacts(artifacts), receipts


def _ht_event(
    treatment: torch.Tensor, domain: torch.Tensor, event: torch.Tensor
) -> dict[str, float | int]:
    denominator = int(domain.sum())
    treatment_rows = int((treatment & domain).sum())
    control_rows = int(((~treatment) & domain).sum())
    if denominator <= 0 or treatment_rows <= 0 or control_rows <= 0:
        raise ValueError("a discovery seed lacks conditional support in one arm")
    treatment_count = int((treatment & domain & event).sum())
    control_count = int(((~treatment) & domain & event).sum())
    treatment_estimate = 2.0 * treatment_count / denominator
    control_estimate = 2.0 * control_count / denominator
    return {
        "treatment": treatment_estimate,
        "control": control_estimate,
        "delta": treatment_estimate - control_estimate,
        "treatment_count": treatment_count,
        "control_count": control_count,
        "domain_rows": denominator,
        "treatment_rows": treatment_rows,
        "control_rows": control_rows,
    }


def _raw_arm_funnel(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    treatment_arm: bool,
) -> dict[str, Any]:
    assigned = 0
    triggered = 0
    counts = {event: 0 for event in EVENTS}
    for seed in SEEDS:
        for replicate in REPLICATES:
            episodes = artifacts[(seed, replicate)]["episodes"]
            arm = episodes["treatment"] if treatment_arm else ~episodes["treatment"]
            domain = arm & episodes["triggered"]
            assigned += int(arm.sum())
            triggered += int(domain.sum())
            for event in EVENTS:
                counts[event] += int((domain & episodes[event]).sum())
    return {
        "assigned": assigned,
        "triggered": triggered,
        "trigger_rate_observed": triggered / assigned if assigned else None,
        **{
            event: {
                "count": count,
                "rate_given_trigger_observed": count / triggered if triggered else None,
            }
            for event, count in counts.items()
        },
    }


def _per_run_step_audit(
    artifact: Mapping[str, Any], *, run: str
) -> dict[str, Any]:
    steps = artifact["steps"]
    active = steps["row_residual_active"]
    baseline_hand = steps["row_baseline_action"][:, 7:]
    candidate_hand = steps["row_candidate_action"][:, 7:]
    if active.numel() != candidate_hand.shape[0]:
        raise ValueError(f"{run} residual-active step mask has the wrong shape")
    new_098 = active.unsqueeze(-1) & (candidate_hand.abs() >= 0.98) & (
        baseline_hand.abs() < 0.98
    )
    new_0999 = active.unsqueeze(-1) & (candidate_hand.abs() >= 0.999) & (
        baseline_hand.abs() < 0.999
    )
    active_elements = int(active.sum()) * HAND_ACTION_DIM
    new_elements = int(new_098.sum())
    saturated_slots = torch.unique(steps["row_env_slot"][new_098.any(dim=-1)])

    delta = steps["row_applied_delta"]
    token_bad = (delta[:, :9].abs() > TOKEN_COMPONENT_CAP + _BUDGET_ATOL).any(dim=-1)
    distal_bad = (delta[:, 9:].abs() > DISTAL_COMPONENT_CAP + _BUDGET_ATOL).any(dim=-1)
    l2_bad = torch.linalg.vector_norm(delta.to(torch.float64), dim=-1) > (
        DELTA_L2_CAP + _BUDGET_ATOL
    )
    inactive_nonzero = (~active) & (delta != 0.0).any(dim=-1)
    budget_bad = token_bad | distal_bad | l2_bad | inactive_nonzero
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
    }


def _paired_direction(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    gradient = torch.zeros(HAND_ACTION_DIM, dtype=torch.float64)
    common_count = 0
    discordant_count = 0
    positive_count = 0
    negative_count = 0
    per_seed: dict[str, Any] = {}
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
        # A SEARCH-side latch followed by release can retire the intervention
        # before handoff while leaving triggered=True.  Such a slot is valid
        # randomized ITT evidence, but it supplied no residual treatment
        # opportunity and therefore cannot identify a residual direction.
        common = both_triggered & (rows_a > 0) & (rows_b > 0)
        treatment_is_a = a["treatment"]
        treatment_outcome = torch.where(
            treatment_is_a,
            a["latched_within_window"],
            b["latched_within_window"],
        )
        control_outcome = torch.where(
            treatment_is_a,
            b["latched_within_window"],
            a["latched_within_window"],
        )
        advantage = (
            treatment_outcome.to(torch.int8) - control_outcome.to(torch.int8)
        )
        paired_advantage = advantage[common].to(torch.float64)
        clipped_z = a["raw_z"][common].to(torch.float64).clamp(-2.0, 2.0)
        gradient += (paired_advantage.unsqueeze(-1) * clipped_z).sum(dim=0)
        common_seed = int(common.sum())
        positive_seed = int((paired_advantage > 0).sum())
        negative_seed = int((paired_advantage < 0).sum())
        common_count += common_seed
        positive_count += positive_seed
        negative_count += negative_seed
        discordant_count += positive_seed + negative_seed
        per_seed[str(seed)] = {
            "triggered_in_both_replicates": int(both_triggered.sum()),
            "common_exposure_opportunity_slots": common_seed,
            "excluded_without_common_intervention_opportunity": int(
                (both_triggered & ~common).sum()
            ),
            "treatment_only_latched": positive_seed,
            "control_only_latched": negative_seed,
            "discordant_latched_within_window_slots": positive_seed + negative_seed,
        }
    finite = bool(torch.isfinite(gradient).all())
    rms = (
        float(torch.sqrt(torch.mean(gradient.square())).item()) if finite else float("nan")
    )
    usable = finite and math.isfinite(rms) and rms > 0.0
    result: dict[str, Any] = {
        "domain": (
            "same seed/env_slot triggered and possessing a residual-window row "
            "in both complementary replicates"
        ),
        "primary_outcome": "latched_within_window",
        "common_exposure_opportunity_slots": common_count,
        "discordant_latched_within_window_slots": discordant_count,
        "treatment_only_latched": positive_count,
        "control_only_latched": negative_count,
        "gradient_finite_nonzero": usable,
        "per_seed": per_seed,
    }
    if usable:
        fixed_z = (gradient / rms).clamp(-2.0, 2.0).to(torch.float32)
        result["gradient_rms"] = rms
        result["gradient"] = [float(value) for value in gradient.tolist()]
        result["fixed_z"] = [float(value) for value in fixed_z.tolist()]
    return result


def compute_discovery(
    artifacts: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the preregistered discovery result from validated artifacts."""

    artifacts = validate_discovery_artifacts(artifacts)
    per_seed_ht: dict[str, Any] = {}
    for seed in SEEDS:
        treatment = torch.stack(
            [artifacts[(seed, replicate)]["episodes"]["treatment"] for replicate in REPLICATES]
        )
        triggered = torch.stack(
            [artifacts[(seed, replicate)]["episodes"]["triggered"] for replicate in REPLICATES]
        )
        per_seed_ht[str(seed)] = {
            event: _ht_event(
                treatment,
                triggered,
                torch.stack(
                    [
                        artifacts[(seed, replicate)]["episodes"][event]
                        for replicate in REPLICATES
                    ]
                ),
            )
            for event in EVENTS
        }

    equal_seed_ht = {
        event: {
            "treatment": sum(
                per_seed_ht[str(seed)][event]["treatment"] for seed in SEEDS
            )
            / len(SEEDS),
            "control": sum(
                per_seed_ht[str(seed)][event]["control"] for seed in SEEDS
            )
            / len(SEEDS),
            "delta": sum(per_seed_ht[str(seed)][event]["delta"] for seed in SEEDS)
            / len(SEEDS),
        }
        for event in EVENTS
    }
    step_runs = [
        _per_run_step_audit(
            artifacts[(seed, replicate)], run=f"{seed}{replicate}"
        )
        for seed, replicate in RUN_ORDER
    ]
    total_active_elements = sum(
        row["residual_active_hand_elements"] for row in step_runs
    )
    total_new_098 = sum(row["new_abs_action_ge_098_elements"] for row in step_runs)
    aggregate_step = {
        "residual_active_hand_elements": total_active_elements,
        "new_abs_action_ge_098_elements": total_new_098,
        "new_abs_action_ge_098_element_fraction": (
            total_new_098 / total_active_elements if total_active_elements else 0.0
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
    }
    paired = _paired_direction(artifacts)
    treatment_dropped = 0
    treatment_unsafe = 0
    for artifact in artifacts.values():
        episode = artifact["episodes"]
        domain = episode["treatment"] & episode["triggered"]
        treatment_dropped += int((domain & episode["dropped"]).sum())
        treatment_unsafe += int((domain & episode["unsafe_force"]).sum())

    def gate(value: Any, threshold: Any, comparison: str, passed: bool) -> dict[str, Any]:
        return {
            "value": value,
            "threshold": threshold,
            "comparison": comparison,
            "pass": bool(passed),
        }

    gates = {
        "common_exposure_opportunity_slots": gate(
            paired["common_exposure_opportunity_slots"],
            GATE_THRESHOLDS[
                "common_exposure_opportunity_slots_min_across_discovery"
            ],
            ">=",
            paired["common_exposure_opportunity_slots"]
            >= GATE_THRESHOLDS[
                "common_exposure_opportunity_slots_min_across_discovery"
            ],
        ),
        "discordant_latched_within_window_slots": gate(
            paired["discordant_latched_within_window_slots"],
            GATE_THRESHOLDS["discordant_latched_within_window_slots_min"],
            ">=",
            paired["discordant_latched_within_window_slots"]
            >= GATE_THRESHOLDS["discordant_latched_within_window_slots_min"],
        ),
        "treatment_triggered_dropped": gate(
            treatment_dropped,
            GATE_THRESHOLDS["treatment_triggered_dropped_max"],
            "<=",
            treatment_dropped <= GATE_THRESHOLDS["treatment_triggered_dropped_max"],
        ),
        "treatment_triggered_unsafe_force": gate(
            treatment_unsafe,
            GATE_THRESHOLDS["treatment_triggered_unsafe_force_max"],
            "<=",
            treatment_unsafe
            <= GATE_THRESHOLDS["treatment_triggered_unsafe_force_max"],
        ),
        "conditional_unlatched_clearance_ge_5cm_delta": gate(
            equal_seed_ht["unlatched_clearance_ge_5cm"]["delta"],
            GATE_THRESHOLDS[
                "conditional_unlatched_clearance_ge_5cm_delta_max"
            ],
            "<=",
            equal_seed_ht["unlatched_clearance_ge_5cm"]["delta"]
            <= GATE_THRESHOLDS[
                "conditional_unlatched_clearance_ge_5cm_delta_max"
            ],
        ),
        "conditional_ever_grasped_delta": gate(
            equal_seed_ht["ever_grasped"]["delta"],
            GATE_THRESHOLDS["conditional_ever_grasped_delta_floor"],
            ">=",
            equal_seed_ht["ever_grasped"]["delta"]
            >= GATE_THRESHOLDS["conditional_ever_grasped_delta_floor"],
        ),
        "new_abs_action_ge_098_element_fraction": gate(
            aggregate_step["new_abs_action_ge_098_element_fraction"],
            GATE_THRESHOLDS["new_abs_action_ge_098_element_fraction_max"],
            "<=",
            aggregate_step["new_abs_action_ge_098_element_fraction"]
            <= GATE_THRESHOLDS["new_abs_action_ge_098_element_fraction_max"],
        ),
        "episodes_with_new_abs_action_ge_098_per_run": gate(
            aggregate_step["max_episodes_with_new_abs_action_ge_098_per_run"],
            GATE_THRESHOLDS[
                "episodes_with_new_abs_action_ge_098_max_per_run"
            ],
            "<=",
            aggregate_step["max_episodes_with_new_abs_action_ge_098_per_run"]
            <= GATE_THRESHOLDS[
                "episodes_with_new_abs_action_ge_098_max_per_run"
            ],
        ),
        "any_new_abs_action_ge_0999": gate(
            aggregate_step["any_new_abs_action_ge_0999"],
            GATE_THRESHOLDS["any_new_abs_action_ge_0999"],
            "==",
            aggregate_step["any_new_abs_action_ge_0999"]
            is GATE_THRESHOLDS["any_new_abs_action_ge_0999"],
        ),
        "component_and_l2_budget_violations": gate(
            aggregate_step["component_and_l2_budget_violations"],
            GATE_THRESHOLDS["component_and_l2_budget_violations"],
            "==",
            aggregate_step["component_and_l2_budget_violations"]
            == GATE_THRESHOLDS["component_and_l2_budget_violations"],
        ),
        "gradient_finite_nonzero": gate(
            paired["gradient_finite_nonzero"], True, "==", paired["gradient_finite_nonzero"]
        ),
    }
    all_pass = all(row["pass"] for row in gates.values())
    # Gradient and candidate z are sensitive to outcomes and have no meaning
    # when a safety, power, or direction-usability gate fails.
    paired_public = dict(paired)
    candidate_direction: dict[str, Any] | None = None
    if all_pass:
        candidate_direction = {
            "design_contract": DESIGN_CONTRACT,
            "design_salt": DESIGN_SALT,
            "gradient": paired_public.pop("gradient"),
            "gradient_rms": paired_public.pop("gradient_rms"),
            "fixed_z": paired_public.pop("fixed_z"),
            "normalization": (
                "float64 gradient=sum advantage*clamp(raw_z,-2,2); "
                "fixed_z=clip(gradient/RMS(gradient),-2,2).float32"
            ),
        }
    else:
        paired_public.pop("gradient", None)
        paired_public.pop("gradient_rms", None)
        paired_public.pop("fixed_z", None)

    result: dict[str, Any] = {
        "kind": REPORT_KIND,
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "development_only": True,
        "estimand_note": (
            "equal-seed Horvitz-Thompson values use known propensity 0.5 and "
            "are not within-arm observed proportions; raw observed proportions "
            "are reported separately in raw_funnel"
        ),
        "raw_funnel": {
            "treatment": _raw_arm_funnel(artifacts, treatment_arm=True),
            "control": _raw_arm_funnel(artifacts, treatment_arm=False),
        },
        "per_seed_equal_seed_ht_inputs": per_seed_ht,
        "equal_seed_horvitz_thompson": equal_seed_ht,
        "step_audit": {"per_run": step_runs, "aggregate": aggregate_step},
        "paired_discovery": paired_public,
        "gates": gates,
        "all_gates_pass": all_pass,
        "decision": "extract_fixed_direction" if all_pass else "reject_direction_extraction",
    }
    if candidate_direction is not None:
        result["candidate_direction"] = candidate_direction
    # This also catches accidental tensors and non-finite floats in the report.
    _strict_json_bytes(result)
    return result


def build_fixed_z_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("kind") != REPORT_KIND or report.get("all_gates_pass") is not True:
        raise ValueError("a fixed direction requires an all-gates-pass discovery report")
    direction = report.get("candidate_direction")
    if not isinstance(direction, Mapping) or set(direction) != {
        "design_contract",
        "design_salt",
        "gradient",
        "gradient_rms",
        "fixed_z",
        "normalization",
    }:
        raise ValueError("discovery report has no complete candidate direction")
    fixed_z = torch.tensor(direction["fixed_z"], dtype=torch.float32)
    if fixed_z.shape != (HAND_ACTION_DIM,) or not bool(torch.isfinite(fixed_z).all()):
        raise ValueError("fixed_z must be finite float32 hand14")
    if bool((fixed_z.abs() > 2.0).any()):
        raise ValueError("fixed_z escaped the registered component clip")
    return {
        "kind": FIXED_Z_KIND,
        "format_version": FORMAT_VERSION,
        "design_contract": DESIGN_CONTRACT,
        "design_salt": DESIGN_SALT,
        "discovery_seeds": torch.tensor(SEEDS, dtype=torch.int64),
        "fixed_z": fixed_z,
        "discovery_report_semantic_sha256": _semantic_sha256(report),
    }


def validate_fixed_z_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "kind",
        "format_version",
        "design_contract",
        "design_salt",
        "discovery_seeds",
        "fixed_z",
        "discovery_report_semantic_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("fixed-z payload schema changed")
    if (
        payload["kind"] != FIXED_Z_KIND
        or payload["format_version"] != FORMAT_VERSION
        or payload["design_contract"] != DESIGN_CONTRACT
        or payload["design_salt"] != DESIGN_SALT
    ):
        raise ValueError("fixed-z payload identity changed")
    seeds = payload["discovery_seeds"]
    fixed_z = payload["fixed_z"]
    if not isinstance(seeds, torch.Tensor) or not torch.equal(
        seeds, torch.tensor(SEEDS, dtype=torch.int64)
    ):
        raise ValueError("fixed-z discovery seeds changed")
    if (
        not isinstance(fixed_z, torch.Tensor)
        or fixed_z.dtype != torch.float32
        or fixed_z.device.type != "cpu"
        or fixed_z.shape != (HAND_ACTION_DIM,)
        or not bool(torch.isfinite(fixed_z).all())
        or bool((fixed_z.abs() > 2.0).any())
    ):
        raise ValueError("fixed-z tensor must be finite clipped CPU float32 hand14")
    digest = payload["discovery_report_semantic_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("fixed-z report receipt must be a lowercase SHA256")
    return {
        **dict(payload),
        "discovery_seeds": seeds.detach().clone(),
        "fixed_z": fixed_z.detach().clone(),
    }


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    encoded = _strict_json_bytes(payload)
    output = Path(os.path.abspath(os.fspath(output)))
    if _owned(output):
        raise FileExistsError(f"JSON output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        linked = True
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if linked and _owned(output) and os.path.samefile(output, temporary):
            output.unlink()
        raise
    finally:
        if _owned(temporary):
            temporary.unlink()


def publish_fixed_z_no_clobber(payload: Mapping[str, Any], output: Path) -> str:
    validated = validate_fixed_z_payload(payload)
    output = Path(os.path.abspath(os.fspath(output)))
    if _owned(output):
        raise FileExistsError(f"fixed-z output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    linked = False
    try:
        with temporary.open("wb") as stream:
            torch.save(validated, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256_bytes(temporary.read_bytes())
        # A successful weights-only round trip is part of the publication
        # contract, not merely a test convenience.
        validate_fixed_z_payload(
            torch.load(temporary, map_location="cpu", weights_only=True)
        )
        os.link(temporary, output)
        linked = True
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return digest
    except BaseException:
        if linked and _owned(output) and os.path.samefile(output, temporary):
            output.unlink()
        raise
    finally:
        if _owned(temporary):
            temporary.unlink()


def analyze_fixed_evidence(
    repository_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifacts, receipts = load_fixed_evidence(repository_root)
    report = compute_discovery(artifacts)
    report["receipts"] = receipts
    _strict_json_bytes(report)
    return report, receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report-output", type=Path, default=Path(REPORT_OUTPUT))
    parser.add_argument("--fixed-z-output", type=Path, default=Path(FIXED_Z_OUTPUT))
    args = parser.parse_args()
    report, _ = analyze_fixed_evidence()
    report_output = args.report_output
    if not report_output.is_absolute():
        report_output = _root() / report_output
    publish_json_no_clobber(report, report_output)
    if report["all_gates_pass"]:
        fixed_output = args.fixed_z_output
        if not fixed_output.is_absolute():
            fixed_output = _root() / fixed_output
        publish_fixed_z_no_clobber(build_fixed_z_payload(report), fixed_output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
