#!/usr/bin/env python3
"""Train an audited conservative SEARCH-versus-FlashSAC handoff selector.

This is deliberately a small, simulation-free model.  At every eligible
handoff candidate the default decision is to route to FlashSAC.  Two
L2-regularized logistic heads estimate the strict-success probability of
continuing SEARCH and routing to FlashSAC; the selector may only veto that
default and continue SEARCH when its continue estimate clears both an absolute
probability floor and a probability-margin threshold.

All model and threshold selection uses leave-one-seed-out predictions from the
training seed groups.  The development seed is evaluated exactly once after
selection and is never used to fit a normalizer, model, or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


DATASET_KIND = "pick_tool_recoverability_pairs_v1"
MODEL_KIND = "pick_tool_recoverability_selector_v1"
OBSERVATION_DIM = 115
ACTION_DIM = 21
EXPECTED_FEATURE_POLICY = (
    "training may use only canonical observation/search_action/flashsac_action "
    "from the continue_search arm; direct preference uses "
    "preference_label_valid, while dual outcome heads use "
    "paired_outcome_label_valid; audit scalars and outcomes are forbidden "
    "as features"
)
EXPECTED_INPUT_FIELDS = ["observation", "search_action", "flashsac_action"]
REQUIRED_FORBIDDEN_PATTERNS = {
    "*_success",
    "*_failure",
    "*_time_out",
    "*_dropped",
    "*_unsafe_force",
    "*clearance*",
    "*pregrasp_score*",
    "*proximity_quality*",
    "*max_force*",
    "*handoff_step*",
    "seed",
    "env_slot",
    "slot_episode_index",
}
FEATURE_DIMS = {
    "local84_v1": 84,
    "public_gate_v1": 152,
    "raw_public_v1": 150,
}
PAD_ORDER = ["middle", "pinky", "ring", "index", "thumb"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_tensor_bytes(value: torch.Tensor) -> bytes:
    canonical = value.detach().cpu().contiguous().clone()
    if canonical.is_floating_point():
        if not bool(torch.isfinite(canonical).all()):
            raise ValueError("cannot hash a tensor containing NaN or infinity")
        canonical[canonical == 0.0] = 0.0
    return canonical.numpy().tobytes()


def semantic_data_sha256(dataset: Mapping[str, Any]) -> str:
    """Hash every model input, mask, group, and dual outcome target."""

    digest = hashlib.sha256()
    for name in (
        "observation",
        "search_action",
        "flashsac_action",
        "seed",
        "paired_outcome_label_valid",
        "continue_success",
        "route_success",
        "continue_dropped",
        "route_dropped",
        "continue_unsafe_force",
        "route_unsafe_force",
    ):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(_canonical_tensor_bytes(dataset[name]))
    return digest.hexdigest()


def _require_tensor(
    dataset: Mapping[str, Any],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = dataset.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        raise ValueError(f"dataset {name} shape must be {shape}, got {actual}")
    if value.dtype != dtype:
        raise TypeError(f"dataset {name} dtype must be {dtype}, got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"dataset {name} must be a CPU tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"dataset {name} contains NaN or infinity")
    return value.contiguous()


def validate_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on provenance, labels, and the selector feature policy."""

    if not isinstance(payload, Mapping):
        raise TypeError("recoverability dataset must be a mapping")
    if payload.get("kind") != DATASET_KIND or payload.get("format_version") != 1:
        raise ValueError("unsupported recoverability dataset kind or version")
    metadata_value = payload.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise TypeError("recoverability dataset metadata must be a mapping")
    metadata = dict(metadata_value)
    json.dumps(metadata, sort_keys=True, allow_nan=False)
    if metadata.get("pairing_semantics") != "independent_gpu_rollout_diagnostic_v1":
        raise ValueError("unsupported pairing semantics")
    if metadata.get("causal_counterfactual_claim_allowed") is not False:
        raise ValueError("dataset must explicitly forbid a causal counterfactual claim")
    if metadata.get("feature_policy") != EXPECTED_FEATURE_POLICY:
        raise ValueError("dataset feature policy does not match the audited v1 contract")
    if metadata.get("selector_input_fields") != EXPECTED_INPUT_FIELDS:
        raise ValueError("dataset changes the canonical selector input fields")
    forbidden = metadata.get("forbidden_selector_feature_patterns")
    if not isinstance(forbidden, list) or not REQUIRED_FORBIDDEN_PATTERNS.issubset(
        set(forbidden)
    ):
        raise ValueError("dataset weakens the forbidden selector feature policy")
    dual = metadata.get("dual_outcome_label_contract")
    if dual != {
        "targets": ["continue_success", "route_success"],
        "valid_mask": "paired_outcome_label_valid",
        "tie_semantics": "valid_for_two_independent_outcome_heads",
    }:
        raise ValueError("dataset changes the audited dual-outcome label contract")
    provenance = metadata.get("canonical_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("dataset lacks canonical collection provenance")
    if (
        provenance.get("task_mode") != "full_task"
        or provenance.get("observation_contract") != "pick_tool_markov115_v1"
        or int(provenance.get("observation_dim", -1)) != OBSERVATION_DIM
        or int(provenance.get("action_dim", -1)) != ACTION_DIM
        or not bool(provenance.get("deterministic_policy_actions", False))
        or float(provenance.get("episode_length_s", -1.0)) != 20.0
        or int(provenance.get("max_episode_steps", -1)) != 1000
    ):
        raise ValueError("dataset collection provenance is incompatible with this trainer")
    source_hashes = provenance.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("dataset collection source hashes are missing")
    for name, digest in source_hashes.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("dataset contains an invalid collection source hash")

    rows = int(metadata.get("rows", -1))
    if rows < 1:
        raise ValueError("recoverability dataset has no rows")
    tensors: dict[str, torch.Tensor] = {}
    for name, width in (
        ("observation", OBSERVATION_DIM),
        ("continue_observation", OBSERVATION_DIM),
        ("search_action", ACTION_DIM),
        ("continue_search_action", ACTION_DIM),
        ("flashsac_action", ACTION_DIM),
        ("continue_flashsac_action", ACTION_DIM),
    ):
        tensors[name] = _require_tensor(
            payload, name, shape=(rows, width), dtype=torch.float32
        )
    tensors["seed"] = _require_tensor(
        payload, "seed", shape=(rows,), dtype=torch.long
    )
    for name in (
        "strong_pair",
        "paired_outcome_label_valid",
        "continue_success",
        "route_success",
        "continue_dropped",
        "route_dropped",
        "continue_unsafe_force",
        "route_unsafe_force",
    ):
        tensors[name] = _require_tensor(
            payload, name, shape=(rows,), dtype=torch.bool
        )
    if not torch.equal(tensors["observation"], tensors["continue_observation"]):
        raise ValueError("canonical observation is not the untreated SEARCH-arm input")
    if not torch.equal(tensors["search_action"], tensors["continue_search_action"]):
        raise ValueError("canonical SEARCH action is not from the untreated arm")
    if not torch.equal(
        tensors["flashsac_action"], tensors["continue_flashsac_action"]
    ):
        raise ValueError("canonical FlashSAC action is not from the untreated arm")
    if not torch.equal(
        tensors["paired_outcome_label_valid"], tensors["strong_pair"]
    ):
        raise ValueError("dual outcome mask must equal the audited strong-pair mask")
    valid = tensors["paired_outcome_label_valid"]
    if not bool(valid.any()) or int(valid.sum()) != int(metadata.get("strong_rows", -1)):
        raise ValueError("strong-row metadata does not match the label-valid mask")
    if bool((tensors["observation"][:, 106] != 0.0).any()):
        raise ValueError("selector candidates must be captured before the latch")
    metadata_seeds = metadata.get("seeds")
    actual_seeds = sorted(set(int(seed) for seed in tensors["seed"].tolist()))
    if metadata_seeds != actual_seeds:
        raise ValueError("dataset seed metadata does not match tensor groups")
    for digest_name in ("builder_source_sha256",):
        digest = metadata.get(digest_name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"dataset has an invalid {digest_name}")
    return {"kind": DATASET_KIND, "format_version": 1, "metadata": metadata, **tensors}


def quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert batched wxyz quaternions to object-to-world rotation matrices."""

    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion tensor must have shape (N, 4)")
    norm = torch.linalg.vector_norm(quaternion, dim=1, keepdim=True)
    if bool((norm < 1.0e-8).any()) or not bool(torch.isfinite(norm).all()):
        raise ValueError("observation contains an invalid object quaternion")
    q = quaternion / norm
    w, x, y, z = q.unbind(dim=1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=1,
    ).reshape(-1, 3, 3)


def build_features(
    observation: torch.Tensor,
    search_action: torch.Tensor,
    flashsac_action: torch.Tensor,
    feature_set: str,
) -> torch.Tensor:
    """Build one of the allow-listed, public-only feature contracts."""

    rows = observation.shape[0]
    if observation.shape != (rows, OBSERVATION_DIM):
        raise ValueError("observation must have shape (N, 115)")
    if search_action.shape != (rows, ACTION_DIM):
        raise ValueError("SEARCH action must have shape (N, 21)")
    if flashsac_action.shape != (rows, ACTION_DIM):
        raise ValueError("FlashSAC action must have shape (N, 21)")
    if feature_set == "raw_public_v1":
        features = torch.cat(
            (observation[:, :63], observation[:, 70:115], search_action, flashsac_action),
            dim=1,
        )
    elif feature_set in {"local84_v1", "public_gate_v1"}:
        object_position = observation[:, 56:59]
        rotation = quaternion_wxyz_to_matrix(observation[:, 59:63])
        pad_world_relative = observation[:, 38:53].reshape(rows, 5, 3) - (
            object_position[:, None, :]
        )
        pad_object_relative = torch.matmul(
            rotation.transpose(1, 2)[:, None, :, :],
            pad_world_relative[..., None],
        ).squeeze(-1)
        palm_world_relative = observation[:, 53:56] - object_position
        palm_object_relative = torch.bmm(
            rotation.transpose(1, 2), palm_world_relative.unsqueeze(-1)
        ).squeeze(-1)
        if feature_set == "local84_v1":
            features = torch.cat(
                (
                    pad_object_relative.reshape(rows, 15),
                    palm_object_relative,
                    observation[:, 86:87],
                    observation[:, 92:115],
                    search_action,
                    flashsac_action,
                ),
                dim=1,
            )
        else:
            # The object's world-frame x and y axes form a sign-invariant 6D
            # rotation representation after quaternion normalization.
            rotation_6d = torch.cat((rotation[:, :, 0], rotation[:, :, 1]), dim=1)
            previous_action = torch.cat(
                (observation[:, 70:86], observation[:, 87:92]), dim=1
            )
            features = torch.cat(
                (
                    observation[:, 0:19],
                    observation[:, 19:38],
                    pad_object_relative.reshape(rows, 15),
                    palm_object_relative,
                    object_position,
                    rotation_6d,
                    previous_action,
                    observation[:, 86:87],
                    observation[:, 92:115],
                    search_action,
                    flashsac_action,
                ),
                dim=1,
            )
    else:
        raise ValueError(f"unsupported feature set: {feature_set!r}")
    expected = FEATURE_DIMS.get(feature_set)
    if expected is None or features.shape != (rows, expected):
        raise AssertionError(
            f"{feature_set} implementation produced {tuple(features.shape)}, "
            f"expected {(rows, expected)}"
        )
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"{feature_set} contains NaN or infinity")
    return features.contiguous()


def feature_contract(feature_set: str) -> dict[str, Any]:
    if feature_set == "local84_v1":
        return {
            "name": feature_set,
            "dimension": 84,
            "source_fields": EXPECTED_INPUT_FIELDS,
            "pad_input_order": PAD_ORDER,
            "quaternion_order": "wxyz_used_only_for_object_frame_transform",
            "object_frame_transform": "R_object_to_world_transpose",
            "slices": {
                "object_frame_pad_relative": [0, 15],
                "object_frame_palm_relative": [15, 18],
                "true_mesh_lift_observation_86": [18, 19],
                "public_phase_and_slip_92_115": [19, 42],
                "search_action": [42, 63],
                "flashsac_action": [63, 84],
            },
            "explicitly_excluded": [
                "q_0_19",
                "qd_19_38",
                "object_world_position_56_59",
                "object_world_orientation_59_63",
                "target_pose_63_70",
                "previous_action_70_86_plus_87_92",
                "private_audit_fields",
            ],
        }
    if feature_set == "public_gate_v1":
        return {
            "name": feature_set,
            "dimension": 152,
            "source_fields": EXPECTED_INPUT_FIELDS,
            "pad_input_order": PAD_ORDER,
            "quaternion_order": "wxyz",
            "object_frame_transform": "R_object_to_world_transpose",
            "rotation_6d": "object_world_x_axis_then_object_world_y_axis",
            "slices": {
                "q": [0, 19],
                "qd": [19, 38],
                "object_frame_pad_relative": [38, 53],
                "object_frame_palm_relative": [53, 56],
                "object_position": [56, 59],
                "rotation_6d": [59, 65],
                "previous_action_70_86_plus_87_92": [65, 86],
                "true_mesh_lift_observation_86": [86, 87],
                "public_phase_and_slip_92_115": [87, 110],
                "search_action": [110, 131],
                "flashsac_action": [131, 152],
            },
        }
    if feature_set == "raw_public_v1":
        return {
            "name": feature_set,
            "dimension": 150,
            "source_fields": EXPECTED_INPUT_FIELDS,
            "slices": {
                "observation_0_63": [0, 63],
                "observation_70_115": [63, 108],
                "search_action": [108, 129],
                "flashsac_action": [129, 150],
            },
            "excluded_observation_slices": {
                "target_pose": [63, 70],
            },
        }
    raise ValueError(f"unsupported feature set: {feature_set!r}")


def fit_normalizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("normalizer needs a non-empty feature matrix")
    features = features.to(torch.float64)
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1.0e-10, scale, torch.ones_like(scale))
    return mean, scale


def fit_dual_logistic(
    normalized_features: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    l2: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit two deterministic logistic heads with zero initialization."""

    if normalized_features.ndim != 2 or normalized_features.shape[0] < 2:
        raise ValueError("logistic fit needs at least two rows")
    if outcomes.shape != (normalized_features.shape[0], 2):
        raise ValueError("dual logistic outcomes must have shape (N, 2)")
    if not math.isfinite(l2) or l2 <= 0.0:
        raise ValueError("L2 strength must be finite and positive")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    x = normalized_features.detach().to(dtype=torch.float64, device="cpu")
    y = outcomes.detach().to(dtype=torch.float64, device="cpu")
    if bool(((y != 0.0) & (y != 1.0)).any()):
        raise ValueError("logistic outcomes must be binary")
    weight = torch.zeros((2, x.shape[1]), dtype=torch.float64, requires_grad=True)
    prevalence = y.mean(dim=0).clamp(1.0e-6, 1.0 - 1.0e-6)
    bias = torch.logit(prevalence).detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weight.transpose(0, 1) + bias
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
        loss = loss + 0.5 * float(l2) * weight.square().sum()
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        logits = x @ weight.transpose(0, 1) + bias
        data_loss = F.binary_cross_entropy_with_logits(
            logits, y, reduction="mean"
        )
        objective = data_loss + 0.5 * float(l2) * weight.square().sum()
    if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(bias).all()):
        raise RuntimeError("logistic optimization produced non-finite parameters")
    diagnostics = {
        "optimizer": "torch_lbfgs_strong_wolfe_zero_initialized_float64",
        "function_evaluations": evaluations,
        "data_bce": float(data_loss),
        "regularized_objective": float(objective),
        "weight_l2_norm": float(torch.linalg.vector_norm(weight)),
    }
    return weight.detach(), bias.detach(), diagnostics


def predict_probabilities(
    features: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x = (features.to(torch.float64) - mean) / scale
    logits = x @ weight.transpose(0, 1) + bias
    # Keep the explicit p_min=1 baseline candidate guaranteed to route.
    epsilon = torch.finfo(torch.float64).eps
    return torch.sigmoid(logits).clamp(epsilon, 1.0 - epsilon)


def policy_metrics(
    select_continue: torch.Tensor,
    continue_success: torch.Tensor,
    route_success: torch.Tensor,
    continue_dropped: torch.Tensor,
    route_dropped: torch.Tensor,
    continue_unsafe_force: torch.Tensor,
    route_unsafe_force: torch.Tensor,
) -> dict[str, Any]:
    select_continue = select_continue.to(torch.bool)
    continue_success = continue_success.to(torch.bool)
    route_success = route_success.to(torch.bool)
    chosen_success = torch.where(select_continue, continue_success, route_success)
    route_gain = route_success & ~continue_success
    route_regression = ~route_success & continue_success
    selected_drop = torch.where(select_continue, continue_dropped, route_dropped)
    selected_unsafe = torch.where(
        select_continue, continue_unsafe_force, route_unsafe_force
    )
    route_gain_count = int(route_gain.sum())
    regression_count = int(route_regression.sum())
    retained_gain_count = int((route_gain & ~select_continue).sum())
    avoided_regression_count = int((route_regression & select_continue).sum())
    return {
        "rows": int(select_continue.numel()),
        "continue_decisions": int(select_continue.sum()),
        "route_decisions": int((~select_continue).sum()),
        "route_baseline_successes": int(route_success.sum()),
        "continue_baseline_successes": int(continue_success.sum()),
        "selected_successes": int(chosen_success.sum()),
        "oracle_successes": int((route_success | continue_success).sum()),
        "net_success_delta_vs_route": int(chosen_success.sum() - route_success.sum()),
        "route_only_gain_count": route_gain_count,
        "retained_route_only_gains": retained_gain_count,
        "route_gain_retention_rate": (
            retained_gain_count / route_gain_count if route_gain_count else 1.0
        ),
        "route_regression_count": regression_count,
        "avoided_route_regressions": avoided_regression_count,
        "route_regression_avoidance_rate": (
            avoided_regression_count / regression_count if regression_count else 1.0
        ),
        "route_baseline_drops": int(route_dropped.sum()),
        "selected_drops": int(selected_drop.sum()),
        "drop_delta_vs_route": int(selected_drop.sum() - route_dropped.sum()),
        "route_baseline_unsafe_force": int(route_unsafe_force.sum()),
        "selected_unsafe_force": int(selected_unsafe.sum()),
        "unsafe_force_delta_vs_route": int(
            selected_unsafe.sum() - route_unsafe_force.sum()
        ),
    }


def evaluate_threshold(
    probabilities: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    seeds: torch.Tensor,
    *,
    p_min: float,
    margin: float,
    minimum_gain_retention: float,
) -> dict[str, Any]:
    if probabilities.shape != (seeds.numel(), 2):
        raise ValueError("probability matrix must have shape (N, 2)")
    # Column 0 is continue, column 1 is route.
    decision = (probabilities[:, 0] >= p_min) & (
        probabilities[:, 0] - probabilities[:, 1] >= margin
    )
    aggregate = policy_metrics(
        decision,
        tensors["continue_success"],
        tensors["route_success"],
        tensors["continue_dropped"],
        tensors["route_dropped"],
        tensors["continue_unsafe_force"],
        tensors["route_unsafe_force"],
    )
    by_seed: dict[str, Any] = {}
    for seed in sorted(set(int(item) for item in seeds.tolist())):
        mask = seeds == seed
        by_seed[str(seed)] = policy_metrics(
            decision[mask],
            tensors["continue_success"][mask],
            tensors["route_success"][mask],
            tensors["continue_dropped"][mask],
            tensors["route_dropped"][mask],
            tensors["continue_unsafe_force"][mask],
            tensors["route_unsafe_force"][mask],
        )
    guard = (
        all(item["net_success_delta_vs_route"] >= 0 for item in by_seed.values())
        and aggregate["route_gain_retention_rate"] >= minimum_gain_retention
        and aggregate["drop_delta_vs_route"] <= 0
        and aggregate["unsafe_force_delta_vs_route"] <= 0
    )
    return {
        "p_continue_min": float(p_min),
        "continue_probability_margin": float(margin),
        "selection_guard_pass": guard,
        "aggregate": aggregate,
        "by_seed": by_seed,
    }


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["threshold_metrics"]
    aggregate = metrics["aggregate"]
    minimum_seed_delta = min(
        item["net_success_delta_vs_route"] for item in metrics["by_seed"].values()
    )
    # Prefer a guarded model, then actual held-seed success improvement.  The
    # remaining terms break ties toward avoiding regressions with fewer vetoes.
    return (
        int(metrics["selection_guard_pass"]),
        aggregate["net_success_delta_vs_route"],
        minimum_seed_delta,
        aggregate["avoided_route_regressions"],
        aggregate["retained_route_only_gains"],
        -aggregate["continue_decisions"],
        {"local84_v1": 2, "public_gate_v1": 1, "raw_public_v1": 0}[
            str(candidate["feature_set"])
        ],
        -float(candidate["l2"]),
        float(metrics["p_continue_min"]),
        float(metrics["continue_probability_margin"]),
    )


def _outcome_tensors(
    dataset: Mapping[str, Any], mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        name: dataset[name][mask].contiguous()
        for name in (
            "continue_success",
            "route_success",
            "continue_dropped",
            "route_dropped",
            "continue_unsafe_force",
            "route_unsafe_force",
        )
    }


def _model_quality(probabilities: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    probabilities = probabilities.clamp(1.0e-12, 1.0 - 1.0e-12)
    targets = targets.to(torch.float64)
    return {
        "dual_head_bce": float(
            F.binary_cross_entropy(probabilities, targets, reduction="mean")
        ),
        "continue_brier": float((probabilities[:, 0] - targets[:, 0]).square().mean()),
        "route_brier": float((probabilities[:, 1] - targets[:, 1]).square().mean()),
        "mean_p_continue": float(probabilities[:, 0].mean()),
        "mean_p_route": float(probabilities[:, 1].mean()),
    }


def train_selector(
    dataset: Mapping[str, Any],
    *,
    dataset_sha256: str,
    dev_seed: int,
    feature_sets: Sequence[str],
    l2_values: Sequence[float],
    p_min_values: Sequence[float],
    margin_values: Sequence[float],
    max_iter: int,
    minimum_cv_net_gain: int,
    minimum_gain_retention: float,
    minimum_regression_avoidance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train with seed-group LOO selection and return model plus JSON report."""

    dataset = validate_dataset(dataset)
    if (
        not isinstance(dataset_sha256, str)
        or len(dataset_sha256) != 64
        or any(character not in "0123456789abcdef" for character in dataset_sha256)
    ):
        raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")
    feature_sets = list(feature_sets)
    if not feature_sets or len(feature_sets) != len(set(feature_sets)):
        raise ValueError("feature sets must be a non-empty unique list")
    for feature_set in feature_sets:
        feature_contract(feature_set)
    l2_values = sorted(set(float(value) for value in l2_values))
    p_min_values = sorted(set(float(value) for value in p_min_values))
    margin_values = sorted(set(float(value) for value in margin_values))
    if not l2_values or any(not math.isfinite(value) or value <= 0 for value in l2_values):
        raise ValueError("L2 grid must contain finite positive values")
    if not p_min_values or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in p_min_values
    ):
        raise ValueError("probability-floor grid values must lie in [0, 1]")
    if not margin_values or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in margin_values
    ):
        raise ValueError("probability-margin grid values must lie in [0, 1]")
    if 1.0 not in p_min_values:
        p_min_values.append(1.0)
        p_min_values.sort()
    if not (0.0 <= minimum_gain_retention <= 1.0):
        raise ValueError("minimum gain retention must lie in [0, 1]")
    if not (0.0 <= minimum_regression_avoidance <= 1.0):
        raise ValueError("minimum regression avoidance must lie in [0, 1]")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    training_configuration = {
        "device": "cpu",
        "parameter_dtype": "float64",
        "torch_version": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "development_seed": int(dev_seed),
        "feature_sets": list(feature_sets),
        "l2_grid": list(l2_values),
        "p_continue_min_grid": list(p_min_values),
        "continue_probability_margin_grid": list(margin_values),
        "max_iter": int(max_iter),
        "minimum_cv_net_gain": int(minimum_cv_net_gain),
        "minimum_gain_retention": float(minimum_gain_retention),
        "minimum_regression_avoidance": float(minimum_regression_avoidance),
        "split_policy": (
            "development seed excluded before grouped leave-one-training-seed-out "
            "model and threshold selection"
        ),
        "normalization": (
            "population mean/std fit only on each fold's training seeds; final "
            "normalizer fit only on all non-development training seeds"
        ),
        "optimizer": {
            "name": "torch.optim.LBFGS",
            "learning_rate": 1.0,
            "max_eval_multiplier": 2,
            "tolerance_grad": 1.0e-10,
            "tolerance_change": 1.0e-12,
            "history_size": 50,
            "line_search_fn": "strong_wolfe",
            "initialization": "zero_weight_prevalence_logit_bias",
            "loss": "mean_dual_binary_cross_entropy_plus_half_l2_weight_square",
            "bias_regularized": False,
        },
        "head_order": [
            "continue_search_strict_success",
            "route_flashsac_strict_success",
        ],
        "default_decision": "route_to_flashsac",
        "veto_decision": (
            "continue_search_only_when_absolute_and_margin_thresholds_pass"
        ),
    }

    valid = dataset["paired_outcome_label_valid"]
    seeds = dataset["seed"][valid]
    all_seeds = sorted(set(int(item) for item in seeds.tolist()))
    if dev_seed not in all_seeds:
        raise ValueError(f"development seed {dev_seed} is absent from strong pairs")
    train_seeds = [seed for seed in all_seeds if seed != dev_seed]
    if len(train_seeds) < 3:
        raise ValueError("at least three non-development seed groups are required for LOO")
    train_mask = valid & (dataset["seed"] != dev_seed)
    dev_mask = valid & (dataset["seed"] == dev_seed)
    train_group = dataset["seed"][train_mask]
    dev_group = dataset["seed"][dev_mask]
    train_outcomes = _outcome_tensors(dataset, train_mask)
    dev_outcomes = _outcome_tensors(dataset, dev_mask)
    train_targets = torch.stack(
        (
            train_outcomes["continue_success"],
            train_outcomes["route_success"],
        ),
        dim=1,
    ).to(torch.float64)
    dev_targets = torch.stack(
        (dev_outcomes["continue_success"], dev_outcomes["route_success"]), dim=1
    ).to(torch.float64)

    hyperparameter_results: list[dict[str, Any]] = []
    global_candidates: list[dict[str, Any]] = []
    feature_cache: dict[str, torch.Tensor] = {}
    for feature_set in feature_sets:
        all_features = build_features(
            dataset["observation"],
            dataset["search_action"],
            dataset["flashsac_action"],
            feature_set,
        )
        feature_cache[feature_set] = all_features
        train_features = all_features[train_mask]
        for l2 in l2_values:
            loo_probabilities = torch.empty((train_mask.sum(), 2), dtype=torch.float64)
            fold_diagnostics: dict[str, Any] = {}
            for held_seed in train_seeds:
                fold_validation = train_group == held_seed
                fold_training = ~fold_validation
                mean, scale = fit_normalizer(train_features[fold_training])
                weight, bias, diagnostics = fit_dual_logistic(
                    (train_features[fold_training].to(torch.float64) - mean) / scale,
                    train_targets[fold_training],
                    l2=l2,
                    max_iter=max_iter,
                )
                loo_probabilities[fold_validation] = predict_probabilities(
                    train_features[fold_validation], mean, scale, weight, bias
                )
                fold_diagnostics[str(held_seed)] = {
                    "fit_seeds": [seed for seed in train_seeds if seed != held_seed],
                    "fit_rows": int(fold_training.sum()),
                    "held_rows": int(fold_validation.sum()),
                    **diagnostics,
                }
            threshold_candidates: list[dict[str, Any]] = []
            for p_min in p_min_values:
                for margin in margin_values:
                    threshold_metrics = evaluate_threshold(
                        loo_probabilities,
                        train_outcomes,
                        train_group,
                        p_min=p_min,
                        margin=margin,
                        minimum_gain_retention=minimum_gain_retention,
                    )
                    candidate = {
                        "feature_set": feature_set,
                        "l2": l2,
                        "threshold_metrics": threshold_metrics,
                    }
                    threshold_candidates.append(candidate)
                    global_candidates.append(candidate)
            best_for_model = max(threshold_candidates, key=_candidate_rank)
            hyperparameter_results.append(
                {
                    "feature_set": feature_set,
                    "dimension": FEATURE_DIMS[feature_set],
                    "l2": l2,
                    "loo_model_quality": _model_quality(
                        loo_probabilities, train_targets
                    ),
                    "fold_diagnostics": fold_diagnostics,
                    "best_threshold": best_for_model["threshold_metrics"],
                }
            )
    selected = max(global_candidates, key=_candidate_rank)
    selected_feature = str(selected["feature_set"])
    selected_l2 = float(selected["l2"])
    selected_threshold = selected["threshold_metrics"]
    cv_aggregate = selected_threshold["aggregate"]
    cv_acceptance = {
        "minimum_net_gain": minimum_cv_net_gain,
        "minimum_gain_retention": minimum_gain_retention,
        "minimum_regression_avoidance": minimum_regression_avoidance,
        "every_held_seed_nonnegative": all(
            item["net_success_delta_vs_route"] >= 0
            for item in selected_threshold["by_seed"].values()
        ),
        "aggregate_net_gain_met": (
            cv_aggregate["net_success_delta_vs_route"] >= minimum_cv_net_gain
        ),
        "gain_retention_met": (
            cv_aggregate["route_gain_retention_rate"] >= minimum_gain_retention
        ),
        "regression_avoidance_met": (
            cv_aggregate["route_regression_avoidance_rate"]
            >= minimum_regression_avoidance
        ),
        "no_drop_worsening": cv_aggregate["drop_delta_vs_route"] <= 0,
        "no_unsafe_force_worsening": (
            cv_aggregate["unsafe_force_delta_vs_route"] <= 0
        ),
    }
    cv_acceptance["accepted"] = all(
        value
        for key, value in cv_acceptance.items()
        if key
        not in {
            "minimum_net_gain",
            "minimum_gain_retention",
            "minimum_regression_avoidance",
        }
    )

    selected_features = feature_cache[selected_feature]
    fit_features = selected_features[train_mask]
    final_mean, final_scale = fit_normalizer(fit_features)
    final_weight, final_bias, final_fit_diagnostics = fit_dual_logistic(
        (fit_features.to(torch.float64) - final_mean) / final_scale,
        train_targets,
        l2=selected_l2,
        max_iter=max_iter,
    )
    dev_probabilities = predict_probabilities(
        selected_features[dev_mask],
        final_mean,
        final_scale,
        final_weight,
        final_bias,
    )
    dev_threshold = evaluate_threshold(
        dev_probabilities,
        dev_outcomes,
        dev_group,
        p_min=float(selected_threshold["p_continue_min"]),
        margin=float(selected_threshold["continue_probability_margin"]),
        minimum_gain_retention=minimum_gain_retention,
    )
    dev_aggregate = dev_threshold["aggregate"]
    dev_acceptance = {
        "nonnegative_vs_route": dev_aggregate["net_success_delta_vs_route"] >= 0,
        "gain_retention_met": (
            dev_aggregate["route_gain_retention_rate"] >= minimum_gain_retention
        ),
        "regression_avoidance_met": (
            dev_aggregate["route_regression_avoidance_rate"]
            >= minimum_regression_avoidance
        ),
        "no_drop_worsening": dev_aggregate["drop_delta_vs_route"] <= 0,
        "no_unsafe_force_worsening": (
            dev_aggregate["unsafe_force_delta_vs_route"] <= 0
        ),
    }
    dev_acceptance["accepted"] = all(dev_acceptance.values())

    trainer_source = Path(__file__).resolve()
    trainer_source_sha256 = sha256_file(trainer_source)
    data_semantic_sha256 = semantic_data_sha256(dataset)
    chosen_feature_digest = hashlib.sha256(
        _canonical_tensor_bytes(selected_features[valid])
    ).hexdigest()
    model_metadata = {
        "dataset_kind": DATASET_KIND,
        "dataset_sha256": dataset_sha256,
        "data_semantic_sha256": data_semantic_sha256,
        "chosen_feature_semantic_sha256": chosen_feature_digest,
        "source_sha256": {
            "scripts/flashsac/train_recoverability_selector.py": trainer_source_sha256
        },
        "torch_version": str(torch.__version__),
        "max_iter": int(max_iter),
        "training_configuration": training_configuration,
        "feature_contract": feature_contract(selected_feature),
        "head_order": ["continue_search_strict_success", "route_flashsac_strict_success"],
        "decision_semantics": (
            "default route to FlashSAC; veto to continue SEARCH iff p_continue >= "
            "p_continue_min and p_continue - p_route >= continue_probability_margin"
        ),
        "p_continue_min": float(selected_threshold["p_continue_min"]),
        "continue_probability_margin": float(
            selected_threshold["continue_probability_margin"]
        ),
        "l2": selected_l2,
        "training_seeds": train_seeds,
        "development_seed": dev_seed,
        "training_rows": int(train_mask.sum()),
        "development_rows": int(dev_mask.sum()),
        "normalization": "training_seed_rows_only_population_std_constant_to_one",
        "dtype": "float64",
        "pairing_semantics": dataset["metadata"]["pairing_semantics"],
        "causal_counterfactual_claim_allowed": False,
        "collection_canonical_provenance": dataset["metadata"][
            "canonical_provenance"
        ],
    }
    model: dict[str, Any] = {
        "kind": MODEL_KIND,
        "format_version": 1,
        "metadata": model_metadata,
        "feature_mean": final_mean.contiguous(),
        "feature_scale": final_scale.contiguous(),
        "head_weight": final_weight.contiguous(),
        "head_bias": final_bias.contiguous(),
    }
    report: dict[str, Any] = {
        "status": "complete",
        "kind": MODEL_KIND,
        "format_version": 1,
        "dataset_sha256": dataset_sha256,
        "data_semantic_sha256": data_semantic_sha256,
        "trainer_source_sha256": trainer_source_sha256,
        "torch_version": str(torch.__version__),
        "max_iter": int(max_iter),
        "training_configuration": training_configuration,
        "pairing_caveat": (
            "independent GPU rollouts are diagnostic matched outcomes, not exact "
            "simulator-state-fork counterfactuals"
        ),
        "selection_bias_caveat": (
            "training-seed LOO predictions are reused to select feature/L2/threshold "
            "candidates, so their acceptance metrics are selection-set diagnostics, "
            "not an unbiased generalization estimate; the development seed is "
            "evaluated only after selection"
        ),
        "strong_rows": int(valid.sum()),
        "training_seeds": train_seeds,
        "development_seed": dev_seed,
        "feature_grid": feature_sets,
        "l2_grid": l2_values,
        "p_continue_min_grid": p_min_values,
        "continue_probability_margin_grid": margin_values,
        "hyperparameter_candidate_count": len(global_candidates),
        "hyperparameter_results": hyperparameter_results,
        "selected": {
            "feature_set": selected_feature,
            "feature_dimension": FEATURE_DIMS[selected_feature],
            "l2": selected_l2,
            "p_continue_min": float(selected_threshold["p_continue_min"]),
            "continue_probability_margin": float(
                selected_threshold["continue_probability_margin"]
            ),
            "loo_threshold_metrics": selected_threshold,
            "final_fit_diagnostics": final_fit_diagnostics,
        },
        "cross_validated_train_seed_report": {
            "model_quality": next(
                item["loo_model_quality"]
                for item in hyperparameter_results
                if item["feature_set"] == selected_feature
                and float(item["l2"]) == selected_l2
            ),
            "policy": selected_threshold,
            "acceptance": cv_acceptance,
        },
        "held_out_development_report": {
            "seed": dev_seed,
            "model_quality": _model_quality(dev_probabilities, dev_targets),
            "policy": dev_threshold,
            "acceptance": dev_acceptance,
        },
        "accepted_for_blind_simulation_evaluation": bool(
            cv_acceptance["accepted"] and dev_acceptance["accepted"]
        ),
        "model_metadata": model_metadata,
    }
    json.dumps(report, sort_keys=True, allow_nan=False)
    return model, report


def _publish_model_and_report(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Transactionally publish both outputs with no-clobber hard links."""

    if output.resolve() == report_path.resolve():
        raise ValueError("model and report outputs must differ")
    for path, label in ((output, "model"), (report_path, "report")):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{label} output already exists: {path}")
    model_temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    report_temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    for path in (model_temporary, report_temporary):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"temporary output already exists: {path}")
    published_model = False
    try:
        with model_temporary.open("xb") as stream:
            torch.save(dict(model), stream)
            stream.flush()
            os.fsync(stream.fileno())
        model_sha256 = sha256_file(model_temporary)
        final_report = {
            **dict(report),
            "model": str(output),
            "model_sha256": model_sha256,
        }
        with report_temporary.open("x", encoding="utf-8") as stream:
            json.dump(final_report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(model_temporary, output)
        published_model = True
        try:
            os.link(report_temporary, report_path)
        except BaseException:
            output.unlink()
            published_model = False
            raise
        return final_report
    finally:
        for path in (model_temporary, report_temporary):
            if path.exists() or path.is_symlink():
                path.unlink()
        if published_model and not report_path.exists():
            output.unlink()


def _float_grid(values: Sequence[str]) -> list[float]:
    output: list[float] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                output.append(float(token))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dev_seed", type=int, default=260)
    parser.add_argument(
        "--feature_sets",
        nargs="+",
        default=["local84_v1", "public_gate_v1", "raw_public_v1"],
        choices=sorted(FEATURE_DIMS),
    )
    parser.add_argument("--l2", nargs="+", default=["0.001,0.01,0.1,1.0"])
    parser.add_argument(
        "--p_continue_min",
        nargs="+",
        default=["0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,1.0"],
    )
    parser.add_argument(
        "--continue_margin",
        nargs="+",
        default=["0.0,0.025,0.05,0.10,0.15,0.20,0.30"],
    )
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--minimum_cv_net_gain", type=int, default=10)
    parser.add_argument("--minimum_gain_retention", type=float, default=0.75)
    parser.add_argument("--minimum_regression_avoidance", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset.is_symlink():
        raise FileNotFoundError(
            f"dataset must not be a symlink: {args.dataset}"
        )
    for path, label in ((args.output, "model"), (args.report, "report")):
        if path.is_symlink():
            raise FileExistsError(f"{label} output must not be a symlink: {path}")
    dataset_path = args.dataset.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset must be a regular non-symlink file: {dataset_path}")
    output = args.output.resolve()
    report_path = args.report.resolve()
    for path, label in ((output, "model"), (report_path, "report")):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{label} output already exists: {path}")
    dataset_sha256 = sha256_file(dataset_path)
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
    if sha256_file(dataset_path) != dataset_sha256:
        raise RuntimeError("recoverability dataset changed while being loaded")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, report = train_selector(
        dataset,
        dataset_sha256=dataset_sha256,
        dev_seed=args.dev_seed,
        feature_sets=args.feature_sets,
        l2_values=_float_grid(args.l2),
        p_min_values=_float_grid(args.p_continue_min),
        margin_values=_float_grid(args.continue_margin),
        max_iter=args.max_iter,
        minimum_cv_net_gain=args.minimum_cv_net_gain,
        minimum_gain_retention=args.minimum_gain_retention,
        minimum_regression_avoidance=args.minimum_regression_avoidance,
    )
    final_report = _publish_model_and_report(
        model, report, output=output, report_path=report_path
    )
    print(json.dumps(final_report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
