#!/usr/bin/env python3
"""Public-state randomized SEARCH/V6 route trial contract.

This module is deliberately simulation-free and does not import ``evaluate``.
The online helpers only inspect tensor metadata, so they can run every Isaac
step without introducing a CUDA-to-host synchronization.  Full numerical and
semantic checks happen when the CPU evidence artifact is built or validated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


OBSERVATION_DIM = 115
ACTION_DIM = 21
FEATURE_DIM = 165
CONFIGURED_MAX_EPISODE_LENGTH = 1000
# Isaac increments the buffer before the task checks >= max_length - 1.
TIMEOUT_ACTION_COUNT = 999
LAST_PREACTION_STEP = 998
# Keep the preregistered feature scaling independent of the reachable maximum.
EPISODE_STEP_DENOMINATOR = 999

READINESS_MIN_SCORE = 0.10
READINESS_HOLD_STEPS = 4
SUCCESS_HOLD_STEPS = 15
STRATUM_G_THRESHOLD = 0.35
STRATUM_CLOSE_THRESHOLD = 0.20

TRIAL_KIND = "pick_tool_public_randomized_route_trial_v2"
FORMAT_VERSION = 2
FEATURE_CONTRACT = "pick_tool_public_route_gate_feature165_v1"
ASSIGNMENT_VERSION = "sha256_rank_balanced_v1"
OUTCOME_CONTRACT = "pick_tool_reset_before_factual_full_task_outcome_v2"
PROPENSITY_ROUTE = 0.5

# Fixed, externally auditable feature layout.  Observation 86 is omitted by
# construction; it has no destination in this table.
FEATURE_LAYOUT: dict[str, tuple[int, int]] = {
    "observation_0_85": (0, 86),
    "observation_87_114": (86, 114),
    "ready_count_after_onehot5": (114, 119),
    "fork_used_before": (119, 120),
    "episode_step_div_999": (120, 121),
    "public_safety_progress2": (121, 123),
    "search_action21": (123, 144),
    "v6_route_action21": (144, 165),
}
STRATUM_NAMES = (
    "g_lt_0.35_close_lt_0.20",
    "g_lt_0.35_close_ge_0.20",
    "g_ge_0.35_close_lt_0.20",
    "g_ge_0.35_close_ge_0.20",
)

_METADATA_INPUT_FIELDS = {
    "seed",
    "num_envs",
    "assignment_salt",
    "replicate",
    "provenance",
}
_METADATA_FIELDS = _METADATA_INPUT_FIELDS | {
    "assignment_version",
    "propensity_route",
    "randomization_unit",
    "feature_contract",
    "feature_dim",
    "feature_layout",
    "observation_contract",
    "observation_dim",
    "action_dim",
    "configured_max_episode_length",
    "timeout_action_count",
    "last_preaction_step",
    "episode_step_denominator",
    "outcome_contract",
    "factual_outcomes_only",
}
_ROW_LONG_FIELDS = (
    "env_slot",
    "slot_episode_index",
    "episode_index",
    "candidate_step",
    "stratum",
    "outcome_episode_length",
)
_ROW_FLOAT_FIELDS = (
    "readiness_score",
    "outcome_max_true_clearance_m",
)
_ROW_BOOL_FIELDS = (
    "factual_treatment_route",
    "outcome_terminated",
    "outcome_truncated",
    "outcome_success",
    "outcome_failure",
    "outcome_time_out",
    "outcome_dropped",
    "outcome_unsafe_force",
    "outcome_unlatched_clearance_ge_5cm",
    "outcome_ever_grasped",
    "outcome_ever_clearance_ge_20cm",
)
ROW_FIELDS = frozenset({"feature", *_ROW_LONG_FIELDS, *_ROW_FLOAT_FIELDS, *_ROW_BOOL_FIELDS})
_TENSOR_FIELDS = frozenset({"assignment_route", "propensity_route", "feature", *_ROW_LONG_FIELDS, *_ROW_FLOAT_FIELDS, *_ROW_BOOL_FIELDS})


def _require_online_batch_tensor(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
    floating: bool = False,
) -> None:
    """Metadata-only validation; never reads a CUDA tensor value."""

    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if floating and not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point")


def update_public_readiness(
    observation: torch.Tensor,
    ready_count_before: torch.Tensor,
    fork_used_before: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the public-only four-frame, once-sticky readiness rule.

    The only observation values read are ``obs[92:96]``, ``obs[96]``,
    ``obs[106]``, plus ``obs[102]`` solely for the audit stratum.
    """

    if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
        raise ValueError("observation must be a rank-two tensor")
    if observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("observation must have shape [N,115]")
    if not observation.dtype.is_floating_point:
        raise TypeError("observation must be floating point")
    batch = observation.shape[0]
    _require_online_batch_tensor(
        "ready_count_before",
        ready_count_before,
        shape=(batch,),
        device=observation.device,
        dtype=torch.long,
    )
    _require_online_batch_tensor(
        "fork_used_before",
        fork_used_before,
        shape=(batch,),
        device=observation.device,
        dtype=torch.bool,
    )

    second_nonthumb = torch.topk(observation[:, 92:96], k=2, dim=-1).values[:, 1]
    score = torch.minimum(observation[:, 96], second_nonthumb)
    eligible = (observation[:, 106] == 0.0) & (score >= READINESS_MIN_SCORE)
    ready_count_after = torch.where(
        eligible,
        torch.clamp(ready_count_before + 1, max=READINESS_HOLD_STEPS),
        torch.zeros_like(ready_count_before),
    )
    trigger = (ready_count_after == READINESS_HOLD_STEPS) & (~fork_used_before)
    fork_used_after = fork_used_before | trigger
    stratum = (
        (score >= STRATUM_G_THRESHOLD).long() * 2
        + (observation[:, 102] >= STRATUM_CLOSE_THRESHOLD).long()
    )
    return {
        "score": score,
        "eligible": eligible,
        "ready_count_before": ready_count_before,
        "ready_count_after": ready_count_after,
        "trigger": trigger,
        "fork_used_before": fork_used_before,
        "fork_used_after": fork_used_after,
        "stratum": stratum,
    }


def build_gate_feature(
    observation: torch.Tensor,
    ready_count_after: torch.Tensor,
    fork_used_before: torch.Tensor,
    episode_step: torch.Tensor,
    public_safety: torch.Tensor,
    search_action: torch.Tensor,
    route_action: torch.Tensor,
) -> torch.Tensor:
    """Build the fixed 165-D gate feature without ever copying ``obs[86]``."""

    if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
        raise ValueError("observation must be a rank-two tensor")
    if observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("observation must have shape [N,115]")
    if not observation.dtype.is_floating_point:
        raise TypeError("observation must be floating point")
    batch = observation.shape[0]
    device = observation.device
    dtype = observation.dtype
    for name, value, shape, expected_dtype, floating in (
        ("ready_count_after", ready_count_after, (batch,), torch.long, False),
        ("fork_used_before", fork_used_before, (batch,), torch.bool, False),
        ("episode_step", episode_step, (batch,), torch.long, False),
        ("public_safety", public_safety, (batch, 2), dtype, True),
        ("search_action", search_action, (batch, ACTION_DIM), dtype, True),
        ("route_action", route_action, (batch, ACTION_DIM), dtype, True),
    ):
        _require_online_batch_tensor(
            name,
            value,
            shape=shape,
            device=device,
            dtype=expected_dtype,
            floating=floating,
        )
    public_observation = torch.cat((observation[:, :86], observation[:, 87:]), dim=-1)
    ready_onehot = F.one_hot(ready_count_after, num_classes=5).to(dtype=dtype)
    feature = torch.cat(
        (
            public_observation,
            ready_onehot,
            fork_used_before[:, None].to(dtype=dtype),
            episode_step[:, None].to(dtype=dtype)
            / float(EPISODE_STEP_DENOMINATOR),
            public_safety,
            search_action,
            route_action,
        ),
        dim=-1,
    )
    if feature.shape != (batch, FEATURE_DIM):
        raise RuntimeError("internal feature layout produced the wrong dimension")
    return feature


def complementary_replicate(replicate: str) -> str:
    """Return the adjacent replicate that reverses every treatment."""

    if replicate not in {"a", "b"}:
        raise ValueError("replicate must be 'a' or 'b'")
    return "b" if replicate == "a" else "a"


def assignment_for(
    cohort: str,
    seed: int,
    num_envs: int,
    salt: str,
    replicate: str,
) -> torch.Tensor:
    """Return a balanced, deterministic route mask; adjacent replicates complement."""

    if not isinstance(cohort, str) or not cohort:
        raise ValueError("cohort must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or num_envs % 2 != 0
    ):
        raise ValueError("num_envs must be a positive even integer")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    if replicate not in {"a", "b"}:
        raise ValueError("replicate must be 'a' or 'b'")
    scored: list[tuple[bytes, int]] = []
    for slot in range(num_envs):
        # This byte sequence is frozen in public_route_trial_manifest.json.
        message = (
            salt
            + "\0"
            + cohort
            + "\0"
            + str(seed)
            + "\0"
            + str(slot)
            + "\0episode=0"
        ).encode("utf-8")
        scored.append((hashlib.sha256(message).digest(), slot))
    order = [slot for _, slot in sorted(scored)]
    base = torch.zeros(num_envs, dtype=torch.bool)
    base[order[: num_envs // 2]] = True
    return ~base if replicate == "b" else base


def assignment_route_mask(
    *, cohort: str, seed: int, num_envs: int, salt: str, replicate: str
) -> torch.Tensor:
    """Keyword-only collector spelling of :func:`assignment_for`."""

    return assignment_for(cohort, seed, num_envs, salt, replicate)


def _plain_json(value: Any, *, path: str = "provenance") -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not torch.isfinite(torch.tensor(value)).item():
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _plain_json(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain_json(item, path=f"{path}[]") for item in value]
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _normalize_metadata(metadata: Mapping[str, Any], *, input_form: bool) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    expected = _METADATA_INPUT_FIELDS if input_form else _METADATA_FIELDS
    if set(metadata) != expected:
        raise KeyError(f"metadata fields must be exactly {sorted(expected)}")
    seed = metadata["seed"]
    num_envs = metadata["num_envs"]
    replicate = metadata["replicate"]
    salt = metadata["assignment_salt"]
    provenance = _plain_json(metadata["provenance"])
    if not isinstance(provenance, dict):
        raise TypeError("provenance must be a mapping")
    cohort = provenance.get("cohort")
    assignment_for(cohort, seed, num_envs, salt, replicate)  # validates inputs
    fixed = {
        "seed": seed,
        "num_envs": num_envs,
        "assignment_salt": salt,
        "replicate": replicate,
        "provenance": provenance,
        "assignment_version": ASSIGNMENT_VERSION,
        "propensity_route": PROPENSITY_ROUTE,
        "randomization_unit": "cohort_x_seed_x_env_slot_x_episode0",
        "feature_contract": FEATURE_CONTRACT,
        "feature_dim": FEATURE_DIM,
        "feature_layout": {key: list(value) for key, value in FEATURE_LAYOUT.items()},
        "observation_contract": "pick_tool_markov115_v1_minus_obs86",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "configured_max_episode_length": CONFIGURED_MAX_EPISODE_LENGTH,
        "timeout_action_count": TIMEOUT_ACTION_COUNT,
        "last_preaction_step": LAST_PREACTION_STEP,
        "episode_step_denominator": EPISODE_STEP_DENOMINATOR,
        "outcome_contract": OUTCOME_CONTRACT,
        "factual_outcomes_only": True,
    }
    if not input_form and dict(metadata) != fixed:
        raise ValueError("artifact metadata disagrees with the fixed trial contract")
    return fixed


def _cpu_tensor(name: str, value: Any, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be a tensor with shape {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    return value.contiguous()


def _validate_semantics(payload: Mapping[str, Any]) -> None:
    metadata = payload["metadata"]
    tensors = payload["tensors"]
    count = tensors["feature"].shape[0]
    assigned = tensors["assignment_route"]
    expected_assignment = assignment_for(
        metadata["provenance"]["cohort"],
        metadata["seed"],
        metadata["num_envs"],
        metadata["assignment_salt"],
        metadata["replicate"],
    )
    if not torch.equal(assigned, expected_assignment):
        raise ValueError("assignment_route does not match the registered SHA256 assignment")
    slots = tensors["env_slot"]
    if bool(((slots < 0) | (slots >= metadata["num_envs"])).any()):
        raise ValueError("env_slot is outside the assignment")
    if count and not torch.equal(tensors["factual_treatment_route"], assigned[slots]):
        raise ValueError("factual treatment does not match randomized assignment")
    if not bool((tensors["propensity_route"] == PROPENSITY_ROUTE).all()):
        raise ValueError("every factual row must have route propensity 0.5")
    slot_episode = tensors["slot_episode_index"]
    if not bool((slot_episode == 0).all()):
        raise ValueError("trial evidence allows only the first episode per env slot")
    if len(slots.tolist()) != len(set(slots.tolist())):
        raise ValueError("duplicate first-episode env_slot trial row")
    episode_indices = tensors["episode_index"]
    if bool(
        ((episode_indices < 0) | (episode_indices >= metadata["num_envs"])).any()
    ) or len(episode_indices.tolist()) != len(set(episode_indices.tolist())):
        raise ValueError("episode_index must be a unique first-episode completion index")

    success = tensors["outcome_success"]
    failure = tensors["outcome_failure"]
    timeout = tensors["outcome_time_out"]
    terminated = tensors["outcome_terminated"]
    truncated = tensors["outcome_truncated"]
    if not torch.equal(terminated, success | failure):
        raise ValueError("terminated must equal success OR failure")
    if not torch.equal(truncated, timeout):
        raise ValueError("truncated must equal time_out")
    if not bool(((success.long() + failure.long() + timeout.long()) == 1).all()):
        raise ValueError("each factual outcome must have exactly one terminal disposition")
    failure_truth = (
        tensors["outcome_dropped"]
        | tensors["outcome_unsafe_force"]
        | tensors["outcome_unlatched_clearance_ge_5cm"]
    )
    if not torch.equal(failure, failure_truth):
        raise ValueError("failure must equal drop OR unsafe force OR unlatched lift")
    strict_success_invalid = success & (
        ~tensors["outcome_ever_grasped"]
        | ~tensors["outcome_ever_clearance_ge_20cm"]
        | tensors["outcome_dropped"]
        | tensors["outcome_unsafe_force"]
        | (tensors["outcome_max_true_clearance_m"] < 0.20)
    )
    if bool(strict_success_invalid.any()):
        raise ValueError("strict success violates grasp, 20 cm, or safety truth")
    clearance = tensors["outcome_max_true_clearance_m"]
    if not torch.equal(tensors["outcome_ever_clearance_ge_20cm"], clearance >= 0.20):
        raise ValueError("20 cm event disagrees with maximum true clearance")
    if bool(
        (
            tensors["outcome_unlatched_clearance_ge_5cm"]
            & (clearance < 0.05)
        ).any()
    ):
        raise ValueError("unlatched 5 cm event exceeds maximum true clearance")

    steps = tensors["candidate_step"]
    lengths = tensors["outcome_episode_length"]
    if bool(
        ((steps < READINESS_HOLD_STEPS - 1) | (steps > LAST_PREACTION_STEP)).any()
    ):
        raise ValueError("candidate_step is earlier than the four-frame gate or above 998")
    if bool(((lengths <= steps) | (lengths > TIMEOUT_ACTION_COUNT)).any()):
        raise ValueError("outcome episode length is inconsistent with candidate_step")
    if bool((timeout & (lengths != TIMEOUT_ACTION_COUNT)).any()):
        raise ValueError("full-task timeout must occur after 999 executed actions")
    if bool((success & ((lengths - steps) < SUCCESS_HOLD_STEPS)).any()):
        raise ValueError("unlatched trigger leaves too few actions for strict success hold")
    if bool(((tensors["stratum"] < 0) | (tensors["stratum"] > 3)).any()):
        raise ValueError("stratum is outside [0,3]")

    feature = tensors["feature"]
    onehot = feature[:, 114:119]
    if not torch.equal(onehot, F.one_hot(torch.full((count,), 4), 5).float()):
        raise ValueError("trial rows must be the four-frame readiness trigger")
    if not bool((feature[:, 119] == 0.0).all()):
        raise ValueError("trial trigger must occur before the sticky fork is used")
    expected_step = steps.float() / float(EPISODE_STEP_DENOMINATOR)
    if not torch.allclose(feature[:, 120], expected_step, rtol=0.0, atol=1.0e-7):
        raise ValueError("feature episode step disagrees with candidate_step")
    hard_count = feature[:, 121] * 10.0
    overforce_count = feature[:, 122] * 2.0
    if bool(
        (hard_count < 0.0).any()
        or (hard_count > 9.0 + 1.0e-6).any()
        or (~torch.isclose(hard_count, hard_count.round(), rtol=0.0, atol=1.0e-6)).any()
        or (overforce_count < 0.0).any()
        or (overforce_count > 1.0 + 1.0e-6).any()
        or (
            ~torch.isclose(
                overforce_count,
                overforce_count.round(),
                rtol=0.0,
                atol=1.0e-6,
            )
        ).any()
        or (hard_count + 1.0e-6 < overforce_count).any()
    ):
        raise ValueError("public safety progress is unreachable for a surviving 10/2-step state")
    minimum_unsafe_actions = torch.minimum(
        10 - hard_count.round().long(),
        2 - overforce_count.round().long(),
    )
    if bool(
        (
            tensors["outcome_unsafe_force"]
            & ((lengths - steps) < minimum_unsafe_actions)
        ).any()
    ):
        raise ValueError("unsafe-force outcome occurs before either public counter can terminate")
    if bool((feature[:, 123:165].abs() > 1.0 + 1.0e-6).any()):
        raise ValueError("candidate actions exceed normalized action bounds")
    if not bool((feature[:, 144:151] == 0.0).all()):
        raise ValueError("unlatched V6 candidate must have zero arm authority")
    if bool((feature[:, 70:91].abs() > 1.0 + 1.0e-6).any()):
        raise ValueError("previous executed actions exceed normalized action bounds")
    if bool(((feature[:, 91:104] < 0.0) | (feature[:, 91:104] > 1.0)).any()):
        raise ValueError("public contact, close, wrap, and hold features must lie in [0,1]")
    if not bool((feature[:, 104] == 1.0).all()):
        raise ValueError("unlatched trigger must expose exact one-minus-latch state")
    confirm_count = feature[:, 106] * 4.0
    release_count = feature[:, 107] * 6.0
    if bool(
        (confirm_count < 0.0).any()
        or (confirm_count > 3.0 + 1.0e-6).any()
        or (
            ~torch.isclose(
                confirm_count, confirm_count.round(), rtol=0.0, atol=1.0e-6
            )
        ).any()
        or (release_count < 0.0).any()
        or (release_count > 6.0 + 1.0e-6).any()
        or (
            ~torch.isclose(
                release_count, release_count.round(), rtol=0.0, atol=1.0e-6
            )
        ).any()
    ):
        raise ValueError("public latch confirmation/release progress is unreachable")
    if bool((feature[:, 108:114].abs() > 2.0 + 1.0e-6).any()):
        raise ValueError("public palm-frame slip features exceed their clipped range")
    second = torch.topk(feature[:, 91:95], k=2, dim=-1).values[:, 1]
    score = torch.minimum(feature[:, 95], second)
    if not torch.allclose(score, tensors["readiness_score"], rtol=0.0, atol=1.0e-6):
        raise ValueError("readiness_score disagrees with public observation feature")
    if bool(((feature[:, 105] != 0.0) | (score < READINESS_MIN_SCORE)).any()):
        raise ValueError("trial row does not satisfy the public readiness gate")
    expected_stratum = (
        (score >= STRATUM_G_THRESHOLD).long() * 2
        + (feature[:, 101] >= STRATUM_CLOSE_THRESHOLD).long()
    )
    if not torch.equal(expected_stratum, tensors["stratum"]):
        raise ValueError("stratum disagrees with g/close thresholds")
    impossible_unlatched = (
        (~tensors["outcome_ever_grasped"])
        & (clearance >= 0.05)
        & (~tensors["outcome_unlatched_clearance_ge_5cm"])
    )
    if bool(impossible_unlatched.any()):
        raise ValueError("never-grasped 5 cm clearance must be an unlatched-lift failure")


def validate_trial_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed validation for a weights-only-loaded factual trial artifact."""

    if not isinstance(payload, Mapping) or set(payload) != {"kind", "format_version", "metadata", "tensors"}:
        raise ValueError("trial artifact has unexpected top-level fields")
    if payload.get("kind") != TRIAL_KIND or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported public route trial artifact")
    metadata = _normalize_metadata(payload.get("metadata"), input_form=False)
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != _TENSOR_FIELDS:
        raise ValueError("trial artifact has unexpected tensor fields")
    count_value = tensors.get("feature")
    if not isinstance(count_value, torch.Tensor) or count_value.ndim != 2:
        raise ValueError("feature must be a rank-two tensor")
    count = count_value.shape[0]
    normalized: dict[str, torch.Tensor] = {
        "assignment_route": _cpu_tensor("assignment_route", tensors["assignment_route"], (metadata["num_envs"],), torch.bool),
        "propensity_route": _cpu_tensor("propensity_route", tensors["propensity_route"], (count,), torch.float32),
        "feature": _cpu_tensor("feature", tensors["feature"], (count, FEATURE_DIM), torch.float32),
    }
    for name in _ROW_LONG_FIELDS:
        normalized[name] = _cpu_tensor(name, tensors[name], (count,), torch.long)
    for name in _ROW_FLOAT_FIELDS:
        normalized[name] = _cpu_tensor(name, tensors[name], (count,), torch.float32)
    for name in _ROW_BOOL_FIELDS:
        normalized[name] = _cpu_tensor(name, tensors[name], (count,), torch.bool)
    result = {"kind": TRIAL_KIND, "format_version": FORMAT_VERSION, "metadata": metadata, "tensors": normalized}
    _validate_semantics(result)
    return result


def build_trial_artifact(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    assignment_route: torch.Tensor,
) -> dict[str, Any]:
    """Build one immutable-schema artifact containing assigned-arm facts only."""

    normalized_metadata = _normalize_metadata(metadata, input_form=True)
    assignment = _cpu_tensor(
        "assignment_route", assignment_route, (normalized_metadata["num_envs"],), torch.bool
    )
    columns: dict[str, list[Any]] = {name: [] for name in ROW_FIELDS}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != ROW_FIELDS:
            raise KeyError(f"trial row {index} fields must be exactly {sorted(ROW_FIELDS)}")
        feature = _cpu_tensor(f"row {index} feature", row["feature"], (FEATURE_DIM,), torch.float32)
        columns["feature"].append(feature)
        for name in _ROW_LONG_FIELDS:
            if isinstance(row[name], bool) or not isinstance(row[name], int):
                raise TypeError(f"trial row {index} {name} must be a Python int")
            columns[name].append(row[name])
        for name in _ROW_FLOAT_FIELDS:
            if type(row[name]) is not float:
                raise TypeError(f"trial row {index} {name} must be a Python float")
            columns[name].append(row[name])
        for name in _ROW_BOOL_FIELDS:
            if type(row[name]) is not bool:
                raise TypeError(f"trial row {index} {name} must be a Python bool")
            columns[name].append(row[name])
    count = len(rows)
    tensors: dict[str, torch.Tensor] = {
        "assignment_route": assignment.clone(),
        "propensity_route": torch.full((count,), PROPENSITY_ROUTE, dtype=torch.float32),
        "feature": torch.stack(columns["feature"]) if count else torch.empty((0, FEATURE_DIM), dtype=torch.float32),
    }
    for name in _ROW_LONG_FIELDS:
        tensors[name] = torch.tensor(columns[name], dtype=torch.long)
    for name in _ROW_FLOAT_FIELDS:
        tensors[name] = torch.tensor(columns[name], dtype=torch.float32)
    for name in _ROW_BOOL_FIELDS:
        tensors[name] = torch.tensor(columns[name], dtype=torch.bool)
    return validate_trial_artifact(
        {"kind": TRIAL_KIND, "format_version": FORMAT_VERSION, "metadata": normalized_metadata, "tensors": tensors}
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_trial_artifact_no_clobber(payload: Mapping[str, Any], output: Path) -> str:
    """Validate, fsync, and atomically hard-link an immutable PT artifact."""

    normalized = validate_trial_artifact(payload)
    output = Path(os.path.abspath(os.fspath(output)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"trial artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            torch.save(normalized, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return digest
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    normalized = _plain_json(payload, path="report")
    if not isinstance(normalized, dict):
        raise TypeError("report must be a mapping")
    return (json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    """Publish strict JSON without replacing a file or dangling symlink."""

    output = Path(os.path.abspath(os.fspath(output)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
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


def publish_trial_and_report_no_clobber(
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    artifact_output: Path,
    report_output: Path,
) -> str:
    """Transactionally publish one validated PT artifact and its JSON report."""

    normalized = validate_trial_artifact(artifact)
    artifact_output = Path(os.path.abspath(os.fspath(artifact_output)))
    report_output = Path(os.path.abspath(os.fspath(report_output)))
    if artifact_output == report_output:
        raise ValueError("artifact and report outputs must differ")
    for path in (artifact_output, report_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"trial evidence output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    artifact_fd, artifact_name = tempfile.mkstemp(
        prefix=f".{artifact_output.name}.tmp-", dir=artifact_output.parent
    )
    os.close(artifact_fd)
    report_fd, report_name = tempfile.mkstemp(
        prefix=f".{report_output.name}.tmp-", dir=report_output.parent
    )
    artifact_temp, report_temp = Path(artifact_name), Path(report_name)
    linked: list[Path] = []
    try:
        with artifact_temp.open("wb") as stream:
            torch.save(normalized, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(artifact_temp)
        final_report = {**dict(report), "artifact_sha256": digest}
        with os.fdopen(report_fd, "wb") as stream:
            stream.write(_strict_json_bytes(final_report))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(artifact_temp, artifact_output)
        linked.append(artifact_output)
        os.link(report_temp, report_output)
        linked.append(report_output)
        for parent in {artifact_output.parent, report_output.parent}:
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return digest
    except BaseException:
        for path in reversed(linked):
            if path.exists() or path.is_symlink():
                path.unlink()
        raise
    finally:
        for path in (artifact_temp, report_temp):
            if path.exists() or path.is_symlink():
                path.unlink()


# Collector-facing names retain the manifest vocabulary while the shorter
# names above remain convenient for offline training code.
update_public_route_readiness = update_public_readiness
build_public_route_feature = build_gate_feature
build_randomized_trial_artifact = build_trial_artifact
