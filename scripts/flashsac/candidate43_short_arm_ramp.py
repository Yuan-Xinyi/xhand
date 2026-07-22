#!/usr/bin/env python3
"""Pure public-state D8 arm-authority ramp for Candidate43.

The common Candidate39 action is evaluated before this module.  Treatment
may scale only its seven arm coordinates after public latch eligibility; the
fourteen hand coordinates, SEARCH actions, and all control actions remain
bit-exact.  Candidate42's public stability predicate and record schemas are
reused directly.  The only scientific change is the treatment clock clamp
and float32 authority denominator: 15 becomes 8.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch

import candidate42_public_arm_ramp as candidate42


OBSERVATION_DIM = candidate42.OBSERVATION_DIM
ACTION_DIM = candidate42.ACTION_DIM
ARM_ACTION_DIM = candidate42.ARM_ACTION_DIM
HAND_ACTION_DIM = candidate42.HAND_ACTION_DIM

FORCE_STRENGTH_START = candidate42.FORCE_STRENGTH_START
FORCE_STRENGTH_STOP = candidate42.FORCE_STRENGTH_STOP
STRICT_WRAP_QUALITY_INDEX = candidate42.STRICT_WRAP_QUALITY_INDEX
HOLD_QUALITY_INDEX = candidate42.HOLD_QUALITY_INDEX
PUBLIC_LATCH_INDEX = candidate42.PUBLIC_LATCH_INDEX
GRASP_QUALITY_MIN = candidate42.GRASP_QUALITY_MIN
HOLD_QUALITY_MIN = candidate42.HOLD_QUALITY_MIN
PUBLIC_FORCE_STRENGTH_MAX = candidate42.PUBLIC_FORCE_STRENGTH_MAX
SAFE_FORCE_STRENGTH_MAX = candidate42.SAFE_FORCE_STRENGTH_MAX
SAFE_FORCE_LIMIT_N = candidate42.SAFE_FORCE_LIMIT_N
FORCE_SATURATION_N = candidate42.FORCE_SATURATION_N
PUBLIC_FORCE_COUNTER_CONTRACT = candidate42.PUBLIC_FORCE_COUNTER_CONTRACT
PUBLIC_FORCE_COUNTER_FEATURES = candidate42.PUBLIC_FORCE_COUNTER_FEATURES

VERIFY_STEPS = 8
RAMP_DENOMINATOR = 8.0
SUPERVISOR_CONTRACT = "public_short_linear_arm_authority_ramp_d8_v1"
ASSIGNMENT_CONTRACT = (
    "candidate43_short_public_arm_ramp_sha256_rank_exact_balanced_complement_v1"
)
ASSIGNMENT_SALT = (
    "pick_tool_candidate43_short_public_arm_ramp_development_20260723_v1"
)
ALLOWED_RUN_NUM_ENVS = frozenset({8, 64})

PublicStableState = candidate42.PublicStableState
PublicArmRampState = candidate42.PublicArmRampState
PublicArmRampStep = candidate42.PublicArmRampStep
public_stable_state = candidate42.public_stable_state


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ranked_slots(*, seed: int, num_envs: int) -> list[int]:
    if not _is_int(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not _is_int(num_envs) or num_envs not in ALLOWED_RUN_NUM_ENVS:
        raise ValueError("num_envs must be exactly 8 or 64")
    return sorted(
        range(num_envs),
        key=lambda slot: (
            hashlib.sha256(
                f"{ASSIGNMENT_SALT}\0rank\0{seed}\0{slot}".encode("utf-8")
            ).digest(),
            slot,
        ),
    )


def exact_balanced_treatment_mask(
    *, seed: int, num_envs: int, replicate: str
) -> torch.Tensor:
    """Return the sealed Candidate43 assignment; replicate B complements A."""

    if replicate not in {"a", "b"}:
        raise ValueError("replicate must be 'a' or 'b'")
    ranked = _ranked_slots(seed=seed, num_envs=num_envs)
    assignment_a = torch.zeros(num_envs, dtype=torch.bool)
    assignment_a[torch.tensor(ranked[: num_envs // 2], dtype=torch.long)] = True
    return assignment_a if replicate == "a" else ~assignment_a


def assignment_rank(*, seed: int, num_envs: int) -> torch.Tensor:
    ranked = _ranked_slots(seed=seed, num_envs=num_envs)
    result = torch.empty(num_envs, dtype=torch.long)
    result[torch.tensor(ranked, dtype=torch.long)] = torch.arange(
        num_envs, dtype=torch.long
    )
    return result


def assignment_mask_sha256(mask: torch.Tensor) -> str:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 1
        or mask.dtype != torch.bool
        or mask.device.type != "cpu"
        or int(mask.numel()) not in ALLOWED_RUN_NUM_ENVS
    ):
        raise ValueError(
            "assignment receipt requires a rank-one 8- or 64-element CPU bool tensor"
        )
    encoded = bytes(int(value) for value in mask.tolist())
    return hashlib.sha256(
        ASSIGNMENT_SALT.encode("utf-8") + b"\0mask-v1\0" + encoded
    ).hexdigest()


def _require_bool_vector(
    name: str, value: torch.Tensor, *, rows: int, device: torch.device
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (rows,)
        or value.dtype != torch.bool
        or value.device != device
    ):
        raise ValueError(f"{name} must be a [{rows}] bool tensor on {device}")
    return value


def initial_public_arm_ramp_state(
    num_envs: int, *, device: torch.device | str | None = None
) -> PublicArmRampState:
    if not _is_int(num_envs) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    target = torch.device("cpu") if device is None else torch.device(device)
    return PublicArmRampState(
        activated=torch.zeros(num_envs, dtype=torch.bool, device=target),
        stable_count=torch.zeros(num_envs, dtype=torch.long, device=target),
        ramp_scale_previous=torch.zeros(
            num_envs, dtype=torch.float32, device=target
        ),
        ever_positive_authority=torch.zeros(
            num_envs, dtype=torch.bool, device=target
        ),
        ever_full_authority=torch.zeros(
            num_envs, dtype=torch.bool, device=target
        ),
    )


def validate_public_arm_ramp_state(
    state: PublicArmRampState, *, rows: int, device: torch.device
) -> None:
    if not isinstance(state, PublicArmRampState):
        raise TypeError("state must be PublicArmRampState")
    for name in (
        "activated",
        "ever_positive_authority",
        "ever_full_authority",
    ):
        _require_bool_vector(
            f"state.{name}", getattr(state, name), rows=rows, device=device
        )
    count = state.stable_count
    if (
        not isinstance(count, torch.Tensor)
        or count.shape != (rows,)
        or count.dtype != torch.long
        or count.device != device
    ):
        raise ValueError(f"state.stable_count must be a [{rows}] long tensor on {device}")
    scale = state.ramp_scale_previous
    if (
        not isinstance(scale, torch.Tensor)
        or scale.shape != (rows,)
        or scale.dtype != torch.float32
        or scale.device != device
    ):
        raise ValueError(
            f"state.ramp_scale_previous must be a [{rows}] float32 tensor on {device}"
        )
    if bool(((count < 0) | (count > VERIFY_STEPS)).any()):
        raise ValueError("state.stable_count escaped [0, 8]")
    if not bool(torch.isfinite(scale).all()) or bool(
        ((scale < 0.0) | (scale > 1.0)).any()
    ):
        raise ValueError("state.ramp_scale_previous escaped finite [0, 1]")
    if bool(((~state.activated) & (count != 0)).any()) or bool(
        ((~state.activated) & (scale != 0.0)).any()
    ):
        raise ValueError("an unactivated slot must have a zero clock and authority")
    if bool(
        ((~state.activated) & state.ever_positive_authority).any()
    ) or bool(((~state.activated) & state.ever_full_authority).any()):
        raise ValueError("an unactivated slot cannot retain authority history")
    if bool(((count == 0) & (scale != 0.0)).any()):
        raise ValueError("a zero stable clock requires zero prior ramp authority")
    expected_from_clock = (
        torch.clamp(count - 1, min=0).to(dtype=torch.float32)
        / torch.tensor(RAMP_DENOMINATOR, dtype=torch.float32, device=device)
    )
    unsaturated = (count > 0) & (count < VERIFY_STEPS)
    if bool((unsaturated & (scale != expected_from_clock)).any()):
        raise ValueError("prior ramp authority disagrees with the public clock")
    saturated_valid = (scale == expected_from_clock) | (scale == 1.0)
    if bool(((count == VERIFY_STEPS) & (~saturated_valid)).any()):
        raise ValueError("saturated prior ramp authority disagrees with the public clock")
    if bool(((scale > 0.0) & (~state.ever_positive_authority)).any()):
        raise ValueError("positive prior authority requires ever-positive memory")
    if bool(((scale == 1.0) & (~state.ever_full_authority)).any()):
        raise ValueError("full prior authority requires ever-full memory")
    if bool((state.ever_full_authority & (~state.ever_positive_authority)).any()):
        raise ValueError("ever-full authority implies ever-positive authority")


def reset_public_arm_ramp_state(
    state: PublicArmRampState, reset_mask: torch.Tensor
) -> PublicArmRampState:
    rows = int(state.activated.numel())
    device = state.activated.device
    validate_public_arm_ramp_state(state, rows=rows, device=device)
    reset = _require_bool_vector(
        "reset_mask", reset_mask, rows=rows, device=device
    )
    zero_bool = torch.zeros_like(state.activated)
    return PublicArmRampState(
        activated=torch.where(reset, zero_bool, state.activated),
        stable_count=torch.where(
            reset, torch.zeros_like(state.stable_count), state.stable_count
        ),
        ramp_scale_previous=torch.where(
            reset,
            torch.zeros_like(state.ramp_scale_previous),
            state.ramp_scale_previous,
        ),
        ever_positive_authority=torch.where(
            reset, zero_bool, state.ever_positive_authority
        ),
        ever_full_authority=torch.where(
            reset, zero_bool, state.ever_full_authority
        ),
    )


def apply_public_arm_ramp(
    baseline_action: torch.Tensor,
    observation: torch.Tensor,
    public_force_counters: Mapping[str, torch.Tensor] | torch.Tensor,
    option_active: torch.Tensor,
    episode_active: torch.Tensor,
    treatment: torch.Tensor,
    state: PublicArmRampState,
    *,
    reset_mask: torch.Tensor,
) -> PublicArmRampStep:
    """Apply the preregistered float32 ``min(count_before, 8) / 8`` ramp."""

    if (
        not isinstance(baseline_action, torch.Tensor)
        or baseline_action.ndim != 2
        or baseline_action.shape[1] != ACTION_DIM
        or baseline_action.dtype != torch.float32
    ):
        raise ValueError(f"baseline_action must be [batch, {ACTION_DIM}] float32")
    if (
        not isinstance(observation, torch.Tensor)
        or observation.shape != (baseline_action.shape[0], OBSERVATION_DIM)
        or observation.dtype != torch.float32
        or observation.device != baseline_action.device
    ):
        raise ValueError(
            f"observation must be [batch, {OBSERVATION_DIM}] float32 on the action device"
        )
    if not bool(torch.isfinite(baseline_action).all()) or bool(
        (baseline_action.abs() > 1.0).any()
    ):
        raise ValueError("baseline action must be finite and bounded by one")
    rows = int(baseline_action.shape[0])
    device = baseline_action.device
    active = _require_bool_vector(
        "episode_active", episode_active, rows=rows, device=device
    )
    option = _require_bool_vector(
        "option_active", option_active, rows=rows, device=device
    )
    assigned = _require_bool_vector(
        "treatment", treatment, rows=rows, device=device
    )
    validate_public_arm_ramp_state(state, rows=rows, device=device)
    working = reset_public_arm_ramp_state(state, reset_mask)
    public = public_stable_state(
        observation, public_force_counters, option, active
    )

    eligible_now = active & option & public.latch
    first_eligible = eligible_now & (~working.activated)
    activated = working.activated | first_eligible
    count_before = working.stable_count
    denominator = torch.tensor(
        RAMP_DENOMINATOR, dtype=torch.float32, device=device
    )
    ramp_scale = torch.where(
        public.stable,
        torch.clamp(count_before, max=VERIFY_STEPS).to(dtype=torch.float32)
        / denominator,
        torch.zeros(rows, dtype=torch.float32, device=device),
    )
    overlay_active = active & assigned & activated & option
    authority_scale = torch.where(
        overlay_active,
        ramp_scale,
        torch.ones(rows, dtype=torch.float32, device=device),
    )
    action = baseline_action.clone()
    action[:, :ARM_ACTION_DIM] = (
        baseline_action[:, :ARM_ACTION_DIM] * authority_scale.unsqueeze(-1)
    )
    if not torch.equal(
        action[:, ARM_ACTION_DIM:], baseline_action[:, ARM_ACTION_DIM:]
    ):
        raise RuntimeError("Candidate43 changed the common hand14 action")
    control = active & (~assigned)
    if bool(control.any()) and not torch.equal(
        action[control], baseline_action[control]
    ):
        raise RuntimeError("Candidate43 changed a control action")
    search = active & (~option)
    if bool(search.any()) and not torch.equal(
        action[search], baseline_action[search]
    ):
        raise RuntimeError("Candidate43 changed a SEARCH action")

    candidate_count = torch.where(
        public.stable,
        torch.clamp(count_before + 1, max=VERIFY_STEPS),
        torch.zeros_like(count_before),
    )
    count_after = torch.where(active, candidate_count, count_before)
    raw_positive = ramp_scale > 0.0
    raw_full = ramp_scale == 1.0
    event_domain = active & assigned & activated
    positive_this_action = overlay_active & raw_positive
    full_this_action = overlay_active & raw_full
    newly_positive = (
        positive_this_action & (~working.ever_positive_authority)
    )
    newly_full = full_this_action & (~working.ever_full_authority)
    relock = (
        event_domain
        & (working.ramp_scale_previous > 0.0)
        & (~raw_positive)
    )
    reramp = (
        positive_this_action
        & working.ever_positive_authority
        & (working.ramp_scale_previous == 0.0)
    )
    verification_complete = (
        active
        & activated
        & public.stable
        & (count_before == VERIFY_STEPS - 1)
    )

    ramp_previous_after = torch.where(
        active, ramp_scale, working.ramp_scale_previous
    )
    ever_positive_after = working.ever_positive_authority | (
        active & activated & raw_positive
    )
    ever_full_after = working.ever_full_authority | (
        active & activated & raw_full
    )
    next_state = PublicArmRampState(
        activated=activated,
        stable_count=count_after,
        ramp_scale_previous=ramp_previous_after,
        ever_positive_authority=ever_positive_after,
        ever_full_authority=ever_full_after,
    )
    validate_public_arm_ramp_state(next_state, rows=rows, device=device)
    return PublicArmRampStep(
        action=action,
        next_state=next_state,
        eligible_now=eligible_now,
        first_eligible=first_eligible,
        activated_before=working.activated,
        activated_this_action=activated,
        stable_current=public.stable,
        public_grasp_quality=public.grasp_quality,
        public_hold_quality=public.hold_quality,
        public_max_force_strength=public.max_force_strength,
        force_counter_features=public.force_counter_features,
        stable_count_before=count_before,
        stable_count_after=count_after,
        ramp_scale=ramp_scale,
        authority_scale=authority_scale,
        positive_authority_this_action=positive_this_action,
        full_authority_this_action=full_this_action,
        verification_complete=verification_complete,
        newly_positive_authority=newly_positive,
        newly_full_authority=newly_full,
        relock=relock,
        reramp=reramp,
        overlay_active=overlay_active,
    )


__all__ = [
    "ACTION_DIM",
    "ALLOWED_RUN_NUM_ENVS",
    "ARM_ACTION_DIM",
    "ASSIGNMENT_CONTRACT",
    "ASSIGNMENT_SALT",
    "FORCE_SATURATION_N",
    "FORCE_STRENGTH_START",
    "FORCE_STRENGTH_STOP",
    "GRASP_QUALITY_MIN",
    "HAND_ACTION_DIM",
    "HOLD_QUALITY_INDEX",
    "HOLD_QUALITY_MIN",
    "OBSERVATION_DIM",
    "PUBLIC_FORCE_COUNTER_CONTRACT",
    "PUBLIC_FORCE_COUNTER_FEATURES",
    "PUBLIC_FORCE_STRENGTH_MAX",
    "PUBLIC_LATCH_INDEX",
    "PublicArmRampState",
    "PublicArmRampStep",
    "PublicStableState",
    "RAMP_DENOMINATOR",
    "SAFE_FORCE_LIMIT_N",
    "SAFE_FORCE_STRENGTH_MAX",
    "STRICT_WRAP_QUALITY_INDEX",
    "SUPERVISOR_CONTRACT",
    "VERIFY_STEPS",
    "apply_public_arm_ramp",
    "assignment_mask_sha256",
    "assignment_rank",
    "exact_balanced_treatment_mask",
    "initial_public_arm_ramp_state",
    "public_stable_state",
    "reset_public_arm_ramp_state",
    "validate_public_arm_ramp_state",
]
