#!/usr/bin/env python3
"""Pure public-state supervisor for Candidate40's verified arm handoff.

The common Candidate39 controller is evaluated before this module.  This
module may replace only its seven arm coordinates, and only for assigned
treatment rows after the first option-active, latch-visible decision.  It has
no simulator dependency and deliberately cannot consume transition truth,
clearance, or raw contact force.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch


OBSERVATION_DIM = 115
ACTION_DIM = 21
ARM_ACTION_DIM = 7
HAND_ACTION_DIM = 14

FORCE_STRENGTH_START = 97
FORCE_STRENGTH_STOP = 102
STRICT_WRAP_QUALITY_INDEX = 103
HOLD_QUALITY_INDEX = 104
PUBLIC_LATCH_INDEX = 106

VERIFY_STEPS = 15
VERIFIER_CONTRACT = "public_verified_live_arm_handoff_v1"
GRASP_QUALITY_MIN = 0.35
HOLD_QUALITY_MIN = 0.50
FORCE_SATURATION_N = 5.0
SAFE_FORCE_LIMIT_N = 30.0
# Sealed float32 value of tanh(30 / 5); do not recompute in rollout code.
PUBLIC_FORCE_STRENGTH_MAX = 0.9999877214431763
# Backward-readable alias; PUBLIC_FORCE_STRENGTH_MAX is the canonical name.
SAFE_FORCE_STRENGTH_MAX = PUBLIC_FORCE_STRENGTH_MAX

PUBLIC_FORCE_COUNTER_CONTRACT = "pick_tool_public_gate_state_v1"
PUBLIC_FORCE_COUNTER_FEATURES = (
    "hard_force_count_progress",
    "overforce_count_progress",
)

ASSIGNMENT_CONTRACT = (
    "candidate40_verified_arm_sha256_rank_exact_balanced_complement_v1"
)
ASSIGNMENT_SALT = (
    "pick_tool_candidate40_verified_arm_handoff_development_20260722_v1"
)


@dataclass(frozen=True)
class VerifiedArmState:
    """Persistent per-slot state, reset at each authoritative episode reset."""

    activated: torch.Tensor
    stable_count: torch.Tensor
    arm_enabled_previous: torch.Tensor
    ever_enabled: torch.Tensor


@dataclass(frozen=True)
class PublicStableState:
    """Public quantities used by the verifier on one pre-action observation."""

    stable: torch.Tensor
    latch: torch.Tensor
    grasp_quality: torch.Tensor
    hold_quality: torch.Tensor
    max_force_strength: torch.Tensor
    force_counter_features: torch.Tensor


@dataclass(frozen=True)
class VerifiedArmStep:
    """Candidate action, next state, and complete auditable row telemetry."""

    action: torch.Tensor
    next_state: VerifiedArmState
    eligible_now: torch.Tensor
    first_eligible: torch.Tensor
    activated_before: torch.Tensor
    activated_this_action: torch.Tensor
    stable_current: torch.Tensor
    public_grasp_quality: torch.Tensor
    public_hold_quality: torch.Tensor
    public_max_force_strength: torch.Tensor
    force_counter_features: torch.Tensor
    stable_count_before: torch.Tensor
    stable_count_after: torch.Tensor
    arm_enabled_this_action: torch.Tensor
    verification_complete: torch.Tensor
    newly_enabled: torch.Tensor
    relock: torch.Tensor
    overlay_active: torch.Tensor


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ranked_slots(*, seed: int, num_envs: int) -> list[int]:
    if not _is_int(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not _is_int(num_envs) or num_envs < 2 or num_envs % 2:
        raise ValueError("num_envs must be an even integer of at least two")
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
    """Return the sealed treatment assignment; replicate B complements A."""

    if replicate not in {"a", "b"}:
        raise ValueError("replicate must be 'a' or 'b'")
    ranked = _ranked_slots(seed=seed, num_envs=num_envs)
    assignment_a = torch.zeros(num_envs, dtype=torch.bool)
    assignment_a[torch.tensor(ranked[: num_envs // 2], dtype=torch.long)] = True
    return assignment_a if replicate == "a" else ~assignment_a


def assignment_rank(*, seed: int, num_envs: int) -> torch.Tensor:
    """Return each stable environment slot's zero-based assignment rank."""

    ranked = _ranked_slots(seed=seed, num_envs=num_envs)
    result = torch.empty(num_envs, dtype=torch.long)
    result[torch.tensor(ranked, dtype=torch.long)] = torch.arange(
        num_envs, dtype=torch.long
    )
    return result


def assignment_mask_sha256(mask: torch.Tensor) -> str:
    """Hash one CPU assignment mask under the sealed Candidate40 salt."""

    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 1
        or mask.dtype != torch.bool
        or mask.device.type != "cpu"
    ):
        raise ValueError("assignment receipt requires a rank-one CPU bool tensor")
    encoded = bytes(int(value) for value in mask.tolist())
    return hashlib.sha256(
        ASSIGNMENT_SALT.encode("utf-8") + b"\0mask-v1\0" + encoded
    ).hexdigest()


def initial_verified_arm_state(
    num_envs: int, *, device: torch.device | str | None = None
) -> VerifiedArmState:
    """Create a reset state for ``num_envs`` stable slots."""

    if not _is_int(num_envs) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    target = torch.device("cpu") if device is None else torch.device(device)
    return VerifiedArmState(
        activated=torch.zeros(num_envs, dtype=torch.bool, device=target),
        stable_count=torch.zeros(num_envs, dtype=torch.long, device=target),
        arm_enabled_previous=torch.zeros(
            num_envs, dtype=torch.bool, device=target
        ),
        ever_enabled=torch.zeros(num_envs, dtype=torch.bool, device=target),
    )


def _require_bool_vector(
    name: str, value: torch.Tensor, *, rows: int, device: torch.device
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (rows,)
        or value.dtype != torch.bool
        or value.device != device
    ):
        raise ValueError(
            f"{name} must be a [{rows}] bool tensor on {device}"
        )
    return value


def validate_verified_arm_state(
    state: VerifiedArmState, *, rows: int, device: torch.device
) -> None:
    """Fail closed on malformed or logically impossible supervisor state."""

    if not isinstance(state, VerifiedArmState):
        raise TypeError("state must be VerifiedArmState")
    for name in ("activated", "arm_enabled_previous", "ever_enabled"):
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
        raise ValueError(
            f"state.stable_count must be a [{rows}] long tensor on {device}"
        )
    if bool(((count < 0) | (count > VERIFY_STEPS)).any()):
        raise ValueError("state.stable_count escaped [0, 15]")
    if bool((state.arm_enabled_previous & (~state.activated)).any()):
        raise ValueError("an unactivated slot cannot have enabled arm authority")
    if bool(((~state.activated) & (count != 0)).any()):
        raise ValueError("an unactivated slot must have a zero stable clock")
    if bool((state.arm_enabled_previous & (count != VERIFY_STEPS)).any()):
        raise ValueError("enabled arm authority requires a saturated stable clock")
    if bool((state.arm_enabled_previous & (~state.ever_enabled)).any()):
        raise ValueError("currently enabled arm must also be marked ever-enabled")
    if bool((state.ever_enabled & (~state.activated)).any()):
        raise ValueError("an unactivated slot cannot have prior arm enablement")


def reset_verified_arm_state(
    state: VerifiedArmState, reset_mask: torch.Tensor
) -> VerifiedArmState:
    """Clear all controller memory on authoritative episode-reset rows."""

    if not isinstance(state, VerifiedArmState):
        raise TypeError("state must be VerifiedArmState")
    rows = int(state.activated.numel())
    device = state.activated.device
    validate_verified_arm_state(state, rows=rows, device=device)
    reset = _require_bool_vector(
        "reset_mask", reset_mask, rows=rows, device=device
    )
    zero_bool = torch.zeros_like(state.activated)
    zero_long = torch.zeros_like(state.stable_count)
    return VerifiedArmState(
        activated=torch.where(reset, zero_bool, state.activated),
        stable_count=torch.where(reset, zero_long, state.stable_count),
        arm_enabled_previous=torch.where(
            reset, zero_bool, state.arm_enabled_previous
        ),
        ever_enabled=torch.where(reset, zero_bool, state.ever_enabled),
    )


def _public_force_counter_tensor(
    value: Mapping[str, torch.Tensor] | torch.Tensor,
    *,
    rows: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, Mapping):
        if set(value) != set(PUBLIC_FORCE_COUNTER_FEATURES):
            raise ValueError("public force-counter sidecar schema changed")
        columns = []
        for name in PUBLIC_FORCE_COUNTER_FEATURES:
            column = value[name]
            if (
                not isinstance(column, torch.Tensor)
                or column.shape != (rows,)
                or column.dtype != torch.float32
                or column.device != device
            ):
                raise ValueError(
                    f"public force counter {name} must be a [{rows}] float32 tensor on {device}"
                )
            columns.append(column)
        features = torch.stack(columns, dim=-1)
    elif isinstance(value, torch.Tensor):
        features = value
        if (
            features.shape != (rows, len(PUBLIC_FORCE_COUNTER_FEATURES))
            or features.dtype != torch.float32
            or features.device != device
        ):
            raise ValueError(
                f"public force counters must be [{rows}, 2] float32 on {device}"
            )
    else:
        raise TypeError("public force counters must be a mapping or tensor")
    if not bool(torch.isfinite(features).all()) or bool(
        ((features < 0.0) | (features > 1.0)).any()
    ):
        raise ValueError("public force counters must be finite and in [0, 1]")
    return features


def public_stable_state(
    observation: torch.Tensor,
    public_force_counters: Mapping[str, torch.Tensor] | torch.Tensor,
    option_active: torch.Tensor,
    episode_active: torch.Tensor,
) -> PublicStableState:
    """Evaluate the sealed stable predicate from public pre-action state only."""

    if (
        not isinstance(observation, torch.Tensor)
        or observation.ndim != 2
        or observation.shape[1] != OBSERVATION_DIM
        or observation.dtype != torch.float32
    ):
        raise ValueError(
            f"observation must be [batch, {OBSERVATION_DIM}] float32"
        )
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("observation contains NaN or infinity")
    rows = int(observation.shape[0])
    device = observation.device
    option = _require_bool_vector(
        "option_active", option_active, rows=rows, device=device
    )
    active = _require_bool_vector(
        "episode_active", episode_active, rows=rows, device=device
    )
    counters = _public_force_counter_tensor(
        public_force_counters, rows=rows, device=device
    )
    latch_float = observation[:, PUBLIC_LATCH_INDEX]
    if bool(((latch_float != 0.0) & (latch_float != 1.0)).any()):
        raise ValueError("public latch must be exactly binary")
    latch = latch_float == 1.0
    hold = observation[:, HOLD_QUALITY_INDEX]
    wrap = observation[:, STRICT_WRAP_QUALITY_INDEX]
    force = observation[:, FORCE_STRENGTH_START:FORCE_STRENGTH_STOP]
    for name, value in (("strict wrap quality", wrap), ("hold quality", hold)):
        if bool(((value < 0.0) | (value > 1.0)).any()):
            raise ValueError(f"public {name} escaped [0, 1]")
    if bool(((force < 0.0) | (force > 1.0)).any()):
        raise ValueError("public force strength escaped [0, 1]")
    grasp_quality = torch.minimum(wrap, hold)
    max_force_strength = force.max(dim=-1).values
    stable = (
        active
        & option
        & latch
        & (grasp_quality >= GRASP_QUALITY_MIN)
        & (hold >= HOLD_QUALITY_MIN)
        & (max_force_strength <= PUBLIC_FORCE_STRENGTH_MAX)
        & (counters[:, 0] == 0.0)
        & (counters[:, 1] == 0.0)
    )
    return PublicStableState(
        stable=stable,
        latch=latch,
        grasp_quality=grasp_quality,
        hold_quality=hold,
        max_force_strength=max_force_strength,
        force_counter_features=counters,
    )


def apply_verified_arm_handoff(
    baseline_action: torch.Tensor,
    observation: torch.Tensor,
    public_force_counters: Mapping[str, torch.Tensor] | torch.Tensor,
    option_active: torch.Tensor,
    episode_active: torch.Tensor,
    treatment: torch.Tensor,
    state: VerifiedArmState,
    *,
    reset_mask: torch.Tensor,
) -> VerifiedArmStep:
    """Overlay Candidate40 arm authority on an exact Candidate39 action.

    The first fifteen stable eligible rows keep treatment arm commands at
    exact zero.  A still-stable sixteenth row may use the common baseline arm.
    Any current stability failure relocks on that same action and resets the
    entire verification clock.  Control and every hand coordinate are copied
    bit-exactly from ``baseline_action``.
    """

    if (
        not isinstance(baseline_action, torch.Tensor)
        or baseline_action.ndim != 2
        or baseline_action.shape[1] != ACTION_DIM
        or baseline_action.dtype != torch.float32
    ):
        raise ValueError(f"baseline_action must be [batch, {ACTION_DIM}] float32")
    if baseline_action.device != observation.device:
        raise ValueError("baseline action and observation must share a device")
    if not bool(torch.isfinite(baseline_action).all()) or bool(
        (baseline_action.abs() > 1.0).any()
    ):
        raise ValueError("baseline action must be finite and bounded by one")
    rows = int(baseline_action.shape[0])
    device = baseline_action.device
    if observation.shape != (rows, OBSERVATION_DIM):
        raise ValueError("baseline action and observation batch shapes disagree")
    active = _require_bool_vector(
        "episode_active", episode_active, rows=rows, device=device
    )
    option = _require_bool_vector(
        "option_active", option_active, rows=rows, device=device
    )
    assigned = _require_bool_vector(
        "treatment", treatment, rows=rows, device=device
    )
    validate_verified_arm_state(state, rows=rows, device=device)
    # Requiring this mask on every call makes the authoritative auto-reset
    # boundary explicit at the integration site.  A caller cannot silently
    # carry arm authority into the next episode by omitting reset handling.
    working = reset_verified_arm_state(state, reset_mask)

    public = public_stable_state(
        observation, public_force_counters, option, active
    )
    eligible_now = active & option & public.latch
    first_eligible = eligible_now & (~working.activated)
    activated = working.activated | first_eligible
    count_before = working.stable_count
    arm_enabled = (
        active
        & activated
        & (count_before == VERIFY_STEPS)
        & public.stable
    )
    # Activation is sticky, but option_active explicitly prevents SEARCH edits.
    overlay_active = active & assigned & activated & option
    action = baseline_action.clone()
    treatment_arm = torch.where(
        arm_enabled.unsqueeze(-1),
        baseline_action[:, :ARM_ACTION_DIM],
        torch.zeros_like(baseline_action[:, :ARM_ACTION_DIM]),
    )
    action[:, :ARM_ACTION_DIM] = torch.where(
        overlay_active.unsqueeze(-1),
        treatment_arm,
        baseline_action[:, :ARM_ACTION_DIM],
    )
    if not torch.equal(
        action[:, ARM_ACTION_DIM:], baseline_action[:, ARM_ACTION_DIM:]
    ):
        raise RuntimeError("Candidate40 changed the common hand14 action")
    control = active & (~assigned)
    if bool(control.any()) and not torch.equal(
        action[control], baseline_action[control]
    ):
        raise RuntimeError("Candidate40 changed a control action")
    search = active & (~option)
    if bool(search.any()) and not torch.equal(
        action[search], baseline_action[search]
    ):
        raise RuntimeError("Candidate40 changed a SEARCH action")

    candidate_count = torch.where(
        public.stable,
        torch.clamp(count_before + 1, max=VERIFY_STEPS),
        torch.zeros_like(count_before),
    )
    count_after = torch.where(active, candidate_count, count_before)
    verification_complete = (
        active
        & activated
        & public.stable
        & (count_before == VERIFY_STEPS - 1)
    )
    newly_enabled = arm_enabled & (~working.arm_enabled_previous)
    relock = (
        active
        & activated
        & working.arm_enabled_previous
        & (~arm_enabled)
    )
    enabled_previous_after = torch.where(
        active, arm_enabled, working.arm_enabled_previous
    )
    ever_enabled_after = working.ever_enabled | newly_enabled
    next_state = VerifiedArmState(
        activated=activated,
        stable_count=count_after,
        arm_enabled_previous=enabled_previous_after,
        ever_enabled=ever_enabled_after,
    )
    validate_verified_arm_state(next_state, rows=rows, device=device)
    return VerifiedArmStep(
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
        arm_enabled_this_action=arm_enabled,
        verification_complete=verification_complete,
        newly_enabled=newly_enabled,
        relock=relock,
        overlay_active=overlay_active,
    )


__all__ = [
    "ACTION_DIM",
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
    "PublicStableState",
    "SAFE_FORCE_LIMIT_N",
    "SAFE_FORCE_STRENGTH_MAX",
    "STRICT_WRAP_QUALITY_INDEX",
    "VERIFIER_CONTRACT",
    "VERIFY_STEPS",
    "VerifiedArmState",
    "VerifiedArmStep",
    "apply_verified_arm_handoff",
    "assignment_mask_sha256",
    "assignment_rank",
    "exact_balanced_treatment_mask",
    "initial_verified_arm_state",
    "public_stable_state",
    "reset_verified_arm_state",
    "validate_verified_arm_state",
]
