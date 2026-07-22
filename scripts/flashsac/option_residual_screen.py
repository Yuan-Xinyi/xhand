#!/usr/bin/env python3
"""Pure-PyTorch contract for Candidate 39 coherent CLOSE residuals.

The contract deliberately keeps randomization outside the actor.  For a fixed
``(seed, env_slot)`` each environment receives one episode-coherent latent.
Slots are paired by a SHA256 ranking; the two members of every pair receive
exact antithetic latents.  Complementary screen replicates only swap treatment
and control--they never redraw the latent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch


ACTION_DIM = 21
ARM_ACTION_DIM = 7
HAND_ACTION_DIM = 14
TOKEN_ACTION_DIM = 9
DISTAL_ACTION_DIM = 5

DESIGN_CONTRACT = "candidate39_sha256_rank_paired_antithetic_v1"
DESIGN_SALT = "pick_tool_candidate39_option_residual_20260722_v1"
DEFAULT_ATANH_EPSILON = 1.0e-6


@dataclass(frozen=True)
class OptionResidualDesign:
    """Auditable randomized design for one complementary replicate."""

    treatment: torch.Tensor
    raw_z: torch.Tensor
    effective_z: torch.Tensor
    pair_slot: torch.Tensor
    antithetic_sign: torch.Tensor


@dataclass(frozen=True)
class OptionResidualResult:
    """Action plus every latent quantity needed to audit the overlay."""

    action: torch.Tensor
    raw_z: torch.Tensor
    effective_z: torch.Tensor
    residual_z: torch.Tensor
    unclipped_pre_tanh_residual: torch.Tensor
    pre_tanh_residual: torch.Tensor
    eligible: torch.Tensor


@dataclass(frozen=True)
class OptionResidualWindow:
    """Sticky first-latch intervention clock for one vector step."""

    close_age: torch.Tensor
    active: torch.Tensor


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_seed_and_size(seed: int, num_envs: int) -> None:
    if not _is_int(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not _is_int(num_envs) or num_envs < 2 or num_envs % 2:
        raise ValueError("num_envs must be an even integer of at least two")


def _ranked_slots(*, seed: int, num_envs: int) -> list[int]:
    _validate_seed_and_size(seed, num_envs)
    return sorted(
        range(num_envs),
        key=lambda env_slot: (
            hashlib.sha256(
                f"{DESIGN_SALT}\0rank\0{seed}\0{env_slot}".encode("utf-8")
            ).digest(),
            env_slot,
        ),
    )


def exact_balanced_treatment_mask(
    *, seed: int, num_envs: int, replicate: str
) -> torch.Tensor:
    """Return the fixed treatment mask; replicate ``b`` exactly complements ``a``."""

    if replicate not in {"a", "b"}:
        raise ValueError("replicate must be 'a' or 'b'")
    ranked = _ranked_slots(seed=seed, num_envs=num_envs)
    assignment_a = torch.zeros(num_envs, dtype=torch.bool)
    assignment_a[torch.tensor(ranked[: num_envs // 2], dtype=torch.long)] = True
    return assignment_a if replicate == "a" else ~assignment_a


def _open_unit_interval(digest_bytes: bytes) -> float:
    """Map eight hash bytes to a double strictly inside ``(0, 1)``."""

    # Retaining 53 bits makes the integer exactly representable as a double.
    integer = int.from_bytes(digest_bytes, byteorder="big", signed=False) >> 11
    return (integer + 1.0) / ((1 << 53) + 1.0)


def _pair_gaussian(*, seed: int, plus_slot: int, minus_slot: int) -> list[float]:
    """Generate fourteen deterministic standard-normal coordinates."""

    values: list[float] = []
    for gaussian_pair in range(HAND_ACTION_DIM // 2):
        digest = hashlib.sha256(
            (
                f"{DESIGN_SALT}\0z\0{seed}\0{plus_slot}\0{minus_slot}"
                f"\0{gaussian_pair}"
            ).encode("utf-8")
        ).digest()
        uniform_radius = _open_unit_interval(digest[:8])
        uniform_angle = _open_unit_interval(digest[8:16])
        radius = math.sqrt(-2.0 * math.log(uniform_radius))
        angle = 2.0 * math.pi * uniform_angle
        values.extend((radius * math.cos(angle), radius * math.sin(angle)))
    if len(values) != HAND_ACTION_DIM or not all(math.isfinite(x) for x in values):
        raise RuntimeError("SHA256 Gaussian construction produced an invalid latent")
    return values


def episode_coherent_raw_z(
    *,
    seed: int,
    num_envs: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(raw_z, pair_slot, sign)`` for every stable environment slot.

    The first half of the SHA256 ranking is paired positionally with the second
    half.  The first member receives ``+z`` and the second receives ``-z``.
    Thus each pair is exactly antithetic, and the result is independent of A/B
    replicate labels and simulator time.
    """

    ranked = _ranked_slots(seed=seed, num_envs=num_envs)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating torch dtype")
    target_device = torch.device("cpu") if device is None else torch.device(device)
    raw_z = torch.zeros(
        (num_envs, HAND_ACTION_DIM), dtype=dtype, device=target_device
    )
    pair_slot = torch.empty(num_envs, dtype=torch.long, device=target_device)
    sign = torch.empty(num_envs, dtype=torch.int8, device=target_device)
    half = num_envs // 2
    for pair_index in range(half):
        plus_slot = ranked[pair_index]
        minus_slot = ranked[pair_index + half]
        base = torch.tensor(
            _pair_gaussian(
                seed=seed, plus_slot=plus_slot, minus_slot=minus_slot
            ),
            dtype=dtype,
            device=target_device,
        )
        raw_z[plus_slot] = base
        raw_z[minus_slot] = -base
        pair_slot[plus_slot] = minus_slot
        pair_slot[minus_slot] = plus_slot
        sign[plus_slot] = 1
        sign[minus_slot] = -1
    if not bool(torch.isfinite(raw_z).all()):
        raise FloatingPointError("raw_z contains NaN or infinity")
    return raw_z, pair_slot, sign


def build_option_design(
    *,
    seed: int,
    num_envs: int,
    replicate: str,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> OptionResidualDesign:
    """Build treatment, replicate-invariant latents and zero control latents."""

    treatment_cpu = exact_balanced_treatment_mask(
        seed=seed, num_envs=num_envs, replicate=replicate
    )
    raw_z, pair_slot, sign = episode_coherent_raw_z(
        seed=seed, num_envs=num_envs, dtype=dtype, device=device
    )
    treatment = treatment_cpu.to(device=raw_z.device)
    effective_z = torch.where(treatment.unsqueeze(-1), raw_z, torch.zeros_like(raw_z))
    return OptionResidualDesign(
        treatment=treatment,
        raw_z=raw_z,
        effective_z=effective_z,
        pair_slot=pair_slot,
        antithetic_sign=sign,
    )


def grouped_pre_tanh_scale(
    *,
    token_scale: float,
    distal_scale: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the fixed hand14 scale vector: token9 followed by distal5."""

    for name, value in (("token_scale", token_scale), ("distal_scale", distal_scale)):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating torch dtype")
    return torch.tensor(
        [float(token_scale)] * TOKEN_ACTION_DIM
        + [float(distal_scale)] * DISTAL_ACTION_DIM,
        dtype=dtype,
        device=device,
    )


def option_residual_window(
    *,
    episode_active: torch.Tensor,
    option_active: torch.Tensor,
    public_latch: torch.Tensor,
    ever_latched: torch.Tensor,
    episode_step: torch.Tensor,
    trigger_step: torch.Tensor,
    window_steps: int,
) -> OptionResidualWindow:
    """Return the trigger-relative window, retired forever after first latch.

    The trigger action has age zero.  ``ever_latched`` is intentionally
    separate from the current public latch: once it becomes true, a later
    latch release cannot reactivate the intervention.
    """

    if (
        not isinstance(window_steps, int)
        or isinstance(window_steps, bool)
        or window_steps < 1
    ):
        raise ValueError("window_steps must be a positive integer")
    if not isinstance(episode_active, torch.Tensor) or episode_active.ndim != 1:
        raise ValueError("episode_active must be a rank-one tensor")
    batch = episode_active.shape[0]
    device = episode_active.device
    for name, value in (
        ("episode_active", episode_active),
        ("option_active", option_active),
        ("public_latch", public_latch),
        ("ever_latched", ever_latched),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (batch,)
            or value.dtype != torch.bool
            or value.device != device
        ):
            raise ValueError(f"{name} must be a co-located bool[{batch}] tensor")
    for name, value in (("episode_step", episode_step), ("trigger_step", trigger_step)):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (batch,)
            or value.dtype != torch.long
            or value.device != device
        ):
            raise ValueError(f"{name} must be a co-located int64[{batch}] tensor")
    if bool((episode_step < 0).any()):
        raise ValueError("episode_step must be non-negative")
    if bool((option_active & (trigger_step < 0)).any()):
        raise ValueError("an active option must have a non-negative trigger_step")
    close_age = episode_step - trigger_step
    active = (
        episode_active
        & option_active
        & (~public_latch)
        & (~ever_latched)
        & (close_age >= 0)
        & (close_age < window_steps)
    )
    return OptionResidualWindow(close_age=close_age, active=active)


def _validate_action_inputs(
    *,
    baseline_action: torch.Tensor,
    option_active: torch.Tensor,
    public_latch: torch.Tensor,
    treatment: torch.Tensor,
    raw_z: torch.Tensor,
    pre_tanh_scale: torch.Tensor,
    raw_z_abs_cap: float | None,
    pre_tanh_abs_cap: torch.Tensor | None,
    pre_tanh_l2_cap: float | None,
    atanh_epsilon: float,
) -> None:
    if (
        not isinstance(baseline_action, torch.Tensor)
        or baseline_action.ndim != 2
        or baseline_action.shape[1] != ACTION_DIM
        or not baseline_action.dtype.is_floating_point
    ):
        raise ValueError(f"baseline_action must be floating [N,{ACTION_DIM}]")
    batch = baseline_action.shape[0]
    for name, value in (
        ("option_active", option_active),
        ("public_latch", public_latch),
        ("treatment", treatment),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (batch,)
            or value.dtype != torch.bool
            or value.device != baseline_action.device
        ):
            raise ValueError(f"{name} must be a co-located bool[{batch}] tensor")
    if (
        not isinstance(raw_z, torch.Tensor)
        or raw_z.shape != (batch, HAND_ACTION_DIM)
        or raw_z.dtype != baseline_action.dtype
        or raw_z.device != baseline_action.device
    ):
        raise ValueError(
            f"raw_z must match baseline dtype/device and have shape [{batch},{HAND_ACTION_DIM}]"
        )
    if (
        not isinstance(pre_tanh_scale, torch.Tensor)
        or pre_tanh_scale.shape
        not in {(HAND_ACTION_DIM,), (batch, HAND_ACTION_DIM)}
        or pre_tanh_scale.dtype != baseline_action.dtype
        or pre_tanh_scale.device != baseline_action.device
    ):
        raise ValueError(
            "pre_tanh_scale must match baseline dtype/device and have shape "
            f"[{HAND_ACTION_DIM}] or [{batch},{HAND_ACTION_DIM}]"
        )
    if pre_tanh_abs_cap is not None and (
        not isinstance(pre_tanh_abs_cap, torch.Tensor)
        or pre_tanh_abs_cap.shape
        not in {(HAND_ACTION_DIM,), (batch, HAND_ACTION_DIM)}
        or pre_tanh_abs_cap.dtype != baseline_action.dtype
        or pre_tanh_abs_cap.device != baseline_action.device
    ):
        raise ValueError(
            "pre_tanh_abs_cap must be None or match baseline dtype/device and "
            f"have shape [{HAND_ACTION_DIM}] or [{batch},{HAND_ACTION_DIM}]"
        )
    if not bool(
        torch.isfinite(baseline_action).all()
        and torch.isfinite(raw_z).all()
        and torch.isfinite(pre_tanh_scale).all()
        and (
            pre_tanh_abs_cap is None
            or torch.isfinite(pre_tanh_abs_cap).all()
        )
    ):
        raise FloatingPointError("action inputs contain NaN or infinity")
    if bool((baseline_action.abs() > 1.0).any()):
        raise ValueError("baseline_action must lie in the closed interval [-1, 1]")
    if bool((pre_tanh_scale < 0.0).any()):
        raise ValueError("pre_tanh_scale must be non-negative")
    if pre_tanh_abs_cap is not None and bool((pre_tanh_abs_cap < 0.0).any()):
        raise ValueError("pre_tanh_abs_cap must be non-negative")
    if raw_z_abs_cap is not None and (
        not isinstance(raw_z_abs_cap, (int, float))
        or isinstance(raw_z_abs_cap, bool)
        or not math.isfinite(float(raw_z_abs_cap))
        or float(raw_z_abs_cap) < 0.0
    ):
        raise ValueError("raw_z_abs_cap must be None or finite and non-negative")
    if pre_tanh_l2_cap is not None and (
        not isinstance(pre_tanh_l2_cap, (int, float))
        or isinstance(pre_tanh_l2_cap, bool)
        or not math.isfinite(float(pre_tanh_l2_cap))
        or float(pre_tanh_l2_cap) < 0.0
    ):
        raise ValueError("pre_tanh_l2_cap must be None or finite and non-negative")
    if (
        not isinstance(atanh_epsilon, (int, float))
        or isinstance(atanh_epsilon, bool)
        or not math.isfinite(float(atanh_epsilon))
        or not 0.0 < float(atanh_epsilon) < 1.0
    ):
        raise ValueError("atanh_epsilon must be finite and strictly between zero and one")


def apply_option_residual(
    *,
    baseline_action: torch.Tensor,
    option_active: torch.Tensor,
    public_latch: torch.Tensor,
    treatment: torch.Tensor,
    raw_z: torch.Tensor,
    pre_tanh_scale: torch.Tensor,
    raw_z_abs_cap: float | None = None,
    pre_tanh_abs_cap: torch.Tensor | None = None,
    pre_tanh_l2_cap: float | None = None,
    atanh_epsilon: float = DEFAULT_ATANH_EPSILON,
) -> OptionResidualResult:
    """Apply a coherent hand-only residual in baseline pre-tanh space.

    Only rows satisfying ``option_active & ~public_latch & treatment`` are
    eligible.  Arm7 is copied from the baseline for every row.  Latched,
    inactive and control rows select the original baseline tensor values, so
    their complete action is bit-exact.  Zero-scaled hand coordinates are also
    selected from the baseline exactly, including at tanh saturation.  Scale
    and residual-cap tensors may be fixed ``[14]`` vectors or explicit
    state-dependent ``[N,14]`` matrices.  Clipping is symmetric; both the
    fully unclipped and actually applied residuals are returned for audit.
    """

    _validate_action_inputs(
        baseline_action=baseline_action,
        option_active=option_active,
        public_latch=public_latch,
        treatment=treatment,
        raw_z=raw_z,
        pre_tanh_scale=pre_tanh_scale,
        raw_z_abs_cap=raw_z_abs_cap,
        pre_tanh_abs_cap=pre_tanh_abs_cap,
        pre_tanh_l2_cap=pre_tanh_l2_cap,
        atanh_epsilon=atanh_epsilon,
    )
    effective_z = torch.where(
        treatment.unsqueeze(-1), raw_z, torch.zeros_like(raw_z)
    )
    eligible = option_active & ~public_latch & treatment
    expanded_scale = (
        pre_tanh_scale.unsqueeze(0)
        if pre_tanh_scale.ndim == 1
        else pre_tanh_scale
    )
    unclipped_residual = torch.where(
        eligible.unsqueeze(-1),
        effective_z * expanded_scale,
        torch.zeros_like(raw_z),
    )
    residual_z = (
        effective_z
        if raw_z_abs_cap is None
        else effective_z.clamp(
            min=-float(raw_z_abs_cap), max=float(raw_z_abs_cap)
        )
    )
    residual = residual_z * expanded_scale
    if pre_tanh_abs_cap is not None:
        expanded_cap = (
            pre_tanh_abs_cap.unsqueeze(0)
            if pre_tanh_abs_cap.ndim == 1
            else pre_tanh_abs_cap
        )
        residual = torch.maximum(
            torch.minimum(residual, expanded_cap), -expanded_cap
        )
    if pre_tanh_l2_cap is not None:
        l2_cap = float(pre_tanh_l2_cap)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
        l2_scale = torch.where(
            residual_norm > l2_cap,
            l2_cap / residual_norm.clamp_min(torch.finfo(residual.dtype).tiny),
            torch.ones_like(residual_norm),
        )
        residual = residual * l2_scale
    residual = torch.where(
        eligible.unsqueeze(-1), residual, torch.zeros_like(residual)
    )

    # The dtype-aware margin prevents +/-1 from reaching atanh even for half
    # precision, where a nominal 1e-6 margin rounds back to one.
    dtype_margin = 4.0 * float(torch.finfo(baseline_action.dtype).eps)
    margin = max(float(atanh_epsilon), dtype_margin)
    if not margin < 1.0:
        raise ValueError("atanh clamp margin is not representable for this dtype")
    baseline_hand = baseline_action[:, ARM_ACTION_DIM:]
    clamped_hand = baseline_hand.clamp(min=-1.0 + margin, max=1.0 - margin)
    overlaid_hand = torch.tanh(torch.atanh(clamped_hand) + residual)
    # Do not let atanh(clamp(x)) perturb a coordinate whose residual is zero.
    selected_hand = torch.where(residual != 0.0, overlaid_hand, baseline_hand)
    candidate_action = torch.cat(
        (baseline_action[:, :ARM_ACTION_DIM], selected_hand), dim=-1
    )
    action = torch.where(eligible.unsqueeze(-1), candidate_action, baseline_action)
    if not bool(torch.isfinite(action).all()):
        raise FloatingPointError("residual overlay produced NaN or infinity")
    if bool((action.abs() > 1.0).any()):
        raise RuntimeError("residual overlay escaped tanh action bounds")
    return OptionResidualResult(
        action=action,
        raw_z=raw_z,
        effective_z=effective_z,
        residual_z=residual_z,
        unclipped_pre_tanh_residual=unclipped_residual,
        pre_tanh_residual=residual,
        eligible=eligible,
    )


__all__ = [
    "ACTION_DIM",
    "ARM_ACTION_DIM",
    "HAND_ACTION_DIM",
    "TOKEN_ACTION_DIM",
    "DISTAL_ACTION_DIM",
    "DESIGN_CONTRACT",
    "DESIGN_SALT",
    "OptionResidualDesign",
    "OptionResidualResult",
    "OptionResidualWindow",
    "exact_balanced_treatment_mask",
    "episode_coherent_raw_z",
    "build_option_design",
    "grouped_pre_tanh_scale",
    "option_residual_window",
    "apply_option_residual",
]
