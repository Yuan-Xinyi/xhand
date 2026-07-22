#!/usr/bin/env python3
"""Candidate42 assignment wrapper around Candidate41's sealed supervisor.

Candidate42 changes only campaign assignment and evidence publication.  Every
scientific state type and supervisor function is imported and aliased directly
from Candidate41 so the arm-authority state machine cannot drift.  The new
salt is used only to construct the preregistered exactly balanced 8- or
64-environment complementary assignments.
"""

from __future__ import annotations

import hashlib

import torch

import candidate41_public_arm_ramp as candidate41


# Scientific dimensions, thresholds, clocks, and contracts are immutable C41
# authorities.  Keep these as aliases rather than restating their values.
OBSERVATION_DIM = candidate41.OBSERVATION_DIM
ACTION_DIM = candidate41.ACTION_DIM
ARM_ACTION_DIM = candidate41.ARM_ACTION_DIM
HAND_ACTION_DIM = candidate41.HAND_ACTION_DIM

FORCE_STRENGTH_START = candidate41.FORCE_STRENGTH_START
FORCE_STRENGTH_STOP = candidate41.FORCE_STRENGTH_STOP
STRICT_WRAP_QUALITY_INDEX = candidate41.STRICT_WRAP_QUALITY_INDEX
HOLD_QUALITY_INDEX = candidate41.HOLD_QUALITY_INDEX
PUBLIC_LATCH_INDEX = candidate41.PUBLIC_LATCH_INDEX
GRASP_QUALITY_MIN = candidate41.GRASP_QUALITY_MIN
HOLD_QUALITY_MIN = candidate41.HOLD_QUALITY_MIN
PUBLIC_FORCE_STRENGTH_MAX = candidate41.PUBLIC_FORCE_STRENGTH_MAX
SAFE_FORCE_STRENGTH_MAX = candidate41.SAFE_FORCE_STRENGTH_MAX
SAFE_FORCE_LIMIT_N = candidate41.SAFE_FORCE_LIMIT_N
FORCE_SATURATION_N = candidate41.FORCE_SATURATION_N
PUBLIC_FORCE_COUNTER_CONTRACT = candidate41.PUBLIC_FORCE_COUNTER_CONTRACT
PUBLIC_FORCE_COUNTER_FEATURES = candidate41.PUBLIC_FORCE_COUNTER_FEATURES

VERIFY_STEPS = candidate41.VERIFY_STEPS
RAMP_DENOMINATOR = candidate41.RAMP_DENOMINATOR
SUPERVISOR_CONTRACT = candidate41.SUPERVISOR_CONTRACT

ASSIGNMENT_CONTRACT = (
    "candidate42_transactional_public_arm_ramp_sha256_rank_exact_balanced_complement_v1"
)
ASSIGNMENT_SALT = (
    "pick_tool_candidate42_public_arm_ramp_transaction_development_20260722_v1"
)
ALLOWED_RUN_NUM_ENVS = frozenset({8, 64})


# These identities are the scientific inheritance contract.  In particular,
# do not subclass the records and do not wrap any function.
PublicStableState = candidate41.PublicStableState
PublicArmRampState = candidate41.PublicArmRampState
PublicArmRampStep = candidate41.PublicArmRampStep
public_stable_state = candidate41.public_stable_state
initial_public_arm_ramp_state = candidate41.initial_public_arm_ramp_state
validate_public_arm_ramp_state = candidate41.validate_public_arm_ramp_state
reset_public_arm_ramp_state = candidate41.reset_public_arm_ramp_state
apply_public_arm_ramp = candidate41.apply_public_arm_ramp


def _ranked_slots(*, seed: int, num_envs: int) -> list[int]:
    if not candidate41._is_int(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not candidate41._is_int(num_envs) or num_envs not in ALLOWED_RUN_NUM_ENVS:
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
    """Return the sealed C42 assignment; replicate B is bitwise complement A."""

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
