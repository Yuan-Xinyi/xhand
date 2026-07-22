#!/usr/bin/env python3
"""Simulation-free identity and assignment tests for Candidate42."""

from __future__ import annotations

import hashlib
import random

import pytest
import torch

import candidate41_public_arm_ramp as c41
import candidate42_public_arm_ramp as c42


def _stable_observation(rows: int, generator: torch.Generator) -> torch.Tensor:
    observation = torch.rand(
        (rows, c42.OBSERVATION_DIM), generator=generator, dtype=torch.float32
    )
    observation[:, c42.PUBLIC_LATCH_INDEX] = 1.0
    observation[:, c42.STRICT_WRAP_QUALITY_INDEX] = 0.35 + 0.65 * torch.rand(
        rows, generator=generator
    )
    observation[:, c42.HOLD_QUALITY_INDEX] = 0.50 + 0.50 * torch.rand(
        rows, generator=generator
    )
    observation[:, c42.FORCE_STRENGTH_START : c42.FORCE_STRENGTH_STOP] = (
        0.5 * torch.rand(
            (rows, c42.FORCE_STRENGTH_STOP - c42.FORCE_STRENGTH_START),
            generator=generator,
        )
    )
    return observation


def _assert_state_equal(left: c42.PublicArmRampState, right: c41.PublicArmRampState) -> None:
    for name in (
        "activated",
        "stable_count",
        "ramp_scale_previous",
        "ever_positive_authority",
        "ever_full_authority",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name))


def test_scientific_types_and_functions_are_direct_candidate41_aliases() -> None:
    assert c42.PublicStableState is c41.PublicStableState
    assert c42.PublicArmRampState is c41.PublicArmRampState
    assert c42.PublicArmRampStep is c41.PublicArmRampStep
    for name in (
        "public_stable_state",
        "initial_public_arm_ramp_state",
        "validate_public_arm_ramp_state",
        "reset_public_arm_ramp_state",
        "apply_public_arm_ramp",
    ):
        assert getattr(c42, name) is getattr(c41, name)
    for name in (
        "SUPERVISOR_CONTRACT",
        "VERIFY_STEPS",
        "RAMP_DENOMINATOR",
        "GRASP_QUALITY_MIN",
        "HOLD_QUALITY_MIN",
        "PUBLIC_FORCE_STRENGTH_MAX",
        "PUBLIC_FORCE_COUNTER_CONTRACT",
        "PUBLIC_FORCE_COUNTER_FEATURES",
    ):
        assert getattr(c42, name) == getattr(c41, name)


def test_registered_assignments_are_balanced_deterministic_and_complementary() -> None:
    for seed, num_envs in ((346, 8), (347, 64), (348, 64)):
        a = c42.exact_balanced_treatment_mask(
            seed=seed, num_envs=num_envs, replicate="a"
        )
        b = c42.exact_balanced_treatment_mask(
            seed=seed, num_envs=num_envs, replicate="b"
        )
        rank = c42.assignment_rank(seed=seed, num_envs=num_envs)
        assert int(a.sum()) == num_envs // 2
        assert torch.equal(b, ~a)
        assert torch.equal(a, rank < num_envs // 2)
        assert torch.equal(torch.sort(rank).values, torch.arange(num_envs))
        assert torch.equal(
            a,
            c42.exact_balanced_treatment_mask(
                seed=seed, num_envs=num_envs, replicate="a"
            ),
        )
        expected_digest = hashlib.sha256(
            c42.ASSIGNMENT_SALT.encode("utf-8")
            + b"\0mask-v1\0"
            + bytes(int(value) for value in a.tolist())
        ).hexdigest()
        assert c42.assignment_mask_sha256(a) == expected_digest


def test_assignment_rejects_every_unregistered_shape_or_label() -> None:
    for count in (0, 2, 16, 63, 128, True):
        with pytest.raises(ValueError):
            c42.exact_balanced_treatment_mask(
                seed=346, num_envs=count, replicate="a"
            )
    for seed in (-1, True, 3.5):
        with pytest.raises(ValueError):
            c42.assignment_rank(seed=seed, num_envs=8)
    with pytest.raises(ValueError):
        c42.exact_balanced_treatment_mask(
            seed=346, num_envs=8, replicate="A"
        )
    with pytest.raises(ValueError):
        c42.assignment_mask_sha256(torch.zeros(16, dtype=torch.bool))


def test_randomized_scientific_outputs_are_bit_exact_to_candidate41() -> None:
    generator = torch.Generator().manual_seed(20260722)
    choices = random.Random(20260722)
    rows = 64
    state42 = c42.initial_public_arm_ramp_state(rows)
    state41 = c41.initial_public_arm_ramp_state(rows)
    treatment = c42.exact_balanced_treatment_mask(
        seed=347, num_envs=rows, replicate="a"
    )
    for _ in range(96):
        observation = _stable_observation(rows, generator)
        # Exercise hard re-locks, inactive rows, SEARCH rows, and reset-before
        # semantics without altering the public scientific function call.
        unstable = torch.rand(rows, generator=generator) < choices.random() * 0.4
        observation[unstable, c42.HOLD_QUALITY_INDEX] = 0.0
        option = torch.rand(rows, generator=generator) > 0.15
        active = torch.rand(rows, generator=generator) > 0.10
        reset = torch.rand(rows, generator=generator) < 0.04
        baseline = 2.0 * torch.rand(
            (rows, c42.ACTION_DIM), generator=generator
        ) - 1.0
        counters = torch.zeros((rows, 2), dtype=torch.float32)
        counters[torch.rand(rows, generator=generator) < 0.03, 0] = 1.0

        result42 = c42.apply_public_arm_ramp(
            baseline,
            observation,
            counters,
            option,
            active,
            treatment,
            state42,
            reset_mask=reset,
        )
        result41 = c41.apply_public_arm_ramp(
            baseline,
            observation,
            counters,
            option,
            active,
            treatment,
            state41,
            reset_mask=reset,
        )
        assert type(result42) is type(result41) is c41.PublicArmRampStep
        for name in result42.__dataclass_fields__:
            left, right = getattr(result42, name), getattr(result41, name)
            if isinstance(left, torch.Tensor):
                assert torch.equal(left, right), name
            else:
                _assert_state_equal(left, right)
        state42, state41 = result42.next_state, result41.next_state


def test_authority_sequence_and_same_action_hard_relock_are_inherited() -> None:
    rows = 2
    treatment = torch.tensor([True, False])
    baseline = torch.linspace(
        -0.9, 0.9, rows * c42.ACTION_DIM, dtype=torch.float32
    ).reshape(rows, c42.ACTION_DIM)
    observation = _stable_observation(rows, torch.Generator().manual_seed(7))
    counters = torch.zeros((rows, 2), dtype=torch.float32)
    state = c42.initial_public_arm_ramp_state(rows)
    expected = torch.arange(16, dtype=torch.float32) / torch.tensor(
        15.0, dtype=torch.float32
    )
    for index, scale in enumerate(expected):
        result = c42.apply_public_arm_ramp(
            baseline,
            observation,
            counters,
            torch.ones(rows, dtype=torch.bool),
            torch.ones(rows, dtype=torch.bool),
            treatment,
            state,
            reset_mask=torch.zeros(rows, dtype=torch.bool),
        )
        assert torch.equal(result.ramp_scale, scale.repeat(rows))
        state = result.next_state
    unstable = observation.clone()
    unstable[:, c42.HOLD_QUALITY_INDEX] = 0.0
    relocked = c42.apply_public_arm_ramp(
        baseline,
        unstable,
        counters,
        torch.ones(rows, dtype=torch.bool),
        torch.ones(rows, dtype=torch.bool),
        treatment,
        state,
        reset_mask=torch.zeros(rows, dtype=torch.bool),
    )
    assert float(relocked.authority_scale[0]) == 0.0
    assert torch.equal(
        relocked.action[0, : c42.ARM_ACTION_DIM],
        torch.zeros(c42.ARM_ACTION_DIM),
    )
    assert int(relocked.stable_count_after[0]) == 0
    assert bool(relocked.relock[0])
    assert torch.equal(relocked.action[1], baseline[1])
