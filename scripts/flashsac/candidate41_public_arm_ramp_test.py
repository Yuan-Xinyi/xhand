#!/usr/bin/env python3
"""Focused CPU tests for Candidate41's pure public arm ramp."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import candidate40_verified_arm_handoff as candidate40
import candidate41_public_arm_ramp as ramp


N = 8


def _observation(rows: int = N) -> torch.Tensor:
    obs = torch.zeros((rows, ramp.OBSERVATION_DIM), dtype=torch.float32)
    obs[:, ramp.FORCE_STRENGTH_START : ramp.FORCE_STRENGTH_STOP] = 0.8
    obs[:, ramp.STRICT_WRAP_QUALITY_INDEX] = 0.6
    obs[:, ramp.HOLD_QUALITY_INDEX] = 0.8
    obs[:, ramp.PUBLIC_LATCH_INDEX] = 1.0
    return obs


def _counters(rows: int = N) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(rows, dtype=torch.float32)
        for name in ramp.PUBLIC_FORCE_COUNTER_FEATURES
    }


def _baseline(rows: int = N) -> torch.Tensor:
    return torch.linspace(
        -0.9, 0.9, rows * ramp.ACTION_DIM, dtype=torch.float32
    ).reshape(rows, ramp.ACTION_DIM)


def _assert_state_equal(
    left: ramp.PublicArmRampState, right: ramp.PublicArmRampState
) -> None:
    for name in (
        "activated",
        "stable_count",
        "ramp_scale_previous",
        "ever_positive_authority",
        "ever_full_authority",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name))


def _step(
    state: ramp.PublicArmRampState,
    *,
    observation: torch.Tensor | None = None,
    counters: dict[str, torch.Tensor] | torch.Tensor | None = None,
    option_active: torch.Tensor | None = None,
    episode_active: torch.Tensor | None = None,
    treatment: torch.Tensor | None = None,
    baseline: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> ramp.PublicArmRampStep:
    rows = int(state.activated.numel())
    return ramp.apply_public_arm_ramp(
        _baseline(rows) if baseline is None else baseline,
        _observation(rows) if observation is None else observation,
        _counters(rows) if counters is None else counters,
        torch.ones(rows, dtype=torch.bool)
        if option_active is None
        else option_active,
        torch.ones(rows, dtype=torch.bool)
        if episode_active is None
        else episode_active,
        torch.ones(rows, dtype=torch.bool) if treatment is None else treatment,
        state,
        reset_mask=torch.zeros(rows, dtype=torch.bool)
        if reset_mask is None
        else reset_mask,
    )


def test_contract_reuses_candidate40_predicate_and_clock_constants() -> None:
    assert ramp.SUPERVISOR_CONTRACT == "public_linear_arm_authority_ramp_v1"
    assert ramp.public_stable_state is candidate40.public_stable_state
    assert ramp.PublicStableState is candidate40.PublicStableState
    assert ramp.VERIFY_STEPS == candidate40.VERIFY_STEPS == 15
    assert ramp.PUBLIC_FORCE_STRENGTH_MAX == candidate40.PUBLIC_FORCE_STRENGTH_MAX
    assert ramp.RAMP_DENOMINATOR == 15.0


def test_assignment_is_balanced_deterministic_and_complementary() -> None:
    for seed in (340, 341, 342):
        a = ramp.exact_balanced_treatment_mask(
            seed=seed, num_envs=64, replicate="a"
        )
        b = ramp.exact_balanced_treatment_mask(
            seed=seed, num_envs=64, replicate="b"
        )
        again = ramp.exact_balanced_treatment_mask(
            seed=seed, num_envs=64, replicate="a"
        )
        rank = ramp.assignment_rank(seed=seed, num_envs=64)
        assert int(a.sum()) == 32
        assert torch.equal(b, ~a)
        assert torch.equal(a, again)
        assert torch.equal(a, rank < 32)
        assert torch.equal(torch.sort(rank).values, torch.arange(64))
        assert ramp.assignment_mask_sha256(a) == ramp.assignment_mask_sha256(
            again
        )
    with pytest.raises(ValueError):
        ramp.exact_balanced_treatment_mask(
            seed=340, num_envs=63, replicate="a"
        )
    with pytest.raises(ValueError):
        ramp.exact_balanced_treatment_mask(
            seed=340, num_envs=64, replicate="c"
        )


def test_sixteen_stable_actions_are_exact_float32_linear_ramp() -> None:
    rows = 2
    treatment = torch.tensor([True, False])
    baseline = _baseline(rows)
    state = ramp.initial_public_arm_ramp_state(rows)
    denominator = torch.tensor(15.0, dtype=torch.float32)
    for index in range(16):
        result = _step(
            state, treatment=treatment, baseline=baseline
        )
        expected = torch.tensor(index, dtype=torch.long).to(torch.float32) / denominator
        assert result.ramp_scale.dtype == torch.float32
        assert torch.equal(result.ramp_scale, expected.repeat(rows))
        assert torch.equal(
            result.authority_scale, torch.tensor([expected.item(), 1.0])
        )
        assert torch.equal(
            result.action[0, : ramp.ARM_ACTION_DIM],
            baseline[0, : ramp.ARM_ACTION_DIM] * expected,
        )
        assert torch.equal(result.action[0, ramp.ARM_ACTION_DIM :], baseline[0, ramp.ARM_ACTION_DIM :])
        assert torch.equal(result.action[1], baseline[1])
        assert bool(result.positive_authority_this_action[0]) == (index > 0)
        assert bool(result.full_authority_this_action[0]) == (index == 15)
        assert bool(result.newly_positive_authority[0]) == (index == 1)
        assert bool(result.newly_full_authority[0]) == (index == 15)
        assert bool(result.verification_complete.all()) == (index == 14)
        state = result.next_state


def test_current_instability_zeros_same_action_resets_and_requires_ramp_restart() -> None:
    rows = 1
    baseline = _baseline(rows)
    state = ramp.initial_public_arm_ramp_state(rows)
    for _ in range(6):
        result = _step(state, baseline=baseline)
        state = result.next_state
    assert float(state.ramp_scale_previous[0]) == pytest.approx(5.0 / 15.0)

    unstable = _observation(rows)
    unstable[:, ramp.HOLD_QUALITY_INDEX] = 0.49
    failed = _step(
        state, observation=unstable, baseline=baseline
    )
    assert not bool(failed.stable_current[0])
    assert float(failed.authority_scale[0]) == 0.0
    assert torch.equal(
        failed.action[0, : ramp.ARM_ACTION_DIM],
        torch.zeros(ramp.ARM_ACTION_DIM),
    )
    assert int(failed.stable_count_after[0]) == 0
    assert bool(failed.relock[0])
    state = failed.next_state

    zero = _step(state, baseline=baseline)
    assert float(zero.ramp_scale[0]) == 0.0
    assert not bool(zero.reramp[0])
    positive = _step(zero.next_state, baseline=baseline)
    assert torch.equal(
        positive.ramp_scale,
        torch.tensor([1], dtype=torch.long).to(torch.float32)
        / torch.tensor(15.0, dtype=torch.float32),
    )
    assert bool(positive.reramp[0])
    assert not bool(positive.newly_positive_authority[0])


def test_control_hand_search_preeligibility_and_inactive_actions_are_exact() -> None:
    rows = 4
    baseline = _baseline(rows)
    state = ramp.initial_public_arm_ramp_state(rows)
    obs = _observation(rows)
    obs[2, ramp.PUBLIC_LATCH_INDEX] = 0.0
    result = _step(
        state,
        observation=obs,
        option_active=torch.tensor([True, True, True, False]),
        episode_active=torch.tensor([True, True, True, False]),
        treatment=torch.tensor([True, False, True, True]),
        baseline=baseline,
    )
    assert torch.equal(result.action[:, ramp.ARM_ACTION_DIM :], baseline[:, ramp.ARM_ACTION_DIM :])
    assert torch.equal(result.action[1:], baseline[1:])
    assert float(result.authority_scale[1]) == 1.0
    assert float(result.authority_scale[2]) == 1.0
    assert float(result.authority_scale[3]) == 1.0
    assert not bool(result.next_state.activated[2:].any())


def test_shadow_clock_is_assignment_invariant() -> None:
    rows = 4
    initial = ramp.initial_public_arm_ramp_state(rows)
    mixed = _step(
        initial,
        treatment=torch.tensor([True, False, True, False]),
    )
    all_control = _step(
        initial,
        treatment=torch.zeros(rows, dtype=torch.bool),
    )
    _assert_state_equal(mixed.next_state, all_control.next_state)
    assert torch.equal(mixed.ramp_scale, all_control.ramp_scale)
    assert not bool(all_control.positive_authority_this_action.any())


def test_reset_clears_all_authority_memory_before_new_episode_action() -> None:
    rows = 1
    baseline = _baseline(rows)
    state = ramp.initial_public_arm_ramp_state(rows)
    for _ in range(4):
        state = _step(state, baseline=baseline).next_state
    assert bool(state.ever_positive_authority[0])
    reset = _step(
        state,
        baseline=baseline,
        reset_mask=torch.ones(rows, dtype=torch.bool),
    )
    assert not bool(reset.activated_before[0])
    assert bool(reset.first_eligible[0])
    assert int(reset.stable_count_before[0]) == 0
    assert float(reset.authority_scale[0]) == 0.0
    assert not bool(reset.next_state.ever_positive_authority[0])
    assert not bool(reset.next_state.ever_full_authority[0])
    assert not bool(reset.newly_positive_authority[0])


def test_inactive_rows_preserve_state_and_common_action() -> None:
    rows = 1
    baseline = _baseline(rows)
    state = ramp.initial_public_arm_ramp_state(rows)
    for _ in range(3):
        state = _step(state, baseline=baseline).next_state
    result = _step(
        state,
        episode_active=torch.zeros(rows, dtype=torch.bool),
        baseline=baseline,
    )
    assert torch.equal(result.action, baseline)
    _assert_state_equal(result.next_state, state)
    assert not bool(result.relock[0])


def test_search_relocks_shadow_clock_without_editing_search_action() -> None:
    rows = 1
    baseline = _baseline(rows)
    state = ramp.initial_public_arm_ramp_state(rows)
    for _ in range(3):
        state = _step(state, baseline=baseline).next_state
    result = _step(
        state,
        option_active=torch.zeros(rows, dtype=torch.bool),
        baseline=baseline,
    )
    assert torch.equal(result.action, baseline)
    assert float(result.authority_scale[0]) == 1.0
    assert float(result.ramp_scale[0]) == 0.0
    assert int(result.stable_count_after[0]) == 0
    assert bool(result.relock[0])


def test_corrupt_state_and_wrong_action_dtype_fail_closed() -> None:
    state = ramp.initial_public_arm_ramp_state(2)
    corrupt = replace(
        state,
        activated=torch.tensor([True, False]),
        stable_count=torch.tensor([3, 0]),
        ramp_scale_previous=torch.tensor([0.9, 0.0]),
        ever_positive_authority=torch.tensor([True, False]),
    )
    with pytest.raises(ValueError, match="disagrees"):
        ramp.validate_public_arm_ramp_state(
            corrupt, rows=2, device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="baseline_action"):
        _step(state, baseline=_baseline(2).to(torch.float64))
    with pytest.raises(ValueError, match="reset_mask"):
        _step(state, reset_mask=torch.zeros(2, dtype=torch.float32))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
