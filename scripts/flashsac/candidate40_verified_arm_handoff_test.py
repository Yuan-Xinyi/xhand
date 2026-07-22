#!/usr/bin/env python3
"""Focused CPU tests for the Candidate40 verified-arm supervisor."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import candidate40_verified_arm_handoff as verifier


N = 8


def _observation(rows: int = N) -> torch.Tensor:
    obs = torch.zeros((rows, verifier.OBSERVATION_DIM), dtype=torch.float32)
    obs[:, verifier.FORCE_STRENGTH_START : verifier.FORCE_STRENGTH_STOP] = 0.8
    obs[:, verifier.STRICT_WRAP_QUALITY_INDEX] = 0.6
    obs[:, verifier.HOLD_QUALITY_INDEX] = 0.8
    obs[:, verifier.PUBLIC_LATCH_INDEX] = 1.0
    return obs


def _counters(rows: int = N) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(rows, dtype=torch.float32)
        for name in verifier.PUBLIC_FORCE_COUNTER_FEATURES
    }


def _baseline(rows: int = N) -> torch.Tensor:
    values = torch.linspace(
        -0.9, 0.9, rows * verifier.ACTION_DIM, dtype=torch.float32
    )
    return values.reshape(rows, verifier.ACTION_DIM)


def _step(
    state: verifier.VerifiedArmState,
    *,
    observation: torch.Tensor | None = None,
    counters: dict[str, torch.Tensor] | torch.Tensor | None = None,
    option_active: torch.Tensor | None = None,
    episode_active: torch.Tensor | None = None,
    treatment: torch.Tensor | None = None,
    baseline: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> verifier.VerifiedArmStep:
    rows = int(state.activated.numel())
    reset = (
        torch.zeros(rows, dtype=torch.bool)
        if reset_mask is None
        else reset_mask
    )
    return verifier.apply_verified_arm_handoff(
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
        reset_mask=reset,
    )


def _enabled_state(rows: int = N) -> verifier.VerifiedArmState:
    state = verifier.initial_verified_arm_state(rows)
    for _ in range(verifier.VERIFY_STEPS + 1):
        result = _step(state)
        state = result.next_state
    assert bool(state.arm_enabled_previous.all())
    return state


def test_sealed_constants_match_plan_float32_value() -> None:
    expected = torch.tanh(torch.tensor(30.0 / 5.0, dtype=torch.float32)).item()
    assert verifier.VERIFIER_CONTRACT == "public_verified_live_arm_handoff_v1"
    assert verifier.PUBLIC_FORCE_STRENGTH_MAX == expected
    assert verifier.SAFE_FORCE_STRENGTH_MAX == expected
    assert verifier.VERIFY_STEPS == 15


def test_assignment_is_balanced_deterministic_and_complementary() -> None:
    for seed in (334, 335, 336):
        a = verifier.exact_balanced_treatment_mask(
            seed=seed, num_envs=64, replicate="a"
        )
        again = verifier.exact_balanced_treatment_mask(
            seed=seed, num_envs=64, replicate="a"
        )
        b = verifier.exact_balanced_treatment_mask(
            seed=seed, num_envs=64, replicate="b"
        )
        rank = verifier.assignment_rank(seed=seed, num_envs=64)
        assert torch.equal(a, again)
        assert int(a.sum()) == 32
        assert torch.equal(b, ~a)
        assert torch.equal(a, rank < 32)
        assert torch.equal(torch.sort(rank).values, torch.arange(64))
        assert verifier.assignment_mask_sha256(a) == verifier.assignment_mask_sha256(
            again
        )
    with pytest.raises(ValueError):
        verifier.exact_balanced_treatment_mask(
            seed=334, num_envs=63, replicate="a"
        )
    with pytest.raises(ValueError):
        verifier.exact_balanced_treatment_mask(
            seed=334, num_envs=64, replicate="c"
        )


def test_public_stable_predicate_uses_every_registered_public_condition() -> None:
    rows = 9
    obs = _observation(rows)
    counters = torch.zeros((rows, 2), dtype=torch.float32)
    option = torch.ones(rows, dtype=torch.bool)
    active = torch.ones(rows, dtype=torch.bool)
    option[1] = False
    active[2] = False
    obs[3, verifier.PUBLIC_LATCH_INDEX] = 0.0
    obs[4, verifier.STRICT_WRAP_QUALITY_INDEX] = 0.349
    obs[5, verifier.HOLD_QUALITY_INDEX] = 0.499
    obs[6, verifier.FORCE_STRENGTH_START] = torch.nextafter(
        torch.tensor(verifier.PUBLIC_FORCE_STRENGTH_MAX, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    counters[7, 0] = 0.1
    counters[8, 1] = 0.5
    public = verifier.public_stable_state(obs, counters, option, active)
    assert torch.equal(
        public.stable,
        torch.tensor([True, False, False, False, False, False, False, False, False]),
    )
    assert torch.equal(
        public.grasp_quality,
        torch.minimum(
            obs[:, verifier.STRICT_WRAP_QUALITY_INDEX],
            obs[:, verifier.HOLD_QUALITY_INDEX],
        ),
    )


def test_force_threshold_is_inclusive_and_mapping_order_is_canonical() -> None:
    obs = _observation(2)
    obs[0, verifier.FORCE_STRENGTH_START] = verifier.PUBLIC_FORCE_STRENGTH_MAX
    obs[1, verifier.FORCE_STRENGTH_START] = torch.nextafter(
        torch.tensor(verifier.PUBLIC_FORCE_STRENGTH_MAX, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    mapping = {
        "overforce_count_progress": torch.zeros(2, dtype=torch.float32),
        "hard_force_count_progress": torch.zeros(2, dtype=torch.float32),
    }
    result = verifier.public_stable_state(
        obs,
        mapping,
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(result.stable, torch.tensor([True, False]))
    assert torch.equal(result.force_counter_features, torch.zeros((2, 2)))


def test_first_fifteen_stable_actions_hold_and_sixteenth_enables() -> None:
    rows = 2
    treatment = torch.tensor([True, False])
    baseline = _baseline(rows)
    state = verifier.initial_verified_arm_state(rows)
    for index in range(verifier.VERIFY_STEPS + 1):
        result = _step(
            state, treatment=treatment, baseline=baseline
        )
        assert torch.equal(result.stable_count_before, torch.full((rows,), min(index, 15)))
        assert torch.equal(result.action[:, verifier.ARM_ACTION_DIM :], baseline[:, verifier.ARM_ACTION_DIM :])
        assert torch.equal(result.action[1], baseline[1])
        if index < verifier.VERIFY_STEPS:
            assert torch.equal(
                result.action[0, : verifier.ARM_ACTION_DIM],
                torch.zeros(verifier.ARM_ACTION_DIM),
            )
            assert not bool(result.arm_enabled_this_action[0])
        else:
            assert torch.equal(result.action[0], baseline[0])
            assert bool(result.arm_enabled_this_action[0])
            assert bool(result.newly_enabled[0])
        if index == verifier.VERIFY_STEPS - 1:
            assert bool(result.verification_complete.all())
            assert not bool(result.arm_enabled_this_action.any())
        else:
            assert not bool(result.verification_complete.any())
        if index == 0:
            assert bool(result.first_eligible.all())
            assert not bool(result.activated_before.any())
            # This field is post-edge state, while first_eligible is the edge.
            assert bool(result.activated_this_action.all())
        state = result.next_state


@pytest.mark.parametrize(
    "failure",
    ["latch", "wrap", "hold", "force", "hard_counter", "overforce_counter"],
)
def test_each_stability_failure_relocks_same_action_and_resets_full_clock(
    failure: str,
) -> None:
    rows = 1
    state = _enabled_state(rows)
    obs = _observation(rows)
    counters = _counters(rows)
    if failure == "latch":
        obs[:, verifier.PUBLIC_LATCH_INDEX] = 0.0
    elif failure == "wrap":
        obs[:, verifier.STRICT_WRAP_QUALITY_INDEX] = 0.34
    elif failure == "hold":
        obs[:, verifier.HOLD_QUALITY_INDEX] = 0.49
    elif failure == "force":
        obs[:, verifier.FORCE_STRENGTH_START] = 1.0
    elif failure == "hard_counter":
        counters["hard_force_count_progress"].fill_(0.1)
    else:
        counters["overforce_count_progress"].fill_(0.5)
    baseline = _baseline(rows)
    failed = _step(state, observation=obs, counters=counters, baseline=baseline)
    assert bool(failed.relock[0])
    assert not bool(failed.arm_enabled_this_action[0])
    assert int(failed.stable_count_after[0]) == 0
    assert torch.equal(
        failed.action[0, : verifier.ARM_ACTION_DIM],
        torch.zeros(verifier.ARM_ACTION_DIM),
    )
    recovered = _step(failed.next_state, baseline=baseline)
    assert int(recovered.stable_count_before[0]) == 0
    assert int(recovered.stable_count_after[0]) == 1
    assert not bool(recovered.arm_enabled_this_action[0])
    assert not bool(recovered.newly_enabled[0])


def test_search_and_pre_eligibility_actions_are_exact_common_candidate39() -> None:
    rows = 3
    baseline = _baseline(rows)
    state = verifier.initial_verified_arm_state(rows)
    obs = _observation(rows)
    obs[1:, verifier.PUBLIC_LATCH_INDEX] = 0.0
    option = torch.tensor([False, True, False])
    result = _step(
        state,
        observation=obs,
        option_active=option,
        baseline=baseline,
    )
    assert torch.equal(result.action, baseline)
    assert not bool(result.first_eligible.any())
    assert not bool(result.next_state.activated.any())


def test_treatment_changes_only_arm_and_control_shadow_state_never_edits() -> None:
    rows = 4
    treatment = torch.tensor([True, False, True, False])
    baseline = _baseline(rows)
    state = verifier.initial_verified_arm_state(rows)
    result = _step(state, treatment=treatment, baseline=baseline)
    assert torch.equal(
        result.action[:, verifier.ARM_ACTION_DIM :],
        baseline[:, verifier.ARM_ACTION_DIM :],
    )
    assert torch.equal(result.action[~treatment], baseline[~treatment])
    assert torch.equal(
        result.action[treatment, : verifier.ARM_ACTION_DIM],
        torch.zeros((2, verifier.ARM_ACTION_DIM)),
    )
    # Assignment never changes the counterfactual verifier state.
    assert torch.equal(result.next_state.stable_count, torch.ones(rows, dtype=torch.long))
    assert bool(result.next_state.activated.all())


def test_live_relock_never_changes_search_even_after_activation() -> None:
    rows = 1
    baseline = _baseline(rows)
    state = _enabled_state(rows)
    result = _step(
        state,
        option_active=torch.zeros(rows, dtype=torch.bool),
        baseline=baseline,
    )
    assert torch.equal(result.action, baseline)
    assert bool(result.relock[0])
    assert int(result.stable_count_after[0]) == 0


def test_relock_requires_full_fifteen_frame_reverification_before_reenable() -> None:
    rows = 1
    baseline = _baseline(rows)
    state = _enabled_state(rows)
    unstable = _observation(rows)
    unstable[:, verifier.PUBLIC_LATCH_INDEX] = 0.0
    failed = _step(state, observation=unstable, baseline=baseline)
    assert bool(failed.relock[0])
    state = failed.next_state

    for index in range(verifier.VERIFY_STEPS):
        held = _step(state, baseline=baseline)
        assert int(held.stable_count_before[0]) == index
        assert not bool(held.arm_enabled_this_action[0])
        assert not bool(held.newly_enabled[0])
        assert torch.equal(
            held.action[0, : verifier.ARM_ACTION_DIM],
            torch.zeros(verifier.ARM_ACTION_DIM),
        )
        state = held.next_state

    reenabled = _step(state, baseline=baseline)
    assert int(reenabled.stable_count_before[0]) == verifier.VERIFY_STEPS
    assert bool(reenabled.arm_enabled_this_action[0])
    assert bool(reenabled.newly_enabled[0])
    assert bool(reenabled.next_state.ever_enabled[0])
    assert torch.equal(reenabled.action, baseline)


def test_reset_clears_activation_clock_and_enablement_before_current_row() -> None:
    rows = 2
    state = _enabled_state(rows)
    obs = _observation(rows)
    obs[:, verifier.PUBLIC_LATCH_INDEX] = 0.0
    result = _step(
        state,
        observation=obs,
        reset_mask=torch.tensor([True, False]),
    )
    assert not bool(result.next_state.activated[0])
    assert int(result.next_state.stable_count[0]) == 0
    assert not bool(result.next_state.arm_enabled_previous[0])
    assert not bool(result.next_state.ever_enabled[0])
    assert bool(result.next_state.activated[1])
    assert int(result.next_state.stable_count[1]) == 0
    assert not bool(result.next_state.arm_enabled_previous[1])
    assert bool(result.next_state.ever_enabled[1])


def test_reset_prevents_old_authority_on_immediately_eligible_new_episode() -> None:
    rows = 1
    state = _enabled_state(rows)
    baseline = _baseline(rows)
    result = _step(
        state,
        baseline=baseline,
        reset_mask=torch.ones(rows, dtype=torch.bool),
    )
    assert not bool(result.activated_before[0])
    assert bool(result.first_eligible[0])
    assert int(result.stable_count_before[0]) == 0
    assert int(result.stable_count_after[0]) == 1
    assert not bool(result.arm_enabled_this_action[0])
    assert not bool(result.next_state.arm_enabled_previous[0])
    assert not bool(result.next_state.ever_enabled[0])
    assert torch.equal(
        result.action[0, : verifier.ARM_ACTION_DIM],
        torch.zeros(verifier.ARM_ACTION_DIM),
    )


def test_inactive_reset_clears_all_state_and_preserves_common_action() -> None:
    rows = 1
    state = _enabled_state(rows)
    baseline = _baseline(rows)
    result = _step(
        state,
        episode_active=torch.zeros(rows, dtype=torch.bool),
        baseline=baseline,
        reset_mask=torch.ones(rows, dtype=torch.bool),
    )
    assert torch.equal(result.action, baseline)
    assert not bool(result.next_state.activated.any())
    assert not bool(result.next_state.stable_count.any())
    assert not bool(result.next_state.arm_enabled_previous.any())
    assert not bool(result.next_state.ever_enabled.any())


def test_inactive_rows_preserve_state_and_action() -> None:
    rows = 1
    state = _enabled_state(rows)
    baseline = _baseline(rows)
    result = _step(
        state,
        episode_active=torch.zeros(rows, dtype=torch.bool),
        baseline=baseline,
    )
    assert torch.equal(result.action, baseline)
    assert torch.equal(result.next_state.activated, state.activated)
    assert torch.equal(result.next_state.stable_count, state.stable_count)
    assert torch.equal(
        result.next_state.arm_enabled_previous, state.arm_enabled_previous
    )
    assert not bool(result.relock.any())


def test_public_counter_and_state_corruption_fail_closed() -> None:
    obs = _observation(2)
    option = torch.ones(2, dtype=torch.bool)
    active = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError):
        verifier.public_stable_state(
            obs,
            {"hard_force_count_progress": torch.zeros(2)},
            option,
            active,
        )
    bad = _counters(2)
    bad["hard_force_count_progress"][0] = float("nan")
    with pytest.raises(ValueError):
        verifier.public_stable_state(obs, bad, option, active)
    with pytest.raises(ValueError):
        verifier.public_stable_state(
            obs,
            torch.tensor([[0.0, 0.0], [0.0, 1.1]], dtype=torch.float32),
            option,
            active,
        )

    state = verifier.initial_verified_arm_state(2)
    corrupt = replace(state, stable_count=torch.tensor([1, 0], dtype=torch.long))
    with pytest.raises(ValueError, match="unactivated slot"):
        verifier.validate_verified_arm_state(corrupt, rows=2, device=torch.device("cpu"))
    corrupt = replace(
        state,
        activated=torch.tensor([True, False]),
        arm_enabled_previous=torch.tensor([True, False]),
        ever_enabled=torch.tensor([True, False]),
    )
    with pytest.raises(ValueError, match="saturated"):
        verifier.validate_verified_arm_state(corrupt, rows=2, device=torch.device("cpu"))


def test_invalid_observation_action_latch_and_masks_fail_closed() -> None:
    state = verifier.initial_verified_arm_state(2)
    obs = _observation(2)
    obs[0, verifier.PUBLIC_LATCH_INDEX] = 0.5
    with pytest.raises(ValueError, match="binary"):
        _step(state, observation=obs)
    with pytest.raises(ValueError, match="baseline_action"):
        _step(state, baseline=torch.zeros((2, 20), dtype=torch.float32))
    with pytest.raises(ValueError, match="bounded"):
        bad_action = _baseline(2)
        bad_action[0, 0] = 1.1
        _step(state, baseline=bad_action)
    with pytest.raises(ValueError, match="treatment"):
        _step(state, treatment=torch.ones(2, dtype=torch.float32))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
