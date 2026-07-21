#!/usr/bin/env python3
"""CPU regression checks for the fixed-pregrasp power-close CEM contract."""

from __future__ import annotations

import torch

from power_close_search_contract import (
    aggregate_replicated_candidates,
    power_close_candidate_score,
    power_close_stable_frame,
    strict_power_close_pass,
    update_power_grasp_latch,
    update_stable_streak,
)


def _score(*, third: float, fourth: float, force: float = 20.0, xy: float = 0.0,
           clearance: float = 0.0, passed: bool = False) -> torch.Tensor:
    scalar = lambda value: torch.tensor([value], dtype=torch.float32)
    return power_close_candidate_score(
        power_close_mean=scalar(0.5),
        thumb_fraction=scalar(1.0),
        third_other_fraction=scalar(third),
        fourth_other_fraction=scalar(fourth),
        power_wrap_mean=scalar(0.4),
        power_grasp_mean=scalar(0.4),
        hold_mean=scalar(0.8),
        stable_fraction=scalar(third),
        stable_streak_at_end=torch.tensor([15 if passed else 5]),
        stable_streak_peak=torch.tensor([15 if passed else 5]),
        force_peak=scalar(force),
        xy_drift_peak=scalar(xy),
        rotation_drift_peak=scalar(0.0),
        clearance_peak=scalar(clearance),
        strict_pass=torch.tensor([passed]),
        unexpected_done=torch.tensor([False]),
    )


def main() -> None:
    replicated_score, replicated_pass, replicated_finite = aggregate_replicated_candidates(
        torch.tensor([4.0, 2.0, 8.0, 1.0]),
        torch.tensor([True, True, True, False]),
        torch.tensor([True, True, True, True]),
        replicates=2,
    )
    assert replicated_pass.tolist() == [True, False]
    assert replicated_finite.tolist() == [True, True]
    assert torch.allclose(replicated_score, torch.tensor([52.5, 2.75]))
    nonfinite_score, nonfinite_pass, nonfinite_group = aggregate_replicated_candidates(
        torch.tensor([4.0, 2.0]),
        torch.tensor([True, True]),
        torch.tensor([True, False]),
        replicates=2,
    )
    assert nonfinite_score.item() == -1.0e9
    assert not nonfinite_pass.item() and not nonfinite_group.item()

    latched = torch.tensor([True, True, True, True, False])
    thumb = torch.tensor([True, True, True, False, True])
    others = torch.tensor([3, 2, 3, 4, 4])
    quality = torch.tensor([0.35, 0.9, 0.34, 0.9, 0.9])
    hold = torch.tensor([0.5, 0.9, 0.9, 0.9, 0.9])
    force = torch.tensor([30.0, 10.0, 10.0, 10.0, 10.0])
    assert power_close_stable_frame(
        latched, thumb, others, quality, hold, force
    ).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]

    # Match DirectRLEnv's dones-before-reward ordering: the fourth high-quality frame latches for
    # the next action, so fifteen post-latch stable frames first complete on action 19.
    power_latch = torch.zeros(1, dtype=torch.bool)
    latch_confirm = torch.zeros(1, dtype=torch.long)
    latch_release = torch.zeros(1, dtype=torch.long)
    option_streak = torch.zeros(1, dtype=torch.long)
    option_peak = torch.zeros_like(option_streak)
    first_success_action = None
    for action_number in range(1, 20):
        stable = power_close_stable_frame(
            power_latch,
            torch.tensor([True]),
            torch.tensor([3]),
            torch.tensor([0.9]),
            torch.tensor([0.9]),
            torch.tensor([10.0]),
        )
        option_streak, option_peak = update_stable_streak(
            option_streak, option_peak, stable
        )
        if option_peak.item() >= 15 and first_success_action is None:
            first_success_action = action_number
        power_latch, latch_confirm, latch_release = update_power_grasp_latch(
            torch.tensor([0.9]), power_latch, latch_confirm, latch_release
        )
    assert first_success_action == 19

    streak = torch.zeros(1, dtype=torch.long)
    peak = torch.zeros_like(streak)
    for stable in [True] * 9 + [False] + [True] * 15:
        streak, peak = update_stable_streak(streak, peak, torch.tensor([stable]))
    assert streak.item() == 15 and peak.item() == 15

    strict = strict_power_close_pass(
        peak,
        torch.tensor([30.0]),
        torch.tensor([0.03]),
        torch.tensor([0.35]),
        torch.tensor([0.015]),
        torch.tensor([False]),
    )
    assert strict.item()
    assert not strict_power_close_pass(
        peak,
        torch.tensor([30.01]),
        torch.tensor([0.03]),
        torch.tensor([0.35]),
        torch.tensor([0.015]),
        torch.tensor([False]),
    ).item()

    no_third = _score(third=0.0, fourth=0.0)
    with_third = _score(third=1.0, fourth=0.0)
    with_fourth = _score(third=1.0, fourth=1.0)
    assert no_third < with_third < with_fourth
    assert _score(third=1.0, fourth=1.0, force=31.0) < with_fourth
    assert _score(third=1.0, fourth=1.0, xy=0.031) < with_fourth
    assert _score(third=1.0, fourth=1.0, clearance=0.016) < with_fourth
    assert _score(third=1.0, fourth=1.0, passed=True) > with_fourth
    print("ALL POWER CLOSE SEARCH CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
