#!/usr/bin/env python3
"""Simulation-free tests for online SEARCH-to-FlashSAC handoff state."""

from __future__ import annotations

import torch

from online_handoff import reset_online_handoff_state, update_online_handoff


def _observation(scores: list[float], latch: list[float] | None = None) -> torch.Tensor:
    obs = torch.zeros((len(scores), 115), dtype=torch.float32)
    for row, score in enumerate(scores):
        obs[row, 92:96] = torch.tensor([score, score, score * 0.5, 0.0])
        obs[row, 96] = score
    obs[:, 106] = torch.tensor(latch or [0.0] * len(scores))
    return obs


def test_four_frame_trigger_owns_trigger_action_and_is_sticky() -> None:
    ready = torch.zeros(3, dtype=torch.long)
    active = torch.zeros(3, dtype=torch.bool)
    for frame in range(4):
        state = update_online_handoff(
            _observation([0.35, 0.29, 0.50], latch=[0.0, 0.0, 1.0]),
            ready_count_before=ready,
            option_active_before=active,
            min_score=0.30,
            hold_steps=4,
        )
        ready = state["ready_count_after"]
        active = state["option_active_after"]
        assert bool(state["trigger"][0]) is (frame == 3)
        assert not bool(state["trigger"][1])
        assert not bool(state["trigger"][2])
    assert active.tolist() == [True, False, False]

    # Losing proximity or becoming latched cannot undo an accepted handoff.
    state = update_online_handoff(
        _observation([0.0, 0.0, 0.0], latch=[1.0, 0.0, 0.0]),
        ready_count_before=ready,
        option_active_before=active,
        min_score=0.30,
        hold_steps=4,
    )
    assert state["option_active_after"].tolist() == [True, False, False]


def test_ineligible_frame_breaks_debounce_and_reset_is_per_environment() -> None:
    ready = torch.tensor([3, 3], dtype=torch.long)
    active = torch.tensor([False, True])
    state = update_online_handoff(
        _observation([0.29, 0.0]),
        ready_count_before=ready,
        option_active_before=active,
        min_score=0.30,
        hold_steps=4,
    )
    assert state["ready_count_after"].tolist() == [0, 3]
    next_ready, next_active = reset_online_handoff_state(
        ready_count=state["ready_count_after"],
        option_active=state["option_active_after"],
        done=torch.tensor([False, True]),
    )
    assert next_ready.tolist() == [0, 0]
    assert next_active.tolist() == [False, False]


def test_public_score_uses_thumb_and_second_best_nonthumb() -> None:
    obs = torch.zeros((2, 115), dtype=torch.float32)
    obs[:, 106] = 0.0
    obs[0, 92:96] = torch.tensor([0.9, 0.4, 0.1, 0.0])
    obs[0, 96] = 0.8
    obs[1, 92:96] = torch.tensor([0.9, 0.8, 0.1, 0.0])
    obs[1, 96] = 0.3
    state = update_online_handoff(
        obs,
        ready_count_before=torch.zeros(2, dtype=torch.long),
        option_active_before=torch.zeros(2, dtype=torch.bool),
        min_score=0.20,
        hold_steps=1,
    )
    torch.testing.assert_close(
        state["score"], torch.tensor([0.4, 0.3]), rtol=0.0, atol=0.0
    )
    assert state["trigger"].tolist() == [True, True]


def main() -> None:
    test_four_frame_trigger_owns_trigger_action_and_is_sticky()
    test_ineligible_frame_breaks_debounce_and_reset_is_per_environment()
    test_public_score_uses_thumb_and_second_best_nonthumb()
    print("All online handoff tests passed.")


if __name__ == "__main__":
    main()
