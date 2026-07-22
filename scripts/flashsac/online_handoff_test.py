#!/usr/bin/env python3
"""Simulation-free tests for online SEARCH-to-FlashSAC handoff state."""

from __future__ import annotations

import torch

from online_handoff import (
    INITIAL_EPISODE_TERMINAL_EVENT_KEYS,
    decode_env_slot_ids,
    encode_env_slot_mask,
    online_handoff_metrics,
    reset_online_handoff_state,
    update_initial_episode_handoff_audit,
    update_online_handoff,
)


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


def _terminal_truth(
    batch: int, **events: list[bool]
) -> dict[str, torch.Tensor]:
    result = {
        name: torch.zeros(batch, dtype=torch.bool)
        for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS
    }
    for name, values in events.items():
        result[name] = torch.tensor(values, dtype=torch.bool)
    return result


def test_initial_episode_audit_counts_each_environment_exactly_once() -> None:
    pending = torch.ones(4, dtype=torch.bool)
    first = update_initial_episode_handoff_audit(
        pending_before=pending,
        triggered_seen_before=torch.zeros(4, dtype=torch.bool),
        trigger=torch.tensor([True, False, True, False]),
        done=torch.tensor([True, True, False, False]),
        terminal_truth=_terminal_truth(
            4,
            success=[True, False, False, False],
            failure=[False, True, False, False],
            dropped=[False, True, False, False],
        ),
    )
    assert first["pending_after"].tolist() == [False, False, True, True]
    assert first["triggered_seen_after"].tolist() == [True, False, True, False]
    assert int(first["trigger_count"].item()) == 2
    assert int(first["completed_triggered_episodes"].item()) == 1
    assert int(first["completed_search_only_episodes"].item()) == 1
    assert int(first["triggered_terminal_counts"]["success"].item()) == 1
    assert int(first["triggered_terminal_counts"]["unsafe_force"].item()) == 0
    # The search-only dropped row must not enter triggered terminal metrics.
    assert int(first["triggered_terminal_counts"]["dropped"].item()) == 0

    second = update_initial_episode_handoff_audit(
        pending_before=first["pending_after"],
        triggered_seen_before=first["triggered_seen_after"],
        # A trigger on already-completed row 1 cannot enter the initial cohort.
        trigger=torch.tensor([False, True, False, False]),
        # Row 0 completes again after auto-reset, but it is no longer pending.
        done=torch.tensor([True, False, True, True]),
        terminal_truth=_terminal_truth(
            4,
            failure=[True, False, True, False],
            time_out=[False, False, False, True],
            unlatched_clearance_ge_5cm=[True, False, True, False],
        ),
    )
    assert second["pending_after"].tolist() == [False, False, False, False]
    assert second["triggered_seen_after"].tolist() == [True, False, True, False]
    assert int(second["trigger_count"].item()) == 0
    assert int(second["completed_triggered_episodes"].item()) == 1
    assert int(second["completed_search_only_episodes"].item()) == 1
    # Row 0 is ignored after its first done; only still-pending row 2 counts.
    assert int(second["triggered_terminal_counts"]["failure"].item()) == 1
    assert (
        int(
            second["triggered_terminal_counts"][
                "unlatched_clearance_ge_5cm"
            ].item()
        )
        == 1
    )

    invalid_truth = _terminal_truth(1, success=[True])
    try:
        update_initial_episode_handoff_audit(
            pending_before=torch.ones(1, dtype=torch.bool),
            triggered_seen_before=torch.zeros(1, dtype=torch.bool),
            trigger=torch.ones(1, dtype=torch.bool),
            done=torch.zeros(1, dtype=torch.bool),
            terminal_truth=invalid_truth,
        )
    except ValueError as error:
        assert "done vector" in str(error)
    else:
        raise AssertionError("inconsistent terminal truth was accepted")


def test_env_slot_encoding_round_trip_is_stable() -> None:
    mask = torch.tensor([True, False, True, False, False, True])
    encoded = encode_env_slot_mask(mask, name="paired cohort")
    assert encoded == [0, 2, 5]
    torch.testing.assert_close(
        decode_env_slot_ids(encoded, num_envs=6), mask, rtol=0.0, atol=0.0
    )
    try:
        decode_env_slot_ids([2, 0], num_envs=6)
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("non-canonical environment-slot IDs were accepted")


def _online_metric_kwargs() -> dict[str, object]:
    triggered_seen = torch.tensor([True, False, True, False])
    terminal_masks = {
        name: torch.zeros(4, dtype=torch.bool)
        for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS
    }
    terminal_masks["success"][0] = True
    terminal_masks["failure"][2] = True
    terminal_masks["dropped"][2] = True
    terminal_masks["unlatched_clearance_ge_5cm"][2] = True
    terminal_counts = {
        name: mask.sum() for name, mask in terminal_masks.items()
    }
    return {
        "enabled": True,
        "search_checkpoint": "search.pth",
        "search_checkpoint_sha256": "a" * 64,
        "min_score": 0.30,
        "hold_steps": 4,
        "trigger_count": torch.tensor(7, dtype=torch.long),
        "search_action_rows": torch.tensor(80, dtype=torch.long),
        "option_action_rows": torch.tensor(20, dtype=torch.long),
        "completed_triggered_episodes": torch.tensor(5, dtype=torch.long),
        "completed_search_only_episodes": torch.tensor(3, dtype=torch.long),
        "initial_episode_pending_rows": torch.tensor(1, dtype=torch.long),
        "initial_episode_trigger_count": torch.tensor(2, dtype=torch.long),
        "initial_episode_completed_triggered_episodes": torch.tensor(
            2, dtype=torch.long
        ),
        "initial_episode_completed_search_only_episodes": torch.tensor(
            1, dtype=torch.long
        ),
        "initial_episode_triggered_seen": triggered_seen,
        "initial_episode_triggered_terminal_counts": terminal_counts,
        "initial_episode_triggered_terminal_masks": terminal_masks,
    }


def test_initial_episode_metrics_are_explicit_and_require_scalar_counts() -> None:
    kwargs = _online_metric_kwargs()
    metrics = online_handoff_metrics(**kwargs)
    assert metrics["online_search_handoff_initial_episode_pending_rows"] == 1
    assert metrics["online_search_handoff_initial_episode_trigger_count"] == 2
    assert metrics["online_search_handoff_initial_episode_triggered_env_ids"] == [
        0,
        2,
    ]
    assert (
        metrics[
            "online_search_handoff_initial_episode_completed_triggered_episodes"
        ]
        == 2
    )
    assert (
        metrics[
            "online_search_handoff_initial_episode_completed_search_only_episodes"
        ]
        == 1
    )
    expected_ids = {
        "success": [0],
        "failure": [2],
        "time_out": [],
        "dropped": [2],
        "unsafe_force": [],
        "unlatched_clearance_ge_5cm": [2],
    }
    for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS:
        assert (
            metrics[
                f"online_search_handoff_initial_episode_triggered_terminal/{name}"
            ]
            == len(expected_ids[name])
        )
        assert (
            metrics[
                f"online_search_handoff_initial_episode_triggered_terminal_env_ids/{name}"
            ]
            == expected_ids[name]
        )

    disabled_kwargs = dict(kwargs)
    disabled_kwargs.update(
        enabled=False,
        search_checkpoint=None,
        search_checkpoint_sha256=None,
    )
    disabled = online_handoff_metrics(**disabled_kwargs)
    assert disabled["online_search_handoff_initial_episode_pending_rows"] == 0
    assert disabled["online_search_handoff_initial_episode_trigger_count"] == 0
    assert disabled["online_search_handoff_initial_episode_triggered_env_ids"] == []

    nonscalar_kwargs = dict(kwargs)
    nonscalar_kwargs["initial_episode_pending_rows"] = torch.zeros(
        2, dtype=torch.long
    )
    try:
        online_handoff_metrics(**nonscalar_kwargs)
    except ValueError as error:
        assert "scalar int64" in str(error)
    else:
        raise AssertionError("non-scalar initial pending count was accepted")

    nonscalar_terminal_kwargs = dict(kwargs)
    nonscalar_terminal_counts = dict(
        kwargs["initial_episode_triggered_terminal_counts"]
    )
    nonscalar_terminal_counts["success"] = torch.zeros(2, dtype=torch.long)
    nonscalar_terminal_kwargs["initial_episode_triggered_terminal_counts"] = (
        nonscalar_terminal_counts
    )
    try:
        online_handoff_metrics(**nonscalar_terminal_kwargs)
    except ValueError as error:
        assert "scalar int64" in str(error)
    else:
        raise AssertionError("non-scalar initial terminal count was accepted")

    incomplete_outcome_kwargs = dict(kwargs)
    incomplete_outcome_masks = dict(
        kwargs["initial_episode_triggered_terminal_masks"]
    )
    incomplete_outcome_counts = dict(
        kwargs["initial_episode_triggered_terminal_counts"]
    )
    incomplete_outcome_masks["failure"] = torch.zeros(4, dtype=torch.bool)
    incomplete_outcome_counts["failure"] = torch.tensor(0, dtype=torch.long)
    incomplete_outcome_kwargs["initial_episode_triggered_terminal_masks"] = (
        incomplete_outcome_masks
    )
    incomplete_outcome_kwargs["initial_episode_triggered_terminal_counts"] = (
        incomplete_outcome_counts
    )
    try:
        online_handoff_metrics(**incomplete_outcome_kwargs)
    except ValueError as error:
        assert "do not partition" in str(error)
    else:
        raise AssertionError("incomplete initial terminal outcomes were accepted")


def main() -> None:
    test_four_frame_trigger_owns_trigger_action_and_is_sticky()
    test_ineligible_frame_breaks_debounce_and_reset_is_per_environment()
    test_public_score_uses_thumb_and_second_best_nonthumb()
    test_initial_episode_audit_counts_each_environment_exactly_once()
    test_env_slot_encoding_round_trip_is_stable()
    test_initial_episode_metrics_are_explicit_and_require_scalar_counts()
    print("All online handoff tests passed.")


if __name__ == "__main__":
    main()
