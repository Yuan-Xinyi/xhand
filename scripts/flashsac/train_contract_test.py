#!/usr/bin/env python3
"""Simulation-free contract tests for the minimal FlashSAC trainer."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapter import ACTION_DIM, build_replay_transition  # noqa: E402
from train import (  # noqa: E402
    EpisodeAccumulator,
    FractionalUpdateBudget,
    TerminalEventAccumulator,
    atomic_write_json,
    resolve_warmup_transitions,
)


def test_update_budget() -> None:
    budget = FractionalUpdateBudget(0.25)
    assert [budget.grant(False) for _ in range(20)] == [0] * 20
    assert [budget.grant(True) for _ in range(8)] == [0, 0, 0, 1, 0, 0, 0, 1]

    budget = FractionalUpdateBudget(1.5)
    assert [budget.grant(True) for _ in range(4)] == [1, 2, 1, 2]


def test_warmup_resolution() -> None:
    assert resolve_warmup_transitions(buffer=128, batch=16, smoke=True, requested=None) == 16
    assert resolve_warmup_transitions(buffer=128, batch=16, smoke=True, requested=64) == 16
    assert resolve_warmup_transitions(buffer=1_000_000, batch=2048, smoke=False, requested=None) == 10_000
    assert resolve_warmup_transitions(buffer=8192, batch=256, smoke=False, requested=None) == 8192
    assert resolve_warmup_transitions(buffer=8192, batch=256, smoke=False, requested=512) == 512


def test_auto_reset_replay_boundary() -> None:
    num_envs, obs_dim = 3, 115
    observation = torch.zeros(num_envs, obs_dim)
    reset_observation = torch.full((num_envs, obs_dim), 9.0)
    final_observation = reset_observation.clone()
    final_observation[0] = 1.0
    final_observation[1] = 2.0
    action = torch.zeros(num_envs, ACTION_DIM)
    reward = torch.arange(num_envs, dtype=torch.float32)
    terminated = torch.tensor([True, False, False])
    truncated = torch.tensor([False, True, False])
    info = {"transition_next_observation": final_observation}

    transition = build_replay_transition(
        observation,
        action,
        reward,
        terminated,
        truncated,
        info,
    )
    # SAC sees the true terminal/time-limit frames, never reset observations.
    assert torch.equal(transition["next_observation"][0], torch.ones(obs_dim))
    assert torch.equal(transition["next_observation"][1], torch.full((obs_dim,), 2.0))
    assert not torch.equal(transition["next_observation"][0], reset_observation[0])
    # Collection still continues from the adapter's returned reset observation.
    rollout_observation = reset_observation
    assert torch.equal(rollout_observation[0], torch.full((obs_dim,), 9.0))
    assert torch.equal(transition["terminated"], terminated)
    assert torch.equal(transition["truncated"], truncated)


def test_episode_accumulator() -> None:
    tracker = EpisodeAccumulator(num_envs=2, device=torch.device("cpu"))
    tracker.step(torch.tensor([1.0, 2.0]), torch.tensor([False, False]))
    tracker.step(torch.tensor([3.0, 4.0]), torch.tensor([True, False]))
    tracker.step(torch.tensor([7.0, 8.0]), torch.tensor([False, True]))
    metrics = tracker.metrics()
    assert metrics["train/completed_episodes"] == 2
    assert metrics["train/mean_episode_return"] == 9.0
    assert metrics["train/mean_episode_length"] == 2.5


def test_terminal_event_accumulator() -> None:
    tracker = TerminalEventAccumulator(num_envs=3, device=torch.device("cpu"))
    false = torch.zeros(3, dtype=torch.bool)
    tracker.step(
        {
            "pick_tool_terminal": {
                "success": torch.tensor([True, False, False]),
                "failure": torch.tensor([False, True, False]),
                "time_out": torch.tensor([False, False, True]),
                "dropped": torch.tensor([False, True, False]),
                "unsafe_force": false,
                "unlatched_clearance_ge_5cm": torch.tensor([False, False, True]),
            }
        }
    )
    assert tracker.metrics() == {
        "pick_tool_terminal/success": 1,
        "pick_tool_terminal/failure": 1,
        "pick_tool_terminal/time_out": 1,
        "pick_tool_terminal/dropped": 1,
        "pick_tool_terminal/unsafe_force": 0,
        "pick_tool_terminal/unlatched_clearance_ge_5cm": 1,
    }

    # A persistent unlatched state is one event, then may fire again only after
    # an episode boundary.
    tracker = TerminalEventAccumulator(num_envs=1, device=torch.device("cpu"))
    quiet = {
        "success": torch.tensor([False]),
        "failure": torch.tensor([False]),
        "time_out": torch.tensor([False]),
        "dropped": torch.tensor([False]),
        "unsafe_force": torch.tensor([False]),
        "unlatched_clearance_ge_5cm": torch.tensor([True]),
    }
    tracker.step({"pick_tool_terminal": quiet})
    tracker.step({"pick_tool_terminal": quiet})
    timeout = dict(quiet)
    timeout["time_out"] = torch.tensor([True])
    tracker.step({"pick_tool_terminal": timeout})
    tracker.step({"pick_tool_terminal": quiet})
    assert tracker.metrics()["pick_tool_terminal/unlatched_clearance_ge_5cm"] == 2


def test_atomic_json() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metrics.json"
        atomic_write_json(path, {"step": 1, "loss": 2.5})
        atomic_write_json(path, {"step": 2, "status": "complete"})
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "status": "complete",
            "step": 2,
        }
        assert not list(path.parent.glob(".metrics.json.tmp-*"))


def main() -> None:
    test_update_budget()
    test_warmup_resolution()
    test_auto_reset_replay_boundary()
    test_episode_accumulator()
    test_terminal_event_accumulator()
    test_atomic_json()
    print("train contract tests passed")


if __name__ == "__main__":
    main()
