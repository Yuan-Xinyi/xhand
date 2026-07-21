#!/usr/bin/env python3
"""Simulation-free contract tests for the strict FlashSAC evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import torch

from evaluate import (
    NOISE_GROUP_SPECS,
    PhysicalTruth,
    PRODUCTION_CRITIC_BINS,
    SMOKE_CRITIC_BINS,
    StrictEpisodeTracker,
    _atomic_write_json,
    build_strict_metrics,
    episode_quotas,
    infer_actor_architecture_from_state,
    physical_truth_from_terminal_info,
    resolve_checkpoint_directory,
    summarize,
    validate_curriculum_config,
    validate_terminal_events,
)


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _events(
    *,
    success=(False, False),
    failure=(False, False),
    time_out=(False, False),
    dropped=(False, False),
    unsafe=(False, False),
    unlatched=(False, False),
    clearance=(-0.001, -0.001),
    grasped=(False, False),
) -> dict[str, torch.Tensor]:
    return {
        "success": torch.tensor(success, dtype=torch.bool),
        "failure": torch.tensor(failure, dtype=torch.bool),
        "time_out": torch.tensor(time_out, dtype=torch.bool),
        "dropped": torch.tensor(dropped, dtype=torch.bool),
        "unsafe_force": torch.tensor(unsafe, dtype=torch.bool),
        "unlatched_clearance_ge_5cm": torch.tensor(unlatched, dtype=torch.bool),
        "true_clearance": torch.tensor(clearance, dtype=torch.float32),
        "is_grasped": torch.tensor(grasped, dtype=torch.bool),
    }


def _truth(clearance: tuple[float, float], grasped: tuple[bool, bool]) -> PhysicalTruth:
    return PhysicalTruth(
        torch.tensor(clearance, dtype=torch.float32),
        torch.tensor(grasped, dtype=torch.bool),
    )


def test_terminal_event_contract() -> None:
    terminated = torch.tensor([True, False])
    truncated = torch.tensor([False, True])
    events = _events(success=(True, False), time_out=(False, True))
    actual = validate_terminal_events(
        {"pick_tool_terminal": events}, terminated, truncated
    )
    assert set(actual) == {
        "success",
        "failure",
        "time_out",
        "dropped",
        "unsafe_force",
        "unlatched_clearance_ge_5cm",
    }

    inconsistent = _events(failure=(True, False), dropped=(False, False))
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": inconsistent},
        torch.tensor([True, False]),
        torch.tensor([False, False]),
    )
    missing = dict(events)
    del missing["success"]
    _expect_error(
        TypeError,
        validate_terminal_events,
        {"pick_tool_terminal": missing},
        terminated,
        truncated,
    )


def test_reset_before_terminal_physical_truth() -> None:
    raw = _events(
        success=(True, False),
        clearance=(0.23, 0.04),
        grasped=(True, True),
    )
    truth = physical_truth_from_terminal_info(
        {"pick_tool_terminal": raw},
        num_envs=2,
        device=torch.device("cpu"),
    )
    # The authoritative terminal payload retains env 0's 23 cm even though
    # the physical state visible after DirectRLEnv.step may already be reset.
    torch.testing.assert_close(
        truth.clearance, torch.tensor([0.23, 0.04]), rtol=0.0, atol=0.0
    )
    assert torch.equal(truth.grasped, torch.tensor([True, True]))

    invalid = _events(
        success=(True, False),
        clearance=(0.19, 0.04),
        grasped=(True, True),
    )
    _expect_error(
        RuntimeError,
        physical_truth_from_terminal_info,
        {"pick_tool_terminal": invalid},
        num_envs=2,
        device=torch.device("cpu"),
    )


def test_exact_episode_quotas_and_strict_tracker() -> None:
    assert torch.equal(episode_quotas(3, 2, device=torch.device("cpu")), torch.tensor([2, 1]))
    assert torch.equal(episode_quotas(2, 4, device=torch.device("cpu")), torch.tensor([1, 1, 0, 0]))

    tracker = StrictEpisodeTracker(
        episodes=3,
        num_envs=2,
        device=torch.device("cpu"),
        initial_truth=_truth((-0.001, -0.001), (False, False)),
    )
    quiet = _events()
    tracker.step(
        reward=torch.tensor([1.0, 2.0]),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, False]),
        events=quiet,
        transition_truth=_truth((0.06, 0.01), (True, False)),
        post_reset_truth=_truth((0.06, 0.01), (True, False)),
    )

    terminal = _events(success=(True, False), time_out=(False, True))
    tracker.step(
        reward=torch.tensor([3.0, 4.0]),
        terminated=torch.tensor([True, False]),
        truncated=torch.tensor([False, True]),
        events=terminal,
        transition_truth=_truth((0.215, 0.04), (True, False)),
        post_reset_truth=_truth((-0.001, -0.001), (False, False)),
    )
    assert len(tracker.records) == 2
    assert tracker.active.tolist() == [True, False]
    assert tracker.records[0]["success"] is True
    assert tracker.records[0]["max_true_clearance_m"] > 0.21
    assert tracker.records[0]["return"] == 4.0
    assert tracker.records[1]["time_out"] is True
    assert tracker.records[1]["return"] == 6.0

    tracker.step(
        reward=torch.tensor([5.0, 999.0]),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, False]),
        events=_events(unlatched=(True, False)),
        transition_truth=_truth((0.07, 0.50), (False, True)),
        post_reset_truth=_truth((0.07, 0.50), (False, True)),
    )
    tracker.step(
        reward=torch.tensor([7.0, 999.0]),
        terminated=torch.tensor([True, False]),
        truncated=torch.tensor([False, False]),
        events=_events(failure=(True, False), unsafe=(True, False)),
        transition_truth=_truth((0.08, 0.50), (False, True)),
        post_reset_truth=_truth((-0.001, 0.50), (False, True)),
    )
    assert tracker.complete
    assert len(tracker.records) == 3
    final = tracker.records[-1]
    assert final["failure"] and final["unsafe_force"]
    assert final["ever_unlatched_clearance_ge_5cm"]
    assert final["return"] == 12.0
    assert final["length"] == 2

    metrics = build_strict_metrics(
        tracker.records,
        checkpoint=Path("/tmp/checkpoint"),
        architecture="production",
        seed=7,
        num_envs=2,
        vector_steps=4,
        max_vector_steps=10,
        episode_length_s=20.0,
        max_episode_steps=1000,
        curriculum_dataset=Path("/tmp/curriculum.pt"),
        curriculum_dataset_sha256="test-curriculum-sha256",
        curriculum_boundary="lift_start",
        curriculum_probability=1.0,
        curriculum_joint_noise=0.01,
        use_compile=False,
        upstream_commit="test-commit",
    )
    assert metrics["events"] == {
        "success": 1,
        "failure": 1,
        "time_out": 1,
        "dropped": 0,
        "unsafe_force": 1,
        "unlatched_clearance_ge_5cm": 1,
    }
    assert metrics["funnel"] == {
        "ever_grasped": 1,
        "ever_clearance_ge_5cm": 2,
        "ever_clearance_ge_20cm": 1,
    }
    assert metrics["strict_success_rate"] == 1 / 3
    assert metrics["episode_length_s"] == 20.0
    assert metrics["max_episode_steps"] == 1000
    assert metrics["curriculum"] == {
        "dataset": "/tmp/curriculum.pt",
        "dataset_sha256": "test-curriculum-sha256",
        "boundary": "lift_start",
        "probability": 1.0,
        "joint_noise": 0.01,
    }


def test_curriculum_argument_contract() -> None:
    validate_curriculum_config(dataset=None, probability=0.0, joint_noise=0.0)
    _expect_error(
        ValueError,
        validate_curriculum_config,
        dataset=None,
        probability=1.0,
        joint_noise=0.0,
    )
    for probability in (-0.01, 1.01, float("nan")):
        _expect_error(
            ValueError,
            validate_curriculum_config,
            dataset=None,
            probability=probability,
            joint_noise=0.0,
        )
    for joint_noise in (-0.01, float("inf")):
        _expect_error(
            ValueError,
            validate_curriculum_config,
            dataset=None,
            probability=0.0,
            joint_noise=joint_noise,
        )
    with tempfile.TemporaryDirectory() as directory:
        dataset = Path(directory) / "curriculum.pt"
        dataset.touch()
        validate_curriculum_config(dataset=dataset, probability=1.0, joint_noise=0.02)
        _expect_error(
            FileNotFoundError,
            validate_curriculum_config,
            dataset=dataset.with_name("missing.pt"),
            probability=0.0,
            joint_noise=0.0,
        )


def _actor_state(*, blocks: int, hidden: int, compiled: bool) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod." if compiled else ""
    state = {
        f"{prefix}embedder.w.w.weight": torch.zeros(hidden, 115),
        f"{prefix}predictor.mean_w.w.weight": torch.zeros(21, hidden),
    }
    for index in range(blocks):
        state[f"{prefix}encoder.{index}.w1.w.weight"] = torch.zeros(4 * hidden, hidden)
    return state


def test_checkpoint_architecture_and_path_contract() -> None:
    assert PRODUCTION_CRITIC_BINS == 101
    assert SMOKE_CRITIC_BINS == 51
    assert NOISE_GROUP_SPECS == (
        ("arm", 0, 7, 1.0, 1.0, 64),
        ("token", 7, 16, 0.5, 1.25, 32),
        ("residual", 16, 21, 0.35, 1.5, 16),
    )
    for compiled in (False, True):
        assert (
            infer_actor_architecture_from_state(
                _actor_state(blocks=2, hidden=128, compiled=compiled)
            )
            == "production"
        )
        assert (
            infer_actor_architecture_from_state(
                _actor_state(blocks=1, hidden=32, compiled=compiled)
            )
            == "smoke"
        )
    mixed = _actor_state(blocks=2, hidden=128, compiled=False)
    mixed["_orig_mod.predictor.std_bias"] = torch.zeros(21)
    _expect_error(RuntimeError, infer_actor_architecture_from_state, mixed)
    _expect_error(
        RuntimeError,
        infer_actor_architecture_from_state,
        _actor_state(blocks=3, hidden=128, compiled=False),
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "checkpoint"
        root.mkdir()
        for name in ("actor.pt", "critic.pt", "target_critic.pt", "temperature.pt"):
            (root / name).touch()
        assert resolve_checkpoint_directory(root) == root.resolve()
        assert resolve_checkpoint_directory(root / "actor.pt") == root.resolve()
        (root / "critic.pt").unlink()
        _expect_error(FileNotFoundError, resolve_checkpoint_directory, root)


def test_strict_json_and_summary() -> None:
    summary = summarize([0.0, 1.0, 2.0])
    assert summary["min"] == 0.0
    assert summary["median"] == 1.0
    assert summary["max"] == 2.0
    _expect_error(FloatingPointError, summarize, [float("nan")])

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metrics.json"
        _atomic_write_json(path, {"status": "complete", "success": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "status": "complete",
            "success": 2,
        }
        _expect_error(ValueError, _atomic_write_json, path, {"bad": float("nan")})


def main() -> None:
    test_terminal_event_contract()
    test_reset_before_terminal_physical_truth()
    test_exact_episode_quotas_and_strict_tracker()
    test_curriculum_argument_contract()
    test_checkpoint_architecture_and_path_contract()
    test_strict_json_and_summary()
    print("evaluate contract tests passed")


if __name__ == "__main__":
    main()
