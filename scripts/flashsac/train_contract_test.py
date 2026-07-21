#!/usr/bin/env python3
"""Simulation-free contract tests for the minimal FlashSAC trainer."""

from __future__ import annotations

import hashlib
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
    CLOSE_OPTION_TASK_MODE,
    CORE_CHECKPOINT_FILENAMES,
    EpisodeAccumulator,
    FULL_TASK_MODE,
    FractionalUpdateBudget,
    INCOMPLETE_CHECKPOINT_FILENAME,
    STALE_OPTIONAL_CHECKPOINT_FILENAMES,
    TASK_CONTRACT_FILENAME,
    TerminalEventAccumulator,
    audit_actor_checkpoint_source,
    atomic_write_json,
    build_latch_conditioned_noise_scale,
    clear_stale_checkpoint_optional_artifacts,
    read_checkpoint_task_contract,
    resolve_warmup_transitions,
    save_final_checkpoint,
    task_mode_from_close_option,
    validate_checkpoint_task_contract,
    validate_checkpoint_output_separation,
    validate_close_option_training_config,
    validate_replay_task_contract,
    validate_training_source_selection,
    write_checkpoint_task_contract,
)


def _expect_error(error_type, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _write_core_checkpoint(checkpoint: Path) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    for filename in CORE_CHECKPOINT_FILENAMES:
        (checkpoint / filename).write_bytes(filename.encode("utf-8"))


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

    close = TerminalEventAccumulator(
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=CLOSE_OPTION_TASK_MODE,
    )
    close.step(
        {
            "pick_tool_terminal": {
                "close_option_success": torch.tensor([True, False]),
                "close_option_failure": torch.tensor([False, True]),
                "close_option_timeout": torch.tensor([False, False]),
                "dropped": torch.tensor([False, False]),
                "unsafe_force": torch.tensor([False, False]),
                "close_option_unlatched_lift": torch.tensor([False, True]),
                "close_option_horizontal_escape": torch.tensor([False, False]),
                "close_option_lost_window": torch.tensor([False, False]),
            }
        }
    )
    assert close.metrics() == {
        "pick_tool_terminal/close_option_success": 1,
        "pick_tool_terminal/close_option_failure": 1,
        "pick_tool_terminal/close_option_timeout": 0,
        "pick_tool_terminal/dropped": 0,
        "pick_tool_terminal/unsafe_force": 0,
        "pick_tool_terminal/close_option_unlatched_lift": 1,
        "pick_tool_terminal/close_option_horizontal_escape": 0,
        "pick_tool_terminal/close_option_lost_window": 0,
    }


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


def test_task_mode_source_and_replay_contracts() -> None:
    assert task_mode_from_close_option(False) == FULL_TASK_MODE
    assert task_mode_from_close_option(True) == CLOSE_OPTION_TASK_MODE
    validate_training_source_selection(
        checkpoint=None,
        actor_checkpoint=Path("actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=True,
        demo=None,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=Path("full"),
        actor_checkpoint=Path("actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        demo=None,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=None,
        actor_checkpoint=None,
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=True,
        demo=[Path("full-transition.pt")],
    )
    validate_checkpoint_output_separation(
        Path("source/checkpoint_final"),
        output_checkpoint=Path("output/checkpoint_final"),
    )
    _expect_error(
        ValueError,
        validate_checkpoint_output_separation,
        Path("same/checkpoint_final"),
        output_checkpoint=Path("same/checkpoint_final"),
    )

    validate_close_option_training_config(
        close_option_mode=True,
        curriculum_dataset=Path("close.pt"),
        curriculum_boundary="close_start",
        curriculum_probability=1.0,
        curriculum_joint_noise=0.005,
        episode_length_s=5.0,
        randomize_episode_lengths=False,
    )
    for override in (
        {"curriculum_dataset": None},
        {"curriculum_boundary": "lift_start"},
        {"curriculum_probability": 0.5},
        {"curriculum_joint_noise": 0.021},
        {"episode_length_s": 0.29},
        {"episode_length_s": 20.0},
        {"randomize_episode_lengths": True},
    ):
        config = {
            "close_option_mode": True,
            "curriculum_dataset": Path("close.pt"),
            "curriculum_boundary": "close_start",
            "curriculum_probability": 1.0,
            "curriculum_joint_noise": 0.005,
            "episode_length_s": 5.0,
            "randomize_episode_lengths": False,
        }
        config.update(override)
        _expect_error(ValueError, validate_close_option_training_config, **config)

    with tempfile.TemporaryDirectory(prefix="flashsac_task_contract_") as directory:
        checkpoint = Path(directory) / "checkpoint"
        checkpoint.mkdir()
        _expect_error(FileNotFoundError, read_checkpoint_task_contract, checkpoint)
        _write_core_checkpoint(checkpoint)
        for missing_filename in CORE_CHECKPOINT_FILENAMES:
            missing_path = checkpoint / missing_filename
            original = missing_path.read_bytes()
            missing_path.unlink()
            _expect_error(FileNotFoundError, read_checkpoint_task_contract, checkpoint)
            missing_path.write_bytes(original)
        legacy = read_checkpoint_task_contract(checkpoint)
        assert legacy == {
            "version": 0,
            "task_mode": FULL_TASK_MODE,
            "legacy_checkpoint": True,
            "replay_n_step": None,
            "replay_gamma": None,
        }
        _expect_error(
            ValueError,
            validate_checkpoint_task_contract,
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )
        _expect_error(
            FileNotFoundError,
            validate_replay_task_contract,
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )
        torch.save(
            {
                "version": 3,
                "online": {"n_step": 3, "gamma": 0.99},
                "demos": {"n_step": 3, "gamma": 0.99},
            },
            checkpoint / "replay_buffer.pt",
        )
        validate_checkpoint_task_contract(
            checkpoint, task_mode=FULL_TASK_MODE, n_step=3, gamma=0.99
        )
        _expect_error(
            ValueError,
            validate_checkpoint_task_contract,
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=1,
            gamma=0.99,
        )
        validate_replay_task_contract(
            checkpoint, task_mode=FULL_TASK_MODE, n_step=3, gamma=0.99
        )
        for n_step, gamma in ((1, 0.99), (3, 0.95)):
            _expect_error(
                ValueError,
                validate_replay_task_contract,
                checkpoint,
                task_mode=FULL_TASK_MODE,
                n_step=n_step,
                gamma=gamma,
            )
        _expect_error(
            ValueError,
            validate_checkpoint_task_contract,
            checkpoint,
            task_mode=CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )
        _expect_error(
            ValueError,
            validate_replay_task_contract,
            checkpoint,
            task_mode=CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )

        write_checkpoint_task_contract(
            checkpoint,
            task_mode=CLOSE_OPTION_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
        )
        assert json.loads((checkpoint / TASK_CONTRACT_FILENAME).read_text()) == {
            "version": 2,
            "task_mode": CLOSE_OPTION_TASK_MODE,
            "replay_n_step": 3,
            "replay_gamma": 0.99,
        }
        contract = validate_replay_task_contract(
            checkpoint,
            task_mode=CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )
        assert contract["legacy_checkpoint"] is False
        (checkpoint / "replay_buffer.pt").unlink()
        _expect_error(
            FileNotFoundError,
            validate_replay_task_contract,
            checkpoint,
            task_mode=CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )
        # V2 sidecar metadata makes the otherwise metadata-free upstream
        # replay format auditable, provided the artifact itself exists.
        torch.save({"observation": torch.zeros(1, 1)}, checkpoint / "replay_buffer.pt")
        validate_replay_task_contract(
            checkpoint,
            task_mode=CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )
        assert validate_checkpoint_task_contract(
            checkpoint,
            task_mode=CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )["task_mode"] == CLOSE_OPTION_TASK_MODE
        _expect_error(
            ValueError,
            validate_checkpoint_task_contract,
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )
        _expect_error(
            ValueError,
            validate_replay_task_contract,
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )

        # Version-1 contracts remain readable for checkpoints produced during
        # the transition, but replay resume still requires embedded metadata.
        (checkpoint / TASK_CONTRACT_FILENAME).write_text(
            json.dumps({"version": 1, "task_mode": FULL_TASK_MODE}),
            encoding="utf-8",
        )
        v1 = read_checkpoint_task_contract(checkpoint)
        assert v1["replay_n_step"] is None and v1["replay_gamma"] is None

        (checkpoint / "replay_buffer.pt").unlink()
        torch.save({"observation": torch.zeros(1, 1)}, checkpoint / "replay_buffer.pt")
        _expect_error(
            ValueError,
            validate_replay_task_contract,
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )


def test_actor_checkpoint_audit() -> None:
    with tempfile.TemporaryDirectory(prefix="flashsac_actor_checkpoint_") as directory:
        root = Path(directory)
        source = root / "source"
        _write_core_checkpoint(source)
        actor_bytes = b"portable actor weights"
        (source / "actor.pt").write_bytes(actor_bytes)
        write_checkpoint_task_contract(
            source,
            task_mode=CLOSE_OPTION_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
        )

        audit = audit_actor_checkpoint_source(
            source,
            output_checkpoint=root / "output" / "checkpoint_final",
        )
        assert audit == {
            "path": str(source.resolve()),
            "actor_sha256": hashlib.sha256(actor_bytes).hexdigest(),
            "source_task_mode": CLOSE_OPTION_TASK_MODE,
        }
        _expect_error(
            ValueError,
            audit_actor_checkpoint_source,
            source,
            output_checkpoint=source,
        )

        legacy_source = root / "legacy"
        _write_core_checkpoint(legacy_source)
        assert audit_actor_checkpoint_source(
            legacy_source,
            output_checkpoint=root / "other" / "checkpoint_final",
        )["source_task_mode"] == FULL_TASK_MODE


def test_final_checkpoint_cleanup_and_contract_order() -> None:
    with tempfile.TemporaryDirectory(prefix="flashsac_final_checkpoint_") as directory:
        checkpoint = Path(directory) / "checkpoint_final"
        checkpoint.mkdir()
        for filename in STALE_OPTIONAL_CHECKPOINT_FILENAMES:
            (checkpoint / filename).write_text("stale", encoding="utf-8")
        (checkpoint / "actor.pt").write_text("old core", encoding="utf-8")
        sentinel = checkpoint / "unrelated.keep"
        sentinel.write_text("preserve", encoding="utf-8")
        events: list[str] = []

        class FakeAgent:
            def save(self, path: str) -> None:
                destination = Path(path)
                assert (destination / INCOMPLETE_CHECKPOINT_FILENAME).is_file()
                assert all(
                    not (destination / filename).exists()
                    for filename in STALE_OPTIONAL_CHECKPOINT_FILENAMES
                )
                assert (destination / "actor.pt").read_text(encoding="utf-8") == "old core"
                events.append("agent")
                _write_core_checkpoint(destination)

            def save_replay_buffer(self, path: str) -> None:
                destination = Path(path)
                assert not (destination / TASK_CONTRACT_FILENAME).exists()
                events.append("replay")
                (destination / "replay_buffer.pt").write_text("new replay", encoding="utf-8")

        class FakeActorRehearsal:
            def save(self, path: Path) -> None:
                assert not (path.parent / TASK_CONTRACT_FILENAME).exists()
                events.append("actor_rehearsal")
                path.write_text("new rehearsal", encoding="utf-8")

        save_final_checkpoint(
            checkpoint,
            agent=FakeAgent(),
            task_mode=CLOSE_OPTION_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            save_replay=True,
            actor_rehearsal=FakeActorRehearsal(),
        )
        assert events == ["agent", "replay", "actor_rehearsal"]
        assert sentinel.read_text(encoding="utf-8") == "preserve"
        assert not (checkpoint / INCOMPLETE_CHECKPOINT_FILENAME).exists()
        assert json.loads((checkpoint / TASK_CONTRACT_FILENAME).read_text()) == {
            "version": 2,
            "task_mode": CLOSE_OPTION_TASK_MODE,
            "replay_n_step": 3,
            "replay_gamma": 0.99,
        }

        class FailingAgent:
            def save(self, path: str) -> None:
                destination = Path(path)
                assert (destination / INCOMPLETE_CHECKPOINT_FILENAME).is_file()
                raise RuntimeError("injected checkpoint failure")

        _expect_error(
            RuntimeError,
            save_final_checkpoint,
            checkpoint,
            agent=FailingAgent(),
            task_mode=CLOSE_OPTION_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            save_replay=False,
            actor_rehearsal=None,
        )
        assert (checkpoint / INCOMPLETE_CHECKPOINT_FILENAME).is_file()
        _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)

        invalid = Path(directory) / "invalid"
        (invalid / "replay_buffer.pt").mkdir(parents=True)
        _expect_error(
            ValueError,
            clear_stale_checkpoint_optional_artifacts,
            invalid,
        )


def test_latch_conditioned_noise_scale() -> None:
    observation = torch.zeros(4, 115)
    observation[:, 106] = torch.tensor([0.0, 1.0, 0.51, 0.5])
    scale = build_latch_conditioned_noise_scale(
        observation,
        unlatched_arm=0.8,
        unlatched_hand=0.6,
        latched_arm=0.3,
        latched_hand=0.1,
    )
    expected_arm = torch.tensor([0.8, 0.3, 0.3, 0.8])
    expected_hand = torch.tensor([0.6, 0.1, 0.1, 0.6])
    torch.testing.assert_close(scale[:, :7], expected_arm[:, None].expand(-1, 7))
    torch.testing.assert_close(scale[:, 7:], expected_hand[:, None].expand(-1, 14))
    assert scale.dtype == torch.float32 and scale.device == observation.device


def main() -> None:
    test_update_budget()
    test_warmup_resolution()
    test_auto_reset_replay_boundary()
    test_episode_accumulator()
    test_terminal_event_accumulator()
    test_atomic_json()
    test_task_mode_source_and_replay_contracts()
    test_actor_checkpoint_audit()
    test_final_checkpoint_cleanup_and_contract_order()
    test_latch_conditioned_noise_scale()
    print("train contract tests passed")


if __name__ == "__main__":
    main()
