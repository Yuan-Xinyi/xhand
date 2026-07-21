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

from adapter import ACTION_DIM, HAND_ACTION_DIM, build_replay_transition  # noqa: E402
from train import (  # noqa: E402
    CLOSE_OPTION_TASK_MODE,
    CORE_CHECKPOINT_FILENAMES,
    EpisodeAccumulator,
    FULL21_TO_HAND14_ACTOR_PROJECTION,
    FULL_ACTION_NOISE_GROUP_SPECS,
    FULL_POLICY_ACTION_LAYOUT,
    FULL_TASK_MODE,
    FractionalUpdateBudget,
    HAND_POLICY_ACTION_LAYOUT,
    IDENTITY_ACTION_PROJECTION,
    INCOMPLETE_CHECKPOINT_FILENAME,
    PICK_TOOL_OBSERVATION_CONTRACT,
    POWER_ACTION_NOISE_GROUP_SPECS,
    POWER_CLOSE_OBSERVATION_CONTRACT,
    POWER_CLOSE_OPTION_TASK_MODE,
    PREPEND_ZERO_ARM_ACTION_PROJECTION,
    STALE_OPTIONAL_CHECKPOINT_FILENAMES,
    TASK_CONTRACT_FILENAME,
    TASK_CONTRACT_VERSION,
    TerminalEventAccumulator,
    action_noise_group_specs,
    audit_actor_checkpoint_source,
    atomic_write_json,
    build_latch_conditioned_noise_scale,
    policy_action_contract,
    clear_stale_checkpoint_optional_artifacts,
    load_audited_actor_checkpoint,
    read_checkpoint_task_contract,
    resolve_warmup_transitions,
    runtime_contract,
    save_final_checkpoint,
    task_mode_from_close_option,
    validate_checkpoint_task_contract,
    validate_checkpoint_output_separation,
    validate_close_option_training_config,
    validate_replay_task_contract,
    validate_training_source_selection,
    write_checkpoint_task_contract,
    _strict_metrics,
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

    hand_action = torch.zeros(num_envs, HAND_ACTION_DIM)
    hand_transition = build_replay_transition(
        observation,
        hand_action,
        reward,
        terminated,
        truncated,
        info,
        action_dim=HAND_ACTION_DIM,
    )
    assert hand_transition["action"].shape == (num_envs, HAND_ACTION_DIM)
    _expect_error(
        ValueError,
        build_replay_transition,
        observation,
        hand_action,
        reward,
        terminated,
        truncated,
        info,
    )


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

    power = TerminalEventAccumulator(
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=POWER_CLOSE_OPTION_TASK_MODE,
    )
    power.step(
        {
            "pick_tool_terminal": {
                "power_close_option_success": torch.tensor([True, False]),
                "power_close_option_failure": torch.tensor([False, True]),
                "power_close_option_timeout": torch.tensor([False, False]),
                "dropped": torch.tensor([False, True]),
                "unsafe_force": torch.tensor([False, False]),
                "close_option_unlatched_lift": torch.tensor([False, False]),
                "close_option_horizontal_escape": torch.tensor([False, False]),
                "close_option_lost_window": torch.tensor([False, False]),
            }
        }
    )
    assert power.metrics() == {
        "pick_tool_terminal/power_close_option_success": 1,
        "pick_tool_terminal/power_close_option_failure": 1,
        "pick_tool_terminal/power_close_option_timeout": 0,
        "pick_tool_terminal/dropped": 1,
        "pick_tool_terminal/unsafe_force": 0,
        "pick_tool_terminal/close_option_unlatched_lift": 0,
        "pick_tool_terminal/close_option_horizontal_escape": 0,
        "pick_tool_terminal/close_option_lost_window": 0,
    }


def test_power_close_strict_metrics_are_power_specific() -> None:
    info = {
        "strict_metrics": {
            "clearance_max": torch.tensor(0.012),
            "success_frac": torch.tensor(0.75),
            "is_grasped_phase_frac": torch.tensor(0.90),
            "grasp_quality_mean": torch.tensor(0.80),
        },
        "pick_tool_terminal": {
            "power_close_quality": torch.tensor([0.10, 0.85, 0.50]),
            "power_wrap_quality": torch.tensor([0.00, 0.45, 0.20]),
            "power_grasp_quality": torch.tensor([0.00, 0.40, 0.15]),
            "power_legal_other_contact_count": torch.tensor(
                [1, 3, 4], dtype=torch.long
            ),
            "power_thumb_contact": torch.tensor([False, True, False]),
            "power_is_grasped": torch.tensor([False, True, False]),
            "power_grasp_latch_confirm_steps": torch.tensor(
                [0, 7, 2], dtype=torch.long
            ),
            "power_close_option_stable_steps": torch.tensor(
                [0, 5, 1], dtype=torch.long
            ),
        },
    }
    metrics = _strict_metrics(info, task_mode=POWER_CLOSE_OPTION_TASK_MODE)
    assert metrics["clearance_max"] == float(info["strict_metrics"]["clearance_max"])
    assert "success_frac" not in metrics
    assert "is_grasped_phase_frac" not in metrics
    assert "grasp_quality_mean" not in metrics
    assert abs(metrics["power_q_close_mean"] - 1.45 / 3.0) < 1.0e-6
    assert abs(metrics["power_q_close_max"] - 0.85) < 1.0e-6
    assert abs(metrics["power_q_wrap_mean"] - 0.65 / 3.0) < 1.0e-6
    assert abs(metrics["power_grasp_quality_max"] - 0.40) < 1.0e-6
    assert metrics["power_legal_other_contacts_max"] == 4.0
    assert abs(metrics["power_thumb_contact_frac"] - 1.0 / 3.0) < 1.0e-6
    assert abs(metrics["power_other_ge3_frac"] - 2.0 / 3.0) < 1.0e-6
    assert abs(metrics["power_thumb_plus_three_frac"] - 1.0 / 3.0) < 1.0e-6
    assert abs(metrics["power_is_grasped_frac"] - 1.0 / 3.0) < 1.0e-6
    assert metrics["power_grasp_phase_frac"] == metrics["power_is_grasped_frac"]
    assert metrics["power_latch_confirm_steps_max"] == 7.0
    assert metrics["power_close_option_stable_steps_max"] == 5.0

    legacy = _strict_metrics(info, task_mode=FULL_TASK_MODE)
    assert legacy["success_frac"] == 0.75
    assert abs(legacy["is_grasped_phase_frac"] - 0.90) < 1.0e-6
    assert "power_q_close_max" not in legacy

    missing = dict(info)
    missing["pick_tool_terminal"] = dict(info["pick_tool_terminal"])
    del missing["pick_tool_terminal"]["power_close_quality"]
    _expect_error(
        TypeError,
        _strict_metrics,
        missing,
        task_mode=POWER_CLOSE_OPTION_TASK_MODE,
    )


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
    assert (
        task_mode_from_close_option(False, True)
        == POWER_CLOSE_OPTION_TASK_MODE
    )
    _expect_error(ValueError, task_mode_from_close_option, True, True)
    assert policy_action_contract(FULL_TASK_MODE) == {
        "policy_action_dim": ACTION_DIM,
        "policy_action_layout": FULL_POLICY_ACTION_LAYOUT,
        "environment_action_dim": ACTION_DIM,
        "action_projection": IDENTITY_ACTION_PROJECTION,
    }
    assert policy_action_contract(CLOSE_OPTION_TASK_MODE) == policy_action_contract(
        FULL_TASK_MODE
    )
    assert policy_action_contract(POWER_CLOSE_OPTION_TASK_MODE) == {
        "policy_action_dim": HAND_ACTION_DIM,
        "policy_action_layout": HAND_POLICY_ACTION_LAYOUT,
        "environment_action_dim": ACTION_DIM,
        "action_projection": PREPEND_ZERO_ARM_ACTION_PROJECTION,
    }
    assert action_noise_group_specs(FULL_TASK_MODE) == FULL_ACTION_NOISE_GROUP_SPECS
    assert (
        action_noise_group_specs(CLOSE_OPTION_TASK_MODE)
        == FULL_ACTION_NOISE_GROUP_SPECS
    )
    assert (
        action_noise_group_specs(POWER_CLOSE_OPTION_TASK_MODE)
        == POWER_ACTION_NOISE_GROUP_SPECS
    )

    validate_training_source_selection(
        checkpoint=None,
        actor_checkpoint=Path("actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=True,
        power_close_option_mode=False,
        demo=None,
        actor_demo=[Path("legacy-close-actor.pt")],
    )
    validate_training_source_selection(
        checkpoint=None,
        actor_checkpoint=Path("actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=True,
        demo=None,
        actor_demo=None,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=Path("full"),
        actor_checkpoint=Path("actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=None,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=None,
        actor_checkpoint=None,
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=True,
        power_close_option_mode=False,
        demo=[Path("full-transition.pt")],
        actor_demo=None,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=None,
        actor_checkpoint=None,
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=True,
        demo=None,
        actor_demo=[Path("uncontracted-power-actor.pt")],
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=None,
        actor_checkpoint=None,
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=True,
        power_close_option_mode=True,
        demo=None,
        actor_demo=None,
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
        power_close_option_mode=False,
        curriculum_dataset=Path("close.pt"),
        curriculum_boundary="close_start",
        curriculum_probability=1.0,
        curriculum_joint_noise=0.005,
        episode_length_s=0.40,
        randomize_episode_lengths=False,
    )
    validate_close_option_training_config(
        close_option_mode=False,
        power_close_option_mode=True,
        curriculum_dataset=Path("close.pt"),
        curriculum_boundary="close_start",
        curriculum_probability=1.0,
        curriculum_joint_noise=0.005,
        episode_length_s=0.40,
        randomize_episode_lengths=False,
    )
    for override in (
        {"curriculum_dataset": None},
        {"curriculum_boundary": "lift_start"},
        {"curriculum_probability": 0.5},
        {"curriculum_joint_noise": 0.021},
        {"episode_length_s": 0.39},
        {"episode_length_s": 20.0},
        {"randomize_episode_lengths": True},
    ):
        config = {
            "close_option_mode": True,
            "power_close_option_mode": False,
            "curriculum_dataset": Path("close.pt"),
            "curriculum_boundary": "close_start",
            "curriculum_probability": 1.0,
            "curriculum_joint_noise": 0.005,
            "episode_length_s": 5.0,
            "randomize_episode_lengths": False,
        }
        config.update(override)
        _expect_error(ValueError, validate_close_option_training_config, **config)

    power_too_short = {
        "close_option_mode": False,
        "power_close_option_mode": True,
        "curriculum_dataset": Path("close.pt"),
        "curriculum_boundary": "close_start",
        "curriculum_probability": 1.0,
        "curriculum_joint_noise": 0.005,
        "episode_length_s": 0.39,
        "randomize_episode_lengths": False,
    }
    _expect_error(
        ValueError,
        validate_close_option_training_config,
        **power_too_short,
    )

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
            **runtime_contract(FULL_TASK_MODE),
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
            "version": TASK_CONTRACT_VERSION,
            "task_mode": CLOSE_OPTION_TASK_MODE,
            "replay_n_step": 3,
            "replay_gamma": 0.99,
            **runtime_contract(CLOSE_OPTION_TASK_MODE),
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

        # V3 contracts fail closed when any policy/environment tensor boundary
        # is missing or altered, before a checkpoint or replay can be restored.
        valid_close_payload = {
            "version": TASK_CONTRACT_VERSION,
            "task_mode": CLOSE_OPTION_TASK_MODE,
            "replay_n_step": 3,
            "replay_gamma": 0.99,
            **runtime_contract(CLOSE_OPTION_TASK_MODE),
        }
        invalid_values = {
            "policy_action_dim": HAND_ACTION_DIM,
            "policy_action_layout": HAND_POLICY_ACTION_LAYOUT,
            "environment_action_dim": HAND_ACTION_DIM,
            "action_projection": PREPEND_ZERO_ARM_ACTION_PROJECTION,
            "observation_dim": 114,
            "observation_contract": POWER_CLOSE_OBSERVATION_CONTRACT,
        }
        for key, invalid_value in invalid_values.items():
            invalid_payload = dict(valid_close_payload)
            invalid_payload[key] = invalid_value
            (checkpoint / TASK_CONTRACT_FILENAME).write_text(
                json.dumps(invalid_payload),
                encoding="utf-8",
            )
            _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)
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
        (checkpoint / TASK_CONTRACT_FILENAME).write_text(
            json.dumps(valid_close_payload),
            encoding="utf-8",
        )
        _expect_error(
            ValueError,
            write_checkpoint_task_contract,
            checkpoint,
            task_mode=CLOSE_OPTION_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            policy_action_dim=HAND_ACTION_DIM,
        )

        # V0--V2 contracts predate policy-action fields and must infer the
        # historic full 21D layout, including legacy close-option checkpoints.
        (checkpoint / TASK_CONTRACT_FILENAME).write_text(
            json.dumps(
                {
                    "version": 2,
                    "task_mode": CLOSE_OPTION_TASK_MODE,
                    "replay_n_step": 3,
                    "replay_gamma": 0.99,
                }
            ),
            encoding="utf-8",
        )
        v2 = read_checkpoint_task_contract(checkpoint)
        assert v2["policy_action_dim"] == ACTION_DIM
        assert v2["policy_action_layout"] == FULL_POLICY_ACTION_LAYOUT
        assert v2["action_projection"] == IDENTITY_ACTION_PROJECTION
        assert v2["observation_contract"] == PICK_TOOL_OBSERVATION_CONTRACT

        # Version-1 contracts remain readable for checkpoints produced during
        # the transition, but replay resume still requires embedded metadata.
        (checkpoint / TASK_CONTRACT_FILENAME).write_text(
            json.dumps({"version": 1, "task_mode": FULL_TASK_MODE}),
            encoding="utf-8",
        )
        v1 = read_checkpoint_task_contract(checkpoint)
        assert v1["replay_n_step"] is None and v1["replay_gamma"] is None
        assert v1["policy_action_dim"] == ACTION_DIM
        assert v1["action_projection"] == IDENTITY_ACTION_PROJECTION

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

        # The hand-only power option is new in V3 and carries both its 14D
        # policy boundary and the exact 14D -> physical 21D projection.
        write_checkpoint_task_contract(
            checkpoint,
            task_mode=POWER_CLOSE_OPTION_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
        )
        power_payload = json.loads(
            (checkpoint / TASK_CONTRACT_FILENAME).read_text(encoding="utf-8")
        )
        assert power_payload == {
            "version": TASK_CONTRACT_VERSION,
            "task_mode": POWER_CLOSE_OPTION_TASK_MODE,
            "replay_n_step": 3,
            "replay_gamma": 0.99,
            **runtime_contract(POWER_CLOSE_OPTION_TASK_MODE),
        }
        assert power_payload["policy_action_dim"] == HAND_ACTION_DIM
        assert power_payload["environment_action_dim"] == ACTION_DIM
        assert (
            power_payload["observation_contract"]
            == POWER_CLOSE_OBSERVATION_CONTRACT
        )
        torch.save({"observation": torch.zeros(1, 1)}, checkpoint / "replay_buffer.pt")
        assert validate_checkpoint_task_contract(
            checkpoint,
            task_mode=POWER_CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )["policy_action_dim"] == HAND_ACTION_DIM
        assert validate_replay_task_contract(
            checkpoint,
            task_mode=POWER_CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )["action_projection"] == PREPEND_ZERO_ARM_ACTION_PROJECTION
        for incompatible_mode in (FULL_TASK_MODE, CLOSE_OPTION_TASK_MODE):
            _expect_error(
                ValueError,
                validate_checkpoint_task_contract,
                checkpoint,
                task_mode=incompatible_mode,
                n_step=3,
                gamma=0.99,
            )
            _expect_error(
                ValueError,
                validate_replay_task_contract,
                checkpoint,
                task_mode=incompatible_mode,
                n_step=3,
                gamma=0.99,
            )

        (checkpoint / TASK_CONTRACT_FILENAME).write_text(
            json.dumps(
                {
                    "version": 2,
                    "task_mode": POWER_CLOSE_OPTION_TASK_MODE,
                    "replay_n_step": 3,
                    "replay_gamma": 0.99,
                }
            ),
            encoding="utf-8",
        )
        _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)


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
            "source_policy_action_dim": ACTION_DIM,
            "source_policy_action_layout": FULL_POLICY_ACTION_LAYOUT,
            "target_task_mode": CLOSE_OPTION_TASK_MODE,
            "target_policy_action_dim": ACTION_DIM,
            "actor_projection": None,
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
        projected = audit_actor_checkpoint_source(
            legacy_source,
            output_checkpoint=root / "power" / "checkpoint_final",
            target_task_mode=POWER_CLOSE_OPTION_TASK_MODE,
        )
        assert projected["source_policy_action_dim"] == ACTION_DIM
        assert projected["target_policy_action_dim"] == HAND_ACTION_DIM
        assert projected["actor_projection"] == FULL21_TO_HAND14_ACTOR_PROJECTION

        class FakeAgent:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def load_actor(self, *args, **kwargs) -> None:
                self.calls.append((args, kwargs))

        projected_agent = FakeAgent()
        load_audited_actor_checkpoint(projected_agent, projected)
        assert len(projected_agent.calls) == 1
        projected_args, projected_kwargs = projected_agent.calls[0]
        assert projected_args == (str(legacy_source.resolve()),)
        assert list(projected_kwargs["source_action_indices"]) == list(range(7, 21))
        assert projected_kwargs["expected_source_action_dim"] == ACTION_DIM

        identity_agent = FakeAgent()
        load_audited_actor_checkpoint(identity_agent, audit)
        assert identity_agent.calls == [((str(source.resolve()),), {})]

        power_source = root / "power_source"
        _write_core_checkpoint(power_source)
        write_checkpoint_task_contract(
            power_source,
            task_mode=POWER_CLOSE_OPTION_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
        )
        power_audit = audit_actor_checkpoint_source(
            power_source,
            output_checkpoint=root / "power_same" / "checkpoint_final",
            target_task_mode=POWER_CLOSE_OPTION_TASK_MODE,
        )
        assert power_audit["source_policy_action_layout"] == HAND_POLICY_ACTION_LAYOUT
        assert power_audit["actor_projection"] is None
        _expect_error(
            ValueError,
            audit_actor_checkpoint_source,
            power_source,
            output_checkpoint=root / "full" / "checkpoint_final",
            target_task_mode=FULL_TASK_MODE,
        )
        malformed_projection = dict(projected)
        malformed_projection["actor_projection"] = "unknown_projection"
        _expect_error(
            ValueError,
            load_audited_actor_checkpoint,
            FakeAgent(),
            malformed_projection,
        )


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
            "version": TASK_CONTRACT_VERSION,
            "task_mode": CLOSE_OPTION_TASK_MODE,
            "replay_n_step": 3,
            "replay_gamma": 0.99,
            **runtime_contract(CLOSE_OPTION_TASK_MODE),
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

    hand_scale = build_latch_conditioned_noise_scale(
        observation,
        unlatched_arm=99.0,
        unlatched_hand=0.6,
        latched_arm=88.0,
        latched_hand=0.1,
        action_dim=HAND_ACTION_DIM,
    )
    assert hand_scale.shape == (4, HAND_ACTION_DIM)
    torch.testing.assert_close(
        hand_scale,
        expected_hand[:, None].expand(-1, HAND_ACTION_DIM),
    )
    _expect_error(
        ValueError,
        build_latch_conditioned_noise_scale,
        observation,
        unlatched_arm=1.0,
        unlatched_hand=1.0,
        latched_arm=1.0,
        latched_hand=1.0,
        action_dim=13,
    )


def main() -> None:
    test_update_budget()
    test_warmup_resolution()
    test_auto_reset_replay_boundary()
    test_episode_accumulator()
    test_terminal_event_accumulator()
    test_power_close_strict_metrics_are_power_specific()
    test_atomic_json()
    test_task_mode_source_and_replay_contracts()
    test_actor_checkpoint_audit()
    test_final_checkpoint_cleanup_and_contract_order()
    test_latch_conditioned_noise_scale()
    print("train contract tests passed")


if __name__ == "__main__":
    main()
