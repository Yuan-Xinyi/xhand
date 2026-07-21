#!/usr/bin/env python3
"""Simulation-free contract tests for the minimal FlashSAC trainer."""

from __future__ import annotations

import copy
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
    COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_CONTRACT,
    COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
    COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
    COUPLED_TEACHER_RESIDUAL_ACTION_PROJECTION,
    COUPLED_TEACHER_RESIDUAL_TASK_MODE,
    CORE_CHECKPOINT_FILENAMES,
    EpisodeAccumulator,
    FULL21_TO_HAND14_ACTOR_PROJECTION,
    FULL115_TO_COUPLED131_ACTOR_PROJECTION,
    FULL_ACTION_NOISE_GROUP_SPECS,
    FULL_POLICY_ACTION_LAYOUT,
    FULL_TASK_MODE,
    FROZEN_LIFT_ACTOR_FILENAME,
    FractionalUpdateBudget,
    HAND_POLICY_ACTION_LAYOUT,
    IDENTITY_ACTION_PROJECTION,
    INCOMPLETE_CHECKPOINT_FILENAME,
    PICK_TOOL_OBSERVATION_CONTRACT,
    PUBLIC_LATCH_ARM_ACTION_AUTHORITY,
    POWER_ACTION_NOISE_GROUP_SPECS,
    POWER_CLOSE_OBSERVATION_CONTRACT,
    POWER_CLOSE_OPTION_TASK_MODE,
    PREPEND_ZERO_ARM_ACTION_PROJECTION,
    STALE_OPTIONAL_CHECKPOINT_FILENAMES,
    TASK_CONTRACT_FILENAME,
    TASK_CONTRACT_VERSION,
    TEACHER_ACTION_PRIOR_FILENAME,
    TEACHER_RESIDUAL_POLICY_ACTION_LAYOUT,
    TEACHER_RESIDUAL_STATE_FILENAME,
    TerminalEventAccumulator,
    action_noise_group_specs,
    audit_actor_checkpoint_source,
    atomic_write_json,
    build_latch_conditioned_noise_scale,
    environment_task_mode_overrides,
    policy_action_contract,
    policy_action_authority_contract,
    public_latch_frozen_actor_router_contract,
    project_coupled_teacher_demo_to_zero_residual,
    project_full_actor_to_coupled_observation_state,
    clear_stale_checkpoint_optional_artifacts,
    load_audited_actor_checkpoint,
    read_checkpoint_task_contract,
    residual_actor_should_unlock,
    resolve_default_episode_length_s,
    resolve_smoke_interaction_steps,
    resolve_warmup_transitions,
    runtime_contract,
    save_final_checkpoint,
    task_mode_from_close_option,
    validate_checkpoint_task_contract,
    validate_checkpoint_output_separation,
    validate_close_option_training_config,
    validate_public_latch_arm_gate_config,
    validate_actor_demo_curriculum_lineage,
    validate_teacher_residual_actor_demo_lineage,
    validate_replay_task_contract,
    validate_training_source_selection,
    validate_teacher_residual_training_state,
    write_checkpoint_task_contract,
    _strict_metrics,
)
from coupled_teacher_prior import CoupledTeacherPrior, HandResidualScales  # noqa: E402


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frozen_actor_sidecar(
    checkpoint: Path,
    *,
    payload: bytes = b"self-contained frozen lift actor",
    network_sha256: str = "1" * 64,
    source_actor_sha256: str = "2" * 64,
) -> dict[str, object]:
    checkpoint.mkdir(parents=True, exist_ok=True)
    sidecar = checkpoint / FROZEN_LIFT_ACTOR_FILENAME
    sidecar.write_bytes(payload)
    return public_latch_frozen_actor_router_contract(
        sidecar_sha256=_sha256(sidecar),
        network_sha256=network_sha256,
        source_actor_sha256=source_actor_sha256,
    )


def _residual_prior() -> CoupledTeacherPrior:
    return CoupledTeacherPrior(
        arm_delta_target_rad=(0.0,) * 7,
        hand_latent=(0.0,) * 14,
        teacher_artifact_sha256="a" * 64,
        curriculum_dataset_sha256="b" * 64,
        teacher_artifact_path="<test-teacher>",
        curriculum_dataset_path="<test-curriculum>",
        residual_scales=HandResidualScales(),
    )


def _write_residual_sidecars(
    checkpoint: Path,
    *,
    threshold: int = 1,
    successes: int = 0,
) -> tuple[str, str, dict[str, object]]:
    checkpoint.mkdir(parents=True, exist_ok=True)
    prior_path = checkpoint / TEACHER_ACTION_PRIOR_FILENAME
    atomic_write_json(prior_path, _residual_prior().contract_payload())
    state = validate_teacher_residual_training_state(
        {
            "version": 1,
            "task_mode": COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            "actor_unlock_successes": threshold,
            "native_strict_successes": successes,
            "actor_unlocked": successes >= threshold,
        },
        source="test residual state",
    )
    state_path = checkpoint / TEACHER_RESIDUAL_STATE_FILENAME
    atomic_write_json(state_path, state)
    return _sha256(prior_path), _sha256(state_path), state


def test_update_budget() -> None:
    budget = FractionalUpdateBudget(0.25)
    assert [budget.grant(False) for _ in range(20)] == [0] * 20
    assert [budget.grant(True) for _ in range(8)] == [0, 0, 0, 1, 0, 0, 0, 1]

    budget = FractionalUpdateBudget(1.5)
    assert [budget.grant(True) for _ in range(4)] == [1, 2, 1, 2]


def test_residual_actor_unlock_boundary() -> None:
    assert not residual_actor_should_unlock(
        native_strict_successes=1,
        actor_unlock_successes=1,
        success_transition_in_replay=False,
    )
    assert residual_actor_should_unlock(
        native_strict_successes=1,
        actor_unlock_successes=1,
        success_transition_in_replay=True,
    )
    assert not residual_actor_should_unlock(
        native_strict_successes=2,
        actor_unlock_successes=3,
        success_transition_in_replay=True,
    )
    _expect_error(
        ValueError,
        residual_actor_should_unlock,
        native_strict_successes=-1,
        actor_unlock_successes=1,
        success_transition_in_replay=True,
    )


def test_warmup_resolution() -> None:
    assert resolve_warmup_transitions(buffer=128, batch=16, smoke=True, requested=None) == 16
    assert resolve_warmup_transitions(buffer=128, batch=16, smoke=True, requested=64) == 16
    assert resolve_warmup_transitions(buffer=1_000_000, batch=2048, smoke=False, requested=None) == 10_000
    assert resolve_warmup_transitions(buffer=8192, batch=256, smoke=False, requested=None) == 8192
    assert resolve_warmup_transitions(buffer=8192, batch=256, smoke=False, requested=512) == 512


def test_default_episode_horizons() -> None:
    assert resolve_default_episode_length_s(
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
        smoke=True,
    ) == 3.0
    assert resolve_default_episode_length_s(
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
        smoke=False,
    ) == 5.0
    assert resolve_default_episode_length_s(
        task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
        smoke=False,
    ) == 5.0
    assert resolve_default_episode_length_s(
        task_mode=POWER_CLOSE_OPTION_TASK_MODE,
        smoke=True,
    ) == 0.5
    assert resolve_default_episode_length_s(
        task_mode=FULL_TASK_MODE,
        smoke=True,
    ) == 0.12
    assert resolve_default_episode_length_s(
        task_mode=FULL_TASK_MODE,
        smoke=False,
    ) is None
    assert resolve_smoke_interaction_steps(
        requested=1_000,
        task_mode=FULL_TASK_MODE,
    ) == 8
    assert resolve_smoke_interaction_steps(
        requested=1,
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
    ) == 64
    assert resolve_smoke_interaction_steps(
        requested=1,
        task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
    ) == 64


def test_environment_task_mode_overrides() -> None:
    assert environment_task_mode_overrides(FULL_TASK_MODE) == {
        "close_option_mode": False,
        "power_close_option_mode": False,
        "coupled_power_align_close_option_mode": False,
    }
    assert environment_task_mode_overrides(CLOSE_OPTION_TASK_MODE) == {
        "close_option_mode": True,
        "power_close_option_mode": False,
        "coupled_power_align_close_option_mode": False,
    }
    assert environment_task_mode_overrides(POWER_CLOSE_OPTION_TASK_MODE) == {
        "close_option_mode": True,
        "power_close_option_mode": True,
        "coupled_power_align_close_option_mode": False,
    }
    assert environment_task_mode_overrides(
        COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE
    ) == {
        "close_option_mode": True,
        "power_close_option_mode": True,
        "coupled_power_align_close_option_mode": True,
        "observation_space": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
        "state_space": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
    }
    assert environment_task_mode_overrides(
        COUPLED_TEACHER_RESIDUAL_TASK_MODE
    ) == environment_task_mode_overrides(
        COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE
    )


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

    coupled = TerminalEventAccumulator(
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
    )
    coupled.step(
        {
            "pick_tool_terminal": {
                "power_close_option_success": torch.tensor([True, False]),
                "power_close_option_failure": torch.tensor([False, True]),
                "power_close_option_timeout": torch.tensor([False, False]),
                "dropped": torch.tensor([False, False]),
                "unsafe_force": torch.tensor([False, False]),
                "close_option_unlatched_lift": torch.tensor([False, False]),
                "close_option_horizontal_escape": torch.tensor([False, False]),
                "close_option_lost_window": torch.tensor([False, False]),
                "coupled_power_pose_escape": torch.tensor([False, True]),
            }
        }
    )
    assert coupled.metrics()["pick_tool_terminal/power_close_option_success"] == 1
    assert coupled.metrics()["pick_tool_terminal/power_close_option_failure"] == 1
    assert coupled.metrics()["pick_tool_terminal/coupled_power_pose_escape"] == 1


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

    coupled_terminal = dict(info["pick_tool_terminal"])
    coupled_terminal.update(
        {
            "coupled_power_pose_escape": torch.tensor([False, True, False]),
            "coupled_power_rotation_drift": torch.tensor([0.01, 0.20, 0.05]),
            "coupled_power_xy_drift": torch.tensor([0.001, 0.02, 0.005]),
            "coupled_power_true_clearance": torch.tensor([-0.001, 0.002, 0.0]),
            "coupled_power_arm_target_offset_abs_max": torch.tensor(
                [0.01, 0.12, 0.04]
            ),
            "coupled_power_arm_target_saturated": torch.tensor(
                [False, True, False]
            ),
            "coupled_power_align_active": torch.tensor([True, False, True]),
        }
    )
    coupled_metrics = _strict_metrics(
        {
            "strict_metrics": info["strict_metrics"],
            "pick_tool_terminal": coupled_terminal,
        },
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
    )
    assert "success_frac" not in coupled_metrics
    assert coupled_metrics["power_q_close_max"] == metrics["power_q_close_max"]
    assert abs(coupled_metrics["coupled_power_pose_escape_frac"] - 1.0 / 3.0) < 1e-6
    assert abs(coupled_metrics["coupled_power_align_active_frac"] - 2.0 / 3.0) < 1e-6
    assert abs(coupled_metrics["coupled_power_rotation_drift_max"] - 0.20) < 1e-6
    assert abs(coupled_metrics["coupled_power_xy_drift_max"] - 0.02) < 1e-6
    assert abs(coupled_metrics["coupled_power_true_clearance_min"] + 0.001) < 1e-6
    assert abs(
        coupled_metrics["coupled_power_arm_target_offset_abs_max_max"] - 0.12
    ) < 1e-6
    missing_coupled = dict(coupled_terminal)
    del missing_coupled["coupled_power_align_active"]
    _expect_error(
        TypeError,
        _strict_metrics,
        {"strict_metrics": {}, "pick_tool_terminal": missing_coupled},
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
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
    assert (
        task_mode_from_close_option(False, False, True)
        == COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE
    )
    assert (
        task_mode_from_close_option(False, False, False, True)
        == COUPLED_TEACHER_RESIDUAL_TASK_MODE
    )
    _expect_error(ValueError, task_mode_from_close_option, True, True)
    _expect_error(ValueError, task_mode_from_close_option, True, False, True)
    _expect_error(ValueError, task_mode_from_close_option, False, True, True)
    _expect_error(ValueError, task_mode_from_close_option, False, False, True, True)
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
    assert policy_action_contract(
        COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE
    ) == {
        "policy_action_dim": ACTION_DIM,
        "policy_action_layout": FULL_POLICY_ACTION_LAYOUT,
        "environment_action_dim": ACTION_DIM,
        "action_projection": IDENTITY_ACTION_PROJECTION,
    }
    assert policy_action_contract(COUPLED_TEACHER_RESIDUAL_TASK_MODE) == {
        "policy_action_dim": HAND_ACTION_DIM,
        "policy_action_layout": TEACHER_RESIDUAL_POLICY_ACTION_LAYOUT,
        "environment_action_dim": ACTION_DIM,
        "action_projection": COUPLED_TEACHER_RESIDUAL_ACTION_PROJECTION,
    }
    assert runtime_contract(COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE) == {
        "observation_dim": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
        "observation_contract": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_CONTRACT,
        "policy_action_dim": ACTION_DIM,
        "policy_action_layout": FULL_POLICY_ACTION_LAYOUT,
        "environment_action_dim": ACTION_DIM,
        "action_projection": IDENTITY_ACTION_PROJECTION,
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
    assert (
        action_noise_group_specs(COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE)
        == FULL_ACTION_NOISE_GROUP_SPECS
    )
    assert (
        action_noise_group_specs(COUPLED_TEACHER_RESIDUAL_TASK_MODE)
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
    validate_training_source_selection(
        checkpoint=None,
        actor_checkpoint=Path("full-actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=None,
        coupled_power_align_close_option_mode=True,
        allow_cross_task_actor=True,
    )
    validate_training_source_selection(
        checkpoint=None,
        actor_checkpoint=Path("coupled-bootstrap"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=[Path("coupled-successes.pt")],
        coupled_power_align_close_option_mode=True,
    )
    validate_training_source_selection(
        checkpoint=Path("coupled-resume"),
        actor_checkpoint=None,
        resume_replay=False,
        resume_actor_demo=True,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=[Path("coupled-successes.pt")],
        coupled_power_align_close_option_mode=True,
    )
    validate_training_source_selection(
        checkpoint=Path("residual-resume"),
        actor_checkpoint=None,
        resume_replay=True,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=None,
        coupled_teacher_residual_mode=True,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=Path("residual-without-replay"),
        actor_checkpoint=None,
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=None,
        coupled_teacher_residual_mode=True,
    )
    validate_training_source_selection(
        checkpoint=None,
        actor_checkpoint=Path("full-actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=[Path("lift-actor.pt")],
        public_latch_arm_gate=True,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=None,
        actor_checkpoint=Path("full-actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=[Path("ungated-transition.pt")],
        actor_demo=None,
        public_latch_arm_gate=True,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=None,
        actor_checkpoint=Path("full-actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=True,
        power_close_option_mode=False,
        demo=None,
        actor_demo=None,
        public_latch_arm_gate=True,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=None,
        actor_checkpoint=Path("absolute-hand-actor"),
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=None,
        coupled_teacher_residual_mode=True,
    )
    validate_training_source_selection(
        checkpoint=None,
        actor_checkpoint=None,
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=[Path("strict-coupled-teacher.pt")],
        coupled_teacher_residual_mode=True,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=None,
        actor_checkpoint=None,
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=[Path("absolute-transition-demo.pt")],
        actor_demo=None,
        coupled_teacher_residual_mode=True,
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
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=[Path("uncontracted-coupled-actor.pt")],
        coupled_power_align_close_option_mode=True,
    )
    _expect_error(
        ValueError,
        validate_training_source_selection,
        checkpoint=None,
        actor_checkpoint=None,
        resume_replay=False,
        resume_actor_demo=False,
        close_option_mode=False,
        power_close_option_mode=False,
        demo=None,
        actor_demo=None,
        coupled_power_align_close_option_mode=True,
        allow_cross_task_actor=True,
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

    curriculum_sha256 = "a" * 64
    validate_actor_demo_curriculum_lineage(
        {
            "path": "coupled-demo.pt",
            "curriculum_dataset_sha256": curriculum_sha256,
        },
        expected_curriculum_sha256=curriculum_sha256,
    )
    _expect_error(
        ValueError,
        validate_actor_demo_curriculum_lineage,
        {
            "path": "wrong-curriculum.pt",
            "curriculum_dataset_sha256": "b" * 64,
        },
        expected_curriculum_sha256=curriculum_sha256,
    )

    teacher_sha256 = "c" * 64
    residual_audit = {
        "path": "strict-coupled-teacher.pt",
        "curriculum_dataset_sha256": curriculum_sha256,
        "teacher_artifact_sha256": teacher_sha256,
    }
    validate_teacher_residual_actor_demo_lineage(
        residual_audit,
        expected_curriculum_sha256=curriculum_sha256,
        expected_teacher_artifact_sha256=teacher_sha256,
    )
    for override in (
        {"curriculum_dataset_sha256": "b" * 64},
        {"teacher_artifact_sha256": "d" * 64},
        {"teacher_artifact_sha256": "not-a-sha"},
    ):
        _expect_error(
            ValueError,
            validate_teacher_residual_actor_demo_lineage,
            {**residual_audit, **override},
            expected_curriculum_sha256=curriculum_sha256,
            expected_teacher_artifact_sha256=teacher_sha256,
        )

    source_observation = torch.randn(
        5, COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM, dtype=torch.float32
    )
    source_action = torch.randn(5, ACTION_DIM, dtype=torch.float32).clamp(-1.0, 1.0)
    projected = project_coupled_teacher_demo_to_zero_residual(
        {"observation": source_observation, "action": source_action}
    )
    assert projected["observation"] is source_observation
    assert projected["action"].shape == (5, HAND_ACTION_DIM)
    assert projected["action"].dtype == torch.float32
    assert not bool(projected["action"].any())
    _expect_error(
        ValueError,
        project_coupled_teacher_demo_to_zero_residual,
        {
            "observation": source_observation,
            "action": source_action[:, :HAND_ACTION_DIM],
        },
    )
    _expect_error(
        ValueError,
        validate_actor_demo_curriculum_lineage,
        {"path": "missing-lineage.pt"},
        expected_curriculum_sha256=curriculum_sha256,
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
        power_close_option_mode=False,
        curriculum_dataset=Path("close.pt"),
        curriculum_boundary="close_start",
        curriculum_probability=1.0,
        curriculum_joint_noise=0.02,
        episode_length_s=3.0,
        randomize_episode_lengths=False,
        coupled_power_align_close_option_mode=True,
    )
    validate_close_option_training_config(
        close_option_mode=False,
        power_close_option_mode=False,
        curriculum_dataset=Path("close.pt"),
        curriculum_boundary="close_start",
        curriculum_probability=1.0,
        curriculum_joint_noise=0.0,
        episode_length_s=3.0,
        randomize_episode_lengths=False,
        coupled_teacher_residual_mode=True,
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
    coupled_config = {
        "close_option_mode": False,
        "power_close_option_mode": False,
        "curriculum_dataset": Path("close.pt"),
        "curriculum_boundary": "close_start",
        "curriculum_probability": 1.0,
        "curriculum_joint_noise": 0.005,
        "episode_length_s": 2.99,
        "randomize_episode_lengths": False,
        "coupled_power_align_close_option_mode": True,
    }
    _expect_error(
        ValueError,
        validate_close_option_training_config,
        **coupled_config,
    )

    assert policy_action_authority_contract(False) == []
    assert policy_action_authority_contract(True) == [
        dict(PUBLIC_LATCH_ARM_ACTION_AUTHORITY[0])
    ]
    public_gate_config = {
        "enabled": True,
        "task_mode": FULL_TASK_MODE,
        "curriculum_dataset": Path("close.pt"),
        "curriculum_boundary": "close_start",
        "curriculum_probability": 1.0,
        "curriculum_joint_noise": 0.01,
        "episode_length_s": 20.0,
        "randomize_episode_lengths": False,
    }
    validate_public_latch_arm_gate_config(**public_gate_config)
    for override in (
        {"task_mode": CLOSE_OPTION_TASK_MODE},
        {"curriculum_dataset": None},
        {"curriculum_boundary": "default"},
        {"curriculum_probability": 0.5},
        {"curriculum_joint_noise": 0.021},
        {"episode_length_s": 0.29},
        {"randomize_episode_lengths": True},
    ):
        invalid_gate = dict(public_gate_config)
        invalid_gate.update(override)
        _expect_error(
            ValueError,
            validate_public_latch_arm_gate_config,
            **invalid_gate,
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
            "policy_action_authority": [],
            "policy_router": None,
            "teacher_action_prior": None,
            "teacher_residual_state": None,
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
            "policy_action_authority": [],
            "policy_router": None,
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
            "policy_action_authority": [],
            "policy_router": None,
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
        assert v1["policy_action_authority"] == []
        assert v1["policy_router"] is None
        _expect_error(
            ValueError,
            validate_replay_task_contract,
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )

        # V4 predates state-dependent policy authority and remains readable as
        # unrestricted. V5 records the public latch gate and full/replay resume
        # must match it exactly in both directions.
        (checkpoint / TASK_CONTRACT_FILENAME).write_text(
            json.dumps(
                {
                    "version": 4,
                    "task_mode": FULL_TASK_MODE,
                    "replay_n_step": 3,
                    "replay_gamma": 0.99,
                    **runtime_contract(FULL_TASK_MODE),
                }
            ),
            encoding="utf-8",
        )
        assert read_checkpoint_task_contract(checkpoint)[
            "policy_action_authority"
        ] == []
        gated_authority = policy_action_authority_contract(True)
        write_checkpoint_task_contract(
            checkpoint,
            task_mode=FULL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            policy_action_authority=gated_authority,
        )
        gated_contract = read_checkpoint_task_contract(checkpoint)
        assert gated_contract["policy_action_authority"] == gated_authority
        validate_checkpoint_task_contract(
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
            policy_action_authority=gated_authority,
        )
        validate_replay_task_contract(
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
            policy_action_authority=gated_authority,
        )
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
            "policy_action_authority": [],
            "policy_router": None,
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

        # Coupled power is a separate 131D/21D identity contract. Full-agent
        # and replay restore remain strictly same-task despite actor-only transfer.
        write_checkpoint_task_contract(
            checkpoint,
            task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
        )
        coupled_payload = json.loads(
            (checkpoint / TASK_CONTRACT_FILENAME).read_text(encoding="utf-8")
        )
        assert coupled_payload == {
            "version": TASK_CONTRACT_VERSION,
            "task_mode": COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
            "replay_n_step": 3,
            "replay_gamma": 0.99,
            "policy_action_authority": [],
            "policy_router": None,
            **runtime_contract(COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE),
        }
        assert validate_checkpoint_task_contract(
            checkpoint,
            task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )["observation_dim"] == COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM
        assert validate_replay_task_contract(
            checkpoint,
            task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )["action_projection"] == IDENTITY_ACTION_PROJECTION
        for incompatible_mode in (
            FULL_TASK_MODE,
            CLOSE_OPTION_TASK_MODE,
            POWER_CLOSE_OPTION_TASK_MODE,
        ):
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

        # Residual V4 binds both the immutable CEM prior and the resumable
        # actor-authority gate. The policy action is canonical residual14,
        # never an absolute hand14 action from another task.
        prior_sha, state_sha, expected_state = _write_residual_sidecars(
            checkpoint,
            threshold=3,
            successes=5,
        )
        write_checkpoint_task_contract(
            checkpoint,
            task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            teacher_action_prior_sha256=prior_sha,
            teacher_residual_state_sha256=state_sha,
        )
        residual_contract = validate_checkpoint_task_contract(
            checkpoint,
            task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )
        assert residual_contract["teacher_residual_state"] == expected_state
        assert residual_contract["teacher_action_prior"] == {
            "filename": TEACHER_ACTION_PRIOR_FILENAME,
            "sha256": prior_sha,
        }
        assert residual_contract["policy_action_dim"] == HAND_ACTION_DIM
        assert (
            residual_contract["action_projection"]
            == COUPLED_TEACHER_RESIDUAL_ACTION_PROJECTION
        )
        assert validate_replay_task_contract(
            checkpoint,
            task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            n_step=3,
            gamma=0.99,
        )["teacher_residual_state"] == expected_state
        state_path = checkpoint / TEACHER_RESIDUAL_STATE_FILENAME
        state_path.write_text("{}", encoding="utf-8")
        _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)
        prior_sha, state_sha, _ = _write_residual_sidecars(
            checkpoint,
            threshold=3,
            successes=5,
        )
        write_checkpoint_task_contract(
            checkpoint,
            task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            teacher_action_prior_sha256=prior_sha,
            teacher_residual_state_sha256=state_sha,
        )
        _expect_error(
            ValueError,
            write_checkpoint_task_contract,
            checkpoint,
            task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            teacher_action_prior_sha256=prior_sha,
        )

        (checkpoint / TASK_CONTRACT_FILENAME).write_text(
            json.dumps(
                {
                    "version": 2,
                    "task_mode": COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
                    "replay_n_step": 3,
                    "replay_gamma": 0.99,
                }
            ),
            encoding="utf-8",
        )
        _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)

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


def test_v6_policy_router_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="flashsac_router_contract_") as directory:
        root = Path(directory)
        checkpoint = root / "routed"
        _write_core_checkpoint(checkpoint)
        torch.save({"observation": torch.zeros(1, 1)}, checkpoint / "replay_buffer.pt")
        authority = policy_action_authority_contract(True)
        router = _write_frozen_actor_sidecar(checkpoint)
        write_checkpoint_task_contract(
            checkpoint,
            task_mode=FULL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            policy_action_authority=authority,
            policy_router=router,
        )
        contract_path = checkpoint / TASK_CONTRACT_FILENAME
        valid_payload = json.loads(contract_path.read_text(encoding="utf-8"))
        assert valid_payload["version"] == 6 == TASK_CONTRACT_VERSION
        assert valid_payload["policy_router"] == router
        contract = read_checkpoint_task_contract(checkpoint)
        assert contract["policy_router"] == router
        assert contract["policy_action_authority"] == authority
        assert contract["policy_router"]["frozen_actor"]["sha256"] == _sha256(
            checkpoint / FROZEN_LIFT_ACTOR_FILENAME
        )
        assert validate_checkpoint_task_contract(
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
            policy_action_authority=authority,
            policy_router_enabled=True,
        )["policy_router"] == router
        assert validate_replay_task_contract(
            checkpoint,
            task_mode=FULL_TASK_MODE,
            n_step=3,
            gamma=0.99,
            policy_action_authority=authority,
            policy_router_enabled=True,
        )["policy_router"] == router

        # A routed checkpoint cannot be resumed by a non-routed run, even when
        # every other task, authority, runtime, and discount field agrees.
        for validator in (
            validate_checkpoint_task_contract,
            validate_replay_task_contract,
        ):
            _expect_error(
                ValueError,
                validator,
                checkpoint,
                task_mode=FULL_TASK_MODE,
                n_step=3,
                gamma=0.99,
                policy_action_authority=authority,
                policy_router_enabled=False,
            )

        sidecar = checkpoint / FROZEN_LIFT_ACTOR_FILENAME
        sidecar_bytes = sidecar.read_bytes()

        sidecar.unlink()
        _expect_error(FileNotFoundError, read_checkpoint_task_contract, checkpoint)

        external_sidecar = root / "external_frozen_actor.pt"
        external_sidecar.write_bytes(sidecar_bytes)
        sidecar.symlink_to(external_sidecar)
        _expect_error(FileNotFoundError, read_checkpoint_task_contract, checkpoint)
        sidecar.unlink()

        sidecar.write_bytes(sidecar_bytes + b"tampered")
        _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)
        sidecar.write_bytes(sidecar_bytes)

        missing_router = dict(valid_payload)
        missing_router.pop("policy_router")
        contract_path.write_text(json.dumps(missing_router), encoding="utf-8")
        _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)

        orphaned_sidecar = dict(valid_payload)
        orphaned_sidecar["policy_router"] = None
        contract_path.write_text(json.dumps(orphaned_sidecar), encoding="utf-8")
        _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)

        # Routing decisions and tensor boundaries are closed static fields;
        # neither substitutions nor forward-compatible-looking extras are
        # accepted silently.
        invalid_static_values = (
            (("kind",), "another_router"),
            (("observation_index",), 105),
            (("close_value",), 1.0),
            (("frozen_value",), 0.0),
            (("trainable_action_slice",), [0, 21]),
            (("close_arm_fill",), "learned"),
            (("frozen_action_slice",), [7, 21]),
            (("frozen_action_sampling",), "stochastic"),
            (("frozen_action_entropy",), "learned"),
            (("frozen_actor", "filename"), "other.pt"),
            (("frozen_actor", "sha256"), "not-a-sha256"),
            (("frozen_actor", "observation_dim"), 114),
            (("frozen_actor", "action_dim"), 20),
        )
        for field_path, invalid_value in invalid_static_values:
            candidate = copy.deepcopy(valid_payload)
            target = candidate["policy_router"]
            for field in field_path[:-1]:
                target = target[field]
            target[field_path[-1]] = invalid_value
            contract_path.write_text(json.dumps(candidate), encoding="utf-8")
            _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)

        extra_outer = copy.deepcopy(valid_payload)
        extra_outer["policy_router"]["unexpected"] = True
        contract_path.write_text(json.dumps(extra_outer), encoding="utf-8")
        _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)

        extra_frozen_actor = copy.deepcopy(valid_payload)
        extra_frozen_actor["policy_router"]["frozen_actor"]["unexpected"] = True
        contract_path.write_text(json.dumps(extra_frozen_actor), encoding="utf-8")
        _expect_error(ValueError, read_checkpoint_task_contract, checkpoint)

        plain = root / "non_routed"
        _write_core_checkpoint(plain)
        torch.save({"observation": torch.zeros(1, 1)}, plain / "replay_buffer.pt")
        write_checkpoint_task_contract(
            plain,
            task_mode=FULL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            policy_action_authority=authority,
        )
        assert read_checkpoint_task_contract(plain)["policy_router"] is None

        # The inverse resume direction also fails closed: enabling routing
        # cannot manufacture the immutable lift branch absent from a source.
        for validator in (
            validate_checkpoint_task_contract,
            validate_replay_task_contract,
        ):
            _expect_error(
                ValueError,
                validator,
                plain,
                task_mode=FULL_TASK_MODE,
                n_step=3,
                gamma=0.99,
                policy_action_authority=authority,
                policy_router_enabled=True,
            )

        # V5 predates policy routing and remains readable as an explicit None.
        (plain / TASK_CONTRACT_FILENAME).write_text(
            json.dumps(
                {
                    "version": 5,
                    "task_mode": FULL_TASK_MODE,
                    "replay_n_step": 3,
                    "replay_gamma": 0.99,
                    "policy_action_authority": authority,
                    **runtime_contract(FULL_TASK_MODE),
                }
            ),
            encoding="utf-8",
        )
        v5 = read_checkpoint_task_contract(plain)
        assert v5["version"] == 5
        assert v5["policy_router"] is None


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
            "source_policy_action_authority": [],
            "source_policy_router": None,
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
        _expect_error(
            ValueError,
            audit_actor_checkpoint_source,
            legacy_source,
            output_checkpoint=root / "coupled_default_reject" / "checkpoint_final",
            target_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
        )
        coupled_projected = audit_actor_checkpoint_source(
            legacy_source,
            output_checkpoint=root / "coupled" / "checkpoint_final",
            target_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
            allow_cross_task_actor=True,
        )
        assert coupled_projected["source_task_mode"] == FULL_TASK_MODE
        assert coupled_projected["source_policy_action_dim"] == ACTION_DIM
        assert coupled_projected["target_policy_action_dim"] == ACTION_DIM
        assert (
            coupled_projected["actor_projection"]
            == FULL115_TO_COUPLED131_ACTOR_PROJECTION
        )
        _expect_error(
            ValueError,
            audit_actor_checkpoint_source,
            source,
            output_checkpoint=root / "close_to_coupled" / "checkpoint_final",
            target_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
            allow_cross_task_actor=True,
        )

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
        coupled_source = root / "coupled_source"
        _write_core_checkpoint(coupled_source)
        write_checkpoint_task_contract(
            coupled_source,
            task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
        )
        coupled_identity = audit_actor_checkpoint_source(
            coupled_source,
            output_checkpoint=root / "coupled_same" / "checkpoint_final",
            target_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
        )
        assert coupled_identity["actor_projection"] is None
        _expect_error(
            ValueError,
            audit_actor_checkpoint_source,
            coupled_source,
            output_checkpoint=root / "coupled_to_full" / "checkpoint_final",
            target_task_mode=FULL_TASK_MODE,
        )
        residual_source = root / "residual_source"
        _write_core_checkpoint(residual_source)
        residual_prior_sha, residual_state_sha, _ = _write_residual_sidecars(
            residual_source,
            successes=1,
        )
        write_checkpoint_task_contract(
            residual_source,
            task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            teacher_action_prior_sha256=residual_prior_sha,
            teacher_residual_state_sha256=residual_state_sha,
        )
        residual_identity = audit_actor_checkpoint_source(
            residual_source,
            output_checkpoint=root / "residual_same" / "checkpoint_final",
            target_task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
        )
        assert residual_identity["actor_projection"] is None
        for target_mode in (
            FULL_TASK_MODE,
            POWER_CLOSE_OPTION_TASK_MODE,
            COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
        ):
            _expect_error(
                ValueError,
                audit_actor_checkpoint_source,
                residual_source,
                output_checkpoint=root / f"residual_to_{target_mode}" / "checkpoint_final",
                target_task_mode=target_mode,
            )
        _expect_error(
            ValueError,
            audit_actor_checkpoint_source,
            power_source,
            output_checkpoint=root / "absolute_to_residual" / "checkpoint_final",
            target_task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
        )
        malformed_projection = dict(projected)
        malformed_projection["actor_projection"] = "unknown_projection"
        _expect_error(
            ValueError,
            load_audited_actor_checkpoint,
            FakeAgent(),
            malformed_projection,
        )


def test_full_actor_coupled_observation_projection() -> None:
    source = {
        "embedder.norm.weight": torch.arange(115, dtype=torch.float32),
        "embedder.norm.bias": torch.arange(115, dtype=torch.float32) + 100.0,
        "embedder.norm.running_mean": torch.arange(115, dtype=torch.float32) + 200.0,
        "embedder.norm.running_var": torch.arange(115, dtype=torch.float32) + 300.0,
        "embedder.w.w.weight": torch.arange(4 * 115, dtype=torch.float32).reshape(4, 115),
        "encoder.0.w1.w.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
    }
    target = {
        "_orig_mod.embedder.norm.weight": torch.full((131,), 1.0),
        "_orig_mod.embedder.norm.bias": torch.full((131,), 2.0),
        "_orig_mod.embedder.norm.running_mean": torch.full((131,), 3.0),
        "_orig_mod.embedder.norm.running_var": torch.full((131,), 4.0),
        "_orig_mod.embedder.w.w.weight": torch.full((4, 131), 5.0),
        "_orig_mod.encoder.0.w1.w.weight": torch.full((4, 4), 6.0),
    }
    projected = project_full_actor_to_coupled_observation_state(source, target)
    assert set(projected) == set(target)
    torch.testing.assert_close(
        projected["_orig_mod.embedder.w.w.weight"][:, :115],
        source["embedder.w.w.weight"],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        projected["_orig_mod.embedder.w.w.weight"][:, 115:],
        torch.zeros(4, 16),
    )
    for key, default in (
        ("weight", 1.0),
        ("bias", 2.0),
        ("running_mean", 3.0),
        ("running_var", 4.0),
    ):
        target_key = f"_orig_mod.embedder.norm.{key}"
        torch.testing.assert_close(
            projected[target_key][:115],
            source[f"embedder.norm.{key}"],
            rtol=0.0,
            atol=0.0,
        )
        assert torch.equal(
            projected[target_key][115:],
            torch.full((16,), default),
        )
    assert torch.equal(
        projected["_orig_mod.encoder.0.w1.w.weight"],
        source["encoder.0.w1.w.weight"],
    )

    malformed = dict(target)
    malformed["_orig_mod.encoder.0.w1.w.weight"] = torch.zeros(5, 4)
    _expect_error(
        ValueError,
        project_full_actor_to_coupled_observation_state,
        source,
        malformed,
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
            "policy_action_authority": [],
            "policy_router": None,
            **runtime_contract(CLOSE_OPTION_TASK_MODE),
        }

        routed_checkpoint = Path(directory) / "routed_checkpoint_final"
        routed_sidecar_bytes = b"published immutable lift actor"
        routed_network_sha256 = "3" * 64
        routed_source_actor_sha256 = "4" * 64

        class RoutedAgent:
            frozen_lift_actor_sha256 = routed_network_sha256
            frozen_lift_actor_source_sha256 = routed_source_actor_sha256

            def save(self, path: str) -> None:
                destination = Path(path)
                assert (destination / INCOMPLETE_CHECKPOINT_FILENAME).is_file()
                assert not (destination / TASK_CONTRACT_FILENAME).exists()
                _write_core_checkpoint(destination)
                (destination / FROZEN_LIFT_ACTOR_FILENAME).write_bytes(
                    routed_sidecar_bytes
                )

            def save_replay_buffer(self, path: str) -> None:
                raise AssertionError("routed test did not request replay save")

        routed_authority = policy_action_authority_contract(True)
        save_final_checkpoint(
            routed_checkpoint,
            agent=RoutedAgent(),
            task_mode=FULL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            save_replay=False,
            actor_rehearsal=None,
            policy_action_authority=routed_authority,
            policy_router_enabled=True,
        )
        assert not (routed_checkpoint / INCOMPLETE_CHECKPOINT_FILENAME).exists()
        assert (
            routed_checkpoint / FROZEN_LIFT_ACTOR_FILENAME
        ).read_bytes() == routed_sidecar_bytes
        routed_contract = read_checkpoint_task_contract(routed_checkpoint)
        assert routed_contract["policy_action_authority"] == routed_authority
        assert routed_contract["policy_router"] == (
            public_latch_frozen_actor_router_contract(
                sidecar_sha256=_sha256(
                    routed_checkpoint / FROZEN_LIFT_ACTOR_FILENAME
                ),
                network_sha256=routed_network_sha256,
                source_actor_sha256=routed_source_actor_sha256,
            )
        )

        residual_checkpoint = Path(directory) / "residual_checkpoint_final"

        class ResidualAgent:
            def save(self, path: str) -> None:
                _write_core_checkpoint(Path(path))

            def save_replay_buffer(self, path: str) -> None:
                raise AssertionError("residual test did not request replay save")

        residual_state = validate_teacher_residual_training_state(
            {
                "version": 1,
                "task_mode": COUPLED_TEACHER_RESIDUAL_TASK_MODE,
                "actor_unlock_successes": 2,
                "native_strict_successes": 3,
                "actor_unlocked": True,
            },
            source="final checkpoint test",
        )
        save_final_checkpoint(
            residual_checkpoint,
            agent=ResidualAgent(),
            task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            save_replay=False,
            actor_rehearsal=None,
            teacher_prior=_residual_prior(),
            teacher_residual_state=residual_state,
        )
        residual_contract = read_checkpoint_task_contract(residual_checkpoint)
        assert residual_contract["teacher_residual_state"] == residual_state
        assert (residual_checkpoint / TEACHER_ACTION_PRIOR_FILENAME).is_file()
        assert (residual_checkpoint / TEACHER_RESIDUAL_STATE_FILENAME).is_file()
        _expect_error(
            ValueError,
            save_final_checkpoint,
            Path(directory) / "invalid_residual_checkpoint",
            agent=ResidualAgent(),
            task_mode=COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            replay_n_step=3,
            replay_gamma=0.99,
            save_replay=False,
            actor_rehearsal=None,
            teacher_prior=_residual_prior(),
            teacher_residual_state=None,
        )
        assert not (Path(directory) / "invalid_residual_checkpoint").exists()

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
    test_residual_actor_unlock_boundary()
    test_warmup_resolution()
    test_default_episode_horizons()
    test_environment_task_mode_overrides()
    test_auto_reset_replay_boundary()
    test_episode_accumulator()
    test_terminal_event_accumulator()
    test_power_close_strict_metrics_are_power_specific()
    test_atomic_json()
    test_task_mode_source_and_replay_contracts()
    test_v6_policy_router_contract()
    test_actor_checkpoint_audit()
    test_full_actor_coupled_observation_projection()
    test_final_checkpoint_cleanup_and_contract_order()
    test_latch_conditioned_noise_scale()
    print("train contract tests passed")


if __name__ == "__main__":
    main()
