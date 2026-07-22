#!/usr/bin/env python3
"""Simulation-free contract tests for the strict FlashSAC evaluator."""

from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import torch

from evaluate import (
    ACTION_DIM,
    CLOSE_OPTION_MODE,
    COUPLED_OBSERVATION_DIM,
    COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    COUPLED_TEACHER_RESIDUAL_ACTION_PROJECTION,
    COUPLED_TEACHER_RESIDUAL_MODE,
    COUPLED_POWER_ARM_TARGET_OFFSET_LIMIT,
    COUPLED_POWER_OBSERVATION_CONTRACT,
    FULL_TASK_MODE,
    FULL115_TO_COUPLED131_OBSERVATION_TRANSFER,
    HAND_ACTION_DIM,
    HAND_ACTION_PROJECTION,
    HAND_NOISE_GROUP_SPECS,
    HAND_POLICY_ACTION_LAYOUT,
    TEACHER_RESIDUAL_POLICY_ACTION_LAYOUT,
    NOISE_GROUP_SPECS,
    POWER_CLOSE_OPTION_MODE,
    POWER_OBSERVATION_CONTRACT,
    PhysicalTruth,
    PRODUCTION_CRITIC_BINS,
    SMOKE_CRITIC_BINS,
    StrictEpisodeTracker,
    _atomic_torch_save,
    _atomic_write_json,
    apply_coupled_controller_ablation,
    apply_public_latch_arm_gate,
    apply_public_latch_hand_hold,
    build_diagnostic_handoff_artifact,
    build_strict_metrics,
    episode_quotas,
    infer_actor_architecture_from_state,
    infer_actor_action_dim_from_state,
    load_checkpoint_teacher_prior,
    physical_truth_from_terminal_info,
    resolve_checkpoint_directory,
    resolve_cross_task_actor_evaluation,
    requested_policy_action_contract,
    select_diagnostic_approach_action,
    task_mode_from_option_flags,
    summarize,
    update_diagnostic_approach_handoff,
    validate_diagnostic_approach_horizon,
    validate_close_option_evaluation_config,
    validate_curriculum_config,
    validate_hierarchical_arm_hold_evaluation_config,
    validate_public_latch_arm_gate_evaluation_config,
    validate_recoverability_selector_evaluation_config,
    validate_arm_hold_handoff_state,
    validate_terminal_events,
    validate_checkpoint_evaluation_contract,
    zero_expand_actor_observation_state,
    run,
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
    full_task_success=None,
    close_success=(False, False),
    close_failure=(False, False),
    close_timeout=(False, False),
    close_unlatched_lift=(False, False),
    close_horizontal_escape=(False, False),
    close_lost_window=(False, False),
    close_stable_steps=(0, 0),
    grasp_quality=(0.0, 0.0),
    hold_quality=(0.0, 0.0),
    max_force=(0.0, 0.0),
    arm_hold_released=(False, False),
    arm_hold_stable_steps=(0, 0),
    arm_hold_release_other_contacts=(-1, -1),
    arm_hold_release_grasp_quality=(0.0, 0.0),
    arm_hold_release_wrap_quality=(0.0, 0.0),
    arm_hold_release_max_force=(0.0, 0.0),
    power_success=(False, False),
    power_failure=(False, False),
    power_timeout=(False, False),
    power_stable_steps=(0, 0),
    power_grasped=(False, False),
    power_thumb_contact=(False, False),
    power_legal_other_contacts=(0, 0),
    power_close_quality=(0.0, 0.0),
    power_wrap_quality=(0.0, 0.0),
    power_grasp_quality=(0.0, 0.0),
    power_latch_confirm_steps=(0, 0),
    coupled_pose_escape=(False, False),
    coupled_rotation_drift=(0.0, 0.0),
    coupled_xy_drift=(0.0, 0.0),
    coupled_true_clearance=None,
    coupled_arm_target_offset=(0.0, 0.0),
    coupled_arm_target_saturated=(False, False),
    coupled_align_active=(False, False),
) -> dict[str, torch.Tensor]:
    if full_task_success is None:
        full_task_success = success
    if coupled_true_clearance is None:
        coupled_true_clearance = clearance
    return {
        "success": torch.tensor(success, dtype=torch.bool),
        "failure": torch.tensor(failure, dtype=torch.bool),
        "time_out": torch.tensor(time_out, dtype=torch.bool),
        "dropped": torch.tensor(dropped, dtype=torch.bool),
        "unsafe_force": torch.tensor(unsafe, dtype=torch.bool),
        "unlatched_clearance_ge_5cm": torch.tensor(unlatched, dtype=torch.bool),
        "true_clearance": torch.tensor(clearance, dtype=torch.float32),
        "is_grasped": torch.tensor(grasped, dtype=torch.bool),
        "full_task_success": torch.tensor(full_task_success, dtype=torch.bool),
        "close_option_success": torch.tensor(close_success, dtype=torch.bool),
        "close_option_failure": torch.tensor(close_failure, dtype=torch.bool),
        "close_option_timeout": torch.tensor(close_timeout, dtype=torch.bool),
        "close_option_unlatched_lift": torch.tensor(
            close_unlatched_lift, dtype=torch.bool
        ),
        "close_option_horizontal_escape": torch.tensor(
            close_horizontal_escape, dtype=torch.bool
        ),
        "close_option_lost_window": torch.tensor(close_lost_window, dtype=torch.bool),
        "close_option_stable_steps": torch.tensor(close_stable_steps, dtype=torch.long),
        "grasp_quality": torch.tensor(grasp_quality, dtype=torch.float32),
        "hold_quality": torch.tensor(hold_quality, dtype=torch.float32),
        "max_force": torch.tensor(max_force, dtype=torch.float32),
        "arm_hold_released": torch.tensor(arm_hold_released, dtype=torch.bool),
        "arm_hold_stable_steps": torch.tensor(
            arm_hold_stable_steps, dtype=torch.long
        ),
        "arm_hold_release_other_contacts": torch.tensor(
            arm_hold_release_other_contacts, dtype=torch.long
        ),
        "arm_hold_release_grasp_quality": torch.tensor(
            arm_hold_release_grasp_quality, dtype=torch.float32
        ),
        "arm_hold_release_wrap_quality": torch.tensor(
            arm_hold_release_wrap_quality, dtype=torch.float32
        ),
        "arm_hold_release_max_force": torch.tensor(
            arm_hold_release_max_force, dtype=torch.float32
        ),
        "power_close_option_success": torch.tensor(power_success, dtype=torch.bool),
        "power_close_option_failure": torch.tensor(power_failure, dtype=torch.bool),
        "power_close_option_timeout": torch.tensor(power_timeout, dtype=torch.bool),
        "power_close_option_stable_steps": torch.tensor(
            power_stable_steps, dtype=torch.long
        ),
        "power_is_grasped": torch.tensor(power_grasped, dtype=torch.bool),
        "power_thumb_contact": torch.tensor(power_thumb_contact, dtype=torch.bool),
        "power_legal_other_contact_count": torch.tensor(
            power_legal_other_contacts, dtype=torch.long
        ),
        "power_close_quality": torch.tensor(power_close_quality, dtype=torch.float32),
        "power_wrap_quality": torch.tensor(power_wrap_quality, dtype=torch.float32),
        "power_grasp_quality": torch.tensor(power_grasp_quality, dtype=torch.float32),
        "power_grasp_latch_confirm_steps": torch.tensor(
            power_latch_confirm_steps, dtype=torch.long
        ),
        "coupled_power_pose_escape": torch.tensor(
            coupled_pose_escape, dtype=torch.bool
        ),
        "coupled_power_rotation_drift": torch.tensor(
            coupled_rotation_drift, dtype=torch.float32
        ),
        "coupled_power_xy_drift": torch.tensor(
            coupled_xy_drift, dtype=torch.float32
        ),
        "coupled_power_true_clearance": torch.tensor(
            coupled_true_clearance, dtype=torch.float32
        ),
        "coupled_power_arm_target_offset_abs_max": torch.tensor(
            coupled_arm_target_offset, dtype=torch.float32
        ),
        "coupled_power_arm_target_saturated": torch.tensor(
            coupled_arm_target_saturated, dtype=torch.bool
        ),
        "coupled_power_align_active": torch.tensor(
            coupled_align_active, dtype=torch.bool
        ),
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
    unlatched_failure = _events(
        failure=(True, False),
        unlatched=(True, False),
        clearance=(0.05, -0.001),
    )
    validate_terminal_events(
        {"pick_tool_terminal": unlatched_failure},
        torch.tensor([True, False]),
        torch.tensor([False, False]),
    )
    terminal_timeout_overlap = _events(
        success=(True, False), time_out=(True, False), grasped=(True, False)
    )
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": terminal_timeout_overlap},
        torch.tensor([True, False]),
        torch.tensor([True, False]),
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


def test_hierarchical_arm_hold_terminal_contract() -> None:
    raw = _events(
        time_out=(True, True),
        arm_hold_released=(True, False),
        arm_hold_stable_steps=(15, 4),
        arm_hold_release_other_contacts=(2, -1),
        arm_hold_release_grasp_quality=(0.4, 0.0),
        arm_hold_release_wrap_quality=(0.5, 0.0),
        arm_hold_release_max_force=(10.0, 0.0),
    )
    state = validate_arm_hold_handoff_state(
        {"pick_tool_terminal": raw},
        num_envs=2,
        device=torch.device("cpu"),
    )
    assert state["arm_hold_released"].tolist() == [True, False]
    assert state["arm_hold_stable_steps"].tolist() == [15, 4]
    assert state["arm_hold_release_other_contacts"].tolist() == [2, -1]

    invalid = dict(raw)
    invalid["arm_hold_stable_steps"] = torch.tensor((14, 4), dtype=torch.long)
    _expect_error(
        RuntimeError,
        validate_arm_hold_handoff_state,
        {"pick_tool_terminal": invalid},
        num_envs=2,
        device=torch.device("cpu"),
    )

    tracker = StrictEpisodeTracker(
        episodes=2,
        num_envs=2,
        device=torch.device("cpu"),
        initial_truth=_truth((-0.001, -0.001), (False, False)),
        track_arm_hold_handoff=True,
    )
    events = validate_terminal_events(
        {"pick_tool_terminal": raw},
        torch.tensor((False, False)),
        torch.tensor((True, True)),
    )
    events.update(state)
    tracker.step(
        reward=torch.zeros(2),
        terminated=torch.tensor((False, False)),
        truncated=torch.tensor((True, True)),
        events=events,
        transition_truth=_truth((0.21, 0.0), (True, False)),
        post_reset_truth=_truth((-0.001, -0.001), (False, False)),
    )
    assert tracker.records[0]["arm_hold_released"] is True
    assert tracker.records[0]["max_arm_hold_stable_steps"] == 15
    assert tracker.records[0]["arm_hold_release_step"] == 1
    assert tracker.records[0]["arm_hold_release_other_contacts"] == 2
    assert abs(tracker.records[0]["arm_hold_release_grasp_quality"] - 0.4) < 1.0e-6
    assert tracker.records[1]["arm_hold_released"] is False
    assert tracker.records[1]["arm_hold_release_step"] == -1

def test_close_option_terminal_event_contract() -> None:
    terminal = _events(
        success=(True, False),
        time_out=(False, True),
        full_task_success=(False, False),
        close_success=(True, False),
        close_timeout=(False, True),
        close_stable_steps=(15, 0),
    )
    actual = validate_terminal_events(
        {"pick_tool_terminal": terminal},
        torch.tensor([True, False]),
        torch.tensor([False, True]),
        task_mode=CLOSE_OPTION_MODE,
    )

    unsafe_success = _events(
        success=(True, False),
        dropped=(True, False),
        full_task_success=(False, False),
        close_success=(True, False),
        close_stable_steps=(15, 0),
    )
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": unsafe_success},
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        task_mode=CLOSE_OPTION_MODE,
    )
    assert actual["close_option_success"].tolist() == [True, False]
    assert actual["full_task_success"].tolist() == [False, False]

    # The option fails pre-latch at 1.5 cm, well before the full task's 5 cm
    # anti-fling boundary.  This is a valid close failure even though the legacy
    # unlatched-5-cm diagnostic remains false.
    early_lift = _events(
        failure=(True, False),
        close_failure=(True, False),
        close_unlatched_lift=(True, False),
        clearance=(0.016, -0.001),
    )
    validate_terminal_events(
        {"pick_tool_terminal": early_lift},
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        task_mode=CLOSE_OPTION_MODE,
    )

    horizontal_escape = _events(
        failure=(True, False),
        close_failure=(True, False),
        close_horizontal_escape=(True, False),
    )
    validate_terminal_events(
        {"pick_tool_terminal": horizontal_escape},
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        task_mode=CLOSE_OPTION_MODE,
    )

    premature = _events(
        success=(True, False),
        close_success=(True, False),
        close_stable_steps=(14, 0),
        full_task_success=(False, False),
    )
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": premature},
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        task_mode=CLOSE_OPTION_MODE,
    )

    inconsistent_alias = dict(terminal)
    inconsistent_alias["close_option_success"] = torch.tensor([False, False])
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": inconsistent_alias},
        torch.tensor([True, False]),
        torch.tensor([False, True]),
        task_mode=CLOSE_OPTION_MODE,
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


def test_close_option_physical_truth_is_not_20cm_success() -> None:
    close_success = _events(
        success=(True, False),
        full_task_success=(False, False),
        close_success=(True, False),
        close_stable_steps=(15, 0),
        clearance=(0.004, -0.001),
        grasped=(True, False),
        grasp_quality=(0.40, 0.0),
        hold_quality=(0.60, 0.0),
        max_force=(12.0, 0.0),
    )
    truth = physical_truth_from_terminal_info(
        {"pick_tool_terminal": close_success},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=CLOSE_OPTION_MODE,
    )
    assert truth.clearance[0] < 0.20

    # The identical generic success alias is invalid under full-task semantics;
    # the explicitly named full_task_success remains authoritative.
    _expect_error(
        RuntimeError,
        physical_truth_from_terminal_info,
        {"pick_tool_terminal": close_success},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=FULL_TASK_MODE,
    )

    no_latch = dict(close_success)
    no_latch["is_grasped"] = torch.tensor([False, False])
    _expect_error(
        RuntimeError,
        physical_truth_from_terminal_info,
        {"pick_tool_terminal": no_latch},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=CLOSE_OPTION_MODE,
    )

    for key, replacement in (
        ("grasp_quality", torch.tensor([0.34, 0.0])),
        ("hold_quality", torch.tensor([0.49, 0.0])),
        ("max_force", torch.tensor([30.01, 0.0])),
    ):
        invalid_quality = dict(close_success)
        invalid_quality[key] = replacement
        _expect_error(
            RuntimeError,
            physical_truth_from_terminal_info,
            {"pick_tool_terminal": invalid_quality},
            num_envs=2,
            device=torch.device("cpu"),
            task_mode=CLOSE_OPTION_MODE,
        )

    dropped_success = dict(close_success)
    dropped_success["dropped"] = torch.tensor([True, False])
    _expect_error(
        RuntimeError,
        physical_truth_from_terminal_info,
        {"pick_tool_terminal": dropped_success},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=CLOSE_OPTION_MODE,
    )

    false_full_success = dict(close_success)
    false_full_success["full_task_success"] = torch.tensor([True, False])
    _expect_error(
        RuntimeError,
        physical_truth_from_terminal_info,
        {"pick_tool_terminal": false_full_success},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=CLOSE_OPTION_MODE,
    )


def _valid_power_events() -> dict[str, torch.Tensor]:
    return _events(
        success=(True, False),
        failure=(False, True),
        full_task_success=(False, False),
        close_success=(True, False),
        close_failure=(False, True),
        close_horizontal_escape=(False, True),
        close_stable_steps=(15, 0),
        power_success=(True, False),
        power_failure=(False, True),
        power_timeout=(False, False),
        power_stable_steps=(15, 0),
        clearance=(0.003, 0.002),
        grasped=(True, False),
        hold_quality=(0.60, 0.0),
        max_force=(12.0, 0.0),
        power_grasped=(True, False),
        power_thumb_contact=(True, False),
        power_legal_other_contacts=(3, 0),
        power_close_quality=(0.85, 0.0),
        power_wrap_quality=(0.45, 0.0),
        power_grasp_quality=(0.40, 0.0),
        power_latch_confirm_steps=(18, 0),
    )


def test_power_close_terminal_and_physical_contract() -> None:
    raw = _valid_power_events()
    events = validate_terminal_events(
        {"pick_tool_terminal": raw},
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=POWER_CLOSE_OPTION_MODE,
    )
    assert events["power_close_option_success"].tolist() == [True, False]
    assert events["power_legal_other_contact_count"].tolist() == [3, 0]
    truth = physical_truth_from_terminal_info(
        {"pick_tool_terminal": raw},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=POWER_CLOSE_OPTION_MODE,
    )
    assert truth.grasped.tolist() == [True, False]
    assert truth.clearance[0] < 0.20

    for key, replacement, error_type in (
        ("max_force", torch.tensor([-0.01, 0.0]), RuntimeError),
        ("hold_quality", torch.tensor([1.01, 0.0]), RuntimeError),
        ("max_force", torch.tensor([float("nan"), 0.0]), FloatingPointError),
    ):
        corrupt = _events()
        corrupt[key] = replacement
        _expect_error(
            error_type,
            validate_terminal_events,
            {"pick_tool_terminal": corrupt},
            torch.tensor([False, False]),
            torch.tensor([False, False]),
            task_mode=POWER_CLOSE_OPTION_MODE,
        )
        _expect_error(
            error_type,
            physical_truth_from_terminal_info,
            {"pick_tool_terminal": corrupt},
            num_envs=2,
            device=torch.device("cpu"),
            task_mode=POWER_CLOSE_OPTION_MODE,
        )

    invalid_replacements = (
        ("success", torch.tensor([False, False], dtype=torch.bool)),
        ("power_thumb_contact", torch.tensor([False, False], dtype=torch.bool)),
        ("power_legal_other_contact_count", torch.tensor([2, 0], dtype=torch.long)),
        ("power_grasp_quality", torch.tensor([0.34, 0.0], dtype=torch.float32)),
        ("power_close_option_stable_steps", torch.tensor([14, 0], dtype=torch.long)),
        ("hold_quality", torch.tensor([0.49, 0.0], dtype=torch.float32)),
        ("max_force", torch.tensor([30.01, 0.0], dtype=torch.float32)),
        ("unsafe_force", torch.tensor([True, False], dtype=torch.bool)),
    )
    for key, replacement in invalid_replacements:
        invalid = dict(raw)
        invalid[key] = replacement
        _expect_error(
            RuntimeError,
            validate_terminal_events,
            {"pick_tool_terminal": invalid},
            torch.tensor([True, True]),
            torch.tensor([False, False]),
            task_mode=POWER_CLOSE_OPTION_MODE,
        )

    premature = dict(raw)
    premature["close_option_stable_steps"] = torch.tensor(
        [14, 0], dtype=torch.long
    )
    premature["power_close_option_stable_steps"] = torch.tensor(
        [14, 0], dtype=torch.long
    )
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": premature},
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=POWER_CLOSE_OPTION_MODE,
    )

    invalid_truth = dict(raw)
    invalid_truth["power_is_grasped"] = torch.tensor([False, False])
    _expect_error(
        RuntimeError,
        physical_truth_from_terminal_info,
        {"pick_tool_terminal": invalid_truth},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=POWER_CLOSE_OPTION_MODE,
    )


def _valid_coupled_power_events() -> dict[str, torch.Tensor]:
    raw = _valid_power_events()
    raw.update(
        {
            "coupled_power_pose_escape": torch.tensor(
                [False, True], dtype=torch.bool
            ),
            "coupled_power_rotation_drift": torch.tensor(
                [0.10, 0.36], dtype=torch.float32
            ),
            "coupled_power_xy_drift": torch.tensor(
                [0.01, 0.04], dtype=torch.float32
            ),
            "coupled_power_true_clearance": torch.tensor(
                [0.003, 0.002], dtype=torch.float32
            ),
            "coupled_power_arm_target_offset_abs_max": torch.tensor(
                [0.119, 0.12], dtype=torch.float32
            ),
            "coupled_power_arm_target_saturated": torch.tensor(
                [False, True], dtype=torch.bool
            ),
            "coupled_power_align_active": torch.tensor(
                [False, False], dtype=torch.bool
            ),
        }
    )
    return raw


def test_coupled_power_terminal_physical_and_metrics_contract() -> None:
    raw = _valid_coupled_power_events()
    info = {"pick_tool_terminal": raw}
    events = validate_terminal_events(
        info,
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    assert events["power_close_option_success"].tolist() == [True, False]
    assert events["coupled_power_pose_escape"].tolist() == [False, True]
    truth = physical_truth_from_terminal_info(
        info,
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    assert truth.grasped.tolist() == [True, False]

    for missing_key in (
        "coupled_power_pose_escape",
        "coupled_power_rotation_drift",
        "coupled_power_xy_drift",
        "coupled_power_true_clearance",
        "coupled_power_arm_target_offset_abs_max",
        "coupled_power_arm_target_saturated",
        "coupled_power_align_active",
    ):
        missing = dict(raw)
        del missing[missing_key]
        _expect_error(
            TypeError,
            validate_terminal_events,
            {"pick_tool_terminal": missing},
            torch.tensor([True, True]),
            torch.tensor([False, False]),
            task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        )

    escaped_success = dict(raw)
    escaped_success["coupled_power_pose_escape"] = torch.tensor([True, True])
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": escaped_success},
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    hidden_rotation_escape = dict(raw)
    hidden_rotation_escape["coupled_power_rotation_drift"] = torch.tensor(
        [0.50, 0.36], dtype=torch.float32
    )
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": hidden_rotation_escape},
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    active_success = dict(raw)
    active_success["coupled_power_align_active"] = torch.tensor(
        [True, False], dtype=torch.bool
    )
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": active_success},
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    saturated_success = dict(raw)
    saturated_success["coupled_power_arm_target_saturated"] = torch.tensor(
        [True, True], dtype=torch.bool
    )
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": saturated_success},
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    _expect_error(
        RuntimeError,
        physical_truth_from_terminal_info,
        {"pick_tool_terminal": saturated_success},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    _expect_error(
        RuntimeError,
        physical_truth_from_terminal_info,
        {"pick_tool_terminal": active_success},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    excessive_arm_offset = dict(raw)
    excessive_arm_offset["coupled_power_arm_target_offset_abs_max"] = torch.tensor(
        [COUPLED_POWER_ARM_TARGET_OFFSET_LIMIT + 2.0e-6, 0.12],
        dtype=torch.float32,
    )
    _expect_error(
        RuntimeError,
        physical_truth_from_terminal_info,
        {"pick_tool_terminal": excessive_arm_offset},
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    inconsistent_clearance = dict(raw)
    inconsistent_clearance["coupled_power_true_clearance"] = torch.tensor(
        [0.004, 0.002], dtype=torch.float32
    )
    _expect_error(
        RuntimeError,
        validate_terminal_events,
        {"pick_tool_terminal": inconsistent_clearance},
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )

    tracker = StrictEpisodeTracker(
        episodes=2,
        num_envs=2,
        device=torch.device("cpu"),
        initial_truth=_truth((-0.001, -0.001), (False, False)),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    # A transient 31 N excursion is below the task's unsafe-force termination
    # in some configurations, but exceeds the conservative teacher contract.
    progress = validate_terminal_events(
        {"pick_tool_terminal": _events(max_force=(31.0, 5.0))},
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
    )
    tracker.step(
        reward=torch.zeros(2),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, False]),
        events=progress,
        transition_truth=_truth((0.0, 0.0), (False, False)),
        post_reset_truth=_truth((0.0, 0.0), (False, False)),
    )
    tracker.step(
        reward=torch.tensor([100.0, -100.0]),
        terminated=torch.tensor([True, True]),
        truncated=torch.tensor([False, False]),
        events=events,
        transition_truth=truth,
        post_reset_truth=_truth((-0.001, -0.001), (False, False)),
    )
    assert tracker.records[0]["terminal_coupled_power_pose_escape"] is False
    assert tracker.records[1]["ever_coupled_power_pose_escape"] is True
    assert tracker.records[0]["episode_max_power_force_n"] == 31.0
    policy = requested_policy_action_contract(
        COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE
    )
    metrics = build_strict_metrics(
        tracker.records,
        checkpoint=Path("/tmp/coupled-close-checkpoint"),
        architecture="production",
        seed=17,
        num_envs=2,
        vector_steps=1,
        max_vector_steps=10,
        episode_length_s=3.0,
        max_episode_steps=150,
        curriculum_dataset=Path("/tmp/coupled-close.pt"),
        curriculum_dataset_sha256="test-coupled-close-dataset-sha256",
        curriculum_boundary="close_start",
        curriculum_probability=1.0,
        curriculum_joint_noise=0.0,
        use_compile=False,
        upstream_commit="test-commit",
        task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        checkpoint_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        policy_action_dim=policy["policy_action_dim"],
        environment_action_dim=policy["environment_action_dim"],
        policy_action_layout=policy["policy_action_layout"],
        action_projection=policy["action_projection"],
        observation_dim=policy["observation_dim"],
        observation_contract=policy["observation_contract"],
    )
    assert metrics["coupled_power_align_close_option_success_rate"] == 0.5
    assert metrics["conservative_teacher_compatible_success_rate"] == 0.0
    assert metrics["conservative_teacher_compatible_success_count"] == 0
    assert "strict_success_rate" not in metrics
    assert metrics["observation_dim"] == COUPLED_OBSERVATION_DIM
    assert metrics["observation_contract"] == COUPLED_POWER_OBSERVATION_CONTRACT
    assert metrics["action_dim"] == ACTION_DIM
    assert metrics["action_projection"] == "identity_v1"
    assert metrics["success_contract"]["full_task_20cm_success"] == "not_evaluated"
    assert metrics["success_contract"]["pose_escape_required_false"] is True
    assert (
        metrics["success_contract"][
            "arm_target_saturation_ever_required_false"
        ]
        is True
    )
    assert metrics["success_contract"]["max_arm_target_offset_rad"] == 0.12
    assert "close_option_only" in metrics["evaluation_scope"]
    coupled_telemetry = metrics["coupled_power_telemetry"]
    assert coupled_telemetry["episodes_ever_pose_escape"] == 1
    assert coupled_telemetry["episodes_ever_arm_target_saturated"] == 1
    assert coupled_telemetry["episode_max_rotation_drift_rad"]["max"] > 0.35
    assert metrics["power_close_telemetry"]["episode_max_force_n"]["max"] == 31.0
    assert metrics["episodes"][0]["conservative_teacher_compatible_success"] is False


def test_teacher_residual_strict_contract_and_metrics() -> None:
    raw = _valid_coupled_power_events()
    info = {"pick_tool_terminal": raw}
    events = validate_terminal_events(
        info,
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=COUPLED_TEACHER_RESIDUAL_MODE,
    )
    truth = physical_truth_from_terminal_info(
        info,
        num_envs=2,
        device=torch.device("cpu"),
        task_mode=COUPLED_TEACHER_RESIDUAL_MODE,
    )
    tracker = StrictEpisodeTracker(
        episodes=2,
        num_envs=2,
        device=torch.device("cpu"),
        initial_truth=_truth((-0.001, -0.001), (False, False)),
        task_mode=COUPLED_TEACHER_RESIDUAL_MODE,
    )
    tracker.step(
        reward=torch.tensor([100.0, -100.0]),
        terminated=torch.tensor([True, True]),
        truncated=torch.tensor([False, False]),
        events=events,
        transition_truth=truth,
        post_reset_truth=_truth((-0.001, -0.001), (False, False)),
    )
    policy = requested_policy_action_contract(COUPLED_TEACHER_RESIDUAL_MODE)
    metrics = build_strict_metrics(
        tracker.records,
        checkpoint=Path("/tmp/residual-checkpoint"),
        architecture="production",
        seed=19,
        num_envs=2,
        vector_steps=1,
        max_vector_steps=10,
        episode_length_s=3.0,
        max_episode_steps=150,
        curriculum_dataset=Path("/tmp/coupled-close.pt"),
        curriculum_dataset_sha256="test-coupled-close-dataset-sha256",
        curriculum_boundary="close_start",
        curriculum_probability=1.0,
        curriculum_joint_noise=0.0,
        use_compile=False,
        upstream_commit="test-commit",
        task_mode=COUPLED_TEACHER_RESIDUAL_MODE,
        checkpoint_task_mode=COUPLED_TEACHER_RESIDUAL_MODE,
        policy_action_dim=policy["policy_action_dim"],
        environment_action_dim=policy["environment_action_dim"],
        policy_action_layout=policy["policy_action_layout"],
        action_projection=policy["action_projection"],
        observation_dim=policy["observation_dim"],
        observation_contract=policy["observation_contract"],
        noise_group_specs=HAND_NOISE_GROUP_SPECS,
    )
    assert metrics["coupled_teacher_residual_close_option_success_rate"] == 0.5
    assert metrics["conservative_teacher_compatible_success_rate"] == 0.5
    assert metrics["policy_action_dim"] == HAND_ACTION_DIM
    assert metrics["environment_action_dim"] == ACTION_DIM
    assert metrics["policy_action_layout"] == TEACHER_RESIDUAL_POLICY_ACTION_LAYOUT
    assert metrics["action_projection"] == COUPLED_TEACHER_RESIDUAL_ACTION_PROJECTION
    assert metrics["observation_dim"] == COUPLED_OBSERVATION_DIM
    assert metrics["coupled_power_telemetry"]["episodes_ever_pose_escape"] == 1


def test_embedded_teacher_prior_sha_and_curriculum_lineage() -> None:
    from coupled_teacher_prior import CoupledTeacherPrior

    with tempfile.TemporaryDirectory(prefix="flashsac_residual_eval_") as directory:
        checkpoint = Path(directory)
        curriculum = checkpoint / "curriculum.pt"
        curriculum.write_bytes(b"exact curriculum bytes")
        curriculum_sha = hashlib.sha256(curriculum.read_bytes()).hexdigest()
        prior = CoupledTeacherPrior(
            arm_delta_target_rad=(0.0,) * 7,
            hand_latent=(0.0,) * 14,
            teacher_artifact_sha256="a" * 64,
            curriculum_dataset_sha256=curriculum_sha,
            teacher_artifact_path="<test>",
            curriculum_dataset_path="<test>",
        )
        prior_path = checkpoint / "teacher_action_prior.json"
        prior_path.write_text(
            json.dumps(prior.contract_payload(), sort_keys=True),
            encoding="utf-8",
        )
        prior_sha = hashlib.sha256(prior_path.read_bytes()).hexdigest()
        contract = {
            "task_mode": COUPLED_TEACHER_RESIDUAL_MODE,
            "teacher_action_prior": {
                "filename": "teacher_action_prior.json",
                "sha256": prior_sha,
            },
        }
        loaded, loaded_sha = load_checkpoint_teacher_prior(
            checkpoint=checkpoint,
            checkpoint_contract=contract,
            curriculum_dataset=curriculum,
        )
        assert loaded.contract_payload() == prior.contract_payload()
        assert loaded_sha == prior_sha

        prior_path.write_text("{}", encoding="utf-8")
        _expect_error(
            ValueError,
            load_checkpoint_teacher_prior,
            checkpoint=checkpoint,
            checkpoint_contract=contract,
            curriculum_dataset=curriculum,
        )
        prior_path.write_text(
            json.dumps(prior.contract_payload(), sort_keys=True),
            encoding="utf-8",
        )
        curriculum.write_bytes(b"different curriculum bytes")
        _expect_error(
            ValueError,
            load_checkpoint_teacher_prior,
            checkpoint=checkpoint,
            checkpoint_contract=contract,
            curriculum_dataset=curriculum,
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


def test_close_option_tracker_and_metrics_are_separate_from_full_success() -> None:
    tracker = StrictEpisodeTracker(
        episodes=2,
        num_envs=2,
        device=torch.device("cpu"),
        initial_truth=_truth((-0.001, -0.001), (False, False)),
        task_mode=CLOSE_OPTION_MODE,
    )
    raw = _events(
        success=(True, False),
        failure=(False, True),
        full_task_success=(False, False),
        close_success=(True, False),
        close_failure=(False, True),
        close_horizontal_escape=(False, True),
        close_stable_steps=(15, 0),
        clearance=(0.003, 0.002),
        grasped=(True, False),
        grasp_quality=(0.40, 0.0),
        hold_quality=(0.60, 0.0),
        max_force=(10.0, 0.0),
    )
    events = validate_terminal_events(
        {"pick_tool_terminal": raw},
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=CLOSE_OPTION_MODE,
    )
    tracker.step(
        reward=torch.tensor([100.0, -100.0]),
        terminated=torch.tensor([True, True]),
        truncated=torch.tensor([False, False]),
        events=events,
        transition_truth=_truth((0.003, 0.002), (True, False)),
        post_reset_truth=_truth((-0.001, -0.001), (False, False)),
    )
    assert tracker.complete
    assert tracker.records[0]["close_option_success"] is True
    assert "success" not in tracker.records[0]
    assert tracker.records[1]["close_option_horizontal_escape"] is True

    timeout_tracker = StrictEpisodeTracker(
        episodes=1,
        num_envs=2,
        device=torch.device("cpu"),
        initial_truth=_truth((-0.001, -0.001), (False, False)),
        task_mode=CLOSE_OPTION_MODE,
    )
    timeout_raw = _events(
        time_out=(True, False),
        close_timeout=(True, False),
    )
    timeout_events = validate_terminal_events(
        {"pick_tool_terminal": timeout_raw},
        torch.tensor([False, False]),
        torch.tensor([True, False]),
        task_mode=CLOSE_OPTION_MODE,
    )
    timeout_tracker.step(
        reward=torch.zeros(2),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([True, False]),
        events=timeout_events,
        transition_truth=_truth((-0.001, -0.001), (False, False)),
        post_reset_truth=_truth((-0.001, -0.001), (False, False)),
    )
    assert timeout_tracker.records[0]["close_option_timeout"] is True
    assert "time_out" not in timeout_tracker.records[0]

    with tempfile.TemporaryDirectory(prefix="flashsac_close_eval_") as directory:
        close_dataset = Path(directory) / "close.pt"
        close_dataset.touch()
        validate_close_option_evaluation_config(
            close_option_mode=True,
            curriculum_dataset=close_dataset,
            curriculum_boundary="close_start",
            curriculum_probability=1.0,
            curriculum_joint_noise=0.0,
            episode_length_s=3.0,
        )
        metrics = build_strict_metrics(
            tracker.records,
            checkpoint=Path("/tmp/close-checkpoint"),
            architecture="production",
            seed=11,
            num_envs=2,
            vector_steps=1,
            max_vector_steps=10,
            episode_length_s=3.0,
            max_episode_steps=150,
            curriculum_dataset=close_dataset,
            curriculum_dataset_sha256="test-close-dataset-sha256",
            curriculum_boundary="close_start",
            curriculum_probability=1.0,
            curriculum_joint_noise=0.0,
            use_compile=False,
            upstream_commit="test-commit",
            task_mode=CLOSE_OPTION_MODE,
            checkpoint_task_mode=CLOSE_OPTION_MODE,
        )
    assert metrics["task_mode"] == CLOSE_OPTION_MODE
    assert metrics["close_option_success_rate"] == 0.5
    assert "strict_success_rate" not in metrics
    assert metrics["success_contract"]["full_task_20cm_success"] == "not_evaluated"
    assert metrics["events"] == {
        "close_option_success": 1,
        "close_option_failure": 1,
        "close_option_timeout": 0,
        "dropped": 0,
        "unsafe_force": 0,
        "close_option_unlatched_lift": 0,
        "close_option_horizontal_escape": 1,
        "close_option_lost_window": 0,
    }
    assert metrics["success_contract"]["min_grasp_quality"] == 0.35
    assert metrics["failure_contract"]["unlatched_lift_limit_m"] == 0.015
    assert all("success" not in episode for episode in metrics["episodes"])


def test_power_close_tracker_metrics_and_hand_action_contract() -> None:
    tracker = StrictEpisodeTracker(
        episodes=2,
        num_envs=2,
        device=torch.device("cpu"),
        initial_truth=_truth((-0.001, -0.001), (False, False)),
        task_mode=POWER_CLOSE_OPTION_MODE,
    )
    progress_raw = _events(
        close_stable_steps=(5, 0),
        power_stable_steps=(5, 0),
        hold_quality=(0.75, 0.20),
        power_grasped=(True, False),
        power_thumb_contact=(True, True),
        power_legal_other_contacts=(4, 2),
        power_close_quality=(0.95, 0.30),
        power_wrap_quality=(0.65, 0.10),
        power_grasp_quality=(0.55, 0.05),
        power_latch_confirm_steps=(7, 1),
    )
    progress_events = validate_terminal_events(
        {"pick_tool_terminal": progress_raw},
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        task_mode=POWER_CLOSE_OPTION_MODE,
    )
    tracker.step(
        reward=torch.tensor([1.0, 2.0]),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, False]),
        events=progress_events,
        transition_truth=_truth((0.001, 0.001), (True, False)),
        post_reset_truth=_truth((0.001, 0.001), (True, False)),
    )
    raw = _valid_power_events()
    events = validate_terminal_events(
        {"pick_tool_terminal": raw},
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        task_mode=POWER_CLOSE_OPTION_MODE,
    )
    tracker.step(
        reward=torch.tensor([80.0, -80.0]),
        terminated=torch.tensor([True, True]),
        truncated=torch.tensor([False, False]),
        events=events,
        transition_truth=_truth((0.003, 0.002), (True, False)),
        post_reset_truth=_truth((-0.001, -0.001), (False, False)),
    )
    assert tracker.complete
    assert tracker.records[0]["power_close_option_success"] is True
    assert tracker.records[0]["power_thumb_contact"] is True
    assert tracker.records[0]["power_legal_other_contact_count"] == 3
    assert tracker.records[0]["terminal_power_legal_other_contact_count"] == 3
    assert tracker.records[0]["max_power_legal_other_contact_count"] == 4
    assert tracker.records[0]["ever_power_thumb_plus_three"] is True
    assert tracker.records[0]["ever_power_grasp_latched"] is True
    assert abs(tracker.records[0]["max_power_staged_close_quality"] - 0.95) < 1.0e-6
    assert abs(tracker.records[0]["max_power_wrap_quality"] - 0.65) < 1.0e-6
    assert abs(tracker.records[0]["max_power_grasp_quality"] - 0.55) < 1.0e-6
    assert tracker.records[0]["max_power_close_option_stable_steps"] == 15
    assert tracker.records[1]["max_power_legal_other_contact_count"] == 2
    assert tracker.records[1]["ever_power_thumb_plus_three"] is False
    assert "success" not in tracker.records[0]
    assert "close_option_success" not in tracker.records[0]

    policy = requested_policy_action_contract(POWER_CLOSE_OPTION_MODE)
    metrics = build_strict_metrics(
        tracker.records,
        checkpoint=Path("/tmp/power-close-checkpoint"),
        architecture="production",
        seed=13,
        num_envs=2,
        vector_steps=1,
        max_vector_steps=10,
        episode_length_s=0.40,
        max_episode_steps=20,
        curriculum_dataset=Path("/tmp/power-close.pt"),
        curriculum_dataset_sha256="test-power-close-dataset-sha256",
        curriculum_boundary="close_start",
        curriculum_probability=1.0,
        curriculum_joint_noise=0.0,
        use_compile=False,
        upstream_commit="test-commit",
        task_mode=POWER_CLOSE_OPTION_MODE,
        checkpoint_task_mode=POWER_CLOSE_OPTION_MODE,
        policy_action_dim=policy["policy_action_dim"],
        environment_action_dim=policy["environment_action_dim"],
        policy_action_layout=policy["policy_action_layout"],
        action_projection=policy["action_projection"],
        observation_contract=policy["observation_contract"],
        noise_group_specs=HAND_NOISE_GROUP_SPECS,
    )
    assert metrics["task_mode"] == POWER_CLOSE_OPTION_MODE
    assert metrics["power_close_option_success_rate"] == 0.5
    assert "strict_success_rate" not in metrics
    assert "close_option_success_rate" not in metrics
    assert metrics["action_dim"] == HAND_ACTION_DIM
    assert metrics["policy_action_dim"] == HAND_ACTION_DIM
    assert metrics["environment_action_dim"] == ACTION_DIM
    assert metrics["policy_action_layout"] == HAND_POLICY_ACTION_LAYOUT
    assert metrics["action_projection"] == HAND_ACTION_PROJECTION
    assert metrics["observation_contract"] == POWER_OBSERVATION_CONTRACT
    assert metrics["noise_groups"] == [
        {
            "name": name,
            "start": start,
            "stop": stop,
            "scale": scale,
            "zeta_mu": zeta_mu,
            "zeta_max": zeta_max,
        }
        for name, start, stop, scale, zeta_mu, zeta_max in HAND_NOISE_GROUP_SPECS
    ]
    assert metrics["events"] == {
        "power_close_option_success": 1,
        "power_close_option_failure": 1,
        "power_close_option_timeout": 0,
        "dropped": 0,
        "unsafe_force": 0,
        "close_option_unlatched_lift": 0,
        "close_option_horizontal_escape": 1,
        "close_option_lost_window": 0,
    }
    telemetry = metrics["power_close_telemetry"]
    assert telemetry["episodes_ever_thumb_contact"] == 2
    assert telemetry["episodes_ever_thumb_plus_three"] == 1
    assert telemetry["episodes_ever_grasp_latched"] == 1
    assert telemetry["episode_max_legal_other_contact_count"]["max"] == 4.0
    assert abs(telemetry["episode_max_staged_close_quality"]["max"] - 0.95) < 1.0e-6
    assert abs(telemetry["episode_max_wrap_quality"]["max"] - 0.65) < 1.0e-6
    assert abs(telemetry["episode_max_grasp_quality"]["max"] - 0.55) < 1.0e-6
    assert telemetry["episode_max_close_option_stable_steps"]["max"] == 15.0
    assert telemetry["episode_max_force_n"]["max"] == 12.0
    assert metrics["success_contract"] == {
        "name": "stable_power_close_option_latch_v1",
        "confirm_steps": 15,
        "thumb_contact_required": True,
        "required_legal_other_contacts": 3,
        "min_power_grasp_quality": 0.35,
        "min_hold_quality": 0.5,
        "safe_force_limit_n": 30.0,
        "full_task_20cm_success": "not_evaluated",
    }


def test_coupled_controller_ablation_composition() -> None:
    actor = torch.arange(42, dtype=torch.float32).reshape(2, 21)
    teacher = -actor - 1.0
    exact = apply_coupled_controller_ablation(actor, teacher, "exact_teacher")
    assert torch.equal(exact, teacher)
    assert exact.data_ptr() != teacher.data_ptr()

    teacher_arm = apply_coupled_controller_ablation(
        actor, teacher, "teacher_arm_actor_hand"
    )
    assert torch.equal(teacher_arm[:, :7], teacher[:, :7])
    assert torch.equal(teacher_arm[:, 7:], actor[:, 7:])

    teacher_hand = apply_coupled_controller_ablation(
        actor, teacher, "actor_arm_teacher_hand"
    )
    assert torch.equal(teacher_hand[:, :7], actor[:, :7])
    assert torch.equal(teacher_hand[:, 7:], teacher[:, 7:])
    _expect_error(
        ValueError,
        apply_coupled_controller_ablation,
        actor,
        teacher,
        "unknown",
    )
    _expect_error(
        ValueError,
        apply_coupled_controller_ablation,
        actor[:, :14],
        teacher[:, :14],
        "exact_teacher",
    )


def test_public_latch_arm_gate_is_memoryless_and_exact() -> None:
    action = torch.arange(42, dtype=torch.float32).reshape(2, 21)
    observation = torch.zeros(2, 115)
    observation[1, 106] = 1.0
    original = action.clone()

    gated = apply_public_latch_arm_gate(action, observation)
    assert torch.equal(gated[0, :7], torch.zeros(7))
    assert torch.equal(gated[0, 7:], action[0, 7:])
    assert torch.equal(gated[1], action[1])
    assert torch.equal(action, original)
    assert gated.data_ptr() != action.data_ptr()

    half_speed = apply_public_latch_arm_gate(
        action,
        observation,
        latched_arm_scale=0.5,
    )
    torch.testing.assert_close(half_speed[1, :7], action[1, :7] * 0.5)
    assert torch.equal(half_speed[1, 7:], action[1, 7:])

    malformed = observation.clone()
    malformed[0, 106] = 0.5
    _expect_error(RuntimeError, apply_public_latch_arm_gate, action, malformed)
    _expect_error(
        ValueError,
        apply_public_latch_arm_gate,
        action[:, :14],
        observation,
    )
    _expect_error(
        ValueError,
        apply_public_latch_arm_gate,
        action,
        observation,
        latched_arm_scale=0.0,
    )


def test_diagnostic_approach_handoff_debounce_is_explicit() -> None:
    validate_diagnostic_approach_horizon(
        enabled=True,
        episode_length_s=None,
        min_step=400,
        hold_steps=4,
    )
    validate_diagnostic_approach_horizon(
        enabled=True,
        episode_length_s=20.0,
        min_step=996,
        hold_steps=4,
    )
    _expect_error(
        ValueError,
        validate_diagnostic_approach_horizon,
        enabled=True,
        episode_length_s=120.0,
        min_step=400,
        hold_steps=4,
    )
    _expect_error(
        ValueError,
        validate_diagnostic_approach_horizon,
        enabled=True,
        episode_length_s=20.0,
        min_step=997,
        hold_steps=4,
    )
    active = torch.tensor([True, True, False])
    ready_steps = torch.tensor([3, 2, 9], dtype=torch.long)
    episode_step = torch.tensor([400, 399, 500], dtype=torch.long)
    score = torch.tensor([0.30, 0.80, 0.90])
    proximity = torch.tensor([0.02, 0.50, 0.50])
    force = torch.tensor([30.0, 1.0, 1.0])
    latch = torch.tensor([False, False, False])
    result = update_diagnostic_approach_handoff(
        approach_active=active,
        ready_steps=ready_steps,
        episode_step=episode_step,
        pregrasp_score=score,
        proximity_quality=proximity,
        max_force_n=force,
        grasp_latch=latch,
        min_step=400,
        score_threshold=0.30,
        proximity_threshold=0.02,
        safe_force_limit_n=30.0,
        hold_steps=4,
    )
    assert torch.equal(result.enter_close, torch.tensor([True, False, False]))
    assert torch.equal(result.approach_active, torch.tensor([False, True, False]))
    assert torch.equal(result.ready_steps, torch.tensor([4, 0, 0]))
    assert torch.equal(ready_steps, torch.tensor([3, 2, 9]))

    blocked = update_diagnostic_approach_handoff(
        approach_active=torch.ones(3, dtype=torch.bool),
        ready_steps=torch.tensor([2, 2, 2]),
        episode_step=torch.full((3,), 500, dtype=torch.long),
        pregrasp_score=torch.full((3,), 0.9),
        proximity_quality=torch.full((3,), 0.9),
        max_force_n=torch.tensor([31.0, 1.0, 1.0]),
        grasp_latch=torch.tensor([False, True, False]),
        min_step=400,
        score_threshold=0.30,
        proximity_threshold=0.02,
        safe_force_limit_n=30.0,
        hold_steps=4,
    )
    assert torch.equal(blocked.ready_steps, torch.tensor([0, 0, 3]))
    assert not bool(blocked.enter_close.any())
    _expect_error(
        TypeError,
        update_diagnostic_approach_handoff,
        approach_active=active,
        ready_steps=ready_steps,
        episode_step=episode_step,
        pregrasp_score=score,
        proximity_quality=proximity,
        max_force_n=force,
        grasp_latch=latch.float(),
        min_step=400,
        score_threshold=0.30,
        proximity_threshold=0.02,
        safe_force_limit_n=30.0,
        hold_steps=4,
    )

    search_action = torch.full((3, 21), 0.25)
    flashsac_action = torch.full((3, 21), -0.75)
    base_only = select_diagnostic_approach_action(
        search_action=search_action,
        flashsac_action=flashsac_action,
        approach_active=torch.ones(3, dtype=torch.bool),
    )
    routed = select_diagnostic_approach_action(
        search_action=search_action,
        flashsac_action=flashsac_action,
        approach_active=torch.tensor([True, False, False]),
    )
    assert torch.equal(base_only, search_action)
    assert torch.equal(routed[0], search_action[0])
    assert torch.equal(routed[1:], flashsac_action[1:])
    assert torch.equal(search_action, torch.full((3, 21), 0.25))


def _handoff_artifact_row(*, env_slot: int = 3) -> dict[str, Any]:
    observation = torch.linspace(-1.0, 1.0, 115, dtype=torch.float32)
    observation[106] = 0.0
    return {
        "observation": observation,
        "search_action": torch.linspace(-0.5, 0.5, 21),
        "flashsac_action": torch.linspace(0.5, -0.5, 21),
        "env_slot": env_slot,
        "slot_episode_index": 0,
        "episode_index": 17,
        "handoff_step": 403,
        "pregrasp_score": 0.34,
        "proximity_quality": 0.08,
        "max_force_n": 12.0,
        "true_clearance_m": 0.001,
        "outcome_episode_length": 612,
        "outcome_max_true_clearance_m": 0.205,
        "outcome_success": True,
        "outcome_failure": False,
        "outcome_time_out": False,
        "outcome_ever_grasped": True,
        "outcome_ever_clearance_ge_5cm": True,
        "outcome_ever_clearance_ge_20cm": True,
        "outcome_dropped": False,
        "outcome_unsafe_force": False,
        "outcome_ever_unlatched_clearance_ge_5cm": False,
        "outcome_ever_post_candidate_latch": True,
    }


def test_diagnostic_handoff_artifact_is_strict_and_weights_only() -> None:
    metadata = {
        "treatment": "handoff_to_flashsac",
        "seed": 257,
        "supervisor": {
            "minimum_zero_based_episode_step": 400,
            "pregrasp_score_threshold": 0.32,
            "minimum_proximity_quality": 0.02,
            "hold_steps": 4,
            "safe_force_limit_n": 30.0,
            "requires_unlatched": True,
            "requires_abs_true_clearance_le_m": 0.005,
        },
    }
    payload = build_diagnostic_handoff_artifact(
        [_handoff_artifact_row(env_slot=8), _handoff_artifact_row(env_slot=2)],
        metadata=metadata,
    )
    assert payload["observation"].shape == (2, 115)
    assert payload["search_action"].shape == (2, 21)
    assert payload["env_slot"].tolist() == [2, 8]
    assert payload["metadata"]["handoff_rows"] == 2
    assert payload["observation"].device.type == "cpu"
    assert payload["observation"].is_contiguous()

    empty = build_diagnostic_handoff_artifact([], metadata=metadata)
    assert empty["observation"].shape == (0, 115)
    assert empty["outcome_success"].shape == (0,)

    duplicate = _handoff_artifact_row()
    _expect_error(
        ValueError,
        build_diagnostic_handoff_artifact,
        [duplicate, dict(duplicate)],
        metadata=metadata,
    )
    invalid_success = _handoff_artifact_row()
    invalid_success["outcome_ever_clearance_ge_20cm"] = False
    _expect_error(
        ValueError,
        build_diagnostic_handoff_artifact,
        [invalid_success],
        metadata=metadata,
    )
    non_finite = _handoff_artifact_row()
    non_finite["observation"] = non_finite["observation"].clone()
    non_finite["observation"][0] = float("nan")
    _expect_error(
        ValueError,
        build_diagnostic_handoff_artifact,
        [non_finite],
        metadata=metadata,
    )
    invalid_bool = _handoff_artifact_row()
    invalid_bool["outcome_success"] = "False"
    _expect_error(
        TypeError,
        build_diagnostic_handoff_artifact,
        [invalid_bool],
        metadata=metadata,
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "handoff.pt"
        _atomic_torch_save(path, payload)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.equal(restored["observation"], payload["observation"])
        _expect_error(FileExistsError, _atomic_torch_save, path, payload)
        dangling = Path(directory) / "dangling.pt"
        dangling.symlink_to(Path(directory) / "missing-target.pt")
        _expect_error(FileExistsError, _atomic_torch_save, dangling, payload)
        assert not list(Path(directory).glob(".*.tmp-*"))


def test_public_latch_hand_hold_uses_observed_previous_action() -> None:
    action = torch.arange(42, dtype=torch.float32).reshape(2, 21) / 42.0
    observation = torch.zeros(2, 115)
    observation[:, 77:86] = torch.arange(18, dtype=torch.float32).reshape(2, 9) / 20.0
    observation[:, 87:92] = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 12.0
    observation[1, 106] = 1.0
    original = action.clone()

    held = apply_public_latch_hand_hold(action, observation)
    assert torch.equal(held[0], action[0])
    assert torch.equal(held[1, :7], action[1, :7])
    assert torch.equal(held[1, 7:16], observation[1, 77:86])
    assert torch.equal(held[1, 16:21], observation[1, 87:92])
    assert torch.equal(action, original)
    assert held.data_ptr() != action.data_ptr()

    malformed = observation.clone()
    malformed[0, 106] = 0.5
    _expect_error(RuntimeError, apply_public_latch_hand_hold, action, malformed)
    _expect_error(
        ValueError,
        apply_public_latch_hand_hold,
        action[:, :14],
        observation,
    )
    _expect_error(
        TypeError,
        apply_public_latch_hand_hold,
        action.to(dtype=torch.float64),
        observation,
    )


def test_curriculum_argument_contract() -> None:
    assert task_mode_from_option_flags(
        close_option_mode=False, power_close_option_mode=False
    ) == FULL_TASK_MODE
    assert task_mode_from_option_flags(
        close_option_mode=True, power_close_option_mode=False
    ) == CLOSE_OPTION_MODE
    assert task_mode_from_option_flags(
        close_option_mode=False, power_close_option_mode=True
    ) == POWER_CLOSE_OPTION_MODE
    assert task_mode_from_option_flags(
        close_option_mode=False,
        power_close_option_mode=False,
        coupled_power_align_close_option_mode=True,
    ) == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE
    assert task_mode_from_option_flags(
        close_option_mode=False,
        power_close_option_mode=False,
        coupled_teacher_residual_mode=True,
    ) == COUPLED_TEACHER_RESIDUAL_MODE
    _expect_error(
        ValueError,
        task_mode_from_option_flags,
        close_option_mode=True,
        power_close_option_mode=True,
    )
    _expect_error(
        ValueError,
        task_mode_from_option_flags,
        close_option_mode=False,
        power_close_option_mode=True,
        coupled_power_align_close_option_mode=True,
    )
    _expect_error(
        ValueError,
        task_mode_from_option_flags,
        close_option_mode=False,
        power_close_option_mode=False,
        coupled_power_align_close_option_mode=True,
        coupled_teacher_residual_mode=True,
    )
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
    with tempfile.TemporaryDirectory() as directory:
        dataset = Path(directory) / "close.pt"
        dataset.touch()
        validate_close_option_evaluation_config(
            close_option_mode=True,
            curriculum_dataset=dataset,
            curriculum_boundary="close_start",
            curriculum_probability=1.0,
            curriculum_joint_noise=0.005,
            episode_length_s=0.40,
        )
        validate_close_option_evaluation_config(
            close_option_mode=True,
            curriculum_dataset=dataset,
            curriculum_boundary="close_start",
            curriculum_probability=1.0,
            curriculum_joint_noise=0.005,
            episode_length_s=3.0,
            coupled_power_align_close_option_mode=True,
        )
        _expect_error(
            ValueError,
            validate_close_option_evaluation_config,
            close_option_mode=True,
            curriculum_dataset=dataset,
            curriculum_boundary="close_start",
            curriculum_probability=1.0,
            curriculum_joint_noise=0.005,
            episode_length_s=2.99,
            coupled_power_align_close_option_mode=True,
        )
        validate_hierarchical_arm_hold_evaluation_config(
            enabled=True,
            close_option_mode=False,
            curriculum_dataset=dataset,
            curriculum_boundary="close_start",
            curriculum_probability=1.0,
            curriculum_joint_noise=0.005,
            episode_length_s=20.0,
            confirm_steps=15,
            grasp_quality_threshold=0.35,
            min_hold_quality=0.5,
            safe_force_limit=30.0,
        )
        validate_public_latch_arm_gate_evaluation_config(
            enabled=True,
            task_mode=FULL_TASK_MODE,
            curriculum_dataset=dataset,
            curriculum_boundary="close_start",
            curriculum_probability=1.0,
            curriculum_joint_noise=0.005,
            episode_length_s=20.0,
        )
        for override in (
            {"task_mode": CLOSE_OPTION_MODE},
            {"curriculum_dataset": None},
            {"curriculum_boundary": "lift_start"},
            {"curriculum_probability": 0.5},
            {"curriculum_joint_noise": 0.021},
            {"episode_length_s": 0.29},
        ):
            config = {
                "enabled": True,
                "task_mode": FULL_TASK_MODE,
                "curriculum_dataset": dataset,
                "curriculum_boundary": "close_start",
                "curriculum_probability": 1.0,
                "curriculum_joint_noise": 0.005,
                "episode_length_s": 20.0,
            }
            config.update(override)
            _expect_error(
                ValueError,
                validate_public_latch_arm_gate_evaluation_config,
                **config,
            )
        for override in (
            {"curriculum_dataset": None},
            {"curriculum_boundary": "lift_start"},
            {"curriculum_probability": 0.5},
            {"curriculum_joint_noise": 0.021},
            {"episode_length_s": 0.39},
            {"episode_length_s": 20.0},
        ):
            config = {
                "close_option_mode": True,
                "curriculum_dataset": dataset,
                "curriculum_boundary": "close_start",
                "curriculum_probability": 1.0,
                "curriculum_joint_noise": 0.005,
                "episode_length_s": 5.0,
            }
            config.update(override)
            _expect_error(ValueError, validate_close_option_evaluation_config, **config)
        for override in (
            {"close_option_mode": True},
            {"curriculum_dataset": None},
            {"curriculum_boundary": "lift_start"},
            {"curriculum_probability": 0.5},
            {"curriculum_joint_noise": 0.021},
            {"episode_length_s": 0.29},
            {"confirm_steps": 0},
            {"grasp_quality_threshold": 1.01},
            {"min_hold_quality": float("nan")},
            {"safe_force_limit": 0.0},
        ):
            config = {
                "enabled": True,
                "close_option_mode": False,
                "curriculum_dataset": dataset,
                "curriculum_boundary": "close_start",
                "curriculum_probability": 1.0,
                "curriculum_joint_noise": 0.005,
                "episode_length_s": 20.0,
                "confirm_steps": 15,
                "grasp_quality_threshold": 0.35,
                "min_hold_quality": 0.5,
                "safe_force_limit": 30.0,
            }
            config.update(override)
            _expect_error(
                ValueError,
                validate_hierarchical_arm_hold_evaluation_config,
                **config,
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


def _actor_state(
    *,
    blocks: int,
    hidden: int,
    compiled: bool,
    action_dim: int = ACTION_DIM,
    observation_dim: int = 115,
) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod." if compiled else ""
    state = {
        f"{prefix}embedder.norm.weight": torch.ones(observation_dim),
        f"{prefix}embedder.norm.bias": torch.zeros(observation_dim),
        f"{prefix}embedder.norm.running_mean": torch.zeros(observation_dim),
        f"{prefix}embedder.norm.running_var": torch.ones(observation_dim),
        f"{prefix}embedder.w.w.weight": torch.zeros(hidden, observation_dim),
        f"{prefix}predictor.mean_w.w.weight": torch.zeros(action_dim, hidden),
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
    assert HAND_NOISE_GROUP_SPECS == (
        ("token", 0, 9, 0.5, 1.25, 32),
        ("residual", 9, 14, 0.35, 1.5, 16),
    )
    for compiled in (False, True):
        for action_dim in (HAND_ACTION_DIM, ACTION_DIM):
            production = _actor_state(
                blocks=2,
                hidden=128,
                compiled=compiled,
                action_dim=action_dim,
            )
            smoke = _actor_state(
                blocks=1,
                hidden=32,
                compiled=compiled,
                action_dim=action_dim,
            )
            assert infer_actor_action_dim_from_state(production) == action_dim
            assert (
                infer_actor_architecture_from_state(
                    production, expected_action_dim=action_dim
                )
                == "production"
            )
            assert infer_actor_architecture_from_state(smoke) == "smoke"
            _expect_error(
                RuntimeError,
                infer_actor_architecture_from_state,
                production,
                expected_action_dim=(
                    ACTION_DIM if action_dim == HAND_ACTION_DIM else HAND_ACTION_DIM
                ),
            )
    coupled = _actor_state(
        blocks=2,
        hidden=128,
        compiled=False,
        observation_dim=COUPLED_OBSERVATION_DIM,
    )
    assert (
        infer_actor_architecture_from_state(
            coupled,
            expected_action_dim=ACTION_DIM,
            expected_observation_dim=COUPLED_OBSERVATION_DIM,
        )
        == "production"
    )
    _expect_error(
        RuntimeError,
        infer_actor_architecture_from_state,
        coupled,
        expected_observation_dim=115,
    )
    source = _actor_state(blocks=2, hidden=128, compiled=False)
    source["embedder.w.w.weight"] = torch.arange(
        128 * 115, dtype=torch.float32
    ).reshape(128, 115)
    expanded = zero_expand_actor_observation_state(source)
    assert expanded["embedder.w.w.weight"].shape == (128, COUPLED_OBSERVATION_DIM)
    assert torch.equal(
        expanded["embedder.w.w.weight"][:, :115],
        source["embedder.w.w.weight"],
    )
    assert torch.count_nonzero(expanded["embedder.w.w.weight"][:, 115:]) == 0
    expanded_vector_keys = {
        "embedder.norm.weight",
        "embedder.norm.bias",
        "embedder.norm.running_mean",
        "embedder.norm.running_var",
    }
    for key in expanded_vector_keys:
        assert expanded[key].shape == (COUPLED_OBSERVATION_DIM,)
        assert torch.equal(expanded[key][:115], source[key])
    assert torch.all(expanded["embedder.norm.weight"][115:] == 1.0)
    assert torch.all(expanded["embedder.norm.bias"][115:] == 0.0)
    assert torch.all(expanded["embedder.norm.running_mean"][115:] == 0.0)
    assert torch.all(expanded["embedder.norm.running_var"][115:] == 1.0)
    for key in source.keys() - expanded_vector_keys - {"embedder.w.w.weight"}:
        assert expanded[key] is source[key]
    assert (
        FULL115_TO_COUPLED131_OBSERVATION_TRANSFER
        == "full115_to_coupled131_zero_pad_input_v1"
    )
    mixed = _actor_state(blocks=2, hidden=128, compiled=False)
    mixed["_orig_mod.predictor.std_bias"] = torch.zeros(21)
    _expect_error(RuntimeError, infer_actor_architecture_from_state, mixed)
    _expect_error(
        RuntimeError,
        infer_actor_architecture_from_state,
        _actor_state(blocks=3, hidden=128, compiled=False),
    )
    _expect_error(
        RuntimeError,
        infer_actor_action_dim_from_state,
        _actor_state(blocks=2, hidden=128, compiled=False, action_dim=13),
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


def test_checkpoint_task_mode_evaluation_contract() -> None:
    assert not resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=FULL_TASK_MODE,
        requested_task_mode=FULL_TASK_MODE,
        allow_cross_task_actor=False,
    )
    assert not resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=CLOSE_OPTION_MODE,
        requested_task_mode=CLOSE_OPTION_MODE,
        allow_cross_task_actor=False,
    )
    assert not resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=POWER_CLOSE_OPTION_MODE,
        requested_task_mode=POWER_CLOSE_OPTION_MODE,
        allow_cross_task_actor=False,
    )
    assert not resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        requested_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        allow_cross_task_actor=False,
    )
    assert not resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=COUPLED_TEACHER_RESIDUAL_MODE,
        requested_task_mode=COUPLED_TEACHER_RESIDUAL_MODE,
        allow_cross_task_actor=False,
    )
    _expect_error(
        ValueError,
        resolve_cross_task_actor_evaluation,
        checkpoint_task_mode=FULL_TASK_MODE,
        requested_task_mode=CLOSE_OPTION_MODE,
        allow_cross_task_actor=False,
    )
    assert resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=FULL_TASK_MODE,
        requested_task_mode=CLOSE_OPTION_MODE,
        allow_cross_task_actor=True,
    )
    assert resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=CLOSE_OPTION_MODE,
        requested_task_mode=FULL_TASK_MODE,
        allow_cross_task_actor=True,
    )
    _expect_error(
        ValueError,
        resolve_cross_task_actor_evaluation,
        checkpoint_task_mode=FULL_TASK_MODE,
        requested_task_mode=POWER_CLOSE_OPTION_MODE,
        allow_cross_task_actor=False,
    )
    assert resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=FULL_TASK_MODE,
        requested_task_mode=POWER_CLOSE_OPTION_MODE,
        allow_cross_task_actor=True,
    )
    assert resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=POWER_CLOSE_OPTION_MODE,
        requested_task_mode=FULL_TASK_MODE,
        allow_cross_task_actor=True,
    )
    _expect_error(
        ValueError,
        resolve_cross_task_actor_evaluation,
        checkpoint_task_mode=FULL_TASK_MODE,
        requested_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        allow_cross_task_actor=False,
    )
    assert resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=FULL_TASK_MODE,
        requested_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        allow_cross_task_actor=True,
    )

    full_contract = {
        "version": 3,
        "task_mode": FULL_TASK_MODE,
        "policy_router": None,
        **requested_policy_action_contract(FULL_TASK_MODE),
    }
    power_contract = {
        "version": 3,
        "task_mode": POWER_CLOSE_OPTION_MODE,
        "policy_router": None,
        **requested_policy_action_contract(POWER_CLOSE_OPTION_MODE),
    }
    coupled_contract = {
        "version": 3,
        "task_mode": COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        "policy_router": None,
        **requested_policy_action_contract(COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE),
    }
    residual_contract = {
        "version": 4,
        "task_mode": COUPLED_TEACHER_RESIDUAL_MODE,
        "policy_router": None,
        **requested_policy_action_contract(COUPLED_TEACHER_RESIDUAL_MODE),
    }
    source, target, indices = validate_checkpoint_evaluation_contract(
        checkpoint_task_mode=COUPLED_TEACHER_RESIDUAL_MODE,
        checkpoint_contract=residual_contract,
        requested_task_mode=COUPLED_TEACHER_RESIDUAL_MODE,
        actor_action_dim=HAND_ACTION_DIM,
    )
    assert source == target == requested_policy_action_contract(
        COUPLED_TEACHER_RESIDUAL_MODE
    )
    assert indices is None
    source, target, indices = validate_checkpoint_evaluation_contract(
        checkpoint_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        checkpoint_contract=coupled_contract,
        requested_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        actor_action_dim=ACTION_DIM,
    )
    assert source == target == requested_policy_action_contract(
        COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE
    )
    assert indices is None
    source, target, indices = validate_checkpoint_evaluation_contract(
        checkpoint_task_mode=POWER_CLOSE_OPTION_MODE,
        checkpoint_contract=power_contract,
        requested_task_mode=POWER_CLOSE_OPTION_MODE,
        actor_action_dim=HAND_ACTION_DIM,
    )
    assert source == target == requested_policy_action_contract(POWER_CLOSE_OPTION_MODE)
    assert indices is None

    source, target, indices = validate_checkpoint_evaluation_contract(
        checkpoint_task_mode=FULL_TASK_MODE,
        checkpoint_contract=full_contract,
        requested_task_mode=POWER_CLOSE_OPTION_MODE,
        actor_action_dim=ACTION_DIM,
    )
    assert source == requested_policy_action_contract(FULL_TASK_MODE)
    assert target == requested_policy_action_contract(POWER_CLOSE_OPTION_MODE)
    assert indices == tuple(range(7, 21))

    legacy_source, _, legacy_indices = validate_checkpoint_evaluation_contract(
        checkpoint_task_mode=FULL_TASK_MODE,
        checkpoint_contract={"version": 2, "task_mode": FULL_TASK_MODE},
        requested_task_mode=POWER_CLOSE_OPTION_MODE,
        actor_action_dim=ACTION_DIM,
    )
    assert legacy_source == requested_policy_action_contract(FULL_TASK_MODE)
    assert legacy_indices == tuple(range(7, 21))

    source, target, indices = validate_checkpoint_evaluation_contract(
        checkpoint_task_mode=FULL_TASK_MODE,
        checkpoint_contract=full_contract,
        requested_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        actor_action_dim=ACTION_DIM,
    )
    assert source == requested_policy_action_contract(FULL_TASK_MODE)
    assert target == requested_policy_action_contract(
        COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE
    )
    assert indices is None

    routed_contract = {
        "version": 6,
        "task_mode": FULL_TASK_MODE,
        "policy_router": {
            "kind": "public_latch_frozen_actor_v1",
            "observation_index": 106,
            "close_value": 0.0,
            "frozen_value": 1.0,
            "trainable_action_slice": [7, 21],
            "close_arm_fill": "exact_zero",
            "frozen_action_slice": [0, 21],
            "frozen_action_sampling": "deterministic_tanh_mean",
            "frozen_action_entropy": "exact_zero",
            "frozen_actor": {
                "filename": "frozen_lift_actor.pt",
                "sha256": "0" * 64,
                "network_sha256": "1" * 64,
                "source_actor_sha256": "2" * 64,
                "observation_dim": 115,
                "action_dim": ACTION_DIM,
            },
        },
        **requested_policy_action_contract(FULL_TASK_MODE),
    }
    source, target, indices = validate_checkpoint_evaluation_contract(
        checkpoint_task_mode=FULL_TASK_MODE,
        checkpoint_contract=routed_contract,
        requested_task_mode=FULL_TASK_MODE,
        actor_action_dim=ACTION_DIM,
    )
    assert source == target == requested_policy_action_contract(FULL_TASK_MODE)
    assert indices is None

    # The outer opt-in may permit ordinary cross-task actor projections, but a
    # routed V6 checkpoint is a self-contained two-actor policy.  The pure
    # contract resolver must therefore reject projection independently.
    assert resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=FULL_TASK_MODE,
        requested_task_mode=POWER_CLOSE_OPTION_MODE,
        allow_cross_task_actor=True,
    )
    _expect_error(
        ValueError,
        validate_checkpoint_evaluation_contract,
        checkpoint_task_mode=FULL_TASK_MODE,
        checkpoint_contract=routed_contract,
        requested_task_mode=POWER_CLOSE_OPTION_MODE,
        actor_action_dim=ACTION_DIM,
    )

    _expect_error(
        ValueError,
        validate_checkpoint_evaluation_contract,
        checkpoint_task_mode=POWER_CLOSE_OPTION_MODE,
        checkpoint_contract=power_contract,
        requested_task_mode=FULL_TASK_MODE,
        actor_action_dim=HAND_ACTION_DIM,
    )
    _expect_error(
        RuntimeError,
        validate_checkpoint_evaluation_contract,
        checkpoint_task_mode=FULL_TASK_MODE,
        checkpoint_contract=full_contract,
        requested_task_mode=FULL_TASK_MODE,
        actor_action_dim=HAND_ACTION_DIM,
    )
    _expect_error(
        ValueError,
        validate_checkpoint_evaluation_contract,
        checkpoint_task_mode=COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
        checkpoint_contract=coupled_contract,
        requested_task_mode=FULL_TASK_MODE,
        actor_action_dim=ACTION_DIM,
    )
    for source_mode, source_contract, target_mode, actor_dim in (
        (
            COUPLED_TEACHER_RESIDUAL_MODE,
            residual_contract,
            POWER_CLOSE_OPTION_MODE,
            HAND_ACTION_DIM,
        ),
        (
            POWER_CLOSE_OPTION_MODE,
            power_contract,
            COUPLED_TEACHER_RESIDUAL_MODE,
            HAND_ACTION_DIM,
        ),
        (
            FULL_TASK_MODE,
            full_contract,
            COUPLED_TEACHER_RESIDUAL_MODE,
            ACTION_DIM,
        ),
    ):
        _expect_error(
            ValueError,
            validate_checkpoint_evaluation_contract,
            checkpoint_task_mode=source_mode,
            checkpoint_contract=source_contract,
            requested_task_mode=target_mode,
            actor_action_dim=actor_dim,
        )
    mismatched_contract = dict(power_contract)
    mismatched_contract["observation_contract"] = "wrong_observation_contract"
    _expect_error(
        ValueError,
        validate_checkpoint_evaluation_contract,
        checkpoint_task_mode=POWER_CLOSE_OPTION_MODE,
        checkpoint_contract=mismatched_contract,
        requested_task_mode=POWER_CLOSE_OPTION_MODE,
        actor_action_dim=HAND_ACTION_DIM,
    )


def test_local_train_contract_import_precedes_upstream_path_mutation() -> None:
    source = inspect.getsource(run)
    local_contract_import = source.index("from train import read_checkpoint_task_contract")
    upstream_bridge_import = source.index("from agent_bridge import")
    assert local_contract_import < upstream_bridge_import
    assert "hand_only_actions=task_mode == POWER_CLOSE_OPTION_MODE" in source
    assert "action_transform=(" in source
    assert "env.canonicalize_policy_action(observation, action)" not in source
    assert "next_observation, reward, terminated, truncated, info = env.step(action)" in source
    assert "load_checkpoint_teacher_prior(" in source
    assert 'metrics["teacher_action_prior_sha256"]' in source
    assert '"coupled_power_align_close_option_mode"' in source
    assert "zero_expanded_actor_checkpoint(checkpoint)" in source
    assert "source_action_indices=source_action_indices" in source
    assert "agent.load_actor(" in source
    assert "agent.load(str(checkpoint))" not in source

    router_constructor = source.index("PublicLatchFrozenActorRouter(")
    actor_load = source.index("agent.load_actor(")
    frozen_sidecar_load = source.index("agent.load_frozen_lift_actor_sidecar(")
    assert router_constructor < actor_load < frozen_sidecar_load
    assert (
        "checkpoint_native_public_latch_gate = bool(\n"
        "        not checkpoint_native_router\n"
        "        and checkpoint_task_mode == FULL_TASK_MODE"
    ) in source
    assert 'metrics["checkpoint_native_policy_router"]' in source
    router_metrics = source.index(
        "if checkpoint_native_router:\n"
        '            frozen_actor_contract = checkpoint_router["frozen_actor"]'
    )
    v5_metrics = source.index(
        "elif checkpoint_native_public_latch_gate:", router_metrics
    )
    assert router_metrics < v5_metrics


def test_recoverability_selector_evaluation_argument_contract() -> None:
    selector = Path("selector.pt")
    report = Path("selector.json")
    approach = Path("search.pth")
    validate_recoverability_selector_evaluation_config(
        selector_checkpoint=None,
        selector_report=None,
        approach_checkpoint=None,
        approach_base_only=False,
        approach_handoff_output=None,
    )
    validate_recoverability_selector_evaluation_config(
        selector_checkpoint=selector,
        selector_report=report,
        approach_checkpoint=approach,
        approach_base_only=False,
        approach_handoff_output=None,
    )
    _expect_error(
        ValueError,
        validate_recoverability_selector_evaluation_config,
        selector_checkpoint=selector,
        selector_report=None,
        approach_checkpoint=approach,
        approach_base_only=False,
        approach_handoff_output=None,
    )
    _expect_error(
        ValueError,
        validate_recoverability_selector_evaluation_config,
        selector_checkpoint=selector,
        selector_report=report,
        approach_checkpoint=None,
        approach_base_only=False,
        approach_handoff_output=None,
    )
    _expect_error(
        ValueError,
        validate_recoverability_selector_evaluation_config,
        selector_checkpoint=selector,
        selector_report=report,
        approach_checkpoint=approach,
        approach_base_only=True,
        approach_handoff_output=None,
    )
    _expect_error(
        ValueError,
        validate_recoverability_selector_evaluation_config,
        selector_checkpoint=selector,
        selector_report=report,
        approach_checkpoint=approach,
        approach_base_only=False,
        approach_handoff_output=Path("capture.pt"),
    )


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
    test_hierarchical_arm_hold_terminal_contract()
    test_close_option_terminal_event_contract()
    test_reset_before_terminal_physical_truth()
    test_close_option_physical_truth_is_not_20cm_success()
    test_power_close_terminal_and_physical_contract()
    test_coupled_power_terminal_physical_and_metrics_contract()
    test_teacher_residual_strict_contract_and_metrics()
    test_embedded_teacher_prior_sha_and_curriculum_lineage()
    test_exact_episode_quotas_and_strict_tracker()
    test_close_option_tracker_and_metrics_are_separate_from_full_success()
    test_power_close_tracker_metrics_and_hand_action_contract()
    test_coupled_controller_ablation_composition()
    test_public_latch_arm_gate_is_memoryless_and_exact()
    test_diagnostic_approach_handoff_debounce_is_explicit()
    test_diagnostic_handoff_artifact_is_strict_and_weights_only()
    test_public_latch_hand_hold_uses_observed_previous_action()
    test_curriculum_argument_contract()
    test_checkpoint_architecture_and_path_contract()
    test_checkpoint_task_mode_evaluation_contract()
    test_recoverability_selector_evaluation_argument_contract()
    test_local_train_contract_import_precedes_upstream_path_mutation()
    test_strict_json_and_summary()
    print("evaluate contract tests passed")


if __name__ == "__main__":
    main()
