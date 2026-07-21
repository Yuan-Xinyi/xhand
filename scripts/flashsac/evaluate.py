#!/usr/bin/env python3
"""Deterministic physical evaluation for PickTool FlashSAC checkpoints.

The evaluator deliberately runs in a fresh Isaac process.  It reconstructs the
same production network architecture as :mod:`train`, loads a checkpoint
without optimizer or reward-normalizer state, and executes ``tanh(actor_mean)``
without exploration noise.

Isaac Lab's DirectRLEnv auto-resets completed sub-environments before ``step``
returns.  The task therefore clones its per-environment event flags, true mesh
clearance and grasp latch into ``pick_tool_terminal`` inside ``_get_dones``.
Those reset-before clones are the sole authority for the episode being closed;
the physical state visible after ``step`` is used only to initialize the next
episode on a reset row.

Full-task success, close-option success, failure, timeout and safety counts are
episode events.  They are never inferred from aggregate log means or from a
post-reset state.  Close-option evaluation deliberately uses a separate output
contract: a stable latch near the table is never reported as 20 cm success.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import traceback
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


OBSERVATION_DIM = 115
COUPLED_OBSERVATION_DIM = 131
ACTION_DIM = 21
ARM_ACTION_DIM = 7
HAND_ACTION_DIM = 14
PRODUCTION_ACTOR_BLOCKS = 2
PRODUCTION_ACTOR_HIDDEN = 128
PRODUCTION_CRITIC_BLOCKS = 2
PRODUCTION_CRITIC_HIDDEN = 256
PRODUCTION_CRITIC_BINS = 101
SMOKE_ACTOR_BLOCKS = 1
SMOKE_ACTOR_HIDDEN = 32
SMOKE_CRITIC_BLOCKS = 1
SMOKE_CRITIC_HIDDEN = 64
SMOKE_CRITIC_BINS = 51
FULL_TASK_MODE = "full_task"
CLOSE_OPTION_MODE = "close_option"
POWER_CLOSE_OPTION_MODE = "power_close_option_v1"
COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE = "coupled_power_align_close_option_v1"
TASK_MODES = (
    FULL_TASK_MODE,
    CLOSE_OPTION_MODE,
    POWER_CLOSE_OPTION_MODE,
    COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
)
POWER_CLOSE_TASK_MODES = (
    POWER_CLOSE_OPTION_MODE,
    COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE,
)
CLOSE_OPTION_TASK_MODES = (CLOSE_OPTION_MODE, *POWER_CLOSE_TASK_MODES)

FULL_POLICY_ACTION_LAYOUT = "arm_delta7|crossdex_token9|distal_residual5"
HAND_POLICY_ACTION_LAYOUT = "crossdex_token9|distal_residual5"
IDENTITY_ACTION_PROJECTION = "identity_v1"
HAND_ACTION_PROJECTION = "prepend_zero_arm7_v1"
STANDARD_OBSERVATION_CONTRACT = "pick_tool_markov115_v1"
POWER_OBSERVATION_CONTRACT = "pick_tool_power_close_markov115_v1"
COUPLED_POWER_OBSERVATION_CONTRACT = (
    "pick_tool_coupled_power_align_close_state131_v1"
)
COUPLED_POWER_ARM_TARGET_OFFSET_LIMIT = 0.12
COUPLED_POWER_ARM_TARGET_OFFSET_TOLERANCE = 1.0e-6
COUPLED_POWER_ROTATION_DRIFT_LIMIT = 0.35
COUPLED_POWER_XY_DRIFT_LIMIT = 0.03
COUPLED_POWER_TRUE_CLEARANCE_LIMIT = 0.015
FULL115_TO_COUPLED131_OBSERVATION_TRANSFER = (
    "full115_to_coupled131_zero_pad_input_v1"
)

# This is checkpoint metadata as well as interaction behavior.  Bridge loading
# intentionally rejects a different grouping, so keep the evaluator contract
# explicit and simulation-free-testable.
NOISE_GROUP_SPECS = (
    ("arm", 0, 7, 1.0, 1.0, 64),
    ("token", 7, 16, 0.5, 1.25, 32),
    ("residual", 16, 21, 0.35, 1.5, 16),
)
HAND_NOISE_GROUP_SPECS = (
    ("token", 0, 9, 0.5, 1.25, 32),
    ("residual", 9, 14, 0.35, 1.5, 16),
)

TERMINAL_EVENT_KEYS = (
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
)

CLOSE_OPTION_EVENT_KEYS = (
    "full_task_success",
    "close_option_success",
    "close_option_failure",
    "close_option_timeout",
    "close_option_unlatched_lift",
    "close_option_horizontal_escape",
    "close_option_lost_window",
    "close_option_stable_steps",
)

POWER_CLOSE_OPTION_EVENT_KEYS = (
    "full_task_success",
    "close_option_success",
    "close_option_failure",
    "close_option_timeout",
    "close_option_unlatched_lift",
    "close_option_horizontal_escape",
    "close_option_lost_window",
    "close_option_stable_steps",
    "power_close_option_success",
    "power_close_option_failure",
    "power_close_option_timeout",
    "power_close_option_stable_steps",
    "power_is_grasped",
    "power_thumb_contact",
    "power_legal_other_contact_count",
    "power_close_quality",
    "power_wrap_quality",
    "power_grasp_quality",
    "power_grasp_latch_confirm_steps",
)

COUPLED_POWER_TELEMETRY_KEYS = (
    "coupled_power_pose_escape",
    "coupled_power_rotation_drift",
    "coupled_power_xy_drift",
    "coupled_power_true_clearance",
    "coupled_power_arm_target_offset_abs_max",
    "coupled_power_arm_target_saturated",
    "coupled_power_align_active",
)

ARM_HOLD_HANDOFF_KEYS = (
    "arm_hold_released",
    "arm_hold_stable_steps",
    "arm_hold_release_other_contacts",
    "arm_hold_release_grasp_quality",
    "arm_hold_release_wrap_quality",
    "arm_hold_release_max_force",
)


def _validate_task_mode(task_mode: str) -> str:
    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}; expected one of {TASK_MODES}")
    return task_mode


def task_mode_from_option_flags(
    *,
    close_option_mode: bool,
    power_close_option_mode: bool,
    coupled_power_align_close_option_mode: bool = False,
) -> str:
    """Resolve mutually exclusive user-facing task flags."""

    enabled = sum(
        bool(value)
        for value in (
            close_option_mode,
            power_close_option_mode,
            coupled_power_align_close_option_mode,
        )
    )
    if enabled > 1:
        raise ValueError(
            "close-option mode flags are mutually exclusive"
        )
    if coupled_power_align_close_option_mode:
        return COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE
    if power_close_option_mode:
        return POWER_CLOSE_OPTION_MODE
    if close_option_mode:
        return CLOSE_OPTION_MODE
    return FULL_TASK_MODE


def apply_coupled_controller_ablation(
    actor_action: torch.Tensor,
    teacher_action: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Compose one explicit diagnostic arm-by-hand controller condition."""

    if (
        not isinstance(actor_action, torch.Tensor)
        or not isinstance(teacher_action, torch.Tensor)
    ):
        raise TypeError("controller ablation actions must be torch tensors")
    if actor_action.shape != teacher_action.shape or actor_action.ndim != 2:
        raise ValueError("actor and teacher actions must share a two-dimensional shape")
    if actor_action.shape[1] != ACTION_DIM:
        raise ValueError(f"controller ablation requires {ACTION_DIM}-D actions")
    if actor_action.device != teacher_action.device:
        raise ValueError("actor and teacher actions must share a device")
    if actor_action.dtype != teacher_action.dtype:
        raise TypeError("actor and teacher actions must share a dtype")
    if mode == "exact_teacher":
        return teacher_action.clone()
    if mode == "teacher_arm_actor_hand":
        return torch.cat((teacher_action[:, :7], actor_action[:, 7:]), dim=-1)
    if mode == "actor_arm_teacher_hand":
        return torch.cat((actor_action[:, :7], teacher_action[:, 7:]), dim=-1)
    raise ValueError(f"unsupported coupled controller ablation {mode!r}")


def requested_policy_action_contract(task_mode: str) -> dict[str, Any]:
    """Return the evaluator-side policy/environment action boundary."""

    task_mode = _validate_task_mode(task_mode)
    if task_mode == POWER_CLOSE_OPTION_MODE:
        return {
            "policy_action_dim": HAND_ACTION_DIM,
            "policy_action_layout": HAND_POLICY_ACTION_LAYOUT,
            "environment_action_dim": ACTION_DIM,
            "action_projection": HAND_ACTION_PROJECTION,
            "observation_dim": OBSERVATION_DIM,
            "observation_contract": POWER_OBSERVATION_CONTRACT,
        }
    if task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
        return {
            "policy_action_dim": ACTION_DIM,
            "policy_action_layout": FULL_POLICY_ACTION_LAYOUT,
            "environment_action_dim": ACTION_DIM,
            "action_projection": IDENTITY_ACTION_PROJECTION,
            "observation_dim": COUPLED_OBSERVATION_DIM,
            "observation_contract": COUPLED_POWER_OBSERVATION_CONTRACT,
        }
    return {
        "policy_action_dim": ACTION_DIM,
        "policy_action_layout": FULL_POLICY_ACTION_LAYOUT,
        "environment_action_dim": ACTION_DIM,
        "action_projection": IDENTITY_ACTION_PROJECTION,
        "observation_dim": OBSERVATION_DIM,
        "observation_contract": STANDARD_OBSERVATION_CONTRACT,
    }


def checkpoint_policy_action_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate V3 policy metadata, with an explicit historical 21-D fallback."""

    version = contract.get("version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError("checkpoint task contract has an invalid version")
    keys = (
        "policy_action_dim",
        "policy_action_layout",
        "environment_action_dim",
        "action_projection",
        "observation_dim",
        "observation_contract",
    )
    if version < 3 and not any(key in contract for key in keys):
        return requested_policy_action_contract(FULL_TASK_MODE)
    missing = [key for key in keys if key not in contract]
    if missing:
        raise ValueError(f"checkpoint task contract is missing policy metadata: {missing}")
    result = {key: contract[key] for key in keys}
    allowed = (
        requested_policy_action_contract(FULL_TASK_MODE),
        requested_policy_action_contract(POWER_CLOSE_OPTION_MODE),
        requested_policy_action_contract(COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE),
    )
    if result not in allowed:
        raise ValueError(f"unsupported checkpoint policy action contract: {result}")
    return result


def validate_checkpoint_evaluation_contract(
    *,
    checkpoint_task_mode: str,
    checkpoint_contract: Mapping[str, Any],
    requested_task_mode: str,
    actor_action_dim: int,
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...] | None]:
    """Resolve exact-load and the two explicit full-task actor transfers."""

    checkpoint_task_mode = _validate_task_mode(checkpoint_task_mode)
    requested_task_mode = _validate_task_mode(requested_task_mode)
    source = checkpoint_policy_action_contract(checkpoint_contract)
    target = requested_policy_action_contract(requested_task_mode)
    if actor_action_dim != source["policy_action_dim"]:
        raise RuntimeError(
            "actor checkpoint action dimension disagrees with task_contract.json: "
            f"actor={actor_action_dim}, contract={source['policy_action_dim']}"
        )
    expected_source = requested_policy_action_contract(checkpoint_task_mode)
    if source != expected_source:
        raise ValueError(
            "checkpoint task mode and policy/observation contract disagree: "
            f"task_mode={checkpoint_task_mode!r}, policy={source}"
        )
    # Cross-mode use is independently gated by ``allow_cross_task_actor``.
    # Preserve historical full/legacy-close exact loads because their tensor
    # contracts are identical.
    if source == target:
        return source, target, None
    if (
        source == requested_policy_action_contract(FULL_TASK_MODE)
        and requested_task_mode == POWER_CLOSE_OPTION_MODE
    ):
        return source, target, tuple(range(ARM_ACTION_DIM, ACTION_DIM))
    if (
        checkpoint_task_mode == FULL_TASK_MODE
        and requested_task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE
        and source == requested_policy_action_contract(FULL_TASK_MODE)
        and target
        == requested_policy_action_contract(COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE)
    ):
        # The action map is identity.  Runtime loading separately zero-expands
        # the observation embedder from 115 to 131 features.
        return source, target, None
    if source["policy_action_dim"] == HAND_ACTION_DIM:
        raise ValueError(
            "a hand-only checkpoint cannot control the full 21-D task in a single-policy "
            "evaluation; compose close and lift policies explicitly"
        )
    raise ValueError(
        "unsupported checkpoint/requested policy action contract transition: "
        f"source={source}, target={target}"
    )


def resolve_cross_task_actor_evaluation(
    *,
    checkpoint_task_mode: str,
    requested_task_mode: str,
    allow_cross_task_actor: bool,
) -> bool:
    """Return whether evaluation crosses task modes, requiring explicit permission."""

    checkpoint_task_mode = _validate_task_mode(checkpoint_task_mode)
    requested_task_mode = _validate_task_mode(requested_task_mode)
    cross_task = checkpoint_task_mode != requested_task_mode
    if cross_task and not allow_cross_task_actor:
        raise ValueError(
            "requested evaluation task mode differs from checkpoint task contract; "
            "pass --allow_cross_task_actor for an explicit actor-only cross-task benchmark: "
            f"checkpoint={checkpoint_task_mode!r}, requested={requested_task_mode!r}"
        )
    return cross_task


def _require_vector(
    name: str,
    value: Any,
    *,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.shape != (num_envs,) or value.dtype != dtype or value.device != device:
        raise ValueError(
            f"{name} must be {dtype}[{num_envs}] on {device}, "
            f"got {value.dtype}{tuple(value.shape)} on {value.device}"
        )
    return value


def _terminal_mapping(info: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(info, Mapping):
        raise TypeError(f"adapter info must be a mapping, got {type(info).__name__}")
    raw = info.get("pick_tool_terminal")
    if not isinstance(raw, Mapping):
        raise KeyError("adapter info has no pick_tool_terminal ground truth")
    return raw


def validate_coupled_power_telemetry(
    info: Mapping[str, Any],
    *,
    num_envs: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Read every coupled pose/arm signal; missing telemetry is always fatal."""

    raw = _terminal_mapping(info)
    bool_keys = {
        "coupled_power_pose_escape",
        "coupled_power_arm_target_saturated",
        "coupled_power_align_active",
    }
    telemetry = {
        name: _require_vector(
            f"pick_tool_terminal[{name!r}]",
            raw.get(name),
            num_envs=num_envs,
            device=device,
            dtype=torch.bool if name in bool_keys else torch.float32,
        )
        for name in COUPLED_POWER_TELEMETRY_KEYS
    }
    for name in COUPLED_POWER_TELEMETRY_KEYS:
        value = telemetry[name]
        if value.dtype == torch.float32 and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains NaN or infinity")
    for name in (
        "coupled_power_rotation_drift",
        "coupled_power_xy_drift",
        "coupled_power_arm_target_offset_abs_max",
    ):
        if bool((telemetry[name] < 0.0).any()):
            raise RuntimeError(f"{name} must be non-negative")
    derived_pose_escape = (
        (telemetry["coupled_power_rotation_drift"] > COUPLED_POWER_ROTATION_DRIFT_LIMIT)
        | (telemetry["coupled_power_xy_drift"] > COUPLED_POWER_XY_DRIFT_LIMIT)
        | (
            telemetry["coupled_power_true_clearance"]
            > COUPLED_POWER_TRUE_CLEARANCE_LIMIT
        )
    )
    if not torch.equal(telemetry["coupled_power_pose_escape"], derived_pose_escape):
        raise RuntimeError(
            "coupled_power_pose_escape disagrees with rotation/xy/true-clearance ground truth"
        )
    return telemetry


def validate_terminal_events(
    info: Mapping[str, Any],
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    task_mode: str = FULL_TASK_MODE,
    close_option_confirm_steps: int = 15,
    power_required_other_contacts: int = 3,
    power_grasp_quality_threshold: float = 0.35,
    close_option_min_hold_quality: float = 0.5,
    close_option_safe_force_limit: float = 30.0,
) -> dict[str, torch.Tensor]:
    """Return task-authored terminal tensors after checking mode-specific consistency."""

    task_mode = _validate_task_mode(task_mode)
    if close_option_confirm_steps < 1:
        raise ValueError("close_option_confirm_steps must be positive")
    if power_required_other_contacts != 3:
        raise ValueError("power close v1 requires exactly three non-thumb contacts")
    for name, value in (
        ("power_grasp_quality_threshold", power_grasp_quality_threshold),
        ("close_option_min_hold_quality", close_option_min_hold_quality),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not math.isfinite(close_option_safe_force_limit) or close_option_safe_force_limit <= 0.0:
        raise ValueError("close_option_safe_force_limit must be finite and positive")

    num_envs = int(terminated.numel())
    if terminated.shape != (num_envs,) or terminated.dtype != torch.bool:
        raise ValueError("terminated must be a one-dimensional bool tensor")
    if truncated.shape != terminated.shape or truncated.dtype != torch.bool:
        raise ValueError("truncated must match the terminated bool vector")
    if truncated.device != terminated.device:
        raise ValueError("terminated and truncated must be on the same device")

    raw = _terminal_mapping(info)
    events = {
        name: _require_vector(
            f"pick_tool_terminal[{name!r}]",
            raw.get(name),
            num_envs=num_envs,
            device=terminated.device,
            dtype=torch.bool,
        )
        for name in TERMINAL_EVENT_KEYS
    }

    task_terminated = events["success"] | events["failure"]
    if not torch.equal(task_terminated, terminated):
        raise RuntimeError(
            "pick_tool_terminal success/failure does not match the reset-before terminated mask"
        )
    if not torch.equal(events["time_out"], truncated):
        raise RuntimeError(
            "pick_tool_terminal time_out does not match the reset-before truncated mask"
        )
    if bool((events["success"] & events["failure"]).any()):
        raise RuntimeError("an episode cannot be both a strict success and task failure")
    if bool((task_terminated & events["time_out"]).any()):
        raise RuntimeError("task termination and timeout must be mutually exclusive")

    full_task_success_raw = raw.get("full_task_success", events["success"])
    full_task_success = _require_vector(
        "pick_tool_terminal['full_task_success']",
        full_task_success_raw,
        num_envs=num_envs,
        device=terminated.device,
        dtype=torch.bool,
    )
    if task_mode == FULL_TASK_MODE:
        if not torch.equal(events["success"], full_task_success):
            raise RuntimeError("full-task generic success alias does not match full_task_success")
        failure_sources = (
            events["dropped"]
            | events["unsafe_force"]
            | events["unlatched_clearance_ge_5cm"]
        )
        if not torch.equal(events["failure"], failure_sources):
            raise RuntimeError(
                "task failure must be exactly drop, unsafe force, or unlatched lift"
            )
        # Preserve the established full-task return schema exactly.
        return events

    if task_mode in POWER_CLOSE_TASK_MODES:
        long_keys = {
            "close_option_stable_steps",
            "power_close_option_stable_steps",
            "power_legal_other_contact_count",
            "power_grasp_latch_confirm_steps",
        }
        float_keys = {
            "power_close_quality",
            "power_wrap_quality",
            "power_grasp_quality",
        }
        power_events = {
            name: _require_vector(
                f"pick_tool_terminal[{name!r}]",
                raw.get(name),
                num_envs=num_envs,
                device=terminated.device,
                dtype=(
                    torch.long
                    if name in long_keys
                    else torch.float32
                    if name in float_keys
                    else torch.bool
                ),
            )
            for name in POWER_CLOSE_OPTION_EVENT_KEYS
        }
        if not torch.equal(power_events["full_task_success"], full_task_success):
            raise RuntimeError("power-close payload has inconsistent full_task_success aliases")
        for generic, objective in (
            ("success", "power_close_option_success"),
            ("failure", "power_close_option_failure"),
            ("time_out", "power_close_option_timeout"),
        ):
            if not torch.equal(events[generic], power_events[objective]):
                raise RuntimeError(
                    f"power-close generic {generic} alias does not match {objective}"
                )
        for legacy, power in (
            ("close_option_success", "power_close_option_success"),
            ("close_option_failure", "power_close_option_failure"),
            ("close_option_timeout", "power_close_option_timeout"),
            ("close_option_stable_steps", "power_close_option_stable_steps"),
        ):
            if not torch.equal(power_events[legacy], power_events[power]):
                raise RuntimeError(
                    f"power-close fixed-schema alias {legacy} does not match {power}"
                )
        power_success = power_events["power_close_option_success"]
        power_failure_sources = (
            events["dropped"]
            | events["unsafe_force"]
            | power_events["close_option_unlatched_lift"]
            | power_events["close_option_horizontal_escape"]
            | power_events["close_option_lost_window"]
        )
        coupled_events: dict[str, torch.Tensor] = {}
        if task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
            coupled_events = validate_coupled_power_telemetry(
                info,
                num_envs=num_envs,
                device=terminated.device,
            )
            power_failure_sources |= coupled_events["coupled_power_pose_escape"]
            true_clearance = _require_vector(
                "pick_tool_terminal['true_clearance']",
                raw.get("true_clearance"),
                num_envs=num_envs,
                device=terminated.device,
                dtype=torch.float32,
            )
            if not torch.allclose(
                coupled_events["coupled_power_true_clearance"],
                true_clearance,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise RuntimeError(
                    "coupled true-clearance telemetry disagrees with mesh ground truth"
                )
        power_failure_sources &= ~power_success
        if not torch.equal(
            power_events["power_close_option_failure"], power_failure_sources
        ):
            raise RuntimeError(
                "power-close failure must be exactly drop, unsafe force, pre-latch lift, "
                "horizontal escape, or lost pregrasp window"
            )
        hold_quality = _require_vector(
            "pick_tool_terminal['hold_quality']",
            raw.get("hold_quality"),
            num_envs=num_envs,
            device=terminated.device,
            dtype=torch.float32,
        )
        max_force = _require_vector(
            "pick_tool_terminal['max_force']",
            raw.get("max_force"),
            num_envs=num_envs,
            device=terminated.device,
            dtype=torch.float32,
        )
        for name in (
            "power_close_quality",
            "power_wrap_quality",
            "power_grasp_quality",
        ):
            quality = power_events[name]
            if not bool(torch.isfinite(quality).all()) or bool(
                ((quality < 0.0) | (quality > 1.0)).any()
            ):
                raise RuntimeError(f"{name} must be finite and in [0, 1]")
        legal_other = power_events["power_legal_other_contact_count"]
        latch_confirm_steps = power_events["power_grasp_latch_confirm_steps"]
        stable_steps = power_events["power_close_option_stable_steps"]
        if bool(((legal_other < 0) | (legal_other > 4)).any()):
            raise RuntimeError("power legal non-thumb contact count must be in [0, 4]")
        if bool((latch_confirm_steps < 0).any()) or bool((stable_steps < 0).any()):
            raise RuntimeError("power latch/stable step counters must be non-negative")
        if not bool(torch.isfinite(hold_quality).all()) or not bool(
            torch.isfinite(max_force).all()
        ):
            raise FloatingPointError("power-close hold quality or force is not finite")
        invalid_success = power_success & (
            (~power_events["power_is_grasped"])
            | (~power_events["power_thumb_contact"])
            | (
                power_events["power_legal_other_contact_count"]
                < power_required_other_contacts
            )
            | (
                power_events["power_grasp_quality"]
                < power_grasp_quality_threshold
            )
            | (
                power_events["power_close_option_stable_steps"]
                < close_option_confirm_steps
            )
            | (hold_quality < close_option_min_hold_quality)
            | (max_force > close_option_safe_force_limit)
            | events["dropped"]
            | events["unsafe_force"]
        )
        if bool(invalid_success.any()):
            raise RuntimeError(
                "power-close success violates thumb+three, power-quality, 15-frame, "
                "hold, or force safety contract"
            )
        if task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
            invalid_coupled_success = power_success & (
                coupled_events["coupled_power_pose_escape"]
                | coupled_events["coupled_power_align_active"]
                | (
                    coupled_events["coupled_power_arm_target_offset_abs_max"]
                    > COUPLED_POWER_ARM_TARGET_OFFSET_LIMIT
                    + COUPLED_POWER_ARM_TARGET_OFFSET_TOLERANCE
                )
            )
            if bool(invalid_coupled_success.any()):
                raise RuntimeError(
                    "coupled power-close success violates pose escape, completed-ALIGN, "
                    "or bounded arm-offset contract"
                )
        return {**events, **power_events, **coupled_events}

    close_events = {
        name: _require_vector(
            f"pick_tool_terminal[{name!r}]",
            raw.get(name),
            num_envs=num_envs,
            device=terminated.device,
            dtype=torch.long if name == "close_option_stable_steps" else torch.bool,
        )
        for name in CLOSE_OPTION_EVENT_KEYS
    }
    if not torch.equal(close_events["full_task_success"], full_task_success):
        raise RuntimeError("close-option payload contains inconsistent full_task_success aliases")
    if not torch.equal(events["success"], close_events["close_option_success"]):
        raise RuntimeError("close-option generic success alias does not match close_option_success")
    if not torch.equal(events["failure"], close_events["close_option_failure"]):
        raise RuntimeError("close-option generic failure alias does not match close_option_failure")
    if not torch.equal(events["time_out"], close_events["close_option_timeout"]):
        raise RuntimeError("close-option generic timeout alias does not match close_option_timeout")
    if bool(
        (
            close_events["close_option_success"]
            & (events["dropped"] | events["unsafe_force"])
        ).any()
    ):
        raise RuntimeError("close-option success cannot coexist with drop or unsafe force")

    close_failure_sources = (
        events["dropped"]
        | events["unsafe_force"]
        | close_events["close_option_unlatched_lift"]
        | close_events["close_option_horizontal_escape"]
        | close_events["close_option_lost_window"]
    ) & ~close_events["close_option_success"]
    if not torch.equal(close_events["close_option_failure"], close_failure_sources):
        raise RuntimeError(
            "close-option failure must be exactly drop, unsafe force, pre-latch lift, "
            "horizontal escape, or lost pregrasp window"
        )
    if bool(
        (
            close_events["close_option_success"]
            & (close_events["close_option_stable_steps"] < close_option_confirm_steps)
        ).any()
    ):
        raise RuntimeError(
            "close-option success reported before the stable-latch confirmation window"
        )
    return {**events, **close_events}


def validate_arm_hold_handoff_state(
    info: Mapping[str, Any],
    *,
    num_envs: int,
    device: torch.device,
    confirm_steps: int = 15,
) -> dict[str, torch.Tensor]:
    """Validate the monotonic close-to-lift supervisor state in terminal truth."""

    if confirm_steps < 1:
        raise ValueError("confirm_steps must be positive")
    raw = _terminal_mapping(info)
    state = {
        "arm_hold_released": _require_vector(
            "pick_tool_terminal['arm_hold_released']",
            raw.get("arm_hold_released"),
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        ),
        "arm_hold_stable_steps": _require_vector(
            "pick_tool_terminal['arm_hold_stable_steps']",
            raw.get("arm_hold_stable_steps"),
            num_envs=num_envs,
            device=device,
            dtype=torch.long,
        ),
        "arm_hold_release_other_contacts": _require_vector(
            "pick_tool_terminal['arm_hold_release_other_contacts']",
            raw.get("arm_hold_release_other_contacts"),
            num_envs=num_envs,
            device=device,
            dtype=torch.long,
        ),
    }
    for name in (
        "arm_hold_release_grasp_quality",
        "arm_hold_release_wrap_quality",
        "arm_hold_release_max_force",
    ):
        state[name] = _require_vector(
            f"pick_tool_terminal[{name!r}]",
            raw.get(name),
            num_envs=num_envs,
            device=device,
            dtype=torch.float32,
        )
    stable_steps = state["arm_hold_stable_steps"]
    released = state["arm_hold_released"]
    if bool((stable_steps < 0).any()):
        raise RuntimeError("arm-hold stable-frame count cannot be negative")
    if bool((released & (stable_steps < confirm_steps)).any()):
        raise RuntimeError("arm hold released before the stable-grasp confirmation window")
    other_contacts = state["arm_hold_release_other_contacts"]
    if bool((released & ((other_contacts < 2) | (other_contacts > 4))).any()):
        raise RuntimeError("released arm-hold state has an invalid opposed-finger count")
    if bool(((~released) & (other_contacts != -1)).any()):
        raise RuntimeError("unreleased arm-hold state contains a release contact snapshot")
    for name in ARM_HOLD_HANDOFF_KEYS[3:]:
        if not bool(torch.isfinite(state[name]).all()):
            raise FloatingPointError(f"{name} contains NaN or infinity")
    return state


@dataclass(frozen=True)
class PhysicalTruth:
    """Per-environment physical state used by strict episode metrics."""

    clearance: torch.Tensor
    grasped: torch.Tensor

    def validate(self, *, num_envs: int, device: torch.device, name: str) -> None:
        _require_vector(
            f"{name}.clearance",
            self.clearance,
            num_envs=num_envs,
            device=device,
            dtype=torch.float32,
        )
        _require_vector(
            f"{name}.grasped",
            self.grasped,
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        )
        if not bool(torch.isfinite(self.clearance).all()):
            raise FloatingPointError(f"{name}.clearance contains NaN or infinity")

def physical_truth_from_terminal_info(
    info: Mapping[str, Any],
    *,
    num_envs: int,
    device: torch.device,
    task_mode: str = FULL_TASK_MODE,
    close_option_confirm_steps: int = 15,
    close_option_min_hold_quality: float = 0.5,
    grasp_quality_threshold: float = 0.35,
    safe_force_limit: float = 30.0,
) -> PhysicalTruth:
    """Read per-step physical truth cloned by ``_get_dones`` before reset."""

    task_mode = _validate_task_mode(task_mode)
    if close_option_confirm_steps < 1:
        raise ValueError("close_option_confirm_steps must be positive")
    for name, value in (
        ("close_option_min_hold_quality", close_option_min_hold_quality),
        ("grasp_quality_threshold", grasp_quality_threshold),
        ("safe_force_limit", safe_force_limit),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    raw = _terminal_mapping(info)
    legacy_grasped = _require_vector(
        "pick_tool_terminal['is_grasped']",
        raw.get("is_grasped"),
        num_envs=num_envs,
        device=device,
        dtype=torch.bool,
    )
    selected_grasped = (
        _require_vector(
            "pick_tool_terminal['power_is_grasped']",
            raw.get("power_is_grasped"),
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        )
        if task_mode in POWER_CLOSE_TASK_MODES
        else legacy_grasped
    )
    truth = PhysicalTruth(
        _require_vector(
            "pick_tool_terminal['true_clearance']",
            raw.get("true_clearance"),
            num_envs=num_envs,
            device=device,
            dtype=torch.float32,
        ),
        selected_grasped,
    )
    truth.validate(num_envs=num_envs, device=device, name="pick_tool_terminal")

    generic_success = raw.get("success")
    full_task_success_raw = raw.get(
        "full_task_success",
        generic_success if task_mode == FULL_TASK_MODE else None,
    )
    full_task_success = _require_vector(
        "pick_tool_terminal['full_task_success']",
        full_task_success_raw,
        num_envs=num_envs,
        device=device,
        dtype=torch.bool,
    )
    # This invariant remains active even while evaluating the close option.  It
    # protects the explicitly named full-task signal without imposing 20 cm on
    # the independent close-option success event.
    if bool((full_task_success & (truth.clearance < 0.20 - 1.0e-6)).any()):
        raise RuntimeError("full-task success reported below 20 cm true mesh clearance")
    if bool((full_task_success & ~legacy_grasped).any()):
        raise RuntimeError("full-task success reported without the grasp latch")

    if task_mode == FULL_TASK_MODE:
        if isinstance(generic_success, torch.Tensor) and not torch.equal(
            generic_success, full_task_success
        ):
            raise RuntimeError("full-task generic success alias is inconsistent")
        return truth

    if task_mode in POWER_CLOSE_TASK_MODES:
        power_success = _require_vector(
            "pick_tool_terminal['power_close_option_success']",
            raw.get("power_close_option_success"),
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        )
        stable_steps = _require_vector(
            "pick_tool_terminal['power_close_option_stable_steps']",
            raw.get("power_close_option_stable_steps"),
            num_envs=num_envs,
            device=device,
            dtype=torch.long,
        )
        thumb_contact = _require_vector(
            "pick_tool_terminal['power_thumb_contact']",
            raw.get("power_thumb_contact"),
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        )
        other_contacts = _require_vector(
            "pick_tool_terminal['power_legal_other_contact_count']",
            raw.get("power_legal_other_contact_count"),
            num_envs=num_envs,
            device=device,
            dtype=torch.long,
        )
        power_quality = _require_vector(
            "pick_tool_terminal['power_grasp_quality']",
            raw.get("power_grasp_quality"),
            num_envs=num_envs,
            device=device,
            dtype=torch.float32,
        )
        hold_quality = _require_vector(
            "pick_tool_terminal['hold_quality']",
            raw.get("hold_quality"),
            num_envs=num_envs,
            device=device,
            dtype=torch.float32,
        )
        max_force = _require_vector(
            "pick_tool_terminal['max_force']",
            raw.get("max_force"),
            num_envs=num_envs,
            device=device,
            dtype=torch.float32,
        )
        dropped = _require_vector(
            "pick_tool_terminal['dropped']",
            raw.get("dropped"),
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        )
        unsafe_force = _require_vector(
            "pick_tool_terminal['unsafe_force']",
            raw.get("unsafe_force"),
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        )
        generic_success = _require_vector(
            "pick_tool_terminal['success']",
            raw.get("success"),
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        )
        if not torch.equal(generic_success, power_success):
            raise RuntimeError("power-close generic success alias is inconsistent")
        invalid_power = power_success & (
            (~truth.grasped)
            | (~thumb_contact)
            | (other_contacts < 3)
            | (stable_steps < close_option_confirm_steps)
            | (power_quality < grasp_quality_threshold)
            | (hold_quality < close_option_min_hold_quality)
            | (max_force > safe_force_limit)
            | dropped
            | unsafe_force
        )
        if bool(invalid_power.any()):
            raise RuntimeError("power-close success violates its physical latch contract")
        if task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
            coupled = validate_coupled_power_telemetry(
                info,
                num_envs=num_envs,
                device=device,
            )
            if not torch.allclose(
                coupled["coupled_power_true_clearance"],
                truth.clearance,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise RuntimeError(
                    "coupled true-clearance telemetry disagrees with physical truth"
                )
            invalid_coupled = power_success & (
                coupled["coupled_power_pose_escape"]
                | coupled["coupled_power_align_active"]
                | (
                    coupled["coupled_power_arm_target_offset_abs_max"]
                    > COUPLED_POWER_ARM_TARGET_OFFSET_LIMIT
                    + COUPLED_POWER_ARM_TARGET_OFFSET_TOLERANCE
                )
            )
            if bool(invalid_coupled.any()):
                raise RuntimeError(
                    "coupled power-close success violates pose, completed-ALIGN, "
                    "or arm-offset truth"
                )
        return truth

    close_success = _require_vector(
        "pick_tool_terminal['close_option_success']",
        raw.get("close_option_success"),
        num_envs=num_envs,
        device=device,
        dtype=torch.bool,
    )
    stable_steps = _require_vector(
        "pick_tool_terminal['close_option_stable_steps']",
        raw.get("close_option_stable_steps"),
        num_envs=num_envs,
        device=device,
        dtype=torch.long,
    )
    grasp_quality = _require_vector(
        "pick_tool_terminal['grasp_quality']",
        raw.get("grasp_quality"),
        num_envs=num_envs,
        device=device,
        dtype=torch.float32,
    )
    hold_quality = _require_vector(
        "pick_tool_terminal['hold_quality']",
        raw.get("hold_quality"),
        num_envs=num_envs,
        device=device,
        dtype=torch.float32,
    )
    max_force = _require_vector(
        "pick_tool_terminal['max_force']",
        raw.get("max_force"),
        num_envs=num_envs,
        device=device,
        dtype=torch.float32,
    )
    dropped = _require_vector(
        "pick_tool_terminal['dropped']",
        raw.get("dropped"),
        num_envs=num_envs,
        device=device,
        dtype=torch.bool,
    )
    unsafe_force = _require_vector(
        "pick_tool_terminal['unsafe_force']",
        raw.get("unsafe_force"),
        num_envs=num_envs,
        device=device,
        dtype=torch.bool,
    )
    invalid_close = close_success & (
        (~truth.grasped)
        | dropped
        | unsafe_force
        | (stable_steps < close_option_confirm_steps)
        | (grasp_quality < grasp_quality_threshold)
        | (hold_quality < close_option_min_hold_quality)
        | (max_force > safe_force_limit)
    )
    if bool(invalid_close.any()):
        raise RuntimeError("close-option success violates the stable, safe latch contract")
    return truth


def _read_physical_truth(
    unwrapped: Any, *, task_mode: str = FULL_TASK_MODE
) -> PhysicalTruth:
    """Read true mesh clearance and the strict task latch from PickTool."""

    task_mode = _validate_task_mode(task_mode)
    unwrapped._compute_intermediate_values()
    clearance = (unwrapped._object_true_min_z() - unwrapped._table_surface_z).detach().clone()
    grasped = (
        unwrapped._power_is_grasped.detach().clone()
        if task_mode in POWER_CLOSE_TASK_MODES
        else unwrapped._is_grasped.detach().clone()
    )
    truth = PhysicalTruth(clearance.to(dtype=torch.float32), grasped.to(dtype=torch.bool))
    truth.validate(
        num_envs=int(unwrapped.num_envs),
        device=torch.device(str(unwrapped.device)),
        name="runtime_truth",
    )
    return truth


def episode_quotas(episodes: int, num_envs: int, *, device: torch.device) -> torch.Tensor:
    """Assign an exact, deterministic number of episodes to every env slot."""

    if episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num_envs must be positive")
    base, remainder = divmod(episodes, num_envs)
    quotas = torch.full((num_envs,), base, dtype=torch.long, device=device)
    quotas[:remainder] += 1
    return quotas


class StrictEpisodeTracker:
    """Track exact per-episode physical events across vector auto-resets."""

    def __init__(
        self,
        *,
        episodes: int,
        num_envs: int,
        device: torch.device,
        initial_truth: PhysicalTruth,
        task_mode: str = FULL_TASK_MODE,
        track_arm_hold_handoff: bool = False,
    ) -> None:
        initial_truth.validate(num_envs=num_envs, device=device, name="initial_truth")
        self.task_mode = _validate_task_mode(task_mode)
        self.track_arm_hold_handoff = bool(track_arm_hold_handoff)
        self.episodes = episodes
        self.num_envs = num_envs
        self.device = device
        self.quotas = episode_quotas(episodes, num_envs, device=device)
        self.completed_by_slot = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_clearance = initial_truth.clearance.clone()
        self.ever_grasped = initial_truth.grasped.clone()
        self.ever_5cm = initial_truth.clearance >= 0.05
        self.ever_20cm = initial_truth.clearance >= 0.20
        self.ever_unlatched_5cm = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ever_arm_hold_released = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.max_arm_hold_stable_steps = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.arm_hold_release_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )
        # Power-close episode accumulators are separate from legacy
        # ``ever_grasped``.  The latter is physical-truth plumbing shared with
        # older reports, while these tensors have thumb+three semantics only.
        self.max_power_legal_other_contact_count = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.ever_power_thumb_contact = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.ever_power_thumb_plus_three = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.ever_power_grasp_latched = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.max_power_staged_close_quality = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self.max_power_wrap_quality = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self.max_power_grasp_quality = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self.max_power_latch_confirm_steps = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.max_power_close_option_stable_steps = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.ever_coupled_pose_escape = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.ever_coupled_arm_target_saturated = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.ever_coupled_align_active = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.max_coupled_rotation_drift = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self.max_coupled_xy_drift = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self.max_coupled_true_clearance = initial_truth.clearance.clone()
        self.max_coupled_arm_target_offset = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        self.records: list[dict[str, Any]] = []

    @property
    def active(self) -> torch.Tensor:
        return self.completed_by_slot < self.quotas

    @property
    def complete(self) -> bool:
        return len(self.records) == self.episodes

    def step(
        self,
        *,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        events: Mapping[str, torch.Tensor],
        transition_truth: PhysicalTruth,
        post_reset_truth: PhysicalTruth,
    ) -> None:
        _require_vector(
            "reward",
            reward,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )
        _require_vector(
            "terminated",
            terminated,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        _require_vector(
            "truncated",
            truncated,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        transition_truth.validate(
            num_envs=self.num_envs, device=self.device, name="transition_truth"
        )
        post_reset_truth.validate(
            num_envs=self.num_envs, device=self.device, name="post_reset_truth"
        )
        for name in TERMINAL_EVENT_KEYS:
            _require_vector(
                f"events[{name!r}]",
                events.get(name),
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )
        if self.task_mode in CLOSE_OPTION_TASK_MODES:
            option_keys = (
                CLOSE_OPTION_EVENT_KEYS
                if self.task_mode == CLOSE_OPTION_MODE
                else POWER_CLOSE_OPTION_EVENT_KEYS
            )
            long_keys = {
                "close_option_stable_steps",
                "power_close_option_stable_steps",
                "power_legal_other_contact_count",
                "power_grasp_latch_confirm_steps",
            }
            float_keys = {
                "power_close_quality",
                "power_wrap_quality",
                "power_grasp_quality",
            }
            for name in option_keys:
                _require_vector(
                    f"events[{name!r}]",
                    events.get(name),
                    num_envs=self.num_envs,
                    device=self.device,
                    dtype=(
                        torch.long
                        if name in long_keys
                        else torch.float32
                        if name in float_keys
                        else torch.bool
                    ),
                )
            if self.task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
                for name in COUPLED_POWER_TELEMETRY_KEYS:
                    _require_vector(
                        f"events[{name!r}]",
                        events.get(name),
                        num_envs=self.num_envs,
                        device=self.device,
                        dtype=(
                            torch.bool
                            if name
                            in {
                                "coupled_power_pose_escape",
                                "coupled_power_arm_target_saturated",
                                "coupled_power_align_active",
                            }
                            else torch.float32
                        ),
                    )
        if self.track_arm_hold_handoff:
            for name, dtype in (
                ("arm_hold_released", torch.bool),
                ("arm_hold_stable_steps", torch.long),
                ("arm_hold_release_other_contacts", torch.long),
                ("arm_hold_release_grasp_quality", torch.float32),
                ("arm_hold_release_wrap_quality", torch.float32),
                ("arm_hold_release_max_force", torch.float32),
            ):
                _require_vector(
                    f"events[{name!r}]",
                    events.get(name),
                    num_envs=self.num_envs,
                    device=self.device,
                    dtype=dtype,
                )

        active = self.active
        self.returns.add_(torch.where(active, reward, 0.0))
        self.lengths.add_(active.long())
        self.max_clearance = torch.where(
            active,
            torch.maximum(self.max_clearance, transition_truth.clearance),
            self.max_clearance,
        )
        self.ever_grasped |= active & transition_truth.grasped
        self.ever_5cm |= active & (transition_truth.clearance >= 0.05)
        self.ever_20cm |= active & (transition_truth.clearance >= 0.20)
        self.ever_unlatched_5cm |= active & events["unlatched_clearance_ge_5cm"]
        if self.task_mode in POWER_CLOSE_TASK_MODES:
            thumb_contact = events["power_thumb_contact"]
            legal_other = events["power_legal_other_contact_count"]
            self.max_power_legal_other_contact_count = torch.where(
                active,
                torch.maximum(self.max_power_legal_other_contact_count, legal_other),
                self.max_power_legal_other_contact_count,
            )
            self.ever_power_thumb_contact |= active & thumb_contact
            self.ever_power_thumb_plus_three |= active & thumb_contact & (legal_other >= 3)
            self.ever_power_grasp_latched |= active & events["power_is_grasped"]
            for accumulator, event_name in (
                (self.max_power_staged_close_quality, "power_close_quality"),
                (self.max_power_wrap_quality, "power_wrap_quality"),
                (self.max_power_grasp_quality, "power_grasp_quality"),
                (
                    self.max_power_latch_confirm_steps,
                    "power_grasp_latch_confirm_steps",
                ),
                (
                    self.max_power_close_option_stable_steps,
                    "power_close_option_stable_steps",
                ),
            ):
                accumulator.copy_(
                    torch.where(
                        active,
                        torch.maximum(accumulator, events[event_name]),
                        accumulator,
                    )
                )
        if self.task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
            self.ever_coupled_pose_escape |= (
                active & events["coupled_power_pose_escape"]
            )
            self.ever_coupled_arm_target_saturated |= (
                active & events["coupled_power_arm_target_saturated"]
            )
            self.ever_coupled_align_active |= (
                active & events["coupled_power_align_active"]
            )
            for accumulator, event_name in (
                (self.max_coupled_rotation_drift, "coupled_power_rotation_drift"),
                (self.max_coupled_xy_drift, "coupled_power_xy_drift"),
                (self.max_coupled_true_clearance, "coupled_power_true_clearance"),
                (
                    self.max_coupled_arm_target_offset,
                    "coupled_power_arm_target_offset_abs_max",
                ),
            ):
                accumulator.copy_(
                    torch.where(
                        active,
                        torch.maximum(accumulator, events[event_name]),
                        accumulator,
                    )
                )
        if self.track_arm_hold_handoff:
            first_released = (
                active
                & events["arm_hold_released"]
                & (~self.ever_arm_hold_released)
            )
            self.arm_hold_release_step = torch.where(
                first_released,
                self.lengths,
                self.arm_hold_release_step,
            )
            self.ever_arm_hold_released |= active & events["arm_hold_released"]
            self.max_arm_hold_stable_steps = torch.where(
                active,
                torch.maximum(
                    self.max_arm_hold_stable_steps,
                    events["arm_hold_stable_steps"],
                ),
                self.max_arm_hold_stable_steps,
            )

        accepted_done = active & (terminated | truncated)
        ids = accepted_done.nonzero(as_tuple=False).squeeze(-1)
        for env_id in ids.detach().cpu().tolist():
            record: dict[str, Any] = {
                "episode_index": len(self.records),
                "env_slot": env_id,
                "slot_episode_index": int(self.completed_by_slot[env_id].item()),
                "return": float(self.returns[env_id].item()),
                "length": int(self.lengths[env_id].item()),
                "max_true_clearance_m": float(self.max_clearance[env_id].item()),
                "ever_grasped": bool(self.ever_grasped[env_id].item()),
                "ever_clearance_ge_5cm": bool(self.ever_5cm[env_id].item()),
                "ever_clearance_ge_20cm": bool(self.ever_20cm[env_id].item()),
                "dropped": bool(events["dropped"][env_id].item()),
                "unsafe_force": bool(events["unsafe_force"][env_id].item()),
                "ever_unlatched_clearance_ge_5cm": bool(
                    self.ever_unlatched_5cm[env_id].item()
                ),
            }
            if self.task_mode == FULL_TASK_MODE:
                record.update(
                    {
                        "success": bool(events["success"][env_id].item()),
                        "failure": bool(events["failure"][env_id].item()),
                        "time_out": bool(events["time_out"][env_id].item()),
                    }
                )
            elif self.task_mode == CLOSE_OPTION_MODE:
                record.update(
                    {
                        "close_option_success": bool(
                            events["close_option_success"][env_id].item()
                        ),
                        "close_option_failure": bool(
                            events["close_option_failure"][env_id].item()
                        ),
                        "close_option_timeout": bool(
                            events["close_option_timeout"][env_id].item()
                        ),
                        "close_option_unlatched_lift": bool(
                            events["close_option_unlatched_lift"][env_id].item()
                        ),
                        "close_option_horizontal_escape": bool(
                            events["close_option_horizontal_escape"][env_id].item()
                        ),
                        "close_option_lost_window": bool(
                            events["close_option_lost_window"][env_id].item()
                        ),
                        "close_option_stable_steps": int(
                            events["close_option_stable_steps"][env_id].item()
                        ),
                    }
                )
            else:
                record.update(
                    {
                        "power_close_option_success": bool(
                            events["power_close_option_success"][env_id].item()
                        ),
                        "power_close_option_failure": bool(
                            events["power_close_option_failure"][env_id].item()
                        ),
                        "power_close_option_timeout": bool(
                            events["power_close_option_timeout"][env_id].item()
                        ),
                        "close_option_unlatched_lift": bool(
                            events["close_option_unlatched_lift"][env_id].item()
                        ),
                        "close_option_horizontal_escape": bool(
                            events["close_option_horizontal_escape"][env_id].item()
                        ),
                        "close_option_lost_window": bool(
                            events["close_option_lost_window"][env_id].item()
                        ),
                        "power_close_option_stable_steps": int(
                            events["power_close_option_stable_steps"][env_id].item()
                        ),
                        "power_is_grasped": bool(
                            events["power_is_grasped"][env_id].item()
                        ),
                        "power_thumb_contact": bool(
                            events["power_thumb_contact"][env_id].item()
                        ),
                        "power_legal_other_contact_count": int(
                            events["power_legal_other_contact_count"][env_id].item()
                        ),
                        "power_close_quality": float(
                            events["power_close_quality"][env_id].item()
                        ),
                        "power_wrap_quality": float(
                            events["power_wrap_quality"][env_id].item()
                        ),
                        "power_grasp_quality": float(
                            events["power_grasp_quality"][env_id].item()
                        ),
                        "power_grasp_latch_confirm_steps": int(
                            events["power_grasp_latch_confirm_steps"][env_id].item()
                        ),
                        # Explicit terminal aliases preserve the old fields
                        # above while making their reset-before/last-frame
                        # semantics unambiguous to downstream analysis.
                        "terminal_power_is_grasped": bool(
                            events["power_is_grasped"][env_id].item()
                        ),
                        "terminal_power_thumb_contact": bool(
                            events["power_thumb_contact"][env_id].item()
                        ),
                        "terminal_power_legal_other_contact_count": int(
                            events["power_legal_other_contact_count"][env_id].item()
                        ),
                        "terminal_power_staged_close_quality": float(
                            events["power_close_quality"][env_id].item()
                        ),
                        "terminal_power_wrap_quality": float(
                            events["power_wrap_quality"][env_id].item()
                        ),
                        "terminal_power_grasp_quality": float(
                            events["power_grasp_quality"][env_id].item()
                        ),
                        "terminal_power_grasp_latch_confirm_steps": int(
                            events["power_grasp_latch_confirm_steps"][env_id].item()
                        ),
                        "terminal_power_close_option_stable_steps": int(
                            events["power_close_option_stable_steps"][env_id].item()
                        ),
                        "max_power_legal_other_contact_count": int(
                            self.max_power_legal_other_contact_count[env_id].item()
                        ),
                        "ever_power_thumb_contact": bool(
                            self.ever_power_thumb_contact[env_id].item()
                        ),
                        "ever_power_thumb_plus_three": bool(
                            self.ever_power_thumb_plus_three[env_id].item()
                        ),
                        "ever_power_grasp_latched": bool(
                            self.ever_power_grasp_latched[env_id].item()
                        ),
                        "max_power_staged_close_quality": float(
                            self.max_power_staged_close_quality[env_id].item()
                        ),
                        "max_power_wrap_quality": float(
                            self.max_power_wrap_quality[env_id].item()
                        ),
                        "max_power_grasp_quality": float(
                            self.max_power_grasp_quality[env_id].item()
                        ),
                        "max_power_grasp_latch_confirm_steps": int(
                            self.max_power_latch_confirm_steps[env_id].item()
                        ),
                        "max_power_close_option_stable_steps": int(
                            self.max_power_close_option_stable_steps[env_id].item()
                        ),
                    }
                )
                if self.task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
                    record.update(
                        {
                            "terminal_coupled_power_pose_escape": bool(
                                events["coupled_power_pose_escape"][env_id].item()
                            ),
                            "terminal_coupled_power_rotation_drift_rad": float(
                                events["coupled_power_rotation_drift"][env_id].item()
                            ),
                            "terminal_coupled_power_xy_drift_m": float(
                                events["coupled_power_xy_drift"][env_id].item()
                            ),
                            "terminal_coupled_power_true_clearance_m": float(
                                events["coupled_power_true_clearance"][env_id].item()
                            ),
                            "terminal_coupled_power_arm_target_offset_abs_max_rad": float(
                                events[
                                    "coupled_power_arm_target_offset_abs_max"
                                ][env_id].item()
                            ),
                            "terminal_coupled_power_arm_target_saturated": bool(
                                events[
                                    "coupled_power_arm_target_saturated"
                                ][env_id].item()
                            ),
                            "terminal_coupled_power_align_active": bool(
                                events["coupled_power_align_active"][env_id].item()
                            ),
                            "ever_coupled_power_pose_escape": bool(
                                self.ever_coupled_pose_escape[env_id].item()
                            ),
                            "ever_coupled_power_arm_target_saturated": bool(
                                self.ever_coupled_arm_target_saturated[env_id].item()
                            ),
                            "ever_coupled_power_align_active": bool(
                                self.ever_coupled_align_active[env_id].item()
                            ),
                            "max_coupled_power_rotation_drift_rad": float(
                                self.max_coupled_rotation_drift[env_id].item()
                            ),
                            "max_coupled_power_xy_drift_m": float(
                                self.max_coupled_xy_drift[env_id].item()
                            ),
                            "max_coupled_power_true_clearance_m": float(
                                self.max_coupled_true_clearance[env_id].item()
                            ),
                            "max_coupled_power_arm_target_offset_abs_max_rad": float(
                                self.max_coupled_arm_target_offset[env_id].item()
                            ),
                        }
                    )
            if self.track_arm_hold_handoff:
                record.update(
                    {
                        "arm_hold_released": bool(
                            self.ever_arm_hold_released[env_id].item()
                        ),
                        "max_arm_hold_stable_steps": int(
                            self.max_arm_hold_stable_steps[env_id].item()
                        ),
                        "arm_hold_release_step": int(
                            self.arm_hold_release_step[env_id].item()
                        ),
                        "arm_hold_release_other_contacts": int(
                            events["arm_hold_release_other_contacts"][env_id].item()
                        ),
                        "arm_hold_release_grasp_quality": float(
                            events["arm_hold_release_grasp_quality"][env_id].item()
                        ),
                        "arm_hold_release_wrap_quality": float(
                            events["arm_hold_release_wrap_quality"][env_id].item()
                        ),
                        "arm_hold_release_max_force_n": float(
                            events["arm_hold_release_max_force"][env_id].item()
                        ),
                    }
                )
            if not math.isfinite(record["return"]) or not math.isfinite(
                record["max_true_clearance_m"]
            ):
                raise FloatingPointError("completed episode contains a non-finite metric")
            self.records.append(record)

        self.completed_by_slot.add_(accepted_done.long())
        # DirectRLEnv already started the next episode on these rows.  Seed its
        # accumulator with that reset state's physical truth, not the old
        # transition terminal state.
        self.returns.masked_fill_(accepted_done, 0.0)
        self.lengths.masked_fill_(accepted_done, 0)
        self.max_clearance = torch.where(
            accepted_done, post_reset_truth.clearance, self.max_clearance
        )
        self.ever_grasped = torch.where(
            accepted_done, post_reset_truth.grasped, self.ever_grasped
        )
        self.ever_5cm = torch.where(
            accepted_done, post_reset_truth.clearance >= 0.05, self.ever_5cm
        )
        self.ever_20cm = torch.where(
            accepted_done, post_reset_truth.clearance >= 0.20, self.ever_20cm
        )
        self.ever_unlatched_5cm.masked_fill_(accepted_done, False)
        self.ever_arm_hold_released.masked_fill_(accepted_done, False)
        self.max_arm_hold_stable_steps.masked_fill_(accepted_done, 0)
        self.arm_hold_release_step.masked_fill_(accepted_done, -1)
        self.max_power_legal_other_contact_count.masked_fill_(accepted_done, 0)
        self.ever_power_thumb_contact.masked_fill_(accepted_done, False)
        self.ever_power_thumb_plus_three.masked_fill_(accepted_done, False)
        self.ever_power_grasp_latched.masked_fill_(accepted_done, False)
        self.max_power_staged_close_quality.masked_fill_(accepted_done, 0.0)
        self.max_power_wrap_quality.masked_fill_(accepted_done, 0.0)
        self.max_power_grasp_quality.masked_fill_(accepted_done, 0.0)
        self.max_power_latch_confirm_steps.masked_fill_(accepted_done, 0)
        self.max_power_close_option_stable_steps.masked_fill_(accepted_done, 0)
        self.ever_coupled_pose_escape.masked_fill_(accepted_done, False)
        self.ever_coupled_arm_target_saturated.masked_fill_(accepted_done, False)
        self.ever_coupled_align_active.masked_fill_(accepted_done, False)
        self.max_coupled_rotation_drift.masked_fill_(accepted_done, 0.0)
        self.max_coupled_xy_drift.masked_fill_(accepted_done, 0.0)
        self.max_coupled_true_clearance = torch.where(
            accepted_done,
            post_reset_truth.clearance,
            self.max_coupled_true_clearance,
        )
        self.max_coupled_arm_target_offset.masked_fill_(accepted_done, 0.0)

        if len(self.records) > self.episodes:
            raise RuntimeError("episode tracker exceeded its exact episode quota")


def summarize(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("summary input contains NaN or infinity")
    quantiles = torch.quantile(
        tensor,
        torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=tensor.dtype),
    )
    return {
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "max": float(quantiles[4]),
        "mean": float(tensor.mean()),
    }


def build_strict_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint: Path,
    architecture: str,
    seed: int,
    num_envs: int,
    vector_steps: int,
    max_vector_steps: int,
    episode_length_s: float,
    max_episode_steps: int,
    curriculum_dataset: Path | None,
    curriculum_dataset_sha256: str | None,
    curriculum_boundary: str,
    curriculum_probability: float,
    curriculum_joint_noise: float,
    use_compile: bool,
    upstream_commit: str,
    task_mode: str = FULL_TASK_MODE,
    close_option_confirm_steps: int = 15,
    close_option_grasp_quality_threshold: float = 0.35,
    close_option_min_hold_quality: float = 0.5,
    close_option_safe_force_limit: float = 30.0,
    close_option_unlatched_lift_limit: float = 0.015,
    close_option_horizontal_drift_limit: float = 0.03,
    close_option_min_proximity: float = 0.01,
    close_option_lost_window_steps: int = 12,
    checkpoint_task_mode: str = FULL_TASK_MODE,
    cross_task_actor_evaluation: bool = False,
    policy_action_dim: int = ACTION_DIM,
    environment_action_dim: int = ACTION_DIM,
    policy_action_layout: str = FULL_POLICY_ACTION_LAYOUT,
    action_projection: str = IDENTITY_ACTION_PROJECTION,
    observation_dim: int = OBSERVATION_DIM,
    observation_contract: str = STANDARD_OBSERVATION_CONTRACT,
    noise_group_specs: Sequence[tuple[str, int, int, float, float, int]] = NOISE_GROUP_SPECS,
    power_required_other_contacts: int = 3,
) -> dict[str, Any]:
    if not records:
        raise ValueError("strict evaluation completed no episodes")
    task_mode = _validate_task_mode(task_mode)
    checkpoint_task_mode = _validate_task_mode(checkpoint_task_mode)
    expected_policy = requested_policy_action_contract(task_mode)
    reported_policy = {
        "policy_action_dim": policy_action_dim,
        "policy_action_layout": policy_action_layout,
        "environment_action_dim": environment_action_dim,
        "action_projection": action_projection,
        "observation_dim": observation_dim,
        "observation_contract": observation_contract,
    }
    if reported_policy != expected_policy:
        raise ValueError(
            "evaluation metrics policy contract disagrees with task mode: "
            f"reported={reported_policy}, expected={expected_policy}"
        )
    expected_cross_task = checkpoint_task_mode != task_mode
    if bool(cross_task_actor_evaluation) != expected_cross_task:
        raise ValueError(
            "cross_task_actor_evaluation is inconsistent with checkpoint/requested task modes"
        )
    if task_mode == FULL_TASK_MODE:
        event_names = TERMINAL_EVENT_KEYS[:-1]
    elif task_mode == CLOSE_OPTION_MODE:
        event_names = (
            "close_option_success",
            "close_option_failure",
            "close_option_timeout",
            "dropped",
            "unsafe_force",
            "close_option_unlatched_lift",
            "close_option_horizontal_escape",
            "close_option_lost_window",
        )
    else:
        event_names = (
            "power_close_option_success",
            "power_close_option_failure",
            "power_close_option_timeout",
            "dropped",
            "unsafe_force",
            "close_option_unlatched_lift",
            "close_option_horizontal_escape",
            "close_option_lost_window",
        )
    event_counts = {
        name: sum(bool(record[name]) for record in records) for name in event_names
    }
    if task_mode == FULL_TASK_MODE:
        event_counts["unlatched_clearance_ge_5cm"] = sum(
            bool(record["ever_unlatched_clearance_ge_5cm"]) for record in records
        )
    funnel = {
        "ever_grasped": sum(bool(record["ever_grasped"]) for record in records),
        "ever_clearance_ge_5cm": sum(
            bool(record["ever_clearance_ge_5cm"]) for record in records
        ),
        "ever_clearance_ge_20cm": sum(
            bool(record["ever_clearance_ge_20cm"]) for record in records
        ),
    }
    if all("arm_hold_released" in record for record in records):
        funnel["arm_hold_released"] = sum(
            bool(record["arm_hold_released"]) for record in records
        )
    episodes = len(records)
    metrics: dict[str, Any] = {
        "status": "complete",
        "task_mode": task_mode,
        "checkpoint_task_mode": checkpoint_task_mode,
        "cross_task_actor_evaluation": bool(cross_task_actor_evaluation),
        "checkpoint": str(checkpoint.resolve()),
        "flashsac_upstream_commit": upstream_commit,
        "policy": "deterministic_tanh_actor_mean",
        "architecture": architecture,
        "critic_bins": (
            PRODUCTION_CRITIC_BINS if architecture == "production" else SMOKE_CRITIC_BINS
        ),
        "noise_groups": [
            {
                "name": name,
                "start": start,
                "stop": stop,
                "scale": scale,
                "zeta_mu": zeta_mu,
                "zeta_max": zeta_max,
            }
            for name, start, stop, scale, zeta_mu, zeta_max in noise_group_specs
        ],
        "use_compile": use_compile,
        "seed": seed,
        "num_envs": num_envs,
        "requested_episodes": episodes,
        "completed_episodes": episodes,
        "vector_steps": vector_steps,
        "max_vector_steps": max_vector_steps,
        "simulated_environment_steps": vector_steps * num_envs,
        "episode_length_s": episode_length_s,
        "max_episode_steps": max_episode_steps,
        "curriculum": {
            "dataset": (
                str(curriculum_dataset.resolve()) if curriculum_dataset is not None else None
            ),
            "dataset_sha256": curriculum_dataset_sha256,
            "boundary": curriculum_boundary if curriculum_dataset is not None else None,
            "probability": curriculum_probability,
            "joint_noise": curriculum_joint_noise,
        },
        "observation_dim": observation_dim,
        "observation_contract": observation_contract,
        "action_dim": policy_action_dim,
        "policy_action_dim": policy_action_dim,
        "environment_action_dim": environment_action_dim,
        "policy_action_layout": policy_action_layout,
        "action_projection": action_projection,
        "events": event_counts,
        "funnel": funnel,
        "max_true_clearance_m": summarize(
            [float(record["max_true_clearance_m"]) for record in records]
        ),
        "episode_return": summarize([float(record["return"]) for record in records]),
        "episode_length": summarize([int(record["length"]) for record in records]),
        "episodes": list(records),
    }
    if all("max_arm_hold_stable_steps" in record for record in records):
        metrics["arm_hold_stable_steps"] = summarize(
            [int(record["max_arm_hold_stable_steps"]) for record in records]
        )
    released_records = [
        record for record in records if bool(record.get("arm_hold_released", False))
    ]
    if released_records and all("arm_hold_release_step" in record for record in records):
        metrics["arm_hold_release_step"] = summarize(
            [int(record["arm_hold_release_step"]) for record in released_records]
        )
        metrics["arm_hold_release_outcomes"] = {
            "released": len(released_records),
            "success": sum(bool(record.get("success", False)) for record in released_records),
            "failure": sum(bool(record.get("failure", False)) for record in released_records),
            "time_out": sum(bool(record.get("time_out", False)) for record in released_records),
        }
        topology: dict[str, Any] = {}
        for cohort_name, cohort in (
            ("all_released", released_records),
            ("success", [record for record in released_records if record.get("success")]),
            ("failure", [record for record in released_records if record.get("failure")]),
        ):
            if cohort:
                topology[cohort_name] = {
                    "episodes": len(cohort),
                    "other_contact_count": summarize(
                        [int(record["arm_hold_release_other_contacts"]) for record in cohort]
                    ),
                    "grasp_quality": summarize(
                        [float(record["arm_hold_release_grasp_quality"]) for record in cohort]
                    ),
                    "wrap_quality": summarize(
                        [float(record["arm_hold_release_wrap_quality"]) for record in cohort]
                    ),
                    "max_force_n": summarize(
                        [float(record["arm_hold_release_max_force_n"]) for record in cohort]
                    ),
                }
        metrics["arm_hold_release_topology"] = topology
    if task_mode in POWER_CLOSE_TASK_MODES:
        ever_thumb = sum(bool(record["ever_power_thumb_contact"]) for record in records)
        ever_thumb_plus_three = sum(
            bool(record["ever_power_thumb_plus_three"]) for record in records
        )
        ever_latched = sum(bool(record["ever_power_grasp_latched"]) for record in records)
        metrics["power_close_telemetry"] = {
            "episodes_ever_thumb_contact": ever_thumb,
            "episodes_ever_thumb_plus_three": ever_thumb_plus_three,
            "episodes_ever_grasp_latched": ever_latched,
            "ever_thumb_contact_rate": ever_thumb / episodes,
            "ever_thumb_plus_three_rate": ever_thumb_plus_three / episodes,
            "ever_grasp_latched_rate": ever_latched / episodes,
            "episode_max_legal_other_contact_count": summarize(
                [
                    int(record["max_power_legal_other_contact_count"])
                    for record in records
                ]
            ),
            "episode_max_staged_close_quality": summarize(
                [float(record["max_power_staged_close_quality"]) for record in records]
            ),
            "episode_max_wrap_quality": summarize(
                [float(record["max_power_wrap_quality"]) for record in records]
            ),
            "episode_max_grasp_quality": summarize(
                [float(record["max_power_grasp_quality"]) for record in records]
            ),
            "episode_max_grasp_latch_confirm_steps": summarize(
                [
                    int(record["max_power_grasp_latch_confirm_steps"])
                    for record in records
                ]
            ),
            "episode_max_close_option_stable_steps": summarize(
                [
                    int(record["max_power_close_option_stable_steps"])
                    for record in records
                ]
            ),
        }
    if task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
        metrics["coupled_power_telemetry"] = {
            "episodes_ever_pose_escape": sum(
                bool(record["ever_coupled_power_pose_escape"]) for record in records
            ),
            "episodes_ever_arm_target_saturated": sum(
                bool(record["ever_coupled_power_arm_target_saturated"])
                for record in records
            ),
            "episodes_ever_align_active": sum(
                bool(record["ever_coupled_power_align_active"])
                for record in records
            ),
            "episode_max_rotation_drift_rad": summarize(
                [
                    float(record["max_coupled_power_rotation_drift_rad"])
                    for record in records
                ]
            ),
            "episode_max_xy_drift_m": summarize(
                [
                    float(record["max_coupled_power_xy_drift_m"])
                    for record in records
                ]
            ),
            "episode_max_true_clearance_m": summarize(
                [
                    float(record["max_coupled_power_true_clearance_m"])
                    for record in records
                ]
            ),
            "episode_max_arm_target_offset_abs_max_rad": summarize(
                [
                    float(
                        record[
                            "max_coupled_power_arm_target_offset_abs_max_rad"
                        ]
                    )
                    for record in records
                ]
            ),
        }
    if task_mode == FULL_TASK_MODE:
        metrics["strict_success_rate"] = event_counts["success"] / episodes
    elif task_mode == CLOSE_OPTION_MODE:
        metrics["close_option_success_rate"] = (
            event_counts["close_option_success"] / episodes
        )
        metrics["success_contract"] = {
            "name": "stable_close_option_latch",
            "confirm_steps": close_option_confirm_steps,
            "min_grasp_quality": close_option_grasp_quality_threshold,
            "min_hold_quality": close_option_min_hold_quality,
            "safe_force_limit_n": close_option_safe_force_limit,
            "full_task_20cm_success": "not_evaluated",
        }
        metrics["failure_contract"] = {
            "unlatched_lift_limit_m": close_option_unlatched_lift_limit,
            "horizontal_drift_limit_m": close_option_horizontal_drift_limit,
            "min_proximity": close_option_min_proximity,
            "lost_window_steps": close_option_lost_window_steps,
            "unsafe_force_or_drop": True,
        }
    elif task_mode == POWER_CLOSE_OPTION_MODE:
        metrics["power_close_option_success_rate"] = (
            event_counts["power_close_option_success"] / episodes
        )
        metrics["success_contract"] = {
            "name": "stable_power_close_option_latch_v1",
            "confirm_steps": close_option_confirm_steps,
            "thumb_contact_required": True,
            "required_legal_other_contacts": power_required_other_contacts,
            "min_power_grasp_quality": close_option_grasp_quality_threshold,
            "min_hold_quality": close_option_min_hold_quality,
            "safe_force_limit_n": close_option_safe_force_limit,
            "full_task_20cm_success": "not_evaluated",
        }
        metrics["failure_contract"] = {
            "unlatched_lift_limit_m": close_option_unlatched_lift_limit,
            "horizontal_drift_limit_m": close_option_horizontal_drift_limit,
            "min_proximity": close_option_min_proximity,
            "lost_window_steps": close_option_lost_window_steps,
            "unsafe_force_or_drop": True,
        }
    else:
        metrics["coupled_power_align_close_option_success_rate"] = (
            event_counts["power_close_option_success"] / episodes
        )
        metrics["evaluation_scope"] = (
            "close_option_only; full-task 20 cm lift success is not evaluated"
        )
        metrics["success_contract"] = {
            "name": "stable_coupled_power_align_close_option_v1",
            "confirm_steps": close_option_confirm_steps,
            "thumb_contact_required": True,
            "required_legal_other_contacts": power_required_other_contacts,
            "min_power_grasp_quality": close_option_grasp_quality_threshold,
            "min_hold_quality": close_option_min_hold_quality,
            "safe_force_limit_n": close_option_safe_force_limit,
            "pose_escape_required_false": True,
            "max_arm_target_offset_rad": COUPLED_POWER_ARM_TARGET_OFFSET_LIMIT,
            "arm_target_offset_tolerance_rad": (
                COUPLED_POWER_ARM_TARGET_OFFSET_TOLERANCE
            ),
            "full_task_20cm_success": "not_evaluated",
        }
        metrics["failure_contract"] = {
            "unlatched_lift_limit_m": close_option_unlatched_lift_limit,
            "horizontal_drift_limit_m": close_option_horizontal_drift_limit,
            "min_proximity": close_option_min_proximity,
            "lost_window_steps": close_option_lost_window_steps,
            "coupled_pose_escape": True,
            "unsafe_force_or_drop": True,
        }
    return metrics


_COMPILED_PREFIX = "_orig_mod."
_ENCODER_BLOCK_PATTERN = re.compile(r"^encoder\.(\d+)\.w1\.w\.weight$")


def _canonical_actor_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not state:
        raise ValueError("actor network state is empty")
    prefixed = [str(key).startswith(_COMPILED_PREFIX) for key in state]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError("actor checkpoint mixes compiled and uncompiled root keys")
    canonical: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("actor network state must map string keys to tensors")
        key = raw_key.removeprefix(_COMPILED_PREFIX) if all(prefixed) else raw_key
        canonical[key] = value
    return canonical


def infer_actor_action_dim_from_state(state: Mapping[str, Any]) -> int:
    """Read and validate the actor head width without trusting sidecar metadata."""

    canonical = _canonical_actor_state(state)
    mean = canonical.get("predictor.mean_w.w.weight")
    if mean is None or mean.ndim != 2:
        raise RuntimeError("actor checkpoint is missing a compatible mean predictor")
    action_dim = int(mean.shape[0])
    if action_dim not in (HAND_ACTION_DIM, ACTION_DIM):
        raise RuntimeError(
            f"actor checkpoint action_dim={action_dim}, expected {HAND_ACTION_DIM} or {ACTION_DIM}"
        )
    for key in (
        "predictor.mean_bias",
        "predictor.std_w.w.weight",
        "predictor.std_bias",
    ):
        value = canonical.get(key)
        if value is not None and int(value.shape[0]) != action_dim:
            raise RuntimeError(
                f"actor checkpoint output tensor {key!r} disagrees with action_dim={action_dim}"
            )
    return action_dim


def infer_actor_architecture_from_state(
    state: Mapping[str, Any],
    *,
    expected_action_dim: int | None = None,
    expected_observation_dim: int | None = None,
) -> str:
    """Recognize the production and smoke architectures saved by train.py."""

    canonical = _canonical_actor_state(state)
    embed = canonical.get("embedder.w.w.weight")
    mean = canonical.get("predictor.mean_w.w.weight")
    if embed is None or mean is None or embed.ndim != 2 or mean.ndim != 2:
        raise RuntimeError("actor checkpoint is missing compatible embedder/predictor weights")
    action_dim = infer_actor_action_dim_from_state(state)
    if expected_action_dim is not None and action_dim != expected_action_dim:
        raise RuntimeError(
            "actor checkpoint action dimension disagrees with the expected contract: "
            f"actor={action_dim}, expected={expected_action_dim}"
        )
    observation_dim = int(embed.shape[1])
    allowed_observation_dims = (OBSERVATION_DIM, COUPLED_OBSERVATION_DIM)
    if observation_dim not in allowed_observation_dims:
        raise RuntimeError(
            "actor checkpoint observation dimension is not a supported PickTool contract: "
            f"embed={tuple(embed.shape)}, mean={tuple(mean.shape)}"
        )
    if (
        expected_observation_dim is not None
        and observation_dim != expected_observation_dim
    ):
        raise RuntimeError(
            "actor checkpoint observation dimension disagrees with the expected contract: "
            f"actor={observation_dim}, expected={expected_observation_dim}"
        )
    hidden = int(embed.shape[0])
    if int(mean.shape[1]) != hidden:
        raise RuntimeError("actor embedder and predictor hidden dimensions differ")
    block_indices = sorted(
        int(match.group(1))
        for key in canonical
        if (match := _ENCODER_BLOCK_PATTERN.match(key)) is not None
    )
    if block_indices != list(range(len(block_indices))):
        raise RuntimeError(f"actor encoder blocks are non-contiguous: {block_indices}")
    signature = (len(block_indices), hidden)
    if signature == (PRODUCTION_ACTOR_BLOCKS, PRODUCTION_ACTOR_HIDDEN):
        return "production"
    if signature == (SMOKE_ACTOR_BLOCKS, SMOKE_ACTOR_HIDDEN):
        return "smoke"
    raise RuntimeError(f"unsupported actor architecture blocks/hidden={signature}")


def resolve_checkpoint_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file() and resolved.name == "actor.pt":
        resolved = resolved.parent
    if not resolved.is_dir():
        raise FileNotFoundError(f"FlashSAC checkpoint directory does not exist: {resolved}")
    required = ("actor.pt", "critic.pt", "target_critic.pt", "temperature.pt")
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint {resolved} is missing files: {missing}")
    return resolved


def _checkpoint_actor_state(checkpoint: Path) -> Mapping[str, Any]:
    payload = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("network_state_dict"), Mapping
    ):
        raise TypeError("actor.pt must contain a mapping network_state_dict")
    return payload["network_state_dict"]


def infer_checkpoint_actor_action_dim(checkpoint: Path) -> int:
    return infer_actor_action_dim_from_state(_checkpoint_actor_state(checkpoint))


def infer_checkpoint_architecture(
    checkpoint: Path,
    *,
    expected_action_dim: int | None = None,
    expected_observation_dim: int | None = None,
) -> str:
    return infer_actor_architecture_from_state(
        _checkpoint_actor_state(checkpoint),
        expected_action_dim=expected_action_dim,
        expected_observation_dim=expected_observation_dim,
    )


def zero_expand_actor_observation_state(
    state: Mapping[str, Any],
    *,
    source_observation_dim: int = OBSERVATION_DIM,
    target_observation_dim: int = COUPLED_OBSERVATION_DIM,
) -> dict[str, torch.Tensor]:
    """Expand the actor input layer while initially ignoring all 16 new features."""

    if source_observation_dim < 1 or target_observation_dim <= source_observation_dim:
        raise ValueError("observation expansion requires 0 < source_dim < target_dim")
    canonical = _canonical_actor_state(state)
    embed = canonical.get("embedder.w.w.weight")
    if embed is None or embed.ndim != 2:
        raise RuntimeError("actor checkpoint is missing a compatible observation embedder")
    if int(embed.shape[1]) != source_observation_dim:
        raise RuntimeError(
            "actor checkpoint observation embedder cannot be zero-expanded: "
            f"source={tuple(embed.shape)}, expected input={source_observation_dim}"
        )
    expanded = dict(state)
    raw_key_by_canonical = {
        raw_key.removeprefix(_COMPILED_PREFIX): raw_key for raw_key in state
    }
    embed_key = raw_key_by_canonical["embedder.w.w.weight"]
    expanded_embed = embed.new_zeros((int(embed.shape[0]), target_observation_dim))
    expanded_embed[:, :source_observation_dim].copy_(embed)
    expanded[embed_key] = expanded_embed
    vector_fill = {
        "embedder.norm.weight": 1.0,
        "embedder.norm.bias": 0.0,
        "embedder.norm.running_mean": 0.0,
        "embedder.norm.running_var": 1.0,
    }
    for canonical_key, fill_value in vector_fill.items():
        raw_key = raw_key_by_canonical.get(canonical_key)
        if raw_key is None:
            continue
        source_value = state[raw_key]
        if source_value.shape != (source_observation_dim,):
            raise RuntimeError(
                f"actor input-normalizer tensor {canonical_key!r} cannot be expanded: "
                f"source={tuple(source_value.shape)}"
            )
        target_value = source_value.new_full(
            (target_observation_dim,), fill_value
        )
        target_value[:source_observation_dim].copy_(source_value)
        expanded[raw_key] = target_value
    return expanded


@contextmanager
def zero_expanded_actor_checkpoint(
    checkpoint: Path,
) -> Iterator[Path]:
    """Create a short-lived actor-only full115-to-coupled131 transfer bundle."""

    actor_path = checkpoint / "actor.pt"
    payload = torch.load(actor_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("network_state_dict"), Mapping
    ):
        raise TypeError("actor.pt must contain a mapping network_state_dict")
    transferred_payload = dict(payload)
    transferred_payload["network_state_dict"] = zero_expand_actor_observation_state(
        payload["network_state_dict"]
    )
    with tempfile.TemporaryDirectory(prefix="flashsac_coupled_actor_") as directory:
        transfer_dir = Path(directory)
        torch.save(transferred_payload, transfer_dir / "actor.pt")
        yield transfer_dir


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--architecture",
        choices=("production", "smoke", "auto"),
        default="production",
        help="Production exactly matches formal train.py; auto also recognizes its smoke architecture.",
    )
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    option_group = parser.add_mutually_exclusive_group()
    option_group.add_argument(
        "--close_option_mode",
        action="store_true",
        help=(
            "Evaluate the short-horizon stable-latch option. Its success is reported separately "
            "and is never treated as full-task 20 cm success."
        ),
    )
    option_group.add_argument(
        "--power_close_option_mode",
        action="store_true",
        help=(
            "Evaluate the POWER close v1 thumb+three topology with a native 14-D hand policy. "
            "This is distinct from the legacy close-option objective."
        ),
    )
    option_group.add_argument(
        "--coupled_power_align_close_option_mode",
        action="store_true",
        help=(
            "Evaluate the 21-D coupled arm+hand POWER align-close option. Success is a "
            "stable near-table close latch, never full-task 20 cm lift success."
        ),
    )
    parser.add_argument(
        "--coupled_controller_ablation",
        choices=(
            "exact_teacher",
            "teacher_arm_actor_hand",
            "actor_arm_teacher_hand",
        ),
        default=None,
        help=(
            "Diagnostic-only exact/actor arm-by-hand controller override for the coupled "
            "option. It requires the strict CEM artifact and its byte-identical curriculum."
        ),
    )
    parser.add_argument(
        "--coupled_teacher_artifact",
        type=Path,
        default=None,
        help="Fail-closed strict coupled CEM artifact used only by controller ablation.",
    )
    parser.add_argument(
        "--allow_cross_task_actor",
        action="store_true",
        help=(
            "Explicitly benchmark actor weights under a task mode different from the "
            "checkpoint's task_contract.json. The critic is evaluation-only and ignored."
        ),
    )
    parser.add_argument(
        "--hold_arm_until_stable_grasp",
        action="store_true",
        help=(
            "On a full-task close_start evaluation, mask the seven arm actions until the strict "
            "grasp contract is stable for 15 frames, then release the same actor in-place."
        ),
    )
    parser.add_argument("--arm_hold_confirm_steps", type=int, default=15)
    parser.add_argument(
        "--arm_hold_grasp_quality_threshold", type=float, default=0.35
    )
    parser.add_argument("--arm_hold_min_hold_quality", type=float, default=0.5)
    parser.add_argument("--arm_hold_safe_force_limit", type=float, default=30.0)
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument(
        "--curriculum_dataset",
        type=Path,
        default=None,
        help="Optional physically captured reset-boundary dataset, matching train.py.",
    )
    parser.add_argument("--curriculum_boundary", default="close_start")
    parser.add_argument("--curriculum_probability", type=float, default=0.0)
    parser.add_argument("--curriculum_joint_noise", type=float, default=0.0)
    parser.add_argument("--max_vector_steps", type=int, default=None)
    parser.add_argument("--validate_finite", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/pick_tool_flashsac_eval.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    return args, launcher


def validate_curriculum_config(
    *,
    dataset: Path | None,
    probability: float,
    joint_noise: float,
) -> None:
    """Apply the same curriculum argument contract as formal training."""

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("--curriculum_probability must be in [0, 1]")
    if not math.isfinite(joint_noise) or joint_noise < 0.0:
        raise ValueError("--curriculum_joint_noise must be finite and non-negative")
    if probability > 0.0 and dataset is None:
        raise ValueError("--curriculum_probability > 0 requires --curriculum_dataset")
    if dataset is not None and not dataset.is_file():
        raise FileNotFoundError(dataset)


def validate_close_option_evaluation_config(
    *,
    close_option_mode: bool,
    curriculum_dataset: Path | None,
    curriculum_boundary: str,
    curriculum_probability: float,
    curriculum_joint_noise: float,
    episode_length_s: float | None,
    coupled_power_align_close_option_mode: bool = False,
) -> None:
    """Reject close-option evaluations outside the captured pregrasp MDP."""

    if not close_option_mode:
        return
    if curriculum_dataset is None:
        raise ValueError("--close_option_mode requires a close-start curriculum dataset")
    if curriculum_boundary != "close_start":
        raise ValueError("--close_option_mode requires --curriculum_boundary close_start")
    if curriculum_probability != 1.0:
        raise ValueError("--close_option_mode requires --curriculum_probability 1")
    if not 0.0 <= curriculum_joint_noise <= 0.02:
        raise ValueError(
            "--close_option_mode requires --curriculum_joint_noise in [0, 0.02]"
        )
    minimum_episode_length_s = (
        3.0 if coupled_power_align_close_option_mode else 0.40
    )
    if (
        episode_length_s is None
        or not minimum_episode_length_s <= episode_length_s <= 5.0
    ):
        raise ValueError(
            "close-option evaluation requires --episode_length_s in "
            f"[{minimum_episode_length_s:.2f}, 5] so its physical contract is reachable"
        )


def validate_hierarchical_arm_hold_evaluation_config(
    *,
    enabled: bool,
    close_option_mode: bool,
    curriculum_dataset: Path | None,
    curriculum_boundary: str,
    curriculum_probability: float,
    curriculum_joint_noise: float,
    episode_length_s: float | None,
    confirm_steps: int,
    grasp_quality_threshold: float,
    min_hold_quality: float,
    safe_force_limit: float,
) -> None:
    """Restrict the in-place close-to-lift handoff to its captured close-start MDP."""

    if not enabled:
        return
    if close_option_mode:
        raise ValueError(
            "--hold_arm_until_stable_grasp is a full-task handoff; "
            "close option already holds the arm"
        )
    if curriculum_dataset is None:
        raise ValueError(
            "--hold_arm_until_stable_grasp requires a close-start curriculum dataset"
        )
    if curriculum_boundary != "close_start" or curriculum_probability != 1.0:
        raise ValueError(
            "--hold_arm_until_stable_grasp requires close_start resets with probability 1"
        )
    if not 0.0 <= curriculum_joint_noise <= 0.02:
        raise ValueError(
            "--hold_arm_until_stable_grasp requires curriculum joint noise in [0, 0.02]"
        )
    if episode_length_s is not None and episode_length_s < 0.30:
        raise ValueError(
            "--hold_arm_until_stable_grasp requires --episode_length_s >= 0.30"
        )
    if confirm_steps < 1:
        raise ValueError("--arm_hold_confirm_steps must be positive")
    for name, value in (
        ("--arm_hold_grasp_quality_threshold", grasp_quality_threshold),
        ("--arm_hold_min_hold_quality", min_hold_quality),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not math.isfinite(safe_force_limit) or safe_force_limit <= 0.0:
        raise ValueError("--arm_hold_safe_force_limit must be finite and positive")


def _validate_args(args: argparse.Namespace) -> None:
    task_mode = task_mode_from_option_flags(
        close_option_mode=args.close_option_mode,
        power_close_option_mode=args.power_close_option_mode,
        coupled_power_align_close_option_mode=(
            args.coupled_power_align_close_option_mode
        ),
    )
    if args.episodes < 1 or args.num_envs < 1:
        raise ValueError("--episodes and --num_envs must be positive")
    if args.max_vector_steps is not None and args.max_vector_steps < 1:
        raise ValueError("--max_vector_steps must be positive")
    if args.episode_length_s is not None and (
        not math.isfinite(args.episode_length_s) or args.episode_length_s <= 0.0
    ):
        raise ValueError("--episode_length_s must be finite and positive")
    validate_curriculum_config(
        dataset=args.curriculum_dataset,
        probability=args.curriculum_probability,
        joint_noise=args.curriculum_joint_noise,
    )
    validate_close_option_evaluation_config(
        close_option_mode=task_mode in CLOSE_OPTION_TASK_MODES,
        curriculum_dataset=args.curriculum_dataset,
        curriculum_boundary=args.curriculum_boundary,
        curriculum_probability=args.curriculum_probability,
        curriculum_joint_noise=args.curriculum_joint_noise,
        episode_length_s=args.episode_length_s,
        coupled_power_align_close_option_mode=(
            task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE
        ),
    )
    validate_hierarchical_arm_hold_evaluation_config(
        enabled=args.hold_arm_until_stable_grasp,
        close_option_mode=task_mode in CLOSE_OPTION_TASK_MODES,
        curriculum_dataset=args.curriculum_dataset,
        curriculum_boundary=args.curriculum_boundary,
        curriculum_probability=args.curriculum_probability,
        curriculum_joint_noise=args.curriculum_joint_noise,
        episode_length_s=args.episode_length_s,
        confirm_steps=args.arm_hold_confirm_steps,
        grasp_quality_threshold=args.arm_hold_grasp_quality_threshold,
        min_hold_quality=args.arm_hold_min_hold_quality,
        safe_force_limit=args.arm_hold_safe_force_limit,
    )
    controller_ablation = getattr(args, "coupled_controller_ablation", None)
    teacher_artifact = getattr(args, "coupled_teacher_artifact", None)
    if controller_ablation is None:
        if teacher_artifact is not None:
            raise ValueError(
                "--coupled_teacher_artifact requires --coupled_controller_ablation"
            )
    else:
        if task_mode != COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
            raise ValueError(
                "--coupled_controller_ablation requires "
                "--coupled_power_align_close_option_mode"
            )
        if teacher_artifact is None or not teacher_artifact.is_file():
            raise FileNotFoundError(
                teacher_artifact
                if teacher_artifact is not None
                else "--coupled_teacher_artifact"
            )
        if args.curriculum_dataset is None:
            raise ValueError(
                "--coupled_controller_ablation requires --curriculum_dataset"
            )
        if args.curriculum_probability != 1.0 or args.curriculum_joint_noise != 0.0:
            raise ValueError(
                "controller ablation requires curriculum probability 1 and exact zero joint noise"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    task_mode = task_mode_from_option_flags(
        close_option_mode=args.close_option_mode,
        power_close_option_mode=args.power_close_option_mode,
        coupled_power_align_close_option_mode=(
            args.coupled_power_align_close_option_mode
        ),
    )
    if task_mode in CLOSE_OPTION_TASK_MODES and args.episode_length_s is None:
        args.episode_length_s = 5.0
    _validate_args(args)
    _seed_everything(args.seed)

    controller_ablation = getattr(args, "coupled_controller_ablation", None)
    teacher_prior = None
    if controller_ablation is not None:
        from coupled_teacher_prior import load_coupled_teacher_prior

        teacher_prior = load_coupled_teacher_prior(
            args.coupled_teacher_artifact,
            args.curriculum_dataset,
        )

    # Import the sibling contract reader before agent_bridge prepends the
    # upstream FlashSAC directory to sys.path; both trees contain train.py.
    from train import read_checkpoint_task_contract

    from adapter import make_pick_tool_env
    from agent_bridge import (
        FLASH_SAC_COMMIT,
        ActionNoiseGroup,
        FlashSACTorchBridge,
        build_agent_config,
    )

    device_string = str(args.device or "cuda:0")
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"PickTool FlashSAC evaluation requires CUDA, got {device_string}")

    checkpoint = resolve_checkpoint_directory(args.checkpoint)
    checkpoint_contract = read_checkpoint_task_contract(checkpoint)
    checkpoint_task_mode = str(checkpoint_contract["task_mode"])
    cross_task_actor_evaluation = resolve_cross_task_actor_evaluation(
        checkpoint_task_mode=checkpoint_task_mode,
        requested_task_mode=task_mode,
        allow_cross_task_actor=args.allow_cross_task_actor,
    )
    checkpoint_actor_action_dim = infer_checkpoint_actor_action_dim(checkpoint)
    source_policy_contract, target_policy_contract, source_action_indices = (
        validate_checkpoint_evaluation_contract(
            checkpoint_task_mode=checkpoint_task_mode,
            checkpoint_contract=checkpoint_contract,
            requested_task_mode=task_mode,
            actor_action_dim=checkpoint_actor_action_dim,
        )
    )
    checkpoint_architecture = infer_checkpoint_architecture(
        checkpoint,
        expected_action_dim=int(source_policy_contract["policy_action_dim"]),
        expected_observation_dim=int(source_policy_contract["observation_dim"]),
    )
    architecture = checkpoint_architecture if args.architecture == "auto" else args.architecture
    if architecture != checkpoint_architecture:
        raise RuntimeError(
            f"requested {architecture} architecture but checkpoint is {checkpoint_architecture}; "
            "use --architecture auto only when smoke-checkpoint evaluation is intentional"
        )

    cfg_overrides: dict[str, Any] = {
        "close_option_mode": task_mode in CLOSE_OPTION_TASK_MODES,
        "power_close_option_mode": task_mode in POWER_CLOSE_TASK_MODES,
        "coupled_power_align_close_option_mode": (
            task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE
        ),
        "hold_arm_until_stable_grasp": bool(args.hold_arm_until_stable_grasp),
    }
    if task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE:
        cfg_overrides.update(
            {
                "observation_space": COUPLED_OBSERVATION_DIM,
                "state_space": COUPLED_OBSERVATION_DIM,
            }
        )
    if args.hold_arm_until_stable_grasp:
        cfg_overrides.update(
            {
                "arm_hold_confirm_steps": args.arm_hold_confirm_steps,
                "arm_hold_grasp_quality_threshold": (
                    args.arm_hold_grasp_quality_threshold
                ),
                "arm_hold_min_hold_quality": args.arm_hold_min_hold_quality,
                "arm_hold_safe_force_limit": args.arm_hold_safe_force_limit,
            }
        )
    if args.episode_length_s is not None:
        cfg_overrides["episode_length_s"] = args.episode_length_s
    if args.curriculum_dataset is not None:
        cfg_overrides.update(
            {
                "curriculum_dataset": str(args.curriculum_dataset.resolve()),
                "curriculum_boundary": args.curriculum_boundary,
                "curriculum_reset_probability": args.curriculum_probability,
                "curriculum_joint_noise": args.curriculum_joint_noise,
            }
        )
    env = make_pick_tool_env(
        num_envs=args.num_envs,
        device=device_string,
        seed=args.seed,
        cfg_overrides=cfg_overrides,
        validate_finite=args.validate_finite,
        hand_only_actions=task_mode == POWER_CLOSE_OPTION_MODE,
    )
    task_cfg = env.unwrapped.cfg
    close_option_confirm_steps = int(task_cfg.close_option_confirm_steps)
    close_option_min_hold_quality = float(task_cfg.close_option_min_hold_quality)
    grasp_quality_threshold = float(
        task_cfg.power_grasp_quality_high
        if task_mode in POWER_CLOSE_TASK_MODES
        else task_cfg.grasp_quality_high
    )
    power_required_other_contacts = int(task_cfg.power_grasp_required_other_contacts)
    close_option_safe_force_limit = float(task_cfg.grasp_bonus_max_force)
    close_option_unlatched_lift_limit = float(task_cfg.close_option_unlatched_lift_limit)
    close_option_horizontal_drift_limit = float(task_cfg.close_option_horizontal_drift_limit)
    close_option_min_proximity = float(task_cfg.close_option_min_proximity)
    close_option_lost_window_steps = int(task_cfg.close_option_lost_window_steps)
    arm_hold_confirm_steps = int(task_cfg.arm_hold_confirm_steps)
    arm_hold_grasp_quality_threshold = float(
        task_cfg.arm_hold_grasp_quality_threshold
    )
    arm_hold_min_hold_quality = float(task_cfg.arm_hold_min_hold_quality)
    arm_hold_safe_force_limit = float(task_cfg.arm_hold_safe_force_limit)

    if architecture == "production":
        actor_blocks, actor_hidden = PRODUCTION_ACTOR_BLOCKS, PRODUCTION_ACTOR_HIDDEN
        critic_blocks, critic_hidden = PRODUCTION_CRITIC_BLOCKS, PRODUCTION_CRITIC_HIDDEN
    else:
        actor_blocks, actor_hidden = SMOKE_ACTOR_BLOCKS, SMOKE_ACTOR_HIDDEN
        critic_blocks, critic_hidden = SMOKE_CRITIC_BLOCKS, SMOKE_CRITIC_HIDDEN

    # All structural values match formal train.py.  Replay/schedule values are
    # deliberately tiny because evaluation neither inserts nor updates data.
    agent_cfg = build_agent_config(
        seed=args.seed,
        normalize_reward=True,
        normalized_G_max=5.0,
        device_type=device_string,
        buffer_device_type=device_string,
        buffer_max_length=max(args.num_envs, 32),
        buffer_min_length=1,
        sample_batch_size=1,
        n_step=3 if architecture == "production" else 1,
        actor_num_blocks=actor_blocks,
        actor_hidden_dim=actor_hidden,
        critic_num_blocks=critic_blocks,
        critic_hidden_dim=critic_hidden,
        critic_num_bins=(
            PRODUCTION_CRITIC_BINS if architecture == "production" else SMOKE_CRITIC_BINS
        ),
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        use_amp=architecture == "production",
        load_optimizer=False,
        load_reward_normalizer=False,
    )
    noise_group_specs = (
        HAND_NOISE_GROUP_SPECS
        if int(target_policy_contract["policy_action_dim"]) == HAND_ACTION_DIM
        else NOISE_GROUP_SPECS
    )
    noise_groups = tuple(
        ActionNoiseGroup(
            name,
            start,
            stop,
            scale=scale,
            zeta_mu=zeta_mu,
            zeta_max=zeta_max,
        )
        for name, start, stop, scale, zeta_mu, zeta_max in noise_group_specs
    )
    agent = FlashSACTorchBridge(
        env.observation_space,
        env.action_space,
        env.env_info,
        agent_cfg,
        noise_groups=noise_groups,
        restore_rng_state_on_load=False,
    )
    full_to_coupled_transfer = (
        checkpoint_task_mode == FULL_TASK_MODE
        and task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_MODE
    )
    if full_to_coupled_transfer:
        with zero_expanded_actor_checkpoint(checkpoint) as transfer_checkpoint:
            agent.load_actor(str(transfer_checkpoint))
    elif source_action_indices is None:
        agent.load_actor(str(checkpoint))
    else:
        agent.load_actor(
            str(checkpoint),
            source_action_indices=source_action_indices,
            expected_source_action_dim=int(source_policy_contract["policy_action_dim"]),
        )
    # A deterministic evaluation never consumes cached noise.  Reset it anyway
    # so a checkpoint trained with another num_envs cannot leak stale shape.
    agent.reset_exploration(batch_size=args.num_envs)

    observation, _ = env.reset(randomize_episode_lengths=False)
    initial_truth = _read_physical_truth(env.unwrapped, task_mode=task_mode)
    tracker = StrictEpisodeTracker(
        episodes=args.episodes,
        num_envs=args.num_envs,
        device=env.device,
        initial_truth=initial_truth,
        task_mode=task_mode,
        track_arm_hold_handoff=args.hold_arm_until_stable_grasp,
    )
    quota_max = int(tracker.quotas.max().item())
    default_max_steps = max(1, env.max_episode_steps * quota_max + quota_max)
    max_vector_steps = args.max_vector_steps or default_max_steps
    vector_steps = 0

    try:
        while not tracker.complete and vector_steps < max_vector_steps:
            vector_steps += 1
            action = agent.sample_actions(
                vector_steps,
                {"next_observation": observation},
                training=False,
            )
            if controller_ablation is not None:
                assert teacher_prior is not None
                teacher_action = teacher_prior.teacher_action(observation)
                action = apply_coupled_controller_ablation(
                    action,
                    teacher_action,
                    controller_ablation,
                )
            # Slots whose deterministic quota is complete continue simulating
            # independently but cannot contribute additional events.
            action = torch.where(tracker.active.unsqueeze(-1), action, torch.zeros_like(action))
            next_observation, reward, terminated, truncated, info = env.step(action)
            events = validate_terminal_events(
                info,
                terminated,
                truncated,
                task_mode=task_mode,
                close_option_confirm_steps=close_option_confirm_steps,
                power_required_other_contacts=power_required_other_contacts,
                power_grasp_quality_threshold=grasp_quality_threshold,
                close_option_min_hold_quality=close_option_min_hold_quality,
                close_option_safe_force_limit=close_option_safe_force_limit,
            )
            if args.hold_arm_until_stable_grasp:
                events.update(
                    validate_arm_hold_handoff_state(
                        info,
                        num_envs=args.num_envs,
                        device=env.device,
                        confirm_steps=arm_hold_confirm_steps,
                    )
                )
            transition_truth = physical_truth_from_terminal_info(
                info,
                num_envs=args.num_envs,
                device=env.device,
                task_mode=task_mode,
                close_option_confirm_steps=close_option_confirm_steps,
                close_option_min_hold_quality=close_option_min_hold_quality,
                grasp_quality_threshold=grasp_quality_threshold,
                safe_force_limit=close_option_safe_force_limit,
            )
            # DirectRLEnv has already reset done rows at this point.  This read
            # is used only to initialize their next episode; old-episode maxima
            # above came exclusively from the reset-before terminal payload.
            post_reset_truth = _read_physical_truth(env.unwrapped, task_mode=task_mode)
            tracker.step(
                reward=reward.to(dtype=torch.float32),
                terminated=terminated,
                truncated=truncated,
                events=events,
                transition_truth=transition_truth,
                post_reset_truth=post_reset_truth,
            )
            observation = next_observation

        if not tracker.complete:
            raise RuntimeError(
                f"completed {len(tracker.records)}/{args.episodes} episodes after "
                f"--max_vector_steps={max_vector_steps}"
            )
        metrics = build_strict_metrics(
            tracker.records,
            checkpoint=checkpoint,
            architecture=architecture,
            seed=args.seed,
            num_envs=args.num_envs,
            vector_steps=vector_steps,
            max_vector_steps=max_vector_steps,
            episode_length_s=float(env.unwrapped.cfg.episode_length_s),
            max_episode_steps=env.max_episode_steps,
            curriculum_dataset=args.curriculum_dataset,
            curriculum_dataset_sha256=(
                _sha256(args.curriculum_dataset)
                if args.curriculum_dataset is not None
                else None
            ),
            curriculum_boundary=args.curriculum_boundary,
            curriculum_probability=args.curriculum_probability,
            curriculum_joint_noise=args.curriculum_joint_noise,
            use_compile=args.use_compile,
            upstream_commit=FLASH_SAC_COMMIT,
            task_mode=task_mode,
            close_option_confirm_steps=close_option_confirm_steps,
            close_option_grasp_quality_threshold=grasp_quality_threshold,
            close_option_min_hold_quality=close_option_min_hold_quality,
            close_option_safe_force_limit=close_option_safe_force_limit,
            close_option_unlatched_lift_limit=close_option_unlatched_lift_limit,
            close_option_horizontal_drift_limit=close_option_horizontal_drift_limit,
            close_option_min_proximity=close_option_min_proximity,
            close_option_lost_window_steps=close_option_lost_window_steps,
            checkpoint_task_mode=checkpoint_task_mode,
            cross_task_actor_evaluation=cross_task_actor_evaluation,
            policy_action_dim=int(target_policy_contract["policy_action_dim"]),
            environment_action_dim=int(target_policy_contract["environment_action_dim"]),
            policy_action_layout=str(target_policy_contract["policy_action_layout"]),
            action_projection=str(target_policy_contract["action_projection"]),
            observation_dim=int(target_policy_contract["observation_dim"]),
            observation_contract=str(target_policy_contract["observation_contract"]),
            noise_group_specs=noise_group_specs,
            power_required_other_contacts=power_required_other_contacts,
        )
        metrics["checkpoint_policy_contract"] = source_policy_contract
        metrics["actor_action_projection_indices"] = (
            list(source_action_indices) if source_action_indices is not None else None
        )
        metrics["actor_observation_transfer"] = (
            FULL115_TO_COUPLED131_OBSERVATION_TRANSFER
            if full_to_coupled_transfer
            else None
        )
        metrics["hold_arm_until_stable_grasp"] = bool(
            args.hold_arm_until_stable_grasp
        )
        metrics["coupled_controller_ablation"] = controller_ablation
        if teacher_prior is not None:
            metrics["coupled_teacher_prior"] = teacher_prior.contract_payload()
            metrics["coupled_teacher_artifact"] = teacher_prior.teacher_artifact_path
            metrics["policy"] = (
                "diagnostic_"
                + str(controller_ablation)
                + "+native_phase_shield"
            )
        if args.hold_arm_until_stable_grasp:
            metrics["policy"] = (
                "deterministic_tanh_actor_mean+arm_hold_supervisor"
            )
            metrics["arm_hold_supervisor"] = {
                "arm_action_width": 7,
                "confirm_steps": arm_hold_confirm_steps,
                "grasp_quality_threshold": arm_hold_grasp_quality_threshold,
                "min_hold_quality": arm_hold_min_hold_quality,
                "safe_force_limit_n": arm_hold_safe_force_limit,
                "release_semantics": (
                    "first unmasked arm action after the confirmed stable-close window"
                ),
            }
        _atomic_write_json(args.output.resolve(), metrics)
        return metrics
    finally:
        env.close()


def main() -> None:
    args, launcher = _parse_args()
    try:
        metrics = run(args)
        print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    except BaseException:
        # SimulationApp.close() can terminate Kit before Python renders an
        # uncaught exception.  Emit it first so a failed evaluation cannot look
        # like a successful, output-free run.
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
