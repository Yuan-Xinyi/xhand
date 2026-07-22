#!/usr/bin/env python3
"""Minimal Torch-native FlashSAC trainer for Pick-Tool-Token-Direct-v0.

Run this script with the Isaac Lab Python launcher.  The simulator is started
before the task, adapter, or FlashSAC modules are imported.  Isaac Lab
auto-resets completed sub-environments inside ``step``; the adapter therefore
provides two different next observations:

* the returned observation continues rollout from the reset state;
* ``transition_next_observation`` is the captured pre-reset terminal state and
  is the only value written to replay.

There is deliberately no in-process periodic evaluation in this first trainer.
Evaluation of a shared Isaac environment would destroy the live collection
state and can silently create a stale transition unless the collector is reset.
Use the separate deterministic evaluator between checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from online_handoff import (
    INITIAL_EPISODE_TERMINAL_EVENT_KEYS,
    online_handoff_metrics,
    reset_online_handoff_state,
    update_initial_episode_handoff_audit,
    update_online_handoff,
    validate_online_handoff_config,
)


@dataclass
class FractionalUpdateBudget:
    """Exact fractional update accounting without floating-point drift."""

    updates_per_interaction: float
    _rate: Fraction = field(init=False, repr=False)
    _credit: Fraction = field(default_factory=Fraction, init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.updates_per_interaction) or self.updates_per_interaction < 0.0:
            raise ValueError("updates_per_interaction must be finite and non-negative")
        self._rate = Fraction(str(self.updates_per_interaction)).limit_denominator(1_000_000)

    def grant(
        self,
        training_ready: bool,
        *,
        weight_numerator: int = 1,
        weight_denominator: int = 1,
    ) -> int:
        """Return updates due for one vector interaction.

        Warm-up interactions earn no deferred credit, matching the upstream
        FlashSAC loop rather than causing a burst of catch-up updates.  An
        optional exact rational weight lets masked online collection scale
        updates by newly materialized replay rows instead of repeatedly
        optimizing old data while SEARCH still owns every environment.
        """

        if (
            not isinstance(weight_numerator, int)
            or isinstance(weight_numerator, bool)
            or weight_numerator < 0
            or not isinstance(weight_denominator, int)
            or isinstance(weight_denominator, bool)
            or weight_denominator < 1
            or weight_numerator > weight_denominator
        ):
            raise ValueError(
                "update-budget weight must satisfy 0 <= numerator <= denominator"
            )
        if not training_ready:
            return 0
        self._credit += self._rate * Fraction(weight_numerator, weight_denominator)
        due = self._credit.numerator // self._credit.denominator
        self._credit -= due
        return due


@dataclass
class EpisodeAccumulator:
    num_envs: int
    device: torch.device
    returns: torch.Tensor = field(init=False)
    lengths: torch.Tensor = field(init=False)
    completed: torch.Tensor = field(init=False)
    return_sum: torch.Tensor = field(init=False)
    length_sum: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.completed = torch.zeros((), dtype=torch.long, device=self.device)
        self.return_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.length_sum = torch.zeros((), dtype=torch.long, device=self.device)

    def step(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        self.returns.add_(reward)
        self.lengths.add_(1)
        self.completed.add_(done.sum())
        self.return_sum.add_(torch.where(done, self.returns, 0.0).sum())
        self.length_sum.add_(torch.where(done, self.lengths, 0).sum())
        self.returns.masked_fill_(done, 0.0)
        self.lengths.masked_fill_(done, 0)

    def metrics(self) -> dict[str, float | int]:
        completed = int(self.completed.item())
        denominator = max(completed, 1)
        return {
            "train/completed_episodes": completed,
            "train/mean_episode_return": float(self.return_sum.item()) / denominator,
            "train/mean_episode_length": int(self.length_sum.item()) / denominator,
        }


TERMINAL_EVENT_KEYS = (
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
)

CLOSE_OPTION_TERMINAL_EVENT_KEYS = (
    "close_option_success",
    "close_option_failure",
    "close_option_timeout",
    "dropped",
    "unsafe_force",
    "close_option_unlatched_lift",
    "close_option_horizontal_escape",
    "close_option_lost_window",
)

POWER_CLOSE_OPTION_TERMINAL_EVENT_KEYS = (
    "power_close_option_success",
    "power_close_option_failure",
    "power_close_option_timeout",
    "dropped",
    "unsafe_force",
    "close_option_unlatched_lift",
    "close_option_horizontal_escape",
    "close_option_lost_window",
)

COUPLED_POWER_ALIGN_CLOSE_OPTION_TERMINAL_EVENT_KEYS = (
    *POWER_CLOSE_OPTION_TERMINAL_EVENT_KEYS,
    "coupled_power_pose_escape",
)

TASK_CONTRACT_FILENAME = "task_contract.json"
TEACHER_ACTION_PRIOR_FILENAME = "teacher_action_prior.json"
TEACHER_RESIDUAL_STATE_FILENAME = "teacher_residual_training_state.json"
INCOMPLETE_CHECKPOINT_FILENAME = ".incomplete_checkpoint.json"
FROZEN_LIFT_ACTOR_FILENAME = "frozen_lift_actor.pt"
PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_KIND = "public_latch_frozen_actor_v1"
TASK_CONTRACT_VERSION = 6
FLASH_SAC_GAMMA = 0.99
CORE_CHECKPOINT_FILENAMES = (
    "actor.pt",
    "critic.pt",
    "target_critic.pt",
    "temperature.pt",
)
STALE_OPTIONAL_CHECKPOINT_FILENAMES = (
    "replay_buffer.pt",
    "actor_rehearsal.pt",
    TEACHER_ACTION_PRIOR_FILENAME,
    TEACHER_RESIDUAL_STATE_FILENAME,
    FROZEN_LIFT_ACTOR_FILENAME,
    TASK_CONTRACT_FILENAME,
)
FULL_TASK_MODE = "full_task"
CLOSE_OPTION_TASK_MODE = "close_option"
POWER_CLOSE_OPTION_TASK_MODE = "power_close_option_v1"
COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE = "coupled_power_align_close_option_v1"
COUPLED_TEACHER_RESIDUAL_TASK_MODE = "coupled_power_teacher_residual_v1"
TASK_MODES = (
    FULL_TASK_MODE,
    CLOSE_OPTION_TASK_MODE,
    POWER_CLOSE_OPTION_TASK_MODE,
    COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
    COUPLED_TEACHER_RESIDUAL_TASK_MODE,
)
PICK_TOOL_LATCH_OBSERVATION_INDEX = 106
PICK_TOOL_ARM_ACTION_DIM = 7
PICK_TOOL_HAND_ACTION_DIM = 14
PICK_TOOL_ACTION_DIM = 21
PICK_TOOL_ENVIRONMENT_ACTION_DIM = 21
FULL_POLICY_ACTION_LAYOUT = "arm_delta7|crossdex_token9|distal_residual5"
HAND_POLICY_ACTION_LAYOUT = "crossdex_token9|distal_residual5"
TEACHER_RESIDUAL_POLICY_ACTION_LAYOUT = (
    "crossdex_token_residual9|distal_action_residual5"
)
IDENTITY_ACTION_PROJECTION = "identity_v1"
PREPEND_ZERO_ARM_ACTION_PROJECTION = "prepend_zero_arm7_v1"
COUPLED_TEACHER_RESIDUAL_ACTION_PROJECTION = (
    "coupled_cem_teacher_prior_plus_bounded_hand_residual_v1"
)
FULL21_TO_HAND14_ACTOR_PROJECTION = "full21_to_hand14_v1"
FULL115_TO_COUPLED131_ACTOR_PROJECTION = (
    "full115_to_coupled131_zero_pad_input_v1"
)
PICK_TOOL_OBSERVATION_DIM = 115
PICK_TOOL_OBSERVATION_CONTRACT = "pick_tool_markov115_v1"
PUBLIC_LATCH_ARM_ACTION_AUTHORITY = (
    {
        "name": "arm_after_public_latch",
        "start": 0,
        "stop": PICK_TOOL_ARM_ACTION_DIM,
        "observation_index": PICK_TOOL_LATCH_OBSERVATION_INDEX,
        "active_value": 1.0,
    },
)
POWER_CLOSE_OBSERVATION_CONTRACT = "pick_tool_power_close_markov115_v1"
COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM = 131
COUPLED_ALIGN_ACTIVE_OBSERVATION_INDEX = 129
COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_CONTRACT = (
    "pick_tool_coupled_power_align_close_state131_v1"
)
FULL_ACTION_NOISE_GROUP_SPECS = (
    ("arm", 0, 7, 1.0, 1.0, 64),
    ("token", 7, 16, 0.5, 1.25, 32),
    ("residual", 16, 21, 0.35, 1.5, 16),
)
POWER_ACTION_NOISE_GROUP_SPECS = (
    ("token", 0, 9, 0.5, 1.25, 32),
    ("residual", 9, 14, 0.35, 1.5, 16),
)


def policy_action_authority_contract(
    public_latch_arm_gate: bool,
) -> list[dict[str, Any]]:
    """Return the JSON/task-contract form of the public action gate."""

    if not isinstance(public_latch_arm_gate, bool):
        raise TypeError("public_latch_arm_gate must be bool")
    return (
        [dict(rule) for rule in PUBLIC_LATCH_ARM_ACTION_AUTHORITY]
        if public_latch_arm_gate
        else []
    )


def validate_policy_action_authority_contract(
    value: Any,
    *,
    task_mode: str,
    source: str,
) -> list[dict[str, Any]]:
    """Validate the only state-dependent policy authority currently supported."""

    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{source} policy_action_authority must be a list")
    normalized: list[dict[str, Any]] = []
    expected_keys = {
        "name",
        "start",
        "stop",
        "observation_index",
        "active_value",
    }
    for index, raw_rule in enumerate(value):
        if not isinstance(raw_rule, Mapping) or set(raw_rule) != expected_keys:
            raise ValueError(
                f"{source} policy_action_authority[{index}] has invalid fields"
            )
        if not isinstance(raw_rule["name"], str) or not raw_rule["name"]:
            raise ValueError(
                f"{source} policy_action_authority[{index}] has invalid name"
            )
        for key in ("start", "stop", "observation_index"):
            if not isinstance(raw_rule[key], int) or isinstance(raw_rule[key], bool):
                raise ValueError(
                    f"{source} policy_action_authority[{index}].{key} must be int"
                )
        active_value = raw_rule["active_value"]
        if (
            not isinstance(active_value, (int, float))
            or isinstance(active_value, bool)
            or not math.isfinite(float(active_value))
        ):
            raise ValueError(
                f"{source} policy_action_authority[{index}].active_value is invalid"
            )
        normalized.append(
            {
                "name": raw_rule["name"],
                "start": raw_rule["start"],
                "stop": raw_rule["stop"],
                "observation_index": raw_rule["observation_index"],
                "active_value": float(active_value),
            }
        )
    unrestricted: list[dict[str, Any]] = []
    public_latch = [dict(rule) for rule in PUBLIC_LATCH_ARM_ACTION_AUTHORITY]
    if normalized == unrestricted:
        return normalized
    if normalized != public_latch:
        raise ValueError(f"{source} has an unsupported policy_action_authority")
    if task_mode != FULL_TASK_MODE:
        raise ValueError(
            f"{source} public latch arm authority requires task_mode={FULL_TASK_MODE!r}"
        )
    return normalized


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_policy_router_contract(
    value: Any,
    *,
    task_mode: str,
    policy_action_authority: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    source: str,
) -> dict[str, Any] | None:
    """Validate the closed V6 dual-policy routing contract."""

    if value is None:
        return None
    expected_outer_keys = {
        "kind",
        "observation_index",
        "close_value",
        "frozen_value",
        "trainable_action_slice",
        "close_arm_fill",
        "frozen_action_slice",
        "frozen_action_sampling",
        "frozen_action_entropy",
        "frozen_actor",
    }
    if not isinstance(value, Mapping) or set(value) != expected_outer_keys:
        raise ValueError(f"{source} policy_router has invalid fields")
    if task_mode != FULL_TASK_MODE:
        raise ValueError(f"{source} policy_router requires task_mode={FULL_TASK_MODE!r}")
    if [dict(rule) for rule in policy_action_authority] != [
        dict(rule) for rule in PUBLIC_LATCH_ARM_ACTION_AUTHORITY
    ]:
        raise ValueError(f"{source} policy_router requires public latch arm authority")
    expected_runtime = runtime_contract(FULL_TASK_MODE)
    if dict(runtime) != expected_runtime:
        raise ValueError(f"{source} policy_router requires the full-task runtime contract")

    if value.get("kind") != PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_KIND:
        raise ValueError(f"{source} policy_router has unsupported kind")
    observation_index = value.get("observation_index")
    if (
        not isinstance(observation_index, int)
        or isinstance(observation_index, bool)
        or observation_index != PICK_TOOL_LATCH_OBSERVATION_INDEX
    ):
        raise ValueError(f"{source} policy_router has invalid observation_index")
    for key, expected in (("close_value", 0.0), ("frozen_value", 1.0)):
        candidate = value.get(key)
        if (
            not isinstance(candidate, (int, float))
            or isinstance(candidate, bool)
            or not math.isfinite(float(candidate))
            or float(candidate) != expected
        ):
            raise ValueError(f"{source} policy_router has invalid {key}")

    def require_exact_int_slice(key: str, expected: tuple[int, int]) -> list[int]:
        candidate = value.get(key)
        if (
            not isinstance(candidate, (list, tuple))
            or len(candidate) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in candidate)
            or tuple(candidate) != expected
        ):
            raise ValueError(f"{source} policy_router has invalid {key}")
        return [int(candidate[0]), int(candidate[1])]

    trainable_slice = require_exact_int_slice(
        "trainable_action_slice",
        (PICK_TOOL_ARM_ACTION_DIM, PICK_TOOL_ACTION_DIM),
    )
    frozen_slice = require_exact_int_slice(
        "frozen_action_slice",
        (0, PICK_TOOL_ACTION_DIM),
    )
    expected_constants = {
        "close_arm_fill": "exact_zero",
        "frozen_action_sampling": "deterministic_tanh_mean",
        "frozen_action_entropy": "exact_zero",
    }
    for key, expected in expected_constants.items():
        if value.get(key) != expected:
            raise ValueError(f"{source} policy_router has invalid {key}")

    frozen_actor = value.get("frozen_actor")
    expected_actor_keys = {
        "filename",
        "sha256",
        "network_sha256",
        "source_actor_sha256",
        "observation_dim",
        "action_dim",
    }
    if not isinstance(frozen_actor, Mapping) or set(frozen_actor) != expected_actor_keys:
        raise ValueError(f"{source} policy_router.frozen_actor has invalid fields")
    if frozen_actor.get("filename") != FROZEN_LIFT_ACTOR_FILENAME:
        raise ValueError(f"{source} policy_router names an unsupported frozen actor file")
    for key in ("sha256", "network_sha256", "source_actor_sha256"):
        if not _is_sha256(frozen_actor.get(key)):
            raise ValueError(f"{source} policy_router.frozen_actor.{key} is invalid")
    for key, expected in (
        ("observation_dim", PICK_TOOL_OBSERVATION_DIM),
        ("action_dim", PICK_TOOL_ACTION_DIM),
    ):
        candidate = frozen_actor.get(key)
        if (
            not isinstance(candidate, int)
            or isinstance(candidate, bool)
            or candidate != expected
        ):
            raise ValueError(f"{source} policy_router.frozen_actor.{key} is invalid")
    return {
        "kind": PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_KIND,
        "observation_index": PICK_TOOL_LATCH_OBSERVATION_INDEX,
        "close_value": 0.0,
        "frozen_value": 1.0,
        "trainable_action_slice": trainable_slice,
        "close_arm_fill": "exact_zero",
        "frozen_action_slice": frozen_slice,
        "frozen_action_sampling": "deterministic_tanh_mean",
        "frozen_action_entropy": "exact_zero",
        "frozen_actor": {
            "filename": FROZEN_LIFT_ACTOR_FILENAME,
            "sha256": str(frozen_actor["sha256"]),
            "network_sha256": str(frozen_actor["network_sha256"]),
            "source_actor_sha256": str(frozen_actor["source_actor_sha256"]),
            "observation_dim": PICK_TOOL_OBSERVATION_DIM,
            "action_dim": PICK_TOOL_ACTION_DIM,
        },
    }


def public_latch_frozen_actor_router_contract(
    *,
    sidecar_sha256: str,
    network_sha256: str,
    source_actor_sha256: str,
) -> dict[str, Any]:
    candidate = {
        "kind": PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_KIND,
        "observation_index": PICK_TOOL_LATCH_OBSERVATION_INDEX,
        "close_value": 0.0,
        "frozen_value": 1.0,
        "trainable_action_slice": [PICK_TOOL_ARM_ACTION_DIM, PICK_TOOL_ACTION_DIM],
        "close_arm_fill": "exact_zero",
        "frozen_action_slice": [0, PICK_TOOL_ACTION_DIM],
        "frozen_action_sampling": "deterministic_tanh_mean",
        "frozen_action_entropy": "exact_zero",
        "frozen_actor": {
            "filename": FROZEN_LIFT_ACTOR_FILENAME,
            "sha256": sidecar_sha256,
            "network_sha256": network_sha256,
            "source_actor_sha256": source_actor_sha256,
            "observation_dim": PICK_TOOL_OBSERVATION_DIM,
            "action_dim": PICK_TOOL_ACTION_DIM,
        },
    }
    normalized = validate_policy_router_contract(
        candidate,
        task_mode=FULL_TASK_MODE,
        policy_action_authority=PUBLIC_LATCH_ARM_ACTION_AUTHORITY,
        runtime=runtime_contract(FULL_TASK_MODE),
        source="policy router builder",
    )
    assert normalized is not None
    return normalized


def policy_router_semantics(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Strip instance fingerprints while retaining every routing decision."""

    if value is None:
        return None
    frozen_actor = value["frozen_actor"]
    return {
        **{key: item for key, item in value.items() if key != "frozen_actor"},
        "frozen_actor": {
            key: item
            for key, item in frozen_actor.items()
            if key not in {"sha256", "network_sha256", "source_actor_sha256"}
        },
    }


def expected_policy_router_semantics(enabled: bool) -> dict[str, Any] | None:
    if not isinstance(enabled, bool):
        raise TypeError("policy router enabled flag must be bool")
    if not enabled:
        return None
    placeholder = "0" * 64
    return policy_router_semantics(
        public_latch_frozen_actor_router_contract(
            sidecar_sha256=placeholder,
            network_sha256=placeholder,
            source_actor_sha256=placeholder,
        )
    )


def task_mode_from_close_option(
    close_option_mode: bool,
    power_close_option_mode: bool = False,
    coupled_power_align_close_option_mode: bool = False,
    coupled_teacher_residual_mode: bool = False,
) -> str:
    """Resolve the mutually-exclusive task flags into a checkpoint mode."""

    selected = sum(
        (
            bool(close_option_mode),
            bool(power_close_option_mode),
            bool(coupled_power_align_close_option_mode),
            bool(coupled_teacher_residual_mode),
        )
    )
    if selected > 1:
        raise ValueError(
            "--close_option_mode, --power_close_option_mode, and "
            "--coupled_power_align_close_option_mode, and "
            "--coupled_teacher_residual_mode are mutually exclusive"
        )
    if coupled_teacher_residual_mode:
        return COUPLED_TEACHER_RESIDUAL_TASK_MODE
    if coupled_power_align_close_option_mode:
        return COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE
    if power_close_option_mode:
        return POWER_CLOSE_OPTION_TASK_MODE
    return CLOSE_OPTION_TASK_MODE if close_option_mode else FULL_TASK_MODE


def is_close_option_task_mode(task_mode: str) -> bool:
    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    return task_mode in (
        CLOSE_OPTION_TASK_MODE,
        POWER_CLOSE_OPTION_TASK_MODE,
        COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
        COUPLED_TEACHER_RESIDUAL_TASK_MODE,
    )


def is_coupled_task_mode(task_mode: str) -> bool:
    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    return task_mode in (
        COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
        COUPLED_TEACHER_RESIDUAL_TASK_MODE,
    )


def resolve_default_episode_length_s(
    *,
    task_mode: str,
    smoke: bool,
) -> float | None:
    """Return a mode-safe default horizon without censoring coupled success."""

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    if is_coupled_task_mode(task_mode):
        return 3.0 if smoke else 5.0
    if is_close_option_task_mode(task_mode):
        return 0.5 if smoke else 5.0
    return 0.12 if smoke else None


def environment_task_mode_overrides(task_mode: str) -> dict[str, bool | int]:
    """Translate a checkpoint task mode to the three independent env switches."""

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    coupled = is_coupled_task_mode(task_mode)
    power = task_mode == POWER_CLOSE_OPTION_TASK_MODE or coupled
    overrides: dict[str, bool | int] = {
        "close_option_mode": is_close_option_task_mode(task_mode),
        "power_close_option_mode": power,
        "coupled_power_align_close_option_mode": coupled,
    }
    if coupled:
        overrides.update(
            {
                "observation_space": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
                "state_space": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
            }
        )
    return overrides


def resolve_smoke_interaction_steps(*, requested: int, task_mode: str) -> int:
    """Keep legacy smoke tiny but exercise both phases of the coupled option."""

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    if is_coupled_task_mode(task_mode):
        return 64
    return min(requested, 8)


def policy_action_contract(task_mode: str) -> dict[str, Any]:
    """Return the policy/environment action boundary fixed by a task mode."""

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    if task_mode in (
        POWER_CLOSE_OPTION_TASK_MODE,
        COUPLED_TEACHER_RESIDUAL_TASK_MODE,
    ):
        return {
            "policy_action_dim": PICK_TOOL_HAND_ACTION_DIM,
            "policy_action_layout": (
                TEACHER_RESIDUAL_POLICY_ACTION_LAYOUT
                if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
                else HAND_POLICY_ACTION_LAYOUT
            ),
            "environment_action_dim": PICK_TOOL_ENVIRONMENT_ACTION_DIM,
            "action_projection": (
                COUPLED_TEACHER_RESIDUAL_ACTION_PROJECTION
                if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
                else PREPEND_ZERO_ARM_ACTION_PROJECTION
            ),
        }
    return {
        "policy_action_dim": PICK_TOOL_ACTION_DIM,
        "policy_action_layout": FULL_POLICY_ACTION_LAYOUT,
        "environment_action_dim": PICK_TOOL_ENVIRONMENT_ACTION_DIM,
        "action_projection": IDENTITY_ACTION_PROJECTION,
    }


def action_noise_group_specs(task_mode: str) -> tuple[tuple[Any, ...], ...]:
    """Return grouped exploration slices in policy-action coordinates."""

    if task_mode in (
        POWER_CLOSE_OPTION_TASK_MODE,
        COUPLED_TEACHER_RESIDUAL_TASK_MODE,
    ):
        return POWER_ACTION_NOISE_GROUP_SPECS
    if task_mode in (
        FULL_TASK_MODE,
        CLOSE_OPTION_TASK_MODE,
        COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
    ):
        return FULL_ACTION_NOISE_GROUP_SPECS
    raise ValueError(f"unsupported task_mode={task_mode!r}")


def observation_contract(task_mode: str) -> dict[str, Any]:
    """Return the actor-visible observation semantics fixed by a task mode."""

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    if is_coupled_task_mode(task_mode):
        return {
            "observation_dim": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
            "observation_contract": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_CONTRACT,
        }
    return {
        "observation_dim": PICK_TOOL_OBSERVATION_DIM,
        "observation_contract": (
            POWER_CLOSE_OBSERVATION_CONTRACT
            if task_mode == POWER_CLOSE_OPTION_TASK_MODE
            else PICK_TOOL_OBSERVATION_CONTRACT
        ),
    }


def runtime_contract(task_mode: str) -> dict[str, Any]:
    """Return every non-discount tensor-layout contract for a task mode."""

    return {
        **observation_contract(task_mode),
        **policy_action_contract(task_mode),
    }


def _validate_serialized_runtime_contract(
    payload: Mapping[str, Any],
    *,
    task_mode: str,
    path: Path,
) -> dict[str, Any]:
    expected = runtime_contract(task_mode)
    actual: dict[str, Any] = {}
    for key, expected_value in expected.items():
        value = payload.get(key)
        if isinstance(expected_value, int):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"checkpoint task contract {path} has invalid {key}")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"checkpoint task contract {path} has invalid {key}")
        if value != expected_value:
            raise ValueError(
                f"checkpoint task contract {path} has {key}={value!r}, "
                f"expected {expected_value!r} for task_mode={task_mode!r}"
            )
        actual[key] = value
    return actual


def validate_teacher_residual_training_state(
    payload: Mapping[str, Any],
    *,
    source: str = "teacher residual training state",
) -> dict[str, Any]:
    """Validate the resumable actor-authority gate for residual training."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"{source} must be a JSON object")
    expected_keys = {
        "version",
        "task_mode",
        "actor_unlock_successes",
        "native_strict_successes",
        "actor_unlocked",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            f"{source} fields differ from the closed contract: "
            f"missing={sorted(expected_keys.difference(payload))}, "
            f"extra={sorted(set(payload).difference(expected_keys))}"
        )
    if payload.get("version") != 1:
        raise ValueError(f"{source}.version must be 1")
    if payload.get("task_mode") != COUPLED_TEACHER_RESIDUAL_TASK_MODE:
        raise ValueError(
            f"{source}.task_mode must be {COUPLED_TEACHER_RESIDUAL_TASK_MODE!r}"
        )
    threshold = payload.get("actor_unlock_successes")
    successes = payload.get("native_strict_successes")
    unlocked = payload.get("actor_unlocked")
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or threshold < 1
    ):
        raise ValueError(f"{source}.actor_unlock_successes must be positive")
    if (
        not isinstance(successes, int)
        or isinstance(successes, bool)
        or successes < 0
    ):
        raise ValueError(f"{source}.native_strict_successes must be non-negative")
    if not isinstance(unlocked, bool):
        raise TypeError(f"{source}.actor_unlocked must be bool")
    expected_unlocked = successes >= threshold
    if unlocked is not expected_unlocked:
        raise ValueError(
            f"{source}.actor_unlocked={unlocked} disagrees with "
            f"native_strict_successes={successes} and threshold={threshold}"
        )
    return {
        "version": 1,
        "task_mode": COUPLED_TEACHER_RESIDUAL_TASK_MODE,
        "actor_unlock_successes": threshold,
        "native_strict_successes": successes,
        "actor_unlocked": unlocked,
    }


def residual_actor_should_unlock(
    *,
    native_strict_successes: int,
    actor_unlock_successes: int,
    success_transition_in_replay: bool,
) -> bool:
    """Authorize learning only after the threshold transition is in replay."""

    for name, value, minimum in (
        ("native_strict_successes", native_strict_successes, 0),
        ("actor_unlock_successes", actor_unlock_successes, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(success_transition_in_replay, bool):
        raise TypeError("success_transition_in_replay must be bool")
    return (
        success_transition_in_replay
        and native_strict_successes >= actor_unlock_successes
    )


def read_teacher_residual_training_state(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read and hash-bind the residual actor gate sidecar."""

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"checkpoint teacher residual state is missing or not a regular file: {path}"
        )
    try:
        raw_bytes = path.read_bytes()
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                f"checkpoint teacher residual state SHA256 mismatch: {path}"
            )
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read checkpoint teacher residual state {path}: {error}"
        ) from error
    return validate_teacher_residual_training_state(payload, source=str(path))


def require_complete_core_checkpoint(checkpoint: Path) -> None:
    """Require the portable network files used to identify a legacy checkpoint."""

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    missing = [
        filename
        for filename in CORE_CHECKPOINT_FILENAMES
        if not (checkpoint / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"incomplete checkpoint {checkpoint}: missing regular files {missing}"
        )


def read_checkpoint_task_contract(checkpoint: Path) -> dict[str, Any]:
    """Read a task contract, treating pre-contract checkpoints as full-task."""

    incomplete = checkpoint / INCOMPLETE_CHECKPOINT_FILENAME
    if incomplete.exists() or incomplete.is_symlink():
        raise ValueError(
            f"checkpoint {checkpoint} is marked incomplete by {incomplete.name}; "
            "refusing a partial or concurrently-written snapshot"
        )
    require_complete_core_checkpoint(checkpoint)
    path = checkpoint / TASK_CONTRACT_FILENAME
    if path.is_symlink():
        raise ValueError(f"checkpoint task contract must not be a symlink: {path}")
    if not path.exists():
        unexpected_sidecars = [
            checkpoint / filename
            for filename in (
                TEACHER_ACTION_PRIOR_FILENAME,
                TEACHER_RESIDUAL_STATE_FILENAME,
                FROZEN_LIFT_ACTOR_FILENAME,
            )
            if (checkpoint / filename).exists()
            or (checkpoint / filename).is_symlink()
        ]
        if unexpected_sidecars:
            raise ValueError(
                "checkpoint has semantic sidecars but no task contract: "
                f"{unexpected_sidecars}"
            )
        # A missing contract is only backward-compatible evidence when the
        # directory is demonstrably a complete pre-contract network snapshot.
        return {
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
    if not path.is_file():
        raise ValueError(f"checkpoint task contract is not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read checkpoint task contract {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint task contract {path} must contain a JSON object")
    version = payload.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in range(1, TASK_CONTRACT_VERSION + 1)
    ):
        raise ValueError(
            f"checkpoint task contract {path} has version={version!r}, "
            f"expected an integer in [1, {TASK_CONTRACT_VERSION}]"
        )
    task_mode = payload.get("task_mode")
    if task_mode not in TASK_MODES:
        raise ValueError(
            f"checkpoint task contract {path} has unsupported task_mode={task_mode!r}"
        )
    if version < 3 and task_mode in (
        POWER_CLOSE_OPTION_TASK_MODE,
        COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
        COUPLED_TEACHER_RESIDUAL_TASK_MODE,
    ):
        raise ValueError(
            f"checkpoint task contract {path} cannot use task_mode={task_mode!r} "
            f"before version 3"
        )
    if version < 4 and task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE:
        raise ValueError(
            f"checkpoint task contract {path} cannot use task_mode={task_mode!r} "
            "before version 4"
        )
    replay_n_step: int | None = None
    replay_gamma: float | None = None
    if version >= 2:
        replay_n_step = payload.get("replay_n_step")
        replay_gamma = payload.get("replay_gamma")
        if not isinstance(replay_n_step, int) or isinstance(replay_n_step, bool) or replay_n_step < 1:
            raise ValueError(f"checkpoint task contract {path} has invalid replay_n_step")
        if not isinstance(replay_gamma, (int, float)) or isinstance(replay_gamma, bool):
            raise ValueError(f"checkpoint task contract {path} has invalid replay_gamma")
        replay_gamma = float(replay_gamma)
        if not math.isfinite(replay_gamma) or not 0.0 <= replay_gamma <= 1.0:
            raise ValueError(f"checkpoint task contract {path} has invalid replay_gamma")
    serialized_runtime = (
        _validate_serialized_runtime_contract(
            payload,
            task_mode=str(task_mode),
            path=path,
        )
        if version >= 3
        else runtime_contract(str(task_mode))
    )
    serialized_authority = validate_policy_action_authority_contract(
        payload.get("policy_action_authority") if version >= 5 else [],
        task_mode=str(task_mode),
        source=f"checkpoint task contract {path}",
    )
    if version < 6 and "policy_router" in payload:
        raise ValueError(
            f"checkpoint task contract {path} declares policy_router before version 6"
        )
    if version >= 6 and "policy_router" not in payload:
        raise ValueError(f"checkpoint task contract {path} is missing policy_router")
    serialized_router = validate_policy_router_contract(
        payload.get("policy_router") if version >= 6 else None,
        task_mode=str(task_mode),
        policy_action_authority=serialized_authority,
        runtime=serialized_runtime,
        source=f"checkpoint task contract {path}",
    )
    frozen_actor_path = checkpoint / FROZEN_LIFT_ACTOR_FILENAME
    if serialized_router is None:
        if frozen_actor_path.exists() or frozen_actor_path.is_symlink():
            raise ValueError(
                "checkpoint without a policy router contains a frozen actor sidecar: "
                f"{frozen_actor_path}"
            )
    else:
        if not frozen_actor_path.is_file() or frozen_actor_path.is_symlink():
            raise FileNotFoundError(
                "checkpoint frozen actor is missing or not a regular file: "
                f"{frozen_actor_path}"
            )
        expected_sidecar_sha256 = serialized_router["frozen_actor"]["sha256"]
        if _sha256(frozen_actor_path) != expected_sidecar_sha256:
            raise ValueError(
                f"checkpoint frozen actor SHA256 mismatch: {frozen_actor_path}"
            )
    teacher_action_prior: dict[str, str] | None = None
    teacher_residual_state: dict[str, Any] | None = None
    prior_payload = payload.get("teacher_action_prior")
    state_reference = payload.get("teacher_residual_state")
    if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE:
        if not isinstance(prior_payload, Mapping) or set(prior_payload) != {
            "filename",
            "sha256",
        }:
            raise ValueError(
                f"checkpoint task contract {path} has invalid teacher_action_prior"
            )
        if prior_payload.get("filename") != TEACHER_ACTION_PRIOR_FILENAME:
            raise ValueError(
                f"checkpoint task contract {path} names an unsupported teacher prior file"
            )
        prior_sha = prior_payload.get("sha256")
        if (
            not isinstance(prior_sha, str)
            or len(prior_sha) != 64
            or any(character not in "0123456789abcdef" for character in prior_sha)
        ):
            raise ValueError(
                f"checkpoint task contract {path} has invalid teacher prior SHA256"
            )
        teacher_action_prior = {
            "filename": TEACHER_ACTION_PRIOR_FILENAME,
            "sha256": prior_sha,
        }
        prior_path = checkpoint / TEACHER_ACTION_PRIOR_FILENAME
        if not prior_path.is_file() or prior_path.is_symlink():
            raise FileNotFoundError(
                f"checkpoint teacher prior is missing or not a regular file: {prior_path}"
            )
        if _sha256(prior_path) != prior_sha:
            raise ValueError(f"checkpoint teacher prior SHA256 mismatch: {prior_path}")
        if not isinstance(state_reference, Mapping) or set(state_reference) != {
            "filename",
            "sha256",
        }:
            raise ValueError(
                f"checkpoint task contract {path} has invalid teacher_residual_state"
            )
        if state_reference.get("filename") != TEACHER_RESIDUAL_STATE_FILENAME:
            raise ValueError(
                f"checkpoint task contract {path} names an unsupported residual state file"
            )
        state_sha = state_reference.get("sha256")
        if (
            not isinstance(state_sha, str)
            or len(state_sha) != 64
            or any(character not in "0123456789abcdef" for character in state_sha)
        ):
            raise ValueError(
                f"checkpoint task contract {path} has invalid residual state SHA256"
            )
        teacher_residual_state = read_teacher_residual_training_state(
            checkpoint / TEACHER_RESIDUAL_STATE_FILENAME,
            expected_sha256=state_sha,
        )
    elif prior_payload is not None:
        raise ValueError(
            f"checkpoint task contract {path} has a teacher prior outside residual mode"
        )
    elif state_reference is not None:
        raise ValueError(
            f"checkpoint task contract {path} has residual state outside residual mode"
        )
    if task_mode != COUPLED_TEACHER_RESIDUAL_TASK_MODE:
        for filename in (
            TEACHER_ACTION_PRIOR_FILENAME,
            TEACHER_RESIDUAL_STATE_FILENAME,
        ):
            unexpected = checkpoint / filename
            if unexpected.exists() or unexpected.is_symlink():
                raise ValueError(
                    f"checkpoint has unexpected residual sidecar outside residual mode: {unexpected}"
                )
    return {
        "version": int(version),
        "task_mode": str(task_mode),
        "legacy_checkpoint": False,
        "replay_n_step": replay_n_step,
        "replay_gamma": replay_gamma,
        "policy_action_authority": serialized_authority,
        "policy_router": serialized_router,
        "teacher_action_prior": teacher_action_prior,
        "teacher_residual_state": teacher_residual_state,
        **serialized_runtime,
    }


def write_checkpoint_task_contract(
    checkpoint: Path,
    *,
    task_mode: str,
    replay_n_step: int,
    replay_gamma: float,
    policy_action_dim: int | None = None,
    policy_action_layout: str | None = None,
    environment_action_dim: int | None = None,
    action_projection: str | None = None,
    observation_dim: int | None = None,
    observation_contract_name: str | None = None,
    policy_action_authority: Sequence[Mapping[str, Any]] = (),
    policy_router: Mapping[str, Any] | None = None,
    teacher_action_prior_sha256: str | None = None,
    teacher_residual_state_sha256: str | None = None,
) -> None:
    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    if (
        not isinstance(replay_n_step, int)
        or isinstance(replay_n_step, bool)
        or replay_n_step < 1
    ):
        raise ValueError("replay_n_step must be a positive integer")
    if not math.isfinite(replay_gamma) or not 0.0 <= replay_gamma <= 1.0:
        raise ValueError("replay_gamma must be finite and in [0, 1]")
    expected_runtime = runtime_contract(task_mode)
    supplied_runtime = {
        "policy_action_dim": policy_action_dim,
        "policy_action_layout": policy_action_layout,
        "environment_action_dim": environment_action_dim,
        "action_projection": action_projection,
        "observation_dim": observation_dim,
        "observation_contract": observation_contract_name,
    }
    resolved_runtime = {
        key: expected_runtime[key] if value is None else value
        for key, value in supplied_runtime.items()
    }
    resolved_authority = validate_policy_action_authority_contract(
        policy_action_authority,
        task_mode=task_mode,
        source="checkpoint writer",
    )
    for key, expected in expected_runtime.items():
        if resolved_runtime[key] != expected:
            raise ValueError(
                f"{key}={resolved_runtime[key]!r} is incompatible with "
                f"task_mode={task_mode!r}; expected {expected!r}"
            )
    resolved_router = validate_policy_router_contract(
        policy_router,
        task_mode=task_mode,
        policy_action_authority=resolved_authority,
        runtime=resolved_runtime,
        source="checkpoint writer",
    )
    frozen_actor_path = checkpoint / FROZEN_LIFT_ACTOR_FILENAME
    if resolved_router is None:
        if frozen_actor_path.exists() or frozen_actor_path.is_symlink():
            raise ValueError(
                "checkpoint writer found a frozen actor sidecar without a policy router: "
                f"{frozen_actor_path}"
            )
    else:
        if not frozen_actor_path.is_file() or frozen_actor_path.is_symlink():
            raise FileNotFoundError(
                "checkpoint writer requires a regular frozen actor sidecar: "
                f"{frozen_actor_path}"
            )
        if _sha256(frozen_actor_path) != resolved_router["frozen_actor"]["sha256"]:
            raise ValueError(
                f"checkpoint writer frozen actor SHA256 mismatch: {frozen_actor_path}"
            )
    prior_entry: dict[str, str] | None = None
    state_entry: dict[str, str] | None = None
    if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE:
        if (
            not isinstance(teacher_action_prior_sha256, str)
            or len(teacher_action_prior_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in teacher_action_prior_sha256
            )
        ):
            raise ValueError("residual task checkpoints require a valid teacher prior SHA256")
        prior_entry = {
            "filename": TEACHER_ACTION_PRIOR_FILENAME,
            "sha256": teacher_action_prior_sha256,
        }
        if (
            not isinstance(teacher_residual_state_sha256, str)
            or len(teacher_residual_state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in teacher_residual_state_sha256
            )
        ):
            raise ValueError(
                "residual task checkpoints require a valid residual state SHA256"
            )
        state_entry = {
            "filename": TEACHER_RESIDUAL_STATE_FILENAME,
            "sha256": teacher_residual_state_sha256,
        }
    elif (
        teacher_action_prior_sha256 is not None
        or teacher_residual_state_sha256 is not None
    ):
        raise ValueError(
            "teacher prior/residual state SHA256 is only valid for residual task checkpoints"
        )
    payload: dict[str, Any] = {
        "version": TASK_CONTRACT_VERSION,
        "task_mode": task_mode,
        "replay_n_step": replay_n_step,
        "replay_gamma": replay_gamma,
        "policy_action_authority": resolved_authority,
        "policy_router": resolved_router,
        **resolved_runtime,
    }
    if prior_entry is not None:
        payload["teacher_action_prior"] = prior_entry
    if state_entry is not None:
        payload["teacher_residual_state"] = state_entry
    atomic_write_json(checkpoint / TASK_CONTRACT_FILENAME, payload)


def clear_stale_checkpoint_optional_artifacts(checkpoint: Path) -> tuple[str, ...]:
    """Remove only optional artifacts whose absence is meaningful on a re-save."""

    if checkpoint.is_symlink():
        raise ValueError(f"checkpoint output must be a real directory: {checkpoint}")
    if not checkpoint.exists():
        return ()
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint output must be a real directory: {checkpoint}")

    removed: list[str] = []
    for filename in STALE_OPTIONAL_CHECKPOINT_FILENAMES:
        path = checkpoint / filename
        if path.is_symlink() or path.is_file():
            path.unlink()
            removed.append(filename)
        elif path.exists():
            raise ValueError(f"stale checkpoint artifact is not a regular file: {path}")
    return tuple(removed)


def save_final_checkpoint(
    checkpoint: Path,
    *,
    agent: Any,
    task_mode: str,
    replay_n_step: int,
    replay_gamma: float,
    save_replay: bool,
    actor_rehearsal: Any | None,
    policy_action_authority: Sequence[Mapping[str, Any]] = (),
    policy_router_enabled: bool = False,
    teacher_prior: Any | None = None,
    teacher_residual_state: Mapping[str, Any] | None = None,
) -> None:
    """Save one final checkpoint, publishing its task contract last."""

    prior_payload: dict[str, Any] | None = None
    residual_state_payload: dict[str, Any] | None = None
    if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE:
        if teacher_prior is None or teacher_residual_state is None:
            raise ValueError(
                "residual checkpoints require both teacher_prior and resumable actor gate state"
            )
        candidate = teacher_prior.contract_payload()
        if not isinstance(candidate, Mapping):
            raise TypeError("teacher_prior.contract_payload() must return a mapping")
        from coupled_teacher_prior import coupled_teacher_prior_from_payload

        validated_prior = coupled_teacher_prior_from_payload(
            candidate,
            source="checkpoint teacher prior",
        )
        prior_payload = validated_prior.contract_payload()
        residual_state_payload = validate_teacher_residual_training_state(
            teacher_residual_state,
            source="checkpoint teacher residual state",
        )
    elif teacher_prior is not None or teacher_residual_state is not None:
        raise ValueError(
            "teacher prior/residual state can only be saved for residual task checkpoints"
        )

    if checkpoint.is_symlink():
        raise ValueError(f"checkpoint output must be a real directory: {checkpoint}")
    checkpoint.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint output must be a real directory: {checkpoint}")
    incomplete = checkpoint / INCOMPLETE_CHECKPOINT_FILENAME
    atomic_write_json(incomplete, {"version": 1, "status": "incomplete"})
    clear_stale_checkpoint_optional_artifacts(checkpoint)
    agent.save(str(checkpoint))
    policy_router: dict[str, Any] | None = None
    frozen_actor_path = checkpoint / FROZEN_LIFT_ACTOR_FILENAME
    if policy_router_enabled:
        network_sha256 = getattr(agent, "frozen_lift_actor_sha256", None)
        source_actor_sha256 = getattr(
            agent, "frozen_lift_actor_source_sha256", None
        )
        if not frozen_actor_path.is_file() or frozen_actor_path.is_symlink():
            raise FileNotFoundError(
                "routed agent did not save a regular frozen actor sidecar: "
                f"{frozen_actor_path}"
            )
        policy_router = public_latch_frozen_actor_router_contract(
            sidecar_sha256=_sha256(frozen_actor_path),
            network_sha256=network_sha256,
            source_actor_sha256=source_actor_sha256,
        )
    elif frozen_actor_path.exists() or frozen_actor_path.is_symlink():
        raise ValueError(
            "non-routed agent unexpectedly saved a frozen actor sidecar: "
            f"{frozen_actor_path}"
        )
    if save_replay:
        agent.save_replay_buffer(str(checkpoint))
    if actor_rehearsal is not None:
        actor_rehearsal.save(checkpoint / "actor_rehearsal.pt")
    teacher_prior_sha256: str | None = None
    teacher_residual_state_sha256: str | None = None
    if prior_payload is not None:
        prior_path = checkpoint / TEACHER_ACTION_PRIOR_FILENAME
        atomic_write_json(prior_path, prior_payload)
        teacher_prior_sha256 = _sha256(prior_path)
    if residual_state_payload is not None:
        state_path = checkpoint / TEACHER_RESIDUAL_STATE_FILENAME
        atomic_write_json(state_path, residual_state_payload)
        teacher_residual_state_sha256 = _sha256(state_path)
    write_checkpoint_task_contract(
        checkpoint,
        task_mode=task_mode,
        replay_n_step=replay_n_step,
        replay_gamma=replay_gamma,
        policy_action_authority=policy_action_authority,
        policy_router=policy_router,
        teacher_action_prior_sha256=teacher_prior_sha256,
        teacher_residual_state_sha256=teacher_residual_state_sha256,
    )
    incomplete.unlink()


def read_legacy_replay_discount_contract(checkpoint: Path) -> tuple[int, float]:
    """Read mixed-replay metadata lazily without materializing its tensor payload."""

    path = checkpoint / "replay_buffer.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"cannot audit legacy replay contract {path}: {error}") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("version") not in (1, 2, 3)
        or not isinstance(payload.get("online"), Mapping)
        or not isinstance(payload.get("demos"), Mapping)
    ):
        raise ValueError(
            f"legacy replay {path} is not a supported mixed replay with auditable metadata; "
            "refusing an unsafe resume"
        )
    online = payload["online"]
    demos = payload["demos"]
    n_step = online.get("n_step")
    gamma = online.get("gamma")
    if not isinstance(n_step, int) or isinstance(n_step, bool) or n_step < 1:
        raise ValueError(f"legacy replay {path} has invalid n_step")
    if not isinstance(gamma, (int, float)) or isinstance(gamma, bool):
        raise ValueError(f"legacy replay {path} has invalid gamma")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"legacy replay {path} has invalid gamma")
    demo_n_step = demos.get("n_step")
    demo_gamma = demos.get("gamma")
    if (
        demo_n_step != n_step
        or not isinstance(demo_gamma, (int, float))
        or isinstance(demo_gamma, bool)
        or not math.isclose(
            float(demo_gamma), gamma, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise ValueError(f"legacy replay {path} has inconsistent online/demo discount metadata")
    return n_step, gamma


def validate_replay_task_contract(
    checkpoint: Path,
    *,
    task_mode: str,
    n_step: int,
    gamma: float,
    policy_action_authority: Sequence[Mapping[str, Any]] = (),
    policy_router_enabled: bool = False,
) -> dict[str, Any]:
    """Reject replay collected under different reward/termination semantics."""

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported current task_mode={task_mode!r}")
    if not isinstance(n_step, int) or isinstance(n_step, bool) or n_step < 1:
        raise ValueError("current replay n_step must be a positive integer")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("current replay gamma must be finite and in [0, 1]")
    replay_path = checkpoint / "replay_buffer.pt"
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)
    contract = read_checkpoint_task_contract(checkpoint)
    current_authority = validate_policy_action_authority_contract(
        policy_action_authority,
        task_mode=task_mode,
        source="current replay",
    )
    if contract["task_mode"] != task_mode:
        raise ValueError(
            "--resume_replay cannot cross task modes: "
            f"checkpoint={contract['task_mode']!r}, current={task_mode!r}"
        )
    current_runtime = runtime_contract(task_mode)
    for key, expected in current_runtime.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"--resume_replay {key} mismatch: checkpoint={contract.get(key)!r}, "
                f"current={expected!r}"
            )
    if contract.get("policy_action_authority") != current_authority:
        raise ValueError(
            "--resume_replay policy_action_authority mismatch: "
            f"checkpoint={contract.get('policy_action_authority')!r}, "
            f"current={current_authority!r}"
        )
    checkpoint_router_semantics = policy_router_semantics(contract.get("policy_router"))
    current_router_semantics = expected_policy_router_semantics(policy_router_enabled)
    if checkpoint_router_semantics != current_router_semantics:
        raise ValueError(
            "--resume_replay policy_router mismatch: "
            f"checkpoint={checkpoint_router_semantics!r}, "
            f"current={current_router_semantics!r}"
        )
    checkpoint_n_step = contract["replay_n_step"]
    checkpoint_gamma = contract["replay_gamma"]
    if checkpoint_n_step is None or checkpoint_gamma is None:
        checkpoint_n_step, checkpoint_gamma = read_legacy_replay_discount_contract(checkpoint)
    if checkpoint_n_step != n_step:
        raise ValueError(
            "--resume_replay n_step mismatch: "
            f"checkpoint={checkpoint_n_step}, current={n_step}"
        )
    if not math.isclose(checkpoint_gamma, gamma, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "--resume_replay gamma mismatch: "
            f"checkpoint={checkpoint_gamma}, current={gamma}"
        )
    return contract


def validate_checkpoint_task_contract(
    checkpoint: Path,
    *,
    task_mode: str,
    n_step: int,
    gamma: float,
    policy_action_authority: Sequence[Mapping[str, Any]] = (),
    policy_router_enabled: bool = False,
) -> dict[str, Any]:
    """Reject a full-agent restore across reward/termination objectives.

    A full checkpoint also restores the critic, target critic, temperature,
    optimizers, scheduler, and reward normalizer. Those states are just as
    objective-specific as replay. Cross-mode transfer must therefore use the
    deliberately actor-only ``--actor_checkpoint`` path.
    """

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported current task_mode={task_mode!r}")
    if not isinstance(n_step, int) or isinstance(n_step, bool) or n_step < 1:
        raise ValueError("current checkpoint n_step must be a positive integer")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("current checkpoint gamma must be finite and in [0, 1]")
    contract = read_checkpoint_task_contract(checkpoint)
    current_authority = validate_policy_action_authority_contract(
        policy_action_authority,
        task_mode=task_mode,
        source="current checkpoint",
    )
    if contract["task_mode"] != task_mode:
        raise ValueError(
            "--checkpoint cannot cross task modes because it restores objective-specific "
            "critic/optimizer/normalizer state; use --actor_checkpoint instead: "
            f"checkpoint={contract['task_mode']!r}, current={task_mode!r}"
        )
    current_runtime = runtime_contract(task_mode)
    for key, expected in current_runtime.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"--checkpoint {key} mismatch; use --actor_checkpoint instead: "
                f"checkpoint={contract.get(key)!r}, current={expected!r}"
            )
    if contract.get("policy_action_authority") != current_authority:
        raise ValueError(
            "--checkpoint policy_action_authority mismatch; use --actor_checkpoint "
            "instead: "
            f"checkpoint={contract.get('policy_action_authority')!r}, "
            f"current={current_authority!r}"
        )
    checkpoint_router_semantics = policy_router_semantics(contract.get("policy_router"))
    current_router_semantics = expected_policy_router_semantics(policy_router_enabled)
    if checkpoint_router_semantics != current_router_semantics:
        raise ValueError(
            "--checkpoint policy_router mismatch; use --actor_checkpoint instead: "
            f"checkpoint={checkpoint_router_semantics!r}, "
            f"current={current_router_semantics!r}"
        )
    checkpoint_n_step = contract["replay_n_step"]
    checkpoint_gamma = contract["replay_gamma"]
    if checkpoint_n_step is None or checkpoint_gamma is None:
        try:
            checkpoint_n_step, checkpoint_gamma = read_legacy_replay_discount_contract(
                checkpoint
            )
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                "legacy --checkpoint has no auditable n_step/gamma contract; use "
                "--actor_checkpoint for a safe actor-only initialization"
            ) from error
    if checkpoint_n_step != n_step or not math.isclose(
        checkpoint_gamma, gamma, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "--checkpoint discount contract mismatch; use --actor_checkpoint instead: "
            f"checkpoint=(n_step={checkpoint_n_step}, gamma={checkpoint_gamma}), "
            f"current=(n_step={n_step}, gamma={gamma})"
        )
    return contract


def validate_training_source_selection(
    *,
    checkpoint: Path | None,
    actor_checkpoint: Path | None,
    resume_replay: bool,
    resume_actor_demo: bool,
    close_option_mode: bool,
    power_close_option_mode: bool,
    demo: list[Path] | None,
    actor_demo: list[Path] | None,
    coupled_power_align_close_option_mode: bool = False,
    coupled_teacher_residual_mode: bool = False,
    allow_cross_task_actor: bool = False,
    public_latch_arm_gate: bool = False,
    public_latch_frozen_lift_router: bool = False,
    frozen_lift_actor_checkpoint: Path | None = None,
) -> None:
    """Validate mutually exclusive initialization and task-specific data sources."""

    if checkpoint is not None and actor_checkpoint is not None:
        raise ValueError("--checkpoint and --actor_checkpoint are mutually exclusive")
    if resume_replay and checkpoint is None:
        raise ValueError("--resume_replay requires --checkpoint")
    if resume_actor_demo and checkpoint is None:
        raise ValueError("--resume_actor_demo requires --checkpoint")
    task_mode = task_mode_from_close_option(
        close_option_mode,
        power_close_option_mode,
        coupled_power_align_close_option_mode,
        coupled_teacher_residual_mode,
    )
    if coupled_teacher_residual_mode:
        if checkpoint is not None and not resume_replay:
            raise ValueError(
                "teacher-residual --checkpoint requires --resume_replay so an "
                "unlocked actor cannot resume without the strict success that "
                "granted its authority"
            )
        if actor_checkpoint is not None:
            raise ValueError(
                "teacher-residual mode rejects absolute-action --actor_checkpoint sources"
            )
        if demo is not None:
            raise ValueError(
                "teacher-residual v1 rejects transition --demo sources; native strict "
                "successes must enter online residual replay. A strictly audited coupled "
                "teacher --actor_demo may only anchor the actor at residual14=0."
            )
    if public_latch_arm_gate:
        if task_mode != FULL_TASK_MODE:
            raise ValueError("--public_latch_arm_gate requires full_task mode")
        if demo is not None:
            raise ValueError(
                "--public_latch_arm_gate rejects transition --demo sources because "
                "their next observations/rewards were generated by ungated actions; "
                "use actor-only --actor_demo supervision"
            )
    if public_latch_frozen_lift_router:
        if task_mode != FULL_TASK_MODE:
            raise ValueError("--public_latch_frozen_lift_router requires full_task mode")
        if not public_latch_arm_gate:
            raise ValueError(
                "--public_latch_frozen_lift_router requires --public_latch_arm_gate"
            )
        if checkpoint is None:
            if actor_checkpoint is None or frozen_lift_actor_checkpoint is None:
                raise ValueError(
                    "fresh routed training requires both --actor_checkpoint and "
                    "--frozen_lift_actor_checkpoint"
                )
        elif frozen_lift_actor_checkpoint is not None:
            raise ValueError(
                "routed --checkpoint resume restores its self-contained frozen actor; "
                "do not pass --frozen_lift_actor_checkpoint"
            )
    elif frozen_lift_actor_checkpoint is not None:
        raise ValueError(
            "--frozen_lift_actor_checkpoint requires --public_latch_frozen_lift_router"
        )
    if allow_cross_task_actor and actor_checkpoint is None:
        raise ValueError("--allow_cross_task_actor requires --actor_checkpoint")
    if allow_cross_task_actor and not coupled_power_align_close_option_mode:
        raise ValueError(
            "--allow_cross_task_actor is only implemented for "
            "--coupled_power_align_close_option_mode"
        )
    if (
        close_option_mode
        or power_close_option_mode
        or coupled_power_align_close_option_mode
        or coupled_teacher_residual_mode
    ) and demo is not None:
        raise ValueError(
            "close-option training rejects full-task transition --demo data; "
            "use allowlisted --actor_demo supervision instead"
        )
    if power_close_option_mode and actor_demo is not None:
        raise ValueError(
            "--power_close_option_mode rejects --actor_demo until a strict, "
            "power-authority hand14 dataset contract is implemented"
        )
    if (
        coupled_power_align_close_option_mode
        and actor_demo is not None
        and actor_checkpoint is None
        and not (checkpoint is not None and resume_actor_demo)
    ):
        raise ValueError(
            "--coupled_power_align_close_option_mode with --actor_demo requires either "
            "an initial --actor_checkpoint or --checkpoint plus --resume_actor_demo; "
            "coupled rehearsal must not start the long-horizon bridge from a random actor"
        )


def validate_actor_demo_curriculum_lineage(
    audit: Mapping[str, Any],
    *,
    expected_curriculum_sha256: str,
) -> None:
    """Bind a strict actor demonstration to the active reset curriculum."""

    for label, value in (
        ("expected curriculum", expected_curriculum_sha256),
        ("actor demo curriculum", audit.get("curriculum_dataset_sha256")),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} SHA256 must be 64-character lowercase hex")
    actual = str(audit["curriculum_dataset_sha256"])
    if actual != expected_curriculum_sha256:
        raise ValueError(
            "coupled actor-demo curriculum lineage mismatch: "
            f"demo={actual}, active={expected_curriculum_sha256}, "
            f"source={audit.get('path')!r}"
        )


def validate_teacher_residual_actor_demo_lineage(
    audit: Mapping[str, Any],
    *,
    expected_curriculum_sha256: str,
    expected_teacher_artifact_sha256: str,
) -> None:
    """Bind zero-residual rehearsal to the exact active teacher/reset pair."""

    validate_actor_demo_curriculum_lineage(
        audit,
        expected_curriculum_sha256=expected_curriculum_sha256,
    )
    for label, value in (
        ("expected teacher artifact", expected_teacher_artifact_sha256),
        ("actor demo teacher artifact", audit.get("teacher_artifact_sha256")),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} SHA256 must be 64-character lowercase hex")
    actual = str(audit["teacher_artifact_sha256"])
    if actual != expected_teacher_artifact_sha256:
        raise ValueError(
            "teacher-residual actor-demo lineage mismatch: "
            f"demo={actual}, active={expected_teacher_artifact_sha256}, "
            f"source={audit.get('path')!r}"
        )


def project_coupled_teacher_demo_to_zero_residual(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Represent an audited native teacher trajectory as residual14=0."""

    observation = batch.get("observation")
    action = batch.get("action")
    if not isinstance(observation, torch.Tensor) or not isinstance(action, torch.Tensor):
        raise TypeError("coupled teacher projection requires tensor observation/action")
    if (
        observation.ndim != 2
        or observation.shape[1] != COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM
    ):
        raise ValueError("coupled teacher projection requires 131-D observations")
    if action.shape != (observation.shape[0], PICK_TOOL_ENVIRONMENT_ACTION_DIM):
        raise ValueError("coupled teacher projection requires native 21-D actions")
    if (
        observation.device != action.device
        or observation.dtype != torch.float32
        or action.dtype != torch.float32
    ):
        raise ValueError("coupled teacher projection requires co-located float32 tensors")
    if not bool(torch.isfinite(observation).all()) or not bool(
        torch.isfinite(action).all()
    ):
        raise ValueError("coupled teacher projection received NaN or infinity")
    return {
        "observation": observation,
        "action": torch.zeros(
            (observation.shape[0], PICK_TOOL_HAND_ACTION_DIM),
            dtype=action.dtype,
            device=action.device,
        ),
    }


def validate_checkpoint_output_separation(
    checkpoint: Path | None,
    *,
    output_checkpoint: Path,
) -> None:
    """Never overwrite the only full-agent source while resuming from it."""

    if checkpoint is not None and checkpoint.resolve() == output_checkpoint.resolve():
        raise ValueError(
            "--checkpoint cannot be the current output checkpoint_final; choose a new "
            "--output_dir so the source remains recoverable"
        )


def validate_close_option_training_config(
    *,
    close_option_mode: bool,
    power_close_option_mode: bool,
    curriculum_dataset: Path | None,
    curriculum_boundary: str,
    curriculum_probability: float,
    curriculum_joint_noise: float,
    episode_length_s: float | None,
    randomize_episode_lengths: bool,
    coupled_power_align_close_option_mode: bool = False,
    coupled_teacher_residual_mode: bool = False,
) -> None:
    """Keep the close option on its physically meaningful pregrasp MDP."""

    task_mode_from_close_option(
        close_option_mode,
        power_close_option_mode,
        coupled_power_align_close_option_mode,
        coupled_teacher_residual_mode,
    )
    if not (
        close_option_mode
        or power_close_option_mode
        or coupled_power_align_close_option_mode
        or coupled_teacher_residual_mode
    ):
        return
    flag = (
        "--coupled_teacher_residual_mode"
        if coupled_teacher_residual_mode
        else "--coupled_power_align_close_option_mode"
        if coupled_power_align_close_option_mode
        else (
            "--power_close_option_mode"
            if power_close_option_mode
            else "--close_option_mode"
        )
    )
    if curriculum_dataset is None:
        raise ValueError(f"{flag} requires a close-start curriculum dataset")
    if curriculum_boundary != "close_start":
        raise ValueError(f"{flag} requires --curriculum_boundary close_start")
    if curriculum_probability != 1.0:
        raise ValueError(f"{flag} requires --curriculum_probability 1")
    if not 0.0 <= curriculum_joint_noise <= 0.02:
        raise ValueError(
            f"{flag} requires --curriculum_joint_noise in [0, 0.02]"
        )
    minimum_episode_length_s = (
        3.0
        if (coupled_power_align_close_option_mode or coupled_teacher_residual_mode)
        else 0.40
    )
    if (
        episode_length_s is None
        or not minimum_episode_length_s <= episode_length_s <= 5.0
    ):
        raise ValueError(
            f"{flag} requires --episode_length_s in "
            f"[{minimum_episode_length_s:.2f}, 5] so the "
            "15-frame confirmation window is physically reachable"
        )
    if randomize_episode_lengths:
        raise ValueError(
            f"{flag} rejects --randomize_episode_lengths because shortened "
            "initial episodes can censor the stable-latch confirmation window"
        )


def validate_public_latch_arm_gate_config(
    *,
    enabled: bool,
    task_mode: str,
    curriculum_dataset: Path | None,
    curriculum_boundary: str,
    curriculum_probability: float,
    curriculum_joint_noise: float,
    episode_length_s: float | None,
    randomize_episode_lengths: bool,
    online_search_handoff: bool = False,
) -> None:
    """Keep latch-gated arm training on the close-to-lift conditional MDP."""

    if not enabled:
        return
    if task_mode != FULL_TASK_MODE:
        raise ValueError("--public_latch_arm_gate requires full_task mode")
    if online_search_handoff:
        # The live SEARCH prefix supplies the conditional state without a
        # PhysX snapshot restore.  Its stricter collection contract is checked
        # independently below.
        return
    if curriculum_dataset is None:
        raise ValueError("--public_latch_arm_gate requires a close-start curriculum dataset")
    if curriculum_boundary != "close_start":
        raise ValueError("--public_latch_arm_gate requires --curriculum_boundary close_start")
    if curriculum_probability != 1.0:
        raise ValueError("--public_latch_arm_gate requires --curriculum_probability 1")
    if not 0.0 <= curriculum_joint_noise <= 0.02:
        raise ValueError(
            "--public_latch_arm_gate requires --curriculum_joint_noise in [0, 0.02]"
        )
    if episode_length_s is None or episode_length_s < 0.30:
        raise ValueError(
            "--public_latch_arm_gate requires --episode_length_s >= 0.30"
        )
    if randomize_episode_lengths:
        raise ValueError(
            "--public_latch_arm_gate rejects --randomize_episode_lengths"
        )


def validate_online_search_handoff_training_config(
    *,
    search_checkpoint: Path | None,
    min_score: float,
    hold_steps: int,
    task_mode: str,
    public_latch_arm_gate: bool,
    public_latch_frozen_lift_router: bool,
    curriculum_dataset: Path | None,
    curriculum_probability: float,
    curriculum_joint_noise: float,
    episode_length_s: float | None,
    randomize_episode_lengths: bool,
) -> None:
    """Require live SEARCH handoff training to match deployment dynamics."""

    validate_online_handoff_config(min_score=min_score, hold_steps=hold_steps)
    if search_checkpoint is None:
        return
    if search_checkpoint.is_symlink() or not search_checkpoint.is_file():
        raise FileNotFoundError(
            "--search_handoff_checkpoint must be a regular non-symlink file: "
            f"{search_checkpoint}"
        )
    if task_mode != FULL_TASK_MODE:
        raise ValueError("online SEARCH handoff requires full_task mode")
    if not public_latch_arm_gate or not public_latch_frozen_lift_router:
        raise ValueError(
            "online SEARCH handoff requires --public_latch_arm_gate and "
            "--public_latch_frozen_lift_router so only CLOSE hand14 is trainable"
        )
    if (
        curriculum_dataset is not None
        or curriculum_probability != 0.0
        or curriculum_joint_noise != 0.0
    ):
        raise ValueError(
            "online SEARCH handoff rejects curriculum resets and joint noise"
        )
    if randomize_episode_lengths:
        raise ValueError(
            "online SEARCH handoff rejects randomized horizons"
        )
    if episode_length_s is not None and not math.isclose(
        episode_length_s, 20.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(
            "online SEARCH handoff requires the authored 20 second full-task horizon"
        )


def build_latch_conditioned_noise_scale(
    observation: torch.Tensor,
    *,
    unlatched_arm: float,
    unlatched_hand: float,
    latched_arm: float,
    latched_hand: float,
    action_dim: int = PICK_TOOL_ACTION_DIM,
) -> torch.Tensor:
    """Build per-row action noise multipliers from the actor-visible latch bit."""

    if not isinstance(observation, torch.Tensor):
        raise TypeError("observation must be a torch.Tensor")
    if observation.ndim != 2 or observation.shape[1] <= PICK_TOOL_LATCH_OBSERVATION_INDEX:
        raise ValueError(
            "observation must be [batch, dim] and contain the PickTool latch bit at index "
            f"{PICK_TOOL_LATCH_OBSERVATION_INDEX}"
        )
    if action_dim not in (PICK_TOOL_HAND_ACTION_DIM, PICK_TOOL_ACTION_DIM):
        raise ValueError(
            f"action_dim must be {PICK_TOOL_HAND_ACTION_DIM} (hand-only) or "
            f"{PICK_TOOL_ACTION_DIM} (arm+hand)"
        )
    values = {
        "unlatched_arm": unlatched_arm,
        "unlatched_hand": unlatched_hand,
        "latched_arm": latched_arm,
        "latched_hand": latched_hand,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    latched = observation[:, PICK_TOOL_LATCH_OBSERVATION_INDEX] > 0.5
    scale = torch.empty(
        (observation.shape[0], action_dim),
        dtype=torch.float32,
        device=observation.device,
    )
    arm_scale = torch.full_like(latched, unlatched_arm, dtype=torch.float32)
    hand_scale = torch.full_like(latched, unlatched_hand, dtype=torch.float32)
    arm_scale.masked_fill_(latched, latched_arm)
    hand_scale.masked_fill_(latched, latched_hand)
    if action_dim == PICK_TOOL_HAND_ACTION_DIM:
        scale[:] = hand_scale.unsqueeze(-1)
    else:
        scale[:, :PICK_TOOL_ARM_ACTION_DIM] = arm_scale.unsqueeze(-1)
        scale[:, PICK_TOOL_ARM_ACTION_DIM:] = hand_scale.unsqueeze(-1)
    return scale


@dataclass
class TerminalEventAccumulator:
    """Accumulate reset-before-clone task truth without per-step host sync."""

    num_envs: int
    device: torch.device
    task_mode: str = FULL_TASK_MODE
    counts: dict[str, torch.Tensor] = field(init=False)
    _event_keys: tuple[str, ...] = field(init=False)
    _unlatched_seen: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.task_mode not in TASK_MODES:
            raise ValueError(f"unsupported task_mode={self.task_mode!r}")
        self._event_keys = {
            FULL_TASK_MODE: TERMINAL_EVENT_KEYS,
            CLOSE_OPTION_TASK_MODE: CLOSE_OPTION_TERMINAL_EVENT_KEYS,
            POWER_CLOSE_OPTION_TASK_MODE: POWER_CLOSE_OPTION_TERMINAL_EVENT_KEYS,
            COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE: (
                COUPLED_POWER_ALIGN_CLOSE_OPTION_TERMINAL_EVENT_KEYS
            ),
            COUPLED_TEACHER_RESIDUAL_TASK_MODE: (
                COUPLED_POWER_ALIGN_CLOSE_OPTION_TERMINAL_EVENT_KEYS
            ),
        }[self.task_mode]
        self.counts = {
            name: torch.zeros((), dtype=torch.long, device=self.device)
            for name in self._event_keys
        }
        self._unlatched_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def step(self, info: Mapping[str, Any]) -> None:
        values = info.get("pick_tool_terminal")
        if not isinstance(values, Mapping):
            raise KeyError("adapter info has no pick_tool_terminal ground truth")
        validated: dict[str, torch.Tensor] = {}
        for name in self._event_keys:
            value = values.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"pick_tool_terminal[{name!r}] must be a torch.Tensor")
            if value.shape != (self.num_envs,) or value.dtype != torch.bool:
                raise ValueError(
                    f"pick_tool_terminal[{name!r}] must be bool[{self.num_envs}], "
                    f"got {value.dtype}{tuple(value.shape)}"
                )
            if value.device != self.device:
                raise ValueError(
                    f"pick_tool_terminal[{name!r}] is on {value.device}, expected {self.device}"
                )
            validated[name] = value

        if self.task_mode == FULL_TASK_MODE:
            # Terminal flags are one-step events. Unlatched 5 cm is a state and
            # can persist, so count its first rising occurrence once per episode.
            for name in TERMINAL_EVENT_KEYS[:-1]:
                self.counts[name].add_(validated[name].sum())
            unlatched = validated["unlatched_clearance_ge_5cm"]
            self.counts["unlatched_clearance_ge_5cm"].add_(
                (unlatched & ~self._unlatched_seen).sum()
            )
            self._unlatched_seen |= unlatched
            episode_done = (
                validated["success"] | validated["failure"] | validated["time_out"]
            )
            self._unlatched_seen &= ~episode_done
            return

        # Every close-option field is a reset-before terminal event. Keep its
        # metrics explicitly option-named so a stable latch cannot be mistaken
        # for full-task 20 cm success in training reports.
        for name in self._event_keys:
            self.counts[name].add_(validated[name].sum())

    def metrics(self) -> dict[str, int]:
        return {
            f"pick_tool_terminal/{name}": int(value.item())
            for name, value in self.counts.items()
        }


def _scalar(value: Any, *, name: str | None = None) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"metric tensor must be scalar, got {tuple(value.shape)}")
        value = value.detach().item()
    elif isinstance(value, np.generic):
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        label = "metric" if name is None else f"metric {name!r}"
        raise FloatingPointError(f"{label} is not finite: {result}")
    return result


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a strict JSON metric file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parse_args() -> tuple[argparse.Namespace, Any]:
    # Importing AppLauncher is intentionally delayed until argument parsing;
    # task and simulator modules are imported only after the app is running.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--steps", type=int, default=1_000, help="Vector-environment interaction steps.")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--buffer", type=int, default=1_000_000, help="Replay capacity in transitions.")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Replay transitions required before updates (default: min(10000, buffer), but at least batch).",
    )
    parser.add_argument("--updates", type=float, default=2.0, help="Gradient updates per vector interaction.")
    parser.add_argument(
        "--critic_burnin_updates",
        type=int,
        default=0,
        help="Initial local updates that train critic/target only and preserve the loaded BC actor.",
    )
    parser.add_argument(
        "--actor_update_period",
        type=int,
        default=2,
        help=(
            "Run one actor/temperature update every N critic updates. The upstream "
            "default is 2; larger values permit conservative fine-tuning of a proven actor."
        ),
    )
    parser.add_argument(
        "--actor_lr_scale",
        type=float,
        default=None,
        help=(
            "Absolute actor-only multiplier on the FlashSAC LR schedule. Fresh runs "
            "default to 1; routed resumes retain the checkpoint value unless explicitly "
            "overridden. Critic and temperature learning rates are unchanged."
        ),
    )
    parser.add_argument(
        "--lr_decay_updates",
        type=int,
        default=None,
        help="Absolute global scheduler decay budget; use the same value across resumed curriculum stages.",
    )
    parser.add_argument(
        "--lr_warmup_updates",
        type=int,
        default=None,
        help="Absolute global scheduler warm-up budget (default: 5%% of decay budget).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_step", type=int, default=3, help="Replay return horizon for non-smoke runs.")
    parser.add_argument("--actor_blocks", type=int, default=2)
    parser.add_argument("--actor_hidden", type=int, default=128)
    parser.add_argument("--critic_blocks", type=int, default=2)
    parser.add_argument("--critic_hidden", type=int, default=256)
    parser.add_argument("--critic_bins", type=int, default=101)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--unlatched_arm_noise_scale",
        type=float,
        default=1.0,
        help="Additional exploration multiplier for arm actions before observed latch.",
    )
    parser.add_argument(
        "--unlatched_hand_noise_scale",
        type=float,
        default=1.0,
        help="Additional exploration multiplier for token/residual actions before observed latch.",
    )
    parser.add_argument(
        "--latched_arm_noise_scale",
        type=float,
        default=1.0,
        help="Additional exploration multiplier for arm actions when observed latch[106] is active.",
    )
    parser.add_argument(
        "--latched_hand_noise_scale",
        type=float,
        default=0.2,
        help="Additional exploration multiplier for token/residual actions after observed latch.",
    )
    parser.add_argument(
        "--public_latch_arm_gate",
        action="store_true",
        help=(
            "Freeze only arm7 at exact zero while public grasp latch observation[106] "
            "is zero; hand14 remains active and arm authority begins when it becomes one."
        ),
    )
    parser.add_argument(
        "--public_latch_frozen_lift_router",
        action="store_true",
        help=(
            "Train only hand14 while latch[106]=0, then route latch[106]=1 to a "
            "self-contained frozen deterministic full21 lift actor."
        ),
    )
    parser.add_argument(
        "--frozen_lift_actor_checkpoint",
        type=Path,
        default=None,
        help=(
            "Fresh-run source checkpoint for the immutable lift actor. Routed resumes "
            "restore frozen_lift_actor.pt from --checkpoint instead."
        ),
    )
    parser.add_argument(
        "--search_handoff_checkpoint",
        type=Path,
        default=None,
        help=(
            "Run this frozen rl_games SEARCH actor from each ordinary reset and switch "
            "sticky to the routed FlashSAC option at the public-observation handoff. "
            "Only option-controlled transitions enter replay."
        ),
    )
    parser.add_argument(
        "--search_handoff_min_score",
        type=float,
        default=0.30,
        help=(
            "Training-curriculum threshold for min(thumb proximity, second-best "
            "non-thumb proximity). Stage 1 defaults to the recoverable high-readiness tail."
        ),
    )
    parser.add_argument(
        "--search_handoff_hold_steps",
        type=int,
        default=4,
        help="Consecutive public-readiness frames required before FlashSAC takes control.",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="Optional task horizon override; smoke defaults to 0.12 s to exercise auto-reset.",
    )
    parser.add_argument(
        "--randomize_episode_lengths",
        action="store_true",
        help="Decorrelate timeout steps. Disabled by default because full lifts often need >600 steps.",
    )
    parser.add_argument(
        "--validate_finite",
        action="store_true",
        help="Synchronously check every observation/action/reward for NaN/Inf (always on in smoke).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional FlashSAC checkpoint to load.")
    parser.add_argument(
        "--actor_checkpoint",
        type=Path,
        default=None,
        help=(
            "Load only actor.pt from a FlashSAC checkpoint. Critic, target, temperature, "
            "optimizers, replay, exploration, and reward normalization remain fresh."
        ),
    )
    parser.add_argument(
        "--allow_cross_task_actor",
        action="store_true",
        help=(
            "Explicitly allow a full-task 115D/21D actor to initialize the coupled-power "
            "131D/21D actor through the audited zero-padded observation projection."
        ),
    )
    parser.add_argument(
        "--resume_replay",
        action="store_true",
        help="Load replay_buffer.pt from --checkpoint; fails if it is absent.",
    )
    parser.add_argument(
        "--save_replay",
        action="store_true",
        help="Save online/permanent-demo replay beside the final network checkpoint.",
    )
    parser.add_argument(
        "--demo",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Successful one-step trajectory datasets. They are converted with the configured "
            "n-step/gamma and kept in a permanent replay reservoir."
        ),
    )
    parser.add_argument(
        "--demo_fraction",
        type=float,
        default=0.25,
        help="Exact fraction of every update batch drawn from permanent demonstrations.",
    )
    parser.add_argument(
        "--actor_demo",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Successful observation/action-only teacher datasets used exclusively for actor "
            "rehearsal; they are never inserted into critic replay. Coupled-power sources "
            "must satisfy the strict 131D phase contract and require --actor_checkpoint "
            "initially (or --checkpoint plus --resume_actor_demo when resuming)."
        ),
    )
    parser.add_argument(
        "--resume_actor_demo",
        action="store_true",
        help="Restore actor_rehearsal.pt sampler state from --checkpoint.",
    )
    parser.add_argument(
        "--demo_bc_weight",
        type=float,
        default=None,
        help="Demo-only actor rehearsal weight (default: 1 with any actor demo source).",
    )
    parser.add_argument("--demo_bc_target_std", type=float, default=0.15)
    parser.add_argument("--demo_bc_std_weight", type=float, default=0.05)
    parser.add_argument("--demo_bc_arm_weight", type=float, default=1.0)
    parser.add_argument("--demo_bc_token_weight", type=float, default=1.0)
    parser.add_argument("--demo_bc_residual_weight", type=float, default=1.0)
    parser.add_argument(
        "--demo_bc_batch",
        type=int,
        default=None,
        help="Demo-only rehearsal rows (default: the fixed demo rows per mixed batch).",
    )
    parser.add_argument(
        "--demo_bc_phases",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional demo phases used only for actor rehearsal. Critic replay keeps its "
            "phase-balanced demo mix; e.g. '--demo_bc_phases 3' anchors a lift curriculum."
        ),
    )
    parser.add_argument(
        "--curriculum_dataset",
        type=Path,
        default=None,
        help="Optional physically captured reset-boundary dataset.",
    )
    parser.add_argument("--curriculum_boundary", default="close_start")
    parser.add_argument("--curriculum_probability", type=float, default=0.0)
    parser.add_argument("--curriculum_joint_noise", type=float, default=0.0)
    task_mode_group = parser.add_mutually_exclusive_group()
    task_mode_group.add_argument(
        "--close_option_mode",
        action="store_true",
        help="Train the legacy 21D close-only task contract.",
    )
    task_mode_group.add_argument(
        "--power_close_option_mode",
        action="store_true",
        help=(
            "Train the power-grasp close option with a 14D hand-only policy; the physical "
            "environment still receives an exact zero arm7 prefix."
        ),
    )
    task_mode_group.add_argument(
        "--coupled_power_align_close_option_mode",
        action="store_true",
        help=(
            "Train the 21D coupled wrist-align/power-close option; both policy and physical "
            "environment use the identity arm7+hand14 action boundary."
        ),
    )
    task_mode_group.add_argument(
        "--coupled_teacher_residual_mode",
        action="store_true",
        help=(
            "Train a 14D canonical hand residual around the audited coupled CEM teacher; "
            "the exact public-observation wrist prior owns ALIGN."
        ),
    )
    parser.add_argument(
        "--coupled_teacher_prior",
        type=Path,
        default=None,
        help="Strict coupled CEM artifact required by --coupled_teacher_residual_mode.",
    )
    parser.add_argument("--residual_close_token_scale", type=float, default=0.04)
    parser.add_argument("--residual_close_distal_scale", type=float, default=0.06)
    parser.add_argument("--residual_hold_token_scale", type=float, default=0.015)
    parser.add_argument("--residual_hold_distal_scale", type=float, default=0.025)
    parser.add_argument(
        "--actor_unlock_successes",
        type=int,
        default=1,
        help=(
            "For teacher-residual training, keep the actor deterministic at zero until this "
            "many native strict close successes have entered online replay."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("logs/flashsac/pick_tool"))
    parser.add_argument("--metrics_every", type=int, default=100)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use a tiny 8-env integration run: 8 interactions normally, or 64 for the "
            "coupled align-close mode so both controller phases are exercised."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    return args, launcher


def _validate_args(args: argparse.Namespace) -> None:
    task_mode = task_mode_from_close_option(
        args.close_option_mode,
        args.power_close_option_mode,
        args.coupled_power_align_close_option_mode,
        args.coupled_teacher_residual_mode,
    )
    validate_training_source_selection(
        checkpoint=args.checkpoint,
        actor_checkpoint=args.actor_checkpoint,
        resume_replay=args.resume_replay,
        resume_actor_demo=args.resume_actor_demo,
        close_option_mode=args.close_option_mode,
        power_close_option_mode=args.power_close_option_mode,
        demo=args.demo,
        actor_demo=args.actor_demo,
        coupled_power_align_close_option_mode=(
            args.coupled_power_align_close_option_mode
        ),
        coupled_teacher_residual_mode=args.coupled_teacher_residual_mode,
        allow_cross_task_actor=args.allow_cross_task_actor,
        public_latch_arm_gate=args.public_latch_arm_gate,
        public_latch_frozen_lift_router=(
            args.public_latch_frozen_lift_router
        ),
        frozen_lift_actor_checkpoint=args.frozen_lift_actor_checkpoint,
    )
    validate_close_option_training_config(
        close_option_mode=args.close_option_mode,
        power_close_option_mode=args.power_close_option_mode,
        curriculum_dataset=args.curriculum_dataset,
        curriculum_boundary=args.curriculum_boundary,
        curriculum_probability=args.curriculum_probability,
        curriculum_joint_noise=args.curriculum_joint_noise,
        episode_length_s=args.episode_length_s,
        randomize_episode_lengths=args.randomize_episode_lengths,
        coupled_power_align_close_option_mode=(
            args.coupled_power_align_close_option_mode
        ),
        coupled_teacher_residual_mode=args.coupled_teacher_residual_mode,
    )
    validate_public_latch_arm_gate_config(
        enabled=args.public_latch_arm_gate,
        task_mode=task_mode,
        curriculum_dataset=args.curriculum_dataset,
        curriculum_boundary=args.curriculum_boundary,
        curriculum_probability=args.curriculum_probability,
        curriculum_joint_noise=args.curriculum_joint_noise,
        episode_length_s=args.episode_length_s,
        randomize_episode_lengths=args.randomize_episode_lengths,
        online_search_handoff=args.search_handoff_checkpoint is not None,
    )
    validate_online_search_handoff_training_config(
        search_checkpoint=args.search_handoff_checkpoint,
        min_score=args.search_handoff_min_score,
        hold_steps=args.search_handoff_hold_steps,
        task_mode=task_mode,
        public_latch_arm_gate=args.public_latch_arm_gate,
        public_latch_frozen_lift_router=args.public_latch_frozen_lift_router,
        curriculum_dataset=args.curriculum_dataset,
        curriculum_probability=args.curriculum_probability,
        curriculum_joint_noise=args.curriculum_joint_noise,
        episode_length_s=args.episode_length_s,
        randomize_episode_lengths=args.randomize_episode_lengths,
    )
    for name in (
        "steps",
        "num_envs",
        "buffer",
        "batch",
        "metrics_every",
        "n_step",
        "actor_hidden",
        "critic_hidden",
        "critic_bins",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    for name in ("actor_blocks", "critic_blocks"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if args.buffer < args.num_envs:
        raise ValueError("--buffer must hold at least one full vector transition")
    if args.batch > args.buffer:
        raise ValueError("--batch cannot exceed --buffer")
    if args.warmup is not None:
        if args.warmup < args.batch:
            raise ValueError("--warmup cannot be smaller than --batch")
        if args.warmup > args.buffer:
            raise ValueError("--warmup cannot exceed --buffer")
    if not math.isfinite(args.updates) or args.updates < 0.0:
        raise ValueError("--updates must be finite and non-negative")
    if args.critic_burnin_updates < 0:
        raise ValueError("--critic_burnin_updates must be non-negative")
    if args.actor_update_period < 1:
        raise ValueError("--actor_update_period must be positive")
    if args.actor_lr_scale is not None and (
        not math.isfinite(args.actor_lr_scale) or args.actor_lr_scale <= 0.0
    ):
        raise ValueError("--actor_lr_scale must be finite and positive")
    for name in ("lr_decay_updates", "lr_warmup_updates"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name} must be positive")
    if args.episode_length_s is not None and (
        not math.isfinite(args.episode_length_s) or args.episode_length_s <= 0.0
    ):
        raise ValueError("--episode_length_s must be finite and positive")
    if not math.isfinite(args.curriculum_probability) or not 0.0 <= args.curriculum_probability <= 1.0:
        raise ValueError("--curriculum_probability must be in [0, 1]")
    if not math.isfinite(args.curriculum_joint_noise) or args.curriculum_joint_noise < 0.0:
        raise ValueError("--curriculum_joint_noise must be finite and non-negative")
    if args.curriculum_probability > 0.0 and args.curriculum_dataset is None:
        raise ValueError("--curriculum_probability > 0 requires --curriculum_dataset")
    if args.curriculum_dataset is not None and not args.curriculum_dataset.is_file():
        raise FileNotFoundError(args.curriculum_dataset)
    if args.demo is not None:
        missing_demos = [path for path in args.demo if not path.is_file()]
        if missing_demos:
            raise FileNotFoundError(f"demonstration datasets do not exist: {missing_demos}")
        if not math.isfinite(args.demo_fraction) or not 0.0 < args.demo_fraction < 1.0:
            raise ValueError("--demo_fraction must be finite and strictly between 0 and 1")
        demo_rows = args.batch * args.demo_fraction
        if not math.isclose(demo_rows, round(demo_rows), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("--batch * --demo_fraction must be an integer")
        if not 0 < round(demo_rows) < args.batch:
            raise ValueError("a mixed batch must contain both online and demonstration rows")
    if args.actor_demo is not None:
        missing_actor_demos = [path for path in args.actor_demo if not path.is_file()]
        if missing_actor_demos:
            raise FileNotFoundError(
                f"actor-only demonstration datasets do not exist: {missing_actor_demos}"
            )
    has_actor_supervision = args.demo is not None or args.actor_demo is not None
    if has_actor_supervision:
        if args.demo_bc_weight is not None and (
            not math.isfinite(args.demo_bc_weight) or args.demo_bc_weight < 0.0
        ):
            raise ValueError("--demo_bc_weight must be finite and non-negative")
        if not math.isfinite(args.demo_bc_target_std) or args.demo_bc_target_std <= 0.0:
            raise ValueError("--demo_bc_target_std must be finite and positive")
        if not math.isfinite(args.demo_bc_std_weight) or args.demo_bc_std_weight < 0.0:
            raise ValueError("--demo_bc_std_weight must be finite and non-negative")
        group_weights = (
            args.demo_bc_arm_weight,
            args.demo_bc_token_weight,
            args.demo_bc_residual_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in group_weights):
            raise ValueError("demo BC action-group weights must be finite and non-negative")
        if not any(value > 0.0 for value in group_weights):
            raise ValueError("at least one demo BC action-group weight must be positive")
        if args.demo_bc_batch is not None and args.demo_bc_batch < 1:
            raise ValueError("--demo_bc_batch must be positive")
        if args.demo_bc_phases is not None:
            if len(set(args.demo_bc_phases)) != len(args.demo_bc_phases):
                raise ValueError("--demo_bc_phases must not contain duplicates")
    else:
        if args.demo_bc_weight not in (None, 0.0):
            raise ValueError("--demo_bc_weight must be 0 without an actor demo source")
        if args.demo_bc_phases is not None:
            raise ValueError("--demo_bc_phases requires --demo or --actor_demo")
        if args.resume_actor_demo:
            raise ValueError("--resume_actor_demo requires --demo or --actor_demo")
    for name in (
        "unlatched_arm_noise_scale",
        "unlatched_hand_noise_scale",
        "latched_arm_noise_scale",
        "latched_hand_noise_scale",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name} must be finite and non-negative")
    if task_mode in (
        POWER_CLOSE_OPTION_TASK_MODE,
        COUPLED_TEACHER_RESIDUAL_TASK_MODE,
    ):
        if args.unlatched_arm_noise_scale != 0.0 or args.latched_arm_noise_scale != 0.0:
            raise ValueError(
                "the selected 14D policy has no arm actions; both arm noise "
                "scales must be exactly 0"
            )
        if args.demo_bc_arm_weight != 0.0:
            raise ValueError(
                "the selected 14D policy has no arm actions; "
                "--demo_bc_arm_weight must be 0"
            )
    residual_mode = task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
    if residual_mode:
        if args.coupled_teacher_prior is None or not args.coupled_teacher_prior.is_file():
            raise FileNotFoundError(
                args.coupled_teacher_prior
                if args.coupled_teacher_prior is not None
                else "--coupled_teacher_prior"
            )
        if args.curriculum_joint_noise != 0.0:
            raise ValueError("teacher-residual v1 requires exact-zero curriculum joint noise")
        if args.actor_unlock_successes < 1:
            raise ValueError("--actor_unlock_successes must be positive")
        for name in (
            "residual_close_token_scale",
            "residual_close_distal_scale",
            "residual_hold_token_scale",
            "residual_hold_distal_scale",
        ):
            value = getattr(args, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name} must be finite and in [0, 1]")
    elif args.coupled_teacher_prior is not None:
        raise ValueError(
            "--coupled_teacher_prior requires --coupled_teacher_residual_mode"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def resolve_warmup_transitions(
    *, buffer: int, batch: int, smoke: bool, requested: int | None
) -> int:
    """Resolve the replay warm-up without hiding short-run no-update tests."""

    if smoke:
        # A smoke run is specifically required to exercise optimizer updates.
        return batch
    if requested is not None:
        return requested
    return min(buffer, max(batch, min(10_000, buffer)))


_POWER_CLOSE_INAPPLICABLE_LEGACY_METRICS = frozenset(
    {
        # These task log fields are authored by the legacy thumb+two/full-task
        # state.  Reporting them during power-close training would silently mix
        # two different grasp definitions.
        "success_frac",
        "is_grasped_phase_frac",
        "grasp_quality_mean",
    }
)


def _strict_metrics(
    info: Mapping[str, Any], *, task_mode: str = FULL_TASK_MODE
) -> dict[str, float]:
    """Return scalar training telemetry with mode-specific grasp authority.

    Power-close metrics are reduced from the per-environment reset-before
    ``pick_tool_terminal`` tensors.  This preserves rare exploration hits that
    an aggregate mean alone would hide and, importantly, never substitutes the
    legacy thumb+two latch or quality.
    """

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    values = info.get("strict_metrics", {})
    if not isinstance(values, Mapping):
        raise TypeError("adapter info['strict_metrics'] must be a mapping")
    metrics = {
        str(name): _scalar(value, name=str(name))
        for name, value in values.items()
        if not (
            task_mode in (
                POWER_CLOSE_OPTION_TASK_MODE,
                COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
                COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            )
            and str(name) in _POWER_CLOSE_INAPPLICABLE_LEGACY_METRICS
        )
    }
    if task_mode not in (
        POWER_CLOSE_OPTION_TASK_MODE,
        COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
        COUPLED_TEACHER_RESIDUAL_TASK_MODE,
    ):
        return metrics

    terminal = info.get("pick_tool_terminal")
    if not isinstance(terminal, Mapping):
        raise KeyError("power-close adapter info has no pick_tool_terminal ground truth")
    specifications = {
        "power_close_quality": torch.float32,
        "power_wrap_quality": torch.float32,
        "power_grasp_quality": torch.float32,
        "power_legal_other_contact_count": torch.long,
        "power_thumb_contact": torch.bool,
        "power_is_grasped": torch.bool,
        "power_grasp_latch_confirm_steps": torch.long,
        "power_close_option_stable_steps": torch.long,
    }
    if is_coupled_task_mode(task_mode):
        specifications.update(
            {
                "coupled_power_pose_escape": torch.bool,
                "coupled_power_rotation_drift": torch.float32,
                "coupled_power_xy_drift": torch.float32,
                "coupled_power_true_clearance": torch.float32,
                "coupled_power_arm_target_offset_abs_max": torch.float32,
                "coupled_power_arm_target_saturated": torch.bool,
                "coupled_power_align_active": torch.bool,
            }
        )
    vectors: dict[str, torch.Tensor] = {}
    vector_shape: tuple[int, ...] | None = None
    vector_device: torch.device | None = None
    for name, dtype in specifications.items():
        value = terminal.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"pick_tool_terminal[{name!r}] must be a torch.Tensor")
        if value.ndim != 1 or value.numel() < 1 or value.dtype != dtype:
            raise ValueError(
                f"pick_tool_terminal[{name!r}] must be non-empty {dtype} vector, "
                f"got {value.dtype}{tuple(value.shape)}"
            )
        if vector_shape is None:
            vector_shape = tuple(value.shape)
            vector_device = value.device
        elif tuple(value.shape) != vector_shape or value.device != vector_device:
            raise ValueError(
                "power-close terminal telemetry tensors must share shape and device"
            )
        vectors[name] = value

    for name in (
        "power_close_quality",
        "power_wrap_quality",
        "power_grasp_quality",
    ):
        value = vectors[name]
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < 0.0) | (value > 1.0)).any()
        ):
            raise ValueError(f"pick_tool_terminal[{name!r}] must be finite and in [0, 1]")
    legal_other = vectors["power_legal_other_contact_count"]
    latch_confirm_steps = vectors["power_grasp_latch_confirm_steps"]
    stable_steps = vectors["power_close_option_stable_steps"]
    if bool(((legal_other < 0) | (legal_other > 4)).any()):
        raise ValueError("power legal non-thumb contact count must be in [0, 4]")
    if bool((latch_confirm_steps < 0).any()) or bool((stable_steps < 0).any()):
        raise ValueError("power latch/stable step counters must be non-negative")

    thumb = vectors["power_thumb_contact"]
    other_ge3 = legal_other >= 3
    power_latch = vectors["power_is_grasped"]
    float_vectors = {
        "power_q_close": vectors["power_close_quality"],
        "power_q_wrap": vectors["power_wrap_quality"],
        "power_grasp_quality": vectors["power_grasp_quality"],
        "power_legal_other_contacts": legal_other.float(),
        "power_latch_confirm_steps": latch_confirm_steps.float(),
        "power_close_option_stable_steps": stable_steps.float(),
    }
    for name, value in float_vectors.items():
        metrics[f"{name}_mean"] = float(value.mean().item())
        metrics[f"{name}_max"] = float(value.max().item())
    metrics.update(
        {
            "power_thumb_contact_frac": float(thumb.float().mean().item()),
            "power_other_ge3_frac": float(other_ge3.float().mean().item()),
            "power_thumb_plus_three_frac": float(
                (thumb & other_ge3).float().mean().item()
            ),
            "power_is_grasped_frac": float(power_latch.float().mean().item()),
            # Explicit phase alias for readers comparing old is_grasped_phase
            # charts; both values come from the power latch above.
            "power_grasp_phase_frac": float(power_latch.float().mean().item()),
        }
    )
    if is_coupled_task_mode(task_mode):
        nonnegative_names = (
            "coupled_power_rotation_drift",
            "coupled_power_xy_drift",
            "coupled_power_arm_target_offset_abs_max",
        )
        finite_names = (*nonnegative_names, "coupled_power_true_clearance")
        for name in finite_names:
            value = vectors[name]
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"pick_tool_terminal[{name!r}] must be finite")
        for name in nonnegative_names:
            if bool((vectors[name] < 0.0).any()):
                raise ValueError(f"pick_tool_terminal[{name!r}] must be non-negative")
        for name in finite_names:
            value = vectors[name]
            metrics[f"{name}_mean"] = float(value.mean().item())
            metrics[f"{name}_max"] = float(value.max().item())
        metrics["coupled_power_true_clearance_min"] = float(
            vectors["coupled_power_true_clearance"].min().item()
        )
        for name in (
            "coupled_power_pose_escape",
            "coupled_power_arm_target_saturated",
            "coupled_power_align_active",
        ):
            metrics[f"{name}_frac"] = float(vectors[name].float().mean().item())
    return metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_actor_checkpoint_source(
    actor_checkpoint: Path,
    *,
    output_checkpoint: Path,
    target_task_mode: str | None = None,
    allow_cross_task_actor: bool = False,
) -> dict[str, Any]:
    """Resolve and fingerprint an actor-only initialization source."""

    source = actor_checkpoint.resolve()
    destination = output_checkpoint.resolve()
    if source == destination:
        raise ValueError(
            "--actor_checkpoint cannot be the current output checkpoint_final: "
            f"{source}"
        )
    if not source.is_dir():
        raise FileNotFoundError(f"actor checkpoint directory does not exist: {source}")
    actor_path = source / "actor.pt"
    if not actor_path.is_file():
        raise FileNotFoundError(f"missing FlashSAC actor checkpoint: {actor_path}")
    source_contract = read_checkpoint_task_contract(source)
    if source_contract.get("policy_router") is not None:
        raise ValueError(
            "a routed checkpoint actor.pt cannot be used alone because that would "
            "discard its frozen lift branch"
        )
    target_task_mode = (
        str(source_contract["task_mode"])
        if target_task_mode is None
        else target_task_mode
    )
    target_runtime = runtime_contract(target_task_mode)
    source_dim = int(source_contract["policy_action_dim"])
    target_dim = int(target_runtime["policy_action_dim"])
    actor_projection: str | None = None
    source_task_mode = str(source_contract["task_mode"])
    if (
        COUPLED_TEACHER_RESIDUAL_TASK_MODE
        in (source_task_mode, target_task_mode)
        and source_task_mode != target_task_mode
    ):
        raise ValueError(
            "teacher-residual actors cannot cross task modes: their 14D outputs are "
            "canonical residuals, not absolute hand actions"
        )
    if (
        target_task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE
        and source_task_mode != target_task_mode
    ):
        if not allow_cross_task_actor:
            raise ValueError(
                "cross-task --actor_checkpoint initialization into coupled power is "
                "disabled; pass --allow_cross_task_actor to authorize the audited "
                "full115-to-coupled131 actor projection"
            )
        required_source = {
            "task_mode": FULL_TASK_MODE,
            "policy_action_dim": PICK_TOOL_ACTION_DIM,
            "policy_action_layout": FULL_POLICY_ACTION_LAYOUT,
            "action_projection": IDENTITY_ACTION_PROJECTION,
            "observation_dim": PICK_TOOL_OBSERVATION_DIM,
            "observation_contract": PICK_TOOL_OBSERVATION_CONTRACT,
        }
        mismatches = {
            key: (source_contract.get(key), expected)
            for key, expected in required_source.items()
            if source_contract.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "--allow_cross_task_actor only permits an exact full-task 115D/21D "
                f"identity actor source for coupled power; mismatches={mismatches}"
            )
        if target_dim != PICK_TOOL_ACTION_DIM:
            raise ValueError("coupled-power actor target must use the full 21D action")
        actor_projection = FULL115_TO_COUPLED131_ACTOR_PROJECTION
    elif (
        source_task_mode == COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE
        and target_task_mode != source_task_mode
    ):
        raise ValueError(
            "a coupled-power 131D actor cannot initialize a different 115D task mode"
        )
    elif source_dim != target_dim:
        if source_dim == PICK_TOOL_ACTION_DIM and target_dim == PICK_TOOL_HAND_ACTION_DIM:
            actor_projection = FULL21_TO_HAND14_ACTOR_PROJECTION
        else:
            raise ValueError(
                "--actor_checkpoint has no allowlisted actor projection: "
                f"source policy_action_dim={source_dim}, target={target_dim}"
            )
    return {
        "path": str(source),
        "actor_sha256": _sha256(actor_path),
        "source_task_mode": source_task_mode,
        "source_policy_action_dim": source_dim,
        "source_policy_action_layout": str(source_contract["policy_action_layout"]),
        "source_policy_action_authority": source_contract[
            "policy_action_authority"
        ],
        "source_policy_router": source_contract["policy_router"],
        "target_task_mode": target_task_mode,
        "target_policy_action_dim": target_dim,
        "actor_projection": actor_projection,
    }


def load_audited_actor_checkpoint(agent: Any, audit: Mapping[str, Any]) -> None:
    """Load an audited actor, applying only the explicit full21 -> hand14 slice."""

    projection = audit.get("actor_projection")
    source_dim = audit.get("source_policy_action_dim")
    target_dim = audit.get("target_policy_action_dim")
    if projection == FULL21_TO_HAND14_ACTOR_PROJECTION:
        if source_dim != PICK_TOOL_ACTION_DIM or target_dim != PICK_TOOL_HAND_ACTION_DIM:
            raise ValueError("invalid full21-to-hand14 actor projection audit")
        agent.load_actor(
            audit["path"],
            source_action_indices=range(
                PICK_TOOL_ARM_ACTION_DIM,
                PICK_TOOL_ACTION_DIM,
            ),
            expected_source_action_dim=PICK_TOOL_ACTION_DIM,
        )
        return
    if projection == FULL115_TO_COUPLED131_ACTOR_PROJECTION:
        if source_dim != PICK_TOOL_ACTION_DIM or target_dim != PICK_TOOL_ACTION_DIM:
            raise ValueError("invalid full115-to-coupled131 actor projection audit")
        _load_actor_with_zero_padded_observation_input(
            agent,
            Path(str(audit["path"])) / "actor.pt",
        )
        return
    if projection is not None:
        raise ValueError(f"unsupported actor projection={projection!r}")
    if source_dim != target_dim:
        raise ValueError(
            "identity actor load requires equal policy action dimensions: "
            f"source={source_dim!r}, target={target_dim!r}"
        )
    agent.load_actor(audit["path"])


def _canonical_actor_state(
    state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> dict[str, tuple[str, torch.Tensor]]:
    """Map compiled/uncompiled actor keys to one unambiguous canonical namespace."""

    prefix = "_orig_mod."
    canonical: dict[str, tuple[str, torch.Tensor]] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} actor state must map string keys to tensors")
        canonical_key = key.removeprefix(prefix)
        if canonical_key in canonical:
            raise ValueError(f"{label} actor state has duplicate key {canonical_key!r}")
        canonical[canonical_key] = (key, value)
    if not canonical:
        raise ValueError(f"{label} actor state is empty")
    return canonical


def project_full_actor_to_coupled_observation_state(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Copy a 115D actor into a 131D actor while initially ignoring new inputs."""

    source = _canonical_actor_state(source_state, label="source")
    target = _canonical_actor_state(target_state, label="target")
    if set(source) != set(target):
        raise ValueError("full115-to-coupled131 actor keys are structurally incompatible")

    expanded: dict[str, torch.Tensor] = {}
    resized: set[str] = set()
    input_vector_keys = {
        "embedder.norm.weight",
        "embedder.norm.bias",
        "embedder.norm.running_mean",
        "embedder.norm.running_var",
    }
    for canonical_key, (target_key, target_value) in target.items():
        source_value = source[canonical_key][1]
        if source_value.dtype != target_value.dtype:
            raise TypeError(
                f"actor tensor {canonical_key!r} dtype mismatch: "
                f"source={source_value.dtype}, target={target_value.dtype}"
            )
        if source_value.shape == target_value.shape:
            expanded[target_key] = source_value.to(device=target_value.device).clone()
            continue
        if canonical_key == "embedder.w.w.weight":
            expected_source_shape = (
                target_value.shape[0],
                PICK_TOOL_OBSERVATION_DIM,
            )
            expected_target_shape = (
                target_value.shape[0],
                COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
            )
            if (
                tuple(source_value.shape) != expected_source_shape
                or tuple(target_value.shape) != expected_target_shape
            ):
                raise ValueError(
                    f"actor input matrix has incompatible shapes: "
                    f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
                )
            value = torch.zeros_like(target_value)
            value[:, :PICK_TOOL_OBSERVATION_DIM].copy_(
                source_value.to(device=target_value.device)
            )
        elif canonical_key in input_vector_keys:
            if (
                tuple(source_value.shape) != (PICK_TOOL_OBSERVATION_DIM,)
                or tuple(target_value.shape)
                != (COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,)
            ):
                raise ValueError(
                    f"actor input-normalizer tensor {canonical_key!r} has incompatible shapes"
                )
            value = target_value.clone()
            value[:PICK_TOOL_OBSERVATION_DIM].copy_(
                source_value.to(device=target_value.device)
            )
        else:
            raise ValueError(
                f"actor tensor {canonical_key!r} changes outside the allowlisted input "
                f"expansion: source={tuple(source_value.shape)}, "
                f"target={tuple(target_value.shape)}"
            )
        expanded[target_key] = value
        resized.add(canonical_key)

    expected_resized = input_vector_keys | {"embedder.w.w.weight"}
    if resized != expected_resized:
        raise ValueError(
            "full115-to-coupled131 actor projection did not resize the exact input layer; "
            f"resized={sorted(resized)}"
        )
    return expanded


def _load_actor_with_zero_padded_observation_input(agent: Any, actor_path: Path) -> None:
    """Load only actor tensors through the audited 115D -> 131D input expansion."""

    payload = torch.load(actor_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"actor checkpoint {actor_path} must contain a mapping")
    source_state = payload.get("network_state_dict")
    if not isinstance(source_state, Mapping):
        raise TypeError(f"actor checkpoint {actor_path} has no mapping network_state_dict")
    actor_bundle = getattr(agent, "_actor", None)
    network = getattr(actor_bundle, "network", None)
    if network is None:
        raise TypeError("FlashSAC agent has no actor network for observation projection")
    target_state = network.state_dict()
    projected = project_full_actor_to_coupled_observation_state(
        source_state,
        target_state,
    )
    network.load_state_dict(projected, strict=True)


def audit_pick_tool_demonstrations(path: Path) -> dict[str, Any]:
    """Reject demo files that do not prove strict, non-hacked task success."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: demonstration payload must be a mapping")
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise TypeError(f"{path}: missing collector metadata")
    required_meta = {
        "format_version": 1,
        "transition_horizon": 1,
        "terminal_observation": "adapter_captured_pre_reset",
        "normal_task_termination": True,
        "collector": "base_close_lift_hierarchy_strict_success",
        "reject_unlatched_clearance_ge_5cm": True,
        "observation_dim": 115,
        "action_dim": 21,
    }
    for key, expected in required_meta.items():
        if meta.get(key) != expected:
            raise ValueError(f"{path}: demo metadata {key}={meta.get(key)!r}, expected {expected!r}")
    offsets = payload.get("episode_offsets")
    observation = payload.get("observation")
    action = payload.get("action")
    if not isinstance(offsets, torch.Tensor) or offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError(f"{path}: invalid episode_offsets")
    if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
        raise ValueError(f"{path}: invalid observation tensor")
    if not isinstance(action, torch.Tensor) or action.shape != (observation.shape[0], 21):
        raise ValueError(f"{path}: invalid action tensor")
    rows = int(observation.shape[0])
    offsets = offsets.to(dtype=torch.long)
    if int(offsets[0]) != 0 or int(offsets[-1]) != rows or bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError(f"{path}: episode offsets do not partition the transition rows")
    episodes = int(offsets.numel() - 1)
    final_rows = offsets[1:] - 1
    expected_terminal = torch.zeros(rows, dtype=torch.bool)
    expected_terminal[final_rows] = True
    terminated = payload.get("terminated")
    truncated = payload.get("truncated")
    if not isinstance(terminated, torch.Tensor) or not torch.equal(terminated.bool(), expected_terminal):
        raise ValueError(f"{path}: every demonstration episode must end in exactly one termination")
    if not isinstance(truncated, torch.Tensor) or bool(truncated.bool().any()):
        raise ValueError(f"{path}: strict successful demonstrations cannot be truncated")

    required_episode_fields: dict[str, tuple[torch.dtype | None, Any]] = {
        "episode_success": (torch.bool, lambda value: bool(value.all())),
        "episode_terminal_is_grasped": (torch.bool, lambda value: bool(value.all())),
        "episode_terminal_true_clearance": (None, lambda value: bool((value >= 0.20).all())),
        "episode_terminal_grasp_quality": (None, lambda value: bool((value >= 0.35).all())),
        "episode_terminal_hold_quality": (None, lambda value: bool((value >= 0.50).all())),
        "episode_terminal_max_force": (None, lambda value: bool((value <= 30.0).all())),
        "episode_terminal_object_lin_speed": (None, lambda value: bool((value < 0.20).all())),
        "episode_terminal_object_ang_speed": (None, lambda value: bool((value < 3.0).all())),
        "episode_terminal_success_steps": (None, lambda value: bool((value >= 15).all())),
        "episode_route": (None, lambda value: bool((value == 1).all())),
    }
    for key, (dtype, predicate) in required_episode_fields.items():
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (episodes,):
            raise ValueError(f"{path}: {key} must have shape ({episodes},)")
        if dtype is not None and value.dtype != dtype:
            raise TypeError(f"{path}: {key} must use {dtype}")
        if not predicate(value):
            raise ValueError(f"{path}: {key} violates the strict-success demo contract")
    if not bool(torch.isfinite(observation).all()) or not bool(torch.isfinite(action).all()):
        raise ValueError(f"{path}: observation/action contains NaN or infinity")
    return {
        "episodes": episodes,
        "terminal_clearance_min": float(payload["episode_terminal_true_clearance"].min()),
        "terminal_hold_quality_min": float(payload["episode_terminal_hold_quality"].min()),
        "terminal_max_force_max": float(payload["episode_terminal_max_force"].max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Create the environment and execute the minimal collection/update loop."""

    task_mode = task_mode_from_close_option(
        args.close_option_mode,
        args.power_close_option_mode,
        args.coupled_power_align_close_option_mode,
        args.coupled_teacher_residual_mode,
    )
    current_policy_action_authority = policy_action_authority_contract(
        args.public_latch_arm_gate
    )
    policy_router_enabled = bool(args.public_latch_frozen_lift_router)
    env_task_mode_overrides = environment_task_mode_overrides(task_mode)
    physical_close_option_mode = bool(env_task_mode_overrides["close_option_mode"])
    if args.smoke:
        args.steps = resolve_smoke_interaction_steps(
            requested=args.steps,
            task_mode=task_mode,
        )
        args.num_envs = min(args.num_envs, 8)
        args.buffer = min(args.buffer, 128)
        args.batch = min(args.batch, 16)
        args.metrics_every = 1
    if args.episode_length_s is None:
        args.episode_length_s = resolve_default_episode_length_s(
            task_mode=task_mode,
            smoke=args.smoke,
        )
    _validate_args(args)
    _seed_everything(args.seed)
    current_runtime_contract = runtime_contract(task_mode)
    teacher_prior = None
    if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE:
        from coupled_teacher_prior import (
            HandResidualScales,
            load_coupled_teacher_prior,
            load_coupled_teacher_prior_payload,
        )

        assert args.curriculum_dataset is not None
        assert args.coupled_teacher_prior is not None
        teacher_prior = load_coupled_teacher_prior(
            args.coupled_teacher_prior,
            args.curriculum_dataset,
            residual_scales=HandResidualScales(
                close_token=args.residual_close_token_scale,
                close_distal=args.residual_close_distal_scale,
                hold_token=args.residual_hold_token_scale,
                hold_distal=args.residual_hold_distal_scale,
            ),
        )
    effective_n_step = 1 if args.smoke else args.n_step
    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoint_final"
    validate_checkpoint_output_separation(
        args.checkpoint,
        output_checkpoint=checkpoint_dir,
    )
    resumed_task_contract: dict[str, Any] | None = None
    if args.checkpoint is not None:
        resumed_task_contract = validate_checkpoint_task_contract(
            args.checkpoint.resolve(),
            task_mode=task_mode,
            n_step=effective_n_step,
            gamma=FLASH_SAC_GAMMA,
            policy_action_authority=current_policy_action_authority,
            policy_router_enabled=policy_router_enabled,
        )
        if teacher_prior is not None:
            prior_entry = resumed_task_contract.get("teacher_action_prior")
            if not isinstance(prior_entry, Mapping):
                raise ValueError("residual checkpoint has no teacher prior lineage")
            checkpoint_prior = load_coupled_teacher_prior_payload(
                args.checkpoint.resolve() / str(prior_entry["filename"]),
                expected_sha256=str(prior_entry["sha256"]),
            )
            if checkpoint_prior.contract_payload() != teacher_prior.contract_payload():
                raise ValueError(
                    "--checkpoint teacher prior/scales differ from the active artifact and curriculum"
                )
        if args.resume_replay:
            # Keep the replay-specific validation as an explicit second guard:
            # it documents why replay continuation is legal in this run.
            validate_replay_task_contract(
                args.checkpoint.resolve(),
                task_mode=task_mode,
                n_step=effective_n_step,
                gamma=FLASH_SAC_GAMMA,
                policy_action_authority=current_policy_action_authority,
                policy_router_enabled=policy_router_enabled,
            )
    residual_native_successes_base = 0
    residual_actor_unlocked = task_mode != COUPLED_TEACHER_RESIDUAL_TASK_MODE
    residual_actor_unlock_interaction_step: int | None = None
    if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE:
        if resumed_task_contract is None:
            residual_state = validate_teacher_residual_training_state(
                {
                    "version": 1,
                    "task_mode": COUPLED_TEACHER_RESIDUAL_TASK_MODE,
                    "actor_unlock_successes": args.actor_unlock_successes,
                    "native_strict_successes": 0,
                    "actor_unlocked": False,
                },
                source="fresh residual training state",
            )
        else:
            candidate_state = resumed_task_contract.get("teacher_residual_state")
            if not isinstance(candidate_state, Mapping):
                raise ValueError("residual checkpoint has no resumable actor gate state")
            residual_state = validate_teacher_residual_training_state(
                candidate_state,
                source="resumed residual training state",
            )
            if residual_state["actor_unlock_successes"] != args.actor_unlock_successes:
                raise ValueError(
                    "--actor_unlock_successes differs from the residual checkpoint: "
                    f"checkpoint={residual_state['actor_unlock_successes']}, "
                    f"current={args.actor_unlock_successes}"
                )
        residual_native_successes_base = int(
            residual_state["native_strict_successes"]
        )
        residual_actor_unlocked = bool(residual_state["actor_unlocked"])
    actor_checkpoint_audit = (
        audit_actor_checkpoint_source(
            args.actor_checkpoint,
            output_checkpoint=checkpoint_dir,
            target_task_mode=task_mode,
            allow_cross_task_actor=args.allow_cross_task_actor,
        )
        if args.actor_checkpoint is not None
        else None
    )
    frozen_lift_actor_audit = (
        audit_actor_checkpoint_source(
            args.frozen_lift_actor_checkpoint,
            output_checkpoint=checkpoint_dir,
            target_task_mode=FULL_TASK_MODE,
            allow_cross_task_actor=False,
        )
        if args.frozen_lift_actor_checkpoint is not None
        else None
    )
    if frozen_lift_actor_audit is not None:
        frozen_requirements = {
            "source_task_mode": FULL_TASK_MODE,
            "source_policy_action_dim": PICK_TOOL_ACTION_DIM,
            "target_policy_action_dim": PICK_TOOL_ACTION_DIM,
            "actor_projection": None,
            "source_policy_router": None,
        }
        frozen_mismatches = {
            key: (frozen_lift_actor_audit.get(key), expected)
            for key, expected in frozen_requirements.items()
            if frozen_lift_actor_audit.get(key) != expected
        }
        if frozen_mismatches:
            raise ValueError(
                "--frozen_lift_actor_checkpoint must be an ordinary full-task "
                f"115D/21D actor; mismatches={frozen_mismatches}"
            )
    demo_bc_weight = (
        1.0
        if (args.demo is not None or args.actor_demo is not None)
        and args.demo_bc_weight is None
        else float(args.demo_bc_weight or 0.0)
    )
    demo_bc_group_weights = (
        {
            "token": float(args.demo_bc_token_weight),
            "residual": float(args.demo_bc_residual_weight),
        }
        if task_mode in (
            POWER_CLOSE_OPTION_TASK_MODE,
            COUPLED_TEACHER_RESIDUAL_TASK_MODE,
        )
        else {
            "arm": float(args.demo_bc_arm_weight),
            "token": float(args.demo_bc_token_weight),
            "residual": float(args.demo_bc_residual_weight),
        }
    )

    # These imports require the simulator process (and, for the task, its USD
    # plugins) to be initialized by AppLauncher first.
    from adapter import build_replay_transition, make_pick_tool_env
    from agent_bridge import (
        FLASH_SAC_COMMIT,
        FLASH_SAC_FORK_COMMIT,
        ActionAuthorityRule,
        ActionNoiseGroup,
        FlashSACTorchBridge,
        PublicLatchFrozenActorRouter,
        build_agent_config,
    )
    from actor_rehearsal import (
        PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
        PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        PICK_TOOL_LIFT_ACTOR_DEMO_CONTRACTS,
        ActorRehearsalReservoir,
        load_actor_rehearsal,
    )
    from demo_replay import (
        PermanentDemoReservoir,
        attach_demo_replay,
        load_and_precompute_n_step,
    )

    device = str(args.device or "cuda:0")
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"FlashSAC PickTool training requires CUDA, got {device}")

    curriculum_metrics: dict[str, Any] = {
        "curriculum_dataset": None,
        "curriculum_dataset_sha256": None,
        "curriculum_boundary": None,
        "curriculum_probability": 0.0,
        "curriculum_joint_noise": 0.0,
    }
    if args.curriculum_dataset is not None:
        curriculum_metrics = {
            "curriculum_dataset": str(args.curriculum_dataset.resolve()),
            "curriculum_dataset_sha256": _sha256(args.curriculum_dataset),
            "curriculum_boundary": args.curriculum_boundary,
            "curriculum_probability": args.curriculum_probability,
            "curriculum_joint_noise": args.curriculum_joint_noise,
        }

    cfg_overrides = dict(env_task_mode_overrides)
    if is_coupled_task_mode(task_mode):
        cfg_overrides.update(
            {
                "observation_space": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
                "state_space": COUPLED_POWER_ALIGN_CLOSE_OBSERVATION_DIM,
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
        device=device,
        seed=args.seed,
        cfg_overrides=cfg_overrides,
        hand_only_actions=task_mode == POWER_CLOSE_OPTION_TASK_MODE,
        action_transform=teacher_prior,
        validate_finite=args.smoke or args.validate_finite,
    )
    online_handoff_enabled = args.search_handoff_checkpoint is not None
    search_handoff_actor = None
    search_handoff_checkpoint_path: Path | None = None
    search_handoff_checkpoint_sha256: str | None = None
    if online_handoff_enabled:
        from evaluate import _load_diagnostic_approach_actor

        assert args.search_handoff_checkpoint is not None
        search_handoff_checkpoint_path = args.search_handoff_checkpoint.resolve()
        search_handoff_checkpoint_sha256 = _sha256(search_handoff_checkpoint_path)
        search_handoff_actor = _load_diagnostic_approach_actor(
            search_handoff_checkpoint_path,
            device=torch.device(device),
        ).eval()
        if _sha256(search_handoff_checkpoint_path) != search_handoff_checkpoint_sha256:
            raise RuntimeError(
                "--search_handoff_checkpoint changed while its actor was loading"
            )
    warmup_transitions = resolve_warmup_transitions(
        buffer=args.buffer,
        batch=args.batch,
        smoke=args.smoke,
        requested=args.warmup,
    )
    planned_updates = max(1, math.ceil(args.steps * args.updates))
    lr_decay_updates = args.lr_decay_updates or planned_updates
    lr_warmup_updates = args.lr_warmup_updates or max(1, lr_decay_updates // 20)
    if lr_warmup_updates > lr_decay_updates:
        raise ValueError("--lr_warmup_updates cannot exceed --lr_decay_updates")
    agent_cfg = build_agent_config(
        seed=args.seed,
        device_type=device,
        buffer_device_type=device,
        buffer_max_length=args.buffer,
        buffer_min_length=warmup_transitions,
        sample_batch_size=args.batch,
        normalize_reward=True,
        normalized_G_max=5.0,
        n_step=effective_n_step,
        gamma=FLASH_SAC_GAMMA,
        actor_num_blocks=1 if args.smoke else args.actor_blocks,
        actor_hidden_dim=32 if args.smoke else args.actor_hidden,
        actor_update_period=args.actor_update_period,
        critic_num_blocks=1 if args.smoke else args.critic_blocks,
        critic_hidden_dim=64 if args.smoke else args.critic_hidden,
        critic_num_bins=51 if args.smoke else args.critic_bins,
        learning_rate_warmup_step=lr_warmup_updates,
        learning_rate_decay_step=lr_decay_updates,
        use_compile=not args.smoke and not args.no_compile,
        compile_mode="default" if args.smoke else "reduce-overhead",
        use_amp=not args.smoke and not args.no_amp,
        load_optimizer=args.checkpoint is not None,
        load_reward_normalizer=args.checkpoint is not None,
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
        for name, start, stop, scale, zeta_mu, zeta_max in action_noise_group_specs(
            task_mode
        )
    )
    agent = FlashSACTorchBridge(
        env.observation_space,
        env.action_space,
        env.env_info,
        agent_cfg,
        noise_groups=noise_groups,
        restore_rng_state_on_load=False,
        actor_action_active_observation_index=(
            COUPLED_ALIGN_ACTIVE_OBSERVATION_INDEX
            if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
            else None
        ),
        action_authority_rules=(
            tuple(ActionAuthorityRule(**rule) for rule in current_policy_action_authority)
        ),
        public_latch_frozen_actor_router=(
            PublicLatchFrozenActorRouter(
                name=PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_KIND,
                observation_index=PICK_TOOL_LATCH_OBSERVATION_INDEX,
                trainable_start=PICK_TOOL_ARM_ACTION_DIM,
                trainable_stop=PICK_TOOL_ACTION_DIM,
            )
            if policy_router_enabled
            else None
        ),
        unit_normalize_actor_mean_head=(
            task_mode != COUPLED_TEACHER_RESIDUAL_TASK_MODE
        ),
    )
    if (
        task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
        and args.checkpoint is None
    ):
        agent.initialize_zero_actor_mean()

    demo_replay = None
    demo_metrics: dict[str, Any] = {
        "demo_sources": [],
        "demo_replay_transitions": 0,
        "demo_fraction": 0.0,
        "demo_rows_per_batch": 0,
        "demo_phase_counts": {},
        "demo_max_abs_n_step_reward": 0.0,
    }
    actor_batches: list[dict[str, torch.Tensor]] = []
    actor_phases: list[torch.Tensor | None] = []
    actor_source_metrics: list[dict[str, Any]] = []
    actor_source_fingerprints: list[str] = []
    demo_max_abs_reward: torch.Tensor | None = None
    if args.demo is not None:
        demo_audits = [audit_pick_tool_demonstrations(path) for path in args.demo]
        loaded_demos = [
            load_and_precompute_n_step(
                path.resolve(),
                device=device,
                n_step=agent_cfg.n_step,
                gamma=agent_cfg.gamma,
            )
            for path in args.demo
        ]
        demo_capacity = sum(int(batch["observation"].shape[0]) for batch, _ in loaded_demos)
        reservoir = PermanentDemoReservoir(
            capacity=demo_capacity,
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            n_step=agent_cfg.n_step,
            gamma=agent_cfg.gamma,
            device=device,
        )
        all_labels: list[torch.Tensor] = []
        reward_maxima: list[torch.Tensor] = []
        source_metrics: list[dict[str, Any]] = []
        for path, audit, (batch, labels) in zip(
            args.demo, demo_audits, loaded_demos, strict=True
        ):
            reservoir.add_precomputed(
                batch,
                n_step=agent_cfg.n_step,
                gamma=agent_cfg.gamma,
                phase=labels,
            )
            if labels is not None:
                all_labels.append(labels)
            reward_maxima.append(batch["reward"].abs().max())
            source_metrics.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "transitions": int(batch["observation"].shape[0]),
                    **audit,
                }
            )
            actor_batches.append(
                {"observation": batch["observation"], "action": batch["action"]}
            )
            actor_phases.append(labels)
            actor_source_metrics.append(
                {
                    "role": "critic_transition_projection",
                    "path": str(path.resolve()),
                    "sha256": source_metrics[-1]["sha256"],
                    "transitions": int(batch["observation"].shape[0]),
                }
            )
            actor_source_fingerprints.append(source_metrics[-1]["sha256"])
        reservoir.seal()
        demo_replay = attach_demo_replay(
            agent,
            reservoir,
            batch_size=args.batch,
            demo_fraction=args.demo_fraction,
            seed=args.seed + 1_000_003,
            demo_fingerprints=tuple(source["sha256"] for source in source_metrics),
        )
        demo_max_abs_reward = torch.stack(reward_maxima).max()
        phase_counts: dict[str, int] = {}
        if all_labels:
            values, counts = torch.unique(torch.cat(all_labels), sorted=True, return_counts=True)
            phase_counts = {
                str(int(value)): int(count)
                for value, count in zip(
                    values.detach().cpu().tolist(),
                    counts.detach().cpu().tolist(),
                    strict=True,
                )
            }
        demo_metrics = {
            "demo_sources": source_metrics,
            "demo_replay_transitions": demo_replay.demo_size,
            "demo_fraction": demo_replay.demo_fraction,
            "demo_rows_per_batch": demo_replay.demo_rows_per_batch,
            "demo_phase_counts": phase_counts,
            "demo_max_abs_n_step_reward": float(demo_max_abs_reward.item()),
        }

    if args.actor_demo is not None:
        if task_mode in (
            COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
            COUPLED_TEACHER_RESIDUAL_TASK_MODE,
        ):
            actor_demo_contracts = PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS
        elif is_close_option_task_mode(task_mode):
            actor_demo_contracts = PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS
        elif policy_router_enabled:
            actor_demo_contracts = PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS
        else:
            actor_demo_contracts = PICK_TOOL_LIFT_ACTOR_DEMO_CONTRACTS
        for path in args.actor_demo:
            source_action_dim = (
                PICK_TOOL_ENVIRONMENT_ACTION_DIM
                if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
                else env.action_dim
            )
            batch, labels, audit = load_actor_rehearsal(
                path.resolve(),
                device=device,
                observation_dim=env.observation_dim,
                action_dim=source_action_dim,
                allowed_contracts=actor_demo_contracts,
            )
            if task_mode in (
                COUPLED_POWER_ALIGN_CLOSE_OPTION_TASK_MODE,
                COUPLED_TEACHER_RESIDUAL_TASK_MODE,
            ):
                initial_curriculum_sha256 = curriculum_metrics[
                    "curriculum_dataset_sha256"
                ]
                if (
                    not isinstance(initial_curriculum_sha256, str)
                    or args.curriculum_dataset is None
                ):
                    raise RuntimeError(
                        "coupled actor-demo validation requires an active curriculum SHA256"
                    )
                expected_curriculum_sha256 = _sha256(args.curriculum_dataset)
                if expected_curriculum_sha256 != initial_curriculum_sha256:
                    raise RuntimeError(
                        "coupled curriculum dataset changed while the run was starting"
                    )
                validate_actor_demo_curriculum_lineage(
                    audit,
                    expected_curriculum_sha256=expected_curriculum_sha256,
                )
            if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE:
                if teacher_prior is None:
                    raise RuntimeError("teacher-residual actor demo requires an active prior")
                validate_teacher_residual_actor_demo_lineage(
                    audit,
                    expected_curriculum_sha256=expected_curriculum_sha256,
                    expected_teacher_artifact_sha256=(
                        teacher_prior.teacher_artifact_sha256
                    ),
                )
                batch = project_coupled_teacher_demo_to_zero_residual(batch)
                audit = {
                    **audit,
                    "source_policy_action_dim": PICK_TOOL_ENVIRONMENT_ACTION_DIM,
                    "projected_policy_action_dim": PICK_TOOL_HAND_ACTION_DIM,
                    "action_projection": (
                        "audited_coupled_teacher21_to_exact_zero_residual14_v1"
                    ),
                }
            if policy_router_enabled:
                row_count = int(batch["observation"].shape[0])
                latch = batch["observation"][:, PICK_TOOL_LATCH_OBSERVATION_INDEX]
                binary = (latch == 0.0) | (latch == 1.0)
                if not bool(binary.all()):
                    raise ValueError(
                        f"{path}: routed actor demo latch observation must be exactly binary"
                    )
                keep = latch == 0.0
                kept_rows = int(keep.sum().item())
                if kept_rows == 0:
                    raise ValueError(
                        f"{path}: routed actor demo has no unlatched close-policy rows"
                    )
                batch = {
                    key: value[keep]
                    for key, value in batch.items()
                }
                if labels is not None:
                    labels = labels[keep]
                audit = {
                    **audit,
                    "transitions": kept_rows,
                    "router_input_rows": row_count,
                    "router_unlatched_rows": kept_rows,
                    "router_removed_latched_rows": row_count - kept_rows,
                }
            if current_policy_action_authority:
                source_action = batch["action"]
                projected_action = (
                    agent.apply_trainable_action_authority(
                        source_action, batch["observation"]
                    )
                    if policy_router_enabled
                    else agent.apply_action_authority(
                        source_action, batch["observation"]
                    )
                )
                changed = projected_action != source_action
                batch = {
                    **batch,
                    "action": projected_action,
                }
                audit = {
                    **audit,
                    "policy_action_authority": current_policy_action_authority,
                    "authority_projected_elements": int(changed.sum().item()),
                    "authority_removed_max_abs_action": (
                        float(source_action[changed].abs().max().item())
                        if bool(changed.any())
                        else 0.0
                    ),
                }
            actor_batches.append(batch)
            actor_phases.append(labels)
            actor_source_metrics.append({"role": "actor_only", **audit})
            actor_source_fingerprints.append(str(audit["sha256"]))

    actor_rehearsal = None
    actor_rehearsal_metrics: dict[str, Any] = {
        "actor_demo_sources": [],
        "actor_rehearsal_transitions": 0,
        "actor_rehearsal_batch": 0,
        "actor_rehearsal_phase_counts": {},
        "actor_rehearsal_stratum_weights": None,
        "actor_rehearsal_resumed": False,
    }
    if actor_batches:
        if any(labels is None for labels in actor_phases) and not all(
            labels is None for labels in actor_phases
        ):
            raise ValueError(
                "all actor rehearsal sources must consistently include or omit phase labels"
            )
        available_phases: tuple[int, ...] = ()
        phase_counts: dict[str, int] = {}
        if actor_phases[0] is not None:
            concatenated_phase = torch.cat(
                [labels for labels in actor_phases if labels is not None]
            )
            values, counts = torch.unique(
                concatenated_phase, sorted=True, return_counts=True
            )
            available_phases = tuple(int(value) for value in values.detach().cpu().tolist())
            phase_counts = {
                str(int(value)): int(count)
                for value, count in zip(
                    values.detach().cpu().tolist(),
                    counts.detach().cpu().tolist(),
                    strict=True,
                )
            }
        actor_stratum_weights: dict[int, float] | None = None
        if args.demo_bc_phases is not None:
            selected_phases = set(args.demo_bc_phases)
            missing_phases = sorted(selected_phases.difference(available_phases))
            if missing_phases:
                raise ValueError(
                    "--demo_bc_phases contains phases absent from actor demonstrations: "
                    f"{missing_phases}; available={list(available_phases)}"
                )
            actor_stratum_weights = {
                phase: float(phase in selected_phases) for phase in available_phases
            }
        actor_rehearsal_batch = (
            int(args.demo_bc_batch)
            if args.demo_bc_batch is not None
            else (
                demo_replay.demo_rows_per_batch
                if demo_replay is not None
                else max(1, args.batch // 4)
            )
        )
        actor_rehearsal = ActorRehearsalReservoir(
            capacity=sum(int(batch["observation"].shape[0]) for batch in actor_batches),
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            device=device,
            seed=args.seed + 2_000_003,
            source_fingerprints=actor_source_fingerprints,
            default_batch_size=actor_rehearsal_batch,
            stratum_weights=actor_stratum_weights,
        )
        for batch, labels in zip(actor_batches, actor_phases, strict=True):
            actor_rehearsal.add(batch, phase=labels)
        actor_rehearsal.seal()
        actor_rehearsal_metrics = {
            "actor_demo_sources": actor_source_metrics,
            "actor_rehearsal_transitions": len(actor_rehearsal),
            "actor_rehearsal_batch": actor_rehearsal_batch,
            "actor_rehearsal_phase_counts": phase_counts,
            "actor_rehearsal_stratum_weights": actor_stratum_weights,
            "actor_rehearsal_resumed": False,
        }

    if args.checkpoint is not None:
        revalidated_contract = validate_checkpoint_task_contract(
            args.checkpoint.resolve(),
            task_mode=task_mode,
            n_step=effective_n_step,
            gamma=FLASH_SAC_GAMMA,
            policy_action_authority=current_policy_action_authority,
            policy_router_enabled=policy_router_enabled,
        )
        if revalidated_contract != resumed_task_contract:
            raise RuntimeError("--checkpoint task contract changed while the run was starting")
        agent.load(str(args.checkpoint.resolve()))
        post_load_contract = validate_checkpoint_task_contract(
            args.checkpoint.resolve(),
            task_mode=task_mode,
            n_step=effective_n_step,
            gamma=FLASH_SAC_GAMMA,
            policy_action_authority=current_policy_action_authority,
            policy_router_enabled=policy_router_enabled,
        )
        if post_load_contract != resumed_task_contract:
            raise RuntimeError("--checkpoint changed while it was being loaded")
        if args.resume_replay:
            replay_path = args.checkpoint.resolve() / "replay_buffer.pt"
            if not replay_path.is_file():
                raise FileNotFoundError(replay_path)
            agent.load_replay_buffer(str(args.checkpoint.resolve()))
        if args.resume_actor_demo:
            if actor_rehearsal is None:
                raise RuntimeError("--resume_actor_demo requires an actor rehearsal reservoir")
            actor_rehearsal_path = args.checkpoint.resolve() / "actor_rehearsal.pt"
            if not actor_rehearsal_path.is_file():
                raise FileNotFoundError(actor_rehearsal_path)
            actor_rehearsal.load(actor_rehearsal_path)
            actor_rehearsal_metrics["actor_rehearsal_resumed"] = True
    elif args.actor_checkpoint is not None:
        assert actor_checkpoint_audit is not None
        revalidated_actor = audit_actor_checkpoint_source(
            args.actor_checkpoint,
            output_checkpoint=checkpoint_dir,
            target_task_mode=task_mode,
            allow_cross_task_actor=args.allow_cross_task_actor,
        )
        if revalidated_actor != actor_checkpoint_audit:
            raise RuntimeError("--actor_checkpoint changed while the run was starting")
        load_audited_actor_checkpoint(agent, actor_checkpoint_audit)
        loaded_actor_sha256 = _sha256(
            Path(actor_checkpoint_audit["path"]) / "actor.pt"
        )
        if loaded_actor_sha256 != actor_checkpoint_audit["actor_sha256"]:
            raise RuntimeError("--actor_checkpoint changed while actor.pt was being loaded")
        if policy_router_enabled:
            if frozen_lift_actor_audit is None or args.frozen_lift_actor_checkpoint is None:
                raise RuntimeError("fresh policy router has no audited frozen lift actor")
            revalidated_frozen = audit_actor_checkpoint_source(
                args.frozen_lift_actor_checkpoint,
                output_checkpoint=checkpoint_dir,
                target_task_mode=FULL_TASK_MODE,
                allow_cross_task_actor=False,
            )
            if revalidated_frozen != frozen_lift_actor_audit:
                raise RuntimeError(
                    "--frozen_lift_actor_checkpoint changed while the run was starting"
                )
            agent.load_frozen_lift_actor(frozen_lift_actor_audit["path"])
            loaded_frozen_sha256 = _sha256(
                Path(frozen_lift_actor_audit["path"]) / "actor.pt"
            )
            if (
                loaded_frozen_sha256 != frozen_lift_actor_audit["actor_sha256"]
                or agent.frozen_lift_actor_source_sha256 != loaded_frozen_sha256
            ):
                raise RuntimeError(
                    "--frozen_lift_actor_checkpoint changed while actor.pt was being loaded"
                )
        if args.public_latch_arm_gate and agent.replay_size != 0:
            raise RuntimeError(
                "public latch arm-gate actor-only initialization must start with fresh replay"
            )
    if args.actor_lr_scale is not None:
        agent.set_actor_learning_rate_scale(args.actor_lr_scale)
    actor_lr_scale = agent.actor_learning_rate_scale
    if demo_max_abs_reward is not None:
        if agent.reward_normalizer is None:
            raise RuntimeError("demonstration replay requires the configured reward normalizer")
        # Demonstration terminal bonuses are present in sampled replay but are
        # absent from online-only running-return statistics until a success is
        # rediscovered.  Prime the hard normalization cap so a 500-point demo
        # success maps to at most normalized_G_max instead of destabilizing the
        # categorical critic.  Preserve a larger value restored from a checkpoint.
        agent.reward_normalizer.G_r_max = torch.maximum(
            agent.reward_normalizer.G_r_max,
            demo_max_abs_reward.reshape_as(agent.reward_normalizer.G_r_max),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    observation, _ = env.reset(randomize_episode_lengths=args.randomize_episode_lengths)
    if observation.shape != (env.num_envs, env.observation_dim):
        raise RuntimeError(
            "environment reset violated its observation contract: "
            f"got {tuple(observation.shape)}"
        )
    agent.start_fresh_rollout(batch_size=env.num_envs)
    episodes = EpisodeAccumulator(env.num_envs, env.device)
    terminal_events = TerminalEventAccumulator(
        env.num_envs,
        env.device,
        task_mode=task_mode,
    )
    update_budget = FractionalUpdateBudget(args.updates)
    update_count = 0
    terminated_count = torch.zeros((), dtype=torch.long, device=env.device)
    truncated_count = torch.zeros((), dtype=torch.long, device=env.device)
    router_route_counts = torch.zeros(2, dtype=torch.long, device=env.device)
    handoff_ready_count = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    handoff_option_active = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    handoff_trigger_count = torch.zeros((), dtype=torch.long, device=env.device)
    handoff_search_action_rows = torch.zeros(
        (), dtype=torch.long, device=env.device
    )
    handoff_option_action_rows = torch.zeros(
        (), dtype=torch.long, device=env.device
    )
    handoff_completed_triggered = torch.zeros(
        (), dtype=torch.long, device=env.device
    )
    handoff_completed_search_only = torch.zeros(
        (), dtype=torch.long, device=env.device
    )
    handoff_terminal_counts = {
        name: torch.zeros((), dtype=torch.long, device=env.device)
        for name in TERMINAL_EVENT_KEYS
    }
    # This is an audit cohort, not routing state: every environment row enters
    # pending exactly once at run start and leaves on its first completed
    # episode, even though the simulator continues auto-resetting that row.
    handoff_initial_episode_pending = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    handoff_initial_episode_triggered_seen = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    handoff_initial_episode_trigger_count = torch.zeros(
        (), dtype=torch.long, device=env.device
    )
    handoff_initial_episode_completed_triggered = torch.zeros(
        (), dtype=torch.long, device=env.device
    )
    handoff_initial_episode_completed_search_only = torch.zeros(
        (), dtype=torch.long, device=env.device
    )
    handoff_initial_episode_triggered_terminal_counts = {
        name: torch.zeros((), dtype=torch.long, device=env.device)
        for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS
    }
    handoff_initial_episode_triggered_terminal_masks = {
        name: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS
    }
    update_sums: dict[str, float] = {}
    update_metric_counts: dict[str, int] = {}
    actor_update_count = 0
    demo_bc_update_count = 0
    instant_strict: dict[str, float] = {}
    run_max_strict: dict[str, float] = {}
    started = time.perf_counter()

    def current_residual_native_successes() -> int:
        if task_mode != COUPLED_TEACHER_RESIDUAL_TASK_MODE:
            return 0
        return residual_native_successes_base + int(
            terminal_events.counts["power_close_option_success"].item()
        )

    try:
        for interaction_step in range(1, args.steps + 1):
            option_control_mask = torch.ones(
                env.num_envs, dtype=torch.bool, device=env.device
            )
            if online_handoff_enabled:
                handoff = update_online_handoff(
                    observation,
                    ready_count_before=handoff_ready_count,
                    option_active_before=handoff_option_active,
                    min_score=args.search_handoff_min_score,
                    hold_steps=args.search_handoff_hold_steps,
                )
                handoff_ready_count = handoff["ready_count_after"]
                handoff_option_active = handoff["option_active_after"]
                option_control_mask = handoff_option_active
                handoff_trigger_count.add_(handoff["trigger"].sum())
            if policy_router_enabled:
                latch = observation[:, PICK_TOOL_LATCH_OBSERVATION_INDEX]
                router_route_counts[0].add_(
                    (option_control_mask & (latch == 0.0)).sum()
                )
                router_route_counts[1].add_(
                    (option_control_mask & (latch == 1.0)).sum()
                )
            training_ready = agent.can_start_training()
            if (
                task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
                and not residual_actor_unlocked
            ):
                # Exact teacher-prior collection is the exploration bridge.
                # Keep the canonical replay action at zero until a native
                # strict success transition has actually entered online replay.
                action = torch.zeros(
                    (env.num_envs, env.action_dim),
                    dtype=torch.float32,
                    device=env.device,
                )
            elif (
                task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
                or args.checkpoint is not None
                or args.actor_checkpoint is not None
                or training_ready
            ):
                noise_scale = None
                if (
                    args.unlatched_arm_noise_scale != 1.0
                    or args.unlatched_hand_noise_scale != 1.0
                    or args.latched_arm_noise_scale != 1.0
                    or args.latched_hand_noise_scale != 1.0
                ):
                    # PickTool's public 115-D Markov observation stores the
                    # grasp-latch bit at index 106.  This is not privileged
                    # simulator state: the actor sees the same bit.  Scale
                    # exploration independently on both sides of the latch.
                    noise_scale = build_latch_conditioned_noise_scale(
                        observation,
                        unlatched_arm=args.unlatched_arm_noise_scale,
                        unlatched_hand=args.unlatched_hand_noise_scale,
                        latched_arm=args.latched_arm_noise_scale,
                        latched_hand=args.latched_hand_noise_scale,
                        action_dim=env.action_dim,
                    )
                action = agent.sample_actions(
                    interaction_step,
                    {"next_observation": observation},
                    # A loaded BC policy controls collection from the first
                    # frame.  Keep it deterministic during replay warm-up and
                    # critic-only burn-in so white noise cannot destroy a
                    # captured close/lift curriculum state before learning.
                    training=(
                        training_ready and update_count >= args.critic_burnin_updates
                    ),
                    noise_scale=noise_scale,
                )
            else:
                action = env.sample_random_actions()

            # Every option proposal source converges here, including random
            # warm-up.  Canonicalize the option before merging it with SEARCH:
            # applying latch authority after the merge would incorrectly zero
            # SEARCH's arm command on pre-handoff rows.
            option_action = agent.apply_action_authority(action, observation)
            if online_handoff_enabled:
                assert search_handoff_actor is not None
                with torch.no_grad():
                    search_action = search_handoff_actor(observation).clamp(-1.0, 1.0)
                if search_action.shape != option_action.shape:
                    raise RuntimeError(
                        "SEARCH actor action shape disagrees with the routed option: "
                        f"search={tuple(search_action.shape)}, "
                        f"option={tuple(option_action.shape)}"
                    )
                if not bool(torch.isfinite(search_action).all()):
                    raise FloatingPointError("frozen SEARCH actor produced NaN or infinity")
                action = torch.where(
                    option_control_mask.unsqueeze(-1),
                    option_action,
                    search_action,
                )
                handoff_option_action_rows.add_(option_control_mask.sum())
                handoff_search_action_rows.add_((~option_control_mask).sum())
            else:
                action = option_action
            next_observation, reward, terminated, truncated, info = env.step(action)
            # Replay the exact canonical policy action that produced this
            # transition, never the 21-D teacher-composed environment command.
            # The adapter owns canonicalization so execution and replay cannot
            # classify the phase from two different observation snapshots.
            executed_policy_action = env.last_executed_policy_action
            if executed_policy_action is None:
                raise RuntimeError("adapter did not publish its executed policy action")
            transition = build_replay_transition(
                observation,
                executed_policy_action,
                reward,
                terminated,
                truncated,
                info,
                action_dim=env.action_dim,
            )
            if online_handoff_enabled:
                materialized_replay_rows = agent.process_transition_masked(
                    transition,
                    replay_valid_mask=option_control_mask,
                )
            else:
                materialized_replay_rows = agent.process_transition(transition)

            done = terminated | truncated
            episodes.step(reward, done)
            terminal_events.step(info)
            if online_handoff_enabled:
                triggered_done = done & option_control_mask
                search_only_done = done & (~option_control_mask)
                handoff_completed_triggered.add_(triggered_done.sum())
                handoff_completed_search_only.add_(search_only_done.sum())
                terminal_truth = info.get("pick_tool_terminal")
                if not isinstance(terminal_truth, Mapping):
                    raise KeyError(
                        "online SEARCH handoff requires pick_tool_terminal ground truth"
                    )
                initial_episode_audit = update_initial_episode_handoff_audit(
                    pending_before=handoff_initial_episode_pending,
                    triggered_seen_before=handoff_initial_episode_triggered_seen,
                    trigger=handoff["trigger"],
                    done=done,
                    terminal_truth=terminal_truth,
                )
                handoff_initial_episode_pending = initial_episode_audit[
                    "pending_after"
                ]
                handoff_initial_episode_triggered_seen = initial_episode_audit[
                    "triggered_seen_after"
                ]
                handoff_initial_episode_trigger_count.add_(
                    initial_episode_audit["trigger_count"]
                )
                handoff_initial_episode_completed_triggered.add_(
                    initial_episode_audit["completed_triggered_episodes"]
                )
                handoff_initial_episode_completed_search_only.add_(
                    initial_episode_audit["completed_search_only_episodes"]
                )
                for name, count in (
                    handoff_initial_episode_triggered_terminal_counts.items()
                ):
                    count.add_(
                        initial_episode_audit["triggered_terminal_counts"][name]
                    )
                    handoff_initial_episode_triggered_terminal_masks[name] |= (
                        initial_episode_audit["triggered_terminal_masks"][name]
                    )
                for name, count in handoff_terminal_counts.items():
                    value = terminal_truth.get(name)
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.shape != (env.num_envs,)
                        or value.device != env.device
                    ):
                        raise RuntimeError(
                            f"pick_tool_terminal[{name!r}] must be a device-local [N] tensor"
                        )
                    count.add_((triggered_done & value.bool()).sum())
            if (
                task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
                and not residual_actor_unlocked
                and residual_actor_should_unlock(
                    native_strict_successes=current_residual_native_successes(),
                    actor_unlock_successes=args.actor_unlock_successes,
                    success_transition_in_replay=(materialized_replay_rows > 0),
                )
            ):
                # This check occurs after process_transition(): the strict
                # success that opens actor authority is already in replay.
                residual_actor_unlocked = True
                residual_actor_unlock_interaction_step = interaction_step
            instant_strict = _strict_metrics(info, task_mode=task_mode)
            for name, value in instant_strict.items():
                run_max_strict[name] = max(run_max_strict.get(name, -math.inf), value)
            terminated_count.add_(terminated.sum())
            truncated_count.add_(truncated.sum())

            # Rollout continues from reset observations.  Replay has already
            # cloned the captured terminal observations in ``transition``.
            observation = next_observation
            if online_handoff_enabled:
                handoff_ready_count, handoff_option_active = (
                    reset_online_handoff_state(
                        ready_count=handoff_ready_count,
                        option_active=handoff_option_active,
                        done=done,
                    )
                )
            agent.reset_exploration(env_ids=done.nonzero(as_tuple=False).squeeze(-1))

            update_weight_numerator = (
                materialized_replay_rows if online_handoff_enabled else env.num_envs
            )
            for _ in range(
                update_budget.grant(
                    agent.can_start_training(),
                    weight_numerator=update_weight_numerator,
                    weight_denominator=env.num_envs,
                )
            ):
                update_info = agent.update(
                    actor_enabled=(
                        update_count >= args.critic_burnin_updates
                        and residual_actor_unlocked
                    ),
                    policy_actions_enabled=residual_actor_unlocked,
                )
                update_count += 1
                actor_was_updated = (
                    update_info.get(
                        "actor/updated",
                        1.0 if "actor/loss" in update_info else 0.0,
                    )
                    > 0.5
                )
                if actor_was_updated:
                    actor_update_count += 1
                    if actor_rehearsal is not None and demo_bc_weight > 0.0:
                        rehearsal_batch = actor_rehearsal.sample()
                        update_info.update(
                            agent.demo_bc_rehearsal(
                                rehearsal_batch,
                                weight=demo_bc_weight,
                                group_weights=demo_bc_group_weights,
                                target_std=args.demo_bc_target_std,
                                std_weight=args.demo_bc_std_weight,
                            )
                        )
                        demo_bc_update_count += 1
                for name, value in update_info.items():
                    update_sums[name] = update_sums.get(name, 0.0) + _scalar(
                        value, name=name
                    )
                    update_metric_counts[name] = update_metric_counts.get(name, 0) + 1

            if interaction_step % args.metrics_every == 0 or interaction_step == args.steps:
                if (
                    online_handoff_enabled
                    and search_handoff_checkpoint_path is not None
                    and _sha256(search_handoff_checkpoint_path)
                    != search_handoff_checkpoint_sha256
                ):
                    raise RuntimeError(
                        "--search_handoff_checkpoint changed during online training"
                    )
                elapsed = max(time.perf_counter() - started, 1.0e-9)
                metrics: dict[str, Any] = {
                    "seed": args.seed,
                    "smoke": bool(args.smoke),
                    "task_mode": task_mode,
                    "interaction_step": interaction_step,
                    "environment_steps": interaction_step * env.num_envs,
                    "gradient_updates": update_count,
                    "actor_updates": actor_update_count,
                    "actor_update_period": args.actor_update_period,
                    "actor_lr_scale": actor_lr_scale,
                    "critic_burnin_updates": args.critic_burnin_updates,
                    "residual_actor_unlocked": residual_actor_unlocked,
                    "residual_actor_unlock_successes": (
                        args.actor_unlock_successes
                        if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
                        else None
                    ),
                    "residual_native_strict_successes": (
                        current_residual_native_successes()
                        if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE
                        else None
                    ),
                    "residual_actor_unlock_interaction_step": (
                        residual_actor_unlock_interaction_step
                    ),
                    "teacher_action_prior": (
                        teacher_prior.contract_payload()
                        if teacher_prior is not None
                        else None
                    ),
                    "demo_bc_updates": demo_bc_update_count,
                    "demo_bc_weight": demo_bc_weight,
                    "demo_bc_group_weights": demo_bc_group_weights,
                    "lr_decay_updates": lr_decay_updates,
                    "lr_warmup_updates": lr_warmup_updates,
                    "initial_checkpoint": (
                        str(args.checkpoint.resolve()) if args.checkpoint is not None else None
                    ),
                    "initial_actor_checkpoint": (
                        actor_checkpoint_audit["path"] if actor_checkpoint_audit else None
                    ),
                    "initial_actor_checkpoint_sha256": (
                        actor_checkpoint_audit["actor_sha256"]
                        if actor_checkpoint_audit
                        else None
                    ),
                    "initial_actor_checkpoint_source_task_mode": (
                        actor_checkpoint_audit["source_task_mode"]
                        if actor_checkpoint_audit
                        else None
                    ),
                    "initial_actor_checkpoint_source_policy_action_dim": (
                        actor_checkpoint_audit["source_policy_action_dim"]
                        if actor_checkpoint_audit
                        else None
                    ),
                    "initial_actor_checkpoint_source_policy_action_layout": (
                        actor_checkpoint_audit["source_policy_action_layout"]
                        if actor_checkpoint_audit
                        else None
                    ),
                    "initial_actor_checkpoint_source_policy_action_authority": (
                        actor_checkpoint_audit["source_policy_action_authority"]
                        if actor_checkpoint_audit
                        else None
                    ),
                    "initial_actor_checkpoint_source_policy_router": (
                        actor_checkpoint_audit["source_policy_router"]
                        if actor_checkpoint_audit
                        else None
                    ),
                    "initial_actor_checkpoint_projection": (
                        actor_checkpoint_audit["actor_projection"]
                        if actor_checkpoint_audit
                        else None
                    ),
                    "allow_cross_task_actor": bool(args.allow_cross_task_actor),
                    "resumed_replay": bool(args.resume_replay),
                    "resumed_task_contract": resumed_task_contract,
                    "policy_action_authority": current_policy_action_authority,
                    "policy_router_enabled": policy_router_enabled,
                    "policy_router_semantics": expected_policy_router_semantics(
                        policy_router_enabled
                    ),
                    "frozen_lift_actor_checkpoint": (
                        frozen_lift_actor_audit["path"]
                        if frozen_lift_actor_audit
                        else None
                    ),
                    "frozen_lift_actor_source_sha256": (
                        agent.frozen_lift_actor_source_sha256
                        if policy_router_enabled
                        else None
                    ),
                    "frozen_lift_actor_network_sha256": (
                        agent.frozen_lift_actor_sha256
                        if policy_router_enabled
                        else None
                    ),
                    "policy_router_close_rows": int(router_route_counts[0].item()),
                    "policy_router_frozen_rows": int(router_route_counts[1].item()),
                    **online_handoff_metrics(
                        enabled=online_handoff_enabled,
                        search_checkpoint=(
                            str(search_handoff_checkpoint_path)
                            if search_handoff_checkpoint_path is not None
                            else None
                        ),
                        search_checkpoint_sha256=search_handoff_checkpoint_sha256,
                        min_score=args.search_handoff_min_score,
                        hold_steps=args.search_handoff_hold_steps,
                        trigger_count=handoff_trigger_count,
                        search_action_rows=handoff_search_action_rows,
                        option_action_rows=handoff_option_action_rows,
                        completed_triggered_episodes=handoff_completed_triggered,
                        completed_search_only_episodes=handoff_completed_search_only,
                        initial_episode_pending_rows=(
                            handoff_initial_episode_pending.sum()
                        ),
                        initial_episode_trigger_count=(
                            handoff_initial_episode_trigger_count
                        ),
                        initial_episode_completed_triggered_episodes=(
                            handoff_initial_episode_completed_triggered
                        ),
                        initial_episode_completed_search_only_episodes=(
                            handoff_initial_episode_completed_search_only
                        ),
                        initial_episode_triggered_terminal_counts=(
                            handoff_initial_episode_triggered_terminal_counts
                        ),
                        initial_episode_triggered_seen=(
                            handoff_initial_episode_triggered_seen
                        ),
                        initial_episode_triggered_terminal_masks=(
                            handoff_initial_episode_triggered_terminal_masks
                        ),
                    ),
                    **{
                        f"online_search_handoff_terminal/{name}": int(count.item())
                        for name, count in handoff_terminal_counts.items()
                    },
                    "resumed_actor_demo": bool(args.resume_actor_demo),
                    "restore_checkpoint_rng": False,
                    "flashsac_upstream_commit": FLASH_SAC_COMMIT,
                    "flashsac_fork_commit": FLASH_SAC_FORK_COMMIT,
                    **current_runtime_contract,
                    "action_dim": env.action_dim,
                    "buffer_capacity": args.buffer,
                    "warmup_transitions": warmup_transitions,
                    "replay_transitions": agent.replay_size,
                    "n_step": agent_cfg.n_step,
                    "actor_blocks": agent_cfg.actor_num_blocks,
                    "actor_hidden": agent_cfg.actor_hidden_dim,
                    "critic_blocks": agent_cfg.critic_num_blocks,
                    "critic_hidden": agent_cfg.critic_hidden_dim,
                    "critic_bins": agent_cfg.critic_num_bins,
                    **curriculum_metrics,
                    "unlatched_arm_noise_scale": args.unlatched_arm_noise_scale,
                    "unlatched_hand_noise_scale": args.unlatched_hand_noise_scale,
                    "latched_arm_noise_scale": args.latched_arm_noise_scale,
                    "latched_hand_noise_scale": args.latched_hand_noise_scale,
                    **demo_metrics,
                    **actor_rehearsal_metrics,
                    "terminated_events": int(terminated_count.item()),
                    "truncated_events": int(truncated_count.item()),
                    "throughput_env_steps_per_second": interaction_step * env.num_envs / elapsed,
                    **episodes.metrics(),
                    **terminal_events.metrics(),
                    **{
                        f"instant_strict/{name}": value
                        for name, value in instant_strict.items()
                    },
                    **{
                        f"run_max_strict/{name}": value
                        for name, value in run_max_strict.items()
                    },
                }
                actor_optimizer = agent._actor.optimizer
                critic_optimizer = agent._critic.optimizer
                if actor_optimizer is not None:
                    metrics["optimizer/actor_lr"] = float(actor_optimizer.param_groups[0]["lr"])
                if critic_optimizer is not None:
                    metrics["optimizer/critic_lr"] = float(critic_optimizer.param_groups[0]["lr"])
                if update_metric_counts:
                    metrics.update(
                        {
                            f"update/{name}": total / update_metric_counts[name]
                            for name, total in update_sums.items()
                        }
                    )
                atomic_write_json(metrics_path, metrics)

        if (
            online_handoff_enabled
            and search_handoff_checkpoint_path is not None
            and _sha256(search_handoff_checkpoint_path)
            != search_handoff_checkpoint_sha256
        ):
            raise RuntimeError(
                "--search_handoff_checkpoint changed before final checkpoint publication"
            )
        final_residual_state = None
        if task_mode == COUPLED_TEACHER_RESIDUAL_TASK_MODE:
            final_residual_state = validate_teacher_residual_training_state(
                {
                    "version": 1,
                    "task_mode": COUPLED_TEACHER_RESIDUAL_TASK_MODE,
                    "actor_unlock_successes": args.actor_unlock_successes,
                    "native_strict_successes": current_residual_native_successes(),
                    "actor_unlocked": residual_actor_unlocked,
                },
                source="final residual training state",
            )
        save_final_checkpoint(
            checkpoint_dir,
            agent=agent,
            task_mode=task_mode,
            replay_n_step=agent_cfg.n_step,
            replay_gamma=agent_cfg.gamma,
            save_replay=bool(args.save_replay),
            actor_rehearsal=actor_rehearsal,
            policy_action_authority=current_policy_action_authority,
            policy_router_enabled=policy_router_enabled,
            teacher_prior=teacher_prior,
            teacher_residual_state=final_residual_state,
        )
        final_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        final_metrics["checkpoint"] = str(checkpoint_dir)
        final_metrics["checkpoint_task_contract"] = read_checkpoint_task_contract(
            checkpoint_dir
        )
        final_metrics["status"] = "complete"
        atomic_write_json(metrics_path, final_metrics)
        return final_metrics
    finally:
        env.close()


def main() -> None:
    args, launcher = _parse_args()
    try:
        metrics = run(args)
        print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    except BaseException:
        # SimulationApp.close() can terminate Kit before Python renders an
        # uncaught exception.  Emit it first so a failed training launch cannot
        # look like a successful, output-free run.
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
