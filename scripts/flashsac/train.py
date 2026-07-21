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
from typing import Any, Mapping

import numpy as np
import torch


@dataclass
class FractionalUpdateBudget:
    """Exact fractional update accounting without floating-point drift."""

    updates_per_interaction: float
    credit_numerator: int = 0
    _rate: Fraction = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.updates_per_interaction) or self.updates_per_interaction < 0.0:
            raise ValueError("updates_per_interaction must be finite and non-negative")
        self._rate = Fraction(str(self.updates_per_interaction)).limit_denominator(1_000_000)

    def grant(self, training_ready: bool) -> int:
        """Return updates due for one vector interaction.

        Warm-up interactions earn no deferred credit, matching the upstream
        FlashSAC loop rather than causing a burst of catch-up updates.
        """

        if not training_ready:
            return 0
        self.credit_numerator += self._rate.numerator
        due, self.credit_numerator = divmod(self.credit_numerator, self._rate.denominator)
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

TASK_CONTRACT_FILENAME = "task_contract.json"
INCOMPLETE_CHECKPOINT_FILENAME = ".incomplete_checkpoint.json"
TASK_CONTRACT_VERSION = 3
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
    TASK_CONTRACT_FILENAME,
)
FULL_TASK_MODE = "full_task"
CLOSE_OPTION_TASK_MODE = "close_option"
POWER_CLOSE_OPTION_TASK_MODE = "power_close_option_v1"
TASK_MODES = (
    FULL_TASK_MODE,
    CLOSE_OPTION_TASK_MODE,
    POWER_CLOSE_OPTION_TASK_MODE,
)
PICK_TOOL_LATCH_OBSERVATION_INDEX = 106
PICK_TOOL_ARM_ACTION_DIM = 7
PICK_TOOL_HAND_ACTION_DIM = 14
PICK_TOOL_ACTION_DIM = 21
PICK_TOOL_ENVIRONMENT_ACTION_DIM = 21
FULL_POLICY_ACTION_LAYOUT = "arm_delta7|crossdex_token9|distal_residual5"
HAND_POLICY_ACTION_LAYOUT = "crossdex_token9|distal_residual5"
IDENTITY_ACTION_PROJECTION = "identity_v1"
PREPEND_ZERO_ARM_ACTION_PROJECTION = "prepend_zero_arm7_v1"
FULL21_TO_HAND14_ACTOR_PROJECTION = "full21_to_hand14_v1"
PICK_TOOL_OBSERVATION_DIM = 115
PICK_TOOL_OBSERVATION_CONTRACT = "pick_tool_markov115_v1"
POWER_CLOSE_OBSERVATION_CONTRACT = "pick_tool_power_close_markov115_v1"
FULL_ACTION_NOISE_GROUP_SPECS = (
    ("arm", 0, 7, 1.0, 1.0, 64),
    ("token", 7, 16, 0.5, 1.25, 32),
    ("residual", 16, 21, 0.35, 1.5, 16),
)
POWER_ACTION_NOISE_GROUP_SPECS = (
    ("token", 0, 9, 0.5, 1.25, 32),
    ("residual", 9, 14, 0.35, 1.5, 16),
)


def task_mode_from_close_option(
    close_option_mode: bool,
    power_close_option_mode: bool = False,
) -> str:
    """Resolve the mutually-exclusive task flag pair into a checkpoint mode."""

    if close_option_mode and power_close_option_mode:
        raise ValueError(
            "--close_option_mode and --power_close_option_mode are mutually exclusive"
        )
    if power_close_option_mode:
        return POWER_CLOSE_OPTION_TASK_MODE
    return CLOSE_OPTION_TASK_MODE if close_option_mode else FULL_TASK_MODE


def is_close_option_task_mode(task_mode: str) -> bool:
    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    return task_mode in (CLOSE_OPTION_TASK_MODE, POWER_CLOSE_OPTION_TASK_MODE)


def policy_action_contract(task_mode: str) -> dict[str, Any]:
    """Return the policy/environment action boundary fixed by a task mode."""

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
    if task_mode == POWER_CLOSE_OPTION_TASK_MODE:
        return {
            "policy_action_dim": PICK_TOOL_HAND_ACTION_DIM,
            "policy_action_layout": HAND_POLICY_ACTION_LAYOUT,
            "environment_action_dim": PICK_TOOL_ENVIRONMENT_ACTION_DIM,
            "action_projection": PREPEND_ZERO_ARM_ACTION_PROJECTION,
        }
    return {
        "policy_action_dim": PICK_TOOL_ACTION_DIM,
        "policy_action_layout": FULL_POLICY_ACTION_LAYOUT,
        "environment_action_dim": PICK_TOOL_ENVIRONMENT_ACTION_DIM,
        "action_projection": IDENTITY_ACTION_PROJECTION,
    }


def action_noise_group_specs(task_mode: str) -> tuple[tuple[Any, ...], ...]:
    """Return grouped exploration slices in policy-action coordinates."""

    if task_mode == POWER_CLOSE_OPTION_TASK_MODE:
        return POWER_ACTION_NOISE_GROUP_SPECS
    if task_mode in (FULL_TASK_MODE, CLOSE_OPTION_TASK_MODE):
        return FULL_ACTION_NOISE_GROUP_SPECS
    raise ValueError(f"unsupported task_mode={task_mode!r}")


def observation_contract(task_mode: str) -> dict[str, Any]:
    """Return the actor-visible observation semantics fixed by a task mode."""

    if task_mode not in TASK_MODES:
        raise ValueError(f"unsupported task_mode={task_mode!r}")
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
    if not path.exists():
        if path.is_symlink():
            raise ValueError(f"checkpoint task contract is a dangling symlink: {path}")
        # A missing contract is only backward-compatible evidence when the
        # directory is demonstrably a complete pre-contract network snapshot.
        return {
            "version": 0,
            "task_mode": FULL_TASK_MODE,
            "legacy_checkpoint": True,
            "replay_n_step": None,
            "replay_gamma": None,
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
        or version not in (1, 2, TASK_CONTRACT_VERSION)
    ):
        raise ValueError(
            f"checkpoint task contract {path} has version={version!r}, "
            f"expected 1, 2 or {TASK_CONTRACT_VERSION}"
        )
    task_mode = payload.get("task_mode")
    if task_mode not in TASK_MODES:
        raise ValueError(
            f"checkpoint task contract {path} has unsupported task_mode={task_mode!r}"
        )
    if version < 3 and task_mode == POWER_CLOSE_OPTION_TASK_MODE:
        raise ValueError(
            f"checkpoint task contract {path} cannot use task_mode={task_mode!r} "
            f"before version 3"
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
    return {
        "version": int(version),
        "task_mode": str(task_mode),
        "legacy_checkpoint": False,
        "replay_n_step": replay_n_step,
        "replay_gamma": replay_gamma,
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
    for key, expected in expected_runtime.items():
        if resolved_runtime[key] != expected:
            raise ValueError(
                f"{key}={resolved_runtime[key]!r} is incompatible with "
                f"task_mode={task_mode!r}; expected {expected!r}"
            )
    atomic_write_json(
        checkpoint / TASK_CONTRACT_FILENAME,
        {
            "version": TASK_CONTRACT_VERSION,
            "task_mode": task_mode,
            "replay_n_step": replay_n_step,
            "replay_gamma": replay_gamma,
            **resolved_runtime,
        },
    )


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
) -> None:
    """Save one final checkpoint, publishing its task contract last."""

    if checkpoint.is_symlink():
        raise ValueError(f"checkpoint output must be a real directory: {checkpoint}")
    checkpoint.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint output must be a real directory: {checkpoint}")
    incomplete = checkpoint / INCOMPLETE_CHECKPOINT_FILENAME
    atomic_write_json(incomplete, {"version": 1, "status": "incomplete"})
    clear_stale_checkpoint_optional_artifacts(checkpoint)
    agent.save(str(checkpoint))
    if save_replay:
        agent.save_replay_buffer(str(checkpoint))
    if actor_rehearsal is not None:
        actor_rehearsal.save(checkpoint / "actor_rehearsal.pt")
    write_checkpoint_task_contract(
        checkpoint,
        task_mode=task_mode,
        replay_n_step=replay_n_step,
        replay_gamma=replay_gamma,
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
) -> None:
    """Validate mutually exclusive initialization and task-specific data sources."""

    if checkpoint is not None and actor_checkpoint is not None:
        raise ValueError("--checkpoint and --actor_checkpoint are mutually exclusive")
    if resume_replay and checkpoint is None:
        raise ValueError("--resume_replay requires --checkpoint")
    if resume_actor_demo and checkpoint is None:
        raise ValueError("--resume_actor_demo requires --checkpoint")
    task_mode_from_close_option(close_option_mode, power_close_option_mode)
    if (close_option_mode or power_close_option_mode) and demo is not None:
        raise ValueError(
            "close-option training rejects full-task transition --demo data; "
            "use allowlisted --actor_demo supervision instead"
        )
    if power_close_option_mode and actor_demo is not None:
        raise ValueError(
            "--power_close_option_mode rejects --actor_demo until a strict, "
            "power-authority hand14 dataset contract is implemented"
        )


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
) -> None:
    """Keep the close option on its physically meaningful pregrasp MDP."""

    task_mode_from_close_option(close_option_mode, power_close_option_mode)
    if not (close_option_mode or power_close_option_mode):
        return
    flag = (
        "--power_close_option_mode"
        if power_close_option_mode
        else "--close_option_mode"
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
    minimum_episode_length_s = 0.40
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


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"metric tensor must be scalar, got {tuple(value.shape)}")
        value = value.detach().item()
    elif isinstance(value, np.generic):
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"metric is not finite: {result}")
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
            "rehearsal; they are never inserted into critic replay."
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
    parser.add_argument("--output_dir", type=Path, default=Path("logs/flashsac/pick_tool"))
    parser.add_argument("--metrics_every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny 8-env, 8-step integration run.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    return args, launcher


def _validate_args(args: argparse.Namespace) -> None:
    task_mode = task_mode_from_close_option(
        args.close_option_mode,
        args.power_close_option_mode,
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
    if task_mode == POWER_CLOSE_OPTION_TASK_MODE:
        if args.unlatched_arm_noise_scale != 0.0 or args.latched_arm_noise_scale != 0.0:
            raise ValueError(
                "--power_close_option_mode has no policy arm actions; both arm noise "
                "scales must be exactly 0"
            )
        if args.demo_bc_arm_weight != 0.0:
            raise ValueError(
                "--power_close_option_mode has no policy arm actions; "
                "--demo_bc_arm_weight must be 0"
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


def _strict_metrics(info: Mapping[str, Any]) -> dict[str, float]:
    values = info.get("strict_metrics", {})
    if not isinstance(values, Mapping):
        raise TypeError("adapter info['strict_metrics'] must be a mapping")
    return {str(name): _scalar(value) for name, value in values.items()}


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
    target_task_mode = (
        str(source_contract["task_mode"])
        if target_task_mode is None
        else target_task_mode
    )
    target_runtime = runtime_contract(target_task_mode)
    source_dim = int(source_contract["policy_action_dim"])
    target_dim = int(target_runtime["policy_action_dim"])
    actor_projection: str | None = None
    if source_dim != target_dim:
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
        "source_task_mode": str(source_contract["task_mode"]),
        "source_policy_action_dim": source_dim,
        "source_policy_action_layout": str(source_contract["policy_action_layout"]),
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
    if projection is not None:
        raise ValueError(f"unsupported actor projection={projection!r}")
    if source_dim != target_dim:
        raise ValueError(
            "identity actor load requires equal policy action dimensions: "
            f"source={source_dim!r}, target={target_dim!r}"
        )
    agent.load_actor(audit["path"])


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

    physical_close_option_mode = bool(
        args.close_option_mode or args.power_close_option_mode
    )
    if args.smoke:
        args.steps = min(args.steps, 8)
        args.num_envs = min(args.num_envs, 8)
        args.buffer = min(args.buffer, 128)
        args.batch = min(args.batch, 16)
        args.metrics_every = 1
        if args.episode_length_s is None:
            args.episode_length_s = 0.5 if physical_close_option_mode else 0.12
    if physical_close_option_mode and args.episode_length_s is None:
        args.episode_length_s = 5.0
    _validate_args(args)
    _seed_everything(args.seed)
    task_mode = task_mode_from_close_option(
        args.close_option_mode,
        args.power_close_option_mode,
    )
    current_runtime_contract = runtime_contract(task_mode)
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
        )
        if args.resume_replay:
            # Keep the replay-specific validation as an explicit second guard:
            # it documents why replay continuation is legal in this run.
            validate_replay_task_contract(
                args.checkpoint.resolve(),
                task_mode=task_mode,
                n_step=effective_n_step,
                gamma=FLASH_SAC_GAMMA,
            )
    actor_checkpoint_audit = (
        audit_actor_checkpoint_source(
            args.actor_checkpoint,
            output_checkpoint=checkpoint_dir,
            target_task_mode=task_mode,
        )
        if args.actor_checkpoint is not None
        else None
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
        if task_mode == POWER_CLOSE_OPTION_TASK_MODE
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
        ActionNoiseGroup,
        FlashSACTorchBridge,
        build_agent_config,
    )
    from actor_rehearsal import (
        PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
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

    cfg_overrides = {
        "close_option_mode": physical_close_option_mode,
        "power_close_option_mode": bool(args.power_close_option_mode),
    }
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
        validate_finite=args.smoke or args.validate_finite,
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
    )

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
        actor_demo_contracts = (
            PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS
            if is_close_option_task_mode(task_mode)
            else PICK_TOOL_LIFT_ACTOR_DEMO_CONTRACTS
        )
        for path in args.actor_demo:
            batch, labels, audit = load_actor_rehearsal(
                path.resolve(),
                device=device,
                observation_dim=env.observation_dim,
                action_dim=env.action_dim,
                allowed_contracts=actor_demo_contracts,
            )
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
        )
        if revalidated_contract != resumed_task_contract:
            raise RuntimeError("--checkpoint task contract changed while the run was starting")
        agent.load(str(args.checkpoint.resolve()))
        post_load_contract = validate_checkpoint_task_contract(
            args.checkpoint.resolve(),
            task_mode=task_mode,
            n_step=effective_n_step,
            gamma=FLASH_SAC_GAMMA,
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
        )
        if revalidated_actor != actor_checkpoint_audit:
            raise RuntimeError("--actor_checkpoint changed while the run was starting")
        load_audited_actor_checkpoint(agent, actor_checkpoint_audit)
        loaded_actor_sha256 = _sha256(
            Path(actor_checkpoint_audit["path"]) / "actor.pt"
        )
        if loaded_actor_sha256 != actor_checkpoint_audit["actor_sha256"]:
            raise RuntimeError("--actor_checkpoint changed while actor.pt was being loaded")
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
    update_sums: dict[str, float] = {}
    update_metric_counts: dict[str, int] = {}
    actor_update_count = 0
    demo_bc_update_count = 0
    instant_strict: dict[str, float] = {}
    run_max_strict: dict[str, float] = {}
    started = time.perf_counter()

    try:
        for interaction_step in range(1, args.steps + 1):
            training_ready = agent.can_start_training()
            if (
                args.checkpoint is not None
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

            next_observation, reward, terminated, truncated, info = env.step(action)
            transition = build_replay_transition(
                observation,
                action,
                reward,
                terminated,
                truncated,
                info,
                action_dim=env.action_dim,
            )
            agent.process_transition(transition)

            done = terminated | truncated
            episodes.step(reward, done)
            terminal_events.step(info)
            instant_strict = _strict_metrics(info)
            for name, value in instant_strict.items():
                run_max_strict[name] = max(run_max_strict.get(name, -math.inf), value)
            terminated_count.add_(terminated.sum())
            truncated_count.add_(truncated.sum())

            # Rollout continues from reset observations.  Replay has already
            # cloned the captured terminal observations in ``transition``.
            observation = next_observation
            agent.reset_exploration(env_ids=done.nonzero(as_tuple=False).squeeze(-1))

            for _ in range(update_budget.grant(agent.can_start_training())):
                update_info = agent.update(
                    actor_enabled=update_count >= args.critic_burnin_updates
                )
                update_count += 1
                if "actor/loss" in update_info:
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
                    update_sums[name] = update_sums.get(name, 0.0) + _scalar(value)
                    update_metric_counts[name] = update_metric_counts.get(name, 0) + 1

            if interaction_step % args.metrics_every == 0 or interaction_step == args.steps:
                elapsed = max(time.perf_counter() - started, 1.0e-9)
                metrics: dict[str, Any] = {
                    "seed": args.seed,
                    "smoke": bool(args.smoke),
                    "task_mode": task_mode,
                    "interaction_step": interaction_step,
                    "environment_steps": interaction_step * env.num_envs,
                    "gradient_updates": update_count,
                    "actor_updates": actor_update_count,
                    "critic_burnin_updates": args.critic_burnin_updates,
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
                    "initial_actor_checkpoint_projection": (
                        actor_checkpoint_audit["actor_projection"]
                        if actor_checkpoint_audit
                        else None
                    ),
                    "resumed_replay": bool(args.resume_replay),
                    "resumed_task_contract": resumed_task_contract,
                    "resumed_actor_demo": bool(args.resume_actor_demo),
                    "restore_checkpoint_rng": False,
                    "flashsac_upstream_commit": FLASH_SAC_COMMIT,
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

        save_final_checkpoint(
            checkpoint_dir,
            agent=agent,
            task_mode=task_mode,
            replay_n_step=agent_cfg.n_step,
            replay_gamma=agent_cfg.gamma,
            save_replay=bool(args.save_replay),
            actor_rehearsal=actor_rehearsal,
        )
        final_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        final_metrics["checkpoint"] = str(checkpoint_dir)
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
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
