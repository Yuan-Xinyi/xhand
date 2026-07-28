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


@dataclass
class TerminalEventAccumulator:
    """Accumulate reset-before-clone task truth without per-step host sync."""

    num_envs: int
    device: torch.device
    counts: dict[str, torch.Tensor] = field(init=False)
    _unlatched_seen: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.counts = {
            name: torch.zeros((), dtype=torch.long, device=self.device)
            for name in TERMINAL_EVENT_KEYS
        }
        self._unlatched_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def step(self, info: Mapping[str, Any]) -> None:
        values = info.get("pick_tool_terminal")
        if not isinstance(values, Mapping):
            raise KeyError("adapter info has no pick_tool_terminal ground truth")
        validated: dict[str, torch.Tensor] = {}
        for name in TERMINAL_EVENT_KEYS:
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

        # Terminal flags are one-step events.  Unlatched 5 cm is a state and
        # can persist, so count its first rising occurrence once per episode.
        for name in TERMINAL_EVENT_KEYS[:-1]:
            self.counts[name].add_(validated[name].sum())
        unlatched = validated["unlatched_clearance_ge_5cm"]
        self.counts["unlatched_clearance_ge_5cm"].add_((unlatched & ~self._unlatched_seen).sum())
        self._unlatched_seen |= unlatched
        episode_done = validated["success"] | validated["failure"] | validated["time_out"]
        self._unlatched_seen &= ~episode_done

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
        "--demo_bc_weight",
        type=float,
        default=None,
        help="Demo-only actor rehearsal weight (default: 1 with --demo, otherwise 0).",
    )
    parser.add_argument("--demo_bc_target_std", type=float, default=0.15)
    parser.add_argument("--demo_bc_std_weight", type=float, default=0.05)
    parser.add_argument(
        "--demo_bc_batch",
        type=int,
        default=None,
        help="Demo-only rehearsal rows (default: the fixed demo rows per mixed batch).",
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
    parser.add_argument(
        "--close_option",
        action="store_true",
        help="Train ONLY the stable-latch (close) phase: episodes terminate on the env's "
        "close-option success/failure contract and pay its bonus/penalties.  The hand must "
        "spawn at a pregrasp boundary, so this requires --curriculum_dataset with "
        "--curriculum_probability 1.0 (e.g. boundary close_start).",
    )
    parser.add_argument(
        "--nudge_option",
        action="store_true",
        help="Train ONLY the non-prehensile pre-grasp reorientation (nudge) phase: from a "
        "normal randomized reset, push the tool on the table back into the graspable pose "
        "family.  Episodes terminate on the env's nudge success/failure contract.  No "
        "curriculum spawn is needed.",
    )
    parser.add_argument(
        "--nudge_yaw_range",
        type=float,
        default=None,
        help="Curriculum knob for --nudge_option: reset yaw is sampled from [-r, r] radians "
        "instead of the full [-pi, pi].  Start e.g. at 1.57 and widen in later runs.",
    )
    parser.add_argument(
        "--nudge_pregrasp",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "OCCUPANCY"),
        help="v7 grasp-ready ending for --nudge_option: success additionally requires the "
        "pregrasp readiness score >= MIN (0.30 = the oracle close_start capture gate) held "
        "through the confirm window; OCCUPANCY pays the score per step while the tool is in "
        "the pose family.  E.g. '0.30 0.3'.",
    )
    parser.add_argument(
        "--nudge_grasp_option",
        action="store_true",
        help="Merged nudge+grasp phase: from the home spawn, reorient (guided, not required) "
        "and END HOLDING the tool -- success is the strict latch contract sustained "
        "close_option_confirm_steps.  Spawn is fixed at home unless --nudge_spawn_anneal.",
    )
    parser.add_argument(
        "--nudge_pregrasp_adaptive",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "TARGET"),
        help="Self-paced contract ratchet for --nudge_option: nudge_pregrasp_min starts at "
        "START and moves toward TARGET driven by the recent success rate (raise a notch at "
        ">=50%%, back off half a notch under 25%%), so the success income is never starved "
        "by a miscalibrated schedule.  Overrides --nudge_pregrasp_anneal.  E.g. '0.005 0.30'.",
    )
    parser.add_argument(
        "--nudge_pregrasp_anneal",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Contract-space anneal for --nudge_option: nudge_pregrasp_min ramps linearly "
        "from START to END over the run.  The policy's existing +100 success income is "
        "ratcheted -- it must gradually END episodes grasp-ready to keep collecting, the "
        "spawn-anneal trick applied to the success contract instead of the spawn.  "
        "E.g. '0.02 0.30'.  Requires --nudge_pregrasp for the occupancy weight.",
    )
    parser.add_argument(
        "--nudge_staging",
        type=float,
        default=None,
        help="Staging-zone occupancy for --nudge_option (reposition training): pays for "
        "hovering the palm above the posed tool while in the pose family.",
    )
    parser.add_argument(
        "--nudge_grasp_staging",
        type=float,
        default=None,
        help="Staging-zone occupancy weight for --nudge_grasp_option (e.g. 0.08): pays for "
        "hovering the palm above the posed tool -- the ladder rung between the post-nudge "
        "push posture and the pregrasp/close/latch stack.",
    )
    parser.add_argument(
        "--inhand_option",
        action="store_true",
        help="In-hand reorientation stage: from curriculum-spawned lifted holds, bring the "
        "INDEX fingertip onto the tool's functional point while keeping the grasp.  "
        "Requires --curriculum_dataset (boundary inhand_start).",
    )
    parser.add_argument(
        "--inhand_dist_adaptive",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "TARGET"),
        help="Self-paced ratchet on inhand_dist_threshold: start loose (e.g. 0.08) and "
        "tighten toward TARGET (e.g. 0.015) driven by the recent success rate, the v9 "
        "contract-ratchet recipe on the success distance.",
    )
    parser.add_argument(
        "--carry_option",
        action="store_true",
        help="Carry stage (SimToolReal-style): from curriculum-spawned fresh-latch states, "
        "move the held tool to consecutive random pose goals (goal in the observation); "
        "each reach pays a bonus and resamples in place; drop terminates.  Requires "
        "--curriculum_dataset (boundary carry_start).",
    )
    parser.add_argument(
        "--carry_pos_adaptive",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "TARGET"),
        help="Ratchet on carry_pos_tolerance (m), driven by window goals-per-episode.",
    )
    parser.add_argument(
        "--carry_rot_adaptive",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "TARGET"),
        help="Ratchet on carry_rot_tolerance (rad), driven by window goals-per-episode.",
    )
    parser.add_argument(
        "--carry_gap_adaptive",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "TARGET"),
        help="Relative-orientation goal curriculum: goal = current orientation rotated by "
        "theta ~ U(0, max); max ratchets START -> TARGET (rad) past the wrist range so "
        "in-hand repositioning becomes necessary.  Overrides absolute rpy goal sampling.",
    )
    parser.add_argument(
        "--carry_lock_arm",
        action="store_true",
        help="Arm-locked in-hand sub-task: zero arm channels; goal positions track the "
        "object; finger gaiting is the only path to orientation goals.",
    )
    parser.add_argument(
        "--carry_timeout_cost",
        type=float,
        default=None,
        help="carry goal-timeout charge override; ~goal bonus makes skipping a hard goal "
        "cancel out a reach instead of being a cheap escape valve.",
    )
    parser.add_argument(
        "--carry_arm_cost",
        type=float,
        default=None,
        help="Arm-expensive motion cost weight (carry_arm_motion_penalty); hand motion free.",
    )
    parser.add_argument(
        "--carry_rot_range",
        type=float,
        default=0.6,
        metavar="RAD",
        help="Goal orientation sampling: roll/pitch ranges become [-RAD, RAD] (yaw stays "
        "full circle).  Full +-pi roll/pitch goals are unreachable for a held hammer.",
    )
    parser.add_argument(
        "--inhand_head_adaptive",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "TARGET"),
        help="Self-paced ratchet on inhand_head_cos_min (head-down attitude): start loose "
        "(e.g. -1.0 = off) and tighten toward TARGET cos (e.g. 0.90 ~= 26 deg) driven by "
        "the recent success rate.",
    )
    parser.add_argument(
        "--nudge_spawn_anneal",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Demo-free spawn curriculum: anneal nudge_spawn_blend_min linearly from START to "
        "END over the run (blend 1 = low-ready pose, 0 = home).  blend_max stays 1.0, so the "
        "spawn mixture always keeps easy near-object starts.  E.g. '1.0 0.0'.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("logs/flashsac/pick_tool"))
    parser.add_argument("--metrics_every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny 8-env, 8-step integration run.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    return args, launcher


# Self-paced contract ratchet cadence: check the recent success window every N interaction
# steps and move the pregrasp gate one notch at most.  ~2 episodes' worth of steps keeps the
# window estimate meaningful at 1024 envs without reacting to single-batch noise.
ADAPTIVE_GATE_WINDOW = 500
ADAPTIVE_GATE_STEP = 0.005


def _validate_args(args: argparse.Namespace) -> None:
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
    if args.close_option:
        # The close-option contract judges failure by pregrasp-window quantities (proximity,
        # horizontal drift from the spawn pose); from a normal far-away reset those conditions
        # fire immediately and every episode is an instant failure.  Demand the pregrasp spawn.
        if args.curriculum_dataset is None or args.curriculum_probability < 1.0:
            raise ValueError(
                "--close_option requires --curriculum_dataset and --curriculum_probability 1.0 "
                "(every episode must spawn at a pregrasp boundary)"
            )
    if args.inhand_option:
        if args.curriculum_dataset is None:
            raise ValueError("--inhand_option requires --curriculum_dataset (inhand_start)")
    if args.inhand_dist_adaptive is not None:
        if not args.inhand_option:
            raise ValueError("--inhand_dist_adaptive only applies with --inhand_option")
        start, target = args.inhand_dist_adaptive
        if not (0.0 < target <= start):
            raise ValueError("--inhand_dist_adaptive needs START >= TARGET > 0")
    if args.inhand_head_adaptive is not None:
        if not args.inhand_option:
            raise ValueError("--inhand_head_adaptive only applies with --inhand_option")
        start, target = args.inhand_head_adaptive
        if not (-1.0 <= start <= target <= 1.0):
            raise ValueError("--inhand_head_adaptive needs -1 <= START <= TARGET <= 1")
    if args.carry_option:
        if args.curriculum_dataset is None:
            raise ValueError("--carry_option requires --curriculum_dataset (carry_start)")
    if args.carry_gap_adaptive is not None:
        if not args.carry_option:
            raise ValueError("--carry_gap_adaptive only applies with --carry_option")
        start, target = args.carry_gap_adaptive
        if not (0.0 < start <= target):
            raise ValueError("--carry_gap_adaptive needs 0 < START <= TARGET (it grows)")
    for name in ("carry_pos_adaptive", "carry_rot_adaptive"):
        pair = getattr(args, name)
        if pair is not None:
            if not args.carry_option:
                raise ValueError(f"--{name} only applies with --carry_option")
            start, target = pair
            if not (0.0 < target <= start):
                raise ValueError(f"--{name} needs START >= TARGET > 0")
    if sum((args.nudge_option, args.close_option, args.nudge_grasp_option, args.inhand_option, args.carry_option)) > 1:
        raise ValueError(
            "--nudge_option, --close_option and --nudge_grasp_option are mutually exclusive"
        )
    if args.nudge_staging is not None:
        if not args.nudge_option:
            raise ValueError("--nudge_staging only applies with --nudge_option")
        if args.nudge_staging < 0.0:
            raise ValueError("--nudge_staging must be >= 0")
    if args.nudge_grasp_staging is not None:
        if not args.nudge_grasp_option:
            raise ValueError("--nudge_grasp_staging only applies with --nudge_grasp_option")
        if args.nudge_grasp_staging < 0.0:
            raise ValueError("--nudge_grasp_staging must be >= 0")
    if args.nudge_pregrasp is not None:
        if not args.nudge_option:
            raise ValueError("--nudge_pregrasp only applies with --nudge_option")
        minimum, occupancy = args.nudge_pregrasp
        if not 0.0 < minimum <= 1.0 or occupancy < 0.0:
            raise ValueError("--nudge_pregrasp MIN must be in (0, 1] and OCCUPANCY >= 0")
    if args.nudge_pregrasp_anneal is not None:
        if args.nudge_pregrasp is None:
            raise ValueError("--nudge_pregrasp_anneal requires --nudge_pregrasp")
        start, end = args.nudge_pregrasp_anneal
        if not (0.0 < start <= 1.0 and 0.0 < end <= 1.0):
            raise ValueError("--nudge_pregrasp_anneal bounds must be in (0, 1]")
    if args.nudge_pregrasp_adaptive is not None:
        if args.nudge_pregrasp is None:
            raise ValueError("--nudge_pregrasp_adaptive requires --nudge_pregrasp")
        start, target = args.nudge_pregrasp_adaptive
        if not (0.0 < start <= target <= 1.0):
            raise ValueError("--nudge_pregrasp_adaptive needs 0 < START <= TARGET <= 1")
    if args.nudge_yaw_range is not None:
        if not (args.nudge_option or args.nudge_grasp_option):
            raise ValueError(
                "--nudge_yaw_range only applies with --nudge_option/--nudge_grasp_option"
            )
        if not math.isfinite(args.nudge_yaw_range) or not 0.0 < args.nudge_yaw_range <= math.pi:
            raise ValueError("--nudge_yaw_range must be in (0, pi]")
    if args.nudge_spawn_anneal is not None:
        if not (args.nudge_option or args.nudge_grasp_option):
            raise ValueError(
                "--nudge_spawn_anneal only applies with --nudge_option/--nudge_grasp_option"
            )
        start, end = args.nudge_spawn_anneal
        for name, value in (("START", start), ("END", end)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"--nudge_spawn_anneal {name} must be in [0, 1]")
    if args.resume_replay and args.checkpoint is None:
        raise ValueError("--resume_replay requires --checkpoint")
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
        if args.demo_bc_weight is not None and (
            not math.isfinite(args.demo_bc_weight) or args.demo_bc_weight < 0.0
        ):
            raise ValueError("--demo_bc_weight must be finite and non-negative")
        if not math.isfinite(args.demo_bc_target_std) or args.demo_bc_target_std <= 0.0:
            raise ValueError("--demo_bc_target_std must be finite and positive")
        if not math.isfinite(args.demo_bc_std_weight) or args.demo_bc_std_weight < 0.0:
            raise ValueError("--demo_bc_std_weight must be finite and non-negative")
        if args.demo_bc_batch is not None and args.demo_bc_batch < 1:
            raise ValueError("--demo_bc_batch must be positive")
    elif args.demo_bc_weight not in (None, 0.0):
        raise ValueError("--demo_bc_weight must be 0 when --demo is absent")
    for name in ("latched_arm_noise_scale", "latched_hand_noise_scale"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name} must be finite and non-negative")


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

    if args.smoke:
        args.steps = min(args.steps, 8)
        args.num_envs = min(args.num_envs, 8)
        args.buffer = min(args.buffer, 128)
        args.batch = min(args.batch, 16)
        args.metrics_every = 1
        if args.episode_length_s is None:
            args.episode_length_s = 0.12
    _validate_args(args)
    _seed_everything(args.seed)
    demo_bc_weight = (
        1.0
        if args.demo is not None and args.demo_bc_weight is None
        else float(args.demo_bc_weight or 0.0)
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
    curriculum_metrics["close_option"] = bool(args.close_option)
    curriculum_metrics["nudge_option"] = bool(args.nudge_option)
    curriculum_metrics["nudge_grasp_option"] = bool(args.nudge_grasp_option)
    curriculum_metrics["nudge_yaw_range"] = args.nudge_yaw_range
    curriculum_metrics["nudge_spawn_anneal"] = (
        list(args.nudge_spawn_anneal) if args.nudge_spawn_anneal is not None else None
    )

    cfg_overrides = {}
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
    if args.close_option:
        # Close-phase-only training: the env terminates each episode on its close-option
        # success/failure contract and pays the corresponding bonus/penalties on top of the
        # dense close/wrap shaping.  All thresholds keep their cfg defaults (the same values
        # the PPO close-option runs used).
        cfg_overrides["close_option_mode"] = True
        if args.episode_length_s is None:
            cfg_overrides["episode_length_s"] = 3.0
    if args.nudge_option:
        # Nudge-phase-only training: normal randomized resets; the env pays potential-based
        # reach/pose shaping plus the nudge success/failure/timeout contract.  Pushing needs
        # more time than closing, so default to a 6s horizon.
        cfg_overrides["nudge_option_mode"] = True
        if args.episode_length_s is None:
            cfg_overrides["episode_length_s"] = 6.0
        if args.nudge_pregrasp is not None:
            cfg_overrides["nudge_pregrasp_min"] = args.nudge_pregrasp[0]
            cfg_overrides["nudge_pregrasp_occupancy"] = args.nudge_pregrasp[1]
        if args.nudge_pregrasp_adaptive is not None:
            cfg_overrides["nudge_pregrasp_min"] = args.nudge_pregrasp_adaptive[0]
        if args.nudge_staging is not None:
            cfg_overrides["nudge_staging_occupancy"] = args.nudge_staging
        if args.nudge_yaw_range is not None:
            cfg_overrides["reset_object_yaw_range"] = (
                -args.nudge_yaw_range,
                args.nudge_yaw_range,
            )
    if args.nudge_grasp_option:
        # Merged nudge+grasp: reorient AND hold the latch.  Longer horizon than nudge alone;
        # spawn fixed at home (the v6 regime) unless the anneal knob is driving blend_min.
        cfg_overrides["nudge_grasp_mode"] = True
        if args.nudge_grasp_staging is not None:
            cfg_overrides["nudge_grasp_staging_occupancy"] = args.nudge_grasp_staging
        if args.episode_length_s is None:
            cfg_overrides["episode_length_s"] = 10.0
        if args.nudge_spawn_anneal is None:
            cfg_overrides["nudge_spawn_blend_min"] = 0.0
            cfg_overrides["nudge_spawn_blend_max"] = 0.0
        if args.nudge_yaw_range is not None:
            cfg_overrides["reset_object_yaw_range"] = (
                -args.nudge_yaw_range,
                args.nudge_yaw_range,
            )
    if args.inhand_option:
        # In-hand reorientation: lifted-hold curriculum spawns; drop guards active in mode
        # logic itself (terminate_on_drop's below-table check never fires from a lifted hold).
        cfg_overrides["inhand_mode"] = True
        if args.inhand_dist_adaptive is not None:
            cfg_overrides["inhand_dist_threshold"] = args.inhand_dist_adaptive[0]
        if args.inhand_head_adaptive is not None:
            cfg_overrides["inhand_head_cos_min"] = args.inhand_head_adaptive[0]
        if args.episode_length_s is None:
            cfg_overrides["episode_length_s"] = 8.0
    if args.carry_option:
        cfg_overrides["carry_mode"] = True
        cfg_overrides["target_rot_range_roll"] = (-args.carry_rot_range, args.carry_rot_range)
        cfg_overrides["target_rot_range_pitch"] = (-args.carry_rot_range, args.carry_rot_range)
        if args.carry_pos_adaptive is not None:
            cfg_overrides["carry_pos_tolerance"] = args.carry_pos_adaptive[0]
        if args.carry_rot_adaptive is not None:
            cfg_overrides["carry_rot_tolerance"] = args.carry_rot_adaptive[0]
        if args.carry_gap_adaptive is not None:
            cfg_overrides["carry_goal_rel_angle_max"] = args.carry_gap_adaptive[0]
        if args.carry_arm_cost is not None:
            cfg_overrides["carry_arm_motion_penalty"] = args.carry_arm_cost
        if args.carry_timeout_cost is not None:
            cfg_overrides["carry_goal_timeout_penalty"] = args.carry_timeout_cost
        if args.carry_lock_arm:
            cfg_overrides["carry_lock_arm"] = True
            cfg_overrides["carry_goal_pos_range"] = (0.02, 0.02, 0.02)
            cfg_overrides["carry_pos_tolerance"] = 0.08
        if args.episode_length_s is None:
            cfg_overrides["episode_length_s"] = 15.0
    env = make_pick_tool_env(
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        cfg_overrides=cfg_overrides,
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
        n_step=1 if args.smoke else args.n_step,
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
    noise_groups = (
        ActionNoiseGroup("arm", 0, 7, scale=1.0, zeta_mu=1.0, zeta_max=64),
        ActionNoiseGroup("token", 7, 16, scale=0.5, zeta_mu=1.25, zeta_max=32),
        ActionNoiseGroup("residual", 16, 21, scale=0.35, zeta_mu=1.5, zeta_max=16),
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

    if args.checkpoint is not None:
        agent.load(str(args.checkpoint.resolve()))
        if args.resume_replay:
            replay_path = args.checkpoint.resolve() / "replay_buffer.pt"
            if not replay_path.is_file():
                raise FileNotFoundError(replay_path)
            agent.load_replay_buffer(str(args.checkpoint.resolve()))
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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    observation, _ = env.reset(randomize_episode_lengths=args.randomize_episode_lengths)
    agent.start_fresh_rollout(batch_size=env.num_envs)
    episodes = EpisodeAccumulator(env.num_envs, env.device)
    terminal_events = TerminalEventAccumulator(env.num_envs, env.device)
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
    adaptive_prev_succ = 0.0
    adaptive_prev_epis = 0.0
    head_prev_succ = 0.0
    head_prev_epis = 0.0
    carry_prev_goals = 0.0
    carry_prev_epis = 0.0
    adaptive_gate_log: dict[str, float] = {}
    started = time.perf_counter()

    try:
        for interaction_step in range(1, args.steps + 1):
            if (args.nudge_option or args.nudge_grasp_option) and args.nudge_spawn_anneal is not None:
                # Demo-free spawn curriculum: anneal the LOWER edge of the spawn-blend range
                # linearly over the run (1 = low-ready over the table, 0 = home) while the
                # upper edge stays at 1, so easy near-object starts never disappear.  Reset
                # code reads cfg live, so mutating it here affects the next resets.
                start, end = args.nudge_spawn_anneal
                progress = min(1.0, interaction_step / max(1, args.steps))
                env.unwrapped.cfg.nudge_spawn_blend_min = start + (end - start) * progress
            if (
                args.nudge_option
                and args.nudge_pregrasp_anneal is not None
                and args.nudge_pregrasp_adaptive is None
            ):
                # Contract-space ratchet: the termination gate reads cfg live each step, so
                # raising nudge_pregrasp_min here tightens what counts as success while the
                # policy keeps collecting the +100 it already earns -- the spawn-anneal trick
                # applied to the success contract.
                start, end = args.nudge_pregrasp_anneal
                progress = min(1.0, interaction_step / max(1, args.steps))
                env.unwrapped.cfg.nudge_pregrasp_min = start + (end - start) * progress
            training_ready = agent.can_start_training()
            if args.checkpoint is not None or training_ready:
                noise_scale = None
                if (
                    args.latched_arm_noise_scale != 1.0
                    or args.latched_hand_noise_scale != 1.0
                ):
                    # PickTool's public 115-D Markov observation stores the
                    # grasp-latch bit at index 106.  This is not privileged
                    # simulator state: the actor sees the same bit.  Preserve
                    # coherent arm exploration while preventing random hand
                    # reopening from destroying a newly discovered grasp.
                    latched = observation[:, 106] > 0.5
                    noise_scale = torch.ones(
                        (env.num_envs, env.action_dim),
                        dtype=torch.float32,
                        device=env.device,
                    )
                    noise_scale[latched, :7] = args.latched_arm_noise_scale
                    noise_scale[latched, 7:] = args.latched_hand_noise_scale
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
                    if demo_replay is not None and demo_bc_weight > 0.0:
                        rehearsal_batch = demo_replay.sample_demonstrations(args.demo_bc_batch)
                        update_info.update(
                            agent.demo_bc_rehearsal(
                                rehearsal_batch,
                                weight=demo_bc_weight,
                                target_std=args.demo_bc_target_std,
                                std_weight=args.demo_bc_std_weight,
                            )
                        )
                        demo_bc_update_count += 1
                for name, value in update_info.items():
                    update_sums[name] = update_sums.get(name, 0.0) + _scalar(value)
                    update_metric_counts[name] = update_metric_counts.get(name, 0) + 1

            if (
                args.inhand_option
                and args.inhand_dist_adaptive is not None
                and interaction_step % ADAPTIVE_GATE_WINDOW == 0
            ):
                # v9 contract ratchet on the success DISTANCE (shrinking = harder).
                u_env = env.unwrapped
                succ = float(u_env._inhand_success_total)
                epis = float(u_env._inhand_episode_total)
                window_succ = (succ - adaptive_prev_succ) / max(epis - adaptive_prev_epis, 1.0)
                adaptive_prev_succ, adaptive_prev_epis = succ, epis
                dist_start, dist_target = args.inhand_dist_adaptive
                current = float(u_env.cfg.inhand_dist_threshold)
                if window_succ >= 0.5:
                    current = max(dist_target, current - 0.002)
                elif window_succ < 0.25:
                    current = min(dist_start, current + 0.001)
                u_env.cfg.inhand_dist_threshold = current
                adaptive_gate_log = {
                    "inhand_dist_gate": current,
                    "inhand_gate_window_success": window_succ,
                }
            if (
                args.carry_option
                and (args.carry_pos_adaptive is not None or args.carry_rot_adaptive is not None)
                and interaction_step % ADAPTIVE_GATE_WINDOW == 0
            ):
                # Dual ratchet on the goal tolerances, driven by window goals-per-episode.
                u_env = env.unwrapped
                goals = float(u_env._carry_goals_total)
                epis = float(u_env._carry_episode_total)
                window_gpe = (goals - carry_prev_goals) / max(epis - carry_prev_epis, 1.0)
                carry_prev_goals, carry_prev_epis = goals, epis
                gate_log = {"carry_window_goals_per_episode": window_gpe}
                if args.carry_pos_adaptive is not None:
                    start, target = args.carry_pos_adaptive
                    current = float(u_env.cfg.carry_pos_tolerance)
                    if window_gpe >= 1.5:
                        current = max(target, current - 0.002)
                    elif window_gpe < 0.5:
                        current = min(start, current + 0.001)
                    u_env.cfg.carry_pos_tolerance = current
                    gate_log["carry_pos_gate"] = current
                if args.carry_rot_adaptive is not None:
                    start, target = args.carry_rot_adaptive
                    current = float(u_env.cfg.carry_rot_tolerance)
                    if window_gpe >= 1.5:
                        current = max(target, current - 0.01)
                    elif window_gpe < 0.5:
                        current = min(start, current + 0.005)
                    u_env.cfg.carry_rot_tolerance = current
                    gate_log["carry_rot_gate"] = current
                if args.carry_gap_adaptive is not None:
                    start, target = args.carry_gap_adaptive
                    current = float(u_env.cfg.carry_goal_rel_angle_max)
                    if window_gpe >= 1.5:
                        current = min(target, current + 0.02)
                    elif window_gpe < 0.5:
                        current = max(start, current - 0.01)
                    u_env.cfg.carry_goal_rel_angle_max = current
                    gate_log["carry_gap_gate"] = current
                adaptive_gate_log.update(gate_log)
            if (
                args.inhand_option
                and args.inhand_head_adaptive is not None
                and interaction_step % ADAPTIVE_GATE_WINDOW == 0
            ):
                # Head-down attitude ratchet (cos scale), driven by its own success window.
                u_env = env.unwrapped
                succ = float(u_env._inhand_success_total)
                epis = float(u_env._inhand_episode_total)
                window_succ = (succ - head_prev_succ) / max(epis - head_prev_epis, 1.0)
                head_prev_succ, head_prev_epis = succ, epis
                cos_start, cos_target = args.inhand_head_adaptive
                current = float(u_env.cfg.inhand_head_cos_min)
                if window_succ >= 0.5:
                    current = min(cos_target, current + 0.02)
                elif window_succ < 0.25:
                    current = max(cos_start, current - 0.01)
                u_env.cfg.inhand_head_cos_min = current
                adaptive_gate_log.update(
                    {
                        "inhand_head_cos_gate": current,
                        "inhand_head_window_success": window_succ,
                    }
                )
            if (
                args.nudge_option
                and args.nudge_pregrasp_adaptive is not None
                and interaction_step % ADAPTIVE_GATE_WINDOW == 0
            ):
                # Self-paced contract ratchet.  The fixed linear anneal is calibration-fragile
                # (v8: a 0.02 start sat above the policy's natural ~0.01 operating point, the
                # success income vanished at step one and the policy collapsed exactly like the
                # hard-gate v7).  Here the threshold moves only as fast as the policy: raise a
                # notch while the recent success rate stays high, back off half a notch when it
                # craters, clamp to [start, target].  Income can never be starved for long.
                u_env = env.unwrapped
                succ = float(u_env._nudge_success_total)
                epis = float(u_env._nudge_episode_total)
                window_succ = (succ - adaptive_prev_succ) / max(epis - adaptive_prev_epis, 1.0)
                adaptive_prev_succ, adaptive_prev_epis = succ, epis
                gate_start, gate_target = args.nudge_pregrasp_adaptive
                current = float(u_env.cfg.nudge_pregrasp_min)
                if window_succ >= 0.5:
                    current = min(gate_target, current + ADAPTIVE_GATE_STEP)
                elif window_succ < 0.25:
                    current = max(gate_start, current - 0.5 * ADAPTIVE_GATE_STEP)
                u_env.cfg.nudge_pregrasp_min = current
                adaptive_gate_log = {
                    "nudge_pregrasp_gate": current,
                    "nudge_gate_window_success": window_succ,
                }

            if interaction_step % args.metrics_every == 0 or interaction_step == args.steps:
                elapsed = max(time.perf_counter() - started, 1.0e-9)
                metrics: dict[str, Any] = {
                    "seed": args.seed,
                    "smoke": bool(args.smoke),
                    "interaction_step": interaction_step,
                    "environment_steps": interaction_step * env.num_envs,
                    "gradient_updates": update_count,
                    "actor_updates": actor_update_count,
                    "critic_burnin_updates": args.critic_burnin_updates,
                    "demo_bc_updates": demo_bc_update_count,
                    "demo_bc_weight": demo_bc_weight,
                    "lr_decay_updates": lr_decay_updates,
                    "lr_warmup_updates": lr_warmup_updates,
                    "initial_checkpoint": (
                        str(args.checkpoint.resolve()) if args.checkpoint is not None else None
                    ),
                    "resumed_replay": bool(args.resume_replay),
                    "restore_checkpoint_rng": False,
                    "flashsac_upstream_commit": FLASH_SAC_COMMIT,
                    "observation_dim": env.observation_dim,
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
                    "latched_arm_noise_scale": args.latched_arm_noise_scale,
                    "latched_hand_noise_scale": args.latched_hand_noise_scale,
                    **demo_metrics,
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
                metrics.update(adaptive_gate_log)
                atomic_write_json(metrics_path, metrics)

        checkpoint_dir = output_dir / "checkpoint_final"
        agent.save(str(checkpoint_dir))
        if args.save_replay:
            agent.save_replay_buffer(str(checkpoint_dir))
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
    except BaseException:
        # SimulationApp.close() in the finally block hard-exits the process with code 0,
        # swallowing any pending traceback.  Print it before the app can eat it.
        import traceback

        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
