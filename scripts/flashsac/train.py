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
    parser.add_argument("--output_dir", type=Path, default=Path("logs/flashsac/pick_tool"))
    parser.add_argument("--metrics_every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny 8-env, 8-step integration run.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    return args, launcher


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
    started = time.perf_counter()

    try:
        for interaction_step in range(1, args.steps + 1):
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
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
