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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_step", type=int, default=3, help="Replay return horizon for non-smoke runs.")
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
    parser.add_argument("--output_dir", type=Path, default=Path("logs/flashsac/pick_tool"))
    parser.add_argument("--metrics_every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny 8-env, 8-step integration run.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    return args, launcher


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("steps", "num_envs", "buffer", "batch", "metrics_every", "n_step"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
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
    if args.episode_length_s is not None and (
        not math.isfinite(args.episode_length_s) or args.episode_length_s <= 0.0
    ):
        raise ValueError("--episode_length_s must be finite and positive")


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
    return {f"strict/{name}": _scalar(value) for name, value in values.items()}


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

    # These imports require the simulator process (and, for the task, its USD
    # plugins) to be initialized by AppLauncher first.
    from adapter import build_replay_transition, make_pick_tool_env
    from agent_bridge import (
        FLASH_SAC_COMMIT,
        ActionNoiseGroup,
        FlashSACTorchBridge,
        build_agent_config,
    )

    device = str(args.device or "cuda:0")
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"FlashSAC PickTool training requires CUDA, got {device}")

    cfg_overrides = {}
    if args.episode_length_s is not None:
        cfg_overrides["episode_length_s"] = args.episode_length_s
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
        actor_num_blocks=1 if args.smoke else 2,
        actor_hidden_dim=32 if args.smoke else 128,
        critic_num_blocks=1 if args.smoke else 2,
        critic_hidden_dim=64 if args.smoke else 256,
        learning_rate_warmup_step=max(1, planned_updates // 20),
        learning_rate_decay_step=planned_updates,
        use_compile=not args.smoke,
        compile_mode="default" if args.smoke else "reduce-overhead",
        use_amp=not args.smoke,
        load_optimizer=args.checkpoint is not None,
        load_reward_normalizer=args.checkpoint is not None,
    )
    noise_groups = (
        ActionNoiseGroup("arm", 0, 7, scale=1.0, zeta_max=32),
        ActionNoiseGroup("token", 7, 16, scale=0.5, zeta_max=16),
        ActionNoiseGroup("residual", 16, 21, scale=0.35, zeta_max=8),
    )
    agent = FlashSACTorchBridge(
        env.observation_space,
        env.action_space,
        env.env_info,
        agent_cfg,
        noise_groups=noise_groups,
    )
    if args.checkpoint is not None:
        agent.load(str(args.checkpoint.resolve()))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    observation, _ = env.reset(randomize_episode_lengths=args.randomize_episode_lengths)
    agent.reset_exploration(batch_size=env.num_envs)
    episodes = EpisodeAccumulator(env.num_envs, env.device)
    terminal_events = TerminalEventAccumulator(env.num_envs, env.device)
    update_budget = FractionalUpdateBudget(args.updates)
    update_count = 0
    terminated_count = torch.zeros((), dtype=torch.long, device=env.device)
    truncated_count = torch.zeros((), dtype=torch.long, device=env.device)
    update_sums: dict[str, float] = {}
    update_metric_count = 0
    started = time.perf_counter()

    try:
        for interaction_step in range(1, args.steps + 1):
            if agent.can_start_training():
                action = agent.sample_actions(
                    interaction_step,
                    {"next_observation": observation},
                    training=True,
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
            terminated_count.add_(terminated.sum())
            truncated_count.add_(truncated.sum())

            # Rollout continues from reset observations.  Replay has already
            # cloned the captured terminal observations in ``transition``.
            observation = next_observation
            agent.reset_exploration(env_ids=done.nonzero(as_tuple=False).squeeze(-1))

            for _ in range(update_budget.grant(agent.can_start_training())):
                update_info = agent.update()
                update_count += 1
                update_metric_count += 1
                for name, value in update_info.items():
                    update_sums[name] = update_sums.get(name, 0.0) + _scalar(value)

            if interaction_step % args.metrics_every == 0 or interaction_step == args.steps:
                elapsed = max(time.perf_counter() - started, 1.0e-9)
                metrics: dict[str, Any] = {
                    "seed": args.seed,
                    "smoke": bool(args.smoke),
                    "interaction_step": interaction_step,
                    "environment_steps": interaction_step * env.num_envs,
                    "gradient_updates": update_count,
                    "flashsac_upstream_commit": FLASH_SAC_COMMIT,
                    "observation_dim": env.observation_dim,
                    "action_dim": env.action_dim,
                    "buffer_capacity": args.buffer,
                    "warmup_transitions": warmup_transitions,
                    "replay_transitions": agent.replay_size,
                    "n_step": agent_cfg.n_step,
                    "terminated_events": int(terminated_count.item()),
                    "truncated_events": int(truncated_count.item()),
                    "throughput_env_steps_per_second": interaction_step * env.num_envs / elapsed,
                    **episodes.metrics(),
                    **terminal_events.metrics(),
                    **_strict_metrics(info),
                }
                if update_metric_count:
                    metrics.update(
                        {
                            f"update/{name}": total / update_metric_count
                            for name, total in update_sums.items()
                        }
                    )
                atomic_write_json(metrics_path, metrics)

        checkpoint_dir = output_dir / "checkpoint_final"
        agent.save(str(checkpoint_dir))
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
