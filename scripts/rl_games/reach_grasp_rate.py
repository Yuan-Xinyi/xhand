# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Measure the grasp success rate of a legacy 16-action/87-observation reach policy.

The question this answers: does a pure-reward PPO reach base actually *close* into a stable
force-closure grasp on its own, or does it only reach a pregrasp (the "hover, never close"
failure)?  We run the policy in the pick-tool env (legacy 16-action/87-observation config,
terminations disabled so a single attempt per env is measured) and count, per env, whether the
environment's own robust-grasp latch is held for a sustained window -- the same latch/quality/force
contract used by the strict success evaluator, minus the lift requirement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Reach-base grasp success-rate probe.")
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--stable_steps", type=int, default=15)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=str, default="/tmp/pick_tool_reach_grasp_rate.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

import xhand_inhand.tasks  # noqa: F401


def _summary(value: torch.Tensor) -> dict[str, float] | None:
    flat = value.detach().float().flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return None
    q = torch.quantile(flat, torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), device=flat.device))
    return dict(zip(("min", "p10", "median", "p90", "max"), (float(x) for x in q), strict=True))


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args_cli.seed)
    n = args_cli.num_envs
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 120.0
    env_cfg.terminate_on_drop = False
    env_cfg.success_hold_steps = 100000
    # Disable the force cutoffs so a single attempt per env is measured (no mid-run auto-reset).
    env_cfg.tactile_terminate_steps = 1_000_000_000
    env_cfg.tactile_hard_terminate_steps = 1_000_000_000
    # Legacy 16-action / 87-observation reach configuration (no distal residual, no grasp obs).
    env_cfg.enable_distal_residual = False
    env_cfg.enable_grasp_observations = False
    env_cfg.action_space = 16
    env_cfg.observation_space = 87
    env_cfg.state_space = 87

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["full_experiment_name"] = "0_reach_grasp_rate"
    agent_cfg["params"]["config"]["num_actors"] = n
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", 5.0)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", 1.0)

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(
        base_env, agent_cfg["params"]["config"]["device"], clip_obs, clip_actions, None, True
    )
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env}
    )
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(args_cli.checkpoint)
    agent.reset()

    u = env.unwrapped
    dev = u.device
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    cfg = u.cfg
    stable_count = torch.zeros(n, dtype=torch.long, device=dev)
    max_stable_run = torch.zeros(n, dtype=torch.long, device=dev)
    ever_latch = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_touch = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_stable_grasp = torch.zeros(n, dtype=torch.bool, device=dev)
    latch_steps = torch.zeros(n, dtype=torch.long, device=dev)
    max_grasp_quality = torch.zeros(n, device=dev)
    max_hold_quality = torch.zeros(n, device=dev)
    max_clearance = torch.full((n,), -float("inf"), device=dev)
    force_peak = torch.zeros(n, device=dev)

    for _ in range(args_cli.steps):
        obs = agent.obs_to_torch(obs)
        actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
        obs, _, _, _ = env.step(actions)
        u._compute_intermediate_values()
        signals = u._compute_grasp_signals()
        force_max = signals["force_magnitude"].max(dim=-1).values
        clearance = u._object_true_min_z() - u._table_surface_z
        stable = (
            u._is_grasped
            & (signals["grasp_quality"] >= cfg.grasp_quality_high)
            & (signals["hold_quality"] >= cfg.close_option_min_hold_quality)
            & (force_max <= cfg.grasp_bonus_max_force)
        )
        stable_count = torch.where(stable, stable_count + 1, torch.zeros_like(stable_count))
        max_stable_run = torch.maximum(max_stable_run, stable_count)
        ever_stable_grasp |= stable_count >= args_cli.stable_steps
        ever_latch |= u._is_grasped
        ever_touch |= force_max >= cfg.contact_force_thr
        latch_steps += u._is_grasped.long()
        max_grasp_quality = torch.maximum(max_grasp_quality, signals["grasp_quality"])
        max_hold_quality = torch.maximum(max_hold_quality, signals["hold_quality"])
        max_clearance = torch.maximum(max_clearance, clearance)
        force_peak = torch.maximum(force_peak, force_max)

    grasp_rate = float(ever_stable_grasp.float().mean())
    metrics = {
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "num_envs": n,
        "steps": args_cli.steps,
        "stable_steps": args_cli.stable_steps,
        "seed": args_cli.seed,
        "stable_grasp_count": int(ever_stable_grasp.sum()),
        "stable_grasp_rate": grasp_rate,
        "funnel": {
            "ever_touch": int(ever_touch.sum()),
            "ever_latch": int(ever_latch.sum()),
            "ever_stable_grasp_15f": int(ever_stable_grasp.sum()),
        },
        "max_stable_run_steps": _summary(max_stable_run.float()),
        "latch_occupancy": _summary(latch_steps.float() / args_cli.steps),
        "max_grasp_quality": _summary(max_grasp_quality),
        "max_hold_quality": _summary(max_hold_quality),
        "max_true_clearance": _summary(max_clearance),
        "force_peak": _summary(force_peak),
    }
    Path(args_cli.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.output).write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"GRASP RATE: {int(ever_stable_grasp.sum())}/{n} = {grasp_rate*100:.1f}% "
        f"(15-frame stable latch); ever_latch={int(ever_latch.sum())}; "
        f"max_clearance p90={metrics['max_true_clearance']['p90']:.3f}m",
        flush=True,
    )
    print(f"wrote {args_cli.output}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
