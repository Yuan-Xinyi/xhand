# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Calibrate the pregrasp-readiness gate: what scores do restored states actually show?

Rolls a policy from curriculum spawns and prints the pregrasp score distribution at reset and
over time, plus the fraction of envs that ever sustain score >= {0.15, 0.20, 0.25, 0.30} for
{4, 8, 15} frames.  Answers whether a reposition contract is miscalibrated or unreachable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--curriculum_dataset", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--joint_noise", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import sys
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "rl_games"))
import agent_bridge  # noqa: E402,F401
from flash_rl.agents.flashSAC.network import FlashSACActor  # noqa: E402

THRESHOLDS = (0.15, 0.20, 0.25, 0.30)
HOLDS = (4, 8, 15)


def load_actor(checkpoint: Path, device: torch.device) -> FlashSACActor:
    payload = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
    state = {
        k.removeprefix("_orig_mod."): v for k, v in payload["network_state_dict"].items()
    }
    actor = FlashSACActor(num_blocks=2, input_dim=115, hidden_dim=128, action_dim=21)
    actor.load_state_dict(state)
    return actor.to(device).eval()


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args_cli.seed)
    n = args_cli.num_envs
    cfg = parse_env_cfg("Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=n)
    cfg.seed = args_cli.seed
    cfg.nudge_option_mode = True
    cfg.episode_length_s = 60.0  # no resets during the probe
    cfg.curriculum_dataset = str(args_cli.curriculum_dataset)
    cfg.curriculum_boundary = "close_start"
    cfg.curriculum_reset_probability = 1.0
    cfg.curriculum_joint_noise = args_cli.joint_noise
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    actor = load_actor(args_cli.checkpoint, u.device)

    obs, _ = env.reset()
    u._compute_intermediate_values()
    score = u._nudge_pregrasp_score()
    q = torch.tensor((0.1, 0.5, 0.9), device=score.device)
    print(
        "reset score quantiles p10/p50/p90:",
        [round(float(x), 3) for x in torch.quantile(score, q)],
        flush=True,
    )
    streak = {t: torch.zeros(n, device=u.device) for t in THRESHOLDS}
    best = {t: torch.zeros(n, device=u.device) for t in THRESHOLDS}
    peak = score.clone()
    for step in range(args_cli.steps):
        mean, _ = actor.get_mean_and_std(obs["policy"], training=False)
        obs, _, _, _, _ = env.step(torch.tanh(mean))
        u._compute_intermediate_values()
        score = u._nudge_pregrasp_score()
        peak = torch.maximum(peak, score)
        for t in THRESHOLDS:
            streak[t] = torch.where(score >= t, streak[t] + 1, torch.zeros_like(streak[t]))
            best[t] = torch.maximum(best[t], streak[t])
        if step in (25, 100, 250):
            print(
                f"step {step:>3} score p10/p50/p90:",
                [round(float(x), 3) for x in torch.quantile(score, q)],
                flush=True,
            )
    print("peak score p10/p50/p90:", [round(float(x), 3) for x in torch.quantile(peak, q)])
    for t in THRESHOLDS:
        row = {h: float((best[t] >= h).float().mean()) for h in HOLDS}
        print(
            f"thr {t:.2f}: ever-held frac", {h: round(v, 3) for h, v in row.items()}, flush=True
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
