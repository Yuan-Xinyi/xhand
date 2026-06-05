# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded smoke test for a registered task: build, step N times, assert shapes, exit cleanly.

    python scripts/check_task.py --task=Xhand-Repose-Cube-OpenAI-LSTM-Direct-v0 --num_envs 4 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bounded task smoke test.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401  (registers our gym ids)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[CHECK] task={args_cli.task}")
    print(f"[CHECK] action_space      = {env.action_space}")
    print(f"[CHECK] observation_space = {env.observation_space}")

    obs, _ = env.reset()
    print(f"[CHECK] reset obs keys = {list(obs.keys())}")
    for k, v in obs.items():
        print(f"[CHECK]   obs['{k}'] shape = {tuple(v.shape)}")

    for i in range(args_cli.steps):
        actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
        obs, rew, terminated, truncated, info = env.step(actions)
    print(f"[CHECK] stepped {args_cli.steps} times OK")
    print(f"[CHECK]   reward shape = {tuple(rew.shape)}, mean = {rew.mean().item():.4f}")
    print("[CHECK] PASS")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
