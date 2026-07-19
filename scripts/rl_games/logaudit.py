# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Log-audit: FORCE is_grasped=True (monkeypatch contact) and trace is_grasped_frac through the exact
training wrapper stack, to find where the nonzero value becomes 0 in the wandb 'Episode/' logging."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=8)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", 5.0)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", 1.0)
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    u = base_env.unwrapped
    wrapped = RlGamesVecEnvWrapper(base_env, agent_cfg["params"]["config"]["device"], clip_obs, clip_actions, None, True)

    # FORCE a valid contact grasp every step so is_grasped latches True after grasp_confirm_steps
    u._finger_contact_state = lambda: (
        torch.ones(u.num_envs, dtype=torch.bool, device=u.device),
        torch.full((u.num_envs,), 2, dtype=torch.long, device=u.device),
    )
    wrapped.reset()
    n_act = u.cfg.action_space
    print("\nstep | env-truth u._is_grasped.mean | wrapper extras['episode']['is_grasped_frac'] | key present?")
    for t in range(10):
        act = torch.zeros((u.num_envs, n_act), device=u.device)
        obs, rew, dones, extras = wrapped.step(act)
        env_truth = u._is_grasped.float().mean().item()
        ep = extras.get("episode", extras.get("log", {}))
        has = "is_grasped_frac" in ep
        logged = ep.get("is_grasped_frac", None)
        logged_v = (logged.item() if hasattr(logged, "item") else logged) if logged is not None else "MISSING"
        print(f"{t:4d} | {env_truth:.3f}                         | {logged_v}                    | {has}  "
              f"(extras keys: {list(extras.keys())})")
    base_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
