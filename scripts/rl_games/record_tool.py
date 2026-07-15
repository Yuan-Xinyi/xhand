# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a close-up video of a trained rl_games policy on a single environment.

play.py records the default wide viewport (all envs tiled, camera 7.5 m out), which is
useless for inspecting a grasp. This frames one environment up close via the env viewer
camera and writes a single mp4.

    python scripts/rl_games/record_tool.py --task=Pick-Tool-Token-Direct-v0 \
        --checkpoint <path.pth> --video_length 300 --seed 0 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a close-up rl_games policy video.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, default="/tmp/xhand_inhand/tool_video")
parser.add_argument("--eye", type=float, nargs=3, default=[1.7, -1.1, 0.9])
parser.add_argument("--lookat", type=float, nargs=3, default=[0.5, 0.0, 0.15])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for offscreen video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import xhand_inhand.tasks  # noqa: F401  (registers our gym ids)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # close-up camera tracking env 0 (fabric clones only render env_0 unless disabled)
    if hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args_cli.eye)
    env_cfg.viewer.lookat = tuple(args_cli.lookat)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    # the SAPG-fork parses policy_idx = int(name.split('_')[0]); give it an int-leading name
    agent_cfg["params"]["config"]["full_experiment_name"] = "0_eval"

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", 5.0)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", 1.0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args_cli.out_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
        name_prefix="tool_grasp",
    )
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, None, True)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(args_cli.checkpoint)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    successes = 0
    max_lift = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    for t in range(args_cli.video_length):
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
        u = env.unwrapped
        lift = u.object_pos_w[:, 2] - u.scene.env_origins[:, 2] - u.object_default_z
        max_lift = torch.maximum(max_lift, lift)
        successes = max(successes, int(u._is_success.sum().item()))
    print(f"[RECORD] wrote video to {args_cli.out_dir}")
    print(f"[RECORD] peak lift (env0) = {max_lift[0].item():.3f} m ; success_envs_seen = {successes}/{u.num_envs}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
