# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose whether the trained policy actually drives fingertips to the handle keypoints.

Loads a checkpoint, runs 1 env, and logs per-step: mean/min fingertip-to-nearest-keypoint
distance (u._curr_fingertip_distances) and object lift. Reveals whether the fingers reach the
keypoints, and whether the object is lifted WITHOUT the fingers ever getting close (a knock).
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["full_experiment_name"] = "0_diag"
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", 5.0)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", 1.0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, agent_cfg["params"]["config"]["device"], clip_obs, clip_actions, None, True)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = 1
    runner = Runner(); runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player(); agent.restore(args_cli.checkpoint); agent.reset()

    u = env.unwrapped
    obs = env.reset()
    if isinstance(obs, dict): obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn: agent.init_rnn()

    print("step  meanFKdist  minFKdist  lift   (FK=fingertip->nearest keypoint, m)")
    rows = []
    for t in range(args_cli.steps):
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
        fk = u._curr_fingertip_distances[0]           # (5,) min dist to nearest keypoint per finger
        lift = (u.object_pos_w[0, 2] - u.scene.env_origins[0, 2] - u.object_default_z[0]).item()
        rows.append((fk.mean().item(), fk.min().item(), lift))
        if t % 15 == 0:
            print(f"{t:4d}  {fk.mean().item():.4f}     {fk.min().item():.4f}    {lift:+.3f}")
    import statistics as st
    meanFK = [r[0] for r in rows]; minFK = [r[1] for r in rows]; lifts = [r[2] for r in rows]
    print("---- summary ----")
    print(f"mean-finger->keypoint dist: overall mean={st.mean(meanFK):.4f}  best(min over episode)={min(meanFK):.4f}")
    print(f"closest any-finger->keypoint ever: {min(minFK):.4f} m")
    print(f"peak lift: {max(lifts):.3f} m")
    # was it lifted while fingers were still far? (knock detection)
    lifted_idx = [i for i, l in enumerate(lifts) if l > 0.10]
    if lifted_idx:
        i0 = lifted_idx[0]
        print(f"first crossed 10cm lift at step {i0}; mean FK dist at that step = {meanFK[i0]:.4f} m "
              f"(small=>grasped, large=>knocked)")
    else:
        print("never crossed 10cm lift in this episode")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
