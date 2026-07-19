# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Experiment 2: CONDITIONAL action statistics while is_grasped=True. Answers 'after grasping, how much
UPWARD end-effector motion does the policy actually produce?' Distinguishes:
  (1) EEF dz~0 AND arm-action std low          -> sigma collapse (under-exploration)
  (2) arm-action std NOT low, EEF dz symmetric -> white noise, no temporal correlation (no sustained rise)
  (3) arm-action large but EEF dz~0            -> action param / joint coupling / gain / pose constraint bug
Records per-dim arm-action mean/std, EEF (palm) dz mean/std, P(dz>0), longest consecutive-positive-dz run."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import gymnasium as gym
import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["full_experiment_name"] = "0_cond"
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
    n_arm = u._n_arm
    obs = env.reset()
    if isinstance(obs, dict): obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)

    # STOCHASTIC actions (is_deterministic=False) so we measure the policy's actual exploration variance
    arm_actions = []   # (M, n_arm) arm action while grasped
    dz_list = []       # (M,) palm dz while grasped
    prev_palm_z = None
    run_pos = 0; max_run_pos = 0
    n_total = 0; n_grasped = 0
    for t in range(args_cli.steps):
        with torch.inference_mode():
            ob = agent.obs_to_torch(obs)
            act = agent.get_action(ob, is_deterministic=False)
            obs, _, dones, _ = env.step(act)
        palm_z = u.palm_center_w[0, 2].item()
        dz = 0.0 if prev_palm_z is None else palm_z - prev_palm_z
        prev_palm_z = palm_z
        isg = bool(u._is_grasped[0].item())
        n_total += 1
        if isg:
            n_grasped += 1
            arm_actions.append(act[0, :n_arm].detach().cpu())
            dz_list.append(dz)
            if dz > 1e-4:
                run_pos += 1; max_run_pos = max(max_run_pos, run_pos)
            else:
                run_pos = 0

    print("\n===== CONDITIONAL ACTION STATS while is_grasped=True =====")
    print(f"steps: {n_total}, grasped steps: {n_grasped} ({n_grasped/n_total*100:.0f}%)")
    if n_grasped < 5:
        print("too few grasped steps for stats"); env.close(); return
    A = torch.stack(arm_actions)               # (M, n_arm)
    dz = torch.tensor(dz_list)                 # (M,)
    print(f"\narm action MEAN per dim: {[round(x, 3) for x in A.mean(0).tolist()]}")
    print(f"arm action STD  per dim: {[round(x, 3) for x in A.std(0).tolist()]}")
    print(f"  (action space is [-1,1]; std ~1 = full exploration, std <0.1 = collapsed)")
    print(f"\nEEF (palm) dz while grasped:  mean={dz.mean().item()*1000:+.2f} mm/step  std={dz.std().item()*1000:.2f} mm")
    print(f"P(dz > 0)                  : {(dz > 1e-4).float().mean().item()*100:.0f}%   (50% = symmetric up/down)")
    print(f"longest consecutive +dz run: {max_run_pos} steps   (need a long run to lift ~20cm)")
    print(f"net palm rise over grasped steps (sum dz): {dz.sum().item()*100:+.2f} cm")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
