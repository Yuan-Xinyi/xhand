# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Report the mvp25 is_grasped hysteresis latch over a rollout: raw contact_grasp vs the latched
is_grasped, plus dense-lift gating. Answers 'did is_grasped ever = 1'."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["full_experiment_name"] = "0_isg"
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

    n_contact = 0; n_isg = 0; ever_isg = False; max_contact_steps = 0; max_rel_lift = 0.0
    batch_thumb = 0.0; batch_isg = 0.0; batch_maxforce = 0.0
    term_sum = {"r_reach_mean": 0.0, "r_grasp_mean": 0.0, "r_lift_dense_mean": 0.0, "r_lift_bonus_mean": 0.0}
    print("step  raw_contact  is_grasped  contact_steps  grasp_rel_lift")
    for t in range(args_cli.steps):
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
        for k in term_sum:
            if k in u.extras.get("log", {}):
                term_sum[k] += float(u.extras["log"][k])
        tc, oc = u._finger_contact_state()
        raw = bool((tc[0] & (oc[0] >= 1)).item())
        isg = bool(u._is_grasped[0].item())
        cs = int(u._contact_steps[0].item())
        oz = (u.object_pos_w[0, 2] - u.scene.env_origins[0, 2] - u.object_default_z[0]).item()
        rel = max(0.0, oz - u._grasp_baseline_lift[0].item()) if u._grasp_baseline_lift[0].item() < 1e5 else 0.0
        n_contact += raw; n_isg += isg; ever_isg = ever_isg or isg
        max_contact_steps = max(max_contact_steps, cs); max_rel_lift = max(max_rel_lift, rel)
        # BATCH stats over ALL envs (what training logs): thumb-contact frac, is_grasped frac, max force
        net = u._contact_sensor.data.net_forces_w
        batch_thumb += tc.float().mean().item()
        batch_isg += u._is_grasped.float().mean().item()
        batch_maxforce = max(batch_maxforce, float(net.norm(dim=-1).max().item()) if net is not None else 0.0)
        if t % 15 == 0 or isg:
            print(f"{t:4d}   {int(raw)}          {int(isg)}          {cs:3d}          {rel:+.3f}")
    N = args_cli.steps
    print("\n---- summary ----")
    print(f"[env-0] raw contact_grasp frac : {n_contact/N*100:.0f}%")
    print(f"[env-0] is_grasped (latch) frac: {n_isg/N*100:.0f}%   EVER is_grasped=1: {ever_isg}")
    print(f"[BATCH n={u.num_envs}] thumb_contact_frac(time-avg): {batch_thumb/N*100:.1f}%   "
          f"is_grasped_frac(time-avg): {batch_isg/N*100:.1f}%   max fingertip force over batch: {batch_maxforce:.1f}N")
    print(f"max consecutive contact_steps reached: {max_contact_steps}  (needs >= {u.cfg.grasp_confirm_steps} to latch is_grasped)")
    print(f"max grasp_rel_lift (dense-lift credited height): {max_rel_lift:+.3f} m")
    print(f"--- reward composition over {N} steps (episode-return-ish) ---")
    for k, v in term_sum.items():
        print(f"    {k:20s}: {v:8.1f}")
    print(f"    {'TOTAL':20s}: {sum(term_sum.values()):8.1f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
