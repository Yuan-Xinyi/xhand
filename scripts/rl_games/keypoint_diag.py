# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose approach, object contact, robust wrap, and true hammer clearance.

The grasp verdict comes only from the environment's shared robust latch.  Raw fingertip force
and thumb/non-thumb contacts are reported as evidence, but are never promoted to a grasp.
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

    print(f"[SENSOR] singleton object-filtered order = {u.ee_names}")
    print(
        f"[SENSOR] thumb_idx={u._contact_thumb_idx}  other_ids={u._contact_other_ids.tolist()}  "
        f"thr={u.cfg.contact_force_thr}N"
    )
    max_object_force = torch.zeros(len(u.ee_names), device=u.device)
    max_net_force = torch.zeros(len(u.ee_names), device=u.device)
    print("step  meanFK   minFK   true_clear  latch  thumb+other  q_wrap  q_grasp  q_hold  objF")
    rows = []
    for t in range(args_cli.steps):
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
        fk = u._curr_fingertip_distances[0]
        clearance = float((u._object_true_min_z()[0] - u._table_surface_z).item())
        object_force = u._finger_object_force_magnitudes()[0]
        net_force = u._finger_net_force_magnitudes()[0]
        max_object_force = torch.maximum(max_object_force, object_force)
        max_net_force = torch.maximum(max_net_force, net_force)
        signals = u._compute_grasp_signals()
        thumb = bool(signals["thumb_contact"][0].item())
        other = int(signals["other_contact_count"][0].item())
        q_wrap = float(signals["quality"][0].item())
        q_grasp = float(signals["grasp_quality"][0].item())
        q_hold = float(signals["hold_quality"][0].item())
        latch = bool(u._is_grasped[0].item())
        to_handle = u.handle_center_w[0] - u.palm_center_w[0]
        to_handle = to_handle / (to_handle.norm() + 1e-6)
        pf = float((u.palm_normal_w[0] * to_handle).sum().item())
        if t == 0 or t == 60 or t == 120:
            print(f"    [palm_facing raw @step{t}] = {pf:+.3f}  (+ = palm faces handle, - = hand-back)")
        rows.append(
            (fk.mean().item(), fk.min().item(), clearance, latch, other, thumb, q_wrap, q_grasp, q_hold)
        )
        if t % 15 == 0:
            print(
                f"{t:4d}  {fk.mean().item():.4f}  {fk.min().item():.4f}  {clearance:+.4f}   "
                f"{int(latch)}       {int(thumb)}+{other}       {q_wrap:.3f}   {q_grasp:.3f}   "
                f"{q_hold:.3f}  {object_force.sum().item():5.1f}"
            )
    import statistics as st
    meanFK = [r[0] for r in rows]; minFK = [r[1] for r in rows]; clearances = [r[2] for r in rows]
    grasp_frac = sum(1 for r in rows if r[3]) / len(rows)
    any_thumb = any(r[5] for r in rows); max_other = max(r[4] for r in rows)
    print(
        f"[GRASP] robust latch steps={grasp_frac*100:.1f}%  ever={any(r[3] for r in rows)}; "
        f"raw topology audit: thumb ever={any_thumb}, max simultaneous others={max_other}"
    )
    object_mf = {u.ee_names[i]: round(max_object_force[i].item(), 3) for i in range(len(u.ee_names))}
    net_mf = {u.ee_names[i]: round(max_net_force[i].item(), 3) for i in range(len(u.ee_names))}
    print(f"[CONTACT] max OBJECT-filtered force per finger (N): {object_mf}")
    print(f"[CONTACT] max unfiltered NET force per finger (N), audit only: {net_mf}")
    print("---- summary ----")
    print(f"mean-finger->keypoint dist: overall mean={st.mean(meanFK):.4f}  best(min over episode)={min(meanFK):.4f}")
    print(f"closest any-finger->keypoint ever: {min(minFK):.4f} m")
    print(f"peak TRUE mesh clearance: {max(clearances):.4f} m")
    print(
        f"peak qualities: q_wrap={max(r[6] for r in rows):.3f}  "
        f"q_grasp={max(r[7] for r in rows):.3f}  q_hold={max(r[8] for r in rows):.3f}"
    )
    lifted_idx = [i for i, clearance in enumerate(clearances) if clearance > 0.10]
    if lifted_idx:
        i0 = lifted_idx[0]
        print(
            f"first crossed 10cm true clearance at step {i0}; robust_latch={int(rows[i0][3])}, "
            f"q_grasp={rows[i0][7]:.3f}, mean FK dist={meanFK[i0]:.4f} m"
        )
    else:
        print("never crossed 10cm true mesh clearance in this episode")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
