# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slip vs no-lift diagnostic for pick_tool_token. Runs the policy in 1 env and logs, per step:
the hand rise, TRUE mesh clearance, robust grasp latch, shared grasp qualities, and object-filtered
force. Distinguishes the two reasons a robust grasp fails to lift:
  * SLIP   : hand rises but object does NOT follow (palm_dz grows, lift lags, contact drops)
  * NO-LIFT: hand never rises (palm_dz ~0) -> the policy isn't commanding an upward move."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=200)
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
    agent_cfg["params"]["config"]["full_experiment_name"] = "0_slip"
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

    palm0 = None
    clearance0 = None
    rows = []
    max_object_force = torch.zeros(len(u.ee_names), device=u.device)
    max_net_force = torch.zeros(len(u.ee_names), device=u.device)
    print("step  palm_dz  true_clear  clear_d  gap    latch  q_wrap  q_grasp  q_hold  objF")
    for t in range(args_cli.steps):
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
        origin_z = u.scene.env_origins[0, 2]
        palm_z = (u.palm_center_w[0, 2] - origin_z).item()
        if palm0 is None:
            palm0 = palm_z
        palm_dz = palm_z - palm0
        clearance = float((u._object_true_min_z()[0] - u._table_surface_z).item())
        if clearance0 is None:
            clearance0 = clearance
        clearance_delta = clearance - clearance0
        object_force = u._finger_object_force_magnitudes()[0]
        net_force = u._finger_net_force_magnitudes()[0]
        max_object_force = torch.maximum(max_object_force, object_force)
        max_net_force = torch.maximum(max_net_force, net_force)
        signals = u._compute_grasp_signals()
        latch = bool(u._is_grasped[0].item())
        q_wrap = float(signals["quality"][0].item())
        q_grasp = float(signals["grasp_quality"][0].item())
        q_hold = float(signals["hold_quality"][0].item())
        fN = object_force.sum().item()
        rows.append((palm_dz, clearance, clearance_delta, latch, fN, q_wrap, q_grasp, q_hold))
        if t % 15 == 0:
            print(
                f"{t:4d}  {palm_dz:+.3f}   {clearance:+.4f}    {clearance_delta:+.3f}  "
                f"{palm_dz-clearance_delta:+.3f}  {int(latch)}      {q_wrap:.3f}   "
                f"{q_grasp:.3f}   {q_hold:.3f}  {fN:5.1f}"
            )

    palm_dz = [r[0] for r in rows]
    clearances = [r[1] for r in rows]
    clearance_delta = [r[2] for r in rows]
    hi = max(range(len(rows)), key=lambda i: palm_dz[i])
    print("\n---- summary ----")
    print(f"peak hand rise (palm_dz): {max(palm_dz):+.3f} m")
    print(f"peak TRUE mesh clearance: {max(clearances):+.4f} m")
    print(f"peak clearance gain     : {max(clearance_delta):+.3f} m")
    print(
        f"robust latch fraction={sum(1 for r in rows if r[3]) / len(rows):.3f}; "
        f"peak q_wrap={max(r[5] for r in rows):.3f}, q_grasp={max(r[6] for r in rows):.3f}"
    )
    print(
        f"at max hand-rise (step {hi}): palm_dz={palm_dz[hi]:+.3f}  "
        f"true_clear={clearances[hi]:+.4f}  clear_d={clearance_delta[hi]:+.3f}  "
        f"latch={int(rows[hi][3])}  objF={rows[hi][4]:.1f}"
    )
    object_mf = {u.ee_names[i]: round(max_object_force[i].item(), 3) for i in range(len(u.ee_names))}
    net_mf = {u.ee_names[i]: round(max_net_force[i].item(), 3) for i in range(len(u.ee_names))}
    print(f"max OBJECT-filtered force per finger (N): {object_mf}")
    print(f"max unfiltered NET force per finger (N), audit only: {net_mf}")
    if not any(r[3] for r in rows):
        v = "NO ROBUST GRASP: the shared grasp latch never formed; slip/no-lift is not yet diagnosable."
    elif max(palm_dz) < 0.05:
        v = "NO-LIFT: hand never rose >5cm -> the policy is NOT commanding an upward arm move."
    elif palm_dz[hi] - clearance_delta[hi] > 0.03:
        v = "SLIP: hand rose but object lagged >3cm -> the grip slips (friction/contact geometry)."
    else:
        v = "OBJECT FOLLOWS: hand & object rise together -> lift mechanically works (just needs more)."
    print(f"VERDICT: {v}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
