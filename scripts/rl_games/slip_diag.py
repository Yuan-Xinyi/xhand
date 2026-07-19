# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slip vs no-lift diagnostic for pick_tool_token. Runs the policy in 1 env and logs, per step:
the HAND height (palm_center z, delta from step 0), the OBJECT height (lift), the contact grasp
state and total normal force. Distinguishes the two reasons a correct-looking grasp fails to lift:
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
    rows = []
    print("step  palm_dz  obj_lift  gap    contact  fN   (palm_dz=hand rise, gap=palm_dz-lift, m)")
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
        lift = (u.object_pos_w[0, 2] - origin_z - u.object_default_z[0]).item()
        fmag = u._contact_sensor.data.force_matrix_w.norm(dim=-1).sum(dim=-1)[0]
        fN = fmag.sum().item()
        tc, oc = u._finger_contact_state()
        cg = bool((tc[0] & (oc[0] >= 1)).item())
        rows.append((palm_dz, lift, cg, fN))
        if t % 15 == 0:
            print(f"{t:4d}  {palm_dz:+.3f}   {lift:+.3f}   {palm_dz-lift:+.3f}  {int(cg)}       {fN:5.1f}")

    palm_dz = [r[0] for r in rows]; lifts = [r[1] for r in rows]
    hi = max(range(len(rows)), key=lambda i: palm_dz[i])
    print("\n---- summary ----")
    print(f"peak hand rise (palm_dz): {max(palm_dz):+.3f} m")
    print(f"peak object lift        : {max(lifts):+.3f} m")
    print(f"at max hand-rise (step {hi}): palm_dz={palm_dz[hi]:+.3f}  obj_lift={lifts[hi]:+.3f}  "
          f"contact={int(rows[hi][2])}  fN={rows[hi][3]:.1f}")
    if max(palm_dz) < 0.05:
        v = "NO-LIFT: hand never rose >5cm -> the policy is NOT commanding an upward arm move."
    elif palm_dz[hi] - lifts[hi] > 0.03:
        v = "SLIP: hand rose but object lagged >3cm -> the grip slips (friction/contact geometry)."
    else:
        v = "OBJECT FOLLOWS: hand & object rise together -> lift mechanically works (just needs more)."
    print(f"VERDICT: {v}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
