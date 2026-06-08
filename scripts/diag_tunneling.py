# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose whether the pick-pen policy is exploiting contact tunneling.

Loads a trained rl_games checkpoint and rolls it out while instrumenting, at the
control-step rate, three tunneling fingerprints on the pen:

  1. SPEED SPIKE   -- the pen's linear speed jumps toward max_depenetration_velocity.
                      A 23 g pen pushed by a position-controlled hand cannot reach
                      several m/s under *normal* contact; such a spike is the
                      solver's depenetration "kick" launching the pen to the other
                      side of the geometry.
  2. POSITION JUMP -- the pen center moves a physically implausible distance in a
                      single step (it teleports across a collision surface).
  3. DEEP OVERLAP  -- the pen center comes closer to a hand link origin than is
                      geometrically possible without interpenetration.

Run (headless):
  python scripts/diag_tunneling.py --task Pick-Pen-Direct-v0 \
      --checkpoint logs/rl_games/pick_pen/2026-06-06_14-59-41/nn/pick_pen.pth \
      --num_envs 16 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose contact tunneling in pick-pen.")
parser.add_argument("--task", type=str, default="Pick-Pen-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to rl_games .pth checkpoint.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=250, help="Control steps to roll out.")
parser.add_argument("--speed_thresh", type=float, default=1.5, help="Pen speed (m/s) flagged as a spike.")
parser.add_argument("--jump_thresh", type=float, default=0.02, help="Per-step pen displacement (m) flagged as a jump.")
parser.add_argument("--overlap_thresh", type=float, default=0.012, help="Pen-center to hand-link distance (m) flagged as overlap.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import xhand_inhand.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False

    resume_path = retrieve_file_path(args_cli.checkpoint)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    base = env.unwrapped
    dt = base.cfg.sim.dt * base.cfg.decimation  # control-step dt
    max_depen = base.cfg.object_cfg.spawn.rigid_props.max_depenetration_velocity
    body_names = base.robot.body_names

    print(f"\n[CFG] sim.dt={base.cfg.sim.dt:.4f}s  decimation={base.cfg.decimation}  control_dt={dt:.4f}s")
    print(f"[CFG] object max_depenetration_velocity = {max_depen} m/s")
    print(f"[CFG] robot solver iters: pos={base.cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count}"
          f" vel={base.cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count}\n")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required by rl_games: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    prev_pos = base.object.data.root_pos_w.clone()
    N = base.num_envs
    max_speed = torch.zeros(N, device=base.device)
    max_jump = torch.zeros(N, device=base.device)
    min_overlap = torch.full((N,), 1e9, device=base.device)
    min_overlap_body = [-1] * N
    n_spike = n_jump = n_overlap = 0
    events = []

    for t in range(args_cli.steps):
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

        base._compute_intermediate_values()
        pos = base.object.data.root_pos_w
        vel = base.object.data.root_lin_vel_w
        speed = torch.norm(vel, dim=-1)
        jump = torch.norm(pos - prev_pos, dim=-1)
        prev_pos = pos.clone()

        # nearest hand-link origin to the pen center (deep-overlap proxy)
        body_pos = base.robot.data.body_pos_w  # (N, B, 3)
        d = torch.norm(body_pos - pos.unsqueeze(1), dim=-1)  # (N, B)
        link_d, link_i = d.min(dim=1)

        max_speed = torch.maximum(max_speed, speed)
        max_jump = torch.maximum(max_jump, jump)
        upd = link_d < min_overlap
        for e in torch.nonzero(upd, as_tuple=False).flatten().tolist():
            min_overlap[e] = link_d[e]
            min_overlap_body[e] = link_i[e].item()

        spike = speed > args_cli.speed_thresh
        bigjump = jump > args_cli.jump_thresh
        overlap = link_d < args_cli.overlap_thresh
        n_spike += int(spike.sum())
        n_jump += int(bigjump.sum())
        n_overlap += int(overlap.sum())

        flagged = torch.nonzero(spike | bigjump | overlap, as_tuple=False).flatten()
        for e in flagged.tolist():
            if len(events) < 40:
                tags = []
                if spike[e]:
                    tags.append(f"SPEED={speed[e]:.2f}")
                if bigjump[e]:
                    tags.append(f"JUMP={jump[e]*1000:.1f}mm")
                if overlap[e]:
                    tags.append(f"OVERLAP={link_d[e]*1000:.1f}mm@{body_names[link_i[e]]}")
                events.append(f"  t={t:3d} env{e:2d}  z={pos[e,2]:.3f}  " + "  ".join(tags))

    print("=" * 78)
    print("TUNNELING FINGERPRINT REPORT  (over {} steps x {} envs)".format(args_cli.steps, N))
    print("=" * 78)
    print(f"  speed spikes (>{args_cli.speed_thresh} m/s) : {n_spike} step-envs   "
          f"max pen speed = {max_speed.max():.2f} m/s  (depen cap = {max_depen})")
    print(f"  position jumps (>{args_cli.jump_thresh*1000:.0f} mm/step): {n_jump} step-envs   "
          f"max single-step jump = {max_jump.max()*1000:.1f} mm")
    print(f"  deep overlaps (<{args_cli.overlap_thresh*1000:.0f} mm to a link): {n_overlap} step-envs   "
          f"closest pen-link approach = {min_overlap.min()*1000:.1f} mm")
    closest_e = int(min_overlap.argmin())
    cb = min_overlap_body[closest_e]
    print(f"     -> closest approach to link '{body_names[cb] if cb >= 0 else '?'}'")
    print("-" * 78)
    if events:
        print("First flagged events (env, step):")
        for line in events:
            print(line)
    else:
        print("No tunneling fingerprints flagged at current thresholds.")
    print("=" * 78)
    verdict = (max_speed.max() > 0.7 * max_depen) or (max_jump.max() > 0.05)
    print("VERDICT:", "LIKELY TUNNELING (speed approaches depen cap / large teleport jumps)"
          if verdict else "no strong tunneling signature")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
