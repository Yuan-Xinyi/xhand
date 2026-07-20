# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Audit robust grasp/contact signals at 1-env and training-relevant parallel scale."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--require_object_contact",
    action="store_true",
    help="Exit nonzero if the filtered singleton sensors never report object contact.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

import xhand_inhand.tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["full_experiment_name"] = "0_isg"
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", 5.0)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", 1.0)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(
        env, agent_cfg["params"]["config"]["device"], clip_obs, clip_actions, None, True
    )
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env}
    )
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = args_cli.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(args_cli.checkpoint)
    agent.reset()

    u = env.unwrapped
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    masses = u.object.root_physx_view.get_masses().reshape(u.num_envs, -1)
    print(
        f"[MASS] shape={tuple(masses.shape)} min/median/max="
        f"{masses.min().item():.6f}/{masses.median().item():.6f}/{masses.max().item():.6f} kg"
    )
    for name in u.ee_names:
        matrix = u._object_contact_sensors[name].data.force_matrix_w
        if matrix is None or matrix.shape[0] != u.num_envs or matrix.shape[1] != 1:
            raise RuntimeError(f"bad filtered contact tensor for {name}: {None if matrix is None else matrix.shape}")
        print(f"[SENSOR] {name:20s} force_matrix shape={tuple(matrix.shape)}")

    accum = {
        "raw": 0.0,
        "latch": 0.0,
        "q_wrap": 0.0,
        "grasp_quality": 0.0,
        "hold": 0.0,
    }
    reward_keys = ("r_reach_mean", "r_grasp_mean", "r_hold_mean", "r_lift_mean", "r_success_mean")
    reward_sum = {key: 0.0 for key in reward_keys}
    max_filtered = torch.zeros(5, device=u.device)
    max_net = torch.zeros(5, device=u.device)
    max_contact_steps = 0
    max_clearance = -1.0e9
    ever_latched = False

    print("step raw latch contact_steps q_wrap grasp_q hold_q true_clearance")
    for step in range(args_cli.steps):
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, _, _ = env.step(actions)

        log = u.extras.get("log", {})
        for key in reward_keys:
            reward_sum[key] += float(log.get(key, 0.0))
        filtered = u._finger_object_force_magnitudes()
        net = u._finger_net_force_magnitudes()
        max_filtered = torch.maximum(max_filtered, filtered.max(dim=0).values)
        max_net = torch.maximum(max_net, net.max(dim=0).values)
        thumb_contact, other_count = u._finger_contact_state()
        raw = thumb_contact & (other_count >= 2)
        signals = u._compute_grasp_signals()
        clearance = u._object_true_min_z() - u._table_surface_z

        accum["raw"] += raw.float().mean().item()
        accum["latch"] += u._is_grasped.float().mean().item()
        accum["q_wrap"] += signals["quality"].mean().item()
        accum["grasp_quality"] += signals["grasp_quality"].mean().item()
        accum["hold"] += signals["hold_quality"].mean().item()
        max_contact_steps = max(max_contact_steps, int(u._contact_steps.max().item()))
        max_clearance = max(max_clearance, float(clearance.max().item()))
        ever_latched = ever_latched or bool(u._is_grasped.any().item())
        if step % 15 == 0:
            print(
                f"{step:4d} {int(raw[0])}   {int(u._is_grasped[0])}      "
                f"{int(u._contact_steps[0]):3d}       {signals['quality'][0]:.3f}  "
                f"{signals['grasp_quality'][0]:.3f}   {signals['hold_quality'][0]:.3f}  "
                f"{clearance[0]:+.4f}"
            )

    n = float(args_cli.steps)
    print("\n---- summary ----")
    print(
        f"[BATCH n={u.num_envs}] raw thumb+2 contact={100*accum['raw']/n:.3f}%  "
        f"latched={100*accum['latch']/n:.3f}%  ever_latched={ever_latched}"
    )
    print(
        f"q_wrap={accum['q_wrap']/n:.4f} grasp_quality={accum['grasp_quality']/n:.4f} "
        f"hold_quality={accum['hold']/n:.4f} max_confirm_steps={max_contact_steps}"
    )
    print(f"max true clearance={max_clearance:+.6f} m")
    print(f"max object-filtered force per finger={max_filtered.tolist()}")
    print(f"max unfiltered net force per finger  ={max_net.tolist()}")
    for key, value in reward_sum.items():
        print(f"{key:20s}: {value:9.3f}")

    env.close()
    if args_cli.require_object_contact and max_filtered.max().item() <= u.cfg.contact_force_thr:
        raise RuntimeError(
            "No object-filtered contact was observed; the singleton force_matrix path failed or the rollout never touched."
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
