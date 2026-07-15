# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the settled resting pose of the pick_tool object + record a zero-action clip.

Spawns the object high with identity orientation, lets it settle under gravity (arm stays
at home), reports the deterministic settled z and quaternion (to bake into tool_asset.py),
and writes a short close-up video so the object's size vs the hand can be eyeballed.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--settle_steps", type=int, default=200)
parser.add_argument("--out_dir", type=str, default="/tmp/xhand_inhand/tool_measure")
parser.add_argument("--eye", type=float, nargs=3, default=[1.7, -1.1, 0.9])
parser.add_argument("--lookat", type=float, nargs=3, default=[0.5, 0.0, 0.15])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # use the REAL cfg spawn pose (TOOL_REST_Z / TOOL_REST_QUAT at x=0.5); just disable the
    # per-reset noise so we get a single canonical settle to check stability + drift.
    env_cfg.reset_object_pos_noise = (0.0, 0.0)
    env_cfg.reset_object_yaw_range = (0.0, 0.0)
    if hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args_cli.eye)
    env_cfg.viewer.lookat = tuple(args_cli.lookat)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env, video_folder=args_cli.out_dir, step_trigger=lambda s: s == 0,
        video_length=args_cli.settle_steps, disable_logger=True, name_prefix="settle",
    )

    u = env.unwrapped
    env.reset()
    zero = torch.zeros((u.num_envs, u.cfg.action_space), device=u.device)
    for _ in range(args_cli.settle_steps):
        env.step(zero)

    z = (u.object_pos_w[:, 2] - u.scene.env_origins[:, 2])
    q = u.object_quat_w
    spawn_z = u.object_default_z[0].item()
    print(f"[MEASURE] spawn_z(default)={spawn_z:.5f}  settled_z={z[0].item():.5f}  DRIFT={z[0].item()-spawn_z:+.5f}")
    if hasattr(u, "grasp_pos_w"):
        g = u.grasp_pos_w[0] - u.scene.env_origins[0]
        print(f"[MEASURE] handle grasp point (env-local) = ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})  [z>0 => above table]")
    print(f"[MEASURE] settled_z = {z[0].item():.5f}  spread={ (z-z[0]).abs().max().item():.5f}")
    print(f"[MEASURE] settled_quat_wxyz = ({q[0,0]:.5f}, {q[0,1]:.5f}, {q[0,2]:.5f}, {q[0,3]:.5f})")
    print(f"[MEASURE] quat spread across envs = {(q-q[0]).abs().max().item():.5f}")
    print(f"[MEASURE] video -> {args_cli.out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
