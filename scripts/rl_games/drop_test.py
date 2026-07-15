# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drop the pick_tool object onto the table and log its z-trajectory to diagnose bouncing."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--drop_z", type=float, default=0.35)
parser.add_argument("--drop_x", type=float, default=0.9)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--record", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.record:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.object_cfg.init_state.pos = (args_cli.drop_x, 0.0, args_cli.drop_z)
    env_cfg.object_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env_cfg.reset_object_pos_noise = (0.0, 0.0)
    env_cfg.reset_object_yaw_range = (0.0, 0.0)
    # make sure a drop-termination or success doesn't cut the rollout short
    env_cfg.terminate_on_drop = False
    if hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False
    if args_cli.record:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (1.3, -0.9, 0.5)
        env_cfg.viewer.lookat = (0.5, 0.0, 0.05)

    render_mode = "rgb_array" if args_cli.record else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.record:
        os.makedirs("/tmp/xhand_inhand/drop_test", exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder="/tmp/xhand_inhand/drop_test",
                                       step_trigger=lambda s: s == 0, video_length=args_cli.steps,
                                       disable_logger=True, name_prefix="drop")
    u = env.unwrapped
    env.reset()
    zero = torch.zeros((u.num_envs, u.cfg.action_space), device=u.device)

    zs, vzs = [], []
    for _ in range(args_cli.steps):
        env.step(zero)
        z = (u.object_pos_w[0, 2] - u.scene.env_origins[0, 2]).item()
        vz = u.object.data.root_lin_vel_w[0, 2].item()
        zs.append(z); vzs.append(vz)

    # analyse bounces: local minima followed by a rise
    zmin_overall = min(zs)
    settle = zs[-1]
    # first contact = first index where z stops decreasing
    contact_i = next((i for i in range(1, len(zs)) if zs[i] > zs[i - 1] - 1e-5 and zs[i - 1] < args_cli.drop_z - 0.02), len(zs) - 1)
    # peak rebound after first contact
    rebound = max(zs[contact_i:contact_i + 60]) if contact_i < len(zs) else settle
    rebound_h = rebound - zs[contact_i]
    max_up_vz = max(vzs)
    # count bounces: sign changes of vz from - to +
    bounces = sum(1 for i in range(1, len(vzs)) if vzs[i - 1] < -0.05 and vzs[i] > 0.05)
    print(f"[DROP] drop_z={args_cli.drop_z:.3f}  settle_z={settle:.4f}  z_min={zmin_overall:.4f}")
    print(f"[DROP] first_contact_step={contact_i}  z_at_contact={zs[contact_i]:.4f}")
    print(f"[DROP] REBOUND after contact = {rebound_h:.4f} m (peak {rebound:.4f})")
    print(f"[DROP] max upward vz = {max_up_vz:.3f} m/s   (bounce events: {bounces})")
    print(f"[DROP] z trajectory (every 8 steps): {[round(zs[i],3) for i in range(0,len(zs),8)]}")
    q = u.object.data.root_quat_w[0]
    print(f"[DROP] SETTLED pose: z={settle:.5f}  quat_wxyz=({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f})")
    print(f"[DROP] final |lin_vel|={u.object.data.root_lin_vel_w[0].norm():.4f}  |ang_vel|={u.object.data.root_ang_vel_w[0].norm():.4f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
