# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw a debug sphere at the computed handle grasp point to confirm it lands on the handle."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--offset", type=float, nargs=3, default=[-0.0157, 0.0245, 0.0894])
parser.add_argument("--out_dir", type=str, default="/tmp/xhand_inhand/grasp_validate")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task if hasattr(args_cli, "task") else "Pick-Tool-Token-Direct-v0",
                            device=args_cli.device, num_envs=1)
    task = "Pick-Tool-Token-Direct-v0"
    env_cfg.object_cfg.init_state.pos = (0.5, 0.0, 0.0735)
    env_cfg.object_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env_cfg.reset_object_pos_noise = (0.0, 0.0)
    env_cfg.reset_object_yaw_range = (0.0, 0.0)
    if hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.3, -0.9, 0.6)
    env_cfg.viewer.lookat = (0.48, 0.02, 0.16)

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_folder=args_cli.out_dir, step_trigger=lambda s: s == 0,
                                   video_length=60, disable_logger=True, name_prefix="grasp")
    u = env.unwrapped

    marker = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/grasp_point",
        markers={"p": sim_utils.SphereCfg(radius=0.015,
                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)))},
    ))
    offset = torch.tensor(args_cli.offset, device=u.device).repeat(u.num_envs, 1)

    env.reset()
    zero = torch.zeros((u.num_envs, u.cfg.action_space), device=u.device)
    for _ in range(60):
        grasp_w = u.object_pos_w + quat_apply(u.object_quat_w, offset)
        marker.visualize(grasp_w)
        env.step(zero)
    gw = grasp_w[0]
    print(f"[VALIDATE] grasp point world = ({gw[0]:.3f}, {gw[1]:.3f}, {gw[2]:.3f})")
    print(f"[VALIDATE] object root  world = {tuple(round(v.item(),3) for v in u.object_pos_w[0])}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
