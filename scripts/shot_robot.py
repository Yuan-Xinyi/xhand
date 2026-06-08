# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a project robot at its home pose to a PNG (to inspect hand orientation)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="FR3_XHAND_CFG")
parser.add_argument("--out", type=str, default="/tmp/robot_shot.png")
parser.add_argument("--eye", type=float, nargs=3, default=[1.1, -0.9, 0.75])
parser.add_argument("--target", type=float, nargs=3, default=[0.25, 0.0, 0.5])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext

import xhand_inhand.robots as robots


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 100, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    robot = Articulation(getattr(robots, args_cli.robot).replace(prim_path="/World/Robot"))
    cam = Camera(
        CameraCfg(
            prim_path="/World/cam",
            height=720,
            width=720,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
            ),
        )
    )
    sim.reset()
    cam.set_world_poses_from_view(
        eyes=torch.tensor([args_cli.eye], device=sim.device),
        targets=torch.tensor([args_cli.target], device=sim.device),
    )
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())

    dt = sim.get_physics_dt()
    for _ in range(40):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        cam.update(dt)

    rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(args_cli.out, rgb)
    except Exception:
        from PIL import Image

        Image.fromarray(rgb).save(args_cli.out)
    print(f"[SHOT] saved {args_cli.out}")
    simulation_app.close()


if __name__ == "__main__":
    main()
