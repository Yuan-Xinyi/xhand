# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the XHand task scene (fixed-base hand + cube) to a PNG for inspection.

    python scripts/shot_hand.py --out /tmp/shot.png --settle 60 --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Screenshot the XHand task scene.")
parser.add_argument("--out", type=str, default="/tmp/xhand_shot.png")
parser.add_argument("--settle", type=int, default=60, help="physics steps to let the cube settle")
parser.add_argument("--eye", type=float, nargs=3, default=[0.5, -0.55, 0.85])
parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.5])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# camera rendering requires this
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext

from xhand_inhand.tasks.direct.xhand_repose.xhand_repose_env_cfg import XHandReposeEnvCfg


def main():
    cfg = XHandReposeEnvCfg()

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device))

    # ground + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    )

    # hand (single instance, fixed base — reuse the task's robot cfg)
    hand = Articulation(cfg.robot_cfg.replace(prim_path="/World/Robot"))
    # cube (reuse the task's object cfg)
    cube = RigidObject(cfg.object_cfg.replace(prim_path="/World/object"))

    # camera
    cam = Camera(
        CameraCfg(
            prim_path="/World/cam",
            update_period=0,
            height=640,
            width=640,
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

    # set the hand to its configured init pose/joints
    hand.write_joint_state_to_sim(hand.data.default_joint_pos.clone(), hand.data.default_joint_vel.clone())

    dt = sim.get_physics_dt()
    for _ in range(args_cli.settle):
        hand.set_joint_position_target(hand.data.default_joint_pos)
        hand.write_data_to_sim()
        sim.step()
        hand.update(dt)
        cube.update(dt)
        cam.update(dt)

    rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(args_cli.out, rgb)
    except Exception:
        from PIL import Image

        Image.fromarray(rgb).save(args_cli.out)
    print(f"[SHOT] saved {args_cli.out}  shape={rgb.shape}")
    print(f"[SHOT] cube pos = {cube.data.root_pos_w[0].cpu().numpy()}")

    simulation_app.close()


if __name__ == "__main__":
    main()
