# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone visualization of the XHand right dexterous hand in Isaac Lab.

Spawns ``--num_envs`` cloned copies of the hand in a grid and (optionally)
animates a flexing motion across all joints.

Usage (with the Isaac Lab conda env active, e.g. ``env_isaaclab``):

    # 8 parallel environments, animated, GUI viewer
    python scripts/visualize_xhand.py --num_envs 8 --wave

    # headless smoke test (no GUI), stop after 120 steps
    python scripts/visualize_xhand.py --num_envs 8 --headless --max-steps 120
"""

import argparse

from isaaclab.app import AppLauncher

# --- CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize the XHand right hand.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments to spawn.")
parser.add_argument("--wave", action="store_true", help="Animate a flexing motion across the joints.")
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Stop after this many physics steps (0 = run until the window is closed). Useful for headless smoke tests.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- everything below requires the app to be running -------------------------
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import XHAND_RIGHT_CFG


@configclass
class XHandSceneCfg(InteractiveSceneCfg):
    """A grid of XHand right hands plus ground and light."""

    # ground plane (shared, not cloned per-env)
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    # dome light (shared)
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

    # the hand (prim_path templated per-env by InteractiveScene)
    robot = XHAND_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    # simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.6, 0.6, 0.5], target=[0.0, 0.0, 0.15])

    # build the cloned scene (spawns ground, light and the cloned hands)
    scene_cfg = XHandSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.4)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    robot = scene["robot"]
    print(f"[INFO] Spawned {scene.num_envs} XHand env(s).")
    print(f"[INFO]   #joints = {robot.num_joints}")
    print(f"[INFO]   joint names = {robot.joint_names}")

    lower = robot.data.soft_joint_pos_limits[..., 0]
    upper = robot.data.soft_joint_pos_limits[..., 1]

    sim_dt = sim.get_physics_dt()
    t = 0.0
    step = 0
    while simulation_app.is_running():
        if args_cli.max_steps and step >= args_cli.max_steps:
            break
        if args_cli.wave:
            # sinusoid sweeping each joint between its limits
            phase = 0.5 * (1.0 - math.cos(t))  # 0..1
            target = lower + phase * (upper - lower)
            robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        t += sim_dt
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
