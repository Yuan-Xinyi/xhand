# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a project robot ArticulationCfg, spawn it, step a few times, print info, exit.

    python scripts/check_robot.py --robot FR3_CFG --steps 20 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="FR3_CFG", help="cfg name in xhand_inhand.robots")
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

import xhand_inhand.robots as robots


def main():
    cfg = getattr(robots, args_cli.robot).replace(prim_path="/World/Robot")
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device))
    sim.set_camera_view(eye=[1.6, 1.6, 1.2], target=[0.0, 0.0, 0.4])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    robot = Articulation(cfg)
    sim.reset()
    print(f"[CHECK] robot = {args_cli.robot}")
    print(f"[CHECK] is_fixed_base = {robot.is_fixed_base}")
    print(f"[CHECK] num_joints = {robot.num_joints}")
    print(f"[CHECK] joint_names = {robot.joint_names}")
    print(f"[CHECK] num_bodies = {robot.num_bodies}")
    lo = robot.data.soft_joint_pos_limits[0, :, 0].tolist()
    hi = robot.data.soft_joint_pos_limits[0, :, 1].tolist()
    print(f"[CHECK] joint_limits_lower = {[round(v, 3) for v in lo]}")
    print(f"[CHECK] joint_limits_upper = {[round(v, 3) for v in hi]}")

    dt = sim.get_physics_dt()
    for _ in range(args_cli.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
    print("[CHECK] stepped OK -> PASS")

    simulation_app.close()


if __name__ == "__main__":
    main()
