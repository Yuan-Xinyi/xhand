"""Headless sanity check: spawn XARM7_XHAND_CFG and print its joint/body layout.

Run:
  conda activate env_isaaclab
  python validate_asset.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from xhand_inhand.robots import XARM7_XHAND_CFG  # noqa: E402


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cpu"))
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg())

    robot = Articulation(XARM7_XHAND_CFG.replace(prim_path="/World/Robot"))

    sim.reset()
    print("\n=========== XARM7_XHAND asset report ===========")
    print(f"num joints (DOF): {robot.num_joints}")
    print(f"joint names ({len(robot.joint_names)}): {robot.joint_names}")
    print(f"num bodies: {robot.num_bodies}")
    print(f"body names ({len(robot.body_names)}): {robot.body_names}")
    # confirm the fingertip + palm bodies the task will reference exist
    for name in ["palm", "index_rota_link2", "mid_link2", "ring_link2", "pinky_link2", "thumb_rota_link2", "link7", "link8"]:
        print(f"  body present: {name:>20} -> {name in robot.body_names}")
    # step a bit so we know physics is stable
    for _ in range(60):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
    print("stepped 60 frames OK; default joint pos:", robot.data.default_joint_pos[0].tolist())
    print("================================================\n")
    simulation_app.close()


if __name__ == "__main__":
    main()
