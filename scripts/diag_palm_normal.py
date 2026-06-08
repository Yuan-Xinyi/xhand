# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find the XHand palm-normal local axis = the direction the fingers curl toward."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply, quat_conjugate

import xhand_inhand.robots as robots


def tips_in_palm(robot, palm_idx, tip_ids):
    pq = robot.data.body_quat_w[:, palm_idx]
    pp = robot.data.body_pos_w[:, palm_idx]
    tips = robot.data.body_pos_w[:, tip_ids]  # (1,5,3)
    rel = tips - pp.unsqueeze(1)
    return quat_apply(quat_conjugate(pq).unsqueeze(1).repeat(1, len(tip_ids), 1), rel)[0]  # (5,3)


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 100))
    sim_utils.GroundPlaneCfg().func("/World/g", sim_utils.GroundPlaneCfg())
    robot = Articulation(robots.FR3_XHAND_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    palm_idx = robot.body_names.index("palm")
    tip_ids, _ = robot.find_bodies(["index_rota_link2", "mid_link2", "ring_link2", "pinky_link2", "thumb_rota_link2"])
    hand_ids, _ = robot.find_joints(["(thumb|index|middle|ring|pinky)_joint.*"])

    qpos = robot.data.default_joint_pos.clone()
    qvel = torch.zeros_like(qpos)

    # open
    robot.write_joint_state_to_sim(qpos, qvel)
    for _ in range(10):
        sim.step()
        robot.update(sim.get_physics_dt())
    open_p = tips_in_palm(robot, palm_idx, tip_ids)

    # flexed (close all hand joints toward +0.9)
    qpos2 = qpos.clone()
    qpos2[:, hand_ids] = 0.9
    robot.write_joint_state_to_sim(qpos2, qvel)
    for _ in range(10):
        sim.step()
        robot.update(sim.get_physics_dt())
    closed_p = tips_in_palm(robot, palm_idx, tip_ids)

    disp = (closed_p - open_p).mean(dim=0)  # mean curl direction in palm frame
    disp_n = disp / disp.norm()
    print(f"[PALM] mean fingertip curl displacement (palm frame) = {disp.cpu().numpy()}")
    print(f"[PALM] normalized (= palm-normal local axis)         = {disp_n.cpu().numpy()}")
    print("[PALM] -> the dominant +/- axis is the palm normal (direction fingers close toward).")
    simulation_app.close()


if __name__ == "__main__":
    main()
