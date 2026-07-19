# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Probe the arm Jacobian: verify dq = J_arm^pinv . [0,0,vz,0,0,0] actually moves the palm +z.
De-risks the task-space arm control before wiring it into _pre_physics_step."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=4)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    r = u.robot
    palm_idx = u.palm_idx
    arm_ids = u._arm_joint_ids
    print(f"palm_idx(body)={palm_idx}  n_bodies={len(r.body_names)}  arm_joint_ids={list(arm_ids)}  n_dof={r.num_joints}")
    J_full = r.root_physx_view.get_jacobians()  # (N, n_jac_bodies, 6, n_dof)
    print(f"jacobians shape = {tuple(J_full.shape)}")
    # fixed base -> jacobian body index = body_idx - 1 (base row dropped). try both, pick the one that works.
    for jidx in (palm_idx, palm_idx - 1):
        if jidx < 0 or jidx >= J_full.shape[1]:
            continue
        J = J_full[:, jidx, :, :][:, :, list(arm_ids)]  # (N, 6, 7)
        # damped pinv, command +z linear velocity
        vz = torch.zeros((u.num_envs, 6), device=u.device); vz[:, 2] = 0.05
        lam = 0.05
        JT = J.transpose(1, 2)
        dq = JT @ torch.linalg.solve(J @ JT + (lam**2) * torch.eye(6, device=u.device), vz.unsqueeze(-1))
        dq = dq.squeeze(-1)  # (N, 7)
        # apply as joint target delta over a few steps, watch palm z
        u._compute_intermediate_values() if hasattr(u, "_compute_intermediate_values") else None
        z0 = u.palm_center_w[:, 2].clone()
        tgt = r.data.joint_pos.clone()
        for _ in range(15):
            tgt[:, arm_ids] += dq
            r.set_joint_position_target(tgt)
            r.write_data_to_sim(); u.sim.step(); r.update(u.sim.get_physics_dt())
        u._compute_intermediate_values()
        dz = (u.palm_center_w[:, 2] - z0).mean().item()
        print(f"[jac idx {jidx}] cmd +z 0.05 x15steps -> mean palm dz = {dz*100:+.2f} cm  "
              f"{'<-- CORRECT (+z lifts)' if dz > 0.02 else '(no/opposite)'}")
        env.reset()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
