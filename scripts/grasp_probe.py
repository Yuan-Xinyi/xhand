# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted grasp-and-hold probe -- can this pen be physically grasped & held at all?

No policy. We hand-script the worst-case-favourable grasp and check whether the closed
hand holds the pen against gravity:

  Phase 1 (CLOSE, pen pinned): pin the pen at the palm-center (re-pinned every step) and
     curl the fingers shut around it.
  Phase 2 (HOLD, pen released): stop pinning, freeze the finger targets and the arm, and
     watch the pen. If the closed hand grips it, the pen stays near the palm; if the grasp
     is too slippery / geometrically impossible, the pen drops away (free-fall).

If even this favourable scripted grasp cannot hold the pen, NO reward shaping will -- the
fix is then physics (friction / contact offsets), not RL. If it DOES hold, the training
plateau is an exploration / reward problem (in-hand reset bootstrap, envelope approach).

Run (headless):
  python scripts/grasp_probe.py --task Pick-Repose-Pen-Direct-v0 --num_envs 32 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted grasp-and-hold probe.")
parser.add_argument("--task", type=str, default="Pick-Repose-Pen-Direct-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--close_steps", type=int, default=60, help="steps spent closing the fingers (pen pinned)")
parser.add_argument("--hold_steps", type=int, default=100, help="steps spent holding (pen released)")
parser.add_argument("--close_frac", type=float, default=0.8, help="fraction toward the joint limit when fully closed")
parser.add_argument("--pen_orient", type=str, default="across", choices=["across", "up"],
                    help="'across' = pen laid across the palm (probe-validated); 'up' = pen vertical (big end up)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym

from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    dev = base.device
    N = base.num_envs

    jnames = base.robot.joint_names
    finger_keys = ("index", "mid", "ring", "pinky", "thumb")
    hand_mask = torch.tensor([any(k in n for k in finger_keys) for n in jnames], dtype=torch.bool, device=dev)
    # curl target = a fraction from the default pose toward the UPPER joint limit
    closed = base.default_joint_pos.clone()
    closed[:, hand_mask] = (
        base.default_joint_pos[:, hand_mask]
        + args_cli.close_frac * (base.dof_upper[:, hand_mask] - base.default_joint_pos[:, hand_mask])
    )

    print("\n" + "=" * 70)
    print(f"[CFG] task={args_cli.task}  envs={N}  close_steps={args_cli.close_steps}  hold_steps={args_cli.hold_steps}")
    print(f"[CFG] pen contact_offset={base.cfg.object_cfg.spawn.collision_props.contact_offset}"
          f"  rest_offset={base.cfg.object_cfg.spawn.collision_props.rest_offset}")
    print("=" * 70 + "\n")

    env.reset()

    def palm_pen_pose():
        base._compute_intermediate_values()
        pose = torch.zeros((N, 7), device=dev)
        pose[:, :3] = base.palm_center_w
        if args_cli.pen_orient == "up":
            # pen VERTICAL, big end up: pen long axis (local z) -> world +Z is identity quat
            pose[:, 3] = 1.0
        else:
            # lay the pen across the palm: long axis (local z) along the palm's local X
            from isaaclab.utils.math import quat_from_angle_axis

            palm_q = base.robot.data.body_quat_w[:, base.palm_idx]
            across = quat_apply(palm_q, torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(N, 1))
            z = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(N, 1)
            cr = torch.cross(z, across, dim=-1)
            crn = cr.norm(dim=-1, keepdim=True)
            ax = torch.where(crn > 1e-6, cr / crn.clamp(min=1e-6), z)
            ang = torch.acos((z * across).sum(-1).clamp(-1, 1))
            pose[:, 3:7] = quat_from_angle_axis(ang, ax)
        return pose

    with torch.inference_mode():
        # PHASE 1: close fingers while pinning the pen at the palm center
        for t in range(args_cli.close_steps):
            frac = min(1.0, (t + 1) / args_cli.close_steps)
            base.dof_targets = base.default_joint_pos + frac * (closed - base.default_joint_pos)
            pose = palm_pen_pose()
            base.object.write_root_pose_to_sim(pose[:, :7], base.robot._ALL_INDICES)
            base.object.write_root_velocity_to_sim(torch.zeros((N, 6), device=dev), base.robot._ALL_INDICES)
            env.step(2.0 * torch.zeros((N, base.cfg.action_space), device=dev))

        # record release state
        base._compute_intermediate_values()
        z_release = base.object_pos_w[:, 2].clone()
        palm_at_release = base.palm_center_w.clone()

        # PHASE 2: hold (no re-pin, frozen finger targets, no arm motion)
        z_track = []
        for t in range(args_cli.hold_steps):
            env.step(torch.zeros((N, base.cfg.action_space), device=dev))
            base._compute_intermediate_values()
            z_track.append(base.object_pos_w[:, 2].clone())

        z_final = base.object_pos_w[:, 2]
        dist_final = torch.norm(base.object_pos_w - palm_at_release, dim=-1)
        drop = z_release - z_final  # how far the pen fell during the hold

    held = (drop < 0.05) & (dist_final < 0.06)
    free_fell = drop > 0.20
    print("=" * 70)
    print(f"GRASP-AND-HOLD PROBE  ({N} envs, {args_cli.hold_steps} hold steps ~ {args_cli.hold_steps*0.02:.1f}s)")
    print("=" * 70)
    print(f"  pen z at release : mean {z_release.mean():.3f} m")
    print(f"  pen z after hold : mean {z_final.mean():.3f} m")
    print(f"  pen drop during hold : mean {drop.mean()*1000:.0f} mm   max {drop.max()*1000:.0f} mm")
    print(f"  pen dist from palm   : mean {dist_final.mean()*1000:.0f} mm")
    print(f"  HELD (drop<5cm & near palm) : {int(held.sum())}/{N} envs  ({100*held.float().mean():.0f}%)")
    print(f"  FREE-FELL (drop>20cm)       : {int(free_fell.sum())}/{N} envs  ({100*free_fell.float().mean():.0f}%)")
    print("-" * 70)
    if held.float().mean() > 0.5:
        print("VERDICT: GRASPABLE -- the closed hand holds the pen. Plateau is EXPLORATION/REWARD.")
    elif free_fell.float().mean() > 0.5:
        print("VERDICT: NOT HELD -- pen slips/falls out. Fix PHYSICS (friction / contact) first.")
    else:
        print("VERDICT: MARGINAL -- grasp is weak. Likely needs friction + better grasp pose.")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
