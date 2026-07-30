# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted grasp-and-hold probe for the pick_hammer_token object (048_hammer).

Hammer analog of ``grasp_probe.py``: pin the hammer with its GRIP CENTROID at the palm
center and the handle axis across the palm, curl the fingers shut, release, and watch
whether the closed hand bears the weight.  Besides the held/fell verdict it samples the
environment's own ``_compute_grasp_signals`` during the hold, which is exactly the shared
robust-grasp quality used by the latch -- the printed quality percentiles calibrate
``grasp_quality_high/low`` for the new grip geometry (the tool thresholds 0.35/0.20 were
measured on a different handle).

Run (headless):
  python scripts/hammer_grasp_probe.py --num_envs 32 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted hammer grasp-and-hold probe.")
parser.add_argument("--task", type=str, default="Pick-Hammer-Token-Direct-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--close_steps", type=int, default=60)
parser.add_argument("--hold_steps", type=int, default=100)
parser.add_argument(
    "--variants",
    type=str,
    default="0.0:0.8,0.07:0.8,0.07:0.95,0.09:0.9",
    help="comma-separated grip_axial_offset:close_frac pairs; the offset shifts the palm "
    "pin point along the handle axis toward the head (choke-up, +m).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import xhand_inhand.tasks  # noqa: F401, E402


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

    grip_center = torch.tensor(base.cfg.handle_center, dtype=torch.float, device=dev)
    handle_axis = torch.tensor(base.cfg.handle_axis, dtype=torch.float, device=dev)
    handle_axis = handle_axis / handle_axis.norm()

    variants = []
    for spec in args_cli.variants.split(","):
        off, frac = spec.split(":")
        variants.append((float(off), float(frac)))

    print("\n" + "=" * 70, flush=True)
    print(f"[CFG] task={args_cli.task}  envs={N}  close={args_cli.close_steps}  hold={args_cli.hold_steps}  variants={variants}", flush=True)
    print("=" * 70 + "\n", flush=True)

    def rotation_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Minimal quaternion rotating batched unit vector a onto b."""
        cr = torch.cross(a, b, dim=-1)
        crn = cr.norm(dim=-1, keepdim=True)
        fallback = torch.tensor([0.0, 0.0, 1.0], device=a.device).expand_as(a)
        ax = torch.where(crn > 1e-6, cr / crn.clamp(min=1e-6), fallback)
        ang = torch.acos((a * b).sum(-1).clamp(-1.0, 1.0))
        return quat_from_angle_axis(ang, ax)

    def palm_hammer_pose(pin_point: torch.Tensor):
        base._compute_intermediate_values()
        palm_q = base.robot.data.body_quat_w[:, base.palm_idx]
        across = quat_apply(palm_q, torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(N, 1))
        q_obj = rotation_between(handle_axis.expand(N, 3), across)
        pose = torch.zeros((N, 7), device=dev)
        # object origin so that the rotated pin point lands at the palm center
        pose[:, :3] = base.palm_center_w - quat_apply(q_obj, pin_point.expand(N, 3))
        pose[:, 3:7] = q_obj
        return pose

    def run_variant(axial_offset: float, close_frac: float):
        closed = base.default_joint_pos.clone()
        closed[:, hand_mask] = (
            base.default_joint_pos[:, hand_mask]
            + close_frac * (base.dof_upper[:, hand_mask] - base.default_joint_pos[:, hand_mask])
        )
        pin_point = grip_center + axial_offset * handle_axis
        q_hist, hold_hist, force_hist = [], [], []
        with torch.inference_mode():
            env.reset()
            for t in range(args_cli.close_steps):
                frac = min(1.0, (t + 1) / args_cli.close_steps)
                base.dof_targets = base.default_joint_pos + frac * (closed - base.default_joint_pos)
                pose = palm_hammer_pose(pin_point)
                base.object.write_root_pose_to_sim(pose[:, :7], base.robot._ALL_INDICES)
                base.object.write_root_velocity_to_sim(torch.zeros((N, 6), device=dev), base.robot._ALL_INDICES)
                env.step(torch.zeros((N, base.cfg.action_space), device=dev))

            base._compute_intermediate_values()
            z_release = base.object_pos_w[:, 2].clone()
            palm_at_release = base.palm_center_w.clone()

            for t in range(args_cli.hold_steps):
                env.step(torch.zeros((N, base.cfg.action_space), device=dev))
                base._compute_intermediate_values()
                if t >= 10:  # skip the release transient
                    sig = base._compute_grasp_signals()
                    q_hist.append(sig["quality"].clone())
                    hold_hist.append(sig["hold_quality"].clone())
                    force_hist.append(sig["force_magnitude"].max(dim=-1).values.clone())

            z_final = base.object_pos_w[:, 2]
            dist_final = torch.norm(base.object_pos_w - palm_at_release, dim=-1)
            drop = z_release - z_final

        held = (drop < 0.05) & (dist_final < 0.08)
        free_fell = drop > 0.20
        q_mean = torch.stack(q_hist).mean(dim=0)
        f_all = torch.stack(force_hist)

        print("-" * 70, flush=True)
        print(f"VARIANT choke-up=+{axial_offset*100:.0f}cm  close_frac={close_frac}", flush=True)
        print(
            f"  drop mean {drop.mean()*1000:.0f}mm max {drop.max()*1000:.0f}mm   "
            f"dist-from-palm mean {dist_final.mean()*1000:.0f}mm",
            flush=True,
        )
        print(
            f"  HELD {int(held.sum())}/{N} ({100*held.float().mean():.0f}%)   "
            f"FREE-FELL {int(free_fell.sum())}/{N}",
            flush=True,
        )
        if held.any():
            qh = q_mean[held]
            print(
                f"  quality over HELD: mean {qh.mean():.3f}  p10 {qh.quantile(0.10):.3f}  "
                f"p50 {qh.quantile(0.50):.3f}  p90 {qh.quantile(0.90):.3f}   "
                f"peak force mean {f_all.mean():.1f}N max {f_all.max():.1f}N",
                flush=True,
            )
        return held.float().mean().item()

    print("=" * 70, flush=True)
    print(f"HAMMER GRASP-AND-HOLD PROBE  ({N} envs/variant, hold {args_cli.hold_steps * 0.02:.1f}s)", flush=True)
    print(f"  cfg grasp_quality_high={base.cfg.grasp_quality_high}  low={base.cfg.grasp_quality_low}", flush=True)
    best = max(run_variant(off, frac) for off, frac in variants)
    print("=" * 70, flush=True)
    verdict = "GRASPABLE" if best > 0.5 else ("MARGINAL" if best > 0.1 else "NOT HELD")
    print(f"VERDICT (best variant held {100*best:.0f}%): {verdict}", flush=True)
    print("=" * 70, flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
