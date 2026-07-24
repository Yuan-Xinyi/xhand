# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Hybrid controller: PPO reaches and grasps, then a DLS IK lifts the tool to 20 cm.

Motivation: the end-to-end PPO can form a stable grasp on some envs but frequently fails to
raise the tool (the "grasps but won't lift" gap).  Instead of learning the lift, we hand each
env off, the instant its grasp latches, to the same GPU damped-least-squares Jacobian IK used by
the scripted oracle -- the arm servos the palm straight up while the hand target is frozen at the
grasp pose (a distal grip servo keeps fingertip contact under load).  We then measure how many
latched grasps IK converts into strict 20 cm successes.

Per-env asynchronous handover on one physics timeline (no snapshot restore):

  GRASP (phase 0): the 115/21 actor drives the env.  When the env's robust-grasp latch
      (is_grasped & grasp_quality>=high & hold_quality>=min & force<=limit) holds for
      ``--grasp_confirm`` consecutive frames, capture the palm body pose and freeze the current
      joint target; the env switches to LIFT.
  LIFT (phase 1): DLS IK raises the captured palm pose by ``--lift_height`` over ``--lift_ramp``
      steps and holds; ``dof_targets`` for this env is overridden with the IK arm + frozen hand.

Strict success = true mesh clearance >= lift_success_height with the grasp latch, quality, force
safety and low object speed held for ``--stable_steps`` consecutive frames.  Force cutoffs and the
built-in success/drop resets are disabled so each env is a single attempt.

``--attempts N`` (with ``--num_envs 1``) runs N single-env episodes in one process, resetting
between them, and saves an mp4 for each that reaches a strict success -- clean single-env clips.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="PPO grasp + DLS IK lift hybrid.")
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--grasp_deadline", type=int, default=500, help="steps allowed to reach a grasp")
parser.add_argument("--grasp_confirm", type=int, default=8, help="consecutive latch frames to hand off")
parser.add_argument("--lift_ramp", type=int, default=240)
parser.add_argument("--hold_steps", type=int, default=60)
parser.add_argument("--stable_steps", type=int, default=15)
parser.add_argument("--lift_height", type=float, default=0.22)
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--max_cart_step", type=float, default=0.004)
parser.add_argument("--max_rot_step", type=float, default=0.05)
parser.add_argument("--max_joint_step", type=float, default=0.04)
# Distal grip servo: maintain fingertip contact force while the arm lifts, so a formed grasp is
# not lost under lift load (the frozen hand target alone lets the tool slide out).
parser.add_argument("--grip_force_target", type=float, default=3.0)
parser.add_argument("--grip_force_limit", type=float, default=20.0)
parser.add_argument("--grip_servo_step", type=float, default=0.006)
parser.add_argument("--grip_servo_range", type=float, default=0.60)
parser.add_argument("--no_grip_servo", action="store_true", help="freeze the hand entirely (ablation)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=str, default="/tmp/pick_tool_ppo_grasp_ik_lift.json")
parser.add_argument("--video", action="store_true", help="record an rgb_array video of the run")
parser.add_argument("--video_folder", type=str, default="/tmp/pick_tool_ik_lift_video")
parser.add_argument("--video_length", type=int, default=0, help="0 = record the whole run")
# Demo mode: run many single-env attempts in one process (reset between), record each and keep
# only the ones that reach a strict 20cm success -- clean, clearly framed single-env clips.
parser.add_argument("--attempts", type=int, default=1, help=">1 enables single-env demo recording")
parser.add_argument("--max_clips", type=int, default=5, help="stop after this many successful clips")
parser.add_argument("--fps", type=int, default=50)
# Render camera (world frame).  The tool sits at ~(0.5, 0, 0.05) and lifts to ~0.25; the default
# viewer sits far away, so frame the tabletop grasp zone up close for recording.
parser.add_argument("--cam_eye", type=float, nargs=3, default=[1.9, 0.95, 0.9])
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.5, 0.0, 0.32])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video or args_cli.attempts > 1:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import compute_pose_error
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401
from bc_pick_tool import MigratedActor, clone_state, load_torch


def _checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    raw = load_torch(path)
    if not isinstance(raw, dict):
        raise TypeError("checkpoint root is not a dictionary")
    payload = raw if isinstance(raw.get("model"), dict) else (raw[0] if 0 in raw else raw.get("0"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise KeyError("checkpoint must contain {'model': state_dict}, optionally below key 0")
    return clone_state(payload["model"])


def _limit_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    return value * torch.clamp(limit / norm, max=1.0)


def _summary(value: torch.Tensor) -> dict[str, float] | None:
    flat = value.detach().float().flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return None
    q = torch.quantile(flat, torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), device=flat.device))
    return dict(zip(("min", "p10", "median", "p90", "max"), (float(x) for x in q), strict=True))


def _save_clip(path: str, frames: list, fps: int) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=fps, macro_block_size=None)


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args_cli.seed)
    n = args_cli.num_envs
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    cfg.seed = args_cli.seed
    # Pull the render camera in close on the tabletop grasp/lift zone (world frame, single env).
    cfg.viewer.eye = tuple(args_cli.cam_eye)
    cfg.viewer.lookat = tuple(args_cli.cam_lookat)
    cfg.viewer.origin_type = "world"
    cfg.episode_length_s = 120.0
    cfg.terminate_on_drop = False
    cfg.success_hold_steps = 100000
    cfg.tactile_terminate_steps = 1_000_000_000
    cfg.tactile_hard_terminate_steps = 1_000_000_000
    # The IK integrates joint targets directly; cap its per-step arm increment at the policy's
    # realizable value (act_moving_average*action_scale) so the lift stays action-space honest.
    realizable_arm_step = float(cfg.act_moving_average * cfg.action_scale)
    max_joint_step = min(args_cli.max_joint_step, realizable_arm_step)

    total_steps = args_cli.grasp_deadline + args_cli.lift_ramp + args_cli.hold_steps
    demo = args_cli.attempts > 1  # single-env: keep only successful clips
    record = args_cli.video or demo
    env = gym.make(args_cli.task, cfg=cfg, render_mode="rgb_array" if record else None)
    if args_cli.video and not demo:
        video_length = args_cli.video_length if args_cli.video_length > 0 else total_steps
        os.makedirs(args_cli.video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_folder,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix="ppo_grasp_ik_lift",
            disable_logger=True,
        )
        print(f"[video] recording {video_length} steps to {args_cli.video_folder}", flush=True)
    if demo:
        os.makedirs(args_cli.video_folder, exist_ok=True)
    u = env.unwrapped
    dev = u.device
    actor = MigratedActor(_checkpoint_model(Path(args_cli.checkpoint))).to(dev).eval()
    if actor.observation_dim not in (115, 120) or actor.action_dim != 21:
        raise RuntimeError(f"expected a 115-or-120/21 actor, got {actor.observation_dim}/{actor.action_dim}")

    # Fingertip -> distal-flexion joint routing for the grip servo (articulation hand order).
    env.reset()
    hand_names = [u.robot.joint_names[i] for i in u._hand_ids_t.tolist()]
    distal_by_fingertip = {
        "thumb_rota_link2": "thumb_joint2",
        "index_rota_link2": "index_joint2",
        "mid_link2": "middle_joint1",
        "ring_link2": "ring_joint1",
        "pinky_link2": "pinky_joint1",
    }
    servo_hand_ids = u._hand_ids_t[
        torch.tensor([hand_names.index(distal_by_fingertip[name]) for name in u.ee_names], device=dev)
    ]

    # Shared with the dof_targets override closure below; every other controller tensor is local
    # to run_attempt so a fresh episode starts from a clean state.
    commanded = u.dof_targets.detach().clone()
    lift_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    eye6 = torch.eye(6, device=dev).unsqueeze(0)

    original_pre_physics = u._pre_physics_step

    def hybrid_pre_physics(actions: torch.Tensor) -> None:
        original_pre_physics(actions)
        if bool(lift_mask.any()):
            u.dof_targets[lift_mask] = commanded[lift_mask]

    u._pre_physics_step = hybrid_pre_physics

    def run_attempt(capture: bool) -> dict:
        nonlocal lift_mask
        obs, _ = env.reset()
        phase = torch.zeros(n, dtype=torch.long, device=dev)
        confirm = torch.zeros(n, dtype=torch.long, device=dev)
        commanded.copy_(u.dof_targets)
        palm_start = torch.zeros((n, 3), device=dev)
        palm_quat_target = torch.zeros((n, 4), device=dev)
        servo_lower = torch.zeros((n, len(u.ee_names)), device=dev)
        servo_upper = torch.zeros((n, len(u.ee_names)), device=dev)
        lift_counter = torch.zeros(n, dtype=torch.long, device=dev)
        lift_mask = phase == 1
        stable_count = torch.zeros(n, dtype=torch.long, device=dev)
        success = torch.zeros(n, dtype=torch.bool, device=dev)
        success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        ever_grasped = torch.zeros(n, dtype=torch.bool, device=dev)
        grasp_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        max_clearance = torch.full((n,), -float("inf"), device=dev)
        force_peak = torch.zeros(n, device=dev)
        frames: list = []

        for step in range(total_steps):
            u._compute_intermediate_values()
            signals = u._compute_grasp_signals()
            force_max = signals["force_magnitude"].max(dim=-1).values
            latched = (
                u._is_grasped
                & (signals["grasp_quality"] >= u.cfg.grasp_quality_high)
                & (signals["hold_quality"] >= u.cfg.close_option_min_hold_quality)
                & (force_max <= u.cfg.grasp_bonus_max_force)
            )

            # grasp handoff: confirm a sustained latch while still in the grasp phase
            grasping = phase == 0
            confirm = torch.where(grasping & latched, confirm + 1, torch.zeros_like(confirm))
            handoff = grasping & (confirm >= args_cli.grasp_confirm) & (step < args_cli.grasp_deadline)
            if bool(handoff.any()):
                phase[handoff] = 1
                lift_counter[handoff] = 0
                commanded[handoff] = u.dof_targets[handoff].detach().clone()
                palm_start[handoff] = u.robot.data.body_pos_w[handoff][:, u.palm_idx]
                palm_quat_target[handoff] = u.robot.data.body_quat_w[handoff][:, u.palm_idx]
                frozen_distal = u.dof_targets[handoff][:, servo_hand_ids]
                servo_lower[handoff] = frozen_distal
                servo_upper[handoff] = torch.minimum(
                    frozen_distal + args_cli.grip_servo_range, u.dof_upper[handoff][:, servo_hand_ids]
                )
                ever_grasped[handoff] = True
                grasp_step[handoff] = step

            # IK lift for every env already in the lift phase
            lift_mask = phase == 1
            if bool(lift_mask.any()):
                height = args_cli.lift_height * (
                    lift_counter.float() / float(args_cli.lift_ramp)
                ).clamp(0.0, 1.0)
                current_pos = u.robot.data.body_pos_w[:, u.palm_idx]
                current_quat = u.robot.data.body_quat_w[:, u.palm_idx]
                desired_pos = palm_start.clone()
                desired_pos[:, 2] += height
                pos_error, rot_error = compute_pose_error(
                    current_pos, current_quat, desired_pos, palm_quat_target, rot_error_type="axis_angle"
                )
                delta = torch.cat(
                    (_limit_norm(pos_error, args_cli.max_cart_step), _limit_norm(rot_error, args_cli.max_rot_step)),
                    dim=-1,
                )
                jacobian = u.robot.root_physx_view.get_jacobians()[:, u._palm_jac_idx, :, :][:, :, u._arm_ids_t]
                jt = jacobian.transpose(1, 2)
                solved = torch.linalg.solve(jacobian @ jt + (args_cli.damping**2) * eye6, delta.unsqueeze(-1))
                delta_q = (jt @ solved).squeeze(-1).clamp(-max_joint_step, max_joint_step)
                next_arm = torch.maximum(
                    torch.minimum(commanded[:, u._arm_ids_t] + delta_q, u.dof_upper[:, u._arm_ids_t]),
                    u.dof_lower[:, u._arm_ids_t],
                )
                commanded[:, u._arm_ids_t] = torch.where(
                    lift_mask.unsqueeze(-1), next_arm, commanded[:, u._arm_ids_t]
                )
                if not args_cli.no_grip_servo:
                    force = u._finger_object_force_magnitudes()
                    distal_cmd = commanded[:, servo_hand_ids]
                    delta_grip = args_cli.grip_servo_step * (
                        (force < args_cli.grip_force_target).float()
                        - (force > args_cli.grip_force_limit).float()
                    )
                    new_distal = torch.clamp(distal_cmd + delta_grip, servo_lower, servo_upper)
                    commanded[:, servo_hand_ids] = torch.where(lift_mask.unsqueeze(-1), new_distal, distal_cmd)
                lift_counter = torch.where(lift_mask, lift_counter + 1, lift_counter)

            action = actor(obs["policy"]).clamp(-1.0, 1.0)
            obs, _, _, _, _ = env.step(action)
            if capture:
                frames.append(env.render())

            slow = (
                (u.object.data.root_com_lin_vel_w.norm(dim=-1) < u.cfg.success_max_obj_lin_speed)
                & (u.object.data.root_com_ang_vel_w.norm(dim=-1) < u.cfg.success_max_obj_ang_speed)
            )
            post_signals = u._compute_grasp_signals()
            post_force = post_signals["force_magnitude"].max(dim=-1).values
            post_clear = u._object_true_min_z() - u._table_surface_z
            strict = (
                (phase == 1)
                & (post_clear >= u.cfg.lift_success_height)
                & u._is_grasped
                & (post_signals["grasp_quality"] >= u.cfg.grasp_quality_high)
                & (post_signals["hold_quality"] >= u.cfg.close_option_min_hold_quality)
                & (post_force <= u.cfg.grasp_bonus_max_force)
                & slow
            )
            stable_count = torch.where(strict, stable_count + 1, torch.zeros_like(stable_count))
            newly = (~success) & (stable_count >= args_cli.stable_steps)
            success_step[newly] = step
            success |= newly
            max_clearance = torch.maximum(max_clearance, torch.where(phase == 1, post_clear, max_clearance))
            force_peak = torch.maximum(force_peak, post_force)

        return {
            "grasped": int(ever_grasped.sum()),
            "lifted": int(success.sum()),
            "frames": frames,
            "grasp_step": _summary(grasp_step[ever_grasped].float()) if bool(ever_grasped.any()) else None,
            "success_step": _summary(success_step[success].float()) if bool(success.any()) else None,
            "max_true_clearance_lift_envs": _summary(max_clearance),
            "force_peak": _summary(force_peak),
        }

    base_metrics = {
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "num_envs": n,
        "seed": args_cli.seed,
        "total_steps": total_steps,
        "realizable_arm_step_rad": realizable_arm_step,
        "effective_max_joint_step": max_joint_step,
        "grip_servo": not args_cli.no_grip_servo,
        "grasp_confirm_frames": args_cli.grasp_confirm,
    }

    if not demo:
        r = run_attempt(capture=False)
        metrics = {
            **base_metrics,
            "grasped_count": r["grasped"],
            "grasped_rate": r["grasped"] / n,
            "ik_lift_success_count": r["lifted"],
            "ik_lift_success_rate": r["lifted"] / n,
            "recovery_of_grasps": (r["lifted"] / r["grasped"]) if r["grasped"] else 0.0,
            "grasp_step": r["grasp_step"],
            "success_step": r["success_step"],
            "max_true_clearance_lift_envs": r["max_true_clearance_lift_envs"],
            "force_peak": r["force_peak"],
        }
        Path(args_cli.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.output).write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        print(
            f"PPO grasp -> IK lift: grasped={r['grasped']}/{n} ({r['grasped']/n*100:.1f}%); "
            f"IK-lifted={r['lifted']}/{n} ({r['lifted']/n*100:.1f}%); "
            f"recovery={metrics['recovery_of_grasps']*100:.0f}% of grasps; "
            f"effective_arm_step={max_joint_step:.3f}",
            flush=True,
        )
        print(f"wrote {args_cli.output}", flush=True)
    else:
        saved: list[str] = []
        for attempt in range(args_cli.attempts):
            r = run_attempt(capture=True)
            ok = r["lifted"] >= 1
            if ok and len(saved) < args_cli.max_clips:
                path = os.path.join(args_cli.video_folder, f"single_env_success_{len(saved) + 1}.mp4")
                _save_clip(path, r["frames"], args_cli.fps)
                saved.append(path)
                print(f"[demo] attempt {attempt}: SUCCESS grasped={r['grasped']} lifted={r['lifted']} -> {path}", flush=True)
            else:
                tag = "success(not saved)" if ok else "no lift"
                print(f"[demo] attempt {attempt}: {tag} grasped={r['grasped']} lifted={r['lifted']}", flush=True)
            if len(saved) >= args_cli.max_clips:
                break
        print(f"[demo] saved {len(saved)} single-env success clip(s):", flush=True)
        for p in saved:
            print(f"  {p}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
