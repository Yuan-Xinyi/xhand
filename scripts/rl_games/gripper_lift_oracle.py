"""Deterministic reach--close--lift feasibility check for the gripper ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Pick-Tool-Gripper-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--pad_height", type=float, default=0.060)
parser.add_argument("--target_force", type=float, default=0.5)
parser.add_argument("--output", default="/tmp/gripper_lift_oracle.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_mul
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401


def _limit_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    return value * torch.clamp(limit / norm, max=1.0)


@torch.inference_mode()
def main() -> None:
    print("[gripper-oracle] entering main", flush=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 0
    cfg.episode_length_s = 120.0
    cfg.success_hold_steps = 100000
    cfg.unlatched_lift_failure_height = 1000.0
    cfg.gripper_terminate_force = 10000.0
    # The oracle establishes physical feasibility at the canonical pose. PPO still trains
    # with the task's full position/yaw and arm-reset randomization.
    cfg.reset_object_pos_noise = (0.0, 0.0)
    cfg.reset_object_yaw_range = (0.0, 0.0)
    cfg.reset_arm_joint_noise = 0.0
    env = gym.make(args.task, cfg=cfg, render_mode=None)
    u = env.unwrapped
    obs, _ = env.reset(seed=0)
    del obs

    n, dev = u.num_envs, u.device
    arm_ids = torch.as_tensor(u._arm_joint_ids, device=dev)
    eye6 = torch.eye(6, device=dev).unsqueeze(0).expand(n, -1, -1)
    palm_jac_idx = u.palm_idx - 1  # fixed-base articulation omits the root body Jacobian
    start_quat = u.robot.data.body_quat_w[:, u.palm_idx].clone()
    handle_axis_local = torch.tensor(cfg.handle_axis, device=dev).expand(n, -1)
    handle_axis_w = quat_apply(u.object.data.root_quat_w, handle_axis_local)
    handle_yaw = torch.atan2(handle_axis_w[:, 1], handle_axis_w[:, 0])
    yaw_quat = torch.zeros((n, 4), device=dev)
    yaw_quat[:, 0] = torch.cos(0.5 * handle_yaw)
    yaw_quat[:, 3] = torch.sin(0.5 * handle_yaw)
    # At home the gripper's local +Y (finger depth) points along world +X. Align it with the
    # resting handle so both jaws touch together instead of one jaw sweeping the hammer away.
    fixed_quat = quat_mul(yaw_quat, start_quat)
    desired_mid = u.handle_center_w.clone()
    desired_mid[:, 2] = args.pad_height
    desired_mid[:, 1] += 0.0035  # compensate the aligned wrist's staged-approach lateral residual
    grip_action = -1.0
    max_force = torch.zeros((n, 2), device=dev)
    max_min_force = torch.zeros(n, device=dev)
    max_clearance = torch.full((n,), -1.0, device=dev)
    first_contact_step = -1

    def action_for_midpoint(target_mid: torch.Tensor) -> torch.Tensor:
        u._compute_intermediate_values()
        midpoint = u.finger_pad_w.mean(dim=1)
        palm_pos = u.robot.data.body_pos_w[:, u.palm_idx]
        desired_palm = palm_pos + target_mid - midpoint
        pos_error, rot_error = compute_pose_error(
            palm_pos,
            u.robot.data.body_quat_w[:, u.palm_idx],
            desired_palm,
            fixed_quat,
            rot_error_type="axis_angle",
        )
        delta = torch.cat((_limit_norm(pos_error, 0.005), _limit_norm(rot_error, 0.04)), dim=-1)
        jac = u.robot.root_physx_view.get_jacobians()
        jac = jac[:, palm_jac_idx, :, :][:, :, arm_ids]
        jt = jac.transpose(1, 2)
        dq = (jt @ torch.linalg.solve(jac @ jt + 0.05**2 * eye6, delta.unsqueeze(-1))).squeeze(-1)
        dq.clamp_(-0.04, 0.04)
        action = torch.zeros((n, 8), device=dev)
        action[:, :7] = (
            dq / (cfg.act_moving_average * cfg.action_scale)
        ).clamp(-1.0, 1.0)
        action[:, 7] = grip_action
        return action

    def step(target_mid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal max_force, max_min_force, max_clearance
        _, _, terminated, truncated, _ = env.step(action_for_midpoint(target_mid))
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError("unexpected reset during gripper feasibility oracle")
        force = u._jaw_forces()
        clearance = u._object_true_min_z() - u._table_surface_z
        max_force = torch.maximum(max_force, force)
        max_min_force = torch.maximum(max_min_force, force.min(dim=1).values)
        max_clearance = torch.maximum(max_clearance, clearance)
        return force, clearance

    # First align at a safe hover, then descend vertically. A direct diagonal path sweeps one
    # long finger through the hammer head and creates a meaningless impact spike.
    hover_mid = desired_mid.clone()
    hover_mid[:, 2] = 0.16
    for _ in range(220):
        step(hover_mid)
    for descend_step in range(160):
        target = desired_mid.clone()
        target[:, 2] = 0.16 + (args.pad_height - 0.16) * (descend_step + 1) / 160.0
        step(target)
    for _ in range(100):
        step(desired_mid)

    approach_max_force = max_force.clone()
    approach_mid = u.finger_pad_w.mean(dim=1).clone()
    approach_error_vec = approach_mid - desired_mid
    approach_error = torch.norm(approach_error_vec, dim=-1)
    object_before_close = u.object.data.root_pos_w.clone()

    # Close slowly and stop increasing the command at the first bilateral touch. Track small
    # object shifts while closing, then use tactile feedback to build force without over-closing.
    contact_force = torch.zeros((n, 2), device=dev)
    for close_step in range(241):
        grip_action = -1.0 + 2.0 * close_step / 240.0
        contact_force, _ = step(desired_mid)
        if bool((contact_force.min(dim=1).values >= args.target_force).all()):
            first_contact_step = close_step
            break

    # A low-gain force servo compensates contact compliance and lets the grasp latch settle.
    for _ in range(420):
        min_force = float(contact_force.min())
        max_contact_force = float(contact_force.max())
        if min_force < 10.0 and max_contact_force < 50.0:
            grip_action = min(grip_action + 0.002, 1.0)
        elif max_contact_force > 60.0:
            grip_action = max(grip_action - 0.002, -1.0)
        contact_force, _ = step(desired_mid)

    close_displacement = torch.norm(
        u.object.data.root_pos_w - object_before_close, dim=-1
    )
    grasped_after_close = u._is_grasped.clone()
    close_max_force = max_force.clone()

    # Lift the same grasp by 24 cm, with a short hold at the top.
    start_mid = desired_mid.clone()
    for lift_step in range(280):
        alpha = (lift_step + 1) / 280.0
        target = start_mid.clone()
        target[:, 2] += 0.24 * alpha
        step(target)
    top_mid = start_mid.clone()
    top_mid[:, 2] += 0.24
    for _ in range(40):
        contact_force, clearance = step(top_mid)

    result = {
        "task": args.task,
        "num_envs": n,
        "pad_height_m": args.pad_height,
        "finger_friction": float(cfg.fingertip_friction),
        "object_scene_friction": 0.5,
        "handle_yaw_rad": handle_yaw.cpu().tolist(),
        "approach_error_m": approach_error.cpu().tolist(),
        "approach_error_xyz_m": approach_error_vec.cpu().tolist(),
        "approach_midpoint_m": approach_mid.cpu().tolist(),
        "desired_midpoint_m": desired_mid.cpu().tolist(),
        "first_contact_close_step": first_contact_step,
        "held_gripper_action": grip_action,
        "close_object_displacement_m": close_displacement.cpu().tolist(),
        "grasped_after_close": grasped_after_close.cpu().tolist(),
        "grasped_at_end": u._is_grasped.cpu().tolist(),
        "final_force_n": contact_force.cpu().tolist(),
        "max_force_n": max_force.cpu().tolist(),
        "approach_max_force_n": approach_max_force.cpu().tolist(),
        "close_max_force_n": close_max_force.cpu().tolist(),
        "max_simultaneous_min_force_n": max_min_force.cpu().tolist(),
        "final_finger_joint_pos_m": u.robot.data.joint_pos[:, u._finger_ids_t].cpu().tolist(),
        "final_object_root_m": u.object.data.root_pos_w.cpu().tolist(),
        "final_clearance_m": clearance.cpu().tolist(),
        "max_clearance_m": max_clearance.cpu().tolist(),
        "pass_20cm": ((max_clearance >= 0.20) & u._is_grasped).cpu().tolist(),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    if not bool(((max_clearance >= 0.20) & u._is_grasped).all()):
        raise RuntimeError("gripper failed the physical 20cm lift check")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
