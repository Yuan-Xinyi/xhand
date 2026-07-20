# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Validate close -> 2cm micro-lift -> 20cm true-clearance with a DLS arm oracle.

Input is the JSON emitted by :mod:`hand_space_feasibility`.  The saved pregrasp is restored with
zero velocity, the selected hand target is closed smoothly while the arm is fixed, and a damped
least-squares controller translates the palm body upward while holding its orientation.  Success
requires the same robust grasp latch/quality used by the task and the true mesh minimum clearing
the table by 20cm; palm/root height proxies are never used for the verdict.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Scripted robust-grasp and DLS lift oracle.")
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--feasibility_json", type=str, required=True)
parser.add_argument("--mode", choices=("token9", "raw12", "hybrid14"), default="hybrid14")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--close_steps", type=int, default=24)
parser.add_argument("--close_hold_steps", type=int, default=12)
parser.add_argument("--micro_height", type=float, default=0.02)
parser.add_argument("--target_height", type=float, default=0.22)
parser.add_argument("--micro_steps", type=int, default=40)
parser.add_argument("--micro_hold_steps", type=int, default=60)
parser.add_argument("--lift_steps", type=int, default=240)
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--max_cart_step", type=float, default=0.004)
parser.add_argument("--max_rot_step", type=float, default=0.05)
parser.add_argument("--max_joint_step", type=float, default=0.04)
parser.add_argument("--grip_force_target", type=float, default=3.0)
parser.add_argument("--grip_force_limit", type=float, default=20.0)
parser.add_argument("--grip_servo_step", type=float, default=0.006)
parser.add_argument("--grip_servo_range", type=float, default=0.60)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=str, default="/tmp/pick_tool_scripted_lift_oracle.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.target_height < 0.20:
    parser.error("--target_height must be at least 0.20m")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import compute_pose_error
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401


def _limit_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    return value * torch.clamp(limit / norm, max=1.0)


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().cpu()
    return {
        "min": float(value.min()),
        "p10": float(torch.quantile(value, 0.10)),
        "median": float(value.median()),
        "p90": float(torch.quantile(value, 0.90)),
        "max": float(value.max()),
    }


def main() -> None:
    torch.manual_seed(args_cli.seed)
    artifact_path = Path(args_cli.feasibility_json)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not artifact["results"][args_cli.mode].get("robust_grasp_pass", False):
        raise RuntimeError(f"{args_cli.mode} was not a robust-grasp pass in {artifact_path}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 120.0
    env_cfg.terminate_on_drop = False
    # Prevent the built-in success termination/reset; this script validates its own stable hold.
    env_cfg.success_hold_steps = 100000
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    dev = u.device
    num_envs = u.num_envs
    all_ids = u.robot._ALL_INDICES
    env.reset()

    saved_names = artifact["hand_joint_names"]
    hand_names = [u.robot.joint_names[i] for i in u._hand_ids_t.tolist()]
    if hand_names != saved_names:
        raise RuntimeError(f"hand joint order changed: artifact={saved_names}, runtime={hand_names}")
    snap_joint = torch.tensor(artifact["pregrasp"]["joint_pos"], device=dev)
    snap_obj_local = torch.tensor(artifact["pregrasp"]["object_local_pos"], device=dev)
    snap_obj_quat = torch.tensor(artifact["pregrasp"]["object_quat"], device=dev)
    hand_target = torch.tensor(artifact["results"][args_cli.mode]["target"], device=dev)
    if snap_joint.numel() != u.robot.num_joints or hand_target.numel() != len(hand_names):
        raise RuntimeError("artifact dimensions do not match the runtime articulation")

    commanded = snap_joint.unsqueeze(0).repeat(num_envs, 1)
    distal_by_fingertip = {
        "thumb_rota_link2": "thumb_joint2",
        "index_rota_link2": "index_joint2",
        "mid_link2": "middle_joint1",
        "ring_link2": "ring_joint1",
        "pinky_link2": "pinky_joint1",
    }
    servo_hand_ids = torch.tensor(
        [hand_names.index(distal_by_fingertip[name]) for name in u.ee_names],
        dtype=torch.long,
        device=dev,
    )
    servo_lower = hand_target[servo_hand_ids]
    servo_upper = torch.minimum(
        servo_lower + args_cli.grip_servo_range,
        u.dof_upper[0, u._hand_ids_t[servo_hand_ids]],
    )

    def oracle_pre_physics(self, actions: torch.Tensor) -> None:
        self.actions = torch.zeros_like(actions)
        self.dof_targets[:] = torch.clamp(commanded, self.dof_lower, self.dof_upper)

    u._pre_physics_step = types.MethodType(oracle_pre_physics, u)
    zero_action = torch.zeros((num_envs, u.cfg.action_space), device=dev)

    @torch.inference_mode()
    def restore_snapshot() -> None:
        joint = snap_joint.unsqueeze(0).repeat(num_envs, 1)
        u.robot.write_joint_state_to_sim(joint, torch.zeros_like(joint), env_ids=all_ids)
        u.robot.set_joint_position_target(joint, env_ids=all_ids)
        commanded[:] = joint
        u.dof_targets[:] = joint
        pose = torch.zeros((num_envs, 7), device=dev)
        pose[:, :3] = snap_obj_local + u.scene.env_origins
        pose[:, 3:7] = snap_obj_quat
        u.object.write_root_pose_to_sim(pose, env_ids=all_ids)
        u.object.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=dev), env_ids=all_ids)
        u.episode_length_buf.zero_()
        u._contact_steps.zero_()
        u._lost_contact_steps.zero_()
        u._is_grasped.zero_()
        u._grasp_bonus_given.zero_()
        u._success_paid.zero_()
        u._success_steps.zero_()
        u._is_success.zero_()
        u.actions.zero_()
        u.prev_actions.zero_()
        u._compute_intermediate_values()

    force_peak = torch.zeros((num_envs, len(u.ee_names)), device=dev)
    min_q_grasp = torch.ones(num_envs, device=dev)
    max_slip_lin = torch.zeros(num_envs, device=dev)
    max_slip_ang = torch.zeros(num_envs, device=dev)

    @torch.inference_mode()
    def step_and_measure() -> dict[str, torch.Tensor]:
        env.step(zero_action)
        signals = u._compute_grasp_signals()
        force = u._finger_object_force_magnitudes()
        clearance = u._object_true_min_z() - u._table_surface_z
        force_peak[:] = torch.maximum(force_peak, force)
        min_q_grasp[:] = torch.minimum(min_q_grasp, signals["grasp_quality"])
        max_slip_lin[:] = torch.maximum(max_slip_lin, signals["slip_lin"])
        max_slip_ang[:] = torch.maximum(max_slip_ang, signals["slip_ang"])
        return {"signals": signals, "force": force, "clearance": clearance}

    @torch.inference_mode()
    def update_grip_servo(force: torch.Tensor) -> None:
        """Tactile distal closure: gain contact without changing the token's proximal pose."""

        distal = commanded[:, u._hand_ids_t[servo_hand_ids]]
        increase = force < args_cli.grip_force_target
        decrease = force > args_cli.grip_force_limit
        delta = args_cli.grip_servo_step * (increase.float() - decrease.float())
        distal = torch.maximum(torch.minimum(distal + delta, servo_upper), servo_lower)
        commanded[:, u._hand_ids_t[servo_hand_ids]] = distal

    restore_snapshot()
    snap_hand = snap_joint[u._hand_ids_t]
    print(
        f"\n=== CLOSE ({args_cli.mode}, {num_envs} exact snapshot replicas) ===\n"
        f"force order={list(u.ee_names)}",
        flush=True,
    )
    close_state = None
    for step in range(args_cli.close_steps + args_cli.close_hold_steps):
        if step < args_cli.close_steps:
            x = float(step + 1) / float(args_cli.close_steps)
            blend = x * x * (3.0 - 2.0 * x)
        else:
            blend = 1.0
        commanded[:, u._hand_ids_t] = snap_hand + blend * (hand_target - snap_hand)
        close_state = step_and_measure()
        if step >= args_cli.close_steps:
            update_grip_servo(close_state["force"])
        if step % 8 == 0 or step == args_cli.close_steps + args_cli.close_hold_steps - 1:
            sig = close_state["signals"]
            print(
                f"close {step:3d}: latch={u._is_grasped.float().mean():.3f} "
                f"q_wrap={sig['quality'].mean():.3f} q_grasp={sig['grasp_quality'].mean():.3f} "
                f"clear={close_state['clearance'].mean():+.5f}m",
                flush=True,
            )

    close_signals = close_state["signals"]
    close_pass = u._is_grasped & (close_signals["grasp_quality"] >= u.cfg.grasp_quality_high)
    print(
        f"CLOSE RESULT: {int(close_pass.sum())}/{num_envs} robust grasps; "
        f"median q={close_signals['grasp_quality'].median():.3f}",
        flush=True,
    )
    if not bool(close_pass.any()):
        raise RuntimeError("scripted closure did not reproduce any robust grasp")

    # DLS controls the palm BODY frame. Keeping its orientation fixed also makes palm-center
    # translation identical, because the center is a fixed link-local offset.
    palm_body_start = u.robot.data.body_pos_w[:, u.palm_idx].detach().clone()
    palm_quat_target = u.robot.data.body_quat_w[:, u.palm_idx].detach().clone()
    palm_center_start_z = u.palm_center_w[:, 2].detach().clone()
    clearance_start = (u._object_true_min_z() - u._table_surface_z).detach().clone()
    eye6 = torch.eye(6, device=dev).unsqueeze(0)

    @torch.inference_mode()
    def dls_step(height: float) -> dict[str, torch.Tensor]:
        u._compute_intermediate_values()
        current_pos = u.robot.data.body_pos_w[:, u.palm_idx]
        current_quat = u.robot.data.body_quat_w[:, u.palm_idx]
        desired_pos = palm_body_start.clone()
        desired_pos[:, 2] += height
        pos_error, rot_error = compute_pose_error(
            current_pos, current_quat, desired_pos, palm_quat_target, rot_error_type="axis_angle"
        )
        delta = torch.cat(
            (
                _limit_norm(pos_error, args_cli.max_cart_step),
                _limit_norm(rot_error, args_cli.max_rot_step),
            ),
            dim=-1,
        )
        jacobian = u.robot.root_physx_view.get_jacobians()
        jacobian = jacobian[:, u._palm_jac_idx, :, :][:, :, u._arm_ids_t]
        jt = jacobian.transpose(1, 2)
        system = jacobian @ jt + (args_cli.damping**2) * eye6
        delta_q = (jt @ torch.linalg.solve(system, delta.unsqueeze(-1))).squeeze(-1)
        delta_q = delta_q.clamp(-args_cli.max_joint_step, args_cli.max_joint_step)
        current_arm = u.robot.data.joint_pos[:, u._arm_ids_t]
        current_target = commanded[:, u._arm_ids_t]
        next_arm = torch.maximum(
            # Integrate in target space.  Using ``current_arm + delta_q`` caps the actuator's
            # position error at one tiny IK increment; under table/contact load that produced too
            # little torque to break static friction and the hand never rose.
            torch.minimum(current_target + delta_q, u.dof_upper[:, u._arm_ids_t]),
            u.dof_lower[:, u._arm_ids_t],
        )
        commanded[:, u._arm_ids_t] = next_arm
        state = step_and_measure()
        update_grip_servo(state["force"])
        state["position_error"] = pos_error
        state["rotation_error"] = rot_error
        state["delta_q"] = delta_q
        state["predicted_delta"] = (jacobian @ delta_q.unsqueeze(-1)).squeeze(-1)
        state["arm_tracking_error"] = next_arm - u.robot.data.joint_pos[:, u._arm_ids_t]
        return state

    trajectory: list[dict[str, float]] = []

    def report(stage: str, step: int, state: dict[str, torch.Tensor]) -> None:
        signals = state["signals"]
        palm_rise = u.palm_center_w[:, 2] - palm_center_start_z
        clear = state["clearance"]
        row = {
            "stage": stage,
            "step": step,
            "palm_rise_mean": float(palm_rise.mean()),
            "clearance_mean": float(clear.mean()),
            "clearance_min": float(clear.min()),
            "latch_fraction": float(u._is_grasped.float().mean()),
            "q_grasp_mean": float(signals["grasp_quality"].mean()),
            "q_wrap_mean": float(signals["quality"].mean()),
            "hold_quality_mean": float(signals["hold_quality"].mean()),
        }
        trajectory.append(row)
        print(
            f"{stage:6s} {step:3d}: palm+={row['palm_rise_mean']:.3f}m "
            f"true_clear={row['clearance_mean']:.3f}m (min={row['clearance_min']:.3f}) "
            f"latch={row['latch_fraction']:.3f} q={row['q_grasp_mean']:.3f} "
            f"hold={row['hold_quality_mean']:.3f} "
            f"parts=[F{float(signals['thumb_strength'].mean()):.2f}/"
            f"O{float(signals['other_coverage'].mean()):.2f}/"
            f"A{float(signals['alignment_score'].mean()):.2f}/"
            f"P{float(signals['opposition_score'].mean()):.2f}/"
            f"p{float(signals['palm_score'].mean()):.2f}] "
            f"F={[round(x, 1) for x in state['force'].mean(dim=0).tolist()]} "
            f"ez={float(state['position_error'][:, 2].mean()):+.4f} "
            f"Jdq_z={float(state['predicted_delta'][:, 2].mean()):+.4f} "
            f"|dq|={float(state['delta_q'].norm(dim=-1).mean()):.4f} "
            f"|qerr|={float(state['arm_tracking_error'].norm(dim=-1).mean()):.4f}",
            flush=True,
        )

    print("\n=== 2CM MICRO-LIFT ===", flush=True)
    micro_state = None
    for step in range(args_cli.micro_steps):
        height = args_cli.micro_height * float(step + 1) / float(args_cli.micro_steps)
        micro_state = dls_step(height)
        if step % 10 == 0 or step == args_cli.micro_steps - 1:
            report("micro", step, micro_state)
    for step in range(args_cli.micro_hold_steps):
        micro_state = dls_step(args_cli.micro_height)
        if step % 15 == 0 or step == args_cli.micro_hold_steps - 1:
            report("mhold", step, micro_state)
    micro_clearance = micro_state["clearance"]
    micro_signals = micro_state["signals"]
    micro_pass = (
        (micro_clearance >= 0.015)
        & u._is_grasped
        & (micro_signals["grasp_quality"] >= u.cfg.grasp_quality_low)
    )
    print(f"MICRO RESULT: {int(micro_pass.sum())}/{num_envs} passed", flush=True)

    print("\n=== CONTINUE TO 20CM TRUE CLEARANCE ===", flush=True)
    lift_state = micro_state
    for step in range(args_cli.lift_steps):
        height = args_cli.micro_height + (args_cli.target_height - args_cli.micro_height) * (
            float(step + 1) / float(args_cli.lift_steps)
        )
        lift_state = dls_step(height)
        if step % 30 == 0 or step == args_cli.lift_steps - 1:
            report("lift", step, lift_state)

    stable_count = torch.zeros(num_envs, dtype=torch.long, device=dev)
    for step in range(args_cli.settle_steps):
        lift_state = dls_step(args_cli.target_height)
        signals = lift_state["signals"]
        slow = (
            (u.object.data.root_com_lin_vel_w.norm(dim=-1) < u.cfg.success_max_obj_lin_speed)
            & (u.object.data.root_com_ang_vel_w.norm(dim=-1) < u.cfg.success_max_obj_ang_speed)
        )
        stable = (
            (lift_state["clearance"] >= u.cfg.lift_success_height)
            & u._is_grasped
            & (signals["grasp_quality"] >= u.cfg.grasp_quality_high)
            & slow
        )
        stable_count = torch.where(stable, stable_count + 1, torch.zeros_like(stable_count))
        if step % 10 == 0 or step == args_cli.settle_steps - 1:
            report("settle", step, lift_state)

    final_clearance = lift_state["clearance"]
    final_signals = lift_state["signals"]
    # cfg was intentionally raised to avoid reset; use the task's original ten-step standard.
    success = stable_count >= 10
    palm_rise = u.palm_center_w[:, 2] - palm_center_start_z
    clearance_gain = final_clearance - clearance_start
    print("\n=== ORACLE VERDICT ===", flush=True)
    print(
        f"close={int(close_pass.sum())}/{num_envs}, micro={int(micro_pass.sum())}/{num_envs}, "
        f"20cm stable success={int(success.sum())}/{num_envs}",
        flush=True,
    )
    print(
        f"final true clearance quantiles={_quantiles(final_clearance)}; "
        f"palm-rise minus clearance-gain median={float((palm_rise-clearance_gain).median()):+.4f}m",
        flush=True,
    )
    print(
        f"final q_grasp={_quantiles(final_signals['grasp_quality'])}; "
        f"peak object-filtered forces per finger={force_peak.max(dim=0).values.tolist()}",
        flush=True,
    )

    output = {
        "source_feasibility_json": str(artifact_path.resolve()),
        "mode": args_cli.mode,
        "num_envs": num_envs,
        "close_pass_count": int(close_pass.sum()),
        "micro_pass_count": int(micro_pass.sum()),
        "success_count": int(success.sum()),
        "true_clearance": _quantiles(final_clearance),
        "palm_rise": _quantiles(palm_rise),
        "clearance_gain": _quantiles(clearance_gain),
        "final_q_grasp": _quantiles(final_signals["grasp_quality"]),
        "max_slip_lin": _quantiles(max_slip_lin),
        "max_slip_ang": _quantiles(max_slip_ang),
        "force_order": list(u.ee_names),
        "force_peak_per_finger": force_peak.max(dim=0).values.detach().cpu().tolist(),
        "trajectory": trajectory,
    }
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)
    env.close()
    if not bool(success.any()):
        raise RuntimeError("oracle failed to achieve one stable 20cm true-clearance lift")


if __name__ == "__main__":
    main()
    simulation_app.close()
