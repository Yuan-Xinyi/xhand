#!/usr/bin/env python3
"""Physical contract smoke for the phased 21-D coupled power-close environment."""

from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--curriculum_dataset", required=True, type=Path)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=209)
parser.add_argument("--align_arm_action", type=float, default=0.01)
parser.add_argument("--close_steps", type=int, default=6)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--watchdog_seconds",
    type=float,
    default=45.0,
    help="Dump all Python thread stacks if the physical smoke stops making progress.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 3:
    parser.error("--num_envs must be at least three for cloned physics and pose-safety probes")
if not 0.0 <= args_cli.align_arm_action <= 0.1:
    parser.error("--align_arm_action must be in [0, 0.1]")
if args_cli.close_steps < 1 or args_cli.close_steps >= 12:
    parser.error("--close_steps must be in [1, 11] to avoid the intentional lost-window timeout")
if args_cli.watchdog_seconds <= 0.0:
    parser.error("--watchdog_seconds must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_mul
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401


@torch.inference_mode()
def main() -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(args_cli.watchdog_seconds, repeat=True)
    print("[contract] constructing environment", flush=True)
    cfg = parse_env_cfg(
        "Pick-Tool-Token-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    cfg.seed = args_cli.seed
    cfg.close_option_mode = True
    cfg.power_close_option_mode = True
    cfg.coupled_power_align_close_option_mode = True
    cfg.observation_space = 131
    cfg.state_space = 131
    cfg.episode_length_s = 3.0
    cfg.curriculum_dataset = str(args_cli.curriculum_dataset.resolve())
    cfg.curriculum_boundary = "close_start"
    cfg.curriculum_reset_probability = 1.0
    cfg.curriculum_joint_noise = 0.0
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    print("[contract] resetting environment", flush=True)
    u = env.unwrapped
    observations, _ = env.reset()
    print("[contract] reset complete", flush=True)
    policy = observations["policy"]
    if policy.shape != (args_cli.num_envs, 131):
        raise AssertionError(f"coupled policy observation has shape {tuple(policy.shape)}")

    hand_hold = u.dof_targets[:, u._hand_ids_t].clone()
    arm_anchor = u._coupled_arm_anchor.clone()
    force_peak = torch.zeros(args_cli.num_envs, device=u.device)
    arm_offset_peak = torch.zeros_like(force_peak)
    hand_hold_error_peak = torch.zeros_like(force_peak)
    tactile_unload_events = torch.zeros(
        args_cli.num_envs, dtype=torch.long, device=u.device
    )
    align_steps = int(cfg.coupled_power_align_steps)
    arm_pattern = torch.tensor(
        (1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75),
        dtype=torch.float32,
        device=u.device,
    )

    for step in range(align_steps + args_cli.close_steps):
        print(f"[contract] action {step + 1}/{align_steps + args_cli.close_steps}", flush=True)
        action = torch.zeros((args_cli.num_envs, 21), device=u.device)
        if step < align_steps:
            jacobian = u.robot.root_physx_view.get_jacobians()
            palm_jacobian = jacobian[:, u._palm_jac_idx, :, :][:, :, u._arm_ids_t]
            palm_up_jacobian = palm_jacobian[:, 2, :]
            # A per-clone downward palm command cannot trigger the upward-action shield, so the
            # executed arm tensor must expose the exact phase multiplier rather than a projection.
            arm_pattern = -palm_up_jacobian / palm_up_jacobian.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
        action[:, :7] = args_cli.align_arm_action * arm_pattern
        action[:, 7:] = 0.75
        latch_before = u._power_is_grasped.clone()
        force_before = u._finger_object_force_magnitudes()
        soft_before = force_before.max(dim=-1).values >= cfg.tactile_soft_force_limit
        arm_before = u.dof_targets[:, u._arm_ids_t].clone()
        observations, _, terminated, truncated, info = env.step(action)
        print(f"[contract] action {step + 1} complete", flush=True)
        if bool((terminated | truncated).any()):
            raise AssertionError(
                f"deterministic phase smoke terminated at action {step + 1}: "
                f"terminated={terminated.tolist()} truncated={truncated.tolist()}"
            )
        if observations["policy"].shape != (args_cli.num_envs, 131):
            raise AssertionError("coupled observation width changed during rollout")
        force = u._finger_object_force_magnitudes().max(dim=-1).values
        force_peak = torch.maximum(force_peak, force)
        arm_offset = (
            u.dof_targets[:, u._arm_ids_t] - arm_anchor
        ).abs().max(dim=-1).values
        arm_offset_peak = torch.maximum(arm_offset_peak, arm_offset)
        if bool((arm_offset > cfg.coupled_power_arm_target_limit + 1.0e-6).any()):
            raise AssertionError("arm target escaped its reset-anchor envelope")

        if step < align_steps:
            expected_arm = action[:, :7] * cfg.coupled_power_arm_action_multiplier
            if not torch.allclose(u.actions[:, :7], expected_arm, rtol=0.0, atol=1.0e-7):
                raise AssertionError("ALIGN did not execute the declared arm action multiplier")
            if torch.count_nonzero(u.actions[:, 7:]).item() != 0:
                raise AssertionError("ALIGN executed a nonzero hand action")
            hand_error = (
                u.dof_targets[:, u._hand_ids_t] - hand_hold
            ).abs().max(dim=-1).values
            hand_hold_error_peak = torch.maximum(hand_hold_error_peak, hand_error)
            target_tolerance = 1.0e-6
            changed = hand_error > target_tolerance
            if bool((changed & (~soft_before)).any()):
                raise AssertionError(
                    "ALIGN changed the captured hand target without a tactile-soft safety event: "
                    f"error={hand_error.tolist()} force_before="
                    f"{force_before.max(dim=-1).values.tolist()}"
                )
            if bool(changed.any()):
                if not torch.allclose(
                    u._coupled_hand_hold_target[changed],
                    u.dof_targets[changed][:, u._hand_ids_t],
                    rtol=0.0,
                    atol=target_tolerance,
                ):
                    raise AssertionError("tactile unload was not captured as the new ALIGN hold")
                tactile_unload_events += changed.long()
                hand_hold[changed] = u.dof_targets[changed][:, u._hand_ids_t]
        else:
            if torch.count_nonzero(u.actions[:, :7]).item() != 0:
                raise AssertionError("CLOSE retained arm action authority")
            if not torch.allclose(
                u.dof_targets[:, u._arm_ids_t], arm_before, rtol=0.0, atol=1.0e-7
            ):
                raise AssertionError("CLOSE changed the frozen arm target")

        terminal = info.get("pick_tool_terminal")
        if not isinstance(terminal, dict):
            raise AssertionError("coupled rollout omitted pick_tool_terminal telemetry")
        for key in (
            "coupled_power_pose_escape",
            "coupled_power_rotation_drift",
            "coupled_power_xy_drift",
            "coupled_power_true_clearance",
            "coupled_power_arm_target_offset_abs_max",
            "coupled_power_arm_target_saturated",
            "coupled_power_align_active",
        ):
            value = terminal.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != (args_cli.num_envs,):
                raise AssertionError(f"invalid coupled terminal telemetry {key!r}")
        expected_align_active = (step < align_steps) & (~latch_before)
        if not torch.equal(
            terminal["coupled_power_align_active"], expected_align_active
        ):
            raise AssertionError("terminal ALIGN-active telemetry disagrees with executed phase")
        derived_pose_escape = (
            (terminal["coupled_power_rotation_drift"] > cfg.coupled_power_rotation_drift_limit)
            | (terminal["coupled_power_xy_drift"] > cfg.close_option_horizontal_drift_limit)
            | (terminal["coupled_power_true_clearance"] > cfg.close_option_unlatched_lift_limit)
        )
        if not torch.equal(terminal["coupled_power_pose_escape"], derived_pose_escape):
            raise AssertionError("pose-escape telemetry disagrees with its physical scalars")
        next_align_active = (
            (u.episode_length_buf < align_steps) & (~u._power_is_grasped)
        ).float()
        next_align_progress = torch.clamp(
            u.episode_length_buf.float() / float(align_steps), 0.0, 1.0
        )
        expected_phase_observation = torch.stack(
            (next_align_active, next_align_progress), dim=-1
        )
        if not torch.allclose(
            observations["policy"][:, -2:],
            expected_phase_observation,
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise AssertionError("131-D suffix does not expose the coupled timed phase state")

    if not bool((force_peak > 0.0).all()):
        raise AssertionError(
            "one or more cloned environments had zero object contact force; "
            f"force_peak={force_peak.tolist()}"
        )

    print("[contract] injecting independent pose-safety violations", flush=True)
    env.reset()
    probe_pose = torch.cat(
        (u.object.data.root_pos_w.clone(), u.object.data.root_quat_w.clone()), dim=-1
    )
    probe_pose[0, 0] += cfg.close_option_horizontal_drift_limit + 0.005
    # Rotation/vertical jumps are deliberately well beyond the boundary: the surrounding hand
    # can resolve a sizable part of an injected overlap during the one physics frame before dones.
    rotation_angle = cfg.coupled_power_rotation_drift_limit + 0.35
    rotation_delta = torch.tensor(
        (
            torch.cos(torch.tensor(rotation_angle / 2.0)).item(),
            0.0,
            0.0,
            torch.sin(torch.tensor(rotation_angle / 2.0)).item(),
        ),
        dtype=torch.float32,
        device=u.device,
    )
    probe_pose[1, 3:] = quat_mul(rotation_delta.unsqueeze(0), probe_pose[1:2, 3:])[0]
    probe_pose[2, 2] += cfg.close_option_unlatched_lift_limit + 0.065
    env_ids = u.robot._ALL_INDICES
    u.object.write_root_pose_to_sim(probe_pose, env_ids=env_ids)
    u.object.write_root_velocity_to_sim(
        torch.zeros((args_cli.num_envs, 6), device=u.device), env_ids=env_ids
    )
    _, _, probe_terminated, probe_truncated, probe_info = env.step(
        torch.zeros((args_cli.num_envs, 21), device=u.device)
    )
    probe_terminal = probe_info.get("pick_tool_terminal")
    if not isinstance(probe_terminal, dict):
        raise AssertionError("pose-safety probe omitted terminal ground truth")
    if not bool(probe_terminated[:3].all()) or bool(probe_truncated[:3].any()):
        raise AssertionError(
            "xy/rotation/clearance violations did not terminate as failures: "
            f"terminated={probe_terminated.tolist()} truncated={probe_truncated.tolist()} "
            f"pose_escape={probe_terminal['coupled_power_pose_escape'].tolist()} "
            f"xy={probe_terminal['coupled_power_xy_drift'].tolist()} "
            f"rotation={probe_terminal['coupled_power_rotation_drift'].tolist()} "
            f"clearance={probe_terminal['coupled_power_true_clearance'].tolist()}"
        )
    if bool(probe_terminated[3:].any()) or bool(probe_truncated[3:].any()):
        raise AssertionError("pose-safety probe terminated an untouched clone")
    if not bool(probe_terminal["coupled_power_pose_escape"][:3].all()):
        raise AssertionError("pose-safety violations were not labeled as pose escapes")
    if bool(probe_terminal["power_close_option_success"][:3].any()):
        raise AssertionError("pose-safety violation tied with or lost to option success")
    if not bool(probe_terminal["power_close_option_failure"][:3].all()):
        raise AssertionError("pose-safety violations were not labeled as option failures")
    safety_probe = {
        "xy_drift_m": float(probe_terminal["coupled_power_xy_drift"][0].item()),
        "rotation_drift_rad": float(
            probe_terminal["coupled_power_rotation_drift"][1].item()
        ),
        "true_clearance_m": float(
            probe_terminal["coupled_power_true_clearance"][2].item()
        ),
        "terminated": probe_terminated[:3].cpu().tolist(),
        "success": probe_terminal["power_close_option_success"][:3].cpu().tolist(),
        "failure": probe_terminal["power_close_option_failure"][:3].cpu().tolist(),
    }
    report = {
        "contract": "coupled_power_align_close_physical_smoke_v1",
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "actions": align_steps + args_cli.close_steps,
        "align_steps": align_steps,
        "close_steps": args_cli.close_steps,
        "observation_dim": int(policy.shape[1]),
        "environment_action_dim": 21,
        "force_peak_n": force_peak.cpu().tolist(),
        "force_peak_max_n": float(force_peak.max().item()),
        "arm_target_offset_peak_rad": arm_offset_peak.cpu().tolist(),
        "arm_target_offset_peak_max_rad": float(arm_offset_peak.max().item()),
        "hand_hold_error_peak_rad": hand_hold_error_peak.cpu().tolist(),
        "tactile_unload_events": tactile_unload_events.cpu().tolist(),
        "pose_safety_probe": safety_probe,
        "strict_success_claimed": False,
        "verdict": "pass",
    }
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    env.close()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac Sim can stall in ``close()`` when the host inotify quota is exhausted.  Emit the
        # actual contract failure before entering simulator teardown so it cannot be hidden.
        traceback.print_exc()
        faulthandler.cancel_dump_traceback_later()
        raise
    else:
        simulation_app.close()
