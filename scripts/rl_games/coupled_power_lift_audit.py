#!/usr/bin/env python3
"""Audit whether the exact coupled POWER grasp can carry a 20 cm tool lift.

The experiment deliberately separates exploration from mechanics:

1. restore the exact ``close_start`` curriculum state;
2. run the immutable coupled CEM teacher until the native close option reports success;
3. hand only the seven arm targets to a damped-least-squares Cartesian controller;
4. retain the teacher's hand action and the task's tactile shield;
5. require 15 consecutive stable frames at both 5 cm and 20 cm *true mesh* clearance.

The close option normally auto-resets on success.  This audit calls its original done
function so all latches and terminal telemetry remain authoritative, but masks the
returned done flags and takes over only conservatively accepted success rows.  Its
own verdict permanently rejects force excursions, airborne unlatching, hand/object
relative-pose escape, and arm joint-limit saturation.  Object-root and AABB heights
are diagnostics only and never enter a pass condition.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Coupled POWER close-to-lift physical audit")
parser.add_argument("--task", default="Pick-Tool-Token-Direct-v0")
parser.add_argument(
    "--teacher_artifact",
    type=Path,
    default=Path("logs/flashsac/pick_tool/coupled_power_curriculum/cem_robust8_s224.json"),
)
parser.add_argument(
    "--curriculum_dataset",
    type=Path,
    default=Path("logs/flashsac/pick_tool/coupled_power_curriculum/rank1_static_v1.pt"),
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=224)
parser.add_argument("--close_step_limit", type=int, default=150)
parser.add_argument("--micro_palm_rise", type=float, default=0.07)
parser.add_argument("--target_palm_rise", type=float, default=0.24)
parser.add_argument("--micro_steps", type=int, default=50)
parser.add_argument("--micro_hold_steps", type=int, default=30)
parser.add_argument("--lift_steps", type=int, default=240)
parser.add_argument("--settle_steps", type=int, default=45)
parser.add_argument("--stable_steps", type=int, default=15)
parser.add_argument(
    "--lift_contract",
    choices=("dual_power", "native_full_task"),
    default="dual_power",
    help="dual_power is the conservative audit; native_full_task matches task lift authority",
)
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--max_cart_step", type=float, default=0.004)
parser.add_argument("--max_rot_step", type=float, default=0.05)
parser.add_argument("--max_joint_step", type=float, default=0.04)
parser.add_argument(
    "--tactile_distal_servo",
    action="store_true",
    help="adapt one distal joint per fingertip after the strict close handoff",
)
parser.add_argument("--grip_force_target", type=float, default=3.0)
parser.add_argument("--grip_force_limit", type=float, default=20.0)
parser.add_argument("--grip_servo_step", type=float, default=0.006)
parser.add_argument("--grip_servo_range", type=float, default=0.60)
parser.add_argument(
    "--hold_hand_artifact",
    type=Path,
    default=None,
    help="optional hand-space feasibility JSON supplying results[mode].latent",
)
parser.add_argument(
    "--hold_hand_mode",
    choices=("token9", "raw12", "hybrid14"),
    default="hybrid14",
)
parser.add_argument("--hold_hand_blend_steps", type=int, default=40)
parser.add_argument("--relative_position_drift_limit", type=float, default=0.03)
parser.add_argument("--relative_rotation_drift_limit", type=float, default=0.35)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("logs/flashsac/pick_tool/coupled_power_lift_audit/s224_n64.json"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

for name in (
    "num_envs",
    "close_step_limit",
    "micro_steps",
    "micro_hold_steps",
    "lift_steps",
    "settle_steps",
    "stable_steps",
    "hold_hand_blend_steps",
):
    if int(getattr(args_cli, name)) < 1:
        parser.error(f"--{name} must be positive")
for name in (
    "micro_palm_rise",
    "target_palm_rise",
    "damping",
    "max_cart_step",
    "max_rot_step",
    "max_joint_step",
    "relative_position_drift_limit",
    "relative_rotation_drift_limit",
    "grip_force_target",
    "grip_force_limit",
    "grip_servo_step",
    "grip_servo_range",
):
    value = float(getattr(args_cli, name))
    if not math.isfinite(value) or value <= 0.0:
        parser.error(f"--{name} must be finite and positive")
if args_cli.target_palm_rise <= args_cli.micro_palm_rise:
    parser.error("--target_palm_rise must exceed --micro_palm_rise")
if args_cli.grip_force_target >= args_cli.grip_force_limit:
    parser.error("--grip_force_target must be below --grip_force_limit")
for path_name in ("teacher_artifact", "curriculum_dataset"):
    path = Path(getattr(args_cli, path_name))
    if path.is_symlink() or not path.is_file():
        parser.error(f"--{path_name} must be a regular non-symlink file: {path}")
if args_cli.hold_hand_artifact is not None and (
    args_cli.hold_hand_artifact.is_symlink()
    or not args_cli.hold_hand_artifact.is_file()
):
    parser.error("--hold_hand_artifact must be a regular non-symlink file")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import compute_pose_error, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
FLASHSAC_SCRIPT_DIR = SCRIPT_DIR.parent / "flashsac"
if str(FLASHSAC_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FLASHSAC_SCRIPT_DIR))

from coupled_power_lift_contract import LiftAuditThresholds, classify_lift_truth
from coupled_teacher_prior import load_coupled_teacher_prior, sha256_file


WAIT_CLOSE = 0
MICRO_RAMP = 1
MICRO_HOLD = 2
LIFT_RAMP = 3
SETTLE = 4
SUCCESS = 5
FAILED = 6

STAGE_NAMES = {
    WAIT_CLOSE: "wait_close",
    MICRO_RAMP: "micro_ramp",
    MICRO_HOLD: "micro_hold",
    LIFT_RAMP: "lift_ramp",
    SETTLE: "settle",
    SUCCESS: "success",
    FAILED: "failed",
}

FAIL_NONE = 0
FAIL_CLOSE_REJECTED = 1
FAIL_CLOSE_TIMEOUT = 2
FAIL_FORCE = 3
FAIL_AIRBORNE_UNLATCH = 4
FAIL_ARM_LIMIT = 5
FAIL_RELATIVE_POSE = 6
FAIL_DROP = 7
FAIL_MICRO_UNSTABLE = 8
FAIL_FINAL_UNSTABLE = 9
FAIL_TRANSPORT_BREAK = 10

FAILURE_NAMES = {
    FAIL_NONE: "none",
    FAIL_CLOSE_REJECTED: "native_close_not_conservative_load_handoff",
    FAIL_CLOSE_TIMEOUT: "close_timeout",
    FAIL_FORCE: "trajectory_force_above_30N",
    FAIL_AIRBORNE_UNLATCH: "airborne_grasp_or_power_latch_lost",
    FAIL_ARM_LIMIT: "dls_arm_target_joint_limit_saturation",
    FAIL_RELATIVE_POSE: "hand_object_relative_pose_escape",
    FAIL_DROP: "object_drop",
    FAIL_MICRO_UNSTABLE: "no_15_frame_stable_5cm_hold",
    FAIL_FINAL_UNSTABLE: "no_15_frame_stable_20cm_hold",
    FAIL_TRANSPORT_BREAK: "two_frame_low_quality_transport_break",
}


def _limit_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    return value * torch.clamp(limit / norm, max=1.0)


def _summary(value: torch.Tensor) -> dict[str, float]:
    finite = value.detach().float().cpu()
    return {
        "min": float(finite.min()),
        "p10": float(torch.quantile(finite, 0.10)),
        "median": float(finite.median()),
        "p90": float(torch.quantile(finite, 0.90)),
        "max": float(finite.max()),
    }


def _counts(values: torch.Tensor, names: dict[int, str]) -> dict[str, int]:
    return {
        name: int((values == code).sum().item())
        for code, name in names.items()
        if bool((values == code).any())
    }


@torch.inference_mode()
def run() -> dict[str, object]:
    torch.manual_seed(args_cli.seed)
    teacher_path = args_cli.teacher_artifact.resolve()
    curriculum_path = args_cli.curriculum_dataset.resolve()
    teacher = load_coupled_teacher_prior(teacher_path, curriculum_path)

    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    # Preserve the exact 3 s teacher-close horizon.  Returned timeouts are masked
    # only after capture so a DLS handoff can continue in the same physical state.
    cfg.episode_length_s = 3.0
    cfg.close_option_mode = True
    cfg.power_close_option_mode = True
    cfg.coupled_power_align_close_option_mode = True
    cfg.observation_space = 131
    cfg.state_space = 131
    cfg.curriculum_dataset = str(curriculum_path)
    cfg.curriculum_boundary = "close_start"
    cfg.curriculum_reset_probability = 1.0
    cfg.curriculum_joint_noise = 0.0
    env = gym.make(args_cli.task, cfg=cfg, render_mode=None)
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    observations, _ = env.reset(seed=args_cli.seed)
    observation = observations["policy"]
    if observation.shape != (n, 131):
        raise RuntimeError(f"expected {(n, 131)} policy observation, got {tuple(observation.shape)}")

    stage = torch.full((n,), WAIT_CLOSE, dtype=torch.long, device=dev)
    stage_step = torch.zeros(n, dtype=torch.long, device=dev)
    handoff_age = torch.zeros(n, dtype=torch.long, device=dev)
    failure_reason = torch.zeros(n, dtype=torch.long, device=dev)
    native_close_seen = torch.zeros(n, dtype=torch.bool, device=dev)
    conservative_handoff = torch.zeros(n, dtype=torch.bool, device=dev)
    handoff_step = torch.zeros(n, dtype=torch.long, device=dev)
    micro_pass = torch.zeros(n, dtype=torch.bool, device=dev)
    micro_stable_count = torch.zeros(n, dtype=torch.long, device=dev)
    final_stable_count = torch.zeros(n, dtype=torch.long, device=dev)
    transport_break_steps = torch.zeros(n, dtype=torch.long, device=dev)

    force_peak = torch.zeros(n, device=dev)
    true_clearance_peak = torch.full((n,), -float("inf"), device=dev)
    close_xy_drift_peak = torch.zeros(n, device=dev)
    close_rotation_drift_peak = torch.zeros(n, device=dev)
    close_pose_escape_ever = torch.zeros(n, dtype=torch.bool, device=dev)
    close_arm_saturated_ever = torch.zeros(n, dtype=torch.bool, device=dev)
    dls_arm_limit_ever = torch.zeros(n, dtype=torch.bool, device=dev)
    airborne_unlatched_ever = torch.zeros(n, dtype=torch.bool, device=dev)
    relative_pose_escape_ever = torch.zeros(n, dtype=torch.bool, device=dev)
    trajectory_unsafe_force_ever = torch.zeros(n, dtype=torch.bool, device=dev)
    min_grasp_quality = torch.ones(n, device=dev)
    min_power_grasp_quality = torch.ones(n, device=dev)
    min_hold_quality = torch.ones(n, device=dev)
    max_relative_position_drift = torch.zeros(n, device=dev)
    max_relative_rotation_drift = torch.zeros(n, device=dev)
    max_arm_tracking_error = torch.zeros(n, device=dev)
    max_object_linear_speed = torch.zeros(n, device=dev)
    max_object_angular_speed = torch.zeros(n, device=dev)

    palm_body_start = torch.zeros((n, 3), device=dev)
    palm_quat_target = torch.zeros((n, 4), device=dev)
    palm_quat_target[:, 0] = 1.0
    palm_center_start_z = torch.zeros(n, device=dev)
    clearance_start = torch.zeros(n, device=dev)
    object_in_palm_start_pos = torch.zeros((n, 3), device=dev)
    object_in_palm_start_quat = torch.zeros((n, 4), device=dev)
    object_in_palm_start_quat[:, 0] = 1.0
    commanded_arm = u.dof_targets[:, u._arm_ids_t].detach().clone()
    hand_names = [u.robot.joint_names[index] for index in u._hand_ids_t.tolist()]
    fingertip_to_distal = {
        "thumb_rota_link2": "thumb_joint2",
        "index_rota_link2": "index_joint2",
        "mid_link2": "middle_joint1",
        "ring_link2": "ring_joint1",
        "pinky_link2": "pinky_joint1",
    }
    try:
        servo_hand_ids = torch.tensor(
            [hand_names.index(fingertip_to_distal[name]) for name in u.ee_names],
            dtype=torch.long,
            device=dev,
        )
    except (KeyError, ValueError) as error:
        raise RuntimeError("fingertip/distal-joint mapping no longer matches the robot") from error
    servo_joint_ids = u._hand_ids_t.index_select(0, servo_hand_ids)
    servo_lower = u.dof_targets[:, servo_joint_ids].detach().clone()
    servo_upper = torch.minimum(
        servo_lower + args_cli.grip_servo_range,
        u.dof_upper[:, servo_joint_ids],
    )
    servo_target = servo_lower.clone()
    servo_max_adjustment = torch.zeros(n, device=dev)
    eye6 = torch.eye(6, device=dev).unsqueeze(0)

    native_terminated_signal = torch.zeros(n, dtype=torch.bool, device=dev)
    native_timeout_signal = torch.zeros(n, dtype=torch.bool, device=dev)
    native_close_success_signal = torch.zeros(n, dtype=torch.bool, device=dev)
    native_close_failure_signal = torch.zeros(n, dtype=torch.bool, device=dev)
    original_get_dones = u._get_dones

    def audit_get_dones(self: object) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, timed_out = original_get_dones()
        native_terminated_signal.copy_(terminated)
        native_timeout_signal.copy_(timed_out)
        native_close_success_signal.copy_(u._close_option_success)
        native_close_failure_signal.copy_(u._close_option_failure)
        # Preserve all original state-machine mutations and extras, but stop DirectRLEnv
        # from resetting the exact physical state before the arm handoff.
        return torch.zeros_like(terminated), torch.zeros_like(timed_out)

    u._get_dones = types.MethodType(audit_get_dones, u)

    original_pre_physics = u._pre_physics_step

    def audit_pre_physics(self: object, actions: torch.Tensor) -> None:
        original_pre_physics(actions)
        active = (
            (stage == MICRO_RAMP)
            | (stage == MICRO_HOLD)
            | (stage == LIFT_RAMP)
            | (stage == SETTLE)
            | (stage == SUCCESS)
        )
        if not bool(active.any()):
            return

        pre_force_by_finger = u._finger_object_force_magnitudes()
        if args_cli.tactile_distal_servo:
            increase = pre_force_by_finger < args_cli.grip_force_target
            decrease = pre_force_by_finger > args_cli.grip_force_limit
            servo_delta = args_cli.grip_servo_step * (
                increase.to(torch.float32) - decrease.to(torch.float32)
            )
            proposed_servo = torch.maximum(
                torch.minimum(servo_target + servo_delta, servo_upper),
                servo_lower,
            )
            servo_target[active] = proposed_servo[active]
            # Preserve the environment's all-hand soft-force unload on this frame.
            servo_safe = active & (
                pre_force_by_finger.max(dim=-1).values
                < float(cfg.tactile_soft_force_limit)
            )
            hand_targets = u.dof_targets[:, u._hand_ids_t].clone()
            selected_targets = hand_targets.index_select(1, servo_hand_ids)
            selected_targets[servo_safe] = servo_target[servo_safe]
            hand_targets[:, servo_hand_ids] = selected_targets
            u.dof_targets[:, u._hand_ids_t] = hand_targets
            servo_max_adjustment.copy_(
                torch.maximum(
                    servo_max_adjustment,
                    (servo_target - servo_lower).abs().max(dim=-1).values * active,
                )
            )

        current_pos = u.robot.data.body_pos_w[:, u.palm_idx]
        current_quat = u.robot.data.body_quat_w[:, u.palm_idx]
        desired_pos = palm_body_start.clone()
        desired_rise = torch.zeros(n, device=dev)
        micro_ramp = stage == MICRO_RAMP
        desired_rise[micro_ramp] = args_cli.micro_palm_rise * (
            (stage_step[micro_ramp].float() + 1.0) / float(args_cli.micro_steps)
        ).clamp(max=1.0)
        desired_rise[stage == MICRO_HOLD] = args_cli.micro_palm_rise
        lift_ramp = stage == LIFT_RAMP
        desired_rise[lift_ramp] = args_cli.micro_palm_rise + (
            args_cli.target_palm_rise - args_cli.micro_palm_rise
        ) * (
            (stage_step[lift_ramp].float() + 1.0) / float(args_cli.lift_steps)
        ).clamp(max=1.0)
        desired_rise[(stage == SETTLE) | (stage == SUCCESS)] = args_cli.target_palm_rise
        desired_pos[:, 2] += desired_rise

        position_error, rotation_error = compute_pose_error(
            current_pos,
            current_quat,
            desired_pos,
            palm_quat_target,
            rot_error_type="axis_angle",
        )
        cartesian_delta = torch.cat(
            (
                _limit_norm(position_error, args_cli.max_cart_step),
                _limit_norm(rotation_error, args_cli.max_rot_step),
            ),
            dim=-1,
        )
        jacobian = u.robot.root_physx_view.get_jacobians()
        jacobian = jacobian[:, u._palm_jac_idx, :, :][:, :, u._arm_ids_t]
        jt = jacobian.transpose(1, 2)
        system = jacobian @ jt + (args_cli.damping**2) * eye6
        delta_q = (jt @ torch.linalg.solve(system, cartesian_delta.unsqueeze(-1))).squeeze(-1)
        delta_q = delta_q.clamp(-args_cli.max_joint_step, args_cli.max_joint_step)
        raw_next = commanded_arm + delta_q
        lower = u.dof_lower[:, u._arm_ids_t]
        upper = u.dof_upper[:, u._arm_ids_t]
        bounded = torch.maximum(torch.minimum(raw_next, upper), lower)
        hit_limit = ((raw_next - bounded).abs() > 1.0e-7).any(dim=-1) & active
        dls_arm_limit_ever.logical_or_(hit_limit)
        commanded_arm[active] = bounded[active]
        # Do not undo the task's hard-force arm arrest.  The trajectory will be
        # permanently failed immediately after this physics step.
        pre_force = pre_force_by_finger.max(dim=-1).values
        safe_active = active & (pre_force < float(cfg.tactile_hard_force_limit))
        arm_targets = u.dof_targets[:, u._arm_ids_t].clone()
        arm_targets[safe_active] = commanded_arm[safe_active]
        u.dof_targets[:, u._arm_ids_t] = arm_targets
        tracking = (commanded_arm - u.robot.data.joint_pos[:, u._arm_ids_t]).norm(dim=-1)
        max_arm_tracking_error.copy_(torch.maximum(max_arm_tracking_error, tracking * active))

    u._pre_physics_step = types.MethodType(audit_pre_physics, u)

    thresholds = LiftAuditThresholds(
        max_object_linear_speed_m_s=float(cfg.success_max_obj_lin_speed),
        max_object_angular_speed_rad_s=float(cfg.success_max_obj_ang_speed),
    )
    zero_residual = torch.zeros((n, 14), device=dev)
    teacher_hand_action = torch.tensor(
        teacher.hand_latent, dtype=torch.float32, device=dev
    ).unsqueeze(0).expand(n, -1)
    hold_hand_source: str | None = None
    hold_hand_sha256: str | None = None
    hold_hand_action = teacher_hand_action
    if args_cli.hold_hand_artifact is not None:
        hold_path = args_cli.hold_hand_artifact.resolve()
        payload = json.loads(hold_path.read_text(encoding="utf-8"))
        if payload.get("hand_joint_names") != hand_names:
            raise RuntimeError("hold-hand artifact joint order does not match the runtime hand")
        mode_payload = payload.get("results", {}).get(args_cli.hold_hand_mode)
        if not isinstance(mode_payload, dict):
            raise RuntimeError(f"hold-hand artifact has no results[{args_cli.hold_hand_mode!r}]")
        latent = mode_payload.get("latent")
        if (
            not isinstance(latent, list)
            or len(latent) != 14
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in latent
            )
        ):
            raise RuntimeError("hold-hand artifact latent must contain 14 finite values")
        if mode_payload.get("robust_grasp_pass") is not True:
            raise RuntimeError("hold-hand artifact mode is not an audited robust grasp pass")
        hold_hand_action = torch.tensor(
            latent, dtype=torch.float32, device=dev
        ).unsqueeze(0).expand(n, -1)
        hold_hand_source = str(hold_path)
        hold_hand_sha256 = sha256_file(hold_path)
    total_step_limit = (
        args_cli.close_step_limit
        + args_cli.micro_steps
        + args_cli.micro_hold_steps
        + args_cli.lift_steps
        + args_cli.settle_steps
        + 10
    )
    trace: list[dict[str, object]] = []

    def fail(mask: torch.Tensor, reason: int) -> None:
        selected = mask & (stage != FAILED)
        failure_reason[selected] = reason
        stage[selected] = FAILED
        stage_step[selected] = 0

    for global_step in range(total_step_limit):
        # Once DLS leaves the close-only +/-0.12 rad arm envelope, its 131-D
        # normalized arm-offset observation intentionally exceeds the teacher
        # prior's domain.  Evaluate the prior only on CLOSE rows; lift rows keep
        # its exact hand latent while the wrapper owns absolute arm targets.
        environment_action = torch.zeros((n, 21), dtype=torch.float32, device=dev)
        environment_action[:, 7:] = teacher_hand_action
        lift_action_rows = (
            (stage == MICRO_RAMP)
            | (stage == MICRO_HOLD)
            | (stage == LIFT_RAMP)
            | (stage == SETTLE)
            | (stage == SUCCESS)
        )
        if bool(lift_action_rows.any()) and hold_hand_source is not None:
            blend = torch.clamp(
                handoff_age.float() / float(args_cli.hold_hand_blend_steps),
                0.0,
                1.0,
            ).unsqueeze(-1)
            blended_hand = teacher_hand_action + blend * (
                hold_hand_action - teacher_hand_action
            )
            environment_action[lift_action_rows, 7:] = blended_hand[lift_action_rows]
        close_rows = stage == WAIT_CLOSE
        if bool(close_rows.any()):
            _, close_action = teacher.compose_action(
                observation[close_rows], zero_residual[close_rows]
            )
            environment_action[close_rows] = close_action
        observations, _, returned_terminated, returned_timeout, _ = env.step(environment_action)
        observation = observations["policy"]
        if bool(returned_terminated.any()) or bool(returned_timeout.any()):
            raise RuntimeError("audit done-mask override failed; DirectRLEnv attempted an auto-reset")

        signals = u._compute_grasp_signals()
        force = signals["force_magnitude"].max(dim=-1).values
        clearance = u._object_true_min_z() - u._table_surface_z
        object_linear_speed = u.object.data.root_com_lin_vel_w.norm(dim=-1)
        object_angular_speed = u.object.data.root_com_ang_vel_w.norm(dim=-1)
        force_peak.copy_(torch.maximum(force_peak, force))
        true_clearance_peak.copy_(torch.maximum(true_clearance_peak, clearance))
        close_xy_drift_peak.copy_(torch.maximum(close_xy_drift_peak, u._coupled_xy_drift))
        close_rotation_drift_peak.copy_(
            torch.maximum(close_rotation_drift_peak, u._coupled_rotation_drift)
        )
        close_pose_escape_ever.logical_or_(u._coupled_pose_escape)
        close_arm_saturated_ever.logical_or_(u._coupled_arm_target_saturated_ever)
        max_object_linear_speed.copy_(
            torch.maximum(max_object_linear_speed, object_linear_speed)
        )
        max_object_angular_speed.copy_(
            torch.maximum(max_object_angular_speed, object_angular_speed)
        )

        waiting = stage == WAIT_CLOSE
        new_native_success = waiting & native_close_success_signal
        native_close_seen.logical_or_(new_native_success)
        conservative = (
            new_native_success
            & (~close_pose_escape_ever)
            & (~close_arm_saturated_ever)
            & (force_peak <= thresholds.max_force_n)
            & u._is_grasped
            & u._power_is_grasped
            & (~u._coupled_align_active)
            & signals["power_thumb_contact"]
            & (
                signals["power_legal_other_contact_count"]
                >= int(cfg.power_grasp_required_other_contacts)
            )
            & (signals["grasp_quality"] >= thresholds.min_grasp_quality)
            & (signals["power_grasp_quality"] >= thresholds.min_grasp_quality)
            & (signals["hold_quality"] >= thresholds.min_hold_quality)
        )
        rejected = new_native_success & ~conservative
        fail(rejected, FAIL_CLOSE_REJECTED)
        if bool(conservative.any()):
            conservative_handoff.logical_or_(conservative)
            handoff_step[conservative] = global_step + 1
            stage[conservative] = MICRO_RAMP
            stage_step[conservative] = 0
            commanded_arm[conservative] = u.dof_targets[conservative][:, u._arm_ids_t]
            servo_lower[conservative] = u.dof_targets[conservative][:, servo_joint_ids]
            servo_upper[conservative] = torch.minimum(
                servo_lower[conservative] + args_cli.grip_servo_range,
                u.dof_upper[conservative][:, servo_joint_ids],
            )
            servo_target[conservative] = servo_lower[conservative]
            palm_body_start[conservative] = u.robot.data.body_pos_w[conservative, u.palm_idx]
            palm_quat_target[conservative] = u.robot.data.body_quat_w[
                conservative, u.palm_idx
            ]
            palm_center_start_z[conservative] = u.palm_center_w[conservative, 2]
            clearance_start[conservative] = clearance[conservative]
            rel_pos, rel_quat = subtract_frame_transforms(
                u.robot.data.body_pos_w[:, u.palm_idx],
                u.robot.data.body_quat_w[:, u.palm_idx],
                u.object_pos_w,
                u.object_quat_w,
            )
            object_in_palm_start_pos[conservative] = rel_pos[conservative]
            object_in_palm_start_quat[conservative] = rel_quat[conservative]

        # A native failure before handoff and the fixed 150-step close budget are
        # terminal for this exact replica.  Post-handoff native close failures are
        # expected because normal lift exceeds the close option's 15 mm window.
        native_close_fail = waiting & native_close_failure_signal
        fail(native_close_fail, FAIL_CLOSE_REJECTED)
        fail(waiting & native_timeout_signal, FAIL_CLOSE_TIMEOUT)
        if global_step + 1 >= args_cli.close_step_limit:
            fail(stage == WAIT_CLOSE, FAIL_CLOSE_TIMEOUT)

        lift_active = (
            (stage == MICRO_RAMP)
            | (stage == MICRO_HOLD)
            | (stage == LIFT_RAMP)
            | (stage == SETTLE)
            | (stage == SUCCESS)
        )
        if bool(lift_active.any()):
            handoff_age[lift_active] += 1
            rel_pos, rel_quat = subtract_frame_transforms(
                u.robot.data.body_pos_w[:, u.palm_idx],
                u.robot.data.body_quat_w[:, u.palm_idx],
                u.object_pos_w,
                u.object_quat_w,
            )
            relative_position_drift = (rel_pos - object_in_palm_start_pos).norm(dim=-1)
            quat_dot = (rel_quat * object_in_palm_start_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
            relative_rotation_drift = 2.0 * torch.acos(quat_dot)
            max_relative_position_drift.copy_(
                torch.maximum(max_relative_position_drift, relative_position_drift * lift_active)
            )
            max_relative_rotation_drift.copy_(
                torch.maximum(max_relative_rotation_drift, relative_rotation_drift * lift_active)
            )
            relative_escape = lift_active & (
                (relative_position_drift > args_cli.relative_position_drift_limit)
                | (relative_rotation_drift > args_cli.relative_rotation_drift_limit)
            )
            relative_pose_escape_ever.logical_or_(relative_escape)

            truth = classify_lift_truth(
                true_clearance_m=clearance,
                is_grasped=u._is_grasped,
                power_latched=u._power_is_grasped,
                grasp_quality=signals["grasp_quality"],
                power_grasp_quality=signals["power_grasp_quality"],
                hold_quality=signals["hold_quality"],
                max_finger_force_n=force,
                object_linear_speed_m_s=object_linear_speed,
                object_angular_speed_rad_s=object_angular_speed,
                thresholds=thresholds,
                require_power_contract=args_cli.lift_contract == "dual_power",
            )
            trajectory_unsafe_force_ever.logical_or_(lift_active & ~truth["safe_force"])
            airborne_unlatched_ever.logical_or_(lift_active & truth["airborne_unlatched"])
            transport_broken = lift_active & (
                (signals["hold_quality"] < float(cfg.grasp_quality_low))
                | (signals["grasp_quality"] < float(cfg.grasp_quality_low))
                | (
                    (signals["power_grasp_quality"] < float(cfg.power_grasp_quality_low))
                    if args_cli.lift_contract == "dual_power"
                    else torch.zeros(n, dtype=torch.bool, device=dev)
                )
            )
            transport_break_steps = torch.where(
                transport_broken,
                transport_break_steps + 1,
                torch.where(lift_active, torch.zeros_like(transport_break_steps), transport_break_steps),
            )
            min_grasp_quality[lift_active] = torch.minimum(
                min_grasp_quality[lift_active], signals["grasp_quality"][lift_active]
            )
            min_power_grasp_quality[lift_active] = torch.minimum(
                min_power_grasp_quality[lift_active],
                signals["power_grasp_quality"][lift_active],
            )
            min_hold_quality[lift_active] = torch.minimum(
                min_hold_quality[lift_active], signals["hold_quality"][lift_active]
            )

            # Permanent trajectory violations outrank a stable/success frame from
            # the same simulation step.
            fail(lift_active & trajectory_unsafe_force_ever, FAIL_FORCE)
            fail(lift_active & airborne_unlatched_ever, FAIL_AIRBORNE_UNLATCH)
            fail(lift_active & dls_arm_limit_ever, FAIL_ARM_LIMIT)
            fail(lift_active & relative_pose_escape_ever, FAIL_RELATIVE_POSE)
            fail(lift_active & (transport_break_steps >= 2), FAIL_TRANSPORT_BREAK)
            dropped = lift_active & (
                u.object_pos_w[:, 2] < (u.object_default_z - cfg.drop_height)
            )
            fail(dropped, FAIL_DROP)

            micro_ramp = stage == MICRO_RAMP
            stage_step[micro_ramp] += 1
            finished_micro_ramp = micro_ramp & (stage_step >= args_cli.micro_steps)
            stage[finished_micro_ramp] = MICRO_HOLD
            stage_step[finished_micro_ramp] = 0

            micro_hold = stage == MICRO_HOLD
            micro_stable_count = torch.where(
                micro_hold & truth["micro_stable"],
                micro_stable_count + 1,
                torch.where(micro_hold, torch.zeros_like(micro_stable_count), micro_stable_count),
            )
            stage_step[micro_hold] += 1
            new_micro_pass = micro_hold & (micro_stable_count >= args_cli.stable_steps)
            micro_pass.logical_or_(new_micro_pass)
            stage[new_micro_pass] = LIFT_RAMP
            stage_step[new_micro_pass] = 0
            fail(
                (stage == MICRO_HOLD) & (stage_step >= args_cli.micro_hold_steps),
                FAIL_MICRO_UNSTABLE,
            )

            lift_ramp = stage == LIFT_RAMP
            stage_step[lift_ramp] += 1
            finished_lift_ramp = lift_ramp & (stage_step >= args_cli.lift_steps)
            stage[finished_lift_ramp] = SETTLE
            stage_step[finished_lift_ramp] = 0

            settle = stage == SETTLE
            final_stable_count = torch.where(
                settle & truth["success_stable"],
                final_stable_count + 1,
                torch.where(settle, torch.zeros_like(final_stable_count), final_stable_count),
            )
            stage_step[settle] += 1
            new_success = settle & (final_stable_count >= args_cli.stable_steps)
            stage[new_success] = SUCCESS
            stage_step[new_success] = 0
            fail(
                (stage == SETTLE) & (stage_step >= args_cli.settle_steps),
                FAIL_FINAL_UNSTABLE,
            )

        if global_step % 25 == 0 or bool((stage == SUCCESS).any()):
            trace.append(
                {
                    "global_step": global_step + 1,
                    "stage_counts": _counts(stage, STAGE_NAMES),
                    "native_close_seen": int(native_close_seen.sum().item()),
                    "conservative_handoff": int(conservative_handoff.sum().item()),
                    "micro_pass": int(micro_pass.sum().item()),
                    "success": int((stage == SUCCESS).sum().item()),
                    "true_clearance_max_m": float(clearance.max().item()),
                    "force_max_n": float(force.max().item()),
                }
            )
            print(
                f"step={global_step + 1:3d} stages={trace[-1]['stage_counts']} "
                f"close={int(native_close_seen.sum())}/{n} "
                f"handoff={int(conservative_handoff.sum())} micro={int(micro_pass.sum())} "
                f"success={int((stage == SUCCESS).sum())} "
                f"clear_max={float(clearance.max()):.4f}m force_max={float(force.max()):.1f}N",
                flush=True,
            )

        if bool(((stage == SUCCESS) | (stage == FAILED)).all()):
            break

    final_clearance = u._object_true_min_z() - u._table_surface_z
    palm_rise = u.palm_center_w[:, 2] - palm_center_start_z
    clearance_gain = final_clearance - clearance_start
    success = stage == SUCCESS
    records = []
    for env_id in range(n):
        records.append(
            {
                "env_id": env_id,
                "stage": STAGE_NAMES[int(stage[env_id].item())],
                "failure_reason": FAILURE_NAMES[int(failure_reason[env_id].item())],
                "native_close_seen": bool(native_close_seen[env_id].item()),
                "conservative_handoff": bool(conservative_handoff[env_id].item()),
                "handoff_step": int(handoff_step[env_id].item()),
                "micro_pass": bool(micro_pass[env_id].item()),
                "success": bool(success[env_id].item()),
                "true_clearance_peak_m": float(true_clearance_peak[env_id].item()),
                "true_clearance_final_m": float(final_clearance[env_id].item()),
                "palm_rise_final_m": float(palm_rise[env_id].item()),
                "clearance_gain_final_m": float(clearance_gain[env_id].item()),
                "force_peak_n": float(force_peak[env_id].item()),
                "min_grasp_quality_after_handoff": float(min_grasp_quality[env_id].item()),
                "min_power_grasp_quality_after_handoff": float(
                    min_power_grasp_quality[env_id].item()
                ),
                "min_hold_quality_after_handoff": float(min_hold_quality[env_id].item()),
                "max_relative_position_drift_m": float(
                    max_relative_position_drift[env_id].item()
                ),
                "max_relative_rotation_drift_rad": float(
                    max_relative_rotation_drift[env_id].item()
                ),
                "max_arm_tracking_error_rad_l2": float(
                    max_arm_tracking_error[env_id].item()
                ),
                "max_tactile_servo_adjustment_rad": float(
                    servo_max_adjustment[env_id].item()
                ),
                "dls_arm_limit_ever": bool(dls_arm_limit_ever[env_id].item()),
                "airborne_unlatched_ever": bool(airborne_unlatched_ever[env_id].item()),
                "unsafe_force_ever": bool(trajectory_unsafe_force_ever[env_id].item()),
            }
        )

    result: dict[str, object] = {
        "status": "complete",
        "scope": "exact coupled teacher close -> DLS arm lift; no learned lift policy",
        "lift_contract": args_cli.lift_contract,
        "tactile_distal_servo": {
            "enabled": bool(args_cli.tactile_distal_servo),
            "force_target_n": args_cli.grip_force_target,
            "force_limit_n": args_cli.grip_force_limit,
            "joint_step_rad": args_cli.grip_servo_step,
            "joint_range_rad": args_cli.grip_servo_range,
            "joint_names_in_force_order": [
                fingertip_to_distal[name] for name in u.ee_names
            ],
        },
        "hold_hand_controller": {
            "source": hold_hand_source,
            "source_sha256": hold_hand_sha256,
            "mode": args_cli.hold_hand_mode if hold_hand_source is not None else None,
            "blend_steps": args_cli.hold_hand_blend_steps,
            "fallback": "exact coupled teacher hand latent",
        },
        "height_authority": "object true convex-mesh minimum minus true table surface",
        "teacher_artifact": str(teacher_path),
        "teacher_artifact_sha256": sha256_file(teacher_path),
        "curriculum_dataset": str(curriculum_path),
        "curriculum_dataset_sha256": sha256_file(curriculum_path),
        "seed": args_cli.seed,
        "num_envs": n,
        "steps_executed": global_step + 1,
        "native_close_success_count": int(native_close_seen.sum().item()),
        "conservative_handoff_count": int(conservative_handoff.sum().item()),
        "micro_5cm_stable_success_count": int(micro_pass.sum().item()),
        "full_20cm_stable_success_count": int(success.sum().item()),
        "full_20cm_stable_success_rate": float(success.float().mean().item()),
        "failure_counts": _counts(failure_reason, FAILURE_NAMES),
        "strict_contract": {
            "micro_true_clearance_m": thresholds.micro_clearance_m,
            "success_true_clearance_m": thresholds.success_clearance_m,
            "consecutive_stable_frames_each_stage": args_cli.stable_steps,
            "minimum_legacy_and_power_grasp_quality": thresholds.min_grasp_quality,
            "power_latch_and_quality_required_during_lift": (
                args_cli.lift_contract == "dual_power"
            ),
            "minimum_rigid_hold_quality": thresholds.min_hold_quality,
            "trajectory_force_limit_n": thresholds.max_force_n,
            "object_linear_speed_limit_m_s": thresholds.max_object_linear_speed_m_s,
            "object_angular_speed_limit_rad_s": thresholds.max_object_angular_speed_rad_s,
            "airborne_unlatch_is_permanent_failure": True,
            "arm_joint_limit_saturation_is_permanent_failure": True,
            "relative_position_drift_limit_m": args_cli.relative_position_drift_limit,
            "relative_rotation_drift_limit_rad": args_cli.relative_rotation_drift_limit,
        },
        "true_clearance_peak_m": _summary(true_clearance_peak),
        "true_clearance_final_m": _summary(final_clearance),
        "force_peak_n": _summary(force_peak),
        "palm_rise_minus_clearance_gain_m": _summary(palm_rise - clearance_gain),
        "records": records,
        "trace": trace,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "AUDIT VERDICT: "
        f"native close={result['native_close_success_count']}/{n}, "
        f"conservative handoff={result['conservative_handoff_count']}/{n}, "
        f"stable 5cm={result['micro_5cm_stable_success_count']}/{n}, "
        f"stable 20cm={result['full_20cm_stable_success_count']}/{n}",
        flush=True,
    )
    print(f"wrote {args_cli.output.resolve()}", flush=True)
    env.close()
    return result


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
