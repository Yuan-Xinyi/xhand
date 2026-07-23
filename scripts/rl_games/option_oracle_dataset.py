#!/usr/bin/env python3
"""Collect state-driven five-option pick-tool demonstrations.

The environment exposes 115 policy observations.  This collector appends the current option as a
five-way one-hot vector and stores a corrective 21-D oracle action at every visited state.  Option
advancement is exclusively guarded by measured pose/contact/transport state; the per-option step
limits below only terminate failed attempts and never advance the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--feasibility_json", required=True)
parser.add_argument("--rollout_checkpoint", default=None, help="optional 120-observation/21-action learner")
parser.add_argument("--teacher_probability", type=float, default=1.0)
parser.add_argument(
    "--retain_all",
    "--retain_all_episodes",
    dest="retain_all",
    action="store_true",
    help="retain failed as well as successful episodes",
)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="/tmp/pick_tool_option_oracle.pt")
parser.add_argument("--metrics", default="/tmp/pick_tool_option_oracle.json")

# Fixed option targets and state guards.
parser.add_argument("--hover_height", type=float, default=0.10)
parser.add_argument("--micro_height", type=float, default=0.04)
parser.add_argument("--target_height", type=float, default=0.24)
# HOVER is a collision-free waypoint 10cm above the handle; it need not meet the millimetre-scale
# initiation tolerance required before closing around the handle.
parser.add_argument("--hover_position_tolerance", type=float, default=0.020)
parser.add_argument("--hover_rotation_tolerance", type=float, default=0.20)
parser.add_argument("--position_tolerance", type=float, default=0.008)
parser.add_argument("--rotation_tolerance", type=float, default=0.12)
parser.add_argument("--max_palm_linear_speed", type=float, default=0.05)
parser.add_argument("--max_palm_angular_speed", type=float, default=0.50)
parser.add_argument("--approach_guard_steps", type=int, default=8)
parser.add_argument("--grasp_guard_steps", type=int, default=4)
parser.add_argument("--success_stable_steps", type=int, default=15)

# These limits only label an attempt as failed; they never cause a successful transition.
parser.add_argument("--hover_max_steps", type=int, default=400)
parser.add_argument("--descend_max_steps", type=int, default=400)
parser.add_argument("--close_max_steps", type=int, default=240)
parser.add_argument("--micro_max_steps", type=int, default=300)
parser.add_argument("--lift_max_steps", type=int, default=600)
parser.add_argument("--total_max_steps", type=int, default=1600)
parser.add_argument("--max_fallbacks", type=int, default=8)
parser.add_argument("--min_successes", type=int, default=1)

# Damped least-squares arm controller and tactile hand servo.
parser.add_argument("--damping", type=float, default=0.05)
parser.add_argument("--max_cart_step", type=float, default=0.004)
parser.add_argument("--max_rot_step", type=float, default=0.05)
parser.add_argument("--max_joint_step", type=float, default=0.04)
parser.add_argument("--grip_force_target", type=float, default=3.0)
parser.add_argument("--grip_force_limit", type=float, default=20.0)
parser.add_argument("--grip_servo_step", type=float, default=0.006)
parser.add_argument("--grip_servo_range", type=float, default=0.60)
parser.add_argument("--cem_population", type=int, default=4096)
parser.add_argument("--cem_elites", type=int, default=64)
parser.add_argument("--cem_iterations", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if not 0.0 <= args_cli.teacher_probability <= 1.0:
    parser.error("--teacher_probability must be in [0, 1]")
if args_cli.rollout_checkpoint is None and args_cli.teacher_probability != 1.0:
    parser.error("--teacher_probability differs from 1 but no --rollout_checkpoint was supplied")
if args_cli.target_height < 0.20 or args_cli.micro_height <= 0.015:
    parser.error("--target_height must be >=0.20m and --micro_height must be >0.015m")
if args_cli.hover_height <= 0.0:
    parser.error("--hover_height must be positive")
if (
    args_cli.hover_position_tolerance <= 0.0
    or args_cli.hover_rotation_tolerance <= 0.0
    or args_cli.position_tolerance <= 0.0
    or args_cli.rotation_tolerance <= 0.0
):
    parser.error("pose tolerances must be positive")
if args_cli.approach_guard_steps < 1 or args_cli.grasp_guard_steps < 1:
    parser.error("guard lengths must be positive")
if args_cli.success_stable_steps < 1 or args_cli.min_successes < 0:
    parser.error("success counts must be non-negative, with --success_stable_steps positive")
for option_limit in (
    args_cli.hover_max_steps,
    args_cli.descend_max_steps,
    args_cli.close_max_steps,
    args_cli.micro_max_steps,
    args_cli.lift_max_steps,
    args_cli.total_max_steps,
):
    if option_limit < 1:
        parser.error("option and total maximum steps must be positive")
if args_cli.max_fallbacks < 0:
    parser.error("--max_fallbacks must be non-negative")
if args_cli.cem_population < 2 or not 1 <= args_cli.cem_elites <= args_cli.cem_population:
    parser.error("CEM needs population>=2 and 1<=elites<=population")
if args_cli.cem_iterations < 1:
    parser.error("--cem_iterations must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn.functional as F
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401
from bc_pick_tool import MigratedActor, clone_state, load_torch
from xhand_inhand.tasks.direct.pick_tool_token.hybrid_action import (
    apply_asymmetric_joint_residual,
    invert_asymmetric_joint_residual,
)


OPTION_NAMES = ("HOVER", "DESCEND", "CLOSE", "MICRO", "LIFT_HOLD")
HOVER, DESCEND, CLOSE, MICRO, LIFT_HOLD = range(len(OPTION_NAMES))
BOUNDARY_NAMES = (
    "hover_start",
    "descend_start",
    "close_start",
    "micro_start",
    "lift_start",
    "micro_end",
    "mid_lift",
    "settle_start",
    "success",
)
FAILURE_NAMES = (
    "active",
    "success",
    "environment_termination",
    "hover_timeout",
    "descend_timeout",
    "close_timeout",
    "micro_timeout",
    "lift_timeout",
    "total_timeout",
    "too_many_fallbacks",
    "object_below_table",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _limit_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    return value * torch.clamp(limit / norm, max=1.0)


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().flatten()
    q = torch.quantile(flat, torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), device=flat.device))
    return dict(zip(("min", "p10", "median", "p90", "max"), (float(x) for x in q), strict=True))


def _checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    raw = load_torch(path)
    if not isinstance(raw, dict):
        raise TypeError("rollout checkpoint root is not a dictionary")
    if isinstance(raw.get("model"), dict):
        payload = raw
    else:
        payload = raw[0] if 0 in raw else raw.get("0")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise KeyError("rollout checkpoint must contain {'model': state_dict}, optionally below key 0")
    return clone_state(payload["model"])


def _capture_boundary(u) -> dict[str, torch.Tensor]:
    return {
        "joint_pos": u.robot.data.joint_pos.detach().clone(),
        "joint_vel": u.robot.data.joint_vel.detach().clone(),
        "dof_targets": u.dof_targets.detach().clone(),
        "object_local_pos": (u.object.data.root_pos_w - u.scene.env_origins).detach().clone(),
        "object_quat": u.object.data.root_quat_w.detach().clone(),
        "object_velocity": torch.cat(
            (u.object.data.root_com_lin_vel_w, u.object.data.root_com_ang_vel_w), dim=-1
        ).detach().clone(),
        "last_action": u.actions.detach().clone(),
        # The grasp latch is Markov-critical: restoring a latched lift boundary without
        # is_grasped=True makes the env's grasp shield block upward arm motion.  The env's
        # curriculum reset requires these three per-episode vectors, so save them too.
        "contact_steps": u._contact_steps.detach().clone(),
        "lost_contact_steps": u._lost_contact_steps.detach().clone(),
        "is_grasped": u._is_grasped.detach().clone(),
    }


def _write_boundary(u, state: dict[str, torch.Tensor]) -> None:
    all_ids = u.robot._ALL_INDICES
    u.robot.write_joint_state_to_sim(state["joint_pos"], state["joint_vel"], env_ids=all_ids)
    u.robot.set_joint_position_target(state["dof_targets"], env_ids=all_ids)
    u.dof_targets.copy_(state["dof_targets"])
    object_pose = torch.empty((u.num_envs, 7), device=u.device)
    object_pose[:, :3] = state["object_local_pos"] + u.scene.env_origins
    object_pose[:, 3:7] = state["object_quat"]
    u.object.write_root_pose_to_sim(object_pose, env_ids=all_ids)
    u.object.write_root_velocity_to_sim(state["object_velocity"], env_ids=all_ids)
    u.actions.copy_(state["last_action"])
    u.prev_actions.copy_(state["last_action"])
    # Restore the latch counters/flag when present so an internal snapshot round-trip matches
    # what the env's curriculum reset would load.
    if "contact_steps" in state:
        u._contact_steps.copy_(state["contact_steps"].to(dtype=u._contact_steps.dtype))
        u._lost_contact_steps.copy_(state["lost_contact_steps"].to(dtype=u._lost_contact_steps.dtype))
        u._is_grasped.copy_(state["is_grasped"].to(dtype=u._is_grasped.dtype))
    u.scene.write_data_to_sim()
    u.sim.forward()
    u.scene.update(dt=u.physics_dt)


def _clear_episode_state(u) -> None:
    u.episode_length_buf.zero_()
    u._contact_steps.zero_()
    u._lost_contact_steps.zero_()
    u._is_grasped.zero_()
    u._grasp_bonus_given.zero_()
    u._safe_grasp_steps.zero_()
    u._success_paid.zero_()
    u._success_steps.zero_()
    u._is_success.zero_()
    u._potential_initialized.zero_()
    u._hard_force_steps.zero_()
    u._overforce_steps.zero_()


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args_cli.seed)
    artifact_path = Path(args_cli.feasibility_json)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    hybrid = artifact["results"]["hybrid14"]
    if not hybrid.get("robust_grasp_pass", False):
        raise RuntimeError("the selected hybrid14 feasibility result is not a robust grasp")

    cfg = parse_env_cfg("Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.episode_length_s = 120.0
    cfg.terminate_on_drop = False
    cfg.success_hold_steps = 100000
    # DirectRLEnv auto-resets inside ``step`` before returning.  Keep the controller shield, but
    # disable only the environment's force done so this collector can measure and terminate the
    # offending post-step state itself instead of accidentally reading the next reset state.
    cfg.tactile_hard_terminate_steps = args_cli.total_max_steps + 1
    cfg.tactile_terminate_steps = args_cli.total_max_steps + 1
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    obs, _ = env.reset()

    rollout_actor = None
    rollout_path = Path(args_cli.rollout_checkpoint) if args_cli.rollout_checkpoint else None
    if rollout_path is not None:
        rollout_actor = MigratedActor(_checkpoint_model(rollout_path)).to(dev).eval()
        if rollout_actor.observation_dim != 120 or rollout_actor.action_dim != 21:
            raise RuntimeError(
                f"DAgger rollout actor must use 120 observations and 21 actions, got "
                f"{rollout_actor.observation_dim}/{rollout_actor.action_dim}"
            )
        print(
            f"DAgger learner={rollout_path.resolve()} teacher_probability="
            f"{args_cli.teacher_probability:.3f}",
            flush=True,
        )

    # Calibrate the palm pose in the object's frame from the physically proven pregrasp, then
    # restore the random reset exactly.  The reference snapshot is never used as an episode reset.
    initial_state = _capture_boundary(u)
    reference_joint = torch.tensor(artifact["pregrasp"]["joint_pos"], device=dev).repeat(n, 1)
    reference_state = {
        "joint_pos": reference_joint,
        "joint_vel": torch.zeros_like(reference_joint),
        "dof_targets": reference_joint.clone(),
        "object_local_pos": torch.tensor(
            artifact["pregrasp"]["object_local_pos"], device=dev
        ).repeat(n, 1),
        "object_quat": torch.tensor(artifact["pregrasp"]["object_quat"], device=dev).repeat(n, 1),
        "object_velocity": torch.zeros((n, 6), device=dev),
        "last_action": torch.zeros_like(u.actions),
    }
    _write_boundary(u, reference_state)
    u._compute_intermediate_values()
    reference_object_pos = reference_state["object_local_pos"] + u.scene.env_origins
    palm_in_object_pos, palm_in_object_quat = subtract_frame_transforms(
        reference_object_pos,
        reference_state["object_quat"],
        u.robot.data.body_pos_w[:, u.palm_idx],
        u.robot.data.body_quat_w[:, u.palm_idx],
    )
    palm_in_object_pos = palm_in_object_pos.detach().clone()
    palm_in_object_quat = palm_in_object_quat.detach().clone()
    _write_boundary(u, initial_state)
    _clear_episode_state(u)
    u._compute_intermediate_values()
    obs = u._get_observations()

    hand_names = [u.robot.joint_names[i] for i in u._hand_ids_t.tolist()]
    if hand_names != artifact["hand_joint_names"]:
        raise RuntimeError("artifact hand-joint order differs from the runtime articulation")
    hand_lower = u.dof_lower[:, u._hand_ids_t]
    hand_upper = u.dof_upper[:, u._hand_ids_t]

    def fit_formal_pregrasp_action(target_hand: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(artifact["pregrasp"]["token"], device=dev).unsqueeze(0)
        sigma = torch.full_like(center, 0.35)
        lower_one = hand_lower[:1]
        upper_one = hand_upper[:1]
        non_distal = torch.ones(len(hand_names), dtype=torch.bool, device=dev)
        non_distal[u._distal_hand_ids] = False
        desired = target_hand[:1]
        best_token = center.clone()
        best_loss = torch.full((), float("inf"), device=dev)
        for _ in range(args_cli.cem_iterations):
            candidates = (
                center
                + sigma
                * torch.randn((args_cli.cem_population, u._n_tokens), device=dev)
            ).clamp(-1.0, 1.0)
            candidates[0] = best_token[0]
            decoded = u.retarget.retarget_from_unit_action(candidates)[:, u._retarget2isaac]
            loss = (decoded[:, non_distal] - desired[:, non_distal]).square().mean(dim=-1)
            iteration_best = torch.argmin(loss)
            if loss[iteration_best] < best_loss:
                best_loss = loss[iteration_best]
                best_token = candidates[iteration_best : iteration_best + 1].clone()
            elite = candidates[torch.topk(loss, k=args_cli.cem_elites, largest=False).indices]
            center = elite.mean(dim=0, keepdim=True)
            sigma = elite.std(dim=0, keepdim=True).clamp_min(0.01)

        token_target = u.retarget.retarget_from_unit_action(best_token)[:, u._retarget2isaac]
        token_target = torch.maximum(torch.minimum(token_target, upper_one), lower_one)
        residual = invert_asymmetric_joint_residual(
            desired, token_target, lower_one, upper_one, u._distal_hand_ids
        )
        decoded, _ = apply_asymmetric_joint_residual(
            token_target, lower_one, upper_one, residual, u._distal_hand_ids
        )
        error = decoded - desired
        print(
            f"formal pregrasp fit rmse={float(error.square().mean().sqrt()):.4f}rad "
            f"max={float(error.abs().max()):.4f}rad",
            flush=True,
        )
        return torch.cat((best_token, residual), dim=-1).repeat(n, 1)

    pregrasp_hand_action = fit_formal_pregrasp_action(reference_joint[:, u._hand_ids_t])
    latent = torch.tensor(hybrid["latent"], device=dev).repeat(n, 1)
    token = latent[:, : u._n_tokens]
    token_base = u.retarget.retarget_from_unit_action(token)[:, u._retarget2isaac]
    token_base = torch.maximum(torch.minimum(token_base, hand_upper), hand_lower)
    expected_target = torch.tensor(hybrid["target"], device=dev).repeat(n, 1)
    decoded_target, _ = apply_asymmetric_joint_residual(
        token_base,
        hand_lower,
        hand_upper,
        latent[:, u._n_tokens :],
        u._distal_hand_ids,
    )
    if float((decoded_target - expected_target).abs().max()) >= 1.0e-6:
        raise RuntimeError("hybrid artifact no longer decodes to its saved hand target")

    desired_distal = expected_target.index_select(1, u._distal_hand_ids).clone()
    servo_lower = desired_distal.clone()
    servo_upper = torch.minimum(
        servo_lower + args_cli.grip_servo_range,
        hand_upper.index_select(1, u._distal_hand_ids),
    )
    fingertip_to_distal = {
        "thumb_rota_link2": "thumb_joint2",
        "index_rota_link2": "index_joint2",
        "mid_link2": "middle_joint1",
        "ring_link2": "ring_joint1",
        "pinky_link2": "pinky_joint1",
    }
    distal_names = list(cfg.distal_residual_joint_names)
    force_to_residual = [distal_names.index(fingertip_to_distal[name]) for name in u.ee_names]

    def grip_hand_action() -> torch.Tensor:
        desired = token_base.clone()
        desired.index_copy_(1, u._distal_hand_ids, desired_distal)
        residual = invert_asymmetric_joint_residual(
            desired, token_base, hand_lower, hand_upper, u._distal_hand_ids
        )
        return torch.cat((token, residual), dim=-1)

    def update_grip(force: torch.Tensor, mask: torch.Tensor) -> None:
        # A DAgger label must be reproducible from the visited state, not a hidden oracle integrator.
        # Decode the action that was actually executed (it is part of obs115), then apply one tactile
        # correction to its absolute distal target.  This retains the position preload needed to
        # support the hammer, while learner-executed actions cannot desynchronize an internal servo.
        executed_token = u.actions[:, u._n_arm : u._n_arm + u._n_tokens]
        executed_residual = u.actions[:, u._n_arm + u._n_tokens :]
        executed_base = u.retarget.retarget_from_unit_action(executed_token)[:, u._retarget2isaac]
        executed_target, _ = apply_asymmetric_joint_residual(
            executed_base,
            hand_lower,
            hand_upper,
            executed_residual,
            u._distal_hand_ids,
        )
        executed_distal = executed_target.index_select(1, u._distal_hand_ids)
        delta = torch.zeros_like(desired_distal)
        for force_index, residual_index in enumerate(force_to_residual):
            delta[:, residual_index] = args_cli.grip_servo_step * (
                (force[:, force_index] < args_cli.grip_force_target).float()
                - (force[:, force_index] > args_cli.grip_force_limit).float()
            )
        executed_distal = torch.maximum(torch.minimum(executed_distal, servo_upper), servo_lower)
        proposed = torch.maximum(torch.minimum(executed_distal + delta, servo_upper), servo_lower)
        desired_distal[mask] = proposed[mask]

    eye6 = torch.eye(6, device=dev).unsqueeze(0)

    def dls_pose_action(desired_pos: torch.Tensor, desired_quat: torch.Tensor) -> torch.Tensor:
        u._compute_intermediate_values()
        pos_error, rot_error = compute_pose_error(
            u.robot.data.body_pos_w[:, u.palm_idx],
            u.robot.data.body_quat_w[:, u.palm_idx],
            desired_pos,
            desired_quat,
            rot_error_type="axis_angle",
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
        return (delta_q / (cfg.act_moving_average * cfg.action_scale)).clamp(-1.0, 1.0)

    def calibrated_targets() -> tuple[torch.Tensor, torch.Tensor]:
        return combine_frame_transforms(
            u.object.data.root_pos_w,
            u.object.data.root_quat_w,
            palm_in_object_pos,
            palm_in_object_quat,
        )

    def pose_guards() -> dict[str, torch.Tensor]:
        target_pos, target_quat = calibrated_targets()
        hover_pos = target_pos.clone()
        hover_pos[:, 2] += args_cli.hover_height
        palm_pos = u.robot.data.body_pos_w[:, u.palm_idx]
        palm_quat = u.robot.data.body_quat_w[:, u.palm_idx]
        hover_pos_error, hover_rot_error = compute_pose_error(
            palm_pos, palm_quat, hover_pos, target_quat, rot_error_type="axis_angle"
        )
        descend_pos_error, descend_rot_error = compute_pose_error(
            palm_pos, palm_quat, target_pos, target_quat, rot_error_type="axis_angle"
        )
        palm_lin_speed = u.robot.data.body_com_lin_vel_w[:, u.palm_idx].norm(dim=-1)
        palm_ang_speed = u.robot.data.body_com_ang_vel_w[:, u.palm_idx].norm(dim=-1)
        low_speed = (
            (palm_lin_speed < args_cli.max_palm_linear_speed)
            & (palm_ang_speed < args_cli.max_palm_angular_speed)
        )
        distances = u._curr_fingertip_distances
        other_distances = distances[:, u._other_ee_idx]
        _, nearest_indices = torch.topk(other_distances, k=2, dim=1, largest=False)
        other_alignment = u._finger_align[:, u._other_ee_idx]
        alignment = (
            u._finger_align[:, u._thumb_ee_idx]
            + torch.gather(other_alignment, 1, nearest_indices).sum(dim=-1)
        ) / 3.0
        to_handle = u.handle_center_w - u.palm_center_w
        to_handle = to_handle / to_handle.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        palm_facing = 0.5 * (1.0 + (u.palm_normal_w * to_handle).sum(dim=-1))
        return {
            "hover_pos": hover_pos_error.norm(dim=-1),
            "hover_rot": hover_rot_error.norm(dim=-1),
            "descend_pos": descend_pos_error.norm(dim=-1),
            "descend_rot": descend_rot_error.norm(dim=-1),
            "low_speed": low_speed,
            "alignment": alignment,
            "palm_facing": palm_facing,
        }

    option = torch.full((n,), HOVER, dtype=torch.long, device=dev)
    option_steps = torch.zeros(n, dtype=torch.long, device=dev)
    total_steps = torch.zeros(n, dtype=torch.long, device=dev)
    guard_steps = torch.zeros(n, dtype=torch.long, device=dev)
    stable_steps = torch.zeros(n, dtype=torch.long, device=dev)
    fallback_count = torch.zeros(n, dtype=torch.long, device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    success = torch.zeros(n, dtype=torch.bool, device=dev)
    outcome = torch.zeros(n, dtype=torch.long, device=dev)
    terminal_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    terminal_clearance = torch.full((n,), float("nan"), device=dev)
    terminal_grasp_quality = torch.full((n,), float("nan"), device=dev)
    terminal_force_max = torch.full((n,), float("nan"), device=dev)
    max_clearance = torch.full((n,), -float("inf"), device=dev)
    force_peak = torch.zeros((n, len(u.ee_names)), device=dev)
    overforce_steps = torch.zeros(n, dtype=torch.long, device=dev)
    option_occupancy = torch.zeros((n, len(OPTION_NAMES)), dtype=torch.long, device=dev)
    transition_counts = torch.zeros(
        (len(OPTION_NAMES), len(OPTION_NAMES)), dtype=torch.long, device=dev
    )
    fallback_counts = torch.zeros((n, 2), dtype=torch.long, device=dev)
    executed_teacher_rows = 0
    executed_total_rows = 0

    close_anchor_pos = u.robot.data.body_pos_w[:, u.palm_idx].detach().clone()
    close_anchor_quat = u.robot.data.body_quat_w[:, u.palm_idx].detach().clone()

    initial_boundary = _capture_boundary(u)
    boundary_storage = {
        name: {key: torch.zeros_like(value) for key, value in initial_boundary.items()}
        for name in BOUNDARY_NAMES
    }
    boundary_valid = {
        name: torch.zeros(n, dtype=torch.bool, device=dev) for name in BOUNDARY_NAMES
    }

    def save_boundary(name: str, mask: torch.Tensor) -> None:
        new = mask & ~boundary_valid[name]
        if not bool(new.any()):
            return
        state = _capture_boundary(u)
        for key, value in state.items():
            boundary_storage[name][key][new] = value[new]
        boundary_valid[name] |= new

    save_boundary("hover_start", active)

    record_obs: list[torch.Tensor] = []
    record_action: list[torch.Tensor] = []
    record_phase: list[torch.Tensor] = []
    record_active: list[torch.Tensor] = []

    option_limits = torch.tensor(
        (
            args_cli.hover_max_steps,
            args_cli.descend_max_steps,
            args_cli.close_max_steps,
            args_cli.micro_max_steps,
            args_cli.lift_max_steps,
        ),
        device=dev,
        dtype=torch.long,
    )
    timeout_outcomes = torch.tensor((3, 4, 5, 6, 7), device=dev, dtype=torch.long)

    print(f"state-driven option oracle: {n} random-reset environments", flush=True)
    for global_step in range(args_cli.total_max_steps):
        if not bool(active.any()):
            break
        policy_obs = obs["policy"]
        if policy_obs.shape[1] != 115:
            raise RuntimeError(f"environment observation changed: expected 115, got {policy_obs.shape[1]}")
        option_one_hot = F.one_hot(option, num_classes=len(OPTION_NAMES)).to(policy_obs.dtype)
        actor_obs = torch.cat((policy_obs.clamp(-5.0, 5.0), option_one_hot), dim=-1)

        target_pos, target_quat = calibrated_targets()
        hover_pos = target_pos.clone()
        hover_pos[:, 2] += args_cli.hover_height
        desired_pos = target_pos.clone()
        desired_quat = target_quat.clone()
        hover_mask = option == HOVER
        desired_pos[hover_mask] = hover_pos[hover_mask]
        transport_mask = option >= CLOSE
        desired_pos[transport_mask] = close_anchor_pos[transport_mask]
        desired_quat[transport_mask] = close_anchor_quat[transport_mask]
        micro_mask = option == MICRO
        lift_mask = option == LIFT_HOLD
        desired_pos[micro_mask, 2] += args_cli.micro_height
        desired_pos[lift_mask, 2] += args_cli.target_height

        teacher_action = torch.zeros((n, cfg.action_space), device=dev)
        teacher_action[:, : u._n_arm] = dls_pose_action(desired_pos, desired_quat)
        approach_mask = option <= DESCEND
        teacher_action[approach_mask, u._n_arm :] = pregrasp_hand_action[approach_mask]
        teacher_action[~approach_mask, u._n_arm :] = grip_hand_action()[~approach_mask]
        teacher_action[~active] = 0.0

        record_obs.append(actor_obs.detach().clone())
        record_action.append(teacher_action.detach().clone())
        record_phase.append(option.detach().clone())
        record_active.append(active.detach().clone())
        option_occupancy[torch.arange(n, device=dev), option] += active.long()
        option_steps[active] += 1
        total_steps[active] += 1

        executed = teacher_action
        if rollout_actor is not None:
            learner_action = rollout_actor(actor_obs).clamp(-1.0, 1.0)
            teacher_mask = (torch.rand(n, device=dev) < args_cli.teacher_probability) & active
            executed = torch.where(teacher_mask.unsqueeze(-1), teacher_action, learner_action)
            executed[~active] = 0.0
            executed_teacher_rows += int(teacher_mask.sum())
            executed_total_rows += int(active.sum())
        obs, _, terminated, truncated, _ = env.step(executed)
        u._compute_intermediate_values()
        signals = u._compute_grasp_signals()
        force = signals["force_magnitude"]
        force_max = force.max(dim=-1).values
        clearance = u._object_true_min_z() - u._table_surface_z
        max_clearance = torch.maximum(max_clearance, torch.where(active, clearance, max_clearance))
        force_peak = torch.maximum(force_peak, torch.where(active.unsqueeze(-1), force, force_peak))

        overforce = force_max > cfg.tactile_terminate_force_limit
        overforce_steps = torch.where(
            active & overforce, overforce_steps + 1, torch.zeros_like(overforce_steps)
        )
        unsafe_force = active & (overforce_steps >= 2)
        if bool(unsafe_force.any()):
            outcome[unsafe_force] = 2
            terminal_step[unsafe_force] = global_step
            terminal_clearance[unsafe_force] = clearance[unsafe_force]
            terminal_grasp_quality[unsafe_force] = signals["grasp_quality"][unsafe_force]
            terminal_force_max[unsafe_force] = force_max[unsafe_force]
            active &= ~unsafe_force

        closure_active = active & (option >= CLOSE)
        update_grip(force, closure_active)
        pose = pose_guards()
        previous = option.clone()
        next_option = previous.clone()
        thumb_and_other = signals["thumb_contact"] & (signals["other_contact_count"] >= 1)
        provisional_grasp = (
            (signals["close_quality"] >= 0.20)
            & thumb_and_other
            & (signals["hold_quality"] >= 0.50)
            & (force_max <= 30.0)
        )
        transport_quality = torch.minimum(signals["quality"], signals["hold_quality"])

        micro_fallback = active & (previous == MICRO) & ~provisional_grasp
        lift_fallback = active & (previous == LIFT_HOLD) & (
            (~u._is_grasped) | (transport_quality < cfg.grasp_quality_low)
        )
        next_option[micro_fallback | lift_fallback] = CLOSE
        fallback_counts[:, 0] += micro_fallback.long()
        fallback_counts[:, 1] += lift_fallback.long()
        fallback_count += (micro_fallback | lift_fallback).long()

        hover_ready = (
            active
            & (previous == HOVER)
            & (pose["hover_pos"] < args_cli.hover_position_tolerance)
            & (pose["hover_rot"] < args_cli.hover_rotation_tolerance)
            & pose["low_speed"]
        )
        descend_ready = (
            active
            & (previous == DESCEND)
            & (pose["descend_pos"] < args_cli.position_tolerance)
            & (pose["descend_rot"] < args_cli.rotation_tolerance)
            & pose["low_speed"]
            & (pose["palm_facing"] >= cfg.grasp_palm_facing_min)
            & (pose["alignment"] >= cfg.grasp_align_min)
        )
        close_ready = active & (previous == CLOSE) & provisional_grasp
        micro_ready = (
            active
            & (previous == MICRO)
            & ~micro_fallback
            & u._is_grasped
            & (transport_quality >= 0.35)
            & (clearance >= 0.015)
            & (force_max <= 30.0)
        )
        guarded = hover_ready | descend_ready | close_ready | micro_ready
        guard_steps = torch.where(guarded, guard_steps + 1, torch.zeros_like(guard_steps))
        hover_transition = hover_ready & (guard_steps >= args_cli.approach_guard_steps)
        descend_transition = descend_ready & (guard_steps >= args_cli.approach_guard_steps)
        close_transition = close_ready & (guard_steps >= args_cli.grasp_guard_steps)
        micro_transition = micro_ready & (guard_steps >= args_cli.grasp_guard_steps)
        next_option[hover_transition] = DESCEND
        next_option[descend_transition] = CLOSE
        next_option[close_transition] = MICRO
        next_option[micro_transition] = LIFT_HOLD

        changed = active & (next_option != previous)
        for source in range(len(OPTION_NAMES)):
            for target in range(len(OPTION_NAMES)):
                transition = changed & (previous == source) & (next_option == target)
                transition_counts[source, target] += transition.sum()

        close_entry = changed & (next_option == CLOSE)
        initial_close_entry = descend_transition
        close_anchor_pos[initial_close_entry] = u.robot.data.body_pos_w[
            initial_close_entry, u.palm_idx
        ]
        close_anchor_quat[initial_close_entry] = u.robot.data.body_quat_w[
            initial_close_entry, u.palm_idx
        ]
        save_boundary("descend_start", hover_transition)
        save_boundary("close_start", initial_close_entry)
        save_boundary("micro_start", close_transition)
        save_boundary("lift_start", micro_transition)
        save_boundary("micro_end", micro_transition)
        save_boundary(
            "mid_lift", active & (previous == LIFT_HOLD) & (clearance >= 0.10)
        )
        save_boundary(
            "settle_start",
            active & (previous == LIFT_HOLD) & (clearance >= cfg.lift_success_height),
        )
        guard_steps[changed] = 0
        option_steps[changed] = 0
        option = next_option

        slow = (
            (u.object.data.root_com_lin_vel_w.norm(dim=-1) < cfg.success_max_obj_lin_speed)
            & (u.object.data.root_com_ang_vel_w.norm(dim=-1) < cfg.success_max_obj_ang_speed)
        )
        strict = (
            active
            & (previous == LIFT_HOLD)
            & (clearance >= cfg.lift_success_height)
            & u._is_grasped
            & (signals["grasp_quality"] >= cfg.grasp_quality_high)
            & (force_max <= 30.0)
            & slow
        )
        stable_steps = torch.where(strict, stable_steps + 1, torch.zeros_like(stable_steps))
        newly_successful = active & (stable_steps >= args_cli.success_stable_steps)
        save_boundary("success", newly_successful)
        success |= newly_successful
        outcome[newly_successful] = 1
        terminal_step[newly_successful] = global_step

        still_active = active & ~newly_successful
        environment_failure = still_active & (terminated | truncated)
        below_table = still_active & (clearance < -0.03)
        too_many_fallbacks = still_active & (fallback_count > args_cli.max_fallbacks)
        total_timeout = still_active & (total_steps >= args_cli.total_max_steps)
        unchanged = option == previous
        option_timeout = (
            still_active
            & unchanged
            & (option_steps >= option_limits.index_select(0, option))
        )

        # Precedence records the most immediate physical failure before controller timeouts.
        outcome[option_timeout] = timeout_outcomes.index_select(0, option[option_timeout])
        outcome[total_timeout] = 8
        outcome[too_many_fallbacks] = 9
        outcome[below_table] = 10
        outcome[environment_failure] = 2
        failed = environment_failure | below_table | too_many_fallbacks | total_timeout | option_timeout
        terminal_step[failed] = global_step
        # Unexpected environment dones have already auto-reset, so their post-step physics is not a
        # terminal measurement.  Leave those fields NaN instead of recording a plausible reset pose.
        measured_terminal_event = (newly_successful | failed) & ~environment_failure
        terminal_clearance[measured_terminal_event] = clearance[measured_terminal_event]
        terminal_grasp_quality[measured_terminal_event] = signals["grasp_quality"][measured_terminal_event]
        terminal_force_max[measured_terminal_event] = force_max[measured_terminal_event]
        active &= ~(newly_successful | failed)

        if global_step % 100 == 0 or not bool(active.any()):
            occupancy = [int(option_occupancy[:, index].sum()) for index in range(len(OPTION_NAMES))]
            print(
                f"step={global_step:4d} active={int(active.sum())}/{n} "
                f"success={int(success.sum())}/{n} options="
                f"{[int(((option == index) & active).sum()) for index in range(len(OPTION_NAMES))]} "
                f"occupancy={occupancy}",
                flush=True,
            )

    # An exhausted Python loop is itself a total timeout (normally caught in-loop on the last step).
    if bool(active.any()):
        outcome[active] = 8
        terminal_step[active] = args_cli.total_max_steps - 1
        terminal_clearance[active] = u._object_true_min_z()[active] - u._table_surface_z
        exhausted_signals = u._compute_grasp_signals()
        terminal_grasp_quality[active] = exhausted_signals["grasp_quality"][active]
        terminal_force_max[active] = exhausted_signals["force_magnitude"][active].max(dim=-1).values
        active.zero_()

    success_ids = success.nonzero(as_tuple=False).squeeze(-1)
    selected_ids = (
        torch.arange(n, device=dev, dtype=torch.long) if args_cli.retain_all else success_ids
    )
    final_signals = u._compute_grasp_signals()
    final_clearance = u._object_true_min_z() - u._table_surface_z

    metrics = {
        "num_envs": n,
        "seed": args_cli.seed,
        "success_count": int(success.sum()),
        "retained_episode_count": int(selected_ids.numel()),
        "executed_teacher_fraction": (
            executed_teacher_rows / executed_total_rows if executed_total_rows else 1.0
        ),
        "outcome_names": list(FAILURE_NAMES),
        "outcome_counts": {
            FAILURE_NAMES[index]: int((outcome == index).sum())
            for index in range(1, len(FAILURE_NAMES))
        },
        "terminal_step": terminal_step.cpu().tolist(),
        "terminal_true_clearance": terminal_clearance.cpu().tolist(),
        "terminal_grasp_quality": terminal_grasp_quality.cpu().tolist(),
        "terminal_force_max": terminal_force_max.cpu().tolist(),
        "success_terminal_true_clearance": (
            _quantiles(terminal_clearance[success]) if bool(success.any()) else None
        ),
        "success_terminal_grasp_quality": (
            _quantiles(terminal_grasp_quality[success]) if bool(success.any()) else None
        ),
        "option_names": list(OPTION_NAMES),
        "option_occupancy": {
            OPTION_NAMES[index]: int(option_occupancy[:, index].sum())
            for index in range(len(OPTION_NAMES))
        },
        "option_occupancy_per_env": option_occupancy.cpu().tolist(),
        "transition_counts": transition_counts.cpu().tolist(),
        "fallback_counts": {
            "MICRO->CLOSE": int(fallback_counts[:, 0].sum()),
            "LIFT_HOLD->CLOSE": int(fallback_counts[:, 1].sum()),
        },
        "boundary_valid_counts": {
            name: int(valid.sum()) for name, valid in boundary_valid.items()
        },
        "max_true_clearance": _quantiles(max_clearance),
        # Earlier-successful envs continue simulating while other attempts finish, so these are
        # explicitly batch-end diagnostics.  ``terminal_*`` above is the success source of truth.
        "batch_end_true_clearance": final_clearance.cpu().tolist(),
        "batch_end_grasp_quality": final_signals["grasp_quality"].cpu().tolist(),
        "force_order": list(u.ee_names),
        "force_peak_per_finger": force_peak.max(dim=0).values.cpu().tolist(),
    }
    metrics_path = Path(args_cli.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if selected_ids.numel() == 0:
        metrics["dataset"] = None
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        env.close()
        raise RuntimeError(f"state-driven oracle produced no successful episodes; wrote {metrics_path}")

    obs_t = torch.stack(record_obs)
    action_t = torch.stack(record_action)
    phase_t = torch.stack(record_phase)
    active_t = torch.stack(record_active)
    episode_obs = []
    episode_action = []
    episode_phase = []
    episode_id = []
    episode_step = []
    offsets = [0]
    selected_id_list = selected_ids.tolist()
    for episode, env_id in enumerate(selected_id_list):
        rows = active_t[:, env_id]
        ep_obs = obs_t[rows, env_id]
        ep_action = action_t[rows, env_id]
        ep_phase = phase_t[rows, env_id]
        length = ep_obs.shape[0]
        episode_obs.append(ep_obs)
        episode_action.append(ep_action)
        episode_phase.append(ep_phase)
        episode_id.append(torch.full((length,), episode, dtype=torch.int32, device=dev))
        episode_step.append(torch.arange(length, dtype=torch.int32, device=dev))
        offsets.append(offsets[-1] + length)

    output_boundaries = {}
    if not args_cli.retain_all:
        for name in BOUNDARY_NAMES:
            selected_valid = boundary_valid[name][selected_ids]
            if not bool(selected_valid.all()):
                raise RuntimeError(f"successful episode is missing required boundary {name!r}")
            output_boundaries[name] = {
                key: value[selected_ids].cpu() for key, value in boundary_storage[name].items()
            }

    dataset = {
        "obs": torch.cat(episode_obs).float().cpu(),
        "action": torch.cat(episode_action).float().cpu(),
        "phase": torch.cat(episode_phase).to(torch.uint8).cpu(),
        "episode_id": torch.cat(episode_id).cpu(),
        "step": torch.cat(episode_step).cpu(),
        "episode_offsets": torch.tensor(offsets, dtype=torch.int64),
        "episode_success": success[selected_ids].cpu(),
        "episode_outcome": outcome[selected_ids].to(torch.int16).cpu(),
        "episode_terminal_clearance": terminal_clearance[selected_ids].cpu(),
        "episode_terminal_grasp_quality": terminal_grasp_quality[selected_ids].cpu(),
        "episode_terminal_force_max": terminal_force_max[selected_ids].cpu(),
        "boundaries": output_boundaries,
        "meta": {
            "format_version": 2,
            "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
            "observation_layout": (
                "legacy_prefix87|distal_action5|grasp_transport23|option_onehot5"
            ),
            "observation_dim": 120,
            "action_dim": 21,
            "option_names": list(OPTION_NAMES),
            "state_driven": True,
            "transition_guards": {
                "position_tolerance_m": args_cli.position_tolerance,
                "rotation_tolerance_rad": args_cli.rotation_tolerance,
                "hover_position_tolerance_m": args_cli.hover_position_tolerance,
                "hover_rotation_tolerance_rad": args_cli.hover_rotation_tolerance,
                "max_palm_linear_speed_mps": args_cli.max_palm_linear_speed,
                "max_palm_angular_speed_radps": args_cli.max_palm_angular_speed,
                "approach_guard_steps": args_cli.approach_guard_steps,
                "grasp_guard_steps": args_cli.grasp_guard_steps,
                "close_quality_min": 0.20,
                "hold_quality_min": 0.50,
                "transport_quality_min": 0.35,
                "micro_clearance_m": 0.015,
                "max_force_N": 30.0,
                "fallback": "MICRO or LIFT_HOLD loss of grasp returns to CLOSE",
            },
            "option_targets": {
                "hover_height": args_cli.hover_height,
                "micro_height": args_cli.micro_height,
                "target_height": args_cli.target_height,
            },
            "option_max_steps_are_failure_only": True,
            "option_max_steps": {
                OPTION_NAMES[index]: int(option_limits[index])
                for index in range(len(OPTION_NAMES))
            },
            "total_max_steps": args_cli.total_max_steps,
            "seed": args_cli.seed,
            "feasibility_json": str(artifact_path.resolve()),
            "feasibility_sha256": _sha256(artifact_path),
            "rollout_checkpoint": str(rollout_path.resolve()) if rollout_path else None,
            "rollout_checkpoint_sha256": _sha256(rollout_path) if rollout_path else None,
            "teacher_probability": args_cli.teacher_probability,
            "retain_all": args_cli.retain_all,
            "boundaries_omitted_for_retain_all": args_cli.retain_all,
            "outcome_names": list(FAILURE_NAMES),
        },
    }
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    metrics["transitions"] = int(dataset["obs"].shape[0])
    metrics["dataset"] = str(output_path.resolve())
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"wrote {output_path} episodes={selected_ids.numel()} transitions={dataset['obs'].shape[0]}",
        flush=True,
    )
    print(f"wrote {metrics_path}", flush=True)

    if success_ids.numel() < args_cli.min_successes:
        env.close()
        raise RuntimeError(
            f"oracle produced {success_ids.numel()} successes, below --min_successes="
            f"{args_cli.min_successes}"
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
