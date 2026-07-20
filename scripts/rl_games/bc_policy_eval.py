#!/usr/bin/env python3
"""Closed-loop physical evaluation for migrated/BC pick-tool actors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument(
    "--reach_checkpoint",
    default=None,
    help="optional legacy 87/16 reach actor; switch irreversibly to --checkpoint at pregrasp",
)
parser.add_argument("--switch_score", type=float, default=0.20)
parser.add_argument("--switch_hold_steps", type=int, default=1)
parser.add_argument("--tactile_close_servo", action="store_true")
parser.add_argument("--close_servo_arm_zero", action="store_true")
parser.add_argument("--close_servo_token_hold", action="store_true")
parser.add_argument("--close_servo_entry_proximity", type=float, default=0.02)
parser.add_argument("--close_servo_exit_proximity", type=float, default=0.01)
parser.add_argument("--close_servo_pregrasp_score", type=float, default=0.25)
parser.add_argument("--close_servo_require_touch", action="store_true")
parser.add_argument("--close_servo_low_force", type=float, default=3.0)
parser.add_argument("--close_servo_high_force", type=float, default=20.0)
parser.add_argument("--close_servo_step", type=float, default=0.005)
parser.add_argument("--close_servo_max_travel", type=float, default=0.10)
parser.add_argument("--close_servo_timeout", type=int, default=120)
parser.add_argument("--distal_adapter", default=None)
parser.add_argument("--distal_adapter_scale", type=float, default=1.0)
parser.add_argument("--distal_adapter_pregrasp_score", type=float, default=0.25)
parser.add_argument("--distal_adapter_timeout", type=int, default=120)
parser.add_argument(
    "--option_fsm",
    action="store_true",
    help="drive a 120-observation actor with a state-derived five-option one-hot suffix",
)
parser.add_argument(
    "--option_checkpoints",
    nargs=5,
    metavar=("HOVER", "DESCEND", "CLOSE", "MICRO", "LIFT_HOLD"),
    default=None,
    help="optional five 120/21 expert checkpoints selected by the option FSM",
)
parser.add_argument("--mode", choices=("fixed", "random"), default="fixed")
parser.add_argument("--feasibility_json", default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--stable_steps", type=int, default=15)
parser.add_argument("--curriculum_dataset", default=None)
parser.add_argument(
    "--curriculum_boundary",
    choices=("close_start", "lift_start", "micro_end", "mid_lift", "settle_start"),
    default="close_start",
)
parser.add_argument("--curriculum_reset_probability", type=float, default=1.0)
parser.add_argument("--curriculum_joint_noise", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="/tmp/pick_tool_bc_eval.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.mode == "fixed" and not args_cli.feasibility_json:
    parser.error("--feasibility_json is required in fixed mode")
if (
    args_cli.num_envs < 1
    or args_cli.steps < 1
    or args_cli.stable_steps < 1
    or args_cli.switch_hold_steps < 1
):
    parser.error("--num_envs, --steps, --stable_steps and --switch_hold_steps must be positive")
if not 0.0 <= args_cli.switch_score <= 1.0:
    parser.error("--switch_score must be in [0, 1]")
if args_cli.option_fsm and not args_cli.feasibility_json:
    parser.error("--feasibility_json is required with --option_fsm for palm/object calibration")
if args_cli.option_fsm and args_cli.reach_checkpoint:
    parser.error("--option_fsm and --reach_checkpoint are mutually exclusive")
if args_cli.option_checkpoints and not args_cli.option_fsm:
    parser.error("--option_checkpoints requires --option_fsm")
if (args_cli.close_servo_arm_zero or args_cli.close_servo_token_hold) and not args_cli.tactile_close_servo:
    parser.error("close-servo ablations require --tactile_close_servo")
if args_cli.tactile_close_servo and args_cli.option_fsm:
    parser.error("--tactile_close_servo currently supports only a single 115-observation actor")
if args_cli.distal_adapter and args_cli.option_fsm:
    parser.error("--distal_adapter currently supports only a single 115-observation actor")
if args_cli.distal_adapter and args_cli.tactile_close_servo:
    parser.error("--distal_adapter and --tactile_close_servo are separate ablations")
if not 0.0 <= args_cli.distal_adapter_scale <= 1.0 or args_cli.distal_adapter_timeout < 1:
    parser.error("distal adapter scale must be in [0,1] and timeout positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, subtract_frame_transforms

import xhand_inhand.tasks  # noqa: F401
from bc_pick_tool import MigratedActor, clone_state, load_torch
from distal_residual_adapter import StatefulCloseGate, load_adapter as load_distal_adapter
from tactile_close_controller import TactileCloseController


OPTION_NAMES = ("HOVER", "DESCEND", "CLOSE", "MICRO", "LIFT_HOLD")
HOVER, DESCEND, CLOSE, MICRO, LIFT_HOLD = range(len(OPTION_NAMES))
OPTION_HOVER_HEIGHT = 0.10
OPTION_HOVER_POSITION_TOLERANCE = 0.020
OPTION_HOVER_ROTATION_TOLERANCE = 0.20
OPTION_POSITION_TOLERANCE = 0.008
OPTION_ROTATION_TOLERANCE = 0.12
OPTION_MAX_PALM_LINEAR_SPEED = 0.05
OPTION_MAX_PALM_ANGULAR_SPEED = 0.50
OPTION_APPROACH_HOLD_STEPS = 8
OPTION_GRASP_HOLD_STEPS = 4


def _summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().flatten()
    q = torch.quantile(flat, torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), device=flat.device))
    return dict(zip(("min", "p10", "median", "p90", "max"), (float(x) for x in q), strict=True))


def _masked_summary(value: torch.Tensor, mask: torch.Tensor) -> dict[str, float] | None:
    selected = value[mask]
    return None if selected.numel() == 0 else _summary(selected)


def _checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    raw = load_torch(path)
    if not isinstance(raw, dict):
        raise TypeError("checkpoint root is not a dictionary")
    if isinstance(raw.get("model"), dict):
        payload = raw
    else:
        payload = raw[0] if 0 in raw else raw.get("0")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise KeyError("checkpoint must contain {'model': state_dict}, optionally below root key 0")
    return clone_state(payload["model"])


def _capture_physical_state(u) -> dict[str, torch.Tensor]:
    """Capture enough state to use an artifact pose as FK calibration without advancing time."""

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
    }


def _write_physical_state(u, state: dict[str, torch.Tensor]) -> None:
    """Write a batched physical state and refresh palm rigid-body kinematics."""

    all_ids = u.robot._ALL_INDICES
    u.robot.write_joint_state_to_sim(state["joint_pos"], state["joint_vel"], env_ids=all_ids)
    u.robot.set_joint_position_target(state["dof_targets"], env_ids=all_ids)
    u.dof_targets.copy_(state["dof_targets"])
    object_pose = torch.zeros((u.num_envs, 7), device=u.device)
    object_pose[:, :3] = state["object_local_pos"] + u.scene.env_origins
    object_pose[:, 3:7] = state["object_quat"]
    u.object.write_root_pose_to_sim(object_pose, env_ids=all_ids)
    u.object.write_root_velocity_to_sim(state["object_velocity"], env_ids=all_ids)
    u.actions.copy_(state["last_action"])
    u.prev_actions.copy_(state["last_action"])
    u.scene.write_data_to_sim()
    u.sim.forward()
    u.scene.update(dt=u.physics_dt)


def _artifact_state(u, artifact: dict) -> dict[str, torch.Tensor]:
    """Broadcast the fixed feasibility snapshot to every environment."""

    n = u.num_envs
    dev = u.device
    joint = torch.tensor(artifact["pregrasp"]["joint_pos"], device=dev).repeat(n, 1)
    token = torch.tensor(artifact["pregrasp"]["token"], device=dev).repeat(n, 1)
    last_action = torch.zeros_like(u.actions)
    last_action[:, u._n_arm : u._n_arm + u._n_tokens] = token
    return {
        "joint_pos": joint,
        "joint_vel": torch.zeros_like(joint),
        "dof_targets": joint.clone(),
        "object_local_pos": torch.tensor(
            artifact["pregrasp"]["object_local_pos"], device=dev
        ).repeat(n, 1),
        "object_quat": torch.tensor(
            artifact["pregrasp"]["object_quat"], device=dev
        ).repeat(n, 1),
        "object_velocity": torch.zeros((n, 6), device=dev),
        "last_action": last_action,
    }


def _clear_episode_state(u) -> None:
    """Clear environment-side grasp/reward history after a diagnostic state write."""

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


def _calibrate_palm_in_object(u, artifact: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure the proven pregrasp palm pose in the object's frame, then restore reset state."""

    initial_state = _capture_physical_state(u)
    reference_state = _artifact_state(u, artifact)
    _write_physical_state(u, reference_state)
    object_pos = reference_state["object_local_pos"] + u.scene.env_origins
    palm_in_object_pos, palm_in_object_quat = subtract_frame_transforms(
        object_pos,
        reference_state["object_quat"],
        u.robot.data.body_pos_w[:, u.palm_idx],
        u.robot.data.body_quat_w[:, u.palm_idx],
    )
    palm_in_object_pos = palm_in_object_pos.detach().clone()
    palm_in_object_quat = palm_in_object_quat.detach().clone()
    _write_physical_state(u, initial_state)
    _clear_episode_state(u)
    u._compute_intermediate_values()
    return palm_in_object_pos, palm_in_object_quat


def _pregrasp_score(u) -> torch.Tensor:
    """Geometry-only readiness score used by the reach-to-grasp option transition."""

    distances = u._curr_fingertip_distances
    other_distances = distances[:, u._other_ee_idx]
    nearest_distances, nearest_indices = torch.topk(other_distances, k=2, dim=1, largest=False)
    grasp_distance = (
        distances[:, u._thumb_ee_idx] + nearest_distances.sum(dim=-1)
    ) / 3.0

    other_alignment = u._finger_align[:, u._other_ee_idx]
    alignment = (
        u._finger_align[:, u._thumb_ee_idx]
        + torch.gather(other_alignment, 1, nearest_indices).sum(dim=-1)
    ) / 3.0
    to_handle = u.handle_center_w - u.palm_center_w
    to_handle = to_handle / to_handle.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    palm_facing = 0.5 * (1.0 + (u.palm_normal_w * to_handle).sum(dim=-1))
    clearance = u._object_true_min_z() - u._table_surface_z
    score = torch.exp(-grasp_distance / 0.025) * alignment * palm_facing
    return torch.where(clearance.abs() <= 0.005, score, torch.zeros_like(score))


def _option_pose_state(
    u,
    palm_in_object_pos: torch.Tensor,
    palm_in_object_quat: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pose and geometry guards for the HOVER and DESCEND transitions."""

    target_pos, target_quat = combine_frame_transforms(
        u.object.data.root_pos_w,
        u.object.data.root_quat_w,
        palm_in_object_pos,
        palm_in_object_quat,
    )
    hover_pos = target_pos.clone()
    hover_pos[:, 2] += OPTION_HOVER_HEIGHT
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
        (palm_lin_speed < OPTION_MAX_PALM_LINEAR_SPEED)
        & (palm_ang_speed < OPTION_MAX_PALM_ANGULAR_SPEED)
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
        "hover_position_error": hover_pos_error.norm(dim=-1),
        "hover_rotation_error": hover_rot_error.norm(dim=-1),
        "descend_position_error": descend_pos_error.norm(dim=-1),
        "descend_rotation_error": descend_rot_error.norm(dim=-1),
        "low_speed": low_speed,
        "alignment": alignment,
        "palm_facing": palm_facing,
    }


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args_cli.seed)
    checkpoint_path = Path(args_cli.checkpoint)
    model_state = _checkpoint_model(checkpoint_path)
    actor = MigratedActor(model_state).to(args_cli.device).eval()
    if actor.observation_dim not in (115, 120) or actor.action_dim != 21:
        raise RuntimeError(
            f"expected a 115-or-120/21 actor, got {actor.observation_dim}/{actor.action_dim}"
        )
    if args_cli.option_fsm and actor.observation_dim != 120:
        raise RuntimeError("--option_fsm requires a 120-observation actor")
    if actor.observation_dim == 120 and not args_cli.option_fsm:
        raise RuntimeError("a 120-observation actor requires --option_fsm")
    option_actors = None
    if args_cli.option_checkpoints:
        option_actors = []
        for option_name, path in zip(OPTION_NAMES, args_cli.option_checkpoints, strict=True):
            expert = MigratedActor(_checkpoint_model(Path(path))).to(args_cli.device).eval()
            if expert.observation_dim != 120 or expert.action_dim != 21:
                raise RuntimeError(
                    f"{option_name} expert must be 120/21, got "
                    f"{expert.observation_dim}/{expert.action_dim}"
                )
            option_actors.append(expert)
    reach_actor = None
    if args_cli.reach_checkpoint:
        reach_actor = MigratedActor(
            _checkpoint_model(Path(args_cli.reach_checkpoint))
        ).to(args_cli.device).eval()
        if reach_actor.observation_dim != 87 or reach_actor.action_dim != 16:
            raise RuntimeError(
                f"expected an 87/16 legacy reach actor, got "
                f"{reach_actor.observation_dim}/{reach_actor.action_dim}"
            )
    distal_adapter = None
    distal_adapter_payload = None
    if args_cli.distal_adapter:
        distal_adapter, distal_adapter_payload = load_distal_adapter(
            Path(args_cli.distal_adapter), checkpoint_path, args_cli.device
        )

    cfg = parse_env_cfg("Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.episode_length_s = 120.0
    cfg.terminate_on_drop = False
    cfg.success_hold_steps = 100000
    if args_cli.curriculum_dataset:
        cfg.curriculum_dataset = args_cli.curriculum_dataset
        cfg.curriculum_boundary = args_cli.curriculum_boundary
        cfg.curriculum_reset_probability = args_cli.curriculum_reset_probability
        cfg.curriculum_joint_noise = args_cli.curriculum_joint_noise
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    dev = u.device
    n = u.num_envs
    close_controller = None
    if args_cli.tactile_close_servo:
        close_controller = TactileCloseController(
            n,
            dev,
            entry_proximity=args_cli.close_servo_entry_proximity,
            exit_proximity=args_cli.close_servo_exit_proximity,
            low_force=args_cli.close_servo_low_force,
            high_force=args_cli.close_servo_high_force,
            step=args_cli.close_servo_step,
            max_travel=args_cli.close_servo_max_travel,
            timeout_steps=args_cli.close_servo_timeout,
            arm_zero=args_cli.close_servo_arm_zero,
            token_hold=args_cli.close_servo_token_hold,
        )
    distal_close_gate = None
    if distal_adapter is not None:
        assert distal_adapter_payload is not None
        distal_close_gate = StatefulCloseGate(
            n,
            dev,
            timeout_steps=args_cli.distal_adapter_timeout,
            entry_proximity=distal_adapter.proximity_threshold,
            exit_proximity=float(
                distal_adapter_payload.get(
                    "gate_exit_proximity", 0.5 * distal_adapter.proximity_threshold
                )
            ),
        )
    curriculum_initial = None
    if args_cli.curriculum_dataset:
        u._compute_intermediate_values()
        initial_signals = u._compute_grasp_signals()
        curriculum_initial = {
            "boundary": args_cli.curriculum_boundary,
            "slip_lin": _summary(initial_signals["slip_lin"]),
            "slip_ang": _summary(initial_signals["slip_ang"]),
            "robot_joint_velocity_norm": _summary(u.robot.data.joint_vel.norm(dim=-1)),
            "object_linear_velocity_norm": _summary(
                u.object.data.root_com_lin_vel_w.norm(dim=-1)
            ),
            "object_angular_velocity_norm": _summary(
                u.object.data.root_com_ang_vel_w.norm(dim=-1)
            ),
        }
    artifact = (
        json.loads(Path(args_cli.feasibility_json).read_text(encoding="utf-8"))
        if args_cli.feasibility_json
        else None
    )
    palm_in_object_pos = None
    palm_in_object_quat = None
    if args_cli.option_fsm:
        assert artifact is not None
        palm_in_object_pos, palm_in_object_quat = _calibrate_palm_in_object(u, artifact)
        obs = u._get_observations()

    if args_cli.mode == "fixed":
        assert artifact is not None
        _write_physical_state(u, _artifact_state(u, artifact))
        _clear_episode_state(u)
        u._compute_intermediate_values()
        obs = u._get_observations()

    stable_count = torch.zeros(n, dtype=torch.long, device=dev)
    success = torch.zeros(n, dtype=torch.bool, device=dev)
    success_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    max_clearance = torch.full((n,), -float("inf"), device=dev)
    max_unlatched_clearance = torch.full((n,), -float("inf"), device=dev)
    max_grasp_quality = torch.zeros(n, device=dev)
    force_peak = torch.zeros((n, len(u.ee_names)), device=dev)
    force_peak_env = torch.zeros(n, device=dev)
    force_above_30_steps = torch.zeros(n, dtype=torch.long, device=dev)
    force_above_60_steps = torch.zeros(n, dtype=torch.long, device=dev)
    force_impulse = torch.zeros(n, device=dev)
    unsafe_force_terminations = torch.zeros(n, dtype=torch.long, device=dev)
    latch_steps = torch.zeros(n, dtype=torch.long, device=dev)
    action_abs_sum = torch.zeros((n, 3), device=dev)
    close_servo_gate_steps = (
        torch.zeros(n, dtype=torch.long, device=dev) if close_controller is not None else None
    )
    close_servo_delta_abs_sum = (
        torch.zeros((n, 3), device=dev) if close_controller is not None else None
    )
    distal_adapter_gate_steps = (
        torch.zeros(n, dtype=torch.long, device=dev) if distal_adapter is not None else None
    )
    distal_adapter_delta_abs_sum = (
        torch.zeros(n, device=dev) if distal_adapter is not None else None
    )
    distal_adapter_delta_abs_max = (
        torch.zeros(n, device=dev) if distal_adapter is not None else None
    )
    option_switched = torch.zeros(n, dtype=torch.bool, device=dev)
    option_ready_steps = torch.zeros(n, dtype=torch.long, device=dev)
    option_switch_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    max_pregrasp_score = torch.zeros(n, device=dev)
    ever_touch = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_latch = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_latched_5cm = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_20cm = torch.zeros(n, dtype=torch.bool, device=dev)
    option_state = torch.full((n,), HOVER, dtype=torch.long, device=dev)
    option_guard_steps = torch.zeros(n, dtype=torch.long, device=dev)
    option_occupancy_steps = torch.zeros((n, len(OPTION_NAMES)), dtype=torch.long, device=dev)
    option_transition_counts = torch.zeros(
        (len(OPTION_NAMES), len(OPTION_NAMES)), dtype=torch.long, device=dev
    )
    option_transition_steps: dict[str, list[int]] = {
        f"{OPTION_NAMES[source]}->{OPTION_NAMES[target]}": []
        for source, target in (
            (HOVER, DESCEND),
            (DESCEND, CLOSE),
            (CLOSE, MICRO),
            (MICRO, LIFT_HOLD),
            (MICRO, CLOSE),
            (LIFT_HOLD, CLOSE),
        )
    }
    option_first_entry_step = torch.full(
        (n, len(OPTION_NAMES)), -1, dtype=torch.long, device=dev
    )
    option_first_entry_step[:, HOVER] = 0
    option_fallback_counts = torch.zeros((n, 2), dtype=torch.long, device=dev)
    option_reset_counts = torch.zeros(n, dtype=torch.long, device=dev)

    for step in range(args_cli.steps):
        policy_obs = obs["policy"]
        if policy_obs.shape[1] != 115:
            raise RuntimeError(f"environment observation changed: expected 115, got {policy_obs.shape[1]}")
        if args_cli.option_fsm:
            option_one_hot = torch.nn.functional.one_hot(
                option_state, num_classes=len(OPTION_NAMES)
            ).to(policy_obs.dtype)
            actor_obs = torch.cat((policy_obs, option_one_hot), dim=-1)
            option_occupancy_steps[torch.arange(n, device=dev), option_state] += 1
        else:
            actor_obs = policy_obs
        if option_actors is None:
            action = actor(actor_obs).clamp(-1.0, 1.0)
        else:
            action = torch.zeros((n, 21), device=dev)
            for option_index, expert in enumerate(option_actors):
                selected = option_state == option_index
                if bool(selected.any()):
                    action[selected] = expert(actor_obs[selected]).clamp(-1.0, 1.0)
        score = _pregrasp_score(u)
        max_pregrasp_score = torch.maximum(max_pregrasp_score, score)
        if reach_actor is not None:
            ready = (~option_switched) & (score >= args_cli.switch_score)
            option_ready_steps = torch.where(
                ready, option_ready_steps + 1, torch.zeros_like(option_ready_steps)
            )
            newly_switched = (~option_switched) & (
                option_ready_steps >= args_cli.switch_hold_steps
            )
            option_switch_step[newly_switched] = step
            option_switched |= newly_switched
            reach_action = torch.zeros_like(action)
            reach_action[:, :16] = reach_actor(obs["policy"][:, :87]).clamp(-1.0, 1.0)
            action = torch.where(option_switched.unsqueeze(-1), action, reach_action)
        if close_controller is not None:
            assert close_servo_gate_steps is not None and close_servo_delta_abs_sum is not None
            control_signals = u._compute_grasp_signals()
            force_by_distal = torch.zeros_like(control_signals["force_magnitude"])
            force_by_distal.index_copy_(
                1, u._force_to_distal_index, control_signals["force_magnitude"]
            )
            external_gate = option_switched if reach_actor is not None else None
            entry_gate = score >= args_cli.close_servo_pregrasp_score
            if args_cli.close_servo_require_touch:
                entry_gate &= control_signals["force_magnitude"].max(dim=-1).values >= cfg.contact_force_thr
            if external_gate is not None:
                entry_gate &= external_gate
            action, servo_gate, servo_delta = close_controller.apply(
                action,
                policy_obs,
                force_by_distal,
                external_gate=external_gate,
                entry_gate=entry_gate,
            )
            close_servo_gate_steps += servo_gate.long()
            close_servo_delta_abs_sum[:, 0] += servo_delta[:, :7].abs().mean(dim=-1)
            close_servo_delta_abs_sum[:, 1] += servo_delta[:, 7:16].abs().mean(dim=-1)
            close_servo_delta_abs_sum[:, 2] += servo_delta[:, 16:21].abs().mean(dim=-1)
        if distal_adapter is not None:
            assert distal_close_gate is not None
            assert distal_adapter_gate_steps is not None
            assert distal_adapter_delta_abs_sum is not None
            assert distal_adapter_delta_abs_max is not None
            allowed = option_switched if reach_actor is not None else None
            close_gate = distal_close_gate.update(
                policy_obs,
                score >= args_cli.distal_adapter_pregrasp_score,
                allowed=allowed,
            )
            actor_latent = actor.encode(policy_obs)
            action, adapter_gate, adapter_delta = distal_adapter.apply(
                action,
                policy_obs,
                actor_latent,
                scale=args_cli.distal_adapter_scale,
                external_gate=close_gate,
            )
            distal_adapter_gate_steps += adapter_gate.long()
            distal_adapter_delta_abs_sum += adapter_delta.abs().mean(dim=-1)
            distal_adapter_delta_abs_max = torch.maximum(
                distal_adapter_delta_abs_max, adapter_delta.abs().max(dim=-1).values
            )
        action_abs_sum[:, 0] += action[:, :7].abs().mean(dim=-1)
        action_abs_sum[:, 1] += action[:, 7:16].abs().mean(dim=-1)
        action_abs_sum[:, 2] += action[:, 16:].abs().mean(dim=-1)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        # Evaluation disables drop and built-in success termination, so every termination here is
        # the sustained-force safety cutoff.  Keep per-env event counts because auto-reset permits
        # a fresh attempt during a long diagnostic rollout.
        unsafe_force_terminations += terminated.long()
        if close_controller is not None:
            close_controller.reset(done)
        if distal_close_gate is not None:
            distal_close_gate.reset(done)
        signals = u._compute_grasp_signals()
        clearance = u._object_true_min_z() - u._table_surface_z
        force = signals["force_magnitude"]
        force_max = force.max(dim=-1).values
        if args_cli.option_fsm:
            assert palm_in_object_pos is not None and palm_in_object_quat is not None
            pose_state = _option_pose_state(u, palm_in_object_pos, palm_in_object_quat)
            active = ~done
            previous_option = option_state.clone()
            next_option = previous_option.clone()
            thumb_and_other_contact = signals["thumb_contact"] & (
                signals["other_contact_count"] >= 1
            )
            provisional_grasp = (
                (signals["close_quality"] >= 0.20)
                & thumb_and_other_contact
                & (signals["hold_quality"] >= 0.50)
                & (force_max <= cfg.grasp_bonus_max_force)
            )
            strict_transport_quality = torch.minimum(
                signals["quality"], signals["hold_quality"]
            )

            # MICRO may begin before the Schmitt latch confirms, so its loss guard uses the same
            # provisional physical grasp that admitted it.  LIFT_HOLD requires the strict latch.
            micro_fallback = active & (previous_option == MICRO) & ~provisional_grasp
            lift_fallback = active & (previous_option == LIFT_HOLD) & (
                (~u._is_grasped) | (strict_transport_quality < cfg.grasp_quality_low)
            )
            next_option[micro_fallback | lift_fallback] = CLOSE
            option_fallback_counts[:, 0] += micro_fallback.long()
            option_fallback_counts[:, 1] += lift_fallback.long()

            hover_ready = (
                (previous_option == HOVER)
                & (pose_state["hover_position_error"] < OPTION_HOVER_POSITION_TOLERANCE)
                & (pose_state["hover_rotation_error"] < OPTION_HOVER_ROTATION_TOLERANCE)
                & pose_state["low_speed"]
            )
            descend_ready = (
                (previous_option == DESCEND)
                & (pose_state["descend_position_error"] < OPTION_POSITION_TOLERANCE)
                & (pose_state["descend_rotation_error"] < OPTION_ROTATION_TOLERANCE)
                & pose_state["low_speed"]
                & (pose_state["palm_facing"] >= cfg.grasp_palm_facing_min)
                & (pose_state["alignment"] >= cfg.grasp_align_min)
            )
            close_ready = (
                (previous_option == CLOSE)
                & provisional_grasp
            )
            micro_ready = (
                (previous_option == MICRO)
                & ~micro_fallback
                & u._is_grasped
                & (strict_transport_quality >= 0.35)
                & (clearance >= 0.015)
                & (force_max <= cfg.grasp_bonus_max_force)
            )
            guarded = active & (hover_ready | descend_ready | close_ready | micro_ready)
            option_guard_steps = torch.where(
                guarded, option_guard_steps + 1, torch.zeros_like(option_guard_steps)
            )
            hover_transition = hover_ready & (
                option_guard_steps >= OPTION_APPROACH_HOLD_STEPS
            )
            descend_transition = descend_ready & (
                option_guard_steps >= OPTION_APPROACH_HOLD_STEPS
            )
            close_transition = close_ready & (
                option_guard_steps >= OPTION_GRASP_HOLD_STEPS
            )
            micro_transition = micro_ready & (
                option_guard_steps >= OPTION_GRASP_HOLD_STEPS
            )
            next_option[hover_transition] = DESCEND
            next_option[descend_transition] = CLOSE
            next_option[close_transition] = MICRO
            next_option[micro_transition] = LIFT_HOLD
            changed = active & (next_option != previous_option)
            option_guard_steps[changed] = 0

            for source, target in (
                (HOVER, DESCEND),
                (DESCEND, CLOSE),
                (CLOSE, MICRO),
                (MICRO, LIFT_HOLD),
                (MICRO, CLOSE),
                (LIFT_HOLD, CLOSE),
            ):
                transition = changed & (previous_option == source) & (next_option == target)
                count = int(transition.sum())
                if count:
                    option_transition_counts[source, target] += count
                    option_transition_steps[
                        f"{OPTION_NAMES[source]}->{OPTION_NAMES[target]}"
                    ].extend([step] * count)
                    first = transition & (option_first_entry_step[:, target] < 0)
                    option_first_entry_step[first, target] = step

            # Auto-reset follows sustained-force termination.  Restart the controller state but
            # keep accumulated evaluation/funnel statistics for the fresh attempt.
            option_reset_counts += done.long()
            next_option[done] = HOVER
            option_guard_steps[done] = 0
            option_state = next_option
        slow = (
            (u.object.data.root_com_lin_vel_w.norm(dim=-1) < cfg.success_max_obj_lin_speed)
            & (u.object.data.root_com_ang_vel_w.norm(dim=-1) < cfg.success_max_obj_ang_speed)
        )
        strict = (
            (clearance >= cfg.lift_success_height)
            & u._is_grasped
            & (signals["grasp_quality"] >= cfg.grasp_quality_high)
            & (force_max <= cfg.grasp_bonus_max_force)
            & slow
        )
        if args_cli.option_fsm:
            strict &= option_state == LIFT_HOLD
        stable_count = torch.where(strict, stable_count + 1, torch.zeros_like(stable_count))
        newly_successful = (~success) & (stable_count >= args_cli.stable_steps)
        success_step[newly_successful] = step
        success |= newly_successful
        max_clearance = torch.maximum(max_clearance, clearance)
        max_unlatched_clearance = torch.maximum(
            max_unlatched_clearance,
            torch.where(u._is_grasped, torch.full_like(clearance, -float("inf")), clearance),
        )
        max_grasp_quality = torch.maximum(max_grasp_quality, signals["grasp_quality"])
        force_peak = torch.maximum(force_peak, force)
        force_peak_env = torch.maximum(force_peak_env, force_max)
        force_above_30_steps += (force_max > 30.0).long()
        force_above_60_steps += (force_max > 60.0).long()
        force_impulse += force_max * (cfg.sim.dt * cfg.decimation)
        latch_steps += u._is_grasped.long()
        ever_touch |= force_max >= cfg.contact_force_thr
        ever_latch |= u._is_grasped
        ever_latched_5cm |= u._is_grasped & (clearance >= 0.05)
        ever_20cm |= clearance >= cfg.lift_success_height

    fling = (~success) & (max_unlatched_clearance >= 0.05)
    metrics = {
        "checkpoint": str(checkpoint_path.resolve()),
        "mode": args_cli.mode,
        "seed": args_cli.seed,
        "num_envs": n,
        "steps": args_cli.steps,
        "stable_steps": args_cli.stable_steps,
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "success_step": success_step.cpu().tolist(),
        "max_true_clearance": _summary(max_clearance),
        "max_unlatched_clearance": _summary(max_unlatched_clearance),
        "ever_unlatched_clearance_ge_5cm_count": int(
            (max_unlatched_clearance >= 0.05).sum()
        ),
        "false_lift_or_fling_count": int(fling.sum()),
        "max_grasp_quality": _summary(max_grasp_quality),
        "latch_occupancy": _summary(latch_steps.float() / args_cli.steps),
        "force_order": list(u.ee_names),
        "force_peak_per_finger": force_peak.max(dim=0).values.cpu().tolist(),
        "mean_action_abs_arm_token_residual": (action_abs_sum / args_cli.steps).mean(dim=0).cpu().tolist(),
        "funnel": {
            "pregrasp_score_ge_0.15": int((max_pregrasp_score >= 0.15).sum()),
            "pregrasp_score_ge_0.25": int((max_pregrasp_score >= 0.25).sum()),
            "touch": int(ever_touch.sum()),
            "grasp_latch": int(ever_latch.sum()),
            "latched_lift_ge_5cm": int(ever_latched_5cm.sum()),
            "true_clearance_ge_20cm": int(ever_20cm.sum()),
            "strict_success": int(success.sum()),
        },
        "force_safety": {
            "peak_all": _summary(force_peak_env),
            "peak_success": _masked_summary(force_peak_env, success),
            "peak_failure": _masked_summary(force_peak_env, ~success),
            "steps_above_30N": _summary(force_above_30_steps.float()),
            "steps_above_60N": _summary(force_above_60_steps.float()),
            "force_time_integral_Ns": _summary(force_impulse),
            "unsafe_termination_events": int(unsafe_force_terminations.sum()),
            "unsafe_terminated_envs": int((unsafe_force_terminations > 0).sum()),
        },
    }
    if curriculum_initial is not None:
        metrics["curriculum_initial"] = curriculum_initial
    if close_controller is not None:
        assert close_servo_gate_steps is not None and close_servo_delta_abs_sum is not None
        metrics["tactile_close_servo"] = {
            "enabled": True,
            "arm_zero": close_controller.arm_zero,
            "token_hold": close_controller.token_hold,
            "entry_proximity": close_controller.entry_proximity,
            "exit_proximity": close_controller.exit_proximity,
            "entry_pregrasp_score": args_cli.close_servo_pregrasp_score,
            "require_touch": args_cli.close_servo_require_touch,
            "low_force_N": close_controller.low_force,
            "high_force_N": close_controller.high_force,
            "normalized_step": close_controller.step,
            "normalized_max_travel": close_controller.max_travel,
            "timeout_steps": close_controller.timeout_steps,
            "entry_count": int(close_controller.entry_count.sum()),
            "entered_envs": int((close_controller.entry_count > 0).sum()),
            "gate_occupancy": _summary(close_servo_gate_steps.float() / args_cli.steps),
            "mean_abs_delta_arm_token_distal": (
                close_servo_delta_abs_sum / args_cli.steps
            ).mean(dim=0).cpu().tolist(),
        }
    if distal_adapter is not None:
        assert distal_adapter_payload is not None
        assert distal_close_gate is not None
        assert distal_adapter_gate_steps is not None
        assert distal_adapter_delta_abs_sum is not None
        assert distal_adapter_delta_abs_max is not None
        metrics["distal_adapter"] = {
            "path": str(Path(args_cli.distal_adapter).resolve()),
            "scale": args_cli.distal_adapter_scale,
            "entry_pregrasp_score": args_cli.distal_adapter_pregrasp_score,
            "timeout_steps": args_cli.distal_adapter_timeout,
            "entry_proximity": distal_close_gate.entry_proximity,
            "exit_proximity": distal_close_gate.exit_proximity,
            "base_sha256": distal_adapter_payload["base_sha256"],
            "delta_limit": distal_adapter.delta_limit,
            "train_meta": distal_adapter_payload.get("train_meta"),
            "gate_occupancy": _summary(distal_adapter_gate_steps.float() / args_cli.steps),
            "mean_abs_delta": float((distal_adapter_delta_abs_sum / args_cli.steps).mean()),
            "max_abs_delta": _summary(distal_adapter_delta_abs_max),
        }
    if reach_actor is not None:
        metrics["option"] = {
            "reach_checkpoint": str(Path(args_cli.reach_checkpoint).resolve()),
            "switch_score": args_cli.switch_score,
            "switch_hold_steps": args_cli.switch_hold_steps,
            "switch_count": int(option_switched.sum()),
            "switch_rate": float(option_switched.float().mean()),
            "switch_step": option_switch_step.cpu().tolist(),
            "max_pregrasp_score": _summary(max_pregrasp_score),
        }
    if args_cli.option_fsm:
        transition_metrics = {}
        for name, steps in option_transition_steps.items():
            step_tensor = torch.tensor(steps, dtype=torch.float, device=dev)
            transition_metrics[name] = {
                "count": len(steps),
                "steps": steps,
                "step_summary": None if not steps else _summary(step_tensor),
            }
        total_occupancy = option_occupancy_steps.sum(dim=0)
        metrics["option_fsm"] = {
            "enabled": True,
            "expert_checkpoints": (
                [str(Path(path).resolve()) for path in args_cli.option_checkpoints]
                if args_cli.option_checkpoints
                else None
            ),
            "order": list(OPTION_NAMES),
            "calibration": "artifact pregrasp palm-in-object FK; reset state restored before rollout",
            "thresholds": {
                "hover_height_m": OPTION_HOVER_HEIGHT,
                "hover_position_error_m": OPTION_HOVER_POSITION_TOLERANCE,
                "hover_rotation_error_rad": OPTION_HOVER_ROTATION_TOLERANCE,
                "position_error_m": OPTION_POSITION_TOLERANCE,
                "rotation_error_rad": OPTION_ROTATION_TOLERANCE,
                "max_palm_linear_speed_mps": OPTION_MAX_PALM_LINEAR_SPEED,
                "max_palm_angular_speed_radps": OPTION_MAX_PALM_ANGULAR_SPEED,
                "approach_debounce_steps": OPTION_APPROACH_HOLD_STEPS,
                "grasp_debounce_steps": OPTION_GRASP_HOLD_STEPS,
                "close_quality": 0.20,
                "hold_quality": 0.50,
                "lift_transport_quality": 0.35,
                "micro_clearance_m": 0.015,
                "max_force_N": cfg.grasp_bonus_max_force,
            },
            "occupancy_steps_total": dict(zip(OPTION_NAMES, total_occupancy.cpu().tolist(), strict=True)),
            "occupancy_fraction": dict(
                zip(
                    OPTION_NAMES,
                    (total_occupancy.float() / max(int(total_occupancy.sum()), 1)).cpu().tolist(),
                    strict=True,
                )
            ),
            "occupancy_per_env": {
                name: _summary(option_occupancy_steps[:, index].float() / args_cli.steps)
                for index, name in enumerate(OPTION_NAMES)
            },
            "transition_count_matrix": option_transition_counts.cpu().tolist(),
            "transitions": transition_metrics,
            "first_entry_step": {
                name: option_first_entry_step[:, index].cpu().tolist()
                for index, name in enumerate(OPTION_NAMES)
            },
            "fallbacks": {
                "MICRO->CLOSE": int(option_fallback_counts[:, 0].sum()),
                "LIFT_HOLD->CLOSE": int(option_fallback_counts[:, 1].sum()),
                "per_env_micro_lift": option_fallback_counts.cpu().tolist(),
            },
            "safety_reset_count": int(option_reset_counts.sum()),
            "final_option_count": {
                name: int((option_state == index).sum())
                for index, name in enumerate(OPTION_NAMES)
            },
        }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"{args_cli.mode} closed-loop: success={int(success.sum())}/{n}; "
        f"fling={int(fling.sum())}/{n}; clearance={metrics['max_true_clearance']}; "
        f"latch={metrics['latch_occupancy']}",
        flush=True,
    )
    if reach_actor is not None:
        print(
            f"option switch={int(option_switched.sum())}/{n}; "
            f"max_pregrasp_score={metrics['option']['max_pregrasp_score']}",
            flush=True,
        )
    if args_cli.option_fsm:
        occupancy = metrics["option_fsm"]["occupancy_steps_total"]
        transitions = {
            name: payload["count"]
            for name, payload in metrics["option_fsm"]["transitions"].items()
        }
        print(f"option occupancy={occupancy}; transitions={transitions}", flush=True)
    print(f"wrote {output.resolve()}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
