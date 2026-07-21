#!/usr/bin/env python3
"""Search hybrid14 hand actions for a strict thumb-plus-three power close.

The input is a recoverable fixed-pregrasp artifact emitted by
``hand_space_feasibility.py``.  Unlike the historical benchmark, this search
optimizes the independent power-close signals: thumb plus three legal opposed
non-thumb contacts, rigid hold, a 30 N force ceiling, and a 15-frame stable
window after the four-frame power latch.  Arm commands are zero by default.
An optional bounded seven-joint arm-target search is a reachability oracle only;
its winner must reproduce through the public incremental controller before use
as a teacher.  Every CEM iteration restores the same robot/object state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import types
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="artifact supplying the fixed pregrasp")
parser.add_argument(
    "--initial_action_input",
    default=None,
    help="optional artifact supplying the initial results.hybrid14 latent",
)
parser.add_argument("--population", type=int, default=512)
parser.add_argument(
    "--replicates_per_candidate",
    type=int,
    default=1,
    help="physical clones sharing each CEM parameter vector; a pass requires every clone",
)
parser.add_argument("--iterations", type=int, default=12)
parser.add_argument(
    "--arm_delta_limit_rad",
    type=float,
    default=0.0,
    help=(
        "when positive, jointly search seven bounded arm-joint target offsets before "
        "closing the hand; zero preserves the hand-only formal-action search"
    ),
)
parser.add_argument(
    "--align_steps",
    type=int,
    default=0,
    help="simulation frames used to blend to a searched arm target before hand closure",
)
parser.add_argument("--arm_delta_penalty", type=float, default=0.20)
parser.add_argument(
    "--public_controller",
    action="store_true",
    help=(
        "evaluate CEM candidates through the formal incremental-arm and hybrid14 hand "
        "controller, including EMA, slew and tactile shields"
    ),
)
parser.add_argument("--close_steps", type=int, default=48)
parser.add_argument("--eval_steps", type=int, default=24)
parser.add_argument("--elite_frac", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="/tmp/pick_tool_power_close_feasibility.json")
parser.add_argument("--require_pass", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.population < 32:
    parser.error("--population must be at least 32")
if args_cli.replicates_per_candidate < 1:
    parser.error("--replicates_per_candidate must be positive")
if args_cli.population % args_cli.replicates_per_candidate != 0:
    parser.error("--population must be divisible by --replicates_per_candidate")
if args_cli.iterations < 1:
    parser.error("--iterations must be positive")
if args_cli.close_steps < 1 or args_cli.eval_steps < 15:
    parser.error("--close_steps must be positive and --eval_steps must be at least 15")
if args_cli.arm_delta_limit_rad < 0.0 or args_cli.arm_delta_limit_rad > 0.35:
    parser.error("--arm_delta_limit_rad must lie in [0, 0.35]")
if args_cli.align_steps < 0:
    parser.error("--align_steps must be non-negative")
if args_cli.arm_delta_limit_rad > 0.0 and args_cli.align_steps < 1:
    parser.error("arm-micro search requires --align_steps >= 1")
if args_cli.arm_delta_limit_rad == 0.0 and args_cli.align_steps != 0:
    parser.error("--align_steps requires a positive --arm_delta_limit_rad")
if args_cli.arm_delta_penalty < 0.0:
    parser.error("--arm_delta_penalty must be non-negative")
if args_cli.public_controller and args_cli.arm_delta_limit_rad <= 0.0:
    parser.error("--public_controller currently requires a positive arm micro-adjustment limit")
if not 0.02 <= args_cli.elite_frac <= 0.5:
    parser.error("--elite_frac must lie in [0.02, 0.5]")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401
from xhand_inhand.tasks.direct.pick_tool_token.hybrid_action import (
    apply_asymmetric_joint_residual,
    invert_asymmetric_joint_residual,
)
from power_close_search_contract import (
    POWER_CLEARANCE_LIMIT,
    POWER_FORCE_LIMIT,
    POWER_GRASP_QUALITY_MIN,
    POWER_HOLD_QUALITY_MIN,
    POWER_LATCH_CONFIRM_FRAMES,
    POWER_LATCH_RELEASE_FRAMES,
    POWER_REQUIRED_OTHER_CONTACTS,
    POWER_ROTATION_DRIFT_LIMIT,
    POWER_STABLE_FRAMES,
    POWER_XY_DRIFT_LIMIT,
    aggregate_replicated_candidates,
    power_close_candidate_score,
    power_close_stable_frame,
    strict_power_close_pass,
    update_power_grasp_latch,
    update_stable_streak,
)


ARM_DIM = 7
HAND_DIM = 14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_list(mapping: dict, name: str, length: int) -> list:
    value = mapping.get(name)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    return value


def _quat_angle(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Shortest sign-invariant quaternion distance in radians."""

    dot = (reference * current).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


@torch.inference_mode()
def main() -> None:
    input_path = Path(args_cli.input).resolve()
    artifact = json.loads(input_path.read_text(encoding="utf-8"))
    pregrasp = artifact.get("pregrasp")
    if not isinstance(pregrasp, dict):
        raise ValueError("input artifact has no pregrasp mapping")
    joint_pos = _require_list(pregrasp, "joint_pos", 19)
    object_local_pos = _require_list(pregrasp, "object_local_pos", 3)
    object_quat = _require_list(pregrasp, "object_quat", 4)

    initial_path = (
        Path(args_cli.initial_action_input).resolve()
        if args_cli.initial_action_input is not None
        else input_path
    )
    initial_artifact = json.loads(initial_path.read_text(encoding="utf-8"))
    try:
        initial_result = initial_artifact["results"]["hybrid14"]
        initial_latent = initial_result["latent"]
    except (KeyError, TypeError) as exc:
        raise ValueError("initial action artifact has no results.hybrid14.latent") from exc
    if not isinstance(initial_latent, list) or len(initial_latent) != HAND_DIM:
        raise ValueError("initial hybrid14 latent must contain 14 values")

    torch.manual_seed(args_cli.seed)
    num_envs = args_cli.population
    replicates = args_cli.replicates_per_candidate
    num_candidates = num_envs // replicates
    cfg = parse_env_cfg(
        "Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=num_envs
    )
    cfg.seed = args_cli.seed
    cfg.episode_length_s = 120.0
    cfg.terminate_on_drop = False
    cfg.success_hold_steps = 100000
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    env.reset()
    dev = u.device
    all_ids = u.robot._ALL_INDICES

    if cfg.action_space != ARM_DIM + HAND_DIM:
        raise RuntimeError(f"expected formal 21-D action space, got {cfg.action_space}")
    if u._n_arm != ARM_DIM or u._n_tokens + u._n_distal_residuals != HAND_DIM:
        raise RuntimeError("runtime hybrid action partition is not arm7|hand14")

    # Search diagnostics own failure accounting.  Preserve the native done truth but prevent
    # DirectRLEnv from auto-resetting a rejected candidate before its trajectory score is read.
    native_get_dones = u._get_dones
    native_terminated_seen = torch.zeros(
        num_envs, dtype=torch.bool, device=dev
    )
    native_timeout_seen = torch.zeros_like(native_terminated_seen)

    def no_auto_reset(self):
        terminated, time_out = native_get_dones()
        native_terminated_seen.logical_or_(terminated)
        native_timeout_seen.logical_or_(time_out)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return zeros, zeros

    u._get_dones = types.MethodType(no_auto_reset, u)

    snapshot_joint = torch.tensor(joint_pos, dtype=torch.float32, device=dev)
    snapshot_object_local = torch.tensor(
        object_local_pos, dtype=torch.float32, device=dev
    )
    snapshot_object_quat = torch.tensor(object_quat, dtype=torch.float32, device=dev)
    repeated_joint = snapshot_joint.unsqueeze(0).repeat(num_envs, 1)
    start_com_xy = torch.zeros((num_envs, 2), dtype=torch.float32, device=dev)
    snapshot_quat_batch = snapshot_object_quat.unsqueeze(0).repeat(num_envs, 1)
    search_arm_micro = args_cli.arm_delta_limit_rad > 0.0
    search_joint_target = repeated_joint.clone()

    if search_arm_micro and not args_cli.public_controller:
        # CEM searches a bounded arm *target offset*, not a constant incremental
        # command that would integrate without bound.  The rollout below blends
        # to this target, then closes the hand.  A later replay/teacher converts
        # the selected offset into the public incremental action sequence.
        def apply_search_target(self, actions: torch.Tensor) -> None:
            self.actions = torch.zeros_like(actions)
            self.dof_targets.copy_(search_joint_target)
            self.dof_targets[:] = torch.clamp(
                self.dof_targets, self.dof_lower, self.dof_upper
            )

        u._pre_physics_step = types.MethodType(apply_search_target, u)

    @torch.inference_mode()
    def restore_snapshot() -> None:
        u._reset_idx(all_ids)
        u.robot.write_joint_state_to_sim(
            repeated_joint, torch.zeros_like(repeated_joint), env_ids=all_ids
        )
        u.robot.set_joint_position_target(repeated_joint, env_ids=all_ids)
        u.dof_targets.copy_(repeated_joint)
        search_joint_target.copy_(repeated_joint)
        pose = torch.zeros((num_envs, 7), dtype=torch.float32, device=dev)
        pose[:, :3] = snapshot_object_local + u.scene.env_origins
        pose[:, 3:7] = snapshot_object_quat
        u.object.write_root_pose_to_sim(pose, env_ids=all_ids)
        u.object.write_root_velocity_to_sim(
            torch.zeros((num_envs, 6), device=dev), env_ids=all_ids
        )
        u.episode_length_buf.zero_()
        u.actions.zero_()
        u.prev_actions.zero_()
        u._compute_intermediate_values()
        start_com_xy.copy_(u._object_com_position_w()[:, :2])
        native_terminated_seen.zero_()
        native_timeout_seen.zero_()

    hand_lower = u.dof_lower[:, u._hand_ids_t]
    hand_upper = u.dof_upper[:, u._hand_ids_t]

    def decode_hybrid14(latent: torch.Tensor) -> torch.Tensor:
        token_target = u.retarget.retarget_from_unit_action(latent[:, : u._n_tokens])[
            :, u._retarget2isaac
        ]
        target, _ = apply_asymmetric_joint_residual(
            token_target,
            hand_lower[: latent.shape[0]],
            hand_upper[: latent.shape[0]],
            latent[:, u._n_tokens :],
            u._distal_hand_ids,
        )
        return target

    initial_hold_action: torch.Tensor | None = None
    initial_hold_decode_error = 0.0
    if args_cli.public_controller:
        pregrasp_token = pregrasp.get("token")
        if not isinstance(pregrasp_token, list) or len(pregrasp_token) != u._n_tokens:
            raise ValueError("public-controller alignment requires pregrasp.token[9]")
        seed_token = torch.tensor(
            pregrasp_token, dtype=torch.float32, device=dev
        )
        fit_population = 4096
        fit_elites = 256
        fit_generator = torch.Generator(device=dev).manual_seed(99173)
        fit_mean = seed_token.clone()
        fit_std = torch.full_like(fit_mean, 0.30)
        fit_lower = hand_lower[:1].expand(fit_population, -1)
        fit_upper = hand_upper[:1].expand(fit_population, -1)
        fit_target = repeated_joint[:1, u._hand_ids_t].expand(
            fit_population, -1
        )
        best_fit_action = None
        best_fit_error = float("inf")
        for _ in range(15):
            candidate_token = (
                fit_mean
                + fit_std
                * torch.randn(
                    (fit_population, u._n_tokens),
                    device=dev,
                    generator=fit_generator,
                )
            ).clamp(-1.0, 1.0)
            candidate_token[0] = fit_mean.clamp(-1.0, 1.0)
            candidate_token[1] = seed_token
            candidate_base = u.retarget.retarget_from_unit_action(candidate_token)[
                :, u._retarget2isaac
            ]
            candidate_residual = invert_asymmetric_joint_residual(
                fit_target,
                candidate_base,
                fit_lower,
                fit_upper,
                u._distal_hand_ids,
            )
            candidate_decoded, _ = apply_asymmetric_joint_residual(
                candidate_base,
                fit_lower,
                fit_upper,
                candidate_residual,
                u._distal_hand_ids,
            )
            candidate_error = candidate_decoded - fit_target
            fit_loss = candidate_error.square().mean(dim=-1) + 0.25 * (
                candidate_error.abs().max(dim=-1).values.square()
            )
            best_index = int(torch.argmin(fit_loss).item())
            best_error = float(candidate_error[best_index].abs().max().item())
            if best_error < best_fit_error:
                best_fit_error = best_error
                best_fit_action = torch.cat(
                    (candidate_token[best_index], candidate_residual[best_index]),
                    dim=0,
                ).clone()
            elite_indices = torch.topk(
                fit_loss, k=fit_elites, largest=False
            ).indices
            elite = candidate_token[elite_indices]
            fit_mean = elite.mean(dim=0)
            fit_std = elite.std(dim=0, unbiased=False).clamp(0.015, 0.50)
        assert best_fit_action is not None
        initial_hold_action = best_fit_action.unsqueeze(0).repeat(num_envs, 1)
        initial_hold_decode_error = best_fit_error

    hand_mean = torch.tensor(initial_latent, dtype=torch.float32, device=dev)
    hand_std = torch.cat(
        (
            torch.full((u._n_tokens,), 0.35, device=dev),
            torch.full((u._n_distal_residuals,), 0.55, device=dev),
        )
    )
    if search_arm_micro:
        initial_arm_delta = initial_result.get("arm_delta_target_rad")
        if initial_arm_delta is None:
            arm_mean = torch.zeros(ARM_DIM, device=dev)
        elif not isinstance(initial_arm_delta, list) or len(initial_arm_delta) != ARM_DIM:
            raise ValueError("initial arm_delta_target_rad must contain 7 values")
        else:
            arm_mean = (
                torch.tensor(initial_arm_delta, dtype=torch.float32, device=dev)
                / args_cli.arm_delta_limit_rad
            ).clamp(-1.0, 1.0)
        mean = torch.cat((arm_mean, hand_mean))
        std = torch.cat((torch.full((ARM_DIM,), 0.40, device=dev), hand_std))
    else:
        mean = hand_mean
        std = hand_std
    original_seed = mean.clone()
    parameter_dim = int(mean.numel())
    elite_count = min(
        num_candidates,
        max(8, int(round(num_candidates * args_cli.elite_frac))),
    )
    best_any: dict | None = None
    best_pass: dict | None = None

    print(
        f"POWER CEM envs={num_envs} candidates={num_candidates} "
        f"replicates={replicates} iterations={args_cli.iterations} "
        f"align={args_cli.align_steps} close={args_cli.close_steps} "
        f"eval={args_cli.eval_steps} elites={elite_count} "
        f"arm_delta_limit={args_cli.arm_delta_limit_rad:.3f}rad "
        f"controller={'public' if args_cli.public_controller else 'oracle'} "
        f"initial_hold_error={initial_hold_decode_error:.4f}rad",
        flush=True,
    )

    for iteration in range(args_cli.iterations):
        candidate_parameter = (
            mean + std * torch.randn((num_candidates, parameter_dim), device=dev)
        ).clamp(
            -1.0, 1.0
        )
        candidate_parameter[0] = mean.clamp(-1.0, 1.0)
        if num_candidates > 1:
            candidate_parameter[1] = original_seed
        parameter = candidate_parameter.repeat_interleave(replicates, dim=0)
        if search_arm_micro:
            arm_parameter = parameter[:, :ARM_DIM]
            hand_latent = parameter[:, ARM_DIM:]
        else:
            arm_parameter = torch.zeros((num_envs, ARM_DIM), device=dev)
            hand_latent = parameter
        action = torch.zeros((num_envs, ARM_DIM + HAND_DIM), device=dev)
        action[:, ARM_DIM:] = hand_latent

        candidate_joint_target = repeated_joint.clone()
        candidate_hand_target = decode_hybrid14(hand_latent)
        candidate_joint_target[:, u._hand_ids_t] = candidate_hand_target
        if search_arm_micro:
            raw_arm_target = repeated_joint[:, u._arm_ids_t] + (
                args_cli.arm_delta_limit_rad * arm_parameter
            )
            arm_lower = u.dof_lower[:, u._arm_ids_t]
            arm_upper = u.dof_upper[:, u._arm_ids_t]
            candidate_arm_target = torch.maximum(
                torch.minimum(raw_arm_target, arm_upper), arm_lower
            )
            candidate_joint_target[:, u._arm_ids_t] = candidate_arm_target
            arm_delta_actual = (
                candidate_arm_target - repeated_joint[:, u._arm_ids_t]
            )
        else:
            arm_delta_actual = torch.zeros((num_envs, ARM_DIM), device=dev)

        restore_snapshot()
        q_close_sum = torch.zeros(num_envs, device=dev)
        q_wrap_sum = torch.zeros(num_envs, device=dev)
        q_grasp_sum = torch.zeros(num_envs, device=dev)
        hold_sum = torch.zeros(num_envs, device=dev)
        thumb_sum = torch.zeros(num_envs, device=dev)
        legal_other_sum = torch.zeros(num_envs, device=dev)
        third_other_sum = torch.zeros(num_envs, device=dev)
        fourth_other_sum = torch.zeros(num_envs, device=dev)
        stable_sum = torch.zeros(num_envs, device=dev)
        stable_streak = torch.zeros(num_envs, dtype=torch.long, device=dev)
        stable_streak_peak = torch.zeros_like(stable_streak)
        first_success_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=dev
        )
        power_is_grasped = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        power_latch_confirm = torch.zeros(num_envs, dtype=torch.long, device=dev)
        power_latch_release = torch.zeros(num_envs, dtype=torch.long, device=dev)
        power_latch_confirm_peak = torch.zeros_like(power_latch_confirm)
        force_peak = torch.zeros((num_envs, len(u.ee_names)), device=dev)
        clearance_peak = torch.full((num_envs,), -float("inf"), device=dev)
        clearance_min = torch.full((num_envs,), float("inf"), device=dev)
        xy_drift_peak = torch.zeros(num_envs, device=dev)
        rotation_drift_peak = torch.zeros(num_envs, device=dev)
        arm_table_clearance_min = torch.full(
            (num_envs,), float("inf"), device=dev
        )
        arm_tracking_error_peak = torch.zeros(num_envs, device=dev)
        eval_count = 0

        eval_start = args_cli.align_steps + args_cli.close_steps
        total_steps = eval_start + args_cli.eval_steps
        for step in range(total_steps):
            desired_arm_target = candidate_joint_target[:, u._arm_ids_t]
            if args_cli.public_controller:
                assert initial_hold_action is not None
                step_action = torch.zeros_like(action)
                if step < args_cli.align_steps:
                    x = float(step + 1) / float(args_cli.align_steps)
                    arm_blend = x * x * (3.0 - 2.0 * x)
                    desired_arm_target = repeated_joint[:, u._arm_ids_t] + (
                        arm_blend * arm_delta_actual
                    )
                    step_action[:, ARM_DIM:] = initial_hold_action
                else:
                    step_action[:, ARM_DIM:] = hand_latent
                current_arm_target = u.dof_targets[:, u._arm_ids_t]
                step_action[:, :ARM_DIM] = torch.clamp(
                    (desired_arm_target - current_arm_target)
                    / (cfg.action_scale * cfg.act_moving_average),
                    -1.0,
                    1.0,
                )
                env.step(step_action)
                arm_tracking_error_peak = torch.maximum(
                    arm_tracking_error_peak,
                    (u.dof_targets[:, u._arm_ids_t] - desired_arm_target)
                    .abs()
                    .max(dim=-1)
                    .values,
                )
            elif search_arm_micro:
                search_joint_target.copy_(repeated_joint)
                if step < args_cli.align_steps:
                    x = float(step + 1) / float(args_cli.align_steps)
                    arm_blend = x * x * (3.0 - 2.0 * x)
                    hand_blend = 0.0
                elif step < eval_start:
                    arm_blend = 1.0
                    x = float(step - args_cli.align_steps + 1) / float(
                        args_cli.close_steps
                    )
                    hand_blend = x * x * (3.0 - 2.0 * x)
                else:
                    arm_blend = 1.0
                    hand_blend = 1.0
                search_joint_target[:, u._arm_ids_t] = repeated_joint[
                    :, u._arm_ids_t
                ] + arm_blend * arm_delta_actual
                search_joint_target[:, u._hand_ids_t] = repeated_joint[
                    :, u._hand_ids_t
                ] + hand_blend * (
                    candidate_hand_target - repeated_joint[:, u._hand_ids_t]
                )
                env.step(action)
            else:
                env.step(action)
            signals = u._compute_grasp_signals()
            max_force = signals["force_magnitude"].max(dim=-1).values
            force_peak = torch.maximum(force_peak, signals["force_magnitude"])
            clearance = u._object_true_min_z() - u._table_surface_z
            clearance_peak = torch.maximum(clearance_peak, clearance)
            clearance_min = torch.minimum(clearance_min, clearance)
            object_com_xy = u._object_com_position_w()[:, :2]
            xy_drift = (object_com_xy - start_com_xy).norm(dim=-1)
            xy_drift_peak = torch.maximum(xy_drift_peak, xy_drift)
            rotation_drift = _quat_angle(
                snapshot_quat_batch, u.object.data.root_quat_w
            )
            rotation_drift_peak = torch.maximum(
                rotation_drift_peak, rotation_drift
            )
            arm_clearance = (
                u.robot.data.body_pos_w[:, u._arm_body_ids, 2]
                - u.scene.env_origins[:, 2].unsqueeze(-1)
                - u._table_surface_z
            ).min(dim=-1).values
            arm_table_clearance_min = torch.minimum(
                arm_table_clearance_min, arm_clearance
            )

            stable = power_close_stable_frame(
                power_is_grasped,
                signals["power_thumb_contact"],
                signals["power_legal_other_contact_count"],
                signals["power_grasp_quality"],
                signals["hold_quality"],
                max_force,
            )
            stable_streak, stable_streak_peak = update_stable_streak(
                stable_streak, stable_streak_peak, stable
            )
            newly_successful = (stable_streak >= POWER_STABLE_FRAMES) & (
                first_success_step < 0
            )
            first_success_step = torch.where(
                newly_successful,
                torch.full_like(first_success_step, step + 1),
                first_success_step,
            )
            # Install the latch after evaluating this action's option state.  This mirrors
            # DirectRLEnv's dones-before-reward ordering and preserves action 19 as the earliest
            # possible strict completion.
            (
                power_is_grasped,
                power_latch_confirm,
                power_latch_release,
            ) = update_power_grasp_latch(
                signals["power_grasp_quality"],
                power_is_grasped,
                power_latch_confirm,
                power_latch_release,
            )
            power_latch_confirm_peak = torch.maximum(
                power_latch_confirm_peak, power_latch_confirm
            )

            if step >= eval_start:
                q_close_sum += signals["power_close_quality"]
                q_wrap_sum += signals["power_wrap_quality"]
                q_grasp_sum += signals["power_grasp_quality"]
                hold_sum += signals["hold_quality"]
                thumb_sum += signals["power_thumb_contact"].float()
                legal_other_sum += signals["power_legal_other_contact_count"].float()
                legal_other = signals["power_legal_other_contact_count"]
                third_other_sum += (legal_other >= 3).float()
                fourth_other_sum += (legal_other >= 4).float()
                stable_sum += stable.float()
                eval_count += 1

        q_close_mean = q_close_sum / eval_count
        q_wrap_mean = q_wrap_sum / eval_count
        q_grasp_mean = q_grasp_sum / eval_count
        hold_mean = hold_sum / eval_count
        thumb_frac = thumb_sum / eval_count
        legal_other_mean = legal_other_sum / eval_count
        third_other_frac = third_other_sum / eval_count
        fourth_other_frac = fourth_other_sum / eval_count
        stable_frac = stable_sum / eval_count
        force_peak_max = force_peak.max(dim=-1).values
        unexpected_done = native_terminated_seen | native_timeout_seen
        strict_env_pass = strict_power_close_pass(
            stable_streak,
            force_peak_max,
            xy_drift_peak,
            rotation_drift_peak,
            clearance_peak,
            unexpected_done,
        )
        env_score = power_close_candidate_score(
            power_close_mean=q_close_mean,
            thumb_fraction=thumb_frac,
            third_other_fraction=third_other_frac,
            fourth_other_fraction=fourth_other_frac,
            power_wrap_mean=q_wrap_mean,
            power_grasp_mean=q_grasp_mean,
            hold_mean=hold_mean,
            stable_fraction=stable_frac,
            stable_streak_at_end=stable_streak,
            stable_streak_peak=stable_streak_peak,
            force_peak=force_peak_max,
            xy_drift_peak=xy_drift_peak,
            rotation_drift_peak=rotation_drift_peak,
            clearance_peak=clearance_peak,
            # A single lucky clone must not receive the lexicographic pass bonus.  The bonus is
            # installed only after all physical replicas of a candidate pass independently.
            strict_pass=torch.zeros_like(strict_env_pass),
            unexpected_done=unexpected_done,
        )
        if search_arm_micro:
            normalized_actual_arm_delta = (
                arm_delta_actual / args_cli.arm_delta_limit_rad
            )
            env_score -= (
                args_cli.arm_delta_penalty
                * normalized_actual_arm_delta.square().mean(dim=-1)
            )
        finite = torch.stack(
            (
                q_close_mean,
                q_wrap_mean,
                q_grasp_mean,
                hold_mean,
                force_peak_max,
                clearance_peak,
                clearance_min,
                xy_drift_peak,
                rotation_drift_peak,
                arm_table_clearance_min,
                arm_tracking_error_peak,
                env_score,
            ),
            dim=-1,
        ).isfinite().all(dim=-1)
        if not bool(finite.any()):
            raise RuntimeError(f"all CEM candidates became non-finite at iteration {iteration + 1}")
        strict_env_pass &= finite
        env_score = torch.where(
            finite, env_score, torch.full_like(env_score, -1.0e9)
        )
        candidate_score, strict_group_pass, finite_group = (
            aggregate_replicated_candidates(
                env_score, strict_env_pass, finite, replicates
            )
        )

        top = torch.topk(candidate_score, k=elite_count, largest=True)
        elite = candidate_parameter[top.indices]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0, unbiased=False).clamp(0.035, 0.75)

        def candidate_record(index: int) -> dict:
            start = index * replicates
            stop = start + replicates
            sl = slice(start, stop)
            first_steps = first_success_step[sl]
            robust_first_step = (
                int(first_steps.max().item())
                if bool((first_steps >= 0).all())
                else -1
            )
            robust_force_peak = force_peak[sl].amax(dim=0)
            return {
                "iteration": iteration + 1,
                "score": float(candidate_score[index].item()),
                "replicates": replicates,
                "strict_replicates": int(strict_env_pass[sl].sum().item()),
                "replicate_strict_pass": strict_env_pass[sl].cpu().tolist(),
                "latent": hand_latent[start].detach().cpu().tolist(),
                "target": candidate_hand_target[start]
                .detach()
                .cpu()
                .tolist(),
                "search_parameter": candidate_parameter[index].detach().cpu().tolist(),
                "arm_parameter": arm_parameter[start].detach().cpu().tolist(),
                "arm_delta_target_rad": arm_delta_actual[start]
                .detach()
                .cpu()
                .tolist(),
                "arm_target": candidate_joint_target[start, u._arm_ids_t]
                .detach()
                .cpu()
                .tolist(),
                "arm_delta_abs_max_rad": float(
                    arm_delta_actual[start].abs().max().item()
                ),
                "strict_power_close_pass": bool(strict_group_pass[index].item()),
                "first_success_step": robust_first_step,
                "stable_streak_at_end": int(stable_streak[sl].min().item()),
                "stable_streak_at_end_per_replicate": stable_streak[sl].cpu().tolist(),
                "stable_streak_peak": int(stable_streak_peak[sl].min().item()),
                "stable_streak_peak_per_replicate": stable_streak_peak[sl]
                .cpu()
                .tolist(),
                "power_latch_confirm_peak": int(
                    power_latch_confirm_peak[sl].min().item()
                ),
                "power_latched_at_end": bool(power_is_grasped[sl].all().item()),
                "stable_fraction": float(stable_frac[sl].mean().item()),
                "q_close": float(q_close_mean[sl].mean().item()),
                "q_wrap": float(q_wrap_mean[sl].mean().item()),
                "q_grasp": float(q_grasp_mean[sl].mean().item()),
                "q_grasp_per_replicate": q_grasp_mean[sl].cpu().tolist(),
                "hold_quality": float(hold_mean[sl].mean().item()),
                "thumb_fraction": float(thumb_frac[sl].mean().item()),
                "legal_other_mean": float(legal_other_mean[sl].mean().item()),
                "third_other_fraction": float(third_other_frac[sl].mean().item()),
                "fourth_other_fraction": float(fourth_other_frac[sl].mean().item()),
                "clearance_peak": float(clearance_peak[sl].max().item()),
                "clearance_min": float(clearance_min[sl].min().item()),
                "xy_drift_peak": float(xy_drift_peak[sl].max().item()),
                "rotation_drift_peak_rad": float(
                    rotation_drift_peak[sl].max().item()
                ),
                "arm_table_clearance_min": float(
                    arm_table_clearance_min[sl].min().item()
                ),
                "arm_tracking_error_peak_rad": float(
                    arm_tracking_error_peak[sl].max().item()
                ),
                "native_terminated_seen": bool(
                    native_terminated_seen[sl].any().item()
                ),
                "native_timeout_seen": bool(native_timeout_seen[sl].any().item()),
                "force_peak": robust_force_peak.detach().cpu().tolist(),
            }

        best_index = int(torch.argmax(candidate_score).item())
        current_any = candidate_record(best_index)
        if best_any is None or current_any["score"] > best_any["score"]:
            best_any = current_any
        pass_ids = strict_group_pass.nonzero(as_tuple=False).squeeze(-1)
        if pass_ids.numel() > 0:
            local = int(torch.argmax(candidate_score[pass_ids]).item())
            current_pass = candidate_record(int(pass_ids[local].item()))
            if best_pass is None or current_pass["score"] > best_pass["score"]:
                best_pass = current_pass

        report = best_pass if best_pass is not None else best_any
        assert report is not None
        print(
            f"iter {iteration + 1:02d}: pass={int(strict_group_pass.sum().item())}/"
            f"{num_candidates} "
            f"best_score={report['score']:.3f} stable={report['stable_streak_peak']} "
            f"legal={report['legal_other_mean']:.2f} q={report['q_grasp']:.3f} "
            f"third={report['third_other_fraction']:.3f} "
            f"F={max(report['force_peak']):.2f}N "
            f"xy={report['xy_drift_peak']:.4f}m "
            f"rot={report['rotation_drift_peak_rad']:.3f}rad "
            f"arm_d={report['arm_delta_abs_max_rad']:.3f}rad "
            f"track={report['arm_tracking_error_peak_rad']:.3f}rad",
            flush=True,
        )

    selected = best_pass if best_pass is not None else best_any
    assert selected is not None
    output = {
        "format_version": 1,
        "contract": (
            "strict_power_close_arm_micro7_hybrid14_public_v1"
            if args_cli.public_controller
            else (
                "strict_power_close_arm_micro7_hybrid14_oracle_v1"
                if search_arm_micro
                else "strict_power_close_hybrid14_v1"
            )
        ),
        "controller": (
            "public_incremental_arm_hybrid14_with_runtime_shields"
            if args_cli.public_controller
            else (
                "oracle_direct_joint_target_requires_public_replay"
                if search_arm_micro
                else "public_hybrid14_zero_arm"
            )
        ),
        "seed": args_cli.seed,
        "population": num_envs,
        "candidate_count": num_candidates,
        "replicates_per_candidate": replicates,
        "iterations": args_cli.iterations,
        "align_steps": args_cli.align_steps,
        "arm_delta_limit_rad": args_cli.arm_delta_limit_rad,
        "arm_delta_penalty": args_cli.arm_delta_penalty,
        "initial_hold_latent": (
            initial_hold_action[0].detach().cpu().tolist()
            if initial_hold_action is not None
            else None
        ),
        "initial_hold_decode_error_rad": initial_hold_decode_error,
        "close_steps": args_cli.close_steps,
        "eval_steps": args_cli.eval_steps,
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "initial_action_input": str(initial_path),
        "initial_action_sha256": _sha256(initial_path),
        "pregrasp": pregrasp,
        "hand_joint_names": [u.robot.joint_names[i] for i in u._hand_ids_t.tolist()],
        "arm_joint_names": [u.robot.joint_names[i] for i in u._arm_ids_t.tolist()],
        "fingertip_force_order": list(u.ee_names),
        "thresholds": {
            "required_legal_other_contacts": POWER_REQUIRED_OTHER_CONTACTS,
            "power_grasp_quality": POWER_GRASP_QUALITY_MIN,
            "hold_quality": POWER_HOLD_QUALITY_MIN,
            "safe_force_n": POWER_FORCE_LIMIT,
            "confirm_steps": POWER_STABLE_FRAMES,
            "power_latch_confirm_steps": POWER_LATCH_CONFIRM_FRAMES,
            "power_latch_release_steps": POWER_LATCH_RELEASE_FRAMES,
            "unlatched_lift_m": POWER_CLEARANCE_LIMIT,
            "horizontal_drift_m": POWER_XY_DRIFT_LIMIT,
            "rotation_drift_rad": POWER_ROTATION_DRIFT_LIMIT,
        },
        "result": selected,
        # Compatibility alias for the replay tool and downstream teacher collectors.
        "results": {"hybrid14": selected},
    }
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"wrote {output_path} strict_pass={selected['strict_power_close_pass']}",
        flush=True,
    )
    env.close()
    if args_cli.require_pass and not selected["strict_power_close_pass"]:
        raise RuntimeError("CEM did not find a strict power-close action")


if __name__ == "__main__":
    main()
    simulation_app.close()
