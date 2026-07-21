#!/usr/bin/env python3
"""Replay a saved hybrid14 closure through the formal 21-D environment action path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Hybrid action decoder/physics regression.")
parser.add_argument("--input", default="/tmp/pick_tool_hand_space_seed0.json")
parser.add_argument(
    "--action_input",
    default=None,
    help=(
        "optional artifact supplying results.hybrid14 while --input supplies the "
        "recoverable pregrasp; useful for replaying refined actions whose compact "
        "artifact omitted simulator state"
    ),
)
parser.add_argument("--num_envs", type=int, default=3)
parser.add_argument(
    "--replay_arm_micro",
    action="store_true",
    help=(
        "track the saved arm_delta_target_rad through the formal incremental arm controller "
        "before applying the saved hybrid14 close"
    ),
)
parser.add_argument(
    "--align_steps",
    type=int,
    default=None,
    help="arm alignment frames; defaults to the action artifact's align_steps",
)
parser.add_argument("--close_steps", type=int, default=48)
parser.add_argument("--eval_steps", type=int, default=12)
parser.add_argument("--output", default=None, help="optional JSON replay report")
parser.add_argument(
    "--require_power_close",
    action="store_true",
    help=(
        "require a safe thumb-plus-three power close for 15 consecutive frames; "
        "the default preserves the historical thumb-plus-two replay regression"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if args_cli.align_steps is not None and args_cli.align_steps < 0:
    parser.error("--align_steps must be non-negative")
if args_cli.require_power_close and args_cli.close_steps + args_cli.eval_steps < 19:
    parser.error("--require_power_close needs at least 19 close/eval physics frames")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401
from xhand_inhand.tasks.direct.pick_cube.pick_cube_env import PickCubeEnv
from xhand_inhand.tasks.direct.pick_tool_token.hybrid_action import (
    apply_asymmetric_joint_residual,
    invert_asymmetric_joint_residual,
)
from power_close_search_contract import (
    POWER_CLEARANCE_LIMIT,
    POWER_FORCE_LIMIT,
    POWER_ROTATION_DRIFT_LIMIT,
    POWER_STABLE_FRAMES,
    POWER_XY_DRIFT_LIMIT,
    power_close_stable_frame,
    strict_power_close_pass,
    update_power_grasp_latch,
    update_stable_streak,
)


def _quat_angle(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    dot = (reference * current).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


@torch.inference_mode()
def main() -> None:
    artifact = json.load(open(args_cli.input))
    action_artifact = (
        json.load(open(args_cli.action_input))
        if args_cli.action_input is not None
        else artifact
    )
    result = action_artifact["results"]["hybrid14"]
    pregrasp = artifact["pregrasp"]
    artifact_align_steps = action_artifact.get("align_steps", 0)
    align_steps = (
        int(artifact_align_steps)
        if args_cli.align_steps is None
        else args_cli.align_steps
    )
    if align_steps < 0:
        raise ValueError("resolved align_steps must be non-negative")
    saved_arm_delta = result.get("arm_delta_target_rad")
    has_arm_micro = isinstance(saved_arm_delta, list) and any(
        abs(float(value)) > 1.0e-8 for value in saved_arm_delta
    )
    if args_cli.replay_arm_micro:
        if not isinstance(saved_arm_delta, list) or len(saved_arm_delta) != 7:
            raise ValueError("--replay_arm_micro requires result.arm_delta_target_rad[7]")
        if align_steps < 1:
            raise ValueError("--replay_arm_micro requires at least one alignment frame")
    elif args_cli.require_power_close and has_arm_micro:
        raise ValueError(
            "power-close artifact contains an arm micro-adjustment; pass --replay_arm_micro "
            "instead of silently dropping it"
        )
    missing_pregrasp = {
        "joint_pos",
        "object_local_pos",
        "object_quat",
    } - pregrasp.keys()
    if missing_pregrasp:
        raise ValueError(
            "--input artifact does not contain recoverable pregrasp fields: "
            + ", ".join(sorted(missing_pregrasp))
        )

    num_envs = args_cli.num_envs
    cfg = parse_env_cfg("Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=num_envs)
    cfg.episode_length_s = 120.0
    cfg.terminate_on_drop = False
    cfg.success_hold_steps = 100000
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    dev = u.device
    env_ids = u.robot._ALL_INDICES

    if obs["policy"].shape != (num_envs, 115) or not torch.isfinite(obs["policy"]).all():
        raise AssertionError("formal observation is not finite (N, 115)")
    base_obs = PickCubeEnv._get_observations(u)["policy"]
    clearance = u._object_true_min_z() - u._table_surface_z
    lift = torch.clamp(clearance / cfg.lift_success_height, 0.0, 1.0).unsqueeze(-1)
    expected_legacy_prefix = torch.cat((base_obs[:, :70], u.actions[:, :16], lift), dim=-1)
    formal_obs = u._get_observations()["policy"]
    prefix_error = float((formal_obs[:, :87] - expected_legacy_prefix).abs().max())
    if prefix_error >= 1.0e-7:
        raise AssertionError(f"old 87-D observation prefix shifted by {prefix_error:.3g}")

    # Validate -1/0/+1 through the actual environment decoder.  Three environments exercise
    # row isolation as well; a one- or two-env physical feasibility run still checks endpoints.
    probe_rows = max(num_envs, 3)
    residual_probe = torch.zeros((probe_rows, u._n_tokens + u._n_distal_residuals), device=dev)
    residual_probe[0, u._n_tokens :] = -1.0
    residual_probe[2, u._n_tokens :] = 1.0
    hand_lower = u.dof_lower[:, u._hand_ids_t]
    hand_upper = u.dof_upper[:, u._hand_ids_t]

    def raw_decode(value: torch.Tensor) -> torch.Tensor:
        token_target = u.retarget.retarget_from_unit_action(value[:, : u._n_tokens])[:, u._retarget2isaac]
        target, _ = apply_asymmetric_joint_residual(
            token_target,
            hand_lower,
            hand_upper,
            value[:, u._n_tokens :],
            u._distal_hand_ids,
        )
        return target

    # Runtime joint limits are identical across cloned environments; repeat row zero when the
    # requested physical batch is smaller than the three-row decoder probe.
    probe_lower = hand_lower[:1].expand(probe_rows, -1)
    probe_upper = hand_upper[:1].expand(probe_rows, -1)

    def probe_decode(value: torch.Tensor) -> torch.Tensor:
        token_target = u.retarget.retarget_from_unit_action(value[:, : u._n_tokens])[
            :, u._retarget2isaac
        ]
        target, _ = apply_asymmetric_joint_residual(
            token_target,
            probe_lower,
            probe_upper,
            value[:, u._n_tokens :],
            u._distal_hand_ids,
        )
        return target

    probe_target = probe_decode(residual_probe)
    probe_zero = probe_decode(torch.zeros_like(residual_probe))
    if not torch.allclose(
        probe_target[0, u._distal_hand_ids], probe_lower[0, u._distal_hand_ids]
    ):
        raise AssertionError("-1 residual did not reach every distal lower limit")
    if not torch.allclose(probe_target[1], probe_zero[1]):
        raise AssertionError("zero-residual row was contaminated by another environment")
    if not torch.allclose(
        probe_target[2, u._distal_hand_ids], probe_upper[2, u._distal_hand_ids]
    ):
        raise AssertionError("+1 residual did not reach every distal upper limit")

    # An asynchronous reset must invalidate only that environment's potential history.
    if num_envs >= 2:
        history = torch.linspace(0.2, 0.2 * num_envs, num_envs, device=dev)
        u._prev_close_quality.copy_(history)
        u._prev_wrap_quality.copy_(history + 0.1)
        u._prev_lift_potential.copy_(history - 0.1)
        u._potential_initialized.fill_(True)
        u._reset_idx(torch.tensor((0,), device=dev))
        if bool(u._potential_initialized[0]) or float(u._prev_close_quality[0]) != 0.0:
            raise AssertionError("partial reset retained stale potential in reset env")
        if not torch.equal(u._prev_close_quality[1:], history[1:]):
            raise AssertionError("partial reset modified non-reset environments")

    joint = torch.tensor(pregrasp["joint_pos"], device=dev).unsqueeze(0).repeat(num_envs, 1)
    u.robot.write_joint_state_to_sim(joint, torch.zeros_like(joint), env_ids=env_ids)
    u.robot.set_joint_position_target(joint, env_ids=env_ids)
    u.dof_targets.copy_(joint)
    pose = torch.zeros((num_envs, 7), device=dev)
    pose[:, :3] = torch.tensor(pregrasp["object_local_pos"], device=dev) + u.scene.env_origins
    pose[:, 3:7] = torch.tensor(pregrasp["object_quat"], device=dev).unsqueeze(0)
    u.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    u.object.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=dev), env_ids=env_ids)
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
    u.actions.zero_()
    u.prev_actions.zero_()
    u._compute_intermediate_values()

    hand_action = torch.tensor(result["latent"], device=dev).unsqueeze(0).repeat(num_envs, 1)
    decoded = raw_decode(hand_action)
    expected = torch.tensor(result["target"], device=dev).unsqueeze(0).repeat(num_envs, 1)
    max_decode_error = float((decoded - expected).abs().max())
    if max_decode_error >= 1.0e-6:
        raise AssertionError(f"formal hybrid decoder differs from benchmark by {max_decode_error:.3g} rad")

    initial_hand_action = hand_action.clone()
    initial_hand_decode_error = 0.0
    if align_steps > 0:
        pregrasp_token = pregrasp.get("token")
        if not isinstance(pregrasp_token, list) or len(pregrasp_token) != u._n_tokens:
            raise ValueError("an alignment phase requires pregrasp.token[9]")
        seed_token = (
            torch.tensor(pregrasp_token, dtype=torch.float32, device=dev)
        )
        # The captured policy action can be far from the instantaneous joint pose because the
        # controller has EMA/actuator lag.  Fit a formal hybrid14 action to that pose before moving
        # the wrist; otherwise the supposed "open-hand hold" can close by >0.5 rad and rotate the
        # hammer before the searched alignment even begins.
        fit_population = 4096
        fit_elites = 256
        fit_generator = torch.Generator(device=dev).manual_seed(99173)
        fit_mean = seed_token.clone()
        fit_std = torch.full_like(fit_mean, 0.30)
        snapshot_hand_one = joint[:1, u._hand_ids_t]
        fit_lower = hand_lower[:1].expand(fit_population, -1)
        fit_upper = hand_upper[:1].expand(fit_population, -1)
        fit_target = snapshot_hand_one.expand(fit_population, -1)
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
        initial_hand_action = best_fit_action.unsqueeze(0).repeat(num_envs, 1)
        snapshot_hand = joint[:, u._hand_ids_t]
        initial_hand_decode_error = float(
            (raw_decode(initial_hand_action) - snapshot_hand).abs().max()
        )

    arm_start = joint[:, u._arm_ids_t].clone()
    if args_cli.replay_arm_micro:
        arm_delta = (
            torch.tensor(saved_arm_delta, dtype=torch.float32, device=dev)
            .unsqueeze(0)
            .repeat(num_envs, 1)
        )
    else:
        arm_delta = torch.zeros_like(arm_start)
    arm_target = torch.maximum(
        torch.minimum(arm_start + arm_delta, u.dof_upper[:, u._arm_ids_t]),
        u.dof_lower[:, u._arm_ids_t],
    )
    if float((arm_target - (arm_start + arm_delta)).abs().max()) > 1.0e-7:
        raise ValueError("saved arm micro target lies outside the runtime joint limits")

    latch_seen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    q_peak = torch.zeros(num_envs, device=dev)
    power_close_peak = torch.zeros(num_envs, device=dev)
    power_wrap_peak = torch.zeros(num_envs, device=dev)
    power_grasp_peak = torch.zeros(num_envs, device=dev)
    hold_peak = torch.zeros(num_envs, device=dev)
    thumb_seen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    legal_other_peak = torch.zeros(num_envs, dtype=torch.long, device=dev)
    strict_streak = torch.zeros(num_envs, dtype=torch.long, device=dev)
    strict_streak_peak = torch.zeros(num_envs, dtype=torch.long, device=dev)
    power_is_grasped = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    power_latch_confirm = torch.zeros(num_envs, dtype=torch.long, device=dev)
    power_latch_release = torch.zeros(num_envs, dtype=torch.long, device=dev)
    overforce_seen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    unlatched_lift_seen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    launch_seen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    horizontal_escape_seen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    rotation_escape_seen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    unexpected_done_seen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    force_peak = torch.zeros((num_envs, len(u.ee_names)), device=dev)
    clearance_peak = torch.full((num_envs,), -float("inf"), device=dev)
    clearance_min = torch.full((num_envs,), float("inf"), device=dev)
    start_xy = u._object_com_position_w()[:, :2].clone()
    snapshot_quat = (
        torch.tensor(pregrasp["object_quat"], dtype=torch.float32, device=dev)
        .unsqueeze(0)
        .repeat(num_envs, 1)
    )
    xy_drift_peak = torch.zeros(num_envs, device=dev)
    rotation_drift_peak = torch.zeros(num_envs, device=dev)
    arm_tracking_error_peak = torch.zeros(num_envs, device=dev)
    arm_table_clearance_min = torch.full((num_envs,), float("inf"), device=dev)
    total_steps = align_steps + args_cli.close_steps + args_cli.eval_steps
    arm_action_denominator = cfg.action_scale * cfg.act_moving_average
    for step in range(total_steps):
        action = torch.zeros((num_envs, cfg.action_space), device=dev)
        if step < align_steps:
            x = float(step + 1) / float(align_steps)
            blend = x * x * (3.0 - 2.0 * x)
            desired_arm_target = arm_start + blend * arm_delta
            action[:, u._n_arm :] = initial_hand_action
        else:
            desired_arm_target = arm_target
            action[:, u._n_arm :] = hand_action
        current_arm_target = u.dof_targets[:, u._arm_ids_t]
        action[:, : u._n_arm] = torch.clamp(
            (desired_arm_target - current_arm_target) / arm_action_denominator,
            -1.0,
            1.0,
        )
        _, _, terminated, truncated, _ = env.step(action)
        unexpected_done_seen |= terminated | truncated
        arm_tracking_error_peak = torch.maximum(
            arm_tracking_error_peak,
            (u.dof_targets[:, u._arm_ids_t] - desired_arm_target)
            .abs()
            .max(dim=-1)
            .values,
        )
        signals = u._compute_grasp_signals()
        latch_seen |= u._is_grasped
        q_peak = torch.maximum(q_peak, signals["grasp_quality"])
        power_close_peak = torch.maximum(power_close_peak, signals["power_close_quality"])
        power_wrap_peak = torch.maximum(power_wrap_peak, signals["power_wrap_quality"])
        power_grasp_peak = torch.maximum(power_grasp_peak, signals["power_grasp_quality"])
        hold_peak = torch.maximum(hold_peak, signals["hold_quality"])
        thumb_seen |= signals["power_thumb_contact"]
        legal_other_peak = torch.maximum(
            legal_other_peak, signals["power_legal_other_contact_count"]
        )
        force_peak = torch.maximum(force_peak, signals["force_magnitude"])
        max_force = signals["force_magnitude"].max(dim=-1).values
        clearance = u._object_true_min_z() - u._table_surface_z
        clearance_peak = torch.maximum(clearance_peak, clearance)
        clearance_min = torch.minimum(clearance_min, clearance)
        xy_drift = (u._object_com_position_w()[:, :2] - start_xy).norm(dim=-1)
        xy_drift_peak = torch.maximum(xy_drift_peak, xy_drift)
        rotation_drift = _quat_angle(snapshot_quat, u.object.data.root_quat_w)
        rotation_drift_peak = torch.maximum(rotation_drift_peak, rotation_drift)
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
        strict_streak, strict_streak_peak = update_stable_streak(
            strict_streak, strict_streak_peak, stable
        )
        was_power_grasped = power_is_grasped
        power_is_grasped, power_latch_confirm, power_latch_release = (
            update_power_grasp_latch(
                signals["power_grasp_quality"],
                power_is_grasped,
                power_latch_confirm,
                power_latch_release,
            )
        )
        overforce_seen |= max_force > POWER_FORCE_LIMIT
        unlatched_lift_seen |= (clearance > POWER_CLEARANCE_LIMIT) & (
            ~was_power_grasped
        )
        launch_seen |= clearance > 0.02
        horizontal_escape_seen |= xy_drift > POWER_XY_DRIFT_LIMIT
        rotation_escape_seen |= rotation_drift > POWER_ROTATION_DRIFT_LIMIT

    strict_power_pass = strict_power_close_pass(
        strict_streak,
        force_peak.max(dim=-1).values,
        xy_drift_peak,
        rotation_drift_peak,
        clearance_peak,
        unexpected_done_seen,
    )

    print(f"PASS finite obs and legacy-prefix max error: {prefix_error:.3g}", flush=True)
    print("PASS multi-env residual endpoints and partial-reset isolation", flush=True)
    print(f"PASS decoder max error: {max_decode_error:.3g} rad", flush=True)
    print(
        f"formal arm replay align_steps={align_steps} "
        f"initial_hand_decode_error={initial_hand_decode_error:.4f}rad "
        f"arm_tracking_error_peak={float(arm_tracking_error_peak.max()):.4f}rad",
        flush=True,
    )
    print(f"closure q_peak={q_peak.tolist()} latch_seen={latch_seen.tolist()}", flush=True)
    print(f"clearance_peak={clearance_peak.tolist()}m force_peak={force_peak.tolist()}", flush=True)
    if args_cli.require_power_close:
        print(
            "power-close "
            f"thumb_seen={thumb_seen.tolist()} legal_other_peak={legal_other_peak.tolist()} "
            f"q_close_peak={power_close_peak.tolist()} q_wrap_peak={power_wrap_peak.tolist()} "
            f"q_grasp_peak={power_grasp_peak.tolist()} hold_peak={hold_peak.tolist()} "
            f"power_latched_end={power_is_grasped.tolist()} "
            f"stable_streak_end={strict_streak.tolist()} "
            f"stable_streak_peak={strict_streak_peak.tolist()}",
            flush=True,
        )
        print(
            "power-safety "
            f"overforce={overforce_seen.tolist()} "
            f"unlatched_lift={unlatched_lift_seen.tolist()} launch={launch_seen.tolist()} "
            f"horizontal_escape={horizontal_escape_seen.tolist()} "
            f"rotation_escape={rotation_escape_seen.tolist()} "
            f"unexpected_done={unexpected_done_seen.tolist()} "
            f"strict_pass={strict_power_pass.tolist()}",
            flush=True,
        )
        print(
            f"trajectory xy_peak={xy_drift_peak.tolist()}m "
            f"rotation_peak={rotation_drift_peak.tolist()}rad "
            f"clearance_min={clearance_min.tolist()}m "
            f"arm_table_clearance_min={arm_table_clearance_min.tolist()}m",
            flush=True,
        )
        if args_cli.output is not None:
            replay_report = {
                "format_version": 1,
                "contract": "formal_21d_power_close_public_replay_v1",
                "input": str(Path(args_cli.input).resolve()),
                "action_input": str(
                    Path(args_cli.action_input or args_cli.input).resolve()
                ),
                "num_envs": num_envs,
                "align_steps": align_steps,
                "close_steps": args_cli.close_steps,
                "eval_steps": args_cli.eval_steps,
                "initial_hand_decode_error_rad": initial_hand_decode_error,
                "arm_tracking_error_peak_rad": float(
                    arm_tracking_error_peak.max().item()
                ),
                "strict_pass_count": int(strict_power_pass.sum().item()),
                "strict_pass": strict_power_pass.cpu().tolist(),
                "stable_streak_at_end": strict_streak.cpu().tolist(),
                "stable_streak_peak": strict_streak_peak.cpu().tolist(),
                "power_latched_at_end": power_is_grasped.cpu().tolist(),
                "max_legal_other_contacts": legal_other_peak.cpu().tolist(),
                "power_grasp_quality_peak": power_grasp_peak.cpu().tolist(),
                "hold_quality_peak": hold_peak.cpu().tolist(),
                "force_peak_n": force_peak.cpu().tolist(),
                "true_clearance_peak_m": clearance_peak.cpu().tolist(),
                "true_clearance_min_m": clearance_min.cpu().tolist(),
                "xy_drift_peak_m": xy_drift_peak.cpu().tolist(),
                "rotation_drift_peak_rad": rotation_drift_peak.cpu().tolist(),
                "arm_table_clearance_min_m": arm_table_clearance_min.cpu().tolist(),
                "unexpected_done": unexpected_done_seen.cpu().tolist(),
            }
            output_path = Path(args_cli.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(replay_report, indent=2), encoding="utf-8"
            )
            print(f"wrote replay report {output_path}", flush=True)
        if not bool(strict_power_pass.all()):
            raise AssertionError(
                "saved hybrid14 closure did not reproduce a safe 15-frame thumb-plus-three "
                "power close"
            )
    else:
        if not bool(latch_seen.all()):
            raise AssertionError("saved hybrid14 closure did not reproduce the robust grasp latch")
        if bool((clearance_peak > 0.02).any()):
            raise AssertionError(f"closure launched the hammer by {float(clearance_peak.max()):.4f}m")
    print("ALL HYBRID ACTION REPLAY TESTS PASSED", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
