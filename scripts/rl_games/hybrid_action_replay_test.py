#!/usr/bin/env python3
"""Replay a saved hybrid14 closure through the formal 21-D environment action path."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Hybrid action decoder/physics regression.")
parser.add_argument("--input", default="/tmp/pick_tool_hand_space_seed0.json")
parser.add_argument("--close_steps", type=int, default=48)
parser.add_argument("--eval_steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401
from xhand_inhand.tasks.direct.pick_cube.pick_cube_env import PickCubeEnv
from xhand_inhand.tasks.direct.pick_tool_token.hybrid_action import apply_asymmetric_joint_residual


@torch.inference_mode()
def main() -> None:
    artifact = json.load(open(args_cli.input))
    result = artifact["results"]["hybrid14"]
    pregrasp = artifact["pregrasp"]

    num_envs = 3
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

    # Validate -1/0/+1 on three independent rows through the actual environment decoder.
    residual_probe = torch.zeros((num_envs, u._n_tokens + u._n_distal_residuals), device=dev)
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

    probe_target = raw_decode(residual_probe)
    probe_zero = raw_decode(torch.zeros_like(residual_probe))
    if not torch.allclose(probe_target[0, u._distal_hand_ids], hand_lower[0, u._distal_hand_ids]):
        raise AssertionError("-1 residual did not reach every distal lower limit")
    if not torch.allclose(probe_target[1], probe_zero[1]):
        raise AssertionError("zero-residual row was contaminated by another environment")
    if not torch.allclose(probe_target[2, u._distal_hand_ids], hand_upper[2, u._distal_hand_ids]):
        raise AssertionError("+1 residual did not reach every distal upper limit")

    # An asynchronous reset must invalidate only that environment's potential history.
    u._prev_close_quality[:] = torch.tensor((0.2, 0.4, 0.6), device=dev)
    u._prev_wrap_quality[:] = torch.tensor((0.3, 0.5, 0.7), device=dev)
    u._prev_lift_potential[:] = torch.tensor((0.1, 0.2, 0.3), device=dev)
    u._potential_initialized.fill_(True)
    u._reset_idx(torch.tensor((0,), device=dev))
    if bool(u._potential_initialized[0]) or float(u._prev_close_quality[0]) != 0.0:
        raise AssertionError("partial reset retained stale potential in reset env")
    if not torch.equal(u._prev_close_quality[1:], torch.tensor((0.4, 0.6), device=dev)):
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
    action = torch.zeros((num_envs, cfg.action_space), device=dev)
    action[:, u._n_arm :] = hand_action
    decoded = raw_decode(hand_action)
    expected = torch.tensor(result["target"], device=dev).unsqueeze(0).repeat(num_envs, 1)
    max_decode_error = float((decoded - expected).abs().max())
    if max_decode_error >= 1.0e-6:
        raise AssertionError(f"formal hybrid decoder differs from benchmark by {max_decode_error:.3g} rad")

    latch_seen = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    q_peak = torch.zeros(num_envs, device=dev)
    force_peak = torch.zeros((num_envs, len(u.ee_names)), device=dev)
    clearance_peak = torch.full((num_envs,), -float("inf"), device=dev)
    total_steps = args_cli.close_steps + args_cli.eval_steps
    for _ in range(total_steps):
        env.step(action)
        signals = u._compute_grasp_signals()
        latch_seen |= u._is_grasped
        q_peak = torch.maximum(q_peak, signals["grasp_quality"])
        force_peak = torch.maximum(force_peak, signals["force_magnitude"])
        clearance_peak = torch.maximum(
            clearance_peak, u._object_true_min_z() - u._table_surface_z
        )

    print(f"PASS finite obs and legacy-prefix max error: {prefix_error:.3g}", flush=True)
    print("PASS multi-env residual endpoints and partial-reset isolation", flush=True)
    print(f"PASS decoder max error: {max_decode_error:.3g} rad", flush=True)
    print(f"closure q_peak={q_peak.tolist()} latch_seen={latch_seen.tolist()}", flush=True)
    print(f"clearance_peak={clearance_peak.tolist()}m force_peak={force_peak.tolist()}", flush=True)
    if not bool(latch_seen.all()):
        raise AssertionError("saved hybrid14 closure did not reproduce the robust grasp latch")
    if bool((clearance_peak > 0.02).any()):
        raise AssertionError(f"closure launched the hammer by {float(clearance_peak.max()):.4f}m")
    print("ALL HYBRID ACTION REPLAY TESTS PASSED", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
