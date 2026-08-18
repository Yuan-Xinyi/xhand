#!/usr/bin/env python3
"""Drop an object from multiple attitudes and verify table-supported settled states."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Functional-Pregrasp-Flashlight-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--drop_height", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 8:
        raise ValueError("the deterministic probe requires at least 8 environments")
    if args.steps < 100:
        raise ValueError("--steps must be at least 100")
    return args


def run(args) -> dict:
    import gymnasium as gym
    import omni.usd
    import torch
    from pxr import PhysxSchema, Usd

    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
    from isaaclab_tasks.utils import parse_env_cfg

    import xhand_inhand.tasks  # noqa: F401

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    control_dt = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = max(float(cfg.episode_length_s), (args.steps + 50) * control_dt)
    cfg.reset_arm_joint_noise = 0.0
    cfg.reset_object_pos_noise = (0.0, 0.0)
    cfg.reset_object_yaw_range = (0.0, 0.0)
    cfg.terminate_on_drop = False
    cfg.nudge_option_mode = False
    cfg.nudge_grasp_mode = False
    cfg.close_option_mode = False
    env = gym.make(args.task, cfg=cfg)
    try:
        env.reset()
        u_env = env.unwrapped
        device = u_env.device
        state = u_env.object.data.default_root_state.clone()
        state[:, :3] += u_env.scene.env_origins
        state[:, 0] = u_env.scene.env_origins[:, 0] + float(cfg.object_cfg.init_state.pos[0])
        state[:, 1] = u_env.scene.env_origins[:, 1] + float(cfg.object_cfg.init_state.pos[1])
        state[0, 2] = float(cfg.object_cfg.init_state.pos[2]) + 0.001
        state[1:, 2] = u_env.scene.env_origins[1:, 2] + float(args.drop_height)

        angles = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5 * math.pi, 0.0, 0.0],
                [0.0, 0.5 * math.pi, 0.0],
                [0.7, -0.4, 0.3],
                [-1.0, 0.6, -0.8],
                [1.2, 0.8, 1.7],
                [-0.5, -1.1, 2.4],
            ],
            dtype=torch.float,
            device=device,
        )
        if args.num_envs > angles.shape[0]:
            repeat = angles[-1:].expand(args.num_envs - angles.shape[0], -1)
            angles = torch.cat((angles, repeat), dim=0)
        state[0, 3:7] = u_env.object.data.default_root_state[0, 3:7]
        state[1:, 3:7] = quat_from_euler_xyz(angles[1:, 0], angles[1:, 1], angles[1:, 2])
        state[:, 7:] = 0.0
        env_ids = torch.arange(args.num_envs, dtype=torch.long, device=device)
        u_env.object.write_root_pose_to_sim(state[:, :7], env_ids)
        u_env.object.write_root_velocity_to_sim(state[:, 7:], env_ids)

        actions = torch.zeros((args.num_envs, cfg.action_space), dtype=torch.float, device=device)
        for _ in range(args.steps):
            env.step(actions)

        u_env._compute_intermediate_values()
        clearance = u_env._object_true_min_z() - u_env._table_surface_z
        linear_speed = u_env.object.data.root_com_lin_vel_w.norm(dim=-1)
        angular_speed = u_env.object.data.root_com_ang_vel_w.norm(dim=-1)
        pose_linear_speed, pose_angular_speed = u_env._nudge_pose_delta_speeds()
        object_local_pos = u_env.object.data.root_pos_w - u_env.scene.env_origins
        fingertip_distance = (
            u_env.ee_pos_w - u_env.object.data.root_pos_w.unsqueeze(1)
        ).norm(dim=-1).min(dim=1).values
        fingertip_force = u_env._finger_object_force_magnitudes().max(dim=1).values
        functional_axis = torch.tensor(cfg.nudge_heading_axis, dtype=torch.float, device=device)
        functional_axis = functional_axis.expand(args.num_envs, 3)
        functional_axis_w = quat_apply(u_env.object.data.root_quat_w, functional_axis)
        support_alignment = u_env._nudge_pose_errors()["tip_cos"]
        authored_rigid_bodies = []
        stage = omni.usd.get_context().get_stage()
        object_prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
        for prim in Usd.PrimRange(object_prim):
            api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if api:
                authored_rigid_bodies.append(
                    {
                        "path": str(prim.GetPath()),
                        "linear_damping": float(api.GetLinearDampingAttr().Get()),
                        "angular_damping": float(api.GetAngularDampingAttr().Get()),
                        "sleep_threshold": float(api.GetSleepThresholdAttr().Get()),
                        "stabilization_threshold": float(api.GetStabilizationThresholdAttr().Get()),
                    }
                )
        stable = (
            (clearance.abs() <= 0.003)
            & (pose_linear_speed <= 0.02)
            & (pose_angular_speed <= 0.10)
            & (support_alignment >= cfg.nudge_tip_cos_min)
        )
        payload = {
            "task": args.task,
            "steps": args.steps,
            "control_dt": control_dt,
            "settle_seconds": args.steps * control_dt,
            "stable_count": int(stable.sum().item()),
            "num_envs": args.num_envs,
            "seed_pose_stable": bool(stable[0].item()),
            "clearance_m": clearance.detach().cpu().tolist(),
            "linear_speed_mps": linear_speed.detach().cpu().tolist(),
            "angular_speed_radps": angular_speed.detach().cpu().tolist(),
            "pose_delta_linear_speed_mps": pose_linear_speed.detach().cpu().tolist(),
            "pose_delta_angular_speed_radps": pose_angular_speed.detach().cpu().tolist(),
            "object_local_pos_m": object_local_pos.detach().cpu().tolist(),
            "minimum_fingertip_distance_m": fingertip_distance.detach().cpu().tolist(),
            "maximum_fingertip_force_n": fingertip_force.detach().cpu().tolist(),
            "functional_axis_world": functional_axis_w.detach().cpu().tolist(),
            "support_alignment": support_alignment.detach().cpu().tolist(),
            "final_quat_wxyz": u_env.object.data.root_quat_w.detach().cpu().tolist(),
            "authored_rigid_bodies": authored_rigid_bodies,
        }
        payload["passed"] = bool(
            payload["seed_pose_stable"] and payload["stable_count"] >= args.num_envs - 1
        )
        return payload
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    launcher = AppLauncher(args)
    try:
        result = run(args)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="", flush=True)
        if not result["passed"]:
            raise RuntimeError("drop-settle contract failed; see the JSON payload above")
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
