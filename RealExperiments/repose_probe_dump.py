#!/usr/bin/env python3
"""Dump ground-truth constants + FK reference data from the repose sim env.

Everything the real-world deployment script must reproduce exactly is printed
and saved to an .npz: joint order, fingertip body order, joint limits,
in_hand_pos, palm root pose, and fingertip poses at random hand configurations
(for verifying the `one`-library FK against Isaac).

Run:
    conda activate env_isaaclab
    python RealExperiments/repose_probe_dump.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Xhand-Repose-Cube-OpenAI-LSTM-Direct-v0")
parser.add_argument("--out", default="RealExperiments/repose_probe.npz")
parser.add_argument("--num_configs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import xhand_inhand.tasks  # noqa: F401, E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # deterministic dump: no obs/action noise
    env_cfg.observation_noise_model = None
    env_cfg.action_noise_model = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()

    print("\n=================== REPOSE PROBE ===================")
    print(f"task                : {args_cli.task}")
    print(f"joint_names (isaac) : {u.hand.joint_names}")
    print(f"actuated_dof_indices: {u.actuated_dof_indices}")
    body_names = u.hand.body_names
    print(f"body_names          : {body_names}")
    finger_body_names = [body_names[i] for i in u.finger_bodies]
    print(f"fingertip order     : {finger_body_names}")
    lower = u.hand_dof_lower_limits[0].cpu().numpy()
    upper = u.hand_dof_upper_limits[0].cpu().numpy()
    print(f"limits lower        : {np.array2string(lower, precision=4)}")
    print(f"limits upper        : {np.array2string(upper, precision=4)}")
    print(f"in_hand_pos         : {u.in_hand_pos[0].cpu().numpy()}")
    root_pos = u.hand.data.root_pos_w[0].cpu().numpy() - u.scene.env_origins[0].cpu().numpy()
    root_quat = u.hand.data.root_quat_w[0].cpu().numpy()
    print(f"palm root pos (env) : {root_pos}")
    print(f"palm root quat wxyz : {root_quat}")
    print(f"obj default pos     : {u.object.data.default_root_state[0, :3].cpu().numpy()}")
    print(f"step_dt             : {u.step_dt}")
    print(f"act_moving_average  : {u.cfg.act_moving_average}")
    print(f"success_tolerance   : {u.success_tolerance} (cfg {u.cfg.success_tolerance})")
    print(f"fall_dist           : {u.cfg.fall_dist}")

    # which body is the articulation root / palm?
    palm_idx = body_names.index("palm") if "palm" in body_names else 0
    origins = u.scene.env_origins[0].cpu().numpy()

    # FK reference: random joint configs -> fingertip + palm poses
    rng = np.random.default_rng(0)
    n = args_cli.num_configs
    qs, tip_pos, tip_quat, palm_pos, palm_quat, obs_list, act_list = [], [], [], [], [], [], []
    obj_pos_l, obj_quat_l, goal_quat_l = [], [], []
    for k in range(n):
        q = rng.uniform(lower, upper).astype(np.float32)
        qt = torch.tensor(q, device=u.device).unsqueeze(0)
        u.hand.set_joint_position_target(qt)
        u.hand.write_joint_state_to_sim(qt, torch.zeros_like(qt))
        u.sim.step(render=False)
        u.hand.update(u.physics_dt)
        u.object.update(u.physics_dt)
        u._compute_intermediate_values()

        # fake an action so the obs has a defined actions block
        act = rng.uniform(-1, 1, u.cfg.action_space).astype(np.float32)
        u.actions = torch.tensor(act, device=u.device).unsqueeze(0)
        obs = u._get_observations()["policy"][0].cpu().numpy()

        qs.append(u.hand_dof_pos[0].cpu().numpy().copy())
        tip_pos.append(u.fingertip_pos[0].cpu().numpy().copy())
        tip_quat.append(u.fingertip_rot[0].cpu().numpy().copy())
        palm_pos.append(u.hand.data.body_pos_w[0, palm_idx].cpu().numpy() - origins)
        palm_quat.append(u.hand.data.body_quat_w[0, palm_idx].cpu().numpy().copy())
        obj_pos_l.append(u.object_pos[0].cpu().numpy().copy())
        obj_quat_l.append(u.object_rot[0].cpu().numpy().copy())
        goal_quat_l.append(u.goal_rot[0].cpu().numpy().copy())
        obs_list.append(obs)
        act_list.append(act)

    np.savez(
        args_cli.out,
        joint_names=np.array(u.hand.joint_names),
        fingertip_names=np.array(finger_body_names),
        lower=lower,
        upper=upper,
        in_hand_pos=u.in_hand_pos[0].cpu().numpy(),
        root_pos=root_pos,
        root_quat=root_quat,
        q=np.array(qs),
        tip_pos=np.array(tip_pos),
        tip_quat=np.array(tip_quat),
        palm_pos=np.array(palm_pos),
        palm_quat=np.array(palm_quat),
        obj_pos=np.array(obj_pos_l),
        obj_quat=np.array(obj_quat_l),
        goal_quat=np.array(goal_quat_l),
        obs=np.array(obs_list),
        actions=np.array(act_list),
    )
    print(f"\nsaved -> {args_cli.out}")
    print("====================================================\n", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
