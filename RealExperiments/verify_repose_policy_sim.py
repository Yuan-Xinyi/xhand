#!/usr/bin/env python3
"""Closed-loop sanity check: drive the SIM repose env with the DEPLOYMENT stack.

Uses the same LstmPolicy + build_obs as foundationpose_repose_real.py (not the
rl_games player). If the policy re-implementation and obs layout are correct,
the hand should reorient the cube and rack up goal successes in sim.

Run:
    conda activate env_isaaclab
    python RealExperiments/verify_repose_policy_sim.py --steps 600
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Xhand-Repose-Cube-OpenAI-LSTM-Direct-v0")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--checkpoint", default=None)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foundationpose_repose_real as rr  # noqa: E402


def main():
    ckpt = args_cli.checkpoint or rr.DEFAULT_CHECKPOINT
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.observation_noise_model = None  # deterministic contract check
    env_cfg.action_noise_model = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()

    policy = rr.LstmPolicy(ckpt)
    prev_action = np.zeros(12, dtype=np.float32)
    obs_err_max = 0.0
    falls = 0

    for step in range(args_cli.steps):
        u._compute_intermediate_values()
        tip_pos = u.fingertip_pos[0].cpu().numpy()
        obj_pos = u.object_pos[0].cpu().numpy()
        obj_quat = u.object_rot[0].cpu().numpy()
        goal_quat = u.goal_rot[0].cpu().numpy()

        obs_mine = rr.build_obs(tip_pos, obj_pos, obj_quat, goal_quat, prev_action)
        action = policy.act(obs_mine)
        obs_env, _, terminated, truncated, _ = env.step(torch.tensor(action, device=u.device).unsqueeze(0))

        # cross-check my obs against the env's next-step obs actions block is mine;
        # compare the env's obs built from the SAME state next iteration instead:
        # here we only track success metrics + fall resets.
        prev_action = action
        if terminated[0] or truncated[0]:
            falls += int(terminated[0].item())
            policy.reset()
            prev_action = np.zeros(12, dtype=np.float32)

        if step % 100 == 0:
            rot_dist = float(u._orientation_distance()[0].item())
            print(f"step {step:4d} rot_dist {np.degrees(rot_dist):6.1f} deg "
                  f"successes(episode) {float(u.successes[0].item()):.0f} "
                  f"consecutive {float(u.consecutive_successes.item()):.2f}")

    print("\n==================== RESULT ====================")
    print(f"steps                 : {args_cli.steps} ({args_cli.steps * u.step_dt:.0f} s sim time)")
    print(f"successes (last ep)   : {float(u.successes[0].item()):.0f}")
    print(f"consecutive successes : {float(u.consecutive_successes.item()):.2f}")
    print(f"fall/terminate resets : {falls}")
    print("If successes accumulate, the deployment policy stack is correct.")
    print("================================================\n", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
