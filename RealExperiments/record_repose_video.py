#!/usr/bin/env python3
"""Record a close-up HD rollout video of a repose policy (deployment stack).

Run:
    conda activate env_isaaclab
    python RealExperiments/record_repose_video.py --task Xhand-Repose-Cube-OpenAI-LSTM-Hard-Direct-v0 \
        --checkpoint <ckpt.pth> --steps 600
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Xhand-Repose-Cube-OpenAI-LSTM-Hard-Direct-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--out", default="/tmp/repose_rollout")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

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
    env_cfg.observation_noise_model = None
    env_cfg.action_noise_model = None
    # close-up camera on the hand (env frame: palm at (0,0,0.5), cube ~(0,0.1,0.55))
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.45, 0.5, 0.85)
    env_cfg.viewer.lookat = (0.0, 0.1, 0.55)
    env_cfg.viewer.resolution = (args_cli.width, args_cli.height)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, video_folder=args_cli.out, video_length=args_cli.steps,
        step_trigger=lambda step: step == 0, disable_logger=True,
    )
    u = env.unwrapped
    env.reset()

    policy = rr.LstmPolicy(ckpt)
    prev_action = np.zeros(12, dtype=np.float32)
    successes = 0
    for step in range(args_cli.steps):
        u._compute_intermediate_values()
        obs = rr.build_obs(
            u.fingertip_pos[0].cpu().numpy(),
            u.object_pos[0].cpu().numpy(),
            u.object_rot[0].cpu().numpy(),
            u.goal_rot[0].cpu().numpy(),
            prev_action,
        )
        action = policy.act(obs)
        _, _, term, trunc, _ = env.step(torch.tensor(action, device=u.device).unsqueeze(0))
        prev_action = action
        if term[0] or trunc[0]:
            policy.reset()
            prev_action = np.zeros(12, dtype=np.float32)
        s = float(u.successes[0].item())
        if s > successes:
            successes = int(s)
            print(f"[video] success #{successes} at step {step}", flush=True)

    env.close()
    print(f"[video] saved to {args_cli.out}, successes on camera: {successes}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
