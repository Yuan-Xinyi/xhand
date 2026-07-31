# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Record the carry policy reaching consecutive pose goals (goal marker visible).

Single-env carry_mode episodes from fresh-latch curriculum spawns; every episode with at
least ``--min_goals`` reached goals is written as an mp4 until ``--max_clips`` are saved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--curriculum_dataset", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=12)
parser.add_argument("--min_goals", type=int, default=2)
parser.add_argument("--max_clips", type=int, default=3)
parser.add_argument("--pos_tolerance", type=float, default=0.04)
parser.add_argument("--rot_tolerance", type=float, default=0.35)
parser.add_argument("--rot_range", type=float, default=0.6)
parser.add_argument("--rel_angle", type=float, default=0.0, help=">0: relative-orientation goals with this max angle (rad)")
parser.add_argument("--lock_arm", action="store_true")
parser.add_argument("--axial", action="store_true")
parser.add_argument("--spindle", action="store_true")
parser.add_argument("--goal_follow", action="store_true")
parser.add_argument("--arm_authority", type=float, default=1.0)
parser.add_argument("--boundary", default="carry_start")
parser.add_argument("--episode_length_s", type=float, default=15.0)
parser.add_argument("--fps", type=int, default=40)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video_folder", type=Path, required=True)
parser.add_argument("--cam_eye", type=float, nargs=3, default=[1.9, 0.95, 0.9])
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.5, 0.0, 0.32])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "rl_games"))
import agent_bridge  # noqa: E402,F401
from flash_rl.agents.flashSAC.network import FlashSACActor  # noqa: E402


def _infer_actor_arch(state: dict) -> tuple[int, int]:
    """(num_blocks, hidden_dim) from weight shapes -- loaders stay valid for any size."""
    hidden = state["embedder.w.w.weight"].shape[0]
    blocks = len({k.split(".")[1] for k in state if k.startswith("encoder.")})
    return blocks, hidden



def load_actor(checkpoint: Path, device: torch.device) -> FlashSACActor:
    payload = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in payload["network_state_dict"].items()}
    blocks, hidden = _infer_actor_arch(canonical if 'canonical' in dir() else state)
    actor = FlashSACActor(num_blocks=blocks, input_dim=115, hidden_dim=hidden, action_dim=21)
    actor.load_state_dict(state)
    return actor.to(device).eval()


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(args_cli.seed)
    cfg = parse_env_cfg("Pick-Tool-Token-Direct-v0", device=args_cli.device, num_envs=1)
    cfg.seed = args_cli.seed
    cfg.carry_mode = True
    cfg.episode_length_s = args_cli.episode_length_s
    cfg.carry_pos_tolerance = args_cli.pos_tolerance
    cfg.carry_rot_tolerance = args_cli.rot_tolerance
    cfg.target_rot_range_roll = (-args_cli.rot_range, args_cli.rot_range)
    cfg.target_rot_range_pitch = (-args_cli.rot_range, args_cli.rot_range)
    if args_cli.rel_angle > 0.0:
        cfg.carry_goal_rel_angle_max = args_cli.rel_angle
    if args_cli.lock_arm:
        cfg.carry_lock_arm = True
        cfg.carry_goal_pos_range = (0.02, 0.02, 0.02)
        cfg.carry_pos_tolerance = 0.08
    if args_cli.goal_follow:
        cfg.carry_goal_follow_object = True
        cfg.carry_goal_pos_range = (0.03, 0.03, 0.03)
        cfg.carry_pos_tolerance = 0.08
    cfg.carry_arm_authority = args_cli.arm_authority
    if args_cli.axial:
        cfg.carry_goal_axial_mode = True
    if args_cli.spindle:
        cfg.carry_spindle_mode = True
        cfg.carry_lock_arm = True
        cfg.carry_goal_axial_mode = True
        cfg.carry_goal_pos_range = (0.0, 0.0, 0.0)
        cfg.carry_pos_tolerance = 0.10
        cfg.carry_goal_follow_object = True
    cfg.curriculum_dataset = str(args_cli.curriculum_dataset)
    cfg.curriculum_boundary = args_cli.boundary
    cfg.curriculum_reset_probability = 1.0
    cfg.curriculum_joint_noise = 0.01
    cfg.viewer.eye = tuple(args_cli.cam_eye)
    cfg.viewer.lookat = tuple(args_cli.cam_lookat)
    cfg.viewer.origin_type = "world"
    env = gym.make("Pick-Tool-Token-Direct-v0", cfg=cfg, render_mode="rgb_array")
    u = env.unwrapped
    actor = load_actor(args_cli.checkpoint, u.device)

    args_cli.video_folder.mkdir(parents=True, exist_ok=True)
    for _ in range(6):
        u.render()
    saved = 0
    import imageio.v2 as imageio

    for episode in range(args_cli.episodes):
        obs, _ = env.reset()
        frames = []
        # The auto-reset inside step() clears per-env counters before step returns, so track
        # goals via the cumulative total instead of the per-env count.
        goals_before = int(u._carry_goals_total)
        timeouts_before = int(u._carry_goal_timeout_total)
        prev_goals, prev_timeouts = goals_before, timeouts_before
        events = []
        flash = 0  # >0: draw border on this frame; sign encodes color
        done = False
        dropped = False
        while not done:
            mean, _ = actor.get_mean_and_std(obs["policy"], training=False)
            obs, _, terminated, truncated, _ = env.step(torch.tanh(mean))
            frame = np.asarray(u.render()).copy()
            now_goals = int(u._carry_goals_total)
            now_timeouts = int(u._carry_goal_timeout_total)
            if now_goals > prev_goals:
                events.append((len(frames), "REACHED"))
                flash = 12
            elif now_timeouts > prev_timeouts:
                events.append((len(frames), "timeout-swap"))
                flash = -12
            prev_goals, prev_timeouts = now_goals, now_timeouts
            if flash != 0:
                color = (60, 220, 120) if flash > 0 else (230, 70, 60)
                w = 14
                frame[:w, :] = color; frame[-w:, :] = color
                frame[:, :w] = color; frame[:, -w:] = color
                flash += -1 if flash > 0 else 1
            frames.append(frame)
            dropped = bool(terminated[0])
            done = bool(terminated[0] or truncated[0])
        goals = int(u._carry_goals_total) - goals_before
        swaps = int(u._carry_goal_timeout_total) - timeouts_before
        outcome = "drop" if dropped else "timeout"
        print(
            f"[demo] episode {episode}: goals={goals} timeout_swaps={swaps} end={outcome} "
            f"events={[(f, t) for f, t in events]}",
            flush=True,
        )
        if goals >= args_cli.min_goals and saved < args_cli.max_clips:
            saved += 1
            path = args_cli.video_folder / f"carry_goals{goals}_{saved}.mp4"
            imageio.mimwrite(
                path, [np.asarray(f) for f in frames], fps=args_cli.fps, macro_block_size=None
            )
            print(f"[demo] saved -> {path}", flush=True)
        if saved >= args_cli.max_clips:
            break
    print(f"[demo] saved {saved} clip(s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
