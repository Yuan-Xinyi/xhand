# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive GUI viewer for inspecting a task's scene setup (robot home pose, table,
cube, goal marker) and its reset randomization.

The robot is held at its home pose (zero relative action every step), so you can orbit
the camera and check that the hand hovers over the grasp region and clears the table.
The scene is re-reset every ``--reset_every`` seconds so you can watch the arm-joint
start randomization (``reset_arm_joint_noise``) and the cube placement vary -- handy for
confirming the perturbed start poses never drive the hand into the table.

Run (GUI is on by default; do NOT pass --headless):
    conda activate env_isaaclab
    python scripts/view_task.py --task Pick-Cube-Direct-v0 --num_envs 4

Tips:
    --reset_every 0   never auto-reset (freeze on the first home pose)
    --num_envs 1      single env, easiest to inspect one robot closely
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive task-setup viewer.")
parser.add_argument("--task", type=str, default="Pick-Cube-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--reset_every",
    type=float,
    default=3.0,
    help="seconds between automatic resets (0 = never auto-reset).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# force the GUI on -- this script is useless headless
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import xhand_inhand.tasks  # noqa: F401  (registers our gym ids)  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device

    print(f"[VIEW] task = {args_cli.task},  num_envs = {args_cli.num_envs}")
    robot = unwrapped.robot
    print(f"[VIEW] arm joints  : {[robot.joint_names[i] for i in unwrapped._arm_joint_ids]}")
    print(f"[VIEW] home pose   : {robot.data.default_joint_pos[0].tolist()}")
    print(f"[VIEW] reset_arm_joint_noise = {getattr(env_cfg, 'reset_arm_joint_noise', 'n/a')} rad")
    print("[VIEW] holding home pose; close the viewer window to exit.")

    env.reset()

    # zero relative action -> the PD target stays at the (randomized) home pose, so the robot
    # holds still and we can inspect the static scene.
    zero_action = torch.zeros(env.action_space.shape, device=device)

    dt = unwrapped.step_dt  # seconds per env step (decimation * sim dt)
    steps_per_reset = int(args_cli.reset_every / dt) if args_cli.reset_every > 0 else 0

    step = 0
    while simulation_app.is_running():
        env.step(zero_action)
        step += 1
        if steps_per_reset and step % steps_per_reset == 0:
            env.reset()
            print(f"[VIEW] reset @ step {step} (new arm start + cube placement)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
