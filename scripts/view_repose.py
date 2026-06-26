# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive GUI viewer for the XHand in-hand repose task.

Unlike ``view_task.py`` (pick_cube: an arm + hand over a table), this viewer is for the
AllegroHand-style ``InHandManipulationEnv`` family (e.g. ``Repose-Cube-XHand-Direct-v0``),
where the wrist is pinned in the air and a cube rests in the fingers.

The hand is held at its reset (open) pose every step, so the cube settles under pure
gravity + finger contact -- handy for confirming the wrist mount pose and the cube's
initial rest position before training. The cube's height drift relative to the in-hand
reference point is printed every second so you can read the settling numerically too.

Run (GUI on by default; do NOT pass --headless):
    conda activate env_isaaclab
    python scripts/view_repose.py --task Repose-Cube-XHand-Direct-v0 --num_envs 1 --reset_every 0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive in-hand repose viewer.")
parser.add_argument("--task", type=str, default="Repose-Cube-XHand-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--reset_every",
    type=float,
    default=0.0,
    help="seconds between automatic resets (0 = never auto-reset, freeze on first placement).",
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
    print(f"[VIEW] hand joints  : {unwrapped.hand.joint_names}")
    print(f"[VIEW] wrist pose   : pos={env_cfg.robot_cfg.init_state.pos}  rot={env_cfg.robot_cfg.init_state.rot}")
    print(f"[VIEW] cube init pos: {env_cfg.object_cfg.init_state.pos}")
    print("[VIEW] holding the open reset pose; close the viewer window to exit.")

    env.reset()

    # The in-hand env maps action a in [-1,1] -> scale(a) joint targets. To HOLD the reset
    # (open) pose we feed the action that scales back to the default joint positions:
    #   a = unscale(default_joint_pos) = (2*q - upper - lower) / (upper - lower)
    idx = unwrapped.actuated_dof_indices
    lower = unwrapped.hand_dof_lower_limits[:, idx]
    upper = unwrapped.hand_dof_upper_limits[:, idx]
    q_default = unwrapped.hand.data.default_joint_pos[:, idx]
    hold_action = ((2.0 * q_default - upper - lower) / (upper - lower)).clamp(-1.0, 1.0)

    dt = unwrapped.step_dt  # seconds per env step (decimation * sim dt)
    steps_per_reset = int(args_cli.reset_every / dt) if args_cli.reset_every > 0 else 0
    log_every = max(1, int(1.0 / dt))  # ~1 s

    step = 0
    while simulation_app.is_running():
        env.step(hold_action)
        step += 1
        if step % log_every == 0:
            # height of the cube relative to the in-hand reference point (env-local frame)
            drop = (unwrapped.object_pos[0, 2] - unwrapped.in_hand_pos[0, 2]).item()
            dist = torch.norm(unwrapped.object_pos[0] - unwrapped.in_hand_pos[0]).item()
            held = "HELD " if dist < env_cfg.fall_dist else "FALL!"
            print(f"[VIEW] t={step * dt:5.1f}s  {held}  cube z-offset={drop:+.3f} m  dist={dist:.3f} m")
        if steps_per_reset and step % steps_per_reset == 0:
            env.reset()
            print(f"[VIEW] reset @ step {step} (new cube placement + goal orientation)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
