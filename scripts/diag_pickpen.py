# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose pick-pen reach geometry: palm-center point vs fingertips vs pen."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg("Pick-Pen-Direct-v0", device=args_cli.device, num_envs=2)
    env = gym.make("Pick-Pen-Direct-v0", cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    act = torch.zeros((2, base.cfg.action_space), device=base.device)
    for _ in range(40):
        env.step(act)
    base._compute_intermediate_values()
    palm = base.robot.data.body_pos_w[0, base.palm_idx].cpu().numpy()
    pc = base.palm_center_w[0].cpu().numpy()
    tips = base.ee_pos_w[0].cpu().numpy()  # (5,3)
    tipmean = tips.mean(axis=0)
    print(f"[DIAG] palm origin (wrist) = {palm}")
    print(f"[DIAG] palm_center point   = {pc}")
    print(f"[DIAG] fingertip mean      = {tipmean}")
    print(f"[DIAG] |palm_center - wrist|     = {((pc-palm)**2).sum()**0.5:.3f} m")
    print(f"[DIAG] |palm_center - tip_mean|  = {((pc-tipmean)**2).sum()**0.5:.3f} m  (should be SMALL: palm-center near fingers)")
    print(f"[DIAG] |wrist - tip_mean|        = {((palm-tipmean)**2).sum()**0.5:.3f} m")
    print("[DIAG] -> palm_center should sit between the wrist and the fingertip cluster (closer to fingers).")
    simulation_app.close()


if __name__ == "__main__":
    main()
