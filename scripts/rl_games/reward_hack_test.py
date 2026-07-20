# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Logic-level regression for the stage-1 robust grasp/lift reward gates."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    dev = u.device

    # Make the reach upper bound exact and replace physics-dependent signals with a truth table.
    u._curr_fingertip_distances[:] = 0.0
    u._finger_align[:] = 1.0
    clearance = torch.zeros(1, device=dev)
    u._object_true_min_z = lambda: u._table_surface_z + clearance
    scenario = {"q_wrap": 0.0, "hold": 1.0, "thumb": False, "others": 0, "palm": 1.0}

    def fake_signals():
        q_wrap = torch.tensor([scenario["q_wrap"]], device=dev)
        hold = torch.tensor([scenario["hold"]], device=dev)
        grasp_q = torch.minimum(q_wrap, hold)
        return {
            "quality": q_wrap,
            "grasp_quality": grasp_q,
            "hold_quality": hold,
            "slip_lin": torch.tensor([0.0 if scenario["hold"] == 1.0 else 3.0], device=dev),
            "slip_ang": torch.zeros(1, device=dev),
            "thumb_contact": torch.tensor([scenario["thumb"]], dtype=torch.bool, device=dev),
            "other_contact_count": torch.tensor([scenario["others"]], dtype=torch.long, device=dev),
            "palm_facing": torch.tensor([scenario["palm"]], device=dev),
            "palm_score": torch.tensor([scenario["palm"]], device=dev),
            "alignment_score": q_wrap.clone(),
            "opposition_raw": q_wrap.clone(),
            "force_magnitude": torch.zeros((1, 5), device=dev),
        }

    u._compute_grasp_signals = fake_signals

    def reset_logic() -> None:
        u._contact_steps.zero_()
        u._lost_contact_steps.zero_()
        u._is_grasped.zero_()
        u._grasp_bonus_given.zero_()
        u._success_paid.zero_()
        u._is_success.zero_()
        u.extras["log"] = {}
        clearance.zero_()

    def reward() -> tuple[float, dict]:
        u.extras["log"] = {}
        total = float(u._get_rewards()[0])
        return total, u.extras["log"]

    # Legal power grasp: thumb + two opposed pads, good geometry and rigid transport.
    reset_logic()
    scenario.update(q_wrap=0.8, hold=1.0, thumb=True, others=2, palm=1.0)
    for _ in range(u.cfg.grasp_confirm_steps - 1):
        _, log = reward()
        check(float(log["r_grasp_mean"]) == 0.0, "grasp bonus waits for confirmation")
    _, log = reward()
    check(bool(u._is_grasped[0]), "legal wrap confirms the grasp latch")
    check(abs(float(log["r_grasp_mean"]) - u.cfg.grasp_bonus) < 1.0e-5, "grasp bonus fires once")
    check(abs(float(log["r_hold_mean"]) - u.cfg.grasp_hold_scale) < 1.0e-5, "confirmed grasp earns hold floor")
    _, log = reward()
    check(float(log["r_grasp_mean"]) == 0.0, "grasp bonus cannot be farmed")

    # The latch survives the Schmitt dead-band, but weak quality must not retain the full hold annuity.
    scenario.update(q_wrap=0.30, hold=1.0, thumb=True, others=3, palm=1.0)
    _, log = reward()
    expected_strength = (0.30 - u.cfg.grasp_quality_low) / (
        u.cfg.grasp_quality_high - u.cfg.grasp_quality_low
    )
    check(bool(u._is_grasped[0]), "dead-band quality preserves the confirmed latch")
    check(
        abs(float(log["r_hold_mean"]) - u.cfg.grasp_hold_scale * expected_strength) < 1.0e-5,
        "hold reward fades inside the Schmitt dead-band",
    )

    # Thumb plus only one finger cannot confirm, regardless of motion agreement.
    reset_logic()
    scenario.update(q_wrap=0.0, hold=1.0, thumb=True, others=1, palm=1.0)
    for _ in range(u.cfg.grasp_confirm_steps + 2):
        reward()
    check(not bool(u._is_grasped[0]), "thumb+one contact is not a power grasp")

    # Back-of-hand press cannot confirm even with large raw contact forces.
    reset_logic()
    scenario.update(q_wrap=0.0, hold=1.0, thumb=True, others=4, palm=0.1)
    for _ in range(u.cfg.grasp_confirm_steps + 2):
        reward()
    check(not bool(u._is_grasped[0]), "back-of-hand contact is rejected")

    # Confirm, then retain raw contacts but destroy geometry/transport quality: Schmitt release must fire.
    reset_logic()
    scenario.update(q_wrap=0.8, hold=1.0, thumb=True, others=3, palm=1.0)
    for _ in range(u.cfg.grasp_confirm_steps):
        reward()
    scenario.update(q_wrap=0.0, hold=0.0, thumb=True, others=3, palm=1.0)
    for _ in range(u.cfg.grasp_release_steps - 1):
        reward()
    check(bool(u._is_grasped[0]), "Schmitt latch tolerates brief quality loss")
    _, log = reward()
    check(not bool(u._is_grasped[0]), "poor quality releases despite persistent raw contacts")
    check(float(log["r_hold_mean"]) == 0.0, "released grasp receives no hold reward")

    # Legal static grasp must dominate the best possible hover reward.
    reset_logic()
    scenario.update(q_wrap=0.8, hold=1.0, thumb=True, others=3, palm=1.0)
    for _ in range(u.cfg.grasp_confirm_steps):
        reward()
    _, held_log = reward()
    held_floor = float(held_log["r_hold_mean"])
    reset_logic()
    scenario.update(q_wrap=0.0, hold=1.0, thumb=False, others=0, palm=1.0)
    _, hover_log = reward()
    check(held_floor > float(hover_log["r_reach_mean"]), "legal hold is worth more than max hover")

    # Height pays only while the same robust grasp is present; fling quality zeros it.
    reset_logic()
    clearance[:] = 0.10
    scenario.update(q_wrap=0.8, hold=1.0, thumb=True, others=3, palm=1.0)
    for _ in range(u.cfg.grasp_confirm_steps):
        reward()
    _, log = reward()
    expected_lift = u.cfg.lift_scale * 0.5 * 0.8
    check(abs(float(log["r_lift_mean"]) - expected_lift) < 1.0e-5, "10cm legal transport gets gated lift reward")
    scenario.update(q_wrap=0.8, hold=0.0, thumb=True, others=3, palm=1.0)
    _, log = reward()
    check(float(log["r_lift_mean"]) == 0.0, "fling/high-slip transport gets zero lift reward")

    # Terminal success payment remains one-shot.
    reset_logic()
    scenario.update(q_wrap=0.8, hold=1.0, thumb=True, others=3, palm=1.0)
    u._is_success[:] = True
    _, log = reward()
    check(abs(float(log["r_success_mean"]) - u.cfg.success_bonus) < 1.0e-5, "success bonus fires once")
    _, log = reward()
    check(float(log["r_success_mean"]) == 0.0, "success bonus cannot be farmed")

    print("ALL REWARD-GATE TESTS PASSED")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
