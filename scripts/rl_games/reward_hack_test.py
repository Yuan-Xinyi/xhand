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
    scenario = {
        "q_close": 0.0,
        "q_wrap": 0.0,
        "hold": 1.0,
        "thumb": False,
        "others": 0,
        "palm": 1.0,
        "force": 0.0,
    }

    def fake_signals():
        q_wrap = torch.tensor([scenario["q_wrap"]], device=dev)
        q_close = torch.tensor([scenario["q_close"]], device=dev)
        hold = torch.tensor([scenario["hold"]], device=dev)
        grasp_q = torch.minimum(q_wrap, hold)
        force = torch.zeros((1, 5), device=dev)
        force[:, 0] = scenario["force"]
        return {
            "quality": q_wrap,
            "close_quality": q_close,
            "contact_quality": q_close,
            "proximity_quality": q_close,
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
            "force_magnitude": force,
        }

    u._compute_grasp_signals = fake_signals

    def reset_logic() -> None:
        u._contact_steps.zero_()
        u._lost_contact_steps.zero_()
        u._is_grasped.zero_()
        u._grasp_bonus_given.zero_()
        u._safe_grasp_steps.zero_()
        u._success_paid.zero_()
        u._is_success.zero_()
        u._prev_close_quality.zero_()
        u._prev_wrap_quality.zero_()
        u._prev_lift_potential.zero_()
        u._potential_initialized.zero_()
        u.actions.zero_()
        u.extras["log"] = {}
        clearance.zero_()

    def reward() -> tuple[float, dict]:
        u.extras["log"] = {}
        total = float(u._get_rewards()[0])
        return total, u.extras["log"]

    def prime() -> None:
        """Initialize per-env potentials without a reset-state jackpot."""
        reward()

    # Hovering is not an annuity. A genuine state improvement pays once and reversing it pays back;
    # gamma-correct shaping also prevents discounted approach/retreat cycles from making a profit.
    reset_logic()
    scenario.update(q_close=0.15, q_wrap=0.0, hold=1.0, thumb=False, others=0, force=0.0)
    prime()
    _, log = reward()
    check(float(log["r_close_progress_mean"]) <= 0.0, "static near-handle hover has no positive annuity")
    scenario["q_close"] = 0.35
    _, log = reward()
    check(float(log["r_close_progress_mean"]) > 0.0, "thumb-stage close progress is paid")
    scenario["q_close"] = 0.60
    _, log = reward()
    check(float(log["r_close_progress_mean"]) > 0.0, "thumb+one-stage progress is paid")
    scenario["q_close"] = 1.0
    _, log = reward()
    check(float(log["r_close_progress_mean"]) > 0.0, "thumb+two-stage progress is paid")

    reset_logic()
    scenario.update(q_close=0.0, q_wrap=0.0, hold=1.0, thumb=False, others=0, force=0.0)
    prime()
    scenario["q_close"] = 1.0
    _, up_log = reward()
    scenario["q_close"] = 0.0
    _, down_log = reward()
    discounted_cycle = float(up_log["r_close_progress_mean"]) + u.cfg.shaping_discount * float(
        down_log["r_close_progress_mean"]
    )
    check(abs(discounted_cycle) < 1.0e-4, "close approach/retreat cycle has zero discounted return")

    # Legal power grasp: thumb + two opposed pads, good geometry and rigid transport.
    reset_logic()
    scenario.update(q_close=1.0, q_wrap=0.8, hold=1.0, thumb=True, others=2, palm=1.0, force=10.0)
    for _ in range(u.cfg.grasp_confirm_steps - 1):
        _, log = reward()
        check(float(log["r_grasp_mean"]) == 0.0, "grasp bonus waits for confirmation")
    _, log = reward()
    check(bool(u._is_grasped[0]), "legal wrap confirms the grasp latch")
    check(abs(float(log["r_grasp_mean"]) - u.cfg.grasp_bonus) < 1.0e-5, "grasp bonus fires once")
    _, log = reward()
    check(float(log["r_grasp_mean"]) == 0.0, "grasp bonus cannot be farmed")
    check(float(log["r_hold_mean"]) == 0.0, "stable hold has no occupancy annuity")

    # A high-force closure may satisfy topology but cannot collect the stable-grasp bonus until the
    # impact settles into the oracle-calibrated force range.
    reset_logic()
    scenario.update(q_close=1.0, q_wrap=0.8, hold=1.0, thumb=True, others=3, palm=1.0, force=300.0)
    for _ in range(u.cfg.grasp_confirm_steps):
        _, log = reward()
        check(float(log["r_grasp_mean"]) == 0.0, "crush closure cannot collect grasp bonus")
    check(bool(u._is_grasped[0]), "force cap does not redefine the grasp-state truth")
    scenario["force"] = 10.0
    for _ in range(u.cfg.grasp_confirm_steps - 1):
        _, log = reward()
        check(float(log["r_grasp_mean"]) == 0.0, "safe bonus waits for consecutive low-impact hold")
    _, log = reward()
    check(abs(float(log["r_grasp_mean"]) - u.cfg.grasp_bonus) < 1.0e-5,
          "four-step safe settled grasp receives the deferred one-shot bonus")

    # The latch survives the Schmitt dead-band, but no hold occupancy is paid there.
    scenario.update(q_close=0.6, q_wrap=0.25, hold=1.0, thumb=True, others=3, palm=1.0)
    _, log = reward()
    check(bool(u._is_grasped[0]), "dead-band quality preserves the confirmed latch")
    check(float(log["r_hold_mean"]) == 0.0, "dead-band latch cannot farm hold reward")

    # Thumb plus only one finger cannot confirm, regardless of motion agreement.
    reset_logic()
    scenario.update(q_close=0.6, q_wrap=0.0, hold=1.0, thumb=True, others=1, palm=1.0, force=10.0)
    for _ in range(u.cfg.grasp_confirm_steps + 2):
        reward()
    check(not bool(u._is_grasped[0]), "thumb+one contact is not a power grasp")

    # Back-of-hand press cannot confirm even with large raw contact forces.
    reset_logic()
    scenario.update(q_close=0.0, q_wrap=0.0, hold=1.0, thumb=True, others=4, palm=0.1, force=300.0)
    for _ in range(u.cfg.grasp_confirm_steps + 2):
        reward()
    check(not bool(u._is_grasped[0]), "back-of-hand contact is rejected")

    # Confirm, then retain raw contacts but destroy geometry/transport quality: Schmitt release must fire.
    reset_logic()
    scenario.update(q_close=1.0, q_wrap=0.8, hold=1.0, thumb=True, others=3, palm=1.0, force=10.0)
    for _ in range(u.cfg.grasp_confirm_steps):
        reward()
    scenario.update(q_close=0.0, q_wrap=0.0, hold=0.0, thumb=True, others=3, palm=1.0, force=10.0)
    for _ in range(u.cfg.grasp_release_steps - 1):
        reward()
    check(bool(u._is_grasped[0]), "Schmitt latch tolerates brief quality loss")
    _, log = reward()
    check(not bool(u._is_grasped[0]), "poor quality releases despite persistent raw contacts")
    check(float(log["r_hold_mean"]) == 0.0, "released grasp receives no hold reward")

    # Excess force saturates: 300N cannot buy more closure than 40N, and both receive the full penalty.
    reset_logic()
    scenario.update(q_close=0.0, q_wrap=0.0, hold=1.0, thumb=False, others=0, palm=1.0, force=40.0)
    prime()
    _, force40 = reward()
    scenario["force"] = 300.0
    _, force300 = reward()
    check(abs(float(force40["r_force_penalty_mean"]) + u.cfg.force_excess_penalty_scale) < 1.0e-5,
          "40N reaches the configured force penalty")
    check(abs(float(force300["r_force_penalty_mean"]) - float(force40["r_force_penalty_mean"])) < 1.0e-5,
          "300N crush has no reward advantage over 40N")

    # A legal 1cm rise is a dense +~20 event and lowering pays it back.
    reset_logic()
    scenario.update(q_close=1.0, q_wrap=0.8, hold=1.0, thumb=True, others=3, palm=1.0, force=10.0)
    for _ in range(u.cfg.grasp_confirm_steps):
        reward()
    reward()
    clearance[:] = 0.01
    _, log = reward()
    expected_lift = u.cfg.lift_progress_scale * u.cfg.shaping_discount * 0.05
    check(abs(float(log["r_lift_progress_mean"]) - expected_lift) < 1.0e-4,
          "1cm legal rise gets dense true-clearance progress")
    clearance.zero_()
    _, log = reward()
    check(abs(float(log["r_lift_progress_mean"]) + 20.0) < 1.0e-4,
          "lowering 1cm pays back lift potential")

    # Fling during latch debounce collapses the transport potential instead of rewarding height.
    clearance[:] = 0.10
    _, _ = reward()
    scenario.update(q_wrap=0.8, hold=0.0, thumb=True, others=3, palm=1.0)
    clearance[:] = 0.15
    _, log = reward()
    check(float(log["r_lift_progress_mean"]) <= 0.0, "fling/high-slip height gets no positive lift reward")

    # Terminal success payment remains one-shot.
    reset_logic()
    scenario.update(q_close=1.0, q_wrap=0.8, hold=1.0, thumb=True, others=3, palm=1.0, force=10.0)
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
