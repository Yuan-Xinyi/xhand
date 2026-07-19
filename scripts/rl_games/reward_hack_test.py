# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the pick_tool_token reward-hack fixes + contact-gated lift (static-review
audit + paper R_contact). Injects reward state (object lift/orientation, contact-grasp state, lift
ratchets) and calls `_get_rewards()` -- which does not recompute intermediates -- to verify the
hacks are blocked and a genuine contact-grasp+lift still pays. The contact grasp is controlled by
monkeypatching `_finger_contact_state` (so no physics contact is needed for the logic test)."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    dev = u.device
    rest_quat = torch.tensor(u.cfg.object_cfg.init_state.rot, dtype=torch.float, device=dev).unsqueeze(0)
    print(f"\ncfg: contact_force_thr={u.cfg.contact_force_thr} grasp_bonus={u.cfg.grasp_bonus} "
          f"confirm/release={u.cfg.grasp_confirm_steps}/{u.cfg.grasp_release_steps} "
          f"lift_step_max={u.cfg.lift_step_max} lift_success_bonus={u.cfg.lift_success_bonus} "
          f"lift_success={u.cfg.lift_success_height}  (reward: reach + grasp_event + lift_HEIGHT + lift_success)")

    def run(lift, thumb_contact, other_count, baseline, grasped, bonus_given=True, transition=False,
            is_success=False, success_paid=False):
        # `grasped` forces the hysteresis latch to is_grasped; `transition=True` simulates a first-grasp
        # transition (was not grasped, contact now confirms). is_success/success_paid inject the strict
        # success latch for the one-shot success-bonus tests.
        u._finger_contact_state = lambda: (
            torch.tensor([thumb_contact], dtype=torch.bool, device=dev),
            torch.tensor([other_count], dtype=torch.long, device=dev),
        )
        opw = u.object_pos_w.clone()
        opw[:, 2] = u.scene.env_origins[:, 2] + u.object_default_z + lift
        u.object_pos_w = opw
        u.object_quat_w = rest_quat.clone()  # rest orientation -> clearance == lift
        u._grasp_baseline_lift[:] = baseline
        u._grasp_bonus_given[:] = bonus_given
        u._is_success[:] = is_success
        u._success_paid[:] = success_paid
        if transition:
            u._is_grasped[:] = False
            u._contact_steps[:] = u.cfg.grasp_confirm_steps + 5
            u._lost_contact_steps[:] = 0
        else:
            u._is_grasped[:] = grasped
            u._contact_steps[:] = (u.cfg.grasp_confirm_steps + 5) if grasped else 0
            u._lost_contact_steps[:] = 0 if grasped else (u.cfg.grasp_release_steps + 5)
        u.extras["log"] = {}
        total = u._get_rewards()[0].item()
        log = u.extras["log"]
        return dict(g=log["is_grasped_frac"].item(), clr=log["clearance_ok_frac"].item(),
                    height=log["r_lift_height_mean"].item(), succ=log["r_lift_success_mean"].item(),
                    rg=log["r_grasp_mean"].item(), total=total, base=u._grasp_baseline_lift[0].item())

    lsm, H = u.cfg.lift_step_max, u.cfg.lift_success_height
    print("\n========== REWARD-HACK REGRESSION (occupancy-height lift, no ratchet) ==========")

    # T0 height: grasped at 11cm -> per-step height = lift_step_max * 0.11/0.20
    r = run(lift=0.11, thumb_contact=True, other_count=2, baseline=0.0, grasped=True)
    exp = lsm * (0.11 / H)
    print(f"[T0 height  ] grasp at 11cm: height={r['height']:.1f} (expect {exp:.1f})  "
          f"-> {'PASS (pays per height)' if abs(r['height'] - exp) < 0.1 else 'FAIL'}")

    # T1 P0-1: knock to 25cm THEN first-grasp -> baseline locks at 25cm -> rel-lift 0 -> height 0
    r = run(lift=0.25, thumb_contact=True, other_count=2, baseline=1e6, grasped=False, bonus_given=False, transition=True)
    ok = r['height'] == 0 and abs(r['base'] - 0.25) < 1e-4
    print(f"[T1 P0-1    ] knock->25cm then grasp: base={r['base']:.2f} height={r['height']:.1f}  "
          f"-> {'FIXED (baseline locks, no retroactive credit)' if ok else 'STILL HACKABLE'}")

    # T2a: thumb only -> not grasped -> no height reward
    r = run(lift=0.15, thumb_contact=True, other_count=0, baseline=0.0, grasped=False)
    print(f"[T2a grasp  ] thumb only: g={r['g']:.0f} height={r['height']:.1f}  "
          f"-> {'PASS (needs thumb + >=1 other)' if r['g'] == 0 and r['height'] == 0 else 'FAIL'}")

    # T2b: FIRST stable grasp -> one-shot R_grasp
    r = run(lift=0.0, thumb_contact=True, other_count=1, baseline=1e6, grasped=False, bonus_given=False, transition=True)
    ok = r['g'] == 1 and abs(r['rg'] - u.cfg.grasp_bonus) < 1e-3
    print(f"[T2b grasp  ] first stable grasp: g={r['g']:.0f} r_grasp={r['rg']:.0f}  "
          f"-> {'PASS (one-shot grasp bonus)' if ok else 'FAIL'}")

    # T3 latch: already-given grasp bonus -> regrasp pays nothing
    r = run(lift=0.0, thumb_contact=True, other_count=1, baseline=1e6, grasped=True, bonus_given=True)
    print(f"[T3 latch   ] regrasp after bonus given: r_grasp={r['rg']:.0f}  "
          f"-> {'PASS (one-shot)' if r['rg'] == 0 else 'FAIL'}")

    # T-nocliff: THE fix. Grasped at 5cm -- occupancy pays for the CURRENT height (5.0) regardless of any
    # earlier bounce peak. The OLD ratchet would pay 0 here (below a prior 10cm peak) -> the 5cm cliff.
    r = run(lift=0.05, thumb_contact=True, other_count=2, baseline=0.0, grasped=True)
    exp = lsm * (0.05 / H)
    print(f"[T-nocliff  ] relift to 5cm (post any peak): height={r['height']:.1f} (expect {exp:.1f}, NOT 0)  "
          f"-> {'PASS (no ratchet cliff)' if abs(r['height'] - exp) < 0.1 else 'FAIL'}")

    # T4: dropped, no contact -> no height reward
    r = run(lift=-0.15, thumb_contact=False, other_count=0, baseline=1e6, grasped=False)
    print(f"[T4 drop    ] object DROPPED: g={r['g']:.0f} height={r['height']:.1f}  "
          f"-> {'OK (no lift credit)' if r['height'] == 0 else 'LEAK'}")

    # T5: grasp 1cm off table (below clearance). height pays (occupancy, NOT clearance-gated) but success
    # bonus needs the strict success (which requires clearance + 20cm + hold) -> 0 here.
    r = run(lift=0.01, thumb_contact=True, other_count=2, baseline=0.0, grasped=True)
    print(f"[T5 height  ] grasp 1cm off table: height={r['height']:.1f}(>0 by design) succ={r['succ']:.0f}  "
          f"-> {'OK' if r['height'] > 0 and r['succ'] == 0 else 'FAIL'}")

    # T7 success bonus: strict success just latched (is_success True, not yet paid) -> one-shot lift_success_bonus
    r = run(lift=0.20, thumb_contact=True, other_count=2, baseline=0.0, grasped=True, is_success=True, success_paid=False)
    ok = abs(r['succ'] - u.cfg.lift_success_bonus) < 1e-3
    print(f"[T7 success ] first stable success: succ={r['succ']:.0f} (expect {u.cfg.lift_success_bonus:.0f})  "
          f"-> {'PASS (one-shot success bonus)' if ok else 'FAIL'}")

    # T8 success latch: already paid -> 0 (not re-farmable)
    r = run(lift=0.20, thumb_contact=True, other_count=2, baseline=0.0, grasped=True, is_success=True, success_paid=True)
    print(f"[T8 succ latch] success already paid: succ={r['succ']:.0f}  "
          f"-> {'PASS (one-shot)' if r['succ'] == 0 else 'FAIL'}")

    print("=======================================================================\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
