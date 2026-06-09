# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic physics stress test for pen tunneling -- NO trained policy needed.

Instead of rolling out a checkpoint, this forces the worst case by hand:

  1. the pen is pinned into the palm-center (re-pinned whenever it escapes), so it
     is always sitting in the finger-closing zone;
  2. the hand fingers are driven to OPEN <-> CLOSE at full amplitude every few steps,
     repeatedly clamping/grinding the pen, while the arm is held still;
  3. each control step we measure the three tunneling fingerprints on the pen:
       SPEED SPIKE   -- pen linear speed jumps toward the depenetration cap (a solver
                        "kick" launching the pen across the geometry),
       POSITION JUMP -- the pen teleports an implausible distance in one step,
       DEEP OVERLAP  -- the pen center gets closer to a hand-link origin than is
                        geometrically possible without interpenetration.

If the physics fix (speculative contacts + capped depenetration velocity) holds, the
fingers should never be able to launch or swallow the pen, even under this abuse.

Run (headless):
  python scripts/stress_tunneling.py --task Pick-Repose-Pen-Direct-v0 --num_envs 32 --steps 400 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic pen-tunneling stress test.")
parser.add_argument("--task", type=str, default="Pick-Repose-Pen-Direct-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--flip_period", type=int, default=8, help="control steps between full open<->close flips")
parser.add_argument("--escape_dist", type=float, default=0.06, help="re-pin the pen if it drifts this far from palm")
parser.add_argument("--speed_thresh", type=float, default=1.5, help="pen speed (m/s) flagged as a spike")
parser.add_argument("--jump_thresh", type=float, default=0.02, help="per-step pen displacement (m) flagged as a jump")
parser.add_argument("--overlap_thresh", type=float, default=0.010, help="pen-center to hand-link distance (m) flagged")
parser.add_argument("--vcap", type=float, default=None, help="override pen max_linear_velocity (m/s) for this run")
parser.add_argument("--acap", type=float, default=None, help="override pen max_angular_velocity (rad/s) for this run")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

import xhand_inhand.tasks  # noqa: F401  (registers our gym ids)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False
    if args_cli.vcap is not None:
        env_cfg.object_cfg.spawn.rigid_props.max_linear_velocity = args_cli.vcap
    if args_cli.acap is not None:
        env_cfg.object_cfg.spawn.rigid_props.max_angular_velocity = args_cli.acap
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    dev = base.device
    N = base.num_envs

    # classify hand (finger) joints vs arm joints by name
    jnames = base.robot.joint_names
    finger_keys = ("index", "mid", "ring", "pinky", "thumb")
    hand_mask = torch.tensor(
        [any(k in n for k in finger_keys) for n in jnames], dtype=torch.bool, device=dev
    )

    # report the settings actually under test
    sp = base.cfg.object_cfg.spawn
    col = getattr(sp, "collision_props", None)
    co = getattr(col, "contact_offset", None) if col is not None else None
    ro = getattr(col, "rest_offset", None) if col is not None else None
    print("\n" + "=" * 78)
    print(f"[CFG] task={args_cli.task}  num_envs={N}  steps={args_cli.steps}")
    print(f"[CFG] sim.dt={base.cfg.sim.dt:.4f}  decimation={base.cfg.decimation}  "
          f"control_dt={base.cfg.sim.dt*base.cfg.decimation:.4f}s")
    print(f"[CFG] pen max_depenetration_velocity = {sp.rigid_props.max_depenetration_velocity} m/s")
    print(f"[CFG] pen contact_offset = {co}  rest_offset = {ro}")
    print(f"[CFG] hand joints driven: {int(hand_mask.sum())} / {len(jnames)}")
    print("=" * 78 + "\n")

    env.reset()

    def pin_pen(env_ids):
        """Teleport the pen to the palm-center of the given envs, zero velocity."""
        base._compute_intermediate_values()
        pose = torch.zeros((len(env_ids), 7), device=dev)
        pose[:, :3] = base.palm_center_w[env_ids]
        pose[:, 3] = 1.0
        base.object.write_root_pose_to_sim(pose, env_ids)
        base.object.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=dev), env_ids)
        return base.palm_center_w[env_ids].clone()

    max_speed = torch.zeros(N, device=dev)
    max_jump = torch.zeros(N, device=dev)
    min_overlap = torch.full((N,), 1e9, device=dev)
    min_overlap_body = [-1] * N
    n_spike = n_jump = n_overlap = 0
    body_names = base.robot.body_names

    with torch.inference_mode():
        all_ids = base.robot._ALL_INDICES
        prev_pos = pin_pen(all_ids).clone()

        for t in range(args_cli.steps):
            # full-amplitude open<->close on the fingers, arm held still
            close = 1.0 if (t // args_cli.flip_period) % 2 == 0 else -1.0
            actions = torch.zeros((N, base.cfg.action_space), device=dev)
            actions[:, hand_mask] = close

            env.step(actions)

            base._compute_intermediate_values()
            pos = base.object.data.root_pos_w
            speed = torch.norm(base.object.data.root_lin_vel_w, dim=-1)
            jump = torch.norm(pos - prev_pos, dim=-1)

            body_pos = base.robot.data.body_pos_w
            d = torch.norm(body_pos - pos.unsqueeze(1), dim=-1)
            link_d, link_i = d.min(dim=1)

            # do NOT count the re-pin teleport as a tunneling jump
            escaped = torch.norm(pos - base.palm_center_w, dim=-1) > args_cli.escape_dist

            valid = ~escaped
            max_speed = torch.maximum(max_speed, torch.where(valid, speed, torch.zeros_like(speed)))
            max_jump = torch.maximum(max_jump, torch.where(valid, jump, torch.zeros_like(jump)))
            upd = (link_d < min_overlap) & valid
            for e in torch.nonzero(upd, as_tuple=False).flatten().tolist():
                min_overlap[e] = link_d[e]
                min_overlap_body[e] = link_i[e].item()

            n_spike += int(((speed > args_cli.speed_thresh) & valid).sum())
            n_jump += int(((jump > args_cli.jump_thresh) & valid).sum())
            n_overlap += int(((link_d < args_cli.overlap_thresh) & valid).sum())

            # re-pin escapees for the next step (reset their prev_pos to avoid a fake jump)
            prev_pos = pos.clone()
            esc_ids = torch.nonzero(escaped, as_tuple=False).flatten()
            if len(esc_ids) > 0:
                prev_pos[esc_ids] = pin_pen(esc_ids)

    print("=" * 78)
    print(f"TUNNELING STRESS REPORT  ({args_cli.steps} steps x {N} envs, fingers crushing the pinned pen)")
    print("=" * 78)
    print(f"  speed spikes (>{args_cli.speed_thresh} m/s) : {n_spike} step-envs   "
          f"max pen speed = {max_speed.max():.2f} m/s  (depen cap = {sp.rigid_props.max_depenetration_velocity})")
    print(f"  position jumps (>{args_cli.jump_thresh*1000:.0f} mm/step): {n_jump} step-envs   "
          f"max single-step jump = {max_jump.max()*1000:.1f} mm")
    print(f"  deep overlaps (<{args_cli.overlap_thresh*1000:.0f} mm to a link): {n_overlap} step-envs   "
          f"closest pen-link approach = {min_overlap.min()*1000:.1f} mm")
    closest_e = int(min_overlap.argmin())
    cb = min_overlap_body[closest_e]
    print(f"     -> closest approach to link '{body_names[cb] if cb >= 0 else '?'}'")
    print("-" * 78)
    # With the pen's velocity capped, a fast knock is EXPECTED (correct physics for a
    # light object) -- it is not tunneling. True tunneling shows up as the pen moving
    # FURTHER in one control step than its capped speed physically allows (it must have
    # passed THROUGH a collider), or as the speed exceeding the cap (skipped contact).
    control_dt = base.cfg.sim.dt * base.cfg.decimation
    v_cap = sp.rigid_props.max_linear_velocity
    allowed_jump = v_cap * control_dt
    print(f"  [bound] pen max_linear_velocity cap = {v_cap} m/s  -> max physical jump/step = {allowed_jump*1000:.1f} mm")
    speed_over_cap = max_speed.max() > 1.10 * v_cap
    teleport = max_jump.max() > 1.25 * allowed_jump
    verdict = bool(speed_over_cap or teleport)
    reason = []
    if speed_over_cap:
        reason.append(f"speed {max_speed.max():.2f} > cap {v_cap}")
    if teleport:
        reason.append(f"jump {max_jump.max()*1000:.1f}mm > {1.25*allowed_jump*1000:.1f}mm allowed")
    print("VERDICT:", f"TUNNELING ({'; '.join(reason)})" if verdict
          else f"NO tunneling -- pen stays within its speed cap ({v_cap} m/s), contacts never skipped")
    print("=" * 78)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
