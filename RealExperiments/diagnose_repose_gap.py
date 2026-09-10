#!/usr/bin/env python3
"""One-step dynamics diagnosis: what SHOULD the cube have done vs what it DID.

For every logged real step k: load the real state (hand joints, cube pose) into
the sim, execute the SAME action through the same control contract, step the
sim once, and compare the sim-predicted cube motion against the real cube
motion at step k+1. The per-step mismatch is the sim2real dynamics gap, in the
cube's own terms.

Run:
    conda activate env_isaaclab
    python RealExperiments/diagnose_repose_gap.py --npz /tmp/repose_run3.npz
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--npz", default="/tmp/repose_run3.npz")
parser.add_argument("--task", default="Xhand-Repose-Cube-OpenAI-LSTM-Hard-Direct-v0")
parser.add_argument("--max-steps", type=int, default=400, help="cap on replayed transitions")
parser.add_argument("--out", default=None, help="save per-step results npz")
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


def quat_delta_deg(q_from: np.ndarray, q_to: np.ndarray) -> float:
    return float(np.degrees(rr.rotation_distance(q_to, q_from)))


def rot_axis(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    qd = rr.quat_mul(np.asarray(q_to, float), rr.quat_conj(np.asarray(q_from, float)))
    v = qd[1:4]
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)


def main():
    d = np.load(args_cli.npz)
    n = len(d["action"])
    t = d["t"]
    if "hand_q" in d.files:
        hand_q = d["hand_q"]
        targets = d["targets"]
    else:
        # legacy logs: reconstruct commanded targets (open-loop assumption)
        print("[diag] legacy log without hand_q/targets — reconstructing from actions")
        targets = np.zeros((n, 12), dtype=np.float32)
        prev = np.zeros(12, dtype=np.float32)
        for i in range(n):
            tg = rr.ACT_MOVING_AVERAGE * rr.scale_action(d["action"][i]) + (1 - rr.ACT_MOVING_AVERAGE) * prev
            tg = np.clip(tg, rr.LOWER, rr.UPPER)
            tg = prev + np.clip(tg - prev, -0.157, 0.157)
            prev = np.clip(tg, rr.LOWER, rr.UPPER).astype(np.float32)
            targets[i] = prev
        hand_q = targets

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.observation_noise_model = None
    env_cfg.action_noise_model = None
    env_cfg.events = None  # no DR: nominal physics for the prediction
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    origins = u.scene.env_origins[0]
    dev = u.device

    dt_real = np.diff(t)
    rows = []
    replayed = 0
    for k in range(1, n - 1):
        if replayed >= args_cli.max_steps:
            break
        # skip stall-reset boundaries / stale gaps
        if dt_real[k - 1] > 0.15 or dt_real[k] > 0.15:
            continue

        q = torch.tensor(hand_q[k], device=dev).unsqueeze(0)
        qd = torch.tensor((hand_q[k] - hand_q[k - 1]) / dt_real[k - 1], device=dev).unsqueeze(0)
        u.hand.write_joint_state_to_sim(q, qd)
        u.prev_targets[:] = torch.tensor(targets[k - 1], device=dev)
        u.hand.set_joint_position_target(u.prev_targets)

        root = torch.zeros((1, 13), device=dev)
        root[0, :3] = torch.tensor(d["obj_pos"][k], device=dev) + origins
        root[0, 3:7] = torch.tensor(d["obj_quat"][k], device=dev)
        root[0, 7:10] = torch.tensor((d["obj_pos"][k] - d["obj_pos"][k - 1]) / dt_real[k - 1], device=dev)
        w_axis = rot_axis(d["obj_quat"][k - 1], d["obj_quat"][k])
        w_mag = np.radians(quat_delta_deg(d["obj_quat"][k - 1], d["obj_quat"][k])) / dt_real[k - 1]
        root[0, 10:13] = torch.tensor(w_axis * w_mag, device=dev)
        u.object.write_root_pose_to_sim(root[:, :7])
        u.object.write_root_velocity_to_sim(root[:, 7:])

        env.step(torch.tensor(d["action"][k], device=dev).unsqueeze(0))

        u._compute_intermediate_values()
        pred_quat = u.object_rot[0].cpu().numpy()
        pred_pos = u.object_pos[0].cpu().numpy()

        real_next_q = d["obj_quat"][k + 1]
        rows.append({
            "pred_rot": quat_delta_deg(d["obj_quat"][k], pred_quat),
            "real_rot": quat_delta_deg(d["obj_quat"][k], real_next_q),
            "pred_vs_real": quat_delta_deg(pred_quat, real_next_q),
            "axis_dot": float(np.dot(rot_axis(d["obj_quat"][k], pred_quat),
                                     rot_axis(d["obj_quat"][k], real_next_q))),
            "pos_err_mm": float(np.linalg.norm(pred_pos - d["obj_pos"][k + 1]) * 1000),
        })
        replayed += 1

    pr = np.array([r["pred_rot"] for r in rows])
    rr_ = np.array([r["real_rot"] for r in rows])
    pv = np.array([r["pred_vs_real"] for r in rows])
    ax = np.array([r["axis_dot"] for r in rows])
    pe = np.array([r["pos_err_mm"] for r in rows])
    moving = pr > 0.5  # steps where sim expects visible motion

    print("\n================= ONE-STEP DYNAMICS GAP =================")
    print(f"transitions replayed        : {len(rows)}")
    print(f"per-step rotation, SIM pred : median {np.median(pr):5.2f} deg  p90 {np.percentile(pr, 90):5.2f}")
    print(f"per-step rotation, REAL     : median {np.median(rr_):5.2f} deg  p90 {np.percentile(rr_, 90):5.2f}")
    print(f"real/pred motion ratio      : {np.median(rr_[moving] / np.maximum(pr[moving], 1e-6)) if moving.any() else float('nan'):.2f}"
          "   (<1 = real cube moves LESS than sim expects)")
    print(f"pred-vs-real mismatch       : median {np.median(pv):5.2f} deg  p90 {np.percentile(pv, 90):5.2f}")
    print(f"rotation-axis alignment     : median {np.median(ax[moving]) if moving.any() else float('nan'):.2f}"
          "   (1 = same direction, <0 = opposite)")
    print(f"position pred error         : median {np.median(pe):5.1f} mm")
    print("=========================================================\n", flush=True)

    if args_cli.out:
        np.savez(args_cli.out, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
        print(f"[diag] per-step rows -> {args_cli.out}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
