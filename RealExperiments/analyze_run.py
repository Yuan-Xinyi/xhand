#!/usr/bin/env python3
"""One-shot post-run report: progress, stalls, per-joint blocking, releases.

    /home/lqin/miniconda3/envs/one/bin/python RealExperiments/analyze_run.py          # latest run
    ... analyze_run.py RealExperiments/runlogs/run_20260911_110520.npz               # a specific one
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foundationpose_repose_real as rr

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runlogs", "latest.npz")
d = np.load(path)
obs, act, oq, gq, t = d["obs"], d["action"], d["obj_quat"], d["goal_quat"], d["t"]
tg, hq = d.get("targets"), d.get("hand_q")
n = len(act)
rd = np.degrees([rr.rotation_distance(oq[i], gq[i]) for i in range(n)])
goal_changes = np.where(np.abs(np.diff(gq, axis=0)).sum(axis=1) > 1e-6)[0]

print(f"=== {os.path.basename(path)}: {n} steps, {t[-1] - t[0]:.0f} s, dt {np.diff(t).mean() * 1000:.1f} ms ===")
print(f"rot_dist: start {rd[0]:.0f} deg  best {rd.min():.0f} deg  end {rd[-1]:.0f} deg   (success < 23)")
print(f"goals attempted: {len(goal_changes) + 1}   successes: {len(goal_changes)}")

# progress segments: where did it move, where did it stall?
w = 40  # 2 s window
prog = np.array([rd[max(0, i - w)] - rd[i] for i in range(n)])
stalled = np.abs(prog) < 3.0
print(f"time making progress: {100 * (prog > 3).mean():.0f}%   stalled (<3 deg / 2 s): {100 * stalled.mean():.0f}%")
gross = np.degrees(sum(rr.rotation_distance(oq[i], oq[i + 1]) for i in range(n - 1)))
print(f"cube rotation: gross {gross:.0f} deg, net progress {rd[0] - rd.min():.0f} deg"
      f"  -> efficiency {100 * (rd[0] - rd.min()) / max(gross, 1e-9):.1f}%")

if tg is not None and hq is not None:
    lead = tg - hq
    print("\nper-joint (whole run):   lead = commanded minus measured")
    print(f"{'joint':16s} {'lead med':>9s} {'pinned%':>8s} {'travel/s':>9s}  state")
    for j, nm in enumerate(rr.ISAAC12):
        pinned = 100 * (lead[:, j] >= 0.057).mean()
        trav = np.median([hq[i:i + 20, j].max() - hq[i:i + 20, j].min() for i in range(0, n - 20, 5)])
        state = "BLOCKED" if (pinned > 30 and trav < 0.02) else ("light" if trav < 0.05 else "active")
        print(f"{nm:16s} {np.median(lead[:, j]):+9.3f} {pinned:7.0f}% {trav:8.3f}r  {state}")

    fingers = {"thumb": [], "index": [], "middle": [], "ring": [], "pinky": []}
    for j, nm in enumerate(rr.ISAAC12):
        fingers[nm.split("_")[0]].append(j)
    print("\nper-finger motion (median travel per second):")
    for f, js in fingers.items():
        trav = np.median([hq[i:i + 20, js].max(axis=0).mean() - hq[i:i + 20, js].min(axis=0).mean()
                          for i in range(0, n - 20, 5)])
        print(f"  {f:7s} {trav:.3f} rad/s")

print(f"\naction: |a|>0.95 on {100 * (np.abs(act) > 0.95).mean():.0f}% of joint-steps,"
      f" mean step change {np.abs(np.diff(act, axis=0)).mean():.3f}")
log = path.replace(".npz", ".log")
if os.path.exists(log):
    rel = [ln for ln in open(log) if "[release]" in ln]
    print(f"releases logged: {len(rel)}" + (f"   e.g. {rel[-1].strip()}" if rel else ""))
