#!/usr/bin/env python3
"""Retarget a MANUS Core ergonomics CSV export onto the XHand's 12 actuated joints.

MANUS gives 20 channels per hand -- 5 fingers x (MCPSpread, MCPStretch,
PIPStretch, DIPStretch).  The XHand has 12 joints and no spread on
middle/ring/pinky, so this is a 20 -> 12 projection:

  * per-finger PIP and DIP are blended into the XHand's single distal joint
  * middle/ring/pinky spread is dropped (no such DOF)
  * thumb keeps all three DOF

Usage:
    python3 manus_csv_to_xhand.py --csv take.csv --out traj.npz --plot traj.png
    python3 manus_csv_to_xhand.py --csv take.csv --inspect     # just dump the header

The output .npz carries `t`, `q` (N,12) in radians and `joint_names`, which is
the format the replay tooling expects.
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------- XHand side
JOINT_NAMES = [
    "thumb_joint0", "thumb_joint1", "thumb_joint2",
    "index_joint0", "index_joint1", "index_joint2",
    "middle_joint0", "middle_joint1",
    "ring_joint0", "ring_joint1",
    "pinky_joint0", "pinky_joint1",
]
# from xarm7_xhand.urdf
JOINT_LIMITS = np.array([
    [0.000, 1.830], [-1.050, 1.570], [-0.175, 1.830],
    [-0.175, 0.175], [0.000, 1.920], [0.000, 1.920],
    [0.000, 1.920], [0.000, 1.920],
    [0.000, 1.920], [0.000, 1.920],
    [0.000, 1.920], [0.000, 1.920],
])

# ---------------------------------------------------------------- MANUS side
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
CHANNELS = ["mcpspread", "mcpstretch", "pipstretch", "dipstretch"]

# Human joint travel in degrees, used to normalise before mapping onto the XHand
# limits.  MEASURED with manus_calibrate.py, not taken from anatomy tables: the
# nominal figures were up to 5x too wide on the distal and spread channels
# (dipstretch really moves 10-37 deg, not 80), so a full fist normalised to about
# a third of travel and the robot never closed.  Re-run the calibration and pass
# --calib for a different operator.
HUMAN_RANGE_DEG = {
    ("thumb", "mcpspread"): (-8.0, 36.0),
    ("thumb", "mcpstretch"): (19.8, 48.7),
    ("thumb", "pipstretch"): (0.0, 70.3),
    ("thumb", "dipstretch"): (-28.4, 33.7),
    ("index", "mcpspread"): (-7.6, 10.1),
    ("index", "mcpstretch"): (-12.1, 54.1),
    ("index", "pipstretch"): (0.0, 122.9),
    ("index", "dipstretch"): (-5.3, 11.5),
    ("middle", "mcpspread"): (-6.9, 2.5),
    ("middle", "mcpstretch"): (-8.6, 57.0),
    ("middle", "pipstretch"): (0.0, 118.0),
    ("middle", "dipstretch"): (-4.1, 18.1),
    ("ring", "mcpspread"): (-17.9, 3.7),
    ("ring", "mcpstretch"): (-11.5, 55.3),
    ("ring", "pipstretch"): (0.0, 128.2),
    ("ring", "dipstretch"): (-5.5, 4.0),
    ("pinky", "mcpspread"): (-34.0, 3.8),
    ("pinky", "mcpstretch"): (-7.5, 47.2),
    ("pinky", "pipstretch"): (0.0, 105.5),
    ("pinky", "dipstretch"): (-11.5, 25.2),
}

# XHand joint <- weighted sum of normalised MANUS channels.
# Weights inside one entry must sum to 1 so the result stays in [0,1].
# The two thumb base channels are NOT what their names suggest.  Measured across
# the calibration poses:
#
#             thumb_out  thumb_across  thumb_up     what it really tracks
#   mcpspread     -7.3         +12.5     +36.0      lift out of the palm plane
#   mcpstretch    +43.1         +19.8     +48.7      sweep across the palm plane
#
# So mcpstretch -- not mcpspread -- is the opposition axis that thumb_joint0
# turns about, and mcpspread drives thumb_joint1's lift.  Wiring them by name
# crossed the two, which no amount of sign flipping could fix.
#
# Distal joints take pip at 0.8: dipstretch measures only 10-37 deg of travel
# against pip's 105-128, so an even blend mostly amplified its noise.
RETARGET = {
    "thumb_joint0":  [(("thumb", "mcpstretch"), 1.0)],
    "thumb_joint1":  [(("thumb", "mcpspread"), 1.0)],
    "thumb_joint2":  [(("thumb", "pipstretch"), 0.8), (("thumb", "dipstretch"), 0.2)],
    "index_joint0":  [(("index", "mcpspread"), 1.0)],
    "index_joint1":  [(("index", "mcpstretch"), 1.0)],
    "index_joint2":  [(("index", "pipstretch"), 0.8), (("index", "dipstretch"), 0.2)],
    "middle_joint0": [(("middle", "mcpstretch"), 1.0)],
    "middle_joint1": [(("middle", "pipstretch"), 0.8), (("middle", "dipstretch"), 0.2)],
    "ring_joint0":   [(("ring", "mcpstretch"), 1.0)],
    "ring_joint1":   [(("ring", "pipstretch"), 0.8), (("ring", "dipstretch"), 0.2)],
    "pinky_joint0":  [(("pinky", "mcpstretch"), 1.0)],
    "pinky_joint1":  [(("pinky", "pipstretch"), 0.8), (("pinky", "dipstretch"), 0.2)],
}

# Joints whose URDF positive direction opposes the channel now driving them.
# thumb_joint0 rising sweeps the tip from x=+0.105 (splayed outboard of the index
# at +0.026) in to x=+0.044, i.e. toward the fingers, while mcpstretch is HIGH
# when the thumb is splayed out and LOW across the palm -- hence the flip.
# thumb_joint1 rising lifts the tip +53 mm clear of the palm and mcpspread rises
# on exactly that motion, so it needs no flip.
INVERT = {"thumb_joint0"}


def norm_key(s):
    """Collapse a CSV header cell to comparable tokens."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_columns(header, hand):
    """Map (finger, channel) -> column index by fuzzy-matching the header.

    Exporting "one file per hand" drops the Left/Right token from the column
    names, so when neither token appears anywhere we match on finger+channel
    alone and trust the caller picked the right file.
    """
    want_hand = hand.lower()
    other_hand = "left" if want_hand == "right" else "right"
    keys = [norm_key(c) for c in header]
    hand_tagged = any(("left" in k or "right" in k) for k in keys)

    found = {}
    for idx, k in enumerate(keys):
        if hand_tagged and (other_hand in k or want_hand not in k):
            continue
        for finger in FINGERS:
            if finger not in k:
                continue
            for ch in CHANNELS:
                if ch in k:  # 'mcp_spread' already collapsed to 'mcpspread'
                    found.setdefault((finger, ch), idx)
                    break
            break
    if not hand_tagged and found:
        print(f"[match] header carries no Left/Right token -- assuming this file "
              f"is the {want_hand} hand")
    return found


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(8192)  # Manus lets you pick the separator
        fh.seek(0)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            delim = ","
        rows = list(csv.reader(fh, delimiter=delim))
    if not rows:
        raise SystemExit(f"{path} is empty")

    # the header is the first row that is mostly non-numeric
    hdr_i = 0
    for i, row in enumerate(rows[:5]):
        numeric = sum(1 for c in row if re.fullmatch(r"[-+]?[\d.eE+]+", c.strip() or "x"))
        if numeric < max(1, len(row) // 2):
            hdr_i = i
            break
    return rows[hdr_i], rows[hdr_i + 1:], delim


def to_float(cell):
    c = cell.strip().replace(",", ".")  # tolerate comma decimal separator
    if not c:
        return np.nan
    try:
        return float(c)
    except ValueError:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="MANUS Core CSV export")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--out", default="manus_traj.npz")
    ap.add_argument("--plot", default="", help="write a PNG preview of the 12 joints")
    ap.add_argument("--src-rate", type=float, default=60.0, help="fps the CSV was exported at")
    ap.add_argument("--rate", type=float, default=20.0, help="resample to this Hz (0 = keep source)")
    ap.add_argument("--smooth", type=float, default=0.15, help="low-pass time constant [s], 0 = off")
    ap.add_argument("--gain", type=float, default=1.0, help="scale motion about its own mean")
    ap.add_argument("--inspect", action="store_true", help="dump header/ranges and exit")
    args = ap.parse_args()

    header, body, delim = load_csv(args.csv)
    print(f"[csv] {args.csv}: {len(header)} columns, {len(body)} data rows, delimiter {delim!r}")

    if args.inspect:
        print("\n--- header ---")
        for i, cell in enumerate(header):
            print(f"  [{i:3d}] {cell}")
        return

    cols = find_columns(header, args.hand)
    print(f"[match] {len(cols)}/20 ergonomics channels for the {args.hand} hand")
    if len(cols) < 8:
        print("\n[!] too few ergonomics columns matched.  Either the export did not")
        print("    include ergonomics data, or the column names differ from what the")
        print("    SDK enum suggests.  Re-run with --inspect and send me the header.")
        sys.exit(2)
    missing = [k for k in HUMAN_RANGE_DEG if k not in cols]
    if missing:
        print("[warn] absent, will hold at neutral: "
              + ", ".join(f"{f}.{c}" for f, c in missing))

    # ---- pull the raw channels -------------------------------------------
    n = len(body)
    raw = {k: np.array([to_float(r[i]) if i < len(r) else np.nan for r in body])
           for k, i in cols.items()}

    finite = np.concatenate([v[np.isfinite(v)] for v in raw.values() if np.isfinite(v).any()])
    span = float(np.max(np.abs(finite))) if finite.size else 0.0
    normalised_source = span <= 1.5
    print(f"[units] max |value| = {span:.3f} -> source is "
          + ("already normalised 0..1" if normalised_source else "degrees"))

    # ---- normalise each channel to 0..1 ----------------------------------
    unit = {}
    for key, series in raw.items():
        s = series.copy()
        bad = ~np.isfinite(s)
        if bad.all():
            s = np.zeros_like(s)
        elif bad.any():  # interpolate over dropped frames rather than punching holes
            good = np.flatnonzero(~bad)
            s = np.interp(np.arange(len(s)), good, s[good])
        if normalised_source:
            unit[key] = np.clip(s, 0.0, 1.0)
        else:
            lo, hi = HUMAN_RANGE_DEG[key]
            unit[key] = np.clip((s - lo) / (hi - lo), 0.0, 1.0)

    # ---- project onto the 12 XHand joints --------------------------------
    q = np.zeros((n, 12))
    for j, name in enumerate(JOINT_NAMES):
        acc, wsum = np.zeros(n), 0.0
        for key, w in RETARGET[name]:
            if key in unit:
                acc += w * unit[key]
                wsum += w
        # a missing channel parks the joint at its neutral: mid-range for the
        # index spread (which straddles zero), fully open for everything else
        u = acc / wsum if wsum > 0 else np.full(n, 0.5 if name == "index_joint0" else 0.0)
        if name in INVERT:
            u = 1.0 - u
        lo, hi = JOINT_LIMITS[j]
        q[:, j] = lo + u * (hi - lo)

    if args.gain != 1.0:  # amplify about the per-joint mean
        mu = q.mean(axis=0, keepdims=True)
        q = mu + (q - mu) * args.gain

    # ---- resample + smooth -----------------------------------------------
    t = np.arange(n) / args.src_rate
    if args.rate > 0 and n > 1:
        t_new = np.arange(0.0, t[-1], 1.0 / args.rate)
        q = np.stack([np.interp(t_new, t, q[:, j]) for j in range(12)], axis=1)
        t = t_new
    if args.smooth > 0 and len(t) > 2:
        dt = float(np.median(np.diff(t)))
        alpha = dt / (args.smooth + dt)
        sm = q.copy()
        for i in range(1, len(sm)):
            sm[i] = sm[i - 1] + alpha * (q[i] - sm[i - 1])
        q = sm
    q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    np.savez(args.out, t=t, q=q, joint_names=np.array(JOINT_NAMES),
             joint_limits=JOINT_LIMITS, source=os.path.abspath(args.csv))
    print(f"\n[out] {args.out}: {len(t)} frames, {t[-1] if len(t) else 0:.2f}s "
          f"@ {args.rate if args.rate > 0 else args.src_rate:.0f}Hz")

    print(f"\n{'joint':>15s} {'min':>8s} {'max':>8s} {'travel':>8s} {'range used':>11s}")
    for j, name in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS[j]
        used = (q[:, j].max() - q[:, j].min()) / (hi - lo) * 100
        print(f"{name:>15s} {q[:, j].min():8.3f} {q[:, j].max():8.3f} "
              f"{q[:, j].max() - q[:, j].min():8.3f} {used:10.0f}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 3, figsize=(14, 9), sharex=True)
        for j, name in enumerate(JOINT_NAMES):
            ax = axes[j // 3][j % 3]
            lo, hi = JOINT_LIMITS[j]
            ax.axhspan(lo, hi, color="0.92", zorder=0)
            ax.plot(t, q[:, j], lw=1.4)
            ax.set_title(name, fontsize=9)
            ax.set_ylim(lo - 0.1, hi + 0.1)
            ax.grid(alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel("t [s]")
        fig.suptitle(f"MANUS -> XHand retarget: {os.path.basename(args.csv)}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"[plot] {args.plot}")


if __name__ == "__main__":
    main()
