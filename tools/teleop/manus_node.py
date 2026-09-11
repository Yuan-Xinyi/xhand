#!/usr/bin/env python3
"""MANUS glove -> XHand joint targets, a drop-in replacement for perception_node.py.

The Unity bridge (ManusUdpBridge.cs, running on the Windows box next to MANUS
Core) sends raw ergonomics as JSON:

    {"ergo": [20 floats, degrees, one hand, MANUS channel order]}

This node retargets those 20 channels onto the XHand's 12 joints and republishes
them in the same binary wire format perception_node.py uses, so real_node.py and
teleop_sim.py drive from the glove without knowing anything changed.

    Unity ──udp:9881 json──▶ manus_node ──udp:51234 packed──▶ real_node ──▶ hand

Usage:
    python manus_node.py --print          # watch the values, publish as usual
    python manus_node.py --listen-only    # decode and print, publish nothing
"""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import sys
import time

import numpy as np

from protocol import HAND_JOINT_NAMES, make_sender, pack

# The retarget table lives with the CSV offline tool so both paths -- recorded
# takes and live streaming -- project the glove identically.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "RealExperiments", "manus_bridge"))
from manus_csv_to_xhand import (  # noqa: E402
    CHANNELS, FINGERS, HUMAN_RANGE_DEG, INVERT, JOINT_LIMITS, JOINT_NAMES, RETARGET,
)

assert tuple(JOINT_NAMES) == tuple(HAND_JOINT_NAMES), "joint order drifted apart"

# MANUS packs one hand as 5 fingers x 4 channels, thumb first.
ERGO_KEYS = [(f, c) for f in FINGERS for c in CHANNELS]
assert len(ERGO_KEYS) == 20


def ranges_from_calibration(path):
    """Per-channel (lo, hi) from a manus_calibrate.py capture of THIS operator."""
    with open(path) as fh:
        cal = json.load(fh)
    mat = np.asarray([cal["poses"][k] for k in cal["poses"]])
    rng = {}
    for i, key in enumerate(ERGO_KEYS):
        lo, hi = float(mat[:, i].min()), float(mat[:, i].max())
        if hi - lo < 3.0:  # channel never moved; keep the shipped default
            rng[key] = HUMAN_RANGE_DEG[key]
            print(f"[manus][WARN] {key[0]}.{key[1]} spans only {hi - lo:.1f} deg "
                  f"in {path}, keeping the default range")
        else:
            rng[key] = (lo, hi)
    return rng


def retarget(ergo_deg: np.ndarray, invert=INVERT, ranges=HUMAN_RANGE_DEG) -> np.ndarray:
    """20 MANUS ergonomics channels (degrees) -> 12 XHand joint angles (rad)."""
    unit = {}
    for i, key in enumerate(ERGO_KEYS):
        lo, hi = ranges[key]
        unit[key] = float(np.clip((ergo_deg[i] - lo) / (hi - lo), 0.0, 1.0))

    q = np.zeros(12)
    for j, name in enumerate(JOINT_NAMES):
        acc = sum(w * unit[k] for k, w in RETARGET[name])
        wsum = sum(w for _, w in RETARGET[name])
        u = acc / wsum
        if name in invert:
            u = 1.0 - u
        lo, hi = JOINT_LIMITS[j]
        q[j] = lo + u * (hi - lo)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0", help="interface to receive glove JSON on")
    ap.add_argument("--glove-port", type=int, default=9881)
    ap.add_argument("--host", default="127.0.0.1", help="where real_node/teleop_sim listens")
    ap.add_argument("--port", type=int, default=51234)
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="EMA on joint targets, 1.0 = no smoothing")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="scale motion about the joint mid-range")
    ap.add_argument("--calib", default="", metavar="cal.json",
                    help="per-channel ranges from a manus_calibrate.py capture; "
                         "overrides the shipped (also measured) defaults")
    ap.add_argument("--invert", default=None, metavar="J1,J2",
                    help="comma-separated joints to flip, replacing the default set "
                         f"({','.join(sorted(INVERT))}). Pass an empty string to flip none. "
                         "Use this to chase a reversed joint live instead of editing code.")
    ap.add_argument("--timeout", type=float, default=1.0,
                    help="no glove packet for this long -> publish valid=0 (driver holds)")
    ap.add_argument("--rate", type=float, default=60.0, help="publish rate cap Hz")
    ap.add_argument("--print", dest="show", action="store_true", help="print joint values")
    ap.add_argument("--listen-only", action="store_true", help="decode only, publish nothing")
    args = ap.parse_args()

    if args.invert is None:
        invert = set(INVERT)
    else:
        invert = {s.strip() for s in args.invert.split(",") if s.strip()}
        unknown = invert - set(JOINT_NAMES)
        if unknown:
            raise SystemExit(f"[manus] --invert names no such joint: {sorted(unknown)}")
    print(f"[manus] inverted joints: {sorted(invert) or '(none)'}")

    ranges = HUMAN_RANGE_DEG
    if args.calib:
        ranges = ranges_from_calibration(args.calib)
        print(f"[manus] channel ranges from {args.calib}")

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.bind, args.glove_port))
    rx.setblocking(False)
    print(f"[manus] listening for glove JSON on udp://{args.bind}:{args.glove_port}")

    tx = send = None
    if not args.listen_only:
        tx, send = make_sender(args.host, args.port)
        print(f"[manus] publishing xhand joints -> udp://{args.host}:{args.port}")
    print(f"[manus] joints: {list(HAND_JOINT_NAMES)}")

    q_ema = None
    seq = 0
    last_rx = 0.0
    next_pub = 0.0
    n_pkt = 0
    t_report = time.time()
    mid = JOINT_LIMITS.mean(axis=1)

    try:
        while True:
            # block briefly so an idle stream does not spin the CPU, then drain
            # to the freshest datagram -- teleop never wants a backlog
            select.select([rx], [], [], 0.005)
            newest = None
            while True:
                try:
                    newest = rx.recv(4096)
                    n_pkt += 1
                except (BlockingIOError, OSError):
                    break

            if newest is not None:
                try:
                    ergo = np.asarray(json.loads(newest.decode("ascii"))["ergo"], dtype=float)
                except (ValueError, KeyError, UnicodeDecodeError):
                    continue
                if ergo.size != 20:
                    print(f"[manus][WARN] expected 20 channels, got {ergo.size}")
                    continue
                q = retarget(ergo, invert, ranges)
                if args.gain != 1.0:
                    q = np.clip(mid + (q - mid) * args.gain,
                                JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
                q_ema = q if q_ema is None else q_ema + args.smooth * (q - q_ema)
                last_rx = time.time()

            now = time.time()
            if q_ema is not None and now >= next_pub:
                next_pub = now + 1.0 / args.rate
                valid = (now - last_rx) < args.timeout
                if send is not None:
                    send(pack(seq, now, valid, q_ema))
                    seq += 1

            if now - t_report >= 1.0:
                if q_ema is None:
                    print("[manus] waiting for glove packets ... "
                          "(is Unity in Play mode, and is linuxHost set to this box?)")
                else:
                    fresh = (now - last_rx) < args.timeout
                    line = (f"[manus] {n_pkt:5d} pkt/s  "
                            f"{'LIVE' if fresh else 'STALE'}")
                    if args.show:
                        line += "  " + " ".join(f"{v:5.2f}" for v in q_ema)
                    print(line, flush=True)
                n_pkt = 0
                t_report = now
    except KeyboardInterrupt:
        print("\n[manus] stopped.")
    finally:
        rx.close()
        if tx is not None:
            tx.close()


if __name__ == "__main__":
    main()
