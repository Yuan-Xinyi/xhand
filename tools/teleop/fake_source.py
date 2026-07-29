# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted teleop source — publishes a fake open<->close hand trajectory.

Stand-in for perception_node.py so the sim driver (teleop_sim.py) can be tested
WITHOUT a camera or WiLoR. Sends joint targets in protocol.HAND_JOINT_NAMES order
at a fixed rate. Pure stdlib — runs in any env.

    python fake_source.py --rate 30 --period 3.0
"""
from __future__ import annotations

import argparse
import math
import time

from protocol import HAND_JOINT_NAMES, make_sender, pack

# Representative open / fist xhand joint targets (rad), in HAND_JOINT_NAMES order,
# measured from the DexPilot retargeter on synthetic open / fist MANO hands. The
# 5 distal joints DexPilot does not drive sit at their ~0.96 rad midpoint.
Q_OPEN = [0.629, 1.026, 0.827, 0.176, 0.665, 0.960,
          0.487, 0.960, 0.315, 0.960, 0.114, 0.960]
Q_FIST = [0.827, 1.069, 0.827, 0.106, 0.916, 0.960,
          0.774, 0.960, 0.636, 0.960, 0.476, 0.960]


def main():
    ap = argparse.ArgumentParser(description="Publish a scripted open<->close hand trajectory.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=51234)
    ap.add_argument("--rate", type=float, default=30.0, help="packets per second")
    ap.add_argument("--period", type=float, default=3.0, help="seconds per open->fist->open cycle")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    args = ap.parse_args()

    sock, send = make_sender(args.host, args.port)
    print(f"[fake_source] publishing open<->close -> udp://{args.host}:{args.port} "
          f"@ {args.rate:.0f} Hz, period {args.period:.1f}s")
    print(f"[fake_source] joint order: {list(HAND_JOINT_NAMES)}")

    dt = 1.0 / args.rate
    t0 = time.time()
    seq = 0
    try:
        while True:
            now = time.time()
            if args.seconds and (now - t0) >= args.seconds:
                break
            a = 0.5 * (1.0 - math.cos(2.0 * math.pi * (now - t0) / args.period))  # 0..1..0
            q = [o + a * (f - o) for o, f in zip(Q_OPEN, Q_FIST)]
            send(pack(seq, now, True, q))
            seq += 1
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print(f"[fake_source] stopped after {seq} packets.")


if __name__ == "__main__":
    main()
