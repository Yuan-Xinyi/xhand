#!/usr/bin/env python3
"""Find the usable XHand torque range: close slowly on the cube at several
tor_max values and report how far the fingers actually travel.

For each tor_max the hand opens, then closes toward a grasp pose at a fixed
target ramp. Comparing COMMANDED vs MEASURED tells you three regimes:
  - measured ~= commanded, stops early at the cube -> torque fine, cube blocks
  - measured lags everywhere                        -> torque below the
                                                       friction/gravity floor
  - measured ~= commanded to the end                -> missed the cube entirely

Cube should be resting in the palm. Read-only to the arm; only the hand moves.

    /home/lqin/miniconda3/envs/one/bin/python RealExperiments/xhand_torque_probe.py
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foundationpose_repose_real as rr  # path sanitize + joint order

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--tor-max", type=int, nargs="+", default=[20, 50, 80, 120, 200, 300])
    ap.add_argument("--kp", type=int, default=100)
    ap.add_argument("--close", type=float, default=0.6, help="closing angle target [rad]")
    ap.add_argument("--secs", type=float, default=2.5, help="ramp duration per trial [s]")
    args = ap.parse_args()

    rr._prepare_one_imports()
    from one.control.end_effector.xhand.xhand_x import XHandX

    hand = XHandX(port=args.port, baudrate=3000000)
    open_q = np.zeros(12, dtype=np.float32)
    # close the four fingers + thumb flexion; splay joints stay put
    close_q = open_q.copy()
    for j, name in enumerate(rr.ISAAC12):
        if name.endswith("joint1") or name.endswith("joint2") or name == "thumb_joint0":
            close_q[j] = args.close

    print(f"{'tor_max':>8s} {'cmd end':>9s} {'meas end':>9s} {'lag':>8s}  verdict")
    try:
        for tm in args.tor_max:
            hand.move_to(open_q[rr.ISAAC_TO_ONE], speed=0.4, freq=50.0, tor_max=300, kp=args.kp)
            time.sleep(0.5)
            n = int(args.secs * 20)
            meas = None
            for i in range(n):
                a = (i + 1) / n
                q = (1 - a) * open_q + a * close_q
                st = hand.move(q[rr.ISAAC_TO_ONE], read=True, kp=args.kp, tor_max=tm)
                if st is not None:
                    try:
                        m = np.array([float(s.position) for s in st], dtype=np.float32)
                        meas = m[rr.ONE_TO_ISAAC]
                    except (AttributeError, TypeError):
                        pass
                time.sleep(0.05)
            if meas is None:
                print(f"{tm:8d}    (no readback)")
                continue
            act = [j for j, nm in enumerate(rr.ISAAC12)
                   if nm.endswith("joint1") or nm.endswith("joint2") or nm == "thumb_joint0"]
            cmd_end = float(close_q[act].mean())
            meas_end = float(meas[act].mean())
            lag = cmd_end - meas_end
            if lag > 0.9 * args.close:
                verdict = "STUCK — torque below the friction floor"
            elif lag > 0.12:
                verdict = "blocked by the cube (good contact regime)"
            else:
                verdict = "free travel — fingers never met the cube"
            print(f"{tm:8d} {cmd_end:8.3f}r {meas_end:8.3f}r {lag:7.3f}r  {verdict}")
        hand.move_to(open_q[rr.ISAAC_TO_ONE], speed=0.4, freq=50.0, tor_max=300, kp=args.kp)
    finally:
        hand.close()
    print("\nPick the LOWEST tor_max still in the 'blocked by the cube' regime,")
    print("then tune --lead-cap for how much it yields once wedged.")


if __name__ == "__main__":
    main()
