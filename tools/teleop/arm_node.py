# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard jog of the REAL xArm7 end-effector (runs alongside hand teleop).

Streams Cartesian set-points to the arm (servo mode 1, xArm-Python-SDK) at a
fixed rate while you hold keys — velocity control with smooth stop on release.
Runs happily in parallel with real_node.py / teleop_sim.py: the arm is on
ethernet, the hand on RS-485, the camera teleop on UDP; nothing collides.

    Translation (BASE frame):        Rotation (TOOL frame, about the TCP):
        W / S   +x / -x  (fwd/back)      Q / E     twist   (about tool z)
        A / D   +y / -y  (left/right)    Up / Down  pitch  (about tool x)
        R / F   +z / -z  (up/down)       Left/Right roll   (about tool y)

    [ / ]   speed down / up (0.25x .. 2x)
    SPACE   hold + re-sync target to the arm's actual pose
    ESC     quit (leaves servo mode cleanly)

Keys are captured globally via pynput (needs X; any window may have focus —
keep the teleop dashboard focused and jog blind-typing). Safety: workspace box
clamp, speed caps, target seeded from the arm's actual pose (no jump on start).

Usage (conda activate wilor):
    python arm_node.py                     # ONE_ARM_IP or 192.168.1.205
    python arm_node.py --ip 192.168.1.205 --lin-speed 0.05 --ang-speed 20
    python arm_node.py --dry-run           # no arm: print the jogged pose
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------- keyboard
class KeyJog:
    """pynput listener -> set of currently held jog keys."""

    def __init__(self):
        from pynput import keyboard
        self._kb = keyboard
        self.held: set[str] = set()
        self.quit = False
        self.hold_sync = False   # set by SPACE, consumed by the main loop
        self.speed_scale = 1.0
        self._listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self._listener.start()

    def _name(self, key):
        kb = self._kb
        if isinstance(key, kb.KeyCode) and key.char:
            return key.char.lower()
        return {kb.Key.up: "up", kb.Key.down: "down", kb.Key.left: "left",
                kb.Key.right: "right", kb.Key.space: "space", kb.Key.esc: "esc"}.get(key)

    def _press(self, key):
        n = self._name(key)
        if n == "esc":
            self.quit = True
        elif n == "space":
            self.hold_sync = True
        elif n == "[":
            self.speed_scale = max(0.25, self.speed_scale / 1.5)
            print(f"[arm] speed x{self.speed_scale:.2f}")
        elif n == "]":
            self.speed_scale = min(2.0, self.speed_scale * 1.5)
            print(f"[arm] speed x{self.speed_scale:.2f}")
        elif n:
            self.held.add(n)

    def _release(self, key):
        n = self._name(key)
        if n:
            self.held.discard(n)

    def twist(self):
        """(v_base[3] in {-1,0,1}, w_tool[3] in {-1,0,1}) from held keys."""
        h = self.held
        v = np.array([("w" in h) - ("s" in h),
                      ("a" in h) - ("d" in h),
                      ("r" in h) - ("f" in h)], dtype=np.float64)
        w = np.array([("up" in h) - ("down" in h),      # pitch about tool x
                      ("left" in h) - ("right" in h),   # roll  about tool y
                      ("q" in h) - ("e" in h)], dtype=np.float64)  # twist about tool z
        return v, w

    def stop(self):
        self._listener.stop()


# ---------------------------------------------------------------------------- arm wrapper
class XArm:
    def __init__(self, ip: str):
        from xarm.wrapper import XArmAPI
        self.api = XArmAPI(ip, is_radian=True)
        self.api.motion_enable(enable=True)
        self.api.clean_error()
        self.api.set_mode(1)     # servo: high-rate Cartesian set-point streaming
        self.api.set_state(0)
        time.sleep(0.2)

    def pose(self):
        """(pos_m[3], R[3x3]) actual TCP pose in the base frame."""
        code, p = self.api.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed (code {code})")
        pos = np.asarray(p[:3], dtype=np.float64) / 1000.0
        rot = Rotation.from_euler("xyz", p[3:6]).as_matrix()
        return pos, rot

    def servo_to(self, pos_m, rot) -> int:
        rpy = Rotation.from_matrix(rot).as_euler("xyz")
        mv = (np.asarray(pos_m) * 1000.0).tolist() + rpy.tolist()
        return self.api.set_servo_cartesian(mv, is_radian=True)

    def close(self):
        try:
            self.api.set_mode(0)
            self.api.set_state(0)
            self.api.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Keyboard (WASD) jog of the real xArm7 TCP.")
    ap.add_argument("--ip", default=os.environ.get("ONE_ARM_IP", "192.168.1.205"))
    ap.add_argument("--rate", type=float, default=100.0, help="servo streaming rate Hz")
    ap.add_argument("--lin-speed", type=float, default=0.05, help="translation speed m/s")
    ap.add_argument("--ang-speed", type=float, default=20.0, help="rotation speed deg/s")
    ap.add_argument("--workspace", default="0.15,0.70,-0.45,0.45,0.06,0.60",
                    help="base-frame clamp box x0,x1,y0,y1,z0,z1 (m)")
    ap.add_argument("--dry-run", action="store_true", help="no arm; print the jogged pose")
    args = ap.parse_args()

    ws = np.array([float(v) for v in args.workspace.split(",")], dtype=np.float64)
    assert ws.shape == (6,), "--workspace needs 6 comma-separated numbers"
    ws_lo, ws_hi = ws[0::2], ws[1::2]

    arm = None
    if args.dry_run:
        pos = np.array([0.35, 0.0, 0.30])
        rot = np.eye(3)
        print("[arm] DRY RUN — no arm connected")
    else:
        arm = XArm(args.ip)
        pos, rot = arm.pose()
        print(f"[arm] connected {args.ip}; TCP at {np.round(pos, 3)} m")
    print("[arm] W/S A/D R/F translate | Q/E twist, arrows pitch/roll | "
          "[ ] speed | SPACE hold | ESC quit")

    jog = KeyJog()
    dt = 1.0 / args.rate
    w_max = np.radians(args.ang_speed)
    n_err = 0
    try:
        while not jog.quit:
            t0 = time.time()
            if jog.hold_sync:
                jog.hold_sync = False
                if arm is not None:
                    pos, rot = arm.pose()
                print(f"[arm] HOLD — target re-synced to {np.round(pos, 3)}")
            v_dir, w_dir = jog.twist()
            s = jog.speed_scale
            if np.any(v_dir) or np.any(w_dir):
                pos = np.clip(pos + v_dir * args.lin_speed * s * dt, ws_lo, ws_hi)
                if np.any(w_dir):  # tool-frame rotation about the TCP
                    rot = rot @ Rotation.from_rotvec(w_dir * w_max * s * dt).as_matrix()
                if args.dry_run and int(t0 * 4) != int((t0 - dt) * 4):
                    rpy = np.degrees(Rotation.from_matrix(rot).as_euler("xyz"))
                    print(f"[arm] pos={np.round(pos,3)} rpy(deg)={np.round(rpy,1)}")
            if arm is not None:
                code = arm.servo_to(pos, rot)
                if code != 0:
                    n_err += 1
                    if n_err in (1, 50):
                        print(f"[arm][WARN] servo code {code} (arm error/limit?) x{n_err}")
                    if n_err >= 200:
                        print("[arm] too many servo errors — stopping for safety")
                        break
                else:
                    n_err = 0
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        jog.stop()
        if arm is not None:
            arm.close()
        print("[arm] stopped.")


if __name__ == "__main__":
    main()
