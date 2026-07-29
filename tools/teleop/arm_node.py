# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad jog of the REAL xArm7 end-effector (runs alongside hand teleop).

Streams Cartesian set-points to the arm (servo mode 1, xArm-Python-SDK) at a
fixed rate — analog velocity control from a Logitech F310 gamepad in X mode
(XInput; slide the bottom switch to "X"). Runs in parallel with real_node.py /
teleop_sim.py: arm on ethernet, hand on RS-485, camera teleop on UDP.

    LEFT STICK    translate x/y (BASE frame: up = forward +x, left = +y)
    LT / RT       translate z up / down (analog)
    RIGHT STICK   horizontal = wrist twist (about tool z)
                  vertical   = pitch      (about tool x)
    LB / RB       roll left / right (about tool y)
    D-pad up/down speed scale up / down (0.25x .. 2x)
    A             hold + re-sync target to the arm's actual pose
    B             reset orientation to the reference rpy (--reset-rpy), at the
                  capped angular rate; any rotation input cancels it
    Back          quit (leaves servo mode cleanly)

Stick deflection maps to velocity with a deadzone + quadratic curve, so small
deflections give fine millimetric motion and full deflection the capped speed.
All rotations are about the TCP (tool frame) — the hand pivots in place.
`--input keyboard` restores the old WASD scheme (pynput).

Safety: workspace box clamp, speed caps, target seeded from the arm's actual
pose (no jump on start), auto-stop on persistent servo errors.

Usage (conda activate wilor):
    python arm_node.py                     # ONE_ARM_IP or 192.168.1.205
    python arm_node.py --ip 192.168.1.205 --lin-speed 0.05 --ang-speed 20
    python arm_node.py --dry-run           # no arm: print the jogged pose
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------- gamepad
class GamepadJog:
    """Logitech F310 (X mode) via the kernel joystick API — pure stdlib.

    xpad layout: axes 0 LX, 1 LY, 2 LT, 3 RX, 4 RY, 5 RT, 6/7 d-pad;
    buttons 0 A, 1 B, 2 X, 3 Y, 4 LB, 5 RB, 6 Back, 7 Start. Triggers rest at
    -32767 (reported by the driver's init events on open, so no touch needed).
    """

    def __init__(self, device: str = "/dev/input/js0", deadzone: float = 0.15):
        self._fd = os.open(device, os.O_RDONLY)
        self._dz = deadzone
        self._axes = [0.0] * 8
        self._axes[2] = self._axes[5] = -1.0   # trigger rest position
        self._btn = [0] * 11
        self.quit = False
        self.hold_sync = False
        self.reset_ori = False
        self.speed_scale = 1.0
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while not self.quit:
            try:
                ev = os.read(self._fd, 8)
            except OSError:
                break
            if len(ev) != 8:
                continue
            _, val, typ, num = struct.unpack("<IhBB", ev)
            typ &= 0x7F                        # init events carry the same payload
            if typ == 0x02 and num < 8:
                prev = self._axes[num]
                self._axes[num] = val / 32767.0
                if num == 7:                   # d-pad vertical: edge -> speed step
                    if val < -16000 and prev > -0.5:
                        self._bump_speed(+1)
                    elif val > 16000 and prev < 0.5:
                        self._bump_speed(-1)
            elif typ == 0x01 and num < 11:
                edge = val and not self._btn[num]
                self._btn[num] = val
                if edge:
                    if num == 6:               # Back
                        self.quit = True
                    elif num == 0:             # A
                        self.hold_sync = True
                    elif num == 1:             # B
                        self.reset_ori = True

    def _bump_speed(self, direction: int):
        self.speed_scale = float(np.clip(
            self.speed_scale * (1.5 if direction > 0 else 1 / 1.5), 0.25, 2.0))
        print(f"[arm] speed x{self.speed_scale:.2f}")

    def _shape(self, x: float) -> float:
        """Deadzone + quadratic response: fine control near center."""
        if abs(x) < self._dz:
            return 0.0
        x = (x - math.copysign(self._dz, x)) / (1.0 - self._dz)
        return x * abs(x)

    def twist(self):
        """(v_base[3], w_tool[3]) each in [-1, 1] — analog velocity commands."""
        ax = self._axes
        lx, ly = self._shape(ax[0]), self._shape(ax[1])
        rx, ry = self._shape(ax[3]), self._shape(ax[4])
        lt, rt = (ax[2] + 1.0) / 2.0, (ax[5] + 1.0) / 2.0
        v = np.array([-ly, -lx, lt - rt])                  # fwd(+x), left(+y), up(+z): LT up, RT down
        w = np.array([-ry,                                  # pitch about tool x
                      float(self._btn[5] - self._btn[4]),   # roll  about tool y (RB/LB)
                      -rx])                                 # twist about tool z
        return v, w

    def stop(self):
        self.quit = True
        try:
            os.close(self._fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------- keyboard
class KeyJog:
    """pynput listener -> set of currently held jog keys."""

    def __init__(self):
        from pynput import keyboard
        self._kb = keyboard
        self.held: set[str] = set()
        self.quit = False
        self.hold_sync = False   # set by SPACE, consumed by the main loop
        self.reset_ori = False   # set by B, consumed by the main loop
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
        elif n == "b":
            self.reset_ori = True
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
    ap = argparse.ArgumentParser(description="Gamepad/keyboard jog of the real xArm7 TCP.")
    ap.add_argument("--ip", default=os.environ.get("ONE_ARM_IP", "192.168.1.205"))
    ap.add_argument("--input", choices=["gamepad", "keyboard"], default="gamepad",
                    help="jog source: Logitech F310 in X mode (default) or WASD keyboard")
    ap.add_argument("--device", default="/dev/input/js0", help="joystick device node")
    ap.add_argument("--deadzone", type=float, default=0.15, help="stick deadzone (0..1)")
    ap.add_argument("--rate", type=float, default=100.0, help="servo streaming rate Hz")
    ap.add_argument("--lin-speed", type=float, default=0.05, help="translation speed m/s")
    ap.add_argument("--ang-speed", type=float, default=20.0, help="rotation speed deg/s")
    ap.add_argument("--workspace", default="0.15,0.70,-0.45,0.45,0.06,0.60",
                    help="base-frame clamp box x0,x1,y0,y1,z0,z1 (m)")
    ap.add_argument("--reset-rpy", default="119.71,-5.78,89.02",
                    help="B-button reference orientation, base-frame rpy in DEGREES "
                         "(captured from the arm's taught neutral pose)")
    ap.add_argument("--reset-speed", type=float, default=10.0,
                    help="angular rate (deg/s) of the B-button orientation reset — "
                         "kept slow on purpose, independent of the speed scale")
    ap.add_argument("--dry-run", action="store_true", help="no arm; print the jogged pose")
    args = ap.parse_args()

    ws = np.array([float(v) for v in args.workspace.split(",")], dtype=np.float64)
    assert ws.shape == (6,), "--workspace needs 6 comma-separated numbers"
    ws_lo, ws_hi = ws[0::2], ws[1::2]

    reset_rot = Rotation.from_euler(
        "xyz", np.radians([float(v) for v in args.reset_rpy.split(",")])).as_matrix()
    reset_step = np.radians(args.reset_speed)  # rad/s, NOT scaled by speed_scale

    arm = None
    if args.dry_run:
        pos = np.array([0.35, 0.0, 0.30])
        rot = np.eye(3)
        print("[arm] DRY RUN — no arm connected")
    else:
        arm = XArm(args.ip)
        pos, rot = arm.pose()
        print(f"[arm] connected {args.ip}; TCP at {np.round(pos, 3)} m")
    if args.input == "gamepad":
        jog = GamepadJog(args.device, deadzone=args.deadzone)
        print(f"[arm] F310 gamepad on {args.device} (X mode) — "
              "L-stick xy, LT up/RT down | R-stick twist+pitch, LB/RB roll | "
              "d-pad speed | A hold | B ori-reset | Back quit")
    else:
        jog = KeyJog()
        print("[arm] W/S A/D R/F translate | Q/E twist, arrows pitch/roll | "
              "[ ] speed | SPACE hold | ESC quit")
    dt = 1.0 / args.rate
    w_max = np.radians(args.ang_speed)
    n_err = 0
    reset_active = False
    try:
        while not jog.quit:
            t0 = time.time()
            if jog.hold_sync:
                jog.hold_sync = False
                reset_active = False
                if arm is not None:
                    pos, rot = arm.pose()
                print(f"[arm] HOLD — target re-synced to {np.round(pos, 3)}")
            if jog.reset_ori:
                jog.reset_ori = False
                reset_active = True
                print(f"[arm] orientation reset -> rpy {args.reset_rpy} deg "
                      f"(slow, {args.reset_speed:.0f} deg/s; move a rotation stick to cancel)")
            v_dir, w_dir = jog.twist()
            s = jog.speed_scale
            if reset_active:
                if np.any(w_dir):          # manual rotation overrides the reset
                    reset_active = False
                    print("[arm] orientation reset cancelled")
                else:                      # creep toward the reference orientation
                    err = Rotation.from_matrix(rot.T @ reset_rot).as_rotvec()
                    ang = float(np.linalg.norm(err))
                    if ang < 1e-3:
                        reset_active = False
                        print("[arm] orientation reset done")
                    else:
                        step = err / ang * min(ang, reset_step * dt)
                        rot = rot @ Rotation.from_rotvec(step).as_matrix()
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
