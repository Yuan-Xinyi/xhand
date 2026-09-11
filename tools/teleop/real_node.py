# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop REAL-HAND driver — drives the physical XHand over RS-485 serial.

Drop-in replacement for teleop_sim.py on the robot side of the UDP link:

    perception_node.py (wilor env) --UDP:51234--> real_node.py --RS-485--> XHand

Protocol (frame 0xAA55, cmd 0x02 = position control, 12x FingerCommand) ported
from `one` repo, branch `dex-hand`, Yuan/dexterous_hand/xhand_con/. Needs only
pyserial + numpy — runs fine in the `wilor` env (or any env).

Safety layers between the raw teleop stream and the motors:
  1. joint-limit clamp with a configurable margin (limits from xhand_right.urdf)
  2. per-joint speed limit (--max-speed rad/s) — also gives a soft start ramp
  3. EMA smoothing (--smooth)
  4. the very first target is initialized FROM THE HAND'S ACTUAL POSITION
     (read back over serial), so enabling teleop never causes a jump
  5. stream watchdog: no valid packet for --timeout s -> HOLD current pose
     (never snaps anywhere on tracking loss)

Bring-up (do these once before real teleop):
    python real_node.py --check          # slow per-joint sweep: verify id<->joint mapping
    python real_node.py --open           # command the open (0 rad) pose and exit
    python real_node.py --dry-run        # follow the UDP stream, print, no serial
Then:
    python real_node.py                  # real teleop follow mode
"""
from __future__ import annotations

import argparse
import struct
import sys
import time

import numpy as np

from protocol import HAND_JOINT_NAMES, N_JOINTS, drain_latest, make_receiver

# ---------------------------------------------------------------------------- hand constants
# Hardware finger id i is assumed to be HAND_JOINT_NAMES[i] (== xhand_right.urdf
# joint order). Run `--check` once to verify on your unit; if a joint moves out
# of turn, fix the mapping HERE (and only here).
HW_JOINT_NAMES = tuple(HAND_JOINT_NAMES)

# [lower, upper] rad per joint, from assets/xhand2R32/xhand_right.urdf (same
# values the sim + retargeter use; keep in lockstep with the URDF).
URDF_LIMITS = np.array([
    [0.000, 1.830],    # thumb_joint0
    [-1.050, 1.570],   # thumb_joint1
    [-0.175, 1.830],   # thumb_joint2
    [-0.175, 0.175],   # index_joint0
    [0.000, 1.920],    # index_joint1
    [0.000, 1.920],    # index_joint2
    [0.000, 1.920],    # middle_joint0
    [0.000, 1.920],    # middle_joint1
    [0.000, 1.920],    # ring_joint0
    [0.000, 1.920],    # ring_joint1
    [0.000, 1.920],    # pinky_joint0
    [0.000, 1.920],    # pinky_joint1
], dtype=np.float32)

OPEN_POSE = np.zeros(N_JOINTS, dtype=np.float32)   # flat open hand

# ---------------------------------------------------------------------------- serial protocol
# CRC-16/XMODEM (poly 0x1021, init 0x0000), table generated at import — matches
# the crc16tab used by the xhand firmware.
_CRC_TABLE = []
for _b in range(256):
    _c = _b << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x1021) if (_c & 0x8000) else (_c << 1)
    _CRC_TABLE.append(_c & 0xFFFF)


def _crc16(data: bytes) -> bytes:
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return struct.pack("<H", crc)


_FINGER_CMD_FMT = "<Hhhh f H H H H H H"   # 24 bytes
# Per-finger state layout: id(B) sensor_id(B) position(f) + error/temp words.
# The reference code's size comments are wrong twice over (state "22 B" is
# really 24, sensor "366 B" is really 3+3*120+20+1 = 384), so match the payload
# size against the plausible combinations; the position float is at offset 2
# in every variant. Measured on our unit: 12*24 + 5*384 = 2208 B.
_STATE_STRIDES = (24, 22)
_SENSOR_BLOCKS = (5 * 384, 5 * 366)


class XHandSerial:
    """Minimal RS-485 link to the XHand: position commands + position readback."""

    def __init__(self, port: str, baudrate: int = 3_000_000,
                 kp: int = 100, kd: int = 10, tor_max: int = 300):
        import serial
        self.ser = serial.Serial(port=port, baudrate=baudrate,
                                 bytesize=8, parity="N", stopbits=1, timeout=0.2)
        self.kp, self.kd, self.tor_max = kp, kd, tor_max

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _send(self, command: int, payload: bytes = b"") -> bytes | None:
        head = struct.pack("<HBBBH", 0xAA55, 0xFE, 0x80, command, len(payload))
        pkt = head + payload + _crc16(head + payload)
        self.ser.reset_input_buffer()
        self.ser.write(pkt)
        self.ser.flush()
        resp_head = self.ser.read(7)
        if len(resp_head) < 7:
            return None
        _, _, _, _, dlen = struct.unpack("<HBBBH", resp_head)
        body = self.ser.read(dlen + 2)
        if len(body) < dlen + 2:
            return None
        data, crc = body[:-2], body[-2:]
        if crc != _crc16(resp_head + data):
            return None
        return data

    def version(self) -> bytes | None:
        """Firmware version query (0x13) — read-only, safe connectivity check."""
        return self._send(0x13)

    def command_positions(self, q) -> np.ndarray | None:
        """Send 12 joint targets (rad, HW order). Returns measured positions or None."""
        payload = b"".join(
            struct.pack(_FINGER_CMD_FMT, i, self.kp, 0, self.kd,
                        float(q[i]), self.tor_max, 3, 0, 0, 0, 0)
            for i in range(N_JOINTS))
        return self._parse_positions(self._send(0x02, payload))

    _warned_payload = False

    @classmethod
    def _parse_positions(cls, data: bytes | None) -> np.ndarray | None:
        """Extract 12 measured joint positions from a state response, if present.

        Handles both known state strides (see _STATE_STRIDES) by matching the
        payload size: 12*stride finger states, optionally + the sensor blob.
        """
        if data is None:
            return None
        for stride in _STATE_STRIDES:
            rem = len(data) - N_JOINTS * stride
            if rem != 0 and rem not in _SENSOR_BLOCKS:
                continue
            pos = np.array([struct.unpack_from("<f", data, i * stride + 2)[0]
                            for i in range(N_JOINTS)], dtype=np.float32)
            if np.all(np.isfinite(pos)) and np.all(np.abs(pos) < 6.3):
                return pos
        if not cls._warned_payload:
            cls._warned_payload = True
            print(f"[real][WARN] unrecognized state payload ({len(data)} B) — "
                  f"running without position readback")
        return None


# ---------------------------------------------------------------------------- helpers
def _rate_limit(target, current, max_step):
    return current + np.clip(target - current, -max_step, max_step)


def _settle_to(hand, goal, start, seconds, hz, label=None):
    """Slow linear ramp start->goal (used by --check / --open). Blocking."""
    steps = max(int(seconds * hz), 1)
    for k in range(steps + 1):
        q = start + (goal - start) * (k / steps)
        hand.command_positions(q)
        time.sleep(1.0 / hz)
    if label:
        print(f"[real] {label}: done")


def _read_start_pose(hand, args) -> np.ndarray:
    """Get the hand's actual pose so the first command never jumps.

    The 0x02 response carries measured positions; send it once with all-zero
    gains / torque cap / mode so nothing can actually move, purely to read.
    """
    pos = XHandSerial._parse_positions(hand._send(0x02, b"".join(
        struct.pack(_FINGER_CMD_FMT, i, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0)  # mode=0: no-op read
        for i in range(N_JOINTS))))
    if pos is not None and np.all(np.isfinite(pos)):
        print(f"[real] start pose read from hand: {np.round(pos, 3)}")
        return np.clip(pos, URDF_LIMITS[:, 0], URDF_LIMITS[:, 1])
    print("[real][WARN] could not read hand pose; starting from open pose")
    return OPEN_POSE.copy()


# ---------------------------------------------------------------------------- modes
def run_check(hand, args):
    """Wiggle each joint alone (small back-and-forth) to verify the id<->joint mapping."""
    q = _read_start_pose(hand, args)
    _settle_to(hand, OPEN_POSE, q, 2.0, args.rate, "settle to open")
    # small amplitude only: min(--check-span, 30% of range), kept inside limits
    span = np.minimum(args.check_span, 0.3 * (URDF_LIMITS[:, 1] - URDF_LIMITS[:, 0]))
    for j, name in enumerate(HW_JOINT_NAMES):
        print(f"[real] check id={j:2d}  ->  should wiggle ONLY {name}")
        goal = OPEN_POSE.copy()
        goal[j] = np.clip(OPEN_POSE[j] + span[j], URDF_LIMITS[j, 0], URDF_LIMITS[j, 1])
        for _ in range(2):  # two gentle back-and-forth cycles
            _settle_to(hand, goal, OPEN_POSE, 0.7, args.rate)
            _settle_to(hand, OPEN_POSE, goal, 0.7, args.rate)
    print("[real] mapping check complete.")


def run_teleop(hand, args):
    sock = make_receiver(args.host, args.port)
    print(f"[real] listening udp://{args.host}:{args.port} "
          f"(rate {args.rate} Hz, max speed {args.max_speed} rad/s)")

    lo = URDF_LIMITS[:, 0] + args.limit_margin
    hi = URDF_LIMITS[:, 1] - args.limit_margin
    dt = 1.0 / args.rate
    max_step = args.max_speed * dt

    cmd = _read_start_pose(hand, args) if hand else OPEN_POSE.copy()
    ema = cmd.copy()
    last_valid_t = None
    got_first = False
    n_sent = 0
    try:
        while True:
            t0 = time.time()
            sample = drain_latest(sock)
            if sample is not None:
                _seq, _t, valid, q = sample
                if valid:
                    tgt = np.clip(np.asarray(q, dtype=np.float32), lo, hi)
                    ema = (1 - args.smooth) * ema + args.smooth * tgt
                    last_valid_t = time.time()
                    if not got_first:
                        got_first = True
                        print("[real] first hand sample — following (speed-limited ramp-in).")
            # watchdog: stale stream -> hold (ema stops moving, cmd converges to it)
            cmd = _rate_limit(ema, cmd, max_step)
            if hand:
                meas = hand.command_positions(cmd)
            else:
                meas = None
            n_sent += 1
            if n_sent % args.rate == 0:  # 1 Hz status
                fresh = last_valid_t is not None and (time.time() - last_valid_t) < args.timeout
                m = f" meas[0]={meas[0]:+.2f}" if meas is not None else ""
                print(f"[real] tracking={fresh} cmd[min/mean/max]="
                      f"{cmd.min():+.2f}/{cmd.mean():+.2f}/{cmd.max():+.2f}{m}", flush=True)
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n[real] stopping — holding last pose (hand keeps position control).")
    finally:
        sock.close()


def main():
    ap = argparse.ArgumentParser(description="Drive the real XHand from the teleop UDP stream.")
    ap.add_argument("--serial-port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=3_000_000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=51234)
    ap.add_argument("--rate", type=int, default=50, help="serial command rate Hz")
    ap.add_argument("--smooth", type=float, default=0.5, help="EMA on incoming targets (0..1]")
    ap.add_argument("--max-speed", type=float, default=2.5, help="per-joint speed limit rad/s")
    ap.add_argument("--limit-margin", type=float, default=0.02,
                    help="shrink URDF limits by this margin (rad)")
    ap.add_argument("--timeout", type=float, default=2.0,
                    help="stream watchdog (s); stale -> hold pose")
    ap.add_argument("--kp", type=int, default=100)
    ap.add_argument("--kd", type=int, default=10)
    ap.add_argument("--tor-max", type=int, default=300, help="firmware torque cap per joint")
    ap.add_argument("--ping", action="store_true",
                    help="read firmware version and exit (read-only, no motion)")
    ap.add_argument("--read", action="store_true",
                    help="read measured joint positions and exit (read-only, no motion). "
                         "Prefer this over --ping to prove the hand is alive: this firmware "
                         "answers 0x02 but stays silent on the 0x13 version query.")
    ap.add_argument("--check", action="store_true",
                    help="small per-joint wiggle to verify id<->joint mapping, then exit")
    ap.add_argument("--check-span", type=float, default=0.25,
                    help="wiggle amplitude for --check (rad, capped at 30%% of joint range)")
    ap.add_argument("--open", action="store_true", help="ramp to open pose, then exit")
    ap.add_argument("--dry-run", action="store_true", help="no serial; print stream only")
    args = ap.parse_args()

    hand = None
    if not args.dry_run:
        hand = XHandSerial(args.serial_port, args.baud,
                           kp=args.kp, kd=args.kd, tor_max=args.tor_max)
        print(f"[real] serial open {args.serial_port} @ {args.baud}")

    try:
        if args.ping:
            if hand is None:
                sys.exit("[real] --ping needs the hand (drop --dry-run)")
            v = hand.version()
            print(f"[real] firmware version: {v.hex() if v else 'NO RESPONSE'}")
        elif args.read:
            if hand is None:
                sys.exit("[real] --read needs the hand (drop --dry-run)")
            q = _read_start_pose(hand, args)
            for name, v, (lo, hi) in zip(HW_JOINT_NAMES, q, URDF_LIMITS):
                bar = int(round(20 * (v - lo) / (hi - lo)))
                print(f"  {name:>15s} {v:7.3f} rad  [{'#' * bar}{'.' * (20 - bar)}]  "
                      f"limits {lo:6.3f}..{hi:5.3f}")
        elif args.check:
            if hand is None:
                sys.exit("[real] --check needs the hand (drop --dry-run)")
            run_check(hand, args)
        elif args.open:
            if hand is None:
                sys.exit("[real] --open needs the hand (drop --dry-run)")
            q = _read_start_pose(hand, args)
            _settle_to(hand, OPEN_POSE, q, 2.5, args.rate, "open pose")
        else:
            run_teleop(hand, args)
    finally:
        if hand is not None:
            # An impatient second Ctrl-C lands mid-close(); retry once so the fd
            # is released rather than left busy for the next run.
            for _ in range(2):
                try:
                    hand.close()
                    break
                except KeyboardInterrupt:
                    continue
            print("[real] serial closed.", flush=True)


if __name__ == "__main__":
    main()
