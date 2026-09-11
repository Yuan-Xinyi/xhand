# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wire protocol shared by the WiLoR perception node and the Isaac Lab sim driver.

PURE STDLIB ONLY (struct / socket) so it can be imported from *both* conda envs
(`wilor` and `env_isaaclab`) without dragging any heavy dependency into either.

The perception node (env `wilor`) PUBLISHES a small UDP datagram every camera
frame; the sim driver (env `env_isaaclab`) SUBSCRIBES and applies the latest one.
UDP (not TCP) on purpose: teleop wants the freshest sample, never a backlog — a
dropped packet is simply the next one, no head-of-line blocking.

Datagram (little-endian, no padding):  <I d B 12f  = 61 bytes
    seq    uint32   monotonically increasing frame counter
    t      float64  perception-side send timestamp (seconds, time.time())
    valid  uint8    1 = a hand was tracked this frame, 0 = no hand (hold last)
    q      12x f32  xhand joint targets (rad) in HAND_JOINT_NAMES order
"""
from __future__ import annotations

import socket
import struct

# Canonical 12-joint order on the wire == the DexPilot config `target_joint_names`
# order (see tools/crossdex_retarget/configs/xhand_right_dexpilot.yml). The
# perception node reorders the retargeter output into THIS order before sending;
# the sim driver permutes from THIS order into its own articulation order. Keep
# the two ends in lockstep by importing this single list.
HAND_JOINT_NAMES = (
    "thumb_joint0", "thumb_joint1", "thumb_joint2",
    "index_joint0", "index_joint1", "index_joint2",
    "middle_joint0", "middle_joint1",
    "ring_joint0", "ring_joint1",
    "pinky_joint0", "pinky_joint1",
)
N_JOINTS = len(HAND_JOINT_NAMES)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 51234

_FMT = "<IdB" + f"{N_JOINTS}f"
PACKET_SIZE = struct.calcsize(_FMT)  # 61 bytes


def pack(seq: int, t: float, valid: bool, q) -> bytes:
    """Serialize one joint-target sample. `q` is an iterable of N_JOINTS floats."""
    q = list(q)
    if len(q) != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} joints, got {len(q)}")
    return struct.pack(_FMT, seq & 0xFFFFFFFF, float(t), 1 if valid else 0, *q)


def unpack(data: bytes):
    """Return (seq, t, valid: bool, q: list[float]) from a datagram."""
    fields = struct.unpack(_FMT, data)
    return fields[0], fields[1], bool(fields[2]), list(fields[3:])


def make_sender(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """UDP sender: returns (sock, send_fn). send_fn(bytes) fires one datagram."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (host, port)
    return sock, lambda payload: sock.sendto(payload, addr)


def make_receiver(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Non-blocking UDP receiver bound to (host, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.setblocking(False)
    return sock


def drain_latest(sock):
    """Read all queued datagrams, return the newest decoded sample or None.

    Teleop only cares about the freshest joint target; older queued packets are
    stale and discarded. Returns (seq, t, valid, q) or None if nothing waiting.
    """
    latest = None
    while True:
        try:
            data, _ = sock.recvfrom(PACKET_SIZE * 4)
        except BlockingIOError:
            break
        except OSError:
            break
        if len(data) == PACKET_SIZE:
            latest = data
    return unpack(latest) if latest is not None else None
