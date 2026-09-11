#!/usr/bin/env python3
"""MANUS 3D hand skeleton -> XHand joints, via the same DexPilot retargeter the
camera path uses.

This replaces manus_node.py's hand-written 20-channel table.  That table needed
a channel assignment, a sign and a range guessed per joint, and a wrong guess
could only be caught by watching the real hand move the wrong way.  Here the
glove's 3D joint positions go into `retarget.HandRetargeter` -- the same
DexPilot optimizer the RL token policy was distilled through -- so the mapping
is geometry, and a mistake shows up as a visibly misplaced finger in sim.

    Unity ──udp:9882 json──▶ manus_skel_node ──udp:51234 packed──▶ real_node/sim

Bring-up order, each step cheap to check:
    python manus_skel_node.py --dump          # what nodes is the glove sending?
    python manus_skel_node.py --print         # joint targets, publishing nothing
    python teleop.py --manus-skel             # drive the SIM hand
    python teleop.py --manus-skel --real      # drive the real hand
"""
from __future__ import annotations

import argparse
import json
import select
import socket
import time

import numpy as np

from protocol import HAND_JOINT_NAMES, make_sender, pack

# CoreSDK.ChainType / CoreSDK.FingerJointType, read out of ManusSDKTypes.cs
CHAIN_HAND = 13
CHAIN_FINGER = {5: "thumb", 6: "index", 7: "middle", 8: "ring", 9: "pinky"}
JT_METACARPAL, JT_PROXIMAL, JT_INTERMEDIATE, JT_DISTAL, JT_TIP = 1, 2, 3, 4, 5

# MANO's 21 keypoints: wrist, then 4 per finger running base -> tip.
MANO_FINGER_BASE = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

# Which MANUS joint fills each of a finger's 4 MANO slots.  The thumb has one
# fewer phalanx, so its chain is metacarpal/proximal/distal/tip while the others
# are proximal/intermediate/distal/tip.
SLOTS_4 = (JT_PROXIMAL, JT_INTERMEDIATE, JT_DISTAL, JT_TIP)
SLOTS_THUMB = (JT_METACARPAL, JT_PROXIMAL, JT_DISTAL, JT_TIP)


def build_index_map(meta):
    """(chainType, fingerJointType) per node -> indices of the 21 MANO slots.

    Returns a list of 21 node indices, or raises if the glove did not send
    enough of the hand to fill them.
    """
    idx = [None] * 21
    for node_i, (chain, jt) in enumerate(meta):
        if chain == CHAIN_HAND and idx[0] is None:
            idx[0] = node_i
            continue
        finger = CHAIN_FINGER.get(chain)
        if finger is None:
            continue
        slots = SLOTS_THUMB if finger == "thumb" else SLOTS_4
        if jt not in slots:
            continue          # e.g. the metacarpal of a non-thumb finger
        idx[MANO_FINGER_BASE[finger] + slots.index(jt)] = node_i

    if idx[0] is None:                       # no explicit hand/wrist node
        idx[0] = 0                           # MANUS roots the chain at the wrist
    missing = [i for i, v in enumerate(idx) if v is None]
    if missing:
        raise ValueError(f"glove did not provide MANO slots {missing}")
    return idx


def canonical_frame(kp):
    """Express the 21 keypoints in a frame derived from the hand itself.

    DexPilot consumes differences between keypoints, so a global translation
    cancels but a global rotation does not -- and MANUS does not publish in
    MANO's frame.  Anchoring to the hand's own geometry (knuckle line, palm
    direction, palm normal) makes the input orientation-independent.
    """
    wrist, idx_mcp, pinky_mcp, mid_mcp = kp[0], kp[5], kp[17], kp[9]

    x = pinky_mcp - idx_mcp                       # across the knuckles
    y = mid_mcp - wrist                           # out along the hand
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx < 1e-6 or ny < 1e-6:
        raise ValueError("degenerate hand geometry (collapsed keypoints)")
    x = x / nx
    y = y - np.dot(y, x) * x                      # Gram-Schmidt against x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)                            # palm normal

    R = np.stack([x, y, z])                       # world -> hand basis
    return (kp - wrist) @ R.T


class SkeletonRetargeter:
    def __init__(self, low_pass_alpha=None, curl_distal=True):
        from retarget import HandRetargeter
        self.rt = HandRetargeter(low_pass_alpha=low_pass_alpha, curl_distal=curl_distal)
        self.index_map = None
        self.scale = 1.0

    def _learn_layout(self, meta, kp_raw):
        self.index_map = build_index_map(meta)
        kp = kp_raw[self.index_map]
        # MANUS may publish centimetres or millimetres; a human hand spans
        # roughly 0.18 m from wrist to middle fingertip.
        span = float(np.linalg.norm(kp[12] - kp[0]))
        self.scale = 1.0 if span < 1.0 else (0.01 if span < 100 else 0.001)
        print(f"[skel] wrist->middle tip spans {span:.3f} raw units "
              f"-> scale {self.scale} (hand = {span * self.scale:.3f} m)")
        named = {v: k for k, v in MANO_FINGER_BASE.items()}
        print("[skel] MANO slot -> MANUS node index:")
        print(f"        wrist: {self.index_map[0]}")
        for finger, base in MANO_FINGER_BASE.items():
            print(f"        {finger:>6s}: {self.index_map[base:base + 4]}")

    def __call__(self, meta, kp_raw):
        if self.index_map is None:
            self._learn_layout(meta, kp_raw)
        kp = kp_raw[self.index_map] * self.scale
        return self.rt.retarget(canonical_frame(kp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--glove-port", type=int, default=9882)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=51234)
    ap.add_argument("--smooth", type=float, default=0.6, help="EMA on joint targets")
    ap.add_argument("--timeout", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=60.0)
    ap.add_argument("--no-curl", action="store_true",
                    help="disable the curl override on DexPilot's gradient-free distal joints")
    ap.add_argument("--print", dest="show", action="store_true")
    ap.add_argument("--listen-only", action="store_true")
    ap.add_argument("--dump", action="store_true",
                    help="print the node inventory of the first packet and exit")
    args = ap.parse_args()

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.bind, args.glove_port))
    rx.setblocking(False)
    print(f"[skel] listening for glove skeleton on udp://{args.bind}:{args.glove_port}")

    def recv_latest():
        select.select([rx], [], [], 0.005)
        newest = None
        while True:
            try:
                newest = rx.recv(65535)
            except (BlockingIOError, OSError):
                break
        if newest is None:
            return None
        try:
            msg = json.loads(newest.decode("ascii"))
            return np.asarray(msg["skel"], dtype=float), msg["meta"]
        except (ValueError, KeyError, UnicodeDecodeError):
            return None

    if args.dump:
        names = {13: "Hand", 5: "Thumb", 6: "Index", 7: "Middle", 8: "Ring", 9: "Pinky"}
        jts = {1: "Metacarpal", 2: "Proximal", 3: "Intermediate", 4: "Distal", 5: "Tip"}
        print("[skel] waiting for one packet ...")
        while True:
            got = recv_latest()
            if got is None:
                continue
            kp, meta = got
            print(f"\n{len(meta)} nodes:\n")
            print(f"{'node':>5s} {'chain':>10s} {'joint':>13s}   position")
            for i, (c, j) in enumerate(meta):
                print(f"{i:5d} {names.get(c, str(c)):>10s} {jts.get(j, str(j)):>13s}   "
                      f"[{kp[i][0]:8.4f} {kp[i][1]:8.4f} {kp[i][2]:8.4f}]")
            try:
                print("\nderived MANO map:", build_index_map(meta))
            except ValueError as e:
                print(f"\n[!] {e}")
            return

    retarget = SkeletonRetargeter(curl_distal=not args.no_curl)
    tx = send = None
    if not args.listen_only:
        tx, send = make_sender(args.host, args.port)
        print(f"[skel] publishing xhand joints -> udp://{args.host}:{args.port}")
    print(f"[skel] joints: {list(HAND_JOINT_NAMES)}")

    q_ema = None
    seq = 0
    last_rx = next_pub = 0.0
    n_pkt = 0
    t_report = time.time()
    try:
        while True:
            got = recv_latest()
            if got is not None:
                n_pkt += 1
                kp_raw, meta = got
                try:
                    q = retarget(meta, kp_raw)
                except ValueError as e:
                    print(f"[skel][WARN] {e}")
                    continue
                q_ema = q if q_ema is None else q_ema + args.smooth * (q - q_ema)
                last_rx = time.time()

            now = time.time()
            if q_ema is not None and now >= next_pub:
                next_pub = now + 1.0 / args.rate
                if send is not None:
                    send(pack(seq, now, (now - last_rx) < args.timeout, q_ema))
                    seq += 1

            if now - t_report >= 1.0:
                if q_ema is None:
                    print("[skel] waiting for skeleton packets ... "
                          "(Unity in Play with Manus Skeleton Bridge attached?)")
                else:
                    line = (f"[skel] {n_pkt:5d} pkt/s  "
                            f"{'LIVE' if (now - last_rx) < args.timeout else 'STALE'}")
                    if args.show:
                        line += "  " + " ".join(f"{v:5.2f}" for v in q_ema)
                    print(line, flush=True)
                n_pkt = 0
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rx.close()
            if tx is not None:
                tx.close()
        except KeyboardInterrupt:
            pass
        print("\n[skel] stopped.", flush=True)


if __name__ == "__main__":
    main()
