#!/usr/bin/env python3
"""Minimal receiver for the MANUS glove stream from Unity. Standard library only.

Reproduces the Windows->Ubuntu link on its own, with no dependency on the rest
of this repository -- run it to confirm the bridge works before wiring anything
downstream.

    python3 manus_recv_example.py
    python3 manus_recv_example.py --port 9881 --raw
"""
import argparse
import json
import socket
import time

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHANNELS = ("MCPSpread", "MCPStretch", "PIPStretch", "DIPStretch")
LABELS = [f"{f}.{c}" for f in FINGERS for c in CHANNELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0",
                    help="0.0.0.0 receives both unicast and subnet broadcast")
    ap.add_argument("--port", type=int, default=9881)
    ap.add_argument("--raw", action="store_true", help="print all 20 channels")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))
    sock.settimeout(1.0)
    print(f"listening on udp://{args.bind}:{args.port}", flush=True)

    n, t0 = 0, time.time()
    while True:
        try:
            payload, addr = sock.recvfrom(4096)
        except socket.timeout:
            print("  no packets (is Unity in Play mode?)", flush=True)
            continue
        try:
            ergo = json.loads(payload.decode("ascii"))["ergo"]
        except (ValueError, KeyError, UnicodeDecodeError):
            continue
        n += 1

        now = time.time()
        if now - t0 >= 1.0:
            print(f"\n{n} pkt/s from {addr[0]}", flush=True)
            if args.raw:
                for i in range(0, 20, 4):
                    print("   " + "  ".join(f"{LABELS[i + k]:>16s}={ergo[i + k]:7.2f}"
                                            for k in range(4)), flush=True)
            else:
                # one number per finger: how bent it is
                print("   " + "  ".join(f"{f}={ergo[4 * i + 2]:6.1f}deg"
                                        for i, f in enumerate(FINGERS)), flush=True)
            n, t0 = 0, now


if __name__ == "__main__":
    main()
