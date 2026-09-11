#!/usr/bin/env python3
"""Print live glove data arriving on udp 9881 — link check before teleop."""
import json
import math
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tp", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "teleop_repose.py"))
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9881
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", port))
sock.settimeout(2.0)
print(f"listening on udp://0.0.0.0:{port} — move your fingers (Ctrl-C to stop)")
n, t0, last = 0, time.time(), {}
while True:
    try:
        data, addr = sock.recvfrom(4096)
    except socket.timeout:
        print("  ... no packets in 2 s")
        continue
    except KeyboardInterrupt:
        break
    try:
        msg = json.loads(data.decode())
    except Exception:
        continue
    human = {}
    if "ergo" in msg:
        for name, v in zip(tp.MANUS_ERGO, msg["ergo"]):
            human[tp.ERGO_ALIAS.get(name, name)] = math.radians(float(v))
    human.update(msg.get("joints", {}))
    last = human or last
    n += 1
    if n % 20 == 0:
        hz = n / max(1e-9, time.time() - t0)
        shown = " ".join(f"{k.split('_')[0][:2]}{k.split('_')[-1][:3]}={math.degrees(v):5.1f}"
                         for k, v in sorted(last.items())[:8])
        print(f"\r{hz:5.1f} Hz from {addr[0]}  {shown}", end="", flush=True)
