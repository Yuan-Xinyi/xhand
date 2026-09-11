#!/usr/bin/env python3
"""Measure what the MANUS channels actually do, instead of trusting the enum.

Walks you through a handful of named poses, samples the 20 raw ergonomics
channels during each, and prints which channel moved for which pose plus the
range you actually reach.  That is enough to fix both failure modes we hit:
a channel wired to the wrong joint, and a nominal range that does not match
your hand.

Run it with the Unity bridge streaming (Unity in Play mode):

    python manus_calibrate.py                  # guided capture
    python manus_calibrate.py --out cal.json   # also save the raw numbers

Nothing touches the robot -- this only listens.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "RealExperiments", "manus_bridge"))
from manus_csv_to_xhand import CHANNELS, FINGERS  # noqa: E402

ERGO_KEYS = [(f, c) for f in FINGERS for c in CHANNELS]
LABELS = [f"{f}.{c}" for f, c in ERGO_KEYS]

# Each pose isolates something we need to know.  Keep the hand still while the
# countdown runs; only the last second is averaged.
POSES = [
    ("flat", "手掌摊平,五指伸直并拢,拇指自然放在手掌边缘"),
    ("fist", "握拳,四指全部蜷起,拇指压在四指外面"),
    ("fingers_spread", "四指伸直并最大张开(手指之间分开)"),
    ("thumb_out", "四指伸直并拢,拇指在手掌平面内最大张开(远离食指)"),
    ("thumb_across", "四指伸直并拢,拇指横过掌心指向小指根部"),
    ("thumb_up", "四指伸直并拢,拇指垂直抬起离开手掌平面"),
    ("thumb_curl", "拇指根部不动,只把拇指最后两节尽量蜷曲"),
]


def drain(rx):
    """Throw away anything queued.  Never let the socket buffer fill: once it is
    full the kernel drops the NEW datagrams and keeps the stale ones."""
    while True:
        try:
            rx.recv(4096)
        except (BlockingIOError, OSError):
            return


def countdown(rx, n):
    """Count down while keeping the socket drained."""
    for k in range(n, 0, -1):
        print(f"\r    {k} ...", end="", flush=True)
        t_end = time.time() + 1.0
        while time.time() < t_end:
            select.select([rx], [], [], 0.02)
            drain(rx)


def sample(rx, seconds, settle):
    """Collect channel vectors for `seconds`, return the mean of the tail."""
    t_end = time.time() + seconds
    keep_after = time.time() + settle
    rows = []
    last_print = 0.0
    while time.time() < t_end:
        select.select([rx], [], [], 0.02)
        newest = None
        while True:
            try:
                newest = rx.recv(4096)
            except (BlockingIOError, OSError):
                break
        if newest is not None:
            try:
                e = np.asarray(json.loads(newest.decode("ascii"))["ergo"], dtype=float)
            except (ValueError, KeyError, UnicodeDecodeError):
                continue
            if e.size == 20 and time.time() >= keep_after:
                rows.append(e)
        left = t_end - time.time()
        if left - last_print < -0.5 or last_print == 0.0:
            print(f"\r    保持姿势 ... {left:3.1f}s  (已采 {len(rows)} 帧)", end="", flush=True)
            last_print = left
    print()
    if not rows:
        return None
    return np.mean(np.asarray(rows), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--glove-port", type=int, default=9881)
    ap.add_argument("--hold", type=float, default=4.0, help="seconds per pose")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds to ignore at the start of each pose while you move into it")
    ap.add_argument("--out", default="", help="write raw per-pose channel values as JSON")
    args = ap.parse_args()

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.bind, args.glove_port))
    rx.setblocking(False)

    print(f"[cal] 监听 udp://{args.bind}:{args.glove_port}")
    print("[cal] 等待 Unity 的手套数据 ...")
    while sample(rx, 1.0, 0.0) is None:
        pass
    print("[cal] 收到数据。\n")

    print("=" * 70)
    print("接下来会依次提示 7 个姿势,每个保持 %.0f 秒(前 %.0f 秒用来摆好,不计入)。"
          % (args.hold, args.settle))
    print("=" * 70)

    captured = {}
    for i, (name, desc) in enumerate(POSES, 1):
        print(f"\n[{i}/{len(POSES)}] {name}")
        print(f"    → {desc}")
        countdown(rx, 3)
        print("\r    开始采集       ")
        v = sample(rx, args.hold, args.settle)
        if v is None:                     # one retry before giving up on the pose
            print("    [!] 没采到数据,重试一次 ...")
            v = sample(rx, args.hold, args.settle)
        if v is None:
            print("    [!] 仍然没有数据 —— Unity 还在 Play 吗?")
            continue
        captured[name] = v

    rx.close()
    if len(captured) < 2:
        sys.exit("[cal] 采到的姿势太少,无法分析")

    mat = np.asarray([captured[k] for k in captured])
    names = list(captured)

    print("\n" + "=" * 70)
    print("各通道实测范围(跨全部姿势)")
    print("=" * 70)
    print(f"{'channel':>22s} {'min':>8s} {'max':>8s} {'span':>8s}")
    lo_all, hi_all = mat.min(axis=0), mat.max(axis=0)
    for i, lab in enumerate(LABELS):
        span = hi_all[i] - lo_all[i]
        flag = "   <- 基本不动" if span < 5 else ""
        print(f"{lab:>22s} {lo_all[i]:8.1f} {hi_all[i]:8.1f} {span:8.1f}{flag}")

    print("\n" + "=" * 70)
    print("拇指通道逐姿势取值(定方向用这个)")
    print("=" * 70)
    thumb_idx = [i for i, (f, _) in enumerate(ERGO_KEYS) if f == "thumb"]
    print(f"{'pose':>16s} " + " ".join(f"{LABELS[i].split('.')[1]:>12s}" for i in thumb_idx))
    for k in names:
        print(f"{k:>16s} " + " ".join(f"{captured[k][i]:12.1f}" for i in thumb_idx))

    print("\n" + "=" * 70)
    print("每个姿势相对 flat 变化最大的通道")
    print("=" * 70)
    if "flat" in captured:
        base = captured["flat"]
        for k in names:
            if k == "flat":
                continue
            d = captured[k] - base
            order = np.argsort(-np.abs(d))[:4]
            bits = ", ".join(f"{LABELS[i]}{d[i]:+.0f}" for i in order if abs(d[i]) > 3)
            print(f"{k:>16s}  {bits or '(无明显变化)'}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"labels": LABELS,
                       "poses": {k: v.tolist() for k, v in captured.items()}}, fh, indent=2)
        print(f"\n[cal] 原始数据 -> {args.out}")
    print("\n把上面三张表贴给我,我据此改映射表和量程。")


if __name__ == "__main__":
    main()
