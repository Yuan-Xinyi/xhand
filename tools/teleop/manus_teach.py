#!/usr/bin/env python3
"""Teach the glove->XHand map by showing the operator poses on the real hand.

Every previous attempt guessed something on the MANUS side -- which channel
drives which joint, its sign, its range -- and each guess could only be falsified
by watching the hand move wrong.  Here nothing is guessed:

    1. drive the real XHand to a joint vector this script chose, so the label is
       exact by construction
    2. the operator looks at the hand and matches it with their gloved hand
    3. record the 20 ergonomics channels

Repeat, then fit e -> q.  Assignment, sign, range and cross-coupling all fall
out of the regression.  The correspondence is established by the operator's own
eyes, which is the one judgement neither of us can get wrong.

    python manus_teach.py --out manus_fit.json     # ~5 minutes, hand moves
    python manus_node.py --fit manus_fit.json      # use it

Needs the ergonomics bridge (ManusUdpBridge, udp 9881) streaming.  Both Unity
bridges can run at once -- they use different ports.
"""
from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import time

import numpy as np

from protocol import HAND_JOINT_NAMES
from real_node import OPEN_POSE, URDF_LIMITS, XHandSerial, _settle_to

N_CH = 20
LO, HI = URDF_LIMITS[:, 0], URDF_LIMITS[:, 1]


def frac(**kw):
    """Joint vector from fractions of each joint's travel; unnamed joints open."""
    q = OPEN_POSE.copy()
    for name, f in kw.items():
        j = HAND_JOINT_NAMES.index(name)
        q[j] = LO[j] + f * (HI[j] - LO[j])
    return q


# Poses are chosen to excite each degree of freedom on its own, so the fit can
# tell them apart, plus a few combined ones so it sees realistic coupling.
def teach_poses():
    P = []
    P.append(("open", OPEN_POSE.copy(), "五指自然张开"))
    P.append(("fist", frac(index_joint1=.85, index_joint2=.85, middle_joint0=.85,
                           middle_joint1=.85, ring_joint0=.85, ring_joint1=.85,
                           pinky_joint0=.85, pinky_joint1=.85, thumb_joint0=.7,
                           thumb_joint2=.7), "握拳"))
    for f, (a, b) in (("index", ("index_joint1", "index_joint2")),
                      ("middle", ("middle_joint0", "middle_joint1")),
                      ("ring", ("ring_joint0", "ring_joint1")),
                      ("pinky", ("pinky_joint0", "pinky_joint1"))):
        P.append((f"{f}_curl", frac(**{a: .85, b: .85}), f"只弯{f},其余伸直"))
    P.append(("half_curl", frac(index_joint1=.45, index_joint2=.45, middle_joint0=.45,
                                middle_joint1=.45, ring_joint0=.45, ring_joint1=.45,
                                pinky_joint0=.45, pinky_joint1=.45), "四指半弯"))
    P.append(("index_spread_a", frac(index_joint0=.0), "食指侧摆到一端"))
    P.append(("index_spread_b", frac(index_joint0=1.), "食指侧摆到另一端"))
    P.append(("thumb_0_lo", frac(thumb_joint0=.0), "拇指根部转到一端"))
    P.append(("thumb_0_hi", frac(thumb_joint0=1.), "拇指根部转到另一端"))
    P.append(("thumb_1_lo", frac(thumb_joint1=.0), "拇指第二关节到一端"))
    P.append(("thumb_1_hi", frac(thumb_joint1=1.), "拇指第二关节到另一端"))
    P.append(("thumb_2_hi", frac(thumb_joint2=1.), "拇指指尖蜷曲"))
    P.append(("thumb_opp", frac(thumb_joint0=.8, thumb_joint1=.5, thumb_joint2=.6),
              "拇指对掌(像要捏东西)"))
    # A few mixed poses: 21 parameters off 15 samples leaves the fit thin, and
    # these cost the operator seconds each.
    P.append(("pinch", frac(thumb_joint0=.8, thumb_joint1=.5, thumb_joint2=.55,
                            index_joint1=.5, index_joint2=.6), "拇指和食指捏合"))
    P.append(("point", frac(middle_joint0=.85, middle_joint1=.85, ring_joint0=.85,
                            ring_joint1=.85, pinky_joint0=.85, pinky_joint1=.85,
                            thumb_joint0=.6), "只伸食指,其余握起"))
    P.append(("hook", frac(index_joint2=.8, middle_joint1=.8, ring_joint1=.8,
                           pinky_joint1=.8), "四指第二节弯,根部伸直(勾状)"))
    P.append(("quarter_curl", frac(index_joint1=.25, index_joint2=.25,
                                   middle_joint0=.25, middle_joint1=.25,
                                   ring_joint0=.25, ring_joint1=.25,
                                   pinky_joint0=.25, pinky_joint1=.25), "四指轻微弯曲"))
    return P


def drain(rx):
    while True:
        try:
            rx.recv(4096)
        except (BlockingIOError, OSError):
            return


def sample(rx, seconds, settle):
    """Mean of the ergonomics channels over the tail of a hold."""
    t_end = time.time() + seconds
    keep = time.time() + settle
    rows = []
    while time.time() < t_end:
        select.select([rx], [], [], 0.02)
        newest = None
        while True:
            try:
                newest = rx.recv(4096)
            except (BlockingIOError, OSError):
                break
        if newest is None:
            continue
        try:
            e = np.asarray(json.loads(newest.decode("ascii"))["ergo"], dtype=float)
        except (ValueError, KeyError, UnicodeDecodeError):
            continue
        if e.size == N_CH and time.time() >= keep:
            rows.append(e)
        left = t_end - time.time()
        print(f"\r      保持 ... {left:3.1f}s  ({len(rows)} 帧)", end="", flush=True)
    print()
    return np.mean(rows, axis=0) if rows else None


def countdown(rx, n, msg):
    for k in range(n, 0, -1):
        print(f"\r      {msg} {k} ...", end="", flush=True)
        t_end = time.time() + 1.0
        while time.time() < t_end:
            select.select([rx], [], [], 0.02)
            drain(rx)          # a full UDP buffer drops NEW packets, keeps stale
    print("\r" + " " * 40, end="\r")


def fit(E, Q, alpha):
    """Ridge-regress q = A e + b. Returns (A, b, per-joint R^2)."""
    X = np.hstack([E, np.ones((len(E), 1))])          # bias column
    reg = alpha * np.eye(X.shape[1])
    reg[-1, -1] = 0.0                                  # never penalise the bias
    W = np.linalg.solve(X.T @ X + reg, X.T @ Q)        # (21, 12)
    pred = X @ W
    ss_res = ((Q - pred) ** 2).sum(axis=0)
    ss_tot = ((Q - Q.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return W[:-1].T, W[-1], r2


def pick_alpha(E, Q):
    """Leave-one-out over a log sweep; few poses, so this is cheap and honest."""
    best, best_err = 1.0, np.inf
    for alpha in np.logspace(-4, 3, 29):
        err = 0.0
        for i in range(len(E)):
            m = np.ones(len(E), bool)
            m[i] = False
            A, b, _ = fit(E[m], Q[m], alpha)
            err += float(((A @ E[i] + b - Q[i]) ** 2).sum())
        if err < best_err:
            best, best_err = alpha, err
    return best, best_err / len(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--glove-port", type=int, default=9881)
    ap.add_argument("--out", default="manus_fit.json")
    ap.add_argument("--hold", type=float, default=3.5, help="seconds recorded per pose")
    ap.add_argument("--settle", type=float, default=1.5, help="ignored while you get set")
    ap.add_argument("--move-time", type=float, default=1.5, help="robot ramp time")
    ap.add_argument("--serial-port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=3_000_000)
    ap.add_argument("--rate", type=int, default=50)
    ap.add_argument("--kp", type=int, default=100)
    ap.add_argument("--kd", type=int, default=10)
    ap.add_argument("--tor-max", type=int, default=150)
    args = ap.parse_args()

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.bind, args.glove_port))
    rx.setblocking(False)
    print(f"[teach] 监听手套数据 udp://{args.bind}:{args.glove_port}")
    print("[teach] 等待手套 ...")
    while sample(rx, 1.0, 0.0) is None:
        pass
    print("[teach] 手套在线。\n")

    hand = XHandSerial(args.serial_port, args.baud,
                       kp=args.kp, kd=args.kd, tor_max=args.tor_max)
    print(f"[teach] 串口已开 {args.serial_port}")

    poses = teach_poses()
    print("=" * 68)
    print(f"共 {len(poses)} 个姿势。每个:机械手先摆好 → 你照着摆 → 采 {args.hold:.0f} 秒。")
    print("⚠️  机械手会动。先把 cube 和障碍物拿开。")
    print("=" * 68)
    countdown(rx, 5, "准备开始")

    E, Q, names = [], [], []
    q_prev = OPEN_POSE.copy()
    try:
        for i, (name, q, desc) in enumerate(poses, 1):
            print(f"\n[{i}/{len(poses)}] {name} — {desc}")
            _settle_to(hand, q, q_prev, args.move_time, args.rate)
            q_prev = q
            countdown(rx, 3, "看着机械手,把你的手摆成一样;开始采集前")
            e = sample(rx, args.hold, args.settle)
            if e is None:
                print("      [!] 没采到手套数据,跳过")
                continue
            E.append(e)
            Q.append(q)
            names.append(name)
        print("\n[teach] 采集完成,机械手回到张开位。")
        _settle_to(hand, OPEN_POSE, q_prev, 2.0, args.rate)
    finally:
        try:
            hand.close()
        except (KeyboardInterrupt, OSError):
            pass
        rx.close()

    if len(E) < 8:
        sys.exit(f"[teach] 只采到 {len(E)} 个姿势,不足以拟合")

    E, Q = np.asarray(E), np.asarray(Q)
    alpha, loo = pick_alpha(E, Q)
    A, b, r2 = fit(E, Q, alpha)
    print(f"\n[teach] ridge alpha={alpha:.4g}  留一法均方误差={loo:.4f} rad^2 "
          f"({np.sqrt(loo):.3f} rad RMS)")

    print(f"\n{'joint':>15s} {'R^2':>7s}   主导通道(权重最大的三个)")
    labels = [f"{f}.{c}" for f in ("thumb", "index", "middle", "ring", "pinky")
              for c in ("spread", "mcp", "pip", "dip")]
    for j, jn in enumerate(HAND_JOINT_NAMES):
        top = np.argsort(-np.abs(A[j]))[:3]
        bits = ", ".join(f"{labels[k]}{A[j][k]:+.3f}" for k in top)
        flag = "  <- 拟合差" if r2[j] < 0.6 else ""
        print(f"{jn:>15s} {r2[j]:7.3f}   {bits}{flag}")

    with open(args.out, "w") as fh:
        json.dump({"A": A.tolist(), "b": b.tolist(), "alpha": float(alpha),
                   "joints": list(HAND_JOINT_NAMES), "poses": names,
                   "r2": r2.tolist()}, fh, indent=2)
    print(f"\n[teach] 写入 {args.out}")
    print(f"[teach] 用它:  python manus_node.py --fit {args.out} --listen-only --print")


if __name__ == "__main__":
    main()
