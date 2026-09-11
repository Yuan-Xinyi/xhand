#!/usr/bin/env python3
"""Teach the glove->XHand map by showing the operator poses on the real hand.

Every previous attempt guessed something on the MANUS side -- which channel
drives which joint, its sign, its range -- and each guess could only be falsified
by watching the hand move wrong.  Here nothing is guessed:

    1. drive the real XHand to a joint vector this script chose, so the label is
       exact by construction
    2. the operator looks at the hand and matches it with their gloved hand
    3. record a 20-feature vector describing the human hand

Repeat, then fit e -> q.  Assignment, sign, range and cross-coupling all fall
out of the regression.  The correspondence is established by the operator's own
eyes, which is the one judgement neither of us can get wrong.

    python manus_teach.py --out manus_fit.json     # ~5 minutes, hand moves
    python manus_node.py --fit manus_fit.json      # use it

Either Unity bridge will do -- both ports are bound and whichever is actually
streaming gets used.  From ManusUdpBridge the 20 ergonomics channels are taken
as-is; from ManusSkeletonBridge an equivalent 20 features (per finger: two bend
angles, elevation out of the palm plane, direction within it) are computed here
from the 3D nodes, so nothing depends on what MANUS calls its channels.
"""
from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import time

import numpy as np

from manus_skel_node import build_index_map, canonical_frame, forward_kinematics
from protocol import HAND_JOINT_NAMES
from real_node import OPEN_POSE, URDF_LIMITS, XHandSerial, _settle_to

N_CH = 20
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# 4 features per finger, same width as MANUS ergonomics but computed by us from
# the 3D skeleton, so nothing depends on what MANUS calls its channels.
FEAT_LABELS = [f"{f}.{k}" for f in FINGERS
               for k in ("curl1", "curl2", "elev", "azim")]


def skeleton_features(world, meta):
    """21 MANO keypoints -> 20 geometric features, in the hand's own frame."""
    kp = canonical_frame(world[build_index_map(meta)])
    feats = []
    for i, f in enumerate(FINGERS):
        b = kp[1 + 4 * i: 5 + 4 * i]                 # mcp, pip, dip, tip
        v = np.diff(b, axis=0)                        # 3 bone vectors
        n = np.linalg.norm(v, axis=1) + 1e-9
        u = v / n[:, None]
        curl1 = float(np.arccos(np.clip(u[0] @ u[1], -1, 1)))
        curl2 = float(np.arccos(np.clip(u[1] @ u[2], -1, 1)))
        elev = float(np.arcsin(np.clip(u[0][2], -1, 1)))   # out of the palm plane
        azim = float(np.arctan2(u[0][0], u[0][1]))         # within it
        feats += [curl1, curl2, elev, azim]
    return np.asarray(feats)


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


def open_sources(bind, ergo_port, skel_port):
    """Bind both bridges; whichever is actually streaming gets used."""
    out = []
    for port, src in ((ergo_port, "ergo"), (skel_port, "skel")):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((bind, port))
        s.setblocking(False)
        out.append((s, src))
    return out


def drain(socks):
    for s, _ in socks:
        while True:
            try:
                s.recv(65535)
            except (BlockingIOError, OSError):
                break


def decode(payload, source):
    """One datagram -> a 20-vector of features, or None."""
    try:
        msg = json.loads(payload.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        return None
    if source == "ergo":
        e = np.asarray(msg.get("ergo", []), dtype=float)
        return e if e.size == N_CH else None
    local = np.asarray(msg.get("skel", []), dtype=float)
    meta = msg.get("meta", [])
    if local.ndim != 2 or local.shape[1] != 7 or any(len(m) < 4 for m in meta):
        return None
    try:
        world = forward_kinematics(local, msg.get("ids") or [m[2] for m in meta],
                                   [m[3] for m in meta])
        return skeleton_features(world, meta)
    except (ValueError, IndexError):
        return None


def sample(socks, seconds, settle, quiet=False):
    """Mean feature vector over the tail of a hold. Returns (vec, source)."""
    t_end = time.time() + seconds
    keep = time.time() + settle
    rows, src_used = [], None
    while time.time() < t_end:
        select.select([s for s, _ in socks], [], [], 0.02)
        for s, src in socks:
            newest = None
            while True:
                try:
                    newest = s.recv(65535)
                except (BlockingIOError, OSError):
                    break
            if newest is None:
                continue
            e = decode(newest, src)
            if e is not None and time.time() >= keep:
                rows.append(e)
                src_used = src
        if not quiet:
            print(f"\r      保持 ... {t_end - time.time():3.1f}s  ({len(rows)} 帧)",
                  end="", flush=True)
    if not quiet:
        print()
    return (np.mean(rows, axis=0), src_used) if rows else (None, None)


def countdown(socks, n, msg):
    for k in range(n, 0, -1):
        print(f"\r      {msg} {k} ...", end="", flush=True)
        t_end = time.time() + 1.0
        while time.time() < t_end:
            select.select([s for s, _ in socks], [], [], 0.02)
            drain(socks)       # a full UDP buffer drops NEW packets, keeps stale
    print("\r" + " " * 40, end="\r")


def fit(E, Q, alpha):
    """Ridge-regress q = A e + b. Returns (A, b, per-joint R^2).

    Features are standardised first. Ridge penalises every coefficient equally,
    so without this the channels carrying 130 degrees of travel are shrunk far
    harder than those carrying 10 -- the regularisation strength ends up being
    an accident of each channel's units. A and b are folded back afterwards so
    the returned map still consumes raw features.
    """
    mu = E.mean(axis=0)
    sd = E.std(axis=0)
    sd[sd < 1e-9] = 1.0                                # a channel that never moved
    Z = (E - mu) / sd

    X = np.hstack([Z, np.ones((len(Z), 1))])
    reg = alpha * np.eye(X.shape[1])
    reg[-1, -1] = 0.0                                  # never penalise the bias
    W = np.linalg.solve(X.T @ X + reg, X.T @ Q)        # (nfeat+1, 12)

    pred = X @ W
    ss_res = ((Q - pred) ** 2).sum(axis=0)
    ss_tot = ((Q - Q.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    A = (W[:-1] / sd[:, None]).T                       # unstandardise
    b = W[-1] - A @ mu
    return A, b, r2


def pick_alpha(E, Q):
    """Leave-one-out over a log sweep; few poses, so this is cheap and honest."""
    best, best_err = 1.0, np.inf
    for alpha in np.logspace(-3, 4, 36):
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
    ap.add_argument("--ergo-port", type=int, default=9881,
                    help="ManusUdpBridge (20 ergonomics channels)")
    ap.add_argument("--skel-port", type=int, default=9882,
                    help="ManusSkeletonBridge (3D nodes; features computed here)")
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
    ap.add_argument("--refit", default="", metavar="manus_fit.json",
                    help="re-fit from a previous session's saved samples and exit; "
                         "touches neither the robot nor the glove")
    args = ap.parse_args()

    if args.refit:
        with open(args.refit) as fh:
            d = json.load(fh)
        if "E" not in d:
            sys.exit(f"[teach] {args.refit} has no raw samples (written before "
                     f"they were saved); a new session is needed")
        E, Q = np.asarray(d["E"]), np.asarray(d["Q"])
        labels, names, source = d["labels"], d["poses"], d.get("source", "?")
        report_and_save(E, Q, labels, names, source, args.out)
        return

    socks = open_sources(args.bind, args.ergo_port, args.skel_port)
    print(f"[teach] 监听 udp://{args.bind}:{args.ergo_port} (ergonomics) "
          f"和 :{args.skel_port} (skeleton)")
    print("[teach] 等待手套 ... (两条流哪条有数据就用哪条)")
    source = None
    while source is None:
        _, source = sample(socks, 1.0, 0.0, quiet=True)
    print(f"[teach] 手套在线,数据源 = {source}"
          f"{' (20 通道 ergonomics)' if source == 'ergo' else ' (3D 骨架, 特征本地计算)'}\n")

    hand = XHandSerial(args.serial_port, args.baud,
                       kp=args.kp, kd=args.kd, tor_max=args.tor_max)
    print(f"[teach] 串口已开 {args.serial_port}")

    poses = teach_poses()
    print("=" * 68)
    print(f"共 {len(poses)} 个姿势。每个:机械手先摆好 → 你照着摆 → 采 {args.hold:.0f} 秒。")
    print("⚠️  机械手会动。先把 cube 和障碍物拿开。")
    print("=" * 68)
    countdown(socks, 5, "准备开始")

    E, Q, names = [], [], []
    q_prev = OPEN_POSE.copy()
    try:
        for i, (name, q, desc) in enumerate(poses, 1):
            print(f"\n[{i}/{len(poses)}] {name} — {desc}")
            _settle_to(hand, q, q_prev, args.move_time, args.rate)
            q_prev = q
            countdown(socks, 3, "看着机械手,把你的手摆成一样;开始采集前")
            e, _ = sample(socks, args.hold, args.settle)
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
        for s, _ in socks:
            s.close()

    if len(E) < 8:
        sys.exit(f"[teach] 只采到 {len(E)} 个姿势,不足以拟合")

    E, Q = np.asarray(E), np.asarray(Q)
    labels = FEAT_LABELS if source == "skel" else [
        f"{f}.{c}" for f in FINGERS for c in ("spread", "mcp", "pip", "dip")]
    report_and_save(E, Q, labels, names, source, args.out)


def report_and_save(E, Q, labels, names, source, out):
    alpha, loo = pick_alpha(E, Q)
    A, b, r2 = fit(E, Q, alpha)
    print(f"\n[teach] ridge alpha={alpha:.4g}  留一法均方误差={loo:.4f} rad^2 "
          f"({np.sqrt(loo):.3f} rad RMS)")

    print(f"\n{'joint':>15s} {'R^2':>7s}   主导通道(权重最大的三个)")
    for j, jn in enumerate(HAND_JOINT_NAMES):
        top = np.argsort(-np.abs(A[j]))[:3]
        bits = ", ".join(f"{labels[k]}{A[j][k]:+.3f}" for k in top)
        flag = "  <- 拟合差" if r2[j] < 0.6 else ""
        print(f"{jn:>15s} {r2[j]:7.3f}   {bits}{flag}")

    with open(out, "w") as fh:
        json.dump({"A": A.tolist(), "b": b.tolist(), "alpha": float(alpha),
                   "joints": list(HAND_JOINT_NAMES), "poses": names,
                   "source": source, "r2": r2.tolist(),
                   "labels": labels,
                   # raw samples, so the fit can be revisited without asking the
                   # operator to pose nineteen times again
                   "E": E.tolist(), "Q": Q.tolist()}, fh, indent=2)
    print(f"\n[teach] 写入 {out}")
    print(f"[teach] 用它:  python manus_node.py --fit {out} --listen-only --print")


if __name__ == "__main__":
    main()
