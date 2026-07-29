# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop PERCEPTION node — run in the `wilor` conda env.

    webcam frame (cv2)
      -> WiLoR: image -> 21 MANO 3D keypoints
        -> DexPilot retarget -> 12 xhand joint targets
          -> UDP publish to the Isaac Lab sim driver (teleop_sim.py)

Decoupled from Isaac Lab on purpose: this process owns a torch 2.5 / WiLoR stack
that would downgrade Isaac Lab's torch if installed into the same env. They talk
only over localhost UDP (see protocol.py).

Usage (conda activate wilor):
    cd tools/teleop
    python perception_node.py --camera 0 --show          # live, with overlay window
    python perception_node.py --camera 0 --hand right     # headless publisher
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from camera import open_camera
from filters import OneEuroArray
from hand_estimator import WiLoREstimator, mirror_to_right
from protocol import make_sender, pack
from retarget import HandRetargeter

def _pick_hand(hands, want_right: bool, mirror: bool):
    """Choose one detected hand and return its (21,3) keypoints as a right hand.

    Prefers a hand matching the requested handedness; with --mirror, accepts the
    opposite hand and reflects it into the right-hand frame. Returns (kp3d, hand)
    or (None, None).
    """
    if not hands:
        return None, None
    # highest-score hand of the wanted handedness, else highest-score overall
    match = [h for h in hands if h.is_right == want_right]
    pool = match if match else (hands if mirror else [])
    if not pool:
        return None, None
    hand = max(pool, key=lambda h: h.score)
    kp = hand.keypoints_3d
    if hand.is_right != want_right and mirror:
        kp = mirror_to_right(kp)
    return kp, hand


def main():
    ap = argparse.ArgumentParser(description="WiLoR -> xhand teleop perception node.")
    ap.add_argument("--source", choices=["webcam", "realsense"], default="realsense",
                    help="frame source; 'realsense' uses the Intel RealSense D435 color stream")
    ap.add_argument("--camera", default="0", help="cv2 camera index or /dev/videoN (webcam source)")
    ap.add_argument("--fps", type=int, default=60, help="requested capture FPS (realsense)")
    ap.add_argument("--rs-serial", default=None,
                    help="RealSense device serial or name substring (e.g. 'D435'); "
                         "default prefers the plain D435 over the D435IF")
    ap.add_argument("--list-cameras", action="store_true", help="list RealSense devices and exit")
    ap.add_argument("--redetect-interval", type=int, default=12,
                    help="frames between full YOLO re-detections; lower = more robust, slower")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=51234)
    ap.add_argument("--hand", choices=["right", "left"], default="right",
                    help="which physical hand you teleop with")
    ap.add_argument("--mirror", action="store_true",
                    help="accept the opposite hand and mirror it into the right frame")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--show", action="store_true", help="open a cv2 overlay window")
    ap.add_argument("--flip", action="store_true", help="horizontally flip (selfie/mirror view)")
    # smoothing / latency knobs
    ap.add_argument("--min-cutoff", type=float, default=1.5,
                    help="One-Euro min cutoff: lower = steadier when still (more lag)")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="One-Euro beta: higher = snappier during fast motion (less lag)")
    ap.add_argument("--no-filter", action="store_true", help="disable One-Euro keypoint filtering")
    ap.add_argument("--retarget-alpha", type=float, default=0.8,
                    help="DexPilot temporal low-pass (0..1); higher = less retargeter lag")
    ap.add_argument("--no-distal-curl", action="store_true",
                    help="disable human-curl drive of the 5 DexPilot-dead distal joints "
                         "(they then sit at their limit midpoints, RL-distillation style)")
    args = ap.parse_args()

    if args.list_cameras:
        from camera import list_realsense
        devs = list_realsense()
        print(f"[perception] {len(devs)} RealSense device(s):")
        for name, serial in devs:
            print(f"    {name}  serial={serial}")
        return

    want_right = args.hand == "right"

    cap = open_camera(args.source, args.camera, args.width, args.height, args.fps, args.rs_serial)
    if not cap.is_opened():
        raise SystemExit(f"[perception] cannot open {args.source} camera {args.camera!r}")
    print(f"[perception] source = {args.source}")

    print("[perception] loading WiLoR ...")
    est = WiLoREstimator(device="cuda", fp16=True, redetect_interval=args.redetect_interval)
    rt = HandRetargeter(low_pass_alpha=args.retarget_alpha,
                        curl_distal=not args.no_distal_curl)
    rt.reset()
    euro = None if args.no_filter else OneEuroArray(min_cutoff=args.min_cutoff, beta=args.beta)
    t_prev = time.time()
    print(f"[perception] output joints (wire order): {rt.output_joint_names}")

    dash = None
    if args.show:
        from viz_panel import TeleopDashboard
        dash = TeleopDashboard(rt.joint_limits, cam_size=(args.width, args.height))

    sock, send = make_sender(args.host, args.port)
    print(f"[perception] publishing xhand joints -> udp://{args.host}:{args.port}")

    seq = 0
    ema_fps = 0.0
    q_hold = np.zeros(12, dtype=np.float32)
    try:
        while True:
            t0 = time.time()
            ok, bgr = cap.read()
            if not ok:
                continue
            if args.flip:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            hands = est.estimate(rgb)
            kp3d, hand = _pick_hand(hands, want_right, args.mirror)
            valid = kp3d is not None
            now = time.time()
            if valid:
                if euro is not None:
                    kp3d = euro(kp3d, now - t_prev)
                q_hold = rt.retarget(kp3d)
            elif euro is not None:
                euro.reset()  # drop stale state so re-acquire doesn't snap through old pose
            t_prev = now
            # when no hand: resend last good target with valid=0 (sim holds pose)
            send(pack(seq, time.time(), valid, q_hold))
            seq += 1

            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            ema_fps = fps if ema_fps == 0.0 else 0.9 * ema_fps + 0.1 * fps
            if args.show:
                canvas = dash.render(bgr, hand, kp3d, q_hold, ema_fps, valid)
                cv2.imshow("xhand teleop (q to quit)", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif seq % 30 == 0:
                print(f"[perception] {ema_fps:5.1f} FPS  valid={valid}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        sock.close()
        print("[perception] stopped.")


if __name__ == "__main__":
    main()
