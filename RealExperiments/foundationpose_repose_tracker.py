#!/usr/bin/env python3
"""FoundationPose continuous cube tracker -> UDP pose stream (for repose sim2real).

Runs in env_isaaclab. Registers once (popup ROI box, or --roi), then tracks every
frame and publishes camera_T_cube to udp://127.0.0.1:9877 as 18 float64:
[seq, unix_time, pose.flatten() (16)].

Normally spawned by foundationpose_repose_real.py; can also run standalone:
    conda activate env_isaaclab
    python RealExperiments/foundationpose_repose_tracker.py
"""
import argparse
import os
import socket
import struct
import sys
import time

# Isaac's omni.pip.compute prebundle ships a HEADLESS cv2 that shadows the env's
# GUI build. Demote that path to the END of sys.path: cv2 then resolves from
# site-packages (GUI), while prebundle-only packages (trimesh) still resolve.
_demoted = [p for p in sys.path if "omni.pip.compute" in p]
sys.path = [p for p in sys.path if p not in _demoted] + _demoted

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FP_CUBE_DIR = "/disk2/FoundationPose/cube"
sys.path.insert(0, FP_CUBE_DIR)
import live_demo  # noqa: E402  (build_estimator, annotate, grabcut, set_*)

UDP_ADDR = ("127.0.0.1", 9877)
POSE_FMT = "<18d"


def gui_available() -> bool:
    try:
        cv2.namedWindow("_probe")
        cv2.destroyWindow("_probe")
        return True
    except cv2.error:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh_file", default=os.path.join(FP_CUBE_DIR, "mesh", "textured.obj"))
    ap.add_argument("--serial", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--est_refine_iter", type=int, default=5)
    ap.add_argument("--track_refine_iter", type=int, default=2)
    ap.add_argument("--roi", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--no-view", action="store_true", help="no live overlay window")
    ap.add_argument("--port", type=int, default=UDP_ADDR[1])
    args = ap.parse_args()

    import pyrealsense2 as rs
    live_demo.set_logging_format()
    live_demo.set_seed(0)
    est, mesh, to_origin, bbox, mt = live_demo.build_estimator(args.mesh_file)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    for _ in range(15):
        pipeline.wait_for_frames()

    def grab():
        fr = align.process(pipeline.wait_for_frames())
        c = np.asarray(fr.get_color_frame().get_data())
        d = np.asarray(fr.get_depth_frame().get_data()).astype(np.float32) * depth_scale
        intr = fr.get_color_frame().profile.as_video_stream_profile().intrinsics
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        return c, d, K

    gui = gui_available() and not args.no_view

    color, depth, K = grab()
    if args.roi is not None:
        rect = tuple(args.roi)
    elif gui:
        bgr = color[..., ::-1].copy()
        rect = cv2.selectROI("tracker init: drag box around cube, ENTER", bgr, showCrosshair=False)
        cv2.destroyWindow("tracker init: drag box around cube, ENTER")
        if rect[2] == 0 or rect[3] == 0:
            print("[tracker] empty ROI, abort")
            return
    else:
        raise RuntimeError("no cv2 GUI and no --roi given; pass --roi X Y W H")

    mask = live_demo.grabcut(color[..., ::-1].copy(), rect) > 0
    print(f"[tracker] ROI={rect}, registering...", flush=True)
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    print("[tracker] registered, tracking + publishing", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (UDP_ADDR[0], args.port)
    win = "FoundationPose tracker (q=quit)"
    seq, t_prev, fps = 0, time.time(), 0.0
    try:
        while True:
            color, depth, K = grab()
            pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
            seq += 1
            sock.sendto(struct.pack(POSE_FMT, float(seq), time.time(), *np.asarray(pose, dtype=np.float64).ravel()), addr)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - t_prev))
            t_prev = now
            if seq % 90 == 0:
                t = pose[:3, 3]
                print(f"[tracker] seq {seq} xyz(m): {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}  {fps:4.1f} FPS", flush=True)
            if gui:
                vis = live_demo.annotate(color, pose, K, to_origin, bbox, mt, fps=fps)
                cv2.imshow(win, vis[..., ::-1])
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if gui:
            cv2.destroyAllWindows()
        print("[tracker] bye", flush=True)


if __name__ == "__main__":
    main()
