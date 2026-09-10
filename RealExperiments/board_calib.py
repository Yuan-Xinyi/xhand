#!/usr/bin/env python3
"""Table-board camera calibration: one 20 cm ArUco at a KNOWN base pose.

base_T_marker is known by construction (board measured on the table):
    position (--board-x, --board-y, --board-z), yaw --board-yaw deg CCW about z
so a single robust detection gives base_T_cam = base_T_marker @ inv(cam_T_marker).
No arm motion, no FK in the loop; a 20 cm marker at ~0.7 m has ~4x the corner
baseline of the palm sticker.

Run:
    /home/lqin/miniconda3/envs/one/bin/python RealExperiments/board_calib.py
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foundationpose_repose_real as rr  # path sanitize + math helpers

import cv2
import numpy as np
import yaml

OUT_YAML = os.path.join(HERE, "palm_env_T_cam.yaml")


def tf(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--size", type=float, default=0.20, help="board marker side [m]")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--board-x", type=float, default=0.369)
    ap.add_argument("--board-y", type=float, default=0.0)
    ap.add_argument("--board-z", type=float, default=0.0)
    ap.add_argument("--board-yaw", type=float, default=90.0, help="deg CCW about base z")
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    yaw = np.radians(args.board_yaw)
    base_T_marker = tf(
        np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]),
        [args.board_x, args.board_y, args.board_z])

    import pyrealsense2 as rs
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(adict, params)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)
    for _ in range(10):
        pipeline.wait_for_frames()

    s = args.size / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
    frames_sol, K, vis = [], None, None
    try:
        for _ in range(args.frames):
            cf = pipeline.wait_for_frames().get_color_frame()
            if K is None:
                intr = cf.profile.as_video_stream_profile().intrinsics
                K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
            color = np.asarray(cf.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or args.id not in ids.ravel():
                continue
            c = corners[list(ids.ravel()).index(args.id)]
            n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj, c.reshape(4, 2).astype(np.float64), K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if n_sol < 1:
                continue
            sols = []
            for rv, tv in zip(rvecs, tvecs):
                rm, _ = cv2.Rodrigues(rv)
                sols.append((rr.rotmat_to_quat(rm), tv.ravel()))
            frames_sol.append(sols)
            if vis is None:
                vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.aruco.drawDetectedMarkers(vis, [c])
                rm0, _ = cv2.Rodrigues(rvecs[0])
                cv2.drawFrameAxes(vis, K, np.zeros(5), rvecs[0], tvecs[0], args.size * 0.6)
    finally:
        pipeline.stop()

    n = len(frames_sol)
    print(f"[board] marker detected in {n}/{args.frames} frames")
    if n < 15:
        print("[board] TOO FEW detections — is the board in view / unoccluded?")
        sys.exit(1)

    # cross-frame consensus over the two IPPE branches (same armor as handeye)
    sel = [0] * n
    for _ in range(4):
        qs = []
        for fs, si in zip(frames_sol, sel):
            q = fs[si][0].copy()
            if qs and np.dot(q, qs[0]) < 0:
                q = -q
            qs.append(q)
        qm = np.mean(qs, axis=0)
        qm /= np.linalg.norm(qm)
        sel = [int(np.argmax([abs(np.dot(sol[0], qm)) for sol in fs])) for fs in frames_sol]
    qs, ts = [], []
    for fs, si in zip(frames_sol, sel):
        q = fs[si][0].copy()
        if qs and np.dot(q, qs[0]) < 0:
            q = -q
        qs.append(q)
        ts.append(fs[si][1])
    qm = np.mean(qs, axis=0)
    qm /= np.linalg.norm(qm)
    spread_rot = np.degrees(2 * np.arccos(np.clip(np.abs(np.array(qs) @ qm), -1, 1))).max()
    spread_pos = float(np.linalg.norm(np.array(ts) - np.mean(ts, axis=0), axis=1).max() * 1000)
    cam_T_marker = tf(rr.quat_to_rotmat(qm), np.mean(ts, axis=0))
    print(f"[board] frame spread: rot {spread_rot:.2f} deg, pos {spread_pos:.1f} mm,"
          f" marker at cam-dist {np.linalg.norm(cam_T_marker[:3, 3]):.3f} m")
    if spread_rot > 1.0:
        print("[board][WARN] detection unstable (>1 deg spread) — check lighting / board flatness")

    base_T_cam = base_T_marker @ np.linalg.inv(cam_T_marker)
    cam_pos = base_T_cam[:3, 3]
    cam_axis = base_T_cam[:3, :3] @ np.array([0, 0, 1.0])
    print(f"[board] camera in base: pos {np.round(cam_pos, 3)} m, optical axis {np.round(cam_axis, 3)}")
    print("[board] sanity: camera above the table (z>0.3)? axis pointing down-ish (z<0)?")
    if cam_pos[2] < 0.2 or cam_axis[2] > -0.2:
        print("[board][WARN] geometry looks implausible — board yaw/offset sign wrong?")

    if os.path.exists(OUT_YAML):
        os.replace(OUT_YAML, OUT_YAML + ".bak")
        print(f"[board] previous calibration backed up -> {OUT_YAML}.bak")
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump({
            "base_T_cam": {"matrix": base_T_cam.tolist()},
            "method": "table_board_20cm",
            "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames_used": int(n),
            "residuals": {"rot_deg_max": float(spread_rot), "pos_mm_max": float(spread_pos)},
            "board": {"dict": args.dict, "id": int(args.id), "size_m": float(args.size),
                      "base_pos": [args.board_x, args.board_y, args.board_z],
                      "yaw_deg": float(args.board_yaw)},
        }, f)
    if vis is not None:
        cv2.imwrite("/tmp/board_detect.png", vis)
        print("[board] detection overlay -> /tmp/board_detect.png")
    print(f"[board] saved -> {OUT_YAML} (pipeline auto-uses it)")


if __name__ == "__main__":
    main()
