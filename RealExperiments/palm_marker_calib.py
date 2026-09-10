#!/usr/bin/env python3
"""Palm-marker extrinsic calibration (ArUco on the palm center).

Measures env_T_cam directly: the env frame is DEFINED by the palm (sim palm
pose is fixed), so a marker stuck on the palm with a known orientation gives
the full camera extrinsic in one shot — no arm FK, no old extrinsic yaml.

Marker convention (IMPORTANT):
  - dictionary 4X4_50, id 0 (defaults), stuck FLAT on the palm center
  - TOP edge of the marker points TOWARD THE FINGERS
  -> marker axes == env axes (x right/thumb side, y fingers, z out of palm)

Translation accuracy barely matters (auto-center re-anchors it); the payload
is ROTATION, which the sliders could never measure properly.

Usage:
    conda activate env_isaaclab
    python RealExperiments/palm_marker_calib.py --make-marker   # print this PNG
    #   stick it on the palm, hand at palm-up home pose, then:
    python RealExperiments/palm_marker_calib.py --size 0.04     # measured side!
"""
import argparse
import os
import sys
import time

_demoted = [p for p in sys.path if "omni.pip.compute" in p]
sys.path = [p for p in sys.path if p not in _demoted] + _demoted

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_YAML = os.path.join(HERE, "palm_env_T_cam.yaml")


def rotmat_to_quat(m):
    tr = float(np.trace(m))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def estimate_marker_pose(corners, size, K, dist):
    s = size / 2.0
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(4, 2).astype(np.float64), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    rm, _ = cv2.Rodrigues(rvec)
    t = np.eye(4)
    t[:3, :3] = rm
    t[:3, 3] = tvec.ravel()
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-marker", action="store_true", help="write a printable marker PNG and exit")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--size", type=float, default=0.04, help="printed marker side length [m] — MEASURE IT")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--marker-env-pos", type=float, nargs=3, default=[0.0, 0.09, 0.52],
                    help="marker center in env frame (translation is auto-centered later; rough is fine)")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))

    if args.make_marker:
        img = cv2.aruco.generateImageMarker(adict, args.id, 600)
        canvas = np.full((760, 680), 255, np.uint8)
        canvas[80:680, 40:640] = img
        cv2.putText(canvas, "^ this edge -> FINGERS", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        cv2.putText(canvas, f"{args.dict} id={args.id}  print & MEASURE side length", (60, 730),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
        out = "/tmp/palm_marker_print.png"
        cv2.imwrite(out, canvas)
        print(f"marker written -> {out}  (print it, measure the black square side, pass --size)")
        return

    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, 30)
    pipeline.start(config)
    for _ in range(15):
        pipeline.wait_for_frames()

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(adict, params)
    quats, tvecs, vis = [], [], None
    K = None
    try:
        for i in range(args.frames):
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if K is None:
                intr = cf.profile.as_video_stream_profile().intrinsics
                K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
            color = np.asarray(cf.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or args.id not in ids.ravel():
                continue
            c = corners[list(ids.ravel()).index(args.id)]
            t = estimate_marker_pose(c, args.size, K, np.zeros(5))
            if t is None:
                continue
            quats.append(rotmat_to_quat(t[:3, :3]))
            tvecs.append(t[:3, 3])
            if vis is None:
                vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                rm, _ = cv2.Rodrigues(t[:3, :3])
                cv2.drawFrameAxes(vis, K, np.zeros(5), rm, t[:3, 3], args.size * 0.75)
                cv2.aruco.drawDetectedMarkers(vis, [c])
    finally:
        pipeline.stop()

    n = len(quats)
    print(f"[palm-calib] marker detected in {n}/{args.frames} frames")
    if n < 10:
        print("[palm-calib] TOO FEW detections — check lighting/occlusion/marker id and retry")
        sys.exit(1)

    q = np.array(quats)
    q[np.sum(q * q[0], axis=1) < 0] *= -1.0  # sign-align
    q_mean = q.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)
    spread = np.degrees(2 * np.arccos(np.clip(np.abs(q @ q_mean), -1, 1))).max()
    t_mean = np.array(tvecs).mean(axis=0)
    print(f"[palm-calib] rotation spread {spread:.2f} deg, marker at cam-dist {np.linalg.norm(t_mean):.3f} m")
    if spread > 2.0:
        print("[palm-calib][WARN] rotation spread > 2 deg — marker/hand moved or detection unstable")

    w, x, y, z = q_mean
    rm = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    cam_T_marker = np.eye(4)
    cam_T_marker[:3, :3] = rm
    cam_T_marker[:3, 3] = t_mean

    env_T_marker = np.eye(4)
    env_T_marker[:3, 3] = np.array(args.marker_env_pos)
    env_T_cam = env_T_marker @ np.linalg.inv(cam_T_marker)

    cam_pos = env_T_cam[:3, 3]
    cam_view = env_T_cam[:3, :3] @ np.array([0, 0, 1.0])  # camera +Z (optical axis) in env
    print(f"[palm-calib] camera position in env: {np.round(cam_pos, 3)} (sanity: above/beside the hand?)")
    print(f"[palm-calib] camera optical axis in env: {np.round(cam_view, 3)} (sanity: pointing DOWN-ish, z<0?)")

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump({
            "env_T_cam": {"matrix": env_T_cam.tolist()},
            "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames_used": int(n),
            "rot_spread_deg": float(spread),
            "marker": {"dict": args.dict, "id": int(args.id), "size_m": float(args.size),
                       "env_pos": [float(v) for v in args.marker_env_pos]},
        }, f)
    if vis is not None:
        cv2.imwrite("/tmp/palm_marker_detect.png", vis)
        print("[palm-calib] detection overlay -> /tmp/palm_marker_detect.png (CHECK the axes!)")
    print(f"[palm-calib] saved -> {OUT_YAML}")
    print("[palm-calib] the pipeline will use it automatically on next start;")
    print("[palm-calib] recommend zeroing the old slider rotation: rx/ry/rz in repose_manual_calib.yaml")


if __name__ == "__main__":
    main()
