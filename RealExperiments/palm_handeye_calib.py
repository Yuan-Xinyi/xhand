#!/usr/bin/env python3
"""20-pose hand-eye calibration: ArUco on the palm, wrist moves, solve base_T_cam.

Unlike the single-pose palm_marker_calib.py, this does NOT assume the sticker
is placed straight: cv2.calibrateHandEye solves base_T_cam AND palm_T_marker
from ~20 (arm FK, marker detection) pairs. Sticker placement can be arbitrary.

Setup: marker stuck anywhere FLAT and RIGID on the palm, hand OPEN, no cube,
arm at the repose working pose. The wrist then tilts +-12 deg through a fixed
pose list (keep the area clear!), capturing at each stop, and returns home.

Run (one env has xarm sdk + realsense + aruco):
    /home/lqin/miniconda3/envs/one/bin/python RealExperiments/palm_handeye_calib.py --size 0.06

Self-test (no hardware; verifies the calibrateHandEye convention):
    ... palm_handeye_calib.py --self-test
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# import rr FIRST: it strips Isaac's cp311 pip_prebundle paths (exported by an
# activated env_isaaclab shell) that would poison numpy/cv2 for this py312 env
import foundationpose_repose_real as rr  # UrdfKinematics, SIM_PALM_*, tf helpers

import cv2
import numpy as np

OUT_YAML = os.path.join(HERE, "palm_env_T_cam.yaml")

# wrist-joint deltas [deg] for j4..j7 (20 poses, home first, marker stays up)
POSE_DELTAS = [
    (0, 0, 0, 0),
    (0, 10, 0, 0), (0, -10, 0, 0), (0, 0, 10, 0), (0, 0, -10, 0),
    (0, 0, 0, 12), (0, 0, 0, -12), (8, 0, 0, 0), (-8, 0, 0, 0),
    (0, 8, 8, 0), (0, -8, 8, 0), (0, 8, -8, 0), (0, -8, -8, 0),
    (0, 8, 0, 10), (0, -8, 0, -10), (0, 0, 8, 10), (0, 0, -8, -10),
    (6, 0, 8, 0), (-6, 0, -8, 0), (0, 12, 0, 12),
]


def tf(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def self_test():
    rng = np.random.default_rng(7)

    def rand_T(rs=1.0, ts=0.5):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        R, _ = cv2.Rodrigues(v * rng.uniform(0.2, 1.0) * rs)
        return tf(R, rng.uniform(-ts, ts, 3))

    base_T_cam = rand_T()
    palm_T_marker = rand_T(0.3, 0.05)
    poses = [rand_T(0.5, 0.3) for _ in range(20)]
    targets = [np.linalg.inv(base_T_cam) @ p @ palm_T_marker for p in poses]
    sol, ptm, res = solve_handeye(poses, targets)
    assert np.linalg.norm(sol - base_T_cam) < 1e-6
    assert np.linalg.norm(ptm - palm_T_marker) < 1e-6
    assert res["rot_deg_max"] < 1e-3 and res["pos_mm_max"] < 1e-3  # float arccos noise
    print("SELF-TEST PASS: convention (base2gripper, target2cam) -> base_T_cam")


def solve_handeye(base_T_palm_list, cam_T_marker_list):
    inv = np.linalg.inv
    R_b2g = [inv(p)[:3, :3] for p in base_T_palm_list]
    t_b2g = [inv(p)[:3, 3] for p in base_T_palm_list]
    R_t2c = [t[:3, :3] for t in cam_T_marker_list]
    t_t2c = [t[:3, 3] for t in cam_T_marker_list]
    R, t = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK)
    base_T_cam = tf(R, t)
    ptms = [inv(p) @ base_T_cam @ c for p, c in zip(base_T_palm_list, cam_T_marker_list)]
    # average palm_T_marker (tight cluster expected)
    qs = np.array([rr.rotmat_to_quat(x[:3, :3]) for x in ptms])
    qs[np.sum(qs * qs[0], axis=1) < 0] *= -1
    qm = qs.mean(axis=0)
    qm /= np.linalg.norm(qm)
    palm_T_marker = tf(rr.quat_to_rotmat(qm), np.array([x[:3, 3] for x in ptms]).mean(axis=0))
    # residuals: predicted vs measured marker pose per view
    rots, poss = [], []
    for p, c in zip(base_T_palm_list, cam_T_marker_list):
        pred = inv(base_T_cam) @ p @ palm_T_marker
        dR = pred[:3, :3] @ c[:3, :3].T
        rots.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        poss.append(np.linalg.norm(pred[:3, 3] - c[:3, 3]) * 1000)
    res = {"rot_deg_mean": float(np.mean(rots)), "rot_deg_max": float(np.max(rots)),
           "pos_mm_mean": float(np.mean(poss)), "pos_mm_max": float(np.max(poss))}
    return base_T_cam, palm_T_marker, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="scripted wrist motion instead of teach mode (default: teach — you drag the arm)")
    ap.add_argument("--min-samples", type=int, default=12)
    ap.add_argument("--xarm-ip", default="192.168.1.205")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--size", type=float, default=0.06, help="printed marker side [m] — MEASURE IT")
    ap.add_argument("--frames-per-pose", type=int, default=12)
    ap.add_argument("--speed", type=float, default=12.0, help="joint speed [deg/s]")
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import pyrealsense2 as rs
    from xarm.wrapper import XArmAPI

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(adict, params)
    kin = rr.UrdfKinematics()
    hand_q0 = np.zeros(12)

    arm = XArmAPI(args.xarm_ip)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.5)
    code, q0 = arm.get_servo_angle()
    if code != 0:
        raise RuntimeError(f"get_servo_angle failed: {code}")
    q0 = np.array(q0[:7], dtype=np.float64)
    print(f"[handeye] home arm q [deg]: {np.round(q0, 2)}")
    if args.auto:
        print(f"[handeye] will visit {len(POSE_DELTAS)} wrist poses, deltas up to +-12 deg on j4..j7,")
        print(f"[handeye] speed {args.speed} deg/s. Marker on palm, hand OPEN, NO cube, area CLEAR.")
        input("[handeye] ENTER to start the motion sequence, Ctrl-C to abort...")
    else:
        print("[handeye] teach mode: marker on palm, hand OPEN, NO cube.")
        print("[handeye] base_T_cam is pose-independent — end wherever you like; the pipeline")
        print("[handeye] composes it with its own FK at runtime.")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)
    for _ in range(10):
        pipeline.wait_for_frames()

    def capture_marker():
        quats, ts, K = [], [], None
        for _ in range(args.frames_per_pose):
            cf = pipeline.wait_for_frames().get_color_frame()
            if K is None:
                intr = cf.profile.as_video_stream_profile().intrinsics
                K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
            gray = cv2.cvtColor(np.asarray(cf.get_data()), cv2.COLOR_RGB2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or args.id not in ids.ravel():
                continue
            c = corners[list(ids.ravel()).index(args.id)]
            s = args.size / 2
            obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
            ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), K, np.zeros(5),
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            rm, _ = cv2.Rodrigues(rvec)
            quats.append(rr.rotmat_to_quat(rm))
            ts.append(tvec.ravel())
        if len(quats) < max(4, args.frames_per_pose // 3):
            return None
        q = np.array(quats)
        q[np.sum(q * q[0], axis=1) < 0] *= -1
        qm = q.mean(axis=0)
        qm /= np.linalg.norm(qm)
        return tf(rr.quat_to_rotmat(qm), np.array(ts).mean(axis=0))

    base_T_palm_list, cam_T_marker_list = [], []
    try:
        if args.auto:
            for i, d in enumerate(POSE_DELTAS):
                target = q0.copy()
                target[3:7] += np.array(d, dtype=np.float64)
                code = arm.set_servo_angle(angle=target.tolist(), speed=args.speed, wait=True)
                if code != 0:
                    print(f"[handeye][WARN] pose {i}: move failed ({code}), skipping")
                    continue
                time.sleep(0.6)
                code, q_act = arm.get_servo_angle()
                if code != 0:
                    continue
                ctm = capture_marker()
                if ctm is None:
                    print(f"[handeye][WARN] pose {i}: marker not detected, skipping")
                    continue
                kin.update(np.radians(np.array(q_act[:7])), hand_q0)
                base_T_palm_list.append(kin.palm_tf_base().astype(np.float64))
                cam_T_marker_list.append(ctm)
                print(f"[handeye] pose {i + 1}/{len(POSE_DELTAS)}: ok ({len(base_T_palm_list)} collected)")
        else:
            # -------- teach mode: user drags the arm, ENTER captures --------
            code = arm.set_mode(2)  # manual (gravity-compensated drag) mode
            arm.set_state(0)
            time.sleep(0.3)
            if code != 0:
                raise RuntimeError(f"set_mode(2) failed ({code}) — enable Manual Mode in xArm Studio")
            print("[handeye] TEACH MODE ON — the arm is free to drag.")
            print("[handeye] Drag to a pose (vary the ORIENTATION between samples, keep the marker")
            print("[handeye] visible to the camera), let go, then press ENTER. ~20 samples recommended.")
            while True:
                n = len(base_T_palm_list)
                cmd = input(f"[handeye] {n} samples | ENTER=capture  u=undo  d=done  q=abort > ").strip().lower()
                if cmd == "q":
                    print("[handeye] aborted by user")
                    sys.exit(1)
                if cmd == "u":
                    if base_T_palm_list:
                        base_T_palm_list.pop()
                        cam_T_marker_list.pop()
                        print("[handeye] last sample removed")
                    continue
                if cmd == "d":
                    if n < args.min_samples:
                        print(f"[handeye] need at least {args.min_samples} samples (have {n})")
                        continue
                    break
                code, qa = arm.get_servo_angle()
                if code != 0:
                    print(f"[handeye][WARN] joint read failed ({code})")
                    continue
                ctm = capture_marker()
                code, qb = arm.get_servo_angle()
                if code != 0:
                    continue
                moved = np.max(np.abs(np.array(qa[:7]) - np.array(qb[:7])))
                if moved > 0.3:
                    print(f"[handeye][WARN] arm moved {moved:.2f} deg during capture — hold still, retry")
                    continue
                if ctm is None:
                    print("[handeye][WARN] marker not detected — adjust the pose / lighting, retry")
                    continue
                q_act = (np.array(qa[:7]) + np.array(qb[:7])) / 2.0
                kin.update(np.radians(q_act), hand_q0)
                base_T_palm_list.append(kin.palm_tf_base().astype(np.float64))
                cam_T_marker_list.append(ctm)
                print(f"[handeye] sample {len(base_T_palm_list)} captured ✓")
    finally:
        if not args.auto:
            print("[handeye] leaving teach mode (position mode restored)")
            arm.set_mode(0)
            arm.set_state(0)
        else:
            print("[handeye] returning to home pose...")
            arm.set_servo_angle(angle=q0.tolist(), speed=args.speed, wait=True)
        pipeline.stop()

    n = len(base_T_palm_list)
    if n < 12:
        print(f"[handeye] only {n} good poses (<12) — aborting without writing calibration")
        sys.exit(1)

    base_T_cam, palm_T_marker, res = solve_handeye(base_T_palm_list, cam_T_marker_list)
    print(f"[handeye] solved from {n} poses.")
    print(f"[handeye] residuals: rot mean {res['rot_deg_mean']:.2f} / max {res['rot_deg_max']:.2f} deg,"
          f" pos mean {res['pos_mm_mean']:.1f} / max {res['pos_mm_max']:.1f} mm")
    if res["rot_deg_max"] > 2.0 or res["pos_mm_max"] > 15.0:
        print("[handeye][WARN] residuals high — sticker not rigid / marker size wrong / FK mismatch?")
    print(f"[handeye] palm_T_marker: t={np.round(palm_T_marker[:3, 3] * 1000, 1)}mm (sticker placement, FYI)")

    # env frame at the HOME pose (same formula as the control script)
    kin.update(np.radians(q0), hand_q0)
    env_T_base = rr.tf_from_pos_quat(rr.SIM_PALM_POS, rr.SIM_PALM_QUAT) @ np.linalg.inv(kin.palm_tf_base())
    env_T_cam = env_T_base @ base_T_cam
    cam_pos = env_T_cam[:3, 3]
    cam_axis = env_T_cam[:3, :3] @ np.array([0, 0, 1.0])
    print(f"[handeye] camera in env: pos {np.round(cam_pos, 3)}, optical axis {np.round(cam_axis, 3)}"
          " (sanity: above the hand, axis z<0?)")

    import yaml
    if os.path.exists(OUT_YAML):
        os.replace(OUT_YAML, OUT_YAML + ".bak")
        print(f"[handeye] previous calibration backed up -> {OUT_YAML}.bak")
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump({
            "env_T_cam": {"matrix": env_T_cam.tolist()},
            "method": "handeye_20pose",
            "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "poses_used": int(n),
            "residuals": res,
            "base_T_cam": {"matrix": base_T_cam.tolist()},
            "palm_T_marker": {"matrix": palm_T_marker.tolist()},
            "marker": {"dict": args.dict, "id": int(args.id), "size_m": float(args.size)},
        }, f)
    print(f"[handeye] saved -> {OUT_YAML} (pipeline auto-uses it; remove the sticker now)")


if __name__ == "__main__":
    main()
