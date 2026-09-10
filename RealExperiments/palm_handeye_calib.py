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
           "pos_mm_mean": float(np.mean(poss)), "pos_mm_max": float(np.max(poss)),
           "per_rot_deg": [float(r) for r in rots], "per_pos_mm": [float(x) for x in poss]}
    return base_T_cam, palm_T_marker, res


def solve_trimmed(base_T_palm_list, cam_T_marker_list, min_keep=10, rot_tol=2.5, pos_tol=25.0,
                  verbose=True):
    """Iteratively drop the worst sample until residuals are sane (outlier armor)."""
    idx = list(range(len(base_T_palm_list)))
    dropped = []
    while True:
        bp = [base_T_palm_list[i] for i in idx]
        cm = [cam_T_marker_list[i] for i in idx]
        base_T_cam, palm_T_marker, res = solve_handeye(bp, cm)
        if (res["rot_deg_max"] <= rot_tol and res["pos_mm_max"] <= pos_tol) or len(idx) <= min_keep:
            return base_T_cam, palm_T_marker, res, idx, dropped
        worst_local = int(np.argmax(res["per_rot_deg"]))
        worst = idx[worst_local]
        dropped.append((worst, res["per_rot_deg"][worst_local], res["per_pos_mm"][worst_local]))
        if verbose:
            print(f"[handeye] dropping outlier sample #{worst}: "
                  f"{res['per_rot_deg'][worst_local]:.1f} deg / {res['per_pos_mm'][worst_local]:.0f} mm")
        idx.pop(worst_local)


# accept gates for a FULL-quality calibration (rotation AND translation)
# pos_max reflects the rig's systematic noise floor (URDF-vs-real FK + sticker
# flex, ~20-25 mm measured in the field); solution quality is guarded by the
# bootstrap gates, which is what actually matters for the output.
GATES = {"rot_max": 3.0, "pos_max": 30.0, "boot_rot": 1.5, "boot_pos": 20.0, "div_min": 40.0}


def bootstrap_spread(bp, cm, ref_btc, iters=25):
    rng = np.random.default_rng(0)
    n = len(bp)
    k = max(6, int(round(0.8 * n)))
    rots, poss = [], []
    for _ in range(iters):
        idx = rng.choice(n, k, replace=False)
        btc, _, _ = solve_handeye([bp[i] for i in idx], [cm[i] for i in idx])
        dR = btc[:3, :3] @ ref_btc[:3, :3].T
        rots.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        poss.append(np.linalg.norm(btc[:3, 3] - ref_btc[:3, 3]) * 1000)
    return float(np.percentile(rots, 95)), float(np.percentile(poss, 95))


def rotation_diversity_deg(bp):
    best = 0.0
    for i in range(len(bp)):
        for j in range(i + 1, len(bp)):
            dR = bp[i][:3, :3] @ bp[j][:3, :3].T
            best = max(best, np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return best


def full_quality(bp, cm, min_keep):
    """Solve + all quality gates. Returns (ok, summary_str, details...)."""
    btc, ptm, res, kept, dropped = solve_trimmed(bp, cm, min_keep=min_keep, verbose=False)
    bpk = [bp[i] for i in kept]
    cmk = [cm[i] for i in kept]
    b_rot, b_pos = bootstrap_spread(bpk, cmk, btc)
    div = rotation_diversity_deg(bpk)
    fails = []
    if res["rot_deg_max"] > GATES["rot_max"]:
        fails.append(f"residual rot {res['rot_deg_max']:.1f}>{GATES['rot_max']} deg")
    if res["pos_mm_max"] > GATES["pos_max"]:
        fails.append(f"residual pos {res['pos_mm_max']:.0f}>{GATES['pos_max']:.0f} mm")
    if b_rot > GATES["boot_rot"]:
        fails.append(f"solution rot spread {b_rot:.1f}>{GATES['boot_rot']} deg")
    if b_pos > GATES["boot_pos"]:
        fails.append(f"solution pos spread {b_pos:.0f}>{GATES['boot_pos']:.0f} mm (need BIGGER wrist rotations)")
    if div < GATES["div_min"]:
        fails.append(f"rotation diversity {div:.0f}<{GATES['div_min']:.0f} deg (tilt the wrist much more)")
    summary = (f"res {res['rot_deg_max']:.1f}deg/{res['pos_mm_max']:.0f}mm  "
               f"boot {b_rot:.1f}deg/{b_pos:.0f}mm  div {div:.0f}deg  "
               f"drop {len(dropped)}")
    return len(fails) == 0, summary, fails, btc, ptm, res, kept, dropped


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
    ap.add_argument("--no-preview", action="store_true",
                    help="terminal-input teach mode instead of the live camera window")
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

    # FK cross-check: one-lib FK flange position vs SDK TCP position.
    # A mismatch that GROWS with pose changes means the URDF FK does not match
    # the real arm (joint sign/offset) — hand-eye would then be garbage.
    try:
        code, tcp = arm.get_position(is_radian=True)
        if code == 0:
            kin.update(np.radians(q0), hand_q0)
            fk_flange = kin.link_tf["link8"][:3, 3] * 1000.0  # mm
            tcp_off = np.array(arm.tcp_offset[:3], dtype=np.float64)  # mm, flange frame
            d = np.linalg.norm(np.array(tcp[:3]) - fk_flange)
            print(f"[handeye] FK check @home: |SDK tcp - FK flange| = {d:.1f} mm"
                  f" (tcp offset set: {np.round(tcp_off, 1)} mm)")
            if d > 30 and np.linalg.norm(tcp_off) < 1e-6:
                print("[handeye][WARN] FK deviates >30 mm with zero tcp-offset — URDF/SDK mismatch?"
                      " Capture 2-3 samples at very different poses and watch the residual report.")
    except Exception as e:  # informational only
        print(f"[handeye] FK check skipped: {e}")
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
        """Multi-frame capture robust to the planar-PnP two-fold ambiguity.

        IPPE on a single flat marker has two candidate poses; near-frontal
        views make them nearly tied and noise picks the wrong branch (this is
        what produced 50-deg outliers in the field). Keep BOTH solutions per
        frame, choose branches by cross-frame consensus, and reject the whole
        sample if the frames still disagree.
        """
        frames_sol, K = [], None
        s = args.size / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
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
            try:
                n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                    obj, c.reshape(4, 2).astype(np.float64), K, np.zeros(5),
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except cv2.error:
                continue
            if n_sol < 1:
                continue
            sols = []
            for rv, tv in zip(rvecs, tvecs):
                rm, _ = cv2.Rodrigues(rv)
                sols.append((rr.rotmat_to_quat(rm), tv.ravel()))
            frames_sol.append(sols)
        if len(frames_sol) < max(4, args.frames_per_pose // 3):
            return None
        # consensus branch selection: iterate mean-rotation -> nearest branch
        sel = [0] * len(frames_sol)
        for _ in range(4):
            qs = []
            for fs, si in zip(frames_sol, sel):
                q = fs[si][0].copy()
                if qs and np.dot(q, qs[0]) < 0:
                    q = -q
                qs.append(q)
            qm = np.mean(qs, axis=0)
            qm /= np.linalg.norm(qm)
            sel = []
            for fs in frames_sol:
                dots = [abs(np.dot(sol[0], qm)) for sol in fs]
                sel.append(int(np.argmax(dots)))
        qs, ts_sel = [], []
        for fs, si in zip(frames_sol, sel):
            q = fs[si][0].copy()
            if qs and np.dot(q, qs[0]) < 0:
                q = -q
            qs.append(q)
            ts_sel.append(fs[si][1])
        qm = np.mean(qs, axis=0)
        qm /= np.linalg.norm(qm)
        spread = np.degrees(2 * np.arccos(np.clip(np.abs(np.array(qs) @ qm), -1, 1))).max()
        if spread > 3.0:
            print(f"[handeye][WARN] frames disagree ({spread:.1f} deg) — view too frontal/unstable;"
                  " TILT the palm 30-45 deg to the camera and retry")
            return None
        return tf(rr.quat_to_rotmat(qm), np.array(ts_sel).mean(axis=0))

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
            # -------- teach mode: user drags the arm, captures on demand ----
            code = arm.set_mode(2)  # manual (gravity-compensated drag) mode
            arm.set_state(0)
            time.sleep(0.3)
            if code != 0:
                raise RuntimeError(f"set_mode(2) failed ({code}) — enable Manual Mode in xArm Studio")
            print("[handeye] TEACH MODE ON — the arm is free to drag.")
            print("[handeye] TIP: keep the marker TILTED 20-45 deg to the camera in most samples —")
            print("[handeye] frontal views are ambiguous for a flat marker and get rejected/deweighted.")

            def try_capture():
                code, qa = arm.get_servo_angle()
                if code != 0:
                    print(f"[handeye][WARN] joint read failed ({code})")
                    return
                ctm = capture_marker()
                code, qb = arm.get_servo_angle()
                if code != 0:
                    return
                moved = np.max(np.abs(np.array(qa[:7]) - np.array(qb[:7])))
                if moved > 0.3:
                    print(f"[handeye][WARN] arm moved {moved:.2f} deg during capture — hold still, retry")
                    return
                if ctm is None:
                    print("[handeye][WARN] marker not detected steadily — adjust pose / lighting, retry")
                    return
                q_act = (np.array(qa[:7]) + np.array(qb[:7])) / 2.0
                kin.update(np.radians(q_act), hand_q0)
                base_T_palm_list.append(kin.palm_tf_base().astype(np.float64))
                cam_T_marker_list.append(ctm)
                np.savez(os.path.join(HERE, "handeye_samples_live.npz"),
                         base_T_palm=np.array(base_T_palm_list),
                         cam_T_marker=np.array(cam_T_marker_list))
                print(f"[handeye] sample {len(base_T_palm_list)} captured ✓")

            if args.no_preview:
                print("[handeye] Drag, let go, press ENTER to capture. ~20 samples recommended.")
                while True:
                    n = len(base_T_palm_list)
                    cmd = input(f"[handeye] {n} samples | ENTER=capture u=undo d=done q=abort > ").strip().lower()
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
                    try_capture()
            else:
                # live preview window with detection overlay; keys act HERE
                win = "handeye teach  (SPACE=capture  u=undo  d=done  q=abort)"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                quality_line, quality_ok = "", False

                def refresh_quality():
                    nonlocal quality_line, quality_ok
                    if len(base_T_palm_list) < args.min_samples:
                        quality_line, quality_ok = f"need {args.min_samples - len(base_T_palm_list)} more samples", False
                        return
                    ok, summary, fails, *_ = full_quality(
                        base_T_palm_list, cam_T_marker_list,
                        max(args.min_samples - 2, int(0.6 * len(base_T_palm_list))))
                    quality_ok = ok
                    quality_line = ("PASS - press d | " if ok else "not yet | ") + summary
                print("[handeye] live window up — keys work IN THE WINDOW, not the terminal.")
                K_prev = None
                s_half = args.size / 2
                obj_pts = np.array([[-s_half, s_half, 0], [s_half, s_half, 0],
                                    [s_half, -s_half, 0], [-s_half, -s_half, 0]])
                while True:
                    cf = pipeline.wait_for_frames().get_color_frame()
                    if K_prev is None:
                        intr = cf.profile.as_video_stream_profile().intrinsics
                        K_prev = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
                    img = cv2.cvtColor(np.asarray(cf.get_data()), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = detector.detectMarkers(gray)
                    seen = ids is not None and args.id in ids.ravel()
                    if seen:
                        c = corners[list(ids.ravel()).index(args.id)]
                        cv2.aruco.drawDetectedMarkers(img, [c])
                        ok, rvec, tvec = cv2.solvePnP(obj_pts, c.reshape(4, 2).astype(np.float64),
                                                      K_prev, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        if ok:
                            cv2.drawFrameAxes(img, K_prev, np.zeros(5), rvec, tvec, args.size * 0.75)
                        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (40, 160, 40), -1)
                        cv2.putText(img, "MARKER OK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (30, 30, 200), -1)
                        cv2.putText(img, "MARKER NOT VISIBLE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2)
                    cv2.putText(img, f"samples: {len(base_T_palm_list)}/{args.min_samples}+  "
                                     "SPACE=capture u=undo d=done q=abort",
                                (10, img.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                    if quality_line:
                        color = (0, 255, 0) if quality_ok else (0, 200, 255)
                        cv2.putText(img, quality_line, (10, img.shape[0] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.imshow(win, img)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord(" "):
                        if not seen:
                            print("[handeye][WARN] marker not visible — not capturing")
                        else:
                            try_capture()
                            refresh_quality()
                            if quality_line:
                                print(f"[handeye] quality: {quality_line}")
                    elif k == ord("u"):
                        if base_T_palm_list:
                            base_T_palm_list.pop()
                            cam_T_marker_list.pop()
                            print("[handeye] last sample removed")
                            refresh_quality()
                    elif k == ord("d"):
                        if len(base_T_palm_list) < args.min_samples:
                            print(f"[handeye] need at least {args.min_samples} samples"
                                  f" (have {len(base_T_palm_list)})")
                            continue
                        ok, summary, fails, *_ = full_quality(
                            base_T_palm_list, cam_T_marker_list,
                            max(args.min_samples - 2, int(0.6 * len(base_T_palm_list))))
                        if ok:
                            break
                        print(f"[handeye] NOT GOOD YET — keep sampling. Failing gates:")
                        for fmsg in fails:
                            print(f"[handeye]   - {fmsg}")
                    elif k in (ord("q"), 27):
                        print("[handeye] aborted by user")
                        cv2.destroyAllWindows()
                        sys.exit(1)
                cv2.destroyAllWindows()
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
    raw_npz = os.path.join(HERE, "handeye_samples.npz")
    np.savez(raw_npz, base_T_palm=np.array(base_T_palm_list), cam_T_marker=np.array(cam_T_marker_list))
    print(f"[handeye] raw samples dumped -> {raw_npz}")
    if n < args.min_samples:
        print(f"[handeye] only {n} good poses (<{args.min_samples}) — aborting without writing calibration")
        sys.exit(1)

    base_T_cam, palm_T_marker, res, kept, dropped = solve_trimmed(
        base_T_palm_list, cam_T_marker_list,
        min_keep=max(args.min_samples - 2, int(0.6 * len(base_T_palm_list))))
    b_rot, b_pos = bootstrap_spread([base_T_palm_list[i] for i in kept],
                                    [cam_T_marker_list[i] for i in kept], base_T_cam)
    div = rotation_diversity_deg([base_T_palm_list[i] for i in kept])
    print(f"[handeye] solution stability (bootstrap p95): rot {b_rot:.2f} deg, pos {b_pos:.0f} mm;"
          f" rotation diversity {div:.0f} deg")
    print(f"[handeye] solved from {len(kept)}/{n} poses ({len(dropped)} outliers dropped).")
    print(f"[handeye] residuals: rot mean {res['rot_deg_mean']:.2f} / max {res['rot_deg_max']:.2f} deg,"
          f" pos mean {res['pos_mm_mean']:.1f} / max {res['pos_mm_max']:.1f} mm")
    if res["rot_deg_max"] > 3.0 or res["pos_mm_max"] > 20.0:
        print("[handeye][FAIL] residuals still high after trimming — NOT writing the calibration.")
        print("[handeye]  likely causes: sticker moved / wrong --size / FK mismatch (see startup check).")
        print(f"[handeye]  raw samples kept at {raw_npz} for offline analysis.")
        sys.exit(1)
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
