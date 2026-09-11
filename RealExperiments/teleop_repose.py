#!/usr/bin/env python3
"""Human teleoperation of the repose task — the sim2real gap's control experiment.

Runs the SAME stack as the policy deployment (FoundationPose tracker, XHand
streaming with lead cap + torque limit, identical npz logging) but the 12 joint
targets come from a HUMAN instead of the LSTM. If a person can rotate the cube
with this hardware and the policy cannot, the gap is the policy/observation, not
the hand.

Glove input is source-agnostic: any client (Manus Core on Windows, a Linux SDK
script, whatever) sends joint angles over UDP as JSON:

    {"t": 1736580000.0, "joints": {"index_mcp": 0.6, "index_pip": 0.9,
     "index_spread": 0.0, "middle_mcp": ..., "ring_mcp": ..., "pinky_mcp": ...,
     "thumb_cmc": ..., "thumb_rot": ..., "thumb_mcp": ...}}

Angles in radians, 0 = straight/open. Missing keys hold their last value.

    # 1) check the whole chain without a glove (synthetic closing/opening hand)
    python RealExperiments/teleop_repose.py --source synthetic --steps 200
    # 2) with hardware + glove client sending to udp 9881
    python RealExperiments/teleop_repose.py --source udp --real --execute
"""
import argparse
import json
import math
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foundationpose_repose_real as rr

import numpy as np

GLOVE_UDP = ("0.0.0.0", 9881)   # bind all: the bridge may come from another host

# Manus Core ergonomics layout: 20 values per hand, degrees, in this order.
MANUS_ERGO = [
    "thumb_cmc_spread", "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_spread", "index_mcp", "index_pip", "index_dip",
    "middle_spread", "middle_mcp", "middle_pip", "middle_dip",
    "ring_spread", "ring_mcp", "ring_pip", "ring_dip",
    "pinky_spread", "pinky_mcp", "pinky_pip", "pinky_dip",
]
# Manus "thumb_cmc_spread" drives the XHand thumb rotation joint
ERGO_ALIAS = {"thumb_cmc_spread": "thumb_rot"}

# human joint -> (xhand joint, human range [rad], invert)
RETARGET = {
    "index_spread": ("index_joint0", (-0.26, 0.26), False),
    "index_mcp": ("index_joint1", (0.0, 1.57), False),
    "index_pip": ("index_joint2", (0.0, 1.75), False),
    "middle_mcp": ("middle_joint0", (0.0, 1.57), False),
    "middle_pip": ("middle_joint1", (0.0, 1.75), False),
    "ring_mcp": ("ring_joint0", (0.0, 1.57), False),
    "ring_pip": ("ring_joint1", (0.0, 1.75), False),
    "pinky_mcp": ("pinky_joint0", (0.0, 1.57), False),
    "pinky_pip": ("pinky_joint1", (0.0, 1.75), False),
    "thumb_cmc": ("thumb_joint0", (0.0, 1.22), False),
    "thumb_rot": ("thumb_joint1", (-0.7, 1.05), False),
    "thumb_mcp": ("thumb_joint2", (0.0, 1.40), False),
}


CALIB_YAML = os.path.join(HERE, "teleop_glove_calib.yaml")


def load_calib() -> dict:
    if not os.path.exists(CALIB_YAML):
        return {}
    import yaml
    with open(CALIB_YAML) as f:
        return yaml.safe_load(f) or {}


def run_calibration(glove, secs: float = 12.0) -> dict:
    """Record each human joint's min/max while the user opens and closes."""
    print(f"[calib] open and close your hand fully, a few times, for {secs:.0f} s ...")
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    t0 = time.time()
    while time.time() - t0 < secs:
        human, _ = glove.poll()
        for k, v in human.items():
            lo[k] = min(lo.get(k, v), v)
            hi[k] = max(hi.get(k, v), v)
        left = secs - (time.time() - t0)
        if int(left * 2) % 4 == 0:
            print(f"\r[calib] {left:4.1f}s  joints seen: {len(lo)}   ", end="", flush=True)
        time.sleep(0.02)
    print()
    calib = {k: [float(lo[k]), float(hi[k])] for k in lo if hi[k] - lo[k] > math.radians(5)}
    import yaml
    with open(CALIB_YAML, "w") as f:
        yaml.safe_dump(calib, f)
    print(f"[calib] saved {len(calib)} joint ranges -> {CALIB_YAML}")
    for k, (a, b) in sorted(calib.items()):
        print(f"    {k:18s} {math.degrees(a):+6.1f} .. {math.degrees(b):+6.1f} deg")
    return calib


def retarget(human: dict, calib: dict | None = None) -> np.ndarray:
    """Human joint angles -> XHand 12 targets (Isaac order), linearly range-mapped."""
    q = np.zeros(12, dtype=np.float32)
    for hname, (xname, (h_lo, h_hi), inv) in RETARGET.items():
        if hname not in human:
            continue
        if calib and hname in calib:
            h_lo, h_hi = calib[hname]
        j = rr.ISAAC12.index(xname)
        a = float(np.clip((human[hname] - h_lo) / max(1e-6, h_hi - h_lo), 0.0, 1.0))
        if inv:
            a = 1.0 - a
        q[j] = rr.LOWER[j] + a * (rr.UPPER[j] - rr.LOWER[j])
    return q


class UdpGlove:
    def __init__(self, addr=GLOVE_UDP):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(addr)
        self.sock.setblocking(False)
        self.human, self.stamp = {}, 0.0

    def poll(self) -> tuple[dict, float]:
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                msg = json.loads(data.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if "ergo" in msg:   # raw Manus ergonomics array (degrees)
                vals = msg["ergo"]
                got = {}
                for name, v in zip(MANUS_ERGO, vals):
                    got[ERGO_ALIAS.get(name, name)] = math.radians(float(v))
                self.human.update(got)
            self.human.update(msg.get("joints", {}))
            self.stamp = time.time()
        return self.human, self.stamp


class SyntheticGlove:
    """Slow open/close sweep so the chain can be verified without hardware."""

    def __init__(self):
        self.t0 = time.time()

    def poll(self):
        a = 0.5 - 0.5 * math.cos(2 * math.pi * (time.time() - self.t0) / 6.0)
        return ({"index_mcp": 1.2 * a, "index_pip": 1.4 * a, "middle_mcp": 1.2 * a,
                 "middle_pip": 1.4 * a, "ring_mcp": 1.2 * a, "ring_pip": 1.4 * a,
                 "pinky_mcp": 1.2 * a, "pinky_pip": 1.4 * a, "thumb_cmc": 0.9 * a,
                 "thumb_rot": 0.6 * a, "thumb_mcp": 1.0 * a}, time.time())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["udp", "synthetic"], default="udp")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--real", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--xarm-ip", default="192.168.1.205")
    p.add_argument("--xhand-port", default="/dev/ttyUSB0")
    p.add_argument("--hand-kp", type=int, default=100)
    p.add_argument("--hand-tor-max", type=int, default=150)
    p.add_argument("--lead-cap", type=float, default=0.06)
    p.add_argument("--max-hand-step", type=float, default=0.25)
    p.add_argument("--ema", type=float, default=0.4, help="glove smoothing (teleop jitter)")
    p.add_argument("--cube-edge", type=float, default=0.062)
    p.add_argument("--pose-source", choices=["tracker", "none"], default="tracker")
    p.add_argument("--run-log-dir", default=os.path.join(HERE, "runlogs"))
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--calibrate", action="store_true",
                   help="record your hand's joint ranges first (open/close for 12 s)")
    args = p.parse_args()
    args.log_npz = None
    log_path, npz_path = rr.start_run_log(args)
    print(f"[teleop] source={args.source}  real={args.real}  execute={args.execute}")

    glove = SyntheticGlove() if args.source == "synthetic" else UdpGlove()
    if args.source == "udp":
        print(f"[teleop] listening for glove packets on udp://{GLOVE_UDP[0]}:{GLOVE_UDP[1]} ...")
        t0 = time.time()
        while not glove.poll()[0] and time.time() - t0 < 60:
            time.sleep(0.1)
        if not glove.poll()[0]:
            raise RuntimeError("no glove packets in 60 s — is the Windows bridge running "
                               "and pointed at this host:9881?")
        print(f"[teleop] glove connected ({len(glove.poll()[0])} joints)")
    calib = run_calibration(glove) if args.calibrate else load_calib()
    if calib:
        print(f"[teleop] using calibrated ranges for {len(calib)} joints")

    hw = None
    if args.real:
        hw = rr.RealHardware(args.xarm_ip, args.xhand_port)
        hw.set_gains(args.hand_kp, args.hand_tor_max)
        print(f"[teleop] XHand gains kp={args.hand_kp} tor_max={args.hand_tor_max}")

    receiver = tracker_proc = None
    if args.pose_source == "tracker":
        args.roi = None
        args.serial = None
        args.no_view = False
        args._tracker_log = log_path.replace(".log", ".tracker.log")
        if args.real:
            input("[teleop] ENTER to home the hand (cube NOT in hand yet)...")
            hw.hand_home(np.zeros(12, dtype=np.float32), 0.25)
            input("[teleop] place the cube in the palm, then ENTER to start the tracker...")
        tracker_proc = rr.spawn_tracker(args)
        receiver = rr.UdpPoseReceiver(rr.UDP_ADDR)
        if not receiver.wait_first(180.0):
            raise RuntimeError("no pose from the tracker")
        print("[teleop] pose stream up")

    base_T_cam = None
    kin = rr.UrdfKinematics()
    arm_q = hw.arm_q() if hw is not None else rr.DEFAULT_ARM_Q
    hand_q = np.zeros(12, dtype=np.float32)
    kin.update(arm_q, hand_q)
    env_T_base = rr.tf_from_pos_quat(rr.SIM_PALM_POS, rr.SIM_PALM_QUAT) @ rr.tf_inv(kin.palm_tf_base())
    palm_yaml = os.path.join(HERE, "palm_env_T_cam.yaml")
    if receiver is not None and os.path.exists(palm_yaml):
        import yaml
        with open(palm_yaml) as f:
            base_T_cam = np.array(yaml.safe_load(f)["base_T_cam"]["matrix"], dtype=np.float64)
        print("[teleop] hand-eye extrinsic loaded")

    if args.real and args.execute:
        input("[teleop] ENTER to hand control to the glove (hand will follow you)...")

    log = {k: [] for k in ["obs", "action", "targets", "hand_q", "obj_pos", "obj_quat", "goal_quat", "t"]}
    prev_targets = hand_q.copy()
    smooth = None
    goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
    next_t = time.perf_counter()
    try:
        for step in range(args.steps):
            human, stamp = glove.poll()
            if args.source == "udp" and (not human or time.time() - stamp > 0.5):
                if step % 40 == 0:
                    print("[teleop] no fresh glove data — holding")
                desired = prev_targets
            else:
                desired = retarget(human, calib)
            smooth = desired if smooth is None else args.ema * desired + (1 - args.ema) * smooth
            targets = smooth.copy()
            if args.lead_cap > 0:
                targets = np.clip(targets, hand_q - args.lead_cap, hand_q + args.lead_cap)
            targets = prev_targets + np.clip(targets - prev_targets, -args.max_hand_step, args.max_hand_step)
            targets = np.clip(targets, rr.LOWER, rr.UPPER).astype(np.float32)

            if hw is not None and args.execute:
                q_meas = hw.hand_stream(targets, read=True)
                hand_q = q_meas if q_meas is not None else targets
            else:
                hand_q = targets
            prev_targets = targets

            obj_pos = obj_quat = None
            if receiver is not None:
                pose_cam = receiver.poll()
                if pose_cam is not None:
                    pose = env_T_base @ (base_T_cam @ pose_cam if base_T_cam is not None else pose_cam)
                    obj_pos, obj_quat = pose[:3, 3], rr.rotmat_to_quat(pose[:3, :3])
            if obj_pos is None:
                obj_pos, obj_quat = rr.REST_POS.copy(), np.array([1.0, 0.0, 0.0, 0.0])

            kin.update(arm_q, hand_q)
            tips = kin.fingertip_pos_base()
            tips_env = np.stack([(env_T_base @ np.array([*t, 1.0]))[:3] for t in tips])
            obs = rr.build_obs(tips_env, obj_pos, obj_quat, goal_quat, np.zeros(12, dtype=np.float32))
            for k, v in zip(log, [obs, smooth, targets, hand_q, obj_pos, obj_quat, goal_quat, time.time()]):
                log[k].append(np.asarray(v).copy())

            if step % args.print_every == 0:
                print(f"[teleop] step {step:5d} cube {np.round(obj_pos, 3)} "
                      f"hand {np.round(hand_q[:4], 2)}...")
            next_t += rr.STEP_DT
            sl = next_t - time.perf_counter()
            if sl > 0:
                time.sleep(sl)
    except KeyboardInterrupt:
        print("\n[teleop] stopped by user")
    finally:
        if tracker_proc is not None and tracker_proc.poll() is None:
            import signal
            os.killpg(os.getpgid(tracker_proc.pid), signal.SIGINT)
        if hw is not None:
            hw.close()
        np.savez(npz_path, **{k: np.asarray(v) for k, v in log.items()})
        print(f"[log] saved {npz_path}")
        n = len(log["t"])
        if n > 2:
            oq = np.asarray(log["obj_quat"])
            gross = math.degrees(sum(rr.rotation_distance(oq[i], oq[i + 1]) for i in range(n - 1)))
            print(f"[teleop] {n} steps, cube gross rotation {gross:.0f} deg "
                  f"({gross / max(1e-9, (log['t'][-1] - log['t'][0])):.1f} deg/s)")


if __name__ == "__main__":
    main()
