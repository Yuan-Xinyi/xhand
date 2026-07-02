#!/usr/bin/env python3
"""FoundationPose pose initialization followed by RL policy execution on real hardware.

Default mode is dry-run: it captures the cube pose, builds observations, loads the
trained rl_games actor, and prints commands without connecting to the robots.

Real execution is opt-in and requires both:

    --real --execute

The control contract mirrors Pick-Cube-Direct-v0:
    target_q <- target_q + 0.1 * action
    target_q <- 0.3 * target_q + 0.7 * previous_target_q
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "logs/rl_games/pick_cube/0_2026-06-25_12-07-39/nn/pick_cube.pth"
DEFAULT_CALIB_YAML = "/home/lqin/one/one/camera/RS435/calibration_result.yaml"

ISAAC_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "index_joint0",
    "middle_joint0",
    "pinky_joint0",
    "ring_joint0",
    "thumb_joint0",
    "index_joint1",
    "middle_joint1",
    "pinky_joint1",
    "ring_joint1",
    "thumb_joint1",
    "index_joint2",
    "thumb_joint2",
]

ONE_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "thumb_joint0",
    "thumb_joint1",
    "thumb_joint2",
    "index_joint0",
    "index_joint1",
    "index_joint2",
    "middle_joint0",
    "middle_joint1",
    "ring_joint0",
    "ring_joint1",
    "pinky_joint0",
    "pinky_joint1",
]

ISAAC_TO_ONE = np.array([ISAAC_JOINT_NAMES.index(n) for n in ONE_JOINT_NAMES], dtype=np.int64)
ONE_TO_ISAAC = np.array([ONE_JOINT_NAMES.index(n) for n in ISAAC_JOINT_NAMES], dtype=np.int64)

DEFAULT_Q_ISAAC = np.array(
    [0.0, -0.7494, 0.0, 1.192, 0.0, 1.9414, 0.0] + [0.0] * 12,
    dtype=np.float32,
)

EE_BODY_NAMES = ["mid_link2", "pinky_link2", "ring_link2", "index_rota_link2", "thumb_rota_link2"]
FINGER_PAD_OFFSETS = {
    "thumb_rota_link2": (0.033409, 0.000346, 0.012429),
    "index_rota_link2": (-0.002238, -0.011313, 0.026695),
    "mid_link2": (0.000509, -0.014334, 0.023363),
    "ring_link2": (0.000705, -0.013922, 0.025485),
    "pinky_link2": (-0.000383, -0.011856, 0.028925),
}
PALM_CENTER_OFFSET = np.array([0.0, -0.02, 0.07], dtype=np.float32)
MOUNT_RPY = 4.71239
ACTION_SCALE = 0.1
ACT_MOVING_AVERAGE = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=250, help="50 Hz control steps. 250 steps = 5 seconds.")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--xarm-ip", default="192.168.1.205")
    parser.add_argument("--xhand-port", default="/dev/ttyUSB0")
    parser.add_argument("--real", action="store_true", help="Connect to xArm7 and XHand.")
    parser.add_argument("--execute", action="store_true", help="Actually stream commands. Requires --real.")
    parser.add_argument("--skip-start-sync", action="store_true", help="Do not move hardware to RL default home first.")
    parser.add_argument("--start-speed", type=float, default=0.25, help="xArm joint speed for start sync, rad/s.")
    parser.add_argument("--hand-start-speed", type=float, default=0.25, help="XHand start sync speed.")
    parser.add_argument("--max-arm-step", type=float, default=0.025, help="Max xArm joint target delta per cycle, rad.")
    parser.add_argument("--max-hand-step", type=float, default=0.035, help="Max XHand target delta per cycle.")
    parser.add_argument("--dry-print-every", type=int, default=10)
    parser.add_argument("--no_pose_capture", action="store_true", help="Use --pose_npy instead of running FoundationPose.")
    parser.add_argument("--pose_npy", default="/tmp/foundationpose_cube_pose.npy")
    parser.add_argument(
        "--calib_yaml",
        default=DEFAULT_CALIB_YAML,
        help="D435 eye-to-hand calibration (T_base_cam). Maps the FoundationPose camera_T_cube into the xArm base frame.",
    )
    parser.add_argument(
        "--no_calib",
        action="store_true",
        help="Treat the cube pose as already expressed in the xArm base frame (skip the T_base_cam transform).",
    )

    # FoundationPose args reused by capture_pose().
    parser.add_argument("--task", default="Pick-Cube-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--play_script", default=str(REPO_ROOT / "scripts" / "rl_games" / "play.py"))
    parser.add_argument("--foundationpose_root", default="/home/lqin/disk2/FoundationPose")
    parser.add_argument("--mesh_file", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--track_frames", type=int, default=5)
    parser.add_argument("--pose_out", default="/tmp/foundationpose_cube_pose.npy")
    parser.add_argument("--image_out", default="/tmp/foundationpose_init_frame.png")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--no_play", action="store_true", default=True)
    parser.add_argument("play_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_base_T_cam(calib_yaml: str) -> np.ndarray:
    """Load the eye-to-hand extrinsic T_base_cam (pose of the D435 in the xArm base frame).

    Convention from the calibration file: T_A_B maps points B->A, so
    p_base = T_base_cam @ p_cam.
    """
    import yaml

    path = Path(calib_yaml).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"D435 calibration not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    mat = np.array(data["T_base_cam"]["matrix"], dtype=np.float32)
    if mat.shape != (4, 4):
        raise RuntimeError(f"T_base_cam.matrix must be 4x4, got {mat.shape} from {path}")
    return mat


def _maybe_reexec_one_for_real(args: argparse.Namespace) -> None:
    """Run perception in env_isaaclab, then switch to the one env for hardware IO."""
    if not args.real:
        return
    one_python = os.environ.get("ONE_PYTHON", "/home/lqin/miniconda3/envs/one/bin/python")
    if not Path(one_python).exists():
        return
    if os.path.realpath(sys.executable) == os.path.realpath(one_python):
        return

    pose_path = args.pose_npy if args.no_pose_capture else args.pose_out
    argv = [
        one_python,
        os.path.abspath(__file__),
        "--no_pose_capture",
        "--pose_npy",
        pose_path,
        "--checkpoint",
        args.checkpoint,
        "--steps",
        str(args.steps),
        "--rate",
        str(args.rate),
        "--xarm-ip",
        args.xarm_ip,
        "--xhand-port",
        args.xhand_port,
        "--start-speed",
        str(args.start_speed),
        "--hand-start-speed",
        str(args.hand_start_speed),
        "--max-arm-step",
        str(args.max_arm_step),
        "--max-hand-step",
        str(args.max_hand_step),
        "--dry-print-every",
        str(args.dry_print_every),
        "--calib_yaml",
        args.calib_yaml,
        "--real",
    ]
    if args.execute:
        argv.append("--execute")
    if args.skip_start_sync:
        argv.append("--skip-start-sync")
    if args.no_calib:
        argv.append("--no_calib")

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        keep = []
        for entry in pythonpath.split(os.pathsep):
            lower = entry.lower()
            if "isaac" in lower or "omni.kit" in lower or "pip_prebundle" in lower:
                continue
            keep.append(entry)
        if keep:
            env["PYTHONPATH"] = os.pathsep.join(keep)
        else:
            env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PATH"] = os.path.dirname(one_python) + os.pathsep + env.get("PATH", "")
    print(f"[real] switching to one Python for hardware control: {one_python}")
    os.execve(one_python, argv, env)


def _rotmat_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    m = np.asarray(rotmat, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s],
            dtype=np.float32,
        )
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = q.astype(np.float32)
    return q / np.linalg.norm(q)


def _transform_point(tf: np.ndarray, local: np.ndarray) -> np.ndarray:
    return tf[:3, 3] + tf[:3, :3] @ local


def _clip_step(target: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    return current + np.clip(target - current, -max_step, max_step)


def _rotmat_from_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _tf_from_pos_rotmat(pos: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = rotmat
    tf[:3, 3] = pos
    return tf


def _import_torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        prebundles = [
            "/disk2/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle",
            "/disk2/IsaacLab/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle",
        ]
        for path in prebundles:
            if Path(path).exists() and path not in sys.path:
                sys.path.insert(0, path)
        import torch

        return torch


def _prepare_one_imports() -> None:
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    try:
        import pyglet

        pyglet.options["headless"] = True
    except Exception:
        pass
    if "matplotlib.pyplot" not in sys.modules:
        mpl = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")

        def _get_cmap(_name):
            colors = [
                (0.121, 0.466, 0.705),
                (0.682, 0.780, 0.909),
                (1.000, 0.498, 0.054),
                (1.000, 0.733, 0.470),
                (0.172, 0.627, 0.172),
                (0.596, 0.875, 0.541),
                (0.839, 0.153, 0.157),
                (1.000, 0.596, 0.588),
                (0.580, 0.404, 0.741),
                (0.773, 0.690, 0.835),
                (0.549, 0.337, 0.294),
                (0.769, 0.612, 0.580),
                (0.890, 0.467, 0.761),
                (0.969, 0.714, 0.824),
                (0.498, 0.498, 0.498),
                (0.780, 0.780, 0.780),
                (0.737, 0.741, 0.133),
                (0.859, 0.859, 0.553),
                (0.090, 0.745, 0.811),
                (0.620, 0.855, 0.898),
            ]
            return types.SimpleNamespace(colors=colors)

        pyplot.get_cmap = _get_cmap
        mpl.pyplot = pyplot
        sys.modules.setdefault("matplotlib", mpl)
        sys.modules.setdefault("matplotlib.pyplot", pyplot)
    one_root = "/home/lqin/one"
    if one_root not in sys.path:
        sys.path.insert(0, one_root)


def _patch_one_mechbase_compat() -> None:
    import one.robots.base.mech_base as mech_base

    init = mech_base.MechBase.__init__
    if getattr(init, "_accepts_is_free", False):
        return

    def _compat_init(self, *args, is_free=None, **kwargs):
        if is_free is not None and "is_floating" not in kwargs:
            kwargs["is_floating"] = is_free
        return init(self, *args, **kwargs)

    _compat_init._accepts_is_free = True
    mech_base.MechBase.__init__ = _compat_init


class OneKinematics:
    def __init__(self):
        _prepare_one_imports()
        _patch_one_mechbase_compat()
        from one.robots.end_effectors.xhand.xhand_right import XHandRight
        from one.robots.manipulators.xarm.xarm7.xarm7 import XArm7

        self.arm = XArm7()
        self.hand = XHandRight()
        mount_tf = _tf_from_pos_rotmat(np.zeros(3, dtype=np.float32), _rotmat_from_z(MOUNT_RPY))
        self.arm.mount(self.hand, self.arm.runtime_lnks[-1], mount_tf, update=True)
        self.link_by_name = {lnk.name: lnk for lnk in self.hand.runtime_lnks}

        self.lower_one = np.concatenate(
            [self.arm._compiled.jlmt_low_by_idx[self.arm._compiled.active_jnt_ids_mask],
             self.hand._compiled.jlmt_low_by_idx[self.hand._compiled.active_jnt_ids_mask]]
        ).astype(np.float32)
        self.upper_one = np.concatenate(
            [self.arm._compiled.jlmt_high_by_idx[self.arm._compiled.active_jnt_ids_mask],
             self.hand._compiled.jlmt_high_by_idx[self.hand._compiled.active_jnt_ids_mask]]
        ).astype(np.float32)
        self.lower_isaac = self.lower_one[ONE_TO_ISAAC]
        self.upper_isaac = self.upper_one[ONE_TO_ISAAC]

    def update(self, q_isaac: np.ndarray) -> None:
        q_one = q_isaac[ISAAC_TO_ONE]
        self.arm.fk(q_one[:7])
        for mounting in self.arm._mountings.values():
            self.arm._update_mounting(mounting)
        self.hand.fk(q_one[7:])

    def observation_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        pads = []
        for name in EE_BODY_NAMES:
            tf = self.link_by_name[name].tf
            pads.append(_transform_point(tf, np.asarray(FINGER_PAD_OFFSETS[name], dtype=np.float32)))
        ee_pos_b = np.asarray(pads, dtype=np.float32).reshape(-1)
        palm_tf = self.link_by_name["palm"].tf
        palm_center_b = _transform_point(palm_tf, PALM_CENTER_OFFSET).astype(np.float32)
        return ee_pos_b, palm_center_b


class RlGamesMlpPolicy:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.torch = _import_torch()
        self.device = self.torch.device(device)
        try:
            raw = self.torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            raw = self.torch.load(checkpoint, map_location=self.device)
        self.model = raw[0]["model"] if isinstance(raw, dict) and 0 in raw else raw["model"]
        self.obs_mean, self.obs_var = self._load_obs_stats(self.model)
        self.layers = self._load_actor_layers(self.model)

    def _load_obs_stats(self, ckpt):
        mean = ckpt.get("running_mean_std.running_mean")
        var = ckpt.get("running_mean_std.running_var")
        if mean is None or var is None:
            return None, None
        return mean.to(self.device).float(), var.to(self.device).float()

    def _load_actor_layers(self, state):
        candidates = [
            ("a2c_network.actor_mlp", "a2c_network.mu"),
            ("actor_mlp", "mu"),
            ("model.a2c_network.actor_mlp", "model.a2c_network.mu"),
        ]
        for mlp_prefix, mu_prefix in candidates:
            keys = [f"{mlp_prefix}.0.weight", f"{mlp_prefix}.0.bias", f"{mlp_prefix}.2.weight", f"{mlp_prefix}.2.bias",
                    f"{mlp_prefix}.4.weight", f"{mlp_prefix}.4.bias", f"{mu_prefix}.weight", f"{mu_prefix}.bias"]
            if all(k in state for k in keys):
                return [(state[keys[i]].to(self.device).float(), state[keys[i + 1]].to(self.device).float())
                        for i in range(0, len(keys), 2)]
        sample = "\n".join(str(k) for k in list(state.keys())[:80])
        raise RuntimeError(f"Could not identify rl_games actor MLP keys. First checkpoint keys:\n{sample}")

    def act(self, obs: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.obs_mean is not None and self.obs_var is not None:
            x = (x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)
            x = torch.clamp(x, -5.0, 5.0)
        for i, (w, b) in enumerate(self.layers):
            x = torch.nn.functional.linear(x, w, b)
            if i < len(self.layers) - 1:
                x = torch.nn.functional.elu(x)
        return torch.clamp(x, -1.0, 1.0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def make_obs(
    q_isaac: np.ndarray,
    qd_isaac: np.ndarray,
    prev_action: np.ndarray,
    robot_t_cube: np.ndarray,
    kin: OneKinematics,
) -> np.ndarray:
    kin.update(q_isaac)
    ee_pos_b, palm_center_b = kin.observation_geometry()
    object_pos_b = robot_t_cube[:3, 3].astype(np.float32)
    object_quat = _rotmat_to_quat_wxyz(robot_t_cube[:3, :3])
    target_pos = np.array([0.5, 0.0, 0.35], dtype=np.float32)
    target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = np.concatenate(
        [
            q_isaac.astype(np.float32),
            qd_isaac.astype(np.float32),
            ee_pos_b,
            palm_center_b,
            object_pos_b,
            object_quat,
            target_pos,
            target_quat,
            prev_action.astype(np.float32),
        ]
    )
    if obs.shape != (89,):
        raise RuntimeError(f"Expected obs shape (89,), got {obs.shape}")
    return obs


class RealHardware:
    def __init__(self, xarm_ip: str, xhand_port: str):
        _prepare_one_imports()
        from one.control.end_effector.xhand.xhand_x import XHandX
        from one.control.manipulators.xarm7.xarm7 import XArm7X

        self.arm = XArm7X(ip=xarm_ip, reset=False)
        self.hand = XHandX(port=xhand_port, baudrate=3000000)

    def get_q_isaac(self, hand_q_one: np.ndarray) -> np.ndarray:
        q_one = np.concatenate([self.arm.get_jnt_values().astype(np.float32), hand_q_one.astype(np.float32)])
        return q_one[ONE_TO_ISAAC].astype(np.float32)

    def sync_start(self, q_start_isaac: np.ndarray, start_speed: float, hand_speed: float):
        q_one = q_start_isaac[ISAAC_TO_ONE]
        print("[real] moving xArm7 to RL default home...")
        self.arm.move_j(q_one[:7], speed=start_speed, wait=True)
        print("[real] opening XHand to RL default home...")
        self.hand.move_to(q_one[7:], speed=hand_speed, freq=50.0)

    def stream(self, q_cmd_isaac: np.ndarray):
        q_one = q_cmd_isaac[ISAAC_TO_ONE]
        arm_ret = self.arm.servo_j(q_one[:7])
        self.hand.move(q_one[7:], read=False)
        if arm_ret != 0:
            raise RuntimeError(f"xArm servo_j returned {arm_ret}")

    def close(self):
        if getattr(self, "hand", None) is not None:
            self.hand.close()


def main() -> None:
    args = parse_args()
    if args.execute and not args.real:
        raise ValueError("--execute requires --real")

    if args.no_pose_capture:
        cam_t_cube = np.load(args.pose_npy).astype(np.float32)
    else:
        from foundationpose_then_play import capture_pose

        cam_t_cube = capture_pose(args).astype(np.float32)
        _maybe_reexec_one_for_real(args)

    _maybe_reexec_one_for_real(args)

    # FoundationPose returns the cube pose in the D435 camera frame. Map it into the
    # xArm base frame (the frame OneKinematics / the policy observation lives in) using
    # the eye-to-hand calibration: base_T_cube = T_base_cam @ camera_T_cube.
    if args.no_calib:
        robot_t_cube = cam_t_cube
    else:
        base_T_cam = load_base_T_cam(args.calib_yaml)
        robot_t_cube = (base_T_cam @ cam_t_cube).astype(np.float32)
        c = cam_t_cube[:3, 3]
        b = robot_t_cube[:3, 3]
        print(f"[calib] camera_T_cube xyz(m): {c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}")
        print(f"[calib] base_T_cube   xyz(m): {b[0]:+.3f} {b[1]:+.3f} {b[2]:+.3f}")

    kin = OneKinematics()
    policy = RlGamesMlpPolicy(args.checkpoint, device="cpu")

    q_target = DEFAULT_Q_ISAAC.copy()
    q_prev = q_target.copy()
    qd = np.zeros(19, dtype=np.float32)
    prev_action = np.zeros(19, dtype=np.float32)

    hw = None
    hand_q_one = DEFAULT_Q_ISAAC[ISAAC_TO_ONE][7:].copy()
    if args.real:
        hw = RealHardware(args.xarm_ip, args.xhand_port)
        if not args.skip_start_sync:
            print("[real] About to sync robot to RL default start pose.")
            print("[real] Make sure the workspace is clear and E-stop is reachable.")
            if args.execute:
                input("[real] Press ENTER to move to start pose, or Ctrl-C to abort.")
                hw.sync_start(DEFAULT_Q_ISAAC, args.start_speed, args.hand_start_speed)
            else:
                print("[dry-run] --real set without --execute; not moving start pose.")
        q_target = hw.get_q_isaac(hand_q_one)
        q_prev = q_target.copy()

    print("[run] starting policy loop")
    if args.real and args.execute:
        input("[real] Press ENTER to start streaming policy commands, or Ctrl-C to abort.")

    dt = 1.0 / args.rate
    next_t = time.perf_counter()
    try:
        for step in range(args.steps):
            qd = (q_target - q_prev) / dt
            obs = make_obs(q_target, qd, prev_action, robot_t_cube, kin)
            action = policy.act(obs)
            raw_target = np.clip(q_target + 0.1 * action, kin.lower_isaac, kin.upper_isaac)
            q_next = 0.3 * raw_target + 0.7 * q_target
            q_next[:7] = _clip_step(q_next[:7], q_target[:7], args.max_arm_step)
            q_next[7:] = _clip_step(q_next[7:], q_target[7:], args.max_hand_step)
            q_next = np.clip(q_next, kin.lower_isaac, kin.upper_isaac).astype(np.float32)

            if hw is not None and args.execute:
                hw.stream(q_next)
                hand_q_one = q_next[ISAAC_TO_ONE][7:]
            elif step % max(1, args.dry_print_every) == 0:
                print(
                    f"[dry-run] step={step:04d} "
                    f"arm={np.array2string(q_next[:7], precision=3)} "
                    f"hand={np.array2string(q_next[7:], precision=3)}"
                )

            q_prev = q_target
            q_target = q_next
            prev_action = action
            next_t += dt
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()
    finally:
        if hw is not None:
            hw.close()


if __name__ == "__main__":
    main()
