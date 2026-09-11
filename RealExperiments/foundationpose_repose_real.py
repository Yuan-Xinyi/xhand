#!/usr/bin/env python3
"""Sim2real pipeline: FoundationPose cube tracking -> repose_cube LSTM policy -> XHand.

Architecture (two processes, UDP IPC — same pattern as the teleop pipeline):

  [env_isaaclab]  foundationpose_repose_tracker.py
      D435 RGB-D -> FoundationPose register + continuous track
      -> UDP 127.0.0.1:9877  (seq, t, camera_T_cube 4x4)

  [one]           this script (control loop, 20 Hz = sim env step_dt)
      XHand joints (open-loop targets) -> one-lib FK -> fingertip positions
      cube pose -> calib (T_base_cam) -> env frame (palm alignment)
      obs(34) -> LSTM policy -> action -> position targets -> XHand

The env frame is defined so the real palm coincides with the sim palm pose
(pos (0,0,0.5), quat (0.7071,-0.7071,0,0), palm up, fingers +Y). The xArm7 only
HOLDS the wrist there; it is never commanded by the policy.

Default is a dry-run (no hardware, no camera needed with --pose-source synthetic).
Real execution requires --real --execute and confirms before each motion.

Examples:
    # offline smoke test (no camera, no robot)
    python RealExperiments/foundationpose_repose_real.py --pose-source synthetic --steps 200

    # camera + policy, print commands, no robot
    python RealExperiments/foundationpose_repose_real.py --steps 400

    # full real run
    python RealExperiments/foundationpose_repose_real.py --real --execute
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import types
from pathlib import Path

# An activated env_isaaclab shell exports Isaac's pip_prebundle (cp311) on
# PYTHONPATH, which breaks numpy/torch for the one env's py312 interpreter.
# If numpy fails to import, strip those entries and retry.
def _sanitize_and_import_numpy():
    try:
        import numpy

        return numpy
    except ImportError:
        bad = [p for p in sys.path if "pip_prebundle" in p or "pip_archive" in p or "isaac" in p.lower()]
        for p in bad:
            sys.path.remove(p)
        keep = [e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e and e not in bad]
        os.environ["PYTHONPATH"] = os.pathsep.join(keep)
        for m in [m for m in sys.modules if m == "numpy" or m.startswith("numpy.")]:
            del sys.modules[m]
        import numpy

        print("[env] stripped incompatible Isaac pip_prebundle paths from sys.path")
        return numpy


np = _sanitize_and_import_numpy()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = str(
    REPO_ROOT / "logs/rl_games/xhand_repose_openai_lstm/0_2026-06-26_17-44-10/nn/xhand_repose_openai_lstm.pth"
)
DEFAULT_CALIB_YAML = "/home/lqin/one-dexhand/one/camera/RS435/camera_extrinsics.yaml"
TRACKER_SCRIPT = str(Path(__file__).resolve().parent / "foundationpose_repose_tracker.py")

# ---------------------------------------------------------------------------
# Contract constants (verified against the sim via repose_probe_dump.py)
# ---------------------------------------------------------------------------

# Isaac joint order of the standalone XHand USD (= action order, = limit order)
ISAAC12 = [
    "index_joint0", "middle_joint0", "pinky_joint0", "ring_joint0", "thumb_joint0",
    "index_joint1", "middle_joint1", "pinky_joint1", "ring_joint1", "thumb_joint1",
    "index_joint2", "thumb_joint2",
]
# one-library XHand joint order (thumb..pinky, as in foundationpose_then_real.py)
ONE12 = [
    "thumb_joint0", "thumb_joint1", "thumb_joint2",
    "index_joint0", "index_joint1", "index_joint2",
    "middle_joint0", "middle_joint1",
    "ring_joint0", "ring_joint1",
    "pinky_joint0", "pinky_joint1",
]
ISAAC_TO_ONE = np.array([ISAAC12.index(n) for n in ONE12], dtype=np.int64)
ONE_TO_ISAAC = np.array([ONE12.index(n) for n in ISAAC12], dtype=np.int64)

# joint limits in ISAAC12 order (repose_probe_dump.py)
LOWER = np.array([-0.175, 0, 0, 0, 0, 0, 0, 0, 0, -1.05, 0, -0.175], dtype=np.float32)
UPPER = np.array([0.175, 1.92, 1.92, 1.92, 1.83, 1.92, 1.92, 1.92, 1.92, 1.57, 1.92, 1.83], dtype=np.float32)

# fingertip bodies in OBSERVATION order (sorted by Isaac body index)
FINGERTIP_BODIES = ["mid_link2", "pinky_link2", "ring_link2", "index_rota_link2", "thumb_rota_link2"]

# sim palm pose in the env frame (XHandReposeEnvCfg.robot_cfg.init_state)
SIM_PALM_POS = np.array([0.0, 0.0, 0.5], dtype=np.float32)
SIM_PALM_QUAT = np.array([0.7071, -0.7071, 0.0, 0.0], dtype=np.float32)  # wxyz

IN_HAND_POS = np.array([0.0, 0.1, 0.51], dtype=np.float32)  # env frame
# settled rest position of the cube in the open palm (measured in sim, 5 s hold)
REST_POS = np.array([0.0, 0.101, 0.551], dtype=np.float32)
FALL_DIST = 0.24
ACT_MOVING_AVERAGE = 0.3  # XHandReposeOpenAIEnvCfg
STEP_DT = 0.05  # 20 Hz (sim dt 1/60 * decimation 3)
DEFAULT_SUCCESS_TOL = 0.4  # rad, trained tolerance (curriculum start; leave margin on real)

# xArm7 home that holds the hand (only used when the arm is not connected)
DEFAULT_ARM_Q = np.array([0.0, -0.7494, 0.0, 1.1920, 0.0, 1.9414, 0.0], dtype=np.float32)

MOUNT_RPY = 4.71239  # link8 -> palm fixed yaw (xarm7_xhand.urdf hand_mount)
MOUNT_DZ = 0.020  # adapter plate between the xArm7 TCP flange and the XHand (tuned on live render)

UDP_ADDR = ("127.0.0.1", 9877)
POSE_FMT = "<18d"  # seq, t, 16 pose floats
GOAL_UDP = ("127.0.0.1", 9878)
GOAL_FMT = "<11d"  # t, R_cam_goal (9), rot_dist (rad, <0 = unknown)
STATE_UDP = ("127.0.0.1", 9879)
STATE_FMT = "<29d"  # t, hand_q (12), env_T_cube (16) — mirror-viewer stream
CALIB_UDP = ("127.0.0.1", 9880)
CALIB_FMT = "<6d"  # dx, dy, dz [m], rx, ry, rz [rad] — manual calib panel
CALIB_COMMIT_FMT = "<7d"  # + flag: 1.0 = saved -> re-run auto-center
MANUAL_CALIB_YAML = str(Path(__file__).resolve().parent / "repose_manual_calib.yaml")


# ---------------------------------------------------------------------------
# math helpers (wxyz quaternions, matching isaaclab.utils.math)
# ---------------------------------------------------------------------------

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), *(axis * s)], dtype=np.float64)


def rotation_distance(q_obj: np.ndarray, q_goal: np.ndarray) -> float:
    qd = quat_mul(np.asarray(q_obj, dtype=np.float64), quat_conj(np.asarray(q_goal, dtype=np.float64)))
    return 2.0 * math.asin(min(1.0, float(np.linalg.norm(qd[1:4]))))


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
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
    return q / np.linalg.norm(q)


def tf_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = quat_to_rotmat(quat_wxyz)
    tf[:3, 3] = np.asarray(pos, dtype=np.float64)
    return tf


def extrapolate_pose(pose_prev: np.ndarray, pose_now: np.ndarray, dt_pair: float, lead: float,
                     max_angle: float = 0.3, max_shift: float = 0.03) -> np.ndarray:
    """Constant-velocity extrapolation of a pose by `lead` seconds (latency compensation).

    Rotation delta is scaled via axis-angle; both the extrapolated angle and the
    translation shift are capped so tracking noise cannot be amplified unboundedly.
    """
    if dt_pair <= 1e-4 or lead <= 0.0:
        return pose_now
    s = lead / dt_pair
    out = pose_now.copy()
    # rotation: R_delta = R_now @ R_prev^T -> axis-angle -> scale -> apply
    r_delta = pose_now[:3, :3] @ pose_prev[:3, :3].T
    cos_a = max(-1.0, min(1.0, (np.trace(r_delta) - 1.0) / 2.0))
    angle = math.acos(cos_a)
    if angle > 1e-6:
        axis = np.array([r_delta[2, 1] - r_delta[1, 2],
                         r_delta[0, 2] - r_delta[2, 0],
                         r_delta[1, 0] - r_delta[0, 1]])
        axis /= max(np.linalg.norm(axis), 1e-9)
        lead_angle = min(angle * s, max_angle)
        out[:3, :3] = _axis_angle_rotmat(axis, lead_angle) @ pose_now[:3, :3]
    # translation
    shift = (pose_now[:3, 3] - pose_prev[:3, 3]) * s
    norm = np.linalg.norm(shift)
    if norm > max_shift:
        shift *= max_shift / norm
    out[:3, 3] = pose_now[:3, 3] + shift
    return out


def tf_inv(tf: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    r = tf[:3, :3].T
    out[:3, :3] = r
    out[:3, 3] = -r @ tf[:3, 3]
    return out


def scale_action(a: np.ndarray) -> np.ndarray:
    """[-1,1] -> absolute joint targets (InHandManipulationEnv contract)."""
    return 0.5 * (a + 1.0) * (UPPER - LOWER) + LOWER


def sample_goal_quat_mode(rng: np.random.Generator, mode: str) -> np.ndarray:
    """Goal sampling constrained to sim-verified difficulty tiers.

    Bucket study (256 envs each, deployment obs stack): yaw 45/90 deg and
    roll(x) 45-180 deg all >=99.6% success; pitch(y) 90/180 and yaw 180 are
    the hard tail (77-89%).
    """
    if mode == "yaw":
        ang = rng.uniform(np.radians(30), np.radians(90)) * rng.choice([-1.0, 1.0])
        return quat_from_angle_axis(ang, np.array([0.0, 0.0, 1.0]))
    if mode == "easy":
        if rng.random() < 0.5:
            ang = rng.uniform(np.radians(30), np.radians(90)) * rng.choice([-1.0, 1.0])
            return quat_from_angle_axis(ang, np.array([0.0, 0.0, 1.0]))
        ang = rng.uniform(np.radians(45), np.radians(180)) * rng.choice([-1.0, 1.0])
        return quat_from_angle_axis(ang, np.array([1.0, 0.0, 0.0]))
    return sample_goal_quat(rng)


def sample_goal_quat(rng: np.random.Generator) -> np.ndarray:
    """Same distribution as the env's randomize_rotation."""
    r0, r1 = rng.uniform(-1.0, 1.0, 2)
    return quat_mul(
        quat_from_angle_axis(r0 * math.pi, np.array([1.0, 0.0, 0.0])),
        quat_from_angle_axis(r1 * math.pi, np.array([0.0, 1.0, 0.0])),
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# observation builder (env obs_type="openai", 34-D)
# ---------------------------------------------------------------------------

def build_obs(
    tip_pos_env: np.ndarray,  # (5,3) fingertip link origins, env frame, FINGERTIP_BODIES order
    obj_pos_env: np.ndarray,  # (3,)
    obj_quat_env: np.ndarray,  # (4,) wxyz
    goal_quat: np.ndarray,  # (4,) wxyz
    prev_action: np.ndarray,  # (12,) ISAAC12 order
) -> np.ndarray:
    rel_quat = quat_mul(np.asarray(obj_quat_env, dtype=np.float64), quat_conj(goal_quat))
    obs = np.concatenate(
        [
            np.asarray(tip_pos_env, dtype=np.float32).reshape(15),
            np.asarray(obj_pos_env, dtype=np.float32),
            rel_quat.astype(np.float32),
            np.asarray(prev_action, dtype=np.float32),
        ]
    )
    assert obs.shape == (34,), obs.shape
    return obs


# ---------------------------------------------------------------------------
# rl_games LSTM actor (obs34 -> LSTM1024+LN -> Linear512+ReLU -> mu12)
# ---------------------------------------------------------------------------

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


class LstmPolicy:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        torch = _import_torch()
        self.torch = torch
        self.device = torch.device(device)
        try:
            raw = torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            raw = torch.load(checkpoint, map_location=self.device)
        state = raw[0]["model"] if isinstance(raw, dict) and 0 in raw else raw["model"]

        self.obs_mean = state["running_mean_std.running_mean"].to(self.device).float()
        self.obs_var = state["running_mean_std.running_var"].to(self.device).float()

        obs_dim = self.obs_mean.shape[0]
        hidden = state["a2c_network.rnn.rnn.weight_hh_l0"].shape[1]
        self.lstm = torch.nn.LSTM(obs_dim, hidden, num_layers=1, batch_first=False).to(self.device)
        with torch.no_grad():
            self.lstm.weight_ih_l0.copy_(state["a2c_network.rnn.rnn.weight_ih_l0"])
            self.lstm.weight_hh_l0.copy_(state["a2c_network.rnn.rnn.weight_hh_l0"])
            self.lstm.bias_ih_l0.copy_(state["a2c_network.rnn.rnn.bias_ih_l0"])
            self.lstm.bias_hh_l0.copy_(state["a2c_network.rnn.rnn.bias_hh_l0"])
        self.lstm.eval()
        self.ln_w = state["a2c_network.layer_norm.weight"].to(self.device).float()
        self.ln_b = state["a2c_network.layer_norm.bias"].to(self.device).float()
        self.mlp_w = state["a2c_network.actor_mlp.0.weight"].to(self.device).float()
        self.mlp_b = state["a2c_network.actor_mlp.0.bias"].to(self.device).float()
        self.mu_w = state["a2c_network.mu.weight"].to(self.device).float()
        self.mu_b = state["a2c_network.mu.bias"].to(self.device).float()
        self.hidden = hidden
        # Training sampled actions from N(mu, sigma); a deterministic (mu-only)
        # deployment can lock into fixed points where obs stops changing ->
        # action stops changing -> the cube wedges. Sim A/B (256 envs, flat
        # start, yaw goals): deterministic 79.7% with 2.0% frozen steps vs
        # 93.0% / 0.2% with the learned sigma, 96.5% with sigma=0.10.
        sig = state.get("a2c_network.sigma")
        self.sigma = torch.exp(sig).to(self.device).float() if sig is not None else None
        self.noise_scale = 0.0
        self.rng = np.random.default_rng(0)
        self.reset()
        print(f"[policy] loaded {checkpoint}")
        print(f"[policy] obs_dim={obs_dim} lstm_hidden={hidden} actions={self.mu_b.shape[0]}")

    def reset(self):
        torch = self.torch
        self.h = torch.zeros(1, 1, self.hidden, device=self.device)
        self.c = torch.zeros(1, 1, self.hidden, device=self.device)

    def act(self, obs: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            x = (x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)
            x = torch.clamp(x, -5.0, 5.0).view(1, 1, -1)  # (seq=1, batch=1, obs)
            out, (self.h, self.c) = self.lstm(x, (self.h, self.c))
            out = torch.nn.functional.layer_norm(out.view(-1), (self.hidden,), self.ln_w, self.ln_b)
            out = torch.nn.functional.relu(torch.nn.functional.linear(out, self.mlp_w, self.mlp_b))
            mu = torch.nn.functional.linear(out, self.mu_w, self.mu_b)
            a = torch.clamp(mu, -1.0, 1.0).cpu().numpy().astype(np.float32)
        if self.noise_scale > 0.0:
            std = (self.sigma.cpu().numpy() if self.sigma is not None else np.full(a.shape, 0.09))
            a = np.clip(a + self.rng.normal(0.0, std * self.noise_scale).astype(np.float32), -1.0, 1.0)
        return a.astype(np.float32)


# ---------------------------------------------------------------------------
# self-contained URDF forward kinematics (same URDF that generated the Isaac USD)
# ---------------------------------------------------------------------------

URDF_PATH = str(REPO_ROOT / "source/xhand_inhand/xhand_inhand/assets/xarm7_xhand/xarm7_xhand.urdf")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


def _rpy_to_rotmat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle_rotmat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


class UrdfKinematics:
    """FK over xarm7_xhand.urdf (base frame = URDF root = xArm base).

    Returns link-frame poses, which is exactly what the Isaac obs uses
    (fingertip body positions are USD link origins).
    """

    def __init__(self, urdf_path: str = URDF_PATH):
        import xml.etree.ElementTree as ET

        root = ET.parse(urdf_path).getroot()
        self.joints = []  # (name, type, parent, child, T_origin(4x4), axis(3))
        for j in root.findall("joint"):
            name = j.get("name")
            jtype = j.get("type")
            parent = j.find("parent").get("link")
            child = j.find("child").get("link")
            origin = j.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            tf = np.eye(4)
            tf[:3, :3] = _rpy_to_rotmat(*rpy)
            tf[:3, 3] = xyz
            axis_el = j.find("axis")
            axis = np.array(
                [float(v) for v in (axis_el.get("xyz") if axis_el is not None else "1 0 0").split()]
            )
            self.joints.append((name, jtype, parent, child, tf, axis))
        children = {j[3] for j in self.joints}
        parents = {j[2] for j in self.joints}
        roots = parents - children
        assert len(roots) == 1, f"URDF should have one root, got {roots}"
        self.root_link = roots.pop()
        self.link_tf: dict[str, np.ndarray] = {}

    def update(self, arm_q7: np.ndarray, hand_q_isaac: np.ndarray) -> None:
        qmap = {n: float(v) for n, v in zip(ARM_JOINTS, arm_q7)}
        qmap.update({n: float(v) for n, v in zip(ISAAC12, hand_q_isaac)})
        self.link_tf = {self.root_link: np.eye(4)}
        pending = list(self.joints)
        while pending:
            progressed = False
            rest = []
            for name, jtype, parent, child, t_origin, axis in pending:
                if parent not in self.link_tf:
                    rest.append((name, jtype, parent, child, t_origin, axis))
                    continue
                tf = self.link_tf[parent] @ t_origin
                if jtype == "revolute" and name in qmap:
                    rot = np.eye(4)
                    rot[:3, :3] = _axis_angle_rotmat(axis, qmap[name])
                    tf = tf @ rot
                self.link_tf[child] = tf
                progressed = True
            pending = rest
            if not progressed and pending:
                raise RuntimeError(f"URDF kinematic tree disconnected at {[p[0] for p in pending]}")

    def palm_tf_base(self) -> np.ndarray:
        return self.link_tf["palm"]

    def fingertip_pos_base(self) -> np.ndarray:
        return np.stack([self.link_tf[n][:3, 3] for n in FINGERTIP_BODIES])


# ---------------------------------------------------------------------------
# one-library imports (hardware drivers only)
# ---------------------------------------------------------------------------

def _prepare_one_imports() -> None:
    """Make the one-lib DRIVER modules importable WITHOUT executing one/__init__.py.

    The package init imports pyglet and builds a GL context at import time,
    which dies in headless shells / under VRAM pressure (EGL NoSuchConfig).
    The drivers themselves (xhand_x, xarm7) only need numpy/scipy/serial/xarm,
    and every package level under one/control is a namespace package — so a
    shell module carrying just __path__ lets them resolve cleanly.
    """
    # the dex-hand branch worktree carries the XHand model/driver sources
    one_root = os.environ.get("ONE_ROOT")
    if not one_root:
        one_root = "/home/lqin/one-dexhand" if os.path.isdir("/home/lqin/one-dexhand/one") else "/home/lqin/one"
    if one_root not in sys.path:
        sys.path.insert(0, one_root)
    if "one" not in sys.modules:
        shell = types.ModuleType("one")
        shell.__path__ = [os.path.join(one_root, "one")]
        shell.__package__ = "one"
        sys.modules["one"] = shell


# ---------------------------------------------------------------------------
# cube pose sources
# ---------------------------------------------------------------------------

class UdpPoseReceiver:
    """Latest camera_T_cube from the tracker process."""

    def __init__(self, addr=UDP_ADDR):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(addr)
        self.sock.setblocking(False)
        self.pose = None
        self.seq = -1
        self.stamp = 0.0

    def poll(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            vals = struct.unpack(POSE_FMT, data)
            self.seq = int(vals[0])
            self.stamp = vals[1]
            self.pose = np.array(vals[2:], dtype=np.float64).reshape(4, 4)
        return self.pose

    def age(self) -> float:
        return time.time() - self.stamp if self.pose is not None else float("inf")

    def wait_first(self, timeout: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.poll() is not None:
                return True
            time.sleep(0.1)
        return False

    def wait_fresh(self, max_age: float, timeout: float) -> bool:
        """Drain stale packets and wait for a genuinely fresh one.

        Needed after any long blocking pause (input() prompts): when the UDP
        receive buffer fills, the kernel drops NEW packets and keeps OLD ones,
        so the freshest packet in the buffer can be arbitrarily stale.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.poll()  # drains the whole buffer
            if self.age() <= max_age:
                return True
            time.sleep(0.02)
        return False


class SyntheticPose:
    """Slowly tumbling cube at the in-hand position, directly in the ENV frame."""

    def __init__(self):
        self.t0 = time.time()

    def pose_env(self):
        t = time.time() - self.t0
        q = quat_mul(
            quat_from_angle_axis(0.3 * t, np.array([0.0, 0.0, 1.0])),
            quat_from_angle_axis(0.15 * t, np.array([1.0, 0.0, 0.0])),
        )
        pos = IN_HAND_POS + np.array([0.0, 0.0, 0.04])
        return tf_from_pos_quat(pos, q)


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------

class RealHardware:
    def __init__(self, xarm_ip: str, xhand_port: str):
        _prepare_one_imports()
        from one.control.end_effector.xhand.xhand_x import XHandX
        from one.control.manipulators.xarm7.xarm7 import XArm7X

        self.arm = XArm7X(ip=xarm_ip, reset=False)
        self.hand = XHandX(port=xhand_port, baudrate=3000000)

    def arm_q(self) -> np.ndarray:
        return self.arm.get_jnt_values().astype(np.float32)

    def hand_home(self, q_isaac: np.ndarray, speed: float):
        self.hand.move_to(q_isaac[ISAAC_TO_ONE], speed=speed, freq=50.0)

    def hand_stream(self, q_isaac: np.ndarray, read: bool = False) -> np.ndarray | None:
        """Stream targets; with read=True also return the MEASURED joints (Isaac order)."""
        states = self.hand.move(q_isaac[ISAAC_TO_ONE], read=read)
        if not read or states is None:
            return None
        try:
            q_one = np.array([float(s.position) for s in states], dtype=np.float32)
        except (AttributeError, TypeError):
            return None
        return q_one[ONE_TO_ISAAC]

    def close(self):
        if getattr(self, "hand", None) is not None:
            self.hand.close()


def load_base_T_cam(calib_yaml: str) -> np.ndarray:
    import yaml

    path = Path(calib_yaml).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"camera calibration not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    entry = data["T_base_cam"]
    mat = np.array(entry["matrix"] if isinstance(entry, dict) else entry, dtype=np.float64)
    if mat.shape != (4, 4):
        raise RuntimeError(f"T_base_cam must be 4x4, got {mat.shape}")
    return mat


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--steps", type=int, default=1200, help="control steps at 20 Hz (1200 = 60 s)")
    p.add_argument("--success-tol", type=float, default=DEFAULT_SUCCESS_TOL, help="rad")
    p.add_argument("--goal-seed", type=int, default=0)
    p.add_argument("--action-noise", type=float, default=1.0,
                   help="exploration dither as a multiple of the policy's trained sigma "
                        "(1.0 = training-like, 0 = deterministic). Deterministic locks into "
                        "frozen fixed points: sim 79.7%% vs 93.0%% with dither.")
    p.add_argument("--goal-mode", choices=["all", "easy", "yaw"], default="all",
                   help="goal difficulty tier: yaw = vertical-axis 30-90 deg only (sim ~100%%), "
                        "easy = yaw + roll(x) (sim >=99.6%%), all = uniform random")
    p.add_argument("--cube-edge", type=float, default=0.06,
                   help="physical cube edge length [m]. Scales the FoundationPose mesh and "
                        "shifts the auto-center rest anchor. Trained size DR band: 0.051-0.069 m.")
    p.add_argument("--stall-timeout", type=float, default=10.0,
                   help="s without a success -> reopen hand, reset LSTM, new goal (sim episodes "
                        "reset every 8 s, the policy never trained past that; 0 = off)")
    p.add_argument("--pose-source", choices=["tracker", "npy", "synthetic"], default="tracker")
    p.add_argument("--pose_npy", default="/tmp/foundationpose_cube_pose.npy")
    p.add_argument("--calib_yaml", default=DEFAULT_CALIB_YAML)
    p.add_argument("--no_calib", action="store_true", help="cube pose already in the xArm base frame")
    p.add_argument("--no-palm-calib", action="store_true",
                   help="ignore palm_env_T_cam.yaml (fall back to camera_extrinsics + arm FK)")
    p.add_argument("--auto-center", action="store_true",
                   help="OPT-IN startup position zeroing. Off by default: the extrinsic is measured "
                        "(table board + 2 cm plate), so the geometric truth needs no anchoring.")
    p.add_argument("--max-pose-age", type=float, default=0.25, help="hold targets if pose older than this [s]")
    p.add_argument("--abort-pose-age", type=float, default=2.0, help="abort real run if pose older than this [s]")
    p.add_argument("--real", action="store_true", help="connect xArm7 (read) + XHand")
    p.add_argument("--execute", action="store_true", help="actually stream commands; requires --real")
    p.add_argument("--xarm-ip", default="192.168.1.205")
    p.add_argument("--xhand-port", default="/dev/ttyUSB0")
    p.add_argument("--hand-start-speed", type=float, default=0.25)
    p.add_argument("--skip-home", action="store_true",
                   help="do not move the hand to the open home pose at startup (NOT recommended: "
                        "the obs and the mirror assume the hand starts at home)")
    p.add_argument("--no-hand-read", action="store_true",
                   help="disable joint read-back (obs falls back to commanded targets)")
    p.add_argument("--pose-lead", type=float, default=0.10,
                   help="extrapolate the cube pose forward by this many seconds to cancel "
                        "the camera->action latency (0 = off)")
    # 0.157 = sim actuator velocity limit (3.14 rad/s) x step_dt (0.05 s). Clamping much
    # tighter starves the policy's stroke depth: the cube gets rocked +-2 deg and springs
    # back instead of tipping over an edge (measured in /tmp/repose_run1.npz: gross
    # rotation ~100 deg / 5 s, net ~2 deg with a 0.03 clamp).
    p.add_argument("--max-hand-step", type=float, default=0.157, help="max joint target delta per cycle [rad]")
    p.add_argument("--arm-q", type=float, nargs=7, default=None,
                   help="arm joints holding the wrist (dry-run only; --real reads the robot)")
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--log-npz", default=None, help="record obs/actions/poses for offline analysis")
    # tracker passthrough
    p.add_argument("--roi", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"))
    p.add_argument("--serial", default=None, help="RealSense serial")
    p.add_argument("--no-view", action="store_true", help="tracker: no live overlay window")
    p.add_argument("--mirror", action="store_true",
                   help="spawn the Isaac mirror viewer (renders what the pipeline believes: "
                        "measured hand joints + estimated cube pose) for calibration checking")
    p.add_argument("--tracker-timeout", type=float, default=300.0)
    return p.parse_args()


def _maybe_reexec_one(args: argparse.Namespace) -> None:
    """Hardware IO needs the one env; re-exec there (perception runs in a subprocess)."""
    if not args.real:
        return
    one_python = os.environ.get("ONE_PYTHON", "/home/lqin/miniconda3/envs/one/bin/python")
    if not Path(one_python).exists():
        return
    if os.path.realpath(sys.executable) == os.path.realpath(one_python):
        return
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    keep = [e for e in pythonpath.split(os.pathsep)
            if e and "isaac" not in e.lower() and "omni.kit" not in e.lower() and "pip_prebundle" not in e.lower()]
    if keep:
        env["PYTHONPATH"] = os.pathsep.join(keep)
    else:
        env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PATH"] = os.path.dirname(one_python) + os.pathsep + env.get("PATH", "")
    print(f"[real] re-exec into one python: {one_python}")
    os.execve(one_python, [one_python, os.path.abspath(__file__)] + sys.argv[1:], env)


FP_MESH_DIR = "/disk2/FoundationPose/cube/mesh"


def scaled_cube_mesh(edge: float) -> str:
    """Write a vertex-scaled copy of the FoundationPose cube obj (same dir, so the
    relative mtl/texture references keep working) and return its path."""
    scale = edge / 0.06
    out = os.path.join(FP_MESH_DIR, f"textured_scaled_{int(round(edge * 1000))}mm.obj")
    if not os.path.exists(out):
        with open(os.path.join(FP_MESH_DIR, "textured.obj")) as f_in, open(out, "w") as f_out:
            for line in f_in:
                if line.startswith("v "):
                    _, x, y, z = line.split()[:4]
                    f_out.write(f"v {float(x) * scale:.8f} {float(y) * scale:.8f} {float(z) * scale:.8f}\n")
                else:
                    f_out.write(line)
        print(f"[mesh] wrote scaled cube mesh ({edge * 1000:.0f} mm): {out}")
    return out


def spawn_tracker(args: argparse.Namespace) -> subprocess.Popen:
    cmd = (
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && "
        f"exec python {TRACKER_SCRIPT}"
    )
    if abs(args.cube_edge - 0.06) > 1e-6:
        cmd += f" --mesh_file {scaled_cube_mesh(args.cube_edge)}"
    if args.roi is not None:
        cmd += " --roi " + " ".join(str(v) for v in args.roi)
    if args.serial:
        cmd += f" --serial {args.serial}"
    if args.no_view:
        cmd += " --no-view"
    print("[tracker] spawning FoundationPose tracker (env_isaaclab)...")
    return subprocess.Popen(["bash", "-c", cmd], start_new_session=True)


def spawn_mirror(args: argparse.Namespace) -> subprocess.Popen:
    mirror_script = str(Path(__file__).resolve().parent / "repose_mirror_viewer.py")
    cmd = (
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && "
        f"exec python {mirror_script} --cube-edge {args.cube_edge}"
    )
    print("[mirror] spawning Isaac mirror viewer (env_isaaclab, GUI)...")
    return subprocess.Popen(["bash", "-c", cmd], start_new_session=True)


PIPELINE_VERSION = "2026-09-10-r6 (calib-persist-check)"


def main() -> None:
    args = parse_args()
    print(f"[version] {PIPELINE_VERSION}")
    if args.execute and not args.real:
        raise ValueError("--execute requires --real")
    _maybe_reexec_one(args)

    if not 0.051 <= args.cube_edge <= 0.069:
        print(f"[WARN] cube edge {args.cube_edge * 1000:.0f} mm is OUTSIDE the trained size band "
              "(51-69 mm) — the policy has never seen this size; sim-check it first.")

    # --- kinematics + frames ----------------------------------------------
    kin = UrdfKinematics()
    hw = None
    if args.real:
        hw = RealHardware(args.xarm_ip, args.xhand_port)
        arm_q = hw.arm_q()
        print(f"[real] arm q: {np.array2string(arm_q, precision=4)}")
    else:
        arm_q = np.array(args.arm_q, dtype=np.float32) if args.arm_q else DEFAULT_ARM_Q
        print(f"[dry] arm q (assumed): {np.array2string(arm_q, precision=4)}")

    hand_q = np.zeros(12, dtype=np.float32)  # repose home: all joints 0 (open)
    kin.update(arm_q, hand_q)
    base_T_palm = kin.palm_tf_base()
    env_T_base = tf_from_pos_quat(SIM_PALM_POS, SIM_PALM_QUAT) @ tf_inv(base_T_palm)

    # gravity direction check: env -Z must stay -Z, else the sim2real gap is large
    g_env = env_T_base[:3, :3] @ np.array([0.0, 0.0, -1.0])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g_env[2]))))
    print(f"[frames] palm(base): xyz={np.array2string(base_T_palm[:3, 3], precision=3)}")
    print(f"[frames] gravity tilt vs sim: {tilt:.1f} deg" + ("  <-- WARNING: palm not palm-up like sim!" if tilt > 8.0 else ""))

    base_T_cam = None
    palm_yaml = Path(__file__).resolve().parent / "palm_env_T_cam.yaml"
    if (args.pose_source in ("tracker", "npy") and not args.no_calib
            and palm_yaml.exists() and not args.no_palm_calib):
        import yaml as _y

        with open(palm_yaml) as f:
            _d = _y.safe_load(f)
        if "base_T_cam" in _d:
            # hand-eye result: pose-independent, compose with runtime FK
            base_T_cam = np.array(_d["base_T_cam"]["matrix"], dtype=np.float64)
            _res = _d.get("residuals", {})
            print(f"[calib] hand-eye base_T_cam loaded ({_d.get('stamp')}, {_d.get('poses_used')} poses, "
                  f"residual max {_res.get('rot_deg_max', 0):.2f} deg / {_res.get('pos_mm_max', 0):.1f} mm)"
                  " — overrides camera_extrinsics.yaml")
        else:
            env_T_base = np.array(_d["env_T_cam"]["matrix"], dtype=np.float64)  # env <- cam directly
            base_T_cam = None  # cam pose feeds straight into env_T_base
            print(f"[calib] palm-marker env_T_cam loaded ({_d.get('stamp')}, spread "
                  f"{_d.get('rot_spread_deg', 0):.2f} deg) — overrides camera_extrinsics + arm FK chain")
    elif args.pose_source in ("tracker", "npy") and not args.no_calib:
        base_T_cam = load_base_T_cam(args.calib_yaml)

    # --- policy (load before any hardware motion so failures abort early) --
    policy = LstmPolicy(args.checkpoint)
    policy.noise_scale = float(args.action_noise)
    if policy.noise_scale > 0:
        sig = policy.sigma.cpu().numpy() if policy.sigma is not None else np.full(12, 0.09)
        print(f"[policy] action dither ON: {args.action_noise:.2f} x trained sigma "
              f"(mean {sig.mean() * args.action_noise:.3f})")
    else:
        print("[policy][WARN] deterministic actions — prone to frozen fixed points (sim: -13%% success)")
    rng = np.random.default_rng(args.goal_seed)
    goal_quat = sample_goal_quat_mode(rng, args.goal_mode)
    successes = 0

    # --- hand to home + cube placement (BEFORE the tracker: the ROI must be
    # drawn around the cube already resting in the palm) ---------------------
    # Homing is NOT gated on --execute: the observation model and the mirror
    # viewer assume the hand starts at the open home pose, so a dry-run with a
    # mis-posed hand would compare apples to oranges.
    if hw is not None and not args.skip_home:
        input("[real] ENTER to move XHand to open home pose (cube NOT in hand yet)...")
        hw.hand_home(hand_q, args.hand_start_speed)
    if hw is not None:
        input("[real] place the cube at rest in the palm, then ENTER to start the tracker...")

    # --- pose source -------------------------------------------------------
    tracker_proc = None
    receiver = None
    synthetic = None
    static_pose_cam = None
    if args.pose_source == "tracker":
        receiver = UdpPoseReceiver()
        tracker_proc = spawn_tracker(args)
        print(f"[tracker] waiting for first pose on udp://{UDP_ADDR[0]}:{UDP_ADDR[1]} "
              f"(load ~40 s + ROI selection)...")
        if not receiver.wait_first(args.tracker_timeout):
            if tracker_proc.poll() is not None:
                raise RuntimeError("tracker process exited before sending a pose")
            raise TimeoutError("no cube pose received from tracker")
        print(f"[tracker] pose stream up (seq={receiver.seq})")
    elif args.pose_source == "npy":
        static_pose_cam = np.load(args.pose_npy).astype(np.float64)
        print(f"[pose] static pose from {args.pose_npy}")
    else:
        synthetic = SyntheticPose()
        print("[pose] synthetic tumbling cube (offline smoke test)")

    mirror_proc = spawn_mirror(args) if args.mirror else None

    # goal visualization: stream the goal orientation (camera view) to the tracker
    goal_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if receiver is not None else None
    T_env_cam = env_T_base @ base_T_cam if base_T_cam is not None else env_T_base

    def send_goal(rot_dist_val: float) -> None:
        if goal_sock is None:
            return
        r_cam_goal = T_env_cam[:3, :3].T @ quat_to_rotmat(goal_quat)
        goal_sock.sendto(struct.pack(GOAL_FMT, time.time(), *r_cam_goal.ravel(), rot_dist_val), GOAL_UDP)

    state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # manual calibration: yaml-persisted, live-updatable from repose_calib_gui.py
    manual = np.zeros(6)  # dx dy dz [m], rx ry rz [rad]
    if os.path.exists(MANUAL_CALIB_YAML):
        import yaml as _yaml

        with open(MANUAL_CALIB_YAML) as f:
            _d = _yaml.safe_load(f) or {}
        manual = np.array([_d.get(k, 0.0) for k in ["dx", "dy", "dz", "rx", "ry", "rz"]])
        print(f"[calib] loaded manual offset {MANUAL_CALIB_YAML}: "
              f"d={np.round(manual[:3] * 1000, 1)}mm r={np.round(np.degrees(manual[3:]), 2)}deg")
    calib_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    calib_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    calib_sock.bind(CALIB_UDP)
    calib_sock.setblocking(False)

    recenter_requested = False

    def poll_manual_calib() -> None:
        nonlocal manual, recenter_requested
        while True:
            try:
                data, _ = calib_sock.recvfrom(64)
            except BlockingIOError:
                return
            if len(data) == struct.calcsize(CALIB_COMMIT_FMT):
                vals = struct.unpack(CALIB_COMMIT_FMT, data)
                cand = np.array(vals[:6])
                vals_flag = vals[6]
                flag = vals_flag > 0.5
            else:
                cand = np.array(struct.unpack(CALIB_FMT, data))
                vals_flag = 0.0
                flag = False
            # reject corrupt packets: the destroyed-trackbar signature (cv2
            # returns -1 -> exactly -51 mm / -15.1 deg on ALL six axes), plus a
            # generous absurdity bound (totals accumulate across saves, so the
            # per-slider range is NOT a valid bound here)
            garbage = (np.allclose(cand[:3], -0.051, atol=1e-9)
                       and np.allclose(cand[3:], np.radians(-15.1), atol=1e-6))
            if garbage or np.any(np.abs(cand[:3]) > 0.15) or np.any(np.abs(cand[3:]) > np.radians(45.0)):
                print(f"[calib][REJECT] {np.round(cand[:3] * 1000, 1)}mm "
                      f"{np.round(np.degrees(cand[3:]), 1)}deg")
                continue
            if (np.any(np.abs(cand[:3] - manual[:3]) > 0.001)
                    or np.any(np.abs(cand[3:] - manual[3:]) > np.radians(0.1))):
                print(f"[calib] manual now d={np.round(cand[:3] * 1000, 1)}mm "
                      f"r={np.round(np.degrees(cand[3:]), 2)}deg")
            manual = cand
            if flag:
                if vals_flag >= 1.5:
                    log_extrinsic_snapshot()
                else:
                    recenter_requested = True

    def log_extrinsic_snapshot() -> None:
        """Print + append the EFFECTIVE extrinsic with the manual calib folded in.

        apply_manual acts in env coords as T_corr = [R_cr | pivot - R_cr@pivot + d],
        so the corrected env_T_cam = T_corr @ env_T_base @ base_T_cam and the
        equivalent camera extrinsic is base_T_cam' = inv(env_T_base) @ env_T_cam'.
        Useful to restore or bake the calibration later.
        """
        cr = (_axis_angle_rotmat(np.array([0, 0, 1.0]), manual[5])
              @ _axis_angle_rotmat(np.array([0, 1.0, 0]), manual[4])
              @ _axis_angle_rotmat(np.array([1.0, 0, 0]), manual[3]))
        pivot = REST_POS.copy()
        pivot[2] += (args.cube_edge - 0.06) / 2.0
        t_corr = np.eye(4)
        t_corr[:3, :3] = cr
        t_corr[:3, 3] = pivot - cr @ pivot + manual[:3]
        env_t_cam = t_corr @ (env_T_base @ base_T_cam if base_T_cam is not None else env_T_base)
        base_t_cam_eff = np.linalg.inv(env_T_base) @ env_t_cam
        quat = rotmat_to_quat(base_t_cam_eff[:3, :3])
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"[extrinsic] snapshot @ {stamp}",
            f"[extrinsic] manual d={np.round(manual[:3] * 1000, 1)}mm r={np.round(np.degrees(manual[3:]), 2)}deg"
            f"  center_offset={np.round(center_offset, 4)}",
            f"[extrinsic] effective base_T_cam (manual folded in){' [NO CALIB YAML: env_T_cam shown]' if base_T_cam is None else ''}:",
        ] + [f"[extrinsic]   {np.array2string(row, precision=6, separator=', ')}" for row in base_t_cam_eff] + [
            f"[extrinsic] as pos(m) + quat(wxyz): {np.round(base_t_cam_eff[:3, 3], 5).tolist()}"
            f" + {np.round(quat, 6).tolist()}",
            "[extrinsic] NOTE: manual translation affects the MIRROR only; policy obs stays rest-anchored",
        ]
        for ln in lines:
            print(ln)
        try:
            with open(str(Path(__file__).resolve().parent / "calib_history.log"), "a") as f:
                f.write("\n".join(lines) + "\n\n")
        except OSError as e:
            print(f"[extrinsic][WARN] history write failed: {e}")

    def apply_manual(pose: np.ndarray) -> np.ndarray:
        """Manual ROTATION calib: acts on the whole pose about the rest anchor.

        An extrinsic rotation error moves positions with a lever arm (worse the
        further the cube is from the pivot) — rotating orientation only cannot
        express that. Pivoting at the rest anchor is equivalent to correcting
        the camera extrinsic rotation up to a constant translation, which
        auto-center absorbs; the rest position stays invariant by construction.
        The manual TRANSLATION is applied by vis_pose() to the mirror stream
        ONLY — it must never shift the policy obs off the sim contract.
        """
        if not np.any(manual[3:]):
            return pose
        out = pose.copy()
        cr = (_axis_angle_rotmat(np.array([0, 0, 1.0]), manual[5])
              @ _axis_angle_rotmat(np.array([0, 1.0, 0]), manual[4])
              @ _axis_angle_rotmat(np.array([1.0, 0, 0]), manual[3]))
        pivot = REST_POS.copy()
        pivot[2] += (args.cube_edge - 0.06) / 2.0
        out[:3, :3] = cr @ pose[:3, :3]
        out[:3, 3] = pivot + cr @ (pose[:3, 3] - pivot)
        return out

    def vis_pose(pose: np.ndarray | None) -> np.ndarray | None:
        """Mirror-only view: manual TRANSLATION on top of the policy pose.

        The real hand's rest geometry differs from the sim's by a few cm; the
        translation sliders reconcile the MIRROR with reality, but the policy
        obs must stay anchored to the sim contract (rest -> REST_POS), so the
        translation is never applied to what the policy sees.
        """
        if pose is None or not np.any(manual[:3]):
            return pose
        out = pose.copy()
        out[:3, 3] = pose[:3, 3] + manual[:3]
        return out

    def send_state(q12: np.ndarray, pose_env: np.ndarray | None) -> None:
        if pose_env is None:
            return
        state_sock.sendto(
            struct.pack(STATE_FMT, time.time(), *np.asarray(q12, dtype=np.float64),
                        *np.asarray(pose_env, dtype=np.float64).ravel()),
            STATE_UDP,
        )

    center_offset = np.zeros(3)

    def cube_env_pose() -> np.ndarray | None:
        poll_manual_calib()
        if synthetic is not None:
            return apply_manual(synthetic.pose_env())
        pose_cam = receiver.poll() if receiver is not None else static_pose_cam
        if pose_cam is None:
            return None
        pose_base = pose_cam if base_T_cam is None else base_T_cam @ pose_cam
        pose = env_T_base @ pose_base
        pose[:3, 3] -= center_offset
        return apply_manual(pose)

    # --- auto-center: cube at rest in the palm defines the position zero ----
    # ALWAYS on: it anchors the rest position to what the policy expects
    # (REST_POS), robust to camera bumps. Consequence: the TRANSLATION part of
    # the manual calibration is neutralized at runtime by construction — use
    # the sliders' translation only for temporary visual exploration; the
    # ROTATION part is the persistent, meaningful correction.
    def run_auto_center(strict: bool = False) -> None:
        nonlocal center_offset
        samples = []
        t0 = time.time()
        while len(samples) < 30 and time.time() - t0 < 5.0:
            p = cube_env_pose()
            if p is not None:
                samples.append(p[:3, 3].copy())
                send_state(hand_q, vis_pose(p))
            time.sleep(0.05)
        if not samples:
            print("[center][WARN] no poses received — auto-center skipped")
            return
        arr = np.asarray(samples)
        spread = float(arr.std(axis=0).max()) if len(arr) > 1 else 0.0
        rest = REST_POS.copy()
        rest[2] += (args.cube_edge - 0.06) / 2.0  # bigger cube rests higher in the palm
        if strict:
            # mid-session re-anchor: refuse rather than jump the world
            dev = float(np.linalg.norm(arr.mean(axis=0) - rest))
            if spread > 0.01 or dev > 0.05:
                print(f"[center] re-anchor REFUSED (std {spread * 1000:.0f} mm, off-rest {dev * 100:.1f} cm)"
                      " — put the cube at rest in the palm first")
                return
        elif spread > 0.01:
            print(f"[center][WARN] cube not still during zeroing (std {spread * 1000:.0f} mm) — offset may be poor")
        # samples already include the CURRENT center_offset -> accumulate
        center_offset = center_offset + (arr.mean(axis=0) - rest)
        mag = float(np.linalg.norm(center_offset))
        print(f"[center] position offset zeroed: {np.array2string(center_offset, precision=3)} (|{mag * 100:.1f} cm|)")
        if mag > 0.15:
            print("[center][WARN] offset > 15 cm — the camera moved a lot since calibration;"
                  " its ROTATION is probably also off. Consider re-calibrating anyway.")

    if synthetic is None and args.auto_center:
        run_auto_center()

    if hw is not None and args.execute:
        input("[real] ENTER to start streaming policy commands, or Ctrl-C to abort...")

    # after any blocking prompt the UDP buffer holds only stale packets; resync
    if receiver is not None:
        if not receiver.wait_fresh(args.max_pose_age, 5.0):
            raise RuntimeError("pose stream not fresh at start — is the tracker still running?")
    send_goal(-1.0)

    # --- control loop ------------------------------------------------------
    prev_action = np.zeros(12, dtype=np.float32)
    prev_targets = hand_q.copy()
    prev_obj_quat = None
    prev_pose_env = None
    prev_pose_t = 0.0
    work_times: list[float] = []
    serial_times: list[float] = []
    overruns = 0
    log = ({"obs": [], "action": [], "obj_pos": [], "obj_quat": [], "goal_quat": [], "t": [],
            "hand_q": [], "targets": []} if args.log_npz else None)

    def soft_reset():
        """Mimic the sim episode reset: ramp the hand open (cube settles back into
        the palm), clear the LSTM state, sample a fresh goal."""
        nonlocal prev_targets, prev_action, hand_q, goal_quat
        for _ in range(40):  # 2 s ramp to the open home pose + settle
            tg = prev_targets + np.clip(-prev_targets, -args.max_hand_step, args.max_hand_step)
            if hw is not None and args.execute:
                hw.hand_stream(tg.astype(np.float32))
            prev_targets = tg
            hand_q = tg
            time.sleep(STEP_DT)
        policy.reset()
        prev_action = np.zeros(12, dtype=np.float32)
        goal_quat = sample_goal_quat_mode(rng, args.goal_mode)
        send_goal(-1.0)
        if receiver is not None:
            receiver.wait_fresh(args.max_pose_age, 3.0)

    print(f"[run] 20 Hz control, success tol {args.success_tol:.2f} rad, goal #1:"
          f" quat {np.array2string(goal_quat, precision=3)}")
    next_t = time.perf_counter()
    stop_reason = "steps done"
    last_event_t = time.perf_counter()
    try:
        for step in range(args.steps):
            cycle_start = time.perf_counter()
            pose_env = cube_env_pose()
            fresh = pose_env is not None and (receiver is None or receiver.age() <= args.max_pose_age)
            if recenter_requested:
                recenter_requested = False
                if hw is not None and args.execute:
                    print("[center] re-center request ignored while EXECUTING (cube must be at rest)")
                else:
                    print("[center] explicit re-anchor requested...")
                    run_auto_center(strict=True)
                    last_event_t = time.perf_counter()
                    next_t = time.perf_counter()
            if receiver is not None and receiver.age() > args.abort_pose_age:
                if hw is not None and args.execute:
                    stop_reason = f"pose stale {receiver.age():.2f}s"
                    break
                if step % 100 == 0:
                    print(f"[dry] pose stale {receiver.age():.1f}s — tracking lost? (no stop in dry run)")

            if fresh:
                # latency compensation: lead the measured pose by the camera->action delay
                now_pose_t = time.perf_counter()
                pose_for_obs = pose_env
                if args.pose_lead > 0 and prev_pose_env is not None:
                    pose_for_obs = extrapolate_pose(
                        prev_pose_env, pose_env, now_pose_t - prev_pose_t, args.pose_lead
                    )
                prev_pose_env = pose_env.copy()
                prev_pose_t = now_pose_t

                send_state(hand_q, vis_pose(pose_env))
                obj_pos = pose_for_obs[:3, 3]
                obj_quat = rotmat_to_quat(pose_for_obs[:3, :3])
                # keep the quaternion sign continuous across frames (PhysX streams
                # are continuous in sim; matrix->quat conversion is not)
                if prev_obj_quat is not None and float(np.dot(obj_quat, prev_obj_quat)) < 0.0:
                    obj_quat = -obj_quat
                prev_obj_quat = obj_quat

                # fall check — hard stop only while actually driving the hand;
                # in dry/calib runs the cube is routinely carried around by hand
                fall_d = float(np.linalg.norm(obj_pos - IN_HAND_POS))
                if fall_d >= FALL_DIST:
                    if hw is not None and args.execute:
                        stop_reason = f"cube fell (dist {fall_d:.3f} m)"
                        break
                    if step % 100 == 0:
                        print(f"[dry] cube {fall_d:.2f} m from palm — calibration move? (no stop)")

                kin.update(arm_q, hand_q)
                tip_env = (env_T_base[:3, :3] @ kin.fingertip_pos_base().T).T + env_T_base[:3, 3]
                obs = build_obs(tip_env, obj_pos, obj_quat, goal_quat, prev_action)
                action = policy.act(obs)

                # sim action contract: absolute targets + moving average + saturation
                targets = ACT_MOVING_AVERAGE * scale_action(action) + (1.0 - ACT_MOVING_AVERAGE) * prev_targets
                targets = np.clip(targets, LOWER, UPPER)
                targets = prev_targets + np.clip(targets - prev_targets, -args.max_hand_step, args.max_hand_step)
                targets = np.clip(targets, LOWER, UPPER).astype(np.float32)

                if hw is not None and args.execute:
                    t_ser = time.perf_counter()
                    q_meas = hw.hand_stream(targets, read=not args.no_hand_read)
                    serial_times.append(time.perf_counter() - t_ser)
                    # closed-loop obs: the policy sees where the fingers ARE (a
                    # finger blocked by the cube no longer lies in the obs)
                    hand_q = q_meas if q_meas is not None else targets
                else:
                    hand_q = targets
                prev_targets = targets
                prev_action = action

                rot_dist = rotation_distance(obj_quat, goal_quat)
                send_goal(rot_dist)
                if rot_dist <= args.success_tol:
                    successes += 1
                    goal_quat = sample_goal_quat_mode(rng, args.goal_mode)
                    last_event_t = time.perf_counter()
                    print(f"[goal] SUCCESS #{successes} at step {step}!"
                          f" next goal quat {np.array2string(goal_quat, precision=3)}")
                elif args.stall_timeout > 0 and time.perf_counter() - last_event_t > args.stall_timeout:
                    print(f"[stall] no success for {args.stall_timeout:.0f}s at step {step} — "
                          "reopening hand, resetting policy, new goal")
                    soft_reset()
                    last_event_t = time.perf_counter()
                    next_t = time.perf_counter()
                    continue

                if log is not None:
                    log["hand_q"].append(hand_q.copy())
                    log["targets"].append(targets.copy())
                    log["obs"].append(obs)
                    log["action"].append(action)
                    log["obj_pos"].append(obj_pos.copy())
                    log["obj_quat"].append(obj_quat.copy())
                    log["goal_quat"].append(goal_quat.copy())
                    log["t"].append(time.time())

                if step % max(1, args.print_every) == 0:
                    age = 0.0 if receiver is None else receiver.age()
                    print(f"[run] step {step:4d} rot_dist {math.degrees(rot_dist):6.1f} deg "
                          f"obj {np.array2string(obj_pos, precision=3)} "
                          f"pose_age {age:.2f}s succ {successes}")
            # not fresh: hold targets, skip policy this cycle

            work_times.append(time.perf_counter() - cycle_start)
            next_t += STEP_DT
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                # hybrid pacing: coarse sleep, then spin the last ~2 ms
                # (plain time.sleep oversleeps ~2 ms -> loop ran at 19.3 Hz, not 20)
                if sleep_t > 0.002:
                    time.sleep(sleep_t - 0.002)
                while time.perf_counter() < next_t:
                    pass
            else:
                overruns += 1
                next_t = time.perf_counter()
    finally:
        print(f"[run] stopped: {stop_reason}; consecutive successes: {successes}")
        if len(work_times) > 10:
            wt = np.array(work_times[1:])
            print(f"[timing] work/cycle p50 {np.percentile(wt, 50) * 1000:.1f}ms "
                  f"p95 {np.percentile(wt, 95) * 1000:.1f}ms  overruns {overruns} "
                  f"({overruns / max(1, len(wt)) * 100:.1f}%)")
        if serial_times:
            st = np.array(serial_times)
            print(f"[timing] hand serial p50 {np.percentile(st, 50) * 1000:.1f}ms "
                  f"p95 {np.percentile(st, 95) * 1000:.1f}ms "
                  f"({'read-back ON' if not args.no_hand_read else 'fire-and-forget'})")
        if hw is not None:
            hw.close()
        if log is not None and log["obs"]:
            np.savez(args.log_npz, **{k: np.array(v) for k, v in log.items()})
            print(f"[log] saved {args.log_npz}")
        if args.mirror and mirror_proc is not None and mirror_proc.poll() is None:
            os.killpg(os.getpgid(mirror_proc.pid), signal.SIGINT)
        if tracker_proc is not None and tracker_proc.poll() is None:
            os.killpg(os.getpgid(tracker_proc.pid), signal.SIGINT)
            try:
                tracker_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(tracker_proc.pid), signal.SIGTERM)


if __name__ == "__main__":
    main()
