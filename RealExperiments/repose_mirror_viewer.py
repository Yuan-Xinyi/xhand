#!/usr/bin/env python3
"""Isaac mirror viewer: render what the deployment pipeline BELIEVES.

Subscribes to the control loop's state stream (udp 9879: measured hand joints +
estimated cube pose in the env frame) and mirrors it kinematically in the
repose sim scene. Put this window next to the camera view: if the rendered
cube-in-hand relationship does not match reality, the calibration / frame math
is off (axis swaps show as mirrored motion, translation offsets as a floating
or sunken cube).

Usually spawned by foundationpose_repose_real.py --mirror; standalone:
    conda activate env_isaaclab
    python RealExperiments/repose_mirror_viewer.py --cube-edge 0.062
"""
import argparse
import socket
import struct
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Xhand-Repose-Cube-OpenAI-LSTM-Hard-Direct-v0")
parser.add_argument("--port", type=int, default=9879)
parser.add_argument("--cube-edge", type=float, default=0.06)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False  # this viewer is pointless without a window

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import xhand_inhand.tasks  # noqa: F401, E402

STATE_FMT = "<29d"


def _rotmat_to_quat(m):
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.observation_noise_model = None
    env_cfg.action_noise_model = None
    env_cfg.events = None  # fixed nominal scene; we mirror states, not simulate
    env_cfg.ui_window_class_type = None
    scale = args_cli.cube_edge / 0.06
    env_cfg.object_cfg.spawn.scale = (scale, scale, scale)
    # kinematic cube: physics must not fight the mirrored pose
    env_cfg.object_cfg.spawn.rigid_props.kinematic_enabled = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    origins = u.scene.env_origins[0]
    u.sim.set_camera_view(eye=[0.4, 0.5, 0.85], target=[0.0, 0.1, 0.55])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args_cli.port))
    sock.setblocking(False)

    print(f"[mirror] listening on udp://127.0.0.1:{args_cli.port} — waiting for the control loop", flush=True)
    last_stamp, shown = 0.0, 0
    q12, pose = None, None
    while simulation_app.is_running():
        while True:
            try:
                data, _ = sock.recvfrom(512)
            except BlockingIOError:
                break
            vals = struct.unpack(STATE_FMT, data)
            last_stamp = vals[0]
            q12 = np.array(vals[1:13])
            pose = np.array(vals[13:29]).reshape(4, 4)

        if q12 is not None:
            qt = torch.tensor(q12, dtype=torch.float32, device=u.device).unsqueeze(0)
            u.hand.write_joint_state_to_sim(qt, torch.zeros_like(qt))
            u.hand.set_joint_position_target(qt)
            root = torch.zeros((1, 7), device=u.device)
            root[0, :3] = torch.tensor(pose[:3, 3], dtype=torch.float32, device=u.device) + origins
            root[0, 3:7] = torch.tensor(_rotmat_to_quat(pose[:3, :3]), dtype=torch.float32, device=u.device)
            u.object.write_root_pose_to_sim(root)

        u.sim.step(render=True)
        shown += 1
        if shown % 200 == 0 and q12 is not None:
            print(f"[mirror] state age {time.time() - last_stamp:.2f}s", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
