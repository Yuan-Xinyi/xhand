# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Animate the CrossDex eigengrasp-token -> xhand retargeting INSIDE Isaac Sim.

Drives the real XHand mesh through a smooth eigengrasp trajectory (sweep PC1, PC2,
PC3, then a random walk through grasps). Each frame the eigengrasp coords are decoded
+ retargeted (the same NN used by Pick-Cube-Token-Direct-v0) and written to the hand
joints, so you watch the actual hand articulate through the token space.

Usage (conda activate env_isaaclab):

    # live GUI window (uses $DISPLAY)
    python scripts/visualize_token_retarget.py

    # headless, record an mp4 you can open anywhere
    python scripts/visualize_token_retarget.py --record --out /tmp/token_retarget.mp4
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Animate eigengrasp-token -> xhand retargeting in Isaac Sim.")
parser.add_argument("--record", action="store_true", help="Headless render to an mp4 instead of opening a GUI.")
parser.add_argument("--out", type=str, default="/disk2/xhand_inhand/xhand_inhand/tools/crossdex_retarget/viz/token_retarget.mp4")
parser.add_argument("--seconds", type=float, default=20.0, help="Animation length (per loop).")
parser.add_argument("--loops", type=int, default=1, help="How many times to repeat the trajectory (GUI mode).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# recording => headless + offscreen camera rendering
if args_cli.record:
    args_cli.headless = True
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------- app up
import pickle

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import XHAND_RIGHT_CFG

import sys
_TOK = "/disk2/xhand_inhand/xhand_inhand/source/xhand_inhand/xhand_inhand/tasks/direct/pick_cube_token"
sys.path.insert(0, _TOK)
from retarget_infer import EigenRetarget  # noqa: E402

_MODELS = "/disk2/xhand_inhand/xhand_inhand/tools/crossdex_retarget/models"


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3500.0, color=(0.95, 0.95, 0.95))
    )
    robot = XHAND_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # camera only attached in --record mode (offscreen rendering needs --enable_cameras)
    camera = None
    if args_cli.record:
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 10.0)),
        )


def _build_trajectory(R, fps, seconds):
    """Yield a list of (N=1, 9) eigengrasp-coord tensors for the whole animation."""
    vmin = R.min_values.cpu().numpy()
    vmax = R.max_values.cpu().numpy()
    dev = R.device
    n_total = int(fps * seconds)
    # 4 phases: PC1 sweep, PC2 sweep, PC3 sweep, random walk
    n_phase = n_total // 4
    coords_seq = []

    def tri(a):  # 0->1->0 triangle
        return 1.0 - abs(2.0 * a - 1.0)

    for pc in (0, 1, 2):
        for i in range(n_phase):
            a = tri(i / max(n_phase - 1, 1))
            c = np.zeros(9)
            c[pc] = vmin[pc] + a * (vmax[pc] - vmin[pc])
            coords_seq.append(c)
    # smooth random walk through grasps
    rng = np.random.default_rng(0)
    n_rest = n_total - len(coords_seq)
    knots = max(n_rest // 25, 2)
    pts = rng.uniform(vmin, vmax, size=(knots, 9))
    pts[0] = 0.0
    for k in range(knots - 1):
        for i in range(25):
            a = i / 25.0
            coords_seq.append(pts[k] * (1 - a) + pts[k + 1] * a)
            if len(coords_seq) >= n_total:
                break
        if len(coords_seq) >= n_total:
            break
    return [torch.as_tensor(c, dtype=torch.float32, device=dev).unsqueeze(0) for c in coords_seq]


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=1.0))
    sim.reset()
    robot = scene["robot"]
    camera = scene["camera"] if args_cli.record else None
    if camera is not None:
        camera.set_world_poses_from_view(
            eyes=torch.tensor([[0.28, 0.28, 0.22]], device=sim.device),
            targets=torch.tensor([[0.0, 0.0, 0.09]], device=sim.device),
        )
    # GUI viewport camera
    sim.set_camera_view(eye=[0.28, 0.28, 0.22], target=[0.0, 0.0, 0.09])

    # retargeting model + permutation NN-order -> robot joint order
    R = EigenRetarget(f"{_MODELS}/retarget_nn_xhand.pt", f"{_MODELS}/retarget_nn_xhand_meta.pkl", device=str(sim.device))
    perm = R.permutation_to(list(robot.joint_names))  # robot.joint_names order
    lower = robot.data.soft_joint_pos_limits[..., 0]
    upper = robot.data.soft_joint_pos_limits[..., 1]
    print(f"[INFO] robot joints = {robot.joint_names}")

    fps = 30
    sub = max(int(round((1.0 / sim.get_physics_dt()) / fps)), 1)  # physics steps per rendered frame
    traj = _build_trajectory(R, fps, args_cli.seconds)
    print(f"[INFO] trajectory frames = {len(traj)}, physics steps/frame = {sub}")

    frames = []
    writer = None
    if args_cli.record:
        import imageio
        os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
        writer = imageio.get_writer(args_cli.out, fps=fps, codec="libx264", quality=8)

    sim_dt = sim.get_physics_dt()
    loops = 1 if args_cli.record else max(args_cli.loops, 1)
    for _ in range(loops):
        for coords in traj:
            if not simulation_app.is_running():
                break
            hand_nn = R.retarget(coords)              # (1,12) NN order
            target = hand_nn[:, perm]                 # -> robot joint order
            target = torch.clamp(target, lower, upper)
            # exact kinematic playback: write state AND target
            robot.write_joint_state_to_sim(target, torch.zeros_like(target))
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            for _s in range(sub):
                sim.step()
            scene.update(sim_dt)
            if args_cli.record:
                camera.update(sim_dt)
                rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
                writer.append_data(rgb)

    if writer is not None:
        writer.close()
        print(f"[INFO] saved video: {args_cli.out}  ({len(traj)} frames @ {fps}fps)")

    simulation_app.close()


if __name__ == "__main__":
    main()
