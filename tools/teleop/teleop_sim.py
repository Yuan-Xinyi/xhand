# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop SIM driver — run in the `env_isaaclab` conda env.

Subscribes to the WiLoR perception node's UDP stream of 12 xhand joint targets
and drives the real XHand mesh in Isaac Sim in real time. Hand-only for now
(``XHAND_RIGHT_CFG``): the arm is out of scope for this MVP, so the wrist is
fixed and only the fingers follow the human hand.

Pairs with perception_node.py (env `wilor`). Start the sim first, then the
perception node — UDP is connectionless so order is not critical, and the sim
simply holds its pose until packets arrive.

Usage (conda activate env_isaaclab):
    cd tools/teleop
    python teleop_sim.py                 # live GUI window
    python teleop_sim.py --port 51234
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive xhand in Isaac Sim from WiLoR teleop UDP.")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=51234)
parser.add_argument("--smooth", type=float, default=0.7,
                    help="sim-side EMA on joint targets in [0,1]; higher = snappier")
parser.add_argument("--timeout", type=float, default=2.0,
                    help="seconds without a valid packet before relaxing to the rest pose "
                         "(hold the last grasp through brief tracking dropouts)")
parser.add_argument("--debug", action="store_true",
                    help="print received seq + applied joint summary periodically")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------- app up
import sys
import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import XHAND_RIGHT_CFG

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import HAND_JOINT_NAMES, drain_latest, make_receiver  # noqa: E402


# Fixed-base XHand: anchor the wrist root and kill gravity on the hand so only the
# fingers move (the base must not drift/wobble). Same recipe as the xhand_repose
# task (tasks/direct/xhand_repose/xhand_repose_env_cfg.py).
_XHAND_FIXED_CFG = XHAND_RIGHT_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=XHAND_RIGHT_CFG.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,  # anchor the wrist root (the "stand")
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        rot=(0.7071, -0.7071, 0.0, 0.0),  # palm-up, wrist fixed
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
)


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3500.0, color=(0.95, 0.95, 0.95))
    )
    robot = _XHAND_FIXED_CFG


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=1.0))
    sim.reset()
    robot = scene["robot"]
    sim.set_camera_view(eye=[0.28, 0.28, 0.22], target=[0.0, 0.0, 0.09])

    # permutation: wire order (HAND_JOINT_NAMES) -> this robot's articulation order
    robot_names = list(robot.joint_names)
    missing = [n for n in robot_names if n not in HAND_JOINT_NAMES]
    if missing:
        print(f"[teleop_sim][WARN] robot joints not driven by teleop (held at rest): {missing}")
    wire_idx = {n: i for i, n in enumerate(HAND_JOINT_NAMES)}
    perm = [wire_idx.get(n, None) for n in robot_names]  # None => not teleop-driven

    lower = robot.data.soft_joint_pos_limits[..., 0]
    upper = robot.data.soft_joint_pos_limits[..., 1]
    rest = robot.data.default_joint_pos.clone()          # (1, ndof)
    target = rest.clone()
    print(f"[teleop_sim] robot joints = {robot_names}")

    sock = make_receiver(args_cli.host, args_cli.port)
    print(f"[teleop_sim] listening udp://{args_cli.host}:{args_cli.port}; waiting for perception node ...")

    alpha = float(args_cli.smooth)
    sim_dt = sim.get_physics_dt()
    last_valid_t = None
    got_first = False
    last_seq = -1
    step = 0

    while simulation_app.is_running():
        sample = drain_latest(sock)
        if sample is not None:
            _seq, _t, valid, q = sample
            last_seq = _seq
            if valid:
                q_t = torch.tensor(q, dtype=target.dtype, device=target.device)
                desired = rest.clone()
                for j, wi in enumerate(perm):
                    if wi is not None:
                        desired[0, j] = q_t[wi]
                target = (1 - alpha) * target + alpha * desired
                last_valid_t = time.time()
                if not got_first:
                    got_first = True
                    print("[teleop_sim] first hand sample received — tracking.")

        # relax toward the rest pose if the hand has been lost for a while
        if last_valid_t is not None and (time.time() - last_valid_t) > args_cli.timeout:
            target = (1 - 0.02) * target + 0.02 * rest

        cmd = torch.clamp(target, lower, upper)
        robot.set_joint_position_target(cmd)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        step += 1
        if args_cli.debug and step % 60 == 0:
            fresh = last_valid_t is not None and (time.time() - last_valid_t) < args_cli.timeout
            print(f"[teleop_sim] step={step} last_seq={last_seq} tracking={fresh} "
                  f"cmd[min/mean/max]={cmd.min():.3f}/{cmd.mean():.3f}/{cmd.max():.3f}", flush=True)

    sock.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
