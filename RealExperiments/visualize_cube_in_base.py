#!/usr/bin/env python3
"""Visualize the FoundationPose-detected cube in the xArm base frame, using `one`.

Loads camera_T_cube (saved by foundationpose_then_real / verify_cube_pose), maps it
into the xArm base frame with the D435 extrinsic (T_base_cam), and renders in the
`one` viewer:

    * the base coordinate frame (world origin)
    * the xArm7 + XHand at the RL home configuration
    * the detected cube as a 6 cm box at base_T_cube, with its own frame drawn

If the cube box overlaps the robot/table sensibly and sits where you expect
relative to the arm, the extrinsic is good.

Run in the `one` env (needs OpenGL):

    python visualize_cube_in_base.py
    python visualize_cube_in_base.py --pose_npy /tmp/foundationpose_cube_pose.npy

Headless validation (no window): ONE_HEADLESS=1 python visualize_cube_in_base.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from foundationpose_then_real import (
    DEFAULT_CALIB_YAML,
    DEFAULT_Q_ISAAC,
    ISAAC_TO_ONE,
    MOUNT_RPY,
    _patch_one_mechbase_compat,
    _rotmat_from_z,
    _tf_from_pos_rotmat,
    load_base_T_cam,
)

CUBE_SIZE = 0.06  # 6 cm cube, matches Pick-Cube-Direct-v0


def _fmt_xyz(v: np.ndarray) -> str:
    return f"{v[0]:+.4f} {v[1]:+.4f} {v[2]:+.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose_npy", default="/tmp/foundationpose_cube_pose.npy",
                        help="camera_T_cube saved by FoundationPose.")
    parser.add_argument("--calib_yaml", default=DEFAULT_CALIB_YAML)
    parser.add_argument("--cube_size", type=float, default=CUBE_SIZE)
    parser.add_argument("--table_z", type=float, default=0.0,
                        help="Table top surface height in the base frame (m). Cube rests on it.")
    parser.add_argument("--table_size", type=float, nargs=2, metavar=("X", "Y"), default=(1.2, 1.2),
                        help="Table top extent in x,y (m).")
    parser.add_argument("--table_thickness", type=float, default=0.02)
    parser.add_argument("--no_table", action="store_true", help="Skip rendering the table.")
    parser.add_argument("--no_robot", action="store_true", help="Skip rendering the arm+hand.")
    parser.add_argument("--no_calib", action="store_true",
                        help="Treat the pose npy as already in the base frame.")
    parser.add_argument("--print_only", action="store_true",
                        help="Print pose/robot diagnostics and exit without opening the viewer.")
    return parser.parse_args()


def _ensure_one_on_path() -> None:
    one_root = "/home/lqin/one"
    if one_root not in sys.path:
        sys.path.insert(0, one_root)


def _link_pos(links, name: str) -> np.ndarray | None:
    for link in links:
        if getattr(link, "name", None) == name:
            return np.asarray(link.tf[:3, 3], dtype=np.float32)
    return None


def _build_robot(q_isaac: np.ndarray):
    """Build xArm7 + mounted XHandRight at q_isaac."""
    _patch_one_mechbase_compat()
    from one.robots.end_effectors.xhand.xhand_right import XHandRight
    from one.robots.manipulators.xarm.xarm7.xarm7 import XArm7

    arm = XArm7()
    hand = XHandRight()
    mount_tf = _tf_from_pos_rotmat(np.zeros(3, dtype=np.float32), _rotmat_from_z(MOUNT_RPY))
    arm.mount(hand, arm.runtime_lnks[-1], mount_tf, update=True)

    q_one = q_isaac[ISAAC_TO_ONE]
    arm.fk(q_one[:7])
    for mounting in arm._mountings.values():
        arm._update_mounting(mounting)
    hand.fk(q_one[7:])

    diag: dict[str, np.ndarray] = {}
    for name in ("link7",):
        pos = _link_pos(arm.runtime_lnks, name)
        if pos is not None:
            diag[name] = pos
    for name in ("palm", "index_rota_link2", "mid_link2", "ring_link2", "pinky_link2", "thumb_rota_link2"):
        pos = _link_pos(hand.runtime_lnks, name)
        if pos is not None:
            diag[name] = pos
    return arm, hand, diag


def _add_robot(scene, q_isaac: np.ndarray) -> dict[str, np.ndarray]:
    """Build xArm7 + mounted XHandRight at q_isaac and attach to the scene."""
    arm, hand, diag = _build_robot(q_isaac)

    arm.attach_to(scene)
    hand.attach_to(scene)
    return diag


def _print_pose_diagnostics(cam_t_cube: np.ndarray, base_T_cam: np.ndarray | None, base_t_cube: np.ndarray) -> None:
    c = cam_t_cube[:3, 3]
    b = base_t_cube[:3, 3]
    print(f"[diag] camera_T_cube xyz(m): {_fmt_xyz(c)}")
    if base_T_cam is None:
        return

    cam_pos = base_T_cam[:3, 3]
    cam_to_cube = b - cam_pos
    print(f"[diag] cam->cube in base(m): {_fmt_xyz(cam_to_cube)}")
    print("[diag] D435 optical axes expressed in base:")
    print(f"       +x image-right : {_fmt_xyz(base_T_cam[:3, 0])}")
    print(f"       +y image-down  : {_fmt_xyz(base_T_cam[:3, 1])}")
    print(f"       +z optical-fwd : {_fmt_xyz(base_T_cam[:3, 2])}")
    print(f"[diag] cam->cube dot optical +z(m): {float(cam_to_cube @ base_T_cam[:3, 2]):+.4f}")


def _print_robot_diagnostics(robot_diag: dict[str, np.ndarray], cube_pos: np.ndarray) -> None:
    for name, pos in robot_diag.items():
        print(f"[diag] robot {name} xyz(m): {_fmt_xyz(pos)}")
    if "palm" in robot_diag:
        print(f"[diag] cube - palm xyz(m): {_fmt_xyz(cube_pos - robot_diag['palm'])}")


def main() -> None:
    args = parse_args()
    _ensure_one_on_path()

    pose_path = Path(args.pose_npy)
    if not pose_path.exists():
        raise FileNotFoundError(
            f"pose npy not found: {pose_path}\n"
            "Run foundationpose_then_real.py or verify_cube_pose.py first to detect the cube."
        )
    cam_t_cube = np.load(pose_path).astype(np.float32)

    base_T_cam = None
    if args.no_calib:
        base_t_cube = cam_t_cube
    else:
        base_T_cam = load_base_T_cam(args.calib_yaml)
        base_t_cube = (base_T_cam @ cam_t_cube).astype(np.float32)

    t = base_t_cube[:3, 3]
    print(f"[viz] base_T_cube xyz(m): {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}")
    _print_pose_diagnostics(cam_t_cube, base_T_cam, base_t_cube)

    if args.print_only:
        if not args.no_robot:
            os.environ.setdefault("PYGLET_HEADLESS", "1")
            try:
                _, _, robot_diag = _build_robot(DEFAULT_Q_ISAAC.copy())
                _print_robot_diagnostics(robot_diag, t)
            except Exception as exc:
                print(f"[diag] warning: could not compute robot diagnostics ({exc}).")
        return

    import one.scene.scene_object_primitive as ossop
    import one.utils.constant as ouc
    import one.viewer.world as ovw

    # View from BEHIND the robot base looking forward along +X (the natural operator
    # vantage). Viewing from the cube/camera side instead mirrors left/right and makes
    # a correct detection look "flipped".
    base = ovw.World(
        cam_pos=(-0.9, 0.0, 0.8),
        cam_lookat_pos=(float(t[0]) * 0.6, float(t[1]) * 0.6, 0.05),
        toggle_auto_cam_orbit=False,
    )

    # base frame (world origin) — large axes
    ossop.frame_from_tf(np.eye(4, dtype=np.float32), length_scale=2.0).attach_to(base.scene)

    # table: a thin slab whose TOP surface sits at table_z (where the cube rests),
    # extending downward by table_thickness, centered around the robot+cube workspace.
    if not args.no_table:
        tx, ty = args.table_size
        table = ossop.box(
            pos=(float(t[0]) * 0.5, float(t[1]) * 0.5, args.table_z - 0.5 * args.table_thickness),
            xyz_lengths=(tx, ty, args.table_thickness),
            rgb=(0.62, 0.52, 0.40),
            alpha=1.0,
            collision_type=None,
            is_floating=False,
        )
        table.attach_to(base.scene)

    # the detected cube + its own (smaller) frame
    cube = ossop.box(
        pos=base_t_cube[:3, 3],
        xyz_lengths=(args.cube_size, args.cube_size, args.cube_size),
        rotmat=base_t_cube[:3, :3],
        rgb=(0.30, 0.55, 0.85),
        alpha=0.85,
        collision_type=ouc.CollisionType.AABB,
        is_floating=True,
    )
    cube.attach_to(base.scene)
    ossop.frame_from_tf(base_t_cube, length_scale=1.0).attach_to(base.scene)

    # The D435 itself, drawn at T_base_cam. Its frame is the OpenCV optical convention
    # (+x right, +y down, +z INTO the scene), so the blue (+z) axis should point from the
    # camera toward the cube. A red sight line connects the camera to the cube. If the
    # camera ends up on the wrong side or +z points away from the cube, the extrinsic
    # axes are flipped — that is what makes detection look "reversed".
    if not args.no_calib:
        cam_pos = base_T_cam[:3, 3]
        ossop.frame_from_tf(base_T_cam, length_scale=1.5).attach_to(base.scene)
        sight = np.stack([cam_pos, base_t_cube[:3, 3]]).reshape(1, 2, 3).astype(np.float32)
        ossop.linsegs(sight, radius=0.004, srgbs=np.array([1.0, 0.0, 0.0], np.float32)).attach_to(base.scene)
        print(f"[viz] D435 in base xyz(m): {cam_pos[0]:+.4f} {cam_pos[1]:+.4f} {cam_pos[2]:+.4f}")

    if not args.no_robot:
        try:
            robot_diag = _add_robot(base.scene, DEFAULT_Q_ISAAC.copy())
            _print_robot_diagnostics(robot_diag, t)
        except Exception as exc:  # robot rendering is best-effort; cube still shows
            print(f"[viz] warning: could not render robot ({exc}); showing cube only.")

    print("[viz] launching one viewer — close the window to exit.")
    base.run()


if __name__ == "__main__":
    main()
