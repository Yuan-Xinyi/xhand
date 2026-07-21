# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author the claw-hammer + guided-nail-fixture USD assets for the functional-grasping task.

Per ``claw_hammer_experiment_plan.md``:

  * ``hammer.usd``       -- a single rigid body with 4 collision shapes: handle (cylinder),
                            head block (box), hammer FACE (cylinder, +X), and two claw
                            prongs (boxes, -X) whose gap forms the nail slot.
  * ``nail_fixture.usd`` -- a fixed-base articulation: a guide block bolted to the world
                            and a nail (shaft + head) on a vertical prismatic joint with
                            joint friction = the "controlled resistance" of the plan's
                            guided-pin fixture (NOT a real nail in wood).

HAMMER LOCAL FRAME (mirrored by ``HammerNailTokenEnvCfg`` -- keep in sync!):
  * origin        = head center
  * +X            = hammer-face side; face plane at x = +0.047, face normal = (1, 0, 0)
  * -X            = claw side; nail slot center at (-0.045, 0, 0), slot open toward -X
  * -Z            = handle direction; grip center at (0, 0, -0.10)
  * claw pull dir = (0, 0, -1): local -Z must point UP (handle up) for the prong plane to
                    sit horizontal under the nail head.

NAIL FIXTURE LOCAL FRAME:
  * origin at the block bottom center (place it on the table top).
  * block top at z = +0.10; the nail body origin sits AT the head underside, i.e. exactly
    on the block top when fully inserted (q = 0). Prismatic range q in [0, 0.05] m, +up.

Run (must be inside the ``env_isaaclab`` conda env):

    python gen_hammer_nail_assets.py --headless
"""

import argparse
import os

# NOTE: no top-level AppLauncher / pxr imports -- this module sits inside the task package
# and `xhand_inhand.tasks` auto-imports every submodule at discovery time (see the
# blacklist in tasks/__init__.py). Everything with side effects lives in main().

_DEFAULT_OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "hammer_nail")
)

# ---------------------------------------------------------------- hammer geometry (m)
HEAD_BLOCK_CENTER = (0.010, 0.0, 0.0)
HEAD_BLOCK_DIMS = (0.050, 0.026, 0.026)
FACE_RADIUS = 0.016
FACE_LEN = 0.012
FACE_CENTER = (0.041, 0.0, 0.0)          # face plane at x = +0.047
PRONG_DIMS = (0.045, 0.008, 0.010)       # prongs span x in [-0.060, -0.015]
PRONG_CENTERS = ((-0.0375, 0.0105, 0.0), (-0.0375, -0.0105, 0.0))  # slot gap = 13 mm
HANDLE_RADIUS = 0.012
HANDLE_LEN = 0.160
HANDLE_CENTER = (0.0, 0.0, -0.095)       # handle spans z in [-0.175, -0.015]
HEAD_DENSITY = 3500.0                    # "not too heavy" head -> total ~0.26 kg
HANDLE_DENSITY = 600.0

# ---------------------------------------------------------------- nail fixture geometry (m)
# block height sized so the CLAW can reach the nail head under the constrained arm: with
# the task home pose (palm center ~0.265 high, palm down) and the hammer rotated handle-up
# in hand, the claw slot sweeps ~0.165 +- a few cm -> nail head base at 0.14 + protrusion.
BLOCK_DIMS = (0.08, 0.08, 0.14)
BLOCK_TOP = 0.14
NAIL_SHAFT_RADIUS = 0.004                # 8 mm shaft fits the 13 mm claw slot
NAIL_SHAFT_LEN = 0.095                   # below the nail origin; must stay INSIDE the block at
                                         # q = 0 (block is 0.10 tall) or the shaft tip pokes into
                                         # the table and depenetration ejects the nail upward
NAIL_HEAD_RADIUS = 0.011                 # 22 mm head cannot pass through the slot
NAIL_HEAD_THICKNESS = 0.008
NAIL_TRAVEL = 0.06                       # prismatic range [0, 0.06]; the task starts the nail at
                                         # ~0.015-0.02 (protruding, so the claw fits UNDER the
                                         # head) and pulls +0.03 from there
NAIL_DENSITY = 7800.0                    # steel -> nail ~0.07 kg
NAIL_JOINT_FRICTION = 2.0                # controlled resistance (N); overridable via actuator cfg
NAIL_JOINT_DAMPING = 10.0


def _add_box(stage, path, center, dims, density, color):
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*dims))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateDensityAttr(density)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cube


def _add_cylinder(stage, path, center, axis, radius, height, density, color):
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    half = height / 2.0
    lo = [-radius, -radius, -radius]
    hi = [radius, radius, radius]
    idx = {"X": 0, "Y": 1, "Z": 2}[axis]
    lo[idx], hi[idx] = -half, half
    cyl.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(density)
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cyl


def _new_stage(path):
    from pxr import Usd, UsdGeom

    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def build_hammer(path):
    from pxr import UsdGeom, UsdPhysics

    stage = _new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/hammer")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())

    steel = (0.45, 0.45, 0.48)
    wood = (0.55, 0.36, 0.18)
    _add_box(stage, "/hammer/head_block", HEAD_BLOCK_CENTER, HEAD_BLOCK_DIMS, HEAD_DENSITY, steel)
    _add_cylinder(stage, "/hammer/face", FACE_CENTER, "X", FACE_RADIUS, FACE_LEN, HEAD_DENSITY, steel)
    for i, c in enumerate(PRONG_CENTERS):
        _add_box(stage, f"/hammer/claw_prong_{i}", c, PRONG_DIMS, HEAD_DENSITY, steel)
    _add_cylinder(stage, "/hammer/handle", HANDLE_CENTER, "Z", HANDLE_RADIUS, HANDLE_LEN, HANDLE_DENSITY, wood)

    stage.Save()
    print(f"[gen] wrote {path}")


def build_nail_fixture(path):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = _new_stage(path)
    root = UsdGeom.Xform.Define(stage, "/nail_fixture")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # guide block (fixed to the world)
    base = UsdGeom.Xform.Define(stage, "/nail_fixture/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    _add_box(
        stage, "/nail_fixture/base/block",
        (0.0, 0.0, BLOCK_DIMS[2] / 2.0), BLOCK_DIMS, 1000.0, (0.25, 0.25, 0.28),
    )
    fix = UsdPhysics.FixedJoint.Define(stage, "/nail_fixture/fix_base")
    fix.CreateBody1Rel().SetTargets(["/nail_fixture/base"])

    # nail: body origin AT the head underside == block top when fully inserted (q = 0).
    # The shaft passes "through" the block: block<->nail collisions are implicitly disabled
    # by the prismatic joint between them (jointCollisionEnabled defaults to off).
    nail = UsdGeom.Xform.Define(stage, "/nail_fixture/nail")
    UsdPhysics.RigidBodyAPI.Apply(nail.GetPrim())
    nail_xf = UsdGeom.Xformable(nail.GetPrim())
    nail_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, BLOCK_TOP))
    silver = (0.75, 0.75, 0.78)
    _add_cylinder(
        stage, "/nail_fixture/nail/shaft",
        (0.0, 0.0, -NAIL_SHAFT_LEN / 2.0), "Z", NAIL_SHAFT_RADIUS, NAIL_SHAFT_LEN, NAIL_DENSITY, silver,
    )
    _add_cylinder(
        stage, "/nail_fixture/nail/head",
        (0.0, 0.0, NAIL_HEAD_THICKNESS / 2.0), "Z", NAIL_HEAD_RADIUS, NAIL_HEAD_THICKNESS, NAIL_DENSITY, silver,
    )

    # vertical prismatic joint: q = 0 fully inserted, q = +NAIL_TRAVEL fully pulled
    pj = UsdPhysics.PrismaticJoint.Define(stage, "/nail_fixture/nail_joint")
    pj.CreateAxisAttr("Z")
    pj.CreateBody0Rel().SetTargets(["/nail_fixture/base"])
    pj.CreateBody1Rel().SetTargets(["/nail_fixture/nail"])
    pj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, BLOCK_TOP))
    pj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    pj.CreateLowerLimitAttr(0.0)
    pj.CreateUpperLimitAttr(NAIL_TRAVEL)
    # passive resistance: zero-stiffness drive with damping + PhysX joint friction. The env's
    # ImplicitActuatorCfg re-writes these at startup, so tuning lives in the env cfg.
    drive = UsdPhysics.DriveAPI.Apply(pj.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(NAIL_JOINT_DAMPING)
    PhysxSchema.PhysxJointAPI.Apply(pj.GetPrim()).CreateJointFrictionAttr(NAIL_JOINT_FRICTION)

    stage.Save()
    print(f"[gen] wrote {path}")


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Generate claw-hammer + nail-fixture USD assets.")
    parser.add_argument(
        "--out_dir", type=str, default=_DEFAULT_OUT_DIR,
        help="Output directory for hammer.usd / nail_fixture.usd.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    os.makedirs(args_cli.out_dir, exist_ok=True)
    build_hammer(os.path.join(args_cli.out_dir, "hammer.usd"))
    build_nail_fixture(os.path.join(args_cli.out_dir, "nail_fixture.usd"))

    simulation_app.close()


if __name__ == "__main__":
    main()
