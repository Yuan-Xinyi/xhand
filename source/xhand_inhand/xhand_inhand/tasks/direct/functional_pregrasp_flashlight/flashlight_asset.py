"""Runtime-converted OakInk ``flashlight_1`` asset for the first new-object specialist."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils


_DEFAULT_DATASET_ROOT = Path("/disk2/xhand_datasets/unidexfpm")
DATASET_ROOT = Path(os.environ.get("XHAND_UNIDEXFPM_ROOT", _DEFAULT_DATASET_ROOT)).expanduser()
FLASHLIGHT_OBJ = DATASET_ROOT / "extracted" / "oakink" / "flashlight_1" / "align" / "design.obj"
if not FLASHLIGHT_OBJ.is_file():
    raise FileNotFoundError(
        f"OakInk flashlight mesh is missing: {FLASHLIGHT_OBJ}. "
        "Set XHAND_UNIDEXFPM_ROOT to the extracted UniDexFPM dataset root."
    )

FLASHLIGHT_SCALE = (1.0, 1.0, 1.0)
FLASHLIGHT_MASS = 0.15

# Geometry-derived side-rest pose, certified on the converted PhysX hull by the deterministic
# 8-attitude drop-settle probe recorded in the functional-pregrasp manifest.
FLASHLIGHT_REST_QUAT = (0.68760010, -0.66562844, 0.20175872, 0.20841856)
FLASHLIGHT_REST_HEIGHT_ABOVE_TABLE = 0.01769763
FLASHLIGHT_REST_Z = -0.003 + FLASHLIGHT_REST_HEIGHT_ABOVE_TABLE

FLASHLIGHT_USD = sim_utils.MeshConverter(
    sim_utils.MeshConverterCfg(
        asset_path=str(FLASHLIGHT_OBJ),
        usd_dir="/tmp/xhand_inhand/functional_pregrasp/flashlight_1",
        usd_file_name="flashlight_1.usd",
        mass_props=sim_utils.MassPropertiesCfg(mass=FLASHLIGHT_MASS),
        # Author these properties on the converted rigid root.  Applying them only through
        # UsdFileCfg may target an instanceable geometry child and leave the actual body at
        # PhysX defaults (the same constraint as mass_props in the parent task).
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.05,
            angular_damping=0.5,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            # Contact reporting forces this to zero at spawn time.  Keep the authored value
            # conventional; explicit angular damping is the actual rolling-loss model.
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=sim_utils.ConvexDecompositionPropertiesCfg(
            max_convex_hulls=64,
            hull_vertex_limit=64,
            shrink_wrap=True,
        ),
    )
).usd_path
