# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""YCB ``048_hammer`` (google_16k scan) asset shared by the pick_hammer_token task.

Same runtime-conversion pattern as ``pick_tool_token/tool_asset.py``: the textured OBJ
(authored in meters, ~0.182 x 0.333 x 0.033 m, a full-size claw hammer lying flat on its
cheek) is converted once at import time to a USD with CONVEX-DECOMPOSITION collision and
cached keyed by the mesh hash.

Unlike the first tool mesh, this scan's authored pose IS the physically settled pose: the
mesh bottom plane sits at z ~ -0.0006 and the hammer rests flat on the head cheek + handle
side.  The exact settled offset below was still measured in-sim by dropping (see the
pick_tool_token TOOL_REST_* history for why spawning in a non-settled pose corrupts
training).
"""

import os

import isaaclab.sim as sim_utils

# Mesh lives in the shared assets tree (source/.../assets/048_hammer_google_16k).
_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")
)

HAMMER_OBJ = os.path.join(_ASSETS_DIR, "048_hammer_google_16k", "048_hammer", "google_16k", "textured.obj")
"""Textured YCB 048_hammer mesh (google_16k, already in meters)."""

HAMMER_SCALE = (1.0, 1.0, 1.0)
"""Mesh is authored in meters at its true size -- no rescale."""

# Kept equal to the first tool's 0.15 kg on purpose: every force gate in the shared reward
# stack (contact_force_thr / safe_contact_force 20N / grasp_bonus_max_force 30N / tactile
# shields) and the lift dynamics recipe were calibrated against a 0.15 kg object.  The real
# YCB 048 hammer weighs 0.665 kg; raise this only together with a recalibration pass.
HAMMER_MASS = 0.15

# Settled rest pose, MEASURED in-sim 2026-07-29 with the converted USD: gentle place at
# identity, flat drop from 0.35 m and two tilted drops from 0.50 m ALL settle to identity
# orientation (residual quat components < 0.023) at z = -0.0004.  Unlike the first tool
# there is no metastable perch -- the authored scan pose is the one true rest.
HAMMER_REST_Z = -0.0004
HAMMER_REST_QUAT = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)

HAMMER_USD = sim_utils.MeshConverter(
    sim_utils.MeshConverterCfg(
        asset_path=HAMMER_OBJ,
        usd_dir="/tmp/xhand_inhand/pick_hammer_token",
        usd_file_name="hammer_048.usd",
        # Mass authored on the converter's rigid root (post-hoc UsdFileCfg mass edits can be
        # silently ignored on instanced prims -- same rationale as tool_asset.py).
        mass_props=sim_utils.MassPropertiesCfg(mass=HAMMER_MASS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=sim_utils.ConvexDecompositionPropertiesCfg(
            max_convex_hulls=64,
            hull_vertex_limit=64,
            shrink_wrap=True,
        ),
    )
).usd_path
"""Runtime-converted USD path for IsaacLab's USD spawner."""
