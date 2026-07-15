# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

""""Tool" asset shared by the pick_tool_token task.

Runtime-converts the high-poly textured tool mesh (``textured_mesh.obj``, authored in
meters, ~0.194 x 0.176 x 0.163 m) to a USD with a CONVEX-DECOMPOSITION collision so the
concave tool can actually be grasped -- a bounding cube (as used for the simple pick-cube)
would let the fingers close on empty space around the real geometry.

Mirrors the ``foundationpose_cube`` pattern: the conversion runs once at import time (the
Isaac Sim app is already up by the time the task packages are imported) and is cached in
``usd_dir`` keyed by the mesh hash, so repeated runs pay nothing.
"""

import os

import isaaclab.sim as sim_utils

_THIS_DIR = os.path.dirname(__file__)

TOOL_OBJ = os.path.join(_THIS_DIR, "textured_mesh.obj")
"""Textured tool mesh (trimesh export, already in meters)."""

TOOL_SCALE = (1.0, 1.0, 1.0)
"""Mesh is authored in meters at its true size -- no rescale."""

# TRUE stable resting pose, measured in-sim by DROPPING the hammer from height (0.35 & 0.50 m,
# away from the arm) and letting it fully tumble+settle -- both drops converge to z=-0.0468,
# quat below, with ~0 rebound. NOTE: the authored identity pose (old z=0.0735) is only a
# METASTABLE perch -- it holds if placed gently but tumbles to this true rest the instant it's
# touched, dropping the root below the drop-termination threshold and resetting the episode
# (the "object bounces then episode resets" bug that sabotaged training). Spawning directly in
# the true rest keeps the object stable when the hand contacts it, and ``actual_lift`` reads ~0
# at rest. A world-Z yaw at reset preserves this resting face (rotates about the vertical).
TOOL_REST_Z = -0.0468
TOOL_REST_QUAT = (0.94733, 0.19237, -0.25430, 0.02996)  # (w, x, y, z) -- true tumbled rest

TOOL_USD = sim_utils.MeshConverter(
    sim_utils.MeshConverterCfg(
        asset_path=TOOL_OBJ,
        usd_dir="/tmp/xhand_inhand/pick_tool_token",
        usd_file_name="tool_pentagon.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        # convex decomposition: many convex hulls approximating the concave tool so the
        # fingers contact the real surface (needed for a graspable, non-tunneling shape).
        mesh_collision_props=sim_utils.ConvexDecompositionPropertiesCfg(
            max_convex_hulls=64,
            hull_vertex_limit=64,
            shrink_wrap=True,
        ),
    )
).usd_path
"""Runtime-converted USD path for IsaacLab's USD spawner."""
