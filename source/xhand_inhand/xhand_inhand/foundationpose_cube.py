# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FoundationPose cube asset shared by cube manipulation tasks."""

import isaaclab.sim as sim_utils

FOUNDATIONPOSE_CUBE_OBJ = "/home/lqin/disk2/FoundationPose/cube/mesh/textured.obj"
"""Textured 6 cm cube mesh used by the FoundationPose cube pipeline."""

FOUNDATIONPOSE_CUBE_SCALE = (1.0, 1.0, 1.0)
"""The FoundationPose cube mesh is already authored in meters with a 0.06 m edge."""

FOUNDATIONPOSE_CUBE_USD = sim_utils.MeshConverter(
    sim_utils.MeshConverterCfg(
        asset_path=FOUNDATIONPOSE_CUBE_OBJ,
        usd_dir="/tmp/xhand_inhand/foundationpose_cube",
        usd_file_name="foundationpose_cube.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=sim_utils.BoundingCubePropertiesCfg(),
    )
).usd_path
"""Runtime-converted USD path for IsaacLab's USD spawner."""
