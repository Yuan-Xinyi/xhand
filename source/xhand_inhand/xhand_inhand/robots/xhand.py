# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the XHand right (12-DOF dexterous hand).

The USD is generated from ``assets/xhand2R32/xhand_right.urdf`` via Isaac Lab's
``scripts/tools/convert_urdf.py`` utility. Joint order (matches the source model):

    thumb  : thumb_joint0..2     -> q[0:3]
    index  : index_joint0..2     -> q[3:6]
    middle : middle_joint0..1    -> q[6:8]
    ring   : ring_joint0..1      -> q[8:10]
    pinky  : pinky_joint0..1     -> q[10:12]
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# absolute path to the converted USD shipped inside this package
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "xhand2R32")
XHAND_USD_PATH = os.path.abspath(os.path.join(_ASSETS_DIR, "xhand_right.usd"))


XHAND_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=XHAND_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=10.0,
            velocity_limit=3.14,
            stiffness=3.0,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Articulation configuration for the XHand right dexterous hand (fixed/free base)."""
