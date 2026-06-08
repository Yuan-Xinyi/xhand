# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the combined Franka FR3 arm + XHand right hand (19 DOF).

USD generated from ``assets/fr3_xhand/fr3_xhand.urdf`` (FR3 + XHand merged via a
fixed ``hand_mount`` joint: FR3 flange ``fr3_link8`` -> XHand ``palm``).

DOF = 7 arm (fr3_joint1..7) + 12 hand (thumb 3, index 3, middle/ring/pinky 2).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "fr3_xhand")
FR3_XHAND_USD_PATH = os.path.abspath(os.path.join(_ASSETS_DIR, "fr3_xhand.usd"))


FR3_XHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FR3_XHAND_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # FR3 arm: standard "ready" home pose
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.785,
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.356,
            "fr3_joint5": 0.0,
            "fr3_joint6": 1.571,
            "fr3_joint7": 0.785,
            # XHand: open
            "(thumb|index|middle|ring|pinky)_joint.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fr3_arm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-7]"],
            stiffness=400.0,
            damping=80.0,
        ),
        "xhand": ImplicitActuatorCfg(
            joint_names_expr=["(thumb|index|middle|ring|pinky)_joint.*"],
            effort_limit_sim=10.0,
            velocity_limit_sim=3.14,
            stiffness=3.0,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Articulation configuration for the FR3 arm + XHand right hand (19 DOF, fixed base)."""
