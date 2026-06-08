# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Research 3 (FR3) 7-DOF arm.

USD generated from ``assets/fr3/fr3.urdf`` via Isaac Lab's ``convert_urdf.py``
(fixed base). Arm-only — no gripper/end-effector in this URDF.

Joint order: fr3_joint1 .. fr3_joint7.
Effort limits (from URDF): joints 1-4 = 87 Nm, joints 5-7 = 12 Nm.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "fr3")
FR3_USD_PATH = os.path.abspath(os.path.join(_ASSETS_DIR, "fr3.usd"))


FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FR3_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # standard Franka "ready" home pose (within FR3 joint limits)
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.785,
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.356,
            "fr3_joint5": 0.0,
            "fr3_joint6": 1.571,
            "fr3_joint7": 0.785,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fr3_arm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-7]"],
            # effort/velocity limits are baked into the USD from the URDF
            stiffness=400.0,
            damping=80.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Articulation configuration for the Franka Research 3 (FR3) 7-DOF arm (fixed base)."""
