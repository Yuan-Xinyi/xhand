# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the combined xArm7 arm + XHand right hand (19 DOF).

USD generated from ``assets/xarm7_xhand/xarm7_xhand.urdf`` (xArm7 + XHand merged via a
fixed ``hand_mount`` joint: xArm7 flange ``link8`` -> XHand ``palm``). This is the robot
used by the ``simtoolreal`` dexterous tool-manipulation task (Isaac Lab port of the
SimToolReal Isaac Gym environment, which originally used an iiwa14 + Sharpa hand).

DOF = 7 arm (joint1..7) + 12 hand (thumb 3, index 3, middle/ring/pinky 2 each).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "xarm7_xhand")
XARM7_XHAND_USD_PATH = os.path.abspath(os.path.join(_ASSETS_DIR, "xarm7_xhand.usd"))


XARM7_XHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=XARM7_XHAND_USD_PATH,
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
        # NOTE: this home pose is a reasonable starting point (hand above/in front of the
        # base); tune the arm joints after a visual check so the hand hovers over the
        # table grasp region. xArm7 limits: j2 in [-2.18, 2.18], j4 in [-0.11, pi],
        # j6 in [-1.75, pi]; the rest are +-pi.
        joint_pos={
            # FINAL home pose (user-confirmed): hand hovers above the tabletop tool.
            "joint1": 0.0,
            "joint2": -0.7494,
            "joint3": 0.0,
            "joint4": 1.1920,
            "joint5": 0.0,
            "joint6": 1.9414,
            "joint7": 0.0,
            # XHand: open
            "(thumb|index|middle|ring|pinky)_joint.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "xarm7_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
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
"""Articulation configuration for the xArm7 arm + XHand right hand (19 DOF, fixed base)."""
