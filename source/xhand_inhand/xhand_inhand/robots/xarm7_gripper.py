"""xArm7 with a simple physical two-finger parallel gripper."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "xarm7_gripper")
XARM7_GRIPPER_USD_PATH = os.path.abspath(os.path.join(_ASSET_DIR, "xarm7_gripper.usd"))


XARM7_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=XARM7_GRIPPER_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.7494,
            "joint3": 0.0,
            "joint4": 1.1920,
            "joint5": 0.0,
            "joint6": 1.9414,
            "joint7": 0.0,
            ".*_finger_joint": 0.045,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "xarm7_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            stiffness=400.0,
            damping=80.0,
        ),
        "parallel_gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_finger_joint"],
            effort_limit_sim=15.0,
            velocity_limit_sim=0.25,
            stiffness=100.0,
            damping=15.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Seven arm joints plus two symmetric finger joints."""
