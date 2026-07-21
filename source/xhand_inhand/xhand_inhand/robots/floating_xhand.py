"""Free-root XHand for the bounded floating-hand ablation."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "floating_xhand")
FLOATING_XHAND_USD_PATH = os.path.abspath(os.path.join(_ASSET_DIR, "floating_xhand_free.usd"))


FLOATING_XHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FLOATING_XHAND_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.19242, 0.01410, 0.60102),
        rot=(-0.015206, -0.690246, -0.723052, 0.022909),
        joint_pos={"(thumb|index|middle|ring|pinky)_joint.*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
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
"""Free root plus the stock 12-DOF XHand."""
