"""Object-specific functional-pregrasp specialist for OakInk ``flashlight_1``."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from ..pick_tool_token.pick_tool_token_env_cfg import PickToolTokenEnvCfg
from .flashlight_asset import (
    FLASHLIGHT_MASS,
    FLASHLIGHT_OBJ,
    FLASHLIGHT_REST_QUAT,
    FLASHLIGHT_REST_Z,
    FLASHLIGHT_SCALE,
    FLASHLIGHT_USD,
)


@configclass
class FunctionalPregraspFlashlightEnvCfg(PickToolTokenEnvCfg):
    """The first calibrated object swap; action/observation/body are unchanged."""

    specialist_object_id = "flashlight_1"
    specialist_intent = "use"

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=FLASHLIGHT_USD,
            scale=FLASHLIGHT_SCALE,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
                # PhysX has no rolling-resistance model for an almost perfect cylinder, and
                # Isaac Lab deliberately disables sleep for bodies with contact reporting.
                # Explicit angular drag therefore represents rolling loss.  Hand contact still
                # wakes/drives the body normally because the object remains fully dynamic.
                linear_damping=0.05,
                angular_damping=0.5,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, FLASHLIGHT_REST_Z), rot=FLASHLIGHT_REST_QUAT
        ),
    )

    object_mesh_obj: str = str(FLASHLIGHT_OBJ)
    object_mesh_scale = FLASHLIGHT_SCALE
    expected_object_mass = FLASHLIGHT_MASS
    object_aabb_min = (-0.0190692, -0.0190692, -0.0694385)
    object_aabb_max = (0.0190692, 0.0190692, 0.0693278)

    # Four points are exact intersections with the visual surface at z=-15 mm.  The source
    # mesh has sparse longitudinal vertex rings, so the analytic section needs a wider slice.
    grasp_keypoints = (
        (0.014484, 0.0, -0.015),
        (0.0, 0.014484, -0.015),
        (-0.014484, 0.0, -0.015),
        (0.0, -0.014484, -0.015),
    )
    grasp_keypoint_dirs = None
    handle_center = (0.0, 0.0, -0.015)
    handle_axis = (0.0, 0.0, 1.0)
    handle_axial_min = -0.050
    handle_axial_max = 0.040
    handle_axial_margin = 0.002
    handle_section_half_width = 0.019
    handle_contact_margin = 0.008

    inhand_point = (0.0, 0.0, 0.068683)
    inhand_head_axis = (0.0, 0.0, 1.0)

    # The lens end (+local Z) is directed, so opposite headings are not equivalent.  Roll
    # about the cylindrical axis is continuous and deliberately omitted from the terminal
    # pose metric; table support plus angular settling remain mandatory.
    nudge_heading_axis = (0.0, 0.0, 1.0)
    nudge_yaw_symmetry_order = 1
    nudge_stable_up_axes = ()
    nudge_terminate_on_support_loss = False
    nudge_pos_tolerance = 0.04
    nudge_yaw_tolerance = 0.20
    nudge_max_obj_speed = 0.08
    nudge_max_obj_ang_speed = 0.15
    nudge_timeout_penalty = 100.0
    reset_object_pos_noise = (0.08, 0.15)
    reset_min_hand_object_dist = 0.08
    carry_goal_z_margin = 0.12

    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=FLASHLIGHT_USD, scale=FLASHLIGHT_SCALE)},
    )
