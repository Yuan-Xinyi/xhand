# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: the pick_tool_token multi-stage chain on the YCB 048 claw hammer.

A pure OBJECT SWAP of ``PickToolTokenEnvCfg``: same 21-D token action pipeline, same staged
reward/latch machinery, same option modes (nudge / close / carry / in-hand reorientation).
Only the object geometry changes -- mesh, rest pose, grasp keypoints, analytic handle frame
and the in-hand functional point are re-derived for the 33 cm google_16k hammer scan.

Geometry (object local frame, measured from the mesh):
  * long axis (handle) direction (-0.373, 0.928, 0.009); handle butt at t=-0.10, head
    junction at t=+0.13 along that axis from the grip centroid (0.001, -0.090, 0.016);
  * rubber grip radius 1.3--1.75 cm (a real graspable handle, vs the first tool's 2 mm web);
  * strike face center (-0.125, 0.053, 0.016), claw center (-0.025, 0.130, 0.014);
  * the scan lies flat (thickness 3.3 cm), authored pose == settled pose.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from ..pick_tool_token.pick_tool_token_env_cfg import PickToolTokenEnvCfg
from .hammer_asset import (
    HAMMER_MASS,
    HAMMER_OBJ,
    HAMMER_REST_QUAT,
    HAMMER_REST_Z,
    HAMMER_SCALE,
    HAMMER_USD,
)


@configclass
class PickHammerTokenEnvCfg(PickToolTokenEnvCfg):
    # ---- the hammer on the table (convex-decomposition collision baked into HAMMER_USD) ----
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HAMMER_USD,
            scale=HAMMER_SCALE,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, HAMMER_REST_Z), rot=HAMMER_REST_QUAT),
    )

    # raw OBJ behind the USD: true-clearance hull + handle cross-section polygon source
    object_mesh_obj: str = HAMMER_OBJ
    object_mesh_scale = HAMMER_SCALE
    expected_object_mass = HAMMER_MASS

    # The 3cm rubber grip settles into the palm on liftoff (thumb+1 pinch + palm rest);
    # accept the physics-proven airborne hold instead of the tabletop thumb+2 topology.
    # Chain diagnostics 2026-07-31: hold 0.97 / in-band pads 1.96 / other_coverage 0.0.
    airborne_pinch_hold = True

    # A 33 cm object: hand spawns a touch farther out so it can never overlap at reset.
    reset_min_hand_object_dist = 0.10

    # HANDLE grasp keypoints: CHOKE-UP cross-section (+7 cm from the grip centroid toward the
    # head), four points ON the mesh surface at ~90 deg spacing around the perimeter -- fingers
    # are driven to their nearest keypoint and wrap the grip just below the neck.  The scripted
    # grasp-and-hold probe (scripts/hammer_grasp_probe.py, 2026-07-30) measured held rates of
    # 12% pinned at the grip centroid vs 62% pinned here: the head-heavy mass distribution
    # (COM ~9 cm toward the head from the grip centroid) torques mid-grip holds out of the
    # closed hand, so the reward geometry points the fingers at the choke-up band instead.
    grasp_keypoints = (
        (-0.015251, -0.021192, 0.006236),
        (-0.033400, -0.028496, 0.006055),
        (-0.034368, -0.029069, 0.025692),
        (-0.017111, -0.022118, 0.025345),
    )
    # auto-seed keypoint directions radial-outward (re-freeze via the GUI drag tool if needed)
    grasp_keypoint_dirs = None

    # Analytic handle frame: choke-up point (grip centroid + 7 cm along the PCA handle axis)
    # with an asymmetric contact band -- [-8 cm, +5 cm] accepts anything from mid-grip to the
    # neck ramp (head junction at +6.4 cm) as valid handle contact, while the keypoints above
    # pull the closure toward the low-torque end of that band.
    handle_center = (-0.0251, -0.0253, 0.0163)
    handle_axis = (-0.3734, 0.9276, 0.0086)
    handle_axial_min = -0.08
    handle_axial_max = 0.05
    # section polygon radius measures 1.10--1.56 cm here (well inside the env sanity window)
    handle_axial_margin = 0.002
    handle_section_half_width = 0.002
    handle_contact_margin = 0.008

    # ---- in-hand reorientation: functional point + head-down operating axis ----
    # Functional point: top of the handle neck (choke-up region just below the head), the
    # analog of the first tool's annotated index-fingertip target.  Refine via
    # tools/functional_point/build_annotator.py; persisted copy in functional_point.json.
    inhand_point = (-0.03162, -0.01035, 0.02656)
    # Head axis = claw -> strike face: pointing it at -z puts the hammer in striking attitude.
    inhand_head_axis = (-0.79400, -0.60773, 0.01500)

    # Loose box for reset collision bounds / diagnostics only (reward uses the real hull).
    object_aabb_min = (-0.1288, -0.1891, -0.0006)
    object_aabb_max = (0.0534, 0.1436, 0.0322)

    # Carry-goal z floor: half diagonal of THIS hammer is ~0.19 m (vs the tool's ~0.13).
    carry_goal_z_margin = 0.22

    # goal-pose marker drawn with the hammer mesh
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=HAMMER_USD, scale=HAMMER_SCALE)},
    )
