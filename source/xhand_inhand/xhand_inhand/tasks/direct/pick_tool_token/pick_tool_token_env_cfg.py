# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: xArm7 + XHand pick-a-tool with CrossDex action tokenization.

Same task / reward / scene / action pipeline as ``pick_cube_token`` (7 arm joint deltas +
a 9-dim eigengrasp token retargeted to 12 absolute xhand joint targets, then the staged
SimToolReal lift reward). The ONLY change is the grasped object: the FoundationPose cube is
swapped for the concave pentagon "tool" mesh (see ``tool_asset.py``).

Action layout (16 = 7 + 9), observation layout (86) and the reward are all inherited
verbatim from ``pick_cube_token`` -- the policy just has to grasp and lift a bulkier,
irregular object instead of a cube.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from ..pick_cube_token.pick_cube_token_env_cfg import PickCubeTokenEnvCfg
from .tool_asset import TOOL_REST_QUAT, TOOL_REST_Z, TOOL_SCALE, TOOL_USD


@configclass
class PickToolTokenEnvCfg(PickCubeTokenEnvCfg):
    # action (16 = 7 arm deltas + 9 hand eigengrasp token) and observation (86) are inherited
    # unchanged from pick_cube_token.

    # the bulky, irregular tool takes a little longer to line the grasp up than a small cube
    episode_length_s = 6.0

    # ---- the tool on the table (convex-decomposition collision baked into TOOL_USD) ----
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TOOL_USD,
            scale=TOOL_SCALE,
            # explicit mass so the weak XHand fingers can hold it and the dynamics stay damped
            # (auto-computed density-based mass on a ~10 cm shell can come out heavy/uneven).
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        # spawn in the measured settled pose (no per-reset tumble); reset yaws it about world Z
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, TOOL_REST_Z), rot=TOOL_REST_QUAT),
    )

    # reset randomization on the table: a bulkier object -> keep the hand a touch farther at
    # reset so it never spawns overlapping the tool, and a slightly tighter xy spread.
    reset_object_pos_noise = (0.08, 0.15)
    reset_object_yaw_range = (-3.14159, 3.14159)
    reset_min_hand_object_dist = 0.08

    # HANDLE grasp KEYPOINTS in the object's local mesh frame (SimToolReal-style: a set of
    # reference points spanning the graspable handle region instead of one point). The mesh
    # ROOT sits at the hammer's ungraspable bottom, so the inherited pick_cube reward -- which
    # drives fingertips to the object root -- makes the policy hover / press with the hand back.
    # Instead each fingertip is driven to its NEAREST handle keypoint, so the fingers wrap the
    # handle centerline. Points found via PCA of the mesh (long axis, thin-cross-section span);
    # tune them by dragging the yellow markers in the GUI (see pick_tool_token_env.py).
    # hand-tuned in the GUI (dragged the yellow markers onto the handle grasp region)
    grasp_keypoints = (
        (-0.0083, 0.0393, 0.0977),
        (-0.0201, 0.0160, 0.0921),
        (-0.0132, 0.0581, 0.0788),
        (-0.0264, 0.0264, 0.0750),
    )

    # GRASP GATE: the lift bonus + dense lift reward pay ONLY when the hand is truly closed on
    # the handle keypoints -- the thumb AND at least ``grasp_min_other_fingers`` other fingertips
    # within ``grasp_close_thr`` of a handle keypoint. This blocks the "knock/scoop the hammer up
    # with fingers 9 cm away" reward hack (a fingertip pad on the ~2 cm-radius handle sits ~2-4 cm
    # from a centerline keypoint, so 0.05 m is a real-grasp threshold, not a loose one).
    grasp_close_thr = 0.07
    grasp_min_other_fingers = 1

    # draggable grasp-keypoint markers (yellow spheres on the object) -- GUI-only, for visually
    # checking / repositioning the handle keypoints. Costs nothing headless (gated on has_gui()).
    debug_grasp_marker = True

    # floating goal-pose marker drawn with the tool mesh (same as the cube task's marker)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=TOOL_USD, scale=TOOL_SCALE)},
    )
