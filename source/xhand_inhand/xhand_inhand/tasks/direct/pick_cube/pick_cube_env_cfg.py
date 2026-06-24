# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: FR3 + XHand pick a cube up and reorient it to a target pose.

This replicates IsaacLab's `Isaac-Lift-Cube-Franka-v0` (the manager-based lift task
that actually trains), keeping the single design decision that makes it learnable --
the lift / tracking rewards are gated on the object's HEIGHT, not on a contact
sensor -- and ADDS a target ORIENTATION on top of the lift target:

  * lift the cube off the table (height-gated bootstrap, +w_lift)
  * track a FIXED target point in the air (gated by lifted)
  * track a per-episode RANDOM target orientation (gated by lifted)

Height-gating is trivially discoverable (any upward nudge of the cube -> instant
lift reward), and lifting a cube off a table is only possible by grasping it, so
the height gate induces the grasp without ever detecting one.  Once the cube is
held, the orientation term shapes it toward the shown target pose.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from xhand_inhand.robots import FR3_XHAND_CFG

_CUBE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"


@configclass
class PickCubeEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_space = 19  # full relative joint control: 7 arm + 12 hand
    # obs = joint_pos(19)+joint_vel(19)+ee_pos_b(5*3=15)+palm_center_b(3)
    #       +object_pos_b(3)+object_quat(4)+target_pos_b(3)+target_quat(4)+actions(19) = 89
    observation_space = 89
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot (no contact sensor needed -- the lift reward is height-gated)
    robot_cfg: ArticulationCfg = FR3_XHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    palm_body_name = "palm"
    # the 5 fingertips form the "end effector" (grasp assembly) for the reach reward
    ee_body_names = ["index_rota_link2", "mid_link2", "ring_link2", "pinky_link2", "thumb_rota_link2"]
    # palm-center point in the PALM BODY frame (the XHand "palm" origin is at the wrist;
    # offset toward the fingers (+Z) and the palm side (-Y) to get the grasp center).
    palm_center_offset = (0.0, -0.02, 0.07)
    # per-fingertip TIP offset in EACH fingertip body's LOCAL frame, for visualization.
    # The link2 body ORIGIN sits at the proximal joint; the actual tip is ~4cm out along the
    # distal mesh axis -- +X for the thumb, +Z for the 4 fingers (measured from the xhand2R32
    # link2 STL bounding boxes; same values as simtoolreal). Keyed by body name so it is robust
    # to ee_body_names ordering / find_bodies reordering.
    fingertip_tip_offsets = {
        "thumb_rota_link2": (0.050, 0.000, -0.005),  # +X
        "index_rota_link2": (0.000, 0.004, 0.040),   # +Z
        "mid_link2": (0.000, 0.004, 0.040),          # +Z
        "ring_link2": (0.000, 0.004, 0.040),         # +Z
        "pinky_link2": (0.000, 0.004, 0.040),        # +Z
    }

    # cube on the table
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CUBE_USD,
            scale=(0.75, 0.75, 0.75),  # dex_cube 0.08 m -> 0.06 m edge
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    table_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    table_pos = (0.5, 0.0, 0.0)
    table_rot = (0.707, 0.0, 0.0, 0.707)

    # action: full RELATIVE joint position control (target += action_scale * action)
    action_scale = 0.1

    # reset randomization of the cube on the table (x/y position + full yaw)
    reset_object_pos_noise = (0.10, 0.20)
    reset_object_yaw_range = (-3.14159, 3.14159)

    # ---- goal: lift the cube to a FIXED point + a target ORIENTATION ----
    # Orientation is specified as roll/pitch/yaw ranges, exactly like the reference
    # lift task's `UniformPoseCommandCfg.Ranges` (which fixes them to 0; here they
    # are opened up so the policy must reorient the cube to the shown pose).
    target_pos = (0.5, 0.0, 0.35)
    target_rot_range_roll = (-3.14159, 3.14159)
    target_rot_range_pitch = (-3.14159, 3.14159)
    target_rot_range_yaw = (-3.14159, 3.14159)

    # ---- lift detection (mirror of franka `object_is_lifted`) ----
    # the cube counts as "lifted" once its center rises this far above its table rest
    lift_margin = 0.04

    # ---- reward weights (franka lift weights + an orientation term) ----
    reach_std = 0.2  # reaching tanh width (m)
    goal_track_std = 0.3  # coarse position-tracking tanh width (m)
    goal_track_fine_std = 0.05  # fine position-tracking tanh width (m)
    orient_track_std = 0.3  # orientation-tracking tanh width (rad)
    success_pos_std = 0.05
    success_rot_std = 0.1
    w_reach = 1.0
    w_lift = 15.0
    w_goal_track = 16.0
    w_goal_track_fine = 5.0
    w_orient_track = 8.0
    w_success = 10.0
    w_action_rate = -1e-4
    w_joint_vel = -1e-4

    # termination: cube fell this far below its rest height
    drop_height = 0.10

    # debug markers: palm-center (red sphere), fingertips (green spheres), palm-normal ray (blue).
    # only built when a GUI is present, so headless training pays nothing.
    debug_markers = True

    # floating goal-pose marker (a cube drawn at the target pose)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=_CUBE_USD, scale=(0.75, 0.75, 0.75))},
    )
