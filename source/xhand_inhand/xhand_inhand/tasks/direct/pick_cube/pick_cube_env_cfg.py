# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: FR3 + XHand pick a cube up and reorient it to a target pose.

Reward follows the SimToolReal-style staged structure:

  * pre-lift fingertip distance progress toward the cube center
  * sparse one-shot lift bonus once the cube clears the table
  * post-lift keypoint progress from cube corners to goal-pose corners

The reward is progress-based instead of occupancy-based so the policy cannot farm
reward by merely hovering near, holding still, or lifting without carrying/reorienting.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from xhand_inhand.foundationpose_cube import FOUNDATIONPOSE_CUBE_SCALE, FOUNDATIONPOSE_CUBE_USD
from xhand_inhand.robots import FR3_XHAND_CFG


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
        # scene default material -> table + object (no explicit material) get robot_friction (0.5),
        # matching SimToolReal (table & object both run through the default 0.5 material). The
        # robot's own shapes are overwritten per-shape after startup (see apply_fingertip_friction).
        physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    # per-shape friction (SimToolReal defaults): all robot shapes 0.5, fingertip distal links
    # 1.5 -> a 3x grip advantage at the fingerpads. Applied init-only via the physx view in
    # the env __init__ (no DR; SimToolReal's friction scale ranges are (1.0, 1.0) by default).
    robot_friction = 0.5
    fingertip_friction = 1.5

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot (no contact sensor needed -- the lift reward is height-gated)
    robot_cfg: ArticulationCfg = FR3_XHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    palm_body_name = "palm"
    # the 5 finger pads form the "end effector" (grasp assembly) for the reach reward
    ee_body_names = ["index_rota_link2", "mid_link2", "ring_link2", "pinky_link2", "thumb_rota_link2"]
    # palm-center point in the PALM BODY frame (the XHand "palm" origin is at the wrist;
    # offset toward the fingers (+Z) and the palm side (-Y) to get the grasp center).
    palm_center_offset = (0.0, -0.02, 0.07)
    # Per-finger-pad offset in each distal link's local frame.
    # These are link-local pad-center points calibrated interactively in Isaac Sim.
    finger_pad_offsets = {
        "thumb_rota_link2": (0.033409, 0.000346, 0.012429),
        "index_rota_link2": (-0.002238, -0.011313, 0.026695),
        "mid_link2": (0.000509, -0.014334, 0.023363),
        "ring_link2": (0.000705, -0.013922, 0.025485),
        "pinky_link2": (-0.000383, -0.011856, 0.028925),
    }

    # cube on the table
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=FOUNDATIONPOSE_CUBE_USD,
            scale=FOUNDATIONPOSE_CUBE_SCALE,
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

    # action: full RELATIVE joint position control, smoothed by target moving average
    action_scale = 0.1
    act_moving_average = 0.3

    # reset randomization of the cube on the table (x/y position + full yaw)
    reset_object_pos_noise = (0.10, 0.20)
    reset_object_yaw_range = (-3.14159, 3.14159)
    reset_min_hand_object_dist = 0.060

    # ---- goal: lift the cube to a FIXED point + a target ORIENTATION ----
    # Orientation is specified as roll/pitch/yaw ranges, exactly like the reference
    # lift task's `UniformPoseCommandCfg.Ranges` (which fixes them to 0; here they
    # are opened up so the policy must reorient the cube to the shown pose).
    target_pos = (0.5, 0.0, 0.35)
    target_rot_range_roll = (-3.14159, 3.14159)
    target_rot_range_pitch = (-3.14159, 3.14159)
    target_rot_range_yaw = (-3.14159, 3.14159)

    # ---- staged reward ----
    lift_z_offset = 0.05  # SimToolReal offset: threshold 0.15 means ~0.10 m actual lift
    lifting_bonus_threshold = 0.15
    lifting_bonus = 300.0
    distance_delta_rew_scale = 50.0
    keypoint_rew_scale = 200.0
    keypoint_half_extent = 0.030  # 6 cm cube half-edge
    success_tolerance = 0.05
    reach_goal_bonus = 1000.0
    success_steps = 10

    # action regularization: L1 joint velocity, arm penalized 10x the hand
    arm_joint_names = ["fr3_joint[1-7]"]
    hand_joint_names = ["(thumb|index|middle|ring|pinky)_joint.*"]
    kuka_actions_penalty_scale = 0.03
    hand_actions_penalty_scale = 0.003

    # termination: cube fell this far below its rest height
    drop_height = 0.10

    # debug markers: palm-center (red sphere), fingertips (green spheres), palm-normal ray (blue).
    # only built when a GUI is present, so headless training pays nothing.
    debug_markers = True

    # floating goal-pose marker (a cube drawn at the target pose)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=FOUNDATIONPOSE_CUBE_USD,
                scale=FOUNDATIONPOSE_CUBE_SCALE,
            )
        },
    )
