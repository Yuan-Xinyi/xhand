# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: FR3 + XHand grasps a cube and reorients it to a target pose.

Faithful port of IsaacLab's Dexsuite Kuka-Allegro REORIENT task to the Direct
workflow with our FR3+XHand and a compact cube (orientation-forgiving -> a clean
grasp baseline before moving to thin objects):
  * full RELATIVE joint position control (arm + hand), scale 0.1
  * fingertip-object contact -> grasp detection (thumb + >=1 finger)
  * reach = max distance over (palm-center + fingertips) to the object center
  * position + orientation tracking to a target pose, GATED by grasp
  * success reward on combined position + orientation error
"""

import os  # noqa: F401

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
class Fr3ReorientEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_space = 19  # full relative joint control: 7 arm + 12 hand
    # obs = joint_pos(19)+joint_vel(19)+ee_pos_b(5*3=15)+contact(5)+object_pos_b(3)
    #       +object_quat(4)+target_pos_b(3)+target_quat(4)+actions(19) = 91
    observation_space = 91
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

    # robot (contact sensors enabled)
    robot_cfg: ArticulationCfg = FR3_XHAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FR3_XHAND_CFG.spawn.replace(activate_contact_sensors=True),
    )
    palm_body_name = "palm"
    fingertip_body_names = ["index_rota_link2", "mid_link2", "ring_link2", "pinky_link2", "thumb_rota_link2"]
    thumb_tip_name = "thumb_rota_link2"
    # palm-center point in the palm BODY frame (XHand "palm" origin is at the wrist;
    # offset toward the fingers (+Z) and the palm side (-Y) to the grasp center).
    palm_center_offset = (0.0, -0.02, 0.07)

    # cube object on the table
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CUBE_USD,
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    table_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    table_pos = (0.5, 0.0, 0.0)
    table_rot = (0.707, 0.0, 0.0, 0.707)

    # action: full RELATIVE joint position control (target += action_scale * action)
    action_scale = 0.1

    # reset randomization of the cube on the table (x/y position + full yaw)
    reset_object_pos_noise = (0.10, 0.20)
    reset_object_yaw_range = (-3.14159, 3.14159)

    # ---- goal: lift + reorient to a target pose (position + full orientation) ----
    target_pos_range_x = (0.40, 0.60)
    target_pos_range_y = (-0.15, 0.15)
    target_pos_range_z = (0.25, 0.40)

    # ---- contact / grasp ----
    contact_force_threshold = 1.0  # N

    # ---- reward (dexsuite weights) ----
    reach_std = 0.4
    pos_track_std = 0.1
    rot_track_std = 0.3
    success_pos_std = 0.05
    success_rot_std = 0.1
    w_reach = 1.0
    w_pos_track = 2.0
    w_orient_track = 4.0
    w_success = 10.0
    w_action_l2 = -0.005
    w_action_rate_l2 = -0.005

    drop_height = 0.10  # cube fell this far below rest -> terminate

    # floating goal-pose marker (a cube at the target pose)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=_CUBE_USD, scale=(0.75, 0.75, 0.75))},
    )
