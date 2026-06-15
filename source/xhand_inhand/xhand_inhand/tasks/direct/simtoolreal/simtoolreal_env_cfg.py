# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab DirectRLEnv config for the SimToolReal dexterous tool-manipulation task.

This is a faithful port of the SimToolReal Isaac Gym environment
(``simtoolreal-main/isaacgymenvs/tasks/simtoolreal/env.py``) to this repo's Isaac Lab
framework, with the robot swapped from iiwa14+Sharpa (29 DOF) to **xArm7 + XHand**
(19 DOF). The task: pick a procedurally-generated handle+head tool off a table and
reorient/move it to a sampled keypoint-pose goal, with a shrinking success tolerance.

The numeric task constants below are copied from ``SimToolReal.yaml``; robot-specific
constants (DOF counts, fingertip/palm bodies, home pose, gains) are adapted to ours.
"""

from __future__ import annotations

import json
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import XARM7_XHAND_CFG

_HERE = os.path.dirname(os.path.abspath(__file__))
_POOL_DIR = os.path.join(_HERE, "objects", "pool")
_MANIFEST = os.path.join(_POOL_DIR, "manifest.json")


def _load_pool():
    """Load the procedural object pool manifest (list of dicts). Empty if not generated."""
    if not os.path.exists(_MANIFEST):
        return []
    with open(_MANIFEST) as f:
        return json.load(f)["objects"]


def _build_object_spawn(pool):
    """Build a MultiAssetSpawnerCfg over the pool USDs (env i -> asset i % pool_size).

    random_choice=False makes the per-env assignment deterministic and index-ordered,
    so the env can recover each env's object dimensions from the manifest order.
    """
    # NOTE: max_depenetration_velocity kept SANE (1.0). The reference's 1000 m/s is an
    # Isaac Gym value; in Isaac Lab it violently ejects the convex-decomposition tools on
    # any tiny spawn/seam penetration -> the tools "jump". A couple of velocity iterations
    # damp contact bounce.
    obj_rigid = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
        max_depenetration_velocity=1.0,
        max_angular_velocity=200.0,
        max_linear_velocity=200.0,
        disable_gravity=False,
    )
    # NOTE: no collision_props here. The converted object USDs have INTERNALLY INSTANCED
    # collision meshes, so per-asset collision_props can't be applied (Isaac Lab floods
    # "Could not perform 'modify_collision_properties' on instanced prim ..." warnings and
    # the values are ignored anyway). The objects use the collision baked at URDF->USD
    # conversion time, which is fine. (Resting-contact stability comes from the solver
    # velocity iterations above, not from contact_offset.)
    assets_cfg = [
        sim_utils.UsdFileCfg(
            usd_path=os.path.join(_POOL_DIR, entry["usd"]),
            rigid_props=obj_rigid,
        )
        for entry in pool
    ]
    return MultiAssetSpawnerCfg(
        assets_cfg=assets_cfg,
        random_choice=False,
        rigid_props=obj_rigid,
    )


@configclass
class SimToolRealEnvCfg(DirectRLEnvCfg):
    # ---------------------------------------------------------------- env / spaces
    # 60 Hz control. Reference ran physics at 60 Hz (dt=1/60, controlFrequencyInv=1);
    # we use dt=1/120 with decimation=2 -> 60 Hz control but a stiffer 120 Hz solve.
    decimation = 2
    episode_length_s = 10.0  # episodeLength=600 steps @ 60 Hz
    # action = 19 (7 arm + 12 hand) relative/absolute joint position targets
    action_space = 19
    # obsList  -> 110 : jpos19+jvel19+prevtarg19+palmpos3+palmrot4+objrot4
    #                   +fingertip_rel_palm(5*3=15)+kp_rel_palm(4*3=12)+kp_rel_goal12+objscale3
    observation_space = 110
    # stateList -> 132 (asymmetric critic): obs(110) - objrot already counted; adds
    #   palm_vel6 + object_vel6 + closest_kp1 + closest_ft5 + lifted1 + progress1
    #   + successes1 + reward1 = 110 + 22 = 132
    state_space = 132

    # ------------------------------------------------------------------------ sim
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        # Isaac Lab native physx: TGS solver + DEFAULT iteration ranges so each body's own
        # solver_*_iteration_count applies (the old global caps forced velocity iters to 0
        # everywhere -> bouncy resting contacts -> tools jumping). Per-body counts: object
        # pos=8/vel=2, robot pos=8/vel=0.
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=1,
            max_position_iteration_count=255,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=255,
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    # replicate_physics=False: each env gets a DIFFERENT procedural object (MultiAsset),
    # so the physics scene cannot be replicated from a single source env.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=1.2, replicate_physics=False
    )

    # --------------------------------------------------------------------- robot
    robot_cfg: ArticulationCfg = XARM7_XHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # arm vs hand joint name patterns (Isaac Lab reorders joints -> address by name)
    arm_joint_expr = "joint[1-7]"
    hand_joint_expr = "(thumb|index|middle|ring|pinky)_joint.*"
    palm_body_name = "palm"
    # 5 fingertip bodies (xhand distal links), order = thumb,index,middle,ring,pinky
    fingertip_body_names = (
        "thumb_rota_link2",
        "index_rota_link2",
        "mid_link2",
        "ring_link2",
        "pinky_link2",
    )
    # palm-center offset in the palm body frame (toward the fingers); TUNE visually.
    palm_offset = (0.0, -0.02, 0.16)
    # per-fingertip tip offset in the fingertip body frame; TUNE visually.
    fingertip_offset = (0.02, 0.002, 0.0)

    # ----------------------------------------------------------------- table layout
    # The xArm7 is MOUNTED ON the tabletop (standard tabletop manipulation rig): the arm
    # base sits at z=table_top_z, and the tool lies on the SAME table in front of it.
    table_top_z = 0.4  # tabletop height = arm base height
    # table box (top flush at table_top_z), spanning under the arm and forward toward the
    # workspace. center_z = table_top_z/2 so the box rises from the floor to the top.
    table_size = (1.2, 1.5, 0.4)
    table_center = (0.30, 0.0, 0.2)
    # where the tool rests on the table, in front of the arm base (env-local). Centered
    # in xArm7's comfortable forward reach; the spawn spread is set by reset_position_noise
    # below so the tool lands within x~[0.30,0.46], y~[-0.12,0.12] (all reachable).
    object_spawn_x = 0.38
    object_spawn_y = 0.0
    # drop clearance above the tabletop: enough that a randomly-oriented (possibly upright)
    # long tool doesn't spawn penetrating the table; it free-falls and settles flat.
    object_rest_z = table_top_z + 0.12
    table_reset_z_range = 0.01  # tableResetZRange (folded into object init z noise)

    # -------------------------------------------------------------------- objects
    _pool = _load_pool()
    num_pool_objects = len(_pool)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=_build_object_spawn(_pool) if _pool else None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(object_spawn_x, object_spawn_y, object_rest_z)),
    )
    object_manifest_path = _MANIFEST

    # =====================================================================
    # ============== ported task constants (SimToolReal.yaml) =============
    # =====================================================================
    clamp_abs_observations = 10.0

    # ---- control / action mapping ----
    dof_speed_scale = 1.5  # dofSpeedScale
    arm_moving_average = 0.1  # armMovingAverage (EMA blend on arm targets)
    hand_moving_average = 0.1  # handMovingAverage (EMA blend on hand targets)
    use_relative_control = False  # useRelativeControl (arm rel-to-prev-target either way)

    # ---- reset randomization ----
    # object spawn spread, sized to xArm7's reachable tabletop region (not the iiwa14
    # defaults): x in [0.30,0.46], y in [-0.12,0.12] around (object_spawn_x, _y).
    reset_position_noise_x = 0.08  # resetPositionNoiseX
    reset_position_noise_y = 0.12  # resetPositionNoiseY
    reset_position_noise_z = 0.02  # resetPositionNoiseZ
    randomize_object_rotation = True  # randomizeObjectRotation
    # start-pose randomization re-enabled but REDUCED on the arm so the randomized hand
    # stays above the tabletop (no clipping into the table). vel noise off for a calm
    # start. Reference values were arm/fingers=0.1, vel=0.5.
    # start-pose randomization (small on the arm so the randomized hand stays above the
    # tabletop). Reference values were arm/fingers=0.1, vel=0.5.
    reset_dof_pos_noise_fingers = 0.1  # resetDofPosRandomIntervalFingers
    reset_dof_pos_noise_arm = 0.05  # resetDofPosRandomIntervalArm
    reset_dof_vel_noise = 0.0  # resetDofVelRandomInterval (kept 0 for a calm start)
    object_scale_noise_multiplier_range = (1.0, 1.0)  # objectScaleNoiseMultiplierRange

    # ---- random object disturbance forces (domain randomization) ----
    force_scale = 20.0  # forceScale
    force_prob_range = (0.001, 0.1)  # forceProbRange
    force_decay = 0.0  # forceDecay
    force_decay_interval = 0.08  # forceDecayInterval
    force_only_when_lifted = True  # forceOnlyWhenLifted
    torque_scale = 2.0  # torqueScale
    torque_prob_range = (0.001, 0.1)  # torqueProbRange
    torque_decay = 0.0  # torqueDecay
    torque_decay_interval = 0.08
    torque_only_when_lifted = True  # torqueOnlyWhenLifted

    # ---- reward scales ----
    lifting_rew_scale = 20.0  # liftingRewScale
    lifting_bonus = 300.0  # liftingBonus
    lifting_bonus_threshold = 0.15  # liftingBonusThreshold
    keypoint_rew_scale = 200.0  # keypointRewScale
    distance_delta_rew_scale = 50.0  # distanceDeltaRewScale
    reach_goal_bonus = 1000.0  # reachGoalBonus
    kuka_actions_penalty_scale = 0.03  # kukaActionsPenaltyScale (arm)
    hand_actions_penalty_scale = 0.003  # handActionsPenaltyScale
    object_lin_vel_penalty_scale = 0.0  # objectLinVelPenaltyScale (disabled)
    object_ang_vel_penalty_scale = 0.0  # objectAngVelPenaltyScale (disabled)

    # ---- keypoints ----
    keypoint_scale = 1.5  # keypointScale
    object_base_size = 0.04  # objectBaseSize
    fixed_size_keypoint_reward = True  # fixedSizeKeypointReward
    fixed_size = (0.141, 0.03025, 0.0271)  # fixedSize (reward keypoint box)
    num_keypoints = 4

    # ---- success / tolerance curriculum ----
    success_tolerance = 0.075  # successTolerance (initial)
    target_success_tolerance = 0.01  # targetSuccessTolerance
    tolerance_curriculum_increment = 0.9  # toleranceCurriculumIncrement (multiplicative)
    tolerance_curriculum_interval = 3000  # toleranceCurriculumInterval (env steps)
    max_consecutive_successes = 50  # maxConsecutiveSuccesses
    success_steps = 10  # successSteps
    force_consecutive_near_goal_steps = False  # forceConsecutiveNearGoalSteps

    # ---- resets ----
    fall_z = table_top_z - 0.15  # object_pos.z < this -> fell off the table
    hand_far_dist = 1.5  # max fingertip-object dist -> reset
    reset_when_dropped = False  # resetWhenDropped

    # ---- goal sampling ----
    # lift/reorient region: above the table, in front of the arm, within xArm7 reach.
    goal_sampling_type = "delta"  # goalSamplingType: "delta" | "absolute" | "coin_flip"
    target_volume_region_scale = 1.0  # targetVolumeRegionScale
    delta_goal_distance = 0.1  # deltaGoalDistance
    delta_rotation_degrees = 90.0  # deltaRotationDegrees
    # lift/reorient goal region, sized to xArm7's reachable space above the table front.
    target_volume_mins = (0.30, -0.15, table_top_z + 0.08)  # targetVolumeMins
    target_volume_maxs = (0.50, 0.15, table_top_z + 0.28)  # targetVolumeMaxs

    # ---- delays / observation noise ----
    use_obs_delay = True  # useObsDelay
    obs_delay_max = 3  # obsDelayMax
    use_action_delay = True  # useActionDelay
    action_delay_max = 3  # actionDelayMax
    use_object_state_delay_noise = True  # useObjectStateDelayNoise
    object_state_delay_max = 10  # objectStateDelayMax
    object_state_xyz_noise_std = 0.01  # objectStateXyzNoiseStd
    object_state_rotation_noise_degrees = 5.0  # objectStateRotationNoiseDegrees
    joint_velocity_obs_noise_std = 0.1  # jointVelocityObsNoiseStd

    # ---- start pose ----
    start_arm_higher = False  # startArmHigher

    # goal marker (floating tool at the target pose). Uses the first pool object's USD.
    goal_marker_cfg: VisualizationMarkersCfg | None = None

    def __post_init__(self):
        # Mount the xArm7 base ON the tabletop (raise it from the floor to table_top_z).
        self.robot_cfg.init_state.pos = (0.0, 0.0, self.table_top_z)
