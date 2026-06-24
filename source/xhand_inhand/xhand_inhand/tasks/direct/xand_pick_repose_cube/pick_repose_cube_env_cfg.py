# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: FR3 + XHand pick a cube up off a table and reorient it to a
target pose. The reward is borrowed VERBATIM in structure from SimToolReal (the
sim2real-validated arm+hand pick-and-reorient task this project is porting), which
sidesteps the reward-farming traps we hit with hand-tuned staged rewards.

Every term is PROGRESS(delta) / GATED / ONE-SHOT -- there is NO persistent occupancy
term, so the policy cannot farm reward by holding the cube still (the failure mode of
both the "hug the cube" and "lift but don't carry" local optima we observed). Two
phases, gated on a latched `lifted` flag:

  PRE-LIFT:
    * lifting   -> dense clamp(z_lift, 0, 0.5) height reward (shut off once lifted)
    * fingertip -> SUM over 5 fingertips of closest-distance PROGRESS to the object
                   (sum, not mean, so every finger is encouraged to participate)
  AT LIFT:
    * a SPARSE one-shot lift bonus the instant the cube clears the table
  POST-LIFT:
    * keypoint  -> PROGRESS on the max distance between the 4 cube-corner keypoints and
                   the goal-pose corners. A keypoint (corner) distance unifies POSITION
                   and ORIENTATION error into one scalar, so this single term drives both
                   carrying to the goal point and reorienting to the goal pose.
  ON SUCCESS (keypoint error within tolerance for `success_steps` steps):
    * an amortized goal bonus, then the goal pose RESAMPLES -> continuous reorientation
      (the moving goalpost prevents camping, exactly as in SimToolReal / xhand_repose).

Action regularization is an L1 joint-velocity penalty, arm penalized ~10x the hand
(the hand must stay free to manipulate). Episode ends on drop, lost grasp, or after
`max_consecutive_successes` goals.
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
class PickReposeCubeEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_space = 19  # full relative joint control: 7 arm + 12 hand
    # obs = joint_pos(19)+joint_vel(19)+ee_pos_b(5*3=15)+palm_center_b(3)
    #       +object_pos_b(3)+object_quat(4)+target_pos_b(3)+target_quat(4)+actions(19)
    #       +tip_contact_mag(5) = 94  (fingertip contact force is in the obs so the policy can
    #       learn the "close fingers -> contact" causality; the real XHand exposes fingertip tactile)
    observation_space = 94
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

    # robot -- contact sensors ENABLED on the fingertips (real XHand has fingertip tactile;
    # we use net contact force to detect a real force-closure grasp, not a geometric "pose")
    robot_cfg: ArticulationCfg = FR3_XHAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FR3_XHAND_CFG.spawn.replace(activate_contact_sensors=True),
    )
    palm_body_name = "palm"
    # the 5 finger pads form the "end effector" (grasp assembly) for the reach reward
    ee_body_names = ["index_rota_link2", "mid_link2", "ring_link2", "pinky_link2", "thumb_rota_link2"]
    thumb_tip_name = "thumb_rota_link2"
    contact_force_threshold = 1.0  # N; per-fingertip net force above this counts as "in contact"
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

    # ====================================================================================
    # Reward: borrowed VERBATIM in structure from SimToolReal (arm+hand pick & reorient,
    # sim2real-validated). All terms are progress(delta)/gated/one-shot -- NO persistent
    # occupancy term, so there is no farming channel. Two phases, gated on `lifted`:
    #   before lift:  dense lifting height + fingertip-distance PROGRESS (sum over 5 tips)
    #   at lift:      one-shot lift bonus
    #   after lift:   keypoint PROGRESS (4 cube corners -> goal corners; unifies pos+orient)
    #   on success:   amortized goal bonus, resample a new goal (continuous reorientation)
    # ====================================================================================
    # joint groups for the (arm-heavy, hand-light) L1 velocity penalty
    arm_joint_names = ["fr3_joint[1-7]"]
    hand_joint_names = ["(thumb|index|middle|ring|pinky)_joint.*"]

    # -- lifting (dense before lift, shut off after; plus a one-shot bonus on crossing) --
    lift_z_offset = 0.05            # SimToolReal offset so z_lift starts positive
    lifting_bonus_threshold = 0.15  # z_lift above this -> "lifted" (i.e. ~0.10 m off the table)
    lifting_bonus = 300.0           # SPARSE one-shot bonus at the instant of lift
    lifting_rew_scale = 20.0        # dense clamp(z_lift,0,0.5) height reward (pre-lift only)

    # -- fingertip approach (progress, pre-lift): sum of per-tip closest-distance improvements --
    distance_delta_rew_scale = 50.0

    # -- keypoint tracking (progress, post-lift): max over 4 cube corners -> goal corners --
    keypoint_rew_scale = 200.0
    keypoint_half_extent = 0.030    # cube half-edge (m) for the 4 corner keypoints

    # -- grasp quality (geometric, no contact sensor): palm-closeness x thumb-opposition.
    # Caging (fingers cage the object center) leaves the palm far and the thumb un-opposed -> low
    # quality; a force-closure palm grasp -> high. Used BOTH as a dense post-lift guide AND to GATE
    # the keypoint (manipulation) reward, so the object must be properly held before reorientation
    # pays -- which a cage cannot do. No contact sensor needed.
    palm_std = 0.06                 # tanh width (m) for palm-center -> object closeness
    w_grasp = 5.0                   # dense grasp-quality reward (active pre-lift too, to bootstrap)
    # keypoint reward is gated by grasp quality, but with a FLOOR so caging still earns enough to
    # keep "lift" valuable (a pure gate deadlocked: no grasp -> no keypoint -> never lifts). The
    # 0.25..1.0 range still makes a real palm grasp pay 4x a cage, driving the transition.
    grasp_gate_floor = 0.25

    # -- success: keypoint max-corner error within tolerance for `success_steps` steps --
    success_tolerance = 0.05        # keypoint max-corner distance tolerance (m)
    success_steps = 10              # steps within tolerance to bank a success
    reach_goal_bonus = 1000.0       # amortized per near-goal step (bonus / success_steps)
    max_consecutive_successes = 50  # terminate after this many goals solved in one episode

    # -- action regularization: L1 joint velocity, arm penalized 10x the hand --
    kuka_actions_penalty_scale = 0.03
    hand_actions_penalty_scale = 0.003

    # termination: cube fell this far below its rest height, or hand lost the object
    drop_height = 0.10
    hand_far_dist = 1.0             # max fingertip-to-object distance (m) before giving up

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
