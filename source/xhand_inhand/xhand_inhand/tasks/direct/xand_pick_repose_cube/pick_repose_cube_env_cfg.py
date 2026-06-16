# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: FR3 + XHand pick a cube up, carry it to a target point,
then REORIENT it in-hand to a target pose -- a STAGED pick-and-repose task.

This builds on IsaacLab's `Isaac-Lift-Cube-Franka-v0` (the manager-based lift task
that actually trains), keeping the single design decision that makes it learnable --
the lift / tracking rewards are gated on the object's HEIGHT, not on a contact
sensor -- and stages the reward into three phases:

  1. PICK:    lift the cube off the table (height-gated bootstrap, +w_lift)
  2. CARRY:   track a FIXED target POINT in the air (gated by `lifted`)
  3. REPOSE:  track a per-episode RANDOM target ORIENTATION via in-hand manipulation,
              only rewarded once the cube is BOTH lifted AND near the target point
              (gated by `lifted & at_center`)

Height-gating is trivially discoverable (any upward nudge of the cube -> instant
lift reward), and lifting a cube off a table is only possible by grasping it, so
the height gate induces the grasp without ever detecting one.  Gating the
orientation term on `at_center` defers reorientation until the cube has been
transported to the goal point, so the policy learns to first carry, then reorient
in-hand -- rather than fighting both objectives during transport.

SUCCESS RATCHET (anti-camping): the rewards never pay for merely OCCUPYING a good
state -- which would let the policy farm reward by holding the cube still.  Instead:
  * reach  -> dense grasp-approach, but DECAYS to zero once the cube is lifted (so it
              cannot be farmed by holding the grasp still)
  * lift   -> a ONE-SHOT latched bonus (paid once, the first time the cube lifts)
  * carry  -> pays only for beating the CLOSEST distance to the goal reached so far

The REPOSE phase is taken VERBATIM from xhand_repose / InHandManipulationEnv -- the proven
ShadowHand/OpenAI in-hand reorientation reward -- only gated behind pick+carry:
  * a dense hyperbolic orientation reward  rot_rew = 1/(|rot_err| + rot_eps) * rot_reward_scale,
    active only while (lifted & at_center)
  * on |rot_err| <= success_tolerance: a one-shot reach_goal_bonus, then the goal orientation
    RESAMPLES (continuous in-hand reorientation -- the moving goalpost is what prevents camping
    here, exactly as in the original).
Dropping the cube terminates the episode.
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
class PickReposeCubeEnvCfg(DirectRLEnvCfg):
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
            scale=(0.8, 0.8, 0.8),
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

    # ---- in-hand REPOSE staging gate ----
    # the orientation (in-hand manipulation) reward only switches on once the cube has
    # been carried to within this radius of the target point -- i.e. the policy must
    # PICK + CARRY first, then REPOSE in-hand. Kept loose enough to be reachable yet
    # tight enough that reorientation happens AT the goal, not mid-flight.
    inhand_pos_thresh = 0.08  # m

    # ---- PICK + CARRY rewards (anti-camping: progress + latched events, no occupancy farming) ----
    reach_std = 0.2          # grasp-approach tanh width (m); dense, DECAYS to 0 after lift
    w_reach = 1.0            # dense grasp-approach (grasp bootstrap)
    w_lift = 5.0             # ONE-SHOT bonus the first time the cube clears the table
    w_carry = 50.0           # per metre of NEW closest-distance progress (carry ratchet, gated by lift)

    # ---- REPOSE: VERBATIM from xhand_repose / InHandManipulationEnv (the proven ShadowHand/OpenAI
    # in-hand reorientation reward) -- dense hyperbolic orientation reward + instant success bonus +
    # goal RESAMPLE on success (continuous reorientation), but STAGED so it only activates once the
    # cube is lifted AND carried to the goal point (lifted & at_center). ----
    rot_eps = 0.1            # softening in rot_rew = 1/(|rot_err| + rot_eps)
    rot_reward_scale = 1.0   # dense orientation-reward weight
    success_tolerance = 0.1  # orientation tolerance (rad) to count a success
    reach_goal_bonus = 250.0 # one-shot bonus on hitting the goal orientation, then the goal resamples
    w_action_rate = -1e-4
    w_joint_vel = -1e-4
    debug_ratchet_asserts = False  # set True to assert the carry-ratchet buffers reset correctly

    # termination: cube fell this far below its rest height
    drop_height = 0.10

    # debug markers: palm-center (red sphere), fingertips (green spheres), palm-normal ray (blue).
    # only built when a GUI is present, so headless training pays nothing.
    debug_markers = True

    # floating goal-pose marker (a cube drawn at the target pose)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=_CUBE_USD, scale=(0.8, 0.8, 0.8))},
    )
