# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: xArm7 + XHand claw-hammer nail pulling / hammering.

Implements ``claw_hammer_experiment_plan.md``: the hammer starts IN the hand (MVP-1
"tool already in hand"), a guided nail sits in a fixture on the table, and the policy
must use the hammer's two natural affordances:

  * PULL phase  -- align the CLAW slot with the nail head and pull the nail up 3 cm
                   (prismatic joint with controlled friction = the plan's guided pin).
  * PRESS phase -- (MVP-3, ``enable_press_phase``) reorient the hammer IN HAND so the
                   hammer FACE points down, then press the nail back near its original
                   depth. Quasi-static pressing, not dynamic impact.

The arm is CONSTRAINED (plan section 6): palm translation beyond ~2 cm and wrist
rotation beyond ~20 deg from the episode-start pose are penalized, so affordance
switching must come from tool-in-hand reorientation, not arm flipping.

Action pipeline is the CrossDex tokenization inherited from ``pick_cube_token``:
7 arm joint deltas + a 9-dim eigengrasp token -> 12 absolute xhand joint targets.

CALIBRATION NOTES (in-GUI, like the finger-pad calibration of pick_cube):
  * ``hammer_in_palm_pos/quat`` and ``reset_hand_grasp_pos`` are reasonable initial
    guesses. Verify in the GUI that the reset places the handle across the palm without
    interpenetration, and tune these fields.
  * ``fixture_pos`` should put the nail head under the home-pose hand within arm reach.

HAMMER LOCAL FRAME (must mirror ``gen_hammer_nail_assets.py``):
  origin = head center; face plane at (+0.047, 0, 0) with normal +X; claw slot center at
  (-0.045, 0, 0) opening toward -X; handle along -Z with grip center at (0, 0, -0.10);
  claw pull direction = local -Z (handle up => prong plane horizontal under the head).
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import XARM7_XHAND_CFG

from ..pick_cube_token.pick_cube_token_env_cfg import PickCubeTokenEnvCfg

_ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "hammer_nail")
)
_HAMMER_USD = os.path.join(_ASSET_DIR, "hammer.usd")
_NAIL_FIXTURE_USD = os.path.join(_ASSET_DIR, "nail_fixture.usd")


@configclass
class HammerNailTokenEnvCfg(PickCubeTokenEnvCfg):
    """MVP-1: hammer already in hand -> align claw -> pull the nail 3 cm."""

    # ---- action (inherited pipeline): 7 arm deltas + 9 hand eigengrasp token ----
    action_space = 16
    n_hand_tokens = 9
    # obs = joint_pos(19)+joint_vel(19)+ee_pos_b(15)+palm_center_b(3)
    #       +object_pos_b(3)+object_quat(4)+claw_slot_b(3)+face_center_b(3)
    #       +claw_pull_dir_w(3)+face_normal_w(3)+nail_engage_b(3)+nail_q(1)+nail_qd(1)
    #       +phase(1)+actions(16) = 97
    observation_space = 97

    episode_length_s = 8.0

    # phase switch: False = MVP-1 (pull only), True = MVP-3 (pull -> in-hand switch -> press)
    enable_press_phase = False

    # ---- physics hardening (same recipe as pick_pen_token: thin prongs / nail shaft) ----
    decimation = 4
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            min_position_iteration_count=32,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=2.5, replicate_physics=True)

    # close-up default camera on env_0's fixture (used by --video recordings)
    viewer: ViewerCfg = ViewerCfg(eye=(0.95, -0.55, 0.55), lookat=(0.44, -0.04, 0.20))

    robot_cfg: ArticulationCfg = XARM7_XHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = 32
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 2

    # the hammer starts in the hand: no arm-home randomization (would break in-palm placement)
    reset_arm_joint_noise = 0.0

    # TASK-SPECIFIC arm home pose (FK-searched, scripts pattern of find_home_pose.py):
    # palm CENTER hovers at (0.44, -0.04, 0.265) with the palm facing DOWN, directly above
    # the nail fixture -- the constrained arm (2 cm translation) can never approach a far
    # fixture, so the nominal pose must start AT the nail. Hand joints stay open (reset
    # overwrites them with the pre-shaped grasp).
    robot_cfg.init_state.joint_pos = {
        "joint1": 0.0772,
        "joint2": 0.1590,
        "joint3": -0.3346,
        "joint4": 0.4637,
        "joint5": -0.3870,
        "joint6": -1.0874,
        "joint7": 0.1836,
        "(thumb|index|middle|ring|pinky)_joint.*": 0.0,
    }

    # ---- hammer (self.object; init pose is irrelevant, reset places it in the palm) ----
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_HAMMER_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # thinner contact shell than the pen (2 mm standoff would visibly float the
            # 8 mm prongs); 8 mm engage distance still guards against press-through.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.008, rest_offset=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.4), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # ---- nail fixture (fixed-base articulation, 1 prismatic dof "nail_joint") ----
    # zero-stiffness actuator = passive joint; friction is the plan's "controlled
    # resistance" (must exceed nail weight ~0.7 N so the pulled nail stays up).
    nail_joint_friction = 2.0
    nail_joint_damping = 10.0
    nail_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/NailFixture",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_NAIL_FIXTURE_USD,
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.44, -0.04, 0.0),  # == fixture_pos below (keep in sync)
            joint_pos={"nail_joint": 0.0},
        ),
        actuators={
            "nail": ImplicitActuatorCfg(
                joint_names_expr=["nail_joint"],
                stiffness=0.0,
                damping=nail_joint_damping,
                friction=nail_joint_friction,
            ),
        },
    )
    fixture_pos = (0.44, -0.04, 0.0)  # env-local, directly under the home-pose palm center
    fixture_block_top = 0.14          # nail head underside height (q = 0), fixture-local
    nail_head_thickness = 0.008
    nail_travel = 0.06                # prismatic upper limit authored in the USD

    # ---- hammer functional geometry (local frame; mirror gen_hammer_nail_assets.py) ----
    claw_slot_local = (-0.045, 0.0, 0.0)
    claw_pull_dir_local = (0.0, 0.0, -1.0)  # must align with world +Z when pulling
    face_center_local = (0.047, 0.0, 0.0)
    face_normal_local = (1.0, 0.0, 0.0)     # must align with world -Z when pressing
    # grip near the MIDDLE of the hammer (closer to the head than the handle center):
    # less head-gravity torque on the grasp and room on both sides for in-hand rolling
    grip_center_local = (0.0, 0.0, -0.065)

    # ---- in-hand reset placement (palm-frame; CALIBRATE in GUI) ----
    # palm frame: fingers ~ +Z, palm surface ~ -Y. Handle axis laid along palm +X
    # (quat = +90 deg about palm Y maps hammer local +Z -> palm +X), grip point at the
    # palm center; head ends up 10 cm along palm +X (thumb side).
    hammer_in_palm_pos = (0.065, -0.035, 0.07)
    hammer_in_palm_quat = (0.7071068, 0.0, 0.7071068, 0.0)  # (w, x, y, z)
    # pre-shaped power grasp around the 24 mm handle (CALIBRATE: no collision check here)
    reset_hand_grasp_pos = {
        "thumb_joint0": 1.40,
        "thumb_joint1": 0.30,
        "thumb_joint2": 0.40,
        "index_joint0": 0.00,
        "index_joint1": 1.20,
        "index_joint2": 0.90,
        "middle_joint0": 1.20,
        "middle_joint1": 0.90,
        "ring_joint0": 1.20,
        "ring_joint1": 0.90,
        "pinky_joint0": 1.20,
        "pinky_joint1": 0.90,
    }
    # The nail starts PROTRUDING: its head underside sits this far above the block top, so
    # the 10 mm-thick claw prongs can slide UNDER the head (flush = no purchase, unpullable).
    nail_init_protrusion = 0.015
    # nail reset depth randomization (plan sec. 11, "initial nail depth"), uniform [0, x]
    # ADDED to the protrusion
    reset_nail_depth_noise = 0.005

    # ---- task thresholds (displacements RELATIVE to the episode-start nail depth q0) ----
    pull_target = 0.03                # plan: pull the nail out by 3 cm (from its start depth)
    press_success_depth = 0.008       # plan: back within 0.5-1 cm of the start depth
    # HOOKED-under-the-head engagement (v2 lesson: a plain radius around the head base was
    # satisfied by RESTING THE CLAW ON TOP of the head -- dist bottomed out at exactly
    # head thickness + prong half-thickness ~2.2 cm and the nail was pressed DOWN, never
    # pulled). The claw counts as engaged only when the slot center sits ON THE SHAFT
    # BELOW THE HEAD: within a small horizontal radius of the shaft axis AND vertically
    # inside a band just under the head underside.
    claw_hook_offset = 0.007          # dense-reward target: this far BELOW the head base
    claw_engage_radius_xy = 0.012     # slot center horizontal distance to the shaft axis
    claw_engage_dz = 0.008            # slot center vertical tolerance around the hook point
    # 50 deg: the v1 policy plateaued at ~47 deg mean tilt right below a 40 deg gate and
    # farmed the align reward instead -- the gate is reward bookkeeping, actual pulling
    # still requires the claw to physically hook the head.
    claw_align_cos = math.cos(math.radians(50.0))   # claw pull dir vs world +Z
    face_engage_radius = 0.025        # face center within this of the nail head top
    face_align_cos = math.cos(math.radians(30.0))   # -face normal vs world +Z
    success_hold_steps = 50           # success state must persist (1 s at 50 Hz control)

    # ---- constrained arm (plan sec. 6): soft limits w.r.t. the episode-start palm pose ----
    arm_translation_limit = 0.02      # m, free palm translation before penalty
    wrist_rotation_limit = math.radians(20.0)  # rad, free palm rotation before penalty
    w_arm_translation = 20.0          # penalty per step per m beyond the translation limit
    w_wrist_rotation = 2.0            # penalty per step per rad beyond the rotation limit

    # ---- tool-in-hand constraints ----
    slip_free_dist = 0.015            # grip-to-palm-center drift allowed before slip penalty
    w_slip = 10.0
    tool_drop_dist = 0.15             # grip point this far from palm center = dropped -> done
    drop_penalty = 50.0

    # ---- PULL phase reward ----
    w_claw_dist = 2.0                 # dense: -w * ||claw_slot - nail_head_base||
    w_claw_align = 3.0                # gated: relu(cos(claw pull dir, +Z)) when claw is near
    claw_near_dist = 0.06             # gate distance for the alignment term
    w_pull = 300.0                    # ratcheted nail-up progress (only while claw engaged)
    w_wrong_pull = 100.0              # nail-up progress WITHOUT claw engagement (penalty)
    pull_bonus = 300.0                # one-shot bonus at nail_disp >= pull_target (claw engaged)
    # Success-state HOLD stream: paid every step the instantaneous success state holds.
    # Must beat the claw-align hover income (~2/step), or the policy farms alignment next
    # to the nail forever instead of pulling (the v1 failure mode).
    w_success_hold = 5.0

    # ---- PRESS phase reward (enable_press_phase) ----
    w_face_dist = 2.0                 # dense: -w * ||face_center - nail_head_top||
    w_face_align = 3.0                # gated: relu(cos(-face normal, +Z)) when face is near
    face_near_dist = 0.08             # gate distance for the alignment term
    w_press = 300.0                   # ratcheted nail-down progress (only while face engaged)
    w_wrong_press = 100.0             # nail-down progress WITHOUT face engagement (penalty)
    success_bonus = 200.0             # full-sequence success (held success_hold_steps)
    # Success does NOT end the episode: terminating on success cuts off the dense reward
    # stream, making success worth LESS return than hovering next to the nail (v1 failure
    # mode). The success-state hold stream keeps paying instead.
    terminate_on_success = False

    # ---- action regularization (inherited fields, re-tuned) ----
    kuka_actions_penalty_scale = 0.03
    hand_actions_penalty_scale = 0.003

    # goal marker: small sphere at the 3 cm pulled nail-head height
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            )
        },
    )
    # target_pos is only used to place the goal marker
    # (fixture + block top + nail_init_protrusion + pull_target)
    target_pos = (0.44, -0.04, 0.185)


@configclass
class HammerNailFullTokenEnvCfg(HammerNailTokenEnvCfg):
    """MVP-3: pull 3 cm -> in-hand claw->face switch -> press the nail back in."""

    enable_press_phase = True
    episode_length_s = 12.0
