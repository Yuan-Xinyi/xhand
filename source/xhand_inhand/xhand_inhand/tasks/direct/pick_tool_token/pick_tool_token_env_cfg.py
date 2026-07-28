# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config for robust xArm7 + XHand hammer pickup.

The policy controls seven arm deltas, nine CrossDex tokens and five independent distal-joint
residuals.  The latter restore hand degrees of freedom that the DexPilot retargeter fixes near
their midpoints.  Reward and observations expose a staged close-to-wrap bridge, while grasp,
lift and success remain gated by real object contact and true mesh clearance.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import XARM7_XHAND_CFG

from ..pick_cube_token.pick_cube_token_env_cfg import PickCubeTokenEnvCfg
from .tool_asset import TOOL_MASS, TOOL_REST_QUAT, TOOL_REST_Z, TOOL_SCALE, TOOL_USD


@configclass
class PickToolTokenEnvCfg(PickCubeTokenEnvCfg):
    # End-to-end DAgger rollouts reach strict 20 cm success at a median of ~630 control steps
    # (observed range extends past step 900).  A 10 s / 500-step horizon therefore censored most
    # successful trajectories before PPO could observe the terminal reward.
    episode_length_s = 20.0

    # Reverse-curriculum reset.  Defaults preserve ordinary random reset exactly.  Training phases
    # opt into a physically captured episode boundary (settle_start -> mid_lift -> micro_end ->
    # close_start), then reduce curriculum_reset_probability until all resets are task-randomized.
    curriculum_dataset: str = ""
    curriculum_boundary = "close_start"
    curriculum_reset_probability = 0.0
    curriculum_joint_noise = 0.0

    # Short-horizon reverse-curriculum task used to train a coupled arm+hand close option.  The
    # default end-to-end task is unchanged.  Option mode deliberately has no height reward: it must
    # hold a safe strict latch for 0.30s, while lifting before latch or pushing the object sideways
    # terminates as a failure.  Deployment then switches to a separately validated lift option.
    close_option_mode = False
    close_option_success_bonus = 100.0
    close_option_confirm_steps = 15
    close_option_min_hold_quality = 0.5
    close_option_unlatched_lift_limit = 0.015
    close_option_horizontal_drift_limit = 0.03
    close_option_min_proximity = 0.01
    close_option_lost_window_steps = 12
    close_option_failure_penalty = 100.0
    close_option_timeout_penalty = 20.0

    # ---- nudge-option mode: non-prehensile pre-grasp reorientation ----
    # Short-horizon task: push/nudge the tool ON the table into the graspable pose family
    # (the close_start distribution the close option was trained from), without grasping it.
    # Success = object COM near the target spot, heading near the target yaw (measured in the
    # rest frame), still flat on the table and settled, held nudge_confirm_steps frames.
    # Mutually exclusive with close_option_mode; the default end-to-end task is unchanged.
    nudge_option_mode = False
    # ---- merged nudge+grasp mode: one SAC policy from reorient to a held latch ----
    # From the home spawn, the policy may reorient the tool (nudge shaping guides but does not
    # require it) and must end holding the tool: success = the strict latch contract
    # (is_grasped & grasp_quality>=high & hold_quality>=min & force<=limit) sustained for
    # close_option_confirm_steps.  Failure = table-hit / escape / drop / sustained force; the
    # tip check is deliberately absent (a grasped tool changing attitude is not a failure).
    # Reward = the v6 nudge shaping stack + the battle-tested close/wrap progress + the
    # one-shot first-stable-grasp bonus + the +-100 contract terms.
    nudge_grasp_mode = False
    nudge_grasp_success_bonus = 100.0
    nudge_grasp_failure_penalty = 100.0
    # Timeout must NOT be a refuge.  With timeout at -10 vs failure at -100, the measured
    # equilibrium (v9-v12 merged runs AND stage-2 close v1/v2: fingertips median 6cm from the
    # tool, grasp_quality exactly 0) is a standoff hover: while P(success|engage)~0, never
    # engaging strictly dominates.  With every non-success outcome worth the same -100,
    # engaging becomes a free option and success probability is the only optimizable term.
    nudge_grasp_timeout_penalty = 100.0
    # Occupancy on close_quality: potential-difference closure shaping telescopes away under
    # FlashSAC's return normalization (27k warm-started steps produced zero latches), exactly
    # like the reach potentials before it.  Occupancy pays every step of partial closure;
    # a full-episode maximum (~25) stays well under the +100 latch bonus, so parking in a
    # half-closed touch is never preferable to finishing the grasp.
    nudge_grasp_close_occupancy = 0.05
    # Occupancy on the GATED approach posture (proximity x alignment x opposition x palm):
    # the v6-inherited pushing fist is un-closable, and self-play spawns from those postures
    # produced zero latches (v10, falsified).  Paying for grasp-like approach postures every
    # step steers the approach itself toward states closure can actually start from.
    # Full-episode maximum ~75 at posture 1.0, but sustained posture~1.0 implies a pregrasp
    # that the +100 latch bonus strictly dominates by finishing.
    nudge_grasp_posture_occupancy = 0.15
    # Anti-standoff terms (v11 home rollouts hover centimetres from the tool without touching:
    # with P(latch|engage)~0 the -100 failure risk makes disengagement rational).  A one-shot
    # milestone for reorienting the tool into the pose family restores the v6 engagement
    # incentive.  Escape matches the uniform -100: with timeout at -100, a cheaper escape
    # would make punting the tool out of the workspace the new refuge.
    nudge_grasp_milestone_bonus = 40.0
    nudge_grasp_escape_penalty = 100.0
    # Staging-zone occupancy: the missing rung between the post-nudge push posture and the
    # pregrasp posture ladder.  From a fist beside the tool, "hover the palm ABOVE the tool"
    # is reachable by contact-free local exploration, and the pregrasp/close/wrap terms take
    # over from there: staging (this) -> posture occupancy -> close occupancy -> +100 latch.
    # Paid only while the tool is in the pose family and unlatched; max ~0.08/step keeps a
    # parked hover strictly dominated by finishing the grasp.
    nudge_grasp_staging_occupancy = 0.0
    nudge_grasp_staging_height = 0.12   # palm target height above the tool COM (m)
    nudge_grasp_staging_sigma = 0.06    # position kernel width (m)
    # ---- in-hand reorientation mode: from a lifted stable hold, bring the INDEX fingertip
    # onto the tool's functional point (object-frame constant, annotated via the drag tool;
    # persisted in functional_point.json next to this file).  Success = fingertip-to-point
    # distance <= inhand_dist_threshold (live-ratcheted by the trainer) while still grasped,
    # held inhand_confirm_steps frames.  No tool-attitude requirement (user decision).
    # Failure = the hold is lost (unlatched inhand_lost_hold_steps frames), the tool falls
    # back below inhand_min_clearance, or sustained unsafe force.
    inhand_mode = False
    inhand_point = (0.01936, -0.00635, 0.07932)   # object-frame functional point (m)
    inhand_dist_threshold = 0.015                 # success distance; ratchet target (m)
    inhand_confirm_steps = 15
    inhand_success_bonus = 100.0
    inhand_failure_penalty = 100.0
    inhand_timeout_penalty = 10.0
    inhand_reach_scale = 0.3      # occupancy on exp(-dist/sigma): dense, farm-capped << +100
    inhand_reach_sigma = 0.05
    inhand_min_clearance = 0.10   # true-mesh clearance below this = dropped -> failure (m)
    inhand_lost_hold_steps = 10   # consecutive unlatched frames -> failure
    # Head-down operating standard: the hammer-head axis (object frame; computed from the
    # mesh -- volume center-of-mass leans 2cm toward the head along the principal axis) must
    # point down: cos(head_axis_world, -z) >= inhand_head_cos_min.  -1 disables (v1
    # behavior); the trainer ratchets this toward its target like the distance gate.
    inhand_head_axis = (0.16574, -0.93217, 0.32187)
    inhand_head_cos_min = -1.0
    inhand_head_scale = 0.15      # occupancy on (1+cos)/2 while held: dense rotation gradient
    # ---- carry mode: SimToolReal-style consecutive pose goals; FlashSAC replaces the IK ----
    # From curriculum-spawned fresh-latch states (chain --capture_latched, boundary
    # carry_start), the policy moves the HELD tool to the goal pose already present in the
    # observation (target_pos 63:66 / target_quat 66:70 -- the base-env goal machinery):
    # object COM within carry_pos_tolerance AND orientation within carry_rot_tolerance,
    # latched, held carry_confirm_steps frames.  Reaching a goal pays carry_goal_bonus and
    # RESAMPLES a fresh goal in place (position uniform in the carry_goal_pos_range box
    # around cfg.target_pos; orientation from the cfg target_rot_range_* ranges) -- the
    # episode continues, goals-per-episode is the metric.  Only a drop (latch lost
    # carry_lost_hold_steps frames) or sustained unsafe force terminates.
    carry_mode = False
    carry_goal_pos_range = (0.10, 0.10, 0.08)   # +- box half-extents around target_pos (m)
    carry_pos_tolerance = 0.05                  # ratcheted
    carry_rot_tolerance = 0.40                  # rad, ratcheted
    carry_confirm_steps = 10
    carry_goal_bonus = 50.0
    carry_drop_penalty = 100.0
    carry_pos_scale = 0.1
    carry_pos_sigma = 0.10
    carry_rot_scale = 0.1
    carry_rot_sigma = 0.7
    # Multiplicative pose occupancy: pays only when BOTH errors shrink.  v1's additive terms
    # financed a stall equilibrium -- park at 1cm position with 1.6rad orientation error and
    # collect the position stream while ignoring wrist-unreachable goals.
    carry_pose_scale = 0.4
    # Goal timeout: v1 goals never expired, so an unreachable goal cost nothing to sit out.
    # Now a goal not reached within the window is charged and resampled -- every goal exerts
    # pressure.
    carry_goal_timeout_steps = 150
    carry_goal_timeout_penalty = 10.0
    # Arm-expensive motion cost: mean |arm dof-target delta| / realizable step, arm channels
    # only -- hand motion stays free, encoding the in-hand-first preference.
    carry_arm_motion_penalty = 0.0
    # Relative-orientation goals (0 disables -> v1 absolute rpy sampling): goal orientation =
    # rotation by theta ~ U(0, carry_goal_rel_angle_max) about a random axis applied to the
    # CURRENT object orientation.  The trainer ratchets the max angle past the wrist range so
    # in-hand repositioning becomes the only way to keep earning.
    carry_goal_rel_angle_max = 0.0
    # Regrasp grace: 10 frames treated any transient contact reorganization as a drop; a
    # finger-gait step needs a longer unlatched window before the -100 verdict.
    carry_lost_hold_steps = 25
    # Arm-locked in-hand sub-task: zero the 7 arm action channels so finger gaiting is the
    # ONLY path to an orientation goal (the exploration space collapses onto the hand and
    # there is no wrist detour).  Goal positions then track the current object position --
    # a locked arm cannot translate the tool to a sampled box position.
    carry_lock_arm = False
    # Gradual arm unlock: arm action channels are scaled by this authority factor (0 = the
    # locked sub-task, 1 = full arm).  The trainer anneals it upward so the arm ASSISTS the
    # learned finger gaiting instead of replacing it; combined with the arm motion cost the
    # in-hand-first preference is preserved.
    carry_arm_authority = 1.0
    # Goal positions follow the current object position (small box) instead of the fixed
    # target point -- used with lifted-hold spawns so goals stay reachable at low arm
    # authority and the tool never has to approach the table.
    carry_goal_follow_object = False
    # SimToolReal-aligned smoothness/streak terms:
    # hand joint-velocity penalty at 1/10 the arm weight (their -0.03/-0.003 recipe) -- the
    # hand stays cheap enough to gait but jitter costs; streak mode resets the episode clock
    # on every reach so runs end by failure or stall, not mid-streak.
    carry_hand_vel_penalty = 0.0
    carry_streak_mode = False
    nudge_target_xy = None            # env-local target COM xy; None -> the default spawn xy
    nudge_target_yaw = 0.0            # target heading vs the rest orientation (rad)
    nudge_pos_tolerance = 0.06        # COM xy distance for success (m)
    nudge_yaw_tolerance = 0.20        # |wrapped heading error| for success (rad)
    nudge_confirm_steps = 15          # consecutive settled in-tolerance frames
    nudge_max_obj_speed = 0.10        # settled COM linear speed for success (m/s)
    nudge_workspace_radius = 0.22     # COM xy escape distance from the target -> failure (m)
    nudge_tip_cos_min = 0.90          # rest-up axis dot world-z below this = tipped -> failure
    nudge_on_table_tolerance = 0.01   # |true clearance| above this = airborne/embedded guard (m)
    nudge_success_bonus = 100.0
    nudge_failure_penalty = 100.0
    nudge_timeout_penalty = 10.0
    # ---- grasp-ready ending (v7) ----
    # Joint-space interpolation between oracle pregrasp and post-nudge handoff states is not a
    # bridge (paired arm configs differ by 2.4 rad median / 4.3 rad max -- different solution
    # branches), and the close policy cannot learn the reposition from the push posture (three
    # reward designs all converged to a standoff hover).  So the handoff distribution itself is
    # moved: with nudge_pregrasp_min > 0, nudge success ALSO requires the geometry-only pregrasp
    # readiness score (same definition and gate value as the oracle close_start capture in
    # pick_tool_shared.PREGRASP_GATE) held through the confirm window, so successful nudge
    # episodes END inside the distribution the close policy already latches from.  The occupancy
    # term pays the score every step once the tool is in the pose family, giving a dense path
    # from "pushed into place" to "hand parked grasp-ready" (max ~0.3/step << the +100 success).
    nudge_pregrasp_min = 0.0
    nudge_pregrasp_occupancy = 0.0
    # Staging rung for the nudge/reposition task (same geometry as the nudge_grasp staging):
    # pays for hovering the palm above the posed tool.  Used by the dedicated REPOSITION stage
    # (curriculum-spawned at the post-nudge handoff, success = pregrasp readiness, no grasping):
    # ladder = staging -> pregrasp occupancy -> pregrasp-gated success.
    nudge_staging_occupancy = 0.0
    nudge_progress_scale = 30.0       # potential-based shaping on the pose error
    # Reach is OCCUPANCY-style, not potential-based.  Two pilots measured zero object contact:
    # a gamma-correct potential telescopes to PHI(end)-PHI(start) (~8 return for a full
    # approach), 30x weaker than the occupancy reward the original PPO reach learned from, and
    # FlashSAC's return normalization (scaled by the +-100 terminals) buries it entirely.
    # Occupancy pays every step near the tool: 0.5*coarse + 0.5*gated, capped so a full-episode
    # hover earns ~30 < the +100 success bonus (hovering is never preferable to finishing).
    # The gated layer keeps the battle-tested stack (dual kernels x region x alignment x
    # opposition x palm facing); the ungated coarse layer supplies the far-field gradient.
    nudge_reach_scale = 0.1           # occupancy reward per step at reach potential 1.0
    nudge_reach_coarse_sigma = 0.12   # mean-fingertip-distance scale of the coarse layer (m)
    nudge_touch_bonus = 10.0          # one-shot bonus at first object contact of the episode
    # Hand-table collision is a HARD constraint, not shaping: any finger pad or the palm
    # center dipping below table + margin terminates the episode as a failure (same -100
    # failure contract as tipping/escaping).  No reward term.
    nudge_table_margin = 0.004        # minimum hand height above the table surface (m)
    # Nudge episodes start from a low-ready arm pose instead of the high home pose (three
    # exploration-only runs measured zero object contact from home).  The pose is DLS-solved
    # and its finger-pad clearance is verified in the TRUE reset context -- the hand at the
    # OPEN home posture, whose pads hang ~4.4cm lower than the curled solve-time posture that
    # fooled the first two attempts (85.5%-100% of spawns started with pads through the
    # table).  Result: palm 160mm over the table center, every finger pad >= 47mm above the
    # surface, >= ~30mm under the dedicated spawn noise below (+-0.1 rad reset noise amplified
    # to +-5cm of palm height and re-broke the clearance on its own).  None keeps home.
    nudge_ready_arm_joints = (-0.0392, 0.4001, 0.0703, 1.1947, 0.081, 1.1331, 0.04)
    nudge_spawn_joint_noise = 0.02    # per-joint spawn noise (rad); NOT reset_arm_joint_noise
    # Spawn-pose curriculum (demo-free bridge of the home->object exploration gap): each
    # episode spawns the arm at home + blend*(ready - home) with blend ~ U[min, max].  The
    # trainer anneals blend_min 1.0 -> 0.0 while blend_max stays 1.0, so the mixture always
    # contains easy near-object starts (success keeps flowing) while the boundary marches
    # toward the home pose.  Defaults reproduce the fixed low-ready spawn.
    nudge_spawn_blend_min = 1.0
    nudge_spawn_blend_max = 1.0
    nudge_pos_sigma = 0.08            # xy error scale inside the pose potential (m)
    # Heading enters the pose potential linearly (1 - err/pi): with full-yaw resets an
    # exponential is numerically flat at large errors and provides no gradient.

    # Action = 7 arm relative deltas + 9 CrossDex tokens + 5 absolute distal residuals.  A residual
    # of -1/0/+1 reaches the runtime lower/token/upper target respectively, before the existing EMA.
    enable_distal_residual = True
    distal_residual_joint_names = (
        "thumb_joint2",
        "index_joint2",
        "middle_joint1",
        "ring_joint1",
        "pinky_joint1",
    )
    action_space = 21

    # Action shields enforce the task ordering in the controller as well as in reward.  The tactile
    # limiter follows the validated oracle servo: remove stored hand preload at 25N and actively
    # unload above 30N.  Ten persistent >30N frames terminate a jammed grasp; two >60N frames are
    # the faster emergency cutoff.
    # While touching without a strict latch (or after losing a
    # previously latched grasp), remove only the palm-up component of the arm delta.
    enable_grasp_action_shield = True
    tactile_soft_force_limit = 25.0
    tactile_hard_force_limit = 30.0
    tactile_release_step = 0.02
    hand_target_max_step = 0.02
    tactile_hard_terminate_steps = 10
    tactile_terminate_force_limit = 60.0
    tactile_terminate_steps = 2

    # The old 87 features remain an exact prefix: core70 + arm/token16 + lift1. New features are
    # residual5 plus 23 bounded close/contact/phase/counter/transport features.  In particular,
    # palm-frame linear and angular slip make held-versus-flung transport observable to the actor.
    enable_grasp_observations = True
    observation_space = 115
    state_space = 115

    # robot with CONTACT REPORTING enabled on its bodies (needed for the fingertip contact sensors /
    # R_contact). A fresh .replace() so the shared XARM7_XHAND_CFG is untouched.
    robot_cfg: ArticulationCfg = XARM7_XHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.activate_contact_sensors = True

    # ---- the tool on the table (convex-decomposition collision baked into TOOL_USD) ----
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TOOL_USD,
            scale=TOOL_SCALE,
            # CONTACT REPORTING on the OBJECT too -- force_matrix_w (filtered fingertip<->object contact)
            # needs BOTH the sensor body (robot) AND the filter body (object) to report contacts. Without
            # this it silently reads 0 at multi-env scale (worked at num_envs=1, dead at >1) -> is_grasped
            # was always False in training -> the grasp/lift reward NEVER fired (the whole "won't lift" bug).
            activate_contact_sensors=True,
            # Mass is authored by MeshConverter on the rigid root before instancing. Applying it
            # here would target an instanceable geometry child and may be ignored by USD/PhysX.
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
        # Projected from the hand-calibrated points to the real central handle cross-section.
        # The old points were 1.0--3.7mm inside the mesh and rewarded impossible penetration.
        (-0.006245, 0.040144, 0.100644),
        (-0.020187, 0.015094, 0.092488),
        (-0.012841, 0.059730, 0.078441),
        (-0.027973, 0.024908, 0.073244),
    )

    # Analytic handle frame. The four calibrated keypoints lie on one handle cross-section; its
    # centroid is the center and the plane normal is the handle axis. At startup the environment
    # takes a thin mesh slice in this frame and builds its convex 2-D surface polygon, yielding a
    # dense handle distance and a real outward normal without a 44k-vertex per-step nearest search.
    handle_center = (-0.017000, 0.034950, 0.085900)
    handle_axis = (0.821886, -0.288917, -0.490949)
    handle_axial_min = -0.020        # m; negative side reaches the hammer head beyond roughly -25mm
    handle_axial_max = 0.025         # m; central 4.5cm graspable band around the calibrated section
    handle_axial_margin = 0.002      # m; contact ROI tolerance, kept separate from radial surface margin
    handle_section_half_width = 0.002  # m; mesh slice used to construct the cross-section polygon
    handle_contact_margin = 0.008    # m; sanity gate in addition to object-filtered contact force

    # ===================== DIRECTIONAL keypoints (calibration step) =====================
    # Each object keypoint and each finger pad gets a DIRECTION (unit vector). A later reward will
    # constrain the finger-pad normal to OPPOSE the object-keypoint normal (a proper opposing grasp),
    # which makes the palm-facing gate unnecessary. This step only ADDS + VISUALIZES the directions
    # (palm-normal-style bead trail + a draggable TIP bead per direction); the reward is unchanged.
    #
    # grasp_keypoint_dirs: outward surface normal at each object keypoint, in the OBJECT LOCAL frame.
    #   Set to None -> auto-seeded to radial-outward-from-centroid so the first render is non-degenerate;
    #   drag the magenta TIP beads in the GUI, then paste the printed tuple back here to freeze them.
    grasp_keypoint_dirs = None
    # finger_pad_normals: the direction each finger PAD faces, in that distal link's LOCAL frame.
    #   Drag the cyan TIP beads in the GUI; the printed dict can be pasted back here.
    finger_pad_normals = {
        # thumb pad normal FROZEN to the value that makes it point exactly opposite the palm normal at
        # the SETTLED home pose. Found offline (tools/diag): the hand takes ~30 steps to settle, so a
        # value read too early is ~40 deg off; this is the step-45+ steady value (cos(thumb,-palm)=1).
        "thumb_rota_link2": (0.2035, 0.4378, 0.8757),
        "index_rota_link2": (0.0, -1.0, 0.0),
        "mid_link2": (0.0, -1.0, 0.0),
        "ring_link2": (0.0, -1.0, 0.0),
        "pinky_link2": (0.0, -1.0, 0.0),
    }
    # keep False -> use the frozen thumb vector above. Set True only to RE-derive it live (auto-compute
    # -palm at the settled pose); needs the long settle countdown in the env (~30+ steps).
    thumb_normal_opposite_palm = False
    # bead-trail visualization (like the palm-normal trail): length in metres and bead count.
    dir_viz_len = 0.05
    dir_viz_beads = 6

    # ================= reward params (stage-1 robust grasp state machine) =================
    # One grasp quality is shared by confirmation, hold, lift and success.  It requires filtered
    # object contact at the handle from thumb + two non-thumb pads, opposed contact sides, palm/pad
    # orientation and low palm/object rigid-body slip.  A Schmitt latch debounces this same quality.
    expected_object_mass = TOOL_MASS
    object_mass_tolerance = 1.0e-4
    contact_force_thr = 0.2       # N; contact detected above this fingertip<->object force magnitude
                                  # (0.5 filtered out the tentative first grazes of a few tenths of a N).
    contact_force_saturation = 5.0  # N; force contribution is tanh(F/F_sat), preventing crush farming
    grasp_confirm_steps = 4       # consecutive high-quality steps to latch (+fire one-shot grasp bonus)
    grasp_release_steps = 6       # consecutive below-low-quality steps to release (>confirm = hysteresis)
    # a valid grasp must be PALM-SIDE and OPPOSED, not a back-of-hand press. palm_facing = 0.5(1+palm·to_obj)
    # in [0,1]; >0.5 means the palm faces the object. align in [0,1] = thumb+2 nearest pads oppose the handle
    # normals. Without these, a dorsal contact (palm_facing ~0.2-0.35, 460N press) satisfied is_grasped but
    # could not lift (the object slipped out). Gating is_grasped on them forces a liftable palm-side grip.
    grasp_palm_facing_min = 0.5   # palm must at least face toward the object (raw palm·to_obj > 0)
    grasp_align_min = 0.3         # thumb+2 pads at least mildly opposed to the handle (a floor, not strict)
    grasp_opposition_min = 0.5    # thumb and the two strongest other contacts must lie on opposing handle sides
    # Calibrated with the scripted gravity-bearing oracle: real fingertip grasps that transport
    # the 0.15kg hammer through 20cm settle at q=0.37--0.44.  The old 0.45 rejected those physical
    # holds even though all topology/orientation/transport components were present.
    grasp_quality_high = 0.35     # confirm only above this shared wrap*transport quality
    grasp_quality_low = 0.20      # release below this threshold (Schmitt dead-band in between)

    # R_grasp: one-shot bonus on the first stable grasp (never re-paid after drop/regrasp).
    grasp_bonus = 100.0
    grasp_bonus_max_force = 30.0  # N; oracle peaks near 21N, while crush-launch spikes are 285--460N

    # R_success is paid once when the stable 20cm true-clearance termination is reached.
    success_bonus = 500.0

    # HELD gate: hold_quality = exp(-rel_lin/hold_v_scale) * exp(-rel_ang/hold_w_scale), where
    # rel_lin/rel_ang are the object's linear/angular speed RELATIVE to the palm. A HELD object moves with
    # the hand (rel ~0 -> hold_quality ~1); a FLUNG object flies off (rel large -> hold_quality -> 0), and it
    # also breaks valid_contact. Together: a throw/fling earns ZERO lift reward.
    hold_v_scale = 0.3    # m/s; relative linear speed that drops hold_quality to ~0.72 (0.1 m/s) / e^-1 (0.3)
    hold_w_scale = 3.0    # rad/s; relative angular speed scale

    # Stage-4 exploration bridge.  Its potential has explicit near -> thumb -> thumb+one ->
    # thumb+two tiers. None of these terms can enable the grasp latch, lift or success.
    close_proximity_scale_far = 0.08
    close_proximity_scale_near = 0.02
    shaping_discount = 0.99
    close_progress_scale = 40.0
    wrap_progress_scale = 80.0

    # Once latched, signed true-clearance progress makes sustained upward arm motion discoverable;
    # there is deliberately no held-height occupancy annuity.
    lift_progress_scale = 400.0

    # A real oracle carry peaks around 21N.  Penalize only excess fingertip-object force so normal
    # closure is free while the historical 285--460N crush/launch solutions are uneconomic.
    safe_contact_force = 20.0
    contact_force_penalty_width = 20.0
    force_excess_penalty_scale = 5.0
    distal_residual_penalty_scale = 0.02

    # The AABB remains only for reset collision bounds and diagnostic comparison. Reward/success use
    # the real mesh convex-hull minimum, never this loose box.
    object_aabb_min = (-0.0986, -0.0878, 0.0034)
    object_aabb_max = (0.0955, 0.0884, 0.1667)

    # Success must remain a slow, safe, strict grasp for this many steps.
    success_hold_steps = 15
    success_max_obj_lin_speed = 0.20   # m/s
    success_max_obj_ang_speed = 3.0    # rad/s

    # True minimum-mesh-point clearance above the measured table surface.
    lift_success_height = 0.20

    # Retained only for press_diag/armhit_diag geometry reports. Arm-table contact is not a task
    # termination; robust object contact and transport gates make table-reaction hacks unprofitable.
    terminate_on_arm_table_contact = False
    arm_contact_bodies = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8")
    arm_table_margin = 0.03    # m; arm link center within this of the TRUE table surface = pressing it
                               # (grasp keeps arm links >= ~+0.047 above origin; true table ~ -0.003 -> safe)

    # draggable grasp-keypoint markers (yellow spheres on the object) -- GUI-only, for visually
    # checking / repositioning the handle keypoints. Costs nothing headless (gated on has_gui()).
    debug_grasp_marker = True

    # floating goal-pose marker drawn with the tool mesh (same as the cube task's marker)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=TOOL_USD, scale=TOOL_SCALE)},
    )
