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
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import XARM7_XHAND_CFG

from ..pick_cube_token.pick_cube_token_env_cfg import PickCubeTokenEnvCfg
from .tool_asset import TOOL_MASS, TOOL_REST_QUAT, TOOL_REST_Z, TOOL_SCALE, TOOL_USD


@configclass
class PickToolTokenEnvCfg(PickCubeTokenEnvCfg):
    # action (16 = 7 arm deltas + 9 hand eigengrasp token) and observation (86) are inherited
    # unchanged from pick_cube_token.

    # the bulky, irregular tool takes a little longer to line the grasp up than a small cube
    episode_length_s = 6.0

    # ===== JOINT-SPACE ARM control (mvp26 reward, joint-space) =====
    # Action = 7 (arm relative joint deltas) + 9 (hand eigengrasp token) = 16, mapped by the inherited
    # PickCubeTokenEnv._pre_physics_step (arm: dof_target += action_scale*a; hand: token -> retarget NN).
    action_space = 16

    observation_space = 87      # 86 base (19 joint_pos + 19 joint_vel + 15 ee + 3 palm + 3 obj_pos + 4
    state_space = 87            # obj_quat + 3 tgt_pos + 4 tgt_quat + 16 action) + 1 lift-progress feature

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

    # R_success: ONE-SHOT bonus when the stable lift first succeeds. Needed because success ENDS the episode
    # (occupancy lift would otherwise pay ~20/step forever at 19cm), so without it the policy is punished for
    # crossing the 20cm line and stalls just below it. 500 ~= 25 steps of max lift reward (a rough
    # continuation-value compensation for the truncated episode).
    success_bonus = 500.0

    # R_lift = lift_scale * clip(clearance/lift_success_height,0,1) * valid_contact * hold_quality, per step.
    # lift_scale=20 => at 20cm success height ~20/step; at 10cm ~10/step. Dominates R_reach (max 2) once up.
    lift_scale = 20.0

    # HELD-gate on R_lift: hold_quality = exp(-rel_lin/hold_v_scale) * exp(-rel_ang/hold_w_scale), where
    # rel_lin/rel_ang are the object's linear/angular speed RELATIVE to the palm. A HELD object moves with
    # the hand (rel ~0 -> hold_quality ~1); a FLUNG object flies off (rel large -> hold_quality -> 0), and it
    # also breaks valid_contact. Together: a throw/fling earns ZERO lift reward.
    hold_v_scale = 0.3    # m/s; relative linear speed that drops hold_quality to ~0.72 (0.1 m/s) / e^-1 (0.3)
    hold_w_scale = 3.0    # rad/s; relative angular speed scale

    # R_hold: quality-weighted floor for a confirmed grasp.  It reaches 3 only at/above the confirmation
    # threshold (so a legal hold beats max reach=2), and fades to zero across the latch dead-band so a weak
    # or slipping latched grasp cannot farm the full reward.
    grasp_hold_scale = 3.0

    # R_reach: directional pre-grasp occupancy kernel (mvp20 kernel), gated to (~is_grasped) so it turns OFF
    # once grasped (no occupancy annuity after the grasp). coarse (reach_scale_far) pulls the arm in from
    # ~15cm, fine (reach_scale) sharpens the final placement; both x palm_facing (whole-hand orient) x align.
    reach_reward_scale = 2.0
    reach_scale = 0.08        # fine kernel (last ~8cm placement)
    reach_scale_far = 0.25    # coarse kernel (approach slope from the ~15cm reset distance)

    # P0-4: table-clearance. The object's AABB lowest corner must rise this far above its at-rest
    # low point before the lift counts -- blocks "tip the hammer up on one end" (root Z rises while
    # the object still rests on the table). AABB from the mesh bbox (tool_asset geometry).
    clearance_margin = 0.03
    object_aabb_min = (-0.0986, -0.0878, 0.0034)
    object_aabb_max = (0.0955, 0.0884, 0.1667)

    # P0-3: success must be a stable grasp held this many steps, with the object nearly still (not
    # flung/tossed through the 30 cm plane for a single frame).
    success_hold_steps = 15
    success_max_obj_lin_speed = 0.20   # m/s
    success_max_obj_ang_speed = 3.0    # rad/s

    # lift success height is now measured RELATIVE to the grasp height (P0-1): 20 cm above where the
    # hand actually closed on the handle -- a solid lift-off, achievable for this hammer.
    lift_success_height = 0.20

    # CLEAN-LIFT gate (user: reward only a straight-up lift that keeps the grasp pose; penalize the arm
    # "wiggle" that tumbles/drags the object). r_lift is multiplied by orient_stay (cos of the twist from
    # the grasp orientation) x xy_stay (1 - tanh(horizontal_drift/scale)). Success also requires the twist
    # to stay small.
    lift_xy_drift_scale = 0.05     # m; horizontal drift at which xy_stay ~ 0.76 (5cm) -> lift reward decays
    success_orient_min = 0.85      # cos(twist) required for success (~0.85 = <=31 deg from grasp orientation)

    # ANTI-HACK: terminate the episode if the ARM presses on the table (the "press-up" hack levers the
    # object up off the table's reaction force instead of a real arm lift). Detected GEOMETRICALLY (no
    # contact sensor): any arm link's world z dropping to within arm_table_margin of the table plane
    # (= the object's at-rest lowest corner) means the arm reached the table. The arm links never
    # legitimately go that low -- only the fingertips reach down to the object -- so this is a clean signal.
    terminate_on_arm_table_contact = True
    arm_contact_bodies = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8")
    arm_table_margin = 0.03    # m; arm link center within this of the TRUE table surface = pressing it
                               # (grasp keeps arm links >= ~+0.047 above origin; true table ~ -0.003 -> safe)

    # BIG lift pay (mvp20, user call): the mvp19 plateau (~324) was a "gentle-hold annuity" -- holding
    # pays a steady 5.0/step while lifting pays one-time ratchet increments with slip risk, so the
    # policy held without lifting (peak lift 8mm). 5x the dense scale: +50 per NEW cm of lift while
    # contact-grasping (full 20cm = 1000 + 300 bonus, dwarfing the ~600 hold annuity). STILL strictly
    # contact-gated (x contact_grasp; no contact -> no lift pay, regression T1) and ratcheted (only new
    # highs pay -- jiggling can't farm it).
    dense_lift_rew_scale = 5000.0

    # draggable grasp-keypoint markers (yellow spheres on the object) -- GUI-only, for visually
    # checking / repositioning the handle keypoints. Costs nothing headless (gated on has_gui()).
    debug_grasp_marker = True

    # floating goal-pose marker drawn with the tool mesh (same as the cube task's marker)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=TOOL_USD, scale=TOOL_SCALE)},
    )
