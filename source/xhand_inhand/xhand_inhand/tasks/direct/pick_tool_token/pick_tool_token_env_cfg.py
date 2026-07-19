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
from .tool_asset import TOOL_REST_QUAT, TOOL_REST_Z, TOOL_SCALE, TOOL_USD


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

    observation_space = 89      # 86 base (19 joint_pos + 19 joint_vel + 15 ee + 3 palm + 3 obj_pos + 4
    state_space = 89            # obj_quat + 3 tgt_pos + 4 tgt_quat + 16 action) + 3 phase (is_grasped etc.)

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

    # ================= reward params (mvp25: hysteresis grasp state machine) =================
    # A "valid contact" = thumb pad contacts the object AND >=1 other fingertip pad contacts it (real
    # contact sensors, force > contact_force_thr). is_grasped is a HYSTERESIS latch over valid_contact:
    # ON after grasp_confirm_steps of consecutive valid contact, OFF after grasp_release_steps of loss.
    # This debounces flickering contact so R_reach switches off cleanly and success is stable.
    contact_force_thr = 0.2       # N; contact detected above this fingertip<->object force magnitude
                                  # (0.5 filtered out the tentative first grazes of a few tenths of a N).
    contact_near_margin = 0.09    # m; PER-FINGER gate: a fingertip's NET force counts as OBJECT contact only
                                  # if THAT fingertip is within this of its nearest handle keypoint. Replaces
                                  # the dead filtered force_matrix_w (multi-env broken) AND the too-loose palm
                                  # gate (stayed True through a crush-launch); now contact drops the instant a
                                  # finger leaves the object, so an object squirting out of the grip isn't credited.
    grasp_confirm_steps = 4       # consecutive valid-contact steps to LATCH is_grasped True
    grasp_release_steps = 6       # consecutive lost-contact steps to release is_grasped (>confirm = hysteresis)

    # R_grasp (mvp26): ONE-SHOT bonus on the first stable grasp of the episode (latched by grasp_bonus_given,
    # never re-paid even after drop+regrasp). Kept MODEST (< the lift payout) so "grasp" matters but never
    # outweighs "actually lift". Ladder: grasp 100 < hold 10cm ~1000 < hold 20cm ~2000 (+200 success).
    grasp_bonus = 100.0

    # R_lift (mvp26): per-step NORMALIZED occupancy height (NOT a ratchet). r_lift = lift_step_max *
    # clip(grasp_rel_lift/lift_success_height,0,1) * is_grasped. lift_step_max=20 => at 20cm it pays 20/step;
    # holding there ~100 steps ~= 2000, so a STABLY HELD lift dominates. A crush-launch pays only while the
    # object is airborne (then falls back to 0) -> can't be farmed like the mvp20 ratchet's transient peak.
    lift_step_max = 20.0
    lift_success_bonus = 200.0

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

    # ANTI-HACK: terminate the episode if the ARM presses on the table (the "press-up" hack levers the
    # object up off the table's reaction force instead of a real arm lift). Detected GEOMETRICALLY (no
    # contact sensor): any arm link's world z dropping to within arm_table_margin of the table plane
    # (= the object's at-rest lowest corner) means the arm reached the table. The arm links never
    # legitimately go that low -- only the fingertips reach down to the object -- so this is a clean signal.
    terminate_on_arm_table_contact = True
    arm_contact_bodies = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8")
    arm_table_margin = 0.04    # m; an arm link center this close (or below) the table plane = pressing it

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
