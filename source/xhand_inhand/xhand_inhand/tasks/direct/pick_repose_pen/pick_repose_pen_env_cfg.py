# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: FR3 + XHand pick a pen up, carry it to a fixed point,
AND stand it upright with the BIG end up / small end down (its big-end axis within
30 deg of the table normal).

This extends the lift-only `pick_pen` task (a faithful Direct port of IsaacLab's
`Isaac-Lift-Cube-Franka-v0`) with an ORIENTATION goal:

  * lift the pen off the table          (height-gated bootstrap, +w_lift)
  * carry it to a FIXED target point     (position tracking, gated by lifted)
  * point the pen's BIG end toward the   (orientation tracking, gated by lifted)
    table normal (world +Z) within 30 deg -- i.e. stand it upright, big end up,
    small end down. This is DIRECTED (the big-end axis must point up, not down).

Height-gating stays the bootstrap (any upward nudge -> lift reward, and a pen can
only leave the table by being grasped), so the grasp is induced without a contact
sensor; the orientation term then shapes the held pen toward upright.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from xhand_inhand.robots import FR3_XHAND_CFG

_PEN_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "pen", "pen.usd")
)


@configclass
class PickReposePenEnvCfg(DirectRLEnvCfg):
    # env
    # NOTE: physics at 200 Hz (dt=1/200) with decimation=4 keeps the control rate
    # at 50 Hz (unchanged from the old 100 Hz / decimation=2), so the trained
    # control timing is preserved while the solver gets twice as many steps to
    # resolve the stiff hand-vs-pen contact -> far less chance of tunneling.
    decimation = 4
    episode_length_s = 5.0
    action_space = 19  # full relative joint control: 7 arm + 12 hand
    # obs = joint_pos(19)+joint_vel(19)+ee_pos_b(5*3=15)+palm_center_b(3)
    #       +object_pos_b(3)+object_quat(4)+pen_big_axis_w(3)+goal_axis(3)+target_pos_b(3)+actions(19) = 91
    observation_space = 91
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            # NOTE: full sweep-based CCD is unsupported on the GPU pipeline. The GPU cure
            # for the slow press-through is to resolve the contact harder (high iteration
            # floor below + high depenetration velocity on the pen) and a wide contact
            # shell (pen contact_offset/rest_offset in object_cfg). This floor forces every
            # body to run >=32 position iterations so the pressed contact never leaks.
            min_position_iteration_count=32,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot (no contact sensor needed -- the lift reward is height-gated)
    # .replace() returns a deep copy, so bumping the solver iterations below only
    # affects this task (the shared FR3_XHAND_CFG is untouched). More position
    # iterations let the solver actually push the pressed pen back out instead of
    # letting it tunnel through the hand; a couple of velocity iterations damp the
    # post-contact velocity spikes we measured (up to ~2.9 m/s on a 23 g pen).
    robot_cfg: ArticulationCfg = FR3_XHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # 32 position iterations on the HAND side of the contact too (was 16): both bodies in
    # the pen-vs-finger contact must resolve hard, or the soft side leaks penetration.
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = 32
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 2
    palm_body_name = "palm"
    # the 5 fingertips form the "end effector" (grasp assembly) for the reach reward
    ee_body_names = ["index_rota_link2", "mid_link2", "ring_link2", "pinky_link2", "thumb_rota_link2"]
    # palm-center point in the PALM BODY frame (the XHand "palm" origin is at the wrist;
    # offset toward the fingers (+Z) and the palm side (-Y) to get the grasp center).
    palm_center_offset = (0.0, -0.02, 0.07)

    # pen on the table (lying flat)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PEN_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # resolve the hand-vs-pen contact HARD every substep so a pressed pen
                # cannot accumulate penetration and slowly creep through a finger.
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                # push overlapping shapes APART fast (HIGH, not low): the cure for the slow
                # press-through is to eject the pen quicker than the finger can sink into it.
                max_depenetration_velocity=10.0,
                disable_gravity=False,
            ),
            # contact "shell": contacts engage 15 mm out (contact_offset) and the pen is
            # held 2 mm OUTSIDE the finger surface (rest_offset) -> a hard standoff the pen
            # must overcome ~15 mm of penetration to even begin to pass through a finger.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.015, rest_offset=0.002),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05), rot=(0.7071, -0.7071, 0.0, 0.0)),
    )

    table_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    table_pos = (0.5, 0.0, 0.0)
    table_rot = (0.707, 0.0, 0.0, 0.707)

    # action: full RELATIVE joint position control (target += action_scale * action)
    action_scale = 0.1

    # reset randomization of the pen on the table (x/y position + full yaw)
    reset_object_pos_noise = (0.10, 0.20)
    reset_object_yaw_range = (-3.14159, 3.14159)

    # ---- goal: lift the pen to a FIXED point + match a RANDOMIZED upright target ----
    target_pos = (0.5, 0.0, 0.35)
    # cone center for sampling the target big-end direction: the table normal (world +Z).
    table_normal = (0.0, 0.0, 1.0)
    # the pen's local long axis that points toward its BIG end (small end is -this).
    # TUNE the sign if big/small ends come out swapped in the GUI.
    pen_big_end_axis = (0.0, 0.0, 1.0)
    # each episode samples a target big-end direction UNIFORMLY within this cone of +Z.
    # 30 deg -> the pen must end up "roughly upright, big end up", but the exact target
    # tilt/azimuth is randomized per episode (and shown by the reference pen marker).
    goal_cone_angle = 0.5236

    # ---- lift / grasp latch (z_lifted = rest + lift_margin) ----
    # the pen counts as "grasped" (I_grasped latches true for the rest of the episode)
    # once its center first rises this far above its table rest -> r_goal takes over.
    lift_margin = 0.04

    # ===================== SOTA reward (progress ratchets + keypoint pose) =====================
    # r = r_smooth + r_grasp + I_grasped * r_goal      (faithful port of the reference paper)
    #
    # r_smooth  : -lambda * L1(joint_vel), split arm / hand
    # r_grasp   : r_approach + (1 - I_grasped) * r_lift
    #   r_approach = lambda_approach * max(min_ft_dist_so_far - mean_fingertip_obj_dist, 0)
    #   r_lift     = lambda_lift * max(z - z_init, 0) + B_lifted (once, when first lifted)
    # r_goal    : max(d* - d, 0) + B_success * I[d < eps]    (d = keypoint pose distance)
    #             d* = min keypoint distance so far for the CURRENT goal (reset on resample)
    #
    # The dense terms are PROGRESS ratchets (reward only improvement over the best so far),
    # so the agent cannot farm reward by hovering near / holding a pose statically.

    # smoothness: L1 of joint velocities, separate arm / hand weights
    lambda_arm_smooth = 0.0005
    lambda_hand_smooth = 0.0005

    # grasp shaping (active before the pen is lifted)
    lambda_approach = 10.0  # fingertip -> pen approach progress ratchet
    lambda_lift = 10.0  # dense height ramp max(z - z_init, 0)
    bonus_lifted = 5.0  # one-time bonus when the pen first crosses z_lifted

    # goal-pose reaching (active after grasped); keypoint pose distance d(o, g)
    lambda_goal = 40.0  # progress ratchet weight on max(d* - d, 0)
    bonus_success = 50.0  # sparse success bonus when d < success_eps
    success_eps = 0.04  # keypoint distance (m) counted as "reached the goal pose"
    resample_goal_on_success = True  # sample a fresh orientation goal after each success

    # 4 object-frame keypoints at (+-sx/2, +-sy/2, +-sz/2). The large scale is on the pen's
    # LONG axis (local z) so the metric is sensitive to TILT of the long axis and nearly
    # free to ROLL about it -- exactly "big-end direction matters, spin about it doesn't".
    keypoint_scales = (0.03, 0.03, 0.14)  # (x, y, z); z is the pen's length direction

    # termination: pen fell this far below its rest height
    drop_height = 0.10

    # floating goal marker: a (tilted, big-end-up) pen at the target showing the goal
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=_PEN_USD, scale=(1.0, 1.0, 1.0))},
    )
