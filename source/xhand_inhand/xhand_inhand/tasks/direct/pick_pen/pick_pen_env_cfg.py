# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: FR3 + XHand pick a pen up and lift it to a FIXED point.

Faithful port of IsaacLab's `Isaac-Lift-Cube-Franka-v0` (the manager-based lift
task that actually trains) to the Direct workflow, with our FR3+XHand and the
whiteboard pen.  The single design decision that makes that task learnable and
that our earlier contact-gated tasks got wrong:

  * the lift / goal-tracking rewards are gated on the object's HEIGHT, not on a
    contact-sensor grasp signal.  Height is trivially discoverable (any upward
    nudge of the pen -> instant +lift reward), and lifting a pen off a table is
    only possible by actually grasping it -- so the height gate *induces* the
    grasp without ever needing to detect one.

Compared to the franka reference (which we mirror term-for-term):
  * robot is the 19-DOF FR3+XHand under full RELATIVE joint position control
  * the "end effector" is the hand's grasp assembly (palm-center + 5 fingertips)
  * the pen is elongated -> reach distance is measured to the pen's axis segment
  * the goal is a single FIXED point in the air (no orientation target)
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
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "bb_pen", "bb_pen.usd")
)


@configclass
class PickPenEnvCfg(DirectRLEnvCfg):
    # env
    # NOTE: physics at 200 Hz (dt=1/200) with decimation=4 keeps the control rate
    # at 50 Hz (unchanged from the old 100 Hz / decimation=2), so the trained
    # control timing is preserved while the solver gets twice as many steps to
    # resolve the stiff hand-vs-pen contact -> far less chance of tunneling.
    decimation = 4
    episode_length_s = 5.0
    action_space = 19  # full relative joint control: 7 arm + 12 hand
    # obs = joint_pos(19)+joint_vel(19)+ee_pos_b(5*3=15)+palm_center_b(3)
    #       +object_pos_b(3)+object_quat(4)+target_pos_b(3)+actions(19) = 85
    observation_space = 85
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
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
    # .replace() returns a deep copy, so bumping the solver iterations below only
    # affects pick-pen (the shared FR3_XHAND_CFG is untouched). More position
    # iterations let the solver actually push the pressed pen back out instead of
    # letting it tunnel through the hand; a couple of velocity iterations damp the
    # post-contact velocity spikes we measured (up to ~2.9 m/s on a 23 g pen).
    robot_cfg: ArticulationCfg = FR3_XHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = 16
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
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
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

    # ---- goal: lift the pen to a single FIXED point in the air (env-local) ----
    target_pos = (0.5, 0.0, 0.35)

    # ---- reach geometry (pen is elongated -> measure distance to its AXIS segment) ----
    pen_long_axis = (0.0, 0.0, 1.0)  # the pen's length direction in its local frame
    pen_half_length = 0.07  # half the pen length (m); pen treated as a line segment

    # ---- lift detection (mirror of franka `object_is_lifted`) ----
    # the pen counts as "lifted" once its center rises this far above its table rest
    lift_margin = 0.04

    # ---- reward weights (mirror of the franka lift task) ----
    reach_std = 0.2  # reaching tanh width (m)
    goal_track_std = 0.3  # coarse goal-tracking tanh width (m)
    goal_track_fine_std = 0.05  # fine goal-tracking tanh width (m)
    w_reach = 1.0
    w_lift = 15.0
    w_goal_track = 16.0
    w_goal_track_fine = 5.0
    w_action_rate = -1e-4
    w_joint_vel = -1e-4

    # termination: pen fell this far below its rest height
    drop_height = 0.10

    # floating goal marker (a small sphere at the fixed target point)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            )
        },
    )
