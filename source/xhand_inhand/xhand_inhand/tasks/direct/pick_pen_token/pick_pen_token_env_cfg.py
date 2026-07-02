# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow config: xArm7 + XHand pick-a-pen with CrossDex action tokenization.

Same action pipeline as ``pick_cube_token`` (7 arm joint deltas + a 9-dim eigengrasp
token retargeted to 12 absolute xhand joint targets), but the object is the pen and the
task adds an ORIENTATION goal on top of lifting:

  1. grasp the pen off the table
  2. lift it ``lift_success_height`` (10 cm) off the table
  3. re-orient it TIP DOWN: the pen's local +Z (big end) points toward the table normal
     (world +Z) within ``upright_success_angle`` (~20 deg) -> the small end (the tip)
     points straight down
  4. the tip is UNOCCLUDED by the fingers: the tip point is the lowest point of the
     hand+pen assembly, at least ``tip_clearance_margin`` below every finger pad / palm.

Action layout (16 = 7 + 9):
  * [0:7]  arm joint deltas   (relative position control, same as pick_cube_token)
  * [7:16] hand eigengrasp token in [-1, 1] -> absolute hand joint targets

Pen geometry (from ``assets/pen``): long axis is the object-local Z. The +Z end is the
"big" end (spans to +0.058 m), the -Z end is the long tapering end -> the writing TIP
(spans to -0.082 m). So "tip down" == "local +Z up".
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import XARM7_XHAND_CFG

from ..pick_cube_token.pick_cube_token_env_cfg import PickCubeTokenEnvCfg

_PEN_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "pen", "pen.usd")
)


@configclass
class PickPenTokenEnvCfg(PickCubeTokenEnvCfg):
    # ---- action (unchanged from pick_cube_token): 7 arm deltas + 9 hand eigengrasp token ----
    action_space = 16
    n_hand_tokens = 9
    # obs = joint_pos(19)+joint_vel(19)+ee_pos_b(15)+palm_center_b(3)+object_pos_b(3)
    #       +object_quat(4)+pen_up_axis_w(3)+tip_pos_b(3)+target_pos(3)+target_quat(4)+actions(16) = 92
    observation_space = 92

    # grasp + lift + re-orient needs more time than a bare lift
    episode_length_s = 6.0

    # Physics hardening for the thin pen (SimToolReal pen tunneling fix): physics at 200 Hz
    # with decimation 4 keeps the control rate at 50 Hz (unchanged) while giving the solver
    # 2x the substeps to resolve the stiff hand-vs-pen contact. A high position-iteration
    # floor + high pen depenetration velocity + a contact shell (below) stop press-through.
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

    # robot: bump the hand-side solver iterations so the pen-vs-finger contact resolves hard
    # on both bodies (a deep copy via .replace(); the shared XARM7_XHAND_CFG is untouched).
    robot_cfg: ArticulationCfg = XARM7_XHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = 32
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 2

    # ---- pen on the table (lying flat along +Y at identity+reset yaw) ----
    # rot (0.7071, -0.7071, 0, 0) = -90 deg about X -> the pen's local +Z (long axis) lies
    # horizontal. z=0.018 is the FATTER pen ORIGIN's settled rest height when lying flat
    # (measured in-sim at 1.8x cross-section; origin is off the geometric center).
    # Spawning at the rest height makes `actual_lift` = true height above the table, so
    # `lift_success_height` (10 cm) means exactly "lifted 10 cm off the table".
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PEN_USD,
            # FATTER pen: non-uniform scale fattens the cross-section (long axis is local Z, so
            # x/y = diameter, z = length). 1.8x -> diameter 1.9 cm -> ~3.4 cm, length unchanged
            # (still ~14 cm, aspect ~4:1). A larger diameter ~doubles the antipodal grasp window
            # and improves roll stability, moving the grasp from near-impossible to learnable.
            scale=(1.8, 1.8, 1.0),
            # HEAVIER pen: 0.10 kg (was 0.023 kg baked in the USD). Enough mass to sit still when
            # brushed and damp the dynamics (light pens fly away / tunnel); still within the weak
            # XHand fingers' holding force.
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                # eject overlapping shapes FAST so a pressed pen cannot creep through a finger
                max_depenetration_velocity=10.0,
                disable_gravity=False,
            ),
            # contact "shell": contacts engage 15 mm out and the pen rests 2 mm outside the
            # finger surface -> a hard standoff against slow press-through.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.015, rest_offset=0.002),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.018), rot=(0.7071, -0.7071, 0.0, 0.0)),
    )

    # pen reset randomization on the table: xy noise + full yaw about world Z (stays flat)
    reset_object_pos_noise = (0.08, 0.15)
    reset_object_yaw_range = (-3.14159, 3.14159)
    reset_min_hand_object_dist = 0.050

    # ---- pen geometry (object-local frame) ----
    # local axis pointing to the BIG end; aligning it with world +Z stands the pen tip-down.
    # (Flip the sign if big/small ends come out swapped in the GUI.)
    pen_up_axis = (0.0, 0.0, 1.0)
    # the writing TIP (small end) in the pen's local frame
    pen_tip_offset = (0.0, 0.0, -0.0819)

    # ---- success thresholds ----
    lift_success_height = 0.10        # required lift for success (10 cm above the table rest)
    success_lift_band = 0.02          # accept lift within this band below the target as "at height"
    upright_success_angle = math.radians(20.0)  # pen long axis within this of the table normal
    tip_clearance_margin = 0.02       # tip must be this far BELOW every finger pad / palm
    success_hold_steps = 60           # success pose must persist this many steps (CrossDex: 60 train)

    # ---- CrossDex / UniDexGrasp++ style DENSE reward ----
    # r = r_dis + r_height + r_xy + r_orient + r_tip + r_success  (see the env docstring). Dense
    # distance terms pull the hand onto the pen every step (no ratchet cap to run out, no
    # hover-and-farm optimum); height/orient/tip are GATED on the hand being on the pen.
    w_ft_dist = 2.0                   # fingertip->object mean-distance penalty weight (2x palm)
    w_palm_dist = 1.0                 # palm->object distance penalty weight
    ft_close_thr = 0.10               # "hand on pen" gate: avg fingertip-to-pen distance below this
    palm_close_thr = 0.15             # "hand on pen" gate: palm-to-pen distance below this
    w_height = 10.0                   # gated lift ramp (0 -> 10 as pen rises to lift_success_height)
    w_xy = 0.3                        # horizontal pen-displacement penalty (anti-slide)
    orient_gate_lift = 0.03           # orient/tip rewards only once lifted at least this much
    w_orient = 10.0                   # gated tip-down reward (relu(cos_tilt) * this)
    w_tip = 2.0                       # gated tip-below-hand clearance reward
    success_bonus = 200.0             # sparse bonus when the success pose is held (CrossDex: 200)

    # goal marker: a floating pen standing tip-down (identity orientation = local +Z up)
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"goal": sim_utils.UsdFileCfg(usd_path=_PEN_USD, scale=(1.8, 1.8, 1.0))},
    )
