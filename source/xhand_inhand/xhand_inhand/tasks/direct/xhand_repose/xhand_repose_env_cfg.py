# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the XHand in-hand cube re-orientation task (OpenAI-LSTM variant).

Adapted from IsaacLab's ``shadow_hand_env_cfg.py``. Plugs the XHand right hand
into the shared ``InHandManipulationEnv``. Only hand-specific fields and the
observation/action dimensions differ from the Shadow Hand version.

Everything tagged ``# TUNE`` is geometry/physics that must be dialed in
visually (palm orientation, grasp pose, cube placement/scale, joint gains).
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from xhand_inhand.robots import XHAND_RIGHT_CFG

##
# XHand-specific constants (derived from the URDF / USD articulation).
##

# all 12 joints are independently actuated (no coupling/tendons)
XHAND_ACTUATED_JOINTS = [
    "thumb_joint0",
    "thumb_joint1",
    "thumb_joint2",
    "index_joint0",
    "index_joint1",
    "index_joint2",
    "middle_joint0",
    "middle_joint1",
    "ring_joint0",
    "ring_joint1",
    "pinky_joint0",
    "pinky_joint1",
]
# distal link of each finger (used for fingertip observations)
XHAND_FINGERTIPS = [
    "index_rota_link2",
    "mid_link2",
    "ring_link2",
    "pinky_link2",
    "thumb_rota_link2",
]


@configclass
class EventCfg:
    """Domain-randomization config (tendon randomization dropped — XHand has none)."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class XHandReposeEnvCfg(DirectRLEnvCfg):
    """Full-observation feed-forward base config (12-DOF XHand)."""

    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 12  # 12 actuated joints (Shadow: 20)
    # full obs = 2*12 (dof pos+vel) + 13 (obj) + 11 (goal) + 5*(3+4+6) (fingertips) + 12 (actions) = 125
    observation_space = 125
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    # robot --------------------------------------------------------------- TUNE
    # Like the original Shadow Hand setup, the hand is mounted on a fixed base
    # (root link anchored in the air, gravity disabled) so it stays put while
    # manipulating the cube. There is no separate "stand" mesh in the original
    # either — the wrist root is simply fixed at `pos`.
    #
    # init pose: palm should face up so the cube can rest in the hand.
    # rot is (w, x, y, z). XHand fingers extend along +Z from the palm root.
    # Adjust rot/pos and the default grasp `joint_pos` in the GUI until the
    # cube sits stably in the palm.
    robot_cfg: ArticulationCfg = XHAND_RIGHT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        # fixed base + no gravity on the hand (matches Shadow Hand)
        spawn=XHAND_RIGHT_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,  # anchor the wrist root (the "stand")
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # TUNE: mounting height of the fixed base
            rot=(0.7071, -0.7071, 0, 0),  # TUNE: orient palm-up
            joint_pos={".*": 0.0},  # TUNE: a slightly-closed grasp posture
            joint_vel={".*": 0.0},
        ),
        # TUNE: position-control gains for manipulation (defaults from viz cfg)
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=10.0,
                velocity_limit_sim=3.14,
                stiffness=3.0,
                damping=0.1,
            ),
        },
    )
    actuated_joint_names = XHAND_ACTUATED_JOINTS
    fingertip_body_names = XHAND_FINGERTIPS

    # in-hand object ------------------------------------------------------ TUNE
    # XHand is smaller than the Shadow Hand, so the dex cube is scaled down and
    # placed above the palm. Tune `scale` and `pos` together with the hand pose.
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.75, 0.75, 0.75),  # dex_cube 0.08 m -> 0.06 m edge
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            semantic_tags=[("class", "cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.1, 0.65), rot=(1.0, 0.0, 0.0, 0.0)),  # TUNE
    )
    # goal object (visualization marker only)
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.75, 0.75, 0.75),  # match object scale (0.06 m edge)
            )
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=0.75, replicate_physics=True, clone_in_fabric=True
    )

    # reset
    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24  # TUNE: cube "dropped" distance, scale to hand size
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


@configclass
class XHandReposeOpenAIEnvCfg(XHandReposeEnvCfg):
    """OpenAI-LSTM variant: asymmetric (reduced policy obs + full critic state)."""

    # env
    decimation = 3
    episode_length_s = 8.0
    action_space = 12
    # reduced obs = 5*3 (fingertip pos) + 3 (obj pos) + 4 (rel quat) + 12 (actions) = 34
    observation_space = 34
    # full state = full obs (125) + 5*6 (fingertip forces) = 155
    state_space = 155
    asymmetric_obs = True
    obs_type = "openai"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # reset
    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24  # TUNE
    vel_obs_scale = 0.2
    success_tolerance = 0.4
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0
    # domain randomization
    events: EventCfg = EventCfg()
    # per-step gaussian action noise + per-reset bias
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # per-step gaussian observation noise + per-reset bias
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
