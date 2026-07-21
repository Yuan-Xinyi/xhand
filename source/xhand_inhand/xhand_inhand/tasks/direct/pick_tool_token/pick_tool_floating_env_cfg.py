"""Arm-free PickTool ablation using a dynamically actuated floating XHand."""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import FLOATING_XHAND_CFG

from .pick_tool_token_env_cfg import PickToolTokenEnvCfg


@configclass
class PickToolFloatingXHandEnvCfg(PickToolTokenEnvCfg):
    """Replace xArm7 by bounded linear/angular velocity control of the free hand root.

    The 20 second horizon is intentionally unchanged: historical successful
    grasp/lift trajectories frequently appeared after 500 control steps.
    """

    episode_length_s = 20.0
    action_space = 20  # bounded palm twist 6 + token 9 + distal residual 5
    observation_space = 100
    state_space = 100

    robot_cfg: ArticulationCfg = FLOATING_XHAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    robot_cfg.spawn.activate_contact_sensors = True

    arm_action_dim = 6
    arm_joint_names = []
    arm_contact_bodies = ("palm",)

    floating_max_linear_speed = 0.25
    floating_max_angular_speed = 1.0
    floating_workspace_min = (0.05, -0.40, 0.08)
    floating_workspace_max = (0.80, 0.40, 0.80)
    reset_arm_joint_noise = 0.0
    enable_grasp_action_shield = False
