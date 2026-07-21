"""xArm7 + parallel-gripper PickTool ablation configuration."""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from xhand_inhand.robots import XARM7_GRIPPER_CFG

from .pick_tool_token_env_cfg import PickToolTokenEnvCfg


@configclass
class PickToolGripperEnvCfg(PickToolTokenEnvCfg):
    episode_length_s = 20.0
    action_space = 8  # arm7 + one symmetric jaw command
    observation_space = 54
    state_space = 54

    robot_cfg: ArticulationCfg = XARM7_GRIPPER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    robot_cfg.spawn.activate_contact_sensors = True

    palm_body_name = "palm"
    ee_body_names = ["thumb_rota_link2", "index_rota_link2"]
    palm_center_offset = (0.0, 0.0, 0.09)
    finger_pad_offsets = {
        "thumb_rota_link2": (-0.007, 0.0, 0.045),
        "index_rota_link2": (0.007, 0.0, 0.045),
    }
    arm_joint_names = ["joint[1-7]"]
    hand_joint_names = [".*_finger_joint"]

    # The hammer remains at the baseline 0.5 scene friction.  Only the rubber
    # jaw pads get the same kind of friction advantage as XHand fingertips.
    fingertip_friction = 2.0
    gripper_open_width = 0.045
    gripper_closed_width = 0.012
    gripper_contact_force_thr = 0.25
    gripper_force_saturation = 6.0
    gripper_safe_force = 60.0
    gripper_terminate_force = 100.0
    gripper_confirm_steps = 4
    gripper_release_steps = 6
    gripper_quality_high = 0.25
    gripper_quality_low = 0.12
    gripper_proximity_scale = 0.08

    # Potential shaping uses the same magnitudes as the XHand task.
    gripper_close_progress_scale = 40.0
    gripper_contact_progress_scale = 80.0
    # A contact-progress burst must never pay more than an unsafe impact costs.
    force_excess_penalty_scale = 100.0
