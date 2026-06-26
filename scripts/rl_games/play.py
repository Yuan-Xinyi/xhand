# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--external_cube_pose_npy",
    type=str,
    default=None,
    help="Path to a 4x4 robot_T_cube .npy pose. The pose is injected into env 0 after reset.",
)
parser.add_argument(
    "--foundationpose_cube",
    action="store_true",
    help="Use D405 + FoundationPose to initialize env 0 cube pose.",
)
parser.add_argument(
    "--camera_to_robot_tf",
    type=str,
    default=None,
    help="Optional path to a 4x4 robot_T_camera .npy transform for FoundationPose. If omitted, camera_T_cube is used as robot_T_cube.",
)
parser.add_argument(
    "--external_cube_track",
    action="store_true",
    help="Continuously overwrite the Isaac cube pose from the external provider. Debug only: this pins the cube.",
)
parser.add_argument("--foundationpose_width", type=int, default=640)
parser.add_argument("--foundationpose_height", type=int, default=480)
parser.add_argument("--foundationpose_fps", type=int, default=30)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import math
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# register this project's gym environments (e.g. Xhand-Repose-Cube-*)
import xhand_inhand.tasks  # noqa: F401, E402

# PLACEHOLDER: Extension template (do not remove this comment)


def _rotmat_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to Isaac/torch quaternion order (w, x, y, z)."""
    m = np.asarray(rotmat, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    return q / np.linalg.norm(q)


class NpyCubePoseProvider:
    def __init__(self, path: str):
        pose = np.load(path).astype(np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"{path} must contain a 4x4 robot_T_cube matrix, got {pose.shape}")
        self.pose = pose

    def get_pose(self) -> np.ndarray:
        return self.pose

    def close(self):
        pass


class FoundationPoseCubeProvider:
    def __init__(self, camera_to_robot_tf: str, width: int, height: int, fps: int):
        if camera_to_robot_tf is None:
            self.robot_T_camera = np.eye(4, dtype=np.float32)
            print(
                "[FoundationPose] --camera_to_robot_tf not provided; using camera_T_cube directly as robot_T_cube. "
                "This follows live_demo.py pose extraction but is not hand-eye calibrated."
            )
        else:
            self.robot_T_camera = np.load(camera_to_robot_tf).astype(np.float32)
            if self.robot_T_camera.shape != (4, 4):
                raise ValueError(f"{camera_to_robot_tf} must contain a 4x4 robot_T_camera matrix")

        foundationpose_root = os.environ.get("FOUNDATIONPOSE_ROOT", "/home/lqin/disk2/FoundationPose")
        cube_dir = os.path.join(foundationpose_root, "cube")
        if cube_dir not in sys.path:
            sys.path.insert(0, cube_dir)

        from live_demo import build_estimator, select_mask
        import pyrealsense2 as rs

        self._rs = rs
        self._select_mask = select_mask
        mesh_file = os.path.join(cube_dir, "mesh", "textured.obj")
        self._est, _, _, _, _ = build_estimator(mesh_file)
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self._pipeline.start(config)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self._align = rs.align(rs.stream.color)
        for _ in range(15):
            self._pipeline.wait_for_frames()
        color, depth, K = self._grab()
        mask = self._select_mask(color)
        if mask is None:
            raise RuntimeError("FoundationPose init mask was cancelled.")
        self._est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)

    def _grab(self):
        fr = self._align.process(self._pipeline.wait_for_frames())
        color_frame = fr.get_color_frame()
        depth_frame = fr.get_depth_frame()
        color = np.asarray(color_frame.get_data())
        depth = np.asarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        return color, depth, K

    def get_pose(self) -> np.ndarray:
        color, depth, K = self._grab()
        camera_T_cube = self._est.track_one(rgb=color, depth=depth, K=K, iteration=2).astype(np.float32)
        return self.robot_T_camera @ camera_T_cube

    def close(self):
        self._pipeline.stop()


def _make_external_cube_provider():
    if args_cli.external_cube_pose_npy is not None:
        return NpyCubePoseProvider(args_cli.external_cube_pose_npy)
    if args_cli.foundationpose_cube:
        return FoundationPoseCubeProvider(
            args_cli.camera_to_robot_tf,
            args_cli.foundationpose_width,
            args_cli.foundationpose_height,
            args_cli.foundationpose_fps,
        )
    return None


def _apply_robot_frame_cube_pose(base_env, robot_T_cube: np.ndarray):
    """Write env-0 cube pose from robot-base frame into Isaac world frame."""
    if base_env.num_envs != 1:
        raise ValueError("External cube pose injection currently requires --num_envs 1")
    device = base_env.device
    root_pos = base_env.robot.data.root_pos_w[0]
    pos_robot = torch.tensor(robot_T_cube[:3, 3], dtype=torch.float32, device=device)
    pos_w = root_pos + pos_robot
    quat_w = torch.tensor(_rotmat_to_quat_wxyz(robot_T_cube[:3, :3]), dtype=torch.float32, device=device)
    pose = torch.cat([pos_w, quat_w], dim=0).unsqueeze(0)
    env_ids = torch.tensor([0], dtype=torch.long, device=device)
    base_env.object.write_root_pose_to_sim(pose, env_ids)
    base_env.object.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device), env_ids)
    base_env._compute_intermediate_values()


def _refresh_obs_from_base(env_wrapper):
    obs_dict = env_wrapper.unwrapped._get_observations()
    processed = env_wrapper._process_obs(obs_dict)
    return processed["obs"] if isinstance(processed, dict) else processed


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # for visualization, render every env: fabric clones only render env_0, so disable it here
    if hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # the SAPG-fork A2CBase parses policy_idx = int(experiment_name.split('_')[0]); use the run
    # dir name (already "0_<timestamp>") as the experiment name so it starts with an int.
    run_name = os.path.basename(log_dir)
    if not str(run_name).split("_")[0].isdigit():
        run_name = "0_" + str(run_name)
    agent_cfg["params"]["config"]["full_experiment_name"] = run_name

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    cube_pose_provider = _make_external_cube_provider()
    if cube_pose_provider is not None:
        robot_T_cube = cube_pose_provider.get_pose()
        _apply_robot_frame_cube_pose(env.unwrapped, robot_T_cube)
        obs = _refresh_obs_from_base(env)
        t = robot_T_cube[:3, 3]
        print(f"[external-cube] initialized env-0 cube pose from robot frame: {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f} m")
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if cube_pose_provider is not None and args_cli.external_cube_track:
                robot_T_cube = cube_pose_provider.get_pose()
                _apply_robot_frame_cube_pose(env.unwrapped, robot_T_cube)
                obs = _refresh_obs_from_base(env)
                if isinstance(obs, dict):
                    obs = obs["obs"]
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if cube_pose_provider is not None:
        cube_pose_provider.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
