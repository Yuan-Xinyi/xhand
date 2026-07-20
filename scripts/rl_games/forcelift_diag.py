# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Force-lift test: does a ROBUST policy grasp hold the object against gravity?

Phase 1 (grip): run the policy until the shared robust grasp latch confirms (freeze the token it was
commanding at that moment). Phase 2 (forced lift): keep the frozen hand token (hold the grip) and
drive the ARM joints back toward their home/reset pose (the high configuration the arm descended
from), a bounded delta per step. Log palm rise vs TRUE mesh clearance. If phase 1 never forms the
robust latch, phase 2 is not run: freezing an arbitrary last token is not a valid lift oracle."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Pick-Tool-Token-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--grip_steps", type=int, default=130)
parser.add_argument("--lift_steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import xhand_inhand.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    # Keep the diagnostic episode alive through the complete two-phase maneuver.  Success and drop
    # are still measured below, but must not auto-reset the state before the readout is captured.
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 30.0)
    env_cfg.terminate_on_drop = False
    env_cfg.success_hold_steps = args_cli.grip_steps + args_cli.lift_steps + 1
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["full_experiment_name"] = "0_forcelift"
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", 5.0)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", 1.0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, agent_cfg["params"]["config"]["device"], clip_obs, clip_actions, None, True)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = 1
    runner = Runner(); runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player(); agent.restore(args_cli.checkpoint); agent.reset()

    u = env.unwrapped
    n_arm = u._n_arm
    arm_ids = u._arm_ids_t
    action_scale = u.cfg.action_scale

    obs = env.reset()
    if isinstance(obs, dict): obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn: agent.init_rnn()

    origin_z = u.scene.env_origins[0, 2]
    home_arm = u.dof_targets[0, arm_ids].clone()  # arm joint targets at reset (high pose)
    def readout():
        object_force = u._finger_object_force_magnitudes()[0]
        net_force = u._finger_net_force_magnitudes()[0]
        signals = u._compute_grasp_signals()
        return {
            "palm_z": float((u.palm_center_w[0, 2] - origin_z).item()),
            "clearance": float((u._object_true_min_z()[0] - u._table_surface_z).item()),
            "latch": bool(u._is_grasped[0].item()),
            "object_force": float(object_force.sum().item()),
            "net_force": float(net_force.sum().item()),
            "q_wrap": float(signals["quality"][0].item()),
            "q_grasp": float(signals["grasp_quality"][0].item()),
            "q_hold": float(signals["hold_quality"][0].item()),
        }

    # ---- phase 1: let the policy grip; freeze the hand token once a grasp forms ----
    frozen_hand = None
    latch_step = None
    for t in range(args_cli.grip_steps):
        with torch.inference_mode():
            ob = agent.obs_to_torch(obs)
            act = agent.get_action(ob, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(act)
        state = readout()
        if state["latch"] and frozen_hand is None:
            frozen_hand = act[0, n_arm:].clone()
            latch_step = t
            print(
                f"[robust latch @ grip step {t}] q_wrap={state['q_wrap']:.3f} "
                f"q_grasp={state['q_grasp']:.3f} q_hold={state['q_hold']:.3f} "
                f"object_force={state['object_force']:.1f}N"
            )
    if frozen_hand is None:
        state = readout()
        print(
            f"\n[grip end] true_clearance={state['clearance']:+.4f} latch=0 "
            f"q_wrap={state['q_wrap']:.3f} q_grasp={state['q_grasp']:.3f} "
            f"q_hold={state['q_hold']:.3f} object_force={state['object_force']:.1f}N "
            f"net_force(audit)={state['net_force']:.1f}N"
        )
        print(
            "VERDICT: ORACLE PRECONDITION FAILED -> policy never formed the robust grasp latch; "
            "forced lift was NOT run and no arbitrary token was frozen."
        )
        env.close()
        return

    state = readout()
    palm_start = state["palm_z"]
    clearance_start = state["clearance"]
    print(
        f"\n[grip end; latch first seen at step {latch_step}] true_clearance={clearance_start:+.4f} "
        f"latch={int(state['latch'])} q_wrap={state['q_wrap']:.3f} "
        f"q_grasp={state['q_grasp']:.3f} q_hold={state['q_hold']:.3f} "
        f"object_force={state['object_force']:.1f}N"
    )

    # ---- phase 2: hold the frozen grip, drive arm joints toward home (up) ----
    print("\n---- FORCED LIFT (hand grip frozen, arm driven to home pose) ----")
    print("step  hand_rise  true_clear  clear_gain  gap    latch  q_wrap  q_grasp  q_hold  objF")
    peak_clearance = clearance_start
    peak_clearance_gain = 0.0
    peak_hand_rise = 0.0
    latch_steps = 0
    for t in range(args_cli.lift_steps):
        cur_arm = u.dof_targets[0, arm_ids]
        arm_delta = torch.clamp((home_arm - cur_arm) / action_scale, -1.0, 1.0)
        act = torch.zeros((1, n_arm + frozen_hand.shape[0]), device=u.device)
        act[0, :n_arm] = arm_delta
        act[0, n_arm:] = frozen_hand
        with torch.inference_mode():
            obs, _, dones, _ = env.step(act)
        if isinstance(obs, dict): obs = obs["obs"]
        state = readout()
        hand_rise = state["palm_z"] - palm_start
        clearance_gain = state["clearance"] - clearance_start
        peak_clearance = max(peak_clearance, state["clearance"])
        peak_clearance_gain = max(peak_clearance_gain, clearance_gain)
        peak_hand_rise = max(peak_hand_rise, hand_rise)
        latch_steps += int(state["latch"])
        if t % 10 == 0:
            print(
                f"{t:4d}  {hand_rise:+.3f}      {state['clearance']:+.4f}    "
                f"{clearance_gain:+.3f}     {hand_rise-clearance_gain:+.3f}  "
                f"{int(state['latch'])}      {state['q_wrap']:.3f}   {state['q_grasp']:.3f}   "
                f"{state['q_hold']:.3f}  {state['object_force']:5.1f}"
            )
        if bool(dones[0]) if not torch.is_tensor(dones) else bool(dones.flatten()[0]):
            print(f"  [unexpected episode termination at lift step {t}]")
            break

    print("\n---- verdict ----")
    print(
        f"peak hand rise={peak_hand_rise:+.3f} m; peak TRUE clearance={peak_clearance:+.4f} m; "
        f"peak clearance gain={peak_clearance_gain:+.3f} m; "
        f"robust latch fraction during lift={latch_steps / max(args_cli.lift_steps, 1):.3f}"
    )
    hand_rise = state["palm_z"] - palm_start
    clearance_gain = state["clearance"] - clearance_start
    print(
        f"final: hand_rise={hand_rise:+.3f} true_clearance={state['clearance']:+.4f} "
        f"clearance_gain={clearance_gain:+.3f} gap={hand_rise-clearance_gain:+.3f} "
        f"latch={int(state['latch'])} q_grasp={state['q_grasp']:.3f}"
    )
    if peak_clearance >= 0.20:
        print("TARGET: reached 20cm TRUE mesh clearance.")
    else:
        print("TARGET: did not reach 20cm TRUE mesh clearance.")
    if peak_hand_rise > 0.05 and peak_clearance_gain > 0.05 and (peak_hand_rise - peak_clearance_gain) < 0.05:
        print("VERDICT: GRIP HOLDS -> true clearance follows the hand. Policy exploration/control is the lift blocker.")
    elif peak_hand_rise > 0.05 and peak_clearance_gain < 0.03:
        print("VERDICT: GRIP SLIPS -> hand rose, object stayed. The grasp is too weak to hold the weight.")
    else:
        print("VERDICT: inconclusive (hand didn't rise enough) -- raise lift_steps or check home pose.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
