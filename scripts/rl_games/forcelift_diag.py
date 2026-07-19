# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Force-lift test: does the policy's GRIP actually hold the object against gravity?

Phase 1 (grip): run the policy until it forms a contact grasp (freeze the hand token it was
commanding at that moment). Phase 2 (forced lift): keep the frozen hand token (hold the grip) and
drive the ARM joints back toward their home/reset pose (the high configuration the arm descended
from), a bounded delta per step. Log palm rise vs object rise. If the object follows the hand up ->
the grasp holds and the failure is purely "policy won't lift"; if the hand rises but the object
stays -> the grip slips and the grasp itself is too weak."""

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
    palm0 = (u.palm_center_w[0, 2] - origin_z).item()

    def readout():
        palm_dz = (u.palm_center_w[0, 2] - origin_z).item() - palm0
        lift = (u.object_pos_w[0, 2] - origin_z - u.object_default_z[0]).item()
        fN = u._contact_sensor.data.net_forces_w.norm(dim=-1)[0].sum().item()
        tc, oc = u._finger_contact_state()
        cg = bool((tc[0] & (oc[0] >= 1)).item())
        return palm_dz, lift, cg, fN

    # ---- phase 1: let the policy grip; freeze the hand token once a grasp forms ----
    frozen_hand = None
    for t in range(args_cli.grip_steps):
        with torch.inference_mode():
            ob = agent.obs_to_torch(obs)
            act = agent.get_action(ob, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(act)
        _, _, cg, _ = readout()
        if cg and frozen_hand is None:
            frozen_hand = act[0, n_arm:].clone()  # hand token at first grasp
    if frozen_hand is None:
        frozen_hand = act[0, n_arm:].clone()
        print("[warn] no contact grasp formed in phase 1; freezing last hand token anyway")
    pd, lf, cg, fN = readout()
    print(f"\n[grip end] palm_dz={pd:+.3f} obj_lift={lf:+.3f} contact={int(cg)} fN={fN:.1f}")

    # ---- phase 2: hold the frozen grip, drive arm joints toward home (up) ----
    print("\n---- FORCED LIFT (hand grip frozen, arm driven to home pose) ----")
    print("step  palm_dz  obj_lift  gap    contact  fN")
    peak_lift = lf
    for t in range(args_cli.lift_steps):
        cur_arm = u.dof_targets[0, arm_ids]
        arm_delta = torch.clamp((home_arm - cur_arm) / action_scale, -1.0, 1.0)
        act = torch.zeros((1, n_arm + frozen_hand.shape[0]), device=u.device)
        act[0, :n_arm] = arm_delta
        act[0, n_arm:] = frozen_hand
        with torch.inference_mode():
            obs, _, dones, _ = env.step(act)
        if isinstance(obs, dict): obs = obs["obs"]
        pd, lf, cg, fN = readout()
        peak_lift = max(peak_lift, lf)
        if t % 10 == 0:
            print(f"{t:4d}  {pd:+.3f}   {lf:+.3f}   {pd-lf:+.3f}  {int(cg)}       {fN:5.1f}")
        if bool(dones[0]) if not torch.is_tensor(dones) else bool(dones.flatten()[0]):
            print(f"  [episode terminated at lift step {t} -> object likely dropped]")
            break

    print("\n---- verdict ----")
    print(f"peak object lift during forced lift: {peak_lift:+.3f} m   (hand target rose ~{-palm0+ (u.dof_targets[0,arm_ids]-home_arm).abs().mean().item()*0:.2f})")
    print(f"final: palm_dz={pd:+.3f} obj_lift={lf:+.3f} gap={pd-lf:+.3f} contact={int(cg)}")
    if pd > 0.05 and lf > 0.05 and (pd - lf) < 0.05:
        print("VERDICT: GRIP HOLDS -> object follows the hand up. Failure is 'policy won't lift' (reward/explore).")
    elif pd > 0.05 and lf < 0.03:
        print("VERDICT: GRIP SLIPS -> hand rose, object stayed. The grasp is too weak to hold the weight.")
    else:
        print("VERDICT: inconclusive (hand didn't rise enough) -- raise lift_steps or check home pose.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
