"""Collect scripted demonstrations + behavior-clone the pi_e prior.

Pipeline:
  1) Roll out a SCRIPTED spiral-search + press-down controller in the OSC
     delta-pose action space (a teleop hook is left as a TODO below).
  2) Behavior-clone a same-structure TanhGaussianActor on (obs, action) pairs.
  3) Save demos/bc_actor.pt = {"actor": state_dict}.

train.py loads this into BOTH actor and actor_prev (pi_e). Under sparse rewards,
the relative-entropy term then anchors learning to this prior — the exploration
source. If the checkpoint is absent, train.py starts pi_e from scratch.

Run: ``python bc_pretrain.py --num_envs 64 --demo_steps 6000``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted demos + BC pretraining")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--demo_steps", type=int, default=6000, help="env steps of scripted data to collect")
parser.add_argument("--bc_epochs", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "demos", "bc_actor.pt"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from csac import TanhGaussianActor  # noqa: E402
from env_cfg import InsertionEnvCfg  # noqa: E402
from insertion_env import InsertionEnv  # noqa: E402


def scripted_action(env, t: int) -> torch.Tensor:
    """Spiral-search toward the hole + steady downward press (OSC delta-pose)."""
    rel = env.fingertip_midpoint_pos - (env.fixed_pos_obs_frame + env.init_fixed_pos_obs_noise)
    a = torch.zeros((env.num_envs, 6), device=env.device)
    # Proportional centering toward the hole axis ...
    a[:, 0] = torch.clamp(-4.0 * rel[:, 0], -1.0, 1.0)
    a[:, 1] = torch.clamp(-4.0 * rel[:, 1], -1.0, 1.0)
    # ... plus an expanding spiral so the peg hunts for the opening ...
    spiral_r = 0.3
    a[:, 0] = a[:, 0] + spiral_r * math.cos(0.3 * t)
    a[:, 1] = a[:, 1] + spiral_r * math.sin(0.3 * t)
    # ... and a constant downward press.
    a[:, 2] = -0.5
    # TODO(teleop): replace the above with a teleop device read here if desired.
    return a.clamp(-1.0, 1.0)


def collect(env, n_steps: int):
    obs_buf, act_buf = [], []
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    for t in range(n_steps):
        action = scripted_action(env, t)
        obs_buf.append(obs.clone())
        act_buf.append(action.clone())
        obs_dict, _, _, _, _ = env.step(action)
        obs = obs_dict["policy"]
    return torch.cat(obs_buf, 0), torch.cat(act_buf, 0)


def behavior_clone(obs, act, obs_dim, act_dim, device, epochs):
    """Fit a TanhGaussianActor by NLL of demo actions (tanh-space targets)."""
    actor = TanhGaussianActor(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    # pre-tanh targets u = atanh(a)
    u_target = torch.atanh(act.clamp(-0.999, 0.999))
    n = obs.shape[0]
    bs = 1024
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logp = actor.log_prob_of_u(obs[idx], u_target[idx])
            loss = (-logp).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"[bc] epoch {ep}: nll={tot / n:.4f}")
    return actor


def main():
    torch.manual_seed(args.seed)
    cfg = InsertionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = InsertionEnv(cfg, render_mode=None)
    device = env.device
    obs_dim = env.cfg.observation_space
    act_dim = env.cfg.action_space

    print(f"[bc] collecting {args.demo_steps} scripted env-steps ...")
    obs, act = collect(env, args.demo_steps)
    print(f"[bc] collected {obs.shape[0]} (obs,action) pairs")

    actor = behavior_clone(obs, act, obs_dim, act_dim, device, args.bc_epochs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"actor": actor.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim}, args.out)
    print(f"[bc] saved BC actor (pi_e prior) to {args.out}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
