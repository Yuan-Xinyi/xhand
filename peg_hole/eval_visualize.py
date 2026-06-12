"""Visualize a trained CSAC checkpoint in the Isaac Sim GUI.

Loads a saved CSAC state_dict (default demos/csac_insert.pt), runs the
DETERMINISTIC policy (tanh(mean), no exploration noise) in a small number of
envs with the GUI window open so you can watch the peg's behavior, and writes
per-step success / force / depth / xy-error stats to a text file (the Isaac app
swallows stdout).

Run (GUI window on DISPLAY=:1, the wrench source that actually initializes):
    conda activate env_isaaclab
    python eval_visualize.py --num_envs 4 --steps 600 --ckpt demos/csac_insert.pt
"""

from __future__ import annotations

import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize a trained CSAC checkpoint")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--ckpt", type=str, default=os.path.join(os.path.dirname(__file__), "demos", "csac_insert.pt"))
parser.add_argument("--stochastic", action="store_true", help="sample actions instead of deterministic mean")
parser.add_argument("--wrench", type=str, default="commanded", choices=["contact", "commanded"])
parser.add_argument("--easy", action="store_true", help="match the easy init-randomization used in --easy training")
parser.add_argument("--out", type=str, default="/tmp/eval_visualize.txt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import torch  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from csac import CSAC, CSACConfig  # noqa: E402
from env_cfg import InsertionEnvCfg  # noqa: E402
from insertion_env import InsertionEnv  # noqa: E402

R = open(args.out, "w")
def out(*a): R.write(" ".join(str(x) for x in a) + "\n"); R.flush()


def main():
    torch.manual_seed(0)
    cfg = InsertionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.wrench_source = args.wrench
    cfg.safety.force_abort = False  # let episodes run so we can watch behavior
    if args.easy:
        cfg.task.fixed_asset_init_pos_noise = [0.005, 0.005, 0.005]
        cfg.task.hand_init_pos_noise = [0.005, 0.005, 0.005]
        cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
        cfg.task.fixed_asset_init_orn_range_deg = 0.0

    env = InsertionEnv(cfg, render_mode=None)
    device = env.device
    obs_dim = env.cfg.observation_space
    act_dim = env.cfg.action_space

    hp = cfg.csac
    csac = CSAC(CSACConfig(obs_dim=obs_dim, act_dim=act_dim, hidden=tuple(hp.hidden),
                           gamma=hp.gamma, lr=hp.lr, rho=hp.rho, device=str(device)))

    if not os.path.isfile(args.ckpt):
        out(f"ERROR: checkpoint not found: {args.ckpt}")
        env.close(); app.close(); return
    sd = torch.load(args.ckpt, map_location=device)
    csac.load_state_dict(sd)
    out(f"loaded {args.ckpt}  obs_dim={obs_dim} act_dim={act_dim} num_envs={args.num_envs} "
        f"deterministic={not args.stochastic} wrench={args.wrench}")
    out("step | reward_mean | F[mean max] | depth[mean max] | xy_err_mean | ori_err_mean | n_success")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    ever_success = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    best_depth = 0.0
    for t in range(args.steps):
        action = csac.act(obs, deterministic=not args.stochastic)
        action = action.clamp(-1.0, 1.0)
        obs_dict, rew, term, trunc, _ = env.step(action)
        obs = obs_dict["policy"]

        sig = env.contact_signals()
        succ = float(env.extras.get("successes", torch.tensor(0.0)))
        ever_success = ever_success | env._success_edge
        best_depth = max(best_depth, float(sig["depth"].max()))

        if t % 20 == 0 or t == args.steps - 1:
            out(f"{t:4d} | {float(rew.mean()):+.4f} | "
                f"[{sig['force_mag'].mean():.2f} {sig['force_mag'].max():.2f}] | "
                f"[{sig['depth'].mean():.4f} {sig['depth'].max():.4f}] | "
                f"{sig['xy_err'].mean():.4f} | {sig['ori_err'].mean():.4f} | {succ:.3f}")

    out("---")
    out(f"envs that EVER hit success edge: {int(ever_success.sum())}/{args.num_envs}")
    out(f"best insertion depth seen over rollout: {best_depth:.4f} m")
    out("EVAL DONE")
    R.close()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
