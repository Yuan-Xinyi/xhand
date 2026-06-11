"""Priority-1 foundation check: does the force signal -> sigma/tau schedule work?

Runs a short rollout with NO learning. Actions are random in xy with a steady
downward bias so the peg reliably descends and makes contact (so we can observe
the free-space -> contact transition). For each step we log force_mag / depth /
xy_err and the schedule's (sigma, tau) + phase distribution.

We confirm:
  * free space -> force_mag ~ 0, schedule sits at sigma~0.2 / tau~0.1 (approach);
  * on contact -> force_mag jumps, schedule switches to sigma~0.05 / tau~0.8.

Run: python verify_signals.py --headless --num_envs 32 --steps 150 [--wrench commanded]
"""

from __future__ import annotations

import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--wrench", type=str, default="contact", choices=["contact", "commanded"])
parser.add_argument("--mode", type=str, default="by_phase", choices=["by_phase", "by_force", "constant"])
parser.add_argument("--out", type=str, default="/tmp/verify_signals.txt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import torch  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from env_cfg import InsertionEnvCfg  # noqa: E402
from insertion_env import InsertionEnv  # noqa: E402
from sigma_tau_schedule import SigmaTauConfig, SigmaTauSchedule, APPROACH, ALIGN, CONTACT, INSERT  # noqa: E402

R = open(args.out, "w")
def out(*a): R.write(" ".join(str(x) for x in a) + "\n"); R.flush()


def main():
    torch.manual_seed(0)
    cfg = InsertionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.wrench_source = args.wrench
    cfg.safety.force_abort = False  # don't end episodes during the probe
    env = InsertionEnv(cfg, render_mode=None)
    device = env.device
    sched = SigmaTauSchedule(SigmaTauConfig(), device=str(device))

    obs, _ = env.reset()
    out(f"wrench_source={args.wrench} schedule_mode={args.mode} num_envs={args.num_envs}")
    out("step |   F[min  mean  max] | depth_mean | xy_mean | phase[app,ali,con,ins] | sig[min mean max] | tau[min mean max]")

    peak_force = 0.0
    for t in range(args.steps):
        a = (torch.rand((args.num_envs, env.cfg.action_space), device=device) * 2 - 1)
        a[:, 2] = -0.7   # steady downward press so the peg actually contacts
        a[:, 3:] *= 0.2  # small rotations
        obs, rew, term, trunc, _ = env.step(a)

        sig = env.contact_signals()
        f = sig["force_mag"]
        sigma, tau = sched(sig, mode=args.mode)
        phase = sched.classify_phase(sig["force_mag"], sig["depth"], sig["xy_err"], sig["ori_err"])
        counts = [int((phase == p).sum()) for p in (APPROACH, ALIGN, CONTACT, INSERT)]
        peak_force = max(peak_force, float(f.max()))

        if t % 10 == 0 or t == args.steps - 1:
            out(f"{t:4d} | [{f.min():.2f} {f.mean():.2f} {f.max():.2f}] | "
                f"{sig['depth'].mean():.4f} | {sig['xy_err'].mean():.4f} | {counts} | "
                f"[{sigma.min():.3f} {sigma.mean():.3f} {sigma.max():.3f}] | "
                f"[{tau.min():.3f} {tau.mean():.3f} {tau.max():.3f}]")

    out("---")
    out(f"peak_force over rollout = {peak_force:.3f} N")
    out("PASS criteria: peak_force >> 0 (signal responds) and phase counts shift "
        "from approach toward contact/insert as the peg descends.")
    out("VERIFY DONE")
    R.close()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
