"""CSAC training loop for Factory PegInsert.  Run: ``python train.py``.

Startup order matters: launch the simulation app via AppLauncher FIRST, then
import any isaaclab / task modules. The env is vectorized (num_envs parallel
envs); every step pushes num_envs transitions into the replay buffer.

Pipeline per step:  env -> sigma/tau schedule -> safety clamp -> CSAC.update.
"""

from __future__ import annotations

import argparse
import os
import sys

# ----------------------------------------------------------------------------
# 1) Launch the simulation app BEFORE importing isaaclab task modules.
# ----------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CSAC peg-in-hole training")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--total_steps", type=int, default=None, help="override total env steps")
parser.add_argument("--learn_start", type=int, default=None, help="override learn-start env steps")
parser.add_argument("--batch_size", type=int, default=None, help="override CSAC batch size")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sparse", action="store_true", help="sparse reward (success + force penalty only)")
parser.add_argument("--reach_only", action="store_true", help="easiest target: reward only centering over the hole")
parser.add_argument("--easy", action="store_true", help="shrink init randomization for a 'can it learn?' check")
parser.add_argument("--algo", type=str, default="csac", choices=["csac", "sac"],
                    help="sac = CSAC with tau->0 and constant sigma (isolates the tau term)")
parser.add_argument("--sac_sigma", type=float, default=0.2, help="constant entropy temp used in --algo sac")
parser.add_argument("--no_bc", action="store_true", help="ignore any BC checkpoint; pi_e starts from scratch")
parser.add_argument("--bc_ckpt", type=str, default=os.path.join(os.path.dirname(__file__), "demos", "bc_actor.pt"))
parser.add_argument("--schedule_mode", type=str, default="by_phase", choices=["by_phase", "by_force", "constant"])
parser.add_argument("--save_path", type=str, default=os.path.join(os.path.dirname(__file__), "demos", "csac_ckpt.pt"))
parser.add_argument("--log_csv", type=str, default=None, help="metrics CSV path (default demos/metrics_<algo>.csv)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------
# 2) Now safe to import torch + isaaclab + our modules.
# ----------------------------------------------------------------------------
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from buffer import ReplayBuffer  # noqa: E402
from csac import CSAC, CSACConfig  # noqa: E402
from env_cfg import InsertionEnvCfg  # noqa: E402
from insertion_env import InsertionEnv  # noqa: E402
from sigma_tau_schedule import SigmaTauConfig, SigmaTauSchedule  # noqa: E402


def build_schedule_cfg(sc) -> SigmaTauConfig:
    return SigmaTauConfig(
        sigma_by_phase=tuple(sc.sigma_by_phase),
        tau_by_phase=tuple(sc.tau_by_phase),
        align_xy_thresh=sc.align_xy_thresh,
        contact_force_thresh=sc.contact_force_thresh,
        insert_depth_thresh=sc.insert_depth_thresh,
        force_lo=sc.force_lo, force_hi=sc.force_hi,
        sigma_free=sc.sigma_free, sigma_contact=sc.sigma_contact,
        tau_free=sc.tau_free, tau_contact=sc.tau_contact,
    )


def main():
    torch.manual_seed(args.seed)

    # --- env ---
    cfg = InsertionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward.sparse = args.sparse
    cfg.reward.reach_only = args.reach_only
    cfg.schedule.mode = args.schedule_mode
    if args.easy:
        # Shrink init randomization so the task is easy enough to confirm learning.
        cfg.task.fixed_asset_init_pos_noise = [0.005, 0.005, 0.005]
        cfg.task.hand_init_pos_noise = [0.005, 0.005, 0.005]
        cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
        cfg.task.fixed_asset_init_orn_range_deg = 0.0
        print("[train] --easy: reduced init randomization")
    env = InsertionEnv(cfg, render_mode=None)
    device = env.device

    obs_dim = env.cfg.observation_space  # set by FactoryEnv.__init__ from obs_order + action
    act_dim = env.cfg.action_space
    print(f"[train] obs_dim={obs_dim} act_dim={act_dim} num_envs={args.num_envs} device={device}")

    # --- CSAC ---
    hp = cfg.csac
    csac = CSAC(CSACConfig(obs_dim=obs_dim, act_dim=act_dim, hidden=tuple(hp.hidden),
                           gamma=hp.gamma, lr=hp.lr, rho=hp.rho, device=str(device)))

    # --- BC prior pi_e: load into BOTH actor and actor_prev if a checkpoint exists ---
    # Without a checkpoint (or with --no_bc) pi_e simply starts as the initial actor;
    # CSAC is self-consistent because the tau term defaults to "pi_e = previous policy".
    if args.no_bc:
        print("[train] --no_bc: skipping BC prior; pi_e = previous-policy trust region only.")
    elif os.path.isfile(args.bc_ckpt):
        sd = torch.load(args.bc_ckpt, map_location=device)
        actor_sd = sd["actor"] if isinstance(sd, dict) and "actor" in sd else sd
        csac.actor.load_state_dict(actor_sd)
        csac.actor_prev.load_state_dict(actor_sd)
        print(f"[train] loaded BC prior pi_e from {args.bc_ckpt}")
    else:
        print(f"[train] no BC checkpoint at {args.bc_ckpt}; starting pi_e from scratch.")

    # --- optional CLI overrides (handy for quick smoke runs) ---
    if args.learn_start is not None:
        hp.learn_start_steps = args.learn_start
    if args.batch_size is not None:
        hp.batch_size = args.batch_size

    # --- schedule + buffer ---
    sched = SigmaTauSchedule(build_schedule_cfg(cfg.schedule), device=str(device))
    sched_mode = args.schedule_mode
    buf = ReplayBuffer(hp.buffer_capacity, obs_dim, act_dim, device=str(device))

    total_steps = args.total_steps or hp.total_env_steps
    action_clip = cfg.safety.action_clip
    abort_thresh = cfg.safety.force_abort_thresh

    def schedule_at(signals):
        """sigma/tau from the schedule; for --algo sac: tau->0, sigma=const (=> plain SAC)."""
        s, t = sched(signals, mode=sched_mode)
        if args.algo == "sac":
            s = torch.full_like(s, args.sac_sigma)
            t = torch.zeros_like(t)
        return s, t

    # metrics CSV (stdout is swallowed by the Isaac app; this is for plotting curves)
    csv_path = args.log_csv or os.path.join(os.path.dirname(__file__), "demos", f"metrics_{args.algo}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv = open(csv_path, "w")
    csv.write("step,reward,critic_loss,actor_loss,q_mean,logp_mean,sigma_mean,tau_mean,"
              "act_sat_frac,n_abort,abort_force_mean,success\n")
    csv.flush()

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    sig, tau = schedule_at(env.contact_signals())  # schedule at s
    logs = {k: float("nan") for k in ("critic_loss", "actor_loss", "q_mean", "logp_mean", "sigma_mean", "tau_mean")}

    for step in range(total_steps):
        # --- act + safety clamp (unit box) ---
        if step < hp.learn_start_steps:
            action = (torch.rand((args.num_envs, act_dim), device=device) * 2 - 1)
        else:
            action = csac.act(obs, deterministic=False)
        action = action.clamp(-action_clip, action_clip)

        # --- env step ---
        next_obs_dict, rew, terminated, truncated, _ = env.step(action)
        next_obs = next_obs_dict["policy"]
        next_signals = env.contact_signals()  # at s'
        next_sig, next_tau = schedule_at(next_signals)

        # done for bootstrapping: ground the value at the insertion moment (success
        # rising edge) and at a force abort. NOT the timeout (that is truncation, and
        # `terminated` returned here is the synchronous timeout — Factory resets all
        # envs together, so we read the per-env masks the env exposes instead).
        done_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        if cfg.safety.ground_done_on_success:
            done_mask = done_mask | env._success_edge
        if cfg.safety.ground_done_on_abort:
            done_mask = done_mask | env._abort_mask
        done = done_mask.float().unsqueeze(-1)
        buf.add(obs, action, rew.unsqueeze(-1), next_obs, done,
                sig.unsqueeze(-1), tau.unsqueeze(-1), next_sig.unsqueeze(-1), next_tau.unsqueeze(-1))

        obs, sig, tau = next_obs, next_sig, next_tau

        # --- learn ---
        if step >= hp.learn_start_steps and buf.size >= hp.batch_size:
            for _ in range(hp.updates_per_step):
                logs = csac.update(buf.sample(hp.batch_size))

        # --- metrics + numerical sanity checks ---
        if step % 200 == 0:
            succ = float(env.extras.get("successes", torch.tensor(0.0)))
            # (1) action saturation: fraction of |action| pinned near the tanh edge
            act_sat = float((action.abs() > 0.99).float().mean())
            # (2) force-abort sanity: how many envs aborted, and at what force
            aborted = env._abort_mask
            n_abort = int(aborted.sum())
            fmag = next_signals["force_mag"]
            abort_fmean = float(fmag[aborted].mean()) if n_abort > 0 else 0.0
            # (3) logp NaN guard
            if logs["logp_mean"] != logs["logp_mean"]:  # NaN
                print(f"[WARN step {step}] logp is NaN — tanh-policy numerics unstable.")
            if act_sat > 0.5:
                print(f"[WARN step {step}] {act_sat:.0%} of actions saturated at +/-1 — entropy may be collapsing.")

            csv.write(f"{step},{float(rew.mean()):.4f},{logs['critic_loss']:.4f},{logs['actor_loss']:.4f},"
                      f"{logs['q_mean']:.4f},{logs['logp_mean']:.4f},{logs['sigma_mean']:.4f},{logs['tau_mean']:.4f},"
                      f"{act_sat:.4f},{n_abort},{abort_fmean:.4f},{succ:.4f}\n")
            csv.flush()
            print(f"[{step:>7}] rew={float(rew.mean()):.3f} critic={logs['critic_loss']:.3f} "
                  f"actor={logs['actor_loss']:.3f} q={logs['q_mean']:.2f} "
                  f"sig={logs['sigma_mean']:.3f} tau={logs['tau_mean']:.3f} "
                  f"sat={act_sat:.2f} abort={n_abort} succ={succ:.3f}")

        if step % 20000 == 0 and step > 0:
            torch.save(csac.state_dict(), args.save_path)

    torch.save(csac.state_dict(), args.save_path)
    csv.close()
    print(f"[train] done. saved CSAC to {args.save_path}; metrics -> {csv_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
