"""Conservative Soft Actor-Critic (CSAC) — pure PyTorch, decoupled from Isaac Lab.

This file is self-contained: it has no Isaac Lab / Isaac Sim imports and can be
compiled and unit-tested on any machine with torch installed (see the
``__main__`` self-test at the bottom that runs ``CSAC.update`` on random data).

CSAC differs from vanilla SAC in exactly two structural ways (see the prompt
spec / Algorithm 1):

  1. The entropy temperature ``sigma`` and the memory-anchor weight ``tau`` are
     PER-SAMPLE tensors (shape (B, 1)), supplied by the contact-phase schedule.
     Every entropy / relative-entropy term is therefore weighted element-wise.

  2. Two extra ``-tau * log pi_e`` relative-entropy terms anchor the policy to
     ``pi_e`` = the *previous* policy. At the start of every ``update`` we
     snapshot the current actor into ``actor_prev`` (Algorithm 1: phi_p <- phi),
     making the relative-entropy term a one-step trust region. ``pi_e`` is also
     where a behavior-cloning prior is injected at the start of training (load
     the BC actor into both ``actor`` and ``actor_prev``).

Key equations (annotated again at the call sites):

  critic target, a' ~ pi(.|s'):
      y = r + gamma*(1-done)*[ min_j Qt_j(s',a')
                               - sigma' * log pi(a'|s')
                               - tau'   * (log pi(a'|s') - log pi_e(a'|s')) ]
  critic loss:
      MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
  actor loss, a~ ~ pi(.|s) (reparameterized):
      J = (tau+sigma)*log pi(a~|s) - tau*log pi_e(a~|s) - min_i Q_i(s,a~)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def mlp(in_dim: int, out_dim: int, hidden=(256, 256)) -> nn.Sequential:
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    """Diagonal Gaussian with tanh squashing (reparameterized sampling).

    Exposes ``sample`` (returns action, log-prob, and the *pre-tanh* sample u)
    and ``log_prob_of_u`` (log-prob of a given pre-tanh u under THIS network),
    so that ``log pi`` and ``log pi_e`` can be evaluated on the same u — the
    tanh Jacobian correction is identical for both and need not be threaded
    through separately.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim

    def _mean_logstd(self, obs):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    @staticmethod
    def _tanh_logprob(normal, u):
        """log N(u) - sum log(1 - tanh(u)^2)  (tanh change-of-variables)."""
        log_prob = normal.log_prob(u).sum(-1, keepdim=True)
        log_prob = log_prob - torch.log(1.0 - torch.tanh(u).pow(2) + EPS).sum(-1, keepdim=True)
        return log_prob

    def sample(self, obs):
        """Return (action=tanh(u), log pi(action|obs), u). Reparameterized."""
        mean, log_std = self._mean_logstd(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = mean + std * torch.randn_like(std)  # rsample, keeps grad
        action = torch.tanh(u)
        log_prob = self._tanh_logprob(normal, u)
        return action, log_prob, u

    def log_prob_of_u(self, obs, u):
        """log pi(tanh(u)|obs) under this net, for an externally supplied u."""
        mean, log_std = self._mean_logstd(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        return self._tanh_logprob(normal, u)

    @torch.no_grad()
    def act_deterministic(self, obs):
        mean, _ = self._mean_logstd(obs)
        return torch.tanh(mean)


class Critic(nn.Module):
    """Twin Q networks; ``forward`` returns (Q1, Q2)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


@dataclass
class CSACConfig:
    obs_dim: int
    act_dim: int
    hidden: tuple = (256, 256)
    gamma: float = 0.99
    lr: float = 3e-4
    rho: float = 0.005  # soft target update rate (a.k.a. tau in SAC; renamed to avoid clash)
    device: str = "cpu"


class CSAC:
    def __init__(self, cfg: CSACConfig):
        self.cfg = cfg
        d = cfg.device
        self.actor = TanhGaussianActor(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(d)
        # pi_e = previous policy. Snapshotted from actor at the start of every update().
        self.actor_prev = copy.deepcopy(self.actor).to(d)
        for p in self.actor_prev.parameters():
            p.requires_grad_(False)

        self.critic = Critic(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(d)
        self.critic_target = copy.deepcopy(self.critic).to(d)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.rho = cfg.rho

    # ------------------------------------------------------------------ act
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        obs = obs.to(self.cfg.device)
        if deterministic:
            return self.actor.act_deterministic(obs)
        a, _, _ = self.actor.sample(obs)
        return a

    # --------------------------------------------------------------- update
    def update(self, batch: dict) -> dict:
        """One CSAC gradient step on a batch of transitions.

        batch keys (all torch tensors on cfg.device):
          obs (B,O), act (B,A), rew (B,1), next_obs (B,O), done (B,1),
          sigma (B,1), tau (B,1)           -- schedule values at s   (actor loss)
          next_sigma (B,1), next_tau (B,1) -- schedule values at s'  (critic target)
        """
        d = self.cfg.device
        obs = batch["obs"].to(d)
        act = batch["act"].to(d)
        rew = batch["rew"].to(d)
        next_obs = batch["next_obs"].to(d)
        done = batch["done"].to(d)
        sigma = batch["sigma"].to(d)
        tau = batch["tau"].to(d)
        next_sigma = batch["next_sigma"].to(d)
        next_tau = batch["next_tau"].to(d)

        # Algorithm 1: phi_p <- phi. Snapshot current actor as pi_e BEFORE the
        # gradient step, so the relative-entropy term is a trust region against
        # the policy at the start of this update.
        self.actor_prev.load_state_dict(self.actor.state_dict())

        # ---------------- critic target ----------------
        # y = r + gamma*(1-done)*[ min_j Qt_j(s',a')
        #                          - sigma'*log pi(a'|s')
        #                          - tau'  *(log pi(a'|s') - log pi_e(a'|s')) ]
        with torch.no_grad():
            next_a, next_logp, next_u = self.actor.sample(next_obs)
            next_logp_e = self.actor_prev.log_prob_of_u(next_obs, next_u)  # log pi_e(a'|s')
            qt1, qt2 = self.critic_target(next_obs, next_a)
            min_qt = torch.min(qt1, qt2)
            soft_q = (
                min_qt
                - next_sigma * next_logp                      # entropy bonus (per-sample temp)
                - next_tau * (next_logp - next_logp_e)        # relative-entropy anchor to pi_e
            )
            y = rew + self.gamma * (1.0 - done) * soft_q

        # ---------------- critic loss ----------------
        # MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------- actor loss ----------------
        # J = (tau+sigma)*log pi(a~|s) - tau*log pi_e(a~|s) - min_i Q_i(s,a~)
        a_new, logp_new, u_new = self.actor.sample(obs)
        with torch.no_grad():
            logp_e_new = self.actor_prev.log_prob_of_u(obs, u_new)  # log pi_e(a~|s)
        q1_pi, q2_pi = self.critic(obs, a_new)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (
            (tau + sigma) * logp_new
            - tau * logp_e_new
            - min_q_pi
        ).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # ---------------- soft target update ----------------
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1.0 - self.rho).add_(self.rho * p)

        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "q_mean": float(min_q_pi.mean().detach()),
            "logp_mean": float(logp_new.mean().detach()),
            "sigma_mean": float(sigma.mean()),
            "tau_mean": float(tau.mean()),
        }

    # --------------------------------------------------------------- io
    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_prev": self.actor_prev.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.actor_prev.load_state_dict(sd["actor_prev"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])


# --------------------------------------------------------------------------- #
# Minimal self-test on random data (no Isaac Lab needed):  python csac.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    O, A, B = 22, 6, 128
    csac = CSAC(CSACConfig(obs_dim=O, act_dim=A, device="cpu"))

    def fake_batch():
        return {
            "obs": torch.randn(B, O),
            "act": torch.tanh(torch.randn(B, A)),
            "rew": torch.randn(B, 1),
            "next_obs": torch.randn(B, O),
            "done": (torch.rand(B, 1) < 0.1).float(),
            # per-sample schedule weights, in a plausible range
            "sigma": torch.rand(B, 1) * 0.2 + 0.02,
            "tau": torch.rand(B, 1) * 0.8 + 0.05,
            "next_sigma": torch.rand(B, 1) * 0.2 + 0.02,
            "next_tau": torch.rand(B, 1) * 0.8 + 0.05,
        }

    print("act() ->", csac.act(torch.randn(4, O)).shape, "/ det:", csac.act(torch.randn(4, O), True).shape)
    for i in range(5):
        logs = csac.update(fake_batch())
        print(f"step {i}: ", {k: round(v, 4) for k, v in logs.items()})
    print("CSAC self-test OK")
