"""GPU ring replay buffer — pure PyTorch, decoupled from Isaac Lab.

Stores vectorized transitions (num_envs per env-step go in at once). Crucially,
it stores the schedule values (sigma, tau) at BOTH s and s', because CSAC needs:
  * sigma / tau  at s  -> actor loss
  * sigma'/ tau' at s' -> critic target
(see csac.py). All tensors live on ``device`` so sampling never touches the CPU.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.full = False

        z = lambda d: torch.zeros((self.capacity, d), device=device)
        self.obs = z(obs_dim)
        self.act = z(act_dim)
        self.rew = z(1)
        self.next_obs = z(obs_dim)
        self.done = z(1)
        self.sigma = z(1)
        self.tau = z(1)
        self.next_sigma = z(1)
        self.next_tau = z(1)

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def add(self, obs, act, rew, next_obs, done, sigma, tau, next_sigma, next_tau):
        """Add a batch of transitions. Leading dim = num_envs (or 1)."""
        def col(x, d):
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
            return x.view(-1, d)

        obs = col(obs, self.obs.shape[1])
        n = obs.shape[0]
        idx = torch.arange(self.ptr, self.ptr + n, device=self.device) % self.capacity

        self.obs[idx] = obs
        self.act[idx] = col(act, self.act.shape[1])
        self.rew[idx] = col(rew, 1)
        self.next_obs[idx] = col(next_obs, self.next_obs.shape[1])
        self.done[idx] = col(done, 1)
        self.sigma[idx] = col(sigma, 1)
        self.tau[idx] = col(tau, 1)
        self.next_sigma[idx] = col(next_sigma, 1)
        self.next_tau[idx] = col(next_tau, 1)

        new_ptr = self.ptr + n
        if new_ptr >= self.capacity:
            self.full = True
        self.ptr = new_ptr % self.capacity

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "sigma": self.sigma[idx],
            "tau": self.tau[idx],
            "next_sigma": self.next_sigma[idx],
            "next_tau": self.next_tau[idx],
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    O, A = 22, 6
    buf = ReplayBuffer(capacity=1000, obs_dim=O, act_dim=A, device="cpu")
    for _ in range(20):
        E = 8  # num_envs
        buf.add(
            obs=torch.randn(E, O), act=torch.randn(E, A), rew=torch.randn(E, 1),
            next_obs=torch.randn(E, O), done=(torch.rand(E, 1) < 0.1).float(),
            sigma=torch.rand(E, 1), tau=torch.rand(E, 1),
            next_sigma=torch.rand(E, 1), next_tau=torch.rand(E, 1),
        )
    print("buffer size:", buf.size)
    b = buf.sample(32)
    print({k: tuple(v.shape) for k, v in b.items()})
    print("buffer self-test OK")
