#!/usr/bin/env python3
"""Distal-only bounded residual adapter for a frozen pick-tool actor."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


OBS_DIM = 115
ACTION_DIM = 21
DISTAL_SLICE = slice(16, 21)
SUFFIX_SLICE = slice(87, 115)
PROXIMITY_SLICE = slice(92, 97)
LATCH_INDEX = 106


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class DistalResidualAdapter(nn.Module):
    """Map frozen actor features and close-state observations to five bounded deltas."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 32,
        delta_limit: float = 0.10,
        proximity_threshold: float = 0.02,
        *,
        zero_head: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0 or delta_limit <= 0.0:
            raise ValueError("latent/hidden dimensions and delta_limit must be positive")
        if not 0.0 <= proximity_threshold <= 1.0:
            raise ValueError("proximity_threshold must be in [0,1]")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_limit = float(delta_limit)
        self.proximity_threshold = float(proximity_threshold)
        input_dim = self.latent_dim + (SUFFIX_SLICE.stop - SUFFIX_SLICE.start) + 5
        self.hidden = nn.Linear(input_dim, self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, 5)
        if zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        actor_latent: torch.Tensor,
        observation: torch.Tensor,
        base_distal: torch.Tensor,
    ) -> torch.Tensor:
        if actor_latent.ndim != 2 or actor_latent.shape[1] != self.latent_dim:
            raise ValueError("actor_latent has an incompatible shape")
        if observation.ndim != 2 or observation.shape[1] != OBS_DIM:
            raise ValueError("observation must have shape [N,115]")
        if base_distal.shape != (observation.shape[0], 5):
            raise ValueError("base_distal must have shape [N,5]")
        features = torch.cat(
            (
                actor_latent,
                observation[:, SUFFIX_SLICE].clamp(-5.0, 5.0),
                base_distal.clamp(-1.0, 1.0),
            ),
            dim=-1,
        )
        return torch.tanh(self.head(F.elu(self.hidden(features))))

    def intrinsic_gate(self, observation: torch.Tensor) -> torch.Tensor:
        near = observation[:, PROXIMITY_SLICE].mean(dim=-1) >= self.proximity_threshold
        return near & (observation[:, LATCH_INDEX] < 0.5)

    def apply(
        self,
        base_action: torch.Tensor,
        observation: torch.Tensor,
        actor_latent: torch.Tensor,
        *,
        scale: float = 1.0,
        external_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if base_action.ndim != 2 or base_action.shape[1] != ACTION_DIM:
            raise ValueError("base_action must have shape [N,21]")
        if not 0.0 <= scale <= 1.0:
            raise ValueError("scale must be in [0,1] so the checkpoint delta limit remains hard")
        if external_gate is None:
            gate = self.intrinsic_gate(observation)
        else:
            if external_gate.shape != (observation.shape[0],):
                raise ValueError("external_gate must have shape [N]")
            # Stateful deployment owns proximity hysteresis.  Reapplying the entry threshold here
            # would silently turn its [exit, entry) dead-band off; latch remains an independent
            # hard safety guard in both modes.
            gate = external_gate.bool() & (observation[:, LATCH_INDEX] < 0.5)
        unit = self(actor_latent, observation, base_action[:, DISTAL_SLICE])
        proposed = unit * (self.delta_limit * float(scale)) * gate.unsqueeze(-1)
        result = base_action.clone()
        result[:, DISTAL_SLICE] = (base_action[:, DISTAL_SLICE] + proposed).clamp(-1.0, 1.0)
        actual_delta = result[:, DISTAL_SLICE] - base_action[:, DISTAL_SLICE]
        return result, gate, actual_delta


class StatefulCloseGate:
    """Latch a qualified pregrasp into the close window without learning phase transitions."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        timeout_steps: int = 120,
        *,
        entry_proximity: float = 0.02,
        exit_proximity: float = 0.01,
    ) -> None:
        if num_envs <= 0 or timeout_steps <= 0:
            raise ValueError("num_envs and timeout_steps must be positive")
        if not 0.0 <= exit_proximity <= entry_proximity <= 1.0:
            raise ValueError("require 0 <= exit_proximity <= entry_proximity <= 1")
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.blocked = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.timeout_steps = int(timeout_steps)
        self.entry_proximity = float(entry_proximity)
        self.exit_proximity = float(exit_proximity)

    def update(
        self,
        observation: torch.Tensor,
        entry_gate: torch.Tensor,
        allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        proximity = observation[:, PROXIMITY_SLICE].mean(dim=-1)
        latched = observation[:, LATCH_INDEX] >= 0.5
        if allowed is None:
            allowed = torch.ones_like(self.active)
        allowed = allowed.bool()
        near = proximity >= self.exit_proximity
        # Once a close attempt times out, require a real phase reset (move away, latch, safety
        # reset, or option deactivation) before another adapter window can begin.
        self.blocked &= near & ~latched & allowed
        timed_out = self.active & (self.age >= self.timeout_steps)
        self.blocked |= timed_out
        remain = (
            self.active
            & near
            & ~latched
            & allowed
            & (self.age < self.timeout_steps)
        )
        enter = (
            ~remain
            & ~self.blocked
            & entry_gate.bool()
            & (proximity >= self.entry_proximity)
            & ~latched
            & allowed
        )
        self.active.copy_(remain | enter)
        self.age[~self.active] = 0
        self.age[enter] = 0
        self.age[self.active] += 1
        return self.active.clone()

    def reset(self, mask: torch.Tensor) -> None:
        self.active[mask] = False
        self.blocked[mask] = False
        self.age[mask] = 0


def make_payload(
    adapter: DistalResidualAdapter,
    base_checkpoint: Path,
    train_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "base_sha256": sha256(base_checkpoint),
        "base_checkpoint": str(base_checkpoint.resolve()),
        "obs_dim": OBS_DIM,
        "action_indices": list(range(DISTAL_SLICE.start, DISTAL_SLICE.stop)),
        "architecture": [adapter.latent_dim + 33, adapter.hidden_dim, 5],
        "latent_dim": adapter.latent_dim,
        "hidden_dim": adapter.hidden_dim,
        "delta_limit": adapter.delta_limit,
        "proximity_threshold": adapter.proximity_threshold,
        "gate_exit_proximity": 0.5 * adapter.proximity_threshold,
        "state_dict": {
            key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()
        },
        "train_meta": train_meta,
    }


def load_adapter(
    path: Path,
    base_checkpoint: Path,
    device: str | torch.device,
) -> tuple[DistalResidualAdapter, dict[str, Any]]:
    payload = load_torch(path)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise RuntimeError("unsupported distal adapter checkpoint")
    expected = {
        "obs_dim": OBS_DIM,
        "action_indices": list(range(16, 21)),
        "base_sha256": sha256(base_checkpoint),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"adapter {key}={payload.get(key)!r}, expected {value!r}")
    if not isinstance(payload.get("state_dict"), dict):
        raise KeyError("adapter checkpoint is missing state_dict")
    adapter = DistalResidualAdapter(
        latent_dim=int(payload["latent_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        delta_limit=float(payload["delta_limit"]),
        proximity_threshold=float(payload["proximity_threshold"]),
        zero_head=False,
    )
    adapter.load_state_dict(payload["state_dict"], strict=True)
    return adapter.to(device).eval(), payload
