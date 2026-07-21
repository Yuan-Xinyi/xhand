"""Torch-only integration boundary for the pinned FlashSAC agent.

The upstream agent at ``third_party/FlashSAC`` is intentionally left unchanged.
This module reuses its networks, replay buffer, and update implementation while
fixing the interaction boundary needed by Isaac Lab:

* actions remain tensors on the agent device (no ``cpu().numpy()`` round trip),
* transitions must already be tensors on that device,
* action groups can use different exploration scales/repetition settings, and
* exploration state and RNG state are included in checkpoints.

The grouped-noise API is deliberately independent of the pick-tool observation
layout.  A future trainer can supply a per-environment ``noise_scale`` tensor to
``sample_actions`` (for example to lower hand noise after a grasp latch) without
changing this bridge.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import types
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch


FLASH_SAC_COMMIT = "87edc9061150ae9e962dd84e6544e27a1554b3ab"
BRIDGE_STATE_FILENAME = "torch_bridge_state.pt"
BRIDGE_CHECKPOINT_VERSION = 1

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_ROOT = _PROJECT_ROOT / "third_party" / "FlashSAC"
if not (_UPSTREAM_ROOT / "flash_rl" / "agents" / "flashSAC" / "agent.py").is_file():
    raise ImportError(
        "Pinned FlashSAC checkout is missing. Expected "
        f"{_UPSTREAM_ROOT} at commit {FLASH_SAC_COMMIT}."
    )
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))


def _install_types_only_jax_fallback() -> None:
    """Avoid making JAX a runtime dependency of the Torch-only Isaac process.

    The pinned upstream imports ``jax.numpy`` solely to construct a union used
    in type annotations.  Its FlashSAC Torch agent never executes JAX code.  We
    install only ``flash_rl.types`` when JAX is unavailable, rather than a fake
    global ``jax`` package that could mask a real dependency elsewhere.
    """

    try:
        import jax.numpy  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "jax" and not (exc.name or "").startswith("jax."):
            raise
        module = types.ModuleType("flash_rl.types")
        module.NDArray = Any
        module.F32NDArray = Any
        module.Tensor = torch.Tensor
        sys.modules.setdefault("flash_rl.types", module)


_install_types_only_jax_fallback()

from flash_rl.agents.flashSAC.agent import (  # noqa: E402
    FlashSACAgent,
    FlashSACConfig,
)


@dataclasses.dataclass(frozen=True)
class ActionNoiseGroup:
    """A contiguous action slice with its own exploration behavior.

    ``scale`` multiplies the actor's learned standard deviation only during
    environment interaction.  SAC actor/temperature updates remain upstream's
    implementation.  ``zeta_mu`` and ``zeta_max`` default to the corresponding
    values in ``FlashSACConfig``.
    """

    name: str
    start: int
    stop: int
    scale: float = 1.0
    zeta_mu: float | None = None
    zeta_max: int | None = None


_REQUIRED_TRANSITION_KEYS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)


def assert_transition_tensors(
    transition: Mapping[str, Any],
    *,
    device: torch.device | str,
    observation_dim: int | None = None,
    action_dim: int | None = None,
) -> None:
    """Validate the zero-copy interaction contract before replay insertion."""

    expected_device = torch.device(device)
    missing = [key for key in _REQUIRED_TRANSITION_KEYS if key not in transition]
    if missing:
        raise KeyError(f"transition is missing required keys: {missing}")

    for key in _REQUIRED_TRANSITION_KEYS:
        value = transition[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"transition[{key!r}] must be a torch.Tensor; got {type(value).__name__}. "
                "Do not convert Isaac tensors to NumPy."
            )
        if value.device != expected_device:
            raise ValueError(
                f"transition[{key!r}] is on {value.device}, expected {expected_device}; "
                "the bridge does not perform hidden host/device transfers."
            )

    observation = transition["observation"]
    next_observation = transition["next_observation"]
    action = transition["action"]
    if observation.ndim != 2 or next_observation.shape != observation.shape:
        raise ValueError(
            "observation and next_observation must have the same [num_envs, obs_dim] shape; "
            f"got {tuple(observation.shape)} and {tuple(next_observation.shape)}"
        )
    if action.ndim != 2 or action.shape[0] != observation.shape[0]:
        raise ValueError(
            "action must have shape [num_envs, action_dim] with the same batch size as observation; "
            f"got {tuple(action.shape)} and {tuple(observation.shape)}"
        )
    if observation_dim is not None and observation.shape[1] != observation_dim:
        raise ValueError(f"expected observation_dim={observation_dim}, got {observation.shape[1]}")
    if action_dim is not None and action.shape[1] != action_dim:
        raise ValueError(f"expected action_dim={action_dim}, got {action.shape[1]}")

    num_envs = observation.shape[0]
    for key in ("reward", "terminated", "truncated"):
        if transition[key].shape != (num_envs,):
            raise ValueError(f"transition[{key!r}] must have shape ({num_envs},), got {tuple(transition[key].shape)}")


def _normalize_noise_groups(
    groups: Sequence[ActionNoiseGroup],
    *,
    action_dim: int,
    default_mu: float,
    default_max: int,
) -> tuple[ActionNoiseGroup, ...]:
    if not groups:
        groups = (ActionNoiseGroup("all", 0, action_dim),)

    normalized: list[ActionNoiseGroup] = []
    occupied = [False] * action_dim
    names: set[str] = set()
    for group in groups:
        if not group.name or group.name in names:
            raise ValueError(f"noise group names must be non-empty and unique; got {group.name!r}")
        names.add(group.name)
        if not 0 <= group.start < group.stop <= action_dim:
            raise ValueError(
                f"noise group {group.name!r} has invalid slice [{group.start}:{group.stop}] "
                f"for action_dim={action_dim}"
            )
        if not math.isfinite(group.scale) or group.scale < 0.0:
            raise ValueError(f"noise group {group.name!r} scale must be finite and non-negative")
        mu = default_mu if group.zeta_mu is None else group.zeta_mu
        max_n = default_max if group.zeta_max is None else group.zeta_max
        if not math.isfinite(mu) or mu <= 0.0:
            raise ValueError(f"noise group {group.name!r} zeta_mu must be finite and positive")
        if not isinstance(max_n, int) or isinstance(max_n, bool) or max_n < 1:
            raise ValueError(f"noise group {group.name!r} zeta_max must be a positive integer")
        for index in range(group.start, group.stop):
            if occupied[index]:
                raise ValueError(f"noise group {group.name!r} overlaps another group at action index {index}")
            occupied[index] = True
        normalized.append(dataclasses.replace(group, zeta_mu=float(mu), zeta_max=max_n))

    uncovered = [index for index, is_occupied in enumerate(occupied) if not is_occupied]
    if uncovered:
        raise ValueError(f"noise groups must cover every action dimension; uncovered indices: {uncovered}")
    return tuple(normalized)


def _truncated_zeta_cdf(mu: float, max_n: int, device: torch.device) -> torch.Tensor:
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    probabilities = ns.pow(-mu)
    return (probabilities / probabilities.sum()).cumsum(dim=0)


def _group_dict(group: ActionNoiseGroup) -> dict[str, Any]:
    return dataclasses.asdict(group)


class FlashSACTorchBridge(FlashSACAgent):
    """Pinned FlashSAC agent with a Torch-native Isaac interaction boundary."""

    def __init__(
        self,
        observation_space: gym.spaces.Space[Any],
        action_space: gym.spaces.Space[Any],
        env_info: dict[str, Any],
        cfg: FlashSACConfig,
        *,
        noise_groups: Sequence[ActionNoiseGroup] = (),
        restore_rng_state_on_load: bool = True,
    ) -> None:
        super().__init__(observation_space, action_space, env_info, cfg)
        self._restore_rng_state_on_load = restore_rng_state_on_load
        self._noise_groups = _normalize_noise_groups(
            noise_groups,
            action_dim=self._action_dim,
            default_mu=self._cfg.actor_noise_zeta_mu,
            default_max=self._cfg.actor_noise_zeta_max,
        )
        self._noise_cdfs = tuple(
            _truncated_zeta_cdf(group.zeta_mu, group.zeta_max, self._device)  # type: ignore[arg-type]
            for group in self._noise_groups
        )
        scale = torch.ones(self._action_dim, dtype=torch.float32, device=self._device)
        for group in self._noise_groups:
            scale[group.start : group.stop] = group.scale
        self._group_noise_scale = scale
        self.reset_exploration()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def noise_groups(self) -> tuple[ActionNoiseGroup, ...]:
        return self._noise_groups

    @property
    def replay_size(self) -> int:
        return len(self._replay_buffer)

    @torch.no_grad()
    def reset_exploration(
        self,
        batch_size: int | None = None,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Drop repeated noise after an environment reset or evaluation pass.

        With no batch size, the next call lazily initializes from its observation
        batch.  Supplying a batch size resets the full batch.  ``env_ids`` instead
        refreshes only those rows whose Isaac environments auto-reset; group-level
        repetition clocks remain shared, matching upstream, but no cached noise
        vector crosses an episode boundary.
        """

        if batch_size is not None and env_ids is not None:
            raise ValueError("batch_size and env_ids are mutually exclusive")
        if env_ids is not None:
            if not isinstance(env_ids, torch.Tensor):
                raise TypeError("env_ids must be a torch.Tensor")
            if env_ids.device != self._device:
                raise ValueError(f"env_ids is on {env_ids.device}, expected {self._device}")
            if env_ids.ndim != 1 or env_ids.dtype != torch.long:
                raise ValueError("env_ids must be a one-dimensional torch.long tensor")
            if self._cached_noise.ndim != 2 or self._cached_noise.shape[1] != self._action_dim:
                raise RuntimeError("exploration batch has not been initialized")
            if env_ids.numel() > 0:
                fresh = torch.randn(
                    (env_ids.numel(), self._action_dim),
                    dtype=self._cached_noise.dtype,
                    device=self._device,
                )
                self._cached_noise.index_copy_(0, env_ids, fresh)
            return
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive when provided")
        shape = (0, self._action_dim) if batch_size is None else (batch_size, self._action_dim)
        self._cached_noise = torch.zeros(shape, dtype=torch.float32, device=self._device)
        num_groups = len(self._noise_groups)
        self._cur_noise_repeat_count = torch.zeros(num_groups, dtype=torch.int32, device=self._device)
        self._cur_noise_repeat_n = torch.ones(num_groups, dtype=torch.int32, device=self._device)

    def _validate_observations(self, observations: Any) -> torch.Tensor:
        if not isinstance(observations, torch.Tensor):
            raise TypeError(
                f"next_observation must be a torch.Tensor; got {type(observations).__name__}. "
                "The Torch bridge intentionally rejects NumPy observations."
            )
        if observations.device != self._device:
            raise ValueError(f"next_observation is on {observations.device}, expected {self._device}")
        if observations.ndim != 2 or observations.shape[1] != self._critic_observation_dim:
            raise ValueError(
                "next_observation must have shape [num_envs, observation_dim]; "
                f"expected second dimension {self._critic_observation_dim}, got {tuple(observations.shape)}"
            )
        return observations.to(dtype=torch.float32)

    def _resolve_runtime_noise_scale(
        self,
        noise_scale: torch.Tensor | None,
        action_shape: torch.Size,
    ) -> torch.Tensor:
        base_scale = self._group_noise_scale
        if noise_scale is None:
            return base_scale
        if not isinstance(noise_scale, torch.Tensor):
            raise TypeError("noise_scale must be a torch.Tensor so it stays on the interaction device")
        if noise_scale.device != self._device:
            raise ValueError(f"noise_scale is on {noise_scale.device}, expected {self._device}")
        allowed_shapes = {(self._action_dim,), tuple(action_shape)}
        if tuple(noise_scale.shape) not in allowed_shapes:
            raise ValueError(
                f"noise_scale must have shape ({self._action_dim},) or {tuple(action_shape)}, "
                f"got {tuple(noise_scale.shape)}"
            )
        return base_scale * noise_scale.to(dtype=torch.float32)

    @torch.no_grad()
    def _sample_grouped_actions(
        self,
        observations: torch.Tensor,
        *,
        temperature: float,
        noise_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        mean, std = self._actor.apply(
            "get_mean_and_std",
            observations=observations,
            training=False,
        )
        if temperature == 0.0:
            return torch.tanh(mean)

        if self._cached_noise.shape != mean.shape:
            self.reset_exploration(batch_size=mean.shape[0])

        next_noise = self._cached_noise.clone()
        next_counts = self._cur_noise_repeat_count.clone()
        next_durations = self._cur_noise_repeat_n.clone()
        for index, (group, cdf) in enumerate(zip(self._noise_groups, self._noise_cdfs, strict=True)):
            reinitialize = (self._cur_noise_repeat_count[index] == 0) | (
                self._cur_noise_repeat_count[index] >= self._cur_noise_repeat_n[index]
            )
            candidate_noise = torch.randn_like(mean[:, group.start : group.stop])
            uniform = torch.rand((), device=self._device)
            candidate_duration = torch.searchsorted(cdf, uniform, right=False).to(torch.int32) + 1
            next_noise[:, group.start : group.stop] = torch.where(
                reinitialize,
                candidate_noise,
                self._cached_noise[:, group.start : group.stop],
            )
            next_durations[index] = torch.where(
                reinitialize,
                candidate_duration,
                self._cur_noise_repeat_n[index],
            )
            next_counts[index] = torch.where(
                reinitialize,
                torch.ones_like(self._cur_noise_repeat_count[index]),
                self._cur_noise_repeat_count[index] + 1,
            )

        self._cached_noise = next_noise
        self._cur_noise_repeat_count = next_counts
        self._cur_noise_repeat_n = next_durations
        effective_scale = self._resolve_runtime_noise_scale(noise_scale, mean.shape)
        return torch.tanh(mean + std * self._cached_noise * effective_scale * temperature)

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Any],
        training: bool,
        *,
        noise_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return actions on the agent device without a NumPy/CPU conversion."""

        del interaction_step  # retained for the upstream BaseAgent API
        observations = self._validate_observations(prev_transition["next_observation"])
        if self._cfg.asymmetric_observation:
            observations = observations[:, : self._actor_observation_dim]
        return self._sample_grouped_actions(
            observations,
            temperature=1.0 if training else 0.0,
            noise_scale=noise_scale,
        )

    def process_transition(self, transition: MutableMapping[str, Any]) -> None:
        assert_transition_tensors(
            transition,
            device=self._device,
            observation_dim=self._critic_observation_dim,
            action_dim=self._action_dim,
        )
        super().process_transition(transition)

    def _bridge_checkpoint_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "version": BRIDGE_CHECKPOINT_VERSION,
            "upstream_commit": FLASH_SAC_COMMIT,
            "action_dim": self._action_dim,
            "noise_groups": [_group_dict(group) for group in self._noise_groups],
            "group_noise_scale": self._group_noise_scale,
            "cached_noise": self._cached_noise,
            "noise_repeat_count": self._cur_noise_repeat_count,
            "noise_repeat_n": self._cur_noise_repeat_n,
            "cpu_rng_state": torch.get_rng_state(),
        }
        if self._device.type == "cuda":
            state["device_rng_state"] = torch.cuda.get_rng_state(self._device)
        return state

    def save(self, path: str) -> None:
        super().save(path)
        torch.save(self._bridge_checkpoint_state(), os.path.join(path, BRIDGE_STATE_FILENAME))

    def load(self, path: str) -> None:
        super().load(path)
        bridge_path = os.path.join(path, BRIDGE_STATE_FILENAME)
        if not os.path.exists(bridge_path):
            warnings.warn(
                f"{bridge_path} is absent; loaded an upstream-only checkpoint and reset exploration state.",
                stacklevel=2,
            )
            self.reset_exploration()
            return

        state = torch.load(bridge_path, map_location=self._device, weights_only=True)
        if state.get("version") != BRIDGE_CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported Torch bridge checkpoint version {state.get('version')!r}; "
                f"expected {BRIDGE_CHECKPOINT_VERSION}"
            )
        if state.get("upstream_commit") != FLASH_SAC_COMMIT:
            raise ValueError(
                f"checkpoint targets upstream commit {state.get('upstream_commit')!r}, "
                f"expected {FLASH_SAC_COMMIT}"
            )
        if state.get("action_dim") != self._action_dim:
            raise ValueError(
                f"checkpoint action_dim={state.get('action_dim')}, current action_dim={self._action_dim}"
            )
        checkpoint_groups = state.get("noise_groups")
        current_groups = [_group_dict(group) for group in self._noise_groups]
        if checkpoint_groups != current_groups:
            raise ValueError(
                "checkpoint noise groups differ from the current bridge configuration; "
                f"checkpoint={checkpoint_groups}, current={current_groups}"
            )

        cached_noise = state["cached_noise"].to(device=self._device, dtype=torch.float32)
        counts = state["noise_repeat_count"].to(device=self._device, dtype=torch.int32)
        durations = state["noise_repeat_n"].to(device=self._device, dtype=torch.int32)
        if cached_noise.ndim != 2 or cached_noise.shape[1] != self._action_dim:
            raise ValueError(f"invalid cached_noise shape in checkpoint: {tuple(cached_noise.shape)}")
        expected_group_shape = (len(self._noise_groups),)
        if counts.shape != expected_group_shape or durations.shape != expected_group_shape:
            raise ValueError(
                "invalid grouped repetition state in checkpoint: "
                f"counts={tuple(counts.shape)}, durations={tuple(durations.shape)}, "
                f"expected={expected_group_shape}"
            )
        self._cached_noise = cached_noise
        self._cur_noise_repeat_count = counts
        self._cur_noise_repeat_n = durations
        self._group_noise_scale = state["group_noise_scale"].to(device=self._device, dtype=torch.float32)

        if self._restore_rng_state_on_load:
            torch.set_rng_state(state["cpu_rng_state"].cpu())
            if self._device.type == "cuda" and "device_rng_state" in state:
                torch.cuda.set_rng_state(state["device_rng_state"].cpu(), self._device)


def build_agent_config(**overrides: Any) -> FlashSACConfig:
    """Build a complete, conservative FlashSAC config for bridge smoke runs.

    Production training should explicitly override buffer sizes, schedules, AMP,
    and compilation.  Unknown keys are rejected by the upstream dataclass.
    """

    values: dict[str, Any] = {
        "seed": 0,
        "normalize_reward": False,
        "normalized_G_max": 5.0,
        "asymmetric_observation": False,
        "device_type": "cuda:0" if torch.cuda.is_available() else "cpu",
        "buffer_max_length": 1024,
        "buffer_min_length": 32,
        "buffer_device_type": "cuda:0" if torch.cuda.is_available() else "cpu",
        "sample_batch_size": 32,
        "learning_rate_init": 3e-4,
        "learning_rate_peak": 3e-4,
        "learning_rate_end": 1.5e-4,
        "learning_rate_warmup_rate": 0.0,
        "learning_rate_warmup_step": 1,
        "learning_rate_decay_rate": 1.0,
        "learning_rate_decay_step": 1_000_000,
        "actor_num_blocks": 1,
        "actor_hidden_dim": 32,
        "actor_bc_alpha": 0.0,
        "actor_noise_zeta_mu": 2.0,
        "actor_noise_zeta_max": 16,
        "actor_update_period": 2,
        "critic_num_blocks": 1,
        "critic_hidden_dim": 64,
        "critic_num_bins": 51,
        "critic_min_v": -5.0,
        "critic_max_v": 5.0,
        "critic_target_update_tau": 0.01,
        "temp_initial_value": 0.01,
        "temp_target_sigma": 0.15,
        "temp_target_entropy": 0.0,
        "gamma": 0.99,
        "n_step": 1,
        "use_compile": False,
        "compile_mode": "default",
        "use_amp": False,
        "load_optimizer": True,
        "load_reward_normalizer": False,
    }
    values.update(overrides)
    return FlashSACConfig(**values)


__all__ = [
    "ActionNoiseGroup",
    "BRIDGE_CHECKPOINT_VERSION",
    "BRIDGE_STATE_FILENAME",
    "FLASH_SAC_COMMIT",
    "FlashSACTorchBridge",
    "assert_transition_tensors",
    "build_agent_config",
]
