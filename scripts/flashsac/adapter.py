#!/usr/bin/env python3
"""Torch-only PickTool adapter for FlashSAC and Isaac Lab DirectRLEnv.

Isaac Lab's ``DirectRLEnv.step`` resets completed sub-environments before it
returns.  Its returned observation is therefore the first observation of the
next episode, not the final observation of the transition that just ended.
That distinction is essential for SAC: a time-limit transition should
bootstrap from the real time-limit state, never from the next episode's reset
state.

This adapter temporarily intercepts ``_reset_idx`` during one call to
``step``.  Immediately before the reset it captures the 115-dimensional
``policy`` observation, then lets the original reset proceed unchanged.  The
rollout observation returned by ``step`` remains the reset observation, while
``info["transition_next_observation"]`` contains the correct replay-buffer
observation.  No NumPy conversion or host-device transfer occurs on the data
path.

The adapter is deliberately task-specific:

* policy observation: 115 floats (the duplicate ``critic`` group is ignored)
* action: 21 normalized values in [-1, 1]
* raw Gymnasium ``terminated`` and ``truncated`` flags are preserved
* timeout transitions bootstrap from their captured final observation
* all original Isaac Lab extras are retained

Call :func:`make_pick_tool_env` only after ``isaaclab.app.AppLauncher`` has
started Isaac Sim.  Unit tests can wrap a small Torch-only fake environment and
do not require Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch


PICK_TOOL_ENV_ID = "Pick-Tool-Token-Direct-v0"
POLICY_OBSERVATION_DIM = 115
ACTION_DIM = 21

# These values already exist under PickTool's ``extras["log"]``.  The adapter
# exposes a reference-only subset at ``info["strict_metrics"]`` for loggers; it
# never recomputes a weaker success proxy.
STRICT_METRIC_KEYS = (
    "success_frac",
    "lift_ge_20cm_frac",
    "clearance_max",
    "is_grasped_phase_frac",
    "grasp_quality_mean",
    "hold_quality_mean",
    "object_contact_force_max",
    "tactile_terminate_fraction",
)

_ADAPTER_INFO_KEYS = frozenset(
    {
        "raw_terminated",
        "raw_truncated",
        "episode_done",
        "bootstrap_mask",
        "transition_next_observation",
        "final_observation",
        "final_observation_mask",
        "reset_observation",
        "reset_observation_mask",
        "strict_metrics",
        "actor_observation_size",
        "asymmetric_obs",
        "auto_reset",
    }
)


class DirectEnvLike(Protocol):
    """Small structural interface used by :class:`PickToolIsaacLabAdapter`."""

    @property
    def unwrapped(self) -> Any: ...

    def reset(self, **kwargs: Any) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]: ...

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        Mapping[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DoneSignals:
    """Raw episode boundaries and the SAC bootstrap mask.

    ``bootstrap_mask`` is zero only for genuine MDP termination.  A timeout has
    mask one because the adapter supplies its true final observation.
    """

    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_done: torch.Tensor
    bootstrap_mask: torch.Tensor


def _require_vector(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape (num_envs,), got {tuple(value.shape)}")


def classify_done(terminated: torch.Tensor, truncated: torch.Tensor) -> DoneSignals:
    """Validate raw Gymnasium done flags and derive SAC bootstrap semantics."""

    _require_vector("terminated", terminated)
    _require_vector("truncated", truncated)
    if terminated.shape != truncated.shape:
        raise ValueError(
            "terminated and truncated shapes differ: "
            f"{tuple(terminated.shape)} vs {tuple(truncated.shape)}"
        )
    if terminated.device != truncated.device:
        raise ValueError(
            f"terminated is on {terminated.device}, truncated is on {truncated.device}"
        )
    if terminated.dtype is not torch.bool or truncated.dtype is not torch.bool:
        raise TypeError("terminated and truncated must both have dtype torch.bool")

    episode_done = terminated | truncated
    return DoneSignals(
        terminated=terminated,
        truncated=truncated,
        episode_done=episode_done,
        bootstrap_mask=(~terminated).to(dtype=torch.float32),
    )


def extract_policy_observation(
    observations: Mapping[str, Any],
    *,
    expected_dim: int = POLICY_OBSERVATION_DIM,
    expected_num_envs: int | None = None,
    expected_device: torch.device | None = None,
) -> torch.Tensor:
    """Return only PickTool's policy tensor, never policy+critic concatenation."""

    if not isinstance(observations, Mapping):
        raise TypeError(
            "Isaac Lab observations must be a mapping containing 'policy'; "
            f"got {type(observations).__name__}"
        )
    if "policy" not in observations:
        raise KeyError("Isaac Lab observations do not contain a 'policy' group")
    policy = observations["policy"]
    if not isinstance(policy, torch.Tensor):
        raise TypeError(f"policy observation must be a torch.Tensor, got {type(policy).__name__}")
    if policy.ndim != 2 or policy.shape[1] != expected_dim:
        raise ValueError(
            f"policy observation must have shape (num_envs, {expected_dim}), "
            f"got {tuple(policy.shape)}"
        )
    if expected_num_envs is not None and policy.shape[0] != expected_num_envs:
        raise ValueError(
            f"policy observation has {policy.shape[0]} envs, expected {expected_num_envs}"
        )
    if expected_device is not None and policy.device != expected_device:
        raise ValueError(f"policy observation is on {policy.device}, expected {expected_device}")
    if not policy.dtype.is_floating_point:
        raise TypeError(f"policy observation must be floating point, got {policy.dtype}")
    return policy


def strict_metrics_from_extras(extras: Mapping[str, Any]) -> dict[str, Any]:
    """Select references to the task's ground-truth log values, if present."""

    log = extras.get("log")
    if not isinstance(log, Mapping):
        return {}
    return {key: log[key] for key in STRICT_METRIC_KEYS if key in log}


def build_replay_transition(
    observation: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    info: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Build the six standard FlashSAC buffer fields with correct final obs.

    FlashSAC intentionally uses only ``terminated`` in its TD target while its
    n-step buffer also uses ``truncated`` to stop returns at episode boundaries.
    Keeping both raw signals and supplying a real timeout observation therefore
    gives the intended time-limit bootstrap behavior.
    """

    signals = classify_done(terminated, truncated)
    next_observation = info.get("transition_next_observation")
    if not isinstance(next_observation, torch.Tensor):
        raise KeyError("info has no tensor 'transition_next_observation'")
    if observation.shape != next_observation.shape:
        raise ValueError(
            "observation and transition_next_observation shapes differ: "
            f"{tuple(observation.shape)} vs {tuple(next_observation.shape)}"
        )
    if observation.device != next_observation.device:
        raise ValueError("observation and transition_next_observation devices differ")
    if action.ndim != 2 or action.shape != (observation.shape[0], ACTION_DIM):
        raise ValueError(
            f"action must have shape ({observation.shape[0]}, {ACTION_DIM}), got {tuple(action.shape)}"
        )
    if reward.shape != signals.terminated.shape:
        raise ValueError(
            f"reward must have shape {tuple(signals.terminated.shape)}, got {tuple(reward.shape)}"
        )
    return {
        "observation": observation,
        "action": action,
        "reward": reward,
        "terminated": signals.terminated,
        "truncated": signals.truncated,
        "next_observation": next_observation,
    }


def _declared_dimension(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    shape = getattr(value, "shape", None)
    if shape and len(shape) == 1:
        return int(shape[0])
    return None


def _set_cfg_override(cfg: Any, dotted_name: str, value: Any) -> None:
    target = cfg
    pieces = dotted_name.split(".")
    for piece in pieces[:-1]:
        if not hasattr(target, piece):
            raise AttributeError(f"unknown config path '{dotted_name}' (missing '{piece}')")
        target = getattr(target, piece)
    leaf = pieces[-1]
    if not hasattr(target, leaf):
        raise AttributeError(f"unknown config override '{dotted_name}'")
    setattr(target, leaf, value)


class PickToolIsaacLabAdapter:
    """A batched, Torch-only adapter around the PickTool DirectRLEnv."""

    observation_dim = POLICY_OBSERVATION_DIM
    action_dim = ACTION_DIM

    def __init__(
        self,
        env: DirectEnvLike,
        *,
        action_clip: float = 1.0,
        strict: bool = True,
        require_cuda: bool = True,
        validate_finite: bool = False,
    ) -> None:
        if not np.isfinite(action_clip) or not 0.0 < action_clip <= 1.0:
            raise ValueError("action_clip must be finite and in (0, 1]")

        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.num_envs = int(getattr(self.unwrapped, "num_envs"))
        self.device = torch.device(str(getattr(self.unwrapped, "device")))
        self.action_clip = float(action_clip)
        self.strict = bool(strict)
        self.validate_finite = bool(validate_finite)
        self.max_episode_steps = int(getattr(self.unwrapped, "max_episode_length", 0))

        if self.num_envs < 1:
            raise ValueError("environment must contain at least one sub-environment")
        if require_cuda and self.device.type != "cuda":
            raise ValueError(f"PickTool FlashSAC training requires a CUDA environment, got {self.device}")
        if not callable(getattr(self.unwrapped, "_reset_idx", None)):
            raise TypeError("environment does not expose DirectRLEnv._reset_idx")
        if not callable(getattr(self.unwrapped, "_get_observations", None)):
            raise TypeError("environment does not expose DirectRLEnv._get_observations")

        cfg = getattr(self.unwrapped, "cfg", None)
        if cfg is not None:
            declared_obs = _declared_dimension(getattr(cfg, "observation_space", None))
            declared_action = _declared_dimension(getattr(cfg, "action_space", None))
            if declared_obs is not None and declared_obs != self.observation_dim:
                raise ValueError(
                    f"environment declares {declared_obs} policy observations, expected {self.observation_dim}"
                )
            if declared_action is not None and declared_action != self.action_dim:
                raise ValueError(
                    f"environment declares {declared_action} actions, expected {self.action_dim}"
                )
            if strict and getattr(cfg, "observation_noise_model", None) is not None:
                raise ValueError(
                    "terminal-observation capture requires PickTool observation_noise_model=None; "
                    "otherwise reset and terminal observations would use different noise semantics"
                )

        # Gym spaces are metadata for the unmodified FlashSAC agent.  Environment
        # interaction and random-action sampling below remain Torch-only.
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.single_action_space = gym.spaces.Box(
            low=-self.action_clip,
            high=self.action_clip,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    @property
    def env_info(self) -> dict[str, Any]:
        """Metadata expected by the upstream FlashSAC agent constructor."""

        return {
            "actor_observation_size": (self.observation_dim,),
            "asymmetric_obs": False,
            "auto_reset": True,
        }

    def _validate_policy(self, observations: Mapping[str, Any]) -> torch.Tensor:
        policy = extract_policy_observation(
            observations,
            expected_num_envs=self.num_envs,
            expected_device=self.device,
        )
        if self.validate_finite and not bool(torch.isfinite(policy).all()):
            raise FloatingPointError("policy observation contains NaN or infinity")
        return policy

    def _validate_action(self, action: torch.Tensor) -> torch.Tensor:
        if not isinstance(action, torch.Tensor):
            raise TypeError(
                "PickToolIsaacLabAdapter accepts torch actions only; "
                f"got {type(action).__name__}"
            )
        if action.ndim != 2 or action.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"action must have shape ({self.num_envs}, {self.action_dim}), "
                f"got {tuple(action.shape)}"
            )
        if not action.dtype.is_floating_point:
            raise TypeError(f"action must be floating point, got {action.dtype}")
        if action.device != self.device:
            raise ValueError(
                f"action is on {action.device}, expected {self.device}; "
                "the adapter does not perform hidden host/device transfers"
            )
        action = action.to(dtype=torch.float32)
        if self.validate_finite and not bool(torch.isfinite(action).all()):
            raise FloatingPointError("action contains NaN or infinity")
        return action.clamp(-self.action_clip, self.action_clip)

    def sample_random_actions(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample normalized actions directly on the environment device."""

        actions = torch.empty(
            (self.num_envs, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        return actions.uniform_(-self.action_clip, self.action_clip, generator=generator)

    def _copy_extras(self, extras: Any) -> dict[str, Any]:
        if not isinstance(extras, Mapping):
            raise TypeError(f"Isaac Lab extras must be a mapping, got {type(extras).__name__}")
        collisions = _ADAPTER_INFO_KEYS.intersection(extras)
        if collisions and self.strict:
            names = ", ".join(sorted(collisions))
            raise KeyError(f"Isaac Lab extras collide with adapter keys: {names}")
        return dict(extras)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        randomize_episode_lengths: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        observations, extras = self.env.reset(seed=seed, options=options)
        policy = self._validate_policy(observations)

        if randomize_episode_lengths:
            episode_length = getattr(self.unwrapped, "episode_length_buf", None)
            if not isinstance(episode_length, torch.Tensor) or self.max_episode_steps < 1:
                raise TypeError("environment does not expose a valid episode_length_buf")
            episode_length.random_(0, self.max_episode_steps)

        info = self._copy_extras(extras)
        info.update(self.env_info)
        info["strict_metrics"] = strict_metrics_from_extras(info)
        return policy, info

    def _step_with_terminal_capture(
        self, action: torch.Tensor
    ) -> tuple[
        tuple[
            Mapping[str, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            dict[str, Any],
        ],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Run one DirectRLEnv step and capture observation immediately pre-reset."""

        captures: list[tuple[torch.Tensor, torch.Tensor]] = []
        unwrapped = self.unwrapped
        original_reset = unwrapped._reset_idx
        instance_dict: MutableMapping[str, Any] = vars(unwrapped)
        had_instance_override = "_reset_idx" in instance_dict
        previous_instance_override = instance_dict.get("_reset_idx")

        def capture_then_reset(env_ids: Any) -> Any:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            terminal_dict = unwrapped._get_observations()
            terminal_policy = self._validate_policy(terminal_dict).detach().clone()
            captures.append((ids.detach().clone(), terminal_policy))
            return original_reset(env_ids)

        # Assigning the closure to the instance deliberately gives it the same
        # one-argument call shape used by ``self._reset_idx(env_ids)``.
        instance_dict["_reset_idx"] = capture_then_reset
        try:
            result = self.env.step(action)
        finally:
            if had_instance_override:
                instance_dict["_reset_idx"] = previous_instance_override
            else:
                del instance_dict["_reset_idx"]
        return result, captures

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        action = self._validate_action(action)
        result, captures = self._step_with_terminal_capture(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("DirectRLEnv.step must return (obs, reward, terminated, truncated, extras)")
        observations, reward, terminated, truncated, extras = result
        next_policy = self._validate_policy(observations)

        _require_vector("reward", reward)
        if reward.shape != (self.num_envs,):
            raise ValueError(f"reward must have shape ({self.num_envs},), got {tuple(reward.shape)}")
        if reward.device != self.device or not reward.dtype.is_floating_point:
            raise TypeError(
                f"reward must be a floating tensor on {self.device}, got {reward.dtype} on {reward.device}"
            )
        if self.validate_finite and not bool(torch.isfinite(reward).all()):
            raise FloatingPointError("reward contains NaN or infinity")

        # DirectRLEnv owns and reuses its done buffers.  Clone them before
        # exposing the tensors to a replay pipeline that may retain references.
        terminated = terminated.detach().clone()
        truncated = truncated.detach().clone()
        signals = classify_done(terminated, truncated)
        if signals.terminated.shape != (self.num_envs,):
            raise ValueError(
                f"done tensors must have shape ({self.num_envs},), got {tuple(signals.terminated.shape)}"
            )
        if signals.terminated.device != self.device:
            raise ValueError(
                f"done tensors are on {signals.terminated.device}, expected {self.device}"
            )

        # The rollout continues from next_policy.  Only the replay-buffer view
        # substitutes terminal observations for auto-reset rows.
        transition_next = next_policy
        final_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if captures:
            transition_next = next_policy.clone()
            for env_ids, terminal_policy in captures:
                transition_next.index_copy_(
                    0,
                    env_ids,
                    terminal_policy.index_select(0, env_ids),
                )
                final_mask[env_ids] = True

        if self.strict and not torch.equal(final_mask, signals.episode_done):
            missing = signals.episode_done & ~final_mask
            unexpected = final_mask & ~signals.episode_done
            raise RuntimeError(
                "terminal observation capture did not match DirectRLEnv done flags "
                f"(missing={int(missing.sum())}, unexpected={int(unexpected.sum())})"
            )

        info = self._copy_extras(extras)
        info.update(
            {
                **self.env_info,
                "raw_terminated": signals.terminated,
                "raw_truncated": signals.truncated,
                "episode_done": signals.episode_done,
                "bootstrap_mask": signals.bootstrap_mask,
                "transition_next_observation": transition_next,
                "final_observation": transition_next,
                "final_observation_mask": final_mask,
                "reset_observation": next_policy,
                "reset_observation_mask": signals.episode_done,
            }
        )
        info["strict_metrics"] = strict_metrics_from_extras(info)
        return next_policy, reward, signals.terminated, signals.truncated, info

    def close(self) -> None:
        self.env.close()


def make_pick_tool_env(
    *,
    num_envs: int,
    device: str,
    seed: int,
    render_mode: str | None = None,
    cfg_overrides: Mapping[str, Any] | None = None,
    action_clip: float = 1.0,
    strict: bool = True,
    validate_finite: bool = False,
) -> PickToolIsaacLabAdapter:
    """Create the registered PickTool task after Isaac Sim has been launched."""

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if torch.device(device).type != "cuda":
        raise ValueError(f"PickTool FlashSAC requires a CUDA device, got {device}")

    # Imports are intentionally lazy: Isaac Lab modules may only be imported
    # after AppLauncher has initialized the simulation application.
    import importlib

    importlib.import_module("xhand_inhand.tasks")
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(PICK_TOOL_ENV_ID, device=device, num_envs=num_envs)
    cfg.seed = seed
    for name, value in (cfg_overrides or {}).items():
        _set_cfg_override(cfg, name, value)

    env = gym.make(PICK_TOOL_ENV_ID, cfg=cfg, render_mode=render_mode)
    return PickToolIsaacLabAdapter(
        env,
        action_clip=action_clip,
        strict=strict,
        require_cuda=True,
        validate_finite=validate_finite,
    )


__all__ = [
    "ACTION_DIM",
    "PICK_TOOL_ENV_ID",
    "POLICY_OBSERVATION_DIM",
    "STRICT_METRIC_KEYS",
    "DoneSignals",
    "PickToolIsaacLabAdapter",
    "build_replay_transition",
    "classify_done",
    "extract_policy_observation",
    "make_pick_tool_env",
    "strict_metrics_from_extras",
]
