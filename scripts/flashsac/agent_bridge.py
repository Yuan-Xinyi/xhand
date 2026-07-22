"""Torch-only integration boundary for the pinned FlashSAC agent.

The audited fork at ``third_party/FlashSAC`` retains the upstream architecture
and adds the minimal per-action log-probability/update seam needed by this
bridge.  This module fixes the interaction boundary needed by Isaac Lab:

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
import hashlib
import io
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
import torch.nn.functional as F


FLASH_SAC_COMMIT = "87edc9061150ae9e962dd84e6544e27a1554b3ab"
FLASH_SAC_FORK_COMMIT = "5ecf331fa11cd457dd39018b3d68af571b257666"
FLASH_SAC_COMPATIBLE_FORK_COMMITS = (
    "4f4daf6f08e112c5d93fd8537eb4d6095d482134",
    FLASH_SAC_FORK_COMMIT,
)
BRIDGE_STATE_FILENAME = "torch_bridge_state.pt"
FROZEN_LIFT_ACTOR_FILENAME = "frozen_lift_actor.pt"
PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_NAME = "public_latch_frozen_actor_v1"
BRIDGE_CHECKPOINT_VERSION = 2
_SUPPORTED_BRIDGE_CHECKPOINT_VERSIONS = (1, BRIDGE_CHECKPOINT_VERSION)
_COMPILED_STATE_PREFIX = "_orig_mod."
_ACTOR_ACTION_OUTPUT_KEYS = (
    "predictor.mean_w.w.weight",
    "predictor.mean_bias",
    "predictor.std_w.w.weight",
    "predictor.std_bias",
)

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
    _update_networks,
)
from flash_rl.agents.flashSAC.network import FlashSACActor  # noqa: E402
from flash_rl.agents.flashSAC.update import (  # noqa: E402
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
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


@dataclasses.dataclass(frozen=True)
class ActionAuthorityRule:
    """Gate one action slice from a public binary observation feature.

    Dimensions not covered by a rule always remain active.  Overlapping rules
    compose with logical AND, so every rule covering a dimension must grant
    authority before that action can reach the environment or an SAC target.
    """

    name: str
    start: int
    stop: int
    observation_index: int
    active_value: float


@dataclasses.dataclass(frozen=True)
class PublicLatchFrozenActorRouter:
    """Use a learned close slice before latch and a frozen full actor after it."""

    name: str
    observation_index: int
    trainable_start: int
    trainable_stop: int
    close_value: float = 0.0
    frozen_value: float = 1.0


def _update_networks_with_action_authority(
    *,
    batch: dict[str, torch.Tensor],
    actor: Any,
    critic: Any,
    target_critic: Any,
    temperature: Any,
    cfg: FlashSACConfig,
    do_actor_update: bool,
    device: torch.device,
    grad_scaler: torch.amp.GradScaler,
    actor_action_active: torch.Tensor,
    actor_next_action_active: torch.Tensor,
    next_action_override: torch.Tensor | None = None,
    next_action_override_rows: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run the pinned update with caller-authored policy action authority.

    The upstream default path remains untouched when no authority observation
    is configured.  This explicit branch passes generic row masks into the
    upstream actor/critic update functions; upstream never needs to know which
    observation feature authored them.
    """

    actor_has_authority = do_actor_update and bool(actor_action_active.any())
    if do_actor_update and actor_has_authority:
        actor_info = update_actor(
            actor=actor,
            critic=critic,
            temperature=temperature,
            batch=batch,  # type: ignore[arg-type]
            bc_alpha=cfg.actor_bc_alpha,
            device=device,
            use_amp=cfg.use_amp,
            grad_scaler=grad_scaler,
            action_active=actor_action_active,
        )
        actor_info["actor/updated"] = torch.ones(
            (), dtype=batch["action"].dtype, device=device
        )
        target_entropy = cfg.temp_target_entropy
        if actor_action_active.ndim == 2:
            active_counts = actor_action_active.sum(dim=-1)
            active_rows = active_counts > 0
            mean_active_dimensions = active_counts[active_rows].float().mean()
            target_entropy *= float(
                mean_active_dimensions / actor_action_active.shape[1]
            )
        temperature_info = update_temperature(
            temperature=temperature,
            entropy=actor_info["actor/entropy"],
            target_entropy=target_entropy,
        )
    elif do_actor_update:
        # No policy action exists in an all-inactive batch. Report finite,
        # exact-zero actor metrics without advancing actor/temperature
        # parameters, Adam state, or LR schedulers.
        zero = torch.zeros((), dtype=batch["action"].dtype, device=device)
        actor_info = {
            "actor/loss": zero,
            "actor/entropy": zero,
            "actor/mean_action": zero,
            "actor/updated": zero,
        }
        temperature_info = {}
    else:
        actor_info = {}
        temperature_info = {}

    critic_info = update_critic(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,
        batch=batch,  # type: ignore[arg-type]
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
        num_bins=cfg.critic_num_bins,
        gamma=cfg.gamma,
        n_step=cfg.n_step,
        device=device,
        use_amp=cfg.use_amp,
        grad_scaler=grad_scaler,
        next_action_active=actor_next_action_active,
        next_action_override=next_action_override,
        next_action_override_rows=next_action_override_rows,
    )
    target_info = update_target_network(target_network=target_critic)
    return {
        **actor_info,
        **critic_info,
        **target_info,
        **temperature_info,
    }


_REQUIRED_TRANSITION_KEYS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)
_MASKED_REPLAY_VALID_KEY = "_bridge_replay_valid_mask"


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


def _normalize_action_authority_rules(
    rules: Sequence[ActionAuthorityRule],
    *,
    action_dim: int,
    actor_observation_dim: int,
) -> tuple[ActionAuthorityRule, ...]:
    normalized: list[ActionAuthorityRule] = []
    names: set[str] = set()
    for rule in rules:
        if not isinstance(rule, ActionAuthorityRule):
            raise TypeError("action authority rules must be ActionAuthorityRule instances")
        if not rule.name or rule.name in names:
            raise ValueError(
                "action authority rule names must be non-empty and unique; "
                f"got {rule.name!r}"
            )
        names.add(rule.name)
        if (
            not isinstance(rule.start, int)
            or isinstance(rule.start, bool)
            or not isinstance(rule.stop, int)
            or isinstance(rule.stop, bool)
            or not 0 <= rule.start < rule.stop <= action_dim
        ):
            raise ValueError(
                f"action authority rule {rule.name!r} has invalid slice "
                f"[{rule.start}:{rule.stop}] for action_dim={action_dim}"
            )
        if (
            not isinstance(rule.observation_index, int)
            or isinstance(rule.observation_index, bool)
            or not 0 <= rule.observation_index < actor_observation_dim
        ):
            raise ValueError(
                f"action authority rule {rule.name!r} has invalid actor-observation "
                f"index {rule.observation_index!r}"
            )
        if (
            not isinstance(rule.active_value, (int, float))
            or isinstance(rule.active_value, bool)
            or float(rule.active_value) not in (0.0, 1.0)
        ):
            raise ValueError(
                f"action authority rule {rule.name!r} active_value must be 0.0 or 1.0"
            )
        normalized.append(dataclasses.replace(rule, active_value=float(rule.active_value)))
    return tuple(normalized)


def _normalize_public_latch_router(
    router: PublicLatchFrozenActorRouter | None,
    *,
    action_dim: int,
    actor_observation_dim: int,
    authority_rules: Sequence[ActionAuthorityRule],
) -> PublicLatchFrozenActorRouter | None:
    if router is None:
        return None
    if not isinstance(router, PublicLatchFrozenActorRouter):
        raise TypeError("policy router must be PublicLatchFrozenActorRouter or None")
    if router.name != PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_NAME:
        raise ValueError(
            "policy router name must be "
            f"{PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_NAME!r}"
        )
    if (
        not isinstance(router.observation_index, int)
        or isinstance(router.observation_index, bool)
        or not 0 <= router.observation_index < actor_observation_dim
    ):
        raise ValueError("policy router observation_index is outside the actor observation")
    if (
        not isinstance(router.trainable_start, int)
        or isinstance(router.trainable_start, bool)
        or not isinstance(router.trainable_stop, int)
        or isinstance(router.trainable_stop, bool)
        or not 0 < router.trainable_start < router.trainable_stop == action_dim
    ):
        raise ValueError(
            "policy router requires one non-empty trainable suffix ending at action_dim"
        )
    if (
        not isinstance(router.close_value, (int, float))
        or isinstance(router.close_value, bool)
        or not isinstance(router.frozen_value, (int, float))
        or isinstance(router.frozen_value, bool)
        or float(router.close_value) != 0.0
        or float(router.frozen_value) != 1.0
    ):
        raise ValueError("policy router requires exact binary close=0 and frozen=1 values")
    expected_authority = (
        ActionAuthorityRule(
            name="arm_after_public_latch",
            start=0,
            stop=router.trainable_start,
            observation_index=router.observation_index,
            active_value=router.frozen_value,
        ),
    )
    if tuple(authority_rules) != expected_authority:
        raise ValueError(
            "public-latch frozen-actor routing requires the matching arm action-authority rule"
        )
    return dataclasses.replace(
        router,
        close_value=float(router.close_value),
        frozen_value=float(router.frozen_value),
    )


def _network_tensor_sha256(network: torch.nn.Module) -> str:
    """Hash canonical parameter/buffer names, metadata and bytes."""

    digest = hashlib.sha256()
    state = {
        key.removeprefix(_COMPILED_STATE_PREFIX): value
        for key, value in network.state_dict().items()
    }
    if len(state) != len(network.state_dict()):
        raise RuntimeError("network state has duplicate canonical keys")
    for key in sorted(state):
        tensor = state[key].detach().contiguous().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file_sha256(path: str | os.PathLike[str], *, label: str) -> str:
    return hashlib.sha256(_read_regular_file_bytes(path, label=label)).hexdigest()


def _read_regular_file_bytes(path: str | os.PathLike[str], *, label: str) -> bytes:
    resolved = os.fspath(path)
    if not os.path.isfile(resolved) or os.path.islink(resolved):
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {resolved}")
    with open(resolved, "rb") as stream:
        return stream.read()


def _truncated_zeta_cdf(mu: float, max_n: int, device: torch.device) -> torch.Tensor:
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    probabilities = ns.pow(-mu)
    return (probabilities / probabilities.sum()).cumsum(dim=0)


def _group_dict(group: ActionNoiseGroup) -> dict[str, Any]:
    return dataclasses.asdict(group)


def _portable_network_state_dict(
    checkpoint_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    path: str,
) -> dict[str, torch.Tensor]:
    """Map root ``torch.compile`` prefixes without weakening strict loading.

    ``torch.compile(module)`` exposes the same parameters under
    ``_orig_mod.<name>``.  FlashSAC saves that wrapper state verbatim, which
    otherwise makes a checkpoint depend on whether the loading process enabled
    compilation.  Only this known, root-level prefix is translated; missing,
    extra, mixed-prefix, or structurally different keys still fail loudly.
    """

    source_keys = set(checkpoint_state)
    target_keys = set(target_state)
    if source_keys == target_keys:
        return dict(checkpoint_state)

    has_prefixed = [key.startswith(_COMPILED_STATE_PREFIX) for key in checkpoint_state]
    if all(has_prefixed):
        translated = {
            key.removeprefix(_COMPILED_STATE_PREFIX): value
            for key, value in checkpoint_state.items()
        }
    elif not any(has_prefixed):
        translated = {
            f"{_COMPILED_STATE_PREFIX}{key}": value
            for key, value in checkpoint_state.items()
        }
    else:
        raise RuntimeError(
            f"network checkpoint {path} mixes compiled and uncompiled parameter keys; "
            "refusing an ambiguous translation"
        )

    if set(translated) != target_keys:
        missing = sorted(target_keys - set(translated))
        unexpected = sorted(set(translated) - target_keys)
        raise RuntimeError(
            f"network checkpoint {path} is structurally incompatible after translating the "
            f"root {_COMPILED_STATE_PREFIX!r} prefix; missing={missing}, unexpected={unexpected}"
        )
    return translated


def _load_network_portably(bundle: Any, path: str, *, load_optimizer: bool) -> None:
    """Load an upstream ``Network`` checkpoint across compile boundaries."""

    device = next(bundle.network.parameters()).device
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"network checkpoint {path} must contain a mapping")
    checkpoint_state = checkpoint.get("network_state_dict")
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError(f"network checkpoint {path} has no mapping network_state_dict")

    target_state = bundle.network.state_dict()
    state = _portable_network_state_dict(checkpoint_state, target_state, path=path)
    bundle.network.load_state_dict(state, strict=True)

    if not load_optimizer:
        return

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if bundle.optimizer is not None and optimizer_state is not None:
        bundle.optimizer.load_state_dict(optimizer_state)
        bundle.update_step = checkpoint["update_step"]
    else:
        print(
            "[Warning] load_optimizer=True but optimizer is None or checkpoint has no optimizer state."
            f" Skipping optimizer load for {path}."
        )

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if bundle.scheduler is not None and scheduler_state is not None:
        bundle.scheduler.load_state_dict(scheduler_state)
    else:
        print(
            "[Warning] load_optimizer=True but scheduler is None or checkpoint has no scheduler state."
            f" Skipping scheduler load for {path}."
        )


def _target_root_key(
    canonical_key: str,
    target_state: Mapping[str, torch.Tensor],
    *,
    path: str,
) -> str:
    """Resolve one canonical actor key after portable root-prefix translation."""

    candidates = (canonical_key, f"{_COMPILED_STATE_PREFIX}{canonical_key}")
    matches = [key for key in candidates if key in target_state]
    if len(matches) != 1:
        raise RuntimeError(
            f"actor checkpoint target for {path} has ambiguous or missing key {canonical_key!r}"
        )
    return matches[0]


def _project_actor_action_outputs(
    checkpoint_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    source_action_indices: Sequence[int],
    expected_source_action_dim: int,
    target_action_dim: int,
    path: str,
) -> dict[str, torch.Tensor]:
    """Strictly reduce only a FlashSAC actor's four action-output tensors.

    The shared observation embedder, encoder, normalization parameters and
    buffers must remain shape compatible. Requiring an explicit source
    dimension prevents an action-layout mistake from being accepted merely
    because the requested indices happen to fit the checkpoint.
    """

    if (
        not isinstance(expected_source_action_dim, int)
        or isinstance(expected_source_action_dim, bool)
        or expected_source_action_dim < 1
    ):
        raise ValueError("expected_source_action_dim must be a positive integer")
    if target_action_dim < 1:
        raise ValueError("target_action_dim must be positive")
    if expected_source_action_dim <= target_action_dim:
        raise ValueError("actor action projection must reduce a larger source action space")
    try:
        indices = tuple(source_action_indices)
    except TypeError as error:
        raise TypeError("source_action_indices must be an integer sequence") from error
    if len(indices) != target_action_dim:
        raise ValueError(
            f"source_action_indices has length {len(indices)}, "
            f"expected target action_dim={target_action_dim}"
        )
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise TypeError("source_action_indices must contain only integers")
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise ValueError("source_action_indices must be unique and strictly increasing")
    if not indices or indices[0] < 0 or indices[-1] >= expected_source_action_dim:
        raise ValueError(
            "source_action_indices are outside the declared source action dimension"
        )

    state = _portable_network_state_dict(checkpoint_state, target_state, path=path)
    output_keys = {
        _target_root_key(key, target_state, path=path) for key in _ACTOR_ACTION_OUTPUT_KEYS
    }
    projected = dict(state)
    index_cache: dict[torch.device, torch.Tensor] = {}
    for key, target_value in target_state.items():
        source_value = state[key]
        if not isinstance(source_value, torch.Tensor):
            raise TypeError(f"actor checkpoint tensor {key!r} in {path} is not a tensor")
        if key not in output_keys:
            if source_value.shape != target_value.shape:
                raise RuntimeError(
                    f"actor checkpoint {path} changes non-output tensor {key!r}: "
                    f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
                )
            continue
        if source_value.ndim not in (1, 2) or target_value.ndim != source_value.ndim:
            raise RuntimeError(
                f"actor output tensor {key!r} in {path} has incompatible ranks: "
                f"source={source_value.ndim}, target={target_value.ndim}"
            )
        if source_value.shape[0] != expected_source_action_dim:
            raise RuntimeError(
                f"actor output tensor {key!r} in {path} declares "
                f"source action_dim={source_value.shape[0]}, "
                f"expected {expected_source_action_dim}"
            )
        if target_value.shape[0] != target_action_dim:
            raise RuntimeError(
                f"target actor output tensor {key!r} has action_dim={target_value.shape[0]}, "
                f"expected {target_action_dim}"
            )
        if source_value.shape[1:] != target_value.shape[1:]:
            raise RuntimeError(
                f"actor output tensor {key!r} in {path} changes hidden dimensions: "
                f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
            )
        index = index_cache.get(source_value.device)
        if index is None:
            index = torch.tensor(indices, dtype=torch.long, device=source_value.device)
            index_cache[source_value.device] = index
        selected = source_value.index_select(0, index)
        if selected.shape != target_value.shape:
            raise RuntimeError(
                f"projected actor output tensor {key!r} has shape {tuple(selected.shape)}, "
                f"expected {tuple(target_value.shape)}"
            )
        projected[key] = selected
    return projected


def _load_actor_portably(
    bundle: Any,
    path: str,
    *,
    source_action_indices: Sequence[int],
    expected_source_action_dim: int,
    target_action_dim: int,
) -> None:
    """Load an actor trunk exactly while projecting a declared action subset."""

    device = next(bundle.network.parameters()).device
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"network checkpoint {path} must contain a mapping")
    checkpoint_state = checkpoint.get("network_state_dict")
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError(f"network checkpoint {path} has no mapping network_state_dict")
    target_state = bundle.network.state_dict()
    state = _project_actor_action_outputs(
        checkpoint_state,
        target_state,
        source_action_indices=source_action_indices,
        expected_source_action_dim=expected_source_action_dim,
        target_action_dim=target_action_dim,
        path=path,
    )
    bundle.network.load_state_dict(state, strict=True)


def _load_grad_scaler_portably(
    scaler: torch.amp.GradScaler,
    state: Any,
    *,
    path: str,
) -> None:
    """Restore AMP state when possible, otherwise keep the target's fresh state.

    A disabled ``GradScaler`` serializes to ``{}``, and PyTorch intentionally
    rejects loading that empty mapping into an enabled scaler.  BC checkpoints
    have exactly this shape because export performs no mixed-precision update.
    Conversely, an enabled scaler's dynamics are irrelevant to a target process
    that disabled AMP.  Both transitions retain the target's freshly initialized
    scaler; equal enabled states are restored exactly.
    """

    if not isinstance(state, Mapping):
        raise TypeError(f"grad_scaler_state_dict in {path} must be a mapping")
    if scaler.is_enabled() and state:
        scaler.load_state_dict(dict(state))
    elif scaler.is_enabled():
        warnings.warn(
            f"checkpoint {path} has disabled/empty AMP scaler state; keeping a fresh enabled scaler",
            stacklevel=2,
        )
    elif state:
        warnings.warn(
            f"checkpoint {path} has enabled AMP scaler state but AMP is disabled; ignoring scaler state",
            stacklevel=2,
        )


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
        actor_action_active_observation_index: int | None = None,
        action_authority_rules: Sequence[ActionAuthorityRule] = (),
        public_latch_frozen_actor_router: PublicLatchFrozenActorRouter | None = None,
        unit_normalize_actor_mean_head: bool = True,
    ) -> None:
        super().__init__(observation_space, action_space, env_info, cfg)
        if actor_action_active_observation_index is not None and action_authority_rules:
            raise ValueError(
                "legacy whole-action authority and action_authority_rules are mutually exclusive"
            )
        if actor_action_active_observation_index is not None and (
            not isinstance(actor_action_active_observation_index, int)
            or isinstance(actor_action_active_observation_index, bool)
            or not 0
            <= actor_action_active_observation_index
            < self._actor_observation_dim
        ):
            raise ValueError(
                "actor_action_active_observation_index must be None or a valid "
                "actor-observation index"
            )
        self._actor_action_active_observation_index = (
            actor_action_active_observation_index
        )
        self._action_authority_rules = _normalize_action_authority_rules(
            action_authority_rules,
            action_dim=self._action_dim,
            actor_observation_dim=self._actor_observation_dim,
        )
        if (
            actor_action_active_observation_index is not None
            and public_latch_frozen_actor_router is not None
        ):
            raise ValueError(
                "legacy whole-action authority cannot be combined with a policy router"
            )
        self._public_latch_frozen_actor_router = _normalize_public_latch_router(
            public_latch_frozen_actor_router,
            action_dim=self._action_dim,
            actor_observation_dim=self._actor_observation_dim,
            authority_rules=self._action_authority_rules,
        )
        self._frozen_lift_actor: FlashSACActor | None = None
        self._frozen_lift_actor_loaded = False
        self._frozen_lift_actor_source_sha256: str | None = None
        if self._public_latch_frozen_actor_router is not None:
            self._frozen_lift_actor = FlashSACActor(
                num_blocks=self._cfg.actor_num_blocks,
                input_dim=self._actor_observation_dim,
                hidden_dim=self._cfg.actor_hidden_dim,
                action_dim=self._action_dim,
            ).to(self._device)
            self._frozen_lift_actor.requires_grad_(False)
            self._frozen_lift_actor.eval()
        if not isinstance(unit_normalize_actor_mean_head, bool):
            raise TypeError("unit_normalize_actor_mean_head must be bool")
        self._unit_normalize_actor_mean_head = unit_normalize_actor_mean_head
        self._actor_learning_rate_scale = 1.0
        mean_modules = [
            module
            for name, module in self._actor.network.named_modules()
            if name.removeprefix(_COMPILED_STATE_PREFIX) == "predictor.mean_w"
        ]
        if len(mean_modules) != 1 or not hasattr(
            mean_modules[0], "parameter_normalization_enabled"
        ):
            raise RuntimeError(
                "FlashSAC actor must expose exactly one configurable predictor.mean_w"
            )
        mean_modules[0].parameter_normalization_enabled = (  # type: ignore[attr-defined]
            unit_normalize_actor_mean_head
        )
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
    def actor_action_active_observation_index(self) -> int | None:
        return self._actor_action_active_observation_index

    @property
    def action_authority_rules(self) -> tuple[ActionAuthorityRule, ...]:
        return self._action_authority_rules

    @property
    def public_latch_frozen_actor_router(
        self,
    ) -> PublicLatchFrozenActorRouter | None:
        return self._public_latch_frozen_actor_router

    @property
    def frozen_lift_actor_sha256(self) -> str | None:
        if not self._frozen_lift_actor_loaded or self._frozen_lift_actor is None:
            return None
        return _network_tensor_sha256(self._frozen_lift_actor)

    @property
    def frozen_lift_actor_source_sha256(self) -> str | None:
        return self._frozen_lift_actor_source_sha256

    @property
    def actor_learning_rate_scale(self) -> float:
        """Return the absolute multiplier applied to the actor LR schedule."""

        return self._actor_learning_rate_scale

    def set_actor_learning_rate_scale(self, scale: float) -> None:
        """Set an absolute actor-only LR multiplier without changing critic LR.

        FlashSAC constructs actor and critic schedulers from one shared learning-
        rate configuration.  Conservative actor fine-tuning needs a smaller
        actor step while retaining the critic schedule, so update both the
        actor optimizer's current LR and the scheduler base LR atomically.
        Repeated calls are absolute rather than multiplicative.
        """

        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("actor learning-rate scale must be a real number")
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("actor learning-rate scale must be finite and positive")
        optimizer = self._actor.optimizer
        scheduler = self._actor.scheduler
        if optimizer is None or scheduler is None:
            raise RuntimeError("FlashSAC actor requires an optimizer and LR scheduler")
        if len(optimizer.param_groups) != len(scheduler.base_lrs):
            raise RuntimeError("actor optimizer and scheduler parameter groups disagree")
        peak_lr = float(self._cfg.learning_rate_peak)
        if not math.isfinite(peak_lr) or peak_lr <= 0.0:
            raise RuntimeError("FlashSAC actor peak learning rate is invalid")
        new_base_lrs: list[float] = []
        new_current_lrs: list[float] = []
        for group, old_base_lr in zip(
            optimizer.param_groups, scheduler.base_lrs, strict=True
        ):
            old_base_lr = float(old_base_lr)
            current_lr = float(group["lr"])
            if (
                not math.isfinite(old_base_lr)
                or old_base_lr <= 0.0
                or not math.isfinite(current_lr)
                or current_lr < 0.0
            ):
                raise RuntimeError("actor optimizer contains an invalid LR schedule state")
            schedule_fraction = current_lr / old_base_lr
            new_base_lr = peak_lr * scale
            new_current_lr = new_base_lr * schedule_fraction
            group["initial_lr"] = new_base_lr
            group["lr"] = new_current_lr
            new_base_lrs.append(new_base_lr)
            new_current_lrs.append(new_current_lr)
        scheduler.base_lrs = new_base_lrs
        scheduler._last_lr = new_current_lrs  # noqa: SLF001 - PyTorch scheduler state
        self._actor_learning_rate_scale = scale

    @property
    def unit_normalize_actor_mean_head(self) -> bool:
        return self._unit_normalize_actor_mean_head

    def _load_frozen_lift_actor_file(
        self,
        actor_path: str | os.PathLike[str],
        *,
        source_sha256: str | None = None,
        bind_source_sha256_to_file: bool = False,
        expected_semantic_sha256: str | None = None,
    ) -> str:
        if self._public_latch_frozen_actor_router is None or self._frozen_lift_actor is None:
            raise RuntimeError("cannot load a frozen lift actor without an active policy router")
        resolved = os.fspath(actor_path)
        file_bytes = _read_regular_file_bytes(resolved, label="frozen lift actor")
        actual_file_sha256 = hashlib.sha256(file_bytes).hexdigest()
        if bind_source_sha256_to_file:
            if source_sha256 is not None:
                raise ValueError(
                    "source_sha256 must be omitted when binding provenance to actor bytes"
                )
            source_sha256 = actual_file_sha256
        if source_sha256 is not None and not _is_sha256(source_sha256):
            raise ValueError("frozen lift actor source SHA-256 must be 64 lowercase hex digits")
        if expected_semantic_sha256 is not None and not _is_sha256(
            expected_semantic_sha256
        ):
            raise ValueError(
                "expected frozen lift actor semantic SHA-256 must be 64 lowercase hex digits"
            )
        checkpoint = torch.load(
            io.BytesIO(file_bytes),
            map_location=self._device,
            weights_only=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"frozen lift actor checkpoint {resolved} must be a mapping")
        checkpoint_state = checkpoint.get("network_state_dict")
        if not isinstance(checkpoint_state, Mapping):
            raise TypeError(
                f"frozen lift actor checkpoint {resolved} has no network_state_dict"
            )
        candidate_actor = FlashSACActor(
            num_blocks=self._cfg.actor_num_blocks,
            input_dim=self._actor_observation_dim,
            hidden_dim=self._cfg.actor_hidden_dim,
            action_dim=self._action_dim,
        ).to(self._device)
        target_state = candidate_actor.state_dict()
        portable_state = _portable_network_state_dict(
            checkpoint_state,
            target_state,
            path=resolved,
        )
        candidate_actor.load_state_dict(portable_state, strict=True)
        candidate_actor.requires_grad_(False)
        candidate_actor.eval()
        candidate_semantic_sha256 = _network_tensor_sha256(candidate_actor)
        if (
            expected_semantic_sha256 is not None
            and candidate_semantic_sha256 != expected_semantic_sha256
        ):
            raise ValueError(
                "frozen lift actor semantic SHA-256 mismatch; "
                f"checkpoint={expected_semantic_sha256!r}, "
                f"actual={candidate_semantic_sha256!r}"
            )
        # Commit only after the independently-loaded candidate passes every
        # provenance and semantic check. A rejected sidecar never contaminates
        # the live policy router.
        self._frozen_lift_actor.load_state_dict(candidate_actor.state_dict(), strict=True)
        self._frozen_lift_actor.requires_grad_(False)
        self._frozen_lift_actor.eval()
        self._frozen_lift_actor_loaded = True
        self._frozen_lift_actor_source_sha256 = source_sha256
        return actual_file_sha256

    def load_frozen_lift_actor(self, checkpoint: str | os.PathLike[str]) -> None:
        """Load the immutable lift actor from an ordinary FlashSAC checkpoint."""

        checkpoint_dir = os.fspath(checkpoint)
        if not os.path.isdir(checkpoint_dir) or os.path.islink(checkpoint_dir):
            raise FileNotFoundError(
                f"frozen lift actor checkpoint must be a real directory: {checkpoint_dir}"
            )
        actor_path = os.path.join(checkpoint_dir, "actor.pt")
        self._load_frozen_lift_actor_file(
            actor_path,
            bind_source_sha256_to_file=True,
        )

    def _require_frozen_lift_actor(self) -> FlashSACActor:
        if not self._frozen_lift_actor_loaded or self._frozen_lift_actor is None:
            raise RuntimeError("policy router requires a loaded frozen lift actor")
        return self._frozen_lift_actor

    @torch.no_grad()
    def frozen_lift_actions(self, actor_observation: torch.Tensor) -> torch.Tensor:
        actor = self._require_frozen_lift_actor()
        if (
            actor_observation.ndim != 2
            or actor_observation.shape[1] != self._actor_observation_dim
            or actor_observation.device != self._device
        ):
            raise ValueError(
                "frozen lift actor requires device-local [batch, actor_observation_dim] input"
            )
        mean, _ = actor.get_mean_and_std(actor_observation, training=False)
        return torch.tanh(mean).clone()

    def _load_frozen_lift_actor_sidecar(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        expected_semantic_sha256: str,
        source_sha256: str | None,
    ) -> None:
        checkpoint_dir = os.fspath(checkpoint)
        actor_path = os.path.join(checkpoint_dir, FROZEN_LIFT_ACTOR_FILENAME)
        self._load_frozen_lift_actor_file(
            actor_path,
            source_sha256=source_sha256,
            expected_semantic_sha256=expected_semantic_sha256,
        )

    @torch.no_grad()
    def initialize_zero_actor_mean(self) -> None:
        """Make a fresh residual actor's deterministic output exactly zero.

        Only the policy mean head is changed. The operation is deliberately
        rejected after Adam has acquired any actor state so it cannot silently
        erase a trained policy; the standard-deviation head, shared trunk, and
        BatchNorm state are left byte-for-byte untouched.
        """

        optimizer = self._actor.optimizer
        if optimizer is None:
            raise RuntimeError("actor has no optimizer")
        if optimizer.state:
            raise RuntimeError(
                "initialize_zero_actor_mean requires a fresh actor optimizer "
                "with no state"
            )
        if self._unit_normalize_actor_mean_head:
            raise RuntimeError(
                "zero actor mean requires unit_normalize_actor_mean_head=False so "
                "the first optimizer step cannot project a tiny residual to unit norm"
            )

        parameters: dict[str, torch.nn.Parameter] = {}
        for name, parameter in self._actor.network.named_parameters():
            canonical_name = name.removeprefix(_COMPILED_STATE_PREFIX)
            if canonical_name in parameters:
                raise RuntimeError(
                    f"actor has duplicate canonical parameter {canonical_name!r}"
                )
            parameters[canonical_name] = parameter
        mean_keys = ("predictor.mean_w.w.weight", "predictor.mean_bias")
        missing = [key for key in mean_keys if key not in parameters]
        if missing:
            raise RuntimeError(f"actor is missing policy mean parameters: {missing}")
        for key in mean_keys:
            parameters[key].zero_()

    @property
    def replay_size(self) -> int:
        return len(self._replay_buffer)

    @torch.no_grad()
    def start_fresh_rollout(self, batch_size: int) -> None:
        """Reset state that is local to the live simulator trajectory.

        Network, optimizer, completed replay rows, reward RMS, and the observed
        maximum return are deliberately preserved.  Pending n-step rows and
        the reward normalizer's per-environment unfinished return cannot be
        carried across a fresh ``env.reset()`` because no simulator state is
        restored with the checkpoint.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        discard_pending = getattr(self._replay_buffer, "discard_pending_n_step", None)
        if discard_pending is not None:
            discard_pending()
        elif hasattr(self._replay_buffer, "_n_step_transitions"):
            self._replay_buffer._n_step_transitions.clear()
        else:
            raise TypeError("replay buffer cannot discard pending n-step transitions")
        if self.reward_normalizer is not None:
            self.reward_normalizer.G_r = torch.zeros(
                batch_size,
                dtype=torch.float32,
                device=self._device,
            )
        self.reset_exploration(batch_size=batch_size)

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

    def _environment_action_active_rows(
        self,
        actor_observation: torch.Tensor,
    ) -> torch.Tensor | None:
        index = self._actor_action_active_observation_index
        if index is None and not self._action_authority_rules:
            return None
        if (
            actor_observation.ndim != 2
            or actor_observation.shape[1] != self._actor_observation_dim
        ):
            raise ValueError(
                "actor action authority requires [batch, actor_observation_dim] observations"
            )
        if index is not None:
            # Legacy residual-task contract: the configured feature is an
            # ALIGN-active bit and whole-action authority begins at zero.
            return actor_observation[:, index] == 0.0

        active = torch.ones(
            (actor_observation.shape[0], self._action_dim),
            dtype=torch.bool,
            device=actor_observation.device,
        )
        for rule in self._action_authority_rules:
            feature = actor_observation[:, rule.observation_index]
            binary = (feature == 0.0) | (feature == 1.0)
            if not bool(binary.all()):
                invalid = feature[~binary]
                raise ValueError(
                    f"action authority feature {rule.observation_index} for rule "
                    f"{rule.name!r} must be exactly binary; got "
                    f"{invalid[:8].detach().cpu().tolist()}"
                )
            grants = feature == rule.active_value
            active[:, rule.start : rule.stop] &= grants.unsqueeze(-1)
        return active

    def _router_frozen_rows(
        self,
        actor_observation: torch.Tensor,
    ) -> torch.Tensor | None:
        router = self._public_latch_frozen_actor_router
        if router is None:
            return None
        if (
            actor_observation.ndim != 2
            or actor_observation.shape[1] != self._actor_observation_dim
        ):
            raise ValueError(
                "policy router requires [batch, actor_observation_dim] observations"
            )
        feature = actor_observation[:, router.observation_index]
        binary = (feature == router.close_value) | (feature == router.frozen_value)
        if not bool(binary.all()):
            invalid = feature[~binary]
            raise ValueError(
                f"policy router feature {router.observation_index} must be exactly "
                f"binary; got {invalid[:8].detach().cpu().tolist()}"
            )
        return feature == router.frozen_value

    def _actor_action_active_rows(
        self,
        actor_observation: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return dimensions owned by the trainable actor during optimization."""

        router = self._public_latch_frozen_actor_router
        if router is None:
            return self._environment_action_active_rows(actor_observation)
        frozen_rows = self._router_frozen_rows(actor_observation)
        assert frozen_rows is not None
        active = torch.zeros(
            (actor_observation.shape[0], self._action_dim),
            dtype=torch.bool,
            device=actor_observation.device,
        )
        active[:, router.trainable_start : router.trainable_stop] = (
            ~frozen_rows
        ).unsqueeze(-1)
        return active

    def _validate_action_authority_inputs(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observations = self._validate_observations(observations)
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if actions.device != self._device:
            raise ValueError(f"actions are on {actions.device}, expected {self._device}")
        if actions.shape != (observations.shape[0], self._action_dim):
            raise ValueError(
                "actions must have shape [num_envs, action_dim]; "
                f"got {tuple(actions.shape)}"
            )
        actor_observation = (
            observations[:, : self._actor_observation_dim]
            if self._cfg.asymmetric_observation
            else observations
        )
        return observations, actor_observation

    @torch.no_grad()
    def apply_trainable_action_authority(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Keep only action dimensions owned by the trainable actor.

        Unlike :meth:`apply_action_authority`, this never invokes a frozen
        policy. It is the canonical projection for actor-only demonstration
        targets and therefore remains usable before the frozen sidecar loads.
        """

        _, actor_observation = self._validate_action_authority_inputs(
            actions, observations
        )
        active = self._actor_action_active_rows(actor_observation)
        if active is None:
            return actions
        if active.ndim == 1:
            active = active.unsqueeze(-1)
        return torch.where(active, actions, torch.zeros_like(actions))

    @torch.no_grad()
    def apply_action_authority(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Project actor or random proposals to the canonical executed action."""

        _, actor_observation = self._validate_action_authority_inputs(
            actions, observations
        )
        active = self._environment_action_active_rows(actor_observation)
        if active is None:
            canonical = actions
        else:
            if active.ndim == 1:
                active = active.unsqueeze(-1)
            canonical = torch.where(active, actions, torch.zeros_like(actions))

        frozen_rows = self._router_frozen_rows(actor_observation)
        if frozen_rows is None:
            return canonical
        frozen_actions = self.frozen_lift_actions(actor_observation)
        return torch.where(
            frozen_rows.unsqueeze(-1),
            frozen_actions.to(dtype=canonical.dtype),
            canonical,
        )

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
        full_observations = observations
        if self._cfg.asymmetric_observation:
            observations = observations[:, : self._actor_observation_dim]
        actions = self._sample_grouped_actions(
            observations,
            temperature=1.0 if training else 0.0,
            noise_scale=noise_scale,
        )
        return self.apply_action_authority(actions, full_observations)

    def process_transition(self, transition: MutableMapping[str, Any]) -> int:
        """Insert a vector transition and report newly materialized replay rows.

        A positive return proves that the just-added vector batch contributed
        to an n-step row. This is stronger than replay size, which stops
        increasing when the circular buffer is full.
        """

        assert_transition_tensors(
            transition,
            device=self._device,
            observation_dim=self._critic_observation_dim,
            action_dim=self._action_dim,
        )
        pending = getattr(self._replay_buffer, "_n_step_transitions", None)
        if pending and _MASKED_REPLAY_VALID_KEY in pending[-1]:
            raise RuntimeError(
                "masked and unmasked transition processing cannot share one "
                "pending n-step rollout; call start_fresh_rollout first"
            )
        canonical_action = self.apply_action_authority(
            transition["action"], transition["observation"]
        )
        if not torch.equal(canonical_action, transition["action"]):
            mismatch = canonical_action != transition["action"]
            raise ValueError(
                "transition action is not canonical under the configured public "
                "action authority/policy router "
                f"({int(mismatch.sum().item())} entries)"
            )
        before = getattr(self._replay_buffer, "total_materialized_rows", None)
        if not isinstance(before, int) or isinstance(before, bool) or before < 0:
            raise TypeError("replay buffer has no monotonic materialized-row counter")
        super().process_transition(transition)
        after = getattr(self._replay_buffer, "total_materialized_rows", None)
        if not isinstance(after, int) or isinstance(after, bool) or after < before:
            raise RuntimeError("replay materialized-row counter is not monotonic")
        return after - before

    @torch.no_grad()
    def process_transition_masked(
        self,
        transition: MutableMapping[str, Any],
        *,
        replay_valid_mask: torch.Tensor,
    ) -> int:
        """Insert only selected per-environment trajectories into replay.

        The complete vector transition remains in the pending n-step window so
        environment row identity and done handling stay identical to the pinned
        Torch buffer.  Once a window matures, the mask from its *oldest*
        transition selects the starting rows that are physically materialized.

        Callers must use this method for every step of a masked rollout.  A row
        that is valid and not done must remain valid on the following call;
        this proves that an n-step sample cannot silently cross from the
        controlled trajectory into an unrecorded policy phase.
        """

        assert_transition_tensors(
            transition,
            device=self._device,
            observation_dim=self._critic_observation_dim,
            action_dim=self._action_dim,
        )
        num_envs = transition["observation"].shape[0]
        if not isinstance(replay_valid_mask, torch.Tensor):
            raise TypeError("replay_valid_mask must be a torch.Tensor")
        if replay_valid_mask.device != self._device:
            raise ValueError(
                f"replay_valid_mask is on {replay_valid_mask.device}, expected "
                f"{self._device}; the bridge does not perform hidden host/device transfers."
            )
        if replay_valid_mask.dtype is not torch.bool:
            raise TypeError("replay_valid_mask must have dtype torch.bool")
        if replay_valid_mask.shape != (num_envs,):
            raise ValueError(
                f"replay_valid_mask must have shape ({num_envs},), got "
                f"{tuple(replay_valid_mask.shape)}"
            )

        # Invalid rows may contain SEARCH actions that are intentionally not
        # canonical for the option actor.  Re-run authority on the *complete*
        # vector batch, then compare only option-owned rows.  Keeping the
        # original batch shape is required for exact frozen-router replay:
        # GEMM kernels can differ by a few ulps when the same actor is evaluated
        # on a boolean-selected sub-batch.
        if bool(replay_valid_mask.any()):
            canonical_action = self.apply_action_authority(
                transition["action"],
                transition["observation"],
            )
            mismatch = (canonical_action != transition["action"]) & (
                replay_valid_mask.unsqueeze(-1)
            )
            if bool(mismatch.any()):
                raise ValueError(
                    "valid transition action is not canonical under the configured "
                    "public action authority/policy router "
                    f"({int(mismatch.sum().item())} entries)"
                )

        replay = self._replay_buffer
        pending = getattr(replay, "_n_step_transitions", None)
        to_buffer_tensor = getattr(replay, "_to_tensor", None)
        if pending is None or not callable(to_buffer_tensor):
            raise TypeError("replay buffer does not expose the audited Torch n-step seam")
        buffer_valid_mask = to_buffer_tensor(replay_valid_mask)
        if pending:
            previous = pending[-1]
            previous_valid = previous.get(_MASKED_REPLAY_VALID_KEY)
            if previous_valid is None:
                raise RuntimeError(
                    "masked and unmasked transition processing cannot share one "
                    "pending n-step rollout; call start_fresh_rollout first"
                )
            previous_done = previous["terminated"].bool() | previous["truncated"].bool()
            missing_continuation = previous_valid & (~previous_done) & (~buffer_valid_mask)
            if bool(missing_continuation.any()):
                raise ValueError(
                    "replay_valid_mask dropped "
                    f"{int(missing_continuation.sum().item())} live environment rows; "
                    "a valid non-done transition must remain valid on the next frame"
                )

        # Invalid collection rows are explicit zero-reward truncation
        # boundaries.  This both resets their return-normalizer accumulator and
        # prevents their values from entering a later valid trajectory.
        masked_transition = dict(transition)
        invalid = ~replay_valid_mask
        if bool(invalid.any()):
            reward = transition["reward"].clone()
            truncated = transition["truncated"].clone()
            reward[invalid] = 0
            truncated[invalid] = True
            masked_transition["reward"] = reward
            masked_transition["truncated"] = truncated
        masked_transition[_MASKED_REPLAY_VALID_KEY] = replay_valid_mask

        before = getattr(replay, "total_materialized_rows", None)
        if not isinstance(before, int) or isinstance(before, bool) or before < 0:
            raise TypeError("replay buffer has no monotonic materialized-row counter")

        expected_materialized = 0
        pending.append(
            {key: to_buffer_tensor(value) for key, value in masked_transition.items()}
        )
        if len(pending) >= replay._n_step:
            n_step_transition = replay._get_n_step_prev_transition()
            oldest_valid = n_step_transition[_MASKED_REPLAY_VALID_KEY]
            add_batch_size = int(oldest_valid.sum().item())
            expected_materialized = add_batch_size
            if add_batch_size:
                end_idx = replay._current_idx + add_batch_size
                if end_idx <= replay._max_length:
                    idxs: Any = slice(replay._current_idx, end_idx)
                else:
                    idxs = (
                        torch.arange(add_batch_size, device=replay._device)
                        + replay._current_idx
                    ) % replay._max_length
                replay._observations[idxs] = n_step_transition["observation"][
                    oldest_valid
                ].to(replay._observations.dtype)
                replay._next_observations[idxs] = n_step_transition[
                    "next_observation"
                ][oldest_valid].to(replay._next_observations.dtype)
                replay._actions[idxs] = n_step_transition["action"][oldest_valid].to(
                    replay._actions.dtype
                )
                replay._rewards[idxs] = n_step_transition["reward"][oldest_valid].to(
                    replay._rewards.dtype
                )
                replay._terminateds[idxs] = n_step_transition["terminated"][
                    oldest_valid
                ].to(replay._terminateds.dtype)
                replay._truncateds[idxs] = n_step_transition["truncated"][
                    oldest_valid
                ].to(replay._truncateds.dtype)
                replay._num_in_buffer = min(
                    replay._num_in_buffer + add_batch_size,
                    replay._max_length,
                )
                replay._current_idx = (
                    replay._current_idx + add_batch_size
                ) % replay._max_length
                replay._total_materialized_rows += add_batch_size

        if self._cfg.normalize_reward:
            if self.reward_normalizer is None:
                raise RuntimeError("normalize_reward=True without a reward normalizer")
            normalizer = self.reward_normalizer
            reward = masked_transition["reward"]
            done = masked_transition["terminated"].bool() | masked_transition[
                "truncated"
            ].bool()
            normalizer.G_r = (
                normalizer.gamma * (~done).float() * normalizer.G_r + reward
            )
            valid_returns = normalizer.G_r[replay_valid_mask]
            if valid_returns.numel():
                normalizer.G_r_max = torch.maximum(
                    normalizer.G_r_max,
                    valid_returns.abs().max(),
                )
                # Deliberately omit invalid zero rows from RMS sample count;
                # otherwise a large SEARCH population would collapse the scale
                # used for the much smaller option-controlled replay cohort.
                normalizer.G_rms.update(valid_returns)

        after = getattr(replay, "total_materialized_rows", None)
        if not isinstance(after, int) or isinstance(after, bool) or after < before:
            raise RuntimeError("replay materialized-row counter is not monotonic")
        if after - before != expected_materialized:
            raise RuntimeError("masked replay materialized-row accounting diverged")
        return after - before

    def update(
        self,
        *,
        actor_enabled: bool = True,
        policy_actions_enabled: bool = True,
    ) -> dict[str, float]:
        """Run one upstream update, optionally withholding actor/temperature.

        A short critic-only burn-in is useful after loading a BC actor around a
        fresh critic.  The replay, reward normalization, critic/target update,
        AMP behavior, schedulers, and global update counter otherwise match the
        pinned upstream implementation exactly.
        """

        if not isinstance(actor_enabled, bool) or not isinstance(
            policy_actions_enabled, bool
        ):
            raise TypeError("actor_enabled and policy_actions_enabled must be bool")
        if (
            not policy_actions_enabled
            and self._actor_action_active_observation_index is None
            and not self._action_authority_rules
        ):
            raise ValueError(
                "policy_actions_enabled=False requires an action-authority observation"
            )

        batch = self._replay_buffer.sample()
        for key, value in batch.items():
            batch[key] = value.to(self._device, non_blocking=True)
        if self._cfg.asymmetric_observation:
            batch["actor_observation"] = batch["observation"][:, : self._actor_observation_dim]
            batch["actor_next_observation"] = batch["next_observation"][
                :, : self._actor_observation_dim
            ]
        else:
            batch["actor_observation"] = batch["observation"]
            batch["actor_next_observation"] = batch["next_observation"]
        if self._cfg.normalize_reward:
            if self.reward_normalizer is None:
                raise RuntimeError("normalize_reward=True without a reward normalizer")
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        do_actor_update = bool(
            actor_enabled
            and policy_actions_enabled
            and self._update_step % self._cfg.actor_update_period == 0
        )
        actor_action_active = self._actor_action_active_rows(
            batch["actor_observation"]
        )
        actor_next_action_active = self._actor_action_active_rows(
            batch["actor_next_observation"]
        )
        if not policy_actions_enabled:
            if actor_action_active is None or actor_next_action_active is None:
                raise RuntimeError(
                    "disabled policy actions require current and next authority masks"
                )
            actor_action_active = torch.zeros_like(actor_action_active)
            actor_next_action_active = torch.zeros_like(actor_next_action_active)
        next_action_override: torch.Tensor | None = None
        next_action_override_rows: torch.Tensor | None = None
        if self._public_latch_frozen_actor_router is not None:
            next_action_override_rows = self._router_frozen_rows(
                batch["actor_next_observation"]
            )
            assert next_action_override_rows is not None
            next_action_override = self.frozen_lift_actions(
                batch["actor_next_observation"]
            )
        if actor_action_active is None:
            if actor_next_action_active is not None:
                raise RuntimeError("current and next actor action authority disagree")
            # Preserve the pinned upstream call bit-for-bit when the optional
            # authority observation is not configured.
            raw_info = _update_networks(
                batch=batch,
                actor=self._actor,
                critic=self._critic,
                target_critic=self._target_critic,
                temperature=self._temperature,
                cfg=self._cfg,
                do_actor_update=do_actor_update,
                device=self._device,
                grad_scaler=self._grad_scaler,
            )
        else:
            if actor_next_action_active is None:
                raise RuntimeError("current and next actor action authority disagree")
            raw_info = _update_networks_with_action_authority(
                batch=batch,
                actor=self._actor,
                critic=self._critic,
                target_critic=self._target_critic,
                temperature=self._temperature,
                cfg=self._cfg,
                do_actor_update=do_actor_update,
                device=self._device,
                grad_scaler=self._grad_scaler,
                actor_action_active=actor_action_active,
                actor_next_action_active=actor_next_action_active,
                next_action_override=next_action_override,
                next_action_override_rows=next_action_override_rows,
            )
        self._update_step += 1
        return {
            key: float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in raw_info.items()
        }

    def demo_bc_rehearsal(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        weight: float = 1.0,
        group_weights: Mapping[str, float] | None = None,
        target_std: float = 0.15,
        std_weight: float = 0.05,
        atanh_epsilon: float = 1.0e-4,
        gradient_clip: float = 10.0,
    ) -> dict[str, float]:
        """Apply one explicit demo-only actor correction after a SAC actor step.

        This intentionally does not use upstream ``actor_bc_alpha`` because
        that term also clones noisy online actions.  Demo actions supervise the
        FlashSAC pre-tanh mean, and the distribution head is softly retained at
        the same explicit standard-deviation prior used by BC bootstrap.
        """

        for name, value in {
            "weight": weight,
            "target_std": target_std,
            "std_weight": std_weight,
            "atanh_epsilon": atanh_epsilon,
            "gradient_clip": gradient_clip,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"demo BC {name} must be finite")
        if weight <= 0.0 or target_std <= 0.0 or std_weight < 0.0:
            raise ValueError("demo BC requires weight>0, target_std>0, and std_weight>=0")
        if not 0.0 < atanh_epsilon < 0.1 or gradient_clip <= 0.0:
            raise ValueError("invalid demo BC atanh epsilon or gradient clip")
        requested_group_weights = {} if group_weights is None else dict(group_weights)
        known_groups = {group.name for group in self._noise_groups}
        unknown_groups = sorted(set(requested_group_weights).difference(known_groups))
        if unknown_groups:
            raise ValueError(f"demo BC weights contain unknown action groups: {unknown_groups}")
        resolved_group_weights: dict[str, float] = {}
        for group in self._noise_groups:
            group_weight = float(requested_group_weights.get(group.name, 1.0))
            if not math.isfinite(group_weight) or group_weight < 0.0:
                raise ValueError(
                    f"demo BC group weight {group.name!r} must be finite and non-negative"
                )
            resolved_group_weights[group.name] = group_weight
        if not any(value > 0.0 for value in resolved_group_weights.values()):
            raise ValueError("at least one demo BC action-group weight must be positive")
        observation = batch.get("observation")
        action = batch.get("action")
        if not isinstance(observation, torch.Tensor) or not isinstance(action, torch.Tensor):
            raise TypeError("demo BC batch requires tensor observation and action")
        if observation.device != self._device or action.device != self._device:
            raise ValueError("demo BC tensors must already be on the agent device")
        if observation.ndim != 2 or observation.shape[1] != self._critic_observation_dim:
            raise ValueError("demo BC observation shape does not match the agent")
        if action.shape != (observation.shape[0], self._action_dim):
            raise ValueError("demo BC action shape does not match the agent")
        actor_observation = (
            observation[:, : self._actor_observation_dim]
            if self._cfg.asymmetric_observation
            else observation
        )
        action_active = self._actor_action_active_rows(actor_observation)
        active_actions: torch.Tensor | None
        if action_active is None:
            active_actions = None
        elif action_active.ndim == 1:
            active_actions = action_active.unsqueeze(-1).expand(
                -1, self._action_dim
            )
        else:
            active_actions = action_active
        if active_actions is not None:
            canonical_action = torch.where(
                active_actions, action, torch.zeros_like(action)
            )
            if not torch.equal(canonical_action, action):
                raise ValueError(
                    "demo BC action is not canonical under the current public "
                    "action-authority rules"
                )
        limit = 1.0 - atanh_epsilon
        target_mean = torch.atanh(action.clamp(-limit, limit))
        action_weights = torch.ones(
            self._action_dim, dtype=torch.float32, device=self._device
        )
        for group in self._noise_groups:
            action_weights[group.start : group.stop] = resolved_group_weights[group.name]
        action_weight_sum = action_weights.sum()
        optimizer = self._actor.optimizer
        if optimizer is None:
            raise RuntimeError("FlashSAC actor has no optimizer")
        if active_actions is not None and not bool(active_actions.any()):
            optimizer.zero_grad(set_to_none=True)
            metrics = {
                "demo_bc/loss": 0.0,
                "demo_bc/action_loss": 0.0,
                "demo_bc/std_loss": 0.0,
                "demo_bc/grad_norm": 0.0,
                "demo_bc/grad_overflow": 0.0,
                "demo_bc/updated": 0.0,
                "demo_bc/active_action_fraction": 0.0,
            }
            for group in self._noise_groups:
                metrics[f"demo_bc/{group.name}_action_rmse"] = 0.0
                metrics[f"demo_bc/{group.name}_active_elements"] = 0.0
                metrics[f"demo_bc/{group.name}_active_fraction"] = 0.0
            return metrics
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=self._cfg.use_amp,
        ):
            predicted_mean, predicted_std = self._actor.apply(
                "get_mean_and_std",
                observations=actor_observation,
                # Rehearse the exact function used by collection/evaluation.
                # Gradients remain enabled; only BatchNorm switches from
                # transient batch statistics to its fixed deployment buffers.
                training=False,
            )
            # ``reduce-overhead`` may return CUDA-Graph-backed output storage
            # which is reused by the next compiled actor invocation (including
            # the parameter-normalization path below). Materialize the
            # pre-update prediction now; retaining ``predicted_mean`` until
            # after ``optimizer.step()`` can otherwise raise an overwritten-
            # CUDAGraph-output error while computing diagnostics.
            predicted_action_for_metrics = torch.tanh(
                predicted_mean.detach().float()
            ).clone()
            action_loss_elementwise = F.smooth_l1_loss(
                predicted_mean,
                target_mean,
                reduction="none",
                beta=1.0,
            )
            predicted_log_std = predicted_std.float().clamp_min(1.0e-8).log()
            std_loss_elementwise = (
                predicted_log_std - math.log(target_std)
            ).square()
            if active_actions is None:
                # Preserve the pre-authority reduction order exactly on the
                # unrestricted path.
                action_loss = (
                    action_loss_elementwise.float()
                    * action_weights.unsqueeze(0)
                ).sum() / (observation.shape[0] * action_weight_sum)
                std_loss = std_loss_elementwise.mean()
            else:
                weighted_active = (
                    active_actions.to(dtype=torch.float32)
                    * action_weights.unsqueeze(0)
                )
                active_weight_sum = weighted_active.sum()
                action_loss = (
                    torch.where(
                        active_actions,
                        action_loss_elementwise,
                        torch.zeros_like(action_loss_elementwise),
                    ).float()
                    * action_weights.unsqueeze(0)
                ).sum() / active_weight_sum.clamp_min(1.0)
                std_loss = torch.where(
                    active_actions,
                    std_loss_elementwise,
                    torch.zeros_like(std_loss_elementwise),
                ).sum() / active_actions.sum().to(
                    dtype=std_loss_elementwise.dtype
                )
            loss = weight * (action_loss + std_weight * std_loss)

        if self._cfg.use_amp:
            scale_before = float(self._grad_scaler.get_scale())
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._actor.network.parameters(), gradient_clip
            )
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
            scale_after = float(self._grad_scaler.get_scale())
            grad_overflow = (not bool(torch.isfinite(grad_norm).item())) or (
                scale_after < scale_before
            )
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._actor.network.parameters(), gradient_clip
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise FloatingPointError(
                    f"demo BC gradient norm is not finite without AMP: {float(grad_norm)}"
                )
            optimizer.step()
            grad_overflow = False
        # The SAC scheduler advances once for the corresponding environment
        # update.  Rehearsal is a correction within that update, not a second
        # unit of the global learning-rate schedule.
        self._actor.normalize_parameters()
        with torch.no_grad():
            if active_actions is None:
                squared_action_error = (
                    predicted_action_for_metrics - action.float()
                ).square()
            else:
                squared_action_error = torch.where(
                    active_actions,
                    (predicted_action_for_metrics - action.float()).square(),
                    torch.zeros_like(predicted_action_for_metrics),
                )
        # AMP overflow is a recoverable GradScaler event: ``step`` is skipped
        # and the scale is reduced. Keep strict-JSON telemetry finite while
        # surfacing the event explicitly instead of hiding it or aborting the
        # entire curriculum run before the scaler can adapt.
        reported_grad_norm = (
            float(grad_norm.detach())
            if bool(torch.isfinite(grad_norm).item())
            else float(gradient_clip)
        )
        metrics = {
            "demo_bc/loss": float(loss.detach()),
            "demo_bc/action_loss": float(action_loss.detach()),
            "demo_bc/std_loss": float(std_loss.detach()),
            "demo_bc/grad_norm": reported_grad_norm,
            "demo_bc/grad_overflow": float(grad_overflow),
            "demo_bc/updated": float(not grad_overflow),
        }
        if active_actions is not None:
            metrics["demo_bc/active_action_fraction"] = float(
                active_actions.float().mean()
            )
        for group in self._noise_groups:
            group_error = squared_action_error[:, group.start : group.stop]
            if active_actions is None:
                group_rmse = group_error.mean().sqrt()
            else:
                group_active = active_actions[:, group.start : group.stop]
                group_active_count = group_active.sum()
                group_rmse = (
                    group_error.sum()
                    / group_active_count.clamp_min(1).to(dtype=group_error.dtype)
                ).sqrt()
                metrics[f"demo_bc/{group.name}_active_elements"] = float(
                    group_active_count
                )
                metrics[f"demo_bc/{group.name}_active_fraction"] = float(
                    group_active.float().mean()
                )
            metrics[f"demo_bc/{group.name}_action_rmse"] = float(group_rmse)
        return metrics

    def _bridge_checkpoint_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "version": BRIDGE_CHECKPOINT_VERSION,
            "upstream_commit": FLASH_SAC_COMMIT,
            "fork_commit": FLASH_SAC_FORK_COMMIT,
            "action_dim": self._action_dim,
            "actor_learning_rate_scale": self._actor_learning_rate_scale,
            "noise_groups": [_group_dict(group) for group in self._noise_groups],
            "group_noise_scale": self._group_noise_scale,
            "cached_noise": self._cached_noise,
            "noise_repeat_count": self._cur_noise_repeat_count,
            "noise_repeat_n": self._cur_noise_repeat_n,
            "cpu_rng_state": torch.get_rng_state(),
        }
        if self._actor_action_active_observation_index is not None:
            state["actor_action_active_observation_index"] = (
                self._actor_action_active_observation_index
            )
        if self._action_authority_rules:
            state["action_authority_rules"] = [
                dataclasses.asdict(rule) for rule in self._action_authority_rules
            ]
        if self._public_latch_frozen_actor_router is not None:
            self._require_frozen_lift_actor()
            frozen_sha256 = self.frozen_lift_actor_sha256
            source_sha256 = self._frozen_lift_actor_source_sha256
            if not _is_sha256(frozen_sha256) or not _is_sha256(source_sha256):
                raise RuntimeError(
                    "policy router checkpoint requires canonical frozen actor and "
                    "source actor SHA-256 fingerprints"
                )
            state["public_latch_frozen_actor_router"] = dataclasses.asdict(
                self._public_latch_frozen_actor_router
            )
            state["frozen_lift_actor_semantic_sha256"] = frozen_sha256
            state["frozen_lift_actor_source_sha256"] = source_sha256
        if not self._unit_normalize_actor_mean_head:
            state["unit_normalize_actor_mean_head"] = False
        if self._device.type == "cuda":
            state["device_rng_state"] = torch.cuda.get_rng_state(self._device)
        return state

    def save(self, path: str) -> None:
        super().save(path)
        if self._public_latch_frozen_actor_router is not None:
            frozen_actor = self._require_frozen_lift_actor()
            sidecar_path = os.path.join(path, FROZEN_LIFT_ACTOR_FILENAME)
            if os.path.islink(sidecar_path):
                raise ValueError(
                    f"refusing to overwrite frozen lift actor symlink: {sidecar_path}"
                )
            torch.save(
                {"network_state_dict": frozen_actor.state_dict()},
                sidecar_path,
            )
        torch.save(self._bridge_checkpoint_state(), os.path.join(path, BRIDGE_STATE_FILENAME))

    def _validate_and_load_router_sidecar(
        self,
        path: str | os.PathLike[str],
        state: Mapping[str, Any],
    ) -> None:
        checkpoint_router = state.get("public_latch_frozen_actor_router")
        current_router = (
            None
            if self._public_latch_frozen_actor_router is None
            else dataclasses.asdict(self._public_latch_frozen_actor_router)
        )
        if checkpoint_router != current_router:
            raise ValueError(
                "checkpoint policy router differs from the current bridge "
                f"configuration; checkpoint={checkpoint_router}, current={current_router}"
            )

        sidecar_path = os.path.join(os.fspath(path), FROZEN_LIFT_ACTOR_FILENAME)
        if current_router is None:
            unexpected_metadata = {
                key: state[key]
                for key in (
                    "frozen_lift_actor_semantic_sha256",
                    "frozen_lift_actor_source_sha256",
                )
                if key in state
            }
            if unexpected_metadata or os.path.lexists(sidecar_path):
                raise ValueError(
                    "checkpoint without a policy router contains frozen lift actor "
                    f"state: metadata={unexpected_metadata}, sidecar={os.path.lexists(sidecar_path)}"
                )
            return

        if state.get("version") != BRIDGE_CHECKPOINT_VERSION:
            raise ValueError(
                "public-latch frozen-actor routing requires bridge checkpoint "
                f"version {BRIDGE_CHECKPOINT_VERSION}"
            )
        semantic_sha256 = state.get("frozen_lift_actor_semantic_sha256")
        source_sha256 = state.get("frozen_lift_actor_source_sha256")
        if not _is_sha256(semantic_sha256) or not _is_sha256(source_sha256):
            raise ValueError(
                "routed checkpoint has invalid frozen actor semantic/source SHA-256 metadata"
            )
        self._load_frozen_lift_actor_sidecar(
            path,
            expected_semantic_sha256=semantic_sha256,
            source_sha256=source_sha256,
        )

    def load_frozen_lift_actor_sidecar(
        self,
        checkpoint: str | os.PathLike[str],
    ) -> None:
        """Load and authenticate the self-contained frozen actor for evaluation."""

        checkpoint_dir = os.fspath(checkpoint)
        if not os.path.isdir(checkpoint_dir) or os.path.islink(checkpoint_dir):
            raise FileNotFoundError(
                f"routed checkpoint must be a real directory: {checkpoint_dir}"
            )
        bridge_path = os.path.join(checkpoint_dir, BRIDGE_STATE_FILENAME)
        _regular_file_sha256(bridge_path, label="Torch bridge checkpoint state")
        state = torch.load(bridge_path, map_location=self._device, weights_only=True)
        if not isinstance(state, Mapping):
            raise TypeError(f"Torch bridge checkpoint state must be a mapping: {bridge_path}")
        if state.get("version") not in _SUPPORTED_BRIDGE_CHECKPOINT_VERSIONS:
            raise ValueError(
                f"unsupported Torch bridge checkpoint version {state.get('version')!r}"
            )
        if state.get("upstream_commit") != FLASH_SAC_COMMIT:
            raise ValueError(
                f"checkpoint targets upstream commit {state.get('upstream_commit')!r}, "
                f"expected {FLASH_SAC_COMMIT}"
            )
        if state.get("fork_commit") != FLASH_SAC_FORK_COMMIT:
            raise ValueError(
                "public-latch frozen-actor routing requires the current audited "
                f"FlashSAC fork {FLASH_SAC_FORK_COMMIT}"
            )
        if state.get("action_dim") != self._action_dim:
            raise ValueError(
                "routed checkpoint action dimension differs from the current bridge"
            )
        if state.get("actor_action_active_observation_index") != (
            self._actor_action_active_observation_index
        ):
            raise ValueError(
                "routed checkpoint legacy action authority differs from the current bridge"
            )
        checkpoint_authority = state.get("action_authority_rules", [])
        current_authority = [
            dataclasses.asdict(rule) for rule in self._action_authority_rules
        ]
        if checkpoint_authority != current_authority:
            raise ValueError(
                "routed checkpoint action authority rules differ from the current bridge"
            )
        if state.get("unit_normalize_actor_mean_head", True) != (
            self._unit_normalize_actor_mean_head
        ):
            raise ValueError(
                "routed checkpoint actor mean-head normalization differs from the current bridge"
            )
        checkpoint_groups = state.get("noise_groups")
        current_groups = [_group_dict(group) for group in self._noise_groups]
        if checkpoint_groups != current_groups:
            raise ValueError(
                "routed checkpoint noise groups differ from the current bridge"
            )
        self._validate_and_load_router_sidecar(checkpoint_dir, state)

    def load_actor(
        self,
        path: str,
        *,
        source_action_indices: Sequence[int] | None = None,
        expected_source_action_dim: int | None = None,
    ) -> None:
        """Strictly load only actor weights from a portable checkpoint directory.

        This transfer path intentionally leaves the actor optimizer/scheduler,
        critic, target critic, temperature, replay, reward normalizer,
        exploration state, agent update counter, AMP scaler, and RNG untouched.

        Supplying both projection arguments permits a declared larger actor to
        initialize a smaller action space. Only the four actor output tensors
        may be sliced; every shared trunk tensor must remain shape compatible.
        It is the safe initialization boundary between different task modes.
        """

        checkpoint_dir = os.fspath(path)
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"actor checkpoint directory does not exist: {checkpoint_dir}")
        actor_path = os.path.join(checkpoint_dir, "actor.pt")
        if not os.path.isfile(actor_path):
            raise FileNotFoundError(f"missing FlashSAC actor checkpoint: {actor_path}")
        if (source_action_indices is None) != (expected_source_action_dim is None):
            raise ValueError(
                "source_action_indices and expected_source_action_dim must be provided together"
            )
        if source_action_indices is None:
            _load_network_portably(self._actor, actor_path, load_optimizer=False)
            detail = ""
        else:
            assert expected_source_action_dim is not None
            _load_actor_portably(
                self._actor,
                actor_path,
                source_action_indices=source_action_indices,
                expected_source_action_dim=expected_source_action_dim,
                target_action_dim=self._action_dim,
            )
            detail = (
                f" with action projection {expected_source_action_dim}"
                f"->{self._action_dim}"
            )
        print(
            f"\033[32m[FlashSAC]\033[0m Successfully loaded actor-only checkpoint "
            f"from {checkpoint_dir}{detail}."
        )

    def load(self, path: str) -> None:
        if self._public_latch_frozen_actor_router is not None:
            # Fail closed on all router/sidecar metadata before any trainable
            # network or optimizer is overwritten by a full resume.
            self.load_frozen_lift_actor_sidecar(path)
        load_optimizer = self._cfg.load_optimizer
        _load_network_portably(
            self._actor,
            os.path.join(path, "actor.pt"),
            load_optimizer=load_optimizer,
        )
        _load_network_portably(
            self._critic,
            os.path.join(path, "critic.pt"),
            load_optimizer=load_optimizer,
        )
        _load_network_portably(
            self._target_critic,
            os.path.join(path, "target_critic.pt"),
            load_optimizer=False,
        )
        _load_network_portably(
            self._temperature,
            os.path.join(path, "temperature.pt"),
            load_optimizer=load_optimizer,
        )

        if load_optimizer:
            agent_state_path = os.path.join(path, "agent_state.pt")
            if not os.path.exists(agent_state_path):
                raise FileNotFoundError(f"missing FlashSAC agent state: {agent_state_path}")
            agent_state = torch.load(agent_state_path, map_location=self._device, weights_only=True)
            self._update_step = agent_state["update_step"]
            _load_grad_scaler_portably(
                self._grad_scaler,
                agent_state["grad_scaler_state_dict"],
                path=agent_state_path,
            )

        if self._cfg.load_reward_normalizer:
            if self.reward_normalizer is None:
                raise RuntimeError(
                    "load_reward_normalizer=True but this agent was constructed without reward normalization"
                )
            self.reward_normalizer.load(os.path.join(path, "reward_normalizer.pt"))

        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded checkpoint from {path}.")

        bridge_path = os.path.join(path, BRIDGE_STATE_FILENAME)
        if not os.path.exists(bridge_path):
            orphan_sidecar = os.path.join(path, FROZEN_LIFT_ACTOR_FILENAME)
            if os.path.lexists(orphan_sidecar):
                raise ValueError(
                    "checkpoint without Torch bridge state contains an orphan frozen "
                    f"lift actor sidecar: {orphan_sidecar}"
                )
            if self._public_latch_frozen_actor_router is not None:
                raise FileNotFoundError(
                    "policy router checkpoint is missing its Torch bridge state: "
                    f"{bridge_path}"
                )
            warnings.warn(
                f"{bridge_path} is absent; loaded an upstream-only checkpoint and reset exploration state.",
                stacklevel=2,
            )
            self.reset_exploration()
            return

        _regular_file_sha256(bridge_path, label="Torch bridge checkpoint state")
        state = torch.load(bridge_path, map_location=self._device, weights_only=True)
        if not isinstance(state, Mapping):
            raise TypeError(f"Torch bridge checkpoint state must be a mapping: {bridge_path}")
        if state.get("version") not in _SUPPORTED_BRIDGE_CHECKPOINT_VERSIONS:
            raise ValueError(
                f"unsupported Torch bridge checkpoint version {state.get('version')!r}; "
                f"expected one of {_SUPPORTED_BRIDGE_CHECKPOINT_VERSIONS}"
            )
        if state.get("upstream_commit") != FLASH_SAC_COMMIT:
            raise ValueError(
                f"checkpoint targets upstream commit {state.get('upstream_commit')!r}, "
                f"expected {FLASH_SAC_COMMIT}"
            )
        checkpoint_fork_commit = state.get("fork_commit")
        if checkpoint_fork_commit not in (None, *FLASH_SAC_COMPATIBLE_FORK_COMMITS):
            raise ValueError(
                f"checkpoint targets FlashSAC fork {checkpoint_fork_commit!r}, "
                f"expected one of {FLASH_SAC_COMPATIBLE_FORK_COMMITS}"
            )
        if (
            self._action_authority_rules
            and checkpoint_fork_commit not in FLASH_SAC_COMPATIBLE_FORK_COMMITS
        ):
            raise ValueError(
                "per-action authority requires a checkpoint created by the audited "
                f"FlashSAC forks {FLASH_SAC_COMPATIBLE_FORK_COMMITS}"
            )
        if (
            self._public_latch_frozen_actor_router is not None
            and checkpoint_fork_commit != FLASH_SAC_FORK_COMMIT
        ):
            raise ValueError(
                "public-latch frozen-actor routing requires the current audited "
                f"FlashSAC fork {FLASH_SAC_FORK_COMMIT}"
            )
        if state.get("action_dim") != self._action_dim:
            raise ValueError(
                f"checkpoint action_dim={state.get('action_dim')}, current action_dim={self._action_dim}"
            )
        checkpoint_active_index = state.get(
            "actor_action_active_observation_index"
        )
        if checkpoint_active_index != self._actor_action_active_observation_index:
            raise ValueError(
                "checkpoint actor action authority differs from the current bridge "
                "configuration; "
                f"checkpoint={checkpoint_active_index}, "
                f"current={self._actor_action_active_observation_index}"
            )
        checkpoint_authority_rules = state.get("action_authority_rules", [])
        current_authority_rules = [
            dataclasses.asdict(rule) for rule in self._action_authority_rules
        ]
        if checkpoint_authority_rules != current_authority_rules:
            raise ValueError(
                "checkpoint action authority rules differ from the current bridge "
                "configuration; "
                f"checkpoint={checkpoint_authority_rules}, "
                f"current={current_authority_rules}"
            )
        self._validate_and_load_router_sidecar(path, state)
        checkpoint_unit_normalize_mean = state.get(
            "unit_normalize_actor_mean_head",
            True,
        )
        if checkpoint_unit_normalize_mean != self._unit_normalize_actor_mean_head:
            raise ValueError(
                "checkpoint actor mean-head normalization differs from the current "
                "bridge configuration; "
                f"checkpoint={checkpoint_unit_normalize_mean}, "
                f"current={self._unit_normalize_actor_mean_head}"
            )
        checkpoint_groups = state.get("noise_groups")
        current_groups = [_group_dict(group) for group in self._noise_groups]
        if checkpoint_groups != current_groups:
            raise ValueError(
                "checkpoint noise groups differ from the current bridge configuration; "
                f"checkpoint={checkpoint_groups}, current={current_groups}"
            )

        checkpoint_actor_lr_scale = state.get("actor_learning_rate_scale", 1.0)
        if (
            isinstance(checkpoint_actor_lr_scale, bool)
            or not isinstance(checkpoint_actor_lr_scale, (int, float))
            or not math.isfinite(float(checkpoint_actor_lr_scale))
            or float(checkpoint_actor_lr_scale) <= 0.0
        ):
            raise ValueError("checkpoint actor learning-rate scale is invalid")
        checkpoint_actor_lr_scale = float(checkpoint_actor_lr_scale)
        if load_optimizer:
            scheduler = self._actor.scheduler
            if scheduler is None or any(
                not math.isclose(
                    float(base_lr),
                    float(self._cfg.learning_rate_peak) * checkpoint_actor_lr_scale,
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                )
                for base_lr in scheduler.base_lrs
            ):
                raise ValueError(
                    "checkpoint actor optimizer LR schedule disagrees with its "
                    "actor_learning_rate_scale metadata"
                )
            self._actor_learning_rate_scale = checkpoint_actor_lr_scale
        else:
            self.set_actor_learning_rate_scale(checkpoint_actor_lr_scale)

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
    "ActionAuthorityRule",
    "ActionNoiseGroup",
    "BRIDGE_CHECKPOINT_VERSION",
    "BRIDGE_STATE_FILENAME",
    "FLASH_SAC_COMMIT",
    "FLASH_SAC_COMPATIBLE_FORK_COMMITS",
    "FLASH_SAC_FORK_COMMIT",
    "FROZEN_LIFT_ACTOR_FILENAME",
    "FlashSACTorchBridge",
    "PUBLIC_LATCH_FROZEN_ACTOR_ROUTER_NAME",
    "PublicLatchFrozenActorRouter",
    "assert_transition_tensors",
    "build_agent_config",
]
