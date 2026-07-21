"""Permanent demonstration replay and fixed-ratio FlashSAC batch mixing.

The pinned FlashSAC agent owns one :class:`TorchUniformBuffer` and samples it
directly inside ``update()``.  This module wraps that buffer instead of changing
the upstream submodule:

* online transitions still use the upstream ring buffer;
* demonstrations live in a separate, seal-once tensor reservoir and therefore
  can never be overwritten by online collection;
* every sampled update contains an exact, fixed number of online and demo rows;
* optional phase/stratum labels balance approach, close, and lift inside that
  demo quota, regardless of their raw dataset frequencies;
* sampling, concatenation, and shuffling stay in Torch on the configured device;
* replay contents, in-flight n-step queues, and sampler RNG are checkpointed.

Demo rows must already be standard FlashSAC transitions.  For ``n_step > 1``,
``reward``, terminal flags, and ``next_observation`` must have been precomputed
with exactly the same ``n_step`` and ``gamma`` as the online buffer.  This
module deliberately does not reconstruct n-step returns from a flat dataset:
doing so without trajectory/environment IDs can silently cross reset boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
import math
import os
from pathlib import Path
from typing import Any

import torch


TRANSITION_KEYS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)
DEMO_MASK_KEY = "is_demo"
DEMO_REPLAY_VERSION = 3

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _canonical_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda:0")
    return resolved


def _require_tensor_batch(
    batch: Mapping[str, Any],
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
    validate_values: bool = True,
) -> int:
    missing = [key for key in TRANSITION_KEYS if key not in batch]
    if missing:
        raise KeyError(f"transition batch is missing required keys: {missing}")
    extras = sorted(set(batch).difference(TRANSITION_KEYS))
    if extras:
        raise KeyError(f"transition batch has unsupported keys: {extras}")

    for key in TRANSITION_KEYS:
        if not isinstance(batch[key], torch.Tensor):
            raise TypeError(f"transition[{key!r}] must be a torch.Tensor")

    observation = batch["observation"]
    next_observation = batch["next_observation"]
    action = batch["action"]
    if observation.ndim != 2 or next_observation.shape != observation.shape:
        raise ValueError(
            "observation and next_observation must share shape [batch, observation_dim]; "
            f"got {tuple(observation.shape)} and {tuple(next_observation.shape)}"
        )
    batch_size = observation.shape[0]
    if batch_size < 1:
        raise ValueError("transition batch must not be empty")
    if action.ndim != 2 or action.shape[0] != batch_size:
        raise ValueError(
            "action must have shape [batch, action_dim] with the same batch size as observation; "
            f"got {tuple(action.shape)}"
        )
    if observation_dim is not None and observation.shape[1] != observation_dim:
        raise ValueError(f"expected observation_dim={observation_dim}, got {observation.shape[1]}")
    if action_dim is not None and action.shape[1] != action_dim:
        raise ValueError(f"expected action_dim={action_dim}, got {action.shape[1]}")
    for key in ("reward", "terminated", "truncated"):
        if batch[key].shape != (batch_size,):
            raise ValueError(f"transition[{key!r}] must have shape ({batch_size},)")

    for key in ("observation", "action", "reward", "next_observation"):
        value = batch[key]
        if not value.is_floating_point():
            raise TypeError(f"transition[{key!r}] must have a floating dtype")
        if validate_values and not torch.isfinite(value).all():
            raise ValueError(f"transition[{key!r}] contains NaN or infinity")
    if validate_values:
        terminated = batch["terminated"].bool()
        truncated = batch["truncated"].bool()
        if torch.any(terminated & truncated):
            raise ValueError("terminated and truncated must be mutually exclusive")
    return batch_size


def _resolve_stratum_labels(
    *,
    stratum: torch.Tensor | None,
    phase: torch.Tensor | None,
    batch_size: int,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if stratum is not None and phase is not None:
        raise ValueError("stratum and phase are aliases; provide only one")
    labels = stratum if stratum is not None else phase
    if labels is None:
        return None
    if not isinstance(labels, torch.Tensor):
        raise TypeError("stratum/phase labels must be a torch.Tensor")
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError(f"stratum/phase labels must have shape ({batch_size},)")
    if labels.dtype not in _INTEGER_DTYPES:
        raise TypeError("stratum/phase labels must use uint8 or a signed integer dtype")
    target_device = labels.device if device is None else device
    return labels.detach().to(device=target_device, dtype=torch.int64, copy=True)


def precompute_n_step_by_episode(
    one_step: Mapping[str, torch.Tensor],
    episode_offsets: torch.Tensor | Sequence[int],
    *,
    n_step: int,
    gamma: float,
    stratum: torch.Tensor | None = None,
    phase: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """Materialize upstream-compatible n-step rows without crossing episodes.

    ``episode_offsets`` uses the conventional CSR layout ``[0, ..., N]``.
    Every episode must end in exactly one ``terminated`` or ``truncated`` row,
    with no earlier done row.  One output is produced for every input start row;
    short episode tails stop at their true final transition.  The upstream
    critic still applies its configured ``gamma**n_step`` when bootstrapping a
    truncated tail, so callers must retain the same global ``n_step`` value.

    Labels are copied from the start row and returned separately from the six
    standard transition tensors.
    """

    if n_step < 1:
        raise ValueError("n_step must be positive")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    rows = _require_tensor_batch(one_step)
    device = one_step["observation"].device
    for key in TRANSITION_KEYS:
        if one_step[key].device != device:
            raise ValueError(f"one-step transition {key!r} is on {one_step[key].device}, expected {device}")

    if isinstance(episode_offsets, torch.Tensor):
        if episode_offsets.dtype not in _INTEGER_DTYPES:
            raise TypeError("episode_offsets must use an integer dtype")
        offsets = episode_offsets.detach().to(device=device, dtype=torch.int64)
    else:
        offsets = torch.as_tensor(tuple(episode_offsets), dtype=torch.int64, device=device)
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("episode_offsets must be one-dimensional with at least [0, N]")
    offsets_host = offsets.detach().cpu().tolist()
    if offsets_host[0] != 0 or offsets_host[-1] != rows:
        raise ValueError(f"episode_offsets must start at 0 and end at {rows}")
    if any(stop <= start for start, stop in zip(offsets_host[:-1], offsets_host[1:], strict=True)):
        raise ValueError("episode_offsets must be strictly increasing; empty episodes are unsupported")

    done = one_step["terminated"].bool() | one_step["truncated"].bool()
    for start, stop in zip(offsets_host[:-1], offsets_host[1:], strict=True):
        if stop - start > 1 and bool(done[start : stop - 1].any()):
            raise ValueError(f"episode [{start}, {stop}) contains a done row before its boundary")
        if not bool(done[stop - 1]):
            raise ValueError(f"episode [{start}, {stop}) does not end in terminated or truncated")

    lengths = offsets[1:] - offsets[:-1]
    episode_ends = torch.repeat_interleave(offsets[1:] - 1, lengths)
    starts = torch.arange(rows, dtype=torch.int64, device=device)
    last_indices = torch.minimum(starts + n_step - 1, episode_ends)
    rewards = torch.zeros(rows, dtype=torch.float32, device=device)
    source_rewards = one_step["reward"].to(dtype=torch.float32)
    # Match TorchUniformBuffer's reversed recurrence, including floating-point
    # operation order (r_t + gamma * R * (1 - done_t)).
    for horizon in reversed(range(n_step)):
        candidates = starts + horizon
        within_episode = candidates <= episode_ends
        safe_indices = torch.minimum(candidates, episode_ends)
        candidate_rewards = source_rewards[safe_indices] + (
            float(gamma)
            * rewards
            * (~done[safe_indices]).to(dtype=torch.float32)
        )
        rewards = torch.where(within_episode, candidate_rewards, rewards)

    precomputed = {
        "observation": one_step["observation"].clone(),
        "action": one_step["action"].clone(),
        "reward": rewards,
        "terminated": one_step["terminated"][last_indices].clone(),
        "truncated": one_step["truncated"][last_indices].clone(),
        "next_observation": one_step["next_observation"][last_indices].clone(),
    }
    labels = _resolve_stratum_labels(
        stratum=stratum,
        phase=phase,
        batch_size=rows,
        device=device,
    )
    return precomputed, labels


def load_and_precompute_n_step(
    path: str | os.PathLike[str],
    *,
    device: torch.device | str,
    n_step: int,
    gamma: float,
    episode_offsets_key: str = "episode_offsets",
    stratum_key: str | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """Load a standard one-step tensor dataset and materialize n-step rows.

    The file must contain the six :data:`TRANSITION_KEYS` plus
    ``episode_offsets``.  When ``stratum_key`` is omitted, ``stratum`` and then
    ``phase`` are detected automatically.  Labels remain outside the returned
    transition mapping.
    """

    resolved_device = _canonical_device(device)
    dataset = torch.load(path, map_location=resolved_device, weights_only=True)
    if not isinstance(dataset, Mapping):
        raise TypeError("one-step demo dataset must contain a mapping")
    if episode_offsets_key not in dataset:
        raise KeyError(f"dataset is missing {episode_offsets_key!r}")
    transitions = {key: dataset[key] for key in TRANSITION_KEYS if key in dataset}
    missing = [key for key in TRANSITION_KEYS if key not in transitions]
    if missing:
        raise KeyError(f"one-step demo dataset is missing transition keys: {missing}")
    selected_stratum_key = stratum_key
    if selected_stratum_key is None:
        selected_stratum_key = next((key for key in ("stratum", "phase") if key in dataset), None)
    labels = None if selected_stratum_key is None else dataset[selected_stratum_key]
    return precompute_n_step_by_episode(
        transitions,
        dataset[episode_offsets_key],
        n_step=n_step,
        gamma=gamma,
        stratum=labels,
    )


def _apportion_weighted_counts(batch_size: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    """Return deterministic largest-remainder integer quotas summing to B."""

    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("stratum weights must have a positive finite sum")
    quotas = tuple(batch_size * weight / total for weight in weights)
    counts = [math.floor(quota) for quota in quotas]
    remainder = batch_size - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    if sum(counts) != batch_size:
        raise RuntimeError("weighted stratum apportionment did not preserve batch size")
    return tuple(counts)


class PermanentDemoReservoir:
    """A separately allocated, immutable-after-seal demonstration dataset."""

    def __init__(
        self,
        *,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        n_step: int,
        gamma: float,
        device: torch.device | str,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        if n_step < 1:
            raise ValueError("n_step must be positive")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        self._capacity = int(capacity)
        self._observation_dim = int(observation_dim)
        self._action_dim = int(action_dim)
        self._n_step = int(n_step)
        self._gamma = float(gamma)
        self._device = _canonical_device(device)
        self._storage = {
            "observation": torch.empty(
                (capacity, observation_dim), dtype=torch.float32, device=self._device
            ),
            "action": torch.empty((capacity, action_dim), dtype=torch.float32, device=self._device),
            "reward": torch.empty(capacity, dtype=torch.float32, device=self._device),
            "terminated": torch.empty(capacity, dtype=torch.float32, device=self._device),
            "truncated": torch.empty(capacity, dtype=torch.float32, device=self._device),
            "next_observation": torch.empty(
                (capacity, observation_dim), dtype=torch.float32, device=self._device
            ),
        }
        self._size = 0
        self._sealed = False
        self._stratum: torch.Tensor | None = None
        self._stratum_values: tuple[int, ...] = ()
        self._stratum_indices: tuple[torch.Tensor, ...] = ()

    def __len__(self) -> int:
        return self._size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def n_step(self) -> int:
        return self._n_step

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def has_strata(self) -> bool:
        return self._stratum is not None

    @property
    def stratum_values(self) -> tuple[int, ...]:
        return self._stratum_values

    def add_precomputed(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        n_step: int,
        gamma: float,
        stratum: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
    ) -> None:
        """Append already materialized n-step transitions before sealing.

        Tensor inputs may be loaded on another device; ingestion performs the
        one explicit copy into permanent storage.  The update/sample path never
        moves rows through CPU or NumPy.
        """

        if self._sealed:
            raise RuntimeError("demonstration reservoir is sealed and immutable")
        if n_step != self._n_step:
            raise ValueError(f"demo n_step={n_step} does not match configured n_step={self._n_step}")
        if not math.isclose(float(gamma), self._gamma, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"demo gamma={gamma} does not match configured gamma={self._gamma}")
        batch_size = _require_tensor_batch(
            batch,
            observation_dim=self._observation_dim,
            action_dim=self._action_dim,
        )
        stop = self._size + batch_size
        if stop > self._capacity:
            raise OverflowError(
                f"demo capacity exceeded: size={self._size}, add={batch_size}, capacity={self._capacity}"
            )
        labels = _resolve_stratum_labels(
            stratum=stratum,
            phase=phase,
            batch_size=batch_size,
            device=self._device,
        )
        if self._size > 0 and (self._stratum is None) != (labels is None):
            raise ValueError("all appended demonstration batches must consistently include or omit strata")
        if self._size == 0 and labels is not None:
            self._stratum = torch.empty(self._capacity, dtype=torch.int64, device=self._device)
        destination = slice(self._size, stop)
        for key in TRANSITION_KEYS:
            self._storage[key][destination].copy_(
                batch[key].detach().to(device=self._device, dtype=self._storage[key].dtype)
            )
        if labels is not None:
            assert self._stratum is not None
            self._stratum[destination].copy_(labels)
        self._size = stop

    def seal(self) -> None:
        if self._size < 1:
            raise RuntimeError("cannot seal an empty demonstration reservoir")
        self._sealed = True

        self._rebuild_stratum_index()

    def _rebuild_stratum_index(self) -> None:
        if self._stratum is None:
            self._stratum_values = ()
            self._stratum_indices = ()
            return
        unique = torch.unique(self._stratum[: self._size], sorted=True)
        self._stratum_values = tuple(int(value) for value in unique.detach().cpu().tolist())
        self._stratum_indices = tuple(
            torch.nonzero(self._stratum[: self._size] == value, as_tuple=False).flatten()
            for value in self._stratum_values
        )

    def normalize_stratum_weights(
        self,
        weights: Mapping[int, float] | None,
    ) -> tuple[tuple[int, float], ...] | None:
        """Validate and canonicalize explicit weights for checkpointing."""

        if weights is None:
            return None
        if not self.has_strata:
            raise ValueError("stratum weights require labeled demonstrations")
        unknown = sorted(set(weights).difference(self._stratum_values))
        if unknown:
            raise ValueError(f"stratum weights contain labels absent from demos: {unknown}")
        canonical: list[tuple[int, float]] = []
        positive = False
        for label in self._stratum_values:
            weight = float(weights.get(label, 0.0))
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"stratum {label} weight must be finite and non-negative")
            canonical.append((label, weight))
            positive |= weight > 0.0
        if not positive:
            raise ValueError("at least one stratum weight must be positive")
        return tuple(canonical)

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        stratum_weights: Mapping[int, float] | tuple[tuple[int, float], ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self._sealed:
            raise RuntimeError("seal the demonstration reservoir before sampling")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.has_strata:
            if stratum_weights is not None:
                raise ValueError("stratum weights require labeled demonstrations")
            indices = torch.randint(
                0,
                self._size,
                (batch_size,),
                device=self._device,
                generator=generator,
            )
        else:
            canonical = self.normalize_stratum_weights(
                None if stratum_weights is None else dict(stratum_weights)
            )
            indices = self._sample_stratified_indices(
                batch_size,
                generator=generator,
                weights=canonical,
            )
        return {key: value[indices] for key, value in self._storage.items()}

    def _sample_stratified_indices(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        weights: tuple[tuple[int, float], ...] | None,
    ) -> torch.Tensor:
        num_strata = len(self._stratum_values)
        if weights is None:
            # A random stratum permutation per batch rotates which phases get
            # the at-most-one remainder row when B is not divisible by K.
            order = torch.randperm(num_strata, device=self._device, generator=generator)
            repeats = (batch_size + num_strata - 1) // num_strata
            slots = order.repeat(repeats)[:batch_size]
        else:
            counts = _apportion_weighted_counts(batch_size, tuple(weight for _, weight in weights))
            slots = torch.repeat_interleave(
                torch.arange(num_strata, device=self._device),
                torch.tensor(counts, dtype=torch.int64, device=self._device),
            )
            slots = slots[torch.randperm(batch_size, device=self._device, generator=generator)]

        sampled = torch.empty(batch_size, dtype=torch.int64, device=self._device)
        for slot, source_indices in enumerate(self._stratum_indices):
            positions = torch.nonzero(slots == slot, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            choices = torch.randint(
                0,
                source_indices.numel(),
                (positions.numel(),),
                device=self._device,
                generator=generator,
            )
            sampled[positions] = source_indices[choices]
        return sampled

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": DEMO_REPLAY_VERSION,
            "capacity": self._capacity,
            "observation_dim": self._observation_dim,
            "action_dim": self._action_dim,
            "n_step": self._n_step,
            "gamma": self._gamma,
            "size": self._size,
            "sealed": self._sealed,
            "storage": {key: value[: self._size].clone() for key, value in self._storage.items()},
            "stratum": None if self._stratum is None else self._stratum[: self._size].clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "capacity": self._capacity,
            "observation_dim": self._observation_dim,
            "action_dim": self._action_dim,
            "n_step": self._n_step,
        }
        if state.get("version") not in (1, 2, DEMO_REPLAY_VERSION):
            raise ValueError(f"unsupported demo replay version {state.get('version')!r}")
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"demo checkpoint {key}={state.get(key)!r}, expected {value!r}")
        if not math.isclose(float(state.get("gamma", float("nan"))), self._gamma, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("demo checkpoint gamma does not match")
        size = int(state["size"])
        if not 0 <= size <= self._capacity:
            raise ValueError(f"invalid demo checkpoint size {size}")
        storage = state["storage"]
        _require_stored_shapes(
            storage,
            size=size,
            observation_dim=self._observation_dim,
            action_dim=self._action_dim,
        )
        for key in TRANSITION_KEYS:
            self._storage[key][:size].copy_(
                storage[key].to(device=self._device, dtype=self._storage[key].dtype)
            )
        stratum = state.get("stratum")
        if stratum is None:
            self._stratum = None
        else:
            labels = _resolve_stratum_labels(
                stratum=stratum,
                phase=None,
                batch_size=size,
                device=self._device,
            )
            assert labels is not None
            self._stratum = torch.empty(self._capacity, dtype=torch.int64, device=self._device)
            self._stratum[:size].copy_(labels)
        self._size = size
        self._sealed = bool(state["sealed"])
        self._rebuild_stratum_index()


def _require_stored_shapes(
    storage: Mapping[str, Any],
    *,
    size: int,
    observation_dim: int,
    action_dim: int,
) -> None:
    expected_shapes = {
        "observation": (size, observation_dim),
        "action": (size, action_dim),
        "reward": (size,),
        "terminated": (size,),
        "truncated": (size,),
        "next_observation": (size, observation_dim),
    }
    for key, shape in expected_shapes.items():
        value = storage.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            actual = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise ValueError(f"stored {key!r} has shape {actual}, expected {shape}")


_UPSTREAM_BUFFER_ATTRIBUTES = (
    "_max_length",
    "_min_length",
    "_n_step",
    "_gamma",
    "_sample_batch_size",
    "_device",
    "_observations",
    "_next_observations",
    "_actions",
    "_rewards",
    "_terminateds",
    "_truncateds",
    "_n_step_transitions",
    "_num_in_buffer",
    "_current_idx",
)


def _assert_upstream_buffer_contract(buffer: Any, *, device: torch.device) -> None:
    missing = [name for name in _UPSTREAM_BUFFER_ATTRIBUTES if not hasattr(buffer, name)]
    if missing:
        raise TypeError(f"online buffer is not the pinned TorchUniformBuffer; missing {missing}")
    if _canonical_device(buffer._device) != device:
        raise ValueError(f"online replay is on {buffer._device}, mixer is on {device}")


def _online_buffer_state_dict(buffer: Any) -> dict[str, Any]:
    n = int(buffer._num_in_buffer)
    return {
        "max_length": int(buffer._max_length),
        "min_length": int(buffer._min_length),
        "n_step": int(buffer._n_step),
        "gamma": float(buffer._gamma),
        "sample_batch_size": int(buffer._sample_batch_size),
        "num_in_buffer": n,
        "current_idx": int(buffer._current_idx),
        "storage": {
            "observation": buffer._observations[:n].clone(),
            "action": buffer._actions[:n].clone(),
            "reward": buffer._rewards[:n].clone(),
            "terminated": buffer._terminateds[:n].clone(),
            "truncated": buffer._truncateds[:n].clone(),
            "next_observation": buffer._next_observations[:n].clone(),
        },
        # Upstream save/load intentionally drops this queue.  Keeping it is
        # required for an exact continuation when n_step > 1.
        "n_step_transitions": [
            {key: value.clone() for key, value in transition.items()}
            for transition in buffer._n_step_transitions
        ],
    }


def _load_online_buffer_state_dict(buffer: Any, state: Mapping[str, Any]) -> None:
    expected = {
        "max_length": int(buffer._max_length),
        "min_length": int(buffer._min_length),
        "n_step": int(buffer._n_step),
        "sample_batch_size": int(buffer._sample_batch_size),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"online replay checkpoint {key}={state.get(key)!r}, expected {value!r}")
    if not math.isclose(float(state.get("gamma", float("nan"))), float(buffer._gamma), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("online replay checkpoint gamma does not match")
    n = int(state["num_in_buffer"])
    current_idx = int(state["current_idx"])
    if not 0 <= n <= buffer._max_length:
        raise ValueError(f"invalid online replay size {n}")
    if not 0 <= current_idx < buffer._max_length:
        raise ValueError(f"invalid online replay current_idx {current_idx}")
    storage = state["storage"]
    _require_stored_shapes(
        storage,
        size=n,
        observation_dim=buffer._observations.shape[1],
        action_dim=buffer._actions.shape[1],
    )
    destinations = {
        "observation": buffer._observations,
        "action": buffer._actions,
        "reward": buffer._rewards,
        "terminated": buffer._terminateds,
        "truncated": buffer._truncateds,
        "next_observation": buffer._next_observations,
    }
    for key, destination in destinations.items():
        destination[:n].copy_(storage[key].to(device=destination.device, dtype=destination.dtype))
    buffer._num_in_buffer = n
    buffer._current_idx = current_idx
    pending = state["n_step_transitions"]
    if len(pending) > buffer._n_step:
        raise ValueError("online replay checkpoint has too many in-flight n-step batches")
    buffer._n_step_transitions.clear()
    for transition in pending:
        _require_tensor_batch(transition)
        buffer._n_step_transitions.append(
            {key: value.to(device=buffer._device, copy=True) for key, value in transition.items()}
        )


class FixedFractionDemoReplay:
    """Drop-in replay wrapper sampled by the unmodified FlashSAC ``update``.

    ``add`` and ``can_sample`` delegate to the online ring buffer.  ``sample``
    draws explicit indices from both buffers using a private device generator,
    concatenates them, and applies a random permutation.  The returned
    ``is_demo`` boolean tensor is ignored by upstream FlashSAC but is available
    for a later demo-only behavior-cloning loss.
    """

    def __init__(
        self,
        online_buffer: Any,
        demos: PermanentDemoReservoir,
        *,
        batch_size: int,
        demo_fraction: float = 0.25,
        device: torch.device | str,
        seed: int,
        stratum_weights: Mapping[int, float] | None = None,
        demo_fingerprints: Sequence[str] = (),
    ) -> None:
        self._device = _canonical_device(device)
        _assert_upstream_buffer_contract(online_buffer, device=self._device)
        if demos.device != self._device:
            raise ValueError(f"demo replay is on {demos.device}, mixer is on {self._device}")
        if demos.n_step != online_buffer._n_step:
            raise ValueError("demo and online n_step differ")
        if not math.isclose(demos.gamma, float(online_buffer._gamma), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("demo and online gamma differ")
        if not demos.sealed:
            raise RuntimeError("seal demonstrations before creating the replay mixer")
        if batch_size < 2:
            raise ValueError("mixed batch_size must be at least 2")
        if not math.isfinite(demo_fraction) or not 0.0 < demo_fraction < 1.0:
            raise ValueError("demo_fraction must be finite and strictly between 0 and 1")
        demo_rows_float = batch_size * demo_fraction
        demo_rows = int(round(demo_rows_float))
        if not math.isclose(demo_rows_float, demo_rows, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "batch_size * demo_fraction must be an integer for an exact fixed-ratio batch; "
                f"got {batch_size} * {demo_fraction} = {demo_rows_float}"
            )
        if not 0 < demo_rows < batch_size:
            raise ValueError("fixed-ratio batch must contain at least one online and one demo row")
        self._online = online_buffer
        self._demos = demos
        self._batch_size = int(batch_size)
        self._demo_fraction = float(demo_fraction)
        self._demo_rows = demo_rows
        self._online_rows = batch_size - demo_rows
        self._stratum_weights = demos.normalize_stratum_weights(stratum_weights)
        self._demo_fingerprints = tuple(str(value) for value in demo_fingerprints)
        if any(not value for value in self._demo_fingerprints):
            raise ValueError("demonstration fingerprints must be non-empty strings")
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(int(seed))

    def __len__(self) -> int:
        """Return online replay size, preserving bridge/trainer semantics."""

        return len(self._online)

    @property
    def demo_size(self) -> int:
        return len(self._demos)

    @property
    def demo_rows_per_batch(self) -> int:
        return self._demo_rows

    @property
    def online_rows_per_batch(self) -> int:
        return self._online_rows

    @property
    def demo_fraction(self) -> float:
        return self._demo_fraction

    def add(self, transition: MutableMapping[str, Any]) -> None:
        self._online.add(transition)

    def reset(self) -> None:
        """Reset online experience only; permanent demonstrations survive."""

        self._online.reset()

    def discard_pending_n_step(self) -> None:
        """Drop trajectory-local rows after starting a fresh simulator state.

        Stored replay remains intact.  Only the upstream deque whose rows have
        not yet formed an n-step transition is cleared; carrying it across a
        process restart would splice two physically unrelated episodes.
        """

        self._online._n_step_transitions.clear()

    def can_sample(self) -> bool:
        return bool(self._online.can_sample()) and self._demos.sealed and len(self._demos) > 0

    def sample(self, sample_idxs: Any = None) -> dict[str, torch.Tensor]:
        if sample_idxs is not None:
            raise ValueError("explicit sample_idxs would violate the fixed demo fraction")
        if not self.can_sample():
            raise RuntimeError("mixed replay cannot sample yet")
        online_indices = torch.randint(
            0,
            len(self._online),
            (self._online_rows,),
            device=self._device,
            generator=self._generator,
        )
        online_batch = self._online.sample(sample_idxs=online_indices)
        demo_batch = self._demos.sample(
            self._demo_rows,
            generator=self._generator,
            stratum_weights=self._stratum_weights,
        )
        _require_mixing_batches(online_batch, demo_batch, device=self._device)
        mixed = {
            key: torch.cat((online_batch[key], demo_batch[key]), dim=0)
            for key in TRANSITION_KEYS
        }
        is_demo = torch.cat(
            (
                torch.zeros(self._online_rows, dtype=torch.bool, device=self._device),
                torch.ones(self._demo_rows, dtype=torch.bool, device=self._device),
            )
        )
        permutation = torch.randperm(self._batch_size, device=self._device, generator=self._generator)
        for key in TRANSITION_KEYS:
            mixed[key] = mixed[key][permutation]
        mixed[DEMO_MASK_KEY] = is_demo[permutation]
        return mixed

    def sample_demonstrations(self, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        """Sample a phase-balanced demo-only rehearsal batch on the replay device."""

        rows = self._demo_rows if batch_size is None else int(batch_size)
        if rows < 1:
            raise ValueError("demonstration rehearsal batch_size must be positive")
        return self._demos.sample(
            rows,
            generator=self._generator,
            stratum_weights=self._stratum_weights,
        )

    def get_observations(self) -> torch.Tensor:
        return self._online.get_observations()

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": DEMO_REPLAY_VERSION,
            "batch_size": self._batch_size,
            "demo_fraction": self._demo_fraction,
            "device_type": str(self._device),
            "stratum_weights": self._stratum_weights,
            "demo_fingerprints": self._demo_fingerprints,
            "generator_state": self._generator.get_state().clone(),
            "online": _online_buffer_state_dict(self._online),
            "demos": self._demos.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        checkpoint_version = state.get("version")
        if checkpoint_version not in (1, 2, DEMO_REPLAY_VERSION):
            raise ValueError(f"unsupported mixed replay version {state.get('version')!r}")
        if state.get("batch_size") != self._batch_size:
            raise ValueError("mixed replay checkpoint batch_size does not match")
        if not math.isclose(
            float(state.get("demo_fraction", float("nan"))),
            self._demo_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("mixed replay checkpoint demo_fraction does not match")
        if state.get("device_type") != str(self._device):
            raise ValueError(
                f"mixed replay checkpoint device={state.get('device_type')!r}, expected {str(self._device)!r}"
            )
        checkpoint_weights = state.get("stratum_weights") if checkpoint_version >= 2 else None
        if checkpoint_weights != self._stratum_weights:
            raise ValueError(
                "mixed replay checkpoint stratum weights differ; "
                f"checkpoint={checkpoint_weights!r}, expected={self._stratum_weights!r}"
            )
        checkpoint_fingerprints = (
            tuple(state.get("demo_fingerprints", ())) if checkpoint_version >= 3 else ()
        )
        if checkpoint_fingerprints != self._demo_fingerprints:
            raise ValueError(
                "mixed replay demonstration fingerprints differ; "
                f"checkpoint={checkpoint_fingerprints!r}, expected={self._demo_fingerprints!r}"
            )
        _load_online_buffer_state_dict(self._online, state["online"])
        self._demos.load_state_dict(state["demos"])
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("mixed replay generator_state must be a tensor")
        self._generator.set_state(generator_state.cpu())

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        torch.save(self.state_dict(), temporary)
        os.replace(temporary, destination)

    def load(self, path: str | os.PathLike[str]) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.load_state_dict(state)


def _require_mixing_batches(
    online: Mapping[str, Any],
    demos: Mapping[str, Any],
    *,
    device: torch.device,
) -> None:
    # Values were validated on insertion into each source buffer.  Avoid
    # reductions here: converting their scalar results to Python would add a
    # CUDA synchronization to every gradient update.
    _require_tensor_batch(online, validate_values=False)
    _require_tensor_batch(demos, validate_values=False)
    for label, batch in (("online", online), ("demo", demos)):
        for key in TRANSITION_KEYS:
            if batch[key].device != device:
                raise ValueError(f"{label} batch {key!r} is on {batch[key].device}, expected {device}")
    for key in TRANSITION_KEYS:
        if online[key].shape[1:] != demos[key].shape[1:]:
            raise ValueError(
                f"online/demo {key!r} shapes are incompatible: "
                f"{tuple(online[key].shape)} vs {tuple(demos[key].shape)}"
            )
        if online[key].dtype != demos[key].dtype:
            raise ValueError(
                f"online/demo {key!r} dtypes differ: {online[key].dtype} vs {demos[key].dtype}"
            )


def attach_demo_replay(
    agent: Any,
    demos: PermanentDemoReservoir,
    *,
    batch_size: int,
    demo_fraction: float = 0.25,
    seed: int,
    stratum_weights: Mapping[int, float] | None = None,
    demo_fingerprints: Sequence[str] = (),
) -> FixedFractionDemoReplay:
    """Attach a mixed replay to a pinned ``FlashSACTorchBridge`` instance.

    This is the sole intentional white-box seam: the pinned upstream samples
    ``agent._replay_buffer`` directly.  The returned wrapper is also installed
    there, so upstream ``process_transition()``, ``update()``, and
    ``save_replay_buffer()/load_replay_buffer()`` continue to work unchanged.
    """

    if not hasattr(agent, "_replay_buffer") or not hasattr(agent, "device"):
        raise TypeError("agent must expose the pinned FlashSAC replay buffer and device")
    if isinstance(agent._replay_buffer, FixedFractionDemoReplay):
        raise RuntimeError("agent already has a demonstration replay attached")
    wrapper = FixedFractionDemoReplay(
        agent._replay_buffer,
        demos,
        batch_size=batch_size,
        demo_fraction=demo_fraction,
        device=agent.device,
        seed=seed,
        stratum_weights=stratum_weights,
        demo_fingerprints=demo_fingerprints,
    )
    agent._replay_buffer = wrapper
    return wrapper
