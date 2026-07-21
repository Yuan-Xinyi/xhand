"""GPU-native observation/action demonstrations for actor-only rehearsal.

The PickTool lift teacher datasets intentionally contain no reward, terminal
flag, or next observation.  They are valid behavior-cloning supervision, but
they are not valid SAC transitions.  This module keeps that distinction
structural: :class:`ActorRehearsalReservoir` stores only observations and
actions and is never installed as an agent replay buffer.

Rows are copied once onto the configured Torch device, then sampled using a
private device-local generator.  Optional phase labels support deterministic
largest-remainder stratification.  Dataset contents, sampler state, source
SHA256 order, and sampling configuration are checkpointed so the first batch
after an exact rehearsal-state resume is bitwise identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch


ACTOR_REHEARSAL_VERSION = 1
ACTOR_REHEARSAL_KEYS = ("observation", "action")

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


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the raw-file SHA256 used as immutable source lineage."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_source_fingerprints(values: Sequence[str]) -> tuple[str, ...]:
    fingerprints = tuple(str(value).lower() for value in values)
    if not fingerprints:
        raise ValueError("actor rehearsal requires at least one source fingerprint")
    for value in fingerprints:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid actor rehearsal SHA256 fingerprint {value!r}")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("actor rehearsal source fingerprints must not contain duplicates")
    return fingerprints


def _require_actor_batch(
    batch: Mapping[str, Any],
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
    validate_values: bool = True,
) -> int:
    missing = [key for key in ACTOR_REHEARSAL_KEYS if key not in batch]
    if missing:
        raise KeyError(f"actor rehearsal batch is missing required keys: {missing}")
    extras = sorted(set(batch).difference(ACTOR_REHEARSAL_KEYS))
    if extras:
        raise KeyError(f"actor rehearsal batch has unsupported keys: {extras}")
    for key in ACTOR_REHEARSAL_KEYS:
        if not isinstance(batch[key], torch.Tensor):
            raise TypeError(f"actor rehearsal {key!r} must be a torch.Tensor")

    observation = batch["observation"]
    action = batch["action"]
    if observation.ndim != 2:
        raise ValueError(
            "actor rehearsal observation must have shape [batch, observation_dim]; "
            f"got {tuple(observation.shape)}"
        )
    rows = int(observation.shape[0])
    if rows < 1:
        raise ValueError("actor rehearsal batch must not be empty")
    if action.ndim != 2 or action.shape[0] != rows:
        raise ValueError(
            "actor rehearsal action must have shape [batch, action_dim] with the same "
            f"batch size as observation; got {tuple(action.shape)}"
        )
    if observation_dim is not None and observation.shape[1] != observation_dim:
        raise ValueError(
            f"expected actor rehearsal observation_dim={observation_dim}, "
            f"got {observation.shape[1]}"
        )
    if action_dim is not None and action.shape[1] != action_dim:
        raise ValueError(
            f"expected actor rehearsal action_dim={action_dim}, got {action.shape[1]}"
        )
    for key in ACTOR_REHEARSAL_KEYS:
        value = batch[key]
        if not value.is_floating_point():
            raise TypeError(f"actor rehearsal {key!r} must have a floating dtype")
        if validate_values and not bool(torch.isfinite(value).all()):
            raise ValueError(f"actor rehearsal {key!r} contains NaN or infinity")
    if validate_values and float(action.abs().max()) > 1.0001:
        raise ValueError("actor rehearsal action exceeds the normalized [-1, 1] range")
    return rows


def _require_phase(
    phase: Any,
    *,
    rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    if phase is None:
        return None
    if not isinstance(phase, torch.Tensor):
        raise TypeError("actor rehearsal phase must be a torch.Tensor")
    if phase.ndim != 1 or phase.shape != (rows,):
        raise ValueError(f"actor rehearsal phase must have shape ({rows},)")
    if phase.dtype not in _INTEGER_DTYPES:
        raise TypeError("actor rehearsal phase must use an integer dtype")
    return phase.detach().to(device=device, dtype=torch.int64, copy=True)


def _audit_episode_partition(payload: Mapping[str, Any], *, rows: int) -> int:
    offsets = payload.get("episode_offsets")
    successes = payload.get("episode_success")
    if not isinstance(offsets, torch.Tensor):
        raise TypeError("actor rehearsal dataset requires tensor episode_offsets")
    if offsets.dtype not in _INTEGER_DTYPES:
        raise TypeError("episode_offsets must use an integer dtype")
    offsets = offsets.detach().to(device="cpu", dtype=torch.int64)
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("episode_offsets must be one-dimensional and contain [0, ..., N]")
    if int(offsets[0]) != 0 or int(offsets[-1]) != rows:
        raise ValueError(f"episode_offsets must start at 0 and end at {rows}")
    if bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError("episode_offsets must be strictly increasing")

    episodes = int(offsets.numel() - 1)
    if not isinstance(successes, torch.Tensor):
        raise TypeError("actor rehearsal dataset requires tensor episode_success")
    if successes.dtype != torch.bool or successes.shape != (episodes,):
        raise ValueError(f"episode_success must be bool with shape ({episodes},)")
    if not bool(successes.all()):
        raise ValueError("actor rehearsal may contain only successful episodes")

    episode_id = payload.get("episode_id")
    if episode_id is not None:
        if not isinstance(episode_id, torch.Tensor) or episode_id.dtype not in _INTEGER_DTYPES:
            raise TypeError("episode_id must be an integer tensor when present")
        if episode_id.shape != (rows,):
            raise ValueError(f"episode_id must have shape ({rows},)")
        ids = episode_id.detach().to(device="cpu", dtype=torch.int64)
        for episode, (start, stop) in enumerate(
            zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True)
        ):
            segment = ids[start:stop]
            if segment.numel() == 0 or bool((segment != segment[0]).any()):
                raise ValueError(f"episode_id changes inside episode_offsets segment {episode}")
        if torch.unique(ids, sorted=False).numel() != episodes:
            raise ValueError("episode_id unique count does not match episode_offsets")
    return episodes


def _audit_metadata(
    payload: Mapping[str, Any],
    *,
    observation_dim: int,
    action_dim: int,
    phase: torch.Tensor | None,
    expected_metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    metadata = payload.get("meta")
    if not isinstance(metadata, Mapping):
        raise TypeError("actor rehearsal dataset requires mapping metadata in 'meta'")
    if metadata.get("format_version") != 1:
        raise ValueError(
            f"actor rehearsal format_version={metadata.get('format_version')!r}, expected 1"
        )
    for key in ("action_layout", "observation_layout", "collector", "dataset_phase"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"actor rehearsal metadata requires non-empty {key!r}")
    for key, expected in (
        ("observation_dim", observation_dim),
        ("action_dim", action_dim),
    ):
        if key in metadata and metadata[key] != expected:
            raise ValueError(
                f"actor rehearsal metadata {key}={metadata[key]!r}, expected {expected}"
            )
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"actor rehearsal metadata {key}={metadata.get(key)!r}, expected {expected!r}"
                )

    phase_names = metadata.get("phase_names")
    if phase_names is not None:
        if not isinstance(phase_names, (list, tuple)) or not phase_names or not all(
            isinstance(name, str) and name for name in phase_names
        ):
            raise ValueError("metadata phase_names must be a non-empty sequence of names")
        if phase is not None and (
            int(phase.min()) < 0 or int(phase.max()) >= len(phase_names)
        ):
            raise ValueError("actor rehearsal phase is outside metadata phase_names")
    for key in ("teacher_probability", "executed_teacher_fraction"):
        if key in metadata:
            value = metadata[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"actor rehearsal metadata {key} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"actor rehearsal metadata {key} must be in [0, 1]")
    return metadata


def load_actor_rehearsal(
    path: str | os.PathLike[str],
    *,
    device: torch.device | str,
    observation_dim: int,
    action_dim: int,
    expected_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, dict[str, Any]]:
    """Load and audit a successful observation/action teacher dataset.

    ``obs`` is the native key used by the PickTool option-teacher collectors;
    ``observation`` is accepted for projected transition demonstrations.  The
    two aliases are deliberately mutually exclusive.  No missing transition
    fields are fabricated.
    """

    if observation_dim < 1 or action_dim < 1:
        raise ValueError("observation_dim and action_dim must be positive")
    source_path = Path(path)
    resolved_device = _canonical_device(device)
    fingerprint = sha256_file(source_path)
    payload = torch.load(source_path, map_location=resolved_device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("actor rehearsal dataset root must be a mapping")
    observation_keys = [key for key in ("obs", "observation") if key in payload]
    if len(observation_keys) != 1:
        raise KeyError("actor rehearsal dataset must contain exactly one of 'obs' or 'observation'")
    if "action" not in payload:
        raise KeyError("actor rehearsal dataset is missing 'action'")
    raw_batch = {
        "observation": payload[observation_keys[0]],
        "action": payload["action"],
    }
    rows = _require_actor_batch(
        raw_batch,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    batch = {
        key: value.detach().to(device=resolved_device, dtype=torch.float32, copy=True)
        for key, value in raw_batch.items()
    }
    phase = _require_phase(payload.get("phase"), rows=rows, device=resolved_device)
    episodes = _audit_episode_partition(payload, rows=rows)
    metadata = _audit_metadata(
        payload,
        observation_dim=observation_dim,
        action_dim=action_dim,
        phase=phase,
        expected_metadata=expected_metadata,
    )
    phase_counts: dict[str, int] = {}
    if phase is not None:
        values, counts = torch.unique(phase, sorted=True, return_counts=True)
        phase_counts = {
            str(int(value)): int(count)
            for value, count in zip(
                values.detach().cpu().tolist(),
                counts.detach().cpu().tolist(),
                strict=True,
            )
        }
    audit = {
        "path": str(source_path.resolve()),
        "sha256": fingerprint,
        "transitions": rows,
        "episodes": episodes,
        "phase_counts": phase_counts,
        "collector": str(metadata["collector"]),
        "dataset_phase": str(metadata["dataset_phase"]),
    }
    return batch, phase, audit


def _apportion_weighted_counts(batch_size: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("actor rehearsal stratum weights need a positive finite sum")
    quotas = tuple(batch_size * weight / total for weight in weights)
    counts = [math.floor(quota) for quota in quotas]
    remainder = batch_size - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


class ActorRehearsalReservoir:
    """Seal-once GPU tensor reservoir with an independent sampler RNG."""

    def __init__(
        self,
        *,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        device: torch.device | str,
        seed: int,
        source_fingerprints: Sequence[str],
        default_batch_size: int | None = None,
        stratum_weights: Mapping[int, float] | None = None,
    ) -> None:
        if capacity < 1 or observation_dim < 1 or action_dim < 1:
            raise ValueError("actor rehearsal capacity and dimensions must be positive")
        if default_batch_size is not None and default_batch_size < 1:
            raise ValueError("actor rehearsal default_batch_size must be positive")
        self._capacity = int(capacity)
        self._observation_dim = int(observation_dim)
        self._action_dim = int(action_dim)
        self._device = _canonical_device(device)
        self._source_fingerprints = _require_source_fingerprints(source_fingerprints)
        self._default_batch_size = (
            None if default_batch_size is None else int(default_batch_size)
        )
        self._requested_stratum_weights = (
            None
            if stratum_weights is None
            else {int(label): float(weight) for label, weight in stratum_weights.items()}
        )
        if self._requested_stratum_weights is not None:
            if not self._requested_stratum_weights:
                raise ValueError("actor rehearsal stratum_weights must not be empty")
            if any(
                not math.isfinite(weight) or weight < 0.0
                for weight in self._requested_stratum_weights.values()
            ):
                raise ValueError("actor rehearsal stratum weights must be finite and non-negative")
            if not any(weight > 0.0 for weight in self._requested_stratum_weights.values()):
                raise ValueError("at least one actor rehearsal stratum weight must be positive")

        self._storage = {
            "observation": torch.empty(
                (capacity, observation_dim), dtype=torch.float32, device=self._device
            ),
            "action": torch.empty(
                (capacity, action_dim), dtype=torch.float32, device=self._device
            ),
        }
        self._size = 0
        self._sealed = False
        self._phase: torch.Tensor | None = None
        self._stratum_values: tuple[int, ...] = ()
        self._stratum_indices: tuple[torch.Tensor, ...] = ()
        self._stratum_weights: tuple[tuple[int, float], ...] | None = None
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(int(seed))
        self._initial_seed = int(seed)
        self._sample_count = 0

    def __len__(self) -> int:
        return self._size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def source_fingerprints(self) -> tuple[str, ...]:
        return self._source_fingerprints

    @property
    def stratum_values(self) -> tuple[int, ...]:
        return self._stratum_values

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def add(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        phase: torch.Tensor | None = None,
    ) -> None:
        if self._sealed:
            raise RuntimeError("actor rehearsal reservoir is sealed and immutable")
        rows = _require_actor_batch(
            batch,
            observation_dim=self._observation_dim,
            action_dim=self._action_dim,
        )
        stop = self._size + rows
        if stop > self._capacity:
            raise OverflowError(
                f"actor rehearsal capacity exceeded: size={self._size}, add={rows}, "
                f"capacity={self._capacity}"
            )
        labels = _require_phase(phase, rows=rows, device=self._device)
        if self._size > 0 and (self._phase is None) != (labels is None):
            raise ValueError("all actor rehearsal sources must consistently include or omit phase")
        if self._size == 0 and labels is not None:
            self._phase = torch.empty(self._capacity, dtype=torch.int64, device=self._device)
        destination = slice(self._size, stop)
        for key in ACTOR_REHEARSAL_KEYS:
            self._storage[key][destination].copy_(
                batch[key].detach().to(
                    device=self._device,
                    dtype=self._storage[key].dtype,
                )
            )
        if labels is not None:
            assert self._phase is not None
            self._phase[destination].copy_(labels)
        self._size = stop

    def seal(self) -> None:
        if self._sealed:
            return
        if self._size < 1:
            raise RuntimeError("cannot seal an empty actor rehearsal reservoir")
        self._sealed = True
        self._rebuild_strata()
        self._stratum_weights = self._canonical_stratum_weights(
            self._requested_stratum_weights
        )

    def _rebuild_strata(self) -> None:
        if self._phase is None:
            self._stratum_values = ()
            self._stratum_indices = ()
            return
        unique = torch.unique(self._phase[: self._size], sorted=True)
        self._stratum_values = tuple(int(value) for value in unique.detach().cpu().tolist())
        self._stratum_indices = tuple(
            torch.nonzero(self._phase[: self._size] == value, as_tuple=False).flatten()
            for value in self._stratum_values
        )

    def _canonical_stratum_weights(
        self,
        weights: Mapping[int, float] | None,
    ) -> tuple[tuple[int, float], ...] | None:
        if weights is None:
            return None
        if self._phase is None:
            raise ValueError("actor rehearsal stratum weights require phase labels")
        unknown = sorted(set(weights).difference(self._stratum_values))
        if unknown:
            raise ValueError(f"actor rehearsal weights name absent phases: {unknown}")
        canonical = tuple(
            (label, float(weights.get(label, 0.0))) for label in self._stratum_values
        )
        if not any(weight > 0.0 for _, weight in canonical):
            raise ValueError("actor rehearsal weights select no available phase")
        return canonical

    def _sample_indices(self, batch_size: int) -> torch.Tensor:
        if self._phase is None:
            return torch.randint(
                0,
                self._size,
                (batch_size,),
                device=self._device,
                generator=self._generator,
            )
        weights = (
            tuple(1.0 for _ in self._stratum_values)
            if self._stratum_weights is None
            else tuple(weight for _, weight in self._stratum_weights)
        )
        counts = _apportion_weighted_counts(batch_size, weights)
        slots = torch.repeat_interleave(
            torch.arange(len(self._stratum_values), device=self._device),
            torch.tensor(counts, dtype=torch.int64, device=self._device),
        )
        if batch_size > 1:
            slots = slots[
                torch.randperm(batch_size, device=self._device, generator=self._generator)
            ]
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
                generator=self._generator,
            )
            sampled[positions] = source_indices[choices]
        return sampled

    def sample(self, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        if not self._sealed:
            raise RuntimeError("seal actor rehearsal reservoir before sampling")
        rows = self._default_batch_size if batch_size is None else int(batch_size)
        if rows is None:
            raise ValueError("sample requires batch_size when no default was configured")
        if rows < 1:
            raise ValueError("actor rehearsal sample batch_size must be positive")
        indices = self._sample_indices(rows)
        self._sample_count += 1
        return {key: value[indices] for key, value in self._storage.items()}

    def state_dict(self) -> dict[str, Any]:
        if not self._sealed:
            raise RuntimeError("seal actor rehearsal reservoir before checkpointing")
        return {
            "version": ACTOR_REHEARSAL_VERSION,
            "capacity": self._capacity,
            "observation_dim": self._observation_dim,
            "action_dim": self._action_dim,
            "device_type": str(self._device),
            "size": self._size,
            "sealed": self._sealed,
            "default_batch_size": self._default_batch_size,
            "source_fingerprints": self._source_fingerprints,
            "stratum_weights": self._stratum_weights,
            "initial_seed": self._initial_seed,
            "sample_count": self._sample_count,
            "generator_state": self._generator.get_state().clone(),
            "storage": {
                key: value[: self._size].clone() for key, value in self._storage.items()
            },
            "phase": None if self._phase is None else self._phase[: self._size].clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not self._sealed:
            raise RuntimeError(
                "build and seal actor rehearsal sources before restoring sampler state"
            )
        expected = {
            "version": ACTOR_REHEARSAL_VERSION,
            "capacity": self._capacity,
            "observation_dim": self._observation_dim,
            "action_dim": self._action_dim,
            "device_type": str(self._device),
            "size": self._size,
            "sealed": True,
            "default_batch_size": self._default_batch_size,
            "source_fingerprints": self._source_fingerprints,
            "stratum_weights": self._stratum_weights,
        }
        for key, value in expected.items():
            checkpoint_value = state.get(key)
            if checkpoint_value != value:
                raise ValueError(
                    f"actor rehearsal checkpoint {key}={checkpoint_value!r}, expected {value!r}"
                )
        storage = state.get("storage")
        if not isinstance(storage, Mapping):
            raise TypeError("actor rehearsal checkpoint storage must be a mapping")
        expected_shapes = {
            "observation": (self._size, self._observation_dim),
            "action": (self._size, self._action_dim),
        }
        for key, shape in expected_shapes.items():
            value = storage.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                actual = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
                raise ValueError(
                    f"actor rehearsal checkpoint {key} shape={actual}, expected {shape}"
                )
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"actor rehearsal checkpoint {key} is not finite floating data")
            self._storage[key][: self._size].copy_(
                value.to(device=self._device, dtype=self._storage[key].dtype)
            )
        checkpoint_phase = state.get("phase")
        if (checkpoint_phase is None) != (self._phase is None):
            raise ValueError("actor rehearsal checkpoint phase presence differs from sources")
        if checkpoint_phase is not None:
            labels = _require_phase(
                checkpoint_phase,
                rows=self._size,
                device=self._device,
            )
            assert labels is not None and self._phase is not None
            self._phase[: self._size].copy_(labels)
            self._rebuild_strata()
            if self._canonical_stratum_weights(self._requested_stratum_weights) != self._stratum_weights:
                raise ValueError("actor rehearsal checkpoint phase/configuration is inconsistent")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("actor rehearsal checkpoint generator_state must be a tensor")
        self._generator.set_state(generator_state.cpu())
        sample_count = state.get("sample_count")
        if not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError("actor rehearsal checkpoint sample_count must be non-negative")
        self._sample_count = sample_count
        initial_seed = state.get("initial_seed")
        if not isinstance(initial_seed, int):
            raise ValueError("actor rehearsal checkpoint initial_seed must be an integer")
        self._initial_seed = initial_seed

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        torch.save(self.state_dict(), temporary)
        os.replace(temporary, destination)

    def load(self, path: str | os.PathLike[str]) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        if not isinstance(state, Mapping):
            raise TypeError("actor rehearsal checkpoint root must be a mapping")
        self.load_state_dict(state)
