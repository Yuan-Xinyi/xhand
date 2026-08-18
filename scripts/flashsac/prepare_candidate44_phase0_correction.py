#!/usr/bin/env python3
"""Project the pinned full-task DAgger corpus into Candidate44 phase-0 labels.

The source stores corrective oracle actions on logged learner/mixed-policy
states.  These rows are actor supervision only: this projector deliberately
does not create reward, next-observation, terminal, discount, or success
fields.  Train/validation assignment is deterministic at source-episode
granularity and every retained row is physically unlatched phase 0.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch


SOURCE_SHA256 = "e4aaf0eda6a33db4a2ed04bc4d7609639da9b0471408808a5dae1b67760ce57f"
SOURCE_ROWS = 797450
SOURCE_EPISODES = 1685
ELIGIBLE_EPISODES = 917
ELIGIBLE_ROWS = 275081
TRAIN_EPISODES = 734
TRAIN_ROWS = 220181
VALIDATION_EPISODES = 183
VALIDATION_ROWS = 54900
LATCH_OBSERVATION_INDEX = 106
OBSERVATION_DIM = 115
ACTION_DIM = 21
SPLIT_SALT = "pick_tool_candidate44_phase0_correction_split_20260723_v1"
SPLIT_METHOD = "sha256_rank_source_episode_80_20_v1"
ACTION_LAYOUT = "arm_delta7|crossdex_token9|distal_residual5"
OBSERVATION_LAYOUT = "legacy_prefix87|distal_action5|grasp_transport23"
PHASE_NAMES = ["approach", "close", "micro", "lift", "settle"]
FORBIDDEN_PROJECTED_KEYS = frozenset(
    {
        "episode_success",
        "reward",
        "next_observation",
        "next_obs",
        "done",
        "terminated",
        "truncated",
        "discount",
    }
)
PROJECTED_KEYS = frozenset(
    {
        "obs",
        "action",
        "phase",
        "episode_id",
        "episode_offsets",
        "source_episode_id",
        "source_step",
        "source_row_index",
        "meta",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_tensor(
    payload: Mapping[str, Any],
    key: str,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"source requires tensor {key!r}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"source {key!r} shape={tuple(value.shape)}, expected {shape}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"source {key!r} dtype={value.dtype}, expected {dtype}")
    return value


def audit_source_payload(
    payload: Mapping[str, Any],
    *,
    source_sha256: str,
    expected_rows: int | None = SOURCE_ROWS,
    expected_episodes: int | None = SOURCE_EPISODES,
) -> tuple[int, int]:
    """Validate source tensors and episode segmentation before selection."""

    if not isinstance(payload, Mapping):
        raise TypeError("DAgger source root must be a mapping")
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError("source_sha256 must be lowercase hexadecimal SHA256")
    observation = _require_tensor(payload, "obs", dtype=torch.float32)
    action = _require_tensor(payload, "action", dtype=torch.float32)
    if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("DAgger source observation must have shape [N,115]")
    rows = int(observation.shape[0])
    if action.shape != (rows, ACTION_DIM):
        raise ValueError("DAgger source action must have shape [N,21]")
    phase = _require_tensor(payload, "phase", shape=(rows,))
    if phase.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("DAgger source phase must use an integer dtype")
    episode_id = _require_tensor(payload, "episode_id", shape=(rows,), dtype=torch.int64)
    step = _require_tensor(payload, "step", shape=(rows,))
    if step.dtype not in (torch.int16, torch.int32, torch.int64):
        raise TypeError("DAgger source step must use an integer dtype")
    offsets = _require_tensor(payload, "episode_offsets", dtype=torch.int64)
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("DAgger source episode_offsets must contain [0,...,N]")
    if int(offsets[0]) != 0 or int(offsets[-1]) != rows:
        raise ValueError("DAgger source episode_offsets do not cover every row")
    if bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError("DAgger source episode_offsets must be strictly increasing")
    episodes = int(offsets.numel() - 1)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"DAgger source rows={rows}, expected {expected_rows}")
    if expected_episodes is not None and episodes != expected_episodes:
        raise ValueError(f"DAgger source episodes={episodes}, expected {expected_episodes}")
    if not bool(torch.isfinite(observation).all()) or not bool(torch.isfinite(action).all()):
        raise ValueError("DAgger source contains NaN or infinity")
    if float(action.abs().max()) > 1.0001:
        raise ValueError("DAgger source action exceeds normalized [-1,1]")
    if bool((phase < 0).any()) or bool((phase >= len(PHASE_NAMES)).any()):
        raise ValueError("DAgger source phase is outside the five-phase contract")
    offset_values = offsets.tolist()
    for source_episode_id, (start, stop) in enumerate(
        zip(offset_values[:-1], offset_values[1:], strict=True)
    ):
        ids = episode_id[start:stop]
        if ids.numel() < 1 or bool((ids != source_episode_id).any()):
            raise ValueError(
                f"DAgger source episode_id disagrees with offsets at episode {source_episode_id}"
            )
        if int(step[start]) != 0 or bool((step[start + 1 : stop] <= step[start : stop - 1]).any()):
            raise ValueError(
                f"DAgger source step must start at zero and increase in episode {source_episode_id}"
            )
    metadata = payload.get("meta")
    if not isinstance(metadata, Mapping):
        raise TypeError("DAgger source requires mapping metadata")
    required_metadata = {
        "format_version": 1,
        "action_layout": ACTION_LAYOUT,
        "observation_layout": OBSERVATION_LAYOUT,
        "phase_names": PHASE_NAMES,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"DAgger source metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    return rows, episodes


def eligible_episode_rows(payload: Mapping[str, Any]) -> dict[int, torch.Tensor]:
    """Return retained absolute source-row indices keyed by source episode."""

    observation = payload["obs"]
    phase = payload["phase"]
    offsets = payload["episode_offsets"].tolist()
    result: dict[int, torch.Tensor] = {}
    for source_episode_id, (start, stop) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        selected = (phase[start:stop] == 0) & (
            observation[start:stop, LATCH_OBSERVATION_INDEX] == 0.0
        )
        local = selected.nonzero(as_tuple=False).flatten()
        if local.numel():
            result[source_episode_id] = local.to(dtype=torch.int64) + start
    return result


def rank_source_episodes(
    source_episode_ids: Sequence[int],
    *,
    source_sha256: str,
    split_salt: str = SPLIT_SALT,
) -> list[int]:
    if not split_salt:
        raise ValueError("split_salt must be non-empty")
    ranked: list[tuple[bytes, int]] = []
    for source_episode_id in source_episode_ids:
        if source_episode_id < 0:
            raise ValueError("source episode IDs must be non-negative")
        message = f"{split_salt}\0{source_sha256}\0{source_episode_id}".encode("utf-8")
        ranked.append((hashlib.sha256(message).digest(), int(source_episode_id)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [source_episode_id for _digest, source_episode_id in ranked]


def _projection_metadata(
    *,
    split: str,
    source_sha256: str,
    source_rows: int,
    source_episodes: int,
    split_episodes: int,
    split_rows: int,
    split_salt: str,
) -> dict[str, Any]:
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    return {
        "format_version": 1,
        "task_mode": "full_task",
        "observation_dim": OBSERVATION_DIM,
        "observation_contract": "pick_tool_markov115_v1",
        "observation_layout": OBSERVATION_LAYOUT,
        "action_dim": ACTION_DIM,
        "action_layout": ACTION_LAYOUT,
        "phase_names": PHASE_NAMES,
        "collector": "pick_tool_dagger_oracle_phase0_correction_v1",
        "dataset_phase": "approach",
        "action_semantics": "oracle_correction_label_on_logged_state_v1",
        "episode_semantics": "correction_labels_without_outcome_claim_v1",
        "critic_replay_eligible": False,
        "source_dataset_sha256": source_sha256,
        "source_dataset_rows": source_rows,
        "source_dataset_episodes": source_episodes,
        "selection": "phase_eq_0_and_observation_106_eq_0_v1",
        "split_method": SPLIT_METHOD,
        "split_salt": split_salt,
        "row_order": "hash_rank_episode_then_source_row_v1",
        "episode_id_semantics": "source_episode_id_v1",
        "split": split,
        "split_episodes": split_episodes,
        "split_rows": split_rows,
    }


def build_projection(
    payload: Mapping[str, Any],
    *,
    source_sha256: str,
    source_episode_ids: Sequence[int],
    split: str,
    split_salt: str = SPLIT_SALT,
) -> dict[str, Any]:
    retained = eligible_episode_rows(payload)
    if not source_episode_ids:
        raise ValueError("a projection split must contain at least one episode")
    if len(set(source_episode_ids)) != len(source_episode_ids):
        raise ValueError("projection source episode IDs must be unique")
    unknown = sorted(set(source_episode_ids).difference(retained))
    if unknown:
        raise ValueError(f"projection requests ineligible source episodes: {unknown[:5]}")

    observation_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    phase_parts: list[torch.Tensor] = []
    episode_id_parts: list[torch.Tensor] = []
    step_parts: list[torch.Tensor] = []
    row_index_parts: list[torch.Tensor] = []
    offsets = [0]
    for source_episode_id in source_episode_ids:
        indices = retained[int(source_episode_id)]
        observation_parts.append(payload["obs"].index_select(0, indices).clone())
        action_parts.append(payload["action"].index_select(0, indices).clone())
        phase_parts.append(payload["phase"].index_select(0, indices).to(torch.uint8).clone())
        episode_id_parts.append(
            torch.full((indices.numel(),), int(source_episode_id), dtype=torch.int64)
        )
        step_parts.append(payload["step"].index_select(0, indices).to(torch.int32).clone())
        row_index_parts.append(indices.clone())
        offsets.append(offsets[-1] + int(indices.numel()))

    rows = offsets[-1]
    episodes = len(source_episode_ids)
    projection: dict[str, Any] = {
        "obs": torch.cat(observation_parts, dim=0).contiguous(),
        "action": torch.cat(action_parts, dim=0).contiguous(),
        "phase": torch.cat(phase_parts, dim=0).contiguous(),
        "episode_id": torch.cat(episode_id_parts, dim=0).contiguous(),
        "episode_offsets": torch.tensor(offsets, dtype=torch.int64),
        "source_episode_id": torch.tensor(source_episode_ids, dtype=torch.int64),
        "source_step": torch.cat(step_parts, dim=0).contiguous(),
        "source_row_index": torch.cat(row_index_parts, dim=0).contiguous(),
        "meta": _projection_metadata(
            split=split,
            source_sha256=source_sha256,
            source_rows=int(payload["obs"].shape[0]),
            source_episodes=int(payload["episode_offsets"].numel() - 1),
            split_episodes=episodes,
            split_rows=rows,
            split_salt=split_salt,
        ),
    }
    validate_projection(
        projection,
        source_payload=payload,
        source_sha256=source_sha256,
        expected_split=split,
        expected_source_episode_ids=source_episode_ids,
        split_salt=split_salt,
    )
    return projection


def validate_projection(
    projection: Mapping[str, Any],
    *,
    source_payload: Mapping[str, Any],
    source_sha256: str,
    expected_split: str,
    expected_source_episode_ids: Sequence[int],
    split_salt: str = SPLIT_SALT,
) -> None:
    unexpected = sorted(set(projection).difference(PROJECTED_KEYS))
    missing = sorted(PROJECTED_KEYS.difference(projection))
    if unexpected or missing:
        raise ValueError(
            "actor-only projection keys differ from the closed contract: "
            f"missing={missing}, unexpected={unexpected}"
        )
    present_forbidden = sorted(FORBIDDEN_PROJECTED_KEYS.intersection(projection))
    if present_forbidden:
        raise ValueError(f"actor-only projection contains forbidden fields: {present_forbidden}")
    rows = len(projection["obs"])
    episodes = len(expected_source_episode_ids)
    observation = _require_tensor(
        projection, "obs", shape=(rows, OBSERVATION_DIM), dtype=torch.float32
    )
    action = _require_tensor(
        projection, "action", shape=(rows, ACTION_DIM), dtype=torch.float32
    )
    phase = _require_tensor(projection, "phase", shape=(rows,), dtype=torch.uint8)
    episode_id = _require_tensor(
        projection, "episode_id", shape=(rows,), dtype=torch.int64
    )
    offsets = _require_tensor(
        projection, "episode_offsets", shape=(episodes + 1,), dtype=torch.int64
    )
    source_episode_id = _require_tensor(
        projection, "source_episode_id", shape=(episodes,), dtype=torch.int64
    )
    source_step = _require_tensor(
        projection, "source_step", shape=(rows,), dtype=torch.int32
    )
    source_row_index = _require_tensor(
        projection, "source_row_index", shape=(rows,), dtype=torch.int64
    )
    retained = eligible_episode_rows(source_payload)
    try:
        expected_row_parts = [retained[int(value)] for value in expected_source_episode_ids]
    except KeyError as error:
        raise ValueError(
            f"projection requests an ineligible source episode: {int(error.args[0])}"
        ) from error
    expected_source_rows = torch.cat(expected_row_parts, dim=0)
    if not torch.equal(source_row_index, expected_source_rows):
        raise ValueError("projection source rows differ from the complete eligible selection")
    expected_offsets = [0]
    for part in expected_row_parts:
        expected_offsets.append(expected_offsets[-1] + int(part.numel()))
    if not torch.equal(offsets, torch.tensor(expected_offsets, dtype=torch.int64)):
        raise ValueError("projection episode_offsets differ from the eligible source rows")
    if int(offsets[0]) != 0 or int(offsets[-1]) != rows or bool(
        (offsets[1:] <= offsets[:-1]).any()
    ):
        raise ValueError("projection episode_offsets are invalid")
    expected_ids = torch.tensor(expected_source_episode_ids, dtype=torch.int64)
    if not torch.equal(source_episode_id, expected_ids):
        raise ValueError("projection source_episode_id order differs from split rank")
    if not bool((phase == 0).all()):
        raise ValueError("projection contains a non-phase0 row")
    if not bool((observation[:, LATCH_OBSERVATION_INDEX] == 0.0).all()):
        raise ValueError("projection contains a latched phase0 observation")
    if not bool(torch.isfinite(observation).all()) or not bool(torch.isfinite(action).all()):
        raise ValueError("projection contains NaN or infinity")
    if float(action.abs().max()) > 1.0001:
        raise ValueError("projection action exceeds normalized [-1,1]")
    if bool((source_row_index < 0).any()) or bool(
        (source_row_index >= source_payload["obs"].shape[0]).any()
    ):
        raise ValueError("projection source_row_index is out of range")
    torch.testing.assert_close(
        observation,
        source_payload["obs"].index_select(0, source_row_index),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        action,
        source_payload["action"].index_select(0, source_row_index),
        rtol=0.0,
        atol=0.0,
    )
    if not torch.equal(
        source_step,
        source_payload["step"].index_select(0, source_row_index).to(torch.int32),
    ):
        raise ValueError("projection source_step does not reconstruct from source")
    offset_values = offsets.tolist()
    for index, (start, stop) in enumerate(
        zip(offset_values[:-1], offset_values[1:], strict=True)
    ):
        source_id = int(source_episode_id[index])
        if bool((episode_id[start:stop] != source_id).any()):
            raise ValueError(f"projection episode_id changes in segment {index}")
        source_start = int(source_payload["episode_offsets"][source_id])
        source_stop = int(source_payload["episode_offsets"][source_id + 1])
        segment_rows = source_row_index[start:stop]
        if bool((segment_rows < source_start).any()) or bool((segment_rows >= source_stop).any()):
            raise ValueError(f"projection segment {index} escapes its source episode")
        if segment_rows.numel() > 1 and bool((segment_rows[1:] <= segment_rows[:-1]).any()):
            raise ValueError(f"projection segment {index} does not preserve source row order")
    expected_metadata = _projection_metadata(
        split=expected_split,
        source_sha256=source_sha256,
        source_rows=int(source_payload["obs"].shape[0]),
        source_episodes=int(source_payload["episode_offsets"].numel() - 1),
        split_episodes=episodes,
        split_rows=rows,
        split_salt=split_salt,
    )
    if projection.get("meta") != expected_metadata:
        raise ValueError("projection metadata differs from the closed contract")


def validate_projection_pair(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    eligible_source_episode_ids: Sequence[int],
) -> None:
    train_ids = set(int(value) for value in train["source_episode_id"].tolist())
    validation_ids = set(int(value) for value in validation["source_episode_id"].tolist())
    if train_ids.intersection(validation_ids):
        raise ValueError("train and validation projections overlap in source episodes")
    if train_ids.union(validation_ids) != set(eligible_source_episode_ids):
        raise ValueError("train/validation projections do not partition eligible episodes")


def _path_is_present(path: Path) -> bool:
    """Include broken symlinks in no-clobber checks."""

    return path.exists() or path.is_symlink()


def _require_regular_file(path: Path, *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file: {path}")


def _assert_exact_value(actual: Any, expected: Any, *, location: str = "root") -> None:
    """Require exact tensor values and an exact nested Python payload schema."""

    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise ValueError(f"{location} is not a tensor")
        if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
            raise ValueError(
                f"{location} tensor contract differs: "
                f"actual={actual.dtype}{tuple(actual.shape)}, "
                f"expected={expected.dtype}{tuple(expected.shape)}"
            )
        if not torch.equal(actual, expected):
            raise ValueError(f"{location} tensor values differ")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"{location} is not a mapping")
        if set(actual) != set(expected):
            raise ValueError(
                f"{location} mapping keys differ: "
                f"actual={sorted(actual)}, expected={sorted(expected)}"
            )
        for key in expected:
            _assert_exact_value(
                actual[key], expected[key], location=f"{location}.{key}"
            )
        return
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"{location} sequence contract differs")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact_value(
                actual_item, expected_item, location=f"{location}[{index}]"
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{location} value differs: {actual!r} != {expected!r}")


def _atomic_copy_no_clobber(source: Path, destination: Path, *, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _path_is_present(destination):
        _require_regular_file(destination, label="canonical source")
        if sha256_file(destination) != expected_sha256:
            raise FileExistsError(f"canonical source exists with different bytes: {destination}")
        return "verified_existing"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise RuntimeError("canonical source copy changed bytes")
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "created"


def _load_exact_torch_payload(path: Path, expected: Mapping[str, Any]) -> None:
    _require_regular_file(path, label="projected actor dataset")
    try:
        actual = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"cannot load existing projected actor dataset: {path}") from error
    _assert_exact_value(actual, expected)


def _atomic_torch_save_or_verify(payload: Mapping[str, Any], path: Path) -> str:
    """Publish without overwrite, or exactly revalidate an existing projection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if _path_is_present(path):
        _load_exact_torch_payload(path, payload)
        return "verified_existing"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if _path_is_present(temporary):
        raise FileExistsError(temporary)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        _load_exact_torch_payload(temporary, payload)
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary.exists():
            temporary.unlink()
    _load_exact_torch_payload(path, payload)
    return "created"


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    return text.encode("utf-8")


def _atomic_json_save_or_verify(payload: Mapping[str, Any], path: Path) -> str:
    """Publish canonical JSON, or require byte-for-byte equality if it exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = _canonical_json_bytes(payload)
    if _path_is_present(path):
        _require_regular_file(path, label="data manifest")
        if path.read_bytes() != expected_bytes:
            raise FileExistsError(f"data manifest exists with different bytes: {path}")
        return "verified_existing"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if _path_is_present(temporary):
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as stream:
            stream.write(expected_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.read_bytes() != expected_bytes:
        raise RuntimeError("published data manifest changed bytes")
    return "created"


def build_manifest(
    *,
    source: Path,
    canonical_source: Path,
    source_rows: int,
    source_episodes: int,
    train_path: Path,
    train: Mapping[str, Any],
    validation_path: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build stable sealed-data metadata; runtime publication status is excluded."""

    return {
        "kind": "pick_tool_candidate44_phase0_correction_data_manifest_v1",
        "status": "complete",
        "source": {
            "input_path": str(source),
            "canonical_path": str(canonical_source),
            "copy_contract": "sha256_verified_no_clobber_v1",
            "sha256": sha256_file(canonical_source),
            "bytes": canonical_source.stat().st_size,
            "rows": source_rows,
            "episodes": source_episodes,
        },
        "split": {
            "method": SPLIT_METHOD,
            "salt": SPLIT_SALT,
            "eligible_rows": ELIGIBLE_ROWS,
            "eligible_episodes": ELIGIBLE_EPISODES,
        },
        "train": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "bytes": train_path.stat().st_size,
            "rows": int(train["obs"].shape[0]),
            "episodes": int(train["source_episode_id"].numel()),
        },
        "validation": {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
            "bytes": validation_path.stat().st_size,
            "rows": int(validation["obs"].shape[0]),
            "episodes": int(validation["source_episode_id"].numel()),
        },
        "train_validation_source_episode_overlap": 0,
        "forbidden_transition_or_outcome_fields_present": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/tmp/pick_tool_ik_dagger_full_iter3_1685ep.pt"),
    )
    parser.add_argument(
        "--canonical_source",
        type=Path,
        default=Path(
            "logs/flashsac/pick_tool/demos/source/"
            "pick_tool_ik_dagger_full_iter3_1685ep.pt"
        ),
    )
    parser.add_argument(
        "--train_output",
        type=Path,
        default=Path("logs/flashsac/pick_tool/demos/c44_dagger_phase0_train.pt"),
    )
    parser.add_argument(
        "--validation_output",
        type=Path,
        default=Path(
            "logs/flashsac/pick_tool/demos/c44_dagger_phase0_validation.pt"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "logs/flashsac/pick_tool/demos/c44_dagger_phase0_manifest.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    canonical_source = args.canonical_source.expanduser().resolve()
    train_path = args.train_output.expanduser().resolve()
    validation_path = args.validation_output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    manifest_preexists = _path_is_present(manifest_path)
    if manifest_preexists:
        _require_regular_file(manifest_path, label="data manifest")
        # A manifest is the final seal.  Once it exists, a missing member is
        # corruption rather than a partial publication that may be repaired.
        for member_label, member_path in (
            ("canonical source", canonical_source),
            ("train projection", train_path),
            ("validation projection", validation_path),
        ):
            _require_regular_file(member_path, label=member_label)
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("DAgger source SHA256 differs from Candidate44 preregistration")
    copy_status = _atomic_copy_no_clobber(
        source, canonical_source, expected_sha256=SOURCE_SHA256
    )
    payload = torch.load(canonical_source, map_location="cpu", weights_only=True)
    source_rows, source_episodes = audit_source_payload(
        payload,
        source_sha256=SOURCE_SHA256,
    )
    eligible = eligible_episode_rows(payload)
    if len(eligible) != ELIGIBLE_EPISODES or sum(
        int(indices.numel()) for indices in eligible.values()
    ) != ELIGIBLE_ROWS:
        raise RuntimeError("eligible phase-0 population differs from preregistration")
    ranked = rank_source_episodes(
        list(eligible), source_sha256=SOURCE_SHA256, split_salt=SPLIT_SALT
    )
    train_ids = ranked[:TRAIN_EPISODES]
    validation_ids = ranked[TRAIN_EPISODES:]
    train = build_projection(
        payload,
        source_sha256=SOURCE_SHA256,
        source_episode_ids=train_ids,
        split="train",
    )
    validation = build_projection(
        payload,
        source_sha256=SOURCE_SHA256,
        source_episode_ids=validation_ids,
        split="validation",
    )
    if train["obs"].shape[0] != TRAIN_ROWS or validation["obs"].shape[0] != VALIDATION_ROWS:
        raise RuntimeError("hash-ranked split row counts differ from preregistration")
    validate_projection_pair(
        train,
        validation,
        eligible_source_episode_ids=ranked,
    )
    train_status = _atomic_torch_save_or_verify(train, train_path)
    validation_status = _atomic_torch_save_or_verify(validation, validation_path)

    reloaded_train = torch.load(train_path, map_location="cpu", weights_only=True)
    reloaded_validation = torch.load(
        validation_path, map_location="cpu", weights_only=True
    )
    validate_projection(
        reloaded_train,
        source_payload=payload,
        source_sha256=SOURCE_SHA256,
        expected_split="train",
        expected_source_episode_ids=train_ids,
    )
    validate_projection(
        reloaded_validation,
        source_payload=payload,
        source_sha256=SOURCE_SHA256,
        expected_split="validation",
        expected_source_episode_ids=validation_ids,
    )
    validate_projection_pair(
        reloaded_train,
        reloaded_validation,
        eligible_source_episode_ids=ranked,
    )
    manifest = build_manifest(
        source=source,
        canonical_source=canonical_source,
        source_rows=source_rows,
        source_episodes=source_episodes,
        train_path=train_path,
        train=reloaded_train,
        validation_path=validation_path,
        validation=reloaded_validation,
    )
    manifest_status = _atomic_json_save_or_verify(manifest, manifest_path)
    publication_status = {
        "canonical_source": copy_status,
        "train": train_status,
        "validation": validation_status,
        "manifest": manifest_status,
    }
    print(
        "publication_status="
        + json.dumps(publication_status, sort_keys=True, allow_nan=False)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
