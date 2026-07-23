#!/usr/bin/env python3
"""Merge episode-complete pick-tool oracle datasets without mixing episode identities."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from typing import Any

import torch


_CORE_EPISODE_KEYS = {"episode_id", "episode_offsets"}
_PHASE_NAME_KEYS = ("phase_names", "option_names")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run deterministic synthetic compatibility tests and exit",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: dataset root is not a dictionary")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_names(path: Path, data: dict[str, Any]) -> tuple[str, list[str]]:
    meta = data.get("meta")
    if not isinstance(meta, dict):
        raise TypeError(f"{path}: meta must be a dictionary")
    for key in _PHASE_NAME_KEYS:
        names = meta.get(key)
        if names is None:
            continue
        if not isinstance(names, (list, tuple)) or not names or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{path}: meta.{key} must be a non-empty sequence of strings")
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: meta.{key} contains duplicate names")
        return key, list(names)
    raise KeyError(f"{path}: meta must contain one of {_PHASE_NAME_KEYS}")


def validate(path: Path, data: dict[str, Any]) -> None:
    required = ("obs", "action", "phase", "episode_id", "step", "episode_offsets")
    for key in required:
        if key not in data or not isinstance(data[key], torch.Tensor):
            raise KeyError(f"{path}: missing tensor {key!r}")
    transitions = data["obs"].shape[0]
    if data["obs"].ndim != 2 or data["obs"].shape[1] not in (115, 120):
        raise ValueError(f"{path}: expected obs [N,115] or [N,120], got {tuple(data['obs'].shape)}")
    if data["action"].shape != (transitions, 21):
        raise ValueError(f"{path}: expected action [N,21], got {tuple(data['action'].shape)}")
    for key in ("phase", "episode_id", "step"):
        if data[key].shape != (transitions,):
            raise ValueError(f"{path}: {key} has shape {tuple(data[key].shape)}, expected [{transitions}]")
    if data["phase"].dtype == torch.bool or data["phase"].is_floating_point():
        raise TypeError(f"{path}: phase must use an integer dtype")

    phase_key, names = _phase_names(path, data)
    if transitions:
        phase_min = int(data["phase"].min())
        phase_max = int(data["phase"].max())
        if phase_min < 0 or phase_max >= len(names):
            raise ValueError(
                f"{path}: phase ids [{phase_min},{phase_max}] are outside meta.{phase_key} "
                f"with {len(names)} entries"
            )

    offsets = data["episode_offsets"].long()
    if offsets.ndim != 1 or offsets.numel() < 2 or int(offsets[0]) != 0:
        raise ValueError(f"{path}: invalid episode_offsets")
    if int(offsets[-1]) != transitions or not bool(torch.all(offsets[1:] > offsets[:-1])):
        raise ValueError(f"{path}: episode_offsets do not partition all transitions")
    episodes = data["episode_id"].long().unique(sorted=True)
    expected = torch.arange(offsets.numel() - 1)
    if not torch.equal(episodes, expected):
        raise ValueError(f"{path}: episode ids must be dense from zero")
    for episode in range(episodes.numel()):
        start, end = int(offsets[episode]), int(offsets[episode + 1])
        if not bool(torch.all(data["episode_id"][start:end] == episode)):
            raise ValueError(f"{path}: episode {episode} is not contiguous")
        if not torch.equal(data["step"][start:end].long(), torch.arange(end - start)):
            raise ValueError(f"{path}: episode {episode} has invalid step indices")
    if not torch.isfinite(data["obs"]).all() or not torch.isfinite(data["action"]).all():
        raise ValueError(f"{path}: non-finite observation or action")

    meta = data["meta"]
    if "observation_dim" in meta and int(meta["observation_dim"]) != data["obs"].shape[1]:
        raise ValueError(f"{path}: meta.observation_dim disagrees with obs")
    if "action_dim" in meta and int(meta["action_dim"]) != data["action"].shape[1]:
        raise ValueError(f"{path}: meta.action_dim disagrees with action")


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_values_equal(a, b) for a, b in zip(left, right, strict=True))
    try:
        equal = left == right
        return bool(equal) if not isinstance(equal, torch.Tensor) else bool(torch.all(equal))
    except (TypeError, ValueError):
        return False


def _common_meta(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    metas = [data["meta"] for data in datasets]
    return {
        key: value
        for key, value in metas[0].items()
        if key != "merged_sources" and all(key in meta and _values_equal(value, meta[key]) for meta in metas[1:])
    }


def _tensor_compatibility(
    field: str,
    values: list[Any],
    episode_counts: list[int],
) -> tuple[list[torch.Tensor] | None, dict[str, Any] | None]:
    missing = [source_id for source_id, value in enumerate(values) if value is None]
    if missing:
        return None, {"field": field, "reason": "missing", "source_ids": missing}
    non_tensors = [source_id for source_id, value in enumerate(values) if not isinstance(value, torch.Tensor)]
    if non_tensors:
        return None, {"field": field, "reason": "not_tensor", "source_ids": non_tensors}

    tensors: list[torch.Tensor] = values
    bad_axes = [
        source_id
        for source_id, (tensor, episodes) in enumerate(zip(tensors, episode_counts, strict=True))
        if tensor.ndim < 1 or tensor.shape[0] != episodes
    ]
    if bad_axes:
        return None, {"field": field, "reason": "invalid_episode_axis", "source_ids": bad_axes}
    dtypes = [tensor.dtype for tensor in tensors]
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        return None, {"field": field, "reason": "dtype_mismatch", "dtypes": [str(dtype) for dtype in dtypes]}
    trailing_shapes = [tuple(tensor.shape[1:]) for tensor in tensors]
    if any(shape != trailing_shapes[0] for shape in trailing_shapes[1:]):
        return None, {"field": field, "reason": "shape_mismatch", "shapes": trailing_shapes}
    return [tensor.cpu() for tensor in tensors], None


def _merge_optional_episode_tensors(
    datasets: list[dict[str, Any]],
    episode_counts: list[int],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    candidates = sorted(
        set().union(
            *(
                {key for key in data if key.startswith("episode_") and key not in _CORE_EPISODE_KEYS}
                for data in datasets
            )
        )
    )
    merged: dict[str, torch.Tensor] = {}
    omitted = []
    for key in candidates:
        parts, issue = _tensor_compatibility(key, [data.get(key) for data in datasets], episode_counts)
        if issue is not None:
            omitted.append(issue)
        else:
            assert parts is not None
            merged[key] = torch.cat(parts)
    return merged, omitted


def _merge_boundaries(
    datasets: list[dict[str, Any]],
    episode_counts: list[int],
) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    boundary_maps = [data.get("boundaries") if isinstance(data.get("boundaries"), dict) else {} for data in datasets]
    phase_names = sorted(set().union(*(set(boundaries) for boundaries in boundary_maps)))
    merged: dict[str, dict[str, torch.Tensor]] = {}
    omitted: list[dict[str, Any]] = []
    for phase_name in phase_names:
        state_maps = [
            boundaries.get(phase_name) if isinstance(boundaries.get(phase_name), dict) else {}
            for boundaries in boundary_maps
        ]
        field_names = sorted(set().union(*(set(state) for state in state_maps)))
        merged_state: dict[str, torch.Tensor] = {}
        for key in field_names:
            field = f"boundaries.{phase_name}.{key}"
            parts, issue = _tensor_compatibility(field, [state.get(key) for state in state_maps], episode_counts)
            if issue is not None:
                omitted.append(issue)
            else:
                assert parts is not None
                merged_state[key] = torch.cat(parts)
        if merged_state:
            merged[phase_name] = merged_state
    return merged, omitted


def merge_datasets(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one input dataset is required")
    datasets = []
    shas = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        data = load(path)
        validate(path, data)
        datasets.append(data)
        shas.append(sha256(path))

    # Merging identical (or overlapping) inputs renumbers the same trajectories as distinct
    # episodes; the episode-disjoint BC split could then place a copy in train and its twin in
    # val, silently inflating the "episode-isolated" validation objective.  Reject exact dupes.
    if len(set(shas)) != len(shas):
        seen: dict[str, Path] = {}
        duplicates = []
        for path, sha in zip(paths, shas, strict=True):
            if sha in seen:
                duplicates.append(f"{seen[sha]} == {path}")
            else:
                seen[sha] = path
        raise ValueError(f"duplicate input datasets (identical content): {duplicates}")

    obs_dims = [int(data["obs"].shape[1]) for data in datasets]
    if any(obs_dim != obs_dims[0] for obs_dim in obs_dims[1:]):
        raise ValueError(f"input observation dimensions must agree, got {obs_dims}")
    phase_schemas = [_phase_names(path, data) for path, data in zip(paths, datasets, strict=True)]
    phase_names = phase_schemas[0][1]
    if any(names != phase_names for _, names in phase_schemas[1:]):
        raise ValueError(f"input phase/option names must agree, got {[names for _, names in phase_schemas]}")
    # Dimensions and phase names agreeing is not enough: two datasets can share a 115-D shape but
    # lay out those columns differently.  Require the declared layout strings to agree when present.
    for layout_key in ("observation_layout", "action_layout"):
        layouts = {
            data.get("meta", {}).get(layout_key)
            for data in datasets
            if isinstance(data.get("meta"), dict) and data["meta"].get(layout_key) is not None
        }
        if len(layouts) > 1:
            raise ValueError(f"input {layout_key} strings disagree: {sorted(layouts)}")
    # Preserve the legacy compact representation when it is sufficient, but do
    # not silently wrap a future schema with more than 256 phase ids.
    phase_dtype = torch.uint8 if len(phase_names) <= 256 else torch.int64

    obs_parts = []
    action_parts = []
    phase_parts = []
    episode_parts = []
    step_parts = []
    source_parts = []
    offsets = [0]
    episode_counts = []
    episode_base = 0
    source_meta = []
    for source_id, (path, data) in enumerate(zip(paths, datasets, strict=True)):
        episodes = data["episode_offsets"].numel() - 1
        transitions = data["obs"].shape[0]
        episode_counts.append(episodes)
        obs_parts.append(data["obs"].float())
        action_parts.append(data["action"].float())
        phase_parts.append(data["phase"].to(phase_dtype))
        episode_parts.append(data["episode_id"].to(torch.int64) + episode_base)
        step_parts.append(data["step"].to(torch.int32))
        source_parts.append(torch.full((transitions,), source_id, dtype=torch.int16))
        lengths = data["episode_offsets"][1:] - data["episode_offsets"][:-1]
        for length in lengths.tolist():
            offsets.append(offsets[-1] + int(length))
        source_meta.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "episodes": episodes,
                "transitions": transitions,
                "meta": data.get("meta", {}),
            }
        )
        episode_base += episodes

    merged_episode_tensors, episode_omissions = _merge_optional_episode_tensors(datasets, episode_counts)
    merged_boundaries, boundary_omissions = _merge_boundaries(datasets, episode_counts)
    merged_meta = _common_meta(datasets)
    phase_name_key = phase_schemas[0][0]
    merged_meta[phase_name_key] = phase_names
    merged_meta["observation_dim"] = obs_dims[0]
    merged_meta["action_dim"] = 21
    merged_meta["merged_sources"] = source_meta
    merged_meta["omitted_fields"] = episode_omissions + boundary_omissions

    merged = {
        "obs": torch.cat(obs_parts),
        "action": torch.cat(action_parts),
        "phase": torch.cat(phase_parts),
        "episode_id": torch.cat(episode_parts),
        "step": torch.cat(step_parts),
        "source_dataset_id": torch.cat(source_parts),
        "episode_offsets": torch.tensor(offsets, dtype=torch.int64),
        **merged_episode_tensors,
        "boundaries": merged_boundaries,
        "meta": merged_meta,
    }
    validate(Path("<merged>"), merged)
    return merged


def _synthetic_dataset(obs_dim: int, episodes: int, source_id: int, *, option_names: bool) -> dict[str, Any]:
    lengths = torch.arange(2, 2 + episodes, dtype=torch.int64)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    transitions = int(offsets[-1])
    episode_id = torch.repeat_interleave(torch.arange(episodes, dtype=torch.int32), lengths)
    step = torch.cat([torch.arange(int(length), dtype=torch.int32) for length in lengths])
    name_key = "option_names" if option_names else "phase_names"
    names = ["HOVER", "DESCEND", "CLOSE"] if option_names else ["approach", "close", "lift"]
    phase = (step % len(names)).to(torch.uint8)
    data: dict[str, Any] = {
        "obs": torch.full((transitions, obs_dim), float(source_id), dtype=torch.float32),
        "action": torch.full((transitions, 21), float(source_id), dtype=torch.float32),
        "phase": phase,
        "episode_id": episode_id,
        "step": step,
        "episode_offsets": offsets,
        "episode_success": torch.arange(episodes) % 2 == 0,
        "episode_outcome": torch.arange(episodes, dtype=torch.int16) + source_id,
        "episode_terminal_force": torch.arange(episodes, dtype=torch.float32),
        "boundaries": {
            "close_start": {
                "joint_pos": torch.full((episodes, 3), float(source_id)),
                "dtype_conflict": torch.zeros(episodes, dtype=torch.float32 if source_id == 0 else torch.float64),
            }
        },
        "meta": {
            "format_version": 2 if option_names else 1,
            "observation_layout": "synthetic120" if option_names else "synthetic115",
            "observation_dim": obs_dim,
            "action_dim": 21,
            name_key: names,
        },
    }
    if source_id == 0:
        data["episode_only_first"] = torch.zeros(episodes)
        data["boundaries"]["close_start"]["only_first"] = torch.zeros(episodes)
    else:
        # Common name, deliberately incompatible trailing shape.
        data["episode_terminal_force"] = torch.zeros(episodes, 1)
    return data


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="merge_oracle_test_") as tmp:
        tmp_path = Path(tmp)
        paths120 = [tmp_path / "state0.pt", tmp_path / "state1.pt"]
        torch.save(_synthetic_dataset(120, 2, 0, option_names=True), paths120[0])
        torch.save(_synthetic_dataset(120, 1, 1, option_names=True), paths120[1])
        merged120 = merge_datasets(paths120)
        assert merged120["obs"].shape == (7, 120)
        assert merged120["episode_offsets"].tolist() == [0, 2, 5, 7]
        assert merged120["episode_success"].shape == (3,)
        assert merged120["episode_outcome"].shape == (3,)
        assert "episode_terminal_force" not in merged120
        assert "episode_only_first" not in merged120
        assert merged120["boundaries"]["close_start"]["joint_pos"].shape == (3, 3)
        assert "dtype_conflict" not in merged120["boundaries"]["close_start"]
        omitted = {entry["field"]: entry["reason"] for entry in merged120["meta"]["omitted_fields"]}
        assert omitted["episode_terminal_force"] == "shape_mismatch"
        assert omitted["episode_only_first"] == "missing"
        assert omitted["boundaries.close_start.dtype_conflict"] == "dtype_mismatch"
        assert omitted["boundaries.close_start.only_first"] == "missing"
        assert merged120["meta"]["option_names"] == ["HOVER", "DESCEND", "CLOSE"]

        path115 = tmp_path / "legacy.pt"
        path115b = tmp_path / "legacy_b.pt"
        torch.save(_synthetic_dataset(115, 1, 0, option_names=False), path115)
        torch.save(_synthetic_dataset(115, 1, 7, option_names=False), path115b)
        merged115 = merge_datasets([path115, path115b])
        assert merged115["obs"].shape == (4, 115)
        assert merged115["meta"]["phase_names"] == ["approach", "close", "lift"]
        try:
            merge_datasets([path115, path115])
        except ValueError as error:
            assert "duplicate input datasets" in str(error)
        else:
            raise AssertionError("exact-duplicate inputs were not rejected")
        try:
            merge_datasets([path115, paths120[0]])
        except ValueError as error:
            assert "observation dimensions must agree" in str(error)
        else:
            raise AssertionError("mixed 115/120 observations were not rejected")
    print("self-test passed: 115/120 schemas, episode tensors, boundaries, and omissions")


def main() -> None:
    args = parse_args()
    if args.self_test:
        if args.inputs or args.output is not None:
            raise ValueError("--self-test does not accept inputs or --output")
        self_test()
        return
    if not args.inputs or args.output is None:
        raise ValueError("normal merge requires one or more inputs and --output")

    merged = merge_datasets(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(
        f"wrote {args.output.resolve()}: {merged['episode_offsets'].numel() - 1} episodes, "
        f"{merged['obs'].shape[0]} transitions from {len(args.inputs)} datasets; "
        f"omitted {len(merged['meta']['omitted_fields'])} incompatible optional fields"
    )


if __name__ == "__main__":
    main()
