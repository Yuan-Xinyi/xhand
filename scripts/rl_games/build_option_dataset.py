#!/usr/bin/env python3
"""Build an option-conditioned pick-tool dataset from 115-D oracle/DAgger data.

The output keeps complete episode identities while filtering unsafe transition labels.  Its policy
observation is ``source_obs115 | option_onehot5`` and the historical ``phase`` tensor is replaced
by the five-way option id used by that one-hot suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SOURCE_OBSERVATION_DIM = 115
OPTION_COUNT = 5
ACTION_DIM = 21

PHASE_APPROACH = 0
PHASE_CLOSE = 1
PHASE_MICRO = 2
PHASE_LIFT = 3
PHASE_SETTLE = 4

OPTION_HOVER = 0
OPTION_DESCEND = 1
OPTION_CLOSE = 2
OPTION_MICRO = 3
OPTION_LIFT_HOLD = 4

OPTION_NAMES = ("HOVER", "DESCEND", "CLOSE", "MICRO", "LIFT_HOLD")
SOURCE_PHASE_NAMES = ("approach", "close", "micro", "lift", "settle")

# Zero-based columns in the 115-D grasp/transport observation.
NONTHUMB_FORCE = slice(97, 101)
THUMB_FORCE = 101
CLOSE_QUALITY = 102
WRAP_QUALITY = 103
HOLD_QUALITY = 104
LATCH = 106

MICRO_CLOSE_MIN = 0.20
MICRO_HOLD_MIN = 0.50
MICRO_FORCE_MIN = math.tanh(0.2 / 5.0)
LIFT_HOLD_QUALITY_MIN = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="115-D oracle/DAgger .pt datasets")
    parser.add_argument("--output", required=True, type=Path, help="output 120-D option dataset")
    return parser.parse_args()


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        return torch.load(path, map_location="cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count(values: torch.Tensor, value: int) -> int:
    return int((values == value).sum())


def _phase_counts(phase: torch.Tensor, keep: torch.Tensor | None = None) -> dict[str, int]:
    if keep is not None:
        phase = phase[keep]
    return {name: _count(phase, index) for index, name in enumerate(SOURCE_PHASE_NAMES)}


def validate_dataset(path: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path}: dataset root must be a dictionary")
    required = ("obs", "action", "phase", "episode_id", "step", "episode_offsets")
    for key in required:
        if key not in value or not isinstance(value[key], torch.Tensor):
            raise KeyError(f"{path}: missing tensor {key!r}")

    obs = value["obs"].detach().cpu().float()
    action = value["action"].detach().cpu().float()
    phase = value["phase"].detach().cpu().long().flatten()
    episode_id = value["episode_id"].detach().cpu().long().flatten()
    step = value["step"].detach().cpu().long().flatten()
    offsets = value["episode_offsets"].detach().cpu().long().flatten()
    transitions = obs.shape[0]

    if obs.ndim != 2 or obs.shape[1] != SOURCE_OBSERVATION_DIM:
        raise ValueError(
            f"{path}: expected obs [N,{SOURCE_OBSERVATION_DIM}], got {tuple(obs.shape)}"
        )
    if action.shape != (transitions, ACTION_DIM):
        raise ValueError(f"{path}: expected action [N,{ACTION_DIM}], got {tuple(action.shape)}")
    for key, tensor in (("phase", phase), ("episode_id", episode_id), ("step", step)):
        if tensor.shape != (transitions,):
            raise ValueError(
                f"{path}: {key} has shape {tuple(tensor.shape)}, expected [{transitions}]"
            )
    if offsets.ndim != 1 or offsets.numel() < 2 or int(offsets[0]) != 0:
        raise ValueError(f"{path}: invalid episode_offsets")
    if int(offsets[-1]) != transitions or not bool(torch.all(offsets[1:] > offsets[:-1])):
        raise ValueError(f"{path}: episode_offsets do not partition all transitions")
    if not torch.isfinite(obs).all() or not torch.isfinite(action).all():
        raise ValueError(f"{path}: non-finite observation or action")
    if not bool(torch.all((phase >= PHASE_APPROACH) & (phase <= PHASE_SETTLE))):
        invalid = phase[(phase < PHASE_APPROACH) | (phase > PHASE_SETTLE)].unique().tolist()
        raise ValueError(f"{path}: unsupported source phase ids {invalid}")

    episodes = offsets.numel() - 1
    if not torch.equal(episode_id.unique(sorted=True), torch.arange(episodes)):
        raise ValueError(f"{path}: episode ids must be dense from zero")
    for episode in range(episodes):
        start, end = int(offsets[episode]), int(offsets[episode + 1])
        if not bool(torch.all(episode_id[start:end] == episode)):
            raise ValueError(f"{path}: episode {episode} is not contiguous")
        if not torch.equal(step[start:end], torch.arange(end - start)):
            raise ValueError(f"{path}: episode {episode} has invalid step indices")

    boundaries = value.get("boundaries", {})
    if not isinstance(boundaries, dict):
        raise TypeError(f"{path}: boundaries must be a dictionary")
    cpu_boundaries: dict[str, dict[str, torch.Tensor]] = {}
    for boundary_name, state in boundaries.items():
        if not isinstance(state, dict):
            raise TypeError(f"{path}: boundary {boundary_name!r} must be a dictionary")
        cpu_state: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            valid = isinstance(tensor, torch.Tensor) and tensor.ndim >= 1
            valid = valid and tensor.shape[0] == episodes
            if not valid:
                shape = (
                    tuple(tensor.shape)
                    if isinstance(tensor, torch.Tensor)
                    else type(tensor).__name__
                )
                raise ValueError(
                    f"{path}: boundary {boundary_name}.{key} has invalid shape {shape}"
                )
            cpu_state[key] = tensor.detach().cpu()
        cpu_boundaries[str(boundary_name)] = cpu_state

    return {
        "obs": obs,
        "action": action,
        "phase": phase,
        "episode_id": episode_id,
        "step": step,
        "episode_offsets": offsets,
        "boundaries": cpu_boundaries,
        "meta": value.get("meta", {}),
    }


def assign_options(phase: torch.Tensor) -> torch.Tensor:
    """Map source phases to option ids, splitting approach progress within one episode."""

    option = torch.full_like(phase, -1)
    approach_rows = (phase == PHASE_APPROACH).nonzero(as_tuple=False).squeeze(-1)
    # For an odd number of approach rows, the middle row remains in the first (hover) half.
    hover_count = (approach_rows.numel() + 1) // 2
    option[approach_rows[:hover_count]] = OPTION_HOVER
    option[approach_rows[hover_count:]] = OPTION_DESCEND
    option[phase == PHASE_CLOSE] = OPTION_CLOSE
    option[phase == PHASE_MICRO] = OPTION_MICRO
    option[(phase == PHASE_LIFT) | (phase == PHASE_SETTLE)] = OPTION_LIFT_HOLD
    if bool(torch.any(option < 0)):
        raise RuntimeError("internal error: at least one source phase was not mapped to an option")
    return option


def legal_label_mask(obs: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Return the requested contact/quality gate for each source transition label."""

    keep = torch.ones(phase.shape, dtype=torch.bool)
    micro_rows = phase == PHASE_MICRO
    nonthumb_contact = (obs[:, NONTHUMB_FORCE] >= MICRO_FORCE_MIN).any(dim=-1)
    micro_legal = (
        (obs[:, CLOSE_QUALITY] >= MICRO_CLOSE_MIN)
        & (obs[:, HOLD_QUALITY] >= MICRO_HOLD_MIN)
        & (obs[:, THUMB_FORCE] >= MICRO_FORCE_MIN)
        & nonthumb_contact
    )
    keep[micro_rows] = micro_legal[micro_rows]

    lift_hold_rows = (phase == PHASE_LIFT) | (phase == PHASE_SETTLE)
    transport_quality = torch.minimum(obs[:, WRAP_QUALITY], obs[:, HOLD_QUALITY])
    lift_hold_legal = (
        (obs[:, LATCH] >= 0.5) & (transport_quality >= LIFT_HOLD_QUALITY_MIN)
    )
    keep[lift_hold_rows] = lift_hold_legal[lift_hold_rows]
    return keep


def _boundary_key_set(data: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (boundary_name, key)
        for boundary_name, state in data["boundaries"].items()
        for key in state
    }


def main() -> None:
    args = parse_args()
    sources: list[tuple[Path, dict[str, Any]]] = []
    seen_sha: dict[str, Path] = {}
    for path in args.inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        # Passing the same shard twice renumbers identical trajectories as distinct episodes; the
        # downstream episode-disjoint BC split could then leak a copy across train/val.  Reject it.
        digest = sha256(path)
        if digest in seen_sha:
            raise ValueError(
                f"duplicate input dataset (identical content): {seen_sha[digest]} == {path}"
            )
        seen_sha[digest] = path
        sources.append((path, validate_dataset(path, load_torch(path))))

    obs_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    option_parts: list[torch.Tensor] = []
    episode_parts: list[torch.Tensor] = []
    step_parts: list[torch.Tensor] = []
    source_parts: list[torch.Tensor] = []
    offsets = [0]
    retained_episode_indices: list[torch.Tensor] = []
    active_source_indices: list[int] = []
    source_meta: list[dict[str, Any]] = []
    episode_base = 0

    global_input_phase = {name: 0 for name in SOURCE_PHASE_NAMES}
    global_kept_phase = {name: 0 for name in SOURCE_PHASE_NAMES}
    total_input_episodes = 0
    total_dropped_empty = 0
    total_input_transitions = 0

    for source_id, (path, data) in enumerate(sources):
        phase = data["phase"]
        keep = legal_label_mask(data["obs"], phase)
        input_phase_counts = _phase_counts(phase)
        kept_phase_counts = _phase_counts(phase, keep)
        dropped_phase_counts = {
            name: input_phase_counts[name] - kept_phase_counts[name] for name in SOURCE_PHASE_NAMES
        }
        for name in SOURCE_PHASE_NAMES:
            global_input_phase[name] += input_phase_counts[name]
            global_kept_phase[name] += kept_phase_counts[name]

        source_retained: list[int] = []
        source_output_transitions = 0
        episodes = data["episode_offsets"].numel() - 1
        total_input_episodes += episodes
        total_input_transitions += phase.numel()
        for old_episode in range(episodes):
            start = int(data["episode_offsets"][old_episode])
            end = int(data["episode_offsets"][old_episode + 1])
            local_keep = keep[start:end]
            if not bool(local_keep.any()):
                continue

            source_retained.append(old_episode)
            ep_obs = data["obs"][start:end][local_keep]
            ep_action = data["action"][start:end][local_keep]
            ep_option = assign_options(phase[start:end])[local_keep]
            onehot = F.one_hot(ep_option, num_classes=OPTION_COUNT).to(ep_obs.dtype)
            ep_obs = torch.cat((ep_obs, onehot), dim=-1)
            length = ep_obs.shape[0]

            obs_parts.append(ep_obs)
            action_parts.append(ep_action)
            option_parts.append(ep_option.to(torch.uint8))
            episode_parts.append(torch.full((length,), episode_base, dtype=torch.int64))
            step_parts.append(torch.arange(length, dtype=torch.int32))
            source_parts.append(torch.full((length,), source_id, dtype=torch.int16))
            offsets.append(offsets[-1] + length)
            source_output_transitions += length
            episode_base += 1

        retained = torch.tensor(source_retained, dtype=torch.long)
        retained_episode_indices.append(retained)
        if retained.numel() > 0:
            active_source_indices.append(source_id)
        dropped_empty = episodes - retained.numel()
        total_dropped_empty += dropped_empty
        source_meta.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "input_episodes": episodes,
                "output_episodes": int(retained.numel()),
                "dropped_empty_episodes": int(dropped_empty),
                "input_transitions": int(phase.numel()),
                "output_transitions": source_output_transitions,
                "dropped_transitions": int(phase.numel()) - source_output_transitions,
                "input_by_source_phase": input_phase_counts,
                "kept_by_source_phase": kept_phase_counts,
                "dropped_by_source_phase": dropped_phase_counts,
                "source_meta": data["meta"],
            }
        )

    if not obs_parts:
        raise RuntimeError("all episodes became empty after option-label filtering")

    # A boundary field is retained only when every source contributing output episodes provides a
    # tensor with a compatible trailing shape and dtype.  Its episode axis is sliced to exactly the
    # surviving episodes before sources are concatenated.
    common_boundary_keys = _boundary_key_set(sources[active_source_indices[0]][1])
    for source_id in active_source_indices[1:]:
        common_boundary_keys &= _boundary_key_set(sources[source_id][1])
    retained_boundary_keys: set[tuple[str, str]] = set()
    omitted_boundary_keys: dict[str, str] = {}
    for boundary_name, key in sorted(common_boundary_keys):
        tensors = [
            sources[source_id][1]["boundaries"][boundary_name][key]
            for source_id in active_source_indices
        ]
        first = tensors[0]
        if any(tensor.shape[1:] != first.shape[1:] for tensor in tensors[1:]):
            omitted_boundary_keys[f"{boundary_name}.{key}"] = "incompatible trailing shapes"
        elif any(tensor.dtype != first.dtype for tensor in tensors[1:]):
            omitted_boundary_keys[f"{boundary_name}.{key}"] = "incompatible dtypes"
        else:
            retained_boundary_keys.add((boundary_name, key))

    union_boundary_keys: set[tuple[str, str]] = set()
    for source_id in active_source_indices:
        union_boundary_keys |= _boundary_key_set(sources[source_id][1])
    for boundary_name, key in sorted(union_boundary_keys - common_boundary_keys):
        omitted_boundary_keys[f"{boundary_name}.{key}"] = "missing from at least one input"

    boundaries: dict[str, dict[str, torch.Tensor]] = {}
    for boundary_name, key in sorted(retained_boundary_keys):
        parts = []
        for source_id in active_source_indices:
            tensor = sources[source_id][1]["boundaries"][boundary_name][key]
            parts.append(tensor.index_select(0, retained_episode_indices[source_id]))
        boundaries.setdefault(boundary_name, {})[key] = torch.cat(parts, dim=0)

    obs = torch.cat(obs_parts)
    action = torch.cat(action_parts)
    option = torch.cat(option_parts)
    episode_id = torch.cat(episode_parts)
    step = torch.cat(step_parts)
    source_dataset_id = torch.cat(source_parts)
    episode_offsets = torch.tensor(offsets, dtype=torch.int64)
    kept_transitions = obs.shape[0]
    kept_option_counts = {
        name: _count(option.long(), option_id) for option_id, name in enumerate(OPTION_NAMES)
    }
    dropped_phase = {
        name: global_input_phase[name] - global_kept_phase[name] for name in SOURCE_PHASE_NAMES
    }

    output = {
        "obs": obs,
        "action": action,
        # Downstream loaders continue to consume the phase key, but its semantics are now option id.
        "phase": option,
        "episode_id": episode_id,
        "step": step,
        "source_dataset_id": source_dataset_id,
        "episode_offsets": episode_offsets,
        "boundaries": boundaries,
        "meta": {
            "format_version": 2,
            "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
            "observation_layout": "source_obs115|option_onehot5",
            "source_observation_dim": SOURCE_OBSERVATION_DIM,
            "observation_dim": SOURCE_OBSERVATION_DIM + OPTION_COUNT,
            "phase_semantics": "option_id",
            "phase_names": list(OPTION_NAMES),
            "option_names": list(OPTION_NAMES),
            "source_phase_names": list(SOURCE_PHASE_NAMES),
            "option_mapping": {
                "approach_first_half": OPTION_HOVER,
                "approach_second_half": OPTION_DESCEND,
                "close": OPTION_CLOSE,
                "micro": OPTION_MICRO,
                "lift_and_settle": OPTION_LIFT_HOLD,
            },
            "label_filters": {
                "MICRO": {
                    "close_quality_obs102_min": MICRO_CLOSE_MIN,
                    "hold_quality_obs104_min": MICRO_HOLD_MIN,
                    "thumb_force_strength_obs101_min": MICRO_FORCE_MIN,
                    "any_nonthumb_force_strength_obs97_101_min": MICRO_FORCE_MIN,
                },
                "LIFT_HOLD": {
                    "latch_obs106_min": 0.5,
                    "min_wrap_obs103_hold_obs104_min": LIFT_HOLD_QUALITY_MIN,
                },
            },
            "filter_stats": {
                "input_episodes": total_input_episodes,
                "output_episodes": episode_base,
                "dropped_empty_episodes": total_dropped_empty,
                "input_transitions": total_input_transitions,
                "output_transitions": kept_transitions,
                "dropped_transitions": total_input_transitions - kept_transitions,
                "input_by_source_phase": global_input_phase,
                "kept_by_source_phase": global_kept_phase,
                "dropped_by_source_phase": dropped_phase,
                "output_by_option": kept_option_counts,
            },
            "retained_boundary_fields": [
                f"{boundary_name}.{key}" for boundary_name, key in sorted(retained_boundary_keys)
            ],
            "omitted_boundary_fields": omitted_boundary_keys,
            "sources": source_meta,
        },
    }

    # Final invariants guard against accidentally emitting option/episode tensors out of sync.
    if obs.shape != (kept_transitions, SOURCE_OBSERVATION_DIM + OPTION_COUNT):
        raise RuntimeError(f"internal error: output observation shape is {tuple(obs.shape)}")
    if action.shape != (kept_transitions, ACTION_DIM):
        raise RuntimeError(f"internal error: output action shape is {tuple(action.shape)}")
    if episode_offsets.shape != (episode_base + 1,) or int(episode_offsets[-1]) != kept_transitions:
        raise RuntimeError("internal error: output episode offsets are inconsistent")
    if not torch.equal(obs[:, -OPTION_COUNT:].argmax(dim=-1).to(torch.uint8), option):
        raise RuntimeError("internal error: option one-hot suffix and phase/option ids disagree")
    if not bool(torch.all(obs[:, -OPTION_COUNT:].sum(dim=-1) == 1.0)):
        raise RuntimeError("internal error: malformed option one-hot suffix")
    for episode in range(episode_base):
        start, end = int(episode_offsets[episode]), int(episode_offsets[episode + 1])
        if not bool(torch.all(episode_id[start:end] == episode)):
            raise RuntimeError(f"internal error: output episode {episode} is not contiguous")
        if not torch.equal(step[start:end].long(), torch.arange(end - start)):
            raise RuntimeError(
                f"internal error: output episode {episode} step indices are not rebuilt"
            )
    for boundary_name, state in boundaries.items():
        for key, tensor in state.items():
            if tensor.shape[0] != episode_base:
                raise RuntimeError(
                    f"internal error: boundary {boundary_name}.{key} has {tensor.shape[0]} episodes"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"wrote {args.output.resolve()}: {episode_base}/{total_input_episodes} episodes, "
        f"{kept_transitions}/{total_input_transitions} transitions, obs={tuple(obs.shape)}"
    )
    print(f"output options: {kept_option_counts}")
    print(f"dropped source phases: {dropped_phase}")
    print(
        f"boundaries: retained {len(retained_boundary_keys)} fields, "
        f"omitted {len(omitted_boundary_keys)} fields"
    )


if __name__ == "__main__":
    main()
