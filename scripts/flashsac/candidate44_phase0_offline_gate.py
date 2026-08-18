#!/usr/bin/env python3
"""Simulation-free release gate for Candidate44's phase-0 actor correction."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actor_rehearsal import (  # noqa: E402
    PICK_TOOL_APPROACH_CORRECTION_VALIDATION_CONTRACTS,
    load_actor_rehearsal,
)
from train import audit_pick_tool_demonstrations  # noqa: E402
from agent_bridge import (  # noqa: E402
    FlashSACTorchBridge,
    build_agent_config,
)


OBSERVATION_DIM = 115
ACTION_DIM = 21
C2_ACTOR_SHA256 = "ccf3a22ae031b6bf4037a5226bfebeeb8699fe4c6ab485e00cc06ff077533e38"
C2_CRITIC_SHA256 = "cd8db7845f55464ef45d403d5a3ccb3cf61e1ed2c45cae1d33b9c6dd4eb1a8ce"
C2_REPLAY_SHA256 = "0817f3ab8dfd28b5be06d192d8cc3a8d41a467d68f72c24d115d76d8ac04c07f"
C2_ACTOR_REHEARSAL_SHA256 = (
    "fadf605c91ebd7233d059cf01709fb66833f4e51269350d27dccdb793d13f508"
)
C2_TASK_CONTRACT_SHA256 = (
    "d47abe6f60627fa05937026e56fb4e44e05644a5a370e59c4e64eec57d142f7a"
)
C2_CHECKPOINT_RELATIVE = Path(
    "logs/flashsac/pick_tool/12_c2_closestart_markov_burnin_s250_r3/"
    "checkpoint_final"
)
CANDIDATE_OUTPUT_RELATIVE = Path(
    "logs/flashsac/pick_tool/56_c44_phase0_correction_s359"
)
CANONICAL_SOURCE_RELATIVE = Path(
    "logs/flashsac/pick_tool/demos/source/"
    "pick_tool_ik_dagger_full_iter3_1685ep.pt"
)
SOURCE_SHA256 = "e4aaf0eda6a33db4a2ed04bc4d7609639da9b0471408808a5dae1b67760ce57f"
TRAIN_ROWS = 220181
TRAIN_EPISODES = 734
VALIDATION_ROWS = 54900
VALIDATION_EPISODES = 183
ELIGIBLE_ROWS = 275081
ELIGIBLE_EPISODES = 917
DATA_MANIFEST_KIND = "pick_tool_candidate44_phase0_correction_data_manifest_v1"
SPLIT_METHOD = "sha256_rank_source_episode_80_20_v1"
SPLIT_SALT = "pick_tool_candidate44_phase0_correction_split_20260723_v1"
CHECKPOINT_TENSOR_FILENAMES = frozenset(
    {
        "actor.pt",
        "actor_rehearsal.pt",
        "agent_state.pt",
        "critic.pt",
        "replay_buffer.pt",
        "reward_normalizer.pt",
        "target_critic.pt",
        "temperature.pt",
        "torch_bridge_state.pt",
    }
)
TRANSITION_DEMOS = (
    (
        "logs/flashsac/pick_tool/demos/hierarchy_transition_smoke_s163.pt",
        "031f2d60c30628142308c5c5cc0b1ccb663d38386f2423761319c83af020c24d",
    ),
    (
        "logs/flashsac/pick_tool/demos/hierarchy_transition_s164.pt",
        "316abdc0da18a038e4f3fe1915b1a5dec36003dfde5e142e47d3d73d021738c0",
    ),
    (
        "logs/flashsac/pick_tool/demos/hierarchy_transition_s165.pt",
        "4d9bd2d1d721aaa9d792c22f2a25f59ddb5997b1bc64ce62bc94a49ddcfd3116",
    ),
)
GROUP_SLICES = {
    "arm": slice(0, 7),
    "token": slice(7, 16),
    "residual": slice(16, 21),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")


def _tensor_tree_summary(value: Any) -> dict[str, Any]:
    """Audit every floating/complex tensor without materializing a file-sized mask."""

    tensor_count = 0
    tensor_elements = 0
    floating_tensor_count = 0
    nonfinite_paths: list[str] = []
    seen_containers: set[int] = set()

    def visit(candidate: Any, path: str) -> None:
        nonlocal tensor_count, tensor_elements, floating_tensor_count
        if isinstance(candidate, torch.Tensor):
            tensor_count += 1
            tensor_elements += candidate.numel()
            if candidate.is_floating_point() or candidate.is_complex():
                floating_tensor_count += 1
                flat = candidate.detach()
                if flat.device.type != "cpu":
                    flat = flat.cpu()
                flat = flat.contiguous().view(-1)
                chunk_elements = 4 * 1024 * 1024
                finite = True
                for start in range(0, flat.numel(), chunk_elements):
                    if not bool(torch.isfinite(flat[start : start + chunk_elements]).all()):
                        finite = False
                        break
                if not finite and len(nonfinite_paths) < 16:
                    nonfinite_paths.append(path)
            return
        if isinstance(candidate, Mapping):
            identity = id(candidate)
            if identity in seen_containers:
                return
            seen_containers.add(identity)
            for key, item in candidate.items():
                visit(item, f"{path}[{key!r}]")
            return
        if isinstance(candidate, (list, tuple)):
            identity = id(candidate)
            if identity in seen_containers:
                return
            seen_containers.add(identity)
            for index, item in enumerate(candidate):
                visit(item, f"{path}[{index}]")

    visit(value, "$payload")
    return {
        "tensor_count": tensor_count,
        "tensor_elements": tensor_elements,
        "floating_or_complex_tensor_count": floating_tensor_count,
        "all_finite": not nonfinite_paths,
        "nonfinite_paths": nonfinite_paths,
    }


def audit_checkpoint_tensor_files(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise NotADirectoryError(
            f"candidate checkpoint must be a real directory: {checkpoint}"
        )
    if (checkpoint / ".incomplete_checkpoint.json").exists():
        raise ValueError("candidate checkpoint still has an incomplete marker")
    actual = {path.name for path in checkpoint.glob("*.pt")}
    if actual != CHECKPOINT_TENSOR_FILENAMES:
        raise ValueError(
            "candidate checkpoint tensor-file schema differs from the sealed full-task "
            f"schema: missing={sorted(CHECKPOINT_TENSOR_FILENAMES - actual)}, "
            f"unexpected={sorted(actual - CHECKPOINT_TENSOR_FILENAMES)}"
        )
    files: dict[str, Any] = {}
    all_finite = True
    for filename in sorted(actual):
        path = checkpoint / filename
        _require_regular_file(path, label=f"checkpoint tensor file {filename}")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        summary = _tensor_tree_summary(payload)
        summary.update(
            {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        files[filename] = summary
        all_finite = all_finite and bool(summary["all_finite"])
        del payload
        gc.collect()
    return {
        "all_finite": all_finite,
        "tensor_file_schema_exact": True,
        "files": files,
    }


def audit_data_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
    train_dataset: Path,
    validation_dataset: Path,
) -> dict[str, Any]:
    _require_regular_file(manifest_path, label="Candidate44 data manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("Candidate44 data manifest root must be an object")
    source = manifest.get("source")
    split = manifest.get("split")
    train = manifest.get("train")
    validation = manifest.get("validation")
    if not all(isinstance(value, Mapping) for value in (source, split, train, validation)):
        raise TypeError("Candidate44 data manifest is missing source/split/train/validation")
    canonical_source = (repository_root / CANONICAL_SOURCE_RELATIVE).resolve()
    _require_regular_file(canonical_source, label="canonical DAgger source")
    _require_regular_file(train_dataset, label="Candidate44 train projection")
    _require_regular_file(validation_dataset, label="Candidate44 validation projection")
    actual_source_sha256 = sha256_file(canonical_source)
    actual_train_sha256 = sha256_file(train_dataset)
    actual_validation_sha256 = sha256_file(validation_dataset)

    checks = {
        "kind_exact": manifest.get("kind") == DATA_MANIFEST_KIND,
        "status_complete": manifest.get("status") == "complete",
        "source_path_exact": Path(str(source.get("canonical_path"))).resolve()
        == canonical_source,
        "source_sha256_exact": source.get("sha256")
        == actual_source_sha256
        == SOURCE_SHA256,
        "source_counts_exact": source.get("rows") == 797450
        and source.get("episodes") == 1685,
        "source_bytes_exact": source.get("bytes") == canonical_source.stat().st_size,
        "split_contract_exact": split.get("method") == SPLIT_METHOD
        and split.get("salt") == SPLIT_SALT
        and split.get("eligible_rows") == ELIGIBLE_ROWS
        and split.get("eligible_episodes") == ELIGIBLE_EPISODES,
        "train_path_exact": Path(str(train.get("path"))).resolve() == train_dataset,
        "train_sha256_exact": train.get("sha256") == actual_train_sha256,
        "train_counts_exact": train.get("rows") == TRAIN_ROWS
        and train.get("episodes") == TRAIN_EPISODES,
        "train_bytes_exact": train.get("bytes") == train_dataset.stat().st_size,
        "validation_path_exact": Path(str(validation.get("path"))).resolve()
        == validation_dataset,
        "validation_sha256_exact": validation.get("sha256")
        == actual_validation_sha256,
        "validation_counts_exact": validation.get("rows") == VALIDATION_ROWS
        and validation.get("episodes") == VALIDATION_EPISODES,
        "validation_bytes_exact": validation.get("bytes")
        == validation_dataset.stat().st_size,
        "episode_overlap_zero": manifest.get(
            "train_validation_source_episode_overlap"
        )
        == 0,
        "forbidden_fields_absent": manifest.get(
            "forbidden_transition_or_outcome_fields_present"
        )
        is False,
    }
    return {
        "all_pass": all(checks.values()),
        "checks": checks,
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": actual_source_sha256,
        "train_sha256": actual_train_sha256,
        "validation_sha256": actual_validation_sha256,
    }


def _semantic_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (
            isinstance(first, torch.Tensor)
            and isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and tuple(first.shape) == tuple(second.shape)
            and torch.equal(first.cpu(), second.cpu())
        )
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return (
            isinstance(first, Mapping)
            and isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_semantic_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == len(second)
            and all(_semantic_equal(a, b) for a, b in zip(first, second, strict=True))
        )
    return type(first) is type(second) and first == second


def action_error_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != ACTION_DIM:
        raise ValueError("predicted and target actions must both have shape [N,21]")
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("action panel contains NaN or infinity")
    squared_error = (predicted - target).square()
    metrics = {"action_rmse": float(squared_error.mean().sqrt())}
    for name, selection in GROUP_SLICES.items():
        metrics[f"{name}_rmse"] = float(squared_error[:, selection].mean().sqrt())
        metrics[f"predicted_{name}_rms"] = float(
            predicted[:, selection].square().mean().sqrt()
        )
        metrics[f"target_{name}_rms"] = float(target[:, selection].square().mean().sqrt())
    return metrics


def build_panel_report(
    baseline_action: torch.Tensor,
    candidate_action: torch.Tensor,
    target_action: torch.Tensor,
) -> dict[str, Any]:
    baseline = action_error_metrics(baseline_action, target_action)
    candidate = action_error_metrics(candidate_action, target_action)
    drift = action_error_metrics(candidate_action, baseline_action)
    relative: dict[str, float] = {}
    delta: dict[str, float] = {}
    for name in ("action", *GROUP_SLICES):
        key = f"{name}_rmse"
        denominator = baseline[key]
        relative[key] = candidate[key] / denominator if denominator > 0.0 else math.inf
        delta[key] = candidate[key] - baseline[key]
    return {
        "rows": int(target_action.shape[0]),
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline_prediction": drift,
        "candidate_to_baseline_rmse_ratio": relative,
        "candidate_minus_baseline_target_rmse": delta,
    }


def _make_actor(checkpoint: Path) -> FlashSACTorchBridge:
    observation_space = gym.spaces.Box(
        low=-math.inf, high=math.inf, shape=(OBSERVATION_DIM,), dtype="float32"
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype="float32"
    )
    cfg = build_agent_config(
        seed=44,
        device_type="cpu",
        buffer_device_type="cpu",
        buffer_max_length=64,
        buffer_min_length=2,
        sample_batch_size=2,
        normalize_reward=True,
        normalized_G_max=5.0,
        n_step=3,
        actor_num_blocks=2,
        actor_hidden_dim=128,
        critic_num_blocks=2,
        critic_hidden_dim=256,
        critic_num_bins=101,
        use_compile=False,
        use_amp=False,
        load_optimizer=False,
        load_reward_normalizer=False,
    )
    actor = FlashSACTorchBridge(
        observation_space,
        action_space,
        {"actor_observation_size": (OBSERVATION_DIM,), "asymmetric_obs": False},
        cfg,
        restore_rng_state_on_load=False,
    )
    actor.load_actor(str(checkpoint))
    return actor


@torch.no_grad()
def predict_actions(
    actor: FlashSACTorchBridge,
    observation: torch.Tensor,
    *,
    batch_size: int = 8192,
) -> torch.Tensor:
    if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("offline actor observations must have shape [N,115]")
    actions: list[torch.Tensor] = []
    for start in range(0, observation.shape[0], batch_size):
        stop = min(start + batch_size, observation.shape[0])
        mean = actor.deterministic_actor_mean(observation[start:stop].to(dtype=torch.float32))
        actions.append(torch.tanh(mean).cpu())
    result = torch.cat(actions, dim=0)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("offline actor prediction contains NaN or infinity")
    return result


def _load_transition_panels(repository_root: Path) -> dict[int, dict[str, torch.Tensor]]:
    observations: dict[int, list[torch.Tensor]] = {1: [], 3: []}
    actions: dict[int, list[torch.Tensor]] = {1: [], 3: []}
    for relative, expected_sha256 in TRANSITION_DEMOS:
        path = (repository_root / relative).resolve()
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"transition demonstration changed bytes: {path}")
        audit_pick_tool_demonstrations(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        phase = payload.get("phase")
        if not isinstance(phase, torch.Tensor) or phase.shape != (
            payload["observation"].shape[0],
        ):
            raise ValueError(f"transition demonstration has invalid phase labels: {path}")
        for phase_value in observations:
            selected = phase == phase_value
            if bool(selected.any()):
                observations[phase_value].append(payload["observation"][selected].clone())
                actions[phase_value].append(payload["action"][selected].clone())
    panels: dict[int, dict[str, torch.Tensor]] = {}
    for phase_value in observations:
        if not observations[phase_value]:
            raise ValueError(f"strict transition demonstrations have no phase={phase_value}")
        panels[phase_value] = {
            "observation": torch.cat(observations[phase_value], dim=0),
            "action": torch.cat(actions[phase_value], dim=0),
        }
    return panels


def _normalized_network_state(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("network_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("actor checkpoint has no network_state_dict")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("actor network state must map names to tensors")
        normalized[name.removeprefix("_orig_mod.")] = value
    if len(normalized) != len(state):
        raise ValueError("actor network state collides after compiled-prefix removal")
    return normalized


def _actor_parameter_names() -> list[str]:
    # Import after agent_bridge installed its types-only JAX fallback.
    from flash_rl.agents.flashSAC.network import FlashSACActor

    network = FlashSACActor(
        num_blocks=2,
        input_dim=OBSERVATION_DIM,
        hidden_dim=128,
        action_dim=ACTION_DIM,
    )
    return [name for name, _parameter in network.named_parameters()]


def _optimizer_state_by_parameter_name(payload: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, Mapping):
        raise TypeError("actor checkpoint has no optimizer_state_dict")
    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(state, Mapping):
        raise ValueError("actor optimizer must contain exactly one parameter group")
    parameter_ids = groups[0].get("params")
    names = _actor_parameter_names()
    if not isinstance(parameter_ids, list) or len(parameter_ids) != len(names):
        raise ValueError("actor optimizer parameter order differs from pinned network")
    return {
        name: state[parameter_id]
        for name, parameter_id in zip(names, parameter_ids, strict=True)
        if parameter_id in state
    }


def checkpoint_integrity(
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    *,
    candidate_metrics_path: Path,
    train_dataset: Path,
    validation_dataset: Path,
    data_manifest_path: Path,
) -> dict[str, Any]:
    repository_root = HERE.parents[1]
    expected_baseline_checkpoint = (repository_root / C2_CHECKPOINT_RELATIVE).resolve()
    expected_candidate_output = (repository_root / CANDIDATE_OUTPUT_RELATIVE).resolve()
    expected_candidate_checkpoint = expected_candidate_output / "checkpoint_final"
    expected_candidate_metrics = expected_candidate_output / "metrics.json"
    if not baseline_checkpoint.is_dir() or baseline_checkpoint.is_symlink():
        raise NotADirectoryError(f"C2 checkpoint is not a real directory: {baseline_checkpoint}")
    if not candidate_checkpoint.is_dir() or candidate_checkpoint.is_symlink():
        raise NotADirectoryError(
            f"Candidate44 checkpoint is not a real directory: {candidate_checkpoint}"
        )
    _require_regular_file(candidate_metrics_path, label="Candidate44 metrics")
    baseline_actor_path = baseline_checkpoint / "actor.pt"
    candidate_actor_path = candidate_checkpoint / "actor.pt"
    for path, label in (
        (baseline_actor_path, "C2 actor"),
        (baseline_checkpoint / "critic.pt", "C2 critic"),
        (baseline_checkpoint / "replay_buffer.pt", "C2 replay"),
        (baseline_checkpoint / "actor_rehearsal.pt", "C2 actor rehearsal"),
        (baseline_checkpoint / "task_contract.json", "C2 task contract"),
        (candidate_actor_path, "Candidate44 actor"),
        (candidate_checkpoint / "task_contract.json", "Candidate44 task contract"),
    ):
        _require_regular_file(path, label=label)
    baseline_actor_sha256 = sha256_file(baseline_actor_path)
    candidate_actor_sha256 = sha256_file(candidate_actor_path)
    baseline_actor = torch.load(baseline_actor_path, map_location="cpu", weights_only=True)
    candidate_actor = torch.load(candidate_actor_path, map_location="cpu", weights_only=True)
    baseline_state = _normalized_network_state(baseline_actor)
    candidate_state = _normalized_network_state(candidate_actor)
    if set(baseline_state) != set(candidate_state):
        raise ValueError("candidate actor network schema differs from C2")
    std_names = ("predictor.std_bias", "predictor.std_w.w.weight")
    std_tensor_exact = all(
        torch.equal(baseline_state[name], candidate_state[name]) for name in std_names
    )
    baseline_optimizer = _optimizer_state_by_parameter_name(baseline_actor)
    candidate_optimizer = _optimizer_state_by_parameter_name(candidate_actor)
    std_optimizer_exact = all(
        name in baseline_optimizer
        and name in candidate_optimizer
        and _semantic_equal(baseline_optimizer[name], candidate_optimizer[name])
        for name in std_names
    )
    scheduler = candidate_actor.get("scheduler_state_dict")
    scheduler_last_epoch = (
        scheduler.get("last_epoch") if isinstance(scheduler, Mapping) else None
    )
    baseline_temperature = torch.load(
        baseline_checkpoint / "temperature.pt", map_location="cpu", weights_only=True
    )
    candidate_temperature = torch.load(
        candidate_checkpoint / "temperature.pt", map_location="cpu", weights_only=True
    )
    task_contract_exact = (
        sha256_file(candidate_checkpoint / "task_contract.json")
        == C2_TASK_CONTRACT_SHA256
        and (candidate_checkpoint / "task_contract.json").read_bytes()
        == (baseline_checkpoint / "task_contract.json").read_bytes()
    )
    metrics = json.loads(candidate_metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, Mapping):
        raise TypeError("Candidate44 metrics root must be an object")
    actor_sources = metrics.get("actor_demo_sources")
    if not isinstance(actor_sources, list):
        raise TypeError("candidate metrics has no actor_demo_sources list")
    if not all(isinstance(entry, Mapping) for entry in actor_sources):
        raise TypeError("candidate actor_demo_sources entries must be objects")
    demo_sources = metrics.get("demo_sources")
    if not isinstance(demo_sources, list) or not all(
        isinstance(entry, Mapping) for entry in demo_sources
    ):
        raise TypeError("candidate metrics has invalid demo_sources")
    train_sha256 = sha256_file(train_dataset)
    validation_sha256 = sha256_file(validation_dataset)
    actor_source_hashes = [entry.get("sha256") for entry in actor_sources]
    actor_source_roles = [entry.get("role") for entry in actor_sources]
    demo_source_hashes = [entry.get("sha256") for entry in demo_sources]
    expected_demo_hashes = [expected for _path, expected in TRANSITION_DEMOS]
    checkpoint_tensor_audit = audit_checkpoint_tensor_files(candidate_checkpoint)
    data_manifest_audit = audit_data_manifest(
        data_manifest_path,
        repository_root=repository_root,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    candidate_task_contract = json.loads(
        (candidate_checkpoint / "task_contract.json").read_text(encoding="utf-8")
    )

    def metric_path(name: str) -> Path | None:
        value = metrics.get(name)
        return Path(value).resolve() if isinstance(value, str) else None

    checks = {
        "baseline_checkpoint_path_exact": baseline_checkpoint
        == expected_baseline_checkpoint,
        "candidate_checkpoint_path_exact": candidate_checkpoint
        == expected_candidate_checkpoint,
        "candidate_metrics_path_exact": candidate_metrics_path
        == expected_candidate_metrics,
        "baseline_actor_sha256_exact": baseline_actor_sha256 == C2_ACTOR_SHA256,
        "baseline_critic_sha256_exact": sha256_file(baseline_checkpoint / "critic.pt")
        == C2_CRITIC_SHA256,
        "baseline_replay_sha256_exact": sha256_file(
            baseline_checkpoint / "replay_buffer.pt"
        )
        == C2_REPLAY_SHA256,
        "baseline_actor_rehearsal_sha256_exact": sha256_file(
            baseline_checkpoint / "actor_rehearsal.pt"
        )
        == C2_ACTOR_REHEARSAL_SHA256,
        "candidate_actor_sha256_differs": candidate_actor_sha256 != baseline_actor_sha256,
        "task_contract_exact_c2": task_contract_exact,
        "all_checkpoint_tensors_finite": checkpoint_tensor_audit["all_finite"],
        "checkpoint_tensor_file_schema_exact": checkpoint_tensor_audit[
            "tensor_file_schema_exact"
        ],
        "std_head_network_tensors_exact_c2": std_tensor_exact,
        "std_head_existing_optimizer_states_exact_c2": std_optimizer_exact,
        "actor_scheduler_last_epoch_exact": scheduler_last_epoch == 3864,
        "temperature_semantically_exact_c2": _semantic_equal(
            baseline_temperature, candidate_temperature
        ),
        "sac_actor_updates_exact": metrics.get("actor_updates") == 0,
        "sac_actor_alias_updates_exact": metrics.get("sac_actor_updates") == 0,
        "demo_bc_updates_exact": metrics.get("demo_bc_updates") == 250,
        "demo_bc_only_target_exact": metrics.get("demo_bc_only_updates_target")
        == 250,
        "demo_bc_only_slots_exact": metrics.get("demo_bc_only_deferred_slots")
        == 250,
        "critic_updates_exact": metrics.get("gradient_updates") == 500,
        "interaction_steps_exact": metrics.get("interaction_step") == 250,
        "requested_interaction_steps_exact": metrics.get(
            "requested_interaction_steps"
        )
        == 250,
        "environment_steps_exact": metrics.get("environment_steps") == 256000,
        "seed_exact": metrics.get("seed") == 359,
        "full_task_non_smoke": metrics.get("task_mode") == "full_task"
        and metrics.get("smoke") is False,
        "candidate_metrics_complete": metrics.get("status") == "complete",
        "metrics_candidate_checkpoint_exact": metric_path("checkpoint")
        == candidate_checkpoint,
        "initial_checkpoint_exact_c2": metric_path("initial_checkpoint")
        == baseline_checkpoint,
        "initial_actor_checkpoint_absent": metrics.get("initial_actor_checkpoint")
        is None,
        "replay_resumed": metrics.get("resumed_replay") is True,
        "replay_saved": metrics.get("save_replay") is True
        and (candidate_checkpoint / "replay_buffer.pt").is_file()
        and not (candidate_checkpoint / "replay_buffer.pt").is_symlink(),
        "actor_rehearsal_not_resumed": metrics.get("resumed_actor_demo") is False,
        "actor_source_order_exact": actor_source_hashes
        == [*expected_demo_hashes, train_sha256],
        "actor_source_roles_exact": actor_source_roles
        == ["critic_transition_projection"] * 3 + ["actor_only"],
        "train_correction_source_present": actor_source_hashes[-1:] == [train_sha256],
        "validation_source_absent": validation_sha256 not in actor_source_hashes,
        "transition_demo_order_exact": demo_source_hashes == expected_demo_hashes,
        "transition_demo_fraction_exact": metrics.get("demo_fraction") == 0.25,
        "transition_demo_rows_per_batch_exact": metrics.get("demo_rows_per_batch")
        == 512,
        "actor_rehearsal_rows_exact": metrics.get("actor_rehearsal_transitions")
        == TRAIN_ROWS + 34810,
        "actor_rehearsal_batch_exact": metrics.get("actor_rehearsal_batch")
        == 2048,
        "phase0_only_rehearsal": metrics.get("actor_rehearsal_stratum_weights")
        == {"0": 1.0, "1": 0.0, "3": 0.0},
        "demo_bc_weight_exact": metrics.get("demo_bc_weight") == 1.0,
        "demo_bc_group_weights_exact": metrics.get("demo_bc_group_weights")
        == {"arm": 2.0, "token": 1.0, "residual": 1.0},
        "demo_bc_target_std_exact": metrics.get("demo_bc_target_std") == 0.15,
        "action_only_std_weight": metrics.get("demo_bc_std_weight", 0.0) == 0.0,
        "demo_bc_phases_exact": metrics.get("demo_bc_phases") == [0],
        "actor_update_period_exact": metrics.get("actor_update_period") == 2,
        "actor_lr_scale_exact": metrics.get("actor_lr_scale") == 0.01,
        "critic_burnin_exact": metrics.get("critic_burnin_updates") == 0,
        "lr_schedule_exact": metrics.get("lr_decay_updates") == 50000
        and metrics.get("lr_warmup_updates") == 1000,
        "vector_width_exact": metrics.get("num_envs") == 1024,
        "batch_size_exact": metrics.get("batch_size") == 2048,
        "updates_per_vector_step_exact": metrics.get("updates_per_vector_step")
        == 2.0,
        "buffer_capacity_exact": metrics.get("buffer_capacity") == 1000000,
        "warmup_exact": metrics.get("warmup_transitions") == 10000,
        "network_shape_exact": metrics.get("n_step") == 3
        and metrics.get("actor_blocks") == 2
        and metrics.get("actor_hidden") == 128
        and metrics.get("critic_blocks") == 2
        and metrics.get("critic_hidden") == 256
        and metrics.get("critic_bins") == 101,
        "ordinary_reset_training_exact": metrics.get("curriculum_probability")
        == 0.0
        and metrics.get("curriculum_dataset") is None
        and metrics.get("curriculum_boundary") is None,
        "noise_scales_exact": metrics.get("unlatched_arm_noise_scale") == 1.0
        and metrics.get("unlatched_hand_noise_scale") == 1.0
        and metrics.get("latched_arm_noise_scale") == 1.0
        and metrics.get("latched_hand_noise_scale") == 0.2,
        "finite_validation_enabled": metrics.get("validate_finite") is True,
        "compile_and_amp_enabled": metrics.get("compile_enabled") is True
        and metrics.get("amp_enabled") is True,
        "rng_restore_disabled": metrics.get("restore_checkpoint_rng") is False,
        "flashsac_fork_exact": metrics.get("flashsac_fork_commit")
        == "5ecf331fa11cd457dd39018b3d68af571b257666",
        "metrics_task_contract_exact": metrics.get("checkpoint_task_contract")
        == candidate_task_contract,
        "resumed_task_contract_exact": metrics.get("resumed_task_contract")
        == candidate_task_contract,
        "sealed_data_manifest_exact": data_manifest_audit["all_pass"],
        "manifest_train_hash_matches_metrics": data_manifest_audit["train_sha256"]
        == train_sha256,
        "manifest_validation_hash_matches_panel": data_manifest_audit[
            "validation_sha256"
        ]
        == validation_sha256,
    }
    return {
        "all_pass": all(checks.values()),
        "checks": checks,
        "baseline_actor_sha256": baseline_actor_sha256,
        "candidate_actor_sha256": candidate_actor_sha256,
        "baseline_task_contract_sha256": sha256_file(
            baseline_checkpoint / "task_contract.json"
        ),
        "candidate_task_contract_sha256": sha256_file(
            candidate_checkpoint / "task_contract.json"
        ),
        "candidate_replay_sha256": checkpoint_tensor_audit["files"][
            "replay_buffer.pt"
        ]["sha256"],
        "actor_scheduler_last_epoch": scheduler_last_epoch,
        "train_dataset_sha256": train_sha256,
        "validation_dataset_sha256": validation_sha256,
        "data_manifest": data_manifest_audit,
        "checkpoint_tensor_audit": checkpoint_tensor_audit,
    }


def apply_registered_gates(
    phase0: Mapping[str, Any],
    phase1: Mapping[str, Any],
    phase3: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    phase0_ratios = phase0["candidate_to_baseline_rmse_ratio"]
    phase1_delta = phase1["candidate_minus_baseline_target_rmse"]
    phase3_delta = phase3["candidate_minus_baseline_target_rmse"]
    checks = {
        "integrity": bool(integrity["all_pass"]),
        "phase0_action_rmse_ratio_le_0_85": phase0_ratios["action_rmse"] <= 0.85,
        "phase0_arm_rmse_ratio_le_0_85": phase0_ratios["arm_rmse"] <= 0.85,
        "phase0_token_rmse_ratio_le_0_95": phase0_ratios["token_rmse"] <= 0.95,
        "phase0_residual_rmse_ratio_le_0_85": phase0_ratios["residual_rmse"] <= 0.85,
        "phase1_action_rmse_delta_le_0_03": phase1_delta["action_rmse"] <= 0.03,
        "phase3_action_rmse_delta_le_0_03": phase3_delta["action_rmse"] <= 0.03,
    }
    return {
        "all_pass": all(checks.values()),
        "checks": checks,
        "decision": (
            "release_to_paired_physical_gate"
            if all(checks.values())
            else "reject_before_candidate_physical_evaluation"
        ),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--baseline_checkpoint",
        type=Path,
        default=Path(
            "logs/flashsac/pick_tool/12_c2_closestart_markov_burnin_s250_r3/"
            "checkpoint_final"
        ),
    )
    parser.add_argument("--candidate_checkpoint", type=Path, required=True)
    parser.add_argument("--candidate_metrics", type=Path, required=True)
    parser.add_argument(
        "--train_dataset",
        type=Path,
        default=Path("logs/flashsac/pick_tool/demos/c44_dagger_phase0_train.pt"),
    )
    parser.add_argument(
        "--validation_dataset",
        type=Path,
        default=Path(
            "logs/flashsac/pick_tool/demos/c44_dagger_phase0_validation.pt"
        ),
    )
    parser.add_argument(
        "--data_manifest",
        type=Path,
        default=Path(
            "logs/flashsac/pick_tool/demos/c44_dagger_phase0_manifest.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository_root = HERE.parents[1]
    baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    candidate_checkpoint = args.candidate_checkpoint.expanduser().resolve()
    candidate_metrics = args.candidate_metrics.expanduser().resolve()
    train_dataset = args.train_dataset.expanduser().resolve()
    validation_dataset = args.validation_dataset.expanduser().resolve()
    data_manifest = args.data_manifest.expanduser().resolve()

    validation_batch, validation_phase, validation_audit = load_actor_rehearsal(
        validation_dataset,
        device="cpu",
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        allowed_contracts=PICK_TOOL_APPROACH_CORRECTION_VALIDATION_CONTRACTS,
    )
    if validation_phase is None or not bool((validation_phase == 0).all()):
        raise ValueError("validation correction dataset is not exclusively phase 0")
    transition_panels = _load_transition_panels(repository_root)
    baseline_actor = _make_actor(baseline_checkpoint)
    candidate_actor = _make_actor(candidate_checkpoint)
    phase0 = build_panel_report(
        predict_actions(baseline_actor, validation_batch["observation"]),
        predict_actions(candidate_actor, validation_batch["observation"]),
        validation_batch["action"],
    )
    transition_reports: dict[str, Any] = {}
    for phase_value, panel in transition_panels.items():
        transition_reports[str(phase_value)] = build_panel_report(
            predict_actions(baseline_actor, panel["observation"]),
            predict_actions(candidate_actor, panel["observation"]),
            panel["action"],
        )
    integrity = checkpoint_integrity(
        baseline_checkpoint,
        candidate_checkpoint,
        candidate_metrics_path=candidate_metrics,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        data_manifest_path=data_manifest,
    )
    gate = apply_registered_gates(
        phase0,
        transition_reports["1"],
        transition_reports["3"],
        integrity,
    )
    result = {
        "kind": "pick_tool_candidate44_phase0_correction_offline_gate_v1",
        "status": "complete",
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_metrics": str(candidate_metrics),
        "data_manifest": str(data_manifest),
        "validation_source": validation_audit,
        "phase0_validation": phase0,
        "strict_success_transition_retention": transition_reports,
        "integrity": integrity,
        "registered_gate": gate,
    }
    _atomic_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
