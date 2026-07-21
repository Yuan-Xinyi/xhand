#!/usr/bin/env python3
"""Bootstrap the PickTool FlashSAC actor from successful demonstration actions.

This is deliberately separate from upstream ``actor_bc_alpha``.  The upstream
term operates on whichever actions happen to be in the online replay buffer and
is Q-scaled; it is not a permanent demonstration objective.  Here the target is
always the recorded 21-dimensional teacher action loaded from an explicit demo
file.

FlashSAC's deterministic action is ``tanh(mean)``.  Supervision therefore uses
the corresponding pre-tanh target
``atanh(clamp(demo_action, -1 + eps, 1 - eps))``.  This avoids the vanishing
gradient obtained by applying ordinary MSE after a saturated tanh.  Deterministic
demos do not identify a behavior variance, so the stochastic standard-deviation
head is calibrated to an explicit 0.15 prior rather than falsely inferred from
the demonstrations.  SAC remains free to update that distribution online.

Full trajectories are sampled with equal probability mass per recorded phase.
Each epoch oversamples every minority phase to the largest phase count, so the
rare close/grasp bridge cannot be drowned out by approach/search rows.  A file
without phase annotations is represented by phase ``-1`` and naturally reduces
to ordinary shuffled sampling of one stratum.  Validation remains on untouched
episodes, while checkpoint selection uses a phase-macro action RMSE.

The exported directory is a complete :class:`FlashSACTorchBridge` checkpoint.
Only actor weights and BatchNorm statistics come from BC.  Critic, target
critic, temperature, optimizers, schedulers, reward normalizer, update counter,
and exploration state are freshly initialized before export, so online RL does
not inherit BC optimizer moments or a meaningless random-critic history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import torch.nn.functional as F

from agent_bridge import ActionNoiseGroup, FlashSACTorchBridge, build_agent_config

# Importing agent_bridge installs the pinned upstream path and its Torch-only
# annotation fallback before these imports are resolved.
from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.agents.utils.network import Network


OBSERVATION_DIM = 115
ACTION_DIM = 21
ACTION_LAYOUT = "arm_delta7|crossdex_token9|distal_residual5"
BC_METADATA_FILENAME = "bc_bootstrap.json"
BC_FORMAT_VERSION = 1

PICK_TOOL_NOISE_GROUPS = (
    ActionNoiseGroup("arm", 0, 7, scale=1.0, zeta_mu=1.0, zeta_max=64),
    ActionNoiseGroup("token", 7, 16, scale=0.5, zeta_mu=1.25, zeta_max=32),
    ActionNoiseGroup("residual", 16, 21, scale=0.35, zeta_mu=1.5, zeta_max=16),
)


@dataclass(frozen=True)
class DemoSource:
    path: str
    sha256: str
    transitions: int
    episodes: int
    phases: dict[str, int]


@dataclass(frozen=True)
class Demonstrations:
    observation: torch.Tensor
    action: torch.Tensor
    episode_id: torch.Tensor
    phase: torch.Tensor
    sources: tuple[DemoSource, ...]

    @property
    def num_transitions(self) -> int:
        return int(self.observation.shape[0])

    @property
    def num_episodes(self) -> int:
        return int(self.episode_id.max().item()) + 1


@dataclass(frozen=True)
class EpisodeSplit:
    train_rows: torch.Tensor
    validation_rows: torch.Tensor
    train_episodes: torch.Tensor
    validation_episodes: torch.Tensor


@dataclass(frozen=True)
class BCArchitecture:
    actor_num_blocks: int = 2
    actor_hidden_dim: int = 128
    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_bins: int = 101
    critic_min_v: float = -5.0
    critic_max_v: float = 5.0


@dataclass(frozen=True)
class BCTrainConfig:
    epochs: int = 200
    batch_size: int = 2048
    learning_rate: float = 3.0e-4
    validation_fraction: float = 0.125
    atanh_epsilon: float = 1.0e-4
    huber_beta: float = 1.0
    weight_decay: float = 0.0
    gradient_clip: float = 10.0
    arm_weight: float = 1.0
    token_weight: float = 1.0
    residual_weight: float = 1.0
    target_std: float = 0.15
    std_anchor_weight: float = 0.05
    use_amp: bool = True
    seed: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_tensor(payload: Mapping[str, Any], names: Sequence[str], source: Path) -> torch.Tensor:
    for name in names:
        value = payload.get(name)
        if isinstance(value, torch.Tensor):
            return value
    joined = " or ".join(repr(name) for name in names)
    raise KeyError(f"{source} has no tensor field {joined}")


def _validate_offsets(offsets: torch.Tensor, rows: int, source: Path) -> torch.Tensor:
    offsets = offsets.to(device="cpu", dtype=torch.long)
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError(f"{source}: episode_offsets must contain at least [0, rows]")
    if int(offsets[0]) != 0 or int(offsets[-1]) != rows:
        raise ValueError(
            f"{source}: episode_offsets must start at 0 and end at {rows}, "
            f"got {int(offsets[0])} and {int(offsets[-1])}"
        )
    if bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError(f"{source}: episode_offsets must describe non-empty increasing episodes")
    return offsets


def _episode_ids_from_offsets(offsets: torch.Tensor) -> torch.Tensor:
    lengths = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(torch.arange(lengths.numel(), dtype=torch.long), lengths)


def _phase_counts(phase: torch.Tensor) -> dict[str, int]:
    values, counts = torch.unique(phase, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}


def phase_balanced_epoch_rows(
    phase: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return local row indices with exactly equal sample count per phase.

    The largest stratum is traversed once in random order.  Smaller strata are
    sampled with replacement to the same size.  Consequently, an epoch retains
    all examples from the majority phase while giving every recorded phase the
    same loss mass.  ``-1`` is an ordinary stratum, including the all-unknown
    case produced for legacy demonstration files.
    """

    if not isinstance(phase, torch.Tensor) or phase.ndim != 1 or phase.numel() < 1:
        raise ValueError("phase must be a non-empty one-dimensional tensor")
    if phase.dtype != torch.long:
        raise TypeError("phase must use torch.long dtype")
    if generator.device != phase.device:
        raise ValueError(
            f"phase sampler generator is on {generator.device}, expected {phase.device}"
        )

    values, counts = torch.unique(phase, sorted=True, return_counts=True)
    samples_per_phase = int(counts.max())
    sampled_parts: list[torch.Tensor] = []
    replacement: dict[str, bool] = {}
    for value, count_tensor in zip(values, counts, strict=True):
        candidates = torch.nonzero(phase == value, as_tuple=False).squeeze(-1)
        count = int(count_tensor)
        key = str(int(value))
        if count == samples_per_phase:
            selection = candidates.index_select(
                0,
                torch.randperm(count, generator=generator, device=phase.device),
            )
            replacement[key] = False
        else:
            selection = candidates.index_select(
                0,
                torch.randint(
                    count,
                    (samples_per_phase,),
                    generator=generator,
                    device=phase.device,
                ),
            )
            replacement[key] = True
        sampled_parts.append(selection)

    rows = torch.cat(sampled_parts)
    rows = rows.index_select(
        0,
        torch.randperm(rows.numel(), generator=generator, device=phase.device),
    )
    summary: dict[str, Any] = {
        "strategy": "phase_balanced_oversample_to_largest_stratum",
        "source_phase_counts": _phase_counts(phase),
        "samples_per_phase_per_epoch": {
            str(int(value)): samples_per_phase for value in values
        },
        "samples_per_epoch": int(rows.numel()),
        "replacement_by_phase": replacement,
    }
    return rows, summary


def _optimizer_batches(rows: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    """Split rows without duplicating a singleton final BatchNorm batch."""

    batches = list(rows.split(batch_size))
    if len(batches) > 1 and batches[-1].numel() == 1:
        batches[-2] = torch.cat((batches[-2], batches[-1]), dim=0)
        batches.pop()
    return batches


def load_demonstrations(paths: Sequence[Path | str]) -> Demonstrations:
    """Load one or more successful full-trajectory Torch demo files.

    Accepted observation keys are ``obs`` and ``observation``.  Episodes must
    be represented by ``episode_offsets``; this is required for leakage-free
    validation.  Extra transition fields are retained on disk but are not
    silently repurposed as supervision.
    """

    if not paths:
        raise ValueError("at least one demonstration path is required")

    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    episode_ids: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    sources: list[DemoSource] = []
    episode_base = 0

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError(f"{path}: demonstration root must be a mapping")

        observation = _require_tensor(payload, ("obs", "observation"), path).detach().to(
            device="cpu", dtype=torch.float32
        )
        action = _require_tensor(payload, ("action",), path).detach().to(
            device="cpu", dtype=torch.float32
        )
        if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
            raise ValueError(
                f"{path}: observation must have shape [N, {OBSERVATION_DIM}], "
                f"got {tuple(observation.shape)}"
            )
        if action.shape != (observation.shape[0], ACTION_DIM):
            raise ValueError(
                f"{path}: action must have shape [{observation.shape[0]}, {ACTION_DIM}], "
                f"got {tuple(action.shape)}"
            )
        if not bool(torch.isfinite(observation).all()):
            raise FloatingPointError(f"{path}: observation contains NaN or infinity")
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError(f"{path}: action contains NaN or infinity")
        if bool((action.abs() > 1.00001).any()):
            maximum = float(action.abs().max())
            raise ValueError(f"{path}: normalized demo action exceeds [-1, 1] (max abs={maximum})")

        offsets_value = payload.get("episode_offsets")
        if not isinstance(offsets_value, torch.Tensor):
            raise KeyError(
                f"{path}: episode_offsets is required; row-random validation would leak trajectories"
            )
        offsets = _validate_offsets(offsets_value, observation.shape[0], path)
        local_episode_id = _episode_ids_from_offsets(offsets)
        stored_episode_id = payload.get("episode_id")
        if isinstance(stored_episode_id, torch.Tensor):
            stored = stored_episode_id.to(device="cpu", dtype=torch.long)
            if stored.shape != local_episode_id.shape or not torch.equal(stored, local_episode_id):
                raise ValueError(f"{path}: episode_id disagrees with episode_offsets")

        episode_success = payload.get("episode_success")
        if isinstance(episode_success, torch.Tensor):
            success = episode_success.to(device="cpu", dtype=torch.bool)
            if success.shape != (offsets.numel() - 1,):
                raise ValueError(f"{path}: episode_success shape does not match episode_offsets")
            if not bool(success.all()):
                raise ValueError(f"{path}: failed episodes are not valid BC demonstrations")

        phase_value = payload.get("phase")
        if phase_value is None:
            phase = torch.full((observation.shape[0],), -1, dtype=torch.long)
        elif isinstance(phase_value, torch.Tensor):
            phase = phase_value.to(device="cpu", dtype=torch.long)
            if phase.shape != (observation.shape[0],):
                raise ValueError(f"{path}: phase must have one value per transition")
        else:
            raise TypeError(f"{path}: phase must be a tensor when present")

        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            declared_action_layout = meta.get("action_layout")
            if declared_action_layout not in (None, ACTION_LAYOUT):
                raise ValueError(
                    f"{path}: unsupported action_layout={declared_action_layout!r}; "
                    f"expected {ACTION_LAYOUT!r}"
                )

        num_episodes = offsets.numel() - 1
        observations.append(observation.contiguous())
        actions.append(action.clamp(-1.0, 1.0).contiguous())
        episode_ids.append(local_episode_id + episode_base)
        phases.append(phase)
        sources.append(
            DemoSource(
                path=str(path),
                sha256=_sha256(path),
                transitions=int(observation.shape[0]),
                episodes=int(num_episodes),
                phases=_phase_counts(phase),
            )
        )
        episode_base += int(num_episodes)

    dataset = Demonstrations(
        observation=torch.cat(observations, dim=0),
        action=torch.cat(actions, dim=0),
        episode_id=torch.cat(episode_ids, dim=0),
        phase=torch.cat(phases, dim=0),
        sources=tuple(sources),
    )
    if dataset.num_transitions < 2 or dataset.num_episodes < 2:
        raise ValueError("BC requires at least two transitions and two episodes")
    return dataset


def split_by_episode(dataset: Demonstrations, validation_fraction: float, seed: int) -> EpisodeSplit:
    """Create a deterministic episode-level split with no shared trajectory."""

    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be finite and in (0, 1)")
    num_episodes = dataset.num_episodes
    validation_count = round(num_episodes * validation_fraction)
    validation_count = min(max(validation_count, 1), num_episodes - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(num_episodes, generator=generator)
    validation_episodes = permutation[:validation_count].sort().values
    train_episodes = permutation[validation_count:].sort().values

    validation_episode_mask = torch.zeros(num_episodes, dtype=torch.bool)
    validation_episode_mask[validation_episodes] = True
    validation_mask = validation_episode_mask.index_select(0, dataset.episode_id)
    train_rows = (~validation_mask).nonzero(as_tuple=False).squeeze(-1)
    validation_rows = validation_mask.nonzero(as_tuple=False).squeeze(-1)
    if train_rows.numel() < 2 or validation_rows.numel() < 1:
        raise ValueError("episode split produced too few train or validation rows")
    return EpisodeSplit(
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
    )


def safe_atanh_action(action: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Map normalized demo actions to finite FlashSAC Gaussian means."""

    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.1:
        raise ValueError("atanh epsilon must be finite and in (0, 0.1)")
    limit = 1.0 - epsilon
    return torch.atanh(action.clamp(-limit, limit))


def _action_weights(config: BCTrainConfig, device: torch.device) -> torch.Tensor:
    values = (config.arm_weight, config.token_weight, config.residual_weight)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("all action-group loss weights must be finite and positive")
    return torch.tensor(
        [config.arm_weight] * 7 + [config.token_weight] * 9 + [config.residual_weight] * 5,
        dtype=torch.float32,
        device=device,
    )


def _validate_train_config(config: BCTrainConfig) -> None:
    if config.epochs < 1 or config.batch_size < 2:
        raise ValueError("epochs must be positive and batch_size must be at least two")
    for name in ("learning_rate", "huber_beta", "gradient_clip"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(config.target_std) or config.target_std <= 0.0:
        raise ValueError("target_std must be finite and positive")
    if not math.isfinite(config.std_anchor_weight) or config.std_anchor_weight < 0.0:
        raise ValueError("std_anchor_weight must be finite and non-negative")
    safe_atanh_action(torch.zeros(1), config.atanh_epsilon)


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _group_rmse(squared_error: torch.Tensor) -> dict[str, float]:
    return {
        "arm_rmse": float(squared_error[:, :7].mean().sqrt()),
        "token_rmse": float(squared_error[:, 7:16].mean().sqrt()),
        "residual_rmse": float(squared_error[:, 16:].mean().sqrt()),
    }


@torch.no_grad()
def evaluate_actor(
    actor: FlashSACActor,
    observation: torch.Tensor,
    action: torch.Tensor,
    *,
    atanh_epsilon: float,
    target_std: float | None = None,
    phase: torch.Tensor | None = None,
    batch_size: int = 8192,
) -> dict[str, Any]:
    if phase is not None and phase.shape != (observation.shape[0],):
        raise ValueError("phase must have one value per evaluated transition")
    actor.eval()
    squared_action_error: list[torch.Tensor] = []
    squared_mean_error: list[torch.Tensor] = []
    std_values: list[torch.Tensor] = []
    target_mean = safe_atanh_action(action, atanh_epsilon)
    for start in range(0, observation.shape[0], batch_size):
        stop = min(start + batch_size, observation.shape[0])
        mean, std = actor.get_mean_and_std(observation[start:stop], training=False)
        predicted_action = torch.tanh(mean)
        squared_action_error.append((predicted_action - action[start:stop]).square())
        squared_mean_error.append((mean - target_mean[start:stop]).square())
        std_values.append(std)
    action_error = torch.cat(squared_action_error, dim=0)
    mean_error = torch.cat(squared_mean_error, dim=0)
    std = torch.cat(std_values, dim=0)
    metrics = {
        "action_rmse": float(action_error.mean().sqrt()),
        "pre_tanh_rmse": float(mean_error.mean().sqrt()),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "target_saturation_fraction": float((action.abs() >= 1.0 - atanh_epsilon).float().mean()),
    }
    if target_std is not None:
        if not math.isfinite(target_std) or target_std <= 0.0:
            raise ValueError("target_std must be finite and positive")
        log_std_error = std.log() - math.log(target_std)
        metrics["log_std_prior_rmse"] = float(log_std_error.square().mean().sqrt())
        metrics["std_median"] = float(std.median())
    metrics.update(_group_rmse(action_error))
    if phase is not None:
        phase_rmse: dict[str, float] = {}
        for value in torch.unique(phase, sorted=True):
            mask = phase == value
            phase_rmse[str(int(value))] = float(action_error[mask].mean().sqrt())
        metrics["phase_action_rmse"] = phase_rmse
        metrics["phase_macro_action_rmse"] = sum(phase_rmse.values()) / len(phase_rmse)
    return metrics


def train_actor(
    dataset: Demonstrations,
    split: EpisodeSplit,
    config: BCTrainConfig,
    architecture: BCArchitecture,
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Fit FlashSAC's deterministic mean to explicit demo actions."""

    _validate_train_config(config)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    observation = dataset.observation.to(device=device, dtype=torch.float32)
    action = dataset.action.to(device=device, dtype=torch.float32)
    train_rows = split.train_rows.to(device=device)
    validation_rows = split.validation_rows.to(device=device)
    train_observation = observation.index_select(0, train_rows)
    train_action = action.index_select(0, train_rows)
    phase = dataset.phase.to(device=device, dtype=torch.long)
    train_phase = phase.index_select(0, train_rows)
    validation_observation = observation.index_select(0, validation_rows)
    validation_action = action.index_select(0, validation_rows)
    validation_phase = phase.index_select(0, validation_rows)

    actor = FlashSACActor(
        num_blocks=architecture.actor_num_blocks,
        input_dim=OBSERVATION_DIM,
        hidden_dim=architecture.actor_hidden_dim,
        action_dim=ACTION_DIM,
    ).to(device)
    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    actor_bundle = Network(
        network=actor,
        optimizer=optimizer,
        scheduler=None,
        compile_network=False,
        use_weight_normalization=True,
    )
    actor_bundle.normalize_parameters()
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    weights = _action_weights(config, device)
    weight_denominator = weights.sum()
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed + 1)

    initial_train = evaluate_actor(
        actor,
        train_observation,
        train_action,
        atanh_epsilon=config.atanh_epsilon,
        target_std=config.target_std,
        phase=train_phase,
    )
    initial_validation = evaluate_actor(
        actor,
        validation_observation,
        validation_action,
        atanh_epsilon=config.atanh_epsilon,
        target_std=config.target_std,
        phase=validation_phase,
    )
    best_validation = initial_validation["phase_macro_action_rmse"]
    best_epoch = 0
    best_state = _clone_state_dict(actor)
    loss_history: list[float] = []

    actor.train()
    for epoch in range(1, config.epochs + 1):
        permutation, sampling_summary = phase_balanced_epoch_rows(
            train_phase,
            generator=generator,
        )
        epoch_loss = torch.zeros((), dtype=torch.float32, device=device)
        epoch_examples = 0
        for row in _optimizer_batches(permutation, config.batch_size):
            batch_observation = train_observation.index_select(0, row)
            batch_action = train_action.index_select(0, row)
            target_mean = safe_atanh_action(batch_action, config.atanh_epsilon)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predicted_mean, predicted_std = actor.get_mean_and_std(
                    batch_observation, training=True
                )
                loss_per_dim = F.smooth_l1_loss(
                    predicted_mean,
                    target_mean,
                    reduction="none",
                    beta=config.huber_beta,
                )
                action_loss = (
                    (loss_per_dim * weights).sum(dim=-1).mean() / weight_denominator
                )
                log_std_target = math.log(config.target_std)
                # FlashSAC permits std down to exp(-10), which is subnormal in
                # fp16.  Evaluate this prior in fp32 so AMP cannot turn it into
                # log(0) on hardware that flushes half subnormals.
                predicted_log_std = predicted_std.float().clamp_min(1.0e-8).log()
                std_anchor_loss = (predicted_log_std - log_std_target).square().mean()
                loss = action_loss + config.std_anchor_weight * std_anchor_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            actor_bundle.normalize_parameters()
            epoch_loss.add_(loss.detach() * row.numel())
            epoch_examples += row.numel()

        loss_history.append(float(epoch_loss / epoch_examples))
        validation = evaluate_actor(
            actor,
            validation_observation,
            validation_action,
            atanh_epsilon=config.atanh_epsilon,
            target_std=config.target_std,
            phase=validation_phase,
        )
        if validation["phase_macro_action_rmse"] < best_validation:
            best_validation = validation["phase_macro_action_rmse"]
            best_epoch = epoch
            best_state = _clone_state_dict(actor)
        actor.train()

    actor.load_state_dict(best_state)
    final_train = evaluate_actor(
        actor,
        train_observation,
        train_action,
        atanh_epsilon=config.atanh_epsilon,
        target_std=config.target_std,
        phase=train_phase,
    )
    final_validation = evaluate_actor(
        actor,
        validation_observation,
        validation_action,
        atanh_epsilon=config.atanh_epsilon,
        target_std=config.target_std,
        phase=validation_phase,
    )
    metrics: dict[str, Any] = {
        "best_epoch": best_epoch,
        "epochs_requested": config.epochs,
        "selection_metric": "validation_phase_macro_action_rmse",
        "sampling": sampling_summary,
        "loss_first": loss_history[0],
        "loss_last": loss_history[-1],
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "best_train": final_train,
        "best_validation": final_validation,
    }
    return best_state, metrics


def _checkpoint_agent_config(
    architecture: BCArchitecture,
    *,
    device: torch.device,
    seed: int,
    normalize_reward: bool,
) -> Any:
    return build_agent_config(
        seed=seed,
        device_type=str(device),
        buffer_device_type=str(device),
        buffer_max_length=64,
        buffer_min_length=2,
        sample_batch_size=2,
        normalize_reward=normalize_reward,
        normalized_G_max=5.0,
        actor_num_blocks=architecture.actor_num_blocks,
        actor_hidden_dim=architecture.actor_hidden_dim,
        actor_bc_alpha=0.0,
        critic_num_blocks=architecture.critic_num_blocks,
        critic_hidden_dim=architecture.critic_hidden_dim,
        critic_num_bins=architecture.critic_num_bins,
        critic_min_v=architecture.critic_min_v,
        critic_max_v=architecture.critic_max_v,
        learning_rate_warmup_step=1,
        learning_rate_decay_step=1_000_000,
        use_compile=False,
        use_amp=device.type == "cuda",
        load_optimizer=True,
        load_reward_normalizer=normalize_reward,
    )


def export_bridge_checkpoint(
    actor_state: Mapping[str, torch.Tensor],
    output_dir: Path | str,
    *,
    architecture: BCArchitecture,
    device: torch.device | str,
    seed: int,
    normalize_reward: bool = True,
) -> FlashSACTorchBridge:
    """Export BC actor weights with fresh online-RL state around them."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    observation_space = gym.spaces.Box(
        low=-math.inf, high=math.inf, shape=(OBSERVATION_DIM,), dtype="float32"
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype="float32")
    bridge = FlashSACTorchBridge(
        observation_space,
        action_space,
        {"actor_observation_size": (OBSERVATION_DIM,), "asymmetric_obs": False},
        _checkpoint_agent_config(
            architecture,
            device=device,
            seed=seed,
            normalize_reward=normalize_reward,
        ),
        noise_groups=PICK_TOOL_NOISE_GROUPS,
    )
    bridge._actor.network.load_state_dict(actor_state)  # noqa: SLF001 - explicit export boundary
    bridge.reset_exploration()
    bridge.save(str(output))
    return bridge


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("only scalar tensors may be written to BC metadata")
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"non-finite metadata value: {value}")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def bootstrap(
    demo_paths: Sequence[Path | str],
    output_dir: Path | str,
    *,
    train_config: BCTrainConfig,
    architecture: BCArchitecture = BCArchitecture(),
    device: torch.device | str = "cuda:0",
    overwrite: bool = False,
    normalize_reward: bool = True,
) -> dict[str, Any]:
    """Load demos, train actor, and write a bridge-loadable checkpoint."""

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"production BC bootstrap requires CUDA, got {device}")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_demonstrations(demo_paths)
    split = split_by_episode(dataset, train_config.validation_fraction, train_config.seed)
    actor_state, training_metrics = train_actor(
        dataset,
        split,
        train_config,
        architecture,
        device,
    )
    bridge = export_bridge_checkpoint(
        actor_state,
        output,
        architecture=architecture,
        device=device,
        seed=train_config.seed,
        normalize_reward=normalize_reward,
    )
    del bridge

    metadata: dict[str, Any] = {
        "format_version": BC_FORMAT_VERSION,
        "checkpoint_type": "FlashSACTorchBridge",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_layout": ACTION_LAYOUT,
        "supervision": {
            "source": "explicit_demo_action",
            "target": "atanh(clamp(action))",
            "loss": "weighted_smooth_l1_pre_tanh_mean",
            "std_head_demo_supervised": False,
            "std_prior": train_config.target_std,
            "std_prior_weight": train_config.std_anchor_weight,
            "upstream_actor_bc_alpha": 0.0,
        },
        "online_state": {
            "actor_from_bc": True,
            "critic_fresh": True,
            "target_critic_fresh": True,
            "temperature_fresh": True,
            "optimizers_fresh": True,
            "amp_scaler_fresh": True,
            "reward_normalizer_fresh": normalize_reward,
            "update_step": 0,
        },
        "dataset": {
            "transitions": dataset.num_transitions,
            "episodes": dataset.num_episodes,
            "train_rows": int(split.train_rows.numel()),
            "validation_rows": int(split.validation_rows.numel()),
            "train_episodes": int(split.train_episodes.numel()),
            "validation_episodes": int(split.validation_episodes.numel()),
            "phase_counts": _phase_counts(dataset.phase),
            "train_phase_counts": _phase_counts(dataset.phase[split.train_rows]),
            "validation_phase_counts": _phase_counts(dataset.phase[split.validation_rows]),
            "sources": [asdict(source) for source in dataset.sources],
        },
        "sampling": training_metrics["sampling"],
        "architecture": asdict(architecture),
        "train_config": asdict(train_config),
        "metrics": training_metrics,
    }
    _atomic_write_json(output / BC_METADATA_FILENAME, metadata)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--demo", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--validation_fraction", type=float, default=0.125)
    parser.add_argument("--atanh_epsilon", type=float, default=1.0e-4)
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_clip", type=float, default=10.0)
    parser.add_argument("--arm_weight", type=float, default=1.0)
    parser.add_argument("--token_weight", type=float, default=1.0)
    parser.add_argument("--residual_weight", type=float, default=1.0)
    parser.add_argument("--target_std", type=float, default=0.15)
    parser.add_argument("--std_anchor_weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = BCTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        validation_fraction=args.validation_fraction,
        atanh_epsilon=args.atanh_epsilon,
        huber_beta=args.huber_beta,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        arm_weight=args.arm_weight,
        token_weight=args.token_weight,
        residual_weight=args.residual_weight,
        target_std=args.target_std,
        std_anchor_weight=args.std_anchor_weight,
        use_amp=not args.no_amp,
        seed=args.seed,
    )
    metadata = bootstrap(
        args.demo,
        args.output,
        train_config=config,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(_json_safe(metadata), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "ACTION_DIM",
    "ACTION_LAYOUT",
    "BCArchitecture",
    "BCTrainConfig",
    "Demonstrations",
    "EpisodeSplit",
    "OBSERVATION_DIM",
    "PICK_TOOL_NOISE_GROUPS",
    "bootstrap",
    "evaluate_actor",
    "export_bridge_checkpoint",
    "load_demonstrations",
    "phase_balanced_epoch_rows",
    "safe_atanh_action",
    "split_by_episode",
    "train_actor",
]
