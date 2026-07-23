#!/usr/bin/env python3
"""Phase-balanced behavior cloning for a 115/120-observation, 21-action pick-tool policy.

The input checkpoint is the deliberately small migration artifact ``{0: {"model": ...}}``.
Only the actor MLP and mean head are optimized.  Observation RMS statistics, value-related
parameters and policy sigma are copied bit-for-bit to the output, whose payload intentionally
contains no optimizer state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_BLOCKS = {
    "arm": slice(0, 7),
    "token": slice(7, 16),
    "residual": slice(16, 21),
}
ACTOR_PREFIXES = ("a2c_network.actor_mlp.", "a2c_network.mu.")
MU_KEYS = {"a2c_network.mu.weight", "a2c_network.mu.bias"}

# Training state whose shape/coupling is stale after BC surgery.  A BC output is a
# weights checkpoint, never a resumable one, so these are dropped from the output payload
# while every other provenance field (option/separate/migrate metadata) is preserved.
INCOMPATIBLE_TRAINING_KEYS = frozenset(
    {
        "assymetric_vf_nets",
        "current_lengths",
        "current_rewards",
        "current_shaped_rewards",
        "dones",
        "env_state",
        "epoch",
        "frame",
        "intr_reward_model",
        "last_mean_rewards",
        "obs",
        "optimizer",
        "rnn_states",
        "scaler",
        "trackers",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a migrated pick-tool actor on physically validated oracle episodes."
    )
    parser.add_argument("--checkpoint", required=True, help="migrated {0: {model, bc_meta}} checkpoint")
    parser.add_argument("--dataset", required=True, help="oracle .pt with obs/action/phase/episode_id")
    parser.add_argument("--output", required=True, help="output {0: {model, bc_meta}} checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--bounds-weight", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1.0e-6)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--saturation-threshold", type=float, default=0.99)
    parser.add_argument(
        "--only-phase",
        type=int,
        choices=range(5),
        default=None,
        help="train only one option/phase id; useful for a non-interfering option expert",
    )
    parser.add_argument(
        "--mu-head-only",
        action="store_true",
        help="freeze actor trunk, sigma, critic/value and RMS; optimize only the separate actor mu head",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("--epochs and batch sizes must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0, 1)")
    if args.smooth_l1_beta <= 0.0:
        parser.error("--smooth-l1-beta must be positive")
    if args.bounds_weight < 0.0 or args.grad_clip <= 0.0:
        parser.error("--bounds-weight must be non-negative and --grad-clip must be positive")
    if args.patience <= 0 or args.min_delta < 0.0:
        parser.error("--patience must be positive and --min-delta must be non-negative")
    if not 0.0 < args.saturation_threshold <= 1.0:
        parser.error("--saturation-threshold must be in (0, 1]")
    return args


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


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model entry {key!r} is not a tensor: {type(value).__name__}")
        result[key] = value.detach().cpu().clone()
    return result


class MigratedActor(nn.Module):
    """RL-Games ELU actor with its observation normalization made explicit."""

    def __init__(self, model_state: dict[str, torch.Tensor]):
        super().__init__()
        mean_key = "running_mean_std.running_mean"
        var_key = "running_mean_std.running_var"
        mu_weight_key = "a2c_network.mu.weight"
        mu_bias_key = "a2c_network.mu.bias"
        for key in (mean_key, var_key, mu_weight_key, mu_bias_key):
            if key not in model_state:
                raise KeyError(f"checkpoint model is missing {key!r}")

        self.register_buffer("obs_mean", model_state[mean_key].float().clone())
        self.register_buffer("obs_var", model_state[var_key].float().clone())

        pattern = re.compile(r"^a2c_network\.actor_mlp\.(\d+)\.weight$")
        numbered = []
        for key in model_state:
            match = pattern.match(key)
            if match:
                numbered.append((int(match.group(1)), key))
        numbered.sort()
        if not numbered:
            raise RuntimeError("checkpoint contains no a2c_network.actor_mlp.*.weight layers")

        self.layer_indices = [number for number, _ in numbered]
        self.layers = nn.ModuleList()
        for number, weight_key in numbered:
            bias_key = f"a2c_network.actor_mlp.{number}.bias"
            if bias_key not in model_state:
                raise KeyError(f"checkpoint model is missing {bias_key!r}")
            weight = model_state[weight_key]
            bias = model_state[bias_key]
            if weight.ndim != 2 or bias.shape != (weight.shape[0],):
                raise RuntimeError(f"invalid linear layer shapes for actor_mlp.{number}")
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(weight.float())
                layer.bias.copy_(bias.float())
            self.layers.append(layer)

        mu_weight = model_state[mu_weight_key]
        mu_bias = model_state[mu_bias_key]
        self.mu = nn.Linear(mu_weight.shape[1], mu_weight.shape[0])
        with torch.no_grad():
            self.mu.weight.copy_(mu_weight.float())
            self.mu.bias.copy_(mu_bias.float())

        if self.layers[0].in_features != self.obs_mean.numel():
            raise RuntimeError(
                f"actor input is {self.layers[0].in_features}, but RMS has {self.obs_mean.numel()} entries"
            )
        if self.layers[-1].out_features != self.mu.in_features:
            raise RuntimeError("last actor layer and mu head have incompatible dimensions")

    @property
    def observation_dim(self) -> int:
        return self.layers[0].in_features

    @property
    def action_dim(self) -> int:
        return self.mu.out_features

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the frozen actor latent before the mean head."""

        x = observation.clamp(-5.0, 5.0)
        x = (x - self.obs_mean) / torch.sqrt(self.obs_var + 1.0e-5)
        x = x.clamp(-5.0, 5.0)
        for layer in self.layers:
            x = F.elu(layer(x))
        return x

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mu(self.encode(observation))

    @torch.no_grad()
    def write_actor_into(self, model_state: dict[str, torch.Tensor]) -> None:
        for number, layer in zip(self.layer_indices, self.layers, strict=True):
            weight_key = f"a2c_network.actor_mlp.{number}.weight"
            bias_key = f"a2c_network.actor_mlp.{number}.bias"
            model_state[weight_key].copy_(layer.weight.detach().cpu().to(model_state[weight_key].dtype))
            model_state[bias_key].copy_(layer.bias.detach().cpu().to(model_state[bias_key].dtype))
        model_state["a2c_network.mu.weight"].copy_(
            self.mu.weight.detach().cpu().to(model_state["a2c_network.mu.weight"].dtype)
        )
        model_state["a2c_network.mu.bias"].copy_(
            self.mu.bias.detach().cpu().to(model_state["a2c_network.mu.bias"].dtype)
        )


def validate_dataset(dataset: Any, observation_dim: int, action_dim: int) -> dict[str, torch.Tensor]:
    if not isinstance(dataset, dict):
        raise TypeError("dataset root must be a dictionary")
    required = ("obs", "action", "phase", "episode_id")
    for key in required:
        if key not in dataset or not isinstance(dataset[key], torch.Tensor):
            raise KeyError(f"dataset is missing tensor {key!r}")

    obs = dataset["obs"].detach().cpu().float()
    action = dataset["action"].detach().cpu().float()
    phase = dataset["phase"].detach().cpu().long().flatten()
    episode_id = dataset["episode_id"].detach().cpu().long().flatten()
    if obs.ndim != 2 or obs.shape[1] != observation_dim:
        raise RuntimeError(f"expected obs [N,{observation_dim}], got {tuple(obs.shape)}")
    if action.ndim != 2 or action.shape != (obs.shape[0], action_dim):
        raise RuntimeError(f"expected action [N,{action_dim}], got {tuple(action.shape)}")
    if phase.shape != (obs.shape[0],) or episode_id.shape != (obs.shape[0],):
        raise RuntimeError("phase and episode_id must each have one entry per transition")
    if obs.shape[0] == 0:
        raise RuntimeError("dataset contains no transitions")
    if not torch.isfinite(obs).all() or not torch.isfinite(action).all():
        raise RuntimeError("dataset contains non-finite observations or actions")
    if float(action.abs().max()) > 1.0001:
        raise RuntimeError(f"dataset action exceeds [-1,1]: max abs={float(action.abs().max()):.6g}")
    if episode_id.unique().numel() < 2:
        raise RuntimeError("at least two episodes are required for an episode-disjoint train/val split")
    return {"obs": obs, "action": action, "phase": phase, "episode_id": episode_id}


def episode_split(
    episode_id: torch.Tensor, val_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    episodes = episode_id.unique(sorted=True)
    generator = torch.Generator().manual_seed(seed)
    shuffled = episodes[torch.randperm(episodes.numel(), generator=generator)]
    val_count = max(1, int(round(episodes.numel() * val_fraction)))
    val_count = min(val_count, episodes.numel() - 1)
    val_episodes = shuffled[:val_count]
    train_episodes = shuffled[val_count:]

    val_mask = torch.zeros_like(episode_id, dtype=torch.bool)
    for episode in val_episodes:
        val_mask |= episode_id == episode
    train_indices = (~val_mask).nonzero(as_tuple=False).squeeze(-1)
    val_indices = val_mask.nonzero(as_tuple=False).squeeze(-1)
    return (
        train_indices,
        val_indices,
        sorted(int(x) for x in train_episodes.tolist()),
        sorted(int(x) for x in val_episodes.tolist()),
    )


def phase_balanced_batches(
    indices: torch.Tensor,
    phase: torch.Tensor,
    requested_batch_size: int,
    generator: torch.Generator,
):
    phase_values = phase[indices].unique(sorted=True)
    if phase_values.numel() == 0:
        raise RuntimeError("training split contains no phases")
    pools = [indices[phase[indices] == value] for value in phase_values]
    batch_size = max(min(requested_batch_size, indices.numel()), phase_values.numel())
    batches = max(1, math.ceil(indices.numel() / batch_size))
    phase_count = len(pools)
    for batch_number in range(batches):
        base, remainder = divmod(batch_size, phase_count)
        parts = []
        for offset, pool in enumerate(pools):
            # Rotate the remainder so no phase always receives the extra item.
            count = base + int(((offset - batch_number) % phase_count) < remainder)
            selected = torch.randint(pool.numel(), (count,), generator=generator)
            parts.append(pool[selected])
        batch = torch.cat(parts)
        yield batch[torch.randperm(batch.numel(), generator=generator)]


def block_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    bounds_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imitation_parts = []
    bounds_parts = []
    for block_slice in ACTION_BLOCKS.values():
        imitation_parts.append(
            F.smooth_l1_loss(prediction[:, block_slice], target[:, block_slice], beta=beta)
        )
        bounds_parts.append(F.relu(prediction[:, block_slice].abs() - 1.0).square().mean())
    imitation = torch.stack(imitation_parts).mean()
    bounds = torch.stack(bounds_parts).mean()
    return imitation + bounds_weight * bounds, imitation, bounds


@torch.inference_mode()
def predict(
    actor: MigratedActor,
    observations: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    actor.eval()
    chunks = []
    for start in range(0, observations.shape[0], batch_size):
        chunk = observations[start : start + batch_size].to(device)
        chunks.append(actor(chunk).cpu())
    return torch.cat(chunks)


def metrics_from_predictions(
    prediction: torch.Tensor,
    target: torch.Tensor,
    phase: torch.Tensor,
    beta: float,
    bounds_weight: float,
    saturation_threshold: float,
) -> dict[str, Any]:
    def summarize(mask: torch.Tensor) -> dict[str, Any]:
        pred = prediction[mask]
        truth = target[mask]
        blocks: dict[str, Any] = {}
        phase_losses = []
        phase_bounds = []
        for name, block_slice in ACTION_BLOCKS.items():
            block_pred = pred[:, block_slice]
            block_truth = truth[:, block_slice]
            error = block_pred - block_truth
            imitation = F.smooth_l1_loss(block_pred, block_truth, beta=beta)
            bounds = F.relu(block_pred.abs() - 1.0).square().mean()
            phase_losses.append(float(imitation))
            phase_bounds.append(float(bounds))
            blocks[name] = {
                "rmse": float(error.square().mean().sqrt()),
                "prediction_saturation_rate": float(
                    (block_pred.abs() >= saturation_threshold).float().mean()
                ),
                "target_saturation_rate": float(
                    (block_truth.abs() >= saturation_threshold).float().mean()
                ),
                "prediction_out_of_bounds_rate": float((block_pred.abs() > 1.0).float().mean()),
            }
        return {
            "samples": int(mask.sum()),
            "objective": sum(phase_losses) / len(phase_losses)
            + bounds_weight * sum(phase_bounds) / len(phase_bounds),
            "blocks": blocks,
        }

    overall_mask = torch.ones(prediction.shape[0], dtype=torch.bool)
    per_phase = {str(int(value)): summarize(phase == value) for value in phase.unique(sorted=True)}
    balanced_objective = sum(entry["objective"] for entry in per_phase.values()) / len(per_phase)
    result = summarize(overall_mask)
    result["phase_balanced_objective"] = balanced_objective
    result["per_phase"] = per_phase
    return result


def phase_balanced_validation_loss(
    actor: MigratedActor,
    observations: torch.Tensor,
    actions: torch.Tensor,
    phases: torch.Tensor,
    device: torch.device,
    eval_batch_size: int,
    beta: float,
    bounds_weight: float,
) -> float:
    actor.eval()
    losses = []
    with torch.inference_mode():
        for value in phases.unique(sorted=True):
            phase_indices = (phases == value).nonzero(as_tuple=False).squeeze(-1)
            weighted_sum = 0.0
            sample_count = 0
            for start in range(0, phase_indices.numel(), eval_batch_size):
                selected = phase_indices[start : start + eval_batch_size]
                prediction = actor(observations[selected].to(device))
                target = actions[selected].to(device)
                loss, _, _ = block_objective(prediction, target, beta, bounds_weight)
                weighted_sum += float(loss) * selected.numel()
                sample_count += selected.numel()
            losses.append(weighted_sum / sample_count)
    return sum(losses) / len(losses)


def max_group_difference(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], predicate
) -> float:
    differences = []
    for key, initial in before.items():
        if predicate(key):
            differences.append(float((after[key] - initial).abs().max()))
    return max(differences, default=0.0)


def print_split_metrics(name: str, metrics: dict[str, Any]) -> None:
    print(
        f"{name}: samples={metrics['samples']} "
        f"phase-balanced-objective={metrics['phase_balanced_objective']:.6f}",
        flush=True,
    )
    for block_name in ACTION_BLOCKS:
        block = metrics["blocks"][block_name]
        print(
            f"  {block_name:8s} RMSE={block['rmse']:.5f} "
            f"sat(pred/target)={block['prediction_saturation_rate']:.3f}/"
            f"{block['target_saturation_rate']:.3f} "
            f"out-of-bounds={block['prediction_out_of_bounds_rate']:.3f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    raw_checkpoint = load_torch(checkpoint_path)
    if not isinstance(raw_checkpoint, dict):
        raise TypeError("checkpoint root must be a dictionary")
    if isinstance(raw_checkpoint.get("model"), dict):
        payload = raw_checkpoint
    elif 0 in raw_checkpoint:
        payload = raw_checkpoint[0]
    elif "0" in raw_checkpoint:
        payload = raw_checkpoint["0"]
    else:
        raise KeyError("checkpoint must contain model directly or below root key 0")
    if not isinstance(payload, dict) or "model" not in payload:
        raise KeyError("checkpoint root 0 must contain model")
    if not isinstance(payload["model"], dict):
        raise TypeError("checkpoint model must be a state dictionary")

    initial_model = clone_state(payload["model"])
    output_model = clone_state(payload["model"])
    actor = MigratedActor(initial_model).to(device)
    if actor.observation_dim not in (115, 120) or actor.action_dim != 21:
        raise RuntimeError(
            f"BC is defined for a migrated 115-or-120/21 policy, got "
            f"{actor.observation_dim}/{actor.action_dim}"
        )
    has_separate_critic = any(
        key.startswith("a2c_network.critic_mlp.") for key in initial_model
    )
    if args.mu_head_only and not has_separate_critic:
        raise RuntimeError("--mu-head-only requires a separate actor/critic checkpoint")
    # Full-actor BC trains the actor trunk.  On a shared-trunk checkpoint that trunk is
    # also the value trunk, so the value function silently changes even though every
    # tensor-level "value" check passes.  The pipeline is only safe because the migrate
    # step zeroes the value head (0 output regardless of trunk).  Refuse full-actor BC on
    # a shared trunk whose value head is still live, rather than emit a checkpoint whose
    # "value bit-identical" guarantee is false.
    if not args.mu_head_only and not has_separate_critic:
        value_weight = initial_model.get("a2c_network.value.weight")
        value_bias = initial_model.get("a2c_network.value.bias")
        value_live = (
            value_weight is not None and bool(torch.count_nonzero(value_weight))
        ) or (value_bias is not None and bool(torch.count_nonzero(value_bias)))
        if value_live:
            raise RuntimeError(
                "full-actor BC on a shared trunk with a live value head would change the "
                "value function while all tensor-level checks pass; separate the critic "
                "first, use --mu-head-only, or zero the value head (as migrate does)"
            )

    raw_dataset = load_torch(dataset_path)
    data = validate_dataset(raw_dataset, actor.observation_dim, actor.action_dim)
    if args.only_phase is not None:
        selected_phase = data["phase"] == args.only_phase
        if not bool(selected_phase.any()):
            raise RuntimeError(f"dataset contains no rows for --only-phase {args.only_phase}")
        data = {key: value[selected_phase] for key, value in data.items()}
    train_indices, val_indices, train_episodes, val_episodes = episode_split(
        data["episode_id"], args.val_fraction, args.seed
    )
    train_phase_values = data["phase"][train_indices].unique(sorted=True).tolist()
    val_phase_values = data["phase"][val_indices].unique(sorted=True).tolist()
    if train_phase_values != val_phase_values:
        raise RuntimeError(
            f"train/val phase sets differ: train={train_phase_values}, val={val_phase_values}"
        )
    print(
        f"device={device}; transitions={data['obs'].shape[0]}; "
        f"episodes train/val={len(train_episodes)}/{len(val_episodes)}; "
        f"phases={train_phase_values}",
        flush=True,
    )

    # This probe isolates the migrated legacy policy: every observation appended after the
    # original 87-D policy input is zero, including option observations when present.
    # It is evaluated before and after BC so reach-policy drift is visible rather than assumed.
    probe_observations = data["obs"][val_indices].clone()
    probe_observations[:, 87:] = 0.0
    probe_description = (
        "validation observations with appended dimensions "
        f"[87:{actor.observation_dim}] zeroed"
    )
    initial_probe = predict(actor, probe_observations, device, args.eval_batch_size)
    initial_val_prediction = predict(actor, data["obs"][val_indices], device, args.eval_batch_size)
    initial_val_metrics = metrics_from_predictions(
        initial_val_prediction,
        data["action"][val_indices],
        data["phase"][val_indices],
        args.smooth_l1_beta,
        args.bounds_weight,
        args.saturation_threshold,
    )
    print_split_metrics("validation before BC", initial_val_metrics)

    if args.mu_head_only:
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        for parameter in actor.mu.parameters():
            parameter.requires_grad_(True)
        trainable_parameters = list(actor.mu.parameters())
    else:
        trainable_parameters = list(actor.parameters())
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed + 1)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_ran = 0
    for epoch in range(1, args.epochs + 1):
        actor.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for selected in phase_balanced_batches(
            train_indices, data["phase"], args.batch_size, generator
        ):
            observation = data["obs"][selected].to(device)
            target = data["action"][selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = actor(observation)
            loss, _, _ = block_objective(
                prediction, target, args.smooth_l1_beta, args.bounds_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_batches += 1

        train_loss = epoch_loss / epoch_batches
        val_loss = phase_balanced_validation_loss(
            actor,
            data["obs"][val_indices],
            data["action"][val_indices],
            data["phase"][val_indices],
            device,
            args.eval_batch_size,
            args.smooth_l1_beta,
            args.bounds_weight,
        )
        epochs_ran = epoch
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or improved or epoch % args.log_every == 0:
            marker = " *" if improved else ""
            print(
                f"epoch {epoch:4d}: train={train_loss:.6f} val-balanced={val_loss:.6f}{marker}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            print(f"early stop at epoch {epoch} (patience={args.patience})", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    actor.load_state_dict(best_state)
    actor.write_actor_into(output_model)

    changed_model_keys = {
        key for key in initial_model if not torch.equal(initial_model[key], output_model[key])
    }
    allowed_changed_keys = MU_KEYS if args.mu_head_only else {
        key for key in initial_model if key.startswith(ACTOR_PREFIXES)
    }
    unexpected_changed_keys = changed_model_keys - allowed_changed_keys
    if unexpected_changed_keys:
        raise RuntimeError(
            "serialized tensors changed outside the requested trainable boundary: "
            f"{sorted(unexpected_changed_keys)}"
        )
    if args.mu_head_only and not (changed_model_keys & MU_KEYS):
        raise RuntimeError("--mu-head-only training did not change either mu tensor")
    frozen_tensors_bit_exact = all(
        torch.equal(initial_model[key], output_model[key])
        for key in initial_model
        if key not in allowed_changed_keys
    )
    if not frozen_tensors_bit_exact:
        raise RuntimeError("a frozen model tensor is not bit-exact")

    # Assert the requested freeze boundary at the serialized-state level, not merely optimizer setup.
    non_actor_diff = max_group_difference(
        initial_model, output_model, lambda key: not key.startswith(ACTOR_PREFIXES)
    )
    rms_diff = max_group_difference(initial_model, output_model, lambda key: "running_mean_std" in key)
    value_diff = max_group_difference(
        initial_model,
        output_model,
        lambda key: "value" in key and not key.startswith(ACTOR_PREFIXES),
    )
    sigma_diff = max_group_difference(initial_model, output_model, lambda key: key.endswith(".sigma"))
    if non_actor_diff != 0.0:
        raise RuntimeError(f"a frozen model tensor changed (max abs diff={non_actor_diff:.9g})")

    final_probe = predict(actor, probe_observations, device, args.eval_batch_size)
    legacy_drift = final_probe[:, :16] - initial_probe[:, :16]
    migration_preservation = {
        "probe": probe_description,
        "legacy_action_first16_rmse": float(legacy_drift.square().mean().sqrt()),
        "legacy_action_first16_max_abs": float(legacy_drift.abs().max()),
        "initial_residual5_abs_max": float(initial_probe[:, 16:].abs().max()),
        "final_residual5_abs_max": float(final_probe[:, 16:].abs().max()),
        "frozen_non_actor_max_abs": non_actor_diff,
        "frozen_rms_max_abs": rms_diff,
        "frozen_value_max_abs": value_diff,
        "frozen_sigma_max_abs": sigma_diff,
        "mu_head_only": args.mu_head_only,
        "changed_model_keys": sorted(changed_model_keys),
        "trainable_model_keys": sorted(MU_KEYS) if args.mu_head_only else sorted(allowed_changed_keys),
        "all_frozen_tensors_bit_exact": frozen_tensors_bit_exact,
    }
    print(
        "migration preservation (new obs zeroed): "
        f"legacy16 drift RMSE={migration_preservation['legacy_action_first16_rmse']:.6f} "
        f"max={migration_preservation['legacy_action_first16_max_abs']:.6f}; "
        f"frozen RMS/value/sigma={rms_diff:.1e}/{value_diff:.1e}/{sigma_diff:.1e}",
        flush=True,
    )

    train_prediction = predict(actor, data["obs"][train_indices], device, args.eval_batch_size)
    val_prediction = predict(actor, data["obs"][val_indices], device, args.eval_batch_size)
    train_metrics = metrics_from_predictions(
        train_prediction,
        data["action"][train_indices],
        data["phase"][train_indices],
        args.smooth_l1_beta,
        args.bounds_weight,
        args.saturation_threshold,
    )
    val_metrics = metrics_from_predictions(
        val_prediction,
        data["action"][val_indices],
        data["phase"][val_indices],
        args.smooth_l1_beta,
        args.bounds_weight,
        args.saturation_threshold,
    )
    print_split_metrics("train after BC", train_metrics)
    print_split_metrics("validation after BC", val_metrics)

    existing_meta = payload.get("bc_meta", {})
    if not isinstance(existing_meta, dict):
        existing_meta = {"input_bc_meta": copy.deepcopy(existing_meta)}
    merged_meta = copy.deepcopy(existing_meta)
    phase_names = {}
    if isinstance(raw_dataset, dict) and isinstance(raw_dataset.get("meta"), dict):
        names = raw_dataset["meta"].get("phase_names")
        if isinstance(names, (list, tuple)):
            phase_names = {str(index): str(name) for index, name in enumerate(names)}
    merged_meta["behavior_cloning"] = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256(dataset_path),
        "only_phase": args.only_phase,
        "mu_head_only": args.mu_head_only,
        "seed": args.seed,
        "device": str(device),
        "transitions": int(data["obs"].shape[0]),
        "train_episode_ids": train_episodes,
        "validation_episode_ids": val_episodes,
        "phase_ids": [int(value) for value in train_phase_values],
        "phase_names": phase_names,
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "epochs_ran": epochs_ran,
            "best_epoch": best_epoch,
            "best_phase_balanced_validation_objective": best_loss,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "val_fraction": args.val_fraction,
            "smooth_l1_beta": args.smooth_l1_beta,
            "bounds_weight": args.bounds_weight,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "initial_validation": initial_val_metrics,
        "train": train_metrics,
        "validation": val_metrics,
        "migration_preservation": migration_preservation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {"model": output_model, "bc_meta": merged_meta}
    # Preserve upstream provenance (option_checkpoint_meta, migrate/separate metadata, ...)
    # so BC does not silently break the audit chain; drop only the shape-stale training
    # state and the keys we (re)generate below.
    for key, value in payload.items():
        if key in {"model", "bc_meta", "separate_actor_critic_meta"}:
            continue
        if key in INCOMPATIBLE_TRAINING_KEYS:
            continue
        output_payload[key] = copy.deepcopy(value)
    separate_meta = payload.get("separate_actor_critic_meta")
    if isinstance(separate_meta, dict):
        output_payload["separate_actor_critic_meta"] = copy.deepcopy(separate_meta)
    elif any(key.startswith("a2c_network.critic_mlp.") for key in output_model):
        output_payload["separate_actor_critic_meta"] = {
            "format_version": 1,
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_checkpoint_sha256": sha256(checkpoint_path),
            "actor_layer_indices": actor.layer_indices,
            "requires_network_separate": True,
            "self_check": {
                "all_frozen_tensors_bit_exact": frozen_tensors_bit_exact,
                "changed_model_keys": sorted(changed_model_keys),
            },
        }
    torch.save({0: output_payload}, output_path)
    print(f"wrote BC checkpoint without optimizer: {output_path.resolve()}", flush=True)
    print(
        "payload keys="
        + json.dumps({"root": [0], "entry": sorted(output_payload)}, separators=(",", ":")),
        flush=True,
    )


if __name__ == "__main__":
    main()
