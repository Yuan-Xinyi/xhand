#!/usr/bin/env python3
"""Fit a close-only distal residual without updating the frozen base actor."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from bc_pick_tool import MigratedActor, clone_state, episode_split, load_torch
from distal_residual_adapter import DistalResidualAdapter, make_payload, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--delta-limit", type=float, default=0.10)
    parser.add_argument("--proximity-threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        parser.error("epochs, batch size and learning rate must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val fraction must be in (0,1)")
    if args.patience <= 0 or args.delta_limit <= 0.0:
        parser.error("patience and delta limit must be positive")
    if not 0.0 <= args.proximity_threshold <= 1.0:
        parser.error("proximity threshold must be in [0,1]")
    checkpoint = Path(args.checkpoint).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    sidecar = output.with_suffix(".json")
    if output.suffix not in (".pt", ".pth"):
        parser.error("--output must end in .pt or .pth so its JSON sidecar cannot overwrite it")
    if len({checkpoint, dataset, output, sidecar}) != 4:
        parser.error("checkpoint, dataset, output and output JSON sidecar must be distinct paths")
    return args


def checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    raw = load_torch(path)
    payload = raw if isinstance(raw, dict) and isinstance(raw.get("model"), dict) else None
    if payload is None and isinstance(raw, dict):
        payload = raw.get(0, raw.get("0"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise KeyError("checkpoint must contain a model state dictionary")
    return clone_state(payload["model"])


@torch.inference_mode()
def actor_features(
    actor: MigratedActor, obs: torch.Tensor, batch_size: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_parts, action_parts = [], []
    for start in range(0, obs.shape[0], batch_size):
        batch = obs[start : start + batch_size].to(actor.obs_mean.device)
        latent = actor.encode(batch)
        latent_parts.append(latent.cpu())
        action_parts.append(actor.mu(latent).clamp(-1.0, 1.0).cpu())
    return torch.cat(latent_parts), torch.cat(action_parts)


@torch.inference_mode()
def evaluate(adapter, latent, obs, base_distal, target, indices) -> dict[str, float]:
    prediction = adapter(
        latent[indices].to(next(adapter.parameters()).device),
        obs[indices].to(next(adapter.parameters()).device),
        base_distal[indices].to(next(adapter.parameters()).device),
    ).cpu()
    return {
        "smooth_l1": float(F.smooth_l1_loss(prediction, target[indices], beta=0.05)),
        "normalized_mae": float((prediction - target[indices]).abs().mean()),
        "saturation_fraction": float((prediction.abs() >= 0.99).float().mean()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    actor = MigratedActor(checkpoint_model(checkpoint)).to(args.device).eval()
    if actor.observation_dim != 115 or actor.action_dim != 21:
        raise RuntimeError(
            f"base actor must use 115 observations and 21 actions, got "
            f"{actor.observation_dim}/{actor.action_dim}"
        )
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    raw = load_torch(dataset_path)
    if not isinstance(raw, dict):
        raise TypeError("dataset root must be a dictionary")
    for key in ("obs", "action", "phase", "episode_id"):
        if not isinstance(raw.get(key), torch.Tensor):
            raise KeyError(f"dataset is missing tensor {key!r}")
    obs = raw["obs"].detach().cpu().float()
    action = raw["action"].detach().cpu().float()
    phase = raw["phase"].detach().cpu().long().flatten()
    episode_id = raw["episode_id"].detach().cpu().long().flatten()
    if obs.ndim != 2 or obs.shape[1] != 115 or action.shape != (obs.shape[0], 21):
        raise RuntimeError("dataset must use 115 observations and 21 actions")
    if phase.shape != (obs.shape[0],) or episode_id.shape != (obs.shape[0],):
        raise RuntimeError("phase and episode_id must contain one value per transition")
    if not torch.isfinite(obs).all() or not torch.isfinite(action).all():
        raise RuntimeError("dataset contains non-finite observations or actions")
    if float(action.abs().max()) > 1.0001:
        raise RuntimeError("dataset actions exceed the normalized [-1,1] range")
    meta = raw.get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("phase_names"), list):
        raise RuntimeError("dataset metadata must declare phase_names")
    phase_names = meta["phase_names"]
    if phase_names.count("close") != 1:
        raise RuntimeError("dataset phase_names must contain exactly one 'close' phase")
    close_phase = phase_names.index("close")
    selected = (phase == close_phase) & (
        obs[:, 92:97].mean(dim=-1) >= args.proximity_threshold
    )
    selected &= obs[:, 106] < 0.5
    rows = selected.nonzero(as_tuple=False).squeeze(-1)
    obs = obs[rows]
    action = action[rows]
    episode_id = episode_id[rows]
    if episode_id.unique().numel() < 2:
        raise RuntimeError("filtered close data contains fewer than two episodes")
    latent, base_action = actor_features(actor, obs)
    base_distal = base_action[:, 16:21]
    raw_delta = action[:, 16:21] - base_distal
    target = (raw_delta / args.delta_limit).clamp(-1.0, 1.0)
    train_idx, val_idx, train_episodes, val_episodes = episode_split(
        episode_id, args.val_fraction, args.seed
    )

    adapter = DistalResidualAdapter(
        latent_dim=latent.shape[1],
        hidden_dim=32,
        delta_limit=args.delta_limit,
        proximity_threshold=args.proximity_threshold,
    ).to(args.device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 1)
    best_state = copy.deepcopy(adapter.state_dict())
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        permutation = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
        imitation_sum = 0.0
        objective_sum = 0.0
        count = 0
        for start in range(0, permutation.numel(), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            prediction = adapter(
                latent[indices].to(args.device),
                obs[indices].to(args.device),
                base_distal[indices].to(args.device),
            )
            desired = target[indices].to(args.device)
            imitation = F.smooth_l1_loss(prediction, desired, beta=0.05)
            loss = imitation + 1.0e-3 * prediction.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            imitation_sum += float(imitation) * prediction.numel()
            objective_sum += float(loss) * prediction.numel()
            count += prediction.numel()
        adapter.eval()
        val = evaluate(adapter, latent, obs, base_distal, target, val_idx)
        history.append(
            {
                "epoch": epoch,
                "train_imitation_loss": imitation_sum / count,
                "train_objective": objective_sum / count,
                **val,
            }
        )
        if val["smooth_l1"] < best_loss - 1.0e-6:
            best_loss = val["smooth_l1"]
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} imitation={imitation_sum/count:.6f} "
                f"objective={objective_sum/count:.6f} val={val['smooth_l1']:.6f} "
                f"mae={val['normalized_mae']:.4f} sat={val['saturation_fraction']:.3f}",
                flush=True,
            )
        if stale >= args.patience:
            break

    adapter.load_state_dict(best_state)
    adapter.eval()
    meta = {
        "kind": "close_only_distal_corrective_dagger",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256(dataset_path),
        "source_rows": int(raw["obs"].shape[0]),
        "selected_rows": int(rows.numel()),
        "train_rows": int(train_idx.numel()),
        "val_rows": int(val_idx.numel()),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "raw_delta_abs_mean": float(raw_delta.abs().mean()),
        "within_limit_fraction": float((raw_delta.abs() <= args.delta_limit).float().mean()),
        "train_metrics": evaluate(adapter, latent, obs, base_distal, target, train_idx),
        "val_metrics": evaluate(adapter, latent, obs, base_distal, target, val_idx),
        "history_tail": history[-10:],
        "seed": args.seed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(make_payload(adapter, checkpoint, meta), output)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"wrote {output.resolve()} rows={rows.numel()} best_epoch={best_epoch} "
        f"val={best_loss:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
