#!/usr/bin/env python3
"""Migrate a legacy pick-tool 87-observation/16-action RL-Games checkpoint.

The output is deliberately an actor-bootstrap checkpoint rather than a resumable
training checkpoint: it contains only ``model`` and migration metadata.  Critic
weights/statistics are reset so stale values cannot leak into the new task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


OLD_OBSERVATIONS = 87
NEW_OBSERVATIONS = 115
OLD_ACTIONS = 16
NEW_ACTIONS = 21
NEW_RESIDUAL_STD = 0.05
RMS_EPSILON = 1.0e-5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand a pick-tool RL-Games actor from 87/16 to 115/21."
    )
    parser.add_argument("--input", required=True, type=Path, help="legacy .pth checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="migrated .pth checkpoint")
    parser.add_argument("--self-check-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.self_check_samples < 1:
        parser.error("--self-check-samples must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_payload(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint root must be a mapping, got {type(checkpoint).__name__}")
    if "model" in checkpoint:
        return checkpoint, "root"
    if 0 in checkpoint and isinstance(checkpoint[0], Mapping):
        return checkpoint[0], "integer key 0"
    if "0" in checkpoint and isinstance(checkpoint["0"], Mapping):
        return checkpoint["0"], "string key '0'"
    raise KeyError("checkpoint has neither a root model nor an outer {0: payload} wrapper")


def _require_tensor(state: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"legacy model is missing {key!r}")
    value = state[key]
    if not torch.is_tensor(value):
        raise TypeError(f"legacy model entry {key!r} is not a tensor")
    return value.detach().cpu()


def _linear_layer_indices(state: Mapping[str, Any]) -> list[int]:
    pattern = re.compile(r"^a2c_network\.actor_mlp\.(\d+)\.weight$")
    indices = sorted(int(match.group(1)) for key in state if (match := pattern.match(key)))
    if not indices:
        raise KeyError("legacy model has no actor_mlp linear weights")
    for index in indices:
        _require_tensor(state, f"a2c_network.actor_mlp.{index}.bias")
    return indices


def _validate_legacy_shapes(state: Mapping[str, Any]) -> list[int]:
    mean = _require_tensor(state, "running_mean_std.running_mean")
    var = _require_tensor(state, "running_mean_std.running_var")
    if mean.shape != (OLD_OBSERVATIONS,) or var.shape != (OLD_OBSERVATIONS,):
        raise ValueError(f"expected 87-D input RMS, got mean={tuple(mean.shape)}, var={tuple(var.shape)}")

    indices = _linear_layer_indices(state)
    first_weight = _require_tensor(state, f"a2c_network.actor_mlp.{indices[0]}.weight")
    if first_weight.ndim != 2 or first_weight.shape[1] != OLD_OBSERVATIONS:
        raise ValueError(f"expected actor first-layer input 87, got {tuple(first_weight.shape)}")

    previous_width = first_weight.shape[0]
    for index in indices:
        weight = _require_tensor(state, f"a2c_network.actor_mlp.{index}.weight")
        bias = _require_tensor(state, f"a2c_network.actor_mlp.{index}.bias")
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"invalid actor layer {index} shapes: {tuple(weight.shape)}, {tuple(bias.shape)}")
        if index != indices[0] and weight.shape[1] != previous_width:
            raise ValueError(f"actor layer {index} does not follow the preceding hidden width")
        previous_width = weight.shape[0]

    mu_weight = _require_tensor(state, "a2c_network.mu.weight")
    mu_bias = _require_tensor(state, "a2c_network.mu.bias")
    sigma = _require_tensor(state, "a2c_network.sigma")
    if mu_weight.shape != (OLD_ACTIONS, previous_width):
        raise ValueError(f"expected mu weight {(OLD_ACTIONS, previous_width)}, got {tuple(mu_weight.shape)}")
    if mu_bias.shape != (OLD_ACTIONS,) or sigma.shape != (OLD_ACTIONS,):
        raise ValueError(f"expected 16-D mu bias/sigma, got {tuple(mu_bias.shape)}/{tuple(sigma.shape)}")

    value_weight = _require_tensor(state, "a2c_network.value.weight")
    value_bias = _require_tensor(state, "a2c_network.value.bias")
    if value_weight.shape != (1, previous_width) or value_bias.shape != (1,):
        raise ValueError(f"unexpected value-head shapes: {tuple(value_weight.shape)}/{tuple(value_bias.shape)}")
    return indices


def _migrate_model(state: Mapping[str, Any], layer_indices: list[int]) -> OrderedDict[str, torch.Tensor]:
    migrated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"model state entry {key!r} is not a tensor")
        migrated[key] = value.detach().cpu().clone()

    input_mean = _require_tensor(state, "running_mean_std.running_mean")
    input_var = _require_tensor(state, "running_mean_std.running_var")
    new_mean = torch.zeros(NEW_OBSERVATIONS, dtype=input_mean.dtype)
    new_var = torch.ones(NEW_OBSERVATIONS, dtype=input_var.dtype)
    new_mean[:OLD_OBSERVATIONS].copy_(input_mean)
    new_var[:OLD_OBSERVATIONS].copy_(input_var)
    migrated["running_mean_std.running_mean"] = new_mean
    migrated["running_mean_std.running_var"] = new_var
    migrated["running_mean_std.count"] = _require_tensor(
        state, "running_mean_std.count"
    ).clone()

    first_key = f"a2c_network.actor_mlp.{layer_indices[0]}.weight"
    first = _require_tensor(state, first_key)
    expanded_first = torch.zeros((first.shape[0], NEW_OBSERVATIONS), dtype=first.dtype)
    expanded_first[:, :OLD_OBSERVATIONS].copy_(first)
    migrated[first_key] = expanded_first

    old_mu_weight = _require_tensor(state, "a2c_network.mu.weight")
    old_mu_bias = _require_tensor(state, "a2c_network.mu.bias")
    new_mu_weight = torch.zeros((NEW_ACTIONS, old_mu_weight.shape[1]), dtype=old_mu_weight.dtype)
    new_mu_bias = torch.zeros(NEW_ACTIONS, dtype=old_mu_bias.dtype)
    new_mu_weight[:OLD_ACTIONS].copy_(old_mu_weight)
    new_mu_bias[:OLD_ACTIONS].copy_(old_mu_bias)
    migrated["a2c_network.mu.weight"] = new_mu_weight
    migrated["a2c_network.mu.bias"] = new_mu_bias

    old_sigma = _require_tensor(state, "a2c_network.sigma")
    new_sigma = torch.full(
        (NEW_ACTIONS,), math.log(NEW_RESIDUAL_STD), dtype=old_sigma.dtype
    )
    new_sigma[:OLD_ACTIONS].copy_(old_sigma)
    migrated["a2c_network.sigma"] = new_sigma

    migrated["a2c_network.value.weight"] = torch.zeros_like(
        _require_tensor(state, "a2c_network.value.weight")
    )
    migrated["a2c_network.value.bias"] = torch.zeros_like(
        _require_tensor(state, "a2c_network.value.bias")
    )
    migrated["value_mean_std.running_mean"] = torch.zeros_like(
        _require_tensor(state, "value_mean_std.running_mean")
    )
    migrated["value_mean_std.running_var"] = torch.ones_like(
        _require_tensor(state, "value_mean_std.running_var")
    )
    migrated["value_mean_std.count"] = torch.ones_like(
        _require_tensor(state, "value_mean_std.count")
    )
    return migrated


def _actor_mu(
    state: Mapping[str, torch.Tensor], raw_observation: torch.Tensor, layer_indices: list[int]
) -> torch.Tensor:
    mean = state["running_mean_std.running_mean"].float()
    var = state["running_mean_std.running_var"].float()
    hidden = (raw_observation - mean) / torch.sqrt(var + RMS_EPSILON)
    hidden = hidden.clamp(-5.0, 5.0)
    for index in layer_indices:
        hidden = F.elu(
            F.linear(
                hidden,
                state[f"a2c_network.actor_mlp.{index}.weight"],
                state[f"a2c_network.actor_mlp.{index}.bias"],
            )
        )
    return F.linear(hidden, state["a2c_network.mu.weight"], state["a2c_network.mu.bias"])


@torch.inference_mode()
def _self_check(
    legacy: Mapping[str, Any],
    migrated: Mapping[str, torch.Tensor],
    layer_indices: list[int],
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    old_obs = torch.randn(samples, OLD_OBSERVATIONS, generator=generator)
    extra_obs = torch.randn(samples, NEW_OBSERVATIONS - OLD_OBSERVATIONS, generator=generator)
    new_obs = torch.cat((old_obs, extra_obs), dim=-1)

    legacy_tensors = {key: _require_tensor(legacy, key) for key in legacy}
    old_mu = _actor_mu(legacy_tensors, old_obs, layer_indices)
    new_mu = _actor_mu(migrated, new_obs, layer_indices)
    copied_error = (new_mu[:, :OLD_ACTIONS] - old_mu).abs().max().item()
    residual_abs_max = new_mu[:, OLD_ACTIONS:].abs().max().item()

    first_key = f"a2c_network.actor_mlp.{layer_indices[0]}.weight"
    structural_checks = {
        "input_mean_prefix": torch.equal(
            migrated["running_mean_std.running_mean"][:OLD_OBSERVATIONS],
            legacy_tensors["running_mean_std.running_mean"],
        ),
        "input_var_prefix": torch.equal(
            migrated["running_mean_std.running_var"][:OLD_OBSERVATIONS],
            legacy_tensors["running_mean_std.running_var"],
        ),
        "first_weight_prefix": torch.equal(
            migrated[first_key][:, :OLD_OBSERVATIONS], legacy_tensors[first_key]
        ),
        "mu_weight_prefix": torch.equal(
            migrated["a2c_network.mu.weight"][:OLD_ACTIONS],
            legacy_tensors["a2c_network.mu.weight"],
        ),
        "mu_bias_prefix": torch.equal(
            migrated["a2c_network.mu.bias"][:OLD_ACTIONS],
            legacy_tensors["a2c_network.mu.bias"],
        ),
    }
    if not all(structural_checks.values()):
        failed = [name for name, passed in structural_checks.items() if not passed]
        raise AssertionError(f"migration did not exactly copy: {failed}")
    if not torch.count_nonzero(migrated[first_key][:, OLD_OBSERVATIONS:]).item() == 0:
        raise AssertionError("new first-layer observation columns are not exactly zero")
    if not torch.count_nonzero(migrated["a2c_network.mu.weight"][OLD_ACTIONS:]).item() == 0:
        raise AssertionError("new mu weight rows are not exactly zero")
    if not torch.count_nonzero(migrated["a2c_network.mu.bias"][OLD_ACTIONS:]).item() == 0:
        raise AssertionError("new mu bias rows are not exactly zero")
    if copied_error >= 1.0e-6:
        raise AssertionError(f"legacy actor equivalence error {copied_error:.9g} is >= 1e-6")
    if residual_abs_max != 0.0:
        raise AssertionError(f"new residual actor outputs are nonzero: {residual_abs_max:.9g}")

    return {
        "samples": samples,
        "seed": seed,
        "legacy_mu_max_abs_error": copied_error,
        "new_residual_mu_abs_max": residual_abs_max,
    }


def main() -> None:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    raw_checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    payload, wrapper = _unwrap_payload(raw_checkpoint)
    if "model" not in payload or not isinstance(payload["model"], Mapping):
        raise KeyError("checkpoint payload does not contain a model state mapping")
    legacy_state = payload["model"]
    layer_indices = _validate_legacy_shapes(legacy_state)
    migrated_state = _migrate_model(legacy_state, layer_indices)
    self_check = _self_check(
        legacy_state, migrated_state, layer_indices, args.self_check_samples, args.seed
    )

    metadata: dict[str, Any] = {
        "format": "pick_tool_actor_bootstrap_v1",
        "source_path": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "source_wrapper": wrapper,
        "old_observations": OLD_OBSERVATIONS,
        "new_observations": NEW_OBSERVATIONS,
        "old_actions": OLD_ACTIONS,
        "new_actions": NEW_ACTIONS,
        "new_residual_log_std": math.log(NEW_RESIDUAL_STD),
        "critic_reset": True,
        "optimizer_carried": False,
        "epoch_carried": False,
        "self_check": self_check,
    }
    output_checkpoint = {0: {"model": migrated_state, "bc_meta": metadata}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_bytes": args.output.stat().st_size,
                "model_tensor_count": len(migrated_state),
                "self_check": self_check,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
