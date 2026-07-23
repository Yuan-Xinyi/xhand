#!/usr/bin/env python3
"""Append five option observations to a 115-observation/21-action actor checkpoint.

The new observations are deliberately inert at conversion time: their RMS mean/variance
are 0/1 and the corresponding actor first-layer columns are zero.  Consequently the old
21-dimensional policy output is independent of every value of the appended option vector.
All other model tensors and payload fields are preserved exactly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


OLD_OBSERVATIONS = 115
OPTION_OBSERVATIONS = 5
NEW_OBSERVATIONS = OLD_OBSERVATIONS + OPTION_OBSERVATIONS
ACTIONS = 21
RMS_EPSILON = 1.0e-5

# The first actor layer changes shape (115 -> 120 input), so any optimizer moment or
# rollout counter carried from the source is stale and would either crash at the first
# optimizer.step() or silently skew schedules if train.py classifies the payload as a
# resumable checkpoint.  Strip the same keys separate_actor_critic_checkpoint.py strips.
INCOMPATIBLE_TRAINING_KEYS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand a {0: {model, ...}} pick-tool actor from 115/21 to 120/21."
    )
    parser.add_argument("--input", required=True, type=Path, help="115-observation checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="120-observation checkpoint")
    parser.add_argument("--self-check-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.self_check_samples < 1:
        parser.error("--self-check-samples must be positive")
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


def require_tensor(state: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"model is missing {key!r}")
    value = state[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"model entry {key!r} is not a tensor: {type(value).__name__}")
    return value.detach().cpu()


def actor_layer_indices(state: Mapping[str, Any]) -> list[int]:
    pattern = re.compile(r"^a2c_network\.actor_mlp\.(\d+)\.weight$")
    indices = sorted(int(match.group(1)) for key in state if (match := pattern.match(key)))
    if not indices:
        raise KeyError("model contains no a2c_network.actor_mlp.*.weight layers")
    return indices


def validate_model(state: Mapping[str, Any]) -> list[int]:
    # This tool widens only the actor first layer and the observation RMS.  A checkpoint
    # that already carries a separate critic MLP would keep its critic at 115-input while
    # the RMS/actor move to 120, and the self-check (actor-only) cannot notice it, so the
    # break surfaces only at online-training load_state_dict.  Refuse it up front.
    critic_keys = sorted(key for key in state if key.startswith("a2c_network.critic_mlp."))
    if critic_keys:
        raise ValueError(
            "checkpoint has a separate critic MLP (e.g. "
            f"{critic_keys[0]!r}); append the option observations before separating the "
            "critic, or widen the critic first layer symmetrically"
        )

    mean = require_tensor(state, "running_mean_std.running_mean")
    var = require_tensor(state, "running_mean_std.running_var")
    if mean.shape != (OLD_OBSERVATIONS,) or var.shape != (OLD_OBSERVATIONS,):
        raise ValueError(
            f"expected 115-D observation RMS, got mean={tuple(mean.shape)}, var={tuple(var.shape)}"
        )

    indices = actor_layer_indices(state)
    first = require_tensor(state, f"a2c_network.actor_mlp.{indices[0]}.weight")
    if first.ndim != 2 or first.shape[1] != OLD_OBSERVATIONS:
        raise ValueError(f"expected actor first-layer input 115, got {tuple(first.shape)}")

    previous_width = first.shape[0]
    for offset, index in enumerate(indices):
        weight = require_tensor(state, f"a2c_network.actor_mlp.{index}.weight")
        bias = require_tensor(state, f"a2c_network.actor_mlp.{index}.bias")
        expected_input = OLD_OBSERVATIONS if offset == 0 else previous_width
        if weight.ndim != 2 or weight.shape[1] != expected_input:
            raise ValueError(f"invalid actor layer {index} weight shape {tuple(weight.shape)}")
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"invalid actor layer {index} bias shape {tuple(bias.shape)}")
        previous_width = weight.shape[0]

    mu_weight = require_tensor(state, "a2c_network.mu.weight")
    mu_bias = require_tensor(state, "a2c_network.mu.bias")
    sigma = require_tensor(state, "a2c_network.sigma")
    if mu_weight.shape != (ACTIONS, previous_width):
        raise ValueError(f"expected mu weight {(ACTIONS, previous_width)}, got {tuple(mu_weight.shape)}")
    if mu_bias.shape != (ACTIONS,) or sigma.shape != (ACTIONS,):
        raise ValueError(
            f"expected 21-D mu bias/sigma, got {tuple(mu_bias.shape)}/{tuple(sigma.shape)}"
        )
    return indices


def expand_model(
    state: Mapping[str, Any], layer_indices: list[int]
) -> OrderedDict[str, torch.Tensor]:
    expanded: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model entry {key!r} is not a tensor: {type(value).__name__}")
        expanded[key] = value.detach().cpu().clone()

    old_mean = require_tensor(state, "running_mean_std.running_mean")
    old_var = require_tensor(state, "running_mean_std.running_var")
    new_mean = old_mean.new_zeros(NEW_OBSERVATIONS)
    new_var = old_var.new_ones(NEW_OBSERVATIONS)
    new_mean[:OLD_OBSERVATIONS].copy_(old_mean)
    new_var[:OLD_OBSERVATIONS].copy_(old_var)
    expanded["running_mean_std.running_mean"] = new_mean
    expanded["running_mean_std.running_var"] = new_var

    first_key = f"a2c_network.actor_mlp.{layer_indices[0]}.weight"
    old_first = require_tensor(state, first_key)
    new_first = old_first.new_zeros((old_first.shape[0], NEW_OBSERVATIONS))
    new_first[:, :OLD_OBSERVATIONS].copy_(old_first)
    expanded[first_key] = new_first
    return expanded


def actor_mu(
    state: Mapping[str, torch.Tensor], observation: torch.Tensor, layer_indices: list[int]
) -> torch.Tensor:
    hidden = observation.clamp(-5.0, 5.0)
    hidden = (hidden - state["running_mean_std.running_mean"].float()) / torch.sqrt(
        state["running_mean_std.running_var"].float() + RMS_EPSILON
    )
    hidden = hidden.clamp(-5.0, 5.0)
    for index in layer_indices:
        hidden = F.elu(
            F.linear(
                hidden,
                state[f"a2c_network.actor_mlp.{index}.weight"].float(),
                state[f"a2c_network.actor_mlp.{index}.bias"].float(),
            )
        )
    return F.linear(
        hidden,
        state["a2c_network.mu.weight"].float(),
        state["a2c_network.mu.bias"].float(),
    )


@torch.inference_mode()
def self_check(
    old: Mapping[str, Any],
    new: Mapping[str, torch.Tensor],
    layer_indices: list[int],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    old_observation = torch.randn(samples, OLD_OBSERVATIONS, generator=generator)
    # Use unconstrained values (including values beyond the observation clip) to verify that
    # the option vector is structurally irrelevant, not merely zero on the probe batch.
    arbitrary_option = 10.0 * torch.randn(
        samples, OPTION_OBSERVATIONS, generator=generator
    )
    new_observation = torch.cat((old_observation, arbitrary_option), dim=-1)

    old_tensors = {key: require_tensor(old, key) for key in old}
    old_output = actor_mu(old_tensors, old_observation, layer_indices)
    new_output = actor_mu(new, new_observation, layer_indices)
    output_error = float((new_output - old_output).abs().max())

    first_key = f"a2c_network.actor_mlp.{layer_indices[0]}.weight"
    exact_checks = {
        "rms_mean_prefix": torch.equal(
            new["running_mean_std.running_mean"][:OLD_OBSERVATIONS],
            old_tensors["running_mean_std.running_mean"],
        ),
        "rms_var_prefix": torch.equal(
            new["running_mean_std.running_var"][:OLD_OBSERVATIONS],
            old_tensors["running_mean_std.running_var"],
        ),
        "actor_first_layer_prefix": torch.equal(
            new[first_key][:, :OLD_OBSERVATIONS], old_tensors[first_key]
        ),
    }
    changed_keys = {
        "running_mean_std.running_mean",
        "running_mean_std.running_var",
        first_key,
    }
    preserved_keys = {
        key: torch.equal(new[key], old_tensors[key]) for key in old_tensors if key not in changed_keys
    }
    failed = [name for name, passed in exact_checks.items() if not passed]
    failed.extend(f"preserved:{key}" for key, passed in preserved_keys.items() if not passed)
    if failed:
        raise AssertionError(f"conversion did not preserve tensors exactly: {failed}")
    if torch.count_nonzero(new["running_mean_std.running_mean"][OLD_OBSERVATIONS:]).item():
        raise AssertionError("appended RMS means are not exactly zero")
    if not torch.equal(
        new["running_mean_std.running_var"][OLD_OBSERVATIONS:],
        torch.ones_like(new["running_mean_std.running_var"][OLD_OBSERVATIONS:]),
    ):
        raise AssertionError("appended RMS variances are not exactly one")
    if torch.count_nonzero(new[first_key][:, OLD_OBSERVATIONS:]).item():
        raise AssertionError("appended actor first-layer columns are not exactly zero")
    if output_error >= 1.0e-6:
        raise AssertionError(f"old action output max error {output_error:.9g} is >= 1e-6")

    return {
        "samples": samples,
        "seed": seed,
        "old_action_output_max_abs_error": output_error,
        "arbitrary_option_abs_max": float(arbitrary_option.abs().max()),
        "all_unmodified_model_tensors_bit_exact": True,
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    checkpoint = load_torch(args.input)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint root must be a mapping, got {type(checkpoint).__name__}")
    if 0 in checkpoint:
        root_key: int | str = 0
    elif "0" in checkpoint:
        root_key = "0"
    else:
        raise KeyError("checkpoint must contain a root 0 payload")
    payload = checkpoint[root_key]
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise KeyError("checkpoint root 0 must contain a model state mapping")

    old_state = payload["model"]
    layer_indices = validate_model(old_state)
    new_state = expand_model(old_state, layer_indices)
    check = self_check(old_state, new_state, layer_indices, args.self_check_samples, args.seed)

    stripped_keys = sorted(key for key in payload if key in INCOMPATIBLE_TRAINING_KEYS)
    metadata = {
        "format": "pick_tool_option_observations_v1",
        "source_path": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "old_observation_dim": OLD_OBSERVATIONS,
        "option_observation_dim": OPTION_OBSERVATIONS,
        "new_observation_dim": NEW_OBSERVATIONS,
        "action_dim": ACTIONS,
        "option_rms_mean": 0.0,
        "option_rms_variance": 1.0,
        "option_actor_columns": "zero",
        "checkpoint_load_mode": "weights",
        "stripped_training_state_keys": stripped_keys,
        "self_check": check,
    }

    # Preserve every non-model payload field except the shape-stale training state; the
    # widened first layer is incompatible with any carried optimizer/rollout counters.
    output_payload = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key != "model" and key not in INCOMPATIBLE_TRAINING_KEYS
    }
    output_payload["model"] = new_state
    output_payload["option_checkpoint_meta"] = metadata
    output_checkpoint = {
        key: copy.deepcopy(value) for key, value in checkpoint.items() if key != root_key
    }
    output_checkpoint[root_key] = output_payload

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    print(
        f"PASS 115/21 -> 120/21; old action max error={check['old_action_output_max_abs_error']:.3g}",
        flush=True,
    )
    print(f"wrote {args.output.resolve()}", flush=True)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
