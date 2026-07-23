#!/usr/bin/env python3
"""Convert a shared-trunk RL-Games checkpoint to separate actor/critic MLPs.

The actor path, Gaussian log standard deviation and normalization statistics are copied
bit-for-bit.  The new critic MLP starts as an exact copy of the old shared actor MLP, so
the initial value prediction is also unchanged.  Subsequent critic gradients can no
longer alter the actor trunk.  Because the topology changes, optimizer and rollout state
are deliberately stripped: the result is a weights-initialization checkpoint, not a
resumable checkpoint.
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


RMS_EPSILON = 1.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--self-check-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--expect-source-sha256",
        type=str,
        default=None,
        help="abort unless the input checkpoint's SHA-256 matches, binding this stage to a "
        "specific upstream checkpoint.",
    )
    args = parser.parse_args()
    if args.self_check_samples < 1:
        parser.error("--self-check-samples must be positive")
    return args


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap(checkpoint: Any) -> tuple[Mapping[str, Any], int | str | None]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint root must be a mapping, got {type(checkpoint).__name__}")
    if isinstance(checkpoint.get("model"), Mapping):
        return checkpoint, None
    for key in (0, "0"):
        payload = checkpoint.get(key)
        if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
            return payload, key
    raise KeyError("checkpoint must contain a model mapping at root or below key 0")


def require_tensor(state: Mapping[str, Any], key: str) -> torch.Tensor:
    value = state.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"model is missing tensor {key!r}")
    return value.detach().cpu()


def actor_layer_indices(state: Mapping[str, Any]) -> list[int]:
    pattern = re.compile(r"^a2c_network\.actor_mlp\.(\d+)\.weight$")
    indices = sorted(int(match.group(1)) for key in state if (match := pattern.match(key)))
    if not indices:
        raise KeyError("model contains no actor MLP layers")
    for index in indices:
        weight = require_tensor(state, f"a2c_network.actor_mlp.{index}.weight")
        bias = require_tensor(state, f"a2c_network.actor_mlp.{index}.bias")
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"invalid actor layer {index}: {tuple(weight.shape)}, {tuple(bias.shape)}")
    return indices


def convert_model(state: Mapping[str, Any], indices: list[int]) -> OrderedDict[str, torch.Tensor]:
    if any(key.startswith("a2c_network.critic_mlp.") for key in state):
        raise ValueError("checkpoint already contains a separate critic MLP")
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"model entry {key!r} is not a tensor")
        converted[key] = value.detach().cpu().clone()
    for index in indices:
        for suffix in ("weight", "bias"):
            actor_key = f"a2c_network.actor_mlp.{index}.{suffix}"
            critic_key = f"a2c_network.critic_mlp.{index}.{suffix}"
            converted[critic_key] = require_tensor(state, actor_key).clone()
    return converted


def forward_heads(
    state: Mapping[str, torch.Tensor], raw_observation: torch.Tensor, indices: list[int], *, critic: bool
) -> torch.Tensor:
    hidden = (raw_observation - state["running_mean_std.running_mean"].float()) / torch.sqrt(
        state["running_mean_std.running_var"].float() + RMS_EPSILON
    )
    hidden = hidden.clamp(-5.0, 5.0)
    trunk = "critic_mlp" if critic else "actor_mlp"
    for index in indices:
        hidden = F.elu(
            F.linear(
                hidden,
                state[f"a2c_network.{trunk}.{index}.weight"].float(),
                state[f"a2c_network.{trunk}.{index}.bias"].float(),
            )
        )
    head = "value" if critic else "mu"
    return F.linear(
        hidden,
        state[f"a2c_network.{head}.weight"].float(),
        state[f"a2c_network.{head}.bias"].float(),
    )


@torch.inference_mode()
def self_check(
    old: Mapping[str, Any], new: Mapping[str, torch.Tensor], indices: list[int], samples: int, seed: int
) -> dict[str, float | int | bool]:
    old_tensors = {key: require_tensor(old, key) for key in old}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    obs_dim = require_tensor(old, "running_mean_std.running_mean").numel()
    observation = torch.randn(samples, obs_dim, generator=generator)
    old_mu = forward_heads(old_tensors, observation, indices, critic=False)
    new_mu = forward_heads(new, observation, indices, critic=False)
    # Reuse the old shared actor path with the value head for a direct value-equivalence check.
    hidden = (observation - old_tensors["running_mean_std.running_mean"].float()) / torch.sqrt(
        old_tensors["running_mean_std.running_var"].float() + RMS_EPSILON
    )
    hidden = hidden.clamp(-5.0, 5.0)
    for index in indices:
        hidden = F.elu(
            F.linear(
                hidden,
                old_tensors[f"a2c_network.actor_mlp.{index}.weight"].float(),
                old_tensors[f"a2c_network.actor_mlp.{index}.bias"].float(),
            )
        )
    old_value = F.linear(
        hidden,
        old_tensors["a2c_network.value.weight"].float(),
        old_tensors["a2c_network.value.bias"].float(),
    )
    new_value = forward_heads(new, observation, indices, critic=True)

    mu_error = float((new_mu - old_mu).abs().max())
    value_error = float((new_value - old_value).abs().max())
    if mu_error != 0.0 or value_error != 0.0:
        raise AssertionError(f"conversion changed outputs: mu={mu_error:.9g}, value={value_error:.9g}")
    for key, tensor in old_tensors.items():
        if not torch.equal(new[key], tensor):
            raise AssertionError(f"source tensor changed: {key}")
    for index in indices:
        for suffix in ("weight", "bias"):
            actor_key = f"a2c_network.actor_mlp.{index}.{suffix}"
            critic_key = f"a2c_network.critic_mlp.{index}.{suffix}"
            if not torch.equal(new[critic_key], old_tensors[actor_key]):
                raise AssertionError(f"critic copy differs: {critic_key}")
    return {
        "samples": samples,
        "seed": seed,
        "actor_mu_max_abs_error": mu_error,
        "value_max_abs_error": value_error,
        "all_source_model_tensors_bit_exact": True,
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.expect_source_sha256 is not None:
        actual_sha = sha256(args.input)
        if actual_sha != args.expect_source_sha256:
            raise RuntimeError(
                f"input SHA-256 {actual_sha} does not match --expect-source-sha256 "
                f"{args.expect_source_sha256}"
            )
    checkpoint = load_torch(args.input)
    payload, wrapper_key = unwrap(checkpoint)
    old_state = payload["model"]
    indices = actor_layer_indices(old_state)
    new_state = convert_model(old_state, indices)
    check = self_check(old_state, new_state, indices, args.self_check_samples, args.seed)

    incompatible_training_keys = {
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
    stripped_keys = sorted(key for key in payload if key in incompatible_training_keys)
    new_payload = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key != "model" and key not in incompatible_training_keys
    }
    new_payload["model"] = new_state
    new_payload["separate_actor_critic_meta"] = {
        "format_version": 1,
        "source_checkpoint": str(args.input.resolve()),
        "source_checkpoint_sha256": sha256(args.input),
        "actor_layer_indices": indices,
        "requires_network_separate": True,
        "checkpoint_load_mode": "weights",
        "stripped_training_state_keys": stripped_keys,
        "self_check": check,
    }
    output: Any = new_payload if wrapper_key is None else {wrapper_key: new_payload}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps(new_payload["separate_actor_critic_meta"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
