#!/usr/bin/env python3
"""Deterministic, strict physical evaluation for PickTool FlashSAC checkpoints.

The evaluator deliberately runs in a fresh Isaac process.  It reconstructs the
same production network architecture as :mod:`train`, loads a checkpoint
without optimizer or reward-normalizer state, and executes ``tanh(actor_mean)``
without exploration noise.

Isaac Lab's DirectRLEnv auto-resets completed sub-environments before ``step``
returns.  The task therefore clones its per-environment event flags, true mesh
clearance and grasp latch into ``pick_tool_terminal`` inside ``_get_dones``.
Those reset-before clones are the sole authority for the episode being closed;
the physical state visible after ``step`` is used only to initialize the next
episode on a reset row.

Success, failure, timeout, drop, unsafe-force and unlatched-5-cm counts are
episode events.  They are never inferred from aggregate log means or from a
post-reset state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch


OBSERVATION_DIM = 115
ACTION_DIM = 21
PRODUCTION_ACTOR_BLOCKS = 2
PRODUCTION_ACTOR_HIDDEN = 128
PRODUCTION_CRITIC_BLOCKS = 2
PRODUCTION_CRITIC_HIDDEN = 256
PRODUCTION_CRITIC_BINS = 101
SMOKE_ACTOR_BLOCKS = 1
SMOKE_ACTOR_HIDDEN = 32
SMOKE_CRITIC_BLOCKS = 1
SMOKE_CRITIC_HIDDEN = 64
SMOKE_CRITIC_BINS = 51

# This is checkpoint metadata as well as interaction behavior.  Bridge loading
# intentionally rejects a different grouping, so keep the evaluator contract
# explicit and simulation-free-testable.
NOISE_GROUP_SPECS = (
    ("arm", 0, 7, 1.0, 1.0, 64),
    ("token", 7, 16, 0.5, 1.25, 32),
    ("residual", 16, 21, 0.35, 1.5, 16),
)

TERMINAL_EVENT_KEYS = (
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
)


def _require_vector(
    name: str,
    value: Any,
    *,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.shape != (num_envs,) or value.dtype != dtype or value.device != device:
        raise ValueError(
            f"{name} must be {dtype}[{num_envs}] on {device}, "
            f"got {value.dtype}{tuple(value.shape)} on {value.device}"
        )
    return value


def _terminal_mapping(info: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(info, Mapping):
        raise TypeError(f"adapter info must be a mapping, got {type(info).__name__}")
    raw = info.get("pick_tool_terminal")
    if not isinstance(raw, Mapping):
        raise KeyError("adapter info has no pick_tool_terminal ground truth")
    return raw


def validate_terminal_events(
    info: Mapping[str, Any],
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return task-authored terminal tensors after checking Gym consistency."""

    num_envs = int(terminated.numel())
    if terminated.shape != (num_envs,) or terminated.dtype != torch.bool:
        raise ValueError("terminated must be a one-dimensional bool tensor")
    if truncated.shape != terminated.shape or truncated.dtype != torch.bool:
        raise ValueError("truncated must match the terminated bool vector")
    if truncated.device != terminated.device:
        raise ValueError("terminated and truncated must be on the same device")

    raw = _terminal_mapping(info)
    events = {
        name: _require_vector(
            f"pick_tool_terminal[{name!r}]",
            raw.get(name),
            num_envs=num_envs,
            device=terminated.device,
            dtype=torch.bool,
        )
        for name in TERMINAL_EVENT_KEYS
    }

    task_terminated = events["success"] | events["failure"]
    if not torch.equal(task_terminated, terminated):
        raise RuntimeError(
            "pick_tool_terminal success/failure does not match the reset-before terminated mask"
        )
    if not torch.equal(events["time_out"], truncated):
        raise RuntimeError(
            "pick_tool_terminal time_out does not match the reset-before truncated mask"
        )
    if bool((events["success"] & events["failure"]).any()):
        raise RuntimeError("an episode cannot be both a strict success and task failure")
    failure_sources = (
        events["dropped"]
        | events["unsafe_force"]
        | events["unlatched_clearance_ge_5cm"]
    )
    if not torch.equal(events["failure"], failure_sources):
        raise RuntimeError(
            "task failure must be exactly drop, unsafe force, or unlatched lift"
        )
    if bool((task_terminated & events["time_out"]).any()):
        raise RuntimeError("task termination and timeout must be mutually exclusive")
    return events


@dataclass(frozen=True)
class PhysicalTruth:
    """Per-environment physical state used by strict episode metrics."""

    clearance: torch.Tensor
    grasped: torch.Tensor

    def validate(self, *, num_envs: int, device: torch.device, name: str) -> None:
        _require_vector(
            f"{name}.clearance",
            self.clearance,
            num_envs=num_envs,
            device=device,
            dtype=torch.float32,
        )
        _require_vector(
            f"{name}.grasped",
            self.grasped,
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        )
        if not bool(torch.isfinite(self.clearance).all()):
            raise FloatingPointError(f"{name}.clearance contains NaN or infinity")

def physical_truth_from_terminal_info(
    info: Mapping[str, Any],
    *,
    num_envs: int,
    device: torch.device,
) -> PhysicalTruth:
    """Read per-step physical truth cloned by ``_get_dones`` before reset."""

    raw = _terminal_mapping(info)
    truth = PhysicalTruth(
        _require_vector(
            "pick_tool_terminal['true_clearance']",
            raw.get("true_clearance"),
            num_envs=num_envs,
            device=device,
            dtype=torch.float32,
        ),
        _require_vector(
            "pick_tool_terminal['is_grasped']",
            raw.get("is_grasped"),
            num_envs=num_envs,
            device=device,
            dtype=torch.bool,
        ),
    )
    truth.validate(num_envs=num_envs, device=device, name="pick_tool_terminal")
    success = raw.get("success")
    if isinstance(success, torch.Tensor):
        if bool((success & (truth.clearance < 0.20 - 1.0e-6)).any()):
            raise RuntimeError("strict success reported below 20 cm true mesh clearance")
        if bool((success & ~truth.grasped).any()):
            raise RuntimeError("strict success reported without the grasp latch")
    return truth


def _read_physical_truth(unwrapped: Any) -> PhysicalTruth:
    """Read true mesh clearance and the strict task latch from PickTool."""

    unwrapped._compute_intermediate_values()
    clearance = (unwrapped._object_true_min_z() - unwrapped._table_surface_z).detach().clone()
    grasped = unwrapped._is_grasped.detach().clone()
    truth = PhysicalTruth(clearance.to(dtype=torch.float32), grasped.to(dtype=torch.bool))
    truth.validate(
        num_envs=int(unwrapped.num_envs),
        device=torch.device(str(unwrapped.device)),
        name="runtime_truth",
    )
    return truth


def episode_quotas(episodes: int, num_envs: int, *, device: torch.device) -> torch.Tensor:
    """Assign an exact, deterministic number of episodes to every env slot."""

    if episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num_envs must be positive")
    base, remainder = divmod(episodes, num_envs)
    quotas = torch.full((num_envs,), base, dtype=torch.long, device=device)
    quotas[:remainder] += 1
    return quotas


class StrictEpisodeTracker:
    """Track exact per-episode physical events across vector auto-resets."""

    def __init__(
        self,
        *,
        episodes: int,
        num_envs: int,
        device: torch.device,
        initial_truth: PhysicalTruth,
    ) -> None:
        initial_truth.validate(num_envs=num_envs, device=device, name="initial_truth")
        self.episodes = episodes
        self.num_envs = num_envs
        self.device = device
        self.quotas = episode_quotas(episodes, num_envs, device=device)
        self.completed_by_slot = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_clearance = initial_truth.clearance.clone()
        self.ever_grasped = initial_truth.grasped.clone()
        self.ever_5cm = initial_truth.clearance >= 0.05
        self.ever_20cm = initial_truth.clearance >= 0.20
        self.ever_unlatched_5cm = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.records: list[dict[str, Any]] = []

    @property
    def active(self) -> torch.Tensor:
        return self.completed_by_slot < self.quotas

    @property
    def complete(self) -> bool:
        return len(self.records) == self.episodes

    def step(
        self,
        *,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        events: Mapping[str, torch.Tensor],
        transition_truth: PhysicalTruth,
        post_reset_truth: PhysicalTruth,
    ) -> None:
        _require_vector(
            "reward",
            reward,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )
        _require_vector(
            "terminated",
            terminated,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        _require_vector(
            "truncated",
            truncated,
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        transition_truth.validate(
            num_envs=self.num_envs, device=self.device, name="transition_truth"
        )
        post_reset_truth.validate(
            num_envs=self.num_envs, device=self.device, name="post_reset_truth"
        )
        for name in TERMINAL_EVENT_KEYS:
            _require_vector(
                f"events[{name!r}]",
                events.get(name),
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )

        active = self.active
        self.returns.add_(torch.where(active, reward, 0.0))
        self.lengths.add_(active.long())
        self.max_clearance = torch.where(
            active,
            torch.maximum(self.max_clearance, transition_truth.clearance),
            self.max_clearance,
        )
        self.ever_grasped |= active & transition_truth.grasped
        self.ever_5cm |= active & (transition_truth.clearance >= 0.05)
        self.ever_20cm |= active & (transition_truth.clearance >= 0.20)
        self.ever_unlatched_5cm |= active & events["unlatched_clearance_ge_5cm"]

        accepted_done = active & (terminated | truncated)
        ids = accepted_done.nonzero(as_tuple=False).squeeze(-1)
        for env_id in ids.detach().cpu().tolist():
            record = {
                "episode_index": len(self.records),
                "env_slot": env_id,
                "slot_episode_index": int(self.completed_by_slot[env_id].item()),
                "return": float(self.returns[env_id].item()),
                "length": int(self.lengths[env_id].item()),
                "max_true_clearance_m": float(self.max_clearance[env_id].item()),
                "ever_grasped": bool(self.ever_grasped[env_id].item()),
                "ever_clearance_ge_5cm": bool(self.ever_5cm[env_id].item()),
                "ever_clearance_ge_20cm": bool(self.ever_20cm[env_id].item()),
                "success": bool(events["success"][env_id].item()),
                "failure": bool(events["failure"][env_id].item()),
                "time_out": bool(events["time_out"][env_id].item()),
                "dropped": bool(events["dropped"][env_id].item()),
                "unsafe_force": bool(events["unsafe_force"][env_id].item()),
                "ever_unlatched_clearance_ge_5cm": bool(
                    self.ever_unlatched_5cm[env_id].item()
                ),
            }
            if not math.isfinite(record["return"]) or not math.isfinite(
                record["max_true_clearance_m"]
            ):
                raise FloatingPointError("completed episode contains a non-finite metric")
            self.records.append(record)

        self.completed_by_slot.add_(accepted_done.long())
        # DirectRLEnv already started the next episode on these rows.  Seed its
        # accumulator with that reset state's physical truth, not the old
        # transition terminal state.
        self.returns.masked_fill_(accepted_done, 0.0)
        self.lengths.masked_fill_(accepted_done, 0)
        self.max_clearance = torch.where(
            accepted_done, post_reset_truth.clearance, self.max_clearance
        )
        self.ever_grasped = torch.where(
            accepted_done, post_reset_truth.grasped, self.ever_grasped
        )
        self.ever_5cm = torch.where(
            accepted_done, post_reset_truth.clearance >= 0.05, self.ever_5cm
        )
        self.ever_20cm = torch.where(
            accepted_done, post_reset_truth.clearance >= 0.20, self.ever_20cm
        )
        self.ever_unlatched_5cm.masked_fill_(accepted_done, False)

        if len(self.records) > self.episodes:
            raise RuntimeError("episode tracker exceeded its exact episode quota")


def summarize(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("summary input contains NaN or infinity")
    quantiles = torch.quantile(
        tensor,
        torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=tensor.dtype),
    )
    return {
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "max": float(quantiles[4]),
        "mean": float(tensor.mean()),
    }


def build_strict_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint: Path,
    architecture: str,
    seed: int,
    num_envs: int,
    vector_steps: int,
    max_vector_steps: int,
    episode_length_s: float,
    max_episode_steps: int,
    curriculum_dataset: Path | None,
    curriculum_dataset_sha256: str | None,
    curriculum_boundary: str,
    curriculum_probability: float,
    curriculum_joint_noise: float,
    use_compile: bool,
    upstream_commit: str,
) -> dict[str, Any]:
    if not records:
        raise ValueError("strict evaluation completed no episodes")
    event_names = TERMINAL_EVENT_KEYS[:-1]
    event_counts = {
        name: sum(bool(record[name]) for record in records) for name in event_names
    }
    event_counts["unlatched_clearance_ge_5cm"] = sum(
        bool(record["ever_unlatched_clearance_ge_5cm"]) for record in records
    )
    funnel = {
        "ever_grasped": sum(bool(record["ever_grasped"]) for record in records),
        "ever_clearance_ge_5cm": sum(
            bool(record["ever_clearance_ge_5cm"]) for record in records
        ),
        "ever_clearance_ge_20cm": sum(
            bool(record["ever_clearance_ge_20cm"]) for record in records
        ),
    }
    episodes = len(records)
    return {
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "flashsac_upstream_commit": upstream_commit,
        "policy": "deterministic_tanh_actor_mean",
        "architecture": architecture,
        "critic_bins": (
            PRODUCTION_CRITIC_BINS if architecture == "production" else SMOKE_CRITIC_BINS
        ),
        "noise_groups": [
            {
                "name": name,
                "start": start,
                "stop": stop,
                "scale": scale,
                "zeta_mu": zeta_mu,
                "zeta_max": zeta_max,
            }
            for name, start, stop, scale, zeta_mu, zeta_max in NOISE_GROUP_SPECS
        ],
        "use_compile": use_compile,
        "seed": seed,
        "num_envs": num_envs,
        "requested_episodes": episodes,
        "completed_episodes": episodes,
        "vector_steps": vector_steps,
        "max_vector_steps": max_vector_steps,
        "simulated_environment_steps": vector_steps * num_envs,
        "episode_length_s": episode_length_s,
        "max_episode_steps": max_episode_steps,
        "curriculum": {
            "dataset": (
                str(curriculum_dataset.resolve()) if curriculum_dataset is not None else None
            ),
            "dataset_sha256": curriculum_dataset_sha256,
            "boundary": curriculum_boundary if curriculum_dataset is not None else None,
            "probability": curriculum_probability,
            "joint_noise": curriculum_joint_noise,
        },
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "events": event_counts,
        "strict_success_rate": event_counts["success"] / episodes,
        "funnel": funnel,
        "max_true_clearance_m": summarize(
            [float(record["max_true_clearance_m"]) for record in records]
        ),
        "episode_return": summarize([float(record["return"]) for record in records]),
        "episode_length": summarize([int(record["length"]) for record in records]),
        "episodes": list(records),
    }


_COMPILED_PREFIX = "_orig_mod."
_ENCODER_BLOCK_PATTERN = re.compile(r"^encoder\.(\d+)\.w1\.w\.weight$")


def _canonical_actor_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not state:
        raise ValueError("actor network state is empty")
    prefixed = [str(key).startswith(_COMPILED_PREFIX) for key in state]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError("actor checkpoint mixes compiled and uncompiled root keys")
    canonical: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("actor network state must map string keys to tensors")
        key = raw_key.removeprefix(_COMPILED_PREFIX) if all(prefixed) else raw_key
        canonical[key] = value
    return canonical


def infer_actor_architecture_from_state(state: Mapping[str, Any]) -> str:
    """Recognize the production and smoke architectures saved by train.py."""

    canonical = _canonical_actor_state(state)
    embed = canonical.get("embedder.w.w.weight")
    mean = canonical.get("predictor.mean_w.w.weight")
    if embed is None or mean is None or embed.ndim != 2 or mean.ndim != 2:
        raise RuntimeError("actor checkpoint is missing compatible embedder/predictor weights")
    if tuple(embed.shape)[1] != OBSERVATION_DIM or tuple(mean.shape)[0] != ACTION_DIM:
        raise RuntimeError(
            f"actor checkpoint dimensions are not PickTool {OBSERVATION_DIM}/{ACTION_DIM}: "
            f"embed={tuple(embed.shape)}, mean={tuple(mean.shape)}"
        )
    hidden = int(embed.shape[0])
    if int(mean.shape[1]) != hidden:
        raise RuntimeError("actor embedder and predictor hidden dimensions differ")
    block_indices = sorted(
        int(match.group(1))
        for key in canonical
        if (match := _ENCODER_BLOCK_PATTERN.match(key)) is not None
    )
    if block_indices != list(range(len(block_indices))):
        raise RuntimeError(f"actor encoder blocks are non-contiguous: {block_indices}")
    signature = (len(block_indices), hidden)
    if signature == (PRODUCTION_ACTOR_BLOCKS, PRODUCTION_ACTOR_HIDDEN):
        return "production"
    if signature == (SMOKE_ACTOR_BLOCKS, SMOKE_ACTOR_HIDDEN):
        return "smoke"
    raise RuntimeError(f"unsupported actor architecture blocks/hidden={signature}")


def resolve_checkpoint_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file() and resolved.name == "actor.pt":
        resolved = resolved.parent
    if not resolved.is_dir():
        raise FileNotFoundError(f"FlashSAC checkpoint directory does not exist: {resolved}")
    required = ("actor.pt", "critic.pt", "target_critic.pt", "temperature.pt")
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint {resolved} is missing files: {missing}")
    return resolved


def infer_checkpoint_architecture(checkpoint: Path) -> str:
    payload = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("network_state_dict"), Mapping
    ):
        raise TypeError("actor.pt must contain a mapping network_state_dict")
    return infer_actor_architecture_from_state(payload["network_state_dict"])


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--architecture",
        choices=("production", "smoke", "auto"),
        default="production",
        help="Production exactly matches formal train.py; auto also recognizes its smoke architecture.",
    )
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument(
        "--curriculum_dataset",
        type=Path,
        default=None,
        help="Optional physically captured reset-boundary dataset, matching train.py.",
    )
    parser.add_argument("--curriculum_boundary", default="close_start")
    parser.add_argument("--curriculum_probability", type=float, default=0.0)
    parser.add_argument("--curriculum_joint_noise", type=float, default=0.0)
    parser.add_argument("--max_vector_steps", type=int, default=None)
    parser.add_argument("--validate_finite", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/pick_tool_flashsac_eval.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    return args, launcher


def validate_curriculum_config(
    *,
    dataset: Path | None,
    probability: float,
    joint_noise: float,
) -> None:
    """Apply the same curriculum argument contract as formal training."""

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("--curriculum_probability must be in [0, 1]")
    if not math.isfinite(joint_noise) or joint_noise < 0.0:
        raise ValueError("--curriculum_joint_noise must be finite and non-negative")
    if probability > 0.0 and dataset is None:
        raise ValueError("--curriculum_probability > 0 requires --curriculum_dataset")
    if dataset is not None and not dataset.is_file():
        raise FileNotFoundError(dataset)


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 1 or args.num_envs < 1:
        raise ValueError("--episodes and --num_envs must be positive")
    if args.max_vector_steps is not None and args.max_vector_steps < 1:
        raise ValueError("--max_vector_steps must be positive")
    if args.episode_length_s is not None and (
        not math.isfinite(args.episode_length_s) or args.episode_length_s <= 0.0
    ):
        raise ValueError("--episode_length_s must be finite and positive")
    validate_curriculum_config(
        dataset=args.curriculum_dataset,
        probability=args.curriculum_probability,
        joint_noise=args.curriculum_joint_noise,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    _seed_everything(args.seed)

    from adapter import make_pick_tool_env
    from agent_bridge import (
        FLASH_SAC_COMMIT,
        ActionNoiseGroup,
        FlashSACTorchBridge,
        build_agent_config,
    )

    device_string = str(args.device or "cuda:0")
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"PickTool FlashSAC evaluation requires CUDA, got {device_string}")

    checkpoint = resolve_checkpoint_directory(args.checkpoint)
    checkpoint_architecture = infer_checkpoint_architecture(checkpoint)
    architecture = checkpoint_architecture if args.architecture == "auto" else args.architecture
    if architecture != checkpoint_architecture:
        raise RuntimeError(
            f"requested {architecture} architecture but checkpoint is {checkpoint_architecture}; "
            "use --architecture auto only when smoke-checkpoint evaluation is intentional"
        )

    cfg_overrides: dict[str, Any] = {}
    if args.episode_length_s is not None:
        cfg_overrides["episode_length_s"] = args.episode_length_s
    if args.curriculum_dataset is not None:
        cfg_overrides.update(
            {
                "curriculum_dataset": str(args.curriculum_dataset.resolve()),
                "curriculum_boundary": args.curriculum_boundary,
                "curriculum_reset_probability": args.curriculum_probability,
                "curriculum_joint_noise": args.curriculum_joint_noise,
            }
        )
    env = make_pick_tool_env(
        num_envs=args.num_envs,
        device=device_string,
        seed=args.seed,
        cfg_overrides=cfg_overrides,
        validate_finite=args.validate_finite,
    )

    if architecture == "production":
        actor_blocks, actor_hidden = PRODUCTION_ACTOR_BLOCKS, PRODUCTION_ACTOR_HIDDEN
        critic_blocks, critic_hidden = PRODUCTION_CRITIC_BLOCKS, PRODUCTION_CRITIC_HIDDEN
    else:
        actor_blocks, actor_hidden = SMOKE_ACTOR_BLOCKS, SMOKE_ACTOR_HIDDEN
        critic_blocks, critic_hidden = SMOKE_CRITIC_BLOCKS, SMOKE_CRITIC_HIDDEN

    # All structural values match formal train.py.  Replay/schedule values are
    # deliberately tiny because evaluation neither inserts nor updates data.
    agent_cfg = build_agent_config(
        seed=args.seed,
        normalize_reward=True,
        normalized_G_max=5.0,
        device_type=device_string,
        buffer_device_type=device_string,
        buffer_max_length=max(args.num_envs, 32),
        buffer_min_length=1,
        sample_batch_size=1,
        n_step=3 if architecture == "production" else 1,
        actor_num_blocks=actor_blocks,
        actor_hidden_dim=actor_hidden,
        critic_num_blocks=critic_blocks,
        critic_hidden_dim=critic_hidden,
        critic_num_bins=(
            PRODUCTION_CRITIC_BINS if architecture == "production" else SMOKE_CRITIC_BINS
        ),
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        use_amp=architecture == "production",
        load_optimizer=False,
        load_reward_normalizer=False,
    )
    noise_groups = tuple(
        ActionNoiseGroup(
            name,
            start,
            stop,
            scale=scale,
            zeta_mu=zeta_mu,
            zeta_max=zeta_max,
        )
        for name, start, stop, scale, zeta_mu, zeta_max in NOISE_GROUP_SPECS
    )
    agent = FlashSACTorchBridge(
        env.observation_space,
        env.action_space,
        env.env_info,
        agent_cfg,
        noise_groups=noise_groups,
        restore_rng_state_on_load=False,
    )
    agent.load(str(checkpoint))
    # A deterministic evaluation never consumes cached noise.  Reset it anyway
    # so a checkpoint trained with another num_envs cannot leak stale shape.
    agent.reset_exploration(batch_size=args.num_envs)

    observation, _ = env.reset(randomize_episode_lengths=False)
    initial_truth = _read_physical_truth(env.unwrapped)
    tracker = StrictEpisodeTracker(
        episodes=args.episodes,
        num_envs=args.num_envs,
        device=env.device,
        initial_truth=initial_truth,
    )
    quota_max = int(tracker.quotas.max().item())
    default_max_steps = max(1, env.max_episode_steps * quota_max + quota_max)
    max_vector_steps = args.max_vector_steps or default_max_steps
    vector_steps = 0

    try:
        while not tracker.complete and vector_steps < max_vector_steps:
            vector_steps += 1
            action = agent.sample_actions(
                vector_steps,
                {"next_observation": observation},
                training=False,
            )
            # Slots whose deterministic quota is complete continue simulating
            # independently but cannot contribute additional events.
            action = torch.where(tracker.active.unsqueeze(-1), action, torch.zeros_like(action))
            next_observation, reward, terminated, truncated, info = env.step(action)
            events = validate_terminal_events(info, terminated, truncated)
            transition_truth = physical_truth_from_terminal_info(
                info,
                num_envs=args.num_envs,
                device=env.device,
            )
            # DirectRLEnv has already reset done rows at this point.  This read
            # is used only to initialize their next episode; old-episode maxima
            # above came exclusively from the reset-before terminal payload.
            post_reset_truth = _read_physical_truth(env.unwrapped)
            tracker.step(
                reward=reward.to(dtype=torch.float32),
                terminated=terminated,
                truncated=truncated,
                events=events,
                transition_truth=transition_truth,
                post_reset_truth=post_reset_truth,
            )
            observation = next_observation

        if not tracker.complete:
            raise RuntimeError(
                f"completed {len(tracker.records)}/{args.episodes} episodes after "
                f"--max_vector_steps={max_vector_steps}"
            )
        metrics = build_strict_metrics(
            tracker.records,
            checkpoint=checkpoint,
            architecture=architecture,
            seed=args.seed,
            num_envs=args.num_envs,
            vector_steps=vector_steps,
            max_vector_steps=max_vector_steps,
            episode_length_s=float(env.unwrapped.cfg.episode_length_s),
            max_episode_steps=env.max_episode_steps,
            curriculum_dataset=args.curriculum_dataset,
            curriculum_dataset_sha256=(
                _sha256(args.curriculum_dataset)
                if args.curriculum_dataset is not None
                else None
            ),
            curriculum_boundary=args.curriculum_boundary,
            curriculum_probability=args.curriculum_probability,
            curriculum_joint_noise=args.curriculum_joint_noise,
            use_compile=args.use_compile,
            upstream_commit=FLASH_SAC_COMMIT,
        )
        _atomic_write_json(args.output.resolve(), metrics)
        return metrics
    finally:
        env.close()


def main() -> None:
    args, launcher = _parse_args()
    try:
        metrics = run(args)
        print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
