#!/usr/bin/env python3
"""Fail-closed, simulation-free evidence contract for Candidate 39.

The artifact contains one first-episode row per environment plus a ragged,
step-major table covering only the pre-latch residual window.  It is deliberately
weights-only loadable: metadata is JSON-safe and all observations are CPU
tensors with fixed dtypes and shapes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from option_residual_screen import DESIGN_CONTRACT, DESIGN_SALT, build_option_design


ARTIFACT_KIND = "pick_tool_candidate39_episode_residual_development_v1"
REPORT_KIND = "pick_tool_candidate39_episode_residual_report_v1"
FORMAT_VERSION = 1
COLLECTOR = "collect_candidate39_episode_residual_ab.py"
COLLECTION_CONTRACT = "search_q030_h4_then_coherent_prelatch_hand_residual32_v1"
DELTA_CONTRACT = "clip_z_scale_component_clip_l2_clip_pre_tanh_v1"
DEVELOPMENT_PLAN = "scripts/flashsac/candidate39_coherent_residual_development_plan.json"
KIT_ARGS = "--/app/extensions/fsWatcherEnabled=false"
REQUIRED_BRANCH = "flashsac-pick-tool-curriculum"

OBSERVATION_DIM = 115
ACTION_DIM = 21
ARM_ACTION_DIM = 7
HAND_ACTION_DIM = 14
TOKEN_ACTION_DIM = 9
WINDOW_STEPS = 32
HANDOFF_MIN_SCORE = 0.30
HANDOFF_HOLD_STEPS = 4
RAW_Z_ABS_CAP = 2.0
TOKEN_SCALE = 0.05
DISTAL_SCALE = 0.025
TOKEN_COMPONENT_CAP = 0.10
DISTAL_COMPONENT_CAP = 0.05
PRE_TANH_L2_CAP = 0.20
MAX_EPISODE_ACTIONS = 999
PUBLIC_LATCH_INDEX = 106
PROXIMITY_SLICE = slice(92, 96)
THUMB_PROXIMITY_INDEX = 96

EPISODE_FIELDS = (
    "env_slot", "treatment", "raw_z", "pair_slot", "antithetic_sign",
    "triggered", "trigger_step", "trigger_score", "first_latch_step",
    "latch_released_after_first", "intervention_steps", "episode_length",
    "trajectory_max_force_n", "max_true_clearance_m", "success", "failure",
    "time_out", "dropped", "unsafe_force", "unlatched_clearance_ge_5cm",
    "ever_grasped", "ever_clearance_ge_20cm", "latched_within_window",
)
STEP_FIELDS = (
    "row_env_slot", "row_episode_step", "row_close_age",
    "row_public_latch_before", "row_residual_active", "row_observation",
    "row_base_mean_hand", "row_applied_delta", "row_baseline_action",
    "row_candidate_action", "row_executed_action", "row_transition_grasped",
    "row_transition_true_clearance_m", "row_grasp_quality", "row_hold_quality",
    "row_max_force_n",
)

_EPISODE_BOOL = {
    "treatment", "triggered", "latch_released_after_first", "success", "failure",
    "time_out", "dropped", "unsafe_force", "unlatched_clearance_ge_5cm",
    "ever_grasped", "ever_clearance_ge_20cm", "latched_within_window",
}
_EPISODE_LONG = {
    "env_slot", "pair_slot", "trigger_step", "first_latch_step",
    "intervention_steps", "episode_length",
}
_STEP_BOOL = {"row_public_latch_before", "row_residual_active", "row_transition_grasped"}
_STEP_LONG = {"row_env_slot", "row_episode_step", "row_close_age"}

GIT_FIELDS = ("commit", "branch", "source_files_dirty", "flashsac_commit", "flashsac_dirty")
RUNTIME_PACKAGE_FIELDS = ("isaaclab", "isaaclab_tasks", "isaaclab_assets", "numpy", "gymnasium")
RUNTIME_FIELDS = (
    "python", "torch", "cuda", "cudnn", "cuda_device_index", "cuda_device_name",
    "cuda_device_capability", "isaac_sim", "packages", "nvidia_smi_inventory",
    "platform", "seed",
)

METADATA_FIELDS = (
    "kind", "version", "development_only", "first_episode_only", "collector",
    "collection_contract", "delta_contract", "design_contract", "assignment_salt",
    "design_salt", "handoff_min_score", "handoff_hold_steps", "window_steps",
    "token_scale", "distal_scale", "raw_z_abs_cap", "token_component_cap",
    "distal_component_cap", "pre_tanh_l2_cap", "observation_dim", "action_dim",
    "max_episode_actions", "kit_args", "seed", "replicate", "num_envs",
    "v6_checkpoint", "search_checkpoint", "development_plan",
    "development_plan_sha256", "v6_actor_sha256", "v6_task_contract_sha256",
    "v6_bridge_state_sha256", "frozen_lift_actor_sha256",
    "frozen_lift_semantic_sha256", "frozen_lift_source_actor_sha256",
    "search_checkpoint_sha256", "source_manifest_sha256",
    "runtime_asset_manifest_sha256", "source_sha256", "runtime_asset_sha256",
    "flashsac_upstream_commit", "flashsac_fork_commit", "git", "runtime",
)

REPORT_FIELDS = (
    "kind", "status", "collector", "seed", "replicate", "num_envs",
    "vector_steps", "v6_actor_sha256", "search_checkpoint_sha256", "summary",
)
PUBLISHED_REPORT_FIELDS = (*REPORT_FIELDS, "artifact_sha256", "artifact_output")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(value: Any, name: str = "value") -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} has a non-string key")
            result[key] = _strict_json(item, f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_strict_json(item, f"{name}[]") for item in value]
    raise TypeError(f"{name} contains unsupported {type(value).__name__}")


def manifest_sha256(manifest: Mapping[str, str]) -> str:
    """Hash a path->SHA manifest independently of insertion order."""
    checked = _sha_manifest(manifest, "manifest")
    encoded = json.dumps(checked, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return value


def _sha_manifest(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty plain dictionary")
    result: dict[str, str] = {}
    for path, digest in value.items():
        if not isinstance(path, str) or not path:
            raise ValueError(f"{name} has an empty/non-string path")
        result[path] = _sha(digest, f"{name}[{path!r}]")
    return result


def _plan_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    return sha256_file(root / DEVELOPMENT_PLAN)


REQUIRED_METADATA = {
    "kind": ARTIFACT_KIND,
    "version": FORMAT_VERSION,
    "development_only": True,
    "first_episode_only": True,
    "collector": COLLECTOR,
    "collection_contract": COLLECTION_CONTRACT,
    "delta_contract": DELTA_CONTRACT,
    "design_contract": DESIGN_CONTRACT,
    "assignment_salt": DESIGN_SALT,
    "design_salt": DESIGN_SALT,
    "handoff_min_score": HANDOFF_MIN_SCORE,
    "handoff_hold_steps": HANDOFF_HOLD_STEPS,
    "window_steps": WINDOW_STEPS,
    "token_scale": TOKEN_SCALE,
    "distal_scale": DISTAL_SCALE,
    "raw_z_abs_cap": RAW_Z_ABS_CAP,
    "token_component_cap": TOKEN_COMPONENT_CAP,
    "distal_component_cap": DISTAL_COMPONENT_CAP,
    "pre_tanh_l2_cap": PRE_TANH_L2_CAP,
    "observation_dim": OBSERVATION_DIM,
    "action_dim": ACTION_DIM,
    "max_episode_actions": MAX_EPISODE_ACTIONS,
    "kit_args": KIT_ARGS,
    "development_plan": DEVELOPMENT_PLAN,
    "development_plan_sha256": _plan_digest(),
}


def _metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(METADATA_FIELDS):
        raise ValueError(f"metadata fields must be exactly {METADATA_FIELDS}")
    result = _strict_json(dict(value), "metadata")
    for key, expected in REQUIRED_METADATA.items():
        if result[key] != expected or type(result[key]) is not type(expected):
            raise ValueError(f"metadata.{key} differs from the fixed contract")
    seed, num_envs, replicate = result["seed"], result["num_envs"], result["replicate"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("metadata.seed must be a non-negative integer")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 2 or num_envs % 2:
        raise ValueError("metadata.num_envs must be positive and even")
    if replicate not in {"a", "b"}:
        raise ValueError("metadata.replicate must be a or b")
    for name in ("v6_checkpoint", "search_checkpoint"):
        if not isinstance(result[name], str) or not result[name]:
            raise ValueError(f"metadata.{name} must be non-empty")
    for name in (
        "development_plan_sha256", "v6_actor_sha256", "v6_task_contract_sha256",
        "v6_bridge_state_sha256", "frozen_lift_actor_sha256",
        "frozen_lift_semantic_sha256", "frozen_lift_source_actor_sha256",
        "search_checkpoint_sha256", "source_manifest_sha256",
        "runtime_asset_manifest_sha256",
    ):
        _sha(result[name], f"metadata.{name}")
    source = _sha_manifest(result["source_sha256"], "metadata.source_sha256")
    assets = _sha_manifest(result["runtime_asset_sha256"], "metadata.runtime_asset_sha256")
    if manifest_sha256(source) != result["source_manifest_sha256"]:
        raise ValueError("source manifest digest disagrees with source_sha256")
    if manifest_sha256(assets) != result["runtime_asset_manifest_sha256"]:
        raise ValueError("runtime asset manifest digest disagrees with runtime_asset_sha256")
    _git_sha(result["flashsac_upstream_commit"], "metadata.flashsac_upstream_commit")
    _git_sha(result["flashsac_fork_commit"], "metadata.flashsac_fork_commit")
    git = result["git"]
    if not isinstance(git, dict) or set(git) != set(GIT_FIELDS):
        raise ValueError("metadata.git has an unexpected schema")
    _git_sha(git["commit"], "metadata.git.commit")
    _git_sha(git["flashsac_commit"], "metadata.git.flashsac_commit")
    if git["branch"] != REQUIRED_BRANCH:
        raise ValueError(f"metadata.git.branch must be {REQUIRED_BRANCH!r}")
    if git["flashsac_commit"] != result["flashsac_fork_commit"]:
        raise ValueError("metadata.git.flashsac_commit disagrees with FlashSAC fork commit")
    if type(git["source_files_dirty"]) is not bool or type(git["flashsac_dirty"]) is not bool:
        raise TypeError("metadata git dirty flags must be bool")
    if git["source_files_dirty"] or git["flashsac_dirty"]:
        raise ValueError("Candidate 39 collection requires clean committed sources")
    runtime = result["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_FIELDS):
        raise ValueError("metadata.runtime has an unexpected schema")
    if runtime["seed"] != seed:
        raise ValueError("metadata.runtime.seed disagrees with seed")
    packages = runtime["packages"]
    if not isinstance(packages, dict) or set(packages) != set(RUNTIME_PACKAGE_FIELDS):
        raise ValueError("metadata.runtime.packages has an unexpected schema")
    return result


def _tensor(name: str, value: Any, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype or value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU {dtype}")
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return value.contiguous()


def _episode_table(value: Any, n: int) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(EPISODE_FIELDS):
        raise ValueError(f"episode fields must be exactly {EPISODE_FIELDS}")
    result: dict[str, torch.Tensor] = {}
    for name in EPISODE_FIELDS:
        shape = (n, HAND_ACTION_DIM) if name == "raw_z" else (n,)
        dtype = torch.bool if name in _EPISODE_BOOL else torch.long if name in _EPISODE_LONG else torch.int8 if name == "antithetic_sign" else torch.float32
        result[name] = _tensor(f"episodes.{name}", value[name], shape, dtype)
    return result


def _step_table(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(STEP_FIELDS):
        raise ValueError(f"step fields must be exactly {STEP_FIELDS}")
    first = value.get("row_env_slot")
    if not isinstance(first, torch.Tensor) or first.ndim != 1:
        raise ValueError("steps.row_env_slot must be rank one")
    rows = int(first.shape[0])
    suffix = {
        "row_observation": (OBSERVATION_DIM,), "row_base_mean_hand": (HAND_ACTION_DIM,),
        "row_applied_delta": (HAND_ACTION_DIM,), "row_baseline_action": (ACTION_DIM,),
        "row_candidate_action": (ACTION_DIM,), "row_executed_action": (ACTION_DIM,),
    }
    result: dict[str, torch.Tensor] = {}
    for name in STEP_FIELDS:
        dtype = torch.bool if name in _STEP_BOOL else torch.long if name in _STEP_LONG else torch.float32
        result[name] = _tensor(f"steps.{name}", value[name], (rows, *suffix.get(name, ())), dtype)
    return result


def _expected_delta(raw_z: torch.Tensor) -> torch.Tensor:
    scale = raw_z.new_tensor([TOKEN_SCALE] * TOKEN_ACTION_DIM + [DISTAL_SCALE] * (HAND_ACTION_DIM - TOKEN_ACTION_DIM))
    cap = raw_z.new_tensor([TOKEN_COMPONENT_CAP] * TOKEN_ACTION_DIM + [DISTAL_COMPONENT_CAP] * (HAND_ACTION_DIM - TOKEN_ACTION_DIM))
    delta = raw_z.clamp(-RAW_Z_ABS_CAP, RAW_Z_ABS_CAP) * scale
    delta = torch.maximum(torch.minimum(delta, cap), -cap)
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    factor = torch.where(norm > PRE_TANH_L2_CAP, PRE_TANH_L2_CAP / norm.clamp_min(torch.finfo(delta.dtype).tiny), torch.ones_like(norm))
    return delta * factor


def _validate_semantics(meta: Mapping[str, Any], ep: Mapping[str, torch.Tensor], st: Mapping[str, torch.Tensor]) -> None:
    n = meta["num_envs"]
    if not torch.equal(ep["env_slot"], torch.arange(n)):
        raise ValueError("episode env_slot must be canonical arange")
    design = build_option_design(seed=meta["seed"], num_envs=n, replicate=meta["replicate"], dtype=torch.float32, device="cpu")
    for name in ("treatment", "raw_z", "pair_slot", "antithetic_sign"):
        if not torch.equal(ep[name], getattr(design, name)):
            raise ValueError(f"episode {name} differs from build_option_design")
    lengths = ep["episode_length"]
    if bool(((lengths < 1) | (lengths > MAX_EPISODE_ACTIONS)).any()):
        raise ValueError("episode_length is outside [1,999]")
    success, failure, timeout = ep["success"], ep["failure"], ep["time_out"]
    if not bool(((success.long() + failure.long() + timeout.long()) == 1).all()):
        raise ValueError("success/failure/time_out must partition episodes")
    if not torch.equal(failure, ep["dropped"] | ep["unsafe_force"] | ep["unlatched_clearance_ge_5cm"]):
        raise ValueError("failure differs from authored failure sources")
    if bool((timeout & (lengths != MAX_EPISODE_ACTIONS)).any()):
        raise ValueError("time_out must occur at the authored horizon")
    clearance = ep["max_true_clearance_m"]
    if not torch.equal(ep["ever_clearance_ge_20cm"], clearance >= 0.20):
        raise ValueError("20 cm truth disagrees with maximum true clearance")
    bad_success = success & (~ep["ever_grasped"] | ~ep["ever_clearance_ge_20cm"] | ep["dropped"] | ep["unsafe_force"] | ep["unlatched_clearance_ge_5cm"])
    if bool(bad_success.any()):
        raise ValueError("success violates physical/safety truth")
    if bool((ep["unlatched_clearance_ge_5cm"] & (clearance < 0.05)).any()):
        raise ValueError("unlatched 5 cm truth exceeds maximum clearance")
    if bool((ep["trajectory_max_force_n"] < 0).any()):
        raise ValueError("trajectory force cannot be negative")

    triggered, trigger_step, trigger_score = ep["triggered"], ep["trigger_step"], ep["trigger_score"]
    if bool((trigger_step[~triggered] != -1).any()) or bool((trigger_score[~triggered] != 0).any()):
        raise ValueError("untriggered episode has trigger data")
    if bool(((trigger_step[triggered] < HANDOFF_HOLD_STEPS - 1) | (trigger_step[triggered] >= lengths[triggered])).any()):
        raise ValueError("trigger step violates q.30/h4 or episode bounds")
    if bool(((trigger_score[triggered] < HANDOFF_MIN_SCORE) | (trigger_score[triggered] > 1)).any()):
        raise ValueError("trigger score violates q.30")
    first = ep["first_latch_step"]
    if bool(((first < -1) | (first >= lengths)).any()):
        raise ValueError("first_latch_step is outside the episode")
    if bool((ep["latch_released_after_first"] & (first < 0)).any()):
        raise ValueError("latch cannot release before a first latch")
    expected_window_latch = triggered & (first >= trigger_step) & ((first - trigger_step) < WINDOW_STEPS)
    if not torch.equal(ep["latched_within_window"], expected_window_latch):
        raise ValueError("latched_within_window disagrees with trigger/latch clocks")

    env, step, age = st["row_env_slot"], st["row_episode_step"], st["row_close_age"]
    rows = env.numel()
    # An entirely empty table is legal when every triggered slot had already
    # latched and released during SEARCH.  Such slots are randomized but have
    # zero intervention exposure and are represented by first_latch<trigger.
    if bool(((env < 0) | (env >= n)).any()):
        raise ValueError("ragged step table has invalid env slots")
    if bool(((step < 0) | (step >= lengths[env])).any()):
        raise ValueError("row episode step is outside its episode")
    order_key = step * n + env
    if rows > 1 and not bool((order_key[1:] > order_key[:-1]).all()):
        raise ValueError("rows must be unique canonical step-major/env-major order")
    counts = torch.bincount(env, minlength=n)
    pretrigger_latch = triggered & (first >= 0) & (first < trigger_step)
    should_have_rows = triggered & ~pretrigger_latch
    if (
        bool((counts[~triggered] != 0).any())
        or bool((counts[pretrigger_latch] != 0).any())
        or bool((counts[should_have_rows] < 1).any())
        or bool((counts > WINDOW_STEPS).any())
    ):
        raise ValueError("ragged row counts disagree with triggered/window contract")
    if not torch.equal(ep["intervention_steps"], torch.bincount(env[st["row_residual_active"]], minlength=n)):
        raise ValueError("intervention_steps disagrees with active rows")
    for slot in range(n):
        mask = env == slot
        count = int(mask.sum())
        if not count:
            continue
        expected_age = torch.arange(count)
        if not torch.equal(age[mask], expected_age) or not torch.equal(step[mask], trigger_step[slot] + expected_age):
            raise ValueError("per-env ragged CLOSE ages are not contiguous from trigger")
        stop = min(WINDOW_STEPS, int(lengths[slot] - trigger_step[slot]))
        latch = int(first[slot])
        if latch >= int(trigger_step[slot]) and latch < int(trigger_step[slot]) + WINDOW_STEPS:
            stop = min(stop, latch - int(trigger_step[slot]) + 1)
        if count != stop:
            raise ValueError("ragged row count disagrees with window/latch/terminal stop")
    if bool(st["row_public_latch_before"].any()):
        raise ValueError("residual-window rows must be pre-latch")
    expected_active = ep["treatment"][env]
    if not torch.equal(st["row_residual_active"], expected_active):
        raise ValueError("residual_active must exactly select treatment rows")

    obs = st["row_observation"]
    if not torch.equal(obs[:, PUBLIC_LATCH_INDEX], torch.zeros(rows)):
        raise ValueError("observation latch disagrees with pre-latch rows")
    nonthumb_second = torch.topk(obs[:, PROXIMITY_SLICE], 2, dim=-1).values[:, 1]
    q = torch.minimum(obs[:, THUMB_PROXIMITY_INDEX], nonthumb_second)
    first_rows = age == 0
    if not torch.allclose(q[first_rows], trigger_score[env[first_rows]], rtol=0, atol=1e-6):
        raise ValueError("trigger score disagrees with trigger observation")

    base, candidate, executed = st["row_baseline_action"], st["row_candidate_action"], st["row_executed_action"]
    if bool((base.abs() > 1).any()) or bool((candidate.abs() > 1).any()) or bool((executed.abs() > 1).any()):
        raise ValueError("action escaped [-1,1]")
    if not torch.equal(candidate[:, :ARM_ACTION_DIM], base[:, :ARM_ACTION_DIM]):
        raise ValueError("candidate changed arm7")
    if not torch.equal(base[:, :ARM_ACTION_DIM], torch.zeros_like(base[:, :ARM_ACTION_DIM])):
        raise ValueError("pre-latch CLOSE baseline arm must be exact zero")
    if not torch.allclose(torch.tanh(st["row_base_mean_hand"]), base[:, ARM_ACTION_DIM:], rtol=0, atol=1e-6):
        raise ValueError("raw actor mean does not reconstruct baseline hand")
    expected_delta = _expected_delta(ep["raw_z"][env])
    expected_delta = torch.where(expected_active[:, None], expected_delta, torch.zeros_like(expected_delta))
    delta = st["row_applied_delta"]
    if not torch.allclose(delta, expected_delta, rtol=0, atol=1e-6):
        raise ValueError("applied delta disagrees with caps/scales/L2 saturation")
    inactive = ~expected_active
    if bool(inactive.any()) and (not torch.equal(candidate[inactive], base[inactive]) or not torch.equal(delta[inactive], torch.zeros_like(delta[inactive]))):
        raise ValueError("control/inactive rows are not bit-exact baseline")
    margin = max(1e-6, 4 * torch.finfo(torch.float32).eps)
    hand = base[:, ARM_ACTION_DIM:]
    overlaid = torch.tanh(torch.atanh(hand.clamp(-1 + margin, 1 - margin)) + delta)
    expected_hand = torch.where(delta != 0, overlaid, hand)
    if not torch.allclose(candidate[:, ARM_ACTION_DIM:], expected_hand, rtol=0, atol=2e-6):
        raise ValueError("candidate hand disagrees with pre-tanh overlay")
    expected_exec = torch.where(expected_active[:, None], candidate, base)
    if not torch.equal(executed, expected_exec):
        raise ValueError("executed action does not select treatment candidate/control baseline")
    if bool((delta[:, :TOKEN_ACTION_DIM].abs() > TOKEN_COMPONENT_CAP + 1e-6).any()) or bool((delta[:, TOKEN_ACTION_DIM:].abs() > DISTAL_COMPONENT_CAP + 1e-6).any()) or bool((torch.linalg.vector_norm(delta, dim=-1) > PRE_TANH_L2_CAP + 1e-6).any()):
        raise ValueError("applied residual exceeds its registered budget")
    for name in ("row_grasp_quality", "row_hold_quality"):
        if bool(((st[name] < 0) | (st[name] > 1)).any()):
            raise ValueError(f"{name} escaped [0,1]")
    if bool((st["row_max_force_n"] < 0).any()):
        raise ValueError("row force cannot be negative")
    if bool((st["row_max_force_n"] > ep["trajectory_max_force_n"][env] + 1e-5).any()) or bool((st["row_transition_true_clearance_m"] > clearance[env] + 1e-6).any()):
        raise ValueError("row physical telemetry exceeds episode maximum")
    if bool((st["row_transition_grasped"] & ~ep["ever_grasped"][env]).any()):
        raise ValueError("row grasp truth disagrees with episode ever_grasped")
    # ``first_latch_step`` is authoritative on the public transition-observation
    # clock.  PickTool terminal telemetry is sampled in ``_get_dones`` while the
    # public latch is updated later in ``_get_rewards``; requiring the former to
    # be true on this same row would incorrectly reject a legitimate one-action
    # clock skew.


def validate_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping) or set(artifact) != {"metadata", "episodes", "steps"}:
        raise ValueError("artifact must contain exactly metadata/episodes/steps")
    meta = _metadata(artifact["metadata"])
    episodes = _episode_table(artifact["episodes"], meta["num_envs"])
    steps = _step_table(artifact["steps"])
    result = {"metadata": meta, "episodes": episodes, "steps": steps}
    _validate_semantics(meta, episodes, steps)
    return result


def build_artifact(metadata: Mapping[str, Any], episodes: Mapping[str, torch.Tensor], steps: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    payload = {
        "metadata": dict(metadata),
        "episodes": {name: episodes[name].detach().cpu().clone() for name in EPISODE_FIELDS},
        "steps": {name: steps[name].detach().cpu().clone() for name in STEP_FIELDS},
    }
    return validate_artifact(payload)


def _arm_summary(ep: Mapping[str, torch.Tensor], mask: torch.Tensor) -> dict[str, int]:
    return {"episodes": int(mask.sum()), **{name: int((mask & ep[name]).sum()) for name in ("triggered", "latched_within_window", "ever_grasped", "ever_clearance_ge_20cm", "success", "failure", "time_out", "dropped", "unsafe_force", "unlatched_clearance_ge_5cm")}}


def summarize_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_artifact(artifact)
    ep, st = value["episodes"], value["steps"]
    treatment = ep["treatment"]
    delta_norm = torch.linalg.vector_norm(st["row_applied_delta"], dim=-1)
    new098 = (st["row_candidate_action"].abs() >= .98) & (st["row_baseline_action"].abs() < .98)
    has_rows = bool(st["row_env_slot"].numel())
    return {
        "episodes": int(treatment.numel()),
        "step_rows": int(st["row_env_slot"].numel()),
        "control": _arm_summary(ep, ~treatment),
        "treatment": _arm_summary(ep, treatment),
        "action_audit": {
            "applied_delta_abs_max": float(st["row_applied_delta"].abs().max()) if has_rows else 0.0,
            "applied_delta_l2_max": float(delta_norm.max()) if has_rows else 0.0,
            "candidate_action_abs_max": float(st["row_candidate_action"].abs().max()) if has_rows else 0.0,
            "new_abs_ge_098_elements": int(new098.sum()),
            "episodes_with_new_abs_ge_098": int(torch.unique(st["row_env_slot"][new098.any(dim=-1)]).numel()),
            "any_new_abs_ge_0999": bool((((st["row_candidate_action"].abs() >= .999) & (st["row_baseline_action"].abs() < .999))).any()),
            "budget_violations": 0,
        },
    }


def validate_report(report: Mapping[str, Any], artifact: Mapping[str, Any], *, published: bool = False) -> dict[str, Any]:
    value = validate_artifact(artifact)
    expected_fields = PUBLISHED_REPORT_FIELDS if published else REPORT_FIELDS
    if not isinstance(report, dict) or set(report) != set(expected_fields):
        raise ValueError(f"report fields must be exactly {expected_fields}")
    checked = _strict_json(report, "report")
    meta, ep = value["metadata"], value["episodes"]
    fixed = {
        "kind": REPORT_KIND, "status": "complete", "collector": COLLECTOR,
        "seed": meta["seed"], "replicate": meta["replicate"], "num_envs": meta["num_envs"],
        "vector_steps": int(ep["episode_length"].max()),
        "v6_actor_sha256": meta["v6_actor_sha256"],
        "search_checkpoint_sha256": meta["search_checkpoint_sha256"],
        "summary": summarize_artifact(value),
    }
    for key, expected in fixed.items():
        if checked[key] != expected:
            raise ValueError(f"report.{key} differs from artifact")
    if published:
        _sha(checked["artifact_sha256"], "report.artifact_sha256")
        if not isinstance(checked["artifact_output"], str) or not checked["artifact_output"]:
            raise ValueError("report.artifact_output must be non-empty")
    return checked


def validate_complementary_artifacts(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left, right = validate_artifact(a), validate_artifact(b)
    ma, mb = left["metadata"], right["metadata"]
    if {ma["replicate"], mb["replicate"]} != {"a", "b"}:
        raise ValueError("complementary artifacts must contain replicates a and b")
    for key in METADATA_FIELDS:
        if key != "replicate" and ma[key] != mb[key]:
            raise ValueError(f"cross-artifact metadata differs at {key}")
    ea, eb = left["episodes"], right["episodes"]
    if not torch.equal(ea["treatment"], ~eb["treatment"]):
        raise ValueError("A/B assignments are not exact complements")
    for key in ("raw_z", "pair_slot", "antithetic_sign"):
        if not torch.equal(ea[key], eb[key]):
            raise ValueError(f"A/B {key} is not bit-exact")
    return left, right


def _owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _unlink_if_ours(output: Path, temporary: Path) -> None:
    try:
        if _owned(output) and _owned(temporary) and os.path.samefile(output, temporary):
            output.unlink()
    except FileNotFoundError:
        pass


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def publish_artifact_and_report_no_clobber(artifact: Mapping[str, Any], report: Mapping[str, Any], artifact_output: Path, report_output: Path) -> str:
    value = validate_artifact(artifact)
    checked_report = validate_report(report, value)
    artifact_output, report_output = Path(os.path.abspath(os.fspath(artifact_output))), Path(os.path.abspath(os.fspath(report_output)))
    if artifact_output == report_output:
        raise ValueError("artifact and report paths must differ")
    for output in (artifact_output, report_output):
        if _owned(output):
            raise FileExistsError(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[Path] = []
    links: dict[Path, Path] = {}
    try:
        fd, name = tempfile.mkstemp(prefix=f".{artifact_output.name}.tmp-", dir=artifact_output.parent); os.close(fd)
        artifact_tmp = Path(name); temporaries.append(artifact_tmp); links[artifact_output] = artifact_tmp
        with artifact_tmp.open("wb") as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        digest = sha256_file(artifact_tmp)
        final_report = {**checked_report, "artifact_sha256": digest, "artifact_output": str(artifact_output)}
        validate_report(final_report, value, published=True)
        fd, name = tempfile.mkstemp(prefix=f".{report_output.name}.tmp-", dir=report_output.parent)
        report_tmp = Path(name); temporaries.append(report_tmp); links[report_output] = report_tmp
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(final_report)); stream.flush(); os.fsync(stream.fileno())
        os.link(artifact_tmp, artifact_output); os.link(report_tmp, report_output)
        for directory in {artifact_output.parent, report_output.parent}:
            dfd = os.open(directory, os.O_RDONLY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        return digest
    except BaseException:
        for output, temporary in reversed(tuple(links.items())):
            _unlink_if_ours(output, temporary)
        raise
    finally:
        for temporary in temporaries:
            if _owned(temporary): temporary.unlink()


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    serialized = _json_bytes(payload)
    output = Path(os.path.abspath(os.fspath(output)))
    if _owned(output): raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(serialized); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, output)
        dfd = os.open(output.parent, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except BaseException:
        _unlink_if_ours(output, temporary)
        raise
    finally:
        if _owned(temporary): temporary.unlink()


__all__ = [
    "ARTIFACT_KIND", "REPORT_KIND", "FORMAT_VERSION", "COLLECTOR",
    "COLLECTION_CONTRACT", "REQUIRED_METADATA", "METADATA_FIELDS",
    "EPISODE_FIELDS", "STEP_FIELDS", "build_artifact", "validate_artifact",
    "summarize_artifact", "validate_report", "validate_complementary_artifacts",
    "publish_artifact_and_report_no_clobber", "publish_json_no_clobber",
    "sha256_file", "manifest_sha256",
]
