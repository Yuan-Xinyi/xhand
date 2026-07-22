#!/usr/bin/env python3
"""Simulation-free contract for successful noisy-V6 CLOSE self-imitation.

The live collector deliberately keeps two same-process cohorts.  Half of the
environment slots execute deterministic V6 CLOSE, while the complementary
half execute the same actor with hand-only exploration before the public
grasp latch.  The dataset contains only executed, pre-latch CLOSE actions from
strictly successful and safe first episodes.  Full first-episode outcome
tensors remain in the artifact so success conditioning cannot hide the yield
or safety of either cohort.
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


COLLECTOR = "pick_tool_v6_success_self_imitation_v1"
REPORT_KIND = "pick_tool_v6_success_self_imitation_report_v1"
FORMAT_VERSION = 1
COHORT_ASSIGNMENT = "balanced_sha256_slot_v1"
COHORT_ASSIGNMENT_SALT = "pick_tool_v6_success_self_imitation_20260722_v1"
HANDOFF_MIN_SCORE = 0.30
HANDOFF_HOLD_STEPS = 4
EXPLORATORY_HAND_NOISE_SCALE = 0.25
OBSERVATION_DIM = 115
ACTION_DIM = 21
ARM_ACTION_DIM = 7
PUBLIC_LATCH_INDEX = 106
MAX_EPISODE_ACTIONS = 999
KIT_ARGS = "--/app/extensions/fsWatcherEnabled=false"
ACTION_LAYOUT = "arm_delta7|crossdex_token9|distal_residual5"
OBSERVATION_LAYOUT = "legacy_prefix87|distal_action5|grasp_transport23"
PHASE_NAMES = ["approach", "close", "micro", "lift", "settle"]

SCORE_BUCKETS = (
    ("0.30_to_0.34", 0.30, 0.34),
    ("0.34_to_0.40", 0.34, 0.40),
    ("ge_0.40", 0.40, None),
)

EPISODE_BOOL_NAMES = (
    "episode_success",
    "episode_native_success",
    "episode_dropped",
    "episode_unsafe_force",
    "episode_unlatched_clearance_ge_5cm",
    "episode_ever_grasped",
    "episode_triggered",
    "episode_ever_latched",
    "episode_latch_released_after_first",
)
EPISODE_LONG_NAMES = (
    "episode_source_env_id",
    "episode_cohort",
    "episode_trigger_step",
    "episode_first_latch_step",
    "episode_terminal_step",
)
EPISODE_FLOAT_NAMES = (
    "episode_hand_noise_scale",
    "episode_trigger_score",
    "episode_max_true_clearance_m",
    "episode_trajectory_max_force",
)
TRIAL_BOOL_NAMES = (
    "trial_triggered",
    "trial_ever_latched",
    "trial_latch_released_after_first",
    "trial_native_success",
    "trial_failure",
    "trial_time_out",
    "trial_dropped",
    "trial_unsafe_force",
    "trial_unlatched_clearance_ge_5cm",
    "trial_ever_grasped",
    "trial_ever_clearance_ge_20cm",
    "trial_retained",
)
TRIAL_LONG_NAMES = (
    "trial_env_slot",
    "trial_cohort",
    "trial_episode_length",
    "trial_trigger_step",
    "trial_first_latch_step",
    "trial_terminal_step",
)
TRIAL_FLOAT_NAMES = (
    "trial_hand_noise_scale",
    "trial_trigger_score",
    "trial_max_true_clearance_m",
    "trial_trajectory_max_force",
)
DATASET_KEYS = (
    "obs",
    "action",
    "phase",
    "episode_id",
    "step",
    "source_step",
    "episode_offsets",
    *EPISODE_BOOL_NAMES,
    *EPISODE_LONG_NAMES,
    *EPISODE_FLOAT_NAMES,
    *TRIAL_BOOL_NAMES,
    *TRIAL_LONG_NAMES,
    *TRIAL_FLOAT_NAMES,
    "meta",
)
SHA_METADATA_NAMES = (
    "search_checkpoint_sha256",
    "v6_actor_sha256",
    "v6_task_contract_sha256",
    "v6_bridge_state_sha256",
    "frozen_lift_actor_sha256",
    "frozen_lift_semantic_sha256",
    "frozen_lift_source_actor_sha256",
    "source_manifest_sha256",
    "runtime_asset_manifest_sha256",
)
REQUIRED_METADATA = {
    "format_version": FORMAT_VERSION,
    "observation_dim": OBSERVATION_DIM,
    "action_dim": ACTION_DIM,
    "observation_layout": OBSERVATION_LAYOUT,
    "action_layout": ACTION_LAYOUT,
    "phase_names": PHASE_NAMES,
    "collector": COLLECTOR,
    "task_mode": "full_task",
    "observation_contract": "pick_tool_markov115_v1",
    "dataset_phase": "close",
    "close_arm_mode": "zero",
    "action_projection": "identity_v1",
    "action_semantics": "executed_v6_action_with_latch_conditioned_exploration_v1",
    "trajectory_acceptance": "initial_episode_triggered_true_success_safe_first_latch_persistent_v1",
    "clearance_authority": "true_mesh_convex_hull_min_z_minus_table_v1",
    "search_handoff_contract": "pick_tool_public_online_handoff_v1",
    "trigger_action_semantics": "option_controls_the_trigger_frame_and_remains_sticky_until_reset",
    "policy_router": "public_latch_frozen_actor_v1",
    "latch_transition_clock": "post_action_transition_observation_public_latch_v1",
    "cohort_assignment": COHORT_ASSIGNMENT,
    "cohort_names": ["deterministic", "exploratory"],
    "handoff_min_score": HANDOFF_MIN_SCORE,
    "handoff_hold_steps": HANDOFF_HOLD_STEPS,
    "exploratory_hand_noise_scale": EXPLORATORY_HAND_NOISE_SCALE,
    "unlatched_arm_noise_scale": 0.0,
    "latched_arm_noise_scale": 0.0,
    "latched_hand_noise_scale": 0.0,
    "terminal_observation": "not_saved_auto_reset_excluded_v1",
    "stored_row_window": "trigger_frame_through_first_latch_transition_v1",
    "initial_episode_only": True,
    "executed_action_max_error": 0.0,
    "max_episode_actions": MAX_EPISODE_ACTIONS,
    "kit_args": KIT_ARGS,
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_git_sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return value


def _require_json_safe(value: Any, *, name: str) -> None:
    if value is None or isinstance(value, (str, bool)) or _is_int(value):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_safe(item, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} contains a non-string key")
            _require_json_safe(item, name=f"{name}.{key}")
        return
    raise TypeError(f"{name} contains non-JSON-safe {type(value).__name__}")


def manifest_sha256(mapping: Mapping[str, str]) -> str:
    """Hash a sorted path-to-content-hash manifest without path ambiguity."""

    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("manifest must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for path, digest in mapping.items():
        if not isinstance(path, str) or not path:
            raise ValueError("manifest paths must be non-empty strings")
        normalized[path] = _require_sha256(digest, name=f"manifest[{path!r}]")
    serialized = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def exploratory_cohort_mask(
    *, seed: int, num_envs: int, replicate: str
) -> torch.Tensor:
    """Return an exact balanced assignment; replicate b complements a."""

    if not _is_int(seed):
        raise TypeError("seed must be an integer")
    if not _is_int(num_envs) or num_envs < 2 or num_envs % 2:
        raise ValueError("num_envs must be an even integer of at least two")
    if replicate not in {"a", "b"}:
        raise ValueError("replicate must be 'a' or 'b'")
    ranked = sorted(
        range(num_envs),
        key=lambda env_id: (
            hashlib.sha256(
                f"{COHORT_ASSIGNMENT_SALT}\0{seed}\0{env_id}".encode("utf-8")
            ).digest(),
            env_id,
        ),
    )
    assignment_a = torch.zeros(num_envs, dtype=torch.bool)
    assignment_a[torch.tensor(ranked[: num_envs // 2], dtype=torch.long)] = True
    return assignment_a if replicate == "a" else ~assignment_a


def build_cohort_noise_scale(
    observation: torch.Tensor,
    exploratory: torch.Tensor,
    *,
    exploratory_hand_scale: float = EXPLORATORY_HAND_NOISE_SCALE,
) -> torch.Tensor:
    """Build arm-zero, pre-latch hand-only exploration multipliers."""

    if (
        not isinstance(observation, torch.Tensor)
        or observation.ndim != 2
        or observation.shape[1] != OBSERVATION_DIM
        or not observation.dtype.is_floating_point
    ):
        raise ValueError(f"observation must be floating [N,{OBSERVATION_DIM}]")
    batch = observation.shape[0]
    if (
        not isinstance(exploratory, torch.Tensor)
        or exploratory.shape != (batch,)
        or exploratory.dtype != torch.bool
        or exploratory.device != observation.device
    ):
        raise ValueError("exploratory must be a co-located bool[N] tensor")
    if (
        not isinstance(exploratory_hand_scale, (int, float))
        or isinstance(exploratory_hand_scale, bool)
        or not math.isfinite(float(exploratory_hand_scale))
        or float(exploratory_hand_scale) < 0.0
    ):
        raise ValueError("exploratory_hand_scale must be finite and non-negative")
    latch = observation[:, PUBLIC_LATCH_INDEX]
    if not bool(((latch == 0.0) | (latch == 1.0)).all()):
        raise ValueError("public latch must be exactly binary")
    result = torch.zeros(
        (batch, ACTION_DIM), dtype=torch.float32, device=observation.device
    )
    active = exploratory & (latch == 0.0)
    result[:, ARM_ACTION_DIM:] = active.to(torch.float32).unsqueeze(-1) * float(
        exploratory_hand_scale
    )
    return result


def select_executed_action(
    *,
    search_action: torch.Tensor,
    routed_v6_action: torch.Tensor,
    option_active: torch.Tensor,
) -> torch.Tensor:
    """Select frozen SEARCH until q=.30/h4, then checkpoint-native V6 routing."""

    if (
        not isinstance(search_action, torch.Tensor)
        or search_action.ndim != 2
        or search_action.shape[1] != ACTION_DIM
        or not search_action.dtype.is_floating_point
    ):
        raise ValueError(f"search_action must be floating [N,{ACTION_DIM}]")
    if (
        not isinstance(routed_v6_action, torch.Tensor)
        or routed_v6_action.shape != search_action.shape
        or routed_v6_action.dtype != search_action.dtype
        or routed_v6_action.device != search_action.device
    ):
        raise ValueError("routed_v6_action must exactly match search_action")
    if (
        not isinstance(option_active, torch.Tensor)
        or option_active.shape != (search_action.shape[0],)
        or option_active.dtype != torch.bool
        or option_active.device != search_action.device
    ):
        raise ValueError("option_active must be a co-located bool[N] tensor")
    if not bool(torch.isfinite(search_action).all()) or not bool(
        torch.isfinite(routed_v6_action).all()
    ):
        raise FloatingPointError("a policy action contains NaN or infinity")
    return torch.where(option_active.unsqueeze(-1), routed_v6_action, search_action)


def public_latch_transition_masks(
    *,
    transition_observation: torch.Tensor,
    active_before: torch.Tensor,
    ever_latched_before: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Audit latch edges on the observation clock used by the policy router.

    ``DirectRLEnv.step`` snapshots task terminal fields before ``_get_rewards``
    and computes policy observations afterwards.  PickTool updates its grasp
    latch in ``_get_rewards``, so the task terminal mapping can lag the public
    observation by one action.  The adapter's transition observation preserves
    this post-reward public observation even on auto-reset rows.  The action
    that first produces index 106 equal to one is therefore the first
    latch-transition action retained by this dataset.
    """

    if (
        not isinstance(transition_observation, torch.Tensor)
        or transition_observation.ndim != 2
        or transition_observation.shape[1] != OBSERVATION_DIM
        or not transition_observation.dtype.is_floating_point
    ):
        raise ValueError(
            f"transition_observation must be floating [N,{OBSERVATION_DIM}]"
        )
    batch = transition_observation.shape[0]
    for name, value in (
        ("active_before", active_before),
        ("ever_latched_before", ever_latched_before),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (batch,)
            or value.dtype != torch.bool
            or value.device != transition_observation.device
        ):
            raise ValueError(f"{name} must be a co-located bool[N] tensor")
    latch = transition_observation[:, PUBLIC_LATCH_INDEX]
    if not bool(((latch == 0.0) | (latch == 1.0)).all()):
        raise ValueError("transition public latch must be exactly binary")
    latched_after = latch == 1.0
    newly_latched = active_before & (~ever_latched_before) & latched_after
    released_after_first = active_before & ever_latched_before & (~latched_after)
    return newly_latched, released_after_first


def _require_cpu_vector(
    payload: Mapping[str, Any],
    name: str,
    *,
    length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = payload.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (length,)
        or value.dtype != dtype
        or value.device.type != "cpu"
    ):
        raise ValueError(f"{name} must be CPU {dtype}[{length}]")
    return value


def _validate_metadata(metadata: Any) -> tuple[int, int, str, int]:
    if not isinstance(metadata, dict):
        raise TypeError("meta must be a plain dictionary")
    _require_json_safe(metadata, name="meta")
    for name, expected in REQUIRED_METADATA.items():
        if metadata.get(name) != expected or type(metadata.get(name)) is not type(expected):
            raise ValueError(f"meta field {name!r} changed")
    seed = metadata.get("seed")
    num_envs = metadata.get("num_envs")
    replicate = metadata.get("cohort_replicate")
    min_retained = metadata.get("min_retained")
    if not _is_int(seed):
        raise TypeError("meta.seed must be an integer")
    if not _is_int(num_envs) or num_envs < 2 or num_envs % 2:
        raise ValueError("meta.num_envs must be an even integer >=2")
    if replicate not in {"a", "b"}:
        raise ValueError("meta.cohort_replicate must be 'a' or 'b'")
    if not _is_int(min_retained) or min_retained < 1:
        raise ValueError("meta.min_retained must be a positive integer")
    if metadata.get("cohort_assignment_salt") != COHORT_ASSIGNMENT_SALT:
        raise ValueError("meta cohort assignment salt changed")
    for key in ("v6_checkpoint", "search_checkpoint"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"meta.{key} must be a non-empty path")
    for key in SHA_METADATA_NAMES:
        _require_sha256(metadata.get(key), name=f"meta.{key}")
    source = metadata.get("source_sha256")
    assets = metadata.get("runtime_asset_sha256")
    if not isinstance(source, dict) or not isinstance(assets, dict):
        raise TypeError("meta source/runtime manifests must be dictionaries")
    if manifest_sha256(source) != metadata["source_manifest_sha256"]:
        raise ValueError("source manifest aggregate SHA256 differs")
    if manifest_sha256(assets) != metadata["runtime_asset_manifest_sha256"]:
        raise ValueError("runtime asset manifest aggregate SHA256 differs")
    _require_git_sha(metadata.get("flashsac_upstream_commit"), name="upstream commit")
    fork = _require_git_sha(metadata.get("flashsac_fork_commit"), name="fork commit")
    git = metadata.get("git")
    if not isinstance(git, dict):
        raise TypeError("meta.git must be a dictionary")
    _require_git_sha(git.get("commit"), name="meta.git.commit")
    if git.get("source_files_dirty") is not False or git.get("flashsac_dirty") is not False:
        raise ValueError("self-imitation evidence requires clean authenticated sources")
    if _require_git_sha(git.get("flashsac_commit"), name="meta.git.flashsac_commit") != fork:
        raise ValueError("FlashSAC submodule differs from the loaded fork")
    if not isinstance(git.get("branch"), str) or not git["branch"]:
        raise ValueError("meta.git.branch must be non-empty")
    runtime = metadata.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("seed") != seed:
        raise ValueError("meta.runtime must bind the collection seed")
    return seed, num_envs, replicate, min_retained


def validate_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a CPU, weights-only self-imitation dataset and full trial audit."""

    if not isinstance(payload, Mapping) or set(payload) != set(DATASET_KEYS):
        raise ValueError(f"dataset must contain exactly {DATASET_KEYS}")
    seed, num_envs, replicate, min_retained = _validate_metadata(payload["meta"])

    obs = payload["obs"]
    action = payload["action"]
    if (
        not isinstance(obs, torch.Tensor)
        or obs.ndim != 2
        or obs.shape[1] != OBSERVATION_DIM
        or obs.dtype != torch.float32
        or obs.device.type != "cpu"
    ):
        raise ValueError(f"obs must be CPU float32[N,{OBSERVATION_DIM}]")
    rows = obs.shape[0]
    if rows < 1 or not isinstance(action, torch.Tensor) or action.shape != (
        rows,
        ACTION_DIM,
    ) or action.dtype != torch.float32 or action.device.type != "cpu":
        raise ValueError(f"action must be CPU float32[N,{ACTION_DIM}] with N>0")
    if not bool(torch.isfinite(obs).all() and torch.isfinite(action).all()):
        raise FloatingPointError("obs/action contains NaN or infinity")
    if float(action.abs().max()) > 1.0001:
        raise ValueError("action exceeds normalized [-1,1]")
    if not torch.equal(action[:, :ARM_ACTION_DIM], torch.zeros_like(action[:, :ARM_ACTION_DIM])):
        raise ValueError("every stored pre-latch CLOSE arm action must be exact zero")
    if not bool((obs[:, PUBLIC_LATCH_INDEX] == 0.0).all()):
        raise ValueError("stored CLOSE observations must all be pre-latch")

    phase = _require_cpu_vector(payload, "phase", length=rows, dtype=torch.uint8)
    episode_id = _require_cpu_vector(payload, "episode_id", length=rows, dtype=torch.int64)
    step = _require_cpu_vector(payload, "step", length=rows, dtype=torch.int64)
    source_step = _require_cpu_vector(payload, "source_step", length=rows, dtype=torch.int64)
    if not bool((phase == 1).all()):
        raise ValueError("self-imitation rows must all use close phase=1")

    offsets = payload.get("episode_offsets")
    if (
        not isinstance(offsets, torch.Tensor)
        or offsets.dtype != torch.int64
        or offsets.device.type != "cpu"
        or offsets.ndim != 1
        or offsets.numel() < 2
        or int(offsets[0]) != 0
        or int(offsets[-1]) != rows
        or bool((offsets[1:] <= offsets[:-1]).any())
    ):
        raise ValueError("episode_offsets must be CPU int64 strictly increasing [0,...,N]")
    episodes = offsets.numel() - 1
    if episodes < min_retained:
        raise ValueError("retained episode count is below meta.min_retained")

    episode: dict[str, torch.Tensor] = {}
    for name in EPISODE_BOOL_NAMES:
        episode[name] = _require_cpu_vector(payload, name, length=episodes, dtype=torch.bool)
    for name in EPISODE_LONG_NAMES:
        episode[name] = _require_cpu_vector(payload, name, length=episodes, dtype=torch.int64)
    for name in EPISODE_FLOAT_NAMES:
        episode[name] = _require_cpu_vector(payload, name, length=episodes, dtype=torch.float32)
        if not bool(torch.isfinite(episode[name]).all()):
            raise FloatingPointError(f"{name} contains NaN or infinity")

    expected_true = (
        "episode_success",
        "episode_native_success",
        "episode_ever_grasped",
        "episode_triggered",
        "episode_ever_latched",
    )
    expected_false = (
        "episode_dropped",
        "episode_unsafe_force",
        "episode_unlatched_clearance_ge_5cm",
        "episode_latch_released_after_first",
    )
    if any(not bool(episode[name].all()) for name in expected_true) or any(
        bool(episode[name].any()) for name in expected_false
    ):
        raise ValueError("retained episodes violate strict success/safety evidence")
    cohort = episode["episode_cohort"]
    noise = episode["episode_hand_noise_scale"]
    if bool(((cohort != 0) & (cohort != 1)).any()):
        raise ValueError("episode_cohort must use 0=deterministic/1=exploratory")
    expected_noise = cohort.to(torch.float32) * EXPLORATORY_HAND_NOISE_SCALE
    if not torch.equal(noise, expected_noise):
        raise ValueError("episode noise scale disagrees with its cohort")
    trigger = episode["episode_trigger_step"]
    first_latch = episode["episode_first_latch_step"]
    terminal = episode["episode_terminal_step"]
    score = episode["episode_trigger_score"]
    if bool(
        (trigger < HANDOFF_HOLD_STEPS - 1).any()
        or (first_latch < trigger).any()
        or (terminal < first_latch).any()
        or (terminal >= MAX_EPISODE_ACTIONS).any()
        or (score < HANDOFF_MIN_SCORE).any()
        or (score > 1.0).any()
        or (episode["episode_max_true_clearance_m"] < 0.20).any()
        or (episode["episode_trajectory_max_force"] < 0.0).any()
        or (episode["episode_trajectory_max_force"] > 30.0).any()
    ):
        raise ValueError("retained episode temporal/physical evidence is inconsistent")

    offsets_list = offsets.tolist()
    for index, (start, stop) in enumerate(zip(offsets_list[:-1], offsets_list[1:], strict=True)):
        length = stop - start
        if not torch.equal(episode_id[start:stop], torch.full((length,), index, dtype=torch.int64)):
            raise ValueError(f"episode_id segment {index} is not canonical")
        if not torch.equal(step[start:stop], torch.arange(length, dtype=torch.int64)):
            raise ValueError(f"episode {index} local step is not contiguous")
        expected_source = torch.arange(
            int(trigger[index]), int(first_latch[index]) + 1, dtype=torch.int64
        )
        if not torch.equal(source_step[start:stop], expected_source):
            raise ValueError(
                f"episode {index} must store every executed action from trigger through first latch"
            )

    trial: dict[str, torch.Tensor] = {}
    for name in TRIAL_BOOL_NAMES:
        trial[name] = _require_cpu_vector(payload, name, length=num_envs, dtype=torch.bool)
    for name in TRIAL_LONG_NAMES:
        trial[name] = _require_cpu_vector(payload, name, length=num_envs, dtype=torch.int64)
    for name in TRIAL_FLOAT_NAMES:
        trial[name] = _require_cpu_vector(payload, name, length=num_envs, dtype=torch.float32)
        if not bool(torch.isfinite(trial[name]).all()):
            raise FloatingPointError(f"{name} contains NaN or infinity")
    if not torch.equal(trial["trial_env_slot"], torch.arange(num_envs, dtype=torch.int64)):
        raise ValueError("trial_env_slot must be canonical arange")
    assignment = exploratory_cohort_mask(seed=seed, num_envs=num_envs, replicate=replicate)
    if not torch.equal(trial["trial_cohort"], assignment.to(torch.int64)):
        raise ValueError("trial cohort differs from its balanced assignment")
    if not torch.equal(
        trial["trial_hand_noise_scale"],
        trial["trial_cohort"].to(torch.float32) * EXPLORATORY_HAND_NOISE_SCALE,
    ):
        raise ValueError("trial noise scale disagrees with cohort")
    length = trial["trial_episode_length"]
    terminal_step = trial["trial_terminal_step"]
    if bool(((length < 1) | (length > MAX_EPISODE_ACTIONS)).any()) or not torch.equal(
        terminal_step, length - 1
    ):
        raise ValueError("trial episode length/terminal step is invalid")
    success = trial["trial_native_success"]
    failure = trial["trial_failure"]
    timeout = trial["trial_time_out"]
    if bool(((success & failure) | (success & timeout) | (failure & timeout)).any()) or not bool(
        (success | failure | timeout).all()
    ):
        raise ValueError("trial primary outcomes must exactly partition episodes")
    failure_sources = (
        trial["trial_dropped"]
        | trial["trial_unsafe_force"]
        | trial["trial_unlatched_clearance_ge_5cm"]
    )
    if not torch.equal(failure, failure_sources):
        raise ValueError("trial failure differs from its task-authored sources")
    if bool((timeout & (length != MAX_EPISODE_ACTIONS)).any()):
        raise ValueError("trial timeout must occur at the exact authored horizon")
    triggered = trial["trial_triggered"]
    trigger_step = trial["trial_trigger_step"]
    trigger_score = trial["trial_trigger_score"]
    if bool((trigger_step[~triggered] != -1).any()) or bool(
        (trigger_score[~triggered] != 0.0).any()
    ) or bool((trigger_step[triggered] < HANDOFF_HOLD_STEPS - 1).any()) or bool(
        (trigger_score[triggered] < HANDOFF_MIN_SCORE).any()
        | (trigger_score[triggered] > 1.0).any()
    ):
        raise ValueError("trial trigger evidence is inconsistent")
    if bool((trigger_step[triggered] > terminal_step[triggered]).any()):
        raise ValueError("trial trigger cannot follow its terminal transition")
    ever_latched = trial["trial_ever_latched"]
    latch_step = trial["trial_first_latch_step"]
    if bool((latch_step[~ever_latched] != -1).any()) or bool(
        (latch_step[ever_latched] < 0).any()
        | (latch_step[ever_latched] > terminal_step[ever_latched]).any()
    ):
        raise ValueError("trial latch evidence is inconsistent")
    if bool((trial["trial_latch_released_after_first"] & ~ever_latched).any()):
        raise ValueError("a latch release cannot precede the first latch")
    if not torch.equal(
        trial["trial_ever_clearance_ge_20cm"],
        trial["trial_max_true_clearance_m"] >= 0.20,
    ):
        raise ValueError("trial 20cm event differs from true-clearance maximum")
    if bool((trial["trial_trajectory_max_force"] < 0.0).any()):
        raise ValueError("trial trajectory force must be non-negative")
    if bool(
        (
            success
            & (
                ~trial["trial_ever_grasped"]
                | ~trial["trial_ever_clearance_ge_20cm"]
                | trial["trial_dropped"]
                | trial["trial_unsafe_force"]
                | trial["trial_unlatched_clearance_ge_5cm"]
            )
        ).any()
    ):
        raise ValueError("trial success violates grasp, clearance, or safety truth")
    if bool(
        (
            trial["trial_unlatched_clearance_ge_5cm"]
            & (trial["trial_max_true_clearance_m"] < 0.05)
        ).any()
    ):
        raise ValueError("unlatched 5cm event disagrees with clearance maximum")
    retained = (
        triggered
        & success
        & ever_latched
        & trial["trial_ever_grasped"]
        & trial["trial_ever_clearance_ge_20cm"]
        & ~trial["trial_dropped"]
        & ~trial["trial_unsafe_force"]
        & ~trial["trial_unlatched_clearance_ge_5cm"]
        & ~trial["trial_latch_released_after_first"]
        & (trial["trial_trajectory_max_force"] <= 30.0)
        & (trial["trial_first_latch_step"] >= trial["trial_trigger_step"])
    )
    if not torch.equal(trial["trial_retained"], retained):
        raise ValueError("trial_retained differs from strict acceptance")
    retained_ids = retained.nonzero(as_tuple=False).flatten()
    retained_cohort = trial["trial_cohort"][retained_ids]
    if not bool((retained_cohort == 0).any()) or not bool(
        (retained_cohort == 1).any()
    ):
        raise ValueError("dataset requires at least one retained episode per cohort")
    if not torch.equal(episode["episode_source_env_id"], retained_ids):
        raise ValueError("retained episode order must match sorted trial env slots")

    episode_to_trial = {
        "episode_native_success": "trial_native_success",
        "episode_dropped": "trial_dropped",
        "episode_unsafe_force": "trial_unsafe_force",
        "episode_unlatched_clearance_ge_5cm": "trial_unlatched_clearance_ge_5cm",
        "episode_ever_grasped": "trial_ever_grasped",
        "episode_triggered": "trial_triggered",
        "episode_ever_latched": "trial_ever_latched",
        "episode_latch_released_after_first": "trial_latch_released_after_first",
        "episode_cohort": "trial_cohort",
        "episode_hand_noise_scale": "trial_hand_noise_scale",
        "episode_trigger_step": "trial_trigger_step",
        "episode_trigger_score": "trial_trigger_score",
        "episode_first_latch_step": "trial_first_latch_step",
        "episode_terminal_step": "trial_terminal_step",
        "episode_max_true_clearance_m": "trial_max_true_clearance_m",
        "episode_trajectory_max_force": "trial_trajectory_max_force",
    }
    for episode_name, trial_name in episode_to_trial.items():
        if not torch.equal(episode[episode_name], trial[trial_name][retained_ids]):
            raise ValueError(f"{episode_name} differs from full trial evidence")

    return {
        name: (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else dict(value)
            if name == "meta"
            else value
        )
        for name, value in payload.items()
    }


def build_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    return validate_dataset(payload)


def _count_summary(dataset: Mapping[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    fields = {
        "triggered": dataset["trial_triggered"],
        "ever_latched": dataset["trial_ever_latched"],
        "success": dataset["trial_native_success"],
        "unsafe_force": dataset["trial_unsafe_force"],
        "dropped": dataset["trial_dropped"],
        "unlatched_clearance_ge_5cm": dataset[
            "trial_unlatched_clearance_ge_5cm"
        ],
        "retained": dataset["trial_retained"],
        "latch_released_after_first": dataset[
            "trial_latch_released_after_first"
        ],
        "trajectory_force_gt_30n": dataset["trial_trajectory_max_force"] > 30.0,
    }
    result: dict[str, Any] = {"episodes": int(mask.sum())}
    for name, value in fields.items():
        result[name] = int((mask & value).sum())
    result["env_ids"] = [
        int(value) for value in mask.nonzero(as_tuple=False).flatten().tolist()
    ]
    return result


def summarize_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    dataset = validate_dataset(payload)
    cohort = dataset["trial_cohort"]
    triggered = dataset["trial_triggered"]
    score = dataset["trial_trigger_score"]
    result: dict[str, Any] = {}
    for cohort_id, name in ((0, "deterministic"), (1, "exploratory")):
        assigned = cohort == cohort_id
        buckets: dict[str, Any] = {}
        for label, lower, upper in SCORE_BUCKETS:
            bucket = assigned & triggered & (score >= lower)
            if upper is not None:
                bucket &= score < upper
            buckets[label] = _count_summary(dataset, bucket)
        result[name] = {
            "all_assigned": _count_summary(dataset, assigned),
            "trigger_score_buckets": buckets,
        }
    return result


REPORT_KEYS = (
    "kind",
    "status",
    "collector",
    "seed",
    "cohort_replicate",
    "num_envs",
    "vector_steps",
    "retained_episodes",
    "transitions",
    "v6_actor_sha256",
    "search_checkpoint_sha256",
    "summary",
)
PUBLISHED_REPORT_KEYS = (*REPORT_KEYS, "dataset_sha256", "dataset_output")


def validate_report(
    report: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    published: bool = False,
) -> dict[str, Any]:
    validated = validate_dataset(dataset)
    if not isinstance(report, dict):
        raise TypeError("report must be a plain dictionary")
    expected_keys = PUBLISHED_REPORT_KEYS if published else REPORT_KEYS
    if set(report) != set(expected_keys):
        raise ValueError(f"report must contain exactly {expected_keys}")
    _require_json_safe(report, name="report")
    metadata = validated["meta"]
    fixed = {
        "kind": REPORT_KIND,
        "status": "complete",
        "collector": COLLECTOR,
        "seed": metadata["seed"],
        "cohort_replicate": metadata["cohort_replicate"],
        "num_envs": metadata["num_envs"],
        "vector_steps": int(validated["trial_episode_length"].max()),
        "retained_episodes": int(validated["trial_retained"].sum()),
        "transitions": int(validated["obs"].shape[0]),
        "v6_actor_sha256": metadata["v6_actor_sha256"],
        "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
        "summary": summarize_dataset(validated),
    }
    for name, expected in fixed.items():
        if report.get(name) != expected:
            raise ValueError(f"report field {name!r} differs from dataset")
    if published:
        _require_sha256(report["dataset_sha256"], name="report.dataset_sha256")
        if not isinstance(report["dataset_output"], str) or not report["dataset_output"]:
            raise ValueError("report.dataset_output must be a non-empty path")
    return dict(report)


def _owned(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _unlink_if_same_file(path: Path, temporary: Path) -> None:
    try:
        if _owned(path) and _owned(temporary) and os.path.samefile(path, temporary):
            path.unlink()
    except FileNotFoundError:
        pass


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def publish_dataset_and_report_no_clobber(
    dataset: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    dataset_output: Path,
    report_output: Path,
) -> str:
    """Atomically publish immutable PT+JSON siblings or publish neither."""

    validated = validate_dataset(dataset)
    validated_report = validate_report(report, validated)
    dataset_output = Path(os.path.abspath(os.fspath(dataset_output)))
    report_output = Path(os.path.abspath(os.fspath(report_output)))
    if dataset_output == report_output:
        raise ValueError("dataset and report outputs must differ")
    for output in (dataset_output, report_output):
        if _owned(output):
            raise FileExistsError(f"self-imitation output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    intended_links: dict[Path, Path] = {}
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{dataset_output.name}.tmp-", dir=dataset_output.parent
        )
        os.close(descriptor)
        dataset_temp = Path(name)
        temporary_paths.append(dataset_temp)
        intended_links[dataset_output] = dataset_temp
        with dataset_temp.open("wb") as stream:
            torch.save(validated, stream)
            stream.flush()
            os.fsync(stream.fileno())
        dataset_sha = sha256_file(dataset_temp)

        final_report = dict(validated_report)
        final_report["dataset_sha256"] = dataset_sha
        final_report["dataset_output"] = str(dataset_output)
        validate_report(final_report, validated, published=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{report_output.name}.tmp-", dir=report_output.parent
        )
        report_temp = Path(name)
        temporary_paths.append(report_temp)
        intended_links[report_output] = report_temp
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_strict_json_bytes(final_report))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(dataset_temp, dataset_output)
        os.link(report_temp, report_output)
        for directory in {dataset_output.parent, report_output.parent}:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return dataset_sha
    except BaseException:
        for output, temporary in reversed(tuple(intended_links.items())):
            _unlink_if_same_file(output, temporary)
        raise
    finally:
        for temporary in temporary_paths:
            if _owned(temporary):
                temporary.unlink()


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    _require_json_safe(dict(payload), name="JSON payload")
    serialized = _strict_json_bytes(payload)
    output = Path(os.path.abspath(os.fspath(output)))
    if _owned(output):
        raise FileExistsError(f"JSON output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        _unlink_if_same_file(output, temporary)
        raise
    finally:
        if _owned(temporary):
            temporary.unlink()


__all__ = [
    "ACTION_DIM",
    "ARM_ACTION_DIM",
    "COHORT_ASSIGNMENT",
    "COLLECTOR",
    "EXPLORATORY_HAND_NOISE_SCALE",
    "HANDOFF_HOLD_STEPS",
    "HANDOFF_MIN_SCORE",
    "KIT_ARGS",
    "MAX_EPISODE_ACTIONS",
    "OBSERVATION_DIM",
    "PUBLIC_LATCH_INDEX",
    "REPORT_KIND",
    "build_cohort_noise_scale",
    "build_dataset",
    "exploratory_cohort_mask",
    "manifest_sha256",
    "publish_dataset_and_report_no_clobber",
    "publish_json_no_clobber",
    "select_executed_action",
    "sha256_file",
    "summarize_dataset",
    "validate_dataset",
    "validate_report",
]
