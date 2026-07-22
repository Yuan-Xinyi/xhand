"""Public-observation SEARCH-to-option handoff state for online training.

The deployment policy runs the frozen SEARCH actor until the hand is close
enough to the handle for several consecutive frames.  This module keeps that
decision Markov from the trainer's point of view and deliberately reads only
the existing 115-D public policy observation.

Unlike :mod:`public_route_trial_contract`, the score threshold is an explicit
training-curriculum parameter.  The randomized trial remains byte-for-byte
sealed at its preregistered 0.10 threshold.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


OBSERVATION_DIM = 115
PUBLIC_LATCH_INDEX = 106
NONTHUMB_PROXIMITY_SLICE = slice(92, 96)
THUMB_PROXIMITY_INDEX = 96
INITIAL_EPISODE_TERMINAL_EVENT_KEYS = (
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
)


def validate_online_handoff_config(
    *, min_score: float, hold_steps: int
) -> tuple[float, int]:
    """Return a normalized public handoff configuration or fail closed."""

    if (
        not isinstance(min_score, (int, float))
        or isinstance(min_score, bool)
        or not math.isfinite(float(min_score))
        or not 0.0 < float(min_score) <= 1.0
    ):
        raise ValueError("online handoff min_score must be finite and in (0, 1]")
    if not isinstance(hold_steps, int) or isinstance(hold_steps, bool) or hold_steps < 1:
        raise ValueError("online handoff hold_steps must be a positive integer")
    return float(min_score), hold_steps


def update_online_handoff(
    observation: torch.Tensor,
    *,
    ready_count_before: torch.Tensor,
    option_active_before: torch.Tensor,
    min_score: float,
    hold_steps: int,
) -> dict[str, torch.Tensor]:
    """Advance the sticky SEARCH-to-option handoff for one pre-action frame.

    ``trigger`` and ``option_active_after`` apply to the action selected from
    this same observation.  The caller resets both state tensors after an
    environment's terminal transition.
    """

    threshold, consecutive = validate_online_handoff_config(
        min_score=min_score, hold_steps=hold_steps
    )
    if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
        raise ValueError("observation must be a rank-two tensor")
    if observation.shape[1] != OBSERVATION_DIM:
        raise ValueError("observation must have shape [N,115]")
    if not observation.dtype.is_floating_point:
        raise TypeError("observation must be floating point")
    batch = observation.shape[0]
    device = observation.device
    if (
        not isinstance(ready_count_before, torch.Tensor)
        or ready_count_before.shape != (batch,)
        or ready_count_before.dtype != torch.long
        or ready_count_before.device != device
    ):
        raise ValueError("ready_count_before must be a device-local int64 [N] tensor")
    if (
        not isinstance(option_active_before, torch.Tensor)
        or option_active_before.shape != (batch,)
        or option_active_before.dtype != torch.bool
        or option_active_before.device != device
    ):
        raise ValueError("option_active_before must be a device-local bool [N] tensor")

    # Same public score used by the randomized route trial: thumb proximity
    # must be accompanied by at least two non-thumb fingers.
    second_nonthumb = torch.topk(
        observation[:, NONTHUMB_PROXIMITY_SLICE], k=2, dim=-1
    ).values[:, 1]
    score = torch.minimum(observation[:, THUMB_PROXIMITY_INDEX], second_nonthumb)
    unlatched = observation[:, PUBLIC_LATCH_INDEX] == 0.0
    eligible = (~option_active_before) & unlatched & (score >= threshold)
    incremented = torch.clamp(ready_count_before + 1, max=consecutive)
    ready_count_after = torch.where(
        option_active_before,
        ready_count_before,
        torch.where(eligible, incremented, torch.zeros_like(ready_count_before)),
    )
    trigger = (~option_active_before) & (ready_count_after == consecutive)
    option_active_after = option_active_before | trigger
    return {
        "score": score,
        "eligible": eligible,
        "ready_count_before": ready_count_before,
        "ready_count_after": ready_count_after,
        "trigger": trigger,
        "option_active_before": option_active_before,
        "option_active_after": option_active_after,
    }


def reset_online_handoff_state(
    *,
    ready_count: torch.Tensor,
    option_active: torch.Tensor,
    done: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clear sticky handoff state only for auto-reset environment rows."""

    if (
        not isinstance(ready_count, torch.Tensor)
        or ready_count.ndim != 1
        or ready_count.dtype != torch.long
    ):
        raise ValueError("ready_count must be a one-dimensional int64 tensor")
    if (
        not isinstance(option_active, torch.Tensor)
        or option_active.shape != ready_count.shape
        or option_active.dtype != torch.bool
        or option_active.device != ready_count.device
    ):
        raise ValueError("option_active must be a co-located bool tensor")
    if (
        not isinstance(done, torch.Tensor)
        or done.shape != ready_count.shape
        or done.dtype != torch.bool
        or done.device != ready_count.device
    ):
        raise ValueError("done must be a co-located bool tensor")
    return (
        torch.where(done, torch.zeros_like(ready_count), ready_count),
        option_active & (~done),
    )


def update_initial_episode_handoff_audit(
    *,
    pending_before: torch.Tensor,
    triggered_seen_before: torch.Tensor,
    trigger: torch.Tensor,
    done: torch.Tensor,
    terminal_truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit only each environment row's first completed episode.

    The audit state is deliberately independent of handoff/action state.  A
    row starts pending, contributes exactly once on its first ``done``, and is
    then permanently cleared for the rest of the run.  ``triggered_seen``
    preserves the stable environment-slot identity of every first-episode
    handoff, including one that occurs on the terminal frame itself.
    """

    if (
        not isinstance(pending_before, torch.Tensor)
        or pending_before.ndim != 1
        or pending_before.dtype != torch.bool
    ):
        raise ValueError("pending_before must be a one-dimensional bool tensor")
    batch = pending_before.shape[0]
    device = pending_before.device
    for name, value in (
        ("triggered_seen_before", triggered_seen_before),
        ("trigger", trigger),
        ("done", done),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (batch,)
            or value.dtype != torch.bool
            or value.device != device
        ):
            raise ValueError(f"{name} must be a co-located bool[{batch}] tensor")
    if not isinstance(terminal_truth, Mapping):
        raise TypeError("terminal_truth must be a mapping")

    validated_terminal: dict[str, torch.Tensor] = {}
    for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS:
        value = terminal_truth.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (batch,)
            or value.dtype != torch.bool
            or value.device != device
        ):
            raise ValueError(
                f"terminal_truth[{name!r}] must be a co-located bool[{batch}] tensor"
            )
        validated_terminal[name] = value

    primary_done = (
        validated_terminal["success"]
        | validated_terminal["failure"]
        | validated_terminal["time_out"]
    )
    if not torch.equal(primary_done, done):
        raise ValueError(
            "success/failure/time_out must exactly reproduce the done vector"
        )
    if bool(
        (
            (validated_terminal["success"] & validated_terminal["failure"])
            | (validated_terminal["success"] & validated_terminal["time_out"])
            | (validated_terminal["failure"] & validated_terminal["time_out"])
        ).any()
    ):
        raise ValueError("success, failure, and time_out must be mutually exclusive")
    failure_sources = (
        validated_terminal["dropped"]
        | validated_terminal["unsafe_force"]
        | validated_terminal["unlatched_clearance_ge_5cm"]
    )
    if not torch.equal(validated_terminal["failure"], failure_sources):
        raise ValueError(
            "failure must exactly equal dropped|unsafe_force|unlatched_clearance_ge_5cm"
        )

    newly_triggered = pending_before & trigger & (~triggered_seen_before)
    triggered_seen_after = triggered_seen_before | newly_triggered
    initial_done = pending_before & done
    triggered_done = initial_done & triggered_seen_after
    search_only_done = initial_done & (~triggered_seen_after)
    return {
        "pending_after": pending_before & (~done),
        "triggered_seen_after": triggered_seen_after,
        "trigger_count": newly_triggered.sum(),
        "completed_triggered_episodes": triggered_done.sum(),
        "completed_search_only_episodes": search_only_done.sum(),
        "triggered_terminal_counts": {
            name: (triggered_done & value).sum()
            for name, value in validated_terminal.items()
        },
        "triggered_terminal_masks": {
            name: triggered_done & value
            for name, value in validated_terminal.items()
        },
    }


def encode_env_slot_mask(mask: torch.Tensor, *, name: str = "mask") -> list[int]:
    """Encode a bool environment-slot mask as sorted JSON-safe row IDs."""

    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 1
        or mask.dtype != torch.bool
    ):
        raise ValueError(f"{name} must be a one-dimensional bool tensor")
    return [int(value) for value in mask.nonzero(as_tuple=False).flatten().cpu()]


def decode_env_slot_ids(env_ids: Sequence[int], *, num_envs: int) -> torch.Tensor:
    """Decode and validate stable environment-slot row IDs on CPU."""

    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 0:
        raise ValueError("num_envs must be a non-negative integer")
    if not isinstance(env_ids, Sequence) or isinstance(env_ids, (str, bytes)):
        raise TypeError("env_ids must be a sequence of integers")
    normalized: list[int] = []
    previous = -1
    for value in env_ids:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("env_ids must contain only integers")
        if value < 0 or value >= num_envs:
            raise ValueError(f"environment slot {value} is outside [0, {num_envs})")
        if value <= previous:
            raise ValueError("env_ids must be unique and strictly increasing")
        normalized.append(value)
        previous = value
    mask = torch.zeros(num_envs, dtype=torch.bool)
    if normalized:
        mask[torch.tensor(normalized, dtype=torch.long)] = True
    return mask


def online_handoff_metrics(
    *,
    enabled: bool,
    search_checkpoint: str | None,
    search_checkpoint_sha256: str | None,
    min_score: float,
    hold_steps: int,
    trigger_count: torch.Tensor,
    search_action_rows: torch.Tensor,
    option_action_rows: torch.Tensor,
    completed_triggered_episodes: torch.Tensor,
    completed_search_only_episodes: torch.Tensor,
    initial_episode_pending_rows: torch.Tensor,
    initial_episode_trigger_count: torch.Tensor,
    initial_episode_completed_triggered_episodes: torch.Tensor,
    initial_episode_completed_search_only_episodes: torch.Tensor,
    initial_episode_triggered_seen: torch.Tensor,
    initial_episode_triggered_terminal_counts: Mapping[str, torch.Tensor],
    initial_episode_triggered_terminal_masks: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Serialize the small set of counters needed to audit online collection."""

    def scalar_count(name: str, value: torch.Tensor) -> int:
        if (
            not isinstance(value, torch.Tensor)
            or value.numel() != 1
            or value.dtype != torch.long
        ):
            raise ValueError(f"{name} must be a scalar int64 tensor")
        result = int(value.item())
        if result < 0:
            raise ValueError(f"{name} must be non-negative")
        return result

    scalar_tensors = {
        "trigger_count": trigger_count,
        "search_action_rows": search_action_rows,
        "option_action_rows": option_action_rows,
        "completed_triggered_episodes": completed_triggered_episodes,
        "completed_search_only_episodes": completed_search_only_episodes,
        "initial_episode_pending_rows": initial_episode_pending_rows,
        "initial_episode_trigger_count": initial_episode_trigger_count,
        "initial_episode_completed_triggered_episodes": (
            initial_episode_completed_triggered_episodes
        ),
        "initial_episode_completed_search_only_episodes": (
            initial_episode_completed_search_only_episodes
        ),
    }
    counts = {
        name: scalar_count(name, value) for name, value in scalar_tensors.items()
    }

    if not isinstance(initial_episode_triggered_seen, torch.Tensor) or (
        initial_episode_triggered_seen.ndim != 1
        or initial_episode_triggered_seen.dtype != torch.bool
    ):
        raise ValueError(
            "initial_episode_triggered_seen must be a one-dimensional bool tensor"
        )
    num_envs = initial_episode_triggered_seen.shape[0]
    triggered_env_ids = encode_env_slot_mask(
        initial_episode_triggered_seen,
        name="initial_episode_triggered_seen",
    )

    if not enabled:
        disabled_metrics: dict[str, Any] = {
            "online_search_handoff": None,
            "online_search_handoff_trigger_count": 0,
            "online_search_handoff_search_action_rows": 0,
            "online_search_handoff_option_action_rows": 0,
            "online_search_handoff_completed_triggered_episodes": 0,
            "online_search_handoff_completed_search_only_episodes": 0,
            "online_search_handoff_initial_episode_pending_rows": 0,
            "online_search_handoff_initial_episode_trigger_count": 0,
            "online_search_handoff_initial_episode_completed_triggered_episodes": 0,
            "online_search_handoff_initial_episode_completed_search_only_episodes": 0,
            "online_search_handoff_initial_episode_triggered_env_ids": [],
            **{
                f"online_search_handoff_initial_episode_triggered_terminal/{name}": 0
                for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS
            },
            **{
                f"online_search_handoff_initial_episode_triggered_terminal_env_ids/{name}": []
                for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS
            },
        }
    else:
        disabled_metrics = {}

    if not isinstance(initial_episode_triggered_terminal_counts, Mapping):
        raise TypeError(
            "initial_episode_triggered_terminal_counts must be a mapping"
        )
    if set(initial_episode_triggered_terminal_counts) != set(
        INITIAL_EPISODE_TERMINAL_EVENT_KEYS
    ):
        raise ValueError(
            "initial_episode_triggered_terminal_counts must contain exactly "
            f"{INITIAL_EPISODE_TERMINAL_EVENT_KEYS}"
        )
    if not isinstance(initial_episode_triggered_terminal_masks, Mapping):
        raise TypeError("initial_episode_triggered_terminal_masks must be a mapping")
    if set(initial_episode_triggered_terminal_masks) != set(
        INITIAL_EPISODE_TERMINAL_EVENT_KEYS
    ):
        raise ValueError(
            "initial_episode_triggered_terminal_masks must contain exactly "
            f"{INITIAL_EPISODE_TERMINAL_EVENT_KEYS}"
        )

    initial_terminal_counts = {
        name: (
            scalar_count(
                f"initial_episode_triggered_terminal_counts[{name!r}]",
                initial_episode_triggered_terminal_counts[name],
            )
        )
        for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS
    }
    initial_terminal_env_ids: dict[str, list[int]] = {}
    for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS:
        mask = initial_episode_triggered_terminal_masks[name]
        if (
            not isinstance(mask, torch.Tensor)
            or mask.shape != (num_envs,)
            or mask.dtype != torch.bool
            or mask.device != initial_episode_triggered_seen.device
        ):
            raise ValueError(
                f"initial_episode_triggered_terminal_masks[{name!r}] must be a "
                f"co-located bool[{num_envs}] tensor"
            )
        if bool((mask & ~initial_episode_triggered_seen).any()):
            raise ValueError(
                f"initial terminal mask {name!r} contains a non-triggered row"
            )
        env_ids = encode_env_slot_mask(
            mask,
            name=f"initial_episode_triggered_terminal_masks[{name!r}]",
        )
        if initial_terminal_counts[name] != len(env_ids):
            raise ValueError(
                f"initial terminal count {name!r} disagrees with its env-slot mask"
            )
        initial_terminal_env_ids[name] = env_ids

    if counts["initial_episode_trigger_count"] != len(triggered_env_ids):
        raise ValueError(
            "initial_episode_trigger_count disagrees with initial_episode_triggered_seen"
        )
    if counts["initial_episode_pending_rows"] > num_envs:
        raise ValueError("initial_episode_pending_rows exceeds the cohort size")
    completed_initial = (
        counts["initial_episode_completed_triggered_episodes"]
        + counts["initial_episode_completed_search_only_episodes"]
    )
    if counts["initial_episode_pending_rows"] + completed_initial != num_envs:
        raise ValueError(
            "initial pending and completed episode counts do not partition the cohort"
        )
    if (
        counts["initial_episode_completed_triggered_episodes"]
        > counts["initial_episode_trigger_count"]
    ):
        raise ValueError("more initial triggered episodes completed than triggered")
    if counts["initial_episode_trigger_count"] > num_envs:
        raise ValueError("initial_episode_trigger_count exceeds the cohort size")
    if counts["initial_episode_completed_search_only_episodes"] > (
        num_envs - counts["initial_episode_trigger_count"]
    ):
        raise ValueError("an initial triggered row was counted as SEARCH-only")
    if counts["initial_episode_pending_rows"] == 0 and (
        counts["initial_episode_completed_triggered_episodes"]
        != counts["initial_episode_trigger_count"]
    ):
        raise ValueError(
            "a complete initial cohort must complete every triggered episode"
        )

    primary_masks = [
        initial_episode_triggered_terminal_masks[name]
        for name in ("success", "failure", "time_out")
    ]
    if bool(
        (
            (primary_masks[0] & primary_masks[1])
            | (primary_masks[0] & primary_masks[2])
            | (primary_masks[1] & primary_masks[2])
        ).any()
    ):
        raise ValueError("initial success/failure/time_out masks overlap")
    if sum(
        initial_terminal_counts[name]
        for name in ("success", "failure", "time_out")
    ) != counts["initial_episode_completed_triggered_episodes"]:
        raise ValueError(
            "initial terminal outcomes do not partition completed triggered episodes"
        )
    failure_source_mask = (
        initial_episode_triggered_terminal_masks["dropped"]
        | initial_episode_triggered_terminal_masks["unsafe_force"]
        | initial_episode_triggered_terminal_masks[
            "unlatched_clearance_ge_5cm"
        ]
    )
    if not torch.equal(
        initial_episode_triggered_terminal_masks["failure"], failure_source_mask
    ):
        raise ValueError(
            "initial failure mask disagrees with its task-authored failure sources"
        )

    if not enabled:
        return disabled_metrics

    threshold, consecutive = validate_online_handoff_config(
        min_score=min_score, hold_steps=hold_steps
    )
    if not isinstance(search_checkpoint, str) or not search_checkpoint:
        raise ValueError("enabled online handoff requires a SEARCH checkpoint path")
    if (
        not isinstance(search_checkpoint_sha256, str)
        or len(search_checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in search_checkpoint_sha256)
    ):
        raise ValueError("enabled online handoff requires a lowercase SHA256")

    return {
        "online_search_handoff": {
            "contract": "pick_tool_public_online_handoff_v1",
            "search_checkpoint": search_checkpoint,
            "search_checkpoint_sha256": search_checkpoint_sha256,
            "score": "min(obs[96], second_largest(obs[92:96]))",
            "requires_unlatched_obs106": True,
            "min_score": threshold,
            "hold_steps": consecutive,
            "trigger_action_semantics": "option_controls_the_trigger_frame_and_remains_sticky_until_reset",
            "replay_semantics": "only_option_controlled_rows",
            "initial_episode_audit_semantics": "each_env_row_is_counted_once_at_its_first_done",
            "initial_episode_pairing_semantics": "stable_zero_based_environment_slot_ids",
        },
        "online_search_handoff_trigger_count": counts["trigger_count"],
        "online_search_handoff_search_action_rows": counts["search_action_rows"],
        "online_search_handoff_option_action_rows": counts["option_action_rows"],
        "online_search_handoff_completed_triggered_episodes": counts[
            "completed_triggered_episodes"
        ],
        "online_search_handoff_completed_search_only_episodes": counts[
            "completed_search_only_episodes"
        ],
        "online_search_handoff_initial_episode_pending_rows": counts[
            "initial_episode_pending_rows"
        ],
        "online_search_handoff_initial_episode_trigger_count": counts[
            "initial_episode_trigger_count"
        ],
        "online_search_handoff_initial_episode_completed_triggered_episodes": counts[
            "initial_episode_completed_triggered_episodes"
        ],
        "online_search_handoff_initial_episode_completed_search_only_episodes": counts[
            "initial_episode_completed_search_only_episodes"
        ],
        "online_search_handoff_initial_episode_triggered_env_ids": triggered_env_ids,
        **{
            f"online_search_handoff_initial_episode_triggered_terminal/{name}": (
                initial_terminal_counts[name]
            )
            for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS
        },
        **{
            f"online_search_handoff_initial_episode_triggered_terminal_env_ids/{name}": (
                initial_terminal_env_ids[name]
            )
            for name in INITIAL_EPISODE_TERMINAL_EVENT_KEYS
        },
    }


__all__ = [
    "OBSERVATION_DIM",
    "PUBLIC_LATCH_INDEX",
    "INITIAL_EPISODE_TERMINAL_EVENT_KEYS",
    "decode_env_slot_ids",
    "encode_env_slot_mask",
    "online_handoff_metrics",
    "reset_online_handoff_state",
    "update_initial_episode_handoff_audit",
    "update_online_handoff",
    "validate_online_handoff_config",
]
