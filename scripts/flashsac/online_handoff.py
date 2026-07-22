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
from typing import Any

import torch


OBSERVATION_DIM = 115
PUBLIC_LATCH_INDEX = 106
NONTHUMB_PROXIMITY_SLICE = slice(92, 96)
THUMB_PROXIMITY_INDEX = 96


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
) -> dict[str, Any]:
    """Serialize the small set of counters needed to audit online collection."""

    if not enabled:
        return {
            "online_search_handoff": None,
            "online_search_handoff_trigger_count": 0,
            "online_search_handoff_search_action_rows": 0,
            "online_search_handoff_option_action_rows": 0,
            "online_search_handoff_completed_triggered_episodes": 0,
            "online_search_handoff_completed_search_only_episodes": 0,
        }
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
        },
        "online_search_handoff_trigger_count": scalar_count(
            "trigger_count", trigger_count
        ),
        "online_search_handoff_search_action_rows": scalar_count(
            "search_action_rows", search_action_rows
        ),
        "online_search_handoff_option_action_rows": scalar_count(
            "option_action_rows", option_action_rows
        ),
        "online_search_handoff_completed_triggered_episodes": scalar_count(
            "completed_triggered_episodes", completed_triggered_episodes
        ),
        "online_search_handoff_completed_search_only_episodes": scalar_count(
            "completed_search_only_episodes", completed_search_only_episodes
        ),
    }


__all__ = [
    "OBSERVATION_DIM",
    "PUBLIC_LATCH_INDEX",
    "online_handoff_metrics",
    "reset_online_handoff_state",
    "update_online_handoff",
    "validate_online_handoff_config",
]
