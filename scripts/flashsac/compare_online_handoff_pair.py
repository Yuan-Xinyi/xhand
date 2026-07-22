#!/usr/bin/env python3
"""Compare same-seed first episodes for two online SEARCH-handoff policies.

The trainer's aggregate counters include episodes after auto-reset and are not
paired.  This comparator deliberately consumes only the stable environment-slot
masks from the first-episode audit.  Any missing, malformed, or incomparable
evidence is an error rather than an implicit zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


COMPARISON_KIND = "pick_tool_online_handoff_first_episode_pair_v1"
HANDOFF_CONTRACT = "pick_tool_public_online_handoff_v1"
TERMINAL_EVENTS = (
    "success",
    "failure",
    "time_out",
    "dropped",
    "unsafe_force",
    "unlatched_clearance_ge_5cm",
)
COMPARED_OUTCOMES = ("success", "dropped", "unsafe_force")
EXPECTED_HANDOFF_SEMANTICS = {
    "contract": HANDOFF_CONTRACT,
    "score": "min(obs[96], second_largest(obs[92:96]))",
    "requires_unlatched_obs106": True,
    "trigger_action_semantics": (
        "option_controls_the_trigger_frame_and_remains_sticky_until_reset"
    ),
    "replay_semantics": "only_option_controlled_rows",
    "initial_episode_audit_semantics": (
        "each_env_row_is_counted_once_at_its_first_done"
    ),
    "initial_episode_pairing_semantics": (
        "stable_zero_based_environment_slot_ids"
    ),
}


@dataclass(frozen=True)
class ValidatedRun:
    seed: int
    steps: int
    num_envs: int
    environment_steps: int
    min_score: float
    hold_steps: int
    search_checkpoint_sha256: str
    triggered_env_ids: tuple[int, ...]
    terminal_env_ids: Mapping[str, tuple[int, ...]]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _required_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int = 0,
) -> int:
    value = payload.get(key)
    if not _is_int(value) or value < minimum:
        raise ValueError(f"{key!r} must be an integer >= {minimum}")
    return value


def _validated_ids(value: Any, *, key: str, num_envs: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{key!r} must be a sequence of environment-slot IDs")
    result: list[int] = []
    previous = -1
    for env_id in value:
        if not _is_int(env_id):
            raise TypeError(f"{key!r} must contain only integer IDs")
        if env_id < 0 or env_id >= num_envs:
            raise ValueError(
                f"{key!r} contains slot {env_id} outside [0, {num_envs})"
            )
        if env_id <= previous:
            raise ValueError(f"{key!r} must be unique and strictly increasing")
        result.append(env_id)
        previous = env_id
    return tuple(result)


def _validate_sha256(value: Any, *, key: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{key!r} must be a lowercase SHA256 digest")
    return value


def validate_metrics(metrics: Mapping[str, Any], *, name: str) -> ValidatedRun:
    """Validate one frozen-evaluation metrics payload and its slot masks."""

    if not isinstance(metrics, Mapping):
        raise TypeError(f"{name} metrics root must be a mapping")
    if metrics.get("status") != "complete":
        raise ValueError(f"{name} status must be exactly 'complete'")
    if _required_int(metrics, "gradient_updates") != 0:
        raise ValueError(f"{name} must be a frozen evaluation (gradient_updates == 0)")

    seed = _required_int(metrics, "seed")
    pending = _required_int(
        metrics, "online_search_handoff_initial_episode_pending_rows"
    )
    if pending != 0:
        raise ValueError(f"{name} has incomplete first episodes (pending_rows != 0)")
    trigger_count = _required_int(
        metrics, "online_search_handoff_initial_episode_trigger_count"
    )
    completed_triggered = _required_int(
        metrics,
        "online_search_handoff_initial_episode_completed_triggered_episodes",
    )
    completed_search_only = _required_int(
        metrics,
        "online_search_handoff_initial_episode_completed_search_only_episodes",
    )
    if completed_triggered != trigger_count:
        raise ValueError(
            f"{name} completed triggered first episodes do not equal trigger count"
        )
    num_envs = pending + completed_triggered + completed_search_only
    if num_envs < 1:
        raise ValueError(f"{name} has an empty first-episode cohort")
    explicit_num_envs = metrics.get("num_envs")
    if explicit_num_envs is not None and (
        not _is_int(explicit_num_envs) or explicit_num_envs != num_envs
    ):
        raise ValueError(f"{name} explicit num_envs disagrees with audit partition")

    steps = _required_int(metrics, "interaction_step", minimum=1)
    explicit_steps = metrics.get("steps")
    if explicit_steps is not None and (
        not _is_int(explicit_steps) or explicit_steps != steps
    ):
        raise ValueError(f"{name} explicit steps disagrees with interaction_step")
    environment_steps = _required_int(metrics, "environment_steps", minimum=1)
    if environment_steps != steps * num_envs:
        raise ValueError(
            f"{name} environment_steps != interaction_step * first-episode cohort"
        )

    handoff = metrics.get("online_search_handoff")
    if not isinstance(handoff, Mapping):
        raise TypeError(f"{name} has no enabled online_search_handoff contract")
    for key, expected in EXPECTED_HANDOFF_SEMANTICS.items():
        if handoff.get(key) != expected:
            raise ValueError(f"{name} handoff semantic {key!r} changed")
    min_score_value = handoff.get("min_score")
    if (
        not isinstance(min_score_value, (int, float))
        or isinstance(min_score_value, bool)
        or not math.isfinite(float(min_score_value))
        or not 0.0 < float(min_score_value) <= 1.0
    ):
        raise ValueError(f"{name} handoff min_score must be finite and in (0, 1]")
    min_score = float(min_score_value)
    hold_steps = handoff.get("hold_steps")
    if not _is_int(hold_steps) or hold_steps < 1:
        raise ValueError(f"{name} handoff hold_steps must be a positive integer")
    search_path = handoff.get("search_checkpoint")
    if not isinstance(search_path, str) or not search_path:
        raise ValueError(f"{name} handoff SEARCH checkpoint path is missing")
    search_sha = _validate_sha256(
        handoff.get("search_checkpoint_sha256"),
        key=f"{name}.online_search_handoff.search_checkpoint_sha256",
    )

    trigger_key = "online_search_handoff_initial_episode_triggered_env_ids"
    triggered = _validated_ids(metrics.get(trigger_key), key=trigger_key, num_envs=num_envs)
    if len(triggered) != trigger_count:
        raise ValueError(f"{name} trigger count disagrees with triggered slot mask")
    if not triggered:
        raise ValueError(f"{name} has no triggered first episodes to compare")
    triggered_set = set(triggered)

    terminals: dict[str, tuple[int, ...]] = {}
    for event in TERMINAL_EVENTS:
        count_key = (
            "online_search_handoff_initial_episode_triggered_terminal/" + event
        )
        ids_key = (
            "online_search_handoff_initial_episode_triggered_terminal_env_ids/"
            + event
        )
        count = _required_int(metrics, count_key)
        env_ids = _validated_ids(metrics.get(ids_key), key=ids_key, num_envs=num_envs)
        if count != len(env_ids):
            raise ValueError(f"{name} {event!r} count disagrees with its slot mask")
        if not set(env_ids).issubset(triggered_set):
            raise ValueError(f"{name} {event!r} mask contains a non-triggered slot")
        terminals[event] = env_ids

    success = set(terminals["success"])
    failure = set(terminals["failure"])
    timeout = set(terminals["time_out"])
    if success & failure or success & timeout or failure & timeout:
        raise ValueError(f"{name} primary terminal masks overlap")
    if success | failure | timeout != triggered_set:
        raise ValueError(f"{name} primary terminal masks do not partition trigger cohort")
    failure_sources = (
        set(terminals["dropped"])
        | set(terminals["unsafe_force"])
        | set(terminals["unlatched_clearance_ge_5cm"])
    )
    if failure_sources != failure:
        raise ValueError(f"{name} failure mask disagrees with failure-source masks")

    return ValidatedRun(
        seed=seed,
        steps=steps,
        num_envs=num_envs,
        environment_steps=environment_steps,
        min_score=min_score,
        hold_steps=hold_steps,
        search_checkpoint_sha256=search_sha,
        triggered_env_ids=triggered,
        terminal_env_ids=terminals,
    )


def exact_mcnemar_two_sided_p(improved: int, lost: int) -> float:
    """Exact two-sided sign/binomial p-value for paired discordant rows."""

    if not _is_int(improved) or not _is_int(lost) or improved < 0 or lost < 0:
        raise ValueError("discordant counts must be non-negative integers")
    discordant = improved + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(improved, lost) + 1))
    probability = min(Fraction(1), Fraction(2 * tail, 1 << discordant))
    return float(probability)


def _compare_outcome(
    baseline_ids: Sequence[int],
    candidate_ids: Sequence[int],
    *,
    cohort_size: int,
    beneficial_when_present: bool,
) -> dict[str, Any]:
    baseline = set(baseline_ids)
    candidate = set(candidate_ids)
    candidate_only = sorted(candidate - baseline)
    baseline_only = sorted(baseline - candidate)
    if beneficial_when_present:
        improved_ids = candidate_only
        lost_ids = baseline_only
        orientation = "event_presence_is_beneficial"
    else:
        improved_ids = baseline_only
        lost_ids = candidate_only
        orientation = "event_presence_is_adverse"
    improved = len(improved_ids)
    lost = len(lost_ids)
    delta_count = len(candidate) - len(baseline)
    return {
        "orientation": orientation,
        "baseline_count": len(baseline),
        "candidate_count": len(candidate),
        "baseline_rate": len(baseline) / cohort_size,
        "candidate_rate": len(candidate) / cohort_size,
        "delta_count_candidate_minus_baseline": delta_count,
        "delta_rate_candidate_minus_baseline": delta_count / cohort_size,
        "candidate_only_env_ids": candidate_only,
        "baseline_only_env_ids": baseline_only,
        "improved_count": improved,
        "improved_env_ids": improved_ids,
        "lost_count": lost,
        "lost_env_ids": lost_ids,
        "discordant_count": improved + lost,
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided_p(improved, lost),
        "mcnemar_method": "two_sided_exact_binomial_discordant_pairs_p0.5",
    }


def compare_metrics(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    baseline_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    """Validate and compare two same-seed, same-slot metrics payloads."""

    baseline_digest = _validate_sha256(baseline_sha256, key="baseline_sha256")
    candidate_digest = _validate_sha256(candidate_sha256, key="candidate_sha256")
    baseline = validate_metrics(baseline_metrics, name="baseline")
    candidate = validate_metrics(candidate_metrics, name="candidate")

    comparable_fields = (
        "seed",
        "steps",
        "num_envs",
        "environment_steps",
        "min_score",
        "hold_steps",
        "search_checkpoint_sha256",
    )
    mismatches = {
        field: {
            "baseline": getattr(baseline, field),
            "candidate": getattr(candidate, field),
        }
        for field in comparable_fields
        if getattr(baseline, field) != getattr(candidate, field)
    }
    if mismatches:
        raise ValueError(f"paired run contracts differ: {sorted(mismatches)}")
    if baseline.triggered_env_ids != candidate.triggered_env_ids:
        raise ValueError("paired runs have different first-episode trigger cohorts")

    cohort_size = len(baseline.triggered_env_ids)
    outcomes = {
        event: _compare_outcome(
            baseline.terminal_env_ids[event],
            candidate.terminal_env_ids[event],
            cohort_size=cohort_size,
            beneficial_when_present=(event == "success"),
        )
        for event in COMPARED_OUTCOMES
    }
    new_drops = outcomes["dropped"]["candidate_only_env_ids"]
    new_unsafe = outcomes["unsafe_force"]["candidate_only_env_ids"]
    success = outcomes["success"]
    pilot_checks = {
        "exact_same_triggered_cohort": True,
        "no_new_candidate_drop": len(new_drops) == 0,
        "no_new_candidate_unsafe_force": len(new_unsafe) == 0,
        "successes_improved_at_least_lost": (
            success["improved_count"] >= success["lost_count"]
        ),
    }
    proof_checks = {
        "success_delta_positive": (
            success["delta_count_candidate_minus_baseline"] > 0
        ),
        "mcnemar_p_below_0.05": success["mcnemar_exact_two_sided_p"] < 0.05,
    }
    return {
        "kind": COMPARISON_KIND,
        "status": "complete",
        "evidence_eligible": True,
        "comparison_layout": "same_seed_same_first_episode_environment_slot_v1",
        "baseline_metrics_sha256": baseline_digest,
        "candidate_metrics_sha256": candidate_digest,
        "seed": baseline.seed,
        "steps": baseline.steps,
        "num_envs": baseline.num_envs,
        "environment_steps": baseline.environment_steps,
        "handoff_contract": {
            "contract": HANDOFF_CONTRACT,
            "min_score": baseline.min_score,
            "hold_steps": baseline.hold_steps,
            "search_checkpoint_sha256": baseline.search_checkpoint_sha256,
        },
        "triggered_cohort_count": cohort_size,
        "triggered_cohort_env_ids": list(baseline.triggered_env_ids),
        "outcomes": outcomes,
        "new_candidate_drop_env_ids": new_drops,
        "new_candidate_unsafe_force_env_ids": new_unsafe,
        "pilot_continuation_gate": {
            "checks": pilot_checks,
            "passed": all(pilot_checks.values()),
            "semantics": (
                "same_cohort_and_no_new_safety_event_and_success_improved_ge_lost"
            ),
        },
        "success_proved_improvement": {
            "checks": proof_checks,
            "passed": all(proof_checks.values()),
            "semantics": "positive_paired_success_delta_and_exact_p_below_0.05",
        },
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_metrics(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"metrics must be a regular, non-symlink file: {path}")
    raw = path.read_bytes()
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise TypeError("metrics JSON root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def publish_json_no_clobber(payload: Mapping[str, Any], path: Path) -> None:
    serialized = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.baseline.resolve(strict=False) == args.candidate.resolve(strict=False):
        parser.error("baseline and candidate must be different metrics files")
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        parser.error("--output already exists; comparison artifacts are immutable")
    return args


def main() -> int:
    args = parse_args()
    try:
        baseline_payload, baseline_sha = load_metrics(args.baseline)
        candidate_payload, candidate_sha = load_metrics(args.candidate)
        report = compare_metrics(
            baseline_payload,
            candidate_payload,
            baseline_sha256=baseline_sha,
            candidate_sha256=candidate_sha,
        )
        report["baseline_metrics"] = str(args.baseline.resolve())
        report["candidate_metrics"] = str(args.candidate.resolve())
        report["comparator_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        if args.output is not None:
            publish_json_no_clobber(report, args.output)
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        failure = {
            "kind": COMPARISON_KIND,
            "status": "failed",
            "evidence_eligible": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "baseline_metrics": str(args.baseline.absolute()),
            "candidate_metrics": str(args.candidate.absolute()),
        }
        print(json.dumps(failure, sort_keys=True, indent=2), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
