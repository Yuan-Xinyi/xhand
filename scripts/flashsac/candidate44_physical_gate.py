#!/usr/bin/env python3
"""Apply Candidate44's preregistered paired physical release gates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


EPISODES = 512
ORDINARY_SEED = 360
CLOSE_START_SEED = 361
OFFLINE_GATE_KIND = "pick_tool_candidate44_phase0_correction_offline_gate_v1"
OFFLINE_GATE_DECISION = "release_to_paired_physical_gate"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_checkpoint(value: Mapping[str, Any], *, label: str) -> Path:
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{label} evaluation has no checkpoint path")
    return Path(checkpoint).expanduser().resolve()


def _require_regular_non_symlink_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular non-symlink file: {path}")


def _require_all_true_checks(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    if any(check is not True for check in value.values()):
        raise ValueError(f"{label} contains a non-passing check")
    return value


def validate_offline_gate_binding(
    offline: Mapping[str, Any],
    baseline_ordinary: Mapping[str, Any],
    candidate_ordinary: Mapping[str, Any],
    baseline_close: Mapping[str, Any],
    candidate_close: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one passing offline report to the exact physical-evaluation bytes."""

    if offline.get("kind") != OFFLINE_GATE_KIND:
        raise ValueError("offline gate kind differs from Candidate44's registered gate")
    if offline.get("status") != "complete":
        raise ValueError("offline gate status is not complete")

    registered = offline.get("registered_gate")
    if not isinstance(registered, Mapping):
        raise ValueError("offline gate has no registered_gate mapping")
    if registered.get("all_pass") is not True:
        raise ValueError("offline registered gate did not pass")
    if registered.get("decision") != OFFLINE_GATE_DECISION:
        raise ValueError(
            "offline registered gate decision does not release physical evaluation"
        )
    registered_checks = _require_all_true_checks(
        registered.get("checks"), label="offline registered_gate checks"
    )
    if registered_checks.get("integrity") is not True:
        raise ValueError("offline registered gate did not explicitly pass integrity")

    integrity = offline.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("offline gate has no integrity mapping")
    if integrity.get("all_pass") is not True:
        raise ValueError("offline gate integrity did not pass")
    _require_all_true_checks(integrity.get("checks"), label="offline integrity checks")

    baseline_checkpoint_value = offline.get("baseline_checkpoint")
    candidate_checkpoint_value = offline.get("candidate_checkpoint")
    if not isinstance(baseline_checkpoint_value, str) or not baseline_checkpoint_value:
        raise ValueError("offline gate has no baseline checkpoint path")
    if not isinstance(candidate_checkpoint_value, str) or not candidate_checkpoint_value:
        raise ValueError("offline gate has no candidate checkpoint path")
    baseline_checkpoint = Path(baseline_checkpoint_value).expanduser().resolve()
    candidate_checkpoint = Path(candidate_checkpoint_value).expanduser().resolve()
    if baseline_checkpoint == candidate_checkpoint:
        raise ValueError("offline baseline and candidate checkpoints are identical")

    evaluation_checkpoints = {
        "baseline_ordinary": _resolved_checkpoint(
            baseline_ordinary, label="baseline ordinary"
        ),
        "candidate_ordinary": _resolved_checkpoint(
            candidate_ordinary, label="candidate ordinary"
        ),
        "baseline_close_start": _resolved_checkpoint(
            baseline_close, label="baseline close-start"
        ),
        "candidate_close_start": _resolved_checkpoint(
            candidate_close, label="candidate close-start"
        ),
    }
    expected_checkpoints = {
        "baseline_ordinary": baseline_checkpoint,
        "baseline_close_start": baseline_checkpoint,
        "candidate_ordinary": candidate_checkpoint,
        "candidate_close_start": candidate_checkpoint,
    }
    for label, expected in expected_checkpoints.items():
        if evaluation_checkpoints[label] != expected:
            raise ValueError(
                f"{label} checkpoint {evaluation_checkpoints[label]} does not match "
                f"offline gate checkpoint {expected}"
            )

    evidence_files = {
        "baseline_actor_sha256": baseline_checkpoint / "actor.pt",
        "candidate_actor_sha256": candidate_checkpoint / "actor.pt",
        "baseline_task_contract_sha256": baseline_checkpoint / "task_contract.json",
        "candidate_task_contract_sha256": candidate_checkpoint / "task_contract.json",
    }
    computed_hashes: dict[str, str] = {}
    reported_hashes: dict[str, str] = {}
    for key, path in evidence_files.items():
        _require_regular_non_symlink_file(path, label=key)
        reported = integrity.get(key)
        if (
            not isinstance(reported, str)
            or len(reported) != 64
            or any(character not in "0123456789abcdef" for character in reported)
        ):
            raise ValueError(f"offline integrity has no valid {key}")
        computed = _sha256_file(path)
        if computed != reported:
            raise ValueError(
                f"offline {key} {reported} does not match physical checkpoint bytes {computed}"
            )
        reported_hashes[key] = reported
        computed_hashes[key] = computed

    return {
        "all_pass": True,
        "offline_kind_exact": True,
        "offline_status_complete": True,
        "offline_registered_gate_passed": True,
        "offline_integrity_passed": True,
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "evaluation_checkpoints": {
            key: str(path) for key, path in evaluation_checkpoints.items()
        },
        "evidence_files": {key: str(path) for key, path in evidence_files.items()},
        "reported_hashes": reported_hashes,
        "computed_hashes": computed_hashes,
    }


def _require_complete_evaluation(
    value: Mapping[str, Any],
    *,
    seed: int,
    curriculum_probability: float,
) -> None:
    required = {
        "status": "complete",
        "seed": seed,
        "requested_episodes": EPISODES,
        "completed_episodes": EPISODES,
        "num_envs": EPISODES,
        "task_mode": "full_task",
        "checkpoint_task_mode": "full_task",
        "policy": "deterministic_tanh_actor_mean",
        "max_episode_steps": 1000,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"evaluation {key}={value.get(key)!r}, expected {expected!r}")
    curriculum = value.get("curriculum")
    if (
        not isinstance(curriculum, Mapping)
        or curriculum.get("probability") != curriculum_probability
    ):
        raise ValueError("evaluation curriculum probability differs from paired contract")
    if curriculum_probability == 0.0:
        if curriculum.get("dataset") is not None:
            raise ValueError("ordinary evaluation unexpectedly uses a curriculum dataset")
    else:
        if (
            curriculum.get("boundary") != "close_start"
            or curriculum.get("joint_noise") != 0.0
            or not isinstance(curriculum.get("dataset_sha256"), str)
        ):
            raise ValueError("close-start evaluation differs from fixed holdout contract")
    for container, keys in (
        (value.get("events"), ("success", "unsafe_force", "unlatched_clearance_ge_5cm")),
        (value.get("funnel"), ("ever_grasped",)),
    ):
        if not isinstance(container, Mapping):
            raise TypeError("evaluation is missing event/funnel mappings")
        for key in keys:
            count = container.get(key)
            if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= EPISODES:
                raise ValueError(f"evaluation count {key!r} is invalid")
    rate = value.get("strict_success_rate")
    if not isinstance(rate, (int, float)) or not math.isfinite(float(rate)):
        raise ValueError("evaluation strict_success_rate is invalid")
    if not math.isclose(
        float(rate), value["events"]["success"] / EPISODES, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("strict_success_rate disagrees with event count")


def compare_physical_gates(
    baseline_ordinary: Mapping[str, Any],
    candidate_ordinary: Mapping[str, Any],
    baseline_close: Mapping[str, Any],
    candidate_close: Mapping[str, Any],
    *,
    offline_gate_passed: bool,
) -> dict[str, Any]:
    _require_complete_evaluation(
        baseline_ordinary, seed=ORDINARY_SEED, curriculum_probability=0.0
    )
    _require_complete_evaluation(
        candidate_ordinary, seed=ORDINARY_SEED, curriculum_probability=0.0
    )
    _require_complete_evaluation(
        baseline_close, seed=CLOSE_START_SEED, curriculum_probability=1.0
    )
    _require_complete_evaluation(
        candidate_close, seed=CLOSE_START_SEED, curriculum_probability=1.0
    )
    baseline_ordinary_checkpoint = _resolved_checkpoint(
        baseline_ordinary, label="baseline ordinary"
    )
    candidate_ordinary_checkpoint = _resolved_checkpoint(
        candidate_ordinary, label="candidate ordinary"
    )
    baseline_close_checkpoint = _resolved_checkpoint(
        baseline_close, label="baseline close-start"
    )
    candidate_close_checkpoint = _resolved_checkpoint(
        candidate_close, label="candidate close-start"
    )
    if baseline_ordinary_checkpoint == candidate_ordinary_checkpoint:
        raise ValueError("ordinary baseline and candidate checkpoints are identical")
    if baseline_close_checkpoint == candidate_close_checkpoint:
        raise ValueError("close-start baseline and candidate checkpoints are identical")
    if baseline_ordinary_checkpoint != baseline_close_checkpoint:
        raise ValueError("paired baselines do not use one checkpoint")
    if candidate_ordinary_checkpoint != candidate_close_checkpoint:
        raise ValueError("paired candidate evaluations do not use one checkpoint")
    if baseline_close["curriculum"]["dataset_sha256"] != candidate_close["curriculum"][
        "dataset_sha256"
    ]:
        raise ValueError("close-start pair uses different holdout bytes")

    ordinary_delta = {
        "ever_grasped_count": candidate_ordinary["funnel"]["ever_grasped"]
        - baseline_ordinary["funnel"]["ever_grasped"],
        "strict_success_count": candidate_ordinary["events"]["success"]
        - baseline_ordinary["events"]["success"],
        "unlatched_clearance_ge_5cm_count": candidate_ordinary["events"][
            "unlatched_clearance_ge_5cm"
        ]
        - baseline_ordinary["events"]["unlatched_clearance_ge_5cm"],
        "unsafe_force_count": candidate_ordinary["events"]["unsafe_force"]
        - baseline_ordinary["events"]["unsafe_force"],
    }
    close_delta = {
        "strict_success_rate": candidate_close["strict_success_rate"]
        - baseline_close["strict_success_rate"],
        "ever_grasped_rate": (
            candidate_close["funnel"]["ever_grasped"]
            - baseline_close["funnel"]["ever_grasped"]
        )
        / EPISODES,
        "unlatched_clearance_ge_5cm_rate": (
            candidate_close["events"]["unlatched_clearance_ge_5cm"]
            - baseline_close["events"]["unlatched_clearance_ge_5cm"]
        )
        / EPISODES,
        "unsafe_force_count": candidate_close["events"]["unsafe_force"]
        - baseline_close["events"]["unsafe_force"],
    }
    checks = {
        "offline_gate_passed": offline_gate_passed,
        "ordinary_ever_grasped_delta_ge_26": ordinary_delta["ever_grasped_count"] >= 26,
        "ordinary_success_delta_ge_0": ordinary_delta["strict_success_count"] >= 0,
        "ordinary_unlatched_5cm_delta_le_5": ordinary_delta[
            "unlatched_clearance_ge_5cm_count"
        ]
        <= 5,
        "ordinary_unsafe_force_delta_le_5": ordinary_delta["unsafe_force_count"] <= 5,
        "close_success_rate_delta_ge_minus_0_05": close_delta["strict_success_rate"]
        >= -0.05,
        "close_ever_grasped_rate_delta_ge_minus_0_05": close_delta["ever_grasped_rate"]
        >= -0.05,
        "close_unlatched_5cm_rate_delta_le_0_03": close_delta[
            "unlatched_clearance_ge_5cm_rate"
        ]
        <= 0.03,
        "close_unsafe_force_delta_le_5": close_delta["unsafe_force_count"] <= 5,
    }
    passed = all(checks.values())
    return {
        "ordinary_delta": ordinary_delta,
        "close_start_delta": close_delta,
        "checks": checks,
        "all_pass": passed,
        "decision": (
            "release_to_new_preregistered_reverse_curriculum_candidate"
            if passed
            else "reject_candidate44_retain_exact_c2"
        ),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline_ordinary", type=Path, required=True)
    parser.add_argument("--candidate_ordinary", type=Path, required=True)
    parser.add_argument("--baseline_close_start", type=Path, required=True)
    parser.add_argument("--candidate_close_start", type=Path, required=True)
    parser.add_argument("--offline_gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    offline = _load_json(args.offline_gate.expanduser().resolve())
    baseline_ordinary = _load_json(args.baseline_ordinary.expanduser().resolve())
    candidate_ordinary = _load_json(args.candidate_ordinary.expanduser().resolve())
    baseline_close = _load_json(args.baseline_close_start.expanduser().resolve())
    candidate_close = _load_json(args.candidate_close_start.expanduser().resolve())
    binding_audit = validate_offline_gate_binding(
        offline,
        baseline_ordinary,
        candidate_ordinary,
        baseline_close,
        candidate_close,
    )
    result = compare_physical_gates(
        baseline_ordinary,
        candidate_ordinary,
        baseline_close,
        candidate_close,
        offline_gate_passed=binding_audit["all_pass"],
    )
    payload = {
        "kind": "pick_tool_candidate44_paired_physical_gate_v1",
        "status": "complete",
        "offline_gate": str(args.offline_gate.expanduser().resolve()),
        "offline_gate_binding": binding_audit,
        **result,
    }
    _atomic_write_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
