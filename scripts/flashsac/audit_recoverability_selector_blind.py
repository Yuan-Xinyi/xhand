#!/usr/bin/env python3
"""CPU-only, fail-closed audit of recoverability-selector blind evidence.

The offline arm replays the signed selector on strong paired handoff rows.  The
actual arm compares only top-level rollout event counts from independently run
fixed-route and selector JSON reports.  Acceptance thresholds are constants on
purpose: this program does not expose post-blind threshold tuning flags.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from recoverability_selector_runtime import (
    DATASET_KIND,
    RecoverabilitySelectorRuntime,
    load_recoverability_selector,
    sha256_file,
)
from train_recoverability_selector import ACTION_DIM, OBSERVATION_DIM, policy_metrics


AUDIT_KIND = "pick_tool_recoverability_selector_blind_audit_v1"
AUDIT_FORMAT_VERSION = 1

# Transcribed unchanged from the criteria registered before blind seeds
# 261--264 were run.  Deliberately not CLI options: this post-run audit must not
# turn those criteria into tunable knobs.
OFFLINE_MIN_AGGREGATE_SUCCESS_DELTA = 10
OFFLINE_MIN_ROUTE_GAIN_RETENTION = 0.75
OFFLINE_MIN_ROUTE_REGRESSION_AVOIDANCE = 0.25
OFFLINE_MAX_CONTINUE_RATE = 0.25
ACTUAL_MIN_AGGREGATE_SUCCESS_DELTA = 10

_OFFLINE_BOOL_FIELDS = (
    "paired_outcome_label_valid",
    "strong_pair",
    "continue_success",
    "route_success",
    "continue_dropped",
    "route_dropped",
    "continue_unsafe_force",
    "route_unsafe_force",
)
_EVENT_FIELDS = ("success", "dropped", "unsafe_force")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _require_regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    source = _require_regular_file(path, label)
    digest = sha256_file(source)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(
            stream,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    if sha256_file(source) != digest:
        raise RuntimeError(f"{label} changed while it was being loaded")
    return payload, digest


def _require_bool_tensor(
    dataset: Mapping[str, Any], name: str, rows: int
) -> torch.Tensor:
    value = dataset.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (rows,):
        raise ValueError(f"dataset {name} must have shape ({rows},)")
    if value.dtype != torch.bool:
        raise TypeError(f"dataset {name} must be torch.bool")
    return value.to(device="cpu").contiguous()


def _require_float_matrix(
    dataset: Mapping[str, Any], name: str, rows: int, columns: int
) -> torch.Tensor:
    value = dataset.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (rows, columns):
        raise ValueError(f"dataset {name} must have shape ({rows}, {columns})")
    if value.dtype != torch.float32:
        raise TypeError(f"dataset {name} must be torch.float32")
    value = value.to(device="cpu").contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"dataset {name} contains NaN or infinity")
    return value


def load_offline_dataset(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load and validate the aggregate paired artifact on CPU."""

    source = _require_regular_file(path, "aggregate paired dataset")
    digest = sha256_file(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if sha256_file(source) != digest:
        raise RuntimeError("aggregate paired dataset changed while it was being loaded")
    if not isinstance(payload, Mapping):
        raise TypeError("aggregate paired dataset must be a mapping")
    if payload.get("kind") != DATASET_KIND or payload.get("format_version") != 1:
        raise ValueError("unsupported aggregate paired dataset kind or version")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("aggregate paired dataset metadata must be a mapping")
    if (
        metadata.get("pairing_semantics")
        != "independent_gpu_rollout_diagnostic_v1"
        or metadata.get("causal_counterfactual_claim_allowed") is not False
    ):
        raise ValueError("aggregate paired dataset has incompatible audit semantics")

    seed_value = payload.get("seed")
    if not isinstance(seed_value, torch.Tensor) or seed_value.ndim != 1:
        raise ValueError("dataset seed must be a one-dimensional tensor")
    if seed_value.dtype == torch.bool or seed_value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("dataset seed must have an integer dtype")
    seeds = seed_value.to(device="cpu", dtype=torch.int64).contiguous()
    rows = int(seeds.numel())
    if rows < 1:
        raise ValueError("aggregate paired dataset has no rows")
    unique_seeds = sorted(set(int(value) for value in seeds.tolist()))
    if len(unique_seeds) < 2:
        raise ValueError("aggregate paired dataset must contain at least two seed groups")
    metadata_seeds = metadata.get("seeds")
    if metadata_seeds != unique_seeds:
        raise ValueError("dataset metadata seed groups disagree with the seed tensor")

    tensors: dict[str, torch.Tensor] = {
        "seed": seeds,
        "observation": _require_float_matrix(
            payload, "observation", rows, OBSERVATION_DIM
        ),
        "search_action": _require_float_matrix(
            payload, "search_action", rows, ACTION_DIM
        ),
        "flashsac_action": _require_float_matrix(
            payload, "flashsac_action", rows, ACTION_DIM
        ),
    }
    for name in _OFFLINE_BOOL_FIELDS:
        tensors[name] = _require_bool_tensor(payload, name, rows)
    valid = tensors["paired_outcome_label_valid"] & tensors["strong_pair"]
    if not bool(valid.any()):
        raise ValueError("aggregate paired dataset has no valid strong rows")
    if "strong_rows" in metadata and metadata.get("strong_rows") != int(
        tensors["strong_pair"].sum()
    ):
        raise ValueError("dataset metadata strong_rows disagrees with tensors")

    provenance = {
        "path": str(source),
        "sha256": digest,
        "rows": rows,
        "valid_strong_rows": int(valid.sum()),
        "excluded_rows": int((~valid).sum()),
        "seeds": unique_seeds,
        "mask_semantics": "paired_outcome_label_valid AND strong_pair",
    }
    return tensors, provenance


def _metric_bundle(
    select_continue: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Call the frozen policy metric and add non-cancelling safety counts."""

    metrics = policy_metrics(
        select_continue,
        tensors["continue_success"],
        tensors["route_success"],
        tensors["continue_dropped"],
        tensors["route_dropped"],
        tensors["continue_unsafe_force"],
        tensors["route_unsafe_force"],
    )
    select_continue = select_continue.to(torch.bool)
    chosen_drop = torch.where(
        select_continue, tensors["continue_dropped"], tensors["route_dropped"]
    )
    chosen_unsafe = torch.where(
        select_continue,
        tensors["continue_unsafe_force"],
        tensors["route_unsafe_force"],
    )
    added_drop = int((chosen_drop & ~tensors["route_dropped"]).sum())
    avoided_drop = int((~chosen_drop & tensors["route_dropped"]).sum())
    added_unsafe = int((chosen_unsafe & ~tensors["route_unsafe_force"]).sum())
    avoided_unsafe = int((~chosen_unsafe & tensors["route_unsafe_force"]).sum())
    if metrics["drop_delta_vs_route"] != added_drop - avoided_drop:
        raise RuntimeError("drop accounting is internally inconsistent")
    if metrics["unsafe_force_delta_vs_route"] != added_unsafe - avoided_unsafe:
        raise RuntimeError("unsafe-force accounting is internally inconsistent")
    rows = int(select_continue.numel())
    return {
        **metrics,
        "continue_decision_rate": int(select_continue.sum()) / rows,
        "added_drops_vs_route": added_drop,
        "avoided_drops_vs_route": avoided_drop,
        "added_unsafe_force_vs_route": added_unsafe,
        "avoided_unsafe_force_vs_route": avoided_unsafe,
        "non_cancelling_safety_semantics": (
            "added counts are policy events absent under route; avoided events are "
            "reported separately and never offset acceptance"
        ),
    }


def audit_offline(
    tensors: Mapping[str, torch.Tensor],
    runtime: RecoverabilitySelectorRuntime,
) -> dict[str, Any]:
    """Replay the signed selector on only valid, strong paired rows."""

    mask = tensors["paired_outcome_label_valid"] & tensors["strong_pair"]
    selected = {name: value[mask] for name, value in tensors.items()}
    decision = runtime.decide(
        selected["observation"],
        selected["search_action"],
        selected["flashsac_action"],
    )
    if decision.continue_search.device.type != "cpu":
        raise RuntimeError("selector audit decision escaped the CPU")
    if tuple(decision.continue_search.shape) != (int(mask.sum()),):
        raise ValueError("selector returned the wrong number of decisions")
    if not torch.equal(decision.route_flashsac, ~decision.continue_search):
        raise ValueError("selector decisions are not complementary")

    aggregate = _metric_bundle(decision.continue_search, selected)
    by_seed: dict[str, Any] = {}
    seed_tensor = selected["seed"]
    for seed in sorted(set(int(value) for value in seed_tensor.tolist())):
        seed_mask = seed_tensor == seed
        seed_tensors = {name: value[seed_mask] for name, value in selected.items()}
        by_seed[str(seed)] = _metric_bundle(
            decision.continue_search[seed_mask], seed_tensors
        )

    checks = {
        "every_seed_success_delta_nonnegative": {
            "threshold": 0,
            "observed_minimum": min(
                item["net_success_delta_vs_route"] for item in by_seed.values()
            ),
            "pass": all(
                item["net_success_delta_vs_route"] >= 0
                for item in by_seed.values()
            ),
        },
        "aggregate_success_delta": {
            "threshold": OFFLINE_MIN_AGGREGATE_SUCCESS_DELTA,
            "observed": aggregate["net_success_delta_vs_route"],
            "pass": aggregate["net_success_delta_vs_route"]
            >= OFFLINE_MIN_AGGREGATE_SUCCESS_DELTA,
        },
        "aggregate_route_gain_retention": {
            "threshold": OFFLINE_MIN_ROUTE_GAIN_RETENTION,
            "observed": aggregate["route_gain_retention_rate"],
            "pass": aggregate["route_gain_retention_rate"]
            >= OFFLINE_MIN_ROUTE_GAIN_RETENTION,
        },
        "aggregate_route_regression_avoidance": {
            "threshold": OFFLINE_MIN_ROUTE_REGRESSION_AVOIDANCE,
            "observed": aggregate["route_regression_avoidance_rate"],
            "pass": aggregate["route_regression_avoidance_rate"]
            >= OFFLINE_MIN_ROUTE_REGRESSION_AVOIDANCE,
        },
        "no_added_drops": {
            "threshold": 0,
            "observed": aggregate["added_drops_vs_route"],
            "pass": aggregate["added_drops_vs_route"] == 0,
        },
        "no_added_unsafe_force": {
            "threshold": 0,
            "observed": aggregate["added_unsafe_force_vs_route"],
            "pass": aggregate["added_unsafe_force_vs_route"] == 0,
        },
        "aggregate_continue_decision_rate": {
            "threshold": OFFLINE_MAX_CONTINUE_RATE,
            "observed": aggregate["continue_decision_rate"],
            "pass": aggregate["continue_decision_rate"] <= OFFLINE_MAX_CONTINUE_RATE,
        },
    }
    return {
        "status": "pass" if all(item["pass"] for item in checks.values()) else "fail",
        "criteria_pass": all(item["pass"] for item in checks.values()),
        "eligible_rows": int(mask.sum()),
        "aggregate": aggregate,
        "by_seed": by_seed,
        "checks": checks,
    }


def _require_event_count(events: Mapping[str, Any], name: str, label: str) -> int:
    value = events.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} events.{name} must be a non-negative integer")
    return value


def _rollout_identity(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    if payload.get("status") != "complete":
        raise ValueError(f"{label} rollout is incomplete")
    result: dict[str, Any] = {}
    for name in ("seed", "requested_episodes", "num_envs", "completed_episodes"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} rollout has invalid {name}")
        result[name] = value
    if result["completed_episodes"] != result["requested_episodes"]:
        raise ValueError(f"{label} rollout did not complete every requested episode")
    for name in ("task_mode", "observation_contract", "max_episode_steps"):
        result[name] = payload.get(name)
    events = payload.get("events")
    if not isinstance(events, Mapping):
        raise TypeError(f"{label} rollout lacks top-level events")
    result["events"] = {
        name: _require_event_count(events, name, label) for name in _EVENT_FIELDS
    }
    for terminal_name in ("failure", "time_out"):
        result["events"][terminal_name] = _require_event_count(
            events, terminal_name, label
        )
    if sum(result["events"][name] for name in ("success", "failure", "time_out")) != result[
        "completed_episodes"
    ]:
        raise ValueError(f"{label} terminal event counts do not cover completed episodes")
    return result


def _validate_actual_pair(
    route: Mapping[str, Any],
    selector: Mapping[str, Any],
    runtime: RecoverabilitySelectorRuntime,
) -> tuple[int, dict[str, Any]]:
    route_identity = _rollout_identity(route, "fixed-route")
    selector_identity = _rollout_identity(selector, "selector")
    comparison_fields = (
        "seed",
        "requested_episodes",
        "num_envs",
        "completed_episodes",
        "task_mode",
        "observation_contract",
        "max_episode_steps",
    )
    mismatches = [
        name
        for name in comparison_fields
        if route_identity[name] != selector_identity[name]
    ]
    if mismatches:
        raise ValueError(f"actual rollout pair differs on {mismatches}")
    seed = int(route_identity["seed"])

    route_hierarchy = route.get("diagnostic_approach_hierarchy")
    selector_hierarchy = selector.get("diagnostic_approach_hierarchy")
    if not isinstance(route_hierarchy, Mapping) or not isinstance(
        selector_hierarchy, Mapping
    ):
        raise TypeError("actual rollout pair lacks hierarchy audit metadata")
    if route_hierarchy.get("mode") != "frozen_rlgames_search_then_flashsac_close_lift":
        raise ValueError("route JSON is not the fixed-route baseline")
    if "recoverability_selector" in route_hierarchy:
        raise ValueError("fixed-route baseline unexpectedly contains a selector")
    if selector_hierarchy.get("mode") != (
        "frozen_rlgames_search_recoverability_selector_then_flashsac_close_lift"
    ):
        raise ValueError("selector JSON is not a recoverability-selector rollout")
    selector_audit = selector_hierarchy.get("recoverability_selector")
    if not isinstance(selector_audit, Mapping):
        raise TypeError("selector rollout lacks recoverability selector audit metadata")
    blind = selector_audit.get("blind_evaluation")
    if not isinstance(blind, Mapping):
        raise TypeError("selector rollout lacks blind evaluation metadata")
    required_blind_values = {
        "blind_requested": True,
        "blind_claim_allowed": True,
        "seen_seed": False,
        "scale_matches_collection": True,
    }
    for name, expected in required_blind_values.items():
        if blind.get(name) is not expected:
            raise ValueError(f"selector blind audit requires {name}={expected}")
    if (
        selector_hierarchy.get("blind_claim_allowed") is not True
        or selector_audit.get("blind_claim_allowed") is not True
    ):
        raise ValueError("selector rollout does not allow a strict blind claim")
    if selector_audit.get("deployment_claim_allowed") is not False:
        raise ValueError("selector rollout must still forbid deployment claims")
    if blind.get("evaluation_seed") != seed:
        raise ValueError("selector blind audit seed disagrees with rollout seed")
    if (
        blind.get("evaluation_episodes") != selector_identity["requested_episodes"]
        or blind.get("evaluation_num_envs") != selector_identity["num_envs"]
    ):
        raise ValueError("selector blind scale metadata disagrees with rollout scale")
    if (
        selector_audit.get("model_sha256") != runtime.model_sha256
        or selector_audit.get("report_sha256") != runtime.report_sha256
    ):
        raise ValueError("actual selector rollout used different signed model/report bytes")
    expected_blind = runtime.validate_blind_configuration(
        seed=seed,
        episodes=selector_identity["requested_episodes"],
        num_envs=selector_identity["num_envs"],
        strict_blind=True,
    )
    for name in (
        "blind_requested",
        "blind_claim_allowed",
        "seen_seed",
        "scale_matches_collection",
        "evaluation_seed",
        "evaluation_episodes",
        "evaluation_num_envs",
    ):
        if blind.get(name) != expected_blind[name]:
            raise ValueError(
                f"selector serialized blind audit disagrees with signed model on {name}"
            )

    route_events = route_identity["events"]
    selector_events = selector_identity["events"]
    metrics = {
        "episodes": route_identity["completed_episodes"],
        "route_events": route_events,
        "selector_events": selector_events,
        "success_delta_vs_route": selector_events["success"] - route_events["success"],
        "drop_delta_vs_route": selector_events["dropped"] - route_events["dropped"],
        "unsafe_force_delta_vs_route": (
            selector_events["unsafe_force"] - route_events["unsafe_force"]
        ),
    }
    return seed, metrics


def audit_actual(
    actual_pairs: Sequence[tuple[Path, Path]],
    runtime: RecoverabilitySelectorRuntime,
) -> dict[str, Any]:
    """Audit paired fixed-route/selector JSONs using top-level events only."""

    if not actual_pairs:
        raise ValueError("at least one actual rollout pair is required")
    by_seed: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    for route_path, selector_path in actual_pairs:
        route_source = _require_regular_file(route_path, "fixed-route JSON")
        selector_source = _require_regular_file(selector_path, "selector JSON")
        route, route_digest = _load_json(route_source, "fixed-route JSON")
        selector, selector_digest = _load_json(selector_source, "selector JSON")
        seed, metrics = _validate_actual_pair(route, selector, runtime)
        if str(seed) in by_seed:
            raise ValueError(f"duplicate actual rollout seed {seed}")
        by_seed[str(seed)] = metrics
        sources.append(
            {
                "seed": seed,
                "route": str(route_source),
                "route_sha256": route_digest,
                "selector": str(selector_source),
                "selector_sha256": selector_digest,
            }
        )
    by_seed = {key: by_seed[key] for key in sorted(by_seed, key=int)}
    sources.sort(key=lambda item: item["seed"])
    aggregate = {
        "seeds": len(by_seed),
        "episodes_per_arm": sum(item["episodes"] for item in by_seed.values()),
        "route_events": {
            name: sum(item["route_events"][name] for item in by_seed.values())
            for name in (*_EVENT_FIELDS, "failure", "time_out")
        },
        "selector_events": {
            name: sum(item["selector_events"][name] for item in by_seed.values())
            for name in (*_EVENT_FIELDS, "failure", "time_out")
        },
        "success_delta_vs_route": sum(
            item["success_delta_vs_route"] for item in by_seed.values()
        ),
        "drop_delta_vs_route": sum(
            item["drop_delta_vs_route"] for item in by_seed.values()
        ),
        "unsafe_force_delta_vs_route": sum(
            item["unsafe_force_delta_vs_route"] for item in by_seed.values()
        ),
        "non_cancelling_added_counts_available": False,
        "safety_note": (
            "actual arms are independent rollouts and this audit intentionally reads "
            "only top-level events; the preregistered actual criterion is aggregate delta"
        ),
    }
    checks = {
        "every_seed_success_delta_nonnegative": {
            "threshold": 0,
            "observed_minimum": min(
                item["success_delta_vs_route"] for item in by_seed.values()
            ),
            "pass": all(
                item["success_delta_vs_route"] >= 0 for item in by_seed.values()
            ),
        },
        "aggregate_success_delta": {
            "threshold": ACTUAL_MIN_AGGREGATE_SUCCESS_DELTA,
            "observed": aggregate["success_delta_vs_route"],
            "pass": aggregate["success_delta_vs_route"]
            >= ACTUAL_MIN_AGGREGATE_SUCCESS_DELTA,
        },
        "aggregate_drop_delta": {
            "threshold": 0,
            "observed": aggregate["drop_delta_vs_route"],
            "pass": aggregate["drop_delta_vs_route"] <= 0,
        },
        "aggregate_unsafe_force_delta": {
            "threshold": 0,
            "observed": aggregate["unsafe_force_delta_vs_route"],
            "pass": aggregate["unsafe_force_delta_vs_route"] <= 0,
        },
    }
    return {
        "status": "pass" if all(item["pass"] for item in checks.values()) else "fail",
        "criteria_pass": all(item["pass"] for item in checks.values()),
        "aggregate": aggregate,
        "by_seed": by_seed,
        "checks": checks,
        "sources": sources,
    }


def build_audit_report(
    *,
    dataset_path: Path,
    model_path: Path,
    report_path: Path,
    actual_pairs: Sequence[tuple[Path, Path]],
) -> dict[str, Any]:
    """Validate every artifact, run both frozen audits, and return strict JSON."""

    runtime = load_recoverability_selector(
        Path(model_path), Path(report_path), device=torch.device("cpu")
    )
    audit_source = _require_regular_file(Path(__file__), "blind audit source")
    audit_source_sha256 = sha256_file(audit_source)
    if runtime.feature_mean.device.type != "cpu":
        raise RuntimeError("signed selector did not load on CPU")
    runtime.verify_unchanged()
    if sha256_file(audit_source) != audit_source_sha256:
        raise RuntimeError("blind audit source changed while the report was built")
    tensors, dataset_provenance = load_offline_dataset(dataset_path)
    offline = audit_offline(tensors, runtime)
    actual = audit_actual(actual_pairs, runtime)
    offline_seeds = set(offline["by_seed"])
    actual_seeds = set(actual["by_seed"])
    if offline_seeds != actual_seeds:
        raise ValueError(
            "offline and actual blind seed groups differ: "
            f"offline={sorted(offline_seeds)}, actual={sorted(actual_seeds)}"
        )
    runtime.verify_unchanged()
    if sha256_file(audit_source) != audit_source_sha256:
        raise RuntimeError("blind audit source changed while the report was built")
    overall = bool(offline["criteria_pass"] and actual["criteria_pass"])
    result = {
        "status": "pass" if overall else "fail",
        "overall_pass": overall,
        "kind": AUDIT_KIND,
        "format_version": AUDIT_FORMAT_VERSION,
        "execution_device": "cpu",
        "thresholds_frozen_in_source": {
            "offline_every_seed_success_delta_min": 0,
            "offline_aggregate_success_delta_min": OFFLINE_MIN_AGGREGATE_SUCCESS_DELTA,
            "offline_route_gain_retention_min": OFFLINE_MIN_ROUTE_GAIN_RETENTION,
            "offline_route_regression_avoidance_min": (
                OFFLINE_MIN_ROUTE_REGRESSION_AVOIDANCE
            ),
            "offline_added_drop_max": 0,
            "offline_added_unsafe_force_max": 0,
            "offline_continue_decision_rate_max": OFFLINE_MAX_CONTINUE_RATE,
            "actual_every_seed_success_delta_min": 0,
            "actual_aggregate_success_delta_min": ACTUAL_MIN_AGGREGATE_SUCCESS_DELTA,
            "actual_aggregate_drop_delta_max": 0,
            "actual_aggregate_unsafe_force_delta_max": 0,
        },
        "criteria_origin": (
            "criteria preregistered before seeds 261--264; this CPU audit was "
            "implemented after collection and transcribes them without CLI overrides"
        ),
        "artifacts": {
            "audit_source": str(audit_source),
            "audit_source_sha256": audit_source_sha256,
            "aggregate_dataset": dataset_provenance,
            "selector_model": str(runtime.model_path),
            "selector_model_sha256": runtime.model_sha256,
            "selector_report": str(runtime.report_path),
            "selector_report_sha256": runtime.report_sha256,
            "selector_trainer_source_sha256": runtime.trainer_source_sha256,
            "selector_runtime_source_sha256": runtime.runtime_source_sha256,
        },
        "offline_paired": offline,
        "actual_rollouts": actual,
        "claim_scope": {
            "blind_diagnostic_claim_allowed": True,
            "deployment_claim_allowed": False,
            "reason": (
                "the selector remains behind a private non-Markov diagnostic "
                "candidate supervisor"
            ),
        },
    }
    # Reject NaN/infinity before publication.
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def publish_json_no_clobber(payload: Mapping[str, Any], output: Path) -> None:
    """Publish strict JSON atomically without replacing any existing path."""

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, output)
    finally:
        staging.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--actual-pair",
        "--actual_pair",
        action="append",
        nargs=2,
        type=Path,
        required=True,
        metavar=("ROUTE_JSON", "SELECTOR_JSON"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_audit_report(
        dataset_path=args.dataset,
        model_path=args.model,
        report_path=args.report,
        actual_pairs=[tuple(pair) for pair in args.actual_pair],
    )
    publish_json_no_clobber(report, args.output)
    print(
        f"blind selector audit: {report['status']} "
        f"(offline={report['offline_paired']['status']}, "
        f"actual={report['actual_rollouts']['status']})"
    )
    print(f"report: {Path(args.output).resolve()}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
