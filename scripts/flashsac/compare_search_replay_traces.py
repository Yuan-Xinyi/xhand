#!/usr/bin/env python3
"""Compare same-seed/same-slot SEARCH traces from independent Isaac runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any, Mapping

import torch

from search_replay_trace import (
    ACTION_DIM,
    OBSERVATION_DIM,
    POLICY_CONTRACT,
    READINESS_CONTRACT,
    TERMINAL_NAMES,
    TRACE_KIND,
    publish_json_no_clobber,
    sha256_file,
)


COMPARISON_KIND = "pick_tool_search_replay_comparison_v1"
FIXED_ATOL = {
    "observation": 1.0e-6,
    "executed_action": 1.0e-7,
    "dof_target": 1.0e-6,
    "joint_pos_target": 1.0e-6,
    "reward": 1.0e-5,
    "readiness_score": 1.0e-6,
}
CONTRACT_KEYS = (
    "kind",
    "task_mode",
    "observation_contract",
    "observation_dim",
    "action_dim",
    "terminal_names",
    "seed",
    "num_envs",
    "env_slots",
    "randomize_episode_lengths",
    "native_max_episode_steps",
    "episode_length_s",
    "enhanced_determinism",
    "device",
    "policy_contract",
    "readiness_contract",
    "checkpoint_sha256",
    "source_sha256",
    "git",
    "runtime",
    "collection_layout",
    "dependency_boundary",
    "controller_target_semantics",
)
DISCRETE_FIELDS = (
    "active",
    "terminated",
    "truncated",
    "terminal_events",
    "readiness_eligible",
    "readiness_count_before",
    "readiness_count_after",
    "readiness_trigger",
    "readiness_fork_used_before",
    "readiness_fork_used_after",
    "readiness_stratum",
)
CONTINUOUS_FIELDS = tuple(FIXED_ATOL)


def _load_trace(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"trace must be a regular, non-symlink file: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"metadata", "tensors"}:
        raise TypeError("trace root must contain exactly metadata and tensors")
    return payload


def validate_trace(payload: Mapping[str, Any], *, name: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    metadata_raw = payload.get("metadata")
    tensors_raw = payload.get("tensors")
    if not isinstance(metadata_raw, Mapping) or not isinstance(tensors_raw, Mapping):
        raise TypeError(f"{name} metadata/tensors must be mappings")
    metadata = dict(metadata_raw)
    json.dumps(metadata, sort_keys=True, allow_nan=False)
    if metadata.get("kind") != TRACE_KIND:
        raise ValueError(f"{name} has an unsupported trace kind")
    if metadata.get("observation_dim") != OBSERVATION_DIM or metadata.get("action_dim") != ACTION_DIM:
        raise ValueError(f"{name} violates the 115/21 contract")
    if metadata.get("terminal_names") != list(TERMINAL_NAMES):
        raise ValueError(f"{name} terminal event ordering changed")
    if metadata.get("policy_contract") != POLICY_CONTRACT:
        raise ValueError(f"{name} policy contract changed")
    if metadata.get("readiness_contract") != READINESS_CONTRACT:
        raise ValueError(f"{name} readiness contract changed")
    checkpoint_sha = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha
    ):
        raise ValueError(f"{name} has an invalid checkpoint SHA256")
    source_sha = metadata.get("source_sha256")
    if not isinstance(source_sha, Mapping) or not source_sha:
        raise ValueError(f"{name} has no source fingerprints")
    for path, digest in source_sha.items():
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} has an invalid source fingerprint")
    num_envs = int(metadata.get("num_envs", -1))
    if num_envs < 1 or metadata.get("env_slots") != list(range(num_envs)):
        raise ValueError(f"{name} has invalid environment slots")
    required = set(DISCRETE_FIELDS) | set(CONTINUOUS_FIELDS)
    if set(tensors_raw) != required:
        missing = sorted(required - set(tensors_raw))
        extra = sorted(set(tensors_raw) - required)
        raise ValueError(f"{name} tensor fields changed: missing={missing}, extra={extra}")
    tensors: dict[str, torch.Tensor] = {}
    steps: int | None = None
    for field in sorted(required):
        value = tensors_raw[field]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise TypeError(f"{name}.{field} must be a CPU tensor")
        if value.ndim < 2 or value.shape[1] != num_envs:
            raise ValueError(f"{name}.{field} must be step-major [T,N,...]")
        if steps is None:
            steps = int(value.shape[0])
        elif value.shape[0] != steps:
            raise ValueError(f"{name} tensor step counts disagree")
        if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name}.{field} contains NaN or infinity")
        tensors[field] = value
    assert steps is not None
    expected_shapes = {
        "active": (steps, num_envs),
        "observation": (steps, num_envs, OBSERVATION_DIM),
        "executed_action": (steps, num_envs, ACTION_DIM),
        "reward": (steps, num_envs),
        "terminated": (steps, num_envs),
        "truncated": (steps, num_envs),
        "terminal_events": (steps, num_envs, len(TERMINAL_NAMES)),
        "readiness_score": (steps, num_envs),
        "readiness_eligible": (steps, num_envs),
        "readiness_count_before": (steps, num_envs),
        "readiness_count_after": (steps, num_envs),
        "readiness_trigger": (steps, num_envs),
        "readiness_fork_used_before": (steps, num_envs),
        "readiness_fork_used_after": (steps, num_envs),
        "readiness_stratum": (steps, num_envs),
    }
    bool_fields = {
        "active", "terminated", "truncated", "terminal_events",
        "readiness_eligible", "readiness_trigger",
        "readiness_fork_used_before", "readiness_fork_used_after",
    }
    long_fields = {"readiness_count_before", "readiness_count_after", "readiness_stratum"}
    for field, value in tensors.items():
        expected = expected_shapes.get(field)
        if expected is not None and tuple(value.shape) != expected:
            raise ValueError(f"{name}.{field} has shape {tuple(value.shape)}, expected {expected}")
        if field in {"dof_target", "joint_pos_target"} and (value.ndim != 3 or value.shape[2] < 1):
            raise ValueError(f"{name}.{field} must be [T,N,D]")
        expected_dtype = torch.bool if field in bool_fields else torch.long if field in long_fields else torch.float32
        if value.dtype != expected_dtype:
            raise TypeError(f"{name}.{field} must be {expected_dtype}, got {value.dtype}")
    active = tensors["active"]
    done = tensors["terminated"] | tensors["truncated"]
    if steps < 1 or bool((active[-1] & (~done[-1])).any()):
        raise ValueError(f"{name} is not a complete first-episode trace")
    if steps > 1 and not torch.equal(active[1:], active[:-1] & (~done[:-1])):
        raise ValueError(f"{name} active mask does not follow prior done masks")
    if bool((done & (~active)).any()) or bool((tensors["terminated"] & tensors["truncated"]).any()):
        raise ValueError(f"{name} has an invalid done mask")
    terminal = tensors["terminal_events"]
    if bool((terminal & (~active.unsqueeze(-1))).any()):
        raise ValueError(f"{name} has terminal truth outside its first-episode mask")
    if not torch.equal(terminal[..., 0] | terminal[..., 1], tensors["terminated"]):
        raise ValueError(f"{name} task termination aliases disagree")
    if not torch.equal(terminal[..., 2], tensors["truncated"]):
        raise ValueError(f"{name} timeout alias disagrees")
    if bool((terminal[..., 0] & terminal[..., 1]).any()):
        raise ValueError(f"{name} overlaps success and failure")
    failure_sources = terminal[..., 3] | terminal[..., 4] | terminal[..., 5]
    if not torch.equal(terminal[..., 1], failure_sources):
        raise ValueError(f"{name} full-task failure sources disagree")
    return metadata, tensors


def _first_mismatch(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    mismatch = a != b
    if not bool(mismatch.any()):
        return None
    index = mismatch.nonzero(as_tuple=False)[0].tolist()
    result: dict[str, Any] = {"index": index, "a": a[tuple(index)].item(), "b": b[tuple(index)].item()}
    if len(index) >= 2:
        result.update({"step_one_based": index[0] + 1, "env_slot": index[1]})
    return result


def _continuous_comparison(a: torch.Tensor, b: torch.Tensor, atol: float) -> dict[str, Any]:
    difference = (a - b).abs()
    max_abs = float(difference.max()) if difference.numel() else 0.0
    mismatch = difference > atol
    first = None
    if bool(mismatch.any()):
        index = mismatch.nonzero(as_tuple=False)[0].tolist()
        first = {
            "index": index,
            "step_one_based": index[0] + 1,
            "env_slot": index[1],
            "a": float(a[tuple(index)]),
            "b": float(b[tuple(index)]),
            "abs_error": float(difference[tuple(index)]),
        }
    return {
        "bitwise_equal": torch.equal(a, b),
        "fixed_atol": atol,
        "within_fixed_tolerance": bool(max_abs <= atol),
        "max_abs_error": max_abs,
        "values_above_tolerance": int(mismatch.sum()),
        "first_above_tolerance": first,
    }


def compare_payloads(
    trace_a: Mapping[str, Any],
    trace_b: Mapping[str, Any],
    *,
    trace_a_sha256: str,
    trace_b_sha256: str,
) -> dict[str, Any]:
    metadata_a, tensors_a = validate_trace(trace_a, name="trace_a")
    metadata_b, tensors_b = validate_trace(trace_b, name="trace_b")
    for name, metadata in (("trace_a", metadata_a), ("trace_b", metadata_b)):
        git = metadata.get("git")
        if not isinstance(git, Mapping) or git.get("source_files_dirty") is not False:
            raise ValueError(f"{name} was collected from dirty or unverified source inputs")
    contract_mismatches = {
        key: {"a": metadata_a.get(key), "b": metadata_b.get(key)}
        for key in CONTRACT_KEYS
        if metadata_a.get(key) != metadata_b.get(key)
    }
    if contract_mismatches:
        raise ValueError(f"trace contracts differ: {sorted(contract_mismatches)}")
    shape_mismatches = {
        field: {"a": list(tensors_a[field].shape), "b": list(tensors_b[field].shape)}
        for field in tensors_a
        if tensors_a[field].shape != tensors_b[field].shape
    }
    if shape_mismatches:
        raise ValueError(f"trace tensor shapes differ: {shape_mismatches}")

    discrete: dict[str, Any] = {}
    discrete_equal = True
    for field in DISCRETE_FIELDS:
        equal = torch.equal(tensors_a[field], tensors_b[field])
        discrete[field] = {
            "equal": equal,
            "mismatch_values": int((tensors_a[field] != tensors_b[field]).sum()),
            "first_mismatch": _first_mismatch(tensors_a[field], tensors_b[field]),
        }
        discrete_equal &= equal
    continuous = {
        field: _continuous_comparison(tensors_a[field], tensors_b[field], FIXED_ATOL[field])
        for field in CONTINUOUS_FIELDS
    }
    continuous_within = all(entry["within_fixed_tolerance"] for entry in continuous.values())
    terminal_exact = all(
        discrete[field]["equal"]
        for field in ("active", "terminated", "truncated", "terminal_events")
    )
    bitwise_equal = discrete_equal and all(entry["bitwise_equal"] for entry in continuous.values())
    passed = terminal_exact and discrete_equal and continuous_within
    terminal_by_name = {}
    terminal_a = tensors_a["terminal_events"]
    terminal_b = tensors_b["terminal_events"]
    for index, name in enumerate(TERMINAL_NAMES):
        a_value = terminal_a[..., index]
        b_value = terminal_b[..., index]
        terminal_by_name[name] = {
            "a_total": int(a_value.sum()),
            "b_total": int(b_value.sum()),
            "mismatch_values": int((a_value != b_value).sum()),
        }
    return {
        "kind": COMPARISON_KIND,
        "status": "passed" if passed else "failed",
        "evidence_eligible": passed,
        "comparison_layout": "same_seed_same_slot_across_independent_processes_v1",
        "seed": metadata_a["seed"],
        "num_envs": metadata_a["num_envs"],
        "steps": int(tensors_a["active"].shape[0]),
        "trace_a_sha256": trace_a_sha256,
        "trace_b_sha256": trace_b_sha256,
        "checkpoint_sha256": metadata_a["checkpoint_sha256"],
        "source_sha256": metadata_a["source_sha256"],
        "policy_contract": metadata_a["policy_contract"],
        "controller_target_semantics": metadata_a["controller_target_semantics"],
        "contract_keys_compared": list(CONTRACT_KEYS),
        "fixed_tolerances": dict(FIXED_ATOL),
        "bitwise_equal": bitwise_equal,
        "within_fixed_tolerance": continuous_within,
        "discrete_equal": discrete_equal,
        "full_terminal_match": terminal_exact,
        "terminal_by_name": terminal_by_name,
        "discrete": discrete,
        "continuous": continuous,
        "comparator_sha256": sha256_file(Path(__file__).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trace_a", "--trace-a", dest="trace_a", type=Path, required=True)
    parser.add_argument("--trace_b", "--trace-b", dest="trace_b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trace_a.resolve(strict=False) == args.trace_b.resolve(strict=False):
        parser.error("--trace_a and --trace_b must be different artifacts")
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output already exists; comparison evidence is immutable")
    return args


def main() -> int:
    args = parse_args()
    try:
        trace_a = _load_trace(args.trace_a)
        trace_b = _load_trace(args.trace_b)
        report = compare_payloads(
            trace_a,
            trace_b,
            trace_a_sha256=sha256_file(args.trace_a),
            trace_b_sha256=sha256_file(args.trace_b),
        )
        report["trace_a"] = str(args.trace_a.resolve())
        report["trace_b"] = str(args.trace_b.resolve())
    except Exception as error:
        publish_json_no_clobber(
            {
                "kind": COMPARISON_KIND,
                "status": "failed",
                "evidence_eligible": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "trace_a": str(args.trace_a.absolute()),
                "trace_b": str(args.trace_b.absolute()),
                "traceback": traceback.format_exc(),
                "comparator_sha256": sha256_file(Path(__file__).resolve()),
            },
            args.output,
        )
        raise
    publish_json_no_clobber(report, args.output)
    print(
        "[compare-search-replay] "
        f"status={report['status']} bitwise={report['bitwise_equal']} "
        f"terminal_match={report['full_terminal_match']}",
        flush=True,
    )
    return 0 if report["evidence_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
