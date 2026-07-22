#!/usr/bin/env python3
"""Simulation-free tests for independent-process SEARCH replay evidence."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from compare_search_replay_traces import compare_payloads, validate_trace
from search_replay_trace import (
    ACTION_DIM,
    OBSERVATION_DIM,
    POLICY_CONTRACT,
    READINESS_CONTRACT,
    TERMINAL_NAMES,
    TRACE_KIND,
    build_trace_payload,
    publish_json_no_clobber,
    publish_trace_and_report_no_clobber,
    update_readiness,
)


def _expect_error(error_type: type[BaseException], function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _metadata() -> dict[str, Any]:
    return {
        "kind": TRACE_KIND,
        "task_mode": "full_task",
        "observation_contract": "pick_tool_markov115_v1",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "terminal_names": list(TERMINAL_NAMES),
        "seed": 271,
        "num_envs": 2,
        "env_slots": [0, 1],
        "randomize_episode_lengths": False,
        "native_max_episode_steps": 1000,
        "episode_length_s": 20.0,
        "enhanced_determinism": False,
        "device": "cuda:0",
        "policy_contract": POLICY_CONTRACT,
        "readiness_contract": READINESS_CONTRACT,
        "checkpoint": "/frozen/search.pth",
        "checkpoint_sha256": "a" * 64,
        "source_sha256": {"env.py": "b" * 64, "actor.py": "c" * 64},
        "git": {"commit": "d" * 40, "branch": "replay", "source_files_dirty": False},
        "runtime": {"python": "3.11", "torch": "2.7", "cuda": "12", "cudnn": 1, "cuda_device_name": "test"},
        "collection_layout": "same_seed_same_slot_independent_process_replay_v1",
        "dependency_boundary": {"repository_inputs": "hashed", "external_simulator": "versioned"},
        "controller_target_semantics": "post-step audit only",
    }


def _rows() -> list[dict[str, torch.Tensor]]:
    rows: list[dict[str, torch.Tensor]] = []
    active_values = ((True, True), (True, True), (False, True))
    done_values = ((False, False), (True, False), (False, True))
    for step, (active_pair, done_pair) in enumerate(zip(active_values, done_values)):
        active = torch.tensor(active_pair, dtype=torch.bool)
        terminated = torch.zeros(2, dtype=torch.bool)
        truncated = torch.tensor(done_pair, dtype=torch.bool)
        terminal = torch.zeros((2, len(TERMINAL_NAMES)), dtype=torch.bool)
        terminal[:, 2] = truncated
        observation = torch.full((2, OBSERVATION_DIM), float(step), dtype=torch.float32)
        action = torch.full((2, ACTION_DIM), float(step) / 10.0, dtype=torch.float32)
        reward = torch.tensor((step + 0.1, step + 0.2), dtype=torch.float32)
        dof = torch.full((2, 19), float(step), dtype=torch.float32)
        score = torch.tensor((0.1 + step, 0.2 + step), dtype=torch.float32)
        eligible = torch.tensor((True, True), dtype=torch.bool)
        before = torch.full((2,), min(step, 3), dtype=torch.long)
        after = torch.full((2,), min(step + 1, 4), dtype=torch.long)
        trigger = after == 4
        fork_before = before == 4
        fork_after = after == 4
        stratum = torch.tensor((0, 3), dtype=torch.long)
        values: dict[str, torch.Tensor] = {
            "active": active,
            "observation": observation,
            "executed_action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "terminal_events": terminal,
            "dof_target": dof,
            "joint_pos_target": dof.clone(),
            "readiness_score": score,
            "readiness_eligible": eligible,
            "readiness_count_before": before,
            "readiness_count_after": after,
            "readiness_trigger": trigger,
            "readiness_fork_used_before": fork_before,
            "readiness_fork_used_after": fork_after,
            "readiness_stratum": stratum,
        }
        for name, value in list(values.items()):
            mask = active
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            values[name] = torch.where(mask, value, torch.zeros_like(value))
        rows.append(values)
    return rows


def _trace() -> dict[str, Any]:
    return build_trace_payload(metadata=_metadata(), rows=_rows())


def test_public_readiness_is_public_sticky_and_stratified() -> None:
    observation = torch.zeros((2, OBSERVATION_DIM), dtype=torch.float32)
    observation[:, 106] = 0.0
    observation[0, 92:96] = torch.tensor((0.8, 0.3, 0.2, 0.1))
    observation[0, 96] = 0.25
    observation[0, 102] = 0.25
    observation[1, 92:96] = 0.05
    observation[1, 96] = 1.0
    ready = torch.tensor((3, 3), dtype=torch.long)
    used = torch.tensor((False, False), dtype=torch.bool)
    result = update_readiness(observation, ready, used)
    assert torch.allclose(result["score"], torch.tensor((0.25, 0.05)))
    assert torch.equal(result["eligible"], torch.tensor((True, False)))
    assert torch.equal(result["ready_count_after"], torch.tensor((4, 0)))
    assert torch.equal(result["trigger"], torch.tensor((True, False)))
    assert torch.equal(result["fork_used_after"], torch.tensor((True, False)))
    assert torch.equal(result["stratum"], torch.tensor((1, 0)))


def test_trace_validation_and_tolerance_comparison() -> None:
    trace = _trace()
    validate_trace(trace, name="trace")
    exact = compare_payloads(trace, copy.deepcopy(trace), trace_a_sha256="1" * 64, trace_b_sha256="2" * 64)
    assert exact["status"] == "passed" and exact["bitwise_equal"]

    near = copy.deepcopy(trace)
    near["tensors"]["observation"][0, 0, 7] += 5.0e-7
    report = compare_payloads(trace, near, trace_a_sha256="1" * 64, trace_b_sha256="2" * 64)
    assert report["status"] == "passed"
    assert not report["bitwise_equal"] and report["within_fixed_tolerance"]

    far = copy.deepcopy(trace)
    far["tensors"]["reward"][0, 0] += 2.0e-5
    report = compare_payloads(trace, far, trace_a_sha256="1" * 64, trace_b_sha256="2" * 64)
    assert report["status"] == "failed"
    assert not report["continuous"]["reward"]["within_fixed_tolerance"]


def test_terminal_and_contract_mismatch_fail_closed() -> None:
    trace = _trace()
    different_outcome = copy.deepcopy(trace)
    # Keep the generic done contract valid while changing timeout into a task
    # termination at the same step.
    different_outcome["tensors"]["truncated"][1, 0] = False
    different_outcome["tensors"]["terminated"][1, 0] = True
    different_outcome["tensors"]["terminal_events"][1, 0, 2] = False
    different_outcome["tensors"]["terminal_events"][1, 0, 0] = True
    report = compare_payloads(trace, different_outcome, trace_a_sha256="1" * 64, trace_b_sha256="2" * 64)
    assert report["status"] == "failed"
    assert not report["full_terminal_match"]

    wrong_seed = copy.deepcopy(trace)
    wrong_seed["metadata"]["seed"] += 1
    _expect_error(ValueError, compare_payloads, trace, wrong_seed, trace_a_sha256="1" * 64, trace_b_sha256="2" * 64)
    dirty = copy.deepcopy(trace)
    dirty["metadata"]["git"]["source_files_dirty"] = True
    _expect_error(ValueError, compare_payloads, dirty, dirty, trace_a_sha256="1" * 64, trace_b_sha256="2" * 64)


def test_atomic_publication_is_strict_and_no_clobber() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trace_path = root / "trace.pt"
        report_path = root / "trace.json"
        sha = publish_trace_and_report_no_clobber(
            _trace(), {"status": "complete"}, trace_path, report_path
        )
        assert len(sha) == 64 and trace_path.is_file() and report_path.is_file()
        _expect_error(
            FileExistsError,
            publish_trace_and_report_no_clobber,
            _trace(),
            {"status": "changed"},
            trace_path,
            report_path,
        )
        dangling = root / "dangling.json"
        dangling.symlink_to(root / "missing")
        _expect_error(FileExistsError, publish_json_no_clobber, {"ok": True}, dangling)
        invalid = root / "invalid.json"
        _expect_error(ValueError, publish_json_no_clobber, {"bad": float("nan")}, invalid)
        assert not invalid.exists()


def test_published_trace_is_weights_only_loadable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trace_path = root / "trace.pt"
        report_path = root / "trace.json"
        publish_trace_and_report_no_clobber(
            _trace(), {"status": "complete"}, trace_path, report_path
        )
        loaded = torch.load(trace_path, map_location="cpu", weights_only=True)
        assert loaded["metadata"]["runtime"]["torch"] == "2.7"


if __name__ == "__main__":
    test_public_readiness_is_public_sticky_and_stratified()
    test_trace_validation_and_tolerance_comparison()
    test_terminal_and_contract_mismatch_fail_closed()
    test_atomic_publication_is_strict_and_no_clobber()
    test_published_trace_is_weights_only_loadable()
    print("search replay trace tests passed")
