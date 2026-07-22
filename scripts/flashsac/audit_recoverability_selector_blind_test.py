#!/usr/bin/env python3
"""Simulation-free tests for the frozen blind-selector audit."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from typing import Any

import torch

from audit_recoverability_selector_blind import (
    audit_actual,
    audit_offline,
    build_audit_report,
    load_offline_dataset,
    publish_json_no_clobber,
)
from recoverability_selector_runtime import load_recoverability_selector
from recoverability_selector_runtime_test import _publish


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _dataset(seeds: tuple[int, ...] = (261, 262, 263, 264)) -> dict[str, Any]:
    per_seed = 20
    rows = len(seeds) * per_seed
    seed = torch.repeat_interleave(torch.tensor(seeds, dtype=torch.int64), per_seed)
    observation = torch.zeros((rows, 115), dtype=torch.float32)
    observation[:, 59] = 1.0
    search_action = torch.full((rows, 21), -1.0, dtype=torch.float32)
    flashsac_action = torch.zeros((rows, 21), dtype=torch.float32)
    zeros = torch.zeros(rows, dtype=torch.bool)
    continue_success = zeros.clone()
    route_success = zeros.clone()
    for group in range(len(seeds)):
        start = group * per_seed
        # Four route regressions are vetoed; four route-only gains stay routed.
        continue_success[start : start + 4] = True
        search_action[start : start + 4, 0] = 1.0
        route_success[start + 4 : start + 8] = True
    valid = torch.ones(rows, dtype=torch.bool)
    return {
        "kind": "pick_tool_recoverability_pairs_v1",
        "format_version": 1,
        "metadata": {
            "pairing_semantics": "independent_gpu_rollout_diagnostic_v1",
            "causal_counterfactual_claim_allowed": False,
            "seeds": list(seeds),
            "strong_rows": rows,
        },
        "seed": seed,
        "observation": observation,
        "search_action": search_action,
        "flashsac_action": flashsac_action,
        "paired_outcome_label_valid": valid.clone(),
        "strong_pair": valid.clone(),
        "continue_success": continue_success,
        "route_success": route_success,
        "continue_dropped": zeros.clone(),
        "route_dropped": zeros.clone(),
        "continue_unsafe_force": zeros.clone(),
        "route_unsafe_force": zeros.clone(),
    }


def _events(*, success: int, dropped: int, unsafe_force: int) -> dict[str, int]:
    return {
        "success": success,
        "failure": 100 - success,
        "time_out": 412,
        "dropped": dropped,
        "unsafe_force": unsafe_force,
    }


def _rollout_pair(
    seed: int,
    runtime,
    *,
    route_success: int = 20,
    selector_success: int = 23,
    route_dropped: int = 1,
    selector_dropped: int = 1,
    route_unsafe: int = 2,
    selector_unsafe: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "status": "complete",
        "seed": seed,
        "requested_episodes": 512,
        "num_envs": 512,
        "completed_episodes": 512,
        "task_mode": "full_task",
        "observation_contract": "pick_tool_markov115_v1",
        "max_episode_steps": 1000,
    }
    route = {
        **common,
        "events": _events(
            success=route_success,
            dropped=route_dropped,
            unsafe_force=route_unsafe,
        ),
        "diagnostic_approach_hierarchy": {
            "mode": "frozen_rlgames_search_then_flashsac_close_lift"
        },
    }
    blind = runtime.validate_blind_configuration(
        seed=seed, episodes=512, num_envs=512, strict_blind=True
    )
    selector = {
        **common,
        "events": _events(
            success=selector_success,
            dropped=selector_dropped,
            unsafe_force=selector_unsafe,
        ),
        "diagnostic_approach_hierarchy": {
            "mode": (
                "frozen_rlgames_search_recoverability_selector_then_"
                "flashsac_close_lift"
            ),
            "blind_claim_allowed": True,
            "recoverability_selector": {
                "blind_claim_allowed": True,
                "deployment_claim_allowed": False,
                "model_sha256": runtime.model_sha256,
                "report_sha256": runtime.report_sha256,
                "blind_evaluation": blind,
            },
        },
    }
    return route, selector


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_actual_pairs(directory: Path, runtime) -> list[tuple[Path, Path]]:
    paths: list[tuple[Path, Path]] = []
    for seed in (261, 262, 263, 264):
        route, selector = _rollout_pair(seed, runtime)
        route_path = directory / f"route_{seed}.json"
        selector_path = directory / f"selector_{seed}.json"
        _write_json(route_path, route)
        _write_json(selector_path, selector)
        paths.append((route_path, selector_path))
    return paths


def test_end_to_end_pass_uses_signed_runtime_and_cpu() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        model_path, report_path = _publish(directory)
        runtime = load_recoverability_selector(model_path, report_path, device="cpu")
        dataset_path = directory / "aggregate.pt"
        torch.save(_dataset(), dataset_path)
        actual_pairs = _write_actual_pairs(directory, runtime)
        report = build_audit_report(
            dataset_path=dataset_path,
            model_path=model_path,
            report_path=report_path,
            actual_pairs=actual_pairs,
        )
        assert report["overall_pass"] is True
        assert report["status"] == "pass"
        assert report["execution_device"] == "cpu"
        assert len(report["artifacts"]["audit_source_sha256"]) == 64
        assert "preregistered before seeds 261--264" in report["criteria_origin"]
        offline = report["offline_paired"]
        assert offline["aggregate"]["net_success_delta_vs_route"] == 16
        assert offline["aggregate"]["continue_decision_rate"] == 0.2
        assert offline["aggregate"]["added_drops_vs_route"] == 0
        assert offline["aggregate"]["added_unsafe_force_vs_route"] == 0
        assert all(
            item["net_success_delta_vs_route"] == 4
            for item in offline["by_seed"].values()
        )
        actual = report["actual_rollouts"]
        assert actual["aggregate"]["success_delta_vs_route"] == 12
        assert actual["aggregate"]["drop_delta_vs_route"] == 0
        assert actual["aggregate"]["unsafe_force_delta_vs_route"] == 0


def test_added_safety_events_cannot_be_net_cancelled() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        model_path, report_path = _publish(directory)
        runtime = load_recoverability_selector(model_path, report_path, device="cpu")
        dataset_path = directory / "aggregate.pt"
        payload = _dataset()
        # Both rows select continue: one adds an unsafe event and the other
        # avoids one.  Net delta is zero, but acceptance must still fail.
        payload["continue_unsafe_force"][0] = True
        payload["route_unsafe_force"][1] = True
        torch.save(payload, dataset_path)
        tensors, _ = load_offline_dataset(dataset_path)
        audit = audit_offline(tensors, runtime)
        aggregate = audit["aggregate"]
        assert aggregate["unsafe_force_delta_vs_route"] == 0
        assert aggregate["added_unsafe_force_vs_route"] == 1
        assert aggregate["avoided_unsafe_force_vs_route"] == 1
        assert audit["checks"]["no_added_unsafe_force"]["pass"] is False
        assert audit["criteria_pass"] is False


def test_actual_safety_delta_and_blind_metadata_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        model_path, report_path = _publish(directory)
        runtime = load_recoverability_selector(model_path, report_path, device="cpu")
        pairs = _write_actual_pairs(directory, runtime)
        route, selector = _rollout_pair(261, runtime, selector_unsafe=3)
        _write_json(pairs[0][0], route)
        _write_json(pairs[0][1], selector)
        actual = audit_actual(pairs, runtime)
        assert actual["aggregate"]["success_delta_vs_route"] == 12
        assert actual["aggregate"]["unsafe_force_delta_vs_route"] == 1
        assert actual["checks"]["aggregate_unsafe_force_delta"]["pass"] is False
        assert actual["criteria_pass"] is False

        bad = copy.deepcopy(selector)
        bad["diagnostic_approach_hierarchy"]["recoverability_selector"][
            "blind_evaluation"
        ]["seen_seed"] = True
        _write_json(pairs[0][1], bad)
        _expect_error(ValueError, audit_actual, pairs, runtime)


def test_no_clobber_and_dataset_mask_validation() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        output = directory / "audit.json"
        publish_json_no_clobber({"overall_pass": False}, output)
        original = output.read_bytes()
        _expect_error(
            FileExistsError,
            publish_json_no_clobber,
            {"overall_pass": True},
            output,
        )
        assert output.read_bytes() == original
        assert not list(directory.glob(".audit.json.tmp-*"))

        dataset_path = directory / "aggregate.pt"
        payload = _dataset()
        payload["paired_outcome_label_valid"].zero_()
        torch.save(payload, dataset_path)
        _expect_error(ValueError, load_offline_dataset, dataset_path)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    test_end_to_end_pass_uses_signed_runtime_and_cpu()
    test_added_safety_events_cannot_be_net_cancelled()
    test_actual_safety_delta_and_blind_metadata_fail_closed()
    test_no_clobber_and_dataset_mask_validation()
    print("audit_recoverability_selector_blind tests passed")


if __name__ == "__main__":
    main()
