#!/usr/bin/env python3
"""Simulation-free tests for first-episode online handoff pairing."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

from compare_online_handoff_pair import (
    compare_metrics,
    exact_mcnemar_two_sided_p,
    load_metrics,
    publish_json_no_clobber,
    validate_metrics,
)


SHA = "a" * 64


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _metrics(*, candidate: bool = False) -> dict[str, object]:
    num_envs = 8
    triggered = list(range(num_envs))
    if candidate:
        success = triggered
        failure: list[int] = []
        dropped: list[int] = []
        unlatched: list[int] = []
    else:
        success = []
        failure = triggered
        dropped = [0]
        unlatched = list(range(1, num_envs))
    terminal = {
        "success": success,
        "failure": failure,
        "time_out": [],
        "dropped": dropped,
        "unsafe_force": [],
        "unlatched_clearance_ge_5cm": unlatched,
    }
    result: dict[str, object] = {
        "status": "complete",
        "gradient_updates": 0,
        "seed": 285,
        "interaction_step": 100,
        "environment_steps": 800,
        "online_search_handoff": {
            "contract": "pick_tool_public_online_handoff_v1",
            "search_checkpoint": "/tmp/search.pth",
            "search_checkpoint_sha256": "b" * 64,
            "score": "min(obs[96], second_largest(obs[92:96]))",
            "requires_unlatched_obs106": True,
            "min_score": 0.3,
            "hold_steps": 4,
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
        },
        "online_search_handoff_initial_episode_pending_rows": 0,
        "online_search_handoff_initial_episode_trigger_count": num_envs,
        "online_search_handoff_initial_episode_completed_triggered_episodes": num_envs,
        "online_search_handoff_initial_episode_completed_search_only_episodes": 0,
        "online_search_handoff_initial_episode_triggered_env_ids": triggered,
    }
    for event, ids in terminal.items():
        result[
            "online_search_handoff_initial_episode_triggered_terminal/" + event
        ] = len(ids)
        result[
            "online_search_handoff_initial_episode_triggered_terminal_env_ids/"
            + event
        ] = ids
    return result


def test_positive_pair_is_slot_exact_and_uses_exact_binomial_test() -> None:
    report = compare_metrics(
        _metrics(candidate=False),
        _metrics(candidate=True),
        baseline_sha256=SHA,
        candidate_sha256="c" * 64,
    )
    assert report["status"] == "complete"
    assert report["num_envs"] == 8
    assert report["triggered_cohort_env_ids"] == list(range(8))
    success = report["outcomes"]["success"]
    assert success["baseline_count"] == 0
    assert success["candidate_count"] == 8
    assert success["improved_env_ids"] == list(range(8))
    assert success["lost_env_ids"] == []
    assert success["delta_rate_candidate_minus_baseline"] == 1.0
    assert success["mcnemar_exact_two_sided_p"] == 2.0 / 256.0
    assert report["outcomes"]["dropped"]["improved_env_ids"] == [0]
    assert report["new_candidate_drop_env_ids"] == []
    assert report["new_candidate_unsafe_force_env_ids"] == []
    assert report["pilot_continuation_gate"]["passed"] is True
    assert report["success_proved_improvement"]["passed"] is True

    assert exact_mcnemar_two_sided_p(4, 1) == 0.375
    assert exact_mcnemar_two_sided_p(0, 0) == 1.0


def test_validation_rejects_corrupt_counts_masks_and_incomplete_runs() -> None:
    base = _metrics()
    corrupt = copy.deepcopy(base)
    corrupt["status"] = "running"
    _expect_error(ValueError, validate_metrics, corrupt, name="corrupt")

    corrupt = copy.deepcopy(base)
    corrupt["gradient_updates"] = 1
    _expect_error(ValueError, validate_metrics, corrupt, name="corrupt")

    corrupt = copy.deepcopy(base)
    corrupt["online_search_handoff_initial_episode_pending_rows"] = 1
    _expect_error(ValueError, validate_metrics, corrupt, name="corrupt")

    corrupt = copy.deepcopy(base)
    corrupt["online_search_handoff_initial_episode_triggered_env_ids"] = [
        0,
        1,
        1,
        3,
        4,
        5,
        6,
        7,
    ]
    _expect_error(ValueError, validate_metrics, corrupt, name="corrupt")

    corrupt = copy.deepcopy(base)
    corrupt[
        "online_search_handoff_initial_episode_triggered_terminal/success"
    ] = 1
    _expect_error(ValueError, validate_metrics, corrupt, name="corrupt")

    corrupt = copy.deepcopy(base)
    corrupt[
        "online_search_handoff_initial_episode_triggered_terminal_env_ids/success"
    ] = [8]
    corrupt[
        "online_search_handoff_initial_episode_triggered_terminal/success"
    ] = 1
    _expect_error(ValueError, validate_metrics, corrupt, name="corrupt")

    corrupt = copy.deepcopy(base)
    corrupt[
        "online_search_handoff_initial_episode_triggered_terminal_env_ids/failure"
    ] = list(range(7))
    corrupt[
        "online_search_handoff_initial_episode_triggered_terminal/failure"
    ] = 7
    _expect_error(ValueError, validate_metrics, corrupt, name="corrupt")


def test_pairing_rejects_contract_seed_step_and_cohort_mismatches() -> None:
    baseline = _metrics()
    candidate = _metrics(candidate=True)
    for mutate in (
        lambda value: value.__setitem__("seed", 286),
        lambda value: value.__setitem__("interaction_step", 101),
        lambda value: value["online_search_handoff"].__setitem__(  # type: ignore[union-attr]
            "min_score", 0.31
        ),
        lambda value: value["online_search_handoff"].__setitem__(  # type: ignore[union-attr]
            "hold_steps", 5
        ),
        lambda value: value["online_search_handoff"].__setitem__(  # type: ignore[union-attr]
            "search_checkpoint_sha256", "d" * 64
        ),
    ):
        mismatch = copy.deepcopy(candidate)
        mutate(mismatch)
        if mismatch.get("interaction_step") == 101:
            mismatch["environment_steps"] = 808
        _expect_error(
            ValueError,
            compare_metrics,
            baseline,
            mismatch,
            baseline_sha256=SHA,
            candidate_sha256="c" * 64,
        )

    mismatch = copy.deepcopy(candidate)
    mismatch["online_search_handoff_initial_episode_trigger_count"] = 7
    mismatch[
        "online_search_handoff_initial_episode_completed_triggered_episodes"
    ] = 7
    mismatch[
        "online_search_handoff_initial_episode_completed_search_only_episodes"
    ] = 1
    mismatch["online_search_handoff_initial_episode_triggered_env_ids"] = list(
        range(7)
    )
    for event in ("success",):
        mismatch[
            "online_search_handoff_initial_episode_triggered_terminal/" + event
        ] = 7
        mismatch[
            "online_search_handoff_initial_episode_triggered_terminal_env_ids/"
            + event
        ] = list(range(7))
    _expect_error(
        ValueError,
        compare_metrics,
        baseline,
        mismatch,
        baseline_sha256=SHA,
        candidate_sha256="c" * 64,
    )


def test_new_safety_event_fails_pilot_gate() -> None:
    baseline = _metrics(candidate=True)
    candidate = copy.deepcopy(baseline)
    candidate[
        "online_search_handoff_initial_episode_triggered_terminal_env_ids/success"
    ] = list(range(1, 8))
    candidate[
        "online_search_handoff_initial_episode_triggered_terminal/success"
    ] = 7
    candidate[
        "online_search_handoff_initial_episode_triggered_terminal_env_ids/failure"
    ] = [0]
    candidate[
        "online_search_handoff_initial_episode_triggered_terminal/failure"
    ] = 1
    candidate[
        "online_search_handoff_initial_episode_triggered_terminal_env_ids/dropped"
    ] = [0]
    candidate[
        "online_search_handoff_initial_episode_triggered_terminal/dropped"
    ] = 1
    report = compare_metrics(
        baseline,
        candidate,
        baseline_sha256=SHA,
        candidate_sha256="c" * 64,
    )
    assert report["new_candidate_drop_env_ids"] == [0]
    assert report["pilot_continuation_gate"]["passed"] is False
    assert report["outcomes"]["success"]["lost_env_ids"] == [0]


def test_strict_json_loading_and_no_clobber_output() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metrics_path = root / "metrics.json"
        metrics_path.write_text(json.dumps(_metrics()), encoding="utf-8")
        loaded, digest = load_metrics(metrics_path)
        assert loaded["seed"] == 285
        assert len(digest) == 64

        duplicate = root / "duplicate.json"
        duplicate.write_text('{"status":"complete","status":"complete"}')
        _expect_error(ValueError, load_metrics, duplicate)

        nonfinite = root / "nonfinite.json"
        nonfinite.write_text('{"value":NaN}')
        _expect_error(ValueError, load_metrics, nonfinite)

        output = root / "comparison.json"
        publish_json_no_clobber({"status": "complete"}, output)
        assert json.loads(output.read_text())["status"] == "complete"
        _expect_error(FileExistsError, publish_json_no_clobber, {}, output)


if __name__ == "__main__":
    test_positive_pair_is_slot_exact_and_uses_exact_binomial_test()
    test_validation_rejects_corrupt_counts_masks_and_incomplete_runs()
    test_pairing_rejects_contract_seed_step_and_cohort_mismatches()
    test_new_safety_event_fails_pilot_gate()
    test_strict_json_loading_and_no_clobber_output()
    print("compare_online_handoff_pair_test: PASS")
