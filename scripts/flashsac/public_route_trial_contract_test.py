#!/usr/bin/env python3
"""Simulation-free tests for the public randomized route trial contract."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from public_route_trial_contract import (
    FEATURE_DIM,
    ROW_FIELDS,
    assignment_for,
    assignment_route_mask,
    build_gate_feature,
    build_trial_artifact,
    complementary_replicate,
    publish_json_no_clobber,
    publish_trial_and_report_no_clobber,
    update_public_readiness,
    validate_trial_artifact,
)


SALT = "pick-tool-public-route-trial-v1-20260722-883bab4"


def _raises(kind: type[BaseException], fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def _candidate() -> tuple[torch.Tensor, float, int]:
    observation = torch.arange(115, dtype=torch.float32).unsqueeze(0) / 1000.0
    observation[:, 92:96] = torch.tensor((0.8, 0.6, 0.4, 0.2))
    observation[:, 96] = 0.5
    observation[:, 102] = 0.25
    observation[:, 105] = 1.0
    observation[:, 106] = 0.0
    observation[:, 107] = 0.5
    observation[:, 108] = 1.0 / 6.0
    search_action = torch.linspace(-1.0, 1.0, 21, dtype=torch.float32).unsqueeze(0)
    route_action = -search_action.clone()
    route_action[:, :7] = 0.0
    feature = build_gate_feature(
        observation,
        torch.tensor((4,), dtype=torch.long),
        torch.tensor((False,), dtype=torch.bool),
        torch.tensor((10,), dtype=torch.long),
        torch.tensor(((0.3, 0.5),), dtype=torch.float32),
        search_action,
        route_action,
    )
    return feature[0].cpu(), 0.5, 3


def _metadata() -> dict[str, Any]:
    return {
        "seed": 266,
        "num_envs": 8,
        "assignment_salt": SALT,
        "replicate": "a",
        "provenance": {"cohort": "pilot", "manifest_sha256": "a" * 64},
    }


def _row(slot: int | None = None) -> dict[str, Any]:
    assignment = assignment_for("pilot", 266, 8, SALT, "a")
    if slot is None:
        slot = int(torch.nonzero(assignment, as_tuple=False)[0])
    feature, score, stratum = _candidate()
    row = {
        "feature": feature,
        "env_slot": slot,
        "slot_episode_index": 0,
        "episode_index": slot,
        "candidate_step": 10,
        "stratum": stratum,
        "outcome_episode_length": 1000,
        "readiness_score": score,
        "outcome_max_true_clearance_m": 0.01,
        "factual_treatment_route": bool(assignment[slot]),
        "outcome_terminated": False,
        "outcome_truncated": True,
        "outcome_success": False,
        "outcome_failure": False,
        "outcome_time_out": True,
        "outcome_dropped": False,
        "outcome_unsafe_force": False,
        "outcome_unlatched_clearance_ge_5cm": False,
        "outcome_ever_grasped": False,
        "outcome_ever_clearance_ge_20cm": False,
    }
    assert set(row) == ROW_FIELDS
    return row


def test_readiness_is_public_four_frame_and_sticky() -> None:
    observation = torch.zeros((4, 115), dtype=torch.float32)
    observation[:, 92:96] = torch.tensor((0.8, 0.4, 0.2, 0.1))
    observation[:, 96] = torch.tensor((0.3, 0.09, 0.5, 0.5))
    observation[:, 102] = torch.tensor((0.3, 0.3, 0.1, 0.3))
    observation[:, 106] = torch.tensor((0.0, 0.0, 1.0, 0.0))
    ready = torch.tensor((3, 3, 3, 4), dtype=torch.long)
    used = torch.tensor((False, False, False, True), dtype=torch.bool)
    result = update_public_readiness(observation, ready, used)
    assert torch.allclose(result["score"], torch.tensor((0.3, 0.09, 0.4, 0.4)))
    assert result["eligible"].tolist() == [True, False, False, True]
    assert result["ready_count_after"].tolist() == [4, 0, 0, 4]
    assert result["trigger"].tolist() == [True, False, False, False]
    assert result["fork_used_after"].tolist() == [True, False, False, True]
    assert result["stratum"].tolist() == [1, 1, 2, 3]
    assert result["ready_count_before"] is ready
    assert result["fork_used_before"] is used


def test_feature_layout_and_direct_obs86_exclusion() -> None:
    feature, _, _ = _candidate()
    observation = torch.arange(115, dtype=torch.float32) / 1000.0
    # Candidate helper overwrote only later observation values; early layout is exact.
    assert torch.equal(feature[:86], observation[:86])
    assert feature[85].item() == observation[85].item()
    assert feature[86].item() == observation[87].item()
    assert not bool((feature[:114] == observation[86]).any())
    assert feature.shape == (FEATURE_DIM,)
    assert feature[114:119].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert feature[119].item() == 0.0
    assert abs(feature[120].item() - 10.0 / 999.0) < 1.0e-8
    assert feature[121:123].tolist() == [0.30000001192092896, 0.5]
    expected_action = torch.linspace(-1.0, 1.0, 21, dtype=torch.float32)
    assert torch.equal(feature[123:144], expected_action)
    expected_route = -expected_action
    expected_route[:7] = 0.0
    assert torch.equal(feature[144:165], expected_route)


def test_assignment_matches_manifest_bytes_balances_and_complements() -> None:
    assigned = assignment_for("pilot", 266, 8, SALT, "a")
    assert assigned.tolist() == [False, False, True, False, True, False, True, True]
    assert torch.equal(
        assigned,
        assignment_route_mask(
            cohort="pilot", seed=266, num_envs=8, salt=SALT, replicate="a"
        ),
    )
    complement = assignment_for("pilot", 266, 8, SALT, complementary_replicate("a"))
    assert torch.equal(complement, ~assigned)
    assert int(assigned.sum()) == int(complement.sum()) == 4
    assert torch.equal(assigned, assignment_for("pilot", 266, 8, SALT, "a"))
    assert not torch.equal(assigned, assignment_for("train", 266, 8, SALT, "a"))
    # Independent implementation of the manifest's canonical byte formula.
    order = sorted(
        range(8),
        key=lambda slot: (
            hashlib.sha256(
                ("\0".join((SALT, "pilot", "266", str(slot), "episode=0"))).encode(
                    "utf-8"
                )
            ).digest(),
            slot,
        ),
    )
    expected = torch.zeros(8, dtype=torch.bool)
    expected[order[:4]] = True
    assert torch.equal(assigned, expected)
    _raises(ValueError, assignment_for, "pilot", 266, 7, SALT, "a")


def test_factual_artifact_is_strict_and_fail_closed() -> None:
    assignment = assignment_for("pilot", 266, 8, SALT, "a")
    artifact = build_trial_artifact([_row()], _metadata(), assignment)
    validate_trial_artifact(artifact)

    wrong_assignment = assignment.clone()
    wrong_assignment[0] = ~wrong_assignment[0]
    _raises(ValueError, build_trial_artifact, [_row()], _metadata(), wrong_assignment)
    fake_dual_arm = _row()
    fake_dual_arm["route_success"] = True
    _raises(KeyError, build_trial_artifact, [fake_dual_arm], _metadata(), assignment)

    wrong_propensity = copy.deepcopy(artifact)
    wrong_propensity["tensors"]["propensity_route"][0] = 0.4
    _raises(ValueError, validate_trial_artifact, wrong_propensity)
    wrong_terminal = copy.deepcopy(artifact)
    wrong_terminal["tensors"]["outcome_truncated"][0] = False
    _raises(ValueError, validate_trial_artifact, wrong_terminal)
    fake_success = copy.deepcopy(artifact)
    tensors = fake_success["tensors"]
    tensors["outcome_success"][0] = True
    tensors["outcome_time_out"][0] = False
    tensors["outcome_terminated"][0] = True
    tensors["outcome_truncated"][0] = False
    _raises(ValueError, validate_trial_artifact, fake_success)
    wrong_episode = copy.deepcopy(artifact)
    wrong_episode["tensors"]["slot_episode_index"][0] = 1
    _raises(ValueError, validate_trial_artifact, wrong_episode)
    wrong_clearance = copy.deepcopy(artifact)
    wrong_clearance["tensors"]["outcome_ever_clearance_ge_20cm"][0] = True
    _raises(ValueError, validate_trial_artifact, wrong_clearance)
    wrong_unlatched = copy.deepcopy(artifact)
    wrong_unlatched["tensors"]["outcome_unlatched_clearance_ge_5cm"][0] = True
    wrong_unlatched["tensors"]["outcome_failure"][0] = True
    wrong_unlatched["tensors"]["outcome_terminated"][0] = True
    wrong_unlatched["tensors"]["outcome_truncated"][0] = False
    wrong_unlatched["tensors"]["outcome_time_out"][0] = False
    _raises(ValueError, validate_trial_artifact, wrong_unlatched)
    wrong_action = copy.deepcopy(artifact)
    wrong_action["tensors"]["feature"][0, 123] = 1.01
    _raises(ValueError, validate_trial_artifact, wrong_action)
    wrong_route_arm = copy.deepcopy(artifact)
    wrong_route_arm["tensors"]["feature"][0, 144] = 0.01
    _raises(ValueError, validate_trial_artifact, wrong_route_arm)
    wrong_step = copy.deepcopy(artifact)
    wrong_step["tensors"]["candidate_step"][0] = 0
    wrong_step["tensors"]["feature"][0, 120] = 0.0
    _raises(ValueError, validate_trial_artifact, wrong_step)
    wrong_safety = copy.deepcopy(artifact)
    wrong_safety["tensors"]["feature"][0, 121] = 1.0
    _raises(ValueError, validate_trial_artifact, wrong_safety)
    inconsistent_safety = copy.deepcopy(artifact)
    inconsistent_safety["tensors"]["feature"][0, 121] = 0.0
    inconsistent_safety["tensors"]["feature"][0, 122] = 0.5
    _raises(ValueError, validate_trial_artifact, inconsistent_safety)
    wrong_latch_complement = copy.deepcopy(artifact)
    wrong_latch_complement["tensors"]["feature"][0, 104] = -1.0e6
    _raises(ValueError, validate_trial_artifact, wrong_latch_complement)
    wrong_confirm = copy.deepcopy(artifact)
    wrong_confirm["tensors"]["feature"][0, 106] = 0.3
    _raises(ValueError, validate_trial_artifact, wrong_confirm)
    wrong_release = copy.deepcopy(artifact)
    wrong_release["tensors"]["feature"][0, 107] = 0.2
    _raises(ValueError, validate_trial_artifact, wrong_release)
    wrong_slip = copy.deepcopy(artifact)
    wrong_slip["tensors"]["feature"][0, 108] = 2.01
    _raises(ValueError, validate_trial_artifact, wrong_slip)
    wrong_quality = copy.deepcopy(artifact)
    wrong_quality["tensors"]["feature"][0, 91] = 1.01
    _raises(ValueError, validate_trial_artifact, wrong_quality)
    missing_unlatched = copy.deepcopy(artifact)
    missing_unlatched["tensors"]["outcome_max_true_clearance_m"][0] = 0.10
    _raises(ValueError, validate_trial_artifact, missing_unlatched)
    impossible_fast_success = copy.deepcopy(artifact)
    fast = impossible_fast_success["tensors"]
    fast["candidate_step"][0] = 999
    fast["feature"][0, 120] = 1.0
    fast["outcome_episode_length"][0] = 1000
    fast["outcome_max_true_clearance_m"][0] = 0.20
    fast["outcome_success"][0] = True
    fast["outcome_time_out"][0] = False
    fast["outcome_terminated"][0] = True
    fast["outcome_truncated"][0] = False
    fast["outcome_ever_grasped"][0] = True
    fast["outcome_ever_clearance_ge_20cm"][0] = True
    _raises(ValueError, validate_trial_artifact, impossible_fast_success)
    impossible_fast_timeout = copy.deepcopy(artifact)
    impossible_fast_timeout["tensors"]["outcome_episode_length"][0] = 11
    _raises(ValueError, validate_trial_artifact, impossible_fast_timeout)
    impossible_fast_unsafe = copy.deepcopy(artifact)
    unsafe = impossible_fast_unsafe["tensors"]
    unsafe["candidate_step"][0] = 999
    unsafe["feature"][0, 120] = 1.0
    unsafe["feature"][0, 121:123] = 0.0
    unsafe["outcome_episode_length"][0] = 1000
    unsafe["outcome_unsafe_force"][0] = True
    unsafe["outcome_failure"][0] = True
    unsafe["outcome_time_out"][0] = False
    unsafe["outcome_terminated"][0] = True
    unsafe["outcome_truncated"][0] = False
    _raises(ValueError, validate_trial_artifact, impossible_fast_unsafe)


def test_weights_only_safe_atomic_pair_and_no_clobber() -> None:
    artifact = build_trial_artifact(
        [_row()], _metadata(), assignment_for("pilot", 266, 8, SALT, "a")
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact_path, report_path = root / "trial.pt", root / "trial.json"
        digest = publish_trial_and_report_no_clobber(
            artifact,
            {"kind": "test_report", "status": "complete"},
            artifact_output=artifact_path,
            report_output=report_path,
        )
        assert len(digest) == 64 and report_path.is_file()
        loaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
        validate_trial_artifact(loaded)
        _raises(
            FileExistsError,
            publish_trial_and_report_no_clobber,
            artifact,
            {"status": "replacement"},
            artifact_output=artifact_path,
            report_output=report_path,
        )
        dangling = root / "dangling.json"
        dangling.symlink_to(root / "missing")
        _raises(FileExistsError, publish_json_no_clobber, {"status": "bad"}, dangling)


if __name__ == "__main__":
    test_readiness_is_public_four_frame_and_sticky()
    test_feature_layout_and_direct_obs86_exclusion()
    test_assignment_matches_manifest_bytes_balances_and_complements()
    test_factual_artifact_is_strict_and_fail_closed()
    test_weights_only_safe_atomic_pair_and_no_clobber()
    print("public route trial contract tests passed")
