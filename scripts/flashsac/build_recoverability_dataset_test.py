#!/usr/bin/env python3
"""Simulation-free tests for audited recoverability pairing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any
from unittest import mock

import torch

import build_recoverability_dataset as recoverability_module
from build_recoverability_dataset import (
    _publish_dataset_and_report,
    build_dataset,
    pair_handoff_artifacts,
    sha256_file,
    validate_handoff_artifact,
)
from evaluate import build_diagnostic_handoff_artifact


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _metadata(treatment: str, *, seed: int = 257) -> dict[str, Any]:
    return {
        "treatment": treatment,
        "task_mode": "full_task",
        "observation_contract": "pick_tool_markov115_v1",
        "observation_dim": 115,
        "action_dim": 21,
        "seed": seed,
        "requested_episodes": 8,
        "num_envs": 8,
        "episode_length_s": 20.0,
        "max_episode_steps": 1000,
        "deterministic_policy_actions": True,
        "use_compile": False,
        "approach_checkpoint_sha256": "a" * 64,
        "flashsac_actor_sha256": "b" * 64,
        "flashsac_task_contract_sha256": "c" * 64,
        "flashsac_fork_commit": "fork-test",
        "flashsac_upstream_commit": "upstream-test",
        "source_sha256": {"task.py": "d" * 64, "hammer.obj": "e" * 64},
        "hierarchy_metrics_output": "/tmp/not-used-by-pure-pair-test.json",
        "supervisor": {
            "minimum_zero_based_episode_step": 400,
            "pregrasp_score_threshold": 0.32,
            "minimum_proximity_quality": 0.02,
            "hold_steps": 4,
            "safe_force_limit_n": 30.0,
            "requires_unlatched": True,
            "requires_abs_true_clearance_le_m": 0.005,
        },
        "selector_eligible_fields": [
            "observation",
            "search_action",
            "flashsac_action",
        ],
        "audit_only_private_fields": [
            "handoff_step",
            "pregrasp_score",
            "proximity_quality",
            "max_force_n",
            "true_clearance_m",
        ],
        "outcome_semantics": "strict true convex clearance test",
    }


def _row(
    env_slot: int,
    *,
    success: bool,
    episode_index: int,
) -> dict[str, Any]:
    observation = torch.linspace(-1.0, 1.0, 115, dtype=torch.float32)
    observation[0] = float(env_slot) / 10.0
    observation[106] = 0.0
    return {
        "observation": observation,
        "search_action": torch.linspace(-0.4, 0.4, 21, dtype=torch.float32),
        "flashsac_action": torch.linspace(0.7, -0.7, 21, dtype=torch.float32),
        "env_slot": env_slot,
        "slot_episode_index": 0,
        "episode_index": episode_index,
        "handoff_step": 403 + env_slot,
        "pregrasp_score": 0.34,
        "proximity_quality": 0.08,
        "max_force_n": 12.0,
        "true_clearance_m": 0.001,
        "outcome_episode_length": 650 + env_slot,
        "outcome_max_true_clearance_m": 0.21 if success else 0.01,
        "outcome_success": success,
        "outcome_failure": False,
        "outcome_time_out": not success,
        "outcome_ever_grasped": success,
        "outcome_ever_clearance_ge_5cm": success,
        "outcome_ever_clearance_ge_20cm": success,
        "outcome_dropped": False,
        "outcome_unsafe_force": False,
        "outcome_ever_unlatched_clearance_ge_5cm": False,
        "outcome_ever_post_candidate_latch": success,
    }


def _artifact(treatment: str, outcomes: list[bool], *, episode_offset: int) -> dict:
    rows = [
        _row(env_slot, success=success, episode_index=episode_offset + env_slot)
        for env_slot, success in enumerate(outcomes)
    ]
    return build_diagnostic_handoff_artifact(
        rows,
        metadata=_metadata(treatment),
    )


def _clone_payload(payload: dict) -> dict:
    return {
        key: (
            value.clone()
            if isinstance(value, torch.Tensor)
            else dict(value)
            if key == "metadata"
            else value
        )
        for key, value in payload.items()
    }


def test_pair_labels_and_canonical_features() -> None:
    # gain, regression, both-success, both-fail
    routed = _artifact("handoff_to_flashsac", [True, False, True, False], episode_offset=0)
    continued = _artifact("continue_search", [False, True, True, False], episode_offset=4)
    paired, report = pair_handoff_artifacts(routed, continued)
    assert report["common_candidates"] == 4
    assert report["strong_pairs"] == 4
    assert paired["strict_success_delta"].tolist() == [1, -1, 0, 0]
    assert paired["route_label"].tolist() == [True, False, False, False]
    assert paired["preference_label_valid"].tolist() == [True, True, False, False]
    assert paired["preference_sample_weight"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert paired["paired_outcome_label_valid"].tolist() == [True, True, True, True]
    assert torch.equal(paired["observation"], paired["continue_observation"])
    # Cross-treatment global completion order is intentionally irrelevant.
    assert paired["route_episode_index"].tolist() == [0, 1, 2, 3]
    assert paired["continue_episode_index"].tolist() == [4, 5, 6, 7]


def test_pair_quality_fails_closed() -> None:
    routed = _artifact("handoff_to_flashsac", [True, False], episode_offset=0)
    continued = _artifact("continue_search", [False, True], episode_offset=2)

    step_mismatch = _clone_payload(continued)
    step_mismatch["handoff_step"][0] += 1
    paired, _ = pair_handoff_artifacts(routed, step_mismatch)
    assert paired["same_handoff_step"].tolist() == [False, True]
    assert paired["strong_pair"].tolist() == [False, True]
    assert not bool(paired["preference_label_valid"][0])

    observation_mismatch = _clone_payload(continued)
    observation_mismatch["observation"][1, 12] += 1.0e-3
    paired, _ = pair_handoff_artifacts(routed, observation_mismatch)
    assert paired["strong_pair"].tolist() == [True, False]

    gate_mismatch = _clone_payload(continued)
    gate_mismatch["max_force_n"][1] += 0.1
    paired, _ = pair_handoff_artifacts(routed, gate_mismatch)
    assert paired["strong_pair"].tolist() == [True, False]

    wrong_seed = _clone_payload(continued)
    wrong_seed["metadata"] = dict(wrong_seed["metadata"])
    wrong_seed["metadata"]["seed"] = 999
    _expect_error(ValueError, pair_handoff_artifacts, routed, wrong_seed)
    _expect_error(
        ValueError,
        validate_handoff_artifact,
        continued,
        source="continued",
        expected_treatment="handoff_to_flashsac",
    )


def _write_artifact_pair(
    directory: Path,
    *,
    seed: int,
    source_digest: str = "d",
) -> tuple[Path, Path]:
    paths: list[Path] = []
    for treatment, outcomes, episode_offset in (
        ("handoff_to_flashsac", [True, False], 0),
        ("continue_search", [False, True], 2),
    ):
        stem = "route" if treatment == "handoff_to_flashsac" else "continue"
        artifact_path = directory / f"{stem}_{seed}.pt"
        metrics_path = directory / f"{stem}_{seed}.json"
        metadata = _metadata(treatment, seed=seed)
        metadata["hierarchy_metrics_output"] = str(metrics_path)
        metadata["source_sha256"] = {
            "task.py": source_digest * 64,
            "hammer.obj": "e" * 64,
        }
        rows = [
            _row(env_slot, success=success, episode_index=episode_offset + env_slot)
            for env_slot, success in enumerate(outcomes)
        ]
        artifact = build_diagnostic_handoff_artifact(rows, metadata=metadata)
        torch.save(artifact, artifact_path)
        artifact_sha256 = sha256_file(artifact_path)
        episodes = []
        for env_slot in range(8):
            if env_slot < len(rows):
                row = rows[env_slot]
                episodes.append(
                    {
                        "env_slot": env_slot,
                        "slot_episode_index": 0,
                        "episode_index": row["episode_index"],
                        "hierarchy_handoff_candidate": True,
                        "hierarchy_handoff": treatment == "handoff_to_flashsac",
                        "hierarchy_handoff_step": row["handoff_step"],
                        "hierarchy_ever_post_handoff_latch": row[
                            "outcome_ever_post_candidate_latch"
                        ],
                        "length": row["outcome_episode_length"],
                        "max_true_clearance_m": row[
                            "outcome_max_true_clearance_m"
                        ],
                        "success": row["outcome_success"],
                        "failure": row["outcome_failure"],
                        "time_out": row["outcome_time_out"],
                        "ever_grasped": row["outcome_ever_grasped"],
                        "ever_clearance_ge_5cm": row[
                            "outcome_ever_clearance_ge_5cm"
                        ],
                        "ever_clearance_ge_20cm": row[
                            "outcome_ever_clearance_ge_20cm"
                        ],
                        "dropped": row["outcome_dropped"],
                        "unsafe_force": row["outcome_unsafe_force"],
                        "ever_unlatched_clearance_ge_5cm": row[
                            "outcome_ever_unlatched_clearance_ge_5cm"
                        ],
                    }
                )
            else:
                episodes.append(
                    {
                        "env_slot": env_slot,
                        "slot_episode_index": 0,
                        "hierarchy_handoff_candidate": False,
                    }
                )
        metrics = {
            "status": "complete",
            "policy": (
                "diagnostic_frozen_rlgames_search_then_checkpoint_native_flashsac_close_lift"
                if treatment == "handoff_to_flashsac"
                else "diagnostic_frozen_rlgames_base_only"
            ),
            "task_mode": metadata["task_mode"],
            "observation_contract": metadata["observation_contract"],
            "observation_dim": metadata["observation_dim"],
            "action_dim": metadata["action_dim"],
            "seed": seed,
            "num_envs": metadata["num_envs"],
            "requested_episodes": metadata["requested_episodes"],
            "completed_episodes": metadata["requested_episodes"],
            "episode_length_s": metadata["episode_length_s"],
            "max_episode_steps": metadata["max_episode_steps"],
            "use_compile": metadata["use_compile"],
            "checkpoint_actor_sha256": metadata["flashsac_actor_sha256"],
            "checkpoint_task_contract_sha256": metadata[
                "flashsac_task_contract_sha256"
            ],
            "flashsac_fork_commit": metadata["flashsac_fork_commit"],
            "flashsac_upstream_commit": metadata["flashsac_upstream_commit"],
            "episodes": episodes,
            "diagnostic_approach_hierarchy": {
                "approach_checkpoint_sha256": metadata[
                    "approach_checkpoint_sha256"
                ],
                "supervisor": metadata["supervisor"],
                "candidate_count": len(rows),
                "handoff_count": (
                    len(rows) if treatment == "handoff_to_flashsac" else 0
                ),
                "handoff_artifact": {
                    "path": str(artifact_path),
                    "sha256": artifact_sha256,
                    "kind": artifact["kind"],
                    "rows": len(rows),
                    "treatment": treatment,
                },
            },
        }
        metrics_path.write_text(
            json.dumps(metrics, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        paths.append(artifact_path)
    return paths[0], paths[1]


def test_build_dataset_checks_companions_and_cross_seed_leakage() -> None:
    kwargs = {
        "max_observation_abs_error": 1.0e-5,
        "max_action_abs_error": 1.0e-5,
        "max_score_abs_error": 1.0e-5,
        "max_proximity_abs_error": 1.0e-5,
        "max_force_abs_error_n": 1.0e-2,
        "max_clearance_abs_error_m": 1.0e-5,
    }
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        pair_257 = _write_artifact_pair(directory, seed=257)
        payload, report = build_dataset([pair_257], **kwargs)
        assert payload["metadata"]["strong_rows"] == 2
        assert report["status"] == "complete"
        assert payload["metadata"]["preference_label_contract"] == {
            "target": "route_label",
            "valid_mask": "preference_label_valid",
            "sample_weight": "preference_sample_weight",
            "tie_semantics": "invalid_for_direct_preference",
        }

        pair_258_mismatch = _write_artifact_pair(
            directory, seed=258, source_digest="f"
        )
        _expect_error(
            ValueError,
            build_dataset,
            [pair_257, pair_258_mismatch],
            **kwargs,
        )

        pair_259_duplicate = _write_artifact_pair(directory, seed=259)
        _expect_error(
            ValueError,
            build_dataset,
            [pair_257, pair_259_duplicate],
            **kwargs,
        )

        bad_metrics = Path(
            str(
                torch.load(
                    pair_257[0], map_location="cpu", weights_only=True
                )["metadata"]["hierarchy_metrics_output"]
            )
        )
        metrics = json.loads(bad_metrics.read_text(encoding="utf-8"))
        metrics["status"] = "partial"
        bad_metrics.write_text(json.dumps(metrics), encoding="utf-8")
        _expect_error(ValueError, build_dataset, [pair_257], **kwargs)


def test_transactional_publication_is_no_clobber() -> None:
    payload = {"kind": "test", "value": torch.tensor([1.0])}
    report = {"status": "complete"}
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        output = directory / "dataset.pt"
        report_path = directory / "report.json"
        final = _publish_dataset_and_report(
            payload, report, output=output, report_path=report_path
        )
        assert output.is_file() and report_path.is_file()
        assert final["dataset_sha256"] == sha256_file(output)
        _expect_error(
            FileExistsError,
            _publish_dataset_and_report,
            payload,
            report,
            output=output,
            report_path=report_path,
        )

        dangling = directory / "dangling.pt"
        dangling.symlink_to(directory / "missing.pt")
        _expect_error(
            FileExistsError,
            _publish_dataset_and_report,
            payload,
            report,
            output=dangling,
            report_path=directory / "unused.json",
        )

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        output = directory / "dataset.pt"
        report_path = directory / "report.json"
        real_link = os.link
        calls = 0

        def fail_second_link(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected report publication failure")
            return real_link(source, target)

        with mock.patch.object(
            recoverability_module.os, "link", side_effect=fail_second_link
        ):
            _expect_error(
                OSError,
                _publish_dataset_and_report,
                payload,
                report,
                output=output,
                report_path=report_path,
            )
        assert not output.exists() and not report_path.exists()
        assert not list(directory.glob(".*.tmp-*"))


def main() -> None:
    test_pair_labels_and_canonical_features()
    test_pair_quality_fails_closed()
    test_build_dataset_checks_companions_and_cross_seed_leakage()
    test_transactional_publication_is_no_clobber()
    print("recoverability dataset tests passed")


if __name__ == "__main__":
    main()
