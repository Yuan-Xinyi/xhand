#!/usr/bin/env python3
"""Simulation-free tests for the preregistered online CLOSE A/B analysis."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import torch

import analyze_online_close_ab as analysis
from analyze_online_close_ab import (
    ADVERSE_EVENTS,
    ANALYSIS_FORMAT_VERSION,
    EVENTS,
    PLAN_KIND,
    PRODUCTION_RUN_ORDER,
    analyze,
    load_analysis_plan,
    load_evidence,
)
from online_close_ab import (
    ARTIFACT_KIND,
    ASSIGNMENT_CONTRACT,
    ASSIGNMENT_SALT,
    COLLECTION_CONTRACT,
    FORMAT_VERSION as ARTIFACT_FORMAT_VERSION,
    HANDOFF_HOLD_STEPS,
    HANDOFF_MIN_SCORE,
    KIT_ARGS,
    MAX_EPISODE_ACTIONS,
    REPORT_KIND,
    SOURCE_PATH_COUNT,
    SOURCE_PATH_SET_SHA256,
    assignment_candidate_mask,
    build_artifact,
    publish_artifact_and_report_no_clobber,
)
from online_close_ab_test import _metadata, _report


def _expect(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _tensors(seed: int, replicate: str) -> dict[str, torch.Tensor]:
    slots = torch.arange(1024, dtype=torch.long)
    candidate = assignment_candidate_mask(
        seed=seed, num_envs=1024, replicate=replicate
    )
    triggered = slots.remainder(5) != 0
    candidate_success = slots.remainder(4) != 0
    baseline_success = slots.remainder(20) < 9
    success = triggered & torch.where(candidate, candidate_success, baseline_success)
    timeout = ~success
    episode_length = torch.where(
        timeout,
        torch.full((1024,), MAX_EPISODE_ACTIONS, dtype=torch.long),
        torch.full((1024,), 100, dtype=torch.long),
    )
    maximum = torch.where(
        success,
        torch.full((1024,), 0.25, dtype=torch.float32),
        torch.full((1024,), 0.01, dtype=torch.float32),
    )
    zero = torch.zeros(1024, dtype=torch.bool)
    return {
        "env_slot": slots,
        "assignment_candidate": candidate,
        "triggered": triggered,
        "trigger_step": torch.where(
            triggered,
            torch.full((1024,), HANDOFF_HOLD_STEPS - 1, dtype=torch.long),
            torch.full((1024,), -1, dtype=torch.long),
        ),
        "trigger_score": torch.where(
            triggered,
            torch.full((1024,), 0.35, dtype=torch.float32),
            torch.zeros(1024, dtype=torch.float32),
        ),
        "episode_length": episode_length,
        "max_true_clearance_m": maximum,
        "success": success,
        "failure": zero.clone(),
        "time_out": timeout,
        "dropped": zero.clone(),
        "unsafe_force": zero.clone(),
        "unlatched_clearance_ge_5cm": zero.clone(),
        "ever_grasped": success.clone(),
        "ever_clearance_ge_20cm": success.clone(),
    }


def _identity(metadata: dict[str, object]) -> dict[str, object]:
    runtime = dict(metadata["runtime"])
    runtime.pop("seed")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "report_kind": REPORT_KIND,
        "collection_contract": COLLECTION_CONTRACT,
        "assignment_contract": ASSIGNMENT_CONTRACT,
        "assignment_salt": ASSIGNMENT_SALT,
        "kit_args": KIT_ARGS,
        "handoff_min_score": HANDOFF_MIN_SCORE,
        "handoff_hold_steps": HANDOFF_HOLD_STEPS,
        "baseline_actor_sha256": metadata["baseline_actor_sha256"],
        "candidate_actor_sha256": metadata["candidate_actor_sha256"],
        "common_frozen_lift_actor_sha256": metadata[
            "common_frozen_lift_actor_sha256"
        ],
        "common_frozen_lift_semantic_sha256": metadata[
            "common_frozen_lift_semantic_sha256"
        ],
        "common_frozen_lift_source_actor_sha256": metadata[
            "common_frozen_lift_source_actor_sha256"
        ],
        "search_checkpoint_sha256": metadata["search_checkpoint_sha256"],
        "baseline_task_contract_sha256": metadata[
            "baseline_task_contract_sha256"
        ],
        "candidate_task_contract_sha256": metadata[
            "candidate_task_contract_sha256"
        ],
        "baseline_bridge_state_sha256": metadata["baseline_bridge_state_sha256"],
        "candidate_bridge_state_sha256": metadata[
            "candidate_bridge_state_sha256"
        ],
        "flashsac_upstream_commit": metadata["flashsac_upstream_commit"],
        "flashsac_fork_commit": metadata["flashsac_fork_commit"],
        "source_sha256_digest": analysis._semantic_sha256(metadata["source_sha256"]),
        "source_path_count": SOURCE_PATH_COUNT,
        "source_path_set_sha256": SOURCE_PATH_SET_SHA256,
        "runtime_asset_sha256_digest": analysis._semantic_sha256(
            metadata["runtime_asset_sha256"]
        ),
        "runtime_static_sha256": analysis._semantic_sha256(runtime),
        "collector_git_branch": metadata["git"]["branch"],
    }


def _plan(root: Path, evidence_root: Path, metadata: dict[str, object]) -> dict[str, object]:
    def relative(path: Path) -> str:
        return str(path.relative_to(root))

    return {
        "kind": PLAN_KIND,
        "format_version": ANALYSIS_FORMAT_VERSION,
        "status": "preregistered",
        "preregistration": {
            "branch": "test-branch",
            "collector_base_commit": "a" * 40,
            "implementation_commit": "b" * 40,
            "seal_tag": "test-online-close-ab-analysis-seal",
            "analysis_source_sha256": analysis.sha256_file(Path(analysis.__file__)),
        },
        "data": {
            "seeds": [290, 291, 292],
            "replicates": ["a", "b"],
            "num_envs": 1024,
            "events": list(EVENTS),
            "identity": _identity(metadata),
            "runs": [
                {
                    "seed": seed,
                    "replicate": replicate,
                    "artifact": relative(evidence_root / f"trial_{seed}_{replicate}.pt"),
                    "report": relative(evidence_root / f"trial_{seed}_{replicate}.json"),
                }
                for seed, replicate in PRODUCTION_RUN_ORDER
            ],
        },
        "estimands": {
            "domains": ["conditional_triggered", "itt_all_assigned"],
            "primary_estimator": "horvitz_thompson",
            "diagnostic_estimator": "hajek",
            "known_propensity": 0.5,
            "seed_weighting": "equal arithmetic mean of the three per-seed estimates",
            "contrast": "candidate minus baseline",
            "conditional_population": "all first episodes whose public SEARCH handoff triggered",
            "itt_population": "all assigned first episodes",
        },
        "inference": {
            "bootstrap": {
                "replicates": 20000,
                "seed": 123456,
                "resampling_unit": (
                    "within each seed resample 1024 env_slot clusters with replacement; "
                    "replicates a,b and their complementary assignments stay together"
                ),
                "lower_rank": 1000,
                "upper_rank": 19001,
            },
            "fisher": {
                "replicates": 20000,
                "seed": 654321,
                "assignment": (
                    "independently per seed choose exactly 512 candidate env slots in replicate a; "
                    "replicate b is the exact complement"
                ),
                "p_value": (
                    "plus-one Monte Carlo; report greater, less, and two-sided absolute-tail probabilities"
                ),
                "primary_statistic": (
                    "equal-seed-weight conditional success Horvitz-Thompson candidate-minus-baseline delta"
                ),
            },
        },
        "acceptance": {
            "minimum_triggered_per_arm_total": 1000,
            "minimum_triggered_per_seed_arm": 300,
            "minimum_triggered_per_run_arm": 100,
            "conditional_success_point_min": 0.05,
            "conditional_success_ht_lower_strict_min": 0.0,
            "itt_success_point_min": 0.015,
            "itt_success_ht_lower_strict_min": 0.0,
            "conditional_success_fisher_greater_p_max": 0.05,
            "positive_seed_count_min": 3,
            "seed_meets_both_success_margins_count_min": 2,
            "conditional_adverse_ht_upper_max": {
                event: 0.01 for event in ADVERSE_EVENTS
            },
            "minimum_baseline_events_for_safety_inference": 10,
            "maximum_candidate_events_when_safety_underpowered": 0,
            "require_every_condition": True,
        },
        "output": {"report": relative(evidence_root / "analysis.json")},
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _install_fake_git(plan_path: Path) -> tuple[object, object, object, object]:
    real_text = analysis._git_text
    real_ancestor = analysis._git_is_ancestor
    real_bytes = analysis._git_bytes
    real_paths_match = analysis._git_paths_match
    relative_plan = plan_path.relative_to(Path(__file__).resolve().parents[2]).as_posix()
    sealed_plan_bytes = plan_path.read_bytes()

    def fake_text(*arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return "c" * 40
        if arguments == ("branch", "--show-current"):
            return "test-branch"
        if arguments[0:1] == ("rev-parse",) and arguments[1].startswith("refs/tags/"):
            return "c" * 40
        if arguments == ("rev-list", "--parents", "-n", "1", "c" * 40):
            return f"{'c' * 40} {'b' * 40}"
        if arguments == ("rev-list", "--parents", "-n", "1", "b" * 40):
            return f"{'b' * 40} {'a' * 40}"
        if arguments == ("diff", "--name-only", "b" * 40, "c" * 40):
            return relative_plan
        raise AssertionError(f"unexpected fake Git query: {arguments}")

    analysis._git_text = fake_text
    analysis._git_is_ancestor = lambda ancestor, descendant: (
        ancestor == "a" * 40 and descendant == "b" * 40
    )
    analysis._git_bytes = lambda *arguments: sealed_plan_bytes
    analysis._git_paths_match = lambda commit, paths: commit == "c" * 40
    return real_text, real_ancestor, real_bytes, real_paths_match


def test_plan_can_be_sealed_before_evidence_and_rejects_drift() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    logs = repository_root / "logs"
    logs.mkdir(exist_ok=True)
    metadata = _metadata(1024, "a")
    metadata["git"]["commit"] = "c" * 40
    with tempfile.TemporaryDirectory(dir=logs) as directory:
        root = Path(directory)
        plan_value = _plan(repository_root, root, metadata)
        plan_path = root / "plan.json"
        _write_json(plan_path, plan_value)
        real_text, real_ancestor, real_bytes, real_paths_match = _install_fake_git(
            plan_path
        )
        try:
            loaded = load_analysis_plan(plan_path)
            assert loaded["_seal_commit"] == "c" * 40
            assert not loaded["data"]["runs"][0]["artifact"].exists()

            dirty_plan = copy.deepcopy(plan_value)
            dirty_plan["acceptance"]["conditional_success_point_min"] = 0.0
            _write_json(plan_path, dirty_plan)
            _expect(ValueError, load_analysis_plan, plan_path)
            _write_json(plan_path, plan_value)

            corrupt = copy.deepcopy(plan_value)
            corrupt["extra"] = True
            corrupt_path = root / "extra.json"
            _write_json(corrupt_path, corrupt)
            _expect(ValueError, load_analysis_plan, corrupt_path)

            corrupt = copy.deepcopy(plan_value)
            corrupt["data"]["runs"][2], corrupt["data"]["runs"][3] = (
                corrupt["data"]["runs"][3],
                corrupt["data"]["runs"][2],
            )
            corrupt_path = root / "order.json"
            _write_json(corrupt_path, corrupt)
            _expect(ValueError, load_analysis_plan, corrupt_path)

            corrupt = copy.deepcopy(plan_value)
            corrupt["inference"]["bootstrap"]["lower_rank"] = 500
            corrupt_path = root / "rank.json"
            _write_json(corrupt_path, corrupt)
            _expect(ValueError, load_analysis_plan, corrupt_path)
        finally:
            analysis._git_text = real_text
            analysis._git_is_ancestor = real_ancestor
            analysis._git_bytes = real_bytes
            analysis._git_paths_match = real_paths_match


def test_complete_analysis_and_identity_fail_closed() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    logs = repository_root / "logs"
    logs.mkdir(exist_ok=True)
    base_metadata = _metadata(1024, "a")
    base_metadata["git"]["commit"] = "c" * 40
    with tempfile.TemporaryDirectory(dir=logs) as directory:
        evidence_root = Path(directory)
        plan_value = _plan(repository_root, evidence_root, base_metadata)
        plan_path = evidence_root / "plan.json"
        _write_json(plan_path, plan_value)

        for seed, replicate in PRODUCTION_RUN_ORDER:
            metadata = copy.deepcopy(base_metadata)
            metadata["seed"] = seed
            metadata["replicate"] = replicate
            metadata["runtime"]["seed"] = seed
            artifact = build_artifact(
                metadata=metadata, tensors=_tensors(seed, replicate)
            )
            artifact_path = evidence_root / f"trial_{seed}_{replicate}.pt"
            report_path = evidence_root / f"trial_{seed}_{replicate}.json"
            publish_artifact_and_report_no_clobber(
                artifact,
                _report(artifact),
                artifact_output=artifact_path,
                report_output=report_path,
            )

        real_text, real_ancestor, real_bytes, real_paths_match = _install_fake_git(
            plan_path
        )
        try:
            plan = load_analysis_plan(plan_path)
            evidence = load_evidence(plan)
            assert evidence["cube"]["success"].shape == (3, 2, 1024)

            wrong_identity = copy.deepcopy(plan)
            wrong_identity["data"]["identity"]["candidate_actor_sha256"] = "f" * 64
            _expect(ValueError, load_evidence, wrong_identity)

            report = analyze(plan)
            assert report["decision"]["passed"] is True
            assert all(report["decision"]["checks"].values())
            assert report["decision"]["positive_seed_count"] == 3
            assert report["decision"]["seed_meets_both_success_margins_count"] == 3
            assert (
                report["point_estimates"]["horvitz_thompson"]
                ["conditional_triggered"]["events"]["success"]["delta"]
                >= 0.05
            )
            assert (
                report["fisher_randomization"]["p_values"]
                ["conditional_triggered"]["success"]["greater"]
                <= 0.05
            )
            for event in ADVERSE_EVENTS:
                safety = report["decision"]["safety"][event]
                assert safety["underpowered"] is True
                assert safety["rule"] == "candidate_count_fallback"
                assert safety["candidate_events"] == 0
                assert "itt_ht_interval" in safety
        finally:
            analysis._git_text = real_text
            analysis._git_is_ancestor = real_ancestor
            analysis._git_bytes = real_bytes
            analysis._git_paths_match = real_paths_match


def test_analysis_publication_rolls_back_after_linked_base_exception() -> None:
    class InjectedAbort(BaseException):
        pass

    repository_root = Path(__file__).resolve().parents[2]
    logs = repository_root / "logs"
    logs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=logs) as directory:
        output = Path(directory) / "analysis.json"
        real_fsync = analysis.os.fsync
        calls = 0

        def abort_directory_sync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InjectedAbort()
            real_fsync(descriptor)

        analysis.os.fsync = abort_directory_sync
        try:
            _expect(InjectedAbort, analysis._publish, output, {"ok": True})
        finally:
            analysis.os.fsync = real_fsync
        assert not output.exists()
        assert list(Path(directory).iterdir()) == []


def main() -> int:
    test_plan_can_be_sealed_before_evidence_and_rejects_drift()
    test_complete_analysis_and_identity_fail_closed()
    test_analysis_publication_rolls_back_after_linked_base_exception()
    print("analyze_online_close_ab_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
