#!/usr/bin/env python3
"""Simulation-free tests for the preregistered public-route collector."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

from collect_public_route_trial import (
    COLLECTOR_REPORT_KIND,
    CollectionSpec,
    DEFAULT_MANIFEST,
    _event_summary,
    canonical_output_paths,
    load_collection_spec,
    next_failure_report_path,
    runtime_asset_fingerprints,
    sha256_file,
    validate_preregistered_run_order,
)


def _expect_error(
    error_type: type[BaseException], function: Callable[..., Any], *args: Any, **kwargs: Any
) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _sealed_manifest() -> dict[str, Any]:
    payload = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    payload["status"] = "preregistered"
    if not all(
        character in "0123456789abcdef"
        for character in payload["preregistration"]["implementation_commit"]
    ):
        payload["preregistration"]["implementation_commit"] = "0" * 40
    return payload


def _write_manifest(root: Path, payload: dict[str, Any]) -> Path:
    path = root / "manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _run_order_manifest(*, seeds: list[int]) -> dict[str, Any]:
    """Return the minimum manifest subset consumed by run-order helpers."""

    manifest = _sealed_manifest()
    manifest["outputs"] = {
        "artifact_template": "{cohort}_s{seed}_{replicate}.pt",
        "failure_report_template": (
            "{cohort}_s{seed}_{replicate}.failed_attempt_{attempt:03d}.json"
        ),
        "report_template": "{cohort}_s{seed}_{replicate}.json",
        "repository_relative_root": "trial_outputs",
        "run_order_enforced": True,
    }
    manifest["cohorts"] = {
        "pilot": {"seeds": seeds, "num_envs": 512},
        "train": {"seeds": [], "num_envs": 512},
        "development": {"seeds": [], "num_envs": 512},
    }
    return manifest


def _collection_spec(
    root: Path,
    *,
    seed: int,
    replicate: str,
    manifest: dict[str, Any] | None = None,
) -> CollectionSpec:
    manifest = manifest if manifest is not None else _run_order_manifest(seeds=[seed])
    return CollectionSpec(
        manifest=manifest,
        manifest_path=root / "manifest.json",
        manifest_sha256="a" * 64,
        repository_root=root.resolve(),
        cohort="pilot",
        seed=seed,
        replicate=replicate,
        num_envs=512,
        assignment_salt="test-assignment-salt",
        search_checkpoint=root / "search.pt",
        search_checkpoint_sha256="b" * 64,
        route_checkpoint=root / "route",
        route_actor_sha256="c" * 64,
        route_task_contract_sha256="d" * 64,
        route_frozen_actor_sha256="e" * 64,
        route_bridge_state_sha256="f" * 64,
    )


def _write_complete_receipt(spec: CollectionSpec) -> None:
    paths = canonical_output_paths(spec)
    paths.artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.artifact.write_bytes(
        f"{spec.cohort}:{spec.seed}:{spec.replicate}".encode("utf-8")
    )
    paths.report.write_text(
        json.dumps(
            {
                "kind": COLLECTOR_REPORT_KIND,
                "status": "complete",
                "cohort": spec.cohort,
                "seed": spec.seed,
                "replicate": spec.replicate,
                "num_envs": spec.num_envs,
                "manifest_sha256": spec.manifest_sha256,
                "artifact_sha256": sha256_file(paths.artifact),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_manifest_lifecycle_rejects_draft_and_loads_preregistered() -> None:
    current = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    if current["status"] == "draft_until_seal_commit":
        _expect_error(
            ValueError,
            load_collection_spec,
            DEFAULT_MANIFEST,
            cohort="pilot",
            seed=266,
            replicate="a",
        )
    elif current["status"] == "preregistered":
        spec = load_collection_spec(
            DEFAULT_MANIFEST, cohort="pilot", seed=266, replicate="a"
        )
        assert spec.num_envs == 512
    else:
        raise AssertionError("unexpected manifest lifecycle state")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        draft = _sealed_manifest()
        draft["status"] = "draft_until_seal_commit"
        draft_path = _write_manifest(root, draft)
        _expect_error(
            ValueError,
            load_collection_spec,
            draft_path,
            cohort="pilot",
            seed=266,
            replicate="a",
        )
        path = _write_manifest(root, _sealed_manifest())
        spec = load_collection_spec(
            path, cohort="pilot", seed=266, replicate="a"
        )
        assert spec.num_envs == 512
        assert spec.seed == 266 and spec.replicate == "a"
        assert len(spec.manifest_sha256) == 64


def test_manifest_rejects_cohort_leakage_and_contract_drift() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _sealed_manifest()
        path = _write_manifest(root, manifest)
        _expect_error(
            ValueError,
            load_collection_spec,
            path,
            cohort="blind",
            seed=278,
            replicate="a",
        )
        _expect_error(
            ValueError,
            load_collection_spec,
            path,
            cohort="train",
            seed=274,
            replicate="a",
        )
        corrupt = copy.deepcopy(manifest)
        corrupt["feature_contract"]["feature_dim"] = 164
        corrupt_path = root / "corrupt.json"
        corrupt_path.write_text(json.dumps(corrupt), encoding="utf-8")
        _expect_error(
            ValueError,
            load_collection_spec,
            corrupt_path,
            cohort="pilot",
            seed=266,
            replicate="a",
        )


def test_manifest_scale_and_seed_partitions_are_frozen() -> None:
    manifest = _sealed_manifest()
    cohorts = manifest["cohorts"]
    assert {entry["num_envs"] for entry in cohorts.values()} == {512}
    seed_sets = [set(entry["seeds"]) for entry in cohorts.values()]
    for index, left in enumerate(seed_sets):
        for right in seed_sets[index + 1 :]:
            assert not left & right
    assert cohorts["blind"]["evaluations_after_model_freeze"] == [
        "fixed_continue",
        "fixed_route",
        "frozen_gate",
    ]


def test_factual_and_tracker_summaries_use_their_own_schema() -> None:
    factual = [
        {
            "outcome_success": True,
            "outcome_failure": False,
            "outcome_time_out": False,
            "outcome_dropped": False,
            "outcome_unsafe_force": False,
            "outcome_ever_grasped": True,
            "outcome_max_true_clearance_m": 0.2,
        }
    ]
    summary = _event_summary(factual)
    assert summary["success"] == 1 and summary["ever_clearance_ge_5cm"] == 1
    tracker = [
        {
            "success": False,
            "failure": False,
            "time_out": True,
            "dropped": False,
            "unsafe_force": False,
            "ever_grasped": False,
            "ever_clearance_ge_5cm": False,
            "max_true_clearance_m": 0.0,
        }
    ]
    summary = _event_summary(tracker)
    assert summary["time_out"] == 1 and summary["success"] == 0


def test_treatment_is_consulted_only_after_candidate_snapshot() -> None:
    source = Path(__file__).with_name("collect_public_route_trial.py").read_text(
        encoding="utf-8"
    )
    assignment_generation = source.index("assignment_cpu = assignment_route_mask(")
    task_import = source.index("from adapter import make_pick_tool_env", assignment_generation)
    environment_creation = source.index("env = make_pick_tool_env(", task_import)
    assert assignment_generation < task_import < environment_creation
    route_candidate = source.index("route_agent.sample_actions(")
    feature_build = source.index("features = build_public_route_feature(", route_candidate)
    snapshot = source.index("candidate_feature[trigger] = features[trigger]", feature_build)
    assignment_use = source.index("route_active |= trigger & assignment_route", snapshot)
    step = source.index("env.step(action)", assignment_use)
    assert route_candidate < feature_build < snapshot < assignment_use < step
    assert 'training=False' in source[route_candidate : route_candidate + 300]
    assert "_diagnostic_pregrasp_measurements" not in source


def test_canonical_output_paths_are_exact_and_cannot_escape() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = _collection_spec(root, seed=266, replicate="a")
        paths = canonical_output_paths(spec)
        output_root = root.resolve() / "trial_outputs"
        assert paths.artifact == output_root / "pilot_s266_a.pt"
        assert paths.report == output_root / "pilot_s266_a.json"

        escaping_root = copy.deepcopy(spec.manifest)
        escaping_root["outputs"]["repository_relative_root"] = "../outside"
        _expect_error(
            ValueError,
            canonical_output_paths,
            _collection_spec(
                root, seed=266, replicate="a", manifest=escaping_root
            ),
        )

        nested_artifact = copy.deepcopy(spec.manifest)
        nested_artifact["outputs"]["artifact_template"] = (
            "nested/{cohort}_s{seed}_{replicate}.pt"
        )
        _expect_error(
            RuntimeError,
            canonical_output_paths,
            _collection_spec(
                root, seed=266, replicate="a", manifest=nested_artifact
            ),
        )


def test_even_seed_requires_a_then_accepts_b_with_complete_receipt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _run_order_manifest(seeds=[266])
        first = _collection_spec(
            root, seed=266, replicate="a", manifest=manifest
        )
        second = _collection_spec(
            root, seed=266, replicate="b", manifest=manifest
        )

        assert validate_preregistered_run_order(first) == canonical_output_paths(first)
        _expect_error(RuntimeError, validate_preregistered_run_order, second)
        _write_complete_receipt(first)
        assert validate_preregistered_run_order(second) == canonical_output_paths(second)


def test_odd_seed_requires_b_then_accepts_a_with_complete_receipt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _run_order_manifest(seeds=[267])
        first = _collection_spec(
            root, seed=267, replicate="b", manifest=manifest
        )
        second = _collection_spec(
            root, seed=267, replicate="a", manifest=manifest
        )

        assert validate_preregistered_run_order(first) == canonical_output_paths(first)
        _expect_error(RuntimeError, validate_preregistered_run_order, second)
        _write_complete_receipt(first)
        assert validate_preregistered_run_order(second) == canonical_output_paths(second)


def test_run_order_fails_closed_for_partial_current_or_future_outputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = _collection_spec(root / "artifact_only", seed=266, replicate="a")
        paths = canonical_output_paths(spec)
        paths.artifact.parent.mkdir(parents=True)
        paths.artifact.write_bytes(b"partial")
        _expect_error(FileExistsError, validate_preregistered_run_order, spec)

        spec = _collection_spec(root / "report_only", seed=266, replicate="a")
        paths = canonical_output_paths(spec)
        paths.report.parent.mkdir(parents=True)
        paths.report.write_text("{}", encoding="utf-8")
        _expect_error(FileExistsError, validate_preregistered_run_order, spec)

        future_root = root / "future_partial"
        manifest = _run_order_manifest(seeds=[266])
        current = _collection_spec(
            future_root, seed=266, replicate="a", manifest=manifest
        )
        future = _collection_spec(
            future_root, seed=266, replicate="b", manifest=manifest
        )
        future_paths = canonical_output_paths(future)
        future_paths.report.parent.mkdir(parents=True)
        future_paths.report.write_text("{}", encoding="utf-8")
        _expect_error(RuntimeError, validate_preregistered_run_order, current)


def test_run_order_rejects_incomplete_or_tampered_predecessor_receipt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _run_order_manifest(seeds=[266])
        first = _collection_spec(
            root, seed=266, replicate="a", manifest=manifest
        )
        second = _collection_spec(
            root, seed=266, replicate="b", manifest=manifest
        )
        _write_complete_receipt(first)
        first_paths = canonical_output_paths(first)
        receipt = json.loads(first_paths.report.read_text(encoding="utf-8"))
        receipt["artifact_sha256"] = "0" * 64
        first_paths.report.write_text(json.dumps(receipt), encoding="utf-8")
        _expect_error(RuntimeError, validate_preregistered_run_order, second)


def test_failure_reports_use_first_unowned_number_without_clobbering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = _collection_spec(root, seed=267, replicate="b")
        output_root = root.resolve() / "trial_outputs"
        output_root.mkdir(parents=True)

        attempt_1 = output_root / "pilot_s267_b.failed_attempt_001.json"
        attempt_2 = output_root / "pilot_s267_b.failed_attempt_002.json"
        attempt_3 = output_root / "pilot_s267_b.failed_attempt_003.json"
        assert next_failure_report_path(spec) == attempt_1
        attempt_1.write_text("{}", encoding="utf-8")
        assert next_failure_report_path(spec) == attempt_2
        attempt_2.symlink_to(output_root / "missing-failure-receipt.json")
        assert next_failure_report_path(spec) == attempt_3

        escaping = copy.deepcopy(spec.manifest)
        escaping["outputs"]["failure_report_template"] = (
            "../{cohort}_s{seed}_{replicate}.failed_attempt_{attempt:03d}.json"
        )
        _expect_error(
            RuntimeError,
            next_failure_report_path,
            _collection_spec(root, seed=267, replicate="b", manifest=escaping),
        )


def test_runtime_assets_require_exact_regular_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        relative = root / "ignored_robot.usd"
        external = root / "cached_tool.usd"
        relative.write_bytes(b"robot-usd")
        external.write_bytes(b"tool-usd")
        spec = _collection_spec(root, seed=266, replicate="a")
        spec.manifest["runtime_assets"] = {
            "files": {
                "ignored_robot.usd": sha256_file(relative),
                str(external): sha256_file(external),
            }
        }
        assert runtime_asset_fingerprints(spec) == spec.manifest["runtime_assets"]["files"]

        external.write_bytes(b"mutated")
        _expect_error(RuntimeError, runtime_asset_fingerprints, spec)
        external.unlink()
        external.symlink_to(relative)
        _expect_error(FileNotFoundError, runtime_asset_fingerprints, spec)


if __name__ == "__main__":
    test_manifest_lifecycle_rejects_draft_and_loads_preregistered()
    test_manifest_rejects_cohort_leakage_and_contract_drift()
    test_manifest_scale_and_seed_partitions_are_frozen()
    test_factual_and_tracker_summaries_use_their_own_schema()
    test_treatment_is_consulted_only_after_candidate_snapshot()
    test_canonical_output_paths_are_exact_and_cannot_escape()
    test_even_seed_requires_a_then_accepts_b_with_complete_receipt()
    test_odd_seed_requires_b_then_accepts_a_with_complete_receipt()
    test_run_order_fails_closed_for_partial_current_or_future_outputs()
    test_run_order_rejects_incomplete_or_tampered_predecessor_receipt()
    test_failure_reports_use_first_unowned_number_without_clobbering()
    test_runtime_assets_require_exact_regular_bytes()
