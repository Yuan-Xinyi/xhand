#!/usr/bin/env python3
"""Simulation-free negative and lifecycle tests for factual gate evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from public_route_gate_dataset import (
    DEFAULT_ANALYSIS_PLAN,
    DEFAULT_MANIFEST,
    build_evidence_ledger,
    load_analysis_plan,
    load_registered_factual_cohort,
    publish_evidence_ledger_no_clobber,
    parse_args,
    semantic_dataset_sha256,
    sha256_file,
    validate_collection_acceptance,
    validate_evidence_ledger,
    validate_factual_dataset,
)
from public_route_trial_contract import assignment_for, build_trial_artifact


def _expect_error(
    error_type: type[BaseException], function: Callable[..., Any], *args: Any, **kwargs: Any
) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _strict_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _synthetic_contract(root: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    manifest["outputs"]["repository_relative_root"] = "trial"
    manifest["cohorts"]["pilot"]["seeds"] = []
    manifest["cohorts"]["train"]["seeds"] = [10]
    manifest["cohorts"]["development"]["seeds"] = [11]
    manifest["cohorts"]["blind"]["seeds"] = []
    for entry in manifest["cohorts"].values():
        entry["num_envs"] = 8
    thresholds = manifest["collection_acceptance"]
    thresholds.update(
        {
            "train_min_factual_rows_per_arm": 8,
            "train_min_factual_rows_per_seed_arm": 8,
            "train_min_factual_rows_total": 16,
            "development_min_factual_rows_per_arm": 8,
            "development_min_factual_rows_per_seed_arm": 8,
            "development_min_factual_rows_total": 16,
            "minimum_rows_per_stratum_arm": 0,
        }
    )
    manifest_path = root / "manifest.json"
    _strict_write_json(manifest_path, manifest)

    plan = json.loads(DEFAULT_ANALYSIS_PLAN.read_text(encoding="utf-8"))
    plan["status"] = "preregistered"
    plan["data"]["train_seeds"] = [10]
    plan["development"]["seeds"] = [11]
    plan["preregistration"]["implementation_commit"] = "1" * 40
    plan["preregistration"]["trial_manifest_sha256"] = sha256_file(manifest_path)
    plan["preregistration"]["train_ledger_tag"] = (
        "pick-tool-public-route-train-ledger-v1-20260722"
    )
    plan["preregistration"]["development_ledger_tag"] = (
        "pick-tool-public-route-development-ledger-v1-20260722"
    )
    plan["preregistration"]["frozen_model_tag"] = (
        "pick-tool-public-route-frozen-gate-v1-20260722"
    )
    plan_path = root / "analysis_plan.json"
    _strict_write_json(plan_path, plan)
    return manifest_path, plan_path, manifest, plan


def _feature(step: int, env_slot: int) -> torch.Tensor:
    feature = torch.zeros(165, dtype=torch.float32)
    # Public readiness: palm plus two non-thumb contacts.  Alternate the close
    # bit only to exercise deterministic stratum recomputation.
    feature[91:93] = 0.2
    feature[95] = 0.2
    feature[101] = 0.2 if env_slot % 2 else 0.1
    feature[104] = 1.0
    feature[105] = 0.0
    feature[118] = 1.0
    feature[120] = float(step) / 999.0
    return feature


def _rows(cohort: str, seed: int, replicate: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    assignment = assignment_for(
        cohort, seed, 8, manifest["assignment"]["salt"], replicate
    )
    rows: list[dict[str, Any]] = []
    for env_slot in reversed(range(8)):
        close = env_slot % 2 == 1
        rows.append(
            {
                "feature": _feature(3, env_slot),
                "env_slot": env_slot,
                "slot_episode_index": 0,
                "episode_index": env_slot,
                "candidate_step": 3,
                "stratum": 1 if close else 0,
                "outcome_episode_length": 999,
                "readiness_score": 0.2,
                "outcome_max_true_clearance_m": 0.0,
                "factual_treatment_route": bool(assignment[env_slot]),
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
        )
    return rows


def _summary(artifact: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    tensors = artifact["tensors"]
    count = int(mask.sum())
    return {
        "episodes": count,
        "success": int(tensors["outcome_success"][mask].sum()),
        "failure": int(tensors["outcome_failure"][mask].sum()),
        "time_out": int(tensors["outcome_time_out"][mask].sum()),
        "dropped": int(tensors["outcome_dropped"][mask].sum()),
        "unsafe_force": int(tensors["outcome_unsafe_force"][mask].sum()),
        "ever_grasped": int(tensors["outcome_ever_grasped"][mask].sum()),
        "ever_clearance_ge_5cm": 0,
        "max_true_clearance_m": 0.0,
    }


def _write_run(
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    cohort: str,
    seed: int,
    replicate: str,
) -> tuple[Path, Path]:
    manifest_sha = sha256_file(manifest_path)
    checkpoint = manifest["checkpoints"]
    common_source = {
        "sealed_source.py": "2" * 64,
        "scripts/flashsac/public_route_trial_manifest.json": manifest_sha,
    }
    common_runtime = dict(manifest["runtime_assets"]["files"])
    common_git = {
        "commit": "4" * 40,
        "branch": manifest["preregistration"]["branch"],
        "seal_tag": manifest["preregistration"]["seal_tag"],
        "seal_commit": "5" * 40,
        "implementation_commit": manifest["preregistration"]["implementation_commit"],
        "source_files_dirty": False,
        "flashsac_commit": manifest["flashsac"]["fork_commit"],
        "flashsac_dirty": False,
    }
    provenance = {
        "kind": "pick_tool_public_route_randomized_factual_trial_v2",
        "format_version": 2,
        "cohort": cohort,
        "seed": seed,
        "replicate": replicate,
        "num_envs": 8,
        "episodes": 8,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "assignment_algorithm": manifest["assignment"]["algorithm"],
        "assignment_salt": manifest["assignment"]["salt"],
        "known_route_propensity": 0.5,
        "pairing_semantics": manifest["claim_boundary"]["pairing_semantics"],
        "causal_counterfactual_claim_allowed": False,
        "feature_contract": manifest["feature_contract"],
        "gate_contract": manifest["gate"],
        "task_contract": manifest["task"],
        "search_checkpoint": "search.pt",
        "search_checkpoint_sha256": checkpoint["search"]["sha256"],
        "route_checkpoint": "route",
        "route_actor_sha256": checkpoint["route_v6"]["actor_sha256"],
        "route_task_contract_sha256": checkpoint["route_v6"]["task_contract_sha256"],
        "route_frozen_actor_sha256": checkpoint["route_v6"]["frozen_lift_actor_sha256"],
        "route_bridge_state_sha256": checkpoint["route_v6"]["torch_bridge_state_sha256"],
        "route_load": {"test": True},
        "source_sha256": common_source,
        "runtime_asset_sha256": common_runtime,
        "git": common_git,
        "runtime": {"seed": seed},
        "collection_semantics": "test factual collection",
    }
    assignment = assignment_for(
        cohort, seed, 8, manifest["assignment"]["salt"], replicate
    )
    artifact = build_trial_artifact(
        _rows(cohort, seed, replicate, manifest),
        metadata={
            "seed": seed,
            "num_envs": 8,
            "assignment_salt": manifest["assignment"]["salt"],
            "replicate": replicate,
            "provenance": provenance,
        },
        assignment_route=assignment,
    )
    output_root = root / "trial"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_path = output_root / f"{cohort}_s{seed}_{replicate}.pt"
    report_path = output_root / f"{cohort}_s{seed}_{replicate}.json"
    torch.save(artifact, artifact_path)
    route = artifact["tensors"]["factual_treatment_route"]
    report = {
        "kind": "pick_tool_public_route_trial_collection_report_v2",
        "status": "complete",
        "cohort": cohort,
        "seed": seed,
        "replicate": replicate,
        "num_envs": 8,
        "vector_steps": 999,
        "candidate_rows": 8,
        "untriggered_episodes": 0,
        "candidate_route_rows": 4,
        "candidate_continue_rows": 4,
        "candidate_route_outcomes": _summary(artifact, route),
        "candidate_continue_outcomes": _summary(artifact, ~route),
        "all_assigned_route_outcomes": _summary(artifact, route),
        "all_assigned_continue_outcomes": _summary(artifact, ~route),
        "assignment_route_slots": 4,
        "assignment_continue_slots": 4,
        "manifest_sha256": manifest_sha,
        "source_sha256": common_source,
        "runtime_asset_sha256": common_runtime,
        "git": common_git,
        "claim_boundary": manifest["claim_boundary"],
        "artifact_sha256": sha256_file(artifact_path),
    }
    _strict_write_json(report_path, report)
    return artifact_path, report_path


def _complete_synthetic(root: Path, cohort: str = "train") -> tuple[Path, Path, dict[str, Any]]:
    manifest_path, plan_path, manifest, _ = _synthetic_contract(root)
    seed = 10 if cohort == "train" else 11
    for replicate in ("a", "b"):
        _write_run(
            root,
            manifest_path,
            manifest,
            cohort=cohort,
            seed=seed,
            replicate=replicate,
        )
    return manifest_path, plan_path, manifest


def _load(root: Path, cohort: str = "train") -> dict[str, Any]:
    return load_registered_factual_cohort(
        cohort,
        manifest_path=root / "manifest.json",
        analysis_plan_path=root / "analysis_plan.json",
        repository_root=root,
    )


def test_plan_lifecycle_and_canonical_dataset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _complete_synthetic(root)
        plan = load_analysis_plan(root / "analysis_plan.json")
        assert plan["status"] == "preregistered"
        dataset = _load(root)
        assert dataset["tensors"]["feature"].shape == (16, 165)
        assert dataset["tensors"]["env_slot"].tolist() == list(range(8)) * 2
        assert dataset["tensors"]["replicate_index"].tolist() == [0] * 8 + [1] * 8
        assert len(dataset["receipts"]) == 2
        assert semantic_dataset_sha256(dataset) == semantic_dataset_sha256(
            validate_factual_dataset(dataset, expected_cohort="train")
        )
        safe_path = root / "safe_dataset.pt"
        torch.save(dataset, safe_path)
        safe_loaded = torch.load(safe_path, map_location="cpu", weights_only=True)
        assert semantic_dataset_sha256(safe_loaded) == semantic_dataset_sha256(dataset)
        _expect_error(ValueError, validate_factual_dataset, dataset, expected_cohort="development")

        draft = json.loads((root / "analysis_plan.json").read_text())
        draft["status"] = "draft_until_seal_commit"
        draft["preregistration"]["implementation_commit"] = "SET_AFTER_DRAFT_IMPLEMENTATION_COMMIT"
        _strict_write_json(root / "draft.json", draft)
        _expect_error(ValueError, load_analysis_plan, root / "draft.json")
        assert load_analysis_plan(root / "draft.json", require_preregistered=False)["status"].startswith("draft")


def test_plan_freezes_optimizer_inference_normalization_and_resampling() -> None:
    mutations = (
        ("optimization", "adamw_eps", 1.0e-7),
        ("inference", "equality_is_continue", False),
        ("normalization", "population_std_correction", 1),
        ("randomization_inference", "safety_events_do_not_cancel", False),
    )
    for section, field, value in mutations:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, plan_path, _, _ = _synthetic_contract(root)
            plan = json.loads(plan_path.read_text())
            plan[section][field] = value
            _strict_write_json(plan_path, plan)
            _expect_error(ValueError, load_analysis_plan, plan_path)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, plan_path, _, _ = _synthetic_contract(root)
        plan = json.loads(plan_path.read_text())
        plan["randomization_inference"]["bootstrap"]["replicates"] = 19999
        _strict_write_json(plan_path, plan)
        _expect_error(ValueError, load_analysis_plan, plan_path)


def test_missing_pair_report_counts_and_cohort_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _complete_synthetic(root)
        missing = root / "trial/train_s10_b.pt"
        missing.unlink()
        _expect_error(FileNotFoundError, _load, root)

    for field, bad in (("candidate_rows", 7), ("cohort", "development")):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _complete_synthetic(root)
            report_path = root / "trial/train_s10_a.json"
            report = json.loads(report_path.read_text())
            report[field] = bad
            _strict_write_json(report_path, report)
            _expect_error(ValueError, _load, root)


def test_artifact_receipt_complement_and_symlink_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _complete_synthetic(root)
        report_path = root / "trial/train_s10_b.json"
        report = json.loads(report_path.read_text())
        report["artifact_sha256"] = "f" * 64
        _strict_write_json(report_path, report)
        _expect_error(ValueError, _load, root)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _complete_synthetic(root)
        report_path = root / "trial/train_s10_b.json"
        target = root / "copied.json"
        target.write_bytes(report_path.read_bytes())
        report_path.unlink()
        report_path.symlink_to(target)
        _expect_error((FileNotFoundError, ValueError), _load, root)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _complete_synthetic(root)
        artifact_path = root / "trial/train_s10_a.pt"
        report_path = root / "trial/train_s10_a.json"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        artifact["metadata"]["provenance"]["runtime_asset_sha256"] = {
            "substituted.usd": "3" * 64
        }
        torch.save(artifact, artifact_path)
        report = json.loads(report_path.read_text())
        report["artifact_sha256"] = sha256_file(artifact_path)
        report["runtime_asset_sha256"] = {"substituted.usd": "3" * 64}
        _strict_write_json(report_path, report)
        _expect_error(ValueError, _load, root)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _complete_synthetic(root)
        dataset = _load(root)
        corrupt = copy.deepcopy(dataset)
        corrupt["receipts"][1]["assignment_route_sha256"] = corrupt["receipts"][0][
            "assignment_route_sha256"
        ]
        _expect_error(ValueError, validate_factual_dataset, corrupt)


def test_deleted_or_reordered_rows_and_receipt_sha_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, _, manifest = _complete_synthetic(root)
        dataset = _load(root)
        deleted = copy.deepcopy(dataset)
        for name in deleted["tensors"]:
            deleted["tensors"][name] = deleted["tensors"][name][1:].contiguous()
        _expect_error(ValueError, validate_factual_dataset, deleted)

        reordered = copy.deepcopy(dataset)
        order = torch.arange(15, -1, -1)
        for name in reordered["tensors"]:
            reordered["tensors"][name] = reordered["tensors"][name][order].contiguous()
        _expect_error(ValueError, validate_factual_dataset, reordered)

        wrong_sha = copy.deepcopy(dataset)
        wrong_sha["receipts"][0]["artifact_sha256"] = "0" * 64
        assert semantic_dataset_sha256(wrong_sha) != semantic_dataset_sha256(dataset)

        accepted = validate_collection_acceptance(dataset, manifest)
        assert accepted["accepted"] is True
        strict = copy.deepcopy(manifest)
        strict["collection_acceptance"]["minimum_rows_per_stratum_arm"] = 99
        rejected = validate_collection_acceptance(dataset, strict)
        assert rejected["accepted"] is False
        assert rejected["checks"]["rows_per_stratum_arm"] is False


def test_ledger_validation_publication_and_no_clobber() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path, plan_path, manifest = _complete_synthetic(root)
        dataset = _load(root)
        ledger = build_evidence_ledger(dataset, manifest)
        assert ledger["acceptance"]["accepted"] is True
        assert validate_evidence_ledger(ledger, dataset=dataset, plan_or_manifest=manifest) == ledger
        corrupt = copy.deepcopy(ledger)
        corrupt["row_count"] -= 1
        _expect_error(
            ValueError,
            validate_evidence_ledger,
            corrupt,
            dataset=dataset,
            plan_or_manifest=manifest,
        )
        published = publish_evidence_ledger_no_clobber(
            "train",
            dataset=dataset,
            manifest_path=manifest_path,
            analysis_plan_path=plan_path,
            repository_root=root,
        )
        output = root / json.loads(plan_path.read_text())["outputs"]["train_ledger"]
        assert json.loads(output.read_text()) == published
        _expect_error(
            FileExistsError,
            publish_evidence_ledger_no_clobber,
            "train",
            dataset=dataset,
            manifest_path=manifest_path,
            analysis_plan_path=plan_path,
            repository_root=root,
        )


def test_pilot_is_never_a_fit_dataset() -> None:
    _expect_error(ValueError, load_registered_factual_cohort, "pilot")
    assert parse_args(["--cohort", "train"]).cohort == "train"
    _expect_error(SystemExit, parse_args, ["--cohort", "pilot"])
    _expect_error(
        SystemExit,
        parse_args,
        ["--cohort", "train", "--manifest", "unsealed.json"],
    )


if __name__ == "__main__":
    test_plan_lifecycle_and_canonical_dataset()
    test_plan_freezes_optimizer_inference_normalization_and_resampling()
    test_missing_pair_report_counts_and_cohort_fail_closed()
    test_artifact_receipt_complement_and_symlink_fail_closed()
    test_deleted_or_reordered_rows_and_receipt_sha_fail_closed()
    test_ledger_validation_publication_and_no_clobber()
    test_pilot_is_never_a_fit_dataset()
    print("public route gate dataset tests passed")
