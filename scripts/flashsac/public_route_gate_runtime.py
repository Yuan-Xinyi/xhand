#!/usr/bin/env python3
"""Fail-closed runtime for the frozen public-route gate.

This module is intentionally narrower than a deployment wrapper.  The model
may be evaluated only on the registered development cohort.  A later blind
runtime must additionally consume a passing immutable development audit; this
module never opens blind collection by itself.

The raw ``.pt`` digest written into the training report is the model identity.
The loader checks that receipt before ``weights_only`` deserialization, binds
the model metadata to the analysis plan, and rechecks both files whenever the
runtime is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from public_route_gate_dataset import (
    DEFAULT_ANALYSIS_PLAN,
    _require_nonsymlink_chain,
    _snapshot_regular_bytes,
    load_analysis_plan,
)
from train_public_route_gate import (
    HEADS,
    REPORT_KIND,
    analysis_seal_provenance,
    decision_from_probabilities,
    ensemble_probabilities,
    semantic_model_sha256,
    validate_gate_model,
)


RUNTIME_KIND = "pick_tool_public_route_gate_runtime_v1"
EXPECTED_COHORT = "development"
TRAINER_SOURCE_KEY = "scripts/flashsac/train_public_route_gate.py"

_TRAINING_REPORT_FIELDS = {
    "analysis_git",
    "analysis_plan_sha256",
    "collection_acceptance",
    "dataset_semantic_sha256",
    "factual_arm_rows",
    "factual_event_counts",
    "format_version",
    "kind",
    "input_ledger_sha256",
    "input_ledger_tag",
    "member_diagnostics",
    "model_semantic_sha256",
    "model_sha256",
    "status",
    "train_continue_decisions",
    "train_continue_rate",
    "training_rows",
}


def _git(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _semantic_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_regular_file(path: Path, label: str) -> Path:
    path = _absolute(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _read_json_unchanged(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _require_regular_file(path, label)
    raw = _snapshot_regular_bytes(path, label)
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{label} root must be a JSON object")
    return payload, digest


def _canonical_output_path(repository_root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a repository-relative path")
    root = _absolute(repository_root)
    path = _absolute(root / relative)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return path


def _validate_training_report(
    report: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    model_sha256: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if set(report) != _TRAINING_REPORT_FIELDS:
        raise ValueError("gate training report schema is not exact")
    if (
        report.get("kind") != REPORT_KIND
        or report.get("format_version") != 1
        or report.get("status") != "complete"
        or report.get("model_sha256") != model_sha256
        or report.get("analysis_plan_sha256") != plan["_sha256"]
        or report.get("input_ledger_sha256")
        != model["metadata"]["input_ledger_sha256"]
        or report.get("input_ledger_tag")
        != model["metadata"]["input_ledger_tag"]
    ):
        raise ValueError("gate training report identity or receipt is invalid")
    if not _is_sha256(report.get("dataset_semantic_sha256")) or not _is_sha256(
        report.get("model_semantic_sha256")
    ):
        raise ValueError("gate training report contains an invalid semantic digest")
    metadata = model["metadata"]
    if (
        report["dataset_semantic_sha256"] != metadata["dataset_semantic_sha256"]
        or report["model_semantic_sha256"] != semantic_model_sha256(model)
        or report.get("training_rows") != metadata["training_rows"]
    ):
        raise ValueError("gate training report disagrees with the frozen model")
    if type(report["training_rows"]) is not int or report["training_rows"] < 1:
        raise ValueError("gate training report row count is invalid")
    if (
        type(report["train_continue_decisions"]) is not int
        or not 0 <= report["train_continue_decisions"] <= report["training_rows"]
        or type(report["train_continue_rate"]) not in (int, float)
        or not math.isfinite(float(report["train_continue_rate"]))
        or not math.isclose(
            float(report["train_continue_rate"]),
            report["train_continue_decisions"] / report["training_rows"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("gate training report decision summary is invalid")
    acceptance = report.get("collection_acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("cohort") != "train":
        raise ValueError("gate was not fitted on an accepted train cohort")
    if acceptance.get("accepted") is not True:
        raise ValueError("gate train cohort failed its preregistered acceptance")
    diagnostics = report.get("member_diagnostics")
    if (
        not isinstance(diagnostics, list)
        or len(diagnostics) != 5
        or [item.get("member_seed") for item in diagnostics if isinstance(item, dict)]
        != [0, 1, 2, 3, 4]
    ):
        raise ValueError("gate training report member diagnostics are invalid")
    analysis_git = report.get("analysis_git")
    expected_analysis_git_fields = {
        "branch",
        "current_commit",
        "analysis_seal_tag",
        "analysis_seal_commit",
        "implementation_commit",
        "trial_seal_tag",
        "trial_seal_commit",
        "source_sha256",
    }
    if not isinstance(analysis_git, dict) or set(analysis_git) != expected_analysis_git_fields:
        raise ValueError("gate training report lacks sealed analysis provenance")
    if (
        analysis_git.get("analysis_seal_tag")
        != plan["preregistration"]["seal_tag"]
        or analysis_git.get("implementation_commit")
        != plan["preregistration"]["implementation_commit"]
    ):
        raise ValueError("gate training report differs from the analysis seal")
    source = analysis_git.get("source_sha256")
    if (
        not isinstance(source, dict)
        or source.get(TRAINER_SOURCE_KEY) != metadata["trainer_source_sha256"]
        or any(
            not isinstance(name, str) or not _is_sha256(digest)
            for name, digest in source.items()
        )
    ):
        raise ValueError("gate trainer source receipt is not frozen")
    commit_names = [
        "current_commit",
        "analysis_seal_commit",
        "trial_seal_commit",
    ]
    if plan.get("status") == "preregistered":
        commit_names.append("implementation_commit")
    for name in commit_names:
        value = analysis_git.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"gate training report {name} is not a Git commit")
    # Copy through strict JSON to detach caller-owned subclasses and reject NaN.
    return json.loads(json.dumps(report, sort_keys=True, allow_nan=False))


def _validate_model_plan_binding(
    model: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    metadata = model["metadata"]
    registration = plan["preregistration"]
    if (
        metadata["analysis_plan_sha256"] != plan["_sha256"]
        or metadata["analysis_seal_tag"] != registration["seal_tag"]
        or metadata["collection_manifest_sha256"]
        != registration["trial_manifest_sha256"]
        or metadata["collection_seal_tag"] != registration["trial_seal_tag"]
        or metadata["trained_cohort"] != "train"
        or metadata["excluded_cohorts"] != ["pilot", "development", "blind"]
        or metadata["head_order"] != list(HEADS)
        or metadata["normalization_contract"] != plan["normalization"]
        or metadata["optimizer_contract"] != plan["optimization"]
        or metadata["decision_contract"] != plan["inference"]
        or metadata["input_ledger_tag"]
        != registration["train_ledger_tag"]
    ):
        raise ValueError("frozen gate metadata is not bound to the analysis plan")
    receipts = metadata.get("run_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("frozen gate lacks immutable train-run receipts")
    if any(
        not isinstance(receipt, dict) or receipt.get("cohort") != "train"
        for receipt in receipts
    ):
        raise ValueError("frozen gate contains a non-train run receipt")


def _validate_live_analysis_binding(
    report: Mapping[str, Any], live: Mapping[str, Any]
) -> None:
    recorded = report["analysis_git"]
    for name in (
        "branch",
        "analysis_seal_tag",
        "analysis_seal_commit",
        "implementation_commit",
        "trial_seal_tag",
        "trial_seal_commit",
    ):
        if recorded.get(name) != live.get(name):
            raise RuntimeError(f"training report analysis provenance drifted on {name}")
    if recorded.get("source_sha256") != live.get("source_sha256"):
        raise RuntimeError("training report source receipts differ from the analysis seal")


def _require_tagged_frozen_inputs(
    *,
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    repository_root: Path,
    model_path: Path,
    report_path: Path,
) -> dict[str, str]:
    """Verify the train ledger and frozen model/report annotated receipts."""

    root = repository_root.resolve()
    relative_model = model_path.relative_to(root).as_posix()
    relative_report = report_path.relative_to(root).as_posix()
    ledger_path = _canonical_output_path(
        root, plan["outputs"]["train_ledger"], "train evidence ledger"
    )
    _require_nonsymlink_chain(ledger_path, root, "train evidence ledger")
    ledger_path = _require_regular_file(ledger_path, "train evidence ledger")
    relative_ledger = ledger_path.relative_to(root).as_posix()
    ledger, ledger_sha = _read_json_unchanged(ledger_path, "train evidence ledger")
    metadata = model["metadata"]
    if (
        ledger_sha != metadata["input_ledger_sha256"]
        or ledger_sha != report["input_ledger_sha256"]
        or ledger.get("kind") != "pick_tool_public_route_evidence_ledger_v1"
        or ledger.get("format_version") != 1
        or ledger.get("status") != "complete"
        or ledger.get("cohort") != "train"
        or ledger.get("analysis_plan_sha256") != plan["_sha256"]
        or ledger.get("manifest_sha256")
        != plan["preregistration"]["trial_manifest_sha256"]
        or ledger.get("dataset_semantic_sha256")
        != metadata["dataset_semantic_sha256"]
        or ledger.get("input_bundle_sha256")
        != metadata["dataset_semantic_sha256"]
        or ledger.get("row_count") != metadata["training_rows"]
        or ledger.get("row_count") != report["training_rows"]
        or ledger.get("expected_runs") != len(metadata["run_receipts"])
        or ledger.get("receipts") != metadata["run_receipts"]
        or ledger.get("acceptance") != report["collection_acceptance"]
        or ledger.get("collection_seal_tag")
        != plan["preregistration"]["trial_seal_tag"]
        or ledger.get("analysis_seal_tag")
        != plan["preregistration"]["seal_tag"]
        or ledger.get("ledger_tag") != plan["preregistration"]["train_ledger_tag"]
        or ledger.get("acceptance", {}).get("accepted") is not True
    ):
        raise ValueError("frozen model is not bound to the accepted tagged train ledger")

    ledger_tag = plan["preregistration"]["train_ledger_tag"]
    ledger_ref = f"refs/tags/{ledger_tag}"
    if _git(root, "cat-file", "-t", ledger_ref) != "tag":
        raise RuntimeError("train ledger tag must be annotated")
    ledger_commit = _git(root, "rev-parse", f"{ledger_ref}^{{commit}}")
    if report["analysis_git"]["current_commit"] != ledger_commit:
        raise RuntimeError("gate training did not start at the tagged train ledger commit")
    analysis_ref = f"refs/tags/{plan['preregistration']['seal_tag']}"
    if _git(root, "cat-file", "-t", analysis_ref) != "tag":
        raise RuntimeError("analysis seal tag must be annotated")
    analysis_commit = _git(root, "rev-parse", f"{analysis_ref}^{{commit}}")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", analysis_commit, ledger_commit),
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    ).returncode != 0:
        raise RuntimeError("train ledger tag does not descend from the analysis seal")
    ledger_changed = _git(
        root, "diff-tree", "--no-commit-id", "--name-only", "-r", ledger_commit
    ).splitlines()
    if ledger_changed != [relative_ledger]:
        raise RuntimeError("train ledger tag commit may change only its ledger")
    if _git(root, "rev-parse", f"{ledger_commit}:{relative_ledger}") != _git(
        root, "hash-object", "--", relative_ledger
    ):
        raise RuntimeError("train ledger bytes differ from its annotated tag")

    frozen_tag = plan["preregistration"]["frozen_model_tag"]
    frozen_ref = f"refs/tags/{frozen_tag}"
    if _git(root, "cat-file", "-t", frozen_ref) != "tag":
        raise RuntimeError("frozen model tag must be annotated")
    frozen_commit = _git(root, "rev-parse", f"{frozen_ref}^{{commit}}")
    frozen_line = _git(root, "rev-list", "--parents", "-n", "1", frozen_commit).split()
    if len(frozen_line) != 2 or frozen_line[1] != ledger_commit:
        raise RuntimeError("frozen model commit must be the sole child of train ledger")
    changed = _git(
        root, "diff-tree", "--no-commit-id", "--name-only", "-r", frozen_commit
    ).splitlines()
    if sorted(changed) != sorted((relative_model, relative_report)):
        raise RuntimeError(
            "frozen model tag commit may change only the model and training report"
        )
    for relative in (relative_model, relative_report):
        if _git(root, "rev-parse", f"{frozen_commit}:{relative}") != _git(
            root, "hash-object", "--", relative
        ):
            raise RuntimeError(f"frozen tagged bytes changed: {relative}")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", ledger_commit, frozen_commit),
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    ).returncode != 0:
        raise RuntimeError("frozen model tag does not descend from the train ledger tag")
    current = _git(root, "rev-parse", "HEAD")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", frozen_commit, current),
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    ).returncode != 0:
        raise RuntimeError("current HEAD does not descend from the frozen model tag")
    dirty = _git(
        root,
        "status",
        "--porcelain",
        "--",
        relative_ledger,
        relative_model,
        relative_report,
    )
    if dirty:
        raise RuntimeError("tagged train ledger/model/report files are dirty")
    return {
        "input_ledger_tag": ledger_tag,
        "input_ledger_commit": ledger_commit,
        "input_ledger_sha256": ledger_sha,
        "frozen_model_tag": frozen_tag,
        "frozen_model_commit": frozen_commit,
    }


@dataclass(frozen=True)
class PublicRouteGateDecision:
    """Frozen gate output; ``False`` is always the fixed-route fallback."""

    probabilities: torch.Tensor
    continue_search: torch.Tensor
    route_v6: torch.Tensor


@dataclass(frozen=True)
class PublicRouteGateRuntime:
    model: Mapping[str, Any]
    training_report: Mapping[str, Any]
    model_path: Path
    training_report_path: Path
    model_sha256: str
    model_semantic_sha256: str
    training_report_sha256: str
    training_report_semantic_sha256: str
    analysis_plan_sha256: str
    frozen_receipt: Mapping[str, str]
    frozen_receipt_semantic_sha256: str

    def verify_unchanged(self) -> None:
        model_raw = _snapshot_regular_bytes(self.model_path, "frozen gate model")
        if hashlib.sha256(model_raw).hexdigest() != self.model_sha256:
            raise RuntimeError("frozen public-route gate changed after loading")
        report_raw = _snapshot_regular_bytes(
            self.training_report_path, "gate training report"
        )
        if hashlib.sha256(report_raw).hexdigest() != self.training_report_sha256:
            raise RuntimeError("public-route gate training report changed after loading")
        try:
            model_semantic = semantic_model_sha256(self.model)
            report_semantic = _semantic_json_sha256(self.training_report)
            receipt_semantic = _semantic_json_sha256(self.frozen_receipt)
        except (TypeError, ValueError) as error:
            raise RuntimeError("frozen public-route runtime state was mutated") from error
        if model_semantic != self.model_semantic_sha256:
            raise RuntimeError("frozen public-route model state was mutated in memory")
        if report_semantic != self.training_report_semantic_sha256:
            raise RuntimeError("gate training report was mutated in memory")
        if receipt_semantic != self.frozen_receipt_semantic_sha256:
            raise RuntimeError("frozen Git receipt was mutated in memory")

    def decide_development(self, feature: torch.Tensor) -> PublicRouteGateDecision:
        """Evaluate only registered development features, checking SHA twice."""

        self.verify_unchanged()
        probabilities = ensemble_probabilities(self.model, feature)
        continue_search = decision_from_probabilities(probabilities)
        self.verify_unchanged()
        return PublicRouteGateDecision(
            probabilities=probabilities,
            continue_search=continue_search,
            route_v6=~continue_search,
        )

    def validate_cohort(self, cohort: str) -> None:
        if cohort != EXPECTED_COHORT:
            raise ValueError(
                "this frozen runtime is development-only; blind remains closed"
            )

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "kind": RUNTIME_KIND,
            "allowed_cohort": EXPECTED_COHORT,
            "default_on_error": "fixed_route_v6",
            "blind_collection_opened": False,
            "model_sha256": self.model_sha256,
            "model_semantic_sha256": self.model_semantic_sha256,
            "training_report_sha256": self.training_report_sha256,
            "analysis_plan_sha256": self.analysis_plan_sha256,
            **dict(self.frozen_receipt),
        }


def load_frozen_public_route_gate(
    model_path: Path,
    training_report_path: Path,
    *,
    analysis_plan_path: Path = DEFAULT_ANALYSIS_PLAN,
    repository_root: Path | None = None,
    cohort: str = EXPECTED_COHORT,
    require_preregistered_plan: bool = True,
    enforce_canonical_paths: bool = True,
    enforce_frozen_model_tag: bool = True,
) -> PublicRouteGateRuntime:
    """Load the one train-only model for a development-only audit.

    The two relaxed keyword arguments exist for simulation-free lifecycle
    tests.  Production callers use the defaults and therefore cannot redirect
    the model or run against a draft plan.
    """

    if cohort != EXPECTED_COHORT:
        raise ValueError("public-route gate runtime may load only for development")
    plan = load_analysis_plan(
        analysis_plan_path, require_preregistered=require_preregistered_plan
    )
    root = (
        _absolute(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    model_path = _absolute(model_path)
    training_report_path = _absolute(training_report_path)
    if enforce_canonical_paths:
        expected_model = _canonical_output_path(
            root, plan["outputs"]["frozen_model"], "frozen model output"
        )
        expected_report = _canonical_output_path(
            root, plan["outputs"]["training_report"], "training report output"
        )
        if model_path != expected_model or training_report_path != expected_report:
            raise ValueError("gate runtime paths are not the registered canonical outputs")
        _require_nonsymlink_chain(model_path, root, "frozen gate model")
        _require_nonsymlink_chain(
            training_report_path, root, "gate training report"
        )

    model_path = _require_regular_file(model_path, "frozen gate model")
    training_report_path = _require_regular_file(
        training_report_path, "gate training report"
    )
    model_raw = _snapshot_regular_bytes(model_path, "frozen gate model")
    model_sha = hashlib.sha256(model_raw).hexdigest()
    payload = torch.load(io.BytesIO(model_raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("frozen gate model root must be a mapping")
    model = validate_gate_model(payload)
    _validate_model_plan_binding(model, plan)

    report, report_sha = _read_json_unchanged(
        training_report_path, "gate training report"
    )
    report = _validate_training_report(
        report, model=model, model_sha256=model_sha, plan=plan
    )
    if enforce_frozen_model_tag:
        live_provenance = analysis_seal_provenance(plan, repository_root=root)
        _validate_live_analysis_binding(report, live_provenance)
    frozen_receipt: Mapping[str, str] = {
        "input_ledger_tag": model["metadata"]["input_ledger_tag"],
        "input_ledger_sha256": model["metadata"]["input_ledger_sha256"],
        "frozen_model_tag": plan["preregistration"]["frozen_model_tag"],
    }
    if enforce_frozen_model_tag:
        frozen_receipt = _require_tagged_frozen_inputs(
            model=model,
            report=report,
            plan=plan,
            repository_root=root,
            model_path=model_path,
            report_path=training_report_path,
        )
    runtime = PublicRouteGateRuntime(
        model=model,
        training_report=report,
        model_path=model_path,
        training_report_path=training_report_path,
        model_sha256=model_sha,
        model_semantic_sha256=semantic_model_sha256(model),
        training_report_sha256=report_sha,
        training_report_semantic_sha256=_semantic_json_sha256(report),
        analysis_plan_sha256=plan["_sha256"],
        frozen_receipt=frozen_receipt,
        frozen_receipt_semantic_sha256=_semantic_json_sha256(frozen_receipt),
    )
    runtime.verify_unchanged()
    return runtime
