#!/usr/bin/env python3
"""CPU-only tests for Candidate42's attempt-local rollout worker."""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import collect_candidate42_public_arm_ramp_child as child


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_application_owner_import_is_unique_and_lazy() -> None:
    path = _root() / "scripts/flashsac/collect_candidate42_public_arm_ramp_child.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owners: list[ast.ImportFrom] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "isaaclab.app":
            owners.append(node)
    assert len(owners) == 1
    owner = owners[0]
    assert [alias.name for alias in owner.names] == ["AppLauncher"]
    full = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_full_arguments"
    )
    assert owner in list(ast.walk(full))


def test_child_has_no_parent_publisher_or_canonical_write_capability() -> None:
    path = _root() / "scripts/flashsac/collect_candidate42_public_arm_ramp_child.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "collect_candidate42_public_arm_ramp_ab" not in imported
    for forbidden in (
        "canonical_artifact_output",
        "canonical_report_output",
        "canonical_commit_output",
        "publish_verified_attempt",
        "recover_verified_publication",
    ):
        assert forbidden not in source

    stage_writers = {
        "write_receipt",
        "durable_stage_bytes",
        "durably_arm_first_step",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            raise AssertionError("child performs a direct filesystem open")
        if not isinstance(node.func, ast.Attribute):
            continue
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "open"
        ):
            raise AssertionError("child performs a direct os.open")
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "transaction"
            and node.func.attr in stage_writers
        ):
            assert node.args
            assert isinstance(node.args[0], ast.Name)
            assert node.args[0].id == "stage"
        if node.func.attr in {
            "write_bytes",
            "write_text",
            "touch",
            "replace",
            "rename",
            "unlink",
        }:
            raise AssertionError(f"child uses direct path writer {node.func.attr}")
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "save"
        ):
            assert len(node.args) >= 2
            assert isinstance(node.args[1], ast.Name)
            assert node.args[1].id == "stream"


def test_authority_is_frozen_before_mutating_launcher() -> None:
    args = argparse.Namespace(device="cuda:3")
    spec = SimpleNamespace(
        argument_receipt={"device": "cuda:3"},
        assignment_mask_sha256="a" * 64,
        source_sha256={"source.py": "b" * 64},
        checkpoint_sha256={"checkpoint": "c" * 64},
        collection_commit="d" * 40,
        preregistration_tag_commit="f" * 40,
        repository_root=Path("/tmp"),
    )
    runtime_assets = {"asset.usd": "e" * 64}
    intent = {
        "argument_receipt": {"device": "cuda:3"},
        "assignment_mask_sha256": spec.assignment_mask_sha256,
        "source_manifest_sha256": child.artifact_contract.manifest_sha256(
            spec.source_sha256
        ),
        "checkpoint_manifest_sha256": child.artifact_contract.canonical_json_sha256(
            spec.checkpoint_sha256
        ),
        "runtime_asset_manifest_sha256": child.artifact_contract.manifest_sha256(
            runtime_assets
        ),
        "collection_commit": spec.collection_commit,
        "preregistration_tag_commit": spec.preregistration_tag_commit,
    }
    with mock.patch.object(
        child.collection, "build_spec", return_value=spec
    ), mock.patch.object(
        child.collection,
        "runtime_asset_fingerprints",
        return_value=runtime_assets,
    ):
        frozen, device, runtime_manifest = child._freeze_child_authority(
            args, intent
        )

    class MutatingLauncher:
        def __init__(self, namespace: argparse.Namespace) -> None:
            namespace.__dict__.pop("device")

    MutatingLauncher(args)
    assert not hasattr(args, "device")
    assert frozen is spec
    assert device == "cuda:3"
    assert runtime_manifest == intent["runtime_asset_manifest_sha256"]


def test_bootstrap_rejects_missing_parent_capabilities() -> None:
    with redirect_stderr(io.StringIO()):
        try:
            child._bootstrap_arguments(["--output_stem", "/tmp/trial"])
        except SystemExit as error:
            assert error.code != 0
        else:
            raise AssertionError("direct child invocation unexpectedly parsed")


def main() -> None:
    test_application_owner_import_is_unique_and_lazy()
    test_child_has_no_parent_publisher_or_canonical_write_capability()
    test_authority_is_frozen_before_mutating_launcher()
    test_bootstrap_rejects_missing_parent_capabilities()
    print("collect_candidate42_public_arm_ramp_child tests passed")


if __name__ == "__main__":
    main()
