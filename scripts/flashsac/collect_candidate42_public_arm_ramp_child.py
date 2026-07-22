#!/usr/bin/env python3
"""Attempt-local Candidate42 rollout worker and sole application owner."""

from __future__ import annotations

import argparse
import fcntl
import io
import os
from pathlib import Path
import traceback
from typing import Any, Mapping

import torch

import candidate42_attempt_transaction as transaction
import candidate42_public_arm_ramp_collection as collection
import candidate42_public_arm_ramp_episode as artifact_contract


def _bootstrap_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output_stem", type=Path, required=True)
    parser.add_argument("--_candidate42_run_id", required=True)
    parser.add_argument("--_candidate42_attempt_id", required=True)
    parser.add_argument("--_candidate42_attempt_number", type=int, required=True)
    parser.add_argument("--_candidate42_lock_fd", type=int, required=True)
    return parser.parse_known_args(argv)[0]


def _identity(args: argparse.Namespace) -> transaction.AttemptIdentity:
    return transaction.AttemptIdentity(
        run_id=args._candidate42_run_id,
        attempt_id=args._candidate42_attempt_id,
        attempt_number=args._candidate42_attempt_number,
    )


def _full_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    # Import occurs only after the durable parent intent and inherited lock
    # have been authenticated by main().  This process is the sole owner.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    collection.add_scientific_arguments(parser)
    parser.add_argument("--_candidate42_run_id", required=True)
    parser.add_argument("--_candidate42_attempt_id", required=True)
    parser.add_argument("--_candidate42_attempt_number", type=int, required=True)
    parser.add_argument("--_candidate42_lock_fd", type=int, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args._candidate42_launcher_type = AppLauncher
    return args


def _receipt_payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    payload = getattr(value, "payload", None)
    if isinstance(payload, Mapping):
        return payload
    raise TypeError("transaction receipt did not expose a strict mapping payload")


def _evidence(stage: transaction.StageOnly, name: str) -> transaction.FileEvidence:
    return transaction.file_evidence(stage, name)


def _validate_intent(
    intent: Mapping[str, Any], identity: transaction.AttemptIdentity
) -> None:
    expected = {
        "run_id": identity.run_id,
        "attempt_id": identity.attempt_id,
        "attempt_number": identity.attempt_number,
    }
    if intent.get("identity") != expected:
        raise ValueError("sealed parent intent identity changed")
    if intent.get("status") != "prepared" or intent.get("receipt") != "00_intent.json":
        raise ValueError("sealed parent intent state changed")
    receipt = intent.get("argument_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("sealed parent intent omitted its argument receipt")


def _serialize_artifact(artifact: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(artifact), stream)
    return stream.getvalue()


def _freeze_child_authority(
    args: argparse.Namespace, intent: Mapping[str, Any]
) -> tuple[collection.CollectionSpec, str, str]:
    """Freeze every value that the launcher is permitted to consume."""

    device_string = str(args.device or "cuda:0")
    spec = collection.build_spec(args)
    runtime_asset_manifest = artifact_contract.manifest_sha256(
        collection.runtime_asset_fingerprints(spec.repository_root)
    )
    if dict(spec.argument_receipt) != dict(intent["argument_receipt"]):
        raise ValueError("child argument receipt differs from sealed intent")
    for name, actual in (
        ("assignment_mask_sha256", spec.assignment_mask_sha256),
        (
            "source_manifest_sha256",
            artifact_contract.manifest_sha256(spec.source_sha256),
        ),
        (
            "checkpoint_manifest_sha256",
            artifact_contract.canonical_json_sha256(spec.checkpoint_sha256),
        ),
        ("runtime_asset_manifest_sha256", runtime_asset_manifest),
        ("collection_commit", spec.collection_commit),
        ("preregistration_tag_commit", spec.preregistration_tag_commit),
    ):
        if intent.get(name) != actual:
            raise ValueError(f"child authority differs from intent at {name}")
    return spec, device_string, runtime_asset_manifest


def _close_with_marker(
    *,
    app: Any,
    stage: transaction.StageOnly,
    identity: transaction.AttemptIdentity,
    marker_name: str,
    predecessor: transaction.FileEvidence,
    reason: str,
) -> None:
    transaction.write_receipt(
        stage,
        marker_name,
        identity,
        {"status": "close_requested", "reason": reason},
        predecessors=(predecessor,),
    )
    app.close()


def run_child(argv: list[str] | None = None) -> None:
    bootstrap = _bootstrap_arguments(argv)
    paths = transaction.TransactionPaths.from_output_stem(bootstrap.output_stem)
    identity = _identity(bootstrap)
    transaction.validate_inherited_lock(
        bootstrap._candidate42_lock_fd, paths.lock_path
    )
    flags = fcntl.fcntl(bootstrap._candidate42_lock_fd, fcntl.F_GETFD)
    fcntl.fcntl(
        bootstrap._candidate42_lock_fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC
    )
    stage = transaction.StageOnly.open(paths, identity.attempt_number)
    intent_evidence = transaction.load_receipt(stage, "00_intent.json")
    intent = _receipt_payload(intent_evidence)
    _validate_intent(intent, identity)
    started = transaction.write_receipt(
        stage,
        "10_child_started.json",
        identity,
        {"status": "started", "pid": os.getpid()},
        predecessors=(_evidence(stage, "00_intent.json"),),
    )

    app = None
    progress = collection.AttemptProgress()
    prepared: transaction.FileEvidence | None = None
    app_started: transaction.FileEvidence | None = None
    try:
        args = _full_arguments(argv)
        if _identity(args) != identity:
            raise ValueError("full child arguments changed transaction identity")
        launcher_type = args._candidate42_launcher_type
        delattr(args, "_candidate42_launcher_type")
        # The launcher consumes several Namespace entries in-place.  Freeze
        # and authenticate the complete user-visible receipt first.
        spec, device_string, runtime_asset_manifest = _freeze_child_authority(
            args, intent
        )

        launcher = launcher_type(args)
        app = launcher.app
        config = getattr(app, "config", None)
        if not isinstance(config, Mapping) or config.get("fast_shutdown") is not True:
            raise RuntimeError("effective fast_shutdown must be exactly true")
        app_started = transaction.write_receipt(
            stage,
            "15_app_started.json",
            identity,
            {"status": "started", "fast_shutdown": True},
            predecessors=(started,),
        )

        boundary: transaction.FileEvidence | None = None

        def arm_boundary() -> None:
            nonlocal boundary
            boundary = transaction.durably_arm_first_step(
                stage,
                identity,
                progress,
                fields={"status": "armed"},
            )
            if boundary is None:
                boundary = _evidence(stage, "20_first_step_armed.json")

        artifact, report_core = collection.run_collection(
            spec,
            device_string=device_string,
            durable_boundary=arm_boundary,
            progress=progress,
        )
        if boundary is None or not progress.first_env_step_invoked:
            raise RuntimeError("rollout returned without an armed first-step boundary")
        artifact = artifact_contract.validate_artifact(artifact)
        report_core = artifact_contract.validate_report(report_core, artifact)
        collection.validate_post_application_authority(spec)

        artifact_bytes = _serialize_artifact(artifact)
        staged_roundtrip = torch.load(
            io.BytesIO(artifact_bytes), map_location="cpu", weights_only=True
        )
        artifact_contract.validate_artifact(staged_roundtrip)
        payload = transaction.durable_stage_bytes(
            stage, "30_payload.pt", artifact_bytes
        )
        report = transaction.durable_stage_bytes(
            stage,
            "31_report_core.json",
            artifact_contract.report_core_bytes(report_core, artifact),
        )
        prepared = transaction.write_receipt(
            stage,
            "32_prepared.json",
            identity,
            {
                "status": "prepared",
                "assignment_mask_sha256": spec.assignment_mask_sha256,
                "source_manifest_sha256": artifact_contract.manifest_sha256(
                    spec.source_sha256
                ),
                "checkpoint_manifest_sha256": artifact_contract.canonical_json_sha256(
                    spec.checkpoint_sha256
                ),
                "runtime_asset_manifest_sha256": runtime_asset_manifest,
            },
            predecessors=(boundary, payload, report),
        )
        _close_with_marker(
            app=app,
            stage=stage,
            identity=identity,
            marker_name="40_close_requested.json",
            predecessor=prepared,
            reason="success",
        )
    except BaseException as error:
        if (
            stage.path("20_first_step_armed.json").exists()
            or stage.path("20_first_step_armed.json").is_symlink()
        ):
            # Durable marker existence is the conservative authority even if
            # a later validation exception prevented the callback returning.
            progress.first_env_step_invoked = True
        if prepared is None and (
            stage.path("32_prepared.json").exists()
            or stage.path("32_prepared.json").is_symlink()
        ):
            prepared = _evidence(stage, "32_prepared.json")
        if app_started is None and (
            stage.path("15_app_started.json").exists()
            or stage.path("15_app_started.json").is_symlink()
        ):
            app_started = _evidence(stage, "15_app_started.json")
        fields = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if not progress.first_env_step_invoked:
            predecessor = app_started if app_started is not None else started
            failed = transaction.write_receipt(
                stage,
                "18_child_preboundary_failed.json",
                identity,
                fields,
                predecessors=(predecessor,),
            )
            if app is not None:
                _close_with_marker(
                    app=app,
                    stage=stage,
                    identity=identity,
                    marker_name="19_close_requested_preboundary.json",
                    predecessor=failed,
                    reason="preboundary_failure",
                )
        elif prepared is None:
            failed = transaction.write_receipt(
                stage,
                "45_child_failed.json",
                identity,
                fields,
                predecessors=(_evidence(stage, "20_first_step_armed.json"),),
            )
            if app is not None:
                _close_with_marker(
                    app=app,
                    stage=stage,
                    identity=identity,
                    marker_name="46_close_requested_failure.json",
                    predecessor=failed,
                    reason="postboundary_failure",
                )
        else:
            transaction.write_receipt(
                stage,
                "46_close_requested_failure.json",
                identity,
                fields | {"reason": "failure_after_prepared"},
                predecessors=(prepared,),
            )
            if app is not None:
                app.close()
        raise


def main() -> None:
    run_child()


if __name__ == "__main__":
    main()
