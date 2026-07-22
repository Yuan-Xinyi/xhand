#!/usr/bin/env python3
"""CPU fault/corruption tests for Candidate42's evidence transaction."""

from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest

import candidate42_attempt_transaction as txn


def _run_inputs(paths: txn.TransactionPaths, *, argv: list[str] | None = None) -> dict:
    return {
        "collection_commit": "a" * 40,
        "parent_argv": ["--seed", "346"] if argv is None else argv,
        "assignment_mask_sha256": "b" * 64,
        "checkpoint_manifest_sha256": "c" * 64,
        "source_manifest_sha256": "d" * 64,
        "runtime_asset_manifest_sha256": "e" * 64,
        "canonical_paths": paths.as_run_id_dict(),
        "seed": 346,
        "replicate": "b",
        "num_envs": 8,
    }


def _run_metadata(paths: txn.TransactionPaths) -> dict:
    return {
        "seed": 346,
        "replicate": "b",
        "num_envs": 8,
        "collection_commit": "a" * 40,
        "preregistration_tag_commit": "f" * 40,
        "assignment_mask_sha256": "b" * 64,
        "source_manifest_sha256": "d" * 64,
        "checkpoint_manifest_sha256": "c" * 64,
        "runtime_asset_manifest_sha256": "e" * 64,
        "canonical_artifact_output": str(paths.artifact),
        "canonical_report_output": str(paths.report),
        "canonical_commit_output": str(paths.commit),
        "argument_receipt": {"seed": 346, "replicate": "b", "num_envs": 8},
    }


def _new_attempt(
    root: Path, *, attempt_number: int = 1, paths: txn.TransactionPaths | None = None
) -> tuple[txn.TransactionPaths, txn.NamespaceLock, txn.StageOnly, txn.AttemptIdentity]:
    paths = txn.TransactionPaths.from_output_stem(root / "trial") if paths is None else paths
    lock = txn.NamespaceLock.acquire(paths)
    run_inputs = _run_inputs(paths)
    run_id = txn.compute_run_id(run_inputs)
    identity = txn.AttemptIdentity(
        run_id=run_id,
        attempt_id=txn.compute_attempt_id(run_id, attempt_number),
        attempt_number=attempt_number,
    )
    stage = txn.StageOnly.create(paths, attempt_number)
    metadata = _run_metadata(paths)
    txn.write_receipt(
        stage,
        txn.INTENT,
        identity,
        {"status": "prepared", "run_metadata": metadata, **metadata},
    )
    return paths, lock, stage, identity


def _through_boundary(
    stage: txn.StageOnly, identity: txn.AttemptIdentity
) -> txn.FileEvidence:
    intent = txn.load_receipt(stage, txn.INTENT)
    started = txn.write_receipt(
        stage,
        txn.CHILD_STARTED,
        identity,
        {"status": "started", "pid": os.getpid()},
        predecessors=(intent,),
    )
    txn.write_receipt(
        stage,
        txn.APP_STARTED,
        identity,
        {"status": "started", "fast_shutdown": True},
        predecessors=(started,),
    )
    progress = SimpleNamespace(first_env_step_invoked=False)
    boundary = txn.durably_arm_first_step(
        stage, identity, progress, fields={"status": "armed"}
    )
    # The collection wrapper owns the in-memory flag immediately after the
    # durable callback returns and before env.step.
    progress.first_env_step_invoked = True
    assert txn.durably_arm_first_step(
        stage, identity, progress, fields={"status": "armed"}
    ) == txn.file_evidence(stage, txn.FIRST_STEP_ARMED)
    return boundary


def _write_preboundary_exit(
    stage: txn.StageOnly,
    identity: txn.AttemptIdentity,
    tip: txn.FileEvidence,
    *,
    raw_wait_status: int | None = 256,
) -> txn.FileEvidence:
    stdout = txn.durable_stage_bytes(stage, txn.STDOUT_LOG, b"")
    stderr = txn.durable_stage_bytes(stage, txn.STDERR_LOG, b"")
    return txn.write_receipt(
        stage,
        txn.CHILD_EXIT,
        identity,
        {
            "status": "observed",
            **txn.decode_wait_status(raw_wait_status),
            "stdout": stdout.as_dict(),
            "stderr": stderr.as_dict(),
        },
        predecessors=(tip,),
    )


def _complete_verified_attempt(
    stage: txn.StageOnly, identity: txn.AttemptIdentity
) -> None:
    paths = stage.paths
    boundary = _through_boundary(stage, identity)
    payload = txn.durable_stage_bytes(stage, txn.STAGED_PAYLOAD, b"weights-only payload")
    core_object = {"kind": "synthetic_report_core", "status": "complete"}
    core_bytes = txn.pretty_json_bytes(core_object)
    report_core = txn.durable_stage_bytes(stage, txn.STAGED_REPORT_CORE, core_bytes)
    prepared = txn.write_receipt(
        stage,
        txn.PREPARED,
        identity,
        {
            "status": "prepared",
            "assignment_mask_sha256": "b" * 64,
            "source_manifest_sha256": "d" * 64,
            "checkpoint_manifest_sha256": "c" * 64,
        },
        predecessors=(boundary, payload, report_core),
    )
    close = txn.write_receipt(
        stage,
        txn.CLOSE_SUCCESS,
        identity,
        {"status": "close_requested", "reason": "success"},
        predecessors=(prepared,),
    )
    stdout = txn.durable_stage_bytes(stage, txn.STDOUT_LOG, b"child stdout\n")
    stderr = txn.durable_stage_bytes(stage, txn.STDERR_LOG, b"")
    child_exit = txn.write_receipt(
        stage,
        txn.CHILD_EXIT,
        identity,
        {
            "status": "observed",
            "raw_wait_status": 0,
            "wifexited": True,
            "exit_code": 0,
            "signal": None,
            "stdout": stdout.as_dict(),
            "stderr": stderr.as_dict(),
        },
        predecessors=(close,),
    )
    final = {
        "transaction_contract": txn.TRANSACTION_CONTRACT,
        "receipt": txn.FINAL_REPORT,
        "identity": identity.as_dict(),
        "predecessors": {txn.CHILD_EXIT: child_exit.as_dict()},
        "kind": txn.FINAL_REPORT_KIND,
        "version": 1,
        "status": "complete",
        **identity.as_dict(),
        "child_exit_sha256": child_exit.sha256,
        "post_exit_authority": {"status": "passed"},
        "source_manifest_sha256": "d" * 64,
        "checkpoint_manifest_sha256": "c" * 64,
        "runtime_asset_manifest_sha256": "e" * 64,
        "artifact_sha256": payload.sha256,
        "artifact_size": payload.size,
        "artifact_output": str(paths.artifact),
        "report_core": core_object,
        "report_core_sha256": report_core.sha256,
    }
    final_evidence = txn.durable_stage_bytes(
        stage, txn.FINAL_REPORT, txn.pretty_json_bytes(final)
    )
    txn.write_receipt(
        stage,
        txn.VERIFIED,
        identity,
        {"status": "verified"},
        predecessors=(final_evidence,),
    )


def _verified_namespace(
    root: Path,
) -> tuple[txn.TransactionPaths, txn.NamespaceLock, txn.StageOnly, txn.AttemptIdentity]:
    paths, lock, stage, identity = _new_attempt(root)
    _complete_verified_attempt(stage, identity)
    return paths, lock, stage, identity


def _successful_namespace(root: Path) -> tuple[txn.TransactionPaths, txn.AttemptIdentity]:
    """Shared CPU fixture used by analyzer corruption tests."""

    paths, lock, _stage, identity = _verified_namespace(root)
    try:
        result = txn.publish_verified_attempt(paths, identity, lock)
        assert result["status"] == "committed"
    finally:
        lock.close()
    return paths, identity


def test_run_and_attempt_ids_are_stable_and_unambiguous() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths = txn.TransactionPaths.from_output_stem(Path(name) / "trial")
        inputs = _run_inputs(paths)
        first = txn.compute_run_id(inputs)
        assert first == txn.compute_run_id(copy.deepcopy(inputs))
        changed = copy.deepcopy(inputs)
        changed["parent_argv"].append("--headless")
        assert txn.compute_run_id(changed) != first
        assert txn.compute_attempt_id(first, 1) != txn.compute_attempt_id(first, 2)
        with pytest.raises((ValueError, TypeError)):
            txn.compute_run_id(inputs | {"extra": True})


def test_namespace_lock_inheritance_and_cloexec() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths = txn.TransactionPaths.from_output_stem(Path(name) / "trial")
        lock = txn.NamespaceLock.acquire(paths)
        duplicate = lock.duplicate_for_child()
        duplicate_open = True
        try:
            assert os.get_inheritable(duplicate)
            fake = os.open(paths.lock_path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                with pytest.raises(txn.TransactionError):
                    txn.validate_inherited_lock(fake, paths.lock_path)
            finally:
                os.close(fake)
            txn.validate_inherited_lock(duplicate, paths.lock_path)
            assert fcntl.fcntl(duplicate, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
            with pytest.raises(txn.TransactionError):
                txn.NamespaceLock.acquire(paths, blocking=False)
            lock.close()
            # Closing the parent descriptor must not explicitly unlock the
            # shared open-file description while a child duplicate is live.
            with pytest.raises(txn.TransactionError):
                txn.NamespaceLock.acquire(paths, blocking=False)
            os.close(duplicate)
            duplicate_open = False
            replacement = txn.NamespaceLock.acquire(paths, blocking=False)
            replacement.close()
        finally:
            if duplicate_open:
                os.close(duplicate)
            lock.close()


def test_stage_capability_rejects_escape_symlink_and_clobber() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, _identity = _new_attempt(Path(name))
        try:
            with pytest.raises(txn.PathSafetyError):
                stage.path("../trial.pt")
            target = stage.path(txn.STAGED_PAYLOAD)
            target.symlink_to(paths.artifact)
            with pytest.raises(FileExistsError):
                txn.durable_stage_bytes(stage, txn.STAGED_PAYLOAD, b"x")
            target.unlink()
            evidence = txn.durable_stage_bytes(stage, txn.STAGED_PAYLOAD, b"one")
            assert evidence.sha256 == txn.sha256_file(evidence.path)
            with pytest.raises(FileExistsError):
                txn.durable_stage_bytes(stage, txn.STAGED_PAYLOAD, b"two")
        finally:
            lock.close()


def test_complete_commit_and_callbacks_validate_exact_hardlinks() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, _identity = _successful_namespace(Path(name))
        calls: list[str] = []
        result = txn.validate_committed_namespace(
            paths,
            artifact_validator=lambda path: calls.append(f"artifact:{path.name}"),
            report_validator=lambda report, artifact: calls.append(
                f"report:{report.name}:{artifact.name}"
            ),
        )
        assert result["status"] == "committed"
        assert result["retryable_prior_attempts"] == []
        assert calls == ["artifact:trial.pt", "report:trial.json:trial.pt"]
        context = txn.open_attempt(paths, 1)
        assert context.run_metadata["seed"] == 346
        assert context.run_identity["parent_argv"] == ["--seed", "346"]
        assert os.path.samefile(context.path(txn.STAGED_PAYLOAD), paths.artifact)
        assert os.path.samefile(context.path(txn.FINAL_REPORT), paths.report)
        assert os.path.samefile(context.path(txn.COMMITTED), paths.commit)


@pytest.mark.parametrize("partial", ("artifact", "pair", "missing_commit"))
def test_partial_publication_recovers_only_from_same_verified_attempt(partial: str) -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _verified_namespace(Path(name))
        try:
            if partial in {"artifact", "pair"}:
                os.link(stage.path(txn.STAGED_PAYLOAD), paths.artifact)
            if partial == "pair":
                os.link(stage.path(txn.FINAL_REPORT), paths.report)
            if partial == "missing_commit":
                txn.publish_verified_attempt(paths, identity, lock)
                paths.commit.unlink()
            result = txn.recover_verified_publication(paths, identity, lock)
            assert result["status"] == "committed"
        finally:
            lock.close()


def test_orphan_60_and_success_failure_conflict_reject() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, identity = _successful_namespace(Path(name))
        # 60 says both canonical pair links were durable.  Losing one makes it
        # an orphan/conflict and recovery must not synthesize history.
        paths.artifact.unlink()
        lock = txn.NamespaceLock.acquire(paths)
        try:
            with pytest.raises(txn.TransactionError):
                txn.recover_verified_publication(paths, identity, lock)
            failure = txn.record_parent_failure(
                paths,
                identity,
                lock,
                reason="orphan 60 after canonical artifact loss",
                raw_wait_status=None,
                spawn_succeeded=True,
                child_exit_observed=True,
                recovery_lock_acquired=True,
            )
            assert failure.payload is not None
            assert failure.payload["boundary_armed"] is True
            assert "report" in failure.payload["canonical_observed"]
            assert "commit" in failure.payload["canonical_observed"]
            with pytest.raises(txn.TransactionError):
                txn.validate_committed_namespace(paths)
        finally:
            lock.close()


def test_preboundary_failure_repair_and_next_attempt_sequence() -> None:
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        paths, lock, stage, identity = _new_attempt(root)
        try:
            intent = txn.load_receipt(stage, txn.INTENT)
            failed = txn.write_receipt(
                stage,
                txn.PARENT_PRESPAWN_FAILED,
                identity,
                {"status": "failed", "error_type": "OSError", "error": "spawn"},
                predecessors=(intent,),
            )
            assert failed.payload is not None
            receipt = txn.record_parent_failure(
                paths,
                identity,
                lock,
                reason="caught pre-spawn failure",
                raw_wait_status=None,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            assert receipt.payload is not None
            assert receipt.payload["retry_permitted"] is True
            canonical_failure = Path(f"{paths.output_stem}.failed_attempt_001.json")
            canonical_failure.unlink()
            txn.repair_failure_link(paths, identity, lock)
            second = txn.StageOnly.create(paths, 2)
            second_identity = txn.AttemptIdentity(
                identity.run_id, txn.compute_attempt_id(identity.run_id, 2), 2
            )
            metadata = _run_metadata(paths)
            # Re-cache exact retry-invariant identity, as a recovery process
            # would do after validating the prior 00.
            txn.compute_run_id(_run_inputs(paths))
            txn.write_receipt(
                second,
                txn.INTENT,
                second_identity,
                {"status": "prepared", "run_metadata": metadata, **metadata},
            )
            sequence = txn.validate_attempt_sequence(paths.output_stem)
            assert sequence.final_state == "unclassified"
            assert len(sequence.attempts) == 2
        finally:
            lock.close()


def test_abrupt_preboundary_requires_exit_or_lock_release_proof() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            intent = txn.load_receipt(stage, txn.INTENT)
            started = txn.write_receipt(
                stage,
                txn.CHILD_STARTED,
                identity,
                {"status": "started", "pid": os.getpid()},
                predecessors=(intent,),
            )
            unproven = txn.classify_attempt(
                paths,
                identity,
                spawn_succeeded=True,
                child_exit_observed=False,
            )
            assert not unproven.boundary_armed and not unproven.retry_permitted
            proven = txn.classify_attempt(
                paths,
                identity,
                spawn_succeeded=True,
                child_exit_observed=True,
            )
            assert not proven.boundary_armed and not proven.retry_permitted
            stdout = txn.durable_stage_bytes(stage, txn.STDOUT_LOG, b"")
            stderr = txn.durable_stage_bytes(stage, txn.STDERR_LOG, b"")
            txn.write_receipt(
                stage,
                txn.CHILD_EXIT,
                identity,
                {
                    "status": "observed",
                    **txn.decode_wait_status(256),
                    "stdout": stdout.as_dict(),
                    "stderr": stderr.as_dict(),
                },
                predecessors=(started,),
            )
            strictly_proven = txn.classify_attempt(
                paths,
                identity,
                spawn_succeeded=True,
                child_exit_observed=True,
            )
            assert (
                not strictly_proven.boundary_armed
                and strictly_proven.retry_permitted
            )
        finally:
            lock.close()


def test_boundary_and_postboundary_temp_are_conservatively_nonretryable() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            _through_boundary(stage, identity)
            classification = txn.classify_attempt(
                paths, identity, spawn_succeeded=True, child_exit_observed=True
            )
            assert classification.boundary_armed and not classification.retry_permitted
        finally:
            lock.close()
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            orphan = stage.attempt_dir / ".30_payload.pt.tmp-123-deadbeef"
            orphan.write_bytes(b"crash window")
            classification = txn.classify_attempt(
                paths, identity, spawn_succeeded=True, child_exit_observed=True
            )
            assert classification.boundary_armed and not classification.retry_permitted
            failure = txn.record_parent_failure(
                paths,
                identity,
                lock,
                reason="orphan post-boundary durable temporary",
                raw_wait_status=None,
                spawn_succeeded=True,
                child_exit_observed=True,
            )
            assert orphan.name in failure.payload["observed"]
        finally:
            lock.close()


def test_receipt_corruption_gap_and_foreign_canonical_collision_reject() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _verified_namespace(Path(name))
        try:
            paths.artifact.write_bytes(b"foreign")
            with pytest.raises(txn.TransactionError):
                txn.publish_verified_attempt(paths, identity, lock)
        finally:
            lock.close()
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, _identity = _new_attempt(Path(name))
        try:
            (paths.namespace / "attempt_003").mkdir()
            with pytest.raises(txn.TransactionError):
                txn.validate_attempt_sequence(paths.output_stem)
            intent = stage.path(txn.INTENT)
            document = json.loads(intent.read_text())
            document["identity"]["run_id"] = "0" * 64
            intent.write_bytes(txn.pretty_json_bytes(document))
            with pytest.raises(txn.TransactionError):
                txn.load_receipt(stage, txn.INTENT)
        finally:
            lock.close()


def test_durable_link_fault_never_overwrites_or_erases_final_evidence() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, _identity = _new_attempt(Path(name))
        try:
            real_link = os.link

            def failed_link(source, target, **kwargs):
                raise OSError("injected link failure")

            with mock.patch("candidate42_attempt_transaction.os.link", failed_link):
                with pytest.raises(OSError):
                    txn.durable_stage_bytes(stage, txn.STAGED_PAYLOAD, b"payload")
            assert not stage.path(txn.STAGED_PAYLOAD).exists()
            assert not any(txn.TEMP_PATTERN.fullmatch(p.name) for p in stage.attempt_dir.iterdir())
            evidence = txn.durable_stage_bytes(stage, txn.STAGED_PAYLOAD, b"payload")
            assert evidence.path.read_bytes() == b"payload"
            assert real_link is os.link
        finally:
            lock.close()


def test_allocate_attempt_writes_the_current_flat_intent_schema() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths = txn.TransactionPaths.from_output_stem(Path(name) / "trial")
        with txn.NamespaceLock.acquire(paths) as lock:
            context = txn.allocate_attempt(
                lock,
                _run_inputs(paths),
                run_metadata=_run_metadata(paths),
            )
            intent = txn.load_receipt(
                txn.StageOnly.open(paths, 1), txn.INTENT
            ).payload
            assert intent is not None
            assert "payload" not in intent
            assert intent["run_identity"] == context.run_identity
            assert context.run_metadata["seed"] == 346


def test_preboundary_temp_is_nonretryable_and_unknown_temp_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            orphan = stage.attempt_dir / ".10_child_started.json.tmp-123-deadbeef"
            orphan.write_bytes(b"incomplete durable write")
            result = txn.classify_attempt(
                paths,
                identity,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            assert not result.boundary_armed and not result.retry_permitted
            failure = txn.record_parent_failure(
                paths,
                identity,
                lock,
                reason="preboundary durable temp",
                raw_wait_status=None,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            assert failure.payload is not None
            assert failure.payload["retry_permitted"] is False
        finally:
            lock.close()
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            (stage.attempt_dir / ".not_registered.tmp-123-deadbeef").write_bytes(b"x")
            with pytest.raises(txn.StateConflictError):
                txn.classify_attempt(
                    paths,
                    identity,
                    spawn_succeeded=False,
                    child_exit_observed=False,
                )
        finally:
            lock.close()


def test_05_conflicts_with_any_child_side_receipt() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            intent = txn.load_receipt(stage, txn.INTENT)
            txn.write_receipt(
                stage,
                txn.PARENT_PRESPAWN_FAILED,
                identity,
                {"status": "failed", "error_type": "OSError", "error": "spawn"},
                predecessors=(intent,),
            )
            started = txn.write_receipt(
                stage,
                txn.CHILD_STARTED,
                identity,
                {"status": "started", "pid": os.getpid()},
                predecessors=(intent,),
            )
            txn.write_receipt(
                stage,
                txn.CHILD_PREBOUNDARY_FAILED,
                identity,
                {
                    "status": "failed",
                    "error_type": "RuntimeError",
                    "error": "preboundary",
                    "traceback": "trace",
                },
                predecessors=(started,),
            )
            result = txn.classify_attempt(
                paths,
                identity,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            assert not result.retry_permitted
            assert "conflicts" in result.reason
        finally:
            lock.close()


@pytest.mark.parametrize("corrupt_name", (txn.CHILD_STARTED, txn.APP_STARTED, txn.CHILD_EXIT))
def test_corrupt_preboundary_receipt_never_authorizes_retry(corrupt_name: str) -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            intent = txn.load_receipt(stage, txn.INTENT)
            started = txn.write_receipt(
                stage,
                txn.CHILD_STARTED,
                identity,
                {"status": "started", "pid": os.getpid()},
                predecessors=(intent,),
            )
            if corrupt_name in {txn.APP_STARTED, txn.CHILD_EXIT}:
                app = txn.write_receipt(
                    stage,
                    txn.APP_STARTED,
                    identity,
                    {"status": "started", "fast_shutdown": True},
                    predecessors=(started,),
                )
            if corrupt_name == txn.CHILD_EXIT:
                _write_preboundary_exit(stage, identity, app)
            document = json.loads(stage.path(corrupt_name).read_text())
            if corrupt_name == txn.CHILD_STARTED:
                document["fd_cloexec"] = False
            elif corrupt_name == txn.APP_STARTED:
                document["fast_shutdown"] = False
            else:
                document["wifexited"] = False
            stage.path(corrupt_name).write_bytes(txn.pretty_json_bytes(document))
            result = txn.classify_attempt(
                paths,
                identity,
                spawn_succeeded=True,
                child_exit_observed=corrupt_name == txn.CHILD_EXIT,
            )
            assert not result.retry_permitted
            assert "malformed observed receipt" in result.reason
        finally:
            lock.close()


def test_prepared_binds_runtime_manifest_from_00_and_rejects_conflict() -> None:
    for conflicting in (False, True):
        with tempfile.TemporaryDirectory() as name:
            paths, lock, stage, identity = _new_attempt(Path(name))
            try:
                boundary = _through_boundary(stage, identity)
                payload = txn.durable_stage_bytes(stage, txn.STAGED_PAYLOAD, b"payload")
                report = txn.durable_stage_bytes(stage, txn.STAGED_REPORT_CORE, b"{}\n")
                fields = {
                    "status": "prepared",
                    "assignment_mask_sha256": "b" * 64,
                    "source_manifest_sha256": "d" * 64,
                    "checkpoint_manifest_sha256": "c" * 64,
                }
                if conflicting:
                    fields["runtime_asset_manifest_sha256"] = "f" * 64
                    with pytest.raises(txn.ReceiptError):
                        txn.write_receipt(
                            stage,
                            txn.PREPARED,
                            identity,
                            fields,
                            predecessors=(boundary, payload, report),
                        )
                    assert not stage.path(txn.PREPARED).exists()
                else:
                    prepared = txn.write_receipt(
                        stage,
                        txn.PREPARED,
                        identity,
                        fields,
                        predecessors=(boundary, payload, report),
                    )
                    assert prepared.payload is not None
                    assert prepared.payload["runtime_asset_manifest_sha256"] == "e" * 64
            finally:
                lock.close()


def test_nullable_recovery_50_with_logs_authorizes_preboundary_retry() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            intent = txn.load_receipt(stage, txn.INTENT)
            started = txn.write_receipt(
                stage,
                txn.CHILD_STARTED,
                identity,
                {"status": "started", "pid": os.getpid()},
                predecessors=(intent,),
            )
            child_exit = _write_preboundary_exit(
                stage, identity, started, raw_wait_status=None
            )
            assert child_exit.payload is not None
            assert child_exit.payload["raw_wait_status"] is None
            result = txn.classify_attempt(
                paths,
                identity,
                spawn_succeeded=True,
                child_exit_observed=True,
                recovery_lock_acquired=True,
            )
            assert not result.boundary_armed and result.retry_permitted
            txn.record_parent_failure(
                paths,
                identity,
                lock,
                reason="recovered abrupt child exit",
                raw_wait_status=None,
                spawn_succeeded=True,
                child_exit_observed=True,
                recovery_lock_acquired=True,
            )
            assert txn.validate_attempt_sequence(paths.output_stem).final_state == "retryable_failure"
        finally:
            lock.close()


def test_70_recomputation_rejects_forgery_and_authority_failure_is_terminal() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            intent = txn.load_receipt(stage, txn.INTENT)
            txn.write_receipt(
                stage,
                txn.PARENT_PRESPAWN_FAILED,
                identity,
                {"status": "failed", "error_type": "OSError", "error": "spawn"},
                predecessors=(intent,),
            )
            txn.record_parent_failure(
                paths,
                identity,
                lock,
                reason="caught spawn failure",
                raw_wait_status=None,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            document = json.loads(stage.path(txn.PARENT_FAILED).read_text())
            document["spawn_succeeded"] = True
            stage.path(txn.PARENT_FAILED).write_bytes(txn.pretty_json_bytes(document))
            with pytest.raises(txn.TransactionError):
                txn.validate_attempt_sequence(paths.output_stem)
        finally:
            lock.close()
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            intent = txn.load_receipt(stage, txn.INTENT)
            txn.write_receipt(
                stage,
                txn.PARENT_PRESPAWN_FAILED,
                identity,
                {"status": "failed", "error_type": "OSError", "error": "spawn"},
                predecessors=(intent,),
            )
            failure = txn.record_parent_failure(
                paths,
                identity,
                lock,
                reason="unsafe evidence",
                authority_failure="missing or unsafe child log",
                raw_wait_status=None,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            assert failure.payload is not None
            assert failure.payload["retry_permitted"] is False
            assert failure.payload["authority_failure"]
            with pytest.raises(txn.TransactionError):
                txn.validate_attempt_sequence(paths.output_stem)
        finally:
            lock.close()


def test_retryable_attempt_one_can_be_followed_by_committed_attempt_two() -> None:
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        paths, lock, first, identity1 = _new_attempt(root)
        try:
            intent = txn.load_receipt(first, txn.INTENT)
            txn.write_receipt(
                first,
                txn.PARENT_PRESPAWN_FAILED,
                identity1,
                {"status": "failed", "error_type": "OSError", "error": "spawn"},
                predecessors=(intent,),
            )
            txn.record_parent_failure(
                paths,
                identity1,
                lock,
                reason="retry one",
                raw_wait_status=None,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            second = txn.StageOnly.create(paths, 2)
            txn.compute_run_id(_run_inputs(paths))
            identity2 = txn.AttemptIdentity(
                identity1.run_id,
                txn.compute_attempt_id(identity1.run_id, 2),
                2,
            )
            metadata = _run_metadata(paths)
            txn.write_receipt(
                second,
                txn.INTENT,
                identity2,
                {"status": "prepared", "run_metadata": metadata, **metadata},
            )
            _complete_verified_attempt(second, identity2)
            result = txn.publish_verified_attempt(paths, identity2, lock)
            assert result["status"] == "committed"
            assert result["retryable_prior_attempts"] == [identity1.as_dict()]
        finally:
            lock.close()


def test_corrupt_lower_retry_is_rejected_before_any_canonical_publication() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, first, identity1 = _new_attempt(Path(name))
        try:
            intent = txn.load_receipt(first, txn.INTENT)
            txn.write_receipt(
                first,
                txn.PARENT_PRESPAWN_FAILED,
                identity1,
                {"status": "failed", "error_type": "OSError", "error": "spawn"},
                predecessors=(intent,),
            )
            txn.record_parent_failure(
                paths,
                identity1,
                lock,
                reason="retry one",
                raw_wait_status=None,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            second = txn.StageOnly.create(paths, 2)
            txn.compute_run_id(_run_inputs(paths))
            identity2 = txn.AttemptIdentity(
                identity1.run_id,
                txn.compute_attempt_id(identity1.run_id, 2),
                2,
            )
            metadata = _run_metadata(paths)
            txn.write_receipt(
                second,
                txn.INTENT,
                identity2,
                {"status": "prepared", "run_metadata": metadata, **metadata},
            )
            _complete_verified_attempt(second, identity2)
            forged = json.loads(first.path(txn.PARENT_FAILED).read_text())
            forged["spawn_succeeded"] = True
            first.path(txn.PARENT_FAILED).write_bytes(txn.pretty_json_bytes(forged))
            with pytest.raises(txn.TransactionError):
                txn.publish_verified_attempt(paths, identity2, lock)
            assert not paths.artifact.exists() and not paths.artifact.is_symlink()
            assert not paths.report.exists() and not paths.report.is_symlink()
            assert not paths.commit.exists() and not paths.commit.is_symlink()
        finally:
            lock.close()


def test_commit_rejects_changed_60_verified_and_extra_failure_link() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, _identity = _successful_namespace(Path(name))
        document = json.loads(paths.commit.read_text())
        document["verified"]["sha256"] = "0" * 64
        paths.commit.write_bytes(txn.pretty_json_bytes(document))
        with pytest.raises(txn.TransactionError):
            txn.validate_committed_namespace(paths)
    with tempfile.TemporaryDirectory() as name:
        paths, _identity = _successful_namespace(Path(name))
        Path(f"{paths.output_stem}.failed_attempt_999.json").write_bytes(b"foreign")
        with pytest.raises(txn.TransactionError):
            txn.validate_committed_namespace(paths)


def test_unsafe_canonical_and_attempt_symlinks_are_sealed_in_nonretryable_70() -> None:
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _verified_namespace(Path(name))
        try:
            paths.artifact.symlink_to(stage.path(txn.STAGED_PAYLOAD))
            with pytest.raises(txn.TransactionError):
                txn.publish_verified_attempt(paths, identity, lock)
            failure = txn.record_parent_failure(
                paths,
                identity,
                lock,
                reason="unsafe canonical artifact symlink",
                authority_failure="canonical artifact is non-regular",
                raw_wait_status=0,
                spawn_succeeded=True,
                child_exit_observed=True,
            )
            assert failure.payload is not None
            observed = failure.payload["canonical_observed"]["artifact"]
            assert observed["object_type"] == "symlink"
            assert "samefile_stage" not in observed
            assert not failure.payload["retry_permitted"]
        finally:
            lock.close()
    with tempfile.TemporaryDirectory() as name:
        paths, lock, stage, identity = _new_attempt(Path(name))
        try:
            outside = Path(name) / "outside.log"
            outside.write_bytes(b"foreign")
            stage.path(txn.STDOUT_LOG).symlink_to(outside)
            failure = txn.record_parent_failure(
                paths,
                identity,
                lock,
                reason="unsafe attempt log symlink",
                authority_failure="stdout is non-regular",
                raw_wait_status=None,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            assert failure.payload is not None
            assert failure.payload["observed"][txn.STDOUT_LOG]["object_type"] == "symlink"
            assert not failure.payload["retry_permitted"]
        finally:
            lock.close()
