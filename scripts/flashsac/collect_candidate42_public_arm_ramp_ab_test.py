#!/usr/bin/env python3
"""CPU-only tests for Candidate42's non-rollout transaction parent."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import candidate42_attempt_transaction as transaction
import candidate42_public_arm_ramp_collection as collection
import candidate42_public_arm_ramp_episode as contract
import collect_candidate42_public_arm_ramp_ab as parent


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_parent_source_has_no_forbidden_import_or_child_import() -> None:
    path = _root() / "scripts/flashsac/collect_candidate42_public_arm_ramp_ab.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("isaacsim", "omni", "pxr", "isaaclab.app")
    assert not any(name.startswith(forbidden) for name in imported)
    assert "collect_candidate42_public_arm_ramp_child" not in imported


def test_fresh_parent_import_does_not_load_runtime_application_modules() -> None:
    script = (
        "import json,sys; import collect_candidate42_public_arm_ramp_ab; "
        "bad=[n for n in sys.modules if n=='omni' or n.startswith(('isaacsim','omni.','pxr','isaaclab.app'))]; "
        "print(json.dumps(bad))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_root() / "scripts/flashsac")
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=_root(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == []


def test_waitpid_preserves_raw_status_and_reaps_once() -> None:
    process = subprocess.Popen(
        (sys.executable, "-c", "import os; os._exit(0)"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    raw = parent.waitpid_exact(process)
    assert os.WIFEXITED(raw)
    assert os.WEXITSTATUS(raw) == 0
    assert process.returncode == 0
    try:
        os.waitpid(process.pid, os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        raise AssertionError("child was not reaped exactly once")


def test_parent_derived_child_argv_has_exact_whitelisted_receipt() -> None:
    with tempfile.TemporaryDirectory() as name:
        stem = Path(name) / "trial"
        args = parent.build_parser().parse_args(
            [
                "--seed", "346", "--replicate", "b", "--num_envs", "8",
                "--output_stem", str(stem), "--headless",
            ]
        )
        identity = transaction.AttemptIdentity(
            run_id="a" * 64, attempt_id="b" * 64, attempt_number=1
        )
        command = parent._child_arguments(args, identity, 17)
        parser = argparse.ArgumentParser(allow_abbrev=False)
        collection.add_scientific_arguments(parser)
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--livestream", type=int, default=-1)
        parser.add_argument("--enable_cameras", action="store_true", default=False)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--experience", default="")
        parser.add_argument("--kit_args", default="")
        parser.add_argument("--_candidate42_run_id")
        parser.add_argument("--_candidate42_attempt_id")
        parser.add_argument("--_candidate42_attempt_number", type=int)
        parser.add_argument("--_candidate42_lock_fd", type=int)
        child = parser.parse_args(command[2:])
        assert collection._argument_receipt(
            args, stem
        ) == collection._argument_receipt(child, stem)


def test_second_log_open_failure_closes_first_descriptor() -> None:
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        real_open = os.open
        first_descriptor: list[int] = []
        calls = 0

        def fail_second(path, flags, mode=0o777):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic stderr open failure")
            descriptor = real_open(path, flags, mode)
            if calls == 1:
                first_descriptor.append(descriptor)
            return descriptor

        with mock.patch.object(parent.os, "open", side_effect=fail_second):
            try:
                parent._open_logs(directory)
            except OSError:
                pass
            else:
                raise AssertionError("expected second log open failure")
        try:
            os.fstat(first_descriptor[0])
        except OSError:
            pass
        else:
            raise AssertionError("first log descriptor leaked")


def test_open_logs_preserves_directory_fsync_error_and_cleans_both_streams() -> None:
    stdout = mock.Mock()
    stderr = mock.Mock()
    stdout.fileno.return_value = 101
    stderr.fileno.return_value = 102
    cleanup_flush_error = OSError("synthetic first-stream cleanup flush failure")
    cleanup_close_error = OSError("synthetic first-stream cleanup close failure")
    stdout.flush.side_effect = cleanup_flush_error
    stdout.close.side_effect = cleanup_close_error
    original_error = OSError("synthetic initial log directory fsync failure")
    directory_fsync_calls: list[Path] = []

    def fsync_directory(path: Path) -> None:
        directory_fsync_calls.append(path)
        if len(directory_fsync_calls) == 1:
            raise original_error

    descriptor_fsyncs: list[int] = []
    directory = Path("/synthetic/candidate42-open-logs")
    with mock.patch.object(parent.os, "open", side_effect=(101, 102)), mock.patch.object(
        parent.os, "fdopen", side_effect=(stdout, stderr)
    ), mock.patch.object(
        parent.os,
        "fsync",
        side_effect=lambda descriptor: descriptor_fsyncs.append(descriptor),
    ), mock.patch.object(parent, "_fsync_directory", side_effect=fsync_directory):
        try:
            parent._open_logs(directory)
        except OSError as error:
            assert error is original_error
        else:
            raise AssertionError("expected initial directory fsync failure")

    stdout.flush.assert_called_once_with()
    stdout.fileno.assert_called_once_with()
    stdout.close.assert_called_once_with()
    stderr.flush.assert_called_once_with()
    stderr.fileno.assert_called_once_with()
    stderr.close.assert_called_once_with()
    assert descriptor_fsyncs == [101, 102]
    assert directory_fsync_calls == [directory, directory]


def test_open_logs_closes_unowned_descriptor_when_fdopen_fails() -> None:
    stdout = mock.Mock()
    stdout.fileno.return_value = 101
    fdopen_error = OSError("synthetic stderr fdopen failure")
    directory = Path("/synthetic/candidate42-fdopen")

    with mock.patch.object(parent.os, "open", side_effect=(101, 102)), mock.patch.object(
        parent.os, "fdopen", side_effect=(stdout, fdopen_error)
    ), mock.patch.object(parent.os, "close") as close_descriptor, mock.patch.object(
        parent.os, "fsync"
    ) as fsync_descriptor, mock.patch.object(parent, "_fsync_directory"):
        try:
            parent._open_logs(directory)
        except OSError as error:
            assert error is fdopen_error
        else:
            raise AssertionError("expected fdopen failure")

    close_descriptor.assert_called_once_with(102)
    stdout.flush.assert_called_once_with()
    stdout.fileno.assert_called_once_with()
    stdout.close.assert_called_once_with()
    fsync_descriptor.assert_called_once_with(101)


def test_spawn_failure_fsyncs_and_closes_both_logs() -> None:
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)

        def fail_spawn(*_args, **_kwargs):
            raise OSError("synthetic spawn failure")

        try:
            parent.spawn_child_exact(
                directory, [sys.executable], 0, popen_factory=fail_spawn
            )
        except OSError:
            pass
        else:
            raise AssertionError("expected spawn failure")
        for log in (directory / "stdout.log", directory / "stderr.log"):
            assert log.is_file() and not log.is_symlink()
            with log.open("ab"):
                pass


def test_finish_logs_preserves_successful_cleanup_semantics() -> None:
    stdout = mock.Mock()
    stderr = mock.Mock()
    stdout.fileno.return_value = 101
    stderr.fileno.return_value = 102
    fsynced: list[int] = []
    directories: list[Path] = []
    directory = Path("/synthetic/candidate42-attempt")

    with mock.patch.object(
        parent.os, "fsync", side_effect=lambda descriptor: fsynced.append(descriptor)
    ), mock.patch.object(
        parent,
        "_fsync_directory",
        side_effect=lambda path: directories.append(path),
    ):
        assert parent._finish_logs(stdout, stderr, directory) is None

    stdout.flush.assert_called_once_with()
    stdout.close.assert_called_once_with()
    stderr.flush.assert_called_once_with()
    stderr.close.assert_called_once_with()
    assert fsynced == [101, 102]
    assert directories == [directory]


def test_finish_logs_cleans_second_stream_after_each_first_stream_failure() -> None:
    for operation in ("flush", "fsync", "close"):
        stdout = mock.Mock()
        stderr = mock.Mock()
        stdout.fileno.return_value = 101
        stderr.fileno.return_value = 102
        expected = OSError(f"synthetic first-stream {operation} failure")
        if operation == "flush":
            stdout.flush.side_effect = expected
        elif operation == "close":
            stdout.close.side_effect = expected

        fsynced: list[int] = []

        def fsync(descriptor: int) -> None:
            fsynced.append(descriptor)
            if operation == "fsync" and descriptor == 101:
                raise expected

        directories: list[Path] = []
        directory = Path(f"/synthetic/candidate42-{operation}")
        with mock.patch.object(parent.os, "fsync", side_effect=fsync), mock.patch.object(
            parent,
            "_fsync_directory",
            side_effect=lambda path: directories.append(path),
        ):
            try:
                parent._finish_logs(stdout, stderr, directory)
            except OSError as error:
                assert error is expected
            else:
                raise AssertionError(f"expected {operation} failure")

        stdout.close.assert_called_once_with()
        stderr.flush.assert_called_once_with()
        stderr.fileno.assert_called_once_with()
        stderr.close.assert_called_once_with()
        assert 102 in fsynced
        assert directories == [directory]


def _integration_fixture(name: str) -> tuple[list[str], SimpleNamespace, dict[str, str]]:
    stem = Path(name) / "trial"
    argv = [
        "--seed", "346", "--replicate", "b", "--num_envs", "8",
        "--output_stem", str(stem), "--headless",
    ]
    args = parent.build_parser().parse_args(argv)
    source = {"source.py": "1" * 64}
    checkpoints = {"checkpoint": "2" * 64}
    runtime_assets = dict(collection.RUNTIME_ASSET_EXPECTED_SHA256)
    spec = SimpleNamespace(
        repository_root=_root(),
        output_stem=stem.resolve(),
        seed=346,
        replicate="b",
        num_envs=8,
        collection_commit="4" * 40,
        implementation_commit="5" * 40,
        preregistration_tag_commit="7" * 40,
        assignment_mask_sha256="6" * 64,
        source_sha256=source,
        checkpoint_sha256=checkpoints,
        argument_receipt=collection._argument_receipt(args, stem),
    )
    return argv, spec, runtime_assets


def _begin_attempt(name: str):
    argv, spec, runtime_assets = _integration_fixture(name)
    paths = transaction.TransactionPaths.from_output_stem(spec.output_stem)
    lock = transaction.NamespaceLock.acquire(paths)
    run_identity = parent._run_identity(spec, runtime_assets, argv)
    run_id = transaction.compute_run_id(run_identity)
    identity = transaction.AttemptIdentity(
        run_id=run_id,
        attempt_id=transaction.compute_attempt_id(run_id, 1),
        attempt_number=1,
    )
    stage = transaction.StageOnly.create(paths, 1)
    metadata = parent._run_metadata(spec, runtime_assets)
    intent = transaction.write_receipt(
        stage,
        transaction.INTENT,
        identity,
        {"status": "prepared", "run_metadata": metadata, **metadata},
    )
    return argv, spec, runtime_assets, paths, lock, identity, stage, intent


def _durable_test_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def test_preregistration_commit_is_in_00_metadata_but_not_fixed_run_id_fields() -> None:
    with tempfile.TemporaryDirectory() as name:
        argv, spec, runtime_assets = _integration_fixture(name)
        metadata = parent._run_metadata(spec, runtime_assets)
        run_identity = parent._run_identity(spec, runtime_assets, argv)
        assert metadata["preregistration_tag_commit"] == (
            spec.preregistration_tag_commit
        )
        assert "preregistration_tag_commit" not in run_identity
        assert set(run_identity) == set(transaction.RUN_ID_FIELDS)


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _fake_fast_exit_spawn(
    stem: Path,
    spec: SimpleNamespace,
    runtime_assets: dict[str, str],
    *,
    boundary: bool = True,
    outcome: str = "success",
):
    def spawn(directory: Path, command, inherited_fd: int):
        identity = transaction.AttemptIdentity(
            run_id=_flag(command, "--_candidate42_run_id"),
            attempt_id=_flag(command, "--_candidate42_attempt_id"),
            attempt_number=int(_flag(command, "--_candidate42_attempt_number")),
        )
        child_code = r"""
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, sys.argv[1])
import candidate42_attempt_transaction as transaction

paths = transaction.TransactionPaths.from_output_stem(Path(sys.argv[2]))
identity = transaction.AttemptIdentity(
    run_id=sys.argv[3],
    attempt_id=sys.argv[4],
    attempt_number=int(sys.argv[5]),
)
lock_fd = int(sys.argv[6])
transaction.validate_inherited_lock(lock_fd, paths.lock)
stage = transaction.StageOnly.open(paths, identity.attempt_number)
intent = transaction.load_receipt(stage, transaction.INTENT)
started = transaction.write_receipt(
    stage,
    transaction.CHILD_STARTED,
    identity,
    {"status": "started", "pid": os.getpid()},
    predecessors=(intent,),
)
app = transaction.write_receipt(
    stage,
    transaction.APP_STARTED,
    identity,
    {"status": "started", "fast_shutdown": True},
    predecessors=(started,),
)
outcome = sys.argv[12]
if sys.argv[11] == "1":
    armed = transaction.write_receipt(
        stage,
        transaction.FIRST_STEP_ARMED,
        identity,
        {"status": "armed"},
        predecessors=(app,),
    )
    if outcome == "child_failed":
        failed = transaction.write_receipt(
            stage,
            transaction.CHILD_FAILED,
            identity,
            {
                "status": "failed",
                "error_type": "RuntimeError",
                "error": "synthetic post-boundary child failure",
                "traceback": "synthetic traceback",
            },
            predecessors=(armed,),
        )
        transaction.write_receipt(
            stage,
            transaction.CLOSE_FAILURE,
            identity,
            {"status": "close_requested", "reason": "postboundary_failure"},
            predecessors=(failed,),
        )
    else:
        payload = transaction.durable_stage_bytes(
            stage, transaction.STAGED_PAYLOAD, b"exact-payload"
        )
        report = transaction.durable_stage_bytes(
            stage,
            transaction.STAGED_REPORT_CORE,
            transaction.pretty_json_bytes({}),
        )
        prepared = transaction.write_receipt(
            stage,
            transaction.PREPARED,
            identity,
            {
                "status": "prepared",
                "assignment_mask_sha256": sys.argv[7],
                "source_manifest_sha256": sys.argv[8],
                "checkpoint_manifest_sha256": sys.argv[9],
                "runtime_asset_manifest_sha256": sys.argv[10],
            },
            predecessors=(armed, payload, report),
        )
        transaction.write_receipt(
            stage,
            transaction.CLOSE_SUCCESS,
            identity,
            {"status": "close_requested", "reason": "success"},
            predecessors=(prepared,),
        )
if outcome == "nonzero":
    os._exit(7)
if outcome == "sigkill":
    os.kill(os.getpid(), signal.SIGKILL)
os._exit(0)
"""
        stdout, stderr = parent._open_logs(directory)
        process = subprocess.Popen(
            (
                sys.executable,
                "-c",
                child_code,
                str(_root() / "scripts/flashsac"),
                str(stem),
                identity.run_id,
                identity.attempt_id,
                str(identity.attempt_number),
                str(inherited_fd),
                spec.assignment_mask_sha256,
                contract.manifest_sha256(spec.source_sha256),
                contract.canonical_json_sha256(spec.checkpoint_sha256),
                contract.manifest_sha256(runtime_assets),
                "1" if boundary else "0",
                outcome,
            ),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            pass_fds=(inherited_fd,),
            close_fds=True,
        )
        return process, stdout, stderr

    return spawn


def _generic_exact_publish(**kwargs):
    stage = kwargs["stage"]
    identity = kwargs["identity"]
    paths = kwargs["paths"]
    lock = kwargs["lock"]
    child_exit = kwargs["child_exit"]
    payload = transaction.file_evidence(stage, transaction.STAGED_PAYLOAD)
    report = transaction.file_evidence(stage, transaction.STAGED_REPORT_CORE)
    intent = transaction.load_receipt(stage, transaction.INTENT)
    assert intent.payload is not None
    document = {
        "transaction_contract": transaction.TRANSACTION_CONTRACT,
        "receipt": transaction.FINAL_REPORT,
        "identity": identity.as_dict(),
        "predecessors": {transaction.CHILD_EXIT: child_exit.as_dict()},
        "kind": transaction.FINAL_REPORT_KIND,
        "version": 1,
        "status": "complete",
        **identity.as_dict(),
        "child_exit_sha256": child_exit.sha256,
        "post_exit_authority": {},
        "source_manifest_sha256": intent.payload["source_manifest_sha256"],
        "checkpoint_manifest_sha256": intent.payload[
            "checkpoint_manifest_sha256"
        ],
        "runtime_asset_manifest_sha256": intent.payload[
            "runtime_asset_manifest_sha256"
        ],
        "artifact_sha256": payload.sha256,
        "artifact_size": payload.size,
        "artifact_output": str(paths.artifact),
        "report_core": {},
        "report_core_sha256": report.sha256,
    }
    final = transaction.durable_stage_bytes(
        stage, transaction.FINAL_REPORT, transaction.pretty_json_bytes(document)
    )
    transaction.write_receipt(
        stage,
        transaction.VERIFIED,
        identity,
        {"status": "verified"},
        predecessors=(final,),
    )
    return transaction.publish_verified_attempt(paths, identity, lock)


def test_fake_fast_exit_after_40_commits_exact_stage_bytes() -> None:
    with tempfile.TemporaryDirectory() as name:
        argv, spec, runtime_assets = _integration_fixture(name)
        spawn = _fake_fast_exit_spawn(
            spec.output_stem, spec, runtime_assets, boundary=True
        )
        with mock.patch.object(
            parent.collection, "build_spec", return_value=spec
        ), mock.patch.object(
            parent.collection,
            "runtime_asset_fingerprints",
            return_value=runtime_assets,
        ), mock.patch.object(
            parent, "spawn_child_exact", side_effect=spawn
        ), mock.patch.object(
            parent, "_verify_and_publish", side_effect=_generic_exact_publish
        ):
            result = parent.run_parent(argv)
        assert result["status"] == "committed"
        attempt = Path(f"{spec.output_stem}.c42_txn/attempt_001")
        bindings = (
            (
                transaction.STAGED_PAYLOAD,
                "canonical_artifact_output",
                "artifact_sha256",
            ),
            (transaction.FINAL_REPORT, "canonical_report_output", "report_sha256"),
            (transaction.COMMITTED, "canonical_commit_output", "commit_sha256"),
        )
        for local_name, output_name, digest_name in bindings:
            local = attempt / local_name
            canonical = Path(result[output_name])
            assert os.path.samefile(local, canonical)
            assert local.read_bytes() == canonical.read_bytes()
            assert transaction.sha256_file(local) == result[digest_name]


def test_lock_dup_failure_after_00_records_05_and_retryable_70() -> None:
    with tempfile.TemporaryDirectory() as name:
        argv, spec, runtime_assets = _integration_fixture(name)
        spawn = mock.Mock()
        with mock.patch.object(
            parent.collection, "build_spec", return_value=spec
        ), mock.patch.object(
            parent.collection,
            "runtime_asset_fingerprints",
            return_value=runtime_assets,
        ), mock.patch.object(
            parent.transaction.NamespaceLock,
            "duplicate_for_child",
            side_effect=OSError("synthetic EMFILE"),
        ), mock.patch.object(
            parent, "spawn_child_exact", spawn
        ):
            try:
                parent.run_parent(argv)
            except OSError as error:
                assert "EMFILE" in str(error)
            else:
                raise AssertionError("lock-dup failure was accepted")
        spawn.assert_not_called()
        paths = transaction.TransactionPaths.from_output_stem(spec.output_stem)
        stage = transaction.StageOnly.open(paths, 1)
        transaction.load_receipt(stage, transaction.INTENT)
        transaction.load_receipt(stage, transaction.PARENT_PRESPAWN_FAILED)
        failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
        assert failure.payload is not None
        assert failure.payload["spawn_succeeded"] is False
        assert failure.payload["retry_permitted"] is True
        assert not stage.path(transaction.STDOUT_LOG).exists()
        assert not stage.path(transaction.STDERR_LOG).exists()


def test_postboundary_nonzero_sigkill_and_45_exit_zero_are_terminal() -> None:
    cases = (
        ("nonzero", True, 7, None),
        ("sigkill", False, None, signal.SIGKILL),
        ("child_failed", True, 0, None),
    )
    for outcome, wifexited, exit_code, signal_number in cases:
        with tempfile.TemporaryDirectory() as name:
            argv, spec, runtime_assets = _integration_fixture(name)
            spawn = _fake_fast_exit_spawn(
                spec.output_stem,
                spec,
                runtime_assets,
                boundary=True,
                outcome=outcome,
            )
            with mock.patch.object(
                parent.collection, "build_spec", return_value=spec
            ), mock.patch.object(
                parent.collection,
                "runtime_asset_fingerprints",
                return_value=runtime_assets,
            ), mock.patch.object(
                parent, "spawn_child_exact", side_effect=spawn
            ):
                try:
                    parent.run_parent(argv)
                except BaseException:
                    pass
                else:
                    raise AssertionError(f"{outcome} unexpectedly committed")
            paths = transaction.TransactionPaths.from_output_stem(spec.output_stem)
            stage = transaction.StageOnly.open(paths, 1)
            child_exit = transaction.load_receipt(stage, transaction.CHILD_EXIT)
            failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
            assert child_exit.payload is not None and failure.payload is not None
            assert child_exit.payload["wifexited"] is wifexited
            assert child_exit.payload["exit_code"] == exit_code
            assert child_exit.payload["signal"] == signal_number
            assert failure.payload["boundary_armed"] is True
            assert failure.payload["retry_permitted"] is False
            if outcome == "child_failed":
                transaction.load_receipt(stage, transaction.CHILD_FAILED)
                transaction.load_receipt(stage, transaction.CLOSE_FAILURE)
            assert not paths.artifact.exists()
            assert not paths.report.exists()
            assert not paths.commit.exists()


def test_verified_recovery_validates_before_any_publication() -> None:
    with tempfile.TemporaryDirectory() as name:
        (
            _argv,
            spec,
            _runtime_assets,
            paths,
            lock,
            identity,
            stage,
            _intent,
        ) = _begin_attempt(name)
        _durable_test_bytes(stage.path(transaction.VERIFIED), b"verified-sentinel")
        events: list[str] = []
        try:
            with mock.patch.object(
                parent,
                "_validate_verified_with_spec",
                side_effect=lambda *_args: events.append("validate"),
            ), mock.patch.object(
                parent.transaction,
                "recover_verified_publication",
                side_effect=lambda *_args: events.append("publish"),
            ), mock.patch.object(
                parent,
                "_validate_committed_with_spec",
                side_effect=lambda *_args: events.append("committed") or {"status": "committed"},
            ):
                result = parent._prepare_attempt_number(
                    paths,
                    lock,
                    spec.output_stem,
                    expected_run_id=identity.run_id,
                    spec=spec,
                )
            assert result is None
            assert events == ["validate", "publish", "committed"]
        finally:
            lock.close()


def test_verified_recovery_validation_failure_never_publishes() -> None:
    with tempfile.TemporaryDirectory() as name:
        (
            _argv,
            spec,
            _runtime_assets,
            paths,
            lock,
            identity,
            stage,
            _intent,
        ) = _begin_attempt(name)
        _durable_test_bytes(stage.path(transaction.VERIFIED), b"verified-sentinel")
        publish = mock.Mock()
        failure = mock.Mock()
        try:
            with mock.patch.object(
                parent,
                "_validate_verified_with_spec",
                side_effect=ValueError("synthetic scientific drift"),
            ), mock.patch.object(
                parent.transaction,
                "recover_verified_publication",
                publish,
            ), mock.patch.object(
                parent.transaction, "record_parent_failure", failure
            ):
                try:
                    parent._prepare_attempt_number(
                        paths,
                        lock,
                        spec.output_stem,
                        expected_run_id=identity.run_id,
                        spec=spec,
                    )
                except ValueError as error:
                    assert "scientific drift" in str(error)
                else:
                    raise AssertionError("invalid verified attempt was accepted")
            publish.assert_not_called()
            failure.assert_called_once()
        finally:
            lock.close()


def test_lock_recovery_without_child_evidence_creates_logs_50_and_retryable_70() -> None:
    with tempfile.TemporaryDirectory() as name:
        (
            _argv,
            spec,
            _runtime_assets,
            paths,
            lock,
            identity,
            stage,
            _intent,
        ) = _begin_attempt(name)
        try:
            next_number = parent._prepare_attempt_number(
                paths,
                lock,
                spec.output_stem,
                expected_run_id=identity.run_id,
                spec=spec,
            )
            assert next_number == 2
            child_exit = transaction.load_receipt(stage, transaction.CHILD_EXIT)
            failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
            assert child_exit.payload is not None and failure.payload is not None
            assert child_exit.payload["raw_wait_status"] is None
            assert child_exit.payload["lock_release_proven"] is True
            assert failure.payload["retry_permitted"] is True
            assert failure.payload["recovery_lock_acquired"] is True
            for log_name in (transaction.STDOUT_LOG, transaction.STDERR_LOG):
                assert stage.path(log_name).is_file()
                assert stage.path(log_name).stat().st_size == 0
        finally:
            lock.close()


def test_lock_recovery_infers_05_as_retryable_without_fabricating_50() -> None:
    with tempfile.TemporaryDirectory() as name:
        (
            _argv,
            spec,
            _runtime_assets,
            paths,
            lock,
            identity,
            stage,
            intent,
        ) = _begin_attempt(name)
        transaction.write_receipt(
            stage,
            transaction.PARENT_PRESPAWN_FAILED,
            identity,
            {
                "status": "failed",
                "error_type": "OSError",
                "error": "synthetic pre-spawn failure",
            },
            predecessors=(intent,),
        )
        try:
            next_number = parent._prepare_attempt_number(
                paths,
                lock,
                spec.output_stem,
                expected_run_id=identity.run_id,
                spec=spec,
            )
            assert next_number == 2
            assert not stage.path(transaction.CHILD_EXIT).exists()
            failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
            assert failure.payload is not None
            assert failure.payload["spawn_succeeded"] is False
            assert failure.payload["retry_permitted"] is True
        finally:
            lock.close()


def test_existing_raw_50_is_copied_exactly_into_retryable_70() -> None:
    with tempfile.TemporaryDirectory() as name:
        (
            _argv,
            spec,
            _runtime_assets,
            paths,
            lock,
            identity,
            stage,
            _intent,
        ) = _begin_attempt(name)
        parent._fsync_recovery_logs(stage, create_missing=True)
        child_exit = parent._write_child_exit(stage, identity, 0)
        assert child_exit.payload is not None
        try:
            next_number = parent._prepare_attempt_number(
                paths,
                lock,
                spec.output_stem,
                expected_run_id=identity.run_id,
                spec=spec,
            )
            assert next_number == 2
            failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
            assert failure.payload is not None
            assert failure.payload["raw_wait_status"] == 0
            assert failure.payload["child_exit_observed"] is True
            assert failure.payload["retry_permitted"] is True
        finally:
            lock.close()


def test_contradictory_05_and_raw_50_are_terminally_rejected() -> None:
    with tempfile.TemporaryDirectory() as name:
        (
            _argv,
            spec,
            _runtime_assets,
            paths,
            lock,
            identity,
            stage,
            intent,
        ) = _begin_attempt(name)
        transaction.write_receipt(
            stage,
            transaction.PARENT_PRESPAWN_FAILED,
            identity,
            {
                "status": "failed",
                "error_type": "OSError",
                "error": "synthetic contradictory pre-spawn failure",
            },
            predecessors=(intent,),
        )
        parent._fsync_recovery_logs(stage, create_missing=True)
        parent._write_child_exit(stage, identity, 0)
        try:
            try:
                parent._prepare_attempt_number(
                    paths,
                    lock,
                    spec.output_stem,
                    expected_run_id=identity.run_id,
                    spec=spec,
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("contradictory 05/50 evidence was accepted")
            assert not paths.attempt_path(2).exists()
            failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
            assert failure.payload is not None
            assert failure.payload["raw_wait_status"] == 0
            assert failure.payload["retry_permitted"] is False
        finally:
            lock.close()


def test_child_evidence_with_missing_logs_is_terminal_and_never_retried() -> None:
    with tempfile.TemporaryDirectory() as name:
        (
            _argv,
            spec,
            _runtime_assets,
            paths,
            lock,
            identity,
            stage,
            intent,
        ) = _begin_attempt(name)
        transaction.write_receipt(
            stage,
            transaction.CHILD_STARTED,
            identity,
            {"status": "started", "pid": os.getpid()},
            predecessors=(intent,),
        )
        try:
            for invocation in range(2):
                try:
                    parent._prepare_attempt_number(
                        paths,
                        lock,
                        spec.output_stem,
                        expected_run_id=identity.run_id,
                        spec=spec,
                    )
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("unsafe log gap was accepted")
                assert not paths.attempt_path(2).exists(), invocation
            failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
            assert failure.payload is not None
            assert failure.payload["retry_permitted"] is False
            assert failure.payload["boundary_armed"] is False
            assert failure.payload["recovery_lock_acquired"] is True
            assert failure.payload["authority_failure"].startswith(
                "unsafe recovery log evidence:"
            )
        finally:
            lock.close()


def test_unsafe_recovery_log_symlink_is_sealed_terminal_without_following_it() -> None:
    with tempfile.TemporaryDirectory() as name:
        (
            _argv,
            spec,
            _runtime_assets,
            paths,
            lock,
            identity,
            stage,
            _intent,
        ) = _begin_attempt(name)
        target = Path(name) / "outside.log"
        _durable_test_bytes(target, b"must-not-be-consumed")
        os.symlink(target, stage.path(transaction.STDOUT_LOG))
        try:
            for _invocation in range(2):
                try:
                    parent._prepare_attempt_number(
                        paths,
                        lock,
                        spec.output_stem,
                        expected_run_id=identity.run_id,
                        spec=spec,
                    )
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("unsafe recovery symlink was accepted")
                assert not paths.attempt_path(2).exists()
            failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
            assert failure.payload is not None
            assert failure.payload["retry_permitted"] is False
            observed = failure.payload["observed"][transaction.STDOUT_LOG]
            assert observed["object_type"] == "symlink"
            assert target.read_bytes() == b"must-not-be-consumed"
        finally:
            lock.close()


def _assert_parent_fault_records_70(fault: str, *, boundary: bool = True) -> dict:
    with tempfile.TemporaryDirectory() as name:
        argv, spec, runtime_assets = _integration_fixture(name)
        spawn = _fake_fast_exit_spawn(
            spec.output_stem, spec, runtime_assets, boundary=boundary
        )
        patches = [
            mock.patch.object(parent.collection, "build_spec", return_value=spec),
            mock.patch.object(
                parent.collection,
                "runtime_asset_fingerprints",
                return_value=runtime_assets,
            ),
            mock.patch.object(parent, "spawn_child_exact", side_effect=spawn),
        ]
        original_wait = parent.waitpid_exact
        if fault == "waitpid":
            def failed_wait(process):
                if process.returncode is None:
                    original_wait(process)
                raise ChildProcessError("synthetic unknown wait status")
            patches.append(mock.patch.object(parent, "waitpid_exact", side_effect=failed_wait))
        elif fault == "finish_logs":
            original_finish = parent._finish_logs
            def failed_finish(*args):
                original_finish(*args)
                raise OSError("synthetic log fsync failure")
            patches.append(mock.patch.object(parent, "_finish_logs", side_effect=failed_finish))
        elif fault == "child_exit":
            patches.append(
                mock.patch.object(
                    parent, "_write_child_exit", side_effect=OSError("synthetic 50 failure")
                )
            )
        else:
            patches.append(
                mock.patch.object(
                    parent,
                    "_verify_and_publish",
                    side_effect=RuntimeError(f"synthetic {fault} drift/failure"),
                )
            )
        entered = [patcher.start() for patcher in patches]
        try:
            try:
                parent.run_parent(argv)
            except BaseException:
                pass
            else:
                raise AssertionError("parent fault unexpectedly committed")
        finally:
            for patcher in reversed(patches):
                patcher.stop()
        paths = transaction.TransactionPaths.from_output_stem(spec.output_stem)
        stage = transaction.StageOnly.open(paths, 1)
        failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
        assert Path(f"{spec.output_stem}.failed_attempt_001.json").is_file()
        return dict(failure.payload or {})


def test_wait_log_50_and_publish_faults_all_record_70() -> None:
    for fault in ("waitpid", "finish_logs", "child_exit", "publish"):
        receipt = _assert_parent_fault_records_70(fault)
        assert receipt["boundary_armed"] is True
        assert receipt["retry_permitted"] is False


def _assert_actual_post_exit_authority_drift_records_70(category: str) -> dict:
    with tempfile.TemporaryDirectory() as name:
        argv, spec, runtime_assets = _integration_fixture(name)
        spawn = _fake_fast_exit_spawn(
            spec.output_stem, spec, runtime_assets, boundary=True
        )
        source = dict(spec.source_sha256)
        checkpoints = dict(spec.checkpoint_sha256)
        runtime = dict(runtime_assets)
        git = {
            "commit": spec.collection_commit,
            "branch": contract.REQUIRED_BRANCH,
            "source_files_dirty": [],
            "flashsac_commit": contract.FLASHSAC_FORK_COMMIT,
            "flashsac_dirty": False,
        }
        if category == "source":
            source["source.py"] = "9" * 64
        elif category == "checkpoint":
            checkpoints["checkpoint"] = "9" * 64
        elif category == "git":
            git["commit"] = "9" * 40

        runtime_calls = 0

        def current_runtime(_root_path):
            nonlocal runtime_calls
            runtime_calls += 1
            if category == "runtime_asset" and runtime_calls >= 2:
                return {"asset.usd": "9" * 64}
            return runtime

        def resolve_tag(_root_path, tag):
            if category == "collection_tag" and tag == collection.COLLECTION_SEAL_TAG:
                return "9" * 40
            if category == "preregistration_tag" and tag == collection.PREREGISTRATION_TAG:
                return "9" * 40
            if tag == collection.PREREGISTRATION_TAG:
                return spec.preregistration_tag_commit
            return spec.collection_commit

        def validate_submodules(_root_path):
            if category == "submodule":
                raise RuntimeError("synthetic dirty submodule")

        def staged(stage, _spec):
            return (
                {},
                {},
                transaction.file_evidence(stage, transaction.STAGED_PAYLOAD),
            )

        runtime_mock = mock.Mock(side_effect=current_runtime)
        source_mock = mock.Mock(return_value=source)
        checkpoint_mock = mock.Mock(return_value=checkpoints)
        git_mock = mock.Mock(return_value=git)
        resolve_mock = mock.Mock(side_effect=resolve_tag)
        submodule_mock = mock.Mock(side_effect=validate_submodules)
        sealed_plan_mock = mock.Mock(
            return_value={
                "preregistration_receipt": {
                    "commit": spec.preregistration_tag_commit
                }
            }
        )
        with mock.patch.object(
            parent.collection, "build_spec", return_value=spec
        ), mock.patch.object(
            parent.collection,
            "runtime_asset_fingerprints",
            runtime_mock,
        ), mock.patch.object(
            parent.collection, "source_fingerprints", source_mock
        ), mock.patch.object(
            parent.collection, "_current_hashes", checkpoint_mock
        ), mock.patch.object(
            parent.collection, "git_provenance", git_mock
        ), mock.patch.object(
            parent.collection, "_git_resolve", resolve_mock
        ), mock.patch.object(
            parent.collection,
            "validate_all_submodules_clean",
            submodule_mock,
        ), mock.patch.object(
            parent.collection.artifact_contract,
            "validate_sealed_plan",
            sealed_plan_mock,
        ), mock.patch.object(
            parent, "spawn_child_exact", side_effect=spawn
        ), mock.patch.object(
            parent, "_validate_staged", side_effect=staged
        ):
            try:
                parent.run_parent(argv)
            except parent.PostExitAuthorityError:
                pass
            else:
                raise AssertionError(f"{category} authority drift was accepted")

        target_calls = {
            "source": source_mock.called,
            "checkpoint": checkpoint_mock.called,
            "runtime_asset": runtime_mock.call_count >= 2,
            "git": git_mock.called,
            "collection_tag": any(
                call.args[1] == collection.COLLECTION_SEAL_TAG
                for call in resolve_mock.call_args_list
            ),
            "preregistration_tag": (
                sealed_plan_mock.called
                and any(
                    call.args[1] == collection.PREREGISTRATION_TAG
                    for call in resolve_mock.call_args_list
                )
            ),
            "submodule": submodule_mock.called,
        }
        assert target_calls[category], f"{category} checker was not reached"

        paths = transaction.TransactionPaths.from_output_stem(spec.output_stem)
        stage = transaction.StageOnly.open(paths, 1)
        failure = transaction.load_receipt(stage, transaction.PARENT_FAILED)
        assert failure.payload is not None
        assert Path(f"{spec.output_stem}.failed_attempt_001.json").is_file()
        return dict(failure.payload)


def test_post_exit_authority_drift_categories_all_record_70() -> None:
    for category in (
        "source",
        "checkpoint",
        "runtime_asset",
        "git",
        "collection_tag",
        "preregistration_tag",
        "submodule",
    ):
        receipt = _assert_actual_post_exit_authority_drift_records_70(category)
        expected_error = {
            "source": "source or checkpoint changed",
            "checkpoint": "source or checkpoint changed",
            "runtime_asset": "runtime asset changed",
            "git": "Git authority changed",
            "collection_tag": "Git authority changed",
            "preregistration_tag": "preregistration tag or sealed receipt changed",
            "submodule": "dirty submodule",
        }[category]
        assert expected_error in receipt["authority_failure"]
        assert receipt["boundary_armed"] is True
        assert receipt["retry_permitted"] is False


def test_unknown_child_liveness_is_never_retryable() -> None:
    receipt = _assert_parent_fault_records_70("waitpid", boundary=False)
    assert receipt["boundary_armed"] is False
    assert receipt["retry_permitted"] is False


def main() -> None:
    test_parent_source_has_no_forbidden_import_or_child_import()
    test_fresh_parent_import_does_not_load_runtime_application_modules()
    test_waitpid_preserves_raw_status_and_reaps_once()
    test_parent_derived_child_argv_has_exact_whitelisted_receipt()
    test_second_log_open_failure_closes_first_descriptor()
    test_spawn_failure_fsyncs_and_closes_both_logs()
    test_preregistration_commit_is_in_00_metadata_but_not_fixed_run_id_fields()
    test_fake_fast_exit_after_40_commits_exact_stage_bytes()
    test_lock_dup_failure_after_00_records_05_and_retryable_70()
    test_postboundary_nonzero_sigkill_and_45_exit_zero_are_terminal()
    test_verified_recovery_validates_before_any_publication()
    test_verified_recovery_validation_failure_never_publishes()
    test_lock_recovery_without_child_evidence_creates_logs_50_and_retryable_70()
    test_lock_recovery_infers_05_as_retryable_without_fabricating_50()
    test_existing_raw_50_is_copied_exactly_into_retryable_70()
    test_contradictory_05_and_raw_50_are_terminally_rejected()
    test_child_evidence_with_missing_logs_is_terminal_and_never_retried()
    test_unsafe_recovery_log_symlink_is_sealed_terminal_without_following_it()
    test_wait_log_50_and_publish_faults_all_record_70()
    test_post_exit_authority_drift_categories_all_record_70()
    test_unknown_child_liveness_is_never_retryable()
    print("collect_candidate42_public_arm_ramp_ab tests passed")


if __name__ == "__main__":
    main()
