#!/usr/bin/env python3
"""Run Candidate42 attempts under a durable parent-owned transaction."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

import candidate42_attempt_transaction as transaction
import candidate42_public_arm_ramp_collection as collection
import candidate42_public_arm_ramp_episode as artifact_contract


_ATTEMPT = re.compile(r"attempt_([0-9]{3})")
_CHILD_BASENAME = "collect_candidate42_public_arm_ramp_child.py"


class PostExitAuthorityError(RuntimeError):
    """Current source/checkpoint/runtime/Git authority did not revalidate."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    collection.add_scientific_arguments(parser)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--livestream", type=int, choices=(-1, 0, 1, 2), default=-1)
    parser.add_argument("--enable_cameras", action="store_true", default=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experience", default="")
    parser.add_argument("--kit_args", default=artifact_contract.KIT_ARGS)
    return parser


def _canonical_outputs(stem: Path) -> dict[str, str]:
    absolute = stem.resolve()
    return {
        "canonical_artifact_output": str(Path(f"{absolute}.pt")),
        "canonical_report_output": str(Path(f"{absolute}.json")),
        "canonical_commit_output": str(Path(f"{absolute}.commit.json")),
    }


def _run_metadata(
    spec: collection.CollectionSpec,
    runtime_assets: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "seed": spec.seed,
        "replicate": spec.replicate,
        "num_envs": spec.num_envs,
        "collection_commit": spec.collection_commit,
        "preregistration_tag_commit": spec.preregistration_tag_commit,
        "assignment_mask_sha256": spec.assignment_mask_sha256,
        "source_manifest_sha256": artifact_contract.manifest_sha256(
            spec.source_sha256
        ),
        "checkpoint_manifest_sha256": artifact_contract.canonical_json_sha256(
            spec.checkpoint_sha256
        ),
        "runtime_asset_manifest_sha256": artifact_contract.manifest_sha256(
            runtime_assets
        ),
        **_canonical_outputs(spec.output_stem),
        "argument_receipt": dict(spec.argument_receipt),
    }


def _run_identity(
    spec: collection.CollectionSpec,
    runtime_assets: Mapping[str, str],
    parent_argv: Sequence[str],
) -> dict[str, Any]:
    canonical = transaction.canonical_paths(spec.output_stem)
    return {
        "collection_commit": spec.collection_commit,
        "parent_argv": list(parent_argv),
        "assignment_mask_sha256": spec.assignment_mask_sha256,
        "checkpoint_manifest_sha256": artifact_contract.canonical_json_sha256(
            spec.checkpoint_sha256
        ),
        "source_manifest_sha256": artifact_contract.manifest_sha256(
            spec.source_sha256
        ),
        "runtime_asset_manifest_sha256": artifact_contract.manifest_sha256(
            runtime_assets
        ),
        "canonical_paths": canonical.as_run_id_dict(),
        "seed": spec.seed,
        "replicate": spec.replicate,
        "num_envs": spec.num_envs,
    }


def _child_arguments(
    args: argparse.Namespace,
    identity: transaction.AttemptIdentity,
    lock_fd: int,
) -> list[str]:
    values: list[str] = [
        sys.executable,
        str((Path(__file__).resolve().parent / _CHILD_BASENAME).resolve()),
        "--v6_checkpoint",
        str(Path(args.v6_checkpoint).resolve()),
        "--search_checkpoint",
        str(Path(args.search_checkpoint).resolve()),
        "--fixed_direction",
        str(Path(args.fixed_direction).resolve()),
        "--seed",
        str(args.seed),
        "--replicate",
        str(args.replicate),
        "--num_envs",
        str(args.num_envs),
        "--window_steps",
        str(args.window_steps),
        "--token_scale",
        repr(args.token_scale),
        "--distal_scale",
        repr(args.distal_scale),
        "--raw_z_abs_cap",
        repr(args.raw_z_abs_cap),
        "--token_component_cap",
        repr(args.token_component_cap),
        "--distal_component_cap",
        repr(args.distal_component_cap),
        "--pre_tanh_l2_cap",
        repr(args.pre_tanh_l2_cap),
        "--verify_steps",
        str(args.verify_steps),
        "--output_stem",
        str(Path(args.output_stem).resolve()),
        "--device",
        str(args.device),
        f"--kit_args={args.kit_args}",
        "--_candidate42_run_id",
        identity.run_id,
        "--_candidate42_attempt_id",
        identity.attempt_id,
        "--_candidate42_attempt_number",
        str(identity.attempt_number),
        "--_candidate42_lock_fd",
        str(lock_fd),
    ]
    if args.headless:
        values.append("--headless")
    if args.enable_cameras:
        values.append("--enable_cameras")
    if args.livestream != -1:
        values.extend(("--livestream", str(args.livestream)))
    if args.experience:
        values.extend(("--experience", str(args.experience)))
    return values


def waitpid_exact(process: subprocess.Popen[bytes]) -> int:
    """Reap exactly once and preserve the kernel's raw wait status."""

    while True:
        try:
            pid, raw_status = os.waitpid(process.pid, 0)
            break
        except OSError as error:
            if error.errno != errno.EINTR:
                raise
    if pid != process.pid:
        raise RuntimeError("waitpid returned a different child pid")
    process.returncode = os.waitstatus_to_exitcode(raw_status)
    return raw_status


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _attempt_directory(stem: Path, number: int) -> Path:
    return Path(f"{stem.resolve()}.c42_txn") / f"attempt_{number:03d}"


def _attempt_numbers(stem: Path) -> list[int]:
    namespace = Path(f"{stem.resolve()}.c42_txn")
    if not namespace.exists():
        return []
    if namespace.is_symlink() or not namespace.is_dir():
        raise ValueError("transaction namespace is not a regular directory")
    found: list[int] = []
    for path in namespace.iterdir():
        if path.name == transaction.LOCK_NAME:
            if path.is_symlink() or not path.is_file():
                raise ValueError("transaction lock path changed type")
            continue
        match = _ATTEMPT.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected transaction namespace entry: {path}")
        if path.is_symlink() or not path.is_dir():
            raise ValueError("attempt path is not a regular directory")
        found.append(int(match.group(1)))
    found.sort()
    if found != list(range(1, len(found) + 1)):
        raise ValueError("attempt directories are not gap-free from 001")
    return found


def _load_json_regular(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"receipt is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not an object: {path}")
    return value


def _identity_from_intent(
    paths: transaction.TransactionPaths, attempt_number: int
) -> transaction.AttemptIdentity:
    stage = transaction.StageOnly.open(paths, attempt_number)
    value = transaction.load_receipt(stage, "00_intent.json").payload
    if not isinstance(value, Mapping):
        raise ValueError("intent did not expose a strict receipt")
    identity = value.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("intent omitted transaction identity")
    return transaction.AttemptIdentity(
        run_id=identity.get("run_id"),
        attempt_id=identity.get("attempt_id"),
        attempt_number=identity.get("attempt_number"),
    )


def _validate_committed_with_spec(
    paths: transaction.TransactionPaths, spec: collection.CollectionSpec
) -> Mapping[str, Any]:
    def validate_artifact(path: Path) -> dict[str, Any]:
        value = torch.load(path, map_location="cpu", weights_only=True)
        checked = artifact_contract.validate_artifact(value)
        metadata = checked["metadata"]
        if (
            metadata["source_sha256"] != dict(spec.source_sha256)
            or metadata["assignment_mask_sha256"]
            != spec.assignment_mask_sha256
            or metadata["git"]["commit"] != spec.collection_commit
        ):
            raise ValueError("canonical artifact differs from current authority")
        return checked

    def validate_report(report_path: Path, artifact_path: Path) -> None:
        report = _load_json_regular(report_path)
        artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=True
        )
        artifact_contract.validate_final_report(report, artifact)

    return transaction.validate_committed_namespace(
        paths,
        artifact_validator=validate_artifact,
        report_validator=validate_report,
    )


def _validate_verified_with_spec(
    paths: transaction.TransactionPaths,
    stage: transaction.StageOnly,
    identity: transaction.AttemptIdentity,
    spec: collection.CollectionSpec,
) -> Mapping[str, transaction.FileEvidence]:
    """Revalidate immutable attempt-local science before any canonical link.

    A durable 56 proves only that a previous parent validated the attempt.  A
    recovering parent independently validates every scientific payload and the
    current authority first; only then may it resume publication from that 56.
    """

    chain = transaction.validate_success_chain(stage, identity)
    intent = chain[transaction.INTENT]
    if (
        intent.payload is None
        or intent.payload.get("preregistration_tag_commit")
        != spec.preregistration_tag_commit
    ):
        raise ValueError(
            "verified intent differs from frozen preregistration tag authority"
        )
    payload = chain[transaction.STAGED_PAYLOAD]
    report_core_evidence = chain[transaction.STAGED_REPORT_CORE]
    final_report_evidence = chain[transaction.FINAL_REPORT]

    artifact = torch.load(payload.path, map_location="cpu", weights_only=True)
    checked_artifact = artifact_contract.validate_artifact(artifact)
    metadata = checked_artifact["metadata"]
    if (
        metadata["source_sha256"] != dict(spec.source_sha256)
        or metadata["assignment_mask_sha256"] != spec.assignment_mask_sha256
        or metadata["git"]["commit"] != spec.collection_commit
    ):
        raise ValueError("verified artifact differs from current authority")

    report_core = _load_json_regular(report_core_evidence.path)
    checked_core = artifact_contract.validate_report(report_core, checked_artifact)
    final_report = _load_json_regular(final_report_evidence.path)
    checked_final = artifact_contract.validate_final_report(
        final_report, checked_artifact
    )
    if checked_final["report_core"] != checked_core:
        raise ValueError("verified final report embeds a different report core")

    current_authority = _checked_post_exit_authority(spec)
    if checked_final["post_exit_authority"] != current_authority:
        raise ValueError("verified post-exit authority differs from current authority")
    if final_report_evidence.path.parent != stage.attempt_dir:
        raise ValueError("verified final report escaped its attempt")
    if paths != stage.paths:
        raise ValueError("verified stage belongs to another canonical namespace")
    return chain


def _durable_empty_recovery_log(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_recovery_logs(
    stage: transaction.StageOnly, *, create_missing: bool
) -> bool:
    """Fsync recovery logs; return whether a missing log had to be created."""

    created = False
    for name in (transaction.STDOUT_LOG, transaction.STDERR_LOG):
        log_path = stage.path(name)
        if not log_path.exists() and not log_path.is_symlink():
            if not create_missing:
                raise RuntimeError(
                    "spawn-successful recovery is missing a child-owned log"
                )
            _durable_empty_recovery_log(log_path)
            created = True
        if log_path.is_symlink() or not log_path.is_file():
            raise RuntimeError("recovery log is not a regular no-follow file")
        descriptor = os.open(log_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _fsync_directory(stage.attempt_dir)
    return created


def _prepare_attempt_number(
    paths: transaction.TransactionPaths,
    lock: transaction.NamespaceLock,
    stem: Path,
    *,
    expected_run_id: str,
    spec: collection.CollectionSpec,
) -> int | None:
    numbers = _attempt_numbers(stem)
    if not numbers:
        return 1
    latest = numbers[-1]
    directory = _attempt_directory(stem, latest)
    identity = _identity_from_intent(paths, latest)
    if identity.run_id != expected_run_id:
        raise ValueError("existing attempt belongs to a different parent run identity")
    failure = directory / "70_parent_failed.json"
    verified = directory / "56_verified.json"
    committed = directory / "60_committed.json"
    canonical_commit = Path(f"{stem.resolve()}.commit.json")
    if failure.exists():
        failure_evidence = transaction.load_receipt(
            transaction.StageOnly.open(paths, latest), "70_parent_failed.json"
        )
        if not isinstance(failure_evidence.payload, Mapping):
            raise ValueError("failure receipt omitted strict payload")
        transaction.repair_failure_link(paths, identity, lock)
        if failure_evidence.payload.get("retry_permitted") is not True:
            raise RuntimeError(
                "latest Candidate42 attempt is terminal and non-retryable"
            )
        sequence = transaction.validate_attempt_sequence(
            stem,
            expected_run_id=expected_run_id,
            repair_failure_links=True,
            lock=lock,
        )
        if sequence.final_state != "retryable_failure":
            raise RuntimeError("attempt sequence is not retryable")
        return latest + 1
    if verified.exists() and not failure.exists():
        stage = transaction.StageOnly.open(paths, latest)
        try:
            _validate_verified_with_spec(paths, stage, identity, spec)
            transaction.recover_verified_publication(paths, identity, lock)
            _validate_committed_with_spec(paths, spec)
            return None
        except BaseException as error:
            recovered_raw_status: int | None = None
            recovered_exit_observed = False
            try:
                recovered_exit = transaction.load_receipt(
                    stage, transaction.CHILD_EXIT
                )
                if recovered_exit.payload is not None:
                    recovered_raw_status = recovered_exit.payload[
                        "raw_wait_status"
                    ]
                    recovered_exit_observed = True
            except BaseException:
                pass
            transaction.record_parent_failure(
                paths,
                identity,
                lock,
                reason=(
                    "verified publication recovery failed: "
                    f"{type(error).__name__}: {error}"
                ),
                authority_failure=(
                    str(error) if isinstance(error, PostExitAuthorityError) else ""
                ),
                raw_wait_status=recovered_raw_status,
                spawn_succeeded=True,
                child_exit_observed=recovered_exit_observed,
                recovery_lock_acquired=True,
            )
            raise
    if committed.exists() or canonical_commit.exists():
        stage = transaction.StageOnly.open(paths, latest)
        recovered_raw_status = None
        recovered_exit_observed = False
        try:
            recovered_exit = transaction.load_receipt(stage, transaction.CHILD_EXIT)
            if recovered_exit.payload is not None:
                recovered_raw_status = recovered_exit.payload["raw_wait_status"]
                recovered_exit_observed = True
        except BaseException:
            pass
        transaction.record_parent_failure(
            paths,
            identity,
            lock,
            reason="commit evidence exists without its verified predecessor",
            raw_wait_status=recovered_raw_status,
            spawn_succeeded=True,
            child_exit_observed=recovered_exit_observed,
            recovery_lock_acquired=True,
        )
        raise RuntimeError("Candidate42 commit chain is missing 56_verified")
    if not failure.exists():
        stage = transaction.StageOnly.open(paths, latest)
        prespawn_failed = (
            stage.path("05_parent_pre_spawn_failed.json").exists()
            or stage.path("05_parent_pre_spawn_failed.json").is_symlink()
        )
        child_exit_exists = (
            stage.path("50_child_exit.json").exists()
            or stage.path("50_child_exit.json").is_symlink()
        )
        child_exit_raw_status: int | None = None
        if child_exit_exists:
            try:
                recovered_child_exit = transaction.load_receipt(
                    stage, transaction.CHILD_EXIT
                )
                if recovered_child_exit.payload is None:
                    raise ValueError("strict 50 omitted its payload")
                child_exit_raw_status = recovered_child_exit.payload[
                    "raw_wait_status"
                ]
            except BaseException as error:
                transaction.record_parent_failure(
                    paths,
                    identity,
                    lock,
                    reason=(
                        "existing child-exit evidence is invalid: "
                        f"{type(error).__name__}: {error}"
                    ),
                    authority_failure="malformed existing child-exit evidence",
                    raw_wait_status=None,
                    spawn_succeeded=not prespawn_failed,
                    child_exit_observed=False,
                    recovery_lock_acquired=True,
                )
                raise RuntimeError(
                    "invalid existing Candidate42 child-exit evidence"
                ) from error
        if not prespawn_failed and not child_exit_exists:
            # Acquiring namespace.lock proves every inherited child duplicate
            # has been released.  Fsync the child-owned logs, then record the
            # plan's nullable recovery wait status before classifying 70.
            child_evidence = any(
                stage.path(name).exists() or stage.path(name).is_symlink()
                for name in transaction.CHILD_STAGE_NAMES
            )
            try:
                _fsync_recovery_logs(stage, create_missing=not child_evidence)
            except BaseException as error:
                # Preserve the acquired-lock fact, but classify the evidence
                # gap as an authority failure so it can never authorize retry.
                # This produces an immutable terminal 70 whenever remaining
                # paths are regular enough to bind; malformed paths remain a
                # persistent fail-closed namespace conflict.
                repair_error: BaseException | None = None
                try:
                    _fsync_recovery_logs(stage, create_missing=True)
                except BaseException as secondary:
                    repair_error = secondary
                authority_failure = (
                    "unsafe recovery log evidence: "
                    f"{type(error).__name__}: {error}"
                )
                if repair_error is not None:
                    authority_failure += (
                        "; repair failed: "
                        f"{type(repair_error).__name__}: {repair_error}"
                    )
                try:
                    transaction.record_parent_failure(
                        paths,
                        identity,
                        lock,
                        reason=(
                            "unsafe unclassified recovery: "
                            f"{type(error).__name__}: {error}"
                        ),
                        authority_failure=authority_failure,
                        raw_wait_status=None,
                        spawn_succeeded=True,
                        child_exit_observed=False,
                        recovery_lock_acquired=True,
                    )
                except BaseException as failure_error:
                    raise RuntimeError(
                        "could not seal unsafe Candidate42 recovery"
                    ) from failure_error
                raise RuntimeError("unsafe unclassified Candidate42 recovery") from error
            _write_child_exit(stage, identity, None)
            child_exit_exists = True
        transaction.record_parent_failure(
            paths,
            identity,
            lock,
            reason="lock-acquired recovery classified an uncommitted attempt",
            raw_wait_status=child_exit_raw_status,
            spawn_succeeded=not prespawn_failed,
            child_exit_observed=child_exit_exists,
            recovery_lock_acquired=True,
        )
    newly_failed = transaction.load_receipt(
        transaction.StageOnly.open(paths, latest), "70_parent_failed.json"
    )
    if not isinstance(newly_failed.payload, Mapping):
        raise ValueError("failure receipt omitted strict payload")
    if newly_failed.payload.get("retry_permitted") is not True:
        raise RuntimeError("recovered attempt is terminal and non-retryable")
    sequence = transaction.validate_attempt_sequence(
        stem,
        expected_run_id=expected_run_id,
        repair_failure_links=True,
        lock=lock,
    )
    if sequence.final_state != "retryable_failure":
        raise RuntimeError("recovered attempt sequence is not retryable")
    return latest + 1


def _open_logs(directory: Path) -> tuple[Any, Any]:
    streams: list[Any] = []
    try:
        for name in ("stdout.log", "stderr.log"):
            descriptor = os.open(
                directory / name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            try:
                stream = os.fdopen(descriptor, "wb", buffering=0)
            except BaseException:
                # ``fdopen`` has not transferred ownership on failure.  Keep
                # the raw descriptor out of the child and out of later retry
                # attempts even when constructing the Python stream fails.
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
                raise
            streams.append(stream)
        _fsync_directory(directory)
        return streams[0], streams[1]
    except BaseException as error:
        original_traceback = error.__traceback__
        _cleanup_log_streams(streams, directory)
        raise error.with_traceback(original_traceback)


def _cleanup_log_streams(
    streams: Sequence[Any], directory: Path
) -> tuple[BaseException | None, Any]:
    first_error: BaseException | None = None
    first_traceback: Any = None

    def remember(error: BaseException) -> None:
        nonlocal first_error, first_traceback
        if first_error is None:
            first_error = error
            first_traceback = error.__traceback__

    for stream in streams:
        try:
            stream.flush()
        except BaseException as error:
            remember(error)
        try:
            os.fsync(stream.fileno())
        except BaseException as error:
            remember(error)
        try:
            stream.close()
        except BaseException as error:
            remember(error)
    try:
        _fsync_directory(directory)
    except BaseException as error:
        remember(error)
    return first_error, first_traceback


def _finish_logs(stdout: Any, stderr: Any, directory: Path) -> None:
    first_error, first_traceback = _cleanup_log_streams(
        (stdout, stderr), directory
    )
    if first_error is not None:
        raise first_error.with_traceback(first_traceback)


def spawn_child_exact(
    directory: Path,
    command: Sequence[str],
    inherited_fd: int,
    *,
    popen_factory: Any = subprocess.Popen,
) -> tuple[subprocess.Popen[bytes], Any, Any]:
    """Open bounded stage logs and close them on every spawn failure."""

    stdout, stderr = _open_logs(directory)
    try:
        process = popen_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            pass_fds=(inherited_fd,),
            close_fds=True,
        )
    except BaseException:
        _finish_logs(stdout, stderr, directory)
        raise
    return process, stdout, stderr


def _last_child_tip(
    stage: transaction.StageOnly,
) -> transaction.FileEvidence:
    for name in (
        "46_close_requested_failure.json",
        "45_child_failed.json",
        "40_close_requested.json",
        "19_close_requested_preboundary.json",
        "18_child_preboundary_failed.json",
        "32_prepared.json",
        "20_first_step_armed.json",
        "15_app_started.json",
        "10_child_started.json",
    ):
        if stage.path(name).exists() or stage.path(name).is_symlink():
            return transaction.file_evidence(stage, name)
    return transaction.file_evidence(stage, "00_intent.json")


def _write_child_exit(
    stage: transaction.StageOnly,
    identity: transaction.AttemptIdentity,
    raw_status: int | None,
) -> transaction.FileEvidence:
    stdout = transaction.file_evidence(stage, "stdout.log")
    stderr = transaction.file_evidence(stage, "stderr.log")
    wait_status = transaction.decode_wait_status(raw_status)
    return transaction.write_receipt(
        stage,
        "50_child_exit.json",
        identity,
        {
            "status": "observed",
            **wait_status,
            "stdout": stdout.as_dict(),
            "stderr": stderr.as_dict(),
        },
        predecessors=(_last_child_tip(stage),),
    )


def _submodule_commits(root: Path) -> dict[str, str]:
    completed = subprocess.run(
        ("git", "submodule", "status", "--recursive"),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line or line[0] != " ":
            raise ValueError("submodule status is not clean and registered")
        parts = line[1:].split()
        if len(parts) < 2:
            raise ValueError("malformed submodule status")
        result[parts[1]] = parts[0]
    return result


def _post_exit_authority(
    spec: collection.CollectionSpec,
) -> dict[str, Any]:
    collection.validate_post_application_authority(spec)
    source = collection.source_fingerprints(spec.repository_root)
    runtime_assets = collection.runtime_asset_fingerprints(spec.repository_root)
    git_paths = tuple(
        sorted(
            set(source)
            | {path for path in runtime_assets if not Path(path).is_absolute()}
        )
    )
    return {
        "status": "passed",
        "collection_commit": spec.collection_commit,
        "implementation_commit": spec.implementation_commit,
        "preregistration_tag_commit": collection._git_resolve(
            spec.repository_root, collection.PREREGISTRATION_TAG
        ),
        "collection_tag_commit": collection._git_resolve(
            spec.repository_root, collection.COLLECTION_SEAL_TAG
        ),
        "superproject_clean": True,
        "submodules_clean": True,
        "git": collection.git_provenance(spec.repository_root, git_paths),
        "source_sha256": source,
        "checkpoint_sha256": collection._current_hashes(spec),
        "runtime_asset_sha256": runtime_assets,
        "submodule_commit_sha256": _submodule_commits(spec.repository_root),
    }


def _checked_post_exit_authority(
    spec: collection.CollectionSpec,
) -> dict[str, Any]:
    try:
        return _post_exit_authority(spec)
    except BaseException as error:
        raise PostExitAuthorityError(
            f"{type(error).__name__}: {error}"
        ) from error


def _validate_staged(
    stage: transaction.StageOnly,
    spec: collection.CollectionSpec,
) -> tuple[dict[str, Any], dict[str, Any], transaction.FileEvidence]:
    for name in (
        "10_child_started.json",
        "15_app_started.json",
        "20_first_step_armed.json",
        "32_prepared.json",
        "40_close_requested.json",
    ):
        transaction.load_receipt(stage, name)
    for forbidden in (
        "18_child_preboundary_failed.json",
        "19_close_requested_preboundary.json",
        "45_child_failed.json",
        "46_close_requested_failure.json",
    ):
        if stage.path(forbidden).exists() or stage.path(forbidden).is_symlink():
            transaction.file_evidence(stage, forbidden)
            raise RuntimeError(f"child terminal conflict: {forbidden}")
    payload_path = stage.path("30_payload.pt")
    report_path = stage.path("31_report_core.json")
    artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    report = _load_json_regular(report_path)
    checked = artifact_contract.validate_artifact(artifact)
    checked_report = artifact_contract.validate_report(report, checked)
    metadata = checked["metadata"]
    if (
        metadata["assignment_mask_sha256"] != spec.assignment_mask_sha256
        or metadata["source_sha256"] != dict(spec.source_sha256)
        or metadata["git"]["commit"] != spec.collection_commit
    ):
        raise ValueError("staged scientific authority differs from parent intent")
    return checked, checked_report, transaction.file_evidence(stage, "30_payload.pt")


def _verify_and_publish(
    *,
    paths: transaction.TransactionPaths,
    stage: transaction.StageOnly,
    identity: transaction.AttemptIdentity,
    lock: transaction.NamespaceLock,
    spec: collection.CollectionSpec,
    child_exit: transaction.FileEvidence,
) -> Mapping[str, Any]:
    artifact, report_core, payload = _validate_staged(stage, spec)
    authority = _checked_post_exit_authority(spec)
    final_report = artifact_contract.build_final_report(
        report_core,
        artifact,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        attempt_number=identity.attempt_number,
        child_exit_sha256=child_exit.sha256,
        child_exit_size=child_exit.size,
        post_exit_authority=authority,
        artifact_sha256=payload.sha256,
        artifact_size=payload.size,
        artifact_output=_canonical_outputs(spec.output_stem)[
            "canonical_artifact_output"
        ],
    )
    final_bytes = artifact_contract.final_report_bytes(final_report, artifact)
    final = transaction.durable_stage_bytes(
        stage, "55_final_report.json", final_bytes
    )
    transaction.write_receipt(
        stage,
        "56_verified.json",
        identity,
        {"status": "verified"},
        predecessors=(final,),
    )
    result = transaction.publish_verified_attempt(paths, identity, lock)
    _validate_committed_with_spec(paths, spec)
    return result


def run_parent(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    parent_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(parent_argv)
    stem = Path(args.output_stem).resolve()
    paths = transaction.TransactionPaths.from_output_stem(stem)
    lock = transaction.NamespaceLock.acquire(paths)
    try:
        spec = collection.build_spec(
            args, allow_current_canonical_evidence=True
        )
        runtime_assets = collection.runtime_asset_fingerprints(spec.repository_root)
        run_metadata = _run_metadata(spec, runtime_assets)
        run_identity = _run_identity(spec, runtime_assets, parent_argv)
        run_id = transaction.compute_run_id(run_identity)
        attempt_number = _prepare_attempt_number(
            paths, lock, stem, expected_run_id=run_id, spec=spec
        )
        if attempt_number is None:
            return _validate_committed_with_spec(paths, spec)
        identity = transaction.AttemptIdentity(
            run_id=run_id,
            attempt_id=transaction.compute_attempt_id(run_id, attempt_number),
            attempt_number=attempt_number,
        )
        stage = transaction.StageOnly.create(paths, attempt_number)
        intent = transaction.write_receipt(
            stage,
            "00_intent.json",
            identity,
            {"status": "prepared", "run_metadata": run_metadata, **run_metadata},
        )
        inherited_fd: int | None = None
        stdout = stderr = None
        process: subprocess.Popen[bytes] | None = None
        try:
            inherited_fd = lock.duplicate_for_child()
            child_command = _child_arguments(args, identity, inherited_fd)
            process, stdout, stderr = spawn_child_exact(
                _attempt_directory(stem, attempt_number),
                child_command,
                inherited_fd,
            )
        except BaseException as error:
            transaction.write_receipt(
                stage,
                "05_parent_pre_spawn_failed.json",
                identity,
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                predecessors=(intent,),
            )
            transaction.record_parent_failure(
                paths,
                identity,
                lock,
                reason=f"pre-spawn failure: {type(error).__name__}: {error}",
                raw_wait_status=None,
                spawn_succeeded=False,
                child_exit_observed=False,
            )
            raise
        finally:
            if inherited_fd is not None:
                os.close(inherited_fd)
        if process is None or stdout is None or stderr is None:
            raise AssertionError("spawn bookkeeping is incomplete")
        raw_status: int | None = None
        logs_finished = False
        try:
            raw_status = waitpid_exact(process)
            _finish_logs(stdout, stderr, _attempt_directory(stem, attempt_number))
            logs_finished = True
            child_exit = _write_child_exit(stage, identity, raw_status)
            if not os.WIFEXITED(raw_status) or os.WEXITSTATUS(raw_status) != 0:
                raise RuntimeError(
                    "child did not exit normally with exact status zero"
                )
            return _verify_and_publish(
                paths=paths,
                stage=stage,
                identity=identity,
                lock=lock,
                spec=spec,
                child_exit=child_exit,
            )
        except BaseException as error:
            if raw_status is None:
                try:
                    raw_status = waitpid_exact(process)
                except BaseException:
                    # Unknown liveness is terminal and non-retryable.  The
                    # inherited descriptor still excludes another parent if
                    # the child remains alive after this parent releases.
                    raw_status = None
            if not logs_finished:
                try:
                    _finish_logs(
                        stdout, stderr, _attempt_directory(stem, attempt_number)
                    )
                    logs_finished = True
                except BaseException:
                    pass
            transaction.record_parent_failure(
                paths,
                identity,
                lock,
                reason=f"post-exit validation/publish failure: {type(error).__name__}: {error}",
                authority_failure=(
                    str(error) if isinstance(error, PostExitAuthorityError) else ""
                ),
                raw_wait_status=raw_status,
                spawn_succeeded=True,
                child_exit_observed=raw_status is not None,
            )
            raise
    finally:
        lock.close()


def main() -> None:
    result = run_parent()
    print(
        "[candidate42-transaction] "
        f"status={result.get('status', 'committed')} "
        f"commit_sha256={result.get('commit_sha256', '')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
