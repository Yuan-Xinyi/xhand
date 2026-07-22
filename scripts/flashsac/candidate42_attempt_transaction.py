#!/usr/bin/env python3
"""Durable parent/child evidence transaction for Candidate42.

This module is deliberately simulation-free.  It owns the filesystem and
process-boundary contract around a rollout, while the scientific artifact
validator remains an injected callback.  Every visible state transition is
no-clobber, hash-addressed, fsynced, and fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Callable, Iterable, Mapping


TRANSACTION_CONTRACT = "candidate42_parent_child_fsync_no_clobber_commit_v1"
NAMESPACE_SUFFIX = ".c42_txn"
LOCK_NAME = "namespace.lock"

INTENT = "00_intent.json"
PARENT_PRESPAWN_FAILED = "05_parent_pre_spawn_failed.json"
CHILD_STARTED = "10_child_started.json"
APP_STARTED = "15_app_started.json"
CHILD_PREBOUNDARY_FAILED = "18_child_preboundary_failed.json"
CLOSE_PREBOUNDARY = "19_close_requested_preboundary.json"
FIRST_STEP_ARMED = "20_first_step_armed.json"
STAGED_PAYLOAD = "30_payload.pt"
STAGED_REPORT_CORE = "31_report_core.json"
PREPARED = "32_prepared.json"
CLOSE_SUCCESS = "40_close_requested.json"
CHILD_FAILED = "45_child_failed.json"
CLOSE_FAILURE = "46_close_requested_failure.json"
CHILD_EXIT = "50_child_exit.json"
FINAL_REPORT = "55_final_report.json"
VERIFIED = "56_verified.json"
COMMITTED = "60_committed.json"
PARENT_FAILED = "70_parent_failed.json"
STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"

RECEIPT_NAMES = frozenset(
    {
        INTENT,
        PARENT_PRESPAWN_FAILED,
        CHILD_STARTED,
        APP_STARTED,
        CHILD_PREBOUNDARY_FAILED,
        CLOSE_PREBOUNDARY,
        FIRST_STEP_ARMED,
        PREPARED,
        CLOSE_SUCCESS,
        CHILD_FAILED,
        CLOSE_FAILURE,
        CHILD_EXIT,
        FINAL_REPORT,
        VERIFIED,
        COMMITTED,
        PARENT_FAILED,
    }
)
STAGE_NAMES = frozenset(
    set(RECEIPT_NAMES)
    | {STAGED_PAYLOAD, STAGED_REPORT_CORE, STDOUT_LOG, STDERR_LOG}
)
# Parent recovery uses this registered set to distinguish a child that never
# entered its bootstrap from one whose evidence/log authority is incomplete.
CHILD_STAGE_NAMES = frozenset(
    {
        CHILD_STARTED,
        APP_STARTED,
        CHILD_PREBOUNDARY_FAILED,
        CLOSE_PREBOUNDARY,
        FIRST_STEP_ARMED,
        STAGED_PAYLOAD,
        STAGED_REPORT_CORE,
        PREPARED,
        CLOSE_SUCCESS,
        CHILD_FAILED,
        CLOSE_FAILURE,
    }
)
IDENTITY_FIELDS = frozenset({"run_id", "attempt_id", "attempt_number"})
RECORD_FIELDS = frozenset({"sha256", "size"})
RECEIPT_HEADER_FIELDS = frozenset(
    {"transaction_contract", "receipt", "identity", "predecessors"}
)
FINAL_REPORT_FIELDS = frozenset(
    {
        "transaction_contract",
        "receipt",
        "identity",
        "predecessors",
        "kind",
        "version",
        "status",
        "run_id",
        "attempt_id",
        "attempt_number",
        "child_exit_sha256",
        "post_exit_authority",
        "source_manifest_sha256",
        "checkpoint_manifest_sha256",
        "runtime_asset_manifest_sha256",
        "artifact_sha256",
        "artifact_size",
        "artifact_output",
        "report_core",
        "report_core_sha256",
    }
)
FINAL_REPORT_KIND = "pick_tool_candidate42_transactional_public_arm_ramp_final_report_v1"
RUN_ID_FIELDS = frozenset(
    {
        "collection_commit",
        "parent_argv",
        "assignment_mask_sha256",
        "checkpoint_manifest_sha256",
        "source_manifest_sha256",
        "runtime_asset_manifest_sha256",
        "canonical_paths",
        "seed",
        "replicate",
        "num_envs",
    }
)
_RUN_IDENTITY_CACHE: dict[str, dict[str, Any]] = {}
CANONICAL_PATH_FIELDS = frozenset({"artifact", "report", "commit"})
ATTEMPT_PATTERN = re.compile(r"^attempt_([0-9]{3})$")
TEMP_PATTERN = re.compile(r"^\..+\.tmp-[0-9]+-[0-9a-f]+$")

POST_BOUNDARY_NAMES = frozenset(
    {
        FIRST_STEP_ARMED,
        STAGED_PAYLOAD,
        STAGED_REPORT_CORE,
        PREPARED,
        CLOSE_SUCCESS,
        CHILD_FAILED,
        CLOSE_FAILURE,
        FINAL_REPORT,
        VERIFIED,
        COMMITTED,
    }
)
SUCCESS_ONLY_NAMES = frozenset(
    {STAGED_PAYLOAD, STAGED_REPORT_CORE, PREPARED, CLOSE_SUCCESS, FINAL_REPORT, VERIFIED, COMMITTED}
)
FAILURE_MARKER_NAMES = frozenset(
    {
        PARENT_PRESPAWN_FAILED,
        CHILD_PREBOUNDARY_FAILED,
        CLOSE_PREBOUNDARY,
        CHILD_FAILED,
        CLOSE_FAILURE,
        PARENT_FAILED,
    }
)


class TransactionError(RuntimeError):
    """Base class for a fail-closed transaction error."""


class PathSafetyError(TransactionError):
    """A path escaped its sealed capability or changed type/inode."""


class ReceiptError(TransactionError):
    """A receipt is malformed, corrupt, or disagrees with its predecessors."""


class StateConflictError(TransactionError):
    """The namespace contains mutually exclusive or foreign evidence."""


class RecoveryRequired(TransactionError):
    """A verified partial publication must be resumed without a child rerun."""


@dataclass(frozen=True)
class TransactionIdentity:
    run_id: str
    attempt_id: str
    attempt_number: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
        }


@dataclass(frozen=True)
class CanonicalPaths:
    output_stem: Path
    artifact: Path
    report: Path
    commit: Path
    namespace: Path
    lock: Path

    @classmethod
    def from_output_stem(
        cls, output_stem: str | os.PathLike[str]
    ) -> "CanonicalPaths":
        return canonical_paths(output_stem)

    @property
    def lock_path(self) -> Path:
        return self.lock

    @property
    def namespace_path(self) -> Path:
        return self.namespace

    @property
    def artifact_path(self) -> Path:
        return self.artifact

    @property
    def report_path(self) -> Path:
        return self.report

    @property
    def commit_path(self) -> Path:
        return self.commit

    def attempt_path(self, attempt_number: int) -> Path:
        if type(attempt_number) is not int or not 1 <= attempt_number <= 999:
            raise ValueError("attempt_number must be in [1, 999]")
        return self.namespace / f"attempt_{attempt_number:03d}"

    def as_run_id_dict(self) -> dict[str, str]:
        return {
            "artifact": str(self.artifact),
            "report": str(self.report),
            "commit": str(self.commit),
        }


@dataclass(frozen=True)
class AttemptContext:
    canonical: CanonicalPaths
    attempt_dir: Path
    identity: TransactionIdentity
    run_identity: Mapping[str, Any]

    @property
    def failure_receipt(self) -> Path:
        return Path(
            f"{self.canonical.output_stem}.failed_attempt_"
            f"{self.identity.attempt_number:03d}.json"
        )

    def path(self, name: str) -> Path:
        _validate_stage_name(name)
        return self.attempt_dir / name

    @property
    def run_metadata(self) -> Mapping[str, Any]:
        value = _load_json_file(self.attempt_dir / INTENT, label=INTENT)
        metadata = value.get("run_metadata")
        return _validate_run_metadata(metadata)


@dataclass(frozen=True)
class AttemptClassification:
    boundary_armed: bool
    retry_permitted: bool
    reason: str
    observed: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class FileEvidence:
    name: str
    path: Path
    sha256: str
    size: int
    payload: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class SequenceValidation:
    attempts: tuple[AttemptContext, ...]
    final_state: str
    recovery_attempt: AttemptContext | None


# Public campaign names retained by the sealed parent/child implementation.
AttemptIdentity = TransactionIdentity
TransactionPaths = CanonicalPaths


def _strict_json(value: Any, name: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float")
        return value
    if isinstance(value, list):
        return [_strict_json(item, f"{name}[]") for item in value]
    if isinstance(value, tuple):
        raise TypeError(f"{name} must use JSON lists, not tuples")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{name} keys must be non-empty strings")
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = _strict_json(item, f"{name}.{key}")
        return result
    raise TypeError(f"{name} is not strict JSON: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    checked = _strict_json(value)
    return (
        json.dumps(
            checked,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    checked = _strict_json(value)
    return (
        json.dumps(checked, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_git_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase Git SHA")
    return value


def _same_lstat(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, name) == getattr(right, name) for name in fields)


def _hash_regular_path(path: Path, *, label: str) -> tuple[str, os.stat_result]:
    """Hash a stable regular inode without ever following a symlink."""

    before = _require_regular_file(path, label=label)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_lstat(before, opened):
            raise PathSafetyError(f"{label} changed before open: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if not _same_lstat(opened, after):
            raise PathSafetyError(f"{label} changed while hashing: {path}")
        try:
            named = path.lstat()
        except FileNotFoundError as error:
            raise PathSafetyError(f"{label} disappeared while hashing: {path}") from error
        if not _same_lstat(after, named):
            raise PathSafetyError(f"{label} path changed while hashing: {path}")
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest, _info = _hash_regular_path(Path(path), label="hash input")
    return digest


def _path_record(path: Path) -> dict[str, Any]:
    digest, info = _hash_regular_path(path, label="evidence")
    return {"sha256": digest, "size": int(info.st_size)}


def validate_record(value: Any, name: str = "record") -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(RECORD_FIELDS):
        raise ReceiptError(f"{name} fields must be exactly {sorted(RECORD_FIELDS)}")
    result = _strict_json(value, name)
    _require_sha(result["sha256"], f"{name}.sha256")
    if type(result["size"]) is not int or result["size"] < 0:
        raise ReceiptError(f"{name}.size must be a non-negative integer")
    return result


def validate_file_record(path: Path, value: Any, name: str = "record") -> dict[str, Any]:
    record = validate_record(value, name)
    actual = _path_record(path)
    if actual != record:
        raise ReceiptError(f"{name} does not match {path}")
    return record


def _normalize_output_stem(output_stem: str | os.PathLike[str]) -> Path:
    raw = Path(output_stem)
    if raw.name in {"", ".", ".."}:
        raise PathSafetyError("output_stem must name one registered trial stem")
    if raw.suffix in {".pt", ".json"} or raw.name.endswith(NAMESPACE_SUFFIX):
        raise PathSafetyError("output_stem must exclude artifact/receipt suffixes")
    return raw.expanduser().absolute().resolve(strict=False)


def canonical_paths(output_stem: str | os.PathLike[str]) -> CanonicalPaths:
    stem = _normalize_output_stem(output_stem)
    namespace = Path(f"{stem}{NAMESPACE_SUFFIX}")
    return CanonicalPaths(
        output_stem=stem,
        artifact=Path(f"{stem}.pt"),
        report=Path(f"{stem}.json"),
        commit=Path(f"{stem}.commit.json"),
        namespace=namespace,
        lock=namespace / LOCK_NAME,
    )


def _require_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise PathSafetyError(f"missing {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PathSafetyError(f"{label} is not a non-symlink directory: {path}")
    return info


def _require_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise PathSafetyError(f"missing {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PathSafetyError(f"{label} is not a non-symlink regular file: {path}")
    return info


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_mkdir(path: Path, *, parents: bool = False) -> None:
    if path.exists() or path.is_symlink():
        _require_directory(path, label="transaction directory")
        return
    if parents:
        path.mkdir(parents=True, exist_ok=False)
    else:
        path.mkdir()
    _require_directory(path, label="transaction directory")
    _fsync_directory(path.parent)


def _validate_stage_name(name: str) -> None:
    if not isinstance(name, str) or name not in STAGE_NAMES:
        raise PathSafetyError(f"unregistered stage name: {name!r}")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise PathSafetyError(f"stage name escaped its capability: {name!r}")


def _identity(value: Any) -> TransactionIdentity:
    if not isinstance(value, Mapping) or set(value) != set(IDENTITY_FIELDS):
        raise ReceiptError("transaction identity fields changed")
    checked = _strict_json(value, "identity")
    run_id = _require_sha(checked["run_id"], "identity.run_id")
    attempt_id = _require_sha(checked["attempt_id"], "identity.attempt_id")
    number = checked["attempt_number"]
    if type(number) is not int or not 1 <= number <= 999:
        raise ReceiptError("identity.attempt_number must be in [1, 999]")
    expected = compute_attempt_id(run_id, number)
    if attempt_id != expected:
        raise ReceiptError("attempt_id differs from run_id/attempt_number")
    return TransactionIdentity(run_id, attempt_id, number)


def validate_run_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(RUN_ID_FIELDS):
        raise ValueError(f"run identity fields must be exactly {sorted(RUN_ID_FIELDS)}")
    result = _strict_json(value, "run_identity")
    _require_git_sha(result["collection_commit"], "collection_commit")
    for name in (
        "assignment_mask_sha256",
        "checkpoint_manifest_sha256",
        "source_manifest_sha256",
        "runtime_asset_manifest_sha256",
    ):
        _require_sha(result[name], name)
    argv = result["parent_argv"]
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item for item in argv
    ):
        raise ValueError("parent_argv must be a non-empty JSON string list")
    paths = result["canonical_paths"]
    if not isinstance(paths, Mapping) or set(paths) != set(CANONICAL_PATH_FIELDS):
        raise ValueError("canonical_paths fields changed")
    for name, path in paths.items():
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"canonical_paths.{name} must be absolute")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if result["replicate"] not in {"a", "b"}:
        raise ValueError("replicate must be a or b")
    if type(result["num_envs"]) is not int or result["num_envs"] not in {8, 64}:
        raise ValueError("num_envs must be exactly 8 or 64")
    return result


def compute_run_id(run_identity: Mapping[str, Any]) -> str:
    checked = validate_run_identity(run_identity)
    digest = canonical_json_sha256(checked)
    # The parent calls compute_run_id immediately before sealing 00.  Cache the
    # exact inputs so the compatibility write API can persist them in 00 even
    # though the pre-existing caller passes only the digest thereafter.
    _RUN_IDENTITY_CACHE[digest] = checked
    return digest


def compute_attempt_id(run_id: str, attempt_number: int) -> str:
    _require_sha(run_id, "run_id")
    if type(attempt_number) is not int or not 1 <= attempt_number <= 999:
        raise ValueError("attempt_number must be in [1, 999]")
    return canonical_json_sha256(
        {"attempt_number": attempt_number, "run_id": run_id}
    )


class NamespaceLock:
    """Exclusive namespace flock retained across classification and commit."""

    def __init__(
        self,
        output_stem: str | os.PathLike[str] | CanonicalPaths,
        *,
        blocking: bool = True,
    ):
        self.canonical = (
            output_stem
            if isinstance(output_stem, CanonicalPaths)
            else canonical_paths(output_stem)
        )
        self.blocking = bool(blocking)
        self.fd: int | None = None

    @classmethod
    def acquire(
        cls, paths: CanonicalPaths, *, blocking: bool = True
    ) -> "NamespaceLock":
        if not isinstance(paths, CanonicalPaths):
            raise TypeError("paths must be TransactionPaths")
        return cls(paths, blocking=blocking)._acquire()

    def _acquire(self) -> "NamespaceLock":
        if self.fd is not None:
            raise TransactionError("namespace lock is already acquired")
        _safe_mkdir(self.canonical.namespace, parents=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.canonical.lock, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise PathSafetyError("namespace lock descriptor is not regular")
            path_info = _require_regular_file(
                self.canonical.lock, label="namespace lock"
            )
            if (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino):
                raise PathSafetyError("namespace lock path changed inode")
            operation = fcntl.LOCK_EX
            if not self.blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(descriptor, operation)
            except BlockingIOError as error:
                raise TransactionError("namespace is locked by another parent") from error
            self.fd = descriptor
            _fsync_directory(self.canonical.namespace)
            return self
        except BaseException:
            os.close(descriptor)
            raise

    def assert_held(self) -> int:
        if self.fd is None:
            raise TransactionError("namespace lock is not held")
        info = os.fstat(self.fd)
        path_info = _require_regular_file(self.canonical.lock, label="namespace lock")
        if (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino):
            raise PathSafetyError("held namespace lock path changed inode")
        return self.fd

    def duplicate_for_child(self) -> int:
        descriptor = os.dup(self.assert_held())
        os.set_inheritable(descriptor, True)
        return descriptor

    def release(self) -> None:
        if self.fd is None:
            return
        descriptor = self.fd
        self.fd = None
        # Never issue LOCK_UN here.  The child inherits a dup of this same
        # open-file description; an explicit unlock would release its liveness
        # exclusion too.  Closing only the parent's descriptor keeps the flock
        # held until the final inherited duplicate is closed.
        os.close(descriptor)

    def close(self) -> None:
        self.release()

    def __enter__(self) -> "NamespaceLock":
        # ``NamespaceLock.acquire(paths)`` is intentionally usable both as a
        # plain acquired handle and directly in a ``with`` statement.  In the
        # latter form the classmethod has already acquired the flock before
        # Python invokes ``__enter__``.
        if self.fd is not None:
            self.assert_held()
            return self
        return self._acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def validate_inherited_lock_fd(
    descriptor: int, expected_lock_path: str | os.PathLike[str]
) -> None:
    """Validate an inherited locked file description, then set FD_CLOEXEC."""

    if type(descriptor) is not int or descriptor < 0:
        raise TypeError("lock descriptor must be a non-negative integer")
    path = Path(expected_lock_path).absolute().resolve(strict=False)
    info = os.fstat(descriptor)
    path_info = _require_regular_file(path, label="inherited namespace lock")
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (
        path_info.st_dev,
        path_info.st_ino,
    ):
        raise PathSafetyError("inherited descriptor does not match namespace.lock")
    probe = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise TransactionError("inherited namespace descriptor is not locked")
    finally:
        os.close(probe)
    # The blocked probe above proves that *some* open-file description owns
    # the namespace flock.  Acquiring through the supplied descriptor
    # distinguishes a legitimate dup of that same description (success) from
    # an independently opened spoof descriptor (would block).  Never unlock:
    # this operation deliberately preserves the inherited liveness lock.
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise TransactionError(
            "inherited descriptor is not the locked open-file description"
        ) from error
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
    fcntl.fcntl(descriptor, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
    if not (fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC):
        raise TransactionError("failed to set FD_CLOEXEC on inherited lock")


class StageOnly:
    """Campaign-facing stage-only capability without canonical write methods."""

    def __init__(self, paths: CanonicalPaths, attempt_number: int) -> None:
        if not isinstance(paths, CanonicalPaths):
            raise TypeError("paths must be TransactionPaths")
        if type(attempt_number) is not int or not 1 <= attempt_number <= 999:
            raise ValueError("attempt_number must be in [1, 999]")
        self.paths = paths
        self.attempt_number = attempt_number
        self.attempt_dir = paths.attempt_path(attempt_number)
        _require_directory(paths.namespace, label="transaction namespace")
        _require_directory(self.attempt_dir, label="attempt directory")
        if self.attempt_dir.parent != paths.namespace:
            raise PathSafetyError("attempt escaped transaction namespace")

    @classmethod
    def create(cls, paths: CanonicalPaths, attempt_number: int) -> "StageOnly":
        if not isinstance(paths, CanonicalPaths):
            raise TypeError("paths must be TransactionPaths")
        _require_directory(paths.namespace, label="transaction namespace")
        target = paths.attempt_path(attempt_number)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"attempt directory already exists: {target}")
        target.mkdir()
        _require_directory(target, label="attempt directory")
        _fsync_directory(paths.namespace)
        return cls(paths, attempt_number)

    @classmethod
    def open(cls, paths: CanonicalPaths, attempt_number: int) -> "StageOnly":
        return cls(paths, attempt_number)

    def path(self, name: str) -> Path:
        _validate_stage_name(name)
        _require_directory(self.attempt_dir, label="attempt directory")
        target = self.attempt_dir / name
        if target.parent != self.attempt_dir:
            raise PathSafetyError("stage target escaped attempt directory")
        return target

    def write_bytes(self, name: str, payload: bytes) -> FileEvidence:
        if not isinstance(payload, bytes):
            raise TypeError("durable payload must be bytes")
        target = self.path(name)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"no-clobber stage target exists: {target}")
        temporary = self.attempt_dir / (
            f".{name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        linked = False
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            _require_regular_file(temporary, label="durable temporary")
            os.link(temporary, target, follow_symlinks=False)
            linked = True
            _fsync_directory(self.attempt_dir)
            evidence = file_evidence(self, name)
            temporary.unlink()
            _fsync_directory(self.attempt_dir)
            return evidence
        except BaseException:
            if temporary.exists() or temporary.is_symlink():
                try:
                    temporary.unlink()
                    _fsync_directory(self.attempt_dir)
                except OSError:
                    pass
            # Never remove a linked final name: it is permanent evidence even
            # if a later fsync or validation operation failed.
            if linked:
                pass
            raise


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    info = _require_regular_file(path, label=label)
    if info.st_size <= 0 or info.st_size > 16 * 1024 * 1024:
        raise ReceiptError(f"{label} size is outside the registered bound")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_json_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReceiptError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cannot decode {label}: {path}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"{label} must be a JSON object")
    checked = _strict_json(value, label)
    if pretty_json_bytes(checked) != raw:
        raise ReceiptError(f"{label} is not in the exact durable JSON encoding")
    return checked


def _predecessors(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ReceiptError("predecessors must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for name, record in value.items():
        _validate_stage_name(name)
        result[name] = validate_record(record, f"predecessors[{name!r}]")
    return result


def _receipt_prefix(name: str) -> int:
    if name in {STDOUT_LOG, STDERR_LOG}:
        return -1
    try:
        return int(name.split("_", 1)[0])
    except (ValueError, IndexError) as error:
        raise ReceiptError(f"receipt has no numeric prefix: {name}") from error


def _validate_predecessor_shape(
    receipt: str,
    predecessors: Mapping[str, Mapping[str, Any]],
) -> None:
    names = set(predecessors)
    exact: dict[str, set[str]] = {
        INTENT: set(),
        PARENT_PRESPAWN_FAILED: {INTENT},
        CHILD_STARTED: {INTENT},
        APP_STARTED: {CHILD_STARTED},
        CLOSE_PREBOUNDARY: {CHILD_PREBOUNDARY_FAILED},
        FIRST_STEP_ARMED: {APP_STARTED},
        PREPARED: {FIRST_STEP_ARMED, STAGED_PAYLOAD, STAGED_REPORT_CORE},
        CLOSE_SUCCESS: {PREPARED},
        CHILD_FAILED: {FIRST_STEP_ARMED},
        FINAL_REPORT: {CHILD_EXIT},
        VERIFIED: {FINAL_REPORT},
        COMMITTED: {VERIFIED},
    }
    if receipt in exact and names != exact[receipt]:
        raise ReceiptError(
            f"{receipt} predecessors changed: {sorted(names)} != "
            f"{sorted(exact[receipt])}"
        )
    if receipt == CHILD_PREBOUNDARY_FAILED and names not in (
        {CHILD_STARTED},
        {APP_STARTED},
    ):
        raise ReceiptError("18 must extend exactly 10 or 15")
    if receipt == CLOSE_FAILURE and names not in ({CHILD_FAILED}, {PREPARED}):
        raise ReceiptError("46 must extend exactly 45 or conflicting 32")
    if receipt == CHILD_EXIT:
        if not {STDOUT_LOG, STDERR_LOG}.issubset(names) or len(names) != 3:
            raise ReceiptError("50 must bind stdout, stderr, and one terminal tip")
        tip = next(iter(names - {STDOUT_LOG, STDERR_LOG}))
        if tip not in {
            INTENT,
            PARENT_PRESPAWN_FAILED,
            CHILD_STARTED,
            APP_STARTED,
            CHILD_PREBOUNDARY_FAILED,
            CLOSE_PREBOUNDARY,
            FIRST_STEP_ARMED,
            PREPARED,
            CLOSE_SUCCESS,
            CHILD_FAILED,
            CLOSE_FAILURE,
        }:
            raise ReceiptError(f"50 has an invalid terminal tip: {tip}")
    if receipt not in exact and receipt not in {
        CHILD_PREBOUNDARY_FAILED,
        CLOSE_FAILURE,
        CHILD_EXIT,
        PARENT_FAILED,
    }:
        raise ReceiptError(f"no predecessor contract registered for {receipt}")
    current = _receipt_prefix(receipt)
    for name in names:
        if name == receipt:
            raise ReceiptError("a receipt cannot bind itself")
        predecessor = _receipt_prefix(name)
        if predecessor >= current and predecessor >= 0:
            raise ReceiptError(f"{receipt} predicts future evidence {name}")


def _wait_status(value: Any, *, allow_null_raw: bool) -> dict[str, Any]:
    fields = {"raw_wait_status", "wifexited", "exit_code", "signal"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReceiptError("wait_status fields changed")
    result = _strict_json(value, "wait_status")
    raw = result["raw_wait_status"]
    if raw is None:
        if not allow_null_raw or any(
            result[name] is not None for name in ("wifexited", "exit_code", "signal")
        ):
            raise ReceiptError("null recovery wait status is inconsistent")
        return result
    if type(raw) is not int or raw < 0:
        raise ReceiptError("raw_wait_status must be a non-negative integer or null")
    exited = os.WIFEXITED(raw)
    exit_code = os.WEXITSTATUS(raw) if exited else None
    signaled = os.WIFSIGNALED(raw)
    signal = os.WTERMSIG(raw) if signaled else None
    expected = {
        "raw_wait_status": raw,
        "wifexited": exited,
        "exit_code": exit_code,
        "signal": signal,
    }
    if result != expected:
        raise ReceiptError("decoded wait_status disagrees with raw_wait_status")
    return result


def decode_wait_status(raw_wait_status: int | None) -> dict[str, Any]:
    if raw_wait_status is None:
        return {
            "raw_wait_status": None,
            "wifexited": None,
            "exit_code": None,
            "signal": None,
        }
    if type(raw_wait_status) is not int or raw_wait_status < 0:
        raise ValueError("raw_wait_status must be a non-negative integer or None")
    return _wait_status(
        {
            "raw_wait_status": raw_wait_status,
            "wifexited": os.WIFEXITED(raw_wait_status),
            "exit_code": (
                os.WEXITSTATUS(raw_wait_status)
                if os.WIFEXITED(raw_wait_status)
                else None
            ),
            "signal": (
                os.WTERMSIG(raw_wait_status)
                if os.WIFSIGNALED(raw_wait_status)
                else None
            ),
        },
        allow_null_raw=True,
    )


def _path_bound_record(value: Any, name: str) -> dict[str, Any]:
    fields = {"path", "sha256", "size"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReceiptError(f"{name} path-record fields changed")
    result = _strict_json(value, name)
    if not isinstance(result["path"], str) or not Path(result["path"]).is_absolute():
        raise ReceiptError(f"{name}.path must be absolute")
    validate_record(
        {"sha256": result["sha256"], "size": result["size"]}, name
    )
    return result


def _scan_attempt_directories(namespace: Path) -> tuple[tuple[int, Path], ...]:
    _require_directory(namespace, label="transaction namespace")
    attempts: list[tuple[int, Path]] = []
    for entry in namespace.iterdir():
        if entry.name == LOCK_NAME:
            _require_regular_file(entry, label="namespace lock")
            continue
        match = ATTEMPT_PATTERN.fullmatch(entry.name)
        if match is None:
            raise StateConflictError(f"unexpected transaction namespace entry: {entry}")
        _require_directory(entry, label="attempt directory")
        attempts.append((int(match.group(1)), entry))
    attempts.sort()
    expected = list(range(1, len(attempts) + 1))
    if [number for number, _ in attempts] != expected:
        raise StateConflictError("attempt directories are not a gap-free sequence")
    return tuple(attempts)


def _validate_attempt_entries(attempt_dir: Path) -> None:
    for entry in attempt_dir.iterdir():
        if entry.name not in STAGE_NAMES and _temp_target(entry.name) is None:
            raise StateConflictError(f"unexpected attempt evidence: {entry}")
        _require_regular_file(entry, label="attempt evidence")


def _context_from_intent(
    canonical: CanonicalPaths,
    attempt_number: int,
    attempt_dir: Path,
) -> AttemptContext:
    intent_path = attempt_dir / INTENT
    document = _load_json_file(intent_path, label=INTENT)
    if document.get("receipt") != INTENT:
        raise ReceiptError("attempt does not begin with 00_intent")
    identity = _identity(document.get("identity"))
    if identity.attempt_number != attempt_number:
        raise ReceiptError("intent attempt number differs from directory")
    document, checked_identity = _validate_flat_document(
        document, receipt=INTENT, expected_identity=identity
    )
    run_identity = validate_run_identity(document["run_identity"])
    run_id = compute_run_id(run_identity)
    if identity.run_id != run_id:
        raise ReceiptError("intent run_id differs from its immutable inputs")
    if run_identity["canonical_paths"] != canonical.as_run_id_dict():
        raise ReceiptError("intent canonical paths differ from namespace")
    context = AttemptContext(canonical, attempt_dir, identity, run_identity)
    if checked_identity != identity:
        raise ReceiptError("intent identity changed during validation")
    return context


def open_attempt(
    output_stem: str | os.PathLike[str] | CanonicalPaths, attempt_number: int
) -> AttemptContext:
    canonical = (
        output_stem
        if isinstance(output_stem, CanonicalPaths)
        else canonical_paths(output_stem)
    )
    if type(attempt_number) is not int or not 1 <= attempt_number <= 999:
        raise ValueError("attempt_number must be in [1, 999]")
    attempt_dir = canonical.namespace / f"attempt_{attempt_number:03d}"
    _require_directory(attempt_dir, label="attempt directory")
    _validate_attempt_entries(attempt_dir)
    return _context_from_intent(canonical, attempt_number, attempt_dir)


def list_attempts(output_stem: str | os.PathLike[str]) -> tuple[AttemptContext, ...]:
    canonical = canonical_paths(output_stem)
    if not canonical.namespace.exists():
        return ()
    return tuple(
        _context_from_intent(canonical, number, path)
        for number, path in _scan_attempt_directories(canonical.namespace)
    )


def allocate_attempt(
    lock: NamespaceLock,
    run_identity: Mapping[str, Any],
    *,
    run_metadata: Mapping[str, Any],
) -> AttemptContext:
    """Allocate the next exact flat-schema attempt under an acquired lock.

    This helper deliberately requires both run-id inputs and the independently
    duplicated 00 metadata.  The former legacy implementation silently wrote
    an incompatible nested-payload 00 receipt and is not a supported format.
    """

    lock.assert_held()
    canonical = lock.canonical
    checked_run_identity = validate_run_identity(run_identity)
    checked_run_metadata = _validate_run_metadata(run_metadata)
    if checked_run_identity["canonical_paths"] != canonical.as_run_id_dict():
        raise ValueError("run identity canonical paths differ from output_stem")
    run_id = compute_run_id(checked_run_identity)
    existing = _scan_attempt_directories(canonical.namespace)
    if existing:
        sequence = validate_attempt_sequence(
            canonical.output_stem,
            expected_run_id=run_id,
            repair_failure_links=True,
            lock=lock,
        )
        if sequence.final_state == "verified_recovery":
            raise RecoveryRequired("verified attempt requires publication recovery")
        if sequence.final_state == "committed":
            raise StateConflictError("namespace is already committed")
        if sequence.final_state != "retryable_failure":
            raise StateConflictError("prior attempt is not retryable")
        number = len(existing) + 1
    else:
        number = 1
    if number > 999:
        raise StateConflictError("attempt number exceeds three-digit namespace")
    identity = TransactionIdentity(
        run_id=run_id,
        attempt_id=compute_attempt_id(run_id, number),
        attempt_number=number,
    )
    intent_fields = {
        "status": "prepared",
        "run_metadata": checked_run_metadata,
        **checked_run_metadata,
        "run_identity": checked_run_identity,
    }
    # Reject inconsistent caller inputs before the gap-free attempt directory
    # itself becomes durable.
    _validate_flat_fields(INTENT, intent_fields, identity)
    stage = StageOnly.create(canonical, number)
    write_receipt(
        stage,
        INTENT,
        identity,
        {name: value for name, value in intent_fields.items() if name != "run_identity"},
    )
    return open_attempt(canonical, number)


RUN_METADATA_FIELDS = frozenset(
    {
        "seed",
        "replicate",
        "num_envs",
        "collection_commit",
        "preregistration_tag_commit",
        "assignment_mask_sha256",
        "source_manifest_sha256",
        "checkpoint_manifest_sha256",
        "runtime_asset_manifest_sha256",
        "canonical_artifact_output",
        "canonical_report_output",
        "canonical_commit_output",
        "argument_receipt",
    }
)


def _validate_run_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(RUN_METADATA_FIELDS):
        raise ReceiptError("run_metadata fields changed")
    result = _strict_json(value, "run_metadata")
    _require_git_sha(result["collection_commit"], "run_metadata.collection_commit")
    _require_git_sha(
        result["preregistration_tag_commit"],
        "run_metadata.preregistration_tag_commit",
    )
    for name in (
        "assignment_mask_sha256",
        "source_manifest_sha256",
        "checkpoint_manifest_sha256",
        "runtime_asset_manifest_sha256",
    ):
        _require_sha(result[name], f"run_metadata.{name}")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ReceiptError("run_metadata.seed is invalid")
    if result["replicate"] not in {"a", "b"}:
        raise ReceiptError("run_metadata.replicate is invalid")
    if type(result["num_envs"]) is not int or result["num_envs"] not in {8, 64}:
        raise ReceiptError("run_metadata.num_envs is invalid")
    for name in (
        "canonical_artifact_output",
        "canonical_report_output",
        "canonical_commit_output",
    ):
        if not isinstance(result[name], str) or not Path(result[name]).is_absolute():
            raise ReceiptError(f"run_metadata.{name} must be absolute")
    if not isinstance(result["argument_receipt"], dict):
        raise ReceiptError("run_metadata.argument_receipt must be a mapping")
    return result


def _validate_flat_fields(
    receipt: str,
    fields: Mapping[str, Any],
    identity: TransactionIdentity,
) -> dict[str, Any]:
    result = _strict_json(fields, f"{receipt}.fields")
    expected: set[str]
    if receipt == INTENT:
        metadata = _validate_run_metadata(result.get("run_metadata"))
        expected = {"status", "run_metadata", "run_identity"} | set(metadata)
        if set(result) != expected or result.get("status") != "prepared":
            raise ReceiptError("00 intent fields changed")
        for name, value in metadata.items():
            if result.get(name) != value:
                raise ReceiptError(f"00 duplicated run metadata changed: {name}")
        run_identity = validate_run_identity(result["run_identity"])
        if compute_run_id(run_identity) != identity.run_id:
            raise ReceiptError("00 run identity differs from attempt identity")
        expected_paths = {
            "artifact": metadata["canonical_artifact_output"],
            "report": metadata["canonical_report_output"],
            "commit": metadata["canonical_commit_output"],
        }
        if run_identity["canonical_paths"] != expected_paths:
            raise ReceiptError("00 run identity/metadata canonical paths differ")
        for name in (
            "seed",
            "replicate",
            "num_envs",
            "collection_commit",
            "assignment_mask_sha256",
            "source_manifest_sha256",
            "checkpoint_manifest_sha256",
            "runtime_asset_manifest_sha256",
        ):
            if run_identity[name] != metadata[name]:
                raise ReceiptError(f"00 run identity/metadata differ at {name}")
        return result
    schemas: dict[str, tuple[set[str], str]] = {
        PARENT_PRESPAWN_FAILED: (
            {
                "status",
                "error_type",
                "error",
                "child_created",
                "lock_transferred",
            },
            "failed",
        ),
        CHILD_STARTED: (
            {
                "status",
                "pid",
                "inherited_lock_validated",
                "fd_cloexec",
            },
            "started",
        ),
        APP_STARTED: ({"status", "fast_shutdown"}, "started"),
        CHILD_PREBOUNDARY_FAILED: (
            {"status", "error_type", "error", "traceback"},
            "failed",
        ),
        CLOSE_PREBOUNDARY: ({"status", "reason"}, "close_requested"),
        FIRST_STEP_ARMED: ({"status", "boundary_armed"}, "armed"),
        PREPARED: (
            {
                "status",
                "assignment_mask_sha256",
                "source_manifest_sha256",
                "checkpoint_manifest_sha256",
                "runtime_asset_manifest_sha256",
            },
            "prepared",
        ),
        CLOSE_SUCCESS: ({"status", "reason"}, "close_requested"),
        CHILD_FAILED: (
            {"status", "error_type", "error", "traceback"},
            "failed",
        ),
        CLOSE_FAILURE: (
            {"status", "reason"},
            "close_requested",
        ),
        CHILD_EXIT: (
            {
                "status",
                "raw_wait_status",
                "wifexited",
                "exit_code",
                "signal",
                "stdout",
                "stderr",
                "lock_release_proven",
            },
            "observed",
        ),
        VERIFIED: (
            {"status", "payload", "final_report", "post_exit_validated"},
            "verified",
        ),
    }
    if receipt == CLOSE_FAILURE and set(result) == {
        "status",
        "error_type",
        "error",
        "traceback",
        "reason",
    }:
        if result["status"] != "failed" or result["reason"] != "failure_after_prepared":
            raise ReceiptError("46 post-prepared conflict fields changed")
        if any(
            not isinstance(result[name], str)
            for name in ("error_type", "error", "traceback")
        ):
            raise ReceiptError("46 post-prepared exception fields changed")
        return result
    if receipt == PARENT_FAILED:
        required = {
            "status",
            "boundary_armed",
            "retry_permitted",
            "reason",
            "authority_failure",
            "raw_wait_status",
            "spawn_succeeded",
            "child_exit_observed",
            "recovery_lock_acquired",
            "observed",
            "canonical_observed",
        }
        if set(result) != required or result.get("status") != "failed":
            raise ReceiptError("70 fields changed")
        if (
            type(result["boundary_armed"]) is not bool
            or type(result["retry_permitted"]) is not bool
            or result["boundary_armed"] and result["retry_permitted"]
        ):
            raise ReceiptError("70 boundary/retry classification is invalid")
        for name in (
            "spawn_succeeded",
            "child_exit_observed",
            "recovery_lock_acquired",
        ):
            if type(result[name]) is not bool:
                raise ReceiptError(f"70.{name} must be bool")
        if not isinstance(result["reason"], str) or not result["reason"]:
            raise ReceiptError("70 reason must be non-empty")
        if not isinstance(result["authority_failure"], str):
            raise ReceiptError("70 authority_failure must be a string")
        if result["authority_failure"] and result["retry_permitted"]:
            raise ReceiptError("70 authority failure cannot permit retry")
        raw_wait_status = result["raw_wait_status"]
        if raw_wait_status is not None and (
            type(raw_wait_status) is not int or raw_wait_status < 0
        ):
            raise ReceiptError("70 raw_wait_status must be non-negative int or null")
        if not isinstance(result["observed"], dict):
            raise ReceiptError("70.observed must be a mapping")
        for name, record in result["observed"].items():
            if (
                not isinstance(name, str)
                or not name
                or name not in STAGE_NAMES
                and _temp_target(name) is None
            ):
                raise ReceiptError("70.observed key is invalid")
            _validate_observation(
                record, f"70.observed[{name!r}]", canonical=False
            )
        _validate_canonical_observed(result["canonical_observed"])
        return result
    if receipt == COMMITTED:
        expected = {
            "status",
            "canonical_artifact",
            "canonical_report",
            "verified",
        }
        if set(result) != expected or result.get("status") != "committed":
            raise ReceiptError("60 fields changed")
        _path_bound_record(result["canonical_artifact"], "60.artifact")
        _path_bound_record(result["canonical_report"], "60.report")
        validate_record(result["verified"], "60.verified")
        return result
    if receipt not in schemas:
        raise ReceiptError(f"no flat receipt schema registered for {receipt}")
    expected, status_value = schemas[receipt]
    if set(result) != expected or result.get("status") != status_value:
        raise ReceiptError(f"{receipt} fields or status changed")
    if receipt == PARENT_PRESPAWN_FAILED:
        if result["child_created"] is not False or result["lock_transferred"] is not False:
            raise ReceiptError("05 did not prove no child/lock transfer")
        if any(not isinstance(result[name], str) for name in ("error_type", "error")):
            raise ReceiptError("05 exception fields changed")
    if receipt == CHILD_STARTED:
        if (
            type(result["pid"]) is not int
            or result["pid"] <= 0
            or result["inherited_lock_validated"] is not True
            or result["fd_cloexec"] is not True
        ):
            raise ReceiptError("10 child/lock fields changed")
    if receipt == APP_STARTED and result["fast_shutdown"] is not True:
        raise ReceiptError("15 must assert fast_shutdown=true")
    if receipt in {CHILD_PREBOUNDARY_FAILED, CHILD_FAILED}:
        if any(not isinstance(result[name], str) for name in ("error_type", "error", "traceback")):
            raise ReceiptError(f"{receipt} exception fields changed")
    if receipt == CLOSE_PREBOUNDARY and result["reason"] != "preboundary_failure":
        raise ReceiptError("19 reason changed")
    if receipt == FIRST_STEP_ARMED and result["boundary_armed"] is not True:
        raise ReceiptError("20 must bind boundary_armed=true")
    if receipt == PREPARED:
        for name in (
            "assignment_mask_sha256",
            "source_manifest_sha256",
            "checkpoint_manifest_sha256",
            "runtime_asset_manifest_sha256",
        ):
            _require_sha(result[name], f"32.{name}")
    if receipt == CLOSE_SUCCESS and result["reason"] != "success":
        raise ReceiptError("40 reason changed")
    if receipt == CLOSE_FAILURE and result["reason"] != "postboundary_failure":
        raise ReceiptError("46 reason changed")
    if receipt == CHILD_EXIT:
        _wait_status(
            {
                "raw_wait_status": result["raw_wait_status"],
                "wifexited": result["wifexited"],
                "exit_code": result["exit_code"],
                "signal": result["signal"],
            },
            # Recovery may acquire namespace.lock only after every inherited
            # duplicate has gone away, then durably record the otherwise
            # unavailable wait status as null.
            allow_null_raw=True,
        )
        validate_record(result["stdout"], "50.stdout")
        validate_record(result["stderr"], "50.stderr")
        if result["lock_release_proven"] is not True:
            raise ReceiptError("50 must prove child lock release")
    if receipt == VERIFIED:
        validate_record(result["payload"], "56.payload")
        validate_record(result["final_report"], "56.final_report")
        if result["post_exit_validated"] is not True:
            raise ReceiptError("56 did not bind post-exit validation")
    return result


def _validate_flat_document(
    value: Any,
    *,
    receipt: str,
    expected_identity: TransactionIdentity | None = None,
) -> tuple[dict[str, Any], TransactionIdentity]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{receipt} must be a mapping")
    checked = _strict_json(value, receipt)
    if not RECEIPT_HEADER_FIELDS.issubset(checked):
        raise ReceiptError(f"{receipt} omitted its transaction header")
    if (
        checked["transaction_contract"] != TRANSACTION_CONTRACT
        or checked["receipt"] != receipt
    ):
        raise ReceiptError(f"{receipt} transaction header changed")
    identity = _identity(checked["identity"])
    if expected_identity is not None and identity != expected_identity:
        raise ReceiptError(f"{receipt} belongs to another attempt")
    predecessors = _predecessors(checked["predecessors"])
    _validate_predecessor_shape(receipt, predecessors)
    fields = {
        name: item
        for name, item in checked.items()
        if name not in RECEIPT_HEADER_FIELDS
    }
    if receipt == FINAL_REPORT:
        if set(checked) != set(FINAL_REPORT_FIELDS):
            raise ReceiptError("55 final-report fields changed")
        for name, expected in identity.as_dict().items():
            if fields.get(name) != expected:
                raise ReceiptError(f"55 top-level {name} differs from identity")
        if (
            fields.get("kind") != FINAL_REPORT_KIND
            or fields.get("version") != 1
            or fields.get("status") != "complete"
        ):
            raise ReceiptError("55 final-report identity changed")
        for name in (
            "child_exit_sha256",
            "source_manifest_sha256",
            "checkpoint_manifest_sha256",
            "runtime_asset_manifest_sha256",
            "artifact_sha256",
            "report_core_sha256",
        ):
            _require_sha(fields.get(name), f"55.{name}")
        if type(fields.get("artifact_size")) is not int or fields["artifact_size"] < 1:
            raise ReceiptError("55 artifact_size must be positive")
        if (
            not isinstance(fields.get("artifact_output"), str)
            or not Path(fields["artifact_output"]).is_absolute()
            or not isinstance(fields.get("post_exit_authority"), dict)
            or not isinstance(fields.get("report_core"), dict)
        ):
            raise ReceiptError("55 nested authority/report/path fields changed")
    else:
        _validate_flat_fields(receipt, fields, identity)
        if receipt == CHILD_EXIT and (
            fields["stdout"] != predecessors[STDOUT_LOG]
            or fields["stderr"] != predecessors[STDERR_LOG]
        ):
            raise ReceiptError("50 log fields differ from their exact predecessors")
    return checked, identity


def file_evidence(stage: StageOnly, name: str) -> FileEvidence:
    if not isinstance(stage, StageOnly):
        raise TypeError("stage must be StageOnly")
    path = stage.path(name)
    record = _path_record(path)
    payload: Mapping[str, Any] | None = None
    if name in RECEIPT_NAMES:
        payload, identity = _validate_flat_document(
            _load_json_file(path, label=name), receipt=name
        )
        if identity.attempt_number != stage.attempt_number:
            raise ReceiptError(f"{name} attempt number differs from directory")
    return FileEvidence(name, path, record["sha256"], record["size"], payload)


def load_receipt(stage: StageOnly, name: str) -> FileEvidence:
    evidence = file_evidence(stage, name)
    if name not in RECEIPT_NAMES or evidence.payload is None:
        raise ReceiptError(f"{name} is not a registered receipt")
    for predecessor, record in evidence.payload["predecessors"].items():
        validate_file_record(
            stage.path(predecessor), record, f"{name} <- {predecessor}"
        )
    if name == PARENT_FAILED:
        expected_predecessors = {
            observed_name: file_record
            for observed_name, record in evidence.payload["observed"].items()
            if observed_name in STAGE_NAMES
            if (file_record := _observation_file_record(record)) is not None
        }
        if evidence.payload["predecessors"] != expected_predecessors:
            raise ReceiptError("70 predecessors do not bind every observed stage file")
        for observed_name, record in evidence.payload["observed"].items():
            path = stage.attempt_dir / observed_name
            if path.parent != stage.attempt_dir:
                raise ReceiptError("70 observed path escaped attempt directory")
            if record["path"] != str(path):
                raise ReceiptError("70 observed path binding changed")
            if _observe_path(path) != record:
                raise ReceiptError(f"70 observed object changed: {observed_name}")
        current = _canonical_observed(stage.paths, stage)
        for role, record in evidence.payload["canonical_observed"].items():
            if current.get(role) != record:
                raise ReceiptError("70 recorded canonical observation changed")
    return evidence


def durable_stage_bytes(stage: StageOnly, name: str, payload: bytes) -> FileEvidence:
    return stage.write_bytes(name, payload)


def write_receipt(
    stage: StageOnly,
    receipt: str,
    identity: TransactionIdentity,
    fields: Mapping[str, Any],
    *,
    predecessors: Iterable[FileEvidence] = (),
) -> FileEvidence:
    if not isinstance(stage, StageOnly):
        raise TypeError("stage must be StageOnly")
    checked_identity = _identity(identity.as_dict())
    if checked_identity.attempt_number != stage.attempt_number:
        raise ReceiptError("identity attempt differs from stage directory")
    if receipt == FINAL_REPORT:
        raise ReceiptError("55 is staged from the episode's prebuilt strict envelope")
    predecessor_map: dict[str, dict[str, Any]] = {}
    for evidence in predecessors:
        if not isinstance(evidence, FileEvidence):
            raise TypeError("predecessors must contain FileEvidence")
        if evidence.path.parent != stage.attempt_dir:
            raise ReceiptError("predecessor escaped this attempt")
        actual = file_evidence(stage, evidence.name)
        if actual.as_dict() != evidence.as_dict():
            raise ReceiptError("predecessor changed before receipt write")
        predecessor_map[evidence.name] = evidence.as_dict()
    checked_fields = dict(_strict_json(fields, f"{receipt}.fields"))
    if receipt == INTENT:
        run_identity = _RUN_IDENTITY_CACHE.get(checked_identity.run_id)
        if run_identity is None:
            raise ReceiptError(
                "compute_run_id must precede 00 so its exact inputs can be sealed"
            )
        checked_fields["run_identity"] = run_identity
    elif receipt == PARENT_PRESPAWN_FAILED:
        checked_fields["child_created"] = False
        checked_fields["lock_transferred"] = False
    elif receipt == CHILD_STARTED:
        checked_fields["inherited_lock_validated"] = True
        checked_fields["fd_cloexec"] = True
    elif receipt == FIRST_STEP_ARMED:
        checked_fields["boundary_armed"] = True
    elif receipt == PREPARED:
        intent = load_receipt(stage, INTENT)
        assert intent.payload is not None
        runtime_manifest = intent.payload["runtime_asset_manifest_sha256"]
        supplied = checked_fields.get("runtime_asset_manifest_sha256")
        if supplied is not None and supplied != runtime_manifest:
            raise ReceiptError("32 runtime asset manifest differs from 00")
        # Older child call sites omitted this duplicated authority field.  It
        # is derived only from the immutable 00 receipt, never from ambient
        # process state, while an explicitly conflicting value fails closed.
        checked_fields["runtime_asset_manifest_sha256"] = runtime_manifest
    elif receipt == CHILD_EXIT:
        checked_fields["lock_release_proven"] = True
        for name, field in ((STDOUT_LOG, "stdout"), (STDERR_LOG, "stderr")):
            evidence = file_evidence(stage, name)
            predecessor_map[name] = evidence.as_dict()
            if checked_fields.get(field) != evidence.as_dict():
                raise ReceiptError(f"50 {field} differs from fsynced log")
    elif receipt == VERIFIED:
        checked_fields["payload"] = file_evidence(stage, STAGED_PAYLOAD).as_dict()
        checked_fields["final_report"] = file_evidence(stage, FINAL_REPORT).as_dict()
        checked_fields["post_exit_validated"] = True
    _validate_predecessor_shape(receipt, predecessor_map)
    document = {
        "transaction_contract": TRANSACTION_CONTRACT,
        "receipt": receipt,
        "identity": checked_identity.as_dict(),
        "predecessors": predecessor_map,
        **checked_fields,
    }
    _validate_flat_document(
        document, receipt=receipt, expected_identity=checked_identity
    )
    evidence = stage.write_bytes(receipt, pretty_json_bytes(document))
    return load_receipt(stage, receipt)


def durably_arm_first_step(
    stage: StageOnly,
    identity: TransactionIdentity,
    progress: Any,
    *,
    fields: Mapping[str, Any],
) -> FileEvidence:
    invoked = getattr(progress, "first_env_step_invoked", None)
    if type(invoked) is not bool:
        raise TypeError("progress must expose bool first_env_step_invoked")
    if invoked:
        return file_evidence(stage, FIRST_STEP_ARMED)
    if stage.path(FIRST_STEP_ARMED).exists() or stage.path(FIRST_STEP_ARMED).is_symlink():
        raise StateConflictError("first-step boundary already exists before progress flag")
    app_started = load_receipt(stage, APP_STARTED)
    return write_receipt(
        stage,
        FIRST_STEP_ARMED,
        identity,
        fields,
        predecessors=(app_started,),
    )


def validate_inherited_lock(
    descriptor: int, expected_lock_path: str | os.PathLike[str]
) -> None:
    validate_inherited_lock_fd(descriptor, expected_lock_path)


def _stage_existing_entries(stage: StageOnly) -> tuple[Path, ...]:
    _require_directory(stage.attempt_dir, label="attempt directory")
    entries = tuple(sorted(stage.attempt_dir.iterdir(), key=lambda item: item.name))
    for entry in entries:
        if entry.name not in STAGE_NAMES and _temp_target(entry.name) is None:
            raise StateConflictError(f"unexpected attempt evidence: {entry}")
    return entries


def _temp_target(name: str) -> str | None:
    if TEMP_PATTERN.fullmatch(name) is None or not name.startswith("."):
        return None
    marker = ".tmp-"
    if marker not in name:
        return None
    target = name[1 : name.index(marker)]
    return target if target in STAGE_NAMES else None


def _object_type(info: os.stat_result) -> str:
    mode = info.st_mode
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISBLK(mode):
        return "block_device"
    if stat.S_ISCHR(mode):
        return "char_device"
    return "other"


def _observe_path(
    path: Path,
    *,
    include_samefile_stage: bool = False,
    stage_source: Path | None = None,
) -> dict[str, Any]:
    """Fingerprint one filesystem object without following a symlink/FIFO."""

    before = path.lstat()
    kind = _object_type(before)
    size = int(before.st_size)
    if kind == "regular":
        fingerprint, after = _hash_regular_path(path, label="observed regular file")
        size = int(after.st_size)
    elif kind == "symlink":
        target = os.readlink(path)
        after = path.lstat()
        if not _same_lstat(before, after):
            raise PathSafetyError(f"observed symlink changed while reading: {path}")
        fingerprint = hashlib.sha256(
            b"candidate42-symlink\0" + os.fsencode(target)
        ).hexdigest()
    else:
        fingerprint = canonical_json_sha256(
            {
                "object_type": kind,
                "st_ctime_ns": int(before.st_ctime_ns),
                "st_dev": int(before.st_dev),
                "st_ino": int(before.st_ino),
                "st_mode": int(before.st_mode),
                "st_mtime_ns": int(before.st_mtime_ns),
                "st_size": size,
            }
        )
    result: dict[str, Any] = {
        "object_type": kind,
        "path": str(path),
        "size": size,
        "fingerprint": fingerprint,
    }
    if include_samefile_stage and kind == "regular":
        samefile = False
        if stage_source is not None:
            try:
                source_info = stage_source.lstat()
                if stat.S_ISREG(source_info.st_mode):
                    samefile = os.path.samefile(path, stage_source)
            except FileNotFoundError:
                pass
        result["samefile_stage"] = samefile
    return result


def _validate_observation(
    value: Any, name: str, *, canonical: bool
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{name} must be a mapping")
    result = _strict_json(value, name)
    kinds = {
        "regular",
        "symlink",
        "directory",
        "fifo",
        "socket",
        "block_device",
        "char_device",
        "other",
    }
    if result.get("object_type") not in kinds:
        raise ReceiptError(f"{name}.object_type is invalid")
    expected = {"object_type", "path", "size", "fingerprint"}
    if canonical and result["object_type"] == "regular":
        expected.add("samefile_stage")
    if set(result) != expected:
        raise ReceiptError(f"{name} observation fields changed")
    if not isinstance(result["path"], str) or not Path(result["path"]).is_absolute():
        raise ReceiptError(f"{name}.path must be absolute")
    if type(result["size"]) is not int or result["size"] < 0:
        raise ReceiptError(f"{name}.size must be a non-negative integer")
    _require_sha(result["fingerprint"], f"{name}.fingerprint")
    if "samefile_stage" in result and type(result["samefile_stage"]) is not bool:
        raise ReceiptError(f"{name}.samefile_stage must be bool")
    return result


def _observation_file_record(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if value.get("object_type") != "regular":
        return None
    return {"sha256": value["fingerprint"], "size": value["size"]}


def _raw_observed(stage: StageOnly, *, exclude: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = set() if exclude is None else set(exclude)
    result: dict[str, dict[str, Any]] = {}
    for entry in _stage_existing_entries(stage):
        if entry.name in excluded:
            continue
        result[entry.name] = _observe_path(entry)
    return result


def _canonical_observed(paths: CanonicalPaths, stage: StageOnly) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    sources = {
        "artifact": (paths.artifact, stage.path(STAGED_PAYLOAD)),
        "report": (paths.report, stage.path(FINAL_REPORT)),
        "commit": (paths.commit, stage.path(COMMITTED)),
    }
    for name, (path, source) in sources.items():
        if not (path.exists() or path.is_symlink()):
            continue
        result[name] = _observe_path(
            path,
            include_samefile_stage=True,
            stage_source=source,
        )
    return result


def _validate_canonical_observed(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError("canonical_observed must be a mapping")
    result = _strict_json(value, "canonical_observed")
    if not set(result).issubset({"artifact", "report", "commit"}):
        raise ReceiptError("canonical_observed contains an unknown canonical role")
    for name, record in result.items():
        _validate_observation(
            record, f"canonical_observed.{name}", canonical=True
        )
    return result


def classify_attempt(
    paths: CanonicalPaths,
    identity: TransactionIdentity,
    *,
    spawn_succeeded: bool,
    child_exit_observed: bool,
    recovery_lock_acquired: bool = False,
    _canonical_present: bool | None = None,
) -> AttemptClassification:
    if not isinstance(paths, CanonicalPaths):
        raise TypeError("paths must be TransactionPaths")
    checked_identity = _identity(identity.as_dict())
    stage = StageOnly.open(paths, checked_identity.attempt_number)
    observed = _raw_observed(stage, exclude={PARENT_FAILED})
    names = set(observed)
    temp_names = {name for name in names if TEMP_PATTERN.fullmatch(name)}
    canonical_present = (
        any(
            path.exists() or path.is_symlink()
            for path in (paths.artifact, paths.report, paths.commit)
        )
        if _canonical_present is None
        else bool(_canonical_present)
    )
    temp_targets = {
        target
        for name in names
        if (target := _temp_target(name)) is not None
    }
    boundary_armed = (
        FIRST_STEP_ARMED in names
        or bool(names & POST_BOUNDARY_NAMES)
        or bool(temp_targets & POST_BOUNDARY_NAMES)
        or canonical_present
    )

    # A retry decision is allowed to consume only a fully parsed, internally
    # consistent receipt prefix.  Hashing names into 70 is useful evidence but
    # is not sufficient authority: every receipt that exists must validate its
    # exact schema, predecessor bytes, and attempt identity first.
    if INTENT not in names:
        return AttemptClassification(
            boundary_armed, False, "attempt is missing immutable 00 intent", observed
        )
    loaded_receipts: dict[str, FileEvidence] = {}
    for receipt in sorted(names & RECEIPT_NAMES):
        try:
            evidence = load_receipt(stage, receipt)
            assert evidence.payload is not None
            if _identity(evidence.payload["identity"]) != checked_identity:
                raise ReceiptError(f"{receipt} crosses attempt identity")
            loaded_receipts[receipt] = evidence
        except (TransactionError, ValueError, TypeError) as error:
            return AttemptClassification(
                boundary_armed,
                False,
                f"malformed observed receipt {receipt}: {type(error).__name__}",
                observed,
            )

    unsafe_names = {
        name
        for name, record in observed.items()
        if record.get("object_type") != "regular"
    }
    if unsafe_names:
        return AttemptClassification(
            boundary_armed,
            False,
            "unsafe non-regular attempt evidence: " + ", ".join(sorted(unsafe_names)),
            observed,
        )

    # A durable-write temporary is an incomplete state transition.  Even when
    # its target precedes the first simulator step it cannot be interpreted as
    # an absent receipt and therefore never authorizes a retry.
    if temp_names:
        return AttemptClassification(
            boundary_armed,
            False,
            "orphan durable temporary requires terminal classification",
            observed,
        )
    if boundary_armed:
        return AttemptClassification(
            True,
            False,
            "first-step or later evidence exists",
            observed,
        )

    child_receipts = names & (
        RECEIPT_NAMES
        - {INTENT, PARENT_PRESPAWN_FAILED, PARENT_FAILED}
    )
    valid_05 = PARENT_PRESPAWN_FAILED in names
    valid_18 = CHILD_PREBOUNDARY_FAILED in names
    if valid_05 and child_receipts:
        return AttemptClassification(
            False,
            False,
            "05 pre-spawn receipt conflicts with child-side evidence",
            observed,
        )
    if not spawn_succeeded and child_receipts:
        return AttemptClassification(
            False,
            False,
            "child-side evidence contradicts spawn_succeeded=false",
            observed,
        )
    if CHILD_EXIT in names and not child_exit_observed:
        return AttemptClassification(
            False,
            False,
            "50 child-exit receipt contradicts child_exit_observed=false",
            observed,
        )
    if spawn_succeeded:
        child_exit = loaded_receipts.get(CHILD_EXIT)
        if child_exit is None or not child_exit_observed:
            return AttemptClassification(
                False,
                False,
                "spawn-successful retry lacks strict 50/log evidence",
                observed,
            )
        assert child_exit.payload is not None
        if (
            child_exit.payload["raw_wait_status"] is None
            and not recovery_lock_acquired
        ):
            return AttemptClassification(
                False,
                False,
                "nullable recovery 50 lacks namespace-lock release proof",
                observed,
            )

    if valid_05:
        retry = not spawn_succeeded and not child_exit_observed
        return AttemptClassification(
            False,
            retry,
            "receipt-proven caught pre-spawn failure"
            if retry
            else "05 contradicts parent spawn bookkeeping",
            observed,
        )
    if valid_18:
        retry = spawn_succeeded and (child_exit_observed or recovery_lock_acquired)
        return AttemptClassification(
            False,
            retry,
            "receipt-proven caught child pre-boundary failure"
            if retry
            else "child exit or inherited-lock release is unproven",
            observed,
        )
    # Abrupt pre-boundary failure has no terminal child receipt.  It may retry
    # only after waitpid or a recovery lock acquisition proves the inherited
    # duplicate can no longer be live.
    retry = spawn_succeeded and (child_exit_observed or recovery_lock_acquired)
    return AttemptClassification(
        False,
        retry,
        "abrupt pre-boundary child exit"
        if retry
        else "pre-boundary liveness is unproven",
        observed,
    )


def _write_parent_failure_document(
    stage: StageOnly,
    identity: TransactionIdentity,
    classification: AttemptClassification,
    *,
    reason: str,
    authority_failure: str,
    raw_wait_status: int | None,
    spawn_succeeded: bool,
    child_exit_observed: bool,
    recovery_lock_acquired: bool,
) -> FileEvidence:
    observed = dict(classification.observed)
    canonical_seen = _canonical_observed(stage.paths, stage)
    predecessor_map = {
        name: file_record
        for name, record in observed.items()
        if name in STAGE_NAMES and name != PARENT_FAILED
        if (file_record := _observation_file_record(record)) is not None
    }
    _validate_predecessor_shape(PARENT_FAILED, predecessor_map)
    fields = {
        "status": "failed",
        "boundary_armed": classification.boundary_armed,
        "retry_permitted": (
            classification.retry_permitted and not authority_failure
        ),
        "reason": reason,
        "authority_failure": authority_failure,
        "raw_wait_status": raw_wait_status,
        "spawn_succeeded": spawn_succeeded,
        "child_exit_observed": child_exit_observed,
        "recovery_lock_acquired": recovery_lock_acquired,
        "observed": observed,
        "canonical_observed": canonical_seen,
    }
    document = {
        "transaction_contract": TRANSACTION_CONTRACT,
        "receipt": PARENT_FAILED,
        "identity": identity.as_dict(),
        "predecessors": predecessor_map,
        **fields,
    }
    _validate_flat_document(
        document, receipt=PARENT_FAILED, expected_identity=identity
    )
    return stage.write_bytes(PARENT_FAILED, pretty_json_bytes(document))


def _samefile_or_collision(source: Path, target: Path, *, label: str) -> None:
    _require_regular_file(source, label=f"{label} source")
    _require_regular_file(target, label=label)
    if not os.path.samefile(source, target) or _path_record(source) != _path_record(target):
        raise StateConflictError(f"foreign or changed {label}: {target}")


def repair_failure_link(
    paths: CanonicalPaths,
    identity: TransactionIdentity,
    lock: NamespaceLock,
) -> FileEvidence:
    lock.assert_held()
    if lock.canonical != paths:
        raise TransactionError("failure repair lock belongs to another namespace")
    stage = StageOnly.open(paths, identity.attempt_number)
    failure = load_receipt(stage, PARENT_FAILED)
    canonical = Path(
        f"{paths.output_stem}.failed_attempt_{identity.attempt_number:03d}.json"
    )
    if canonical.exists() or canonical.is_symlink():
        _samefile_or_collision(failure.path, canonical, label="canonical failure receipt")
    else:
        os.link(failure.path, canonical, follow_symlinks=False)
        _fsync_directory(canonical.parent)
        _samefile_or_collision(failure.path, canonical, label="canonical failure receipt")
    return failure


def record_parent_failure(
    paths: CanonicalPaths,
    identity: TransactionIdentity,
    lock: NamespaceLock,
    *,
    reason: str,
    authority_failure: str = "",
    raw_wait_status: int | None,
    spawn_succeeded: bool,
    child_exit_observed: bool,
    recovery_lock_acquired: bool = False,
) -> FileEvidence:
    lock.assert_held()
    if lock.canonical != paths:
        raise TransactionError("failure lock belongs to another namespace")
    if not isinstance(reason, str) or not reason:
        raise ValueError("failure reason must be non-empty")
    if not isinstance(authority_failure, str):
        raise TypeError("authority_failure must be a string")
    checked_identity = _identity(identity.as_dict())
    stage = StageOnly.open(paths, checked_identity.attempt_number)
    if stage.path(PARENT_FAILED).exists() or stage.path(PARENT_FAILED).is_symlink():
        failure = load_receipt(stage, PARENT_FAILED)
        repair_failure_link(paths, checked_identity, lock)
        return failure
    classification = classify_attempt(
        paths,
        checked_identity,
        spawn_succeeded=spawn_succeeded,
        child_exit_observed=child_exit_observed,
        recovery_lock_acquired=recovery_lock_acquired,
    )
    failure = _write_parent_failure_document(
        stage,
        checked_identity,
        classification,
        reason=reason,
        authority_failure=authority_failure,
        raw_wait_status=raw_wait_status,
        spawn_succeeded=spawn_succeeded,
        child_exit_observed=child_exit_observed,
        recovery_lock_acquired=recovery_lock_acquired,
    )
    repair_failure_link(paths, checked_identity, lock)
    return failure


def _absent(path: Path, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise StateConflictError(f"forbidden {label} exists: {path}")


def validate_success_chain(
    stage: StageOnly, identity: TransactionIdentity
) -> dict[str, FileEvidence]:
    checked_identity = _identity(identity.as_dict())
    if checked_identity.attempt_number != stage.attempt_number:
        raise ReceiptError("success identity differs from stage")
    _validate_attempt_entries(stage.attempt_dir)
    for entry in stage.attempt_dir.iterdir():
        if TEMP_PATTERN.fullmatch(entry.name):
            raise StateConflictError("orphan durable temporary conflicts with success")
    for name in FAILURE_MARKER_NAMES:
        _absent(stage.path(name), label=f"success/failure conflict {name}")
    required_receipts = (
        INTENT,
        CHILD_STARTED,
        APP_STARTED,
        FIRST_STEP_ARMED,
        PREPARED,
        CLOSE_SUCCESS,
        CHILD_EXIT,
        FINAL_REPORT,
        VERIFIED,
    )
    result = {name: load_receipt(stage, name) for name in required_receipts}
    payload = file_evidence(stage, STAGED_PAYLOAD)
    report_core = file_evidence(stage, STAGED_REPORT_CORE)
    result[STAGED_PAYLOAD] = payload
    result[STAGED_REPORT_CORE] = report_core
    for evidence in result.values():
        if evidence.payload is not None:
            receipt_identity = _identity(evidence.payload["identity"])
            if receipt_identity != checked_identity:
                raise ReceiptError("success chain crosses attempt identities")

    intent = result[INTENT].payload
    prepared = result[PREPARED].payload
    child_exit = result[CHILD_EXIT].payload
    final = result[FINAL_REPORT].payload
    verified = result[VERIFIED].payload
    assert intent is not None and prepared is not None and child_exit is not None
    assert final is not None and verified is not None
    if (
        child_exit["wifexited"] is not True
        or child_exit["exit_code"] != 0
        or child_exit["signal"] is not None
    ):
        raise ReceiptError("success requires exact normal child exit zero")
    if child_exit["predecessors"].get(CLOSE_SUCCESS) != result[
        CLOSE_SUCCESS
    ].as_dict():
        raise ReceiptError("success 50 does not extend exact 40 close receipt")
    for receipt_name, intent_name in (
        ("assignment_mask_sha256", "assignment_mask_sha256"),
        ("source_manifest_sha256", "source_manifest_sha256"),
        ("checkpoint_manifest_sha256", "checkpoint_manifest_sha256"),
        ("runtime_asset_manifest_sha256", "runtime_asset_manifest_sha256"),
    ):
        if prepared[receipt_name] != intent[intent_name]:
            raise ReceiptError(f"32 authority differs from 00 at {receipt_name}")
    _require_sha(final.get("artifact_sha256"), "55.artifact_sha256")
    if (
        final["artifact_sha256"] != payload.sha256
        or final.get("artifact_size") != payload.size
        or final.get("artifact_output") != str(stage.paths.artifact)
    ):
        raise ReceiptError("55 artifact binding differs from staged 30/canonical path")
    if final.get("report_core_sha256") != report_core.sha256:
        raise ReceiptError("55 report-core hash differs from staged 31")
    for name in (
        "source_manifest_sha256",
        "checkpoint_manifest_sha256",
        "runtime_asset_manifest_sha256",
    ):
        if final.get(name) != intent.get(name):
            raise ReceiptError(f"55 differs from 00 at {name}")
    if (
        final.get("child_exit_sha256") != result[CHILD_EXIT].sha256
        or final["predecessors"].get(CHILD_EXIT)
        != result[CHILD_EXIT].as_dict()
    ):
        raise ReceiptError("55 differs from 50 predecessor")
    if verified["payload"] != payload.as_dict() or verified["final_report"] != result[
        FINAL_REPORT
    ].as_dict():
        raise ReceiptError("56 differs from staged payload/final report")
    return result


def _link_or_validate(source: Path, target: Path, *, label: str) -> None:
    _require_regular_file(source, label=f"{label} source")
    if target.exists() or target.is_symlink():
        _samefile_or_collision(source, target, label=label)
        return
    os.link(source, target, follow_symlinks=False)
    _fsync_directory(target.parent)
    _samefile_or_collision(source, target, label=label)


def _canonical_path_record(path: Path) -> dict[str, Any]:
    record = _path_record(path)
    return {"path": str(path), **record}


def publish_verified_attempt(
    paths: CanonicalPaths,
    identity: TransactionIdentity,
    lock: NamespaceLock,
) -> dict[str, Any]:
    """Publish or resume exactly one verified attempt without rerunning a child."""

    lock.assert_held()
    if lock.canonical != paths:
        raise TransactionError("publish lock belongs to another namespace")
    checked_identity = _identity(identity.as_dict())
    # Validate the complete attempt history before creating even the first
    # canonical artifact/report hardlink.  Otherwise a forged lower 70 could
    # be discovered only by the final analyzer call, after publication had
    # already escaped the attempt namespace.
    sequence = validate_attempt_sequence(
        paths.output_stem,
        expected_run_id=checked_identity.run_id,
        repair_failure_links=True,
        lock=lock,
    )
    if (
        sequence.final_state not in {"verified_recovery", "committed"}
        or not sequence.attempts
        or sequence.attempts[-1].identity != checked_identity
        or (
            sequence.final_state == "verified_recovery"
            and (
                sequence.recovery_attempt is None
                or sequence.recovery_attempt.identity != checked_identity
            )
        )
    ):
        raise StateConflictError("publication target is not the validated final attempt")
    stage = StageOnly.open(paths, checked_identity.attempt_number)
    _absent(stage.path(PARENT_FAILED), label="attempt-local 70")
    failure_link = Path(
        f"{paths.output_stem}.failed_attempt_{checked_identity.attempt_number:03d}.json"
    )
    _absent(failure_link, label="canonical failure receipt")
    chain = validate_success_chain(stage, checked_identity)
    payload = chain[STAGED_PAYLOAD]
    final = chain[FINAL_REPORT]
    verified = chain[VERIFIED]

    local_commit_exists = stage.path(COMMITTED).exists() or stage.path(COMMITTED).is_symlink()
    canonical_commit_exists = paths.commit.exists() or paths.commit.is_symlink()
    if local_commit_exists:
        # 60 is created only after both canonical pair links.  Their absence or
        # foreign replacement is an orphan-60 terminal conflict, not recoverable.
        _samefile_or_collision(payload.path, paths.artifact, label="canonical artifact")
        _samefile_or_collision(final.path, paths.report, label="canonical report")
        committed = load_receipt(stage, COMMITTED)
    else:
        if canonical_commit_exists:
            raise StateConflictError("canonical commit exists without attempt-local 60")
        _link_or_validate(payload.path, paths.artifact, label="canonical artifact")
        _link_or_validate(final.path, paths.report, label="canonical report")
        committed = write_receipt(
            stage,
            COMMITTED,
            checked_identity,
            {
                "status": "committed",
                "canonical_artifact": _canonical_path_record(paths.artifact),
                "canonical_report": _canonical_path_record(paths.report),
                "verified": verified.as_dict(),
            },
            predecessors=(verified,),
        )
    committed_document = committed.payload
    assert committed_document is not None
    if (
        committed_document["canonical_artifact"]
        != _canonical_path_record(paths.artifact)
        or committed_document["canonical_report"]
        != _canonical_path_record(paths.report)
        or committed_document["verified"] != verified.as_dict()
    ):
        raise ReceiptError("60 canonical bindings changed")
    _link_or_validate(committed.path, paths.commit, label="canonical commit")
    _fsync_directory(paths.commit.parent)
    return validate_committed_namespace(paths)


def recover_verified_publication(
    paths: CanonicalPaths,
    identity: TransactionIdentity,
    lock: NamespaceLock,
) -> dict[str, Any]:
    """Resume only publication from the same immutable 56; never launch a child."""

    return publish_verified_attempt(paths, identity, lock)


def _failure_link(paths: CanonicalPaths, attempt_number: int) -> Path:
    return Path(f"{paths.output_stem}.failed_attempt_{attempt_number:03d}.json")


def _canonical_failure_links(paths: CanonicalPaths) -> dict[str, Path]:
    """Return every failure-like canonical name for this exact output stem."""

    parent = paths.output_stem.parent
    _require_directory(parent, label="canonical output directory")
    prefix = f"{paths.output_stem.name}.failed_attempt_"
    result: dict[str, Path] = {}
    for entry in parent.iterdir():
        if entry.name.startswith(prefix) and entry.name.endswith(".json"):
            result[entry.name] = entry
    return result


def _validate_retryable_failure(
    paths: CanonicalPaths,
    stage: StageOnly,
    identity: TransactionIdentity,
    *,
    repair: bool,
    lock: NamespaceLock | None,
    allow_later_canonical: bool,
) -> FileEvidence:
    failure = load_receipt(stage, PARENT_FAILED)
    assert failure.payload is not None
    if failure.payload["authority_failure"]:
        raise StateConflictError("authority-failed attempt is terminal and non-retryable")
    raw_wait_status = failure.payload["raw_wait_status"]
    if raw_wait_status is not None:
        decode_wait_status(raw_wait_status)
        if failure.payload["child_exit_observed"] is not True:
            raise StateConflictError("70 raw wait status lacks child-exit bookkeeping")
    elif (
        failure.payload["child_exit_observed"] is True
        and failure.payload["recovery_lock_acquired"] is not True
    ):
        raise StateConflictError("70 child exit has neither wait nor recovery-lock proof")
    if failure.payload["spawn_succeeded"]:
        child_exit = load_receipt(stage, CHILD_EXIT)
        assert child_exit.payload is not None
        if child_exit.payload["raw_wait_status"] != raw_wait_status:
            raise StateConflictError("70 raw wait status differs from strict 50")
    current_canonical = _canonical_observed(paths, stage)
    recorded_canonical = failure.payload["canonical_observed"]
    if allow_later_canonical:
        for role, record in recorded_canonical.items():
            if current_canonical.get(role) != record:
                raise StateConflictError("lower 70 canonical snapshot changed")
    elif current_canonical != recorded_canonical:
        raise StateConflictError("final 70 canonical snapshot changed")
    recomputed = classify_attempt(
        paths,
        identity,
        spawn_succeeded=failure.payload["spawn_succeeded"],
        child_exit_observed=failure.payload["child_exit_observed"],
        recovery_lock_acquired=failure.payload["recovery_lock_acquired"],
        # A lower attempt cannot own canonical links later published by a
        # higher attempt.  Recompute against the immutable snapshot sealed in
        # its own 70, while the final attempt must match current globals above.
        _canonical_present=bool(recorded_canonical),
    )
    if (
        recomputed.boundary_armed != failure.payload["boundary_armed"]
        or recomputed.retry_permitted != failure.payload["retry_permitted"]
        or dict(recomputed.observed) != failure.payload["observed"]
    ):
        raise StateConflictError("70 classification differs from current strict evidence")
    if (
        failure.payload["boundary_armed"] is not False
        or failure.payload["retry_permitted"] is not True
    ):
        raise StateConflictError("lower attempt is not an exact retryable failure")
    for forbidden in SUCCESS_ONLY_NAMES:
        if forbidden in failure.payload["observed"] or (
            stage.path(forbidden).exists() or stage.path(forbidden).is_symlink()
        ):
            raise StateConflictError("retryable lower attempt contains success evidence")
    canonical = _failure_link(paths, identity.attempt_number)
    if not (canonical.exists() or canonical.is_symlink()):
        if not repair or lock is None:
            raise StateConflictError("retryable failure lacks repaired canonical link")
        repair_failure_link(paths, identity, lock)
    _samefile_or_collision(failure.path, canonical, label="canonical failure receipt")
    return failure


def validate_attempt_sequence(
    output_stem: str | os.PathLike[str],
    *,
    expected_run_id: str | None = None,
    repair_failure_links: bool = False,
    lock: NamespaceLock | None = None,
) -> SequenceValidation:
    paths = canonical_paths(output_stem)
    directories = _scan_attempt_directories(paths.namespace)
    if not directories:
        return SequenceValidation((), "empty", None)
    contexts: list[AttemptContext] = []
    for number, directory in directories:
        _validate_attempt_entries(directory)
        context = _context_from_intent(paths, number, directory)
        if expected_run_id is not None and context.identity.run_id != expected_run_id:
            raise StateConflictError("attempt sequence changed deterministic run_id")
        if contexts and context.identity.run_id != contexts[0].identity.run_id:
            raise StateConflictError("attempt sequence contains mixed run ids")
        contexts.append(context)
    for context in contexts[:-1]:
        _validate_retryable_failure(
            paths,
            StageOnly.open(paths, context.identity.attempt_number),
            context.identity,
            repair=repair_failure_links,
            lock=lock,
            allow_later_canonical=True,
        )
    final = contexts[-1]
    stage = StageOnly.open(paths, final.identity.attempt_number)
    has_70 = stage.path(PARENT_FAILED).exists() or stage.path(PARENT_FAILED).is_symlink()
    has_56 = stage.path(VERIFIED).exists() or stage.path(VERIFIED).is_symlink()
    has_60 = stage.path(COMMITTED).exists() or stage.path(COMMITTED).is_symlink()
    if has_70:
        _validate_retryable_failure(
            paths,
            stage,
            final.identity,
            repair=repair_failure_links,
            lock=lock,
            allow_later_canonical=False,
        )
        state = "retryable_failure"
        recovery = None
    elif has_60 or paths.commit.exists() or paths.commit.is_symlink():
        state = "committed"
        recovery = None
    elif has_56:
        load_receipt(stage, VERIFIED)
        state = "verified_recovery"
        recovery = final
    else:
        state = "unclassified"
        recovery = None
    return SequenceValidation(tuple(contexts), state, recovery)


def validate_committed_namespace(
    paths: CanonicalPaths,
    *,
    artifact_validator: Callable[[Path], Any] | None = None,
    report_validator: Callable[[Path, Path], Any] | None = None,
) -> dict[str, Any]:
    """Validate the sole analyzer commit point and the complete attempt history."""

    if not isinstance(paths, CanonicalPaths):
        raise TypeError("paths must be TransactionPaths")
    sequence = validate_attempt_sequence(paths.output_stem)
    if sequence.final_state != "committed" or not sequence.attempts:
        raise StateConflictError("namespace has no final committed attempt")
    final = sequence.attempts[-1]
    stage = StageOnly.open(paths, final.identity.attempt_number)
    _absent(stage.path(PARENT_FAILED), label="final attempt 70")
    _absent(_failure_link(paths, final.identity.attempt_number), label="final canonical 70")
    chain = validate_success_chain(stage, final.identity)
    committed = load_receipt(stage, COMMITTED)
    _samefile_or_collision(stage.path(STAGED_PAYLOAD), paths.artifact, label="canonical artifact")
    _samefile_or_collision(stage.path(FINAL_REPORT), paths.report, label="canonical report")
    _samefile_or_collision(stage.path(COMMITTED), paths.commit, label="canonical commit")
    assert committed.payload is not None
    if committed.payload["canonical_artifact"] != _canonical_path_record(paths.artifact):
        raise ReceiptError("60 artifact record changed")
    if committed.payload["canonical_report"] != _canonical_path_record(paths.report):
        raise ReceiptError("60 report record changed")
    if committed.payload["verified"] != chain[VERIFIED].as_dict():
        raise ReceiptError("60 verified record differs from current 56")
    expected_failure_links = {
        _failure_link(paths, context.identity.attempt_number).name
        for context in sequence.attempts[:-1]
    }
    observed_failure_links = set(_canonical_failure_links(paths))
    if observed_failure_links != expected_failure_links:
        raise StateConflictError(
            "canonical failure-receipt set differs from committed attempt history"
        )
    if artifact_validator is not None:
        artifact_validator(paths.artifact)
    if report_validator is not None:
        report_validator(paths.report, paths.artifact)
    artifact_record = _path_record(paths.artifact)
    report_record = _path_record(paths.report)
    commit_record = _path_record(paths.commit)
    return {
        "status": "committed",
        "identity": final.identity.as_dict(),
        "attempt_number": final.identity.attempt_number,
        "run_id": final.identity.run_id,
        "canonical_artifact_output": str(paths.artifact),
        "canonical_report_output": str(paths.report),
        "canonical_commit_output": str(paths.commit),
        "artifact_sha256": artifact_record["sha256"],
        "report_sha256": report_record["sha256"],
        "commit_sha256": commit_record["sha256"],
        "retryable_prior_attempts": [
            context.identity.as_dict() for context in sequence.attempts[:-1]
        ],
    }
