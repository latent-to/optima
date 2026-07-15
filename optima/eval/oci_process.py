"""Host-owned OCI process lifecycle and crash recovery.

This module knows nothing about candidates, models, scoring, or engine protocol.  It
owns only a Docker-compatible client process, the exact container name/CID, and
lease-scoped local resources.  The same primitive is used by offline prebuild and
streaming runtime so timeout/cancellation/controller-restart cleanup has one authority.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
import secrets
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable, Protocol


LEASE_SCHEMA = "optima.oci-process-lease.v1"
EXECUTOR_LABEL = "optima.executor_id"
LEASE_LABEL = "optima.lease_id"
GPU_RESERVATION_ENV = "OPTIMA_GPU_RESERVATION_ID"
GPU_RESERVATION_LABEL = "optima.gpu_reservation_id"
ATTACHED_STDERR_MAX_BYTES = 64 << 10
ATTACHED_STDERR_EXCERPT_BYTES = 2 << 10
ATTACHED_STDERR_ARTIFACT_MAX_BYTES = 16 << 20
STDERR_ARTIFACT_SCHEMA = "optima.oci-stderr-artifact.v1"
_ATTACHED_STDERR_READ_BYTES = 64 << 10
_ATTACHED_STDERR_JOIN_SECONDS = 2.0
_SIMPLE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_GPU_RESERVATION_ID = re.compile(r"[0-9a-f]{12}\Z")
_CID = re.compile(r"[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STDERR_ARTIFACT_NAME = re.compile(
    r"(?P<lease>[a-z0-9][a-z0-9_.-]{0,127})\.(?P<nonce>[0-9a-f]{32})\.stderr\Z"
)
_INSPECT_FORMAT = (
    '--format={"Id":{{json .Id}},"Name":{{json .Name}},'
    '"Labels":{{json .Config.Labels}}}'
)


class OCIProcessError(RuntimeError):
    pass


class OCIProcessTimeout(OCIProcessError):
    def __init__(
        self, message: str, diagnostic: "OCIAttachedDiagnostic | None" = None
    ) -> None:
        super().__init__(message)
        self._message = message
        self.diagnostic = (
            diagnostic if type(diagnostic) is OCIAttachedDiagnostic else None
        )

    def __str__(self) -> str:
        if self.diagnostic is None:
            return self._message
        return f"{self._message}; {self.diagnostic.summary}"


@dataclass(frozen=True)
class OCIStderrArtifactReceipt:
    """Host-owned identity for one private bounded stderr prefix artifact.

    ``stream_*`` authenticates every byte drained from the pipe. ``artifact_*``
    authenticates the retained prefix and is therefore independently reopenable
    even when the complete stream exceeded the hard disk cap.
    """

    schema: str
    executor_id: str
    lease_id: str
    artifact_path: Path
    receipt_path: Path
    artifact_sha256: str
    stream_sha256: str
    artifact_bytes: int
    stream_bytes: int
    truncated: bool
    owner_uid: int
    owner_gid: int
    mode: int

    def __post_init__(self) -> None:
        artifact = Path(self.artifact_path)
        receipt = Path(self.receipt_path)
        object.__setattr__(self, "artifact_path", artifact)
        object.__setattr__(self, "receipt_path", receipt)
        match = _STDERR_ARTIFACT_NAME.fullmatch(artifact.name)
        if (
            self.schema != STDERR_ARTIFACT_SCHEMA
            or not isinstance(self.executor_id, str)
            or _SIMPLE_ID.fullmatch(self.executor_id) is None
            or not isinstance(self.lease_id, str)
            or _SIMPLE_ID.fullmatch(self.lease_id) is None
            or match is None
            or match.group("lease") != self.lease_id
            or not artifact.is_absolute()
            or not receipt.is_absolute()
            or len(os.fsencode(artifact)) > 1024
            or len(os.fsencode(receipt)) > 1024
            or artifact.parent != receipt.parent
            or receipt.name != artifact.name + ".json"
            or any(part in ("", ".", "..") for part in artifact.parts)
            or any(part in ("", ".", "..") for part in receipt.parts)
            or not isinstance(self.artifact_sha256, str)
            or _SHA256.fullmatch(self.artifact_sha256) is None
            or not isinstance(self.stream_sha256, str)
            or _SHA256.fullmatch(self.stream_sha256) is None
            or type(self.artifact_bytes) is not int
            or not 0 <= self.artifact_bytes <= ATTACHED_STDERR_ARTIFACT_MAX_BYTES
            or type(self.stream_bytes) is not int
            or self.stream_bytes < self.artifact_bytes
            or type(self.truncated) is not bool
            or self.truncated != (self.stream_bytes > self.artifact_bytes)
            or (
                not self.truncated
                and self.artifact_sha256 != self.stream_sha256
            )
            or type(self.owner_uid) is not int
            or self.owner_uid < 0
            or type(self.owner_gid) is not int
            or self.owner_gid < 0
            or self.mode != 0o600
        ):
            raise OCIProcessError("OCI stderr artifact receipt is malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_bytes": self.artifact_bytes,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "executor_id": self.executor_id,
            "lease_id": self.lease_id,
            "mode": self.mode,
            "owner_gid": self.owner_gid,
            "owner_uid": self.owner_uid,
            "receipt_path": str(self.receipt_path),
            "schema": self.schema,
            "stream_bytes": self.stream_bytes,
            "stream_sha256": self.stream_sha256,
            "truncated": self.truncated,
        }

    @property
    def receipt_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.receipt_bytes).hexdigest()

    @property
    def digest(self) -> str:
        return self.receipt_sha256


@dataclass(frozen=True)
class OCIAttachedDiagnostic:
    """One bounded, private diagnostic snapshot from an attached OCI client.

    The terminal-safe tail and private prefix artifact are never part of launch
    identity or the worker-visible protocol. The host hashes and drains the whole
    stream while bounding both retained representations.
    """

    stderr_tail: bytes
    stderr_truncated: bool
    capture_complete: bool
    capture_error: str | None = None
    client_returncode: int | None = None
    stream_bytes: int | None = None
    stream_sha256: str | None = None
    artifact: OCIStderrArtifactReceipt | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stderr_tail, bytes)
            or len(self.stderr_tail) > ATTACHED_STDERR_MAX_BYTES
            or type(self.stderr_truncated) is not bool
            or type(self.capture_complete) is not bool
            or (
                self.client_returncode is not None
                and type(self.client_returncode) is not int
            )
            or (
                self.capture_error is not None
                and (
                    not isinstance(self.capture_error, str)
                    or not self.capture_error
                    or len(self.capture_error) > 512
                    or "\x00" in self.capture_error
                )
            )
            or (
                self.stream_bytes is not None
                and (type(self.stream_bytes) is not int or self.stream_bytes < 0)
            )
            or (
                self.stream_sha256 is not None
                and (
                    not isinstance(self.stream_sha256, str)
                    or _SHA256.fullmatch(self.stream_sha256) is None
                )
            )
            or (self.stream_bytes is None) != (self.stream_sha256 is None)
            or (
                self.artifact is not None
                and (
                    type(self.artifact) is not OCIStderrArtifactReceipt
                    or self.stream_bytes != self.artifact.stream_bytes
                    or self.stream_sha256 != self.artifact.stream_sha256
                )
            )
        ):
            raise OCIProcessError("attached OCI diagnostic is malformed")

    @property
    def stderr_sha256(self) -> str:
        return hashlib.sha256(self.stderr_tail).hexdigest()

    @property
    def summary(self) -> str:
        """A terminal-safe bounded summary; the complete bounded tail stays typed."""

        fields = [
            f"outer_oci_stderr_bytes={len(self.stderr_tail)}",
            f"outer_oci_stderr_sha256={self.stderr_sha256}",
            f"outer_oci_stderr_truncated={str(self.stderr_truncated).lower()}",
            f"outer_oci_stderr_complete={str(self.capture_complete).lower()}",
        ]
        if self.client_returncode is not None:
            fields.append(f"outer_oci_client_returncode={self.client_returncode}")
        if self.stream_bytes is not None:
            fields.extend(
                (
                    f"outer_oci_stderr_stream_bytes={self.stream_bytes}",
                    f"outer_oci_stderr_stream_sha256={self.stream_sha256}",
                )
            )
        if self.artifact is not None:
            fields.extend(
                (
                    "outer_oci_stderr_artifact_path="
                    f"{str(self.artifact.artifact_path)!r}",
                    "outer_oci_stderr_artifact_sha256="
                    f"{self.artifact.artifact_sha256}",
                    "outer_oci_stderr_artifact_bytes="
                    f"{self.artifact.artifact_bytes}",
                    "outer_oci_stderr_artifact_truncated="
                    f"{str(self.artifact.truncated).lower()}",
                    "outer_oci_stderr_receipt_path="
                    f"{str(self.artifact.receipt_path)!r}",
                    "outer_oci_stderr_receipt_sha256="
                    f"{self.artifact.receipt_sha256}",
                )
            )
        if self.capture_error is not None:
            fields.append(f"outer_oci_stderr_capture_error={self.capture_error!r}")
        if self.stderr_tail:
            # bytes.__repr__ escapes terminal controls.  Bound the input before
            # formatting so even worst-case escaping remains small.
            excerpt = self.stderr_tail[-ATTACHED_STDERR_EXCERPT_BYTES:]
            fields.append(f"outer_oci_stderr_tail={excerpt!r}")
        return " ".join(fields)


class _StderrArtifactWriter:
    """Stream only the bounded prefix to a private host-owned regular file."""

    def __init__(self, manager: "OCIProcessManager", lease: "OCILease") -> None:
        self.manager = manager
        self.lease = lease
        token = secrets.token_hex(16)
        self.artifact_path = manager.diagnostics_root / (
            f"{lease.lease_id}.{token}.stderr"
        )
        self.receipt_path = Path(str(self.artifact_path) + ".json")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(self.artifact_path, flags, 0o600)
        self._artifact_hasher = hashlib.sha256()
        self._artifact_bytes = 0
        self._closed = False
        try:
            os.fchmod(self.fd, 0o600)
            info = os.fstat(self.fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid()
                or info.st_gid != os.getegid()
            ):
                raise OCIProcessError(
                    "OCI stderr artifact is not a private controller-owned file"
                )
        except BaseException:
            os.close(self.fd)
            self._closed = True
            self.artifact_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_all(fd: int, payload: bytes) -> None:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            try:
                count = os.write(fd, view[offset:])
            except InterruptedError:
                continue
            if count <= 0:
                raise OSError("stderr artifact write made no progress")
            offset += count

    def append(self, chunk: bytes) -> None:
        remaining = ATTACHED_STDERR_ARTIFACT_MAX_BYTES - self._artifact_bytes
        if remaining <= 0:
            return
        retained = chunk[:remaining]
        self._write_all(self.fd, retained)
        self._artifact_hasher.update(retained)
        self._artifact_bytes += len(retained)

    def finish(
        self, *, stream_bytes: int, stream_sha256: str
    ) -> OCIStderrArtifactReceipt:
        if self._closed:
            raise OCIProcessError("OCI stderr artifact writer is already closed")
        try:
            os.fsync(self.fd)
            info = os.fstat(self.fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size != self._artifact_bytes
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid()
                or info.st_gid != os.getegid()
            ):
                raise OCIProcessError(
                    "OCI stderr artifact changed before receipt publication"
                )
        finally:
            os.close(self.fd)
            self._closed = True
        receipt = OCIStderrArtifactReceipt(
            STDERR_ARTIFACT_SCHEMA,
            self.lease.executor_id,
            self.lease.lease_id,
            self.artifact_path,
            self.receipt_path,
            self._artifact_hasher.hexdigest(),
            stream_sha256,
            self._artifact_bytes,
            stream_bytes,
            stream_bytes > self._artifact_bytes,
            info.st_uid,
            info.st_gid,
            stat.S_IMODE(info.st_mode),
        )
        self.manager._publish_stderr_receipt(receipt)
        return receipt

    def abort(self) -> None:
        if not self._closed:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self._closed = True
        self.receipt_path.unlink(missing_ok=True)
        self.artifact_path.unlink(missing_ok=True)


class _AttachedStderrCapture:
    """Drain one pipe while retaining a tail and streaming a bounded prefix."""

    def __init__(
        self,
        stream: BinaryIO | None,
        artifact_writer: _StderrArtifactWriter | None = None,
        artifact_error: str | None = None,
    ) -> None:
        self._tail = bytearray()
        self._truncated = False
        self._complete = False
        self._error: str | None = artifact_error
        self._stream_hasher = hashlib.sha256()
        self._stream_bytes = 0
        self._artifact_writer = artifact_writer
        self._artifact: OCIStderrArtifactReceipt | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if stream is None:
            self._error = "attached OCI client did not expose a stderr pipe"
            return
        try:
            fd = stream.fileno()
            if type(fd) is not int or fd < 0:
                raise OSError("invalid stderr file descriptor")
            self._thread = threading.Thread(
                target=self._drain,
                args=(fd,),
                name="optima-oci-stderr-drain",
                daemon=True,
            )
            self._thread.start()
        except (OSError, RuntimeError, ValueError) as exc:
            if artifact_writer is not None:
                artifact_writer.abort()
                self._artifact_writer = None
            self._error = self._bounded_error(exc)

    @staticmethod
    def _bounded_error(exc: BaseException) -> str:
        detail = str(exc)
        return f"{type(exc).__name__}: {detail[:448]}"[:512]

    def _append(self, chunk: bytes) -> None:
        with self._lock:
            self._stream_hasher.update(chunk)
            self._stream_bytes += len(chunk)
            if len(chunk) >= ATTACHED_STDERR_MAX_BYTES:
                self._tail[:] = chunk[-ATTACHED_STDERR_MAX_BYTES:]
                self._truncated = True
            else:
                overflow = len(self._tail) + len(chunk) - ATTACHED_STDERR_MAX_BYTES
                if overflow > 0:
                    del self._tail[:overflow]
                    self._truncated = True
                self._tail.extend(chunk)
        writer = self._artifact_writer
        if writer is not None:
            try:
                writer.append(chunk)
            except (OSError, OCIProcessError) as exc:
                writer.abort()
                self._artifact_writer = None
                with self._lock:
                    self._error = self._bounded_error(exc)

    def _drain(self, fd: int) -> None:
        reached_eof = False
        try:
            while True:
                try:
                    chunk = os.read(fd, _ATTACHED_STDERR_READ_BYTES)
                except InterruptedError:
                    continue
                if not chunk:
                    reached_eof = True
                    break
                self._append(chunk)
        except OSError as exc:
            with self._lock:
                self._error = self._bounded_error(exc)
        finally:
            writer = self._artifact_writer
            if writer is not None and reached_eof:
                with self._lock:
                    stream_bytes = self._stream_bytes
                    stream_sha256 = self._stream_hasher.hexdigest()
                try:
                    artifact = writer.finish(
                        stream_bytes=stream_bytes,
                        stream_sha256=stream_sha256,
                    )
                except (OSError, OCIProcessError) as exc:
                    writer.abort()
                    with self._lock:
                        self._error = self._bounded_error(exc)
                else:
                    with self._lock:
                        self._artifact = artifact
            elif writer is not None:
                writer.abort()
                self._artifact_writer = None
            with self._lock:
                self._complete = True

    def finish(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=_ATTACHED_STDERR_JOIN_SECONDS)
        if thread.is_alive():
            with self._lock:
                self._error = "stderr drain did not reach EOF after client teardown"

    def snapshot(self) -> OCIAttachedDiagnostic:
        with self._lock:
            return OCIAttachedDiagnostic(
                bytes(self._tail),
                self._truncated,
                self._complete,
                self._error,
                None,
                self._stream_bytes,
                self._stream_hasher.copy().hexdigest(),
                self._artifact,
            )

    def discard_artifact(self) -> None:
        with self._lock:
            artifact = self._artifact
            writer = self._artifact_writer
        if artifact is None:
            return
        if writer is None:
            raise OCIProcessError("stderr artifact lacks its manager authority")
        writer.manager.reopen_stderr_artifact(artifact)
        artifact.receipt_path.unlink(missing_ok=True)
        OCIProcessManager._fsync_directory(artifact.artifact_path.parent)
        artifact.artifact_path.unlink(missing_ok=True)
        OCIProcessManager._fsync_directory(artifact.artifact_path.parent)
        with self._lock:
            if self._artifact == artifact:
                self._artifact = None


@dataclass(frozen=True)
class OCIQuiescenceReceipt:
    """Manager-owned proof that one executor namespace has no live resources."""

    schema: str
    executor_id: str
    manager_instance_id: str
    namespace_digest: str
    sequence: int
    observed_monotonic_s: float
    lease_records: tuple[str, ...]
    resource_entries: tuple[str, ...]
    container_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.schema != "optima.oci-quiescence.v1"
            or not isinstance(self.executor_id, str)
            or _SIMPLE_ID.fullmatch(self.executor_id) is None
            or not isinstance(self.manager_instance_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", self.manager_instance_id) is None
            or not isinstance(self.namespace_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.namespace_digest) is None
            or type(self.sequence) is not int
            or self.sequence < 1
            or type(self.observed_monotonic_s) is not float
            or not math.isfinite(self.observed_monotonic_s)
            or self.observed_monotonic_s < 0
            or any((self.lease_records, self.resource_entries, self.container_ids))
        ):
            raise OCIProcessError("OCI quiescence receipt is malformed or nonempty")

    @property
    def digest(self) -> str:
        payload = {
            "container_ids": list(self.container_ids),
            "executor_id": self.executor_id,
            "manager_instance_id": self.manager_instance_id,
            "namespace_digest": self.namespace_digest,
            "lease_records": list(self.lease_records),
            "observed_monotonic_s": format(self.observed_monotonic_s, ".17g"),
            "resource_entries": list(self.resource_entries),
            "schema": self.schema,
            "sequence": self.sequence,
        }
        return hashlib.sha256(b"optima.oci-quiescence.v1\0" + _canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class CommandRunner(Protocol):
    def __call__(
        self, argv: tuple[str, ...], *, timeout_s: float, max_output_bytes: int
    ) -> CommandResult: ...


class CaptureRunner(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout_s: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
    ) -> CommandResult: ...


def _bounded_command_runner(
    argv: tuple[str, ...], *, timeout_s: float, max_output_bytes: int
) -> CommandResult:
    try:
        result = subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OCIProcessError(f"OCI control command failed: {exc}") from None
    stdout = bytes(result.stdout)
    stderr = bytes(result.stderr)
    if len(stdout) > max_output_bytes or len(stderr) > max_output_bytes:
        raise OCIProcessError("OCI control command exceeded its output bound")
    return CommandResult(int(result.returncode), stdout, stderr)


@dataclass(frozen=True)
class OCILease:
    executor_id: str
    lease_id: str
    container_name: str
    recovery_root: Path
    resource_root: Path
    record_path: Path
    cid_path: Path
    mount_paths: tuple[Path, ...]
    stage_paths: tuple[Path, ...]
    gpu_reservation_id: str | None = None

    def run_prefix(self, docker_binary: str) -> tuple[str, ...]:
        prefix = (
            docker_binary,
            "run",
            f"--name={self.container_name}",
            f"--cidfile={self.cid_path}",
            f"--label={EXECUTOR_LABEL}={self.executor_id}",
            f"--label={LEASE_LABEL}={self.lease_id}",
        )
        if self.gpu_reservation_id is not None:
            prefix += (
                f"--label={GPU_RESERVATION_LABEL}={self.gpu_reservation_id}",
            )
        return prefix


@dataclass(frozen=True)
class OCIProcessResult:
    returncode: int
    elapsed_seconds: float
    stderr_diagnostic: OCIAttachedDiagnostic | None = None


class OCIAttachedClient:
    """One foreground OCI client whose teardown remains manager-owned.

    The transport layer may use only the byte streams; pipe EOF is its liveness
    signal. It cannot replace the lease-aware cleanup sequence. ``finalize`` and ``abort``
    have identical process/container cleanup authority; normal completion discards
    the diagnostic artifact while exceptional teardown retains its sealed receipt.
    """

    def __init__(
        self,
        manager: OCIProcessManager,
        lease: OCILease,
        process: subprocess.Popen[bytes],
    ) -> None:
        self._manager = manager
        self._lease = lease
        self._process = process
        self._closed = False
        self._stderr_capture = manager._stderr_capture(process.stderr, lease)

    @property
    def lease(self) -> OCILease:
        return self._lease

    @property
    def stdin(self) -> BinaryIO:
        if self._closed or self._process.stdin is None:
            raise OCIProcessError("attached OCI client stdin is unavailable")
        return self._process.stdin

    @property
    def stdout(self) -> BinaryIO:
        if self._closed or self._process.stdout is None:
            raise OCIProcessError("attached OCI client stdout is unavailable")
        return self._process.stdout

    @property
    def closed(self) -> bool:
        return self._closed

    def stderr_diagnostic(self) -> OCIAttachedDiagnostic:
        """Return the current bounded stderr suffix without waiting or reaping."""

        snapshot = self._stderr_capture.snapshot()
        return OCIAttachedDiagnostic(
            snapshot.stderr_tail,
            snapshot.stderr_truncated,
            snapshot.capture_complete,
            snapshot.capture_error,
            self._process.returncode,
            snapshot.stream_bytes,
            snapshot.stream_sha256,
            snapshot.artifact,
        )

    def _finish_stderr_capture(self) -> None:
        self._stderr_capture.finish()

    def finalize(self) -> None:
        self._manager.finalize_attached(self)

    def abort(self) -> None:
        self._manager.abort_attached(self)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _simple_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SIMPLE_ID.fullmatch(value) is None:
        raise OCIProcessError(f"{field} must be a lowercase simple identifier")
    return value


def _gpu_reservation_id(value: object) -> str:
    if not isinstance(value, str) or _GPU_RESERVATION_ID.fullmatch(value) is None:
        raise OCIProcessError(
            f"{GPU_RESERVATION_ENV} must be a 12-character lowercase hex identity"
        )
    return value


def _relative_resource(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise OCIProcessError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise OCIProcessError(f"{field} must be a canonical relative path")
    canonical = path.as_posix()
    if canonical != value:
        raise OCIProcessError(f"{field} must be a canonical relative path")
    return canonical


def _within(root: Path, path: Path, *, field: str) -> Path:
    resolved = path.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise OCIProcessError(f"{field} escapes its lease resource root")
    for parent in (resolved, *resolved.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise OCIProcessError(f"{field} traverses a symlink")
    return resolved


class OCIProcessManager:
    """Create, run, reap, and recover exact lease-owned OCI resources."""

    def __init__(
        self,
        *,
        docker_binary: str,
        recovery_root: str | Path,
        executor_id: str,
        runner: CommandRunner = _bounded_command_runner,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(docker_binary, str) or not docker_binary.startswith("/"):
            raise OCIProcessError("docker_binary must be an absolute path")
        self.docker_binary = docker_binary
        self.executor_id = _simple_id(executor_id, field="executor_id")
        root = Path(recovery_root)
        if root.is_symlink():
            raise OCIProcessError("recovery_root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        self.recovery_root = root.resolve()
        root_info = self.recovery_root.stat()
        locks_root = self.recovery_root / "locks"
        locks_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = locks_root / f"{self.executor_id}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            self._namespace_lock_fd = os.open(lock_path, flags, 0o600)
            fcntl.flock(
                self._namespace_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except (OSError, BlockingIOError) as exc:
            if getattr(self, "_namespace_lock_fd", -1) >= 0:
                os.close(self._namespace_lock_fd)
            self._namespace_lock_fd = -1
            raise OCIProcessError(
                f"OCI executor namespace is already owned: {exc}"
            ) from None
        self.manager_instance_id = secrets.token_hex(16)
        self.namespace_digest = hashlib.sha256(
            b"optima.oci-namespace.v1\0"
            + _canonical_json(
                {
                    "docker_binary": self.docker_binary,
                    "executor_id": self.executor_id,
                    "recovery_device": root_info.st_dev,
                    "recovery_inode": root_info.st_ino,
                }
            )
        ).hexdigest()
        self.leases_root = self.recovery_root / "leases" / self.executor_id
        self.resources_root = self.recovery_root / "resources" / self.executor_id
        self.diagnostics_root = self.recovery_root / "diagnostics" / self.executor_id
        self.leases_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.resources_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.diagnostics_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        diagnostics_info = self.diagnostics_root.lstat()
        if (
            not stat.S_ISDIR(diagnostics_info.st_mode)
            or stat.S_ISLNK(diagnostics_info.st_mode)
            or diagnostics_info.st_uid != os.geteuid()
            or diagnostics_info.st_gid != os.getegid()
        ):
            raise OCIProcessError(
                "OCI diagnostics root must be a controller-owned real directory"
            )
        os.chmod(self.diagnostics_root, 0o700)
        diagnostics_info = self.diagnostics_root.stat()
        if stat.S_IMODE(diagnostics_info.st_mode) != 0o700:
            raise OCIProcessError("OCI diagnostics root is not private")
        self._diagnostics_identity = (
            diagnostics_info.st_dev,
            diagnostics_info.st_ino,
            diagnostics_info.st_uid,
            diagnostics_info.st_gid,
            stat.S_IMODE(diagnostics_info.st_mode),
        )
        self.runner = runner
        self.clock = clock
        # A trusted controller may reserve the complete B/C/B-prime/T
        # transaction while re-entering for each individual engine lifetime.
        self.transaction_lock = threading.RLock()
        self._quiescence_sequence = 0

    def _require_namespace_owner(self) -> None:
        if getattr(self, "_namespace_lock_fd", -1) < 0:
            raise OCIProcessError("OCI executor namespace manager is closed")

    def prove_quiescent(self) -> OCIQuiescenceReceipt:
        """Fail closed unless this executor has no lease, resource, or container."""

        self._require_namespace_owner()

        def entries(root: Path) -> tuple[str, ...]:
            if root.is_symlink() or not root.is_dir():
                raise OCIProcessError("OCI executor namespace is not a real directory")
            return tuple(sorted(path.name for path in root.iterdir()))

        leases = entries(self.leases_root)
        resources = entries(self.resources_root)
        listed = self._run_control(
            (
                self.docker_binary,
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                f"label={EXECUTOR_LABEL}={self.executor_id}",
                "--format={{.ID}}",
            ),
            allow_failure=False,
        )
        if listed.stderr.strip():
            raise OCIProcessError("executor-labelled container listing emitted stderr")
        try:
            containers = tuple(
                row for row in listed.stdout.decode("ascii", errors="strict").splitlines()
                if row
            )
        except UnicodeError as exc:
            raise OCIProcessError(f"executor-labelled container listing is not ASCII: {exc}") from None
        if len(set(containers)) != len(containers) or any(
            _CID.fullmatch(row) is None for row in containers
        ):
            raise OCIProcessError("executor-labelled container listing is malformed")
        # Close a late registration/removal race around the container query.
        if (leases, resources) != (entries(self.leases_root), entries(self.resources_root)):
            raise OCIProcessError("OCI executor namespace changed during quiescence proof")
        if leases or resources or containers:
            raise OCIProcessError("OCI executor is not quiescent")
        try:
            observed = float(self.clock())
        except Exception as exc:
            raise OCIProcessError(f"OCI quiescence clock failed: {exc}") from None
        if not math.isfinite(observed) or observed < 0:
            raise OCIProcessError("OCI quiescence clock is invalid")
        self._quiescence_sequence += 1
        return OCIQuiescenceReceipt(
            "optima.oci-quiescence.v1",
            self.executor_id,
            self.manager_instance_id,
            self.namespace_digest,
            self._quiescence_sequence,
            observed,
            leases,
            resources,
            containers,
        )

    def close(self) -> None:
        """Release process-wide ownership; normally only controller exit does this."""

        fd = getattr(self, "_namespace_lock_fd", -1)
        if fd >= 0:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            self._namespace_lock_fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def register(
        self,
        *,
        lease_id: str,
        container_name: str,
        mount_relpaths: Iterable[str] = (),
        stage_relpaths: Iterable[str] = (),
    ) -> OCILease:
        self._require_namespace_owner()
        lease_id = _simple_id(lease_id, field="lease_id")
        container_name = _simple_id(container_name, field="container_name")
        reservation_env = os.environ.get(GPU_RESERVATION_ENV)
        gpu_reservation_id = (
            None if reservation_env is None else _gpu_reservation_id(reservation_env)
        )
        mount_rows = tuple(
            _relative_resource(value, field="mount path") for value in mount_relpaths
        )
        stage_rows = tuple(
            _relative_resource(value, field="stage path") for value in stage_relpaths
        )
        if len(set(mount_rows + stage_rows)) != len(mount_rows) + len(stage_rows):
            raise OCIProcessError("lease resource paths must be unique")
        resource_root = self.resources_root / lease_id
        record_path = self.leases_root / f"{lease_id}.json"
        if resource_root.exists() or resource_root.is_symlink() or record_path.exists():
            raise OCIProcessError(f"OCI lease already exists: {lease_id}")
        resource_root.mkdir(mode=0o700)
        mount_paths = tuple(resource_root.joinpath(*PurePosixPath(row).parts) for row in mount_rows)
        stage_paths = tuple(resource_root.joinpath(*PurePosixPath(row).parts) for row in stage_rows)
        cid_path = resource_root / "container.cid"
        payload = {
            "schema": LEASE_SCHEMA,
            "executor_id": self.executor_id,
            "lease_id": lease_id,
            "container_name": container_name,
            "mount_relpaths": list(mount_rows),
            "stage_relpaths": list(stage_rows),
        }
        if gpu_reservation_id is not None:
            payload["gpu_reservation_id"] = gpu_reservation_id
        temporary = self.leases_root / (
            f".{lease_id}.{secrets.token_hex(16)}.tmp"
        )
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(_canonical_json(payload) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Hard-link publication is atomic and no-replace.  The temporary is
            # already complete/fsynced; a crash before the link leaves no canonical
            # record, while a crash after it leaves a complete record.
            os.link(temporary, record_path, follow_symlinks=False)
            temporary.unlink()
            self._fsync_directory(self.leases_root)
        except BaseException:
            shutil.rmtree(resource_root, ignore_errors=True)
            record_path.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            raise
        return OCILease(
            self.executor_id,
            lease_id,
            container_name,
            self.recovery_root,
            resource_root,
            record_path,
            cid_path,
            mount_paths,
            stage_paths,
            gpu_reservation_id,
        )

    def mount_tmpfs(
        self,
        lease: OCILease,
        path: str | Path,
        *,
        size_bytes: int,
        inode_limit: int,
        uid: int,
        gid: int,
        executable: bool = False,
    ) -> Path:
        """Mount one quota-backed tmpfs declared by ``lease``.

        Build/runtime callers must not use an ordinary host directory for hostile
        writable state: post-hoc copy bounds do not prevent a compiler or worker
        from filling the host filesystem before publication.  The lease record is
        durable before this method runs, so controller-restart recovery can unmount
        the exact path.
        """
        if not isinstance(lease, OCILease) or lease.executor_id != self.executor_id:
            raise OCIProcessError("tmpfs lease does not belong to this executor")
        selected = _within(lease.resource_root, Path(path), field="mount path")
        if selected not in lease.mount_paths:
            raise OCIProcessError("tmpfs path was not declared in the lease record")
        for field, value, low, high in (
            ("size_bytes", size_bytes, 1 << 20, 1 << 40),
            ("inode_limit", inode_limit, 1_024, 10_000_000),
            ("uid", uid, 1, 2_147_483_647),
            ("gid", gid, 1, 2_147_483_647),
        ):
            if type(value) is not int or not low <= value <= high:
                raise OCIProcessError(f"tmpfs {field} is outside its hard bound")
        if type(executable) is not bool:
            raise OCIProcessError("tmpfs executable policy must be boolean")
        if selected.exists() or selected.is_symlink():
            raise OCIProcessError("tmpfs mountpoint already exists")
        selected.mkdir(parents=True, mode=0o700)
        options = [
            "rw",
            "nosuid",
            "nodev",
            "exec" if executable else "noexec",
            f"size={size_bytes}",
            f"nr_inodes={inode_limit}",
            "mode=0700",
            f"uid={uid}",
            f"gid={gid}",
        ]
        source = f"optima-{self.executor_id}-{lease.lease_id}"
        try:
            result = self._run_control(
                (
                    "/usr/bin/mount",
                    "-t",
                    "tmpfs",
                    "-o",
                    ",".join(options),
                    source,
                    str(selected),
                ),
                allow_failure=False,
            )
            if result.stderr.strip() or not os.path.ismount(selected):
                raise OCIProcessError("could not prove quota-backed tmpfs mount")
            info = selected.stat()
            if (
                stat.S_IMODE(info.st_mode) != 0o700
                or info.st_uid != uid
                or info.st_gid != gid
            ):
                raise OCIProcessError("tmpfs owner or mode differs from lease policy")
        except BaseException:
            if os.path.ismount(selected):
                try:
                    self._run_control(("/usr/bin/umount", str(selected)), allow_failure=True)
                except OCIProcessError:
                    pass
            if not os.path.ismount(selected):
                shutil.rmtree(selected, ignore_errors=True)
            raise
        return selected

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _validate_diagnostics_root(self) -> None:
        try:
            info = self.diagnostics_root.lstat()
        except OSError as exc:
            raise OCIProcessError(
                f"cannot reopen OCI diagnostics root: {exc}"
            ) from None
        observed = (
            info.st_dev,
            info.st_ino,
            info.st_uid,
            info.st_gid,
            stat.S_IMODE(info.st_mode),
        )
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or observed != self._diagnostics_identity
        ):
            raise OCIProcessError("OCI diagnostics root identity changed")

    def _new_stderr_artifact_writer(
        self, lease: OCILease
    ) -> _StderrArtifactWriter:
        if not isinstance(lease, OCILease) or lease.executor_id != self.executor_id:
            raise OCIProcessError("stderr artifact lease belongs to another executor")
        self._validate_diagnostics_root()
        return _StderrArtifactWriter(self, lease)

    def _stderr_capture(
        self, stream: BinaryIO | None, lease: OCILease
    ) -> _AttachedStderrCapture:
        if stream is None:
            return _AttachedStderrCapture(stream)
        try:
            writer = self._new_stderr_artifact_writer(lease)
        except (OSError, OCIProcessError) as exc:
            return _AttachedStderrCapture(
                stream,
                artifact_error=_AttachedStderrCapture._bounded_error(exc),
            )
        return _AttachedStderrCapture(stream, writer)

    def _publish_stderr_receipt(
        self, receipt: OCIStderrArtifactReceipt
    ) -> None:
        if type(receipt) is not OCIStderrArtifactReceipt:
            raise OCIProcessError("stderr artifact publication is not typed")
        self._validate_diagnostics_root()
        if (
            receipt.executor_id != self.executor_id
            or receipt.artifact_path.parent != self.diagnostics_root
            or receipt.receipt_path.parent != self.diagnostics_root
        ):
            raise OCIProcessError("stderr artifact publication escapes its executor root")
        temporary = self.diagnostics_root / (
            f".{receipt.lease_id}.{secrets.token_hex(16)}.receipt.tmp"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = -1
        temporary_created = False
        receipt_linked = False
        try:
            fd = os.open(temporary, flags, 0o600)
            temporary_created = True
            os.fchmod(fd, 0o600)
            _StderrArtifactWriter._write_all(fd, receipt.receipt_bytes)
            os.fsync(fd)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size != len(receipt.receipt_bytes)
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != receipt.owner_uid
                or info.st_gid != receipt.owner_gid
            ):
                raise OCIProcessError("stderr receipt is not a private regular file")
            os.close(fd)
            fd = -1
            os.link(temporary, receipt.receipt_path, follow_symlinks=False)
            receipt_linked = True
            temporary.unlink()
            temporary_created = False
            self._fsync_directory(self.diagnostics_root)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            if receipt_linked:
                receipt.receipt_path.unlink(missing_ok=True)
            if temporary_created:
                temporary.unlink(missing_ok=True)
            raise

    def reopen_stderr_artifact(
        self, receipt: OCIStderrArtifactReceipt
    ) -> Path:
        """Reopen and authenticate one persisted failure diagnostic."""

        if type(receipt) is not OCIStderrArtifactReceipt:
            raise OCIProcessError("stderr artifact receipt is not typed")
        self._validate_diagnostics_root()
        if (
            receipt.executor_id != self.executor_id
            or receipt.artifact_path.parent != self.diagnostics_root
            or receipt.receipt_path.parent != self.diagnostics_root
        ):
            raise OCIProcessError("stderr artifact receipt escapes its executor root")

        def open_regular(path: Path) -> tuple[int, os.stat_result]:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(path, flags)
            except OSError as exc:
                raise OCIProcessError(
                    f"cannot securely reopen stderr artifact: {exc}"
                ) from None
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != receipt.owner_uid
                or info.st_gid != receipt.owner_gid
            ):
                os.close(fd)
                raise OCIProcessError("stderr artifact is not a private regular file")
            return fd, info

        receipt_fd, receipt_info = open_regular(receipt.receipt_path)
        try:
            if receipt_info.st_size != len(receipt.receipt_bytes):
                raise OCIProcessError("stderr artifact receipt size changed")
            chunks: list[bytes] = []
            remaining = len(receipt.receipt_bytes) + 1
            while remaining:
                try:
                    chunk = os.read(receipt_fd, remaining)
                except InterruptedError:
                    continue
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if payload != receipt.receipt_bytes:
                raise OCIProcessError("stderr artifact receipt content changed")
        finally:
            os.close(receipt_fd)

        artifact_fd, artifact_info = open_regular(receipt.artifact_path)
        observed = hashlib.sha256()
        observed_bytes = 0
        try:
            while True:
                chunk = os.read(artifact_fd, _ATTACHED_STDERR_READ_BYTES)
                if not chunk:
                    break
                observed.update(chunk)
                observed_bytes += len(chunk)
                if observed_bytes > ATTACHED_STDERR_ARTIFACT_MAX_BYTES:
                    raise OCIProcessError("stderr artifact exceeds its hard cap")
        finally:
            os.close(artifact_fd)
        if (
            artifact_info.st_size != receipt.artifact_bytes
            or observed_bytes != receipt.artifact_bytes
            or observed.hexdigest() != receipt.artifact_sha256
        ):
            raise OCIProcessError("stderr artifact content changed")
        return receipt.artifact_path

    def _run_control(self, argv: tuple[str, ...], *, allow_failure: bool) -> CommandResult:
        result = self.runner(argv, timeout_s=30.0, max_output_bytes=4096)
        if not isinstance(result, CommandResult):
            raise OCIProcessError("OCI command runner returned an invalid result")
        if result.returncode != 0 and not allow_failure:
            raise OCIProcessError(
                f"OCI control command exited {result.returncode}: "
                + result.stderr[:256].decode("utf-8", errors="replace")
            )
        return result

    def _listed_container_id(self, lease: OCILease) -> str | None:
        listed = self._run_control(
            (
                self.docker_binary,
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                f"name=^/{lease.container_name}$",
                "--format={{.ID}}",
            ),
            allow_failure=False,
        )
        if listed.stderr.strip():
            raise OCIProcessError("container listing emitted stderr")
        try:
            rows = listed.stdout.decode("ascii", errors="strict").splitlines()
        except UnicodeError as exc:
            raise OCIProcessError(f"container listing is not ASCII: {exc}") from None
        if not rows:
            return None
        if len(rows) != 1 or _CID.fullmatch(rows[0]) is None:
            raise OCIProcessError("container listing returned a malformed identity")
        return rows[0]

    def _verify_container_lease(self, lease: OCILease, container_id: str) -> None:
        inspected = self._run_control(
            (
                self.docker_binary,
                "container",
                "inspect",
                _INSPECT_FORMAT,
                container_id,
            ),
            allow_failure=False,
        )
        if inspected.stderr.strip():
            raise OCIProcessError("container inspection emitted stderr")
        try:
            row = json.loads(inspected.stdout)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise OCIProcessError(f"container inspection is malformed: {exc}") from None
        if not isinstance(row, dict) or set(row) != {"Id", "Name", "Labels"}:
            raise OCIProcessError("container inspection schema mismatch")
        labels = row["Labels"]
        if (
            row["Id"] != container_id
            or row["Name"] != f"/{lease.container_name}"
            or not isinstance(labels, dict)
            or labels.get(EXECUTOR_LABEL) != lease.executor_id
            or labels.get(LEASE_LABEL) != lease.lease_id
            or (
                lease.gpu_reservation_id is not None
                and labels.get(GPU_RESERVATION_LABEL) != lease.gpu_reservation_id
            )
        ):
            raise OCIProcessError(
                "refusing to remove a container without the exact lease labels"
            )
        try:
            if lease.cid_path.is_file() and not lease.cid_path.is_symlink():
                cid = lease.cid_path.read_text(encoding="ascii").strip()
                if _CID.fullmatch(cid) is None or cid != container_id:
                    raise OCIProcessError("container CID file differs from lease identity")
        except (OSError, UnicodeError) as exc:
            raise OCIProcessError(f"cannot verify lease CID file: {exc}") from None

    def force_remove_container(self, lease: OCILease) -> None:
        container_id = self._listed_container_id(lease)
        if container_id is None:
            return
        self._verify_container_lease(lease, container_id)
        try:
            self._run_control(
                (self.docker_binary, "rm", "--force", container_id),
                allow_failure=True,
            )
        except OCIProcessError:
            # The independent bounded listing below is authoritative. A timed-out
            # remove may still have completed in the daemon.
            pass
        if self._listed_container_id(lease) is not None:
            raise OCIProcessError("lease container still exists after forced removal")

    @staticmethod
    def _terminate_client(process: subprocess.Popen[bytes]) -> None:
        # The attached-client API deliberately exposes no reaping poll.  A non-null
        # cached return code therefore means another manager path already waited;
        # avoid signalling a PID/process-group identifier that may have been reused.
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            process.terminate()
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
        process.wait(timeout=10)

    def _reap_failed_process(
        self, lease: OCILease, process: subprocess.Popen[bytes]
    ) -> None:
        failures: list[BaseException] = []
        try:
            self.force_remove_container(lease)
        except BaseException as exc:
            failures.append(exc)
        try:
            self._terminate_client(process)
        except BaseException as exc:
            failures.append(exc)
        # Docker may create the named container after the first absence proof but
        # before its client is terminated.  Only the proof after client death closes
        # that race; it must run even when the first cleanup reported success.
        try:
            self.force_remove_container(lease)
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise OCIProcessError(
                "OCI failure cleanup could not prove both client and container removal"
            ) from failures[0]

    def spawn_attached(
        self,
        lease: OCILease,
        argv: tuple[str, ...],
    ) -> OCIAttachedClient:
        """Spawn one foreground byte-stream client under a durable OCI lease."""
        if not isinstance(lease, OCILease) or lease.executor_id != self.executor_id:
            raise OCIProcessError("OCI lease does not belong to this executor")
        prefix = lease.run_prefix(self.docker_binary)
        if tuple(argv[: len(prefix)]) != prefix:
            raise OCIProcessError("OCI attached argv lacks the exact lease prefix")
        if self._listed_container_id(lease) is not None:
            raise OCIProcessError("OCI lease container name is already occupied")
        try:
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
                shell=False,
            )
        except OSError as exc:
            self.force_remove_container(lease)
            raise OCIProcessError(f"could not start attached OCI client: {exc}") from None
        if process.stdin is None or process.stdout is None:
            self._reap_failed_process(lease, process)
            raise OCIProcessError("attached OCI client did not expose byte streams")
        return OCIAttachedClient(self, lease, process)

    def _close_attached(
        self, client: OCIAttachedClient, *, retain_stderr_artifact: bool
    ) -> None:
        if not isinstance(client, OCIAttachedClient) or client._manager is not self:
            raise OCIProcessError("attached OCI client belongs to another manager")
        if client.closed:
            return
        process = client._process
        failures: list[BaseException] = []
        try:
            self.force_remove_container(client.lease)
        except BaseException as exc:
            failures.append(exc)
        try:
            self._terminate_client(process)
        except BaseException as exc:
            failures.append(exc)
        client._finish_stderr_capture()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is None or stream.closed:
                continue
            try:
                stream.close()
            except BaseException as exc:
                failures.append(exc)
        client._finish_stderr_capture()
        # A foreground Docker client can create the named container after the first
        # absence proof.  Only this proof after process-group death closes that race.
        try:
            self.force_remove_container(client.lease)
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise OCIProcessError(
                "attached OCI cleanup could not prove both client and container removal"
            ) from failures[0]
        if not retain_stderr_artifact:
            try:
                client._stderr_capture.discard_artifact()
            except (OSError, OCIProcessError) as exc:
                raise OCIProcessError(
                    f"could not discard successful OCI stderr artifact: {exc}"
                ) from None
        client._closed = True

    def finalize_attached(self, client: OCIAttachedClient) -> None:
        """Normal host teardown after the caller consumed its final exact response."""
        self._close_attached(client, retain_stderr_artifact=False)

    def abort_attached(self, client: OCIAttachedClient) -> None:
        """Exceptional host teardown; cleanup authority is identical to finalize."""
        self._close_attached(client, retain_stderr_artifact=True)

    def run(
        self,
        lease: OCILease,
        argv: tuple[str, ...],
        *,
        timeout_s: float,
        stdin_bytes: bytes = b"",
    ) -> OCIProcessResult:
        if not isinstance(lease, OCILease) or lease.executor_id != self.executor_id:
            raise OCIProcessError("OCI lease does not belong to this executor")
        prefix = lease.run_prefix(self.docker_binary)
        if tuple(argv[: len(prefix)]) != prefix:
            raise OCIProcessError("OCI run argv lacks the exact lease name/cid/labels prefix")
        if not isinstance(stdin_bytes, bytes):
            raise OCIProcessError("OCI stdin must be bytes")
        if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise OCIProcessError("OCI timeout must be finite and positive")
        if self._listed_container_id(lease) is not None:
            raise OCIProcessError("OCI lease container name is already occupied")
        try:
            started = float(self.clock())
        except Exception as exc:
            raise OCIProcessError(f"OCI clock failed: {exc}") from None
        if not math.isfinite(started):
            raise OCIProcessError("OCI clock returned a non-finite value")
        stderr_read_fd: int | None = None
        stderr_write_fd: int | None = None
        stderr_stream: BinaryIO | None = None
        stderr_capture: _AttachedStderrCapture | None = None
        try:
            stderr_read_fd, stderr_write_fd = os.pipe()
            stderr_stream = os.fdopen(stderr_read_fd, "rb", buffering=0)
            stderr_read_fd = None
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                # Keep Docker stderr off ``communicate`` so the dedicated drain
                # can hash every byte while retaining only a bounded memory tail
                # and bounded private disk prefix.
                stderr=stderr_write_fd,
                close_fds=True,
                start_new_session=True,
                shell=False,
            )
        except OSError as exc:
            for fd in (stderr_read_fd, stderr_write_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            if stderr_stream is not None:
                stderr_stream.close()
            self.force_remove_container(lease)
            raise OCIProcessError(f"could not start OCI client: {exc}") from None
        finally:
            if stderr_write_fd is not None:
                try:
                    os.close(stderr_write_fd)
                except OSError:
                    pass
        stderr_capture = self._stderr_capture(stderr_stream, lease)
        timed_out = False
        try:
            process.communicate(input=stdin_bytes, timeout=float(timeout_s))
        except subprocess.TimeoutExpired:
            self._reap_failed_process(lease, process)
            timed_out = True
        except BaseException:
            self._reap_failed_process(lease, process)
            raise
        finally:
            stderr_capture.finish()
            if stderr_stream is not None:
                stderr_stream.close()
            stderr_capture.finish()
        if timed_out:
            snapshot = stderr_capture.snapshot()
            diagnostic = OCIAttachedDiagnostic(
                snapshot.stderr_tail,
                snapshot.stderr_truncated,
                snapshot.capture_complete,
                snapshot.capture_error,
                process.returncode,
                snapshot.stream_bytes,
                snapshot.stream_sha256,
                snapshot.artifact,
            )
            raise OCIProcessTimeout(
                f"OCI process exceeded its {float(timeout_s):g}s deadline",
                diagnostic,
            ) from None
        self.force_remove_container(lease)
        if int(process.returncode) == 0:
            stderr_capture.discard_artifact()
        diagnostic = stderr_capture.snapshot()
        diagnostic = OCIAttachedDiagnostic(
            diagnostic.stderr_tail,
            diagnostic.stderr_truncated,
            diagnostic.capture_complete,
            diagnostic.capture_error,
            int(process.returncode),
            diagnostic.stream_bytes,
            diagnostic.stream_sha256,
            diagnostic.artifact,
        )
        try:
            finished = float(self.clock())
        except Exception as exc:
            raise OCIProcessError(f"OCI clock failed: {exc}") from None
        if not math.isfinite(finished) or finished < started:
            raise OCIProcessError("OCI clock moved backwards or became non-finite")
        return OCIProcessResult(
            int(process.returncode), finished - started, diagnostic
        )

    def run_capture(
        self,
        lease: OCILease,
        argv: tuple[str, ...],
        *,
        timeout_s: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        capture_runner: CaptureRunner,
    ) -> CommandResult:
        """Run an attached bounded-output client under the same lease authority.

        The injected capture primitive owns live pipe caps and process-group reaping;
        this method adds durable container labels, pre-name absence, forced removal,
        and restart-recoverable lease state around it.
        """
        if not isinstance(lease, OCILease) or lease.executor_id != self.executor_id:
            raise OCIProcessError("OCI lease does not belong to this executor")
        prefix = lease.run_prefix(self.docker_binary)
        if tuple(argv[: len(prefix)]) != prefix:
            raise OCIProcessError("OCI capture argv lacks the exact lease prefix")
        if (
            type(timeout_s) not in (int, float)
            or not math.isfinite(float(timeout_s))
            or timeout_s <= 0
        ):
            raise OCIProcessError("OCI capture timeout must be finite and positive")
        for field, value in (
            ("max_stdout_bytes", max_stdout_bytes),
            ("max_stderr_bytes", max_stderr_bytes),
        ):
            if type(value) is not int or not 1 <= value <= 64 << 20:
                raise OCIProcessError(f"OCI capture {field} is outside its hard bound")
        if self._listed_container_id(lease) is not None:
            raise OCIProcessError("OCI lease container name is already occupied")
        try:
            result = capture_runner(
                argv,
                timeout_s=float(timeout_s),
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            )
        except BaseException:
            self.force_remove_container(lease)
            raise
        self.force_remove_container(lease)
        if (
            not isinstance(result, CommandResult)
            or len(result.stdout) > max_stdout_bytes
            or len(result.stderr) > max_stderr_bytes
        ):
            raise OCIProcessError("OCI capture runner returned invalid bounded output")
        return result

    def _lease_from_record(self, record_path: Path) -> OCILease:
        if record_path.is_symlink() or not record_path.is_file():
            raise OCIProcessError(f"lease record is not a regular file: {record_path}")
        try:
            info = record_path.stat()
            raw = record_path.read_bytes()
            row = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OCIProcessError(f"invalid OCI lease record: {exc}") from None
        required_fields = {
            "schema",
            "executor_id",
            "lease_id",
            "container_name",
            "mount_relpaths",
            "stage_relpaths",
        }
        if (
            not isinstance(row, dict)
            or set(row) not in (
                required_fields,
                required_fields | {"gpu_reservation_id"},
            )
            or row["schema"] != LEASE_SCHEMA
        ):
            raise OCIProcessError("OCI lease record schema mismatch")
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or raw != _canonical_json(row) + b"\n"
        ):
            raise OCIProcessError("OCI lease record is not canonical and single-linked")
        executor_id = _simple_id(row["executor_id"], field="executor_id")
        lease_id = _simple_id(row["lease_id"], field="lease_id")
        container_name = _simple_id(row["container_name"], field="container_name")
        gpu_reservation_id = (
            None
            if "gpu_reservation_id" not in row
            else _gpu_reservation_id(row["gpu_reservation_id"])
        )
        if record_path.name != f"{lease_id}.json":
            raise OCIProcessError("OCI lease filename and identity differ")
        if executor_id != self.executor_id:
            raise OCIProcessError("OCI lease record is in the wrong executor namespace")
        for field in ("mount_relpaths", "stage_relpaths"):
            if not isinstance(row[field], list) or not all(isinstance(item, str) for item in row[field]):
                raise OCIProcessError(f"OCI lease {field} is invalid")
        resource_root = self.resources_root / lease_id
        mount_paths = tuple(
            _within(
                resource_root,
                resource_root.joinpath(*PurePosixPath(_relative_resource(item, field="mount path")).parts),
                field="mount path",
            )
            for item in row["mount_relpaths"]
        )
        stage_paths = tuple(
            _within(
                resource_root,
                resource_root.joinpath(*PurePosixPath(_relative_resource(item, field="stage path")).parts),
                field="stage path",
            )
            for item in row["stage_relpaths"]
        )
        return OCILease(
            executor_id,
            lease_id,
            container_name,
            self.recovery_root,
            resource_root,
            record_path,
            resource_root / "container.cid",
            mount_paths,
            stage_paths,
            gpu_reservation_id,
        )

    def release(self, lease: OCILease) -> None:
        if lease.executor_id != self.executor_id:
            raise OCIProcessError("cannot release another executor's lease")
        self.force_remove_container(lease)
        for path in sorted(lease.mount_paths, key=lambda item: len(item.parts), reverse=True):
            _within(lease.resource_root, path, field="mount path")
            if os.path.ismount(path):
                result = self._run_control(("/usr/bin/umount", str(path)), allow_failure=False)
                if result.stderr.strip() or os.path.ismount(path):
                    raise OCIProcessError(f"could not prove tmpfs unmounted: {path}")
        for path in lease.stage_paths:
            _within(lease.resource_root, path, field="stage path")
        if lease.resource_root.is_symlink():
            raise OCIProcessError("lease resource root must not be a symlink")
        if lease.resource_root.exists():
            shutil.rmtree(lease.resource_root, ignore_errors=False)
        lease.record_path.unlink(missing_ok=True)
        self._fsync_directory(self.leases_root)

    def recover_stale(self, *, active_lease_ids: Iterable[str] = ()) -> tuple[str, ...]:
        active = {_simple_id(value, field="active lease id") for value in active_lease_ids}
        recovered: list[str] = []
        # Registration publishes by hard-linking a complete temporary record and
        # then unlinking the temporary.  A kill between those operations leaves the
        # canonical record with nlink=2.  Remove executor-owned temporaries first so
        # canonical record validation observes the intended single-link shape.
        for temporary in sorted(self.leases_root.glob(".*.tmp")):
            if temporary.is_symlink() or not temporary.is_file():
                raise OCIProcessError("stale lease temporary is not a regular file")
            temporary.unlink()
        for record_path in sorted(self.leases_root.glob("*.json")):
            lease = self._lease_from_record(record_path)
            if lease.executor_id != self.executor_id or lease.lease_id in active:
                continue
            self.release(lease)
            recovered.append(lease.lease_id)
        # Registration publishes a complete record before returning.  Therefore an
        # unrecorded resource directory or temporary record can only be residue from
        # a controller crash during registration; no container/mount was yet handed
        # to a caller.  The executor-specific namespace prevents cross-owner cleanup.
        recorded = {path.stem for path in self.leases_root.glob("*.json")}
        for resource in sorted(self.resources_root.iterdir()):
            lease_id = _simple_id(resource.name, field="orphan lease id")
            if lease_id in recorded or lease_id in active:
                continue
            if resource.is_symlink() or not resource.is_dir():
                raise OCIProcessError("orphan lease resource is not a real directory")
            shutil.rmtree(resource)
            recovered.append(lease_id)
        return tuple(recovered)
