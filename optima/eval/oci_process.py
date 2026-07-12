"""Host-owned OCI process lifecycle and crash recovery.

This module knows nothing about candidates, models, scoring, or engine protocol.  It
owns only a Docker-compatible client process, the exact container name/CID, and
lease-scoped local resources.  The same primitive is used by offline prebuild and
streaming runtime so timeout/cancellation/controller-restart cleanup has one authority.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import time
import secrets
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Protocol


LEASE_SCHEMA = "optima.oci-process-lease.v1"
EXECUTOR_LABEL = "optima.executor_id"
LEASE_LABEL = "optima.lease_id"
_SIMPLE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_CID = re.compile(r"[0-9a-f]{64}\Z")
_INSPECT_FORMAT = (
    '--format={"Id":{{json .Id}},"Name":{{json .Name}},'
    '"Labels":{{json .Config.Labels}}}'
)


class OCIProcessError(RuntimeError):
    pass


class OCIProcessTimeout(OCIProcessError):
    pass


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

    def run_prefix(self, docker_binary: str) -> tuple[str, ...]:
        return (
            docker_binary,
            "run",
            f"--name={self.container_name}",
            f"--cidfile={self.cid_path}",
            f"--label={EXECUTOR_LABEL}={self.executor_id}",
            f"--label={LEASE_LABEL}={self.lease_id}",
        )


@dataclass(frozen=True)
class OCIProcessResult:
    returncode: int
    elapsed_seconds: float


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _simple_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SIMPLE_ID.fullmatch(value) is None:
        raise OCIProcessError(f"{field} must be a lowercase simple identifier")
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
        self.leases_root = self.recovery_root / "leases" / self.executor_id
        self.resources_root = self.recovery_root / "resources" / self.executor_id
        self.leases_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.resources_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.runner = runner
        self.clock = clock

    def register(
        self,
        *,
        lease_id: str,
        container_name: str,
        mount_relpaths: Iterable[str] = (),
        stage_relpaths: Iterable[str] = (),
    ) -> OCILease:
        lease_id = _simple_id(lease_id, field="lease_id")
        container_name = _simple_id(container_name, field="container_name")
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
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

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
        try:
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
                shell=False,
            )
        except OSError as exc:
            self.force_remove_container(lease)
            raise OCIProcessError(f"could not start OCI client: {exc}") from None
        try:
            process.communicate(input=stdin_bytes, timeout=float(timeout_s))
        except subprocess.TimeoutExpired:
            self._reap_failed_process(lease, process)
            raise OCIProcessTimeout(
                f"OCI process exceeded its {float(timeout_s):g}s deadline"
            ) from None
        except BaseException:
            self._reap_failed_process(lease, process)
            raise
        self.force_remove_container(lease)
        try:
            finished = float(self.clock())
        except Exception as exc:
            raise OCIProcessError(f"OCI clock failed: {exc}") from None
        if not math.isfinite(finished) or finished < started:
            raise OCIProcessError("OCI clock moved backwards or became non-finite")
        return OCIProcessResult(int(process.returncode), finished - started)

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
        fields = {
            "schema",
            "executor_id",
            "lease_id",
            "container_name",
            "mount_relpaths",
            "stage_relpaths",
        }
        if not isinstance(row, dict) or set(row) != fields or row["schema"] != LEASE_SCHEMA:
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
