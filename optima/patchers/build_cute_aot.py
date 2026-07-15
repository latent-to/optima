"""Shared hermetic CuTe compiler-child helpers for the device-only provider.

The bundle declares factories, not compiler commands.  Each factory executes in a
separate child with only its declared validator profile values and returns an exact
``CuteAOTCompileRequest``.  The child receives no stage path or stage environment;
the parent snapshots the mounted stage across candidate execution (catching even a
hard-coded access), accepts only the exact product set requested by the device
provider, and never loads candidate host code.
"""

from __future__ import annotations

import hashlib
import os
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath

from optima.cute_aot import (
    CUTE_AOT_CHILD_SCHEMA,
    CuteAOTError,
    deterministic_export_names,
    export_launch_plan,
)
from optima.stack_identity import canonical_json_bytes, require_sha256_hex


_MAX_PRODUCT_BYTES = 1 << 30
_MAX_PATCHER_BYTES = 4 << 20
_PR_SET_CHILD_SUBREAPER = 36


def _log(message: str) -> None:
    print(f"[optima.build_cute_aot] {message}", flush=True)


def _digest(value: object, *, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise CuteAOTError(str(exc)) from None
    if result == "0" * 64:
        raise CuteAOTError(f"{field} must not be the all-zero digest")
    return result


def _phase() -> str:
    phase = os.environ.get("OPTIMA_REBUILD_PHASE", "all").strip().lower()
    if phase not in {"all", "build", "load"}:
        raise CuteAOTError(f"unsupported CuTe AOT rebuild phase: {phase!r}")
    return phase


def _absolute_directory_env(name: str) -> Path:
    raw = os.environ.get(name, "").strip()
    path = Path(raw)
    if not raw or not path.is_absolute():
        raise CuteAOTError(f"{name} must be an existing absolute directory")
    try:
        info = path.lstat()
    except OSError as exc:
        raise CuteAOTError(f"cannot inspect {name}: {exc}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CuteAOTError(f"{name} must be a non-symlink directory")
    return path


def _stable_digest(path: Path, *, maximum: int) -> tuple[str, int]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CuteAOTError(f"cannot inspect CuTe AOT product {path}: {exc}") from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 1 <= before.st_size <= maximum
    ):
        raise CuteAOTError(f"CuTe AOT product has an unsafe shape: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 << 20), b""):
                digest.update(chunk)
        after = path.lstat()
    except OSError as exc:
        raise CuteAOTError(f"cannot hash CuTe AOT product {path}: {exc}") from None
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise CuteAOTError(f"CuTe AOT product changed while hashing: {path}")
    return digest.hexdigest(), before.st_size


def _patcher_sha256() -> str:
    digest, _size = _stable_digest(Path(__file__).resolve(), maximum=_MAX_PATCHER_BYTES)
    return digest


def _stage_snapshot(stage: Path) -> tuple[tuple[object, ...], ...]:
    """Byte/inode snapshot used to detect candidate writes to the shared stage."""

    rows: list[tuple[object, ...]] = []
    try:
        paths = sorted(stage.rglob("*"), key=lambda path: path.relative_to(stage).as_posix())
        for path in paths:
            relative = path.relative_to(stage).as_posix()
            info = path.lstat()
            common: tuple[object, ...] = (
                relative,
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_nlink,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
            if stat.S_ISLNK(info.st_mode):
                raise CuteAOTError("native artifact stage contains a symlink")
            if stat.S_ISREG(info.st_mode):
                digest, _size = _stable_digest(path, maximum=_MAX_PRODUCT_BYTES)
                rows.append((*common, "file", digest))
            elif stat.S_ISDIR(info.st_mode):
                rows.append((*common, "directory"))
            else:
                raise CuteAOTError("native artifact stage contains a non-file object")
    except OSError as exc:
        raise CuteAOTError(f"cannot snapshot native artifact stage: {exc}") from None
    return tuple(rows)


def _safe_bundle_source(bundle: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or "." in path.parts or ".." in path.parts or str(path) != relative:
        raise CuteAOTError("CuTe AOT source path is not canonical")
    requested = bundle.joinpath(*path.parts)
    try:
        if requested.is_symlink():
            raise CuteAOTError("CuTe AOT source must not be a symlink")
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise CuteAOTError(f"CuTe AOT source is unavailable: {exc}") from None
    if bundle not in resolved.parents or not resolved.is_file():
        raise CuteAOTError("CuTe AOT source escapes the engine tree")
    return resolved


def _copy_product(source: Path, destination: Path) -> tuple[str, int]:
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise CuteAOTError("CuTe AOT destination directory is unavailable")
    try:
        before = source.lstat()
    except OSError as exc:
        raise CuteAOTError(f"cannot inspect CuTe AOT compiler output: {exc}") from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 1 <= before.st_size <= _MAX_PRODUCT_BYTES
    ):
        raise CuteAOTError("CuTe AOT compiler output has an unsafe shape")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(destination, flags, 0o400)
    digest = hashlib.sha256()
    try:
        read_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            source_fd = os.open(source, read_flags)
        except OSError as exc:
            raise CuteAOTError(
                f"cannot open CuTe AOT compiler output without following links: {exc}"
            ) from None
        with os.fdopen(source_fd, "rb", closefd=True) as reader:
            opened = os.fstat(reader.fileno())
            source_identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            opened_identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            if source_identity != opened_identity:
                raise CuteAOTError("CuTe AOT compiler output changed while opening")
            remaining = before.st_size
            while True:
                chunk = reader.read(min(4 << 20, remaining)) if remaining else b""
                if not chunk and remaining:
                    raise CuteAOTError("CuTe AOT compiler output was truncated")
                if not remaining:
                    break
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise CuteAOTError("CuTe AOT product copy stalled")
                    view = view[written:]
                remaining -= len(chunk)
            if reader.read(1):
                raise CuteAOTError("CuTe AOT compiler output grew while copying")
            after_fd = os.fstat(reader.fileno())
        after_path = source.lstat()
        if source_identity != (
            after_fd.st_dev,
            after_fd.st_ino,
            after_fd.st_mode,
            after_fd.st_nlink,
            after_fd.st_size,
            after_fd.st_mtime_ns,
            after_fd.st_ctime_ns,
        ) or source_identity != (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_mode,
            after_path.st_nlink,
            after_path.st_size,
            after_path.st_mtime_ns,
            after_path.st_ctime_ns,
        ):
            raise CuteAOTError("CuTe AOT compiler output changed while copying")
        os.fsync(fd)
    finally:
        os.close(fd)
    size = before.st_size
    digest_hex = digest.hexdigest()
    copied_digest, copied_size = _stable_digest(destination, maximum=_MAX_PRODUCT_BYTES)
    if (copied_digest, copied_size) != (digest_hex, size):
        raise CuteAOTError("CuTe AOT copied product differs from compiler output")
    return digest_hex, size


def _arm_child_subreaper() -> None:
    if sys.platform != "linux":
        raise CuteAOTError("authoritative CuTe AOT compiler isolation requires Linux")
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise CuteAOTError(
            f"cannot arm CuTe compiler child subreaper: errno {error}"
        )


def _proc_parent_map() -> dict[int, int]:
    """Return the visible Linux PID parent relation or fail closed."""

    parents: dict[int, int] = {}
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as exc:
        raise CuteAOTError(f"cannot enumerate compiler PID namespace: {exc}") from None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            rows = (entry / "status").read_text(encoding="ascii").splitlines()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, UnicodeError) as exc:
            raise CuteAOTError(
                f"cannot inspect compiler PID {entry.name}: {exc}"
            ) from None
        pid_rows = [row for row in rows if row.startswith("Pid:\t")]
        parent_rows = [row for row in rows if row.startswith("PPid:\t")]
        if len(pid_rows) != 1 or len(parent_rows) != 1:
            raise CuteAOTError("compiler PID status has an unexpected shape")
        try:
            pid = int(pid_rows[0].split("\t", 1)[1])
            parent = int(parent_rows[0].split("\t", 1)[1])
        except (IndexError, ValueError):
            raise CuteAOTError("compiler PID status is malformed") from None
        if pid > 0 and parent >= 0:
            parents[pid] = parent
    return parents


def _compiler_descendants() -> set[int]:
    """Find every descendant, including fork+setsid escapees.

    The patcher is armed as a child subreaper before spawning the compiler.  Once
    an intermediate exits, every orphan remains (or becomes) our descendant, so a
    full parent-graph walk is stronger than process-group membership.
    """

    root = os.getpid()
    parents = _proc_parent_map()
    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid == root or pid in descendants:
                continue
            if parent == root or parent in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _signal_compiler_descendants(sig: int) -> None:
    for pid in sorted(_compiler_descendants(), reverse=True):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except OSError as exc:
            raise CuteAOTError(
                f"cannot signal compiler descendant {pid}: {exc}"
            ) from None


def _reap_adopted_children() -> None:
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid > 0:
            continue
        return


def _quiesce_compiler_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill/reap the complete compiler descendant tree before any stage read."""

    _signal_compiler_descendants(signal.SIGTERM)
    if process.poll() is None:
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + 0.25
    while _compiler_descendants() and time.monotonic() < deadline:
        time.sleep(0.01)

    _signal_compiler_descendants(signal.SIGKILL)
    if process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            raise CuteAOTError("direct CuTe compiler child resisted SIGKILL") from None

    # A killed subreaper/fork daemon can expose another generation only after its
    # parent is reaped.  Iterate kill -> reap -> enumerate until the namespace is
    # stably empty; never publish when it fails to quiesce.
    deadline = time.monotonic() + 2.0
    empty_observations = 0
    while time.monotonic() < deadline:
        _reap_adopted_children()
        descendants = _compiler_descendants()
        if not descendants:
            empty_observations += 1
            if empty_observations >= 3:
                return
        else:
            empty_observations = 0
            for pid in sorted(descendants, reverse=True):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
                except OSError as exc:
                    raise CuteAOTError(
                        f"cannot kill compiler descendant {pid}: {exc}"
                    ) from None
        time.sleep(0.01)
    raise CuteAOTError("CuTe compiler descendants did not quiesce")


def _bounded_stderr_text(raw: bytearray) -> str:
    decoded = bytes(raw).decode("utf-8", errors="replace")
    return "".join(
        character
        if character in {"\n", "\t"} or 0x20 <= ord(character) <= 0x7E
        else "?"
        for character in decoded
    ).strip()


def _run_isolated_compiler(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> tuple[int, str]:
    """Run and fully reap one compiler session while retaining a bounded error tail."""

    _arm_child_subreaper()
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except OSError as exc:
        raise CuteAOTError(f"cannot start isolated CuTe AOT compiler: {exc}") from None
    assert process.stderr is not None
    stderr_tail = bytearray()

    def drain_stderr() -> None:
        while True:
            try:
                chunk = process.stderr.read(8 << 10)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            stderr_tail.extend(chunk)
            if len(stderr_tail) > (64 << 10):
                del stderr_tail[: len(stderr_tail) - (64 << 10)]

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()
    timed_out = False
    returncode: int | None = None
    cleanup_error: CuteAOTError | None = None
    try:
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
    finally:
        try:
            _quiesce_compiler_tree(process)
        except CuteAOTError as exc:
            cleanup_error = exc
        reader.join(timeout=1.0)
        if reader.is_alive():
            process.stderr.close()
            reader.join(timeout=1.0)
        if reader.is_alive():
            raise CuteAOTError("CuTe compiler stderr drain did not quiesce")
        process.stderr.close()
    if cleanup_error is not None:
        raise cleanup_error
    tail = _bounded_stderr_text(stderr_tail)
    if timed_out:
        detail = f"; stderr tail: {tail}" if tail else ""
        raise CuteAOTError(
            f"isolated CuTe AOT compiler timed out after {timeout_seconds}s{detail}"
        )
    if returncode is None:
        raise CuteAOTError("isolated CuTe AOT compiler has no exit status")
    return returncode, tail


def _child_environment(private_tmp: Path) -> dict[str, str]:
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_DISABLE_FILE_CACHING": "1",
        "HOME": str(private_tmp / "home"),
        "NVIDIA_VISIBLE_DEVICES": "void",
        "OPTIMA_REBUILD_CONTAINER": "1",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "TMPDIR": str(private_tmp),
    }
    for name in ("CUDA_HOME", "LD_LIBRARY_PATH"):
        value = os.environ.get(name, "").strip()
        if value:
            environment[name] = value
    return environment


def _run_build_child(
    *,
    bundle: Path,
    source: str,
    slot: str,
    variant: str,
    target_authority: dict[str, object],
    target_authority_sha256: str,
    resource_plan: dict[str, object],
    resource_plan_sha256: str,
    export,
    profile,
    private_root: Path,
    timeout_seconds: int,
    stage: Path,
    expected_suffixes: tuple[str, ...] = (".cubin",),
) -> tuple[Path, str, str, dict[str, int]]:
    if expected_suffixes != (".cubin",):
        raise CuteAOTError("CuTe compiler may publish only one retained CUBIN")
    launch_plan = export_launch_plan(export)
    artifact_id, function_prefix = deterministic_export_names(
        source=source,
        slot=slot,
        variant=variant,
        name=export.name,
        factory=export.factory,
        plan=launch_plan,
        artifact_resource_plan_sha256=resource_plan_sha256,
        artifact_target_authority_sha256=target_authority_sha256,
    )
    resolved_profile = {
        key: profile.require_int(key) for key in export.profile_inputs
    }
    child_root = private_root / artifact_id
    child_root.mkdir(mode=0o700)
    output = child_root / "output"
    output.mkdir(mode=0o700)
    (child_root / "home").mkdir(mode=0o700)
    spec = {
        "artifact_id": artifact_id,
        "artifact_target_authority": target_authority,
        "artifact_target_authority_sha256": target_authority_sha256,
        "artifact_resource_plan": resource_plan,
        "artifact_resource_plan_sha256": resource_plan_sha256,
        "bundle": str(bundle),
        "compiler_architecture": profile.compiler_architecture,
        "factory": export.factory,
        "function_prefix": function_prefix,
        "name": export.name,
        "output_directory": str(output),
        "profile_values": resolved_profile,
        "provider": export.provider,
        "launch_plan": launch_plan,
        "schema": CUTE_AOT_CHILD_SCHEMA,
        "slot": slot,
        "source": source,
        "variant": variant,
    }
    spec_path = child_root / "request.json"
    spec_path.write_bytes(canonical_json_bytes(spec) + b"\n")
    spec_path.chmod(0o400)
    before = _stage_snapshot(stage)
    returncode, stderr_tail = _run_isolated_compiler(
        [
            sys.executable,
            "-I",
            "-m",
            "optima.cute_aot_child",
            "--build-export",
            str(spec_path),
        ],
        cwd=child_root,
        environment=_child_environment(child_root),
        timeout_seconds=timeout_seconds,
    )
    after = _stage_snapshot(stage)
    if after != before:
        raise CuteAOTError("candidate CuTe AOT factory modified the native artifact stage")
    if returncode != 0:
        detail = f"; stderr tail: {stderr_tail}" if stderr_tail else ""
        raise CuteAOTError(
            f"isolated CuTe AOT compiler exited {returncode} for "
            f"{(slot, variant, export.name)!r}{detail}"
        )
    expected_names = {f"{artifact_id}{suffix}" for suffix in expected_suffixes}
    try:
        products = tuple(output.iterdir())
    except OSError as exc:
        raise CuteAOTError(f"cannot enumerate CuTe AOT child output: {exc}") from None
    if (
        len(products) != len(expected_suffixes)
        or {path.name for path in products} != expected_names
    ):
        raise CuteAOTError("isolated CuTe AOT compiler emitted unexpected products")
    for product in products:
        _stable_digest(product, maximum=_MAX_PRODUCT_BYTES)
    return output, artifact_id, function_prefix, resolved_profile
