"""Trusted, candidate-free runtime image preflight.

The controller performs two shell-free Docker operations before any miner tree,
model, artifact, or GPU is exposed:

* bind an immutable ``name@sha256:...`` reference to one local image ID and
  expected OCI platform;
* run a fixed standard-library-only probe by local image ID with no network,
  mounts, GPU runtime, or Linux capabilities.

The probe inspects installed distribution metadata without importing Optima,
Torch, SGLang, or candidate code.  It hashes the installed Optima distribution
files and emits one bounded exact-schema receipt.  Every error is validator
owned; this module cannot disqualify a miner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import secrets
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Protocol

from optima.eval.oci_process import (
    CommandResult,
    OCIProcessError,
    OCIProcessManager,
)


INSPECT_SCHEMA_KEYS = frozenset(
    {"Id", "RepoDigests", "Volumes", "Os", "Architecture"}
)
CONTAINER_RECEIPT_SCHEMA = "optima-runtime-container-preflight-v2"
HOST_RECEIPT_SCHEMA = "optima-runtime-preflight-v2"
WORKER_DISTRIBUTION = "optima-harness"
WORKER_DIGEST_SCHEMA = "optima-installed-distribution-v1"
PLATFORM_DIGEST_SCHEMA = "optima-runtime-platform-v1"
MAX_INSPECT_STDOUT_BYTES = 16 * 1024
MAX_RECEIPT_STDOUT_BYTES = 16 * 1024
MAX_STDERR_BYTES = 8 * 1024

_IMAGE = re.compile(r"[a-z0-9][a-z0-9._/:+-]{0,255}@sha256:[0-9a-f]{64}\Z")
_SHA_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z._+-]{0,127}\Z")
_DISTRIBUTION = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_OCI_PLATFORM = re.compile(r"[a-z0-9][a-z0-9._-]{0,31}/[a-z0-9][a-z0-9._-]{0,31}\Z")
_SMALL_TEXT = re.compile(r"[^\x00\r\n]{0,255}\Z")
_CONTAINER_NAME = re.compile(r"optima-runtime-preflight-[0-9a-f]{20}\Z")
_PACKAGE_NAMES = (
    "cuda-python",
    "flashinfer-python",
    "nvidia-cuda-runtime-cu12",
    "torch",
    "triton",
)

_INSPECT_FORMAT = (
    '--format={"Id":{{json .Id}},"RepoDigests":{{json .RepoDigests}},'
    '"Volumes":{{json .Config.Volumes}},"Os":{{json .Os}},'
    '"Architecture":{{json .Architecture}}}'
)

# Fixed source.  No config, path, image, or miner value is interpolated into it.
# It deliberately inspects distribution metadata and files without importing
# any inspected distribution.  The limits make even a malformed trusted image
# fail closed instead of turning the controller probe into an unbounded reader.
_CONTAINER_SCRIPT = r'''import ctypes.util, hashlib, importlib.metadata, json, os, pathlib, platform, re, stat, sys, sysconfig
MAX_FILES = 4096
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
WORKER = "optima-harness"
def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()
roots = []
for raw in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib")):
    if raw:
        root = pathlib.Path(raw).resolve(strict=True)
        if root not in roots:
            roots.append(root)
        if str(root) not in sys.path:
            sys.path.append(str(root))
versions = {}
worker_dists = []
for dist in importlib.metadata.distributions(path=[str(root) for root in roots]):
    name = dist.metadata.get("Name")
    if not name:
        continue
    key = norm(name)
    prior = versions.get(key)
    if prior is not None and prior != dist.version:
        raise RuntimeError("conflicting installed distribution metadata: " + name)
    versions[key] = dist.version
    if key == norm(WORKER):
        worker_dists.append(dist)
if len(worker_dists) != 1:
    raise RuntimeError("expected exactly one installed Optima worker distribution")
dist = worker_dists[0]
files = dist.files
if files is None or not files:
    raise RuntimeError("worker distribution has no installed file inventory")
records = []
seen = set()
total = 0
has_module = False
has_metadata = False
for entry in files:
    rel_meta = pathlib.PurePosixPath(str(entry))
    if rel_meta.is_absolute() or "\x00" in str(rel_meta):
        raise RuntimeError("unsafe worker distribution file path")
    located = pathlib.Path(dist.locate_file(entry))
    try:
        resolved = located.resolve(strict=False)
    except (OSError, RuntimeError):
        raise RuntimeError("worker distribution file path cannot be resolved")
    selected = None
    for root in roots:
        try:
            selected = resolved.relative_to(root).as_posix()
            break
        except ValueError:
            pass
    # Wheel RECORD files commonly include console scripts as paths such as
    # ../../../bin/optima.  They are intentionally outside the in-image worker
    # distribution root and therefore outside this identity.  Resolve only to
    # classify them; never open an out-of-root path.
    if selected is None:
        continue
    try:
        resolved = located.resolve(strict=True)
    except (OSError, RuntimeError):
        raise RuntimeError("worker distribution file is missing")
    parts = pathlib.PurePosixPath(selected).parts
    if "__pycache__" in parts or selected.endswith((".pyc", ".pyo")):
        continue
    if selected in seen:
        raise RuntimeError("duplicate canonical worker distribution path")
    seen.add(selected)
    if located.is_symlink():
        raise RuntimeError("worker distribution contains a symlink")
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("worker distribution contains a non-regular file")
    if info.st_size < 0 or info.st_size > MAX_FILE_BYTES:
        raise RuntimeError("worker distribution file exceeds size policy")
    total += info.st_size
    if total > MAX_TOTAL_BYTES:
        raise RuntimeError("worker distribution exceeds total size policy")
    if len(records) >= MAX_FILES:
        raise RuntimeError("worker distribution exceeds file-count policy")
    digest = hashlib.sha256()
    read_count = 0
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            read_count += len(chunk)
            if read_count > info.st_size or read_count > MAX_FILE_BYTES:
                raise RuntimeError("worker distribution file changed during hashing")
            digest.update(chunk)
    after = resolved.stat()
    if read_count != info.st_size or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
        raise RuntimeError("worker distribution file changed during hashing")
    records.append([selected, info.st_size, digest.hexdigest()])
    has_module = has_module or selected == "optima/__init__.py"
    has_metadata = has_metadata or (".dist-info/" in selected and selected.endswith("/METADATA"))
if not records or not has_module or not has_metadata:
    raise RuntimeError("worker distribution inventory is incomplete")
records.sort(key=lambda record: record[0])
identity = {
    "schema": "optima-installed-distribution-v1",
    "distribution": WORKER,
    "version": dist.version,
    "files": records,
}
encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
def v(name):
    return versions.get(norm(name))
out = {
    "schema": "optima-runtime-container-preflight-v2",
    "sglang_version": v("sglang"),
    "worker": {
        "distribution": WORKER,
        "version": dist.version,
        "digest": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(records),
        "total_bytes": total,
    },
    "python": {
        "executable": os.path.realpath(sys.executable),
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
        "abi": str(sysconfig.get_config_var("SOABI") or ""),
        "platform": sysconfig.get_platform(),
        "machine": platform.machine(),
    },
    "packages": {
        "cuda-python": v("cuda-python"),
        "flashinfer-python": v("flashinfer-python"),
        "nvidia-cuda-runtime-cu12": v("nvidia-cuda-runtime-cu12"),
        "torch": v("torch"),
        "triton": v("triton"),
    },
    "cuda": {
        "cudart_library": ctypes.util.find_library("cudart"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    },
}
print(json.dumps(out, sort_keys=True, separators=(",", ":")))'''


class RuntimePreflightError(RuntimeError):
    """Trusted validator image/runtime state is invalid or unavailable."""

    validator_fault = True
    retryable = False


class Runner(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        timeout_s: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
    ) -> CommandResult: ...


@dataclass(frozen=True)
class RuntimePreflightConfig:
    """Expected path-free identity for one trusted runtime image probe."""

    image: str
    expected_oci_platform: str
    expected_python_platform: str
    expected_machine: str
    expected_python_executable: str
    expected_sglang_version: str
    expected_worker_distribution: str
    expected_worker_version: str
    expected_worker_digest: str
    uid: int
    gid: int
    docker_binary: str
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or _IMAGE.fullmatch(self.image) is None:
            raise RuntimePreflightError(
                "preflight image must be immutable lowercase name@sha256"
            )
        if (
            not isinstance(self.expected_oci_platform, str)
            or _OCI_PLATFORM.fullmatch(self.expected_oci_platform) is None
        ):
            raise RuntimePreflightError("expected OCI platform is invalid")
        for name in ("expected_python_platform", "expected_machine"):
            if _small_text_value(getattr(self, name), allow_empty=False) is None:
                raise RuntimePreflightError(f"{name} is invalid")
        if not _safe_absolute_path(self.expected_python_executable):
            raise RuntimePreflightError("expected_python_executable is invalid")
        for name in ("expected_sglang_version", "expected_worker_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or _VERSION.fullmatch(value) is None:
                raise RuntimePreflightError(f"{name} is invalid")
        if (
            not isinstance(self.expected_worker_distribution, str)
            or _DISTRIBUTION.fullmatch(self.expected_worker_distribution) is None
            or _normalize_distribution(self.expected_worker_distribution)
            != WORKER_DISTRIBUTION
        ):
            raise RuntimePreflightError(
                f"expected worker distribution must be {WORKER_DISTRIBUTION!r}"
            )
        if (
            not isinstance(self.expected_worker_digest, str)
            or _DIGEST.fullmatch(self.expected_worker_digest) is None
        ):
            raise RuntimePreflightError(
                "expected worker distribution digest must be lowercase sha256 hex"
            )
        for name in ("uid", "gid"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 2_147_483_647:
                raise RuntimePreflightError(
                    f"preflight {name} must be a fixed nonzero integer"
                )
        if not _safe_docker_binary(self.docker_binary):
            raise RuntimePreflightError(
                "docker_binary must be an absolute normalized path ending in /docker"
            )
        if (
            type(self.timeout_s) not in (int, float)
            or not math.isfinite(float(self.timeout_s))
            or not 1.0 <= float(self.timeout_s) <= 300.0
        ):
            raise RuntimePreflightError("preflight timeout must be in [1, 300] seconds")


@dataclass(frozen=True)
class RuntimePreflightReceipt:
    """Canonical preflight evidence consumed by ``EngineLaunchSpec``.

    ``launch_identity()`` is the deliberately small interface between this
    module and launch-spec construction.  It contains only the three canonical
    digests that a resolved launch must match; version and local-cache details
    remain auditable diagnostics in ``canonical_payload()``.
    """

    schema: str
    requested_image: str
    image_digest: str
    local_image_id: str
    repo_digests: tuple[str, ...]
    oci_platform: str
    platform_digest: str
    docker_binary: str
    uid: int
    gid: int
    sglang_version: str
    worker_distribution: str
    worker_version: str
    worker_distribution_digest: str
    worker_file_count: int
    worker_total_bytes: int
    python_implementation: str
    python_executable: str
    python_version: str
    python_abi: str
    python_platform: str
    machine: str
    package_versions: tuple[tuple[str, str | None], ...]
    cudart_library: str | None
    cuda_visible_devices: str
    nvidia_visible_devices: str
    security_argv_sha256: str

    def launch_identity(self) -> dict[str, str]:
        return {
            "image_digest": self.image_digest,
            "platform_digest": self.platform_digest,
            "worker_distribution_digest": self.worker_distribution_digest,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "requested_image": self.requested_image,
            "image_digest": self.image_digest,
            "local_image_id": self.local_image_id,
            "repo_digests": list(self.repo_digests),
            "oci_platform": self.oci_platform,
            "platform_digest": self.platform_digest,
            "docker_binary": self.docker_binary,
            "uid": self.uid,
            "gid": self.gid,
            "sglang_version": self.sglang_version,
            "worker": {
                "distribution": self.worker_distribution,
                "version": self.worker_version,
                "digest": self.worker_distribution_digest,
                "file_count": self.worker_file_count,
                "total_bytes": self.worker_total_bytes,
            },
            "python": {
                "executable": self.python_executable,
                "implementation": self.python_implementation,
                "version": self.python_version,
                "abi": self.python_abi,
                "platform": self.python_platform,
                "machine": self.machine,
            },
            "packages": dict(self.package_versions),
            "cuda": {
                "cudart_library": self.cudart_library,
                "cuda_visible_devices": self.cuda_visible_devices,
                "nvidia_visible_devices": self.nvidia_visible_devices,
            },
            "security_argv_sha256": self.security_argv_sha256,
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("ascii")).hexdigest()


def _normalize_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _small_text_value(value: object, *, allow_empty: bool) -> str | None:
    if (
        not isinstance(value, str)
        or _SMALL_TEXT.fullmatch(value) is None
        or (not allow_empty and not value)
    ):
        return None
    return value


def _safe_docker_binary(value: object) -> bool:
    if (
        not isinstance(value, str)
        or len(value) > 4096
        or "\x00" in value
        or value.startswith("//")
    ):
        return False
    path = PurePosixPath(value)
    return bool(
        path.is_absolute()
        and path.name == "docker"
        and ".." not in path.parts
        and str(path) == value
        and all(re.fullmatch(r"[A-Za-z0-9._+-]+", part) for part in path.parts[1:])
    )


def _safe_absolute_path(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 4096 or "\x00" in value or value.startswith("//"):
        return False
    path = PurePosixPath(value)
    return bool(
        path.is_absolute()
        and ".." not in path.parts
        and str(path) == value
        and all(re.fullmatch(r"[A-Za-z0-9._+-]+", part) for part in path.parts[1:])
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _bounded_argv_runner(
    argv: tuple[str, ...],
    *,
    timeout_s: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> CommandResult:
    """Run one shell-free argv with an absolute deadline and live output caps."""
    if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise RuntimePreflightError("runner received invalid argv")
    deadline = time.monotonic() + float(timeout_s)
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimePreflightError(f"cannot execute preflight command: {exc}") from None
    assert process.stdout is not None and process.stderr is not None
    selector = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, ("stdout", max_stdout_bytes))
        selector.register(process.stderr, selectors.EVENT_READ, ("stderr", max_stderr_bytes))
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout_s)
            events = selector.select(remaining)
            if not events:
                raise subprocess.TimeoutExpired(argv, timeout_s)
            for key, _ in events:
                name, limit = key.data
                chunk = os.read(key.fd, min(4096, limit + 1 - len(buffers[name])))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffers[name].extend(chunk)
                if len(buffers[name]) > limit:
                    raise RuntimePreflightError(
                        f"preflight {name} exceeded its {limit}-byte bound"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(argv, timeout_s)
        returncode = process.wait(timeout=remaining)
    except BaseException:
        _terminate(process)
        raise
    finally:
        if selector is not None:
            selector.close()
    return CommandResult(
        returncode=returncode,
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
    )


def _strict_json(raw: bytes, *, max_bytes: int, label: str) -> object:
    if not isinstance(raw, bytes) or not raw or len(raw) > max_bytes:
        raise RuntimePreflightError(f"{label} JSON is empty or exceeds its byte bound")
    try:
        text = raw.decode("utf-8", errors="strict")

        def object_pairs(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    raise ValueError(f"duplicate key {key!r}")
                out[key] = value
            return out

        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise RuntimePreflightError(f"{label} emitted malformed JSON: {exc}") from None


def _exact_object(value: object, keys: frozenset[str], *, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimePreflightError(
            f"{label} keys/type mismatch: expected={sorted(keys)!r} actual={actual!r}"
        )
    return value


def _small_text(value: object, *, label: str, allow_empty: bool = False) -> str:
    validated = _small_text_value(value, allow_empty=allow_empty)
    if validated is None:
        raise RuntimePreflightError(f"{label} must be a bounded string")
    return validated


def _bounded_int(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise RuntimePreflightError(f"{label} must be a bounded integer")
    return value


def _inspect_argv(config: RuntimePreflightConfig) -> tuple[str, ...]:
    return (
        config.docker_binary,
        "image",
        "inspect",
        _INSPECT_FORMAT,
        config.image,
    )


def _container_argv(
    config: RuntimePreflightConfig,
    *,
    local_image_id: str,
    container_name: str,
    run_prefix: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    prefix = run_prefix or (
        config.docker_binary,
        "run",
        "--rm",
        "--pull=never",
        f"--platform={config.expected_oci_platform}",
        "--network=none",
        "--read-only",
        "--runtime=runc",
        "--ipc=none",
        f"--name={container_name}",
    )
    return (
        *prefix,
        *(() if run_prefix is None else (
            "--rm",
            "--pull=never",
            f"--platform={config.expected_oci_platform}",
            "--network=none",
            "--read-only",
            "--runtime=runc",
            "--ipc=none",
        )),
        "--stop-timeout=1",
        "--no-healthcheck",
        f"--user={config.uid}:{config.gid}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        "--security-opt=seccomp=builtin",
        "--pids-limit=32",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1.0",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m",
        "--workdir=/tmp",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=CUDA_VISIBLE_DEVICES=",
        "--log-driver=none",
        f"--entrypoint={config.expected_python_executable}",
        local_image_id,
        "-I",
        "-S",
        "-c",
        _CONTAINER_SCRIPT,
    )


def _new_container_name() -> str:
    try:
        name = "optima-runtime-preflight-" + secrets.token_hex(10)
    except Exception as exc:
        raise RuntimePreflightError(
            f"cannot allocate trusted preflight container name: {exc}"
        ) from None
    if _CONTAINER_NAME.fullmatch(name) is None:
        raise RuntimePreflightError("trusted preflight container name is invalid")
    return name


def _clock_now(clock: Callable[[], float]) -> float:
    try:
        value = float(clock())
    except Exception as exc:
        raise RuntimePreflightError(f"preflight clock failed: {exc}") from None
    if not math.isfinite(value):
        raise RuntimePreflightError("preflight clock returned a non-finite value")
    return value


def _invoke(
    runner: Runner,
    argv: tuple[str, ...],
    *,
    deadline: float,
    clock: Callable[[], float],
    max_stdout_bytes: int,
) -> CommandResult:
    remaining = deadline - _clock_now(clock)
    if not math.isfinite(remaining) or remaining <= 0:
        raise RuntimePreflightError("runtime preflight absolute deadline expired")
    try:
        result = runner(
            argv,
            timeout_s=remaining,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=MAX_STDERR_BYTES,
        )
    except (RuntimePreflightError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, RuntimePreflightError):
            raise
        raise RuntimePreflightError("runtime preflight timed out") from None
    except Exception as exc:
        raise RuntimePreflightError(f"runtime preflight runner failed: {exc}") from None
    if not isinstance(result, CommandResult):
        raise RuntimePreflightError("preflight runner returned an invalid result type")
    if (
        type(result.returncode) is not int
        or not isinstance(result.stdout, bytes)
        or not isinstance(result.stderr, bytes)
    ):
        raise RuntimePreflightError("preflight runner returned invalid field types")
    if result.returncode != 0:
        detail = result.stderr[:512].decode("utf-8", errors="replace")
        raise RuntimePreflightError(
            f"preflight command exited {result.returncode}: {detail}"
        )
    if result.stderr.strip():
        raise RuntimePreflightError("preflight command emitted unexpected stderr")
    if len(result.stdout) > max_stdout_bytes or len(result.stderr) > MAX_STDERR_BYTES:
        raise RuntimePreflightError("preflight runner violated output bounds")
    return result


def _cleanup_container(
    runner: Runner,
    config: RuntimePreflightConfig,
    *,
    container_name: str,
    clock: Callable[[], float],
) -> None:
    """Force-remove a named container after a failed attached run."""
    cleanup_deadline = _clock_now(clock) + min(5.0, float(config.timeout_s))
    try:
        _invoke(
            runner,
            (
                config.docker_binary,
                "rm",
                "--force",
                "--volumes",
                container_name,
            ),
            deadline=cleanup_deadline,
            clock=clock,
            max_stdout_bytes=1024,
        )
    except RuntimePreflightError as exc:
        raise RuntimePreflightError(
            "runtime preflight launch failed and forced container cleanup "
            f"could not be confirmed: {exc}"
        ) from None


def _platform_digest(
    *,
    oci_platform: str,
    python_implementation: str,
    python_executable: str,
    python_version: str,
    python_abi: str,
    python_platform: str,
    machine: str,
) -> str:
    payload = {
        "schema": PLATFORM_DIGEST_SCHEMA,
        "oci_platform": oci_platform,
        "python_implementation": python_implementation,
        "python_executable": python_executable,
        "python_version": python_version,
        "python_abi": python_abi,
        "python_platform": python_platform,
        "machine": machine,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _run_runtime_preflight(
    config: RuntimePreflightConfig,
    *,
    runner: Runner = _bounded_argv_runner,
    clock: Callable[[], float] = time.monotonic,
    process_manager: OCIProcessManager | None = None,
) -> RuntimePreflightReceipt:
    """Attest one immutable runtime image without candidate data, mounts, or GPUs."""
    if not isinstance(config, RuntimePreflightConfig):
        raise RuntimePreflightError("runtime preflight requires a validated config")
    if process_manager is not None and process_manager.docker_binary != config.docker_binary:
        raise RuntimePreflightError("runtime preflight manager uses another Docker client")
    started = _clock_now(clock)
    deadline = started + float(config.timeout_s)

    inspect_result = _invoke(
        runner,
        _inspect_argv(config),
        deadline=deadline,
        clock=clock,
        max_stdout_bytes=MAX_INSPECT_STDOUT_BYTES,
    )
    inspected = _exact_object(
        _strict_json(
            inspect_result.stdout,
            max_bytes=MAX_INSPECT_STDOUT_BYTES,
            label="docker image inspect",
        ),
        INSPECT_SCHEMA_KEYS,
        label="docker image inspect",
    )
    local_image_id = inspected["Id"]
    if not isinstance(local_image_id, str) or _SHA_ID.fullmatch(local_image_id) is None:
        raise RuntimePreflightError("docker inspect returned an invalid local image ID")
    repo_digests_raw = inspected["RepoDigests"]
    if (
        not isinstance(repo_digests_raw, list)
        or not 1 <= len(repo_digests_raw) <= 64
        or any(
            not isinstance(item, str) or _IMAGE.fullmatch(item) is None
            for item in repo_digests_raw
        )
        or len(set(repo_digests_raw)) != len(repo_digests_raw)
    ):
        raise RuntimePreflightError("docker inspect returned invalid RepoDigests")
    if config.image not in repo_digests_raw:
        raise RuntimePreflightError(
            "requested manifest digest is not bound to the inspected local image ID"
        )
    repo_digests = tuple(sorted(repo_digests_raw))
    if inspected["Volumes"] not in (None, {}):
        raise RuntimePreflightError("runtime preflight image declares Dockerfile volumes")
    image_os = _small_text(inspected["Os"], label="image OS")
    image_arch = _small_text(inspected["Architecture"], label="image architecture")
    oci_platform = f"{image_os}/{image_arch}"
    if oci_platform != config.expected_oci_platform:
        raise RuntimePreflightError(
            f"OCI platform mismatch: {oci_platform!r} != {config.expected_oci_platform!r}"
        )

    container_name = _new_container_name()
    lease = None
    run_prefix = None
    if process_manager is not None:
        suffix = container_name.rsplit("-", 1)[1]
        try:
            lease = process_manager.register(
                lease_id=f"preflight-{suffix}", container_name=container_name
            )
        except OCIProcessError as exc:
            raise RuntimePreflightError(
                f"runtime preflight lease registration failed: {exc}"
            ) from None
        run_prefix = lease.run_prefix(config.docker_binary)
    container_argv = _container_argv(
        config,
        local_image_id=local_image_id,
        container_name=container_name,
        run_prefix=run_prefix,
    )
    try:
        if process_manager is None:
            container_result = _invoke(
                runner,
                container_argv,
                deadline=deadline,
                clock=clock,
                max_stdout_bytes=MAX_RECEIPT_STDOUT_BYTES,
            )
        else:
            assert lease is not None

            def leased_runner(argv, *, timeout_s, max_stdout_bytes, max_stderr_bytes):
                return process_manager.run_capture(
                    lease,
                    argv,
                    timeout_s=timeout_s,
                    max_stdout_bytes=max_stdout_bytes,
                    max_stderr_bytes=max_stderr_bytes,
                    capture_runner=runner,
                )

            container_result = _invoke(
                leased_runner,
                container_argv,
                deadline=deadline,
                clock=clock,
                max_stdout_bytes=MAX_RECEIPT_STDOUT_BYTES,
            )
    except RuntimePreflightError:
        if process_manager is None:
            _cleanup_container(
                runner,
                config,
                container_name=container_name,
                clock=clock,
            )
        raise
    finally:
        if process_manager is not None and lease is not None:
            try:
                process_manager.release(lease)
            except OCIProcessError as exc:
                raise RuntimePreflightError(
                    f"runtime preflight lease cleanup failed: {exc}"
                ) from None
    container = _exact_object(
        _strict_json(
            container_result.stdout,
            max_bytes=MAX_RECEIPT_STDOUT_BYTES,
            label="runtime container",
        ),
        frozenset({"schema", "sglang_version", "worker", "python", "packages", "cuda"}),
        label="runtime container",
    )
    if container["schema"] != CONTAINER_RECEIPT_SCHEMA:
        raise RuntimePreflightError("runtime container schema mismatch")
    sglang_version = _small_text(container["sglang_version"], label="sglang_version")
    if sglang_version != config.expected_sglang_version:
        raise RuntimePreflightError(
            f"installed sglang mismatch: {sglang_version!r} != "
            f"{config.expected_sglang_version!r}"
        )

    worker = _exact_object(
        container["worker"],
        frozenset({"distribution", "version", "digest", "file_count", "total_bytes"}),
        label="worker receipt",
    )
    worker_distribution = _small_text(
        worker["distribution"], label="worker distribution"
    )
    worker_version = _small_text(worker["version"], label="worker version")
    worker_digest = worker["digest"]
    if not isinstance(worker_digest, str) or _DIGEST.fullmatch(worker_digest) is None:
        raise RuntimePreflightError("worker distribution digest is invalid")
    worker_file_count = _bounded_int(
        worker["file_count"], label="worker file_count", maximum=4096
    )
    worker_total_bytes = _bounded_int(
        worker["total_bytes"], label="worker total_bytes", maximum=64 * 1024 * 1024
    )
    if worker_file_count == 0 or worker_total_bytes == 0:
        raise RuntimePreflightError("worker distribution inventory is empty")
    if _normalize_distribution(worker_distribution) != _normalize_distribution(
        config.expected_worker_distribution
    ):
        raise RuntimePreflightError("installed worker distribution mismatch")
    if worker_version != config.expected_worker_version:
        raise RuntimePreflightError(
            f"installed worker version mismatch: {worker_version!r} != "
            f"{config.expected_worker_version!r}"
        )
    if worker_digest != config.expected_worker_digest:
        raise RuntimePreflightError("installed worker distribution digest mismatch")

    python = _exact_object(
        container["python"],
        frozenset({"executable", "implementation", "version", "abi", "platform", "machine"}),
        label="python receipt",
    )
    python_implementation = _small_text(
        python["implementation"], label="python implementation"
    )
    python_executable = _small_text(
        python["executable"], label="python executable"
    )
    if not _safe_absolute_path(python_executable):
        raise RuntimePreflightError("Python executable path is invalid")
    if python_executable != config.expected_python_executable:
        raise RuntimePreflightError(
            f"Python executable mismatch: {python_executable!r} != "
            f"{config.expected_python_executable!r}"
        )
    python_version = _small_text(python["version"], label="python version")
    python_abi = _small_text(python["abi"], label="python ABI", allow_empty=True)
    python_platform = _small_text(python["platform"], label="python platform")
    machine = _small_text(python["machine"], label="machine")
    if python_platform != config.expected_python_platform:
        raise RuntimePreflightError(
            f"Python platform mismatch: {python_platform!r} != "
            f"{config.expected_python_platform!r}"
        )
    if machine != config.expected_machine:
        raise RuntimePreflightError(
            f"machine mismatch: {machine!r} != {config.expected_machine!r}"
        )
    platform_digest = _platform_digest(
        oci_platform=oci_platform,
        python_implementation=python_implementation,
        python_executable=python_executable,
        python_version=python_version,
        python_abi=python_abi,
        python_platform=python_platform,
        machine=machine,
    )

    packages = _exact_object(
        container["packages"], frozenset(_PACKAGE_NAMES), label="package receipt"
    )
    package_versions: list[tuple[str, str | None]] = []
    for name in _PACKAGE_NAMES:
        value = packages[name]
        if value is not None:
            value = _small_text(value, label=f"package {name}")
        package_versions.append((name, value))
    cuda = _exact_object(
        container["cuda"],
        frozenset(
            {"cudart_library", "cuda_visible_devices", "nvidia_visible_devices"}
        ),
        label="CUDA receipt",
    )
    cudart = cuda["cudart_library"]
    if cudart is not None:
        cudart = _small_text(cudart, label="cudart_library")
    cuda_visible = _small_text(
        cuda["cuda_visible_devices"], label="cuda_visible_devices", allow_empty=True
    )
    nvidia_visible = _small_text(
        cuda["nvidia_visible_devices"], label="nvidia_visible_devices"
    )
    if cuda_visible != "" or nvidia_visible != "void":
        raise RuntimePreflightError(
            "runtime preflight container did not preserve no-GPU policy"
        )

    image_digest = config.image.rsplit("@sha256:", 1)[1]
    security_argv_sha256 = hashlib.sha256(
        json.dumps(container_argv, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimePreflightReceipt(
        schema=HOST_RECEIPT_SCHEMA,
        requested_image=config.image,
        image_digest=image_digest,
        local_image_id=local_image_id,
        repo_digests=repo_digests,
        oci_platform=oci_platform,
        platform_digest=platform_digest,
        docker_binary=config.docker_binary,
        uid=config.uid,
        gid=config.gid,
        sglang_version=sglang_version,
        worker_distribution=worker_distribution,
        worker_version=worker_version,
        worker_distribution_digest=worker_digest,
        worker_file_count=worker_file_count,
        worker_total_bytes=worker_total_bytes,
        python_implementation=python_implementation,
        python_executable=python_executable,
        python_version=python_version,
        python_abi=python_abi,
        python_platform=python_platform,
        machine=machine,
        package_versions=tuple(package_versions),
        cudart_library=cudart,
        cuda_visible_devices=cuda_visible,
        nvidia_visible_devices=nvidia_visible,
        security_argv_sha256=security_argv_sha256,
    )


def run_runtime_preflight(
    config: RuntimePreflightConfig,
    *,
    process_manager: OCIProcessManager | None = None,
    runner: Runner = _bounded_argv_runner,
    clock: Callable[[], float] = time.monotonic,
) -> RuntimePreflightReceipt:
    """Run the production preflight under a durable OCI process lease."""
    if process_manager is None:
        raise RuntimePreflightError(
            "production runtime preflight requires an OCIProcessManager lease"
        )
    return _run_runtime_preflight(
        config,
        runner=runner,
        clock=clock,
        process_manager=process_manager,
    )


def _run_runtime_preflight_unleased_for_test(
    config: RuntimePreflightConfig,
    *,
    runner: Runner,
    clock: Callable[[], float] = time.monotonic,
) -> RuntimePreflightReceipt:
    """Exercise receipt validation with a scripted runner; never a production API."""
    if runner is _bounded_argv_runner:
        raise RuntimePreflightError("unleased test preflight requires a scripted runner")
    return _run_runtime_preflight(config, runner=runner, clock=clock)


__all__ = [
    "CommandResult",
    "CONTAINER_RECEIPT_SCHEMA",
    "HOST_RECEIPT_SCHEMA",
    "PLATFORM_DIGEST_SCHEMA",
    "RuntimePreflightConfig",
    "RuntimePreflightError",
    "RuntimePreflightReceipt",
    "WORKER_DIGEST_SCHEMA",
    "WORKER_DISTRIBUTION",
    "run_runtime_preflight",
]
