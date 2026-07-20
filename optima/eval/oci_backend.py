"""Trusted host assembly for one isolated, content-addressed engine session.

This module is the narrow join between path-free launch identity, immutable host
mounts, the OCI lifecycle, device observations, native prebuild, and the raw outer
session.  It contains no scheduling role, quality policy, settlement state, chain
client, inference runtime import, or candidate execution.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Protocol

from optima.cute_aot import CUTE_COMPILE_PROFILE_DIGEST_ENV
from optima.eval.device_state import (
    CommandRunner as DeviceCommandRunner,
    DeviceStateActiveReceipt,
    DeviceStateGuard,
    DeviceStatePolicy,
    DeviceStateReceipt,
    subprocess_runner as device_subprocess_runner,
    validate_device_state_policy,
)
from optima.eval.engine_launch import (
    EngineLaunchError,
    EngineLaunchSpec,
    ResolvedEngineLaunch,
    TrustedLaunchBinding,
    reopen_launch_tree,
    resolve_engine_launch,
)
from optima.eval.native_artifact import (
    NativeArtifactLimits,
    NativeArtifactPublication,
    reopen_native_artifact,
)
from optima.eval.oci_cpuset import validate_cpuset_pair
from optima.eval.oci_outer_session import (
    AttachedSessionTransport,
    OpenedOuterSession,
    OuterSessionError,
    SessionExecutionEvidence,
    SessionExecutionPlan,
    run_outer_session,
)
from optima.eval.oci_prebuild import (
    OCIPrebuildConfig,
    OCIPrebuildResult,
    PREBUILD_RECEIPT,
    run_oci_prebuild,
)
from optima.eval.oci_process import (
    OCIAttachedDiagnostic,
    OCILease,
    OCIProcessManager,
    OCIQuiescenceReceipt,
)
from optima.eval.oci_reference_session import (
    AttachedReferenceTransport,
    ReferenceSessionEvidence,
    ReferenceSessionPlan,
    run_reference_session,
)
from optima.eval.oci_resident_session import (
    ResidentOuterSession,
    ResidentSessionEvidence,
    ResidentSessionPlan,
)
from optima.eval.oci_session_protocol import (
    CONTAINER_SWAP_INTAKE_PATH,
    RuntimePreflightFacts,
)
from optima.eval.runtime_preflight import (
    HOST_RECEIPT_SCHEMA,
    RuntimePreflightReceipt,
    WORKER_DISTRIBUTION,
)
from optima.stack_identity import canonical_digest
from optima._strict import require_digest


CONTAINER_TREE = "/optima/engine-tree"
CONTAINER_MODEL = "/optima/input/model"
CONTAINER_ARTIFACT_BASE = "/optima/native-artifacts"
CONTAINER_CACHE = "/optima/runtime-cache"

_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IMAGE_REF = re.compile(r"[a-z0-9][a-z0-9._/:+-]{0,255}@sha256:[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9_.:+/@-]{1,256}\Z")
_OCI_PLATFORM = re.compile(
    r"[a-z0-9][a-z0-9._-]{0,31}/[a-z0-9][a-z0-9._-]{0,31}\Z"
)
_OPAQUE_ID = re.compile(r"runtime-[0-9a-f]{32}\Z")


class OCIBackendError(RuntimeError):
    """A trusted identity, resource, launch, or cleanup fact is invalid."""


class OCIBackendRuntimeError(OCIBackendError):
    """Runtime cleanup failure with a typed, bounded host diagnostic receipt."""

    def __init__(
        self, message: str, diagnostic: OCIAttachedDiagnostic | None = None
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


def _reference_publication_is_control_only(
    publication: NativeArtifactPublication,
) -> bool:
    """Return whether a pristine publication contains only trusted metadata."""

    return (
        publication.directories == ()
        and tuple(row.path for row in publication.files) == (PREBUILD_RECEIPT,)
    )


class OCIBackendDeadlineError(OCIBackendError):
    """The caller-owned absolute execution deadline is unavailable or expired."""


def _digest(value: object, *, field: str) -> str:
    return require_digest(value, field=field, error=OCIBackendError)


def _absolute_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise OCIBackendError(f"{field} must be a canonical absolute path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise OCIBackendError(f"{field} must be a canonical absolute path")
    return value


def _now(clock: Callable[[], float]) -> float:
    try:
        value = float(clock())
    except Exception as exc:
        raise OCIBackendDeadlineError(f"executor monotonic clock failed: {exc}") from None
    if not math.isfinite(value):
        raise OCIBackendDeadlineError("executor monotonic clock returned a non-finite value")
    return value


def _deadline(value: object, *, clock: Callable[[], float]) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise OCIBackendDeadlineError("deadline must be a finite absolute monotonic value")
    result = float(value)
    if result <= _now(clock):
        raise OCIBackendDeadlineError("executor absolute deadline has expired")
    return result


def _remaining(deadline: float, *, clock: Callable[[], float], stage: str) -> float:
    remaining = deadline - _now(clock)
    if not math.isfinite(remaining) or remaining <= 0:
        raise OCIBackendDeadlineError(f"executor deadline expired during {stage}")
    return remaining


@dataclass(frozen=True)
class CandidateFreeRuntimeIdentity:
    """Path-free runtime identities derived only from the preflighted image."""

    runtime_digest: str
    base_engine_digest: str
    validator_overlay_digest: str

    def __post_init__(self) -> None:
        for field in (
            "runtime_digest",
            "base_engine_digest",
            "validator_overlay_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))


def runtime_identity_from_preflight(
    receipt: RuntimePreflightReceipt,
) -> CandidateFreeRuntimeIdentity:
    """Derive runtime/base/validator-overlay identity without candidate inputs."""
    if type(receipt) is not RuntimePreflightReceipt:
        raise OCIBackendError("runtime preflight receipt has the wrong type")
    if receipt.schema != HOST_RECEIPT_SCHEMA:
        raise OCIBackendError("runtime preflight receipt schema mismatch")
    for field in ("image_digest", "platform_digest", "worker_distribution_digest"):
        _digest(getattr(receipt, field), field=f"preflight {field}")
    if (
        not isinstance(receipt.requested_image, str)
        or _IMAGE_REF.fullmatch(receipt.requested_image) is None
        or not receipt.requested_image.endswith("@sha256:" + receipt.image_digest)
        or not isinstance(receipt.local_image_id, str)
        or _IMAGE_ID.fullmatch(receipt.local_image_id) is None
        or not isinstance(receipt.oci_platform, str)
        or _OCI_PLATFORM.fullmatch(receipt.oci_platform) is None
        or not isinstance(receipt.sglang_version, str)
        or _TOKEN.fullmatch(receipt.sglang_version) is None
        or receipt.worker_distribution != WORKER_DISTRIBUTION
        or not isinstance(receipt.worker_version, str)
        or _TOKEN.fullmatch(receipt.worker_version) is None
    ):
        raise OCIBackendError("runtime preflight receipt identity fields are malformed")
    runtime = canonical_digest(
        "optima.eval.preflighted-runtime",
        {
            "image_digest": receipt.image_digest,
            "oci_platform": receipt.oci_platform,
            "platform_digest": receipt.platform_digest,
            "sglang_version": receipt.sglang_version,
        },
    )
    base = canonical_digest(
        "optima.eval.preflighted-base-engine",
        {"runtime_digest": runtime, "sglang_version": receipt.sglang_version},
    )
    overlay = canonical_digest(
        "optima.eval.installed-worker-overlay",
        {
            "distribution": receipt.worker_distribution,
            "version": receipt.worker_version,
            "worker_distribution_digest": receipt.worker_distribution_digest,
        },
    )
    return CandidateFreeRuntimeIdentity(runtime, base, overlay)


@dataclass(frozen=True)
class TrustedArenaModelMountReceipt:
    """Trusted local binding for one pre-verified model and arena identity."""

    model_root: Path
    arena_digest: str
    model_revision_digest: str
    model_manifest_digest: str
    model_content_digest: str
    root_device: int
    root_inode: int
    root_mode: int
    root_uid: int
    root_gid: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_root", Path(self.model_root))
        for field in (
            "arena_digest",
            "model_revision_digest",
            "model_manifest_digest",
            "model_content_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        for field in ("root_device", "root_inode", "root_mode", "root_uid", "root_gid"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise OCIBackendError(f"model receipt {field} must be a nonnegative integer")

    @classmethod
    def capture(
        cls,
        model_root: str | Path,
        *,
        arena_digest: str,
        model_revision_digest: str,
        model_manifest_digest: str,
        model_content_digest: str,
    ) -> "TrustedArenaModelMountReceipt":
        root, info = _reopen_directory(model_root, field="model_root")
        return cls(
            root,
            arena_digest,
            model_revision_digest,
            model_manifest_digest,
            model_content_digest,
            info.st_dev,
            info.st_ino,
            stat.S_IMODE(info.st_mode),
            info.st_uid,
            info.st_gid,
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.eval.arena-model-mount",
            {
                "arena_digest": self.arena_digest,
                "model_content_digest": self.model_content_digest,
                "model_manifest_digest": self.model_manifest_digest,
                "model_revision_digest": self.model_revision_digest,
            },
        )

    def reopen(self) -> Path:
        root, info = _reopen_directory(self.model_root, field="model_root")
        observed = (
            info.st_dev,
            info.st_ino,
            stat.S_IMODE(info.st_mode),
            info.st_uid,
            info.st_gid,
        )
        expected = (
            self.root_device,
            self.root_inode,
            self.root_mode,
            self.root_uid,
            self.root_gid,
        )
        if observed != expected:
            raise OCIBackendError("model root identity changed after its trusted receipt")
        return root


def _reopen_directory(value: str | Path, *, field: str) -> tuple[Path, os.stat_result]:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise OCIBackendError(f"{field} must be an absolute host path")
    try:
        if stat.S_ISLNK(requested.lstat().st_mode):
            raise OCIBackendError(f"{field} must not be a symlink")
        root = requested.resolve(strict=True)
        info = root.stat()
    except OCIBackendError:
        raise
    except (OSError, RuntimeError) as exc:
        raise OCIBackendError(f"{field} is unavailable: {exc}") from None
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o022:
        raise OCIBackendError(f"{field} must be a non-group/world-writable directory")
    if any(token in str(root) for token in (",", "\x00", "\r", "\n")):
        raise OCIBackendError(f"{field} cannot be represented as a closed OCI mount")
    return root, info


@dataclass(frozen=True)
class OCIRuntimeResourcePolicy:
    """Path-free runtime limits bound into the launch's composite resource policy."""

    uid: int
    gid: int
    cpu_millis: int
    memory_bytes: int
    pids_limit: int
    nofile_limit: int
    cache_bytes: int
    cache_inodes: int
    tmpfs_bytes: int
    shm_bytes: int
    init_timeout_seconds: float
    batch_timeout_seconds: float
    container_python: str
    cpuset_cpus: str | None = None
    cpuset_mems: str | None = None

    def __post_init__(self) -> None:
        bounds = {
            "uid": (1, 2_147_483_647),
            "gid": (1, 2_147_483_647),
            "cpu_millis": (100, 256_000),
            "memory_bytes": (256 << 20, 1 << 50),
            "pids_limit": (64, 1_048_576),
            "nofile_limit": (1_024, 1_048_576),
            "cache_bytes": (16 << 20, 1 << 40),
            "cache_inodes": (1_024, 10_000_000),
            "tmpfs_bytes": (16 << 20, 1 << 40),
            "shm_bytes": (16 << 20, 1 << 40),
        }
        for field, (low, high) in bounds.items():
            value = getattr(self, field)
            if type(value) is not int or not low <= value <= high:
                raise OCIBackendError(f"runtime resource {field} is outside its hard bound")
        for field in ("init_timeout_seconds", "batch_timeout_seconds"):
            value = getattr(self, field)
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not 0 < float(value) <= 86_400
            ):
                raise OCIBackendError(f"runtime resource {field} is invalid")
        object.__setattr__(
            self,
            "container_python",
            _absolute_path(self.container_python, field="container_python"),
        )
        try:
            cpus, mems = validate_cpuset_pair(
                self.cpuset_cpus,
                self.cpuset_mems,
                cpu_millis=self.cpu_millis,
            )
        except ValueError as exc:
            raise OCIBackendError(f"runtime resource {exc}") from None
        object.__setattr__(self, "cpuset_cpus", cpus)
        object.__setattr__(self, "cpuset_mems", mems)

    @property
    def digest(self) -> str:
        payload: dict[str, object] = {
            "batch_timeout_milliseconds": int(
                round(float(self.batch_timeout_seconds) * 1000)
            ),
            "cache_bytes": self.cache_bytes,
            "cache_inodes": self.cache_inodes,
            "container_python": self.container_python,
            "cpu_millis": self.cpu_millis,
            "gid": self.gid,
            "init_timeout_milliseconds": int(
                round(float(self.init_timeout_seconds) * 1000)
            ),
            "memory_bytes": self.memory_bytes,
            "nofile_limit": self.nofile_limit,
            "pids_limit": self.pids_limit,
            "shm_bytes": self.shm_bytes,
            "tmpfs_bytes": self.tmpfs_bytes,
            "uid": self.uid,
        }
        # Preserve the historical no-affinity policy identity byte-for-byte.
        # Cpuset placement is an optional authority extension, not an identity
        # epoch for every existing runtime policy.
        if self.cpuset_cpus is not None:
            payload["cpuset_cpus"] = self.cpuset_cpus
            payload["cpuset_mems"] = self.cpuset_mems
        return canonical_digest("optima.eval.oci-runtime-resource-policy", payload)


@dataclass(frozen=True)
class OCIBackendConfig:
    prebuild: OCIPrebuildConfig
    runtime: OCIRuntimeResourcePolicy
    native_limits: NativeArtifactLimits = NativeArtifactLimits()

    def __post_init__(self) -> None:
        if type(self.prebuild) is not OCIPrebuildConfig:
            raise OCIBackendError("backend prebuild config has the wrong type")
        if type(self.runtime) is not OCIRuntimeResourcePolicy:
            raise OCIBackendError("backend runtime policy has the wrong type")
        if type(self.native_limits) is not NativeArtifactLimits:
            raise OCIBackendError("backend native limits have the wrong type")
        policy = self.prebuild.policy
        if (
            policy.runtime_policy_digest != self.runtime.digest
            or policy.uid != self.runtime.uid
            or policy.gid != self.runtime.gid
            or policy.container_python != self.runtime.container_python
        ):
            raise OCIBackendError(
                "prebuild and runtime resource policies do not share one identity"
            )


def _copy_seccomp(source: Path, destination: Path, *, expected_digest: str) -> None:
    expected = _digest(expected_digest, field="seccomp_policy_digest")
    try:
        if source.is_symlink():
            raise OCIBackendError("seccomp profile must not be a symlink")
        before = source.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OCIBackendError("seccomp profile must be a single-linked regular file")
        raw = source.read_bytes()
        after = source.stat()
    except OCIBackendError:
        raise
    except OSError as exc:
        raise OCIBackendError(f"cannot read seccomp profile: {exc}") from None
    stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable):
        raise OCIBackendError("seccomp profile changed while being copied")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise OCIBackendError("seccomp profile digest differs from launch identity")
    try:
        parsed = json.loads(raw)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise OCIBackendError(f"seccomp profile is malformed: {exc}") from None
    if (
        not isinstance(parsed, dict)
        or parsed.get("defaultAction") != "SCMP_ACT_ERRNO"
        or not isinstance(parsed.get("syscalls"), list)
    ):
        raise OCIBackendError("seccomp profile does not have the pinned deny-by-default shape")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(destination, flags, 0o400)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise OCIBackendError(f"cannot stage seccomp profile: {exc}") from None


def _mount(source: Path, destination: str, *, readonly: bool) -> str:
    root, _ = _reopen_directory(source, field="OCI mount source")
    suffix = ",readonly" if readonly else ""
    return (
        f"--mount=type=bind,src={root},dst={destination},"
        f"bind-propagation=rprivate{suffix}"
    )


def expected_runtime_preflight(
    launch: EngineLaunchSpec,
    receipt: RuntimePreflightReceipt,
) -> RuntimePreflightFacts:
    return RuntimePreflightFacts(
        launch_digest=launch.digest,
        runtime_digest=launch.runtime_digest,
        stack_digest=launch.stack_digest,
        tree_digest=launch.tree_digest,
        engine_config_digest=launch.engine_config_digest,
        worker_distribution_digest=launch.worker_distribution_digest,
        model_revision_digest=launch.model_revision_digest,
        model_manifest_digest=launch.model_manifest_digest,
        model_content_digest=launch.model_content_digest,
        sglang_version=receipt.sglang_version,
        gpu_architectures=(launch.hardware.architecture,)
        * launch.hardware.visible_gpu_count,
        topology_digest=launch.hardware.topology_digest,
        loopback_only=True,
        read_only_inputs=True,
        private_writable_cache=True,
    )


def build_runtime_argv(
    *,
    lease: OCILease,
    resolved: ResolvedEngineLaunch,
    preflight: RuntimePreflightReceipt,
    model_root: Path,
    publication: NativeArtifactPublication,
    cache_root: Path,
    seccomp_path: Path,
    runtime: OCIRuntimeResourcePolicy,
    session_protocol: str = "ordinary",
    discovery_overlay_identity_digest: str | None = None,
    swap_intake_root: Path | None = None,
) -> tuple[str, ...]:
    """Construct the exact runtime argv from trusted, already-reopened inputs."""
    if session_protocol not in {"ordinary", "reference", "resident"}:
        raise OCIBackendError("runtime session protocol is not registered")
    if (session_protocol == "resident") != (swap_intake_root is not None):
        raise OCIBackendError(
            "swap-intake mounts exist exactly for resident sessions"
        )
    launch = resolved.spec
    if preflight.local_image_id is None or _IMAGE_ID.fullmatch(preflight.local_image_id) is None:
        raise OCIBackendError("runtime preflight local image ID is malformed")
    artifact_destination = (
        f"{CONTAINER_ARTIFACT_BASE}/{publication.build_spec_digest[:2]}/"
        f"{publication.build_spec_digest}"
    )
    environment = {
        "CUDA_CACHE_PATH": f"{CONTAINER_CACHE}/cuda",
        "FLASHINFER_WORKSPACE_BASE": f"{CONTAINER_CACHE}/flashinfer",
        "HF_HOME": f"{CONTAINER_CACHE}/huggingface",
        "HF_HUB_OFFLINE": "1",
        "HOME": f"{CONTAINER_CACHE}/home",
        "OPTIMA_ENGINE_CONFIG_DIGEST": launch.engine_config_digest,
        "OPTIMA_ENGINE_TREE_DIGEST": launch.tree_digest,
        "OPTIMA_ENGINE_WORKER": "1",
        "OPTIMA_EXTERNAL_NO_EGRESS": "1",
        "OPTIMA_LAUNCH_DIGEST": launch.digest,
        "OPTIMA_MODEL_CONTENT_DIGEST": launch.model_content_digest,
        "OPTIMA_MODEL_MANIFEST_DIGEST": launch.model_manifest_digest,
        "OPTIMA_MODEL_REVISION_DIGEST": launch.model_revision_digest,
        "OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST": publication.publication_digest,
        "OPTIMA_NATIVE_ARTIFACT_ROOT": artifact_destination,
        "OPTIMA_NATIVE_BUILD_SPEC_DIGEST": resolved.native_build_spec.digest,
        "OPTIMA_PREBUILT_ARTIFACTS": "1",
        "OPTIMA_REBUILD_PHASE": "load",
        "OPTIMA_RUNTIME_DIGEST": launch.runtime_digest,
        "OPTIMA_SESSION_PROTOCOL": session_protocol,
        "OPTIMA_STACK_DIGEST": launch.stack_digest,
        "OPTIMA_TARGET_GPU_ARCH": resolved.native_build_spec.target_architecture,
        "OPTIMA_WORKER_DISTRIBUTION_DIGEST": launch.worker_distribution_digest,
        "OPTIMA_EXPECTED_SGLANG_VERSION": preflight.sglang_version,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "TMPDIR": "/tmp",
        "TORCH_EXTENSIONS_DIR": f"{CONTAINER_CACHE}/torch-extensions",
        "TRANSFORMERS_OFFLINE": "1",
        "TRITON_CACHE_DIR": f"{CONTAINER_CACHE}/triton",
        "XDG_CACHE_HOME": f"{CONTAINER_CACHE}/xdg",
    }
    if resolved.native_compile_profile is not None:
        environment[CUTE_COMPILE_PROFILE_DIGEST_ENV] = (
            resolved.native_compile_profile.digest
        )
    discovery_rows = tuple(
        row
        for row in publication.files
        if row.path.startswith("dep_overlays/discovery/")
    )
    if discovery_overlay_identity_digest is None:
        if discovery_rows:
            raise OCIBackendError(
                "ordinary runtime publication contains a discovery overlay"
            )
    else:
        if session_protocol != "ordinary":
            raise OCIBackendError(
                "reference/resident runtimes cannot activate discovery"
            )
        from optima.discovery import DiscoveryError, reopen_discovery_overlay
        from optima.discovery_overlay import ARMED, EXPECTED_IDENTITY

        try:
            reopen_discovery_overlay(
                publication,
                expected_identity_digest=discovery_overlay_identity_digest,
            )
        except (DiscoveryError, OSError, TypeError, ValueError) as exc:
            raise OCIBackendError(
                f"discovery runtime publication cannot reopen: {exc}"
            ) from None
        environment[ARMED] = "1"
        environment[EXPECTED_IDENTITY] = discovery_overlay_identity_digest
    gpu_csv = ",".join(resolved.physical_hardware.physical_gpu_ids)
    # Docker parses --gpus with a CSV decoder. A multi-device request must be one
    # quoted CSV field even though argv is passed directly without a shell;
    # otherwise Docker interprets the tail as a second Count request.
    gpu_request = f"device={gpu_csv}"
    if len(resolved.physical_hardware.physical_gpu_ids) > 1:
        gpu_request = f'"{gpu_request}"'
    argv = [
        *lease.run_prefix(preflight.docker_binary),
        "--rm",
        "--init",
        "--interactive",
        "--pull=never",
        f"--platform={preflight.oci_platform}",
        "--runtime=runc",
        "--network=none",
        "--read-only",
        "--ipc=private",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        f"--security-opt=seccomp={seccomp_path}",
        f"--user={runtime.uid}:{runtime.gid}",
        f"--cpus={runtime.cpu_millis / 1000:g}",
        *(
            (
                f"--cpuset-cpus={runtime.cpuset_cpus}",
                f"--cpuset-mems={runtime.cpuset_mems}",
            )
            if runtime.cpuset_cpus is not None
            else ()
        ),
        f"--memory={runtime.memory_bytes}",
        f"--memory-swap={runtime.memory_bytes}",
        f"--pids-limit={runtime.pids_limit}",
        f"--ulimit=nofile={runtime.nofile_limit}:{runtime.nofile_limit}",
        "--ulimit=core=0:0",
        f"--tmpfs=/tmp:rw,nosuid,nodev,noexec,size={runtime.tmpfs_bytes},"
        f"uid={runtime.uid},gid={runtime.gid},mode=0700",
        f"--shm-size={runtime.shm_bytes}",
        f"--gpus={gpu_request}",
        "--stop-timeout=1",
        "--no-healthcheck",
        "--log-driver=none",
        "--workdir=/tmp",
        _mount(model_root, CONTAINER_MODEL, readonly=True),
        _mount(resolved.materialized_tree_root, CONTAINER_TREE, readonly=True),
        _mount(publication.root, artifact_destination, readonly=True),
        _mount(cache_root, CONTAINER_CACHE, readonly=False),
        *(
            (_mount(swap_intake_root, CONTAINER_SWAP_INTAKE_PATH, readonly=True),)
            if swap_intake_root is not None
            else ()
        ),
    ]
    argv.extend(f"--env={key}={environment[key]}" for key in sorted(environment))
    argv.extend(
        (
            f"--entrypoint={runtime.container_python}",
            preflight.local_image_id,
            "-I",
            "-m",
            "optima.eval.oci_session_worker",
        )
    )
    return tuple(argv)


@dataclass(frozen=True)
class EngineExecutionEvidence:
    schema: str
    launch_digest: str
    runtime_identity: CandidateFreeRuntimeIdentity
    runtime_preflight_receipt_sha256: str
    arena_model_receipt_digest: str
    resource_policy_digest: str
    prebuild: OCIPrebuildResult
    native_publication_digest: str
    runtime_argv_sha256: str
    recovered_lease_ids: tuple[str, ...]
    device_receipts: tuple[
        DeviceStateReceipt, DeviceStateActiveReceipt, DeviceStateReceipt
    ]
    session: SessionExecutionEvidence


@dataclass(frozen=True)
class PristineReferenceExecutionEvidence:
    """Raw, verdict-free evidence from one separately launched empty-stack T."""

    schema: str
    launch_digest: str
    runtime_identity: CandidateFreeRuntimeIdentity
    runtime_preflight_receipt_sha256: str
    arena_model_receipt_digest: str
    resource_policy_digest: str
    prebuild: OCIPrebuildResult
    native_publication_digest: str
    runtime_argv_sha256: str
    recovered_lease_ids: tuple[str, ...]
    device_receipts: tuple[DeviceStateReceipt, DeviceStateReceipt]
    session: ReferenceSessionEvidence


@dataclass(frozen=True)
class ResidentEngineExecutionEvidence:
    """Raw evidence from one resident (hot-swap) engine lifetime.

    Screen/routing tier: one stock launch served an ordered stream of swaps and
    timed reads. Device state is proven pre/post the whole lifetime (like the
    reference role); per-read noise policy lives in the queue layer's
    bracketing, not in an active-conditioning receipt.
    """

    schema: str
    launch_digest: str
    runtime_identity: CandidateFreeRuntimeIdentity
    runtime_preflight_receipt_sha256: str
    arena_model_receipt_digest: str
    resource_policy_digest: str
    prebuild: OCIPrebuildResult
    native_publication_digest: str
    runtime_argv_sha256: str
    recovered_lease_ids: tuple[str, ...]
    device_receipts: tuple[DeviceStateReceipt, DeviceStateReceipt]
    session: ResidentSessionEvidence


class ResidentSessionDriver(Protocol):
    """Trusted host callback that drives one open resident engine lifetime."""

    def __call__(self, session: ResidentOuterSession) -> ResidentSessionEvidence: ...


class OuterSessionRunner(Protocol):
    def __call__(self, plan: SessionExecutionPlan, **kwargs: object) -> SessionExecutionEvidence: ...


class OpenedSessionDriver(Protocol):
    """Trusted host callback that drives one already-started resident engine."""

    def __call__(self, session: OpenedOuterSession) -> SessionExecutionEvidence: ...


class ReferenceSessionRunner(Protocol):
    def __call__(
        self, plan: ReferenceSessionPlan, **kwargs: object
    ) -> ReferenceSessionEvidence: ...


@dataclass(frozen=True)
class _RawRuntimeExecution:
    launch_id: str
    prebuild: OCIPrebuildResult
    publication_digest: str
    argv_digest: str
    pre_receipt: DeviceStateReceipt
    post_receipt: DeviceStateReceipt
    value: object


class _FinalWarmupConditioner:
    """Collect one active device receipt without interpreting session throughput."""

    def __init__(
        self,
        guard: DeviceStateGuard,
        *,
        launch_id: str,
        final_warmup_index: int,
        first_timed_index: int,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        self.guard = guard
        self.launch_id = launch_id
        self.final_warmup_index = final_warmup_index
        self.first_timed_index = first_timed_index
        self.deadline = deadline
        self.clock = clock
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.started = threading.Event()
        self.thread: threading.Thread | None = None
        self.receipt: DeviceStateActiveReceipt | None = None
        self.error: BaseException | None = None

    def _run(self) -> None:
        self.started.set()
        try:
            self.receipt = self.guard.condition_active(
                self.launch_id,
                "final-warmup",
                deadline=self.deadline,
                release=self.release.is_set,
                wait_for_release=self.release.wait,
                cancel=self.cancelled.is_set,
            )
        except BaseException as exc:
            self.error = exc

    def _join(self) -> None:
        if self.thread is None:
            raise OCIBackendError("final-warmup device observation was never started")
        self.thread.join(timeout=max(0.0, self.deadline - _now(self.clock)))
        if self.thread.is_alive():
            raise OCIBackendDeadlineError(
                "final-warmup device observation exceeded the absolute deadline"
            )

    def boundary(self, event: str, batch_index: int, deadline: float) -> None:
        if deadline != self.deadline:
            raise OCIBackendError("outer session changed the caller-owned deadline")
        if event == "before_final_warmup":
            if batch_index != self.final_warmup_index or self.thread is not None:
                raise OCIBackendError("final-warmup device observation order is invalid")
            self.thread = threading.Thread(
                target=self._run,
                name="optima-device-active",
                daemon=True,
            )
            self.thread.start()
            if not self.started.wait(timeout=max(0.0, deadline - _now(self.clock))):
                raise OCIBackendDeadlineError("final-warmup device observer did not start")
            return
        if event == "after_final_warmup":
            if batch_index != self.final_warmup_index:
                raise OCIBackendError("final-warmup release index is invalid")
            self.release.set()
            self._join()
            if self.error is not None:
                raise OCIBackendError(
                    f"final-warmup device observation failed: {self.error}"
                ) from None
            if type(self.receipt) is not DeviceStateActiveReceipt:
                raise OCIBackendError("final-warmup device receipt is missing")
            return
        if event == "before_first_timed":
            if batch_index != self.first_timed_index or self.receipt is None:
                raise OCIBackendError("timed work was released without active device evidence")
            return
        raise OCIBackendError("outer session emitted an unknown device boundary")

    def require_complete(self) -> DeviceStateActiveReceipt:
        if type(self.receipt) is not DeviceStateActiveReceipt or self.error is not None:
            raise OCIBackendError("successful session lacks final-warmup device evidence")
        return self.receipt

    def cancel(self) -> None:
        if self.thread is None or not self.thread.is_alive():
            return
        self.cancelled.set()
        self.release.set()
        self._join()


class OCIEngineExecutor:
    """Sequential trusted-host executor for generic materialized engine trees."""

    def __init__(
        self,
        config: OCIBackendConfig,
        device_policy: DeviceStatePolicy,
        *,
        manager: OCIProcessManager | None = None,
        device_runner: DeviceCommandRunner = device_subprocess_runner,
        device_sleep: Callable[[float], None] = time.sleep,
        session_runner: OuterSessionRunner = run_outer_session,
        reference_session_runner: ReferenceSessionRunner = run_reference_session,
    ) -> None:
        if type(config) is not OCIBackendConfig:
            raise OCIBackendError("executor config has the wrong type")
        if type(device_policy) is not DeviceStatePolicy:
            raise OCIBackendError("executor device policy has the wrong type")
        self.config = config
        self.manager = manager or OCIProcessManager(
            docker_binary=config.prebuild.docker_binary,
            recovery_root=config.prebuild.recovery_root,
            executor_id=config.prebuild.executor_id,
        )
        if (
            self.manager.docker_binary != config.prebuild.docker_binary
            or self.manager.executor_id != config.prebuild.executor_id
        ):
            raise OCIBackendError("executor manager differs from its backend config")
        self.device_policy = device_policy
        self.device_guard = DeviceStateGuard(
            device_policy,
            runner=device_runner,
            clock=self.manager.clock,
            sleep=device_sleep,
        )
        self.session_runner = session_runner
        self.reference_session_runner = reference_session_runner
        # One executor owns one manager, so its transaction lock serializes every lifecycle.
        self._lock = self.manager.transaction_lock
        self._recovered: tuple[str, ...] | None = None

    def _recover_once(self) -> tuple[str, ...]:
        if self._recovered is None:
            self._recovered = self.manager.recover_stale()
        return self._recovered

    def prove_quiescent(self) -> OCIQuiescenceReceipt:
        """Serialize an executor-owned absence proof between engine lifetimes."""

        if not self._lock.acquire(blocking=False):
            raise OCIBackendError("cannot prove quiescence during an active session")
        try:
            return self.manager.prove_quiescent()
        finally:
            self._lock.release()

    @contextlib.contextmanager
    def exclusive_transaction(self) -> Iterator["OCIEngineExecutor"]:
        """Reserve this manager across one causal multi-engine qualification."""

        if not self._lock.acquire(blocking=False):
            raise OCIBackendError("executor manager already has an active transaction")
        try:
            yield self
        finally:
            self._lock.release()

    def _validate_launch_identity(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        *,
        engine_config_digest: str,
        engine_tp_size: int,
        expected_preflight: RuntimePreflightFacts,
    ) -> tuple[
        ResolvedEngineLaunch,
        RuntimePreflightReceipt,
        CandidateFreeRuntimeIdentity,
        RuntimePreflightFacts,
        Path,
    ]:
        if type(launch) is not EngineLaunchSpec or type(binding) is not TrustedLaunchBinding:
            raise OCIBackendError("executor launch/binding types are invalid")
        if type(mount) is not TrustedArenaModelMountReceipt:
            raise OCIBackendError("arena/model mount receipt has the wrong type")
        try:
            resolved = resolve_engine_launch(launch, binding)
        except EngineLaunchError as exc:
            raise OCIBackendError(f"engine launch binding failed: {exc}") from None
        preflight = binding.runtime_preflight_receipt
        if type(preflight) is not RuntimePreflightReceipt:
            raise OCIBackendError("launch lacks a typed runtime preflight receipt")
        identity = runtime_identity_from_preflight(preflight)
        expected_identity = (
            identity.runtime_digest,
            identity.base_engine_digest,
            identity.validator_overlay_digest,
        )
        if expected_identity != (
            launch.runtime_digest,
            launch.base_engine_digest,
            launch.validator_overlay_digest,
        ):
            raise OCIBackendError("launch runtime/base/worker-overlay identity is unsubstantiated")
        if (
            preflight.docker_binary != self.manager.docker_binary
            or preflight.uid != self.config.runtime.uid
            or preflight.gid != self.config.runtime.gid
            or preflight.python_executable != self.config.runtime.container_python
        ):
            raise OCIBackendError("runtime preflight and executor policy differ")
        if (
            launch.arena_digest != mount.arena_digest
            or launch.model_revision_digest != mount.model_revision_digest
            or launch.model_manifest_digest != mount.model_manifest_digest
            or launch.model_content_digest != mount.model_content_digest
        ):
            raise OCIBackendError("arena/model mount receipt differs from launch identity")
        if (
            engine_config_digest != launch.engine_config_digest
            or engine_tp_size != launch.hardware.tp_size
        ):
            raise OCIBackendError("engine session configuration differs from launch identity")
        validate_device_state_policy(
            self.device_policy,
            logical_hardware=launch.hardware,
            physical_hardware=resolved.physical_hardware,
        )
        expected = expected_runtime_preflight(launch, preflight)
        if expected_preflight != expected:
            raise OCIBackendError("outer session expected preflight differs from host policy")
        model_root = mount.reopen()
        return resolved, preflight, identity, expected, model_root

    def _validate_launch(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        plan: SessionExecutionPlan,
    ) -> tuple[
        ResolvedEngineLaunch,
        RuntimePreflightReceipt,
        CandidateFreeRuntimeIdentity,
        RuntimePreflightFacts,
        Path,
    ]:
        if type(plan) is not SessionExecutionPlan:
            raise OCIBackendError("outer session plan has the wrong type")
        if plan.engine_config.digest != plan.expected_engine_config_digest:
            raise OCIBackendError("outer session plan has inconsistent engine identity")
        validated = self._validate_launch_identity(
            launch,
            binding,
            mount,
            engine_config_digest=plan.expected_engine_config_digest,
            engine_tp_size=plan.engine_config.tp_size,
            expected_preflight=plan.expected_preflight,
        )
        if plan.launch_digest != launch.digest:
            raise OCIBackendError("outer session plan names another launch")
        has_discovery_tree = any(
            row.path == "metadata/optima_discovery.json"
            for row in validated[0].materialized_tree.files
        )
        if has_discovery_tree != (
            plan.expected_discovery_overlay_identity_digest is not None
        ):
            raise OCIBackendError(
                "session discovery requirement differs from its engine tree"
            )
        return validated

    def _validate_reference_launch(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        plan: ReferenceSessionPlan,
    ) -> tuple[
        ResolvedEngineLaunch,
        RuntimePreflightReceipt,
        CandidateFreeRuntimeIdentity,
        RuntimePreflightFacts,
        Path,
    ]:
        if type(plan) is not ReferenceSessionPlan:
            raise OCIBackendError("reference session plan has the wrong type")
        if plan.reference.pristine_launch_digest != launch.digest:
            raise OCIBackendError("reference session plan names another launch")
        validated = self._validate_launch_identity(
            launch,
            binding,
            mount,
            engine_config_digest=plan.expected_engine_config_digest,
            engine_tp_size=plan.engine_config.tp_size,
            expected_preflight=plan.expected_preflight,
        )
        resolved = validated[0]
        if resolved.materialized_tree.runtime_manifest is not None:
            raise OCIBackendError("pristine reference tree contains a contribution manifest")
        from optima.eval.marginal_runtime import MaterializedArmBinding

        try:
            rebuilt = type(plan.reference).from_pristine(
                plan.pristine_stack,
                launch,
                MaterializedArmBinding(resolved.materialized_tree, binding),
                workload_digest=plan.reference.workload_digest,
                tokenizer_digest=plan.reference.tokenizer_digest,
                hidden_corpus_commitment=plan.reference.hidden_corpus_commitment,
                hidden_judge_digest=plan.reference.hidden_judge_digest,
                selection_policy_digest=plan.reference.selection_policy_digest,
            )
        except (TypeError, ValueError) as exc:
            raise OCIBackendError(f"pristine reference identity failed to reopen: {exc}") from None
        if rebuilt != plan.reference:
            raise OCIBackendError("reference manifest differs from the reopened empty stack")
        return validated

    def _execute_runtime(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        *,
        absolute: float,
        resolved: ResolvedEngineLaunch,
        preflight: RuntimePreflightReceipt,
        model_root: Path,
        session_protocol: str,
        discovery_overlay_identity_digest: str | None,
        run: Callable[[AttachedSessionTransport, float, str], object],
        swap_intake_root: Path | None = None,
    ) -> _RawRuntimeExecution:
        """Own the common prebuild/lease/mount/teardown shell for ordinary and T."""

        prebuild = run_oci_prebuild(
            launch,
            binding,
            self.config.prebuild,
            manager=self.manager,
            limits=self.config.native_limits,
            deadline=absolute,
        )
        if (
            prebuild.discovery_overlay_identity_digest
            != discovery_overlay_identity_digest
        ):
            raise OCIBackendError(
                "native prebuild discovery result differs from the session plan"
            )
        publication = reopen_native_artifact(
            prebuild.publication.root,
            expected_build_spec_digest=resolved.native_build_spec.digest,
            expected_publication_digest=prebuild.publication.publication_digest,
            limits=self.config.native_limits,
        )
        if session_protocol in (
            "reference",
            "resident",
        ) and not _reference_publication_is_control_only(publication):
            raise OCIBackendError(
                "stock-launched session exposes candidate native artifacts"
            )
        _validate_mount_roots(
            model_root,
            resolved.materialized_tree_root,
            publication.root,
            self.config.prebuild.recovery_root,
            *(() if swap_intake_root is None else (swap_intake_root,)),
        )
        launch_id = _new_runtime_id()
        pre_receipt = self.device_guard.before_launch(launch_id, deadline=absolute)
        post_receipt: DeviceStateReceipt | None = None
        value: object | None = None
        argv_digest = ""
        lease: OCILease | None = None
        transport: AttachedSessionTransport | None = None
        primary: BaseException | None = None
        cleanup_failures: list[BaseException] = []
        try:
            lease = self.manager.register(
                lease_id=launch_id,
                container_name="optima-" + launch_id,
                mount_relpaths=("runtime-cache",),
                stage_relpaths=("seccomp.json",),
            )
            cache_root = lease.mount_paths[0]
            seccomp_copy = lease.stage_paths[0]
            _copy_seccomp(
                self.config.prebuild.seccomp_profile,
                seccomp_copy,
                expected_digest=launch.seccomp_policy_digest,
            )
            self.manager.mount_tmpfs(
                lease,
                cache_root,
                size_bytes=self.config.runtime.cache_bytes,
                inode_limit=self.config.runtime.cache_inodes,
                uid=self.config.runtime.uid,
                gid=self.config.runtime.gid,
                executable=True,
            )
            resolved = resolve_engine_launch(launch, binding)
            model_root = mount.reopen()
            publication = reopen_native_artifact(
                publication.root,
                expected_build_spec_digest=resolved.native_build_spec.digest,
                expected_publication_digest=publication.publication_digest,
                limits=self.config.native_limits,
            )
            if session_protocol in ("reference", "resident") and (
                resolved.materialized_tree.runtime_manifest is not None
                or not _reference_publication_is_control_only(publication)
            ):
                raise OCIBackendError(
                    "stock-launched session inputs acquired contribution state"
                )
            reopen_launch_tree(launch, resolved.materialized_tree_root)
            argv = build_runtime_argv(
                lease=lease,
                resolved=resolved,
                preflight=preflight,
                model_root=model_root,
                publication=publication,
                cache_root=cache_root,
                seccomp_path=seccomp_copy,
                runtime=self.config.runtime,
                session_protocol=session_protocol,
                discovery_overlay_identity_digest=(
                    discovery_overlay_identity_digest
                ),
                swap_intake_root=swap_intake_root,
            )
            argv_digest = hashlib.sha256(
                json.dumps(argv, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            reserve = float(self.device_policy.drain_timeout_s)
            remaining = _remaining(
                absolute, clock=self.manager.clock, stage="runtime session"
            )
            if remaining <= reserve:
                raise OCIBackendDeadlineError(
                    "executor lacks time for a session and mandatory post-drain"
                )
            session_deadline = absolute - reserve
            transport_type = (
                AttachedReferenceTransport
                if session_protocol == "reference"
                else AttachedSessionTransport
            )
            transport = transport_type(
                self.manager, lease, argv, clock=self.manager.clock
            )
            value = run(transport, session_deadline, launch_id)
        except BaseException as exc:
            primary = exc
        finally:
            if transport is not None:
                try:
                    transport.abort()
                except BaseException as exc:
                    cleanup_failures.append(exc)
            if lease is not None:
                try:
                    self.manager.release(lease)
                except BaseException as exc:
                    cleanup_failures.append(exc)
            try:
                post_receipt = self.device_guard.after_launch(
                    launch_id, deadline=absolute
                )
            except BaseException as exc:
                cleanup_failures.append(exc)
        diagnostic: OCIAttachedDiagnostic | None = None
        if transport is not None:
            try:
                observed = transport.stderr_diagnostic()
            except BaseException:
                pass
            else:
                if type(observed) is OCIAttachedDiagnostic:
                    diagnostic = observed
        if isinstance(primary, OuterSessionError) and diagnostic is not None:
            primary.attach_diagnostic(lambda diagnostic=diagnostic: diagnostic)
        if cleanup_failures:
            cause = primary or cleanup_failures[0]
            raise OCIBackendRuntimeError(
                "runtime cleanup or mandatory post-drain could not be proven: "
                + "; ".join(str(item)[:256] for item in cleanup_failures),
                diagnostic,
            ) from cause
        if primary is not None:
            raise primary
        if value is None or type(post_receipt) is not DeviceStateReceipt:
            raise OCIBackendError("runtime returned incomplete raw evidence")
        return _RawRuntimeExecution(
            launch_id,
            prebuild,
            publication.publication_digest,
            argv_digest,
            pre_receipt,
            post_receipt,
            value,
        )

    def _execute_ordinary(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        plan: SessionExecutionPlan,
        *,
        deadline: float,
        opened_driver: OpenedSessionDriver | None,
    ) -> EngineExecutionEvidence:
        if not self._lock.acquire(blocking=False):
            raise OCIBackendError("one executor instance cannot run concurrent sessions")
        try:
            absolute = _deadline(deadline, clock=self.manager.clock)
            recovered = self._recover_once()
            resolved, preflight, identity, expected, model_root = self._validate_launch(
                launch, binding, mount, plan
            )

            def run(
                transport: AttachedSessionTransport,
                session_deadline: float,
                launch_id: str,
            ) -> tuple[SessionExecutionEvidence, DeviceStateActiveReceipt]:
                conditioner = _FinalWarmupConditioner(
                    self.device_guard,
                    launch_id=launch_id,
                    final_warmup_index=plan.warmup_count - 1,
                    first_timed_index=plan.warmup_count,
                    deadline=session_deadline,
                    clock=self.manager.clock,
                )
                try:
                    kwargs = {
                        "transport": transport,
                        "deadline": session_deadline,
                        "init_timeout_s": self.config.runtime.init_timeout_seconds,
                        "batch_timeout_s": self.config.runtime.batch_timeout_seconds,
                        "clock": self.manager.clock,
                        "boundary_callback": conditioner.boundary,
                    }
                    if opened_driver is None:
                        session = self.session_runner(plan, **kwargs)
                    else:
                        controller = OpenedOuterSession(plan, **kwargs)
                        controller.start()
                        try:
                            session = opened_driver(controller)
                            if not controller.closed:
                                raise OCIBackendError(
                                    "opened session driver returned without closing the engine"
                                )
                            if (
                                type(session) is not SessionExecutionEvidence
                                or session.session_id != controller.session_id
                            ):
                                raise OCIBackendError(
                                    "opened session driver returned another session receipt"
                                )
                        finally:
                            controller.abort()
                    return session, conditioner.require_complete()
                finally:
                    conditioner.cancel()

            raw = self._execute_runtime(
                launch,
                binding,
                mount,
                absolute=absolute,
                resolved=resolved,
                preflight=preflight,
                model_root=model_root,
                session_protocol="ordinary",
                discovery_overlay_identity_digest=(
                    plan.expected_discovery_overlay_identity_digest
                ),
                run=run,
            )
            if (
                type(raw.value) is not tuple
                or len(raw.value) != 2
                or type(raw.value[0]) is not SessionExecutionEvidence
                or type(raw.value[1]) is not DeviceStateActiveReceipt
            ):
                raise OCIBackendError("runtime returned malformed raw execution evidence")
            session, active_receipt = raw.value
            if (
                session.launch_digest != launch.digest
                or session.preflight != expected
            ):
                raise OCIBackendError("runtime returned malformed raw execution evidence")
            receipts = (raw.pre_receipt, active_receipt, raw.post_receipt)
            _validate_device_receipts(receipts, launch_id=raw.launch_id)
            return EngineExecutionEvidence(
                (
                    "optima.oci-engine-execution.v1"
                    if opened_driver is None
                    else "optima.oci-resident-engine-execution.v1"
                ),
                launch.digest,
                identity,
                preflight.sha256,
                mount.digest,
                self.config.runtime.digest,
                raw.prebuild,
                raw.publication_digest,
                raw.argv_digest,
                recovered,
                receipts,
                session,
            )
        finally:
            self._lock.release()

    def execute(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        plan: SessionExecutionPlan,
        *,
        deadline: float,
    ) -> EngineExecutionEvidence:
        """Execute one complete ordinary session (the historical API)."""

        return self._execute_ordinary(
            launch,
            binding,
            mount,
            plan,
            deadline=deadline,
            opened_driver=None,
        )

    def execute_opened(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        plan: SessionExecutionPlan,
        *,
        deadline: float,
        driver: OpenedSessionDriver,
    ) -> EngineExecutionEvidence:
        """Drive a resident engine incrementally while retaining normal teardown."""

        if not callable(driver):
            raise OCIBackendError("opened session driver must be callable")
        return self._execute_ordinary(
            launch,
            binding,
            mount,
            plan,
            deadline=deadline,
            opened_driver=driver,
        )

    def execute_reference(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        plan: ReferenceSessionPlan,
        *,
        deadline: float,
    ) -> PristineReferenceExecutionEvidence:
        """Launch a separate empty-stack T and return only its raw transcript."""

        if not self._lock.acquire(blocking=False):
            raise OCIBackendError("one executor instance cannot run concurrent sessions")
        try:
            absolute = _deadline(deadline, clock=self.manager.clock)
            recovered = self._recover_once()
            resolved, preflight, identity, expected, model_root = (
                self._validate_reference_launch(launch, binding, mount, plan)
            )

            def run(
                transport: AttachedSessionTransport,
                session_deadline: float,
                _launch_id: str,
            ) -> ReferenceSessionEvidence:
                if type(transport) is not AttachedReferenceTransport:
                    raise OCIBackendError("reference runtime received the wrong transport")
                return self.reference_session_runner(
                    plan,
                    transport=transport,
                    deadline=session_deadline,
                    init_timeout_s=self.config.runtime.init_timeout_seconds,
                    batch_timeout_s=self.config.runtime.batch_timeout_seconds,
                    clock=self.manager.clock,
                )

            raw = self._execute_runtime(
                launch,
                binding,
                mount,
                absolute=absolute,
                resolved=resolved,
                preflight=preflight,
                model_root=model_root,
                session_protocol="reference",
                discovery_overlay_identity_digest=None,
                run=run,
            )
            if (
                type(raw.value) is not ReferenceSessionEvidence
                or raw.value.launch_digest != launch.digest
                or raw.value.preflight != expected
                or raw.value.reference_manifest_digest != plan.reference.digest
                or raw.value.session_plan_digest != plan.digest
                or raw.value.request_plan_digest != plan.request_plan_digest
                or tuple(row.request for row in raw.value.exchanges) != plan.requests
            ):
                raise OCIBackendError("reference runtime returned malformed raw evidence")
            receipts = (raw.pre_receipt, raw.post_receipt)
            _validate_reference_device_receipts(receipts, launch_id=raw.launch_id)
            return PristineReferenceExecutionEvidence(
                "optima.oci-pristine-reference-execution.v1",
                launch.digest,
                identity,
                preflight.sha256,
                mount.digest,
                self.config.runtime.digest,
                raw.prebuild,
                raw.publication_digest,
                raw.argv_digest,
                recovered,
                receipts,
                raw.value,
            )
        finally:
            self._lock.release()

    def execute_resident(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        plan: ResidentSessionPlan,
        *,
        deadline: float,
        swap_intake_root: str | Path,
        driver: ResidentSessionDriver,
    ) -> ResidentEngineExecutionEvidence:
        """Launch ONE stock engine and drive a whole candidate queue through it.

        The engine loads once per call; the driver issues swap/read verbs for
        any number of candidates and must close the session before returning.
        Swap timeouts reuse the init timeout: a swap is a graph recapture, which
        is strictly cheaper than the engine boot the init timeout already
        covers. Quiescence between queue passes is unchanged — teardown at the
        end of this call proves the usual zero-state.
        """

        if not callable(driver):
            raise OCIBackendError("resident session driver must be callable")
        if type(plan) is not ResidentSessionPlan:
            raise OCIBackendError("resident session plan has the wrong type")
        if not self._lock.acquire(blocking=False):
            raise OCIBackendError("one executor instance cannot run concurrent sessions")
        try:
            absolute = _deadline(deadline, clock=self.manager.clock)
            recovered = self._recover_once()
            resolved, preflight, identity, expected, model_root = (
                self._validate_launch_identity(
                    launch,
                    binding,
                    mount,
                    engine_config_digest=plan.expected_engine_config_digest,
                    engine_tp_size=plan.engine_config.tp_size,
                    expected_preflight=plan.expected_preflight,
                )
            )
            if plan.launch_digest != launch.digest:
                raise OCIBackendError("resident session plan names another launch")
            if resolved.materialized_tree.runtime_manifest is not None:
                raise OCIBackendError(
                    "resident launch tree contains a contribution manifest"
                )
            swap_root, _ = _reopen_directory(
                swap_intake_root, field="swap intake root"
            )

            def run(
                transport: AttachedSessionTransport,
                session_deadline: float,
                _launch_id: str,
            ) -> ResidentSessionEvidence:
                controller = ResidentOuterSession(
                    plan,
                    transport=transport,
                    deadline=session_deadline,
                    init_timeout_s=self.config.runtime.init_timeout_seconds,
                    batch_timeout_s=self.config.runtime.batch_timeout_seconds,
                    swap_timeout_s=self.config.runtime.init_timeout_seconds,
                    clock=self.manager.clock,
                )
                controller.start()
                try:
                    session = driver(controller)
                    if not controller.closed:
                        raise OCIBackendError(
                            "resident driver returned without closing the engine"
                        )
                    if (
                        type(session) is not ResidentSessionEvidence
                        or session.session_id != controller.session_id
                    ):
                        raise OCIBackendError(
                            "resident driver returned another session receipt"
                        )
                finally:
                    controller.abort()
                return session

            raw = self._execute_runtime(
                launch,
                binding,
                mount,
                absolute=absolute,
                resolved=resolved,
                preflight=preflight,
                model_root=model_root,
                session_protocol="resident",
                discovery_overlay_identity_digest=None,
                run=run,
                swap_intake_root=swap_root,
            )
            if (
                type(raw.value) is not ResidentSessionEvidence
                or raw.value.launch_digest != launch.digest
                or raw.value.preflight != expected
            ):
                raise OCIBackendError(
                    "resident runtime returned malformed raw evidence"
                )
            receipts = (raw.pre_receipt, raw.post_receipt)
            _validate_reference_device_receipts(receipts, launch_id=raw.launch_id)
            return ResidentEngineExecutionEvidence(
                "optima.oci-resident-queue-execution.v1",
                launch.digest,
                identity,
                preflight.sha256,
                mount.digest,
                self.config.runtime.digest,
                raw.prebuild,
                raw.publication_digest,
                raw.argv_digest,
                recovered,
                receipts,
                raw.value,
            )
        finally:
            self._lock.release()


def stage_swap_bundle(
    swap_intake_root: str | Path,
    source_tree: str | Path,
    *,
    expected_digest: str | None = None,
) -> str:
    """Publish one validated worker tree into the content-addressed swap intake.

    The destination is ``<swap_intake_root>/<content_hash>``, published by
    atomic rename so the in-container worker never observes a partial tree.
    The worker independently re-hashes before loading, so this helper is a
    convenience, not a trust boundary.  Symlinks are preserved (not followed):
    bundle identity skips them and the load-time scan rejects them, so a
    symlinked source fails closed rather than folding foreign bytes in.

    Worker-storage metadata is not part of bundle identity: the root-level
    native-artifact receipt that immutable publication plants next to the
    committed bytes is skipped, so staging from a worker publication
    reproduces the chain-committed content hash exactly.  The digest is
    computed over the staged COPY (the bytes actually published), never over
    the source it was copied from.
    """

    from optima.bundle_hash import content_hash
    from optima.eval.native_artifact import _MANIFEST as _PUBLICATION_RECEIPT

    root, _ = _reopen_directory(swap_intake_root, field="swap intake root")
    source, _ = _reopen_directory(source_tree, field="staged bundle source")

    def _ignore_receipt(directory: str, names: list[str]) -> set[str]:
        if Path(directory) == source:
            return {name for name in names if name == _PUBLICATION_RECEIPT}
        return set()

    staging = root / f".staging-{secrets.token_hex(16)}"
    try:
        shutil.copytree(source, staging, symlinks=True, ignore=_ignore_receipt)
        try:
            digest = content_hash(staging)
        except (OSError, ValueError) as exc:
            raise OCIBackendError(
                f"staged bundle source is unhashable: {exc}"
            ) from None
        if expected_digest is not None and digest != _digest(
            expected_digest, field="expected bundle digest"
        ):
            raise OCIBackendError("staged bundle differs from its committed digest")
        destination = root / digest
        if destination.is_symlink():
            raise OCIBackendError("swap intake destination must not be a symlink")
        if destination.exists():
            try:
                if content_hash(destination) != digest:
                    raise OCIBackendError(
                        "swap intake already holds different bytes for this digest"
                    )
            except (OSError, ValueError) as exc:
                raise OCIBackendError(
                    f"existing staged bundle is unreadable: {exc}"
                ) from None
            return digest
        os.rename(staging, destination)
        return digest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _new_runtime_id() -> str:
    value = "runtime-" + secrets.token_hex(16)
    if _OPAQUE_ID.fullmatch(value) is None:
        raise OCIBackendError("system RNG returned an invalid runtime label")
    return value


def _validate_mount_roots(*roots: Path) -> None:
    resolved = tuple(_reopen_directory(root, field="immutable/writable root")[0] for root in roots)
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            try:
                common = Path(os.path.commonpath((left, right)))
            except ValueError:
                continue
            if common in {left, right}:
                raise OCIBackendError("backend host roots must be pairwise disjoint")
    forbidden = {Path.cwd().resolve(), Path(__file__).resolve().parents[2]}
    for root in resolved[:3]:
        for blocked in forbidden:
            try:
                common = Path(os.path.commonpath((root, blocked)))
            except ValueError:
                continue
            if common in {root, blocked}:
                raise OCIBackendError("runtime mounts must not expose cwd/referee source")


def _validate_device_receipts(
    receipts: tuple[DeviceStateReceipt, DeviceStateActiveReceipt, DeviceStateReceipt],
    *,
    launch_id: str,
) -> None:
    pre, active, post = receipts
    if (
        type(pre) is not DeviceStateReceipt
        or type(active) is not DeviceStateActiveReceipt
        or type(post) is not DeviceStateReceipt
        or (pre.phase, active.event, post.phase) != ("pre", "final-warmup", "post")
        or any(row.launch_id != launch_id for row in receipts)
        or not (pre.sequence < active.sequence < post.sequence)
        or len({row.selected_physical_gpu_ids for row in receipts}) != 1
        or len({row.configuration_sha256 for row in receipts}) != 1
        or len({row.policy_sha256 for row in receipts}) != 1
        or not (
            pre.completed_monotonic_s
            <= active.started_monotonic_s
            <= active.completed_monotonic_s
            <= post.started_monotonic_s
            <= post.completed_monotonic_s
        )
    ):
        raise OCIBackendError("device pre/active/post receipt triplet is invalid")


def _validate_reference_device_receipts(
    receipts: tuple[DeviceStateReceipt, DeviceStateReceipt],
    *,
    launch_id: str,
) -> None:
    pre, post = receipts
    if (
        type(pre) is not DeviceStateReceipt
        or type(post) is not DeviceStateReceipt
        or (pre.phase, post.phase) != ("pre", "post")
        or pre.launch_id != launch_id
        or post.launch_id != launch_id
        or pre.sequence >= post.sequence
        or pre.selected_physical_gpu_ids != post.selected_physical_gpu_ids
        or pre.configuration_sha256 != post.configuration_sha256
        or pre.policy_sha256 != post.policy_sha256
        or pre.completed_monotonic_s > post.started_monotonic_s
        or post.started_monotonic_s > post.completed_monotonic_s
    ):
        raise OCIBackendError("reference device pre/post receipt pair is invalid")


__all__ = [
    "CandidateFreeRuntimeIdentity",
    "EngineExecutionEvidence",
    "OCIBackendConfig",
    "OCIBackendDeadlineError",
    "OCIBackendError",
    "OCIBackendRuntimeError",
    "OCIEngineExecutor",
    "OCIRuntimeResourcePolicy",
    "PristineReferenceExecutionEvidence",
    "ResidentEngineExecutionEvidence",
    "ResidentSessionDriver",
    "TrustedArenaModelMountReceipt",
    "build_runtime_argv",
    "expected_runtime_preflight",
    "runtime_identity_from_preflight",
    "stage_swap_bundle",
]
