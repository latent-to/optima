"""Host-hermetic native prebuild for one resolved engine launch.

The trusted controller supplies only a digest-pinned image, an already materialized
engine tree, a validator-owned seccomp profile, and one lease-scoped quota-backed
output mount.  The disposable container has no network, GPU, model, home, wallet,
Docker socket, or ambient host mount.  It parses candidate files as data and invokes
only validator-registered rebuild patchers in ``build`` phase.  The host never imports
candidate Python or loads a produced native object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from optima.artifact_provider import (
    ARTIFACT_PROVIDERS,
    ArtifactKind,
    ArtifactProviderDescriptor,
    ArtifactProviderPolicyError,
)
from optima.cute_aot import (
    CUTE_COMPILE_PROFILE_DIGEST_ENV,
    CUTE_COMPILE_PROFILE_ENV,
    CuteAOTError,
    load_compile_profile,
)
from optima.cute_cubin import (
    CuteCubinError,
    reopen_cute_cubin_index,
)
from optima.eval.engine_launch import (
    EngineLaunchSpec,
    ResolvedEngineLaunch,
    TrustedLaunchBinding,
    reopen_launch_tree,
    resolve_engine_launch,
)
from optima.eval.native_artifact import (
    NativeArtifactLimits,
    NativeArtifactPublication,
    publish_native_artifact,
    reopen_native_artifact,
)
from optima.eval.oci_cpuset import validate_cpuset_pair
from optima.eval.oci_process import OCILease, OCIProcessManager
from optima.eval.runtime_preflight import RuntimePreflightReceipt
from optima.stack_identity import canonical_digest, canonical_json_bytes, require_sha256_hex


CONTAINER_TREE = "/optima/engine-tree"
CONTAINER_STAGE = "/optima/native-stage"
CONTAINER_CUTE_COMPILE_PROFILE = "/optima/cute-compile-profile.json"
PREBUILD_RECEIPT = "prebuild.json"
PREBUILD_SCHEMA = "optima.oci-native-prebuild.v1"
_PUBLICATION_MANIFEST = ".optima-native-artifact.json"
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_ARCH = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SIZE = re.compile(r"[1-9][0-9]{0,15}\Z")
_ARTIFACT_PUBLICATION_DIRECTORIES = frozenset(
    descriptor.publication_directory
    for descriptor in ARTIFACT_PROVIDERS.descriptors()
)
_ALLOWED_STAGE_TOP_LEVEL = frozenset(
    {
        "cuda",
        "dep_modules",
        "dep_overlays",
        PREBUILD_RECEIPT,
    }
) | _ARTIFACT_PUBLICATION_DIRECTORIES
_DISCOVERY_ENGINE_METADATA = "metadata/optima_discovery.json"
_DISCOVERY_RECEIPT_FIELD = "discovery_overlay_identity_digest"
_CUTE_AOT_RECEIPT_FIELD = "cute_aot_compile_profile_digest"


class OCIPrebuildError(RuntimeError):
    """Trusted prebuild configuration, execution, or evidence is invalid."""


def _artifact_provider_descriptors(
    manifest: object, rebuild_plan: object
) -> tuple[ArtifactProviderDescriptor, ...]:
    """Resolve manifest provider IDs to the closed prebuild policy table.

    This function deliberately returns descriptors, not compiler/loader callbacks.
    The reviewed rebuild plan remains the only authority that can execute a build.
    """

    ops = getattr(manifest, "ops", None)
    if not isinstance(ops, tuple):
        raise OCIPrebuildError("artifact provider resolution requires a typed manifest")
    provider_ids = tuple(
        sorted(
            {
                export.provider
                for op in ops
                for export in getattr(op, "aot_exports", ())
            }
        )
    )
    try:
        descriptors = tuple(
            ARTIFACT_PROVIDERS.require(provider_id)
            for provider_id in provider_ids
        )
    except ArtifactProviderPolicyError as exc:
        raise OCIPrebuildError(
            f"engine tree selected an unregistered artifact provider: {exc}"
        ) from None
    steps = () if rebuild_plan is None else getattr(rebuild_plan, "steps", None)
    if not isinstance(steps, tuple):
        raise OCIPrebuildError("artifact provider resolution requires a typed rebuild plan")
    patcher_ids = {getattr(step, "patcher_id", None) for step in steps}
    missing_provider_patchers = tuple(
        descriptor.build_patcher_id
        for descriptor in descriptors
        if descriptor.build_patcher_id not in patcher_ids
    )
    if missing_provider_patchers:
        raise OCIPrebuildError(
            "artifact providers lack their reviewed build patchers "
            f"{missing_provider_patchers!r}"
        )
    return descriptors


def _compile_profile_provider(
    descriptors: tuple[ArtifactProviderDescriptor, ...],
) -> ArtifactProviderDescriptor | None:
    selected = tuple(
        descriptor
        for descriptor in descriptors
        if descriptor.requires_compile_profile
    )
    if not selected:
        return None
    provider_ids = {descriptor.provider_id for descriptor in selected}
    if len(provider_ids) != 1:
        raise OCIPrebuildError(
            "one engine tree cannot mix artifact compile-profile providers"
        )
    return selected[0]


def _reopen_provider_publication(
    descriptor: ArtifactProviderDescriptor,
    root: Path,
    *,
    build_spec_digest: str,
    tree_digest: str,
    logical_architecture: str,
    compile_profile_digest: str,
    verify_distributions: bool,
):
    common = {
        "expected_build_spec_digest": build_spec_digest,
        "expected_tree_digest": tree_digest,
        "expected_logical_architecture": logical_architecture,
        "expected_compile_profile_digest": compile_profile_digest,
        "verify_distributions": verify_distributions,
    }
    if descriptor.artifact_kind is ArtifactKind.CUDA_CUBIN:
        try:
            return reopen_cute_cubin_index(root, **common)
        except CuteCubinError as exc:
            raise OCIPrebuildError(
                f"CuTe CUBIN publication cannot reopen: {exc}"
            ) from None
    raise OCIPrebuildError(
        "registered artifact kind has no sealed prebuild validator: "
        f"{descriptor.artifact_kind.value!r}"
    )


def _digest(value: object, *, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise OCIPrebuildError(str(exc)) from None
    if result == "0" * 64:
        raise OCIPrebuildError(f"{field} must not be all zeroes")
    return result


def _absolute_container_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or ":" in value:
        raise OCIPrebuildError(f"{field} must be a canonical absolute container path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise OCIPrebuildError(f"{field} must be a canonical absolute container path")
    return value


@dataclass(frozen=True)
class OCIPrebuildPolicy:
    """Path-free resource and dependency policy bound by a launch specification.

    ``runtime_policy_digest`` reserves the independently specified PR2b runtime
    policy, so the launch's resource identity can bind both stages without changing
    the PR2a format later.
    """

    uid: int
    gid: int
    cpu_millis: int
    memory_bytes: int
    pids_limit: int
    tmpfs_bytes: int
    stage_bytes: int
    stage_inodes: int
    timeout_seconds: float
    native_compile_timeout_seconds: int
    container_python: str
    build_path: tuple[str, ...]
    build_tmpdir: str
    pinned_build_roots: tuple[str, ...]
    runtime_policy_digest: str
    cpuset_cpus: str | None = None
    cpuset_mems: str | None = None

    def __post_init__(self) -> None:
        for field in ("uid", "gid"):
            value = getattr(self, field)
            if type(value) is not int or not 1 <= value <= 2_147_483_647:
                raise OCIPrebuildError(f"{field} must be a nonzero integer")
        bounds = {
            "cpu_millis": (100, 256_000),
            "memory_bytes": (256 << 20, 1 << 50),
            "pids_limit": (32, 1_048_576),
            "tmpfs_bytes": (16 << 20, 1 << 40),
            "stage_bytes": (16 << 20, 1 << 40),
            "stage_inodes": (1_024, 10_000_000),
            "native_compile_timeout_seconds": (1, 7_200),
        }
        for field, (low, high) in bounds.items():
            value = getattr(self, field)
            if type(value) is not int or not low <= value <= high:
                raise OCIPrebuildError(f"{field} is outside its hard bound")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(float(self.timeout_seconds))
            or not 1 <= float(self.timeout_seconds) <= 86_400
        ):
            raise OCIPrebuildError("timeout_seconds must be finite and in [1, 86400]")
        object.__setattr__(
            self,
            "container_python",
            _absolute_container_path(self.container_python, field="container_python"),
        )
        if type(self.build_path) is not tuple or not self.build_path:
            raise OCIPrebuildError("build_path must be a nonempty tuple")
        build_path = tuple(
            _absolute_container_path(value, field="build PATH entry")
            for value in self.build_path
        )
        if len(set(build_path)) != len(build_path):
            raise OCIPrebuildError("build_path entries must be unique")
        object.__setattr__(self, "build_path", build_path)
        object.__setattr__(
            self,
            "build_tmpdir",
            _absolute_container_path(self.build_tmpdir, field="build_tmpdir"),
        )
        if type(self.pinned_build_roots) is not tuple or not self.pinned_build_roots:
            raise OCIPrebuildError("pinned_build_roots must be a nonempty tuple")
        roots = tuple(
            _absolute_container_path(value, field="pinned build root")
            for value in self.pinned_build_roots
        )
        if roots != tuple(sorted(set(roots))):
            raise OCIPrebuildError("pinned_build_roots must be sorted and unique")
        object.__setattr__(self, "pinned_build_roots", roots)
        object.__setattr__(
            self,
            "runtime_policy_digest",
            _digest(self.runtime_policy_digest, field="runtime_policy_digest"),
        )
        try:
            cpus, mems = validate_cpuset_pair(
                self.cpuset_cpus,
                self.cpuset_mems,
                cpu_millis=self.cpu_millis,
            )
        except ValueError as exc:
            raise OCIPrebuildError(str(exc)) from None
        object.__setattr__(self, "cpuset_cpus", cpus)
        object.__setattr__(self, "cpuset_mems", mems)

    def resource_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "cpu_millis": self.cpu_millis,
            "container_python": self.container_python,
            "gid": self.gid,
            "memory_bytes": self.memory_bytes,
            "pids_limit": self.pids_limit,
            "runtime_policy_digest": self.runtime_policy_digest,
            "stage_bytes": self.stage_bytes,
            "stage_inodes": self.stage_inodes,
            "timeout_milliseconds": int(round(float(self.timeout_seconds) * 1000)),
            "tmpfs_bytes": self.tmpfs_bytes,
            "uid": self.uid,
        }
        if self.cpuset_cpus is not None:
            payload["cpuset_cpus"] = self.cpuset_cpus
            payload["cpuset_mems"] = self.cpuset_mems
        return payload

    @property
    def resource_policy_digest(self) -> str:
        return canonical_digest("optima.eval.executor-resource-policy", self.resource_payload())

    @property
    def dependency_policy_digest(self) -> str:
        return canonical_digest(
            "optima.eval.native-dependency-policy",
            {
                "build_path": list(self.build_path),
                "build_tmpdir": self.build_tmpdir,
                "container_python": self.container_python,
                "native_compile_timeout_seconds": self.native_compile_timeout_seconds,
                "pinned_build_roots": list(self.pinned_build_roots),
            },
        )


@dataclass(frozen=True)
class OCIPrebuildConfig:
    docker_binary: str
    recovery_root: Path
    publication_root: Path
    seccomp_profile: Path
    executor_id: str
    policy: OCIPrebuildPolicy

    def __post_init__(self) -> None:
        docker = _absolute_container_path(self.docker_binary, field="docker_binary")
        if PurePosixPath(docker).name != "docker":
            raise OCIPrebuildError("docker_binary must end in /docker")
        object.__setattr__(self, "docker_binary", docker)
        if not isinstance(self.executor_id, str) or _SAFE_ID.fullmatch(self.executor_id) is None:
            raise OCIPrebuildError("executor_id must be a lowercase simple identifier")
        if not isinstance(self.policy, OCIPrebuildPolicy):
            raise OCIPrebuildError("policy must be OCIPrebuildPolicy")
        for field in ("recovery_root", "publication_root", "seccomp_profile"):
            try:
                value = Path(getattr(self, field)).expanduser()
            except TypeError:
                raise OCIPrebuildError(f"{field} must be a trusted host path") from None
            if not value.is_absolute():
                raise OCIPrebuildError(f"{field} must be absolute")
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class OCIPrebuildResult:
    launch_digest: str
    build_spec_digest: str
    publication: NativeArtifactPublication
    container_elapsed_seconds: float | None
    security_argv_digest: str | None
    discovery_overlay_identity_digest: str | None = None

    def __post_init__(self) -> None:
        if self.discovery_overlay_identity_digest is not None:
            object.__setattr__(
                self,
                "discovery_overlay_identity_digest",
                _digest(
                    self.discovery_overlay_identity_digest,
                    field="discovery overlay identity digest",
                ),
            )

    @property
    def reused(self) -> bool:
        return self.container_elapsed_seconds is None or self.publication.reused


def _sha256_file_stable(path: Path) -> str:
    if path.is_symlink():
        raise OCIPrebuildError(f"trusted file must not be a symlink: {path}")
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OCIPrebuildError(f"trusted file must be regular and single-linked: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise OCIPrebuildError(f"cannot hash trusted file {path}: {exc}") from None
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise OCIPrebuildError(f"trusted file changed while hashing: {path}")
    return digest.hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    try:
        common = Path(os.path.commonpath((left, right)))
    except ValueError:
        return False
    return common == left or common == right


def _copy_seccomp(source: Path, destination: Path, *, expected_digest: str) -> None:
    if _sha256_file_stable(source) != expected_digest:
        raise OCIPrebuildError("seccomp profile digest does not match launch specification")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(destination, flags, 0o400)
    try:
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            raise OCIPrebuildError("seccomp profile changed while copying")
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OCIPrebuildError("seccomp profile copy stalled")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if _sha256_file_stable(destination) != expected_digest:
        raise OCIPrebuildError("lease seccomp copy failed its digest check")


def _write_compile_profile(profile: object, destination: Path) -> None:
    """Write one already validated profile into the private lease staging area."""

    from optima.eval.native_compile_profile import NativeCuTeCompileProfile

    if type(profile) is not NativeCuTeCompileProfile:
        raise OCIPrebuildError("lease compile profile has the wrong type")
    raw = profile.canonical_bytes
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(destination, flags, 0o444)
    except OSError as exc:
        raise OCIPrebuildError(f"cannot create lease compile profile: {exc}") from None
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OCIPrebuildError("lease compile profile write stalled")
            view = view[written:]
        # Creation mode is filtered through the controller's umask.  The worker
        # runs as an unprivileged UID, so override that mask explicitly before
        # bind-mounting this validator-owned data file read-only.
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    except OSError as exc:
        raise OCIPrebuildError(f"cannot seal lease compile profile: {exc}") from None
    finally:
        os.close(fd)
    try:
        observed = destination.read_bytes()
    except OSError as exc:
        raise OCIPrebuildError(f"cannot reopen lease compile profile: {exc}") from None
    if observed != raw:
        raise OCIPrebuildError("lease compile profile changed while staging")
    try:
        reopened = load_compile_profile(destination, expected_digest=profile.digest)
    except CuteAOTError as exc:
        raise OCIPrebuildError(f"lease compile profile cannot reopen: {exc}") from None
    if reopened != profile:
        raise OCIPrebuildError("lease compile profile round trip changed its value")


def _validate_binding(
    launch: EngineLaunchSpec,
    binding: TrustedLaunchBinding,
    config: OCIPrebuildConfig,
) -> tuple[ResolvedEngineLaunch, RuntimePreflightReceipt]:
    resolved = resolve_engine_launch(launch, binding)
    receipt = binding.runtime_preflight_receipt
    if not isinstance(receipt, RuntimePreflightReceipt):
        raise OCIPrebuildError("prebuild requires a typed runtime preflight receipt")
    if receipt.docker_binary != config.docker_binary:
        raise OCIPrebuildError("preflight and prebuild Docker clients differ")
    if (receipt.uid, receipt.gid) != (config.policy.uid, config.policy.gid):
        raise OCIPrebuildError("preflight and prebuild UID/GID policies differ")
    if receipt.python_executable != config.policy.container_python:
        raise OCIPrebuildError("preflight and prebuild Python interpreters differ")
    if _IMAGE_ID.fullmatch(receipt.local_image_id) is None:
        raise OCIPrebuildError("preflight local image ID is malformed")
    if launch.resource_policy_digest != config.policy.resource_policy_digest:
        raise OCIPrebuildError("prebuild resource policy does not match launch specification")
    if resolved.native_build_spec.dependency_policy_digest != config.policy.dependency_policy_digest:
        raise OCIPrebuildError("native dependency policy does not match launch specification")
    if _sha256_file_stable(config.seccomp_profile) != launch.seccomp_policy_digest:
        raise OCIPrebuildError("seccomp profile does not match launch specification")
    for name, path in (
        ("publication_root", config.publication_root),
        ("recovery_root", config.recovery_root),
    ):
        if _paths_overlap(resolved.materialized_tree_root, path):
            raise OCIPrebuildError(
                f"{name} must not overlap the identity-bound materialized tree"
            )
    if _paths_overlap(config.publication_root, config.recovery_root):
        raise OCIPrebuildError(
            "publication_root and recovery_root must not overlap"
        )
    return resolved, receipt


def _prepare_publication_root_for_lease(
    publication_root: Path, *, recovery_resources_root: Path
) -> None:
    """Fail before container work if lease staging cannot publish atomically."""
    try:
        publication_root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise OCIPrebuildError(f"cannot create publication_root: {exc}") from None
    try:
        publication = publication_root.lstat()
        recovery = recovery_resources_root.stat()
    except OSError as exc:
        raise OCIPrebuildError(
            f"cannot inspect publication/recovery filesystems: {exc}"
        ) from None
    if (
        not stat.S_ISDIR(publication.st_mode)
        or stat.S_ISLNK(publication.st_mode)
        or stat.S_IMODE(publication.st_mode) & 0o022
        or (hasattr(os, "geteuid") and publication.st_uid != os.geteuid())
    ):
        raise OCIPrebuildError(
            "publication_root must be a validator-owned private directory"
        )
    if publication.st_dev != recovery.st_dev:
        raise OCIPrebuildError(
            "publication_root and recovery_root must share a filesystem"
        )


def _absolute_deadline(value: float | None) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise OCIPrebuildError("prebuild deadline must be a finite monotonic timestamp")
    return float(value)


def _remaining_deadline(
    manager: OCIProcessManager | None,
    deadline: float | None,
    *,
    stage: str,
) -> float | None:
    if deadline is None:
        return None
    if manager is None:  # Internal invariant: a deadline always selects one clock.
        raise OCIPrebuildError("prebuild deadline has no OCI manager clock")
    try:
        now = float(manager.clock())
    except Exception as exc:
        raise OCIPrebuildError(f"prebuild monotonic clock failed: {exc}") from None
    if not math.isfinite(now):
        raise OCIPrebuildError("prebuild monotonic clock returned a non-finite value")
    remaining = deadline - now
    if not math.isfinite(remaining) or remaining <= 0:
        raise OCIPrebuildError(f"prebuild deadline expired during {stage}")
    return remaining


def build_prebuild_argv(
    *,
    lease: OCILease,
    resolved: ResolvedEngineLaunch,
    preflight: RuntimePreflightReceipt,
    config: OCIPrebuildConfig,
    stage_path: Path,
    seccomp_path: Path,
    compile_profile_path: Path | None = None,
) -> tuple[str, ...]:
    """Construct the closed, candidate-independent Docker argv."""
    policy = config.policy
    env = {
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "/tmp/home",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "OPTIMA_ENGINE_TREE_DIGEST": resolved.spec.tree_digest,
        "OPTIMA_BUILD_PATH": ":".join(policy.build_path),
        "OPTIMA_BUILD_TMPDIR": policy.build_tmpdir,
        "OPTIMA_NATIVE_ARTIFACT_STAGE": CONTAINER_STAGE,
        "OPTIMA_NATIVE_BUILD_SPEC_DIGEST": resolved.native_build_spec.digest,
        "OPTIMA_NATIVE_COMPILE_TIMEOUT_S": str(policy.native_compile_timeout_seconds),
        "OPTIMA_PINNED_BUILD_ROOTS": ":".join(policy.pinned_build_roots),
        "OPTIMA_REBUILD_CONTAINER": "1",
        "OPTIMA_TARGET_GPU_ARCH": resolved.native_build_spec.target_architecture,
        "PATH": ":".join(policy.build_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": "/tmp",
    }
    compile_profile = resolved.native_compile_profile
    if (compile_profile is None) != (compile_profile_path is None):
        raise OCIPrebuildError("prebuild compile profile mount does not match launch authority")
    if compile_profile is not None:
        assert compile_profile_path is not None
        if not compile_profile_path.is_absolute():
            raise OCIPrebuildError("prebuild compile profile host path must be absolute")
        env[CUTE_COMPILE_PROFILE_ENV] = CONTAINER_CUTE_COMPILE_PROFILE
        env[CUTE_COMPILE_PROFILE_DIGEST_ENV] = compile_profile.digest
    argv = [
        *lease.run_prefix(config.docker_binary),
        "--rm",
        "--pull=never",
        f"--platform={preflight.oci_platform}",
        "--runtime=runc",
        "--network=none",
        "--read-only",
        "--ipc=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        f"--security-opt=seccomp={seccomp_path}",
        f"--user={policy.uid}:{policy.gid}",
        f"--cpus={policy.cpu_millis / 1000:g}",
        *(
            (
                f"--cpuset-cpus={policy.cpuset_cpus}",
                f"--cpuset-mems={policy.cpuset_mems}",
            )
            if policy.cpuset_cpus is not None
            else ()
        ),
        f"--memory={policy.memory_bytes}",
        f"--memory-swap={policy.memory_bytes}",
        f"--pids-limit={policy.pids_limit}",
        f"--tmpfs=/tmp:rw,nosuid,nodev,noexec,size={policy.tmpfs_bytes},uid={policy.uid},gid={policy.gid},mode=0700",
        "--workdir=/tmp",
        "--no-healthcheck",
        "--stop-timeout=1",
        "--log-driver=none",
        f"--mount=type=bind,src={resolved.materialized_tree_root},dst={CONTAINER_TREE},readonly,bind-propagation=rprivate",
        f"--mount=type=bind,src={stage_path},dst={CONTAINER_STAGE},bind-propagation=rprivate",
    ]
    if compile_profile_path is not None:
        argv.append(
            f"--mount=type=bind,src={compile_profile_path},"
            f"dst={CONTAINER_CUTE_COMPILE_PROFILE},readonly,bind-propagation=rprivate"
        )
    argv.extend(f"--env={key}={env[key]}" for key in sorted(env))
    argv.extend(
        (
            f"--entrypoint={policy.container_python}",
            preflight.local_image_id,
            "-I",
            "-m",
            "optima.eval.oci_prebuild",
            "--container-build",
        )
    )
    return tuple(argv)


def _discovery_binding(
    resolved: ResolvedEngineLaunch,
    *,
    preflight: RuntimePreflightReceipt | None = None,
):
    """Reopen discovery authority only when its typed metadata is inventoried."""

    if not any(
        row.path == _DISCOVERY_ENGINE_METADATA
        for row in resolved.materialized_tree.files
    ):
        return None
    from optima.discovery import DiscoveryError, reopen_discovery_engine_binding

    try:
        binding = reopen_discovery_engine_binding(resolved.materialized_tree)
    except (DiscoveryError, OSError, TypeError, ValueError) as exc:
        raise OCIPrebuildError(
            f"discovery engine binding cannot reopen: {exc}"
        ) from None
    profile = binding.build_profile
    if (
        profile.architecture != resolved.native_build_spec.target_architecture
        or profile.tensor_parallel_size != resolved.spec.hardware.tp_size
    ):
        raise OCIPrebuildError(
            "discovery build profile differs from launch architecture or TP size"
        )
    if preflight is not None and profile.sglang_version != preflight.sglang_version:
        raise OCIPrebuildError(
            "discovery build profile differs from image SGLang preflight"
        )
    return binding


def _read_prebuild_receipt(
    root: Path, *, resolved: ResolvedEngineLaunch
) -> tuple[dict[str, object], object | None]:
    path = root / PREBUILD_RECEIPT
    try:
        raw = path.read_bytes()
        row = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise OCIPrebuildError(f"native prebuild receipt is unreadable: {exc}") from None
    observed = {child.name for child in root.iterdir()}
    observed.discard(_PUBLICATION_MANIFEST)
    artifact_entries = observed & _ARTIFACT_PUBLICATION_DIRECTORIES
    has_artifact_publication = bool(artifact_entries)
    keys = {
        "schema",
        "build_spec_digest",
        "rebuild_applied",
        "stage_entries",
        "target_architecture",
        "tree_digest",
    }
    discovery = _discovery_binding(resolved)
    if discovery is not None:
        keys.add(_DISCOVERY_RECEIPT_FIELD)
    if has_artifact_publication:
        keys.add(_CUTE_AOT_RECEIPT_FIELD)
    if not isinstance(row, dict) or set(row) != keys or row.get("schema") != PREBUILD_SCHEMA:
        raise OCIPrebuildError("native prebuild receipt schema mismatch")
    if raw != canonical_json_bytes(row) + b"\n":
        raise OCIPrebuildError("native prebuild receipt is not canonical")
    expected = {
        "build_spec_digest": resolved.native_build_spec.digest,
        "target_architecture": resolved.native_build_spec.target_architecture,
        "tree_digest": resolved.spec.tree_digest,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise OCIPrebuildError(f"native prebuild receipt {field} mismatch")
    entries = row.get("stage_entries")
    if not isinstance(entries, list) or entries != sorted(set(entries)):
        raise OCIPrebuildError("native prebuild receipt stage inventory is invalid")
    if any(not isinstance(value, str) or value not in _ALLOWED_STAGE_TOP_LEVEL for value in entries):
        raise OCIPrebuildError("native prebuild receipt contains an unexpected stage entry")
    if observed != set(entries):
        raise OCIPrebuildError("native prebuild stage differs from its receipt")
    if type(row.get("rebuild_applied")) is not bool:
        raise OCIPrebuildError("native prebuild receipt rebuild flag is invalid")
    if discovery is not None:
        _digest(
            row.get(_DISCOVERY_RECEIPT_FIELD),
            field="discovery overlay identity digest",
        )
    compile_profile = resolved.native_compile_profile
    if (compile_profile is None) == has_artifact_publication:
        raise OCIPrebuildError(
            "artifact publication does not match launch compile-profile authority"
        )
    if compile_profile is not None:
        receipt_profile_digest = _digest(
            row.get(_CUTE_AOT_RECEIPT_FIELD),
            field="CuTe AOT receipt compile profile digest",
        )
        if receipt_profile_digest != compile_profile.digest:
            raise OCIPrebuildError("CuTe AOT receipt compile profile digest mismatch")
        try:
            descriptor = ARTIFACT_PROVIDERS.require(compile_profile.provider)
        except ArtifactProviderPolicyError as exc:
            raise OCIPrebuildError(str(exc)) from None
        if artifact_entries != {descriptor.publication_directory}:
            raise OCIPrebuildError(
                "artifact publication directory differs from compile-profile provider"
            )
        index = _reopen_provider_publication(
            descriptor,
            root,
            build_spec_digest=resolved.native_build_spec.digest,
            tree_digest=resolved.spec.tree_digest,
            logical_architecture=resolved.native_build_spec.target_architecture,
            compile_profile_digest=compile_profile.digest,
            verify_distributions=False,
        )
        if index.compile_profile_digest != receipt_profile_digest:
            raise OCIPrebuildError("CuTe AOT index differs from prebuild receipt")
        if index.compiler_architecture != compile_profile.compiler_architecture:
            raise OCIPrebuildError("CuTe AOT compiler architecture differs from profile")
        for export in index.exports:
            expected_profile = {
                key: compile_profile.require_int(key)
                for key in export.profile_inputs
            }
            if dict(export.resolved_profile) != expected_profile:
                raise OCIPrebuildError(
                    "CuTe AOT export values differ from validator compile profile"
                )
    return row, discovery


def _reopen_discovery_publication(
    publication: NativeArtifactPublication,
    receipt: dict[str, object],
    discovery: object | None,
) -> str | None:
    """Trust a discovery receipt only after the immutable publication reopens."""

    if discovery is None:
        if _DISCOVERY_RECEIPT_FIELD in receipt or any(
            row.path.startswith("dep_overlays/discovery/")
            for row in publication.files
        ):
            raise OCIPrebuildError("ordinary prebuild acquired discovery state")
        return None
    from optima.discovery import (
        DiscoveryEngineBinding,
        DiscoveryError,
        reopen_discovery_overlay,
    )

    if type(discovery) is not DiscoveryEngineBinding:
        raise OCIPrebuildError("discovery publication lacks a typed engine binding")
    expected = _digest(
        receipt.get(_DISCOVERY_RECEIPT_FIELD),
        field="discovery overlay identity digest",
    )
    try:
        overlay = reopen_discovery_overlay(
            publication,
            expected_identity_digest=expected,
        )
    except (DiscoveryError, OSError, TypeError, ValueError) as exc:
        raise OCIPrebuildError(
            f"discovery overlay publication cannot reopen: {exc}"
        ) from None
    identity = overlay.identity
    if (
        identity.proposal_digest != discovery.discovery.proposal_digest
        or identity.policy_digest != discovery.policy.digest
        or identity.build_profile_digest != discovery.build_profile.digest
    ):
        raise OCIPrebuildError(
            "discovery overlay identity differs from its engine-tree authority"
        )
    return overlay.identity_digest


def run_oci_prebuild(
    launch: EngineLaunchSpec,
    binding: TrustedLaunchBinding,
    config: OCIPrebuildConfig,
    *,
    manager: OCIProcessManager | None = None,
    limits: NativeArtifactLimits | None = None,
    deadline: float | None = None,
) -> OCIPrebuildResult:
    """Build, seal, publish, and reopen one native artifact tree."""
    if not isinstance(config, OCIPrebuildConfig):
        raise OCIPrebuildError("config must be OCIPrebuildConfig")
    deadline = _absolute_deadline(deadline)
    # A caller-owned deadline and every phase timeout must read the same monotonic
    # clock.  Construct the manager before identity work only when that deadline is
    # present; the legacy no-deadline cache-hit path remains side-effect free.
    if deadline is not None and manager is None:
        manager = OCIProcessManager(
            docker_binary=config.docker_binary,
            recovery_root=config.recovery_root,
            executor_id=config.executor_id,
        )
    if manager is not None and (
        manager.docker_binary != config.docker_binary
        or manager.executor_id != config.executor_id
    ):
        raise OCIPrebuildError("OCI process manager does not match prebuild config")
    _remaining_deadline(manager, deadline, stage="binding validation")
    resolved, preflight = _validate_binding(launch, binding, config)
    expected_discovery = _discovery_binding(resolved, preflight=preflight)
    _remaining_deadline(manager, deadline, stage="binding validation")
    digest = resolved.native_build_spec.digest
    existing = config.publication_root / digest[:2] / digest
    if existing.exists() or existing.is_symlink():
        _remaining_deadline(manager, deadline, stage="cached artifact reopen")
        publication = reopen_native_artifact(
            existing, expected_build_spec_digest=digest, limits=limits
        )
        receipt, receipt_discovery = _read_prebuild_receipt(
            publication.root, resolved=resolved
        )
        if receipt_discovery != expected_discovery:
            raise OCIPrebuildError("cached discovery binding changed during reopen")
        discovery_digest = _reopen_discovery_publication(
            publication, receipt, receipt_discovery
        )
        _remaining_deadline(manager, deadline, stage="cached artifact validation")
        return OCIPrebuildResult(
            launch.digest,
            digest,
            publication,
            None,
            None,
            discovery_digest,
        )

    if manager is None:
        manager = OCIProcessManager(
            docker_binary=config.docker_binary,
            recovery_root=config.recovery_root,
            executor_id=config.executor_id,
        )
    _remaining_deadline(manager, deadline, stage="publication-root preparation")
    _prepare_publication_root_for_lease(
        config.publication_root, recovery_resources_root=manager.resources_root
    )
    _remaining_deadline(manager, deadline, stage="lease registration")
    lease_id = "prebuild-" + secrets.token_hex(10)
    lease = manager.register(
        lease_id=lease_id,
        container_name="optima-" + lease_id,
        mount_relpaths=("native-stage",),
        stage_relpaths=(
            "seccomp.json",
            "publication-work",
            *(
                ("cute-compile-profile.json",)
                if resolved.native_compile_profile is not None
                else ()
            ),
        ),
    )
    stage_path = lease.mount_paths[0]
    seccomp_copy = lease.stage_paths[0]
    publication_work = lease.stage_paths[1]
    compile_profile_copy = (
        lease.stage_paths[2]
        if resolved.native_compile_profile is not None
        else None
    )
    try:
        _remaining_deadline(manager, deadline, stage="seccomp staging")
        _copy_seccomp(
            config.seccomp_profile,
            seccomp_copy,
            expected_digest=launch.seccomp_policy_digest,
        )
        if resolved.native_compile_profile is not None:
            assert compile_profile_copy is not None
            _remaining_deadline(manager, deadline, stage="compile-profile staging")
            _write_compile_profile(
                resolved.native_compile_profile,
                compile_profile_copy,
            )
        _remaining_deadline(manager, deadline, stage="native-stage mount")
        manager.mount_tmpfs(
            lease,
            stage_path,
            size_bytes=config.policy.stage_bytes,
            inode_limit=config.policy.stage_inodes,
            uid=config.policy.uid,
            gid=config.policy.gid,
            executable=False,
        )
        _remaining_deadline(manager, deadline, stage="launch-tree reopen")
        # Reopen immediately before passing the host path to Docker; the container
        # independently reopens and hashes the mounted tree as well.
        reopen_launch_tree(launch, resolved.materialized_tree_root)
        argv = build_prebuild_argv(
            lease=lease,
            resolved=resolved,
            preflight=preflight,
            config=config,
            stage_path=stage_path,
            seccomp_path=seccomp_copy,
            compile_profile_path=compile_profile_copy,
        )
        remaining = _remaining_deadline(
            manager, deadline, stage="native prebuild container"
        )
        container_timeout = float(config.policy.timeout_seconds)
        if remaining is not None:
            container_timeout = min(container_timeout, remaining)
        execution = manager.run(
            lease, argv, timeout_s=container_timeout
        )
        _remaining_deadline(manager, deadline, stage="container completion")
        if execution.returncode != 0:
            diagnostic = execution.stderr_diagnostic
            detail = "" if diagnostic is None else f"; {diagnostic.summary}"
            raise OCIPrebuildError(
                f"native prebuild container exited {execution.returncode}{detail}"
            )
        reopen_launch_tree(launch, resolved.materialized_tree_root)
        stage_receipt, stage_discovery = _read_prebuild_receipt(
            stage_path, resolved=resolved
        )
        if stage_discovery != expected_discovery:
            raise OCIPrebuildError("discovery binding changed across container execution")
        _remaining_deadline(manager, deadline, stage="artifact publication")
        publication_work.mkdir(mode=0o700)
        published = publish_native_artifact(
            stage_path,
            config.publication_root,
            build_spec_digest=digest,
            work_root=publication_work,
            limits=limits,
        )
        publication = reopen_native_artifact(
            published.root,
            expected_build_spec_digest=digest,
            expected_publication_digest=published.publication_digest,
            limits=limits,
        )
        if published.reused:
            publication = replace(publication, reused=True)
        receipt, receipt_discovery = _read_prebuild_receipt(
            publication.root, resolved=resolved
        )
        if receipt != stage_receipt or receipt_discovery != expected_discovery:
            raise OCIPrebuildError(
                "published prebuild receipt differs from its container output"
            )
        discovery_digest = _reopen_discovery_publication(
            publication, receipt, receipt_discovery
        )
        _remaining_deadline(manager, deadline, stage="published artifact validation")
        argv_digest = hashlib.sha256(
            json.dumps(argv, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _remaining_deadline(manager, deadline, stage="prebuild result assembly")
        return OCIPrebuildResult(
            launch.digest,
            digest,
            publication,
            execution.elapsed_seconds,
            argv_digest,
            discovery_digest,
        )
    finally:
        manager.release(lease)


def _container_value(name: str, *, pattern: re.Pattern[str] | None = None) -> str:
    value = os.environ.get(name, "")
    if not value or "\x00" in value or (pattern is not None and pattern.fullmatch(value) is None):
        raise OCIPrebuildError(f"container prebuild environment {name} is invalid")
    return value


def _stock_sglang_site_root(
    *, expected_version: str, pinned_build_roots: tuple[str, ...]
) -> Path:
    """Locate image-owned SGLang as data without importing its package."""

    import importlib.metadata
    import importlib.util

    try:
        distribution = importlib.metadata.distribution("sglang")
    except importlib.metadata.PackageNotFoundError:
        raise OCIPrebuildError("image has no installed SGLang distribution") from None
    if distribution.version != expected_version:
        raise OCIPrebuildError(
            "image SGLang distribution version differs from discovery policy"
        )
    try:
        requested_package = Path(distribution.locate_file("sglang"))
        if not requested_package.exists():
            # Editable image installs keep distribution metadata in
            # site-packages while the package itself remains under a pinned
            # source root. Resolving the top-level spec does not import or run
            # SGLang; the package initializer remains unopened data.
            spec = importlib.util.find_spec("sglang")
            locations = (
                ()
                if spec is None
                else tuple(spec.submodule_search_locations or ())
            )
            if (
                spec is None
                or len(locations) != 1
                or spec.origin is None
                or Path(spec.origin) != Path(locations[0]) / "__init__.py"
            ):
                raise OCIPrebuildError(
                    "image SGLang editable package cannot be resolved exactly"
                )
            requested_package = Path(locations[0])
        initializer = requested_package / "__init__.py"
        if requested_package.is_symlink() or initializer.is_symlink():
            raise OCIPrebuildError(
                "image SGLang package and initializer must not be symlinks"
            )
        package = requested_package.resolve(strict=True)
        resolved_initializer = initializer.resolve(strict=True)
        site = package.parent
        if (
            package.name != "sglang"
            or not package.is_dir()
            or resolved_initializer != package / "__init__.py"
            or not resolved_initializer.is_file()
        ):
            raise OCIPrebuildError("image SGLang distribution has no package tree")
        roots = tuple(Path(value).resolve(strict=True) for value in pinned_build_roots)
    except OCIPrebuildError:
        raise
    except (OSError, RuntimeError) as exc:
        raise OCIPrebuildError(
            f"image SGLang distribution cannot be resolved: {exc}"
        ) from None
    if not any(package == root or root in package.parents for root in roots):
        raise OCIPrebuildError(
            "image SGLang package is outside validator-pinned build roots"
        )
    return site


def container_build() -> Path:
    """Fixed in-image worker entry; imports no candidate module and dlopens nothing."""
    build_digest = _container_value("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", pattern=re.compile(r"[0-9a-f]{64}"))
    tree_digest = _container_value("OPTIMA_ENGINE_TREE_DIGEST", pattern=re.compile(r"[0-9a-f]{64}"))
    architecture = _container_value("OPTIMA_TARGET_GPU_ARCH", pattern=_ARCH)
    stage_raw = _container_value("OPTIMA_NATIVE_ARTIFACT_STAGE")
    roots_raw = _container_value("OPTIMA_PINNED_BUILD_ROOTS")
    build_path_raw = _container_value("OPTIMA_BUILD_PATH")
    build_tmpdir = _container_value("OPTIMA_BUILD_TMPDIR")
    compile_timeout = _container_value(
        "OPTIMA_NATIVE_COMPILE_TIMEOUT_S", pattern=_SIZE
    )
    compile_profile_raw = os.environ.get(CUTE_COMPILE_PROFILE_ENV, "").strip()
    compile_profile_digest_raw = os.environ.get(
        CUTE_COMPILE_PROFILE_DIGEST_ENV, ""
    ).strip()
    if bool(compile_profile_raw) != bool(compile_profile_digest_raw):
        raise OCIPrebuildError("container CuTe compile-profile binding is incomplete")
    if compile_profile_raw and compile_profile_raw != CONTAINER_CUTE_COMPILE_PROFILE:
        raise OCIPrebuildError("container CuTe compile-profile path differs from policy")
    if os.environ.get("OPTIMA_REBUILD_CONTAINER") != "1":
        raise OCIPrebuildError("container build is not armed")
    if stage_raw != CONTAINER_STAGE:
        raise OCIPrebuildError("container stage path differs from policy")
    roots = tuple(roots_raw.split(":"))
    if not roots or roots != tuple(sorted(set(roots))):
        raise OCIPrebuildError("container pinned build roots are not canonical")
    for root in roots:
        _absolute_container_path(root, field="container pinned build root")
    build_path = tuple(build_path_raw.split(":"))
    if not build_path or len(set(build_path)) != len(build_path):
        raise OCIPrebuildError("container build PATH is not canonical")
    for root in build_path:
        _absolute_container_path(root, field="container build PATH entry")
    _absolute_container_path(build_tmpdir, field="container build tmpdir")

    preserved = {
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "/tmp/home",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "OPTIMA_BUILD_PATH": build_path_raw,
        "OPTIMA_BUILD_TMPDIR": build_tmpdir,
        "OPTIMA_BUNDLE_PATH": CONTAINER_TREE,
        "OPTIMA_ENGINE_TREE_DIGEST": tree_digest,
        "OPTIMA_NATIVE_ARTIFACT_STAGE": CONTAINER_STAGE,
        "OPTIMA_NATIVE_BUILD_SPEC_DIGEST": build_digest,
        "OPTIMA_NATIVE_COMPILE_TIMEOUT_S": compile_timeout,
        "OPTIMA_PINNED_BUILD_ROOTS": roots_raw,
        "OPTIMA_REBUILD_CONTAINER": "1",
        "OPTIMA_REBUILD_PHASE": "build",
        "OPTIMA_TARGET_GPU_ARCH": architecture,
        "PATH": build_path_raw,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": "/tmp",
    }
    if compile_profile_raw:
        preserved[CUTE_COMPILE_PROFILE_ENV] = compile_profile_raw
        preserved[CUTE_COMPILE_PROFILE_DIGEST_ENV] = compile_profile_digest_raw
    os.environ.clear()
    os.environ.update(preserved)
    from optima.engine_tree import reopen_materialized_engine_tree

    reopened = reopen_materialized_engine_tree(
        CONTAINER_TREE, expected_tree_digest=tree_digest
    )
    resolved_discovery = None
    if any(row.path == _DISCOVERY_ENGINE_METADATA for row in reopened.files):
        from optima.discovery import DiscoveryError, reopen_discovery_engine_binding

        try:
            resolved_discovery = reopen_discovery_engine_binding(reopened)
        except (DiscoveryError, OSError, TypeError, ValueError) as exc:
            raise OCIPrebuildError(
                f"container discovery binding cannot reopen: {exc}"
            ) from None
        if (
            resolved_discovery.build_profile.architecture != architecture
            or resolved_discovery.build_profile.sglang_version
            != resolved_discovery.policy.sglang_version
        ):
            raise OCIPrebuildError(
                "container discovery profile differs from build architecture or policy"
            )
    stage = Path(CONTAINER_STAGE)
    if stage.is_symlink() or not stage.is_dir():
        raise OCIPrebuildError("container native stage mount is unavailable")
    if any(stage.iterdir()):
        raise OCIPrebuildError("container native stage is not empty")

    from optima.manifest import all_declared_cuda_sources, load_manifest
    from optima.rebuild import apply_rebuild_plan, parse_rebuild_plan

    manifest = None
    requires_rebuild = False
    artifact_descriptors = ()
    if reopened.runtime_manifest is not None:
        manifest = load_manifest(CONTAINER_TREE)
        plan = parse_rebuild_plan(CONTAINER_TREE)
        artifact_descriptors = _artifact_provider_descriptors(manifest, plan)
        requires_rebuild = bool(
            manifest.dep_patches
            or all_declared_cuda_sources(CONTAINER_TREE, manifest)
            or artifact_descriptors
        )
    compile_profile_provider = _compile_profile_provider(artifact_descriptors)
    requires_compile_profile = compile_profile_provider is not None
    if requires_compile_profile != bool(compile_profile_raw):
        raise OCIPrebuildError(
            "artifact provider declaration does not match launch compile-profile authority"
        )
    compile_profile = None
    if requires_compile_profile:
        try:
            compile_profile = load_compile_profile(
                compile_profile_raw,
                expected_digest=compile_profile_digest_raw,
            )
        except CuteAOTError as exc:
            raise OCIPrebuildError(f"container CuTe compile profile is invalid: {exc}") from None
        if compile_profile.logical_architecture != architecture:
            raise OCIPrebuildError(
                "container CuTe compile profile differs from target architecture"
            )
        assert compile_profile_provider is not None
        if compile_profile.provider != compile_profile_provider.provider_id:
            raise OCIPrebuildError(
                "container CuTe compile profile names a different artifact provider"
            )
    rebuilt = apply_rebuild_plan(CONTAINER_TREE, phase="build")
    if requires_rebuild and not rebuilt:
        raise OCIPrebuildError("declared native inputs lack a reviewed rebuild plan")
    for descriptor in artifact_descriptors:
        assert compile_profile is not None
        _reopen_provider_publication(
            descriptor,
            stage,
            build_spec_digest=build_digest,
            tree_digest=tree_digest,
            logical_architecture=architecture,
            compile_profile_digest=compile_profile.digest,
            verify_distributions=True,
        )
    discovery_overlay_identity_digest = None
    if resolved_discovery is not None:
        from optima.discovery import DiscoveryError, build_discovery_overlay_stage

        stock_site = _stock_sglang_site_root(
            expected_version=resolved_discovery.policy.sglang_version,
            pinned_build_roots=roots,
        )
        try:
            identity = build_discovery_overlay_stage(
                resolved_discovery.discovery,
                stock_site_root=stock_site,
                native_stage_root=stage,
                policy=resolved_discovery.policy,
                build_profile=resolved_discovery.build_profile,
            )
        except (DiscoveryError, OSError, TypeError, ValueError) as exc:
            raise OCIPrebuildError(
                f"discovery overlay build failed: {exc}"
            ) from None
        discovery_overlay_identity_digest = identity.digest
    entries = sorted(child.name for child in stage.iterdir())
    if any(name not in _ALLOWED_STAGE_TOP_LEVEL - {PREBUILD_RECEIPT} for name in entries):
        raise OCIPrebuildError("reviewed patcher emitted an unexpected stage entry")
    receipt = {
        "build_spec_digest": build_digest,
        "rebuild_applied": rebuilt,
        "schema": PREBUILD_SCHEMA,
        "stage_entries": sorted((*entries, PREBUILD_RECEIPT)),
        "target_architecture": architecture,
        "tree_digest": tree_digest,
    }
    if discovery_overlay_identity_digest is not None:
        receipt[_DISCOVERY_RECEIPT_FIELD] = discovery_overlay_identity_digest
    if compile_profile is not None:
        receipt[_CUTE_AOT_RECEIPT_FIELD] = compile_profile.digest
    destination = stage / PREBUILD_RECEIPT
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(destination, flags, 0o444)
    try:
        raw = canonical_json_bytes(receipt) + b"\n"
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OCIPrebuildError("container prebuild receipt write stalled")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m optima.eval.oci_prebuild")
    parser.add_argument("--container-build", action="store_true")
    args = parser.parse_args(argv)
    if not args.container_build:
        parser.error("only the fixed --container-build worker entry is supported")
    container_build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTAINER_CUTE_COMPILE_PROFILE",
    "CONTAINER_STAGE",
    "CONTAINER_TREE",
    "OCIPrebuildConfig",
    "OCIPrebuildError",
    "OCIPrebuildPolicy",
    "OCIPrebuildResult",
    "PREBUILD_RECEIPT",
    "PREBUILD_SCHEMA",
    "build_prebuild_argv",
    "container_build",
    "run_oci_prebuild",
]
