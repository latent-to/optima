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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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
from optima.eval.oci_process import OCILease, OCIProcessManager
from optima.eval.runtime_preflight import RuntimePreflightReceipt
from optima.stack_identity import canonical_digest, canonical_json_bytes, require_sha256_hex


CONTAINER_TREE = "/optima/engine-tree"
CONTAINER_STAGE = "/optima/native-stage"
PREBUILD_RECEIPT = "prebuild.json"
PREBUILD_SCHEMA = "optima.oci-native-prebuild.v1"
_PUBLICATION_MANIFEST = ".optima-native-artifact.json"
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_ARCH = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SIZE = re.compile(r"[1-9][0-9]{0,15}\Z")
_ALLOWED_STAGE_TOP_LEVEL = frozenset(
    {"cuda", "dep_modules", "dep_overlays", PREBUILD_RECEIPT}
)


class OCIPrebuildError(RuntimeError):
    """Trusted prebuild configuration, execution, or evidence is invalid."""


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

    def resource_payload(self) -> dict[str, object]:
        return {
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


def build_prebuild_argv(
    *,
    lease: OCILease,
    resolved: ResolvedEngineLaunch,
    preflight: RuntimePreflightReceipt,
    config: OCIPrebuildConfig,
    stage_path: Path,
    seccomp_path: Path,
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


def _read_prebuild_receipt(root: Path, *, resolved: ResolvedEngineLaunch) -> dict[str, object]:
    path = root / PREBUILD_RECEIPT
    try:
        raw = path.read_bytes()
        row = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise OCIPrebuildError(f"native prebuild receipt is unreadable: {exc}") from None
    keys = {
        "schema",
        "build_spec_digest",
        "rebuild_applied",
        "stage_entries",
        "target_architecture",
        "tree_digest",
    }
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
    observed = {child.name for child in root.iterdir()}
    observed.discard(_PUBLICATION_MANIFEST)
    if observed != set(entries):
        raise OCIPrebuildError("native prebuild stage differs from its receipt")
    if type(row.get("rebuild_applied")) is not bool:
        raise OCIPrebuildError("native prebuild receipt rebuild flag is invalid")
    return row


def run_oci_prebuild(
    launch: EngineLaunchSpec,
    binding: TrustedLaunchBinding,
    config: OCIPrebuildConfig,
    *,
    manager: OCIProcessManager | None = None,
    limits: NativeArtifactLimits | None = None,
) -> OCIPrebuildResult:
    """Build, seal, publish, and reopen one native artifact tree."""
    if not isinstance(config, OCIPrebuildConfig):
        raise OCIPrebuildError("config must be OCIPrebuildConfig")
    resolved, preflight = _validate_binding(launch, binding, config)
    digest = resolved.native_build_spec.digest
    existing = config.publication_root / digest[:2] / digest
    if existing.exists() or existing.is_symlink():
        publication = reopen_native_artifact(
            existing, expected_build_spec_digest=digest, limits=limits
        )
        _read_prebuild_receipt(publication.root, resolved=resolved)
        return OCIPrebuildResult(launch.digest, digest, publication, None, None)

    if manager is None:
        manager = OCIProcessManager(
            docker_binary=config.docker_binary,
            recovery_root=config.recovery_root,
            executor_id=config.executor_id,
        )
    if manager.docker_binary != config.docker_binary or manager.executor_id != config.executor_id:
        raise OCIPrebuildError("OCI process manager does not match prebuild config")
    _prepare_publication_root_for_lease(
        config.publication_root, recovery_resources_root=manager.resources_root
    )
    lease_id = "prebuild-" + secrets.token_hex(10)
    lease = manager.register(
        lease_id=lease_id,
        container_name="optima-" + lease_id,
        mount_relpaths=("native-stage",),
        stage_relpaths=("seccomp.json", "publication-work"),
    )
    stage_path = lease.mount_paths[0]
    seccomp_copy = lease.stage_paths[0]
    publication_work = lease.stage_paths[1]
    try:
        _copy_seccomp(
            config.seccomp_profile,
            seccomp_copy,
            expected_digest=launch.seccomp_policy_digest,
        )
        manager.mount_tmpfs(
            lease,
            stage_path,
            size_bytes=config.policy.stage_bytes,
            inode_limit=config.policy.stage_inodes,
            uid=config.policy.uid,
            gid=config.policy.gid,
            executable=False,
        )
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
        )
        execution = manager.run(
            lease, argv, timeout_s=float(config.policy.timeout_seconds)
        )
        if execution.returncode != 0:
            raise OCIPrebuildError(
                f"native prebuild container exited {execution.returncode}"
            )
        reopen_launch_tree(launch, resolved.materialized_tree_root)
        _read_prebuild_receipt(stage_path, resolved=resolved)
        publication_work.mkdir(mode=0o700)
        publication = publish_native_artifact(
            stage_path,
            config.publication_root,
            build_spec_digest=digest,
            work_root=publication_work,
            limits=limits,
        )
        _read_prebuild_receipt(publication.root, resolved=resolved)
        argv_digest = hashlib.sha256(
            json.dumps(argv, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return OCIPrebuildResult(
            launch.digest,
            digest,
            publication,
            execution.elapsed_seconds,
            argv_digest,
        )
    finally:
        manager.release(lease)


def _container_value(name: str, *, pattern: re.Pattern[str] | None = None) -> str:
    value = os.environ.get(name, "")
    if not value or "\x00" in value or (pattern is not None and pattern.fullmatch(value) is None):
        raise OCIPrebuildError(f"container prebuild environment {name} is invalid")
    return value


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
    os.environ.clear()
    os.environ.update(preserved)
    from optima.engine_tree import reopen_materialized_engine_tree

    reopened = reopen_materialized_engine_tree(
        CONTAINER_TREE, expected_tree_digest=tree_digest
    )
    stage = Path(CONTAINER_STAGE)
    if stage.is_symlink() or not stage.is_dir():
        raise OCIPrebuildError("container native stage mount is unavailable")
    if any(stage.iterdir()):
        raise OCIPrebuildError("container native stage is not empty")

    from optima.manifest import all_declared_cuda_sources, load_manifest
    from optima.rebuild import apply_rebuild_plan

    requires_rebuild = False
    if reopened.runtime_manifest is not None:
        manifest = load_manifest(CONTAINER_TREE)
        requires_rebuild = bool(
            manifest.dep_patches or all_declared_cuda_sources(CONTAINER_TREE, manifest)
        )
    rebuilt = apply_rebuild_plan(CONTAINER_TREE, phase="build")
    if requires_rebuild and not rebuilt:
        raise OCIPrebuildError("declared native inputs lack a reviewed rebuild plan")
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
