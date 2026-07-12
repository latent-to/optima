"""Immutable publication of native build outputs.

This module is intentionally independent of the engine launcher and native
loader.  A trusted controller copies an untrusted build stage into a bounded,
validator-owned tree, seals a canonical inventory, and publishes the tree at a
path derived only from the complete native-build specification digest.  It
never imports or loads any published byte.
"""

from __future__ import annotations

import ctypes
import dataclasses
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


_SCHEMA = "optima.native-artifact-publication.v1"
_MANIFEST = ".optima-native-artifact.json"
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@=-]{0,254}$")
_STAGE_RE = re.compile(r"^\.stage-[0-9a-f]{16}-[0-9a-f]{32}$")
_FILE_KEYS = frozenset({"path", "sha256", "size"})
_MANIFEST_KEYS = frozenset(
    {"schema", "build_spec_digest", "publication_digest", "directories", "files"}
)
_STAT_STABILITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_CURRENT_OWNER = object()


class NativeArtifactError(RuntimeError):
    """A native artifact cannot be safely published or reopened."""


class NativeArtifactCollisionError(NativeArtifactError):
    """A build-spec address already names different canonical bytes."""


class NativeArtifactRaceError(NativeArtifactError):
    """A publication or source changed during a security-sensitive operation."""


@dataclass(frozen=True)
class NativeArtifactLimits:
    """Hard resource and path bounds for one native artifact tree."""

    max_files: int = 4_096
    max_directories: int = 4_096
    max_file_bytes: int = 1 << 30
    max_total_bytes: int = 8 << 30
    max_depth: int = 32
    max_path_bytes: int = 1_024
    max_manifest_bytes: int = 16 << 20

    def __post_init__(self) -> None:
        for name in (
            "max_files",
            "max_directories",
            "max_file_bytes",
            "max_total_bytes",
            "max_depth",
            "max_path_bytes",
            "max_manifest_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise NativeArtifactError(f"{name} must be a positive integer")
        if self.max_file_bytes > self.max_total_bytes:
            raise NativeArtifactError("max_file_bytes cannot exceed max_total_bytes")


@dataclass(frozen=True, order=True)
class NativeArtifactFile:
    """One immutable regular file in a native artifact inventory."""

    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.path, limits=NativeArtifactLimits())
        _require_digest(self.sha256, field="native artifact file sha256")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise NativeArtifactError("native artifact file size must be a nonnegative integer")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class NativeArtifactPublication:
    """A reopened, path-bound native artifact publication."""

    root: Path
    build_spec_digest: str
    publication_digest: str
    directories: tuple[str, ...]
    files: tuple[NativeArtifactFile, ...]
    reused: bool = False

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute():
            raise NativeArtifactError("native artifact publication root must be absolute")
        object.__setattr__(self, "root", root)
        _require_digest(self.build_spec_digest, field="build_spec_digest")
        _require_digest(self.publication_digest, field="publication_digest")
        if type(self.directories) is not tuple or type(self.files) is not tuple:
            raise NativeArtifactError("native artifact inventories must be tuples")
        if type(self.reused) is not bool:
            raise NativeArtifactError("native artifact reused flag must be boolean")
        if self.root.name != self.build_spec_digest or self.root.parent.name != self.build_spec_digest[:2]:
            raise NativeArtifactError("native artifact path is not derived from build_spec_digest")
        if self.directories != tuple(sorted(set(self.directories))):
            raise NativeArtifactError("native artifact directories are not canonical")
        if self.files != tuple(sorted(set(self.files), key=lambda row: row.path)):
            raise NativeArtifactError("native artifact files are not canonical")
        if len({row.path for row in self.files}) != len(self.files):
            raise NativeArtifactError("native artifact file paths are not unique")
        expected_directories = _required_directories(row.path for row in self.files)
        if self.directories != expected_directories:
            raise NativeArtifactError("native artifact contains empty or unmanifested directories")
        payload = _identity_payload(
            self.build_spec_digest, self.directories, self.files
        )
        if _publication_digest(payload) != self.publication_digest:
            raise NativeArtifactError("native artifact publication digest is inconsistent")

    @property
    def tree_digest(self) -> str:
        """Compatibility-neutral name for the exact published tree digest."""

        return self.publication_digest

    @property
    def path(self) -> Path:
        return self.root

    def identity_dict(self) -> dict[str, object]:
        """Return the path-free canonical publication identity."""

        return {
            "schema": _SCHEMA,
            "build_spec_digest": self.build_spec_digest,
            "publication_digest": self.publication_digest,
            "directories": list(self.directories),
            "files": [row.to_dict() for row in self.files],
        }


@dataclass
class _CopyState:
    limits: NativeArtifactLimits
    directories: list[str] = dataclasses.field(default_factory=list)
    files: list[NativeArtifactFile] = dataclasses.field(default_factory=list)
    total_bytes: int = 0


def _require_digest(value: object, *, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise NativeArtifactError(str(exc)) from None


def _validate_component(component: str) -> None:
    if not isinstance(component, str) or _COMPONENT_RE.fullmatch(component) is None:
        raise NativeArtifactError(f"native artifact path component is unsafe: {component!r}")
    try:
        component.encode("ascii", "strict")
    except UnicodeError:
        raise NativeArtifactError("native artifact paths must be canonical ASCII") from None


def _validate_relative_path(path: object, *, limits: NativeArtifactLimits) -> str:
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        raise NativeArtifactError(f"native artifact path is not canonical: {path!r}")
    parts = path.split("/")
    if len(parts) > limits.max_depth:
        raise NativeArtifactError(f"native artifact path exceeds depth bound: {path!r}")
    for component in parts:
        _validate_component(component)
    if len(path.encode("ascii")) > limits.max_path_bytes:
        raise NativeArtifactError(f"native artifact path exceeds byte bound: {path!r}")
    return path


def _relative(prefix: str, name: str, *, limits: NativeArtifactLimits) -> str:
    return _validate_relative_path(f"{prefix}/{name}" if prefix else name, limits=limits)


def _required_directories(paths: Any) -> tuple[str, ...]:
    directories: set[str] = set()
    for raw in paths:
        parts = raw.split("/")
        for index in range(1, len(parts)):
            directories.add("/".join(parts[:index]))
    return tuple(sorted(directories))


def _identity_payload(
    build_spec_digest: str,
    directories: tuple[str, ...],
    files: tuple[NativeArtifactFile, ...],
) -> dict[str, object]:
    return {
        "build_spec_digest": build_spec_digest,
        "directories": list(directories),
        "files": [row.to_dict() for row in files],
    }


def _publication_digest(payload: dict[str, object]) -> str:
    return canonical_digest("optima.native-artifact-publication", payload)


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in _STAT_STABILITY_FIELDS)


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_uid == right.st_uid
        and left.st_gid == right.st_gid
    )


def _is_sparse(info: os.stat_result) -> bool:
    blocks = getattr(info, "st_blocks", None)
    return info.st_size > 0 and blocks is not None and blocks * 512 < info.st_size


def _open_directory(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before = path.lstat()
        fd = os.open(path, flags)
    except OSError as exc:
        raise NativeArtifactError(f"cannot open directory without following links: {path}: {exc}") from None
    opened = os.fstat(fd)
    if not stat.S_ISDIR(before.st_mode) or not _same_stat(before, opened):
        os.close(fd)
        raise NativeArtifactRaceError(f"directory changed while opening: {path}")
    return fd, opened


def _open_child_directory(parent_fd: int, name: str, expected: os.stat_result) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise NativeArtifactError(f"cannot open native artifact directory {name!r}: {exc}") from None
    if not _same_stat(expected, os.fstat(fd)):
        os.close(fd)
        raise NativeArtifactRaceError(f"native artifact directory changed while opening: {name!r}")
    return fd


def _copy_regular(
    source_fd: int,
    destination_fd: int,
    name: str,
    relative: str,
    expected: os.stat_result,
    state: _CopyState,
) -> NativeArtifactFile:
    if expected.st_nlink != 1:
        raise NativeArtifactError(f"native artifact file is hardlinked: {relative}")
    if expected.st_size < 0 or expected.st_size > state.limits.max_file_bytes:
        raise NativeArtifactError(f"native artifact file exceeds its hard bound: {relative}")
    if _is_sparse(expected):
        raise NativeArtifactError(f"native artifact file is sparse: {relative}")
    if len(state.files) >= state.limits.max_files:
        raise NativeArtifactError("native artifact exceeds its file-count bound")
    if state.total_bytes + expected.st_size > state.limits.max_total_bytes:
        raise NativeArtifactError("native artifact exceeds its total-byte bound")

    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_file = destination_file = -1
    try:
        source_file = os.open(name, read_flags, dir_fd=source_fd)
        if not _same_stat(expected, os.fstat(source_file)):
            raise NativeArtifactRaceError(f"native artifact file changed while opening: {relative}")
        destination_file = os.open(name, write_flags, 0o600, dir_fd=destination_fd)
        digest = hashlib.sha256()
        remaining = expected.st_size
        while remaining:
            chunk = os.read(source_file, min(4 << 20, remaining))
            if not chunk:
                raise NativeArtifactRaceError(f"native artifact file was truncated: {relative}")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_file, view)
                if written <= 0:
                    raise NativeArtifactError(f"native artifact copy stalled: {relative}")
                view = view[written:]
            remaining -= len(chunk)
        if os.read(source_file, 1):
            raise NativeArtifactRaceError(f"native artifact file grew while copying: {relative}")
        if not _same_stat(expected, os.fstat(source_file)):
            raise NativeArtifactRaceError(f"native artifact file changed while copying: {relative}")
        copied = os.fstat(destination_file)
        if not stat.S_ISREG(copied.st_mode) or copied.st_nlink != 1 or copied.st_size != expected.st_size:
            raise NativeArtifactError(f"native artifact copy has an unsafe shape: {relative}")
        os.fchmod(destination_file, 0o444)
        os.fsync(destination_file)
        state.total_bytes += expected.st_size
        return NativeArtifactFile(relative, digest.hexdigest(), expected.st_size)
    except OSError as exc:
        raise NativeArtifactError(f"cannot copy native artifact file {relative}: {exc}") from None
    finally:
        if source_file >= 0:
            os.close(source_file)
        if destination_file >= 0:
            os.close(destination_file)


def _copy_directory(
    source_fd: int,
    destination_fd: int,
    *,
    prefix: str,
    state: _CopyState,
    freeze_destination: bool,
) -> None:
    before = os.fstat(source_fd)
    try:
        names = sorted(os.listdir(source_fd))
    except OSError as exc:
        raise NativeArtifactError(f"cannot enumerate native artifact stage: {exc}") from None
    if len(names) != len(set(names)):
        raise NativeArtifactError("native artifact directory has duplicate names")
    for name in names:
        _validate_component(name)
        relative = _relative(prefix, name, limits=state.limits)
        if not prefix and name == _MANIFEST:
            raise NativeArtifactError("native artifact stage contains the reserved host manifest")
        try:
            info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError as exc:
            raise NativeArtifactRaceError(f"native artifact entry vanished: {relative}: {exc}") from None
        if stat.S_ISLNK(info.st_mode):
            raise NativeArtifactError(f"native artifact contains a symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            if len(state.directories) >= state.limits.max_directories:
                raise NativeArtifactError("native artifact exceeds its directory-count bound")
            state.directories.append(relative)
            try:
                os.mkdir(name, 0o700, dir_fd=destination_fd)
                child_destination = _open_child_directory(
                    destination_fd,
                    name,
                    os.stat(name, dir_fd=destination_fd, follow_symlinks=False),
                )
            except OSError as exc:
                raise NativeArtifactError(f"cannot create native artifact directory {relative}: {exc}") from None
            child_source = _open_child_directory(source_fd, name, info)
            try:
                _copy_directory(
                    child_source,
                    child_destination,
                    prefix=relative,
                    state=state,
                    freeze_destination=True,
                )
            finally:
                os.close(child_source)
                os.close(child_destination)
        elif stat.S_ISREG(info.st_mode):
            state.files.append(
                _copy_regular(source_fd, destination_fd, name, relative, info, state)
            )
        else:
            raise NativeArtifactError(f"native artifact contains a non-regular object: {relative}")
    if not _same_stat(before, os.fstat(source_fd)):
        raise NativeArtifactRaceError(f"native artifact directory changed while copying: {prefix or '.'}")
    if freeze_destination:
        os.fchmod(destination_fd, 0o555)
    os.fsync(destination_fd)


def _copy_stage(
    source: Path,
    destination_fd: int,
    *,
    expected_source: os.stat_result,
    limits: NativeArtifactLimits,
) -> tuple[tuple[str, ...], tuple[NativeArtifactFile, ...]]:
    source_fd, source_info = _open_directory(source)
    if not _same_stat(expected_source, source_info):
        os.close(source_fd)
        raise NativeArtifactRaceError("native artifact stage root changed before copying")
    state = _CopyState(limits=limits)
    try:
        _copy_directory(
            source_fd,
            destination_fd,
            prefix="",
            state=state,
            freeze_destination=False,
        )
        if not _same_stat(source_info, os.fstat(source_fd)):
            raise NativeArtifactRaceError("native artifact stage root changed while copying")
        try:
            path_info = source.lstat()
        except OSError:
            path_info = None
        if path_info is None or not _same_object(path_info, source_info):
            raise NativeArtifactRaceError("native artifact stage root was replaced while copying")
    finally:
        os.close(source_fd)
    if not state.files:
        raise NativeArtifactError("native artifact stage must contain at least one file")
    files = tuple(sorted(state.files, key=lambda row: row.path))
    directories = tuple(sorted(state.directories))
    if directories != _required_directories(row.path for row in files):
        raise NativeArtifactError("native artifact stage contains an empty directory")
    return directories, files


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NativeArtifactError(f"native artifact manifest repeats key {key!r}")
        result[key] = value
    return result


def _write_manifest(
    stage_fd: int,
    *,
    build_spec_digest: str,
    directories: tuple[str, ...],
    files: tuple[NativeArtifactFile, ...],
    maximum_bytes: int,
) -> tuple[str, bytes]:
    payload = _identity_payload(build_spec_digest, directories, files)
    digest = _publication_digest(payload)
    manifest = {
        "schema": _SCHEMA,
        **payload,
        "publication_digest": digest,
    }
    raw = canonical_json_bytes(manifest) + b"\n"
    if len(raw) > maximum_bytes:
        raise NativeArtifactError("native artifact manifest exceeds its hard bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(_MANIFEST, flags, 0o600, dir_fd=stage_fd)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise NativeArtifactError("native artifact manifest write stalled")
            view = view[written:]
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    except OSError as exc:
        raise NativeArtifactError(f"cannot write native artifact manifest: {exc}") from None
    finally:
        if fd >= 0:
            os.close(fd)
    return digest, raw


def _read_regular_at(
    parent_fd: int,
    name: str,
    *,
    maximum: int,
    required_mode: int,
    expected_owner_uid: int | None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        fd = os.open(name, flags, dir_fd=parent_fd)
        before = os.fstat(fd)
        if not _same_stat(expected, before):
            raise NativeArtifactRaceError(f"native artifact file changed while opening: {name}")
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                expected_owner_uid is not None
                and before.st_uid != expected_owner_uid
            )
            or stat.S_IMODE(before.st_mode) != required_mode
            or before.st_size < 0
            or before.st_size > maximum
            or _is_sparse(before)
        ):
            raise NativeArtifactError(f"native artifact file has an unsafe shape: {name}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise NativeArtifactRaceError(f"native artifact file was truncated: {name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise NativeArtifactRaceError(f"native artifact file grew while reading: {name}")
        if not _same_stat(before, os.fstat(fd)):
            raise NativeArtifactRaceError(f"native artifact file changed while reading: {name}")
        return b"".join(chunks)
    except FileNotFoundError:
        raise NativeArtifactError(f"native artifact publication lacks {name}") from None
    except OSError as exc:
        raise NativeArtifactError(f"cannot read native artifact file {name}: {exc}") from None
    finally:
        if fd >= 0:
            os.close(fd)


def _hash_regular_at(
    parent_fd: int,
    name: str,
    relative: str,
    info: os.stat_result,
    limits: NativeArtifactLimits,
    expected_owner_uid: int | None,
) -> NativeArtifactFile:
    if (
        info.st_nlink != 1
        or (expected_owner_uid is not None and info.st_uid != expected_owner_uid)
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_size < 0
        or info.st_size > limits.max_file_bytes
        or _is_sparse(info)
    ):
        raise NativeArtifactError(f"published native artifact file has an unsafe shape: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
        if not _same_stat(info, os.fstat(fd)):
            raise NativeArtifactRaceError(f"published native artifact changed while opening: {relative}")
        digest = hashlib.sha256()
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(4 << 20, remaining))
            if not chunk:
                raise NativeArtifactRaceError(f"published native artifact was truncated: {relative}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1) or not _same_stat(info, os.fstat(fd)):
            raise NativeArtifactRaceError(f"published native artifact changed while hashing: {relative}")
        return NativeArtifactFile(relative, digest.hexdigest(), info.st_size)
    except OSError as exc:
        raise NativeArtifactError(f"cannot hash published native artifact {relative}: {exc}") from None
    finally:
        if fd >= 0:
            os.close(fd)


def _scan_published_directory(
    directory_fd: int,
    *,
    prefix: str,
    limits: NativeArtifactLimits,
    directories: list[str],
    files: list[NativeArtifactFile],
    totals: list[int],
    expected_owner_uid: int | None,
) -> None:
    before = os.fstat(directory_fd)
    required_mode = 0o555
    if not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode) != required_mode:
        raise NativeArtifactError(f"published native artifact directory is not read-only: {prefix or '.'}")
    if expected_owner_uid is not None and before.st_uid != expected_owner_uid:
        raise NativeArtifactError(f"published native artifact directory is not validator-owned: {prefix or '.'}")
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise NativeArtifactError(f"cannot enumerate published native artifact: {exc}") from None
    for name in names:
        if not prefix and name == _MANIFEST:
            continue
        _validate_component(name)
        relative = _relative(prefix, name, limits=limits)
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise NativeArtifactRaceError(f"published native artifact entry vanished: {relative}: {exc}") from None
        if stat.S_ISLNK(info.st_mode):
            raise NativeArtifactError(f"published native artifact contains a symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            if len(directories) >= limits.max_directories:
                raise NativeArtifactError("published native artifact exceeds its directory-count bound")
            directories.append(relative)
            child = _open_child_directory(directory_fd, name, info)
            try:
                _scan_published_directory(
                    child,
                    prefix=relative,
                    limits=limits,
                    directories=directories,
                    files=files,
                    totals=totals,
                    expected_owner_uid=expected_owner_uid,
                )
            finally:
                os.close(child)
        elif stat.S_ISREG(info.st_mode):
            if len(files) >= limits.max_files:
                raise NativeArtifactError("published native artifact exceeds its file-count bound")
            if totals[0] + info.st_size > limits.max_total_bytes:
                raise NativeArtifactError("published native artifact exceeds its total-byte bound")
            files.append(
                _hash_regular_at(
                    directory_fd,
                    name,
                    relative,
                    info,
                    limits,
                    expected_owner_uid,
                )
            )
            totals[0] += info.st_size
        else:
            raise NativeArtifactError(f"published native artifact contains a non-regular object: {relative}")
    if not _same_stat(before, os.fstat(directory_fd)):
        raise NativeArtifactRaceError(f"published native artifact directory changed while reading: {prefix or '.'}")


def _parse_manifest(raw: bytes, *, limits: NativeArtifactLimits) -> tuple[str, str, tuple[str, ...], tuple[NativeArtifactFile, ...]]:
    try:
        decoded = raw.decode("utf-8", "strict")
        manifest = json.loads(decoded, object_pairs_hook=_strict_json_object)
    except NativeArtifactError:
        raise
    except (UnicodeError, ValueError, TypeError) as exc:
        raise NativeArtifactError(f"native artifact manifest is malformed: {exc}") from None
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise NativeArtifactError("native artifact manifest schema is not closed")
    if manifest.get("schema") != _SCHEMA:
        raise NativeArtifactError("native artifact manifest schema mismatch")
    build_digest = _require_digest(manifest.get("build_spec_digest"), field="build_spec_digest")
    publication_digest = _require_digest(manifest.get("publication_digest"), field="publication_digest")
    raw_directories = manifest.get("directories")
    raw_files = manifest.get("files")
    if type(raw_directories) is not list or type(raw_files) is not list:
        raise NativeArtifactError("native artifact manifest inventories must be arrays")
    directories: list[str] = []
    for path in raw_directories:
        directories.append(_validate_relative_path(path, limits=limits))
    files: list[NativeArtifactFile] = []
    for row in raw_files:
        if not isinstance(row, dict) or set(row) != _FILE_KEYS:
            raise NativeArtifactError("native artifact manifest file row is malformed")
        path = _validate_relative_path(row.get("path"), limits=limits)
        digest = _require_digest(row.get("sha256"), field="native artifact file sha256")
        size = row.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise NativeArtifactError("native artifact manifest file size is malformed")
        if size > limits.max_file_bytes:
            raise NativeArtifactError("native artifact manifest file exceeds its hard bound")
        files.append(NativeArtifactFile(path, digest, size))
    directory_tuple = tuple(directories)
    file_tuple = tuple(files)
    if directory_tuple != tuple(sorted(set(directory_tuple))):
        raise NativeArtifactError("native artifact manifest directories are not canonical")
    if file_tuple != tuple(sorted(set(file_tuple), key=lambda row: row.path)):
        raise NativeArtifactError("native artifact manifest files are not canonical")
    if len({row.path for row in file_tuple}) != len(file_tuple):
        raise NativeArtifactError("native artifact manifest file paths are not unique")
    if sum(row.size for row in file_tuple) > limits.max_total_bytes:
        raise NativeArtifactError("native artifact manifest exceeds its total-byte bound")
    if len(file_tuple) > limits.max_files or len(directory_tuple) > limits.max_directories:
        raise NativeArtifactError("native artifact manifest exceeds its count bounds")
    if not file_tuple or directory_tuple != _required_directories(row.path for row in file_tuple):
        raise NativeArtifactError("native artifact manifest has empty/unmanifested directories")
    payload = _identity_payload(build_digest, directory_tuple, file_tuple)
    if _publication_digest(payload) != publication_digest:
        raise NativeArtifactError("native artifact manifest publication digest mismatch")
    if raw != canonical_json_bytes(manifest) + b"\n":
        raise NativeArtifactError("native artifact manifest is not canonical JSON")
    return build_digest, publication_digest, directory_tuple, file_tuple


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        common = Path(os.path.commonpath((left, right)))
    except ValueError:
        return False
    return common == left or common == right


def _canonical_existing_directory(
    path: Path,
    *,
    name: str,
    writable: bool,
    expected_owner_uid: int | None | object = _CURRENT_OWNER,
) -> tuple[Path, int, os.stat_result]:
    try:
        canonical = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise NativeArtifactError(f"{name} is unavailable: {exc}") from None
    fd, info = _open_directory(canonical)
    mode = stat.S_IMODE(info.st_mode)
    if expected_owner_uid is _CURRENT_OWNER:
        expected_owner_uid = os.geteuid() if hasattr(os, "geteuid") else None
    if expected_owner_uid is not None and info.st_uid != expected_owner_uid:
        os.close(fd)
        raise NativeArtifactError(f"{name} must be owned by the validator user")
    if mode & 0o022:
        os.close(fd)
        raise NativeArtifactError(f"{name} must not be group/world writable")
    if writable and not mode & 0o200:
        os.close(fd)
        raise NativeArtifactError(f"{name} must be writable by the validator owner")
    return canonical, fd, info


def _prepare_publication_root(path: Path, *, source: Path) -> tuple[Path, int, os.stat_result]:
    requested = path.expanduser()
    prospective = requested.resolve(strict=False)
    if _paths_overlap(source, prospective):
        raise NativeArtifactError("staging_root and publication_root must not overlap")
    try:
        requested.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise NativeArtifactError(f"cannot create publication_root: {exc}") from None
    try:
        if stat.S_ISLNK(requested.lstat().st_mode):
            raise NativeArtifactError("publication_root must not be a symlink")
    except OSError as exc:
        raise NativeArtifactError(f"cannot inspect publication_root: {exc}") from None
    canonical, fd, info = _canonical_existing_directory(requested, name="publication_root", writable=True)
    if _paths_overlap(source, canonical):
        os.close(fd)
        raise NativeArtifactError("staging_root and publication_root must not overlap")
    return canonical, fd, info


def _ensure_shard(root_fd: int, root: Path, shard: str) -> tuple[Path, int, os.stat_result]:
    _validate_component(shard)
    try:
        os.mkdir(shard, 0o700, dir_fd=root_fd)
        os.fsync(root_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise NativeArtifactError(f"cannot create native artifact shard: {exc}") from None
    try:
        info = os.stat(shard, dir_fd=root_fd, follow_symlinks=False)
        fd = _open_child_directory(root_fd, shard, info)
    except OSError as exc:
        raise NativeArtifactError(f"cannot open native artifact shard: {exc}") from None
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISDIR(info.st_mode)
        or mode & 0o022
        or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
    ):
        os.close(fd)
        raise NativeArtifactError("native artifact shard is not validator-controlled")
    return root / shard, fd, info


def _create_stage(parent_fd: int, build_spec_digest: str) -> tuple[str, int]:
    name = f".stage-{build_spec_digest[:16]}-{secrets.token_hex(16)}"
    if _STAGE_RE.fullmatch(name) is None:
        raise NativeArtifactError("native artifact stage name generation failed")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        fd = _open_child_directory(parent_fd, name, info)
    except FileExistsError:
        raise NativeArtifactRaceError("native artifact stage name already exists") from None
    except OSError as exc:
        raise NativeArtifactError(f"cannot create native artifact stage: {exc}") from None
    return name, fd


def _remove_private_tree(path: Path) -> None:
    if not os.path.lexists(path):
        return
    try:
        for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            try:
                info = child.lstat()
                if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    os.chmod(child, 0o700, follow_symlinks=False)
                else:
                    os.chmod(child, 0o600, follow_symlinks=False)
            except OSError:
                pass
        os.chmod(path, 0o700, follow_symlinks=False)
        import shutil

        shutil.rmtree(path)
    except OSError:
        pass


def _rename_noreplace(
    source_directory_fd: int,
    source: str,
    destination_directory_fd: int,
    destination: str,
) -> None:
    """Atomically rename across same-filesystem directories without replacement."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_b = os.fsencode(source)
    destination_b = os.fsencode(destination)
    if hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(
            source_directory_fd,
            source_b,
            destination_directory_fd,
            destination_b,
            1,
        )
    elif hasattr(libc, "renameatx_np"):
        function = libc.renameatx_np
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(
            source_directory_fd,
            source_b,
            destination_directory_fd,
            destination_b,
            0x00000004,
        )
    else:
        raise NativeArtifactError("platform lacks an atomic no-replace directory rename")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise NativeArtifactRaceError("native artifact publication appeared during publish")
    raise NativeArtifactError(f"atomic native artifact publication failed: {os.strerror(error)}")


def _entry_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise NativeArtifactError(f"cannot inspect native artifact destination: {exc}") from None


def _assert_bound_directory(path: Path, fd: int, *, name: str) -> None:
    try:
        path_info = path.lstat()
    except OSError:
        path_info = None
    if path_info is None or not _same_object(path_info, os.fstat(fd)):
        raise NativeArtifactRaceError(f"{name} changed during native artifact publication")


def publish_native_artifact(
    staging_root: str | os.PathLike[str],
    publication_root: str | os.PathLike[str],
    *,
    build_spec_digest: str,
    work_root: str | os.PathLike[str] | None = None,
    limits: NativeArtifactLimits | None = None,
) -> NativeArtifactPublication:
    """Copy, seal, and atomically publish one native build output tree.

    ``staging_root`` is treated as hostile and may not overlap the
    validator-owned ``publication_root``.  The final path is always
    ``publication_root/<digest[:2]>/<digest>``.  Production callers provide a
    lease-owned ``work_root`` on the same filesystem so a controller kill during
    the bounded copy is restart-recoverable; the internal staging fallback exists
    for the standalone verifier and unit tests only.
    """

    digest = _require_digest(build_spec_digest, field="build_spec_digest")
    if work_root is not None and sys.platform != "linux":
        raise NativeArtifactError(
            "lease-owned work_root publication requires Linux renameat2"
        )
    if limits is None:
        limits = NativeArtifactLimits()
    if type(limits) is not NativeArtifactLimits:
        raise NativeArtifactError("limits must be NativeArtifactLimits")
    try:
        requested_source = Path(staging_root).expanduser()
        if stat.S_ISLNK(requested_source.lstat().st_mode):
            raise NativeArtifactError("staging_root must not be a symlink")
        source = requested_source.resolve(strict=True)
    except OSError as exc:
        raise NativeArtifactError(f"staging_root is unavailable: {exc}") from None
    source_info = source.lstat()
    if not stat.S_ISDIR(source_info.st_mode) or stat.S_ISLNK(source_info.st_mode):
        raise NativeArtifactError("staging_root must be a real directory")

    root, root_fd, root_info = _prepare_publication_root(Path(publication_root), source=source)
    shard_path: Path | None = None
    shard_fd = stage_fd = work_fd = -1
    work_path: Path | None = None
    work_info: os.stat_result | None = None
    stage_name = ""
    stage_path: Path | None = None
    renamed = False
    try:
        shard_path, shard_fd, shard_info = _ensure_shard(root_fd, root, digest[:2])
        if work_root is None:
            work_path, work_fd, work_info = shard_path, shard_fd, shard_info
        else:
            requested_work = Path(work_root).expanduser()
            if requested_work.is_symlink():
                raise NativeArtifactError("work_root must not be a symlink")
            work_path, work_fd, work_info = _canonical_existing_directory(
                requested_work, name="work_root", writable=True
            )
            if _paths_overlap(source, work_path) or _paths_overlap(root, work_path):
                raise NativeArtifactError(
                    "work_root must not overlap staging_root or publication_root"
                )
            if work_info.st_dev != shard_info.st_dev:
                raise NativeArtifactError(
                    "work_root and publication_root must share a filesystem"
                )
            if os.listdir(work_fd):
                raise NativeArtifactError("work_root must be empty before publication")
        existed_before = _entry_exists(shard_fd, digest)
        existing_info = (
            os.stat(digest, dir_fd=shard_fd, follow_symlinks=False)
            if existed_before
            else None
        )
        stage_name, stage_fd = _create_stage(work_fd, digest)
        stage_path = work_path / stage_name
        directories, files = _copy_stage(
            source,
            stage_fd,
            expected_source=source_info,
            limits=limits,
        )
        publication_digest, _ = _write_manifest(
            stage_fd,
            build_spec_digest=digest,
            directories=directories,
            files=files,
            maximum_bytes=limits.max_manifest_bytes,
        )
        os.fchmod(stage_fd, 0o555)
        os.fsync(stage_fd)
        os.close(stage_fd)
        stage_fd = -1

        destination = shard_path / digest
        if existed_before:
            try:
                current_existing = os.stat(
                    digest, dir_fd=shard_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise NativeArtifactRaceError(
                    f"existing native artifact changed during staging: {exc}"
                ) from None
            assert existing_info is not None
            if not _same_stat(existing_info, current_existing):
                raise NativeArtifactRaceError(
                    "existing native artifact changed during staging"
                )
            existing = reopen_native_artifact(
                destination,
                expected_build_spec_digest=digest,
                expected_publication_digest=publication_digest,
                limits=limits,
            )
            return dataclasses.replace(existing, reused=True)
        if _entry_exists(shard_fd, digest):
            raise NativeArtifactRaceError("native artifact publication appeared during staging")
        _rename_noreplace(work_fd, stage_name, shard_fd, digest)
        renamed = True
        os.fsync(work_fd)
        os.fsync(shard_fd)
        _assert_bound_directory(shard_path, shard_fd, name="native artifact shard")
        reopened = reopen_native_artifact(
            destination,
            expected_build_spec_digest=digest,
            expected_publication_digest=publication_digest,
            limits=limits,
        )
        if reopened.directories != directories or reopened.files != files:
            raise NativeArtifactRaceError("native artifact changed across atomic publication")
        return reopened
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)
        if stage_path is not None and not renamed:
            _remove_private_tree(stage_path)
        if work_fd >= 0 and work_fd != shard_fd:
            try:
                assert work_path is not None
                _assert_bound_directory(work_path, work_fd, name="native artifact work root")
            finally:
                os.close(work_fd)
        if shard_fd >= 0:
            os.close(shard_fd)
        try:
            root_path_info = root.lstat()
        except OSError:
            root_path_info = None
        if root_path_info is None or not _same_object(root_path_info, os.fstat(root_fd)):
            # Cleanup has already been attempted.  A caller must treat the root
            # replacement as terminal even if another exception is in flight.
            os.close(root_fd)
            raise NativeArtifactRaceError("publication_root changed during publication")
        os.close(root_fd)


def reopen_native_artifact(
    path: str | os.PathLike[str],
    *,
    expected_build_spec_digest: str,
    expected_publication_digest: str | None = None,
    expected_owner_uid: int | None | object = _CURRENT_OWNER,
    limits: NativeArtifactLimits | None = None,
) -> NativeArtifactPublication:
    """Reopen and fully rederive an immutable native artifact publication.

    Host callers retain the default and require validator ownership.  A non-root
    engine container may pass ``expected_owner_uid=None`` for a root-owned read-only
    bind; PR2b separately proves that mount read-only before candidate entry.  This
    never relaxes canonical paths, 0444/0555 modes, link shape, inventory, hashes, or
    the two externally bound digests.
    """

    build_digest = _require_digest(expected_build_spec_digest, field="expected_build_spec_digest")
    if expected_publication_digest is not None:
        expected_publication_digest = _require_digest(
            expected_publication_digest, field="expected_publication_digest"
        )
    if limits is None:
        limits = NativeArtifactLimits()
    if type(limits) is not NativeArtifactLimits:
        raise NativeArtifactError("limits must be NativeArtifactLimits")
    if expected_owner_uid is _CURRENT_OWNER:
        expected_owner_uid = os.geteuid() if hasattr(os, "geteuid") else None
    elif expected_owner_uid is not None and (
        type(expected_owner_uid) is not int or expected_owner_uid < 0
    ):
        raise NativeArtifactError("expected_owner_uid must be a nonnegative integer or None")
    requested = Path(path).expanduser()
    if requested.name != build_digest or requested.parent.name != build_digest[:2]:
        raise NativeArtifactError("native artifact path is not derived from expected build digest")
    try:
        if stat.S_ISLNK(requested.lstat().st_mode):
            raise NativeArtifactError("native artifact publication must not be a symlink")
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise NativeArtifactError(f"native artifact publication is unavailable: {exc}") from None
    if root.name != build_digest or root.parent.name != build_digest[:2]:
        raise NativeArtifactError("resolved native artifact path has the wrong build address")
    for parent, label in (
        (root.parent, "native artifact shard"),
        (root.parent.parent, "native artifact publication root"),
    ):
        _, parent_fd, _ = _canonical_existing_directory(
            parent,
            name=label,
            writable=False,
            expected_owner_uid=expected_owner_uid,
        )
        os.close(parent_fd)
    directory_fd, directory_info = _open_directory(root)
    try:
        if (
            stat.S_IMODE(directory_info.st_mode) != 0o555
            or (
                expected_owner_uid is not None
                and directory_info.st_uid != expected_owner_uid
            )
        ):
            raise NativeArtifactError("native artifact publication root must have mode 0555")
        raw = _read_regular_at(
            directory_fd,
            _MANIFEST,
            maximum=limits.max_manifest_bytes,
            required_mode=0o444,
            expected_owner_uid=expected_owner_uid,
        )
        manifest_build, publication_digest, directories, files = _parse_manifest(raw, limits=limits)
        if manifest_build != build_digest:
            raise NativeArtifactCollisionError("native artifact manifest names another build spec")
        if expected_publication_digest is not None and publication_digest != expected_publication_digest:
            raise NativeArtifactCollisionError("native artifact publication digest differs from expected bytes")
        observed_directories: list[str] = []
        observed_files: list[NativeArtifactFile] = []
        _scan_published_directory(
            directory_fd,
            prefix="",
            limits=limits,
            directories=observed_directories,
            files=observed_files,
            totals=[0],
            expected_owner_uid=expected_owner_uid,
        )
        if (
            tuple(sorted(observed_directories)) != directories
            or tuple(sorted(observed_files, key=lambda row: row.path)) != files
        ):
            raise NativeArtifactError("native artifact publication inventory differs from its manifest")
        if not _same_stat(directory_info, os.fstat(directory_fd)):
            raise NativeArtifactRaceError("native artifact publication changed while reopening")
        return NativeArtifactPublication(
            root=root,
            build_spec_digest=build_digest,
            publication_digest=publication_digest,
            directories=directories,
            files=files,
        )
    finally:
        os.close(directory_fd)


__all__ = [
    "NativeArtifactCollisionError",
    "NativeArtifactError",
    "NativeArtifactFile",
    "NativeArtifactLimits",
    "NativeArtifactPublication",
    "NativeArtifactRaceError",
    "publish_native_artifact",
    "reopen_native_artifact",
]
