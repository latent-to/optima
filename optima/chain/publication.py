"""Immutable worker publication for one finalized submitted bundle.

The fetched tree is private validator intake.  It is never mounted into a worker
directly.  This module first proves that every source byte participates in the
committed bundle identity, then delegates the race-safe copy, 0444/0555 sealing,
no-replace rename, fsync, and complete reopen to the common native-artifact
publisher.  The resulting carrier is wrapped in a bundle-specific typed receipt;
none of its bytes are imported here.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from optima.eval.native_artifact import (
    NativeArtifactError,
    NativeArtifactFile,
    NativeArtifactLimits,
    NativeArtifactPublication,
    publish_native_artifact,
    reopen_native_artifact,
)
from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    require_sha256_hex,
)


_SCHEMA = "optima.worker-bundle-publication.v1"
_EXCLUDED_DIRECTORIES = frozenset({".git", "__pycache__"})
_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})
_CURRENT_OWNER = object()


class WorkerBundlePublicationError(RuntimeError):
    """A submitted tree cannot be safely published or reopened."""


class WorkerBundleSourceError(WorkerBundlePublicationError):
    """Submitted paths cannot be represented by the immutable worker carrier."""


@dataclass(frozen=True)
class WorkerBundlePublication:
    """Path-bound immutable publication of exactly one committed bundle tree."""

    root: Path
    content_hash: str
    address_digest: str
    publication_digest: str
    directories: tuple[str, ...]
    files: tuple[NativeArtifactFile, ...]
    reused: bool = False

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute():
            raise WorkerBundlePublicationError("worker publication root must be absolute")
        object.__setattr__(self, "root", root)
        for name in ("content_hash", "address_digest", "publication_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), field=name))
        if root.name != self.address_digest or root.parent.name != self.address_digest[:2]:
            raise WorkerBundlePublicationError(
                "worker publication path is not derived from its address"
            )
        if type(self.directories) is not tuple or self.directories != tuple(
            sorted(set(self.directories))
        ):
            raise WorkerBundlePublicationError(
                "worker publication directory inventory is not canonical"
            )
        if type(self.files) is not tuple or self.files != tuple(
            sorted(set(self.files), key=lambda row: row.path)
        ):
            raise WorkerBundlePublicationError(
                "worker publication file inventory is not canonical"
            )
        if (
            not self.files
            or any(type(row) is not NativeArtifactFile for row in self.files)
            or len({row.path for row in self.files}) != len(self.files)
        ):
            raise WorkerBundlePublicationError(
                "worker publication file inventory is empty or ambiguous"
            )
        required_directories = {
            parent.as_posix()
            for row in self.files
            for parent in tuple(PurePosixPath(row.path).parents)[:-1]
        }
        if set(self.directories) != required_directories:
            raise WorkerBundlePublicationError(
                "worker publication contains empty or unmanifested directories"
            )
        if type(self.reused) is not bool:
            raise WorkerBundlePublicationError("worker publication reused flag is not boolean")
        expected_address = _address(self.content_hash, self.directories, self.files)
        if self.address_digest != expected_address:
            raise WorkerBundlePublicationError(
                "worker publication address differs from its exact inventory"
            )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.worker-bundle-publication",
            {
                "address_digest": self.address_digest,
                "content_hash": self.content_hash,
                "directories": list(self.directories),
                "files": [row.to_dict() for row in self.files],
                "publication_digest": self.publication_digest,
                "schema": _SCHEMA,
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "address_digest": self.address_digest,
            "content_hash": self.content_hash,
            "directories": list(self.directories),
            "files": [row.to_dict() for row in self.files],
            "publication_digest": self.publication_digest,
            "schema": _SCHEMA,
        }


def _digest(value: object, *, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise WorkerBundlePublicationError(str(exc)) from None
    if result == "0" * 64:
        raise WorkerBundlePublicationError(f"{field} must not be the all-zero digest")
    return result


def _excluded(relative: PurePosixPath) -> bool:
    return (
        any(part in _EXCLUDED_DIRECTORIES for part in relative.parts)
        or relative.suffix in _EXCLUDED_SUFFIXES
        or relative.name.startswith("._")
    )


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, name) == getattr(right, name)
        for name in (
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
    )


def _private_root(value: str | os.PathLike[str]) -> Path:
    requested = Path(value).expanduser()
    try:
        before = requested.lstat()
        root = requested.resolve(strict=True)
        after = root.lstat()
    except OSError as exc:
        raise WorkerBundlePublicationError(
            f"private bundle root is unavailable: {exc}"
        ) from None
    owner = os.geteuid() if hasattr(os, "geteuid") else after.st_uid
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or not _same_stat(before, after)
        or after.st_uid != owner
        or stat.S_IMODE(after.st_mode) & 0o077
        or stat.S_IMODE(after.st_mode) & 0o700 != 0o700
    ):
        raise WorkerBundlePublicationError(
            "private bundle root must be a validator-owned mode-0700-style directory"
        )
    return root


def _stable_file(path: Path, *, owner_uid: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != owner_uid
            or stat.S_IMODE(before.st_mode) & 0o077
            or not stat.S_IMODE(before.st_mode) & 0o400
        ):
            raise WorkerBundlePublicationError(
                f"private bundle file has unsafe ownership/mode/link shape: {path}"
            )
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if not _same_stat(before, opened):
            raise WorkerBundlePublicationError(
                f"private bundle file changed while opening: {path}"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(4 << 20, remaining))
            if not chunk:
                raise WorkerBundlePublicationError(
                    f"private bundle file was truncated: {path}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1) or not _same_stat(opened, os.fstat(fd)):
            raise WorkerBundlePublicationError(
                f"private bundle file changed while reading: {path}"
            )
        return b"".join(chunks), opened
    except OSError as exc:
        raise WorkerBundlePublicationError(
            f"cannot read private bundle file {path}: {exc}"
        ) from None
    finally:
        if fd >= 0:
            os.close(fd)


def _source_inventory(
    root: Path,
) -> tuple[tuple[str, ...], tuple[NativeArtifactFile, ...], str]:
    owner = os.geteuid() if hasattr(os, "geteuid") else root.stat().st_uid
    directories: list[str] = []
    observed_files: list[tuple[str, Path, NativeArtifactFile]] = []
    for current, names, leaves in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        names[:] = sorted(names)
        for name in names:
            child = current_path / name
            relative = PurePosixPath(child.relative_to(root).as_posix())
            try:
                info = child.lstat()
            except OSError as exc:
                raise WorkerBundlePublicationError(
                    f"cannot inspect private bundle directory {relative}: {exc}"
                ) from None
            if _excluded(relative):
                raise WorkerBundlePublicationError(
                    f"bundle contains bytes excluded from committed identity: {relative}"
                )
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_uid != owner
                or stat.S_IMODE(info.st_mode) & 0o077
                or stat.S_IMODE(info.st_mode) & 0o500 != 0o500
            ):
                raise WorkerBundlePublicationError(
                    f"private bundle directory has unsafe ownership/mode: {relative}"
                )
            directories.append(relative.as_posix())
        for name in sorted(leaves):
            path = current_path / name
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if _excluded(relative):
                raise WorkerBundlePublicationError(
                    f"bundle contains bytes excluded from committed identity: {relative}"
                )
            raw, _ = _stable_file(path, owner_uid=owner)
            logical = relative.as_posix()
            try:
                row = NativeArtifactFile(
                    logical, hashlib.sha256(raw).hexdigest(), len(raw)
                )
            except NativeArtifactError as exc:
                raise WorkerBundleSourceError(
                    f"private bundle path is not publishable: {exc}"
                ) from exc
            observed_files.append((logical, path, row))
    if not observed_files:
        raise WorkerBundlePublicationError("private bundle contains no committed files")
    # `bundle_hash._iter_files` hashes in global relative-path order, not os.walk's
    # root-files-before-child-files order. Read again in that exact order and bind
    # the inventory snapshot used to derive the immutable address.
    content = hashlib.sha256()
    files: list[NativeArtifactFile] = []
    for logical, path, row in sorted(observed_files):
        raw, _ = _stable_file(path, owner_uid=owner)
        if len(raw) != row.size or hashlib.sha256(raw).hexdigest() != row.sha256:
            raise WorkerBundlePublicationError(
                f"private bundle file changed between identity reads: {logical}"
            )
        encoded = logical.encode("utf-8")
        content.update(len(encoded).to_bytes(4, "big"))
        content.update(encoded)
        content.update(len(raw).to_bytes(8, "big"))
        content.update(raw)
        files.append(row)
    return tuple(sorted(directories)), tuple(files), content.hexdigest()


def _address(
    content_hash: str,
    directories: tuple[str, ...],
    files: tuple[NativeArtifactFile, ...],
) -> str:
    return canonical_digest(
        "optima.chain.worker-bundle-address",
        {
            "content_hash": content_hash,
            "directories": list(directories),
            "files": [row.to_dict() for row in files],
            "schema": _SCHEMA,
        },
    )


def _published_content_hash(publication: NativeArtifactPublication) -> str:
    result = hashlib.sha256()
    for row in publication.files:
        path = publication.root.joinpath(*PurePosixPath(row.path).parts)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        fd = -1
        try:
            before = path.lstat()
            fd = os.open(path, flags)
            opened = os.fstat(fd)
            if not _same_stat(before, opened):
                raise WorkerBundlePublicationError(
                    f"immutable worker file changed while opening: {row.path}"
                )
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(fd, min(4 << 20, remaining))
                if not chunk:
                    raise WorkerBundlePublicationError(
                        f"immutable worker file was truncated: {row.path}"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise WorkerBundlePublicationError(
                    f"immutable worker file grew while reading: {row.path}"
                )
            after = os.fstat(fd)
            raw = b"".join(chunks)
        except WorkerBundlePublicationError:
            raise
        except OSError as exc:
            raise WorkerBundlePublicationError(
                f"cannot read immutable worker file {row.path}: {exc}"
            ) from None
        finally:
            if fd >= 0:
                os.close(fd)
        if (
            not _same_stat(opened, after)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or stat.S_IMODE(after.st_mode) != 0o444
            or len(raw) != row.size
            or hashlib.sha256(raw).hexdigest() != row.sha256
        ):
            raise WorkerBundlePublicationError(
                f"immutable worker file differs from publication inventory: {row.path}"
            )
        encoded = row.path.encode("utf-8")
        result.update(len(encoded).to_bytes(4, "big"))
        result.update(encoded)
        result.update(len(raw).to_bytes(8, "big"))
        result.update(raw)
    return result.hexdigest()


def _wrap(
    native: NativeArtifactPublication, *, expected_content_hash: str
) -> WorkerBundlePublication:
    if _published_content_hash(native) != expected_content_hash:
        raise WorkerBundlePublicationError(
            "immutable worker tree differs from the committed content hash"
        )
    address = _address(expected_content_hash, native.directories, native.files)
    if native.build_spec_digest != address:
        raise WorkerBundlePublicationError(
            "native carrier address differs from worker bundle identity"
        )
    return WorkerBundlePublication(
        native.root,
        expected_content_hash,
        address,
        native.publication_digest,
        native.directories,
        native.files,
        native.reused,
    )


def publish_worker_bundle(
    private_root: str | os.PathLike[str],
    publication_root: str | os.PathLike[str],
    expected_content_hash: str,
    *,
    work_root: str | os.PathLike[str] | None = None,
    limits: NativeArtifactLimits | None = None,
) -> WorkerBundlePublication:
    """Copy one private, hash-complete intake tree into immutable worker storage."""

    expected = _digest(expected_content_hash, field="expected_content_hash")
    source = _private_root(private_root)
    directories, files, observed = _source_inventory(source)
    if observed != expected:
        raise WorkerBundlePublicationError(
            "private bundle differs from its committed content hash"
        )
    address = _address(expected, directories, files)
    try:
        native = publish_native_artifact(
            source,
            publication_root,
            build_spec_digest=address,
            work_root=work_root,
            limits=limits,
        )
    except NativeArtifactError as exc:
        raise WorkerBundlePublicationError(
            f"worker bundle publication failed: {exc}"
        ) from exc
    if native.directories != directories or native.files != files:
        raise WorkerBundlePublicationError(
            "private bundle changed across immutable publication"
        )
    return _wrap(native, expected_content_hash=expected)


def reopen_worker_bundle(
    path: str | os.PathLike[str],
    expected_content_hash: str,
    *,
    expected_publication_digest: str | None = None,
    expected_receipt_digest: str | None = None,
    expected_owner_uid: int | None | object = _CURRENT_OWNER,
    limits: NativeArtifactLimits | None = None,
) -> WorkerBundlePublication:
    """Reopen and independently rederive one immutable worker publication."""

    expected = _digest(expected_content_hash, field="expected_content_hash")
    requested = Path(path).expanduser()
    address = _digest(requested.name, field="worker publication address")
    try:
        kwargs: dict[str, object] = {
            "expected_build_spec_digest": address,
            "expected_publication_digest": expected_publication_digest,
            "limits": limits,
        }
        if expected_owner_uid is not _CURRENT_OWNER:
            kwargs["expected_owner_uid"] = expected_owner_uid
        native = reopen_native_artifact(requested, **kwargs)
    except NativeArtifactError as exc:
        raise WorkerBundlePublicationError(
            f"worker bundle reopen failed: {exc}"
        ) from exc
    result = _wrap(native, expected_content_hash=expected)
    if expected_receipt_digest is not None and result.digest != _digest(
        expected_receipt_digest, field="expected_receipt_digest"
    ):
        raise WorkerBundlePublicationError(
            "worker bundle receipt differs from expected authority"
        )
    return result


__all__ = [
    "WorkerBundlePublication",
    "WorkerBundlePublicationError",
    "WorkerBundleSourceError",
    "publish_worker_bundle",
    "reopen_worker_bundle",
]
