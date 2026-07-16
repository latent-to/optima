"""Deterministic, content-addressed model-tree provisioning receipts.

This module has no arena or serving-runtime policy.  It establishes one narrow
fact: a concrete directory contained exactly the regular files named by a
canonical receipt when the receipt was produced or reopened.  Receipts are
published outside the model tree and are never replaced in place.
"""

from __future__ import annotations

import concurrent.futures
import errno
import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from optima.stack_identity import canonical_digest, canonical_json_bytes
from optima._strict import require_digest, require_exact_fields


MODEL_PROVISION_SCHEMA_VERSION = 1
MODEL_RECEIPT_PREFIX = "model-provision-sha256-"
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024
_READ_SIZE = 16 * 1024 * 1024
_TRANSIENT_MODEL_PATH_NAMES = frozenset({".cache"})
_STAT_FIELDS = (
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
_NOFOLLOW = getattr(os, "O_NOFOLLOW", None)


class ModelProvisionError(RuntimeError):
    """A model tree or its retained receipt cannot be trusted."""


def _digest(value: object, *, field: str) -> str:
    return require_digest(value, field=field, error=ModelProvisionError)


def _strict_object(
    value: object, *, fields: frozenset[str], label: str
) -> Mapping[str, object]:
    return require_exact_fields(value, fields=fields, label=label, error=ModelProvisionError)


def _logical_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ModelProvisionError("model file path must be non-empty text")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or logical.as_posix() != value
        or any(part in {"", ".", ".."} for part in logical.parts)
        or "\\" in value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ModelProvisionError(f"model file path is not canonical: {value!r}")
    return value


@dataclass(frozen=True, order=True)
class ModelFileRecord:
    """Stable identity row for one model file."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _logical_path(self.path))
        if type(self.size) is not int or self.size < 0:
            raise ModelProvisionError("model file size must be a non-negative integer")
        object.__setattr__(self, "sha256", _digest(self.sha256, field="file sha256"))

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, value: object) -> "ModelFileRecord":
        row = _strict_object(
            value,
            fields=frozenset({"path", "sha256", "size"}),
            label="model file record",
        )
        return cls(
            path=row["path"],  # type: ignore[arg-type]
            size=row["size"],  # type: ignore[arg-type]
            sha256=row["sha256"],  # type: ignore[arg-type]
        )


def _content_digest(files: Iterable[ModelFileRecord]) -> str:
    return canonical_digest(
        "optima.model-content",
        {"files": [record.to_dict() for record in files]},
        schema_version=MODEL_PROVISION_SCHEMA_VERSION,
    )


@dataclass(frozen=True)
class ModelProvisionReceipt:
    """Canonical, path-independent receipt for one complete model tree."""

    content_digest: str
    files: tuple[ModelFileRecord, ...]
    schema_version: int = MODEL_PROVISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ModelProvisionError("unsupported model provision schema_version")
        if not isinstance(self.files, tuple) or not self.files:
            raise ModelProvisionError("model provision receipt must contain files")
        if any(not isinstance(item, ModelFileRecord) for item in self.files):
            raise ModelProvisionError("model provision files must be ModelFileRecord values")
        ordered = tuple(sorted(self.files, key=lambda item: item.path))
        if ordered != self.files:
            raise ModelProvisionError("model provision files are not path-sorted")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)) or len(paths) != len({path.casefold() for path in paths}):
            raise ModelProvisionError("model provision receipt contains duplicate paths")
        supplied = _digest(self.content_digest, field="model content_digest")
        if supplied != _content_digest(self.files):
            raise ModelProvisionError("model content_digest does not match its file records")
        object.__setattr__(self, "content_digest", supplied)

    def to_dict(self) -> dict[str, object]:
        return {
            "content_digest": self.content_digest,
            "files": [record.to_dict() for record in self.files],
            "schema_version": self.schema_version,
            "type": "optima.model-provision",
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_digest(self) -> str:
        return canonical_digest(
            "optima.model-provision-receipt",
            self.to_dict(),
            schema_version=self.schema_version,
        )

    @classmethod
    def from_dict(cls, value: object) -> "ModelProvisionReceipt":
        row = _strict_object(
            value,
            fields=frozenset({"content_digest", "files", "schema_version", "type"}),
            label="model provision receipt",
        )
        if row["type"] != "optima.model-provision":
            raise ModelProvisionError("invalid model provision receipt type")
        raw_files = row["files"]
        if not isinstance(raw_files, list):
            raise ModelProvisionError("model provision files must be an array")
        return cls(
            content_digest=row["content_digest"],  # type: ignore[arg-type]
            files=tuple(ModelFileRecord.from_dict(item) for item in raw_files),
            schema_version=row["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ProvisionedModel:
    receipt: ModelProvisionReceipt
    receipt_path: Path


@dataclass(frozen=True)
class _DiscoveredFile:
    path: str
    native_path: Path
    stat_values: tuple[int, ...]


def _stat_values(info: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(info, field) for field in _STAT_FIELDS)


def _concrete_directory(value: str | os.PathLike[str], *, label: str) -> Path:
    raw = Path(value)
    try:
        info = raw.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ModelProvisionError(f"{label} must be a concrete directory: {raw}")
        return raw.resolve(strict=True)
    except ModelProvisionError:
        raise
    except (OSError, RuntimeError) as exc:
        raise ModelProvisionError(f"cannot resolve {label} {raw}: {exc}") from None


def _discover(
    root: Path,
) -> tuple[
    tuple[_DiscoveredFile, ...],
    tuple[tuple[str, tuple[int, ...]], ...],
]:
    files: list[_DiscoveredFile] = []
    snapshot: list[tuple[str, tuple[int, ...]]] = []
    folded: dict[str, str] = {}

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        try:
            directory_info = directory.stat(follow_symlinks=False)
            if not stat.S_ISDIR(directory_info.st_mode):
                raise ModelProvisionError(f"model directory changed during scan: {directory}")
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except ModelProvisionError:
            raise
        except OSError as exc:
            raise ModelProvisionError(f"cannot scan model directory {directory}: {exc}") from None
        directory_name = PurePosixPath(*prefix).as_posix() if prefix else "."
        snapshot.append((directory_name, _stat_values(directory_info)))
        for entry in entries:
            relative = _logical_path(PurePosixPath(*prefix, entry.name).as_posix())
            if entry.name in _TRANSIENT_MODEL_PATH_NAMES:
                raise ModelProvisionError(
                    f"model tree contains a transient cache path: {relative}"
                )
            previous = folded.get(relative.casefold())
            if previous is not None and previous != relative:
                raise ModelProvisionError(
                    f"case-colliding model paths are not portable: {previous!r}, {relative!r}"
                )
            folded[relative.casefold()] = relative
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ModelProvisionError(f"cannot inspect model path {relative}: {exc}") from None
            snapshot.append((relative, _stat_values(info)))
            if stat.S_ISLNK(info.st_mode):
                raise ModelProvisionError(f"model tree contains a symlink: {relative}")
            if stat.S_ISDIR(info.st_mode):
                visit(Path(entry.path), (*prefix, entry.name))
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise ModelProvisionError(f"model file is hardlinked: {relative}")
                files.append(_DiscoveredFile(relative, Path(entry.path), _stat_values(info)))
            else:
                raise ModelProvisionError(f"model tree contains a special file: {relative}")

    visit(root, ())
    if not files:
        raise ModelProvisionError("model tree has no regular files")
    return tuple(sorted(files, key=lambda item: item.path)), tuple(sorted(snapshot))


def _hash_file(item: _DiscoveredFile) -> ModelFileRecord:
    if _NOFOLLOW is None:
        raise ModelProvisionError("this platform lacks O_NOFOLLOW")
    try:
        fd = os.open(
            item.native_path,
            os.O_RDONLY | os.O_CLOEXEC | _NOFOLLOW,
        )
    except OSError as exc:
        raise ModelProvisionError(f"cannot safely open model file {item.path}: {exc}") from None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_values(before) != item.stat_values
        ):
            raise ModelProvisionError(f"model file changed before hashing: {item.path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, _READ_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        if _stat_values(after) != _stat_values(before):
            raise ModelProvisionError(f"model file changed while hashing: {item.path}")
        return ModelFileRecord(item.path, before.st_size, digest.hexdigest())
    finally:
        os.close(fd)


def _build_receipt(root: Path, *, workers: int) -> ModelProvisionReceipt:
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ModelProvisionError("workers must be an integer in [1, 64]")
    discovered, before = _discover(root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        records = tuple(sorted(executor.map(_hash_file, discovered), key=lambda item: item.path))
    after_files, after = _discover(root)
    if before != after or tuple(item.path for item in discovered) != tuple(
        item.path for item in after_files
    ):
        raise ModelProvisionError("model tree changed while it was being sealed")
    return ModelProvisionReceipt(content_digest=_content_digest(records), files=records)


def _decode_receipt(raw: bytes) -> ModelProvisionReceipt:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ModelProvisionError(f"duplicate JSON key in model receipt: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
    except ModelProvisionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelProvisionError(f"model provision receipt is not valid JSON: {exc}") from None
    receipt = ModelProvisionReceipt.from_dict(value)
    if raw != receipt.canonical_bytes:
        raise ModelProvisionError("model provision receipt bytes are not canonical")
    return receipt


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    if _NOFOLLOW is None:
        raise ModelProvisionError("this platform lacks O_NOFOLLOW")
    try:
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | _NOFOLLOW, dir_fd=directory_fd)
    except OSError as exc:
        raise ModelProvisionError(f"cannot open model provision receipt: {exc}") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ModelProvisionError("model provision receipt is not a standalone regular file")
        if before.st_size > _MAX_RECEIPT_BYTES:
            raise ModelProvisionError("model provision receipt exceeds its size bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ModelProvisionError("model provision receipt was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ModelProvisionError("model provision receipt grew while reading")
        if _stat_values(os.fstat(fd)) != _stat_values(before):
            raise ModelProvisionError("model provision receipt changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _receipt_name(digest: str) -> str:
    return f"{MODEL_RECEIPT_PREFIX}{digest}.json"


def _open_publication_directory(path: Path) -> int:
    if _NOFOLLOW is None:
        raise ModelProvisionError("this platform lacks O_NOFOLLOW")
    try:
        return os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | _NOFOLLOW)
    except OSError as exc:
        raise ModelProvisionError(f"cannot open receipt publication directory: {exc}") from None


def _read_published(
    path: Path, *, require_content_addressed_name: bool = True
) -> ModelProvisionReceipt:
    parent = _concrete_directory(path.parent, label="receipt publication directory")
    directory_fd = _open_publication_directory(parent)
    try:
        receipt = _decode_receipt(_read_regular_at(directory_fd, path.name))
    finally:
        os.close(directory_fd)
    if require_content_addressed_name and path.name != _receipt_name(receipt.receipt_digest):
        raise ModelProvisionError("model receipt filename does not match its canonical digest")
    return receipt


def _publish_no_clobber(directory: Path, receipt: ModelProvisionReceipt) -> Path:
    name = _receipt_name(receipt.receipt_digest)
    payload = receipt.canonical_bytes
    directory_fd = _open_publication_directory(directory)
    temporary = f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    linked = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if _NOFOLLOW is not None:
            flags |= _NOFOLLOW
        fd = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise ModelProvisionError("short write while publishing model receipt")
                view = view[written:]
            os.fchmod(fd, 0o444)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            pass
        finally:
            os.unlink(temporary, dir_fd=directory_fd)
            os.fsync(directory_fd)
    except Exception as exc:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except OSError as cleanup:
            if cleanup.errno != errno.ENOENT:
                raise ModelProvisionError(f"cannot clean receipt publication: {cleanup}") from None
        if isinstance(exc, ModelProvisionError):
            raise
        raise ModelProvisionError(f"cannot publish model provision receipt: {exc}") from None
    finally:
        os.close(directory_fd)
    path = directory / name
    if not linked:
        existing = _read_published(path)
        if existing.canonical_bytes != payload:
            raise ModelProvisionError("content-addressed model receipt collision")
    return path


def provision_model(
    model_root: str | os.PathLike[str],
    publication_directory: str | os.PathLike[str],
    *,
    expected_content_digest: str | None = None,
    workers: int = 8,
) -> ProvisionedModel:
    """Seal ``model_root`` and publish an immutable content-addressed receipt."""

    root = _concrete_directory(model_root, label="model root")
    publication = _concrete_directory(
        publication_directory, label="receipt publication directory"
    )
    try:
        publication.relative_to(root)
    except ValueError:
        pass
    else:
        raise ModelProvisionError("receipt publication directory must be outside the model tree")
    expected = (
        None
        if expected_content_digest is None
        else _digest(expected_content_digest, field="expected_content_digest")
    )
    receipt = _build_receipt(root, workers=workers)
    if expected is not None and receipt.content_digest != expected:
        raise ModelProvisionError("model tree does not match expected_content_digest")
    path = _publish_no_clobber(publication, receipt)
    reopened = _read_published(path)
    if reopened != receipt:
        raise ModelProvisionError("published model receipt did not reopen exactly")
    return ProvisionedModel(reopened, path)


def reopen_model_provision(
    model_root: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    *,
    expected_content_digest: str | None = None,
    expected_receipt_digest: str | None = None,
    workers: int = 8,
) -> ProvisionedModel:
    """Reopen a retained receipt and re-hash the complete model tree against it."""

    root = _concrete_directory(model_root, label="model root")
    path = Path(receipt_path)
    receipt = _read_published(path)
    if expected_content_digest is not None and receipt.content_digest != _digest(
        expected_content_digest, field="expected_content_digest"
    ):
        raise ModelProvisionError("model receipt does not match expected_content_digest")
    if expected_receipt_digest is not None and receipt.receipt_digest != _digest(
        expected_receipt_digest, field="expected_receipt_digest"
    ):
        raise ModelProvisionError("model receipt does not match expected_receipt_digest")
    observed = _build_receipt(root, workers=workers)
    if observed != receipt:
        raise ModelProvisionError("model tree does not match its retained provision receipt")
    return ProvisionedModel(receipt, path)


def reopen_embedded_model_provision(
    model_root: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    *,
    expected_content_digest: str,
    expected_receipt_digest: str,
    workers: int = 8,
) -> ProvisionedModel:
    """Reopen a signed-release copy whose enclosing descriptor supplies its address.

    Ordinary provision receipts remain content-addressed by filename.  A release
    ships the same canonical bytes under one fixed artifact role name, so this path
    requires both independently signed expected digests before re-hashing the model.
    """

    root = _concrete_directory(model_root, label="model root")
    path = Path(receipt_path)
    expected_content = _digest(
        expected_content_digest, field="expected_content_digest"
    )
    expected_receipt = _digest(
        expected_receipt_digest, field="expected_receipt_digest"
    )
    receipt = _read_published(path, require_content_addressed_name=False)
    if (
        receipt.content_digest != expected_content
        or receipt.receipt_digest != expected_receipt
    ):
        raise ModelProvisionError("embedded model receipt differs from signed identity")
    observed = _build_receipt(root, workers=workers)
    if observed != receipt:
        raise ModelProvisionError("model tree does not match its embedded provision receipt")
    return ProvisionedModel(receipt, path)


__all__ = [
    "MODEL_PROVISION_SCHEMA_VERSION",
    "MODEL_RECEIPT_PREFIX",
    "ModelFileRecord",
    "ModelProvisionError",
    "ModelProvisionReceipt",
    "ProvisionedModel",
    "provision_model",
    "reopen_embedded_model_provision",
    "reopen_model_provision",
]
