"""Content-addressed storage for semantically opaque referee evidence bytes."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path

from optima.stack_identity import canonical_json_bytes, require_sha256_hex


DEFAULT_MAX_EVIDENCE_BYTES = 64 << 20
HARD_MAX_EVIDENCE_BYTES = 1 << 30
_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_SCHEMA = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_MEDIA = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/[a-z0-9][a-z0-9!#$&^_.+-]{0,63}$"
)


class EvidenceStoreError(ValueError):
    """Evidence identity, bytes, or filesystem state is invalid."""


@dataclass(frozen=True)
class EvidenceArtifactRef:
    domain: str
    sha256: str
    size: int
    media_type: str
    schema: str

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or _DOMAIN.fullmatch(self.domain) is None:
            raise EvidenceStoreError("evidence domain is invalid")
        try:
            digest = require_sha256_hex(self.sha256, field="evidence sha256")
        except ValueError as exc:
            raise EvidenceStoreError(str(exc)) from None
        object.__setattr__(self, "sha256", digest)
        if (isinstance(self.size, bool) or not isinstance(self.size, int)
                or not 0 <= self.size <= HARD_MAX_EVIDENCE_BYTES):
            raise EvidenceStoreError("evidence size is invalid")
        if not isinstance(self.media_type, str) or _MEDIA.fullmatch(self.media_type) is None:
            raise EvidenceStoreError("evidence media_type is invalid")
        if not isinstance(self.schema, str) or _SCHEMA.fullmatch(self.schema) is None:
            raise EvidenceStoreError("evidence schema is invalid")

    def to_dict(self) -> dict[str, object]:
        return {"domain": self.domain, "media_type": self.media_type,
                "schema": self.schema, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceArtifactRef":
        fields = {"domain", "media_type", "schema", "sha256", "size"}
        if type(value) is not dict or set(value) != fields:
            raise EvidenceStoreError("evidence reference schema is not closed")
        return cls(**value)  # type: ignore[arg-type]


def _limit(value: object) -> int:
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 1 <= value <= HARD_MAX_EVIDENCE_BYTES):
        raise EvidenceStoreError("max evidence size is invalid")
    return value


def _absolute(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise EvidenceStoreError("evidence root must be a path")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise EvidenceStoreError("evidence root must be canonical and absolute")
    return path


def _directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise EvidenceStoreError(f"evidence directory is unavailable: {exc}") from None
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise EvidenceStoreError("evidence directory has an unsafe owner or mode")
    try:
        if path.resolve(strict=True) != path:
            raise EvidenceStoreError("evidence directory traverses a symlink")
    except OSError as exc:
        raise EvidenceStoreError(f"cannot resolve evidence directory: {exc}") from None


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise EvidenceStoreError(f"cannot fsync evidence directory: {exc}") from None


def prepare_evidence_root(root: str | Path) -> Path:
    path = _absolute(root)
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        pass
    except OSError as exc:
        raise EvidenceStoreError(f"cannot create evidence root: {exc}") from None
    _directory(path)
    _fsync_dir(path)
    return path


def _target(root: Path, reference: EvidenceArtifactRef, *, create: bool) -> Path:
    domain, shard = root / reference.domain, root / reference.domain / reference.sha256[:2]
    if create:
        for directory in (domain, shard):
            try:
                directory.mkdir(mode=0o700, exist_ok=False)
                _fsync_dir(directory.parent)
            except FileExistsError:
                pass
            except OSError as exc:
                raise EvidenceStoreError(f"cannot create evidence directory: {exc}") from None
            _directory(directory)
    else:
        _directory(domain)
        _directory(shard)
    target = shard / reference.sha256
    try:
        target.relative_to(root)
    except ValueError:
        raise EvidenceStoreError("evidence path escapes its store root") from None
    return target


def publish_evidence(
    root: str | Path,
    payload: bytes,
    *,
    domain: str,
    media_type: str,
    schema: str,
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> EvidenceArtifactRef:
    """Atomically publish bytes, or accept an exact existing duplicate."""

    limit = _limit(max_bytes)
    if not isinstance(payload, bytes):
        raise EvidenceStoreError("evidence payload must be exact bytes")
    if len(payload) > limit:
        raise EvidenceStoreError("evidence payload exceeds its size limit")
    reference = EvidenceArtifactRef(domain, hashlib.sha256(payload).hexdigest(),
                                    len(payload), media_type, schema)
    store = prepare_evidence_root(root)
    target = _target(store, reference, create=True)
    if os.path.lexists(target):
        if reopen_evidence(store, reference, max_bytes=limit) != payload:
            raise EvidenceStoreError("existing evidence is not an exact duplicate")
        return reference
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                if stream.write(payload) != len(payload):
                    raise EvidenceStoreError("evidence artifact write stalled")
                stream.flush()
                os.fchmod(fd, 0o400)
                os.fsync(fd)
            sealed = os.fstat(fd)
        finally:
            os.close(fd)
        if (not stat.S_ISREG(sealed.st_mode) or sealed.st_nlink != 1
                or sealed.st_uid != os.geteuid() or stat.S_IMODE(sealed.st_mode) != 0o400
                or sealed.st_size != reference.size):
            raise EvidenceStoreError("staged evidence file did not seal safely")
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            if reopen_evidence(store, reference, max_bytes=limit) != payload:
                raise EvidenceStoreError("concurrent evidence publication differs")
        else:
            temporary.unlink()
            _fsync_dir(target.parent)
            reopen_evidence(store, reference, max_bytes=limit)
        return reference
    except OSError as exc:
        raise EvidenceStoreError(f"cannot publish evidence artifact: {exc}") from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def reopen_evidence(
    root: str | Path,
    reference: EvidenceArtifactRef,
    *,
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> bytes:
    """Reopen and authenticate exact bytes without interpreting their schema."""

    limit = _limit(max_bytes)
    if type(reference) is not EvidenceArtifactRef:
        raise EvidenceStoreError("evidence reference must be exact and typed")
    if reference.size > limit:
        raise EvidenceStoreError("evidence reference exceeds its size limit")
    store = _absolute(root)
    _directory(store)
    target = _target(store, reference, create=False)
    try:
        before = target.lstat()
    except OSError as exc:
        raise EvidenceStoreError(f"evidence artifact is unavailable: {exc}") from None
    if (stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1 or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o400 or before.st_size != reference.size
            or before.st_size > limit):
        raise EvidenceStoreError("evidence artifact has an unsafe shape, owner, mode, or size")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise EvidenceStoreError(f"cannot open evidence artifact: {exc}") from None
    try:
        opened = os.fstat(fd)
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(opened, name) for name in stable):
            raise EvidenceStoreError("evidence artifact changed while opening")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        if len(payload) != reference.size:
            raise EvidenceStoreError("evidence artifact was truncated or grew while reading")
        after = os.fstat(fd)
        if any(getattr(opened, name) != getattr(after, name) for name in stable):
            raise EvidenceStoreError("evidence artifact changed while reading")
    finally:
        os.close(fd)
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise EvidenceStoreError("evidence artifact digest mismatch")
    return payload


def publish_canonical_json_evidence(
    root: str | Path,
    value: object,
    *,
    domain: str,
    schema: str,
    media_type: str = "application/json",
    max_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> EvidenceArtifactRef:
    """Canonicalize trusted input, then store it as semantically opaque bytes."""

    return publish_evidence(root, canonical_json_bytes(value), domain=domain,
                            media_type=media_type, schema=schema, max_bytes=max_bytes)


__all__ = ["DEFAULT_MAX_EVIDENCE_BYTES", "EvidenceArtifactRef", "EvidenceStoreError",
           "prepare_evidence_root", "publish_canonical_json_evidence",
           "publish_evidence", "reopen_evidence"]
