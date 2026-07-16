"""Chain-independent, reviewed Optima Engine release products.

Release construction consumes only an ``EngineReleaseManifest``, exact approved
integration records, a reopened materialized engine tree, and validator-owned build
artifacts.  It never reads chain state, miner URLs, wallets, or evaluation incumbents.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from optima.engine_tree import MaterializedEngineTree, reopen_materialized_engine_tree
from optima.stack_identity import canonical_digest, canonical_json_bytes
from optima.stack_manifest import (
    EngineReleaseManifest,
    IntegrationReviewRecord,
    StackManifestError,
)
from optima._strict import require_digest, require_exact_fields


RELEASE_SCHEMA_VERSION = 1
RUNTIME_DISTRIBUTION = "optima-engine"
RUNTIME_VERSION = "0.0.1"
RUNTIME_WHEEL = "optima_engine-0.0.1-py3-none-any.whl"
SOURCE_ARCHIVE = "optima-engine-source-0.0.1.tar.gz"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE = re.compile(r"[a-z0-9][a-z0-9._/:+-]{0,255}@sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
_FILE_MODE = 0o444
_DIR_MODE = 0o555
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_ARTIFACT_ROLES = {
    "runtime_source": (SOURCE_ARCHIVE, "application/gzip"),
    "runtime_wheel": (RUNTIME_WHEEL, "application/vnd.python.wheel"),
    "model_receipt": ("model-provision.json", "application/vnd.optima.model-provision+json"),
    "seccomp": ("seccomp.json", "application/vnd.optima.seccomp+json"),
    "reference_manifest": ("reference-manifest.json", "application/vnd.optima.reference-manifest+json"),
    "calibration_manifest": ("calibration-manifest.json", "application/vnd.optima.calibration-manifest+json"),
    "sbom": ("sbom.spdx.json", "application/spdx+json"),
    "provenance": ("provenance.intoto.json", "application/vnd.in-toto+json"),
}
_FORBIDDEN_ENV_PREFIXES = (
    "OPTIMA_", "PYTHON", "LD_", "BT_", "BITTENSOR", "WALLET", "MINER",
)
_FORBIDDEN_ENV_KEYS = frozenset(
    {
        "HOME", "LANG", "LC_ALL", "PATH", "SGLANG_PLUGINS",
        "SGLANG_PLUGIN_PATH", "TMPDIR",
    }
)
_RESERVED_ENGINE_FLAGS = frozenset(
    {
        "--model", "--model-path", "--served-model-name", "--tp", "--tp-size",
        "--tensor-parallel-size", "--tensor-parallel-degree",
    }
)

_RUNTIME_TOP_LEVEL = frozenset(
    {
        "__init__.py",
        "_strict.py",
        "bootstrap.py",
        "bundle_hash.py",
        "capabilities.py",
        "dep_policy.py",
        "deppatch.py",
        "dispatch.py",
        "manifest.py",
        "moe_export.py",
        "receipts.py",
        "rebuild.py",
        "registry.py",
        "release.py",
        "release_runtime.py",
        "model_provision.py",
        "sandbox.py",
        "seam.py",
        "seams.py",
        "slots.py",
        "stack_identity.py",
        "stack_manifest.py",
        "engine_tree.py",
        "tensor_spec.py",
        "vendor_provenance.json",
    }
)
_RUNTIME_PREFIXES = (
    "optima/integrations/",
    "optima/patchers/",
    "optima/arena_assets/",
    "optima_kernels/",
)
_RUNTIME_EXACT = frozenset(
    {
        "optima/eval/__init__.py",
        "optima/eval/native_artifact.py",
        "optima/eval/engine_launch.py",
        "optima/eval/evidence_store.py",
        "optima/eval/calibration.py",
        "optima/eval/qualification.py",
        "optima/eval/seccomp_moby_v0_2_1.json",
    }
)
_RELEASE_LEGAL = ("LICENSE", "NOTICE", "LICENSES/MINIMAX_COMMUNITY_LICENSE.txt", "LICENSES/SGLANG.txt")


class ReleaseError(RuntimeError):
    """A release identity, build artifact, signature, or publication is invalid."""


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=ReleaseError)


def _strict(value: object, fields: set[str], name: str) -> dict[str, Any]:
    return require_exact_fields(
        value, fields=frozenset(fields), label=name, error=ReleaseError, exact_dict=True
    )


def _stable_regular(path: Path, *, limit: int = 1 << 30) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ReleaseError("release construction requires O_NOFOLLOW")
    try:
        fd = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise ReleaseError(f"cannot open release input {path}: {exc}") from None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > limit
        ):
            raise ReleaseError(f"release input is not one bounded regular file: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ReleaseError(f"release input was truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ReleaseError(f"release input grew while reading: {path}")
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise ReleaseError(f"release input changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ReleaseArtifact:
    name: str
    media_type: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or PurePosixPath(self.name).name != self.name
            or not isinstance(self.media_type, str)
            or not self.media_type
            or type(self.size) is not int
            or self.size < 0
        ):
            raise ReleaseError("release artifact metadata is malformed")
        object.__setattr__(self, "sha256", _digest(self.sha256, "artifact sha256"))

    @classmethod
    def from_bytes(cls, name: str, media_type: str, payload: bytes) -> "ReleaseArtifact":
        return cls(name, media_type, _sha(payload), len(payload))

    def to_dict(self) -> dict[str, object]:
        return {"media_type": self.media_type, "name": self.name, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, value: object) -> "ReleaseArtifact":
        return cls(**_strict(value, {"media_type", "name", "sha256", "size"}, "release artifact"))


@dataclass(frozen=True)
class ModelReleaseIdentity:
    model_id: str
    revision: str
    manifest_digest: str
    content_digest: str
    receipt_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id or self.model_id.strip() != self.model_id:
            raise ReleaseError("model_id is malformed")
        if not isinstance(self.revision, str) or _REVISION.fullmatch(self.revision) is None:
            raise ReleaseError("model revision is not immutable")
        object.__setattr__(self, "manifest_digest", _digest(self.manifest_digest, "model manifest"))
        object.__setattr__(self, "content_digest", _digest(self.content_digest, "model content"))
        object.__setattr__(self, "receipt_digest", _digest(self.receipt_digest, "model provision receipt"))

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_provisioning(
        cls,
        model_id: str,
        revision: str,
        manifest_digest: str,
        receipt: object,
    ) -> "ModelReleaseIdentity":
        from optima.model_provision import ModelProvisionReceipt

        if not isinstance(receipt, ModelProvisionReceipt):
            raise ReleaseError("model release requires an exact provision receipt")
        return cls(
            model_id, revision, manifest_digest, receipt.content_digest,
            receipt.receipt_digest,
        )

    @classmethod
    def from_dict(cls, value: object) -> "ModelReleaseIdentity":
        return cls(**_strict(value, set(cls.__dataclass_fields__), "model release identity"))


@dataclass(frozen=True)
class ServeSpec:
    base_image: str
    oci_platform: str
    model_mount: str
    tp_size: int
    engine_arguments: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_image, str) or _IMAGE.fullmatch(self.base_image) is None:
            raise ReleaseError("serve base_image must be digest pinned")
        if not isinstance(self.oci_platform, str) or re.fullmatch(r"[a-z0-9._-]+/[a-z0-9._-]+", self.oci_platform) is None:
            raise ReleaseError("serve OCI platform is malformed")
        mount = PurePosixPath(self.model_mount)
        if not mount.is_absolute() or ".." in mount.parts or mount.as_posix() != self.model_mount:
            raise ReleaseError("serve model mount is not canonical")
        if type(self.tp_size) is not int or self.tp_size < 1:
            raise ReleaseError("serve tp_size must be positive")
        args = tuple(self.engine_arguments)
        env = tuple(self.environment)
        reserved_flags = {
            value.split("=", 1)[0] for value in args if value.startswith("--")
        } & _RESERVED_ENGINE_FLAGS
        if (
            any(not isinstance(value, str) or not value or "\x00" in value for value in args)
            or reserved_flags
            or env != tuple(sorted(env))
            or len({key for key, _ in env}) != len(env)
            or any(
                not isinstance(key, str)
                or re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None
                or not isinstance(value, str)
                or any(char in value for char in "\x00\r\n")
                for key, value in env
            )
            or any(
                key in _FORBIDDEN_ENV_KEYS
                or key.startswith(_FORBIDDEN_ENV_PREFIXES)
                for key, _ in env
            )
        ):
            raise ReleaseError(
                "serve arguments/environment are not canonical or override validator-owned settings"
            )
        object.__setattr__(self, "engine_arguments", args)
        object.__setattr__(self, "environment", env)

    @property
    def command_arguments(self) -> tuple[str, ...]:
        """Complete SGLang arguments with validator-owned model and TP identity."""

        return (
            "--model-path", self.model_mount,
            "--tp-size", str(self.tp_size),
            *self.engine_arguments,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "base_image": self.base_image,
            "engine_arguments": list(self.engine_arguments),
            "environment": [list(row) for row in self.environment],
            "model_mount": self.model_mount,
            "oci_platform": self.oci_platform,
            "tp_size": self.tp_size,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ServeSpec":
        row = _strict(value, {"base_image", "engine_arguments", "environment", "model_mount", "oci_platform", "tp_size"}, "serve spec")
        if type(row["engine_arguments"]) is not list or type(row["environment"]) is not list:
            raise ReleaseError("serve arrays are malformed")
        return cls(**{**row, "engine_arguments": tuple(row["engine_arguments"]), "environment": tuple(tuple(item) for item in row["environment"])})


@dataclass(frozen=True, order=True)
class NativeReleaseFile:
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        logical = PurePosixPath(self.path)
        if (
            not isinstance(self.path, str)
            or not self.path
            or logical.is_absolute()
            or logical.as_posix() != self.path
            or ".." in logical.parts
            or type(self.size) is not int
            or self.size < 0
        ):
            raise ReleaseError("native release file is malformed")
        object.__setattr__(self, "sha256", _digest(self.sha256, "native file sha256"))

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, value: object) -> "NativeReleaseFile":
        return cls(**_strict(value, {"path", "sha256", "size"}, "native release file"))


@dataclass(frozen=True)
class NativeReleaseIdentity:
    build_spec: Mapping[str, object]
    build_spec_digest: str
    publication_digest: str
    directories: tuple[str, ...]
    files: tuple[NativeReleaseFile, ...]

    def __post_init__(self) -> None:
        from optima.eval.engine_launch import NativeBuildSpec

        if not isinstance(self.build_spec, Mapping):
            raise ReleaseError("native build spec is malformed")
        try:
            spec = NativeBuildSpec.from_dict(dict(self.build_spec))
        except Exception as exc:
            raise ReleaseError(f"native build spec is invalid: {exc}") from None
        supplied_build = _digest(self.build_spec_digest, "native build spec")
        if supplied_build != spec.digest:
            raise ReleaseError("native build spec digest differs from its typed spec")
        directories = tuple(self.directories)
        files = tuple(self.files)
        if any(not isinstance(row, NativeReleaseFile) for row in files):
            raise ReleaseError("native artifact inventory is not typed")
        if (
            directories != tuple(sorted(set(directories)))
            or files != tuple(sorted(files, key=lambda row: row.path))
            or len({row.path for row in files}) != len(files)
        ):
            raise ReleaseError("native artifact inventory is not canonical")
        identity = {
            "build_spec_digest": supplied_build,
            "directories": list(directories),
            "files": [row.to_dict() for row in files],
        }
        supplied_publication = _digest(self.publication_digest, "native publication")
        if supplied_publication != canonical_digest(
            "optima.native-artifact-publication", identity
        ):
            raise ReleaseError("native publication digest differs from its inventory")
        object.__setattr__(self, "build_spec", MappingProxyType(spec.to_dict()))
        object.__setattr__(self, "build_spec_digest", supplied_build)
        object.__setattr__(self, "publication_digest", supplied_publication)
        object.__setattr__(self, "directories", directories)
        object.__setattr__(self, "files", files)

    @classmethod
    def from_publication(cls, build_spec: object, publication: object) -> "NativeReleaseIdentity":
        from optima.eval.engine_launch import NativeBuildSpec
        from optima.eval.native_artifact import NativeArtifactPublication

        if not isinstance(build_spec, NativeBuildSpec) or not isinstance(
            publication, NativeArtifactPublication
        ):
            raise ReleaseError("native release requires typed build and publication values")
        if build_spec.digest != publication.build_spec_digest:
            raise ReleaseError("native publication names another build spec")
        return cls(
            build_spec=build_spec.to_dict(),
            build_spec_digest=build_spec.digest,
            publication_digest=publication.publication_digest,
            directories=publication.directories,
            files=tuple(
                NativeReleaseFile(row.path, row.sha256, row.size)
                for row in publication.files
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "build_spec": dict(self.build_spec),
            "build_spec_digest": self.build_spec_digest,
            "directories": list(self.directories),
            "files": [row.to_dict() for row in self.files],
            "publication_digest": self.publication_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NativeReleaseIdentity":
        row = _strict(
            value,
            {"build_spec", "build_spec_digest", "directories", "files", "publication_digest"},
            "native release identity",
        )
        if type(row["directories"]) is not list or type(row["files"]) is not list:
            raise ReleaseError("native release inventory arrays are malformed")
        return cls(
            build_spec=row["build_spec"],
            build_spec_digest=row["build_spec_digest"],
            publication_digest=row["publication_digest"],
            directories=tuple(row["directories"]),
            files=tuple(NativeReleaseFile.from_dict(item) for item in row["files"]),
        )


@dataclass(frozen=True)
class EngineReleaseDescriptor:
    release_manifest: EngineReleaseManifest
    engine_tree_digest: str
    runtime_source: ReleaseArtifact
    runtime_wheel: ReleaseArtifact
    model_receipt: ReleaseArtifact
    seccomp: ReleaseArtifact
    reference_manifest: ReleaseArtifact
    calibration_manifest: ReleaseArtifact
    sbom: ReleaseArtifact
    provenance: ReleaseArtifact
    upstream_repository: str
    upstream_revision: str
    sglang_version: str
    model: ModelReleaseIdentity
    native: NativeReleaseIdentity
    integration_records: tuple[IntegrationReviewRecord, ...]
    serve: ServeSpec
    schema_version: int = RELEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.release_manifest, EngineReleaseManifest):
            raise ReleaseError("release descriptor requires an EngineReleaseManifest")
        for field in ("engine_tree_digest",):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        for field in _ARTIFACT_ROLES:
            if not isinstance(getattr(self, field), ReleaseArtifact):
                raise ReleaseError("release descriptor artifacts are untyped")
            expected_name, expected_media = _ARTIFACT_ROLES[field]
            artifact = getattr(self, field)
            if (artifact.name, artifact.media_type) != (expected_name, expected_media):
                raise ReleaseError(f"release artifact role {field} has the wrong name/media type")
        artifacts = tuple(getattr(self, field) for field in _ARTIFACT_ROLES)
        if len({row.name for row in artifacts}) != len(artifacts):
            raise ReleaseError("release artifact role names are not unique")
        if (
            not isinstance(self.upstream_repository, str)
            or not self.upstream_repository.startswith("https://")
            or not isinstance(self.upstream_revision, str)
            or _REVISION.fullmatch(self.upstream_revision) is None
            or not isinstance(self.sglang_version, str)
            or not self.sglang_version
            or not isinstance(self.model, ModelReleaseIdentity)
            or not isinstance(self.native, NativeReleaseIdentity)
            or not isinstance(self.serve, ServeSpec)
            or type(self.schema_version) is not int
            or self.schema_version != RELEASE_SCHEMA_VERSION
        ):
            raise ReleaseError("release descriptor identity is malformed")
        if self.native.build_spec.get("tree_digest") != self.engine_tree_digest:
            raise ReleaseError(
                "native build product is not bound to the released engine tree"
            )
        records = tuple(self.integration_records)
        if any(not isinstance(row, IntegrationReviewRecord) for row in records):
            raise ReleaseError("release integration records are untyped")
        by_target = {row.target_id: row for row in records}
        if len(by_target) != len(records):
            raise ReleaseError("release integration records are duplicated")
        try:
            self.release_manifest.validate_integrations(by_target)
        except StackManifestError as exc:
            raise ReleaseError(f"release integration authority differs: {exc}") from None
        object.__setattr__(self, "integration_records", tuple(sorted(records, key=lambda row: row.target_id)))

    @property
    def digest(self) -> str:
        return canonical_digest("optima.engine-release.descriptor", self.to_dict())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_manifest": self.calibration_manifest.to_dict(),
            "engine_tree_digest": self.engine_tree_digest,
            "integration_records": [row.to_dict() for row in self.integration_records],
            "model": self.model.to_dict(),
            "model_receipt": self.model_receipt.to_dict(),
            "native": self.native.to_dict(),
            "provenance": self.provenance.to_dict(),
            "reference_manifest": self.reference_manifest.to_dict(),
            "release_manifest": self.release_manifest.to_dict(),
            "runtime_source": self.runtime_source.to_dict(),
            "runtime_wheel": self.runtime_wheel.to_dict(),
            "sbom": self.sbom.to_dict(),
            "schema_version": self.schema_version,
            "seccomp": self.seccomp.to_dict(),
            "serve": self.serve.to_dict(),
            "sglang_version": self.sglang_version,
            "upstream_repository": self.upstream_repository,
            "upstream_revision": self.upstream_revision,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EngineReleaseDescriptor":
        fields = set(cls.__dataclass_fields__)
        row = _strict(value, fields, "engine release descriptor")
        return cls(
            **{
                **row,
                "release_manifest": EngineReleaseManifest.from_dict(row["release_manifest"]),
                "runtime_source": ReleaseArtifact.from_dict(row["runtime_source"]),
                "runtime_wheel": ReleaseArtifact.from_dict(row["runtime_wheel"]),
                "model_receipt": ReleaseArtifact.from_dict(row["model_receipt"]),
                "seccomp": ReleaseArtifact.from_dict(row["seccomp"]),
                "reference_manifest": ReleaseArtifact.from_dict(row["reference_manifest"]),
                "calibration_manifest": ReleaseArtifact.from_dict(row["calibration_manifest"]),
                "sbom": ReleaseArtifact.from_dict(row["sbom"]),
                "provenance": ReleaseArtifact.from_dict(row["provenance"]),
                "model": ModelReleaseIdentity.from_dict(row["model"]),
                "native": NativeReleaseIdentity.from_dict(row["native"]),
                "serve": ServeSpec.from_dict(row["serve"]),
                "integration_records": tuple(IntegrationReviewRecord.from_dict(item) for item in row["integration_records"]),
            }
        )


@dataclass(frozen=True)
class ReleaseSignature:
    descriptor_digest: str
    public_key: str
    signature: str
    algorithm: str = "ed25519"

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptor_digest", _digest(self.descriptor_digest, "descriptor digest"))
        if (
            self.algorithm != "ed25519"
            or not isinstance(self.public_key, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.public_key) is None
            or not isinstance(self.signature, str)
            or re.fullmatch(r"[0-9a-f]{128}", self.signature) is None
        ):
            raise ReleaseError("release signature is malformed")

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "ReleaseSignature":
        return cls(**_strict(value, set(cls.__dataclass_fields__), "release signature"))


def sign_release(descriptor: EngineReleaseDescriptor, private_key: bytes) -> ReleaseSignature:
    if not isinstance(descriptor, EngineReleaseDescriptor) or type(private_key) is not bytes or len(private_key) != 32:
        raise ReleaseError("release signing requires a descriptor and raw 32-byte Ed25519 key")
    return _sign_digest(descriptor.digest, private_key)


def _sign_digest(digest: str, private_key: bytes) -> ReleaseSignature:
    digest = _digest(digest, "signed digest")
    if type(private_key) is not bytes or len(private_key) != 32:
        raise ReleaseError("signing requires a raw 32-byte Ed25519 key")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        key = Ed25519PrivateKey.from_private_bytes(private_key)
        public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        signature = key.sign(bytes.fromhex(digest))
    except (ImportError, ValueError) as exc:
        raise ReleaseError(f"Ed25519 signing is unavailable: {exc}") from None
    return ReleaseSignature(digest, public.hex(), signature.hex())


def _expected_public_key(value: bytes | str) -> str:
    if type(value) is bytes:
        value = value.hex()
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReleaseError("trusted expected public key must be raw 32-byte Ed25519 hex")
    return value


def verify_release_signature(
    descriptor: EngineReleaseDescriptor,
    signature: ReleaseSignature,
    *,
    expected_public_key: bytes | str,
) -> None:
    if not isinstance(descriptor, EngineReleaseDescriptor) or not isinstance(signature, ReleaseSignature) or signature.descriptor_digest != descriptor.digest:
        raise ReleaseError("release signature names another descriptor")
    if signature.public_key != _expected_public_key(expected_public_key):
        raise ReleaseError("release signature is not from the trusted release authority")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        raise ReleaseError("release signature verification failed") from None
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(signature.public_key)).verify(
            bytes.fromhex(signature.signature), bytes.fromhex(descriptor.digest)
        )
    except (InvalidSignature, ValueError):
        raise ReleaseError("release signature verification failed") from None


def _runtime_files(source_root: str | Path) -> dict[str, bytes]:
    root = Path(source_root).resolve(strict=True)
    files: dict[str, bytes] = {}
    candidates = [*(root / "optima").rglob("*"), *(root / "optima_kernels").rglob("*"), *(root / "LICENSES").rglob("*")]
    for legal in (root / "LICENSE", root / "NOTICE"):
        candidates.append(legal)
    for path in sorted(set(candidates)):
        if path.is_symlink():
            raise ReleaseError(f"runtime release source contains a symlink: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        include = (
            relative in _RUNTIME_EXACT
            or relative in _RELEASE_LEGAL
            or any(relative.startswith(prefix) for prefix in _RUNTIME_PREFIXES)
            or (
                relative.startswith("optima/")
                and PurePosixPath(relative).parent == PurePosixPath("optima")
                and PurePosixPath(relative).name in _RUNTIME_TOP_LEVEL
            )
        )
        if not include:
            continue
        if "__pycache__" in PurePosixPath(relative).parts or relative.endswith((".pyc", ".pyo")):
            continue
        if not path.is_file():
            raise ReleaseError(f"runtime release source contains a special object: {relative}")
        files[relative] = _stable_regular(path)
    required = {
        "optima/__init__.py",
        "optima/bootstrap.py",
        "optima/seam.py",
        "optima/integrations/sglang_plugin.py",
        "optima_kernels/__init__.py",
        "LICENSE",
        "NOTICE",
    }
    if not required <= set(files):
        raise ReleaseError(f"runtime release source is incomplete: {sorted(required - set(files))}")
    banned = ("optima/chain/", "optima/commit_reveal.py", "optima/economics.py", "optima/settlement.py", "optima/cli.py")
    if any(path.startswith(banned) for path in files):
        raise ReleaseError("runtime release contains subnet/control-plane code")
    return files


def _tar_gz(files: Mapping[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path, payload in sorted(files.items()):
            info = tarfile.TarInfo(f"optima-engine-{RUNTIME_VERSION}/{path}")
            info.size = len(payload)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0) as zipped:
        zipped.write(raw.getvalue())
    return out.getvalue()


def _zip_entry(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, _ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 << 16)
    archive.writestr(info, payload)


def _wheel(files: Mapping[str, bytes]) -> bytes:
    dist = f"optima_engine-{RUNTIME_VERSION}.dist-info"
    payloads = dict(files)
    payloads["optima_engine_bootstrap.pth"] = b"import optima.bootstrap\n"
    payloads[f"{dist}/METADATA"] = (
        "Metadata-Version: 2.1\nName: optima-engine\nVersion: " + RUNTIME_VERSION
        + "\nSummary: Chain-independent Optima inference runtime\n\n"
    ).encode("utf-8")
    payloads[f"{dist}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: optima-release-v1\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    payloads[f"{dist}/entry_points.txt"] = b"[sglang.srt.plugins]\noptima = optima.integrations.sglang_plugin:register\n"
    for legal in _RELEASE_LEGAL:
        payloads[f"{dist}/licenses/{legal}"] = files[legal]
    rows: list[tuple[str, str, str]] = []
    for name, payload in sorted(payloads.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
        rows.append((name, "sha256=" + encoded, str(len(payload))))
    record_name = f"{dist}/RECORD"
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows([*rows, (record_name, "", "")])
    payloads[record_name] = output.getvalue().encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(payloads.items()):
            _zip_entry(archive, name, payload)
    return buffer.getvalue()


def build_runtime_artifacts(source_root: str | Path) -> tuple[tuple[ReleaseArtifact, bytes], tuple[ReleaseArtifact, bytes]]:
    files = _runtime_files(source_root)
    source = _tar_gz(files)
    wheel = _wheel(files)
    return (
        (ReleaseArtifact.from_bytes(SOURCE_ARCHIVE, "application/gzip", source), source),
        (ReleaseArtifact.from_bytes(RUNTIME_WHEEL, "application/vnd.python.wheel", wheel), wheel),
    )


def require_reproducible_build(first: tuple[ReleaseArtifact, bytes], second: tuple[ReleaseArtifact, bytes]) -> ReleaseArtifact:
    if first[0] != second[0] or first[1] != second[1] or _sha(first[1]) != first[0].sha256:
        raise ReleaseError("double build did not reproduce byte-for-byte")
    return first[0]


def build_spdx_sbom(
    *, release_manifest: EngineReleaseManifest, engine_tree: MaterializedEngineTree,
    artifacts: Iterable[ReleaseArtifact], integrations: Iterable[IntegrationReviewRecord],
    native: NativeReleaseIdentity, model: ModelReleaseIdentity,
    upstream_repository: str, upstream_revision: str, sglang_version: str,
    base_image: str,
) -> bytes:
    tree = reopen_materialized_engine_tree(engine_tree.root, expected_tree_digest=engine_tree.tree_digest)
    if tree.stack_digest != release_manifest.digest:
        raise ReleaseError("SBOM engine tree differs from the release manifest")
    records = tuple(sorted(integrations, key=lambda row: row.target_id))
    release_manifest.validate_integrations({row.target_id: row for row in records})
    release_artifacts = tuple(sorted(artifacts, key=lambda row: row.name))
    if len({row.name for row in release_artifacts}) != len(release_artifacts):
        raise ReleaseError("SBOM artifact names are duplicated")
    document = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {"created": "1970-01-01T00:00:00Z", "creators": ["Tool: optima-release-v1"]},
        "dataLicense": "CC0-1.0",
        "documentNamespace": "urn:optima:engine-release:" + release_manifest.digest,
        "name": "Optima Engine " + release_manifest.digest[:12],
        "packages": [
            {
                "SPDXID": "SPDXRef-Engine",
                "checksums": [{"algorithm": "SHA256", "checksumValue": tree.tree_digest}],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "name": "optima-engine-tree",
                "versionInfo": release_manifest.digest,
            },
            {
                "SPDXID": "SPDXRef-BaseEngine",
                "checksums": [{"algorithm": "SHA256", "checksumValue": release_manifest.base_engine_digest}],
                "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
                "name": "validator-base-engine", "versionInfo": release_manifest.base_engine_digest,
            },
            {
                "SPDXID": "SPDXRef-Upstream",
                "downloadLocation": upstream_repository, "filesAnalyzed": False,
                "name": "sglang", "versionInfo": f"{sglang_version}+{upstream_revision}",
            },
            {
                "SPDXID": "SPDXRef-BaseImage",
                "checksums": [{"algorithm": "SHA256", "checksumValue": base_image.rsplit("sha256:", 1)[1]}],
                "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
                "name": "serving-base-image", "versionInfo": base_image,
            },
            {
                "SPDXID": "SPDXRef-Model",
                "checksums": [{"algorithm": "SHA256", "checksumValue": model.content_digest}],
                "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
                "name": model.model_id, "versionInfo": model.revision,
            },
            {
                "SPDXID": "SPDXRef-Native",
                "checksums": [{"algorithm": "SHA256", "checksumValue": native.publication_digest}],
                "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
                "name": "optima-native-artifact", "versionInfo": native.build_spec_digest,
            },
            *(
                {
                    "SPDXID": f"SPDXRef-Artifact-{index}",
                    "checksums": [{"algorithm": "SHA256", "checksumValue": artifact.sha256}],
                    "downloadLocation": "NOASSERTION",
                    "filesAnalyzed": False,
                    "name": artifact.name,
                    "versionInfo": str(artifact.size),
                }
                for index, artifact in enumerate(release_artifacts)
            ),
            *(
                {
                    "SPDXID": f"SPDXRef-Integration-{index}",
                    "checksums": [{"algorithm": "SHA256", "checksumValue": row.integrated_source_tree_digest}],
                    "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
                    "name": row.target_id, "versionInfo": row.digest,
                }
                for index, row in enumerate(records)
            ),
        ],
        "spdxVersion": "SPDX-2.3",
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"


def build_provenance(
    *, release_manifest: EngineReleaseManifest, engine_tree: MaterializedEngineTree,
    source: ReleaseArtifact, wheel: ReleaseArtifact, upstream_revision: str,
    upstream_repository: str, sglang_version: str, base_image: str,
    model: ModelReleaseIdentity, native: NativeReleaseIdentity,
    model_receipt: ReleaseArtifact, seccomp: ReleaseArtifact,
    reference_manifest: ReleaseArtifact, calibration_manifest: ReleaseArtifact,
    integrations: Iterable[IntegrationReviewRecord],
) -> bytes:
    tree = reopen_materialized_engine_tree(engine_tree.root, expected_tree_digest=engine_tree.tree_digest)
    records = tuple(sorted(integrations, key=lambda row: row.target_id))
    release_manifest.validate_integrations({row.target_id: row for row in records})
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://optima.engine/build/v1",
                "externalParameters": {
                    "engine_release_manifest": release_manifest.digest,
                    "base_engine_digest": release_manifest.base_engine_digest,
                    "base_image": base_image,
                    "model": model.to_dict(),
                    "native_build_spec": dict(native.build_spec),
                    "native_publication_digest": native.publication_digest,
                    "reference_manifest_digest": reference_manifest.sha256,
                    "calibration_manifest_digest": calibration_manifest.sha256,
                    "seccomp_digest": seccomp.sha256,
                    "upstream_repository": upstream_repository,
                    "upstream_revision": upstream_revision,
                    "sglang_version": sglang_version,
                },
                "internalParameters": {},
                "resolvedDependencies": [
                    {"digest": {"sha256": model_receipt.sha256}, "uri": "optima:model-provision-receipt"},
                    {"digest": {"sha256": native.publication_digest}, "uri": "optima:native-artifact"},
                    {"digest": {"sha256": seccomp.sha256}, "uri": "optima:seccomp-profile"},
                    {"digest": {"sha256": reference_manifest.sha256}, "uri": "optima:reference-manifest"},
                    {"digest": {"sha256": calibration_manifest.sha256}, "uri": "optima:calibration-manifest"},
                    *(
                        {"digest": {"sha256": row.integrated_source_tree_digest}, "uri": f"optima:integration:{row.target_id}"}
                        for row in records
                    ),
                ],
            },
            "runDetails": {"builder": {"id": "https://optima.engine/builder/v1"}, "metadata": {"invocationId": release_manifest.digest}},
        },
        "predicateType": "https://slsa.dev/provenance/v1",
        "subject": [
            {"digest": {"sha256": tree.tree_digest}, "name": "engine-tree"},
            {"digest": {"sha256": source.sha256}, "name": source.name},
            {"digest": {"sha256": wheel.sha256}, "name": wheel.name},
        ],
    }
    return json.dumps(statement, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"


@dataclass(frozen=True)
class PreparedRelease:
    descriptor: EngineReleaseDescriptor
    artifact_payloads: tuple[tuple[str, bytes], ...]
    native_publication: object

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, EngineReleaseDescriptor):
            raise ReleaseError("prepared release descriptor is untyped")
        names = tuple(name for name, _ in self.artifact_payloads)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ReleaseError("prepared release artifacts are not canonical")

    @property
    def payloads(self) -> dict[str, bytes]:
        return dict(self.artifact_payloads)


def prepare_release(
    *,
    source_root: str | Path,
    release_manifest: EngineReleaseManifest,
    engine_tree: MaterializedEngineTree,
    integrations: Iterable[IntegrationReviewRecord],
    model_id: str,
    model_revision: str,
    model_manifest_digest: str,
    model_provision: object,
    native_build_spec: object,
    native_publication: object,
    seccomp_payload: bytes,
    reference_manifest_payload: bytes,
    calibration_manifest_payload: bytes,
    upstream_repository: str,
    upstream_revision: str,
    sglang_version: str,
    serve: ServeSpec,
) -> PreparedRelease:
    """Construct all canonical release evidence from reopened typed inputs."""

    from optima.eval.engine_launch import NativeBuildSpec
    from optima.eval.native_artifact import NativeArtifactPublication, reopen_native_artifact
    from optima.model_provision import ProvisionedModel

    if not isinstance(model_provision, ProvisionedModel):
        raise ReleaseError("release preparation requires a provisioned model receipt")
    if not isinstance(native_build_spec, NativeBuildSpec) or not isinstance(
        native_publication, NativeArtifactPublication
    ):
        raise ReleaseError("release preparation requires typed native build outputs")
    reopened_native = reopen_native_artifact(
        native_publication.root,
        expected_build_spec_digest=native_build_spec.digest,
        expected_publication_digest=native_publication.publication_digest,
    )
    native = NativeReleaseIdentity.from_publication(native_build_spec, reopened_native)
    _reopen_reference_manifest(reference_manifest_payload)
    _reopen_calibration_manifest(calibration_manifest_payload)
    if seccomp_payload != _reviewed_seccomp_bytes():
        raise ReleaseError("seccomp bytes differ from the reviewed runtime profile")
    first_source, first_wheel = build_runtime_artifacts(source_root)
    second_source, second_wheel = build_runtime_artifacts(source_root)
    source = require_reproducible_build(first_source, second_source)
    wheel = require_reproducible_build(first_wheel, second_wheel)
    model = ModelReleaseIdentity.from_provisioning(
        model_id, model_revision, model_manifest_digest, model_provision.receipt
    )
    model_receipt_payload = model_provision.receipt.canonical_bytes
    inputs = {
        "runtime_source": (source, first_source[1]),
        "runtime_wheel": (wheel, first_wheel[1]),
        "model_receipt": (
            ReleaseArtifact.from_bytes(*_ARTIFACT_ROLES["model_receipt"], model_receipt_payload),
            model_receipt_payload,
        ),
        "seccomp": (
            ReleaseArtifact.from_bytes(*_ARTIFACT_ROLES["seccomp"], seccomp_payload),
            seccomp_payload,
        ),
        "reference_manifest": (
            ReleaseArtifact.from_bytes(
                *_ARTIFACT_ROLES["reference_manifest"], reference_manifest_payload
            ),
            reference_manifest_payload,
        ),
        "calibration_manifest": (
            ReleaseArtifact.from_bytes(
                *_ARTIFACT_ROLES["calibration_manifest"], calibration_manifest_payload
            ),
            calibration_manifest_payload,
        ),
    }
    records = tuple(sorted(integrations, key=lambda row: row.target_id))
    sbom_payload = build_spdx_sbom(
        release_manifest=release_manifest,
        engine_tree=engine_tree,
        artifacts=tuple(row[0] for row in inputs.values()),
        integrations=records,
        native=native,
        model=model,
        upstream_repository=upstream_repository,
        upstream_revision=upstream_revision,
        sglang_version=sglang_version,
        base_image=serve.base_image,
    )
    sbom = ReleaseArtifact.from_bytes(*_ARTIFACT_ROLES["sbom"], sbom_payload)
    provenance_payload = build_provenance(
        release_manifest=release_manifest,
        engine_tree=engine_tree,
        source=source,
        wheel=wheel,
        upstream_repository=upstream_repository,
        upstream_revision=upstream_revision,
        sglang_version=sglang_version,
        base_image=serve.base_image,
        model=model,
        native=native,
        model_receipt=inputs["model_receipt"][0],
        seccomp=inputs["seccomp"][0],
        reference_manifest=inputs["reference_manifest"][0],
        calibration_manifest=inputs["calibration_manifest"][0],
        integrations=records,
    )
    provenance = ReleaseArtifact.from_bytes(
        *_ARTIFACT_ROLES["provenance"], provenance_payload
    )
    descriptor = EngineReleaseDescriptor(
        release_manifest=release_manifest,
        engine_tree_digest=engine_tree.tree_digest,
        runtime_source=source,
        runtime_wheel=wheel,
        model_receipt=inputs["model_receipt"][0],
        seccomp=inputs["seccomp"][0],
        reference_manifest=inputs["reference_manifest"][0],
        calibration_manifest=inputs["calibration_manifest"][0],
        sbom=sbom,
        provenance=provenance,
        upstream_repository=upstream_repository,
        upstream_revision=upstream_revision,
        sglang_version=sglang_version,
        model=model,
        native=native,
        integration_records=records,
        serve=serve,
    )
    payloads = {
        **{artifact.name: payload for artifact, payload in inputs.values()},
        sbom.name: sbom_payload,
        provenance.name: provenance_payload,
    }
    return PreparedRelease(
        descriptor,
        tuple(sorted(payloads.items())),
        reopened_native,
    )


@dataclass(frozen=True)
class ContainerReproducibility:
    descriptor_digest: str
    first_oci_digest: str
    second_oci_digest: str
    oci_platform: str

    def __post_init__(self) -> None:
        for field in ("descriptor_digest", "first_oci_digest", "second_oci_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if self.first_oci_digest != self.second_oci_digest:
            raise ReleaseError("container double build did not reproduce")
        if not isinstance(self.oci_platform, str) or re.fullmatch(r"[a-z0-9._-]+/[a-z0-9._-]+", self.oci_platform) is None:
            raise ReleaseError("container platform is malformed")

    @property
    def digest(self) -> str:
        return canonical_digest("optima.engine-release.container-reproducibility", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "ContainerReproducibility":
        return cls(**_strict(value, set(cls.__dataclass_fields__), "container reproducibility"))


@dataclass(frozen=True)
class SignedContainerReproducibility:
    attestation: ContainerReproducibility
    signature: ReleaseSignature

    def __post_init__(self) -> None:
        if not isinstance(self.attestation, ContainerReproducibility) or not isinstance(
            self.signature, ReleaseSignature
        ):
            raise ReleaseError("container reproducibility signature is untyped")
        if self.signature.descriptor_digest != self.attestation.digest:
            raise ReleaseError("container reproducibility signature names another attestation")

    def to_dict(self) -> dict[str, object]:
        return {
            "attestation": self.attestation.to_dict(),
            "signature": self.signature.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "SignedContainerReproducibility":
        row = _strict(value, {"attestation", "signature"}, "signed container reproducibility")
        return cls(
            ContainerReproducibility.from_dict(row["attestation"]),
            ReleaseSignature.from_dict(row["signature"]),
        )


def sign_container_reproducibility(
    attestation: ContainerReproducibility, private_key: bytes
) -> SignedContainerReproducibility:
    if not isinstance(attestation, ContainerReproducibility):
        raise ReleaseError("container attestation is untyped")
    # The signature envelope deliberately reuses the audited Ed25519 wire format;
    # its digest field names the attestation rather than an engine descriptor.
    signature = _sign_digest(attestation.digest, private_key)
    return SignedContainerReproducibility(attestation, signature)


def verify_container_reproducibility(
    signed: SignedContainerReproducibility, *, expected_public_key: bytes | str
) -> None:
    if not isinstance(signed, SignedContainerReproducibility):
        raise ReleaseError("container attestation is untyped")
    trusted = _expected_public_key(expected_public_key)
    if signed.signature.public_key != trusted:
        raise ReleaseError("container attestation is not from the trusted release authority")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted)).verify(
            bytes.fromhex(signed.signature.signature),
            bytes.fromhex(signed.attestation.digest),
        )
    except (ImportError, InvalidSignature, ValueError):
        raise ReleaseError("container reproducibility signature verification failed") from None


@dataclass(frozen=True)
class PublishedRelease:
    root: Path
    descriptor: EngineReleaseDescriptor
    signature: ReleaseSignature
    native_publication: object
    release_tree_digest: str


@dataclass(frozen=True)
class ServeReceiptVerification:
    descriptor_digest: str
    expected_slots: tuple[str, ...]
    expected_ranks: int
    active_count: int
    routed_count: int
    completed_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptor_digest", _digest(self.descriptor_digest, "descriptor digest"))
        if (
            not self.expected_slots
            or self.expected_slots != tuple(sorted(set(self.expected_slots)))
            or type(self.expected_ranks) is not int
            or self.expected_ranks < 1
            or any(type(value) is not int or value < 1 for value in (
                self.active_count, self.routed_count, self.completed_count
            ))
        ):
            raise ReleaseError("serve receipt verification is malformed")

    @property
    def digest(self) -> str:
        return canonical_digest("optima.engine-release.serve-receipts", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            field: list(getattr(self, field)) if field == "expected_slots" else getattr(self, field)
            for field in self.__dataclass_fields__
        }


def verify_serve_receipts(
    release: PublishedRelease, receipt_root: str | Path
) -> ServeReceiptVerification:
    """Require active, routed, and completed evidence for every release slot/rank."""

    from optima import receipts
    from optima.manifest import load_manifest

    if not isinstance(release, PublishedRelease):
        raise ReleaseError("serve receipt verification requires a reopened release")
    root = Path(receipt_root)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ReleaseError(f"serve receipt directory is unavailable: {exc}") from None
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ReleaseError("serve receipt root must be a concrete directory")
    entries = tuple(root.iterdir())
    if len(entries) > 4_096:
        raise ReleaseError("serve receipt directory exceeds its file bound")
    for entry in entries:
        info = entry.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > (1 << 20)
            or entry.suffix != ".json"
        ):
            raise ReleaseError("serve receipt directory contains an unsafe entry")
    manifest = load_manifest(release.root / "engine-tree")
    slots = tuple(sorted({row.slot for row in manifest.ops}))
    if not slots:
        raise ReleaseError("release has no runtime slots to receipt")
    try:
        active = receipts.require(root, "active", context="signed release serve smoke")
        routed = receipts.require(root, "fired", context="signed release serve smoke")
        completed = receipts.require(root, "completed", context="signed release serve smoke")
        load_failed = receipts.collect(root, "load_failed")
        fallbacks = receipts.collect(root, "fallback")
    except (RuntimeError, receipts.ReceiptFormatError) as exc:
        raise ReleaseError(str(exc)) from None
    if load_failed or fallbacks:
        raise ReleaseError("signed release serve smoke recorded load failure or fallback")
    for row in active:
        declared = row.get("slots")
        if not isinstance(declared, list) or not set(slots) <= set(declared):
            raise ReleaseError("active receipt does not name every expected release slot")
    routed_detail = receipts.coverage_matrix(
        routed,
        expected_slots=slots,
        member_receipts=active,
        expected_member_count=release.descriptor.serve.tp_size,
    )
    completed_ok, completed_detail = receipts.completed_gate(
        completed,
        expected_slots=slots,
        member_receipts=active,
        expected_member_count=release.descriptor.serve.tp_size,
        fallback_receipts=fallbacks,
    )
    if not routed_detail["ok"]:
        raise ReleaseError(f"signed release routing coverage failed: {routed_detail}")
    if not completed_ok:
        raise ReleaseError(f"signed release completion coverage failed: {completed_detail}")
    return ServeReceiptVerification(
        release.descriptor.digest,
        slots,
        release.descriptor.serve.tp_size,
        len(active),
        len(routed),
        len(completed),
    )


def _descriptor_artifacts(descriptor: EngineReleaseDescriptor) -> tuple[ReleaseArtifact, ...]:
    return tuple(getattr(descriptor, role) for role in _ARTIFACT_ROLES)


def _decode_unique_json(payload: bytes, *, label: str) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=unique)
    except ReleaseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"{label} is not valid JSON: {exc}") from None


def _reopen_reference_manifest(payload: bytes) -> object:
    from optima.eval.qualification import ReferenceManifest

    try:
        reference = ReferenceManifest.from_dict(
            _decode_unique_json(payload, label="reference manifest")
        )
    except Exception as exc:
        raise ReleaseError(f"reference manifest is invalid: {exc}") from None
    if payload != canonical_json_bytes(reference.to_dict()) + b"\n":
        raise ReleaseError("reference manifest bytes are not canonical")
    return reference


def _reopen_calibration_manifest(payload: bytes) -> object:
    from optima.eval.calibration import CalibrationManifest

    try:
        calibration = CalibrationManifest.from_dict(
            _decode_unique_json(payload, label="calibration manifest")
        )
    except Exception as exc:
        raise ReleaseError(f"calibration manifest is invalid: {exc}") from None
    if not calibration.thresholds_frozen:
        raise ReleaseError("release calibration thresholds are not frozen")
    if payload != canonical_json_bytes(calibration.to_dict()) + b"\n":
        raise ReleaseError("calibration manifest bytes are not canonical")
    return calibration


def _reviewed_seccomp_bytes() -> bytes:
    return _stable_regular(Path(__file__).parent / "eval" / "seccomp_moby_v0_2_1.json")


def _recompute_release_evidence(
    descriptor: EngineReleaseDescriptor,
    tree: MaterializedEngineTree,
    payloads: Mapping[str, bytes],
) -> None:
    from optima.model_provision import ModelProvisionReceipt

    model_value = _decode_unique_json(
        payloads[descriptor.model_receipt.name], label="model provision receipt"
    )
    try:
        model_receipt = ModelProvisionReceipt.from_dict(model_value)
    except Exception as exc:
        raise ReleaseError(f"model provision receipt is invalid: {exc}") from None
    if (
        model_receipt.canonical_bytes != payloads[descriptor.model_receipt.name]
        or model_receipt.receipt_digest != descriptor.model.receipt_digest
        or model_receipt.content_digest != descriptor.model.content_digest
    ):
        raise ReleaseError("model provision receipt differs from the release model identity")
    reference = _reopen_reference_manifest(payloads[descriptor.reference_manifest.name])
    calibration = _reopen_calibration_manifest(payloads[descriptor.calibration_manifest.name])
    reference_expected = {
        "runtime_digest": descriptor.release_manifest.runtime_digest,
        "base_engine_digest": descriptor.release_manifest.base_engine_digest,
        "catalog_digest": descriptor.release_manifest.catalog_digest,
        "model_manifest_digest": descriptor.model.manifest_digest,
        "model_content_digest": descriptor.model.content_digest,
    }
    if any(getattr(reference, field) != expected for field, expected in reference_expected.items()):
        raise ReleaseError("reference manifest differs from the release runtime/model authority")
    calibration_expected = {
        "reference_manifest_digest": reference.digest,
        "arena_digest": reference.arena_digest,
        "runtime_digest": reference.runtime_digest,
        "base_engine_digest": reference.base_engine_digest,
        "model_revision_digest": reference.model_revision_digest,
        "model_manifest_digest": reference.model_manifest_digest,
        "model_content_digest": reference.model_content_digest,
        "logical_hardware_digest": reference.logical_hardware_digest,
        "workload_digest": reference.workload_digest,
        "controller_distribution_digest": reference.controller_distribution_digest,
    }
    if any(
        getattr(calibration.context, field) != expected
        for field, expected in calibration_expected.items()
    ):
        raise ReleaseError("calibration context differs from the retained reference authority")
    seccomp_value = _decode_unique_json(
        payloads[descriptor.seccomp.name], label="seccomp"
    )
    if not isinstance(seccomp_value, dict) or not isinstance(
        seccomp_value.get("defaultAction"), str
    ):
        raise ReleaseError("seccomp profile lacks a defaultAction")
    if payloads[descriptor.seccomp.name] != _reviewed_seccomp_bytes():
        raise ReleaseError("seccomp bytes differ from the reviewed runtime profile")
    evidence_artifacts = (
        descriptor.runtime_source, descriptor.runtime_wheel, descriptor.model_receipt,
        descriptor.seccomp, descriptor.reference_manifest, descriptor.calibration_manifest,
    )
    expected_sbom = build_spdx_sbom(
        release_manifest=descriptor.release_manifest,
        engine_tree=tree,
        artifacts=evidence_artifacts,
        integrations=descriptor.integration_records,
        native=descriptor.native,
        model=descriptor.model,
        upstream_repository=descriptor.upstream_repository,
        upstream_revision=descriptor.upstream_revision,
        sglang_version=descriptor.sglang_version,
        base_image=descriptor.serve.base_image,
    )
    expected_provenance = build_provenance(
        release_manifest=descriptor.release_manifest,
        engine_tree=tree,
        source=descriptor.runtime_source,
        wheel=descriptor.runtime_wheel,
        upstream_repository=descriptor.upstream_repository,
        upstream_revision=descriptor.upstream_revision,
        sglang_version=descriptor.sglang_version,
        base_image=descriptor.serve.base_image,
        model=descriptor.model,
        native=descriptor.native,
        model_receipt=descriptor.model_receipt,
        seccomp=descriptor.seccomp,
        reference_manifest=descriptor.reference_manifest,
        calibration_manifest=descriptor.calibration_manifest,
        integrations=descriptor.integration_records,
    )
    if payloads[descriptor.sbom.name] != expected_sbom:
        raise ReleaseError("SBOM is not the canonical document derived from reopened inputs")
    if payloads[descriptor.provenance.name] != expected_provenance:
        raise ReleaseError("provenance is not the canonical statement derived from reopened inputs")


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        relative = path.relative_to(root)
        directory_mode = 0o755 if "engine-tree" in relative.parts else _DIR_MODE
        os.chmod(path, directory_mode if path.is_dir() else _FILE_MODE)
        os.utime(path, (0, 0), follow_symlinks=False)
    os.chmod(root, _DIR_MODE)
    os.utime(root, (0, 0), follow_symlinks=False)


def _release_tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ReleaseError(f"release contains a special object: {relative}")
        if path.is_file():
            data = _stable_regular(path)
            rows.append({"path": relative, "sha256": _sha(data), "size": len(data)})
    return canonical_digest("optima.engine-release.tree", {"files": rows})


def publish_release(
    publication_root: str | Path, descriptor: EngineReleaseDescriptor,
    signature: ReleaseSignature, engine_tree: MaterializedEngineTree,
    artifact_payloads: Mapping[str, bytes], native_publication: object,
    *, expected_public_key: bytes | str,
) -> PublishedRelease:
    from optima.eval.engine_launch import NativeBuildSpec
    from optima.eval.native_artifact import reopen_native_artifact

    verify_release_signature(
        descriptor, signature, expected_public_key=expected_public_key
    )
    tree = reopen_materialized_engine_tree(engine_tree.root, expected_tree_digest=descriptor.engine_tree_digest)
    if tree.stack_digest != descriptor.release_manifest.digest:
        raise ReleaseError("release engine tree names another manifest")
    expected = {row.name: row for row in _descriptor_artifacts(descriptor)}
    if set(artifact_payloads) != set(expected):
        raise ReleaseError("release artifact payload coverage differs")
    for name, artifact in expected.items():
        payload = artifact_payloads[name]
        if len(payload) != artifact.size or _sha(payload) != artifact.sha256:
            raise ReleaseError(f"release artifact payload differs: {name}")
    reopened_native = reopen_native_artifact(
        getattr(native_publication, "root", ""),
        expected_build_spec_digest=descriptor.native.build_spec_digest,
        expected_publication_digest=descriptor.native.publication_digest,
    )
    if NativeReleaseIdentity.from_publication(
        # Parsing the descriptor's complete spec prevents a caller from pairing the
        # publication with an opaque digest-only build identity.
        NativeBuildSpec.from_dict(dict(descriptor.native.build_spec)),
        reopened_native,
    ) != descriptor.native:
        raise ReleaseError("reopened native publication differs from the release identity")
    _recompute_release_evidence(descriptor, tree, artifact_payloads)
    parent = Path(publication_root).resolve()
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = parent / descriptor.digest
    if destination.exists():
        return reopen_release(
            destination,
            expected_descriptor_digest=descriptor.digest,
            expected_public_key=expected_public_key,
        )
    stage = Path(tempfile.mkdtemp(prefix=".release-", dir=parent))
    try:
        (stage / "artifacts").mkdir()
        (stage / "engine-tree").mkdir()
        native_destination = (
            stage / "native-artifacts" / descriptor.native.build_spec_digest[:2]
            / descriptor.native.build_spec_digest
        )
        native_destination.parent.mkdir(parents=True)
        native_destination.mkdir()
        (stage / "release.json").write_bytes(descriptor.canonical_bytes + b"\n")
        (stage / "release.sig.json").write_bytes(canonical_json_bytes(signature.to_dict()) + b"\n")
        for name, payload in artifact_payloads.items():
            (stage / "artifacts" / name).write_bytes(payload)
        for source in sorted(tree.root.rglob("*")):
            relative = source.relative_to(tree.root)
            target = stage / "engine-tree" / relative
            if source.is_dir():
                target.mkdir()
            else:
                target.write_bytes(_stable_regular(source))
        for source in sorted(reopened_native.root.rglob("*")):
            relative = source.relative_to(reopened_native.root)
            target = native_destination / relative
            if source.is_dir():
                target.mkdir()
            else:
                target.write_bytes(_stable_regular(source))
        _freeze_tree(stage)
        os.replace(stage, destination)
    except BaseException:
        for path in sorted(stage.rglob("*"), reverse=True):
            try:
                os.chmod(path, 0o700 if path.is_dir() else 0o600)
            except OSError:
                pass
        try:
            os.chmod(stage, 0o700)
        except OSError:
            pass
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return reopen_release(
        destination,
        expected_descriptor_digest=descriptor.digest,
        expected_public_key=expected_public_key,
    )


def reopen_release(
    root: str | Path, *, expected_public_key: bytes | str,
    expected_descriptor_digest: str | None = None,
) -> PublishedRelease:
    requested = Path(root)
    if requested.is_symlink():
        raise ReleaseError("published release root must not be a symlink")
    path = requested.resolve(strict=True)
    if stat.S_IMODE(path.stat().st_mode) != _DIR_MODE:
        raise ReleaseError("published release root is not read-only")
    try:
        descriptor_raw = _stable_regular(path / "release.json")
        signature_raw = _stable_regular(path / "release.sig.json")
        descriptor = EngineReleaseDescriptor.from_dict(
            _decode_unique_json(descriptor_raw, label="release descriptor")
        )
        signature = ReleaseSignature.from_dict(
            _decode_unique_json(signature_raw, label="release signature")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReleaseError(f"published release metadata is invalid: {exc}") from None
    if (
        descriptor_raw != descriptor.canonical_bytes + b"\n"
        or signature_raw != canonical_json_bytes(signature.to_dict()) + b"\n"
    ):
        raise ReleaseError("published release metadata is not canonical")
    if expected_descriptor_digest is not None and descriptor.digest != _digest(expected_descriptor_digest, "expected descriptor"):
        raise ReleaseError("published release descriptor digest differs")
    verify_release_signature(
        descriptor, signature, expected_public_key=expected_public_key
    )
    tree = reopen_materialized_engine_tree(path / "engine-tree", expected_tree_digest=descriptor.engine_tree_digest)
    if tree.stack_digest != descriptor.release_manifest.digest:
        raise ReleaseError("published release engine tree differs")
    expected = {row.name: row for row in _descriptor_artifacts(descriptor)}
    artifact_root = path / "artifacts"
    top_level = {item.name for item in path.iterdir()}
    if top_level != {
        "artifacts", "engine-tree", "native-artifacts", "release.json", "release.sig.json"
    }:
        raise ReleaseError("published release top-level inventory differs")
    artifact_items = tuple(artifact_root.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in artifact_items):
        raise ReleaseError("published release artifacts contain a non-file")
    actual = {item.name for item in artifact_items}
    if actual != set(expected):
        raise ReleaseError("published release artifact inventory differs")
    for name, artifact in expected.items():
        payload = _stable_regular(artifact_root / name)
        if len(payload) != artifact.size or _sha(payload) != artifact.sha256:
            raise ReleaseError(f"published release artifact changed: {name}")
    from optima.eval.native_artifact import reopen_native_artifact
    native_path = (
        path / "native-artifacts" / descriptor.native.build_spec_digest[:2]
        / descriptor.native.build_spec_digest
    )
    native = reopen_native_artifact(
        native_path,
        expected_build_spec_digest=descriptor.native.build_spec_digest,
        expected_publication_digest=descriptor.native.publication_digest,
    )
    native_root_items = tuple((path / "native-artifacts").iterdir())
    shard_items = tuple((path / "native-artifacts" / descriptor.native.build_spec_digest[:2]).iterdir())
    if (
        {item.name for item in native_root_items} != {descriptor.native.build_spec_digest[:2]}
        or {item.name for item in shard_items} != {descriptor.native.build_spec_digest}
    ):
        raise ReleaseError("published native artifact address space contains extra entries")
    _recompute_release_evidence(
        descriptor,
        tree,
        {name: _stable_regular(artifact_root / name) for name in expected},
    )
    for item in path.rglob("*"):
        relative = item.relative_to(path)
        directory_mode = 0o755 if "engine-tree" in relative.parts else _DIR_MODE
        if item.is_symlink() or stat.S_IMODE(item.stat().st_mode) != (directory_mode if item.is_dir() else _FILE_MODE):
            raise ReleaseError("published release mode or link policy differs")
    return PublishedRelease(path, descriptor, signature, native, _release_tree_digest(path))


def _container_deployment(
    release: PublishedRelease, trusted_key: str, overlay_digest: str
) -> dict[str, object]:
    return {
        "descriptor_digest": release.descriptor.digest,
        "required_read_only_rootfs": True,
        "required_writable_tmpfs": "/tmp",
        "serve_receipt_directory": "/tmp/optima-release-receipts",
        "required_seccomp_profile": "seccomp.json",
        "runtime_overlay_digest": overlay_digest,
        "seccomp_sha256": release.descriptor.seccomp.sha256,
        "trusted_release_public_key": trusted_key,
    }


def _container_dockerfile(release: PublishedRelease, overlay_digest: str) -> bytes:
    descriptor = release.descriptor
    return (
        f"FROM {descriptor.serve.base_image}\n"
        "COPY release/artifacts/" + RUNTIME_WHEEL + " /tmp/optima-engine.whl\n"
        "RUN /usr/bin/python3 -m pip install --no-deps /tmp/optima-engine.whl && rm /tmp/optima-engine.whl\n"
        + "RUN /usr/bin/python3 -m optima.release_runtime install-reviewed-overlays"
        + " --expected-sglang-version " + shlex.quote(descriptor.sglang_version)
        + " --expected-upstream-revision " + shlex.quote(descriptor.upstream_revision)
        + " --expected-overlay-digest " + overlay_digest + "\n"
        "COPY release /optima\n"
        "COPY trusted-release-key /etc/optima/trusted-release-key\n"
        "COPY seccomp.json /etc/optima/seccomp.json\n"
        + "".join(
            f"ENV {key}={json.dumps(value, ensure_ascii=True)}\n"
            for key, value in descriptor.serve.environment
        )
        + "LABEL org.optima.release.descriptor=\"" + descriptor.digest + "\" "
        + "org.optima.seccomp.sha256=\"" + descriptor.seccomp.sha256 + "\" "
        + "org.optima.runtime-overlays=\"" + overlay_digest + "\"\n"
        + "ENTRYPOINT [\"/usr/bin/python3\",\"-m\",\"optima.release_runtime\","
        + "\"--release-root\",\"/optima\","
        + "\"--expected-public-key-file\",\"/etc/optima/trusted-release-key\","
        + "\"--model-root\"," + json.dumps(descriptor.serve.model_mount) + ","
        + "\"--require-seccomp\",\"--\",\"/usr/bin/python3\",\"-m\",\"sglang.launch_server\"]\n"
        + "CMD "
        + json.dumps(list(descriptor.serve.command_arguments), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def container_context(
    release: PublishedRelease, destination: str | Path, *,
    expected_public_key: bytes | str,
) -> Path:
    """Create one deterministic BuildKit context; no chain or credentials enter it."""

    from optima.release_runtime import reviewed_runtime_overlay_digest

    trusted_key = _expected_public_key(expected_public_key)
    reopened = reopen_release(
        release.root,
        expected_descriptor_digest=release.descriptor.digest,
        expected_public_key=trusted_key,
    )
    dest = Path(destination)
    if dest.exists() or dest.is_symlink():
        raise ReleaseError("container context destination already exists")
    dest.mkdir(parents=True)
    shutil.copytree(reopened.root, dest / "release", copy_function=shutil.copyfile)
    # This is the Ed25519 public verification key; private signing material never
    # enters the container context.
    # codeql[py/clear-text-storage-sensitive-data]
    (dest / "trusted-release-key").write_text(trusted_key + "\n", encoding="ascii")
    shutil.copyfile(
        reopened.root / "artifacts" / reopened.descriptor.seccomp.name,
        dest / "seccomp.json",
    )
    overlay_digest = reviewed_runtime_overlay_digest(
        expected_sglang_version=reopened.descriptor.sglang_version,
        expected_upstream_revision=reopened.descriptor.upstream_revision,
    )
    (dest / "deployment.json").write_bytes(
        # The deployment policy intentionally carries the same public verifier.
        # codeql[py/clear-text-storage-sensitive-data]
        canonical_json_bytes(
            _container_deployment(reopened, trusted_key, overlay_digest)
        )
        + b"\n"
    )
    (dest / "Dockerfile").write_bytes(
        _container_dockerfile(reopened, overlay_digest)
    )
    _freeze_tree(dest)
    return dest


__all__ = [
    "ContainerReproducibility", "EngineReleaseDescriptor", "ModelReleaseIdentity",
    "NativeReleaseFile", "NativeReleaseIdentity", "PreparedRelease",
    "PublishedRelease", "ReleaseArtifact", "ReleaseError", "ReleaseSignature",
    "ServeSpec", "SignedContainerReproducibility", "build_provenance",
    "build_runtime_artifacts", "build_spdx_sbom", "container_context",
    "prepare_release", "publish_release", "reopen_release",
    "require_reproducible_build", "sign_container_reproducibility", "sign_release",
    "verify_container_reproducibility", "verify_release_signature",
]
