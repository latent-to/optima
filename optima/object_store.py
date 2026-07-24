"""Provider-agnostic object storage for Optima control-plane artifacts.

Backends are swappable through :class:`ObjectStoreConfig.provider` (and optional
endpoint overrides). The S3-compatible path uses boto3 only when selected; the
core package does not hard-depend on it. Hippius is one preset endpoint/region
profile over the same S3-compatible client — not a fork of any provider SDK.

boto3/botocore are Apache-2.0. This module contains no GPL code and does not
vendor third-party sources; it only calls the public S3 API shape.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ObjectStoreError(RuntimeError):
    """Object-store configuration or I/O failed closed."""

    validator_fault = True
    retryable = False


class ObjectStoreRetryableError(ObjectStoreError):
    """Transient remote object-store failure."""

    retryable = True


class ObjectStore(Protocol):
    """Minimal byte-object API shared by every provider backend."""

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...


# Known S3-compatible presets. Operators swap providers by changing ``provider``
# (and credentials/endpoint when needed); call sites keep the same ObjectStore API.
_PROVIDER_DEFAULTS: dict[str, dict[str, str | None]] = {
    "hippius": {
        "endpoint_url": "https://s3.hippius.com",
        "region_name": "decentralized",
        "addressing_style": "path",
    },
    # Generic AWS S3 (or any path that accepts the AWS SDK defaults).
    "s3": {
        "endpoint_url": None,
        "region_name": "us-east-1",
        "addressing_style": "auto",
    },
    "minio": {
        "endpoint_url": "http://127.0.0.1:9000",
        "region_name": "us-east-1",
        "addressing_style": "path",
    },
    # Local filesystem rooted at ``root_dir`` (tests / air-gapped dry runs).
    "local": {
        "endpoint_url": None,
        "region_name": None,
        "addressing_style": None,
    },
    # In-process map (unit tests only).
    "memory": {
        "endpoint_url": None,
        "region_name": None,
        "addressing_style": None,
    },
}

KNOWN_OBJECT_STORE_PROVIDERS = frozenset(_PROVIDER_DEFAULTS)


@dataclass(frozen=True)
class ObjectStoreConfig:
    """Declarative object-store selection. Swap providers without code changes."""

    provider: str
    bucket: str = ""
    key_prefix: str = ""
    endpoint_url: str | None = None
    region_name: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    addressing_style: str | None = None
    root_dir: str | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower() if isinstance(self.provider, str) else ""
        if provider not in _PROVIDER_DEFAULTS:
            raise ObjectStoreError(
                "object-store provider must be one of: "
                + ", ".join(sorted(KNOWN_OBJECT_STORE_PROVIDERS))
            )
        object.__setattr__(self, "provider", provider)
        if provider in {"hippius", "s3", "minio"}:
            if not isinstance(self.bucket, str) or not self.bucket.strip():
                raise ObjectStoreError(f"{provider} object store requires a bucket")
            if self.bucket.strip() != self.bucket or "/" in self.bucket or ".." in self.bucket:
                raise ObjectStoreError("object-store bucket is malformed")
        if self.key_prefix is None or not isinstance(self.key_prefix, str):
            raise ObjectStoreError("object-store key_prefix is malformed")
        if any(part == ".." for part in self.key_prefix.split("/")):
            raise ObjectStoreError("object-store key_prefix must not contain '..'")
        if provider == "local":
            root = self.root_dir
            if not isinstance(root, str) or not root.strip():
                raise ObjectStoreError("local object store requires root_dir")

    def resolve_key(self, logical_key: str) -> str:
        key = _require_object_key(logical_key)
        prefix = self.key_prefix.strip("/")
        return f"{prefix}/{key}" if prefix else key

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str = "OPTIMA_OBJECT_STORE_",
        defaults: "ObjectStoreConfig | None" = None,
    ) -> "ObjectStoreConfig":
        """Build config from environment, overlaying optional CLI defaults."""

        base = defaults or cls(provider="hippius", bucket="")
        def _env(name: str) -> str | None:
            value = os.environ.get(prefix + name)
            if value is None or value == "":
                return None
            return value

        provider = (_env("PROVIDER") or base.provider).lower()
        return cls(
            provider=provider,
            bucket=_env("BUCKET") or base.bucket,
            key_prefix=_env("KEY_PREFIX") if _env("KEY_PREFIX") is not None else base.key_prefix,
            endpoint_url=_env("ENDPOINT_URL")
            if _env("ENDPOINT_URL") is not None
            else base.endpoint_url,
            region_name=_env("REGION") if _env("REGION") is not None else base.region_name,
            access_key_id=_env("ACCESS_KEY_ID")
            if _env("ACCESS_KEY_ID") is not None
            else base.access_key_id,
            secret_access_key=_env("SECRET_ACCESS_KEY")
            if _env("SECRET_ACCESS_KEY") is not None
            else base.secret_access_key,
            addressing_style=_env("ADDRESSING_STYLE")
            if _env("ADDRESSING_STYLE") is not None
            else base.addressing_style,
            root_dir=_env("ROOT_DIR") if _env("ROOT_DIR") is not None else base.root_dir,
        )


def _require_object_key(key: str) -> str:
    if (
        not isinstance(key, str)
        or not key
        or key.strip() != key
        or key.startswith("/")
        or ".." in key.split("/")
        or len(key) > 1024
    ):
        raise ObjectStoreError("object-store key is malformed")
    return key


@dataclass
class MemoryObjectStore:
    """In-process store for tests; not durable."""

    _objects: dict[str, tuple[bytes, str]] | None = None
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        if self._objects is None:
            self._objects = {}
        if self._lock is None:
            self._lock = threading.Lock()

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        key = _require_object_key(key)
        if not isinstance(data, (bytes, bytearray)):
            raise ObjectStoreError("object-store put requires bytes")
        assert self._lock is not None and self._objects is not None
        with self._lock:
            self._objects[key] = (bytes(data), content_type)

    def get_bytes(self, key: str) -> bytes:
        key = _require_object_key(key)
        assert self._lock is not None and self._objects is not None
        with self._lock:
            row = self._objects.get(key)
        if row is None:
            raise ObjectStoreError(f"object-store key is missing: {key}")
        return row[0]


@dataclass(frozen=True)
class LocalDirectoryObjectStore:
    """Filesystem-backed store rooted at ``root_dir`` (provider=local)."""

    root_dir: Path

    def __post_init__(self) -> None:
        root = Path(self.root_dir)
        if root.exists() and not root.is_dir():
            raise ObjectStoreError("local object-store root_dir is not a directory")
        object.__setattr__(self, "root_dir", root)

    def _path_for(self, key: str) -> Path:
        key = _require_object_key(key)
        path = (self.root_dir / key).resolve()
        root = self.root_dir.resolve()
        if root not in path.parents and path != root:
            raise ObjectStoreError("local object-store key escapes root_dir")
        return path

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        del content_type  # filesystem backend does not retain content-type
        if not isinstance(data, (bytes, bytearray)):
            raise ObjectStoreError("object-store put requires bytes")
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(bytes(data))
        os.replace(tmp, path)

    def get_bytes(self, key: str) -> bytes:
        path = self._path_for(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ObjectStoreError(f"object-store key is missing: {key}") from exc


@dataclass(frozen=True)
class S3CompatibleObjectStore:
    """Any S3-compatible endpoint via boto3 (Hippius, AWS, MinIO, custom)."""

    client: object
    bucket: str

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        key = _require_object_key(key)
        if not isinstance(data, (bytes, bytearray)):
            raise ObjectStoreError("object-store put requires bytes")
        try:
            self.client.put_object(  # type: ignore[attr-defined]
                Bucket=self.bucket,
                Key=key,
                Body=bytes(data),
                ContentType=content_type,
            )
        except Exception as exc:
            name = type(exc).__name__
            if name in {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError"}:
                raise ObjectStoreRetryableError(
                    f"s3-compatible put failed transiently: {exc}"
                ) from None
            raise ObjectStoreError(f"s3-compatible put failed: {exc}") from None

    def get_bytes(self, key: str) -> bytes:
        key = _require_object_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)  # type: ignore[attr-defined]
            body = response["Body"].read()
        except Exception as exc:
            name = type(exc).__name__
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404", "NotFound"} or name == "NoSuchKey":
                raise ObjectStoreError(f"object-store key is missing: {key}") from None
            if name in {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError"}:
                raise ObjectStoreRetryableError(
                    f"s3-compatible get failed transiently: {exc}"
                ) from None
            raise ObjectStoreError(f"s3-compatible get failed: {exc}") from None
        if not isinstance(body, (bytes, bytearray)):
            raise ObjectStoreError("s3-compatible get returned non-bytes body")
        return bytes(body)


def _boto3_client(config: ObjectStoreConfig):
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:
        raise ObjectStoreError(
            "s3-compatible object stores require boto3; "
            'install with pip install -e ".[object-store]"'
        ) from exc
    defaults = _PROVIDER_DEFAULTS[config.provider]
    endpoint = (
        config.endpoint_url
        if config.endpoint_url is not None
        else defaults["endpoint_url"]
    )
    region = (
        config.region_name
        if config.region_name is not None
        else defaults["region_name"]
    )
    addressing = (
        config.addressing_style
        if config.addressing_style is not None
        else defaults["addressing_style"]
    )
    access_key = config.access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = config.secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise ObjectStoreError(
            f"{config.provider} object store requires access_key_id and secret_access_key"
        )
    boto_kwargs: dict[str, object] = {
        "service_name": "s3",
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": region or "us-east-1",
    }
    if endpoint:
        boto_kwargs["endpoint_url"] = endpoint
    if addressing and addressing != "auto":
        boto_kwargs["config"] = BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": addressing},
        )
    else:
        boto_kwargs["config"] = BotoConfig(signature_version="s3v4")
    return boto3.client(**boto_kwargs)


def open_object_store(config: ObjectStoreConfig) -> ObjectStore:
    """Construct the backend selected by ``config.provider``."""

    if type(config) is not ObjectStoreConfig:
        raise ObjectStoreError("object-store config is untyped")
    if config.provider == "memory":
        return MemoryObjectStore()
    if config.provider == "local":
        assert config.root_dir is not None
        return LocalDirectoryObjectStore(Path(config.root_dir))
    if config.provider in {"hippius", "s3", "minio"}:
        return S3CompatibleObjectStore(_boto3_client(config), config.bucket)
    raise ObjectStoreError(f"unsupported object-store provider: {config.provider}")


def prefixed_store(store: ObjectStore, prefix: str) -> ObjectStore:
    """Return a view that prefixes every key (used with ObjectStoreConfig.key_prefix)."""

    prefix = prefix.strip("/")
    if not prefix:
        return store
    if ".." in prefix.split("/"):
        raise ObjectStoreError("object-store key prefix is malformed")

    class _Prefixed:
        def put_bytes(
            self,
            key: str,
            data: bytes,
            *,
            content_type: str = "application/octet-stream",
        ) -> None:
            store.put_bytes(
                f"{prefix}/{_require_object_key(key)}",
                data,
                content_type=content_type,
            )

        def get_bytes(self, key: str) -> bytes:
            return store.get_bytes(f"{prefix}/{_require_object_key(key)}")

    return _Prefixed()


def open_configured_object_store(config: ObjectStoreConfig) -> ObjectStore:
    """Open the provider backend and apply ``key_prefix`` if set."""

    return prefixed_store(open_object_store(config), config.key_prefix)


__all__ = [
    "KNOWN_OBJECT_STORE_PROVIDERS",
    "LocalDirectoryObjectStore",
    "MemoryObjectStore",
    "ObjectStore",
    "ObjectStoreConfig",
    "ObjectStoreError",
    "ObjectStoreRetryableError",
    "S3CompatibleObjectStore",
    "open_configured_object_store",
    "open_object_store",
    "prefixed_store",
]
