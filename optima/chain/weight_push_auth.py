"""Rotatable shared-secret credentials for eval → weights-service push.

The eval host must not hold the chain-signing wallet path for weight
publication. It authenticates to ``serve-weights`` with HMAC credentials that
operators can rotate by adding a new active secret and retiring the old id.

This module is original Optima code (Apache-2.0). It does not vendor third-party
auth libraries.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

from optima.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


CREDENTIALS_SCHEMA = "optima.weight-push-credentials.v1"
PUSH_AUTH_DOMAIN = "optima.weight-share.push.v1"
DEFAULT_MAX_SKEW_SECONDS = 60
# File path to a PushCredentialSet JSON (serve + eval).
ENV_PUSH_CREDENTIALS = "OPTIMA_WEIGHT_PUSH_CREDENTIALS"
# Inline single active secret for eval (and optionally serve with one key).
ENV_PUSH_KEY = "OPTIMA_WEIGHT_PUSH_KEY"
ENV_PUSH_CREDENTIAL_ID = "OPTIMA_WEIGHT_PUSH_CREDENTIAL_ID"
DEFAULT_ENV_CREDENTIAL_ID = "env"


class WeightPushAuthError(RuntimeError):
    """Push credential or signature verification failed closed."""

    validator_fault = True
    retryable = False


@dataclass(frozen=True)
class PushCredential:
    credential_id: str
    secret: str
    status: str = "active"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.credential_id, str)
            or not self.credential_id
            or self.credential_id.strip() != self.credential_id
            or len(self.credential_id) > 128
            or "/" in self.credential_id
            or any(char in self.credential_id for char in " \t\r\n")
        ):
            raise WeightPushAuthError("push credential id is malformed")
        if (
            not isinstance(self.secret, str)
            or len(self.secret) < 32
            or len(self.secret) > 512
            or any(char in self.secret for char in "\x00\r\n")
        ):
            raise WeightPushAuthError("push credential secret is malformed")
        if self.status not in {"active", "retired"}:
            raise WeightPushAuthError("push credential status is unsupported")


@dataclass(frozen=True)
class PushCredentialSet:
    credentials: tuple[PushCredential, ...]

    def __post_init__(self) -> None:
        rows = tuple(self.credentials)
        if not rows:
            raise WeightPushAuthError("push credential set is empty")
        if any(type(row) is not PushCredential for row in rows):
            raise WeightPushAuthError("push credentials are untyped")
        ids = tuple(row.credential_id for row in rows)
        if len(ids) != len(set(ids)):
            raise WeightPushAuthError("push credential ids must be unique")
        if not any(row.status == "active" for row in rows):
            raise WeightPushAuthError("push credential set has no active secret")
        object.__setattr__(self, "credentials", rows)

    def active(self) -> tuple[PushCredential, ...]:
        return tuple(row for row in self.credentials if row.status == "active")

    def get(self, credential_id: str) -> PushCredential | None:
        for row in self.credentials:
            if row.credential_id == credential_id:
                return row
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "credentials": [
                {
                    "credential_id": row.credential_id,
                    "secret": row.secret,
                    "status": row.status,
                }
                for row in self.credentials
            ],
            "schema": CREDENTIALS_SCHEMA,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PushCredentialSet":
        if type(value) is not dict or set(value) != {"credentials", "schema"}:
            raise WeightPushAuthError("push credential file fields do not match")
        if value["schema"] != CREDENTIALS_SCHEMA:
            raise WeightPushAuthError("push credential schema is unsupported")
        raw = value["credentials"]
        if type(raw) is not list or not raw:
            raise WeightPushAuthError("push credentials array is malformed")
        rows: list[PushCredential] = []
        for item in raw:
            if type(item) is not dict or set(item) != {
                "credential_id",
                "secret",
                "status",
            }:
                raise WeightPushAuthError("push credential row fields do not match")
            rows.append(
                PushCredential(
                    str(item["credential_id"]),
                    str(item["secret"]),
                    str(item["status"]),
                )
            )
        return cls(tuple(rows))


def load_push_credentials(path: str | Path) -> PushCredentialSet:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WeightPushAuthError("push credential file is missing") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightPushAuthError(f"push credential file is unreadable: {exc}") from None
    return PushCredentialSet.from_dict(raw)


def push_credentials_from_env_key(
    *,
    secret: str | None = None,
    credential_id: str | None = None,
) -> PushCredentialSet:
    """Build a one-credential set from ``OPTIMA_WEIGHT_PUSH_KEY`` (and optional id)."""

    import os

    raw_secret = secret if secret is not None else os.environ.get(ENV_PUSH_KEY, "")
    if not isinstance(raw_secret, str) or not raw_secret:
        raise WeightPushAuthError(
            f"{ENV_PUSH_KEY} is unset or empty; provide --push-credentials, "
            f"{ENV_PUSH_CREDENTIALS}, or {ENV_PUSH_KEY}"
        )
    raw_id = (
        credential_id
        if credential_id is not None
        else os.environ.get(ENV_PUSH_CREDENTIAL_ID, "") or DEFAULT_ENV_CREDENTIAL_ID
    )
    return PushCredentialSet((PushCredential(str(raw_id), raw_secret, "active"),))


def resolve_push_credentials(
    path: str | Path | None = None,
    *,
    required: bool = False,
) -> PushCredentialSet | None:
    """Resolve push credentials: CLI path → file env → inline key env.

    Precedence:
    1. Explicit ``path`` / ``--push-credentials``
    2. ``OPTIMA_WEIGHT_PUSH_CREDENTIALS`` (JSON file path)
    3. ``OPTIMA_WEIGHT_PUSH_KEY`` (+ optional ``OPTIMA_WEIGHT_PUSH_CREDENTIAL_ID``)
    """

    import os

    explicit = str(path).strip() if path is not None else ""
    if explicit:
        return load_push_credentials(explicit)
    env_path = os.environ.get(ENV_PUSH_CREDENTIALS, "").strip()
    if env_path:
        return load_push_credentials(env_path)
    if os.environ.get(ENV_PUSH_KEY, "").strip():
        return push_credentials_from_env_key()
    if required:
        raise WeightPushAuthError(
            "push credentials required: pass --push-credentials, set "
            f"{ENV_PUSH_CREDENTIALS}, or set {ENV_PUSH_KEY}"
        )
    return None


def write_push_credentials(path: str | Path, credentials: PushCredentialSet) -> Path:
    if type(credentials) is not PushCredentialSet:
        raise WeightPushAuthError("push credentials are untyped")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(credentials.to_dict()) + b"\n"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.chmod(0o600)
    tmp.replace(target)
    target.chmod(0o600)
    return target


def mint_push_credential(*, credential_id: str) -> PushCredential:
    return PushCredential(credential_id, secrets.token_urlsafe(48), "active")


def push_auth_digest(
    *,
    credential_id: str,
    method: str,
    path: str,
    timestamp: int,
    body_digest: str,
) -> str:
    if method != "PUT" or path != "/v1/current-weights":
        raise WeightPushAuthError("push auth route is unsupported")
    if type(timestamp) is not int or timestamp <= 0:
        raise WeightPushAuthError("push timestamp is malformed")
    body_digest = require_sha256_hex(body_digest, field="body_digest")
    return canonical_digest(
        PUSH_AUTH_DOMAIN,
        {
            "body_digest": body_digest,
            "credential_id": credential_id,
            "method": method,
            "path": path,
            "timestamp": timestamp,
        },
    )


def sign_push_request(
    credential: PushCredential,
    *,
    timestamp: int,
    body: bytes,
    method: str = "PUT",
    path: str = "/v1/current-weights",
) -> dict[str, str]:
    if type(credential) is not PushCredential or credential.status != "active":
        raise WeightPushAuthError("push signing requires an active credential")
    if not isinstance(body, (bytes, bytearray)):
        raise WeightPushAuthError("push body must be bytes")
    body_digest = hashlib.sha256(bytes(body)).hexdigest()
    digest = push_auth_digest(
        credential_id=credential.credential_id,
        method=method,
        path=path,
        timestamp=timestamp,
        body_digest=body_digest,
    )
    signature = hmac.new(
        credential.secret.encode("utf-8"),
        digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "Content-Type": "application/json; charset=utf-8",
        "X-Optima-Push-Credential-Id": credential.credential_id,
        "X-Optima-Push-Timestamp": str(timestamp),
        "X-Optima-Push-Body-Digest": body_digest,
        "X-Optima-Push-Signature": signature,
    }


def verify_push_request(
    credentials: PushCredentialSet,
    *,
    headers: dict[str, str],
    body: bytes,
    now: int,
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    method: str = "PUT",
    path: str = "/v1/current-weights",
) -> str:
    """Verify a push request; return the accepted credential id."""

    if type(credentials) is not PushCredentialSet:
        raise WeightPushAuthError("push credential set is untyped")
    if not isinstance(body, (bytes, bytearray)):
        raise WeightPushAuthError("push body must be bytes")
    if type(now) is not int or now <= 0:
        raise WeightPushAuthError("clock reading is malformed")
    if type(max_skew_seconds) is not int or not 1 <= max_skew_seconds <= 600:
        raise WeightPushAuthError("push skew bound is malformed")

    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    credential_id = normalized.get("x-optima-push-credential-id", "")
    timestamp_raw = normalized.get("x-optima-push-timestamp", "")
    body_digest_header = normalized.get("x-optima-push-body-digest", "")
    signature = normalized.get("x-optima-push-signature", "")
    try:
        timestamp = int(timestamp_raw)
    except ValueError as exc:
        raise WeightPushAuthError("push timestamp is malformed") from exc
    if abs(now - timestamp) > max_skew_seconds:
        raise WeightPushAuthError("push timestamp is outside the accepted skew window")

    body_digest = hashlib.sha256(bytes(body)).hexdigest()
    declared = require_sha256_hex(body_digest_header, field="X-Optima-Push-Body-Digest")
    if not hmac.compare_digest(declared, body_digest):
        raise WeightPushAuthError("push body digest mismatch")

    credential = credentials.get(credential_id)
    if credential is None or credential.status != "active":
        raise WeightPushAuthError("push credential is unknown or retired")

    digest = push_auth_digest(
        credential_id=credential.credential_id,
        method=method,
        path=path,
        timestamp=timestamp,
        body_digest=body_digest,
    )
    expected = hmac.new(
        credential.secret.encode("utf-8"),
        digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if (
        not isinstance(signature, str)
        or len(signature) != 64
        or any(char not in "0123456789abcdef" for char in signature)
        or not hmac.compare_digest(signature, expected)
    ):
        raise WeightPushAuthError("push signature is invalid")
    return credential.credential_id


__all__ = [
    "CREDENTIALS_SCHEMA",
    "DEFAULT_ENV_CREDENTIAL_ID",
    "ENV_PUSH_CREDENTIAL_ID",
    "ENV_PUSH_CREDENTIALS",
    "ENV_PUSH_KEY",
    "PushCredential",
    "PushCredentialSet",
    "WeightPushAuthError",
    "load_push_credentials",
    "mint_push_credential",
    "push_auth_digest",
    "push_credentials_from_env_key",
    "resolve_push_credentials",
    "sign_push_request",
    "verify_push_request",
    "write_push_credentials",
]
