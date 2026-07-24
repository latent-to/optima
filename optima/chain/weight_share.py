"""Permit-gated sharing of the current on-chain weight projection.

Authority validators persist the exact ``WeightProjection`` that should be
published — locally on the eval/signer host and asynchronously to a swappable
object store (Hippius/S3/MinIO/local). A separate cheap ``serve-weights``
process reads only from that object store and gates access with timestamp-bound
hotkey signatures so DoS traffic never hits the eval box. Followers rebind the
projection to their own signer and publish through the same
``reconcile_weight_publication`` / ``set_weights`` commit-reveal path used by
``optima set-weights``.

This module is original Optima code (Apache-2.0). Similar subnet patterns
(public weight APIs, hotkey-signed request headers) exist elsewhere; no third-
party sources were copied. Object-store I/O goes through :mod:`optima.object_store`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from optima import chain
from optima.chain.weights import (
    WeightProjection,
    WeightPublicationError,
    WeightPublicationJournal,
    reconcile_weight_publication,
)
from optima.object_store import ObjectStore, ObjectStoreError
from optima.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)


logger = logging.getLogger("optima.chain.weight_share")

OFFER_SCHEMA = "optima.current-weight-offer.v1"
REQUEST_DOMAIN = "optima.weight-share.request.v1"
RESPONSE_DOMAIN = "optima.weight-share.response.v1"
CURRENT_WEIGHTS_PATH = "/v1/current-weights"
DEFAULT_MAX_SKEW_SECONDS = 60
DEFAULT_HTTP_TIMEOUT_SECONDS = 30
DEFAULT_REMOTE_OFFER_KEY = "current_weights.json"
OFFER_CONTENT_TYPE = "application/json; charset=utf-8"

SignFn = Callable[[bytes], bytes]
VerifyFn = Callable[[str, bytes, bytes], bool]
OfferLoader = Callable[[], "CurrentWeightOffer"]


class WeightShareError(RuntimeError):
    """Auth, offer, or transport for shared weights failed closed."""

    validator_fault = True
    retryable = False


class WeightShareRetryableError(WeightShareError):
    """Transient transport or chain read failure that a follower may retry."""

    retryable = True


class HotkeySigner(Protocol):
    """Minimal wallet hotkey surface used for weight-share signatures."""

    ss58_address: str

    def sign(self, data: bytes) -> bytes: ...


@dataclass(frozen=True)
class CurrentWeightOffer:
    """Exact on-disk / on-wire weight projection peers may publish."""

    projection: WeightProjection

    def __post_init__(self) -> None:
        if type(self.projection) is not WeightProjection:
            raise WeightShareError("current weight offer projection is untyped")

    @property
    def digest(self) -> str:
        return self.projection.digest

    def to_dict(self) -> dict[str, object]:
        return {
            "projection": self.projection.to_dict(),
            "projection_digest": self.projection.digest,
            "schema": OFFER_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_dict(cls, value: object) -> "CurrentWeightOffer":
        if type(value) is not dict or set(value) != {
            "projection",
            "projection_digest",
            "schema",
        }:
            raise WeightShareError("current weight offer fields do not match")
        if value["schema"] != OFFER_SCHEMA:
            raise WeightShareError("current weight offer schema is unsupported")
        try:
            projection = WeightProjection.from_dict(value["projection"])
        except WeightPublicationError as exc:
            raise WeightShareError(
                f"current weight offer projection is malformed: {exc}"
            ) from None
        digest = require_sha256_hex(
            value["projection_digest"], field="projection_digest"
        )
        if projection.digest != digest:
            raise WeightShareError(
                "current weight offer projection digest does not match"
            )
        return cls(projection)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CurrentWeightOffer":
        if not isinstance(raw, (bytes, bytearray)):
            raise WeightShareError("current weight offer bytes are malformed")
        try:
            value = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeightShareError(
                f"current weight offer is not canonical JSON: {exc}"
            ) from None
        return cls.from_dict(value)


def default_offer_path(intake_db: str | Path) -> Path:
    """Sibling file next to the intake DB (local durability on the eval host)."""

    path = Path(intake_db)
    return path.with_name(path.name + ".current_weights.json")


def write_current_weight_offer(path: str | Path, projection: WeightProjection) -> Path:
    """Atomically persist the projection locally."""

    if type(projection) is not WeightProjection:
        raise WeightShareError("weight offer requires an exact WeightProjection")
    target = Path(path)
    if target.exists() and not target.is_file():
        raise WeightShareError("weight offer path is not a regular file")
    offer = CurrentWeightOffer(projection)
    payload = offer.to_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
        os.chmod(target, 0o600)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def read_current_weight_offer(path: str | Path) -> CurrentWeightOffer:
    """Reopen one persisted local current-weight offer."""

    target = Path(path)
    try:
        raw = target.read_bytes()
    except FileNotFoundError as exc:
        raise WeightShareError("current weight offer is missing") from exc
    except OSError as exc:
        raise WeightShareError(f"current weight offer cannot be read: {exc}") from None
    return CurrentWeightOffer.from_bytes(raw)


def put_current_weight_offer(
    store: ObjectStore,
    projection: WeightProjection,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> str:
    """Upload the exact offer bytes to a swappable object store."""

    if type(projection) is not WeightProjection:
        raise WeightShareError("weight offer requires an exact WeightProjection")
    offer = CurrentWeightOffer(projection)
    try:
        store.put_bytes(key, offer.to_bytes(), content_type=OFFER_CONTENT_TYPE)
    except ObjectStoreError as exc:
        if getattr(exc, "retryable", False):
            raise WeightShareRetryableError(str(exc)) from None
        raise WeightShareError(str(exc)) from None
    return key


def load_current_weight_offer_from_store(
    store: ObjectStore,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> CurrentWeightOffer:
    """Load the current offer from object storage (serve-weights path)."""

    try:
        raw = store.get_bytes(key)
    except ObjectStoreError as exc:
        if getattr(exc, "retryable", False):
            raise WeightShareRetryableError(str(exc)) from None
        raise WeightShareError(str(exc)) from None
    return CurrentWeightOffer.from_bytes(raw)


def publish_current_weight_offer(
    projection: WeightProjection,
    *,
    local_path: str | Path,
    remote_store: ObjectStore | None = None,
    remote_key: str = DEFAULT_REMOTE_OFFER_KEY,
    async_remote: bool = True,
) -> Path:
    """Write locally, then publish to the object store (optionally in the background).

    Local durability is synchronous so the eval/signer never depends on remote
    latency. Remote upload is best-effort async by default so a slow object-store
    path cannot stall weight publication on the eval host.
    """

    path = write_current_weight_offer(local_path, projection)
    if remote_store is None:
        return path

    def _upload() -> None:
        try:
            put_current_weight_offer(remote_store, projection, key=remote_key)
            logger.info(
                "published weight offer %s to object store key %s",
                projection.digest,
                remote_key,
            )
        except Exception:
            logger.exception(
                "async weight-offer object-store publish failed for %s",
                projection.digest,
            )

    if async_remote:
        threading.Thread(
            target=_upload,
            name="optima-weight-offer-upload",
            daemon=True,
        ).start()
    else:
        put_current_weight_offer(remote_store, projection, key=remote_key)
    return path


def local_offer_loader(path: str | Path) -> OfferLoader:
    target = Path(path)

    def load() -> CurrentWeightOffer:
        return read_current_weight_offer(target)

    return load


def object_store_offer_loader(
    store: ObjectStore,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> OfferLoader:
    def load() -> CurrentWeightOffer:
        return load_current_weight_offer_from_store(store, key=key)

    return load


def rebind_projection_signer(
    projection: WeightProjection, signer_hotkey: str
) -> WeightProjection:
    """Keep the economic vector; bind publication to the follower's hotkey."""

    if type(projection) is not WeightProjection:
        raise WeightShareError("rebind requires an exact WeightProjection")
    if (
        not isinstance(signer_hotkey, str)
        or not signer_hotkey
        or signer_hotkey.strip() != signer_hotkey
        or len(signer_hotkey) > 256
    ):
        raise WeightShareError("follower signer hotkey is malformed")
    if signer_hotkey == projection.validator_hotkey:
        return projection
    return replace(projection, validator_hotkey=signer_hotkey)


def request_auth_digest(
    *,
    hotkey: str,
    method: str,
    netuid: int,
    path: str,
    timestamp: int,
) -> str:
    if (
        not isinstance(hotkey, str)
        or not hotkey
        or hotkey.strip() != hotkey
        or len(hotkey) > 256
    ):
        raise WeightShareError("request hotkey is malformed")
    if method != "GET" or path != CURRENT_WEIGHTS_PATH:
        raise WeightShareError("weight-share request route is unsupported")
    if type(netuid) is not int or netuid < 0:
        raise WeightShareError("request netuid is malformed")
    if type(timestamp) is not int or timestamp <= 0:
        raise WeightShareError("request timestamp is malformed")
    return canonical_digest(
        REQUEST_DOMAIN,
        {
            "hotkey": hotkey,
            "method": method,
            "netuid": netuid,
            "path": path,
            "timestamp": timestamp,
        },
    )


def response_auth_digest(
    *,
    authority_hotkey: str,
    body_digest: str,
    netuid: int,
    timestamp: int,
) -> str:
    if (
        not isinstance(authority_hotkey, str)
        or not authority_hotkey
        or authority_hotkey.strip() != authority_hotkey
        or len(authority_hotkey) > 256
    ):
        raise WeightShareError("response authority hotkey is malformed")
    body_digest = require_sha256_hex(body_digest, field="body_digest")
    if type(netuid) is not int or netuid < 0:
        raise WeightShareError("response netuid is malformed")
    if type(timestamp) is not int or timestamp <= 0:
        raise WeightShareError("response timestamp is malformed")
    return canonical_digest(
        RESPONSE_DOMAIN,
        {
            "authority_hotkey": authority_hotkey,
            "body_digest": body_digest,
            "netuid": netuid,
            "timestamp": timestamp,
        },
    )


def _message_bytes(digest: str) -> bytes:
    return require_sha256_hex(digest, field="auth_digest").encode("ascii")


def sign_auth_digest(signer: HotkeySigner, digest: str) -> str:
    try:
        signature = signer.sign(_message_bytes(digest))
    except Exception as exc:
        raise WeightShareError(f"weight-share signing failed: {exc}") from None
    if not isinstance(signature, (bytes, bytearray)) or not signature:
        raise WeightShareError("weight-share signature bytes are malformed")
    return bytes(signature).hex()


def verify_auth_digest(
    hotkey: str,
    digest: str,
    signature_hex: str,
    *,
    verify: VerifyFn,
) -> None:
    if (
        not isinstance(signature_hex, str)
        or len(signature_hex) < 64
        or len(signature_hex) > 256
        or len(signature_hex) % 2 != 0
        or any(char not in "0123456789abcdef" for char in signature_hex)
    ):
        raise WeightShareError("weight-share signature encoding is malformed")
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError as exc:
        raise WeightShareError("weight-share signature encoding is malformed") from exc
    try:
        ok = bool(verify(hotkey, _message_bytes(digest), signature))
    except Exception as exc:
        raise WeightShareError(
            f"weight-share signature verification failed: {exc}"
        ) from None
    if not ok:
        raise WeightShareError("weight-share signature is invalid")


def assert_fresh_timestamp(
    timestamp: int, *, now: int, max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS
) -> None:
    if type(timestamp) is not int or timestamp <= 0:
        raise WeightShareError("timestamp is malformed")
    if type(now) is not int or now <= 0:
        raise WeightShareError("clock reading is malformed")
    if type(max_skew_seconds) is not int or max_skew_seconds <= 0 or max_skew_seconds > 600:
        raise WeightShareError("timestamp skew bound is malformed")
    if abs(now - timestamp) > max_skew_seconds:
        raise WeightShareError("timestamp is outside the accepted skew window")


def assert_validator_permit(metagraph: chain.MetagraphView, hotkey: str) -> int:
    if type(metagraph) is not chain.MetagraphView:
        raise WeightShareError("permit check requires an exact MetagraphView")
    if (
        not isinstance(hotkey, str)
        or not hotkey
        or hotkey.strip() != hotkey
        or len(hotkey) > 256
    ):
        raise WeightShareError("permit hotkey is malformed")
    uid = metagraph.uid_of(hotkey)
    if uid is None:
        raise WeightShareError("hotkey is not registered on the metagraph")
    if uid >= len(metagraph.validator_permit) or not bool(
        metagraph.validator_permit[uid]
    ):
        raise WeightShareError("hotkey does not currently hold validator_permit")
    return uid


def default_verify_fn(hotkey: str, message: bytes, signature: bytes) -> bool:
    """Verify an sr25519 hotkey signature via the installed wallet stack."""

    try:
        import bittensor as bt
    except ImportError as exc:
        raise WeightShareError(
            "bittensor is required to verify weight-share signatures"
        ) from exc
    keypair_cls = getattr(bt, "Keypair", None)
    if keypair_cls is None:
        wallet_mod = getattr(bt, "wallet", None)
        keypair_cls = getattr(wallet_mod, "Keypair", None) if wallet_mod else None
    if keypair_cls is None:
        try:
            from bittensor_wallet import Keypair as keypair_cls  # type: ignore
        except ImportError as exc:
            raise WeightShareError(
                "no Keypair implementation is available for weight-share verify"
            ) from exc
    keypair = keypair_cls(ss58_address=hotkey)
    return bool(keypair.verify(message, signature))


def build_signed_offer_response(
    offer: CurrentWeightOffer,
    *,
    authority: HotkeySigner,
    netuid: int,
    timestamp: int,
) -> tuple[bytes, dict[str, str]]:
    if offer.projection.netuid != netuid:
        raise WeightShareError("offer netuid differs from the served netuid")
    if authority.ss58_address != offer.projection.validator_hotkey:
        raise WeightShareError(
            "authority hotkey differs from the stored projection authority"
        )
    body_obj = {
        "authority_hotkey": authority.ss58_address,
        "netuid": netuid,
        "offer": offer.to_dict(),
        "timestamp": timestamp,
    }
    body = canonical_json_bytes(body_obj) + b"\n"
    body_digest = sha256_hex(body)
    digest = response_auth_digest(
        authority_hotkey=authority.ss58_address,
        body_digest=body_digest,
        netuid=netuid,
        timestamp=timestamp,
    )
    signature = sign_auth_digest(authority, digest)
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Optima-Authority-Hotkey": authority.ss58_address,
        "X-Optima-Netuid": str(netuid),
        "X-Optima-Timestamp": str(timestamp),
        "X-Optima-Signature": signature,
        "X-Optima-Body-Digest": body_digest,
    }
    return body, headers


def parse_signed_offer_response(
    body: bytes,
    headers: dict[str, str],
    *,
    netuid: int,
    now: int,
    max_skew_seconds: int,
    verify: VerifyFn,
    expected_authority: str | None = None,
    metagraph: chain.MetagraphView | None = None,
) -> CurrentWeightOffer:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightShareError(f"weight-share response is not JSON: {exc}") from None
    if type(payload) is not dict or set(payload) != {
        "authority_hotkey",
        "netuid",
        "offer",
        "timestamp",
    }:
        raise WeightShareError("weight-share response fields do not match")
    authority_hotkey = payload["authority_hotkey"]
    timestamp = payload["timestamp"]
    response_netuid = payload["netuid"]
    if response_netuid != netuid:
        raise WeightShareError("weight-share response netuid mismatch")
    if type(timestamp) is not int:
        raise WeightShareError("weight-share response timestamp is malformed")
    assert_fresh_timestamp(
        timestamp, now=now, max_skew_seconds=max_skew_seconds
    )
    header_authority = headers.get("X-Optima-Authority-Hotkey", "")
    header_timestamp = headers.get("X-Optima-Timestamp", "")
    header_netuid = headers.get("X-Optima-Netuid", "")
    signature = headers.get("X-Optima-Signature", "")
    header_body_digest = headers.get("X-Optima-Body-Digest", "")
    if (
        header_authority != authority_hotkey
        or header_timestamp != str(timestamp)
        or header_netuid != str(netuid)
    ):
        raise WeightShareError("weight-share response headers disagree with body")
    body_digest = sha256_hex(body)
    if header_body_digest:
        declared = require_sha256_hex(
            header_body_digest, field="X-Optima-Body-Digest"
        )
        if declared != body_digest:
            raise WeightShareError("weight-share response body digest mismatch")
    digest = response_auth_digest(
        authority_hotkey=authority_hotkey,
        body_digest=body_digest,
        netuid=netuid,
        timestamp=timestamp,
    )
    verify_auth_digest(authority_hotkey, digest, signature, verify=verify)
    if expected_authority is not None and authority_hotkey != expected_authority:
        raise WeightShareError("weight-share authority hotkey is not the pinned authority")
    if metagraph is not None:
        assert_validator_permit(metagraph, authority_hotkey)
    offer = CurrentWeightOffer.from_dict(payload["offer"])
    if offer.projection.validator_hotkey != authority_hotkey:
        raise WeightShareError(
            "stored projection authority differs from the response signer"
        )
    if offer.projection.netuid != netuid:
        raise WeightShareError("offer projection netuid mismatch")
    return offer


def _normalize_headers(headers: object) -> dict[str, str]:
    if hasattr(headers, "items"):
        return {str(key): str(value) for key, value in headers.items()}  # type: ignore[arg-type]
    raise WeightShareError("response headers are malformed")


class _WeightShareHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler,
        *,
        load_offer: OfferLoader,
        authority: HotkeySigner,
        subtensor,
        netuid: int,
        max_skew_seconds: int,
        verify: VerifyFn,
        clock: Callable[[], int],
    ) -> None:
        super().__init__(server_address, handler)
        self.load_offer = load_offer
        self.authority = authority
        self.subtensor = subtensor
        self.netuid = netuid
        self.max_skew_seconds = max_skew_seconds
        self.verify = verify
        self.clock = clock
        self._offer_lock = threading.Lock()


class _WeightShareHandler(BaseHTTPRequestHandler):
    server: _WeightShareHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        logger.info("weight-share: " + format, *args)

    def _send(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {"Content-Type": "application/json"}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        payload = canonical_json_bytes({"error": message}) + b"\n"
        self._send(status, payload, {"Content-Type": "application/json; charset=utf-8"})

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path.split("?", 1)[0] != CURRENT_WEIGHTS_PATH:
            self._error(404, "not found")
            return
        server = self.server
        try:
            hotkey = self.headers.get("X-Optima-Hotkey", "")
            timestamp_raw = self.headers.get("X-Optima-Timestamp", "")
            signature = self.headers.get("X-Optima-Signature", "")
            header_netuid = self.headers.get("X-Optima-Netuid", "")
            if header_netuid != str(server.netuid):
                raise WeightShareError("request netuid header mismatch")
            try:
                timestamp = int(timestamp_raw)
            except ValueError as exc:
                raise WeightShareError("request timestamp is malformed") from exc
            now = int(server.clock())
            assert_fresh_timestamp(
                timestamp, now=now, max_skew_seconds=server.max_skew_seconds
            )
            digest = request_auth_digest(
                hotkey=hotkey,
                method="GET",
                netuid=server.netuid,
                path=CURRENT_WEIGHTS_PATH,
                timestamp=timestamp,
            )
            verify_auth_digest(hotkey, digest, signature, verify=server.verify)
            metagraph = chain.fetch_metagraph(server.subtensor, server.netuid)
            assert_validator_permit(metagraph, hotkey)
            with server._offer_lock:
                offer = server.load_offer()
            if offer.projection.netuid != server.netuid:
                raise WeightShareError("stored offer netuid differs from server netuid")
            if offer.projection.validator_hotkey != server.authority.ss58_address:
                raise WeightShareError(
                    "stored offer authority differs from the serving wallet"
                )
            body, headers = build_signed_offer_response(
                offer,
                authority=server.authority,
                netuid=server.netuid,
                timestamp=now,
            )
        except WeightShareError as exc:
            self._error(403, str(exc))
            return
        except Exception as exc:
            logger.exception("weight-share handler failed")
            self._error(500, f"internal error: {type(exc).__name__}")
            return
        self._send(200, body, headers)


def serve_current_weights(
    *,
    host: str,
    port: int,
    authority: HotkeySigner,
    subtensor,
    netuid: int,
    load_offer: OfferLoader | None = None,
    offer_path: str | Path | None = None,
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    verify: VerifyFn | None = None,
    clock: Callable[[], int] | None = None,
) -> ThreadingHTTPServer:
    """Start the permit-gated current-weights HTTP server (caller serves forever).

    Prefer ``load_offer`` from an object store on a host separate from eval.
    ``offer_path`` remains a local-file convenience for tests and air-gapped runs.
    """

    if not isinstance(host, str) or not host.strip():
        raise WeightShareError("weight-share host is malformed")
    if type(port) is not int or not 0 <= port <= 65535:
        raise WeightShareError("weight-share port is malformed")
    if load_offer is None:
        if offer_path is None:
            raise WeightShareError(
                "serve-weights requires load_offer or offer_path"
            )
        load_offer = local_offer_loader(offer_path)
    server = _WeightShareHTTPServer(
        (host, port),
        _WeightShareHandler,
        load_offer=load_offer,
        authority=authority,
        subtensor=subtensor,
        netuid=netuid,
        max_skew_seconds=max_skew_seconds,
        verify=verify or default_verify_fn,
        clock=clock or (lambda: int(time.time())),
    )
    return server


def fetch_current_weights(
    url: str,
    *,
    signer: HotkeySigner,
    netuid: int,
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    verify: VerifyFn | None = None,
    clock: Callable[[], int] | None = None,
    expected_authority: str | None = None,
    metagraph: chain.MetagraphView | None = None,
    opener: Callable[..., object] | None = None,
) -> CurrentWeightOffer:
    """Authenticated GET of the current publishable weight offer."""

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise WeightShareError("weight-share URL must be http(s)")
    if type(timeout_seconds) is not float and type(timeout_seconds) is not int:
        raise WeightShareError("timeout is malformed")
    timeout = float(timeout_seconds)
    if not 0.1 <= timeout <= 600:
        raise WeightShareError("timeout is out of bounds")
    now = int((clock or (lambda: int(time.time())))())
    digest = request_auth_digest(
        hotkey=signer.ss58_address,
        method="GET",
        netuid=netuid,
        path=CURRENT_WEIGHTS_PATH,
        timestamp=now,
    )
    signature = sign_auth_digest(signer, digest)
    endpoint = url.rstrip("/") + CURRENT_WEIGHTS_PATH
    request = Request(
        endpoint,
        method="GET",
        headers={
            "Accept": "application/json",
            "X-Optima-Hotkey": signer.ss58_address,
            "X-Optima-Netuid": str(netuid),
            "X-Optima-Timestamp": str(now),
            "X-Optima-Signature": signature,
        },
    )
    open_url = opener or urlopen
    try:
        with open_url(request, timeout=timeout) as response:  # type: ignore[arg-type]
            body = response.read()
            headers = _normalize_headers(getattr(response, "headers", {}))
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        if 500 <= int(exc.code) <= 599:
            raise WeightShareRetryableError(
                f"weight-share server error {exc.code}: {detail}"
            ) from None
        raise WeightShareError(
            f"weight-share request rejected ({exc.code}): {detail}"
        ) from None
    except URLError as exc:
        raise WeightShareRetryableError(
            f"weight-share transport failed: {exc}"
        ) from None
    if status != 200:
        raise WeightShareError(f"weight-share unexpected status {status}")
    return parse_signed_offer_response(
        body,
        headers,
        netuid=netuid,
        now=now,
        max_skew_seconds=max_skew_seconds,
        verify=verify or default_verify_fn,
        expected_authority=expected_authority,
        metagraph=metagraph,
    )


def publish_followed_weights(
    *,
    subtensor,
    signer_wallet,
    offer: CurrentWeightOffer,
    journal: WeightPublicationJournal,
    refresh_blocks: int,
    dry_run: bool = False,
):
    """Publish a fetched offer through the normal weight reconciler."""

    try:
        follower_hotkey = signer_wallet.hotkey.ss58_address
    except AttributeError as exc:
        raise WeightShareError("follower publish requires a signer wallet") from exc
    projection = rebind_projection_signer(offer.projection, follower_hotkey)
    return reconcile_weight_publication(
        subtensor,
        None if dry_run else signer_wallet,
        projection,
        journal,
        refresh_blocks=refresh_blocks,
        dry_run=dry_run,
        require_current_crown=projection.crown_count > 0,
    )


__all__ = [
    "CURRENT_WEIGHTS_PATH",
    "CurrentWeightOffer",
    "DEFAULT_MAX_SKEW_SECONDS",
    "DEFAULT_REMOTE_OFFER_KEY",
    "WeightShareError",
    "WeightShareRetryableError",
    "assert_fresh_timestamp",
    "assert_validator_permit",
    "build_signed_offer_response",
    "default_offer_path",
    "default_verify_fn",
    "fetch_current_weights",
    "load_current_weight_offer_from_store",
    "local_offer_loader",
    "object_store_offer_loader",
    "parse_signed_offer_response",
    "publish_current_weight_offer",
    "publish_followed_weights",
    "put_current_weight_offer",
    "read_current_weight_offer",
    "rebind_projection_signer",
    "request_auth_digest",
    "response_auth_digest",
    "serve_current_weights",
    "sign_auth_digest",
    "verify_auth_digest",
    "write_current_weight_offer",
]
