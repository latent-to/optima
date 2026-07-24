"""Permit-gated sharing of current publishable weights (V1 or debt/composition).

Eval builds a :class:`CurrentWeightOffer` — legacy V1 projection or a full
:class:`DebtWeightPublicationBinding` — and pushes it to ``serve-weights`` with
rotatable HMAC credentials. Eval never opens a chain-signing weight path.

Cheap ``serve-weights`` hosts persist the offer (object store or local file),
accept authenticated PUT from eval, and serve permit-gated GET to validators.
Followers rebind the signer-facing projection and publish via
``reconcile_weight_publication`` / commit-reveal (``follow-weights``).

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
from optima.chain.debt_publication import (
    PUBLICATION_KIND_COMPOSED,
    PUBLICATION_KIND_CORE,
    DebtPublicationError,
    DebtWeightPublicationBinding,
)
from optima.chain.weight_push_auth import (
    PushCredential,
    PushCredentialSet,
    WeightPushAuthError,
    sign_push_request,
    verify_push_request,
)
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

OFFER_SCHEMA_V1 = "optima.current-weight-offer.v1"
OFFER_SCHEMA = "optima.current-weight-offer.v2"
LANE_LEGACY_V1 = "legacy_v1"
LANE_COMPOSED = "incentive_composition"
LANE_CORE = "finite_debt"
OFFER_LANES = frozenset({LANE_LEGACY_V1, LANE_COMPOSED, LANE_CORE})
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
OfferSink = Callable[["CurrentWeightOffer"], None]


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
    """Exact publishable weights for peer validators.

    V2 debt / composition offers carry the full
    :class:`DebtWeightPublicationBinding` so followers can rebind the signer
    hotkey and publish the same economic vector through the debt reconciler.
    Legacy V1 offers carry only a :class:`WeightProjection`.
    """

    lane: str
    projection: WeightProjection
    debt_binding: DebtWeightPublicationBinding | None = None

    def __post_init__(self) -> None:
        if self.lane not in OFFER_LANES:
            raise WeightShareError("current weight offer lane is unsupported")
        if type(self.projection) is not WeightProjection:
            raise WeightShareError("current weight offer projection is untyped")
        if self.lane == LANE_LEGACY_V1:
            if self.debt_binding is not None:
                raise WeightShareError("legacy V1 offer cannot carry a debt binding")
            return
        if type(self.debt_binding) is not DebtWeightPublicationBinding:
            raise WeightShareError("debt-lane offer requires an exact debt binding")
        expected_kind = (
            PUBLICATION_KIND_COMPOSED
            if self.lane == LANE_COMPOSED
            else PUBLICATION_KIND_CORE
        )
        if (
            self.debt_binding.publication_kind != expected_kind
            or self.debt_binding.weight_projection != self.projection
            or self.debt_binding.weight_projection.digest != self.projection.digest
        ):
            raise WeightShareError(
                "debt-lane offer projection differs from its economic binding"
            )
        economic_weights = tuple(
            (row.hotkey, row.units) for row in self.debt_binding.weights
        )
        if self.projection.weights_ppm != economic_weights:
            raise WeightShareError(
                "offer weights_ppm differ from the debt economic projection"
            )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.current-weight-offer", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "lane": self.lane,
            "projection": self.projection.to_dict(),
            "projection_digest": self.projection.digest,
            "schema": OFFER_SCHEMA,
        }
        if self.debt_binding is not None:
            row["debt_binding"] = self.debt_binding.to_dict()
            row["debt_binding_digest"] = self.debt_binding.digest
        else:
            row["debt_binding"] = None
            row["debt_binding_digest"] = None
        return row

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_legacy_projection(cls, projection: WeightProjection) -> "CurrentWeightOffer":
        return cls(LANE_LEGACY_V1, projection, None)

    @classmethod
    def from_debt_binding(cls, binding: DebtWeightPublicationBinding) -> "CurrentWeightOffer":
        if type(binding) is not DebtWeightPublicationBinding:
            raise WeightShareError("debt offer requires an exact binding")
        lane = (
            LANE_COMPOSED
            if binding.publication_kind == PUBLICATION_KIND_COMPOSED
            else LANE_CORE
            if binding.publication_kind == PUBLICATION_KIND_CORE
            else ""
        )
        if not lane:
            raise WeightShareError("debt offer publication kind is unsupported")
        return cls(lane, binding.weight_projection, binding)

    @classmethod
    def from_dict(cls, value: object) -> "CurrentWeightOffer":
        if type(value) is not dict:
            raise WeightShareError("current weight offer fields do not match")
        schema = value.get("schema")
        if schema == OFFER_SCHEMA_V1:
            # Historical local files: projection-only legacy V1.
            if set(value) != {"projection", "projection_digest", "schema"}:
                raise WeightShareError("legacy weight offer fields do not match")
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
            return cls.from_legacy_projection(projection)
        if schema != OFFER_SCHEMA:
            raise WeightShareError("current weight offer schema is unsupported")
        expected = {
            "debt_binding",
            "debt_binding_digest",
            "lane",
            "projection",
            "projection_digest",
            "schema",
        }
        if set(value) != expected:
            raise WeightShareError("current weight offer fields do not match")
        lane = value["lane"]
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
        binding = None
        if value["debt_binding"] is not None:
            try:
                binding = DebtWeightPublicationBinding.from_dict(value["debt_binding"])
            except DebtPublicationError as exc:
                raise WeightShareError(
                    f"current weight offer debt binding is malformed: {exc}"
                ) from None
            binding_digest = require_sha256_hex(
                value["debt_binding_digest"], field="debt_binding_digest"
            )
            if binding.digest != binding_digest:
                raise WeightShareError(
                    "current weight offer debt binding digest does not match"
                )
        elif value["debt_binding_digest"] is not None:
            raise WeightShareError("debt binding digest present without binding")
        return cls(str(lane), projection, binding)

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


def write_current_weight_offer(
    path: str | Path, offer: CurrentWeightOffer | WeightProjection
) -> Path:
    """Atomically persist the current offer locally."""

    if type(offer) is WeightProjection:
        offer = CurrentWeightOffer.from_legacy_projection(offer)
    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("weight offer requires an exact CurrentWeightOffer")
    target = Path(path)
    if target.exists() and not target.is_file():
        raise WeightShareError("weight offer path is not a regular file")
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
    offer: CurrentWeightOffer | WeightProjection,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> str:
    """Upload the exact offer bytes to a swappable object store."""

    if type(offer) is WeightProjection:
        offer = CurrentWeightOffer.from_legacy_projection(offer)
    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("weight offer requires an exact CurrentWeightOffer")
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
    offer: CurrentWeightOffer | WeightProjection,
    *,
    local_path: str | Path,
    remote_store: ObjectStore | None = None,
    remote_key: str = DEFAULT_REMOTE_OFFER_KEY,
    async_remote: bool = True,
) -> Path:
    """Write locally, then publish to the object store (optionally in the background)."""

    if type(offer) is WeightProjection:
        offer = CurrentWeightOffer.from_legacy_projection(offer)
    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("weight offer requires an exact CurrentWeightOffer")
    path = write_current_weight_offer(local_path, offer)
    if remote_store is None:
        return path

    def _upload() -> None:
        try:
            put_current_weight_offer(remote_store, offer, key=remote_key)
            logger.info(
                "published weight offer %s to object store key %s",
                offer.digest,
                remote_key,
            )
        except Exception:
            logger.exception(
                "async weight-offer object-store publish failed for %s",
                offer.digest,
            )

    if async_remote:
        threading.Thread(
            target=_upload,
            name="optima-weight-offer-upload",
            daemon=True,
        ).start()
    else:
        put_current_weight_offer(remote_store, offer, key=remote_key)
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


def object_store_offer_sink(
    store: ObjectStore,
    *,
    key: str = DEFAULT_REMOTE_OFFER_KEY,
) -> OfferSink:
    def save(offer: CurrentWeightOffer) -> None:
        put_current_weight_offer(store, offer, key=key)

    return save


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


def rebind_offer_signer(
    offer: CurrentWeightOffer, signer_hotkey: str
) -> CurrentWeightOffer:
    """Rebind the signer-facing projection (and debt binding, when present)."""

    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("rebind requires an exact CurrentWeightOffer")
    projection = rebind_projection_signer(offer.projection, signer_hotkey)
    if offer.debt_binding is None:
        return CurrentWeightOffer.from_legacy_projection(projection)
    binding = DebtWeightPublicationBinding(
        offer.debt_binding.publication_kind,
        offer.debt_binding.activation_digest,
        offer.debt_binding.effective_block_hash,
        offer.debt_binding.economic_projection,
        projection,
    )
    return CurrentWeightOffer.from_debt_binding(binding)


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
    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("signed response requires an exact CurrentWeightOffer")
    if offer.projection.netuid != netuid:
        raise WeightShareError("offer netuid differs from the served netuid")
    # The HTTP response signer is the weights-service hotkey. It need not equal
    # the offer's projection.validator_hotkey: eval builds the economic vector,
    # followers rebind before chain publish.
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
        save_offer: OfferSink | None,
        push_credentials: PushCredentialSet | None,
        authority: HotkeySigner,
        subtensor,
        netuid: int,
        max_skew_seconds: int,
        verify: VerifyFn,
        clock: Callable[[], int],
    ) -> None:
        super().__init__(server_address, handler)
        self.load_offer = load_offer
        self.save_offer = save_offer
        self.push_credentials = push_credentials
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

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path.split("?", 1)[0] != CURRENT_WEIGHTS_PATH:
            self._error(404, "not found")
            return
        server = self.server
        if server.push_credentials is None or server.save_offer is None:
            self._error(405, "weight push is not enabled on this server")
            return
        try:
            length_raw = self.headers.get("Content-Length", "")
            try:
                length = int(length_raw)
            except ValueError as exc:
                raise WeightShareError("push Content-Length is malformed") from exc
            if length < 2 or length > 8_000_000:
                raise WeightShareError("push body length is out of bounds")
            body = self.rfile.read(length)
            if len(body) != length:
                raise WeightShareError("push body length mismatch")
            now = int(server.clock())
            credential_id = verify_push_request(
                server.push_credentials,
                headers=dict(self.headers.items()),
                body=body,
                now=now,
                max_skew_seconds=server.max_skew_seconds,
            )
            offer = CurrentWeightOffer.from_bytes(body)
            if offer.projection.netuid != server.netuid:
                raise WeightShareError("pushed offer netuid differs from server netuid")
            with server._offer_lock:
                server.save_offer(offer)
            logger.info(
                "accepted weight offer %s lane=%s via push credential %s",
                offer.digest,
                offer.lane,
                credential_id,
            )
            response = canonical_json_bytes(
                {
                    "credential_id": credential_id,
                    "offer_digest": offer.digest,
                    "projection_digest": offer.projection.digest,
                    "status": "accepted",
                }
            ) + b"\n"
        except (WeightShareError, WeightPushAuthError) as exc:
            self._error(403, str(exc))
            return
        except Exception as exc:
            logger.exception("weight-share push failed")
            self._error(500, f"internal error: {type(exc).__name__}")
            return
        self._send(
            200,
            response,
            {"Content-Type": "application/json; charset=utf-8"},
        )


def serve_current_weights(
    *,
    host: str,
    port: int,
    authority: HotkeySigner,
    subtensor,
    netuid: int,
    load_offer: OfferLoader | None = None,
    save_offer: OfferSink | None = None,
    push_credentials: PushCredentialSet | None = None,
    offer_path: str | Path | None = None,
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    verify: VerifyFn | None = None,
    clock: Callable[[], int] | None = None,
) -> ThreadingHTTPServer:
    """Start the permit-gated current-weights HTTP server (caller serves forever).

    Prefer ``load_offer`` / ``save_offer`` backed by object storage on a host
    separate from eval. ``PUT /v1/current-weights`` accepts eval pushes only when
    rotatable ``push_credentials`` are configured. Eval must not chain-publish.
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
        if save_offer is None:
            path = Path(offer_path)

            def _save(offer: CurrentWeightOffer) -> None:
                write_current_weight_offer(path, offer)

            save_offer = _save
    if push_credentials is not None and save_offer is None:
        raise WeightShareError("push credentials require a configured save_offer")
    if push_credentials is not None and type(push_credentials) is not PushCredentialSet:
        raise WeightShareError("push credentials are untyped")
    server = _WeightShareHTTPServer(
        (host, port),
        _WeightShareHandler,
        load_offer=load_offer,
        save_offer=save_offer,
        push_credentials=push_credentials,
        authority=authority,
        subtensor=subtensor,
        netuid=netuid,
        max_skew_seconds=max_skew_seconds,
        verify=verify or default_verify_fn,
        clock=clock or (lambda: int(time.time())),
    )
    return server


def push_current_weights(
    url: str,
    offer: CurrentWeightOffer,
    *,
    credential: PushCredential,
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    clock: Callable[[], int] | None = None,
    opener: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Eval-side: push one offer to the weights service. Never touches the chain."""

    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("push requires an exact CurrentWeightOffer")
    if type(credential) is not PushCredential:
        raise WeightShareError("push requires an exact PushCredential")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise WeightShareError("weight-share URL must be http(s)")
    timeout = float(timeout_seconds)
    if not 0.1 <= timeout <= 600:
        raise WeightShareError("timeout is out of bounds")
    body = offer.to_bytes()
    now = int((clock or (lambda: int(time.time())))())
    headers = sign_push_request(credential, timestamp=now, body=body)
    endpoint = url.rstrip("/") + CURRENT_WEIGHTS_PATH
    request = Request(endpoint, data=body, method="PUT", headers=headers)
    open_url = opener or urlopen
    try:
        with open_url(request, timeout=timeout) as response:  # type: ignore[arg-type]
            raw = response.read()
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        if 500 <= int(exc.code) <= 599:
            raise WeightShareRetryableError(
                f"weight-share push server error {exc.code}: {detail}"
            ) from None
        raise WeightShareError(
            f"weight-share push rejected ({exc.code}): {detail}"
        ) from None
    except URLError as exc:
        raise WeightShareRetryableError(
            f"weight-share push transport failed: {exc}"
        ) from None
    if status != 200:
        raise WeightShareError(f"weight-share push unexpected status {status}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightShareError(f"weight-share push response is not JSON: {exc}") from None
    if type(payload) is not dict:
        raise WeightShareError("weight-share push response is malformed")
    return payload


def publish_followed_weights(
    *,
    subtensor,
    signer_wallet,
    offer: CurrentWeightOffer,
    journal: WeightPublicationJournal,
    refresh_blocks: int,
    dry_run: bool = False,
):
    """Publish a fetched offer through the normal weight reconciler / commit-reveal."""

    if type(offer) is not CurrentWeightOffer:
        raise WeightShareError("follow publish requires an exact CurrentWeightOffer")
    try:
        follower_hotkey = signer_wallet.hotkey.ss58_address
    except AttributeError as exc:
        raise WeightShareError("follower publish requires a signer wallet") from exc
    rebound = rebind_offer_signer(offer, follower_hotkey)
    # Debt / composition epochs are crownless by construction relative to the
    # legacy V1 require_current_crown gate; match set-debt-weights.
    require_crown = rebound.lane == LANE_LEGACY_V1 and rebound.projection.crown_count > 0
    return reconcile_weight_publication(
        subtensor,
        None if dry_run else signer_wallet,
        rebound.projection,
        journal,
        refresh_blocks=refresh_blocks,
        dry_run=dry_run,
        require_current_crown=require_crown,
    )


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


__all__ = [
    "CURRENT_WEIGHTS_PATH",
    "CurrentWeightOffer",
    "DEFAULT_MAX_SKEW_SECONDS",
    "DEFAULT_REMOTE_OFFER_KEY",
    "LANE_COMPOSED",
    "LANE_CORE",
    "LANE_LEGACY_V1",
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
    "object_store_offer_sink",
    "parse_signed_offer_response",
    "publish_current_weight_offer",
    "publish_followed_weights",
    "push_current_weights",
    "put_current_weight_offer",
    "read_current_weight_offer",
    "rebind_offer_signer",
    "rebind_projection_signer",
    "request_auth_digest",
    "response_auth_digest",
    "serve_current_weights",
    "sign_auth_digest",
    "verify_auth_digest",
    "write_current_weight_offer",
]
