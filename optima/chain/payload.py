"""The on-chain submission payload: a tiny JSON binding a bundle hash to a fetch URL.

This is the ONLY thing a miner puts on chain — `{"v": 1, "h": <content_hash>,
"u": <url>}` — committed via the chain's timelock commit-reveal. Everything the
validator later trusts about the bundle flows from ``h``: the fetched artifact is
extracted and re-hashed with ``optima.bundle_hash.content_hash``, and a mismatch is
a rejected submission. Decoding is fail-quiet (returns ``None``), because payloads
arrive from arbitrary registered hotkeys: garbage on chain must never crash the
validator loop.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("optima.chain.payload")

PAYLOAD_VERSION = 1
# Chain-side cap for TimelockEncrypted commitments (subtensor commitments pallet).
MAX_PAYLOAD_BYTES = 1024
# Fetch transports the validator will follow. file:// is for local/dev loops only.
ALLOWED_URL_SCHEMES = ("https", "http", "file")

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEME_RE = re.compile(r"^([a-z][a-z0-9+.-]*)://")


class PayloadError(ValueError):
    """A payload WE are building is invalid (miner-side, loud)."""


@dataclass(frozen=True)
class SubmissionRef:
    """A decoded, validated on-chain submission reference."""
    hotkey: str
    content_hash: str
    url: str
    block: int  # reveal block — the consensus anti-copy priority timestamp


def encode_payload(content_hash: str, url: str) -> str:
    """Miner side: build the commitment JSON. Raises PayloadError on anything that
    would be rejected by the validator or the chain — fail loud before signing."""
    if not _HASH_RE.match(content_hash or ""):
        raise PayloadError(f"content_hash must be 64 lowercase hex chars, got {content_hash!r}")
    scheme = _SCHEME_RE.match(url or "")
    if not scheme or scheme.group(1) not in ALLOWED_URL_SCHEMES:
        raise PayloadError(f"url scheme must be one of {ALLOWED_URL_SCHEMES}, got {url!r}")
    data = json.dumps({"v": PAYLOAD_VERSION, "h": content_hash, "u": url},
                      separators=(",", ":"))
    n = len(data.encode("utf-8"))
    if n > MAX_PAYLOAD_BYTES:
        raise PayloadError(f"payload is {n} bytes; chain cap is {MAX_PAYLOAD_BYTES}")
    return data


def decode_payload(hotkey: str, block: int, data: str) -> "SubmissionRef | None":
    """Validator side: parse an untrusted on-chain payload. Returns None (and logs)
    on anything malformed — never raises on chain-sourced data."""
    if not isinstance(data, str) or len(data.encode("utf-8", "replace")) > MAX_PAYLOAD_BYTES:
        logger.warning("payload from %s: oversized or non-string; ignored", hotkey)
        return None
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("payload from %s: not JSON; ignored", hotkey)
        return None
    if not isinstance(obj, dict) or obj.get("v") != PAYLOAD_VERSION:
        logger.warning("payload from %s: missing/unknown version; ignored", hotkey)
        return None
    content_hash, url = obj.get("h"), obj.get("u")
    if not isinstance(content_hash, str) or not _HASH_RE.match(content_hash):
        logger.warning("payload from %s: bad content hash; ignored", hotkey)
        return None
    scheme = _SCHEME_RE.match(url or "") if isinstance(url, str) else None
    if not scheme or scheme.group(1) not in ALLOWED_URL_SCHEMES:
        logger.warning("payload from %s: bad/disallowed url; ignored", hotkey)
        return None
    return SubmissionRef(hotkey=hotkey, content_hash=content_hash, url=url, block=int(block))
