"""Strict on-chain submission references for finalized validator intake.

Production payloads are exact three-field JSON objects and may name only a
canonical HTTPS URL.  ``file://`` exists behind separate test-only functions so
an on-chain value can never select local files or plaintext HTTP.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger("optima.chain.payload")

PAYLOAD_VERSION = 1
MAX_PAYLOAD_BYTES = 1024
ALLOWED_URL_SCHEMES = ("https",)
_TEST_ONLY_URL_SCHEMES = ("https", "file")

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEME_RE = re.compile(r"^([a-z][a-z0-9+.-]*)://")


class PayloadError(ValueError):
    """A locally constructed payload violates the production wire policy."""


@dataclass(frozen=True)
class SubmissionRef:
    hotkey: str
    content_hash: str
    url: str
    block: int


def _url_allowed(url: object, *, schemes: tuple[str, ...]) -> bool:
    if (
        not isinstance(url, str)
        or not url
        or len(url) > 8_192
        or not url.isascii()
        or any(ord(char) <= 32 or ord(char) == 127 for char in url)
    ):
        return False
    scheme = _SCHEME_RE.match(url)
    if scheme is None or scheme.group(1) not in schemes:
        return False
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        return False
    if parsed.scheme == "https":
        return bool(parsed.hostname) and (port is None or 1 <= port <= 65_535)
    if parsed.scheme == "file":
        return parsed.netloc in ("", "localhost") and bool(parsed.path) and not parsed.query
    return False


def _encode_payload(content_hash: str, url: str, *, schemes: tuple[str, ...]) -> str:
    if _HASH_RE.fullmatch(content_hash or "") is None:
        raise PayloadError(
            f"content_hash must be 64 lowercase hex chars, got {content_hash!r}"
        )
    if not _url_allowed(url, schemes=schemes):
        raise PayloadError(f"url must be canonical and use one of {schemes}, got {url!r}")
    data = json.dumps(
        {"v": PAYLOAD_VERSION, "h": content_hash, "u": url},
        separators=(",", ":"),
    )
    if len(data.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise PayloadError(f"payload exceeds the {MAX_PAYLOAD_BYTES}-byte chain cap")
    return data


def encode_payload(content_hash: str, url: str) -> str:
    """Encode one production, HTTPS-only submission reference."""

    return _encode_payload(content_hash, url, schemes=ALLOWED_URL_SCHEMES)


def encode_payload_for_testing(content_hash: str, url: str) -> str:
    """Encode a hermetic-test reference which may additionally use ``file://``."""

    return _encode_payload(content_hash, url, schemes=_TEST_ONLY_URL_SCHEMES)


def _decode_payload(
    hotkey: str,
    block: int,
    data: object,
    *,
    schemes: tuple[str, ...],
) -> SubmissionRef | None:
    if not isinstance(data, str) or len(data.encode("utf-8", "replace")) > MAX_PAYLOAD_BYTES:
        logger.warning("payload from %s: oversized or non-string; ignored", hotkey)
        return None

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        obj = json.loads(
            data,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        logger.warning("payload from %s: not canonical JSON; ignored", hotkey)
        return None
    if (
        not isinstance(obj, dict)
        or set(obj) != {"v", "h", "u"}
        or type(obj.get("v")) is not int
        or obj.get("v") != PAYLOAD_VERSION
        or json.dumps(
            {"v": obj.get("v"), "h": obj.get("h"), "u": obj.get("u")},
            separators=(",", ":"),
        )
        != data
    ):
        logger.warning("payload from %s: schema/version/encoding mismatch; ignored", hotkey)
        return None
    content_hash, url = obj["h"], obj["u"]
    if not isinstance(content_hash, str) or _HASH_RE.fullmatch(content_hash) is None:
        logger.warning("payload from %s: bad content hash; ignored", hotkey)
        return None
    if not _url_allowed(url, schemes=schemes):
        logger.warning("payload from %s: bad/disallowed url; ignored", hotkey)
        return None
    if type(block) is not int or block < 0:
        logger.warning("payload from %s: invalid reveal block; ignored", hotkey)
        return None
    if (
        not isinstance(hotkey, str)
        or not hotkey
        or len(hotkey) > 256
        or hotkey.strip() != hotkey
        or any(char in hotkey for char in "\x00\r\n")
    ):
        logger.warning("payload contains an invalid hotkey identity; ignored")
        return None
    return SubmissionRef(hotkey, content_hash, url, block)


def decode_payload(hotkey: str, block: int, data: object) -> SubmissionRef | None:
    """Decode hostile production bytes under the HTTPS-only policy."""

    return _decode_payload(hotkey, block, data, schemes=ALLOWED_URL_SCHEMES)


def decode_payload_for_testing(
    hotkey: str, block: int, data: object
) -> SubmissionRef | None:
    """Decode bytes under the explicit hermetic-test transport policy."""

    return _decode_payload(hotkey, block, data, schemes=_TEST_ONLY_URL_SCHEMES)
