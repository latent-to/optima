"""Submission payload encode/decode — miner side fails loud, validator side fails quiet."""

from __future__ import annotations

import json

import pytest

from optima.chain.payload import (
    MAX_PAYLOAD_BYTES,
    PAYLOAD_VERSION,
    PayloadError,
    decode_payload,
    encode_payload,
)

HASH = "a" * 64


def test_roundtrip():
    data = encode_payload(HASH, "https://example.com/b.tar.gz")
    ref = decode_payload("hk1", 42, data)
    assert ref is not None
    assert ref.content_hash == HASH and ref.url == "https://example.com/b.tar.gz"
    assert ref.hotkey == "hk1" and ref.block == 42


def test_encode_rejects_bad_hash_and_scheme():
    with pytest.raises(PayloadError):
        encode_payload("nothex", "https://x")
    with pytest.raises(PayloadError):
        encode_payload(HASH.upper(), "https://x")  # uppercase hex is not canonical
    with pytest.raises(PayloadError):
        encode_payload(HASH, "ftp://example.com/b.tar.gz")
    with pytest.raises(PayloadError):
        encode_payload(HASH, "not-a-url")


def test_encode_rejects_oversize():
    with pytest.raises(PayloadError):
        encode_payload(HASH, "https://example.com/" + "x" * MAX_PAYLOAD_BYTES)


def test_decode_never_raises_on_garbage():
    for garbage in ("", "not json", "[]", '{"v":99,"h":"x","u":"y"}',
                    json.dumps({"v": PAYLOAD_VERSION}),  # missing fields
                    json.dumps({"v": PAYLOAD_VERSION, "h": "short", "u": "https://x"}),
                    json.dumps({"v": PAYLOAD_VERSION, "h": HASH, "u": "javascript:x"}),
                    json.dumps({"v": PAYLOAD_VERSION, "h": HASH, "u": 7}),
                    "x" * (MAX_PAYLOAD_BYTES + 1)):
        assert decode_payload("hk", 1, garbage) is None
    assert decode_payload("hk", 1, None) is None  # type: ignore[arg-type]


def test_decode_accepts_file_url_for_dev_loops():
    data = encode_payload(HASH, "file:///tmp/b.tar.gz")
    ref = decode_payload("hk", 7, data)
    assert ref is not None and ref.url.startswith("file://")
