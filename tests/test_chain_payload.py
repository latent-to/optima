"""Submission payload encode/decode — miner side fails loud, validator side fails quiet."""

from __future__ import annotations

import json

import pytest

from optima.chain.payload import (
    MAX_PAYLOAD_BYTES,
    PAYLOAD_VERSION,
    PayloadError,
    decode_payload,
    decode_payload_for_testing,
    encode_payload,
    encode_payload_for_testing,
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
                    '{"v":1,"v":1,"h":"' + HASH + '","u":"https://x"}',
                    '{"h":"' + HASH + '","u":"https://x","v":1}',
                    '{"v":1, "h":"' + HASH + '","u":"https://x"}',
                    '{"v":1,"h":"' + HASH + '","u":"https://x","x":1}',
                    "x" * (MAX_PAYLOAD_BYTES + 1)):
        assert decode_payload("hk", 1, garbage) is None
    assert decode_payload("hk", 1, None) is None  # type: ignore[arg-type]


def test_file_url_is_available_only_through_explicit_test_api():
    with pytest.raises(PayloadError):
        encode_payload(HASH, "file:///tmp/b.tar.gz")
    data = encode_payload_for_testing(HASH, "file:///tmp/b.tar.gz")
    assert decode_payload("hk", 7, data) is None
    ref = decode_payload_for_testing("hk", 7, data)
    assert ref is not None and ref.url.startswith("file://")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/x",
        "https://user@example.com/x",
        "https://user:pw@example.com/x",
        "https://example.com/x#fragment",
        "https://example.com:0/x",
        "https://example.com:65536/x",
        "https://example.com/has space",
        "https://example.com/\nnext",
        "HTTPS://example.com/x",
    ],
)
def test_production_payload_rejects_noncanonical_urls(url):
    with pytest.raises(PayloadError):
        encode_payload(HASH, url)


def test_decode_rejects_invalid_chain_identity_without_coercion():
    wire = encode_payload(HASH, "https://example.com/x")
    assert decode_payload(" hk", 1, wire) is None
    assert decode_payload("hk", True, wire) is None
    assert decode_payload("hk", -1, wire) is None
