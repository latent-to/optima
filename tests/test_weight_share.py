"""Tests for permit-gated current-weight sharing and follower publish reuse."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima import chain
from optima.chain.weight_share import (
    CURRENT_WEIGHTS_PATH,
    CurrentWeightOffer,
    WeightShareError,
    assert_fresh_timestamp,
    assert_validator_permit,
    build_signed_offer_response,
    default_offer_path,
    fetch_current_weights,
    parse_signed_offer_response,
    publish_followed_weights,
    read_current_weight_offer,
    rebind_projection_signer,
    request_auth_digest,
    serve_current_weights,
    sign_auth_digest,
    write_current_weight_offer,
)
from optima.chain.weights import WeightProjection
from optima.stack_identity import canonical_digest, sha256_hex


def _d(label: str) -> str:
    return sha256_hex(label.encode())


def _projection(*, hotkey: str = "authority", block: int = 10) -> WeightProjection:
    scope = _d("scope")
    metagraph_digest = canonical_digest(
        "optima.economics.metagraph-membership",
        {
            "block": block,
            "block_hash": "0x" + f"{block:064x}",
            "chain_scope_digest": scope,
            "members": [
                {"hotkey": "authority", "uid": 0},
                {"hotkey": "follower", "uid": 1},
                {"hotkey": "miner", "uid": 2},
            ],
        },
    )
    return WeightProjection(
        scope,
        307,
        hotkey,
        _d("policy"),
        _d("settlement"),
        _d("evaluation"),
        metagraph_digest,
        (_d("arena"),),
        1,
        block,
        1,
        (_d("evidence"),),
        (("miner", 1_000_000),),
    )


class _FakeHotkey:
    def __init__(self, address: str, secret: bytes) -> None:
        self.ss58_address = address
        self._secret = secret

    def sign(self, data: bytes) -> bytes:
        return hashlib.sha256(self._secret + data).digest()


def _verify_factory(secrets: dict[str, bytes]):
    def verify(hotkey: str, message: bytes, signature: bytes) -> bool:
        secret = secrets.get(hotkey)
        if secret is None:
            return False
        return signature == hashlib.sha256(secret + message).digest()

    return verify


def _view(block: int = 10) -> chain.MetagraphView:
    return chain.MetagraphView(
        307,
        block,
        "0x" + f"{block:064x}",
        [0, 1, 2],
        ["authority", "follower", "miner"],
        [True, True, False],
        [0, 0, 0],
    )


def test_offer_roundtrip_and_default_path(tmp_path: Path) -> None:
    intake = tmp_path / "intake.sqlite3"
    path = default_offer_path(intake)
    assert path.name == "intake.sqlite3.current_weights.json"
    projection = _projection()
    write_current_weight_offer(path, projection)
    offer = read_current_weight_offer(path)
    assert offer.projection == projection
    assert offer.digest == projection.digest
    raw = json.loads(path.read_text())
    assert raw["schema"] == "optima.current-weight-offer.v1"
    assert raw["projection_digest"] == projection.digest


def test_rebind_keeps_weights_changes_signer() -> None:
    projection = _projection()
    rebound = rebind_projection_signer(projection, "follower")
    assert rebound.validator_hotkey == "follower"
    assert rebound.weights_ppm == projection.weights_ppm
    assert rebound.digest != projection.digest
    assert rebind_projection_signer(projection, "authority") is projection


def test_timestamp_skew_and_permit_gates() -> None:
    assert_fresh_timestamp(100, now=130, max_skew_seconds=60)
    with pytest.raises(WeightShareError, match="skew"):
        assert_fresh_timestamp(100, now=200, max_skew_seconds=60)
    view = _view()
    assert assert_validator_permit(view, "authority") == 0
    with pytest.raises(WeightShareError, match="validator_permit"):
        assert_validator_permit(view, "miner")
    with pytest.raises(WeightShareError, match="not registered"):
        assert_validator_permit(view, "missing")


def test_signed_response_rejects_tampered_body() -> None:
    secrets = {"authority": b"auth-secret"}
    verify = _verify_factory(secrets)
    authority = _FakeHotkey("authority", secrets["authority"])
    offer = CurrentWeightOffer(_projection())
    body, headers = build_signed_offer_response(
        offer, authority=authority, netuid=307, timestamp=1_700_000_000
    )
    parsed = parse_signed_offer_response(
        body,
        headers,
        netuid=307,
        now=1_700_000_000,
        max_skew_seconds=60,
        verify=verify,
        metagraph=_view(),
    )
    assert parsed.projection.digest == offer.digest

    tampered = body.replace(b"miner", b"Miner")
    with pytest.raises(WeightShareError, match="digest mismatch|signature"):
        parse_signed_offer_response(
            tampered,
            headers,
            netuid=307,
            now=1_700_000_000,
            max_skew_seconds=60,
            verify=verify,
            metagraph=_view(),
        )


def test_http_endpoint_requires_permit_and_fresh_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = {"authority": b"auth", "follower": b"follow", "miner": b"miner"}
    verify = _verify_factory(secrets)
    authority = _FakeHotkey("authority", secrets["authority"])
    follower = _FakeHotkey("follower", secrets["follower"])
    miner = _FakeHotkey("miner", secrets["miner"])
    offer_path = tmp_path / "offer.json"
    write_current_weight_offer(offer_path, _projection())

    monkeypatch.setattr(chain, "fetch_metagraph", lambda *_a, **_k: _view())
    server = serve_current_weights(
        host="127.0.0.1",
        port=0,
        offer_path=offer_path,
        authority=authority,
        subtensor=object(),
        netuid=307,
        max_skew_seconds=60,
        verify=verify,
        clock=lambda: 1_700_000_100,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        base = f"http://127.0.0.1:{port}"
        offer = fetch_current_weights(
            base,
            signer=follower,
            netuid=307,
            max_skew_seconds=60,
            verify=verify,
            clock=lambda: 1_700_000_100,
            expected_authority="authority",
            metagraph=_view(),
        )
        assert offer.projection.weights_ppm == (("miner", 1_000_000),)

        with pytest.raises(WeightShareError, match="rejected|permit|validator_permit"):
            fetch_current_weights(
                base,
                signer=miner,
                netuid=307,
                max_skew_seconds=60,
                verify=verify,
                clock=lambda: 1_700_000_100,
                metagraph=_view(),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fetch_binds_request_signature_over_timestamp() -> None:
    secrets = {"follower": b"follow"}
    follower = _FakeHotkey("follower", secrets["follower"])
    captured: dict[str, object] = {}

    class _Resp:
        status = 200
        headers = {
            "X-Optima-Authority-Hotkey": "authority",
            "X-Optima-Netuid": "307",
            "X-Optima-Timestamp": "100",
            "X-Optima-Signature": "00",
            "X-Optima-Body-Digest": "00",
        }

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def opener(request, timeout=30):
        captured["headers"] = dict(request.headers.items())
        captured["timeout"] = timeout
        return _Resp()

    with pytest.raises(WeightShareError):
        fetch_current_weights(
            "http://example.test",
            signer=follower,
            netuid=307,
            clock=lambda: 100,
            opener=opener,
            verify=lambda *_a: False,
        )
    headers = {str(k).lower(): v for k, v in captured["headers"].items()}  # type: ignore[union-attr]
    assert headers["x-optima-hotkey"] == "follower"
    assert headers["x-optima-timestamp"] == "100"
    digest = request_auth_digest(
        hotkey="follower",
        method="GET",
        netuid=307,
        path=CURRENT_WEIGHTS_PATH,
        timestamp=100,
    )
    assert headers["x-optima-signature"] == sign_auth_digest(follower, digest)


class _Journal:
    def __init__(self) -> None:
        self.row = None
        self.history = []

    def load(self):
        return self.row

    def compare_and_swap(self, expected_record_digest, replacement):
        assert expected_record_digest == (self.row.digest if self.row else None)
        self.row = replacement
        self.history.append(replacement)

    def retained_projection(self, projection_digest):
        raise AssertionError("not used")


def test_publish_followed_weights_uses_reconciler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offer = CurrentWeightOffer(_projection())
    wallet = SimpleNamespace(hotkey=_FakeHotkey("follower", b"f"))
    journal = _Journal()
    seen = {}

    def fake_reconcile(
        subtensor,
        signer_wallet,
        projection,
        journal_arg,
        *,
        refresh_blocks,
        dry_run=False,
        reconcile_only=False,
        allow_stale_initial=False,
        require_current_crown=True,
    ):
        seen["projection"] = projection
        seen["wallet"] = signer_wallet
        seen["refresh_blocks"] = refresh_blocks
        seen["dry_run"] = dry_run
        seen["require_current_crown"] = require_current_crown
        assert journal_arg is journal
        return SimpleNamespace(
            projection_digest=projection.digest,
            status="dry_run",
            chain_matches=False,
            submitted=False,
            refresh_due=False,
        )

    monkeypatch.setattr(
        "optima.chain.weight_share.reconcile_weight_publication",
        fake_reconcile,
    )
    result = publish_followed_weights(
        subtensor=object(),
        signer_wallet=wallet,
        offer=offer,
        journal=journal,
        refresh_blocks=100,
        dry_run=True,
    )
    assert result.status == "dry_run"
    assert seen["projection"].validator_hotkey == "follower"
    assert seen["projection"].weights_ppm == offer.projection.weights_ppm
    assert seen["wallet"] is None
    assert seen["dry_run"] is True
    assert seen["require_current_crown"] is True


def test_set_weights_persists_offer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import optima.cli as cli
    from optima.chain.intake import FinalizedIntakeStore, IntakeScope

    scope = IntakeScope("0x" + "0" * 64, 307)
    path = tmp_path / "private" / "intake.sqlite3"
    with FinalizedIntakeStore(path, scope=scope):
        pass

    projection = _projection(hotkey="validator")
    offer_path = tmp_path / "offer.json"

    class _Subtensor:
        def get_block_hash(self, block):
            return "0x" + "0" * 64

    monkeypatch.setattr(chain, "connect", lambda _network: _Subtensor())
    monkeypatch.setattr(
        chain,
        "fetch_metagraph",
        lambda *_a, **_k: chain.MetagraphView(
            307,
            10,
            "0x" + f"{10:064x}",
            [0, 1],
            ["validator", "miner"],
            [True, True],
            [0, 0],
        ),
    )

    def fake_build(self, **_kwargs):
        return projection

    monkeypatch.setattr(
        FinalizedIntakeStore, "build_burn_weight_projection", fake_build
    )

    def fake_reconcile(*_a, **_k):
        return SimpleNamespace(
            status="dry_run",
            chain_matches=False,
            submitted=False,
            refresh_due=False,
        )

    monkeypatch.setattr(
        "optima.chain.weights.reconcile_weight_publication", fake_reconcile
    )
    monkeypatch.setattr(
        cli,
        "_cmd_set_weights_once",
        cli._cmd_set_weights_once,
    )

    import sys
    import types

    class _Hotkey:
        ss58_address = "validator"

    class _Wallet:
        def __init__(self, name, hotkey):
            self.hotkey = _Hotkey()

    monkeypatch.setitem(
        sys.modules, "bittensor", types.SimpleNamespace(Wallet=_Wallet)
    )

    args = SimpleNamespace(
        reconcile_only=False,
        dry_run=True,
        release_hold="",
        burn_hotkey="miner",
        validator_hotkey="",
        wallet="default",
        hotkey="default",
        intake_db=str(path),
        netuid=307,
        network="finney",
        half_life_blocks=100,
        discovery_lifetime_blocks=20,
        discovery_pool_ppm=100_000,
        refresh_blocks=10,
        weight_offer_path=str(offer_path),
        watch=False,
    )
    assert cli.cmd_set_weights(args) == 0
    stored = read_current_weight_offer(offer_path)
    assert stored.projection.digest == projection.digest
