"""Chain I/O logic — pure helpers + RPC wrappers against a mock subtensor (no network)."""

from __future__ import annotations

import types

import pytest

from optima import chain


def _scale_reveal_frame(data: str) -> bytes:
    payload = data.encode("utf-8")
    size = len(payload)
    if size < 64:
        prefix = bytes([size << 2])
    elif size < 16_384:
        prefix = ((size << 2) | 1).to_bytes(2, "little")
    else:
        raise ValueError("test reveal exceeds the bounded production format")
    return prefix + payload


def _scale_reveal_hex(data: str) -> str:
    return "0x" + _scale_reveal_frame(data).hex()


# --- a minimal stand-in for bittensor's subtensor (records what was called) ---

class _MockMetagraph:
    def __init__(self, hotkeys, permits=None, last_updates=None, *, block=0):
        self.uids = list(range(len(hotkeys)))
        self.hotkeys = list(hotkeys)
        self.validator_permit = list(permits) if permits is not None else [True] * len(hotkeys)
        self.last_update = list(last_updates) if last_updates is not None else [0] * len(hotkeys)
        self.block = block


class _MockSubtensor:
    def __init__(self, *, hotkeys, commitments=None, revealed=None, block=100,
                 registered=None, events=None, weight_rows=None, last_updates=None):
        self._hotkeys = list(hotkeys)
        self._commitments = dict(commitments or {})
        self._revealed = dict(revealed or {})  # hotkey -> ((block, data), ...)
        self._block = block
        self._registered = set(hotkeys if registered is None else registered)
        self._events = events
        self._weight_rows = list(weight_rows or [])
        self._last_updates = last_updates
        self.substrate = types.SimpleNamespace(
            get_events=self._get_events,
            get_chain_finalised_head=lambda: self.get_block_hash(self._block),
            get_block_number=lambda block_hash: int(block_hash[2:], 16),
        )
        self.set_weights_calls: list[dict] = []
        self.metagraph_calls: list[int | None] = []
        self.weight_calls: list[int | None] = []
        self.set_commitment_calls: list[str] = []
        self.set_reveal_commitment_calls: list[tuple] = []

    def metagraph(self, netuid=None, block=None):
        self.metagraph_calls.append(block)
        return _MockMetagraph(
            self._hotkeys,
            last_updates=self._last_updates,
            block=self._block if block is None else block,
        )

    def weights(self, netuid=None, block=None):
        self.weight_calls.append(block)
        return list(self._weight_rows)

    def get_current_block(self):
        return self._block

    def get_block_hash(self, block):
        return "0x" + f"{block:064x}"

    def get_finalized_block_number(self):
        return self._block

    def get_all_commitments(self, netuid=None):
        return dict(self._commitments)

    def query_map(self, module, name, params=None, block=None):
        assert (module, name, params) == ("Commitments", "RevealedCommitments", [1])
        return [
            (
                hotkey,
                [
                    (_scale_reveal_hex(data), reveal_block)
                    for reveal_block, data in history
                    if block is None or reveal_block <= block
                ][-chain.CHAIN_REVEAL_HISTORY_CAP :],
            )
            for hotkey, history in self._revealed.items()
        ]

    def _get_events(self, *, block_hash):
        block = int(block_hash[2:], 16)
        if self._events is not None:
            return list(self._events.get(block, ()))
        rows = []
        for hotkey, history in self._revealed.items():
            for reveal_block, data in history:
                if reveal_block == block:
                    rows.append(_reveal_event(hotkey, data))
        return rows

    def set_reveal_commitment(self, *, wallet, netuid, data, blocks_until_reveal):
        self.set_reveal_commitment_calls.append((data, blocks_until_reveal))
        return True

    def set_weights(self, *, wallet, netuid, uids, weights, version_key,
                    wait_for_inclusion, wait_for_finalization):
        self.set_weights_calls.append({
            "uids": uids,
            "weights": weights,
            "version_key": version_key,
            "wait_for_inclusion": wait_for_inclusion,
            "wait_for_finalization": wait_for_finalization,
        })
        return True

    def is_hotkey_registered(self, *, hotkey_ss58, netuid):
        return hotkey_ss58 in self._registered

    def set_commitment(self, *, wallet, netuid, data):
        self.set_commitment_calls.append(data)
        return True


def _wallet(ss58: str):
    return types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address=ss58))


def _reveal_event(hotkey: str, data: str, *, netuid: int = 1, extrinsic: int = 0):
    return {
        "phase": {"ApplyExtrinsic": extrinsic},
        "event": {
            "module_id": "Commitments",
            "event_id": "CommitmentRevealed",
            "attributes": {
                "netuid": netuid,
                "who": hotkey,
            },
        },
    }


# ---- pure helpers ----

def test_normalize():
    assert chain.normalize({"a": 2, "b": 2}) == {"a": 0.5, "b": 0.5}
    assert chain.normalize({"a": 0, "b": -1}) == {}
    assert chain.normalize({}) == {}


def test_weights_map_to_uids():
    mg = chain.MetagraphView(1, 1, "h", uids=[0, 1, 2], hotkeys=["a", "b", "c"],
                             validator_permit=[True] * 3)
    assert chain.weights_to_uid_vector({"b": 1.0}, mg) == ([1], [1.0])


def test_weights_refuse_to_redistribute_deregistered_recipient():
    mg = chain.MetagraphView(1, 1, "h", uids=[5, 6], hotkeys=["a", "b"],
                             validator_permit=[True, True])
    with pytest.raises(chain.ChainWeightStateError, match="absent"):
        chain.weights_to_uid_vector({"a": 1.0, "ghost": 3.0}, mg)


def test_uid_of():
    mg = chain.MetagraphView(1, 1, "h", uids=[0, 1], hotkeys=["a", "b"])
    assert mg.uid_of("b") == 1 and mg.uid_of("ghost") is None


# ---- RPC wrappers (mock subtensor) ----

def test_fetch_metagraph():
    st = _MockSubtensor(hotkeys=["a", "b"], block=42)
    mg = chain.fetch_metagraph(st, netuid=1)
    assert mg.uids == [0, 1] and mg.hotkeys == ["a", "b"]
    assert mg.block == 42 and mg.block_hash == "0x" + f"{42:064x}"
    assert st.metagraph_calls == [42]


def test_fetch_metagraph_pins_finalized_height_and_rejects_false_provenance():
    st = _MockSubtensor(hotkeys=["a", "b"], block=42)
    st.get_current_block = lambda: 99
    mg = chain.fetch_metagraph(st, netuid=1)
    assert mg.block == 42
    assert st.metagraph_calls == [42]

    class Mislabelled(_MockSubtensor):
        def metagraph(self, netuid=None, block=None):
            result = super().metagraph(netuid=netuid, block=block)
            result.block = int(block) + 1
            return result

    with pytest.raises(chain.ChainWeightStateError, match="requested finalized block"):
        chain.fetch_metagraph(Mislabelled(hotkeys=["a"], block=42), netuid=1)


def test_sparse_weight_row_and_uid_mapping_share_one_finalized_block():
    calls: list[tuple[str, int | None]] = []

    class ReassignedAtBestHead:
        def get_finalized_block_number(self):
            return 100

        def get_current_block(self):
            return 101

        def get_block_hash(self, block):
            return "0x" + f"{block:064x}"

        def metagraph(self, *, netuid, block):
            calls.append(("metagraph", block))
            assert block == 100
            return types.SimpleNamespace(
                # UID 1 belongs to mallory at best head 101, but to alice at
                # the exact finalized authority used for this sparse row.
                uids=[0, 1],
                hotkeys=["validator", "alice"],
                last_update=[90, 0],
                validator_permit=[True, True],
                block=block,
            )

        def weights(self, *, netuid, block):
            calls.append(("weights", block))
            assert block == 100
            return [(0, [(1, 65_535)])]

    snapshot = chain.read_validator_weight_snapshot(
        ReassignedAtBestHead(), 1, "validator"
    )
    assert snapshot.weights == {"alice": 1.0}
    assert calls == [("metagraph", 100), ("weights", 100)]


def test_read_commitments():
    st = _MockSubtensor(hotkeys=["a", "b"], commitments={"a": "hashA", "b": "hashB"}, block=7)
    cs = chain.read_commitments(st, netuid=1)
    assert cs["a"].data == "hashA" and cs["a"].block == 7 and cs["b"].hotkey == "b"


def test_set_weights_dry_run_does_not_submit():
    st = _MockSubtensor(hotkeys=["a", "b"])
    res = chain.set_weights(st, wallet=None, netuid=1, weights_by_hotkey={"b": 1.0}, dry_run=True)
    assert res["submitted"] is False and res["dry_run"] is True
    assert res["uids"] == [1] and res["weights"] == [1.0]
    assert st.set_weights_calls == []  # nothing went to chain


def test_set_weights_submits_with_version_key():
    st = _MockSubtensor(hotkeys=["a", "b"])
    res = chain.set_weights(st, wallet=object(), netuid=1, weights_by_hotkey={"b": 1.0})
    assert res["submitted"] is True
    assert st.set_weights_calls == [
        {
            "uids": [1],
            "weights": [1.0],
            "version_key": chain.WEIGHTS_VERSION_KEY,
            "wait_for_inclusion": True,
            "wait_for_finalization": True,
        }
    ]


def test_set_weights_refuses_nonfinal_submission_mode_before_signing():
    st = _MockSubtensor(hotkeys=["a", "b"])
    with pytest.raises(chain.ChainWeightStateError, match="wait for inclusion and finalization"):
        chain.set_weights(
            st,
            wallet=object(),
            netuid=1,
            weights_by_hotkey={"b": 1.0},
            wait_for_finalization=False,
        )
    assert st.metagraph_calls == [] and st.set_weights_calls == []


def test_set_weights_chain_side_failure_reported():
    # An included extrinsic can still fail on-chain (rate limit / permit / CR
    # window) — submitted must reflect the chain's verdict, not the SDK call
    # returning (measured on 307: a rate-limited CR commit was silently inert).
    class _Failed:
        success = False
        message = "CommittingWeightsTooFast"

    st = _MockSubtensor(hotkeys=["a", "b"])
    st.set_weights = lambda **kw: _Failed()
    res = chain.set_weights(st, wallet=object(), netuid=1, weights_by_hotkey={"b": 1.0})
    assert res["submitted"] is False
    assert "TooFast" in res["message"]


def test_set_weights_deregistered_champion_does_not_submit():
    st = _MockSubtensor(hotkeys=["a", "b"])
    with pytest.raises(chain.ChainWeightStateError, match="absent"):
        chain.set_weights(
            st, wallet=object(), netuid=1, weights_by_hotkey={"ghost": 1.0}
        )
    assert st.set_weights_calls == []


def test_read_validator_weights_uses_authoritative_sparse_sdk_row():
    st = _MockSubtensor(
        hotkeys=["validator", "alice", "bob"],
        weight_rows=[(0, [(1, 10_000), (2, 30_000)])],
        last_updates=[91, 0, 0],
    )
    snapshot = chain.read_validator_weight_snapshot(st, 1, "validator")
    assert snapshot.last_update_block == 91
    assert snapshot.weights == pytest.approx({"alice": 0.25, "bob": 0.75})
    assert st.metagraph_calls == [100]
    assert st.weight_calls == [100]


def test_read_validator_weights_fails_closed_without_sparse_weight_api():
    st = _MockSubtensor(hotkeys=["validator"], last_updates=[1])
    st.weights = None
    with pytest.raises(chain.ChainWeightStateError, match="on-chain weights"):
        chain.read_validator_weight_snapshot(st, 1, "validator")


@pytest.mark.parametrize(
    "rows",
    [
        [(0, [(1, 1)]), (0, [(1, 1)])],
        [(0, [(1, 1), (1, 2)])],
        [(0, [(9, 1)])],
        [(0, [(1, 65_536)])],
    ],
)
def test_read_validator_weights_rejects_ambiguous_or_invalid_sparse_rows(rows):
    st = _MockSubtensor(
        hotkeys=["validator", "alice"], weight_rows=rows, last_updates=[1, 0]
    )
    with pytest.raises(chain.ChainWeightStateError):
        chain.read_validator_weight_snapshot(st, 1, "validator")


def test_post_commitment_dry_run_and_submit():
    st = _MockSubtensor(hotkeys=["a"])
    assert chain.post_commitment(st, None, 1, "thehash", dry_run=True)["submitted"] is False
    assert st.set_commitment_calls == []
    assert chain.post_commitment(st, object(), 1, "thehash")["submitted"] is True
    assert st.set_commitment_calls == ["thehash"]


def test_commitment_wrappers_report_failed_extrinsics_honestly():
    class _FailedResponse:
        success = False
        message = "wallet could not be deserialized"

    class _SucceededResponse:
        success = True
        message = ""

    for failed in (_FailedResponse(), (False, "rate limited"), None, False):
        st = _MockSubtensor(hotkeys=["a"])
        st.set_commitment = lambda *, wallet, netuid, data, _r=failed: _r
        st.set_reveal_commitment = (
            lambda *, wallet, netuid, data, blocks_until_reveal, _r=failed: _r
        )
        assert chain.post_commitment(st, object(), 1, "h")["submitted"] is False
        assert chain.post_reveal_commitment(st, object(), 1, "p")["submitted"] is False

    st = _MockSubtensor(hotkeys=["a"])
    st.set_commitment = lambda *, wallet, netuid, data: _SucceededResponse()
    st.set_reveal_commitment = (
        lambda *, wallet, netuid, data, blocks_until_reveal: (True, "")
    )
    assert chain.post_commitment(st, object(), 1, "h")["submitted"] is True
    assert chain.post_reveal_commitment(st, object(), 1, "p")["submitted"] is True


def test_read_revealed_commitments_takes_latest_per_hotkey():
    st = _MockSubtensor(hotkeys=["a", "b"], revealed={
        "a": ((5, "old"), (9, "new")),
        "b": ((7, "only"),),
        "c": (),  # a hotkey with an empty history is skipped
    })
    out = chain.read_revealed_commitments(st, netuid=1)
    assert out["a"].data == "new" and out["a"].block == 9
    assert out["b"].data == "only" and out["b"].block == 7
    assert "c" not in out


def test_raw_reveal_decoder_accepts_live_text_and_historical_hex_forms():
    live = (
        '{"v":1,"h":"af97a4f6656078784848976ecefc6ffda96bc95e20394b6f35c122a0d0cbc58c",'
        '"u":"https://github.com/latent-to/optima/releases/download/'
        "b300-testnet307-bfaa0511/miner_m3_blockscore_cute-"
        'af97a4f6656078784848976ecefc6ffda96bc95e20394b6f35c122a0d0cbc58c.tar.gz"}'
    )
    live_frame = _scale_reveal_frame(live)
    assert len(live.encode("utf-8")) == 260
    assert live_frame[:2] == b"\x11\x04"
    assert chain._decode_raw_reveal(live_frame.decode("utf-8")) == live
    st = _MockSubtensor(
        hotkeys=["alice"], block=12, revealed={"alice": ((9, live),)}
    )
    st.query_map = lambda **_kwargs: [("alice", [(live_frame.decode("utf-8"), 9)])]
    st.get_all_revealed_commitments = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("lossy SDK decoder must not be called")
    )
    assert chain.read_finalized_reveal_history(st, 1).reveals[0].data == live

    historical = "x" * 176
    historical_frame = _scale_reveal_frame(historical)
    assert historical_frame[:2] == b"\xc1\x02"
    assert chain._decode_raw_reveal("0x" + historical_frame.hex()) == historical


def test_raw_reveal_decoder_fails_closed_on_malformed_scale():
    for raw in (
        b"\x02",  # compact mode 2 cannot encode a bounded message
        b"\x03",  # compact mode 3 cannot encode a bounded message
        b"\x05\x00x",  # non-canonical mode 1 for a one-byte payload
        b"\x10abc",  # declared length four, only three bytes present
        b"\x04\xff",  # invalid UTF-8 payload
        "0xnot-hex",
        [4, 256],
    ):
        with pytest.raises(chain.ChainRevealHistoryError, match="SCALE bytes"):
            chain._decode_raw_reveal(raw)


def test_raw_reveal_decoder_fails_closed_on_ambiguous_candidates():
    with pytest.raises(chain.ChainRevealHistoryError, match="ambiguous"):
        chain._decode_reveal_candidates(
            (_scale_reveal_frame("first"), _scale_reveal_frame("second"))
        )


def test_raw_reveal_page_rejects_duplicate_or_malformed_rows():
    st = _MockSubtensor(hotkeys=["alice"], block=12)
    for page in (
        [("alice", [(_scale_reveal_hex("x"), 9)]), ("alice", [])],
        ["not-a-map-pair"],
        [("alice", object())],
        [("alice", [("only-one-field",)])],
        [("alice", [(9, _scale_reveal_hex("wrong-order"))])],
    ):
        st.query_map = lambda **_kwargs: page
        with pytest.raises(chain.ChainRevealHistoryError):
            chain._raw_reveal_page(st, 1, 12)


def test_raw_reveal_page_streams_and_enforces_both_budgets(monkeypatch):
    monkeypatch.setattr(chain, "MAX_REVEAL_HISTORY_ROWS", 2)
    consumed: list[int] = []

    def rows():
        for index in range(10):
            consumed.append(index)
            yield (f"hotkey-{index}", [])

    st = _MockSubtensor(hotkeys=[], block=12)
    st.query_map = lambda **_kwargs: rows()
    with pytest.raises(chain.ChainRevealHistoryError, match="map-row budget"):
        chain._raw_reveal_page(st, 1, 12)
    assert consumed == [0, 1, 2]

    st.query_map = lambda **_kwargs: [
        ("alice", [(object(), 1), (object(), 2), (object(), 3)])
    ]
    with pytest.raises(chain.ChainRevealHistoryError, match="reveal row budget"):
        chain._raw_reveal_page(st, 1, 12)


def test_raw_reveal_page_rejects_11_entries_before_payload_decode(monkeypatch):
    st = _MockSubtensor(hotkeys=["alice"], block=12)
    st.query_map = lambda **_kwargs: [
        ("alice", [(object(), block) for block in range(11)])
    ]
    monkeypatch.setattr(
        chain,
        "_decode_raw_reveal",
        lambda _raw: (_ for _ in ()).throw(AssertionError("payload decoded")),
    )
    with pytest.raises(chain.ChainRevealHistoryError, match="per-hotkey"):
        chain._raw_reveal_page(st, 1, 12)


def test_raw_reveal_page_normalizes_generator_exceptions():
    def broken_page():
        yield ("alice", [])
        raise RuntimeError("query page broke")

    st = _MockSubtensor(hotkeys=["alice"], block=12)
    st.query_map = lambda **_kwargs: broken_page()
    with pytest.raises(
        chain.ChainRevealHistoryError,
        match="cannot retrieve complete historical reveal state: RuntimeError",
    ):
        chain._raw_reveal_page(st, 1, 12)


def test_finalized_snapshot_uses_event_order_not_lexical_hotkey_order():
    events = {
        9: [
            _reveal_event("zeta", "first", extrinsic=4),
            _reveal_event("alpha", "second", extrinsic=5),
        ]
    }
    st = _MockSubtensor(
        hotkeys=["alpha", "zeta"],
        block=12,
        revealed={"alpha": ((9, "second"),), "zeta": ((9, "first"),)},
        events=events,
    )
    snapshot = chain.read_finalized_reveal_history(st, netuid=1)
    assert snapshot.finalized_block == 12
    assert snapshot.finalized_block_hash == "0x" + f"{12:064x}"
    assert [(row.hotkey, row.data, row.event_index) for row in snapshot.reveals] == [
        ("zeta", "first", 0),
        ("alpha", "second", 1),
    ]
    assert [row.extrinsic_index for row in snapshot.reveals] == [4, 5]
    assert all(row.block_hash == "0x" + f"{9:064x}" for row in snapshot.reveals)


def test_finalized_snapshot_scans_only_blocks_after_durable_cursor():
    st = _MockSubtensor(
        hotkeys=["alice"],
        block=12,
        revealed={"alice": ((7, "old"), (11, "new"))},
    )
    visited: list[int] = []
    original = st.substrate.get_events

    def get_events(*, block_hash):
        visited.append(int(block_hash[2:], 16))
        return original(block_hash=block_hash)

    st.substrate.get_events = get_events
    snapshot = chain.read_finalized_reveal_history(st, netuid=1, after_block=9)
    assert [(row.block, row.data) for row in snapshot.reveals] == [(11, "new")]
    assert visited == [10, 11, 12]


def test_stale_cursor_uses_complete_history_without_scanning_empty_blocks():
    st = _MockSubtensor(
        hotkeys=["alice"],
        block=100,
        revealed={"alice": ((7, "old"), (99, "new"))},
    )
    visited: list[int] = []
    original = st.substrate.get_events

    def get_events(*, block_hash):
        visited.append(int(block_hash[2:], 16))
        return original(block_hash=block_hash)

    st.substrate.get_events = get_events
    snapshot = chain.read_finalized_reveal_history(st, netuid=1, after_block=9)
    assert [(row.block, row.data) for row in snapshot.reveals] == [(99, "new")]
    assert visited == [99]


def test_same_hotkey_same_block_payloads_use_lexical_suborder_only():
    st = _MockSubtensor(
        hotkeys=["alice", "bob"],
        block=12,
        revealed={
            "alice": ((9, "z-payload"), (9, "a-payload")),
            "bob": ((9, "middle"),),
        },
        events={
            9: [
                _reveal_event("alice", "not-exposed"),
                _reveal_event("bob", "not-exposed"),
                _reveal_event("alice", "not-exposed"),
            ]
        },
    )
    rows = chain.read_reveal_history(st, 1)
    assert [(row.event_index, row.hotkey, row.data) for row in rows] == [
        (0, "alice", "a-payload"),
        (1, "bob", "middle"),
        (2, "alice", "z-payload"),
    ]


def test_finalized_history_paginates_and_rejects_same_block_overflow():
    history = tuple((block, f"payload-{block}") for block in range(1, 13))
    st = _MockSubtensor(hotkeys=["alice"], revealed={"alice": history}, block=20)
    assert [row.block for row in chain.read_reveal_history(st, 1)] == list(range(1, 13))

    same_block = tuple((10, f"same-{index}") for index in range(12))
    saturated = _MockSubtensor(
        hotkeys=["alice"], revealed={"alice": same_block}, block=20
    )
    with pytest.raises(chain.ChainRevealHistoryError, match="storage/event mismatch"):
        chain.read_reveal_history(saturated, 1)


def test_finalized_history_fails_closed_on_missing_or_extra_event():
    missing = _MockSubtensor(
        hotkeys=["alice"],
        revealed={"alice": ((9, "payload"),)},
        events={9: []},
    )
    with pytest.raises(chain.ChainRevealHistoryError, match="storage/event mismatch"):
        chain.read_finalized_reveal_history(missing, 1)

    extra = _MockSubtensor(
        hotkeys=["alice"],
        revealed={"alice": ((9, "payload"),)},
        events={9: [_reveal_event("alice", "payload"), _reveal_event("bob", "extra")]},
    )
    with pytest.raises(chain.ChainRevealHistoryError, match="storage/event mismatch"):
        chain.read_finalized_reveal_history(extra, 1)


def test_finalized_history_fails_closed_on_pruned_events_and_hash_mismatch():
    st = _MockSubtensor(hotkeys=["alice"], revealed={"alice": ((9, "payload"),)})
    st.substrate.get_events = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("state pruned")
    )
    with pytest.raises(chain.ChainRevealHistoryError, match="cannot read finalized events"):
        chain.read_finalized_reveal_history(st, 1)

    class Fallback(_MockSubtensor):
        get_finalized_block_number = None

    fallback = Fallback(hotkeys=[], revealed={}, block=12)
    fallback.substrate.get_chain_finalised_head = lambda: "0x" + "f" * 64
    fallback.substrate.get_block_number = lambda block_hash: 12
    with pytest.raises(chain.ChainRevealHistoryError, match="inconsistent"):
        chain.read_finalized_reveal_history(fallback, 1)


def test_finalized_history_rejects_unnamed_commitment_event_attributes():
    st = _MockSubtensor(
        hotkeys=["alice"],
        revealed={"alice": ((9, "payload"),)},
        events={
            9: [
                {
                    "phase": {"ApplyExtrinsic": 0},
                    "event": {
                        "module_id": "Commitments",
                        "event_id": "CommitmentRevealed",
                        "attributes": [1, "alice", "payload"],
                    },
                }
            ]
        },
    )
    with pytest.raises(chain.ChainRevealHistoryError, match="named object"):
        chain.read_finalized_reveal_history(st, 1)


def test_post_reveal_commitment_dry_run_and_submit():
    st = _MockSubtensor(hotkeys=["a"])
    res = chain.post_reveal_commitment(st, None, 1, "payload", dry_run=True)
    assert res["submitted"] is False and st.set_reveal_commitment_calls == []
    res = chain.post_reveal_commitment(st, object(), 1, "payload", blocks_until_reveal=10)
    assert res["submitted"] is True
    assert st.set_reveal_commitment_calls == [("payload", 10)]


def test_preflight_registered_with_permit():
    st = _MockSubtensor(hotkeys=["valX"], registered=["valX"])
    checks = chain.preflight(st, _wallet("valX"), netuid=1)
    assert checks[0].ok is True       # registered
    assert checks[1].ok is True       # permit (mock defaults to permitted)


def test_preflight_unregistered_short_circuits():
    st = _MockSubtensor(hotkeys=["other"], registered=[])
    checks = chain.preflight(st, _wallet("valX"), netuid=1)
    assert checks[0].ok is False and len(checks) == 1  # no permit check when unregistered
