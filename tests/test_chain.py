"""Chain I/O logic — pure helpers + RPC wrappers against a mock subtensor (no network)."""

from __future__ import annotations

import types

import pytest

from optima import chain


# --- a minimal stand-in for bittensor's subtensor (records what was called) ---

class _MockMetagraph:
    def __init__(self, hotkeys, permits=None):
        self.uids = list(range(len(hotkeys)))
        self.hotkeys = list(hotkeys)
        self.validator_permit = list(permits) if permits is not None else [True] * len(hotkeys)


class _MockSubtensor:
    def __init__(self, *, hotkeys, commitments=None, revealed=None, block=100,
                 registered=None):
        self._hotkeys = list(hotkeys)
        self._commitments = dict(commitments or {})
        self._revealed = dict(revealed or {})  # hotkey -> ((block, data), ...)
        self._block = block
        self._registered = set(hotkeys if registered is None else registered)
        self.set_weights_calls: list[dict] = []
        self.set_commitment_calls: list[str] = []
        self.set_reveal_commitment_calls: list[tuple] = []

    def metagraph(self, netuid=None):
        return _MockMetagraph(self._hotkeys)

    def get_current_block(self):
        return self._block

    def get_block_hash(self, block):
        return f"0xhash{block}"

    def get_all_commitments(self, netuid=None):
        return dict(self._commitments)

    def get_all_revealed_commitments(self, netuid=None):
        return dict(self._revealed)

    def set_reveal_commitment(self, *, wallet, netuid, data, blocks_until_reveal):
        self.set_reveal_commitment_calls.append((data, blocks_until_reveal))
        return True

    def set_weights(self, *, wallet, netuid, uids, weights, version_key,
                    wait_for_inclusion, wait_for_finalization):
        self.set_weights_calls.append({"uids": uids, "weights": weights, "version_key": version_key})
        return True

    def is_hotkey_registered(self, *, hotkey_ss58, netuid):
        return hotkey_ss58 in self._registered

    def set_commitment(self, *, wallet, netuid, data):
        self.set_commitment_calls.append(data)
        return True


def _wallet(ss58: str):
    return types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address=ss58))


# ---- pure helpers ----

def test_normalize():
    assert chain.normalize({"a": 2, "b": 2}) == {"a": 0.5, "b": 0.5}
    assert chain.normalize({"a": 0, "b": -1}) == {}
    assert chain.normalize({}) == {}


def test_weights_map_to_uids():
    mg = chain.MetagraphView(1, 1, "h", uids=[0, 1, 2], hotkeys=["a", "b", "c"],
                             validator_permit=[True] * 3)
    assert chain.weights_to_uid_vector({"b": 1.0}, mg) == ([1], [1.0])


def test_weights_drop_deregistered_and_renormalize():
    mg = chain.MetagraphView(1, 1, "h", uids=[5, 6], hotkeys=["a", "b"],
                             validator_permit=[True, True])
    # "ghost" isn't on the metagraph -> dropped; "a" renormalized to 1.0
    assert chain.weights_to_uid_vector({"a": 1.0, "ghost": 3.0}, mg) == ([5], [1.0])


def test_uid_of():
    mg = chain.MetagraphView(1, 1, "h", uids=[0, 1], hotkeys=["a", "b"])
    assert mg.uid_of("b") == 1 and mg.uid_of("ghost") is None


# ---- RPC wrappers (mock subtensor) ----

def test_fetch_metagraph():
    st = _MockSubtensor(hotkeys=["a", "b"], block=42)
    mg = chain.fetch_metagraph(st, netuid=1)
    assert mg.uids == [0, 1] and mg.hotkeys == ["a", "b"]
    assert mg.block == 42 and mg.block_hash == "0xhash42"


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
        {"uids": [1], "weights": [1.0], "version_key": chain.WEIGHTS_VERSION_KEY}
    ]


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
    res = chain.set_weights(st, wallet=object(), netuid=1, weights_by_hotkey={"ghost": 1.0})
    assert res["submitted"] is False and res["uids"] == []
    assert st.set_weights_calls == []


def test_post_commitment_dry_run_and_submit():
    st = _MockSubtensor(hotkeys=["a"])
    assert chain.post_commitment(st, None, 1, "thehash", dry_run=True)["submitted"] is False
    assert st.set_commitment_calls == []
    assert chain.post_commitment(st, object(), 1, "thehash")["submitted"] is True
    assert st.set_commitment_calls == ["thehash"]


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


def test_post_reveal_commitment_dry_run_and_submit():
    st = _MockSubtensor(hotkeys=["a"])
    res = chain.post_reveal_commitment(st, None, 1, "payload", dry_run=True)
    assert res["submitted"] is False and st.set_reveal_commitment_calls == []
    res = chain.post_reveal_commitment(st, object(), 1, "payload", blocks_until_reveal=10)
    assert res["submitted"] is True
    assert st.set_reveal_commitment_calls == [("payload", 10)]


def test_ledger_current_weights_is_the_policy_seam():
    from optima.commit_reveal import Champion, Ledger

    led = Ledger()
    assert led.current_weights() == {}
    led.champion = Champion("h1", "hkA", 1.05, 0)
    assert led.current_weights(per_slot=False) == {"hkA": 1.0}
    assert led.current_weights() == {"hkA": 1.0}  # no per-slot state -> falls back
    led.champions = {
        "slot.x": Champion("h1", "hkA", 1.05, 0),
        "slot.y": Champion("h2", "hkB", 1.10, 0),
        "slot.z": Champion("h3", "hkA", 1.02, 0),
    }
    w = led.current_weights()
    assert w["hkA"] == pytest.approx(2 / 3) and w["hkB"] == pytest.approx(1 / 3)


def test_preflight_registered_with_permit():
    st = _MockSubtensor(hotkeys=["valX"], registered=["valX"])
    checks = chain.preflight(st, _wallet("valX"), netuid=1)
    assert checks[0].ok is True       # registered
    assert checks[1].ok is True       # permit (mock defaults to permitted)


def test_preflight_unregistered_short_circuits():
    st = _MockSubtensor(hotkeys=["other"], registered=[])
    checks = chain.preflight(st, _wallet("valX"), netuid=1)
    assert checks[0].ok is False and len(checks) == 1  # no permit check when unregistered
