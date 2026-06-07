"""Chain I/O logic — pure helpers + RPC wrappers against a mock subtensor (no network)."""

from __future__ import annotations

import types

from optima import chain


# --- a minimal stand-in for bittensor's subtensor (records what was called) ---

class _MockMetagraph:
    def __init__(self, hotkeys, permits=None):
        self.uids = list(range(len(hotkeys)))
        self.hotkeys = list(hotkeys)
        self.validator_permit = list(permits) if permits is not None else [True] * len(hotkeys)


class _MockSubtensor:
    def __init__(self, *, hotkeys, commitments=None, block=100, registered=None):
        self._hotkeys = list(hotkeys)
        self._commitments = dict(commitments or {})
        self._block = block
        self._registered = set(hotkeys if registered is None else registered)
        self.set_weights_calls: list[dict] = []
        self.set_commitment_calls: list[str] = []

    def metagraph(self, netuid=None):
        return _MockMetagraph(self._hotkeys)

    def get_current_block(self):
        return self._block

    def get_block_hash(self, block):
        return f"0xhash{block}"

    def get_all_commitments(self, netuid=None):
        return dict(self._commitments)

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


def test_preflight_registered_with_permit():
    st = _MockSubtensor(hotkeys=["valX"], registered=["valX"])
    checks = chain.preflight(st, _wallet("valX"), netuid=1)
    assert checks[0].ok is True       # registered
    assert checks[1].ok is True       # permit (mock defaults to permitted)


def test_preflight_unregistered_short_circuits():
    st = _MockSubtensor(hotkeys=["other"], registered=[])
    checks = chain.preflight(st, _wallet("valX"), netuid=1)
    assert checks[0].ok is False and len(checks) == 1  # no permit check when unregistered
