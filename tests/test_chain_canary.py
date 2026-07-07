"""Chain-SDK canary — pure introspection logic against a mocked bittensor.

No real bittensor needed: we inject a fake module into sys.modules so the
introspection path is exercised on CI / a laptop. The real run (``optima
chain-compat``) introspects the actually-installed SDK on the validator box.
"""

from __future__ import annotations

import sys
import types

from optima.chain_canary import Check, format_checks, run_checks


def test_format_checks_green():
    out = format_checks([Check("a", True), Check("b", True, "detail")])
    assert "CHAIN SDK API PRESENT" in out
    assert "[ok  ] a" in out


def test_format_checks_red():
    out = format_checks([Check("a", True), Check("b", False, "missing")])
    assert "FAIL" in out
    assert "1 CHECK(S) FAILED" in out


def test_missing_bittensor_is_graceful(monkeypatch):
    monkeypatch.setitem(sys.modules, "bittensor", None)  # `import bittensor` -> ImportError
    checks = run_checks()
    assert checks[0].name == "import bittensor"
    assert checks[0].ok is False


def _fake_bittensor(methods: tuple[str, ...]) -> types.ModuleType:
    mod = types.ModuleType("bittensor")
    mod.__version__ = "9.9.9-fake"
    wcls = type("Wallet", (), {})
    mod.Wallet = wcls
    mod.wallet = wcls
    ns = {m: (lambda self, *a, **k: None) for m in methods}
    cls = type("Subtensor", (), ns)
    mod.Subtensor = cls
    mod.subtensor = cls
    return mod


def test_full_fake_sdk_passes(monkeypatch):
    fake = _fake_bittensor((
        "set_weights", "metagraph", "get_all_commitments", "set_commitment",
        "set_reveal_commitment", "get_all_revealed_commitments",
        "is_hotkey_registered", "burned_register", "get_current_block", "get_block_hash",
        "commit", "reveal_commitment",
    ))
    monkeypatch.setitem(sys.modules, "bittensor", fake)
    by_name = {c.name: c for c in run_checks()}
    assert by_name["bittensor installed"].ok
    assert by_name["subtensor.set_weights"].ok
    assert by_name["subtensor.metagraph"].ok
    assert by_name["commitment/reveal API present"].ok   # found commit + reveal_commitment
    assert by_name["weights API present"].ok             # found set_weights
    assert all(c.ok for c in by_name.values())


def test_missing_method_is_flagged(monkeypatch):
    fake = _fake_bittensor(("metagraph", "is_hotkey_registered", "get_current_block"))
    monkeypatch.setitem(sys.modules, "bittensor", fake)
    by_name = {c.name: c for c in run_checks()}
    assert by_name["subtensor.set_weights"].ok is False
    assert "MISSING" in by_name["subtensor.set_weights"].detail
    assert not all(c.ok for c in by_name.values())
