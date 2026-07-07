"""The validator loop end-to-end against a mock subtensor: chain commitments in,
fetch + copy-detection + evaluation + settlement, weights out. No network, no GPU —
the evaluator is stubbed; subprocess evaluators get their own contract tests."""

from __future__ import annotations

import json

from optima.chain.fetch import package_bundle
from optima.chain.payload import encode_payload
from optima.chain.validator_loop import (
    EvalOutcome,
    command_evaluator,
    run_pass,
    run_validator,
)
from optima.commit_reveal import Ledger


# --------------------------------------------------------------------------- #
# fixtures: a mock subtensor + real mini-bundles served over file://
# --------------------------------------------------------------------------- #

class _MockMetagraph:
    def __init__(self, hotkeys):
        self.uids = list(range(len(hotkeys)))
        self.hotkeys = list(hotkeys)
        self.validator_permit = [True] * len(hotkeys)


class _MockSubtensor:
    def __init__(self, *, hotkeys, revealed=None, block=100):
        self._hotkeys = list(hotkeys)
        self.revealed = dict(revealed or {})  # hotkey -> ((block, data), ...)
        self._block = block
        self.set_weights_calls: list[dict] = []

    def metagraph(self, netuid=None):
        return _MockMetagraph(self._hotkeys)

    def get_current_block(self):
        return self._block

    def get_block_hash(self, block):
        return f"0xhash{block}"

    def get_all_revealed_commitments(self, netuid=None):
        return dict(self.revealed)

    def set_weights(self, *, wallet, netuid, uids, weights, version_key,
                    wait_for_inclusion, wait_for_finalization):
        self.set_weights_calls.append(
            {"uids": uids, "weights": weights, "version_key": version_key})
        return True


def _mini_bundle(root, name, body):
    """A minimal VALID bundle (manifest parses, fingerprints compute)."""
    b = root / name
    (b / "kernels").mkdir(parents=True)
    (b / "manifest.toml").write_text(
        f'bundle_id = "{name}"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[[ops]]\n"
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/k.py"\n'
        'entry = "k"\n'
        'dtypes = ["float32"]\n'
    )
    (b / "kernels" / "k.py").write_text(body)
    return b


def _submission(root, name, body):
    """Package a mini-bundle; return (hotkey-agnostic) payload pieces."""
    bundle = _mini_bundle(root / "src", name, body)
    archive, ch = package_bundle(bundle, root / "hosted" / f"{name}.tar.gz")
    return ch, archive.as_uri()


def _loop_env(tmp_path, revealed, hotkeys):
    st = _MockSubtensor(hotkeys=hotkeys, revealed=revealed, block=400)
    return st, dict(ledger_path=str(tmp_path / "ledger.json"),
                    bundles_dir=str(tmp_path / "cache"))


def _pass_all(bundle_dir):
    return EvalOutcome(True, 1.05)


# --------------------------------------------------------------------------- #
# the referee cycle
# --------------------------------------------------------------------------- #

def test_full_pass_crowns_and_pushes_weights(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, object(), 1, evaluator=_pass_all, **env)
    assert res.seen == 1 and res.new == [ch]
    assert res.evaluated == {ch: True}
    assert res.weights == {"miner1": 1.0}
    assert res.weights_pushed and len(st.set_weights_calls) == 1
    assert st.set_weights_calls[0]["uids"] == [1]

    # ledger state: score + audit record + champion
    led = Ledger.load(env["ledger_path"])
    assert led.is_known("miner1", ch)
    assert led.champions and led.current_weights() == {"miner1": 1.0}


def test_second_pass_is_idempotent(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    calls = {"n": 0}

    def counting(bundle_dir):
        calls["n"] += 1
        return EvalOutcome(True, 1.05)

    run_pass(st, object(), 1, evaluator=counting, **env)
    res2 = run_pass(st, object(), 1, evaluator=counting, **env)
    assert calls["n"] == 1          # not re-evaluated
    assert res2.new == []           # nothing new
    assert len(st.set_weights_calls) == 1  # unchanged weights not re-pushed


def test_copy_is_demoted_not_evaluated(tmp_path):
    body = "def k(x):\n    return x + 1\n"
    ch, url = _submission(tmp_path, "orig", body)
    # the copycat commits the SAME content at a LATER block
    revealed = {
        "author": ((5, encode_payload(ch, url)),),
        "copycat": ((9, encode_payload(ch, url)),),
    }
    st, env = _loop_env(tmp_path, revealed, hotkeys=["val", "author", "copycat"])
    evaluated = []

    def tracking(bundle_dir):
        evaluated.append(bundle_dir)
        return EvalOutcome(True, 1.05)

    res = run_pass(st, object(), 1, evaluator=tracking, **env)
    assert len(evaluated) == 1      # only the original ran
    assert res.copies == [ch]
    assert res.weights == {"author": 1.0}
    led = Ledger.load(env["ledger_path"])
    assert led.eval_for("copycat", ch).dq_reason == "copy"


def test_reformatted_copy_is_demoted_by_fingerprint(tmp_path):
    ch1, url1 = _submission(tmp_path, "orig", "def k(x):\n    return x + 1\n")
    # different bytes (comment + spacing) -> different content hash, same normalized code
    ch2, url2 = _submission(tmp_path, "theft",
                            "# totally my own work\ndef k(x):\n    return (x + 1)\n")
    assert ch1 != ch2
    revealed = {
        "author": ((5, encode_payload(ch1, url1)),),
        "copycat": ((9, encode_payload(ch2, url2)),),
    }
    st, env = _loop_env(tmp_path, revealed, hotkeys=["val", "author", "copycat"])
    res = run_pass(st, object(), 1, evaluator=_pass_all, **env)
    assert ch2 in res.copies
    assert res.weights == {"author": 1.0}


def test_fetch_failure_is_recorded_and_not_retried(tmp_path):
    ch = "c" * 64
    revealed = {"miner1": ((5, encode_payload(ch, "file:///nonexistent/x.tar.gz")),)}
    st, env = _loop_env(tmp_path, revealed, hotkeys=["val", "miner1"])
    res = run_pass(st, object(), 1, evaluator=_pass_all, **env)
    assert ch in res.rejected and not res.weights
    led = Ledger.load(env["ledger_path"])
    assert led.eval_for("miner1", ch).dq_reason.startswith("fetch:")
    # second pass skips it entirely (no infinite refetch of a dead URL)
    res2 = run_pass(st, object(), 1, evaluator=_pass_all, **env)
    assert res2.new == [] and res2.rejected == {}


def test_hash_mismatch_rejects_submission(tmp_path):
    _, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    lie = "d" * 64  # committed hash does not match the hosted artifact
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(lie, url)),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, object(), 1, evaluator=_pass_all, **env)
    assert lie in res.rejected and "mismatch" in res.rejected[lie]


def test_garbage_payload_is_ignored(tmp_path):
    st, env = _loop_env(tmp_path, {"miner1": ((5, "not json at all"),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, object(), 1, evaluator=_pass_all, **env)
    assert res.seen == 0 and res.new == []


def test_failed_gates_earn_no_weight(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, object(), 1,
                   evaluator=lambda p: EvalOutcome(False, 0.0, detail="verify failed"),
                   **env)
    assert res.evaluated == {ch: False}
    assert res.weights == {} and not st.set_weights_calls


def test_dry_run_weights_never_submits(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    res = run_pass(st, None, 1, evaluator=_pass_all, dry_run_weights=True, **env)
    assert res.weights == {"miner1": 1.0}
    assert not res.weights_pushed and not st.set_weights_calls


def test_weights_repushed_after_refresh_interval(tmp_path):
    ch, url = _submission(tmp_path, "m1", "def k(x):\n    return x\n")
    st, env = _loop_env(tmp_path, {"miner1": ((5, encode_payload(ch, url)),)},
                        hotkeys=["val", "miner1"])
    run_pass(st, object(), 1, evaluator=_pass_all, **env)
    st._block += 1000  # well past the refresh cadence
    run_pass(st, object(), 1, evaluator=_pass_all, **env)
    assert len(st.set_weights_calls) == 2


def test_run_validator_once_returns_pass_result(tmp_path):
    st, env = _loop_env(tmp_path, {}, hotkeys=["val"])
    res = run_validator(st, object(), 1, evaluator=_pass_all, once=True, **env)
    assert res is not None and res.seen == 0


# --------------------------------------------------------------------------- #
# subprocess evaluator contract
# --------------------------------------------------------------------------- #

def test_command_evaluator_report_and_exit_codes(tmp_path):
    bundle = tmp_path / "cache" / ("e" * 64)
    bundle.mkdir(parents=True)

    ok = command_evaluator("true")(bundle)
    assert ok.passed and ok.score == 1.0

    bad = command_evaluator("exit 3")(bundle)
    assert not bad.passed and bad.score == 0.0

    report = {"score": 1.07, "kl_mean": 0.002, "slot": "moe.fused_experts"}
    writer = tmp_path / "write_report.sh"
    writer.write_text(f"#!/bin/sh\necho '{json.dumps(report)}' > \"$1\"\n")
    writer.chmod(0o755)
    rich = command_evaluator(f"sh {writer} {{report}} # {{bundle}}")(bundle)
    assert rich.passed and rich.score == 1.07
    assert rich.kl_mean == 0.002 and rich.slot == "moe.fused_experts"
