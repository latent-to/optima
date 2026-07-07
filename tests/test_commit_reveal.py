"""Tests for bundle hashing + commit-reveal + king-of-the-hill scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from optima.bundle_hash import content_hash
from optima.commit_reveal import Ledger, RevealError, make_commitment

TRITON = "examples/miner_silu_triton"
BROKEN = "examples/miner_silu_broken"


# ---- content hash ----


def test_content_hash_stable_and_distinct():
    h1 = content_hash(TRITON)
    h2 = content_hash(TRITON)
    assert h1 == h2 and len(h1) == 64
    assert content_hash(BROKEN) != h1  # different bundle -> different hash


def test_content_hash_changes_with_content(tmp_path: Path):
    b = tmp_path / "b"
    (b / "kernels").mkdir(parents=True)
    (b / "manifest.toml").write_text("bundle_id='x'\n")
    (b / "kernels" / "k.py").write_text("x = 1\n")
    h1 = content_hash(b)
    (b / "kernels" / "k.py").write_text("x = 2\n")
    assert content_hash(b) != h1


def test_content_hash_ignores_symlinks_so_identity_stays_in_bundle(tmp_path: Path):
    # A symlink must not fold an out-of-bundle file's bytes into the identity hash
    # (nor let the hash depend on the symlink target mutating). Two bundles with the
    # same real files hash identically regardless of a symlink's presence/target.
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 1\n")
    b = tmp_path / "b"
    (b / "kernels").mkdir(parents=True)
    (b / "manifest.toml").write_text("bundle_id='x'\n")
    (b / "kernels" / "k.py").write_text("x = 1\n")
    h_plain = content_hash(b)
    (b / "kernels" / "link.py").symlink_to(outside)
    assert content_hash(b) == h_plain  # symlink ignored
    outside.write_text("SECRET = 2\n")   # mutating the target does not move the hash
    assert content_hash(b) == h_plain


# ---- commit / reveal ----


def test_reveal_requires_matching_commitment():
    led = Ledger()
    ch = content_hash(TRITON)
    led.commit("alice", make_commitment(ch, "alice", "s3cr3t"), round_id=0)
    # wrong salt -> no matching commitment
    with pytest.raises(RevealError):
        led.reveal("alice", ch, "wrong-salt", round_id=0)
    # right salt -> accepted, original
    rev = led.reveal("alice", ch, "s3cr3t", round_id=0)
    assert rev.original


def test_cannot_reveal_without_committing():
    led = Ledger()
    ch = content_hash(TRITON)
    with pytest.raises(RevealError):
        led.reveal("mallory", ch, "salt", round_id=0)


def test_copy_detection_earliest_commit_wins():
    led = Ledger()
    ch = content_hash(TRITON)
    # alice commits first (seq 0), bob second (seq 1), both to the same content
    led.commit("alice", make_commitment(ch, "alice", "a"), 0)
    led.commit("bob", make_commitment(ch, "bob", "b"), 0)
    # bob reveals first, then alice
    bob_rev = led.reveal("bob", ch, "b", 0)
    alice_rev = led.reveal("alice", ch, "a", 0)
    assert alice_rev.original is True
    # bob is demoted to a copy because alice committed earlier
    assert bob_rev.original is False


# ---- settle / king of the hill ----


def _setup_scored(led, hotkey, content, score, *, salt="s", round_id=0, passed=True):
    led.commit(hotkey, make_commitment(content, hotkey, salt), round_id)
    led.reveal(hotkey, content, salt, round_id)
    led.record_score(hotkey, content, round_id, score, kl_mean=1e-3, passed=passed)


def test_first_champion_must_clear_margin():
    led = Ledger()
    _setup_scored(led, "alice", "hashA", 1.30)
    res = led.settle(0, margin=0.02)
    assert res.title_changed and res.champion.hotkey == "alice"
    assert res.weights == {"alice": 1.0}


def test_challenger_needs_to_beat_by_margin():
    led = Ledger()
    _setup_scored(led, "alice", "hashA", 1.30, round_id=0)
    led.settle(0, margin=0.05)
    # round 1: bob ties (within margin) -> no title change
    _setup_scored(led, "bob", "hashB", 1.33, round_id=1)
    res = led.settle(1, margin=0.05)  # needs >= 1.30*1.05 = 1.365
    assert not res.title_changed and res.champion.hotkey == "alice"
    # round 2: carol clearly beats -> takes title
    _setup_scored(led, "carol", "hashC", 1.50, round_id=2)
    res2 = led.settle(2, margin=0.05)
    assert res2.title_changed and res2.champion.hotkey == "carol"


def test_copy_earns_nothing_at_settle():
    led = Ledger()
    # alice (original) and bob (copy of same content) both committed; alice first
    ch = "sharedhash"
    led.commit("alice", make_commitment(ch, "alice", "a"), 0)
    led.commit("bob", make_commitment(ch, "bob", "b"), 0)
    led.reveal("alice", ch, "a", 0)
    led.reveal("bob", ch, "b", 0)  # demoted to copy
    led.record_score("alice", ch, 0, 1.40, 1e-3, True)
    led.record_score("bob", ch, 0, 1.40, 1e-3, True)
    res = led.settle(0, margin=0.02)
    assert res.champion.hotkey == "alice"
    assert "bob" in res.rejected_copies


def test_failed_quality_cannot_win():
    led = Ledger()
    _setup_scored(led, "alice", "hashA", 9.9, passed=False)  # huge speedup but cheated
    res = led.settle(0, margin=0.02)
    assert res.champion is None and res.weights == {}


def test_ledger_roundtrip(tmp_path: Path):
    led = Ledger()
    _setup_scored(led, "alice", "hashA", 1.30)
    led.settle(0)
    p = tmp_path / "led.json"
    led.save(p)
    led2 = Ledger.load(p)
    assert led2.champion and led2.champion.hotkey == "alice"
    assert len(led2.commitments) == 1 and len(led2.reveals) == 1
