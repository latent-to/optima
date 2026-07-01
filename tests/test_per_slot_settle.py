"""Per-slot championships (report misalignment M4): pay specialists, not only the
single best end-to-end bundle. One champion per slot; emission split across slots."""

from optima.commit_reveal import Ledger, make_commitment


def _score(led, hotkey, ch, slot, score, *, rnd=0, pin="0.5.12.post1", passed=True):
    led.commit(hotkey, make_commitment(ch, hotkey, "s"), rnd)
    led.reveal(hotkey, ch, "s", rnd, fingerprint=ch)
    led.record_score(hotkey, ch, rnd, score, kl_mean=0.0, passed=passed, sglang_version=pin, slot=slot)


def test_two_specialists_split_emission_across_slots():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.20)
    _score(led, "bob", "H_B", "collective.all_reduce", 1.15)
    res = led.settle_per_slot(0, margin=0.02, current_sglang_version="0.5.12.post1")
    assert res.champions["moe.fused_experts"].hotkey == "alice"
    assert res.champions["collective.all_reduce"].hotkey == "bob"
    # Each owns one of two slots -> 50/50 split (winner-take-all would have given alice 100%).
    assert res.weights == {"alice": 0.5, "bob": 0.5}


def test_one_hotkey_owning_two_slots_gets_full_share():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.20)
    _score(led, "alice", "H_A2", "norm.rmsnorm", 1.10)
    res = led.settle_per_slot(0, margin=0.02, current_sglang_version="0.5.12.post1")
    assert res.weights == {"alice": 1.0}


def test_per_slot_king_of_the_hill_within_slot():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.30, rnd=0)
    led.settle_per_slot(0, margin=0.05, current_sglang_version="0.5.12.post1")
    # A weak challenger in the SAME slot doesn't clear the margin.
    _score(led, "bob", "H_B", "moe.fused_experts", 1.33, rnd=1)
    res = led.settle_per_slot(1, margin=0.05, current_sglang_version="0.5.12.post1")
    assert res.champions["moe.fused_experts"].hotkey == "alice"
    assert not res.title_changes.get("moe.fused_experts")


def test_per_slot_copy_excluded():
    led = Ledger()
    ch = "shared"
    led.commit("alice", make_commitment(ch, "alice", "a"), 0)
    led.commit("bob", make_commitment(ch, "bob", "b"), 0)
    led.reveal("alice", ch, "a", 0, fingerprint=ch)
    led.reveal("bob", ch, "b", 0, fingerprint=ch)  # demoted
    led.record_score("alice", ch, 0, 1.40, 0.0, True, slot="moe.fused_experts")
    led.record_score("bob", ch, 0, 1.40, 0.0, True, slot="moe.fused_experts")
    res = led.settle_per_slot(0, margin=0.02)
    assert res.champions["moe.fused_experts"].hotkey == "alice"
    assert "bob" in res.rejected_copies


def test_per_slot_persists_across_load(tmp_path):
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.20)
    led.settle_per_slot(0, current_sglang_version="0.5.12.post1")
    p = tmp_path / "l.json"
    led.save(p)
    led2 = Ledger.load(p)
    assert led2.champions["moe.fused_experts"].hotkey == "alice"


def test_stale_champion_flagged_even_when_its_slot_gets_no_submissions():
    # A pin bump makes alice's frozen score incomparable; her slot receives no
    # challengers this round (only ANOTHER slot does), but she still holds emission —
    # the slot must be flagged stale for re-baseline anyway.
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.20, pin="0.5.11")
    led.settle_per_slot(0, current_sglang_version="0.5.11")
    _score(led, "bob", "H_B", "norm.rmsnorm", 1.10, rnd=1, pin="0.5.12")
    res = led.settle_per_slot(1, current_sglang_version="0.5.12")
    assert "moe.fused_experts" in res.stale_slots
    # And with NO submissions anywhere the flag still raises.
    res2 = led.settle_per_slot(2, current_sglang_version="0.5.12")
    assert "moe.fused_experts" in res2.stale_slots
