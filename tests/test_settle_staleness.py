"""King-of-the-hill: noise-confirmed scores + stale-champion (sglang-pin) handling.

Two behaviors pinned: (1) under the SAME pin a challenger must beat the champion by
the margin (unchanged); (2) when the champion was crowned under a DIFFERENT sglang pin
than the current one, its frozen speedup is not comparable to the new stock baseline, so
it must NOT gate the round — the best confident challenger re-establishes the title by
clearing the floor over current stock, and the result flags the staleness.
"""

from optima.commit_reveal import Ledger, make_commitment


def _commit_reveal_score(led: Ledger, hotkey: str, ch: str, rnd: int, score: float, pin: str):
    led.commit(hotkey, make_commitment(ch, hotkey, "s"), rnd)
    led.reveal(hotkey, ch, "s", rnd, fingerprint=ch)  # distinct fp per hash -> all original
    led.record_score(hotkey, ch, rnd, score, kl_mean=0.0, passed=True, sglang_version=pin)


def test_same_pin_challenger_must_beat_champion_by_margin():
    led = Ledger()
    _commit_reveal_score(led, "alice", "H_A", 0, 1.10, "0.5.12.post1")
    led.settle(0, margin=0.02, current_sglang_version="0.5.12.post1")
    assert led.champion and led.champion.hotkey == "alice"

    # A 1.11 challenger does NOT clear 1.10 * 1.02 = 1.122 -> no title change.
    _commit_reveal_score(led, "bob", "H_B", 1, 1.11, "0.5.12.post1")
    res = led.settle(1, margin=0.02, current_sglang_version="0.5.12.post1")
    assert not res.title_changed
    assert not res.champion_stale
    assert led.champion.hotkey == "alice"


def test_stale_champion_does_not_block_on_an_incomparable_frozen_score():
    led = Ledger()
    # Alice crowned under the OLD pin at a big 1.40 (vs old stock).
    _commit_reveal_score(led, "alice", "H_A", 0, 1.40, "0.5.11")
    led.settle(0, margin=0.02, current_sglang_version="0.5.11")
    assert led.champion.hotkey == "alice" and led.champion.score == 1.40

    # New pin: stock changed, so alice's frozen 1.40 is not comparable. Bob is a real
    # 1.05 win vs the NEW stock. He should re-crown (clears the floor over current stock),
    # not be blocked by 1.40 * 1.02.
    _commit_reveal_score(led, "bob", "H_B", 1, 1.05, "0.5.12.post1")
    res = led.settle(1, margin=0.02, current_sglang_version="0.5.12.post1")
    assert res.title_changed
    assert led.champion.hotkey == "bob"
    assert led.champion.sglang_version == "0.5.12.post1"


def test_stale_flag_raised_when_no_challenger_re_establishes():
    led = Ledger()
    _commit_reveal_score(led, "alice", "H_A", 0, 1.20, "0.5.11")
    led.settle(0, margin=0.02, current_sglang_version="0.5.11")
    # New pin, no challenger this round -> champion kept but FLAGGED stale for re-baseline.
    res = led.settle(1, margin=0.02, current_sglang_version="0.5.12.post1")
    assert res.champion_stale
    assert not res.title_changed
    assert led.champion.hotkey == "alice"


def test_noise_zero_score_never_crowns():
    led = Ledger()
    # A faithful-but-not-confident kernel records score 0.0 (the eval's crownable-or-0 rule).
    _commit_reveal_score(led, "alice", "H_A", 0, 0.0, "0.5.12.post1")
    res = led.settle(0, margin=0.02, current_sglang_version="0.5.12.post1")
    assert not res.title_changed
    assert led.champion is None
