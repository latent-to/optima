"""Arena registry + per-model KL-floor override + per-arena settle isolation.

Arenas make "try a new model" a config row (its sglang pin, image, seam subset, KL
floors, engine kwargs) without disturbing the validated default path. Pin: the default
arena equals pre-arena behavior; a non-default arena's KL floor overrides the slot
default; and scores from different arenas don't share a championship bracket.
"""

import argparse

import pytest

from optima.arenas import ARENAS, DEFAULT_ARENA, Arena, arena_for_model, get_arena, list_arenas
from optima.commit_reveal import Ledger, make_commitment


def test_default_arena_equals_pre_arena_pin():
    # PINNED_SGLANG must alias the default arena's version (back-compat for importers).
    from optima.compat import PINNED_SGLANG
    assert DEFAULT_ARENA.sglang_version == PINNED_SGLANG
    assert get_arena(None) is DEFAULT_ARENA
    assert get_arena("default") is DEFAULT_ARENA


def test_unknown_arena_raises_with_known_list():
    with pytest.raises(KeyError, match="unknown arena"):
        get_arena("does-not-exist")


def test_seam_subset_semantics():
    full = Arena(name="f", model_path="m", sglang_version="x")  # empty subset = all
    assert full.applies_seam("moe") and full.applies_seam("collective")
    subset = Arena(name="s", model_path="m", sglang_version="x", seam_adapters=("attention", "moe"))
    assert subset.applies_seam("attention") and subset.applies_seam("moe")
    assert not subset.applies_seam("collective")


def test_kl_floor_override_and_competable():
    a = Arena(name="m3", model_path="MiniMax/M3", sglang_version="0.5.13",
              kl_floors={"attention.decode": 0.04})
    assert a.kl_floor_for("attention.decode") == 0.04
    assert a.kl_floor_for("norm.rmsnorm") is None  # falls back to slot/CLI
    assert a.competable()
    stub = Arena(name="stub", model_path="x", sglang_version="")  # declared, no pin yet
    assert not stub.competable()


def test_compat_runs_per_arena_seam_subset():
    # A subset arena's canary only iterates its seams (table-driven loop). We assert the
    # subset is honored by checking the labels mention only the subset's chokepoints.
    from optima.compat import run_checks
    subset = Arena(name="attn-only", model_path="m", sglang_version="z", seam_adapters=("attention",))
    checks = run_checks(subset)
    names = " ".join(c.name for c in checks)
    # sglang isn't installed in CI -> import fails fast; but the arena label must appear.
    assert "attn-only" in names
    # When sglang IS importable, the table loop would skip non-attention seams; we at least
    # confirm no 'seam table: moe' check is emitted for an attention-only arena.
    assert "seam table: moe" not in names


def _score(led, hotkey, ch, slot, score, arena, *, rnd=0, pin="0.5.12.post1"):
    led.commit(hotkey, make_commitment(ch, hotkey, "s"), rnd)
    led.reveal(hotkey, ch, "s", rnd, fingerprint=ch)
    led.record_score(hotkey, ch, rnd, score, kl_mean=0.0, passed=True, sglang_version=pin,
                     slot=slot, arena=arena)


def test_settle_arena_filter_isolates_brackets():
    led = Ledger()
    # Same slot, two arenas: a big speedup on model B must NOT beat model A's champion.
    _score(led, "alice", "H_A", "moe.fused_experts", 1.10, "gpt-oss")
    _score(led, "bob", "H_B", "moe.fused_experts", 1.90, "minimax-m3")
    res_a = led.settle(0, margin=0.02, arena="gpt-oss")
    assert res_a.champion.hotkey == "alice"  # bob's 1.90 (different arena) is excluded
    assert res_a.challenger_score == 1.10


def test_settle_default_arena_does_not_mix_non_default_scores():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.30, "default")
    _score(led, "bob", "H_B", "moe.fused_experts", 1.90, "minimax-m3")
    res = led.settle(0, margin=0.02)  # omitted arena -> default arena only
    assert res.champion.hotkey == "alice"
    assert res.weights == {"alice": 1.0}


def test_arena_champions_remain_isolated_after_settling_other_arena():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.10, "gpt-oss")
    _score(led, "bob", "H_B", "moe.fused_experts", 1.90, "minimax-m3")
    assert led.settle(0, margin=0.02, arena="gpt-oss").champion.hotkey == "alice"
    assert led.settle(0, margin=0.02, arena="minimax-m3").champion.hotkey == "bob"

    # A later empty gpt-oss round must keep paying the gpt-oss champion, not the
    # last-settled minimax champion.
    res = led.settle(1, margin=0.02, arena="gpt-oss")
    assert res.champion.hotkey == "alice"
    assert res.weights == {"alice": 1.0}


def test_per_slot_champions_are_arena_scoped():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.10, "gpt-oss")
    _score(led, "bob", "H_B", "moe.fused_experts", 1.90, "minimax-m3")
    assert led.settle_per_slot(0, margin=0.02, arena="gpt-oss").weights == {"alice": 1.0}
    assert led.settle_per_slot(0, margin=0.02, arena="minimax-m3").weights == {"bob": 1.0}
    assert led.settle_per_slot(1, margin=0.02, arena="gpt-oss").weights == {"alice": 1.0}


def test_arena_score_persists(tmp_path):
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.20, "minimax-m3")
    p = tmp_path / "l.json"
    led.save(p)
    led2 = Ledger.load(p)
    assert led2.scores[0].arena == "minimax-m3"


def test_arena_champion_persists_by_arena(tmp_path):
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.10, "gpt-oss")
    _score(led, "bob", "H_B", "moe.fused_experts", 1.90, "minimax-m3")
    led.settle(0, margin=0.02, arena="gpt-oss")
    led.settle(0, margin=0.02, arena="minimax-m3")
    p = tmp_path / "l.json"
    led.save(p)

    led2 = Ledger.load(p)
    assert led2.arena_champions["gpt-oss"].hotkey == "alice"
    assert led2.arena_champions["minimax-m3"].hotkey == "bob"


def test_arena_for_model_resolves_then_defaults():
    assert arena_for_model("no-such-model") is DEFAULT_ARENA
    assert "default" in list_arenas()


def test_cli_rejects_model_that_does_not_match_arena():
    from optima.cli import _resolve_scoring_arena

    ARENAS["tmp-m3"] = Arena(name="tmp-m3", model_path="MiniMax/M3", sglang_version="0.5.13")
    try:
        with pytest.raises(ValueError, match="--model"):
            _resolve_scoring_arena(argparse.Namespace(arena="tmp-m3", model="Qwen/Qwen2.5"))
    finally:
        ARENAS.pop("tmp-m3", None)


def test_engine_kwargs_precedence_arena_then_typed_then_json():
    from optima.eval._launch import engine_kwargs
    from optima.eval.throughput_kl import EvalConfig

    cfg = EvalConfig(
        model_path="model",
        base_engine_kwargs={"tp_size": 4, "custom": "arena"},
        tp_size=2,
        extra_engine_kwargs={"custom": "json"},
    )
    kw = engine_kwargs(cfg)
    assert kw["tp_size"] == 2
    assert kw["custom"] == "json"


def test_engine_kwargs_drops_harness_owned_base_keys():
    # A harness-owned key in arena base_engine_kwargs is dropped (the typed cfg wins),
    # but a non-reserved base kwarg with no typed field passes straight through.
    from optima.eval._launch import engine_kwargs
    from optima.eval.throughput_kl import EvalConfig

    cfg = EvalConfig(
        model_path="model",
        mem_fraction_static=0.6,
        base_engine_kwargs={"mem_fraction_static": 0.99, "schedule_policy": "fcfs"},
    )
    kw = engine_kwargs(cfg)
    assert kw["mem_fraction_static"] == 0.6  # reserved: arena's 0.99 dropped, not honored
    assert kw["schedule_policy"] == "fcfs"  # non-reserved: survives


def test_set_weights_reads_authoritative_arena_champion_not_legacy_view():
    # The bug this guards: set-weights must emit the ACTIVE arena's champion, not the
    # last-settled one. arena_champions is per-arena correct; the legacy `champion` is not.
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.10, "gpt-oss")
    _score(led, "bob", "H_B", "moe.fused_experts", 1.90, "minimax-m3")
    led.settle(0, margin=0.02, arena="gpt-oss")
    led.settle(0, margin=0.02, arena="minimax-m3")  # settled last
    assert led.arena_champions["gpt-oss"].hotkey == "alice"
    assert led.arena_champions["minimax-m3"].hotkey == "bob"
    assert led.champion.hotkey == "bob"  # legacy view = last-settled -> wrong for gpt-oss


def test_legacy_views_alias_authoritative_maps_after_load(tmp_path):
    # The legacy champion/champions views must be the SAME objects as their arena entries
    # after load, so the two serialized blobs can never drift apart.
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.10, "gpt-oss")
    _score(led, "bob", "H_B", "moe.fused_experts", 1.90, "minimax-m3")
    led.settle(0, margin=0.02, arena="gpt-oss")
    led.settle(0, margin=0.02, arena="minimax-m3")
    led.settle_per_slot(0, margin=0.02, arena="default")  # seed the default slot map too
    p = tmp_path / "l.json"
    led.save(p)

    led2 = Ledger.load(p)
    assert led2.champion is led2.arena_champions[led2.champion.arena]
    assert led2.champions is led2.arena_slot_champions["default"]


def test_pre_arena_ledger_migrates_champion_into_default(tmp_path):
    import json

    # A ledger written by pre-arena code has only `champion` (no arena maps). Load must
    # migrate it into the default arena and alias the legacy view to it.
    p = tmp_path / "old.json"
    p.write_text(json.dumps({
        "schema_version": 1,
        "champion": {"content_hash": "H", "hotkey": "alice", "score": 1.5, "round_id": 0},
    }))
    led = Ledger.load(p)
    assert led.arena_champions["default"].hotkey == "alice"
    assert led.champion is led.arena_champions["default"]
