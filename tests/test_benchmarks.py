"""Torch-free tests for the realistic-workload eval pieces.

Covers the answer extractors (numeric + multiple-choice), the GSM8K/MMLU
``check`` logic, the benchmark registry, and the shared KL helpers
(``extract_per_prompt`` / ``aligned_kl``). All pure-Python: no GPU, no torch, no
network (the HF ``load`` paths are exercised on the pod, not here).
"""

from __future__ import annotations

from optima.eval.benchmarks import (
    GSM8K,
    MMLU,
    Problem,
    extract_choice_letter,
    extract_final_number,
    get_benchmark,
    list_benchmarks,
)
from optima.eval.kl import aligned_kl, extract_per_prompt


# ---- numeric extraction (GSM8K / MATH-style) --------------------------------


def test_extract_final_number_answer_cue():
    assert extract_final_number("lots of words. The answer is 42.") == 42.0


def test_extract_final_number_handles_commas_and_dollar():
    assert extract_final_number("So the total is $1,234.50 in the end.") == 1234.5


def test_extract_final_number_falls_back_to_last():
    assert extract_final_number("we add 3 and 4 to get 7") == 7.0


def test_extract_final_number_none_when_no_digits():
    assert extract_final_number("no numbers anywhere here") is None


# ---- multiple-choice extraction (MMLU / GPQA-style) -------------------------


def test_extract_choice_letter_cue_parenthesized():
    assert extract_choice_letter("...therefore The answer is (C).") == "C"


def test_extract_choice_letter_cue_bare():
    assert extract_choice_letter("Answer: B") == "B"


def test_extract_choice_letter_paren_fallback():
    assert extract_choice_letter("I'll go with (D) here") == "D"


def test_extract_choice_letter_out_of_range_is_none():
    # 'E' is beyond a 4-option question, so it must not be returned.
    assert extract_choice_letter("The answer is (E).", num_choices=4) is None


def test_extract_choice_letter_none():
    assert extract_choice_letter("no idea, honestly") is None


# ---- benchmark check logic --------------------------------------------------


def test_gsm8k_check():
    g = GSM8K()
    p = Problem(id="x", prompt="", answer="18")
    assert g.check(p, "work work work. The answer is 18.")
    assert not g.check(p, "work work work. The answer is 17.")


def test_mmlu_format_and_check():
    m = MMLU()
    body = m._format_question("What is 2+2?", ["3", "4", "5", "6"])
    assert "(B) 4" in body and "(A) 3" in body
    p = Problem(id="x", prompt="", answer="B", meta={"num_choices": 4})
    assert m.check(p, "reasoning... The answer is (B).")
    assert not m.check(p, "reasoning... The answer is (A).")


def test_registry_has_gsm8k_and_mmlu():
    assert get_benchmark("gsm8k").name == "gsm8k"
    assert get_benchmark("mmlu").name == "mmlu"
    assert {"gsm8k", "mmlu"} <= set(list_benchmarks())


# ---- shared KL helpers ------------------------------------------------------


def _out(ids, topk):
    """A minimal sglang-shaped generate() output."""
    return {"output_ids": ids, "meta_info": {"output_top_logprobs": topk}}


def test_aligned_kl_zero_on_identical():
    pos = [[(-0.1, 5, None), (-2.0, 9, None)], [(-0.2, 7, None), (-1.0, 3, None)]]
    base = extract_per_prompt([_out([5, 7], pos)])
    rep = aligned_kl(base, base)
    assert rep.num_positions == 2
    assert rep.mean_kl == 0.0
    assert rep.argmax_disagreements == 0


def test_aligned_kl_stops_at_first_divergence_but_scores_position_zero():
    base_pos = [[(-0.1, 5, None), (-2.0, 9, None)], [(-0.2, 7, None), (-1.0, 3, None)]]
    base = extract_per_prompt([_out([5, 7], base_pos)])
    # Candidate flips the very first token (argmax 5 -> 6); later positions must be
    # dropped (different context) but position 0 is still scored with a real KL.
    cand_pos = [[(-2.0, 5, None), (-0.1, 6, None)], [(-0.2, 7, None), (-1.0, 3, None)]]
    cand = extract_per_prompt([_out([6, 7], cand_pos)])
    rep = aligned_kl(base, cand)
    assert rep.num_positions == 1
    assert rep.mean_kl > 0.0
    assert rep.argmax_disagreements == 1


# ---- KL degenerate/NaN guard (a broken kernel that blows up the logits) ------


def test_kl_position_nan_candidate_is_max_not_zero():
    import math

    from optima.eval.kl import DEGENERATE_KL, kl_position

    ref = [(-0.05, 5, None), (-3.0, 9, None)]
    cand = [(float("nan"), 7, None)]  # blown-up model -> NaN logprob
    v = kl_position(ref, cand)
    assert v == DEGENERATE_KL  # the old `max(0.0, nan)` returned 0.0 here
    assert math.isfinite(v)


def test_kl_position_inf_candidate_is_max_not_zero():
    from optima.eval.kl import DEGENERATE_KL, kl_position

    assert kl_position([(-0.05, 5, None)], [(float("inf"), 7, None)]) == DEGENERATE_KL


def test_kl_position_confident_mismatch_is_large():
    from optima.eval.kl import kl_position

    # two near-certain but different tokens -> large finite KL, not ~0
    assert kl_position([(-1e-6, 5, None)], [(-1e-6, 7, None)]) > 1.0


def test_kl_position_shared_nan_tail_is_not_flagged():
    from optima.eval.kl import kl_position

    # the deterministic backend emits NaN tail entries; an otherwise-identical
    # candidate must read as KL 0, not a false "degenerate" divergence.
    ref = [(-0.05, 5, None), (float("nan"), 9, None)]
    cand = [(-0.05, 5, None), (float("nan"), 9, None)]
    assert kl_position(ref, cand) == 0.0


def test_aligned_kl_flags_nan_candidate():
    from optima.eval.kl import DEGENERATE_KL, aligned_kl, extract_per_prompt

    base = extract_per_prompt([_out([5], [[(-0.05, 5, None), (-3.0, 9, None)]])])
    cand = extract_per_prompt([_out([7], [[(float("nan"), 7, None)]])])
    rep = aligned_kl(base, cand)
    assert rep.num_positions == 1
    assert rep.mean_kl >= DEGENERATE_KL  # the NaN->0 bug would report 0.0 here


# ---- the gate: mean + p99 + argmax-rate (sparse-cheat hardening, 2b) ----------


def _report(num_positions, mean_kl, *, p99_kl=0.0, max_kl=0.0, disagreements=0):
    from optima.eval.kl import KLReport

    return KLReport(num_positions=num_positions, mean_kl=mean_kl, max_kl=max_kl,
                    p99_kl=p99_kl, argmax_disagreements=disagreements)


def test_gate_catches_sparse_cheat_that_mean_misses():
    from optima.eval.kl import kl_gate_ok

    # The adversary the harness never faced: bit-exact almost everywhere, a few
    # tokens flipped. mean_kl stays under threshold (the flips average out) but the
    # argmax-disagreement RATE is 3% (30/1000).
    sparse = _report(1000, mean_kl=0.001, disagreements=30)
    # OLD behaviour (mean only) would PASS it:
    assert kl_gate_ok(sparse, kl_threshold=5e-3, argmax_disagree_rate_threshold=None) is True
    # NEW gate with a 1% rate cap FAILS it:
    assert kl_gate_ok(sparse, kl_threshold=5e-3, argmax_disagree_rate_threshold=0.01) is False


def test_gate_passes_faithful_kernel():
    from optima.eval.kl import kl_gate_ok

    faithful = _report(1000, mean_kl=0.0, disagreements=0)  # deterministic floor
    assert kl_gate_ok(faithful, kl_threshold=5e-3, argmax_disagree_rate_threshold=0.01) is True


def test_gate_advisory_never_blocks():
    from optima.eval.kl import kl_gate_ok

    catastrophic = _report(1000, mean_kl=10.0, p99_kl=50.0, disagreements=900)
    assert kl_gate_ok(catastrophic, kl_threshold=None, argmax_disagree_rate_threshold=0.01) is True


def test_gate_p99_catches_catastrophic_tail():
    from optima.eval.kl import kl_gate_ok

    # low mean, few flips, but a handful of positions are wildly off (high p99)
    tail = _report(1000, mean_kl=0.004, p99_kl=20.0, disagreements=2)
    assert kl_gate_ok(tail, kl_threshold=5e-3, p99_kl_threshold=1.0) is False
    assert kl_gate_ok(tail, kl_threshold=5e-3, p99_kl_threshold=None) is True  # p99 opt-in


def test_gate_num_positions_zero_defers():
    from optima.eval.kl import kl_gate_ok

    assert kl_gate_ok(_report(0, mean_kl=0.0), kl_threshold=5e-3) is True  # no logprobs -> defer


# ---- LongMath answer checking: FINAL lines are authoritative; no raw substring ----


def _lm_problem(answer="42"):
    from optima.eval.benchmarks import Problem

    return Problem(id="t", prompt="p", answer=answer)


def test_long_math_final_lines_checked():
    lm = get_benchmark("long_math")
    assert lm.check(_lm_problem(), "FINAL_FIRST: 42\n...consistency...\nFINAL_LAST: 42")
    # a wrong FINAL answer is wrong even if the gold digits appear in the reasoning
    assert not lm.check(_lm_problem(), "FINAL_FIRST: 41\nwe considered 42 but rejected it\nFINAL_LAST: 41")


def test_long_math_substring_of_another_number_is_not_correct():
    lm = get_benchmark("long_math")
    # no FINAL lines: "42" inside "142"/"3.42"/"-42" must NOT pass (the old raw
    # substring check made the gate near-vacuous for short gold answers)
    assert not lm.check(_lm_problem(), "the running total is 142 here")
    assert not lm.check(_lm_problem(), "we get 3.42 as an intermediate")
    assert not lm.check(_lm_problem(), "the delta is -42 overall")
    # but a clean standalone mention still counts as a fallback
    assert lm.check(_lm_problem(), "so the result is 42.")
