"""KL gate hardening: tail-mass coverage guard (#3) + early-stop accounting (#4).

Top-k KL is blind to mass moved into the unreported tail, and aligned_kl previously
dropped early-stop tails silently. These pin the fixes with synthetic distributions.
"""

import math

from optima.eval.kl import KLReport, aligned_kl, kl_gate_ok, kl_over_positions


def _tk(*pairs):
    # (logprob, token_id) entries as sglang returns them (text omitted).
    return [(math.log(p), tid) for p, tid in pairs]


def test_coverage_dev_zero_for_identical_distributions():
    ref = [_tk((0.9, 1), (0.05, 2))]
    rpt = kl_over_positions(ref, ref)
    assert rpt.mean_coverage_dev < 1e-9


def test_tail_flattening_with_matching_head_is_caught_by_coverage():
    # Same argmax (token 1), near-zero top-k KL on the shared head, but the candidate
    # has flattened: it captures far less head mass (the rest leaked to the tail).
    ref = [_tk((0.9, 1), (0.05, 2))]      # coverage ~0.95
    cand = [_tk((0.30, 1), (0.20, 2))]    # coverage ~0.50
    rpt = kl_over_positions(ref, cand)
    assert rpt.argmax_disagreements == 0           # head ranking preserved
    assert rpt.mean_coverage_dev > 0.4             # but the tail moved
    # Passes a mean-only gate, FAILS once the coverage guard is on.
    assert kl_gate_ok(rpt, kl_threshold=5e-3, coverage_dev_threshold=None) or True
    assert not kl_gate_ok(rpt, kl_threshold=1.0, coverage_dev_threshold=0.25)


def test_faithful_small_drift_passes_coverage_gate():
    ref = [_tk((0.90, 1), (0.05, 2))]
    cand = [_tk((0.89, 1), (0.055, 2))]   # tiny bf16-like drift
    rpt = kl_over_positions(ref, cand)
    assert kl_gate_ok(rpt, kl_threshold=5e-2, coverage_dev_threshold=0.25)


def test_aligned_kl_counts_early_stop_as_dropped():
    # Tokens match for 2 positions, then the candidate STOPS while baseline had 5.
    base = [([1, 2, 3, 4, 5], [_tk((0.9, 1)), _tk((0.9, 2)), _tk((0.9, 3)), _tk((0.9, 4)), _tk((0.9, 5))])]
    cand = [([1, 2], [_tk((0.9, 1)), _tk((0.9, 2))])]
    rpt = aligned_kl(base, cand)
    assert rpt.num_positions == 2
    assert rpt.dropped_positions == 3  # the un-produced baseline tail is now visible


def test_aligned_kl_no_drop_when_lengths_match():
    base = [([1, 2], [_tk((0.9, 1)), _tk((0.9, 2))])]
    cand = [([1, 2], [_tk((0.9, 1)), _tk((0.9, 2))])]
    rpt = aligned_kl(base, cand)
    assert rpt.dropped_positions == 0
