"""Noise-robust speedup scoring for a validator that CANNOT lock GPU clocks.

The validator runs on rented pods where ``nvidia-smi -lgc`` is denied and the
profiling counters are blocked, so per-launch throughput carries a large
warmup/thermal/boost component (measured at **±7-17%**, worst-case ~±32% cold).
Crucially that component is a *between-launch systematic offset*, so median-of-K
*within* a single launch cannot remove it: if the baseline launch runs cold and
the candidate launch runs warm, the candidate "wins" on clock state alone. The
old gate compared one cold baseline to one warm candidate against a hand-picked
2% margin that sits an order of magnitude *below* that noise — the exact source
of the project's repeated phantom wins.

This module turns raw per-launch tok/s into a trustworthy verdict using only what
is available without privileged clock control:

* **Bookended / interleaved A/B.** The baseline is measured both *before* and
  *after* the candidate (``B, C, B'``), so the candidate is bracketed and the two
  baseline reads bound the drift that occurred across it. More rounds tighten it.
* **Paired speedup.** Speedup is ``candidate / mean(bracketing baselines)``, so a
  monotonic ramp across the run partly cancels rather than biasing one side.
* **Noise-derived margin.** The bar a speedup must clear is
  ``1 + max(min_margin, k * noise)``, where ``noise`` is the *measured* relative
  spread of the repeated baseline reads — not a constant that ignores the box.
* **Drift rejection (no-decision).** If the bracketing baselines disagree by more
  than ``max_noise``, the box was too unstable this round to trust: the verdict is
  ``confident=False`` and it must NOT crown. A real subnet re-queues such a round;
  it never pays emissions on a measurement it cannot reproduce.

Every function here is pure and unit-tested on CPU with synthetic samples — the
statistics are hardware-independent, so they are validated here and run unchanged
on the pod. This is the piece that makes a sub-10% real win resolvable on a noisy
box, which is the regime every win on a mature model lives in.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeedupVerdict:
    """The outcome of comparing a candidate launch to bracketing baseline launches."""

    speedup: float  # robust paired estimate: candidate / mean(baseline reads)
    noise: float  # measured relative spread of the baseline reads (the floor)
    required: float  # the bar it had to clear: 1 + max(min_margin, k*noise)
    passed_speedup: bool  # cleared `required` AND the round was trustworthy
    confident: bool  # False -> box too noisy this round; treat as NO-DECISION, never crown
    n_baselines: int
    detail: str = ""


def relative_spread(samples: list[float]) -> float:
    """A scale-free measure of run-to-run noise across point estimates.

    For >=3 reads use population stdev / mean (smooth, uses all points). For exactly
    2 reads (the default bookend ``B, B'``) stdev underestimates, so use the range /
    mean — the honest worst-case gap between the two bracketing baselines. Returns
    ``inf`` for <2 reads (noise unmeasurable -> the caller must not claim confidence).
    """
    vals = [s for s in samples if s > 0]
    if len(vals) < 2:
        return float("inf")
    mean = statistics.fmean(vals)
    if mean <= 0:
        return float("inf")
    if len(vals) == 2:
        return (max(vals) - min(vals)) / mean
    return statistics.pstdev(vals) / mean


def score_speedup(
    baseline_reads: list[float],
    candidate_read: float,
    *,
    min_margin: float = 0.02,
    k: float = 2.0,
    max_noise: float = 0.10,
) -> SpeedupVerdict:
    """Decide whether ``candidate_read`` is a *real* speedup over the bracketing baselines.

    ``baseline_reads`` are the tok/s of the bookending baseline launches (>=2 for a
    real verdict: e.g. the ``B`` and ``B'`` around the candidate ``C``). The speedup
    is paired against their mean; the required margin scales with the measured
    baseline noise; and a round whose baselines disagree by more than ``max_noise``
    is flagged ``confident=False`` (no-decision) so noise can never mint a champion.

    * ``min_margin`` — floor on the required improvement even on a perfectly stable box.
    * ``k`` — how many noise-widths above 1.0 the speedup must sit (2.0 ~= a 2-sigma-ish bar).
    * ``max_noise`` — relative baseline spread above which the round is untrustworthy.
    """
    reads = [b for b in baseline_reads if b > 0]
    if not reads or candidate_read <= 0:
        return SpeedupVerdict(0.0, float("inf"), 1.0 + min_margin, False, False,
                              len(reads), "missing/zero throughput sample")
    base = statistics.fmean(reads)
    noise = relative_spread(reads)
    speedup = candidate_read / base
    required = 1.0 + max(min_margin, k * (noise if noise != float("inf") else 0.0))
    confident = len(reads) >= 2 and noise <= max_noise
    passed = confident and speedup >= required
    if not confident:
        if len(reads) < 2:
            detail = "single baseline read -> noise unmeasured; cannot crown (bookend the baseline)"
        else:
            detail = f"baseline drift {noise:.1%} > max_noise {max_noise:.0%}; NO-DECISION (re-queue)"
    elif passed:
        detail = f"speedup {speedup:.3f} >= required {required:.3f} (noise {noise:.1%})"
    else:
        detail = f"speedup {speedup:.3f} < required {required:.3f} (noise {noise:.1%})"
    return SpeedupVerdict(
        speedup=speedup, noise=noise, required=required,
        passed_speedup=passed, confident=confident, n_baselines=len(reads), detail=detail,
    )
