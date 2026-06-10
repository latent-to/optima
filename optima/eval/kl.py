"""KL between two runs' per-position top-k token distributions.

SGLang returns, per position, a list of ``(logprob, token_id, text|None)`` for
the top-k tokens (``output_top_logprobs`` / ``input_top_logprobs``). We turn each
into a distribution and compute KL(reference || candidate) per position, then
average.

Caveat, stated honestly: top-k truncation means each distribution only carries
the head mass, so this is an *approximation* of the true full-vocab KL. It is
sensitive enough to catch the cheats that matter (calibration collapse, biased
quant, dropped precision) when k is reasonably large (e.g. 20+), but it is not a
substitute for a full-vocab teacher-forced KL when you can afford the logits.
The production path should capture full logits at the reference seam; this MVP
uses top-k because that is what the stock Engine API exposes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# A per-position top-k entry as returned by sglang: (logprob, token_id, text|None)
TopK = Sequence[tuple]


def _dist_from_topk(topk: TopK) -> dict[int, float]:
    d: dict[int, float] = {}
    for entry in topk:
        lp = float(entry[0])
        if not math.isfinite(lp):
            # NaN / +inf top-k entries (the deterministic pytorch sampling backend
            # emits them in the tail) carry no usable mass; -inf is prob 0 anyway.
            # Drop them rather than poison the whole position's sum.
            continue
        tid = int(entry[1])
        d[tid] = d.get(tid, 0.0) + math.exp(lp)
    return d


def _coverage(topk: TopK) -> float:
    """Probability mass the reported top-k actually captures (sum of exp(logprob)).

    The top-k KL is blind to mass that moves into the UNREPORTED tail: a candidate
    whose visible head matches the baseline yields KL~0 even if it flattened the
    distribution (calibration collapse / diversity loss). But flattening drops the
    head's captured mass — a well-calibrated decode step keeps ~0.9+ here; a flattened
    one is much lower. Comparing baseline-vs-candidate coverage is a magnitude-aware
    complement that *does* see the tail redistribution. Clamped to [0, 1]."""
    m = sum(p for p in _dist_from_topk(topk).values())
    return min(1.0, max(0.0, m))


# A degenerate candidate distribution — a broken kernel that drives the logits to
# inf/NaN, so sglang returns non-finite logprobs — is MAXIMAL divergence, not zero.
# Mapping it to a large finite sentinel (instead of letting `max(0.0, nan)` silently
# return 0.0) keeps aggregation finite while making it unmistakably fail any gate.
DEGENERATE_KL = 1e3


def kl_position(ref_topk: TopK, cand_topk: TopK, *, eps: float = 1e-8) -> float:
    """KL(ref || cand) over the union of the two (sanitized) top-k supports."""
    P = _dist_from_topk(ref_topk)
    Q = _dist_from_topk(cand_topk)
    if not P and not Q:
        return 0.0
    if not Q:
        # The reference has a distribution but the candidate produced NONE — every
        # candidate logprob was non-finite (a blown-up model). Maximal divergence,
        # never 0 (the old `max(0.0, nan)` silently returned 0.0 here). This fires
        # only on a genuinely degenerate candidate, not on a shared NaN tail.
        return DEGENERATE_KL
    # Renormalize each over the shared support with a floor for missing mass.
    support = set(P) | set(Q)
    pz = sum(P.get(t, 0.0) + eps for t in support)
    qz = sum(Q.get(t, 0.0) + eps for t in support)
    kl = 0.0
    for t in support:
        p = (P.get(t, 0.0) + eps) / pz
        q = (Q.get(t, 0.0) + eps) / qz
        kl += p * math.log(p / q)
    return max(0.0, kl) if math.isfinite(kl) else DEGENERATE_KL


@dataclass
class KLReport:
    num_positions: int
    mean_kl: float
    max_kl: float
    p99_kl: float
    # number of positions where the argmax token differs between ref and cand
    argmax_disagreements: int
    # mean |coverage(ref) - coverage(cand)| — head-mass shift the top-k KL can't see
    mean_coverage_dev: float = 0.0
    # positions baseline produced that the candidate never matched (early stop / length
    # deficit, NOT greedy divergence). Set by aligned_kl; a fidelity signal of its own.
    dropped_positions: int = 0

    @property
    def argmax_disagree_rate(self) -> float:
        """Fraction of compared positions where the candidate picked a different
        top token. Catches sparse cheats the mean misses: a flip counts regardless
        of its KL magnitude, so a kernel that's bit-exact almost everywhere but
        corrupts a few positions still shows a non-trivial rate."""
        return self.argmax_disagreements / self.num_positions if self.num_positions else 0.0


def kl_gate_ok(
    report: "KLReport",
    *,
    kl_threshold: Optional[float],
    p99_kl_threshold: Optional[float] = None,
    argmax_disagree_rate_threshold: Optional[float] = None,
    coverage_dev_threshold: Optional[float] = None,
) -> bool:
    """The fidelity gate over a ``KLReport``.

    ``kl_threshold is None`` -> advisory (always OK; rely on the accuracy gate, e.g.
    on big MoE where KL is noise-dominated). Otherwise the candidate must clear ALL
    configured checks:

      * ``mean_kl``   <= ``kl_threshold``                 — diffuse drift
      * ``p99_kl``    <= ``p99_kl_threshold`` (if set)    — a catastrophic tail
      * ``argmax_disagree_rate`` <= rate threshold (if set) — sparse argmax flips
      * ``mean_coverage_dev`` <= ``coverage_dev_threshold`` (if set) — tail-mass shift
        the top-k-truncated KL can't see (calibration collapse / diversity loss)

    Mean alone is blind to a sparse/targeted cheat (bit-exact 99.9% of the time,
    wrong on a rare pattern) because the few bad positions average out. The rate
    check is magnitude-independent; the coverage check is tail-aware — together they
    close the head-only-KL blind spots.
    """
    if kl_threshold is None:
        return True
    if report.num_positions == 0:
        return True  # no logprobs returned -> defer to the accuracy gate
    if report.mean_kl > kl_threshold:
        return False
    if p99_kl_threshold is not None and report.p99_kl > p99_kl_threshold:
        return False
    if argmax_disagree_rate_threshold is not None and report.argmax_disagree_rate > argmax_disagree_rate_threshold:
        return False
    if coverage_dev_threshold is not None and report.mean_coverage_dev > coverage_dev_threshold:
        return False
    return True


def _argmax(topk: TopK) -> Optional[int]:
    best_lp = -math.inf
    best_tid: Optional[int] = None
    for entry in topk:
        lp = float(entry[0])
        if lp > best_lp:
            best_lp = lp
            best_tid = int(entry[1])
    return best_tid


def kl_over_positions(
    ref: Sequence[TopK], cand: Sequence[TopK], *, eps: float = 1e-8
) -> KLReport:
    """Aggregate KL across aligned positions.

    ``ref`` and ``cand`` are per-position top-k lists; they must already be aligned
    (same positions). Only ``min(len(ref), len(cand))`` positions are scored here;
    any positions ``ref`` has beyond that are a LENGTH deficit, which ``aligned_kl``
    accounts for separately as ``dropped_positions`` (this function scores only the
    overlap).
    """
    n = min(len(ref), len(cand))
    kls: list[float] = []
    cov_devs: list[float] = []
    disagree = 0
    for i in range(n):
        kls.append(kl_position(ref[i], cand[i], eps=eps))
        cov_devs.append(abs(_coverage(ref[i]) - _coverage(cand[i])))
        if _argmax(ref[i]) != _argmax(cand[i]):
            disagree += 1
    if not kls:
        return KLReport(0, 0.0, 0.0, 0.0, 0)
    kls_sorted = sorted(kls)
    p99 = kls_sorted[min(len(kls_sorted) - 1, int(0.99 * len(kls_sorted)))]
    return KLReport(
        num_positions=n,
        mean_kl=sum(kls) / len(kls),
        max_kl=max(kls),
        p99_kl=p99,
        argmax_disagreements=disagree,
        mean_coverage_dev=sum(cov_devs) / len(cov_devs),
    )


# A single prompt's run, as KL consumes it: (generated token ids, per-position top-k).
PromptRun = tuple[Sequence[int], Sequence[TopK]]


def extract_per_prompt(outputs: Sequence[dict]) -> list[tuple[list[int], list]]:
    """Pull ``(output_ids, per-position top-k)`` out of sglang's generate() outputs.

    Shared by the throughput+KL eval and the benchmark eval so both build the exact
    same structure for ``aligned_kl``.
    """
    per_prompt: list[tuple[list[int], list]] = []
    for o in outputs:
        meta = o.get("meta_info", {})
        output_ids = o.get("output_ids") or meta.get("output_ids") or []
        topk = meta.get("output_top_logprobs") or []
        per_prompt.append(([int(t) for t in output_ids], topk))
    return per_prompt


def aligned_kl(
    baseline: Sequence[PromptRun], candidate: Sequence[PromptRun], *, eps: float = 1e-8
) -> KLReport:
    """KL between two runs, aligned per prompt up to the first token divergence.

    Greedy decoding means the candidate can diverge from the baseline mid-sequence;
    once the generated token at position ``i`` differs, the two runs no longer share
    a context and later positions aren't comparable. So we compare position ``i``
    and then stop at the first mismatch. Position 0 always shares the prompt, so a
    kernel that derails the very first token still gets scored (a large KL) instead
    of silently contributing zero comparable positions.

    EARLY STOP is distinct from divergence: if the tokens MATCH up to where the
    candidate simply ran out (it stopped before the baseline did), those un-produced
    baseline positions are recorded as ``dropped_positions`` — a real fidelity signal
    (the candidate stopped doing the work) that the per-position KL would otherwise
    miss. (It can't arise from an upstream-of-sampler op kernel, but an engine-patching
    submission could trigger it; token_match_rate also catches it in framework mode.)
    """
    ref_positions: list = []
    cand_positions: list = []
    dropped = 0
    for (b_ids, b_topk), (c_ids, c_topk) in zip(baseline, candidate):
        n = min(len(b_topk), len(c_topk))
        diverged = False
        for i in range(n):
            ref_positions.append(b_topk[i])
            cand_positions.append(c_topk[i])
            if i < len(b_ids) and i < len(c_ids) and b_ids[i] != c_ids[i]:
                diverged = True
                break
        # Tokens matched but the candidate produced fewer positions than the baseline
        # -> early stop / length deficit. Count the un-matched baseline tail as dropped.
        if not diverged and len(b_topk) > len(c_topk):
            dropped += len(b_topk) - len(c_topk)
    report = kl_over_positions(ref_positions, cand_positions, eps=eps)
    report.dropped_positions = dropped
    return report


def token_match_rate(baseline: Sequence[PromptRun], candidate: Sequence[PromptRun]) -> tuple[int, int]:
    """Fraction of generated positions where the candidate emits the SAME token as the
    (trusted) baseline. The FRAMEWORK-MODE correctness check: when the miner can patch
    the engine, its self-reported logprobs aren't trustworthy — but its emitted tokens
    are only correct if it actually computed correctly (and, under no-egress isolation,
    it can't fetch them). So we compare tokens, not KL. Returns ``(matched, total)``; a
    length mismatch counts the extra/missing positions as non-matches.
    """
    matched = 0
    total = 0
    for (b_ids, _), (c_ids, _) in zip(baseline, candidate):
        n = min(len(b_ids), len(c_ids))
        for i in range(n):
            total += 1
            if int(b_ids[i]) == int(c_ids[i]):
                matched += 1
        total += abs(len(b_ids) - len(c_ids))
    return matched, total
