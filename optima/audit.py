"""In-engine slot audit — direct fidelity, replacing rollout-KL as the primary gate.

WHY (2026-07-07, measured): rollout-KL between two engine launches measures
BATCHING, not fidelity, on non-deterministic stacks — a bit-stock candidate
running at 0.545x speed scored mean_kl 0.96 purely because timing shifts batch
composition and kernels are batch-variant; and sglang's deterministic mode
refuses the arena's fa4 attention backend outright. The invariant the referee
actually needs is direct: "in the SCORED engine, the miner kernel computes the
slot's declared function." So the validator audits exactly that: on randomly
sampled dispatcher calls, run the captured STOCK baseline on pre-call clones and
compare the miner outputs within the slot's own verify tolerances (the same
numeric contract offline and in-engine). Backend-agnostic; no determinism
assumptions; uniform across dispatchers (every seam holds the baseline in
closure).

Gate stack this belongs to: verify (fp32 ground truth, jittered/temporal/burst)
-> THIS audit (untimed quality launch) -> paired benchmark no-regression ->
rollout-KL demoted to advisory (still computed and reported; it is calibration
data, not a razor).

Threat notes:
  * Sampling comes from a process-private RNG seeded with os.urandom — a kernel
    cannot know ex-ante whether a call is audited, so behaving only on audited
    calls is not a strategy. (In-process miner code could in principle
    introspect this module; full isolation is the existing roadmap item that
    closes that class for good.)
  * Audits run only when OPTIMA_SLOT_AUDIT is set (the eval's untimed quality
    launch sets it); timed launches never carry the overhead.
  * A failed comparison NEVER crashes the engine: violations are counted and
    receipted; the eval driver reads the receipts and fails the bundle.
  * The baseline call itself may be collective (e.g. the fused AR+norm
    chokepoint): safe only because every rank reaches the dispatcher for the
    same calls in lockstep AND the sampling RNG is seeded identically across
    ranks of one launch (OPTIMA_SLOT_AUDIT_SEED, set by the driver) — a
    rank-divergent sample of a collective baseline would deadlock.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional, Sequence

import torch

from optima import receipts

logger = logging.getLogger("optima.audit")

# Per-call pass bar for "allclose" slots: fraction of elements within the slot's
# (atol, rtol) bound vs the STOCK baseline. Verify uses ratio 1.0 against an fp32
# reference; in-engine both sides are low-precision, so the worst few elements sit
# at the tolerance edge (outlier channels: one bf16 ULP at magnitude 4096 is 32).
# Garbage is nowhere near this: a wrong-function kernel misses on MOST elements.
_ALLCLOSE_MIN_RATIO = 0.995

# In-engine margin under a matched_ratio slot's verify bar. Verify compares the
# candidate against an FP32 reference; the audit compares it against the STOCK
# low-precision kernel, so BOTH sides carry rounding and honest implementations sit
# a hair below the verify-calibrated bar. MEASURED on the M3 arena control bundles
# (2026-07-07, 4xB300): an fp32-EXACT reference audited at worst_frac 0.9894 (8/3308
# calls under the raw 0.99 bar — a must-pass control failing), the honest v6 kernel
# at 0.9901, and residual-dropping sabotage at 0.0029. The margin splits honest
# rounding (~0.989+) from garbage (~0.003) with three orders of magnitude to spare.
_MATCHED_RATIO_AUDIT_MARGIN = 0.005

_state: dict = {"rate": None, "rng": None}
_stats: dict[str, dict] = {}


def _rate() -> float:
    if _state["rate"] is None:
        try:
            _state["rate"] = min(1.0, max(0.0, float(os.environ.get("OPTIMA_SLOT_AUDIT", "0"))))
        except ValueError:
            _state["rate"] = 0.0
        if _state["rate"] > 0.0:
            seed = os.environ.get("OPTIMA_SLOT_AUDIT_SEED")
            # Collective-safe sampling REQUIRES rank-identical decisions; the driver
            # sets the seed. Without one, fall back to urandom (fine for pure-op
            # slots, unsafe for collective baselines — the driver always seeds).
            _state["rng"] = random.Random(int(seed) if seed else os.urandom(8))
    return _state["rate"]


def enabled() -> bool:
    return _rate() > 0.0


def sampled() -> bool:
    """Decide (per dispatcher call) whether this call is audited."""
    r = _rate()
    return r > 0.0 and _state["rng"].random() < r


def _slot_stats(slot: str) -> dict:
    return _stats.setdefault(slot, {
        "slot": slot, "n": 0, "violations": 0, "baseline_refused": 0,
        "compare_errors": 0, "worst_frac": 1.0, "min_ratio": None, "mode": None,
    })


def _receipt(slot: str) -> None:
    # receipts.write names the file kind.tag.pid.json -> same-call-site writes
    # OVERWRITE, giving a rolling per-rank summary; the driver reads the final state.
    receipts.write("audit", _stats[slot], tag=slot)


def baseline_refused(slot: str) -> None:
    """The stock baseline declined this call (e.g. returned (None, None)) — a
    coverage note, not a violation: there is nothing to compare against."""
    s = _slot_stats(slot)
    s["baseline_refused"] += 1
    _receipt(slot)


def record(slot: str, actual: Sequence[torch.Tensor],
           expected: Sequence[Optional[torch.Tensor]]) -> None:
    """Compare miner outputs vs the stock baseline's, under the slot's verify
    tolerances, and fold the result into the receipted stats. Never raises."""
    try:
        from optima.slots import SLOTS

        spec = SLOTS.get(slot)
        s = _slot_stats(slot)
        if spec is None:
            s["compare_errors"] += 1
            _receipt(slot)
            return
        if any(e is None for e in expected) or len(actual) != len(expected):
            baseline_refused(slot)
            return

        corr = spec.correctness
        s["mode"] = corr.mode
        ok = True
        worst = 1.0
        for a, e in zip(actual, expected):
            af, ef = a.detach().float(), e.detach().float()
            if corr.mode == "cosine":
                cos = torch.nn.functional.cosine_similarity(
                    af.flatten(), ef.flatten(), dim=0).item()
                worst = min(worst, cos)
                ok = ok and cos >= corr.min_cosine
                s["min_ratio"] = corr.min_cosine
            elif corr.mode == "topk_overlap":
                # Selection slots: the dispatcher audits the CONSUMED product — the
                # top-k index rows its (validator-owned) selector produced from miner
                # scores vs the rows the stock function produced on the same pristine
                # inputs (the stock baseline runs BEFORE the miner path on audited
                # calls, so no input clones are needed). Per-row set overlap over the
                # stock row's valid (>= 0) entries, vacuous rows skipped, mean gated
                # at the slot's own min_overlap — verify-parity semantics.
                ai = a.detach().to(torch.long)
                ei = e.detach().to(torch.long)
                valid = ei >= 0
                rows = valid.any(dim=-1)
                if not bool(rows.any()):
                    baseline_refused(slot)  # nothing selected: coverage note only
                    return
                hit = (ei.unsqueeze(-1) == ai.unsqueeze(-2)).any(dim=-1)
                per_row = ((hit & valid).sum(-1).float()
                           / valid.sum(-1).clamp(min=1).float())
                ov = float(per_row[rows].mean())
                worst = min(worst, ov)
                ok = ok and ov >= corr.min_overlap
                s["min_ratio"] = corr.min_overlap
            else:
                tol = spec.tolerance_for(a.dtype)
                within = ((af - ef).abs() <= tol.atol + tol.rtol * ef.abs())
                frac = within.float().mean().item()
                bar = (max(0.0, corr.min_ratio - _MATCHED_RATIO_AUDIT_MARGIN)
                       if corr.mode == "matched_ratio" else _ALLCLOSE_MIN_RATIO)
                worst = min(worst, frac)
                ok = ok and frac >= bar
                s["min_ratio"] = bar
        s["n"] += 1
        s["worst_frac"] = min(s["worst_frac"], worst)
        if not ok:
            s["violations"] += 1
            if s["violations"] <= 4:
                logger.warning(
                    "optima.audit VIOLATION slot=%s call=%d frac/cos=%.4f (bar %.4f) "
                    "shapes=%s", slot, s["n"], worst, s["min_ratio"],
                    [tuple(a.shape) for a in actual])
        _receipt(slot)
    except Exception:  # noqa: BLE001 — an audit must never take down an engine
        try:
            s = _slot_stats(slot)
            s["compare_errors"] += 1
            _receipt(slot)
        except Exception:  # noqa: BLE001
            pass
        logger.exception("optima.audit: compare failed (slot=%s)", slot)


def run(slot: str, actual: Sequence[torch.Tensor], baseline_thunk) -> None:
    """Dispatcher-side one-liner: call the captured stock baseline (thunk) and
    record the comparison. Never raises — a baseline error is a compare_error,
    not an engine crash. (For COLLECTIVE baselines the thunk itself is a
    collective; if it errors on one rank the engine is already unrecoverable —
    hang-avoidance beyond that is out of scope here.)"""
    try:
        expected = baseline_thunk()
        if not isinstance(expected, (tuple, list)):
            expected = (expected,)
        record(slot, actual, tuple(expected))
    except Exception:  # noqa: BLE001
        try:
            s = _slot_stats(slot)
            s["compare_errors"] += 1
            _receipt(slot)
        except Exception:  # noqa: BLE001
            pass
        logger.exception("optima.audit: baseline call failed (slot=%s)", slot)


def gate(audit_receipts: list[dict], *, min_calls: int) -> tuple[bool, str]:
    """Eval-driver side: fold per-rank rolling receipts into a verdict.

    Pass iff every audited slot has zero violations and the total audited-call
    count is at least ``min_calls`` (insufficient coverage is a FAIL — an
    unaudited kernel is unproven, not innocent)."""
    if not audit_receipts:
        return False, f"no audit receipts (need >= {min_calls} audited calls)"
    total_n = sum(r.get("n", 0) for r in audit_receipts)
    total_viol = sum(r.get("violations", 0) for r in audit_receipts)
    total_err = sum(r.get("compare_errors", 0) for r in audit_receipts)
    worst = min((r.get("worst_frac", 1.0) for r in audit_receipts), default=1.0)
    desc = (f"{total_n} audited calls, {total_viol} violations, "
            f"worst_frac={worst:.4f}, compare_errors={total_err}")
    if total_viol > 0:
        return False, desc
    if total_err > 0:
        return False, desc + " (audit could not compare; refusing to pass unproven)"
    if total_n < min_calls:
        return False, desc + f" (insufficient coverage; need >= {min_calls})"
    return True, desc
