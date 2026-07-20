"""Queue scheduler for the resident (hot-swap) speed screen.

Runs N candidates through ONE resident engine lifetime with the same-lane
bracket structure proven by the 2026-07-20 pod probes:

    B_0  swap(k1)  C_1  swap(stock)  B_1  swap(k2)  C_2  swap(stock)  B_2 ...

Every stock read doubles as (a) the closing bracket of the previous candidate,
(b) the opening bracket of the next, and (c) a contamination canary — the
engine provably dispatches stock (the swap-out ack registered zero slots), so a
stock read that leaves the lifetime's stock band flags in-process tampering or
state rot and stops the lifetime for a recycle.

Verdicts reuse :func:`optima.eval.scoring.score_speedup` (noise-derived bar,
NO-DECISION on disagreeing brackets).  Borderline candidates escalate to the
five-leg shape (B C B' C' B'') by swapping back in — an escalation costs two
swaps and two reads, never an engine reload.

Trust tier: screen/routing only.  Payment and crown evidence still come from
the isolated per-candidate qualification path.  Non-swappable bundles
(aot_exports device artifacts, dep-patched trees) never enter this queue — the
seam refuses them — and are scheduled as dedicated launches by the caller.

This module is deliberately free of executor imports: it drives the
:class:`~optima.eval.oci_resident_session.ResidentOuterSession` API only, so it
tests without GPUs, containers, or engines.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Protocol, Sequence

from optima.eval.oci_resident_session import (
    ResidentBatchEvidence,
    SwapReceipt,
)
from optima.eval.scoring import SpeedupVerdict, score_speedup
from optima.stack_identity import require_sha256_hex


_CANDIDATE_ID = re.compile(r"[A-Za-z0-9_.:+-]{1,128}\Z")

# The exact failure recorded when a tripped canary voids the just-closed
# verdict.  Consumers branch on CandidateScreenVerdict.withdrawn, never on
# this string.
WITHDRAWN_FAILURE = "stock canary drifted beyond tolerance; evidence withdrawn"


class ResidentQueueError(ValueError):
    """A queue plan, policy, or session interaction is invalid."""


class ScreenSession(Protocol):
    """The subset of ResidentOuterSession the screen scheduler drives."""

    def swap(self, bundle_digest: str | None) -> SwapReceipt: ...
    def execute_batch(
        self, prompts: Sequence[str], *, canary: bool = False
    ) -> ResidentBatchEvidence: ...


@dataclass(frozen=True)
class ScreenCandidate:
    """One swappable candidate, already staged in the swap intake."""

    candidate_id: str
    bundle_digest: str
    expected_slots: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or _CANDIDATE_ID.fullmatch(self.candidate_id) is None
        ):
            raise ResidentQueueError("screen candidate_id is invalid")
        try:
            require_sha256_hex(self.bundle_digest, field="screen bundle digest")
        except ValueError as exc:
            raise ResidentQueueError(str(exc)) from None
        slots = tuple(self.expected_slots)
        if (
            not slots
            or slots != tuple(sorted(set(slots)))
            or any(not isinstance(slot, str) or not slot for slot in slots)
        ):
            raise ResidentQueueError(
                "screen expected_slots must be nonempty sorted unique names"
            )
        object.__setattr__(self, "expected_slots", slots)


@dataclass(frozen=True)
class ScreenPolicy:
    """Bar, escalation, canary, and recycle policy for one screen pass.

    Defaults are pinned from the 2026-07-21 noise-qualification campaign on
    the production 4xB300 lane (8 interleaved null/bundle swap cycles, 9 stock
    reads, 16 recaptures): stock band 0.30%, worst null-cycle excursion 0.21%,
    zero nulls above a 1.005 bar.  min_margin sits above 2x the worst null
    excursion; canary_tolerance sits at ~4x the stock band so contamination
    trips it but honest drift does not.
    """

    min_margin: float = 0.0075
    noise_multiplier: float = 2.0
    max_noise: float = 0.10
    escalation_band: float = 0.02
    canary_tolerance: float = 0.012
    max_candidates_per_lifetime: int = 8

    def __post_init__(self) -> None:
        for name, value, low, high in (
            ("min_margin", self.min_margin, 0.0, 1.0),
            ("noise_multiplier", self.noise_multiplier, 0.0, 100.0),
            ("max_noise", self.max_noise, 0.0, 1.0),
            ("escalation_band", self.escalation_band, 0.0, 1.0),
            ("canary_tolerance", self.canary_tolerance, 0.0, 1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not low < float(value) < high
            ):
                raise ResidentQueueError(f"screen policy {name} is invalid")
        if (
            type(self.max_candidates_per_lifetime) is not int
            or not 1 <= self.max_candidates_per_lifetime <= 1_000
        ):
            raise ResidentQueueError(
                "screen policy max_candidates_per_lifetime is invalid"
            )


@dataclass(frozen=True)
class CandidateScreenVerdict:
    """Routing verdict for one candidate; never payment evidence."""

    candidate_id: str
    bundle_digest: str
    slots: tuple[str, ...]
    baseline_throughputs: tuple[float, ...]
    candidate_throughputs: tuple[float, ...]
    verdict: SpeedupVerdict | None
    escalated: bool
    failure: str | None
    swap_receipts: tuple[SwapReceipt, ...]
    batch_indices: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return (
            self.failure is None
            and self.verdict is not None
            and self.verdict.passed_speedup
        )

    @property
    def withdrawn(self) -> bool:
        """Evidence voided by a tripped canary; re-screen on a fresh lifetime."""
        return self.failure == WITHDRAWN_FAILURE

    @property
    def rejected_dispatch(self) -> bool:
        """The engine registered slots other than the candidate declared."""
        return self.failure is not None and not self.withdrawn

    def to_dict(self) -> dict[str, object]:
        verdict = self.verdict
        return {
            "baseline_throughputs": [
                format(row, ".17g") for row in self.baseline_throughputs
            ],
            "batch_indices": list(self.batch_indices),
            "bundle_digest": self.bundle_digest,
            "candidate_id": self.candidate_id,
            "candidate_throughputs": [
                format(row, ".17g") for row in self.candidate_throughputs
            ],
            "escalated": self.escalated,
            "failure": self.failure,
            "slots": list(self.slots),
            "swap_receipts": [row.to_dict() for row in self.swap_receipts],
            "verdict": None
            if verdict is None
            else {
                "confident": verdict.confident,
                "detail": verdict.detail,
                "n_baselines": verdict.n_baselines,
                "n_candidates": verdict.n_candidates,
                "noise": format(verdict.noise, ".17g"),
                "passed_speedup": verdict.passed_speedup,
                "required": format(verdict.required, ".17g"),
                "speedup": format(verdict.speedup, ".17g"),
            },
        }


@dataclass(frozen=True)
class ScreenReport:
    """The outcome of one resident lifetime's screen pass."""

    verdicts: tuple[CandidateScreenVerdict, ...]
    stock_throughputs: tuple[float, ...]
    unprocessed_candidate_ids: tuple[str, ...]
    stopped_reason: str | None


def _throughput(row: ResidentBatchEvidence) -> float:
    elapsed = row.elapsed_seconds
    if elapsed <= 0:
        raise ResidentQueueError("screen read clock did not advance")
    return row.token_numerator / elapsed


def _canary_drifted(
    stock_reads: Sequence[float], latest: float, *, tolerance: float
) -> bool:
    if len(stock_reads) < 2:
        return False
    reference = statistics.fmean(stock_reads[:-1])
    if reference <= 0:
        return True
    return abs(latest - reference) / reference > tolerance


def _is_borderline(verdict: SpeedupVerdict, *, band: float) -> bool:
    if not verdict.confident:
        return True
    return abs(verdict.speedup - verdict.required) <= band


class ResidentScreenLoop:
    """Incremental screen: one candidate at a time on one live session.

    The batch entry point :func:`run_resident_screen` drives this over a fixed
    list; the arena provider drives it over arrivals — candidates trickle in
    while the engine stays resident between them, so the loop carries the
    shared bracket (the last stock read), the lifetime's full stock band (the
    canary reference), and the stop condition across calls.

    ``screen`` returns ``None`` when the lifetime cannot accept the candidate
    (budget exhausted or already stopped) — the candidate was NOT touched and
    must be re-screened on a fresh lifetime.  A returned verdict can still be
    terminal for the lifetime: a tripped canary returns the WITHDRAWN verdict
    and sets ``stopped_reason``, so callers check it after every call.
    """

    def __init__(
        self,
        session: ScreenSession,
        *,
        prompts: Sequence[str],
        policy: ScreenPolicy = ScreenPolicy(),
    ) -> None:
        if type(policy) is not ScreenPolicy:
            raise ResidentQueueError("screen policy has the wrong type")
        prompt_plan = tuple(prompts)
        if not prompt_plan:
            raise ResidentQueueError("screen prompt plan is empty")
        self._session = session
        self._prompts = prompt_plan
        self._policy = policy
        self._stock: list[float] = []
        self._baseline_prev: float | None = None
        self._processed = 0
        self._stopped: str | None = None

    @property
    def stopped_reason(self) -> str | None:
        return self._stopped

    @property
    def stock_throughputs(self) -> tuple[float, ...]:
        return tuple(self._stock)

    @property
    def processed(self) -> int:
        """Candidates with retained verdicts (a withdrawn one does not count)."""
        return self._processed

    def screen(self, candidate: ScreenCandidate) -> CandidateScreenVerdict | None:
        if type(candidate) is not ScreenCandidate:
            raise ResidentQueueError("screen candidate is not exactly typed")
        if self._stopped is not None:
            return None
        policy = self._policy
        if self._processed >= policy.max_candidates_per_lifetime:
            self._stopped = "lifetime candidate budget exhausted"
            return None
        session = self._session
        prompt_plan = self._prompts
        if self._baseline_prev is None:
            opening = session.execute_batch(prompt_plan, canary=True)
            self._baseline_prev = _throughput(opening)
            self._stock.append(self._baseline_prev)

        receipts: list[SwapReceipt] = []
        batch_indices: list[int] = []
        failure: str | None = None
        candidate_reads: list[float] = []
        baseline_reads: list[float] = [self._baseline_prev]
        verdict: SpeedupVerdict | None = None
        escalated = False

        swap_in = session.swap(candidate.bundle_digest)
        receipts.append(swap_in)
        slots = swap_in.slots
        if slots != candidate.expected_slots:
            # The engine is live with unexpected dispatch; return to stock
            # before deciding anything else.
            failure = (
                f"registered slots {list(slots)!r} differ from expected "
                f"{list(candidate.expected_slots)!r}"
            )
        else:
            candidate_row = session.execute_batch(prompt_plan)
            batch_indices.append(candidate_row.batch_index)
            candidate_reads.append(_throughput(candidate_row))

        swap_out = session.swap(None)
        receipts.append(swap_out)
        closing = session.execute_batch(prompt_plan, canary=True)
        batch_indices.append(closing.batch_index)
        closing_throughput = _throughput(closing)
        self._stock.append(closing_throughput)
        baseline_reads.append(closing_throughput)

        if failure is None:
            verdict = score_speedup(
                baseline_reads,
                candidate_reads,
                min_margin=policy.min_margin,
                k=policy.noise_multiplier,
                max_noise=policy.max_noise,
            )
            if _is_borderline(verdict, band=policy.escalation_band):
                escalated = True
                swap_in_2 = session.swap(candidate.bundle_digest)
                receipts.append(swap_in_2)
                if swap_in_2.slots != candidate.expected_slots:
                    failure = "escalation swap registered different slots"
                else:
                    candidate_row_2 = session.execute_batch(prompt_plan)
                    batch_indices.append(candidate_row_2.batch_index)
                    candidate_reads.append(_throughput(candidate_row_2))
                swap_out_2 = session.swap(None)
                receipts.append(swap_out_2)
                closing_2 = session.execute_batch(prompt_plan, canary=True)
                batch_indices.append(closing_2.batch_index)
                closing_throughput = _throughput(closing_2)
                self._stock.append(closing_throughput)
                baseline_reads.append(closing_throughput)
                if failure is None:
                    verdict = score_speedup(
                        baseline_reads,
                        candidate_reads,
                        min_margin=policy.min_margin,
                        k=policy.noise_multiplier,
                        max_noise=policy.max_noise,
                    )

        result = CandidateScreenVerdict(
            candidate.candidate_id,
            candidate.bundle_digest,
            slots,
            tuple(baseline_reads),
            tuple(candidate_reads),
            verdict,
            escalated,
            failure,
            tuple(receipts),
            tuple(batch_indices),
        )
        self._processed += 1
        self._baseline_prev = closing_throughput

        if _canary_drifted(
            self._stock, closing_throughput, tolerance=policy.canary_tolerance
        ):
            # The drifted read closed THIS candidate's bracket, so its verdict
            # is built on suspect evidence: withdraw it and re-screen the
            # candidate on the fresh lifetime along with the remainder.
            result = CandidateScreenVerdict(
                result.candidate_id,
                result.bundle_digest,
                result.slots,
                result.baseline_throughputs,
                result.candidate_throughputs,
                None,
                result.escalated,
                WITHDRAWN_FAILURE,
                result.swap_receipts,
                result.batch_indices,
            )
            self._processed -= 1
            self._stopped = (
                "stock canary drifted beyond tolerance after "
                f"{candidate.candidate_id}; lifetime requires recycle"
            )
        return result


def run_resident_screen(
    session: ScreenSession,
    candidates: Sequence[ScreenCandidate],
    *,
    prompts: Sequence[str],
    policy: ScreenPolicy = ScreenPolicy(),
) -> ScreenReport:
    """Screen every candidate through one live engine; stop early on drift.

    The caller owns the engine lifetime: it opens the resident session, calls
    this, then closes.  When ``stopped_reason`` is set, the remaining
    candidates in ``unprocessed_candidate_ids`` must be re-screened on a fresh
    lifetime (recycle) — their absence here is scheduling state, not a verdict.
    A canary drift additionally WITHDRAWS the just-closed candidate's verdict
    (failure set, verdict ``None``, receipts retained for the record) and lists
    that candidate as unprocessed too: re-screen it on the fresh lifetime.
    """

    rows = tuple(candidates)
    if not rows or any(type(row) is not ScreenCandidate for row in rows):
        raise ResidentQueueError("screen candidates must be typed and nonempty")
    if len({row.candidate_id for row in rows}) != len(rows):
        raise ResidentQueueError("screen candidate ids must be unique")

    loop = ResidentScreenLoop(session, prompts=prompts, policy=policy)
    verdicts: list[CandidateScreenVerdict] = []
    for candidate in rows:
        result = loop.screen(candidate)
        if result is None:
            break
        verdicts.append(result)
        if loop.stopped_reason is not None:
            break
    # loop.processed is the exact list position of the first candidate that
    # must be re-screened: the loop was fresh and fed in list order, and a
    # withdrawn candidate does not count as processed.
    unprocessed = tuple(row.candidate_id for row in rows[loop.processed :])
    return ScreenReport(
        tuple(verdicts),
        loop.stock_throughputs,
        unprocessed,
        loop.stopped_reason,
    )


__all__ = [
    "CandidateScreenVerdict",
    "ResidentQueueError",
    "ResidentScreenLoop",
    "ScreenCandidate",
    "ScreenPolicy",
    "ScreenReport",
    "ScreenSession",
    "WITHDRAWN_FAILURE",
    "run_resident_screen",
]
