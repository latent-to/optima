"""CPU tests for the resident screen queue scheduler (fake session, no GPU)."""

from __future__ import annotations

import pytest

from optima.eval.oci_resident_session import ResidentBatchEvidence, SwapReceipt
from optima.eval.oci_session_protocol import BatchEvidence, PromptEvidence
from optima.eval.resident_queue import (
    ResidentQueueError,
    ScreenCandidate,
    ScreenPolicy,
    run_resident_screen,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _evidence(tokens: int = 4) -> BatchEvidence:
    return BatchEvidence(
        (
            PromptEvidence(
                tuple(range(tokens)),
                tuple(((-0.5, 0),) for _ in range(tokens)),
            ),
        )
    )


class FakeSession:
    """Plays back configured throughputs; models generation bookkeeping."""

    def __init__(self, stock_rate: float, candidate_rates: dict[str, float],
                 slots: dict[str, tuple[str, ...]] | None = None,
                 stock_drift_after: int | None = None) -> None:
        self.stock_rate = stock_rate
        self.candidate_rates = candidate_rates
        self.slots = slots or {
            digest: ("moe.fused_experts",) for digest in candidate_rates
        }
        self.stock_drift_after = stock_drift_after
        self.generation = 0
        self.active: str | None = None
        self.batch_count = 0
        self.stock_reads = 0
        self.swaps: list[str | None] = []
        self.clock = 0.0

    def swap(self, bundle_digest: str | None) -> SwapReceipt:
        self.generation += 1
        self.active = bundle_digest
        self.swaps.append(bundle_digest)
        self.clock += 30.0
        return SwapReceipt(
            len(self.swaps) - 1,
            self.generation,
            bundle_digest,
            () if bundle_digest is None else self.slots[bundle_digest],
            self.clock - 30.0,
            self.clock,
        )

    def execute_batch(self, prompts, *, canary: bool = False):
        assert not canary or self.active is None
        tokens = 1000
        if self.active is None:
            rate = self.stock_rate
            self.stock_reads += 1
            if (
                self.stock_drift_after is not None
                and self.stock_reads > self.stock_drift_after
            ):
                rate *= 0.90
        else:
            rate = self.candidate_rates[self.active]
        elapsed = tokens / rate
        started = self.clock
        self.clock += elapsed
        row = ResidentBatchEvidence(
            self.batch_count,
            f"{self.batch_count + 5:032x}",
            f"{self.batch_count + 6:032x}".replace("0", "9", 1),
            self.generation,
            () if self.active is None else self.slots[self.active],
            canary,
            started,
            self.clock,
            tokens,
            _evidence(),
        )
        self.batch_count += 1
        return row


def _candidate(digest: str, name: str = "cand") -> ScreenCandidate:
    return ScreenCandidate(name, digest, ("moe.fused_experts",))


class TestScreenQueue:
    def test_clear_winner_passes_without_escalation(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 112.0})
        report = run_resident_screen(
            session, [_candidate(DIGEST_A)], prompts=("p",),
        )
        [verdict] = report.verdicts
        assert verdict.passed
        assert not verdict.escalated
        assert report.stopped_reason is None
        # swap in, swap out — exactly two swaps for a clear verdict
        assert session.swaps == [DIGEST_A, None]

    def test_clear_loser_fails_without_escalation(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 80.0})
        report = run_resident_screen(
            session, [_candidate(DIGEST_A)], prompts=("p",),
        )
        [verdict] = report.verdicts
        assert not verdict.passed
        assert verdict.failure is None
        assert not verdict.escalated

    def test_borderline_escalates_to_five_legs(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 101.0})
        report = run_resident_screen(
            session, [_candidate(DIGEST_A)], prompts=("p",),
        )
        [verdict] = report.verdicts
        assert verdict.escalated
        assert len(verdict.candidate_throughputs) == 2
        assert len(verdict.baseline_throughputs) == 3
        # in, out, in, out — four swaps for an escalated verdict
        assert session.swaps == [DIGEST_A, None, DIGEST_A, None]

    def test_queue_reuses_brackets_across_candidates(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 112.0, DIGEST_B: 80.0})
        report = run_resident_screen(
            session,
            [_candidate(DIGEST_A, "a"), _candidate(DIGEST_B, "b")],
            prompts=("p",),
        )
        assert [v.passed for v in report.verdicts] == [True, False]
        # Zero engine reloads: 1 opening stock read + per candidate (C + closing B)
        assert session.batch_count == 5
        assert report.stopped_reason is None

    def test_slot_mismatch_fails_closed_and_returns_to_stock(self) -> None:
        session = FakeSession(
            100.0, {DIGEST_A: 112.0}, slots={DIGEST_A: ("other.slot",)}
        )
        report = run_resident_screen(
            session, [_candidate(DIGEST_A)], prompts=("p",),
        )
        [verdict] = report.verdicts
        assert not verdict.passed
        assert "differ from expected" in (verdict.failure or "")
        assert session.swaps == [DIGEST_A, None]
        assert session.active is None

    def test_canary_drift_stops_lifetime_and_withdraws_verdict(self) -> None:
        session = FakeSession(
            100.0,
            {DIGEST_A: 112.0, DIGEST_B: 112.0},
            stock_drift_after=2,
        )
        report = run_resident_screen(
            session,
            [_candidate(DIGEST_A, "a"), _candidate(DIGEST_B, "b")],
            prompts=("p",),
        )
        assert report.stopped_reason is not None
        assert "recycle" in report.stopped_reason
        withdrawn = report.verdicts[-1]
        assert withdrawn.verdict is None
        assert "withdrawn" in (withdrawn.failure or "")
        assert withdrawn.candidate_id in report.unprocessed_candidate_ids

    def test_lifetime_budget_stops_queue(self) -> None:
        session = FakeSession(100.0, {DIGEST_A: 112.0, DIGEST_B: 112.0})
        report = run_resident_screen(
            session,
            [_candidate(DIGEST_A, "a"), _candidate(DIGEST_B, "b")],
            prompts=("p",),
            policy=ScreenPolicy(max_candidates_per_lifetime=1),
        )
        assert len(report.verdicts) == 1
        assert report.unprocessed_candidate_ids == ("b",)
        assert report.stopped_reason == "lifetime candidate budget exhausted"

    def test_duplicate_candidate_ids_rejected(self) -> None:
        with pytest.raises(ResidentQueueError, match="unique"):
            run_resident_screen(
                FakeSession(100.0, {DIGEST_A: 110.0}),
                [_candidate(DIGEST_A, "x"), _candidate(DIGEST_B, "x")],
                prompts=("p",),
            )

    def test_empty_prompts_rejected(self) -> None:
        with pytest.raises(ResidentQueueError, match="prompt plan"):
            run_resident_screen(
                FakeSession(100.0, {DIGEST_A: 110.0}),
                [_candidate(DIGEST_A)],
                prompts=(),
            )
