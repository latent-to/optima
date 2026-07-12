"""Finalized chain intake, immutable publication, and causal qualification.

This production loop deliberately stops before settlement and weights.  It reserves
the complete finalized event order before network transport, publishes submitted bytes
into a separate immutable worker tree, and optionally invokes the current batch causal
qualification authority.  The old shell/CPU fake-score evaluator and immediate JSON
Ledger settlement do not exist on this path.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from optima import chain
from optima.chain.fetch import FetchError, FetchTransientError, fetch_bundle
from optima.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakePolicy,
    IntakeReservation,
    IntakeScope,
)
from optima.chain.payload import decode_payload
from optima.chain.publication import (
    WorkerBundlePublication,
    WorkerBundlePublicationError,
    WorkerBundleSourceError,
    publish_worker_bundle,
    reopen_worker_bundle,
)
from optima.copy_fingerprint import fingerprint_submitted_delta
from optima.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationPlanFactory,
    QualificationReservation,
    run_qualification_intake,
)
from optima.stack_identity import canonical_digest


logger = logging.getLogger("optima.chain.validator")
DEFAULT_INTERVAL_S = 60.0


class IntakeControllerError(RuntimeError):
    """Validator-owned intake/qualification authority is inconsistent."""


@dataclass(frozen=True)
class QualificationWork:
    """Exact authorities required for one already-planned cohort execution."""

    factory: QualificationPlanFactory
    executor: object
    entropy_provider: object
    hidden_judge: object
    deadline: float

    def __post_init__(self) -> None:
        if type(self.factory) is not QualificationPlanFactory:
            raise IntakeControllerError("qualification work has no exact plan factory")
        if not callable(self.entropy_provider) or not callable(self.hidden_judge):
            raise IntakeControllerError("qualification work authorities are not callable")
        if (
            isinstance(self.deadline, bool)
            or not isinstance(self.deadline, (int, float))
            or not math.isfinite(float(self.deadline))
            or float(self.deadline) <= 0
        ):
            raise IntakeControllerError("qualification deadline is invalid")


QualificationPlanner = Callable[
    [
        tuple[IntakeReservation, ...],
        tuple[WorkerBundlePublication, ...],
        tuple[QualificationReservation, ...],
    ],
    QualificationWork,
]


@dataclass
class PassResult:
    finalized_block: int
    finalized_block_hash: str
    seen: int = 0
    reserved: list[str] = field(default_factory=list)
    published: dict[str, str] = field(default_factory=dict)
    copies: dict[str, str] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    decisions: dict[str, str] = field(default_factory=dict)
    held: list[str] = field(default_factory=list)


def _finalized_arrivals(snapshot) -> tuple[FinalizedArrival, ...]:
    rows: list[FinalizedArrival] = []
    for reveal in snapshot.reveals:
        payload_digest = hashlib.sha256(reveal.data.encode("utf-8")).hexdigest()
        ref = decode_payload(reveal.hotkey, reveal.block, reveal.data)
        if ref is None:
            rows.append(
                FinalizedArrival(
                    reveal.hotkey,
                    "",
                    "",
                    reveal.block,
                    reveal.block_hash.lower(),
                    reveal.event_index,
                    0,
                    payload_digest,
                    "invalid_payload",
                )
            )
            continue
        rows.append(
            FinalizedArrival(
                ref.hotkey,
                ref.content_hash,
                ref.url,
                reveal.block,
                reveal.block_hash.lower(),
                reveal.event_index,
                0,
                payload_digest,
            )
        )
    return tuple(rows)


def _fingerprint_private_bundle(root: Path):
    """Choose a lane by exact parser success; never by miner-provided mode alone."""

    component_error: Exception | None = None
    try:
        return fingerprint_submitted_delta(root)
    except (OSError, TypeError, ValueError) as exc:
        component_error = exc
    try:
        return fingerprint_submitted_delta(root, discovery=True)
    except (OSError, TypeError, ValueError) as discovery_error:
        raise ValueError(
            "submission is neither a registered component nor a closed discovery "
            f"proposal: component={component_error}; discovery={discovery_error}"
        ) from None


def _qualification_reservations(
    reservations: tuple[IntakeReservation, ...],
    publications: tuple[WorkerBundlePublication, ...],
) -> tuple[QualificationReservation, ...]:
    if len(reservations) != len(publications):
        raise IntakeControllerError("qualification publication coverage differs")
    rows: list[QualificationReservation] = []
    for index, (reservation, publication) in enumerate(
        zip(reservations, publications, strict=True)
    ):
        fingerprint = reservation.delta_fingerprint
        if (
            fingerprint is None
            or reservation.publication_digest != publication.digest
            or reservation.arrival.content_hash != publication.content_hash
        ):
            raise IntakeControllerError("qualification intake provenance differs")
        rows.append(
            QualificationReservation(
                reservation.reservation_id,
                publication.digest,
                fingerprint.target_id,
                fingerprint.selected_delta_digest,
                index,
            )
        )
    return tuple(rows)


def _validate_work(
    work: QualificationWork,
    expected: tuple[QualificationReservation, ...],
) -> None:
    if type(work) is not QualificationWork:
        raise IntakeControllerError("qualification planner returned an untyped work item")
    if work.factory.manifest.reservations != expected:
        raise IntakeControllerError("qualification factory changed finalized cohort order")


def _apply_qualification(
    store: FinalizedIntakeStore,
    reservations: tuple[IntakeReservation, ...],
    publications: tuple[WorkerBundlePublication, ...],
    planner: QualificationPlanner,
) -> QualificationIntakeBatch:
    authority_rows = _qualification_reservations(reservations, publications)
    work = planner(reservations, publications, authority_rows)
    _validate_work(work, authority_rows)
    authority_digest = work.factory.manifest.digest
    authority_manifest = work.factory.manifest.to_dict()
    for row in reservations:
        store.mark_qualifying(
            row.reservation_id, authority_digest, authority_manifest
        )
    batch = run_qualification_intake(
        work.factory,
        executor=work.executor,
        entropy_provider=work.entropy_provider,
        hidden_judge=work.hidden_judge,
        deadline=float(work.deadline),
    )
    if (
        type(batch) is not QualificationIntakeBatch
        or batch.authority_manifest_digest != authority_digest
        or tuple(row.reservation_digest for row in batch.outcomes)
        != tuple(row.reservation_id for row in reservations)
        or tuple(row.selected_delta_digest for row in batch.outcomes)
        != tuple(row.selected_delta_digest for row in authority_rows)
    ):
        raise IntakeControllerError("qualification outcomes changed cohort authority")
    for outcome in batch.outcomes:
        store.mark_outcome(
            outcome.reservation_digest,
            decision=outcome.decision.value,
            attempt_ref=(
                batch.attempt_ref
                if outcome.attempt_artifact_sha256 is not None
                else None
            ),
            report_digest=outcome.report_digest or "",
            failure_digest=outcome.failure_digest or "",
            reason=outcome.reason,
        )
    if batch.retry_plan is not None:
        for group_index, group in enumerate(
            batch.retry_plan.reservation_groups
        ):
            group_digest = canonical_digest(
                "optima.chain.qualification-retry-group",
                {
                    "authority_manifest_digest": batch.authority_manifest_digest,
                    "group_index": group_index,
                    "members": list(group),
                    "strategy": batch.retry_plan.strategy,
                },
            )
            for retry_position, reservation_id in enumerate(group):
                store.requeue_qualification(
                    reservation_id,
                    reason=f"qualification_{batch.retry_plan.strategy}",
                    retry_group_digest=group_digest,
                    retry_position=retry_position,
                )
    return batch


def run_pass(
    subtensor,
    netuid: int,
    *,
    intake_db: str | Path,
    private_root: str | Path,
    publication_root: str | Path,
    policy: IntakePolicy = IntakePolicy(),
    qualification_planner: QualificationPlanner | None = None,
) -> PassResult:
    """Run one non-emitting finalized intake/qualification pass."""

    scope = IntakeScope(str(subtensor.get_block_hash(0)).lower(), netuid)
    with FinalizedIntakeStore(intake_db, policy, scope=scope) as store:
        cursor = store.finalized_cursor()
        snapshot = chain.read_finalized_reveal_history(
            subtensor,
            netuid,
            after_block=None if cursor is None else cursor[0],
        )
        result = PassResult(snapshot.finalized_block, snapshot.finalized_block_hash)
        arrivals = _finalized_arrivals(snapshot)
        result.seen = len(arrivals)
        inserted = store.reserve_finalized(
            arrivals,
            finalized_block=snapshot.finalized_block,
            finalized_block_hash=snapshot.finalized_block_hash.lower(),
        )
        result.reserved.extend(row.reservation_id for row in inserted)

        for pending in store.pending(limit=policy.max_cohort):
            active = store.mark_fetching(pending.reservation_id)
            if active.status != "fetching":
                result.held.append(active.reservation_id)
                continue
            try:
                private = fetch_bundle(
                    active.arrival.url,
                    active.arrival.content_hash,
                    private_root,
                )
            except FetchTransientError as exc:
                store.mark_transport_retry(active.reservation_id, str(exc))
                continue
            except FetchError as exc:
                rejected = store.mark_failed(active.reservation_id, f"fetch:{exc}")
                result.rejected[rejected.reservation_id] = rejected.reason
                continue
            try:
                fingerprint = _fingerprint_private_bundle(private)
            except (OSError, TypeError, ValueError) as exc:
                rejected = store.mark_failed(active.reservation_id, f"manifest:{exc}")
                result.rejected[rejected.reservation_id] = rejected.reason
                continue
            try:
                publication = publish_worker_bundle(
                    private,
                    publication_root,
                    active.arrival.content_hash,
                )
            except WorkerBundleSourceError as exc:
                rejected = store.mark_failed(
                    active.reservation_id, f"publication_source:{exc}"
                )
                result.rejected[rejected.reservation_id] = rejected.reason
                continue
            except WorkerBundlePublicationError as exc:
                # Publication/storage faults are validator-side NO_DECISION, never a
                # miner loss. The bounded transport retry policy eventually holds it.
                store.mark_transport_retry(active.reservation_id, f"publication:{exc}")
                continue
            published = store.mark_published(
                active.reservation_id,
                delta_fingerprint=fingerprint,
                publication_digest=publication.digest,
                publication_root=publication.root,
            )
            if published.status != "published":
                result.rejected[published.reservation_id] = published.reason
                continue
            predecessors = store.copy_predecessors(published.reservation_id)
            if predecessors:
                copied = store.mark_copy(
                    published.reservation_id, predecessors[0].reservation_id
                )
                result.copies[copied.reservation_id] = predecessors[0].reservation_id
                continue
            result.published[published.reservation_id] = publication.digest
            for successor in store.copy_successors(published.reservation_id):
                copied = store.mark_copy(
                    successor.reservation_id, published.reservation_id
                )
                result.copies[copied.reservation_id] = published.reservation_id

        if qualification_planner is not None:
            cohort = store.published(limit=policy.max_cohort)
            if cohort:
                publications = tuple(
                    reopen_worker_bundle(
                        row.publication_root,
                        row.arrival.content_hash,
                        expected_receipt_digest=row.publication_digest,
                    )
                    for row in cohort
                )
                batch = _apply_qualification(
                    store, cohort, publications, qualification_planner
                )
                result.decisions.update(
                    (row.reservation_digest, row.decision.value)
                    for row in batch.outcomes
                )
        result.rejected.update(
            (row.reservation_id, row.reason)
            for row in inserted
            if row.status == "failed"
        )
        result.held.extend(
            row.reservation_id for row in store.all() if row.status == "held"
        )
    result.held = sorted(set(result.held))
    return result


def run_validator(
    subtensor,
    netuid: int,
    *,
    intake_db: str | Path,
    private_root: str | Path,
    publication_root: str | Path,
    policy: IntakePolicy = IntakePolicy(),
    qualification_planner: QualificationPlanner | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
    once: bool = False,
    max_consecutive_failures: int = 10,
) -> Optional[PassResult]:
    """Run finalized intake forever, containing validator-side pass failures."""

    failures = 0
    last: Optional[PassResult] = None
    while True:
        try:
            last = run_pass(
                subtensor,
                netuid,
                intake_db=intake_db,
                private_root=private_root,
                publication_root=publication_root,
                policy=policy,
                qualification_planner=qualification_planner,
            )
            failures = 0
            logger.info(
                "intake @finalized %d: seen=%d reserved=%d published=%d copies=%d "
                "rejected=%d decisions=%d held=%d",
                last.finalized_block,
                last.seen,
                len(last.reserved),
                len(last.published),
                len(last.copies),
                len(last.rejected),
                len(last.decisions),
                len(last.held),
            )
        except Exception:  # validator-side fault; a supervisor may restart cleanly
            failures += 1
            logger.exception("validator intake pass failed (%d consecutive)", failures)
            if once or failures >= max_consecutive_failures:
                raise
        if once:
            return last
        time.sleep(float(interval_s) * (1 + min(failures, 5)))


__all__ = [
    "IntakeControllerError", "PassResult", "QualificationPlanner",
    "QualificationWork", "run_pass", "run_validator",
]
