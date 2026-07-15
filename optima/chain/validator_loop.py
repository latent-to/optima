"""Finalized chain intake, immutable publication, qualification, and settlement.

This production loop deliberately stops before weight signing.  It reserves
the complete finalized event order before network transport, publishes submitted bytes
into a separate immutable worker tree, optionally invokes the current batch causal
qualification authority, and transactionally adopts its retained PASS projection. The
old shell/CPU fake-score evaluator and JSON Ledger settlement do not exist on this path;
wallet access belongs only to the separate control-plane signer.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from optima import chain
from optima.arena_service import (
    AdmissionDecision,
    ArenaCandidateBinding,
    ArenaQualificationWork,
    ArenaService,
    ArenaServiceRegistry,
)
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
    QualificationAuthorityManifest,
    QualificationIntakeBatch,
    QualificationReservation,
    run_qualification_intake,
)


logger = logging.getLogger("optima.chain.validator")
DEFAULT_INTERVAL_S = 60.0


class IntakeControllerError(RuntimeError):
    """Validator-owned intake/qualification authority is inconsistent."""


# Compatibility names for code constructing trusted providers.  The live loop
# accepts only a closed ArenaServiceRegistry, never an arbitrary planner callback.
QualificationWork = ArenaQualificationWork


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
    settlements: dict[str, str] = field(default_factory=dict)
    screens: dict[str, str] = field(default_factory=dict)


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
                reservation.arrival.hotkey,
                reservation.arrival.block,
                reservation.arrival.event_index,
                reservation.arrival.event_subindex,
                reservation.target_members,
            )
        )
    return tuple(rows)


def _validate_work(
    work: ArenaQualificationWork,
    expected: tuple[QualificationReservation, ...],
) -> None:
    if type(work) is not ArenaQualificationWork:
        raise IntakeControllerError("qualification planner returned an untyped work item")
    if work.factory.manifest.reservations != expected:
        raise IntakeControllerError("qualification factory changed finalized cohort order")


def _apply_qualification(
    store: FinalizedIntakeStore,
    reservations: tuple[IntakeReservation, ...],
    publications: tuple[WorkerBundlePublication, ...],
    service: ArenaService,
) -> QualificationIntakeBatch:
    authority_rows = _qualification_reservations(reservations, publications)
    candidates = tuple(
        ArenaCandidateBinding(authority, publication, reservation.screen_attempts)
        for reservation, publication, authority in zip(
            reservations, publications, authority_rows, strict=True
        )
    )
    receipts = tuple(
        store.latest_promoted_screen(row.reservation_id) for row in reservations
    )
    work = service.plan_qualification(candidates, receipts, state=store)
    _validate_work(work, authority_rows)
    prepared = None
    if type(work.factory.manifest) is QualificationAuthorityManifest:
        prepared = work.factory.build()
        arms = tuple(row.arm for row in prepared.prepared.candidates)
        if (
            not arms
            or len({row.baseline_before for row in arms}) != 1
            or any(row.incumbent != arms[0].incumbent for row in arms)
        ):
            raise IntakeControllerError("qualification planner has no single incumbent")
        store.initialize_evaluation_stack(
            arms[0].incumbent,
            tree_digest=arms[0].baseline_before.tree_digest,
        )
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
    store.apply_qualification_batch(
        batch,
        evidence_root=None if prepared is None else prepared.evidence_root,
    )
    return batch


def _screen_pending(
    store: FinalizedIntakeStore,
    service: ArenaService,
    *,
    current_block: int,
) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for row in store.screenable(limit=store.policy.max_cohort):
        admission = service.admit(
            store.arena_queue_snapshot(current_block=current_block)
        )
        if admission is AdmissionDecision.QUEUE:
            break
        if admission is AdmissionDecision.HOLD:
            store.mark_held(row.reservation_id, "arena_screen_capacity_hold")
            decisions[row.reservation_id] = "hold"
            continue
        publication = reopen_worker_bundle(
            row.publication_root,
            row.arrival.content_hash,
            expected_receipt_digest=row.publication_digest,
        )
        active = store.begin_screen(
            row.reservation_id, service_digest=service.identity
        )
        authority = _qualification_reservations((active,), (publication,))[0]
        candidate = ArenaCandidateBinding(
            authority, publication, active.screen_attempts
        )
        receipt = service.screen(candidate)
        store.apply_screen_receipt(
            active.reservation_id,
            candidate_digest=candidate.digest,
            receipt=receipt,
        )
        decisions[active.reservation_id] = receipt.decision.value
    return decisions


def _settle_pending(
    store: FinalizedIntakeStore,
    *,
    current_block: int,
    finalized_block_provider: Callable[[], int],
) -> dict[str, str]:
    """Settle every causally ready retained PASS without chain or wallet access."""

    from optima.settlement import plan_settlement

    committed: dict[str, str] = {}
    while store.has_pending_settlement():
        lease_block = finalized_block_provider()
        if type(lease_block) is not int or lease_block < current_block:
            raise IntakeControllerError("finalized settlement clock regressed")
        current_block = lease_block
        lease = store.lease_settlement_cohort(current_block=current_block)
        if lease is None:
            return committed
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(candidate)
            for candidate in lease.candidates
        )
        refreshed_block = finalized_block_provider()
        if type(refreshed_block) is not int or refreshed_block < current_block:
            raise IntakeControllerError("finalized settlement clock regressed")
        store.commit_settlement(
            lease,
            plan,
            evidence,
            current_block=refreshed_block,
        )
        current_block = refreshed_block
        committed[lease.lease_id] = plan.digest
    return committed


def run_pass(
    subtensor,
    netuid: int,
    *,
    intake_db: str | Path,
    private_root: str | Path,
    publication_root: str | Path,
    policy: IntakePolicy = IntakePolicy(),
    arena_registry: ArenaServiceRegistry | None = None,
    arena_id: str | None = None,
    intake_only: bool = False,
    retained_only: bool = False,
) -> PassResult:
    """Run one non-emitting intake/qualification pass.

    ``retained_only`` evaluates the already-durable queue at the current
    finalized head without rereading or advancing reveal history.
    """

    if type(intake_only) is not bool or type(retained_only) is not bool:
        raise IntakeControllerError("pass mode flags must be exact booleans")
    if intake_only and retained_only:
        raise IntakeControllerError("intake-only and retained-only modes conflict")
    if intake_only:
        if arena_registry is not None or arena_id is not None:
            raise IntakeControllerError("intake-only mode cannot receive arena authority")
        service = None
    else:
        if type(arena_registry) is not ArenaServiceRegistry or not arena_id:
            raise IntakeControllerError(
                "live validation requires an injected registered arena service"
            )
        service = arena_registry.require(arena_id)

    scope = IntakeScope(str(subtensor.get_block_hash(0)).lower(), netuid)
    with FinalizedIntakeStore(intake_db, policy, scope=scope) as store:
        cursor = store.finalized_cursor()
        if retained_only:
            if cursor is None:
                raise IntakeControllerError("retained-only pass has no finalized cursor")
            finalized_block, finalized_hash = chain.read_finalized_head(subtensor)
            if finalized_block < cursor[0]:
                raise IntakeControllerError("retained-only finalized head regressed")
            result = PassResult(finalized_block, finalized_hash)
            inserted = ()
        else:
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
            result.published[published.reservation_id] = publication.digest

        # Publication and copy disposition are separate durable operations. Run a
        # complete idempotent reconciliation every pass so a crash in that window
        # cannot permanently bypass finalized priority.
        for copied, predecessor in store.reconcile_copies():
            result.copies[copied] = predecessor
            result.published.pop(copied, None)

        if service is not None:
            result.screens.update(
                _screen_pending(
                    store, service, current_block=result.finalized_block
                )
            )
            cohort = store.promoted(limit=policy.max_cohort)
            if cohort:
                admission = service.admit_qualification(
                    store.arena_queue_snapshot(
                        current_block=result.finalized_block
                    ),
                    cohort_size=len(cohort),
                )
                if admission is AdmissionDecision.HOLD:
                    for row in cohort:
                        store.mark_held(
                            row.reservation_id, "arena_qualification_capacity_hold"
                        )
                    cohort = ()
                elif admission is AdmissionDecision.QUEUE:
                    cohort = ()
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
                    store, cohort, publications, service
                )
                result.decisions.update(
                    (row.reservation_digest, row.decision.value)
                    for row in batch.outcomes
                )
            result.settlements.update(
                _settle_pending(
                    store,
                    current_block=result.finalized_block,
                    finalized_block_provider=lambda: chain.read_finalized_head(subtensor)[0],
                )
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
    arena_registry: ArenaServiceRegistry | None = None,
    arena_id: str | None = None,
    intake_only: bool = False,
    retained_only: bool = False,
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
                arena_registry=arena_registry,
                arena_id=arena_id,
                intake_only=intake_only,
                retained_only=retained_only,
            )
            failures = 0
            logger.info(
                "intake @finalized %d: seen=%d reserved=%d published=%d copies=%d "
                "rejected=%d decisions=%d settlements=%d held=%d",
                last.finalized_block,
                last.seen,
                len(last.reserved),
                len(last.published),
                len(last.copies),
                len(last.rejected),
                len(last.decisions),
                len(last.settlements),
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
    "IntakeControllerError", "PassResult", "QualificationWork", "run_pass",
    "run_validator",
]
