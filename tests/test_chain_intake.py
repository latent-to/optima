from __future__ import annotations

import os
import json

import pytest

import optima.cli as cli
from optima import chain
from optima.arena_service import (
    SCREEN_STAGES, ArenaScreenReceipt, PromotionDecision, ScreenGrade,
    ScreenStageResult,
)
from optima.chain.intake import (
    FinalizedArrival, FinalizedIntakeStore, IntakeError, IntakePolicy,
    IntakeScope, SQLiteWeightPublicationJournal,
)
from optima.chain.weights import WeightProjection, WeightPublicationRecord
from optima.copy_fingerprint import SubmittedDeltaFingerprint
from optima.discovery import DiscoveryArmPlan
from optima.eval.evidence_store import EvidenceArtifactRef, publish_evidence
from optima.eval.oci_session_protocol import SlotAuditPolicy
from optima.eval.qualification import QualificationDecision
from optima.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationRetryPlan,
)
from optima.economics import (
    DiscoveryBountyClaim, EconomicsError,
    EmissionsPolicyManifest,
    GlobalRewardProjectionContext,
    MetagraphMember,
    StandingRewardClaim,
)
from optima.settlement import (
    SettlementCandidate, SettlementEventType, SettlementQualification,
    plan_settlement,
)
from optima.stack_identity import sha256_hex
from optima.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    ProposalContributionRef,
)
from optima.stack_plan import plan_marginal_arm
from optima.target_catalog import TargetCatalog, default_target_catalog


SCOPE = IntakeScope("0x" + "0" * 64, 307)
AUTHORITY = {"schema": "test-authority"}
ATTEMPT = EvidenceArtifactRef(
    "qualification.cohort-attempt",
    "9" * 64,
    1,
    "application/json",
    "optima.qualification.cohort-attempt.v1",
)


def _arrival(index: int, *, hotkey: str = "miner", block: int = 10) -> FinalizedArrival:
    return FinalizedArrival(
        hotkey=hotkey,
        content_hash=f"{index + 1:064x}",
        url=f"https://example.com/{index}.tar.gz",
        block=block,
        block_hash="0x" + f"{block:064x}",
        event_index=index,
    )


def _store(tmp_path, **policy):
    return FinalizedIntakeStore(
        tmp_path / "private" / "intake.sqlite3",
        IntakePolicy(**policy),
        scope=SCOPE,
    )


def _fingerprint(
    target: str,
    member: str,
    marker: str = "a",
    *,
    selected_delta: str = "3" * 64,
):
    return SubmittedDeltaFingerprint(
        "component", target, "1" * 64, (member,), "2" * 64,
        selected_delta, "4" * 64, (marker * 64,), ("5" * 64,),
    )


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _audit_policy(label: str, slots: tuple[str, ...]) -> SlotAuditPolicy:
    return SlotAuditPolicy(_h(f"audit-seed:{label}")[:32], 100_000, 32, slots, 1)


def _promote(store: FinalizedIntakeStore, reservation_id: str) -> None:
    active = store.begin_screen(reservation_id, service_digest=_h("service"))
    candidate_digest = _h(f"candidate:{reservation_id}:{active.screen_attempts}")
    receipt = ArenaScreenReceipt(
        _h("service"),
        candidate_digest,
        active.screen_attempts,
        tuple(
            ScreenStageResult(stage, ScreenGrade.PASS, _h(stage), 1)
            for stage in SCREEN_STAGES
        ),
        PromotionDecision.PROMOTE,
    )
    store.apply_screen_receipt(
        reservation_id, candidate_digest=candidate_digest, receipt=receipt
    )


def _stack_context(catalog: TargetCatalog) -> EvaluationStackContext:
    targets = catalog.snapshot()["targets"]
    assert isinstance(targets, list)
    return EvaluationStackContext(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests={
            row["target_id"]: catalog.target_spec_digest(row["target_id"])
            for row in targets
        },
    )


def _qualified_settlement_candidate(
    store: FinalizedIntakeStore,
    *,
    primary_only: bool = False,
    retained_block: int = 10,
) -> SettlementCandidate | str:
    catalog = default_target_catalog()
    target = "activation.silu_and_mul"
    incumbent = EvaluationStackManifest(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )
    replacement = ProposalContributionRef(
        target_id=target,
        target_spec_digest=catalog.target_spec_digest(target),
        artifact_digest=_h("artifact"),
        selected_payload_digest=_h("payload"),
        attribution_digest=_h("attribution"),
    )
    arm = plan_marginal_arm(
        incumbent,
        replacement,
        catalog=catalog,
        incumbent_tree_digest=_h("incumbent-tree"),
        candidate_tree_digest=_h("candidate-tree"),
        expected_context=_stack_context(catalog),
    )
    store.initialize_evaluation_stack(
        incumbent, tree_digest=arm.baseline_before.tree_digest
    )
    evidence_root = store.path.parent / "evidence"
    primary_attempt = publish_evidence(
        evidence_root,
        b"retained primary qualification attempt",
        domain="qualification.cohort-attempt",
        media_type="application/json",
        schema="optima.qualification.cohort-attempt.v1",
    )
    reproduction_attempt = publish_evidence(
        evidence_root,
        b"retained reproduction qualification attempt",
        domain="qualification.cohort-attempt",
        media_type="application/json",
        schema="optima.qualification.cohort-attempt.v1",
    )
    row = store.reserve_finalized(
        (_arrival(0),),
        finalized_block=10,
        finalized_block_hash="0x" + f"{10:064x}",
    )[0]
    store.mark_fetching(row.reservation_id)
    store.mark_published(
        row.reservation_id,
        delta_fingerprint=_fingerprint(
            target,
            target,
            selected_delta=arm.selected_delta_digest,
        ),
        publication_digest="d" * 64,
        publication_root="/published/candidate",
    )
    _promote(store, row.reservation_id)
    def qualification(marker: str, authority: str, attempt, speedup: str):
        audit_policy = _audit_policy(marker, (target,))
        return SettlementQualification(
            lane="registered",
            arena_digest=incumbent.arena_digest,
            reservation_digest=row.reservation_id,
            finalized_block=row.arrival.block,
            event_index=row.arrival.event_index,
            event_subindex=row.arrival.event_subindex,
            hotkey=row.arrival.hotkey,
            target_id=target,
            members=(target,),
            selected_delta_digest=arm.selected_delta_digest,
            qualification_authority_digest=authority,
            qualification_plan_digest=_h("plan-" + marker),
            qualification_attempt_digest=attempt.sha256,
            qualification_report_digest=_h("report-" + marker),
            selection_commitment_digest=_h("commitment-" + marker),
            selection_secret_commitment_digest=_h("secret-" + marker),
            selection_evidence_digest=_h("selection-" + marker),
            arm_digest=arm.digest,
            incumbent_stack_digest=arm.baseline_before.stack_digest,
            incumbent_tree_digest=arm.baseline_before.tree_digest,
            candidate_stack_digest=arm.challenger.stack_digest,
            candidate_tree_digest=arm.challenger.tree_digest,
            speedup=speedup,
            incumbent_manifest=incumbent,
            candidate_manifest=arm.candidate,
            audit_control_digest=audit_policy.control.digest,
            audit_policy=audit_policy,
            audit_evidence_digest=_h("audit-evidence-" + marker),
        )

    authorities = (_h("primary-authority"), _h("reproduction-authority"))
    qualifications = (
        qualification("primary", authorities[0], primary_attempt, "1.05"),
        qualification("reproduction", authorities[1], reproduction_attempt, "1.04"),
    )
    for index, (authority, attempt, settled) in enumerate(
        zip(
            authorities,
            (primary_attempt, reproduction_attempt),
            qualifications,
            strict=True,
        )
    ):
        if index:
            _promote(store, row.reservation_id)
        store.mark_qualifying(row.reservation_id, authority, AUTHORITY)
        outcome = QualificationIntakeOutcome(
            row.reservation_id,
            arm.selected_delta_digest,
            authority,
            QualificationDecision.PASS,
            "qualified",
            False,
            attempt_artifact_sha256=attempt.sha256,
            report_digest=settled.qualification_report_digest,
            settlement_qualification=settled,
        )
        store.apply_qualification_batch(
            QualificationIntakeBatch(authority, (outcome,), attempt),
            current_finalized_block=retained_block,
            evidence_root=evidence_root,
        )
        if index == 0:
            assert store.get(row.reservation_id).status == "reproduction_pending"
            assert store.lease_settlement_cohort(
                current_block=max(11, retained_block)
            ) is None
            if primary_only:
                return row.reservation_id
    return SettlementCandidate.from_reproductions(*qualifications)


def _discovery_fingerprint(
    proposal_digest: str, *, selected_delta_digest: str | None = None
) -> SubmittedDeltaFingerprint:
    return SubmittedDeltaFingerprint(
        "discovery",
        "discovery",
        "",
        ("discovery",),
        proposal_digest,
        selected_delta_digest or proposal_digest,
        _h(f"normalized:{proposal_digest}"),
        (),
        (),
    )


def _qualified_discovery_candidate(
    store: FinalizedIntakeStore,
    *,
    index: int,
    proposal_digest: str,
    hotkey: str = "miner",
    bypass_publication_dedup: bool = False,
) -> SettlementCandidate:
    catalog = default_target_catalog()
    incumbent = EvaluationStackManifest(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )
    arm = DiscoveryArmPlan.create(
        incumbent=incumbent,
        incumbent_tree_digest=_h("incumbent-tree"),
        candidate_tree_digest=_h(f"discovery-tree:{index}"),
        proposal_digest=proposal_digest,
        policy_digest=_h("discovery-policy"),
        build_profile_digest=_h("build-profile"),
        overlay_identity_digest=_h(f"overlay:{index}"),
    )
    store.initialize_evaluation_stack(
        incumbent, tree_digest=arm.baseline_before.tree_digest
    )
    evidence_root = store.path.parent / "evidence"
    attempts = tuple(
        publish_evidence(
            evidence_root,
            f"retained discovery {index} {lane}".encode(),
            domain="qualification.cohort-attempt",
            media_type="application/json",
            schema="optima.qualification.cohort-attempt.v1",
        )
        for lane in ("primary", "reproduction")
    )
    row = store.reserve_finalized(
        (_arrival(index, hotkey=hotkey),),
        finalized_block=10,
        finalized_block_hash="0x" + f"{10:064x}",
    )[0]
    store.mark_fetching(row.reservation_id)
    published = store.mark_published(
        row.reservation_id,
        delta_fingerprint=_discovery_fingerprint(
            proposal_digest, selected_delta_digest=arm.selected_delta_digest
        ),
        publication_digest=_h(f"publication:{index}"),
        publication_root=f"/published/discovery-{index}",
    )
    if bypass_publication_dedup:
        assert published.status == "failed"
        awarded = store._db.execute(
            "SELECT 1 FROM discovery_bounty_claims WHERE proposal_digest=?",
            (proposal_digest,),
        ).fetchone()
        assert published.reason == (
            "already_awarded" if awarded is not None else "duplicate_proposal"
        )
        store._db.execute(
            "UPDATE reservations SET status='published',decision='',reason='' "
            "WHERE reservation_id=?",
            (row.reservation_id,),
        )
    else:
        assert published.status == "published"

    def qualification(lane: str, attempt, speedup: str) -> SettlementQualification:
        audit_policy = _audit_policy(
            f"discovery:{index}:{lane}", ("discovery",)
        )
        return SettlementQualification(
            lane="discovery",
            arena_digest=incumbent.arena_digest,
            reservation_digest=row.reservation_id,
            finalized_block=row.arrival.block,
            event_index=row.arrival.event_index,
            event_subindex=row.arrival.event_subindex,
            hotkey=row.arrival.hotkey,
            target_id="discovery",
            members=("discovery",),
            selected_delta_digest=arm.selected_delta_digest,
            qualification_authority_digest=_h(f"authority:{index}:{lane}"),
            qualification_plan_digest=_h(f"plan:{index}:{lane}"),
            qualification_attempt_digest=attempt.sha256,
            qualification_report_digest=_h(f"report:{index}:{lane}"),
            selection_commitment_digest=_h(f"commitment:{index}:{lane}"),
            selection_secret_commitment_digest=_h(f"secret:{index}:{lane}"),
            selection_evidence_digest=_h(f"selection:{index}:{lane}"),
            arm_digest=arm.digest,
            incumbent_stack_digest=arm.baseline_before.stack_digest,
            incumbent_tree_digest=arm.baseline_before.tree_digest,
            candidate_stack_digest=arm.challenger.stack_digest,
            candidate_tree_digest=arm.challenger.tree_digest,
            speedup=speedup,
            incumbent_manifest=incumbent,
            proposal_digest=proposal_digest,
            audit_control_digest=audit_policy.control.digest,
            audit_policy=audit_policy,
            audit_evidence_digest=_h(f"audit-evidence:{index}:{lane}"),
        )

    qualifications = (
        qualification("primary", attempts[0], "1.03"),
        qualification("reproduction", attempts[1], "1.02"),
    )
    for attempt, settled in zip(attempts, qualifications, strict=True):
        _promote(store, row.reservation_id)
        store.mark_qualifying(
            row.reservation_id,
            settled.qualification_authority_digest,
            AUTHORITY,
        )
        outcome = QualificationIntakeOutcome(
            row.reservation_id,
            arm.selected_delta_digest,
            settled.qualification_authority_digest,
            QualificationDecision.PASS,
            "qualified",
            False,
            attempt_artifact_sha256=attempt.sha256,
            report_digest=settled.qualification_report_digest,
            settlement_qualification=settled,
        )
        store.apply_qualification_batch(
            QualificationIntakeBatch(
                settled.qualification_authority_digest, (outcome,), attempt
            ),
            current_finalized_block=10,
            evidence_root=evidence_root,
        )
    return SettlementCandidate.from_reproductions(*qualifications)


def test_finalized_batch_is_reserved_atomically_before_transport(tmp_path):
    rows = (_arrival(0), _arrival(1, hotkey="other"))
    with _store(tmp_path) as store:
        reserved = store.reserve_finalized(
            rows, finalized_block=10, finalized_block_hash="0x" + f"{10:064x}"
        )
        assert tuple(row.arrival for row in reserved) == rows
        assert store.pending() == reserved
        assert oct(os.stat(store.path).st_mode & 0o777) == "0o600"
        for suffix in ("-wal", "-shm"):
            sidecar = store.path.with_name(store.path.name + suffix)
            if sidecar.exists():
                assert oct(os.stat(sidecar).st_mode & 0o777) == "0o600"

    with _store(tmp_path) as reopened:
        assert tuple(row.arrival for row in reopened.all()) == rows
        assert reopened.reserve_finalized(
            rows, finalized_block=10, finalized_block_hash="0x" + f"{10:064x}"
        ) == ()


def test_malformed_payload_still_reserves_its_finalized_position(tmp_path):
    invalid = FinalizedArrival(
        "miner", "", "", 10, "0x" + f"{10:064x}", 4, 0,
        "9" * 64, "invalid_payload",
    )
    with _store(tmp_path) as store:
        row = store.reserve_finalized(
            (invalid,), finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        assert row.status == "failed" and row.reason == "invalid_payload"
        assert store.pending() == ()


def test_store_rejects_a_symlink_database(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    target = tmp_path / "elsewhere"
    target.write_text("do not overwrite")
    (private / "intake.sqlite3").symlink_to(target)
    with pytest.raises(IntakeError, match="symlink"):
        FinalizedIntakeStore(private / "intake.sqlite3", scope=SCOPE)


def test_store_rejects_an_existing_nonprivate_parent(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o755)
    with pytest.raises(IntakeError, match="mode 0700"):
        FinalizedIntakeStore(parent / "intake.sqlite3", scope=SCOPE)


def test_store_binds_chain_scope_and_excludes_a_second_controller(tmp_path):
    path = tmp_path / "private" / "intake.sqlite3"
    with _store(tmp_path):
        with pytest.raises(IntakeError, match="another intake controller"):
            FinalizedIntakeStore(path, scope=SCOPE)
    with pytest.raises(IntakeError, match="another chain scope"):
        FinalizedIntakeStore(
            path, scope=IntakeScope("0x" + "1" * 64, SCOPE.netuid)
        )


def test_finalized_cursor_rejects_hash_change_or_regression(tmp_path):
    with _store(tmp_path) as store:
        store.reserve_finalized(
            (_arrival(0),), finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )
        with pytest.raises(IntakeError, match="cursor"):
            store.reserve_finalized(
                (), finalized_block=10, finalized_block_hash="0x" + "f" * 64
            )
        with pytest.raises(IntakeError, match="cursor"):
            store.reserve_finalized(
                (), finalized_block=9, finalized_block_hash="0x" + f"{9:064x}"
            )


def test_cursor_rejection_rolls_back_automatic_expiry(tmp_path, monkeypatch):
    with _store(tmp_path, expiry_blocks=20) as store:
        row = store.reserve_finalized(
            (_arrival(0),), finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        # Force the cursor check after the in-transaction expiry update to reject
        # this proposed head.  No partial liveness transition may survive.
        monkeypatch.setattr(
            store,
            "_cursor",
            lambda: (31, "0x" + f"{31:064x}"),
        )
        with pytest.raises(IntakeError, match="cursor"):
            store.reserve_finalized(
                (), finalized_block=30,
                finalized_block_hash="0x" + f"{30:064x}",
            )
        assert store.get(row.reservation_id).status == "reserved"


def test_restart_holds_interrupted_work_instead_of_replaying(tmp_path):
    with _store(tmp_path) as store:
        row = store.reserve_finalized(
            (_arrival(0),), finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        store.mark_fetching(row.reservation_id)
    with _store(tmp_path) as reopened:
        held = reopened.get(row.reservation_id)
        assert held.status == "held" and held.decision == "NO_DECISION"
        assert reopened.pending() == ()


def test_restart_applies_finalized_sla_before_admitting_a_new_arrival(tmp_path):
    with _store(tmp_path, max_pending=1, max_cohort=1, expiry_blocks=20) as store:
        stale = store.reserve_finalized(
            (_arrival(0),), finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        store.mark_fetching(stale.reservation_id)

    with _store(
        tmp_path, max_pending=1, max_cohort=1, expiry_blocks=20
    ) as reopened:
        assert reopened.get(stale.reservation_id).status == "held"
        delayed, admitted = reopened.reserve_finalized(
            (
                _arrival(2, hotkey="late-miner", block=10),
                _arrival(1, block=30),
            ), finalized_block=30,
            finalized_block_hash="0x" + f"{30:064x}",
        )
        expired = reopened.get(stale.reservation_id)
        assert (expired.status, expired.decision, expired.reason) == (
            "expired", "NO_DECISION", "finalized_block_sla_expired",
        )
        assert (delayed.status, delayed.reason) == (
            "expired", "finalized_block_sla_expired",
        )
        assert admitted.status == "reserved"
        assert reopened.pending() == (admitted,)


def test_admission_bounds_and_epoch_cutoff_are_durable(tmp_path):
    policy = dict(
        max_per_hotkey_epoch=1,
        max_pending=2,
        max_cohort=2,
        epoch_blocks=100,
        cutoff_blocks=10,
    )
    with _store(tmp_path, **policy) as store:
        rows = (
            _arrival(0, block=95),
            _arrival(1, block=96),
            _arrival(2, hotkey="other", block=97),
        )
        result = store.reserve_finalized(
            rows, finalized_block=100, finalized_block_hash="0x" + f"{100:064x}"
        )
        assert [row.admission_epoch for row in result] == [1, 1, 1]
        assert [row.status for row in result] == ["reserved", "failed", "reserved"]


def test_unknown_older_and_overlapping_target_block_later_settlement(tmp_path):
    with _store(tmp_path) as store:
        first, second, third = store.reserve_finalized(
            (_arrival(0), _arrival(1, hotkey="b"), _arrival(2, hotkey="c")),
            finalized_block=10, finalized_block_hash="0x" + f"{10:064x}",
        )
        for row, target, members in (
            (second, "target.a", ("slot.a",)),
            (third, "target.b", ("slot.b",)),
        ):
            store.mark_fetching(row.reservation_id)
            store.mark_published(
                row.reservation_id, delta_fingerprint=_fingerprint(target, members[0]),
                publication_digest="d" * 64, publication_root=f"/published/{target}",
            )
        assert store.settlement_blockers(second.reservation_id) == (first,)
        assert store.settlement_blockers(third.reservation_id) == (first,)

        store.mark_fetching(first.reservation_id)
        store.mark_published(
            first.reservation_id, delta_fingerprint=_fingerprint("target.a", "slot.a", "b"),
            publication_digest="e" * 64, publication_root="/published/first",
        )
        assert store.settlement_blockers(second.reservation_id) == (store.get(first.reservation_id),)
        assert store.settlement_blockers(third.reservation_id) == ()


def test_copy_decision_uses_only_durable_delta_fingerprints(tmp_path):
    with _store(tmp_path) as store:
        first, second = store.reserve_finalized(
            (_arrival(0, hotkey="author"), _arrival(1, hotkey="copycat")),
            finalized_block=10, finalized_block_hash="0x" + f"{10:064x}",
        )
        for row in (first, second):
            store.mark_fetching(row.reservation_id)
            store.mark_published(
                row.reservation_id,
                delta_fingerprint=_fingerprint("target.a", "slot.a"),
                publication_digest="d" * 64,
                publication_root=f"/published/{row.reservation_id}",
            )
        assert store.copy_predecessors(second.reservation_id) == (
            store.get(first.reservation_id),
        )
        copied = store.mark_copy(second.reservation_id, first.reservation_id)
        assert copied.status == "failed" and copied.decision == "FAIL"


def test_expiry_and_retry_release_are_explicit(tmp_path):
    with _store(tmp_path, expiry_blocks=20) as store:
        row = store.reserve_finalized(
            (_arrival(0),), finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        store.mark_fetching(row.reservation_id)
        store.mark_transport_retry(row.reservation_id, "host unavailable")
        with pytest.raises(IntakeError, match="not old enough"):
            store.expire(row.reservation_id, current_block=29, reason="operator expiry")
        expired = store.expire(row.reservation_id, current_block=30, reason="operator expiry")
        assert expired.status == "expired" and expired.decision == "NO_DECISION"


def test_transport_retry_exhaustion_becomes_an_explicit_hold(tmp_path):
    with _store(tmp_path, max_transport_retries=1) as store:
        row = store.reserve_finalized(
            (_arrival(0),),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        store.mark_fetching(row.reservation_id)
        held = store.mark_transport_retry(row.reservation_id, "host unavailable")
        assert held.status == "held"
        assert held.decision == "NO_DECISION"
        assert held.reason == "transport_retry_limit"
        released = store.release_hold(
            row.reservation_id, reason="operator granted one fresh transport budget"
        )
        assert released.status == "transport_retry"
        assert released.transport_attempts == 0
        assert store.pending() == (released,)


def test_finalized_sla_removes_old_blocker_but_preserves_settled_candidate(tmp_path):
    with _store(tmp_path, max_transport_retries=1, expiry_blocks=20) as store:
        blocker = store.reserve_finalized(
            (_arrival(99, block=9),), finalized_block=9,
            finalized_block_hash="0x" + f"{9:064x}",
        )[0]
        store.mark_fetching(blocker.reservation_id)
        blocker = store.mark_transport_retry(
            blocker.reservation_id, "host unavailable"
        )
        assert blocker.status == "held" and blocker.target_members == ()

        candidate = _qualified_settlement_candidate(store)
        assert store.lease_settlement_cohort(current_block=28) is None
        lease = store.lease_settlement_cohort(current_block=29)
        assert lease is not None and lease.candidates == (candidate,)
        expired = store.get(blocker.reservation_id)
        assert (expired.status, expired.reason) == (
            "expired", "finalized_block_sla_expired",
        )
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        store.commit_settlement(lease, plan, evidence, current_block=30)
        assert store.get(candidate.reservation_digest).status == "qualified"
        assert store.expire_stale(current_block=100) == ()
        assert store.get(candidate.reservation_digest).status == "qualified"
        assert store.reopen_active_crown(
            candidate.arena_digest, candidate.target_id
        ).candidate == candidate


def test_finalized_sla_resets_on_retained_primary_and_survives_restart(tmp_path):
    with _store(tmp_path, expiry_blocks=20) as store:
        reservation_id = _qualified_settlement_candidate(
            store, primary_only=True, retained_block=29
        )
        assert isinstance(reservation_id, str)

        # Arrival block 10 would expire at 30.  The primary PASS retained at 29
        # resets the same 20-block SLA, giving reproduction through block 48.
        assert store.expire_stale(current_block=30) == ()
        retained = store.get(reservation_id)
        assert retained.status == "reproduction_pending"
        progress = store._db.execute(
            "SELECT retained_block FROM settlement_qualifications "
            "WHERE reservation_id=? AND reproduction_index=0",
            (reservation_id,),
        ).fetchone()
        assert progress["retained_block"] == 29

    with _store(tmp_path, expiry_blocks=20) as reopened:
        assert reopened.expire_stale(current_block=48) == ()
        expired = reopened.expire_stale(current_block=49)
        assert tuple(row.reservation_id for row in expired) == (reservation_id,)
        assert (
            expired[0].status,
            expired[0].decision,
            expired[0].reason,
        ) == (
            "expired", "NO_DECISION", "finalized_block_sla_expired"
        )


def test_legacy_retained_primary_unknown_block_stays_manual(tmp_path):
    with _store(tmp_path, expiry_blocks=20) as store:
        reservation_id = _qualified_settlement_candidate(
            store, primary_only=True
        )
        assert isinstance(reservation_id, str)
        # Simulate the exact additive migration input: an existing schema-3
        # qualification table from before retained progress was recorded.
        store._db.execute(
            "ALTER TABLE settlement_qualifications DROP COLUMN retained_block"
        )

    with _store(tmp_path, expiry_blocks=20) as reopened:
        progress = reopened._db.execute(
            "SELECT retained_block FROM settlement_qualifications "
            "WHERE reservation_id=? AND reproduction_index=0",
            (reservation_id,),
        ).fetchone()
        assert progress["retained_block"] == 0
        assert reopened.expire_stale(current_block=100) == ()
        expired = reopened.expire(
            reservation_id,
            current_block=100,
            reason="operator archived legacy retained PASS",
        )
        assert (expired.status, expired.reason) == (
            "expired", "operator archived legacy retained PASS"
        )


def test_schema3_migration_hold_survives_all_generic_expiry_paths(tmp_path):
    with _store(tmp_path, expiry_blocks=20) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        # Reopen through the real v2 -> v3 migration path.
        store._db.execute("UPDATE metadata SET value='2' WHERE key='schema'")

    with _store(tmp_path, expiry_blocks=20) as reopened:
        held = reopened.get(candidate.reservation_digest)
        assert (held.status, held.decision, held.reason) == (
            "held",
            "NO_DECISION",
            "schema3_reproduction_required",
        )
        assert reopened.expire_stale(current_block=100) == ()
        assert reopened.get(candidate.reservation_digest) == held
        with pytest.raises(IntakeError, match="archival migration"):
            reopened.expire(
                candidate.reservation_digest,
                current_block=100,
                reason="generic operator expiry",
            )
        with pytest.raises(IntakeError, match="archival migration"):
            reopened.release_hold(
                candidate.reservation_digest,
                reason="generic operator release",
            )


def test_schema3_archival_is_terminal_preserves_evidence_and_releases_priority(
    tmp_path,
):
    with _store(tmp_path, expiry_blocks=20) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        store._db.execute("UPDATE metadata SET value='2' WHERE key='schema'")

    with _store(tmp_path, expiry_blocks=20) as reopened:
        legacy = reopened.get(candidate.reservation_digest)
        candidate_before = dict(
            reopened._db.execute(
                "SELECT * FROM settlement_candidates WHERE reservation_id=?",
                (candidate.reservation_digest,),
            ).fetchone()
        )
        qualifications_before = tuple(
            tuple(row)
            for row in reopened._db.execute(
                "SELECT * FROM settlement_qualifications WHERE reservation_id=? "
                "ORDER BY reproduction_index",
                (candidate.reservation_digest,),
            )
        )

        later = reopened.reserve_finalized(
            (_arrival(1, block=11),),
            finalized_block=11,
            finalized_block_hash="0x" + f"{11:064x}",
        )[0]
        reopened.mark_fetching(later.reservation_id)
        reopened.mark_published(
            later.reservation_id,
            delta_fingerprint=_fingerprint(
                candidate.target_id,
                candidate.target_id,
                "b",
                selected_delta="6" * 64,
            ),
            publication_digest="e" * 64,
            publication_root="/published/later",
        )
        assert reopened.settlement_blockers(later.reservation_id) == (legacy,)

        archived = reopened.archive_schema3_migration_hold(
            candidate.reservation_digest,
            current_finalized_block=11,
            reason="operator verified legacy evidence remains audit-only",
        )
        assert (archived.status, archived.decision) == ("expired", "NO_DECISION")
        assert archived.reason.startswith("schema3_archived@11:")
        assert reopened.settlement_blockers(later.reservation_id) == ()

        candidate_after = dict(
            reopened._db.execute(
                "SELECT * FROM settlement_candidates WHERE reservation_id=?",
                (candidate.reservation_digest,),
            ).fetchone()
        )
        assert candidate_after["status"] == "held"
        assert candidate_after["reason"] == archived.reason
        assert candidate_after["candidate_json"] == candidate_before["candidate_json"]
        assert candidate_after["candidate_digest"] == candidate_before["candidate_digest"]
        assert candidate_after["evidence_root"] == candidate_before["evidence_root"]
        assert candidate_after["reproduction_evidence_root"] == candidate_before[
            "reproduction_evidence_root"
        ]
        assert tuple(
            tuple(row)
            for row in reopened._db.execute(
                "SELECT * FROM settlement_qualifications WHERE reservation_id=? "
                "ORDER BY reproduction_index",
                (candidate.reservation_digest,),
            )
        ) == qualifications_before
        assert reopened.has_pending_settlement() is False
        assert reopened.lease_settlement_cohort(current_block=11) is None
        with pytest.raises(IntakeError, match="only held intake"):
            reopened.release_hold(
                candidate.reservation_digest,
                reason="must not restore crown eligibility",
            )
        with pytest.raises(IntakeError, match="active crown"):
            reopened.reopen_active_crown(candidate.arena_digest, candidate.target_id)
        with pytest.raises(IntakeError, match="exact schema3"):
            reopened.archive_schema3_migration_hold(
                candidate.reservation_digest,
                current_finalized_block=12,
                reason="must not archive twice",
            )


def test_schema3_archival_rejects_ordinary_or_inconsistent_holds(tmp_path):
    with _store(tmp_path) as store:
        ordinary = store.reserve_finalized(
            (_arrival(0),),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        ordinary = store.mark_held(ordinary.reservation_id, "ordinary operator hold")
        with pytest.raises(IntakeError, match="exact schema3"):
            store.archive_schema3_migration_hold(
                ordinary.reservation_id,
                current_finalized_block=10,
                reason="must not archive an ordinary hold",
            )
        assert store.get(ordinary.reservation_id) == ordinary

    other_root = tmp_path / "inconsistent"
    with _store(other_root) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        store._db.execute("UPDATE metadata SET value='2' WHERE key='schema'")
    with _store(other_root) as reopened:
        reopened._db.execute(
            "UPDATE settlement_candidates SET status='pending' WHERE reservation_id=?",
            (candidate.reservation_digest,),
        )
        held = reopened.get(candidate.reservation_digest)
        with pytest.raises(IntakeError, match="settlement authority"):
            reopened.archive_schema3_migration_hold(
                candidate.reservation_digest,
                current_finalized_block=10,
                reason="must fail closed on inconsistent authority",
            )
        assert reopened.get(candidate.reservation_digest) == held


def test_schema3_archival_cli_uses_finalized_public_scope_without_a_wallet(
    tmp_path, monkeypatch, capsys
):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        assert isinstance(candidate, SettlementCandidate)
        store._db.execute("UPDATE metadata SET value='2' WHERE key='schema'")

    class Subtensor:
        def get_block_hash(self, block):
            assert block == 0
            return SCOPE.genesis_hash

    monkeypatch.setattr(chain, "connect", lambda network: Subtensor())
    monkeypatch.setattr(
        chain,
        "read_finalized_head",
        lambda _subtensor: (12, "0x" + f"{12:064x}"),
    )
    args = cli.build_parser().parse_args(
        [
            "chain-archive-schema3-hold",
            "--network",
            "mock",
            "--netuid",
            str(SCOPE.netuid),
            "--intake-db",
            str(tmp_path / "private" / "intake.sqlite3"),
            "--reservation-id",
            candidate.reservation_digest,
            "--reason",
            "reviewed before testnet restart",
        ]
    )
    assert args.func is cli.cmd_chain_archive_schema3_hold
    result = args.func(args)
    assert result == 0
    assert "retained evidence remains non-crownable" in capsys.readouterr().out
    with _store(tmp_path) as reopened:
        archived = reopened.get(candidate.reservation_digest)
        assert archived.status == "expired"
        assert archived.reason.startswith("schema3_archived@12:")


def test_qualification_no_decision_is_retained_before_bounded_requeue(tmp_path):
    with _store(tmp_path, max_qualification_retries=1) as store:
        row = store.reserve_finalized(
            (_arrival(0),), finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        store.mark_fetching(row.reservation_id)
        store.mark_published(
            row.reservation_id,
            delta_fingerprint=_fingerprint("target.a", "slot.a"),
            publication_digest="d" * 64,
            publication_root="/published/a",
        )
        _promote(store, row.reservation_id)
        store.mark_qualifying(row.reservation_id, "6" * 64, AUTHORITY)
        store.mark_outcome(
            row.reservation_id,
            decision="NO_DECISION",
            failure_digest="7" * 64,
            reason="shared_reference_failure",
        )
        assert store.qualification_dispositions(row.reservation_id) == ({
            "attempt_index": 0,
            "authority_digest": "6" * 64,
            "authority_manifest": AUTHORITY,
            "evidence_digest": "7" * 64,
            "attempt_ref": None,
            "report_digest": "",
            "failure_digest": "7" * 64,
            "decision": "NO_DECISION",
            "reason": "shared_reference_failure",
        },)
        held = store.requeue_qualification(
            row.reservation_id,
            reason="retry budget checked",
            retry_group_digest="8" * 64,
            retry_position=0,
        )
        assert held.status == "held"
        assert store.qualification_dispositions(row.reservation_id)[0]["decision"] == "NO_DECISION"


def test_retry_groups_are_selected_separately_in_finalized_order(tmp_path):
    with _store(tmp_path, max_cohort=2) as store:
        first, second = store.reserve_finalized(
            (_arrival(0), _arrival(1, hotkey="other")),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )
        for row, marker in ((first, "a"), (second, "b")):
            store.mark_fetching(row.reservation_id)
            store.mark_published(
                row.reservation_id,
                delta_fingerprint=_fingerprint(
                    f"target.{marker}", f"slot.{marker}", marker
                ),
                publication_digest=marker * 64,
                publication_root=f"/published/{marker}",
            )
            _promote(store, row.reservation_id)
            store.mark_qualifying(row.reservation_id, "7" * 64, AUTHORITY)
            store.mark_outcome(
                row.reservation_id,
                decision="NO_DECISION",
                failure_digest="6" * 64,
                reason="shared_failure",
            )
            store.requeue_qualification(
                row.reservation_id,
                reason="qualification_bisect",
                retry_group_digest=marker * 64,
                retry_position=0,
            )

        assert store.published() == (store.get(first.reservation_id),)
        _promote(store, first.reservation_id)
        store.mark_qualifying(first.reservation_id, "5" * 64, AUTHORITY)
        store.mark_outcome(
            first.reservation_id,
            decision="FAIL",
            attempt_ref=ATTEMPT,
            report_digest="4" * 64,
            reason="qualified",
        )
        assert store.published() == (store.get(second.reservation_id),)


def test_qualification_batch_persists_dispositions_and_groups_atomically(tmp_path):
    with _store(tmp_path, max_cohort=2) as store:
        rows = store.reserve_finalized(
            (_arrival(0), _arrival(1, hotkey="other")),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )
        for row, marker in zip(rows, ("a", "b"), strict=True):
            store.mark_fetching(row.reservation_id)
            store.mark_published(
                row.reservation_id,
                delta_fingerprint=_fingerprint(
                    f"target.{marker}", f"slot.{marker}", marker
                ),
                publication_digest=marker * 64,
                publication_root=f"/published/{marker}",
            )
            _promote(store, row.reservation_id)
            store.mark_qualifying(row.reservation_id, "7" * 64, AUTHORITY)
        failure = "6" * 64
        outcomes = tuple(
            QualificationIntakeOutcome(
                row.reservation_id,
                "3" * 64,
                "7" * 64,
                QualificationDecision.NO_DECISION,
                "shared_failure",
                True,
                failure_digest=failure,
            )
            for row in rows
        )
        retry = QualificationRetryPlan(
            "7" * 64,
            "bisect",
            tuple((row.reservation_id,) for row in rows),
            failure,
        )
        stored = store.apply_qualification_batch(
            QualificationIntakeBatch("7" * 64, outcomes, retry_plan=retry),
            current_finalized_block=10,
        )
        assert [row.status for row in stored] == ["published", "published"]
        assert store.published() == (store.get(rows[0].reservation_id),)
        assert store.qualification_dispositions(rows[0].reservation_id)[0][
            "authority_manifest"
        ] == AUTHORITY


def test_worker_failure_retry_holds_offender_without_stranding_peer(tmp_path):
    with _store(
        tmp_path, max_cohort=2, max_qualification_retries=2
    ) as store:
        offender, peer = store.reserve_finalized(
            (_arrival(0), _arrival(1, hotkey="peer")),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )
        for row, marker in zip((offender, peer), ("a", "b"), strict=True):
            store.mark_fetching(row.reservation_id)
            store.mark_published(
                row.reservation_id,
                delta_fingerprint=_fingerprint(
                    f"target.{marker}", f"slot.{marker}", marker
                ),
                publication_digest=marker * 64,
                publication_root=f"/published/{marker}",
            )
            _promote(store, row.reservation_id)
            store.mark_qualifying(row.reservation_id, "7" * 64, AUTHORITY)

        first_failure = "6" * 64
        first_outcomes = tuple(
            QualificationIntakeOutcome(
                row.reservation_id,
                "3" * 64,
                "7" * 64,
                QualificationDecision.NO_DECISION,
                "candidate_worker",
                True,
                failure_digest=first_failure,
            )
            for row in (offender, peer)
        )
        first_retry = QualificationRetryPlan(
            "7" * 64,
            "bisect",
            ((offender.reservation_id,), (peer.reservation_id,)),
            first_failure,
        )
        store.apply_qualification_batch(
            QualificationIntakeBatch(
                "7" * 64, first_outcomes, retry_plan=first_retry
            ),
            current_finalized_block=10,
        )

        # Finalized order selects the offender's isolated retry without pulling
        # the unrelated retry group back into the same failing cohort.
        assert tuple(row.reservation_id for row in store.published()) == (
            offender.reservation_id,
        )
        _promote(store, offender.reservation_id)
        store.mark_qualifying(offender.reservation_id, "8" * 64, AUTHORITY)
        singleton_failure = "9" * 64
        store.apply_qualification_batch(
            QualificationIntakeBatch(
                "8" * 64,
                (
                    QualificationIntakeOutcome(
                        offender.reservation_id,
                        "3" * 64,
                        "8" * 64,
                        QualificationDecision.NO_DECISION,
                        "candidate_worker",
                        True,
                        failure_digest=singleton_failure,
                    ),
                ),
                retry_plan=QualificationRetryPlan(
                    "8" * 64,
                    "requeue",
                    ((offender.reservation_id,),),
                    singleton_failure,
                ),
            ),
            current_finalized_block=10,
        )

        held = store.get(offender.reservation_id)
        assert held.status == "held"
        assert held.decision == "NO_DECISION"
        assert len(store.qualification_dispositions(offender.reservation_id)) == 2

        # Once the bounded offender is held, the peer's isolated group remains
        # runnable and can retain an independently evidenced terminal decision.
        assert tuple(row.reservation_id for row in store.published()) == (
            peer.reservation_id,
        )
        _promote(store, peer.reservation_id)
        store.mark_qualifying(peer.reservation_id, "a" * 64, AUTHORITY)
        store.apply_qualification_batch(
            QualificationIntakeBatch(
                "a" * 64,
                (
                    QualificationIntakeOutcome(
                        peer.reservation_id,
                        "3" * 64,
                        "a" * 64,
                        QualificationDecision.FAIL,
                        "peer_completed",
                        False,
                        attempt_artifact_sha256=ATTEMPT.sha256,
                        report_digest="4" * 64,
                    ),
                ),
                ATTEMPT,
            ),
            current_finalized_block=10,
        )
        completed = store.get(peer.reservation_id)
        assert completed.status == "failed"
        assert completed.decision == "FAIL"
        assert completed.reason == "peer_completed"


def test_late_earlier_fingerprint_retroactively_identifies_a_qualified_copy(tmp_path):
    with _store(tmp_path) as store:
        first, later = store.reserve_finalized(
            (_arrival(0, hotkey="author"), _arrival(1, hotkey="copycat")),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )
        store.mark_fetching(later.reservation_id)
        store.mark_published(
            later.reservation_id,
            delta_fingerprint=_fingerprint("target.a", "slot.a"),
            publication_digest="b" * 64,
            publication_root="/published/later",
        )
        _promote(store, later.reservation_id)
        store.mark_qualifying(later.reservation_id, "5" * 64, AUTHORITY)
        store.mark_outcome(
            later.reservation_id,
            decision="NO_DECISION",
            failure_digest="4" * 64,
            reason="not_decided",
        )

        store.mark_fetching(first.reservation_id)
        store.mark_published(
            first.reservation_id,
            delta_fingerprint=_fingerprint("target.a", "slot.a"),
            publication_digest="a" * 64,
            publication_root="/published/first",
        )
        assert store.reconcile_copies() == (
            (later.reservation_id, first.reservation_id),
        )
        copied = store.get(later.reservation_id)
        assert copied.status == "failed" and copied.decision == "FAIL"


def test_pass_projection_settles_atomically_and_recovers_stack_and_claim(tmp_path):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        genesis = store.evaluation_stack(candidate.arena_digest)
        assert genesis.generation == 0
        assert genesis.manifest.digest == candidate.incumbent_stack_digest
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (candidate,)
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        current = store.commit_settlement(
            lease, plan, evidence, current_block=11
        )
        assert current.generation == 1
        assert current.manifest.digest == candidate.candidate_stack_digest
        standing, discovery = store.active_reward_claims()
        assert discovery == ()
        assert len(standing) == 1
        assert standing[0].arena_digest == candidate.arena_digest
        assert standing[0].retained_evidence_digest == evidence[0].digest
        crown = store.reopen_active_crown(candidate.arena_digest, candidate.target_id)
        assert crown.candidate == candidate
        assert crown.evidence == evidence[0]
        assert crown.event.event_type is SettlementEventType.CROWN
        assert store.lease_settlement_cohort(current_block=12) is None

    with _store(tmp_path) as reopened:
        current = reopened.evaluation_stack(candidate.arena_digest)
        assert current.generation == 1
        assert current.manifest.digest == candidate.candidate_stack_digest
        assert reopened.active_reward_claims()[0][0].hotkey == candidate.hotkey
        crown = reopened.reopen_active_crown(
            candidate.arena_digest, candidate.target_id
        )
        assert crown.candidate == candidate
        reopened._db.execute(
            "UPDATE settlement_events SET event_json='{}' WHERE event_id=?",
            (crown.event.digest,),
        )
        with pytest.raises(IntakeError, match="event is corrupt"):
            reopened.reopen_active_crown(candidate.arena_digest, candidate.target_id)


def test_interrupted_settlement_lease_requeues_retained_evidence_without_gpu(tmp_path):
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        first = store.lease_settlement_cohort(current_block=11, lease_blocks=10)
        assert first is not None
    with _store(tmp_path) as reopened:
        second = reopened.lease_settlement_cohort(current_block=12, lease_blocks=10)
        assert second is not None
        assert second.candidates == (candidate,)
        assert second.generation > first.generation
        assert second.lease_id != first.lease_id


@pytest.mark.parametrize("hotkey", ("miner", "other-miner"))
def test_discovery_proposal_replay_is_terminal_before_screening(tmp_path, hotkey):
    proposal = _h("one discovery proposal")
    with _store(tmp_path) as store:
        first, replay, distinct = store.reserve_finalized(
            (
                _arrival(0),
                _arrival(1, hotkey=hotkey),
                _arrival(2, hotkey=hotkey),
            ),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )
        for row, fingerprint in (
            (first, _discovery_fingerprint(proposal)),
            (replay, _discovery_fingerprint(proposal)),
            (distinct, _discovery_fingerprint(_h("distinct proposal"))),
        ):
            store.mark_fetching(row.reservation_id)
            store.mark_published(
                row.reservation_id,
                delta_fingerprint=fingerprint,
                publication_digest=_h(f"publication:{row.reservation_id}"),
                publication_root=f"/published/{row.reservation_id}",
            )

        rejected = store.get(replay.reservation_id)
        assert rejected.status == "failed"
        assert rejected.decision == "FAIL"
        assert rejected.reason == "duplicate_proposal"
        assert replay.reservation_id not in {
            row.reservation_id for row in store.screenable()
        }
        assert store.get(distinct.reservation_id).status == "published"
        assert store.reconcile_copies() == ()
        if hotkey == first.arrival.hotkey:
            # Same-hotkey exemption remains a plagiarism-policy rule only; it
            # no longer permits repeated validator work or repeated bounty.
            assert store.copy_predecessors(replay.reservation_id) == ()


def test_legacy_awarded_discovery_replay_stays_terminal_across_restart(tmp_path):
    proposal = _h("legacy duplicate proposal")
    with _store(tmp_path) as store:
        first = _qualified_discovery_candidate(
            store, index=0, proposal_digest=proposal
        )
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (first,)
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        store.commit_settlement(lease, plan, evidence, current_block=11)
        assert len(store.active_reward_claims()[1]) == 1

        # Simulate a pending row retained by a pre-fix database. The current
        # publication path would stop it before screening; lease recovery must
        # also dispose it so restart cannot reproduce the old crash loop.
        replay = _qualified_discovery_candidate(
            store,
            index=1,
            proposal_digest=proposal,
            bypass_publication_dedup=True,
        )
        assert store.copy_predecessors(replay.reservation_digest) == ()
        assert store.lease_settlement_cohort(current_block=12) is None
        duplicate = store._db.execute(
            "SELECT status,reason FROM settlement_candidates WHERE reservation_id=?",
            (replay.reservation_digest,),
        ).fetchone()
        assert tuple(duplicate) == ("duplicate_proposal", "already_awarded")

    with _store(tmp_path) as reopened:
        assert not reopened.has_pending_settlement()
        duplicate = reopened._db.execute(
            "SELECT status,reason FROM settlement_candidates WHERE reservation_id=?",
            (replay.reservation_digest,),
        ).fetchone()
        assert tuple(duplicate) == ("duplicate_proposal", "already_awarded")


def test_legacy_pending_discovery_replays_are_deduplicated_before_lease(tmp_path):
    proposal = _h("pending duplicate proposal")
    with _store(tmp_path) as store:
        first = _qualified_discovery_candidate(
            store, index=0, proposal_digest=proposal
        )
        replay = _qualified_discovery_candidate(
            store,
            index=1,
            proposal_digest=proposal,
            bypass_publication_dedup=True,
        )
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (first,)
        duplicate = store._db.execute(
            "SELECT status,reason FROM settlement_candidates WHERE reservation_id=?",
            (replay.reservation_digest,),
        ).fetchone()
        assert tuple(duplicate) == ("duplicate_proposal", "duplicate_proposal")


def test_discovery_award_race_is_an_idempotent_no_bounty_commit(tmp_path):
    proposal = _h("raced discovery proposal")
    with _store(tmp_path) as store:
        candidate = _qualified_discovery_candidate(
            store, index=0, proposal_digest=proposal
        )
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None and lease.candidates == (candidate,)
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )

        competing = DiscoveryBountyClaim(
            proposal,
            _h("competing retained evidence"),
            "other-miner",
            1,
            candidate.finalized_block,
        )
        store._db.execute(
            "INSERT INTO discovery_bounty_claims(claim_digest,proposal_digest,"
            "claim_json,status,event_id) VALUES(?,?,?,?,?)",
            (
                competing.digest,
                competing.proposal_digest,
                json.dumps(
                    competing.to_dict(), separators=(",", ":"), sort_keys=True
                ),
                "active",
                _h("competing event"),
            ),
        )

        state = store.commit_settlement(lease, plan, evidence, current_block=11)
        assert state == lease.stack
        disposition = store._db.execute(
            "SELECT status,reason FROM settlement_candidates WHERE reservation_id=?",
            (candidate.reservation_digest,),
        ).fetchone()
        assert tuple(disposition) == ("duplicate_proposal", "already_awarded")
        assert store._event_head() == (
            lease.initial_event_sequence,
            lease.previous_event_digest,
        )
        assert store.active_reward_claims()[1] == (competing,)
        assert not store.has_pending_settlement()


def test_weight_projection_reopens_every_active_crown_and_holds_on_loss(tmp_path):
    catalog = default_target_catalog()
    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        store.commit_settlement(lease, plan, evidence, current_block=11)
        context = GlobalRewardProjectionContext(
            SCOPE.digest,
            "validator",
            12,
            "0x" + f"{12:064x}",
            (MetagraphMember(0, "validator"), MetagraphMember(1, "miner")),
        )
        policy = EmissionsPolicyManifest(100, 20, 100_000)
        with pytest.raises(IntakeError, match="catalogs"):
            store.build_weight_projection(
                policy=policy,
                context=context,
                catalogs={},
                netuid=SCOPE.netuid,
            )
        assert store._db.execute(
            "SELECT value FROM metadata WHERE key='emissions_policy_digest'"
        ).fetchone() is None
        projection = store.build_weight_projection(
            policy=policy,
            context=context,
            catalogs={candidate.arena_digest: catalog},
            netuid=SCOPE.netuid,
        )
        assert projection.crown_count == 1
        assert projection.weights_ppm == (("miner", 1_000_000),)
        pending = WeightPublicationRecord(
            projection.digest,
            "pending",
            submit_block=projection.effective_block,
            retry_after_block=projection.effective_block + 20,
            reason="sdk_result_unconfirmed",
        )
        journal = SQLiteWeightPublicationJournal(store, projection)
        journal.compare_and_swap(None, pending)
        standing = store.active_reward_claims()[0][0]
        orphan = StandingRewardClaim(
            _h("orphan-arena"),
            standing.target_id,
            standing.target_spec_digest,
            standing.contribution_digest,
            standing.hotkey,
            standing.speedup_ppm,
            standing.crowned_block,
            standing.retained_evidence_digest,
        )
        store._db.execute(
            "INSERT INTO standing_reward_claims(arena_id,target_id,claim_digest,"
            "claim_json,status,event_id) VALUES(?,?,?,?, 'active',?)",
            (
                orphan.arena_digest,
                orphan.target_id,
                orphan.digest,
                json.dumps(orphan.to_dict(), separators=(",", ":"), sort_keys=True),
                _h("orphan-event"),
            ),
        )
        with pytest.raises(IntakeError, match="absent evaluation arena"):
            store.build_weight_projection(
                policy=policy,
                context=context,
                catalogs={candidate.arena_digest: catalog},
                netuid=SCOPE.netuid,
            )
        store._db.execute(
            "DELETE FROM standing_reward_claims WHERE arena_id=?",
            (orphan.arena_digest,),
        )
        with pytest.raises(IntakeError, match="emissions policy"):
            store.build_weight_projection(
                policy=EmissionsPolicyManifest(101, 20, 100_000),
                context=context,
                catalogs={candidate.arena_digest: catalog},
                netuid=SCOPE.netuid,
            )

        artifact = (
            store.path.parent
            / "evidence"
            / evidence[0].primary_attempt_ref.domain
            / evidence[0].primary_attempt_ref.sha256[:2]
            / evidence[0].primary_attempt_ref.sha256
        )
        artifact.unlink()
        with pytest.raises(IntakeError, match="cannot reopen"):
            store.build_weight_projection(
                policy=policy,
                context=context,
                catalogs={candidate.arena_digest: catalog},
                netuid=SCOPE.netuid,
            )
        retained = SQLiteWeightPublicationJournal.reopen_from_head(store)
        assert retained.projection == projection
        assert retained.load() == pending


def test_uncrowned_arena_is_staging_and_cannot_halt_a_crowned_arena(tmp_path):
    catalog = default_target_catalog()
    policy = EmissionsPolicyManifest(100, 20, 100_000)
    context = GlobalRewardProjectionContext(
        SCOPE.digest,
        "validator",
        12,
        "0x" + f"{12:064x}",
        (MetagraphMember(0, "validator"), MetagraphMember(1, "miner")),
    )
    staging = EvaluationStackManifest(
        runtime_digest=_h("staging-runtime"),
        base_engine_digest=_h("staging-base"),
        arena_digest=_h("staging-arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )

    with _store(tmp_path) as store:
        candidate = _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11)
        assert lease is not None
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        store.commit_settlement(lease, plan, evidence, current_block=11)
        store.initialize_evaluation_stack(staging, tree_digest=_h("staging-tree"))

        projection = store.build_weight_projection(
            policy=policy,
            context=context,
            catalogs={candidate.arena_digest: catalog, staging.arena_digest: catalog},
            netuid=SCOPE.netuid,
        )
        assert projection.weights_ppm == (("miner", 1_000_000),)
        assert projection.crown_count == 1
        assert len(projection.arena_state_digests) == 1
        assert store.evaluation_stack(staging.arena_digest).generation == 0

    with _store(tmp_path) as reopened:
        # A restart must not reactivate a persisted generation-zero arena.  Its
        # catalog is optional because it has no economic authority yet.
        projection = reopened.build_weight_projection(
            policy=policy,
            context=context,
            catalogs={candidate.arena_digest: catalog},
            netuid=SCOPE.netuid,
        )
        assert projection.weights_ppm == (("miner", 1_000_000),)
        assert len(projection.arena_state_digests) == 1


def test_all_uncrowned_bootstrap_remains_an_explicit_fail_closed_policy(tmp_path):
    catalog = default_target_catalog()
    staging = EvaluationStackManifest(
        runtime_digest=_h("bootstrap-runtime"),
        base_engine_digest=_h("bootstrap-base"),
        arena_digest=_h("bootstrap-arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )
    context = GlobalRewardProjectionContext(
        SCOPE.digest,
        "validator",
        12,
        "0x" + f"{12:064x}",
        (MetagraphMember(0, "validator"),),
    )
    with _store(tmp_path) as store:
        store.initialize_evaluation_stack(staging, tree_digest=_h("bootstrap-tree"))
        with pytest.raises(EconomicsError, match="typed arena authorities"):
            store.build_weight_projection(
                policy=EmissionsPolicyManifest(100, 20, 100_000),
                context=context,
                catalogs={staging.arena_digest: catalog},
                netuid=SCOPE.netuid,
            )
        assert store._db.execute(
            "SELECT value FROM metadata WHERE key='emissions_policy_digest'"
        ).fetchone() is None


def test_expired_settlement_lease_cannot_commit(tmp_path):
    with _store(tmp_path) as store:
        _qualified_settlement_candidate(store)
        lease = store.lease_settlement_cohort(current_block=11, lease_blocks=2)
        assert lease is not None
        plan = plan_settlement(
            lease.candidates,
            current_manifest=lease.stack.manifest,
            current_tree_digest=lease.stack.tree_digest,
            initial_event_sequence=lease.initial_event_sequence,
            previous_event_digest=lease.previous_event_digest,
        )
        evidence = tuple(
            store.reopen_settlement_evidence(row) for row in lease.candidates
        )
        with pytest.raises(IntakeError, match="deadline"):
            store.commit_settlement(lease, plan, evidence, current_block=13)
        assert store.evaluation_stack(lease.stack.arena_digest) == lease.stack


def test_pass_without_exact_settlement_projection_is_rejected_atomically(tmp_path):
    with _store(tmp_path) as store:
        row = store.reserve_finalized(
            (_arrival(0),), finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
        )[0]
        store.mark_fetching(row.reservation_id)
        store.mark_published(
            row.reservation_id,
            delta_fingerprint=_fingerprint("target.a", "slot.a"),
            publication_digest="d" * 64,
            publication_root="/published/a",
        )
        _promote(store, row.reservation_id)
        store.mark_qualifying(row.reservation_id, "7" * 64, AUTHORITY)
        outcome = QualificationIntakeOutcome(
            row.reservation_id,
            "3" * 64,
            "7" * 64,
            QualificationDecision.PASS,
            "qualified",
            False,
            attempt_artifact_sha256=ATTEMPT.sha256,
            report_digest="4" * 64,
        )
        with pytest.raises(IntakeError, match="settlement projection"):
            store.apply_qualification_batch(
                QualificationIntakeBatch("7" * 64, (outcome,), ATTEMPT),
                current_finalized_block=10,
            )
        assert store.get(row.reservation_id).status == "qualifying"
        assert store.qualification_dispositions(row.reservation_id) == ()


def test_sqlite_weight_journal_is_cas_bound_and_restart_reopenable(tmp_path):
    projection = WeightProjection(
        _h("scope"),
        307,
        "validator",
        _h("policy"),
        _h("settlement"),
        _h("evaluation"),
        _h("metagraph"),
        (_h("arena-state"),),
        1,
        10,
        1,
        (_h("evidence"),),
        (("miner", 1_000_000),),
    )
    intent = WeightPublicationRecord(
        projection.digest,
        "intent",
        submit_block=10,
        retry_after_block=20,
        reason="before_sdk_submission",
    )
    with _store(tmp_path) as store:
        journal = SQLiteWeightPublicationJournal(store, projection)
        assert journal.load() is None
        journal.compare_and_swap(None, intent)
        assert journal.load() == intent
        with pytest.raises(IntakeError, match="compare-and-swap"):
            journal.compare_and_swap(None, intent)

    with _store(tmp_path) as reopened:
        journal = SQLiteWeightPublicationJournal.reopen_from_head(reopened)
        assert journal.projection == projection
        assert journal.load() == intent
        assert journal.retained_projection(projection.digest) == projection
        pending = WeightPublicationRecord(
            projection.digest,
            "pending",
            prior_record_digest=intent.digest,
            submit_block=10,
            retry_after_block=20,
            reason="sdk_result_unconfirmed",
        )
        journal.compare_and_swap(intent.digest, pending)
        assert journal.load() == pending
        assert reopened._db.execute(
            "SELECT COUNT(*) AS n FROM weight_publications"
        ).fetchone()["n"] == 2


def test_sqlite_weight_journal_reopen_rejects_corrupt_head_projection(tmp_path):
    projection = WeightProjection(
        _h("scope"),
        307,
        "validator",
        _h("policy"),
        _h("settlement"),
        _h("evaluation"),
        _h("metagraph"),
        (_h("arena-state"),),
        1,
        10,
        1,
        (_h("evidence"),),
        (("miner", 1_000_000),),
    )
    pending = WeightPublicationRecord(
        projection.digest,
        "pending",
        submit_block=10,
        retry_after_block=20,
        reason="sdk_result_unconfirmed",
    )
    with _store(tmp_path) as store:
        journal = SQLiteWeightPublicationJournal(store, projection)
        journal.compare_and_swap(None, pending)
        store._db.execute(
            "UPDATE weight_publications SET projection_json='{}' "
            "WHERE record_digest=?",
            (pending.digest,),
        )
        with pytest.raises(IntakeError, match="projection is corrupt"):
            SQLiteWeightPublicationJournal.reopen_from_head(store)
