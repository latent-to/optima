from __future__ import annotations

import os
import json

import pytest

from optima.chain.intake import (
    FinalizedArrival, FinalizedIntakeStore, IntakeError, IntakePolicy,
    IntakeScope, SQLiteWeightPublicationJournal,
)
from optima.chain.weights import WeightProjection, WeightPublicationRecord
from optima.copy_fingerprint import SubmittedDeltaFingerprint
from optima.eval.evidence_store import EvidenceArtifactRef, publish_evidence
from optima.eval.qualification import QualificationDecision
from optima.eval.qualification_intake import (
    QualificationIntakeBatch,
    QualificationIntakeOutcome,
    QualificationRetryPlan,
)
from optima.economics import (
    EmissionsPolicyManifest,
    GlobalRewardProjectionContext,
    MetagraphMember,
    StandingRewardClaim,
)
from optima.settlement import SettlementCandidate, plan_settlement
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


def _qualified_settlement_candidate(store: FinalizedIntakeStore) -> SettlementCandidate:
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
    attempt = publish_evidence(
        evidence_root,
        b"retained qualification attempt",
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
    store.mark_qualifying(row.reservation_id, "7" * 64, AUTHORITY)
    candidate = SettlementCandidate(
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
        qualification_authority_digest="7" * 64,
        qualification_plan_digest="6" * 64,
        qualification_attempt_digest=attempt.sha256,
        qualification_report_digest="4" * 64,
        arm_digest=arm.digest,
        incumbent_stack_digest=arm.baseline_before.stack_digest,
        incumbent_tree_digest=arm.baseline_before.tree_digest,
        candidate_stack_digest=arm.challenger.stack_digest,
        candidate_tree_digest=arm.challenger.tree_digest,
        speedup="1.05",
        incumbent_manifest=incumbent,
        candidate_manifest=arm.candidate,
    )
    outcome = QualificationIntakeOutcome(
        row.reservation_id,
        arm.selected_delta_digest,
        "7" * 64,
        QualificationDecision.PASS,
        "qualified",
        False,
        attempt_artifact_sha256=attempt.sha256,
        report_digest="4" * 64,
        settlement_candidate=candidate,
    )
    store.apply_qualification_batch(
        QualificationIntakeBatch("7" * 64, (outcome,), attempt),
        evidence_root=evidence_root,
    )
    return candidate


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
        store.mark_qualifying(first.reservation_id, "5" * 64, AUTHORITY)
        store.mark_outcome(
            first.reservation_id,
            decision="PASS",
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
            QualificationIntakeBatch("7" * 64, outcomes, retry_plan=retry)
        )
        assert [row.status for row in stored] == ["published", "published"]
        assert store.published() == (store.get(rows[0].reservation_id),)
        assert store.qualification_dispositions(rows[0].reservation_id)[0][
            "authority_manifest"
        ] == AUTHORITY


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
        store.mark_qualifying(later.reservation_id, "5" * 64, AUTHORITY)
        store.mark_outcome(
            later.reservation_id,
            decision="PASS",
            attempt_ref=ATTEMPT,
            report_digest="4" * 64,
            reason="qualified",
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
        assert store.lease_settlement_cohort(current_block=12) is None

    with _store(tmp_path) as reopened:
        current = reopened.evaluation_stack(candidate.arena_digest)
        assert current.generation == 1
        assert current.manifest.digest == candidate.candidate_stack_digest
        assert reopened.active_reward_claims()[0][0].hotkey == candidate.hotkey


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
            / evidence[0].attempt_ref.domain
            / evidence[0].attempt_ref.sha256[:2]
            / evidence[0].attempt_ref.sha256
        )
        artifact.unlink()
        with pytest.raises(IntakeError, match="cannot reopen"):
            store.build_weight_projection(
                policy=policy,
                context=context,
                catalogs={candidate.arena_digest: catalog},
                netuid=SCOPE.netuid,
            )


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
                QualificationIntakeBatch("7" * 64, (outcome,), ATTEMPT)
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
        journal = SQLiteWeightPublicationJournal(reopened, projection)
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
