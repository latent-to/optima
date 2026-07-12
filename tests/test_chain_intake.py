from __future__ import annotations

import os

import pytest

from optima.chain.intake import (
    FinalizedArrival, FinalizedIntakeStore, IntakeError, IntakePolicy,
    IntakeScope,
)
from optima.copy_fingerprint import SubmittedDeltaFingerprint
from optima.eval.evidence_store import EvidenceArtifactRef


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


def _fingerprint(target: str, member: str, marker: str = "a"):
    return SubmittedDeltaFingerprint(
        "component", target, "1" * 64, (member,), "2" * 64, "3" * 64,
        "4" * 64, (marker * 64,), ("5" * 64,),
    )


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
        assert store.copy_successors(first.reservation_id) == (
            store.get(later.reservation_id),
        )
        copied = store.mark_copy(later.reservation_id, first.reservation_id)
        assert copied.status == "failed" and copied.decision == "FAIL"
