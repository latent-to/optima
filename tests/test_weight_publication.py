from __future__ import annotations

import types

import pytest

from optima.chain.weights import (
    WeightProjection,
    WeightPublicationError,
    WeightPublicationRecord,
    release_weight_publication_hold,
    reconcile_weight_publication,
    resume_weight_projection,
)
from optima.stack_identity import canonical_digest


def _d(char: str) -> str:
    return char * 64


class Journal:
    def __init__(self, row=None, retained=()):
        self.row = row
        self.history = []
        self.retained = {projection.digest: projection for projection in retained}

    def load(self):
        return self.row

    def compare_and_swap(self, expected_record_digest, replacement):
        assert expected_record_digest == (self.row.digest if self.row else None)
        self.row = replacement
        self.history.append(replacement)

    def retained_projection(self, projection_digest):
        return self.retained[projection_digest]


class Chain:
    def __init__(self, *, apply=False, block=100):
        self.block = block
        self.hotkeys = ["validator", "alice", "bob"]
        self.last_update = [0, 0, 0]
        self.row = []
        self.apply = apply
        self.submit_calls = 0
        self.weight_reads = 0

    def metagraph(self, netuid=None):
        return types.SimpleNamespace(
            uids=[0, 1, 2], hotkeys=self.hotkeys,
            last_update=self.last_update, validator_permit=[True, True, True],
        )

    def weights(self, netuid=None):
        self.weight_reads += 1
        return [(0, list(self.row))] if self.row else []

    def get_current_block(self):
        return self.block

    def get_block_hash(self, block):
        return "0x" + f"{block:064x}"

    def set_weights(self, **kwargs):
        self.submit_calls += 1
        if self.apply:
            self.row = [
                (uid, round(weight * 65_535))
                for uid, weight in zip(kwargs["uids"], kwargs["weights"], strict=True)
            ]
            self.last_update[0] = self.block
        return True

    def install(self, weights, *, update):
        index = {hotkey: uid for uid, hotkey in enumerate(self.hotkeys)}
        self.row = [
            (index[hotkey], round(weight * 65_535))
            for hotkey, weight in weights.items()
        ]
        self.last_update[0] = update


def _projection(
    *, crowns=1, weights=(("alice", 1_000_000),), marker="a", block=100
):
    metagraph_digest = canonical_digest(
        "optima.economics.metagraph-membership",
        {
            "block": block,
            "block_hash": "0x" + f"{block:064x}",
            "chain_scope_digest": _d("1"),
            "members": [
                {"hotkey": hotkey, "uid": uid}
                for uid, hotkey in enumerate(("validator", "alice", "bob"))
            ],
        },
    )
    return WeightProjection(
        _d("1"), 1, "validator", _d("2"), _d(marker), _d("4"),
        metagraph_digest, (_d("6"),), 3, block, crowns,
        ((_d("5"),) if crowns else ()), tuple(weights),
    )


def _wallet(hotkey="validator"):
    return types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address=hotkey))


def test_dry_run_refreshes_without_creating_journal_intent():
    chain, journal = Chain(), Journal()
    result = reconcile_weight_publication(
        chain, None, _projection(crowns=0), journal, refresh_blocks=20, dry_run=True
    )
    assert result.status == "dry_run" and result.record is None
    assert chain.weight_reads == 1 and chain.submit_calls == 0 and journal.history == []


def test_real_submit_journals_intent_pending_then_authoritative_confirmation():
    chain, journal = Chain(apply=True), Journal()
    result = reconcile_weight_publication(
        chain, _wallet(), _projection(), journal, refresh_blocks=20
    )
    assert result.status == "confirmed" and result.chain_matches
    assert [row.status for row in journal.history] == ["intent", "pending", "confirmed"]
    assert chain.weight_reads == 2 and chain.submit_calls == 1
    assert journal.row.confirmed_last_update == 100


def test_pending_is_not_resubmitted_and_confirms_only_after_exact_readback():
    chain, journal = Chain(), Journal()
    projection = _projection()
    first = reconcile_weight_publication(
        chain, _wallet(), projection, journal, refresh_blocks=20
    )
    assert first.status == "pending" and chain.submit_calls == 1
    chain.install(projection.weights, update=100)
    second = reconcile_weight_publication(
        chain, _wallet(), projection, journal, refresh_blocks=20
    )
    assert second.status == "confirmed" and chain.submit_calls == 1


def test_restart_reopens_pending_projection_and_confirms_after_chain_head_advances():
    chain = Chain()
    original = _projection()
    journal = Journal(retained=(original,))
    first = reconcile_weight_publication(
        chain, _wallet(), original, journal, refresh_blocks=20
    )
    assert first.status == "pending" and chain.submit_calls == 1

    chain.block = 101
    chain.install(original.weights, update=100)
    rebuilt = _projection(block=101)
    resumed = resume_weight_projection(rebuilt, journal)
    second = reconcile_weight_publication(
        chain, _wallet(), resumed, journal, refresh_blocks=20
    )

    assert resumed == original
    assert second.status == "confirmed" and second.projection_digest == original.digest
    assert chain.submit_calls == 1
    assert [row.status for row in journal.history] == [
        "intent", "pending", "confirmed"
    ]
    assert {row.projection_digest for row in journal.history} == {original.digest}


def test_pending_resume_rejects_a_different_chain_authority():
    original = _projection()
    pending = WeightPublicationRecord(
        original.digest,
        "pending",
        submit_block=100,
        retry_after_block=120,
        reason="sdk_result_unconfirmed",
    )
    journal = Journal(pending, retained=(original,))
    proposed = WeightProjection.from_dict(
        {**_projection(block=101).to_dict(), "validator_hotkey": "other"}
    )

    with pytest.raises(WeightPublicationError, match="current chain authority"):
        resume_weight_projection(proposed, journal)


def test_unresolved_or_changed_pending_projection_holds_without_signing():
    chain, journal = Chain(), Journal()
    projection = _projection()
    reconcile_weight_publication(chain, _wallet(), projection, journal, refresh_blocks=20)
    changed = _projection(marker="b")
    result = reconcile_weight_publication(
        chain, _wallet(), changed, journal, refresh_blocks=20
    )
    assert result.status == "held" and chain.submit_calls == 1
    assert journal.row.projection_digest == projection.digest

    chain2, journal2 = Chain(), Journal()
    reconcile_weight_publication(chain2, _wallet(), projection, journal2, refresh_blocks=20)
    chain2.block = 120
    expired = reconcile_weight_publication(
        chain2, _wallet(), _projection(block=120), journal2, refresh_blocks=20
    )
    assert expired.status == "held" and chain2.submit_calls == 1


def test_real_submission_requires_crown_and_exact_signer_before_intent():
    for projection, wallet, message in (
        (_projection(crowns=0), _wallet(), "current crown"),
        (_projection(), _wallet("other"), "signer wallet"),
    ):
        chain, journal = Chain(), Journal()
        with pytest.raises(WeightPublicationError, match=message):
            reconcile_weight_publication(
                chain, wallet, projection, journal, refresh_blocks=20
            )
        assert journal.history == [] and chain.submit_calls == 0


def test_confirmed_vector_mismatch_holds_and_refresh_due_resubmits():
    projection = _projection()
    confirmed = WeightPublicationRecord(
        projection.digest, "confirmed", confirmed_block=90,
        confirmed_last_update=90, reason="readback",
    )
    chain, journal = Chain(), Journal(confirmed)
    mismatch = reconcile_weight_publication(
        chain, _wallet(), projection, journal, refresh_blocks=20
    )
    assert mismatch.status == "held" and chain.submit_calls == 0

    chain2, journal2 = Chain(apply=True), Journal(confirmed)
    chain2.install(projection.weights, update=70)
    refreshed = reconcile_weight_publication(
        chain2, _wallet(), projection, journal2, refresh_blocks=20
    )
    assert refreshed.status == "confirmed" and chain2.submit_calls == 1
    assert [row.status for row in journal2.history] == ["intent", "pending", "confirmed"]


def test_stale_projection_is_rejected_before_journal_or_signing():
    chain, journal = Chain(block=101), Journal()
    with pytest.raises(WeightPublicationError, match="stale"):
        reconcile_weight_publication(
            chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
        )
    assert journal.history == [] and chain.submit_calls == 0


def test_held_publication_requires_explicit_append_only_release():
    projection = _projection()
    held = WeightPublicationRecord(
        projection.digest,
        "held",
        reason="operator_review_required",
    )
    journal = Journal(held)
    released = release_weight_publication_hold(
        journal, reason="review_ticket_123"
    )
    assert released.status == "released"
    assert released.prior_record_digest == held.digest
    chain = Chain(apply=True)
    result = reconcile_weight_publication(
        chain, _wallet(), projection, journal, refresh_blocks=20
    )
    assert result.status == "confirmed"
    assert [row.status for row in journal.history] == [
        "released", "intent", "pending", "confirmed"
    ]
