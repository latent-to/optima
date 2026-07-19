from __future__ import annotations

import types

import pytest

from optima import chain as chain_module
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
        self.metagraph_reads: list[int | None] = []
        self.weight_read_blocks: list[int | None] = []
        self.submit_options: list[tuple[bool, bool]] = []

    def metagraph(self, netuid=None, block=None):
        self.metagraph_reads.append(block)
        return types.SimpleNamespace(
            uids=[0, 1, 2], hotkeys=self.hotkeys,
            last_update=self.last_update, validator_permit=[True, True, True],
            block=self.block if block is None else block,
        )

    def weights(self, netuid=None, block=None):
        self.weight_reads += 1
        self.weight_read_blocks.append(block)
        return [(0, list(self.row))] if self.row else []

    def get_current_block(self):
        return self.block

    def get_finalized_block_number(self):
        return self.block

    def get_block_hash(self, block):
        return "0x" + f"{block:064x}"

    def set_weights(self, **kwargs):
        self.submit_calls += 1
        self.submit_options.append(
            (kwargs["wait_for_inclusion"], kwargs["wait_for_finalization"])
        )
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


class ReassigningChain(Chain):
    """Historical UID 1 is alice; finalized block 101 reassigns it."""

    def __init__(self, *, finalized_heads, apply=False):
        super().__init__(apply=apply, block=100)
        self._finalized_heads = iter(finalized_heads)
        self._last_finalized_head = 100

    def get_finalized_block_number(self):
        try:
            self._last_finalized_head = next(self._finalized_heads)
        except StopIteration:
            pass
        return self._last_finalized_head

    def metagraph(self, netuid=None, block=None):
        self.metagraph_reads.append(block)
        requested = self._last_finalized_head if block is None else block
        hotkeys = (
            ["validator", "alice", "bob"]
            if requested <= 100
            else ["validator", "mallory", "alice"]
        )
        return types.SimpleNamespace(
            uids=[0, 1, 2],
            hotkeys=hotkeys,
            last_update=self.last_update,
            validator_permit=[True, True, True],
            block=requested,
        )


class AdvancingStableChain(ReassigningChain):
    """Finality advances while every authority-relevant UID stays stable."""

    def metagraph(self, netuid=None, block=None):
        self.metagraph_reads.append(block)
        requested = self._last_finalized_head if block is None else block
        return types.SimpleNamespace(
            uids=[0, 1, 2],
            hotkeys=["validator", "alice", "bob"],
            last_update=self.last_update,
            validator_permit=[True, True, True],
            block=requested,
        )

    def set_weights(self, **kwargs):
        result = super().set_weights(**kwargs)
        if self.apply:
            self.last_update[0] = self._last_finalized_head
        return result


class LateRevealChain(AdvancingStableChain):
    """The exact row becomes finalized only at the pre-sign authority read."""

    def metagraph(self, netuid=None, block=None):
        requested = self._last_finalized_head if block is None else block
        if requested >= 101 and not self.row:
            self.install({"alice": 1.0}, update=101)
        return super().metagraph(netuid=netuid, block=block)


class SameHeightHashChangingChain(Chain):
    """Returns a different canonical hash on the pre-sign same-height fetch."""

    def __init__(self):
        super().__init__(block=100)
        self._hash_reads = 0

    def get_block_hash(self, block):
        self._hash_reads += 1
        suffix = block if self._hash_reads <= 6 else block + 1
        return "0x" + f"{suffix:064x}"


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


def test_stale_preexisting_vector_without_a_journal_is_refreshed_once():
    chain, journal = Chain(apply=True, block=121), Journal()
    projection = _projection(block=121)
    chain.install(projection.weights, update=100)

    result = reconcile_weight_publication(
        chain, _wallet(), projection, journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.submitted is True
    assert chain.submit_calls == 1
    assert [row.status for row in journal.history] == [
        "intent", "pending", "confirmed"
    ]
    assert journal.row.confirmed_last_update == 121


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


def test_explicit_stale_initial_catch_up_requires_stable_recipient_uids():
    projection = _projection(block=100)
    stable = Chain(apply=True, block=101)
    journal = Journal()

    result = reconcile_weight_publication(
        stable,
        _wallet(),
        projection,
        journal,
        refresh_blocks=20,
        allow_stale_initial=True,
    )

    assert result.status == "confirmed"
    assert result.submitted is True
    assert stable.submit_calls == 1

    reassigned = ReassigningChain(finalized_heads=[101, 101])
    reassigned.block = 101
    journal = Journal()
    with pytest.raises(WeightPublicationError, match="UID mapping changed"):
        reconcile_weight_publication(
            reassigned,
            _wallet(),
            projection,
            journal,
            refresh_blocks=20,
            allow_stale_initial=True,
        )
    assert journal.history == [] and reassigned.submit_calls == 0


def test_explicit_v2_mode_can_publish_reserve_only_projection():
    chain, journal = Chain(apply=True), Journal()

    result = reconcile_weight_publication(
        chain,
        _wallet(),
        _projection(crowns=0),
        journal,
        refresh_blocks=20,
        require_current_crown=False,
    )

    assert result.status == "confirmed"
    assert result.submitted is True
    assert chain.submit_calls == 1


def test_finalized_head_advance_before_signing_aborts_on_uid_reassignment():
    chain = ReassigningChain(finalized_heads=[100, 100, 101])
    journal = Journal()

    with pytest.raises(WeightPublicationError, match="UID mapping changed before signing"):
        reconcile_weight_publication(
            chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
        )

    assert journal.history == []
    assert chain.submit_calls == 0


def test_finalized_head_advance_before_signing_uses_stable_current_uid_mapping():
    chain = AdvancingStableChain(
        finalized_heads=[100, 100, 101, 101, 101, 101], apply=True
    )
    journal = Journal()

    result = reconcile_weight_publication(
        chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.submitted is True
    assert chain.submit_calls == 1
    assert journal.history[0].status == "intent"
    assert journal.history[0].submit_block == 101
    assert journal.history[-1].confirmed_last_update == 101


def test_finalized_head_advance_observes_late_reveal_without_resigning():
    projection = _projection(block=100)
    released = WeightPublicationRecord(
        projection.digest,
        "released",
        submit_block=100,
        retry_after_block=120,
        reason="reviewed late reveal",
    )
    chain = LateRevealChain(finalized_heads=[100, 100, 101, 101])
    journal = Journal(released)

    result = reconcile_weight_publication(
        chain, _wallet(), projection, journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.chain_matches is True
    assert result.submitted is False
    assert chain.submit_calls == 0
    assert [row.status for row in journal.history] == ["confirmed"]
    assert journal.row.submit_block == 100
    assert journal.row.confirmed_last_update == 101


def test_finalized_head_regression_before_signing_aborts_without_intent():
    chain = AdvancingStableChain(finalized_heads=[100, 100, 99])
    journal = Journal()

    with pytest.raises(
        chain_module.ChainWeightStateError,
        match="regressed before signing",
    ):
        reconcile_weight_publication(
            chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
        )

    assert journal.history == []
    assert chain.submit_calls == 0


def test_same_height_authority_hash_change_aborts_without_intent():
    chain = SameHeightHashChangingChain()
    journal = Journal()

    with pytest.raises(
        chain_module.ChainWeightStateError,
        match="changed at the projection block",
    ):
        reconcile_weight_publication(
            chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
        )

    assert journal.history == []
    assert chain.submit_calls == 0


def test_same_height_post_submit_authority_change_holds_not_confirms(
    monkeypatch,
):
    projection = _projection(block=100)
    subtensor = Chain(apply=True, block=100)
    journal = Journal()
    original_fetch = chain_module.fetch_metagraph
    calls = 0

    def changing_fetch(target, netuid, *, block=None):
        nonlocal calls
        calls += 1
        view = original_fetch(target, netuid, block=block)
        if calls == 5:
            return chain_module.MetagraphView(
                netuid=view.netuid,
                block=view.block,
                block_hash="0x" + f"{view.block + 1:064x}",
                uids=list(view.uids),
                hotkeys=list(view.hotkeys),
                validator_permit=list(view.validator_permit),
                last_update=list(view.last_update),
            )
        return view

    monkeypatch.setattr(chain_module, "fetch_metagraph", changing_fetch)
    result = reconcile_weight_publication(
        subtensor, _wallet(), projection, journal, refresh_blocks=20
    )

    assert result.status == "held"
    assert result.record is not None
    assert result.record.reason == "post_submit_authority_unavailable"
    assert result.submitted is True
    assert [row.status for row in journal.history] == ["intent", "pending", "held"]


def test_pending_restart_accepts_later_submit_block_than_effective_block():
    projection = _projection(block=100)
    pending = WeightPublicationRecord(
        projection.digest,
        "pending",
        submit_block=101,
        retry_after_block=121,
        reason="sdk_result_unconfirmed",
    )
    chain = Chain(block=102)
    chain.install(projection.weights, update=101)
    journal = Journal(pending, retained=(projection,))

    result = reconcile_weight_publication(
        chain, _wallet(), projection, journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.submitted is False
    assert chain.submit_calls == 0
    assert [row.status for row in journal.history] == ["confirmed"]
    assert journal.row.submit_block == 101
    assert journal.row.confirmed_last_update == 101


def test_pending_readback_holds_if_recipient_uid_was_reassigned():
    projection = _projection(block=100)
    pending = WeightPublicationRecord(
        projection.digest,
        "pending",
        submit_block=100,
        retry_after_block=120,
        reason="sdk_result_unconfirmed",
    )
    journal = Journal(pending, retained=(projection,))
    chain = ReassigningChain(finalized_heads=[101])

    result = reconcile_weight_publication(
        chain, _wallet(), projection, journal, refresh_blocks=20
    )

    assert result.status == "held"
    assert result.record is not None
    assert result.record.reason == "metagraph_uid_mapping_changed"
    assert chain.submit_calls == 0


def test_post_submit_uid_reassignment_cannot_be_confirmed():
    chain = ReassigningChain(
        finalized_heads=[100, 100, 100, 100, 101], apply=True
    )
    journal = Journal()

    result = reconcile_weight_publication(
        chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
    )

    assert result.status == "held"
    assert result.record is not None
    assert result.record.reason == "post_submit_uid_mapping_changed"
    assert result.submitted is True
    assert [row.status for row in journal.history] == ["intent", "pending", "held"]
    assert chain.submit_options == [(True, True)]


def test_best_head_only_sparse_row_never_confirms():
    class BestHeadOnlyChain(Chain):
        def set_weights(self, **kwargs):
            self.submit_calls += 1
            self.submit_options.append(
                (kwargs["wait_for_inclusion"], kwargs["wait_for_finalization"])
            )
            # Simulate a misleading best-head row which the exact finalized
            # weights(block=100) reader below intentionally never exposes.
            self.best_head_row = [(1, 65_535)]
            return True

    chain = BestHeadOnlyChain()
    journal = Journal()
    result = reconcile_weight_publication(
        chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
    )

    assert result.status == "pending"
    assert result.chain_matches is False
    assert result.submitted is True
    assert set(chain.weight_read_blocks) == {100}
    assert chain.submit_options == [(True, True)]


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


def test_released_late_reveal_confirms_without_duplicate_signature():
    original = _projection(block=100)
    held = WeightPublicationRecord(
        original.digest,
        "held",
        submit_block=100,
        retry_after_block=120,
        reason="publication_readback_deadline_expired",
    )
    journal = Journal(held)
    release_weight_publication_hold(journal, reason="late reveal review passed")
    chain = Chain(block=121)
    chain.install(original.weights, update=120)
    current = _projection(block=121)

    result = reconcile_weight_publication(
        chain, _wallet(), current, journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.chain_matches is True
    assert result.submitted is False
    assert chain.submit_calls == 0
    assert [row.status for row in journal.history] == ["released", "confirmed"]
    assert journal.row.projection_digest == current.digest
    assert journal.row.submit_block == 100
    assert journal.row.retry_after_block == 120
    assert journal.row.confirmed_last_update == 120


def test_released_pre_submit_exact_row_is_refreshed_with_one_signature():
    original = _projection(block=100)
    held = WeightPublicationRecord(
        original.digest,
        "held",
        submit_block=100,
        retry_after_block=120,
        reason="publication_readback_deadline_expired",
    )
    journal = Journal(held)
    release_weight_publication_hold(journal, reason="review passed")
    chain = Chain(apply=True, block=121)
    chain.install(original.weights, update=99)
    current = _projection(block=121)

    result = reconcile_weight_publication(
        chain, _wallet(), current, journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.submitted is True
    assert chain.submit_calls == 1
    assert [row.status for row in journal.history] == [
        "released", "intent", "pending", "confirmed"
    ]
    assert journal.row.projection_digest == current.digest
    assert journal.row.confirmed_last_update == 121


def test_released_post_submit_but_refresh_due_row_is_refreshed_once():
    original = _projection(block=100)
    released = WeightPublicationRecord(
        original.digest,
        "released",
        submit_block=100,
        retry_after_block=120,
        reason="late reveal reviewed",
    )
    journal = Journal(released)
    chain = Chain(apply=True, block=121)
    chain.install(original.weights, update=100)
    current = _projection(block=121)

    result = reconcile_weight_publication(
        chain, _wallet(), current, journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.submitted is True
    assert chain.submit_calls == 1
    assert [row.status for row in journal.history] == [
        "intent", "pending", "confirmed"
    ]
    assert journal.row.confirmed_last_update == 121


def test_fresh_projection_with_unchanged_recent_vector_does_not_resubmit():
    original = _projection(block=100)
    confirmed = WeightPublicationRecord(
        original.digest,
        "confirmed",
        confirmed_block=100,
        confirmed_last_update=100,
        reason="authoritative_readback",
    )
    journal = Journal(confirmed)
    chain = Chain(block=101)
    chain.install(original.weights, update=100)
    current = _projection(block=101)

    result = reconcile_weight_publication(
        chain, _wallet(), current, journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.chain_matches is True
    assert result.submitted is False
    assert chain.submit_calls == 0
    assert [row.status for row in journal.history] == ["confirmed"]
    assert journal.row.projection_digest == current.digest


def test_reconcile_only_pending_exact_readback_confirms_without_signer():
    projection = _projection(block=100)
    pending = WeightPublicationRecord(
        projection.digest,
        "pending",
        submit_block=100,
        retry_after_block=120,
        reason="sdk_result_unconfirmed",
    )
    journal = Journal(pending, retained=(projection,))
    chain = Chain(block=121)
    chain.install(projection.weights, update=120)

    result = reconcile_weight_publication(
        chain,
        None,
        projection,
        journal,
        refresh_blocks=20,
        reconcile_only=True,
    )

    assert result.status == "confirmed"
    assert result.chain_matches is True
    assert result.submitted is False
    assert chain.submit_calls == 0
    assert [row.status for row in journal.history] == ["confirmed"]
    assert journal.row.submit_block == 100
    assert journal.row.confirmed_last_update == 120


def test_reconcile_only_refresh_due_confirmed_row_fails_without_mutation():
    projection = _projection(block=100)
    confirmed = WeightPublicationRecord(
        projection.digest,
        "confirmed",
        confirmed_block=100,
        confirmed_last_update=100,
        reason="authoritative_readback",
    )
    journal = Journal(confirmed)
    chain = Chain(block=121)
    chain.install(projection.weights, update=100)
    current = _projection(block=121)

    with pytest.raises(WeightPublicationError, match="refresh-due"):
        reconcile_weight_publication(
            chain,
            None,
            current,
            journal,
            refresh_blocks=20,
            reconcile_only=True,
        )

    assert journal.row == confirmed
    assert journal.history == []
    assert chain.submit_calls == 0


def test_reconcile_only_advances_fresh_confirmed_projection_from_recent_readback():
    original = _projection(block=100)
    confirmed = WeightPublicationRecord(
        original.digest,
        "confirmed",
        confirmed_block=100,
        confirmed_last_update=100,
        reason="authoritative_readback",
    )
    journal = Journal(confirmed)
    chain = Chain(block=101)
    chain.install(original.weights, update=100)
    current = _projection(block=101)

    result = reconcile_weight_publication(
        chain,
        None,
        current,
        journal,
        refresh_blocks=20,
        reconcile_only=True,
    )

    assert result.status == "confirmed"
    assert result.projection_digest == current.digest
    assert result.submitted is False
    assert chain.submit_calls == 0
    assert [row.status for row in journal.history] == ["confirmed"]
    assert journal.row.projection_digest == current.digest
    assert journal.row.prior_record_digest == confirmed.digest


def test_reconcile_only_released_exact_readback_confirms_without_signer():
    original = _projection(block=100)
    released = WeightPublicationRecord(
        original.digest,
        "released",
        submit_block=100,
        retry_after_block=120,
        reason="late reveal reviewed",
    )
    journal = Journal(released)
    chain = Chain(block=121)
    chain.install(original.weights, update=120)
    result = reconcile_weight_publication(
        chain,
        None,
        original,
        journal,
        refresh_blocks=20,
        reconcile_only=True,
    )

    assert result.status == "confirmed"
    assert result.submitted is False
    assert chain.submit_calls == 0
    assert [row.status for row in journal.history] == ["confirmed"]
    assert journal.row.submit_block == 100
    assert journal.row.retry_after_block == 120
    assert journal.row.confirmed_last_update == 120


def test_reconcile_only_released_pre_submit_row_fails_without_mutation():
    projection = _projection(block=100)
    released = WeightPublicationRecord(
        projection.digest,
        "released",
        submit_block=100,
        retry_after_block=120,
        reason="late reveal reviewed",
    )
    journal = Journal(released)
    chain = Chain(block=121)
    chain.install(projection.weights, update=99)

    with pytest.raises(WeightPublicationError, match="post-submit"):
        reconcile_weight_publication(
            chain,
            None,
            projection,
            journal,
            refresh_blocks=20,
            reconcile_only=True,
        )

    assert journal.row == released
    assert journal.history == []
    assert chain.submit_calls == 0


def test_reconcile_only_released_refresh_due_row_fails_without_mutation():
    projection = _projection(block=100)
    released = WeightPublicationRecord(
        projection.digest,
        "released",
        submit_block=100,
        retry_after_block=120,
        reason="late reveal reviewed",
    )
    journal = Journal(released)
    chain = Chain(block=121)
    chain.install(projection.weights, update=100)

    with pytest.raises(WeightPublicationError, match="refresh-due"):
        reconcile_weight_publication(
            chain,
            None,
            projection,
            journal,
            refresh_blocks=20,
            reconcile_only=True,
        )

    assert journal.row == released
    assert journal.history == []
    assert chain.submit_calls == 0


def test_reconcile_only_rejects_a_signer_before_chain_or_journal_access():
    chain, journal = Chain(), Journal()

    with pytest.raises(WeightPublicationError, match="forbids a signer"):
        reconcile_weight_publication(
            chain,
            _wallet(),
            _projection(),
            journal,
            refresh_blocks=20,
            reconcile_only=True,
        )

    assert chain.metagraph_reads == []
    assert chain.weight_reads == 0
    assert journal.history == []
