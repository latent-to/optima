from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from optima.chain.debt_publication import (
    PUBLICATION_KIND_COMPOSED,
    PUBLICATION_KIND_CORE,
    ConfirmedDebtWeightPublication,
    DebtPublicationError,
    DebtWeightReadback,
    SQLiteDebtWeightPublicationJournal,
    build_confirmed_debt_weight_publication,
    build_debt_weight_publication_binding,
    ensure_debt_publication_schema,
    next_debt_boundary_schedule,
    reopen_confirmed_debt_publication,
    retain_confirmed_debt_publication,
)
from optima.chain.weights import WeightPublicationRecord
from optima.finite_debt import PPM, DebtEpochProjection, DebtHotkeyWeight
from optima.stack_identity import canonical_digest


def _h(value: str) -> str:
    return canonical_digest("test.debt-publication", value)


def _confirmation(*, projection: str | None = None):
    projection_digest = projection or _h("projection")
    weights = (
        DebtHotkeyWeight("miner", 850_000),
        DebtHotkeyWeight("reserve", 150_000),
    )
    readback = DebtWeightReadback(
        _h("scope"),
        307,
        "validator",
        7_250,
        "0x" + f"{7250:064x}",
        7_240,
        weights,
    )
    publication = WeightPublicationRecord(
        projection_digest,
        "confirmed",
        submit_block=7_230,
        retry_after_block=7_240,
        confirmed_block=readback.block,
        confirmed_last_update=readback.last_update_block,
        reason="finalized_exact_readback",
    )
    return ConfirmedDebtWeightPublication(
        readback.chain_scope_digest,
        PUBLICATION_KIND_COMPOSED,
        _h("policy"),
        projection_digest,
        projection_digest,
        7_210,
        "0x" + f"{7210:064x}",
        weights,
        readback,
        publication,
    )


def _db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.row_factory = sqlite3.Row
    ensure_debt_publication_schema(db)
    return db


def test_confirmation_requires_exact_confirmed_readback() -> None:
    confirmation = _confirmation()
    confirmation.validate_projection(
        chain_scope_digest=confirmation.chain_scope_digest,
        publication_kind=confirmation.publication_kind,
        policy_digest=confirmation.policy_digest,
        projection_digest=confirmation.projection_digest,
        effective_block=confirmation.effective_block,
        effective_block_hash=confirmation.effective_block_hash,
        weights=confirmation.weights,
    )

    pending = WeightPublicationRecord(
        confirmation.projection_digest,
        "pending",
        submit_block=7_230,
        retry_after_block=7_240,
    )
    with pytest.raises(DebtPublicationError, match="confirmed finalized readback"):
        ConfirmedDebtWeightPublication(
            confirmation.chain_scope_digest,
            confirmation.publication_kind,
            confirmation.policy_digest,
            confirmation.projection_digest,
            confirmation.weight_projection_digest,
            confirmation.effective_block,
            confirmation.effective_block_hash,
            confirmation.weights,
            confirmation.readback,
            pending,
        )

    with pytest.raises(DebtPublicationError, match="projected epoch"):
        confirmation.validate_projection(
            chain_scope_digest=confirmation.chain_scope_digest,
            publication_kind=confirmation.publication_kind,
            policy_digest=confirmation.policy_digest,
            projection_digest=_h("another projection"),
            effective_block=confirmation.effective_block,
            effective_block_hash=confirmation.effective_block_hash,
            weights=confirmation.weights,
        )


def test_confirmation_retention_is_immutable_idempotent_and_reopenable() -> None:
    db = _db()
    confirmation = _confirmation()
    db.execute("BEGIN IMMEDIATE")
    assert retain_confirmed_debt_publication(db, confirmation) == confirmation
    assert retain_confirmed_debt_publication(db, confirmation) == confirmation
    db.execute("COMMIT")
    assert reopen_confirmed_debt_publication(db, confirmation.digest) == confirmation

    another = _confirmation(projection=_h("another projection"))
    db.execute("BEGIN IMMEDIATE")
    with pytest.raises(DebtPublicationError, match="already binds other bytes"):
        retain_confirmed_debt_publication(db, another)
    db.execute("ROLLBACK")

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute(
            "UPDATE debt_weight_publication_confirmations SET record_json='{}'"
        )


def test_boundary_schedule_retains_the_earliest_gapless_boundary() -> None:
    policy = _h("policy")
    not_due = next_debt_boundary_schedule(
        policy_digest=policy,
        activation_block=10,
        epoch_blocks=100,
        closed_effective_blocks=(),
        finalized_block=109,
    )
    assert (not_due.next_epoch_index, not_due.next_effective_block, not_due.status) == (
        1,
        110,
        "not_due",
    )
    ready = next_debt_boundary_schedule(
        policy_digest=policy,
        activation_block=10,
        epoch_blocks=100,
        closed_effective_blocks=(),
        finalized_block=110,
    )
    assert ready.status == "ready"
    missed = next_debt_boundary_schedule(
        policy_digest=policy,
        activation_block=10,
        epoch_blocks=100,
        closed_effective_blocks=(),
        finalized_block=350,
    )
    assert (missed.next_epoch_index, missed.next_effective_block, missed.status) == (
        1,
        110,
        "catch_up_required",
    )
    caught_up = next_debt_boundary_schedule(
        policy_digest=policy,
        activation_block=10,
        epoch_blocks=100,
        closed_effective_blocks=(110, 210),
        finalized_block=350,
        previous_confirmed_block=250,
    )
    assert (
        caught_up.next_epoch_index,
        caught_up.next_effective_block,
        caught_up.not_before_block,
    ) == (3, 310, 350)
    rate_limited = next_debt_boundary_schedule(
        policy_digest=policy,
        activation_block=10,
        epoch_blocks=100,
        closed_effective_blocks=(110,),
        finalized_block=350,
        previous_confirmed_block=350,
    )
    assert rate_limited.next_effective_block == 210
    assert rate_limited.not_before_block == 450
    assert rate_limited.status == "not_due"
    with pytest.raises(DebtPublicationError, match="gapless prefix"):
        next_debt_boundary_schedule(
            policy_digest=policy,
            activation_block=10,
            epoch_blocks=100,
            closed_effective_blocks=(210,),
            finalized_block=350,
            previous_confirmed_block=250,
        )


def test_restart_safe_journal_reopens_exact_economic_weight_binding() -> None:
    economic = DebtEpochProjection(
        _h("core policy"),
        110,
        PPM,
        900_000,
        0,
        0,
        "reserve",
        PPM,
        (),
        (),
        (DebtHotkeyWeight("reserve", PPM),),
    )
    metagraph = SimpleNamespace(
        block=110,
        block_hash="0x" + f"{110:064x}",
        hotkeys=["reserve", "validator"],
        uids=[1, 2],
    )
    binding = build_debt_weight_publication_binding(
        economic,
        publication_kind=PUBLICATION_KIND_CORE,
        activation_digest=_h("activation"),
        chain_scope_digest=_h("scope"),
        netuid=307,
        validator_hotkey="validator",
        boundary_metagraph=metagraph,
        epoch_index=1,
    )
    db = _db()

    class Transaction:
        def __enter__(self):
            db.execute("BEGIN IMMEDIATE")

        def __exit__(self, exc_type, _exc, _tb):
            db.execute("ROLLBACK" if exc_type else "COMMIT")

    validation_transactions = []
    journal = SQLiteDebtWeightPublicationJournal(
        db,
        transaction=Transaction,
        binding=binding,
        validate_new_binding=lambda _binding: validation_transactions.append(
            db.in_transaction
        ),
    )
    intent = WeightPublicationRecord(
        binding.weight_projection.digest,
        "intent",
        submit_block=110,
        retry_after_block=120,
        reason="before_sdk_submission",
    )
    journal.compare_and_swap(None, intent)
    assert validation_transactions == [True]
    reopened = SQLiteDebtWeightPublicationJournal.reopen_from_head(
        db, transaction=Transaction
    )
    assert reopened.load() == intent
    assert reopened.head_binding() == binding
    assert reopened.retained_projection(binding.weight_projection.digest) == (
        binding.weight_projection
    )
    assert reopened.retained_authority(intent.digest) == (intent, binding)


def test_confirmation_builder_binds_exact_finalized_sparse_row() -> None:
    economic = DebtEpochProjection(
        _h("core policy"),
        110,
        PPM,
        900_000,
        0,
        0,
        "reserve",
        PPM,
        (),
        (),
        (DebtHotkeyWeight("reserve", PPM),),
    )
    boundary = SimpleNamespace(
        block=110,
        block_hash="0x" + f"{110:064x}",
        hotkeys=["reserve", "validator"],
        uids=[1, 2],
    )
    binding = build_debt_weight_publication_binding(
        economic,
        publication_kind=PUBLICATION_KIND_CORE,
        activation_digest=_h("activation"),
        chain_scope_digest=_h("scope"),
        netuid=307,
        validator_hotkey="validator",
        boundary_metagraph=boundary,
        epoch_index=1,
    )
    record = WeightPublicationRecord(
        binding.weight_projection.digest,
        "confirmed",
        confirmed_block=120,
        confirmed_last_update=115,
        reason="exact readback",
    )
    metagraph = SimpleNamespace(
        block=120,
        block_hash="0x" + f"{120:064x}",
    )
    snapshot = SimpleNamespace(weights={"reserve": 1.0}, last_update_block=115)
    confirmation = build_confirmed_debt_weight_publication(
        binding,
        record,
        confirmed_metagraph=metagraph,
        confirmed_snapshot=snapshot,
    )
    assert confirmation.publication_record == record
    assert confirmation.readback.block_hash == metagraph.block_hash

    with pytest.raises(DebtPublicationError, match="sparse row"):
        build_confirmed_debt_weight_publication(
            binding,
            record,
            confirmed_metagraph=metagraph,
            confirmed_snapshot=SimpleNamespace(
                weights={"reserve": 0.5}, last_update_block=115
            ),
        )
