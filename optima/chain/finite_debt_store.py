"""Durable, inactive-by-default authority for finite innovation debt.

The arithmetic lives in :mod:`optima.finite_debt`.  This module only binds that
pure policy to finalized chain authority and stores append-only policy, clock,
claim, balance, event, and epoch records.  It deliberately does not publish
weights and does not infer claims from the pre-existing settlement journal.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Iterable

from optima._strict import require_digest, require_exact_fields, require_int
from optima.finite_debt import (
    IMPROVEMENT_GROSS,
    PPM,
    DebtClaimBalance,
    DebtClaimState,
    DebtEpochProjection,
    FiniteDebtError,
    FiniteDebtPolicyManifest,
    InnovationDebtClaim,
    apply_debt_epoch_projection,
    cancel_claim_balance,
    expire_claim_balance,
    issue_innovation_claim,
    log_improvement_units_ppm,
    project_debt_epoch,
    resets_family_clock,
)
from optima.stack_identity import canonical_digest


SCHEMA_VERSION = 4
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_HOTKEY = re.compile(r"[^\s]{1,256}\Z")
_TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
# Core policy activation is a lower-level half of the selected incentive
# cutover.  Keeping the capability private prevents the owning intake API from
# durably activating arbitrary core bytes without the composition fence in the
# same SQLite transaction.
_ATOMIC_COMPOSITION_ACTIVATION = object()
_CLOCK_SOURCES = frozenset(
    {"seed", "crown", "crown_no_debt", "invalidation"}
)
_EVENT_TYPES = frozenset(
    {
        "policy_activated",
        "claim_issued",
        "claim_expired",
        "claim_cancelled",
        "claim_not_issued",
        "family_invalidated",
        "epoch_paid",
        # Schema-5 composition appends to this same chain so restart audit has
        # one total order across both reward classes.
        "composition_policy_activated",
        "discovery_win_retained",
        "discovery_review_expired",
        "discovery_disposition_recorded",
        "discovery_claim_issued",
        "discovery_claim_expired",
        "discovery_claim_cancelled",
        "composed_epoch_paid",
    }
)


class FiniteDebtStoreError(RuntimeError):
    """Durable finite-debt authority is malformed, stale, or inconsistent."""


def _digest(value: object, field: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    return require_digest(value, field=field, error=FiniteDebtStoreError)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    return require_int(
        value,
        field=field,
        error=FiniteDebtStoreError,
        minimum=minimum,
    )


def _block_hash(value: object, field: str = "block_hash") -> str:
    if not isinstance(value, str) or _BLOCK_HASH.fullmatch(value) is None:
        raise FiniteDebtStoreError(f"{field} is malformed")
    return value


def _hotkey(value: object, field: str = "hotkey") -> str:
    if not isinstance(value, str) or _HOTKEY.fullmatch(value) is None:
        raise FiniteDebtStoreError(f"{field} is malformed")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def reward_family_id(
    arena_digest: str,
    target_id: str,
    target_spec_digest: str,
) -> str:
    """Return the stable reward-family identity shared with V1 standing claims."""

    arena = _digest(arena_digest, "arena_digest")
    target_spec = _digest(target_spec_digest, "target_spec_digest")
    if not isinstance(target_id, str) or _TARGET.fullmatch(target_id) is None:
        raise FiniteDebtStoreError("target_id is malformed")
    return canonical_digest(
        "optima.economics.standing-family",
        {
            "arena_digest": arena,
            "target_id": target_id,
            "target_spec_digest": target_spec,
        },
    )


@dataclass(frozen=True)
class SeededFamilyClock:
    """Explicit pre-activation accepted-crown order, without retroactive debt."""

    family_id: str
    accepted_crown_block: int
    accepted_crown_block_hash: str
    event_index: int
    event_subindex: int
    reservation_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _digest(self.family_id, "family_id"))
        _integer(self.accepted_crown_block, "accepted_crown_block")
        object.__setattr__(
            self,
            "accepted_crown_block_hash",
            _block_hash(self.accepted_crown_block_hash, "accepted_crown_block_hash"),
        )
        _integer(self.event_index, "event_index")
        _integer(self.event_subindex, "event_subindex")
        object.__setattr__(
            self,
            "reservation_digest",
            _digest(self.reservation_digest, "reservation_digest"),
        )

    @property
    def finalized_order(self) -> tuple[int, int, int, str]:
        return (
            self.accepted_crown_block,
            self.event_index,
            self.event_subindex,
            self.reservation_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "SeededFamilyClock":
        row = require_exact_fields(
            value,
            fields=frozenset(cls.__dataclass_fields__),
            label="seeded family clock",
            error=FiniteDebtStoreError,
            exact_dict=True,
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FiniteDebtPolicyActivation:
    """One exact policy activation and its explicitly seeded family clocks."""

    chain_scope_digest: str
    policy: FiniteDebtPolicyManifest
    activation_block: int
    activation_block_hash: str
    previous_policy_digest: str
    seeded_family_clocks: tuple[SeededFamilyClock, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chain_scope_digest",
            _digest(self.chain_scope_digest, "chain_scope_digest"),
        )
        if type(self.policy) is not FiniteDebtPolicyManifest:
            raise FiniteDebtStoreError("activation policy is not exactly typed")
        _integer(self.activation_block, "activation_block")
        object.__setattr__(
            self,
            "activation_block_hash",
            _block_hash(self.activation_block_hash, "activation_block_hash"),
        )
        object.__setattr__(
            self,
            "previous_policy_digest",
            _digest(
                self.previous_policy_digest,
                "previous_policy_digest",
                optional=True,
            ),
        )
        seeds = tuple(self.seeded_family_clocks)
        if (
            any(type(row) is not SeededFamilyClock for row in seeds)
            or tuple(row.family_id for row in seeds)
            != tuple(sorted({row.family_id for row in seeds}))
            or any(row.accepted_crown_block > self.activation_block for row in seeds)
            or any(row.family_id not in self.policy.family_ids for row in seeds)
        ):
            raise FiniteDebtStoreError("activation family-clock seeds are not canonical")
        object.__setattr__(self, "seeded_family_clocks", seeds)

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_block": self.activation_block,
            "activation_block_hash": self.activation_block_hash,
            "chain_scope_digest": self.chain_scope_digest,
            "policy": self.policy.to_dict(),
            "previous_policy_digest": self.previous_policy_digest,
            "seeded_family_clocks": [
                row.to_dict() for row in self.seeded_family_clocks
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> "FiniteDebtPolicyActivation":
        fields = {
            "activation_block",
            "activation_block_hash",
            "chain_scope_digest",
            "policy",
            "previous_policy_digest",
            "seeded_family_clocks",
        }
        row = dict(
            require_exact_fields(
                value,
                fields=frozenset(fields),
                label="finite-debt policy activation",
                error=FiniteDebtStoreError,
                exact_dict=True,
            )
        )
        seeds = row["seeded_family_clocks"]
        if type(seeds) is not list:
            raise FiniteDebtStoreError("seeded_family_clocks must be an array")
        row["policy"] = FiniteDebtPolicyManifest.from_dict(row["policy"])
        row["seeded_family_clocks"] = tuple(
            SeededFamilyClock.from_dict(item) for item in seeds
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.finite-debt-policy-activation",
            self.to_dict(),
        )


@dataclass(frozen=True)
class FiniteDebtFamilyClock:
    """One append-only family-clock history entry with full finalized order."""

    policy_digest: str
    family_id: str
    accepted_crown_block: int
    accepted_crown_block_hash: str
    event_index: int
    event_subindex: int
    reservation_digest: str
    source: str
    claim_digest: str
    reward_event_digest: str

    def __post_init__(self) -> None:
        for field in ("policy_digest", "family_id", "reservation_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        _integer(self.accepted_crown_block, "accepted_crown_block")
        object.__setattr__(
            self,
            "accepted_crown_block_hash",
            _block_hash(self.accepted_crown_block_hash, "accepted_crown_block_hash"),
        )
        _integer(self.event_index, "event_index")
        _integer(self.event_subindex, "event_subindex")
        if self.source not in _CLOCK_SOURCES:
            raise FiniteDebtStoreError("family clock source is unsupported")
        object.__setattr__(
            self,
            "claim_digest",
            _digest(
                self.claim_digest,
                "claim_digest",
                optional=self.source in {"seed", "crown_no_debt", "invalidation"},
            ),
        )
        object.__setattr__(
            self,
            "reward_event_digest",
            _digest(self.reward_event_digest, "reward_event_digest"),
        )
        if self.source == "crown" and not self.claim_digest:
            raise FiniteDebtStoreError("crown family clock lacks a claim")
        if self.source == "seed" and self.claim_digest:
            raise FiniteDebtStoreError("seeded family clock cannot name a claim")
        if self.source == "invalidation" and self.claim_digest:
            raise FiniteDebtStoreError("invalidated family clock cannot name a claim")
        if self.source == "crown_no_debt" and self.claim_digest:
            raise FiniteDebtStoreError("no-debt crown clock cannot name a claim")

    @property
    def finalized_order(self) -> tuple[int, int, int, str]:
        return (
            self.accepted_crown_block,
            self.event_index,
            self.event_subindex,
            self.reservation_digest,
        )


@dataclass(frozen=True)
class FiniteDebtRewardEpoch:
    """One confirmed, immutable claim-pool epoch closure."""

    chain_scope_digest: str
    activation_digest: str
    policy_digest: str
    epoch_index: int
    start_block: int
    effective_block: int
    effective_block_hash: str
    projection: DebtEpochProjection
    publication_record_digest: str
    payout_event_digest: str

    def __post_init__(self) -> None:
        for field in (
            "chain_scope_digest",
            "activation_digest",
            "policy_digest",
            "publication_record_digest",
            "payout_event_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        _integer(self.epoch_index, "epoch_index", minimum=1)
        _integer(self.start_block, "start_block")
        _integer(self.effective_block, "effective_block", minimum=1)
        object.__setattr__(
            self,
            "effective_block_hash",
            _block_hash(self.effective_block_hash, "effective_block_hash"),
        )
        if (
            type(self.projection) is not DebtEpochProjection
            or self.projection.policy_digest != self.policy_digest
            or self.projection.effective_block != self.effective_block
            or self.start_block >= self.effective_block
        ):
            raise FiniteDebtStoreError("reward epoch projection is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_digest": self.activation_digest,
            "chain_scope_digest": self.chain_scope_digest,
            "effective_block": self.effective_block,
            "effective_block_hash": self.effective_block_hash,
            "epoch_index": self.epoch_index,
            "payout_event_digest": self.payout_event_digest,
            "policy_digest": self.policy_digest,
            "projection": self.projection.to_dict(),
            "publication_record_digest": self.publication_record_digest,
            "start_block": self.start_block,
        }

    @classmethod
    def from_dict(cls, value: object) -> "FiniteDebtRewardEpoch":
        row = dict(
            require_exact_fields(
                value,
                fields=frozenset(cls.__dataclass_fields__),
                label="finite-debt reward epoch",
                error=FiniteDebtStoreError,
                exact_dict=True,
            )
        )
        row["projection"] = DebtEpochProjection.from_dict(row["projection"])
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.chain.finite-debt-reward-epoch", self.to_dict())


_TABLE_DEFINITIONS = (
    """
    CREATE TABLE finite_debt_reward_events (
        sequence INTEGER PRIMARY KEY,
        event_digest TEXT NOT NULL UNIQUE,
        previous_event_digest TEXT NOT NULL,
        chain_scope_digest TEXT NOT NULL,
        event_type TEXT NOT NULL,
        block INTEGER NOT NULL CHECK(block>=0),
        block_hash TEXT NOT NULL,
        payload_json TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_policy_activations (
        activation_digest TEXT PRIMARY KEY,
        chain_scope_digest TEXT NOT NULL,
        policy_digest TEXT NOT NULL UNIQUE,
        policy_json TEXT NOT NULL,
        activation_block INTEGER NOT NULL CHECK(activation_block>=0),
        activation_block_hash TEXT NOT NULL,
        previous_policy_digest TEXT NOT NULL,
        seeded_clocks_json TEXT NOT NULL,
        activation_json TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_family_clocks (
        clock_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        policy_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(policy_digest),
        family_id TEXT NOT NULL,
        accepted_crown_block INTEGER NOT NULL CHECK(accepted_crown_block>=0),
        accepted_crown_block_hash TEXT NOT NULL,
        event_index INTEGER NOT NULL CHECK(event_index>=0),
        event_subindex INTEGER NOT NULL CHECK(event_subindex>=0),
        reservation_digest TEXT NOT NULL,
        source TEXT NOT NULL
            CHECK(source IN ('seed','crown','crown_no_debt','invalidation')),
        claim_digest TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL
            REFERENCES finite_debt_reward_events(event_digest),
        UNIQUE(policy_digest,family_id,accepted_crown_block,event_index,
               event_subindex,reservation_digest)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_claims (
        claim_digest TEXT PRIMARY KEY,
        policy_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(policy_digest),
        family_id TEXT NOT NULL,
        candidate_digest TEXT NOT NULL UNIQUE,
        retained_evidence_digest TEXT NOT NULL,
        hotkey TEXT NOT NULL,
        accepted_crown_block INTEGER NOT NULL CHECK(accepted_crown_block>=0),
        accepted_crown_block_hash TEXT NOT NULL,
        event_index INTEGER NOT NULL CHECK(event_index>=0),
        event_subindex INTEGER NOT NULL CHECK(event_subindex>=0),
        reservation_digest TEXT NOT NULL,
        settlement_block INTEGER NOT NULL CHECK(settlement_block>=0),
        settlement_block_hash TEXT NOT NULL,
        settlement_event_digest TEXT NOT NULL UNIQUE,
        principal_units INTEGER NOT NULL CHECK(principal_units>0),
        claim_json TEXT NOT NULL,
        issuance_reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_claim_balances (
        claim_digest TEXT NOT NULL REFERENCES finite_debt_claims(claim_digest),
        revision INTEGER NOT NULL CHECK(revision>=0),
        balance_digest TEXT NOT NULL UNIQUE,
        principal_units INTEGER NOT NULL CHECK(principal_units>0),
        paid_units INTEGER NOT NULL CHECK(paid_units>=0),
        forfeited_units INTEGER NOT NULL CHECK(forfeited_units>=0),
        remaining_units INTEGER NOT NULL CHECK(remaining_units>=0),
        status TEXT NOT NULL CHECK(status IN ('open','paid','expired','cancelled')),
        terminal_block INTEGER,
        terminal_reason TEXT NOT NULL,
        balance_json TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL
            REFERENCES finite_debt_reward_events(event_digest),
        PRIMARY KEY(claim_digest,revision),
        CHECK(paid_units+forfeited_units+remaining_units=principal_units)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_reward_epochs (
        epoch_digest TEXT PRIMARY KEY,
        chain_scope_digest TEXT NOT NULL,
        activation_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(activation_digest),
        policy_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(policy_digest),
        epoch_index INTEGER NOT NULL CHECK(epoch_index>0),
        start_block INTEGER NOT NULL CHECK(start_block>=0),
        effective_block INTEGER NOT NULL CHECK(effective_block>start_block),
        effective_block_hash TEXT NOT NULL,
        projection_digest TEXT NOT NULL UNIQUE,
        projection_json TEXT NOT NULL,
        publication_record_digest TEXT NOT NULL UNIQUE,
        payout_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest),
        payout_units INTEGER NOT NULL CHECK(payout_units>=0),
        reserve_units INTEGER NOT NULL CHECK(reserve_units>=0),
        epoch_json TEXT NOT NULL,
        UNIQUE(policy_digest,epoch_index),
        UNIQUE(policy_digest,effective_block),
        CHECK(payout_units+reserve_units=1000000)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_epoch_allocations (
        epoch_digest TEXT NOT NULL
            REFERENCES finite_debt_reward_epochs(epoch_digest),
        claim_digest TEXT NOT NULL REFERENCES finite_debt_claims(claim_digest),
        hotkey TEXT NOT NULL,
        units INTEGER NOT NULL CHECK(units>0),
        PRIMARY KEY(epoch_digest,claim_digest)
    ) STRICT
    """,
)

_INDEX_DEFINITIONS = (
    "CREATE INDEX finite_debt_activations_block ON "
    "finite_debt_policy_activations(activation_block,activation_digest)",
    "CREATE INDEX finite_debt_clocks_latest ON "
    "finite_debt_family_clocks(policy_digest,family_id,accepted_crown_block DESC,"
    "event_index DESC,event_subindex DESC,reservation_digest DESC)",
    "CREATE INDEX finite_debt_balances_latest ON "
    "finite_debt_claim_balances(claim_digest,revision DESC)",
)

_IMMUTABLE_TABLES = (
    "finite_debt_reward_events",
    "finite_debt_policy_activations",
    "finite_debt_family_clocks",
    "finite_debt_claims",
    "finite_debt_claim_balances",
    "finite_debt_reward_epochs",
    "finite_debt_epoch_allocations",
)

_REQUIRED_COLUMNS = {
    "finite_debt_reward_events": {
        "sequence", "event_digest", "previous_event_digest", "chain_scope_digest",
        "event_type", "block", "block_hash", "payload_json",
    },
    "finite_debt_policy_activations": {
        "activation_digest", "chain_scope_digest", "policy_digest", "policy_json",
        "activation_block", "activation_block_hash", "previous_policy_digest",
        "seeded_clocks_json", "activation_json", "reward_event_digest",
    },
    "finite_debt_family_clocks": {
        "clock_sequence", "policy_digest", "family_id", "accepted_crown_block",
        "accepted_crown_block_hash", "event_index", "event_subindex",
        "reservation_digest", "source", "claim_digest", "reward_event_digest",
    },
    "finite_debt_claims": {
        "claim_digest", "policy_digest", "family_id", "candidate_digest",
        "retained_evidence_digest", "hotkey", "accepted_crown_block",
        "accepted_crown_block_hash", "event_index", "event_subindex",
        "reservation_digest", "settlement_block", "settlement_block_hash",
        "settlement_event_digest", "principal_units", "claim_json",
        "issuance_reward_event_digest",
    },
    "finite_debt_claim_balances": {
        "claim_digest", "revision", "balance_digest", "principal_units",
        "paid_units", "forfeited_units", "remaining_units", "status",
        "terminal_block", "terminal_reason", "balance_json", "reward_event_digest",
    },
    "finite_debt_reward_epochs": {
        "epoch_digest", "chain_scope_digest", "activation_digest", "policy_digest",
        "epoch_index", "start_block", "effective_block", "effective_block_hash",
        "projection_digest", "projection_json", "publication_record_digest",
        "payout_event_digest", "payout_units", "reserve_units", "epoch_json",
    },
    "finite_debt_epoch_allocations": {
        "epoch_digest", "claim_digest", "hotkey", "units",
    },
}


def _verify_schema(db: sqlite3.Connection) -> None:
    tables = {
        row["name"]: row
        for row in db.execute("PRAGMA table_list")
        if row["type"] == "table"
    }
    for table, columns in _REQUIRED_COLUMNS.items():
        row = tables.get(table)
        if row is None or row["strict"] != 1:
            raise FiniteDebtStoreError(f"{table} is missing or not STRICT")
        observed = {item["name"] for item in db.execute(f"PRAGMA table_info({table})")}
        if observed != columns:
            raise FiniteDebtStoreError(f"{table} columns differ from schema 4")
    triggers = {
        row["name"]
        for row in db.execute(
            "SELECT name FROM sqlite_schema WHERE type='trigger'"
        )
    }
    required_triggers = {
        f"{table}_reject_{action}"
        for table in _IMMUTABLE_TABLES
        for action in ("update", "delete")
    }
    if not required_triggers.issubset(triggers):
        raise FiniteDebtStoreError("finite-debt immutability triggers are incomplete")


def migrate_schema3_to4(db: sqlite3.Connection) -> None:
    """Create only empty finite-debt tables and advance metadata 3 -> 4."""

    schema = db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
    if schema is None:
        raise FiniteDebtStoreError("intake schema metadata is absent")
    # Schema 5 is an additive composition extension.  Reopening it must still
    # verify every schema-4 authority table before the schema-5 verifier runs.
    if schema["value"] in {str(SCHEMA_VERSION), "5", "6"}:
        _verify_schema(db)
        return
    if schema["value"] != "3":
        raise FiniteDebtStoreError("finite-debt migration requires intake schema 3")
    existing = {
        row["name"]
        for row in db.execute("PRAGMA table_list")
        if row["type"] == "table"
    }
    try:
        db.execute("BEGIN IMMEDIATE")
        for definition in _TABLE_DEFINITIONS:
            table = definition.split("CREATE TABLE ", 1)[1].split(" ", 1)[0]
            if table not in existing:
                db.execute(definition)
            elif db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                raise FiniteDebtStoreError(
                    "schema-3 database contains non-authoritative finite-debt rows"
                )
        for definition in _INDEX_DEFINITIONS:
            db.execute(definition.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS "))
        for table in _IMMUTABLE_TABLES:
            for action in ("UPDATE", "DELETE"):
                name = f"{table}_reject_{action.lower()}"
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {name} BEFORE {action} ON {table} "
                    "BEGIN SELECT RAISE(ABORT,'finite-debt rows are immutable'); END"
                )
        _verify_schema(db)
        db.execute(
            "UPDATE metadata SET value=? WHERE key='schema' AND value='3'",
            (str(SCHEMA_VERSION),),
        )
        if db.execute("SELECT changes() AS n").fetchone()["n"] != 1:
            raise FiniteDebtStoreError("intake schema changed during migration")
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise


class FiniteDebtStore:
    """Persistence helper owned by one :class:`FinalizedIntakeStore` writer."""

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        chain_scope_digest: str,
        transaction: Callable[[], object],
        finalized_cursor: Callable[[], tuple[int, str] | None],
    ):
        if not isinstance(db, sqlite3.Connection):
            raise FiniteDebtStoreError("finite-debt store requires SQLite authority")
        self.db = db
        self.chain_scope_digest = _digest(chain_scope_digest, "chain_scope_digest")
        self._transaction = transaction
        self._finalized_cursor = finalized_cursor
        _verify_schema(db)
        # Policy v1 was never activated.  Refuse a retained legacy or corrupt
        # activation at open time rather than discovering incompatible bytes
        # only when the first payout path is exercised.
        self.policy_activations()

    def _require_finalized_authority(self, block: int, block_hash: str) -> None:
        height = _integer(block, "finalized block")
        digest = _block_hash(block_hash, "finalized block hash")
        if self._finalized_cursor() != (height, digest):
            raise FiniteDebtStoreError(
                "finite-debt action lacks the exact finalized block/hash authority"
            )

    def composition_active_at(self, block: int) -> bool:
        """Whether schema-5 composition owns new projections at ``block``.

        The query deliberately has no import dependency on the composition
        store.  Schema 4 remains independently reopenable, while schema 5 can
        establish a hard cutover that legacy callers cannot accidentally skip.
        """

        height = _integer(block, "composition lookup block")
        table = self.db.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' "
            "AND name='incentive_composition_activations'"
        ).fetchone()
        if table is None:
            return False
        return self.db.execute(
            "SELECT 1 FROM incentive_composition_activations "
            "WHERE activation_block<=? LIMIT 1",
            (height,),
        ).fetchone() is not None

    def composition_activation_block_at(self, block: int) -> int | None:
        """Return the active composition cutover block, if any."""

        height = _integer(block, "composition lookup block")
        table = self.db.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' "
            "AND name='incentive_composition_activations'"
        ).fetchone()
        if table is None:
            return None
        row = self.db.execute(
            "SELECT MAX(activation_block) AS value FROM "
            "incentive_composition_activations WHERE activation_block<=?",
            (height,),
        ).fetchone()
        return row["value"]

    def composition_activated(self) -> bool:
        """Whether any durable composition cutover has occurred."""

        table = self.db.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' "
            "AND name='incentive_composition_activations'"
        ).fetchone()
        return table is not None and self.db.execute(
            "SELECT 1 FROM incentive_composition_activations LIMIT 1"
        ).fetchone() is not None

    @staticmethod
    def _event_digest(
        *,
        chain_scope_digest: str,
        sequence: int,
        previous_event_digest: str,
        event_type: str,
        block: int,
        block_hash: str,
        payload: dict[str, object],
    ) -> str:
        return canonical_digest(
            "optima.chain.finite-debt-reward-event",
            {
                "block": block,
                "block_hash": block_hash,
                "chain_scope_digest": chain_scope_digest,
                "event_type": event_type,
                "payload": payload,
                "previous_event_digest": previous_event_digest,
                "sequence": sequence,
            },
        )

    def reward_events(self) -> tuple[dict[str, object], ...]:
        """Reopen and verify the complete append-only reward-event hash chain."""

        result: list[dict[str, object]] = []
        previous = ""
        expected_sequence = 0
        for row in self.db.execute(
            "SELECT * FROM finite_debt_reward_events ORDER BY sequence"
        ):
            if (
                row["sequence"] != expected_sequence
                or row["previous_event_digest"] != previous
                or row["chain_scope_digest"] != self.chain_scope_digest
                or row["event_type"] not in _EVENT_TYPES
            ):
                raise FiniteDebtStoreError("finite-debt reward-event chain is not contiguous")
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise FiniteDebtStoreError(
                    f"finite-debt reward event is corrupt: {exc}"
                ) from None
            if type(payload) is not dict or _canonical_json(payload) != row["payload_json"]:
                raise FiniteDebtStoreError("finite-debt reward event payload is not canonical")
            expected = self._event_digest(
                chain_scope_digest=self.chain_scope_digest,
                sequence=row["sequence"],
                previous_event_digest=previous,
                event_type=row["event_type"],
                block=row["block"],
                block_hash=_block_hash(row["block_hash"]),
                payload=payload,
            )
            if expected != row["event_digest"]:
                raise FiniteDebtStoreError("finite-debt reward event digest differs")
            result.append(
                {
                    "block": row["block"],
                    "block_hash": row["block_hash"],
                    "event_digest": row["event_digest"],
                    "event_type": row["event_type"],
                    "payload": payload,
                    "previous_event_digest": previous,
                    "sequence": row["sequence"],
                }
            )
            previous = row["event_digest"]
            expected_sequence += 1
        return tuple(result)

    def _append_event(
        self,
        event_type: str,
        *,
        block: int,
        block_hash: str,
        payload: dict[str, object],
    ) -> str:
        if not self.db.in_transaction:
            raise FiniteDebtStoreError(
                "finite-debt reward events require the owning write transaction"
            )
        if event_type not in _EVENT_TYPES:
            raise FiniteDebtStoreError("finite-debt reward event type is unsupported")
        height = _integer(block, "reward event block")
        authority_hash = _block_hash(block_hash, "reward event block hash")
        payload_json = _canonical_json(payload)
        if type(json.loads(payload_json)) is not dict:
            raise FiniteDebtStoreError("finite-debt reward event payload must be an object")
        events = self.reward_events()
        sequence = len(events)
        previous = "" if not events else str(events[-1]["event_digest"])
        event_digest = self._event_digest(
            chain_scope_digest=self.chain_scope_digest,
            sequence=sequence,
            previous_event_digest=previous,
            event_type=event_type,
            block=height,
            block_hash=authority_hash,
            payload=payload,
        )
        self.db.execute(
            "INSERT INTO finite_debt_reward_events(sequence,event_digest,"
            "previous_event_digest,chain_scope_digest,event_type,block,block_hash,"
            "payload_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                sequence,
                event_digest,
                previous,
                self.chain_scope_digest,
                event_type,
                height,
                authority_hash,
                payload_json,
            ),
        )
        return event_digest

    def _activation_from_row(self, row: sqlite3.Row) -> FiniteDebtPolicyActivation:
        try:
            activation = FiniteDebtPolicyActivation.from_dict(
                json.loads(row["activation_json"])
            )
        except (FiniteDebtError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FiniteDebtStoreError(
                f"finite-debt policy activation is corrupt: {exc}"
            ) from None
        if (
            activation.digest != row["activation_digest"]
            or activation.chain_scope_digest != self.chain_scope_digest
            or activation.policy.digest != row["policy_digest"]
            or _canonical_json(activation.policy.to_dict()) != row["policy_json"]
            or activation.activation_block != row["activation_block"]
            or activation.activation_block_hash != row["activation_block_hash"]
            or activation.previous_policy_digest != row["previous_policy_digest"]
            or _canonical_json(
                [item.to_dict() for item in activation.seeded_family_clocks]
            )
            != row["seeded_clocks_json"]
        ):
            raise FiniteDebtStoreError("finite-debt activation differs from retained bytes")
        event = self.db.execute(
            "SELECT event_type,block,block_hash,payload_json FROM "
            "finite_debt_reward_events WHERE event_digest=?",
            (row["reward_event_digest"],),
        ).fetchone()
        expected_payload = {
            "activation": activation.to_dict(),
            "activation_digest": activation.digest,
        }
        if (
            event is None
            or event["event_type"] != "policy_activated"
            or event["block"] != activation.activation_block
            or event["block_hash"] != activation.activation_block_hash
            or event["payload_json"] != _canonical_json(expected_payload)
        ):
            raise FiniteDebtStoreError("finite-debt activation event differs")
        return activation

    def policy_activations(self) -> tuple[FiniteDebtPolicyActivation, ...]:
        self.reward_events()
        activations = tuple(
            self._activation_from_row(row)
            for row in self.db.execute(
                "SELECT * FROM finite_debt_policy_activations "
                "ORDER BY activation_block,activation_digest"
            )
        )
        previous = ""
        for activation in activations:
            if activation.previous_policy_digest != previous:
                raise FiniteDebtStoreError("finite-debt policy activation chain differs")
            previous = activation.policy.digest
        return activations

    def active_policy_activation(
        self,
        *,
        at_block: int,
    ) -> FiniteDebtPolicyActivation | None:
        block = _integer(at_block, "policy lookup block")
        rows = tuple(
            activation
            for activation in self.policy_activations()
            if activation.activation_block <= block
        )
        return None if not rows else rows[-1]

    def _activation_by_policy(self, policy_digest: str) -> FiniteDebtPolicyActivation:
        digest = _digest(policy_digest, "policy_digest")
        self.reward_events()
        row = self.db.execute(
            "SELECT * FROM finite_debt_policy_activations WHERE policy_digest=?",
            (digest,),
        ).fetchone()
        if row is None:
            raise FiniteDebtStoreError("finite-debt claim names an absent policy")
        activation = self._activation_from_row(row)
        if activation.policy.digest != digest:
            raise FiniteDebtStoreError("finite-debt activation policy digest differs")
        return activation

    def activate_policy(
        self,
        policy: FiniteDebtPolicyManifest,
        *,
        activation_block: int,
        activation_block_hash: str,
        seeded_family_clocks: Iterable[SeededFamilyClock] = (),
        _atomic_composition_authority: object | None = None,
    ) -> FiniteDebtPolicyActivation:
        """Activate the core half of one atomic selected-composition cutover.

        This method is intentionally inaccessible through the public intake
        authority.  The composition store supplies the private capability and
        calls it inside the same outer transaction that inserts the selected
        composition activation.  A savepoint may be released here, but the
        durable commit remains owned by that outer transaction.
        """

        if _atomic_composition_authority is not _ATOMIC_COMPOSITION_ACTIVATION:
            raise FiniteDebtStoreError(
                "standalone finite-debt activation is disabled; use the atomic "
                "selected incentive cutover"
            )

        if type(policy) is not FiniteDebtPolicyManifest:
            raise FiniteDebtStoreError("finite-debt activation policy is not exactly typed")
        if policy.improvement_basis != IMPROVEMENT_GROSS:
            raise FiniteDebtStoreError("durable finite debt requires gross improvement")
        height = _integer(activation_block, "activation_block")
        authority_hash = _block_hash(activation_block_hash, "activation_block_hash")
        seeds = tuple(sorted(tuple(seeded_family_clocks), key=lambda row: row.family_id))
        if any(type(row) is not SeededFamilyClock for row in seeds):
            raise FiniteDebtStoreError("seeded family clocks are not exactly typed")
        with self._transaction():
            self._require_finalized_authority(height, authority_hash)
            exact = self.db.execute(
                "SELECT * FROM finite_debt_policy_activations "
                "WHERE activation_block=? AND activation_block_hash=?",
                (height, authority_hash),
            ).fetchone()
            if exact is not None:
                existing = self._activation_from_row(exact)
                if existing.policy == policy and existing.seeded_family_clocks == seeds:
                    return existing
                raise FiniteDebtStoreError("activation block already binds another policy")
            if self.composition_active_at(height):
                raise FiniteDebtStoreError(
                    "core policy upgrades are disabled after incentive composition activation"
                )
            activations = self.policy_activations()
            prior = None if not activations else activations[-1]
            if prior is not None:
                if height <= prior.activation_block:
                    raise FiniteDebtStoreError("policy activation block did not advance")
                states = self._claim_states()
                if any(state.balance.status == "open" for state in states):
                    raise FiniteDebtStoreError("cannot upgrade policy with open debt")
            if self.db.execute(
                "SELECT 1 FROM finite_debt_policy_activations WHERE policy_digest=?",
                (policy.digest,),
            ).fetchone() is not None:
                raise FiniteDebtStoreError("finite-debt policy was already activated")
            activation = FiniteDebtPolicyActivation(
                self.chain_scope_digest,
                policy,
                height,
                authority_hash,
                "" if prior is None else prior.policy.digest,
                seeds,
            )
            event_digest = self._append_event(
                "policy_activated",
                block=height,
                block_hash=authority_hash,
                payload={
                    "activation": activation.to_dict(),
                    "activation_digest": activation.digest,
                },
            )
            self.db.execute(
                "INSERT INTO finite_debt_policy_activations(activation_digest,"
                "chain_scope_digest,policy_digest,policy_json,activation_block,"
                "activation_block_hash,previous_policy_digest,seeded_clocks_json,"
                "activation_json,reward_event_digest) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    activation.digest,
                    self.chain_scope_digest,
                    policy.digest,
                    _canonical_json(policy.to_dict()),
                    height,
                    authority_hash,
                    activation.previous_policy_digest,
                    _canonical_json([row.to_dict() for row in seeds]),
                    _canonical_json(activation.to_dict()),
                    event_digest,
                ),
            )
            for seed in seeds:
                self.db.execute(
                    "INSERT INTO finite_debt_family_clocks(policy_digest,family_id,"
                    "accepted_crown_block,accepted_crown_block_hash,event_index,"
                    "event_subindex,reservation_digest,source,claim_digest,"
                    "reward_event_digest) VALUES(?,?,?,?,?,?,?,'seed','',?)",
                    (
                        policy.digest,
                        seed.family_id,
                        seed.accepted_crown_block,
                        seed.accepted_crown_block_hash,
                        seed.event_index,
                        seed.event_subindex,
                        seed.reservation_digest,
                        event_digest,
                    ),
                )
        return activation

    @staticmethod
    def _clock_from_row(row: sqlite3.Row) -> FiniteDebtFamilyClock:
        return FiniteDebtFamilyClock(
            row["policy_digest"],
            row["family_id"],
            row["accepted_crown_block"],
            row["accepted_crown_block_hash"],
            row["event_index"],
            row["event_subindex"],
            row["reservation_digest"],
            row["source"],
            row["claim_digest"],
            row["reward_event_digest"],
        )

    @staticmethod
    def _clock_identity(clock: FiniteDebtFamilyClock) -> dict[str, object]:
        return {
            "accepted_crown_block": clock.accepted_crown_block,
            "claim_digest": clock.claim_digest,
            "event_index": clock.event_index,
            "event_subindex": clock.event_subindex,
            "reservation_digest": clock.reservation_digest,
            "source": clock.source,
        }

    @staticmethod
    def _claim_prior_clock_identity(
        clock: FiniteDebtFamilyClock | None,
    ) -> dict[str, object] | None:
        if clock is None:
            return None
        return {
            "accepted_crown_block": clock.accepted_crown_block,
            "event_index": clock.event_index,
            "event_subindex": clock.event_subindex,
            "reservation_digest": clock.reservation_digest,
        }

    def _retained_crown_authority(
        self,
        *,
        candidate_digest: str,
        reservation_digest: str,
        settlement_event_digest: str,
        retained_evidence_digest: str,
    ):
        """Reopen the exact typed CROWN authority used by finite debt.

        This is intentionally a database-only reconstruction.  Qualification
        artifact bytes were independently reopened before settlement; restart
        must nevertheless bind the retained typed candidate, both qualification
        identities, their evidence receipt, and the exact CROWN journal event.
        """

        from optima.eval.evidence_store import EvidenceArtifactRef, EvidenceStoreError
        from optima.settlement import (
            SettlementCandidate,
            SettlementError,
            SettlementEvent,
            SettlementEventType,
            SettlementEvidence,
            SettlementQualification,
        )

        candidate_id = _digest(candidate_digest, "candidate_digest")
        reservation = _digest(reservation_digest, "reservation_digest")
        settlement_id = _digest(
            settlement_event_digest, "settlement_event_digest"
        )
        evidence_id = _digest(
            retained_evidence_digest, "retained_evidence_digest"
        )
        candidate_row = self.db.execute(
            "SELECT * FROM settlement_candidates WHERE candidate_digest=? "
            "AND reservation_id=?",
            (candidate_id, reservation),
        ).fetchone()
        arrival = self.db.execute(
            "SELECT block,block_hash,event_index,event_subindex,hotkey FROM "
            "reservations WHERE reservation_id=?",
            (reservation,),
        ).fetchone()
        event_row = self.db.execute(
            "SELECT * FROM settlement_events WHERE event_digest=?",
            (settlement_id,),
        ).fetchone()
        retained = tuple(
            self.db.execute(
                "SELECT reproduction_index,qualification_digest,qualification_json,"
                "attempt_ref_json,evidence_root FROM settlement_qualifications "
                "WHERE reservation_id=? ORDER BY reproduction_index",
                (reservation,),
            )
        )
        if (
            candidate_row is None
            or arrival is None
            or event_row is None
            or len(retained) != 2
            or tuple(row["reproduction_index"] for row in retained) != (0, 1)
        ):
            raise FiniteDebtStoreError(
                "finite-debt CROWN lacks retained typed settlement authority"
            )
        try:
            candidate = SettlementCandidate.from_dict(
                json.loads(candidate_row["candidate_json"])
            )
            event = SettlementEvent.from_dict(json.loads(event_row["event_json"]))
            qualifications = []
            references = []
            for row in retained:
                qualification = SettlementQualification.from_dict(
                    json.loads(row["qualification_json"])
                )
                reference = EvidenceArtifactRef.from_dict(
                    json.loads(row["attempt_ref_json"])
                )
                if (
                    qualification.digest != row["qualification_digest"]
                    or _canonical_json(qualification.to_dict())
                    != row["qualification_json"]
                    or reference.sha256
                    != qualification.qualification_attempt_digest
                    or _canonical_json(reference.to_dict()) != row["attempt_ref_json"]
                ):
                    raise FiniteDebtStoreError(
                        "finite-debt CROWN qualification bytes differ"
                    )
                dispositions = tuple(
                    self.db.execute(
                        "SELECT authority_digest,report_digest,decision FROM "
                        "qualification_dispositions WHERE reservation_id=? "
                        "AND evidence_digest=?",
                        (reservation, qualification.qualification_attempt_digest),
                    )
                )
                if (
                    len(dispositions) != 1
                    or dispositions[0]["decision"] != "PASS"
                    or dispositions[0]["authority_digest"]
                    != qualification.qualification_authority_digest
                    or dispositions[0]["report_digest"]
                    != qualification.qualification_report_digest
                ):
                    raise FiniteDebtStoreError(
                        "finite-debt CROWN qualification lost PASS authority"
                    )
                qualifications.append(qualification)
                references.append(reference)
            evidence = SettlementEvidence.bind(
                candidate,
                primary_attempt_ref=references[0],
                reproduction_attempt_ref=references[1],
            )
        except FiniteDebtStoreError:
            raise
        except (
            EvidenceStoreError,
            SettlementError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise FiniteDebtStoreError(
                f"finite-debt CROWN authority is corrupt: {exc}"
            ) from None
        contribution = (
            None
            if candidate.candidate_manifest is None
            else candidate.candidate_manifest.entries.get(candidate.target_id)
        )
        if (
            candidate.digest != candidate_id
            or _canonical_json(candidate.to_dict()) != candidate_row["candidate_json"]
            or candidate_row["authority_digest"]
            != candidate.qualification_authority_digest
            or candidate_row["status"] != "crowned"
            or candidate_row["settlement_evidence_digest"] != evidence_id
            or evidence.digest != evidence_id
            or tuple(qualifications) != (candidate.primary, candidate.reproduction)
            or candidate_row["evidence_root"] != retained[0]["evidence_root"]
            or candidate_row["reproduction_evidence_root"]
            != retained[1]["evidence_root"]
            or candidate.lane != "registered"
            or contribution is None
            or candidate.reservation_digest != reservation
            or arrival["block"] != candidate.finalized_block
            or arrival["block_hash"]
            != _block_hash(arrival["block_hash"], "accepted crown block hash")
            or arrival["event_index"] != candidate.event_index
            or arrival["event_subindex"] != candidate.event_subindex
            or arrival["hotkey"] != candidate.hotkey
            or event.digest != settlement_id
            or event_row["sequence"] != event.sequence
            or event_row["event_id"] != event.digest
            or event_row["event_type"] != "CROWN"
            or event_row["reservation_id"] != reservation
            or event_row["arena_id"] != candidate.arena_digest
            or event_row["target_id"] != candidate.target_id
            or event_row["event_json"] != _canonical_json(event.to_dict())
            or event.event_type is not SettlementEventType.CROWN
            or event.candidate_digest != candidate.digest
            or event.subject_digest != contribution.digest
            or event.target_id != candidate.target_id
            or event.from_stack_digest != candidate.incumbent_stack_digest
            or event.from_tree_digest != candidate.incumbent_tree_digest
            or event.to_stack_digest != candidate.incumbent_stack_digest
            or event.to_tree_digest != candidate.incumbent_tree_digest
            or event.reason != "qualified_win"
        ):
            raise FiniteDebtStoreError(
                "finite-debt CROWN differs from retained typed settlement authority"
            )
        return candidate, evidence, event, arrival, contribution

    def _no_issue_authority(self, event: dict[str, object]):
        payload = event["payload"]
        fields = {
            "accepted_crown_block",
            "accepted_crown_block_hash",
            "activation_digest",
            "candidate_digest",
            "event_index",
            "event_subindex",
            "family_id",
            "hotkey",
            "reason",
            "reservation_digest",
            "retained_evidence_digest",
            "settlement_event_digest",
        }
        if type(payload) is not dict or set(payload) != fields:
            raise FiniteDebtStoreError("finite-debt no-issue event payload differs")
        activation_row = self.db.execute(
            "SELECT * FROM finite_debt_policy_activations "
            "WHERE activation_digest=?",
            (payload["activation_digest"],),
        ).fetchone()
        if activation_row is None:
            raise FiniteDebtStoreError("finite-debt no-issue activation is absent")
        activation = self._activation_from_row(activation_row)
        active = self.active_policy_activation(at_block=event["block"])
        candidate, evidence, settlement_event, arrival, contribution = (
            self._retained_crown_authority(
                candidate_digest=payload["candidate_digest"],
                reservation_digest=payload["reservation_digest"],
                settlement_event_digest=payload["settlement_event_digest"],
                retained_evidence_digest=payload["retained_evidence_digest"],
            )
        )
        family = reward_family_id(
            candidate.arena_digest,
            candidate.target_id,
            contribution.target_spec_digest,
        )
        expected = {
            "accepted_crown_block": candidate.finalized_block,
            "accepted_crown_block_hash": arrival["block_hash"],
            "activation_digest": activation.digest,
            "candidate_digest": candidate.digest,
            "event_index": candidate.event_index,
            "event_subindex": candidate.event_subindex,
            "family_id": family,
            "hotkey": candidate.hotkey,
            "reason": payload["reason"],
            "reservation_digest": candidate.reservation_digest,
            "retained_evidence_digest": evidence.digest,
            "settlement_event_digest": settlement_event.digest,
        }
        if (
            active is None
            or active.digest != activation.digest
            or event["block"] < candidate.finalized_block
            or payload != expected
            or payload["reason"]
            not in {
                "accepted_before_policy_activation",
                "accepted_before_family_invalidation",
                "improvement_rounds_to_zero",
                "principal_rounds_to_zero",
            }
        ):
            raise FiniteDebtStoreError(
                "finite-debt no-issue event differs from CROWN authority"
            )
        return activation, candidate, str(payload["reason"]), arrival

    def _validate_family_invalidation(
        self,
        event: dict[str, object],
        *,
        prior_clock: FiniteDebtFamilyClock | None,
    ) -> FiniteDebtFamilyClock:
        payload = event["payload"]
        fields = {
            "activation_digest",
            "claim_balance_transitions",
            "family_id",
            "invalidation_digest",
            "policy_digest",
            "prior_family_clock",
        }
        if type(payload) is not dict or set(payload) != fields:
            raise FiniteDebtStoreError(
                "finite-debt family invalidation event payload differs"
            )
        activation_row = self.db.execute(
            "SELECT * FROM finite_debt_policy_activations "
            "WHERE activation_digest=? AND policy_digest=?",
            (payload["activation_digest"], payload["policy_digest"]),
        ).fetchone()
        if activation_row is None:
            raise FiniteDebtStoreError(
                "finite-debt family invalidation activation is absent"
            )
        activation = self._activation_from_row(activation_row)
        active = self.active_policy_activation(at_block=event["block"])
        family = _digest(payload["family_id"], "family_id")
        invalidation = _digest(
            payload["invalidation_digest"], "invalidation_digest"
        )
        try:
            activation.policy.campaign_for_family(family)
        except FiniteDebtError as exc:
            raise FiniteDebtStoreError(
                f"finite-debt family invalidation is outside its policy: {exc}"
            ) from None
        marker = FiniteDebtFamilyClock(
            activation.policy.digest,
            family,
            _integer(event["block"], "family invalidation block"),
            _block_hash(event["block_hash"], "family invalidation block hash"),
            9223372036854775807,
            9223372036854775807,
            invalidation,
            "invalidation",
            "",
            _digest(event["event_digest"], "reward_event_digest"),
        )
        if (
            active is None
            or active.digest != activation.digest
            or payload["policy_digest"] != activation.policy.digest
            or (prior_clock is not None and marker.accepted_crown_block
                <= prior_clock.accepted_crown_block)
        ):
            raise FiniteDebtStoreError(
                "finite-debt family invalidation authority differs"
            )

        expected_by_claim: dict[str, tuple[sqlite3.Row, DebtClaimBalance]] = {}
        for claim_row in self.db.execute(
            "SELECT * FROM finite_debt_claims WHERE policy_digest=? "
            "AND family_id=? ORDER BY claim_digest",
            (activation.policy.digest, family),
        ):
            claim = self._claim_from_row(claim_row)
            issuance = self.db.execute(
                "SELECT sequence FROM finite_debt_reward_events "
                "WHERE event_digest=?",
                (claim_row["issuance_reward_event_digest"],),
            ).fetchone()
            if issuance is None or issuance["sequence"] >= event["sequence"]:
                continue
            before_rows = tuple(
                self.db.execute(
                    "SELECT b.*,e.sequence AS event_sequence FROM "
                    "finite_debt_claim_balances AS b JOIN "
                    "finite_debt_reward_events AS e ON "
                    "e.event_digest=b.reward_event_digest "
                    "WHERE b.claim_digest=? AND e.sequence<? ORDER BY b.revision",
                    (claim.digest, event["sequence"]),
                )
            )
            if not before_rows:
                raise FiniteDebtStoreError(
                    "finite-debt family invalidation lacks issued balance"
                )
            before_row = before_rows[-1]
            before = self._balance_from_row(before_row)
            if before.status != "open":
                continue
            try:
                after = cancel_claim_balance(
                    claim,
                    before,
                    at_block=marker.accepted_crown_block,
                    reason="runtime_invalidation",
                )
            except FiniteDebtError as exc:
                raise FiniteDebtStoreError(
                    f"finite-debt family invalidation cannot replay: {exc}"
                ) from None
            expected_by_claim[claim.digest] = (before_row, after)

        actual_rows = tuple(
            self.db.execute(
                "SELECT b.*,c.policy_digest AS claim_policy_digest,"
                "c.family_id AS claim_family_id FROM "
                "finite_debt_claim_balances AS b JOIN finite_debt_claims AS c "
                "USING(claim_digest) WHERE b.reward_event_digest=? "
                "ORDER BY b.claim_digest",
                (marker.reward_event_digest,),
            )
        )
        if any(
            row["claim_policy_digest"] != activation.policy.digest
            or row["claim_family_id"] != family
            for row in actual_rows
        ):
            raise FiniteDebtStoreError(
                "finite-debt family invalidation crossed policy or family"
            )
        if len(actual_rows) != len(expected_by_claim):
            raise FiniteDebtStoreError(
                "finite-debt family invalidation balance set differs"
            )
        transitions: list[dict[str, str]] = []
        for balance_row in actual_rows:
            expected_pair = expected_by_claim.get(balance_row["claim_digest"])
            if (
                expected_pair is None
            ):
                raise FiniteDebtStoreError(
                    "finite-debt family invalidation balance set differs"
                )
            before_row, expected_after = expected_pair
            actual_after = self._balance_from_row(balance_row)
            if (
                balance_row["revision"] != before_row["revision"] + 1
                or actual_after != expected_after
            ):
                raise FiniteDebtStoreError(
                    "finite-debt family invalidation balance differs"
                )
            transitions.append(
                {
                    "after_balance_digest": actual_after.digest,
                    "before_balance_digest": before_row["balance_digest"],
                    "claim_digest": actual_after.claim_digest,
                }
            )
        expected_payload = {
            "activation_digest": activation.digest,
            "claim_balance_transitions": transitions,
            "family_id": family,
            "invalidation_digest": invalidation,
            "policy_digest": activation.policy.digest,
            "prior_family_clock": (
                None
                if prior_clock is None
                else self._clock_identity(prior_clock)
            ),
        }
        if payload != expected_payload:
            raise FiniteDebtStoreError(
                "finite-debt family invalidation event differs"
            )
        return marker

    def family_clocks(
        self,
        *,
        policy_digest: str | None = None,
        family_id: str | None = None,
    ) -> tuple[FiniteDebtFamilyClock, ...]:
        selected_policy = (
            None
            if policy_digest is None
            else _digest(policy_digest, "policy_digest")
        )
        selected_family = (
            None if family_id is None else _digest(family_id, "family_id")
        )
        events = self.reward_events()
        activation_rows = {
            row["reward_event_digest"]: row
            for row in self.db.execute(
                "SELECT * FROM finite_debt_policy_activations"
            )
        }
        claim_rows = {
            row["issuance_reward_event_digest"]: row
            for row in self.db.execute("SELECT * FROM finite_debt_claims")
        }
        expected: list[FiniteDebtFamilyClock] = []
        latest: dict[tuple[str, str], FiniteDebtFamilyClock] = {}
        seen_activations: set[str] = set()
        seen_claims: set[str] = set()
        crown_outcomes: set[str] = set()

        def append_clock(clock: FiniteDebtFamilyClock) -> None:
            key = (clock.policy_digest, clock.family_id)
            prior = latest.get(key)
            if prior is not None and clock.finalized_order <= prior.finalized_order:
                raise FiniteDebtStoreError(
                    "finite-debt family clock order did not advance"
                )
            latest[key] = clock
            expected.append(clock)

        for event in events:
            event_type = event["event_type"]
            event_digest = str(event["event_digest"])
            if event_type == "policy_activated":
                activation_row = activation_rows.get(event_digest)
                if activation_row is None:
                    raise FiniteDebtStoreError(
                        "finite-debt policy activation event is orphaned"
                    )
                activation = self._activation_from_row(activation_row)
                seen_activations.add(activation.digest)
                for seed in activation.seeded_family_clocks:
                    append_clock(
                        FiniteDebtFamilyClock(
                            activation.policy.digest,
                            seed.family_id,
                            seed.accepted_crown_block,
                            seed.accepted_crown_block_hash,
                            seed.event_index,
                            seed.event_subindex,
                            seed.reservation_digest,
                            "seed",
                            "",
                            event_digest,
                        )
                    )
                continue

            if event_type == "claim_issued":
                claim_row = claim_rows.get(event_digest)
                if claim_row is None:
                    raise FiniteDebtStoreError(
                        "finite-debt claim-issued event is orphaned"
                    )
                claim = self._claim_from_row(claim_row)
                if claim.settlement_block != event["block"]:
                    raise FiniteDebtStoreError(
                        "finite-debt claim-issued settlement block differs"
                    )
                if claim_row["settlement_event_digest"] in crown_outcomes:
                    raise FiniteDebtStoreError(
                        "finite-debt CROWN has duplicate issuance outcomes"
                    )
                crown_outcomes.add(claim_row["settlement_event_digest"])
                seen_claims.add(claim.digest)
                key = (claim.policy_digest, claim.family_id)
                prior = latest.get(key)
                expected_prior_block = (
                    None
                    if prior is None or prior.source == "invalidation"
                    else prior.accepted_crown_block
                )
                payload = event["payload"]
                if (
                    claim.prior_accepted_crown_block != expected_prior_block
                    or type(payload) is not dict
                    or payload.get("prior_family_clock")
                    != self._claim_prior_clock_identity(prior)
                ):
                    raise FiniteDebtStoreError(
                        "finite-debt claim prior family clock differs"
                    )
                if claim.resets_clock:
                    append_clock(
                        FiniteDebtFamilyClock(
                            claim.policy_digest,
                            claim.family_id,
                            claim.accepted_crown_block,
                            claim_row["accepted_crown_block_hash"],
                            claim_row["event_index"],
                            claim_row["event_subindex"],
                            claim_row["reservation_digest"],
                            "crown",
                            claim.digest,
                            event_digest,
                        )
                    )
                continue

            if event_type == "claim_not_issued":
                activation, candidate, reason, arrival = self._no_issue_authority(
                    event
                )
                payload = event["payload"]
                settlement_event_digest = str(payload["settlement_event_digest"])
                if settlement_event_digest in crown_outcomes:
                    raise FiniteDebtStoreError(
                        "finite-debt CROWN has duplicate issuance outcomes"
                    )
                crown_outcomes.add(settlement_event_digest)
                family = str(payload["family_id"])
                key = (activation.policy.digest, family)
                prior = latest.get(key)
                accepted_order = candidate.finalized_order
                units: int | None = None
                if reason == "accepted_before_policy_activation":
                    valid_reason = (
                        candidate.finalized_block < activation.activation_block
                    )
                elif reason == "accepted_before_family_invalidation":
                    valid_reason = (
                        candidate.hotkey != activation.policy.reserve_hotkey
                        and prior is not None
                        and prior.source == "invalidation"
                        and accepted_order <= prior.finalized_order
                    )
                else:
                    valid_reason = (
                        candidate.finalized_block >= activation.activation_block
                        and candidate.hotkey != activation.policy.reserve_hotkey
                        and (prior is None or accepted_order > prior.finalized_order)
                    )
                    try:
                        units = log_improvement_units_ppm(
                            candidate.speedup,
                            basis=activation.policy.improvement_basis,
                            threshold_speedup="1",
                        )
                    except FiniteDebtError as exc:
                        valid_reason = valid_reason and (
                            reason == "improvement_rounds_to_zero"
                            and str(exc)
                            == "improvement rounds to zero 1%-log-unit ppm"
                        )
                    else:
                        if reason != "principal_rounds_to_zero":
                            valid_reason = False
                        else:
                            try:
                                issue_innovation_claim(
                                    activation.policy,
                                    family_id=family,
                                    candidate_digest=candidate.digest,
                                    retained_evidence_digest=str(
                                        payload["retained_evidence_digest"]
                                    ),
                                    hotkey=candidate.hotkey,
                                    settled_speedup=candidate.speedup,
                                    threshold_speedup="1",
                                    accepted_crown_block=candidate.finalized_block,
                                    prior_accepted_crown_block=(
                                        None
                                        if prior is None
                                        or prior.source == "invalidation"
                                        else prior.accepted_crown_block
                                    ),
                                    settlement_block=int(event["block"]),
                                )
                            except FiniteDebtError as exc:
                                valid_reason = valid_reason and (
                                    str(exc) == "claim principal rounds to zero"
                                )
                            else:
                                valid_reason = False
                if not valid_reason:
                    raise FiniteDebtStoreError(
                        "finite-debt no-issue reason cannot replay"
                    )
                if (
                    reason == "principal_rounds_to_zero"
                    and units is not None
                    and resets_family_clock(activation.policy, units)
                ):
                    append_clock(
                        FiniteDebtFamilyClock(
                            activation.policy.digest,
                            family,
                            candidate.finalized_block,
                            arrival["block_hash"],
                            candidate.event_index,
                            candidate.event_subindex,
                            candidate.reservation_digest,
                            "crown_no_debt",
                            "",
                            event_digest,
                        )
                    )
                continue

            if event_type == "family_invalidated":
                payload = event["payload"]
                if type(payload) is not dict:
                    raise FiniteDebtStoreError(
                        "finite-debt family invalidation payload differs"
                    )
                key = (
                    _digest(payload.get("policy_digest"), "policy_digest"),
                    _digest(payload.get("family_id"), "family_id"),
                )
                append_clock(
                    self._validate_family_invalidation(
                        event,
                        prior_clock=latest.get(key),
                    )
                )

        if len(seen_activations) != len(activation_rows):
            raise FiniteDebtStoreError(
                "finite-debt activation rows differ from activation events"
            )
        if len(seen_claims) != len(claim_rows):
            raise FiniteDebtStoreError(
                "finite-debt claim rows differ from claim-issued events"
            )
        actual = tuple(
            self._clock_from_row(row)
            for row in self.db.execute(
                "SELECT * FROM finite_debt_family_clocks ORDER BY "
                "policy_digest,family_id,accepted_crown_block,event_index,"
                "event_subindex,reservation_digest"
            )
        )
        expected_rows = tuple(
            sorted(
                expected,
                key=lambda row: (
                    row.policy_digest,
                    row.family_id,
                    *row.finalized_order,
                ),
            )
        )
        if actual != expected_rows:
            raise FiniteDebtStoreError(
                "finite-debt family clocks differ from exact reward authority"
            )
        return tuple(
            row
            for row in actual
            if (selected_policy is None or row.policy_digest == selected_policy)
            and (selected_family is None or row.family_id == selected_family)
        )

    def _latest_clock(
        self,
        policy_digest: str,
        family_id: str,
    ) -> FiniteDebtFamilyClock | None:
        rows = self.family_clocks(
            policy_digest=policy_digest,
            family_id=family_id,
        )
        return None if not rows else rows[-1]

    def _claim_from_row(self, row: sqlite3.Row) -> InnovationDebtClaim:
        try:
            claim = InnovationDebtClaim.from_dict(json.loads(row["claim_json"]))
            activation = self._activation_by_policy(row["policy_digest"])
            claim.validate_policy(activation.policy)
        except (FiniteDebtError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FiniteDebtStoreError(f"finite-debt claim is corrupt: {exc}") from None
        if (
            claim.digest != row["claim_digest"]
            or claim.policy_digest != row["policy_digest"]
            or claim.family_id != row["family_id"]
            or claim.candidate_digest != row["candidate_digest"]
            or claim.retained_evidence_digest != row["retained_evidence_digest"]
            or claim.hotkey != row["hotkey"]
            or claim.accepted_crown_block != row["accepted_crown_block"]
            or claim.settlement_block != row["settlement_block"]
            or claim.principal_units != row["principal_units"]
            or _canonical_json(claim.to_dict()) != row["claim_json"]
        ):
            raise FiniteDebtStoreError("finite-debt claim differs from retained bytes")
        _block_hash(row["accepted_crown_block_hash"], "accepted crown block hash")
        _block_hash(row["settlement_block_hash"], "settlement block hash")
        _digest(row["reservation_digest"], "reservation_digest")
        _digest(row["settlement_event_digest"], "settlement_event_digest")
        _digest(row["issuance_reward_event_digest"], "issuance_reward_event_digest")
        candidate, evidence, settlement, arrival, contribution = (
            self._retained_crown_authority(
                candidate_digest=claim.candidate_digest,
                reservation_digest=row["reservation_digest"],
                settlement_event_digest=row["settlement_event_digest"],
                retained_evidence_digest=claim.retained_evidence_digest,
            )
        )
        event = self.db.execute(
            "SELECT event_type,block,block_hash,payload_json FROM "
            "finite_debt_reward_events WHERE event_digest=?",
            (row["issuance_reward_event_digest"],),
        ).fetchone()
        active = self.active_policy_activation(at_block=claim.settlement_block)
        family = reward_family_id(
            candidate.arena_digest,
            candidate.target_id,
            contribution.target_spec_digest,
        )
        if (
            active is None
            or active.digest != activation.digest
            or candidate.digest != claim.candidate_digest
            or candidate.speedup != claim.settled_speedup
            or candidate.hotkey != claim.hotkey
            or candidate.finalized_block != claim.accepted_crown_block
            or candidate.event_index != row["event_index"]
            or candidate.event_subindex != row["event_subindex"]
            or candidate.reservation_digest != row["reservation_digest"]
            or evidence.digest != claim.retained_evidence_digest
            or family != claim.family_id
            or arrival["block"] != row["accepted_crown_block"]
            or arrival["block_hash"] != row["accepted_crown_block_hash"]
            or arrival["event_index"] != row["event_index"]
            or arrival["event_subindex"] != row["event_subindex"]
            or arrival["hotkey"] != claim.hotkey
            or settlement.digest != row["settlement_event_digest"]
            or event is None
            or event["event_type"] != "claim_issued"
            or event["block"] != claim.settlement_block
            or event["block_hash"] != row["settlement_block_hash"]
        ):
            raise FiniteDebtStoreError("finite-debt claim authority differs")
        try:
            payload = json.loads(event["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FiniteDebtStoreError(
                f"finite-debt claim event is corrupt: {exc}"
            ) from None
        expected_keys = {
            "accepted_crown_block_hash",
            "activation_digest",
            "claim",
            "claim_digest",
            "event_index",
            "event_subindex",
            "prior_family_clock",
            "reservation_digest",
            "settlement_event_digest",
        }
        if (
            type(payload) is not dict
            or set(payload) != expected_keys
            or payload.get("claim") != claim.to_dict()
            or payload.get("claim_digest") != claim.digest
            or payload.get("accepted_crown_block_hash")
            != row["accepted_crown_block_hash"]
            or payload.get("activation_digest") != activation.digest
            or payload.get("event_index") != row["event_index"]
            or payload.get("event_subindex") != row["event_subindex"]
            or payload.get("reservation_digest") != row["reservation_digest"]
            or payload.get("settlement_event_digest")
            != row["settlement_event_digest"]
        ):
            raise FiniteDebtStoreError("finite-debt claim event payload differs")
        return claim

    def invalidate_family(
        self,
        *,
        policy_digest: str,
        family_id: str,
        invalidation_digest: str,
        current_block: int,
        current_block_hash: str,
    ) -> tuple[DebtClaimState, ...]:
        """Cancel one family's open debt and reset its next-crown clock.

        The invalidation is an append-only finalized authority. Its clock marker
        makes a later accepted crown behave as the family's first crown while
        preserving the complete pre-invalidation history.
        """

        policy_id = _digest(policy_digest, "policy_digest")
        family = _digest(family_id, "family_id")
        invalidation = _digest(invalidation_digest, "invalidation_digest")
        height = _integer(current_block, "family invalidation block")
        authority_hash = _block_hash(
            current_block_hash, "family invalidation block hash"
        )
        with self._transaction():
            exact = self.db.execute(
                "SELECT * FROM finite_debt_family_clocks WHERE policy_digest=? "
                "AND family_id=? AND source='invalidation' AND reservation_digest=?",
                (policy_id, family, invalidation),
            ).fetchone()
            if exact is not None:
                clock = self._clock_from_row(exact)
                if (
                    clock.accepted_crown_block != height
                    or clock.accepted_crown_block_hash != authority_hash
                ):
                    raise FiniteDebtStoreError(
                        "family invalidation retry differs from retained authority"
                    )
                self.family_clocks(policy_digest=policy_id, family_id=family)
                return tuple(
                    state
                    for state in self._claim_states(policy_digest=policy_id)
                    if state.claim.family_id == family
                    and state.balance.status == "cancelled"
                    and state.balance.terminal_block == height
                    and state.balance.terminal_reason == "runtime_invalidation"
                )

            self._require_finalized_authority(height, authority_hash)
            activation = self.active_policy_activation(at_block=height)
            if activation is None or activation.policy.digest != policy_id:
                raise FiniteDebtStoreError(
                    "family invalidation policy is not the active finalized policy"
                )
            try:
                activation.policy.campaign_for_family(family)
            except FiniteDebtError as exc:
                raise FiniteDebtStoreError(
                    f"family invalidation is outside the policy: {exc}"
                ) from None

            if self.db.execute(
                "SELECT 1 FROM finite_debt_family_clocks WHERE policy_digest=? "
                "AND family_id=? AND source='invalidation' "
                "AND accepted_crown_block=?",
                (policy_id, family, height),
            ).fetchone() is not None:
                raise FiniteDebtStoreError(
                    "family already has a different invalidation at this block"
                )
            prior_clock = self._latest_clock(policy_id, family)
            if prior_clock is not None and height <= prior_clock.accepted_crown_block:
                raise FiniteDebtStoreError(
                    "family invalidation must follow its latest clock block"
                )

            changed: list[DebtClaimState] = []
            before_balance_digests: dict[str, str] = {}
            for state in self._claim_states(policy_digest=policy_id):
                if state.claim.family_id != family or state.balance.status != "open":
                    continue
                try:
                    updated = cancel_claim_balance(
                        state.claim,
                        state.balance,
                        at_block=height,
                        reason="runtime_invalidation",
                    )
                except FiniteDebtError as exc:
                    raise FiniteDebtStoreError(
                        f"family invalidation cannot cancel debt: {exc}"
                    ) from None
                changed.append(DebtClaimState(state.claim, updated))
                before_balance_digests[state.claim.digest] = state.balance.digest

            event_digest = self._append_event(
                "family_invalidated",
                block=height,
                block_hash=authority_hash,
                payload={
                    "activation_digest": activation.digest,
                    "claim_balance_transitions": sorted(
                        (
                            {
                                "after_balance_digest": state.balance.digest,
                                "before_balance_digest": before_balance_digests[
                                    state.claim.digest
                                ],
                                "claim_digest": state.claim.digest,
                            }
                            for state in changed
                        ),
                        key=lambda item: item["claim_digest"],
                    ),
                    "family_id": family,
                    "invalidation_digest": invalidation,
                    "policy_digest": policy_id,
                    "prior_family_clock": (
                        None
                        if prior_clock is None
                        else self._clock_identity(prior_clock)
                    ),
                },
            )
            self.db.execute(
                "INSERT INTO finite_debt_family_clocks(policy_digest,family_id,"
                "accepted_crown_block,accepted_crown_block_hash,event_index,"
                "event_subindex,reservation_digest,source,claim_digest,"
                "reward_event_digest) VALUES(?,?,?,?,9223372036854775807,"
                "9223372036854775807,?,'invalidation','',?)",
                (
                    policy_id,
                    family,
                    height,
                    authority_hash,
                    invalidation,
                    event_digest,
                ),
            )
            for state in changed:
                revision = self.db.execute(
                    "SELECT MAX(revision) AS value FROM finite_debt_claim_balances "
                    "WHERE claim_digest=?",
                    (state.claim.digest,),
                ).fetchone()["value"] + 1
                self._insert_balance(
                    state.balance,
                    revision=revision,
                    reward_event_digest=event_digest,
                )
        return tuple(changed)

    @staticmethod
    def _balance_from_row(row: sqlite3.Row) -> DebtClaimBalance:
        try:
            balance = DebtClaimBalance.from_dict(json.loads(row["balance_json"]))
        except (FiniteDebtError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FiniteDebtStoreError(f"finite-debt balance is corrupt: {exc}") from None
        if (
            balance.digest != row["balance_digest"]
            or balance.claim_digest != row["claim_digest"]
            or balance.principal_units != row["principal_units"]
            or balance.paid_units != row["paid_units"]
            or balance.forfeited_units != row["forfeited_units"]
            or balance.remaining_units != row["remaining_units"]
            or balance.status != row["status"]
            or balance.terminal_block != row["terminal_block"]
            or balance.terminal_reason != row["terminal_reason"]
            or _canonical_json(balance.to_dict()) != row["balance_json"]
        ):
            raise FiniteDebtStoreError("finite-debt balance differs from retained bytes")
        return balance

    def _claim_states(
        self,
        *,
        policy_digest: str | None = None,
    ) -> tuple[DebtClaimState, ...]:
        self.family_clocks(policy_digest=policy_digest)
        params: tuple[str, ...] = ()
        where = ""
        if policy_digest is not None:
            where = " WHERE policy_digest=?"
            params = (_digest(policy_digest, "policy_digest"),)
        states = []
        for claim_row in self.db.execute(
            "SELECT * FROM finite_debt_claims" + where + " ORDER BY claim_digest",
            params,
        ):
            claim = self._claim_from_row(claim_row)
            balance_rows = tuple(
                self.db.execute(
                    "SELECT * FROM finite_debt_claim_balances WHERE claim_digest=? "
                    "ORDER BY revision",
                    (claim.digest,),
                )
            )
            if not balance_rows or tuple(row["revision"] for row in balance_rows) != tuple(
                range(len(balance_rows))
            ):
                raise FiniteDebtStoreError("finite-debt balance revisions are not contiguous")
            balances = tuple(self._balance_from_row(row) for row in balance_rows)
            if balances[0] != DebtClaimBalance.open(claim):
                raise FiniteDebtStoreError("finite-debt initial balance is not open principal")
            for index, (balance_row, current) in enumerate(
                zip(balance_rows, balances, strict=True)
            ):
                event = self.db.execute(
                    "SELECT event_type,block,block_hash,payload_json FROM "
                    "finite_debt_reward_events "
                    "WHERE event_digest=?",
                    (balance_row["reward_event_digest"],),
                ).fetchone()
                if index == 0:
                    expected_events = {"claim_issued"}
                elif current.status == "expired":
                    expected_events = {"claim_expired"}
                elif current.status == "cancelled":
                    expected_events = {"claim_cancelled", "family_invalidated"}
                else:
                    expected_events = {"epoch_paid", "composed_epoch_paid"}
                if event is None or event["event_type"] not in expected_events:
                    raise FiniteDebtStoreError(
                        "finite-debt balance revision event differs"
                    )
                if index == 0:
                    continue
                prior = balances[index - 1]
                try:
                    payload = json.loads(event["payload_json"])
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise FiniteDebtStoreError(
                        f"finite-debt balance event is corrupt: {exc}"
                    ) from None
                if type(payload) is not dict:
                    raise FiniteDebtStoreError(
                        "finite-debt balance event payload differs"
                    )
                transition = {
                    "after_balance_digest": current.digest,
                    "before_balance_digest": prior.digest,
                    "claim_digest": claim.digest,
                }
                event_type = event["event_type"]
                if event_type in {"claim_expired", "claim_cancelled"}:
                    try:
                        recomputed = (
                            expire_claim_balance(claim, prior, at_block=event["block"])
                            if event_type == "claim_expired"
                            else cancel_claim_balance(
                                claim,
                                prior,
                                at_block=event["block"],
                                reason=current.terminal_reason,
                            )
                        )
                    except FiniteDebtError as exc:
                        raise FiniteDebtStoreError(
                            f"finite-debt lifecycle event cannot replay: {exc}"
                        ) from None
                    expected_payload = {
                        "after_balance": current.to_dict(),
                        "after_balance_digest": current.digest,
                        "before_balance_digest": prior.digest,
                        "claim_digest": claim.digest,
                    }
                    allowed_keys = {frozenset(expected_payload)}
                    if "composition_activation_digest" in payload:
                        allowed_keys.add(
                            frozenset(
                                (*expected_payload, "composition_activation_digest")
                            )
                        )
                    if (
                        recomputed != current
                        or frozenset(payload) not in allowed_keys
                        or any(payload.get(key) != value for key, value in expected_payload.items())
                    ):
                        raise FiniteDebtStoreError(
                            "finite-debt lifecycle event payload differs"
                        )
                else:
                    transition_field = {
                        "epoch_paid": "balance_transitions",
                        "composed_epoch_paid": "core_balance_transitions",
                        "family_invalidated": "claim_balance_transitions",
                    }[event_type]
                    transitions = payload.get(transition_field)
                    if (
                        type(transitions) is not list
                        or sum(item == transition for item in transitions) != 1
                    ):
                        raise FiniteDebtStoreError(
                            "finite-debt balance transition is not exactly authorized"
                        )
            for prior, current in zip(balances, balances[1:]):
                if (
                    prior.status != "open"
                    or current.paid_units < prior.paid_units
                    or current.forfeited_units < prior.forfeited_units
                    or current.remaining_units >= prior.remaining_units
                ):
                    raise FiniteDebtStoreError("finite-debt balance history regressed")
            states.append(DebtClaimState(claim, balances[-1]))
        return tuple(states)

    def claim_states(
        self,
        *,
        policy_digest: str | None = None,
    ) -> tuple[DebtClaimState, ...]:
        self.reward_events()
        return self._claim_states(policy_digest=policy_digest)

    def _insert_balance(
        self,
        balance: DebtClaimBalance,
        *,
        revision: int,
        reward_event_digest: str,
    ) -> None:
        if not self.db.in_transaction:
            raise FiniteDebtStoreError(
                "finite-debt balance revisions require the owning write transaction"
            )
        if type(balance) is not DebtClaimBalance:
            raise FiniteDebtStoreError("finite-debt balance is not exactly typed")
        self.db.execute(
            "INSERT INTO finite_debt_claim_balances(claim_digest,revision,"
            "balance_digest,principal_units,paid_units,forfeited_units,remaining_units,"
            "status,terminal_block,terminal_reason,balance_json,reward_event_digest) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                balance.claim_digest,
                revision,
                balance.digest,
                balance.principal_units,
                balance.paid_units,
                balance.forfeited_units,
                balance.remaining_units,
                balance.status,
                balance.terminal_block,
                balance.terminal_reason,
                _canonical_json(balance.to_dict()),
                _digest(reward_event_digest, "reward_event_digest"),
            ),
        )

    def issue_crown_in_transaction(
        self,
        *,
        arena_digest: str,
        target_id: str,
        target_spec_digest: str,
        candidate_digest: str,
        retained_evidence_digest: str,
        hotkey: str,
        settled_speedup: str,
        accepted_crown_block: int,
        accepted_crown_block_hash: str,
        event_index: int,
        event_subindex: int,
        reservation_digest: str,
        settlement_block: int,
        settlement_block_hash: str | None,
        settlement_event_digest: str,
    ) -> InnovationDebtClaim | None:
        """Issue one CROWN claim inside the caller's settlement transaction."""

        if not self.db.in_transaction:
            raise FiniteDebtStoreError(
                "finite-debt CROWN issuance requires the settlement transaction"
            )
        activation = self.active_policy_activation(at_block=settlement_block)
        if activation is None:
            return None
        if settlement_block_hash is None:
            raise FiniteDebtStoreError(
                "active finite-debt policy requires settlement block hash authority"
            )
        settlement_hash = _block_hash(
            settlement_block_hash,
            "settlement_block_hash",
        )
        self._require_finalized_authority(settlement_block, settlement_hash)
        accepted = _integer(accepted_crown_block, "accepted_crown_block")
        accepted_hash = _block_hash(
            accepted_crown_block_hash,
            "accepted_crown_block_hash",
        )
        index = _integer(event_index, "event_index")
        subindex = _integer(event_subindex, "event_subindex")
        reservation = _digest(reservation_digest, "reservation_digest")
        candidate = _digest(candidate_digest, "candidate_digest")
        evidence = _digest(retained_evidence_digest, "retained_evidence_digest")
        settlement_event = _digest(settlement_event_digest, "settlement_event_digest")
        owner = _hotkey(hotkey)
        family = reward_family_id(arena_digest, target_id, target_spec_digest)
        if accepted < activation.activation_block:
            self._append_event(
                "claim_not_issued",
                block=settlement_block,
                block_hash=settlement_hash,
                payload={
                    "accepted_crown_block": accepted,
                    "accepted_crown_block_hash": accepted_hash,
                    "activation_digest": activation.digest,
                    "candidate_digest": candidate,
                    "event_index": index,
                    "event_subindex": subindex,
                    "family_id": family,
                    "hotkey": owner,
                    "reason": "accepted_before_policy_activation",
                    "reservation_digest": reservation,
                    "retained_evidence_digest": evidence,
                    "settlement_event_digest": settlement_event,
                },
            )
            return None
        if owner == activation.policy.reserve_hotkey:
            raise FiniteDebtStoreError("reserve hotkey cannot receive finite debt")
        prior_clock = self._latest_clock(activation.policy.digest, family)
        accepted_order = (accepted, index, subindex, reservation)
        if prior_clock is not None and accepted_order <= prior_clock.finalized_order:
            if prior_clock.source == "invalidation":
                self._append_event(
                    "claim_not_issued",
                    block=settlement_block,
                    block_hash=settlement_hash,
                    payload={
                        "accepted_crown_block": accepted,
                        "accepted_crown_block_hash": accepted_hash,
                        "activation_digest": activation.digest,
                        "candidate_digest": candidate,
                        "event_index": index,
                        "event_subindex": subindex,
                        "family_id": family,
                        "hotkey": owner,
                        "reason": "accepted_before_family_invalidation",
                        "reservation_digest": reservation,
                        "retained_evidence_digest": evidence,
                        "settlement_event_digest": settlement_event,
                    },
                )
                return None
            raise FiniteDebtStoreError("accepted family crown order did not advance")
        try:
            claim = issue_innovation_claim(
                activation.policy,
                family_id=family,
                candidate_digest=candidate,
                retained_evidence_digest=evidence,
                hotkey=owner,
                settled_speedup=settled_speedup,
                threshold_speedup="1",
                accepted_crown_block=accepted,
                prior_accepted_crown_block=(
                    None
                    if prior_clock is None or prior_clock.source == "invalidation"
                    else prior_clock.accepted_crown_block
                ),
                settlement_block=settlement_block,
            )
        except FiniteDebtError as exc:
            if str(exc) in {
                "improvement rounds to zero 1%-log-unit ppm",
                "claim principal rounds to zero",
            }:
                reason = (
                    "principal_rounds_to_zero"
                    if str(exc) == "claim principal rounds to zero"
                    else "improvement_rounds_to_zero"
                )
                event_digest = self._append_event(
                    "claim_not_issued",
                    block=settlement_block,
                    block_hash=settlement_hash,
                    payload={
                        "accepted_crown_block": accepted,
                        "accepted_crown_block_hash": accepted_hash,
                        "activation_digest": activation.digest,
                        "candidate_digest": candidate,
                        "event_index": index,
                        "event_subindex": subindex,
                        "family_id": family,
                        "hotkey": owner,
                        "reason": reason,
                        "reservation_digest": reservation,
                        "retained_evidence_digest": evidence,
                        "settlement_event_digest": settlement_event,
                    },
                )
                if reason == "principal_rounds_to_zero":
                    units = log_improvement_units_ppm(
                        settled_speedup,
                        basis=activation.policy.improvement_basis,
                        threshold_speedup="1",
                    )
                    if resets_family_clock(activation.policy, units):
                        self.db.execute(
                            "INSERT INTO finite_debt_family_clocks(policy_digest,"
                            "family_id,accepted_crown_block,accepted_crown_block_hash,"
                            "event_index,event_subindex,reservation_digest,source,"
                            "claim_digest,reward_event_digest) "
                            "VALUES(?,?,?,?,?,?,?,'crown_no_debt','',?)",
                            (
                                activation.policy.digest,
                                family,
                                accepted,
                                accepted_hash,
                                index,
                                subindex,
                                reservation,
                                event_digest,
                            ),
                        )
                return None
            raise FiniteDebtStoreError(f"finite-debt claim cannot issue: {exc}") from None
        event_digest = self._append_event(
            "claim_issued",
            block=settlement_block,
            block_hash=settlement_hash,
            payload={
                "accepted_crown_block_hash": accepted_hash,
                "activation_digest": activation.digest,
                "claim": claim.to_dict(),
                "claim_digest": claim.digest,
                "event_index": index,
                "event_subindex": subindex,
                "prior_family_clock": (
                    None
                    if prior_clock is None
                    else {
                        "accepted_crown_block": prior_clock.accepted_crown_block,
                        "event_index": prior_clock.event_index,
                        "event_subindex": prior_clock.event_subindex,
                        "reservation_digest": prior_clock.reservation_digest,
                    }
                ),
                "reservation_digest": reservation,
                "settlement_event_digest": settlement_event,
            },
        )
        self.db.execute(
            "INSERT INTO finite_debt_claims(claim_digest,policy_digest,family_id,"
            "candidate_digest,retained_evidence_digest,hotkey,accepted_crown_block,"
            "accepted_crown_block_hash,event_index,event_subindex,reservation_digest,"
            "settlement_block,settlement_block_hash,settlement_event_digest,"
            "principal_units,claim_json,issuance_reward_event_digest) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                claim.digest,
                claim.policy_digest,
                claim.family_id,
                claim.candidate_digest,
                claim.retained_evidence_digest,
                claim.hotkey,
                accepted,
                accepted_hash,
                index,
                subindex,
                reservation,
                settlement_block,
                settlement_hash,
                settlement_event,
                claim.principal_units,
                _canonical_json(claim.to_dict()),
                event_digest,
            ),
        )
        self._insert_balance(
            DebtClaimBalance.open(claim),
            revision=0,
            reward_event_digest=event_digest,
        )
        if claim.resets_clock:
            self.db.execute(
                "INSERT INTO finite_debt_family_clocks(policy_digest,family_id,"
                "accepted_crown_block,accepted_crown_block_hash,event_index,"
                "event_subindex,reservation_digest,source,claim_digest,"
                "reward_event_digest) VALUES(?,?,?,?,?,?,?,'crown',?,?)",
                (
                    claim.policy_digest,
                    claim.family_id,
                    accepted,
                    accepted_hash,
                    index,
                    subindex,
                    reservation,
                    claim.digest,
                    event_digest,
                ),
            )
        return claim

    @staticmethod
    def _eligible_hotkeys(values: Iterable[str]) -> frozenset[str]:
        try:
            rows = tuple(values)
        except TypeError:
            raise FiniteDebtStoreError("eligible hotkeys are not iterable") from None
        if any(_hotkey(row, "eligible hotkey") != row for row in rows):
            raise FiniteDebtStoreError("eligible hotkeys are malformed")
        return frozenset(rows)

    def reconcile_lifecycle(
        self,
        *,
        current_block: int,
        current_block_hash: str,
        eligible_hotkeys: Iterable[str],
    ) -> tuple[DebtClaimState, ...]:
        """Forfeit expired or departed open debt under exact finalized authority."""

        height = _integer(current_block, "lifecycle block")
        authority_hash = _block_hash(current_block_hash, "lifecycle block hash")
        eligible = self._eligible_hotkeys(eligible_hotkeys)
        changed: list[DebtClaimState] = []
        with self._transaction():
            self._require_finalized_authority(height, authority_hash)
            for state in self._claim_states():
                if state.balance.status != "open":
                    continue
                if height >= state.claim.expires_block:
                    updated = expire_claim_balance(
                        state.claim,
                        state.balance,
                        at_block=height,
                    )
                    event_type = "claim_expired"
                elif state.claim.hotkey not in eligible:
                    try:
                        updated = cancel_claim_balance(
                            state.claim,
                            state.balance,
                            at_block=height,
                            reason="hotkey_departed",
                        )
                    except FiniteDebtError as exc:
                        raise FiniteDebtStoreError(
                            f"finite-debt departure cannot reconcile: {exc}"
                        ) from None
                    event_type = "claim_cancelled"
                else:
                    continue
                if updated == state.balance:
                    continue
                event_digest = self._append_event(
                    event_type,
                    block=height,
                    block_hash=authority_hash,
                    payload={
                        "after_balance": updated.to_dict(),
                        "after_balance_digest": updated.digest,
                        "before_balance_digest": state.balance.digest,
                        "claim_digest": state.claim.digest,
                    },
                )
                revision = self.db.execute(
                    "SELECT MAX(revision) AS value FROM finite_debt_claim_balances "
                    "WHERE claim_digest=?",
                    (state.claim.digest,),
                ).fetchone()["value"] + 1
                self._insert_balance(
                    updated,
                    revision=revision,
                    reward_event_digest=event_digest,
                )
                changed.append(DebtClaimState(state.claim, updated))
        return tuple(changed)

    @staticmethod
    def _epoch_coordinates(
        activation: FiniteDebtPolicyActivation,
        effective_block: int,
    ) -> tuple[int, int]:
        block = _integer(effective_block, "effective_block")
        elapsed = block - activation.activation_block
        if elapsed <= 0 or elapsed % activation.policy.epoch_blocks:
            raise FiniteDebtStoreError(
                "finite-debt epoch is not an exact activated-policy cadence boundary"
            )
        return elapsed // activation.policy.epoch_blocks, block - activation.policy.epoch_blocks

    def _epoch_from_row(self, row: sqlite3.Row) -> FiniteDebtRewardEpoch:
        try:
            epoch = FiniteDebtRewardEpoch.from_dict(json.loads(row["epoch_json"]))
        except (FiniteDebtError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FiniteDebtStoreError(f"finite-debt reward epoch is corrupt: {exc}") from None
        if (
            epoch.digest != row["epoch_digest"]
            or epoch.chain_scope_digest != self.chain_scope_digest
            or epoch.activation_digest != row["activation_digest"]
            or epoch.policy_digest != row["policy_digest"]
            or epoch.epoch_index != row["epoch_index"]
            or epoch.start_block != row["start_block"]
            or epoch.effective_block != row["effective_block"]
            or epoch.effective_block_hash != row["effective_block_hash"]
            or epoch.projection.digest != row["projection_digest"]
            or _canonical_json(epoch.projection.to_dict()) != row["projection_json"]
            or epoch.publication_record_digest != row["publication_record_digest"]
            or epoch.payout_event_digest != row["payout_event_digest"]
            or epoch.projection.payout_units != row["payout_units"]
            or epoch.projection.reserve_units != row["reserve_units"]
            or _canonical_json(epoch.to_dict()) != row["epoch_json"]
        ):
            raise FiniteDebtStoreError("finite-debt reward epoch differs from retained bytes")
        allocations = tuple(
            self.db.execute(
                "SELECT claim_digest,hotkey,units FROM finite_debt_epoch_allocations "
                "WHERE epoch_digest=? ORDER BY claim_digest",
                (epoch.digest,),
            )
        )
        if tuple(
            (item["claim_digest"], item["hotkey"], item["units"])
            for item in allocations
        ) != tuple(
            (item.claim_digest, item.hotkey, item.units)
            for item in epoch.projection.allocations
        ):
            raise FiniteDebtStoreError("finite-debt epoch allocations differ")
        event = self.db.execute(
            "SELECT event_type,block,block_hash,payload_json FROM "
            "finite_debt_reward_events WHERE event_digest=?",
            (epoch.payout_event_digest,),
        ).fetchone()
        if (
            event is None
            or event["event_type"] != "epoch_paid"
            or event["block"] != epoch.effective_block
            or event["block_hash"] != epoch.effective_block_hash
        ):
            raise FiniteDebtStoreError("finite-debt epoch payout event differs")
        try:
            payload = json.loads(event["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FiniteDebtStoreError(
                f"finite-debt epoch payout event is corrupt: {exc}"
            ) from None
        input_digests = set(epoch.projection.input_state_digests)
        prior_by_claim: dict[str, tuple[DebtClaimState, int]] = {}
        matched_inputs: set[str] = set()
        for claim_row in self.db.execute(
            "SELECT * FROM finite_debt_claims WHERE policy_digest=? "
            "ORDER BY claim_digest",
            (epoch.policy_digest,),
        ):
            claim = self._claim_from_row(claim_row)
            for balance_row in self.db.execute(
                "SELECT * FROM finite_debt_claim_balances WHERE claim_digest=? "
                "ORDER BY revision",
                (claim.digest,),
            ):
                balance = self._balance_from_row(balance_row)
                state = DebtClaimState(claim, balance)
                if state.digest not in input_digests:
                    continue
                if state.digest in matched_inputs or claim.digest in prior_by_claim:
                    raise FiniteDebtStoreError(
                        "finite-debt epoch input state is ambiguous"
                    )
                matched_inputs.add(state.digest)
                prior_by_claim[claim.digest] = (state, balance_row["revision"])
        if matched_inputs != input_digests:
            raise FiniteDebtStoreError(
                "finite-debt epoch input states cannot be reopened"
            )
        try:
            replayed = apply_debt_epoch_projection(
                tuple(row[0] for row in prior_by_claim.values()),
                epoch.projection,
            )
        except FiniteDebtError as exc:
            raise FiniteDebtStoreError(
                f"finite-debt epoch transition cannot replay: {exc}"
            ) from None
        after_by_claim = {state.claim.digest: state for state in replayed}
        expected_transitions = [
            {
                "after_balance_digest": after_by_claim[digest].balance.digest,
                "before_balance_digest": prior_by_claim[digest][0].balance.digest,
                "claim_digest": digest,
            }
            for digest in sorted(prior_by_claim)
            if prior_by_claim[digest][0].balance
            != after_by_claim[digest].balance
        ]
        transition_rows = tuple(
            self.db.execute(
                "SELECT b.*,c.policy_digest AS claim_policy_digest FROM "
                "finite_debt_claim_balances AS b JOIN finite_debt_claims AS c "
                "USING(claim_digest) WHERE b.reward_event_digest=? "
                "ORDER BY b.claim_digest",
                (epoch.payout_event_digest,),
            )
        )
        if len(transition_rows) != len(expected_transitions):
            raise FiniteDebtStoreError(
                "finite-debt epoch balance revision set differs"
            )
        for balance_row, transition in zip(
            transition_rows, expected_transitions, strict=True
        ):
            digest = transition["claim_digest"]
            prior = prior_by_claim.get(digest)
            if (
                prior is None
                or balance_row["claim_digest"] != digest
                or balance_row["claim_policy_digest"] != epoch.policy_digest
                or balance_row["revision"] != prior[1] + 1
                or self._balance_from_row(balance_row)
                != after_by_claim[digest].balance
            ):
                raise FiniteDebtStoreError(
                    "finite-debt epoch balance revision differs"
                )
        expected_payload = {
            "activation_digest": epoch.activation_digest,
            "balance_transitions": expected_transitions,
            "epoch_index": epoch.epoch_index,
            "projection_digest": epoch.projection.digest,
            "publication_record_digest": epoch.publication_record_digest,
        }
        if type(payload) is not dict or payload != expected_payload:
            raise FiniteDebtStoreError("finite-debt epoch payout payload differs")
        activation = self._activation_by_policy(epoch.policy_digest)
        expected_index, expected_start = self._epoch_coordinates(
            activation,
            epoch.effective_block,
        )
        if (
            activation.digest != epoch.activation_digest
            or expected_index != epoch.epoch_index
            or expected_start != epoch.start_block
        ):
            raise FiniteDebtStoreError("finite-debt reward epoch cadence differs")
        return epoch

    def reward_epochs(self) -> tuple[FiniteDebtRewardEpoch, ...]:
        self.reward_events()
        return tuple(
            self._epoch_from_row(row)
            for row in self.db.execute(
                "SELECT * FROM finite_debt_reward_epochs "
                "ORDER BY effective_block,epoch_digest"
            )
        )

    @staticmethod
    def _require_projection_eligibility(
        projection: DebtEpochProjection,
        eligible: frozenset[str],
    ) -> None:
        if projection.reserve_hotkey not in eligible:
            raise FiniteDebtStoreError(
                "finite-debt reserve hotkey is absent from eligibility authority"
            )
        positive = {row.hotkey for row in projection.weights if row.units > 0}
        missing = positive - eligible
        if missing:
            raise FiniteDebtStoreError(
                "finite-debt projection contains an ineligible positive miner"
            )

    def _project_epoch(
        self,
        *,
        effective_block: int,
        eligible: frozenset[str],
        allow_retained: bool,
    ) -> DebtEpochProjection:
        activation = self.active_policy_activation(at_block=effective_block)
        if activation is None:
            raise FiniteDebtStoreError("finite-debt policy is not active")
        epoch_index, _start = self._epoch_coordinates(activation, effective_block)
        existing = self.db.execute(
            "SELECT * FROM finite_debt_reward_epochs WHERE policy_digest=? "
            "AND effective_block=?",
            (activation.policy.digest, effective_block),
        ).fetchone()
        if existing is not None:
            if not allow_retained:
                raise FiniteDebtStoreError("finite-debt epoch is already closed")
            epoch = self._epoch_from_row(existing)
            self._require_projection_eligibility(epoch.projection, eligible)
            return epoch.projection
        if self.composition_active_at(effective_block):
            raise FiniteDebtStoreError(
                "core-only projection is disabled after incentive composition activation"
            )
        latest = self.db.execute(
            "SELECT MAX(epoch_index) AS value FROM finite_debt_reward_epochs "
            "WHERE policy_digest=?",
            (activation.policy.digest,),
        ).fetchone()["value"]
        expected_prior = epoch_index - 1
        if (0 if latest is None else latest) != expected_prior:
            raise FiniteDebtStoreError(
                "finite-debt epochs must be projected and closed without gaps"
            )
        states = self._claim_states(policy_digest=activation.policy.digest)
        try:
            projection = project_debt_epoch(
                activation.policy,
                effective_block=effective_block,
                states=states,
            )
        except FiniteDebtError as exc:
            raise FiniteDebtStoreError(
                f"finite-debt epoch cannot project: {exc}"
            ) from None
        self._require_projection_eligibility(projection, eligible)
        return projection

    def project_epoch(
        self,
        *,
        effective_block: int,
        eligible_hotkeys: Iterable[str],
    ) -> DebtEpochProjection:
        """Build a read-only next-epoch projection; balances are never debited."""

        eligible = self._eligible_hotkeys(eligible_hotkeys)
        return self._project_epoch(
            effective_block=effective_block,
            eligible=eligible,
            allow_retained=True,
        )

    def close_confirmed_epoch(
        self,
        projection: DebtEpochProjection,
        *,
        confirmation,
        eligible_hotkeys: Iterable[str],
    ) -> FiniteDebtRewardEpoch:
        """Apply one journal-bound publication once, or reopen its closure."""

        from optima.chain.debt_publication import (
            PUBLICATION_KIND_CORE,
            ConfirmedDebtWeightPublication,
            DebtPublicationError,
            SQLiteDebtWeightPublicationJournal,
            reopen_confirmed_debt_publication,
            retain_confirmed_debt_publication,
        )

        if type(projection) is not DebtEpochProjection:
            raise FiniteDebtStoreError("finite-debt projection is not exactly typed")
        if type(confirmation) is not ConfirmedDebtWeightPublication:
            raise FiniteDebtStoreError(
                "finite-debt close requires a typed publication confirmation"
            )
        expected = projection.digest
        publication = confirmation.digest
        height = projection.effective_block
        authority_hash = confirmation.effective_block_hash
        eligible = self._eligible_hotkeys(eligible_hotkeys)

        with self._transaction():
            if self.composition_active_at(height):
                raise FiniteDebtStoreError(
                    "core-only close is disabled after incentive composition activation"
                )
            try:
                journal = SQLiteDebtWeightPublicationJournal.reopen_from_head(
                    self.db, transaction=self._transaction
                )
                retained_record, binding = journal.retained_authority(
                    confirmation.publication_record.digest
                )
                confirmation.validate_binding(binding)
            except DebtPublicationError as exc:
                raise FiniteDebtStoreError(
                    f"finite-debt publication authority failed: {exc}"
                ) from None
            if (
                retained_record != confirmation.publication_record
                or binding.publication_kind != PUBLICATION_KIND_CORE
                or binding.weight_projection.chain_scope_digest
                != self.chain_scope_digest
                or binding.economic_projection.to_dict() != projection.to_dict()
            ):
                raise FiniteDebtStoreError(
                    "finite-debt publication differs from retained projection authority"
                )
            retained = self.db.execute(
                "SELECT * FROM finite_debt_reward_epochs WHERE policy_digest=? "
                "AND effective_block=?",
                (projection.policy_digest, height),
            ).fetchone()
            if retained is not None:
                epoch = self._epoch_from_row(retained)
                if (
                    epoch.projection.to_dict() != projection.to_dict()
                    or epoch.projection.digest != expected
                    or epoch.effective_block_hash != authority_hash
                    or epoch.publication_record_digest != publication
                ):
                    raise FiniteDebtStoreError(
                        "confirmed finite-debt epoch retry differs from retained closure"
                    )
                self._require_projection_eligibility(epoch.projection, eligible)
                return epoch

            activation = self.active_policy_activation(at_block=height)
            if activation is None or activation.policy.digest != projection.policy_digest:
                raise FiniteDebtStoreError(
                    "finite-debt projection policy is not active at its boundary"
                )
            if binding.activation_digest != activation.digest:
                raise FiniteDebtStoreError(
                    "finite-debt publication differs from active policy authority"
                )
            cursor = self._finalized_cursor()
            if (
                cursor is None
                or cursor[0] < confirmation.readback.block
                or (
                    cursor[0] == confirmation.readback.block
                    and cursor[1] != confirmation.readback.block_hash
                )
            ):
                raise FiniteDebtStoreError(
                    "finite-debt confirmation is newer than retained finalized intake"
                )
            prior_row = self.db.execute(
                "SELECT * FROM finite_debt_reward_epochs WHERE policy_digest=? "
                "AND effective_block<? ORDER BY effective_block DESC LIMIT 1",
                (projection.policy_digest, height),
            ).fetchone()
            if prior_row is not None:
                prior_epoch = self._epoch_from_row(prior_row)
                try:
                    prior_confirmation = reopen_confirmed_debt_publication(
                        self.db, prior_epoch.publication_record_digest
                    )
                except DebtPublicationError as exc:
                    raise FiniteDebtStoreError(
                        f"prior finite-debt publication cannot reopen: {exc}"
                    ) from None
                if (
                    confirmation.readback.block
                    < prior_confirmation.readback.block
                    + activation.policy.epoch_blocks
                ):
                    raise FiniteDebtStoreError(
                        "finite-debt catch-up would compress live emission epochs"
                    )
            epoch_index, start_block = self._epoch_coordinates(activation, height)
            authoritative = self._project_epoch(
                effective_block=height,
                eligible=eligible,
                allow_retained=False,
            )
            if authoritative.to_dict() != projection.to_dict():
                raise FiniteDebtStoreError(
                    "finite-debt balances changed after projection was built"
                )
            try:
                retain_confirmed_debt_publication(self.db, confirmation)
            except DebtPublicationError as exc:
                raise FiniteDebtStoreError(
                    f"finite-debt confirmation cannot be retained: {exc}"
                ) from None
            states = self._claim_states(policy_digest=activation.policy.digest)
            try:
                updated = apply_debt_epoch_projection(states, authoritative)
            except FiniteDebtError as exc:
                raise FiniteDebtStoreError(
                    f"finite-debt epoch cannot apply: {exc}"
                ) from None
            before = {row.claim.digest: row for row in states}
            after = {row.claim.digest: row for row in updated}
            event_digest = self._append_event(
                "epoch_paid",
                block=height,
                block_hash=authority_hash,
                payload={
                    "activation_digest": activation.digest,
                    "balance_transitions": [
                        {
                            "after_balance_digest": after[digest].balance.digest,
                            "before_balance_digest": before[digest].balance.digest,
                            "claim_digest": digest,
                        }
                        for digest in sorted(before)
                        if before[digest].balance != after[digest].balance
                    ],
                    "epoch_index": epoch_index,
                    "projection_digest": authoritative.digest,
                    "publication_record_digest": publication,
                },
            )
            epoch = FiniteDebtRewardEpoch(
                self.chain_scope_digest,
                activation.digest,
                activation.policy.digest,
                epoch_index,
                start_block,
                height,
                authority_hash,
                authoritative,
                publication,
                event_digest,
            )
            self.db.execute(
                "INSERT INTO finite_debt_reward_epochs(epoch_digest,chain_scope_digest,"
                "activation_digest,policy_digest,epoch_index,start_block,effective_block,"
                "effective_block_hash,projection_digest,projection_json,"
                "publication_record_digest,payout_event_digest,payout_units,reserve_units,"
                "epoch_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    epoch.digest,
                    self.chain_scope_digest,
                    activation.digest,
                    activation.policy.digest,
                    epoch_index,
                    start_block,
                    height,
                    authority_hash,
                    authoritative.digest,
                    _canonical_json(authoritative.to_dict()),
                    publication,
                    event_digest,
                    authoritative.payout_units,
                    authoritative.reserve_units,
                    _canonical_json(epoch.to_dict()),
                ),
            )
            for allocation in authoritative.allocations:
                self.db.execute(
                    "INSERT INTO finite_debt_epoch_allocations(epoch_digest,claim_digest,"
                    "hotkey,units) VALUES(?,?,?,?)",
                    (
                        epoch.digest,
                        allocation.claim_digest,
                        allocation.hotkey,
                        allocation.units,
                    ),
                )
            for digest in sorted(before):
                if before[digest].balance == after[digest].balance:
                    continue
                revision = self.db.execute(
                    "SELECT MAX(revision) AS value FROM finite_debt_claim_balances "
                    "WHERE claim_digest=?",
                    (digest,),
                ).fetchone()["value"] + 1
                self._insert_balance(
                    after[digest].balance,
                    revision=revision,
                    reward_event_digest=event_digest,
                )
        return epoch


__all__ = [
    "FiniteDebtFamilyClock",
    "FiniteDebtPolicyActivation",
    "FiniteDebtRewardEpoch",
    "FiniteDebtStore",
    "FiniteDebtStoreError",
    "SCHEMA_VERSION",
    "SeededFamilyClock",
    "migrate_schema3_to4",
    "reward_family_id",
]
