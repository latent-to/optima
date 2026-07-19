"""Typed retained publication authority for finite-debt reward epochs.

The legacy publisher retains a :class:`WeightPublicationRecord`, but that record
alone names only a projection digest.  Debt may be debited only after reopening
the complete V2 projection/readback binding below: exact policy, boundary,
weights, finalized chain readback, and the confirmed publication journal row.

This module deliberately owns no wallet or chain client.  A publisher constructs
the typed confirmation only after its independently verified finalized readback;
the reward stores retain and reopen these bytes before consuming debt.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Iterable

from optima._strict import require_digest, require_exact_fields, require_int
from optima.chain.weights import (
    WeightProjection,
    WeightPublicationError,
    WeightPublicationRecord,
)
from optima.finite_debt import PPM, DebtEpochProjection, DebtHotkeyWeight
from optima.incentive_composition import ComposedEpochProjection
from optima.stack_identity import canonical_digest


PUBLICATION_KIND_CORE = "finite_debt"
PUBLICATION_KIND_COMPOSED = "incentive_composition"
PUBLICATION_KINDS = frozenset(
    {PUBLICATION_KIND_CORE, PUBLICATION_KIND_COMPOSED}
)
CONFIRMATION_SCHEMA_VERSION = 1
CONFIRMATION_VERSION = "optima.debt-weight-publication.v1"
ACTIVE_INTAKE_SCHEMA_VERSION = "6"

_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_TABLE = "debt_weight_publication_confirmations"
_JOURNAL_TABLE = "debt_weight_publication_journal"
_COLUMNS = {
    "record_digest",
    "chain_scope_digest",
    "publication_kind",
    "policy_digest",
    "projection_digest",
    "weight_projection_digest",
    "effective_block",
    "effective_block_hash",
    "confirmed_block",
    "confirmed_block_hash",
    "record_json",
}
_JOURNAL_COLUMNS = {
    "sequence",
    "record_digest",
    "prior_record_digest",
    "binding_digest",
    "weight_projection_digest",
    "record_json",
    "binding_json",
}


class DebtPublicationError(RuntimeError):
    """A debt publication confirmation or boundary schedule is invalid."""


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=DebtPublicationError)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    return require_int(
        value,
        field=field,
        error=DebtPublicationError,
        minimum=minimum,
    )


def _block_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _BLOCK_HASH.fullmatch(value) is None:
        raise DebtPublicationError(f"{field} is malformed")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _strict(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    return dict(
        require_exact_fields(
            value,
            fields=fields,
            label=label,
            error=DebtPublicationError,
            exact_dict=True,
        )
    )


def _weights(values: Iterable[DebtHotkeyWeight]) -> tuple[DebtHotkeyWeight, ...]:
    rows = tuple(values)
    if (
        any(type(row) is not DebtHotkeyWeight for row in rows)
        or tuple(row.hotkey for row in rows)
        != tuple(sorted({row.hotkey for row in rows}))
        or sum(row.units for row in rows) != PPM
    ):
        raise DebtPublicationError(
            "debt publication weights must be unique, canonical, and conserve PPM"
        )
    return rows


@dataclass(frozen=True)
class DebtWeightReadback:
    """Exact finalized on-chain readback attributed to one publication."""

    chain_scope_digest: str
    netuid: int
    validator_hotkey: str
    block: int
    block_hash: str
    last_update_block: int
    weights: tuple[DebtHotkeyWeight, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chain_scope_digest",
            _digest(self.chain_scope_digest, "chain_scope_digest"),
        )
        _integer(self.netuid, "netuid")
        if (
            not isinstance(self.validator_hotkey, str)
            or not self.validator_hotkey
            or len(self.validator_hotkey) > 256
            or any(char.isspace() for char in self.validator_hotkey)
        ):
            raise DebtPublicationError("validator_hotkey is malformed")
        block = _integer(self.block, "readback block")
        object.__setattr__(
            self,
            "block_hash",
            _block_hash(self.block_hash, "readback block_hash"),
        )
        updated = _integer(self.last_update_block, "last_update_block")
        if updated > block:
            raise DebtPublicationError("weight readback update is after its block")
        object.__setattr__(self, "weights", _weights(self.weights))

    def to_dict(self) -> dict[str, object]:
        return {
            "block": self.block,
            "block_hash": self.block_hash,
            "chain_scope_digest": self.chain_scope_digest,
            "last_update_block": self.last_update_block,
            "netuid": self.netuid,
            "validator_hotkey": self.validator_hotkey,
            "weights": [row.to_dict() for row in self.weights],
        }

    @classmethod
    def from_dict(cls, value: object) -> "DebtWeightReadback":
        row = _strict(
            value,
            frozenset(cls.__dataclass_fields__),
            "debt weight readback",
        )
        raw_weights = row["weights"]
        if type(raw_weights) is not list:
            raise DebtPublicationError("debt weight readback weights must be an array")
        try:
            row["weights"] = tuple(
                DebtHotkeyWeight.from_dict(item) for item in raw_weights
            )
        except (TypeError, ValueError) as exc:
            raise DebtPublicationError(f"debt weight readback is invalid: {exc}") from None
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.debt-weight-readback", self.to_dict())


@dataclass(frozen=True)
class ConfirmedDebtWeightPublication:
    """One complete, immutable proof that a V2 projection reached chain state."""

    chain_scope_digest: str
    publication_kind: str
    policy_digest: str
    projection_digest: str
    weight_projection_digest: str
    effective_block: int
    effective_block_hash: str
    weights: tuple[DebtHotkeyWeight, ...]
    readback: DebtWeightReadback
    publication_record: WeightPublicationRecord
    schema_version: int = CONFIRMATION_SCHEMA_VERSION
    confirmation_version: str = CONFIRMATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chain_scope_digest",
            _digest(self.chain_scope_digest, "chain_scope_digest"),
        )
        if self.publication_kind not in PUBLICATION_KINDS:
            raise DebtPublicationError("debt publication kind is unsupported")
        for field in (
            "policy_digest",
            "projection_digest",
            "weight_projection_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        effective = _integer(self.effective_block, "effective_block")
        object.__setattr__(
            self,
            "effective_block_hash",
            _block_hash(self.effective_block_hash, "effective_block_hash"),
        )
        weights = _weights(self.weights)
        object.__setattr__(self, "weights", weights)
        if type(self.readback) is not DebtWeightReadback:
            raise DebtPublicationError("debt publication readback is not exactly typed")
        if type(self.publication_record) is not WeightPublicationRecord:
            raise DebtPublicationError(
                "debt publication journal record is not exactly typed"
            )
        record = self.publication_record
        if (
            self.readback.chain_scope_digest != self.chain_scope_digest
            or self.readback.weights != weights
            or self.readback.block < effective
            or record.status != "confirmed"
            or record.projection_digest != self.weight_projection_digest
            or record.confirmed_block != self.readback.block
            or record.confirmed_last_update != self.readback.last_update_block
            or record.confirmed_last_update < record.submit_block
        ):
            raise DebtPublicationError(
                "debt publication does not bind an exact confirmed finalized readback"
            )
        if self.schema_version != CONFIRMATION_SCHEMA_VERSION:
            raise DebtPublicationError("debt publication schema is unsupported")
        if self.confirmation_version != CONFIRMATION_VERSION:
            raise DebtPublicationError("debt publication version is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_scope_digest": self.chain_scope_digest,
            "confirmation_version": self.confirmation_version,
            "effective_block": self.effective_block,
            "effective_block_hash": self.effective_block_hash,
            "policy_digest": self.policy_digest,
            "projection_digest": self.projection_digest,
            "publication_kind": self.publication_kind,
            "publication_record": self.publication_record.to_dict(),
            "readback": self.readback.to_dict(),
            "schema_version": self.schema_version,
            "weights": [row.to_dict() for row in self.weights],
            "weight_projection_digest": self.weight_projection_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConfirmedDebtWeightPublication":
        row = _strict(
            value,
            frozenset(cls.__dataclass_fields__),
            "confirmed debt weight publication",
        )
        raw_weights = row["weights"]
        if type(raw_weights) is not list:
            raise DebtPublicationError("debt publication weights must be an array")
        try:
            row["weights"] = tuple(
                DebtHotkeyWeight.from_dict(item) for item in raw_weights
            )
            row["readback"] = DebtWeightReadback.from_dict(row["readback"])
            row["publication_record"] = WeightPublicationRecord.from_dict(
                row["publication_record"]
            )
        except (TypeError, ValueError, DebtPublicationError) as exc:
            raise DebtPublicationError(f"debt publication is invalid: {exc}") from None
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.debt-weight-publication", self.to_dict())

    @property
    def confirmed_block_hash(self) -> str:
        return self.readback.block_hash

    def validate_projection(
        self,
        *,
        chain_scope_digest: str,
        publication_kind: str,
        policy_digest: str,
        projection_digest: str,
        effective_block: int,
        effective_block_hash: str,
        weights: Iterable[DebtHotkeyWeight],
        weight_projection_digest: str | None = None,
    ) -> None:
        expected_weights = _weights(weights)
        if (
            self.chain_scope_digest != _digest(
                chain_scope_digest, "expected chain_scope_digest"
            )
            or self.publication_kind != publication_kind
            or self.policy_digest != _digest(policy_digest, "expected policy_digest")
            or self.projection_digest
            != _digest(projection_digest, "expected projection_digest")
            or (
                weight_projection_digest is not None
                and self.weight_projection_digest
                != _digest(
                    weight_projection_digest,
                    "expected weight_projection_digest",
                )
            )
            or self.effective_block
            != _integer(effective_block, "expected effective_block")
            or self.effective_block_hash
            != _block_hash(effective_block_hash, "expected effective_block_hash")
            or self.weights != expected_weights
        ):
            raise DebtPublicationError(
                "confirmed debt publication differs from the projected epoch"
            )

    def validate_binding(self, binding: "DebtWeightPublicationBinding") -> None:
        """Require the retained signer and economic bytes that produced this proof."""

        if type(binding) is not DebtWeightPublicationBinding:
            raise DebtPublicationError(
                "debt publication binding is not exactly typed"
            )
        self.validate_projection(
            chain_scope_digest=binding.weight_projection.chain_scope_digest,
            publication_kind=binding.publication_kind,
            policy_digest=binding.policy_digest,
            projection_digest=binding.economic_projection_digest,
            effective_block=binding.economic_projection.effective_block,
            effective_block_hash=binding.effective_block_hash,
            weights=binding.weights,
            weight_projection_digest=binding.weight_projection.digest,
        )
        if (
            self.readback.netuid != binding.weight_projection.netuid
            or self.readback.validator_hotkey
            != binding.weight_projection.validator_hotkey
            or self.publication_record.projection_digest
            != binding.weight_projection.digest
        ):
            raise DebtPublicationError(
                "confirmed debt publication differs from signer authority"
            )


def build_confirmed_debt_weight_publication(
    binding: "DebtWeightPublicationBinding",
    publication_record: WeightPublicationRecord,
    *,
    confirmed_metagraph: object,
    confirmed_snapshot: object,
) -> ConfirmedDebtWeightPublication:
    """Construct confirmation only from one exact finalized sparse-row readback."""

    if type(binding) is not DebtWeightPublicationBinding:
        raise DebtPublicationError("debt publication binding is not exactly typed")
    if type(publication_record) is not WeightPublicationRecord:
        raise DebtPublicationError("debt publication record is not exactly typed")
    try:
        block = getattr(confirmed_metagraph, "block")
        block_hash = str(getattr(confirmed_metagraph, "block_hash")).lower()
        snapshot_weights = dict(getattr(confirmed_snapshot, "weights"))
        last_update = getattr(confirmed_snapshot, "last_update_block")
    except Exception as exc:
        raise DebtPublicationError(
            f"confirmed debt publication readback is malformed: {exc}"
        ) from None
    expected = binding.weight_projection.weights
    if (
        type(block) is not int
        or _BLOCK_HASH.fullmatch(block_hash) is None
        or type(last_update) is not int
        or set(snapshot_weights) != set(expected)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), expected[hotkey], rel_tol=2e-5, abs_tol=2e-5
            )
            for hotkey, value in snapshot_weights.items()
        )
    ):
        raise DebtPublicationError(
            "confirmed debt publication sparse row differs from projected weights"
        )
    if (
        publication_record.status != "confirmed"
        or publication_record.projection_digest
        != binding.weight_projection.digest
        or publication_record.confirmed_block != block
        or publication_record.confirmed_last_update != last_update
    ):
        raise DebtPublicationError(
            "confirmed debt publication record differs from finalized readback"
        )
    return ConfirmedDebtWeightPublication(
        binding.weight_projection.chain_scope_digest,
        binding.publication_kind,
        binding.policy_digest,
        binding.economic_projection_digest,
        binding.weight_projection.digest,
        binding.economic_projection.effective_block,
        binding.effective_block_hash,
        binding.weights,
        DebtWeightReadback(
            binding.weight_projection.chain_scope_digest,
            binding.weight_projection.netuid,
            binding.weight_projection.validator_hotkey,
            block,
            block_hash,
            last_update,
            binding.weights,
        ),
        publication_record,
    )


@dataclass(frozen=True)
class DebtWeightPublicationBinding:
    """Economic projection plus the exact signer-facing weight projection."""

    publication_kind: str
    activation_digest: str
    effective_block_hash: str
    economic_projection: DebtEpochProjection | ComposedEpochProjection
    weight_projection: WeightProjection

    def __post_init__(self) -> None:
        if self.publication_kind not in PUBLICATION_KINDS:
            raise DebtPublicationError("debt publication binding kind is unsupported")
        object.__setattr__(
            self,
            "activation_digest",
            _digest(self.activation_digest, "activation_digest"),
        )
        object.__setattr__(
            self,
            "effective_block_hash",
            _block_hash(self.effective_block_hash, "effective_block_hash"),
        )
        expected_type = (
            DebtEpochProjection
            if self.publication_kind == PUBLICATION_KIND_CORE
            else ComposedEpochProjection
        )
        if type(self.economic_projection) is not expected_type:
            raise DebtPublicationError(
                "debt publication binding economic projection has the wrong kind"
            )
        if type(self.weight_projection) is not WeightProjection:
            raise DebtPublicationError(
                "debt publication binding weight projection is not exactly typed"
            )
        economic = self.economic_projection
        policy_digest = (
            economic.policy_digest
            if type(economic) is DebtEpochProjection
            else economic.composition_policy_digest
        )
        economic_weights = tuple(
            (row.hotkey, row.units) for row in economic.weights
        )
        input_digests = (
            economic.input_state_digests
            if type(economic) is DebtEpochProjection
            else tuple(
                sorted(
                    (
                        *economic.discovery_input_state_digests,
                        *economic.innovation_input_state_digests,
                    )
                )
            )
        )
        expected_evaluation = canonical_digest(
            "optima.debt-weight-projection.evaluation",
            {
                "activation_digest": self.activation_digest,
                "input_state_digests": list(input_digests),
                "projection_digest": economic.digest,
            },
        )
        if (
            self.weight_projection.policy_digest != policy_digest
            or self.weight_projection.settlement_state_digest != economic.digest
            or self.weight_projection.evaluation_state_digest != expected_evaluation
            or self.weight_projection.effective_block != economic.effective_block
            or self.weight_projection.arena_state_digests
            != (self.activation_digest,)
            or self.weight_projection.weights_ppm != economic_weights
        ):
            raise DebtPublicationError(
                "signer-facing weights differ from their economic projection"
            )

    @property
    def policy_digest(self) -> str:
        if type(self.economic_projection) is DebtEpochProjection:
            return self.economic_projection.policy_digest
        return self.economic_projection.composition_policy_digest

    @property
    def economic_projection_digest(self) -> str:
        return self.economic_projection.digest

    @property
    def weights(self) -> tuple[DebtHotkeyWeight, ...]:
        return self.economic_projection.weights

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_digest": self.activation_digest,
            "economic_projection": self.economic_projection.to_dict(),
            "effective_block_hash": self.effective_block_hash,
            "publication_kind": self.publication_kind,
            "weight_projection": self.weight_projection.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "DebtWeightPublicationBinding":
        row = _strict(
            value,
            frozenset(cls.__dataclass_fields__),
            "debt weight publication binding",
        )
        kind = row["publication_kind"]
        try:
            row["economic_projection"] = (
                DebtEpochProjection.from_dict(row["economic_projection"])
                if kind == PUBLICATION_KIND_CORE
                else ComposedEpochProjection.from_dict(row["economic_projection"])
                if kind == PUBLICATION_KIND_COMPOSED
                else row["economic_projection"]
            )
            row["weight_projection"] = WeightProjection.from_dict(
                row["weight_projection"]
            )
        except (TypeError, ValueError, WeightPublicationError) as exc:
            raise DebtPublicationError(
                f"debt weight publication binding is invalid: {exc}"
            ) from None
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.debt-weight-publication.binding", self.to_dict())


def build_debt_weight_publication_binding(
    economic_projection: DebtEpochProjection | ComposedEpochProjection,
    *,
    publication_kind: str,
    activation_digest: str,
    chain_scope_digest: str,
    netuid: int,
    validator_hotkey: str,
    boundary_metagraph: object,
    epoch_index: int,
) -> DebtWeightPublicationBinding:
    """Adapt one debt projection to the existing signer without losing authority."""

    if publication_kind == PUBLICATION_KIND_CORE:
        if type(economic_projection) is not DebtEpochProjection:
            raise DebtPublicationError("core publication requires a core projection")
        policy_digest = economic_projection.policy_digest
        input_digests = economic_projection.input_state_digests
        crown_count = len(input_digests)
    elif publication_kind == PUBLICATION_KIND_COMPOSED:
        if type(economic_projection) is not ComposedEpochProjection:
            raise DebtPublicationError(
                "composed publication requires a composed projection"
            )
        policy_digest = economic_projection.composition_policy_digest
        input_digests = tuple(
            sorted(
                (
                    *economic_projection.discovery_input_state_digests,
                    *economic_projection.innovation_input_state_digests,
                )
            )
        )
        crown_count = len(economic_projection.innovation_input_state_digests)
    else:
        raise DebtPublicationError("debt publication binding kind is unsupported")
    try:
        block = getattr(boundary_metagraph, "block")
        block_hash = str(getattr(boundary_metagraph, "block_hash")).lower()
        hotkeys = tuple(getattr(boundary_metagraph, "hotkeys"))
        uids = tuple(getattr(boundary_metagraph, "uids"))
    except Exception as exc:
        raise DebtPublicationError(f"boundary metagraph is malformed: {exc}") from None
    if (
        type(block) is not int
        or block != economic_projection.effective_block
        or _BLOCK_HASH.fullmatch(block_hash) is None
        or len(hotkeys) != len(uids)
        or len(set(hotkeys)) != len(hotkeys)
        or len(set(uids)) != len(uids)
    ):
        raise DebtPublicationError(
            "boundary metagraph differs from the economic projection boundary"
        )
    members = [
        {"hotkey": hotkey, "uid": uid}
        for uid, hotkey in sorted(
            zip(uids, hotkeys, strict=True), key=lambda item: (item[0], item[1])
        )
    ]
    scope = _digest(chain_scope_digest, "chain_scope_digest")
    activation = _digest(activation_digest, "activation_digest")
    projection = WeightProjection(
        scope,
        _integer(netuid, "netuid"),
        validator_hotkey,
        policy_digest,
        economic_projection.digest,
        canonical_digest(
            "optima.debt-weight-projection.evaluation",
            {
                "activation_digest": activation,
                "input_state_digests": list(input_digests),
                "projection_digest": economic_projection.digest,
            },
        ),
        canonical_digest(
            "optima.economics.metagraph-membership",
            {
                "block": block,
                "block_hash": block_hash,
                "chain_scope_digest": scope,
                "members": members,
            },
        ),
        (activation,),
        _integer(epoch_index, "epoch_index", minimum=1),
        block,
        crown_count,
        tuple(sorted(set(input_digests))),
        tuple((row.hotkey, row.units) for row in economic_projection.weights),
    )
    return DebtWeightPublicationBinding(
        publication_kind,
        activation,
        block_hash,
        economic_projection,
        projection,
    )


def ensure_debt_publication_schema(db: sqlite3.Connection) -> None:
    """Create or verify the additive immutable confirmation table."""

    if not isinstance(db, sqlite3.Connection):
        raise DebtPublicationError("debt publication schema requires SQLite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS debt_weight_publication_confirmations("
        "record_digest TEXT PRIMARY KEY,chain_scope_digest TEXT NOT NULL,"
        "publication_kind TEXT NOT NULL,policy_digest TEXT NOT NULL,"
        "projection_digest TEXT NOT NULL UNIQUE,"
        "weight_projection_digest TEXT NOT NULL UNIQUE,"
        "effective_block INTEGER NOT NULL,"
        "effective_block_hash TEXT NOT NULL,confirmed_block INTEGER NOT NULL,"
        "confirmed_block_hash TEXT NOT NULL,record_json TEXT NOT NULL,"
        "UNIQUE(publication_kind,policy_digest,effective_block)) STRICT"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS debt_weight_publication_journal("
        "sequence INTEGER PRIMARY KEY,record_digest TEXT NOT NULL UNIQUE,"
        "prior_record_digest TEXT NOT NULL,binding_digest TEXT NOT NULL,"
        "weight_projection_digest TEXT NOT NULL,record_json TEXT NOT NULL,"
        "binding_json TEXT NOT NULL) STRICT"
    )
    for action in ("UPDATE", "DELETE"):
        db.execute(
            f"CREATE TRIGGER IF NOT EXISTS {_TABLE}_reject_{action.lower()} "
            f"BEFORE {action} ON {_TABLE} BEGIN SELECT "
            "RAISE(ABORT,'debt publication confirmations are immutable'); END"
        )
        db.execute(
            f"CREATE TRIGGER IF NOT EXISTS {_JOURNAL_TABLE}_reject_{action.lower()} "
            f"BEFORE {action} ON {_JOURNAL_TABLE} BEGIN SELECT "
            "RAISE(ABORT,'debt publication journal rows are immutable'); END"
        )
    row = db.execute(
        "SELECT strict FROM pragma_table_list WHERE name=?", (_TABLE,)
    ).fetchone()
    columns = {item["name"] for item in db.execute(f"PRAGMA table_info({_TABLE})")}
    if row is None or row["strict"] != 1 or columns != _COLUMNS:
        raise DebtPublicationError("debt publication confirmation schema differs")
    journal_row = db.execute(
        "SELECT strict FROM pragma_table_list WHERE name=?", (_JOURNAL_TABLE,)
    ).fetchone()
    journal_columns = {
        item["name"] for item in db.execute(f"PRAGMA table_info({_JOURNAL_TABLE})")
    }
    if (
        journal_row is None
        or journal_row["strict"] != 1
        or journal_columns != _JOURNAL_COLUMNS
    ):
        raise DebtPublicationError("debt publication journal schema differs")


def activate_debt_publication_schema(db: sqlite3.Connection) -> None:
    """Raise the intake schema floor atomically when V2 economics activates.

    Pre-activation databases remain schema 5 and can run legacy V1.  Once an
    incentive composition is activated, metadata moves to 6 in that same outer
    transaction.  A schema-5 runtime therefore refuses to reopen the active
    database, preventing a binary rollback from silently restoring V1.
    """

    if not isinstance(db, sqlite3.Connection) or not db.in_transaction:
        raise DebtPublicationError(
            "debt publication schema activation requires the owning transaction"
        )
    ensure_debt_publication_schema(db)
    row = db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
    if row is None:
        raise DebtPublicationError("intake schema metadata is absent")
    if row["value"] == ACTIVE_INTAKE_SCHEMA_VERSION:
        return
    if row["value"] != "5":
        raise DebtPublicationError(
            "active debt publication requires intake schema 5"
        )
    db.execute(
        "UPDATE metadata SET value=? WHERE key='schema' AND value='5'",
        (ACTIVE_INTAKE_SCHEMA_VERSION,),
    )
    if db.execute("SELECT changes() AS value").fetchone()["value"] != 1:
        raise DebtPublicationError(
            "intake schema changed during debt publication activation"
        )


def retain_confirmed_debt_publication(
    db: sqlite3.Connection,
    confirmation: ConfirmedDebtWeightPublication,
) -> ConfirmedDebtWeightPublication:
    """Retain one confirmation inside the owning store transaction."""

    if not isinstance(db, sqlite3.Connection) or not db.in_transaction:
        raise DebtPublicationError(
            "debt publication retention requires the owning SQLite transaction"
        )
    if type(confirmation) is not ConfirmedDebtWeightPublication:
        raise DebtPublicationError("debt publication confirmation is not exactly typed")
    ensure_debt_publication_schema(db)
    exact = db.execute(
        f"SELECT * FROM {_TABLE} WHERE record_digest=?", (confirmation.digest,)
    ).fetchone()
    if exact is not None:
        return _confirmation_from_row(exact)
    encoded = _canonical_json(confirmation.to_dict())
    try:
        db.execute(
            f"INSERT INTO {_TABLE}(record_digest,chain_scope_digest,publication_kind,"
            "policy_digest,projection_digest,weight_projection_digest,effective_block,"
            "effective_block_hash,confirmed_block,confirmed_block_hash,record_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                confirmation.digest,
                confirmation.chain_scope_digest,
                confirmation.publication_kind,
                confirmation.policy_digest,
                confirmation.projection_digest,
                confirmation.weight_projection_digest,
                confirmation.effective_block,
                confirmation.effective_block_hash,
                confirmation.readback.block,
                confirmation.readback.block_hash,
                encoded,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise DebtPublicationError(
            "debt publication boundary or projection already binds other bytes"
        ) from exc
    return confirmation


def _confirmation_from_row(row: sqlite3.Row) -> ConfirmedDebtWeightPublication:
    try:
        confirmation = ConfirmedDebtWeightPublication.from_dict(
            json.loads(row["record_json"])
        )
    except (TypeError, ValueError, json.JSONDecodeError, DebtPublicationError) as exc:
        raise DebtPublicationError(f"debt publication row is corrupt: {exc}") from None
    if (
        confirmation.digest != row["record_digest"]
        or confirmation.chain_scope_digest != row["chain_scope_digest"]
        or confirmation.publication_kind != row["publication_kind"]
        or confirmation.policy_digest != row["policy_digest"]
        or confirmation.projection_digest != row["projection_digest"]
        or confirmation.weight_projection_digest != row["weight_projection_digest"]
        or confirmation.effective_block != row["effective_block"]
        or confirmation.effective_block_hash != row["effective_block_hash"]
        or confirmation.readback.block != row["confirmed_block"]
        or confirmation.readback.block_hash != row["confirmed_block_hash"]
        or _canonical_json(confirmation.to_dict()) != row["record_json"]
    ):
        raise DebtPublicationError(
            "debt publication confirmation differs from retained bytes"
        )
    return confirmation


def reopen_confirmed_debt_publication(
    db: sqlite3.Connection,
    record_digest: str,
) -> ConfirmedDebtWeightPublication:
    """Reopen and fully validate one retained confirmation by digest."""

    if not isinstance(db, sqlite3.Connection):
        raise DebtPublicationError("debt publication reopening requires SQLite")
    ensure_debt_publication_schema(db)
    digest = _digest(record_digest, "publication_record_digest")
    row = db.execute(
        f"SELECT * FROM {_TABLE} WHERE record_digest=?", (digest,)
    ).fetchone()
    if row is None:
        raise DebtPublicationError(
            "confirmed debt publication record is not retained"
        )
    return _confirmation_from_row(row)


class SQLiteDebtWeightPublicationJournal:
    """Restart-safe CAS journal for one current V2 publication binding."""

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        transaction: Callable[[], object],
        binding: DebtWeightPublicationBinding,
        validate_new_binding: (
            Callable[[DebtWeightPublicationBinding], None] | None
        ) = None,
    ):
        if (
            not isinstance(db, sqlite3.Connection)
            or not callable(transaction)
            or type(binding) is not DebtWeightPublicationBinding
            or (
                validate_new_binding is not None
                and not callable(validate_new_binding)
            )
        ):
            raise DebtPublicationError("debt publication journal authority is malformed")
        self.db = db
        self._transaction = transaction
        self.binding = binding
        self._validate_new_binding = validate_new_binding
        ensure_debt_publication_schema(db)
        self.load()

    @staticmethod
    def _binding_from_row(row: sqlite3.Row) -> DebtWeightPublicationBinding:
        try:
            binding = DebtWeightPublicationBinding.from_dict(
                json.loads(row["binding_json"])
            )
        except (TypeError, ValueError, json.JSONDecodeError, DebtPublicationError) as exc:
            raise DebtPublicationError(
                f"debt publication journal binding is corrupt: {exc}"
            ) from None
        if (
            binding.digest != row["binding_digest"]
            or binding.weight_projection.digest != row["weight_projection_digest"]
            or _canonical_json(binding.to_dict()) != row["binding_json"]
        ):
            raise DebtPublicationError(
                "debt publication journal binding differs from retained bytes"
            )
        return binding

    @classmethod
    def reopen_from_head(
        cls,
        db: sqlite3.Connection,
        *,
        transaction: Callable[[], object],
        validate_new_binding: (
            Callable[[DebtWeightPublicationBinding], None] | None
        ) = None,
    ) -> "SQLiteDebtWeightPublicationJournal":
        ensure_debt_publication_schema(db)
        row = db.execute(
            f"SELECT * FROM {_JOURNAL_TABLE} ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise DebtPublicationError("debt publication journal has no retained head")
        return cls(
            db,
            transaction=transaction,
            binding=cls._binding_from_row(row),
            validate_new_binding=validate_new_binding,
        )

    def _records(
        self,
    ) -> tuple[tuple[WeightPublicationRecord, DebtWeightPublicationBinding], ...]:
        result: list[
            tuple[WeightPublicationRecord, DebtWeightPublicationBinding]
        ] = []
        previous: str | None = None
        expected_sequence = 1
        for row in self.db.execute(
            f"SELECT * FROM {_JOURNAL_TABLE} ORDER BY sequence"
        ):
            try:
                record = WeightPublicationRecord.from_dict(
                    json.loads(row["record_json"])
                )
            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
                WeightPublicationError,
            ) as exc:
                raise DebtPublicationError(
                    f"debt publication journal record is corrupt: {exc}"
                ) from None
            binding = self._binding_from_row(row)
            encoded_prior = "" if previous is None else previous
            if (
                row["sequence"] != expected_sequence
                or record.digest != row["record_digest"]
                or row["prior_record_digest"] != encoded_prior
                or record.prior_record_digest != previous
                or record.projection_digest != binding.weight_projection.digest
                or _canonical_json(record.to_dict()) != row["record_json"]
            ):
                raise DebtPublicationError(
                    "debt publication journal chain differs from retained bytes"
                )
            result.append((record, binding))
            previous = record.digest
            expected_sequence += 1
        return tuple(result)

    def load(self) -> WeightPublicationRecord | None:
        records = self._records()
        return None if not records else records[-1][0]

    def head_binding(self) -> DebtWeightPublicationBinding | None:
        records = self._records()
        return None if not records else records[-1][1]

    def retained_authorities(
        self,
    ) -> tuple[tuple[WeightPublicationRecord, DebtWeightPublicationBinding], ...]:
        """Reopen and validate every immutable record/binding pair in order."""

        return self._records()

    def compare_and_swap(
        self,
        expected_record_digest: str | None,
        replacement: WeightPublicationRecord,
    ) -> None:
        if type(replacement) is not WeightPublicationRecord:
            raise DebtPublicationError(
                "debt publication replacement is not exactly typed"
            )
        with self._transaction():
            records = self._records()
            current = None if not records else records[-1][0]
            observed = None if current is None else current.digest
            if observed != expected_record_digest:
                raise DebtPublicationError(
                    "debt publication journal compare-and-swap failed"
                )
            if replacement.prior_record_digest != expected_record_digest:
                raise DebtPublicationError(
                    "debt publication replacement does not bind the CAS head"
                )
            retained_bindings = tuple(
                retained_binding
                for retained_record, retained_binding in records
                if retained_record.projection_digest
                == replacement.projection_digest
            )
            if retained_bindings:
                binding = retained_bindings[-1]
                if any(
                    retained.to_dict() != binding.to_dict()
                    for retained in retained_bindings
                ):
                    raise DebtPublicationError(
                        "one weight projection has conflicting retained bindings"
                    )
            elif (
                replacement.projection_digest
                == self.binding.weight_projection.digest
            ):
                binding = self.binding
                if self._validate_new_binding is None:
                    raise DebtPublicationError(
                        "new debt publication binding lacks authoritative "
                        "state validation"
                    )
                # This callback must reopen the economic projection from the
                # owning store.  It runs after BEGIN IMMEDIATE and before the
                # first immutable row, closing the projection-to-intent race.
                self._validate_new_binding(binding)
            else:
                binding = None
            if binding is None:
                raise DebtPublicationError(
                    "publication record has no retained debt projection binding"
                )
            self.db.execute(
                f"INSERT INTO {_JOURNAL_TABLE}(sequence,record_digest,"
                "prior_record_digest,binding_digest,weight_projection_digest,"
                "record_json,binding_json) VALUES(?,?,?,?,?,?,?)",
                (
                    len(records) + 1,
                    replacement.digest,
                    "" if replacement.prior_record_digest is None
                    else replacement.prior_record_digest,
                    binding.digest,
                    binding.weight_projection.digest,
                    _canonical_json(replacement.to_dict()),
                    _canonical_json(binding.to_dict()),
                ),
            )

    def retained_projection(self, projection_digest: str) -> WeightProjection:
        digest = _digest(projection_digest, "weight projection digest")
        for record, binding in reversed(self._records()):
            if record.projection_digest == digest:
                return binding.weight_projection
        raise DebtPublicationError("debt weight projection is not retained")

    def retained_authority(
        self, record_digest: str
    ) -> tuple[WeightPublicationRecord, DebtWeightPublicationBinding]:
        """Reopen one exact record and its economic binding from the full chain."""

        digest = _digest(record_digest, "publication record digest")
        for record, binding in self._records():
            if record.digest == digest:
                return record, binding
        raise DebtPublicationError("debt publication record is not retained")


@dataclass(frozen=True)
class DebtBoundarySchedule:
    """Deterministic next gapless boundary at one finalized head."""

    policy_digest: str
    activation_block: int
    epoch_blocks: int
    finalized_block: int
    next_epoch_index: int
    next_effective_block: int
    not_before_block: int
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_digest", _digest(self.policy_digest, "policy_digest")
        )
        activation = _integer(self.activation_block, "activation_block")
        cadence = _integer(self.epoch_blocks, "epoch_blocks", minimum=1)
        finalized = _integer(self.finalized_block, "finalized_block")
        index = _integer(self.next_epoch_index, "next_epoch_index", minimum=1)
        effective = _integer(self.next_effective_block, "next_effective_block")
        not_before = _integer(self.not_before_block, "not_before_block")
        if effective != activation + index * cadence:
            raise DebtPublicationError("debt boundary schedule cadence differs")
        if not_before < effective:
            raise DebtPublicationError("debt boundary schedule rate limit differs")
        expected = (
            "not_due"
            if finalized < not_before
            else "ready"
            if finalized == effective and not_before == effective
            else "catch_up_required"
        )
        if self.status != expected:
            raise DebtPublicationError("debt boundary schedule status differs")


def next_debt_boundary_schedule(
    *,
    policy_digest: str,
    activation_block: int,
    epoch_blocks: int,
    closed_effective_blocks: Iterable[int],
    finalized_block: int,
    previous_confirmed_block: int | None = None,
) -> DebtBoundarySchedule:
    """Return the earliest unclosed boundary without compressing payout epochs.

    A delayed publication leaves the nominal effective-block sequence unchanged,
    but the next vector cannot be consumed until one full cadence after the prior
    confirmation.  This permits deterministic catch-up without rapidly debiting
    several historical epochs against one interval of live emissions.
    """

    activation = _integer(activation_block, "activation_block")
    cadence = _integer(epoch_blocks, "epoch_blocks", minimum=1)
    closed = tuple(closed_effective_blocks)
    if any(type(value) is not int or value < 0 for value in closed):
        raise DebtPublicationError("closed debt boundaries are malformed")
    expected = tuple(
        activation + index * cadence for index in range(1, len(closed) + 1)
    )
    if closed != expected:
        raise DebtPublicationError(
            "closed debt boundaries are not an exact gapless prefix"
        )
    index = len(closed) + 1
    effective = activation + index * cadence
    finalized = _integer(finalized_block, "finalized_block")
    if previous_confirmed_block is None:
        if closed:
            raise DebtPublicationError(
                "closed debt boundaries require prior confirmation authority"
            )
        prior_confirmed = None
    else:
        prior_confirmed = _integer(
            previous_confirmed_block, "previous_confirmed_block"
        )
        if not closed:
            raise DebtPublicationError(
                "prior confirmation authority has no closed debt boundary"
            )
    not_before = max(
        effective,
        effective if prior_confirmed is None else prior_confirmed + cadence,
    )
    status = (
        "not_due"
        if finalized < not_before
        else "ready"
        if finalized == effective and not_before == effective
        else "catch_up_required"
    )
    return DebtBoundarySchedule(
        _digest(policy_digest, "policy_digest"),
        activation,
        cadence,
        finalized,
        index,
        effective,
        not_before,
        status,
    )


__all__ = [
    "ACTIVE_INTAKE_SCHEMA_VERSION",
    "CONFIRMATION_SCHEMA_VERSION",
    "CONFIRMATION_VERSION",
    "ConfirmedDebtWeightPublication",
    "DebtBoundarySchedule",
    "DebtPublicationError",
    "DebtWeightPublicationBinding",
    "DebtWeightReadback",
    "PUBLICATION_KIND_COMPOSED",
    "PUBLICATION_KIND_CORE",
    "SQLiteDebtWeightPublicationJournal",
    "activate_debt_publication_schema",
    "build_debt_weight_publication_binding",
    "build_confirmed_debt_weight_publication",
    "ensure_debt_publication_schema",
    "next_debt_boundary_schedule",
    "reopen_confirmed_debt_publication",
    "retain_confirmed_debt_publication",
]
