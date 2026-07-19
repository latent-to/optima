"""Durable schema-5 authority for reviewed discovery/core composition.

The pure arithmetic lives in :mod:`optima.incentive_composition`.  This module
binds those bytes to one finalized chain scope, one already-active finite-debt
policy, a reviewed discovery-disposition ledger, append-only balances, and
confirmed composed epoch closures.  Migration is deliberately empty: neither
legacy discovery awards nor historical registered crowns are re-priced.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Iterable

from optima._strict import require_digest, require_exact_fields, require_int
from optima.chain.finite_debt_store import (
    _ATOMIC_COMPOSITION_ACTIVATION,
    FiniteDebtPolicyActivation,
    FiniteDebtStore,
    FiniteDebtStoreError,
    SeededFamilyClock,
)
from optima.finite_debt import (
    IMPROVEMENT_GROSS,
    PPM,
    DebtClaimState,
    FiniteDebtError,
    cancel_claim_balance,
    equal_campaign_budget_shares,
    expire_claim_balance,
)
from optima.incentive_composition import (
    DISCOVERY_BOUNTY_ONLY,
    DISCOVERY_REGISTERED_PROMOTION,
    ComposedEpochProjection,
    DiscoveryClaimBalance,
    DiscoveryClaimState,
    DiscoveryDebtClaim,
    IncentiveCompositionError,
    IncentiveCompositionPolicyManifest,
    ReviewedDiscoveryDisposition,
    apply_composed_epoch,
    cancel_discovery_balance,
    expire_discovery_balance,
    issue_discovery_claim,
    project_composed_epoch,
)
from optima.stack_identity import canonical_digest


SCHEMA_VERSION = 5
SELECTED_DISCOVERY_CAP_UNITS = 50_000
SELECTED_PER_AWARD_PRINCIPAL_CAP_EPOCHS = 1
SELECTED_DISCOVERY_LIFETIME_BLOCKS = 648_000
SELECTED_RESERVE_PPM = 100_000
SELECTED_EPOCH_BLOCKS = 7_200
SELECTED_SELECTION_REPORT_DIGEST = (
    "6bdfce26e4e6090e0dcc8814a636c665f28d1ff20945a09d43a9a90dc94151fc"
)
SELECTED_CORE_SELECTION_REPORT_DIGEST = (
    "7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590"
)

_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_HOTKEY = re.compile(r"[^\s]{1,256}\Z")


class IncentiveCompositionStoreError(RuntimeError):
    """Composition authority is malformed, stale, or inconsistent."""


def _digest(value: object, field: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    return require_digest(value, field=field, error=IncentiveCompositionStoreError)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    return require_int(
        value,
        field=field,
        error=IncentiveCompositionStoreError,
        minimum=minimum,
    )


def _block_hash(value: object, field: str = "block_hash") -> str:
    if not isinstance(value, str) or _BLOCK_HASH.fullmatch(value) is None:
        raise IncentiveCompositionStoreError(f"{field} is malformed")
    return value


def _hotkey(value: object, field: str = "hotkey") -> str:
    if not isinstance(value, str) or _HOTKEY.fullmatch(value) is None:
        raise IncentiveCompositionStoreError(f"{field} is malformed")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class SelectedIncentiveActivationApproval:
    """Operator-reviewed identity for the one immutable launch campaign.

    The selected economic constants are still enforced independently.  This
    manifest binds the deployment-specific facts those constants cannot know:
    the chain scope, exact policy bytes, MiniMax campaign, retained launch
    arena/evaluation stack/catalog, finalized metagraph membership, registered
    reward families, reserve recipient, production audit controls and canary,
    explicit residual-risk acceptance, and the finalized cutover point.
    """

    chain_scope_digest: str
    core_policy_digest: str
    composition_policy_digest: str
    campaign_id: str
    arena_digest: str
    evaluation_stack_digest: str
    catalog_digest: str
    membership_digest: str
    audit_control_manifest_digest: str
    audit_canary_receipt_digest: str
    audit_residual_risk_acceptance_digest: str
    reward_family_ids: tuple[str, ...]
    reserve_hotkey: str
    activation_block: int
    activation_block_hash: str

    def __post_init__(self) -> None:
        for field in (
            "chain_scope_digest",
            "core_policy_digest",
            "composition_policy_digest",
            "campaign_id",
            "arena_digest",
            "evaluation_stack_digest",
            "catalog_digest",
            "membership_digest",
            "audit_control_manifest_digest",
            "audit_canary_receipt_digest",
            "audit_residual_risk_acceptance_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        families = tuple(self.reward_family_ids)
        if (
            not families
            or any(_digest(row, "reward_family_id") != row for row in families)
            or families != tuple(sorted(set(families)))
        ):
            raise IncentiveCompositionStoreError(
                "approved reward-family roster is not canonical"
            )
        object.__setattr__(self, "reward_family_ids", families)
        expected_campaign = canonical_digest(
            "optima.economics.model-campaign.v1",
            {
                "arena_digest": self.arena_digest,
                "catalog_digest": self.catalog_digest,
                "reward_family_ids": list(families),
            },
        )
        if self.campaign_id != expected_campaign:
            raise IncentiveCompositionStoreError(
                "approved campaign_id is not derived from its "
                "arena/catalog/reward-family roster"
            )
        object.__setattr__(self, "reserve_hotkey", _hotkey(self.reserve_hotkey))
        _integer(self.activation_block, "activation_block")
        object.__setattr__(
            self,
            "activation_block_hash",
            _block_hash(self.activation_block_hash, "activation_block_hash"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_block": self.activation_block,
            "activation_block_hash": self.activation_block_hash,
            "arena_digest": self.arena_digest,
            "audit_canary_receipt_digest": self.audit_canary_receipt_digest,
            "audit_control_manifest_digest": self.audit_control_manifest_digest,
            "audit_residual_risk_acceptance_digest": (
                self.audit_residual_risk_acceptance_digest
            ),
            "campaign_id": self.campaign_id,
            "catalog_digest": self.catalog_digest,
            "chain_scope_digest": self.chain_scope_digest,
            "composition_policy_digest": self.composition_policy_digest,
            "core_policy_digest": self.core_policy_digest,
            "evaluation_stack_digest": self.evaluation_stack_digest,
            "membership_digest": self.membership_digest,
            "reserve_hotkey": self.reserve_hotkey,
            "reward_family_ids": list(self.reward_family_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> "SelectedIncentiveActivationApproval":
        row = dict(
            require_exact_fields(
                value,
                fields=frozenset(cls.__dataclass_fields__),
                label="selected incentive activation approval",
                error=IncentiveCompositionStoreError,
                exact_dict=True,
            )
        )
        families = row["reward_family_ids"]
        if type(families) is not list:
            raise IncentiveCompositionStoreError(
                "approved reward_family_ids must be an array"
            )
        row["reward_family_ids"] = tuple(families)
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.selected-incentive-activation-approval", self.to_dict()
        )


@dataclass(frozen=True)
class IncentiveCompositionActivation:
    """One exact selection activated over one exact core policy."""

    chain_scope_digest: str
    policy: IncentiveCompositionPolicyManifest
    core_activation_digest: str
    selection_report_digest: str
    activation_block: int
    activation_block_hash: str
    approval: SelectedIncentiveActivationApproval

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chain_scope_digest",
            _digest(self.chain_scope_digest, "chain_scope_digest"),
        )
        if type(self.policy) is not IncentiveCompositionPolicyManifest:
            raise IncentiveCompositionStoreError(
                "composition activation policy is not exactly typed"
            )
        object.__setattr__(
            self,
            "core_activation_digest",
            _digest(self.core_activation_digest, "core_activation_digest"),
        )
        object.__setattr__(
            self,
            "selection_report_digest",
            _digest(self.selection_report_digest, "selection_report_digest"),
        )
        _integer(self.activation_block, "activation_block")
        object.__setattr__(
            self,
            "activation_block_hash",
            _block_hash(self.activation_block_hash, "activation_block_hash"),
        )
        if type(self.approval) is not SelectedIncentiveActivationApproval:
            raise IncentiveCompositionStoreError(
                "composition activation approval is not exactly typed"
            )
        if (
            self.approval.chain_scope_digest != self.chain_scope_digest
            or self.approval.composition_policy_digest != self.policy.digest
            or self.approval.activation_block != self.activation_block
            or self.approval.activation_block_hash != self.activation_block_hash
        ):
            raise IncentiveCompositionStoreError(
                "composition activation differs from its approval"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_block": self.activation_block,
            "activation_block_hash": self.activation_block_hash,
            "chain_scope_digest": self.chain_scope_digest,
            "core_activation_digest": self.core_activation_digest,
            "policy": self.policy.to_dict(),
            "selection_report_digest": self.selection_report_digest,
            "approval": self.approval.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "IncentiveCompositionActivation":
        row = dict(
            require_exact_fields(
                value,
                fields=frozenset(cls.__dataclass_fields__),
                label="incentive-composition activation",
                error=IncentiveCompositionStoreError,
                exact_dict=True,
            )
        )
        try:
            row["policy"] = IncentiveCompositionPolicyManifest.from_dict(row["policy"])
            row["approval"] = SelectedIncentiveActivationApproval.from_dict(
                row["approval"]
            )
        except (IncentiveCompositionError, ValueError, TypeError) as exc:
            raise IncentiveCompositionStoreError(
                f"composition activation policy is invalid: {exc}"
            ) from None
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.incentive-composition-activation", self.to_dict()
        )


@dataclass(frozen=True)
class ReviewPendingDiscoveryWin:
    """Canonical retained candidate/evidence identity awaiting one review."""

    chain_scope_digest: str
    activation_digest: str
    candidate_digest: str
    reservation_digest: str
    proposal_digest: str
    retained_evidence_digest: str
    arm_digest: str
    selected_delta_digest: str
    candidate_tree_digest: str
    hotkey: str
    settlement_block: int
    settlement_block_hash: str
    settlement_event_digest: str

    def __post_init__(self) -> None:
        for field in (
            "chain_scope_digest",
            "activation_digest",
            "candidate_digest",
            "reservation_digest",
            "proposal_digest",
            "retained_evidence_digest",
            "arm_digest",
            "selected_delta_digest",
            "candidate_tree_digest",
            "settlement_event_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        _integer(self.settlement_block, "settlement_block")
        object.__setattr__(
            self,
            "settlement_block_hash",
            _block_hash(self.settlement_block_hash, "settlement_block_hash"),
        )

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "ReviewPendingDiscoveryWin":
        row = require_exact_fields(
            value,
            fields=frozenset(cls.__dataclass_fields__),
            label="review-pending discovery win",
            error=IncentiveCompositionStoreError,
            exact_dict=True,
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.review-pending-discovery-win", self.to_dict()
        )


@dataclass(frozen=True)
class ReviewedDiscoveryDispositionRecord:
    """Finalized bounded-bounty record; promotion transport is not yet retained."""

    chain_scope_digest: str
    activation_digest: str
    disposition: ReviewedDiscoveryDisposition
    authority_block: int
    authority_block_hash: str
    claim_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chain_scope_digest",
            _digest(self.chain_scope_digest, "chain_scope_digest"),
        )
        object.__setattr__(
            self,
            "activation_digest",
            _digest(self.activation_digest, "activation_digest"),
        )
        if type(self.disposition) is not ReviewedDiscoveryDisposition:
            raise IncentiveCompositionStoreError(
                "reviewed discovery disposition is not exactly typed"
            )
        _integer(self.authority_block, "authority_block")
        if self.authority_block != self.disposition.authority_block:
            raise IncentiveCompositionStoreError(
                "disposition reviewed block differs from finalized authority"
            )
        object.__setattr__(
            self,
            "authority_block_hash",
            _block_hash(self.authority_block_hash, "authority_block_hash"),
        )
        object.__setattr__(
            self,
            "claim_digest",
            _digest(
                self.claim_digest,
                "claim_digest",
                optional=self.disposition.decision
                == DISCOVERY_REGISTERED_PROMOTION,
            ),
        )
        if (self.disposition.decision == DISCOVERY_BOUNTY_ONLY) != bool(
            self.claim_digest
        ):
            raise IncentiveCompositionStoreError(
                "only a bounty disposition may name an issued claim"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_digest": self.activation_digest,
            "authority_block": self.authority_block,
            "authority_block_hash": self.authority_block_hash,
            "chain_scope_digest": self.chain_scope_digest,
            "claim_digest": self.claim_digest,
            "disposition": self.disposition.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ReviewedDiscoveryDispositionRecord":
        row = dict(
            require_exact_fields(
                value,
                fields=frozenset(cls.__dataclass_fields__),
                label="reviewed discovery disposition record",
                error=IncentiveCompositionStoreError,
                exact_dict=True,
            )
        )
        try:
            row["disposition"] = ReviewedDiscoveryDisposition.from_dict(
                row["disposition"]
            )
        except (IncentiveCompositionError, ValueError, TypeError) as exc:
            raise IncentiveCompositionStoreError(
                f"reviewed discovery disposition is invalid: {exc}"
            ) from None
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.reviewed-discovery-disposition-record", self.to_dict()
        )


@dataclass(frozen=True)
class IncentiveCompositionRewardEpoch:
    """One immutable confirmed closure debiting both claim classes."""

    chain_scope_digest: str
    activation_digest: str
    composition_policy_digest: str
    core_policy_digest: str
    epoch_index: int
    start_block: int
    effective_block: int
    effective_block_hash: str
    projection: ComposedEpochProjection
    publication_record_digest: str
    payout_event_digest: str

    def __post_init__(self) -> None:
        for field in (
            "chain_scope_digest",
            "activation_digest",
            "composition_policy_digest",
            "core_policy_digest",
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
            type(self.projection) is not ComposedEpochProjection
            or self.projection.composition_policy_digest
            != self.composition_policy_digest
            or self.projection.innovation_policy_digest != self.core_policy_digest
            or self.projection.effective_block != self.effective_block
            or self.start_block >= self.effective_block
        ):
            raise IncentiveCompositionStoreError(
                "composed reward epoch projection is inconsistent"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "activation_digest": self.activation_digest,
            "chain_scope_digest": self.chain_scope_digest,
            "composition_policy_digest": self.composition_policy_digest,
            "core_policy_digest": self.core_policy_digest,
            "effective_block": self.effective_block,
            "effective_block_hash": self.effective_block_hash,
            "epoch_index": self.epoch_index,
            "payout_event_digest": self.payout_event_digest,
            "projection": self.projection.to_dict(),
            "publication_record_digest": self.publication_record_digest,
            "start_block": self.start_block,
        }

    @classmethod
    def from_dict(cls, value: object) -> "IncentiveCompositionRewardEpoch":
        row = dict(
            require_exact_fields(
                value,
                fields=frozenset(cls.__dataclass_fields__),
                label="incentive-composition reward epoch",
                error=IncentiveCompositionStoreError,
                exact_dict=True,
            )
        )
        try:
            row["projection"] = ComposedEpochProjection.from_dict(row["projection"])
        except (IncentiveCompositionError, ValueError, TypeError) as exc:
            raise IncentiveCompositionStoreError(
                f"composed epoch projection is invalid: {exc}"
            ) from None
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain.incentive-composition-reward-epoch", self.to_dict()
        )


@dataclass(frozen=True)
class ComposedLifecycleChanges:
    """Atomic lifecycle revisions from both claim classes."""

    core_states: tuple[DebtClaimState, ...]
    discovery_states: tuple[DiscoveryClaimState, ...]


_TABLE_DEFINITIONS = (
    """
    CREATE TABLE incentive_composition_activations (
        activation_digest TEXT PRIMARY KEY,
        chain_scope_digest TEXT NOT NULL,
        composition_policy_digest TEXT NOT NULL UNIQUE,
        core_activation_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(activation_digest),
        core_policy_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(policy_digest),
        selection_report_digest TEXT NOT NULL,
        policy_json TEXT NOT NULL,
        activation_block INTEGER NOT NULL CHECK(activation_block>=0),
        activation_block_hash TEXT NOT NULL,
        activation_json TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_discovery_wins (
        win_digest TEXT PRIMARY KEY,
        composition_policy_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(composition_policy_digest),
        candidate_digest TEXT NOT NULL UNIQUE,
        reservation_digest TEXT NOT NULL UNIQUE,
        proposal_digest TEXT NOT NULL UNIQUE,
        retained_evidence_digest TEXT NOT NULL UNIQUE,
        arm_digest TEXT NOT NULL UNIQUE,
        selected_delta_digest TEXT NOT NULL UNIQUE,
        candidate_tree_digest TEXT NOT NULL UNIQUE,
        hotkey TEXT NOT NULL,
        settlement_block INTEGER NOT NULL CHECK(settlement_block>=0),
        settlement_block_hash TEXT NOT NULL,
        settlement_event_digest TEXT NOT NULL UNIQUE,
        win_json TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_discovery_dispositions (
        disposition_digest TEXT PRIMARY KEY,
        composition_policy_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(composition_policy_digest),
        proposal_digest TEXT NOT NULL UNIQUE,
        review_digest TEXT NOT NULL UNIQUE,
        win_digest TEXT NOT NULL UNIQUE
            REFERENCES incentive_discovery_wins(win_digest),
        candidate_digest TEXT NOT NULL UNIQUE,
        retained_evidence_digest TEXT NOT NULL UNIQUE,
        hotkey TEXT NOT NULL,
        decision TEXT NOT NULL
            CHECK(decision='bounty_only'),
        disposition_json TEXT NOT NULL,
        win_block INTEGER NOT NULL CHECK(win_block>=0),
        authority_block INTEGER NOT NULL CHECK(authority_block>=0),
        authority_block_hash TEXT NOT NULL,
        claim_digest TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_discovery_claims (
        claim_digest TEXT PRIMARY KEY,
        composition_policy_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(composition_policy_digest),
        disposition_digest TEXT NOT NULL UNIQUE
            REFERENCES incentive_discovery_dispositions(disposition_digest),
        hotkey TEXT NOT NULL,
        awarded_block INTEGER NOT NULL CHECK(awarded_block>=0),
        expires_block INTEGER NOT NULL CHECK(expires_block>awarded_block),
        principal_units INTEGER NOT NULL CHECK(principal_units>0),
        claim_json TEXT NOT NULL,
        issuance_reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_discovery_balances (
        claim_digest TEXT NOT NULL
            REFERENCES incentive_discovery_claims(claim_digest),
        revision INTEGER NOT NULL CHECK(revision>=0),
        balance_digest TEXT NOT NULL UNIQUE,
        principal_units INTEGER NOT NULL CHECK(principal_units>0),
        paid_units INTEGER NOT NULL CHECK(paid_units>=0),
        forfeited_units INTEGER NOT NULL CHECK(forfeited_units>=0),
        remaining_units INTEGER NOT NULL CHECK(remaining_units>=0),
        status TEXT NOT NULL
            CHECK(status IN ('open','paid','expired','cancelled')),
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
    CREATE TABLE incentive_composed_epochs (
        epoch_digest TEXT PRIMARY KEY,
        chain_scope_digest TEXT NOT NULL,
        activation_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(activation_digest),
        composition_policy_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(composition_policy_digest),
        core_policy_digest TEXT NOT NULL
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
        discovery_payout_units INTEGER NOT NULL CHECK(discovery_payout_units>=0),
        core_payout_units INTEGER NOT NULL CHECK(core_payout_units>=0),
        reserve_units INTEGER NOT NULL CHECK(reserve_units>=0),
        epoch_json TEXT NOT NULL,
        UNIQUE(composition_policy_digest,epoch_index),
        UNIQUE(composition_policy_digest,effective_block),
        CHECK(discovery_payout_units+core_payout_units+reserve_units=1000000)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_composed_allocations (
        epoch_digest TEXT NOT NULL
            REFERENCES incentive_composed_epochs(epoch_digest),
        reward_class TEXT NOT NULL CHECK(reward_class IN ('discovery','core')),
        claim_digest TEXT NOT NULL,
        hotkey TEXT NOT NULL,
        units INTEGER NOT NULL CHECK(units>0),
        PRIMARY KEY(epoch_digest,reward_class,claim_digest)
    ) STRICT
    """,
)

_INDEX_DEFINITIONS = (
    "CREATE INDEX incentive_composition_activation_block ON "
    "incentive_composition_activations(activation_block,activation_digest)",
    "CREATE INDEX incentive_discovery_balances_latest ON "
    "incentive_discovery_balances(claim_digest,revision DESC)",
)

_IMMUTABLE_TABLES = (
    "incentive_composition_activations",
    "incentive_discovery_wins",
    "incentive_discovery_dispositions",
    "incentive_discovery_claims",
    "incentive_discovery_balances",
    "incentive_composed_epochs",
    "incentive_composed_allocations",
)

_REQUIRED_COLUMNS = {
    "incentive_composition_activations": {
        "activation_digest", "chain_scope_digest", "composition_policy_digest",
        "core_activation_digest", "core_policy_digest", "selection_report_digest",
        "policy_json", "activation_block", "activation_block_hash",
        "activation_json", "reward_event_digest",
    },
    "incentive_discovery_wins": {
        "win_digest", "composition_policy_digest", "candidate_digest",
        "reservation_digest", "proposal_digest", "retained_evidence_digest",
        "arm_digest", "selected_delta_digest", "candidate_tree_digest", "hotkey",
        "settlement_block", "settlement_block_hash", "settlement_event_digest",
        "win_json", "reward_event_digest",
    },
    "incentive_discovery_dispositions": {
        "disposition_digest", "composition_policy_digest", "proposal_digest",
        "review_digest", "win_digest", "candidate_digest",
        "retained_evidence_digest", "hotkey", "decision", "disposition_json",
        "win_block", "authority_block", "authority_block_hash", "claim_digest",
        "reward_event_digest",
    },
    "incentive_discovery_claims": {
        "claim_digest", "composition_policy_digest", "disposition_digest", "hotkey",
        "awarded_block", "expires_block", "principal_units", "claim_json",
        "issuance_reward_event_digest",
    },
    "incentive_discovery_balances": {
        "claim_digest", "revision", "balance_digest", "principal_units", "paid_units",
        "forfeited_units", "remaining_units", "status", "terminal_block",
        "terminal_reason", "balance_json", "reward_event_digest",
    },
    "incentive_composed_epochs": {
        "epoch_digest", "chain_scope_digest", "activation_digest",
        "composition_policy_digest", "core_policy_digest", "epoch_index",
        "start_block", "effective_block", "effective_block_hash",
        "projection_digest", "projection_json", "publication_record_digest",
        "payout_event_digest", "discovery_payout_units", "core_payout_units",
        "reserve_units", "epoch_json",
    },
    "incentive_composed_allocations": {
        "epoch_digest", "reward_class", "claim_digest", "hotkey", "units",
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
            raise IncentiveCompositionStoreError(f"{table} is missing or not STRICT")
        observed = {item["name"] for item in db.execute(f"PRAGMA table_info({table})")}
        if observed != columns:
            raise IncentiveCompositionStoreError(
                f"{table} columns differ from schema 5"
            )
    triggers = {
        row["name"]
        for row in db.execute("SELECT name FROM sqlite_schema WHERE type='trigger'")
    }
    required = {
        f"{table}_reject_{action}"
        for table in _IMMUTABLE_TABLES
        for action in ("update", "delete")
    }
    if not required.issubset(triggers):
        raise IncentiveCompositionStoreError(
            "incentive-composition immutability triggers are incomplete"
        )


def migrate_schema4_to5(db: sqlite3.Connection) -> None:
    """Create only empty composition tables and advance metadata 4 -> 5."""

    schema = db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
    if schema is None:
        raise IncentiveCompositionStoreError("intake schema metadata is absent")
    if schema["value"] in {str(SCHEMA_VERSION), "6"}:
        _verify_schema(db)
        return
    if schema["value"] != "4":
        raise IncentiveCompositionStoreError(
            "incentive-composition migration requires intake schema 4"
        )
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
                raise IncentiveCompositionStoreError(
                    "schema-4 database contains non-authoritative composition rows"
                )
        for definition in _INDEX_DEFINITIONS:
            db.execute(definition.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS "))
        for table in _IMMUTABLE_TABLES:
            for action in ("UPDATE", "DELETE"):
                name = f"{table}_reject_{action.lower()}"
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {name} BEFORE {action} ON {table} "
                    "BEGIN SELECT RAISE(ABORT,'incentive composition rows are immutable'); END"
                )
        _verify_schema(db)
        db.execute(
            "UPDATE metadata SET value=? WHERE key='schema' AND value='4'",
            (str(SCHEMA_VERSION),),
        )
        if db.execute("SELECT changes() AS n").fetchone()["n"] != 1:
            raise IncentiveCompositionStoreError(
                "intake schema changed during composition migration"
            )
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise


class IncentiveCompositionStore:
    """Composition persistence owned by one finalized intake writer."""

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        chain_scope_digest: str,
        transaction: Callable[[], object],
        finalized_cursor: Callable[[], tuple[int, str] | None],
        core_store: FiniteDebtStore,
        reopen_settlement_evidence: Callable[[object], object],
    ):
        if (
            not isinstance(db, sqlite3.Connection)
            or type(core_store) is not FiniteDebtStore
            or core_store.db is not db
        ):
            raise IncentiveCompositionStoreError(
                "composition store requires the owning SQLite/core authority"
            )
        self.db = db
        self.chain_scope_digest = _digest(chain_scope_digest, "chain_scope_digest")
        self._transaction = transaction
        self._finalized_cursor = finalized_cursor
        self.core_store = core_store
        if not callable(reopen_settlement_evidence):
            raise IncentiveCompositionStoreError(
                "composition store requires settlement-evidence reopening authority"
            )
        self._reopen_settlement_evidence = reopen_settlement_evidence
        _verify_schema(db)
        # Reopen exact selected bytes eagerly.  A D-013 composition over the
        # superseded family-share core must not appear healthy until payout.
        self.policy_activations()

    def _require_finalized_authority(self, block: int, block_hash: str) -> None:
        height = _integer(block, "finalized block")
        digest = _block_hash(block_hash, "finalized block hash")
        if self._finalized_cursor() != (height, digest):
            raise IncentiveCompositionStoreError(
                "composition action lacks the exact finalized block/hash authority"
            )

    @staticmethod
    def _selected_policy(
        policy: IncentiveCompositionPolicyManifest,
        core: FiniteDebtPolicyActivation,
        approval: SelectedIncentiveActivationApproval,
    ) -> None:
        if type(policy) is not IncentiveCompositionPolicyManifest:
            raise IncentiveCompositionStoreError(
                "composition policy is not exactly typed"
            )
        expected = (
            policy.discovery_cap_units == SELECTED_DISCOVERY_CAP_UNITS
            and policy.per_award_principal_cap_epochs
            == SELECTED_PER_AWARD_PRINCIPAL_CAP_EPOCHS
            and policy.discovery_lifetime_blocks
            == SELECTED_DISCOVERY_LIFETIME_BLOCKS
            and policy.reserve_ppm == SELECTED_RESERVE_PPM
            and policy.epoch_blocks == SELECTED_EPOCH_BLOCKS
            and policy.selection_report_digest
            == SELECTED_SELECTION_REPORT_DIGEST
        )
        core_policy = core.policy
        campaigns = core_policy.campaign_budget_shares
        families = tuple(row.family_id for row in core_policy.reward_family_campaigns)
        selected_core = (
            core_policy.epoch_blocks == SELECTED_EPOCH_BLOCKS
            and core_policy.reserve_ppm == SELECTED_RESERVE_PPM
            and core_policy.beta_ppm == 100_000
            and core_policy.tau_blocks == 648_000
            and core_policy.lifetime_blocks == 648_000
            and core_policy.k_ppm == PPM
            and core_policy.improvement_basis == IMPROVEMENT_GROSS
            and core_policy.clock_reset_threshold_log_units_ppm == 1
            and core_policy.selection_report_digest
            == SELECTED_CORE_SELECTION_REPORT_DIGEST
            # Tomorrow's launch is deliberately one immutable MiniMax
            # campaign.  A later model rotation or one-to-two expansion needs
            # a separately reviewed successor protocol; it cannot be smuggled
            # into these activation bytes.
            and len(campaigns) == 1
            and campaigns == equal_campaign_budget_shares(
                row.campaign_id for row in campaigns
            )
            and approval.chain_scope_digest == core.chain_scope_digest
            and approval.core_policy_digest == core_policy.digest
            and approval.composition_policy_digest == policy.digest
            and approval.campaign_id == campaigns[0].campaign_id
            and approval.reward_family_ids == families
            and approval.reserve_hotkey == core_policy.reserve_hotkey
        )
        try:
            policy.validate_innovation_policy(core_policy)
        except IncentiveCompositionError as exc:
            raise IncentiveCompositionStoreError(
                f"composition/core policy binding differs: {exc}"
            ) from None
        if not expected or not selected_core:
            raise IncentiveCompositionStoreError(
                "composition activation differs from the exact D-013/D-015 selection"
            )

    def _activation_from_row(
        self, row: sqlite3.Row
    ) -> IncentiveCompositionActivation:
        schema = self.db.execute(
            "SELECT value FROM metadata WHERE key='schema'"
        ).fetchone()
        if schema is None or schema["value"] != "6":
            raise IncentiveCompositionStoreError(
                "active incentive composition requires schema-6 rollback fencing"
            )
        try:
            activation = IncentiveCompositionActivation.from_dict(
                json.loads(row["activation_json"])
            )
        except (
            IncentiveCompositionError,
            IncentiveCompositionStoreError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise IncentiveCompositionStoreError(
                f"composition activation is corrupt: {exc}"
            ) from None
        core = self.core_store._activation_by_policy(row["core_policy_digest"])
        self._selected_policy(activation.policy, core, activation.approval)
        if (
            activation.digest != row["activation_digest"]
            or activation.chain_scope_digest != self.chain_scope_digest
            or activation.policy.digest != row["composition_policy_digest"]
            or activation.core_activation_digest != row["core_activation_digest"]
            or activation.core_activation_digest != core.digest
            or activation.policy.innovation_policy_digest
            != row["core_policy_digest"]
            or activation.selection_report_digest
            != row["selection_report_digest"]
            or activation.selection_report_digest
            != activation.policy.selection_report_digest
            or activation.approval.core_policy_digest != core.policy.digest
            or _canonical_json(activation.policy.to_dict()) != row["policy_json"]
            or activation.activation_block != row["activation_block"]
            or activation.activation_block_hash != row["activation_block_hash"]
            or _canonical_json(activation.to_dict()) != row["activation_json"]
        ):
            raise IncentiveCompositionStoreError(
                "composition activation differs from retained bytes"
            )
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
            or event["event_type"] != "composition_policy_activated"
            or event["block"] != activation.activation_block
            or event["block_hash"] != activation.activation_block_hash
            or event["payload_json"] != _canonical_json(expected_payload)
        ):
            raise IncentiveCompositionStoreError(
                "composition activation reward event differs"
            )
        return activation

    def policy_activations(self) -> tuple[IncentiveCompositionActivation, ...]:
        self.core_store.reward_events()
        rows = tuple(
            self._activation_from_row(row)
            for row in self.db.execute(
                "SELECT * FROM incentive_composition_activations "
                "ORDER BY activation_block,activation_digest"
            )
        )
        schema = self.db.execute(
            "SELECT value FROM metadata WHERE key='schema'"
        ).fetchone()
        if schema is None or (
            (schema["value"] == "6" and len(rows) != 1)
            or (schema["value"] != "6" and bool(rows))
        ):
            raise IncentiveCompositionStoreError(
                "schema-6 rollback fence requires exactly one selected activation"
            )
        if len(rows) > 1:
            raise IncentiveCompositionStoreError(
                "schema-5 selected composition permits one activation"
            )
        return rows

    def active_policy_activation(
        self, *, at_block: int
    ) -> IncentiveCompositionActivation | None:
        height = _integer(at_block, "composition lookup block")
        rows = tuple(
            row for row in self.policy_activations() if row.activation_block <= height
        )
        return None if not rows else rows[-1]

    def activate_policy(
        self,
        policy: IncentiveCompositionPolicyManifest,
        *,
        activation_block: int,
        activation_block_hash: str,
    ) -> IncentiveCompositionActivation:
        """Reject the removed second half of the former two-call cutover."""

        del policy, activation_block, activation_block_hash
        raise IncentiveCompositionStoreError(
            "standalone composition activation is disabled; use the atomic "
            "selected incentive cutover"
        )

    def _require_quiescent_cutover(self, activation_block: int) -> None:
        """Prove no pre-cutover intake can later acquire V2 economics."""

        height = _integer(activation_block, "activation_block")
        unsettled = self.db.execute(
            "SELECT r.reservation_id FROM reservations AS r LEFT JOIN "
            "settlement_candidates AS s USING(reservation_id) WHERE r.block<=? AND "
            "(r.status NOT IN ('failed','expired','qualified') OR "
            "(r.status='qualified' AND (s.reservation_id IS NULL OR s.status NOT IN "
            "('crowned','neutralized','discovery_bounty','duplicate_proposal',"
            "'review_pending','reviewed_bounty','reviewed_promotion',"
            "'review_ineligible','review_expired')))) LIMIT 1",
            (height,),
        ).fetchone()
        if unsettled is not None:
            raise IncentiveCompositionStoreError(
                "atomic incentive cutover requires quiescent pre-activation intake"
            )

    def activate_selected_policy(
        self,
        core_policy,
        policy: IncentiveCompositionPolicyManifest,
        approval: SelectedIncentiveActivationApproval,
        *,
        expected_approval_digest: str,
        seeded_family_clocks: Iterable[SeededFamilyClock] = (),
    ) -> IncentiveCompositionActivation:
        """Atomically activate the exact one-campaign D-013/D-015 selection."""

        from optima.finite_debt import FiniteDebtPolicyManifest

        if type(core_policy) is not FiniteDebtPolicyManifest:
            raise IncentiveCompositionStoreError(
                "core activation policy is not exactly typed"
            )
        if type(policy) is not IncentiveCompositionPolicyManifest:
            raise IncentiveCompositionStoreError(
                "composition policy is not exactly typed"
            )
        if type(approval) is not SelectedIncentiveActivationApproval:
            raise IncentiveCompositionStoreError(
                "selected incentive approval is not exactly typed"
            )
        expected_approval = _digest(
            expected_approval_digest, "expected_approval_digest"
        )
        if approval.digest != expected_approval:
            raise IncentiveCompositionStoreError(
                "selected incentive approval differs from the pinned digest"
            )
        height = approval.activation_block
        authority_hash = approval.activation_block_hash
        seeds = tuple(seeded_family_clocks)
        if any(type(row) is not SeededFamilyClock for row in seeds):
            raise IncentiveCompositionStoreError(
                "seeded family clocks are not exactly typed"
            )
        seeds = tuple(sorted(seeds, key=lambda row: row.family_id))

        with self._transaction():
            self._require_finalized_authority(height, authority_hash)
            exact = self.db.execute(
                "SELECT * FROM incentive_composition_activations "
                "WHERE activation_block=? AND activation_block_hash=?",
                (height, authority_hash),
            ).fetchone()
            if exact is not None:
                retained = self._activation_from_row(exact)
                retained_core = self.core_store._activation_by_policy(
                    core_policy.digest
                )
                if (
                    retained.policy == policy
                    and retained.approval == approval
                    and retained.core_activation_digest == retained_core.digest
                    and retained_core.policy == core_policy
                    and retained_core.seeded_family_clocks == seeds
                ):
                    return retained
                raise IncentiveCompositionStoreError(
                    "composition activation block already binds other bytes"
                )
            if self.policy_activations():
                raise IncentiveCompositionStoreError(
                    "selected incentive campaign is immutable; successor activation "
                    "is disabled"
                )
            if self.core_store.policy_activations():
                raise IncentiveCompositionStoreError(
                    "retained core-only activation proves a non-atomic cutover"
                )
            if any(
                row.balance.status == "open"
                for row in self.core_store._claim_states()
            ):
                raise IncentiveCompositionStoreError(
                    "composition activation requires zero open core debt"
                )
            if any(row.balance.status == "open" for row in self._discovery_states()):
                raise IncentiveCompositionStoreError(
                    "composition activation requires zero open discovery debt"
                )
            # Legacy discovery claims have no retained lifecycle journal: their
            # mutable status column is not authority for proving that an award
            # became terminal.  Any retained row therefore blocks the one-way
            # cutover, including a row whose status was manually changed.
            if self.db.execute(
                "SELECT 1 FROM discovery_bounty_claims LIMIT 1"
            ).fetchone() is not None:
                raise IncentiveCompositionStoreError(
                    "legacy discovery claims block reviewed-only composition"
                )
            if (
                self.db.execute(
                    "SELECT 1 FROM weight_publications LIMIT 1"
                ).fetchone()
                is not None
                or self.db.execute(
                    "SELECT 1 FROM metadata WHERE key IN "
                    "('weight_publication_head','emissions_policy_digest') LIMIT 1"
                ).fetchone()
                is not None
            ):
                raise IncentiveCompositionStoreError(
                    "legacy V1 projection/publication state requires an explicit cutover"
                )
            self._require_quiescent_cutover(height)
            try:
                from optima.chain.debt_publication import (
                    DebtPublicationError,
                    activate_debt_publication_schema,
                )

                activate_debt_publication_schema(self.db)
            except DebtPublicationError as exc:
                raise IncentiveCompositionStoreError(
                    f"debt publication schema activation failed: {exc}"
                ) from None
            try:
                core = self.core_store.activate_policy(
                    core_policy,
                    activation_block=height,
                    activation_block_hash=authority_hash,
                    seeded_family_clocks=seeds,
                    _atomic_composition_authority=(
                        _ATOMIC_COMPOSITION_ACTIVATION
                    ),
                )
            except FiniteDebtStoreError as exc:
                raise IncentiveCompositionStoreError(
                    f"atomic core activation failed: {exc}"
                ) from None
            self._selected_policy(policy, core, approval)
            activation = IncentiveCompositionActivation(
                self.chain_scope_digest,
                policy,
                core.digest,
                policy.selection_report_digest,
                height,
                authority_hash,
                approval,
            )
            event_digest = self.core_store._append_event(
                "composition_policy_activated",
                block=height,
                block_hash=authority_hash,
                payload={
                    "activation": activation.to_dict(),
                    "activation_digest": activation.digest,
                },
            )
            self.db.execute(
                "INSERT INTO incentive_composition_activations(activation_digest,"
                "chain_scope_digest,composition_policy_digest,core_activation_digest,"
                "core_policy_digest,selection_report_digest,policy_json,activation_block,"
                "activation_block_hash,activation_json,reward_event_digest) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    activation.digest,
                    self.chain_scope_digest,
                    policy.digest,
                    core.digest,
                    core.policy.digest,
                    policy.selection_report_digest,
                    _canonical_json(policy.to_dict()),
                    height,
                    authority_hash,
                    _canonical_json(activation.to_dict()),
                    event_digest,
                ),
            )
        return activation

    def _activation_by_policy(
        self, policy_digest: str
    ) -> IncentiveCompositionActivation:
        digest = _digest(policy_digest, "composition_policy_digest")
        row = self.db.execute(
            "SELECT * FROM incentive_composition_activations "
            "WHERE composition_policy_digest=?",
            (digest,),
        ).fetchone()
        if row is None:
            raise IncentiveCompositionStoreError(
                "composition row names an absent policy"
            )
        return self._activation_from_row(row)

    def _win_from_row(
        self,
        row: sqlite3.Row,
        *,
        validate_lifecycle: bool = True,
    ) -> ReviewPendingDiscoveryWin:
        from optima.settlement import (
            SettlementCandidate,
            SettlementError,
            SettlementEvent,
            SettlementEventType,
            SettlementEvidence,
        )

        try:
            win = ReviewPendingDiscoveryWin.from_dict(json.loads(row["win_json"]))
        except (
            IncentiveCompositionStoreError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise IncentiveCompositionStoreError(
                f"review-pending discovery win is corrupt: {exc}"
            ) from None
        activation = self._activation_by_policy(row["composition_policy_digest"])
        if (
            win.digest != row["win_digest"]
            or win.activation_digest != activation.digest
            or win.chain_scope_digest != self.chain_scope_digest
            or win.candidate_digest != row["candidate_digest"]
            or win.reservation_digest != row["reservation_digest"]
            or win.proposal_digest != row["proposal_digest"]
            or win.retained_evidence_digest != row["retained_evidence_digest"]
            or win.arm_digest != row["arm_digest"]
            or win.selected_delta_digest != row["selected_delta_digest"]
            or win.candidate_tree_digest != row["candidate_tree_digest"]
            or win.hotkey != row["hotkey"]
            or win.settlement_block != row["settlement_block"]
            or win.settlement_block_hash != row["settlement_block_hash"]
            or win.settlement_event_digest != row["settlement_event_digest"]
            or _canonical_json(win.to_dict()) != row["win_json"]
        ):
            raise IncentiveCompositionStoreError(
                "review-pending discovery win differs from retained bytes"
            )
        candidate_row = self.db.execute(
            "SELECT candidate_json,status,reason,settlement_evidence_digest FROM "
            "settlement_candidates WHERE candidate_digest=? AND reservation_id=?",
            (win.candidate_digest, win.reservation_digest),
        ).fetchone()
        if candidate_row is None or candidate_row["status"] not in {
            "review_pending",
            "reviewed_bounty",
            "reviewed_promotion",
            "review_expired",
        }:
            raise IncentiveCompositionStoreError(
                "review-pending win lacks its retained settlement candidate"
            )
        try:
            candidate = SettlementCandidate.from_dict(
                json.loads(candidate_row["candidate_json"])
            )
            evidence = self._reopen_settlement_evidence(candidate)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise IncentiveCompositionStoreError(
                f"review-pending win evidence cannot reopen: {exc}"
            ) from None
        if (
            type(evidence) is not SettlementEvidence
            or candidate.digest != win.candidate_digest
            or candidate.lane != "discovery"
            or candidate.reservation_digest != win.reservation_digest
            or candidate.proposal_digest != win.proposal_digest
            or candidate.arm_digest != win.arm_digest
            or candidate.selected_delta_digest != win.selected_delta_digest
            or candidate.candidate_tree_digest != win.candidate_tree_digest
            or candidate.hotkey != win.hotkey
            or evidence.digest != win.retained_evidence_digest
            or candidate_row["settlement_evidence_digest"]
            != win.retained_evidence_digest
        ):
            raise IncentiveCompositionStoreError(
                "review-pending win differs from reopened candidate/evidence"
            )
        settlement = self.db.execute(
            "SELECT * FROM settlement_events "
            "WHERE event_digest=?",
            (win.settlement_event_digest,),
        ).fetchone()
        settlement_event = None
        if settlement is not None:
            try:
                settlement_event = SettlementEvent.from_dict(
                    json.loads(settlement["event_json"])
                )
            except (
                SettlementError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise IncentiveCompositionStoreError(
                    f"review-pending settlement event is corrupt: {exc}"
                ) from None
        event = self.db.execute(
            "SELECT event_type,block,block_hash,payload_json FROM "
            "finite_debt_reward_events WHERE event_digest=?",
            (row["reward_event_digest"],),
        ).fetchone()
        expected_payload = {"win": win.to_dict(), "win_digest": win.digest}
        if (
            settlement is None
            or settlement_event is None
            or settlement["event_type"] != "DISCOVERY_BOUNTY"
            or settlement["event_id"] != win.settlement_event_digest
            or settlement["reservation_id"] != win.reservation_digest
            or settlement["arena_id"] != candidate.arena_digest
            or settlement["target_id"] != candidate.target_id
            or settlement["event_digest"] != win.settlement_event_digest
            or settlement["sequence"] != settlement_event.sequence
            or settlement["event_json"]
            != _canonical_json(settlement_event.to_dict())
            or settlement_event.digest != win.settlement_event_digest
            or settlement_event.event_type is not SettlementEventType.DISCOVERY_BOUNTY
            or settlement_event.candidate_digest != candidate.digest
            or settlement_event.subject_digest != candidate.selected_delta_digest
            or settlement_event.target_id != candidate.target_id
            or settlement_event.from_stack_digest
            != candidate.incumbent_stack_digest
            or settlement_event.from_tree_digest != candidate.incumbent_tree_digest
            or settlement_event.to_stack_digest != candidate.incumbent_stack_digest
            or settlement_event.to_tree_digest != candidate.incumbent_tree_digest
            or settlement_event.reason != "qualified_discovery"
            or event is None
            or event["event_type"] != "discovery_win_retained"
            or event["block"] != win.settlement_block
            or event["block_hash"] != win.settlement_block_hash
            or event["payload_json"] != _canonical_json(expected_payload)
        ):
            raise IncentiveCompositionStoreError(
                "review-pending win event authority differs"
            )
        if not validate_lifecycle:
            return win

        disposition_rows = tuple(
            self.db.execute(
                "SELECT * FROM incentive_discovery_dispositions WHERE "
                "win_digest=? OR candidate_digest=? OR proposal_digest=? OR "
                "retained_evidence_digest=? ORDER BY disposition_digest",
                (
                    win.digest,
                    win.candidate_digest,
                    win.proposal_digest,
                    win.retained_evidence_digest,
                ),
            )
        )
        status = candidate_row["status"]
        reason = candidate_row["reason"]
        if status == "review_pending":
            if reason != "review_pending" or disposition_rows:
                raise IncentiveCompositionStoreError(
                    "review-pending discovery lifecycle cardinality differs"
                )
        elif status == "reviewed_bounty":
            if reason != "reviewed_bounty" or len(disposition_rows) != 1:
                raise IncentiveCompositionStoreError(
                    "reviewed discovery bounty lifecycle cardinality differs"
                )
            record = self._disposition_from_row(disposition_rows[0])
            if (
                record.disposition.win_digest != win.digest
                or record.disposition.decision != DISCOVERY_BOUNTY_ONLY
                or not record.claim_digest
            ):
                raise IncentiveCompositionStoreError(
                    "reviewed discovery bounty disposition differs"
                )
            claim_rows = tuple(
                self.db.execute(
                    "SELECT * FROM incentive_discovery_claims WHERE "
                    "disposition_digest=? OR claim_digest=? ORDER BY claim_digest",
                    (record.digest, record.claim_digest),
                )
            )
            if len(claim_rows) != 1:
                raise IncentiveCompositionStoreError(
                    "reviewed discovery bounty claim cardinality differs"
                )
            claim = self._claim_from_row(claim_rows[0])
            initial_rows = tuple(
                self.db.execute(
                    "SELECT * FROM incentive_discovery_balances "
                    "WHERE claim_digest=? AND revision=0",
                    (claim.digest,),
                )
            )
            if (
                claim.digest != record.claim_digest
                or len(initial_rows) != 1
                or self._balance_from_row(initial_rows[0])
                != DiscoveryClaimBalance.open(claim)
                or initial_rows[0]["reward_event_digest"]
                != disposition_rows[0]["reward_event_digest"]
            ):
                raise IncentiveCompositionStoreError(
                    "reviewed discovery bounty claim authority differs"
                )
        elif status == "reviewed_promotion":
            raise IncentiveCompositionStoreError(
                "registered promotion lacks retained DiscoveryWinRecord/"
                "DiscoveryPromotion authority"
            )
        elif disposition_rows:
            raise IncentiveCompositionStoreError(
                "expired discovery review has a reviewed disposition"
            )

        if status == "review_expired":
            prefix = "review_expired:"
            if not isinstance(reason, str) or not reason.startswith(prefix):
                raise IncentiveCompositionStoreError(
                    "expired discovery review lacks its terminal event"
                )
            expiry_event_digest = _digest(
                reason[len(prefix):], "discovery review expiry event digest"
            )
            expiry_event = self.db.execute(
                "SELECT event_type,block,block_hash,payload_json FROM "
                "finite_debt_reward_events WHERE event_digest=?",
                (expiry_event_digest,),
            ).fetchone()
            deadline = (
                win.settlement_block
                + activation.policy.discovery_lifetime_blocks
            )
            if (
                expiry_event is None
                or expiry_event["event_type"] != "discovery_review_expired"
                or expiry_event["block"] < deadline
                or expiry_event["payload_json"]
                != _canonical_json(
                    {"deadline_block": deadline, "win_digest": win.digest}
                )
            ):
                raise IncentiveCompositionStoreError(
                    "expired discovery review event authority differs"
                )
        return win

    def _validated_wins(
        self,
    ) -> tuple[tuple[ReviewPendingDiscoveryWin, str], ...]:
        """Reopen every retained win before any lifecycle-status filter."""

        result: list[tuple[ReviewPendingDiscoveryWin, str]] = []
        for row in self.db.execute(
            "SELECT * FROM incentive_discovery_wins "
            "ORDER BY settlement_block,win_digest"
        ):
            win = self._win_from_row(row)
            candidate = self.db.execute(
                "SELECT status FROM settlement_candidates "
                "WHERE candidate_digest=? AND reservation_id=?",
                (win.candidate_digest, win.reservation_digest),
            ).fetchone()
            if candidate is None:
                raise IncentiveCompositionStoreError(
                    "review-pending win lacks its retained settlement candidate"
                )
            result.append((win, candidate["status"]))
        return tuple(result)

    def review_pending_wins(self) -> tuple[ReviewPendingDiscoveryWin, ...]:
        self.core_store.reward_events()
        return tuple(
            win for win, status in self._validated_wins()
            if status == "review_pending"
        )

    def expire_review_pending_wins(
        self,
        *,
        current_block: int,
        current_block_hash: str,
    ) -> tuple[ReviewPendingDiscoveryWin, ...]:
        """Terminalize unreviewed wins once their bounty window has elapsed."""

        height = _integer(current_block, "discovery review expiry block")
        authority_hash = _block_hash(
            current_block_hash, "discovery review expiry block hash"
        )
        expired: list[ReviewPendingDiscoveryWin] = []
        with self._transaction():
            self._require_finalized_authority(height, authority_hash)
            activation = self.active_policy_activation(at_block=height)
            if activation is None:
                raise IncentiveCompositionStoreError(
                    "discovery review expiry requires active composition"
                )
            wins = tuple(
                win for win, status in self._validated_wins()
                if status == "review_pending"
            )
            for win in wins:
                deadline = (
                    win.settlement_block
                    + activation.policy.discovery_lifetime_blocks
                )
                if height < deadline:
                    continue
                event_digest = self.core_store._append_event(
                    "discovery_review_expired",
                    block=height,
                    block_hash=authority_hash,
                    payload={
                        "deadline_block": deadline,
                        "win_digest": win.digest,
                    },
                )
                cursor = self.db.execute(
                    "UPDATE settlement_candidates SET status='review_expired',"
                    "reason=?,lease_id='',lease_expires_block=0 "
                    "WHERE candidate_digest=? AND status='review_pending'",
                    (
                        f"review_expired:{event_digest}",
                        win.candidate_digest,
                    ),
                )
                if cursor.rowcount != 1:
                    raise IncentiveCompositionStoreError(
                        "review-pending discovery win changed during expiry"
                    )
                expired.append(win)
        return tuple(expired)

    def _win_by_digest(
        self,
        win_digest: str,
        *,
        validate_lifecycle: bool = True,
    ) -> ReviewPendingDiscoveryWin:
        digest = _digest(win_digest, "win_digest")
        row = self.db.execute(
            "SELECT * FROM incentive_discovery_wins WHERE win_digest=?",
            (digest,),
        ).fetchone()
        if row is None:
            raise IncentiveCompositionStoreError(
                "reviewed disposition names no retained discovery win"
            )
        return self._win_from_row(
            row,
            validate_lifecycle=validate_lifecycle,
        )

    def retain_review_pending_win_in_transaction(
        self,
        candidate: object,
        evidence: object,
        *,
        settlement_block: int,
        settlement_block_hash: str,
        settlement_event_digest: str,
    ) -> ReviewPendingDiscoveryWin:
        """Retain a qualified discovery win without minting either reward path."""

        from optima.settlement import SettlementCandidate, SettlementEvidence

        if not self.db.in_transaction:
            raise IncentiveCompositionStoreError(
                "review-pending win requires the settlement transaction"
            )
        if (
            type(candidate) is not SettlementCandidate
            or candidate.lane != "discovery"
            or type(evidence) is not SettlementEvidence
            or evidence.candidate_digest != candidate.digest
            or evidence.reservation_digest != candidate.reservation_digest
        ):
            raise IncentiveCompositionStoreError(
                "review-pending win candidate/evidence authority differs"
            )
        height = _integer(settlement_block, "settlement_block")
        authority_hash = _block_hash(settlement_block_hash, "settlement_block_hash")
        settlement_event = _digest(settlement_event_digest, "settlement_event_digest")
        self._require_finalized_authority(height, authority_hash)
        activation = self.active_policy_activation(at_block=height)
        if activation is None:
            raise IncentiveCompositionStoreError(
                "review-pending discovery composition is not active"
            )
        if candidate.finalized_block < activation.activation_block:
            raise IncentiveCompositionStoreError(
                "pre-activation discovery candidate cannot enter reviewed composition"
            )
        win = ReviewPendingDiscoveryWin(
            self.chain_scope_digest,
            activation.digest,
            candidate.digest,
            candidate.reservation_digest,
            candidate.proposal_digest,
            evidence.digest,
            candidate.arm_digest,
            candidate.selected_delta_digest,
            candidate.candidate_tree_digest,
            candidate.hotkey,
            height,
            authority_hash,
            settlement_event,
        )
        exact = self.db.execute(
            "SELECT * FROM incentive_discovery_wins WHERE win_digest=?",
            (win.digest,),
        ).fetchone()
        if exact is not None:
            return self._win_from_row(exact)
        for field, value in (
            ("candidate_digest", win.candidate_digest),
            ("reservation_digest", win.reservation_digest),
            ("proposal_digest", win.proposal_digest),
            ("retained_evidence_digest", win.retained_evidence_digest),
            ("arm_digest", win.arm_digest),
            ("selected_delta_digest", win.selected_delta_digest),
            ("candidate_tree_digest", win.candidate_tree_digest),
        ):
            if self.db.execute(
                f"SELECT 1 FROM incentive_discovery_wins WHERE {field}=?",
                (value,),
            ).fetchone() is not None:
                raise IncentiveCompositionStoreError(
                    "discovery candidate/evidence already has a retained win"
                )
        settlement = self.db.execute(
            "SELECT event_type,reservation_id,event_digest FROM settlement_events "
            "WHERE event_digest=?",
            (settlement_event,),
        ).fetchone()
        if (
            settlement is None
            or settlement["event_type"] != "DISCOVERY_BOUNTY"
            or settlement["reservation_id"] != candidate.reservation_digest
        ):
            raise IncentiveCompositionStoreError(
                "review-pending win lacks its exact settlement event"
            )
        event_digest = self.core_store._append_event(
            "discovery_win_retained",
            block=height,
            block_hash=authority_hash,
            payload={"win": win.to_dict(), "win_digest": win.digest},
        )
        self.db.execute(
            "INSERT INTO incentive_discovery_wins(win_digest,"
            "composition_policy_digest,candidate_digest,reservation_digest,"
            "proposal_digest,retained_evidence_digest,arm_digest,selected_delta_digest,"
            "candidate_tree_digest,hotkey,settlement_block,settlement_block_hash,"
            "settlement_event_digest,win_json,reward_event_digest) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                win.digest,
                activation.policy.digest,
                win.candidate_digest,
                win.reservation_digest,
                win.proposal_digest,
                win.retained_evidence_digest,
                win.arm_digest,
                win.selected_delta_digest,
                win.candidate_tree_digest,
                win.hotkey,
                height,
                authority_hash,
                settlement_event,
                _canonical_json(win.to_dict()),
                event_digest,
            ),
        )
        return win

    def _disposition_from_row(
        self, row: sqlite3.Row
    ) -> ReviewedDiscoveryDispositionRecord:
        try:
            disposition = ReviewedDiscoveryDisposition.from_dict(
                json.loads(row["disposition_json"])
            )
        except (IncentiveCompositionError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IncentiveCompositionStoreError(
                f"reviewed discovery disposition is corrupt: {exc}"
            ) from None
        if disposition.decision == DISCOVERY_REGISTERED_PROMOTION:
            raise IncentiveCompositionStoreError(
                "registered promotion lacks retained DiscoveryWinRecord/"
                "DiscoveryPromotion authority"
            )
        activation = self._activation_by_policy(row["composition_policy_digest"])
        try:
            disposition.validate_policy(activation.policy)
        except IncentiveCompositionError as exc:
            raise IncentiveCompositionStoreError(
                f"reviewed discovery disposition differs: {exc}"
            ) from None
        record = ReviewedDiscoveryDispositionRecord(
            self.chain_scope_digest,
            activation.digest,
            disposition,
            row["authority_block"],
            row["authority_block_hash"],
            row["claim_digest"],
        )
        win = self._win_by_digest(
            disposition.win_digest,
            validate_lifecycle=False,
        )
        if (
            record.digest != row["disposition_digest"]
            or disposition.proposal_digest != row["proposal_digest"]
            or disposition.review_digest != row["review_digest"]
            or disposition.win_digest != row["win_digest"]
            or row["candidate_digest"] != win.candidate_digest
            or disposition.win_block != row["win_block"]
            or disposition.win_block != win.settlement_block
            or disposition.retained_evidence_digest
            != row["retained_evidence_digest"]
            or disposition.hotkey != row["hotkey"]
            or disposition.decision != row["decision"]
            or _canonical_json(disposition.to_dict()) != row["disposition_json"]
        ):
            raise IncentiveCompositionStoreError(
                "reviewed discovery disposition differs from retained bytes"
            )
        event = self.db.execute(
            "SELECT event_type,block,block_hash,payload_json FROM "
            "finite_debt_reward_events WHERE event_digest=?",
            (row["reward_event_digest"],),
        ).fetchone()
        claim_value: object = None
        if record.claim_digest:
            claim_row = self.db.execute(
                "SELECT claim_json FROM incentive_discovery_claims WHERE claim_digest=?",
                (record.claim_digest,),
            ).fetchone()
            if claim_row is None:
                raise IncentiveCompositionStoreError(
                    "bounty disposition lacks its issued claim"
                )
            claim_value = json.loads(claim_row["claim_json"])
        expected_payload = {
            "claim": claim_value,
            "record": record.to_dict(),
            "record_digest": record.digest,
        }
        expected_type = (
            "discovery_claim_issued"
            if record.claim_digest
            else "discovery_disposition_recorded"
        )
        if (
            event is None
            or event["event_type"] != expected_type
            or event["block"] != record.authority_block
            or event["block_hash"] != record.authority_block_hash
            or event["payload_json"] != _canonical_json(expected_payload)
        ):
            raise IncentiveCompositionStoreError(
                "reviewed discovery disposition event differs"
            )
        return record

    def disposition_records(self) -> tuple[ReviewedDiscoveryDispositionRecord, ...]:
        self.core_store.reward_events()
        self._validated_wins()
        return tuple(
            self._disposition_from_row(row)
            for row in self.db.execute(
                "SELECT * FROM incentive_discovery_dispositions "
                "ORDER BY authority_block,disposition_digest"
            )
        )

    def record_disposition(
        self,
        disposition: ReviewedDiscoveryDisposition,
        *,
        authority_block_hash: str,
    ) -> ReviewedDiscoveryDispositionRecord:
        """Atomically issue one bounded bounty from a retained reviewed win.

        The controller-supplied ``review_digest`` is content-bound here; this
        store does not independently reopen or grade an external governance
        review system. Registered promotion remains fail-closed until the
        existing typed DiscoveryWinRecord/DiscoveryPromotion authority is
        transported and reopened durably.
        """

        if type(disposition) is not ReviewedDiscoveryDisposition:
            raise IncentiveCompositionStoreError(
                "discovery disposition is not exactly typed"
            )
        height = _integer(disposition.authority_block, "authority_block")
        authority_hash = _block_hash(authority_block_hash, "authority_block_hash")
        with self._transaction():
            self._require_finalized_authority(height, authority_hash)
            activation = self.active_policy_activation(at_block=height)
            if activation is None:
                raise IncentiveCompositionStoreError(
                    "reviewed discovery composition is not active"
                )
            if disposition.decision == DISCOVERY_REGISTERED_PROMOTION:
                raise IncentiveCompositionStoreError(
                    "registered promotion requires retained DiscoveryWinRecord/"
                    "DiscoveryPromotion authority; settlement does not transport it"
                )
            try:
                disposition.validate_policy(activation.policy)
                win = self._win_by_digest(disposition.win_digest)
                if (
                    disposition.proposal_digest != win.proposal_digest
                    or disposition.retained_evidence_digest
                    != win.retained_evidence_digest
                    or disposition.hotkey != win.hotkey
                    or disposition.win_block != win.settlement_block
                    or height < win.settlement_block
                ):
                    raise IncentiveCompositionError(
                        "reviewed disposition differs from retained candidate/evidence"
                    )
                claim = issue_discovery_claim(activation.policy, disposition)
            except IncentiveCompositionError as exc:
                raise IncentiveCompositionStoreError(
                    f"reviewed discovery disposition cannot settle: {exc}"
                ) from None
            core = self.core_store._activation_by_policy(
                activation.policy.innovation_policy_digest
            )
            if disposition.hotkey == core.policy.reserve_hotkey:
                raise IncentiveCompositionStoreError(
                    "reserve hotkey cannot receive discovery debt"
                )
            record = ReviewedDiscoveryDispositionRecord(
                self.chain_scope_digest,
                activation.digest,
                disposition,
                height,
                authority_hash,
                "" if claim is None else claim.digest,
            )
            exact = self.db.execute(
                "SELECT * FROM incentive_discovery_dispositions "
                "WHERE disposition_digest=?",
                (record.digest,),
            ).fetchone()
            if exact is not None:
                return self._disposition_from_row(exact)
            for field, value in (
                ("proposal_digest", disposition.proposal_digest),
                ("review_digest", disposition.review_digest),
                ("win_digest", disposition.win_digest),
                ("candidate_digest", win.candidate_digest),
                (
                    "retained_evidence_digest",
                    disposition.retained_evidence_digest,
                ),
            ):
                if self.db.execute(
                    f"SELECT 1 FROM incentive_discovery_dispositions WHERE {field}=?",
                    (value,),
                ).fetchone() is not None:
                    raise IncentiveCompositionStoreError(
                        "discovery proposal/review/win already has a disposition"
                    )
            payload = {
                "claim": None if claim is None else claim.to_dict(),
                "record": record.to_dict(),
                "record_digest": record.digest,
            }
            event_digest = self.core_store._append_event(
                "discovery_disposition_recorded"
                if claim is None
                else "discovery_claim_issued",
                block=height,
                block_hash=authority_hash,
                payload=payload,
            )
            self.db.execute(
                "INSERT INTO incentive_discovery_dispositions(disposition_digest,"
                "composition_policy_digest,proposal_digest,review_digest,win_digest,"
                "candidate_digest,retained_evidence_digest,hotkey,decision,"
                "disposition_json,win_block,authority_block,"
                "authority_block_hash,claim_digest,reward_event_digest) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.digest,
                    activation.policy.digest,
                    disposition.proposal_digest,
                    disposition.review_digest,
                    disposition.win_digest,
                    win.candidate_digest,
                    disposition.retained_evidence_digest,
                    disposition.hotkey,
                    disposition.decision,
                    _canonical_json(disposition.to_dict()),
                    disposition.win_block,
                    height,
                    authority_hash,
                    record.claim_digest,
                    event_digest,
                ),
            )
            if claim is not None:
                self.db.execute(
                    "INSERT INTO incentive_discovery_claims(claim_digest,"
                    "composition_policy_digest,disposition_digest,hotkey,awarded_block,"
                    "expires_block,principal_units,claim_json,issuance_reward_event_digest) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        claim.digest,
                        activation.policy.digest,
                        record.digest,
                        claim.hotkey,
                        claim.awarded_block,
                        claim.expires_block,
                        claim.principal_units,
                        _canonical_json(claim.to_dict()),
                        event_digest,
                    ),
                )
                self._insert_discovery_balance(
                    DiscoveryClaimBalance.open(claim),
                    revision=0,
                    reward_event_digest=event_digest,
                )
            status = (
                "reviewed_bounty"
                if disposition.decision == DISCOVERY_BOUNTY_ONLY
                else "reviewed_promotion"
            )
            cursor = self.db.execute(
                "UPDATE settlement_candidates SET status=?,reason=?,lease_id='',"
                "lease_expires_block=0 WHERE candidate_digest=? "
                "AND status='review_pending'",
                (status, status, win.candidate_digest),
            )
            if cursor.rowcount != 1:
                raise IncentiveCompositionStoreError(
                    "retained discovery win was already consumed or changed"
                )
        return record

    def _claim_from_row(self, row: sqlite3.Row) -> DiscoveryDebtClaim:
        try:
            claim = DiscoveryDebtClaim.from_dict(json.loads(row["claim_json"]))
        except (IncentiveCompositionError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IncentiveCompositionStoreError(
                f"discovery debt claim is corrupt: {exc}"
            ) from None
        activation = self._activation_by_policy(row["composition_policy_digest"])
        disposition_row = self.db.execute(
            "SELECT * FROM incentive_discovery_dispositions "
            "WHERE disposition_digest=?",
            (row["disposition_digest"],),
        ).fetchone()
        if disposition_row is None:
            raise IncentiveCompositionStoreError(
                "discovery claim lacks its disposition"
            )
        record = self._disposition_from_row(disposition_row)
        try:
            claim.validate_policy(activation.policy, record.disposition)
        except IncentiveCompositionError as exc:
            raise IncentiveCompositionStoreError(
                f"discovery debt claim differs: {exc}"
            ) from None
        if (
            claim.digest != row["claim_digest"]
            or claim.policy_digest != row["composition_policy_digest"]
            or claim.disposition_digest != record.disposition.digest
            or record.claim_digest != claim.digest
            or claim.hotkey != row["hotkey"]
            or claim.awarded_block != row["awarded_block"]
            or claim.expires_block != row["expires_block"]
            or claim.principal_units != row["principal_units"]
            or _canonical_json(claim.to_dict()) != row["claim_json"]
            or row["issuance_reward_event_digest"]
            != disposition_row["reward_event_digest"]
        ):
            raise IncentiveCompositionStoreError(
                "discovery debt claim differs from retained bytes"
            )
        return claim

    @staticmethod
    def _balance_from_row(row: sqlite3.Row) -> DiscoveryClaimBalance:
        try:
            balance = DiscoveryClaimBalance.from_dict(
                json.loads(row["balance_json"])
            )
        except (IncentiveCompositionError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IncentiveCompositionStoreError(
                f"discovery balance is corrupt: {exc}"
            ) from None
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
            raise IncentiveCompositionStoreError(
                "discovery balance differs from retained bytes"
            )
        return balance

    def _discovery_states(
        self, *, policy_digest: str | None = None
    ) -> tuple[DiscoveryClaimState, ...]:
        params: tuple[str, ...] = ()
        where = ""
        if policy_digest is not None:
            where = " WHERE composition_policy_digest=?"
            params = (_digest(policy_digest, "composition_policy_digest"),)
        states: list[DiscoveryClaimState] = []
        for claim_row in self.db.execute(
            "SELECT * FROM incentive_discovery_claims" + where + " ORDER BY claim_digest",
            params,
        ):
            claim = self._claim_from_row(claim_row)
            balance_rows = tuple(
                self.db.execute(
                    "SELECT * FROM incentive_discovery_balances WHERE claim_digest=? "
                    "ORDER BY revision",
                    (claim.digest,),
                )
            )
            if not balance_rows or tuple(row["revision"] for row in balance_rows) != tuple(
                range(len(balance_rows))
            ):
                raise IncentiveCompositionStoreError(
                    "discovery balance revisions are not contiguous"
                )
            balances = tuple(self._balance_from_row(row) for row in balance_rows)
            if balances[0] != DiscoveryClaimBalance.open(claim):
                raise IncentiveCompositionStoreError(
                    "initial discovery balance is not open principal"
                )
            for index, (balance_row, current) in enumerate(
                zip(balance_rows, balances, strict=True)
            ):
                event = self.db.execute(
                    "SELECT event_type,block,block_hash,payload_json FROM "
                    "finite_debt_reward_events "
                    "WHERE event_digest=?",
                    (balance_row["reward_event_digest"],),
                ).fetchone()
                expected = (
                    "discovery_claim_issued"
                    if index == 0
                    else "discovery_claim_expired"
                    if current.status == "expired"
                    else "discovery_claim_cancelled"
                    if current.status == "cancelled"
                    else "composed_epoch_paid"
                )
                if event is None or event["event_type"] != expected:
                    raise IncentiveCompositionStoreError(
                        "discovery balance revision event differs"
                    )
                if index == 0:
                    continue
                prior = balances[index - 1]
                try:
                    payload = json.loads(event["payload_json"])
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise IncentiveCompositionStoreError(
                        f"discovery balance event is corrupt: {exc}"
                    ) from None
                if type(payload) is not dict:
                    raise IncentiveCompositionStoreError(
                        "discovery balance event payload differs"
                    )
                transition = {
                    "after_balance_digest": current.digest,
                    "before_balance_digest": prior.digest,
                    "claim_digest": claim.digest,
                }
                if event["event_type"] in {
                    "discovery_claim_expired",
                    "discovery_claim_cancelled",
                }:
                    try:
                        replayed = (
                            expire_discovery_balance(
                                claim, prior, at_block=event["block"]
                            )
                            if event["event_type"] == "discovery_claim_expired"
                            else cancel_discovery_balance(
                                claim,
                                prior,
                                at_block=event["block"],
                                reason=current.terminal_reason,
                            )
                        )
                    except IncentiveCompositionError as exc:
                        raise IncentiveCompositionStoreError(
                            f"discovery lifecycle event cannot replay: {exc}"
                        ) from None
                    expected_payload = {
                        "after_balance": current.to_dict(),
                        "after_balance_digest": current.digest,
                        "before_balance_digest": prior.digest,
                        "claim_digest": claim.digest,
                        "composition_activation_digest": self._activation_by_policy(
                            claim.policy_digest
                        ).digest,
                    }
                    if replayed != current or payload != expected_payload:
                        raise IncentiveCompositionStoreError(
                            "discovery lifecycle event payload differs"
                        )
                else:
                    transitions = payload.get("discovery_balance_transitions")
                    if (
                        type(transitions) is not list
                        or sum(item == transition for item in transitions) != 1
                    ):
                        raise IncentiveCompositionStoreError(
                            "discovery balance transition is not exactly authorized"
                        )
            for prior, current in zip(balances, balances[1:]):
                if (
                    prior.status != "open"
                    or current.paid_units < prior.paid_units
                    or current.forfeited_units < prior.forfeited_units
                    or current.remaining_units >= prior.remaining_units
                ):
                    raise IncentiveCompositionStoreError(
                        "discovery balance history regressed"
                    )
            states.append(DiscoveryClaimState(claim, balances[-1]))
        return tuple(states)

    def discovery_claim_states(
        self, *, policy_digest: str | None = None
    ) -> tuple[DiscoveryClaimState, ...]:
        self.core_store.reward_events()
        self._validated_wins()
        return self._discovery_states(policy_digest=policy_digest)

    def _insert_discovery_balance(
        self,
        balance: DiscoveryClaimBalance,
        *,
        revision: int,
        reward_event_digest: str,
    ) -> None:
        if not self.db.in_transaction or type(balance) is not DiscoveryClaimBalance:
            raise IncentiveCompositionStoreError(
                "discovery balance revision requires the owning transaction"
            )
        self.db.execute(
            "INSERT INTO incentive_discovery_balances(claim_digest,revision,"
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

    def reconcile_lifecycle(
        self,
        *,
        current_block: int,
        current_block_hash: str,
        eligible_hotkeys: Iterable[str],
    ) -> ComposedLifecycleChanges:
        """Atomically expire/depart both classes under one finalized authority."""

        height = _integer(current_block, "lifecycle block")
        authority_hash = _block_hash(current_block_hash, "lifecycle block hash")
        eligible = self.core_store._eligible_hotkeys(eligible_hotkeys)
        changed_core: list[DebtClaimState] = []
        changed_discovery: list[DiscoveryClaimState] = []
        with self._transaction():
            self._require_finalized_authority(height, authority_hash)
            activation = self.active_policy_activation(at_block=height)
            if activation is None:
                raise IncentiveCompositionStoreError(
                    "composition lifecycle requires an active policy"
                )
            core_states = self.core_store._claim_states(
                policy_digest=activation.policy.innovation_policy_digest
            )
            for state in core_states:
                if state.balance.status != "open":
                    continue
                try:
                    if height >= state.claim.expires_block:
                        updated = expire_claim_balance(
                            state.claim, state.balance, at_block=height
                        )
                        event_type = "claim_expired"
                    elif state.claim.hotkey not in eligible:
                        updated = cancel_claim_balance(
                            state.claim,
                            state.balance,
                            at_block=height,
                            reason="hotkey_departed",
                        )
                        event_type = "claim_cancelled"
                    else:
                        continue
                except FiniteDebtError as exc:
                    raise IncentiveCompositionStoreError(
                        f"core lifecycle cannot reconcile: {exc}"
                    ) from None
                if updated == state.balance:
                    continue
                event_digest = self.core_store._append_event(
                    event_type,
                    block=height,
                    block_hash=authority_hash,
                    payload={
                        "after_balance": updated.to_dict(),
                        "after_balance_digest": updated.digest,
                        "before_balance_digest": state.balance.digest,
                        "claim_digest": state.claim.digest,
                        "composition_activation_digest": activation.digest,
                    },
                )
                revision = self.db.execute(
                    "SELECT MAX(revision) AS value FROM finite_debt_claim_balances "
                    "WHERE claim_digest=?",
                    (state.claim.digest,),
                ).fetchone()["value"] + 1
                self.core_store._insert_balance(
                    updated,
                    revision=revision,
                    reward_event_digest=event_digest,
                )
                changed_core.append(DebtClaimState(state.claim, updated))

            for state in self._discovery_states(
                policy_digest=activation.policy.digest
            ):
                if state.balance.status != "open":
                    continue
                try:
                    if height >= state.claim.expires_block:
                        updated_discovery = expire_discovery_balance(
                            state.claim, state.balance, at_block=height
                        )
                        event_type = "discovery_claim_expired"
                    elif state.claim.hotkey not in eligible:
                        updated_discovery = cancel_discovery_balance(
                            state.claim,
                            state.balance,
                            at_block=height,
                            reason="hotkey_departed",
                        )
                        event_type = "discovery_claim_cancelled"
                    else:
                        continue
                except IncentiveCompositionError as exc:
                    raise IncentiveCompositionStoreError(
                        f"discovery lifecycle cannot reconcile: {exc}"
                    ) from None
                if updated_discovery == state.balance:
                    continue
                event_digest = self.core_store._append_event(
                    event_type,
                    block=height,
                    block_hash=authority_hash,
                    payload={
                        "after_balance": updated_discovery.to_dict(),
                        "after_balance_digest": updated_discovery.digest,
                        "before_balance_digest": state.balance.digest,
                        "claim_digest": state.claim.digest,
                        "composition_activation_digest": activation.digest,
                    },
                )
                revision = self.db.execute(
                    "SELECT MAX(revision) AS value FROM incentive_discovery_balances "
                    "WHERE claim_digest=?",
                    (state.claim.digest,),
                ).fetchone()["value"] + 1
                self._insert_discovery_balance(
                    updated_discovery,
                    revision=revision,
                    reward_event_digest=event_digest,
                )
                changed_discovery.append(
                    DiscoveryClaimState(state.claim, updated_discovery)
                )
        return ComposedLifecycleChanges(
            tuple(changed_core), tuple(changed_discovery)
        )

    @staticmethod
    def _epoch_coordinates(
        activation: IncentiveCompositionActivation,
        effective_block: int,
    ) -> tuple[int, int]:
        block = _integer(effective_block, "effective_block")
        elapsed = block - activation.activation_block
        if elapsed <= 0 or elapsed % activation.policy.epoch_blocks:
            raise IncentiveCompositionStoreError(
                "composed epoch is not an exact activated-policy cadence boundary"
            )
        return (
            elapsed // activation.policy.epoch_blocks,
            block - activation.policy.epoch_blocks,
        )

    @staticmethod
    def _require_projection_eligibility(
        projection: ComposedEpochProjection,
        eligible: frozenset[str],
    ) -> None:
        if projection.reserve_hotkey not in eligible:
            raise IncentiveCompositionStoreError(
                "composed reserve hotkey is absent from eligibility authority"
            )
        missing = {row.hotkey for row in projection.weights if row.units > 0} - eligible
        if missing:
            raise IncentiveCompositionStoreError(
                "composed projection contains an ineligible positive miner"
            )

    def _epoch_from_row(self, row: sqlite3.Row) -> IncentiveCompositionRewardEpoch:
        try:
            epoch = IncentiveCompositionRewardEpoch.from_dict(
                json.loads(row["epoch_json"])
            )
        except (
            IncentiveCompositionError,
            IncentiveCompositionStoreError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise IncentiveCompositionStoreError(
                f"composed reward epoch is corrupt: {exc}"
            ) from None
        if (
            epoch.digest != row["epoch_digest"]
            or epoch.chain_scope_digest != self.chain_scope_digest
            or epoch.activation_digest != row["activation_digest"]
            or epoch.composition_policy_digest
            != row["composition_policy_digest"]
            or epoch.core_policy_digest != row["core_policy_digest"]
            or epoch.epoch_index != row["epoch_index"]
            or epoch.start_block != row["start_block"]
            or epoch.effective_block != row["effective_block"]
            or epoch.effective_block_hash != row["effective_block_hash"]
            or epoch.projection.digest != row["projection_digest"]
            or _canonical_json(epoch.projection.to_dict()) != row["projection_json"]
            or epoch.publication_record_digest != row["publication_record_digest"]
            or epoch.payout_event_digest != row["payout_event_digest"]
            or epoch.projection.discovery_payout_units
            != row["discovery_payout_units"]
            or epoch.projection.innovation_payout_units != row["core_payout_units"]
            or epoch.projection.reserve_units != row["reserve_units"]
            or _canonical_json(epoch.to_dict()) != row["epoch_json"]
        ):
            raise IncentiveCompositionStoreError(
                "composed reward epoch differs from retained bytes"
            )
        allocations = tuple(
            self.db.execute(
                "SELECT reward_class,claim_digest,hotkey,units FROM "
                "incentive_composed_allocations WHERE epoch_digest=? "
                "ORDER BY reward_class,claim_digest",
                (epoch.digest,),
            )
        )
        expected_allocations = tuple(
            sorted(
                (
                    *(('discovery', item.claim_digest, item.hotkey, item.units)
                      for item in epoch.projection.discovery_allocations),
                    *(('core', item.claim_digest, item.hotkey, item.units)
                      for item in epoch.projection.innovation_allocations),
                )
            )
        )
        if tuple(
            (item["reward_class"], item["claim_digest"], item["hotkey"], item["units"])
            for item in allocations
        ) != expected_allocations:
            raise IncentiveCompositionStoreError(
                "composed epoch allocations differ"
            )
        event = self.db.execute(
            "SELECT event_type,block,block_hash,payload_json FROM "
            "finite_debt_reward_events WHERE event_digest=?",
            (epoch.payout_event_digest,),
        ).fetchone()
        if (
            event is None
            or event["event_type"] != "composed_epoch_paid"
            or event["block"] != epoch.effective_block
            or event["block_hash"] != epoch.effective_block_hash
        ):
            raise IncentiveCompositionStoreError(
                "composed epoch payout event differs"
            )
        try:
            payload = json.loads(event["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IncentiveCompositionStoreError(
                f"composed epoch payout event is corrupt: {exc}"
            ) from None
        core_inputs = set(epoch.projection.innovation_input_state_digests)
        core_prior: dict[str, tuple[DebtClaimState, int]] = {}
        matched_core: set[str] = set()
        for claim_row in self.db.execute(
            "SELECT * FROM finite_debt_claims WHERE policy_digest=? "
            "ORDER BY claim_digest",
            (epoch.core_policy_digest,),
        ):
            claim = self.core_store._claim_from_row(claim_row)
            for balance_row in self.db.execute(
                "SELECT * FROM finite_debt_claim_balances WHERE claim_digest=? "
                "ORDER BY revision",
                (claim.digest,),
            ):
                balance = self.core_store._balance_from_row(balance_row)
                state = DebtClaimState(claim, balance)
                if state.digest not in core_inputs:
                    continue
                if state.digest in matched_core or claim.digest in core_prior:
                    raise IncentiveCompositionStoreError(
                        "composed core input state is ambiguous"
                    )
                matched_core.add(state.digest)
                core_prior[claim.digest] = (state, balance_row["revision"])

        discovery_inputs = set(epoch.projection.discovery_input_state_digests)
        discovery_prior: dict[str, tuple[DiscoveryClaimState, int]] = {}
        matched_discovery: set[str] = set()
        for claim_row in self.db.execute(
            "SELECT * FROM incentive_discovery_claims "
            "WHERE composition_policy_digest=? ORDER BY claim_digest",
            (epoch.composition_policy_digest,),
        ):
            claim = self._claim_from_row(claim_row)
            for balance_row in self.db.execute(
                "SELECT * FROM incentive_discovery_balances WHERE claim_digest=? "
                "ORDER BY revision",
                (claim.digest,),
            ):
                balance = self._balance_from_row(balance_row)
                state = DiscoveryClaimState(claim, balance)
                if state.digest not in discovery_inputs:
                    continue
                if (
                    state.digest in matched_discovery
                    or claim.digest in discovery_prior
                ):
                    raise IncentiveCompositionStoreError(
                        "composed discovery input state is ambiguous"
                    )
                matched_discovery.add(state.digest)
                discovery_prior[claim.digest] = (
                    state,
                    balance_row["revision"],
                )
        if matched_core != core_inputs or matched_discovery != discovery_inputs:
            raise IncentiveCompositionStoreError(
                "composed epoch input states cannot be reopened"
            )
        try:
            replayed_core, replayed_discovery = apply_composed_epoch(
                tuple(row[0] for row in core_prior.values()),
                tuple(row[0] for row in discovery_prior.values()),
                epoch.projection,
            )
        except IncentiveCompositionError as exc:
            raise IncentiveCompositionStoreError(
                f"composed epoch transition cannot replay: {exc}"
            ) from None
        core_after = {state.claim.digest: state for state in replayed_core}
        discovery_after = {
            state.claim.digest: state for state in replayed_discovery
        }
        expected_core_transitions = [
            {
                "after_balance_digest": core_after[digest].balance.digest,
                "before_balance_digest": core_prior[digest][0].balance.digest,
                "claim_digest": digest,
            }
            for digest in sorted(core_prior)
            if core_prior[digest][0].balance != core_after[digest].balance
        ]
        expected_discovery_transitions = [
            {
                "after_balance_digest": discovery_after[digest].balance.digest,
                "before_balance_digest": discovery_prior[digest][0].balance.digest,
                "claim_digest": digest,
            }
            for digest in sorted(discovery_prior)
            if discovery_prior[digest][0].balance
            != discovery_after[digest].balance
        ]
        core_rows = tuple(
            self.db.execute(
                "SELECT b.*,c.policy_digest AS claim_policy_digest FROM "
                "finite_debt_claim_balances AS b JOIN finite_debt_claims AS c "
                "USING(claim_digest) WHERE b.reward_event_digest=? "
                "ORDER BY b.claim_digest",
                (epoch.payout_event_digest,),
            )
        )
        discovery_rows = tuple(
            self.db.execute(
                "SELECT b.*,c.composition_policy_digest AS claim_policy_digest "
                "FROM incentive_discovery_balances AS b JOIN "
                "incentive_discovery_claims AS c USING(claim_digest) "
                "WHERE b.reward_event_digest=? ORDER BY b.claim_digest",
                (epoch.payout_event_digest,),
            )
        )
        if (
            len(core_rows) != len(expected_core_transitions)
            or len(discovery_rows) != len(expected_discovery_transitions)
        ):
            raise IncentiveCompositionStoreError(
                "composed epoch balance revision set differs"
            )
        for balance_row, transition in zip(
            core_rows, expected_core_transitions, strict=True
        ):
            digest = transition["claim_digest"]
            prior = core_prior.get(digest)
            if (
                prior is None
                or balance_row["claim_digest"] != digest
                or balance_row["claim_policy_digest"] != epoch.core_policy_digest
                or balance_row["revision"] != prior[1] + 1
                or self.core_store._balance_from_row(balance_row)
                != core_after[digest].balance
            ):
                raise IncentiveCompositionStoreError(
                    "composed core balance revision differs"
                )
        for balance_row, transition in zip(
            discovery_rows, expected_discovery_transitions, strict=True
        ):
            digest = transition["claim_digest"]
            prior = discovery_prior.get(digest)
            if (
                prior is None
                or balance_row["claim_digest"] != digest
                or balance_row["claim_policy_digest"]
                != epoch.composition_policy_digest
                or balance_row["revision"] != prior[1] + 1
                or self._balance_from_row(balance_row)
                != discovery_after[digest].balance
            ):
                raise IncentiveCompositionStoreError(
                    "composed discovery balance revision differs"
                )
        expected_payload = {
            "activation_digest": epoch.activation_digest,
            "core_balance_transitions": expected_core_transitions,
            "discovery_balance_transitions": expected_discovery_transitions,
            "epoch_index": epoch.epoch_index,
            "projection_digest": epoch.projection.digest,
            "publication_record_digest": epoch.publication_record_digest,
        }
        if type(payload) is not dict or payload != expected_payload:
            raise IncentiveCompositionStoreError(
                "composed epoch payout payload differs"
            )
        activation = self._activation_by_policy(
            epoch.composition_policy_digest
        )
        expected_index, expected_start = self._epoch_coordinates(
            activation, epoch.effective_block
        )
        if (
            activation.digest != epoch.activation_digest
            or activation.policy.innovation_policy_digest
            != epoch.core_policy_digest
            or expected_index != epoch.epoch_index
            or expected_start != epoch.start_block
        ):
            raise IncentiveCompositionStoreError(
                "composed reward epoch cadence differs"
            )
        return epoch

    def reward_epochs(self) -> tuple[IncentiveCompositionRewardEpoch, ...]:
        self.core_store.reward_events()
        self._validated_wins()
        return tuple(
            self._epoch_from_row(row)
            for row in self.db.execute(
                "SELECT * FROM incentive_composed_epochs "
                "ORDER BY effective_block,epoch_digest"
            )
        )

    def _project_epoch(
        self,
        *,
        effective_block: int,
        eligible: frozenset[str],
        allow_retained: bool,
    ) -> ComposedEpochProjection:
        activation = self.active_policy_activation(at_block=effective_block)
        if activation is None:
            raise IncentiveCompositionStoreError(
                "incentive composition is not active"
            )
        epoch_index, _start = self._epoch_coordinates(activation, effective_block)
        existing = self.db.execute(
            "SELECT * FROM incentive_composed_epochs "
            "WHERE composition_policy_digest=? AND effective_block=?",
            (activation.policy.digest, effective_block),
        ).fetchone()
        if existing is not None:
            if not allow_retained:
                raise IncentiveCompositionStoreError(
                    "composed epoch is already closed"
                )
            epoch = self._epoch_from_row(existing)
            self._require_projection_eligibility(epoch.projection, eligible)
            return epoch.projection
        latest = self.db.execute(
            "SELECT MAX(epoch_index) AS value FROM incentive_composed_epochs "
            "WHERE composition_policy_digest=?",
            (activation.policy.digest,),
        ).fetchone()["value"]
        if (0 if latest is None else latest) != epoch_index - 1:
            raise IncentiveCompositionStoreError(
                "composed epochs must be projected and closed without gaps"
            )
        core = self.core_store._activation_by_policy(
            activation.policy.innovation_policy_digest
        )
        core_states = self.core_store._claim_states(
            policy_digest=core.policy.digest
        )
        discovery_states = self._discovery_states(
            policy_digest=activation.policy.digest
        )
        try:
            projection = project_composed_epoch(
                core.policy,
                activation.policy,
                effective_block=effective_block,
                innovation_states=core_states,
                discovery_states=discovery_states,
            )
        except IncentiveCompositionError as exc:
            raise IncentiveCompositionStoreError(
                f"composed epoch cannot project: {exc}"
            ) from None
        self._require_projection_eligibility(projection, eligible)
        return projection

    def project_epoch(
        self,
        *,
        effective_block: int,
        eligible_hotkeys: Iterable[str],
    ) -> ComposedEpochProjection:
        """Build a read-only two-class projection; no balance is debited."""

        eligible = self.core_store._eligible_hotkeys(eligible_hotkeys)
        return self._project_epoch(
            effective_block=effective_block,
            eligible=eligible,
            allow_retained=True,
        )

    def close_confirmed_epoch(
        self,
        projection: ComposedEpochProjection,
        *,
        confirmation,
        eligible_hotkeys: Iterable[str],
    ) -> IncentiveCompositionRewardEpoch:
        """Debit both classes once after a retained, exact chain readback."""

        from optima.chain.debt_publication import (
            PUBLICATION_KIND_COMPOSED,
            ConfirmedDebtWeightPublication,
            DebtPublicationError,
            SQLiteDebtWeightPublicationJournal,
            reopen_confirmed_debt_publication,
            retain_confirmed_debt_publication,
        )

        if type(projection) is not ComposedEpochProjection:
            raise IncentiveCompositionStoreError(
                "composed projection is not exactly typed"
            )
        if type(confirmation) is not ConfirmedDebtWeightPublication:
            raise IncentiveCompositionStoreError(
                "composed close requires a typed publication confirmation"
            )
        expected = projection.digest
        publication = confirmation.digest
        height = projection.effective_block
        authority_hash = confirmation.effective_block_hash
        eligible = self.core_store._eligible_hotkeys(eligible_hotkeys)

        with self._transaction():
            try:
                journal = SQLiteDebtWeightPublicationJournal.reopen_from_head(
                    self.db, transaction=self._transaction
                )
                retained_record, binding = journal.retained_authority(
                    confirmation.publication_record.digest
                )
                confirmation.validate_binding(binding)
            except DebtPublicationError as exc:
                raise IncentiveCompositionStoreError(
                    f"composed publication authority failed: {exc}"
                ) from None
            if (
                retained_record != confirmation.publication_record
                or binding.publication_kind != PUBLICATION_KIND_COMPOSED
                or binding.weight_projection.chain_scope_digest
                != self.chain_scope_digest
                or binding.economic_projection.to_dict() != projection.to_dict()
            ):
                raise IncentiveCompositionStoreError(
                    "composed publication differs from retained projection authority"
                )
            retained = self.db.execute(
                "SELECT * FROM incentive_composed_epochs "
                "WHERE composition_policy_digest=? AND effective_block=?",
                (projection.composition_policy_digest, height),
            ).fetchone()
            if retained is not None:
                epoch = self._epoch_from_row(retained)
                if (
                    epoch.projection.to_dict() != projection.to_dict()
                    or epoch.projection.digest != expected
                    or epoch.effective_block_hash != authority_hash
                    or epoch.publication_record_digest != publication
                ):
                    raise IncentiveCompositionStoreError(
                        "confirmed composed epoch retry differs from retained closure"
                    )
                self._require_projection_eligibility(epoch.projection, eligible)
                return epoch

            activation = self.active_policy_activation(at_block=height)
            if (
                activation is None
                or activation.policy.digest
                != projection.composition_policy_digest
            ):
                raise IncentiveCompositionStoreError(
                    "composed projection policy is not active at its boundary"
                )
            if binding.activation_digest != activation.digest:
                raise IncentiveCompositionStoreError(
                    "composed publication differs from active policy authority"
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
                raise IncentiveCompositionStoreError(
                    "composed confirmation is newer than retained finalized intake"
                )
            prior_row = self.db.execute(
                "SELECT * FROM incentive_composed_epochs WHERE "
                "composition_policy_digest=? AND effective_block<? "
                "ORDER BY effective_block DESC LIMIT 1",
                (projection.composition_policy_digest, height),
            ).fetchone()
            if prior_row is not None:
                prior_epoch = self._epoch_from_row(prior_row)
                try:
                    prior_confirmation = reopen_confirmed_debt_publication(
                        self.db, prior_epoch.publication_record_digest
                    )
                except DebtPublicationError as exc:
                    raise IncentiveCompositionStoreError(
                        f"prior composed publication cannot reopen: {exc}"
                    ) from None
                if (
                    confirmation.readback.block
                    < prior_confirmation.readback.block
                    + activation.policy.epoch_blocks
                ):
                    raise IncentiveCompositionStoreError(
                        "composed catch-up would compress live emission epochs"
                    )
            epoch_index, start_block = self._epoch_coordinates(activation, height)
            authoritative = self._project_epoch(
                effective_block=height,
                eligible=eligible,
                allow_retained=False,
            )
            if authoritative.to_dict() != projection.to_dict():
                raise IncentiveCompositionStoreError(
                    "composed balances changed after projection was built"
                )
            try:
                retain_confirmed_debt_publication(self.db, confirmation)
            except DebtPublicationError as exc:
                raise IncentiveCompositionStoreError(
                    f"composed confirmation cannot be retained: {exc}"
                ) from None
            core_states = self.core_store._claim_states(
                policy_digest=activation.policy.innovation_policy_digest
            )
            discovery_states = self._discovery_states(
                policy_digest=activation.policy.digest
            )
            try:
                updated_core, updated_discovery = apply_composed_epoch(
                    core_states,
                    discovery_states,
                    authoritative,
                )
            except IncentiveCompositionError as exc:
                raise IncentiveCompositionStoreError(
                    f"composed epoch cannot apply: {exc}"
                ) from None
            core_before = {row.claim.digest: row for row in core_states}
            core_after = {row.claim.digest: row for row in updated_core}
            discovery_before = {
                row.claim.digest: row for row in discovery_states
            }
            discovery_after = {
                row.claim.digest: row for row in updated_discovery
            }
            event_digest = self.core_store._append_event(
                "composed_epoch_paid",
                block=height,
                block_hash=authority_hash,
                payload={
                    "activation_digest": activation.digest,
                    "core_balance_transitions": [
                        {
                            "after_balance_digest": core_after[digest].balance.digest,
                            "before_balance_digest": core_before[digest].balance.digest,
                            "claim_digest": digest,
                        }
                        for digest in sorted(core_before)
                        if core_before[digest].balance
                        != core_after[digest].balance
                    ],
                    "discovery_balance_transitions": [
                        {
                            "after_balance_digest": discovery_after[
                                digest
                            ].balance.digest,
                            "before_balance_digest": discovery_before[
                                digest
                            ].balance.digest,
                            "claim_digest": digest,
                        }
                        for digest in sorted(discovery_before)
                        if discovery_before[digest].balance
                        != discovery_after[digest].balance
                    ],
                    "epoch_index": epoch_index,
                    "projection_digest": authoritative.digest,
                    "publication_record_digest": publication,
                },
            )
            epoch = IncentiveCompositionRewardEpoch(
                self.chain_scope_digest,
                activation.digest,
                activation.policy.digest,
                activation.policy.innovation_policy_digest,
                epoch_index,
                start_block,
                height,
                authority_hash,
                authoritative,
                publication,
                event_digest,
            )
            self.db.execute(
                "INSERT INTO incentive_composed_epochs(epoch_digest,chain_scope_digest,"
                "activation_digest,composition_policy_digest,core_policy_digest,"
                "epoch_index,start_block,effective_block,effective_block_hash,"
                "projection_digest,projection_json,publication_record_digest,"
                "payout_event_digest,discovery_payout_units,core_payout_units,"
                "reserve_units,epoch_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    epoch.digest,
                    self.chain_scope_digest,
                    activation.digest,
                    activation.policy.digest,
                    activation.policy.innovation_policy_digest,
                    epoch_index,
                    start_block,
                    height,
                    authority_hash,
                    authoritative.digest,
                    _canonical_json(authoritative.to_dict()),
                    publication,
                    event_digest,
                    authoritative.discovery_payout_units,
                    authoritative.innovation_payout_units,
                    authoritative.reserve_units,
                    _canonical_json(epoch.to_dict()),
                ),
            )
            for reward_class, allocations in (
                ("discovery", authoritative.discovery_allocations),
                ("core", authoritative.innovation_allocations),
            ):
                for allocation in allocations:
                    self.db.execute(
                        "INSERT INTO incentive_composed_allocations(epoch_digest,"
                        "reward_class,claim_digest,hotkey,units) VALUES(?,?,?,?,?)",
                        (
                            epoch.digest,
                            reward_class,
                            allocation.claim_digest,
                            allocation.hotkey,
                            allocation.units,
                        ),
                    )
            for digest in sorted(core_before):
                if core_before[digest].balance == core_after[digest].balance:
                    continue
                revision = self.db.execute(
                    "SELECT MAX(revision) AS value FROM finite_debt_claim_balances "
                    "WHERE claim_digest=?",
                    (digest,),
                ).fetchone()["value"] + 1
                self.core_store._insert_balance(
                    core_after[digest].balance,
                    revision=revision,
                    reward_event_digest=event_digest,
                )
            for digest in sorted(discovery_before):
                if (
                    discovery_before[digest].balance
                    == discovery_after[digest].balance
                ):
                    continue
                revision = self.db.execute(
                    "SELECT MAX(revision) AS value FROM incentive_discovery_balances "
                    "WHERE claim_digest=?",
                    (digest,),
                ).fetchone()["value"] + 1
                self._insert_discovery_balance(
                    discovery_after[digest].balance,
                    revision=revision,
                    reward_event_digest=event_digest,
                )
        return epoch


__all__ = [
    "ComposedLifecycleChanges",
    "IncentiveCompositionActivation",
    "IncentiveCompositionRewardEpoch",
    "IncentiveCompositionStore",
    "IncentiveCompositionStoreError",
    "ReviewPendingDiscoveryWin",
    "ReviewedDiscoveryDispositionRecord",
    "SelectedIncentiveActivationApproval",
    "SCHEMA_VERSION",
    "SELECTED_CORE_SELECTION_REPORT_DIGEST",
    "SELECTED_DISCOVERY_CAP_UNITS",
    "SELECTED_DISCOVERY_LIFETIME_BLOCKS",
    "SELECTED_EPOCH_BLOCKS",
    "SELECTED_PER_AWARD_PRINCIPAL_CAP_EPOCHS",
    "SELECTED_RESERVE_PPM",
    "SELECTED_SELECTION_REPORT_DIGEST",
    "migrate_schema4_to5",
]
