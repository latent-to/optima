"""Pure, content-addressed emissions projection.

This module owns no chain client, wallet, database, or settlement transition.  It
accepts only crowns whose retained evidence has already been reopened by the
settlement authority and either returns one complete integer weight projection or
fails without a partial vector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from optima.stack_identity import canonical_digest, require_sha256_hex
from optima.stack_manifest import EvaluationStackManifest
from optima.target_catalog import TargetCatalog, TargetResolutionError


POLICY_SCHEMA_VERSION = 1
POLICY_VERSION = "optima.emissions.v1"
WEIGHT_PPM = 1_000_000
_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}\Z")
_HOTKEY = re.compile(r"[^\s]{1,256}\Z")


class EconomicsError(ValueError):
    """The policy, authority, or complete reward projection is invalid."""


def _digest(value: object, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise EconomicsError(str(exc)) from None
    if result == "0" * 64:
        raise EconomicsError(f"{field} must not be the all-zero digest")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EconomicsError(f"{field} must be an integer >= {minimum}")
    return value


def _hotkey(value: object, field: str = "hotkey") -> str:
    if not isinstance(value, str) or _HOTKEY.fullmatch(value) is None:
        raise EconomicsError(f"{field} is malformed")
    return value


def _strict(value: object, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise EconomicsError(f"{name} fields do not match the schema")
    return value


@dataclass(frozen=True)
class EmissionsPolicyManifest:
    """Validator-consensus parameters for one deterministic reward policy."""

    half_life_blocks: int
    discovery_lifetime_blocks: int
    discovery_pool_ppm: int
    schema_version: int = POLICY_SCHEMA_VERSION
    policy_version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        _integer(self.half_life_blocks, "half_life_blocks", minimum=1)
        _integer(
            self.discovery_lifetime_blocks,
            "discovery_lifetime_blocks",
            minimum=1,
        )
        _integer(self.discovery_pool_ppm, "discovery_pool_ppm")
        if self.discovery_pool_ppm >= WEIGHT_PPM:
            raise EconomicsError("discovery_pool_ppm must leave standing reward capacity")
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise EconomicsError("emissions policy schema_version is unsupported")
        if self.policy_version != POLICY_VERSION:
            raise EconomicsError("emissions policy_version is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "discovery_lifetime_blocks": self.discovery_lifetime_blocks,
            "discovery_pool_ppm": self.discovery_pool_ppm,
            "half_life_blocks": self.half_life_blocks,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EmissionsPolicyManifest":
        row = _strict(
            value,
            {
                "discovery_lifetime_blocks",
                "discovery_pool_ppm",
                "half_life_blocks",
                "policy_version",
                "schema_version",
            },
            "emissions policy",
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.economics.policy", self.to_dict())


@dataclass(frozen=True)
class MetagraphMember:
    uid: int
    hotkey: str

    def __post_init__(self) -> None:
        _integer(self.uid, "metagraph uid")
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey, "metagraph hotkey"))

    def to_dict(self) -> dict[str, object]:
        return {"hotkey": self.hotkey, "uid": self.uid}


@dataclass(frozen=True)
class RewardProjectionContext:
    """Current control-plane and metagraph authority bound into the vector."""

    chain_scope_digest: str
    validator_hotkey: str
    stack_generation: int
    current_block: int
    current_block_hash: str
    metagraph_members: tuple[MetagraphMember, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chain_scope_digest",
            _digest(self.chain_scope_digest, "chain_scope_digest"),
        )
        object.__setattr__(self, "validator_hotkey", _hotkey(self.validator_hotkey))
        _integer(self.stack_generation, "stack_generation")
        _integer(self.current_block, "current_block")
        if (
            not isinstance(self.current_block_hash, str)
            or _BLOCK_HASH.fullmatch(self.current_block_hash) is None
        ):
            raise EconomicsError("current_block_hash is malformed")
        members = tuple(self.metagraph_members)
        if any(type(row) is not MetagraphMember for row in members):
            raise EconomicsError("metagraph_members must be exactly typed")
        members = tuple(sorted(members, key=lambda row: (row.uid, row.hotkey)))
        if (
            not members
            or len({row.uid for row in members}) != len(members)
            or len({row.hotkey for row in members}) != len(members)
        ):
            raise EconomicsError("metagraph membership is empty or duplicated")
        if self.validator_hotkey not in {row.hotkey for row in members}:
            raise EconomicsError("validator is absent from the current metagraph")
        object.__setattr__(self, "metagraph_members", members)

    @property
    def eligible_hotkeys(self) -> frozenset[str]:
        return frozenset(row.hotkey for row in self.metagraph_members)

    @property
    def metagraph_digest(self) -> str:
        return canonical_digest(
            "optima.economics.metagraph-membership",
            {
                "block": self.current_block,
                "block_hash": self.current_block_hash,
                "chain_scope_digest": self.chain_scope_digest,
                "members": [row.to_dict() for row in self.metagraph_members],
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_scope_digest": self.chain_scope_digest,
            "current_block": self.current_block,
            "current_block_hash": self.current_block_hash,
            "metagraph_digest": self.metagraph_digest,
            "metagraph_members": [row.to_dict() for row in self.metagraph_members],
            "stack_generation": self.stack_generation,
            "validator_hotkey": self.validator_hotkey,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.economics.projection-context", self.to_dict())


@dataclass(frozen=True)
class GlobalRewardProjectionContext:
    """Chain authority shared by every arena in one global projection."""

    chain_scope_digest: str
    validator_hotkey: str
    current_block: int
    current_block_hash: str
    metagraph_members: tuple[MetagraphMember, ...]

    def __post_init__(self) -> None:
        single = RewardProjectionContext(
            self.chain_scope_digest,
            self.validator_hotkey,
            0,
            self.current_block,
            self.current_block_hash,
            self.metagraph_members,
        )
        object.__setattr__(self, "chain_scope_digest", single.chain_scope_digest)
        object.__setattr__(self, "validator_hotkey", single.validator_hotkey)
        object.__setattr__(self, "metagraph_members", single.metagraph_members)

    @property
    def eligible_hotkeys(self) -> frozenset[str]:
        return frozenset(row.hotkey for row in self.metagraph_members)

    @property
    def metagraph_digest(self) -> str:
        return canonical_digest(
            "optima.economics.metagraph-membership",
            {
                "block": self.current_block,
                "block_hash": self.current_block_hash,
                "chain_scope_digest": self.chain_scope_digest,
                "members": [row.to_dict() for row in self.metagraph_members],
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_scope_digest": self.chain_scope_digest,
            "current_block": self.current_block,
            "current_block_hash": self.current_block_hash,
            "metagraph_digest": self.metagraph_digest,
            "metagraph_members": [row.to_dict() for row in self.metagraph_members],
            "validator_hotkey": self.validator_hotkey,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.economics.global-context", self.to_dict())


@dataclass(frozen=True)
class StandingRewardClaim:
    """One reopened crown for one currently active registered target."""

    arena_digest: str
    target_id: str
    target_spec_digest: str
    contribution_digest: str
    hotkey: str
    speedup_ppm: int
    crowned_block: int
    retained_evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id:
            raise EconomicsError("standing target_id is malformed")
        for field in (
            "arena_digest",
            "target_spec_digest",
            "contribution_digest",
            "retained_evidence_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        _integer(self.speedup_ppm, "speedup_ppm", minimum=WEIGHT_PPM + 1)
        _integer(self.crowned_block, "crowned_block")

    @property
    def family_id(self) -> str:
        return canonical_digest(
            "optima.economics.standing-family",
            {
                "arena_digest": self.arena_digest,
                "target_id": self.target_id,
                "target_spec_digest": self.target_spec_digest,
            },
        )

    def credit_at(self, block: int, policy: EmissionsPolicyManifest) -> int:
        _integer(block, "credit block")
        if block < self.crowned_block:
            raise EconomicsError("crown is newer than projection authority")
        age = block - self.crowned_block
        improvement = self.speedup_ppm - WEIGHT_PPM
        return improvement * policy.half_life_blocks // (
            policy.half_life_blocks + age
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "contribution_digest": self.contribution_digest,
            "crowned_block": self.crowned_block,
            "hotkey": self.hotkey,
            "retained_evidence_digest": self.retained_evidence_digest,
            "speedup_ppm": self.speedup_ppm,
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "StandingRewardClaim":
        row = _strict(value, set(cls.__dataclass_fields__), "standing reward claim")
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.economics.standing-claim", self.to_dict())


@dataclass(frozen=True)
class DiscoveryBountyClaim:
    """One non-renewable, expiring discovery reward claim."""

    proposal_digest: str
    retained_evidence_digest: str
    hotkey: str
    bounty_units: int
    awarded_block: int

    def __post_init__(self) -> None:
        for field in ("proposal_digest", "retained_evidence_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hotkey", _hotkey(self.hotkey))
        _integer(self.bounty_units, "bounty_units", minimum=1)
        _integer(self.awarded_block, "awarded_block")

    def live_at(self, block: int, policy: EmissionsPolicyManifest) -> bool:
        _integer(block, "bounty block")
        if block < self.awarded_block:
            raise EconomicsError("discovery claim is newer than projection authority")
        return block - self.awarded_block < policy.discovery_lifetime_blocks

    def to_dict(self) -> dict[str, object]:
        return {
            "awarded_block": self.awarded_block,
            "bounty_units": self.bounty_units,
            "hotkey": self.hotkey,
            "proposal_digest": self.proposal_digest,
            "retained_evidence_digest": self.retained_evidence_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryBountyClaim":
        row = _strict(value, set(cls.__dataclass_fields__), "discovery bounty claim")
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.economics.discovery-claim", self.to_dict())


@dataclass(frozen=True)
class ArenaRewardAuthority:
    """One arena's complete active stack generation and reopened crowns."""

    catalog: TargetCatalog
    stack: EvaluationStackManifest
    stack_generation: int
    standing_claims: tuple[StandingRewardClaim, ...]

    def __post_init__(self) -> None:
        if type(self.catalog) is not TargetCatalog:
            raise EconomicsError("arena reward catalog is not exactly typed")
        if type(self.stack) is not EvaluationStackManifest:
            raise EconomicsError("arena reward stack is not exactly typed")
        _integer(self.stack_generation, "arena stack_generation")
        claims = tuple(self.standing_claims)
        if any(type(row) is not StandingRewardClaim for row in claims):
            raise EconomicsError("arena standing claims are not exactly typed")
        object.__setattr__(self, "standing_claims", claims)

    @property
    def arena_digest(self) -> str:
        return self.stack.arena_digest

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "catalog_digest": self.catalog.digest,
            "stack_digest": self.stack.digest,
            "stack_generation": self.stack_generation,
            "standing_claims": [
                row.digest for row in sorted(self.standing_claims, key=lambda row: row.target_id)
            ],
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.economics.arena-authority", self.to_dict())


@dataclass(frozen=True)
class StandingFamilyCredit:
    arena_digest: str
    family_id: str
    target_id: str
    claim_digest: str
    hotkey: str
    credit: int

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "claim_digest": self.claim_digest,
            "credit": self.credit,
            "family_id": self.family_id,
            "hotkey": self.hotkey,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class DiscoveryBountyCredit:
    claim_digest: str
    hotkey: str
    credit: int

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_digest": self.claim_digest,
            "credit": self.credit,
            "hotkey": self.hotkey,
        }


@dataclass(frozen=True)
class HotkeyWeight:
    hotkey: str
    weight_ppm: int

    def to_dict(self) -> dict[str, object]:
        return {"hotkey": self.hotkey, "weight_ppm": self.weight_ppm}


@dataclass(frozen=True)
class RewardProjection:
    policy_digest: str
    catalog_digest: str
    evaluation_stack_digest: str
    context: RewardProjectionContext
    standing: tuple[StandingFamilyCredit, ...]
    discovery: tuple[DiscoveryBountyCredit, ...]
    expired_discovery_claims: tuple[str, ...]
    weights: tuple[HotkeyWeight, ...]

    def __post_init__(self) -> None:
        for field in ("policy_digest", "catalog_digest", "evaluation_stack_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if type(self.context) is not RewardProjectionContext:
            raise EconomicsError("reward projection context is not exactly typed")
        if any(type(row) is not StandingFamilyCredit for row in self.standing):
            raise EconomicsError("standing credits are not exactly typed")
        if any(type(row) is not DiscoveryBountyCredit for row in self.discovery):
            raise EconomicsError("discovery credits are not exactly typed")
        if any(type(row) is not HotkeyWeight for row in self.weights):
            raise EconomicsError("hotkey weights are not exactly typed")
        if (
            tuple((row.arena_digest, row.target_id) for row in self.standing)
            != tuple(sorted((row.arena_digest, row.target_id) for row in self.standing))
            or tuple(row.hotkey for row in self.weights)
            != tuple(sorted(row.hotkey for row in self.weights))
            or len({row.hotkey for row in self.weights}) != len(self.weights)
            or sum(row.weight_ppm for row in self.weights) != WEIGHT_PPM
        ):
            raise EconomicsError("reward projection is not canonical or normalized")
        for digest in self.expired_discovery_claims:
            _digest(digest, "expired discovery claim")

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_digest": self.catalog_digest,
            "context": self.context.to_dict(),
            "discovery": [row.to_dict() for row in self.discovery],
            "evaluation_stack_digest": self.evaluation_stack_digest,
            "expired_discovery_claims": list(self.expired_discovery_claims),
            "policy_digest": self.policy_digest,
            "standing": [row.to_dict() for row in self.standing],
            "weights": [row.to_dict() for row in self.weights],
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.economics.reward-projection", self.to_dict())

    @property
    def weights_by_hotkey(self) -> Mapping[str, int]:
        return {row.hotkey: row.weight_ppm for row in self.weights}


@dataclass(frozen=True)
class GlobalRewardProjection:
    policy_digest: str
    context: GlobalRewardProjectionContext
    arena_authority_digests: tuple[str, ...]
    standing: tuple[StandingFamilyCredit, ...]
    discovery: tuple[DiscoveryBountyCredit, ...]
    expired_discovery_claims: tuple[str, ...]
    weights: tuple[HotkeyWeight, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_digest", _digest(self.policy_digest, "policy_digest"))
        if type(self.context) is not GlobalRewardProjectionContext:
            raise EconomicsError("global reward context is not exactly typed")
        for digest in self.arena_authority_digests:
            _digest(digest, "arena authority digest")
        if tuple(self.arena_authority_digests) != tuple(sorted(self.arena_authority_digests)):
            raise EconomicsError("arena authorities are not canonical")
        if tuple((row.arena_digest, row.target_id) for row in self.standing) != tuple(
            sorted((row.arena_digest, row.target_id) for row in self.standing)
        ):
            raise EconomicsError("global standing families are not canonical")
        if tuple(row.hotkey for row in self.weights) != tuple(sorted(row.hotkey for row in self.weights)):
            raise EconomicsError("global weights are not canonical")
        if len({row.hotkey for row in self.weights}) != len(self.weights) or sum(
            row.weight_ppm for row in self.weights
        ) != WEIGHT_PPM:
            raise EconomicsError("global weights are not exactly normalized")

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_authority_digests": list(self.arena_authority_digests),
            "context": self.context.to_dict(),
            "discovery": [row.to_dict() for row in self.discovery],
            "expired_discovery_claims": list(self.expired_discovery_claims),
            "policy_digest": self.policy_digest,
            "standing": [row.to_dict() for row in self.standing],
            "weights": [row.to_dict() for row in self.weights],
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.economics.global-reward-projection", self.to_dict())

    @property
    def weights_by_hotkey(self) -> Mapping[str, int]:
        return {row.hotkey: row.weight_ppm for row in self.weights}


def _allocate_pool(credits: Mapping[str, int], pool: int) -> dict[str, int]:
    positive = {hotkey: credit for hotkey, credit in credits.items() if credit > 0}
    total = sum(positive.values())
    if total <= 0:
        raise EconomicsError("a non-empty reward pool has no positive credit")
    result: dict[str, int] = {}
    remainders = []
    for hotkey in sorted(positive):
        quotient, remainder = divmod(positive[hotkey] * pool, total)
        result[hotkey] = quotient
        remainders.append((remainder, hotkey))
    missing = pool - sum(result.values())
    for _remainder, hotkey in sorted(remainders, key=lambda row: (-row[0], row[1]))[
        :missing
    ]:
        result[hotkey] += 1
    return result


def project_global_rewards(
    policy: EmissionsPolicyManifest,
    context: GlobalRewardProjectionContext,
    arenas: Iterable[ArenaRewardAuthority],
    discovery_claims: Iterable[DiscoveryBountyClaim] = (),
) -> GlobalRewardProjection:
    """Pool every registered arena before producing one indivisible vector."""

    if type(policy) is not EmissionsPolicyManifest:
        raise EconomicsError("policy is not exactly typed")
    if type(context) is not GlobalRewardProjectionContext:
        raise EconomicsError("global projection context is not exactly typed")
    authorities = tuple(arenas)
    if not authorities or any(type(row) is not ArenaRewardAuthority for row in authorities):
        raise EconomicsError("global projection requires typed arena authorities")
    if len({row.arena_digest for row in authorities}) != len(authorities):
        raise EconomicsError("global projection contains duplicate arenas")
    authorities = tuple(sorted(authorities, key=lambda row: row.arena_digest))
    eligible = context.eligible_hotkeys
    family_credits: list[StandingFamilyCredit] = []
    standing_by_hotkey: dict[str, int] = {}
    for authority in authorities:
        catalog, stack = authority.catalog, authority.stack
        if stack.catalog_digest != catalog.digest or stack.catalog_snapshot != catalog.snapshot():
            raise EconomicsError("evaluation stack and reward catalog differ")
        try:
            active_targets = catalog.validate_active_targets(stack.entries)
        except TargetResolutionError as exc:
            raise EconomicsError(
                f"active reward families overlap or are incomplete: {exc}"
            ) from None
        if not active_targets:
            raise EconomicsError("every registered arena requires an active crown")
        by_target = {row.target_id: row for row in authority.standing_claims}
        if len(by_target) != len(authority.standing_claims) or set(by_target) != set(active_targets):
            raise EconomicsError("every active target requires exactly one standing claim")
        for target_id in active_targets:
            claim = by_target[target_id]
            contribution = stack.entries[target_id]
            if claim.arena_digest != stack.arena_digest:
                raise EconomicsError(f"standing claim for {target_id!r} names another arena")
            if (
                claim.target_spec_digest != catalog.target_spec_digest(target_id)
                or claim.target_spec_digest != contribution.target_spec_digest
                or claim.contribution_digest != contribution.digest
            ):
                raise EconomicsError(f"standing claim for {target_id!r} is stale or incompatible")
            if claim.hotkey not in eligible:
                raise EconomicsError(f"standing hotkey for {target_id!r} left the metagraph")
            credit = claim.credit_at(context.current_block, policy)
            family_credits.append(
                StandingFamilyCredit(
                    stack.arena_digest,
                    claim.family_id,
                    target_id,
                    claim.digest,
                    claim.hotkey,
                    credit,
                )
            )
            standing_by_hotkey[claim.hotkey] = standing_by_hotkey.get(claim.hotkey, 0) + credit
    if not any(standing_by_hotkey.values()):
        raise EconomicsError("all standing crown credit has decayed to zero")

    discoveries = tuple(discovery_claims)
    if any(type(row) is not DiscoveryBountyClaim for row in discoveries):
        raise EconomicsError("discovery claims are not exactly typed")
    if (
        len({row.proposal_digest for row in discoveries}) != len(discoveries)
        or len({row.retained_evidence_digest for row in discoveries}) != len(discoveries)
        or len({row.digest for row in discoveries}) != len(discoveries)
    ):
        raise EconomicsError("discovery claims are renewed or duplicated")
    live = []
    expired = []
    discovery_by_hotkey: dict[str, int] = {}
    for claim in sorted(discoveries, key=lambda row: row.digest):
        if claim.live_at(context.current_block, policy):
            if claim.hotkey not in eligible:
                raise EconomicsError("a live discovery hotkey left the metagraph")
            live.append(
                DiscoveryBountyCredit(claim.digest, claim.hotkey, claim.bounty_units)
            )
            discovery_by_hotkey[claim.hotkey] = (
                discovery_by_hotkey.get(claim.hotkey, 0) + claim.bounty_units
            )
        else:
            expired.append(claim.digest)

    discovery_pool = policy.discovery_pool_ppm if live else 0
    if live and discovery_pool == 0:
        raise EconomicsError("live discovery claims exist while bounties are disabled")
    standing_pool = WEIGHT_PPM - discovery_pool
    combined = _allocate_pool(standing_by_hotkey, standing_pool)
    if live:
        for hotkey, value in _allocate_pool(discovery_by_hotkey, discovery_pool).items():
            combined[hotkey] = combined.get(hotkey, 0) + value
    weights = tuple(
        HotkeyWeight(hotkey, combined[hotkey]) for hotkey in sorted(combined)
    )
    return GlobalRewardProjection(
        policy.digest,
        context,
        tuple(sorted(row.digest for row in authorities)),
        tuple(family_credits),
        tuple(live),
        tuple(expired),
        weights,
    )


def project_rewards(
    policy: EmissionsPolicyManifest,
    catalog: TargetCatalog,
    stack: EvaluationStackManifest,
    context: RewardProjectionContext,
    standing_claims: Iterable[StandingRewardClaim],
    discovery_claims: Iterable[DiscoveryBountyClaim] = (),
) -> RewardProjection:
    """Compatibility wrapper for a single arena; production uses the global API."""

    if type(context) is not RewardProjectionContext:
        raise EconomicsError("projection context is not exactly typed")
    authority = ArenaRewardAuthority(
        catalog, stack, context.stack_generation, tuple(standing_claims)
    )
    global_context = GlobalRewardProjectionContext(
        context.chain_scope_digest,
        context.validator_hotkey,
        context.current_block,
        context.current_block_hash,
        context.metagraph_members,
    )
    result = project_global_rewards(
        policy, global_context, (authority,), discovery_claims
    )
    return RewardProjection(
        result.policy_digest,
        catalog.digest,
        stack.digest,
        context,
        result.standing,
        result.discovery,
        result.expired_discovery_claims,
        result.weights,
    )


__all__ = [
    "ArenaRewardAuthority",
    "DiscoveryBountyClaim",
    "EconomicsError",
    "EmissionsPolicyManifest",
    "GlobalRewardProjection",
    "GlobalRewardProjectionContext",
    "HotkeyWeight",
    "MetagraphMember",
    "POLICY_SCHEMA_VERSION",
    "POLICY_VERSION",
    "RewardProjection",
    "RewardProjectionContext",
    "StandingRewardClaim",
    "WEIGHT_PPM",
    "project_global_rewards",
    "project_rewards",
]
