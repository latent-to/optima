"""Validator-owned contribution target identity.

The manifest answers which implementation rows should be loaded.  This module
answers the separate question: which smallest validator-registered semantic
delta does that set of rows propose to replace?

The catalog is deliberately policy-only.  It contains no score, champion,
settlement, chain, qualification, execution-trust, or whole-serving authority.
Untrusted implementation code still runs as a complete isolated engine; its
execution form does not create or deny a contribution identity.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Mapping

from optima.artifact_provider import (
    ARTIFACT_PROVIDERS,
    CUTE_CUBIN_MANIFEST_FEATURE,
    CUTE_CUBIN_REBUILD_FEATURE,
    ArtifactProviderPolicyError,
)
from optima.manifest import CompetitionEntry, DEFAULT_VARIANT, Manifest
from optima.stack_identity import canonical_digest


class TargetKind(str, Enum):
    SLOT = "slot"
    ATOMIC = "atomic"


# Feature names are validator vocabulary, never miner-selected permissions.
# Dynamic/unknown manifest fields and rebuild steps are still observed, but no
# registered target admits them until validator code names the capability here.
FEATURE_ENTRY = "entry"
FEATURE_VARIANTS = "variants"
FEATURE_PREPARE = "prepare"
FEATURE_SETUP = "setup"
FEATURE_OVERRIDE = "override"
FEATURE_CUDA_SOURCES = "cuda_sources"
FEATURE_AOT_CUTE_OBJECT = CUTE_CUBIN_MANIFEST_FEATURE
FEATURE_REBUILD_BUILD_CUDA_EXT = "rebuild:build_cuda_ext"
FEATURE_REBUILD_BUILD_CUTE_AOT = CUTE_CUBIN_REBUILD_FEATURE
FEATURE_REBUILD_APPLY_DEP_PATCH = "rebuild:apply_dep_patch"
FEATURE_DEP_PATCH_FLASHINFER = "dep_patch:flashinfer"

_ARTIFACT_PROVIDER_TARGET_FEATURES = frozenset(
    feature
    for descriptor in ARTIFACT_PROVIDERS.descriptors()
    for feature in descriptor.required_target_features
)

KNOWN_CONTRIBUTION_FEATURES = frozenset(
    {
        FEATURE_ENTRY,
        FEATURE_VARIANTS,
        FEATURE_PREPARE,
        FEATURE_SETUP,
        FEATURE_OVERRIDE,
        FEATURE_CUDA_SOURCES,
        FEATURE_REBUILD_BUILD_CUDA_EXT,
        FEATURE_REBUILD_APPLY_DEP_PATCH,
        FEATURE_DEP_PATCH_FLASHINFER,
    }
) | _ARTIFACT_PROVIDER_TARGET_FEATURES

_STANDARD_COMPONENT_FEATURES = frozenset(
    {
        FEATURE_ENTRY,
        FEATURE_VARIANTS,
        FEATURE_PREPARE,
        FEATURE_OVERRIDE,
        FEATURE_CUDA_SOURCES,
        FEATURE_REBUILD_BUILD_CUDA_EXT,
    }
) | _ARTIFACT_PROVIDER_TARGET_FEATURES
_FLASHINFER_FEATURES = frozenset(
    {FEATURE_DEP_PATCH_FLASHINFER, FEATURE_REBUILD_APPLY_DEP_PATCH}
)

_ID_RE = re.compile(r"^[0-9A-Za-z._\-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CATALOG_SCHEMA_VERSION = 1
_CATALOG_POLICY_VERSION = "target-catalog.v1"


class TargetCatalogError(ValueError):
    """Validator target policy is internally invalid."""


class TargetResolutionError(ValueError):
    """A bundle cannot resolve to the registered target it requested."""


def _decimal_string(value: object, *, field: str) -> str:
    """Return the frozen, float-free decimal representation used in identity JSON."""
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TargetCatalogError(f"{field} must be a finite decimal") from exc
    if not number.is_finite():
        raise TargetCatalogError(f"{field} must be a finite decimal")
    if number == 0:
        return "0"
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


@dataclass(frozen=True)
class CorrectnessContractRef:
    mode: str = "allclose"
    top_k: int = 0
    min_ratio: str = "1"
    min_cosine: str = "0"
    max_rel_norm_err: str = "0"
    min_overlap: str = "0"

    def __post_init__(self) -> None:
        if self.mode not in {
            "allclose",
            "matched_ratio",
            "cosine",
            "topk_overlap",
        }:
            raise TargetCatalogError("correctness mode is not registered")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise TargetCatalogError("correctness top_k must be a non-negative integer")
        for name in (
            "min_ratio",
            "min_cosine",
            "max_rel_norm_err",
            "min_overlap",
        ):
            object.__setattr__(
                self,
                name,
                _decimal_string(getattr(self, name), field=f"correctness {name}"),
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "min_ratio": self.min_ratio,
            "min_cosine": self.min_cosine,
            "max_rel_norm_err": self.max_rel_norm_err,
            "min_overlap": self.min_overlap,
        }


@dataclass(frozen=True)
class ToleranceContractRef:
    dtype: str
    atol: str
    rtol: str

    def __post_init__(self) -> None:
        _simple_id(self.dtype, field="tolerance dtype")
        object.__setattr__(
            self, "atol", _decimal_string(self.atol, field=f"{self.dtype} atol")
        )
        object.__setattr__(
            self, "rtol", _decimal_string(self.rtol, field=f"{self.dtype} rtol")
        )

    def snapshot(self) -> dict[str, str]:
        return {"dtype": self.dtype, "atol": self.atol, "rtol": self.rtol}


@dataclass(frozen=True)
class TargetContractRef:
    """Stdlib-only, versioned projection of one live ``SlotSpec`` contract."""

    schema_version: int
    slot_id: str
    kind: str
    entry: str
    prepare: str | None
    graph_dynamic_inputs: tuple[str, ...]
    input_abi_id: str
    output_abi_id: str
    reference_id: str
    verification_profile_id: str
    binding_family_id: str
    correctness: CorrectnessContractRef
    tolerances: tuple[ToleranceContractRef, ...]
    kl_threshold: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise TargetCatalogError("target contract schema_version must be 1")
        _simple_id(self.slot_id, field="contract slot_id")
        if self.kind not in {"op", "block", "collective"}:
            raise TargetCatalogError(f"contract {self.slot_id!r} has invalid kind")
        if not isinstance(self.entry, str) or not self.entry.isidentifier():
            raise TargetCatalogError(f"contract {self.slot_id!r} entry is invalid")
        if self.prepare is not None and (
            not isinstance(self.prepare, str) or not self.prepare.isidentifier()
        ):
            raise TargetCatalogError(f"contract {self.slot_id!r} prepare is invalid")
        if isinstance(self.graph_dynamic_inputs, str):
            raise TargetCatalogError("graph_dynamic_inputs must be an ordered sequence")
        dynamic = tuple(self.graph_dynamic_inputs)
        if len(set(dynamic)) != len(dynamic) or not all(
            isinstance(name, str) and name.isidentifier() for name in dynamic
        ):
            raise TargetCatalogError(
                f"contract {self.slot_id!r} graph_dynamic_inputs are invalid"
            )
        object.__setattr__(self, "graph_dynamic_inputs", dynamic)
        for name in (
            "input_abi_id",
            "output_abi_id",
            "reference_id",
            "verification_profile_id",
            "binding_family_id",
        ):
            _simple_id(getattr(self, name), field=f"contract {name}")
        if not isinstance(self.correctness, CorrectnessContractRef):
            raise TargetCatalogError("contract correctness must be CorrectnessContractRef")
        tolerances = tuple(self.tolerances)
        if not all(isinstance(row, ToleranceContractRef) for row in tolerances):
            raise TargetCatalogError("contract tolerances must be ToleranceContractRef rows")
        dtype_names = tuple(row.dtype for row in tolerances)
        if dtype_names != tuple(sorted(dtype_names)) or len(set(dtype_names)) != len(
            dtype_names
        ):
            raise TargetCatalogError("contract tolerances must be dtype-sorted and unique")
        object.__setattr__(self, "tolerances", tolerances)
        if self.kl_threshold is not None:
            object.__setattr__(
                self,
                "kl_threshold",
                _decimal_string(self.kl_threshold, field="contract kl_threshold"),
            )

    def snapshot(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "slot_id": self.slot_id,
            "kind": self.kind,
            "entry": self.entry,
            "prepare": self.prepare,
            "graph_dynamic_inputs": list(self.graph_dynamic_inputs),
            "input_abi_id": self.input_abi_id,
            "output_abi_id": self.output_abi_id,
            "reference_id": self.reference_id,
            "verification_profile_id": self.verification_profile_id,
            "binding_family_id": self.binding_family_id,
            "correctness": self.correctness.snapshot(),
            "tolerances": [row.snapshot() for row in self.tolerances],
            "kl_threshold": self.kl_threshold,
        }
        return result

    @property
    def digest(self) -> str:
        return canonical_digest("optima.target-contract", self.snapshot())


@dataclass(frozen=True)
class CompositionRule:
    schema_version: int
    rule_id: str
    target_ids: tuple[str, ...]
    precedence: tuple[str, ...]
    mode: str
    binding_family_id: str
    binding_contract_digest: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise TargetCatalogError("composition rule schema_version must be 1")
        if isinstance(self.target_ids, (str, bytes)) or isinstance(
            self.precedence, (str, bytes)
        ):
            raise TargetCatalogError(
                "composition rule targets and precedence must be sequences"
            )
        _simple_id(self.rule_id, field="composition rule_id")
        targets = tuple(self.target_ids)
        precedence = tuple(self.precedence)
        if len(targets) != 2 or targets != tuple(sorted(targets)):
            raise TargetCatalogError(
                "composition rule target_ids must contain exactly two sorted IDs"
            )
        if len(set(targets)) != len(targets):
            raise TargetCatalogError("composition rule target_ids contain duplicates")
        for target_id in targets:
            _simple_id(target_id, field="composition target_id")
        if len(precedence) != len(targets) or set(precedence) != set(targets):
            raise TargetCatalogError(
                "composition rule precedence must order the exact target_ids"
            )
        if self.mode != "first_applicable":
            raise TargetCatalogError("first_applicable is the only composition mode")
        _simple_id(self.binding_family_id, field="composition binding_family_id")
        if not isinstance(self.binding_contract_digest, str) or not _SHA256_RE.fullmatch(
            self.binding_contract_digest
        ):
            raise TargetCatalogError(
                "composition binding_contract_digest must be lowercase SHA-256"
            )
        object.__setattr__(self, "target_ids", targets)
        object.__setattr__(self, "precedence", precedence)

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rule_id": self.rule_id,
            "target_ids": list(self.target_ids),
            "precedence": list(self.precedence),
            "mode": self.mode,
            "binding_family_id": self.binding_family_id,
            "binding_contract_digest": self.binding_contract_digest,
        }


@dataclass(frozen=True)
class TargetSpec:
    """One validator-owned reward-unit identity.

    ``members`` are canonical semantic slot IDs, not manifest rows; variants of
    one slot never add members.  ``displaces`` is directional and represents a
    mutually exclusive registered target.  ``compatible_with`` records a known
    semantic overlap that the validator binding deliberately composes (for
    example the ordered MoE reduce-owning/plain fallback pair).
    """

    target_id: str
    kind: TargetKind
    members: tuple[str, ...]
    displaces: frozenset[str] = frozenset()
    compatible_with: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    allowed_features: frozenset[str] = frozenset({FEATURE_ENTRY})
    contract_ref: TargetContractRef | None = None
    atomic_semantics_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.members, str):
            object.__setattr__(self, "members", tuple(self.members))
        if not isinstance(self.displaces, str):
            object.__setattr__(self, "displaces", frozenset(self.displaces))
        if not isinstance(self.compatible_with, str):
            object.__setattr__(self, "compatible_with", frozenset(self.compatible_with))
        if not isinstance(self.requires, str):
            object.__setattr__(self, "requires", frozenset(self.requires))
        if not isinstance(self.allowed_features, str):
            object.__setattr__(self, "allowed_features", frozenset(self.allowed_features))


@dataclass(frozen=True)
class ResolvedTarget:
    """Canonical proposal identity, independent of manifest row order.

    ``registered=False`` is a discovery result, not a crownability judgment.
    Explicit requests that lie about a registered target fail instead of
    producing an unregistered result.
    """

    target_id: str | None
    kind: TargetKind | None
    members: tuple[str, ...]
    registered: bool
    implicit: bool
    observed_features: frozenset[str]
    features_complete: bool
    reason: str | None = None
    contract_digest: str | None = None

    def require_registered(self) -> "ResolvedTarget":
        if not self.registered:
            raise TargetResolutionError(
                self.reason or "proposal has no registered contribution target"
            )
        return self

    def require_complete_features(self) -> "ResolvedTarget":
        if not self.features_complete:
            raise TargetResolutionError(
                "target identity lacks complete trusted bundle-feature evidence"
            )
        return self


def _simple_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TargetCatalogError(f"{field} must be a non-empty canonical string")
    if not _ID_RE.fullmatch(value):
        raise TargetCatalogError(f"{field} has illegal characters: {value!r}")
    return value


def _semantic_members(manifest: Manifest) -> tuple[str, ...]:
    """Distinct slots in declaration order; variant rows count once."""
    return tuple(dict.fromkeys(op.slot for op in manifest.ops))


def manifest_declared_features(manifest: Manifest) -> frozenset[str]:
    """Derive contribution features from parsed manifest data.

    This does not inspect ``rebuild.json``.  Trusted intake must pass exact
    observed patcher capabilities through ``observed_features``; the catalog
    deliberately does not duplicate the rebuild parser or execute a plan.
    """

    features: set[str] = {FEATURE_ENTRY}
    counts: dict[str, int] = {}
    for op in manifest.ops:
        counts[op.slot] = counts.get(op.slot, 0) + 1
        if op.variant != DEFAULT_VARIANT:
            features.add(FEATURE_VARIANTS)
        if op.prepare is not None:
            features.add(FEATURE_PREPARE)
        if op.setup is not None:
            features.add(FEATURE_SETUP)
        if op.base_kernel is not None or op.override_point is not None:
            features.add(FEATURE_OVERRIDE)
        if op.cuda_sources:
            features.add(FEATURE_CUDA_SOURCES)
        if op.aot_exports:
            features.update(f"aot:{export.provider}" for export in op.aot_exports)
        # Unknown op keys are retained by Manifest for forward compatibility.
        # They are observable capabilities, not an implicit permission bypass.
        features.update(f"op_extra:{key}" for key in op.extra)
    if any(count > 1 for count in counts.values()):
        features.add(FEATURE_VARIANTS)
    features.update(f"dep_patch:{patch.target}" for patch in manifest.dep_patches)
    return frozenset(features)


def _validate_complete_feature_evidence(
    manifest: Manifest, features: frozenset[str]
) -> None:
    """Validate trusted external build evidence for static targets."""

    patches = {feature for feature in features if feature.startswith("dep_patch:")}
    applies_patch = FEATURE_REBUILD_APPLY_DEP_PATCH in features
    if patches and not applies_patch:
        raise TargetResolutionError(
            "complete feature evidence has a dependency patch without "
            "rebuild:apply_dep_patch"
        )
    if applies_patch and not patches:
        raise TargetResolutionError(
            "complete feature evidence selects rebuild:apply_dep_patch "
            "without a declared dependency patch"
        )
    has_cuda = FEATURE_CUDA_SOURCES in features
    builds_cuda = FEATURE_REBUILD_BUILD_CUDA_EXT in features
    if has_cuda and not builds_cuda:
        raise TargetResolutionError(
            "complete feature evidence has CUDA sources without "
            "rebuild:build_cuda_ext"
        )
    if builds_cuda and not has_cuda:
        raise TargetResolutionError(
            "complete feature evidence selects rebuild:build_cuda_ext "
            "without declared CUDA sources"
        )
    if builds_cuda and not any(
        path.endswith(".cu") for op in manifest.ops for path in op.cuda_sources
    ):
        raise TargetResolutionError(
            "rebuild:build_cuda_ext requires a declared .cu compilation unit"
        )
    declared_provider_ids = {
        export.provider for op in manifest.ops for export in op.aot_exports
    }
    try:
        declared_descriptors = tuple(
            ARTIFACT_PROVIDERS.require(provider_id)
            for provider_id in sorted(declared_provider_ids)
        )
    except ArtifactProviderPolicyError as exc:
        raise TargetResolutionError(str(exc)) from None
    for descriptor in declared_descriptors:
        missing = descriptor.required_target_features - features
        if missing:
            raise TargetResolutionError(
                "complete feature evidence has artifact-provider AOT exports "
                f"without required features {tuple(sorted(missing))!r}"
            )
    for descriptor in ARTIFACT_PROVIDERS.descriptors():
        if descriptor.rebuild_feature not in features:
            continue
        matching_declared = {
            row.provider_id
            for row in declared_descriptors
            if row.rebuild_feature == descriptor.rebuild_feature
        }
        if not matching_declared:
            raise TargetResolutionError(
                f"complete feature evidence selects {descriptor.rebuild_feature} "
                "without declared artifact-provider AOT exports"
            )


class TargetCatalog:
    """Immutable, deterministic validator policy for registered targets."""

    def __init__(
        self,
        specs: Iterable[TargetSpec],
        *,
        composition_rules: Iterable[CompositionRule] = (),
    ):
        if isinstance(specs, (str, bytes, Mapping)):
            raise TargetCatalogError("target specs must be an iterable of TargetSpec")
        rows = tuple(specs)
        if not rows:
            raise TargetCatalogError("target catalog must not be empty")

        by_id: dict[str, TargetSpec] = {}
        member_sets: dict[frozenset[str], str] = {}
        for index, spec in enumerate(rows):
            if not isinstance(spec, TargetSpec):
                raise TargetCatalogError(
                    f"target spec {index} is not a TargetSpec: {type(spec).__name__}"
                )
            target_id = _simple_id(spec.target_id, field="target_id")
            if target_id in by_id:
                raise TargetCatalogError(f"duplicate target ID {target_id!r}")
            if not isinstance(spec.kind, TargetKind):
                raise TargetCatalogError(
                    f"target {target_id!r} kind must be TargetKind"
                )
            if isinstance(spec.members, str) or not spec.members:
                raise TargetCatalogError(
                    f"target {target_id!r} members must be a non-empty sequence"
                )
            members = tuple(
                _simple_id(member, field=f"target {target_id!r} member")
                for member in spec.members
            )
            if len(set(members)) != len(members):
                raise TargetCatalogError(
                    f"target {target_id!r} has duplicate members {members!r}"
                )
            if spec.kind is TargetKind.SLOT:
                if members != (target_id,):
                    raise TargetCatalogError(
                        f"slot target {target_id!r} must have itself as its sole member"
                    )
                if not isinstance(spec.contract_ref, TargetContractRef):
                    raise TargetCatalogError(
                        f"slot target {target_id!r} requires a TargetContractRef"
                    )
                if spec.contract_ref.slot_id != target_id:
                    raise TargetCatalogError(
                        f"slot target {target_id!r} contract_ref names "
                        f"{spec.contract_ref.slot_id!r}"
                    )
                if spec.atomic_semantics_id is not None:
                    raise TargetCatalogError(
                        f"slot target {target_id!r} may not declare atomic_semantics_id"
                    )
            elif len(members) < 2:
                raise TargetCatalogError(
                    f"atomic target {target_id!r} requires at least two members"
                )
            else:
                if spec.contract_ref is not None:
                    raise TargetCatalogError(
                        f"atomic target {target_id!r} may not declare contract_ref"
                    )
                _simple_id(
                    spec.atomic_semantics_id,
                    field=f"atomic target {target_id!r} atomic_semantics_id",
                )

            member_set = frozenset(members)
            previous = member_sets.get(member_set)
            if previous is not None:
                raise TargetCatalogError(
                    "multiple targets register the same exact member set: "
                    f"{previous!r}, {target_id!r}"
                )
            member_sets[member_set] = target_id

            if isinstance(spec.allowed_features, str):
                raise TargetCatalogError(
                    f"target {target_id!r} allowed_features must be a set"
                )
            unknown_features = spec.allowed_features - KNOWN_CONTRIBUTION_FEATURES
            if unknown_features:
                raise TargetCatalogError(
                    f"target {target_id!r} allows unknown features "
                    f"{tuple(sorted(unknown_features))!r}"
                )
            if FEATURE_ENTRY not in spec.allowed_features:
                raise TargetCatalogError(
                    f"target {target_id!r} must allow the entry feature"
                )
            if FEATURE_SETUP in spec.allowed_features:
                raise TargetCatalogError(
                    f"target {target_id!r} may not allow engine-wide setup"
                )
            by_id[target_id] = spec

        for spec in by_id.values():
            for relation_name, related in (
                ("displaces", spec.displaces),
                ("compatible_with", spec.compatible_with),
                ("requires", spec.requires),
            ):
                if isinstance(related, str):
                    raise TargetCatalogError(
                        f"target {spec.target_id!r} {relation_name} must be a set"
                    )
                if spec.target_id in related:
                    raise TargetCatalogError(
                        f"target {spec.target_id!r} may not {relation_name} itself"
                    )
                unknown = set(related) - set(by_id)
                if unknown:
                    raise TargetCatalogError(
                        f"target {spec.target_id!r} {relation_name} unknown targets "
                        f"{tuple(sorted(unknown))!r}"
                    )
            overlap = spec.displaces & spec.compatible_with
            if overlap:
                raise TargetCatalogError(
                    f"target {spec.target_id!r} both displaces and is compatible with "
                    f"{tuple(sorted(overlap))!r}"
                )
            required_displaced = spec.requires & spec.displaces
            if required_displaced:
                raise TargetCatalogError(
                    f"target {spec.target_id!r} requires targets it displaces "
                    f"{tuple(sorted(required_displaced))!r}"
                )

            if spec.kind is TargetKind.ATOMIC:
                missing_singletons = [
                    member
                    for member in spec.members
                    if member not in by_id
                    or by_id[member].kind is not TargetKind.SLOT
                    or by_id[member].members != (member,)
                ]
                if missing_singletons:
                    raise TargetCatalogError(
                        f"atomic target {spec.target_id!r} has members without registered "
                        f"singletons {tuple(missing_singletons)!r}"
                    )
                missing_displacement = set(spec.members) - set(spec.displaces)
                if missing_displacement:
                    raise TargetCatalogError(
                        f"atomic target {spec.target_id!r} must explicitly displace "
                        f"member targets {tuple(sorted(missing_displacement))!r}"
                    )

        for spec in by_id.values():
            for other_id in spec.compatible_with:
                if spec.target_id not in by_id[other_id].compatible_with:
                    raise TargetCatalogError(
                        "compatible overlap must be symmetric: "
                        f"{spec.target_id!r} -> {other_id!r}"
                    )
                if (
                    other_id in spec.displaces
                    or spec.target_id in by_id[other_id].displaces
                ):
                    raise TargetCatalogError(
                        f"targets {spec.target_id!r} and {other_id!r} cannot be both "
                        "compatible and displaced"
                    )

        for left, right in combinations(by_id.values(), 2):
            shared = set(left.members) & set(right.members)
            if not shared:
                continue
            if right.target_id in left.compatible_with:
                raise TargetCatalogError(
                    f"targets {left.target_id!r} and {right.target_id!r} share members "
                    "but first_applicable rules do not define member ownership"
                )
            related = (
                right.target_id in left.displaces
                or left.target_id in right.displaces
            )
            if not related:
                raise TargetCatalogError(
                    f"targets {left.target_id!r} and {right.target_id!r} share "
                    f"members {tuple(sorted(shared))!r} without explicit "
                    "displacement or compatible overlap"
                )

        self._validate_relation_dag(by_id, relation="displaces")
        self._validate_relation_dag(by_id, relation="requires")

        def relation_closure(target_id: str, relation: str) -> set[str]:
            found: set[str] = set()
            pending = list(getattr(by_id[target_id], relation))
            while pending:
                current = pending.pop()
                if current in found:
                    continue
                found.add(current)
                pending.extend(getattr(by_id[current], relation))
            return found

        displacement_closures = {
            target_id: frozenset(relation_closure(target_id, "displaces"))
            for target_id in by_id
        }
        requirement_closures = {
            target_id: frozenset(relation_closure(target_id, "requires"))
            for target_id in by_id
        }
        for spec in by_id.values():
            displaced = displacement_closures[spec.target_id]
            required = requirement_closures[spec.target_id]
            contradiction = displaced & required
            if contradiction:
                raise TargetCatalogError(
                    f"target {spec.target_id!r} requires its displacement closure "
                    f"{tuple(sorted(contradiction))!r}"
                )
            reverse = {
                dependency
                for dependency in required
                if spec.target_id in displacement_closures[dependency]
            }
            if reverse:
                raise TargetCatalogError(
                    f"target {spec.target_id!r} requires targets that displace it "
                    f"{tuple(sorted(reverse))!r}"
                )

        rules = tuple(composition_rules)
        if not all(isinstance(rule, CompositionRule) for rule in rules):
            raise TargetCatalogError("composition_rules must contain CompositionRule rows")
        by_rule_id: dict[str, CompositionRule] = {}
        by_rule_targets: dict[tuple[str, ...], CompositionRule] = {}
        by_rule_pair: dict[tuple[str, str], CompositionRule] = {}
        for rule in rules:
            if rule.rule_id in by_rule_id:
                raise TargetCatalogError(f"duplicate composition rule ID {rule.rule_id!r}")
            if rule.target_ids in by_rule_targets:
                raise TargetCatalogError(
                    f"duplicate composition rule targets {rule.target_ids!r}"
                )
            unknown = set(rule.target_ids) - set(by_id)
            if unknown:
                raise TargetCatalogError(
                    f"composition rule {rule.rule_id!r} names unknown targets "
                    f"{tuple(sorted(unknown))!r}"
                )
            for left, right in combinations(rule.target_ids, 2):
                if (
                    right not in by_id[left].compatible_with
                    or left not in by_id[right].compatible_with
                ):
                    raise TargetCatalogError(
                        f"composition rule {rule.rule_id!r} targets must be "
                        "explicitly compatible"
                    )
                pair = (left, right)
                previous = by_rule_pair.get(pair)
                if previous is not None:
                    raise TargetCatalogError(
                        f"compatible target pair {pair!r} is covered by multiple "
                        f"composition rules {previous.rule_id!r}, {rule.rule_id!r}"
                    )
                by_rule_pair[pair] = rule
            for target_id in rule.target_ids:
                contract = by_id[target_id].contract_ref
                if (
                    contract is not None
                    and contract.binding_family_id != rule.binding_family_id
                ):
                    raise TargetCatalogError(
                        f"composition rule {rule.rule_id!r} binding family does not "
                        f"match target {target_id!r}"
                    )
            by_rule_id[rule.rule_id] = rule
            by_rule_targets[rule.target_ids] = rule

        for left, right in combinations(sorted(by_id), 2):
            compatible = right in by_id[left].compatible_with
            rule = by_rule_pair.get((left, right))
            if compatible and rule is None:
                raise TargetCatalogError(
                    f"compatible targets {(left, right)!r} require a CompositionRule"
                )
            if not compatible and rule is not None:
                raise TargetCatalogError(
                    f"composition rule {rule.rule_id!r} has no compatible target pair"
                )

        outgoing: dict[str, set[str]] = {target_id: set() for target_id in by_id}
        incoming: dict[str, int] = {target_id: 0 for target_id in by_id}
        for rule in rules:
            for earlier, later in zip(rule.precedence, rule.precedence[1:]):
                if later not in outgoing[earlier]:
                    outgoing[earlier].add(later)
                    incoming[later] += 1
        ready = [target_id for target_id, count in incoming.items() if count == 0]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for child in outgoing[current]:
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
        if visited != len(by_id):
            raise TargetCatalogError("composition precedence contains a cycle")

        ordered = dict(sorted(by_id.items()))
        self._by_id = ordered
        self._by_members = {
            members: ordered[target_id] for members, target_id in member_sets.items()
        }
        self._composition_by_id = dict(sorted(by_rule_id.items()))
        self._composition_by_targets = by_rule_targets
        self._composition_by_pair = by_rule_pair
        self._displacement_closures = displacement_closures
        self._requirement_closures = requirement_closures
        self._target_snapshots = {
            target_id: self._build_target_snapshot(spec)
            for target_id, spec in ordered.items()
        }
        self._snapshot = {
            "schema_version": _CATALOG_SCHEMA_VERSION,
            "policy_version": _CATALOG_POLICY_VERSION,
            # Provider admission/build/load policy is consensus-bearing. Changing
            # crownability, artifact kind, ABI, or capability vocabulary must
            # rotate catalog/stack authority even when target feature names do not.
            "artifact_provider_registry": ARTIFACT_PROVIDERS.snapshot(),
            "artifact_provider_registry_digest": ARTIFACT_PROVIDERS.digest,
            "targets": [self._target_snapshots[target_id] for target_id in ordered],
            "composition_rules": [
                self._composition_by_id[rule_id].snapshot()
                for rule_id in self._composition_by_id
            ],
        }
        self._digest = canonical_digest("optima.target-catalog", self._snapshot)

    @staticmethod
    def _validate_relation_dag(
        by_id: Mapping[str, TargetSpec], *, relation: str
    ) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(target_id: str) -> None:
            if target_id in visiting:
                raise TargetCatalogError(
                    f"target {relation} graph contains a cycle at {target_id!r}"
                )
            if target_id in visited:
                return
            visiting.add(target_id)
            for child in getattr(by_id[target_id], relation):
                visit(child)
            visiting.remove(target_id)
            visited.add(target_id)

        for target_id in by_id:
            visit(target_id)

    def _build_target_snapshot(self, spec: TargetSpec) -> dict[str, object]:
        common: dict[str, object] = {
            "target_id": spec.target_id,
            "kind": spec.kind.value,
            "members": list(spec.members),
            "displaces": sorted(spec.displaces),
            "compatible_with": sorted(spec.compatible_with),
            "requires": sorted(spec.requires),
            "allowed_features": sorted(spec.allowed_features),
        }
        if spec.kind is TargetKind.SLOT:
            assert spec.contract_ref is not None
            common["contract_ref"] = spec.contract_ref.snapshot()
            common["contract_digest"] = spec.contract_ref.digest
        else:
            assert spec.atomic_semantics_id is not None
            member_digests = [
                self._by_id[member].contract_ref.digest  # type: ignore[union-attr]
                for member in spec.members
            ]
            contract_payload = {
                "schema_version": 1,
                "target_id": spec.target_id,
                "atomic_semantics_id": spec.atomic_semantics_id,
                "member_contract_digests": member_digests,
            }
            common["atomic_semantics_id"] = spec.atomic_semantics_id
            common["member_contract_digests"] = member_digests
            common["contract_digest"] = canonical_digest(
                "optima.atomic-target-contract", contract_payload
            )
        return common

    def require(self, target_id: str) -> TargetSpec:
        if not isinstance(target_id, str):
            raise TargetResolutionError("target ID must be a string")
        try:
            return self._by_id[target_id]
        except KeyError:
            raise TargetResolutionError(
                f"unknown contribution target {target_id!r}; target IDs are validator-owned"
            ) from None

    def snapshot(self) -> dict[str, object]:
        """Return a fresh canonical JSON projection of complete catalog policy."""
        return deepcopy(self._snapshot)

    @property
    def digest(self) -> str:
        return self._digest

    def target_spec_digest(self, target_id: str) -> str:
        self.require(target_id)
        return canonical_digest(
            "optima.target-spec", self._target_snapshots[target_id]
        )

    def contract_digest(self, target_id: str) -> str:
        self.require(target_id)
        value = self._target_snapshots[target_id]["contract_digest"]
        assert isinstance(value, str)
        return value

    def displacement_closure(self, target_id: str) -> frozenset[str]:
        self.require(target_id)
        return self._displacement_closures[target_id]

    def requires_closure(self, target_id: str) -> frozenset[str]:
        self.require(target_id)
        return self._requirement_closures[target_id]

    def composition_rule(self, left: str, right: str) -> CompositionRule:
        targets = tuple(sorted((left, right)))
        if len(set(targets)) != 2:
            raise TargetResolutionError("composition requires two distinct target IDs")
        self.require(left)
        self.require(right)
        try:
            return self._composition_by_pair[targets]
        except KeyError:
            raise TargetResolutionError(
                f"targets {targets!r} have no registered composition rule"
            ) from None

    def ordered_active_targets(self, target_ids: Iterable[str]) -> tuple[str, ...]:
        """Order active rows with validator-owned composition precedence."""
        active = self.validate_active_targets(target_ids)
        active_set = set(active)
        outgoing: dict[str, set[str]] = {target_id: set() for target_id in active}
        incoming: dict[str, int] = {target_id: 0 for target_id in active}
        for rule in self._composition_by_id.values():
            precedence = tuple(
                target_id for target_id in rule.precedence if target_id in active_set
            )
            if len(precedence) < 2:
                continue
            for earlier, later in zip(precedence, precedence[1:]):
                if later not in outgoing[earlier]:
                    outgoing[earlier].add(later)
                    incoming[later] += 1
        ready = sorted(target_id for target_id, count in incoming.items() if count == 0)
        ordered: list[str] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for child in sorted(outgoing[current]):
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(ordered) != len(active):
            raise TargetResolutionError("active composition precedence contains a cycle")
        return tuple(ordered)

    def validate_active_targets(self, target_ids: Iterable[str]) -> tuple[str, ...]:
        if isinstance(target_ids, (str, bytes)):
            raise TargetResolutionError("active target IDs must be an iterable")
        active = tuple(target_ids)
        if not all(isinstance(target_id, str) for target_id in active):
            raise TargetResolutionError("active target IDs must be strings")
        if len(set(active)) != len(active):
            raise TargetResolutionError("active target IDs contain duplicates")
        for target_id in active:
            self.require(target_id)
        active_set = set(active)
        for target_id in active:
            conflicts = self.displacement_closure(target_id) & active_set
            if conflicts:
                raise TargetResolutionError(
                    f"active target {target_id!r} displaces "
                    f"{tuple(sorted(conflicts))!r}"
                )
            missing = self.requires_closure(target_id) - active_set
            if missing:
                raise TargetResolutionError(
                    f"active target {target_id!r} requires active contributions "
                    f"{tuple(sorted(missing))!r}; stock does not satisfy requires"
                )
        return tuple(sorted(active))

    def resolve_manifest(
        self,
        manifest: Manifest,
        *,
        observed_features: Iterable[str] | None = None,
    ) -> ResolvedTarget:
        if not isinstance(manifest, Manifest):
            raise TypeError("manifest must be an optima.manifest.Manifest")
        features_complete = observed_features is not None
        if isinstance(observed_features, (str, bytes)):
            raise TargetResolutionError("observed_features must be an iterable of strings")
        extra_features = tuple(observed_features or ())
        if not all(isinstance(feature, str) and feature for feature in extra_features):
            raise TargetResolutionError(
                "observed_features must contain non-empty strings"
            )
        features = frozenset(
            set(manifest_declared_features(manifest)) | set(extra_features)
        )
        members_in_manifest = _semantic_members(manifest)
        member_set = frozenset(members_in_manifest)
        request = manifest.competition

        if request is not None:
            if not isinstance(request, CompetitionEntry):
                raise TargetResolutionError(
                    "manifest competition request must be a CompetitionEntry"
                )
            if (
                not isinstance(request.target, str)
                or not _ID_RE.fullmatch(request.target)
                or not isinstance(request.mode, str)
            ):
                raise TargetResolutionError(
                    "manifest competition target/mode are malformed"
                )

        if request is not None and request.mode == "system":
            members = tuple(sorted(member_set))
            return ResolvedTarget(
                target_id=None,
                kind=None,
                members=members,
                registered=False,
                implicit=False,
                observed_features=features,
                features_complete=features_complete,
                reason=(
                    "legacy competition mode 'system' is unregistered; migrate to "
                    "a slot/atomic target or the future discovery lane"
                ),
            )

        if request is None:
            spec = self._by_members.get(member_set)
            if spec is None:
                members = tuple(sorted(member_set))
                return ResolvedTarget(
                    target_id=None,
                    kind=None,
                    members=members,
                    registered=False,
                    implicit=True,
                    observed_features=features,
                    features_complete=features_complete,
                    reason=(
                        f"proposal {manifest.bundle_id!r} has no registered exact target "
                        f"for members {members!r}; classify it for future discovery"
                    ),
                )
            implicit = True
        else:
            if request.mode not in {kind.value for kind in TargetKind}:
                raise TargetResolutionError(
                    f"unknown competition mode {request.mode!r}"
                )
            spec = self.require(request.target)
            if request.mode != spec.kind.value:
                raise TargetResolutionError(
                    f"target {spec.target_id!r} is catalog kind {spec.kind.value!r}, "
                    f"not requested mode {request.mode!r}"
                )
            implicit = False

        if member_set != frozenset(spec.members):
            raise TargetResolutionError(
                f"target {spec.target_id!r} requires exact members {spec.members!r}; "
                f"manifest declares {members_in_manifest!r}"
            )
        unexpected = features - spec.allowed_features
        if unexpected:
            if FEATURE_SETUP in unexpected:
                detail = "engine-wide setup belongs in the fenced discovery lane"
            else:
                detail = "features are not registered for this target"
            raise TargetResolutionError(
                f"target {spec.target_id!r} rejects observed features "
                f"{tuple(sorted(unexpected))!r}: {detail}"
            )
        if features_complete:
            _validate_complete_feature_evidence(manifest, features)
        return ResolvedTarget(
            target_id=spec.target_id,
            kind=spec.kind,
            members=spec.members,
            registered=True,
            implicit=implicit,
            observed_features=features,
            features_complete=features_complete,
        )

    def resolve_intake(
        self,
        manifest: Manifest,
        *,
        observed_features: Iterable[str],
    ) -> ResolvedTarget:
        """Resolve with a required trusted projection of external features."""
        return self.resolve_manifest(
            manifest, observed_features=observed_features
        ).require_registered().require_complete_features()


# Target IDs are intentionally lightweight policy data.  Importing this module
# must not import Torch through optima.slots; a focused test checks that every
# live SlotSpec has exactly one singleton target and vice versa.
SINGLETON_TARGET_IDS = (
    "activation.silu_and_mul",
    "attention.decode",
    "attention.msa_block_score",
    "attention.msa_prefill_block_score",
    "attention.sdpa",
    "collective.all_reduce",
    "collective.ar_residual_rmsnorm",
    "collective.moe_finalize_ar_rmsnorm",
    "moe.fused_experts",
    "moe.fused_experts_reduce",
    "norm.rmsnorm",
)

MOE_EPILOGUE_ATOMIC_TARGET = "collective.moe_epilogue.v1"
MOE_EPILOGUE_MEMBERS = (
    "collective.ar_residual_rmsnorm",
    "collective.moe_finalize_ar_rmsnorm",
)

_STANDARD_TOLERANCES = (
    ToleranceContractRef("bfloat16", "0.02", "0.02"),
    ToleranceContractRef("float16", "0.01", "0.01"),
    ToleranceContractRef("float32", "0.00001", "0.00001"),
)


def _contract_ref(
    slot_id: str,
    *,
    kind: str,
    entry: str,
    prepare: str | None,
    graph_dynamic_inputs: tuple[str, ...],
    input_abi_id: str,
    output_abi_id: str,
    reference_id: str,
    verification_profile_id: str,
    binding_family_id: str,
    correctness: CorrectnessContractRef,
    kl_threshold: str | None = None,
) -> TargetContractRef:
    return TargetContractRef(
        schema_version=1,
        slot_id=slot_id,
        kind=kind,
        entry=entry,
        prepare=prepare,
        graph_dynamic_inputs=graph_dynamic_inputs,
        input_abi_id=input_abi_id,
        output_abi_id=output_abi_id,
        reference_id=reference_id,
        verification_profile_id=verification_profile_id,
        binding_family_id=binding_family_id,
        correctness=correctness,
        tolerances=_STANDARD_TOLERANCES,
        kl_threshold=kl_threshold,
    )


_SINGLETON_CONTRACTS = {
    "activation.silu_and_mul": _contract_ref(
        "activation.silu_and_mul",
        kind="op",
        entry="silu_and_mul",
        prepare=None,
        graph_dynamic_inputs=("x",),
        input_abi_id="activation.silu_and_mul.input.v1",
        output_abi_id="activation.silu_and_mul.output.v1",
        reference_id="activation.silu_and_mul.reference.v1",
        verification_profile_id="activation.silu_and_mul.verify.v1",
        binding_family_id="sglang.activation.silu_and_mul.v1",
        correctness=CorrectnessContractRef(),
    ),
    "attention.decode": _contract_ref(
        "attention.decode",
        kind="block",
        entry="attention_decode",
        prepare=None,
        graph_dynamic_inputs=("q", "k", "v", "seq_lens"),
        input_abi_id="attention.decode.input.v1",
        output_abi_id="attention.decode.output.v1",
        reference_id="attention.decode.reference.v1",
        verification_profile_id="attention.decode.verify.v1",
        binding_family_id="sglang.attention.decode.v1",
        correctness=CorrectnessContractRef(mode="matched_ratio", min_ratio="0.99"),
        kl_threshold="0.03",
    ),
    "attention.msa_block_score": _contract_ref(
        "attention.msa_block_score",
        kind="block",
        entry="msa_block_score",
        prepare=None,
        graph_dynamic_inputs=("q", "index_k", "seq_lens"),
        input_abi_id="attention.msa_block_score.input.v1",
        output_abi_id="attention.msa_block_score.output.v1",
        reference_id="attention.msa_block_score.reference.v1",
        verification_profile_id="attention.msa_block_score.verify.v1",
        binding_family_id="sglang.attention.msa.decode-score.v1",
        correctness=CorrectnessContractRef(
            mode="topk_overlap", top_k=8, min_overlap="0.875"
        ),
        kl_threshold="0.03",
    ),
    "attention.msa_prefill_block_score": _contract_ref(
        "attention.msa_prefill_block_score",
        kind="block",
        entry="msa_prefill_block_score",
        prepare=None,
        graph_dynamic_inputs=("q", "index_k"),
        input_abi_id="attention.msa_prefill_block_score.input.v1",
        output_abi_id="attention.msa_prefill_block_score.output.v1",
        reference_id="attention.msa_prefill_block_score.reference.v1",
        verification_profile_id="attention.msa_prefill_block_score.verify.v1",
        binding_family_id="sglang.attention.msa.prefill-score.v1",
        correctness=CorrectnessContractRef(
            mode="topk_overlap", top_k=8, min_overlap="0.9"
        ),
        kl_threshold="0.03",
    ),
    "attention.sdpa": _contract_ref(
        "attention.sdpa",
        kind="block",
        entry="attention",
        prepare=None,
        graph_dynamic_inputs=("q", "k", "v"),
        input_abi_id="attention.sdpa.input.v1",
        output_abi_id="attention.sdpa.output.v1",
        reference_id="attention.sdpa.reference.v1",
        verification_profile_id="attention.sdpa.verify.v1",
        binding_family_id="sglang.attention.radix.v1",
        correctness=CorrectnessContractRef(mode="matched_ratio", min_ratio="0.99"),
        kl_threshold="0.03",
    ),
    "collective.all_reduce": _contract_ref(
        "collective.all_reduce",
        kind="collective",
        entry="all_reduce",
        prepare=None,
        graph_dynamic_inputs=("x",),
        input_abi_id="collective.all_reduce.input.v1",
        output_abi_id="collective.all_reduce.output.v1",
        reference_id="collective.all_reduce.reference.v1",
        verification_profile_id="collective.all_reduce.verify.v1",
        binding_family_id="sglang.collective.all_reduce.v1",
        correctness=CorrectnessContractRef(mode="matched_ratio", min_ratio="0.99"),
    ),
    "collective.ar_residual_rmsnorm": _contract_ref(
        "collective.ar_residual_rmsnorm",
        kind="collective",
        entry="ar_residual_rmsnorm",
        prepare=None,
        graph_dynamic_inputs=("x", "residual"),
        input_abi_id="collective.ar_residual_rmsnorm.input.v1",
        output_abi_id="collective.ar_residual_rmsnorm.output.v1",
        reference_id="collective.ar_residual_rmsnorm.reference.v1",
        verification_profile_id="collective.ar_residual_rmsnorm.verify.v1",
        binding_family_id="sglang.collective.ar-fusion.v1",
        correctness=CorrectnessContractRef(mode="matched_ratio", min_ratio="0.99"),
    ),
    "collective.moe_finalize_ar_rmsnorm": _contract_ref(
        "collective.moe_finalize_ar_rmsnorm",
        kind="collective",
        entry="moe_finalize_ar_rmsnorm",
        prepare=None,
        graph_dynamic_inputs=("gemm_out", "row_map", "scales", "residual"),
        input_abi_id="collective.moe_finalize_ar_rmsnorm.input.v1",
        output_abi_id="collective.moe_finalize_ar_rmsnorm.output.v1",
        reference_id="collective.moe_finalize_ar_rmsnorm.reference.v1",
        verification_profile_id="collective.moe_finalize_ar_rmsnorm.verify.v1",
        binding_family_id="sglang.collective.moe-finalize.v1",
        correctness=CorrectnessContractRef(mode="matched_ratio", min_ratio="0.99"),
    ),
    "moe.fused_experts": _contract_ref(
        "moe.fused_experts",
        kind="block",
        entry="fused_experts",
        prepare="prepare",
        graph_dynamic_inputs=("x", "topk_ids", "topk_weights"),
        input_abi_id="moe.fused_experts.input.v1",
        output_abi_id="moe.fused_experts.output.v1",
        reference_id="moe.fused_experts.reference.v1",
        verification_profile_id="moe.fused_experts.verify.v1",
        binding_family_id="sglang.moe.fused-experts.dispatch.v1",
        correctness=CorrectnessContractRef(mode="matched_ratio", min_ratio="0.97"),
    ),
    "moe.fused_experts_reduce": _contract_ref(
        "moe.fused_experts_reduce",
        kind="collective",
        entry="fused_experts_reduce",
        prepare="prepare",
        graph_dynamic_inputs=("x", "topk_ids", "topk_weights"),
        input_abi_id="moe.fused_experts_reduce.input.v1",
        output_abi_id="moe.fused_experts_reduce.output.v1",
        reference_id="moe.fused_experts_reduce.reference.v1",
        verification_profile_id="moe.fused_experts_reduce.verify.v1",
        binding_family_id="sglang.moe.fused-experts.dispatch.v1",
        correctness=CorrectnessContractRef(mode="matched_ratio", min_ratio="0.97"),
    ),
    "norm.rmsnorm": _contract_ref(
        "norm.rmsnorm",
        kind="op",
        entry="rmsnorm",
        prepare=None,
        graph_dynamic_inputs=("x",),
        input_abi_id="norm.rmsnorm.input.v1",
        output_abi_id="norm.rmsnorm.output.v1",
        reference_id="norm.rmsnorm.reference.v1",
        verification_profile_id="norm.rmsnorm.verify.v1",
        binding_family_id="sglang.norm.rmsnorm.v1",
        correctness=CorrectnessContractRef(),
    ),
}


@lru_cache(maxsize=1)
def default_target_catalog() -> TargetCatalog:
    moe_pair = frozenset({"moe.fused_experts", "moe.fused_experts_reduce"})
    specs: list[TargetSpec] = []
    for target_id in SINGLETON_TARGET_IDS:
        compatible = moe_pair - {target_id} if target_id in moe_pair else frozenset()
        features = _STANDARD_COMPONENT_FEATURES
        if target_id == "collective.moe_finalize_ar_rmsnorm":
            features = features | _FLASHINFER_FEATURES
        specs.append(
            TargetSpec(
                target_id=target_id,
                kind=TargetKind.SLOT,
                members=(target_id,),
                compatible_with=compatible,
                allowed_features=features,
                contract_ref=_SINGLETON_CONTRACTS[target_id],
            )
        )
    specs.append(
        TargetSpec(
            target_id=MOE_EPILOGUE_ATOMIC_TARGET,
            kind=TargetKind.ATOMIC,
            members=MOE_EPILOGUE_MEMBERS,
            displaces=frozenset(MOE_EPILOGUE_MEMBERS),
            allowed_features=frozenset(
                _STANDARD_COMPONENT_FEATURES | _FLASHINFER_FEATURES
            ),
            atomic_semantics_id="collective.moe_epilogue.v1.atomic-semantics.v1",
        )
    )
    moe_rule = CompositionRule(
        schema_version=1,
        rule_id="sglang.moe.reduce-first.v1",
        target_ids=tuple(sorted(moe_pair)),
        precedence=("moe.fused_experts_reduce", "moe.fused_experts"),
        mode="first_applicable",
        binding_family_id="sglang.moe.fused-experts.dispatch.v1",
        binding_contract_digest=canonical_digest(
            "optima.binding-contract",
            {
                "schema_version": 1,
                "binding_family_id": "sglang.moe.fused-experts.dispatch.v1",
                "precedence": ["moe.fused_experts_reduce", "moe.fused_experts"],
                "mode": "first_applicable",
            },
        ),
    )
    return TargetCatalog(specs, composition_rules=(moe_rule,))


def resolve_target(
    manifest: Manifest,
    *,
    catalog: TargetCatalog | None = None,
) -> ResolvedTarget:
    """Resolve semantic identity; external bundle features remain incomplete."""
    return (catalog or default_target_catalog()).resolve_manifest(manifest)


def resolve_intake_target(
    manifest: Manifest,
    *,
    observed_features: Iterable[str],
    catalog: TargetCatalog | None = None,
) -> ResolvedTarget:
    """Resolve admission with required trusted external-feature evidence."""
    return (catalog or default_target_catalog()).resolve_intake(
        manifest, observed_features=observed_features
    )
