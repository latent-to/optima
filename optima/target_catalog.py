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
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Mapping

from optima.manifest import CompetitionEntry, DEFAULT_VARIANT, Manifest


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
FEATURE_REBUILD_BUILD_CUDA_EXT = "rebuild:build_cuda_ext"
FEATURE_REBUILD_APPLY_DEP_PATCH = "rebuild:apply_dep_patch"
FEATURE_DEP_PATCH_FLASHINFER = "dep_patch:flashinfer"

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
)

_STANDARD_COMPONENT_FEATURES = frozenset(
    {
        FEATURE_ENTRY,
        FEATURE_VARIANTS,
        FEATURE_PREPARE,
        FEATURE_OVERRIDE,
        FEATURE_CUDA_SOURCES,
        FEATURE_REBUILD_BUILD_CUDA_EXT,
    }
)
_FLASHINFER_FEATURES = frozenset(
    {FEATURE_DEP_PATCH_FLASHINFER, FEATURE_REBUILD_APPLY_DEP_PATCH}
)

_ID_RE = re.compile(r"^[0-9A-Za-z._\-]+$")


class TargetCatalogError(ValueError):
    """Validator target policy is internally invalid."""


class TargetResolutionError(ValueError):
    """A bundle cannot resolve to the registered target it requested."""


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
    allowed_features: frozenset[str] = frozenset({FEATURE_ENTRY})

    def __post_init__(self) -> None:
        if not isinstance(self.members, str):
            object.__setattr__(self, "members", tuple(self.members))
        if not isinstance(self.displaces, str):
            object.__setattr__(self, "displaces", frozenset(self.displaces))
        if not isinstance(self.compatible_with, str):
            object.__setattr__(self, "compatible_with", frozenset(self.compatible_with))
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
        # Unknown op keys are retained by Manifest for forward compatibility.
        # They are observable capabilities, not an implicit permission bypass.
        features.update(f"op_extra:{key}" for key in op.extra)
    if any(count > 1 for count in counts.values()):
        features.add(FEATURE_VARIANTS)
    features.update(f"dep_patch:{patch.target}" for patch in manifest.dep_patches)
    return frozenset(features)


class TargetCatalog:
    """Immutable, deterministic validator policy for registered targets."""

    def __init__(self, specs: Iterable[TargetSpec]):
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
            elif len(members) < 2:
                raise TargetCatalogError(
                    f"atomic target {target_id!r} requires at least two members"
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
            related = (
                right.target_id in left.displaces
                or left.target_id in right.displaces
                or right.target_id in left.compatible_with
            )
            if not related:
                raise TargetCatalogError(
                    f"targets {left.target_id!r} and {right.target_id!r} share "
                    f"members {tuple(sorted(shared))!r} without explicit "
                    "displacement or compatible overlap"
                )

        self._validate_displacement_dag(by_id)
        ordered = dict(sorted(by_id.items()))
        self._by_id = ordered
        self._by_members = {
            members: ordered[target_id] for members, target_id in member_sets.items()
        }

    @staticmethod
    def _validate_displacement_dag(by_id: Mapping[str, TargetSpec]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(target_id: str) -> None:
            if target_id in visiting:
                raise TargetCatalogError(
                    f"target displacement graph contains a cycle at {target_id!r}"
                )
            if target_id in visited:
                return
            visiting.add(target_id)
            for child in by_id[target_id].displaces:
                visit(child)
            visiting.remove(target_id)
            visited.add(target_id)

        for target_id in by_id:
            visit(target_id)

    def require(self, target_id: str) -> TargetSpec:
        if not isinstance(target_id, str):
            raise TargetResolutionError("target ID must be a string")
        try:
            return self._by_id[target_id]
        except KeyError:
            raise TargetResolutionError(
                f"unknown contribution target {target_id!r}; target IDs are validator-owned"
            ) from None

    def displacement_closure(self, target_id: str) -> frozenset[str]:
        root = self.require(target_id)
        found: set[str] = set()
        stack = list(root.displaces)
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(self._by_id[current].displaces)
        return frozenset(found)

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
            patches = {
                feature for feature in features if feature.startswith("dep_patch:")
            }
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
        )
    )
    return TargetCatalog(specs)


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
