"""Closed validator policy for executable artifact providers.

The manifest carries a provider *identifier*, never an implementation callback.
This module is the single trusted mapping from that identifier to the provider's
artifact shape and execution policy.  Provider-specific compilers/loaders remain
validator-owned modules; registering one here grants only the declarative policy
described by :class:`ArtifactProviderDescriptor`.

In particular, a provider may be registered for bounded bring-up without being
authoritative or crownable.  Intake can then retain and fingerprint its declaration
while evaluation/release admission fails closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from optima.stack_identity import canonical_digest


_PROVIDER_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_POLICY_ID = re.compile(r"[a-z0-9][a-z0-9._:-]{0,191}\Z")
_PROFILE_INPUT = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*\Z")


class ArtifactProviderPolicyError(ValueError):
    """A trusted provider descriptor or provider selection is invalid."""


class ArtifactKind(str, Enum):
    """Native artifact class consumed by validator-owned runtime code."""

    CUDA_CUBIN = "cuda_cubin"


class ArtifactBindingABI(str, Enum):
    """Validator-owned argument-packing contract for an artifact kind."""

    CUDA_DRIVER_PARAMS_V1 = "cuda_driver_params.v1"


class ArtifactBuildPhase(str, Enum):
    """Only environment in which candidate-influenced compilation may occur."""

    OCI_PREBUILD_ONLY = "oci_prebuild_only"


class ArtifactLoadPhase(str, Enum):
    """Only environment in which a sealed artifact may be loaded."""

    ISOLATED_ENGINE_WORKER_ONLY = "isolated_engine_worker_only"


@dataclass(frozen=True)
class ArtifactProviderDescriptor:
    """Data-only policy for one validator-shipped executable provider.

    No field is executable.  Adding a provider therefore cannot smuggle a loader,
    compiler, capability resolver, or host callback through manifest data.
    """

    provider_id: str
    artifact_kind: ArtifactKind
    binding_abi: ArtifactBindingABI
    authoritative: bool
    crownable: bool
    build_phase: ArtifactBuildPhase
    load_phase: ArtifactLoadPhase
    build_patcher_id: str
    rebuild_feature: str
    required_target_features: frozenset[str]
    provider_capabilities: frozenset[str]
    requires_compile_profile: bool
    compile_profile_inputs: frozenset[str]
    publication_directory: str
    supports_static_slots: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or _PROVIDER_ID.fullmatch(
            self.provider_id
        ) is None:
            raise ArtifactProviderPolicyError("artifact provider ID is not canonical")
        if type(self.artifact_kind) is not ArtifactKind:
            raise ArtifactProviderPolicyError("artifact kind is not registered")
        if type(self.binding_abi) is not ArtifactBindingABI:
            raise ArtifactProviderPolicyError("artifact binding ABI is not registered")
        compatible_bindings = {
            ArtifactKind.CUDA_CUBIN: frozenset(
                {ArtifactBindingABI.CUDA_DRIVER_PARAMS_V1}
            ),
        }
        if self.binding_abi not in compatible_bindings[self.artifact_kind]:
            raise ArtifactProviderPolicyError(
                "artifact binding ABI is incompatible with its artifact kind"
            )
        if type(self.build_phase) is not ArtifactBuildPhase or type(
            self.load_phase
        ) is not ArtifactLoadPhase:
            raise ArtifactProviderPolicyError("artifact build/load phase is not registered")
        for field in (
            "authoritative",
            "crownable",
            "requires_compile_profile",
            "supports_static_slots",
        ):
            if type(getattr(self, field)) is not bool:
                raise ArtifactProviderPolicyError(
                    f"artifact provider {field} must be boolean"
                )
        if self.crownable and not self.authoritative:
            raise ArtifactProviderPolicyError(
                "a crownable artifact provider must be authoritative"
            )
        if not isinstance(self.build_patcher_id, str) or _POLICY_ID.fullmatch(
            self.build_patcher_id
        ) is None:
            raise ArtifactProviderPolicyError("artifact build patcher ID is malformed")
        if (
            not isinstance(self.rebuild_feature, str)
            or not self.rebuild_feature.startswith("rebuild:")
            or _POLICY_ID.fullmatch(self.rebuild_feature) is None
        ):
            raise ArtifactProviderPolicyError("artifact rebuild feature is malformed")
        if type(self.required_target_features) is not frozenset or any(
            not isinstance(value, str) or _POLICY_ID.fullmatch(value) is None
            for value in self.required_target_features
        ):
            raise ArtifactProviderPolicyError(
                "artifact required target features are malformed"
            )
        required = {self.manifest_feature, self.rebuild_feature}
        if not required <= self.required_target_features:
            raise ArtifactProviderPolicyError(
                "artifact target features omit provider or rebuild authority"
            )
        if type(self.provider_capabilities) is not frozenset or any(
            not isinstance(value, str) or _POLICY_ID.fullmatch(value) is None
            for value in self.provider_capabilities
        ):
            raise ArtifactProviderPolicyError(
                "artifact provider capabilities are malformed"
            )
        if type(self.compile_profile_inputs) is not frozenset or any(
            not isinstance(value, str) or _PROFILE_INPUT.fullmatch(value) is None
            for value in self.compile_profile_inputs
        ):
            raise ArtifactProviderPolicyError(
                "artifact compile-profile inputs are malformed"
            )
        if self.compile_profile_inputs and not self.requires_compile_profile:
            raise ArtifactProviderPolicyError(
                "artifact profile inputs require compile-profile authority"
            )
        if (
            not isinstance(self.publication_directory, str)
            or not self.publication_directory
            or "/" in self.publication_directory
            or self.publication_directory in {".", ".."}
            or _PROVIDER_ID.fullmatch(self.publication_directory) is None
        ):
            raise ArtifactProviderPolicyError(
                "artifact publication directory is not canonical"
            )

    @property
    def manifest_feature(self) -> str:
        return f"aot:{self.provider_id}"

    @property
    def bringup_only(self) -> bool:
        return not self.authoritative

    def snapshot(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind.value,
            "authoritative": self.authoritative,
            "binding_abi": self.binding_abi.value,
            "build_patcher_id": self.build_patcher_id,
            "build_phase": self.build_phase.value,
            "compile_profile_inputs": sorted(self.compile_profile_inputs),
            "crownable": self.crownable,
            "load_phase": self.load_phase.value,
            "provider_capabilities": sorted(self.provider_capabilities),
            "provider_id": self.provider_id,
            "publication_directory": self.publication_directory,
            "rebuild_feature": self.rebuild_feature,
            "required_target_features": sorted(self.required_target_features),
            "requires_compile_profile": self.requires_compile_profile,
            "supports_static_slots": self.supports_static_slots,
        }


class ArtifactProviderRegistry:
    """Immutable trusted provider table; it contains no dynamic registration API."""

    def __init__(self, descriptors: Iterable[ArtifactProviderDescriptor]):
        if isinstance(descriptors, (str, bytes, Mapping)):
            raise ArtifactProviderPolicyError(
                "artifact provider descriptors must be an iterable"
            )
        rows = tuple(descriptors)
        by_id: dict[str, ArtifactProviderDescriptor] = {}
        patcher_features: dict[str, str] = {}
        rebuild_features: dict[str, set[str]] = {}
        publication_directories: dict[str, str] = {}
        for index, descriptor in enumerate(rows):
            if type(descriptor) is not ArtifactProviderDescriptor:
                raise ArtifactProviderPolicyError(
                    f"artifact provider descriptor {index} has the wrong type"
                )
            if descriptor.provider_id in by_id:
                raise ArtifactProviderPolicyError(
                    f"duplicate artifact provider {descriptor.provider_id!r}"
                )
            prior = patcher_features.setdefault(
                descriptor.build_patcher_id, descriptor.rebuild_feature
            )
            if prior != descriptor.rebuild_feature:
                raise ArtifactProviderPolicyError(
                    "one artifact patcher cannot imply multiple rebuild features"
                )
            rebuild_features.setdefault(descriptor.rebuild_feature, set()).add(
                descriptor.provider_id
            )
            prior_provider = publication_directories.setdefault(
                descriptor.publication_directory, descriptor.provider_id
            )
            if prior_provider != descriptor.provider_id:
                raise ArtifactProviderPolicyError(
                    "artifact providers may not share a publication directory"
                )
            by_id[descriptor.provider_id] = descriptor
        self._by_id = MappingProxyType(dict(sorted(by_id.items())))
        self._patcher_features = MappingProxyType(dict(sorted(patcher_features.items())))
        self._providers_by_rebuild_feature = MappingProxyType(
            {
                feature: frozenset(provider_ids)
                for feature, provider_ids in sorted(rebuild_features.items())
            }
        )
        self._snapshot = {
            "providers": [self._by_id[key].snapshot() for key in self._by_id],
            "schema_version": 1,
        }
        self._digest = canonical_digest("optima.artifact-provider-registry", self._snapshot)

    @property
    def digest(self) -> str:
        return self._digest

    def snapshot(self) -> dict[str, object]:
        return {
            "providers": [descriptor.snapshot() for descriptor in self._by_id.values()],
            "schema_version": 1,
        }

    def descriptors(self) -> tuple[ArtifactProviderDescriptor, ...]:
        return tuple(self._by_id.values())

    def get(self, provider_id: object) -> ArtifactProviderDescriptor | None:
        if not isinstance(provider_id, str):
            return None
        return self._by_id.get(provider_id)

    def require(self, provider_id: object) -> ArtifactProviderDescriptor:
        descriptor = self.get(provider_id)
        if descriptor is None:
            raise ArtifactProviderPolicyError(
                f"artifact provider {provider_id!r} is not registered"
            )
        return descriptor

    def build_feature_for_patcher(self, patcher_id: object) -> str | None:
        if not isinstance(patcher_id, str):
            return None
        return self._patcher_features.get(patcher_id)

    def providers_for_rebuild_feature(
        self, rebuild_feature: object
    ) -> frozenset[str]:
        if not isinstance(rebuild_feature, str):
            return frozenset()
        return self._providers_by_rebuild_feature.get(rebuild_feature, frozenset())

    def require_crownable(
        self, provider_ids: Iterable[str], *, context: str
    ) -> tuple[ArtifactProviderDescriptor, ...]:
        if isinstance(provider_ids, (str, bytes, Mapping)):
            raise ArtifactProviderPolicyError(
                "artifact provider selection must be an iterable of IDs"
            )
        selected = tuple(provider_ids)
        if any(
            not isinstance(provider_id, str)
            or _PROVIDER_ID.fullmatch(provider_id) is None
            for provider_id in selected
        ):
            raise ArtifactProviderPolicyError(
                "artifact provider selection contains a malformed ID"
            )
        unique = tuple(sorted(set(selected)))
        descriptors = tuple(self.require(provider_id) for provider_id in unique)
        blocked = tuple(
            descriptor.provider_id
            for descriptor in descriptors
            if not descriptor.authoritative or not descriptor.crownable
        )
        if blocked:
            raise ArtifactProviderPolicyError(
                f"{context} rejects non-authoritative/non-crownable artifact "
                f"providers {blocked!r}"
            )
        return descriptors


CUTE_CUBIN_PROFILE_INPUTS = frozenset(
    f"max_active_clusters.cluster_size_{cluster_size}"
    for cluster_size in (1, 2, 4, 8, 16)
)

# Device-only CuTe/CUTLASS provider.  It deliberately shares the same bounded
# arena-measured compile inputs as the bring-up object provider, but publishes a
# disjoint artifact set and is built by a disjoint reviewed patcher.  Its sealed
# CUBIN admission, validator-owned launch materialization, and isolated runtime
# are the authoritative path; qualification still owns correctness, graph,
# fidelity, throughput, reproduction, and settlement authority.
CUTE_CUBIN_PROVIDER_ID = "cutlass.cute.cubin.v1"
CUTE_CUBIN_MANIFEST_FEATURE = f"aot:{CUTE_CUBIN_PROVIDER_ID}"
CUTE_CUBIN_REBUILD_FEATURE = "rebuild:build_cute_cubin"
CUTE_CUBIN_PATCHER_ID = "optima.build-cute-cubin.v1"
CUTE_CUBIN_PROVIDER = ArtifactProviderDescriptor(
    provider_id=CUTE_CUBIN_PROVIDER_ID,
    artifact_kind=ArtifactKind.CUDA_CUBIN,
    binding_abi=ArtifactBindingABI.CUDA_DRIVER_PARAMS_V1,
    authoritative=True,
    crownable=True,
    build_phase=ArtifactBuildPhase.OCI_PREBUILD_ONLY,
    load_phase=ArtifactLoadPhase.ISOLATED_ENGINE_WORKER_ONLY,
    build_patcher_id=CUTE_CUBIN_PATCHER_ID,
    rebuild_feature=CUTE_CUBIN_REBUILD_FEATURE,
    required_target_features=frozenset(
        {
            CUTE_CUBIN_MANIFEST_FEATURE,
            CUTE_CUBIN_REBUILD_FEATURE,
        }
    ),
    provider_capabilities=frozenset(
        {
            "group.native_handle.v1",
            "group.peer_ptr_table.v1",
        }
    ),
    requires_compile_profile=True,
    compile_profile_inputs=CUTE_CUBIN_PROFILE_INPUTS,
    publication_directory="cute_cubin",
    supports_static_slots=True,
)


ARTIFACT_PROVIDERS = ArtifactProviderRegistry((CUTE_CUBIN_PROVIDER,))


__all__ = [
    "ARTIFACT_PROVIDERS",
    "CUTE_CUBIN_MANIFEST_FEATURE",
    "CUTE_CUBIN_PATCHER_ID",
    "CUTE_CUBIN_PROFILE_INPUTS",
    "CUTE_CUBIN_PROVIDER",
    "CUTE_CUBIN_PROVIDER_ID",
    "CUTE_CUBIN_REBUILD_FEATURE",
    "ArtifactBindingABI",
    "ArtifactBuildPhase",
    "ArtifactKind",
    "ArtifactLoadPhase",
    "ArtifactProviderDescriptor",
    "ArtifactProviderPolicyError",
    "ArtifactProviderRegistry",
]
