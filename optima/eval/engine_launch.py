"""Canonical identity and trusted local binding for one isolated engine launch.

The public launch specification is deliberately path-free.  It describes one
complete, content-addressed engine environment but contains no scheduling role,
miner identity, score, request nonce, physical GPU identifier, or host path.
Trusted host code resolves those identities immediately before mounting and
independently reopens the materialized engine tree.

This module is standard-library-only apart from Optima's data-only identity and
engine-tree validators.  It never imports an inference runtime or candidate
code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from optima.engine_tree import (
    EngineTreeError,
    MaterializedEngineTree,
    reopen_materialized_engine_tree,
)
from optima.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


ENGINE_LAUNCH_SCHEMA_VERSION = 1
NATIVE_BUILD_SCHEMA_VERSION = 1
LOGICAL_HARDWARE_SCHEMA_VERSION = 1

_ARCHITECTURE_RE = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")
_POLICY_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_PHYSICAL_GPU_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")
_PREFLIGHT_IDENTITY_FIELDS = frozenset(
    {"image_digest", "platform_digest", "worker_distribution_digest"}
)


class EngineLaunchError(ValueError):
    """One launch identity or trusted local binding is invalid."""


def _strict_object(
    value: object, *, fields: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EngineLaunchError(f"{name} schema mismatch")
    return value


def _version(value: object, *, expected: int, field: str) -> int:
    if type(value) is not int or value != expected:
        raise EngineLaunchError(f"{field} must be {expected}")
    return value


def _digest(value: object, *, field: str) -> str:
    try:
        checked = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise EngineLaunchError(str(exc)) from exc
    if checked == "0" * 64:
        raise EngineLaunchError(f"{field} must not be the all-zero digest")
    return checked


def _architecture(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _ARCHITECTURE_RE.fullmatch(value):
        raise EngineLaunchError(
            f"{field} must be a canonical architecture such as 'sm103' or 'sm120'"
        )
    return value


def _policy_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _POLICY_NAME_RE.fullmatch(value):
        raise EngineLaunchError(
            f"{field} must match [a-z0-9][a-z0-9._-]*"
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise EngineLaunchError(f"{field} must be a positive integer")
    return value


def native_toolchain_digest(*, image_digest: str, platform_digest: str) -> str:
    """Identity of toolchain bytes sealed by one immutable OCI platform image."""
    return canonical_digest(
        "optima.eval.native-toolchain-policy",
        {
            "image_digest": _digest(image_digest, field="image_digest"),
            "platform_digest": _digest(platform_digest, field="platform_digest"),
        },
    )


def native_patcher_digest(*, worker_distribution_digest: str) -> str:
    """Identity of reviewed patcher bytes in the attested worker distribution."""
    return canonical_digest(
        "optima.eval.native-patcher-policy",
        {
            "worker_distribution_digest": _digest(
                worker_distribution_digest, field="worker_distribution_digest"
            )
        },
    )


def native_compiler_policy_digest(
    *,
    image_digest: str,
    worker_distribution_digest: str,
    dependency_policy_digest: str,
    target_architecture: str,
) -> str:
    """Identity of compiler argv policy and image-owned dependency generators."""
    return canonical_digest(
        "optima.eval.native-compiler-policy",
        {
            "dependency_policy_digest": _digest(
                dependency_policy_digest, field="dependency_policy_digest"
            ),
            "image_digest": _digest(image_digest, field="image_digest"),
            "target_architecture": _architecture(
                target_architecture, field="native target_architecture"
            ),
            "worker_distribution_digest": _digest(
                worker_distribution_digest, field="worker_distribution_digest"
            ),
        },
    )


@dataclass(frozen=True)
class NativeBuildSpec:
    """Complete identity of one hermetic native build.

    The digest is safe to use as the native artifact/publication namespace: it
    includes the whole materialized tree and immutable build environment rather
    than a bundle name, target, source stem, or selected delta.
    """

    tree_digest: str
    image_digest: str
    platform_digest: str
    worker_distribution_digest: str
    toolchain_digest: str
    patcher_digest: str
    compiler_flags_digest: str
    target_architecture: str
    dependency_policy_digest: str
    schema_version: int = NATIVE_BUILD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "tree_digest",
            "image_digest",
            "platform_digest",
            "worker_distribution_digest",
            "toolchain_digest",
            "patcher_digest",
            "compiler_flags_digest",
            "dependency_policy_digest",
        ):
            object.__setattr__(
                self, field, _digest(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "target_architecture",
            _architecture(
                self.target_architecture, field="native target_architecture"
            ),
        )
        _version(
            self.schema_version,
            expected=NATIVE_BUILD_SCHEMA_VERSION,
            field="native build schema_version",
        )
        expected = {
            "toolchain_digest": native_toolchain_digest(
                image_digest=self.image_digest,
                platform_digest=self.platform_digest,
            ),
            "patcher_digest": native_patcher_digest(
                worker_distribution_digest=self.worker_distribution_digest
            ),
            "compiler_flags_digest": native_compiler_policy_digest(
                image_digest=self.image_digest,
                worker_distribution_digest=self.worker_distribution_digest,
                dependency_policy_digest=self.dependency_policy_digest,
                target_architecture=self.target_architecture,
            ),
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise EngineLaunchError(
                    f"native build {field} is not grounded in its attested image/worker policy"
                )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.eval.native-build", self.to_dict())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "compiler_flags_digest": self.compiler_flags_digest,
            "dependency_policy_digest": self.dependency_policy_digest,
            "image_digest": self.image_digest,
            "patcher_digest": self.patcher_digest,
            "platform_digest": self.platform_digest,
            "schema_version": self.schema_version,
            "target_architecture": self.target_architecture,
            "toolchain_digest": self.toolchain_digest,
            "tree_digest": self.tree_digest,
            "type": "native_build",
            "worker_distribution_digest": self.worker_distribution_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NativeBuildSpec":
        row = _strict_object(
            value,
            fields=frozenset(
                {
                    "compiler_flags_digest",
                    "dependency_policy_digest",
                    "image_digest",
                    "patcher_digest",
                    "platform_digest",
                    "schema_version",
                    "target_architecture",
                    "toolchain_digest",
                    "tree_digest",
                    "type",
                    "worker_distribution_digest",
                }
            ),
            name="native build",
        )
        if row["type"] != "native_build":
            raise EngineLaunchError("native build type must be 'native_build'")
        return cls(
            tree_digest=row["tree_digest"],  # type: ignore[arg-type]
            image_digest=row["image_digest"],  # type: ignore[arg-type]
            platform_digest=row["platform_digest"],  # type: ignore[arg-type]
            worker_distribution_digest=row["worker_distribution_digest"],  # type: ignore[arg-type]
            toolchain_digest=row["toolchain_digest"],  # type: ignore[arg-type]
            patcher_digest=row["patcher_digest"],  # type: ignore[arg-type]
            compiler_flags_digest=row["compiler_flags_digest"],  # type: ignore[arg-type]
            target_architecture=row["target_architecture"],  # type: ignore[arg-type]
            dependency_policy_digest=row["dependency_policy_digest"],  # type: ignore[arg-type]
            schema_version=row["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LogicalHardwareSpec:
    """Path-free logical hardware and distributed-runtime expectation."""

    visible_gpu_count: int
    architecture: str
    topology_class: str
    topology_digest: str
    tp_size: int
    ep_size: int
    dp_size: int
    device_policy_digest: str
    schema_version: int = LOGICAL_HARDWARE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "visible_gpu_count",
            _positive_int(self.visible_gpu_count, field="visible_gpu_count"),
        )
        object.__setattr__(
            self,
            "architecture",
            _architecture(self.architecture, field="hardware architecture"),
        )
        object.__setattr__(
            self,
            "topology_class",
            _policy_name(self.topology_class, field="topology_class"),
        )
        object.__setattr__(
            self,
            "topology_digest",
            _digest(self.topology_digest, field="topology_digest"),
        )
        for field in ("tp_size", "ep_size", "dp_size"):
            degree = _positive_int(getattr(self, field), field=field)
            if degree > self.visible_gpu_count:
                raise EngineLaunchError(
                    f"{field} cannot exceed visible_gpu_count"
                )
            object.__setattr__(self, field, degree)
        object.__setattr__(
            self,
            "device_policy_digest",
            _digest(self.device_policy_digest, field="device_policy_digest"),
        )
        _version(
            self.schema_version,
            expected=LOGICAL_HARDWARE_SCHEMA_VERSION,
            field="logical hardware schema_version",
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.eval.logical-hardware", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "device_policy_digest": self.device_policy_digest,
            "dp_size": self.dp_size,
            "ep_size": self.ep_size,
            "schema_version": self.schema_version,
            "topology_class": self.topology_class,
            "topology_digest": self.topology_digest,
            "tp_size": self.tp_size,
            "type": "logical_hardware",
            "visible_gpu_count": self.visible_gpu_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "LogicalHardwareSpec":
        row = _strict_object(
            value,
            fields=frozenset(
                {
                    "architecture",
                    "device_policy_digest",
                    "dp_size",
                    "ep_size",
                    "schema_version",
                    "topology_class",
                    "topology_digest",
                    "tp_size",
                    "type",
                    "visible_gpu_count",
                }
            ),
            name="logical hardware",
        )
        if row["type"] != "logical_hardware":
            raise EngineLaunchError(
                "logical hardware type must be 'logical_hardware'"
            )
        return cls(
            visible_gpu_count=row["visible_gpu_count"],  # type: ignore[arg-type]
            architecture=row["architecture"],  # type: ignore[arg-type]
            topology_class=row["topology_class"],  # type: ignore[arg-type]
            topology_digest=row["topology_digest"],  # type: ignore[arg-type]
            tp_size=row["tp_size"],  # type: ignore[arg-type]
            ep_size=row["ep_size"],  # type: ignore[arg-type]
            dp_size=row["dp_size"],  # type: ignore[arg-type]
            device_policy_digest=row["device_policy_digest"],  # type: ignore[arg-type]
            schema_version=row["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class EngineLaunchSpec:
    """Strict canonical identity for one complete isolated engine launch."""

    runtime_digest: str
    base_engine_digest: str
    arena_digest: str
    stack_digest: str
    tree_digest: str
    image_digest: str
    platform_digest: str
    controller_distribution_digest: str
    worker_distribution_digest: str
    model_revision_digest: str
    model_manifest_digest: str
    model_content_digest: str
    validator_overlay_digest: str
    engine_config_digest: str
    seccomp_policy_digest: str
    resource_policy_digest: str
    native_build_spec_digest: str
    hardware: LogicalHardwareSpec
    schema_version: int = ENGINE_LAUNCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "runtime_digest",
            "base_engine_digest",
            "arena_digest",
            "stack_digest",
            "tree_digest",
            "image_digest",
            "platform_digest",
            "controller_distribution_digest",
            "worker_distribution_digest",
            "model_revision_digest",
            "model_manifest_digest",
            "model_content_digest",
            "validator_overlay_digest",
            "engine_config_digest",
            "seccomp_policy_digest",
            "resource_policy_digest",
            "native_build_spec_digest",
        ):
            object.__setattr__(
                self, field, _digest(getattr(self, field), field=field)
            )
        if not isinstance(self.hardware, LogicalHardwareSpec):
            raise EngineLaunchError("hardware must be a LogicalHardwareSpec")
        _version(
            self.schema_version,
            expected=ENGINE_LAUNCH_SCHEMA_VERSION,
            field="engine launch schema_version",
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.eval.engine-launch", self.to_dict())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def runtime_preflight_identity(self) -> Mapping[str, str]:
        """Exact image/worker identity a candidate-free preflight must report."""

        return MappingProxyType(
            {
                "image_digest": self.image_digest,
                "platform_digest": self.platform_digest,
                "worker_distribution_digest": self.worker_distribution_digest,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "base_engine_digest": self.base_engine_digest,
            "controller_distribution_digest": self.controller_distribution_digest,
            "engine_config_digest": self.engine_config_digest,
            "hardware": self.hardware.to_dict(),
            "image_digest": self.image_digest,
            "model_content_digest": self.model_content_digest,
            "model_manifest_digest": self.model_manifest_digest,
            "model_revision_digest": self.model_revision_digest,
            "native_build_spec_digest": self.native_build_spec_digest,
            "platform_digest": self.platform_digest,
            "resource_policy_digest": self.resource_policy_digest,
            "runtime_digest": self.runtime_digest,
            "schema_version": self.schema_version,
            "seccomp_policy_digest": self.seccomp_policy_digest,
            "stack_digest": self.stack_digest,
            "tree_digest": self.tree_digest,
            "type": "engine_launch",
            "validator_overlay_digest": self.validator_overlay_digest,
            "worker_distribution_digest": self.worker_distribution_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EngineLaunchSpec":
        row = _strict_object(
            value,
            fields=frozenset(
                {
                    "arena_digest",
                    "base_engine_digest",
                    "controller_distribution_digest",
                    "engine_config_digest",
                    "hardware",
                    "image_digest",
                    "model_content_digest",
                    "model_manifest_digest",
                    "model_revision_digest",
                    "native_build_spec_digest",
                    "platform_digest",
                    "resource_policy_digest",
                    "runtime_digest",
                    "schema_version",
                    "seccomp_policy_digest",
                    "stack_digest",
                    "tree_digest",
                    "type",
                    "validator_overlay_digest",
                    "worker_distribution_digest",
                }
            ),
            name="engine launch",
        )
        if row["type"] != "engine_launch":
            raise EngineLaunchError("engine launch type must be 'engine_launch'")
        return cls(
            runtime_digest=row["runtime_digest"],  # type: ignore[arg-type]
            base_engine_digest=row["base_engine_digest"],  # type: ignore[arg-type]
            arena_digest=row["arena_digest"],  # type: ignore[arg-type]
            stack_digest=row["stack_digest"],  # type: ignore[arg-type]
            tree_digest=row["tree_digest"],  # type: ignore[arg-type]
            image_digest=row["image_digest"],  # type: ignore[arg-type]
            platform_digest=row["platform_digest"],  # type: ignore[arg-type]
            controller_distribution_digest=row[
                "controller_distribution_digest"
            ],  # type: ignore[arg-type]
            worker_distribution_digest=row["worker_distribution_digest"],  # type: ignore[arg-type]
            model_revision_digest=row["model_revision_digest"],  # type: ignore[arg-type]
            model_manifest_digest=row["model_manifest_digest"],  # type: ignore[arg-type]
            model_content_digest=row["model_content_digest"],  # type: ignore[arg-type]
            validator_overlay_digest=row["validator_overlay_digest"],  # type: ignore[arg-type]
            engine_config_digest=row["engine_config_digest"],  # type: ignore[arg-type]
            seccomp_policy_digest=row["seccomp_policy_digest"],  # type: ignore[arg-type]
            resource_policy_digest=row["resource_policy_digest"],  # type: ignore[arg-type]
            native_build_spec_digest=row["native_build_spec_digest"],  # type: ignore[arg-type]
            hardware=LogicalHardwareSpec.from_dict(row["hardware"]),
            schema_version=row["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PhysicalHardwareBinding:
    """Trusted host-local realization of a logical hardware specification.

    Physical identifiers intentionally have no canonical serialization and do
    not participate in ``EngineLaunchSpec.digest``.
    """

    physical_gpu_ids: tuple[str, ...]
    architecture: str
    topology_class: str
    topology_digest: str
    tp_size: int
    ep_size: int
    dp_size: int
    device_policy_digest: str

    def __post_init__(self) -> None:
        if isinstance(self.physical_gpu_ids, (str, bytes)) or not isinstance(
            self.physical_gpu_ids, Sequence
        ):
            raise EngineLaunchError("physical_gpu_ids must be a sequence")
        identifiers = tuple(self.physical_gpu_ids)
        if not identifiers:
            raise EngineLaunchError("physical_gpu_ids must not be empty")
        if any(
            not isinstance(value, str)
            or not _PHYSICAL_GPU_ID_RE.fullmatch(value)
            for value in identifiers
        ):
            raise EngineLaunchError("physical_gpu_ids contains an invalid identifier")
        if len(set(identifiers)) != len(identifiers):
            raise EngineLaunchError("physical_gpu_ids contains duplicates")
        object.__setattr__(self, "physical_gpu_ids", identifiers)
        object.__setattr__(
            self,
            "architecture",
            _architecture(self.architecture, field="physical GPU architecture"),
        )
        object.__setattr__(
            self,
            "topology_class",
            _policy_name(self.topology_class, field="physical topology_class"),
        )
        object.__setattr__(
            self,
            "topology_digest",
            _digest(self.topology_digest, field="physical topology_digest"),
        )
        for field in ("tp_size", "ep_size", "dp_size"):
            degree = _positive_int(getattr(self, field), field=f"physical {field}")
            if degree > len(identifiers):
                raise EngineLaunchError(
                    f"physical {field} cannot exceed selected GPU count"
                )
            object.__setattr__(self, field, degree)
        object.__setattr__(
            self,
            "device_policy_digest",
            _digest(
                self.device_policy_digest, field="physical device_policy_digest"
            ),
        )

    def validate_against(self, expected: LogicalHardwareSpec) -> None:
        if not isinstance(expected, LogicalHardwareSpec):
            raise EngineLaunchError("expected hardware must be a LogicalHardwareSpec")
        actual: dict[str, object] = {
            "visible_gpu_count": len(self.physical_gpu_ids),
            "architecture": self.architecture,
            "topology_class": self.topology_class,
            "topology_digest": self.topology_digest,
            "tp_size": self.tp_size,
            "ep_size": self.ep_size,
            "dp_size": self.dp_size,
            "device_policy_digest": self.device_policy_digest,
        }
        for field, value in actual.items():
            if value != getattr(expected, field):
                raise EngineLaunchError(
                    f"physical hardware {field} does not match launch specification"
                )


@dataclass(frozen=True)
class TrustedLaunchBinding:
    """Host-local inputs resolved from validator-owned digest registries."""

    materialized_tree_root: Path
    controller_distribution_digest: str
    native_build_spec: NativeBuildSpec
    runtime_preflight_receipt: object
    physical_hardware: PhysicalHardwareBinding

    def __post_init__(self) -> None:
        if isinstance(self.materialized_tree_root, bytes):
            raise EngineLaunchError("materialized_tree_root must be a local path")
        try:
            root = Path(self.materialized_tree_root)
        except TypeError as exc:
            raise EngineLaunchError("materialized_tree_root must be a local path") from exc
        if not str(root):
            raise EngineLaunchError("materialized_tree_root must not be empty")
        object.__setattr__(self, "materialized_tree_root", root)
        object.__setattr__(
            self,
            "controller_distribution_digest",
            _digest(
                self.controller_distribution_digest,
                field="resolved controller_distribution_digest",
            ),
        )
        if not isinstance(self.native_build_spec, NativeBuildSpec):
            raise EngineLaunchError("native_build_spec must be a NativeBuildSpec")
        if not isinstance(self.physical_hardware, PhysicalHardwareBinding):
            raise EngineLaunchError(
                "physical_hardware must be a PhysicalHardwareBinding"
            )


@dataclass(frozen=True)
class ResolvedEngineLaunch:
    """Verified host-local launch inputs; never a portable or hashed record."""

    spec: EngineLaunchSpec
    materialized_tree: MaterializedEngineTree
    native_build_spec: NativeBuildSpec
    physical_hardware: PhysicalHardwareBinding
    runtime_preflight_identity: tuple[tuple[str, str], ...]

    @property
    def materialized_tree_root(self) -> Path:
        return self.materialized_tree.root


def validate_native_build_spec(
    launch: EngineLaunchSpec, native: NativeBuildSpec
) -> None:
    """Reject a build record that is not exactly the launch's whole-tree build."""

    if not isinstance(launch, EngineLaunchSpec):
        raise EngineLaunchError("launch must be an EngineLaunchSpec")
    if not isinstance(native, NativeBuildSpec):
        raise EngineLaunchError("native must be a NativeBuildSpec")
    expected: tuple[tuple[str, object, object], ...] = (
        ("native_build_spec_digest", native.digest, launch.native_build_spec_digest),
        ("tree_digest", native.tree_digest, launch.tree_digest),
        ("image_digest", native.image_digest, launch.image_digest),
        ("platform_digest", native.platform_digest, launch.platform_digest),
        (
            "worker_distribution_digest",
            native.worker_distribution_digest,
            launch.worker_distribution_digest,
        ),
        (
            "target_architecture",
            native.target_architecture,
            launch.hardware.architecture,
        ),
    )
    for field, actual, wanted in expected:
        if actual != wanted:
            raise EngineLaunchError(
                f"native build {field} does not match launch specification"
            )


def validate_runtime_preflight_receipt(
    launch: EngineLaunchSpec, receipt: object
) -> Mapping[str, str]:
    """Validate a candidate-free runtime preflight without importing its type.

    ``runtime_preflight`` owns the receipt class.  Duck-typing the small mapping
    here avoids a circular import while retaining an exact, closed identity.
    """

    if not isinstance(launch, EngineLaunchSpec):
        raise EngineLaunchError("launch must be an EngineLaunchSpec")
    try:
        identity: Any = getattr(receipt, "launch_identity")
        if callable(identity):
            identity = identity()
    except (AttributeError, TypeError, ValueError) as exc:
        raise EngineLaunchError(
            "runtime preflight receipt lacks a valid launch_identity"
        ) from exc
    row = _strict_object(
        identity,
        fields=_PREFLIGHT_IDENTITY_FIELDS,
        name="runtime preflight launch_identity",
    )
    normalized = {
        field: _digest(row[field], field=f"preflight {field}")
        for field in sorted(_PREFLIGHT_IDENTITY_FIELDS)
    }
    if normalized != dict(launch.runtime_preflight_identity):
        raise EngineLaunchError(
            "runtime preflight launch_identity does not match launch specification"
        )
    return MappingProxyType(normalized)


def reopen_launch_tree(
    launch: EngineLaunchSpec, root: str | Path
) -> MaterializedEngineTree:
    """Reopen the exact tree and prove its embedded stack identity."""

    if not isinstance(launch, EngineLaunchSpec):
        raise EngineLaunchError("launch must be an EngineLaunchSpec")
    try:
        tree = reopen_materialized_engine_tree(
            root, expected_tree_digest=launch.tree_digest
        )
    except (EngineTreeError, OSError, TypeError, ValueError) as exc:
        raise EngineLaunchError(f"materialized engine tree reopen failed: {exc}") from exc
    if tree.stack_digest != launch.stack_digest:
        raise EngineLaunchError(
            "materialized engine tree stack_digest does not match launch specification"
        )
    return tree


def resolve_engine_launch(
    launch: EngineLaunchSpec, binding: TrustedLaunchBinding
) -> ResolvedEngineLaunch:
    """Resolve all host-local launch facts and reopen the tree immediately.

    Callers should invoke this directly before constructing read-only mounts.
    The returned object is intentionally not serializable and its physical GPU
    identifiers never enter the canonical launch identity.
    """

    if not isinstance(launch, EngineLaunchSpec):
        raise EngineLaunchError("launch must be an EngineLaunchSpec")
    if not isinstance(binding, TrustedLaunchBinding):
        raise EngineLaunchError("binding must be a TrustedLaunchBinding")
    if (
        binding.controller_distribution_digest
        != launch.controller_distribution_digest
    ):
        raise EngineLaunchError(
            "resolved controller distribution does not match launch specification"
        )
    validate_native_build_spec(launch, binding.native_build_spec)
    preflight = validate_runtime_preflight_receipt(
        launch, binding.runtime_preflight_receipt
    )
    binding.physical_hardware.validate_against(launch.hardware)
    tree = reopen_launch_tree(launch, binding.materialized_tree_root)
    return ResolvedEngineLaunch(
        spec=launch,
        materialized_tree=tree,
        native_build_spec=binding.native_build_spec,
        physical_hardware=binding.physical_hardware,
        runtime_preflight_identity=tuple(sorted(preflight.items())),
    )


__all__ = [
    "ENGINE_LAUNCH_SCHEMA_VERSION",
    "EngineLaunchError",
    "EngineLaunchSpec",
    "LOGICAL_HARDWARE_SCHEMA_VERSION",
    "LogicalHardwareSpec",
    "NATIVE_BUILD_SCHEMA_VERSION",
    "NativeBuildSpec",
    "PhysicalHardwareBinding",
    "ResolvedEngineLaunch",
    "TrustedLaunchBinding",
    "reopen_launch_tree",
    "resolve_engine_launch",
    "validate_native_build_spec",
    "validate_runtime_preflight_receipt",
    "native_compiler_policy_digest",
    "native_patcher_digest",
    "native_toolchain_digest",
]
