"""Declarative call ABI for sealed native/CuTe artifacts.

The semantic call boundary belongs to the validator's :class:`SlotSpec`; a miner
may only project that boundary into a compiled artifact's ordered argument list.
This module is deliberately torch-free so manifest intake can validate the
projection without importing a GPU runtime or candidate code.

The first consumer is CuTe AOT, but the types are provider-neutral.  A future
CUTLASS/CUDA provider can consume the same binding plan.  Manifests may request
bounded validator-allocated workspace/state/prepared tensors through the closed
artifact resource plan below; process-group/stream authority remains owned by
the slot. Runtime support may fail closed on a resource kind it cannot yet
materialize; syntax support does not grant runtime authority.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class ArtifactABIError(ValueError):
    """A slot resource or artifact launch projection is malformed."""


_RESOURCE_RE = re.compile(
    r"(?:input|output|workspace|prepared|state|group|stream)"
    r"(?:\.[a-z][a-z0-9_]{0,63}){0,3}\Z"
)
_CAPABILITY_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_PROVIDER_CAPABILITY_RE = re.compile(
    r"group\.[a-z][a-z0-9_]{0,63}\.v[1-9][0-9]{0,5}\Z"
)
_RESOURCE_KINDS = frozenset(
    {"tensor", "scalar", "pointer", "stream", "group", "opaque"}
)
_ACCESS_MODES = frozenset({"read", "write", "readwrite"})
_SCALAR_TYPES = frozenset(
    {
        "bool",
        "i8",
        "u8",
        "i16",
        "u16",
        "i32",
        "u32",
        "i64",
        "u64",
        "f16",
        "bf16",
        "f32",
        "tf32",
        "f64",
    }
)
_INTEGER_TYPES = frozenset({"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"})
_FLOAT_TYPES = frozenset({"f16", "bf16", "f32", "tf32", "f64"})
_INTEGER_CAST_BOUNDS = MappingProxyType(
    {
        "i8": (-(1 << 7), (1 << 7) - 1),
        "u8": (0, (1 << 8) - 1),
        "i16": (-(1 << 15), (1 << 15) - 1),
        "u16": (0, (1 << 16) - 1),
        "i32": (-(1 << 31), (1 << 31) - 1),
        "u32": (0, (1 << 32) - 1),
        "i64": (-(1 << 63), (1 << 63) - 1),
        "u64": (0, (1 << 64) - 1),
    }
)
_PROJECTIONS = frozenset(
    {
        "identity",
        "descriptor",
        "device_ptr",
        "value",
        "tuple",
        "shape",
        "stride",
        "numel",
        "rank",
        "size",
        "storage_offset",
        "native_handle",
        "peer_ptr_table",
    }
)
_DERIVED_INTEGER_PROJECTIONS = frozenset(
    {"shape", "stride", "numel", "rank", "size", "storage_offset"}
)
_AXIS_PROJECTIONS = frozenset({"shape", "stride"})
_DEFAULT_PROJECTION_BY_BINDING_KIND = MappingProxyType(
    {
        "tensor": "descriptor",
        "scalar": "value",
        "pointer": "identity",
        "stream": "identity",
        "group": "identity",
        "opaque": "identity",
        "aggregate": "tuple",
    }
)
# This is deliberately a closed relation, not a coercion registry.  Adding a
# source or argument style requires a validator-owned semantic definition here;
# candidate manifests cannot name Python callbacks or arbitrary constructors.
_SOURCE_BINDING_PROJECTIONS = MappingProxyType(
    {
        "tensor": frozenset(
            {
                ("tensor", "descriptor"),
                ("pointer", "device_ptr"),
            }
            | {
                ("scalar", projection)
                for projection in {
                    "shape",
                    "stride",
                    "numel",
                    "rank",
                    "storage_offset",
                }
            }
        ),
        "scalar": frozenset({("scalar", "value")}),
        "pointer": frozenset({("pointer", "identity")}),
        "stream": frozenset({("stream", "identity")}),
        "group": frozenset(
            {
                ("group", "identity"),
                ("scalar", "rank"),
                ("scalar", "size"),
                ("pointer", "native_handle"),
                ("pointer", "peer_ptr_table"),
            }
        ),
        "opaque": frozenset({("opaque", "identity")}),
    }
)
_ARTIFACT_ROLES = frozenset({"init", "prepare", "run", "reset", "destroy"})
_MAX_BINDINGS = 64
_MAX_PRELAUNCH = 16
_MAX_UNSQUEEZE = 8
_MAX_TENSOR_RANK = 16
_AGGREGATE_TYPES = frozenset({"Shape", "Coord", "Tile", "IntTuple", "Stride"})
_AGGREGATE_COMPONENT_CASTS = frozenset({"i32", "i64"})
_MAX_AGGREGATE_DEPTH = 4
_MAX_AGGREGATE_CHILDREN = 16
_MAX_AGGREGATE_LEAVES = 32
_MAX_AGGREGATE_DYNAMIC_ARITY = 16
_GROUP_PROVIDER_CAPABILITIES = frozenset(
    {"group.native_handle.v1", "group.peer_ptr_table.v1"}
)
_GROUP_PROJECTION_CAPABILITY = MappingProxyType(
    {
        "native_handle": "group.native_handle.v1",
        "peer_ptr_table": "group.peer_ptr_table.v1",
    }
)

# Artifact-owned storage is deliberately much narrower than the semantic slot
# resource vocabulary.  A manifest may request validator-allocated tensors, but
# it cannot mint streams, groups, pointers, opaque handles, or new semantic
# inputs/outputs.  These are syntax ceilings; an arena/runtime may impose a
# smaller live allocation budget.
_ARTIFACT_RESOURCE_RE = re.compile(
    r"(?:workspace|prepared|state)(?:\.[a-z][a-z0-9_]{0,63}){1,3}\Z"
)
_ARTIFACT_RESOURCE_LIFETIME = MappingProxyType(
    {
        "workspace": "call",
        "prepared": "prepared",
        "state": "engine",
    }
)
_ARTIFACT_RESOURCE_SCOPES = frozenset({"rank_local", "group_ipc"})
_ARTIFACT_STORAGE_BYTES = MappingProxyType(
    {
        "bool": 1,
        "i8": 1,
        "u8": 1,
        "f8e4m3fn": 1,
        "f8e4m3fnuz": 1,
        "f8e5m2": 1,
        "f8e5m2fnuz": 1,
        "i16": 2,
        "u16": 2,
        "f16": 2,
        "bf16": 2,
        "i32": 4,
        "u32": 4,
        "f32": 4,
        "i64": 8,
        "u64": 8,
        "f64": 8,
    }
)
_ARTIFACT_SHAPE_PROJECTIONS = frozenset({"shape", "numel", "rank", "value", "size"})
_MAX_ARTIFACT_RESOURCES = 32
_MAX_ARTIFACT_RESOURCE_RANK = 16
_MAX_ARTIFACT_FACTORS_PER_EXTENT = 8
_MAX_ARTIFACT_FACTOR = (1 << 31) - 1
_MAX_ARTIFACT_FACTOR_PRODUCT = (1 << 63) - 1
_MAX_ARTIFACT_EXTENT = (1 << 31) - 1
_MAX_ARTIFACT_ELEMENTS = 1 << 36
_MAX_ARTIFACT_RESOURCE_BYTES = 64 << 30
_MAX_ARTIFACT_TOTAL_BYTES = 128 << 30
_MAX_ARTIFACT_ALIGNMENT = 1 << 20


def _canonical_resource(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _RESOURCE_RE.fullmatch(value) is None:
        raise ArtifactABIError(
            f"{field} must be a canonical slot resource (for example 'input.x')"
        )
    return value


def _power_of_two(value: object, *, field: str, maximum: int = 1 << 20) -> int:
    if (
        type(value) is not int
        or value < 1
        or value > maximum
        or value & (value - 1)
    ):
        raise ArtifactABIError(
            f"{field} must be a positive power of two no larger than {maximum}"
        )
    return value


def integer_cast_bounds(cast: str) -> tuple[int, int]:
    """Return the normative range for a native integer artifact cast."""

    try:
        return _INTEGER_CAST_BOUNDS[cast]
    except (KeyError, TypeError):
        raise ArtifactABIError(f"unsupported integer artifact cast {cast!r}") from None


def checked_integer_cast(value: object, cast: str, *, field: str) -> int:
    """Validate a derived integer before the runtime packs a native argument.

    Runtime providers must use this contract (or an equivalent checked packing
    primitive); truncation and signed wraparound are never artifact ABI semantics.
    """

    minimum, maximum = integer_cast_bounds(cast)
    if type(value) is not int:
        raise ArtifactABIError(f"{field} must resolve to an integer")
    if not minimum <= value <= maximum:
        raise ArtifactABIError(
            f"{field} value {value} is outside {cast} range [{minimum}, {maximum}]"
        )
    return value


@dataclass(frozen=True)
class ArtifactAggregateComponent:
    """One bounded leaf in a validator-constructed CuTe algebra value.

    A dynamic leaf references either one tensor shape/stride coordinate or one
    validator-owned scalar resource.  A static leaf is a bounded Python integer
    and contributes no native packed argument.  Tuple nesting is represented by
    tuples of these leaves; there are no expressions, names, or callbacks.
    """

    source: str | None = None
    projection: str | None = None
    axis: int | None = None
    cast: str | None = None
    static: int | None = None

    def __post_init__(self) -> None:
        dynamic = self.source is not None
        if dynamic == (self.static is not None):
            raise ArtifactABIError(
                "aggregate component must declare exactly one of source or static"
            )
        if not dynamic:
            if type(self.static) is not int:
                raise ArtifactABIError("aggregate static component must be an integer")
            checked_integer_cast(
                self.static, "i64", field="aggregate static component"
            )
            if self.projection is not None or self.axis is not None or self.cast is not None:
                raise ArtifactABIError(
                    "aggregate static component may not declare projection/axis/cast"
                )
            return

        _canonical_resource(self.source, field="aggregate component source")
        if self.projection not in {"value", "shape", "stride"}:
            raise ArtifactABIError(
                "aggregate dynamic component projection must be value/shape/stride"
            )
        if self.cast not in _AGGREGATE_COMPONENT_CASTS:
            raise ArtifactABIError(
                "aggregate dynamic component cast must be explicit i32 or i64"
            )
        if self.projection in _AXIS_PROJECTIONS:
            if type(self.axis) is not int or not 0 <= self.axis < _MAX_TENSOR_RANK:
                raise ArtifactABIError(
                    f"aggregate component axis must be in [0, {_MAX_TENSOR_RANK - 1}]"
                )
        elif self.axis is not None:
            raise ArtifactABIError(
                "aggregate scalar value component may not declare an axis"
            )

    @property
    def dynamic(self) -> bool:
        return self.source is not None

    def to_dict(self) -> dict[str, object]:
        if not self.dynamic:
            return {"static": self.static}  # type: ignore[dict-item]
        row: dict[str, object] = {
            "cast": self.cast,  # type: ignore[dict-item]
            "projection": self.projection,  # type: ignore[dict-item]
            "source": self.source,  # type: ignore[dict-item]
        }
        if self.axis is not None:
            row["axis"] = self.axis
        return row


def _binding_writes_resource(
    binding: ArtifactBinding,
    resource: SlotResource,
) -> bool:
    """Whether an argument exposes writable storage, not merely metadata."""

    if resource.access not in {"write", "readwrite"}:
        return False
    if resource.kind == "tensor":
        return (binding.kind, binding.projection) in {
            ("tensor", "descriptor"),
            ("pointer", "device_ptr"),
        }
    if resource.kind == "pointer":
        return (binding.kind, binding.projection) == ("pointer", "identity")
    return False


def _validate_scalar_cast(
    *,
    source: SlotResource,
    cast: str | None,
    field: str,
) -> None:
    if source.scalar_type == "bool":
        allowed = cast == "bool"
    elif source.scalar_type in _INTEGER_TYPES:
        allowed = cast in _INTEGER_TYPES
    else:
        allowed = source.scalar_type in _FLOAT_TYPES and cast in _FLOAT_TYPES
    if not allowed:
        raise ArtifactABIError(
            f"{field} cannot cast slot scalar {source.scalar_type!r} to {cast!r}"
        )


ArtifactAggregateNode = ArtifactAggregateComponent | tuple["ArtifactAggregateNode", ...]


def _walk_aggregate_nodes(
    nodes: Sequence[ArtifactAggregateNode],
    *,
    depth: int = 1,
) -> tuple[ArtifactAggregateComponent, ...]:
    if depth > _MAX_AGGREGATE_DEPTH:
        raise ArtifactABIError(
            f"aggregate component tree exceeds depth {_MAX_AGGREGATE_DEPTH}"
        )
    if not 1 <= len(nodes) <= _MAX_AGGREGATE_CHILDREN:
        raise ArtifactABIError(
            "each aggregate tuple must contain "
            f"1..{_MAX_AGGREGATE_CHILDREN} children"
        )
    leaves: list[ArtifactAggregateComponent] = []
    for node in nodes:
        if isinstance(node, ArtifactAggregateComponent):
            leaves.append(node)
        elif type(node) is tuple:
            leaves.extend(_walk_aggregate_nodes(node, depth=depth + 1))
        else:
            raise ArtifactABIError("aggregate component tree has an invalid node")
        if len(leaves) > _MAX_AGGREGATE_LEAVES:
            raise ArtifactABIError(
                f"aggregate component tree exceeds {_MAX_AGGREGATE_LEAVES} leaves"
            )
    return tuple(leaves)


def _aggregate_node_to_data(node: ArtifactAggregateNode) -> object:
    if isinstance(node, ArtifactAggregateComponent):
        return node.to_dict()
    return [_aggregate_node_to_data(child) for child in node]


@dataclass(frozen=True)
class SlotResource:
    """One validator-owned value visible at a semantic slot call.

    ``capability_field`` names the canonical, validator-produced descriptor fact
    which can prove a miner specialization.  A scalar without such a field may be
    passed at runtime, but may not disappear into a compile-time specialization.
    """

    name: str
    kind: str
    access: str = "read"
    scalar_type: str | None = None
    capability_field: str | None = None
    provider_capabilities: tuple[str, ...] = ()
    # Derived validator facts name their exact provider resource.  This keeps
    # multi-group rank/size materialization generic instead of relying on magic
    # names such as one global ``group.rank``.
    provider_resource: str | None = None
    provider_projection: str | None = None

    def __post_init__(self) -> None:
        name = _canonical_resource(self.name, field="slot resource name")
        if not isinstance(self.kind, str) or self.kind not in _RESOURCE_KINDS:
            raise ArtifactABIError(
                f"slot resource {name!r} has unsupported kind {self.kind!r}"
            )
        if not isinstance(self.access, str) or self.access not in _ACCESS_MODES:
            raise ArtifactABIError(
                f"slot resource {name!r} has unsupported access {self.access!r}"
            )
        if self.kind == "scalar":
            if not isinstance(self.scalar_type, str) or self.scalar_type not in _SCALAR_TYPES:
                raise ArtifactABIError(
                    f"scalar slot resource {name!r} requires a canonical scalar_type"
                )
        elif self.scalar_type is not None:
            raise ArtifactABIError(
                f"non-scalar slot resource {name!r} may not declare scalar_type"
            )
        if self.capability_field is not None:
            if (
                self.kind != "scalar"
                or _CAPABILITY_RE.fullmatch(self.capability_field) is None
            ):
                raise ArtifactABIError(
                    f"slot resource {name!r} has an invalid capability_field"
                )
        if type(self.provider_capabilities) is not tuple or any(
            not isinstance(capability, str)
            or capability not in _GROUP_PROVIDER_CAPABILITIES
            for capability in self.provider_capabilities
        ):
            raise ArtifactABIError(
                f"slot resource {name!r} has unsupported provider_capabilities"
            )
        if len(set(self.provider_capabilities)) != len(self.provider_capabilities):
            raise ArtifactABIError(
                f"slot resource {name!r} repeats a provider capability"
            )
        if self.provider_capabilities and self.kind != "group":
            raise ArtifactABIError(
                "only a validator-owned group resource may expose provider capabilities"
            )
        if (self.provider_resource is None) != (self.provider_projection is None):
            raise ArtifactABIError(
                f"slot resource {name!r} provider provenance must declare both "
                "provider_resource and provider_projection"
            )
        if self.provider_resource is not None:
            provider_name = _canonical_resource(
                self.provider_resource,
                field=f"slot resource {name!r} provider_resource",
            )
            if (
                self.kind != "scalar"
                or not provider_name.startswith("group.")
                or self.provider_projection not in {"rank", "size"}
            ):
                raise ArtifactABIError(
                    f"slot resource {name!r} has unsupported provider provenance"
                )
            expected_capability = {
                "rank": "rank",
                "size": "world_size",
            }[self.provider_projection]
            if self.capability_field != expected_capability:
                raise ArtifactABIError(
                    f"slot resource {name!r} provider projection "
                    f"{self.provider_projection!r} requires capability_field "
                    f"{expected_capability!r}"
                )
        prefix = name.split(".", 1)[0]
        if prefix == "input" and self.access != "read":
            raise ArtifactABIError("slot inputs must be read-only resources")
        if prefix == "output" and self.access not in {"write", "readwrite"}:
            raise ArtifactABIError("slot outputs must be writable resources")
        if prefix == "stream" and self.kind != "stream":
            raise ArtifactABIError("stream resources must have kind='stream'")
        if prefix == "group" and self.kind not in {
            "group",
            "scalar",
            "pointer",
            "opaque",
        }:
            raise ArtifactABIError(
                "group resources must be group/scalar/pointer/opaque capabilities"
            )

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "access": self.access,
            "kind": self.kind,
            "name": self.name,
        }
        if self.scalar_type is not None:
            row["scalar_type"] = self.scalar_type
        if self.capability_field is not None:
            row["capability_field"] = self.capability_field
        if self.provider_capabilities:
            row["provider_capabilities"] = list(self.provider_capabilities)
        if self.provider_resource is not None:
            row["provider_resource"] = self.provider_resource
            row["provider_projection"] = self.provider_projection
        return row

    def identity_dict(self) -> dict[str, object]:
        """Semantic call identity, excluding optional policy availability.

        ``provider_capabilities`` and ``capability_field`` are validator-owned
        eligibility allowlists.  Growing either must not rename an existing call
        boundary or invalidate an artifact that did not request it.  Exact used
        projections are sealed separately in the artifact launch plan.
        """

        row = self.to_dict()
        row.pop("provider_capabilities", None)
        row.pop("capability_field", None)
        return row


@dataclass(frozen=True, order=True)
class ProviderCapabilityRequirement:
    """One versioned optional provider projection used by an artifact step."""

    source: str
    projection: str
    capability: str

    def __post_init__(self) -> None:
        source = _canonical_resource(
            self.source, field="provider capability requirement source"
        )
        if not source.startswith("group."):
            raise ArtifactABIError(
                "provider capability requirement source must be a group resource"
            )
        if (
            not isinstance(self.projection, str)
            or self.projection not in _GROUP_PROJECTION_CAPABILITY
        ):
            raise ArtifactABIError(
                "provider capability requirement projection is unsupported"
            )
        if (
            not isinstance(self.capability, str)
            or _PROVIDER_CAPABILITY_RE.fullmatch(self.capability) is None
        ):
            raise ArtifactABIError(
                "provider capability requirement must be a versioned identifier"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "capability": self.capability,
            "projection": self.projection,
            "source": self.source,
        }


@dataclass(frozen=True, order=True)
class SpecializationCapabilityRequirement:
    """Exact validator metadata predicate used by one specialization."""

    source: str
    capability_field: str

    def __post_init__(self) -> None:
        _canonical_resource(
            self.source, field="specialization capability requirement source"
        )
        if (
            not isinstance(self.capability_field, str)
            or _CAPABILITY_RE.fullmatch(self.capability_field) is None
        ):
            raise ArtifactABIError(
                "specialization capability requirement field is invalid"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "capability_field": self.capability_field,
            "source": self.source,
        }


@dataclass(frozen=True)
class ArtifactShapeFactor:
    """One leaf of a bounded, data-only artifact allocation extent.

    A leaf is either a positive static integer or one non-negative integer
    projection of a validator-owned :class:`SlotResource`.  Dynamic leaves carry
    an explicit live upper bound.  There are no expression strings, callbacks,
    environment lookups, or references to another artifact-owned allocation.
    """

    static: int | None = None
    source: str | None = None
    projection: str | None = None
    axis: int | None = None
    upper_bound: int | None = None

    def __post_init__(self) -> None:
        dynamic = self.source is not None
        if dynamic == (self.static is not None):
            raise ArtifactABIError(
                "artifact shape factor must declare exactly one of static or source"
            )
        if not dynamic:
            if (
                type(self.static) is not int
                or not 1 <= self.static <= _MAX_ARTIFACT_FACTOR
            ):
                raise ArtifactABIError(
                    "artifact static shape factor must be an integer in "
                    f"[1, {_MAX_ARTIFACT_FACTOR}]"
                )
            if (
                self.projection is not None
                or self.axis is not None
                or self.upper_bound is not None
            ):
                raise ArtifactABIError(
                    "artifact static shape factor may not declare dynamic fields"
                )
            return

        _canonical_resource(self.source, field="artifact shape factor source")
        if self.projection not in _ARTIFACT_SHAPE_PROJECTIONS:
            raise ArtifactABIError(
                "artifact shape factor projection must be one of "
                f"{sorted(_ARTIFACT_SHAPE_PROJECTIONS)!r}"
            )
        if (
            type(self.upper_bound) is not int
            or not 1 <= self.upper_bound <= _MAX_ARTIFACT_FACTOR
        ):
            raise ArtifactABIError(
                "artifact dynamic shape factor upper_bound must be an integer in "
                f"[1, {_MAX_ARTIFACT_FACTOR}]"
            )
        if self.projection == "shape":
            if type(self.axis) is not int or not 0 <= self.axis < _MAX_TENSOR_RANK:
                raise ArtifactABIError(
                    f"artifact shape factor axis must be in [0, {_MAX_TENSOR_RANK - 1}]"
                )
        elif self.axis is not None:
            raise ArtifactABIError(
                f"artifact shape projection {self.projection!r} may not declare an axis"
            )

    @property
    def maximum(self) -> int:
        return self.static if self.static is not None else self.upper_bound  # type: ignore[return-value]

    def validate_source(self, resource: SlotResource) -> None:
        """Authorize the dynamic projection against the immutable slot ABI."""

        if self.static is not None:
            return
        if resource.kind == "tensor":
            allowed = {"shape", "numel", "rank"}
        elif resource.kind == "scalar":
            allowed = {"value"} if resource.scalar_type in _INTEGER_TYPES else set()
            if resource.access == "write":
                allowed = set()
        elif resource.kind == "group":
            allowed = {"size"}
        else:
            allowed = set()
        if self.projection not in allowed:
            raise ArtifactABIError(
                f"artifact shape factor projection {self.projection!r} is not allowed "
                f"for validator resource {resource.name!r} kind {resource.kind!r}"
            )

    def to_dict(self) -> dict[str, object]:
        if self.static is not None:
            return {"static": self.static}
        row: dict[str, object] = {
            "projection": self.projection,  # type: ignore[dict-item]
            "source": self.source,  # type: ignore[dict-item]
            "upper_bound": self.upper_bound,  # type: ignore[dict-item]
        }
        if self.axis is not None:
            row["axis"] = self.axis
        return row


@dataclass(frozen=True)
class ArtifactShapeExtent:
    """One tensor dimension: ``ceil(product(factors) / divisor)``.

    This fixed operation is the entire allocation-expression language.  It is
    sufficient for tiled workspace dimensions while remaining straightforward
    for the validator to bound and re-evaluate from live slot facts.
    """

    factors: tuple[ArtifactShapeFactor, ...]
    divisor: int = 1

    def __post_init__(self) -> None:
        if type(self.factors) is not tuple or not 1 <= len(
            self.factors
        ) <= _MAX_ARTIFACT_FACTORS_PER_EXTENT:
            raise ArtifactABIError(
                "artifact shape extent must contain "
                f"1..{_MAX_ARTIFACT_FACTORS_PER_EXTENT} factors"
            )
        if not all(isinstance(factor, ArtifactShapeFactor) for factor in self.factors):
            raise ArtifactABIError("artifact shape extent factors have the wrong type")
        if (
            type(self.divisor) is not int
            or not 1 <= self.divisor <= _MAX_ARTIFACT_FACTOR
        ):
            raise ArtifactABIError(
                "artifact shape extent divisor must be an integer in "
                f"[1, {_MAX_ARTIFACT_FACTOR}]"
            )
        product = 1
        for factor in self.factors:
            product *= factor.maximum
            if product > _MAX_ARTIFACT_FACTOR_PRODUCT:
                raise ArtifactABIError(
                    "artifact shape extent maximum product exceeds signed 64-bit bounds"
                )
        maximum = (product + self.divisor - 1) // self.divisor
        if not 1 <= maximum <= _MAX_ARTIFACT_EXTENT:
            raise ArtifactABIError(
                "artifact shape extent maximum is outside "
                f"[1, {_MAX_ARTIFACT_EXTENT}]"
            )

    @property
    def maximum(self) -> int:
        product = math.prod(factor.maximum for factor in self.factors)
        return (product + self.divisor - 1) // self.divisor

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "factors": [factor.to_dict() for factor in self.factors]
        }
        if self.divisor != 1:
            row["divisor"] = self.divisor
        return row


@dataclass(frozen=True)
class ArtifactResource:
    """One miner-named, validator-allocated artifact tensor.

    Prefix and lifetime are a closed relation: ``workspace.*`` is call-local,
    ``state.*`` lives with the engine artifact entry, and ``prepared.*`` is
    populated at the validator-owned prepare boundary then consumed by run.
    ``scope`` is also closed: existing buffers are rank-local by default, while
    ``group_ipc`` authorizes a validator collective provider to allocate one
    persistent local buffer per rank and exchange mappings for that exact buffer.
    """

    name: str
    dtype: str
    alignment: int
    lifetime: str
    shape: tuple[ArtifactShapeExtent, ...]
    scope: str = "rank_local"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or _ARTIFACT_RESOURCE_RE.fullmatch(self.name) is None
        ):
            raise ArtifactABIError(
                "artifact resource name must be a miner-named "
                "workspace.*, prepared.*, or state.* resource"
            )
        prefix = self.name.split(".", 1)[0]
        expected_lifetime = _ARTIFACT_RESOURCE_LIFETIME[prefix]
        if self.lifetime != expected_lifetime:
            raise ArtifactABIError(
                f"artifact resource {self.name!r} requires lifetime "
                f"{expected_lifetime!r}"
            )
        if (
            not isinstance(self.scope, str)
            or self.scope not in _ARTIFACT_RESOURCE_SCOPES
        ):
            raise ArtifactABIError(
                f"artifact resource {self.name!r} has unsupported allocation scope "
                f"{self.scope!r}"
            )
        if self.scope == "group_ipc" and self.lifetime == "call":
            raise ArtifactABIError(
                f"group-IPC artifact resource {self.name!r} must have a persistent "
                "prepared or engine lifetime"
            )
        if not isinstance(self.dtype, str) or self.dtype not in _ARTIFACT_STORAGE_BYTES:
            raise ArtifactABIError(
                f"artifact resource {self.name!r} has unsupported storage dtype "
                f"{self.dtype!r}"
            )
        _power_of_two(
            self.alignment,
            field=f"artifact resource {self.name!r} alignment",
            maximum=_MAX_ARTIFACT_ALIGNMENT,
        )
        if self.alignment < _ARTIFACT_STORAGE_BYTES[self.dtype]:
            raise ArtifactABIError(
                f"artifact resource {self.name!r} alignment is smaller than its dtype"
            )
        if type(self.shape) is not tuple or not 1 <= len(
            self.shape
        ) <= _MAX_ARTIFACT_RESOURCE_RANK:
            raise ArtifactABIError(
                f"artifact resource shape must have rank 1..{_MAX_ARTIFACT_RESOURCE_RANK}"
            )
        if not all(isinstance(extent, ArtifactShapeExtent) for extent in self.shape):
            raise ArtifactABIError("artifact resource shape has the wrong extent type")
        if self.max_elements > _MAX_ARTIFACT_ELEMENTS:
            raise ArtifactABIError(
                f"artifact resource {self.name!r} exceeds the element-count cap"
            )
        if self.max_bytes > _MAX_ARTIFACT_RESOURCE_BYTES:
            raise ArtifactABIError(
                f"artifact resource {self.name!r} exceeds the per-buffer byte cap "
                f"of {_MAX_ARTIFACT_RESOURCE_BYTES}"
            )

    @property
    def max_elements(self) -> int:
        return math.prod(extent.maximum for extent in self.shape)

    @property
    def max_bytes(self) -> int:
        return self.max_elements * _ARTIFACT_STORAGE_BYTES[self.dtype]

    @property
    def slot_resource(self) -> SlotResource:
        return SlotResource(self.name, "tensor", access="readwrite")

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "alignment": self.alignment,
            "dtype": self.dtype,
            "lifetime": self.lifetime,
            "name": self.name,
            "shape": [extent.to_dict() for extent in self.shape],
        }
        if self.scope != "rank_local":
            row["scope"] = self.scope
        return row


@dataclass(frozen=True)
class ArtifactBinding:
    """One ordered argument of a sealed artifact export.

    ``projection`` is selected from a closed validator vocabulary.  A tensor can
    remain a descriptor, expose its raw device pointer, or yield bounded metadata
    scalars.  ``kind='aggregate'`` constructs a sealed CuTe algebra value from a
    bounded data-only component tree.  No form can name candidate Python.

    Descriptor transformations are intentionally view-only. ``unsqueeze`` cannot
    allocate/copy storage; ``assumed_align`` is checked against the live pointer;
    and ``leading_dim`` selects CuTe's bounded dynamic-layout annotation.  A
    ``peer_ptr_table`` carries the canonical name of the exact artifact-owned
    ``group_ipc`` buffer mapped across the validator's group.
    """

    source: str | None
    kind: str
    cast: str | None = None
    unsqueeze: tuple[int, ...] = ()
    assumed_align: int | None = None
    leading_dim: int | None = None
    projection: str | None = None
    axis: int | None = None
    component_cast: str | None = None
    components: tuple[ArtifactAggregateNode, ...] = ()
    peer_resource: str | None = None

    def __post_init__(self) -> None:
        binding_kinds = _RESOURCE_KINDS | {"aggregate"}
        if not isinstance(self.kind, str) or self.kind not in binding_kinds:
            raise ArtifactABIError(
                f"artifact binding {self.source!r} has unsupported kind {self.kind!r}"
            )
        if self.kind == "aggregate":
            if self.source is not None:
                raise ArtifactABIError("aggregate binding may not declare source")
        else:
            _canonical_resource(self.source, field="artifact binding source")

        projection = self.projection
        if projection is None:
            projection = _DEFAULT_PROJECTION_BY_BINDING_KIND[self.kind]
            object.__setattr__(self, "projection", projection)
        if not isinstance(projection, str) or projection not in _PROJECTIONS:
            raise ArtifactABIError(
                f"artifact binding {self.source!r} has unsupported projection "
                f"{projection!r}"
            )

        if self.kind == "scalar":
            if not isinstance(self.cast, str) or self.cast not in _SCALAR_TYPES:
                raise ArtifactABIError(
                    f"scalar binding {self.source!r} requires a canonical cast"
                )
            if projection in _DERIVED_INTEGER_PROJECTIONS and self.cast not in _INTEGER_TYPES:
                raise ArtifactABIError(
                    f"derived tensor scalar {self.source!r} requires an integer cast"
                )
        elif self.kind == "aggregate":
            if self.cast not in _AGGREGATE_TYPES:
                raise ArtifactABIError(
                    "aggregate binding cast must be Shape/Coord/Tile/IntTuple/Stride"
                )
        elif self.cast is not None:
            raise ArtifactABIError(
                f"binding kind {self.kind!r} may not declare a cast"
            )

        if self.kind == "tensor":
            if projection != "descriptor":
                raise ArtifactABIError("tensor binding projection must be descriptor")
            if type(self.unsqueeze) is not tuple or len(self.unsqueeze) > _MAX_UNSQUEEZE:
                raise ArtifactABIError(
                    f"tensor binding {self.source!r} has too many unsqueeze operations"
                )
            if any(type(dim) is not int or not -16 <= dim <= 16 for dim in self.unsqueeze):
                raise ArtifactABIError(
                    f"tensor binding {self.source!r} has an invalid unsqueeze dimension"
                )
            if self.assumed_align is not None:
                _power_of_two(
                    self.assumed_align,
                    field=f"tensor binding {self.source!r} assumed_align",
                )
            if self.leading_dim is not None and (
                type(self.leading_dim) is not int or not 0 <= self.leading_dim <= 16
            ):
                raise ArtifactABIError(
                    f"tensor binding {self.source!r} leading_dim is outside [0, 16]"
                )
        elif self.kind == "pointer":
            if projection not in {
                "identity",
                "device_ptr",
                "native_handle",
                "peer_ptr_table",
            }:
                raise ArtifactABIError(
                    "pointer binding has an unsupported pointer projection"
                )
            if self.assumed_align is not None:
                _power_of_two(
                    self.assumed_align,
                    field=f"pointer binding {self.source!r} assumed_align",
                )
            if self.unsqueeze or self.leading_dim is not None:
                raise ArtifactABIError(
                    f"pointer binding {self.source!r} has descriptor-only options"
                )
        elif self.unsqueeze or self.assumed_align is not None or self.leading_dim is not None:
            raise ArtifactABIError(
                f"non-tensor binding {self.source!r} has tensor-only options"
            )

        if projection in _AXIS_PROJECTIONS:
            if type(self.axis) is not int or not 0 <= self.axis < _MAX_TENSOR_RANK:
                raise ArtifactABIError(
                    f"artifact binding axis must be in [0, {_MAX_TENSOR_RANK - 1}]"
                )
        elif self.axis is not None:
            raise ArtifactABIError(
                f"artifact projection {projection!r} may not declare an axis"
            )

        if projection == "peer_ptr_table":
            _canonical_resource(
                self.peer_resource,
                field="peer_ptr_table binding peer_resource",
            )
        elif self.peer_resource is not None:
            raise ArtifactABIError(
                "peer_resource is valid only for a peer_ptr_table projection"
            )

        if self.kind == "aggregate":
            if projection != "tuple":
                raise ArtifactABIError("aggregate binding projection must be tuple")
            if self.component_cast not in _AGGREGATE_COMPONENT_CASTS:
                raise ArtifactABIError(
                    "aggregate binding component_cast must be i32 or i64"
                )
            if type(self.components) is not tuple:
                raise ArtifactABIError("aggregate binding components must be a tuple")
            leaves = _walk_aggregate_nodes(self.components)
            dynamic = tuple(leaf for leaf in leaves if leaf.dynamic)
            if len(dynamic) > _MAX_AGGREGATE_DYNAMIC_ARITY:
                raise ArtifactABIError(
                    "aggregate binding exceeds dynamic arity "
                    f"{_MAX_AGGREGATE_DYNAMIC_ARITY}"
                )
            if any(leaf.cast != self.component_cast for leaf in dynamic):
                raise ArtifactABIError(
                    "aggregate dynamic leaf casts must equal component_cast"
                )
        elif self.component_cast is not None or self.components:
            raise ArtifactABIError(
                f"non-aggregate binding {self.source!r} has aggregate-only options"
            )

    @property
    def dynamic_arity(self) -> int:
        if self.kind != "aggregate":
            return 1
        return sum(leaf.dynamic for leaf in _walk_aggregate_nodes(self.components))

    @property
    def integer_range(self) -> tuple[int, int] | None:
        if self.kind == "scalar" and self.cast in _INTEGER_TYPES:
            return integer_cast_bounds(self.cast)
        return None

    @property
    def aggregate_leaves(self) -> tuple[ArtifactAggregateComponent, ...]:
        if self.kind != "aggregate":
            return ()
        return _walk_aggregate_nodes(self.components)

    @property
    def native_projection(self) -> tuple[object, ...]:
        """Canonical flattened native signature consumed by artifact ABI gates.

        The semantic aggregate class/tree and pointer provenance remain in the
        full binding row.  This projection describes only what the compiled
        header can prove at the native call boundary.
        """

        if self.kind == "aggregate":
            return ("aggregate", self.component_cast, self.dynamic_arity)
        if self.kind == "scalar":
            native_cast = {"bool": "u8", "tf32": "f32"}.get(self.cast, self.cast)
            return ("scalar", native_cast)
        if self.kind == "pointer":
            return ("pointer", None)
        if self.kind == "opaque":
            return ("pointer", None)
        return (self.kind, None)

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "kind": self.kind,
            "projection": self.projection,  # type: ignore[dict-item]
        }
        if self.source is not None:
            row["source"] = self.source
        if self.cast is not None:
            row["cast"] = self.cast
        if self.unsqueeze:
            row["unsqueeze"] = list(self.unsqueeze)
        if self.assumed_align is not None:
            row["assumed_align"] = self.assumed_align
        if self.leading_dim is not None:
            row["leading_dim"] = self.leading_dim
        if self.axis is not None:
            row["axis"] = self.axis
        if self.component_cast is not None:
            row["component_cast"] = self.component_cast
        if self.components:
            row["components"] = [
                _aggregate_node_to_data(node) for node in self.components
            ]
        if self.peer_resource is not None:
            row["peer_resource"] = self.peer_resource
        return row


@dataclass(frozen=True)
class ArtifactPrelaunch:
    """A bounded validator-executed operation included in the candidate call."""

    op: str
    target: str
    value: bool | int | float | str

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or self.op != "fill":
            raise ArtifactABIError(
                f"unsupported artifact prelaunch operation {self.op!r}"
            )
        _canonical_resource(self.target, field="artifact prelaunch target")
        value = self.value
        if isinstance(value, str):
            if value not in {"-inf", "+inf", "nan"}:
                raise ArtifactABIError(
                    "string fill values must be '-inf', '+inf', or 'nan'"
                )
        elif isinstance(value, float) and not math.isfinite(value):
            raise ArtifactABIError(
                "non-finite fill values must use their canonical string spelling"
            )
        elif not isinstance(value, (bool, int, float)):
            raise ArtifactABIError("artifact fill value must be a bounded scalar")
        elif type(value) is int and not -(1 << 63) <= value <= (1 << 64) - 1:
            raise ArtifactABIError("artifact integer fill value is outside 64-bit bounds")

    def to_dict(self) -> dict[str, object]:
        return {"op": self.op, "target": self.target, "value": self.value}


@dataclass(frozen=True)
class ArtifactResourcePlan:
    """Closed artifact-owned storage namespace authorized by one slot ABI."""

    slot: str
    resources: tuple[ArtifactResource, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.slot, str) or not self.slot or len(self.slot) > 128:
            raise ArtifactABIError("artifact resource plan requires a bounded slot name")
        if type(self.resources) is not tuple or len(
            self.resources
        ) > _MAX_ARTIFACT_RESOURCES:
            raise ArtifactABIError(
                "artifact resource plan may contain at most "
                f"{_MAX_ARTIFACT_RESOURCES} buffers"
            )
        if not all(isinstance(resource, ArtifactResource) for resource in self.resources):
            raise ArtifactABIError("artifact resource plan rows have the wrong type")
        names = tuple(resource.name for resource in self.resources)
        if len(set(names)) != len(names):
            raise ArtifactABIError("artifact resource plan repeats a resource name")
        total = sum(resource.max_bytes for resource in self.resources)
        if total > _MAX_ARTIFACT_TOTAL_BYTES:
            raise ArtifactABIError(
                "artifact resource plan exceeds the aggregate byte cap "
                f"of {_MAX_ARTIFACT_TOTAL_BYTES}"
            )

    @property
    def by_name(self) -> Mapping[str, ArtifactResource]:
        return MappingProxyType({resource.name: resource for resource in self.resources})

    @property
    def slot_resources(self) -> tuple[SlotResource, ...]:
        return tuple(resource.slot_resource for resource in self.resources)

    def validate_for(self, call_abi: SlotCallABI) -> None:
        """Bind allocation facts only to the immutable validator slot boundary."""

        if self.slot != call_abi.slot:
            raise ArtifactABIError(
                f"artifact resource plan for {self.slot!r} cannot extend slot "
                f"{call_abi.slot!r}"
            )
        base = call_abi.by_name
        collisions = set(base) & set(self.by_name)
        if collisions:
            raise ArtifactABIError(
                "artifact resources collide with validator slot resources "
                f"{sorted(collisions)!r}"
            )
        # Prepared allocation shapes may use only resources present at the exact
        # validator-owned allocation boundary.  An explicit prepare-result slot
        # allocates in ``prepare(*prepare_args)``; a result-free slot allocates
        # immediately before run from the positional call frame.  This grants the
        # allocator tensor metadata only: prepare-export bindings remain restricted
        # to ``prepare_args`` by ``SlotCallABI.validate_plan`` and therefore cannot
        # read activation contents merely because an allocation shape names one.
        prepared_shape_sources = set(
            call_abi.prepare_args
            if call_abi.prepare_result is not None
            else call_abi.call_args
        )
        for declaration in self.resources:
            for extent in declaration.shape:
                for factor in extent.factors:
                    if factor.source is None:
                        continue
                    try:
                        source = base[factor.source]
                    except KeyError:
                        raise ArtifactABIError(
                            f"artifact resource {declaration.name!r} shape references "
                            f"unknown validator slot resource {factor.source!r}"
                        ) from None
                    factor.validate_source(source)
                    if (
                        declaration.lifetime == "prepared"
                        and factor.source not in prepared_shape_sources
                    ):
                        raise ArtifactABIError(
                            f"prepared artifact resource {declaration.name!r} shape "
                            f"source {factor.source!r} is unavailable at the validator "
                            "prepare boundary"
                        )

    def _storage_roles(
        self,
        steps: Sequence[
            tuple[str, Sequence[ArtifactBinding], Sequence[ArtifactPrelaunch]]
        ],
        *,
        include_prelaunch: bool = True,
    ) -> Mapping[str, frozenset[str]]:
        generated = {resource.name: resource.slot_resource for resource in self.resources}
        roles: dict[str, set[str]] = {name: set() for name in generated}
        for role, bindings, prelaunch in steps:
            for binding in bindings:
                if binding.source in generated and _binding_writes_resource(
                    binding, generated[binding.source]
                ):
                    roles[binding.source].add(role)  # type: ignore[index]
                if binding.peer_resource in generated:
                    # A peer table exposes the declared local buffer through every
                    # rank's mapped pointer, even when the artifact does not also
                    # request a direct tensor descriptor for its own rank.
                    roles[binding.peer_resource].add(role)  # type: ignore[index]
            if include_prelaunch:
                for operation in prelaunch:
                    if operation.target in generated:
                        roles[operation.target].add(role)
        return MappingProxyType(
            {name: frozenset(resource_roles) for name, resource_roles in roles.items()}
        )

    def validate_pipeline(
        self,
        steps: Sequence[
            tuple[str, Sequence[ArtifactBinding], Sequence[ArtifactPrelaunch]]
        ],
        *,
        require_all: bool = False,
    ) -> None:
        """Validate storage lifetimes for one plan or an op's complete plan set.

        A ``prepared.*`` tensor is validator-allocated, exposed to a prepare
        export for population, retained under the prepare key, then exposed to a
        run export.  This models a prepare result without executing candidate
        Python in the engine.  ``workspace.*`` cannot escape a call into init or
        destroy phases; ``state.*`` is the explicitly engine-persistent tier.
        """

        roles = self._storage_roles(steps)
        binding_roles = self._storage_roles(steps, include_prelaunch=False)
        generated = self.by_name
        for role, bindings, prelaunch in steps:
            if role != "destroy":
                continue
            names = {
                binding.source
                for binding in bindings
                if binding.source in generated
            }
            names.update(
                leaf.source
                for binding in bindings
                if binding.kind == "aggregate"
                for leaf in binding.aggregate_leaves
                if leaf.source in generated
            )
            names.update(
                binding.peer_resource
                for binding in bindings
                if binding.peer_resource in generated
            )
            names.update(
                operation.target
                for operation in prelaunch
                if operation.target in generated
            )
            classes = {
                (generated[name].lifetime, generated[name].scope)
                for name in names
            }
            if len(classes) > 1:
                raise ArtifactABIError(
                    "one artifact destroy export may not mix generated resources "
                    "with different lifetime/scope classes"
                )
        for resource in self.resources:
            used = roles[resource.name]
            if require_all and not used:
                raise ArtifactABIError(
                    f"artifact resource {resource.name!r} is declared but never exposed"
                )
            if not used:
                continue
            prefix = resource.name.split(".", 1)[0]
            if prefix == "workspace" and used & {"init", "destroy"}:
                raise ArtifactABIError(
                    f"call-lifetime artifact resource {resource.name!r} may not be "
                    "used by init/destroy exports"
                )
            if prefix == "prepared":
                forbidden = used & {"init", "reset"}
                if forbidden:
                    raise ArtifactABIError(
                        f"prepared artifact resource {resource.name!r} has invalid "
                        f"lifecycle roles {sorted(forbidden)!r}"
                    )
                if "prepare" not in used or "run" not in binding_roles[resource.name]:
                    raise ArtifactABIError(
                        f"prepared artifact resource {resource.name!r} requires both "
                        "prepare producer and run consumer exports"
                    )

    def to_dict(self) -> dict[str, object]:
        return {
            "resources": [resource.to_dict() for resource in self.resources],
            "slot": self.slot,
        }


@dataclass(frozen=True)
class SlotCallABI:
    """The validator-owned resources available to implementations of one slot."""

    slot: str
    resources: tuple[SlotResource, ...]
    # Canonical positional ABI seen by legacy Python entries and dispatchers.
    # Provider-added resources such as ``stream.current`` do not appear here.
    # Keeping this mapping explicit prevents resource-table ordering from silently
    # changing the semantic entry signature.
    call_args: tuple[str, ...]
    # Prepare+forward slots have a second positional boundary. The prepare result
    # is validator-held state and appears as one resource in the forward call.
    prepare_args: tuple[str, ...] = ()
    prepare_result: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slot, str) or not self.slot or len(self.slot) > 128:
            raise ArtifactABIError("slot call ABI requires a bounded slot name")
        if type(self.resources) is not tuple or not self.resources:
            raise ArtifactABIError("slot call ABI requires a nonempty resource tuple")
        if not all(isinstance(resource, SlotResource) for resource in self.resources):
            raise ArtifactABIError("slot call ABI resources have the wrong type")
        names = tuple(resource.name for resource in self.resources)
        if len(set(names)) != len(names):
            raise ArtifactABIError(f"slot call ABI {self.slot!r} repeats a resource")
        resources_by_name = {resource.name: resource for resource in self.resources}
        for resource in self.resources:
            if resource.provider_resource is None:
                continue
            provider = resources_by_name.get(resource.provider_resource)
            if provider is None or provider.kind != "group":
                raise ArtifactABIError(
                    f"slot resource {resource.name!r} derives from a missing/non-group "
                    f"provider {resource.provider_resource!r}"
                )
        if not any(name.startswith("output.") for name in names):
            raise ArtifactABIError(f"slot call ABI {self.slot!r} declares no output")
        if type(self.call_args) is not tuple or not self.call_args:
            raise ArtifactABIError(
                f"slot call ABI {self.slot!r} requires explicit positional call_args"
            )
        for index, name in enumerate(self.call_args):
            _canonical_resource(name, field=f"slot call_args[{index}]")
            if name not in names:
                raise ArtifactABIError(
                    f"slot call_args[{index}] references unknown resource {name!r}"
                )
        if len(set(self.call_args)) != len(self.call_args):
            raise ArtifactABIError(f"slot call ABI {self.slot!r} repeats a call_arg")
        if type(self.prepare_args) is not tuple:
            raise ArtifactABIError(
                f"slot call ABI {self.slot!r} prepare_args must be a tuple"
            )
        for index, name in enumerate(self.prepare_args):
            _canonical_resource(name, field=f"slot prepare_args[{index}]")
            if name not in names:
                raise ArtifactABIError(
                    f"slot prepare_args[{index}] references unknown resource {name!r}"
                )
            if name.startswith("output."):
                raise ArtifactABIError("slot prepare_args may not consume run outputs")
        if len(set(self.prepare_args)) != len(self.prepare_args):
            raise ArtifactABIError(f"slot call ABI {self.slot!r} repeats a prepare_arg")

        required_outputs = {name for name in names if name.startswith("output.")}
        missing_outputs = required_outputs - set(self.call_args)
        if missing_outputs:
            raise ArtifactABIError(
                f"slot call ABI {self.slot!r} positional call_args omit outputs "
                f"{sorted(missing_outputs)!r}"
            )
        required_inputs = {name for name in names if name.startswith("input.")}
        missing_inputs = required_inputs - set(self.call_args) - set(self.prepare_args)
        if missing_inputs:
            raise ArtifactABIError(
                f"slot call ABI {self.slot!r} positional boundaries omit inputs "
                f"{sorted(missing_inputs)!r}"
            )
        if self.prepare_result is not None:
            if not self.prepare_args:
                raise ArtifactABIError(
                    "slot prepare_result requires a nonempty prepare_args boundary"
                )
            result = _canonical_resource(
                self.prepare_result, field="slot prepare_result"
            )
            if result not in names or not result.startswith("prepared."):
                raise ArtifactABIError(
                    "slot prepare_result must name a declared prepared.* resource"
                )
            if result not in self.call_args:
                raise ArtifactABIError(
                    "slot prepare_result must be consumed by positional call_args"
                )
        elif self.prepare_args and not set(self.prepare_args) <= set(self.call_args):
            raise ArtifactABIError(
                "result-free prepare_args must remain available in call_args for "
                "validator-owned implicit pre-run preparation"
            )

    @property
    def by_name(self) -> Mapping[str, SlotResource]:
        return MappingProxyType({resource.name: resource for resource in self.resources})

    def identity_snapshot(self) -> dict[str, object]:
        """Stable semantic ABI identity without optional policy allowlists."""

        return {
            "slot": self.slot,
            "resources": [resource.identity_dict() for resource in self.resources],
            "call_args": list(self.call_args),
            "prepare_args": list(self.prepare_args),
            "prepare_result": self.prepare_result,
        }

    def provider_capability_requirements(
        self,
        bindings: Sequence[ArtifactBinding],
        *,
        artifact_resources: ArtifactResourcePlan | None = None,
    ) -> tuple[ProviderCapabilityRequirement, ...]:
        """Derive exact versioned optional capabilities used by bindings.

        The live call ABI retains the complete current allowlist for admission.
        This projection includes only capabilities actually requested by this
        artifact step, so unrelated allowlist growth is identity-neutral while a
        removed or remapped requested capability cannot reopen silently.
        """

        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
            raise ArtifactABIError("artifact bindings must be a sequence")
        resources = self.resource_table(artifact_resources)
        requirements: set[ProviderCapabilityRequirement] = set()
        for index, binding in enumerate(bindings):
            if not isinstance(binding, ArtifactBinding):
                raise ArtifactABIError(
                    f"artifact binding[{index}] has the wrong type"
                )
            if binding.source is None:
                continue
            resource = resources.get(binding.source)
            if resource is None or resource.kind != "group":
                continue
            capability = _GROUP_PROJECTION_CAPABILITY.get(binding.projection)
            if capability is None:
                continue
            if capability not in resource.provider_capabilities:
                raise ArtifactABIError(
                    f"artifact binding[{index}] group projection "
                    f"{binding.projection!r} requires validator capability "
                    f"{capability!r}"
                )
            requirements.add(
                ProviderCapabilityRequirement(
                    source=binding.source,
                    projection=str(binding.projection),
                    capability=capability,
                )
            )
        return tuple(sorted(requirements))

    def specialization_capability_requirements(
        self,
        specializes: Sequence[tuple[str, bool | int | float | str]],
        *,
        artifact_resources: ArtifactResourcePlan | None = None,
    ) -> tuple[SpecializationCapabilityRequirement, ...]:
        """Derive exact optional metadata predicates used by specializations."""

        if not isinstance(specializes, Sequence) or isinstance(
            specializes, (str, bytes)
        ):
            raise ArtifactABIError("artifact specializations must be a sequence")
        resources = self.resource_table(artifact_resources)
        requirements: set[SpecializationCapabilityRequirement] = set()
        for index, row in enumerate(specializes):
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
            ):
                raise ArtifactABIError(
                    f"artifact specialization[{index}] has the wrong type"
                )
            source = row[0]
            resource = resources.get(source)
            if resource is None or resource.capability_field is None:
                raise ArtifactABIError(
                    f"artifact specialization {source!r} lacks a validator "
                    "capability fact"
                )
            requirements.add(
                SpecializationCapabilityRequirement(
                    source=source,
                    capability_field=resource.capability_field,
                )
            )
        return tuple(sorted(requirements))

    def resource_table(
        self,
        artifact_resources: ArtifactResourcePlan | None = None,
    ) -> Mapping[str, SlotResource]:
        """Return semantic resources plus a validated artifact-owned namespace."""

        if artifact_resources is None:
            return self.by_name
        if not isinstance(artifact_resources, ArtifactResourcePlan):
            raise ArtifactABIError("artifact resources have the wrong plan type")
        artifact_resources.validate_for(self)
        resources = dict(self.by_name)
        resources.update(
            (resource.name, resource) for resource in artifact_resources.slot_resources
        )
        return MappingProxyType(resources)

    def validate_plan(
        self,
        *,
        role: str,
        bindings: Sequence[ArtifactBinding],
        specializes: Mapping[str, bool | int | float | str],
        prelaunch: Sequence[ArtifactPrelaunch],
        require_outputs: bool = True,
        artifact_resources: ArtifactResourcePlan | None = None,
    ) -> tuple[tuple[str, bool | int | float | str], ...]:
        """Validate and canonicalize one implementation projection.

        The return value is the sorted specialization tuple used in the sealed
        artifact identity.  It contains only validator-descriptor-backed scalar
        resources; a miner cannot hide a dynamic input by calling it constexpr.
        """

        if not isinstance(role, str) or role not in _ARTIFACT_ROLES:
            raise ArtifactABIError(f"unsupported artifact export role {role!r}")
        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
            raise ArtifactABIError("artifact bindings must be a sequence")
        if not 1 <= len(bindings) <= _MAX_BINDINGS:
            raise ArtifactABIError(
                f"artifact export must declare 1..{_MAX_BINDINGS} bindings"
            )
        resources = self.resource_table(artifact_resources)
        written: set[str] = set()
        for index, binding in enumerate(bindings):
            if not isinstance(binding, ArtifactBinding):
                raise ArtifactABIError(f"artifact binding[{index}] has the wrong type")
            if binding.kind == "aggregate":
                for leaf_index, leaf in enumerate(binding.aggregate_leaves):
                    if not leaf.dynamic:
                        continue
                    try:
                        component_resource = resources[leaf.source]  # type: ignore[index]
                    except KeyError:
                        raise ArtifactABIError(
                            f"artifact binding[{index}] aggregate leaf[{leaf_index}] "
                            f"references unknown slot resource {leaf.source!r}"
                        ) from None
                    if leaf.projection in {"shape", "stride"}:
                        if component_resource.kind != "tensor":
                            raise ArtifactABIError(
                                f"artifact binding[{index}] aggregate leaf[{leaf_index}] "
                                "shape/stride projection requires a tensor resource"
                            )
                    elif component_resource.kind != "scalar":
                        raise ArtifactABIError(
                            f"artifact binding[{index}] aggregate leaf[{leaf_index}] "
                            "value projection requires a scalar resource"
                        )
                    else:
                        if component_resource.access == "write":
                            raise ArtifactABIError(
                                f"artifact binding[{index}] aggregate leaf[{leaf_index}] "
                                "cannot read a write-only scalar resource"
                            )
                        _validate_scalar_cast(
                            source=component_resource,
                            cast=leaf.cast,
                            field=(
                                f"artifact binding[{index}] aggregate leaf[{leaf_index}]"
                            ),
                        )
                continue

            try:
                resource = resources[binding.source]  # type: ignore[index]
            except KeyError:
                raise ArtifactABIError(
                    f"artifact binding[{index}] references unknown slot resource "
                    f"{binding.source!r}"
                ) from None
            if (
                role == "run"
                and self.prepare_result is not None
                and binding.source == self.prepare_result
            ):
                raise ArtifactABIError(
                    "sealed artifacts may not bind the opaque semantic prepare "
                    "envelope directly; declare validator-allocated prepared.* "
                    "tensor resources and populate them in prepare exports"
                )
            allowed = _SOURCE_BINDING_PROJECTIONS[resource.kind]
            if (binding.kind, binding.projection) not in allowed:
                raise ArtifactABIError(
                    f"artifact binding[{index}] projection "
                    f"{(binding.kind, binding.projection)!r} is not allowed for "
                    f"slot resource {binding.source!r} kind {resource.kind!r}"
                )
            if resource.kind == "scalar":
                if resource.access == "write":
                    raise ArtifactABIError(
                        f"artifact binding[{index}] cannot pass write-only scalar "
                        f"resource {binding.source!r} by value"
                    )
                _validate_scalar_cast(
                    source=resource,
                    cast=binding.cast,
                    field=f"artifact binding[{index}]",
                )
            elif resource.kind == "group":
                capability = _GROUP_PROJECTION_CAPABILITY.get(binding.projection)
                if (
                    capability is not None
                    and capability not in resource.provider_capabilities
                ):
                    raise ArtifactABIError(
                        f"artifact binding[{index}] group projection "
                        f"{binding.projection!r} requires validator capability "
                        f"{capability!r}"
                    )
                if binding.projection == "peer_ptr_table":
                    assert binding.peer_resource is not None
                    if artifact_resources is None:
                        raise ArtifactABIError(
                            f"artifact binding[{index}] peer_ptr_table must name an "
                            "exact declared artifact buffer"
                        )
                    try:
                        peer_resource = artifact_resources.by_name[
                            binding.peer_resource
                        ]
                    except KeyError:
                        raise ArtifactABIError(
                            f"artifact binding[{index}] peer_ptr_table references "
                            "undeclared artifact buffer "
                            f"{binding.peer_resource!r}"
                        ) from None
                    if (
                        peer_resource.scope != "group_ipc"
                        or peer_resource.lifetime == "call"
                    ):
                        raise ArtifactABIError(
                            f"artifact binding[{index}] peer_ptr_table buffer "
                            f"{binding.peer_resource!r} must be a persistent "
                            "group-IPC artifact resource"
                        )
            if _binding_writes_resource(binding, resource):
                written.add(resource.name)

        if not isinstance(specializes, Mapping):
            raise ArtifactABIError("artifact specializes must be a mapping")
        normalized_specializes: list[tuple[str, bool | int | float | str]] = []
        for source, value in specializes.items():
            source = _canonical_resource(source, field="artifact specialization source")
            try:
                resource = resources[source]
            except KeyError:
                raise ArtifactABIError(
                    f"artifact specialization references unknown slot resource {source!r}"
                ) from None
            if resource.kind != "scalar" or resource.capability_field is None:
                raise ArtifactABIError(
                    f"artifact specialization {source!r} lacks a validator capability fact"
                )
            if resource.scalar_type == "bool":
                valid_value = type(value) is bool
            elif resource.scalar_type in _INTEGER_TYPES:
                valid_value = type(value) is int
            else:
                valid_value = type(value) in {int, float}
            if not valid_value:
                raise ArtifactABIError(
                    f"artifact specialization {source!r} has the wrong scalar type"
                )
            if resource.scalar_type in _INTEGER_CAST_BOUNDS:
                minimum, maximum = integer_cast_bounds(resource.scalar_type)
                if not minimum <= value <= maximum:  # type: ignore[operator]
                    raise ArtifactABIError(
                        f"artifact specialization {source!r} is outside "
                        f"{resource.scalar_type}"
                    )
            elif resource.scalar_type in _FLOAT_TYPES and not math.isfinite(
                float(value)
            ):
                raise ArtifactABIError(
                    f"artifact specialization {source!r} must be finite"
                )
            normalized_specializes.append((source, value))
        normalized_specializes.sort(key=lambda row: row[0])

        if not isinstance(prelaunch, Sequence) or isinstance(prelaunch, (str, bytes)):
            raise ArtifactABIError("artifact prelaunch must be a sequence")
        if len(prelaunch) > _MAX_PRELAUNCH:
            raise ArtifactABIError(
                f"artifact prelaunch may contain at most {_MAX_PRELAUNCH} operations"
            )
        initialized: set[str] = set()
        for index, operation in enumerate(prelaunch):
            if not isinstance(operation, ArtifactPrelaunch):
                raise ArtifactABIError(
                    f"artifact prelaunch[{index}] has the wrong type"
                )
            try:
                target = resources[operation.target]
            except KeyError:
                raise ArtifactABIError(
                    f"artifact prelaunch[{index}] references unknown resource "
                    f"{operation.target!r}"
                ) from None
            if target.kind != "tensor" or target.access not in {"write", "readwrite"}:
                raise ArtifactABIError(
                    f"artifact prelaunch target {operation.target!r} is not a writable tensor"
                )
            initialized.add(operation.target)

        used_sources = {
            binding.source
            for binding in bindings
            if binding.source is not None
        }
        used_sources.update(
            leaf.source
            for binding in bindings
            if binding.kind == "aggregate"
            for leaf in binding.aggregate_leaves
            if leaf.source is not None
        )
        used_sources.update(
            binding.peer_resource
            for binding in bindings
            if binding.peer_resource is not None
        )
        used_sources.update(operation.target for operation in prelaunch)
        provider_prefixes = ("workspace.", "state.", "group.", "stream.")
        lifecycle_prefixes = provider_prefixes + ("prepared.",)
        if role == "run":
            phase_allowed = set(self.call_args) | {
                name for name in resources if name.startswith(lifecycle_prefixes)
            }
        elif role == "prepare":
            phase_allowed = set(self.prepare_args) | {
                name for name in resources if name.startswith(lifecycle_prefixes)
            }
        elif role == "init":
            phase_allowed = {
                name for name in resources if name.startswith(provider_prefixes)
            }
        else:  # reset/destroy operate only on validator-held lifecycle resources.
            phase_allowed = {
                name for name in resources if name.startswith(lifecycle_prefixes)
            }
        unavailable = used_sources - phase_allowed
        if unavailable:
            raise ArtifactABIError(
                f"artifact {role} plan references resources outside its lifecycle "
                f"boundary {sorted(unavailable)!r}"
            )
        if role != "run":
            output_mutations = {
                source for source in used_sources if source.startswith("output.")
            }
            if output_mutations:
                raise ArtifactABIError(
                    f"artifact {role} plan may not mutate call outputs "
                    f"{sorted(output_mutations)!r}"
                )

        if role == "run" and require_outputs:
            required_outputs = {
                resource.name
                for resource in self.resources
                if resource.name.startswith("output.")
            }
            missing = required_outputs - written - initialized
            if missing:
                raise ArtifactABIError(
                    f"artifact run plan does not write slot outputs {sorted(missing)!r}"
                )
        return tuple(normalized_specializes)

    def validate_pipeline(
        self,
        steps: Sequence[
            tuple[
                str,
                Sequence[ArtifactBinding],
                Sequence[ArtifactPrelaunch],
            ]
        ],
        *,
        artifact_resources: ArtifactResourcePlan | None = None,
    ) -> None:
        """Require a multi-export plan to collectively write every slot output."""

        resources = self.resource_table(artifact_resources)
        run_steps = [step for step in steps if step[0] == "run"]
        if not run_steps:
            raise ArtifactABIError("artifact plan contains no run export")
        role_order = {"init": 0, "prepare": 1, "reset": 2, "run": 3, "destroy": 4}
        try:
            ranks = [role_order[step[0]] for step in steps]
        except KeyError as exc:
            raise ArtifactABIError(
                f"artifact pipeline has unsupported role {exc.args[0]!r}"
            ) from None
        if ranks != sorted(ranks):
            raise ArtifactABIError(
                "artifact pipeline lifecycle roles must be ordered "
                "init/prepare/reset/run/destroy"
            )
        covered: set[str] = set()
        for _role, bindings, prelaunch in run_steps:
            for binding in bindings:
                if binding.source is None:
                    continue
                resource = resources.get(binding.source)
                if resource is not None and _binding_writes_resource(binding, resource):
                    covered.add(binding.source)
            covered.update(operation.target for operation in prelaunch)
        required_outputs = {
            resource.name
            for resource in self.resources
            if resource.name.startswith("output.")
        }
        missing = required_outputs - covered
        if missing:
            raise ArtifactABIError(
                f"artifact plan does not write slot outputs {sorted(missing)!r}"
            )
        if artifact_resources is not None:
            artifact_resources.validate_pipeline(steps)

    def specialization_capabilities(
        self,
        specializes: Sequence[tuple[str, bool | int | float | str]],
    ) -> dict[str, bool | int | float | str]:
        resources = self.by_name
        return {
            resources[source].capability_field: value  # type: ignore[misc]
            for source, value in specializes
        }

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "call_args": list(self.call_args),
            "resources": [resource.to_dict() for resource in self.resources],
            "slot": self.slot,
        }
        if self.prepare_args:
            row["prepare_args"] = list(self.prepare_args)
            row["prepare_result"] = self.prepare_result
        return row


def _parse_artifact_shape_factor(raw: object, *, field: str) -> ArtifactShapeFactor:
    if not isinstance(raw, Mapping):
        raise ArtifactABIError(f"{field} must be a shape-factor table")
    if set(raw) == {"static"}:
        return ArtifactShapeFactor(static=raw["static"])  # type: ignore[arg-type]
    required = {"source", "projection", "upper_bound"}
    optional = {"axis"}
    missing = required - set(raw)
    unknown = set(raw) - required - optional
    if missing or unknown:
        raise ArtifactABIError(
            f"{field} fields mismatch; missing={sorted(missing)!r}, "
            f"unknown={sorted(unknown)!r}"
        )
    return ArtifactShapeFactor(
        source=raw["source"],  # type: ignore[arg-type]
        projection=raw["projection"],  # type: ignore[arg-type]
        axis=raw.get("axis"),  # type: ignore[arg-type]
        upper_bound=raw["upper_bound"],  # type: ignore[arg-type]
    )


def parse_artifact_resources(
    raw: object,
    *,
    call_abi: SlotCallABI,
    field: str,
) -> ArtifactResourcePlan:
    """Parse and authorize per-op artifact buffers without importing a runtime."""

    if not isinstance(raw, (list, tuple)):
        raise ArtifactABIError(f"{field} must be a list of resource tables")
    if len(raw) > _MAX_ARTIFACT_RESOURCES:
        raise ArtifactABIError(
            f"{field} may contain at most {_MAX_ARTIFACT_RESOURCES} buffers"
        )
    resources: list[ArtifactResource] = []
    required = {"name", "dtype", "alignment", "lifetime", "shape"}
    optional = {"scope"}
    for resource_index, value in enumerate(raw):
        resource_field = f"{field}[{resource_index}]"
        if not isinstance(value, Mapping):
            raise ArtifactABIError(f"{resource_field} must be a table")
        missing = required - set(value)
        unknown = set(value) - required - optional
        if missing or unknown:
            raise ArtifactABIError(
                f"{resource_field} fields mismatch; missing={sorted(missing)!r}, "
                f"unknown={sorted(unknown)!r}"
            )
        raw_shape = value["shape"]
        if not isinstance(raw_shape, (list, tuple)):
            raise ArtifactABIError(
                f"{resource_field}.shape must be a list of extent tables"
            )
        extents: list[ArtifactShapeExtent] = []
        for extent_index, extent_raw in enumerate(raw_shape):
            extent_field = f"{resource_field}.shape[{extent_index}]"
            if not isinstance(extent_raw, Mapping):
                raise ArtifactABIError(f"{extent_field} must be an extent table")
            required_extent = {"factors"}
            optional_extent = {"divisor"}
            missing_extent = required_extent - set(extent_raw)
            unknown_extent = set(extent_raw) - required_extent - optional_extent
            if missing_extent or unknown_extent:
                raise ArtifactABIError(
                    f"{extent_field} fields mismatch; "
                    f"missing={sorted(missing_extent)!r}, "
                    f"unknown={sorted(unknown_extent)!r}"
                )
            raw_factors = extent_raw["factors"]
            if not isinstance(raw_factors, (list, tuple)):
                raise ArtifactABIError(
                    f"{extent_field}.factors must be a list of factor tables"
                )
            factors = tuple(
                _parse_artifact_shape_factor(
                    factor,
                    field=f"{extent_field}.factors[{factor_index}]",
                )
                for factor_index, factor in enumerate(raw_factors)
            )
            extents.append(
                ArtifactShapeExtent(
                    factors=factors,
                    divisor=extent_raw.get("divisor", 1),  # type: ignore[arg-type]
                )
            )
        resources.append(
            ArtifactResource(
                name=value["name"],  # type: ignore[arg-type]
                dtype=value["dtype"],  # type: ignore[arg-type]
                alignment=value["alignment"],  # type: ignore[arg-type]
                lifetime=value["lifetime"],  # type: ignore[arg-type]
                shape=tuple(extents),
                scope=value.get("scope", "rank_local"),  # type: ignore[arg-type]
            )
        )
    plan = ArtifactResourcePlan(slot=call_abi.slot, resources=tuple(resources))
    plan.validate_for(call_abi)
    return plan


def _parse_aggregate_node(raw: object, *, field: str) -> ArtifactAggregateNode:
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ArtifactABIError(f"{field} nested tuple may not be empty")
        return tuple(
            _parse_aggregate_node(value, field=f"{field}[{index}]")
            for index, value in enumerate(raw)
        )
    if not isinstance(raw, Mapping):
        raise ArtifactABIError(
            f"{field} must be a component table or nested component list"
        )
    if "static" in raw:
        if set(raw) != {"static"}:
            raise ArtifactABIError(
                f"{field} static component must contain exactly ['static']"
            )
        return ArtifactAggregateComponent(static=raw["static"])  # type: ignore[arg-type]

    required = {"source", "projection", "cast"}
    optional = {"axis"}
    missing = required - set(raw)
    unknown = set(raw) - required - optional
    if missing or unknown:
        raise ArtifactABIError(
            f"{field} component fields mismatch; missing={sorted(missing)!r}, "
            f"unknown={sorted(unknown)!r}"
        )
    return ArtifactAggregateComponent(
        source=raw["source"],  # type: ignore[arg-type]
        projection=raw["projection"],  # type: ignore[arg-type]
        axis=raw.get("axis"),  # type: ignore[arg-type]
        cast=raw["cast"],  # type: ignore[arg-type]
    )


def parse_artifact_bindings(raw: object, *, field: str) -> tuple[ArtifactBinding, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ArtifactABIError(f"{field} must be a list of binding tables")
    if not 1 <= len(raw) <= _MAX_BINDINGS:
        raise ArtifactABIError(f"{field} must contain 1..{_MAX_BINDINGS} rows")
    rows: list[ArtifactBinding] = []
    common = {"source", "kind"}
    options = {
        "tensor": {"projection", "unsqueeze", "assumed_align", "leading_dim"},
        "scalar": {"projection", "axis", "cast"},
        "pointer": {"projection", "assumed_align", "peer_resource"},
        "stream": {"projection"},
        "group": {"projection"},
        "opaque": {"projection"},
        "aggregate": {
            "projection",
            "cast",
            "component_cast",
            "components",
        },
    }
    for index, value in enumerate(raw):
        row_field = f"{field}[{index}]"
        if not isinstance(value, Mapping):
            raise ArtifactABIError(f"{row_field} must be a table")
        kind = value.get("kind")
        if not isinstance(kind, str) or kind not in options:
            raise ArtifactABIError(f"{row_field} has unsupported kind {kind!r}")
        if kind == "aggregate":
            required = {"kind", "cast", "component_cast", "components"}
            expected = required | options[kind]
        else:
            required = common
            expected = common | options[kind]
        unknown = set(value) - expected
        missing = required - set(value)
        if unknown or missing:
            raise ArtifactABIError(
                f"{row_field} fields mismatch; missing={sorted(missing)!r}, "
                f"unknown={sorted(unknown)!r}"
            )
        unsqueeze = value.get("unsqueeze", ())
        if not isinstance(unsqueeze, (list, tuple)):
            raise ArtifactABIError(f"{row_field}.unsqueeze must be an integer list")
        raw_components = value.get("components", ())
        if kind == "aggregate":
            if not isinstance(raw_components, (list, tuple)):
                raise ArtifactABIError(
                    f"{row_field}.components must be a component list"
                )
            components = tuple(
                _parse_aggregate_node(
                    component, field=f"{row_field}.components[{component_index}]"
                )
                for component_index, component in enumerate(raw_components)
            )
        else:
            components = ()
        rows.append(
            ArtifactBinding(
                source=value.get("source"),  # type: ignore[arg-type]
                kind=kind,
                cast=value.get("cast"),  # type: ignore[arg-type]
                unsqueeze=tuple(unsqueeze),
                assumed_align=value.get("assumed_align"),  # type: ignore[arg-type]
                leading_dim=value.get("leading_dim"),  # type: ignore[arg-type]
                projection=value.get("projection"),  # type: ignore[arg-type]
                axis=value.get("axis"),  # type: ignore[arg-type]
                component_cast=value.get("component_cast"),  # type: ignore[arg-type]
                components=components,
                peer_resource=value.get("peer_resource"),  # type: ignore[arg-type]
            )
        )
    return tuple(rows)


def parse_provider_capability_requirements(
    raw: object,
    *,
    field: str,
) -> tuple[ProviderCapabilityRequirement, ...]:
    """Reopen a canonical sealed list without consulting current allowlists.

    The requirement may name an older version that is no longer available; it
    must remain parseable so the caller can report an exact mismatch against the
    requirements derived from the current validator-owned call ABI.
    """

    if not isinstance(raw, (list, tuple)):
        raise ArtifactABIError(f"{field} must be a list of requirement tables")
    if len(raw) > _MAX_BINDINGS:
        raise ArtifactABIError(
            f"{field} may contain at most {_MAX_BINDINGS} requirements"
        )
    expected = {"capability", "projection", "source"}
    rows: list[ProviderCapabilityRequirement] = []
    for index, value in enumerate(raw):
        row_field = f"{field}[{index}]"
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ArtifactABIError(
                f"{row_field} must contain exactly {sorted(expected)!r}"
            )
        rows.append(
            ProviderCapabilityRequirement(
                source=value["source"],  # type: ignore[arg-type]
                projection=value["projection"],  # type: ignore[arg-type]
                capability=value["capability"],  # type: ignore[arg-type]
            )
        )
    canonical = tuple(sorted(rows))
    if len(set(rows)) != len(rows):
        raise ArtifactABIError(f"{field} repeats a provider capability requirement")
    if tuple(rows) != canonical:
        raise ArtifactABIError(f"{field} must be sorted canonically")
    return canonical


def parse_specialization_capability_requirements(
    raw: object,
    *,
    field: str,
) -> tuple[SpecializationCapabilityRequirement, ...]:
    """Reopen exact sealed specialization predicates without current policy."""

    if not isinstance(raw, (list, tuple)):
        raise ArtifactABIError(f"{field} must be a list of requirement tables")
    if len(raw) > _MAX_BINDINGS:
        raise ArtifactABIError(
            f"{field} may contain at most {_MAX_BINDINGS} requirements"
        )
    expected = {"capability_field", "source"}
    rows: list[SpecializationCapabilityRequirement] = []
    for index, value in enumerate(raw):
        row_field = f"{field}[{index}]"
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ArtifactABIError(
                f"{row_field} must contain exactly {sorted(expected)!r}"
            )
        rows.append(
            SpecializationCapabilityRequirement(
                source=value["source"],  # type: ignore[arg-type]
                capability_field=value["capability_field"],  # type: ignore[arg-type]
            )
        )
    canonical = tuple(sorted(rows))
    if len(set(rows)) != len(rows):
        raise ArtifactABIError(
            f"{field} repeats a specialization capability requirement"
        )
    if tuple(rows) != canonical:
        raise ArtifactABIError(f"{field} must be sorted canonically")
    return canonical


def parse_artifact_prelaunch(raw: object, *, field: str) -> tuple[ArtifactPrelaunch, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ArtifactABIError(f"{field} must be a list of operation tables")
    if len(raw) > _MAX_PRELAUNCH:
        raise ArtifactABIError(f"{field} contains too many operations")
    rows: list[ArtifactPrelaunch] = []
    expected = {"op", "target", "value"}
    for index, value in enumerate(raw):
        row_field = f"{field}[{index}]"
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ArtifactABIError(
                f"{row_field} must contain exactly {sorted(expected)!r}"
            )
        rows.append(
            ArtifactPrelaunch(
                op=value["op"],  # type: ignore[arg-type]
                target=value["target"],  # type: ignore[arg-type]
                value=value["value"],  # type: ignore[arg-type]
            )
        )
    return tuple(rows)


# Canonical semantic call ABIs. Both ``manifest`` and ``SlotSpec`` refer to these
# immutable objects, keeping intake torch-free without a parallel adapter registry.
# Resource rows describe stable slot semantics; a miner only selects bounded native
# projections of these values.
_STREAM_RESOURCE = SlotResource("stream.current", "stream")
_GROUP_RESOURCES = (
    SlotResource(
        "group.current",
        "group",
        provider_capabilities=(
            "group.native_handle.v1",
            "group.peer_ptr_table.v1",
        ),
    ),
    SlotResource(
        "group.rank",
        "scalar",
        scalar_type="i32",
        capability_field="rank",
        provider_resource="group.current",
        provider_projection="rank",
    ),
    SlotResource(
        "group.size",
        "scalar",
        scalar_type="i32",
        capability_field="world_size",
        provider_resource="group.current",
        provider_projection="size",
    ),
)


SILU_AND_MUL_CALL_ABI = SlotCallABI(
    slot="activation.silu_and_mul",
    resources=(
        SlotResource("input.x", "tensor"),
        SlotResource("output.out", "tensor", access="write"),
        _STREAM_RESOURCE,
    ),
    call_args=("input.x", "output.out"),
)

RMSNORM_CALL_ABI = SlotCallABI(
    slot="norm.rmsnorm",
    resources=(
        SlotResource("input.x", "tensor"),
        SlotResource("input.weight", "tensor"),
        SlotResource(
            "input.eps", "scalar", scalar_type="f64", capability_field="eps"
        ),
        SlotResource("output.out", "tensor", access="write"),
        _STREAM_RESOURCE,
    ),
    call_args=("input.x", "input.weight", "output.out", "input.eps"),
)

ATTENTION_SDPA_CALL_ABI = SlotCallABI(
    slot="attention.sdpa",
    resources=(
        SlotResource("input.q", "tensor"),
        SlotResource("input.k", "tensor"),
        SlotResource("input.v", "tensor"),
        SlotResource(
            "input.sm_scale",
            "scalar",
            scalar_type="f64",
            capability_field="sm_scale",
        ),
        SlotResource(
            "input.causal",
            "scalar",
            scalar_type="bool",
            capability_field="causal",
        ),
        SlotResource("output.out", "tensor", access="write"),
        _STREAM_RESOURCE,
    ),
    call_args=(
        "input.q",
        "input.k",
        "input.v",
        "output.out",
        "input.sm_scale",
        "input.causal",
    ),
)

ATTENTION_DECODE_CALL_ABI = SlotCallABI(
    slot="attention.decode",
    resources=(
        SlotResource("input.q", "tensor"),
        SlotResource("input.k", "tensor"),
        SlotResource("input.v", "tensor"),
        SlotResource("input.seq_lens", "tensor"),
        SlotResource(
            "input.sm_scale",
            "scalar",
            scalar_type="f64",
            capability_field="sm_scale",
        ),
        SlotResource("output.out", "tensor", access="write"),
        _STREAM_RESOURCE,
    ),
    call_args=(
        "input.q",
        "input.k",
        "input.v",
        "input.seq_lens",
        "input.sm_scale",
        "output.out",
    ),
)


def _moe_resources(*, collective: bool) -> tuple[SlotResource, ...]:
    common = (
        SlotResource("input.x", "tensor"),
        SlotResource("input.topk_ids", "tensor"),
        SlotResource("input.topk_weights", "tensor"),
        SlotResource("input.w13", "tensor"),
        SlotResource("input.w2", "tensor"),
        SlotResource("prepared.state", "opaque", access="readwrite"),
        SlotResource("output.out", "tensor", access="write"),
    )
    return common + (_GROUP_RESOURCES if collective else ()) + (_STREAM_RESOURCE,)


MOE_FUSED_EXPERTS_CALL_ABI = SlotCallABI(
    slot="moe.fused_experts",
    resources=_moe_resources(collective=False),
    call_args=(
        "input.x",
        "input.topk_ids",
        "input.topk_weights",
        "prepared.state",
        "output.out",
    ),
    prepare_args=("input.w13", "input.w2"),
    prepare_result="prepared.state",
)

COLLECTIVE_ALL_REDUCE_CALL_ABI = SlotCallABI(
    slot="collective.all_reduce",
    resources=(
        SlotResource("input.x", "tensor"),
        SlotResource("output.out", "tensor", access="write"),
        *_GROUP_RESOURCES,
        _STREAM_RESOURCE,
    ),
    call_args=("input.x", "output.out", "group.current"),
)

COLLECTIVE_AR_RESIDUAL_RMSNORM_CALL_ABI = SlotCallABI(
    slot="collective.ar_residual_rmsnorm",
    resources=(
        SlotResource("input.x", "tensor"),
        SlotResource("input.residual", "tensor"),
        SlotResource("input.weight", "tensor"),
        SlotResource(
            "input.eps", "scalar", scalar_type="f64", capability_field="eps"
        ),
        SlotResource("output.out_norm", "tensor", access="write"),
        SlotResource("output.out_residual", "tensor", access="write"),
        *_GROUP_RESOURCES,
        _STREAM_RESOURCE,
    ),
    call_args=(
        "input.x",
        "input.residual",
        "input.weight",
        "input.eps",
        "output.out_norm",
        "output.out_residual",
        "group.current",
    ),
)

COLLECTIVE_MOE_FINALIZE_AR_RMSNORM_CALL_ABI = SlotCallABI(
    slot="collective.moe_finalize_ar_rmsnorm",
    resources=(
        SlotResource("input.gemm_out", "tensor"),
        SlotResource("input.row_map", "tensor"),
        SlotResource("input.scales", "tensor"),
        SlotResource("input.residual", "tensor"),
        SlotResource("input.weight", "tensor"),
        SlotResource(
            "input.eps", "scalar", scalar_type="f64", capability_field="eps"
        ),
        SlotResource("output.out_norm", "tensor", access="write"),
        SlotResource("output.out_residual", "tensor", access="write"),
        *_GROUP_RESOURCES,
        _STREAM_RESOURCE,
    ),
    call_args=(
        "input.gemm_out",
        "input.row_map",
        "input.scales",
        "input.residual",
        "input.weight",
        "input.eps",
        "output.out_norm",
        "output.out_residual",
        "group.current",
    ),
)

MOE_FUSED_EXPERTS_REDUCE_CALL_ABI = SlotCallABI(
    slot="moe.fused_experts_reduce",
    resources=_moe_resources(collective=True),
    call_args=(
        "input.x",
        "input.topk_ids",
        "input.topk_weights",
        "prepared.state",
        "output.out",
        "group.current",
    ),
    prepare_args=("input.w13", "input.w2"),
    prepare_result="prepared.state",
)

MSA_BLOCK_SCORE_CALL_ABI = SlotCallABI(
    slot="attention.msa_block_score",
    resources=(
        SlotResource("input.q", "tensor"),
        SlotResource("input.index_k", "tensor"),
        SlotResource("input.seq_lens", "tensor"),
        SlotResource(
            "input.block_size",
            "scalar",
            scalar_type="i64",
            capability_field="block_size",
        ),
        SlotResource("output.block_scores", "tensor", access="write"),
        _STREAM_RESOURCE,
    ),
    call_args=(
        "input.q",
        "input.index_k",
        "input.seq_lens",
        "input.block_size",
        "output.block_scores",
    ),
)

MSA_PREFILL_BLOCK_SCORE_CALL_ABI = SlotCallABI(
    slot="attention.msa_prefill_block_score",
    resources=(
        SlotResource("input.q", "tensor"),
        SlotResource("input.index_k", "tensor"),
        SlotResource(
            "input.prefix_len",
            "scalar",
            scalar_type="i64",
        ),
        SlotResource(
            "input.scale",
            "scalar",
            scalar_type="f64",
        ),
        SlotResource(
            "input.block_size",
            "scalar",
            scalar_type="i64",
            capability_field="block_size",
        ),
        SlotResource("output.block_scores", "tensor", access="write"),
        SlotResource("stream.current", "stream"),
    ),
    call_args=(
        "input.q",
        "input.index_k",
        "input.prefix_len",
        "input.scale",
        "input.block_size",
        "output.block_scores",
    ),
)

SLOT_CALL_ABIS: Mapping[str, SlotCallABI] = MappingProxyType(
    {
        abi.slot: abi
        for abi in (
            SILU_AND_MUL_CALL_ABI,
            RMSNORM_CALL_ABI,
            ATTENTION_SDPA_CALL_ABI,
            ATTENTION_DECODE_CALL_ABI,
            MOE_FUSED_EXPERTS_CALL_ABI,
            COLLECTIVE_ALL_REDUCE_CALL_ABI,
            COLLECTIVE_AR_RESIDUAL_RMSNORM_CALL_ABI,
            COLLECTIVE_MOE_FINALIZE_AR_RMSNORM_CALL_ABI,
            MOE_FUSED_EXPERTS_REDUCE_CALL_ABI,
            MSA_BLOCK_SCORE_CALL_ABI,
            MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
        )
    }
)


def slot_call_abi(slot: object) -> SlotCallABI | None:
    return SLOT_CALL_ABIS.get(slot) if isinstance(slot, str) else None


__all__ = [
    "ArtifactABIError",
    "ArtifactAggregateComponent",
    "ArtifactAggregateNode",
    "ArtifactBinding",
    "ArtifactPrelaunch",
    "ArtifactResource",
    "ArtifactResourcePlan",
    "ArtifactShapeExtent",
    "ArtifactShapeFactor",
    "ATTENTION_DECODE_CALL_ABI",
    "ATTENTION_SDPA_CALL_ABI",
    "COLLECTIVE_ALL_REDUCE_CALL_ABI",
    "COLLECTIVE_AR_RESIDUAL_RMSNORM_CALL_ABI",
    "COLLECTIVE_MOE_FINALIZE_AR_RMSNORM_CALL_ABI",
    "MSA_PREFILL_BLOCK_SCORE_CALL_ABI",
    "MSA_BLOCK_SCORE_CALL_ABI",
    "MOE_FUSED_EXPERTS_CALL_ABI",
    "MOE_FUSED_EXPERTS_REDUCE_CALL_ABI",
    "RMSNORM_CALL_ABI",
    "SLOT_CALL_ABIS",
    "SILU_AND_MUL_CALL_ABI",
    "ProviderCapabilityRequirement",
    "SpecializationCapabilityRequirement",
    "SlotCallABI",
    "SlotResource",
    "checked_integer_cast",
    "integer_cast_bounds",
    "parse_artifact_bindings",
    "parse_artifact_prelaunch",
    "parse_artifact_resources",
    "parse_provider_capability_requirements",
    "parse_specialization_capability_requirements",
    "slot_call_abi",
]
