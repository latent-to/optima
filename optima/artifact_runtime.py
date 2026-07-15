"""Provider-neutral execution of declarative sealed-artifact launch plans.

Candidate manifests describe only projections of a validator-owned
``SlotCallABI``.  This module turns those data rows into a callable without a
per-slot or per-submission Python adapter.  Provider code is captured once during
trusted runtime setup; a later candidate import cannot redirect tensor, stream,
group, or executor authority.

The core deliberately knows nothing about torch, CUDA, or CuTe. Device-provider
adapters are validator-owned and shared by every admitted static slot.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Sequence

from optima.artifact_abi import (
    ArtifactABIError,
    ArtifactAggregateComponent,
    ArtifactAggregateNode,
    ArtifactBinding,
    ArtifactPrelaunch,
    ArtifactResource,
    ArtifactResourcePlan,
    ArtifactShapeFactor,
    SlotCallABI,
    SlotResource,
    checked_integer_cast,
)


class ArtifactRuntimeError(RuntimeError):
    """A sealed launch plan cannot be safely materialized or executed."""


@dataclass(frozen=True)
class ArtifactRuntimeProvider:
    """Trusted operations needed to project semantic resources for one provider."""

    provider: str
    tensor_descriptor: Callable[[object, ArtifactBinding], object]
    tensor_pointer: Callable[[object, ArtifactBinding], object]
    current_stream: Callable[[], object]
    group_rank: Callable[[object], int]
    group_size: Callable[[object], int]
    group_pointer: Callable[[object, str, ArtifactBinding, object | None], object]
    pointer_identity: Callable[[object, ArtifactBinding], object]
    # Versioned optional capabilities which this exact trusted provider can
    # materialize.  The declarative ABI seals requirements per artifact step;
    # construction rejects a missing capability before an engine can invoke the
    # artifact.  An empty inventory is the safe default for ordinary op kernels.
    provider_capabilities: frozenset[str] = frozenset()
    # Artifact-owned tensors are allocated only by this captured provider.  The
    # callback must return a tensor with the exact requested shape/dtype/device
    # and at least the requested alignment.  It is optional so forward-only
    # plans without artifact_resources retain the minimal provider surface.
    allocate_tensor: Callable[[tuple[int, ...], str, int], object] | None = None
    # Validate the exact allocation returned by ``allocate_tensor``.  Keeping
    # this provider-owned avoids teaching the generic binder about torch dtype
    # or device objects while still making the allocator contract enforceable.
    validate_allocation: (
        Callable[[object, tuple[int, ...], str, int], None] | None
    ) = None
    # Missing cached storage during CUDA capture is a hard error: allocating in
    # capture would make the graph depend on a lazy path and unstable pointers.
    is_capturing: Callable[[], bool] | None = None
    # Stable execution-domain key (normally CUDA device + stream).  Call/state
    # storage is never shared across distinct domains.
    execution_scope: Callable[[], object] | None = None
    # Lifecycle teardown and explicit prepare need a trusted completion fence.
    # The CuTe provider uses a captured CUDA-device synchronize callable.
    synchronize: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider:
            raise ArtifactRuntimeError("artifact runtime provider ID is empty")
        for field_name in (
            "tensor_descriptor",
            "tensor_pointer",
            "current_stream",
            "group_rank",
            "group_size",
            "group_pointer",
            "pointer_identity",
        ):
            if not callable(getattr(self, field_name)):
                raise ArtifactRuntimeError(
                    f"artifact runtime provider {field_name} is not callable"
                )
        if type(self.provider_capabilities) is not frozenset or any(
            not isinstance(capability, str)
            or not capability
            or len(capability) > 128
            or not capability.isascii()
            for capability in self.provider_capabilities
        ):
            raise ArtifactRuntimeError(
                "artifact runtime provider capabilities are malformed"
            )
        optional = (
            "allocate_tensor",
            "validate_allocation",
            "is_capturing",
            "execution_scope",
            "synchronize",
        )
        for field_name in optional:
            value = getattr(self, field_name)
            if value is not None and not callable(value):
                raise ArtifactRuntimeError(
                    f"artifact runtime provider {field_name} is not callable"
                )
        present = {name for name in optional if getattr(self, name) is not None}
        if present and present != set(optional):
            raise ArtifactRuntimeError(
                "artifact storage providers must supply allocator, allocation "
                "validation, capture, execution-scope, and synchronization authority"
            )


@dataclass(frozen=True)
class ArtifactRuntimeLimits:
    """Validator-owned live allocation limits for one artifact entry.

    Manifest ceilings protect intake; these smaller live limits protect an arena.
    They are supplied by trusted launch policy and must be included in the arena
    identity/receipt by the caller that constructs the runtime entry.
    """

    max_live_bytes: int
    max_allocation_keys: int = 4_096

    def __post_init__(self) -> None:
        if (
            type(self.max_live_bytes) is not int
            or not 1 <= self.max_live_bytes <= 128 << 30
        ):
            raise ArtifactRuntimeError(
                "artifact runtime max_live_bytes must be in [1, 128 GiB]"
            )
        if (
            type(self.max_allocation_keys) is not int
            or not 1 <= self.max_allocation_keys <= 16_384
        ):
            raise ArtifactRuntimeError(
                "artifact runtime max_allocation_keys must be in [1, 16384]"
            )


class ArtifactAllocationBudget:
    """Shared process-wide reservation authority for artifact-owned storage."""

    def __init__(self, limits: ArtifactRuntimeLimits) -> None:
        if not isinstance(limits, ArtifactRuntimeLimits):
            raise ArtifactRuntimeError("artifact allocation budget requires limits")
        self.limits = limits
        self._lock = threading.RLock()
        self._reservations: dict[tuple[object, object], tuple[int, int]] = {}
        self._bytes = 0
        self._keys = 0

    @property
    def allocated_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def allocated_keys(self) -> int:
        with self._lock:
            return self._keys

    def reserve(
        self,
        owner: object,
        key: object,
        byte_count: int,
        allocation_count: int,
    ) -> None:
        if type(byte_count) is not int or byte_count < 0:
            raise ArtifactRuntimeError("artifact allocation reservation is malformed")
        if type(allocation_count) is not int or allocation_count < 1:
            raise ArtifactRuntimeError(
                "artifact allocation-count reservation is malformed"
            )
        identity = (owner, key)
        with self._lock:
            if identity in self._reservations:
                raise ArtifactRuntimeError("artifact allocation reservation repeats")
            if self._keys + allocation_count > self.limits.max_allocation_keys:
                raise ArtifactRuntimeError(
                    "artifact runtime exceeded the process allocation-key budget"
                )
            if self._bytes + byte_count > self.limits.max_live_bytes:
                raise ArtifactRuntimeError(
                    "artifact runtime exceeded the process live-byte budget"
                )
            self._reservations[identity] = (byte_count, allocation_count)
            self._bytes += byte_count
            self._keys += allocation_count

    def cancel(self, owner: object, key: object) -> None:
        identity = (owner, key)
        with self._lock:
            reservation = self._reservations.pop(identity, None)
            if reservation is not None:
                byte_count, allocation_count = reservation
                self._bytes -= byte_count
                self._keys -= allocation_count

    def release_owner(self, owner: object) -> None:
        with self._lock:
            identities = [key for key in self._reservations if key[0] is owner]
            for identity in identities:
                byte_count, allocation_count = self._reservations.pop(identity)
                self._bytes -= byte_count
                self._keys -= allocation_count


@dataclass(frozen=True)
class ArtifactPreparedState:
    """Opaque validator-produced result of a declarative prepare pipeline."""

    _owner: object
    _key: object
    _token: object


@dataclass(frozen=True)
class ArtifactRuntimeStep:
    """One reopened executor plus its sealed declarative launch row."""

    name: str
    plan: str
    step: int
    role: str
    bindings: tuple[ArtifactBinding, ...]
    specializes: tuple[tuple[str, bool | int | float | str], ...]
    prelaunch: tuple[ArtifactPrelaunch, ...]
    executor: object

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ArtifactRuntimeError("artifact runtime step name is empty")
        if not isinstance(self.plan, str) or not self.plan:
            raise ArtifactRuntimeError("artifact runtime plan name is empty")
        if type(self.step) is not int or not 0 <= self.step <= 255:
            raise ArtifactRuntimeError("artifact runtime step is outside [0, 255]")
        if self.role not in {"init", "prepare", "reset", "run", "destroy"}:
            raise ArtifactRuntimeError(
                f"artifact runtime step has unsupported role {self.role!r}"
            )
        if type(self.bindings) is not tuple or not all(
            isinstance(binding, ArtifactBinding) for binding in self.bindings
        ):
            raise ArtifactRuntimeError("artifact runtime bindings are malformed")
        if type(self.prelaunch) is not tuple or not all(
            isinstance(operation, ArtifactPrelaunch)
            for operation in self.prelaunch
        ):
            raise ArtifactRuntimeError("artifact runtime prelaunch rows are malformed")
        if not callable(self.executor):
            raise ArtifactRuntimeError("artifact runtime executor is not callable")


@dataclass(frozen=True)
class _ArtifactPlan:
    name: str
    specializes: tuple[tuple[str, bool | int | float | str], ...]
    steps: tuple[ArtifactRuntimeStep, ...]


_CALL_RESOURCES: ContextVar[Mapping[str, object]] = ContextVar(
    "optima_artifact_call_resources", default=MappingProxyType({})
)


@contextmanager
def artifact_call_resources(resources: Mapping[str, object]) -> Iterator[None]:
    """Supply validator-owned workspace/state resources to one artifact call.

    Ordinary slot inputs and outputs remain positional according to ``call_args``.
    This scoped channel is only for resources the validator allocated outside the
    historical Python entry signature (for example ``workspace.scratch``).  A
    manifest cannot populate it and cannot override a positional resource.
    """

    if not isinstance(resources, Mapping):
        raise ArtifactRuntimeError("artifact call resources must be a mapping")
    copied: dict[str, object] = {}
    for name, value in resources.items():
        if not isinstance(name, str) or not name:
            raise ArtifactRuntimeError("artifact call resource name is malformed")
        if name in copied:
            raise ArtifactRuntimeError(
                f"artifact call resource repeats {name!r}"
            )
        copied[name] = value
    token = _CALL_RESOURCES.set(MappingProxyType(copied))
    try:
        yield
    finally:
        _CALL_RESOURCES.reset(token)


def _fill_value(value: bool | int | float | str) -> bool | int | float:
    if value == "-inf":
        return float("-inf")
    if value == "+inf":
        return float("inf")
    if value == "nan":
        return float("nan")
    if isinstance(value, (bool, int, float)):
        return value
    raise ArtifactRuntimeError(f"unsupported artifact fill value {value!r}")


def _scalar_cast(value: object, cast: str, *, field: str) -> object:
    if cast == "bool":
        if type(value) is not bool:
            raise ArtifactRuntimeError(f"{field} must resolve to bool")
        return value
    if cast in {"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"}:
        try:
            return checked_integer_cast(value, cast, field=field)
        except ValueError as exc:
            raise ArtifactRuntimeError(str(exc)) from None
    if cast in {"f16", "bf16", "f32", "tf32", "f64"}:
        if type(value) not in {int, float}:
            raise ArtifactRuntimeError(f"{field} must resolve to a real scalar")
        result = float(value)
        if not math.isfinite(result):
            raise ArtifactRuntimeError(f"{field} must resolve to a finite scalar")
        return result
    raise ArtifactRuntimeError(f"{field} has unsupported scalar cast {cast!r}")


def _tensor_axis(value: object, axis: int, *, projection: str, field: str) -> int:
    try:
        sequence = value.shape if projection == "shape" else value.stride()
        rank = len(value.shape)
    except (AttributeError, TypeError) as exc:
        raise ArtifactRuntimeError(f"{field} is not a tensor descriptor: {exc}") from None
    if not 0 <= axis < rank:
        raise ArtifactRuntimeError(
            f"{field} axis {axis} is outside live rank {rank}"
        )
    try:
        return int(sequence[axis])
    except (IndexError, TypeError, ValueError, OverflowError) as exc:
        raise ArtifactRuntimeError(
            f"{field} cannot resolve tensor {projection}: {exc}"
        ) from None


def _tensor_metadata(value: object, projection: str, axis: int | None, *, field: str) -> int:
    if projection in {"shape", "stride"}:
        if axis is None:
            raise ArtifactRuntimeError(f"{field} lacks an axis")
        return _tensor_axis(value, axis, projection=projection, field=field)
    try:
        if projection == "numel":
            return int(value.numel())
        if projection == "rank":
            return int(len(value.shape))
        if projection == "storage_offset":
            return int(value.storage_offset())
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ArtifactRuntimeError(
            f"{field} cannot resolve tensor {projection}: {exc}"
        ) from None
    raise ArtifactRuntimeError(
        f"{field} has unsupported tensor metadata projection {projection!r}"
    )


def _resource_value(
    name: str,
    resources: Mapping[str, object],
    *,
    field: str,
) -> object:
    try:
        return resources[name]
    except KeyError:
        raise ArtifactRuntimeError(
            f"{field} requires unavailable validator resource {name!r}"
        ) from None


def _component_value(
    component: ArtifactAggregateComponent,
    resources: Mapping[str, object],
    resource_table: Mapping[str, SlotResource],
    provider: ArtifactRuntimeProvider,
    *,
    field: str,
) -> int:
    if not component.dynamic:
        return int(component.static)  # type: ignore[arg-type]
    source_name = component.source
    assert source_name is not None
    resource = resource_table[source_name]
    value = _resource_value(source_name, resources, field=field)
    if component.projection == "value":
        projected = value
    elif resource.kind == "tensor":
        projected = _tensor_metadata(
            value,
            component.projection,  # type: ignore[arg-type]
            component.axis,
            field=field,
        )
    elif resource.kind == "group" and component.projection == "rank":
        projected = provider.group_rank(value)
    elif resource.kind == "group" and component.projection == "size":
        projected = provider.group_size(value)
    else:
        raise ArtifactRuntimeError(
            f"{field} cannot project {component.projection!r} from {resource.kind!r}"
        )
    return int(_scalar_cast(projected, component.cast, field=field))  # type: ignore[arg-type]


def _aggregate_value(
    nodes: Sequence[ArtifactAggregateNode],
    resources: Mapping[str, object],
    resource_table: Mapping[str, SlotResource],
    provider: ArtifactRuntimeProvider,
    *,
    field: str,
) -> tuple[object, ...]:
    values: list[object] = []
    for index, node in enumerate(nodes):
        child_field = f"{field}[{index}]"
        if isinstance(node, ArtifactAggregateComponent):
            values.append(
                _component_value(
                    node,
                    resources,
                    resource_table,
                    provider,
                    field=child_field,
                )
            )
        else:
            values.append(
                _aggregate_value(
                    node,
                    resources,
                    resource_table,
                    provider,
                    field=child_field,
                )
            )
    return tuple(values)


def _project_binding(
    binding: ArtifactBinding,
    resources: Mapping[str, object],
    resource_table: Mapping[str, SlotResource],
    provider: ArtifactRuntimeProvider,
    *,
    field: str,
) -> object:
    if binding.kind == "aggregate":
        return _aggregate_value(
            binding.components, resources, resource_table, provider, field=field
        )

    source_name = binding.source
    assert source_name is not None
    resource = resource_table[source_name]
    if resource.kind == "stream":
        if binding.kind != "stream":
            raise ArtifactRuntimeError(f"{field} has a non-stream stream projection")
        return provider.current_stream()
    value = _resource_value(source_name, resources, field=field)

    if binding.kind == "tensor":
        return provider.tensor_descriptor(value, binding)
    if binding.kind == "scalar":
        projection = binding.projection
        if resource.kind == "scalar" and projection == "value":
            projected = value
        elif resource.kind == "tensor":
            projected = _tensor_metadata(
                value, projection, binding.axis, field=field  # type: ignore[arg-type]
            )
        elif resource.kind == "group" and projection == "rank":
            projected = provider.group_rank(value)
        elif resource.kind == "group" and projection == "size":
            projected = provider.group_size(value)
        else:
            raise ArtifactRuntimeError(
                f"{field} cannot project scalar {projection!r} from {resource.kind!r}"
            )
        return _scalar_cast(projected, binding.cast, field=field)  # type: ignore[arg-type]
    if binding.kind == "pointer":
        if resource.kind == "tensor" and binding.projection == "device_ptr":
            return provider.tensor_pointer(value, binding)
        if resource.kind == "group" and binding.projection in {
            "native_handle",
            "peer_ptr_table",
        }:
            peer_value = (
                _resource_value(
                    binding.peer_resource,
                    resources,
                    field=f"{field} peer resource",
                )
                if binding.peer_resource is not None
                else None
            )
            return provider.group_pointer(
                value, binding.projection, binding, peer_value
            )
        if binding.projection == "identity":
            return provider.pointer_identity(value, binding)
        raise ArtifactRuntimeError(
            f"{field} has unsupported pointer projection {binding.projection!r}"
        )
    if binding.kind in {"group", "opaque"}:
        return value
    raise ArtifactRuntimeError(
        f"{field} has unsupported artifact binding kind {binding.kind!r}"
    )


def _resource_identity(value: object) -> object:
    """Bounded identity for once-per-resource lifecycle preparation."""

    try:
        pointer = int(value.data_ptr())
        shape = tuple(int(dim) for dim in value.shape)
        stride = tuple(int(dim) for dim in value.stride())
        return (
            "tensor",
            pointer,
            shape,
            stride,
            str(value.dtype),
            str(value.device),
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass
    if type(value) in {bool, int, float, str, type(None)}:
        return (type(value).__name__, value)
    return ("object", id(value))


def _shape_factor_value(
    factor: ArtifactShapeFactor,
    resources: Mapping[str, object],
    resource_table: Mapping[str, SlotResource],
    provider: ArtifactRuntimeProvider,
    *,
    field: str,
) -> int:
    if factor.static is not None:
        return factor.static
    source_name = factor.source
    assert source_name is not None and factor.upper_bound is not None
    descriptor = resource_table[source_name]
    value = _resource_value(source_name, resources, field=field)
    if factor.projection == "value":
        projected = value
    elif descriptor.kind == "tensor":
        projected = _tensor_metadata(
            value,
            factor.projection,  # type: ignore[arg-type]
            factor.axis,
            field=field,
        )
    elif descriptor.kind == "group" and factor.projection == "size":
        projected = provider.group_size(value)
    else:  # Admission validated this relation; fail closed on runtime drift.
        raise ArtifactRuntimeError(
            f"{field} cannot project {factor.projection!r} from "
            f"{descriptor.kind!r}"
        )
    if type(projected) is not int:
        try:
            projected = int(projected)
        except (TypeError, ValueError, OverflowError):
            raise ArtifactRuntimeError(f"{field} did not resolve to an integer") from None
    if type(projected) is bool or not 0 <= projected <= factor.upper_bound:
        raise ArtifactRuntimeError(
            f"{field} value {projected!r} is outside live bound "
            f"[0, {factor.upper_bound}]"
        )
    return projected


def _artifact_resource_shape(
    declaration: ArtifactResource,
    resources: Mapping[str, object],
    resource_table: Mapping[str, SlotResource],
    provider: ArtifactRuntimeProvider,
) -> tuple[int, ...]:
    shape: list[int] = []
    for extent_index, extent in enumerate(declaration.shape):
        product = 1
        for factor_index, factor in enumerate(extent.factors):
            product *= _shape_factor_value(
                factor,
                resources,
                resource_table,
                provider,
                field=(
                    f"artifact resource {declaration.name!r} "
                    f"shape[{extent_index}].factor[{factor_index}]"
                ),
            )
            if product > (1 << 63) - 1:
                raise ArtifactRuntimeError(
                    f"artifact resource {declaration.name!r} live shape overflows i64"
                )
        value = (product + extent.divisor - 1) // extent.divisor
        if not 0 <= value <= extent.maximum:
            raise ArtifactRuntimeError(
                f"artifact resource {declaration.name!r} live extent is outside "
                "its sealed maximum"
            )
        shape.append(value)
    result = tuple(shape)
    elements = math.prod(result)
    if elements > declaration.max_elements:
        raise ArtifactRuntimeError(
            f"artifact resource {declaration.name!r} live element count exceeds "
            "its sealed maximum"
        )
    return result


def _artifact_resource_bytes(
    declaration: ArtifactResource, shape: tuple[int, ...]
) -> int:
    # The declaration already validated a nonzero maximum shape and dtype.  Derive
    # element width from its own sealed maxima so runtime cannot drift from the ABI
    # module's dtype table.
    element_bytes = declaration.max_bytes // declaration.max_elements
    return math.prod(shape) * element_bytes


def _artifact_resource_reservation_bytes(
    declaration: ArtifactResource, shape: tuple[int, ...]
) -> int:
    """Upper-bound the backing allocation retained by an aligned tensor view.

    The CuTe provider over-allocates by ``alignment - 1`` bytes and retains that
    backing storage through the returned view.  Charging only logical payload
    bytes makes zero/small buffers with large alignment a budget bypass.
    Providers admitted to this runtime must fit this same upper bound.
    """

    return _artifact_resource_bytes(declaration, shape) + declaration.alignment - 1


def _step_resource_names(
    steps: Sequence[ArtifactRuntimeStep], role: str
) -> tuple[str, ...]:
    names = {
        binding.source
        for step in steps
        if step.role == role
        for binding in step.bindings
        if binding.source is not None
    }
    names.update(
        leaf.source
        for step in steps
        if step.role == role
        for binding in step.bindings
        if binding.kind == "aggregate"
        for leaf in binding.aggregate_leaves
        if leaf.source is not None
    )
    names.update(
        binding.peer_resource
        for step in steps
        if step.role == role
        for binding in step.bindings
        if binding.peer_resource is not None
    )
    names.update(
        operation.target
        for step in steps
        if step.role == role
        for operation in step.prelaunch
    )
    return tuple(sorted(names))


class ArtifactRuntimeEntry:
    """Validator-built caller for one sealed declarative artifact contract.

    Artifact-owned storage is allocated only through the captured provider and is
    cached at stable addresses.  A first allocation/init/prepare while CUDA capture
    is active fails closed; graph warmup must establish every required frame first.
    """

    def __init__(
        self,
        *,
        call_abi: SlotCallABI,
        steps: Sequence[ArtifactRuntimeStep],
        provider: ArtifactRuntimeProvider,
        artifact_resources: ArtifactResourcePlan | None = None,
        limits: ArtifactRuntimeLimits | None = None,
        allocation_budget: ArtifactAllocationBudget | None = None,
    ) -> None:
        if not isinstance(call_abi, SlotCallABI):
            raise ArtifactRuntimeError("artifact runtime requires a SlotCallABI")
        if not isinstance(provider, ArtifactRuntimeProvider):
            raise ArtifactRuntimeError("artifact runtime provider has the wrong type")
        if artifact_resources is not None and not isinstance(
            artifact_resources, ArtifactResourcePlan
        ):
            raise ArtifactRuntimeError("artifact resources have the wrong plan type")
        if artifact_resources is not None:
            try:
                artifact_resources.validate_for(call_abi)
            except ArtifactABIError as exc:
                raise ArtifactRuntimeError(str(exc)) from None
            if not artifact_resources.resources:
                artifact_resources = None
        if artifact_resources is not None:
            if not isinstance(limits, ArtifactRuntimeLimits):
                raise ArtifactRuntimeError(
                    "artifact resources require validator-owned runtime limits"
                )
            if not isinstance(allocation_budget, ArtifactAllocationBudget):
                raise ArtifactRuntimeError(
                    "artifact resources require a shared allocation budget"
                )
            if allocation_budget.limits != limits:
                raise ArtifactRuntimeError(
                    "artifact allocation budget and entry limits disagree"
                )
            if provider.allocate_tensor is None or provider.is_capturing is None:
                raise ArtifactRuntimeError(
                    "artifact resources require allocator and capture authority"
                )
        elif limits is not None and not isinstance(limits, ArtifactRuntimeLimits):
            raise ArtifactRuntimeError("artifact runtime limits have the wrong type")
        elif allocation_budget is not None and not isinstance(
            allocation_budget, ArtifactAllocationBudget
        ):
            raise ArtifactRuntimeError("artifact allocation budget has the wrong type")

        self._call_abi = call_abi
        self._provider = provider
        self._artifact_resources = artifact_resources
        self._limits = limits
        self._allocation_budget = allocation_budget
        try:
            self._resource_table = call_abi.resource_table(artifact_resources)
        except ArtifactABIError as exc:
            raise ArtifactRuntimeError(str(exc)) from None
        self._plans = self._build_plans(steps)
        if (
            any(
                step.role in {"init", "prepare", "destroy"}
                for plan in self._plans
                for step in plan.steps
            )
            and provider.is_capturing is None
        ):
            raise ArtifactRuntimeError(
                "artifact lifecycle steps require capture-state authority"
            )
        self._validate_plans()
        self._owner = object()
        self._lock = threading.RLock()
        self._initialized: set[object] = set()
        self._prepared_by_key: dict[object, ArtifactPreparedState] = {}
        self._prepared_frames: dict[
            object, Mapping[str, Mapping[str, object]]
        ] = {}
        self._implicit_cache: dict[object, Mapping[str, object]] = {}
        self._resource_frames: dict[object, Mapping[str, object]] = {}
        self._destroy_frames: dict[
            object, tuple[ArtifactRuntimeStep, Mapping[str, object]]
        ] = {}
        self._allocated_bytes = 0
        self._allocated_keys = 0
        self._key_limit = limits.max_allocation_keys if limits is not None else 4_096
        self._closed = False
        self._closing = False
        grouped: dict[str, list[ArtifactResource]] = {
            "workspace": [],
            "prepared": [],
            "state": [],
        }
        for declaration in (
            artifact_resources.resources if artifact_resources is not None else ()
        ):
            grouped[declaration.name.split(".", 1)[0]].append(declaration)
        self._declarations = MappingProxyType(
            {name: tuple(rows) for name, rows in grouped.items()}
        )

    @staticmethod
    def _build_plans(steps: Sequence[ArtifactRuntimeStep]) -> tuple[_ArtifactPlan, ...]:
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
            raise ArtifactRuntimeError("artifact runtime requires at least one step")
        grouped: dict[str, list[ArtifactRuntimeStep]] = {}
        for row in steps:
            if not isinstance(row, ArtifactRuntimeStep):
                raise ArtifactRuntimeError("artifact runtime step has the wrong type")
            grouped.setdefault(row.plan, []).append(row)
        plans: list[_ArtifactPlan] = []
        role_order = {"init": 0, "prepare": 1, "reset": 2, "run": 3, "destroy": 4}
        for plan_name, rows in sorted(grouped.items()):
            rows.sort(key=lambda row: row.step)
            if len({row.step for row in rows}) != len(rows):
                raise ArtifactRuntimeError(
                    f"artifact plan {plan_name!r} repeats a step number"
                )
            predicates = {row.specializes for row in rows}
            if len(predicates) != 1:
                raise ArtifactRuntimeError(
                    f"artifact plan {plan_name!r} has inconsistent specializations"
                )
            if not any(row.role == "run" for row in rows):
                raise ArtifactRuntimeError(
                    f"artifact plan {plan_name!r} contains no run step"
                )
            ranks = [role_order[row.role] for row in rows]
            if ranks != sorted(ranks):
                raise ArtifactRuntimeError(
                    f"artifact plan {plan_name!r} lifecycle roles are out of order"
                )
            plans.append(
                _ArtifactPlan(plan_name, rows[0].specializes, tuple(rows))
            )
        return tuple(plans)

    def _validate_plans(self) -> None:
        try:
            required_capabilities: set[str] = set()
            for plan in self._plans:
                for step in plan.steps:
                    normalized = self._call_abi.validate_plan(
                        role=step.role,
                        bindings=step.bindings,
                        specializes=dict(step.specializes),
                        prelaunch=step.prelaunch,
                        require_outputs=False,
                        artifact_resources=self._artifact_resources,
                    )
                    if normalized != step.specializes:
                        raise ArtifactRuntimeError(
                            f"artifact step {step.name!r} specializations are not canonical"
                        )
                    required_capabilities.update(
                        requirement.capability
                        for requirement in self._call_abi.provider_capability_requirements(
                            step.bindings,
                            artifact_resources=self._artifact_resources,
                        )
                    )
                rows = tuple(
                    (step.role, step.bindings, step.prelaunch)
                    for step in plan.steps
                )
                self._call_abi.validate_pipeline(
                    rows, artifact_resources=self._artifact_resources
                )
                if self._artifact_resources is not None:
                    self._artifact_resources.validate_pipeline(rows, require_all=True)
            missing_capabilities = (
                required_capabilities - self._provider.provider_capabilities
            )
            if missing_capabilities:
                raise ArtifactRuntimeError(
                    "artifact runtime provider lacks sealed capabilities "
                    f"{sorted(missing_capabilities)!r}"
                )
        except ArtifactABIError as exc:
            raise ArtifactRuntimeError(
                f"artifact runtime launch plan failed sealed ABI validation: {exc}"
            ) from None

    @property
    def plans(self) -> tuple[str, ...]:
        return tuple(plan.name for plan in self._plans)

    @property
    def allocated_bytes(self) -> int:
        return self._allocated_bytes

    @property
    def allocated_keys(self) -> int:
        return self._allocated_keys

    def _base_frame(
        self, names: tuple[str, ...], args: tuple[object, ...]
    ) -> Mapping[str, object]:
        if len(args) != len(names):
            raise ArtifactRuntimeError(
                f"slot {self._call_abi.slot!r} requires {len(names)} positional "
                f"resources, got {len(args)}"
            )
        frame = dict(zip(names, args))
        for name, value in _CALL_RESOURCES.get().items():
            if name not in self._call_abi.by_name:
                raise ArtifactRuntimeError(
                    f"scoped resource {name!r} is not declared by slot "
                    f"{self._call_abi.slot!r}"
                )
            if name in frame:
                raise ArtifactRuntimeError(
                    f"scoped resource may not override positional resource {name!r}"
                )
            frame[name] = value
        group_facts: dict[str, dict[str, int]] = {}
        for name, descriptor in self._call_abi.by_name.items():
            provider_name = descriptor.provider_resource
            if provider_name is None or name in frame or provider_name not in frame:
                continue
            provider_value = frame[provider_name]
            if descriptor.provider_projection == "rank":
                rank = self._provider.group_rank(provider_value)
                if type(rank) is not int or rank < 0:
                    raise ArtifactRuntimeError(
                        f"slot resource {name!r} resolved an invalid group rank"
                    )
                frame[name] = rank
                group_facts.setdefault(provider_name, {})["rank"] = rank
            elif descriptor.provider_projection == "size":
                size = self._provider.group_size(provider_value)
                if type(size) is not int or size <= 0:
                    raise ArtifactRuntimeError(
                        f"slot resource {name!r} resolved an invalid group size"
                    )
                frame[name] = size
                group_facts.setdefault(provider_name, {})["size"] = size
            else:  # SlotResource admission makes this unreachable; fail closed.
                raise ArtifactRuntimeError(
                    f"slot resource {name!r} has unsupported provider projection"
                )
        for provider_name, facts in group_facts.items():
            if "rank" in facts and "size" in facts and facts["rank"] >= facts["size"]:
                raise ArtifactRuntimeError(
                    f"slot group provider {provider_name!r} resolved rank outside world"
                )
        return MappingProxyType(frame)

    def _select(self, resources: Mapping[str, object]) -> _ArtifactPlan:
        matches: list[_ArtifactPlan] = []
        for plan in self._plans:
            if all(
                source in resources
                and type(resources[source]) is type(expected)
                and resources[source] == expected
                for source, expected in plan.specializes
            ):
                matches.append(plan)
        if not matches:
            raise ArtifactRuntimeError(
                f"slot {self._call_abi.slot!r} has no compatible artifact plan"
            )
        specificity = max(len(plan.specializes) for plan in matches)
        winners = [plan for plan in matches if len(plan.specializes) == specificity]
        if len(winners) != 1:
            raise ArtifactRuntimeError(
                f"slot {self._call_abi.slot!r} has ambiguous artifact plans "
                f"{[plan.name for plan in winners]!r}"
            )
        return winners[0]

    def _capturing(self) -> bool:
        query = self._provider.is_capturing
        if query is None:
            return False
        try:
            result = query()
        except Exception as exc:  # noqa: BLE001 - runtime providers vary
            raise ArtifactRuntimeError(
                f"cannot query artifact capture state: {exc}"
            ) from None
        if type(result) is not bool:
            raise ArtifactRuntimeError("artifact capture query did not return bool")
        return result

    def _execution_scope(self) -> object:
        query = self._provider.execution_scope
        if query is None:
            return ("provider-default",)
        try:
            result = query()
            hash(result)
        except Exception as exc:  # noqa: BLE001 - provider surfaces vary
            raise ArtifactRuntimeError(
                f"cannot resolve artifact execution scope: {exc}"
            ) from None
        if isinstance(result, (str, bytes)) and len(result) > 256:
            raise ArtifactRuntimeError("artifact execution scope is unbounded")
        if isinstance(result, tuple) and len(result) > 16:
            raise ArtifactRuntimeError("artifact execution scope is unbounded")
        return result

    def _synchronize(self, action: str) -> None:
        synchronize = self._provider.synchronize
        if synchronize is None:
            raise ArtifactRuntimeError(
                f"artifact {action} lacks provider synchronization authority"
            )
        try:
            synchronize()
        except Exception as exc:  # noqa: BLE001 - provider surfaces vary
            raise ArtifactRuntimeError(
                f"artifact {action} synchronization failed: {exc}"
            ) from None

    def _group_scope(self, resources: Mapping[str, object]) -> object:
        """Bind state/workspace frames to every live validator group.

        Two independent process groups may have identical world sizes and use
        the same CUDA stream.  Sharing mutable collective state between them is
        still invalid, so group object identity is part of allocation authority.
        """

        return tuple(
            (
                name,
                _resource_identity(resources[name]),
            )
            for name, descriptor in sorted(self._call_abi.by_name.items())
            if descriptor.kind == "group" and name in resources
        )

    def _require_pre_capture(self, action: str) -> None:
        if self._capturing():
            raise ArtifactRuntimeError(
                f"artifact {action} was not prepared before CUDA graph capture"
            )

    def _resource_frame(
        self,
        plan: _ArtifactPlan,
        prefix: str,
        resources: Mapping[str, object],
        *,
        instance_key: object,
    ) -> Mapping[str, object]:
        declarations = self._declarations[prefix]
        if not declarations:
            return MappingProxyType({})
        shape_rows = tuple(
            (
                declaration.name,
                _artifact_resource_shape(
                    declaration,
                    resources,
                    self._call_abi.by_name,
                    self._provider,
                ),
            )
            for declaration in declarations
        )
        cache_key = (plan.name, prefix, instance_key, shape_rows)
        cached = self._resource_frames.get(cache_key)
        if cached is not None:
            return cached
        self._require_pre_capture(f"{prefix} allocation")
        limits = self._limits
        allocator = self._provider.allocate_tensor
        assert limits is not None and allocator is not None
        allocation_count = len(declarations)
        if self._allocated_keys + allocation_count > limits.max_allocation_keys:
            raise ArtifactRuntimeError(
                "artifact runtime exceeded its live allocation-key limit"
            )
        additional = sum(
            _artifact_resource_reservation_bytes(declaration, shape)
            for declaration, (_name, shape) in zip(declarations, shape_rows)
        )
        if self._allocated_bytes + additional > limits.max_live_bytes:
            raise ArtifactRuntimeError(
                "artifact runtime allocation exceeds the arena live-byte budget"
            )
        budget = self._allocation_budget
        assert budget is not None
        budget.reserve(self._owner, cache_key, additional, allocation_count)
        allocated: dict[str, object] = {}
        try:
            validate = self._provider.validate_allocation
            assert validate is not None
            for declaration, (_name, shape) in zip(declarations, shape_rows):
                tensor = allocator(shape, declaration.dtype, declaration.alignment)
                observed_shape = tuple(int(value) for value in tensor.shape)
                pointer = int(tensor.data_ptr())
                if observed_shape != shape:
                    raise ArtifactRuntimeError(
                        f"artifact allocator returned wrong shape for "
                        f"{declaration.name!r}"
                    )
                if pointer % declaration.alignment:
                    raise ArtifactRuntimeError(
                        f"artifact allocator violated {declaration.name!r} alignment"
                    )
                validate(tensor, shape, declaration.dtype, declaration.alignment)
                allocated[declaration.name] = tensor
        except Exception as exc:  # noqa: BLE001 - provider/runtime surfaces vary
            budget.cancel(self._owner, cache_key)
            if isinstance(exc, ArtifactRuntimeError):
                raise
            raise ArtifactRuntimeError(
                f"cannot allocate artifact {prefix} resources: {exc}"
            ) from None
        frame = MappingProxyType(allocated)
        self._resource_frames[cache_key] = frame
        self._allocated_bytes += additional
        self._allocated_keys += allocation_count
        return frame

    @staticmethod
    def _can_shape(
        declarations: Sequence[ArtifactResource], resources: Mapping[str, object]
    ) -> bool:
        return all(
            factor.source is None or factor.source in resources
            for declaration in declarations
            for extent in declaration.shape
            for factor in extent.factors
        )

    def _invoke_step(
        self,
        step: ArtifactRuntimeStep,
        resources: Mapping[str, object],
    ) -> None:
        for operation in step.prelaunch:
            target = _resource_value(
                operation.target,
                resources,
                field=f"artifact step {step.name!r} prelaunch",
            )
            fill = getattr(target, "fill_", None)
            if not callable(fill):
                raise ArtifactRuntimeError(
                    f"artifact prelaunch target {operation.target!r} is not fillable"
                )
            fill(_fill_value(operation.value))
        projected = tuple(
            _project_binding(
                binding,
                resources,
                self._resource_table,
                self._provider,
                field=f"artifact step {step.name!r} binding[{index}]",
            )
            for index, binding in enumerate(step.bindings)
        )
        step.executor(*projected)

    def _lifecycle_key(
        self,
        plan: _ArtifactPlan,
        role: str,
        resources: Mapping[str, object],
        *,
        steps: Sequence[ArtifactRuntimeStep] | None = None,
    ) -> object:
        rows: list[tuple[str, object]] = []
        selected = plan.steps if steps is None else steps
        for name in _step_resource_names(selected, role):
            descriptor = self._resource_table[name]
            if descriptor.kind == "stream" and name not in resources:
                identity: object = (
                    "provider-current-stream",
                    self._execution_scope(),
                )
            else:
                identity = _resource_identity(
                    _resource_value(name, resources, field=f"artifact {role}")
                )
            rows.append((name, identity))
        return (plan.name, role, tuple(rows) if rows else (("process",),))

    def _ensure_initialized(
        self, plan: _ArtifactPlan, resources: Mapping[str, object]
    ) -> None:
        steps = tuple(step for step in plan.steps if step.role == "init")
        if not steps:
            return
        key = self._lifecycle_key(plan, "init", resources)
        if key in self._initialized:
            return
        self._require_pre_capture("initialization")
        for step in steps:
            self._invoke_step(step, resources)
        self._initialized.add(key)

    def _record_destroy(
        self, plan: _ArtifactPlan, resources: Mapping[str, object]
    ) -> None:
        frozen = MappingProxyType(dict(resources))
        for step in plan.steps:
            if step.role != "destroy":
                continue
            key = self._lifecycle_key(
                plan, "destroy", resources, steps=(step,)
            )
            self._destroy_frames.setdefault(
                (step.name, step.step, key), (step, frozen)
            )

    def _prepared_key(self, resources: Mapping[str, object]) -> object:
        names = self._call_abi.prepare_args
        if not names:
            return ("process",)
        return tuple(
            (
                name,
                _resource_identity(
                    _resource_value(name, resources, field="artifact prepare")
                ),
            )
            for name in names
        )

    def prepare(self, *args, **kwargs) -> ArtifactPreparedState:
        """Run validator-owned prepare pipelines outside graph capture.

        The result is an exact, owner-bound token.  Tensor frames stay exclusively
        inside this entry so external dispatcher caches cannot outlive allocation
        accounting. Run accepts no miner-created replacement.
        """

        if kwargs:
            raise ArtifactRuntimeError("artifact prepare accepts positional resources only")
        base = self._base_frame(self._call_abi.prepare_args, tuple(args))
        key = self._prepared_key(base)
        with self._lock:
            if self._closed or self._closing:
                raise ArtifactRuntimeError("artifact runtime entry is closing or closed")
            cached = self._prepared_by_key.get(key)
            if cached is not None:
                return cached
            self._require_pre_capture("prepare")
            if len(self._prepared_by_key) >= self._key_limit:
                raise ArtifactRuntimeError(
                    "artifact runtime exceeded its prepared-key limit"
                )
            frames: dict[str, Mapping[str, object]] = {}
            launched = False
            execution_scope = self._execution_scope()
            group_scope = self._group_scope(base)
            for plan in self._plans:
                frame = dict(base)
                state_rows = self._declarations["state"]
                if self._can_shape(state_rows, frame):
                    frame.update(
                        self._resource_frame(
                            plan,
                            "state",
                            frame,
                            instance_key=("state", execution_scope, group_scope),
                        )
                    )
                prepared_rows = self._declarations["prepared"]
                if prepared_rows:
                    if not self._can_shape(prepared_rows, frame):
                        raise ArtifactRuntimeError(
                            "prepared artifact resource shape is unavailable at the "
                            "validator prepare boundary"
                        )
                    frame.update(
                        self._resource_frame(
                            plan, "prepared", frame, instance_key=key
                        )
                    )
                frozen = MappingProxyType(frame)
                self._ensure_initialized(plan, frozen)
                for step in plan.steps:
                    if step.role == "prepare":
                        self._invoke_step(step, frozen)
                        launched = True
                self._record_destroy(plan, frozen)
                frames[plan.name] = frozen
            # Explicit prepare may run on a different stream from graph warmup.
            # Complete it before publishing the owner-bound token to another
            # engine scope.  This is a one-time preparation cost, never timed run.
            if launched:
                self._synchronize("prepare")
            token = object()
            result = ArtifactPreparedState(self._owner, key, token)
            self._prepared_frames[token] = MappingProxyType(frames)
            self._prepared_by_key[key] = result
            return result

    def _implicit_frame(
        self,
        plan: _ArtifactPlan,
        base: Mapping[str, object],
        execution_scope: object,
    ) -> Mapping[str, object]:
        names = {
            name
            for name in _step_resource_names(plan.steps, "prepare")
            if name in self._call_abi.by_name
            and self._call_abi.by_name[name].kind != "stream"
        }
        names.update(
            factor.source
            for declaration in self._declarations["prepared"]
            for extent in declaration.shape
            for factor in extent.factors
            if factor.source is not None
        )
        key = (
            plan.name,
            execution_scope,
            tuple(
                (name, _resource_identity(_resource_value(name, base, field="prepare")))
                for name in sorted(names)
            ),
        )
        cached = self._implicit_cache.get(key)
        if cached is not None:
            return cached
        self._require_pre_capture("implicit prepare")
        if len(self._implicit_cache) >= self._key_limit:
            raise ArtifactRuntimeError("artifact runtime exceeded implicit prepare keys")
        frame = dict(base)
        group_scope = self._group_scope(base)
        state_rows = self._declarations["state"]
        if self._can_shape(state_rows, frame):
            frame.update(
                self._resource_frame(
                    plan,
                    "state",
                    frame,
                    instance_key=("state", execution_scope, group_scope),
                )
            )
        prepared_rows = self._declarations["prepared"]
        if prepared_rows:
            frame.update(
                self._resource_frame(plan, "prepared", frame, instance_key=key)
            )
        frozen = MappingProxyType(frame)
        self._ensure_initialized(plan, frozen)
        for step in plan.steps:
            if step.role == "prepare":
                self._invoke_step(step, frozen)
        self._record_destroy(plan, frozen)
        self._implicit_cache[key] = frozen
        return frozen

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise ArtifactRuntimeError("artifact entries accept positional resources only")
        base = self._base_frame(self._call_abi.call_args, tuple(args))
        plan = self._select(base)
        with self._lock:
            if self._closed or self._closing:
                raise ArtifactRuntimeError("artifact runtime entry is closing or closed")
            execution_scope = self._execution_scope()
            group_scope = self._group_scope(base)
            frame = dict(base)
            prepare_result = self._call_abi.prepare_result
            if prepare_result is not None:
                state = _resource_value(
                    prepare_result, base, field="artifact prepared state"
                )
                if (
                    type(state) is not ArtifactPreparedState
                    or state._owner is not self._owner
                    or self._prepared_by_key.get(state._key) is not state
                ):
                    raise ArtifactRuntimeError(
                        "artifact run received a foreign prepared-state frame"
                    )
                try:
                    frames = self._prepared_frames[state._token]
                    frame.update(frames[plan.name])
                except KeyError:
                    raise ArtifactRuntimeError(
                        f"prepared state lacks plan {plan.name!r}"
                    ) from None
            elif any(step.role == "prepare" for step in plan.steps) or self._declarations[
                "prepared"
            ]:
                frame.update(self._implicit_frame(plan, base, execution_scope))

            state_rows = self._declarations["state"]
            if self._can_shape(state_rows, frame):
                frame.update(
                    self._resource_frame(
                        plan,
                        "state",
                        frame,
                        instance_key=("state", execution_scope, group_scope),
                    )
                )
            frame.update(
                self._resource_frame(
                    plan,
                    "workspace",
                    frame,
                    instance_key=("execution", execution_scope, group_scope),
                )
            )
            frozen = MappingProxyType(frame)
            self._ensure_initialized(plan, frozen)
            self._record_destroy(plan, frozen)
            # Keep teardown and another host thread from interleaving between
            # frame selection and launch.  Distinct execution scopes have
            # distinct writable state/workspace; same-stream CUDA ordering
            # serializes their asynchronous launches.
            for step in plan.steps:
                if step.role in {"reset", "run"}:
                    self._invoke_step(step, frozen)
        return None

    def close(self) -> None:
        """Run deduplicated destroy phases and release every retained frame."""

        with self._lock:
            if self._closed:
                return
            self._require_pre_capture("teardown")
            self._closing = True
            failures: list[str] = []
            if self._resource_frames or self._destroy_frames:
                try:
                    self._synchronize("pre-destroy")
                except ArtifactRuntimeError as exc:
                    failures.append(str(exc))
            if not failures:
                for step, resources in tuple(self._destroy_frames.values()):
                    try:
                        self._invoke_step(step, resources)
                    except Exception as exc:  # noqa: BLE001 - attempt every cleanup
                        failures.append(f"{step.name}: {exc}")
                if self._destroy_frames and not failures:
                    try:
                        self._synchronize("destroy")
                    except ArtifactRuntimeError as exc:
                        failures.append(str(exc))
            if failures:
                # Keep all frames and budget reservations alive.  The entry is
                # terminally closing and cannot race another run; process-level
                # teardown remains the final containment boundary.
                raise ArtifactRuntimeError(
                    "artifact destroy phase failed: " + "; ".join(failures)
                )
            self._closed = True
            self._destroy_frames.clear()
            self._prepared_by_key.clear()
            self._prepared_frames.clear()
            self._implicit_cache.clear()
            self._resource_frames.clear()
            self._allocated_bytes = 0
            self._allocated_keys = 0
            if self._allocation_budget is not None:
                self._allocation_budget.release_owner(self._owner)


def resolve_direct_artifact_entry(op: object) -> object | None:
    """Route one sealed direct op to its already-prepared provider state."""

    exports = getattr(op, "aot_exports", ())
    if not exports:
        return None
    providers = {getattr(export, "provider", None) for export in exports}
    if len(providers) != 1:
        raise ArtifactRuntimeError(
            "direct artifact op does not name exactly one provider"
        )
    provider = next(iter(providers))
    from optima.artifact_provider import CUTE_CUBIN_PROVIDER_ID

    if provider == CUTE_CUBIN_PROVIDER_ID:
        from optima.cute_cubin import resolve_cute_cubin_entry

        return resolve_cute_cubin_entry(op)
    raise ArtifactRuntimeError(
        f"direct artifact provider {provider!r} has no runtime resolver"
    )


def shutdown_direct_artifact_runtimes() -> None:
    """Close the device-artifact provider state."""

    from optima.cute_cubin import shutdown_cute_cubin_runtime

    shutdown_cute_cubin_runtime()


__all__ = [
    "ArtifactAllocationBudget",
    "ArtifactRuntimeEntry",
    "ArtifactRuntimeError",
    "ArtifactRuntimeLimits",
    "ArtifactPreparedState",
    "ArtifactRuntimeProvider",
    "ArtifactRuntimeStep",
    "artifact_call_resources",
    "resolve_direct_artifact_entry",
    "shutdown_direct_artifact_runtimes",
]
