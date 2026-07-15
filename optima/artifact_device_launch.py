"""Import-light schema and runtime for sealed device-only CUDA launch plans.

The serialized plan is the single ABI authority: it declares a complete logical
kernel inventory, every formal parameter width, and an ordered bounded launch
list.  Canonical logical-name order defines kernel ordinal; runtime binds those
ordinals to the complete physical inventory observed from the exact sealed
CUBIN.  Runtime admission and launch retain one exact CUDA library handle; there
is no candidate host object or executable callback.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

from optima.cuda_cubin import (
    CudaCubinContract,
    CudaCubinError,
    CudaCubinLibrary,
    CudaKernelContract,
    cuda_cubin_identity,
)
from optima.cuda_launch import (
    CudaDim3,
    CudaLaunchAttributes,
    CudaLaunchError,
    CudaLaunchSpec,
    launch_cuda_kernel,
)
from optima.cuda_materialize import (
    CUDA_CHECKED_EXPRESSION_CAPABILITY,
    CUDA_PACKED_STRUCT_CAPABILITY,
    CUDA_TMA_DESCRIPTOR_CAPABILITY,
    CUTLASS_FAST_DIVMOD_CAPABILITY,
    CUTE_FAST_DIVMOD_CAPABILITY,
    CudaCheckedExpression,
    CudaMaterializeError,
    CudaParameterPlan,
    CudaPrimitiveRegistry,
    evaluate_cuda_expression,
    materialize_cuda_parameter,
    validate_parameter_bindings,
)
from optima.stack_identity import canonical_digest, require_sha256_hex


DEVICE_LAUNCH_PLAN_SCHEMA = "optima.device-launch-plan.v1"
DEVICE_ARTIFACT_ADMISSION_SCHEMA = "optima.device-artifact-admission.v1"

_MAX_KERNELS = 1_024
_MAX_LAUNCHES = 256
_MAX_BINDINGS = 64
_UINT32_MAX = (1 << 32) - 1
_MAX_SHARED_MEMORY = 1 << 20
_KERNEL_INTRINSIC_CAPABILITIES = frozenset(
    {
        CUDA_CHECKED_EXPRESSION_CAPABILITY,
        CUDA_PACKED_STRUCT_CAPABILITY,
        CUDA_TMA_DESCRIPTOR_CAPABILITY,
        CUTLASS_FAST_DIVMOD_CAPABILITY,
        CUTE_FAST_DIVMOD_CAPABILITY,
    }
)


class DeviceLaunchError(RuntimeError):
    """A serialized device plan, admission, or materialized launch is invalid."""


def _device_error(exc: Exception) -> DeviceLaunchError:
    return DeviceLaunchError(str(exc))


@dataclass(frozen=True)
class DeviceArtifactAdmission:
    """Handle-free evidence observed from the exact retained CUDA library."""

    cubin_sha256: str
    cubin_size: int
    observed_abi_digest: str
    observed_contract_digest: str
    schema: str = DEVICE_ARTIFACT_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "cubin_sha256",
            "observed_abi_digest",
            "observed_contract_digest",
        ):
            try:
                require_sha256_hex(getattr(self, field_name), field=field_name)
            except (TypeError, ValueError) as exc:
                raise DeviceLaunchError(str(exc)) from None
        if type(self.cubin_size) is not int or not 1 <= self.cubin_size <= 1 << 30:
            raise DeviceLaunchError("device admission CUBIN size is outside policy")
        if self.schema != DEVICE_ARTIFACT_ADMISSION_SCHEMA:
            raise DeviceLaunchError("device admission schema mismatch")

    @property
    def digest(self) -> str:
        return canonical_digest("optima.device-artifact-admission", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "cubin_sha256": self.cubin_sha256,
            "cubin_size": self.cubin_size,
            "observed_abi_digest": self.observed_abi_digest,
            "observed_contract_digest": self.observed_contract_digest,
            "schema": self.schema,
        }


@dataclass(frozen=True)
class DeviceDim3Plan:
    """Three checked expressions for a CUDA grid, block, or cluster extent."""

    x: CudaCheckedExpression
    y: CudaCheckedExpression
    z: CudaCheckedExpression

    def __post_init__(self) -> None:
        if any(
            type(value) is not CudaCheckedExpression
            for value in (self.x, self.y, self.z)
        ):
            raise DeviceLaunchError("device dimension expressions are malformed")

    @property
    def expressions(self) -> tuple[CudaCheckedExpression, ...]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict[str, object]:
        return {
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
            "z": self.z.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "DeviceDim3Plan":
        if type(value) is not dict or set(value) != {"x", "y", "z"}:
            raise DeviceLaunchError("device dimension fields mismatch")
        try:
            return cls(
                CudaCheckedExpression.from_dict(value["x"]),
                CudaCheckedExpression.from_dict(value["y"]),
                CudaCheckedExpression.from_dict(value["z"]),
            )
        except CudaMaterializeError as exc:
            raise _device_error(exc) from None


@dataclass(frozen=True)
class DeviceLaunchInvocation:
    """One ordinal in a validator-executed, same-stream launch sequence."""

    ordinal: int
    kernel: str
    grid: DeviceDim3Plan
    block: DeviceDim3Plan
    cluster: DeviceDim3Plan | None
    shared_mem_bytes: CudaCheckedExpression
    parameters: tuple[CudaParameterPlan, ...]
    stream_binding: int | None
    attributes: CudaLaunchAttributes = field(default_factory=CudaLaunchAttributes)

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < _MAX_LAUNCHES:
            raise DeviceLaunchError("device launch ordinal is outside policy")
        try:
            CudaKernelContract(name=self.kernel, parameter_sizes=())
        except CudaCubinError as exc:
            raise _device_error(exc) from None
        if type(self.grid) is not DeviceDim3Plan or type(self.block) is not DeviceDim3Plan:
            raise DeviceLaunchError("device launch grid/block plan is malformed")
        if self.cluster is not None and type(self.cluster) is not DeviceDim3Plan:
            raise DeviceLaunchError("device launch cluster plan is malformed")
        if type(self.shared_mem_bytes) is not CudaCheckedExpression:
            raise DeviceLaunchError("device launch shared-memory expression is malformed")
        if (
            type(self.parameters) is not tuple
            or len(self.parameters) > 256
            or not all(type(parameter) is CudaParameterPlan for parameter in self.parameters)
        ):
            raise DeviceLaunchError("device launch parameters are malformed")
        if self.stream_binding is not None and (
            type(self.stream_binding) is not int
            or not 0 <= self.stream_binding < _MAX_BINDINGS
        ):
            raise DeviceLaunchError("device launch stream binding is outside policy")
        if type(self.attributes) is not CudaLaunchAttributes:
            raise DeviceLaunchError("device launch attributes are malformed")

    @property
    def expressions(self) -> tuple[CudaCheckedExpression, ...]:
        rows = list(self.grid.expressions + self.block.expressions)
        if self.cluster is not None:
            rows.extend(self.cluster.expressions)
        rows.append(self.shared_mem_bytes)
        return tuple(rows)

    @property
    def required_capabilities(self) -> frozenset[str]:
        return frozenset(
            capability
            for parameter in self.parameters
            for capability in parameter.required_capabilities
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "attributes": self.attributes.to_dict(),
            "block": self.block.to_dict(),
            "cluster": None if self.cluster is None else self.cluster.to_dict(),
            "grid": self.grid.to_dict(),
            "kernel": self.kernel,
            "ordinal": self.ordinal,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "shared_mem_bytes": self.shared_mem_bytes.to_dict(),
            "stream_binding": self.stream_binding,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DeviceLaunchInvocation":
        expected = {
            "attributes",
            "block",
            "cluster",
            "grid",
            "kernel",
            "ordinal",
            "parameters",
            "shared_mem_bytes",
            "stream_binding",
        }
        required = expected - {"cluster"}
        if (
            type(value) is not dict
            or not required <= set(value)
            or not set(value) <= expected
        ):
            raise DeviceLaunchError("device invocation fields mismatch")
        parameters = value["parameters"]
        if type(parameters) is not list:
            raise DeviceLaunchError("device invocation parameters must be a list")
        cluster = value.get("cluster")
        try:
            return cls(
                ordinal=value["ordinal"],  # type: ignore[arg-type]
                kernel=value["kernel"],  # type: ignore[arg-type]
                grid=DeviceDim3Plan.from_dict(value["grid"]),
                block=DeviceDim3Plan.from_dict(value["block"]),
                cluster=(
                    None if cluster is None else DeviceDim3Plan.from_dict(cluster)
                ),
                shared_mem_bytes=CudaCheckedExpression.from_dict(
                    value["shared_mem_bytes"]
                ),
                parameters=tuple(
                    CudaParameterPlan.from_dict(parameter)
                    for parameter in parameters
                ),
                stream_binding=value["stream_binding"],  # type: ignore[arg-type]
                attributes=CudaLaunchAttributes.from_dict(value["attributes"]),
            )
        except (CudaMaterializeError, CudaLaunchError) as exc:
            raise _device_error(exc) from None


@dataclass(frozen=True)
class DeviceLaunchPlan:
    """Complete bounded symbol contract plus ordered device invocation sequence."""

    kernels: tuple[CudaKernelContract, ...]
    launches: tuple[DeviceLaunchInvocation, ...]
    schema: str = DEVICE_LAUNCH_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.kernels) is not tuple
            or not 1 <= len(self.kernels) <= _MAX_KERNELS
            or not all(type(kernel) is CudaKernelContract for kernel in self.kernels)
        ):
            raise DeviceLaunchError("device kernel contract inventory is outside policy")
        names = tuple(kernel.name for kernel in self.kernels)
        if names != tuple(sorted(set(names))):
            raise DeviceLaunchError("device kernel contract inventory is not canonical")
        if (
            type(self.launches) is not tuple
            or not 1 <= len(self.launches) <= _MAX_LAUNCHES
            or not all(type(launch) is DeviceLaunchInvocation for launch in self.launches)
        ):
            raise DeviceLaunchError("device invocation inventory is outside policy")
        if tuple(launch.ordinal for launch in self.launches) != tuple(
            range(len(self.launches))
        ):
            raise DeviceLaunchError("device invocation ordinals are not contiguous")
        referenced = {launch.kernel for launch in self.launches}
        missing = referenced - set(names)
        if missing:
            raise DeviceLaunchError(
                f"device launch references undeclared CUBIN symbols {sorted(missing)!r}"
            )
        by_name = {kernel.name: kernel for kernel in self.kernels}
        for launch in self.launches:
            declared = by_name[launch.kernel].parameter_sizes
            observed = tuple(parameter.size for parameter in launch.parameters)
            if declared != observed:
                raise DeviceLaunchError(
                    f"device launch {launch.ordinal} parameter widths differ from "
                    f"kernel contract {launch.kernel!r}"
                )
        if self.schema != DEVICE_LAUNCH_PLAN_SCHEMA:
            raise DeviceLaunchError("device launch plan schema mismatch")

    @property
    def digest(self) -> str:
        return canonical_digest("optima.device-launch-plan", self.to_dict())

    @property
    def required_capabilities(self) -> frozenset[str]:
        return frozenset(
            capability
            for launch in self.launches
            for capability in launch.required_capabilities
        )

    def cuda_contract(self, cubin_sha256: str, cubin_size: int) -> CudaCubinContract:
        try:
            return CudaCubinContract(
                cubin_sha256=cubin_sha256,
                cubin_size=cubin_size,
                kernels=self.kernels,
            )
        except CudaCubinError as exc:
            raise _device_error(exc) from None

    def bind_observed_contract(
        self, observed: CudaCubinContract
    ) -> "DeviceLaunchPlan":
        """Resolve logical aliases against a complete physical inventory.

        Both inventories are canonical, so their tuple positions are the only
        selector.  Widths are checked per position rather than searched or
        matched, which remains unambiguous when multiple kernels share an ABI.
        """

        if type(observed) is not CudaCubinContract:
            raise DeviceLaunchError("observed device contract has the wrong type")
        declared_widths = tuple(kernel.parameter_sizes for kernel in self.kernels)
        observed_widths = tuple(kernel.parameter_sizes for kernel in observed.kernels)
        if observed_widths != declared_widths:
            raise DeviceLaunchError(
                "observed device kernel ordinal widths differ from the declaration"
            )
        names = {
            declared.name: physical.name
            for declared, physical in zip(self.kernels, observed.kernels, strict=True)
        }
        return DeviceLaunchPlan(
            kernels=observed.kernels,
            launches=tuple(
                replace(launch, kernel=names[launch.kernel])
                for launch in self.launches
            ),
            schema=self.schema,
        )

    def validate_bindings(
        self,
        bindings: Sequence[object],
        *,
        provider_capabilities: Sequence[str] | frozenset[str],
    ) -> None:
        """Join every plan reference to the sealed semantic ArtifactBinding list."""

        if not isinstance(bindings, Sequence) or len(bindings) > _MAX_BINDINGS:
            raise DeviceLaunchError("device plan bindings are outside policy")
        if not isinstance(provider_capabilities, (tuple, list, set, frozenset)) or any(
            type(capability) is not str for capability in provider_capabilities
        ):
            raise DeviceLaunchError("device provider capabilities are malformed")
        missing = (
            self.required_capabilities - _KERNEL_INTRINSIC_CAPABILITIES
        ) - frozenset(provider_capabilities)
        if missing:
            raise DeviceLaunchError(
                f"device provider lacks required capabilities {sorted(missing)!r}"
            )

        def validate_expression(expression: CudaCheckedExpression) -> None:
            for index, op in expression.binding_references:
                if not 0 <= index < len(bindings):
                    raise DeviceLaunchError(
                        f"device expression references missing binding {index}"
                    )
                expected = "scalar" if op == "binding_scalar" else "tensor"
                observed = getattr(bindings[index], "kind", None)
                if observed != expected:
                    raise DeviceLaunchError(
                        f"device expression {op!r} requires {expected} binding; "
                        f"binding {index} is {observed!r}"
                    )

        try:
            for launch in self.launches:
                if launch.stream_binding is not None:
                    if launch.stream_binding >= len(bindings) or getattr(
                        bindings[launch.stream_binding], "kind", None
                    ) != "stream":
                        raise DeviceLaunchError(
                            "device launch stream reference is not a sealed stream binding"
                        )
                for expression in launch.expressions:
                    validate_expression(expression)
                for parameter in launch.parameters:
                    validate_parameter_bindings(parameter, bindings)
        except CudaMaterializeError as exc:
            raise _device_error(exc) from None

    def to_dict(self) -> dict[str, object]:
        return {
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "launches": [launch.to_dict() for launch in self.launches],
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DeviceLaunchPlan":
        if type(value) is not dict or set(value) != {"kernels", "launches", "schema"}:
            raise DeviceLaunchError("device launch plan fields mismatch")
        kernels = value["kernels"]
        launches = value["launches"]
        if type(kernels) is not list or type(launches) is not list:
            raise DeviceLaunchError("device launch plan inventories must be lists")
        try:
            return cls(
                kernels=tuple(CudaKernelContract.from_dict(kernel) for kernel in kernels),
                launches=tuple(
                    DeviceLaunchInvocation.from_dict(launch) for launch in launches
                ),
                schema=value["schema"],  # type: ignore[arg-type]
            )
        except CudaCubinError as exc:
            raise _device_error(exc) from None


def make_device_artifact_runtime_provider(
    *,
    driver: object,
    group_capabilities: Mapping[
        str, Callable[[object, object, object | None], object]
    ]
    | None = None,
) -> object:
    """Capture the generic ArtifactRuntimeProvider for raw device plans.

    ``kind='tensor'`` remains a checked validator-owned tensor/view. The
    downstream TMA/pointer materializers need
    its live shape, element-stride, element-size, and address; converting it to a
    CuTe Python descriptor would lose that authority and reintroduce a host ABI.
    """

    from optima.artifact_provider import CUTE_CUBIN_PROVIDER_ID
    from optima.artifact_runtime import ArtifactRuntimeError, ArtifactRuntimeProvider

    try:
        import torch
        import torch.distributed as torch_distributed
    except Exception as exc:  # noqa: BLE001 - optional engine dependency
        raise DeviceLaunchError(
            f"device artifact runtime dependencies are unavailable: {exc}"
        ) from None

    current_stream = torch.cuda.current_stream
    current_device = torch.cuda.current_device
    is_current_stream_capturing = torch.cuda.is_current_stream_capturing
    synchronize_device = torch.cuda.synchronize
    torch_empty = torch.empty
    torch_uint8 = torch.uint8
    distributed_rank = torch_distributed.get_rank
    distributed_size = torch_distributed.get_world_size
    stream_type = getattr(driver, "CUstream", None)
    if not isinstance(stream_type, type):
        raise DeviceLaunchError("CUDA driver lacks the CUstream handle type")
    capabilities = dict(group_capabilities or {})
    unknown = set(capabilities) - {"native_handle", "peer_ptr_table"}
    if unknown or not all(callable(value) for value in capabilities.values()):
        raise DeviceLaunchError(
            f"device group capability resolvers are invalid: {sorted(unknown)!r}"
        )

    storage_dtype_names = {
        "bool": "bool",
        "i8": "int8",
        "u8": "uint8",
        "f8e4m3fn": "float8_e4m3fn",
        "f8e4m3fnuz": "float8_e4m3fnuz",
        "f8e5m2": "float8_e5m2",
        "f8e5m2fnuz": "float8_e5m2fnuz",
        "i16": "int16",
        "u16": "uint16",
        "f16": "float16",
        "bf16": "bfloat16",
        "i32": "int32",
        "u32": "uint32",
        "f32": "float32",
        "i64": "int64",
        "u64": "uint64",
        "f64": "float64",
    }
    storage_dtypes = {
        name: getattr(torch, attribute, None)
        for name, attribute in storage_dtype_names.items()
    }

    def aligned(value: object, binding: object) -> None:
        alignment = getattr(binding, "assumed_align", None)
        if alignment is None:
            return
        try:
            pointer = int(value.data_ptr())
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ArtifactRuntimeError(f"cannot inspect tensor alignment: {exc}") from None
        if pointer % alignment:
            raise ArtifactRuntimeError(
                f"tensor pointer is not {alignment}-byte aligned"
            )

    def tensor_descriptor(value: object, binding: object) -> object:
        aligned(value, binding)
        view = value
        for dimension in getattr(binding, "unsqueeze", ()):
            try:
                view = view.unsqueeze(dimension)
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                raise ArtifactRuntimeError(
                    f"cannot apply device-view unsqueeze({dimension}): {exc}"
                ) from None
        return view

    def tensor_pointer(value: object, binding: object) -> int:
        aligned(value, binding)
        try:
            return int(value.data_ptr())
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ArtifactRuntimeError(
                f"cannot project validator tensor pointer: {exc}"
            ) from None

    def stream() -> object:
        try:
            return stream_type(current_stream().cuda_stream)
        except Exception as exc:  # noqa: BLE001 - CUDA binding variants
            raise ArtifactRuntimeError(
                f"cannot project validator CUDA stream: {exc}"
            ) from None

    def group_rank(group: object) -> int:
        try:
            return int(distributed_rank(group=group))
        except Exception:  # noqa: BLE001 - reviewed group wrappers
            try:
                return int(group.rank())  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                raise ArtifactRuntimeError(
                    f"cannot resolve validator group rank: {exc}"
                ) from None

    def group_size(group: object) -> int:
        try:
            size = int(distributed_size(group=group))
        except Exception:  # noqa: BLE001 - reviewed group wrappers
            try:
                size = int(group.size())  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                raise ArtifactRuntimeError(
                    f"cannot resolve validator group size: {exc}"
                ) from None
        if size <= 0:
            raise ArtifactRuntimeError("validator group size is not positive")
        return size

    def group_pointer(
        group: object,
        projection: str,
        binding: object,
        peer_resource: object | None,
    ) -> object:
        try:
            resolver = capabilities[projection]
        except KeyError:
            raise ArtifactRuntimeError(
                f"device provider cannot materialize group projection {projection!r}"
            ) from None
        return resolver(group, binding, peer_resource)

    def pointer_identity(value: object, binding: object) -> int:
        try:
            pointer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ArtifactRuntimeError(
                f"cannot project validator pointer identity: {exc}"
            ) from None
        alignment = getattr(binding, "assumed_align", None)
        if alignment is not None and pointer % alignment:
            raise ArtifactRuntimeError(
                f"native pointer is not {alignment}-byte aligned"
            )
        return pointer

    def allocate_tensor(shape: tuple[int, ...], dtype: str, alignment: int) -> object:
        torch_dtype = storage_dtypes.get(dtype)
        if torch_dtype is None:
            raise ArtifactRuntimeError(
                f"device provider cannot allocate artifact dtype {dtype!r}"
            )
        try:
            element_size = int(torch_empty((), dtype=torch_dtype).element_size())
            byte_count = math.prod(shape) * element_size
            raw = torch_empty(
                byte_count + alignment - 1,
                dtype=torch_uint8,
                device=int(current_device()),
            )
            offset = (-int(raw.data_ptr())) % alignment
            view = raw.narrow(0, offset, byte_count).view(torch_dtype).view(shape)
        except Exception as exc:  # noqa: BLE001 - runtime dtype/device variants
            raise ArtifactRuntimeError(
                f"device provider artifact allocation failed: {exc}"
            ) from None
        return view

    def validate_allocation(
        value: object,
        shape: tuple[int, ...],
        dtype: str,
        alignment: int,
    ) -> None:
        expected_dtype = storage_dtypes.get(dtype)
        try:
            observed_shape = tuple(int(dimension) for dimension in value.shape)
            pointer = int(value.data_ptr())
            device = value.device
            valid = (
                observed_shape == shape
                and value.dtype == expected_dtype
                and getattr(device, "type", None) == "cuda"
                and int(getattr(device, "index", current_device()))
                == int(current_device())
                and pointer % alignment == 0
                and bool(value.is_contiguous())
            )
        except Exception as exc:  # noqa: BLE001
            raise ArtifactRuntimeError(
                f"device provider allocation is uninspectable: {exc}"
            ) from None
        if not valid:
            raise ArtifactRuntimeError(
                "device provider allocation differs from sealed request"
            )

    def execution_scope() -> object:
        live = current_stream()
        return ("cuda", int(current_device()), int(live.cuda_stream))

    return ArtifactRuntimeProvider(
        provider=CUTE_CUBIN_PROVIDER_ID,
        tensor_descriptor=tensor_descriptor,
        tensor_pointer=tensor_pointer,
        current_stream=stream,
        group_rank=group_rank,
        group_size=group_size,
        group_pointer=group_pointer,
        pointer_identity=pointer_identity,
        provider_capabilities=frozenset(
            {
                "native_handle": "group.native_handle.v1",
                "peer_ptr_table": "group.peer_ptr_table.v1",
            }[projection]
            for projection in capabilities
        ),
        allocate_tensor=allocate_tensor,
        validate_allocation=validate_allocation,
        is_capturing=lambda: bool(is_current_stream_capturing()),
        execution_scope=execution_scope,
        synchronize=lambda: synchronize_device(),
    )


def _materialized_dim3(
    plan: DeviceDim3Plan,
    bindings: tuple[object, ...],
    registry: CudaPrimitiveRegistry,
    *,
    field: str,
) -> CudaDim3:
    values = tuple(
        evaluate_cuda_expression(expression, bindings, registry)
        for expression in plan.expressions
    )
    if any(type(value) is not int or not 1 <= value <= _UINT32_MAX for value in values):
        raise DeviceLaunchError(f"{field} dimensions are not positive uint32 values")
    try:
        return CudaDim3(*values)  # type: ignore[arg-type]
    except CudaLaunchError as exc:
        raise _device_error(exc) from None


class DeviceArtifactRuntime:
    """One admitted library and its validator-owned ordered launch executor."""

    __slots__ = ("_admission", "_library", "_lock", "_plan", "_registry")

    def __init__(
        self,
        library: CudaCubinLibrary,
        plan: DeviceLaunchPlan,
        registry: CudaPrimitiveRegistry,
    ) -> None:
        if type(library) is not CudaCubinLibrary or type(plan) is not DeviceLaunchPlan:
            raise DeviceLaunchError("device runtime authority is malformed")
        if type(registry) is not CudaPrimitiveRegistry:
            raise DeviceLaunchError("device runtime primitive registry is malformed")
        if registry.driver is not object.__getattribute__(library, "_driver"):
            raise DeviceLaunchError(
                "device primitive registry did not capture the admitted library driver"
            )
        observed_abi = library.abi
        observed_contract = observed_abi.contract
        if observed_contract.kernels != plan.kernels:
            raise DeviceLaunchError(
                "admitted CUDA library symbol contract differs from device plan"
            )
        missing = plan.required_capabilities - registry.capabilities
        if missing:
            raise DeviceLaunchError(
                f"runtime primitive registry lacks {sorted(missing)!r}"
            )
        self._admission = DeviceArtifactAdmission(
            cubin_sha256=observed_abi.cubin_sha256,
            cubin_size=observed_abi.cubin_size,
            observed_abi_digest=observed_abi.digest,
            observed_contract_digest=observed_contract.digest,
        )
        self._library = library
        self._lock = threading.RLock()
        self._plan = plan
        self._registry = registry

    @classmethod
    def admit(
        cls,
        cubin: bytes | bytearray | memoryview,
        plan: DeviceLaunchPlan,
        registry: CudaPrimitiveRegistry,
        *,
        driver: object | None = None,
    ) -> "DeviceArtifactRuntime":
        if type(plan) is not DeviceLaunchPlan:
            raise DeviceLaunchError("device admission plan has the wrong type")
        if type(registry) is not CudaPrimitiveRegistry:
            raise DeviceLaunchError("device admission registry has the wrong type")
        captured_driver = registry.driver if driver is None else driver
        if captured_driver is not registry.driver:
            raise DeviceLaunchError(
                "device admission driver differs from the primitive registry driver"
            )
        try:
            raw, cubin_sha256, cubin_size = cuda_cubin_identity(cubin)
            contract = plan.cuda_contract(cubin_sha256, cubin_size)
            library = CudaCubinLibrary.open_ordered_contract(
                raw,
                expected_contract=contract,
                driver=captured_driver,
            )
        except (CudaCubinError, TypeError, ValueError) as exc:
            raise _device_error(exc) from None
        try:
            resolved_plan = plan.bind_observed_contract(library.abi.contract)
            return cls(library, resolved_plan, registry)
        except BaseException:
            library.close()
            raise

    @property
    def closed(self) -> bool:
        return self._library.closed

    @property
    def admission(self) -> DeviceArtifactAdmission:
        return self._admission

    def __call__(self, *bindings: object) -> None:
        with self._lock:
            if self._library.closed:
                raise DeviceLaunchError("device artifact runtime is closed")
            frame = tuple(bindings)
            for invocation in self._plan.launches:
                try:
                    grid = _materialized_dim3(
                        invocation.grid, frame, self._registry, field="CUDA grid"
                    )
                    block = _materialized_dim3(
                        invocation.block, frame, self._registry, field="CUDA block"
                    )
                    cluster = (
                        None
                        if invocation.cluster is None
                        else _materialized_dim3(
                            invocation.cluster,
                            frame,
                            self._registry,
                            field="CUDA cluster",
                        )
                    )
                    shared_mem = evaluate_cuda_expression(
                        invocation.shared_mem_bytes, frame, self._registry
                    )
                    if (
                        type(shared_mem) is not int
                        or not 0 <= shared_mem <= _MAX_SHARED_MEMORY
                    ):
                        raise DeviceLaunchError(
                            "CUDA shared-memory expression is outside policy"
                        )
                    spec = CudaLaunchSpec(
                        kernel=invocation.kernel,
                        grid=grid,
                        block=block,
                        cluster=cluster,
                        shared_mem_bytes=shared_mem,
                        attributes=invocation.attributes,
                    )
                    parameters = tuple(
                        materialize_cuda_parameter(parameter, frame, self._registry)
                        for parameter in invocation.parameters
                    )
                    stream = (
                        None
                        if invocation.stream_binding is None
                        else frame[invocation.stream_binding]
                    )
                    launch_cuda_kernel(
                        self._library,
                        spec,
                        parameters,
                        stream=stream,
                    )
                except (CudaLaunchError, CudaMaterializeError, IndexError) as exc:
                    raise _device_error(exc) from None

    def close(self) -> None:
        with self._lock:
            if self._library.closed:
                return
            try:
                # Device launches are asynchronous even for run-only artifacts with
                # no workspace/destroy frames.  Fence unconditionally before the
                # admitted library can be unloaded.
                self._registry.synchronize()
            except Exception as exc:  # noqa: BLE001 - retain handle for worker death
                raise DeviceLaunchError(
                    "device synchronize failed; admitted CUBIN remains loaded and "
                    f"the isolated worker must terminate: {exc}"
                ) from None
            try:
                self._library.close()
            except CudaCubinError as exc:
                raise _device_error(exc) from None

    def __enter__(self) -> "DeviceArtifactRuntime":
        if self.closed:
            raise DeviceLaunchError("device artifact runtime is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class DeviceArtifactEntry:
    """Close an ArtifactRuntimeEntry before fencing/unloading its CUBINs.

    ``ArtifactRuntimeEntry.close`` owns lifecycle destroy steps and retained
    validator allocations.  A run-only entry may legitimately have neither and
    therefore may not synchronize itself; each device runtime below always does.
    This wrapper makes the required teardown order explicit and fail-closed.
    """

    __slots__ = ("_entry", "_runtimes")

    def __init__(
        self,
        entry: object,
        runtimes: tuple[DeviceArtifactRuntime, ...],
    ) -> None:
        if not callable(entry) or not callable(getattr(entry, "close", None)):
            raise DeviceLaunchError("device artifact entry is not closeable")
        if (
            type(runtimes) is not tuple
            or not runtimes
            or not all(type(runtime) is DeviceArtifactRuntime for runtime in runtimes)
        ):
            raise DeviceLaunchError("device artifact runtime inventory is malformed")
        self._entry = entry
        self._runtimes = runtimes

    def prepare(self, *args: object, **kwargs: object) -> object:
        prepare = getattr(self._entry, "prepare", None)
        if not callable(prepare):
            raise DeviceLaunchError("device artifact entry has no prepare boundary")
        return prepare(*args, **kwargs)

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._entry(*args, **kwargs)

    def close(self) -> None:
        try:
            self._entry.close()
        except Exception as exc:  # noqa: BLE001 - retain all state for worker death
            raise DeviceLaunchError(
                "artifact lifecycle close failed; device runtimes remain loaded and "
                f"the isolated worker must terminate: {exc}"
            ) from None
        for runtime in self._runtimes:
            try:
                runtime.close()
            except Exception as exc:  # noqa: BLE001 - worker teardown is final containment
                raise DeviceLaunchError(
                    "device runtime close failed; remaining handles stay retained and "
                    f"the isolated worker must terminate: {exc}"
                ) from None


__all__ = [
    "DEVICE_ARTIFACT_ADMISSION_SCHEMA",
    "DEVICE_LAUNCH_PLAN_SCHEMA",
    "DeviceArtifactAdmission",
    "DeviceArtifactEntry",
    "DeviceArtifactRuntime",
    "DeviceDim3Plan",
    "DeviceLaunchError",
    "DeviceLaunchInvocation",
    "DeviceLaunchPlan",
    "make_device_artifact_runtime_provider",
]
