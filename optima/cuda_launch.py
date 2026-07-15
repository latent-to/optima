"""Validator-owned, declarative CUDA Driver launch boundary.

The device image authority lives in :mod:`optima.cuda_cubin`.  This module
adds only the other half of that boundary: a bounded, JSON-shaped launch
description and exact packing of validator-materialized values into the
driver-observed kernel parameter layout.  It never imports or executes a
candidate host object, Python callback, expression, or argument converter.

CUDA remains an optional runtime dependency.  All CUDA binding objects are
obtained from the already-open :class:`~optima.cuda_cubin.CudaCubinLibrary`
inside the isolated worker.
"""

from __future__ import annotations

import ctypes
import math
import operator
import re
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from optima.cuda_cubin import (
    CudaCubinLibrary,
    CudaKernelABI,
)


CUDA_LAUNCH_SCHEMA = "optima.cuda-launch.v1"

_KERNEL_NAME_RE = re.compile(r"[A-Za-z_.$][A-Za-z0-9_.$@]{0,4095}\Z")
_UINT32_MAX = (1 << 32) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_UINT64_MAX = (1 << 64) - 1
_MAX_THREADS_PER_BLOCK = 1_024
_MAX_CLUSTER_BLOCKS = 32
_MAX_DYNAMIC_SHARED_MEMORY_BYTES = 1 << 20
_MAX_OPAQUE_PARAMETER_BYTES = 4_096


class CudaLaunchError(RuntimeError):
    """A declarative launch, materialized binding, or driver call is invalid."""


class CudaScalarType(str, Enum):
    """Scalar encodings with an exact little-endian CUDA parameter width."""

    BOOL = "bool"
    I8 = "i8"
    U8 = "u8"
    I16 = "i16"
    U16 = "u16"
    I32 = "i32"
    U32 = "u32"
    I64 = "i64"
    U64 = "u64"
    F16 = "f16"
    F32 = "f32"
    F64 = "f64"


class CudaClusterSchedulingPolicy(str, Enum):
    DEFAULT = "default"
    SPREAD = "spread"
    LOAD_BALANCING = "load_balancing"


class CudaPortableClusterMode(str, Enum):
    DEFAULT = "default"
    REQUIRE_PORTABLE = "require_portable"
    ALLOW_NON_PORTABLE = "allow_non_portable"


class CudaSharedMemoryMode(str, Enum):
    DEFAULT = "default"
    REQUIRE_PORTABLE = "require_portable"
    ALLOW_NON_PORTABLE = "allow_non_portable"


def _enum_value(enum_type: type[Enum], value: object, *, field_name: str) -> Enum:
    if type(value) is not str:
        raise CudaLaunchError(f"{field_name} must be a string enum")
    try:
        return enum_type(value)
    except ValueError:
        raise CudaLaunchError(f"{field_name} is not supported") from None


def _strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise CudaLaunchError(f"{field_name} must be a boolean")
    return value


@dataclass(frozen=True)
class CudaDim3:
    """A positive, bounded CUDA three-dimensional extent."""

    x: int
    y: int = 1
    z: int = 1

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or not 1 <= value <= _UINT32_MAX
            for value in (self.x, self.y, self.z)
        ):
            raise CudaLaunchError("CUDA dimensions must be positive uint32 values")

    @property
    def volume(self) -> int:
        return self.x * self.y * self.z

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, value: object) -> "CudaDim3":
        if type(value) is not dict or set(value) != {"x", "y", "z"}:
            raise CudaLaunchError("CUDA dimension fields mismatch")
        return cls(
            x=value["x"],  # type: ignore[arg-type]
            y=value["y"],  # type: ignore[arg-type]
            z=value["z"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaLaunchAttributes:
    """The fixed allowlist of launch-time attributes exposed by the waist."""

    cooperative: bool = False
    cluster_scheduling: CudaClusterSchedulingPolicy | None = None
    programmatic_stream_serialization: bool = False
    priority: int | None = None
    portable_cluster_mode: CudaPortableClusterMode | None = None
    shared_memory_mode: CudaSharedMemoryMode | None = None

    def __post_init__(self) -> None:
        _strict_bool(self.cooperative, field_name="cooperative")
        _strict_bool(
            self.programmatic_stream_serialization,
            field_name="programmatic_stream_serialization",
        )
        if self.cluster_scheduling is not None and type(
            self.cluster_scheduling
        ) is not CudaClusterSchedulingPolicy:
            raise CudaLaunchError("cluster_scheduling has the wrong type")
        if self.portable_cluster_mode is not None and type(
            self.portable_cluster_mode
        ) is not CudaPortableClusterMode:
            raise CudaLaunchError("portable_cluster_mode has the wrong type")
        if self.shared_memory_mode is not None and type(
            self.shared_memory_mode
        ) is not CudaSharedMemoryMode:
            raise CudaLaunchError("shared_memory_mode has the wrong type")
        if self.priority is not None and (
            type(self.priority) is not int
            or not _INT32_MIN <= self.priority <= _INT32_MAX
        ):
            raise CudaLaunchError("CUDA launch priority is outside int32 policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_scheduling": (
                None
                if self.cluster_scheduling is None
                else self.cluster_scheduling.value
            ),
            "cooperative": self.cooperative,
            "portable_cluster_mode": (
                None
                if self.portable_cluster_mode is None
                else self.portable_cluster_mode.value
            ),
            "priority": self.priority,
            "programmatic_stream_serialization": (
                self.programmatic_stream_serialization
            ),
            "shared_memory_mode": (
                None
                if self.shared_memory_mode is None
                else self.shared_memory_mode.value
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaLaunchAttributes":
        allowed = {
            "cluster_scheduling",
            "cooperative",
            "portable_cluster_mode",
            "priority",
            "programmatic_stream_serialization",
            "shared_memory_mode",
        }
        if type(value) is not dict or set(value) - allowed:
            raise CudaLaunchError("CUDA launch attribute fields mismatch")

        raw_cluster_scheduling = value.get("cluster_scheduling")
        cluster_scheduling = (
            None
            if raw_cluster_scheduling is None
            else _enum_value(
                CudaClusterSchedulingPolicy,
                raw_cluster_scheduling,
                field_name="cluster_scheduling",
            )
        )
        raw_portable = value.get("portable_cluster_mode")
        portable = (
            None
            if raw_portable is None
            else _enum_value(
                CudaPortableClusterMode,
                raw_portable,
                field_name="portable_cluster_mode",
            )
        )
        raw_shared = value.get("shared_memory_mode")
        shared = (
            None
            if raw_shared is None
            else _enum_value(
                CudaSharedMemoryMode,
                raw_shared,
                field_name="shared_memory_mode",
            )
        )
        return cls(
            cooperative=_strict_bool(
                value.get("cooperative", False), field_name="cooperative"
            ),
            cluster_scheduling=cluster_scheduling,  # type: ignore[arg-type]
            programmatic_stream_serialization=_strict_bool(
                value.get("programmatic_stream_serialization", False),
                field_name="programmatic_stream_serialization",
            ),
            priority=value.get("priority"),  # type: ignore[arg-type]
            portable_cluster_mode=portable,  # type: ignore[arg-type]
            shared_memory_mode=shared,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaLaunchSpec:
    """One sealed, data-only invocation of a named CUBIN kernel."""

    kernel: str
    grid: CudaDim3
    block: CudaDim3
    cluster: CudaDim3 | None = None
    shared_mem_bytes: int = 0
    attributes: CudaLaunchAttributes = field(default_factory=CudaLaunchAttributes)
    schema: str = CUDA_LAUNCH_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.kernel) is not str
            or _KERNEL_NAME_RE.fullmatch(self.kernel) is None
        ):
            raise CudaLaunchError("CUDA launch kernel is not a canonical symbol")
        if type(self.grid) is not CudaDim3 or type(self.block) is not CudaDim3:
            raise CudaLaunchError("CUDA launch grid/block have the wrong type")
        if self.block.volume > _MAX_THREADS_PER_BLOCK:
            raise CudaLaunchError("CUDA block exceeds the thread-count policy")
        if self.cluster is not None:
            if type(self.cluster) is not CudaDim3:
                raise CudaLaunchError("CUDA cluster has the wrong type")
            if self.cluster.volume > _MAX_CLUSTER_BLOCKS:
                raise CudaLaunchError("CUDA cluster exceeds the block-count policy")
            if any(
                grid_axis % cluster_axis
                for grid_axis, cluster_axis in zip(
                    (self.grid.x, self.grid.y, self.grid.z),
                    (self.cluster.x, self.cluster.y, self.cluster.z),
                )
            ):
                raise CudaLaunchError("CUDA cluster dimensions must divide the grid")
        if (
            type(self.shared_mem_bytes) is not int
            or not 0
            <= self.shared_mem_bytes
            <= _MAX_DYNAMIC_SHARED_MEMORY_BYTES
        ):
            raise CudaLaunchError("CUDA dynamic shared memory is outside policy")
        if type(self.attributes) is not CudaLaunchAttributes:
            raise CudaLaunchError("CUDA launch attributes have the wrong type")
        if self.cluster is None and (
            self.attributes.cluster_scheduling is not None
            or self.attributes.portable_cluster_mode is not None
        ):
            raise CudaLaunchError(
                "CUDA cluster attributes require an explicit cluster dimension"
            )
        if type(self.schema) is not str or self.schema != CUDA_LAUNCH_SCHEMA:
            raise CudaLaunchError("CUDA launch schema mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "attributes": self.attributes.to_dict(),
            "block": self.block.to_dict(),
            "cluster": None if self.cluster is None else self.cluster.to_dict(),
            "grid": self.grid.to_dict(),
            "kernel": self.kernel,
            "schema": self.schema,
            "shared_mem_bytes": self.shared_mem_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaLaunchSpec":
        expected = {
            "attributes",
            "block",
            "cluster",
            "grid",
            "kernel",
            "schema",
            "shared_mem_bytes",
        }
        if type(value) is not dict or set(value) != expected:
            raise CudaLaunchError("CUDA launch fields mismatch")
        raw_cluster = value["cluster"]
        return cls(
            kernel=value["kernel"],  # type: ignore[arg-type]
            grid=CudaDim3.from_dict(value["grid"]),
            block=CudaDim3.from_dict(value["block"]),
            cluster=(
                None
                if raw_cluster is None
                else CudaDim3.from_dict(raw_cluster)
            ),
            shared_mem_bytes=value["shared_mem_bytes"],  # type: ignore[arg-type]
            attributes=CudaLaunchAttributes.from_dict(value["attributes"]),
            schema=value["schema"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaScalar:
    """One explicitly typed scalar that is already owned by the validator."""

    scalar_type: CudaScalarType
    value: bool | int | float

    def __post_init__(self) -> None:
        if type(self.scalar_type) is not CudaScalarType:
            raise CudaLaunchError("CUDA scalar type is not supported")
        _pack_scalar(self.scalar_type, self.value)


@dataclass(frozen=True)
class CudaPointer:
    """A materialized 64-bit CUDA virtual address; zero is a valid null."""

    address: int

    def __post_init__(self) -> None:
        if type(self.address) is not int or not 0 <= self.address <= _UINT64_MAX:
            raise CudaLaunchError("CUDA pointer is outside uint64 policy")


@dataclass(frozen=True)
class CudaOpaqueBytes:
    """An exact by-value parameter (for driver-owned opaque structs)."""

    value: bytes

    def __post_init__(self) -> None:
        if (
            type(self.value) is not bytes
            or not 1 <= len(self.value) <= _MAX_OPAQUE_PARAMETER_BYTES
        ):
            raise CudaLaunchError("opaque CUDA parameter bytes are outside policy")


CudaMaterializedParameter: TypeAlias = CudaScalar | CudaPointer | CudaOpaqueBytes


_INTEGER_SCALARS: dict[CudaScalarType, tuple[str, int, int]] = {
    CudaScalarType.I8: ("b", -(1 << 7), (1 << 7) - 1),
    CudaScalarType.U8: ("B", 0, (1 << 8) - 1),
    CudaScalarType.I16: ("h", -(1 << 15), (1 << 15) - 1),
    CudaScalarType.U16: ("H", 0, (1 << 16) - 1),
    CudaScalarType.I32: ("i", _INT32_MIN, _INT32_MAX),
    CudaScalarType.U32: ("I", 0, _UINT32_MAX),
    CudaScalarType.I64: ("q", -(1 << 63), (1 << 63) - 1),
    CudaScalarType.U64: ("Q", 0, _UINT64_MAX),
}

_FLOAT_SCALARS: dict[CudaScalarType, str] = {
    CudaScalarType.F16: "e",
    CudaScalarType.F32: "f",
    CudaScalarType.F64: "d",
}


def _pack_scalar(scalar_type: CudaScalarType, value: object) -> bytes:
    if scalar_type is CudaScalarType.BOOL:
        if type(value) is not bool:
            raise CudaLaunchError("CUDA bool scalar requires an exact bool")
        return struct.pack("<?", value)

    integer = _INTEGER_SCALARS.get(scalar_type)
    if integer is not None:
        code, minimum, maximum = integer
        if type(value) is not int or not minimum <= value <= maximum:
            raise CudaLaunchError(
                f"CUDA {scalar_type.value} scalar is outside its exact range"
            )
        return struct.pack("<" + code, value)

    float_code = _FLOAT_SCALARS.get(scalar_type)
    if float_code is not None:
        if type(value) is not float or not math.isfinite(value):
            raise CudaLaunchError(
                f"CUDA {scalar_type.value} scalar requires a finite exact float"
            )
        try:
            return struct.pack("<" + float_code, value)
        except (OverflowError, struct.error):
            raise CudaLaunchError(
                f"CUDA {scalar_type.value} scalar is outside its exact range"
            ) from None

    raise CudaLaunchError("CUDA scalar type is not supported")


def _pack_materialized_parameter(value: object) -> bytes:
    # Exact types are deliberate: a miner-controlled subclass cannot add a
    # conversion hook or override behavior at this boundary.
    if type(value) is CudaScalar:
        return _pack_scalar(value.scalar_type, value.value)
    if type(value) is CudaPointer:
        return struct.pack("<Q", value.address)
    if type(value) is CudaOpaqueBytes:
        return value.value
    raise CudaLaunchError(
        "CUDA parameter is not a validator-materialized scalar, pointer, or bytes"
    )


def pack_kernel_parameters(
    abi: CudaKernelABI,
    parameters: tuple[CudaMaterializedParameter, ...],
) -> bytes:
    """Pack values at exactly the offsets and sizes observed by the driver."""

    if type(abi) is not CudaKernelABI:
        raise CudaLaunchError("CUDA kernel ABI has the wrong type")
    if type(parameters) is not tuple:
        raise CudaLaunchError("CUDA materialized parameters must be a tuple")
    if len(parameters) != len(abi.parameters):
        raise CudaLaunchError("CUDA materialized parameter count differs from ABI")

    packed = bytearray(abi.parameter_buffer_size)
    for formal, materialized in zip(abi.parameters, parameters):
        raw = _pack_materialized_parameter(materialized)
        if len(raw) != formal.size:
            raise CudaLaunchError(
                f"CUDA parameter {formal.index} has {len(raw)} bytes; "
                f"sealed ABI requires {formal.size}"
            )
        packed[formal.offset : formal.offset + formal.size] = raw
    return bytes(packed)


def _driver_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise CudaLaunchError(f"CUDA driver returned a malformed {field_name}")
    try:
        return operator.index(value)
    except TypeError:
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, bool):
            raise CudaLaunchError(f"CUDA driver returned a malformed {field_name}")
        try:
            return operator.index(enum_value)
        except TypeError:
            raise CudaLaunchError(
                f"CUDA driver returned a malformed {field_name}"
            ) from None


def _driver_member(owner: object, name: str, *, field_name: str) -> object:
    try:
        return getattr(owner, name)
    except AttributeError:
        raise CudaLaunchError(f"CUDA driver lacks {field_name} {name}") from None


def _driver_type(driver: object, name: str) -> type:
    value = _driver_member(driver, name, field_name="type")
    if not isinstance(value, type):
        raise CudaLaunchError(f"CUDA driver {name} is not a type")
    return value


def _new_attribute(
    driver: object,
    *,
    attribute_id: str,
    value_field: str,
    value: object,
) -> object:
    attr_type = _driver_type(driver, "CUlaunchAttribute")
    value_type = _driver_type(driver, "CUlaunchAttributeValue")
    attr_ids = _driver_member(
        driver, "CUlaunchAttributeID", field_name="enum type"
    )
    attr = attr_type()
    union = value_type()
    try:
        attr.id = _driver_member(
            attr_ids, attribute_id, field_name="launch attribute"
        )
        setattr(union, value_field, value)
        attr.value = union
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise CudaLaunchError(
            f"CUDA launch attribute {attribute_id} could not be materialized: {exc}"
        ) from None
    return attr


def _cluster_attribute(driver: object, cluster: CudaDim3) -> object:
    value_type = _driver_type(driver, "CUlaunchAttributeValue")
    attr_type = _driver_type(driver, "CUlaunchAttribute")
    attr_ids = _driver_member(
        driver, "CUlaunchAttributeID", field_name="enum type"
    )
    union = value_type()
    try:
        cluster_value = union.clusterDim
        cluster_value.x = cluster.x
        cluster_value.y = cluster.y
        cluster_value.z = cluster.z
        # cuda.bindings extension fields can return copies.  Use the setter to
        # commit the modified anonymous struct, then commit the union as well.
        union.clusterDim = cluster_value
        attr = attr_type()
        attr.id = _driver_member(
            attr_ids,
            "CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION",
            field_name="launch attribute",
        )
        attr.value = union
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise CudaLaunchError(
            f"CUDA cluster launch attribute could not be materialized: {exc}"
        ) from None
    return attr


def _enum_attribute(
    driver: object,
    *,
    enum_type_name: str,
    enum_member_name: str,
    attribute_id: str,
    value_field: str,
) -> object:
    enum_type = _driver_member(driver, enum_type_name, field_name="enum type")
    enum_value = _driver_member(
        enum_type, enum_member_name, field_name=enum_type_name
    )
    return _new_attribute(
        driver,
        attribute_id=attribute_id,
        value_field=value_field,
        value=enum_value,
    )


_CLUSTER_SCHEDULING_DRIVER_NAMES = {
    CudaClusterSchedulingPolicy.DEFAULT: "CU_CLUSTER_SCHEDULING_POLICY_DEFAULT",
    CudaClusterSchedulingPolicy.SPREAD: "CU_CLUSTER_SCHEDULING_POLICY_SPREAD",
    CudaClusterSchedulingPolicy.LOAD_BALANCING: (
        "CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING"
    ),
}

_PORTABLE_CLUSTER_DRIVER_NAMES = {
    CudaPortableClusterMode.DEFAULT: "CU_LAUNCH_PORTABLE_CLUSTER_MODE_DEFAULT",
    CudaPortableClusterMode.REQUIRE_PORTABLE: (
        "CU_LAUNCH_PORTABLE_CLUSTER_MODE_REQUIRE_PORTABLE"
    ),
    CudaPortableClusterMode.ALLOW_NON_PORTABLE: (
        "CU_LAUNCH_PORTABLE_CLUSTER_MODE_ALLOW_NON_PORTABLE"
    ),
}

_SHARED_MEMORY_DRIVER_NAMES = {
    CudaSharedMemoryMode.DEFAULT: "CU_SHARED_MEMORY_MODE_DEFAULT",
    CudaSharedMemoryMode.REQUIRE_PORTABLE: "CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE",
    CudaSharedMemoryMode.ALLOW_NON_PORTABLE: (
        "CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE"
    ),
}


def _build_launch_config(
    driver: object,
    spec: CudaLaunchSpec,
    *,
    stream: object | None,
) -> object:
    config_type = _driver_type(driver, "CUlaunchConfig")
    stream_type = _driver_type(driver, "CUstream")
    if stream is None:
        try:
            captured_stream = stream_type(0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CudaLaunchError(
                f"CUDA default stream could not be materialized: {exc}"
            ) from None
    else:
        if type(stream) is not stream_type:
            raise CudaLaunchError(
                "CUDA stream must be an exact handle from the captured driver"
            )
        captured_stream = stream

    attrs: list[object] = []
    if spec.cluster is not None:
        attrs.append(_cluster_attribute(driver, spec.cluster))
    launch_attrs = spec.attributes
    if launch_attrs.cluster_scheduling is not None:
        attrs.append(
            _enum_attribute(
                driver,
                enum_type_name="CUclusterSchedulingPolicy",
                enum_member_name=_CLUSTER_SCHEDULING_DRIVER_NAMES[
                    launch_attrs.cluster_scheduling
                ],
                attribute_id=(
                    "CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE"
                ),
                value_field="clusterSchedulingPolicyPreference",
            )
        )
    if launch_attrs.cooperative:
        attrs.append(
            _new_attribute(
                driver,
                attribute_id="CU_LAUNCH_ATTRIBUTE_COOPERATIVE",
                value_field="cooperative",
                value=1,
            )
        )
    if launch_attrs.programmatic_stream_serialization:
        attrs.append(
            _new_attribute(
                driver,
                attribute_id=(
                    "CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION"
                ),
                value_field="programmaticStreamSerializationAllowed",
                value=1,
            )
        )
    if launch_attrs.priority is not None:
        attrs.append(
            _new_attribute(
                driver,
                attribute_id="CU_LAUNCH_ATTRIBUTE_PRIORITY",
                value_field="priority",
                value=launch_attrs.priority,
            )
        )
    if launch_attrs.portable_cluster_mode is not None:
        attrs.append(
            _enum_attribute(
                driver,
                enum_type_name="CUlaunchAttributePortableClusterMode",
                enum_member_name=_PORTABLE_CLUSTER_DRIVER_NAMES[
                    launch_attrs.portable_cluster_mode
                ],
                attribute_id="CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE",
                value_field="portableClusterSizeMode",
            )
        )
    if launch_attrs.shared_memory_mode is not None:
        attrs.append(
            _enum_attribute(
                driver,
                enum_type_name="CUsharedMemoryMode",
                enum_member_name=_SHARED_MEMORY_DRIVER_NAMES[
                    launch_attrs.shared_memory_mode
                ],
                attribute_id="CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE",
                value_field="sharedMemoryMode",
            )
        )

    try:
        config = config_type()
        config.gridDimX = spec.grid.x
        config.gridDimY = spec.grid.y
        config.gridDimZ = spec.grid.z
        config.blockDimX = spec.block.x
        config.blockDimY = spec.block.y
        config.blockDimZ = spec.block.z
        config.sharedMemBytes = spec.shared_mem_bytes
        config.hStream = captured_stream
        # cuda.bindings requires assigning the whole list through the setter.
        config.attrs = attrs
        config.numAttrs = len(attrs)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise CudaLaunchError(
            f"CUDA launch configuration could not be materialized: {exc}"
        ) from None
    return config


def _success_code(driver: object) -> int:
    result_enum = _driver_member(driver, "CUresult", field_name="enum type")
    return _driver_integer(
        _driver_member(result_enum, "CUDA_SUCCESS", field_name="status"),
        field_name="success status",
    )


def _set_dynamic_shared_memory_opt_in(
    driver: object,
    function: object,
    shared_mem_bytes: int,
) -> None:
    """Apply the sealed launch requirement to the admitted ``CUfunction``.

    The requested maximum is not a second miner-controlled field: it is exactly
    the already-validated dynamic shared-memory byte count in ``CudaLaunchSpec``.
    Zero needs no opt-in.  Resolving the setter and enum from the retained
    library's captured driver prevents context/handle substitution.
    """

    if shared_mem_bytes == 0:
        return
    attributes = _driver_member(
        driver, "CUfunction_attribute", field_name="enum type"
    )
    attribute = _driver_member(
        attributes,
        "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
        field_name="function attribute",
    )
    setter = _driver_member(driver, "cuFuncSetAttribute", field_name="callable")
    if not callable(setter):
        raise CudaLaunchError("CUDA driver cuFuncSetAttribute is not callable")
    try:
        result = setter(function, attribute, shared_mem_bytes)
    except Exception as exc:  # noqa: BLE001 - normalize optional binding
        raise CudaLaunchError(
            "CUDA cuFuncSetAttribute binding raised "
            f"{type(exc).__name__}: {exc}"
        ) from None
    if not isinstance(result, tuple) or len(result) != 1:
        raise CudaLaunchError(
            "CUDA cuFuncSetAttribute returned a malformed result"
        )
    status = _driver_integer(result[0], field_name="function attribute status")
    if status != _success_code(driver):
        raise CudaLaunchError(
            "CUDA cuFuncSetAttribute for dynamic shared memory failed with "
            f"status {status}"
        )


def _parameter_buffer_extra(
    driver: object, raw: bytes
) -> tuple[int, tuple[object, ...]]:
    if not raw:
        return 0, ()

    tokens: list[int] = []
    for name, expected in (
        ("CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT", 1),
        ("CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT", 2),
        ("CU_LAUNCH_PARAM_END_AS_INT", 0),
    ):
        token = _driver_integer(
            _driver_member(driver, name, field_name="launch parameter token"),
            field_name=name,
        )
        if token != expected:
            raise CudaLaunchError(f"CUDA driver {name} differs from the CUDA ABI")
        tokens.append(token)

    parameter_buffer = ctypes.create_string_buffer(raw, len(raw))
    parameter_size = ctypes.c_size_t(len(raw))
    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(tokens[0]),
        ctypes.cast(parameter_buffer, ctypes.c_void_p),
        ctypes.c_void_p(tokens[1]),
        ctypes.cast(ctypes.pointer(parameter_size), ctypes.c_void_p),
        ctypes.c_void_p(tokens[2]),
    )
    # Keep every pointee alive until cuLaunchKernelEx has copied the values.
    keepalive: tuple[object, ...] = (parameter_buffer, parameter_size, extra)
    return ctypes.addressof(extra), keepalive


def launch_cuda_kernel(
    library: CudaCubinLibrary,
    spec: CudaLaunchSpec,
    parameters: tuple[CudaMaterializedParameter, ...],
    *,
    stream: object | None,
) -> None:
    """Launch sealed device code using only the declarative CUDA Driver waist.

    ``parameters`` and ``stream`` must already have been materialized by
    validator-owned binding code.  This function deliberately performs no
    tensor lookup, pointer discovery, expression evaluation, or callback.
    """

    if type(library) is not CudaCubinLibrary:
        raise CudaLaunchError("CUDA launch requires an exact sealed CUBIN library")
    if type(spec) is not CudaLaunchSpec:
        raise CudaLaunchError("CUDA launch spec has the wrong type")

    # The CUBIN authority intentionally does not expose its captured driver.
    # This sibling trusted module uses the private handle so a caller cannot
    # substitute a different binding/context between ABI admission and launch.
    driver = object.__getattribute__(library, "_driver")
    lock = object.__getattribute__(library, "_lock")
    with lock:
        if object.__getattribute__(library, "_closed"):
            raise CudaLaunchError("CUDA CUBIN library is closed")
        abi = library.abi.by_name.get(spec.kernel)
        if abi is None:
            raise CudaLaunchError(
                f"CUDA CUBIN has no sealed kernel {spec.kernel!r}"
            )
        parameter_buffer = pack_kernel_parameters(abi, parameters)
        config = _build_launch_config(driver, spec, stream=stream)
        function = library.function(spec.kernel)
        extra_pointer, keepalive = _parameter_buffer_extra(
            driver, parameter_buffer
        )
        _set_dynamic_shared_memory_opt_in(
            driver, function, spec.shared_mem_bytes
        )

        launch = _driver_member(
            driver, "cuLaunchKernelEx", field_name="callable"
        )
        if not callable(launch):
            raise CudaLaunchError("CUDA driver cuLaunchKernelEx is not callable")
        try:
            result = launch(config, function, 0, extra_pointer)
        except Exception as exc:  # noqa: BLE001 - normalize optional binding
            raise CudaLaunchError(
                f"CUDA cuLaunchKernelEx binding raised {type(exc).__name__}: {exc}"
            ) from None
        finally:
            # A named reference makes the lifetime requirement explicit.
            _ = keepalive
        if not isinstance(result, tuple) or len(result) != 1:
            raise CudaLaunchError("CUDA cuLaunchKernelEx returned a malformed result")
        status = _driver_integer(result[0], field_name="launch status")
        if status != _success_code(driver):
            raise CudaLaunchError(
                f"CUDA cuLaunchKernelEx failed with status {status}"
            )


__all__ = [
    "CUDA_LAUNCH_SCHEMA",
    "CudaClusterSchedulingPolicy",
    "CudaDim3",
    "CudaLaunchAttributes",
    "CudaLaunchError",
    "CudaLaunchSpec",
    "CudaMaterializedParameter",
    "CudaOpaqueBytes",
    "CudaPointer",
    "CudaPortableClusterMode",
    "CudaScalar",
    "CudaScalarType",
    "CudaSharedMemoryMode",
    "launch_cuda_kernel",
    "pack_kernel_parameters",
]
