"""Closed validator-owned materializers for device-only CUDA artifacts.

This module is deliberately CUDA- and torch-free at import time.  Miner data can
select only the versioned primitives below; it cannot provide a callback, type
converter, or arbitrary expression.  Live tensor/TMA/group authority is captured
by :func:`make_cuda_primitive_registry` inside the isolated engine worker.
"""

from __future__ import annotations

import math
import operator
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from optima.cuda_launch import (
    CudaMaterializedParameter,
    CudaOpaqueBytes,
    CudaPointer,
    CudaScalar,
    CudaScalarType,
)


CUDA_EXPRESSION_SCHEMA = "optima.cuda-expression-dag.v1"
CUDA_TMA_DESCRIPTOR_CAPABILITY = "cuda.tma_descriptor.v1"
CUTLASS_FAST_DIVMOD_CAPABILITY = "cutlass.fast_divmod.i32.v1"
CUTE_FAST_DIVMOD_CAPABILITY = "cute.fast_divmod.i32.v1"
CUDA_PACKED_STRUCT_CAPABILITY = "cuda.packed_struct.v1"
CUDA_CHECKED_EXPRESSION_CAPABILITY = "cuda.checked_expression.v1"
GROUP_NATIVE_HANDLE_CAPABILITY = "group.native_handle.v1"
GROUP_PEER_POINTER_TABLE_CAPABILITY = "group.peer_ptr_table.v1"

_MAX_EXPRESSION_NODES = 128
_MAX_EXPRESSION_OPERANDS = 8
_MAX_BINDINGS = 64
_MAX_TENSOR_RANK = 16
_MAX_PACKED_FIELDS = 64
_MAX_PARAMETER_BYTES = 4_096
# Static bytes are already covered by the exact per-parameter ABI width and the
# 4 KiB CUDA parameter cap.  A smaller CuTe/CUTLASS-specific allowance would
# make the generic waist reactive to each new compile-time struct layout without
# removing any device-code authority from a miner who already supplies a CUBIN.
_MAX_STATIC_LITERAL_BYTES = _MAX_PARAMETER_BYTES
_MAX_STATIC_LITERAL_TOTAL_BYTES = _MAX_PARAMETER_BYTES
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT64_MAX = (1 << 64) - 1

_SCALAR_WIDTH = MappingProxyType(
    {
        "bool": 1,
        "i8": 1,
        "u8": 1,
        "i16": 2,
        "u16": 2,
        "i32": 4,
        "u32": 4,
        "i64": 8,
        "u64": 8,
        "f16": 2,
        "f32": 4,
        "f64": 8,
    }
)
_SCALAR_STRUCT = MappingProxyType(
    {
        "bool": "?",
        "i8": "b",
        "u8": "B",
        "i16": "h",
        "u16": "H",
        "i32": "i",
        "u32": "I",
        "i64": "q",
        "u64": "Q",
        "f16": "e",
        "f32": "f",
        "f64": "d",
    }
)
_UNARY_OPS = frozenset({"neg", "not"})
_BINARY_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "floor_div",
        "ceil_div",
        "mod",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
    }
)
_NARY_OPS = frozenset({"min", "max"})
_SOURCE_OPS = frozenset(
    {
        "binding_scalar",
        "tensor_dim",
        "tensor_stride",
        "tensor_numel",
        "tensor_storage_offset",
        "tensor_element_size",
    }
)


class CudaMaterializeError(RuntimeError):
    """A device declaration or live validator resource cannot be materialized."""


def _binding_index(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value < _MAX_BINDINGS:
        raise CudaMaterializeError(f"{field} is outside the binding-index policy")
    return value


def _checked_number(value: object, *, field: str) -> bool | int | float:
    if type(value) is bool:
        return value
    if type(value) is int:
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise CudaMaterializeError(f"{field} is outside signed 64-bit policy")
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise CudaMaterializeError(f"{field} is not a finite exact scalar")


def _number_to_wire(value: bool | int | float | None) -> object:
    """Return a canonical-json-safe exact scalar representation.

    Artifact identity intentionally rejects native JSON floats.  IEEE-754
    constants therefore travel as their canonical Python hexadecimal spelling,
    which preserves signed zero and every finite binary64 bit pattern without
    admitting alternate textual representations into the sealed identity.
    """

    if type(value) is float:
        _checked_number(value, field="CUDA expression constant")
        return {"f64_hex": value.hex()}
    return value


def _number_from_wire(value: object) -> bool | int | float | None:
    if type(value) is dict:
        if set(value) != {"f64_hex"} or type(value["f64_hex"]) is not str:
            raise CudaMaterializeError("CUDA float constant tag is malformed")
        encoded = value["f64_hex"]
        try:
            decoded = float.fromhex(encoded)
        except ValueError:
            raise CudaMaterializeError("CUDA float constant is not hexadecimal") from None
        if not math.isfinite(decoded) or decoded.hex() != encoded:
            raise CudaMaterializeError(
                "CUDA float constant is non-finite or non-canonical"
            )
        return decoded
    if type(value) is float:
        raise CudaMaterializeError("CUDA float constant must use the f64_hex tag")
    if value is None or type(value) in {bool, int}:
        return value
    raise CudaMaterializeError("CUDA expression value is malformed")


@dataclass(frozen=True)
class CudaTensorFacts:
    """Bounded live facts derived from one validator-owned tensor/view."""

    address: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    element_size: int
    storage_offset: int = 0

    def __post_init__(self) -> None:
        if type(self.address) is not int or not 0 <= self.address <= _UINT64_MAX:
            raise CudaMaterializeError("tensor address is outside uint64 policy")
        if (
            type(self.shape) is not tuple
            or not 1 <= len(self.shape) <= _MAX_TENSOR_RANK
            or any(type(value) is not int or value < 0 for value in self.shape)
        ):
            raise CudaMaterializeError("tensor shape is outside policy")
        if (
            type(self.stride) is not tuple
            or len(self.stride) != len(self.shape)
            or any(type(value) is not int for value in self.stride)
        ):
            raise CudaMaterializeError("tensor stride is outside policy")
        if type(self.element_size) is not int or not 1 <= self.element_size <= 32:
            raise CudaMaterializeError("tensor element size is outside policy")
        if type(self.storage_offset) is not int or self.storage_offset < 0:
            raise CudaMaterializeError("tensor storage offset is outside policy")

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class CudaExpressionNode:
    """One topologically ordered node in the closed expression DAG."""

    op: str
    operands: tuple[int, ...] = ()
    value: bool | int | float | None = None
    binding: int | None = None
    axis: int | None = None

    def __post_init__(self) -> None:
        if type(self.op) is not str:
            raise CudaMaterializeError("CUDA expression op must be a string")
        if type(self.operands) is not tuple or len(self.operands) > _MAX_EXPRESSION_OPERANDS:
            raise CudaMaterializeError("CUDA expression operand set is outside policy")
        if any(type(index) is not int or index < 0 for index in self.operands):
            raise CudaMaterializeError("CUDA expression operand index is malformed")
        if self.op == "const":
            _checked_number(self.value, field="CUDA expression constant")
            if self.operands or self.binding is not None or self.axis is not None:
                raise CudaMaterializeError("CUDA const expression has extra fields")
        elif self.op in _SOURCE_OPS:
            _binding_index(self.binding, field="CUDA expression binding")
            if self.operands or self.value is not None:
                raise CudaMaterializeError("CUDA source expression has extra fields")
            axis_required = self.op in {"tensor_dim", "tensor_stride"}
            if axis_required:
                if type(self.axis) is not int or not 0 <= self.axis < _MAX_TENSOR_RANK:
                    raise CudaMaterializeError("CUDA tensor expression axis is outside policy")
            elif self.axis is not None:
                raise CudaMaterializeError("CUDA source expression has an unexpected axis")
        else:
            if self.value is not None or self.binding is not None or self.axis is not None:
                raise CudaMaterializeError("CUDA operation expression has source fields")
            expected = 1 if self.op in _UNARY_OPS else 2 if self.op in _BINARY_OPS else None
            if self.op == "select":
                expected = 3
            if expected is not None and len(self.operands) != expected:
                raise CudaMaterializeError(
                    f"CUDA expression {self.op!r} requires {expected} operands"
                )
            if self.op in _NARY_OPS and not 1 <= len(self.operands) <= _MAX_EXPRESSION_OPERANDS:
                raise CudaMaterializeError(
                    f"CUDA expression {self.op!r} requires bounded operands"
                )
            if self.op not in _UNARY_OPS | _BINARY_OPS | _NARY_OPS | {"select"}:
                raise CudaMaterializeError(f"unsupported CUDA expression op {self.op!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "binding": self.binding,
            "op": self.op,
            "operands": list(self.operands),
            "value": _number_to_wire(self.value),
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaExpressionNode":
        allowed = {
            "axis",
            "binding",
            "op",
            "operands",
            "value",
        }
        if (
            type(value) is not dict
            or set(value) - allowed
            or "op" not in value
        ):
            raise CudaMaterializeError("CUDA expression node fields mismatch")
        operands = value.get("operands", [])
        if type(operands) is not list:
            raise CudaMaterializeError("CUDA expression operands must be a list")
        return cls(
            op=value["op"],  # type: ignore[arg-type]
            operands=tuple(operands),  # type: ignore[arg-type]
            value=_number_from_wire(value.get("value")),
            binding=value.get("binding"),  # type: ignore[arg-type]
            axis=value.get("axis"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaCheckedExpression:
    """A bounded acyclic scalar program evaluated only by validator code."""

    nodes: tuple[CudaExpressionNode, ...]
    result: int
    schema: str = CUDA_EXPRESSION_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.nodes) is not tuple
            or not 1 <= len(self.nodes) <= _MAX_EXPRESSION_NODES
            or not all(type(node) is CudaExpressionNode for node in self.nodes)
        ):
            raise CudaMaterializeError("CUDA expression node inventory is outside policy")
        for ordinal, node in enumerate(self.nodes):
            if any(index >= ordinal for index in node.operands):
                raise CudaMaterializeError(
                    "CUDA expression operands must reference prior nodes"
                )
        if type(self.result) is not int or not 0 <= self.result < len(self.nodes):
            raise CudaMaterializeError("CUDA expression result index is outside policy")
        if self.schema != CUDA_EXPRESSION_SCHEMA:
            raise CudaMaterializeError("CUDA expression schema mismatch")

    @property
    def binding_references(self) -> tuple[tuple[int, str], ...]:
        return tuple(
            (node.binding, node.op)  # type: ignore[arg-type]
            for node in self.nodes
            if node.binding is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "result": self.result,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaCheckedExpression":
        if type(value) is not dict or set(value) != {"nodes", "result", "schema"}:
            raise CudaMaterializeError("CUDA expression fields mismatch")
        nodes = value["nodes"]
        if type(nodes) is not list:
            raise CudaMaterializeError("CUDA expression nodes must be a list")
        return cls(
            nodes=tuple(CudaExpressionNode.from_dict(node) for node in nodes),
            result=value["result"],  # type: ignore[arg-type]
            schema=value["schema"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaPackedField:
    """One non-overlapping validator-materialized packed-struct field.

    ``tensor_binding`` is deliberately narrower than the top-level ``pointer``
    primitive: it embeds only the exact base address of a sealed tensor binding.
    There is no integer-address input, pointer expression, offset, or candidate
    conversion hook.  Dynamic memref shape/stride members remain ordinary
    checked scalar expressions in adjacent declared fields.
    """

    offset: int
    scalar_type: str | None = None
    expression: CudaCheckedExpression | None = None
    data_hex: str | None = None
    tensor_binding: int | None = None

    def __post_init__(self) -> None:
        if type(self.offset) is not int or not 0 <= self.offset < _MAX_PARAMETER_BYTES:
            raise CudaMaterializeError("packed field offset is outside policy")
        scalar = self.scalar_type is not None or self.expression is not None
        literal = self.data_hex is not None
        tensor_pointer = self.tensor_binding is not None
        if sum((scalar, literal, tensor_pointer)) != 1:
            raise CudaMaterializeError(
                "packed field must declare exactly one scalar expression, literal, "
                "or tensor pointer"
            )
        if scalar:
            if (
                self.scalar_type not in _SCALAR_WIDTH
                or type(self.expression) is not CudaCheckedExpression
            ):
                raise CudaMaterializeError("packed scalar field is malformed")
        elif tensor_pointer:
            _binding_index(
                self.tensor_binding, field="packed tensor-pointer binding"
            )
        else:
            if (
                type(self.data_hex) is not str
                or len(self.data_hex) == 0
                or len(self.data_hex) % 2
                or len(self.data_hex) > 2 * _MAX_STATIC_LITERAL_BYTES
            ):
                raise CudaMaterializeError("packed literal field is malformed")
            try:
                bytes.fromhex(self.data_hex)
            except ValueError:
                raise CudaMaterializeError("packed literal field is not canonical hex") from None
            if self.data_hex != self.data_hex.lower():
                raise CudaMaterializeError("packed literal field is not lowercase hex")

    @property
    def size(self) -> int:
        if self.scalar_type is not None:
            return _SCALAR_WIDTH[self.scalar_type]
        if self.tensor_binding is not None:
            return 8
        assert self.data_hex is not None
        return len(self.data_hex) // 2

    def to_dict(self) -> dict[str, object]:
        return {
            "data_hex": self.data_hex,
            "expression": None if self.expression is None else self.expression.to_dict(),
            "offset": self.offset,
            "scalar_type": self.scalar_type,
            "tensor_binding": self.tensor_binding,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaPackedField":
        allowed = {
            "data_hex",
            "expression",
            "offset",
            "scalar_type",
            "tensor_binding",
        }
        if (
            type(value) is not dict
            or set(value) - allowed
            or "offset" not in value
        ):
            raise CudaMaterializeError("packed field fields mismatch")
        expression = value.get("expression")
        return cls(
            offset=value["offset"],  # type: ignore[arg-type]
            scalar_type=value.get("scalar_type"),  # type: ignore[arg-type]
            expression=(
                None if expression is None else CudaCheckedExpression.from_dict(expression)
            ),
            data_hex=value.get("data_hex"),  # type: ignore[arg-type]
            tensor_binding=value.get("tensor_binding"),  # type: ignore[arg-type]
        )


_TMA_ELEMENT_TYPES = frozenset(
    {
        "u8",
        "u16",
        "u32",
        "i32",
        "i64",
        "f16",
        "f32",
        "f64",
        "bf16",
        "tf32",
        "tf32_32b",
        "f8e4m3",
        "f8e5m2",
    }
)
_TMA_INTERLEAVES = frozenset({"none", "interleave_16b", "interleave_32b"})
_TMA_SWIZZLES = frozenset(
    {"none", "swizzle_32b", "swizzle_64b", "swizzle_128b", "swizzle_128b_atom_32b"}
)
_TMA_L2 = frozenset({"none", "l2_64b", "l2_128b", "l2_256b"})
_TMA_OOB = frozenset({"none", "zero", "nan_request_zero_fma"})
_TMA_ELEMENT_BYTES = MappingProxyType(
    {
        "u8": 1,
        "u16": 2,
        "u32": 4,
        "i32": 4,
        "i64": 8,
        "f16": 2,
        "f32": 4,
        "f64": 8,
        "bf16": 2,
        "tf32": 4,
        "tf32_32b": 4,
        "f8e4m3": 1,
        "f8e5m2": 1,
    }
)
_TMA_FLOAT_ELEMENTS = frozenset(
    {"f16", "f32", "f64", "bf16", "tf32", "tf32_32b"}
)
_TMA_SWIZZLE_BYTES = MappingProxyType(
    {
        "swizzle_32b": 32,
        "swizzle_64b": 64,
        "swizzle_128b": 128,
        "swizzle_128b_atom_32b": 128,
    }
)


@dataclass(frozen=True)
class CudaTmaDescriptorPlan:
    """Complete data-only input to CUDA's versioned tensor-map encoder."""

    binding: int
    element_type: str
    global_dims: tuple[CudaCheckedExpression, ...]
    global_strides: tuple[CudaCheckedExpression, ...]
    box_dims: tuple[CudaCheckedExpression, ...]
    element_strides: tuple[CudaCheckedExpression, ...]
    interleave: str = "none"
    swizzle: str = "none"
    l2_promotion: str = "none"
    oob_fill: str = "none"

    def __post_init__(self) -> None:
        _binding_index(self.binding, field="TMA tensor binding")
        rank = len(self.global_dims)
        if not 1 <= rank <= 5:
            raise CudaMaterializeError("TMA rank is outside CUDA policy")
        if (
            type(self.global_dims) is not tuple
            or type(self.global_strides) is not tuple
            or type(self.box_dims) is not tuple
            or type(self.element_strides) is not tuple
            or len(self.global_strides) != max(0, rank - 1)
            or len(self.box_dims) != rank
            or len(self.element_strides) != rank
            or not all(
                type(expr) is CudaCheckedExpression
                for rows in (
                    self.global_dims,
                    self.global_strides,
                    self.box_dims,
                    self.element_strides,
                )
                for expr in rows
            )
        ):
            raise CudaMaterializeError("TMA dimension expressions are malformed")
        if self.element_type not in _TMA_ELEMENT_TYPES:
            raise CudaMaterializeError("TMA element type is unsupported")
        if self.interleave not in _TMA_INTERLEAVES:
            raise CudaMaterializeError("TMA interleave is unsupported")
        if self.swizzle not in _TMA_SWIZZLES:
            raise CudaMaterializeError("TMA swizzle is unsupported")
        if self.l2_promotion not in _TMA_L2:
            raise CudaMaterializeError("TMA L2 promotion is unsupported")
        if self.oob_fill not in _TMA_OOB:
            raise CudaMaterializeError("TMA OOB fill is unsupported")

    @property
    def expressions(self) -> tuple[CudaCheckedExpression, ...]:
        return self.global_dims + self.global_strides + self.box_dims + self.element_strides

    def to_dict(self) -> dict[str, object]:
        return {
            "binding": self.binding,
            "box_dims": [expr.to_dict() for expr in self.box_dims],
            "element_strides": [expr.to_dict() for expr in self.element_strides],
            "element_type": self.element_type,
            "global_dims": [expr.to_dict() for expr in self.global_dims],
            "global_strides": [expr.to_dict() for expr in self.global_strides],
            "interleave": self.interleave,
            "l2_promotion": self.l2_promotion,
            "oob_fill": self.oob_fill,
            "swizzle": self.swizzle,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaTmaDescriptorPlan":
        expected = {
            "binding",
            "box_dims",
            "element_strides",
            "element_type",
            "global_dims",
            "global_strides",
            "interleave",
            "l2_promotion",
            "oob_fill",
            "swizzle",
        }
        if type(value) is not dict or set(value) != expected:
            raise CudaMaterializeError("TMA descriptor fields mismatch")

        def expressions(name: str) -> tuple[CudaCheckedExpression, ...]:
            rows = value[name]
            if type(rows) is not list:
                raise CudaMaterializeError(f"TMA {name} must be a list")
            return tuple(CudaCheckedExpression.from_dict(row) for row in rows)

        return cls(
            binding=value["binding"],  # type: ignore[arg-type]
            element_type=value["element_type"],  # type: ignore[arg-type]
            global_dims=expressions("global_dims"),
            global_strides=expressions("global_strides"),
            box_dims=expressions("box_dims"),
            element_strides=expressions("element_strides"),
            interleave=value["interleave"],  # type: ignore[arg-type]
            swizzle=value["swizzle"],  # type: ignore[arg-type]
            l2_promotion=value["l2_promotion"],  # type: ignore[arg-type]
            oob_fill=value["oob_fill"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaParameterPlan:
    """One closed primitive that materializes one CUDA kernel parameter."""

    kind: str
    size: int
    binding: int | None = None
    scalar_type: str | None = None
    expression: CudaCheckedExpression | None = None
    fields: tuple[CudaPackedField, ...] = ()
    tma: CudaTmaDescriptorPlan | None = None
    group_projection: str | None = None

    def __post_init__(self) -> None:
        if type(self.size) is not int or not 1 <= self.size <= _MAX_PARAMETER_BYTES:
            raise CudaMaterializeError("CUDA parameter plan size is outside policy")
        if type(self.fields) is not tuple or len(self.fields) > _MAX_PACKED_FIELDS:
            raise CudaMaterializeError("CUDA packed field inventory is outside policy")
        if self.kind == "scalar":
            if (
                self.scalar_type not in _SCALAR_WIDTH
                or type(self.expression) is not CudaCheckedExpression
                or self.size != _SCALAR_WIDTH[self.scalar_type]
            ):
                raise CudaMaterializeError("CUDA scalar parameter plan is malformed")
        elif self.kind == "pointer":
            _binding_index(self.binding, field="CUDA pointer binding")
            if self.size != 8:
                raise CudaMaterializeError("CUDA pointer parameter must be 8 bytes")
        elif self.kind == "packed_struct":
            if not self.fields:
                raise CudaMaterializeError("CUDA packed struct must contain fields")
            # Static literal bytes are for bounded compile-time algebra atoms.
            # They have no binding source.  A packed live address can only be
            # projected from an admitted tensor through ``tensor_binding``;
            # arbitrary integer addresses and pointer arithmetic stay absent.
            if (
                sum(
                    field.size
                    for field in self.fields
                    if field.data_hex is not None
                )
                > _MAX_STATIC_LITERAL_TOTAL_BYTES
            ):
                raise CudaMaterializeError(
                    "CUDA packed struct static literals exceed policy"
                )
            occupied: set[int] = set()
            for field in self.fields:
                if type(field) is not CudaPackedField or field.offset + field.size > self.size:
                    raise CudaMaterializeError("CUDA packed struct field exceeds its size")
                span = set(range(field.offset, field.offset + field.size))
                if occupied & span:
                    raise CudaMaterializeError("CUDA packed struct fields overlap")
                occupied.update(span)
        elif self.kind == "tma_descriptor":
            if type(self.tma) is not CudaTmaDescriptorPlan or self.size != 128:
                raise CudaMaterializeError("CUDA TMA descriptor parameter is malformed")
        elif self.kind == "cutlass_fast_divmod_i32_v1":
            if type(self.expression) is not CudaCheckedExpression or self.size != 12:
                raise CudaMaterializeError("CUTLASS FastDivmod parameter is malformed")
        elif self.kind == "cute_fast_divmod_i32_v1":
            if type(self.expression) is not CudaCheckedExpression or self.size != 12:
                raise CudaMaterializeError("CuTe FastDivmod parameter is malformed")
        elif self.kind == "group_handle":
            _binding_index(self.binding, field="CUDA group binding")
            if self.group_projection not in {"native_handle", "peer_ptr_table"} or self.size != 8:
                raise CudaMaterializeError("CUDA group handle parameter is malformed")
        else:
            raise CudaMaterializeError(
                f"unsupported CUDA parameter primitive {self.kind!r}"
            )

        used = {
            "binding": self.kind in {"pointer", "group_handle"},
            "scalar_type": self.kind == "scalar",
            "expression": self.kind
            in {
                "scalar",
                "cutlass_fast_divmod_i32_v1",
                "cute_fast_divmod_i32_v1",
            },
            "fields": self.kind == "packed_struct",
            "tma": self.kind == "tma_descriptor",
            "group_projection": self.kind == "group_handle",
        }
        values = {
            "binding": self.binding is not None,
            "scalar_type": self.scalar_type is not None,
            "expression": self.expression is not None,
            "fields": bool(self.fields),
            "tma": self.tma is not None,
            "group_projection": self.group_projection is not None,
        }
        if used != values:
            raise CudaMaterializeError("CUDA parameter primitive has extraneous fields")

    @property
    def expressions(self) -> tuple[CudaCheckedExpression, ...]:
        rows: list[CudaCheckedExpression] = []
        if self.expression is not None:
            rows.append(self.expression)
        rows.extend(
            field.expression for field in self.fields if field.expression is not None
        )
        if self.tma is not None:
            rows.extend(self.tma.expressions)
        return tuple(rows)

    @property
    def required_capabilities(self) -> frozenset[str]:
        capabilities = {CUDA_CHECKED_EXPRESSION_CAPABILITY}
        if self.kind == "packed_struct":
            capabilities.add(CUDA_PACKED_STRUCT_CAPABILITY)
        elif self.kind == "tma_descriptor":
            capabilities.add(CUDA_TMA_DESCRIPTOR_CAPABILITY)
        elif self.kind == "cutlass_fast_divmod_i32_v1":
            capabilities.add(CUTLASS_FAST_DIVMOD_CAPABILITY)
        elif self.kind == "cute_fast_divmod_i32_v1":
            capabilities.add(CUTE_FAST_DIVMOD_CAPABILITY)
        elif self.kind == "group_handle":
            capabilities.add(
                {
                    "native_handle": GROUP_NATIVE_HANDLE_CAPABILITY,
                    "peer_ptr_table": GROUP_PEER_POINTER_TABLE_CAPABILITY,
                }[self.group_projection]  # type: ignore[index]
            )
        return frozenset(capabilities)

    def to_dict(self) -> dict[str, object]:
        return {
            "binding": self.binding,
            "expression": None if self.expression is None else self.expression.to_dict(),
            "fields": [field.to_dict() for field in self.fields],
            "group_projection": self.group_projection,
            "kind": self.kind,
            "scalar_type": self.scalar_type,
            "size": self.size,
            "tma": None if self.tma is None else self.tma.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaParameterPlan":
        allowed = {
            "binding",
            "expression",
            "fields",
            "group_projection",
            "kind",
            "scalar_type",
            "size",
            "tma",
        }
        if (
            type(value) is not dict
            or set(value) - allowed
            or not {"kind", "size"} <= set(value)
        ):
            raise CudaMaterializeError("CUDA parameter plan fields mismatch")
        expression = value.get("expression")
        fields = value.get("fields", [])
        tma = value.get("tma")
        if type(fields) is not list:
            raise CudaMaterializeError("CUDA parameter fields must be a list")
        return cls(
            kind=value["kind"],  # type: ignore[arg-type]
            size=value["size"],  # type: ignore[arg-type]
            binding=value.get("binding"),  # type: ignore[arg-type]
            scalar_type=value.get("scalar_type"),  # type: ignore[arg-type]
            expression=(
                None if expression is None else CudaCheckedExpression.from_dict(expression)
            ),
            fields=tuple(CudaPackedField.from_dict(field) for field in fields),
            tma=None if tma is None else CudaTmaDescriptorPlan.from_dict(tma),
            group_projection=value.get("group_projection"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaResolvedTmaDescriptor:
    address: int
    element_type: str
    global_dims: tuple[int, ...]
    global_strides: tuple[int, ...]
    box_dims: tuple[int, ...]
    element_strides: tuple[int, ...]
    interleave: str
    swizzle: str
    l2_promotion: str
    oob_fill: str


def _validate_resolved_tma_descriptor(
    descriptor: CudaResolvedTmaDescriptor,
) -> None:
    """Reject CUDA-invalid tiled tensor maps before crossing the driver API."""

    if type(descriptor) is not CudaResolvedTmaDescriptor:
        raise CudaMaterializeError("CUDA TMA validator received the wrong type")
    rank = len(descriptor.global_dims)
    if not 1 <= rank <= 5:
        raise CudaMaterializeError("TMA rank is outside CUDA limits")
    if descriptor.interleave != "none" and rank < 3:
        raise CudaMaterializeError("interleaved TMA rank must be at least three")

    address_alignment = 32 if descriptor.interleave == "interleave_32b" else 16
    if descriptor.address % address_alignment:
        raise CudaMaterializeError(
            f"TMA global address must be {address_alignment}-byte aligned"
        )
    if any(dimension > 1 << 32 for dimension in descriptor.global_dims):
        raise CudaMaterializeError("TMA global dimension exceeds CUDA's 2^32 limit")
    if any(stride >= 1 << 40 for stride in descriptor.global_strides):
        raise CudaMaterializeError("TMA global stride exceeds CUDA's 2^40 limit")
    stride_alignment = 32 if descriptor.interleave == "interleave_32b" else 16
    if any(stride % stride_alignment for stride in descriptor.global_strides):
        raise CudaMaterializeError(
            f"TMA global stride must be {stride_alignment}-byte aligned"
        )

    element_bytes = _TMA_ELEMENT_BYTES[descriptor.element_type]
    minimum_stride = descriptor.global_dims[0] * element_bytes
    for ordinal, stride in enumerate(descriptor.global_strides):
        if stride < minimum_stride:
            raise CudaMaterializeError(
                "TMA global strides do not describe nested tensor dimensions"
            )
        minimum_stride = stride * descriptor.global_dims[ordinal + 1]
    if any(dimension > 256 for dimension in descriptor.box_dims):
        raise CudaMaterializeError("TMA box dimension exceeds CUDA's 256 limit")
    if any(stride > 8 for stride in descriptor.element_strides):
        raise CudaMaterializeError("TMA element stride exceeds CUDA's 8 limit")

    inner_bytes = descriptor.box_dims[0] * element_bytes
    if descriptor.interleave == "none":
        if inner_bytes % 16:
            raise CudaMaterializeError(
                "TMA non-interleaved inner box must span a multiple of 16 bytes"
            )
        swizzle_limit = _TMA_SWIZZLE_BYTES.get(descriptor.swizzle)
        if swizzle_limit is not None and inner_bytes > swizzle_limit:
            raise CudaMaterializeError(
                f"TMA inner box exceeds {swizzle_limit}-byte swizzle limit"
            )
    if (
        descriptor.interleave == "interleave_32b"
        and descriptor.swizzle != "swizzle_32b"
    ):
        raise CudaMaterializeError("TMA 32-byte interleave requires 32-byte swizzle")
    if (
        descriptor.oob_fill == "nan_request_zero_fma"
        and descriptor.element_type not in _TMA_FLOAT_ELEMENTS
    ):
        raise CudaMaterializeError("TMA NaN OOB fill requires a floating element type")


@dataclass(frozen=True)
class CudaPrimitiveRegistry:
    """Captured, validator-owned live authority for the closed primitives."""

    driver: object
    tensor_facts: Callable[[object], CudaTensorFacts]
    tma_descriptor: Callable[[CudaResolvedTmaDescriptor], bytes] | None
    group_handles: Mapping[str, Callable[[object], int]]
    synchronize: Callable[[], None]
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        if not callable(self.tensor_facts):
            raise CudaMaterializeError("CUDA tensor-facts provider is not callable")
        if self.tma_descriptor is not None and not callable(self.tma_descriptor):
            raise CudaMaterializeError("CUDA TMA provider is not callable")
        if not isinstance(self.group_handles, Mapping) or set(self.group_handles) - {
            "native_handle",
            "peer_ptr_table",
        }:
            raise CudaMaterializeError("CUDA group-handle registry is malformed")
        if not all(callable(resolver) for resolver in self.group_handles.values()):
            raise CudaMaterializeError("CUDA group-handle resolver is not callable")
        if not callable(self.synchronize):
            raise CudaMaterializeError("CUDA synchronization provider is not callable")
        expected = {
            CUDA_CHECKED_EXPRESSION_CAPABILITY,
            CUDA_PACKED_STRUCT_CAPABILITY,
            CUTLASS_FAST_DIVMOD_CAPABILITY,
            CUTE_FAST_DIVMOD_CAPABILITY,
        }
        if self.tma_descriptor is not None:
            expected.add(CUDA_TMA_DESCRIPTOR_CAPABILITY)
        expected.update(
            {
                "native_handle": GROUP_NATIVE_HANDLE_CAPABILITY,
                "peer_ptr_table": GROUP_PEER_POINTER_TABLE_CAPABILITY,
            }[projection]
            for projection in self.group_handles
        )
        if self.capabilities != frozenset(expected):
            raise CudaMaterializeError(
                "CUDA primitive capabilities differ from captured providers"
            )


def _default_tensor_facts(value: object) -> CudaTensorFacts:
    try:
        address = int(value.data_ptr())
        shape = tuple(int(dimension) for dimension in value.shape)
        stride = tuple(int(dimension) for dimension in value.stride())
        element_size = int(value.element_size())
        storage_offset = int(value.storage_offset())
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise CudaMaterializeError(f"cannot inspect validator tensor: {exc}") from None
    return CudaTensorFacts(address, shape, stride, element_size, storage_offset)


_TMA_DRIVER_ELEMENT = MappingProxyType(
    {
        "u8": "CU_TENSOR_MAP_DATA_TYPE_UINT8",
        "u16": "CU_TENSOR_MAP_DATA_TYPE_UINT16",
        "u32": "CU_TENSOR_MAP_DATA_TYPE_UINT32",
        "i32": "CU_TENSOR_MAP_DATA_TYPE_INT32",
        "i64": "CU_TENSOR_MAP_DATA_TYPE_INT64",
        "f16": "CU_TENSOR_MAP_DATA_TYPE_FLOAT16",
        "f32": "CU_TENSOR_MAP_DATA_TYPE_FLOAT32",
        "f64": "CU_TENSOR_MAP_DATA_TYPE_FLOAT64",
        "bf16": "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
        "tf32": "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32",
        "tf32_32b": "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ",
        # CuTe/CUTLASS represents both FP8 formats as one opaque byte at the
        # tensor-map boundary; interpretation remains in device code.
        "f8e4m3": "CU_TENSOR_MAP_DATA_TYPE_UINT8",
        "f8e5m2": "CU_TENSOR_MAP_DATA_TYPE_UINT8",
    }
)
_TMA_DRIVER_INTERLEAVE = MappingProxyType(
    {
        "none": "CU_TENSOR_MAP_INTERLEAVE_NONE",
        "interleave_16b": "CU_TENSOR_MAP_INTERLEAVE_16B",
        "interleave_32b": "CU_TENSOR_MAP_INTERLEAVE_32B",
    }
)
_TMA_DRIVER_SWIZZLE = MappingProxyType(
    {
        "none": "CU_TENSOR_MAP_SWIZZLE_NONE",
        "swizzle_32b": "CU_TENSOR_MAP_SWIZZLE_32B",
        "swizzle_64b": "CU_TENSOR_MAP_SWIZZLE_64B",
        "swizzle_128b": "CU_TENSOR_MAP_SWIZZLE_128B",
        "swizzle_128b_atom_32b": "CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B",
    }
)
_TMA_DRIVER_L2 = MappingProxyType(
    {
        "none": "CU_TENSOR_MAP_L2_PROMOTION_NONE",
        "l2_64b": "CU_TENSOR_MAP_L2_PROMOTION_L2_64B",
        "l2_128b": "CU_TENSOR_MAP_L2_PROMOTION_L2_128B",
        "l2_256b": "CU_TENSOR_MAP_L2_PROMOTION_L2_256B",
    }
)
_TMA_DRIVER_OOB = MappingProxyType(
    {
        "none": "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE",
        # CUDA has no distinct zero enum: NONE is the ordinary zero-fill mode.
        "zero": "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE",
        "nan_request_zero_fma": (
            "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA"
        ),
    }
)


def _driver_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise CudaMaterializeError(f"CUDA driver returned malformed {field}")
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError):
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, bool):
            raise CudaMaterializeError(f"CUDA driver returned malformed {field}")
        try:
            return operator.index(enum_value)
        except (TypeError, ValueError, OverflowError):
            raise CudaMaterializeError(
                f"CUDA driver returned malformed {field}"
            ) from None


def _driver_enum(driver: object, owner: str, member: str) -> object:
    try:
        return getattr(getattr(driver, owner), member)
    except AttributeError:
        raise CudaMaterializeError(
            f"CUDA driver lacks {owner}.{member} required by TMA v1"
        ) from None


def _driver_success(driver: object, status: object, *, operation: str) -> None:
    success = _driver_integer(
        _driver_enum(driver, "CUresult", "CUDA_SUCCESS"), field="success status"
    )
    observed = _driver_integer(status, field=f"{operation} status")
    if observed != success:
        raise CudaMaterializeError(f"CUDA {operation} failed with status {observed}")


def _make_driver_tma_encoder(
    driver: object,
) -> Callable[[CudaResolvedTmaDescriptor], bytes]:
    """Capture CUDA's real 128-byte ``cuTensorMapEncodeTiled`` authority."""

    encoder = getattr(driver, "cuTensorMapEncodeTiled", None)
    tensor_map_type = getattr(driver, "CUtensorMap", None)
    uint32_type = getattr(driver, "cuuint32_t", None)
    uint64_type = getattr(driver, "cuuint64_t", None)
    if (
        not callable(encoder)
        or not isinstance(tensor_map_type, type)
        or not isinstance(uint32_type, type)
        or not isinstance(uint64_type, type)
    ):
        raise CudaMaterializeError(
            "CUDA driver lacks cuTensorMapEncodeTiled/CUtensorMap/"
            "cuuint32_t/cuuint64_t"
        )
    # Resolve every closed enum now, before any candidate invocation can mutate
    # module attributes.  Duplicate logical values (FP8/zero) are intentional.
    element = {
        name: _driver_enum(driver, "CUtensorMapDataType", member)
        for name, member in _TMA_DRIVER_ELEMENT.items()
    }
    interleave = {
        name: _driver_enum(driver, "CUtensorMapInterleave", member)
        for name, member in _TMA_DRIVER_INTERLEAVE.items()
    }
    swizzle = {
        name: _driver_enum(driver, "CUtensorMapSwizzle", member)
        for name, member in _TMA_DRIVER_SWIZZLE.items()
    }
    l2 = {
        name: _driver_enum(driver, "CUtensorMapL2promotion", member)
        for name, member in _TMA_DRIVER_L2.items()
    }
    oob = {
        name: _driver_enum(driver, "CUtensorMapFloatOOBfill", member)
        for name, member in _TMA_DRIVER_OOB.items()
    }

    def encode(descriptor: CudaResolvedTmaDescriptor) -> bytes:
        if type(descriptor) is not CudaResolvedTmaDescriptor:
            raise CudaMaterializeError("CUDA TMA encoder received the wrong type")
        try:
            result = encoder(
                element[descriptor.element_type],
                len(descriptor.global_dims),
                descriptor.address,
                [uint64_type(value) for value in descriptor.global_dims],
                [uint64_type(value) for value in descriptor.global_strides],
                [uint32_type(value) for value in descriptor.box_dims],
                [uint32_type(value) for value in descriptor.element_strides],
                interleave[descriptor.interleave],
                swizzle[descriptor.swizzle],
                l2[descriptor.l2_promotion],
                oob[descriptor.oob_fill],
            )
        except Exception as exc:  # noqa: BLE001 - normalize optional binding
            raise CudaMaterializeError(
                f"CUDA cuTensorMapEncodeTiled binding raised "
                f"{type(exc).__name__}: {exc}"
            ) from None
        if not isinstance(result, tuple) or len(result) != 2:
            raise CudaMaterializeError(
                "CUDA cuTensorMapEncodeTiled returned a malformed result"
            )
        _driver_success(
            driver, result[0], operation="cuTensorMapEncodeTiled"
        )
        tensor_map = result[1]
        if type(tensor_map) is not tensor_map_type:
            raise CudaMaterializeError(
                "CUDA cuTensorMapEncodeTiled returned a malformed tensor map"
            )
        try:
            opaque = tuple(tensor_map.opaque)
        except (AttributeError, TypeError) as exc:
            raise CudaMaterializeError(
                f"CUDA tensor map does not expose its sealed 128 bytes: {exc}"
            ) from None
        if len(opaque) != 16:
            raise CudaMaterializeError("CUDA tensor map opaque word count is not 16")
        try:
            words = tuple(
                int(value)
                if type(value) is uint64_type
                else _driver_integer(value, field="tensor-map opaque word")
                for value in opaque
            )
        except (TypeError, ValueError, OverflowError):
            raise CudaMaterializeError(
                "CUDA driver returned malformed tensor-map opaque word"
            ) from None
        if any(not 0 <= value <= _UINT64_MAX for value in words):
            raise CudaMaterializeError("CUDA tensor map opaque word is outside uint64")
        return struct.pack("<16Q", *words)

    return encode


def _make_driver_synchronize(driver: object) -> Callable[[], None]:
    synchronize = getattr(driver, "cuCtxSynchronize", None)
    if not callable(synchronize):
        raise CudaMaterializeError("CUDA driver lacks callable cuCtxSynchronize")

    def captured() -> None:
        try:
            result = synchronize()
        except Exception as exc:  # noqa: BLE001 - normalize optional binding
            raise CudaMaterializeError(
                f"CUDA cuCtxSynchronize binding raised {type(exc).__name__}: {exc}"
            ) from None
        if not isinstance(result, tuple) or len(result) != 1:
            raise CudaMaterializeError("CUDA cuCtxSynchronize returned malformed result")
        _driver_success(driver, result[0], operation="cuCtxSynchronize")

    return captured


def make_cuda_primitive_registry(
    *,
    driver: object,
    tensor_facts: Callable[[object], CudaTensorFacts] | None = None,
    tma_descriptor: Callable[[CudaResolvedTmaDescriptor], bytes] | None = None,
    group_handles: Mapping[str, Callable[[object], int]] | None = None,
    synchronize: Callable[[], None] | None = None,
) -> CudaPrimitiveRegistry:
    """Capture the one generic registry; no slot or kernel key is accepted."""

    groups = MappingProxyType(dict(group_handles or {}))
    if tma_descriptor is None:
        tma_descriptor = _make_driver_tma_encoder(driver)
    if synchronize is None:
        synchronize = _make_driver_synchronize(driver)
    capabilities = {
        CUDA_CHECKED_EXPRESSION_CAPABILITY,
        CUDA_PACKED_STRUCT_CAPABILITY,
        CUTLASS_FAST_DIVMOD_CAPABILITY,
        CUTE_FAST_DIVMOD_CAPABILITY,
    }
    if tma_descriptor is not None:
        capabilities.add(CUDA_TMA_DESCRIPTOR_CAPABILITY)
    capabilities.update(
        {
            "native_handle": GROUP_NATIVE_HANDLE_CAPABILITY,
            "peer_ptr_table": GROUP_PEER_POINTER_TABLE_CAPABILITY,
        }[projection]
        for projection in groups
    )
    return CudaPrimitiveRegistry(
        driver=driver,
        tensor_facts=tensor_facts or _default_tensor_facts,
        tma_descriptor=tma_descriptor,
        group_handles=groups,
        synchronize=synchronize,
        capabilities=frozenset(capabilities),
    )


class _ExpressionContext:
    def __init__(
        self,
        bindings: tuple[object, ...],
        registry: CudaPrimitiveRegistry,
    ) -> None:
        self.bindings = bindings
        self.registry = registry
        self._tensor_cache: dict[int, CudaTensorFacts] = {}

    def binding(self, index: int) -> object:
        try:
            return self.bindings[index]
        except IndexError:
            raise CudaMaterializeError(
                f"CUDA device plan references missing binding {index}"
            ) from None

    def tensor(self, index: int) -> CudaTensorFacts:
        cached = self._tensor_cache.get(index)
        if cached is None:
            value = self.registry.tensor_facts(self.binding(index))
            if type(value) is not CudaTensorFacts:
                raise CudaMaterializeError(
                    "CUDA tensor-facts provider returned the wrong type"
                )
            self._tensor_cache[index] = value
            return value
        return cached


def _integer(value: object, *, field: str) -> int:
    if type(value) is not int or not _INT64_MIN <= value <= _INT64_MAX:
        raise CudaMaterializeError(f"{field} must resolve to a signed 64-bit integer")
    return value


def _eval_expression(
    expression: CudaCheckedExpression,
    context: _ExpressionContext,
) -> bool | int | float:
    values: list[bool | int | float] = []
    for ordinal, node in enumerate(expression.nodes):
        args = tuple(values[index] for index in node.operands)
        field = f"CUDA expression node {ordinal}"
        if node.op == "const":
            result = node.value
        elif node.op == "binding_scalar":
            result = context.binding(node.binding)  # type: ignore[arg-type]
        elif node.op.startswith("tensor_"):
            tensor = context.tensor(node.binding)  # type: ignore[arg-type]
            if node.op == "tensor_dim":
                try:
                    result = tensor.shape[node.axis]  # type: ignore[index]
                except IndexError:
                    raise CudaMaterializeError(f"{field} axis exceeds live tensor rank") from None
            elif node.op == "tensor_stride":
                try:
                    result = tensor.stride[node.axis]  # type: ignore[index]
                except IndexError:
                    raise CudaMaterializeError(f"{field} axis exceeds live tensor rank") from None
            elif node.op == "tensor_numel":
                result = tensor.numel
            elif node.op == "tensor_element_size":
                result = tensor.element_size
            else:
                result = tensor.storage_offset
        elif node.op == "neg":
            if type(args[0]) is bool:
                raise CudaMaterializeError(f"{field} cannot negate bool")
            result = -args[0]
        elif node.op == "not":
            if type(args[0]) is not bool:
                raise CudaMaterializeError(f"{field} logical operand is not bool")
            result = not args[0]
        elif node.op in {"and", "or"}:
            if any(type(arg) is not bool for arg in args):
                raise CudaMaterializeError(f"{field} logical operand is not bool")
            result = (args[0] and args[1]) if node.op == "and" else (args[0] or args[1])
        elif node.op == "select":
            if type(args[0]) is not bool or type(args[1]) is not type(args[2]):
                raise CudaMaterializeError(f"{field} select types are incompatible")
            result = args[1] if args[0] else args[2]
        elif node.op in {"eq", "ne", "lt", "le", "gt", "ge"}:
            if type(args[0]) is not type(args[1]):
                raise CudaMaterializeError(f"{field} comparison types differ")
            result = {
                "eq": args[0] == args[1],
                "ne": args[0] != args[1],
                "lt": args[0] < args[1],
                "le": args[0] <= args[1],
                "gt": args[0] > args[1],
                "ge": args[0] >= args[1],
            }[node.op]
        elif node.op in {"min", "max"}:
            if any(type(arg) is not type(args[0]) or type(arg) is bool for arg in args):
                raise CudaMaterializeError(f"{field} numeric operand types differ")
            result = min(args) if node.op == "min" else max(args)
        else:
            left, right = args
            if type(left) is not type(right) or type(left) is bool:
                raise CudaMaterializeError(f"{field} arithmetic operand types differ")
            if node.op in {"floor_div", "ceil_div", "mod"}:
                left = _integer(left, field=field)
                right = _integer(right, field=field)
                if right <= 0:
                    raise CudaMaterializeError(f"{field} divisor must be positive")
                if node.op == "floor_div":
                    result = left // right
                elif node.op == "ceil_div":
                    result = -(-left // right)
                else:
                    result = left % right
            else:
                result = {
                    "add": lambda: left + right,
                    "sub": lambda: left - right,
                    "mul": lambda: left * right,
                }[node.op]()
        values.append(_checked_number(result, field=field))
    return values[expression.result]


def evaluate_cuda_expression(
    expression: CudaCheckedExpression,
    bindings: tuple[object, ...],
    registry: CudaPrimitiveRegistry,
) -> bool | int | float:
    if type(expression) is not CudaCheckedExpression or type(bindings) is not tuple:
        raise CudaMaterializeError("CUDA expression evaluation authority is malformed")
    return _eval_expression(expression, _ExpressionContext(bindings, registry))


def _pack_scalar(scalar_type: str, value: object) -> bytes:
    enum = CudaScalarType(scalar_type)
    materialized = CudaScalar(enum, value)  # validates exact type/range
    try:
        return struct.pack("<" + _SCALAR_STRUCT[scalar_type], materialized.value)
    except (OverflowError, struct.error) as exc:
        raise CudaMaterializeError(f"cannot pack CUDA scalar: {exc}") from None


def _checked_i32_divisor(divisor: object, *, family: str) -> int:
    if type(divisor) is not int or not 1 <= divisor <= (1 << 31) - 1:
        raise CudaMaterializeError(
            f"{family} FastDivmod divisor is outside i32 policy"
        )
    return divisor


def _cutlass_fast_divmod_i32(divisor: object) -> bytes:
    divisor = _checked_i32_divisor(divisor, family="CUTLASS")
    multiplier = 0
    shift_right = 0
    if divisor != 1:
        ceil_log2 = (divisor - 1).bit_length()
        p = 31 + ceil_log2
        multiplier = ((1 << p) + divisor - 1) // divisor
        shift_right = p - 32
    return struct.pack("<iII", divisor, multiplier, shift_right)


def _cute_fast_divmod_i32(divisor: object) -> bytes:
    """Materialize CuTe DSL's four-field reciprocal divisor ABI."""

    divisor = _checked_i32_divisor(divisor, family="CuTe")
    ceil_log2 = (divisor - 1).bit_length()
    multiplier = (
        ((1 << 32) * ((1 << ceil_log2) - divisor)) // divisor
    ) + 1
    shift_1 = min(ceil_log2, 1)
    shift_2 = max(ceil_log2 - 1, 0)
    return struct.pack("<iIBB2x", divisor, multiplier, shift_1, shift_2)


def materialize_cuda_parameter(
    plan: CudaParameterPlan,
    bindings: tuple[object, ...],
    registry: CudaPrimitiveRegistry,
) -> CudaMaterializedParameter:
    """Materialize one parameter without importing or invoking miner code."""

    if type(plan) is not CudaParameterPlan or type(bindings) is not tuple:
        raise CudaMaterializeError("CUDA parameter materialization input is malformed")
    context = _ExpressionContext(bindings, registry)
    if plan.kind == "scalar":
        assert plan.expression is not None and plan.scalar_type is not None
        value = _eval_expression(plan.expression, context)
        return CudaScalar(CudaScalarType(plan.scalar_type), value)
    if plan.kind == "pointer":
        assert plan.binding is not None
        value = context.binding(plan.binding)
        if type(value) is int:
            address = value
        else:
            address = context.tensor(plan.binding).address
        return CudaPointer(address)
    if plan.kind == "packed_struct":
        packed = bytearray(plan.size)
        for field in plan.fields:
            if field.data_hex is not None:
                raw = bytes.fromhex(field.data_hex)
            elif field.tensor_binding is not None:
                raw = struct.pack(
                    "<Q", context.tensor(field.tensor_binding).address
                )
            else:
                raw = _pack_scalar(
                    field.scalar_type,  # type: ignore[arg-type]
                    _eval_expression(field.expression, context),  # type: ignore[arg-type]
                )
            packed[field.offset : field.offset + len(raw)] = raw
        return CudaOpaqueBytes(bytes(packed))
    if plan.kind == "cutlass_fast_divmod_i32_v1":
        assert plan.expression is not None
        return CudaOpaqueBytes(
            _cutlass_fast_divmod_i32(_eval_expression(plan.expression, context))
        )
    if plan.kind == "cute_fast_divmod_i32_v1":
        assert plan.expression is not None
        return CudaOpaqueBytes(
            _cute_fast_divmod_i32(_eval_expression(plan.expression, context))
        )
    if plan.kind == "group_handle":
        assert plan.binding is not None and plan.group_projection is not None
        try:
            resolver = registry.group_handles[plan.group_projection]
        except KeyError:
            raise CudaMaterializeError(
                f"CUDA group provider lacks projection {plan.group_projection!r}"
            ) from None
        address = resolver(context.binding(plan.binding))
        if type(address) is not int:
            raise CudaMaterializeError("CUDA group provider returned a non-integer handle")
        return CudaPointer(address)
    assert plan.kind == "tma_descriptor" and plan.tma is not None
    encoder = registry.tma_descriptor
    if encoder is None:
        raise CudaMaterializeError("CUDA primitive registry lacks TMA authority")
    tma = plan.tma
    tensor = context.tensor(tma.binding)

    def positive(rows: Sequence[CudaCheckedExpression], *, field: str) -> tuple[int, ...]:
        values = tuple(_integer(_eval_expression(row, context), field=field) for row in rows)
        if any(value <= 0 for value in values):
            raise CudaMaterializeError(f"{field} must resolve to positive integers")
        return values

    resolved = CudaResolvedTmaDescriptor(
        address=tensor.address,
        element_type=tma.element_type,
        global_dims=positive(tma.global_dims, field="TMA global dimension"),
        global_strides=positive(tma.global_strides, field="TMA global stride"),
        box_dims=positive(tma.box_dims, field="TMA box dimension"),
        element_strides=positive(tma.element_strides, field="TMA element stride"),
        interleave=tma.interleave,
        swizzle=tma.swizzle,
        l2_promotion=tma.l2_promotion,
        oob_fill=tma.oob_fill,
    )
    _validate_resolved_tma_descriptor(resolved)
    raw = encoder(resolved)
    if type(raw) is not bytes or len(raw) != 128:
        raise CudaMaterializeError("CUDA TMA provider returned a malformed descriptor")
    return CudaOpaqueBytes(raw)


def validate_parameter_bindings(
    plan: CudaParameterPlan,
    bindings: Sequence[object],
) -> None:
    """Statically join all expression/primitive references to ArtifactBinding."""

    def require(index: int, allowed: frozenset[str], *, field: str) -> None:
        if not 0 <= index < len(bindings):
            raise CudaMaterializeError(f"{field} references missing binding {index}")
        kind = getattr(bindings[index], "kind", None)
        if kind not in allowed:
            raise CudaMaterializeError(
                f"{field} binding {index} kind {kind!r} is not in {sorted(allowed)!r}"
            )

    if plan.kind == "pointer":
        require(plan.binding, frozenset({"tensor", "pointer", "opaque"}), field="pointer")  # type: ignore[arg-type]
    if plan.kind == "group_handle":
        require(plan.binding, frozenset({"group"}), field="group handle")  # type: ignore[arg-type]
    if plan.tma is not None:
        require(plan.tma.binding, frozenset({"tensor"}), field="TMA descriptor")
    for field in plan.fields:
        if field.tensor_binding is not None:
            require(
                field.tensor_binding,
                frozenset({"tensor"}),
                field="packed tensor pointer",
            )
    for expression in plan.expressions:
        for index, op in expression.binding_references:
            allowed = frozenset({"scalar"}) if op == "binding_scalar" else frozenset({"tensor"})
            require(index, allowed, field=f"expression {op}")


__all__ = [
    "CUDA_CHECKED_EXPRESSION_CAPABILITY",
    "CUDA_EXPRESSION_SCHEMA",
    "CUDA_PACKED_STRUCT_CAPABILITY",
    "CUDA_TMA_DESCRIPTOR_CAPABILITY",
    "CUTLASS_FAST_DIVMOD_CAPABILITY",
    "CUTE_FAST_DIVMOD_CAPABILITY",
    "GROUP_NATIVE_HANDLE_CAPABILITY",
    "GROUP_PEER_POINTER_TABLE_CAPABILITY",
    "CudaCheckedExpression",
    "CudaExpressionNode",
    "CudaMaterializeError",
    "CudaPackedField",
    "CudaParameterPlan",
    "CudaPrimitiveRegistry",
    "CudaResolvedTmaDescriptor",
    "CudaTensorFacts",
    "CudaTmaDescriptorPlan",
    "evaluate_cuda_expression",
    "make_cuda_primitive_registry",
    "materialize_cuda_parameter",
    "validate_parameter_bindings",
]
