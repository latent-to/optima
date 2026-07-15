from __future__ import annotations

import math
import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest

from optima.cuda_launch import CudaOpaqueBytes, CudaPointer
from optima.cuda_materialize import (
    CudaCheckedExpression,
    CudaExpressionNode,
    CudaMaterializeError,
    CudaPackedField,
    CudaParameterPlan,
    CudaTmaDescriptorPlan,
    evaluate_cuda_expression,
    make_cuda_primitive_registry,
    materialize_cuda_parameter,
    validate_parameter_bindings,
)


def _const(value: bool | int | float) -> CudaCheckedExpression:
    return CudaCheckedExpression((CudaExpressionNode("const", value=value),), 0)


def _source(op: str, binding: int, axis: int | None = None) -> CudaCheckedExpression:
    return CudaCheckedExpression(
        (CudaExpressionNode(op, binding=binding, axis=axis),), 0
    )


def _registry(*, group_handles=None):
    return make_cuda_primitive_registry(
        driver=object(),
        tma_descriptor=lambda _descriptor: bytes(128),
        group_handles=group_handles,
        synchronize=lambda: None,
    )


class _Tensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        *,
        pointer: int = 0x4000,
        element_size: int = 2,
    ) -> None:
        self.shape = shape
        self._stride = stride
        self._pointer = pointer
        self._element_size = element_size

    def data_ptr(self) -> int:
        return self._pointer

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def element_size(self) -> int:
        return self._element_size

    def storage_offset(self) -> int:
        return 0


def test_float_constants_have_exact_canonical_wire_identity() -> None:
    expression = _const(-0.0)
    encoded = expression.to_dict()

    assert encoded["nodes"][0]["value"] == {"f64_hex": "-0x0.0p+0"}
    decoded = CudaCheckedExpression.from_dict(encoded)
    assert decoded == expression
    assert math.copysign(1.0, decoded.nodes[0].value) == -1.0


@pytest.mark.parametrize(
    "wire_value",
    (
        1.0,
        {"f64_hex": "nan"},
        {"f64_hex": "inf"},
        {"f64_hex": "0x0p+0"},
        {"f64_hex": 1},
        {"float": "0x0.0p+0"},
    ),
)
def test_float_wire_rejects_raw_nonfinite_and_noncanonical_values(
    wire_value: object,
) -> None:
    node = {
        "axis": None,
        "binding": None,
        "op": "const",
        "operands": [],
        "value": wire_value,
    }
    with pytest.raises(CudaMaterializeError, match="float constant|value"):
        CudaExpressionNode.from_dict(node)


def test_checked_expression_derives_dynamic_geometry_and_tensor_element_size() -> None:
    expression = CudaCheckedExpression(
        (
            CudaExpressionNode("tensor_dim", binding=0, axis=0),
            CudaExpressionNode("const", value=128),
            CudaExpressionNode("ceil_div", operands=(0, 1)),
            CudaExpressionNode("tensor_dim", binding=0, axis=1),
            CudaExpressionNode("ceil_div", operands=(3, 1)),
            CudaExpressionNode("mul", operands=(2, 4)),
            CudaExpressionNode("const", value=148),
            CudaExpressionNode("min", operands=(5, 6)),
        ),
        7,
    )
    tensor = _Tensor((300, 129), (129, 1), element_size=4)

    assert evaluate_cuda_expression(expression, (tensor,), _registry()) == 6
    assert evaluate_cuda_expression(
        _source("tensor_element_size", 0), (tensor,), _registry()
    ) == 4


def test_fast_divmod_families_and_static_packed_atoms_have_exact_layout() -> None:
    registry = _registry()
    cutlass_fast_divmod = CudaParameterPlan(
        kind="cutlass_fast_divmod_i32_v1",
        size=12,
        expression=_const(128),
    )
    cute_fast_divmod = CudaParameterPlan(
        kind="cute_fast_divmod_i32_v1",
        size=12,
        expression=_const(128),
    )
    packed = CudaParameterPlan(
        kind="packed_struct",
        size=8,
        fields=(
            CudaPackedField(offset=0, data_hex="010203"),
            CudaPackedField(
                offset=4,
                scalar_type="i32",
                expression=_const(7),
            ),
        ),
    )

    assert materialize_cuda_parameter(
        cutlass_fast_divmod, (), registry
    ) == CudaOpaqueBytes(struct.pack("<iII", 128, 0x8000_0000, 6))
    assert materialize_cuda_parameter(
        cute_fast_divmod, (), registry
    ) == CudaOpaqueBytes(
        struct.pack("<iIBB2x", 128, 1, 1, 6)
    )
    assert materialize_cuda_parameter(packed, (), registry) == CudaOpaqueBytes(
        b"\x01\x02\x03\x00" + struct.pack("<i", 7)
    )

    with pytest.raises(CudaMaterializeError, match="fields mismatch"):
        CudaPackedField.from_dict(
            {
                "binding": 0,
                "data_hex": "010203",
                "expression": None,
                "offset": 0,
                "scalar_type": None,
                "tensor_binding": None,
            }
        )
    # The generic boundary accepts any sealed compile-time struct up to the
    # existing CUDA parameter cap; there is no CuTe-shape-specific literal cap.
    assert CudaParameterPlan(
        kind="packed_struct",
        size=4096,
        fields=(CudaPackedField(offset=0, data_hex="00" * 4096),),
    ).size == 4096
    with pytest.raises(CudaMaterializeError, match="literal field"):
        CudaPackedField(offset=0, data_hex="00" * 4097)


@pytest.mark.parametrize(
    ("divisor", "multiplier", "shift_1", "shift_2"),
    (
        (1, 0x0000_0001, 0, 0),
        (2, 0x0000_0001, 1, 0),
        (3, 0x5555_5556, 1, 1),
        (5, 0x9999_999A, 1, 2),
        (7, 0x2492_4925, 1, 2),
        (10, 0x9999_999A, 1, 3),
        (32, 0x0000_0001, 1, 4),
        (127, 0x0204_0811, 1, 6),
        (128, 0x0000_0001, 1, 6),
        (129, 0xFC07_F020, 1, 7),
        (148, 0xBACF_914D, 1, 7),
        (256, 0x0000_0001, 1, 7),
        (1000, 0x0624_DD30, 1, 9),
    ),
)
def test_cute_fast_divmod_exact_bytes_and_device_formula(
    divisor: int, multiplier: int, shift_1: int, shift_2: int
) -> None:
    plan = CudaParameterPlan(
        kind="cute_fast_divmod_i32_v1",
        size=12,
        expression=_const(divisor),
    )
    materialized = materialize_cuda_parameter(plan, (), _registry())
    assert materialized == CudaOpaqueBytes(
        struct.pack("<iIBB2x", divisor, multiplier, shift_1, shift_2)
    )
    for dividend in {
        0,
        1,
        divisor - 1,
        divisor,
        min((1 << 32) - 1, divisor + 1),
        (1 << 31) - 1,
        (1 << 32) - 1,
    }:
        high = (dividend * multiplier) >> 32
        quotient = (high + ((dividend - high) >> shift_1)) >> shift_2
        remainder = dividend - quotient * divisor
        assert (quotient, remainder) == divmod(dividend, divisor)


def test_packed_memref_uses_only_live_tensor_pointer_shape_and_stride() -> None:
    output = _Tensor(
        (37, 11),
        (16, 1),
        pointer=0x1234_5678,
        element_size=4,
    )
    plan = CudaParameterPlan(
        kind="packed_struct",
        size=24,
        fields=(
            CudaPackedField(offset=0, tensor_binding=0),
            CudaPackedField(
                offset=8,
                scalar_type="i32",
                expression=_source("tensor_dim", 0, 0),
            ),
            CudaPackedField(
                offset=12,
                scalar_type="i32",
                expression=_source("tensor_dim", 0, 1),
            ),
            CudaPackedField(
                offset=16,
                scalar_type="i32",
                expression=_source("tensor_stride", 0, 0),
            ),
        ),
    )

    assert CudaParameterPlan.from_dict(plan.to_dict()) == plan
    sparse = {
        "kind": "packed_struct",
        "size": 24,
        "fields": [
            {"offset": 0, "tensor_binding": 0},
            {
                "offset": 8,
                "scalar_type": "i32",
                "expression": {
                    "nodes": [
                        {"op": "tensor_dim", "binding": 0, "axis": 0}
                    ],
                    "result": 0,
                    "schema": "optima.cuda-expression-dag.v1",
                },
            },
            {
                "offset": 12,
                "scalar_type": "i32",
                "expression": _source("tensor_dim", 0, 1).to_dict(),
            },
            {
                "offset": 16,
                "scalar_type": "i32",
                "expression": _source("tensor_stride", 0, 0).to_dict(),
            },
        ],
    }
    assert CudaParameterPlan.from_dict(sparse) == plan
    assert CudaParameterPlan.from_dict(sparse).to_dict() == plan.to_dict()
    validate_parameter_bindings(
        plan,
        (SimpleNamespace(kind="tensor"),),
    )
    assert materialize_cuda_parameter(
        plan, (output,), _registry()
    ) == CudaOpaqueBytes(
        struct.pack("<Qiii", 0x1234_5678, 37, 11, 16) + b"\0" * 4
    )


def test_packed_tensor_pointer_rejects_raw_or_non_tensor_authority() -> None:
    with pytest.raises(CudaMaterializeError, match="exactly one"):
        CudaPackedField(
            offset=0,
            tensor_binding=0,
            data_hex="0000000000000000",
        )
    with pytest.raises(CudaMaterializeError, match="binding-index"):
        CudaPackedField(offset=0, tensor_binding=64)

    pointer_field = CudaPackedField(offset=0, tensor_binding=0).to_dict()
    with pytest.raises(CudaMaterializeError, match="fields mismatch"):
        CudaPackedField.from_dict({**pointer_field, "pointer_address": 0x1234})

    plan = CudaParameterPlan(
        kind="packed_struct",
        size=8,
        fields=(CudaPackedField(offset=0, tensor_binding=0),),
    )
    with pytest.raises(CudaMaterializeError, match="kind 'scalar'"):
        validate_parameter_bindings(
            plan,
            (SimpleNamespace(kind="scalar"),),
        )


class _FakeCuuint32:
    def __init__(self, value: object) -> None:
        if type(value) is not int or not 0 <= value < 2**32:
            raise OverflowError("outside cuuint32_t")
        self._value = value

    def __int__(self) -> int:
        return self._value


class _FakeCuuint64:
    def __init__(self, value: object) -> None:
        if type(value) is not int or not 0 <= value < 2**64:
            raise OverflowError("outside cuuint64_t")
        self._value = value

    def __int__(self) -> int:
        return self._value


class _FakeTmaDriver:
    cuuint32_t = _FakeCuuint32
    cuuint64_t = _FakeCuuint64
    CUresult = type("CUresult", (), {"CUDA_SUCCESS": 0})
    CUtensorMapDataType = type(
        "CUtensorMapDataType",
        (),
        {
            name: ordinal
            for ordinal, name in enumerate(
                (
                    "CU_TENSOR_MAP_DATA_TYPE_UINT8",
                    "CU_TENSOR_MAP_DATA_TYPE_UINT16",
                    "CU_TENSOR_MAP_DATA_TYPE_UINT32",
                    "CU_TENSOR_MAP_DATA_TYPE_INT32",
                    "CU_TENSOR_MAP_DATA_TYPE_INT64",
                    "CU_TENSOR_MAP_DATA_TYPE_FLOAT16",
                    "CU_TENSOR_MAP_DATA_TYPE_FLOAT32",
                    "CU_TENSOR_MAP_DATA_TYPE_FLOAT64",
                    "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
                    "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32",
                    "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ",
                )
            )
        },
    )
    CUtensorMapInterleave = type(
        "CUtensorMapInterleave",
        (),
        {
            "CU_TENSOR_MAP_INTERLEAVE_NONE": 0,
            "CU_TENSOR_MAP_INTERLEAVE_16B": 1,
            "CU_TENSOR_MAP_INTERLEAVE_32B": 2,
        },
    )
    CUtensorMapSwizzle = type(
        "CUtensorMapSwizzle",
        (),
        {
            "CU_TENSOR_MAP_SWIZZLE_NONE": 0,
            "CU_TENSOR_MAP_SWIZZLE_32B": 1,
            "CU_TENSOR_MAP_SWIZZLE_64B": 2,
            "CU_TENSOR_MAP_SWIZZLE_128B": 3,
            "CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B": 4,
        },
    )
    CUtensorMapL2promotion = type(
        "CUtensorMapL2promotion",
        (),
        {
            "CU_TENSOR_MAP_L2_PROMOTION_NONE": 0,
            "CU_TENSOR_MAP_L2_PROMOTION_L2_64B": 1,
            "CU_TENSOR_MAP_L2_PROMOTION_L2_128B": 2,
            "CU_TENSOR_MAP_L2_PROMOTION_L2_256B": 3,
        },
    )
    CUtensorMapFloatOOBfill = type(
        "CUtensorMapFloatOOBfill",
        (),
        {
            "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE": 0,
            "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA": 1,
        },
    )

    class CUtensorMap:
        def __init__(self) -> None:
            self.opaque = tuple(_FakeCuuint64(value) for value in range(16))

    def __init__(self) -> None:
        self.encode_calls: list[tuple[object, ...]] = []
        self.sync_calls = 0

    def cuTensorMapEncodeTiled(self, *args: object) -> tuple[int, object]:
        self.encode_calls.append(args)
        return 0, self.CUtensorMap()

    def cuCtxSynchronize(self) -> tuple[int]:
        self.sync_calls += 1
        return (0,)


def test_real_driver_tma_materializer_uses_live_byte_strides_and_exact_128_bytes() -> None:
    driver = _FakeTmaDriver()
    registry = make_cuda_primitive_registry(driver=driver)
    byte_stride = CudaCheckedExpression(
        (
            CudaExpressionNode("tensor_stride", binding=0, axis=0),
            CudaExpressionNode("tensor_element_size", binding=0),
            CudaExpressionNode("mul", operands=(0, 1)),
        ),
        2,
    )
    tma = CudaTmaDescriptorPlan(
        binding=0,
        element_type="f16",
        global_dims=(
            _source("tensor_dim", 0, 1),
            _source("tensor_dim", 0, 0),
        ),
        global_strides=(byte_stride,),
        box_dims=(_const(8), _const(4)),
        element_strides=(_const(1), _const(1)),
    )
    plan = CudaParameterPlan(kind="tma_descriptor", size=128, tma=tma)

    materialized = materialize_cuda_parameter(
        plan,
        (_Tensor((3, 8), (8, 1), pointer=0x8000, element_size=2),),
        registry,
    )

    assert materialized == CudaOpaqueBytes(struct.pack("<16Q", *range(16)))
    assert len(driver.encode_calls) == 1
    call = driver.encode_calls[0]
    assert call[1:3] == (2, 0x8000)
    assert tuple(tuple(int(value) for value in row) for row in call[3:7]) == (
        (8, 3),
        (16,),
        (8, 4),
        (1, 1),
    )
    assert all(type(value) is driver.cuuint64_t for value in call[3])
    assert all(type(value) is driver.cuuint64_t for value in call[4])
    assert all(type(value) is driver.cuuint32_t for value in call[5])
    assert all(type(value) is driver.cuuint32_t for value in call[6])
    registry.synchronize()
    assert driver.sync_calls == 1


@pytest.mark.parametrize("field", ("cuuint32_t", "cuuint64_t"))
def test_real_driver_tma_materializer_requires_unsigned_wrapper_types(field) -> None:
    driver = _FakeTmaDriver()
    setattr(driver, field, None)

    with pytest.raises(CudaMaterializeError, match="cuuint32_t/cuuint64_t"):
        make_cuda_primitive_registry(driver=driver)


def test_real_driver_tma_materializer_rejects_box_overflow_before_encode() -> None:
    driver = _FakeTmaDriver()
    registry = make_cuda_primitive_registry(driver=driver)
    tma = CudaTmaDescriptorPlan(
        binding=0,
        element_type="f16",
        global_dims=(_const(3), _const(4)),
        global_strides=(_const(16),),
        box_dims=(_const(2**32), _const(4)),
        element_strides=(_const(1), _const(1)),
    )

    with pytest.raises(CudaMaterializeError, match="256 limit"):
        materialize_cuda_parameter(
            CudaParameterPlan(kind="tma_descriptor", size=128, tma=tma),
            (_Tensor((3, 4), (4, 1), pointer=0x8000, element_size=2),),
            registry,
        )
    assert driver.encode_calls == []


def _valid_tma_plan() -> CudaTmaDescriptorPlan:
    return CudaTmaDescriptorPlan(
        binding=0,
        element_type="bf16",
        global_dims=(_const(64), _const(128)),
        global_strides=(_const(128),),
        box_dims=(_const(64), _const(128)),
        element_strides=(_const(1), _const(1)),
        swizzle="swizzle_128b",
    )


_INTERLEAVED_TMA = {
    "global_dims": (_const(64), _const(128), _const(2)),
    "global_strides": (_const(128), _const(16384)),
    "box_dims": (_const(64), _const(128), _const(1)),
    "element_strides": (_const(1), _const(1), _const(1)),
    "interleave": "interleave_32b",
    "swizzle": "swizzle_32b",
}


@pytest.mark.parametrize(
    ("changes", "pointer", "message"),
    (
        ({}, 0x8008, "16-byte aligned"),
        ({"global_dims": (_const((1 << 32) + 1), _const(128))}, 0x8000, "2\\^32 limit"),
        ({"global_strides": (_const(136),)}, 0x8000, "16-byte aligned"),
        ({"global_strides": (_const(1 << 40),)}, 0x8000, "2\\^40 limit"),
        ({"global_strides": (_const(112),)}, 0x8000, "nested tensor dimensions"),
        ({"box_dims": (_const(257), _const(128))}, 0x8000, "256 limit"),
        ({"element_strides": (_const(9), _const(1))}, 0x8000, "8 limit"),
        ({"box_dims": (_const(63), _const(128))}, 0x8000, "multiple of 16 bytes"),
        ({"box_dims": (_const(128), _const(64))}, 0x8000, "128-byte swizzle limit"),
        ({"interleave": "interleave_16b"}, 0x8000, "rank must be at least three"),
        ({**_INTERLEAVED_TMA, "swizzle": "swizzle_128b"}, 0x8000, "requires 32-byte swizzle"),
        ({**_INTERLEAVED_TMA, "global_strides": (_const(128), _const(16400))}, 0x8000, "32-byte aligned"),
        ({**_INTERLEAVED_TMA}, 0x8010, "32-byte aligned"),
        (
            {
                "element_type": "i32",
                "global_dims": (_const(32), _const(128)),
                "global_strides": (_const(128),),
                "box_dims": (_const(32), _const(128)),
                "oob_fill": "nan_request_zero_fma",
            },
            0x8000,
            "floating element type",
        ),
    ),
)
def test_tma_constraints_fail_before_driver_encode(
    changes: dict[str, object], pointer: int, message: str
) -> None:
    driver = _FakeTmaDriver()
    registry = make_cuda_primitive_registry(driver=driver)
    plan = CudaParameterPlan(
        kind="tma_descriptor",
        size=128,
        tma=replace(_valid_tma_plan(), **changes),
    )

    with pytest.raises(CudaMaterializeError, match=message):
        materialize_cuda_parameter(
            plan,
            (_Tensor((128, 128), (128, 1), pointer=pointer, element_size=2),),
            registry,
        )
    assert driver.encode_calls == []


def test_group_handle_is_a_closed_validator_resolver() -> None:
    group = object()
    registry = _registry(group_handles={"native_handle": lambda value: 17 if value is group else 0})
    plan = CudaParameterPlan(
        kind="group_handle",
        size=8,
        binding=0,
        group_projection="native_handle",
    )

    assert materialize_cuda_parameter(plan, (group,), registry) == CudaPointer(17)
