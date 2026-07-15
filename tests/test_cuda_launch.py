from __future__ import annotations

import ctypes
import hashlib
import math
import struct

import pytest

from optima.cuda_cubin import (
    CudaCubinABI,
    CudaCubinLibrary,
    CudaKernelABI,
    CudaKernelParameter,
)
from optima.cuda_launch import (
    CUDA_LAUNCH_SCHEMA,
    CudaClusterSchedulingPolicy,
    CudaDim3,
    CudaLaunchAttributes,
    CudaLaunchError,
    CudaLaunchSpec,
    CudaOpaqueBytes,
    CudaPointer,
    CudaPortableClusterMode,
    CudaScalar,
    CudaScalarType,
    CudaSharedMemoryMode,
    launch_cuda_kernel,
    pack_kernel_parameters,
)


def _kernel_abi(
    parameters: tuple[CudaKernelParameter, ...],
    *,
    name: str = "sealed_kernel",
) -> CudaKernelABI:
    return CudaKernelABI(name=name, parameters=parameters)


def _fake_cubin() -> bytes:
    raw = bytearray(64)
    raw[:7] = b"\x7fELF\x02\x01\x01"
    raw[16:18] = (2).to_bytes(2, "little")
    raw[18:20] = (190).to_bytes(2, "little")
    raw[20:24] = (1).to_bytes(4, "little")
    raw[52:54] = (64).to_bytes(2, "little")
    return bytes(raw)


def _cubin_abi(kernel: CudaKernelABI, cubin: bytes) -> CudaCubinABI:
    return CudaCubinABI(
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        cubin_size=len(cubin),
        kernels=(kernel,),
    )


def test_launch_schema_round_trip_is_strict_and_complete() -> None:
    spec = CudaLaunchSpec(
        kernel="cute.block_score$1",
        grid=CudaDim3(148, 2, 1),
        block=CudaDim3(256, 1, 1),
        cluster=CudaDim3(2, 1, 1),
        shared_mem_bytes=96 * 1024,
        attributes=CudaLaunchAttributes(
            cooperative=True,
            cluster_scheduling=CudaClusterSchedulingPolicy.SPREAD,
            programmatic_stream_serialization=True,
            priority=-2,
            portable_cluster_mode=CudaPortableClusterMode.ALLOW_NON_PORTABLE,
            shared_memory_mode=CudaSharedMemoryMode.ALLOW_NON_PORTABLE,
        ),
    )

    encoded = spec.to_dict()
    assert encoded["schema"] == CUDA_LAUNCH_SCHEMA
    assert CudaLaunchSpec.from_dict(encoded) == spec
    assert CudaLaunchAttributes.from_dict({}) == CudaLaunchAttributes()

    with pytest.raises(CudaLaunchError, match="fields mismatch"):
        CudaLaunchSpec.from_dict({**encoded, "candidate_callback": "run_me"})
    with pytest.raises(CudaLaunchError, match="attribute fields mismatch"):
        CudaLaunchAttributes.from_dict(
            {**spec.attributes.to_dict(), "raw_attribute_id": 12}
        )


def test_launch_schema_enforces_bounded_geometry_and_cluster_relationships() -> None:
    with pytest.raises(CudaLaunchError, match="thread-count"):
        CudaLaunchSpec(
            kernel="k",
            grid=CudaDim3(1),
            block=CudaDim3(1024, 2),
        )
    with pytest.raises(CudaLaunchError, match="block-count"):
        CudaLaunchSpec(
            kernel="k",
            grid=CudaDim3(64),
            block=CudaDim3(1),
            cluster=CudaDim3(64),
        )
    with pytest.raises(CudaLaunchError, match="divide the grid"):
        CudaLaunchSpec(
            kernel="k",
            grid=CudaDim3(7),
            block=CudaDim3(1),
            cluster=CudaDim3(2),
        )
    with pytest.raises(CudaLaunchError, match="require an explicit cluster"):
        CudaLaunchSpec(
            kernel="k",
            grid=CudaDim3(1),
            block=CudaDim3(1),
            attributes=CudaLaunchAttributes(
                cluster_scheduling=CudaClusterSchedulingPolicy.DEFAULT
            ),
        )
    with pytest.raises(CudaLaunchError, match="shared memory"):
        CudaLaunchSpec(
            kernel="k",
            grid=CudaDim3(1),
            block=CudaDim3(1),
            shared_mem_bytes=(1 << 20) + 1,
        )


def test_parameter_buffer_uses_driver_offsets_sizes_and_zero_padding() -> None:
    abi = _kernel_abi(
        (
            CudaKernelParameter(index=0, offset=0, size=8),
            CudaKernelParameter(index=1, offset=8, size=4),
            CudaKernelParameter(index=2, offset=16, size=4),
        )
    )
    raw = pack_kernel_parameters(
        abi,
        (
            CudaPointer(0x1234_5678_9ABC_DEF0),
            CudaScalar(CudaScalarType.U32, 0xFEED_BEEF),
            CudaOpaqueBytes(b"TMA!"),
        ),
    )

    assert len(raw) == 20
    assert raw[:8] == struct.pack("<Q", 0x1234_5678_9ABC_DEF0)
    assert raw[8:12] == struct.pack("<I", 0xFEED_BEEF)
    assert raw[12:16] == b"\0" * 4
    assert raw[16:20] == b"TMA!"


@pytest.mark.parametrize(
    ("scalar_type", "value", "expected"),
    [
        (CudaScalarType.BOOL, True, b"\x01"),
        (CudaScalarType.I8, -7, struct.pack("<b", -7)),
        (CudaScalarType.U16, 65535, struct.pack("<H", 65535)),
        (CudaScalarType.I32, -1234, struct.pack("<i", -1234)),
        (CudaScalarType.U64, (1 << 64) - 1, b"\xff" * 8),
        (CudaScalarType.F16, 1.5, struct.pack("<e", 1.5)),
        (CudaScalarType.F32, -2.5, struct.pack("<f", -2.5)),
        (CudaScalarType.F64, 3.25, struct.pack("<d", 3.25)),
    ],
)
def test_checked_scalar_encodings_are_exact(
    scalar_type: CudaScalarType,
    value: bool | int | float,
    expected: bytes,
) -> None:
    abi = _kernel_abi(
        (CudaKernelParameter(index=0, offset=0, size=len(expected)),)
    )
    assert pack_kernel_parameters(abi, (CudaScalar(scalar_type, value),)) == expected


def test_parameter_packing_rejects_implicit_conversions_and_size_mismatch() -> None:
    abi = _kernel_abi((CudaKernelParameter(index=0, offset=0, size=8),))

    with pytest.raises(CudaLaunchError, match="count differs"):
        pack_kernel_parameters(abi, ())
    with pytest.raises(CudaLaunchError, match="must be a tuple"):
        pack_kernel_parameters(abi, [CudaPointer(1)])  # type: ignore[arg-type]
    with pytest.raises(CudaLaunchError, match="sealed ABI requires 8"):
        pack_kernel_parameters(abi, (CudaScalar(CudaScalarType.U32, 1),))
    with pytest.raises(CudaLaunchError, match="finite exact float"):
        CudaScalar(CudaScalarType.F32, math.inf)
    with pytest.raises(CudaLaunchError, match="exact range"):
        CudaScalar(CudaScalarType.U8, 256)
    with pytest.raises(CudaLaunchError, match="exact bool"):
        CudaScalar(CudaScalarType.BOOL, 1)
    with pytest.raises(CudaLaunchError, match="opaque"):
        CudaOpaqueBytes(bytearray(b"x"))  # type: ignore[arg-type]

    called = False

    class ConversionTrap:
        def __index__(self) -> int:
            nonlocal called
            called = True
            return 1

    with pytest.raises(CudaLaunchError, match="uint64"):
        CudaPointer(ConversionTrap())  # type: ignore[arg-type]
    assert called is False


class _FakeStream:
    def __init__(self, address: int) -> None:
        self.address = address


class _FakeDim:
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.z = 0


class _FakeLaunchAttributeValue:
    def __init__(self) -> None:
        self.clusterDim = _FakeDim()
        self.clusterSchedulingPolicyPreference = 0
        self.cooperative = 0
        self.programmaticStreamSerializationAllowed = 0
        self.priority = 0
        self.portableClusterSizeMode = 0
        self.sharedMemoryMode = 0


class _FakeLaunchAttribute:
    def __init__(self) -> None:
        self.id = 0
        self.value = _FakeLaunchAttributeValue()


class _FakeLaunchConfig:
    def __init__(self) -> None:
        self.gridDimX = 0
        self.gridDimY = 0
        self.gridDimZ = 0
        self.blockDimX = 0
        self.blockDimY = 0
        self.blockDimZ = 0
        self.sharedMemBytes = 0
        self.hStream = None
        self.attrs: list[_FakeLaunchAttribute] = []
        self.numAttrs = 0


class _FakeDriver:
    class CUresult:
        CUDA_SUCCESS = 0
        CUDA_ERROR_INVALID_VALUE = 1

    class CUlaunchAttributeID:
        CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2
        CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6
        CU_LAUNCH_ATTRIBUTE_PRIORITY = 8
        CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE = 17
        CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE = 19

    class CUclusterSchedulingPolicy:
        CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = 0
        CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1
        CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2

    class CUlaunchAttributePortableClusterMode:
        CU_LAUNCH_PORTABLE_CLUSTER_MODE_DEFAULT = 0
        CU_LAUNCH_PORTABLE_CLUSTER_MODE_REQUIRE_PORTABLE = 1
        CU_LAUNCH_PORTABLE_CLUSTER_MODE_ALLOW_NON_PORTABLE = 2

    class CUsharedMemoryMode:
        CU_SHARED_MEMORY_MODE_DEFAULT = 0
        CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE = 1
        CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE = 2

    class CUfunction_attribute:
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

    CUstream = _FakeStream
    CUlaunchAttribute = _FakeLaunchAttribute
    CUlaunchAttributeValue = _FakeLaunchAttributeValue
    CUlaunchConfig = _FakeLaunchConfig
    CU_LAUNCH_PARAM_END_AS_INT = 0
    CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT = 1
    CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT = 2

    def __init__(
        self,
        *,
        launch_status: int = 0,
        function_attribute_status: int = 0,
    ) -> None:
        self.launch_status = launch_status
        self.function_attribute_status = function_attribute_status
        self.parameters: tuple[CudaKernelParameter, ...] = ()
        self.calls: list[dict[str, object]] = []
        self.function_attribute_calls: list[tuple[object, object, int]] = []
        self.loaded = 0
        self.unloaded = 0

    def cuInit(self, flags: int) -> tuple[int]:
        assert flags == 0
        return (0,)

    def cuLibraryLoadData(
        self,
        cubin: bytes,
        jit_options: object,
        jit_option_values: object,
        num_jit_options: int,
        library_options: object,
        library_option_values: object,
        num_library_options: int,
    ) -> tuple[int, str]:
        assert cubin == _fake_cubin()
        assert (jit_options, jit_option_values, num_jit_options) == (None, None, 0)
        assert (library_options, library_option_values, num_library_options) == (
            None,
            None,
            0,
        )
        self.loaded += 1
        return (0, f"library-handle-{self.loaded}")

    def cuLibraryGetKernelCount(self, library: object) -> tuple[int, int]:
        assert str(library).startswith("library-handle-")
        return (0, 1)

    def cuLibraryEnumerateKernels(
        self, count: int, library: object
    ) -> tuple[int, list[str]]:
        assert count == 1
        assert str(library).startswith("library-handle-")
        return (0, ["kernel-handle"])

    def cuKernelGetName(self, kernel: object) -> tuple[int, bytes]:
        assert kernel == "kernel-handle"
        return (0, b"sealed_kernel")

    def cuKernelGetParamInfo(
        self, kernel: object, index: int
    ) -> tuple[int, int, int]:
        assert kernel == "kernel-handle"
        if index >= len(self.parameters):
            return (1, 0, 0)
        parameter = self.parameters[index]
        return (0, parameter.offset, parameter.size)

    def cuKernelGetFunction(self, kernel: object) -> tuple[int, str]:
        assert kernel == "kernel-handle"
        return (0, "function-handle")

    def cuLibraryUnload(self, library: object) -> tuple[int]:
        assert str(library).startswith("library-handle-")
        self.unloaded += 1
        return (0,)

    def cuFuncSetAttribute(
        self,
        function: object,
        attribute: object,
        value: int,
    ) -> tuple[int]:
        self.function_attribute_calls.append((function, attribute, value))
        return (self.function_attribute_status,)

    def cuLaunchKernelEx(
        self,
        config: _FakeLaunchConfig,
        function: object,
        kernel_params: object,
        extra_pointer: int,
    ) -> tuple[int]:
        parameter_bytes = b""
        tokens: tuple[int | None, ...] = ()
        if extra_pointer:
            raw_extra = ctypes.cast(
                extra_pointer,
                ctypes.POINTER(ctypes.c_void_p * 5),
            ).contents
            tokens = tuple(raw_extra)
            size = ctypes.cast(
                raw_extra[3], ctypes.POINTER(ctypes.c_size_t)
            ).contents.value
            parameter_bytes = ctypes.string_at(raw_extra[1], size)
        self.calls.append(
            {
                "config": config,
                "extra_pointer": extra_pointer,
                "function": function,
                "kernel_params": kernel_params,
                "parameter_bytes": parameter_bytes,
                "tokens": tokens,
            }
        )
        return (self.launch_status,)


def _library(
    driver: _FakeDriver,
    parameters: tuple[CudaKernelParameter, ...],
) -> CudaCubinLibrary:
    driver.parameters = parameters
    kernel = _kernel_abi(parameters)
    cubin = _fake_cubin()
    return CudaCubinLibrary.open(
        cubin,
        driver=driver,
        expected_abi=_cubin_abi(kernel, cubin),
    )


def test_launch_uses_sealed_library_driver_parameter_buffer_and_attributes() -> None:
    driver = _FakeDriver()
    library = _library(
        driver,
        (
            CudaKernelParameter(index=0, offset=0, size=8),
            CudaKernelParameter(index=1, offset=8, size=4),
        ),
    )
    stream = driver.CUstream(0xCAFE)
    spec = CudaLaunchSpec(
        kernel="sealed_kernel",
        grid=CudaDim3(12, 2, 1),
        block=CudaDim3(256, 1, 1),
        cluster=CudaDim3(2, 1, 1),
        shared_mem_bytes=65_536,
        attributes=CudaLaunchAttributes(
            cooperative=True,
            cluster_scheduling=CudaClusterSchedulingPolicy.LOAD_BALANCING,
            programmatic_stream_serialization=True,
            priority=-3,
            portable_cluster_mode=CudaPortableClusterMode.ALLOW_NON_PORTABLE,
            shared_memory_mode=CudaSharedMemoryMode.ALLOW_NON_PORTABLE,
        ),
    )

    launch_cuda_kernel(
        library,
        spec,
        (
            CudaPointer(0xABCD),
            CudaScalar(CudaScalarType.I32, -9),
        ),
        stream=stream,
    )

    assert len(driver.calls) == 1
    assert driver.function_attribute_calls == [
        ("function-handle", 8, 65_536)
    ]
    call = driver.calls[0]
    assert call["function"] == "function-handle"
    assert call["kernel_params"] == 0
    tokens = call["tokens"]
    assert isinstance(tokens, tuple)
    assert len(tokens) == 5
    assert (tokens[0], tokens[2], tokens[4]) == (1, 2, None)
    assert isinstance(tokens[1], int) and tokens[1] != 0
    assert isinstance(tokens[3], int) and tokens[3] != 0
    assert call["parameter_bytes"] == struct.pack("<Qi", 0xABCD, -9)

    config = call["config"]
    assert isinstance(config, _FakeLaunchConfig)
    assert (config.gridDimX, config.gridDimY, config.gridDimZ) == (12, 2, 1)
    assert (config.blockDimX, config.blockDimY, config.blockDimZ) == (256, 1, 1)
    assert config.sharedMemBytes == 65_536
    assert config.hStream is stream
    assert config.numAttrs == 7
    attrs = {attribute.id: attribute.value for attribute in config.attrs}
    cluster = attrs[4].clusterDim
    assert (cluster.x, cluster.y, cluster.z) == (2, 1, 1)
    assert attrs[5].clusterSchedulingPolicyPreference == 2
    assert attrs[2].cooperative == 1
    assert attrs[6].programmaticStreamSerializationAllowed == 1
    assert attrs[8].priority == -3
    assert attrs[17].portableClusterSizeMode == 2
    assert attrs[19].sharedMemoryMode == 2


def test_zero_parameter_launch_uses_default_stream_and_no_extra_buffer() -> None:
    driver = _FakeDriver()
    library = _library(driver, ())
    spec = CudaLaunchSpec(
        kernel="sealed_kernel",
        grid=CudaDim3(1),
        block=CudaDim3(1),
    )

    launch_cuda_kernel(library, spec, (), stream=None)

    call = driver.calls[0]
    assert call["extra_pointer"] == 0
    assert call["parameter_bytes"] == b""
    config = call["config"]
    assert isinstance(config, _FakeLaunchConfig)
    assert isinstance(config.hStream, _FakeStream)
    assert config.hStream.address == 0
    assert config.attrs == []
    assert config.numAttrs == 0
    assert driver.function_attribute_calls == []


def test_launch_rejects_foreign_stream_closed_library_and_driver_failure() -> None:
    driver = _FakeDriver()
    library = _library(driver, ())
    spec = CudaLaunchSpec(
        kernel="sealed_kernel",
        grid=CudaDim3(1),
        block=CudaDim3(1),
    )

    with pytest.raises(CudaLaunchError, match="exact handle"):
        launch_cuda_kernel(library, spec, (), stream=object())
    assert driver.calls == []

    library.close()
    with pytest.raises(CudaLaunchError, match="closed"):
        launch_cuda_kernel(library, spec, (), stream=None)

    failing_driver = _FakeDriver(launch_status=700)
    failing_library = _library(failing_driver, ())
    with pytest.raises(CudaLaunchError, match="status 700"):
        launch_cuda_kernel(failing_library, spec, (), stream=None)


def test_dynamic_shared_memory_opt_in_failure_prevents_launch() -> None:
    driver = _FakeDriver(function_attribute_status=1)
    library = _library(driver, ())
    spec = CudaLaunchSpec(
        kernel="sealed_kernel",
        grid=CudaDim3(1),
        block=CudaDim3(1),
        shared_mem_bytes=231_424,
    )

    with pytest.raises(CudaLaunchError, match="dynamic shared memory failed"):
        launch_cuda_kernel(library, spec, (), stream=None)

    assert driver.function_attribute_calls == [
        ("function-handle", 8, 231_424)
    ]
    assert driver.calls == []
