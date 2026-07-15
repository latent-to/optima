from __future__ import annotations

import threading
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

import optima.artifact_device_launch as device_launch
from optima.artifact_abi import ArtifactBinding, SILU_AND_MUL_CALL_ABI
from optima.artifact_device_launch import (
    DeviceArtifactEntry,
    DeviceArtifactRuntime,
    DeviceDim3Plan,
    DeviceLaunchError,
    DeviceLaunchInvocation,
    DeviceLaunchPlan,
)
from optima.artifact_runtime import (
    ArtifactRuntimeEntry,
    ArtifactRuntimeError,
    ArtifactRuntimeProvider,
    ArtifactRuntimeStep,
)
from optima.cuda_cubin import (
    CudaCubinABI,
    CudaCubinContract,
    CudaCubinLibrary,
    CudaKernelABI,
    CudaKernelContract,
    CudaKernelParameter,
)
from optima.cuda_launch import CudaLaunchAttributes, CudaPointer
from optima.cuda_materialize import (
    CudaCheckedExpression,
    CudaExpressionNode,
    CudaParameterPlan,
    make_cuda_primitive_registry,
)


def _const(value: bool | int | float) -> CudaCheckedExpression:
    return CudaCheckedExpression((CudaExpressionNode("const", value=value),), 0)


def _tensor_dim(binding: int, axis: int) -> CudaCheckedExpression:
    return CudaCheckedExpression(
        (CudaExpressionNode("tensor_dim", binding=binding, axis=axis),), 0
    )


def _dim3(x: CudaCheckedExpression) -> DeviceDim3Plan:
    return DeviceDim3Plan(x, _const(1), _const(1))


def _invocation(
    parameters: tuple[CudaParameterPlan, ...],
    *,
    kernel: str = "run",
    stream_binding: int | None = None,
) -> DeviceLaunchInvocation:
    return DeviceLaunchInvocation(
        ordinal=0,
        kernel=kernel,
        grid=_dim3(_tensor_dim(0, 0)),
        block=_dim3(_const(256)),
        cluster=None,
        shared_mem_bytes=_const(0),
        parameters=parameters,
        stream_binding=stream_binding,
        attributes=CudaLaunchAttributes(),
    )


def _plan(
    parameters: tuple[CudaParameterPlan, ...],
    *,
    auxiliary: bool = False,
) -> DeviceLaunchPlan:
    kernels = [
        CudaKernelContract("run", tuple(parameter.size for parameter in parameters))
    ]
    if auxiliary:
        kernels.append(CudaKernelContract("auxiliary", (4,)))
    return DeviceLaunchPlan(
        kernels=tuple(sorted(kernels, key=lambda row: row.name)),
        launches=(_invocation(parameters),),
    )


def test_plan_round_trip_digest_preserves_signed_zero_and_auxiliary_symbols() -> None:
    scalar = CudaParameterPlan(
        kind="scalar",
        size=8,
        scalar_type="f64",
        expression=_const(-0.0),
    )
    plan = _plan((scalar,), auxiliary=True)
    encoded = plan.to_dict()

    assert encoded["launches"][0]["parameters"][0]["expression"]["nodes"][0][
        "value"
    ] == {"f64_hex": "-0x0.0p+0"}
    assert DeviceLaunchPlan.from_dict(encoded) == plan
    assert plan.digest == DeviceLaunchPlan.from_dict(encoded).digest
    assert len(plan.digest) == 64
    assert tuple(kernel.name for kernel in plan.cuda_contract("1" * 64, 64).kernels) == (
        "auxiliary",
        "run",
    )


def test_invocation_from_dict_allows_omitted_optional_cluster_only() -> None:
    encoded = _invocation(()).to_dict()
    encoded.pop("cluster")

    parsed = DeviceLaunchInvocation.from_dict(encoded)

    assert parsed.cluster is None
    assert parsed == _invocation(())

    missing_required = dict(encoded)
    missing_required.pop("grid")
    with pytest.raises(DeviceLaunchError, match="fields mismatch"):
        DeviceLaunchInvocation.from_dict(missing_required)

    with_unknown = {**encoded, "miner_extension": {}}
    with pytest.raises(DeviceLaunchError, match="fields mismatch"):
        DeviceLaunchInvocation.from_dict(with_unknown)


def test_plan_rejects_referenced_missing_symbol_but_allows_sealed_auxiliary_symbol() -> None:
    pointer = CudaParameterPlan(kind="pointer", size=8, binding=0)
    assert _plan((pointer,), auxiliary=True).launches[0].kernel == "run"

    with pytest.raises(DeviceLaunchError, match="undeclared CUBIN symbols"):
        DeviceLaunchPlan(
            kernels=(CudaKernelContract("run", (8,)),),
            launches=(_invocation((pointer,), kernel="missing"),),
        )


def test_plan_binds_generated_symbols_by_ordinal_even_with_duplicate_widths() -> None:
    pointer = CudaParameterPlan(kind="pointer", size=8, binding=0)
    plan = DeviceLaunchPlan(
        kernels=(
            CudaKernelContract("logical_a", (8,)),
            CudaKernelContract("logical_b", (8,)),
        ),
        launches=(_invocation((pointer,), kernel="logical_b"),),
    )
    observed = CudaCubinABI(
        "1" * 64,
        64,
        (
            CudaKernelABI(
                "generated_alpha",
                (CudaKernelParameter(index=0, offset=0, size=8),),
            ),
            CudaKernelABI(
                "generated_zeta",
                (CudaKernelParameter(index=0, offset=16, size=8),),
            ),
        ),
    ).contract

    resolved = plan.bind_observed_contract(observed)

    assert resolved.kernels == observed.kernels
    assert resolved.launches[0].kernel == "generated_zeta"
    assert plan.launches[0].kernel == "logical_b"


def test_plan_rejects_width_multiset_that_differs_by_ordinal() -> None:
    pointer = CudaParameterPlan(kind="pointer", size=8, binding=0)
    plan = DeviceLaunchPlan(
        kernels=(
            CudaKernelContract("logical_a", (4,)),
            CudaKernelContract("logical_b", (8,)),
        ),
        launches=(_invocation((pointer,), kernel="logical_b"),),
    )
    observed = CudaCubinABI(
        "1" * 64,
        64,
        (
            CudaKernelABI(
                "generated_alpha",
                (CudaKernelParameter(index=0, offset=0, size=8),),
            ),
            CudaKernelABI(
                "generated_zeta",
                (CudaKernelParameter(index=0, offset=0, size=4),),
            ),
        ),
    ).contract

    with pytest.raises(DeviceLaunchError, match="ordinal widths differ"):
        plan.bind_observed_contract(observed)


def test_plan_statically_joins_tensor_stream_and_parameter_widths() -> None:
    parameters = (
        CudaParameterPlan(kind="pointer", size=8, binding=0),
        CudaParameterPlan(
            kind="scalar",
            size=4,
            scalar_type="i32",
            expression=_tensor_dim(0, 0),
        ),
    )
    plan = DeviceLaunchPlan(
        kernels=(CudaKernelContract("run", (8, 4)),),
        launches=(_invocation(parameters, stream_binding=1),),
    )
    plan.validate_bindings(
        (SimpleNamespace(kind="tensor"), SimpleNamespace(kind="stream")),
        provider_capabilities=(),
    )

    with pytest.raises(DeviceLaunchError, match="sealed stream binding"):
        plan.validate_bindings(
            (SimpleNamespace(kind="tensor"), SimpleNamespace(kind="scalar")),
            provider_capabilities=(),
        )
    with pytest.raises(DeviceLaunchError, match="parameter widths"):
        DeviceLaunchPlan(
            kernels=(CudaKernelContract("run", (8, 8)),),
            launches=(_invocation(parameters),),
        )


class _Driver:
    class CUresult:
        CUDA_SUCCESS = 0

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.library = object()

    def cuLibraryUnload(self, library: object) -> tuple[int]:
        assert library is self.library
        self.events.append("unload")
        return (0,)


def _library(driver: _Driver, plan: DeviceLaunchPlan) -> CudaCubinLibrary:
    kernels: list[CudaKernelABI] = []
    handles: dict[str, object] = {}
    for contract in plan.kernels:
        offset = 0
        parameters = []
        for ordinal, size in enumerate(contract.parameter_sizes):
            parameters.append(CudaKernelParameter(ordinal, offset, size))
            offset += size
        kernels.append(CudaKernelABI(contract.name, tuple(parameters)))
        handles[contract.name] = object()
    abi = CudaCubinABI("1" * 64, 64, tuple(kernels))
    library = object.__new__(CudaCubinLibrary)
    library._driver = driver  # type: ignore[attr-defined]
    library._library = driver.library  # type: ignore[attr-defined]
    library._abi = abi  # type: ignore[attr-defined]
    library._kernels = MappingProxyType(handles)  # type: ignore[attr-defined]
    library._closed = False  # type: ignore[attr-defined]
    library._lock = threading.RLock()  # type: ignore[attr-defined]
    return library


class _Tensor:
    def __init__(self, shape: tuple[int, ...], pointer: int) -> None:
        self.shape = shape
        self._pointer = pointer

    def data_ptr(self) -> int:
        return self._pointer

    def stride(self) -> tuple[int, ...]:
        return (1,) * len(self.shape)

    def element_size(self) -> int:
        return 2

    def storage_offset(self) -> int:
        return 0


def _provider(stream: object) -> ArtifactRuntimeProvider:
    return ArtifactRuntimeProvider(
        provider="test.device.v1",
        tensor_descriptor=lambda value, _binding: value,
        tensor_pointer=lambda value, _binding: value.data_ptr(),
        current_stream=lambda: stream,
        group_rank=lambda _group: 0,
        group_size=lambda _group: 1,
        group_pointer=lambda _group, _projection, _binding, _peer: 0,
        pointer_identity=lambda value, _binding: value,
    )


def test_artifact_entry_calls_device_runtime_with_live_tensors_and_orders_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    driver = _Driver(events)
    parameters = (
        CudaParameterPlan(kind="pointer", size=8, binding=0),
        CudaParameterPlan(kind="pointer", size=8, binding=1),
    )
    invocation = _invocation(parameters, stream_binding=2)
    plan = DeviceLaunchPlan(
        kernels=(CudaKernelContract("run", (8, 8)),),
        launches=(invocation,),
    )
    registry = make_cuda_primitive_registry(
        driver=driver,
        tma_descriptor=lambda _descriptor: bytes(128),
        synchronize=lambda: events.append("sync"),
    )
    runtime = DeviceArtifactRuntime(_library(driver, plan), plan, registry)
    launches: list[tuple[object, ...]] = []

    def capture_launch(library, spec, materialized, *, stream):
        launches.append((library, spec, materialized, stream))

    monkeypatch.setattr(device_launch, "launch_cuda_kernel", capture_launch)
    stream = object()
    bindings = (
        ArtifactBinding("input.x", "tensor"),
        ArtifactBinding("output.out", "tensor"),
        ArtifactBinding("stream.current", "stream"),
    )
    plan.validate_bindings(bindings, provider_capabilities=())
    entry = ArtifactRuntimeEntry(
        call_abi=SILU_AND_MUL_CALL_ABI,
        steps=(
            ArtifactRuntimeStep(
                name="device-run",
                plan="default",
                step=0,
                role="run",
                bindings=bindings,
                specializes=(),
                prelaunch=(),
                executor=runtime,
            ),
        ),
        provider=_provider(stream),
    )
    wrapped = DeviceArtifactEntry(entry, (runtime,))
    source = _Tensor((513,), 0x1000)
    output = _Tensor((513,), 0x2000)

    wrapped(source, output)

    assert len(launches) == 1
    _, spec, materialized, observed_stream = launches[0]
    assert spec.grid.x == 513
    assert materialized == (CudaPointer(0x1000), CudaPointer(0x2000))
    assert observed_stream is stream
    assert runtime.admission.cubin_sha256 == "1" * 64
    assert len(runtime.admission.observed_abi_digest) == 64
    assert len(runtime.admission.observed_contract_digest) == 64

    wrapped.close()
    wrapped.close()
    assert events == ["sync", "unload"]
    assert runtime.closed
    with pytest.raises(ArtifactRuntimeError, match="closing or closed"):
        wrapped(source, output)


def test_admit_launches_driver_observed_symbol_from_logical_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    driver = _Driver(events)
    pointer = CudaParameterPlan(kind="pointer", size=8, binding=0)
    invocation = replace(
        _invocation((pointer,), kernel="entry_000"),
        grid=_dim3(_const(1)),
    )
    declared = DeviceLaunchPlan(
        kernels=(CudaKernelContract("entry_000", (8,)),),
        launches=(invocation,),
    )
    registry = make_cuda_primitive_registry(
        driver=driver,
        tma_descriptor=lambda _descriptor: bytes(128),
        synchronize=lambda: events.append("sync"),
    )
    physical = CudaCubinContract(
        cubin_sha256="1" * 64,
        cubin_size=64,
        kernels=(CudaKernelContract("generated_kernel", (8,)),),
    )
    library = _library(driver, declared.bind_observed_contract(physical))
    admitted: list[CudaCubinContract] = []

    def open_ordered(
        cls, cubin, *, expected_contract, driver
    ) -> CudaCubinLibrary:
        assert cls is CudaCubinLibrary
        assert cubin == b"sealed"
        assert driver is registry.driver
        admitted.append(expected_contract)
        return library

    monkeypatch.setattr(
        device_launch,
        "cuda_cubin_identity",
        lambda cubin: (b"sealed", "1" * 64, 64),
    )
    monkeypatch.setattr(
        CudaCubinLibrary,
        "open_ordered_contract",
        classmethod(open_ordered),
    )
    runtime = DeviceArtifactRuntime.admit(b"sealed", declared, registry)
    launches: list[tuple[object, ...]] = []

    def capture_launch(library, spec, materialized, *, stream):
        launches.append((library, spec, materialized, stream))

    monkeypatch.setattr(device_launch, "launch_cuda_kernel", capture_launch)
    runtime(0x1000)

    assert len(launches) == 1
    assert admitted == [declared.cuda_contract("1" * 64, 64)]
    assert launches[0][1].kernel == "generated_kernel"
    assert launches[0][2] == (CudaPointer(0x1000),)
    assert (
        runtime.admission.observed_contract_digest
        == runtime._library.abi.contract.digest
    )

    runtime.close()
    assert events == ["sync", "unload"]


def test_entry_close_failure_retains_device_handle_without_synchronizing() -> None:
    events: list[str] = []
    driver = _Driver(events)
    plan = _plan((CudaParameterPlan(kind="pointer", size=8, binding=0),))
    registry = make_cuda_primitive_registry(
        driver=driver,
        tma_descriptor=lambda _descriptor: bytes(128),
        synchronize=lambda: events.append("sync"),
    )
    runtime = DeviceArtifactRuntime(_library(driver, plan), plan, registry)

    class FailingEntry:
        def __call__(self, *_args: object) -> None:
            return None

        def close(self) -> None:
            events.append("entry-close")
            raise RuntimeError("destroy failed")

    with pytest.raises(DeviceLaunchError, match="runtimes remain loaded"):
        DeviceArtifactEntry(FailingEntry(), (runtime,)).close()

    assert events == ["entry-close"]
    assert not runtime.closed


def test_synchronize_failure_retains_library_and_requires_worker_death() -> None:
    events: list[str] = []
    driver = _Driver(events)
    plan = _plan((CudaParameterPlan(kind="pointer", size=8, binding=0),))

    def fail_sync() -> None:
        events.append("sync")
        raise RuntimeError("device lost")

    registry = make_cuda_primitive_registry(
        driver=driver,
        tma_descriptor=lambda _descriptor: bytes(128),
        synchronize=fail_sync,
    )
    runtime = DeviceArtifactRuntime(_library(driver, plan), plan, registry)

    with pytest.raises(DeviceLaunchError, match="remains loaded"):
        runtime.close()

    assert events == ["sync"]
    assert not runtime.closed
