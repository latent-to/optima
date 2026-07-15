from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from optima.cuda_cubin import (
    CudaCubinABI,
    CudaCubinCleanupError,
    CudaCubinContract,
    CudaCubinError,
    CudaCubinLibrary,
    CudaKernelABI,
    CudaKernelContract,
    CudaKernelParameter,
    cuda_cubin_identity,
)


class _Status:
    # cuda.bindings 13.x enums expose an integer ``value`` but are not
    # guaranteed to be ``IntEnum`` instances.
    def __init__(self, value: int) -> None:
        self.value = value


class _CUresult:
    CUDA_SUCCESS = _Status(0)
    CUDA_ERROR_INVALID_VALUE = _Status(1)


_KernelRow = tuple[str, bytes, tuple[tuple[object, object], ...]]


class _FakeDriver:
    CUresult = _CUresult

    def __init__(self, kernels: tuple[_KernelRow, ...]) -> None:
        self.kernels = kernels
        self.library = object()
        self.init_calls: list[int] = []
        self.load_calls: list[tuple[object, ...]] = []
        self.enumerate_calls: list[tuple[int, object]] = []
        self.unload_calls: list[object] = []
        self.function_calls: list[str] = []
        self.count_override: object | None = None
        self.terminal_result: tuple[object, ...] | None = None
        self.unload_status: object = self.CUresult.CUDA_SUCCESS

    def _row(self, handle: str) -> _KernelRow:
        return next(row for row in self.kernels if row[0] == handle)

    def cuInit(self, flags: int) -> tuple[object]:
        self.init_calls.append(flags)
        return (self.CUresult.CUDA_SUCCESS,)

    def cuLibraryLoadData(
        self,
        code: object,
        jit_options: object,
        jit_option_values: object,
        num_jit_options: object,
        library_options: object,
        library_option_values: object,
        num_library_options: object,
    ) -> tuple[object, object]:
        self.load_calls.append(
            (
                code,
                jit_options,
                jit_option_values,
                num_jit_options,
                library_options,
                library_option_values,
                num_library_options,
            )
        )
        return self.CUresult.CUDA_SUCCESS, self.library

    def cuLibraryGetKernelCount(
        self, library: object
    ) -> tuple[object, object]:
        assert library is self.library
        count = len(self.kernels)
        if self.count_override is not None:
            count = self.count_override  # type: ignore[assignment]
        return self.CUresult.CUDA_SUCCESS, count

    def cuLibraryEnumerateKernels(
        self, count: int, library: object
    ) -> tuple[object, list[str]]:
        self.enumerate_calls.append((count, library))
        return self.CUresult.CUDA_SUCCESS, [row[0] for row in self.kernels]

    def cuKernelGetName(self, handle: str) -> tuple[object, bytes]:
        return self.CUresult.CUDA_SUCCESS, self._row(handle)[1]

    def cuKernelGetParamInfo(
        self, handle: str, index: int
    ) -> tuple[object, ...]:
        parameters = self._row(handle)[2]
        if index >= len(parameters):
            if self.terminal_result is not None:
                return self.terminal_result
            return self.CUresult.CUDA_ERROR_INVALID_VALUE, 0, 0
        offset, size = parameters[index]
        return self.CUresult.CUDA_SUCCESS, offset, size

    def cuKernelGetFunction(self, handle: str) -> tuple[object, str]:
        self.function_calls.append(handle)
        return self.CUresult.CUDA_SUCCESS, f"function:{handle}"

    def cuLibraryUnload(self, library: object) -> tuple[object]:
        self.unload_calls.append(library)
        return (self.unload_status,)


def _cubin(label: bytes = b"image", *, elf_type: int = 2) -> bytes:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4:7] = bytes((2, 1, 1))  # ELF64, little-endian, current version.
    header[16:18] = elf_type.to_bytes(2, "little")
    header[18:20] = (190).to_bytes(2, "little")  # EM_CUDA.
    header[20:24] = (1).to_bytes(4, "little")
    header[52:54] = (64).to_bytes(2, "little")
    return bytes(header) + label


def _default_rows() -> tuple[_KernelRow, ...]:
    # Deliberately return driver handles in non-canonical name order.
    return (
        ("handle-zeta", b"zeta_kernel", ((0, 8),)),
        ("handle-alpha", b"alpha_kernel", ((0, 8), (16, 4))),
    )


def _expected_abi(
    cubin: bytes,
    rows: tuple[_KernelRow, ...] | None = None,
) -> CudaCubinABI:
    kernels = []
    for _handle, raw_name, parameters in rows or _default_rows():
        kernels.append(
            CudaKernelABI(
                name=raw_name.decode("ascii"),
                parameters=tuple(
                    CudaKernelParameter(
                        index=index,
                        offset=offset,  # type: ignore[arg-type]
                        size=size,  # type: ignore[arg-type]
                    )
                    for index, (offset, size) in enumerate(parameters)
                ),
            )
        )
    return CudaCubinABI(
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        cubin_size=len(cubin),
        kernels=tuple(sorted(kernels, key=lambda row: row.name)),
    )


def test_inspect_abi_seals_complete_sorted_kernel_and_parameter_inventory() -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())

    abi = CudaCubinLibrary.inspect_abi(cubin, driver=driver)

    assert abi == _expected_abi(cubin)
    assert tuple(abi.by_name) == ("alpha_kernel", "zeta_kernel")
    assert abi.by_name["alpha_kernel"].parameters == (
        CudaKernelParameter(index=0, offset=0, size=8),
        CudaKernelParameter(index=1, offset=16, size=4),
    )
    assert abi.by_name["alpha_kernel"].parameter_buffer_size == 20
    assert CudaCubinABI.from_dict(abi.to_dict()) == abi
    assert driver.init_calls == [0]
    assert driver.load_calls == [(cubin, None, None, 0, None, None, 0)]
    assert driver.enumerate_calls == [(2, driver.library)]
    assert driver.unload_calls == [driver.library]


def test_gpu_free_contract_round_trip_projects_exact_symbols_and_ordinal_widths() -> None:
    cubin = _cubin()
    abi = _expected_abi(cubin)
    contract = abi.contract

    assert contract.kernels == (
        CudaKernelContract("alpha_kernel", (8, 4)),
        CudaKernelContract("zeta_kernel", (8,)),
    )
    assert CudaCubinContract.from_dict(contract.to_dict()) == contract
    assert len(contract.digest) == 64
    raw, digest, size = cuda_cubin_identity(memoryview(cubin))
    assert (raw, digest, size) == (cubin, abi.cubin_sha256, len(cubin))


def test_open_contract_observes_and_retains_the_same_exact_library_handle() -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())
    contract = _expected_abi(cubin).contract

    library = CudaCubinLibrary.open_contract(
        cubin,
        expected_contract=contract,
        driver=driver,
    )

    assert library.abi.contract == contract
    assert driver.load_calls == [(cubin, None, None, 0, None, None, 0)]
    assert driver.unload_calls == []
    assert library.function("alpha_kernel") == "function:handle-alpha"
    library.close()
    assert driver.unload_calls == [driver.library]


def test_open_contract_unloads_an_observed_ordinal_width_mismatch() -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())
    expected = _expected_abi(cubin).contract
    alpha, zeta = expected.kernels
    mismatched = replace(
        expected,
        kernels=(replace(alpha, parameter_sizes=(8, 8)), zeta),
    )

    with pytest.raises(CudaCubinError, match="contract differs"):
        CudaCubinLibrary.open_contract(
            cubin,
            expected_contract=mismatched,
            driver=driver,
        )

    assert driver.unload_calls == [driver.library]


def test_open_ordered_contract_accepts_generated_names_and_retains_handle() -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())
    observed = _expected_abi(cubin).contract
    logical = replace(
        observed,
        kernels=(
            CudaKernelContract("entry_000", (8, 4)),
            CudaKernelContract("entry_001", (8,)),
        ),
    )

    library = CudaCubinLibrary.open_ordered_contract(
        cubin,
        expected_contract=logical,
        driver=driver,
    )

    assert library.abi.contract == observed
    assert library.function("alpha_kernel") == "function:handle-alpha"
    assert driver.unload_calls == []
    library.close()
    assert driver.unload_calls == [driver.library]


@pytest.mark.parametrize(
    "kernels",
    (
        (CudaKernelContract("entry_000", (8,)),),
        (
            CudaKernelContract("entry_000", (8,)),
            CudaKernelContract("entry_001", (8, 4)),
        ),
    ),
)
def test_open_ordered_contract_rejects_count_or_per_ordinal_width_mismatch(
    kernels: tuple[CudaKernelContract, ...],
) -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())
    observed = _expected_abi(cubin).contract
    logical = replace(observed, kernels=kernels)

    with pytest.raises(CudaCubinError, match="ordinal widths differ"):
        CudaCubinLibrary.open_ordered_contract(
            cubin,
            expected_contract=logical,
            driver=driver,
        )

    assert driver.unload_calls == [driver.library]


def test_open_rejects_wrong_bytes_before_initializing_driver() -> None:
    sealed_cubin = _cubin(b"sealed")
    different_cubin = _cubin(b"different")
    driver = _FakeDriver(_default_rows())

    with pytest.raises(CudaCubinError, match="differ from sealed identity"):
        CudaCubinLibrary.open(
            different_cubin,
            expected_abi=_expected_abi(sealed_cubin),
            driver=driver,
        )

    assert driver.init_calls == []
    assert driver.load_calls == []


def test_open_requires_sealed_abi_and_direct_construction_is_closed() -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())

    with pytest.raises(CudaCubinError, match="expected CUDA CUBIN ABI"):
        CudaCubinLibrary.open(
            cubin,
            expected_abi=None,  # type: ignore[arg-type]
            driver=driver,
        )
    with pytest.raises(CudaCubinError, match=r"sealed open\(\) admission"):
        CudaCubinLibrary(
            driver=driver,
            library=driver.library,
            abi=_expected_abi(cubin),
            kernels={"alpha_kernel": "handle-alpha"},
        )

    assert driver.init_calls == []
    assert driver.load_calls == []


def test_open_exposes_only_exactly_sealed_handles_and_closes_once() -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())
    expected = _expected_abi(cubin)

    library = CudaCubinLibrary.open(
        cubin, expected_abi=expected, driver=driver
    )

    assert library.abi == expected
    assert library.kernel("alpha_kernel") == "handle-alpha"
    assert library.function("zeta_kernel") == "function:handle-zeta"
    with pytest.raises(CudaCubinError, match="no sealed kernel"):
        library.function("missing_kernel")

    library.close()
    library.close()
    assert library.closed
    assert driver.unload_calls == [driver.library]
    with pytest.raises(CudaCubinError, match="library is closed"):
        library.kernel("alpha_kernel")
    with pytest.raises(CudaCubinError, match="library is closed"):
        library.function("alpha_kernel")


@pytest.mark.parametrize(
    "mismatch",
    ("name", "kernel_count", "parameter_count", "offset", "size"),
)
def test_open_rejects_every_kernel_abi_mismatch_and_unloads(
    mismatch: str,
) -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())
    expected = _expected_abi(cubin)
    alpha, zeta = expected.kernels
    if mismatch == "name":
        expected = replace(
            expected,
            kernels=(
                replace(alpha, name="altered_kernel"),
                zeta,
            ),
        )
    elif mismatch == "kernel_count":
        expected = replace(expected, kernels=(alpha,))
    else:
        first, second = alpha.parameters
        if mismatch == "parameter_count":
            expected = replace(
                expected,
                kernels=(replace(alpha, parameters=(first,)), zeta),
            )
        elif mismatch == "offset":
            second = replace(second, offset=24)
        else:
            second = replace(second, size=8)
        if mismatch != "parameter_count":
            expected = replace(
                expected,
                kernels=(replace(alpha, parameters=(first, second)), zeta),
            )

    with pytest.raises(CudaCubinError, match="differs from sealed authority"):
        CudaCubinLibrary.open(
            cubin, expected_abi=expected, driver=driver
        )

    assert driver.unload_calls == [driver.library]


def test_duplicate_observed_kernel_name_is_rejected_and_unloaded() -> None:
    rows: tuple[_KernelRow, ...] = (
        ("handle-a", b"same_kernel", ((0, 8),)),
        ("handle-b", b"same_kernel", ((0, 8),)),
    )
    driver = _FakeDriver(rows)

    with pytest.raises(CudaCubinError, match="repeats a kernel symbol"):
        CudaCubinLibrary.inspect_abi(_cubin(), driver=driver)

    assert driver.unload_calls == [driver.library]


@pytest.mark.parametrize("malformed", ("count", "offset", "terminal_arity"))
def test_malformed_driver_abi_results_fail_closed_and_unload(
    malformed: str,
) -> None:
    rows = _default_rows()
    if malformed == "offset":
        rows = (
            ("handle-zeta", b"zeta_kernel", ((0.0, 8),)),
            rows[1],
        )
    driver = _FakeDriver(rows)
    if malformed == "count":
        driver.count_override = 2.0
    elif malformed == "terminal_arity":
        driver.terminal_result = (
            driver.CUresult.CUDA_ERROR_INVALID_VALUE,
            0,
        )

    with pytest.raises(CudaCubinError, match="malformed|arity mismatch"):
        CudaCubinLibrary.inspect_abi(_cubin(), driver=driver)

    assert driver.unload_calls == [driver.library]


def test_unload_error_permanently_closes_handle_access() -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())
    library = CudaCubinLibrary.open(
        cubin,
        expected_abi=_expected_abi(cubin),
        driver=driver,
    )
    driver.unload_status = _Status(999)

    with pytest.raises(CudaCubinCleanupError, match="cuLibraryUnload failed"):
        library.close()

    assert library.closed
    assert driver.unload_calls == [driver.library]
    with pytest.raises(CudaCubinError, match="library is closed"):
        library.function("alpha_kernel")
    # The handle state is indeterminate after an unload error; close is
    # intentionally idempotent instead of retrying it.
    library.close()
    assert driver.unload_calls == [driver.library]


def test_inspection_does_not_publish_abi_when_unload_fails() -> None:
    driver = _FakeDriver(_default_rows())
    driver.unload_status = _Status(999)

    with pytest.raises(CudaCubinCleanupError, match="cuLibraryUnload failed"):
        CudaCubinLibrary.inspect_abi(_cubin(), driver=driver)

    assert driver.unload_calls == [driver.library]


def test_admission_failure_plus_unload_failure_is_worker_fatal() -> None:
    cubin = _cubin()
    driver = _FakeDriver(_default_rows())
    expected = _expected_abi(cubin)
    alpha, zeta = expected.kernels
    expected = replace(expected, kernels=(alpha, replace(zeta, name="other")))
    driver.unload_status = _Status(999)

    with pytest.raises(
        CudaCubinCleanupError,
        match="admission failed.*worker must terminate",
    ):
        CudaCubinLibrary.open(
            cubin,
            expected_abi=expected,
            driver=driver,
        )

    assert driver.unload_calls == [driver.library]


@pytest.mark.parametrize(
    "payload",
    (
        b".version 8.8\n.target sm_103a\n",
        b"\x50\xed\x55\xba" + b"\x00" * 60,
        b"\x7fELF" + b"\x00" * 60,
        _cubin(elf_type=1),
    ),
)
def test_non_cubin_driver_jit_inputs_are_rejected_before_driver_use(
    payload: bytes,
) -> None:
    driver = _FakeDriver(_default_rows())

    with pytest.raises(CudaCubinError, match="ELF64 CUDA image"):
        CudaCubinLibrary.inspect_abi(payload, driver=driver)

    assert driver.init_calls == []
    assert driver.load_calls == []


def test_driver_surface_is_validated_before_initialization_or_load() -> None:
    driver = _FakeDriver(_default_rows())
    driver.cuKernelGetFunction = None  # type: ignore[method-assign]

    with pytest.raises(CudaCubinError, match="lacks callable cuKernelGetFunction"):
        CudaCubinLibrary.inspect_abi(_cubin(), driver=driver)

    assert driver.init_calls == []
    assert driver.load_calls == []


def test_sealed_abi_rejects_lossy_numeric_and_noncanonical_inventory() -> None:
    cubin = _cubin()
    expected = _expected_abi(cubin)

    with pytest.raises(CudaCubinError, match="parameter offset"):
        CudaKernelParameter(index=0, offset=0.0, size=8)  # type: ignore[arg-type]
    with pytest.raises(CudaCubinError, match="CUBIN size"):
        replace(expected, cubin_size=True)
    with pytest.raises(CudaCubinError, match="inventory is not canonical"):
        replace(expected, kernels=tuple(reversed(expected.kernels)))
