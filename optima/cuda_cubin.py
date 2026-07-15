"""Validator-owned CUDA CUBIN reopen and device-parameter authority.

Core artifact providers may execute candidate *device* code, but they must not
load a candidate-produced host object or launcher.  This module is the narrow
CUDA Library/Driver boundary for that rule.  It lazily loads a sealed CUBIN,
enumerates every device kernel, and asks the CUDA driver for each formal
parameter's device-side offset and size.  A later declarative launch runtime may
obtain a ``CUfunction`` only after that independently observed ABI matches the
sealed validator expectation.

Importing this module is CUDA-free.  The driver binding is captured only by
``CudaCubinLibrary.inspect_abi`` or ``CudaCubinLibrary.open`` inside an isolated
GPU worker; intake and trusted controllers can parse the JSON-shaped ABI types
without importing CUDA Python.
"""

from __future__ import annotations

import hashlib
import operator
import re
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from optima.stack_identity import canonical_digest, require_sha256_hex


CUDA_CUBIN_ABI_SCHEMA = "optima.cuda-cubin-abi.v1"
CUDA_CUBIN_CONTRACT_SCHEMA = "optima.cuda-cubin-contract.v1"

_KERNEL_NAME_RE = re.compile(r"[A-Za-z_.$][A-Za-z0-9_.$@]{0,4095}\Z")
_MAX_CUBIN_BYTES = 1 << 30
_MAX_KERNELS = 4_096
_MAX_KERNEL_PARAMETERS = 256
# CUDA's kernel parameter buffer is bounded.  Keep a conservative explicit cap
# rather than trusting a malformed image or a future driver to allocate freely.
_MAX_PARAMETER_BUFFER_BYTES = 32_764

_ELF64_HEADER_BYTES = 64
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_EV_CURRENT = 1
_ET_EXEC = 2
_EM_CUDA = 190
_LIBRARY_CONSTRUCTION_TOKEN = object()


class CudaCubinError(RuntimeError):
    """A device image, observed kernel ABI, or CUDA driver call is invalid."""


class CudaCubinCleanupError(CudaCubinError):
    """A loaded CUDA library could not be proven unloaded; kill the worker."""


def _cubin_bytes(cubin: object) -> bytes:
    """Copy a bounded bytes-like value without allocating before the size gate."""

    if type(cubin) is bytes:
        size = len(cubin)
        raw = cubin
    elif type(cubin) is bytearray:
        size = len(cubin)
        if not 1 <= size <= _MAX_CUBIN_BYTES:
            raise CudaCubinError("CUDA CUBIN size is outside policy")
        raw = bytes(cubin)
    elif type(cubin) is memoryview:
        size = cubin.nbytes
        if not 1 <= size <= _MAX_CUBIN_BYTES:
            raise CudaCubinError("CUDA CUBIN size is outside policy")
        raw = cubin.tobytes()
    else:
        raise CudaCubinError("CUDA CUBIN must be an exact bytes-like value")
    if not 1 <= size <= _MAX_CUBIN_BYTES:
        raise CudaCubinError("CUDA CUBIN size is outside policy")
    if len(raw) != size:
        raise CudaCubinError("CUDA CUBIN bytes-like value changed size while copied")
    return raw


def cuda_cubin_identity(
    cubin: bytes | bytearray | memoryview,
) -> tuple[bytes, str, int]:
    """Return one bounded copied image and its exact content identity."""

    raw = _cubin_bytes(cubin)
    _require_elf_cubin(raw)
    return raw, hashlib.sha256(raw).hexdigest(), len(raw)


def _require_elf_cubin(raw: bytes) -> None:
    """Reject driver-JIT inputs: this boundary accepts only CUDA ELF CUBINs."""

    if (
        len(raw) < _ELF64_HEADER_BYTES
        or raw[:4] != b"\x7fELF"
        or raw[4] != _ELFCLASS64
        or raw[5] != _ELFDATA2LSB
        or raw[6] != _EV_CURRENT
        or int.from_bytes(raw[16:18], "little") != _ET_EXEC
        or int.from_bytes(raw[18:20], "little") != _EM_CUDA
        or int.from_bytes(raw[20:24], "little") != _EV_CURRENT
        or int.from_bytes(raw[52:54], "little") != _ELF64_HEADER_BYTES
    ):
        raise CudaCubinError(
            "CUDA CUBIN must be a canonical little-endian ELF64 CUDA image"
        )


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise CudaCubinError(
            f"{field} must be a lowercase 64-hex SHA-256 digest"
        )
    try:
        digest = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise CudaCubinError(str(exc)) from None
    if digest == "0" * 64:
        raise CudaCubinError(f"{field} must not be the all-zero digest")
    return digest


def _kernel_name(value: object, *, field: str) -> str:
    if type(value) is bytes:
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            raise CudaCubinError(f"{field} is not ASCII") from None
    if type(value) is not str or _KERNEL_NAME_RE.fullmatch(value) is None:
        raise CudaCubinError(f"{field} is not a canonical CUDA kernel symbol")
    return value


@dataclass(frozen=True, order=True)
class CudaKernelParameter:
    """One formal device parameter as reported by the CUDA driver."""

    index: int
    offset: int
    size: int

    def __post_init__(self) -> None:
        if (
            type(self.index) is not int
            or not 0 <= self.index < _MAX_KERNEL_PARAMETERS
        ):
            raise CudaCubinError("CUDA kernel parameter index is outside policy")
        if (
            type(self.offset) is not int
            or not 0 <= self.offset < _MAX_PARAMETER_BUFFER_BYTES
        ):
            raise CudaCubinError("CUDA kernel parameter offset is outside policy")
        if (
            type(self.size) is not int
            or not 1 <= self.size <= _MAX_PARAMETER_BUFFER_BYTES
            or self.offset + self.size > _MAX_PARAMETER_BUFFER_BYTES
        ):
            raise CudaCubinError("CUDA kernel parameter size is outside policy")

    def to_dict(self) -> dict[str, int]:
        return {"index": self.index, "offset": self.offset, "size": self.size}

    @classmethod
    def from_dict(cls, value: object) -> "CudaKernelParameter":
        if not isinstance(value, dict) or set(value) != {"index", "offset", "size"}:
            raise CudaCubinError("sealed CUDA kernel parameter fields mismatch")
        return cls(
            index=value["index"],  # type: ignore[arg-type]
            offset=value["offset"],  # type: ignore[arg-type]
            size=value["size"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaKernelABI:
    """Complete ordered formal-parameter layout for one device entry."""

    name: str
    parameters: tuple[CudaKernelParameter, ...]

    def __post_init__(self) -> None:
        name = _kernel_name(self.name, field="CUDA kernel name")
        if type(self.parameters) is not tuple or len(
            self.parameters
        ) > _MAX_KERNEL_PARAMETERS:
            raise CudaCubinError("CUDA kernel parameter set is outside policy")
        if not all(
            type(parameter) is CudaKernelParameter
            for parameter in self.parameters
        ):
            raise CudaCubinError("CUDA kernel parameter rows have the wrong type")
        indices = tuple(parameter.index for parameter in self.parameters)
        if indices != tuple(range(len(self.parameters))):
            raise CudaCubinError("CUDA kernel parameter indices are not contiguous")
        previous_end = 0
        for parameter in self.parameters:
            if parameter.offset < previous_end:
                raise CudaCubinError("CUDA kernel parameters overlap or regress")
            previous_end = parameter.offset + parameter.size
        object.__setattr__(self, "name", name)

    @property
    def parameter_buffer_size(self) -> int:
        if not self.parameters:
            return 0
        last = self.parameters[-1]
        return last.offset + last.size

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaKernelABI":
        if not isinstance(value, dict) or set(value) != {"name", "parameters"}:
            raise CudaCubinError("sealed CUDA kernel ABI fields mismatch")
        raw_parameters = value["parameters"]
        if not isinstance(raw_parameters, list):
            raise CudaCubinError("sealed CUDA kernel parameters must be a list")
        return cls(
            name=value["name"],  # type: ignore[arg-type]
            parameters=tuple(
                CudaKernelParameter.from_dict(row) for row in raw_parameters
            ),
        )


@dataclass(frozen=True)
class CudaKernelContract:
    """GPU-free contract for one kernel's ordered by-value parameter widths.

    CUDA is the authority for parameter offsets.  The compiler-side declaration
    must nevertheless bind every ordinal and its exact width, so runtime admission
    cannot reinterpret an unexpected device entry or silently ignore one.  The
    name may be a logical alias when the caller uses ordered-contract admission;
    exact-name admission continues to treat it as a physical CUDA symbol.
    """

    name: str
    parameter_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        name = _kernel_name(self.name, field="CUDA kernel contract name")
        if (
            type(self.parameter_sizes) is not tuple
            or len(self.parameter_sizes) > _MAX_KERNEL_PARAMETERS
            or any(
                type(size) is not int
                or not 1 <= size <= _MAX_PARAMETER_BUFFER_BYTES
                for size in self.parameter_sizes
            )
        ):
            raise CudaCubinError(
                "CUDA kernel contract parameter sizes are outside policy"
            )
        object.__setattr__(self, "name", name)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "parameter_sizes": list(self.parameter_sizes)}

    @classmethod
    def from_dict(cls, value: object) -> "CudaKernelContract":
        if type(value) is not dict or set(value) != {"name", "parameter_sizes"}:
            raise CudaCubinError("sealed CUDA kernel contract fields mismatch")
        sizes = value["parameter_sizes"]
        if type(sizes) is not list:
            raise CudaCubinError(
                "sealed CUDA kernel contract parameter sizes must be a list"
            )
        return cls(
            name=value["name"],  # type: ignore[arg-type]
            parameter_sizes=tuple(sizes),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaCubinContract:
    """Exact CUBIN bytes plus a complete GPU-free kernel inventory.

    ``open_contract`` interprets names as physical symbols.  The ordered path
    interprets them as logical aliases whose canonical order defines ordinals.
    """

    cubin_sha256: str
    cubin_size: int
    kernels: tuple[CudaKernelContract, ...]
    schema: str = CUDA_CUBIN_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        digest = _digest(self.cubin_sha256, field="CUDA CUBIN contract sha256")
        if (
            type(self.cubin_size) is not int
            or not 1 <= self.cubin_size <= _MAX_CUBIN_BYTES
        ):
            raise CudaCubinError("CUDA CUBIN contract size is outside policy")
        if self.schema != CUDA_CUBIN_CONTRACT_SCHEMA:
            raise CudaCubinError("CUDA CUBIN contract schema mismatch")
        if (
            type(self.kernels) is not tuple
            or not 1 <= len(self.kernels) <= _MAX_KERNELS
            or not all(type(kernel) is CudaKernelContract for kernel in self.kernels)
        ):
            raise CudaCubinError("CUDA CUBIN contract inventory is outside policy")
        names = tuple(kernel.name for kernel in self.kernels)
        if names != tuple(sorted(set(names))):
            raise CudaCubinError("CUDA CUBIN contract inventory is not canonical")
        object.__setattr__(self, "cubin_sha256", digest)

    @property
    def by_name(self) -> Mapping[str, CudaKernelContract]:
        return MappingProxyType({kernel.name: kernel for kernel in self.kernels})

    @property
    def digest(self) -> str:
        return canonical_digest("optima.cuda-cubin-contract", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "cubin_sha256": self.cubin_sha256,
            "cubin_size": self.cubin_size,
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaCubinContract":
        if type(value) is not dict or set(value) != {
            "cubin_sha256",
            "cubin_size",
            "kernels",
            "schema",
        }:
            raise CudaCubinError("sealed CUDA CUBIN contract fields mismatch")
        kernels = value["kernels"]
        if type(kernels) is not list:
            raise CudaCubinError("sealed CUDA CUBIN contract kernels must be a list")
        return cls(
            cubin_sha256=value["cubin_sha256"],  # type: ignore[arg-type]
            cubin_size=value["cubin_size"],  # type: ignore[arg-type]
            kernels=tuple(CudaKernelContract.from_dict(row) for row in kernels),
            schema=value["schema"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CudaCubinABI:
    """Canonical CUBIN content identity plus its complete kernel inventory."""

    cubin_sha256: str
    cubin_size: int
    kernels: tuple[CudaKernelABI, ...]
    schema: str = CUDA_CUBIN_ABI_SCHEMA

    def __post_init__(self) -> None:
        digest = _digest(self.cubin_sha256, field="CUDA CUBIN sha256")
        if (
            type(self.cubin_size) is not int
            or not 1 <= self.cubin_size <= _MAX_CUBIN_BYTES
        ):
            raise CudaCubinError("CUDA CUBIN size is outside policy")
        if (
            type(self.schema) is not str
            or self.schema != CUDA_CUBIN_ABI_SCHEMA
        ):
            raise CudaCubinError("CUDA CUBIN ABI schema mismatch")
        if (
            type(self.kernels) is not tuple
            or not 1 <= len(self.kernels) <= _MAX_KERNELS
            or not all(type(kernel) is CudaKernelABI for kernel in self.kernels)
        ):
            raise CudaCubinError("CUDA CUBIN kernel inventory is outside policy")
        names = tuple(kernel.name for kernel in self.kernels)
        if names != tuple(sorted(set(names))):
            raise CudaCubinError("CUDA CUBIN kernel inventory is not canonical")
        object.__setattr__(self, "cubin_sha256", digest)

    @property
    def by_name(self) -> Mapping[str, CudaKernelABI]:
        return MappingProxyType({kernel.name: kernel for kernel in self.kernels})

    @property
    def digest(self) -> str:
        return canonical_digest("optima.cuda-cubin-abi", self.to_dict())

    @property
    def contract(self) -> CudaCubinContract:
        return CudaCubinContract(
            cubin_sha256=self.cubin_sha256,
            cubin_size=self.cubin_size,
            kernels=tuple(
                CudaKernelContract(
                    name=kernel.name,
                    parameter_sizes=tuple(
                        parameter.size for parameter in kernel.parameters
                    ),
                )
                for kernel in self.kernels
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cubin_sha256": self.cubin_sha256,
            "cubin_size": self.cubin_size,
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CudaCubinABI":
        if not isinstance(value, dict) or set(value) != {
            "cubin_sha256",
            "cubin_size",
            "kernels",
            "schema",
        }:
            raise CudaCubinError("sealed CUDA CUBIN ABI fields mismatch")
        raw_kernels = value["kernels"]
        if not isinstance(raw_kernels, list):
            raise CudaCubinError("sealed CUDA CUBIN kernels must be a list")
        return cls(
            cubin_sha256=value["cubin_sha256"],  # type: ignore[arg-type]
            cubin_size=value["cubin_size"],  # type: ignore[arg-type]
            kernels=tuple(CudaKernelABI.from_dict(row) for row in raw_kernels),
            schema=value["schema"],  # type: ignore[arg-type]
        )


def _driver_integer(value: object, *, field: str) -> int:
    """Accept integer-protocol CUDA values without lossy ``int()`` coercion."""

    if isinstance(value, bool):
        raise CudaCubinError(f"CUDA driver returned a malformed {field}")
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError):
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, bool):
            raise CudaCubinError(f"CUDA driver returned a malformed {field}")
        try:
            return operator.index(enum_value)
        except (TypeError, ValueError, OverflowError):
            pass
    raise CudaCubinError(f"CUDA driver returned a malformed {field}")


def _result_code(value: object) -> int:
    return _driver_integer(value, field="status")


def _call_result(value: object, *, operation: str) -> tuple[object, ...]:
    if not isinstance(value, tuple) or not value:
        raise CudaCubinError(f"CUDA {operation} returned a malformed result")
    return value


def _driver_method(driver: object, name: str) -> Any:
    method = getattr(driver, name, None)
    if not callable(method):
        raise CudaCubinError(f"CUDA driver binding lacks callable {name}")
    return method


def _driver_call(driver: object, name: str, *args: object) -> object:
    method = _driver_method(driver, name)
    try:
        return method(*args)
    except Exception as exc:  # noqa: BLE001 - normalize the optional binding
        raise CudaCubinError(
            f"CUDA {name} binding raised {type(exc).__name__}: {exc}"
        ) from None


def _success_code(driver: object) -> int:
    try:
        return _result_code(driver.CUresult.CUDA_SUCCESS)  # type: ignore[attr-defined]
    except AttributeError:
        raise CudaCubinError("CUDA driver binding lacks CUDA_SUCCESS") from None


def _invalid_value_code(driver: object) -> int:
    try:
        raw = driver.CUresult.CUDA_ERROR_INVALID_VALUE  # type: ignore[attr-defined]
        return _result_code(raw)
    except AttributeError:
        raise CudaCubinError(
            "CUDA driver binding lacks CUDA_ERROR_INVALID_VALUE"
        ) from None


def _validate_driver(driver: object) -> None:
    # Validate every binding surface and both status constants before acquiring
    # a CUDA resource.  Otherwise a malformed binding could leak a successfully
    # loaded library while error handling itself discovers a missing method.
    _success_code(driver)
    _invalid_value_code(driver)
    for name in (
        "cuInit",
        "cuLibraryLoadData",
        "cuLibraryGetKernelCount",
        "cuLibraryEnumerateKernels",
        "cuKernelGetName",
        "cuKernelGetParamInfo",
        "cuKernelGetFunction",
        "cuLibraryUnload",
    ):
        _driver_method(driver, name)


def _require_success(
    driver: object,
    result: tuple[object, ...],
    *,
    operation: str,
    arity: int,
) -> tuple[object, ...]:
    if len(result) != arity:
        raise CudaCubinError(f"CUDA {operation} result arity mismatch")
    status = _result_code(result[0])
    if status != _success_code(driver):
        raise CudaCubinError(f"CUDA {operation} failed with status {status}")
    return result[1:]


def _unload_library(driver: object, library: object) -> None:
    try:
        result = _call_result(
            _driver_call(driver, "cuLibraryUnload", library),
            operation="cuLibraryUnload",
        )
        _require_success(
            driver,
            result,
            operation="cuLibraryUnload",
            arity=1,
        )
    except CudaCubinError as exc:
        raise CudaCubinCleanupError(
            f"{exc}; isolated CUDA worker must terminate"
        ) from exc


def _observe_kernel_parameters(
    driver: object, kernel: object
) -> tuple[CudaKernelParameter, ...]:
    parameters: list[CudaKernelParameter] = []
    for index in range(_MAX_KERNEL_PARAMETERS + 1):
        result = _call_result(
            _driver_call(driver, "cuKernelGetParamInfo", kernel, index),
            operation="cuKernelGetParamInfo",
        )
        status = _result_code(result[0])
        if status == _invalid_value_code(driver):
            if len(result) not in {1, 3}:
                raise CudaCubinError(
                    "CUDA cuKernelGetParamInfo terminal result arity mismatch"
                )
            break
        values = _require_success(
            driver,
            result,
            operation="cuKernelGetParamInfo",
            arity=3,
        )
        if index == _MAX_KERNEL_PARAMETERS:
            raise CudaCubinError("CUDA kernel exceeds the parameter-count cap")
        parameters.append(
            CudaKernelParameter(
                index=index,
                offset=_driver_integer(
                    values[0], field="kernel parameter offset"
                ),
                size=_driver_integer(
                    values[1], field="kernel parameter size"
                ),
            )
        )
    else:  # pragma: no cover - loop always breaks or raises at the sentinel row
        raise CudaCubinError("CUDA kernel parameter enumeration did not terminate")
    return tuple(parameters)


class CudaCubinLibrary:
    """One captured CUDA library handle whose complete ABI was checked."""

    __slots__ = ("_abi", "_closed", "_driver", "_kernels", "_library", "_lock")

    def __init__(
        self,
        *,
        driver: object,
        library: object,
        abi: CudaCubinABI,
        kernels: Mapping[str, object],
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _LIBRARY_CONSTRUCTION_TOKEN:
            raise CudaCubinError(
                "CUDA CUBIN library handles require sealed open() admission"
            )
        self._driver = driver
        self._library = library
        self._abi = abi
        self._kernels = MappingProxyType(dict(kernels))
        self._closed = False
        self._lock = threading.RLock()

    @classmethod
    def inspect_abi(
        cls,
        cubin: bytes | bytearray | memoryview,
        *,
        driver: object | None = None,
    ) -> CudaCubinABI:
        """Observe a CUBIN ABI and unload it without exposing launch handles.

        This is the prebuild/sealing surface.  Runtime admission must use
        :meth:`open` with the resulting validator-owned ABI.
        """

        if cls is not CudaCubinLibrary:
            raise CudaCubinError(
                "CUDA CUBIN library authority may not be subclassed"
            )
        raw = _cubin_bytes(cubin)
        _require_elf_cubin(raw)
        cubin_sha256 = hashlib.sha256(raw).hexdigest()
        captured_driver = cls._capture_driver(driver)
        library, abi, _kernels = cls._load_and_observe(
            raw,
            cubin_sha256=cubin_sha256,
            driver=captured_driver,
            expected_abi=None,
        )
        _unload_library(captured_driver, library)
        return abi

    @classmethod
    def open(
        cls,
        cubin: bytes | bytearray | memoryview,
        *,
        expected_abi: CudaCubinABI,
        driver: object | None = None,
    ) -> "CudaCubinLibrary":
        """Load device bytes and require an exact independently observed ABI."""

        if cls is not CudaCubinLibrary:
            raise CudaCubinError(
                "CUDA CUBIN library authority may not be subclassed"
            )
        raw = _cubin_bytes(cubin)
        _require_elf_cubin(raw)
        cubin_sha256 = hashlib.sha256(raw).hexdigest()
        if type(expected_abi) is not CudaCubinABI:
            raise CudaCubinError("expected CUDA CUBIN ABI has the wrong type")
        if (
            expected_abi.cubin_sha256 != cubin_sha256
            or expected_abi.cubin_size != len(raw)
        ):
            raise CudaCubinError("CUDA CUBIN bytes differ from sealed identity")

        captured_driver = cls._capture_driver(driver)
        library, abi, kernels = cls._load_and_observe(
            raw,
            cubin_sha256=cubin_sha256,
            driver=captured_driver,
            expected_abi=expected_abi,
        )
        return cls(
            driver=captured_driver,
            library=library,
            abi=abi,
            kernels=kernels,
            _construction_token=_LIBRARY_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def open_contract(
        cls,
        cubin: bytes | bytearray | memoryview,
        *,
        expected_contract: CudaCubinContract,
        driver: object | None = None,
    ) -> "CudaCubinLibrary":
        """Admit a GPU-free contract and retain that exact observed handle.

        Observation and later launch deliberately share one loaded library.  A
        symbol/ordinal-width mismatch unloads the rejected handle before the
        error escapes; no inspect-unload-reopen race is part of this authority.
        """

        if cls is not CudaCubinLibrary:
            raise CudaCubinError(
                "CUDA CUBIN library authority may not be subclassed"
            )
        if type(expected_contract) is not CudaCubinContract:
            raise CudaCubinError("expected CUDA CUBIN contract has the wrong type")
        raw = _cubin_bytes(cubin)
        _require_elf_cubin(raw)
        cubin_sha256 = hashlib.sha256(raw).hexdigest()
        if (
            expected_contract.cubin_sha256 != cubin_sha256
            or expected_contract.cubin_size != len(raw)
        ):
            raise CudaCubinError("CUDA CUBIN bytes differ from sealed contract")

        captured_driver = cls._capture_driver(driver)
        library, abi, kernels = cls._load_and_observe(
            raw,
            cubin_sha256=cubin_sha256,
            driver=captured_driver,
            expected_abi=None,
        )
        if abi.contract != expected_contract:
            try:
                _unload_library(captured_driver, library)
            except CudaCubinCleanupError as cleanup:
                raise CudaCubinCleanupError(
                    "CUDA CUBIN contract admission failed and its library could not "
                    "be unloaded; isolated CUDA worker must terminate: "
                    f"{cleanup}"
                ) from None
            raise CudaCubinError(
                "CUDA driver-observed CUBIN contract differs from sealed authority"
            )
        return cls(
            driver=captured_driver,
            library=library,
            abi=abi,
            kernels=kernels,
            _construction_token=_LIBRARY_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def open_ordered_contract(
        cls,
        cubin: bytes | bytearray | memoryview,
        *,
        expected_contract: CudaCubinContract,
        driver: object | None = None,
    ) -> "CudaCubinLibrary":
        """Bind logical kernel aliases to the exact sealed CUBIN by ordinal.

        CuTe derives physical symbols from its materialized Python module name,
        so a submission cannot stably declare those names.  Canonical contract
        order is instead the selector: the complete driver-observed inventory
        must have the same count and the same parameter-width vector at every
        ordinal.  Names and offsets remain captured in ``abi`` evidence, and the
        successfully checked library handle is the one retained for launch.
        """

        if cls is not CudaCubinLibrary:
            raise CudaCubinError(
                "CUDA CUBIN library authority may not be subclassed"
            )
        if type(expected_contract) is not CudaCubinContract:
            raise CudaCubinError("expected CUDA CUBIN contract has the wrong type")
        raw = _cubin_bytes(cubin)
        _require_elf_cubin(raw)
        cubin_sha256 = hashlib.sha256(raw).hexdigest()
        if (
            expected_contract.cubin_sha256 != cubin_sha256
            or expected_contract.cubin_size != len(raw)
        ):
            raise CudaCubinError("CUDA CUBIN bytes differ from sealed contract")

        captured_driver = cls._capture_driver(driver)
        library, abi, kernels = cls._load_and_observe(
            raw,
            cubin_sha256=cubin_sha256,
            driver=captured_driver,
            expected_abi=None,
        )
        expected_widths = tuple(
            kernel.parameter_sizes for kernel in expected_contract.kernels
        )
        observed_widths = tuple(
            kernel.parameter_sizes for kernel in abi.contract.kernels
        )
        if observed_widths != expected_widths:
            try:
                _unload_library(captured_driver, library)
            except CudaCubinCleanupError as cleanup:
                raise CudaCubinCleanupError(
                    "CUDA CUBIN ordered-contract admission failed and its library "
                    "could not be unloaded; isolated CUDA worker must terminate: "
                    f"{cleanup}"
                ) from None
            raise CudaCubinError(
                "CUDA driver-observed CUBIN ordinal widths differ from sealed authority"
            )
        return cls(
            driver=captured_driver,
            library=library,
            abi=abi,
            kernels=kernels,
            _construction_token=_LIBRARY_CONSTRUCTION_TOKEN,
        )

    @staticmethod
    def _capture_driver(driver: object | None) -> object:
        if driver is None:
            try:
                import cuda.bindings.driver as captured_driver
            except Exception as exc:  # noqa: BLE001 - optional runtime dependency
                raise CudaCubinError(
                    f"CUDA driver binding is unavailable: {exc}"
                ) from None
            driver = captured_driver
        _validate_driver(driver)
        init = _call_result(
            _driver_call(driver, "cuInit", 0), operation="cuInit"
        )
        _require_success(driver, init, operation="cuInit", arity=1)
        return driver

    @staticmethod
    def _load_and_observe(
        raw: bytes,
        *,
        cubin_sha256: str,
        driver: object,
        expected_abi: CudaCubinABI | None,
    ) -> tuple[object, CudaCubinABI, Mapping[str, object]]:
        load = _call_result(
            _driver_call(
                driver,
                "cuLibraryLoadData",
                raw,
                None,
                None,
                0,
                None,
                None,
                0,
            ),
            operation="cuLibraryLoadData",
        )
        (library,) = _require_success(
            driver, load, operation="cuLibraryLoadData", arity=2
        )
        try:
            count_result = _call_result(
                _driver_call(driver, "cuLibraryGetKernelCount", library),
                operation="cuLibraryGetKernelCount",
            )
            (count,) = _require_success(
                driver,
                count_result,
                operation="cuLibraryGetKernelCount",
                arity=2,
            )
            count = _driver_integer(count, field="kernel count")
            if not 1 <= count <= _MAX_KERNELS:
                raise CudaCubinError("CUDA kernel count is outside policy")
            enumerate_result = _call_result(
                _driver_call(
                    driver, "cuLibraryEnumerateKernels", count, library
                ),
                operation="cuLibraryEnumerateKernels",
            )
            (handles,) = _require_success(
                driver,
                enumerate_result,
                operation="cuLibraryEnumerateKernels",
                arity=2,
            )
            if not isinstance(handles, (tuple, list)) or len(handles) != count:
                raise CudaCubinError("CUDA kernel enumeration count mismatch")

            observed: dict[str, tuple[object, CudaKernelABI]] = {}
            for handle in handles:
                name_result = _call_result(
                    _driver_call(driver, "cuKernelGetName", handle),
                    operation="cuKernelGetName",
                )
                (raw_name,) = _require_success(
                    driver,
                    name_result,
                    operation="cuKernelGetName",
                    arity=2,
                )
                name = _kernel_name(raw_name, field="observed CUDA kernel name")
                if name in observed:
                    raise CudaCubinError("CUDA CUBIN repeats a kernel symbol")
                observed[name] = (
                    handle,
                    CudaKernelABI(
                        name=name,
                        parameters=_observe_kernel_parameters(driver, handle),
                    ),
                )
            abi = CudaCubinABI(
                cubin_sha256=cubin_sha256,
                cubin_size=len(raw),
                kernels=tuple(
                    observed[name][1] for name in sorted(observed)
                ),
            )
            if expected_abi is not None and abi != expected_abi:
                raise CudaCubinError(
                    "CUDA driver-observed CUBIN ABI differs from sealed authority"
                )
            return (
                library,
                abi,
                {name: row[0] for name, row in observed.items()},
            )
        except BaseException as original:
            # Loading succeeded, so any later rejection must drop the driver
            # library before propagating.  There is no candidate callback here.
            try:
                _unload_library(driver, library)
            except CudaCubinCleanupError as cleanup:
                raise CudaCubinCleanupError(
                    "CUDA CUBIN admission failed and its library could not be "
                    "unloaded; isolated CUDA worker must terminate: "
                    f"{cleanup}"
                ) from original
            raise

    @property
    def abi(self) -> CudaCubinABI:
        return self._abi

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def kernel(self, name: str) -> object:
        """Return the captured ``CUkernel`` only after exact name validation."""

        name = _kernel_name(name, field="requested CUDA kernel name")
        with self._lock:
            if self._closed:
                raise CudaCubinError("CUDA CUBIN library is closed")
            try:
                return self._kernels[name]
            except KeyError:
                raise CudaCubinError(
                    f"CUDA CUBIN has no sealed kernel {name!r}"
                ) from None

    def function(self, name: str) -> object:
        """Resolve a context-bound function for a previously sealed kernel."""

        name = _kernel_name(name, field="requested CUDA kernel name")
        with self._lock:
            if self._closed:
                raise CudaCubinError("CUDA CUBIN library is closed")
            try:
                kernel = self._kernels[name]
            except KeyError:
                raise CudaCubinError(
                    f"CUDA CUBIN has no sealed kernel {name!r}"
                ) from None
            result = _call_result(
                _driver_call(
                    self._driver, "cuKernelGetFunction", kernel
                ),
                operation="cuKernelGetFunction",
            )
            (function,) = _require_success(
                self._driver,
                result,
                operation="cuKernelGetFunction",
                arity=2,
            )
            return function

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            # Invalidate access before unloading.  Even if the binding reports
            # an unload error, the handle's lifetime is indeterminate and must
            # never be exposed for a retry or launch.
            self._closed = True
            library = self._library
            self._library = None
            self._kernels = MappingProxyType({})
            _unload_library(self._driver, library)

    def __enter__(self) -> "CudaCubinLibrary":
        if self.closed:
            raise CudaCubinError("CUDA CUBIN library is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


__all__ = [
    "CUDA_CUBIN_ABI_SCHEMA",
    "CUDA_CUBIN_CONTRACT_SCHEMA",
    "CudaCubinABI",
    "CudaCubinCleanupError",
    "CudaCubinContract",
    "CudaCubinError",
    "CudaCubinLibrary",
    "CudaKernelABI",
    "CudaKernelContract",
    "CudaKernelParameter",
    "cuda_cubin_identity",
]
