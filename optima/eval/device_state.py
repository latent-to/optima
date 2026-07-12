"""Trusted-host GPU state attestation for isolated OCI engine launches.

The contained engine must never decide whether its selected devices are clean.
This module runs on the outer host and has deliberately *no* management commands:
it observes an arena-pinned set of physical GPUs, verifies their immutable runtime
configuration, and waits for a bounded sequence of idle samples before returning a
monotonic receipt.

``DeviceStateGuard`` is stateful so pre/active/post receipts have a strict sequence.
Callers supply absolute monotonic deadlines.  Tests inject the
runner, clock, and sleeper; production uses an argv-only ``/usr/bin/nvidia-smi``
runner with a subprocess timeout derived from the absolute deadline.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
from typing import Callable, Protocol

from optima.eval.engine_launch import LogicalHardwareSpec, PhysicalHardwareBinding


NVIDIA_SMI = "/usr/bin/nvidia-smi"
_MAX_STDOUT_BYTES = 4 * 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_MAX_PROCESS_NAME_CHARS = 256
_LABEL = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_UUID = re.compile(r"GPU-[0-9A-Fa-f-]{16,64}\Z")
_PCI_BUS_ID = re.compile(
    r"(?:[0-9A-Fa-f]{4,8}:)?[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]\Z"
)
_DRIVER = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}\Z")
_PSTATE = re.compile(r"P[0-9]{1,2}\Z")
_PROCESS_KIND = re.compile(r"[A-Z][A-Z+/]{0,7}\Z")
_COMPUTE_MODES = frozenset(
    {"Default", "Exclusive_Process", "Prohibited", "Exclusive_Thread"}
)
_PERSISTENCE_MODES = frozenset({"Enabled", "Disabled"})
_UNSUPPORTED = frozenset(
    {
        "n/a",
        "not supported",
        "[not supported]",
        "requested functionality has been deprecated",
        "[requested functionality has been deprecated]",
        "-",
    }
)

_GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "pci.bus_id",
    "name",
    "memory.total",
    "driver_version",
    "power.limit",
    "compute_mode",
    "persistence_mode",
    "clocks.applications.graphics",
    "clocks.applications.memory",
    "clocks.max.graphics",
    "clocks.max.memory",
    "pstate",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.graphics",
    "clocks.current.memory",
    "power.draw",
)


class DeviceStateError(RuntimeError):
    """The trusted host could not produce a valid clean-device receipt."""


class DeviceStatePolicyError(ValueError):
    """The arena's immutable device-state policy is malformed."""


class DeviceStateCommandError(DeviceStateError):
    """The host telemetry command could not execute successfully."""


class DeviceStateParseError(DeviceStateError):
    """Host telemetry was malformed, ambiguous, or outside its bounds."""


class DeviceStateConfigurationError(DeviceStateError):
    """The selected GPUs or their immutable configuration changed."""


class DeviceStateTimeoutError(DeviceStateError):
    """The selected devices did not become clean before the absolute deadline."""


class DeviceStateEnvelopeTimeoutError(DeviceStateTimeoutError):
    """A live launch failed to enter or retain its active envelope in time.

    This error reports the observed device fact only.  Evaluation policy decides
    how a failed launch affects retry or qualification.
    """

class DeviceStateCancelledError(DeviceStateError):
    """The controller cancelled an in-flight launch observation."""


class DeviceStateEnvelopeError(DeviceStateError):
    """An active post-warmup sample violated the arena-pinned telemetry envelope."""


class DeviceStateClockError(DeviceStateError):
    """The injected/host monotonic clock moved backwards or became non-finite."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded primitive result returned by an injected command runner."""

    returncode: int
    stdout: str
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(
        self, argv: tuple[str, ...], *, timeout_s: float
    ) -> CommandResult: ...


def subprocess_runner(
    argv: tuple[str, ...], *, timeout_s: float
) -> CommandResult:
    """Run one read-only host telemetry command with no shell or inherited stdin."""

    if not argv or argv[0] != NVIDIA_SMI:
        raise DeviceStateCommandError("device telemetry executable is not the fixed nvidia-smi path")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise DeviceStateTimeoutError("device telemetry command has no remaining deadline")
    try:
        completed = subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout_s,
            check=False,
            shell=False,
            close_fds=True,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        raise DeviceStateTimeoutError("host nvidia-smi command exceeded its absolute deadline") from None
    except (OSError, UnicodeError) as exc:
        raise DeviceStateCommandError(f"host nvidia-smi command failed: {exc}") from None
    return CommandResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise DeviceStatePolicyError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _optional_clock_policy(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _exact_int(value, name, 1, 100_000)


@dataclass(frozen=True, slots=True)
class GPUConfiguration:
    """Arena-pinned identity and management configuration of one physical GPU."""

    physical_id: int
    uuid: str
    pci_bus_id: str
    name: str
    memory_total_mib: int
    driver_version: str
    power_limit_mw: int
    compute_mode: str
    persistence_mode: str
    application_graphics_clock_mhz: int | None
    application_memory_clock_mhz: int | None
    max_graphics_clock_mhz: int
    max_memory_clock_mhz: int

    def __post_init__(self) -> None:
        _exact_int(self.physical_id, "physical_id", 0, 65_535)
        if not isinstance(self.uuid, str) or _UUID.fullmatch(self.uuid) is None:
            raise DeviceStatePolicyError("GPU uuid must be a canonical physical GPU UUID")
        if (
            not isinstance(self.pci_bus_id, str)
            or _PCI_BUS_ID.fullmatch(self.pci_bus_id) is None
            or self.pci_bus_id != self.pci_bus_id.lower()
        ):
            raise DeviceStatePolicyError("GPU PCI bus id must be canonical lowercase PCI notation")
        if (
            not isinstance(self.name, str)
            or not self.name
            or len(self.name) > 256
            or any(char in self.name for char in "\x00\r\n")
        ):
            raise DeviceStatePolicyError("GPU name is invalid")
        _exact_int(self.memory_total_mib, "memory_total_mib", 1, 16_777_216)
        if (
            not isinstance(self.driver_version, str)
            or _DRIVER.fullmatch(self.driver_version) is None
        ):
            raise DeviceStatePolicyError("GPU driver version is invalid")
        _exact_int(self.power_limit_mw, "power_limit_mw", 1, 10_000_000)
        if self.compute_mode not in _COMPUTE_MODES:
            raise DeviceStatePolicyError("GPU compute mode is unsupported")
        if self.persistence_mode not in _PERSISTENCE_MODES:
            raise DeviceStatePolicyError("GPU persistence mode is unsupported")
        _optional_clock_policy(
            self.application_graphics_clock_mhz,
            "application_graphics_clock_mhz",
        )
        _optional_clock_policy(
            self.application_memory_clock_mhz,
            "application_memory_clock_mhz",
        )
        _exact_int(self.max_graphics_clock_mhz, "max_graphics_clock_mhz", 1, 100_000)
        _exact_int(self.max_memory_clock_mhz, "max_memory_clock_mhz", 1, 100_000)

    def canonical_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class DeviceStatePolicy:
    """Immutable arena policy for selected devices and pre/post idle drainage."""

    expected_gpus: tuple[GPUConfiguration, ...]
    maximum_temperature_c: int = 65
    maximum_gpu_utilization_percent: int = 5
    maximum_memory_utilization_percent: int = 5
    allowed_active_pstates: tuple[str, ...] = ("P0",)
    active_maximum_graphics_clock_mhz: int = 100_000
    active_memory_clock_mhz: int = 4_000
    active_maximum_power_draw_mw: int = 10_000_000
    active_require_process_on_every_gpu: bool = True
    required_consecutive_idle_samples: int = 3
    poll_interval_s: float = 2.0
    ready_poll_interval_s: float = 0.1
    drain_timeout_s: float = 300.0
    maximum_samples: int = 256

    def __post_init__(self) -> None:
        if (
            not isinstance(self.expected_gpus, tuple)
            or not self.expected_gpus
            or any(type(gpu) is not GPUConfiguration for gpu in self.expected_gpus)
        ):
            raise DeviceStatePolicyError(
                "expected_gpus must be a non-empty tuple of exact GPUConfiguration values"
            )
        ids = tuple(gpu.physical_id for gpu in self.expected_gpus)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise DeviceStatePolicyError("physical GPU IDs must be unique and sorted")
        for name, values in (
            ("UUID", tuple(gpu.uuid for gpu in self.expected_gpus)),
            ("PCI bus ID", tuple(gpu.pci_bus_id for gpu in self.expected_gpus)),
        ):
            if len(set(values)) != len(values):
                raise DeviceStatePolicyError(f"physical GPU {name}s must be unique")
        drivers = {gpu.driver_version for gpu in self.expected_gpus}
        if len(drivers) != 1:
            raise DeviceStatePolicyError("all selected GPUs must expose one host driver version")
        _exact_int(self.maximum_temperature_c, "maximum_temperature_c", 0, 120)
        _exact_int(
            self.maximum_gpu_utilization_percent,
            "maximum_gpu_utilization_percent",
            0,
            25,
        )
        _exact_int(
            self.maximum_memory_utilization_percent,
            "maximum_memory_utilization_percent",
            0,
            25,
        )
        if (
            not isinstance(self.allowed_active_pstates, tuple)
            or not self.allowed_active_pstates
            or any(
                not isinstance(value, str) or _PSTATE.fullmatch(value) is None
                for value in self.allowed_active_pstates
            )
            or len(set(self.allowed_active_pstates)) != len(self.allowed_active_pstates)
        ):
            raise DeviceStatePolicyError(
                "allowed_active_pstates must be a non-empty tuple of unique P-states"
            )
        _exact_int(
            self.active_maximum_graphics_clock_mhz,
            "active_maximum_graphics_clock_mhz",
            1,
            100_000,
        )
        _exact_int(
            self.active_memory_clock_mhz,
            "active_memory_clock_mhz",
            1,
            100_000,
        )
        _exact_int(
            self.active_maximum_power_draw_mw,
            "active_maximum_power_draw_mw",
            1,
            10_000_000,
        )
        if type(self.active_require_process_on_every_gpu) is not bool:
            raise DeviceStatePolicyError(
                "active_require_process_on_every_gpu must be boolean"
            )
        _exact_int(
            self.required_consecutive_idle_samples,
            "required_consecutive_idle_samples",
            2,
            32,
        )
        if (
            isinstance(self.poll_interval_s, bool)
            or not isinstance(self.poll_interval_s, (int, float))
            or not math.isfinite(float(self.poll_interval_s))
            or not 0.05 <= float(self.poll_interval_s) <= 60.0
        ):
            raise DeviceStatePolicyError("poll_interval_s must be finite and in [0.05, 60]")
        if (
            isinstance(self.ready_poll_interval_s, bool)
            or not isinstance(self.ready_poll_interval_s, (int, float))
            or not math.isfinite(float(self.ready_poll_interval_s))
            or not 0.05 <= float(self.ready_poll_interval_s) <= 5.0
        ):
            raise DeviceStatePolicyError(
                "ready_poll_interval_s must be finite and in [0.05, 5]"
            )
        if (
            isinstance(self.drain_timeout_s, bool)
            or not isinstance(self.drain_timeout_s, (int, float))
            or not math.isfinite(float(self.drain_timeout_s))
            or not 1.0 <= float(self.drain_timeout_s) <= 3600.0
        ):
            raise DeviceStatePolicyError("drain_timeout_s must be finite and in [1, 3600]")
        _exact_int(self.maximum_samples, "maximum_samples", 2, 4096)
        if self.maximum_samples < self.required_consecutive_idle_samples:
            raise DeviceStatePolicyError(
                "maximum_samples must cover the required consecutive idle samples"
            )

    @property
    def physical_gpu_ids(self) -> tuple[int, ...]:
        return tuple(gpu.physical_id for gpu in self.expected_gpus)

    @property
    def configuration_sha256(self) -> str:
        payload = json.dumps(
            [gpu.canonical_dict() for gpu in self.expected_gpus],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    @property
    def policy_sha256(self) -> str:
        payload = {
            "active_maximum_graphics_clock_mhz": self.active_maximum_graphics_clock_mhz,
            "active_maximum_power_draw_mw": self.active_maximum_power_draw_mw,
            "active_memory_clock_mhz": self.active_memory_clock_mhz,
            "active_require_process_on_every_gpu": self.active_require_process_on_every_gpu,
            "allowed_active_pstates": list(self.allowed_active_pstates),
            "configuration_sha256": self.configuration_sha256,
            "drain_timeout_s": float(self.drain_timeout_s),
            "maximum_gpu_utilization_percent": self.maximum_gpu_utilization_percent,
            "maximum_memory_utilization_percent": self.maximum_memory_utilization_percent,
            "maximum_samples": self.maximum_samples,
            "maximum_temperature_c": self.maximum_temperature_c,
            "poll_interval_s": float(self.poll_interval_s),
            "ready_poll_interval_s": float(self.ready_poll_interval_s),
            "required_consecutive_idle_samples": self.required_consecutive_idle_samples,
        }
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class GPUTelemetry:
    physical_id: int
    uuid: str
    pstate: str
    temperature_c: int
    gpu_utilization_percent: int
    memory_utilization_percent: int
    current_graphics_clock_mhz: int | None
    current_memory_clock_mhz: int | None
    power_draw_mw: int | None


@dataclass(frozen=True, slots=True)
class GPUProcess:
    physical_id: int
    pid: int
    kind: str
    process_name: str


@dataclass(frozen=True, slots=True)
class DeviceStateSample:
    monotonic_s: float
    telemetry: tuple[GPUTelemetry, ...]
    processes: tuple[GPUProcess, ...]
    idle: bool
    idle_reason: str
    active_envelope_passed: bool = False
    active_envelope_reason: str = "not evaluated"


@dataclass(frozen=True, slots=True)
class DeviceStateReceipt:
    """One trusted-host, monotonically ordered pre/post launch receipt."""

    schema: str
    sequence: int
    launch_id: str
    phase: str
    selected_physical_gpu_ids: tuple[int, ...]
    configuration_sha256: str
    policy_sha256: str
    started_monotonic_s: float
    completed_monotonic_s: float
    consecutive_idle_samples: int
    samples: tuple[DeviceStateSample, ...]


@dataclass(frozen=True, slots=True)
class DeviceStateActiveReceipt:
    """Concurrent warmup telemetry plus bounded post-response readiness evidence."""

    schema: str
    sequence: int
    launch_id: str
    event: str
    selected_physical_gpu_ids: tuple[int, ...]
    configuration_sha256: str
    policy_sha256: str
    started_monotonic_s: float
    completed_monotonic_s: float
    consecutive_active_samples: int
    release_sample_index: int
    post_release_ready_samples: int
    samples: tuple[DeviceStateSample, ...]


def validate_device_state_policy(
    policy: DeviceStatePolicy,
    *,
    logical_hardware: LogicalHardwareSpec,
    physical_hardware: PhysicalHardwareBinding,
) -> None:
    """Bind one host observation policy to the resolved launch hardware.

    Physical GPU indices are host-local, while launch bindings may select devices
    by index or UUID.  Both forms are accepted only when every ordered identity and
    the complete policy digest match the already-resolved logical specification.
    """

    if type(policy) is not DeviceStatePolicy:
        raise DeviceStatePolicyError("device state policy has the wrong type")
    if type(logical_hardware) is not LogicalHardwareSpec:
        raise DeviceStatePolicyError("logical hardware has the wrong type")
    if type(physical_hardware) is not PhysicalHardwareBinding:
        raise DeviceStatePolicyError("physical hardware binding has the wrong type")
    physical_hardware.validate_against(logical_hardware)
    if policy.policy_sha256 != logical_hardware.device_policy_digest:
        raise DeviceStatePolicyError(
            "device state policy digest does not match logical hardware"
        )
    if len(policy.expected_gpus) != logical_hardware.visible_gpu_count:
        raise DeviceStatePolicyError(
            "device state policy GPU count does not match logical hardware"
        )
    selected = physical_hardware.physical_gpu_ids
    indices = tuple(str(gpu.physical_id) for gpu in policy.expected_gpus)
    uuids = tuple(gpu.uuid for gpu in policy.expected_gpus)
    if selected not in {indices, uuids}:
        raise DeviceStatePolicyError(
            "device state policy identities do not match physical hardware binding"
        )


@dataclass(frozen=True, slots=True)
class _ObservedGPU:
    configuration: GPUConfiguration
    telemetry: GPUTelemetry


def _parse_int(text: str, name: str, low: int, high: int) -> int:
    try:
        value = int(text.strip(), 10)
    except (TypeError, ValueError):
        raise DeviceStateParseError(f"{name} is not an integer: {text!r}") from None
    if not low <= value <= high:
        raise DeviceStateParseError(f"{name} is outside [{low}, {high}]: {value}")
    return value


def _parse_milli(text: str, name: str, *, allow_unsupported: bool) -> int | None:
    clean = text.strip()
    if clean.lower() in _UNSUPPORTED:
        if allow_unsupported:
            return None
        raise DeviceStateParseError(f"{name} may not be unsupported")
    try:
        value = Decimal(clean)
    except (InvalidOperation, ValueError):
        raise DeviceStateParseError(f"{name} is not a decimal: {text!r}") from None
    if not value.is_finite() or value < 0 or value > Decimal("10000"):
        raise DeviceStateParseError(f"{name} is outside its finite bound")
    scaled = value * 1000
    if scaled != scaled.to_integral_value():
        raise DeviceStateParseError(f"{name} has more than milliwatt precision")
    return int(scaled)


def _parse_clock(text: str, name: str, *, allow_unsupported: bool) -> int | None:
    clean = text.strip()
    if clean.lower() in _UNSUPPORTED:
        if allow_unsupported:
            return None
        raise DeviceStateParseError(f"{name} may not be unsupported")
    return _parse_int(clean, name, 1, 100_000)


def _safe_text(text: str, name: str, *, maximum: int = 256) -> str:
    clean = text.strip()
    if not clean or len(clean) > maximum or any(char in clean for char in "\x00\r\n"):
        raise DeviceStateParseError(f"{name} is empty, oversized, or contains control bytes")
    return clean


def _configuration_difference(
    expected: GPUConfiguration, observed: GPUConfiguration
) -> str:
    changed = [
        field.name
        for field in fields(GPUConfiguration)
        if getattr(expected, field.name) != getattr(observed, field.name)
    ]
    return ",".join(changed) or "unknown"


def _normalize_selected_gpu_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise DeviceStatePolicyError(
            "selected physical GPU IDs must be a non-empty tuple"
        )
    for item in value:
        _exact_int(item, "selected physical GPU ID", 0, 65_535)
    if value != tuple(sorted(value)) or len(set(value)) != len(value):
        raise DeviceStatePolicyError(
            "selected physical GPU IDs must be unique and sorted"
        )
    return value


def parse_gpu_query(
    text: str,
    *,
    policy: DeviceStatePolicy | None = None,
    selected_gpu_ids: tuple[int, ...] | None = None,
) -> tuple[_ObservedGPU, ...]:
    """Parse one exact ``--query-gpu`` CSV response and bind it to the policy."""

    if (policy is None) == (selected_gpu_ids is None):
        raise DeviceStatePolicyError(
            "GPU query parsing requires exactly one policy or selected-ID set"
        )
    selected_tuple = (
        policy.physical_gpu_ids
        if policy is not None
        else _normalize_selected_gpu_ids(selected_gpu_ids)
    )

    if not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_STDOUT_BYTES:
        raise DeviceStateParseError("GPU query output exceeds its byte bound")
    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except (csv.Error, UnicodeError) as exc:
        raise DeviceStateParseError(f"GPU query output is not strict CSV: {exc}") from None
    if any(not row or all(not cell.strip() for cell in row) for row in rows):
        raise DeviceStateParseError("GPU query output contains a blank row")
    for row in rows:
        if len(row) != len(_GPU_QUERY_FIELDS):
            raise DeviceStateParseError(
                f"GPU query row has {len(row)} fields; expected {len(_GPU_QUERY_FIELDS)}"
            )
    if len(rows) != len(selected_tuple):
        raise DeviceStateConfigurationError(
            f"GPU query returned {len(rows)} rows for {len(selected_tuple)} selected GPUs"
        )

    observed: dict[int, _ObservedGPU] = {}
    seen_uuids: set[str] = set()
    seen_pci: set[str] = set()
    for row in rows:
        physical_id = _parse_int(row[0], "GPU index", 0, 65_535)
        if physical_id in observed:
            raise DeviceStateConfigurationError(f"GPU query duplicated physical ID {physical_id}")
        uuid = _safe_text(row[1], "GPU UUID")
        if _UUID.fullmatch(uuid) is None or uuid in seen_uuids:
            raise DeviceStateConfigurationError(f"GPU query returned invalid/duplicate UUID {uuid!r}")
        pci = _safe_text(row[2], "GPU PCI bus ID").lower()
        if _PCI_BUS_ID.fullmatch(pci) is None or pci in seen_pci:
            raise DeviceStateConfigurationError(
                f"GPU query returned invalid/duplicate PCI bus ID {pci!r}"
            )
        compute_mode = _safe_text(row[7], "GPU compute mode").replace(" ", "_")
        persistence_mode = _safe_text(row[8], "GPU persistence mode")
        try:
            configuration = GPUConfiguration(
                physical_id=physical_id,
                uuid=uuid,
                pci_bus_id=pci,
                name=_safe_text(row[3], "GPU name"),
                memory_total_mib=_parse_int(
                    row[4], "GPU total memory MiB", 1, 16_777_216
                ),
                driver_version=_safe_text(row[5], "GPU driver version", maximum=64),
                power_limit_mw=int(
                    _parse_milli(row[6], "GPU power limit", allow_unsupported=False)
                ),
                compute_mode=compute_mode,
                persistence_mode=persistence_mode,
                application_graphics_clock_mhz=_parse_clock(
                    row[9], "GPU application graphics clock", allow_unsupported=True
                ),
                application_memory_clock_mhz=_parse_clock(
                    row[10], "GPU application memory clock", allow_unsupported=True
                ),
                max_graphics_clock_mhz=int(
                    _parse_clock(row[11], "GPU maximum graphics clock", allow_unsupported=False)
                ),
                max_memory_clock_mhz=int(
                    _parse_clock(row[12], "GPU maximum memory clock", allow_unsupported=False)
                ),
            )
        except DeviceStatePolicyError as exc:
            raise DeviceStateParseError(f"GPU configuration row is invalid: {exc}") from None
        telemetry = GPUTelemetry(
            physical_id=physical_id,
            uuid=uuid,
            pstate=_safe_text(row[13], "GPU performance state", maximum=8),
            temperature_c=_parse_int(row[14], "GPU temperature", -20, 150),
            gpu_utilization_percent=_parse_int(row[15], "GPU utilization", 0, 100),
            memory_utilization_percent=_parse_int(
                row[16], "GPU memory utilization", 0, 100
            ),
            current_graphics_clock_mhz=_parse_clock(
                row[17], "GPU current graphics clock", allow_unsupported=True
            ),
            current_memory_clock_mhz=_parse_clock(
                row[18], "GPU current memory clock", allow_unsupported=True
            ),
            power_draw_mw=_parse_milli(
                row[19], "GPU power draw", allow_unsupported=True
            ),
        )
        if _PSTATE.fullmatch(telemetry.pstate) is None:
            raise DeviceStateParseError(
                f"GPU performance state is invalid: {telemetry.pstate!r}"
            )
        observed[physical_id] = _ObservedGPU(configuration, telemetry)
        seen_uuids.add(uuid)
        seen_pci.add(pci)

    selected = set(selected_tuple)
    returned = set(observed)
    if returned != selected:
        raise DeviceStateConfigurationError(
            "GPU query selected-set mismatch: "
            f"missing={sorted(selected - returned)} extra={sorted(returned - selected)}"
        )
    ordered = tuple(observed[physical_id] for physical_id in selected_tuple)
    if policy is not None:
        for expected, item in zip(policy.expected_gpus, ordered):
            if item.configuration != expected:
                raise DeviceStateConfigurationError(
                    f"GPU {expected.physical_id} immutable configuration changed: "
                    f"{_configuration_difference(expected, item.configuration)}"
                )
    return ordered


def provision_gpu_configurations(
    selected_gpu_ids: tuple[int, ...],
    *,
    deadline: float,
    runner: CommandRunner = subprocess_runner,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[GPUConfiguration, ...]:
    """Read exact per-host GPU identities for subsequent arena-class validation.

    This helper is provisioning evidence, not an acceptance policy. The
    caller must compare every returned management/class field with the immutable
    arena profile, then freeze these UUID/PCI/index rows into ``DeviceStatePolicy``.
    """

    selected = _normalize_selected_gpu_ids(selected_gpu_ids)

    def now() -> float:
        value = clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise DeviceStateClockError(
                "host monotonic clock returned a non-finite provisioning value"
            )
        return float(value)

    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(float(deadline))
    ):
        raise DeviceStatePolicyError(
            "provisioning deadline must be an absolute finite monotonic value"
        )
    started = now()
    remaining = float(deadline) - started
    if remaining <= 0:
        raise DeviceStateTimeoutError("GPU provisioning started after its deadline")
    ids = ",".join(str(value) for value in selected)
    argv = (
        NVIDIA_SMI,
        f"--id={ids}",
        "--query-gpu=" + ",".join(_GPU_QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    )
    try:
        result = runner(argv, timeout_s=remaining)
    except DeviceStateError:
        raise
    except subprocess.TimeoutExpired:
        raise DeviceStateTimeoutError(
            "GPU provisioning query exceeded its deadline"
        ) from None
    except (OSError, UnicodeError) as exc:
        raise DeviceStateCommandError(f"GPU provisioning query failed: {exc}") from None
    completed = now()
    if completed < started:
        raise DeviceStateClockError("host monotonic clock moved backwards during provisioning")
    if completed > float(deadline):
        raise DeviceStateTimeoutError("GPU provisioning query returned after its deadline")
    if (
        type(result) is not CommandResult
        or type(result.returncode) is not int
        or not isinstance(result.stdout, str)
        or not isinstance(result.stderr, str)
    ):
        raise DeviceStateCommandError("GPU provisioning runner returned an invalid result")
    if (
        len(result.stdout.encode("utf-8")) > _MAX_STDOUT_BYTES
        or len(result.stderr.encode("utf-8")) > _MAX_STDERR_BYTES
    ):
        raise DeviceStateCommandError("GPU provisioning output exceeds its byte bound")
    if result.returncode != 0:
        raise DeviceStateCommandError(
            "GPU provisioning nvidia-smi failed with status "
            f"{result.returncode}: {result.stderr.strip()[:1024]}"
        )
    observed = parse_gpu_query(
        result.stdout, selected_gpu_ids=selected
    )
    return tuple(item.configuration for item in observed)


def parse_process_monitor(
    text: str, *, selected_gpu_ids: tuple[int, ...]
) -> tuple[GPUProcess, ...]:
    """Parse ``nvidia-smi pmon`` rows, including compute and graphics contexts."""

    if not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_STDOUT_BYTES:
        raise DeviceStateParseError("GPU process output exceeds its byte bound")
    selected = set(selected_gpu_ids)
    seen_gpus: set[int] = set()
    seen_processes: set[tuple[int, int, str]] = set()
    processes: list[GPUProcess] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not 3 <= len(parts) <= 64:
            raise DeviceStateParseError(f"GPU process row is malformed: {line!r}")
        physical_id = _parse_int(parts[0], "process GPU index", 0, 65_535)
        if physical_id not in selected:
            raise DeviceStateConfigurationError(
                f"process monitor returned unselected physical GPU {physical_id}"
            )
        seen_gpus.add(physical_id)
        pid_text, kind = parts[1], parts[2]
        if pid_text == "-":
            if kind != "-":
                raise DeviceStateParseError("idle process row has a non-idle process kind")
            continue
        pid = _parse_int(pid_text, "GPU process PID", 1, 2_147_483_647)
        if _PROCESS_KIND.fullmatch(kind) is None:
            raise DeviceStateParseError(f"GPU process type is invalid: {kind!r}")
        key = (physical_id, pid, kind)
        if key in seen_processes:
            raise DeviceStateParseError(f"GPU process monitor duplicated {key!r}")
        name = _safe_text(
            parts[-1], "GPU process name", maximum=_MAX_PROCESS_NAME_CHARS
        )
        processes.append(GPUProcess(physical_id, pid, kind, name))
        seen_processes.add(key)
    if seen_gpus != selected:
        raise DeviceStateConfigurationError(
            "process monitor selected-set mismatch: "
            f"missing={sorted(selected - seen_gpus)} extra={sorted(seen_gpus - selected)}"
        )
    return tuple(sorted(processes, key=lambda item: (item.physical_id, item.pid, item.kind)))


class DeviceStateGuard:
    """Trusted, observation-only guard for one selected GPU set."""

    def __init__(
        self,
        policy: DeviceStatePolicy,
        *,
        runner: CommandRunner = subprocess_runner,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(policy) is not DeviceStatePolicy:
            raise DeviceStatePolicyError("DeviceStateGuard requires an exact DeviceStatePolicy")
        self.policy = policy
        self._runner = runner
        self._clock = clock
        self._sleep = sleep
        self._last_clock = -math.inf
        self._last_receipt_completed = -math.inf
        self._sequence = 0

    def _now(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise DeviceStateClockError("host monotonic clock returned a non-finite value")
        result = float(value)
        if result < self._last_clock:
            raise DeviceStateClockError("host monotonic clock moved backwards")
        self._last_clock = result
        return result

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._now()
        if remaining <= 0:
            raise DeviceStateTimeoutError("device drain absolute deadline expired")
        return remaining

    def _command(self, argv: tuple[str, ...], *, deadline: float) -> str:
        if not argv or argv[0] != NVIDIA_SMI:
            raise DeviceStateCommandError("device guard constructed a non-nvidia-smi command")
        try:
            result = self._runner(argv, timeout_s=self._remaining(deadline))
        except DeviceStateError:
            raise
        except subprocess.TimeoutExpired:
            raise DeviceStateTimeoutError("host telemetry runner exceeded its deadline") from None
        except (OSError, UnicodeError) as exc:
            raise DeviceStateCommandError(f"host telemetry runner failed: {exc}") from None
        if type(result) is not CommandResult:
            raise DeviceStateCommandError("host telemetry runner returned an invalid result type")
        if type(result.returncode) is not int:
            raise DeviceStateCommandError("host telemetry runner returned an invalid exit status")
        if self._now() > deadline:
            raise DeviceStateTimeoutError("host telemetry runner returned after its deadline")
        if (
            not isinstance(result.stdout, str)
            or not isinstance(result.stderr, str)
            or len(result.stdout.encode("utf-8")) > _MAX_STDOUT_BYTES
            or len(result.stderr.encode("utf-8")) > _MAX_STDERR_BYTES
        ):
            raise DeviceStateCommandError("host telemetry output exceeds its byte bound")
        if result.returncode != 0:
            diagnostic = result.stderr.strip()[:1024]
            raise DeviceStateCommandError(
                f"host nvidia-smi exited with status {result.returncode}: {diagnostic}"
            )
        return result.stdout

    def _query(self, *, deadline: float) -> tuple[tuple[GPUTelemetry, ...], tuple[GPUProcess, ...]]:
        ids = ",".join(str(value) for value in self.policy.physical_gpu_ids)
        query_argv = (
            NVIDIA_SMI,
            f"--id={ids}",
            "--query-gpu=" + ",".join(_GPU_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        )
        process_argv = (
            NVIDIA_SMI,
            "pmon",
            f"--id={ids}",
            "--count=1",
            "--select=u",
        )
        # A wedged nvidia-smi must not make cancellation wait for the whole
        # multi-hour bracket deadline. Each live query receives a small bounded
        # command window while the outer polling loop retains the arena deadline.
        command_window = max(2.0, 4.0 * float(self.policy.poll_interval_s))
        query_deadline = min(deadline, self._now() + command_window)
        observed = parse_gpu_query(
            self._command(query_argv, deadline=query_deadline), policy=self.policy
        )
        process_deadline = min(deadline, self._now() + command_window)
        processes = parse_process_monitor(
            self._command(process_argv, deadline=process_deadline),
            selected_gpu_ids=self.policy.physical_gpu_ids,
        )
        return tuple(item.telemetry for item in observed), processes

    def _idle_verdict(
        self,
        telemetry: tuple[GPUTelemetry, ...],
        processes: tuple[GPUProcess, ...],
    ) -> tuple[bool, str]:
        if processes:
            identities = ",".join(
                f"gpu{item.physical_id}:pid{item.pid}:{item.kind}"
                for item in processes[:16]
            )
            return False, f"active GPU processes: {identities}"
        hot = [
            item.physical_id
            for item in telemetry
            if item.temperature_c > self.policy.maximum_temperature_c
        ]
        if hot:
            return False, f"temperature above policy on GPUs {hot}"
        busy = [
            item.physical_id
            for item in telemetry
            if item.gpu_utilization_percent
            > self.policy.maximum_gpu_utilization_percent
        ]
        if busy:
            return False, f"GPU utilization above policy on GPUs {busy}"
        memory_busy = [
            item.physical_id
            for item in telemetry
            if item.memory_utilization_percent
            > self.policy.maximum_memory_utilization_percent
        ]
        if memory_busy:
            return False, f"memory utilization above policy on GPUs {memory_busy}"
        return True, "no processes; temperature/utilization within policy"

    def _active_envelope_verdict(
        self,
        telemetry: tuple[GPUTelemetry, ...],
        processes: tuple[GPUProcess, ...],
    ) -> tuple[bool, str]:
        violations: list[str] = []
        process_gpu_ids = {item.physical_id for item in processes}
        if self.policy.active_require_process_on_every_gpu:
            missing = set(self.policy.physical_gpu_ids) - process_gpu_ids
            if missing:
                violations.append(f"missing active process on GPUs {sorted(missing)}")
        graphics_only = [
            (item.physical_id, item.pid, item.kind)
            for item in processes
            if "C" not in item.kind and "M" not in item.kind
        ]
        if graphics_only:
            violations.append(f"unexpected graphics-only processes {graphics_only[:16]}")
        for item in telemetry:
            prefix = f"gpu{item.physical_id}"
            if item.pstate not in self.policy.allowed_active_pstates:
                violations.append(f"{prefix} pstate={item.pstate}")
            if (
                item.current_graphics_clock_mhz is None
                or item.current_graphics_clock_mhz
                > self.policy.active_maximum_graphics_clock_mhz
            ):
                violations.append(
                    f"{prefix} graphics_clock={item.current_graphics_clock_mhz}"
                )
            if item.current_memory_clock_mhz != self.policy.active_memory_clock_mhz:
                violations.append(
                    f"{prefix} memory_clock={item.current_memory_clock_mhz}"
                )
            if (
                item.power_draw_mw is None
                or item.power_draw_mw > self.policy.active_maximum_power_draw_mw
            ):
                violations.append(f"{prefix} power_draw_mw={item.power_draw_mw}")
            if item.temperature_c > self.policy.maximum_temperature_c:
                violations.append(f"{prefix} temperature_c={item.temperature_c}")
        if violations:
            return False, "; ".join(violations)[:4096]
        return True, "active processes and telemetry satisfy the pinned envelope"

    def _ready_envelope_verdict(
        self,
        telemetry: tuple[GPUTelemetry, ...],
        processes: tuple[GPUProcess, ...],
    ) -> tuple[bool, str]:
        """Post-response quiescence without pretending processes have exited.

        The engine must stay loaded, so the ordinary idle verdict cannot apply.
        Require the pinned active configuration plus low GPU/memory utilization;
        this rejects background kernels crossing into the timed request.
        Thermal equivalence is not inferred from this receipt—the trusted final
        warmup rate is separately charged as a throughput lower bound.
        """

        passed, reason = self._active_envelope_verdict(telemetry, processes)
        violations: list[str] = [] if passed else [reason]
        for item in telemetry:
            prefix = f"gpu{item.physical_id}"
            if (
                item.gpu_utilization_percent
                > self.policy.maximum_gpu_utilization_percent
            ):
                violations.append(
                    f"{prefix} gpu_utilization={item.gpu_utilization_percent}"
                )
            if (
                item.memory_utilization_percent
                > self.policy.maximum_memory_utilization_percent
            ):
                violations.append(
                    f"{prefix} memory_utilization={item.memory_utilization_percent}"
                )
        if violations:
            return False, "; ".join(violations)[:4096]
        return True, "loaded engine is quiescent in the pinned active configuration"

    def _drain(
        self, *, launch_id: str, phase: str, deadline: float | None
    ) -> DeviceStateReceipt:
        if not isinstance(launch_id, str) or _LABEL.fullmatch(launch_id) is None:
            raise DeviceStatePolicyError("launch_id must be a simple 1..128 character identifier")
        if phase not in {"pre", "post"}:
            raise DeviceStatePolicyError("device receipt phase must be 'pre' or 'post'")
        started = self._now()
        if started < self._last_receipt_completed:
            raise DeviceStateClockError("device receipt order is not monotonic")
        local_deadline = started + float(self.policy.drain_timeout_s)
        if deadline is not None:
            if (
                isinstance(deadline, bool)
                or not isinstance(deadline, (int, float))
                or not math.isfinite(float(deadline))
            ):
                raise DeviceStatePolicyError("deadline must be an absolute finite monotonic value")
            local_deadline = min(local_deadline, float(deadline))
        if local_deadline <= started:
            raise DeviceStateTimeoutError("device drain started after its absolute deadline")

        samples: list[DeviceStateSample] = []
        consecutive = 0
        last_reason = "no telemetry sample completed"
        for _ in range(self.policy.maximum_samples):
            telemetry, processes = self._query(deadline=local_deadline)
            sampled_at = self._now()
            if sampled_at > local_deadline:
                raise DeviceStateTimeoutError("device sample completed after its deadline")
            idle, reason = self._idle_verdict(telemetry, processes)
            samples.append(
                DeviceStateSample(sampled_at, telemetry, processes, idle, reason)
            )
            last_reason = reason
            consecutive = consecutive + 1 if idle else 0
            if consecutive >= self.policy.required_consecutive_idle_samples:
                completed = self._now()
                if completed > local_deadline:
                    raise DeviceStateTimeoutError("device receipt completed after its deadline")
                self._sequence += 1
                receipt = DeviceStateReceipt(
                    schema="optima.device-state-receipt.v1",
                    sequence=self._sequence,
                    launch_id=launch_id,
                    phase=phase,
                    selected_physical_gpu_ids=self.policy.physical_gpu_ids,
                    configuration_sha256=self.policy.configuration_sha256,
                    policy_sha256=self.policy.policy_sha256,
                    started_monotonic_s=started,
                    completed_monotonic_s=completed,
                    consecutive_idle_samples=consecutive,
                    samples=tuple(samples),
                )
                self._last_receipt_completed = completed
                return receipt
            remaining = local_deadline - self._now()
            if remaining <= 0:
                break
            self._sleep(min(float(self.policy.poll_interval_s), remaining))
        raise DeviceStateTimeoutError(
            "selected GPUs did not satisfy the idle envelope before the bounded "
            f"drain ended: {last_reason}"
        )

    def before_launch(
        self, launch_id: str, *, deadline: float | None = None
    ) -> DeviceStateReceipt:
        """Drain and attest the exact selected GPUs before one launch."""

        return self._drain(launch_id=launch_id, phase="pre", deadline=deadline)

    def after_launch(
        self, launch_id: str, *, deadline: float | None = None
    ) -> DeviceStateReceipt:
        """Drain and attest the exact selected GPUs after one launch."""

        return self._drain(launch_id=launch_id, phase="post", deadline=deadline)

    def condition_active(
        self,
        launch_id: str,
        event: str = "post-warmup",
        *,
        deadline: float | None = None,
        release: Callable[[], bool] | None = None,
        wait_for_release: Callable[[float], bool] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> DeviceStateActiveReceipt:
        """Poll until consecutive active-envelope samples permit timed work.

        When ``release`` is supplied, sampling begins while the final warmup is
        still running but authority is withheld until the controller reports that
        the warmup response has crossed the hostile boundary. The receipt requires
        the policy's full consecutive active run *before* that boundary and exactly
        one stricter ready-envelope pass afterward. High utilization after release
        does not erase already-proven serving state; an underlying P-state, clock,
        process, power, or temperature failure does. The charged warmup/tail rate
        separately makes sleeping or cooling before the response a throughput loss.

        ``wait_for_release`` lets the host event interrupt a long telemetry polling
        interval. It changes no authority: the loop still performs a post-response
        query before returning, but does not insert an artificial full polling
        sleep between the final warmup and the first timed request.
        """

        if not isinstance(launch_id, str) or _LABEL.fullmatch(launch_id) is None:
            raise DeviceStatePolicyError("launch_id must be a simple 1..128 character identifier")
        if not isinstance(event, str) or _LABEL.fullmatch(event) is None:
            raise DeviceStatePolicyError("event must be a simple 1..128 character identifier")
        if release is not None and not callable(release):
            raise DeviceStatePolicyError("release must be callable or null")
        if wait_for_release is not None and (
            release is None or not callable(wait_for_release)
        ):
            raise DeviceStatePolicyError(
                "wait_for_release requires release and must be callable"
            )
        if cancel is not None and not callable(cancel):
            raise DeviceStatePolicyError("cancel must be callable or null")
        started = self._now()
        if started < self._last_receipt_completed:
            raise DeviceStateClockError("device receipt order is not monotonic")
        local_deadline = started + float(self.policy.drain_timeout_s)
        if deadline is not None:
            if (
                isinstance(deadline, bool)
                or not isinstance(deadline, (int, float))
                or not math.isfinite(float(deadline))
            ):
                raise DeviceStatePolicyError(
                    "deadline must be an absolute finite monotonic value"
                )
            local_deadline = min(local_deadline, float(deadline))
        if local_deadline <= started:
            raise DeviceStateTimeoutError(
                "active device conditioning started after its deadline"
            )

        samples: list[DeviceStateSample] = []
        required = self.policy.required_consecutive_idle_samples
        consecutive = 0
        released = release is None
        release_sample_index: int | None = 0 if release is None else None
        last_reason = "no active telemetry sample completed"

        def release_now() -> bool:
            assert release is not None
            try:
                return bool(release())
            except BaseException as exc:  # noqa: BLE001 - trusted callback
                raise DeviceStatePolicyError(
                    f"active-conditioning release callback failed: {exc}"
                ) from None

        def released_receipt(*, post_release_ready_samples: int) -> DeviceStateActiveReceipt:
            nonlocal release_sample_index
            completed = self._now()
            if completed > local_deadline:
                raise DeviceStateTimeoutError(
                    "active device receipt completed after its deadline"
                )
            self._sequence += 1
            receipt = DeviceStateActiveReceipt(
                schema="optima.device-state-active-receipt.v2",
                sequence=self._sequence,
                launch_id=launch_id,
                event=event,
                selected_physical_gpu_ids=self.policy.physical_gpu_ids,
                configuration_sha256=self.policy.configuration_sha256,
                policy_sha256=self.policy.policy_sha256,
                started_monotonic_s=started,
                completed_monotonic_s=completed,
                # Claim exactly the policy run instead of an unbounded warmup-long
                # count; retained verification independently checks those samples.
                consecutive_active_samples=required,
                release_sample_index=(
                    int(release_sample_index)
                    if release_sample_index is not None else 0
                ),
                post_release_ready_samples=post_release_ready_samples,
                samples=tuple(samples),
            )
            self._last_receipt_completed = completed
            return receipt

        for _ in range(self.policy.maximum_samples):
            if cancel is not None and cancel():
                raise DeviceStateCancelledError(
                    "active device observation cancelled by the controller"
                )
            if not released and release is not None:
                released = release_now()
                if released:
                    release_sample_index = len(samples)
                    if consecutive < required:
                        raise DeviceStateEnvelopeError(
                            "final warmup ended before the required consecutive "
                            "pre-release active samples were observed"
                        )
            telemetry, processes = self._query(deadline=local_deadline)
            if cancel is not None and cancel():
                raise DeviceStateCancelledError(
                    "active device observation cancelled by the controller"
                )
            sampled_at = self._now()
            if sampled_at > local_deadline:
                raise DeviceStateTimeoutError(
                    "active device sample completed after its deadline"
                )
            # If release crossed while the two nvidia-smi queries were in flight,
            # this observation is neither provably pre-release nor a ready query.
            # Discard it and issue a fresh, explicitly post-release sample.
            if release is not None and not released and release_now():
                released = True
                release_sample_index = len(samples)
                if consecutive < required:
                    raise DeviceStateEnvelopeError(
                        "final warmup ended before the required consecutive "
                        "pre-release active samples were observed"
                    )
                continue
            idle, idle_reason = self._idle_verdict(telemetry, processes)
            active_passed, active_reason = self._active_envelope_verdict(
                telemetry, processes
            )
            post_release = release is not None and released
            if post_release:
                passed, reason = self._ready_envelope_verdict(telemetry, processes)
            else:
                passed, reason = active_passed, active_reason
            samples.append(
                DeviceStateSample(
                    sampled_at,
                    telemetry,
                    processes,
                    idle,
                    idle_reason,
                    passed,
                    reason,
                )
            )
            last_reason = reason
            if release is None:
                consecutive = consecutive + 1 if active_passed else 0
                if consecutive >= required:
                    return released_receipt(post_release_ready_samples=0)
            elif not released:
                consecutive = consecutive + 1 if active_passed else 0
            else:
                if not active_passed:
                    raise DeviceStateEnvelopeError(
                        "active GPU configuration failed after the final-warmup "
                        f"boundary: {active_reason}"
                    )
                if passed:
                    return released_receipt(post_release_ready_samples=1)
            remaining = local_deadline - self._now()
            if remaining <= 0:
                break
            delay = min(
                float(
                    self.policy.ready_poll_interval_s
                    if post_release else self.policy.poll_interval_s
                ),
                remaining,
            )
            if release is not None and not released and wait_for_release is not None:
                try:
                    wait_for_release(delay)
                except BaseException as exc:  # noqa: BLE001 - trusted callback
                    raise DeviceStatePolicyError(
                        f"active-conditioning release wait failed: {exc}"
                    ) from None
            else:
                self._sleep(delay)
        raise DeviceStateEnvelopeTimeoutError(
            "selected GPUs did not satisfy the active post-warmup envelope before "
            f"the bounded conditioning window ended: {last_reason}"
        )
