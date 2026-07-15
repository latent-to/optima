"""Validator-owned hardware inputs for GPU-free CuTe AOT compilation.

CuTe kernels may need hardware constants while they are traced (for example the
maximum number of active clusters).  Candidate code must not discover those
values itself and the no-egress prebuild intentionally has no GPU.  This module
defines the small, canonical profile that the trusted arena measures first and
mounts read-only into prebuild.

The profile is data only.  It neither imports CuTe/CUDA nor executes a probe.
Probe collection is arena provisioning authority; this type makes its result
portable, digest-bound, and independently checkable against one launch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from optima.artifact_provider import (
    ARTIFACT_PROVIDERS,
    CUTE_CUBIN_PROVIDER_ID,
    ArtifactKind,
)
from optima.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


PROFILE_SCHEMA = "optima.native-cute-compile-profile.v1"

_LOGICAL_ARCH = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")
_COMPILER_ARCH = re.compile(r"sm_[0-9]{2,3}[a-z]?\Z")
_PROFILE_KEY = re.compile(
    r"[a-z][a-z0-9_]{0,31}(?:\.[a-z][a-z0-9_]{0,31}){1,3}\Z"
)
_MAX_CONSTANTS = 32
_MAX_DEGREE = 65_536
_MAX_CONSTANT = 1_048_576


class NativeCompileProfileError(ValueError):
    """A native compile profile is malformed or differs from its launch."""


def _digest(value: object, *, field: str) -> str:
    try:
        checked = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise NativeCompileProfileError(str(exc)) from None
    if checked == "0" * 64:
        raise NativeCompileProfileError(f"{field} must not be the all-zero digest")
    return checked


def _degree(value: object, *, field: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_DEGREE:
        raise NativeCompileProfileError(
            f"compile profile {field} must be an integer in [1, {_MAX_DEGREE}]"
        )
    return value


def _architecture(value: object, *, compiler: bool) -> str:
    pattern = _COMPILER_ARCH if compiler else _LOGICAL_ARCH
    field = "compiler_architecture" if compiler else "logical_architecture"
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        example = "sm_103a" if compiler else "sm103"
        raise NativeCompileProfileError(
            f"compile profile {field} must be canonical (for example {example!r})"
        )
    return value


def _compile_profile_provider(provider: object):
    descriptor = ARTIFACT_PROVIDERS.get(provider)
    if (
        descriptor is None
        or descriptor.artifact_kind is not ArtifactKind.CUDA_CUBIN
        or not descriptor.requires_compile_profile
    ):
        raise NativeCompileProfileError(
            "native CuTe compile profile provider is unregistered"
        )
    return descriptor


def _normalize_constants(
    value: object, *, allowed: frozenset[str]
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping) or len(value) > _MAX_CONSTANTS:
        raise NativeCompileProfileError(
            f"compile profile constants must be a mapping with at most {_MAX_CONSTANTS} rows"
        )
    rows: list[tuple[str, int]] = []
    for key, raw in value.items():
        if (
            not isinstance(key, str)
            or len(key) > 128
            or _PROFILE_KEY.fullmatch(key) is None
            or key not in allowed
        ):
            raise NativeCompileProfileError(
                f"compile profile constant is not registered: {key!r}"
            )
        if type(raw) is not int or not 1 <= raw <= _MAX_CONSTANT:
            raise NativeCompileProfileError(
                f"compile profile constant {key!r} is outside its hard bound"
            )
        rows.append((key, raw))
    rows.sort()
    if len(rows) != len({key for key, _ in rows}):
        raise NativeCompileProfileError("compile profile constants contain duplicates")
    return tuple(rows)


@dataclass(frozen=True)
class NativeCuTeCompileProfile:
    """Canonical, validator-measured inputs consumed by one CuTe AOT build."""

    logical_architecture: str
    compiler_architecture: str
    image_digest: str
    platform_digest: str
    worker_distribution_digest: str
    logical_hardware_digest: str
    device_policy_digest: str
    topology_digest: str
    visible_gpu_count: int
    tp_size: int
    ep_size: int
    dp_size: int
    constants: tuple[tuple[str, int], ...] | Mapping[str, int]
    measurement_digest: str
    provider: str = CUTE_CUBIN_PROVIDER_ID
    schema: str = PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != PROFILE_SCHEMA:
            raise NativeCompileProfileError(
                "native CuTe compile profile schema mismatch"
            )
        if type(self.provider) is not str:
            raise NativeCompileProfileError(
                "native CuTe compile profile provider is unregistered"
            )
        provider = _compile_profile_provider(self.provider)
        logical = _architecture(self.logical_architecture, compiler=False)
        compiler = _architecture(self.compiler_architecture, compiler=True)
        compiler_family = compiler.removeprefix("sm_").rstrip("a")
        logical_family = logical.removeprefix("sm").rstrip("a")
        if compiler_family != logical_family:
            raise NativeCompileProfileError(
                "compiler architecture does not name the logical architecture family"
            )
        object.__setattr__(self, "logical_architecture", logical)
        object.__setattr__(self, "compiler_architecture", compiler)
        for field in (
            "image_digest",
            "platform_digest",
            "worker_distribution_digest",
            "logical_hardware_digest",
            "device_policy_digest",
            "topology_digest",
            "measurement_digest",
        ):
            object.__setattr__(
                self, field, _digest(getattr(self, field), field=field)
            )
        visible = _degree(self.visible_gpu_count, field="visible_gpu_count")
        object.__setattr__(self, "visible_gpu_count", visible)
        for field in ("tp_size", "ep_size", "dp_size"):
            degree = _degree(getattr(self, field), field=field)
            if degree > visible:
                raise NativeCompileProfileError(
                    f"compile profile {field} cannot exceed visible_gpu_count"
                )
            object.__setattr__(self, field, degree)
        raw_constants: object
        if isinstance(self.constants, Mapping):
            raw_constants = self.constants
        elif type(self.constants) is tuple:
            try:
                raw_constants = dict(self.constants)
            except (TypeError, ValueError) as exc:
                raise NativeCompileProfileError(
                    f"compile profile constants are malformed: {exc}"
                ) from None
            if len(raw_constants) != len(self.constants):
                raise NativeCompileProfileError(
                    "compile profile constants contain duplicates"
                )
        else:
            raise NativeCompileProfileError(
                "compile profile constants have the wrong type"
            )
        object.__setattr__(
            self,
            "constants",
            _normalize_constants(
                raw_constants,
                allowed=provider.compile_profile_inputs,
            ),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.eval.native-cute-compile-profile", self.to_dict()
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    @property
    def values(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.constants))

    def require_int(self, key: str) -> int:
        allowed = _compile_profile_provider(self.provider).compile_profile_inputs
        if not isinstance(key, str) or key not in allowed:
            raise NativeCompileProfileError(
                f"compile profile input is not registered: {key!r}"
            )
        try:
            return dict(self.constants)[key]
        except KeyError:
            raise NativeCompileProfileError(
                f"compile profile does not provide required input {key!r}"
            ) from None

    def validate_launch(
        self,
        *,
        image_digest: str,
        platform_digest: str,
        worker_distribution_digest: str,
        logical_hardware_digest: str,
        logical_architecture: str,
        device_policy_digest: str,
        topology_digest: str,
        visible_gpu_count: int,
        tp_size: int,
        ep_size: int,
        dp_size: int,
    ) -> None:
        expected: dict[str, object] = {
            "image_digest": _digest(image_digest, field="launch image_digest"),
            "platform_digest": _digest(platform_digest, field="launch platform_digest"),
            "worker_distribution_digest": _digest(
                worker_distribution_digest,
                field="launch worker_distribution_digest",
            ),
            "logical_hardware_digest": _digest(
                logical_hardware_digest, field="launch logical_hardware_digest"
            ),
            "logical_architecture": _architecture(
                logical_architecture, compiler=False
            ),
            "device_policy_digest": _digest(
                device_policy_digest, field="launch device_policy_digest"
            ),
            "topology_digest": _digest(
                topology_digest, field="launch topology_digest"
            ),
            "visible_gpu_count": _degree(
                visible_gpu_count, field="launch visible_gpu_count"
            ),
            "tp_size": _degree(tp_size, field="launch tp_size"),
            "ep_size": _degree(ep_size, field="launch ep_size"),
            "dp_size": _degree(dp_size, field="launch dp_size"),
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise NativeCompileProfileError(
                    f"compile profile {field} differs from launch authority"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "compiler_architecture": self.compiler_architecture,
            "constants": dict(self.constants),
            "degrees": {
                "dp_size": self.dp_size,
                "ep_size": self.ep_size,
                "tp_size": self.tp_size,
                "visible_gpu_count": self.visible_gpu_count,
            },
            "device_policy_digest": self.device_policy_digest,
            "image_digest": self.image_digest,
            "logical_architecture": self.logical_architecture,
            "logical_hardware_digest": self.logical_hardware_digest,
            "measurement_digest": self.measurement_digest,
            "platform_digest": self.platform_digest,
            "provider": self.provider,
            "schema": self.schema,
            "topology_digest": self.topology_digest,
            "worker_distribution_digest": self.worker_distribution_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NativeCuTeCompileProfile":
        fields = {
            "compiler_architecture",
            "constants",
            "degrees",
            "device_policy_digest",
            "image_digest",
            "logical_architecture",
            "logical_hardware_digest",
            "measurement_digest",
            "platform_digest",
            "provider",
            "schema",
            "topology_digest",
            "worker_distribution_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise NativeCompileProfileError(
                "native CuTe compile profile fields mismatch"
            )
        degrees = value["degrees"]
        degree_fields = {"visible_gpu_count", "tp_size", "ep_size", "dp_size"}
        if not isinstance(degrees, Mapping) or set(degrees) != degree_fields:
            raise NativeCompileProfileError(
                "native CuTe compile profile degrees mismatch"
            )
        return cls(
            logical_architecture=value["logical_architecture"],  # type: ignore[arg-type]
            compiler_architecture=value["compiler_architecture"],  # type: ignore[arg-type]
            image_digest=value["image_digest"],  # type: ignore[arg-type]
            platform_digest=value["platform_digest"],  # type: ignore[arg-type]
            worker_distribution_digest=value["worker_distribution_digest"],  # type: ignore[arg-type]
            logical_hardware_digest=value["logical_hardware_digest"],  # type: ignore[arg-type]
            device_policy_digest=value["device_policy_digest"],  # type: ignore[arg-type]
            topology_digest=value["topology_digest"],  # type: ignore[arg-type]
            visible_gpu_count=degrees["visible_gpu_count"],  # type: ignore[arg-type]
            tp_size=degrees["tp_size"],  # type: ignore[arg-type]
            ep_size=degrees["ep_size"],  # type: ignore[arg-type]
            dp_size=degrees["dp_size"],  # type: ignore[arg-type]
            constants=value["constants"],  # type: ignore[arg-type]
            measurement_digest=value["measurement_digest"],  # type: ignore[arg-type]
            provider=value["provider"],  # type: ignore[arg-type]
            schema=value["schema"],  # type: ignore[arg-type]
        )


__all__ = [
    "NativeCompileProfileError",
    "NativeCuTeCompileProfile",
    "PROFILE_SCHEMA",
]
