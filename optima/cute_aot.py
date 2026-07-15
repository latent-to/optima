"""Shared validator-owned CuTe compiler-factory contract.

Candidate code may define ``@cute.jit`` kernels and a small factory that returns a
:class:`CuteAOTCompileRequest`.  The factory is executed only by the disposable,
GPU-free native-prebuild worker. The validator fixes compiler options and retains
only the device CUBIN consumed by the separate device-only provider.

This module is intentionally import-light: importing it does not import torch,
CuTe/CUTLASS, CUDA, or candidate code. Those imports occur only in the explicit
build child. There is deliberately no host-object runtime API.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from optima.artifact_provider import (
    ARTIFACT_PROVIDERS,
    CUTE_CUBIN_PROVIDER_ID,
    ArtifactKind,
)
from optima.artifact_resource_identity import (
    ARTIFACT_RESOURCE_PLAN_SCHEMA,
    ArtifactResourceIdentityError,
    artifact_resource_plan_identity as _artifact_resource_plan_identity,
)
from optima.stack_identity import canonical_json_bytes, require_sha256_hex

if TYPE_CHECKING:
    from optima.artifact_abi import (
        ArtifactResourcePlan,
        ProviderCapabilityRequirement,
        SpecializationCapabilityRequirement,
    )
    from optima.manifest import ArtifactTargetAuthority


CUTE_AOT_CHILD_SCHEMA = "optima.cute-aot-build-child.v5"
CUTE_COMPILE_PROFILE_ENV = "OPTIMA_CUTE_COMPILE_PROFILE"
CUTE_COMPILE_PROFILE_DIGEST_ENV = "OPTIMA_CUTE_COMPILE_PROFILE_DIGEST"

_CHILD_SCHEMA = CUTE_AOT_CHILD_SCHEMA
_COMPILER_ARCH_RE = re.compile(r"sm_[0-9]{2,3}[a-z]?\Z")
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_PROFILE_KEY_RE = re.compile(
    r"[a-z][a-z0-9_]{0,31}(?:\.[a-z][a-z0-9_]{0,31}){1,3}\Z"
)
_DISTRIBUTIONS = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu13",
)
_MAX_ARTIFACT_BYTES = 1 << 30


class CuteAOTError(RuntimeError):
    """A CuTe AOT declaration, build product, or runtime reopen is invalid."""


def _digest(value: object, *, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise CuteAOTError(str(exc)) from None
    if result == "0" * 64:
        raise CuteAOTError(f"{field} must not be the all-zero digest")
    return result


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CuteAOTError(f"CuTe AOT JSON repeats key {key!r}")
        result[key] = value
    return result


def _safe_relative(value: object, *, field: str) -> str:
    try:
        encoded_length = len(value.encode("utf-8")) if isinstance(value, str) else -1
    except UnicodeEncodeError:
        encoded_length = -1
    if (
        not isinstance(value, str)
        or not value
        or encoded_length < 0
        or encoded_length > 4_096
        or "\x00" in value
        or "\\" in value
    ):
        raise CuteAOTError(f"{field} must be a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or len(path.parts) > 32
        or str(path) != value
    ):
        raise CuteAOTError(f"{field} must be a canonical relative path")
    return value


def _simple(value: object, *, field: str, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise CuteAOTError(f"{field} must be a nonempty bounded string")
    if identifier:
        if not value.isascii() or not value.isidentifier():
            raise CuteAOTError(f"{field} must be a Python/C identifier")
    elif _SAFE_COMPONENT_RE.fullmatch(value) is None:
        raise CuteAOTError(f"{field} contains unsupported characters")
    return value


def _stable_file_digest(path: Path, *, maximum: int = _MAX_ARTIFACT_BYTES) -> tuple[str, int]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CuteAOTError(f"CuTe AOT file is unavailable: {path}: {exc}") from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 1 <= before.st_size <= maximum
    ):
        raise CuteAOTError(f"CuTe AOT file has an unsafe shape: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 << 20), b""):
                digest.update(chunk)
        after = path.lstat()
    except OSError as exc:
        raise CuteAOTError(f"cannot hash CuTe AOT file {path}: {exc}") from None
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise CuteAOTError(f"CuTe AOT file changed while hashing: {path}")
    return digest.hexdigest(), before.st_size


def _stable_file_bytes(path: Path, *, maximum: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CuteAOTError(f"CuTe AOT file is unavailable: {path}: {exc}") from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 1 <= before.st_size <= maximum
    ):
        raise CuteAOTError(f"CuTe AOT file has an unsafe shape: {path}")
    try:
        with path.open("rb") as handle:
            raw = handle.read(maximum + 1)
        after = path.lstat()
    except OSError as exc:
        raise CuteAOTError(f"cannot read CuTe AOT file {path}: {exc}") from None
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(raw) != before.st_size:
        raise CuteAOTError(f"CuTe AOT file changed while reading: {path}")
    return raw


def _write_device_cubin(path: Path, value: object) -> None:
    """Write one bounded compiler-retained device image without a host export."""

    if type(value) is bytes:
        raw = value
    elif type(value) is bytearray:
        raw = bytes(value)
    elif type(value) is memoryview:
        raw = value.tobytes()
    else:
        raise CuteAOTError("CuTe compiler retained CUBIN has the wrong type")
    if not 64 <= len(raw) <= _MAX_ARTIFACT_BYTES or raw[:4] != b"\x7fELF":
        raise CuteAOTError("CuTe compiler retained CUBIN is not a bounded ELF image")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o400)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise CuteAOTError("CuTe CUBIN write stalled")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    _stable_file_digest(path)


@dataclass(frozen=True)
class CuteAOTCompileRequest:
    """One candidate factory result consumed by the validator-owned compiler.

    ``args`` and ``kwargs`` are CuTe fake/dynamic compile arguments, not live GPU
    tensors.  The request cannot alter validator compile options; in particular,
    ``no_jit_engine`` is reserved and always forced by the validator compiler.
    """

    callable: object
    args: tuple[object, ...] = ()
    kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not builtins.callable(self.callable):
            raise CuteAOTError("CuTe AOT request callable is not callable")
        if type(self.args) is not tuple:
            raise CuteAOTError("CuTe AOT request args must be an exact tuple")
        if not isinstance(self.kwargs, Mapping):
            raise CuteAOTError("CuTe AOT request kwargs must be a mapping")
        normalized: dict[str, object] = {}
        for key, value in self.kwargs.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise CuteAOTError("CuTe AOT request kwargs keys must be identifiers")
            if key in normalized:
                raise CuteAOTError(f"CuTe AOT request repeats kwarg {key!r}")
            if key in {"no_jit_engine", "options"}:
                raise CuteAOTError(f"{key} is validator-owned")
            normalized[key] = value
        object.__setattr__(self, "kwargs", MappingProxyType(normalized))


@dataclass(frozen=True)
class CuteAOTCompileContext:
    """The bounded validator-owned values visible to one candidate factory."""

    slot: str
    variant: str
    export_name: str
    provider: str
    profile_values: Mapping[str, int]

    def __post_init__(self) -> None:
        for field_name in ("slot", "variant", "export_name"):
            _simple(getattr(self, field_name), field=field_name)
        descriptor = ARTIFACT_PROVIDERS.get(self.provider)
        if (
            self.provider != CUTE_CUBIN_PROVIDER_ID
            or descriptor is None
            or descriptor.artifact_kind is not ArtifactKind.CUDA_CUBIN
        ):
            raise CuteAOTError("CuTe AOT context provider is unregistered")
        if not isinstance(self.profile_values, Mapping):
            raise CuteAOTError("CuTe AOT context profile_values must be a mapping")
        undeclared = set(self.profile_values) - descriptor.compile_profile_inputs
        if undeclared:
            raise CuteAOTError(
                "CuTe AOT context contains provider-undeclared profile inputs"
            )
        values: dict[str, int] = {}
        for key, value in self.profile_values.items():
            if (
                not isinstance(key, str)
                or len(key) > 128
                or _PROFILE_KEY_RE.fullmatch(key) is None
            ):
                raise CuteAOTError("CuTe AOT profile key is malformed")
            if type(value) is not int or value <= 0:
                raise CuteAOTError(f"CuTe AOT profile value {key!r} is not positive")
            values[key] = value
        object.__setattr__(self, "profile_values", MappingProxyType(dict(sorted(values.items()))))

    def require_int(self, key: str) -> int:
        if not isinstance(key, str):
            raise CuteAOTError("CuTe AOT profile key must be a string")
        try:
            return self.profile_values[key]
        except KeyError:
            raise CuteAOTError(
                f"CuTe AOT factory requested undeclared profile input {key!r}"
            ) from None


def compile_options_snapshot(compiler_architecture: str) -> list[dict[str, object]]:
    if (
        not isinstance(compiler_architecture, str)
        or _COMPILER_ARCH_RE.fullmatch(compiler_architecture) is None
    ):
        raise CuteAOTError("CuTe compiler architecture is not canonical")
    return [
        {"name": "GPUArch", "value": compiler_architecture},
        {"name": "OptLevel", "value": 3},
        {"name": "EnableAssertions", "value": False},
        {"name": "GenerateLineInfo", "value": False},
        {"name": "no_jit_engine", "value": True},
    ]


def installed_cute_distributions() -> dict[str, str]:
    rows: dict[str, str] = {}
    for distribution in _DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            raise CuteAOTError(
                f"CuTe AOT image lacks required distribution {distribution!r}"
            ) from None
        if not version or len(version) > 128 or not version.isascii():
            raise CuteAOTError(f"CuTe AOT distribution version is malformed: {distribution}")
        rows[distribution] = version
    return rows


def artifact_resource_plan_identity(
    *,
    slot: object | None = None,
    authority: ArtifactTargetAuthority | None = None,
    call_frame: object | None = None,
    resources: object,
) -> tuple[ArtifactResourcePlan, dict[str, object], str]:
    """Compatibility wrapper for provider-neutral artifact resource identity."""
    try:
        return _artifact_resource_plan_identity(
            slot=slot,
            authority=authority,
            call_frame=call_frame,
            resources=resources,
        )
    except ArtifactResourceIdentityError as exc:
        raise CuteAOTError(str(exc)) from None


def reopen_artifact_resource_plan_identity(
    value: object,
    *,
    expected_slot: object | None = None,
    authority: ArtifactTargetAuthority | None = None,
    expected_sha256: object,
) -> tuple[ArtifactResourcePlan, dict[str, object], str]:
    """Reparse and compare a sealed canonical resource plan."""

    from optima.artifact_abi import (
        ArtifactABIError,
        parse_artifact_resources,
    )
    from optima.manifest import (
        ArtifactTargetAuthority,
        ManifestError,
        static_artifact_target_authority,
    )

    try:
        if authority is None:
            if not isinstance(expected_slot, str):
                raise ArtifactABIError(
                    "sealed artifact resource plan requires target authority or static slot"
                )
            authority = static_artifact_target_authority(expected_slot)
        elif not isinstance(authority, ArtifactTargetAuthority):
            raise ArtifactABIError("sealed artifact target authority has the wrong type")
        elif expected_slot is not None and expected_slot != authority.dispatch_slot:
            raise ArtifactABIError("sealed artifact target dispatch slot mismatch")
        call_abi = authority.call_abi
        target_id = authority.target_id
        if not isinstance(value, dict) or set(value) != {"resources", "slot"}:
            raise ArtifactABIError("sealed artifact resource plan fields mismatch")
        if value["slot"] != target_id:
            raise ArtifactABIError("sealed artifact resource plan slot mismatch")
        plan = parse_artifact_resources(
            value["resources"],
            call_abi=call_abi,
            field="sealed artifact resource plan resources",
        )
    except (ArtifactABIError, ManifestError) as exc:
        raise CuteAOTError(f"sealed artifact resource plan is invalid: {exc}") from None
    canonical_plan, data, digest = artifact_resource_plan_identity(
        authority=authority,
        resources=plan,
    )
    if value != data:
        raise CuteAOTError("sealed artifact resource plan is not canonical")
    expected = _digest(
        expected_sha256, field="artifact resource plan sha256"
    )
    if digest != expected:
        raise CuteAOTError("artifact resource plan digest mismatch")
    return canonical_plan, data, digest


def deterministic_export_names(
    *,
    source: str,
    slot: str,
    variant: str,
    name: str,
    factory: str,
    plan: Mapping[str, object],
    artifact_resource_plan_sha256: str,
    artifact_target_authority_sha256: str | None = None,
) -> tuple[str, str]:
    canonical_plan = _canonical_launch_plan(plan)
    resource_plan_sha256 = _digest(
        artifact_resource_plan_sha256,
        field="artifact resource plan sha256",
    )
    if artifact_target_authority_sha256 is None:
        from optima.manifest import static_artifact_target_authority

        target_authority_sha256 = static_artifact_target_authority(slot).digest
    else:
        target_authority_sha256 = _digest(
            artifact_target_authority_sha256,
            field="artifact target authority sha256",
        )
    identity = canonical_json_bytes(
        {
            "artifact_resource_plan_sha256": resource_plan_sha256,
            "artifact_target_authority_sha256": target_authority_sha256,
            "factory": factory,
            "name": name,
            "plan": canonical_plan,
            "slot": slot,
            "source": source,
            "variant": variant,
        }
    )
    digest = hashlib.sha256(identity).hexdigest()
    return f"cute_{digest}", f"optima_cute_{digest}"


def _canonical_launch_plan(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the provider-neutral launch-plan projection used in identities.

    Full semantic validation against ``SlotCallABI`` happens at manifest intake and
    again while reopening the sealed index.  This helper deliberately accepts only
    the already-normalized JSON-shaped projection so candidate objects, callables,
    or arbitrary mappings can never enter an artifact identity.
    """

    if not isinstance(value, Mapping):
        raise CuteAOTError("CuTe AOT launch plan must be a mapping")
    required = {
        "bindings",
        "plan",
        "prelaunch",
        "provider_capability_requirements",
        "role",
        "specialization_capability_requirements",
        "specializes",
        "step",
    }
    optional = {"device_plan"}
    if not required <= set(value) or set(value) - required - optional:
        raise CuteAOTError("CuTe AOT launch plan fields mismatch")
    bindings = value["bindings"]
    prelaunch = value["prelaunch"]
    provider_capability_requirements = value[
        "provider_capability_requirements"
    ]
    specialization_capability_requirements = value[
        "specialization_capability_requirements"
    ]
    specializes = value["specializes"]
    if (
        not isinstance(bindings, list)
        or not all(isinstance(row, dict) for row in bindings)
        or not isinstance(prelaunch, list)
        or not all(isinstance(row, dict) for row in prelaunch)
        or not isinstance(provider_capability_requirements, list)
        or not all(
            isinstance(row, dict) for row in provider_capability_requirements
        )
        or not isinstance(specialization_capability_requirements, list)
        or not all(
            isinstance(row, dict)
            for row in specialization_capability_requirements
        )
        or not isinstance(specializes, dict)
    ):
        raise CuteAOTError("CuTe AOT launch plan has a non-canonical shape")
    plan = _simple(value["plan"], field="CuTe AOT plan")
    role = _simple(value["role"], field="CuTe AOT role")
    step = value["step"]
    if type(step) is not int or not 0 <= step <= 255:
        raise CuteAOTError("CuTe AOT plan step is outside [0, 255]")
    from optima.artifact_abi import (
        ArtifactABIError,
        parse_provider_capability_requirements,
        parse_specialization_capability_requirements,
    )

    try:
        capability_requirements = parse_provider_capability_requirements(
            provider_capability_requirements,
            field="CuTe AOT provider capability requirements",
        )
        specialization_requirements = (
            parse_specialization_capability_requirements(
                specialization_capability_requirements,
                field="CuTe AOT specialization capability requirements",
            )
        )
    except ArtifactABIError as exc:
        raise CuteAOTError(f"CuTe AOT launch plan is invalid: {exc}") from None
    # A JSON round-trip strips Mapping subclasses and proves the projection is made
    # solely of bounded JSON scalars/containers before canonical hashing.
    try:
        normalized = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CuteAOTError(f"CuTe AOT launch plan is not canonical JSON: {exc}") from None
    normalized["plan"] = plan
    normalized["provider_capability_requirements"] = [
        requirement.to_dict() for requirement in capability_requirements
    ]
    normalized["specialization_capability_requirements"] = [
        requirement.to_dict() for requirement in specialization_requirements
    ]
    normalized["role"] = role
    normalized["step"] = step
    if "device_plan" in value:
        from optima.artifact_device_launch import (
            DeviceLaunchError,
            DeviceLaunchPlan,
        )

        try:
            device_plan = DeviceLaunchPlan.from_dict(value["device_plan"])
        except DeviceLaunchError as exc:
            raise CuteAOTError(
                f"CuTe AOT device launch plan is invalid: {exc}"
            ) from None
        normalized["device_plan"] = device_plan.to_dict()
    return normalized


def export_launch_plan(export: object) -> dict[str, object]:
    """Project one validated manifest/index export into canonical launch data."""

    try:
        bindings = [binding.to_dict() for binding in export.bindings]
        prelaunch = [operation.to_dict() for operation in export.prelaunch]
        specializes = dict(export.specializes)
        value = {
            "bindings": bindings,
            "plan": export.plan,
            "prelaunch": prelaunch,
            "provider_capability_requirements": [
                requirement.to_dict()
                for requirement in export.provider_capability_requirements
            ],
            "role": export.role,
            "specialization_capability_requirements": [
                requirement.to_dict()
                for requirement in export.specialization_capability_requirements
            ],
            "specializes": specializes,
            "step": export.step,
        }
        device_plan = getattr(export, "device_plan", None)
        if device_plan is not None:
            value["device_plan"] = device_plan.to_dict()
    except (AttributeError, TypeError, ValueError) as exc:
        raise CuteAOTError(f"CuTe AOT export launch plan is malformed: {exc}") from None
    return _canonical_launch_plan(value)


def _read_canonical_json(path: Path, *, maximum: int) -> tuple[dict[str, Any], bytes]:
    raw = _stable_file_bytes(path, maximum=maximum)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except CuteAOTError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CuteAOTError(f"CuTe AOT JSON is malformed: {path}: {exc}") from None
    if not isinstance(value, dict):
        raise CuteAOTError(f"CuTe AOT JSON has an unsafe shape: {path}")
    if raw != canonical_json_bytes(value) + b"\n":
        raise CuteAOTError(f"CuTe AOT JSON is not canonical: {path}")
    return value, raw


def load_compile_profile(path: str | Path, *, expected_digest: str):
    """Reopen an exact validator profile without importing CuTe/CUDA."""

    from optima.eval.native_compile_profile import (
        NativeCompileProfileError,
        NativeCuTeCompileProfile,
    )

    requested = Path(path)
    row, raw = _read_canonical_json(requested, maximum=1 << 20)
    try:
        profile = NativeCuTeCompileProfile.from_dict(row)
    except NativeCompileProfileError as exc:
        raise CuteAOTError(f"CuTe AOT compile profile is invalid: {exc}") from None
    if raw != profile.canonical_bytes:
        raise CuteAOTError("CuTe AOT compile profile is not its canonical projection")
    if profile.digest != _digest(expected_digest, field="compile profile digest"):
        raise CuteAOTError("CuTe AOT compile profile digest mismatch")
    return profile


def build_export_in_child(spec_path: str | Path) -> None:
    """Child-only compiler entry.  It never opens the native publication stage."""

    spec, _raw = _read_canonical_json(Path(spec_path), maximum=1 << 20)
    expected_fields = {
        "schema",
        "bundle",
        "source",
        "slot",
        "variant",
        "name",
        "factory",
        "provider",
        "launch_plan",
        "profile_values",
        "compiler_architecture",
        "artifact_id",
        "artifact_target_authority",
        "artifact_target_authority_sha256",
        "artifact_resource_plan",
        "artifact_resource_plan_sha256",
        "function_prefix",
        "output_directory",
    }
    if set(spec) != expected_fields or spec.get("schema") != _CHILD_SCHEMA:
        raise CuteAOTError("CuTe AOT build-child specification mismatch")
    if os.environ.get("OPTIMA_REBUILD_CONTAINER") != "1":
        raise CuteAOTError("CuTe AOT child requires the disposable rebuild container")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CuteAOTError(
            "CuTe AOT child requires CUDA_VISIBLE_DEVICES to be explicitly empty"
        )
    bundle = Path(spec["bundle"])
    source_relative = _safe_relative(spec["source"], field="CuTe AOT child source")
    try:
        bundle = bundle.resolve(strict=True)
        source = bundle.joinpath(*PurePosixPath(source_relative).parts).resolve(strict=True)
    except OSError as exc:
        raise CuteAOTError(f"CuTe AOT child source cannot reopen: {exc}") from None
    if bundle not in source.parents or source.is_symlink() or not source.is_file():
        raise CuteAOTError("CuTe AOT child source escapes its bundle")
    output = Path(spec["output_directory"])
    if not output.is_absolute() or output.is_symlink() or not output.is_dir():
        raise CuteAOTError("CuTe AOT child output directory is invalid")
    if any(output.iterdir()):
        raise CuteAOTError("CuTe AOT child output directory is not empty")
    context = CuteAOTCompileContext(
        slot=spec["slot"],
        variant=spec["variant"],
        export_name=spec["name"],
        provider=spec["provider"],
        profile_values=spec["profile_values"],
    )
    compiler_architecture = spec["compiler_architecture"]
    from optima.cute_cubin import (
        compile_options_snapshot as cubin_compile_options_snapshot,
    )

    cubin_compile_options_snapshot(compiler_architecture)
    artifact_id = _simple(spec["artifact_id"], field="artifact ID", identifier=True)
    function_prefix = _simple(
        spec["function_prefix"], field="function prefix", identifier=True
    )
    launch_plan = _canonical_launch_plan(spec["launch_plan"])
    from optima.manifest import ManifestError, reopen_artifact_target_authority

    try:
        target_authority = reopen_artifact_target_authority(
            spec["artifact_target_authority"],
            expected_dispatch_slot=context.slot,
        )
    except ManifestError as exc:
        raise CuteAOTError(f"CuTe AOT child target authority is invalid: {exc}") from None
    target_authority_sha256 = _digest(
        spec["artifact_target_authority_sha256"],
        field="CuTe AOT child target authority sha256",
    )
    if target_authority_sha256 != target_authority.digest:
        raise CuteAOTError("CuTe AOT child target authority digest mismatch")
    _resource_plan, _resource_plan_data, resource_plan_sha256 = (
        reopen_artifact_resource_plan_identity(
            spec["artifact_resource_plan"],
            authority=target_authority,
            expected_sha256=spec["artifact_resource_plan_sha256"],
        )
    )
    expected_artifact_id, expected_function_prefix = deterministic_export_names(
        source=source_relative,
        slot=context.slot,
        variant=context.variant,
        name=context.export_name,
        factory=_simple(spec["factory"], field="CuTe AOT factory", identifier=True),
        plan=launch_plan,
        artifact_resource_plan_sha256=resource_plan_sha256,
        artifact_target_authority_sha256=target_authority_sha256,
    )
    if (artifact_id, function_prefix) != (
        expected_artifact_id,
        expected_function_prefix,
    ):
        raise CuteAOTError("CuTe AOT child export names differ from its declaration")

    # Capture the validator compiler/options before executing candidate module code.
    # The no-egress OCI worker remains the security boundary; capturing these objects
    # prevents ordinary module-global rebinding from selecting a different compiler.
    import cutlass.cute as cute

    compiler_options = [
        cute.GPUArch(compiler_architecture),
        cute.OptLevel(3),
        cute.EnableAssertions(False),
        cute.GenerateLineInfo(False),
        cute.KeepCUBIN(),
    ]
    compiler = cute.compile[tuple(compiler_options)]
    compile_call = compiler.__call__
    sys.path.insert(0, str(bundle))
    try:
        from optima.sandbox import callable_from, load_module

        module = load_module(source)
        factory = callable_from(
            module,
            _simple(spec["factory"], field="CuTe AOT factory", identifier=True),
        )
        request = factory(context)
        if type(request) is not CuteAOTCompileRequest:
            raise CuteAOTError(
                "CuTe AOT factory must return an exact CuteAOTCompileRequest"
            )
        compiled = compile_call(
            request.callable,
            *request.args,
            **dict(request.kwargs),
            no_jit_engine=True,
        )
        try:
            cubin = compiled.__cubin__
            if builtins.callable(cubin):
                cubin = cubin()
        except (AttributeError, TypeError) as exc:
            raise CuteAOTError(
                f"CuTe compiler did not retain a CUBIN: {exc}"
            ) from None
        _write_device_cubin(output / f"{artifact_id}.cubin", cubin)
    finally:
        try:
            sys.path.remove(str(bundle))
        except ValueError:
            pass
    expected = {f"{artifact_id}.cubin"}
    if {path.name for path in output.iterdir()} != expected:
        raise CuteAOTError("CuTe AOT compiler emitted an unexpected file set")
    _stable_file_digest(output / f"{artifact_id}.cubin")


def _entry_with_use_evidence(entry: object, payload: Mapping[str, object]) -> object:
    """Wrap one trusted entry with a once-per-process successful-use receipt."""

    if not builtins.callable(entry):
        raise CuteAOTError("CuTe AOT evidence wrapper received a non-callable entry")
    from optima import receipts

    entry_call = entry.__call__
    write_receipt = receipts.write
    lock = threading.Lock()
    body = dict(payload)
    tag = f"{body['slot']}.{body['variant']}"
    written = False

    def evidenced(*args, **kwargs):
        nonlocal written
        result = entry_call(*args, **kwargs)
        if not written:
            with lock:
                if not written:
                    write_receipt("aot_invoked", body, tag=tag)
                    written = True
        return result

    close = getattr(entry, "close", None)
    if builtins.callable(close):
        # Preserve the provider-neutral lifecycle close hook without exposing the
        # underlying entry or executor table to candidate code.
        evidenced.close = close  # type: ignore[attr-defined]
    prepare = getattr(entry, "prepare", None)
    if builtins.callable(prepare):
        # This is validator-generated lifecycle code, not a candidate callback.
        # MoE dispatchers call it outside the captured forward to obtain the
        # owner-bound ArtifactPreparedState consumed by the direct run entry.
        evidenced.prepare = prepare  # type: ignore[attr-defined]
    return evidenced


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m optima.cute_aot")
    parser.add_argument("--build-export", metavar="SPEC")
    args = parser.parse_args(argv)
    if not args.build_export:
        parser.error("only the isolated --build-export entry is supported")
    build_export_in_child(args.build_export)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ARTIFACT_RESOURCE_PLAN_SCHEMA",
    "CUTE_AOT_CHILD_SCHEMA",
    "CUTE_COMPILE_PROFILE_DIGEST_ENV",
    "CUTE_COMPILE_PROFILE_ENV",
    "CuteAOTCompileContext",
    "CuteAOTCompileRequest",
    "CuteAOTError",
    "artifact_resource_plan_identity",
    "build_export_in_child",
    "compile_options_snapshot",
    "deterministic_export_names",
    "installed_cute_distributions",
    "load_compile_profile",
    "reopen_artifact_resource_plan_identity",
]
