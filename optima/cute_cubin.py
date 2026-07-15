"""Sealed device-only CuTe artifact contract.

The compiler factory remains the bounded :mod:`optima.cute_aot` factory.  Its
authoritative product for this provider is only a CUDA CUBIN; generated host
objects, Python launchers, PTX, and JIT engines are never published.

This module is the GPU-free publication boundary.  Reopening an index parses
only bounded data, exact file identities, and the fixed CUDA ELF header gate.
The isolated engine worker performs the authoritative Driver-API admission on
the same library handle that it later launches; prebuild never imports or calls
the CUDA driver.
"""

from __future__ import annotations

import builtins
import hashlib
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Mapping

from optima.artifact_provider import (
    CUTE_CUBIN_PATCHER_ID,
    CUTE_CUBIN_PROVIDER,
    CUTE_CUBIN_PROVIDER_ID,
    ArtifactBindingABI,
)
from optima.artifact_device_launch import DeviceLaunchError, DeviceLaunchPlan
from optima.cuda_cubin import CudaCubinError, _require_elf_cubin
from optima.cute_aot import (
    CuteAOTError,
    _canonical_launch_plan,
    _digest,
    _read_canonical_json,
    _safe_relative,
    _simple,
    _stable_file_bytes,
    _stable_file_digest,
    compile_options_snapshot as _host_compile_options_snapshot,
    deterministic_export_names,
    installed_cute_distributions,
    reopen_artifact_resource_plan_identity,
)
from optima.stack_identity import canonical_json_bytes


CUTE_CUBIN_SCHEMA = "optima.cute-cubin-set.v1"
CUTE_CUBIN_PROVIDER_NAME = CUTE_CUBIN_PROVIDER_ID
CUTE_CUBIN_PATCHER = CUTE_CUBIN_PATCHER_ID
CUTE_CUBIN_BINDING_ABI = ArtifactBindingABI.CUDA_DRIVER_PARAMS_V1.value
CUTE_CUBIN_STAGE_DIRECTORY = CUTE_CUBIN_PROVIDER.publication_directory
CUTE_CUBIN_INDEX_NAME = "index.json"
CUTE_CUBIN_INDEX_RELPATH = (
    f"{CUTE_CUBIN_STAGE_DIRECTORY}/{CUTE_CUBIN_INDEX_NAME}"
)
CUTE_CUBIN_ARTIFACT_MAX_LIVE_BYTES = 16 << 30
CUTE_CUBIN_ARTIFACT_MAX_ALLOCATION_KEYS = 4_096
_LOGICAL_ARCH_RE = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")
_COMPILER_ARCH_RE = re.compile(r"sm_[0-9]{2,3}[a-z]?\Z")
_PROFILE_KEY_RE = re.compile(
    r"[a-z][a-z0-9_]{0,31}(?:\.[a-z][a-z0-9_]{0,31}){1,3}\Z"
)
_INDEX_FIELDS = frozenset(
    {
        "binding_abi",
        "build_spec_digest",
        "compile_options",
        "compile_profile_digest",
        "compiler_architecture",
        "distributions",
        "exports",
        "logical_architecture",
        "patcher_id",
        "patcher_sha256",
        "provider",
        "schema",
        "tree_digest",
    }
)
_EXPORT_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_resource_plan",
        "artifact_resource_plan_sha256",
        "artifact_target_authority",
        "artifact_target_authority_sha256",
        "cubin",
        "device_plan",
        "device_plan_sha256",
        "factory",
        "launch_plan",
        "launch_plan_sha256",
        "name",
        "profile_inputs",
        "resolved_profile",
        "slot",
        "source",
        "variant",
    }
)
_FILE_FIELDS = frozenset({"path", "sha256", "size"})
_MAX_INDEX_BYTES = 16 << 20
_MAX_CUBIN_BYTES = 1 << 30
_MAX_EXPORTS = 1_024
_DISTRIBUTIONS = frozenset(
    {
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-cu13",
    }
)


class CuteCubinError(RuntimeError):
    """A device-only CuTe declaration, product, or receipt is invalid."""


@dataclass(frozen=True)
class CuteCubinFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class CuteCubinExportArtifact:
    source: str
    slot: str
    variant: str
    name: str
    factory: str
    artifact_id: str
    launch_plan: Mapping[str, object]
    launch_plan_sha256: str
    device_plan: DeviceLaunchPlan
    device_plan_sha256: str
    profile_inputs: tuple[str, ...]
    resolved_profile: tuple[tuple[str, int], ...]
    artifact_target_authority: object
    artifact_target_authority_sha256: str
    artifact_resource_plan: object
    artifact_resource_plan_sha256: str
    cubin: CuteCubinFile


@dataclass(frozen=True)
class CuteCubinIndex:
    root: Path
    build_spec_digest: str
    tree_digest: str
    compile_profile_digest: str
    logical_architecture: str
    compiler_architecture: str
    patcher_sha256: str
    distributions: tuple[tuple[str, str], ...]
    compile_options: tuple[tuple[str, object], ...]
    exports: tuple[CuteCubinExportArtifact, ...]


@dataclass(frozen=True)
class _CuteCubinRuntimeState:
    authority: tuple[object, ...]
    bindings: Mapping[tuple[str, str], object]
    keepalive: tuple[object, ...]


_RUNTIME_LOCK = threading.RLock()
_RUNTIME_STATE: _CuteCubinRuntimeState | None = None


def compile_options_snapshot(compiler_architecture: str) -> list[dict[str, object]]:
    """Exact compiler policy for a retained device image."""

    rows = _host_compile_options_snapshot(compiler_architecture)
    # Retention changes only what the trusted compiler returns.  PTX/MLIR are
    # intentionally not retained by the authoritative provider.
    rows.insert(-1, {"name": "KeepCUBIN", "value": True})
    return rows


def _file_row(
    root: Path,
    row: object,
    *,
    artifact_id: str,
    allowed_paths: set[str],
) -> CuteCubinFile:
    if not isinstance(row, dict) or set(row) != _FILE_FIELDS:
        raise CuteCubinError("CuTe CUBIN file receipt fields mismatch")
    relative = _safe_relative(row["path"], field="CuTe CUBIN path")
    expected_relative = (
        f"{CUTE_CUBIN_STAGE_DIRECTORY}/cubins/{artifact_id}/{artifact_id}.cubin"
    )
    if relative != expected_relative or relative in allowed_paths:
        raise CuteCubinError("CuTe CUBIN path differs from its artifact identity")
    size = row["size"]
    if type(size) is not int or not 64 <= size <= _MAX_CUBIN_BYTES:
        raise CuteCubinError("CuTe CUBIN size is outside policy")
    expected_sha256 = _digest(row["sha256"], field="CuTe CUBIN sha256")
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        observed_sha256, observed_size = _stable_file_digest(
            path, maximum=_MAX_CUBIN_BYTES
        )
    except Exception as exc:
        raise CuteCubinError(f"CuTe CUBIN cannot reopen: {exc}") from None
    if (observed_sha256, observed_size) != (expected_sha256, size):
        raise CuteCubinError("CuTe CUBIN bytes differ from their receipt")
    allowed_paths.add(relative)
    return CuteCubinFile(relative, size, expected_sha256)


def _canonical_options(
    value: object, *, compiler_architecture: str
) -> tuple[tuple[str, object], ...]:
    expected = compile_options_snapshot(compiler_architecture)
    if value != expected:
        raise CuteCubinError("CuTe CUBIN compile options differ from policy")
    return tuple((row["name"], row["value"]) for row in expected)


def _reopen_launch_plan(
    value: object, *, target_authority: object, resource_plan: object
) -> dict[str, object]:
    try:
        canonical = _canonical_launch_plan(value)  # type: ignore[arg-type]
        from optima.artifact_abi import (
            ArtifactABIError,
            parse_artifact_bindings,
            parse_artifact_prelaunch,
            parse_provider_capability_requirements,
            parse_specialization_capability_requirements,
        )

        call_abi = target_authority.call_abi
        bindings = parse_artifact_bindings(
            canonical["bindings"], field="sealed CuTe CUBIN bindings"
        )
        prelaunch = parse_artifact_prelaunch(
            canonical["prelaunch"], field="sealed CuTe CUBIN prelaunch"
        )
        provider_requirements = parse_provider_capability_requirements(
            canonical["provider_capability_requirements"],
            field="sealed CuTe CUBIN provider capability requirements",
        )
        specialization_requirements = (
            parse_specialization_capability_requirements(
                canonical["specialization_capability_requirements"],
                field=(
                    "sealed CuTe CUBIN specialization capability requirements"
                ),
            )
        )
        try:
            device_plan = DeviceLaunchPlan.from_dict(canonical["device_plan"])
            device_plan.validate_bindings(
                bindings,
                provider_capabilities=CUTE_CUBIN_PROVIDER.provider_capabilities,
            )
        except (KeyError, DeviceLaunchError) as exc:
            raise ArtifactABIError(
                f"sealed device launch plan is invalid: {exc}"
            ) from None
        specializes = call_abi.validate_plan(
            role=canonical["role"],
            bindings=bindings,
            specializes=canonical["specializes"],
            prelaunch=prelaunch,
            require_outputs=False,
            artifact_resources=resource_plan,
        )
        current_provider_requirements = call_abi.provider_capability_requirements(
            bindings,
            artifact_resources=resource_plan,
        )
        current_specialization_requirements = (
            call_abi.specialization_capability_requirements(
                specializes,
                artifact_resources=resource_plan,
            )
        )
        if provider_requirements != current_provider_requirements:
            raise ArtifactABIError(
                "sealed provider requirements differ from validator projection"
            )
        if specialization_requirements != current_specialization_requirements:
            raise ArtifactABIError(
                "sealed specialization requirements differ from validator projection"
            )
        reopened = _canonical_launch_plan(
            {
                "bindings": [binding.to_dict() for binding in bindings],
                "device_plan": device_plan.to_dict(),
                "plan": canonical["plan"],
                "prelaunch": [operation.to_dict() for operation in prelaunch],
                "provider_capability_requirements": [
                    requirement.to_dict()
                    for requirement in current_provider_requirements
                ],
                "role": canonical["role"],
                "specialization_capability_requirements": [
                    requirement.to_dict()
                    for requirement in current_specialization_requirements
                ],
                "specializes": dict(specializes),
                "step": canonical["step"],
            }
        )
    except (AttributeError, TypeError, ValueError, ArtifactABIError, CuteAOTError) as exc:
        raise CuteCubinError(f"CuTe CUBIN launch plan is invalid: {exc}") from None
    if reopened != canonical:
        raise CuteCubinError(
            "CuTe CUBIN launch plan differs from current validator projection"
        )
    return reopened


def _publication_inventory(root: Path, *, allowed_paths: set[str]) -> None:
    publication = root / CUTE_CUBIN_STAGE_DIRECTORY
    try:
        root_info = publication.lstat()
    except OSError as exc:
        raise CuteCubinError(f"CuTe CUBIN publication is unavailable: {exc}") from None
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise CuteCubinError("CuTe CUBIN publication must be a real directory")
    observed_files: set[str] = set()
    observed_directories: set[str] = {CUTE_CUBIN_STAGE_DIRECTORY}
    try:
        for path in publication.rglob("*"):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise CuteCubinError("CuTe CUBIN publication contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                observed_directories.add(path.relative_to(root).as_posix())
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise CuteCubinError(
                    "CuTe CUBIN publication contains an unsafe filesystem object"
                )
            observed_files.add(path.relative_to(root).as_posix())
    except OSError as exc:
        raise CuteCubinError(f"cannot inventory CuTe CUBIN publication: {exc}") from None
    if observed_files != allowed_paths:
        raise CuteCubinError("CuTe CUBIN publication contains unreceipted files")
    expected_directories = {CUTE_CUBIN_STAGE_DIRECTORY}
    for relative in allowed_paths:
        path = PurePosixPath(relative)
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            if parent.parts and parent.parts[0] == CUTE_CUBIN_STAGE_DIRECTORY:
                expected_directories.add(parent.as_posix())
    if observed_directories != expected_directories:
        raise CuteCubinError(
            "CuTe CUBIN publication contains unreceipted directories"
        )


def _validate_reopened_pipelines(
    exports: list[CuteCubinExportArtifact],
) -> None:
    """Repeat the manifest's cross-export ABI checks over sealed rows.

    An artifact ID binds each individual launch plan, but identity is not a
    substitute for semantic validation: a consistently re-hashed publication
    must still collectively write every slot output and obey lifecycle/storage
    ordering.  This reconstruction consumes only validator-owned ABI objects and
    the closed launch language.
    """

    from optima.artifact_abi import (
        ArtifactABIError,
        parse_artifact_bindings,
        parse_artifact_prelaunch,
    )

    def pipeline_step(
        export: CuteCubinExportArtifact,
    ) -> tuple[str, tuple[object, ...], tuple[object, ...]]:
        try:
            return (
                str(export.launch_plan["role"]),
                tuple(
                    parse_artifact_bindings(
                        export.launch_plan["bindings"],
                        field="sealed CuTe CUBIN pipeline bindings",
                    )
                ),
                tuple(
                    parse_artifact_prelaunch(
                        export.launch_plan["prelaunch"],
                        field="sealed CuTe CUBIN pipeline prelaunch",
                    )
                ),
            )
        except (ArtifactABIError, KeyError, TypeError, ValueError) as exc:
            raise CuteCubinError(
                f"CuTe CUBIN launch plan pipeline cannot reopen: {exc}"
            ) from None

    by_op: dict[tuple[str, str], list[CuteCubinExportArtifact]] = {}
    for export in exports:
        by_op.setdefault((export.slot, export.variant), []).append(export)
    for op_key, op_exports in by_op.items():
        if (
            len({export.source for export in op_exports}) != 1
            or len(
                {
                    export.artifact_target_authority_sha256
                    for export in op_exports
                }
            )
            != 1
            or len(
                {
                    export.artifact_resource_plan_sha256
                    for export in op_exports
                }
            )
            != 1
        ):
            raise CuteCubinError(
                f"CuTe CUBIN launch plan authority is inconsistent for {op_key!r}"
            )
        by_plan: dict[str, list[CuteCubinExportArtifact]] = {}
        for export in op_exports:
            plan_name = export.launch_plan["plan"]
            assert isinstance(plan_name, str)
            by_plan.setdefault(plan_name, []).append(export)
        for plan_name, plan_exports in by_plan.items():
            ordered = sorted(
                plan_exports,
                key=lambda export: int(export.launch_plan["step"]),
            )
            steps = [int(export.launch_plan["step"]) for export in ordered]
            if len(steps) != len(set(steps)):
                raise CuteCubinError(
                    f"CuTe CUBIN launch plan {plan_name!r} repeats a step"
                )
            specializes = [
                tuple(
                    (key, type(value), value)
                    for key, value in export.launch_plan["specializes"].items()
                )
                for export in ordered
            ]
            if any(value != specializes[0] for value in specializes[1:]):
                raise CuteCubinError(
                    f"CuTe CUBIN launch plan {plan_name!r} has inconsistent "
                    "specializations"
                )
            pipeline = tuple(pipeline_step(export) for export in ordered)
            authority = ordered[0].artifact_target_authority
            resource_plan = ordered[0].artifact_resource_plan
            try:
                authority.call_abi.validate_pipeline(
                    pipeline,
                    artifact_resources=resource_plan,
                )
            except (ArtifactABIError, AttributeError, TypeError, ValueError) as exc:
                raise CuteCubinError(
                    f"CuTe CUBIN launch plan {plan_name!r} pipeline is invalid: {exc}"
                ) from None
        complete_pipeline = tuple(
            pipeline_step(export)
            for export in sorted(
                op_exports,
                key=lambda export: (
                    str(export.launch_plan["plan"]),
                    int(export.launch_plan["step"]),
                    export.name,
                ),
            )
        )
        try:
            op_exports[0].artifact_resource_plan.validate_pipeline(
                complete_pipeline,
                require_all=True,
            )
        except (ArtifactABIError, AttributeError, TypeError, ValueError) as exc:
            raise CuteCubinError(
                f"CuTe CUBIN launch plan resources are invalid: {exc}"
            ) from None


def _reopen_cute_cubin_index(
    artifact_root: str | Path,
    *,
    expected_build_spec_digest: str | None = None,
    expected_tree_digest: str | None = None,
    expected_logical_architecture: str | None = None,
    expected_compile_profile_digest: str | None = None,
    expected_patcher_sha256: str | None = None,
    verify_distributions: bool = True,
) -> CuteCubinIndex:
    """Reopen the complete device publication without loading CUDA."""

    root = Path(artifact_root)
    try:
        info = root.lstat()
    except OSError as exc:
        raise CuteCubinError(f"CuTe CUBIN artifact root is unavailable: {exc}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CuteCubinError("CuTe CUBIN artifact root must be a real directory")
    try:
        row, _raw = _read_canonical_json(
            root / CUTE_CUBIN_INDEX_RELPATH,
            maximum=_MAX_INDEX_BYTES,
        )
    except Exception as exc:
        raise CuteCubinError(f"CuTe CUBIN index cannot reopen: {exc}") from None
    if set(row) != _INDEX_FIELDS or row.get("schema") != CUTE_CUBIN_SCHEMA:
        raise CuteCubinError("CuTe CUBIN index fields or schema mismatch")
    if row.get("provider") != CUTE_CUBIN_PROVIDER_NAME:
        raise CuteCubinError("CuTe CUBIN index provider mismatch")
    if row.get("binding_abi") != CUTE_CUBIN_BINDING_ABI:
        raise CuteCubinError("CuTe CUBIN binding ABI mismatch")
    if row.get("patcher_id") != CUTE_CUBIN_PATCHER:
        raise CuteCubinError("CuTe CUBIN patcher ID mismatch")

    build_spec = _digest(row["build_spec_digest"], field="CuTe CUBIN build spec digest")
    tree_digest = _digest(row["tree_digest"], field="CuTe CUBIN tree digest")
    profile_digest = _digest(
        row["compile_profile_digest"], field="CuTe CUBIN compile profile digest"
    )
    patcher_sha256 = _digest(
        row["patcher_sha256"], field="CuTe CUBIN patcher sha256"
    )
    for actual, expected, field in (
        (build_spec, expected_build_spec_digest, "build spec digest"),
        (tree_digest, expected_tree_digest, "tree digest"),
        (profile_digest, expected_compile_profile_digest, "compile profile digest"),
        (patcher_sha256, expected_patcher_sha256, "patcher sha256"),
    ):
        if expected is not None and actual != _digest(expected, field=f"expected {field}"):
            raise CuteCubinError(f"CuTe CUBIN {field} mismatch")

    logical_architecture = row["logical_architecture"]
    compiler_architecture = row["compiler_architecture"]
    if (
        not isinstance(logical_architecture, str)
        or _LOGICAL_ARCH_RE.fullmatch(logical_architecture) is None
        or not isinstance(compiler_architecture, str)
        or _COMPILER_ARCH_RE.fullmatch(compiler_architecture) is None
        or compiler_architecture.removeprefix("sm_").rstrip("a")
        != logical_architecture.removeprefix("sm").rstrip("a")
    ):
        raise CuteCubinError("CuTe CUBIN architecture pair is invalid")
    if (
        expected_logical_architecture is not None
        and logical_architecture != expected_logical_architecture
    ):
        raise CuteCubinError("CuTe CUBIN logical architecture mismatch")

    distributions = row["distributions"]
    if (
        not isinstance(distributions, dict)
        or set(distributions) != _DISTRIBUTIONS
        or not all(
            isinstance(key, str)
            and isinstance(value, str)
            and bool(value)
            and len(value) <= 128
            and value.isascii()
            for key, value in distributions.items()
        )
    ):
        raise CuteCubinError("CuTe CUBIN distribution inventory is malformed")
    if verify_distributions and distributions != installed_cute_distributions():
        raise CuteCubinError("CuTe CUBIN distributions differ from this worker")
    options = _canonical_options(
        row["compile_options"], compiler_architecture=compiler_architecture
    )

    raw_exports = row["exports"]
    if (
        not isinstance(raw_exports, list)
        or not raw_exports
        or len(raw_exports) > _MAX_EXPORTS
    ):
        raise CuteCubinError("CuTe CUBIN exports must be a nonempty bounded list")
    from optima.manifest import ManifestError, reopen_artifact_target_authority

    allowed_paths = {CUTE_CUBIN_INDEX_RELPATH}
    identities: set[tuple[str, str, str, str]] = set()
    exports: list[CuteCubinExportArtifact] = []
    for raw_export in raw_exports:
        if not isinstance(raw_export, dict) or set(raw_export) != _EXPORT_FIELDS:
            raise CuteCubinError("CuTe CUBIN export fields mismatch")
        source = _safe_relative(raw_export["source"], field="CuTe CUBIN source")
        slot = _simple(raw_export["slot"], field="CuTe CUBIN slot")
        variant = _simple(raw_export["variant"], field="CuTe CUBIN variant")
        name = _simple(raw_export["name"], field="CuTe CUBIN export name")
        factory = _simple(
            raw_export["factory"], field="CuTe CUBIN factory", identifier=True
        )
        artifact_id = _simple(
            raw_export["artifact_id"], field="CuTe CUBIN artifact ID", identifier=True
        )
        try:
            target_authority = reopen_artifact_target_authority(
                raw_export["artifact_target_authority"],
                expected_dispatch_slot=slot,
            )
        except ManifestError as exc:
            raise CuteCubinError(f"CuTe CUBIN target authority is invalid: {exc}") from None
        target_sha256 = _digest(
            raw_export["artifact_target_authority_sha256"],
            field="CuTe CUBIN target authority sha256",
        )
        if target_sha256 != target_authority.digest:
            raise CuteCubinError("CuTe CUBIN target authority digest mismatch")
        try:
            resource_plan, _resource_data, resource_sha256 = (
                reopen_artifact_resource_plan_identity(
                    raw_export["artifact_resource_plan"],
                    authority=target_authority,
                    expected_sha256=raw_export["artifact_resource_plan_sha256"],
                )
            )
        except Exception as exc:
            raise CuteCubinError(f"CuTe CUBIN resource plan is invalid: {exc}") from None
        launch_plan = _reopen_launch_plan(
            raw_export["launch_plan"],
            target_authority=target_authority,
            resource_plan=resource_plan,
        )
        launch_sha256 = _digest(
            raw_export["launch_plan_sha256"], field="CuTe CUBIN launch plan sha256"
        )
        if hashlib.sha256(canonical_json_bytes(launch_plan)).hexdigest() != launch_sha256:
            raise CuteCubinError("CuTe CUBIN launch plan digest mismatch")
        try:
            device_plan = DeviceLaunchPlan.from_dict(raw_export["device_plan"])
        except DeviceLaunchError as exc:
            raise CuteCubinError(
                f"CuTe CUBIN device launch plan is invalid: {exc}"
            ) from None
        device_plan_sha256 = _digest(
            raw_export["device_plan_sha256"],
            field="CuTe CUBIN device launch plan sha256",
        )
        if (
            device_plan.digest != device_plan_sha256
            or launch_plan.get("device_plan") != device_plan.to_dict()
        ):
            raise CuteCubinError(
                "CuTe CUBIN device launch plan differs from launch authority"
            )
        expected_artifact_id, _prefix = deterministic_export_names(
            source=source,
            slot=slot,
            variant=variant,
            name=name,
            factory=factory,
            plan=launch_plan,
            artifact_resource_plan_sha256=resource_sha256,
            artifact_target_authority_sha256=target_sha256,
        )
        if artifact_id != expected_artifact_id:
            raise CuteCubinError("CuTe CUBIN artifact ID differs from its declaration")

        raw_profile_inputs = raw_export["profile_inputs"]
        if (
            not isinstance(raw_profile_inputs, list)
            or raw_profile_inputs != sorted(set(raw_profile_inputs))
            or any(
                not isinstance(key, str)
                or _PROFILE_KEY_RE.fullmatch(key) is None
                or key not in CUTE_CUBIN_PROVIDER.compile_profile_inputs
                for key in raw_profile_inputs
            )
        ):
            raise CuteCubinError("CuTe CUBIN profile inputs are invalid")
        raw_resolved = raw_export["resolved_profile"]
        if (
            not isinstance(raw_resolved, dict)
            or set(raw_resolved) != set(raw_profile_inputs)
            or any(type(value) is not int or value <= 0 for value in raw_resolved.values())
        ):
            raise CuteCubinError("CuTe CUBIN resolved profile is invalid")
        cubin = _file_row(
            root,
            raw_export["cubin"],
            artifact_id=artifact_id,
            allowed_paths=allowed_paths,
        )
        try:
            raw_cubin = _stable_file_bytes(
                root.joinpath(*PurePosixPath(cubin.path).parts),
                maximum=_MAX_CUBIN_BYTES,
            )
            _require_elf_cubin(raw_cubin)
        except (CudaCubinError, OSError) as exc:
            raise CuteCubinError(f"CuTe CUBIN ELF gate failed: {exc}") from None

        identity = (slot, variant, str(launch_plan["plan"]), name)
        if identity in identities:
            raise CuteCubinError("CuTe CUBIN export identity is duplicated")
        identities.add(identity)
        exports.append(
            CuteCubinExportArtifact(
                source=source,
                slot=slot,
                variant=variant,
                name=name,
                factory=factory,
                artifact_id=artifact_id,
                launch_plan=MappingProxyType(launch_plan),
                launch_plan_sha256=launch_sha256,
                device_plan=device_plan,
                device_plan_sha256=device_plan_sha256,
                profile_inputs=tuple(raw_profile_inputs),
                resolved_profile=tuple(sorted(raw_resolved.items())),
                artifact_target_authority=target_authority,
                artifact_target_authority_sha256=target_sha256,
                artifact_resource_plan=resource_plan,
                artifact_resource_plan_sha256=resource_sha256,
                cubin=cubin,
            )
        )
    _validate_reopened_pipelines(exports)
    _publication_inventory(root, allowed_paths=allowed_paths)
    return CuteCubinIndex(
        root=root,
        build_spec_digest=build_spec,
        tree_digest=tree_digest,
        compile_profile_digest=profile_digest,
        logical_architecture=logical_architecture,
        compiler_architecture=compiler_architecture,
        patcher_sha256=patcher_sha256,
        distributions=tuple(sorted(distributions.items())),
        compile_options=options,
        exports=tuple(exports),
    )


def reopen_cute_cubin_index(
    artifact_root: str | Path,
    *,
    expected_build_spec_digest: str | None = None,
    expected_tree_digest: str | None = None,
    expected_logical_architecture: str | None = None,
    expected_compile_profile_digest: str | None = None,
    expected_patcher_sha256: str | None = None,
    verify_distributions: bool = True,
) -> CuteCubinIndex:
    """Normalize shared AOT helper failures to the device-provider boundary."""

    try:
        return _reopen_cute_cubin_index(
            artifact_root,
            expected_build_spec_digest=expected_build_spec_digest,
            expected_tree_digest=expected_tree_digest,
            expected_logical_architecture=expected_logical_architecture,
            expected_compile_profile_digest=expected_compile_profile_digest,
            expected_patcher_sha256=expected_patcher_sha256,
            verify_distributions=verify_distributions,
        )
    except CuteCubinError:
        raise
    except CuteAOTError as exc:
        raise CuteCubinError(str(exc)) from None


def _runtime_step(export: CuteCubinExportArtifact, executor: object) -> object:
    """Reconstruct one generic runtime step from the sealed launch authority."""

    from optima.artifact_abi import (
        ArtifactABIError,
        parse_artifact_bindings,
        parse_artifact_prelaunch,
        parse_provider_capability_requirements,
        parse_specialization_capability_requirements,
    )
    from optima.artifact_runtime import ArtifactRuntimeStep

    launch = export.launch_plan
    try:
        bindings = parse_artifact_bindings(
            launch["bindings"], field="runtime CuTe CUBIN bindings"
        )
        prelaunch = parse_artifact_prelaunch(
            launch["prelaunch"], field="runtime CuTe CUBIN prelaunch"
        )
        provider_requirements = parse_provider_capability_requirements(
            launch["provider_capability_requirements"],
            field="runtime CuTe CUBIN provider requirements",
        )
        specialization_requirements = (
            parse_specialization_capability_requirements(
                launch["specialization_capability_requirements"],
                field="runtime CuTe CUBIN specialization requirements",
            )
        )
        call_abi = export.artifact_target_authority.call_abi
        specializes = call_abi.validate_plan(
            role=launch["role"],
            bindings=bindings,
            specializes=launch["specializes"],
            prelaunch=prelaunch,
            require_outputs=False,
            artifact_resources=export.artifact_resource_plan,
        )
        current_provider = call_abi.provider_capability_requirements(
            bindings,
            artifact_resources=export.artifact_resource_plan,
        )
        current_specialization = call_abi.specialization_capability_requirements(
            specializes,
            artifact_resources=export.artifact_resource_plan,
        )
        if (
            provider_requirements != current_provider
            or specialization_requirements != current_specialization
        ):
            raise ArtifactABIError(
                "runtime CuTe CUBIN requirements differ from validator projection"
            )
        return ArtifactRuntimeStep(
            name=export.name,
            plan=str(launch["plan"]),
            step=int(launch["step"]),
            role=str(launch["role"]),
            bindings=bindings,
            specializes=specializes,
            prelaunch=prelaunch,
            executor=executor,
        )
    except (ArtifactABIError, KeyError, TypeError, ValueError) as exc:
        raise CuteCubinError(
            f"CuTe CUBIN runtime launch authority is invalid: {exc}"
        ) from None


def _require_live_provider_capabilities(
    exports: tuple[CuteCubinExportArtifact, ...],
    provider_capabilities: frozenset[str],
) -> None:
    from optima.artifact_abi import (
        ArtifactABIError,
        parse_provider_capability_requirements,
    )

    try:
        required = frozenset(
            requirement.capability
            for export in exports
            for requirement in parse_provider_capability_requirements(
                export.launch_plan["provider_capability_requirements"],
                field="runtime CuTe CUBIN provider requirements",
            )
        )
    except (ArtifactABIError, KeyError, TypeError, ValueError) as exc:
        raise CuteCubinError(
            f"CuTe CUBIN provider requirements are invalid: {exc}"
        ) from None
    missing = required - provider_capabilities
    if missing:
        raise CuteCubinError(
            "CuTe CUBIN runtime lacks reviewed provider capabilities "
            f"{sorted(missing)!r}"
        )


def prepare_cute_cubin_runtime(
    bundle: str | Path,
    artifact_root: str | Path,
    *,
    expected_publication_digest: str,
    expected_build_spec_digest: str,
    expected_tree_digest: str,
    expected_logical_architecture: str,
    expected_compile_profile_digest: str,
    expected_patcher_sha256: str,
    driver: object | None = None,
    group_capabilities: Mapping[
        str, Callable[[object, object, object | None], object]
    ]
    | None = None,
    group_handles: Mapping[str, Callable[[object], int]] | None = None,
) -> CuteCubinIndex:
    """Admit CUBINs and bind declarative entries inside the engine worker."""

    root = Path(artifact_root)
    try:
        owner_uid = root.lstat().st_uid
    except OSError as exc:
        raise CuteCubinError(
            f"CuTe CUBIN publication root is unavailable: {exc}"
        ) from None
    publication_digest = _digest(
        expected_publication_digest, field="native artifact publication digest"
    )
    build_spec = _digest(expected_build_spec_digest, field="native build spec digest")
    tree_digest = _digest(expected_tree_digest, field="engine tree digest")
    profile_digest = _digest(
        expected_compile_profile_digest,
        field="runtime CuTe compile profile digest",
    )
    patcher_sha256 = _digest(
        expected_patcher_sha256, field="runtime CuTe CUBIN patcher sha256"
    )
    architecture = _simple(
        expected_logical_architecture,
        field="runtime CuTe CUBIN logical architecture",
    )

    from optima.eval.native_artifact import reopen_native_artifact

    publication = reopen_native_artifact(
        root,
        expected_build_spec_digest=build_spec,
        expected_publication_digest=publication_digest,
        expected_owner_uid=owner_uid,
    )
    index = reopen_cute_cubin_index(
        publication.root,
        expected_build_spec_digest=build_spec,
        expected_tree_digest=tree_digest,
        expected_logical_architecture=architecture,
        expected_compile_profile_digest=profile_digest,
        expected_patcher_sha256=patcher_sha256,
    )

    from optima.manifest import load_manifest

    from optima.cute_aot import artifact_resource_plan_identity, export_launch_plan

    manifest = load_manifest(bundle)
    declaration_rows: list[tuple[object, ...]] = []
    for op in manifest.ops:
        for export in op.aot_exports:
            if export.provider != CUTE_CUBIN_PROVIDER_NAME:
                raise CuteCubinError(
                    "CuTe CUBIN runtime cannot mix artifact providers"
                )
            target = manifest.artifact_target_authority(op)
            _resources, resource_data, resource_digest = (
                artifact_resource_plan_identity(
                    authority=target, resources=op.artifact_resources
                )
            )
            declaration_rows.append(
                (
                    op.source,
                    op.slot,
                    op.variant,
                    export.name,
                    export.factory,
                    export_launch_plan(export),
                    target.snapshot(),
                    target.digest,
                    resource_data,
                    resource_digest,
                    tuple(export.profile_inputs),
                    export.device_plan.digest,
                )
            )
    declarations = tuple(
        sorted(
            declaration_rows,
            key=lambda row: (row[1], row[2], row[5]["plan"], row[5]["step"], row[3]),  # type: ignore[index]
        )
    )
    indexed = tuple(
        (
            export.source,
            export.slot,
            export.variant,
            export.name,
            export.factory,
            dict(export.launch_plan),
            export.artifact_target_authority.snapshot(),
            export.artifact_target_authority_sha256,
            export.artifact_resource_plan.to_dict(),
            export.artifact_resource_plan_sha256,
            export.profile_inputs,
            export.device_plan_sha256,
        )
        for export in index.exports
    )
    if indexed != declarations:
        raise CuteCubinError(
            "CuTe CUBIN publication differs from manifest declarations"
        )

    provider_groups = dict(group_capabilities or {})
    registry_groups = dict(group_handles or {})
    if set(provider_groups) != set(registry_groups):
        raise CuteCubinError(
            "CuTe CUBIN provider and parameter group authorities differ"
        )
    if driver is None:
        try:
            import cuda.bindings.driver as driver
        except Exception as exc:  # noqa: BLE001 - optional engine dependency
            raise CuteCubinError(f"CUDA driver binding is unavailable: {exc}") from None

    from optima.artifact_device_launch import (
        DeviceArtifactEntry,
        DeviceArtifactRuntime,
        DeviceLaunchError,
        make_device_artifact_runtime_provider,
    )
    from optima.artifact_runtime import (
        ArtifactAllocationBudget,
        ArtifactRuntimeEntry,
        ArtifactRuntimeError,
        ArtifactRuntimeLimits,
    )
    from optima.cuda_materialize import (
        CudaMaterializeError,
        make_cuda_primitive_registry,
    )

    try:
        registry = make_cuda_primitive_registry(
            driver=driver,
            group_handles=registry_groups,
        )
        provider = make_device_artifact_runtime_provider(
            driver=driver,
            group_capabilities=provider_groups,
        )
        import torch

        if not torch.cuda.is_available():
            raise CuteCubinError(
                "CuTe CUBIN runtime requires an available CUDA device"
            )
        device = int(torch.cuda.current_device())
    except (ArtifactRuntimeError, CudaMaterializeError, DeviceLaunchError) as exc:
        raise CuteCubinError(f"CuTe CUBIN runtime provider failed: {exc}") from None
    _require_live_provider_capabilities(index.exports, provider.provider_capabilities)
    resource_authority = hashlib.sha256(
        canonical_json_bytes(
            sorted({(row[1], row[2], row[7], row[9]) for row in declarations})
        )
    ).hexdigest()
    authority: tuple[object, ...] = (
        str(publication.root),
        publication_digest,
        build_spec,
        tree_digest,
        architecture,
        profile_digest,
        patcher_sha256,
        resource_authority,
        CUTE_CUBIN_ARTIFACT_MAX_LIVE_BYTES,
        CUTE_CUBIN_ARTIFACT_MAX_ALLOCATION_KEYS,
        owner_uid,
        device,
        driver,
        tuple(sorted(provider_groups.items())),
        tuple(sorted(registry_groups.items())),
    )

    global _RUNTIME_STATE
    with _RUNTIME_LOCK:
        if _RUNTIME_STATE is not None:
            if _RUNTIME_STATE.authority != authority:
                raise CuteCubinError(
                    "this engine process is already bound to another CuTe CUBIN authority"
                )
            return index

        limits = ArtifactRuntimeLimits(
            max_live_bytes=CUTE_CUBIN_ARTIFACT_MAX_LIVE_BYTES,
            max_allocation_keys=CUTE_CUBIN_ARTIFACT_MAX_ALLOCATION_KEYS,
        )
        budget = ArtifactAllocationBudget(limits)
        runtimes: list[object] = []
        bindings: dict[tuple[str, str], object] = {}
        keepalive: list[object] = [provider, registry, limits, budget]
        evidence_by_op: dict[tuple[str, str], dict[str, object]] = {}
        try:
            steps_by_op: dict[tuple[str, str], list[object]] = {}
            runtime_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
            runtimes_by_op: dict[tuple[str, str], list[object]] = {}
            for export in index.exports:
                raw = _stable_file_bytes(
                    index.root.joinpath(*PurePosixPath(export.cubin.path).parts),
                    maximum=_MAX_CUBIN_BYTES,
                )
                runtime = DeviceArtifactRuntime.admit(
                    raw,
                    export.device_plan,
                    registry,
                )
                runtimes.append(runtime)
                key = (export.slot, export.variant)
                runtimes_by_op.setdefault(key, []).append(runtime)
                steps_by_op.setdefault(key, []).append(
                    _runtime_step(export, runtime)
                )
                runtime_rows.setdefault(key, []).append(
                    {
                        "admission": runtime.admission.to_dict(),
                        "admission_sha256": runtime.admission.digest,
                        "device_plan_sha256": export.device_plan_sha256,
                        "name": export.name,
                        "plan": export.launch_plan["plan"],
                        "sha256": export.cubin.sha256,
                        "size": export.cubin.size,
                        "step": export.launch_plan["step"],
                    }
                )

            from optima.cute_aot import _entry_with_use_evidence

            for key, steps in sorted(steps_by_op.items()):
                op_exports = tuple(
                    export
                    for export in index.exports
                    if (export.slot, export.variant) == key
                )
                target_digests = {
                    export.artifact_target_authority_sha256
                    for export in op_exports
                }
                resource_plans = {
                    canonical_json_bytes(export.artifact_resource_plan.to_dict())
                    for export in op_exports
                }
                if len(target_digests) != 1 or len(resource_plans) != 1:
                    raise CuteCubinError(
                        f"CuTe CUBIN runtime op {key!r} has inconsistent authority"
                    )
                target = op_exports[0].artifact_target_authority
                resources = op_exports[0].artifact_resource_plan
                runtime_resources = resources if resources.resources else None
                base_entry = ArtifactRuntimeEntry(
                    call_abi=target.call_abi,
                    steps=tuple(steps),
                    provider=provider,
                    artifact_resources=runtime_resources,
                    limits=limits if runtime_resources is not None else None,
                    allocation_budget=budget if runtime_resources is not None else None,
                )
                entry = DeviceArtifactEntry(
                    base_entry,
                    tuple(runtimes_by_op[key]),
                )
                evidence = {
                    "artifact_abi": "optima.device-launch-plan.v1",
                    "artifact_resource_plan_sha256": (
                        op_exports[0].artifact_resource_plan_sha256
                    ),
                    "artifact_runtime_limits": {
                        "max_allocation_keys": CUTE_CUBIN_ARTIFACT_MAX_ALLOCATION_KEYS,
                        "max_live_bytes": CUTE_CUBIN_ARTIFACT_MAX_LIVE_BYTES,
                    },
                    "artifact_target_authority": target.snapshot(),
                    "artifact_target_authority_sha256": target.digest,
                    "build_spec_digest": build_spec,
                    "compile_profile_digest": profile_digest,
                    "cubins": sorted(runtime_rows[key], key=lambda row: (row["plan"], row["step"], row["name"])),
                    "plans": list(base_entry.plans),
                    "provider": CUTE_CUBIN_PROVIDER_NAME,
                    "provider_capabilities": sorted(provider.provider_capabilities),
                    "publication_digest": publication_digest,
                    "runtime_primitive_capabilities": sorted(registry.capabilities),
                    "slot": key[0],
                    "target_id": target.target_id,
                    "tree_digest": tree_digest,
                    "variant": key[1],
                }
                evidence_by_op[key] = evidence
                evidenced = _entry_with_use_evidence(entry, evidence)
                bindings[key] = evidenced
                keepalive.extend((base_entry, entry, evidenced))

            expected = {
                (op.slot, op.variant)
                for op in manifest.ops
                if op.aot_exports
            }
            if set(bindings) != expected:
                raise CuteCubinError(
                    "CuTe CUBIN runtime binding coverage differs from manifest"
                )
        except Exception as exc:  # noqa: BLE001 - cleanup before normalizing
            cleanup_failures: list[str] = []
            for entry in reversed(tuple(bindings.values())):
                close = getattr(entry, "close", None)
                if builtins.callable(close):
                    try:
                        close()
                    except Exception as cleanup:  # noqa: BLE001
                        cleanup_failures.append(str(cleanup))
            if not cleanup_failures:
                for runtime in reversed(runtimes):
                    try:
                        runtime.close()
                    except Exception as cleanup:  # noqa: BLE001
                        cleanup_failures.append(str(cleanup))
            if cleanup_failures:
                raise CuteCubinError(
                    "CuTe CUBIN preparation failed and cleanup is indeterminate; "
                    "the isolated worker must terminate: "
                    + "; ".join(cleanup_failures)
                ) from None
            if isinstance(exc, CuteCubinError):
                raise
            if isinstance(exc, (ArtifactRuntimeError, DeviceLaunchError)):
                raise CuteCubinError(
                    f"CuTe CUBIN declarative runtime failed: {exc}"
                ) from None
            raise

        keepalive.extend(runtimes)
        _RUNTIME_STATE = _CuteCubinRuntimeState(
            authority=authority,
            bindings=MappingProxyType(bindings),
            keepalive=tuple(keepalive),
        )
        from optima import receipts

        for key in sorted(bindings):
            receipts.write("aot_loaded", evidence_by_op[key], tag=f"{key[0]}.{key[1]}")
    return index


def resolve_cute_cubin_entry(op: object) -> object | None:
    """Return the prebound entry for one device-only CuTe manifest row."""

    exports = getattr(op, "aot_exports", ())
    if not exports:
        return None
    providers = {getattr(export, "provider", None) for export in exports}
    if providers != {CUTE_CUBIN_PROVIDER_NAME}:
        raise CuteCubinError(
            "direct artifact row does not belong to the CuTe CUBIN provider"
        )
    slot = _simple(getattr(op, "slot", None), field="CuTe CUBIN runtime slot")
    variant = _simple(
        getattr(op, "variant", None), field="CuTe CUBIN runtime variant"
    )
    with _RUNTIME_LOCK:
        if _RUNTIME_STATE is None:
            raise CuteCubinError(
                "CuTe CUBIN runtime was not prepared before candidate resolution"
            )
        try:
            return _RUNTIME_STATE.bindings[(slot, variant)]
        except KeyError:
            raise CuteCubinError(
                f"CuTe CUBIN runtime lacks direct binding {(slot, variant)!r}"
            ) from None


def shutdown_cute_cubin_runtime() -> None:
    """Fence lifecycle work and unload every admitted CUBIN at engine exit."""

    global _RUNTIME_STATE
    with _RUNTIME_LOCK:
        state = _RUNTIME_STATE
        if state is None:
            return
        failures: list[str] = []
        for key, entry in sorted(state.bindings.items()):
            close = getattr(entry, "close", None)
            if not builtins.callable(close):
                failures.append(f"{key!r}: entry is not closeable")
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - close independent entries
                failures.append(f"{key!r}: {exc}")
        if failures:
            raise CuteCubinError(
                "CuTe CUBIN runtime teardown failed: " + "; ".join(failures)
            )
        _RUNTIME_STATE = None


__all__ = [
    "CUTE_CUBIN_ARTIFACT_MAX_ALLOCATION_KEYS",
    "CUTE_CUBIN_ARTIFACT_MAX_LIVE_BYTES",
    "CUTE_CUBIN_BINDING_ABI",
    "CUTE_CUBIN_INDEX_NAME",
    "CUTE_CUBIN_INDEX_RELPATH",
    "CUTE_CUBIN_PATCHER",
    "CUTE_CUBIN_PROVIDER_NAME",
    "CUTE_CUBIN_SCHEMA",
    "CUTE_CUBIN_STAGE_DIRECTORY",
    "CuteCubinError",
    "CuteCubinExportArtifact",
    "CuteCubinFile",
    "CuteCubinIndex",
    "compile_options_snapshot",
    "prepare_cute_cubin_runtime",
    "reopen_cute_cubin_index",
    "resolve_cute_cubin_entry",
    "shutdown_cute_cubin_runtime",
]
