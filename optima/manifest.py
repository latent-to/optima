"""Bundle manifest parsing and validation.

A *bundle* is what a miner submits: a directory (or tarball) containing a
``manifest.toml`` plus kernel source and optional eligibility metadata. The
manifest is **data**, not code — it declares which slots the bundle claims to
implement and where the source lives. The validator reads it; the miner never
runs code at this stage.

This module is pure-Python (no torch/GPU) so the whole intake/validation path
runs anywhere.

Bundle layout::

    bundle/
      manifest.toml
      kernels/
        silu_and_mul.py        # exposes the slot's `entry` callable
        silu_and_mul.cu        # optional: declared via ops.cuda_sources (sanctioned
                                # inspectable CUDA source; compiled only by a
                                # validator-reviewed patcher, see optima/rebuild.py)
      metadata/
        silu_and_mul.json      # optional eligibility (dtypes, arch, max dims)
"""

from __future__ import annotations

import json
import keyword
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optima.artifact_provider import (
    ARTIFACT_PROVIDERS,
    ArtifactBindingABI,
)
from optima.artifact_abi import (
    ArtifactABIError,
    ArtifactBinding,
    ArtifactPrelaunch,
    ArtifactResource,
    ArtifactResourcePlan,
    ProviderCapabilityRequirement,
    SpecializationCapabilityRequirement,
    SlotCallABI,
    parse_artifact_bindings,
    parse_artifact_prelaunch,
    parse_artifact_resources,
    parse_provider_capability_requirements,
    parse_specialization_capability_requirements,
    slot_call_abi,
)
from optima.artifact_device_launch import DeviceLaunchError, DeviceLaunchPlan
from optima.stack_identity import canonical_digest

def _load_toml(p: Path) -> dict:
    """Parse a TOML file using whatever backend is available.

    Prefers stdlib ``tomllib`` (3.11+), then ``tomli`` (the de-facto 3.10
    backend, same binary-mode API), then the older ``toml`` package.
    """
    try:
        import tomllib  # type: ignore

        with p.open("rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        pass
    try:
        import tomli  # type: ignore

        with p.open("rb") as f:
            return tomli.load(f)
    except ModuleNotFoundError:
        pass
    import toml  # type: ignore

    return toml.loads(p.read_text())


ABI_VERSION = "optima-op-abi-v0"
_ID_RE = re.compile(r"^[0-9A-Za-z._\-]+$")
DEFAULT_VARIANT = "default"
ARTIFACT_TARGET_AUTHORITY_SCHEMA = "optima.artifact-target-authority.v1"
_AOT_EXPORT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_AOT_PROVIDER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_AOT_PLAN_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_AOT_PROFILE_COMPONENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_MAX_AOT_EXPORTS_PER_OP = 16
_MAX_AOT_PROFILE_INPUTS = 16
_MAX_AOT_FACTORY_LENGTH = 128
_MAX_AOT_PROFILE_INPUT_LENGTH = 128
_MAX_AOT_PLAN_STEP = 255
class ManifestError(ValueError):
    """Raised when a manifest is malformed or violates a structural rule."""


def _call_abi_snapshot(call_abi: SlotCallABI) -> dict[str, object]:
    return call_abi.identity_snapshot()


@dataclass(frozen=True)
class ArtifactTargetAuthority:
    """Validator-reconstructed target and call ABI for one static dispatch slot."""

    dispatch_slot: str
    call_abi: SlotCallABI

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dispatch_slot, str)
            or not self.dispatch_slot
            or len(self.dispatch_slot) > 128
            or _ID_RE.fullmatch(self.dispatch_slot) is None
        ):
            raise ManifestError("artifact target authority has an invalid dispatch slot")
        if not isinstance(self.call_abi, SlotCallABI):
            raise ManifestError("artifact target authority has an invalid call ABI")
        if self.call_abi.slot != self.dispatch_slot:
            raise ManifestError(
                "static artifact target and dispatch slot must be identical"
            )

    @property
    def target_id(self) -> str:
        return self.call_abi.slot

    @property
    def call_abi_digest(self) -> str:
        return canonical_digest("optima.artifact-call-abi", _call_abi_snapshot(self.call_abi))

    def snapshot(self) -> dict[str, object]:
        return {
            "call_abi": _call_abi_snapshot(self.call_abi),
            "call_abi_sha256": self.call_abi_digest,
            "dispatch_slot": self.dispatch_slot,
            "schema": ARTIFACT_TARGET_AUTHORITY_SCHEMA,
            "target_id": self.target_id,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.artifact-target-authority", self.snapshot())


def static_artifact_target_authority(dispatch_slot: str) -> ArtifactTargetAuthority:
    call_abi = slot_call_abi(dispatch_slot)
    if call_abi is None:
        raise ManifestError(
            f"slot {dispatch_slot!r} has no validator-owned artifact call ABI"
        )
    return ArtifactTargetAuthority(dispatch_slot=dispatch_slot, call_abi=call_abi)


def reopen_artifact_target_authority(
    value: object,
    *,
    expected_dispatch_slot: str | None = None,
) -> ArtifactTargetAuthority:
    """Rebuild a sealed target from validator tables and require exact equality.

    The serialized ABI is evidence, never authority.  In particular, a miner
    cannot provide a different ABI snapshot for the same event declaration.
    """

    expected_fields = {
        "call_abi",
        "call_abi_sha256",
        "dispatch_slot",
        "schema",
        "target_id",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("schema") != ARTIFACT_TARGET_AUTHORITY_SCHEMA
    ):
        raise ManifestError("sealed artifact target authority fields mismatch")
    dispatch_slot = value.get("dispatch_slot")
    if not isinstance(dispatch_slot, str):
        raise ManifestError("sealed artifact target dispatch slot is malformed")
    if expected_dispatch_slot is not None and dispatch_slot != expected_dispatch_slot:
        raise ManifestError("sealed artifact target dispatch slot mismatch")

    authority = static_artifact_target_authority(dispatch_slot)
    if value != authority.snapshot():
        raise ManifestError(
            "sealed artifact target authority differs from validator reconstruction"
        )
    return authority


@dataclass(frozen=True)
class AOTExport:
    """One validator-prebuilt artifact exported by an op source module.

    For the CuTe CUBIN provider, ``factory`` names a scanned Python symbol
    executed only by the no-egress validator compiler worker.
    ``profile_inputs`` names bounded, validator-owned values rather than
    miner-selected compile arguments. The
    ordered ``bindings`` project only resources exposed by the slot's immutable
    :class:`~optima.artifact_abi.SlotCallABI` plus declared validator-allocated
    ``artifact_resources``; there is no per-submission runtime Python adapter.
    """

    provider: str
    name: str
    factory: str
    profile_inputs: tuple[str, ...]
    bindings: tuple[ArtifactBinding, ...]
    specializes: tuple[tuple[str, bool | int | float | str], ...]
    prelaunch: tuple[ArtifactPrelaunch, ...]
    provider_capability_requirements: tuple[
        ProviderCapabilityRequirement, ...
    ] = ()
    specialization_capability_requirements: tuple[
        SpecializationCapabilityRequirement, ...
    ] = ()
    role: str = "run"
    plan: str = "default"
    step: int = 0
    # Closed, import-light launch/materialization declaration. It can never
    # carry a callback, launcher, or candidate host executable.
    device_plan: DeviceLaunchPlan | None = None

    @property
    def specialization_map(self) -> Mapping[str, bool | int | float | str]:
        return dict(self.specializes)


@dataclass(frozen=True)
class OpEntry:
    slot: str
    source: str  # bundle-relative path to the kernel module
    entry: str  # callable name inside the module
    dtypes: tuple[str, ...]
    architectures: tuple[str, ...]
    metadata: str | None
    variant: str = DEFAULT_VARIANT
    prepare: str | None = None  # optional 2nd callable (weight-prep) for (prepare, forward) slots
    setup: str | None = None  # optional callable run ONCE at engine init (framework mode)
    # Override-point submission (the swigluoai class): the bundle does NOT ship a whole kernel —
    # it fills a typed hole in a validator-owned base kernel from optima_kernels. ``entry`` then
    # names the override device fn (e.g. a CuTe-DSL epilogue), ``base_kernel`` names the base
    # (e.g. "nvfp4_moe_megakernel"), ``override_point`` the hole (e.g. "gemm1_epilogue"). The
    # validator JIT-composes base+override at load (see optima_kernels.override). ``prepare`` is
    # omitted: the validator owns the weight-prep for the base kernel.
    base_kernel: str | None = None
    override_point: str | None = None
    # Sanctioned "CUDA source" tier: bundle-relative paths to inspectable ``.cu``/``.cuh``
    # sources declared for this op. Compiled only by a validator-reviewed patcher
    # (rebuild.json — a different track); this module only validates the declaration is
    # well-formed and safe to point at. Declaring a path here is what lets scan_tree (see
    # optima/sandbox.py) treat the file as sanctioned instead of an unscanned stray binary.
    cuda_sources: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)
    # Validator-owned CuTe AOT prebuild declarations.  Appended after historical
    # fields so positional construction (if any downstream code uses it) retains
    # its prior meaning.
    aot_exports: tuple[AOTExport, ...] = ()
    # Miner-named storage is still validator allocated.  Rows are a closed,
    # bounded data contract shared by every artifact provider; no callback or
    # per-slot adapter is carried in the manifest.
    artifact_resources: tuple[ArtifactResource, ...] = ()

    @property
    def is_override(self) -> bool:
        return self.override_point is not None


@dataclass(frozen=True)
class DepPatchEntry:
    """One bundle-declared dependency patch (the ``dep_patches`` tier).

    ``target`` names a PINNED dependency (e.g. "flashinfer") — whether that target is
    patchable at all, and WHERE inside it a patch may land, is arena policy enforced by
    the one reviewed applier (optima/patchers/apply_dep_patch.py), never bundle content.
    ``path`` is a bundle-relative TEXT unified diff, structurally validated at load
    (optima/deppatch.py): modifications + new files only, no binary/rename/delete.
    """

    target: str
    path: str


@dataclass(frozen=True)
class CompetitionEntry:
    """A bundle's requested validator-owned contribution target.

    This table is syntax only.  The miner may request an identifier and assert
    whether it expects a singleton slot or a registered atomic target, but it
    cannot declare members, overlap, displacement, or allowed features.  Those
    are resolved independently by :mod:`optima.target_catalog`.
    """

    target: str
    mode: str


@dataclass(frozen=True)
class Manifest:
    bundle_id: str
    abi_version: str
    ops: tuple[OpEntry, ...]
    dep_patches: tuple[DepPatchEntry, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)
    # Appended after every historical field so positional construction retains
    # the old meaning of arguments four and five.
    competition: CompetitionEntry | None = None

    def ops_for(self, slot: str) -> tuple[OpEntry, ...]:
        """Return every implementation row for one semantic slot."""
        return tuple(op for op in self.ops if op.slot == slot)

    def op_for(self, slot: str, variant: str | None = None) -> OpEntry | None:
        """Return one row without allowing manifest order to select a variant."""
        matches = self.ops_for(slot)
        if variant is not None:
            return next((op for op in matches if op.variant == variant), None)
        if len(matches) > 1:
            raise ManifestError(
                f"slot {slot!r} has multiple variants; specify one of "
                f"{tuple(op.variant for op in matches)!r}"
            )
        return matches[0] if matches else None

    def artifact_target_authority(self, op: OpEntry) -> ArtifactTargetAuthority:
        """Return the validator-owned target boundary for an artifact op row."""

        if not isinstance(op, OpEntry):
            raise ManifestError("artifact target authority requires an OpEntry")
        return static_artifact_target_authority(op.slot)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ManifestError(msg)


def _safe_relpath(root: Path, rel: str, *, kind: str) -> Path:
    """Resolve ``rel`` under ``root`` and refuse to escape the bundle.

    Rejects absolute paths, ``..`` traversal, and symlinks that resolve outside
    ``root``. Returns the resolved path (which must exist).
    """
    _require(isinstance(rel, str) and rel != "", f"{kind} path must be a non-empty string")
    _require(not rel.startswith("/"), f"{kind} path must be relative: {rel!r}")
    p = (root / rel).resolve()
    root_resolved = root.resolve()
    _require(
        p == root_resolved or root_resolved in p.parents,
        f"{kind} path escapes bundle root: {rel!r}",
    )
    _require(p.exists(), f"{kind} not found: {rel!r}")
    _require(p.is_file(), f"{kind} must be a file: {rel!r}")
    return p


_CUDA_SOURCE_SUFFIXES = (".cu", ".cuh")


def _validate_cuda_source(root: Path, rel: str, *, slot: str) -> Path:
    """Validate one declared ``cuda_sources`` entry: exists, resolves inside the bundle,
    is a regular non-symlink file, and has a ``.cu``/``.cuh`` suffix.

    Mirrors ``_safe_relpath``'s containment check but additionally refuses symlinks
    (a bundle-relative symlink could point a "reviewed" .cu path at file contents
    outside the reviewed tree even while resolving inside the bundle boundary) and
    restricts the suffix so this declaration can't be used to sneak an arbitrary file
    past scan_tree's binary-suffix rejection under the "sanctioned CUDA source" cover.
    """
    kind = "cuda_sources"
    _require(isinstance(rel, str) and rel != "", f"ops ({slot}) {kind} path must be a non-empty string")
    _require(not rel.startswith("/"), f"ops ({slot}) {kind} path must be relative: {rel!r}")
    unresolved = root / rel
    _require(not unresolved.is_symlink(), f"ops ({slot}) {kind} must not be a symlink: {rel!r}")
    p = unresolved.resolve()
    root_resolved = root.resolve()
    _require(
        p == root_resolved or root_resolved in p.parents,
        f"ops ({slot}) {kind} path escapes bundle root: {rel!r}",
    )
    _require(p.exists(), f"ops ({slot}) {kind} not found: {rel!r}")
    _require(not p.is_symlink(), f"ops ({slot}) {kind} must not be a symlink: {rel!r}")
    _require(p.is_file(), f"ops ({slot}) {kind} must be a regular file: {rel!r}")
    _require(
        p.suffix in _CUDA_SOURCE_SUFFIXES,
        f"ops ({slot}) {kind} must be .cu or .cuh: {rel!r}",
    )
    return p


_DEP_PATCH_SUFFIXES = (".patch", ".diff")


def _validate_dep_patch(root: Path, rel: str, *, target: str) -> Path:
    """Validate one declared ``dep_patches`` entry: same containment/symlink posture as
    ``_validate_cuda_source`` (suffix-restricted so the declaration can't sanction an
    arbitrary file), PLUS a structural parse of the diff itself — a bundle carrying a
    binary/rename/delete "patch" fails at intake, not at apply time."""
    kind = "dep_patches"
    _require(isinstance(rel, str) and rel != "", f"{kind} ({target}) path must be a non-empty string")
    _require(not rel.startswith("/"), f"{kind} ({target}) path must be relative: {rel!r}")
    unresolved = root / rel
    _require(not unresolved.is_symlink(), f"{kind} ({target}) must not be a symlink: {rel!r}")
    p = unresolved.resolve()
    root_resolved = root.resolve()
    _require(
        p == root_resolved or root_resolved in p.parents,
        f"{kind} ({target}) path escapes bundle root: {rel!r}",
    )
    _require(p.exists(), f"{kind} ({target}) not found: {rel!r}")
    _require(not p.is_symlink(), f"{kind} ({target}) must not be a symlink: {rel!r}")
    _require(p.is_file(), f"{kind} ({target}) must be a regular file: {rel!r}")
    _require(
        p.suffix in _DEP_PATCH_SUFFIXES,
        f"{kind} ({target}) must be .patch or .diff: {rel!r}",
    )
    from optima.deppatch import DepPatchError, parse_patch_text

    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{kind} ({target}) {rel!r} is not UTF-8 text: {exc}") from exc
    try:
        parse_patch_text(text)
    except DepPatchError as exc:
        raise ManifestError(f"{kind} ({target}) {rel!r} rejected: {exc}") from exc
    return p


def _validate_aot_profile_input(
    value: object, *, field: str, allowed: frozenset[str]
) -> str:
    _require(
        isinstance(value, str) and value == value.strip() and bool(value),
        f"{field} must be a non-empty canonical string",
    )
    _require(
        len(value) <= _MAX_AOT_PROFILE_INPUT_LENGTH,
        f"{field} exceeds {_MAX_AOT_PROFILE_INPUT_LENGTH} characters",
    )
    components = value.split(".")
    _require(
        all(_AOT_PROFILE_COMPONENT_RE.fullmatch(component) for component in components),
        f"{field} must contain dot-separated lowercase identifiers: {value!r}",
    )
    _require(
        value in allowed,
        f"{field} is not registered for the selected artifact provider: {value!r}",
    )
    return value


def _validate_unsealed_aot_profile_input(value: object, *, field: str) -> str:
    """Retain bounded future-provider descriptor data without authorizing it."""

    _require(
        isinstance(value, str) and value == value.strip() and bool(value),
        f"{field} must be a non-empty canonical string",
    )
    _require(
        len(value) <= _MAX_AOT_PROFILE_INPUT_LENGTH,
        f"{field} exceeds {_MAX_AOT_PROFILE_INPUT_LENGTH} characters",
    )
    components = value.split(".")
    _require(
        all(_AOT_PROFILE_COMPONENT_RE.fullmatch(component) for component in components),
        f"{field} must contain dot-separated lowercase identifiers: {value!r}",
    )
    return value


def _parse_aot_exports(
    raw: object,
    *,
    op_index: int,
    slot: str,
    call_abi: SlotCallABI | None,
    artifact_resources: ArtifactResourcePlan | None = None,
    allow_unsealed_provider: bool = False,
) -> tuple[AOTExport, ...]:
    field = f"ops[{op_index}] ({slot}) 'aot_exports'"
    _require(isinstance(raw, (list, tuple)), f"{field} must be a list of tables")
    _require(
        len(raw) <= _MAX_AOT_EXPORTS_PER_OP,
        f"{field} may contain at most {_MAX_AOT_EXPORTS_PER_OP} entries",
    )
    if not raw:
        return ()
    exports: list[AOTExport] = []
    seen_names: set[str] = set()
    required_keys = {
        "provider",
        "name",
        "factory",
        "profile_inputs",
        "bindings",
    }
    optional_keys = {
        "device_plan",
        "plan",
        "prelaunch",
        "provider_capability_requirements",
        "role",
        "specialization_capability_requirements",
        "specializes",
        "step",
    }
    _require(
        call_abi is not None,
        f"{field} targets a slot without a declarative artifact call ABI",
    )
    for export_index, row in enumerate(raw):
        row_field = f"{field}[{export_index}]"
        _require(isinstance(row, dict), f"{row_field} must be a table")
        missing = required_keys - set(row)
        unknown = set(row) - required_keys - optional_keys
        _require(
            not missing and not unknown,
            f"{row_field} fields mismatch; missing={sorted(missing)!r}, "
            f"unknown={sorted(unknown)!r}",
        )

        provider = row["provider"]
        _require(
            isinstance(provider, str)
            and _AOT_PROVIDER_RE.fullmatch(provider) is not None,
            (
                f"{row_field} provider must be a bounded canonical identifier"
            ),
        )
        provider_descriptor = ARTIFACT_PROVIDERS.get(provider)
        _require(
            provider_descriptor is not None or allow_unsealed_provider,
            f"{row_field} provider {provider!r} is not registered",
        )
        if provider_descriptor is not None:
            _require(
                provider_descriptor.supports_static_slots,
                f"{row_field} provider {provider!r} is not enabled for this target kind",
            )
        name = row["name"]
        _require(
            isinstance(name, str) and bool(_AOT_EXPORT_NAME_RE.fullmatch(name)),
            f"{row_field} name must match {_AOT_EXPORT_NAME_RE.pattern!r}",
        )
        _require(name not in seen_names, f"{row_field} duplicates AOT export name {name!r}")
        seen_names.add(name)

        factory = row["factory"]
        _require(
            isinstance(factory, str)
            and len(factory) <= _MAX_AOT_FACTORY_LENGTH
            and factory.isascii()
            and factory.isidentifier()
            and not keyword.iskeyword(factory),
            f"{row_field} factory/implementation must be an identifier of at most "
            f"{_MAX_AOT_FACTORY_LENGTH} characters",
        )

        raw_inputs = row["profile_inputs"]
        _require(
            isinstance(raw_inputs, (list, tuple)),
            f"{row_field} profile_inputs must be a list of strings",
        )
        _require(
            len(raw_inputs) <= _MAX_AOT_PROFILE_INPUTS,
            f"{row_field} profile_inputs may contain at most "
            f"{_MAX_AOT_PROFILE_INPUTS} entries",
        )
        if provider_descriptor is not None:
            profile_inputs = tuple(
                _validate_aot_profile_input(
                    value,
                    field=f"{row_field} profile_inputs[{input_index}]",
                    allowed=provider_descriptor.compile_profile_inputs,
                )
                for input_index, value in enumerate(raw_inputs)
            )
        else:
            profile_inputs = tuple(
                _validate_unsealed_aot_profile_input(
                    value, field=f"{row_field} profile_inputs[{input_index}]"
                )
                for input_index, value in enumerate(raw_inputs)
            )
        _require(
            len(set(profile_inputs)) == len(profile_inputs),
            f"{row_field} profile_inputs must be unique",
        )
        role = row.get("role", "run")
        _require(
            isinstance(role, str) and role == role.strip() and bool(role),
            f"{row_field} role must be a non-empty canonical string",
        )
        plan = row.get("plan", "default")
        _require(
            isinstance(plan, str) and _AOT_PLAN_RE.fullmatch(plan) is not None,
            f"{row_field} plan must match {_AOT_PLAN_RE.pattern!r}",
        )
        step = row.get("step", 0)
        _require(
            type(step) is int and 0 <= step <= _MAX_AOT_PLAN_STEP,
            f"{row_field} step must be an integer in [0, {_MAX_AOT_PLAN_STEP}]",
        )
        raw_specializes = row.get("specializes", {})
        _require(
            isinstance(raw_specializes, Mapping),
            f"{row_field} specializes must be a table",
        )
        try:
            bindings = parse_artifact_bindings(
                row["bindings"], field=f"{row_field} bindings"
            )
            prelaunch = parse_artifact_prelaunch(
                row.get("prelaunch", ()), field=f"{row_field} prelaunch"
            )
            specializes = call_abi.validate_plan(
                role=role,
                bindings=bindings,
                specializes=raw_specializes,
                prelaunch=prelaunch,
                require_outputs=False,
                artifact_resources=artifact_resources,
            )
            provider_capability_requirements = (
                call_abi.provider_capability_requirements(
                    bindings,
                    artifact_resources=artifact_resources,
                )
            )
            specialization_capability_requirements = (
                call_abi.specialization_capability_requirements(
                    specializes,
                    artifact_resources=artifact_resources,
                )
            )
            if provider_descriptor is not None:
                required_provider_capabilities = {
                    requirement.capability
                    for requirement in provider_capability_requirements
                }
                unsupported_provider_capabilities = (
                    required_provider_capabilities
                    - provider_descriptor.provider_capabilities
                )
                if unsupported_provider_capabilities:
                    raise ArtifactABIError(
                        f"provider {provider!r} does not support capabilities "
                        f"{tuple(sorted(unsupported_provider_capabilities))!r}"
                    )
            if "provider_capability_requirements" in row:
                sealed_provider_requirements = (
                    parse_provider_capability_requirements(
                        row["provider_capability_requirements"],
                        field=(
                            f"{row_field} provider_capability_requirements"
                        ),
                    )
                )
                if (
                    sealed_provider_requirements
                    != provider_capability_requirements
                ):
                    raise ArtifactABIError(
                        "sealed provider capability requirements differ from "
                        "validator reconstruction"
                    )
            if "specialization_capability_requirements" in row:
                sealed_specialization_requirements = (
                    parse_specialization_capability_requirements(
                        row["specialization_capability_requirements"],
                        field=(
                            f"{row_field} specialization_capability_requirements"
                        ),
                    )
                )
                if (
                    sealed_specialization_requirements
                    != specialization_capability_requirements
                ):
                    raise ArtifactABIError(
                        "sealed specialization capability requirements differ "
                        "from validator reconstruction"
                    )
        except ArtifactABIError as exc:
            raise ManifestError(f"{row_field} artifact ABI rejected: {exc}") from None
        raw_device_plan = row.get("device_plan")
        device_binding = (
            provider_descriptor is not None
            and provider_descriptor.binding_abi
            is ArtifactBindingABI.CUDA_DRIVER_PARAMS_V1
        )
        _require(
            not device_binding or raw_device_plan is not None,
            f"{row_field} device provider requires device_plan",
        )
        device_plan = None
        if raw_device_plan is not None:
            try:
                device_plan = DeviceLaunchPlan.from_dict(raw_device_plan)
                device_plan.validate_bindings(
                    bindings,
                    provider_capabilities=(
                        frozenset()
                        if provider_descriptor is None
                        else provider_descriptor.provider_capabilities
                    ),
                )
            except DeviceLaunchError as exc:
                raise ManifestError(
                    f"{row_field} device launch plan rejected: {exc}"
                ) from None
        exports.append(
            AOTExport(
                provider=provider,
                name=name,
                factory=factory,
                profile_inputs=tuple(sorted(profile_inputs)),
                bindings=bindings,
                specializes=specializes,
                prelaunch=prelaunch,
                provider_capability_requirements=provider_capability_requirements,
                specialization_capability_requirements=(
                    specialization_capability_requirements
                ),
                role=role,
                plan=plan,
                step=step,
                device_plan=device_plan,
            )
        )
    # Declaration order is not executable authority.  Canonicalize once before
    # every cross-export validation so no lifecycle/resource decision can depend
    # on TOML row order (the selected-payload projection uses the same order).
    exports.sort(
        key=lambda item: (
            item.provider,
            item.plan,
            item.step,
            item.name,
        )
    )
    plans: dict[str, list[AOTExport]] = {}
    providers = {export.provider for export in exports}
    _require(
        len(providers) == 1,
        f"{field} may not mix artifact providers within one op",
    )
    seen_steps: set[tuple[str, int]] = set()
    for export in exports:
        key = (export.plan, export.step)
        _require(
            key not in seen_steps,
            f"{field} repeats artifact plan/step {key!r}",
        )
        seen_steps.add(key)
        plans.setdefault(export.plan, []).append(export)
    for plan, plan_exports in plans.items():
        specializations = {export.specializes for export in plan_exports}
        _require(
            len(specializations) == 1,
            f"{field} plan {plan!r} has inconsistent specializations",
        )
        try:
            call_abi.validate_pipeline(
                tuple(
                    (export.role, export.bindings, export.prelaunch)
                    for export in sorted(plan_exports, key=lambda item: item.step)
                ),
                artifact_resources=artifact_resources,
            )
            if artifact_resources is not None:
                artifact_resources.validate_pipeline(
                    tuple(
                        (export.role, export.bindings, export.prelaunch)
                        for export in sorted(
                            plan_exports, key=lambda item: item.step
                        )
                    ),
                    require_all=True,
                )
        except ArtifactABIError as exc:
            raise ManifestError(
                f"{field} plan {plan!r} artifact ABI rejected: {exc}"
            ) from None

    if artifact_resources is not None:
        try:
            artifact_resources.validate_pipeline(
                tuple(
                    (export.role, export.bindings, export.prelaunch)
                    for export in exports
                ),
                require_all=True,
            )
        except ArtifactABIError as exc:
            raise ManifestError(f"{field} artifact resources rejected: {exc}") from None

    # Runtime selects the unique most-specific equality predicate.  Two plans
    # may overlap only as a strict fallback chain (for example {} then
    # {block_size=128}); incomparable predicates would make selection depend on
    # ordering, and identical predicates would be duplicate defaults.
    plan_predicates = {
        plan: dict(plan_exports[0].specializes)
        for plan, plan_exports in plans.items()
    }

    def exact_equal(left: object, right: object) -> bool:
        return type(left) is type(right) and left == right

    def strict_subset(
        left: Mapping[str, object], right: Mapping[str, object]
    ) -> bool:
        return len(left) < len(right) and all(
            key in right and exact_equal(value, right[key])
            for key, value in left.items()
        )

    plan_names = sorted(plan_predicates)
    for left_index, left_name in enumerate(plan_names):
        left = plan_predicates[left_name]
        for right_name in plan_names[left_index + 1 :]:
            right = plan_predicates[right_name]
            disjoint = any(
                key in right and not exact_equal(value, right[key])
                for key, value in left.items()
            )
            if disjoint:
                continue
            if strict_subset(left, right) or strict_subset(right, left):
                continue
            raise ManifestError(
                f"{field} plans {left_name!r} and {right_name!r} have ambiguous "
                "overlapping specialization predicates"
            )
    return tuple(exports)


def _metadata_exact_capabilities(
    root: Path,
    relative: str | None,
    *,
    op_index: int,
    slot: str,
) -> Mapping[str, object]:
    """Return only capability predicates which prove one exact value.

    A compile-time specialization may not rely on a range, a multi-valued
    predicate, or descriptive metadata. The runtime selector must independently
    prove the exact fact before a specialized artifact can be selected.
    """

    if relative is None:
        return {}
    path = _safe_relpath(root, relative, kind="metadata")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ManifestError(
            f"ops[{op_index}] ({slot}) cannot read metadata: {exc}"
        ) from None
    _require(
        len(raw) <= 1 << 20,
        f"ops[{op_index}] ({slot}) metadata exceeds 1 MiB",
    )

    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(
                    f"ops[{op_index}] ({slot}) metadata repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object)
    except ManifestError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(
            f"ops[{op_index}] ({slot}) metadata is invalid JSON: {exc}"
        ) from None
    _require(
        isinstance(value, dict),
        f"ops[{op_index}] ({slot}) metadata must be an object",
    )
    capabilities = value.get("capabilities", {})
    _require(
        isinstance(capabilities, dict),
        f"ops[{op_index}] ({slot}) metadata capabilities must be an object",
    )
    exact: dict[str, object] = {}
    for field, predicate in capabilities.items():
        if not isinstance(field, str):
            continue
        if isinstance(predicate, dict):
            if set(predicate) == {"exact"}:
                exact[field] = predicate["exact"]
            elif set(predicate) == {"one_of"} and isinstance(
                predicate["one_of"], list
            ) and len(predicate["one_of"]) == 1:
                exact[field] = predicate["one_of"][0]
        elif isinstance(predicate, list):
            if len(predicate) == 1:
                exact[field] = predicate[0]
        else:
            exact[field] = predicate
    return exact


def _validate_aot_specializations(
    exports: tuple[AOTExport, ...],
    *,
    root: Path,
    metadata: str | None,
    op_index: int,
    slot: str,
    call_abi: SlotCallABI | None,
) -> None:
    if call_abi is None or not exports:
        return
    exact = _metadata_exact_capabilities(
        root, metadata, op_index=op_index, slot=slot
    )
    for export in exports:
        required = call_abi.specialization_capabilities(export.specializes)
        for field, value in required.items():
            _require(
                field in exact
                and type(exact[field]) is type(value)
                and exact[field] == value,
                f"ops[{op_index}] ({slot}) AOT export {export.name!r} specializes "
                f"{field}={value!r} without an identical exact capability predicate",
            )


def load_manifest(bundle_root: str | Path) -> Manifest:
    """Load and structurally validate ``manifest.toml`` under ``bundle_root``.

    Validates schema and path-safety only. It does NOT import or execute any
    miner code (that is ``optima.sandbox``'s job) and it does NOT check the slot
    contract numerically (that is ``optima.verify``'s job).
    """
    root = Path(bundle_root).resolve()
    _require(root.is_dir(), f"bundle root is not a directory: {root}")
    manifest_path = root / "manifest.toml"
    _require(manifest_path.is_file(), f"manifest.toml not found in {root}")

    try:
        data = _load_toml(manifest_path)
    except Exception as exc:  # noqa: BLE001 - surface parse errors cleanly
        raise ManifestError(f"failed to parse manifest.toml: {exc}") from exc

    bundle_id = str(data.get("bundle_id", "")).strip()
    _require(bool(bundle_id), "manifest must set a non-empty bundle_id")
    _require(bool(_ID_RE.match(bundle_id)), f"bundle_id has illegal chars: {bundle_id!r}")

    abi = str(data.get("abi_version", "")).strip()
    _require(
        abi == ABI_VERSION,
        f"unsupported abi_version {abi!r}; this validator speaks {ABI_VERSION!r}",
    )

    ops_raw = data.get("ops")
    _require(isinstance(ops_raw, list) and ops_raw, "manifest must contain a non-empty [[ops]] list")

    ops: list[OpEntry] = []
    seen_variants: dict[str, dict[str, bool]] = {}
    seen_aot_exports: set[tuple[str, str, str]] = set()
    for i, op in enumerate(ops_raw):
        _require(isinstance(op, dict), f"ops[{i}] must be a table")
        slot = str(op.get("slot", "")).strip()
        _require(bool(slot), f"ops[{i}] missing 'slot'")
        variant_explicit = "variant" in op
        raw_variant = op.get("variant", DEFAULT_VARIANT)
        _require(
            isinstance(raw_variant, str),
            f"ops[{i}] ({slot}) 'variant' must be a string",
        )
        variant = raw_variant.strip()
        _require(
            bool(variant) and bool(_ID_RE.match(variant)),
            f"ops[{i}] ({slot}) 'variant' must be a simple identifier: {variant!r}",
        )
        prior = seen_variants.setdefault(slot, {})
        if prior:
            _require(
                variant_explicit and all(prior.values()),
                f"duplicate slot {slot!r} requires every row to declare an explicit "
                "unique 'variant' identifier",
            )
        _require(
            variant not in prior,
            f"duplicate variant {variant!r} for slot {slot!r}",
        )
        prior[variant] = variant_explicit

        source = str(op.get("source", "")).strip()
        entry = str(op.get("entry", "")).strip()
        _require(bool(source), f"ops[{i}] ({slot}) missing 'source'")
        _require(bool(entry) and entry.isidentifier(), f"ops[{i}] ({slot}) 'entry' must be a python identifier")

        prepare = op.get("prepare")
        if prepare is not None:
            prepare = str(prepare).strip()
            _require(prepare.isidentifier(), f"ops[{i}] ({slot}) 'prepare' must be a python identifier")

        setup = op.get("setup")
        if setup is not None:
            setup = str(setup).strip()
            _require(setup.isidentifier(), f"ops[{i}] ({slot}) 'setup' must be a python identifier")

        # Path-safety check now (existence + containment); content scanning later.
        _safe_relpath(root, source, kind="source")

        metadata = op.get("metadata")
        if metadata is not None:
            metadata = str(metadata).strip()
            _safe_relpath(root, metadata, kind="metadata")

        def _string_array(name: str) -> tuple[str, ...]:
            raw = op.get(name, ())
            _require(
                isinstance(raw, (list, tuple)),
                f"ops[{i}] ({slot}) {name!r} must be an array of strings",
            )
            _require(
                all(isinstance(value, str) and value.strip() for value in raw),
                f"ops[{i}] ({slot}) {name!r} must contain non-empty strings",
            )
            return tuple(value.strip() for value in raw)

        dtypes = _string_array("dtypes")
        archs = _string_array("architectures")

        # Override-point fields (optional). override_point requires base_kernel; the names
        # are resolved against optima_kernels at load (here we only check structure).
        base_kernel = op.get("base_kernel")
        if base_kernel is not None:
            base_kernel = str(base_kernel).strip()
            _require(bool(base_kernel), f"ops[{i}] ({slot}) 'base_kernel' must be non-empty when set")
        override_point = op.get("override_point")
        if override_point is not None:
            override_point = str(override_point).strip()
            _require(bool(override_point), f"ops[{i}] ({slot}) 'override_point' must be non-empty when set")
        _require(
            override_point is None or base_kernel is not None,
            f"ops[{i}] ({slot}) 'override_point' requires 'base_kernel'",
        )

        cuda_sources_raw = op.get("cuda_sources", ())
        _require(
            isinstance(cuda_sources_raw, (list, tuple)),
            f"ops[{i}] ({slot}) 'cuda_sources' must be a list of paths",
        )
        for cs in cuda_sources_raw:
            _validate_cuda_source(root, cs, slot=slot)
        cuda_sources = tuple(str(cs) for cs in cuda_sources_raw)

        call_abi = slot_call_abi(slot)

        raw_artifact_resources = op.get("artifact_resources", ())
        artifact_resource_plan: ArtifactResourcePlan | None = None
        if raw_artifact_resources:
            _require(
                call_abi is not None,
                f"ops[{i}] ({slot}) artifact_resources require a declarative "
                "artifact call ABI",
            )
            try:
                artifact_resource_plan = parse_artifact_resources(
                    raw_artifact_resources,
                    call_abi=call_abi,
                    field=f"ops[{i}] ({slot}) 'artifact_resources'",
                )
            except ArtifactABIError as exc:
                raise ManifestError(
                    f"ops[{i}] ({slot}) artifact resources rejected: {exc}"
                ) from None
        elif not isinstance(raw_artifact_resources, (list, tuple)):
            raise ManifestError(
                f"ops[{i}] ({slot}) 'artifact_resources' must be a list of tables"
            )

        aot_exports = _parse_aot_exports(
            op.get("aot_exports", ()),
            op_index=i,
            slot=slot,
            call_abi=call_abi,
            artifact_resources=artifact_resource_plan,
            allow_unsealed_provider=False,
        )
        _require(
            artifact_resource_plan is None or bool(aot_exports),
            f"ops[{i}] ({slot}) artifact_resources require AOT exports",
        )
        _validate_aot_specializations(
            aot_exports,
            root=root,
            metadata=metadata,
            op_index=i,
            slot=slot,
            call_abi=call_abi,
        )
        for export in aot_exports:
            export_key = (slot, variant, export.name)
            _require(
                export_key not in seen_aot_exports,
                f"duplicate AOT export for slot/variant/name {export_key!r}",
            )
            seen_aot_exports.add(export_key)
        if aot_exports:
            providers = {export.provider for export in aot_exports}
            _require(
                prepare is None
                and setup is None
                and base_kernel is None
                and override_point is None,
                f"ops[{i}] ({slot}) direct artifact rows may not declare prepare, "
                "setup, override, or base-kernel runtime paths",
            )
            _require(
                all(
                    (descriptor := ARTIFACT_PROVIDERS.get(provider)) is not None
                    and descriptor.supports_static_slots
                    for provider in providers
                ),
                f"ops[{i}] ({slot}) direct artifact rows require a registered "
                "static-slot provider",
            )
            _require(
                not cuda_sources,
                f"ops[{i}] ({slot}) direct artifact rows may not declare "
                "CUDA-source runtime paths",
            )

        known = {"slot", "variant", "source", "entry", "prepare", "setup", "dtypes", "architectures",
                 "metadata", "base_kernel", "override_point", "cuda_sources", "aot_exports",
                 "artifact_resources"}
        extra = {k: v for k, v in op.items() if k not in known}

        ops.append(
            OpEntry(
                slot=slot,
                source=source,
                entry=entry,
                dtypes=dtypes,
                architectures=archs,
                metadata=metadata,
                variant=variant,
                prepare=prepare,
                setup=setup,
                base_kernel=base_kernel,
                override_point=override_point,
                cuda_sources=cuda_sources,
                extra=extra,
                aot_exports=aot_exports,
                artifact_resources=(
                    ()
                    if artifact_resource_plan is None
                    else artifact_resource_plan.resources
                ),
            )
        )

    competition_raw = data.get("competition")
    competition: CompetitionEntry | None = None
    if competition_raw is not None:
        _require(
            isinstance(competition_raw, dict),
            "top-level 'competition' must be a {target, mode} table",
        )
        unknown = set(competition_raw) - {"target", "mode"}
        _require(not unknown, f"competition has unknown keys: {sorted(unknown)}")

        raw_target = competition_raw.get("target")
        _require(
            isinstance(raw_target, str),
            "competition 'target' must be a string",
        )
        target = raw_target.strip()
        _require(
            bool(target) and bool(_ID_RE.match(target)),
            f"competition 'target' must be a simple identifier: {target!r}",
        )

        raw_mode = competition_raw.get("mode")
        _require(
            isinstance(raw_mode, str),
            "competition 'mode' must be a string",
        )
        mode = raw_mode.strip()
        # Donor-era local bundles already carry ``mode = "system"``.  Preserve
        # syntax compatibility for inspection/migration, but TargetCatalog never
        # registers a system title.
        _require(
            mode in {"slot", "atomic", "system"},
            "competition 'mode' must be 'slot', 'atomic', or legacy 'system'",
        )
        competition = CompetitionEntry(target=target, mode=mode)

    dep_raw = data.get("dep_patches", ())
    _require(
        isinstance(dep_raw, (list, tuple)),
        "top-level 'dep_patches' must be a list of {target, path} tables",
    )
    dep_patches: list[DepPatchEntry] = []
    seen_dep: set[tuple[str, str]] = set()
    for i, dp in enumerate(dep_raw):
        _require(isinstance(dp, dict), f"dep_patches[{i}] must be a table")
        target = str(dp.get("target", "")).strip()
        _require(bool(target) and bool(_ID_RE.match(target)),
                 f"dep_patches[{i}] 'target' must be a simple identifier: {target!r}")
        rel = str(dp.get("path", "")).strip()
        _validate_dep_patch(root, rel, target=target)
        key = (target, rel)
        _require(key not in seen_dep, f"duplicate dep_patches entry: {key!r}")
        seen_dep.add(key)
        unknown = set(dp) - {"target", "path"}
        _require(not unknown, f"dep_patches[{i}] has unknown keys: {sorted(unknown)}")
        dep_patches.append(DepPatchEntry(target=target, path=rel))

    return Manifest(
        bundle_id=bundle_id,
        abi_version=abi,
        ops=tuple(ops),
        dep_patches=tuple(dep_patches),
        raw=data,
        competition=competition,
    )


def resolve_source(bundle_root: str | Path, op: OpEntry) -> Path:
    """Return the absolute, containment-checked path to an op's source file."""
    return _safe_relpath(Path(bundle_root), op.source, kind="source")


def resolve_cuda_sources(bundle_root: str | Path, op: OpEntry) -> tuple[Path, ...]:
    """Return the absolute, containment-checked paths to an op's declared ``cuda_sources``.

    Re-validates (cheap; these are small source files) rather than trusting the
    manifest was loaded from this exact ``bundle_root`` — same posture as
    ``resolve_source``.
    """
    root = Path(bundle_root)
    return tuple(_validate_cuda_source(root, cs, slot=op.slot) for cs in op.cuda_sources)


def all_declared_cuda_sources(bundle_root: str | Path, manifest: Manifest) -> frozenset[Path]:
    """Resolved, deduped set of every ``cuda_sources`` path declared across all ops.

    The shape ``optima.sandbox.scan_tree`` wants for its declared-allowlist parameter:
    a flat set of resolved paths, independent of which op declared them.
    """
    root = Path(bundle_root)
    out: set[Path] = set()
    for op in manifest.ops:
        out.update(resolve_cuda_sources(root, op))
    return frozenset(out)


def resolve_dep_patches(bundle_root: str | Path, manifest: Manifest) -> tuple[Path, ...]:
    """Absolute, containment-checked (re-validated) paths of all declared dep patches."""
    root = Path(bundle_root)
    return tuple(_validate_dep_patch(root, dp.path, target=dp.target)
                 for dp in manifest.dep_patches)


def all_declared_dep_patches(bundle_root: str | Path, manifest: Manifest) -> frozenset[Path]:
    """``scan_tree``-shaped allowlist of every declared dep patch (see cuda variant)."""
    return frozenset(resolve_dep_patches(bundle_root, manifest))
