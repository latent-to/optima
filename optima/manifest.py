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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


class ManifestError(ValueError):
    """Raised when a manifest is malformed or violates a structural rule."""


@dataclass(frozen=True)
class OpEntry:
    slot: str
    source: str  # bundle-relative path to the kernel module
    entry: str  # callable name inside the module
    dtypes: tuple[str, ...]
    architectures: tuple[str, ...]
    metadata: str | None
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
class Manifest:
    bundle_id: str
    abi_version: str
    ops: tuple[OpEntry, ...]
    dep_patches: tuple[DepPatchEntry, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    def op_for(self, slot: str) -> OpEntry | None:
        for op in self.ops:
            if op.slot == slot:
                return op
        return None


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
    seen_slots: set[str] = set()
    for i, op in enumerate(ops_raw):
        _require(isinstance(op, dict), f"ops[{i}] must be a table")
        slot = str(op.get("slot", "")).strip()
        _require(bool(slot), f"ops[{i}] missing 'slot'")
        _require(slot not in seen_slots, f"duplicate slot in manifest: {slot!r}")
        seen_slots.add(slot)

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

        dtypes = tuple(str(d) for d in op.get("dtypes", ()))
        archs = tuple(str(a) for a in op.get("architectures", ()))

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

        known = {"slot", "source", "entry", "prepare", "setup", "dtypes", "architectures",
                 "metadata", "base_kernel", "override_point", "cuda_sources"}
        extra = {k: v for k, v in op.items() if k not in known}

        ops.append(
            OpEntry(
                slot=slot,
                source=source,
                entry=entry,
                dtypes=dtypes,
                architectures=archs,
                metadata=metadata,
                prepare=prepare,
                setup=setup,
                base_kernel=base_kernel,
                override_point=override_point,
                cuda_sources=cuda_sources,
                extra=extra,
            )
        )

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

    return Manifest(bundle_id=bundle_id, abi_version=abi, ops=tuple(ops),
                    dep_patches=tuple(dep_patches), raw=data)


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
