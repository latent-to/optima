"""Validator-owned dependency-patch policy and immutable overlay validation.

Dependency patches are data.  A reviewed patcher applies them inside the hermetic
native-build container; engine ranks only reopen the published result.  Overlay
locations are derived from the complete native-build specification digest, never
from a bundle ID (materialized engine trees intentionally share one bundle ID).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


OVERLAY_SCHEMA = "optima.dep-overlay.v3"
_STAMP_NAME = "overlay.json"
_STAMP_KEYS = frozenset(
    {
        "schema",
        "build_spec_digest",
        "target",
        "policy",
        "patches",
        "subtree",
        "touched_files",
        "tree_digest",
        "tree_files",
        "prebuilt_modules",
    }
)


class DepOverlayError(RuntimeError):
    """A dependency overlay is absent, malformed, or not the exact publication."""


@dataclass(frozen=True)
class PrebuiltJITModule:
    """One validator-approved dependency generator compiled before engine launch."""

    name: str
    target_architecture: str
    cuda_arch_list: str
    generator_module: str
    generator_attr: str
    consumer_module: str
    consumer_attr: str


@dataclass(frozen=True)
class DepPolicy:
    # Importable package whose install anchors site-root-relative patch paths.
    package: str
    # Complete subtree copied into the build stage so relative includes keep working.
    overlay_subtree: str
    # Validator-controlled paths that a text patch may touch.
    allowed_globs: tuple[str, ...]
    # (module, attr) of the dependency's late-bound source-root constant.
    env_rebind: tuple[str, str] | None = None
    # Exact dependency generators compiled in build and replaced by load-only proxies.
    prebuilt_modules: tuple[PrebuiltJITModule, ...] = ()


PATCHABLE_DEPS: dict[str, DepPolicy] = {
    "flashinfer": DepPolicy(
        package="flashinfer",
        overlay_subtree="flashinfer/data/csrc",
        allowed_globs=("flashinfer/data/csrc/fused_moe/*",),
        env_rebind=("flashinfer.jit.env", "FLASHINFER_CSRC_DIR"),
        prebuilt_modules=(
            PrebuiltJITModule(
                name="fused_moe_103",
                target_architecture="sm103",
                cuda_arch_list="10.3",
                generator_module="flashinfer.jit.fused_moe",
                generator_attr="gen_cutlass_fused_moe_sm103_module",
                consumer_module="flashinfer.fused_moe.core",
                consumer_attr="gen_cutlass_fused_moe_sm103_module",
            ),
        ),
    ),
}


@dataclass(frozen=True)
class OverlayFile:
    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class ValidatedOverlay:
    target: str
    root: Path
    subtree: Path
    build_spec_digest: str
    tree_digest: str
    files: tuple[OverlayFile, ...]
    modules: tuple["ValidatedPrebuiltModule", ...]
    stamp: dict[str, Any]


@dataclass(frozen=True)
class ValidatedPrebuiltModule:
    policy: PrebuiltJITModule
    path: Path
    sha256: str
    size: int


def _digest(value: object, *, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise DepOverlayError(str(exc)) from None


def rebuild_phase() -> str:
    phase = os.environ.get("OPTIMA_REBUILD_PHASE", "all").strip().lower()
    if phase not in {"all", "build", "load"}:
        raise DepOverlayError(f"unsupported OPTIMA_REBUILD_PHASE: {phase!r}")
    return phase


def native_build_spec_digest(*, phase: str | None = None) -> str:
    """Return the controller-supplied whole-build identity.

    Production ``build`` and ``load`` phases require it.  ``all`` is an explicit
    development compatibility lane whose caller may derive a whole-input identity.
    """

    phase = phase or rebuild_phase()
    raw = os.environ.get("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", "").strip()
    if not raw:
        raise DepOverlayError(
            f"OPTIMA_NATIVE_BUILD_SPEC_DIGEST is required in rebuild phase {phase!r}"
        )
    return _digest(raw, field="OPTIMA_NATIVE_BUILD_SPEC_DIGEST")


def resolved_build_spec_digest(bundle: str | Path, *, phase: str | None = None) -> str:
    """Resolve production identity, or derive the explicit ``all``-lane identity."""

    phase = phase or rebuild_phase()
    if os.environ.get("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", "").strip():
        return native_build_spec_digest(phase=phase)
    if phase != "all":
        return native_build_spec_digest(phase=phase)  # raises with the phase in context
    from optima.manifest import load_manifest

    manifest = load_manifest(bundle)
    targets = [dp.target for dp in manifest.dep_patches]
    return derive_development_build_digest(bundle, targets)


def _absolute_directory_env(name: str) -> Path:
    raw = os.environ.get(name, "").strip()
    path = Path(raw)
    if not raw or not path.is_absolute():
        raise DepOverlayError(f"{name} must name an existing absolute directory")
    try:
        info = path.lstat()
    except OSError as exc:
        raise DepOverlayError(f"cannot inspect {name}: {exc}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise DepOverlayError(f"{name} must name a non-symlink directory")
    return path


def native_artifact_root(*, phase: str | None = None, build_spec_digest: str | None = None) -> Path:
    """Resolve the build-stage or published-artifact root for one phase."""

    phase = phase or rebuild_phase()
    if phase == "build":
        return _absolute_directory_env("OPTIMA_NATIVE_ARTIFACT_STAGE")
    if phase == "load":
        return _absolute_directory_env("OPTIMA_NATIVE_ARTIFACT_ROOT")

    digest = build_spec_digest or native_build_spec_digest(phase=phase)
    base_raw = os.environ.get("OPTIMA_DEP_OVERLAY_CACHE", "").strip()
    base = Path(base_raw) if base_raw else Path.home() / ".cache" / "optima" / "dep_overlay"
    if not base.is_absolute():
        raise DepOverlayError("OPTIMA_DEP_OVERLAY_CACHE must be absolute")
    # Development-only cache; still content-addressed by a whole-build digest.
    return base / "v3" / digest[:2] / digest


def overlay_base(build_spec_digest: str, *, artifact_root: Path | None = None) -> Path:
    """Compatibility path constructor, now keyed only by a full build digest."""

    digest = _digest(build_spec_digest, field="native build-spec digest")
    root = artifact_root or native_artifact_root(build_spec_digest=digest)
    return root / "dep_overlays"


def overlay_path(artifact_root: Path, target: str) -> Path:
    if target not in PATCHABLE_DEPS:
        raise DepOverlayError(f"dependency target is not patchable: {target!r}")
    return Path(artifact_root) / "dep_overlays" / target


def prebuilt_module_relative_path(target: str, module: PrebuiltJITModule) -> str:
    if target not in PATCHABLE_DEPS:
        raise DepOverlayError(f"dependency target is not patchable: {target!r}")
    for component in (target, module.name):
        if not component or not component.replace("_", "").isalnum():
            raise DepOverlayError(f"unsafe prebuilt dependency module name: {component!r}")
    return f"dep_modules/{target}/{module.name}/{module.name}.so"


def dependency_site_root(policy: DepPolicy) -> Path | None:
    """Locate a dependency without importing its Python package."""

    spec = importlib.util.find_spec(policy.package)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def policy_snapshot(policy: DepPolicy) -> dict[str, object]:
    # Round-trip normalizes tuples to JSON arrays for byte-exact stamp comparison.
    return json.loads(canonical_json_bytes(asdict(policy)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_relative(path: Path) -> str:
    value = path.as_posix()
    pure = PurePosixPath(value)
    if not value or value.startswith("/") or ".." in pure.parts or "\\" in value:
        raise DepOverlayError(f"non-canonical overlay path: {value!r}")
    try:
        value.encode("ascii", "strict")
    except UnicodeError:
        raise DepOverlayError(f"overlay path must be ASCII: {value!r}") from None
    return value


def tree_inventory(root: Path, *, require_read_only: bool = False) -> tuple[str, tuple[OverlayFile, ...]]:
    """Hash every regular file under ``root`` and reject unsafe filesystem shapes."""

    root = Path(root)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise DepOverlayError(f"overlay subtree is missing: {root}: {exc}") from None
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise DepOverlayError(f"overlay subtree is not a regular directory: {root}")
    if require_read_only and root_info.st_mode & 0o222:
        raise DepOverlayError(f"published overlay directory is writable: {root}")

    files: list[OverlayFile] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            child = current_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise DepOverlayError(f"overlay contains unsafe directory entry: {child}")
            if require_read_only and info.st_mode & 0o222:
                raise DepOverlayError(f"published overlay directory is writable: {child}")
        for name in sorted(file_names):
            child = current_path / name
            info = child.lstat()
            relative = _canonical_relative(child.relative_to(root))
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise DepOverlayError(f"overlay contains a non-regular file: {relative}")
            if info.st_nlink != 1:
                raise DepOverlayError(f"overlay contains a hardlinked file: {relative}")
            if require_read_only and info.st_mode & 0o222:
                raise DepOverlayError(f"published overlay file is writable: {relative}")
            files.append(OverlayFile(relative, sha256_file(child), info.st_size))
    files.sort(key=lambda row: row.path)
    if not files:
        raise DepOverlayError(f"overlay subtree is empty: {root}")
    payload = [row.to_dict() for row in files]
    return canonical_digest("optima.dep-overlay-tree", payload), tuple(files)


def _patch_rows(bundle: Path, target: str, manifest: Any) -> list[dict[str, str]]:
    return [
        {"path": dp.path, "sha256": sha256_file(bundle / dp.path)}
        for dp in manifest.dep_patches
        if dp.target == target
    ]


def expected_overlay_stamp(
    bundle: str | Path,
    target: str,
    *,
    build_spec_digest: str,
    manifest: Any | None = None,
) -> dict[str, object]:
    from optima.manifest import load_manifest

    root = Path(bundle).resolve()
    manifest = manifest or load_manifest(root)
    policy = PATCHABLE_DEPS.get(target)
    if policy is None:
        raise DepOverlayError(f"active tree declares unapproved dep target {target!r}")
    patches = _patch_rows(root, target, manifest)
    if not patches:
        raise DepOverlayError(f"active tree does not declare a patch for {target!r}")
    return {
        "schema": OVERLAY_SCHEMA,
        "build_spec_digest": _digest(build_spec_digest, field="build_spec_digest"),
        "target": target,
        "policy": policy_snapshot(policy),
        "patches": patches,
        "subtree": policy.overlay_subtree,
    }


def validate_overlay(
    bundle: str | Path,
    target: str,
    *,
    artifact_root: Path,
    build_spec_digest: str,
    require_read_only: bool = False,
) -> ValidatedOverlay:
    """Validate exact stamp, complete source tree, and publication shape."""

    from optima.manifest import load_manifest

    bundle_path = Path(bundle).resolve()
    manifest = load_manifest(bundle_path)
    root = overlay_path(artifact_root, target)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise DepOverlayError(f"declared dep overlay {target!r} is missing: {root}: {exc}") from None
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise DepOverlayError(f"dep overlay root is not a regular directory: {root}")
    if require_read_only and root_info.st_mode & 0o222:
        raise DepOverlayError(f"published overlay root is writable: {root}")
    stamp_path = root / _STAMP_NAME
    try:
        stamp_info = stamp_path.lstat()
    except OSError as exc:
        raise DepOverlayError(
            f"declared dep overlay {target!r} is missing at {stamp_path}: {exc}"
        ) from None
    if stat.S_ISLNK(stamp_info.st_mode) or not stat.S_ISREG(stamp_info.st_mode):
        raise DepOverlayError(f"dep overlay stamp is not a regular file: {stamp_path}")
    if require_read_only and stamp_info.st_mode & 0o222:
        raise DepOverlayError(f"published overlay stamp is writable: {stamp_path}")
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DepOverlayError(f"dep overlay stamp is unreadable: {stamp_path}: {exc}") from None
    if not isinstance(stamp, dict) or set(stamp) != _STAMP_KEYS:
        raise DepOverlayError(f"dep overlay stamp schema mismatch: {stamp_path}")

    expected = expected_overlay_stamp(
        bundle_path,
        target,
        build_spec_digest=build_spec_digest,
        manifest=manifest,
    )
    for key, value in expected.items():
        if stamp.get(key) != value:
            raise DepOverlayError(f"dep overlay stamp field {key!r} mismatches exact build")

    policy = PATCHABLE_DEPS[target]
    subtree = root / policy.overlay_subtree
    # The envelope contains only overlay.json and the one exact subtree chain.
    current = root
    components = PurePosixPath(policy.overlay_subtree).parts
    for index, component in enumerate(components):
        expected_names = {component, _STAMP_NAME} if index == 0 else {component}
        if {entry.name for entry in current.iterdir()} != expected_names:
            raise DepOverlayError(f"dep overlay contains unexpected entries: {current}")
        current = current / component
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise DepOverlayError(f"dep overlay subtree chain is unsafe: {current}")
        if require_read_only and info.st_mode & 0o222:
            raise DepOverlayError(f"published overlay directory is writable: {current}")
    tree_digest, files = tree_inventory(subtree, require_read_only=require_read_only)
    if stamp.get("tree_digest") != tree_digest:
        raise DepOverlayError(f"dep overlay tree digest mismatch for {target!r}")
    rows = [row.to_dict() for row in files]
    if stamp.get("tree_files") != rows:
        raise DepOverlayError(f"dep overlay file inventory mismatch for {target!r}")
    by_path = {
        f"{policy.overlay_subtree}/{row.path}": row.sha256 for row in files
    }
    touched = stamp.get("touched_files")
    if not isinstance(touched, dict) or not touched:
        raise DepOverlayError(f"dep overlay touched-file inventory is malformed for {target!r}")
    if any(by_path.get(path) != digest for path, digest in touched.items()):
        raise DepOverlayError(f"dep overlay touched-file hash mismatch for {target!r}")

    module_rows = stamp.get("prebuilt_modules")
    if not isinstance(module_rows, list) or len(module_rows) != len(policy.prebuilt_modules):
        raise DepOverlayError(f"dep overlay prebuilt-module inventory mismatch for {target!r}")
    modules: list[ValidatedPrebuiltModule] = []
    module_keys = {
        "name",
        "target_architecture",
        "cuda_arch_list",
        "generator_module",
        "generator_attr",
        "consumer_module",
        "consumer_attr",
        "path",
        "sha256",
        "size",
    }
    for row, module in zip(module_rows, policy.prebuilt_modules, strict=True):
        if not isinstance(row, dict) or set(row) != module_keys:
            raise DepOverlayError(f"malformed prebuilt-module row for {target!r}")
        expected_module = {
            **asdict(module),
            "path": prebuilt_module_relative_path(target, module),
        }
        for key, value in expected_module.items():
            if row.get(key) != value:
                raise DepOverlayError(
                    f"prebuilt dependency module field {key!r} mismatches policy"
                )
        module_digest = _digest(row.get("sha256"), field="prebuilt module sha256")
        module_size = row.get("size")
        if type(module_size) is not int or module_size < 1:
            raise DepOverlayError("prebuilt dependency module size is invalid")
        module_path = Path(artifact_root) / row["path"]
        try:
            module_info = module_path.lstat()
        except OSError as exc:
            raise DepOverlayError(f"prebuilt dependency module is missing: {module_path}: {exc}") from None
        if (
            stat.S_ISLNK(module_info.st_mode)
            or not stat.S_ISREG(module_info.st_mode)
            or module_info.st_nlink != 1
            or module_info.st_size != module_size
        ):
            raise DepOverlayError(f"prebuilt dependency module has an unsafe shape: {module_path}")
        if require_read_only and module_info.st_mode & 0o222:
            raise DepOverlayError(f"published dependency module is writable: {module_path}")
        if sha256_file(module_path) != module_digest:
            raise DepOverlayError(f"prebuilt dependency module hash mismatch: {module_path}")
        modules.append(
            ValidatedPrebuiltModule(module, module_path, module_digest, module_size)
        )

        module_dir = module_path.parent
        if {entry.name for entry in module_dir.iterdir()} != {module_path.name}:
            raise DepOverlayError(f"prebuilt dependency module directory has extra entries: {module_dir}")
        if require_read_only:
            parent = module_path.parent
            while True:
                if parent.lstat().st_mode & 0o222:
                    raise DepOverlayError(f"published dependency module directory is writable: {parent}")
                if parent == Path(artifact_root):
                    break
                parent = parent.parent
    if policy.prebuilt_modules:
        target_module_root = Path(artifact_root) / "dep_modules" / target
        expected_module_names = {module.name for module in policy.prebuilt_modules}
        if {entry.name for entry in target_module_root.iterdir()} != expected_module_names:
            raise DepOverlayError(
                f"prebuilt dependency target contains unexpected modules: {target_module_root}"
            )

    if require_read_only:
        artifact_info = Path(artifact_root).lstat()
        if artifact_info.st_mode & 0o222:
            raise DepOverlayError(f"published native artifact root is writable: {artifact_root}")
    return ValidatedOverlay(
        target=target,
        root=root,
        subtree=subtree,
        build_spec_digest=build_spec_digest,
        tree_digest=tree_digest,
        files=files,
        modules=tuple(modules),
        stamp=stamp,
    )


def read_validated_overlay(bundle: str | Path, target: str) -> ValidatedOverlay:
    """Side-effect-free runtime reopen of one exact read-only overlay."""

    phase = rebuild_phase()
    digest = resolved_build_spec_digest(bundle, phase=phase)
    root = native_artifact_root(phase=phase, build_spec_digest=digest)
    if phase == "load":
        publication_digest = _digest(
            os.environ.get("OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST", "").strip(),
            field="OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST",
        )
        from optima.eval.native_artifact import reopen_native_artifact

        publication = reopen_native_artifact(
            root,
            expected_build_spec_digest=digest,
            expected_publication_digest=publication_digest,
            expected_owner_uid=None,
        )
        root = publication.root
    return validate_overlay(
        bundle,
        target,
        artifact_root=root,
        build_spec_digest=digest,
        require_read_only=phase == "load",
    )


def derive_development_build_digest(bundle: str | Path, targets: list[str]) -> str:
    """Whole-input identity for the explicit in-process development lane only."""

    from optima.bundle_hash import content_hash

    bundle_path = Path(bundle).resolve()
    dependencies: list[dict[str, object]] = []
    for target in sorted(set(targets)):
        policy = PATCHABLE_DEPS.get(target)
        if policy is None:
            raise DepOverlayError(f"dependency target is not patchable: {target!r}")
        site_root = dependency_site_root(policy)
        source_digest: str | None = None
        if site_root is not None:
            source_digest, _ = tree_inventory(site_root / policy.overlay_subtree)
        dependencies.append(
            {
                "target": target,
                "policy": policy_snapshot(policy),
                "source_tree_digest": source_digest,
            }
        )
    patcher = Path(__file__).resolve().parent / "patchers" / "apply_dep_patch.py"
    return canonical_digest(
        "optima.dev-native-build",
        {
            "bundle_tree_digest": content_hash(bundle_path),
            "dependencies": dependencies,
            "patcher_sha256": sha256_file(patcher),
        },
    )


__all__ = [
    "DepOverlayError",
    "DepPolicy",
    "OVERLAY_SCHEMA",
    "OverlayFile",
    "PATCHABLE_DEPS",
    "PrebuiltJITModule",
    "ValidatedOverlay",
    "ValidatedPrebuiltModule",
    "dependency_site_root",
    "derive_development_build_digest",
    "expected_overlay_stamp",
    "native_artifact_root",
    "native_build_spec_digest",
    "overlay_base",
    "overlay_path",
    "policy_snapshot",
    "prebuilt_module_relative_path",
    "read_validated_overlay",
    "rebuild_phase",
    "resolved_build_spec_digest",
    "sha256_file",
    "tree_inventory",
    "validate_overlay",
]
