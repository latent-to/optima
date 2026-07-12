"""Apply declared text patches to a hermetic dependency-source overlay.

``build`` writes only beneath ``OPTIMA_NATIVE_ARTIFACT_STAGE``.  ``load`` performs
a side-effect-free validation of the exact publication mounted at
``OPTIMA_NATIVE_ARTIFACT_ROOT``.  The complete native-build digest is the namespace;
the manifest's intentionally shared bundle ID is never used.

This reviewed patcher parses bundle files as data.  It never imports bundle Python.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import sys
import tempfile
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath


def _log(message: str) -> None:
    print(f"[optima.apply_dep_patch] {message}", flush=True)


def _check_policy(target: str, file_patches) -> object:
    from optima.dep_policy import PATCHABLE_DEPS

    policy = PATCHABLE_DEPS.get(target)
    if policy is None:
        raise RuntimeError(
            f"dep_patches target {target!r} is not on the validator's patchable-deps "
            f"allowlist ({sorted(PATCHABLE_DEPS)}) — rejecting the bundle"
        )
    subtree = PurePosixPath(policy.overlay_subtree)
    for file_patch in file_patches:
        path = PurePosixPath(file_patch.path)
        if subtree not in path.parents:
            raise RuntimeError(
                f"dep patch touches {file_patch.path!r} outside the overlay subtree "
                f"{policy.overlay_subtree!r} — rejecting the bundle"
            )
        if not any(fnmatch(file_patch.path, pattern) for pattern in policy.allowed_globs):
            raise RuntimeError(
                f"dep patch touches {file_patch.path!r} not matching the arena's allowed "
                f"globs {list(policy.allowed_globs)} — rejecting the bundle"
            )
    return policy


def _make_writable(root: Path) -> None:
    for current, directories, files in os.walk(root):
        Path(current).chmod(0o700)
        for name in directories:
            (Path(current) / name).chmod(0o700)
        for name in files:
            (Path(current) / name).chmod(0o600)


def _prune_empty_directories(root: Path) -> None:
    for current, _directories, _files in os.walk(root, topdown=False):
        path = Path(current)
        if path != root and not any(path.iterdir()):
            path.rmdir()


def _apply_to_overlay(policy, parsed_by_patch, site_root: Path, destination: Path) -> dict[str, str]:
    """Copy exact source input and apply all patches with byte-exact context."""

    from optima.deppatch import apply_file_patch

    source_subtree = site_root / policy.overlay_subtree
    if not source_subtree.is_dir():
        raise RuntimeError(f"pinned dependency subtree missing: {source_subtree}")
    destination_subtree = destination / policy.overlay_subtree
    shutil.copytree(source_subtree, destination_subtree, symlinks=False)
    _make_writable(destination_subtree)

    touched: dict[str, str] = {}
    for _patch_rel, file_patches in parsed_by_patch:
        for file_patch in file_patches:
            target_file = destination / file_patch.path
            original = target_file.read_text(encoding="utf-8") if target_file.exists() else None
            new_text = apply_file_patch(original, file_patch)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(new_text, encoding="utf-8")
            touched[file_patch.path] = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
    _prune_empty_directories(destination_subtree)
    return dict(sorted(touched.items()))


def _materialize(
    bundle: Path,
    target: str,
    entries,
    policy,
    *,
    artifact_root: Path,
    build_spec_digest: str,
    manifest,
) -> None:
    from optima.dep_policy import (
        expected_overlay_stamp,
        overlay_path,
        tree_inventory,
        validate_overlay,
    )

    from optima.dep_policy import dependency_site_root

    site_root = dependency_site_root(policy)
    if site_root is None:
        raise RuntimeError(
            f"dependency {policy.package!r} is absent from the hermetic build image"
        )
    destination = overlay_path(artifact_root, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(
            f"refusing to replace existing dep-overlay build output: {destination}"
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{target}.", dir=destination.parent))
    try:
        touched = _apply_to_overlay(policy, entries, site_root, temporary)
        tree_digest, files = tree_inventory(temporary / policy.overlay_subtree)
        prebuilt_modules = _build_prebuilt_modules(
            policy,
            target=target,
            overlay_subtree=temporary / policy.overlay_subtree,
            artifact_root=artifact_root,
        )
        stamp = expected_overlay_stamp(
            bundle,
            target,
            build_spec_digest=build_spec_digest,
            manifest=manifest,
        )
        stamp.update(
            {
                "touched_files": touched,
                "tree_digest": tree_digest,
                "tree_files": [row.to_dict() for row in files],
                "prebuilt_modules": prebuilt_modules,
            }
        )
        (temporary / "overlay.json").write_bytes(
            json.dumps(stamp, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        os.rename(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    validated = validate_overlay(
        bundle,
        target,
        artifact_root=artifact_root,
        build_spec_digest=build_spec_digest,
    )
    _log(
        f"overlay staged for {target}: {validated.subtree} "
        f"({len(validated.files)} source files, build={build_spec_digest[:12]})"
    )


def _required_target_architecture(policy) -> str:
    architectures = {module.target_architecture for module in policy.prebuilt_modules}
    if not architectures:
        return ""
    if len(architectures) != 1:
        raise RuntimeError("one dependency policy cannot prebuild multiple target architectures")
    expected = next(iter(architectures))
    actual = os.environ.get("OPTIMA_TARGET_GPU_ARCH", "").strip().lower()
    if not actual and os.environ.get("OPTIMA_REBUILD_PHASE", "all") == "all":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "dependency development rebuild requires a live CUDA device to "
                "derive OPTIMA_TARGET_GPU_ARCH"
            )
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        actual = f"sm{major}{minor}"
        os.environ["OPTIMA_TARGET_GPU_ARCH"] = actual
    if actual != expected:
        raise RuntimeError(
            f"dependency prebuild requires target architecture {expected!r}, got {actual!r}"
        )
    return expected


def _copy_built_module(source: Path, destination: Path) -> tuple[str, int]:
    try:
        info = source.lstat()
    except OSError as exc:
        raise RuntimeError(f"dependency generator did not produce its shared object: {exc}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"dependency generator produced an unsafe shared object: {source}")
    if info.st_size < 1 or info.st_size > (1 << 30):
        raise RuntimeError(f"dependency shared object has an invalid size: {source}")
    destination.parent.mkdir(parents=True, exist_ok=False)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=4 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    destination.chmod(0o444)
    return hashlib.sha256(destination.read_bytes()).hexdigest(), info.st_size


def _build_prebuilt_modules(
    policy,
    *,
    target: str,
    overlay_subtree: Path,
    artifact_root: Path,
) -> list[dict[str, object]]:
    """Compile exact validator-declared FlashInfer specs without loading them."""

    from dataclasses import asdict
    from optima.dep_policy import prebuilt_module_relative_path

    if not policy.prebuilt_modules:
        return []
    _required_target_architecture(policy)
    if any(name == "flashinfer" or name.startswith("flashinfer.") for name in sys.modules):
        raise RuntimeError(
            "flashinfer was imported before the hermetic dependency prebuild environment"
        )

    cuda_arch_lists = {module.cuda_arch_list for module in policy.prebuilt_modules}
    if len(cuda_arch_lists) != 1:
        raise RuntimeError("one dependency policy cannot use multiple CUDA arch lists")
    cuda_arch_list = next(iter(cuda_arch_lists))
    existing_arch = os.environ.get("FLASHINFER_CUDA_ARCH_LIST")
    if existing_arch not in {None, cuda_arch_list}:
        raise RuntimeError(
            "FLASHINFER_CUDA_ARCH_LIST conflicts with the validator prebuild policy"
        )

    rows: list[dict[str, object]] = []
    # Keep compiler intermediates inside the quota-backed build stage.  /tmp is a
    # deliberately small noexec control tmpfs and must not absorb a multi-GB
    # FlashInfer/Ninja build.
    with tempfile.TemporaryDirectory(
        prefix=".flashinfer-build-", dir=artifact_root
    ) as scratch_raw:
        scratch = Path(scratch_raw)
        old_workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
        os.environ["FLASHINFER_WORKSPACE_BASE"] = str(scratch)
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = cuda_arch_list
        try:
            jit_environment = importlib.import_module("flashinfer.jit.env")
            jit_environment.FLASHINFER_CSRC_DIR = overlay_subtree
            for module in policy.prebuilt_modules:
                generator_module = importlib.import_module(module.generator_module)
                generator = getattr(generator_module, module.generator_attr, None)
                if not callable(generator):
                    raise RuntimeError(
                        f"dependency generator is missing: "
                        f"{module.generator_module}.{module.generator_attr}"
                    )
                spec = generator(False)
                if getattr(spec, "name", None) != module.name:
                    raise RuntimeError(
                        f"dependency generator returned unexpected spec name: "
                        f"{getattr(spec, 'name', None)!r}"
                    )
                # Deliberately compile only.  build_and_load()/tvm_ffi must never run
                # in the prebuild worker.
                spec.build(verbose=False)
                built = Path(spec.jit_library_path)
                try:
                    built.resolve().relative_to(scratch.resolve())
                except (OSError, ValueError):
                    raise RuntimeError(
                        f"dependency generator wrote outside its private scratch: {built}"
                    ) from None
                relative = prebuilt_module_relative_path(target, module)
                destination = artifact_root / relative
                if destination.exists() or destination.is_symlink():
                    raise RuntimeError(
                        f"refusing to replace dependency module build output: {destination}"
                    )
                digest, size = _copy_built_module(built, destination)
                rows.append(
                    {
                        **asdict(module),
                        "path": relative,
                        "sha256": digest,
                        "size": size,
                    }
                )
        finally:
            if old_workspace is None:
                os.environ.pop("FLASHINFER_WORKSPACE_BASE", None)
            else:
                os.environ["FLASHINFER_WORKSPACE_BASE"] = old_workspace
            if existing_arch is None:
                os.environ.pop("FLASHINFER_CUDA_ARCH_LIST", None)
            else:
                os.environ["FLASHINFER_CUDA_ARCH_LIST"] = existing_arch
    return rows

def _development_workspace(build_spec_digest: str) -> Path:
    raw = os.environ.get("OPTIMA_DEP_OVERLAY_CACHE", "").strip()
    base = Path(raw) if raw else Path.home() / ".cache" / "optima" / "dep_overlay"
    return base / "jit_workspace" / "v3" / build_spec_digest[:2] / build_spec_digest


def main() -> None:
    bundle_raw = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not bundle_raw:
        _log("no OPTIMA_BUNDLE_PATH set; nothing to apply")
        return

    from optima.deppatch import parse_patch_text
    from optima.dep_policy import (
        dependency_site_root,
        native_artifact_root,
        rebuild_phase,
        resolved_build_spec_digest,
        validate_overlay,
    )
    from optima.manifest import load_manifest

    bundle = Path(bundle_raw).resolve()
    manifest = load_manifest(bundle)
    phase = rebuild_phase()
    if not manifest.dep_patches:
        _log("bundle declares no dep_patches; nothing to apply")
        return

    # Parse and policy-check every declaration before touching the output stage.
    by_target: dict[str, list[tuple[str, tuple]]] = {}
    for declaration in manifest.dep_patches:
        raw = (bundle / declaration.path).read_bytes()
        parsed = parse_patch_text(raw.decode("utf-8"))
        by_target.setdefault(declaration.target, []).append((declaration.path, parsed))
    policies = {
        target: _check_policy(
            target,
            [file_patch for _path, parsed in entries for file_patch in parsed],
        )
        for target, entries in by_target.items()
    }

    build_spec_digest = resolved_build_spec_digest(bundle, phase=phase)
    artifact_root = native_artifact_root(
        phase=phase, build_spec_digest=build_spec_digest
    )
    if phase == "all":
        artifact_root.mkdir(parents=True, exist_ok=True)
        # Preserve the old direct-eval lane without letting a patched and stock JIT
        # race in one workspace.  Production runtime supplies a launch-private path.
        if "FLASHINFER_WORKSPACE_BASE" not in os.environ:
            workspace = _development_workspace(build_spec_digest)
            workspace.mkdir(parents=True, exist_ok=True)
            os.environ["FLASHINFER_WORKSPACE_BASE"] = str(workspace)

    for target, entries in sorted(by_target.items()):
        if phase == "load":
            validated = validate_overlay(
                bundle,
                target,
                artifact_root=artifact_root,
                build_spec_digest=build_spec_digest,
                require_read_only=True,
            )
            _log(
                f"overlay reopened for {target}: {validated.subtree} "
                f"(build={build_spec_digest[:12]})"
            )
            continue

        if phase == "all" and dependency_site_root(policies[target]) is None:
            _log(
                f"dependency {policies[target].package!r} is not installed; "
                "policy checks passed, skipping explicit development materialization"
            )
            continue

        destination = artifact_root / "dep_overlays" / target
        if destination.exists() and phase == "all":
            validated = validate_overlay(
                bundle,
                target,
                artifact_root=artifact_root,
                build_spec_digest=build_spec_digest,
            )
            _log(f"development overlay cache hit for {target}: {validated.subtree}")
            continue
        _materialize(
            bundle,
            target,
            entries,
            policies[target],
            artifact_root=artifact_root,
            build_spec_digest=build_spec_digest,
            manifest=manifest,
        )


main()
