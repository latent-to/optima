"""Compile and seal device-only CuTe CUBIN publications.

Candidate factories execute only in the existing disposable compiler child.
This patcher accepts exactly one retained ``.cubin`` per export and writes the
canonical device-provider index.  It never asks CuTe to export/load a host
object and never imports or calls the CUDA driver; authoritative ABI admission
belongs to the isolated engine worker.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from optima.cuda_cubin import CudaCubinError, _require_elf_cubin
from optima.cute_aot import (
    CUTE_COMPILE_PROFILE_DIGEST_ENV,
    CUTE_COMPILE_PROFILE_ENV,
    CuteAOTError,
    artifact_resource_plan_identity,
    export_launch_plan,
    installed_cute_distributions,
    load_compile_profile,
    _stable_file_bytes,
)
from optima.cute_cubin import (
    CUTE_CUBIN_BINDING_ABI,
    CUTE_CUBIN_INDEX_NAME,
    CUTE_CUBIN_PATCHER,
    CUTE_CUBIN_PROVIDER_NAME,
    CUTE_CUBIN_SCHEMA,
    CUTE_CUBIN_STAGE_DIRECTORY,
    CuteCubinError,
    compile_options_snapshot,
    prepare_cute_cubin_runtime,
    reopen_cute_cubin_index,
)
from optima.patchers.build_cute_aot import (
    _absolute_directory_env,
    _copy_product,
    _digest,
    _phase,
    _run_build_child,
    _safe_bundle_source,
    _stable_digest,
)
from optima.stack_identity import canonical_json_bytes


_MAX_PATCHER_BYTES = 4 << 20


def _log(message: str) -> None:
    print(f"[optima.build_cute_cubin] {message}", flush=True)


def _patcher_sha256() -> str:
    digest, _size = _stable_digest(Path(__file__).resolve(), maximum=_MAX_PATCHER_BYTES)
    return digest


def _manifest_exports(bundle: Path):
    from optima.manifest import load_manifest

    manifest = load_manifest(bundle)
    rows = []
    for op in manifest.ops:
        selected = tuple(
            export
            for export in op.aot_exports
            if export.provider == CUTE_CUBIN_PROVIDER_NAME
        )
        if not selected:
            continue
        _safe_bundle_source(bundle, op.source)
        target_authority = manifest.artifact_target_authority(op)
        _resource_plan, resource_data, resource_sha256 = (
            artifact_resource_plan_identity(
                authority=target_authority,
                resources=op.artifact_resources,
            )
        )
        for export in selected:
            rows.append(
                (
                    op.source,
                    op.slot,
                    op.variant,
                    target_authority.snapshot(),
                    target_authority.digest,
                    resource_data,
                    resource_sha256,
                    export,
                )
            )
    rows.sort(
        key=lambda row: (
            row[1],
            row[2],
            row[7].plan,
            row[7].step,
            row[7].name,
        )
    )
    if not rows:
        raise CuteCubinError(
            "build_cute_cubin selected without declared device exports"
        )
    return tuple(rows)


def _write_index(path: Path, value: dict[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o400)
    try:
        view = memoryview(canonical_json_bytes(value) + b"\n")
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise CuteCubinError("CuTe CUBIN index write stalled")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def build_cute_cubin_stage(bundle: Path, *, stage: Path) -> str:
    if os.environ.get("OPTIMA_REBUILD_CONTAINER") != "1":
        raise CuteCubinError(
            "CuTe CUBIN build requires the disposable rebuild container"
        )
    profile_path = os.environ.get(CUTE_COMPILE_PROFILE_ENV, "").strip()
    profile_digest = os.environ.get(CUTE_COMPILE_PROFILE_DIGEST_ENV, "").strip()
    if not profile_path or not Path(profile_path).is_absolute():
        raise CuteCubinError(
            "CuTe CUBIN build lacks its read-only compile profile mount"
        )
    try:
        profile = load_compile_profile(profile_path, expected_digest=profile_digest)
    except CuteAOTError as exc:
        raise CuteCubinError(str(exc)) from None
    if profile.provider != CUTE_CUBIN_PROVIDER_NAME:
        raise CuteCubinError("CuTe CUBIN compile profile names a different provider")
    build_spec = _digest(
        os.environ.get("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", "").strip(),
        field="native build spec digest",
    )
    tree_digest = _digest(
        os.environ.get("OPTIMA_ENGINE_TREE_DIGEST", "").strip(),
        field="engine tree digest",
    )
    target_architecture = os.environ.get("OPTIMA_TARGET_GPU_ARCH", "").strip()
    if target_architecture != profile.logical_architecture:
        raise CuteCubinError("CuTe CUBIN profile architecture differs from build target")
    timeout_raw = os.environ.get("OPTIMA_NATIVE_COMPILE_TIMEOUT_S", "").strip()
    if not timeout_raw.isdigit() or not 1 <= int(timeout_raw) <= 7_200:
        raise CuteCubinError("CuTe CUBIN compile timeout is invalid")
    timeout_seconds = int(timeout_raw)
    rows = _manifest_exports(bundle)
    tmp_root_raw = os.environ.get("OPTIMA_BUILD_TMPDIR", "").strip()
    tmp_root = Path(tmp_root_raw)
    if not tmp_root_raw or not tmp_root.is_absolute():
        raise CuteCubinError("CuTe CUBIN build tmpdir is not absolute")
    tmp_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="optima-cute-cubin-", dir=tmp_root))
    try:
        destination_root = stage / CUTE_CUBIN_STAGE_DIRECTORY
        destination_root.mkdir(mode=0o700)
        export_rows: list[dict[str, object]] = []
        for (
            source,
            slot,
            variant,
            target_authority,
            target_authority_sha256,
            resource_plan,
            resource_plan_sha256,
            export,
        ) in rows:
            output, artifact_id, _function_prefix, resolved_profile = _run_build_child(
                bundle=bundle,
                source=source,
                slot=slot,
                variant=variant,
                target_authority=target_authority,
                target_authority_sha256=target_authority_sha256,
                resource_plan=resource_plan,
                resource_plan_sha256=resource_plan_sha256,
                export=export,
                profile=profile,
                private_root=private_root,
                timeout_seconds=timeout_seconds,
                stage=stage,
                expected_suffixes=(".cubin",),
            )
            destination_directory = (
                destination_root / "cubins" / artifact_id
            )
            destination_directory.mkdir(mode=0o700, parents=True, exist_ok=False)
            source_cubin = output / f"{artifact_id}.cubin"
            destination_cubin = destination_directory / f"{artifact_id}.cubin"
            cubin_sha256, cubin_size = _copy_product(
                source_cubin, destination_cubin
            )
            try:
                cubin_bytes = _stable_file_bytes(
                    destination_cubin, maximum=1 << 30
                )
                _require_elf_cubin(cubin_bytes)
            except (CudaCubinError, OSError) as exc:
                raise CuteCubinError(
                    f"static CUBIN gate failed for "
                    f"{(slot, variant, export.plan, export.step, export.name)!r}: {exc}"
                ) from None
            launch_plan = export_launch_plan(export)
            device_plan = export.device_plan
            if device_plan is None:
                raise CuteCubinError("device export lacks its sealed device plan")
            export_rows.append(
                {
                    "artifact_id": artifact_id,
                    "artifact_resource_plan": resource_plan,
                    "artifact_resource_plan_sha256": resource_plan_sha256,
                    "artifact_target_authority": target_authority,
                    "artifact_target_authority_sha256": target_authority_sha256,
                    "cubin": {
                        "path": destination_cubin.relative_to(stage).as_posix(),
                        "sha256": cubin_sha256,
                        "size": cubin_size,
                    },
                    "device_plan": device_plan.to_dict(),
                    "device_plan_sha256": device_plan.digest,
                    "factory": export.factory,
                    "launch_plan": launch_plan,
                    "launch_plan_sha256": hashlib.sha256(
                        canonical_json_bytes(launch_plan)
                    ).hexdigest(),
                    "name": export.name,
                    "profile_inputs": list(export.profile_inputs),
                    "resolved_profile": resolved_profile,
                    "slot": slot,
                    "source": source,
                    "variant": variant,
                }
            )
            shutil.rmtree(output.parent)
        patcher_sha256 = _patcher_sha256()
        index = {
            "binding_abi": CUTE_CUBIN_BINDING_ABI,
            "build_spec_digest": build_spec,
            "compile_options": compile_options_snapshot(
                profile.compiler_architecture
            ),
            "compile_profile_digest": profile.digest,
            "compiler_architecture": profile.compiler_architecture,
            "distributions": installed_cute_distributions(),
            "exports": export_rows,
            "logical_architecture": profile.logical_architecture,
            "patcher_id": CUTE_CUBIN_PATCHER,
            "patcher_sha256": patcher_sha256,
            "provider": CUTE_CUBIN_PROVIDER_NAME,
            "schema": CUTE_CUBIN_SCHEMA,
            "tree_digest": tree_digest,
        }
        _write_index(destination_root / CUTE_CUBIN_INDEX_NAME, index)
        reopen_cute_cubin_index(
            stage,
            expected_build_spec_digest=build_spec,
            expected_tree_digest=tree_digest,
            expected_logical_architecture=profile.logical_architecture,
            expected_compile_profile_digest=profile.digest,
            expected_patcher_sha256=patcher_sha256,
        )
        _log(f"built and sealed {len(export_rows)} device-only CuTe CUBIN(s)")
        return profile.digest
    finally:
        shutil.rmtree(private_root, ignore_errors=True)


def validate_cute_cubin_publication(bundle: Path, *, root: Path) -> str:
    if os.environ.get("OPTIMA_ENGINE_WORKER") != "1":
        raise CuteCubinError(
            "CuTe CUBIN load validation requires an isolated engine worker"
        )
    if os.environ.get(CUTE_COMPILE_PROFILE_ENV, "").strip():
        raise CuteCubinError(
            "runtime CuTe CUBIN validation must not receive the profile file"
        )
    build_spec = _digest(
        os.environ.get("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", "").strip(),
        field="native build spec digest",
    )
    tree_digest = _digest(
        os.environ.get("OPTIMA_ENGINE_TREE_DIGEST", "").strip(),
        field="engine tree digest",
    )
    profile_digest = _digest(
        os.environ.get(CUTE_COMPILE_PROFILE_DIGEST_ENV, "").strip(),
        field="runtime CuTe compile profile digest",
    )
    architecture = os.environ.get("OPTIMA_TARGET_GPU_ARCH", "").strip()
    publication_digest = _digest(
        os.environ.get("OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST", "").strip(),
        field="native artifact publication digest",
    )
    index = prepare_cute_cubin_runtime(
        bundle,
        root,
        expected_publication_digest=publication_digest,
        expected_build_spec_digest=build_spec,
        expected_tree_digest=tree_digest,
        expected_logical_architecture=architecture,
        expected_compile_profile_digest=profile_digest,
        expected_patcher_sha256=_patcher_sha256(),
    )
    _log(
        f"validated, admitted, and bound {len(index.exports)} "
        "sealed device-only CuTe CUBIN(s)"
    )
    return index.compile_profile_digest


def main() -> None:
    phase = _phase()
    bundle = _absolute_directory_env("OPTIMA_BUNDLE_PATH")
    if phase == "build":
        build_cute_cubin_stage(
            bundle,
            stage=_absolute_directory_env("OPTIMA_NATIVE_ARTIFACT_STAGE"),
        )
        return
    if phase == "load":
        validate_cute_cubin_publication(
            bundle,
            root=_absolute_directory_env("OPTIMA_NATIVE_ARTIFACT_ROOT")
        )
        return
    raise CuteCubinError(
        "combined CuTe CUBIN build/load is disabled; use isolated publication"
    )


if __name__ == "__main__":
    main()
