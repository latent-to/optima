from __future__ import annotations

import copy
import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import optima.cute_cubin as cute_cubin
import optima.patchers.build_cute_aot as build_cute_aot
import optima.patchers.build_cute_cubin as build_cute_cubin
from optima.artifact_abi import (
    ArtifactBinding,
    ArtifactPrelaunch,
    parse_artifact_bindings,
    parse_artifact_prelaunch,
    parse_provider_capability_requirements,
    parse_specialization_capability_requirements,
    slot_call_abi,
)
from optima.artifact_device_launch import (
    DeviceDim3Plan,
    DeviceLaunchInvocation,
    DeviceLaunchPlan,
)
from optima.artifact_identity import direct_artifact_execution_identity
from optima.artifact_runtime import (
    ArtifactRuntimeProvider,
    resolve_direct_artifact_entry,
    shutdown_direct_artifact_runtimes,
)
from optima.cuda_cubin import CudaCubinLibrary, CudaKernelContract
from optima.cuda_materialize import (
    CudaCheckedExpression,
    CudaExpressionNode,
    CudaParameterPlan,
)
from optima.cute_aot import (
    CuteAOTError,
    artifact_resource_plan_identity,
    deterministic_export_names,
)
from optima.cute_cubin import (
    CUTE_CUBIN_BINDING_ABI,
    CUTE_CUBIN_INDEX_RELPATH,
    CUTE_CUBIN_PATCHER,
    CUTE_CUBIN_PROVIDER_NAME,
    CUTE_CUBIN_SCHEMA,
    CUTE_CUBIN_STAGE_DIRECTORY,
    CuteCubinError,
    compile_options_snapshot,
    prepare_cute_cubin_runtime,
    reopen_cute_cubin_index,
)
from optima.manifest import (
    ABI_VERSION,
    Manifest,
    ManifestError,
    OpEntry,
    _parse_aot_exports,
    static_artifact_target_authority,
)
from optima.stack_identity import canonical_json_bytes


SLOT = "attention.msa_prefill_block_score"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _cubin() -> bytes:
    raw = bytearray(64)
    raw[:4] = b"\x7fELF"
    raw[4] = 2  # ELFCLASS64
    raw[5] = 1  # little endian
    raw[6] = 1  # current ELF identification
    raw[16:18] = (2).to_bytes(2, "little")  # ET_EXEC
    raw[18:20] = (190).to_bytes(2, "little")  # EM_CUDA
    raw[20:24] = (1).to_bytes(4, "little")
    raw[52:54] = (64).to_bytes(2, "little")
    return bytes(raw)


def test_shared_build_child_accepts_exact_single_cubin_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id = "artifact"
    output = tmp_path / artifact_id / "output"

    def emit(*_args, **_kwargs):
        (output / f"{artifact_id}.cubin").write_bytes(_cubin())
        return 0, ""

    monkeypatch.setattr(
        build_cute_aot,
        "deterministic_export_names",
        lambda **_kwargs: (artifact_id, "function_prefix"),
    )
    monkeypatch.setattr(build_cute_aot, "export_launch_plan", lambda _export: {})
    monkeypatch.setattr(build_cute_aot, "_child_environment", lambda _root: {})
    monkeypatch.setattr(
        build_cute_aot,
        "_run_isolated_compiler",
        emit,
    )
    monkeypatch.setattr(build_cute_aot, "_stage_snapshot", lambda _stage: ())
    monkeypatch.setattr(
        build_cute_aot,
        "_stable_digest",
        lambda path, **_kwargs: (_digest(path.name), path.stat().st_size),
    )
    export = SimpleNamespace(
        factory="build_blockscore",
        name="blockscore",
        profile_inputs=(),
        provider=CUTE_CUBIN_PROVIDER_NAME,
    )
    profile = SimpleNamespace(
        compiler_architecture="sm_103a",
        require_int=lambda _key: 1,
    )

    child_output, observed_id, _prefix, resolved = build_cute_aot._run_build_child(
        bundle=tmp_path,
        source="kernels/blockscore.py",
        slot=SLOT,
        variant="cute",
        target_authority={},
        target_authority_sha256=_digest("target"),
        resource_plan={},
        resource_plan_sha256=_digest("resources"),
        export=export,
        profile=profile,
        private_root=tmp_path,
        timeout_seconds=60,
        stage=tmp_path / "stage",
        expected_suffixes=(".cubin",),
    )

    assert child_output == output
    assert observed_id == artifact_id
    assert resolved == {}


@pytest.mark.parametrize(
    "emitted_names",
    (
        ("artifact.cubin", "artifact.ptx"),
        ("artifact.ptx",),
    ),
)
def test_shared_build_child_rejects_cubin_product_count_or_name_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    emitted_names: tuple[str, ...],
) -> None:
    artifact_id = "artifact"
    output = tmp_path / artifact_id / "output"

    def emit(*_args, **_kwargs):
        for name in emitted_names:
            (output / name).write_bytes(_cubin())
        return 0, ""

    monkeypatch.setattr(
        build_cute_aot,
        "deterministic_export_names",
        lambda **_kwargs: (artifact_id, "function_prefix"),
    )
    monkeypatch.setattr(build_cute_aot, "export_launch_plan", lambda _export: {})
    monkeypatch.setattr(build_cute_aot, "_child_environment", lambda _root: {})
    monkeypatch.setattr(build_cute_aot, "_run_isolated_compiler", emit)
    monkeypatch.setattr(build_cute_aot, "_stage_snapshot", lambda _stage: ())
    export = SimpleNamespace(
        factory="build_blockscore",
        name="blockscore",
        profile_inputs=(),
        provider=CUTE_CUBIN_PROVIDER_NAME,
    )
    profile = SimpleNamespace(
        compiler_architecture="sm_103a",
        require_int=lambda _key: 1,
    )

    with pytest.raises(CuteAOTError, match="unexpected products"):
        build_cute_aot._run_build_child(
            bundle=tmp_path,
            source="kernels/blockscore.py",
            slot=SLOT,
            variant="cute",
            target_authority={},
            target_authority_sha256=_digest("target"),
            resource_plan={},
            resource_plan_sha256=_digest("resources"),
            export=export,
            profile=profile,
            private_root=tmp_path,
            timeout_seconds=60,
            stage=tmp_path / "stage",
            expected_suffixes=(".cubin",),
        )


def _constant(value: int) -> CudaCheckedExpression:
    return CudaCheckedExpression(
        nodes=(CudaExpressionNode(op="const", value=value),),
        result=0,
    )


def _binding_scalar(binding: int) -> CudaCheckedExpression:
    return CudaCheckedExpression(
        nodes=(CudaExpressionNode(op="binding_scalar", binding=binding),),
        result=0,
    )


def _output_grid() -> CudaCheckedExpression:
    return CudaCheckedExpression(
        nodes=(
            CudaExpressionNode(op="tensor_numel", binding=2),
            CudaExpressionNode(op="const", value=128),
            CudaExpressionNode(op="ceil_div", operands=(0, 1)),
        ),
        result=2,
    )


def _device_plan() -> DeviceLaunchPlan:
    parameters = (
        CudaParameterPlan(kind="pointer", size=8, binding=0),
        CudaParameterPlan(kind="pointer", size=8, binding=1),
        CudaParameterPlan(kind="pointer", size=8, binding=2),
        CudaParameterPlan(
            kind="scalar",
            size=4,
            scalar_type="i32",
            expression=_binding_scalar(3),
        ),
        CudaParameterPlan(
            kind="scalar",
            size=4,
            scalar_type="f32",
            expression=_binding_scalar(4),
        ),
    )
    return DeviceLaunchPlan(
        kernels=(
            CudaKernelContract(
                name="blockscore_kernel",
                parameter_sizes=tuple(parameter.size for parameter in parameters),
            ),
        ),
        launches=(
            DeviceLaunchInvocation(
                ordinal=0,
                kernel="blockscore_kernel",
                grid=DeviceDim3Plan(_output_grid(), _constant(1), _constant(1)),
                block=DeviceDim3Plan(_constant(128), _constant(1), _constant(1)),
                cluster=None,
                shared_mem_bytes=_constant(0),
                parameters=parameters,
                stream_binding=5,
            ),
        ),
    )


def _launch_plan() -> dict[str, object]:
    bindings = (
        ArtifactBinding(
            "input.q", "tensor", unsqueeze=(-1,), assumed_align=16, leading_dim=1
        ),
        ArtifactBinding(
            "input.index_k",
            "tensor",
            unsqueeze=(-1,),
            assumed_align=16,
            leading_dim=1,
        ),
        ArtifactBinding(
            "output.block_scores", "tensor", assumed_align=4, leading_dim=1
        ),
        ArtifactBinding("input.prefix_len", "scalar", cast="i32"),
        ArtifactBinding("input.scale", "scalar", cast="f32"),
        ArtifactBinding("stream.current", "stream"),
    )
    prelaunch = (ArtifactPrelaunch("fill", "output.block_scores", "-inf"),)
    return {
        "bindings": [binding.to_dict() for binding in bindings],
        "device_plan": _device_plan().to_dict(),
        "plan": "default",
        "prelaunch": [operation.to_dict() for operation in prelaunch],
        "provider_capability_requirements": [],
        "role": "run",
        "specialization_capability_requirements": [
            {
                "capability_field": "block_size",
                "source": "input.block_size",
            }
        ],
        "specializes": {"input.block_size": 128},
        "step": 0,
    }


def _manifest_export_row(
    *, provider: str = CUTE_CUBIN_PROVIDER_NAME
) -> dict[str, object]:
    plan = _launch_plan()
    return {
        "bindings": plan["bindings"],
        "device_plan": plan["device_plan"],
        "factory": "build_blockscore",
        "name": "blockscore",
        "plan": plan["plan"],
        "prelaunch": plan["prelaunch"],
        "profile_inputs": ["max_active_clusters.cluster_size_1"],
        "provider": provider,
        "role": plan["role"],
        "specializes": plan["specializes"],
        "step": plan["step"],
    }


def _publication(root: Path) -> tuple[dict[str, object], Path]:
    source = "kernels/blockscore.py"
    variant = "cute"
    name = "blockscore"
    factory = "build_blockscore"
    plan = _launch_plan()
    target = static_artifact_target_authority(SLOT)
    _resources, resource_data, resource_sha256 = artifact_resource_plan_identity(
        authority=target,
        resources=(),
    )
    artifact_id, _prefix = deterministic_export_names(
        source=source,
        slot=SLOT,
        variant=variant,
        name=name,
        factory=factory,
        plan=plan,
        artifact_resource_plan_sha256=resource_sha256,
        artifact_target_authority_sha256=target.digest,
    )
    raw = _cubin()
    cubin_sha256 = hashlib.sha256(raw).hexdigest()
    cubin_path = (
        root
        / CUTE_CUBIN_STAGE_DIRECTORY
        / "cubins"
        / artifact_id
        / f"{artifact_id}.cubin"
    )
    cubin_path.parent.mkdir(parents=True)
    cubin_path.write_bytes(raw)
    launch_sha256 = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    row: dict[str, object] = {
        "binding_abi": CUTE_CUBIN_BINDING_ABI,
        "build_spec_digest": _digest("build"),
        "compile_options": compile_options_snapshot("sm_103a"),
        "compile_profile_digest": _digest("profile"),
        "compiler_architecture": "sm_103a",
        "distributions": {
            "nvidia-cutlass-dsl": "4.5.2",
            "nvidia-cutlass-dsl-libs-base": "4.5.2",
            "nvidia-cutlass-dsl-libs-cu13": "4.5.2",
        },
        "exports": [
            {
                "artifact_id": artifact_id,
                "artifact_resource_plan": resource_data,
                "artifact_resource_plan_sha256": resource_sha256,
                "artifact_target_authority": target.snapshot(),
                "artifact_target_authority_sha256": target.digest,
                "cubin": {
                    "path": cubin_path.relative_to(root).as_posix(),
                    "sha256": cubin_sha256,
                    "size": len(raw),
                },
                "device_plan": _device_plan().to_dict(),
                "device_plan_sha256": _device_plan().digest,
                "factory": factory,
                "launch_plan": plan,
                "launch_plan_sha256": launch_sha256,
                "name": name,
                "profile_inputs": ["max_active_clusters.cluster_size_1"],
                "resolved_profile": {
                    "max_active_clusters.cluster_size_1": 148
                },
                "slot": SLOT,
                "source": source,
                "variant": variant,
            }
        ],
        "logical_architecture": "sm103",
        "patcher_id": CUTE_CUBIN_PATCHER,
        "patcher_sha256": _digest("patcher"),
        "provider": CUTE_CUBIN_PROVIDER_NAME,
        "schema": CUTE_CUBIN_SCHEMA,
        "tree_digest": _digest("tree"),
    }
    index_path = root / CUTE_CUBIN_INDEX_RELPATH
    index_path.write_bytes(canonical_json_bytes(row) + b"\n")
    return row, index_path


def test_device_publication_reopens_without_loading_candidate_host_code(
    tmp_path: Path,
) -> None:
    row, _index_path = _publication(tmp_path)

    reopened = reopen_cute_cubin_index(
        tmp_path,
        expected_build_spec_digest=row["build_spec_digest"],  # type: ignore[arg-type]
        expected_tree_digest=row["tree_digest"],  # type: ignore[arg-type]
        expected_logical_architecture="sm103",
        expected_compile_profile_digest=row["compile_profile_digest"],  # type: ignore[arg-type]
        expected_patcher_sha256=row["patcher_sha256"],  # type: ignore[arg-type]
        verify_distributions=False,
    )

    assert len(reopened.exports) == 1
    assert reopened.exports[0].cubin.sha256 == hashlib.sha256(_cubin()).hexdigest()
    assert reopened.exports[0].launch_plan["bindings"] == _launch_plan()["bindings"]
    assert reopened.compile_options[-1] == ("no_jit_engine", True)


def test_manifest_requires_device_plan_and_rejects_removed_host_provider() -> None:
    call_abi = slot_call_abi(SLOT)
    assert call_abi is not None
    row = _manifest_export_row()

    parsed = _parse_aot_exports(
        [row],
        op_index=0,
        slot=SLOT,
        call_abi=call_abi,
    )
    assert parsed[0].device_plan == _device_plan()

    missing = copy.deepcopy(row)
    del missing["device_plan"]
    with pytest.raises(ManifestError, match="device provider requires device_plan"):
        _parse_aot_exports(
            [missing],
            op_index=0,
            slot=SLOT,
            call_abi=call_abi,
        )

    host = copy.deepcopy(row)
    host["provider"] = "cutlass.cute.object.v1"
    with pytest.raises(ManifestError, match="provider .* is not registered"):
        _parse_aot_exports(
            [host],
            op_index=0,
            slot=SLOT,
            call_abi=call_abi,
        )


def test_device_plan_rotates_artifact_identity() -> None:
    original = _launch_plan()
    changed = copy.deepcopy(original)
    changed_device = changed["device_plan"]
    assert isinstance(changed_device, dict)
    launches = changed_device["launches"]
    assert isinstance(launches, list)
    block_x = launches[0]["block"]["x"]  # type: ignore[index]
    block_x["nodes"][0]["value"] = 64  # type: ignore[index]
    changed_plan = DeviceLaunchPlan.from_dict(changed_device)
    changed["device_plan"] = changed_plan.to_dict()
    target = static_artifact_target_authority(SLOT)
    _resources, _resource_data, resource_sha256 = artifact_resource_plan_identity(
        authority=target,
        resources=(),
    )
    common = {
        "source": "kernels/blockscore.py",
        "slot": SLOT,
        "variant": "cute",
        "name": "blockscore",
        "factory": "build_blockscore",
        "artifact_resource_plan_sha256": resource_sha256,
        "artifact_target_authority_sha256": target.digest,
    }

    original_id, _ = deterministic_export_names(plan=original, **common)
    changed_id, _ = deterministic_export_names(plan=changed, **common)

    assert _device_plan().digest != changed_plan.digest
    assert original_id != changed_id


def test_device_plan_float_constant_has_exact_sealable_identity() -> None:
    base = _device_plan()
    parameters = list(base.launches[0].parameters)
    parameters[4] = CudaParameterPlan(
        kind="scalar",
        size=4,
        scalar_type="f32",
        expression=CudaCheckedExpression(
            nodes=(CudaExpressionNode(op="const", value=-0.0),),
            result=0,
        ),
    )
    plan = DeviceLaunchPlan(
        kernels=base.kernels,
        launches=(replace(base.launches[0], parameters=tuple(parameters)),),
    )
    wire = plan.to_dict()
    encoded_value = wire["launches"][0]["parameters"][4]["expression"][  # type: ignore[index]
        "nodes"
    ][0]["value"]

    assert encoded_value == {"f64_hex": "-0x0.0p+0"}
    assert DeviceLaunchPlan.from_dict(wire) == plan
    assert len(plan.digest) == 64
    assert canonical_json_bytes(wire)


def test_device_plan_rotates_direct_artifact_submission_identity() -> None:
    call_abi = slot_call_abi(SLOT)
    assert call_abi is not None
    original_row = _manifest_export_row()
    changed_row = copy.deepcopy(original_row)
    raw_plan = changed_row["device_plan"]
    raw_plan["launches"][0]["block"]["x"]["nodes"][0]["value"] = 64  # type: ignore[index]
    changed_row["device_plan"] = DeviceLaunchPlan.from_dict(raw_plan).to_dict()

    def identity(row: dict[str, object]) -> dict[str, object]:
        exports = _parse_aot_exports(
            [row],
            op_index=0,
            slot=SLOT,
            call_abi=call_abi,
        )
        op = OpEntry(
            slot=SLOT,
            source="kernels/blockscore.py",
            entry="unused_direct_entry",
            dtypes=("bfloat16",),
            architectures=("sm103",),
            metadata=None,
            variant="cute",
            aot_exports=exports,
        )
        manifest = Manifest(
            bundle_id="device-plan-identity",
            abi_version=ABI_VERSION,
            ops=(op,),
        )
        return direct_artifact_execution_identity(manifest, op)

    original_identity = identity(original_row)
    changed_identity = identity(changed_row)

    assert original_identity["exports"][0]["device_plan"] == original_row[  # type: ignore[index]
        "device_plan"
    ]
    assert original_identity != changed_identity


def test_publication_rejects_device_plan_divergent_from_launch_identity(
    tmp_path: Path,
) -> None:
    row, index_path = _publication(tmp_path)
    tampered = copy.deepcopy(row)
    export = tampered["exports"][0]  # type: ignore[index]
    raw_plan = export["device_plan"]
    raw_plan["launches"][0]["block"]["x"]["nodes"][0]["value"] = 64  # type: ignore[index]
    changed_plan = DeviceLaunchPlan.from_dict(raw_plan)
    export["device_plan"] = changed_plan.to_dict()
    export["device_plan_sha256"] = changed_plan.digest
    index_path.write_bytes(canonical_json_bytes(tampered) + b"\n")

    with pytest.raises(CuteCubinError, match="differs from launch authority"):
        reopen_cute_cubin_index(tmp_path, verify_distributions=False)


def test_device_publication_static_reopen_never_calls_cuda_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _row, _index_path = _publication(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("GPU-free publication reopen called the CUDA driver")

    monkeypatch.setattr(CudaCubinLibrary, "inspect_abi", classmethod(forbidden))
    monkeypatch.setattr(CudaCubinLibrary, "open", classmethod(forbidden))
    monkeypatch.setattr(CudaCubinLibrary, "open_contract", classmethod(forbidden))
    monkeypatch.setattr(CudaCubinLibrary, "_capture_driver", staticmethod(forbidden))

    reopened = reopen_cute_cubin_index(
        tmp_path,
        verify_distributions=False,
    )

    assert len(reopened.exports) == 1


def test_device_publication_build_stage_never_calls_cuda_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    profile_path = tmp_path / "compile-profile.json"
    profile_path.write_text("{}\n")
    private_tmp = tmp_path / "private"
    private_tmp.mkdir()
    target = static_artifact_target_authority(SLOT)
    _resources, resource_data, resource_sha256 = artifact_resource_plan_identity(
        authority=target,
        resources=(),
    )
    plan = _launch_plan()
    export = SimpleNamespace(
        bindings=parse_artifact_bindings(
            plan["bindings"], field="test device bindings"
        ),
        factory="build_blockscore",
        name="blockscore",
        plan="default",
        prelaunch=parse_artifact_prelaunch(
            plan["prelaunch"], field="test device prelaunch"
        ),
        profile_inputs=("max_active_clusters.cluster_size_1",),
        provider_capability_requirements=parse_provider_capability_requirements(
            plan["provider_capability_requirements"],
            field="test device provider requirements",
        ),
        role="run",
        specialization_capability_requirements=(
            parse_specialization_capability_requirements(
                plan["specialization_capability_requirements"],
                field="test device specialization requirements",
            )
        ),
        specializes=tuple(plan["specializes"].items()),
        step=0,
        device_plan=_device_plan(),
    )
    profile = SimpleNamespace(
        compiler_architecture="sm_103a",
        digest=_digest("profile"),
        logical_architecture="sm103",
        provider=CUTE_CUBIN_PROVIDER_NAME,
        require_int=lambda key: 148,
    )
    distributions = {
        "nvidia-cutlass-dsl": "4.5.2",
        "nvidia-cutlass-dsl-libs-base": "4.5.2",
        "nvidia-cutlass-dsl-libs-cu13": "4.5.2",
    }

    monkeypatch.setattr(
        build_cute_cubin,
        "load_compile_profile",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        build_cute_cubin,
        "_manifest_exports",
        lambda _bundle: (
            (
                "kernels/blockscore.py",
                SLOT,
                "cute",
                target.snapshot(),
                target.digest,
                resource_data,
                resource_sha256,
                export,
            ),
        ),
    )
    monkeypatch.setattr(
        build_cute_cubin,
        "installed_cute_distributions",
        lambda: distributions,
    )
    monkeypatch.setattr(
        cute_cubin,
        "installed_cute_distributions",
        lambda: distributions,
    )

    def fake_build_child(**kwargs):
        artifact_id, function_prefix = deterministic_export_names(
            source=kwargs["source"],
            slot=kwargs["slot"],
            variant=kwargs["variant"],
            name=kwargs["export"].name,
            factory=kwargs["export"].factory,
            plan=plan,
            artifact_resource_plan_sha256=resource_sha256,
            artifact_target_authority_sha256=target.digest,
        )
        output = kwargs["private_root"] / "compiler-child" / "output"
        output.mkdir(parents=True)
        (output / f"{artifact_id}.cubin").write_bytes(_cubin())
        return (
            output,
            artifact_id,
            function_prefix,
            {"max_active_clusters.cluster_size_1": 148},
        )

    monkeypatch.setattr(build_cute_cubin, "_run_build_child", fake_build_child)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("GPU-free publication build called the CUDA driver")

    monkeypatch.setattr(CudaCubinLibrary, "inspect_abi", classmethod(forbidden))
    monkeypatch.setattr(CudaCubinLibrary, "open", classmethod(forbidden))
    monkeypatch.setattr(CudaCubinLibrary, "open_contract", classmethod(forbidden))
    monkeypatch.setattr(CudaCubinLibrary, "_capture_driver", staticmethod(forbidden))
    monkeypatch.setenv("OPTIMA_REBUILD_CONTAINER", "1")
    monkeypatch.setenv("OPTIMA_CUTE_COMPILE_PROFILE", str(profile_path))
    monkeypatch.setenv("OPTIMA_CUTE_COMPILE_PROFILE_DIGEST", profile.digest)
    monkeypatch.setenv("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", _digest("build"))
    monkeypatch.setenv("OPTIMA_ENGINE_TREE_DIGEST", _digest("tree"))
    monkeypatch.setenv("OPTIMA_TARGET_GPU_ARCH", "sm103")
    monkeypatch.setenv("OPTIMA_NATIVE_COMPILE_TIMEOUT_S", "60")
    monkeypatch.setenv("OPTIMA_BUILD_TMPDIR", str(private_tmp))

    assert build_cute_cubin.build_cute_cubin_stage(bundle, stage=stage) == profile.digest
    assert reopen_cute_cubin_index(
        stage,
        expected_build_spec_digest=_digest("build"),
        expected_tree_digest=_digest("tree"),
        expected_logical_architecture="sm103",
        expected_compile_profile_digest=profile.digest,
        verify_distributions=False,
    ).exports[0].cubin.size == len(_cubin())


def test_device_publication_load_resolve_invoke_and_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import optima.artifact_device_launch as device_launch
    import optima.cuda_materialize as cuda_materialize
    import optima.eval.native_artifact as native_artifact
    import optima.manifest as manifest_module
    import optima.receipts as receipts

    row, _index_path = _publication(tmp_path)
    index = reopen_cute_cubin_index(tmp_path, verify_distributions=False)
    call_abi = slot_call_abi(SLOT)
    assert call_abi is not None
    exports = _parse_aot_exports(
        [_manifest_export_row()],
        op_index=0,
        slot=SLOT,
        call_abi=call_abi,
    )
    op = OpEntry(
        slot=SLOT,
        source="kernels/blockscore.py",
        entry="unused_direct_entry",
        dtypes=("bfloat16",),
        architectures=("sm103",),
        metadata=None,
        variant="cute",
        aot_exports=exports,
    )
    manifest = Manifest(
        bundle_id="device-runtime",
        abi_version=ABI_VERSION,
        ops=(op,),
    )
    driver = object()
    calls: list[tuple[object, ...]] = []
    closed: list[str] = []

    class FakeAdmission:
        digest = _digest("admission")

        def to_dict(self) -> dict[str, object]:
            return {
                "cubin_sha256": index.exports[0].cubin.sha256,
                "cubin_size": index.exports[0].cubin.size,
                "observed_abi_digest": _digest("abi"),
                "observed_contract_digest": _digest("contract"),
                "schema": "optima.device-artifact-admission.v1",
            }

    class FakeRuntime:
        admission = FakeAdmission()

        @classmethod
        def admit(cls, raw, plan, registry):
            assert raw == _cubin()
            assert plan == _device_plan()
            assert registry.driver is driver
            return cls()

        def __call__(self, *bindings):
            calls.append(bindings)

        def close(self):
            closed.append("runtime")

    class FakeDeviceEntry:
        def __init__(self, entry, runtimes):
            self.entry = entry
            self.runtimes = runtimes

        def __call__(self, *args, **kwargs):
            return self.entry(*args, **kwargs)

        def close(self):
            self.entry.close()
            for runtime in self.runtimes:
                runtime.close()

    provider = ArtifactRuntimeProvider(
        provider=CUTE_CUBIN_PROVIDER_NAME,
        tensor_descriptor=lambda value, _binding: value,
        tensor_pointer=lambda value, _binding: value,
        current_stream=lambda: "stream",
        group_rank=lambda _group: 0,
        group_size=lambda _group: 1,
        group_pointer=lambda *_args: 0,
        pointer_identity=lambda value, _binding: value,
    )
    registry = SimpleNamespace(
        driver=driver,
        capabilities=frozenset(
            {
                "cuda.checked_expression.v1",
                "cuda.packed_struct.v1",
                "cuda.tma_descriptor.v1",
                "cutlass.fast_divmod.i32.v1",
            }
        ),
    )
    monkeypatch.setattr(
        native_artifact,
        "reopen_native_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(root=tmp_path),
    )
    monkeypatch.setattr(cute_cubin, "reopen_cute_cubin_index", lambda *_a, **_k: index)
    monkeypatch.setattr(manifest_module, "load_manifest", lambda _bundle: manifest)
    monkeypatch.setattr(device_launch, "DeviceArtifactRuntime", FakeRuntime)
    monkeypatch.setattr(device_launch, "DeviceArtifactEntry", FakeDeviceEntry)
    monkeypatch.setattr(
        device_launch,
        "make_device_artifact_runtime_provider",
        lambda **_kwargs: provider,
    )
    monkeypatch.setattr(
        cuda_materialize,
        "make_cuda_primitive_registry",
        lambda **_kwargs: registry,
    )
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        current_device=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    written: list[tuple[str, dict[str, object], str | None]] = []
    monkeypatch.setattr(
        receipts,
        "write",
        lambda kind, payload, *, tag=None: written.append((kind, payload, tag)),
    )
    cute_cubin._RUNTIME_STATE = None

    prepare_cute_cubin_runtime(
        tmp_path,
        tmp_path,
        expected_publication_digest=_digest("publication"),
        expected_build_spec_digest=row["build_spec_digest"],  # type: ignore[arg-type]
        expected_tree_digest=row["tree_digest"],  # type: ignore[arg-type]
        expected_logical_architecture="sm103",
        expected_compile_profile_digest=row["compile_profile_digest"],  # type: ignore[arg-type]
        expected_patcher_sha256=row["patcher_sha256"],  # type: ignore[arg-type]
        driver=driver,
    )
    entry = resolve_direct_artifact_entry(op)
    assert entry is not None

    class Output:
        def fill_(self, _value):
            return self

    q, index_k, output = object(), object(), Output()
    entry(q, index_k, 7, 0.125, 128, output)
    shutdown_direct_artifact_runtimes()

    assert calls == [(q, index_k, output, 7, 0.125, "stream")]
    assert closed == ["runtime"]
    assert [kind for kind, _payload, _tag in written] == [
        "aot_loaded",
        "aot_invoked",
    ]
    assert written[0][1]["cubins"][0]["admission_sha256"] == _digest(  # type: ignore[index]
        "admission"
    )
    assert cute_cubin._RUNTIME_STATE is None


def test_device_publication_rejects_non_cubin_and_unreceipted_files(
    tmp_path: Path,
) -> None:
    row, index_path = _publication(tmp_path)
    cubin_path = tmp_path / row["exports"][0]["cubin"]["path"]  # type: ignore[index]
    raw = bytearray(cubin_path.read_bytes())
    raw[:4] = b"NOPE"
    cubin_path.write_bytes(raw)
    cubin_sha256 = hashlib.sha256(raw).hexdigest()
    export = row["exports"][0]  # type: ignore[index]
    export["cubin"]["sha256"] = cubin_sha256  # type: ignore[index]
    index_path.write_bytes(canonical_json_bytes(row) + b"\n")

    with pytest.raises(CuteCubinError, match="ELF gate"):
        reopen_cute_cubin_index(tmp_path, verify_distributions=False)

    valid = _cubin()
    cubin_path.write_bytes(valid)
    valid_sha256 = hashlib.sha256(valid).hexdigest()
    export["cubin"]["sha256"] = valid_sha256  # type: ignore[index]
    index_path.write_bytes(canonical_json_bytes(row) + b"\n")
    extra = cubin_path.parent / "candidate.ptx"
    extra.write_text("not authoritative")
    with pytest.raises(CuteCubinError, match="unreceipted files"):
        reopen_cute_cubin_index(tmp_path, verify_distributions=False)


def test_device_publication_rejects_launch_plan_semantic_tampering(
    tmp_path: Path,
) -> None:
    row, index_path = _publication(tmp_path)
    tampered = copy.deepcopy(row)
    plan = tampered["exports"][0]["launch_plan"]  # type: ignore[index]
    plan["bindings"] = [  # type: ignore[index]
        binding
        for binding in plan["bindings"]  # type: ignore[index]
        if binding["source"] != "output.block_scores"
    ]
    plan["prelaunch"] = [  # type: ignore[index]
        operation
        for operation in plan["prelaunch"]  # type: ignore[index]
        if operation["target"] != "output.block_scores"
    ]
    export = tampered["exports"][0]  # type: ignore[index]
    export["launch_plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(plan)
    ).hexdigest()
    # Re-hash every derived per-export identity so this reaches the independent
    # pipeline semantic gate instead of merely proving that artifact_id seals the
    # plan bytes.
    artifact_id, _prefix = deterministic_export_names(
        source=export["source"],
        slot=export["slot"],
        variant=export["variant"],
        name=export["name"],
        factory=export["factory"],
        plan=plan,
        artifact_resource_plan_sha256=export[
            "artifact_resource_plan_sha256"
        ],
        artifact_target_authority_sha256=export[
            "artifact_target_authority_sha256"
        ],
    )
    old_cubin = tmp_path / export["cubin"]["path"]
    new_cubin = (
        tmp_path
        / CUTE_CUBIN_STAGE_DIRECTORY
        / "cubins"
        / artifact_id
        / f"{artifact_id}.cubin"
    )
    new_cubin.parent.mkdir()
    old_directory = old_cubin.parent
    old_cubin.rename(new_cubin)
    old_directory.rmdir()
    export["artifact_id"] = artifact_id
    export["cubin"]["path"] = new_cubin.relative_to(tmp_path).as_posix()
    index_path.write_bytes(canonical_json_bytes(tampered) + b"\n")

    with pytest.raises(CuteCubinError, match="launch plan"):
        reopen_cute_cubin_index(tmp_path, verify_distributions=False)
