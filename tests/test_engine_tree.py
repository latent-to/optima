from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

import optima.engine_tree as engine_tree
from optima.bundle_hash import content_hash
from optima.engine_tree import (
    EngineTreeError,
    inspect_contribution,
    integrated_source_tree_digest,
    materialize_engine_tree,
    reopen_materialized_engine_tree,
)
from optima.manifest import load_manifest
from optima.sandbox import load_module
from optima.stack_manifest import (
    EngineReleaseManifest,
    EvaluationStackContext,
    EvaluationStackManifest,
    IntegratedContributionRef,
    ProposalContributionRef,
    ReleaseStackContext,
)
from optima.stack_plan import RollbackPlan, plan_candidate_stack, plan_marginal_arm
from optima.target_catalog import TargetCatalog, default_target_catalog


FIXTURES = Path(__file__).parent / "fixtures"
MSA = FIXTURES / "stack_msa_singleton"
FUSED = FIXTURES / "stack_fused_epilogue_atomic"
OVERRIDE = Path(__file__).parents[1] / "examples" / "miner_m3_swigluoai_override"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _spec_digests(catalog: TargetCatalog) -> dict[str, str]:
    targets = catalog.snapshot()["targets"]
    assert isinstance(targets, list)
    return {
        row["target_id"]: catalog.target_spec_digest(row["target_id"])
        for row in targets
    }


def _evaluation_context(catalog: TargetCatalog) -> EvaluationStackContext:
    return EvaluationStackContext(
        runtime_digest=_digest("runtime"),
        base_engine_digest=_digest("base"),
        arena_digest=_digest("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests=_spec_digests(catalog),
    )


def _proposal_ref(source: Path, catalog: TargetCatalog) -> ProposalContributionRef:
    inspected = inspect_contribution(source, catalog=catalog)
    return ProposalContributionRef(
        target_id=inspected.target_id,
        target_spec_digest=inspected.target_spec_digest,
        artifact_digest=content_hash(source),
        selected_payload_digest=inspected.selected_payload_digest,
        attribution_digest=_digest(f"attribution:{inspected.target_id}"),
    )


def _sources(
    *rows: tuple[ProposalContributionRef | IntegratedContributionRef, Path]
) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for ref, source in rows:
        key = (
            ("proposal", ref.artifact_digest)
            if isinstance(ref, ProposalContributionRef)
            else ("integrated", ref.integrated_source_tree_digest)
        )
        result[key] = source
    return result


def _evaluation_stack(
    catalog: TargetCatalog,
    context: EvaluationStackContext,
    *refs: ProposalContributionRef,
) -> EvaluationStackManifest:
    return EvaluationStackManifest(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        arena_digest=context.arena_digest,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={ref.target_id: ref for ref in refs},
    )


def _write_moe_fixture(root: Path, target: str, entry: str) -> Path:
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "fused_epilogue.py").write_text(
        "from kernels.helper import marker\n"
        "import fused_epilogue_sm103 as native\n\n"
        "def prepare(*args):\n"
        "    return marker\n\n"
        f"def {entry}(*args):\n"
        "    return native, marker\n"
    )
    (root / "kernels" / "helper.py").write_text(f"marker = {target!r}\n")
    (root / "kernels" / "fused_epilogue_sm103.cu").write_text(
        "#include <torch/extension.h>\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}\n"
    )
    (root / "rebuild.json").write_text(
        '{"steps":[{"type":"repo_python","path":"build_cuda_ext.py"}]}\n'
    )
    (root / "manifest.toml").write_text(
        f'bundle_id = "fixture-{entry}"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[competition]\n"
        f'target = "{target}"\n'
        'mode = "slot"\n\n'
        "[[ops]]\n"
        f'slot = "{target}"\n'
        'source = "kernels/fused_epilogue.py"\n'
        f'entry = "{entry}"\n'
        'prepare = "prepare"\n'
        'dtypes = ["bfloat16"]\n'
        'architectures = ["sm103"]\n'
        'cuda_sources = ["kernels/fused_epilogue_sm103.cu"]\n'
    )
    return root


def test_singleton_materialization_projects_metadata_and_reopens(tmp_path: Path) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(MSA, catalog)
    stack = _evaluation_stack(catalog, context, ref)

    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, MSA)),
        destination=tmp_path / "engine",
    )

    assert result.stack_digest == stack.digest
    assert result.runtime_manifest == "manifest.toml"
    assert reopen_materialized_engine_tree(
        result.root, expected_tree_digest=result.tree_digest
    ) == result
    manifest = load_manifest(result.root)
    assert manifest.bundle_id == "optima-materialized-v1"
    assert manifest.competition is None
    assert [op.slot for op in manifest.ops] == [ref.target_id]
    metadata = json.loads((result.root / manifest.ops[0].metadata).read_text())
    assert metadata["graph_safe"] is True
    assert "notes" not in metadata
    assert "regime" not in metadata
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in result.root.rglob("*") if path.is_file())


def test_multiple_variants_share_selected_source_without_order_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    wide_metadata = source / "metadata" / "blockscore_wide.json"
    wide = json.loads((source / "metadata" / "blockscore.json").read_text())
    wide["capabilities"]["block_size"] = {"exact": 256}
    wide_metadata.write_text(json.dumps(wide, indent=2) + "\n")
    with (source / "manifest.toml").open("a") as manifest:
        manifest.write(
            "\n[[ops]]\n"
            'slot = "attention.msa_prefill_block_score"\n'
            'variant = "wide"\n'
            'source = "kernels/blockscore.py"\n'
            'entry = "blockscore"\n'
            'dtypes = ["bfloat16"]\n'
            'architectures = ["sm103"]\n'
            'metadata = "metadata/blockscore_wide.json"\n'
        )
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(source, catalog)
    stack = _evaluation_stack(catalog, context, ref)
    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, source)),
        destination=tmp_path / "engine",
    )
    manifest = load_manifest(result.root)
    assert [op.variant for op in manifest.ops] == ["fixture", "wide"]
    assert manifest.ops[0].source == manifest.ops[1].source


def test_overlapping_variant_domains_reject_before_ref_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    with (source / "manifest.toml").open("a") as manifest:
        manifest.write(
            "\n[[ops]]\n"
            'slot = "attention.msa_prefill_block_score"\n'
            'variant = "overlap"\n'
            'source = "kernels/blockscore.py"\n'
            'entry = "blockscore"\n'
            'dtypes = ["bfloat16"]\n'
            'architectures = ["sm103"]\n'
            'metadata = "metadata/blockscore.json"\n'
        )
    with pytest.raises(EngineTreeError, match="overlapping capability domains"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_atomic_materialization_namespaces_native_patch_and_rebuild(
    tmp_path: Path,
) -> None:
    source_hash = content_hash(FUSED)
    source_modes = {
        path.relative_to(FUSED): path.stat().st_mode
        for path in FUSED.rglob("*")
        if path.is_file()
    }
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(FUSED, catalog)
    stack = _evaluation_stack(catalog, context, ref)

    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, FUSED)),
        destination=tmp_path / "engine",
    )

    manifest = load_manifest(result.root)
    assert {op.slot for op in manifest.ops} == {
        "collective.ar_residual_rmsnorm",
        "collective.moe_finalize_ar_rmsnorm",
    }
    assert len(manifest.dep_patches) == 1
    assert all(op.source.startswith("entries/optima_c_") for op in manifest.ops)
    assert all(op.cuda_sources[0].startswith("cuda/optima_c_") for op in manifest.ops)
    source = (result.root / manifest.ops[0].source).read_text()
    assert "from optima_c_" in source
    assert "import fused_epilogue_sm103" not in source
    assert json.loads((result.root / "rebuild.json").read_text()) == {
        "steps": [
            {
                "path": "optima/patchers/apply_dep_patch.py",
                "type": "repo_python",
            },
            {
                "path": "optima/patchers/build_cuda_ext.py",
                "type": "repo_python",
            },
        ]
    }
    assert content_hash(FUSED) == source_hash
    assert source_modes == {
        path.relative_to(FUSED): path.stat().st_mode
        for path in FUSED.rglob("*")
        if path.is_file()
    }


def test_stock_only_stack_has_no_runtime_bundle(tmp_path: Path) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    stack = _evaluation_stack(catalog, context)

    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver={},
        destination=tmp_path / "stock",
    )

    assert result.runtime_manifest is None
    assert not (result.root / "manifest.toml").exists()
    assert [row.path for row in result.files] == ["metadata/optima_engine_tree.json"]


def test_independent_contributions_compose_without_source_name_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _write_moe_fixture(
        tmp_path / "experts", "moe.fused_experts", "fused_experts"
    )
    reduce = _write_moe_fixture(
        tmp_path / "reduce", "moe.fused_experts_reduce", "fused_experts_reduce"
    )
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    experts_ref = _proposal_ref(experts, catalog)
    reduce_ref = _proposal_ref(reduce, catalog)
    stack = _evaluation_stack(catalog, context, experts_ref, reduce_ref)

    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((experts_ref, experts), (reduce_ref, reduce)),
        destination=tmp_path / "engine",
    )

    manifest = load_manifest(result.root)
    assert [op.slot for op in manifest.ops] == [
        "moe.fused_experts_reduce",
        "moe.fused_experts",
    ]
    assert manifest.ops[0].source != manifest.ops[1].source
    assert Path(manifest.ops[0].source).stem != Path(manifest.ops[1].source).stem
    assert len(set(result.root.glob("optima_c_*/kernels/fused_epilogue.py"))) == 2
    assert len(set(result.root.glob("optima_c_*/kernels/helper.py"))) == 2
    assert len(
        set(result.root.glob("cuda/optima_c_*/kernels/optima_c_*__fused_epilogue_sm103_*.cu"))
    ) == 2
    for op in manifest.ops:
        shim = (result.root / op.source).read_text()
        assert "from optima_c_" in shim
        assert "from kernels.helper" not in shim
        assert "import fused_epilogue_sm103" not in shim
    for emitted_path in result.root.glob("optima_c_*/kernels/fused_epilogue.py"):
        emitted = emitted_path.read_text()
        assert "from optima_c_" in emitted
        assert "from kernels.helper" not in emitted
        assert "import optima_c_" in emitted
        assert "import fused_epilogue_sm103" not in emitted

    monkeypatch.syspath_prepend(str(result.root))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    before_modules = set(sys.modules)
    loaded = []
    try:
        for op in manifest.ops:
            native_name = Path(op.cuda_sources[0]).stem
            monkeypatch.setitem(sys.modules, native_name, ModuleType(native_name))
            module = load_module(result.root / op.source)
            loaded.append(module)
            _native, marker = getattr(module, op.entry)()
            assert marker == op.slot
            assert sys.modules[module.__name__] is module
        assert loaded[0].__name__ != loaded[1].__name__
        assert getattr(loaded[0], manifest.ops[0].entry).__module__ != getattr(
            loaded[1], manifest.ops[1].entry
        ).__module__
    finally:
        for name in set(sys.modules) - before_modules:
            if name.startswith(("optima_c_", "optima_kernel_optima_c_")):
                sys.modules.pop(name, None)


def test_override_entry_shim_preserves_required_ref_and_optional_device_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(OVERRIDE, catalog)
    stack = _evaluation_stack(catalog, context, ref)
    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, OVERRIDE)),
        destination=tmp_path / "engine",
    )
    op = load_manifest(result.root).ops[0]
    shim = (result.root / op.source).read_text()
    assert f"import {op.entry}_ref as {op.entry}_ref" in shim
    assert "try:" in shim and f"import {op.entry} as {op.entry}" in shim

    monkeypatch.syspath_prepend(str(result.root))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    # The SGLang/CUDA validation image installs CuTeDSL even when no GPU is
    # exposed. Make this specifically the portable-reference branch that the
    # test names, independent of ambient toolchain packages.
    monkeypatch.setitem(sys.modules, "cutlass", None)
    monkeypatch.setitem(sys.modules, "cutlass.cute", None)
    before_modules = set(sys.modules)
    try:
        module = load_module(result.root / op.source)
        assert callable(getattr(module, op.entry + "_ref"))
        assert getattr(module, op.entry, None) is None
        from optima_kernels.override import build_override

        entry, prepare = build_override(
            op.slot,
            op.override_point,
            op.entry,
            lambda name: getattr(module, name, None),
        )
        assert callable(entry) and callable(prepare)
    finally:
        for name in set(sys.modules) - before_modules:
            if name.startswith(("optima_c_", "optima_kernel_optima_c_")):
                sys.modules.pop(name, None)


def test_integrated_release_revalidates_reviewed_source(tmp_path: Path) -> None:
    catalog = default_target_catalog()
    inspected = inspect_contribution(MSA, catalog=catalog)
    ref = IntegratedContributionRef(
        target_id=inspected.target_id,
        target_spec_digest=inspected.target_spec_digest,
        integrated_source_tree_digest=integrated_source_tree_digest(MSA),
        selected_payload_digest=inspected.selected_payload_digest,
        attribution_digest=_digest("attribution"),
        integration_record_digest=_digest("integration-record"),
    )
    context = ReleaseStackContext(
        runtime_digest=_digest("runtime"),
        base_engine_digest=_digest("base"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests=_spec_digests(catalog),
    )
    release = EngineReleaseManifest(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={ref.target_id: ref},
    )

    result = materialize_engine_tree(
        release,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, MSA)),
        destination=tmp_path / "release",
    )
    assert result.stack_digest == release.digest

    padded = tmp_path / "padded"
    shutil.copytree(MSA, padded)
    (padded / "padding.txt").write_text("reviewed source changed\n")
    with pytest.raises(EngineTreeError, match="integrated source digest mismatch"):
        materialize_engine_tree(
            release,
            context=context,
            catalog=catalog,
            resolver=_sources((ref, padded)),
            destination=tmp_path / "wrong-source",
        )

    wrong_payload = replace(ref, selected_payload_digest=_digest("wrong-payload"))
    wrong_release = EngineReleaseManifest(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={wrong_payload.target_id: wrong_payload},
    )
    with pytest.raises(EngineTreeError, match="selected payload digest mismatch"):
        materialize_engine_tree(
            wrong_release,
            context=context,
            catalog=catalog,
            resolver=_sources((wrong_payload, MSA)),
            destination=tmp_path / "wrong-payload",
        )


def test_inert_padding_changes_artifact_not_selected_payload(tmp_path: Path) -> None:
    padded = tmp_path / "padded"
    shutil.copytree(MSA, padded)
    (padded / "README.txt").write_text("not selected by the target\n")
    catalog = default_target_catalog()

    plain = inspect_contribution(MSA, catalog=catalog)
    extra = inspect_contribution(padded, catalog=catalog)

    assert content_hash(MSA) != content_hash(padded)
    assert plain.selected_payload_digest == extra.selected_payload_digest
    assert plain.selected_delta_digest == extra.selected_delta_digest
    assert _proposal_ref(MSA, catalog).digest != _proposal_ref(padded, catalog).digest
    context = _evaluation_context(catalog)
    ref = _proposal_ref(padded, catalog)
    result = materialize_engine_tree(
        _evaluation_stack(catalog, context, ref),
        context=context,
        catalog=catalog,
        resolver=_sources((ref, padded)),
        destination=tmp_path / "engine",
    )
    assert not any("README" in row.path for row in result.files)


def test_imported_local_inputs_enter_selected_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    catalog = default_target_catalog()
    before = inspect_contribution(source, catalog=catalog)
    kernel = source / "kernels" / "blockscore.py"
    kernel.write_text("from kernels.helper import scale\n" + kernel.read_text())
    (source / "kernels" / "helper.py").write_text("scale = 1\n")

    after = inspect_contribution(source, catalog=catalog)

    assert after.python_files == (
        "kernels/blockscore.py",
        "kernels/helper.py",
    )
    assert after.selected_payload_digest != before.selected_payload_digest


def test_native_from_import_is_rewritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    kernel = source / "kernels" / "fused_epilogue.py"
    kernel.write_text(
        "from fused_epilogue_sm103 import ar_residual_rmsnorm as native_ar\n"
        + kernel.read_text()
    )
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(source, catalog)
    stack = _evaluation_stack(catalog, context, ref)

    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, source)),
        destination=tmp_path / "engine",
    )

    manifest = load_manifest(result.root)
    emitted = (result.root / manifest.ops[0].source).read_text()
    assert "from optima_c_" in emitted
    assert "from fused_epilogue_sm103" not in emitted


@pytest.mark.parametrize(
    "source_text",
    [
        "module = __import__(\"kernels.helper\")\n",
        "import importlib\nmodule = importlib.import_module(\"kernels.helper\")\n",
        "import importlib as il\nmodule = il.import_module(\"kernels.helper\")\n",
        "from importlib import import_module\nmodule = import_module(\"kernels.helper\")\n",
        "import builtins\nmodule = builtins.__import__(\"kernels.helper\")\n",
        "from builtins import __import__ as imp\nmodule = imp(\"kernels.helper\")\n",
        "exec(\"import kernels.helper\")\n",
        "code = compile(\"import kernels.helper\", \"<miner>\", \"exec\")\n",
    ],
)
def test_dynamic_imports_fail_closed(tmp_path: Path, source_text: str) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    (source / "kernels" / "blockscore.py").write_text(source_text)
    with pytest.raises(EngineTreeError, match="dynamic import"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_package_imports_are_closed_rewritten_and_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    package = source / "kernels" / "pkg"
    package.mkdir()
    (source / "kernels" / "__init__.py").write_text("PARENT = 3\n")
    (package / "__init__.py").write_text("PACKAGE = True\n")
    (package / "helper.py").write_text("VALUE = 7\n")
    # CPython resolves the package before this same-name module.
    (source / "kernels" / "pkg.py").write_text("VALUE = -1\n")
    (source / "kernels" / "blockscore.py").write_text(
        "if True:\n"
        "    from .pkg import helper\n"
        "    import kernels.pkg.helper\n"
        "VALUE = helper.VALUE + kernels.pkg.helper.VALUE\n\n"
        "def blockscore(q, k, out):\n"
        "    return VALUE\n"
    )
    catalog = default_target_catalog()
    inspected = inspect_contribution(source, catalog=catalog)
    assert "kernels/pkg/__init__.py" in inspected.python_files
    assert "kernels/__init__.py" in inspected.python_files
    assert "kernels/pkg/helper.py" in inspected.python_files
    assert "kernels/pkg.py" not in inspected.python_files
    context = _evaluation_context(catalog)
    ref = _proposal_ref(source, catalog)
    stack = _evaluation_stack(catalog, context, ref)
    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, source)),
        destination=tmp_path / "engine",
    )
    manifest = load_manifest(result.root)
    emitted = (result.root / manifest.ops[0].source).read_text()
    compile(emitted, manifest.ops[0].source, "exec")
    assert "from optima_c_" in emitted
    implementation = next(result.root.glob("optima_c_*/kernels/blockscore.py")).read_text()
    assert "import optima_c_" in implementation
    monkeypatch.syspath_prepend(str(result.root))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    module_name = manifest.ops[0].source.removesuffix(".py").replace("/", ".")
    module = importlib.import_module(module_name)
    entry = getattr(module, manifest.ops[0].entry)
    assert entry(None, None, None) == 14
    namespace = entry.__module__.split(".", 1)[0]
    assert importlib.import_module(namespace + ".kernels").PARENT == 3
    reopen_materialized_engine_tree(
        result.root, expected_tree_digest=result.tree_digest
    )


@pytest.mark.parametrize(
    "source_text,message",
    [
        ("from .missing import value\n", "unresolved relative"),
        ("from kernels import helper, missing\n", "partially local"),
        ("import kernels.missing\n", "partially local"),
        ("from kernels.missing import value\n", "partially local"),
        ("from . import *\n", "unresolved relative"),
    ],
)
def test_unresolved_or_partial_local_imports_fail_closed(
    tmp_path: Path, source_text: str, message: str
) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    (source / "kernels" / "helper.py").write_text("value = 1\n")
    (source / "kernels" / "blockscore.py").write_text(source_text)
    with pytest.raises(EngineTreeError, match=message):
        inspect_contribution(source, catalog=default_target_catalog())


@pytest.mark.parametrize(
    "source_text",
    (
        "import fused_epilogue_sm103.missing\n",
        "from fused_epilogue_sm103.missing import value\n",
    ),
)
def test_partial_declared_native_import_fails_closed(
    tmp_path: Path, source_text: str,
) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    (source / "kernels" / "fused_epilogue.py").write_text(
        source_text
        + "def ar_residual_rmsnorm(*args):\n    return None\n"
        + "def moe_finalize_ar_rmsnorm(*args):\n    return None\n"
    )
    with pytest.raises(EngineTreeError, match="partially local"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_bare_namespace_and_nonidentifier_module_paths_fail_closed(tmp_path: Path) -> None:
    namespace = tmp_path / "namespace"
    shutil.copytree(MSA, namespace)
    (namespace / "kernels" / "blockscore.py").write_text("import kernels\n")
    with pytest.raises(EngineTreeError, match="bare local namespace"):
        inspect_contribution(namespace, catalog=default_target_catalog())

    invalid = tmp_path / "invalid"
    shutil.copytree(MSA, invalid)
    (invalid / "kernels-v2").mkdir()
    shutil.copy2(
        invalid / "kernels" / "blockscore.py",
        invalid / "kernels-v2" / "blockscore.py",
    )
    manifest = invalid / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace("kernels/blockscore.py", "kernels-v2/blockscore.py")
    )
    with pytest.raises(EngineTreeError, match="non-identifier component"):
        inspect_contribution(invalid, catalog=default_target_catalog())


def test_python_native_name_collision_fails_during_inspection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    (source / "fused_epilogue_sm103.py").write_text("collision = True\n")
    with pytest.raises(EngineTreeError, match="both local Python and declared native"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_ambiguous_declared_native_stems_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    duplicate = source / "other" / "fused_epilogue_sm103.cu"
    duplicate.parent.mkdir()
    shutil.copy2(source / "kernels" / "fused_epilogue_sm103.cu", duplicate)
    manifest = source / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace(
            'cuda_sources = ["kernels/fused_epilogue_sm103.cu"]',
            'cuda_sources = ["kernels/fused_epilogue_sm103.cu", '
            '"other/fused_epilogue_sm103.cu"]',
        )
    )
    with pytest.raises(EngineTreeError, match="ambiguous native module stem"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_nonregular_source_tree_entries_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    fifo = source / "host-pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError):
        pytest.skip("FIFO creation unavailable")
    with pytest.raises(EngineTreeError, match="nonregular"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_declared_cuda_headers_enter_identity_and_undeclared_headers_reject(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    cuda = source / "kernels" / "fused_epilogue_sm103.cu"
    cuda.write_text('#include "helper.cuh" /* selected */\n' + cuda.read_text())
    (source / "kernels" / "helper.cuh").write_text("#define HELPER 1\n")
    with pytest.raises(EngineTreeError, match="undeclared local input"):
        inspect_contribution(source, catalog=default_target_catalog())

    manifest = source / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace(
            'cuda_sources = ["kernels/fused_epilogue_sm103.cu"]',
            'cuda_sources = ["kernels/fused_epilogue_sm103.cu", "kernels/helper.cuh"]',
        )
    )
    inspected = inspect_contribution(source, catalog=default_target_catalog())
    assert "kernels/helper.cuh" in inspected.cuda_files


def test_dynamic_cuda_include_directives_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    cuda = source / "kernels" / "fused_epilogue_sm103.cu"
    cuda.write_text('#define HEADER "unbound.cuh"\n#include HEADER\n' + cuda.read_text())
    with pytest.raises(EngineTreeError, match="dynamic include"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_literal_cuda_includes_allow_comment_only_suffixes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    cuda = source / "kernels" / "fused_epilogue_sm103.cu"
    cuda.write_text(
        cuda.read_text().replace(
            "#include <torch/extension.h>",
            "#include <torch/extension.h> // pinned toolchain header",
        )
    )
    inspect_contribution(source, catalog=default_target_catalog())


@pytest.mark.parametrize("header", ["/tmp/unbound.cuh", "../../unbound.cuh"])
def test_unsafe_system_cuda_includes_fail_closed(tmp_path: Path, header: str) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    cuda = source / "kernels" / "fused_epilogue_sm103.cu"
    cuda.write_text(f"#include <{header}>\n" + cuda.read_text())
    with pytest.raises(EngineTreeError, match="unsafe system include"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_missing_quoted_cuda_include_cannot_escape_dependency_roots(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    cuda = source / "kernels" / "fused_epilogue_sm103.cu"
    cuda.write_text('#include "../unbound.cuh"\n' + cuda.read_text())
    with pytest.raises(EngineTreeError, match="unsafe dependency include"):
        inspect_contribution(source, catalog=default_target_catalog())


@pytest.mark.parametrize(
    "directive",
    [
        '#include_next "/tmp/unbound.cuh"',
        '%:include "/tmp/unbound.cuh"',
    ],
)
def test_alternate_cuda_include_directives_fail_closed(
    tmp_path: Path, directive: str
) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    cuda = source / "kernels" / "fused_epilogue_sm103.cu"
    cuda.write_text(directive + "\n" + cuda.read_text())
    with pytest.raises(EngineTreeError, match="unsupported include"):
        inspect_contribution(source, catalog=default_target_catalog())


def test_line_spliced_cuda_include_is_still_validated(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    cuda = source / "kernels" / "fused_epilogue_sm103.cu"
    cuda.write_text('#inc\\\nlude "/tmp/unbound.cuh"\n' + cuda.read_text())
    with pytest.raises(EngineTreeError, match="safe relative path"):
        inspect_contribution(source, catalog=default_target_catalog())


@pytest.mark.parametrize(
    "selected",
    [".git/blockscore.py", "kernels/._blockscore.py", "kernels/blockscore.pyc"],
)
def test_bundle_hash_excluded_paths_cannot_be_selected(
    tmp_path: Path, selected: str
) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    selected_path = source / selected
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text((source / "kernels" / "blockscore.py").read_text())
    manifest = source / "manifest.toml"
    manifest.write_text(manifest.read_text().replace("kernels/blockscore.py", selected))
    with pytest.raises(ValueError):
        inspect_contribution(source, catalog=default_target_catalog())


def test_root_source_symlink_is_rejected(tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(MSA, target_is_directory=True)
    with pytest.raises(EngineTreeError, match="must not be a symlink"):
        inspect_contribution(alias, catalog=default_target_catalog())


def test_materialization_is_location_mode_and_umask_independent(tmp_path: Path) -> None:
    left = tmp_path / "left-source"
    right = tmp_path / "right-source"
    shutil.copytree(MSA, left)
    shutil.copytree(MSA, right)
    os.chmod(left / "kernels" / "blockscore.py", 0o600)
    os.chmod(right / "kernels" / "blockscore.py", 0o755)
    assert integrated_source_tree_digest(left) == integrated_source_tree_digest(right)

    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    left_ref = _proposal_ref(left, catalog)
    right_ref = _proposal_ref(right, catalog)
    assert left_ref == right_ref
    stack = _evaluation_stack(catalog, context, left_ref)
    previous = os.umask(0o077)
    try:
        left_tree = materialize_engine_tree(
            stack,
            context=context,
            catalog=catalog,
            resolver=_sources((left_ref, left)),
            destination=tmp_path / "left-engine",
        )
    finally:
        os.umask(previous)
    previous = os.umask(0o002)
    try:
        right_tree = materialize_engine_tree(
            stack,
            context=context,
            catalog=catalog,
            resolver=_sources((right_ref, right)),
            destination=tmp_path / "right-engine",
        )
    finally:
        os.umask(previous)

    assert left_tree.tree_digest == right_tree.tree_digest
    assert [row.identity_data() for row in left_tree.files] == [
        row.identity_data() for row in right_tree.files
    ]


def test_semantic_aliases_and_set_order_share_selected_identity(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    aliases = tmp_path / "aliases"
    shutil.copytree(MSA, canonical)
    shutil.copytree(MSA, aliases)
    manifest = aliases / "manifest.toml"
    manifest.write_text(
        manifest.read_text()
        .replace('dtypes = ["bfloat16"]', 'dtypes = ["bfloat16", "bf16"]')
        .replace('architectures = ["sm103"]', 'architectures = ["sm_103", "sm103"]')
    )
    metadata_path = aliases / "metadata" / "blockscore.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dtypes"] = ["bf16", "bfloat16"]
    metadata["architectures"] = ["sm_103", "sm103"]
    metadata["capabilities"] = {
        field: {"one_of": [spec["exact"], spec["exact"]]}
        for field, spec in reversed(tuple(metadata["capabilities"].items()))
    }
    metadata_path.write_text(json.dumps(metadata, indent=4) + "\n")

    left = inspect_contribution(canonical, catalog=default_target_catalog())
    right = inspect_contribution(aliases, catalog=default_target_catalog())
    assert left.selected_payload_digest == right.selected_payload_digest
    assert left.selected_delta_digest == right.selected_delta_digest


def test_packaging_order_ids_and_json_whitespace_do_not_choose_namespace(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    reordered = tmp_path / "reordered"
    shutil.copytree(FUSED, canonical)
    shutil.copytree(FUSED, reordered)
    manifest = reordered / "manifest.toml"
    prefix, first_op, second_op = manifest.read_text().split("[[ops]]")
    manifest.write_text(
        (prefix + "[[ops]]" + second_op + "[[ops]]" + first_op)
        .replace("fixture-fused-epilogue-atomic", "ignored-packaging-id")
    )
    rebuild = json.loads((reordered / "rebuild.json").read_text())
    rebuild["steps"].reverse()
    (reordered / "rebuild.json").write_text(json.dumps(rebuild, separators=(",", ":")))
    for metadata_path in (reordered / "metadata").glob("*.json"):
        metadata = json.loads(metadata_path.read_text())
        metadata_path.write_text(json.dumps(metadata, indent=6) + "\n")

    left = inspect_contribution(canonical, catalog=default_target_catalog())
    right = inspect_contribution(reordered, catalog=default_target_catalog())
    assert content_hash(canonical) != content_hash(reordered)
    assert left.selected_payload_digest == right.selected_payload_digest
    assert left.selected_delta_digest == right.selected_delta_digest
    assert f"optima_c_{left.selected_delta_digest}" == (
        f"optima_c_{right.selected_delta_digest}"
    )


@pytest.mark.parametrize(
    "input_class",
    ["op", "metadata", "python", "cuda", "header", "patch"],
)
def test_every_selected_executable_input_class_rotates_delta(
    tmp_path: Path, input_class: str
) -> None:
    source = tmp_path / "source"
    shutil.copytree(FUSED, source)
    if input_class == "header":
        header = source / "kernels" / "helper.cuh"
        header.write_text("#define VALUE 1\n")
        cuda = source / "kernels" / "fused_epilogue_sm103.cu"
        cuda.write_text('#include "helper.cuh"\n' + cuda.read_text())
        manifest = source / "manifest.toml"
        manifest.write_text(
            manifest.read_text().replace(
                'cuda_sources = ["kernels/fused_epilogue_sm103.cu"]',
                'cuda_sources = ["kernels/fused_epilogue_sm103.cu", '
                '"kernels/helper.cuh"]',
            )
        )
    before = inspect_contribution(source, catalog=default_target_catalog())

    if input_class == "op":
        manifest = source / "manifest.toml"
        manifest.write_text(
            manifest.read_text().replace(
                'entry = "ar_residual_rmsnorm"',
                'entry = "ar_residual_rmsnorm_v2"',
                1,
            )
        )
    elif input_class == "metadata":
        path = source / "metadata" / "ar_norm.json"
        metadata = json.loads(path.read_text())
        metadata["max_num_tokens"] = 999
        path.write_text(json.dumps(metadata))
    elif input_class == "python":
        path = source / "kernels" / "fused_epilogue.py"
        path.write_text(path.read_text() + "\n# selected source revision\n")
    elif input_class == "cuda":
        path = source / "kernels" / "fused_epilogue_sm103.cu"
        path.write_text(path.read_text() + "\n// selected CUDA revision\n")
    elif input_class == "header":
        path = source / "kernels" / "helper.cuh"
        path.write_text("#define VALUE 2\n")
    else:
        path = source / "patches" / "flashinfer.patch"
        path.write_text(path.read_text().replace("export_prefinalize", "export_prefinalize_v2"))

    after = inspect_contribution(source, catalog=default_target_catalog())
    assert after.selected_payload_digest != before.selected_payload_digest
    assert after.selected_delta_digest != before.selected_delta_digest


def test_registered_patcher_source_identity_rotates_selected_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    patchers = repo / "optima" / "patchers"
    patchers.mkdir(parents=True)
    real_patchers = Path(engine_tree.__file__).parent / "patchers"
    for name in ("apply_dep_patch.py", "build_cuda_ext.py"):
        shutil.copy2(real_patchers / name, patchers / name)
    monkeypatch.setenv("OPTIMA_REPO_ROOT", str(repo))
    before = inspect_contribution(FUSED, catalog=default_target_catalog())
    builder = patchers / "build_cuda_ext.py"
    builder.write_text(builder.read_text() + "\n# reviewed patcher revision\n")
    after = inspect_contribution(FUSED, catalog=default_target_catalog())
    assert after.selected_delta_digest != before.selected_delta_digest


def test_source_mutation_cannot_diverge_identity_from_emitted_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(source, catalog)
    stack = _evaluation_stack(catalog, context, ref)
    stable_read = engine_tree._stable_read
    changed = False

    def racing_read(root: Path, relative: str) -> bytes:
        nonlocal changed
        data = stable_read(root, relative)
        if root == source.resolve() and relative == "kernels/blockscore.py" and not changed:
            (source / relative).write_text("def blockscore(*args):\n    return None\n")
            changed = True
        return data

    monkeypatch.setattr(engine_tree, "_stable_read", racing_read)
    with pytest.raises(EngineTreeError, match="changed"):
        materialize_engine_tree(
            stack,
            context=context,
            catalog=catalog,
            resolver=_sources((ref, source)),
            destination=tmp_path / "engine",
        )
    assert not (tmp_path / "engine").exists()


def test_reopen_rejects_root_mode_extra_directories_and_root_symlinks(
    tmp_path: Path,
) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(MSA, catalog)
    stack = _evaluation_stack(catalog, context, ref)
    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, MSA)),
        destination=tmp_path / "engine",
    )
    metadata = json.loads((result.root / "metadata/optima_engine_tree.json").read_text())
    assert metadata["contributions"][0]["namespace"] == (
        "optima_c_" + ref.selected_delta_digest
    )
    with pytest.raises(EngineTreeError, match="tree digest mismatch"):
        reopen_materialized_engine_tree(
            result.root, expected_tree_digest=_digest("wrong-tree-receipt")
        )

    ghost = result.root / "ghost"
    ghost.mkdir()
    with pytest.raises(EngineTreeError, match="directory inventory"):
        reopen_materialized_engine_tree(result.root, expected_tree_digest=result.tree_digest)
    ghost.rmdir()

    os.chmod(result.root, 0o700)
    try:
        with pytest.raises(EngineTreeError, match="root directory mode"):
            reopen_materialized_engine_tree(result.root)
    finally:
        os.chmod(result.root, 0o755)

    alias = tmp_path / "engine-link"
    alias.symlink_to(result.root, target_is_directory=True)
    with pytest.raises(EngineTreeError, match="must not be a symlink"):
        reopen_materialized_engine_tree(alias)

    metadata_path = result.root / "metadata/optima_engine_tree.json"
    os.chmod(metadata_path, 0o644)
    metadata["contributions"][0]["namespace"] = "optima_c_" + "0" * 64
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.chmod(metadata_path, 0o444)
    with pytest.raises(EngineTreeError, match="namespace mismatch"):
        reopen_materialized_engine_tree(result.root)


def test_failed_preinstall_verification_leaves_no_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(MSA, catalog)
    stack = _evaluation_stack(catalog, context, ref)
    reopen = engine_tree.reopen_materialized_engine_tree

    def fail_temp(root: str | Path, *, expected_tree_digest: str | None = None):
        if Path(root).name.startswith(".engine."):
            raise EngineTreeError("forced preinstall verification failure")
        return reopen(root, expected_tree_digest=expected_tree_digest)

    monkeypatch.setattr(engine_tree, "reopen_materialized_engine_tree", fail_temp)
    destination = tmp_path / "engine"
    with pytest.raises(EngineTreeError, match="forced preinstall"):
        materialize_engine_tree(
            stack,
            context=context,
            catalog=catalog,
            resolver=_sources((ref, MSA)),
            destination=destination,
        )
    assert not destination.exists()


def test_runtime_rebuild_order_is_global_not_contribution_order() -> None:
    raw = engine_tree._runtime_rebuild(
        [
            {"type": "repo_python", "path": "optima/patchers/build_cuda_ext.py"},
            {"type": "repo_python", "path": "optima/patchers/apply_dep_patch.py"},
        ]
    )
    assert raw is not None
    assert [row["path"] for row in json.loads(raw)["steps"]] == [
        "optima/patchers/apply_dep_patch.py",
        "optima/patchers/build_cuda_ext.py",
    ]


def test_dependency_patch_destinations_cannot_overlap_by_order() -> None:
    inspected = inspect_contribution(FUSED, catalog=default_target_catalog())
    destinations: set[tuple[str, str]] = set()
    engine_tree._contribution_files(
        inspected,
        delta_digest=inspected.selected_delta_digest,
        patch_destinations=destinations,
    )
    with pytest.raises(EngineTreeError, match="patch destination collision"):
        engine_tree._contribution_files(
            inspected,
            delta_digest=_digest("other-delta"),
            patch_destinations=destinations,
        )


@pytest.mark.parametrize("fixture", [MSA, FUSED], ids=["msa", "atomic-fused"])
def test_fixture_materialization_binds_marginal_arm_and_exact_rollback(
    tmp_path: Path, fixture: Path,
) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    incumbent = _evaluation_stack(catalog, context)
    baseline = materialize_engine_tree(
        incumbent,
        context=context,
        catalog=catalog,
        resolver={},
        destination=tmp_path / "baseline",
    )
    ref = _proposal_ref(fixture, catalog)
    candidate = plan_candidate_stack(
        incumbent,
        ref,
        catalog=catalog,
        expected_context=context,
    )
    challenger = materialize_engine_tree(
        candidate,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, fixture)),
        destination=tmp_path / "challenger",
    )
    arm = plan_marginal_arm(
        incumbent,
        ref,
        catalog=catalog,
        incumbent_tree_digest=baseline.tree_digest,
        candidate_tree_digest=challenger.tree_digest,
        expected_context=context,
    )
    assert arm.candidate == candidate
    assert arm.baseline_before == arm.baseline_after
    assert arm.baseline_before is not arm.baseline_after
    rollback = RollbackPlan.from_arm(
        arm, catalog=catalog, expected_context=context
    )
    restored, restored_tree = rollback.reconstruct(
        candidate,
        tree_digest=challenger.tree_digest,
        source_arm=arm,
        catalog=catalog,
        expected_context=context,
    )
    assert restored == incumbent
    assert restored_tree == baseline.tree_digest


def test_source_and_materialized_symlinks_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    (source / "padding-link").symlink_to(source / "manifest.toml")
    with pytest.raises(EngineTreeError, match="symlink"):
        inspect_contribution(source, catalog=default_target_catalog())

    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(MSA, catalog)
    stack = _evaluation_stack(catalog, context, ref)
    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, MSA)),
        destination=tmp_path / "engine",
    )
    link = result.root / "link"
    link.symlink_to(result.root / "manifest.toml")
    with pytest.raises(EngineTreeError, match="symlink"):
        reopen_materialized_engine_tree(result.root)


def test_wrong_source_identity_and_post_write_tampering_fail_closed(tmp_path: Path) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(MSA, catalog)
    stack = _evaluation_stack(catalog, context, ref)
    padded = tmp_path / "padded"
    shutil.copytree(MSA, padded)
    (padded / "padding.txt").write_text("changes the proposal artifact")

    with pytest.raises(EngineTreeError, match="artifact digest mismatch"):
        materialize_engine_tree(
            stack,
            context=context,
            catalog=catalog,
            resolver=_sources((ref, padded)),
            destination=tmp_path / "wrong",
        )

    wrong_payload = replace(ref, selected_payload_digest=_digest("wrong-payload"))
    wrong_stack = _evaluation_stack(catalog, context, wrong_payload)
    with pytest.raises(EngineTreeError, match="selected payload digest mismatch"):
        materialize_engine_tree(
            wrong_stack,
            context=context,
            catalog=catalog,
            resolver=_sources((wrong_payload, MSA)),
            destination=tmp_path / "wrong-payload",
        )

    result = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver=_sources((ref, MSA)),
        destination=tmp_path / "engine",
    )
    kernel = next(result.root.glob("optima_c_*/kernels/*.py"))
    os.chmod(kernel, 0o644)
    kernel.write_text(kernel.read_text() + "\n# tampered\n")
    os.chmod(kernel, 0o444)
    with pytest.raises(EngineTreeError, match="inventory mismatch"):
        reopen_materialized_engine_tree(result.root)


def test_destination_and_context_are_fail_closed(tmp_path: Path) -> None:
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(MSA, catalog)
    stack = _evaluation_stack(catalog, context, ref)
    destination = tmp_path / "already-there"
    destination.mkdir()
    with pytest.raises(EngineTreeError, match="already exists"):
        materialize_engine_tree(
            stack,
            context=context,
            catalog=catalog,
            resolver=_sources((ref, MSA)),
            destination=destination,
        )

    wrong_context = EvaluationStackContext(
        runtime_digest=_digest("wrong-runtime"),
        base_engine_digest=context.base_engine_digest,
        arena_digest=context.arena_digest,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests=_spec_digests(catalog),
    )
    with pytest.raises(ValueError, match="runtime digest"):
        materialize_engine_tree(
            stack,
            context=wrong_context,
            catalog=catalog,
            resolver=_sources((ref, MSA)),
            destination=tmp_path / "stale",
        )


def test_destination_cannot_mutate_a_resolved_contribution_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(MSA, source)
    before = content_hash(source)
    catalog = default_target_catalog()
    context = _evaluation_context(catalog)
    ref = _proposal_ref(source, catalog)
    stack = _evaluation_stack(catalog, context, ref)

    with pytest.raises(EngineTreeError, match="outside contribution source"):
        materialize_engine_tree(
            stack,
            context=context,
            catalog=catalog,
            resolver=_sources((ref, source)),
            destination=source / "emitted-engine",
        )
    assert content_hash(source) == before
    assert not (source / "emitted-engine").exists()
