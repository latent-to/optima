"""Hermetic build and read-only load invariants for declared CUDA products."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest


def _digest(label: str) -> str:
    return __import__("hashlib").sha256(label.encode()).hexdigest()


def _patcher():
    return importlib.import_module("optima.patchers.build_cuda_ext")


def _bundle(
    root: Path,
    *,
    marker: Path | None = None,
    header: str = "#define VALUE 1\n",
    two_units: bool = False,
) -> Path:
    kernels = root / "kernels"
    kernels.mkdir(parents=True)
    body = "def entry(x, out):\n    out.copy_(x)\n"
    if marker is not None:
        body = f"open({str(marker)!r}, 'w').close()\n" + body
    (kernels / "shim.py").write_text(body)
    (kernels / "native.cu").write_text('// unit\n#include "values.cuh"\n')
    (kernels / "values.cuh").write_text(header)
    sources = '"kernels/native.cu", "kernels/values.cuh"'
    if two_units:
        (kernels / "second.cu").write_text('// second\n#include "values.cuh"\n')
        sources += ', "kernels/second.cu"'
    (root / "manifest.toml").write_text(
        'bundle_id = "miner-controlled-display-name"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        '[[ops]]\n'
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/shim.py"\n'
        'entry = "entry"\n'
        f"cuda_sources = [{sources}]\n"
    )
    return root


def _architecture_variant_bundle(root: Path) -> tuple[Path, dict[str, Path]]:
    markers: dict[str, Path] = {}
    rows: list[str] = []
    for architecture in ("sm103", "sm120"):
        kernel_root = root / "kernels" / architecture
        kernel_root.mkdir(parents=True)
        marker = root / f"{architecture}-python-ran"
        markers[architecture] = marker
        (kernel_root / "shim.py").write_text(
            f"open({str(marker)!r}, 'w').close()\n"
            "def entry(x, out):\n    out.copy_(x)\n"
        )
        (kernel_root / "native.cu").write_text(
            f"// {architecture} unit\n#include \"values.cuh\"\n"
        )
        (kernel_root / "values.cuh").write_text(
            f"#define OPTIMA_VARIANT_{architecture.upper()} 1\n"
        )
        rows.extend(
            (
                "[[ops]]",
                'slot = "activation.silu_and_mul"',
                f'variant = "{architecture}"',
                f'source = "kernels/{architecture}/shim.py"',
                'entry = "entry"',
                f'architectures = ["{architecture}"]',
                f'cuda_sources = ["kernels/{architecture}/native.cu", '
                f'"kernels/{architecture}/values.cuh"]',
                "",
            )
        )
    (root / "manifest.toml").write_text(
        'bundle_id = "architecture-variants"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        + "\n".join(rows)
    )
    return root, markers


def _context(mod, image_root: Path, *, salt: str = "") -> dict[str, object]:
    return {
        "compiler_env_digest": _digest("env" + salt),
        "cxx11_abi": 1,
        "link_libraries": list(mod._LINK_LIBRARIES),
        "nvcc": {
            "path": "/image/cuda/bin/nvcc",
            "sha256": _digest("nvcc" + salt),
            "version": "fake nvcc " + salt,
        },
        "nvcc_architecture": "sm_120a",
        "nvcc_flags": list(mod._COMPILE_FLAGS),
        "pinned_build_roots": [str(image_root.resolve())],
        "ptxas": {
            "path": "/image/cuda/bin/ptxas",
            "sha256": _digest("ptxas" + salt),
            "version": "fake ptxas " + salt,
        },
        "python_include": "/image/python/include",
        "python_soabi": "fake-soabi",
        "python_version": "3.12.0",
        "torch_api_include": "/image/torch/api/include",
        "torch_cuda_version": "13.0",
        "torch_include": "/image/torch/include",
        "torch_lib": "/image/torch/lib",
        "torch_version": "2.fake",
    }


def test_development_with_sealed_architecture_does_not_probe_cuda(
    tmp_path, monkeypatch
):
    mod = _patcher()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda name, path=None: (
            f"/toolchain/{name}" if name in {"nvcc", "ptxas"} else None
        ),
    )
    monkeypatch.setattr(
        mod,
        "_declared_sources",
        lambda selected_bundle, architecture: (
            observed.update(bundle=selected_bundle, architecture=architecture)
            or ([], [], frozenset())
        ),
    )
    monkeypatch.setattr(
        mod, "_compiler_environment", lambda: {"PATH": "/toolchain"}
    )
    monkeypatch.setattr(
        mod,
        "_build_context",
        lambda architecture, env, production: {
            "architecture": architecture,
            "production": production,
        },
    )
    monkeypatch.setattr(mod, "_canonical_hash", lambda value: _digest("development"))
    monkeypatch.setattr(mod, "_patcher_hash", lambda: _digest("patcher"))
    monkeypatch.setattr(mod, "_build_set", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_validate_index", lambda **kwargs: [])
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("OPTIMA_CUDA_EXT_CACHE", str(tmp_path / "cache"))

    mod._development_all(bundle, "", _digest("tree"), "sm103")

    assert observed == {"bundle": bundle, "architecture": "sm103"}


def test_development_sealed_architecture_resolves_container_cuda_root(
    tmp_path, monkeypatch
):
    mod = _patcher()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    cuda_home = tmp_path / "cuda"
    cuda_bin = cuda_home / "bin"
    cuda_bin.mkdir(parents=True)
    for tool in ("nvcc", "ptxas"):
        path = cuda_bin / tool
        path.write_text("tool")
        path.chmod(0o755)
    observed: dict[str, object] = {}

    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setenv("OPTIMA_CUDA_EXT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(
        mod, "_compiler_environment", lambda: {"PATH": "/usr/bin:/bin"}
    )
    real_which = mod.shutil.which
    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda name, path=None: real_which(name, path=path),
    )
    monkeypatch.setattr(
        mod,
        "_declared_sources",
        lambda selected_bundle, architecture: (
            observed.update(bundle=selected_bundle, architecture=architecture)
            or ([], [], frozenset())
        ),
    )
    monkeypatch.setattr(
        mod,
        "_build_context",
        lambda architecture, env, production: (
            observed.update(path=env["PATH"], production=production)
            or {"architecture": architecture}
        ),
    )
    monkeypatch.setattr(mod, "_canonical_hash", lambda value: _digest("development"))
    monkeypatch.setattr(mod, "_patcher_hash", lambda: _digest("patcher"))
    monkeypatch.setattr(mod, "_validate_index", lambda **kwargs: [])

    mod._development_all(bundle, "", _digest("tree"), "sm103")

    assert observed == {
        "architecture": "sm103",
        "bundle": bundle,
        "path": f"{cuda_bin.resolve()}:/usr/bin:/bin",
        "production": False,
    }


@pytest.fixture()
def fake_build(monkeypatch, tmp_path):
    mod = _patcher()
    image_root = tmp_path / "image-root"
    image_root.mkdir()
    image_header = image_root / "cuda_runtime.h"
    image_header.write_text("// image-bound header\n")
    context = _context(mod, image_root)
    compiler_env = {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/image/cuda/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SOURCE_DATE_EPOCH": "0",
        "TEMP": "/tmp",
        "TMP": "/tmp",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
    }
    monkeypatch.setattr(mod, "_compiler_environment", lambda: dict(compiler_env))
    monkeypatch.setattr(
        mod, "_build_context", lambda architecture, env, production: dict(context)
    )
    compiles: list[tuple[str, dict[str, str]]] = []

    def compile_fixture(*, bundle, source, output, depfile, module_name, context, env):
        compiles.append((source, dict(env)))
        output.write_bytes(f"{module_name}:{Path(bundle, source).read_text()}".encode())
        local_header = Path(source).with_name("values.cuh").as_posix()
        dependencies = [source, local_header, str(image_header)]
        depfile.write_text(f"{module_name}: " + " ".join(dependencies) + "\n")

    monkeypatch.setattr(mod, "_compile", compile_fixture)
    return mod, image_root, context, compiler_env, compiles


def _set_build_env(
    monkeypatch,
    *,
    bundle: Path,
    stage: Path,
    build_spec: str,
    tree_digest: str,
    architecture: str = "sm120",
) -> None:
    monkeypatch.setenv("OPTIMA_BUNDLE_PATH", str(bundle))
    monkeypatch.setenv("OPTIMA_REBUILD_PHASE", "build")
    monkeypatch.setenv("OPTIMA_REBUILD_CONTAINER", "1")
    monkeypatch.setenv("OPTIMA_BUILD_PATH", "/usr/bin:/bin")
    monkeypatch.setenv("OPTIMA_BUILD_TMPDIR", "/tmp")
    monkeypatch.setenv("OPTIMA_NATIVE_COMPILE_TIMEOUT_S", "60")
    monkeypatch.setenv("OPTIMA_NATIVE_ARTIFACT_STAGE", str(stage))
    monkeypatch.setenv("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", build_spec)
    monkeypatch.setenv("OPTIMA_ENGINE_TREE_DIGEST", tree_digest)
    monkeypatch.setenv("OPTIMA_TARGET_GPU_ARCH", architecture)


def _build(
    monkeypatch,
    mod,
    tmp_path: Path,
    bundle: Path,
    *,
    build_spec: str | None = None,
    tree_digest: str | None = None,
    architecture: str = "sm120",
) -> tuple[Path, str, str, dict]:
    stage = tmp_path / f"stage-{len(list(tmp_path.glob('stage-*')))}"
    stage.mkdir()
    build_spec = build_spec or _digest("build-spec")
    tree_digest = tree_digest or _digest("tree")
    _set_build_env(
        monkeypatch,
        bundle=bundle,
        stage=stage,
        build_spec=build_spec,
        tree_digest=tree_digest,
        architecture=architecture,
    )
    mod.main()
    index = json.loads((stage / "cuda" / "extensions.json").read_text())
    return stage, build_spec, tree_digest, index


def _publish_and_arm_load(
    monkeypatch,
    tmp_path: Path,
    bundle: Path,
    stage: Path,
    build_spec: str,
    tree_digest: str,
    *,
    architecture: str = "sm120",
):
    from optima.eval.native_artifact import publish_native_artifact

    publication_root = tmp_path / "publications"
    publication = publish_native_artifact(
        stage, publication_root, build_spec_digest=build_spec
    )
    monkeypatch.setenv("OPTIMA_BUNDLE_PATH", str(bundle))
    monkeypatch.setenv("OPTIMA_REBUILD_PHASE", "load")
    monkeypatch.setenv("OPTIMA_ENGINE_WORKER", "1")
    monkeypatch.setenv("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", build_spec)
    monkeypatch.setenv("OPTIMA_ENGINE_TREE_DIGEST", tree_digest)
    monkeypatch.setenv("OPTIMA_TARGET_GPU_ARCH", architecture)
    monkeypatch.setenv("OPTIMA_NATIVE_ARTIFACT_ROOT", str(publication.root))
    monkeypatch.setenv(
        "OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST",
        publication.publication_digest,
    )
    return publication


def test_build_is_gpu_free_does_not_import_candidate_or_load_elf(
    tmp_path, monkeypatch, fake_build
):
    mod, _, _, compiler_env, compiles = fake_build
    marker = tmp_path / "candidate-python-ran"
    bundle = _bundle(tmp_path / "bundle", marker=marker)
    monkeypatch.setattr(
        mod,
        "_load",
        lambda *args: pytest.fail("build phase attempted to dlopen a native product"),
    )

    stage, build_spec, tree_digest, index = _build(
        monkeypatch, mod, tmp_path, bundle
    )

    assert not marker.exists()
    assert build_spec == index["build_spec_digest"]
    assert tree_digest == index["tree_digest"]
    assert index["target_architecture"] == "sm120"
    assert len(index["units"]) == 1
    assert compiles == [("kernels/native.cu", compiler_env)]
    assert (stage / index["units"][0]["artifact_path"]).is_file()


def test_architecture_variants_build_and_load_only_matching_native_units(
    tmp_path, monkeypatch, fake_build
):
    mod, _, _, compiler_env, compiles = fake_build
    bundle, markers = _architecture_variant_bundle(tmp_path / "bundle")

    stage, build_spec, tree_digest, index = _build(
        monkeypatch,
        mod,
        tmp_path,
        bundle,
        architecture="sm120",
    )

    assert compiles == [("kernels/sm120/native.cu", compiler_env)]
    assert index["selected_sources"] == [
        "kernels/sm120/native.cu",
        "kernels/sm120/values.cuh",
    ]
    assert [row["path"] for row in index["source_inventory"]] == [
        "kernels/sm103/native.cu",
        "kernels/sm103/values.cuh",
        "kernels/sm120/native.cu",
        "kernels/sm120/values.cuh",
    ]
    assert [row["source"] for row in index["units"]] == [
        "kernels/sm120/native.cu"
    ]
    assert all(not marker.exists() for marker in markers.values())

    publication = _publish_and_arm_load(
        monkeypatch,
        tmp_path,
        bundle,
        stage,
        build_spec,
        tree_digest,
        architecture="sm120",
    )
    loaded: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(mod, "_compile", lambda **kwargs: pytest.fail("load compiled"))
    monkeypatch.setattr(mod, "_load", lambda *args: loaded.append(args))

    mod.main()

    loaded_paths = [
        (alias, path.relative_to(publication.root).as_posix())
        for alias, _, path in loaded
    ]
    assert loaded_paths == [("native", index["units"][0]["artifact_path"])]
    assert all(not marker.exists() for marker in markers.values())


def test_normative_architecture_capability_selects_without_importing_python(
    tmp_path, monkeypatch, fake_build
):
    mod, *_, compiles = fake_build
    marker = tmp_path / "candidate-python-ran"
    bundle = _bundle(tmp_path / "bundle", marker=marker)
    metadata = bundle / "metadata"
    metadata.mkdir()
    (metadata / "native.json").write_text(
        json.dumps({"capabilities": {"architecture": {"exact": "sm120"}}})
    )
    manifest = bundle / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace(
            'entry = "entry"\n',
            'entry = "entry"\nmetadata = "metadata/native.json"\n',
        )
    )

    _, _, _, off_domain = _build(
        monkeypatch,
        mod,
        tmp_path,
        bundle,
        architecture="sm103",
    )
    assert off_domain["units"] == []
    assert off_domain["selected_sources"] == []
    assert [row["path"] for row in off_domain["source_inventory"]] == [
        "kernels/native.cu",
        "kernels/values.cuh",
    ]
    assert compiles == []
    assert not marker.exists()

    _, _, _, selected = _build(
        monkeypatch,
        mod,
        tmp_path,
        bundle,
        architecture="sm120",
    )
    assert [row["source"] for row in selected["units"]] == ["kernels/native.cu"]
    assert selected["selected_sources"] == [
        "kernels/native.cu",
        "kernels/values.cuh",
    ]
    assert len(compiles) == 1
    assert not marker.exists()


def test_architecture_projection_keeps_disjoint_shape_variants(
    tmp_path, monkeypatch, fake_build
):
    mod, *_, compiles = fake_build
    bundle = _bundle(tmp_path / "bundle")
    kernels = bundle / "kernels"
    (kernels / "second.py").write_text("def entry(x, out):\n    out.copy_(x)\n")
    (kernels / "second.cu").write_text('// q2 unit\n#include "values.cuh"\n')
    metadata = bundle / "metadata"
    metadata.mkdir()
    for variant, q_len in (("q1", 1), ("q2", 2)):
        (metadata / f"{variant}.json").write_text(
            json.dumps(
                {
                    "capabilities": {
                        "architecture": "sm120",
                        "q_len": q_len,
                    }
                }
            )
        )
    manifest = bundle / "manifest.toml"
    first = manifest.read_text().replace(
        'source = "kernels/shim.py"\n',
        'variant = "q1"\nsource = "kernels/shim.py"\n',
    ).replace(
        'entry = "entry"\n',
        'entry = "entry"\nmetadata = "metadata/q1.json"\n',
        1,
    )
    manifest.write_text(
        first
        + "\n[[ops]]\n"
        + 'slot = "activation.silu_and_mul"\n'
        + 'variant = "q2"\n'
        + 'source = "kernels/second.py"\n'
        + 'entry = "entry"\n'
        + 'metadata = "metadata/q2.json"\n'
        + 'cuda_sources = ["kernels/second.cu", "kernels/values.cuh"]\n'
    )

    _, _, _, index = _build(monkeypatch, mod, tmp_path, bundle)

    assert [row[0] for row in compiles] == [
        "kernels/native.cu",
        "kernels/second.cu",
    ]
    assert [row["source"] for row in index["units"]] == [
        "kernels/native.cu",
        "kernels/second.cu",
    ]


def test_selected_unit_cannot_include_an_off_domain_variant_header(
    tmp_path, monkeypatch, fake_build
):
    mod, _, _, _, compiles = fake_build
    bundle, _ = _architecture_variant_bundle(tmp_path / "bundle")
    image_header = tmp_path / "image-root" / "cuda_runtime.h"

    def compile_with_off_domain_header(
        *, bundle, source, output, depfile, module_name, context, env
    ):
        compiles.append((source, dict(env)))
        output.write_bytes(b"must-not-publish")
        depfile.write_text(
            f"{module_name}: {source} kernels/sm103/values.cuh {image_header}\n"
        )

    monkeypatch.setattr(mod, "_compile", compile_with_off_domain_header)
    with pytest.raises(
        mod.CUDAExtensionError,
        match="undeclared or off-domain native source.*sm103/values.cuh",
    ):
        _build(monkeypatch, mod, tmp_path, bundle, architecture="sm120")
    assert [row[0] for row in compiles] == ["kernels/sm120/native.cu"]


def test_bundle_display_id_cannot_alias_header_or_whole_tree_identity(
    tmp_path, monkeypatch, fake_build
):
    mod, *_ = fake_build
    first = _bundle(tmp_path / "first", header="#define VALUE 1\n")
    second = _bundle(tmp_path / "second", header="#define VALUE 2\n")
    _, _, _, first_index = _build(
        monkeypatch, mod, tmp_path, first, build_spec=_digest("first-build")
    )
    _, _, _, second_index = _build(
        monkeypatch, mod, tmp_path, second, build_spec=_digest("second-build")
    )
    assert first_index["units"][0]["artifact_id"] != second_index["units"][0]["artifact_id"]
    assert first_index["source_inventory"] != second_index["source_inventory"]


@pytest.mark.parametrize(
    "changed",
    ["build_spec", "tree", "architecture", "patcher", "toolchain", "source"],
)
def test_every_native_unit_identity_dimension_rotates(changed, tmp_path, fake_build):
    mod, image_root, context, *_ = fake_build
    inventory = [{"path": "kernels/native.cu", "sha256": _digest("source") }]
    arguments = {
        "build_spec_digest": _digest("build"),
        "tree_digest": _digest("tree"),
        "architecture": "sm120",
        "source": "kernels/native.cu",
        "alias": "native",
        "source_inventory": inventory,
        "patcher_sha256": _digest("patcher"),
        "context": context,
    }
    before = mod._artifact_identity(**arguments)[0]
    revised = dict(arguments)
    if changed == "build_spec":
        revised["build_spec_digest"] = _digest("other-build")
    elif changed == "tree":
        revised["tree_digest"] = _digest("other-tree")
    elif changed == "architecture":
        revised["architecture"] = "sm103"
    elif changed == "patcher":
        revised["patcher_sha256"] = _digest("other-patcher")
    elif changed == "toolchain":
        revised["context"] = _context(mod, image_root, salt="changed")
    else:
        revised["source_inventory"] = [
            {"path": "kernels/native.cu", "sha256": _digest("other-source")}
        ]
    assert mod._artifact_identity(**revised)[0] != before


def test_compiler_environment_is_an_allowlist_not_inherited(monkeypatch):
    mod = _patcher()
    monkeypatch.setenv("OPTIMA_BUILD_PATH", "/usr/bin:/bin")
    monkeypatch.setenv("OPTIMA_BUILD_TMPDIR", "/tmp")
    for name in (
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "CC",
        "CXX",
        "HTTP_PROXY",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.setenv(name, "attacker-controlled")

    environment = mod._compiler_environment()

    assert set(environment) == {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "SOURCE_DATE_EPOCH",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
    }
    assert environment["PATH"] == "/usr/bin:/bin"
    assert not set(environment) & {
        "CPATH",
        "NVCC_PREPEND_FLAGS",
        "LD_PRELOAD",
        "PYTHONPATH",
        "HTTP_PROXY",
    }


def test_nvcc_architecture_uses_only_supported_arch_specific_suffixes():
    mod = _patcher()
    assert mod._nvcc_architecture("sm80") == "sm_80"
    assert mod._nvcc_architecture("sm89") == "sm_89"
    assert mod._nvcc_architecture("sm90") == "sm_90a"
    assert mod._nvcc_architecture("sm103") == "sm_103a"
    assert mod._nvcc_architecture("sm120") == "sm_120a"


def test_build_rejects_dependency_outside_tree_and_pinned_image_roots(
    tmp_path, monkeypatch, fake_build
):
    mod, *_ = fake_build
    bundle = _bundle(tmp_path / "bundle")
    outside = tmp_path / "host-secret"
    outside.write_text("secret")

    def hostile_depfile(*, bundle, source, output, depfile, module_name, context, env):
        output.write_bytes(b"not-loaded")
        depfile.write_text(f"{module_name}: {source} {outside}\n")

    monkeypatch.setattr(mod, "_compile", hostile_depfile)
    monkeypatch.setattr(
        mod,
        "_load",
        lambda *args: pytest.fail("invalid dependency reached native load"),
    )
    with pytest.raises(mod.CUDAExtensionError, match="outside.*pinned image roots"):
        _build(monkeypatch, mod, tmp_path, bundle)


@pytest.mark.parametrize(
    "missing",
    [
        "OPTIMA_NATIVE_BUILD_SPEC_DIGEST",
        "OPTIMA_ENGINE_TREE_DIGEST",
        "OPTIMA_TARGET_GPU_ARCH",
    ],
)
def test_production_build_identity_is_mandatory(
    missing, tmp_path, monkeypatch, fake_build
):
    mod, *_ = fake_build
    bundle = _bundle(tmp_path / "bundle")
    stage = tmp_path / "stage"
    stage.mkdir()
    _set_build_env(
        monkeypatch,
        bundle=bundle,
        stage=stage,
        build_spec=_digest("build"),
        tree_digest=_digest("tree"),
    )
    monkeypatch.delenv(missing)
    with pytest.raises(mod.CUDAExtensionError):
        mod.main()
    assert not (stage / "cuda").exists()


def test_production_build_requires_container_marker(tmp_path, monkeypatch, fake_build):
    mod, *_ = fake_build
    bundle = _bundle(tmp_path / "bundle")
    stage = tmp_path / "stage"
    stage.mkdir()
    _set_build_env(
        monkeypatch,
        bundle=bundle,
        stage=stage,
        build_spec=_digest("build"),
        tree_digest=_digest("tree"),
    )
    monkeypatch.delenv("OPTIMA_REBUILD_CONTAINER")
    with pytest.raises(mod.CUDAExtensionError, match="disposable rebuild container"):
        mod.main()


@pytest.mark.parametrize(
    "missing",
    (
        "OPTIMA_BUILD_PATH",
        "OPTIMA_BUILD_TMPDIR",
        "OPTIMA_NATIVE_COMPILE_TIMEOUT_S",
    ),
)
def test_production_build_requires_explicit_compiler_policy(
    missing, tmp_path, monkeypatch, fake_build
):
    mod, *_ = fake_build
    bundle = _bundle(tmp_path / "bundle")
    stage = tmp_path / "stage"
    stage.mkdir()
    _set_build_env(
        monkeypatch,
        bundle=bundle,
        stage=stage,
        build_spec=_digest("build"),
        tree_digest=_digest("tree"),
    )
    monkeypatch.delenv(missing)
    with pytest.raises(mod.CUDAExtensionError, match=missing):
        mod.main()


def test_pinned_build_roots_are_required_and_root_is_forbidden(
    tmp_path, monkeypatch
):
    mod = _patcher()
    monkeypatch.delenv("OPTIMA_PINNED_BUILD_ROOTS", raising=False)
    with pytest.raises(mod.CUDAExtensionError, match="requires OPTIMA_PINNED_BUILD_ROOTS"):
        mod._pinned_roots((), production=True)
    monkeypatch.setenv("OPTIMA_PINNED_BUILD_ROOTS", "/")
    with pytest.raises(mod.CUDAExtensionError, match="unsafe"):
        mod._pinned_roots((), production=True)


def test_load_reopens_publication_validates_everything_then_dlopens_in_worker(
    tmp_path, monkeypatch, fake_build
):
    mod, *_ = fake_build
    marker = tmp_path / "candidate-python-ran"
    bundle = _bundle(tmp_path / "bundle", marker=marker, two_units=True)
    stage, build_spec, tree_digest, index = _build(
        monkeypatch, mod, tmp_path, bundle
    )
    publication = _publish_and_arm_load(
        monkeypatch, tmp_path, bundle, stage, build_spec, tree_digest
    )
    loaded: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(mod, "_compile", lambda **kwargs: pytest.fail("load compiled"))
    monkeypatch.setattr(mod, "_load", lambda *args: loaded.append(args))
    before = {
        path.relative_to(publication.root).as_posix(): (path.stat().st_mode, path.stat().st_mtime_ns)
        for path in publication.root.rglob("*")
    }

    mod.main()

    after = {
        path.relative_to(publication.root).as_posix(): (path.stat().st_mode, path.stat().st_mtime_ns)
        for path in publication.root.rglob("*")
    }
    assert before == after
    assert not marker.exists()
    assert [row[0] for row in loaded] == [unit["alias"] for unit in index["units"]]
    assert all(Path(row[2]).is_relative_to(publication.root) for row in loaded)


def test_load_requires_worker_and_exact_publication_digest(
    tmp_path, monkeypatch, fake_build
):
    mod, *_ = fake_build
    bundle = _bundle(tmp_path / "bundle")
    stage, build_spec, tree_digest, _ = _build(monkeypatch, mod, tmp_path, bundle)
    _publish_and_arm_load(monkeypatch, tmp_path, bundle, stage, build_spec, tree_digest)
    monkeypatch.delenv("OPTIMA_ENGINE_WORKER")
    with pytest.raises(mod.CUDAExtensionError, match="isolated engine worker"):
        mod.main()
    monkeypatch.setenv("OPTIMA_ENGINE_WORKER", "1")
    monkeypatch.setenv("OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST", _digest("wrong"))
    with pytest.raises(Exception, match="publication"):
        mod.main()


@pytest.mark.parametrize("kind", ["artifact", "depfile", "index"])
def test_tampered_publication_refuses_load_before_any_native_initializer(
    kind, tmp_path, monkeypatch, fake_build
):
    mod, *_ = fake_build
    bundle = _bundle(tmp_path / "bundle")
    stage, build_spec, tree_digest, index = _build(monkeypatch, mod, tmp_path, bundle)
    publication = _publish_and_arm_load(
        monkeypatch, tmp_path, bundle, stage, build_spec, tree_digest
    )
    row = index["units"][0]
    relative = {
        "artifact": row["artifact_path"],
        "depfile": row["depfile_path"],
        "index": "cuda/extensions.json",
    }[kind]
    victim = publication.root / relative
    victim.chmod(0o644)
    victim.write_bytes(victim.read_bytes() + b"tamper")
    loaded: list[object] = []
    monkeypatch.setattr(mod, "_load", lambda *args: loaded.append(args))

    with pytest.raises(Exception, match="native artifact|native extension"):
        mod.main()
    assert loaded == []


def test_all_units_are_validated_before_first_native_initializer(
    tmp_path, monkeypatch, fake_build
):
    mod, *_ = fake_build
    bundle = _bundle(tmp_path / "bundle", two_units=True)
    stage, build_spec, tree_digest, index = _build(monkeypatch, mod, tmp_path, bundle)
    publication = _publish_and_arm_load(
        monkeypatch, tmp_path, bundle, stage, build_spec, tree_digest
    )
    second = publication.root / index["units"][1]["artifact_path"]
    second.chmod(0o644)
    second.write_bytes(b"tampered second unit")
    loaded: list[object] = []
    monkeypatch.setattr(mod, "_load", lambda *args: loaded.append(args))
    with pytest.raises(Exception):
        mod.main()
    assert loaded == []


def test_duplicate_source_stems_reject_before_compile(
    tmp_path, monkeypatch, fake_build
):
    mod, *_, compiles = fake_build
    bundle = _bundle(tmp_path / "bundle")
    other = bundle / "other"
    other.mkdir()
    (other / "native.cu").write_text("// collision\n")
    manifest = bundle / "manifest.toml"
    manifest.write_text(
        manifest.read_text().replace(
            '"kernels/values.cuh"]', '"kernels/values.cuh", "other/native.cu"]'
        )
    )
    with pytest.raises(mod.CUDAExtensionError, match="share import alias"):
        _build(monkeypatch, mod, tmp_path, bundle)
    assert compiles == []


def test_native_import_alias_cannot_overwrite_existing_module(tmp_path):
    mod = _patcher()
    assert "json" in sys.modules
    with pytest.raises(mod.CUDAExtensionError, match="import alias collision"):
        mod._load("json", "_optima_cuda_unique_deadbeef", tmp_path / "missing.so")


def test_depfile_escape_parser_is_deterministic():
    mod = _patcher()
    assert mod._depfile_tokens("out: one\\ two \\\n three\n") == ["one two", "three"]
    with pytest.raises(mod.CUDAExtensionError, match="target separator"):
        mod._depfile_tokens("no target here")
