"""Read-only FlashInfer dependency-overlay consumption."""

from __future__ import annotations

import functools
import hashlib
import json
import sys
import textwrap
import types
from dataclasses import asdict
from pathlib import Path

import pytest

import optima.integrations.flashinfer_overlay as fov
from optima import receipts


BUILD_DIGEST = "a" * 64


@pytest.fixture()
def fresh(monkeypatch):
    monkeypatch.setattr(fov, "_installed", False)
    return fov


def _stub_flashinfer(monkeypatch):
    environment = types.ModuleType("flashinfer.jit.env")
    environment.FLASHINFER_CSRC_DIR = Path("/stock/csrc")

    source = types.ModuleType("flashinfer.jit.fused_moe")

    def stock_generator(use_fast_build=False):
        return ("stock", use_fast_build)

    source.gen_cutlass_fused_moe_sm103_module = stock_generator

    consumer = types.ModuleType("flashinfer.fused_moe.core")
    consumer.gen_cutlass_fused_moe_sm103_module = stock_generator

    @functools.cache
    def get_cutlass_fused_moe_module(backend):
        return f"stock-{backend}"

    consumer.get_cutlass_fused_moe_module = get_cutlass_fused_moe_module

    loaded: list[str] = []
    tvm_ffi = types.ModuleType("tvm_ffi")
    tvm_ffi.load_module = lambda path: loaded.append(path) or f"loaded:{path}"

    for name, module in (
        ("flashinfer.jit.env", environment),
        ("flashinfer.jit.fused_moe", source),
        ("flashinfer.fused_moe.core", consumer),
        ("tvm_ffi", tvm_ffi),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return environment, source, consumer, loaded


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "patches").mkdir()
    (bundle / "kernels/k.py").write_text("def entry(x, out):\n    out.copy_(x)\n")
    (bundle / "patches/p.patch").write_text(
        "--- a/flashinfer/data/csrc/fused_moe/x.cu\n"
        "+++ b/flashinfer/data/csrc/fused_moe/x.cu\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    (bundle / "manifest.toml").write_text(
        textwrap.dedent(
            """\
            bundle_id = "optima-materialized-v1"
            abi_version = "optima-op-abi-v0"

            [[ops]]
            slot = "activation.silu_and_mul"
            source = "kernels/k.py"
            entry = "entry"
            dtypes = ["float32"]

            [[dep_patches]]
            target = "flashinfer"
            path = "patches/p.patch"
            """
        )
    )
    return bundle


def _candidate(tmp_path, monkeypatch, *, materialize=True):
    from optima.dep_policy import (
        PATCHABLE_DEPS,
        expected_overlay_stamp,
        prebuilt_module_relative_path,
        tree_inventory,
    )

    bundle = _bundle(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    if materialize:
        policy = PATCHABLE_DEPS["flashinfer"]
        overlay = artifact / "dep_overlays/flashinfer"
        subtree = overlay / policy.overlay_subtree
        (subtree / "fused_moe").mkdir(parents=True)
        source = subtree / "fused_moe/x.cu"
        source.write_text("new\n")

        module = policy.prebuilt_modules[0]
        module_relative = prebuilt_module_relative_path("flashinfer", module)
        module_path = artifact / module_relative
        module_path.parent.mkdir(parents=True)
        module_bytes = b"synthetic-shared-object"
        module_path.write_bytes(module_bytes)

        tree_digest, files = tree_inventory(subtree)
        stamp = expected_overlay_stamp(
            bundle, "flashinfer", build_spec_digest=BUILD_DIGEST
        )
        stamp.update(
            {
                "touched_files": {
                    "flashinfer/data/csrc/fused_moe/x.cu": hashlib.sha256(b"new\n").hexdigest()
                },
                "tree_digest": tree_digest,
                "tree_files": [row.to_dict() for row in files],
                "prebuilt_modules": [
                    {
                        **asdict(module),
                        "path": module_relative,
                        "sha256": hashlib.sha256(module_bytes).hexdigest(),
                        "size": len(module_bytes),
                    }
                ],
            }
        )
        (overlay / "overlay.json").write_text(
            json.dumps(stamp, sort_keys=True, separators=(",", ":"))
        )
    else:
        # A canonical publication can still lack the declared overlay; the runtime
        # must reject that absence rather than stock-fallback.
        (artifact / "prebuild.json").write_text("{}\n")

    from optima.eval.native_artifact import publish_native_artifact

    publication = publish_native_artifact(
        artifact, tmp_path / "published", build_spec_digest=BUILD_DIGEST
    )
    artifact = publication.root

    monkeypatch.setenv("OPTIMA_BUNDLE_PATH", str(bundle))
    monkeypatch.setenv("OPTIMA_ACTIVE", "1")
    monkeypatch.setenv("OPTIMA_REBUILD_PHASE", "load")
    monkeypatch.setenv("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", BUILD_DIGEST)
    monkeypatch.setenv("OPTIMA_NATIVE_ARTIFACT_ROOT", str(artifact))
    monkeypatch.setenv(
        "OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST",
        publication.publication_digest,
    )
    monkeypatch.setenv("OPTIMA_TARGET_GPU_ARCH", "sm103")
    return bundle, artifact


def test_installs_rebinds_load_only_generator_and_receipt(tmp_path, monkeypatch, fresh):
    environment, source, consumer, loaded = _stub_flashinfer(monkeypatch)
    _bundle_path, artifact = _candidate(tmp_path, monkeypatch)
    receipt_dir = tmp_path / "receipts"
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(receipt_dir))
    assert consumer.get_cutlass_fused_moe_module("103") == "stock-103"

    fov.install(registry=None)

    expected_subtree = artifact / "dep_overlays/flashinfer/flashinfer/data/csrc"
    assert environment.FLASHINFER_CSRC_DIR == expected_subtree
    assert source.gen_cutlass_fused_moe_sm103_module is consumer.gen_cutlass_fused_moe_sm103_module
    spec = consumer.gen_cutlass_fused_moe_sm103_module(False)
    assert not hasattr(spec, "build")
    expected_so = artifact / "dep_modules/flashinfer/fused_moe_103/fused_moe_103.so"
    assert spec.build_and_load() == f"loaded:{expected_so}"
    assert loaded == [str(expected_so)]
    assert consumer.get_cutlass_fused_moe_module.cache_info().currsize == 0
    with pytest.raises(RuntimeError, match="use_fast_build=False"):
        source.gen_cutlass_fused_moe_sm103_module(True)

    (receipt,) = receipts.collect(receipt_dir, "overlay")
    assert receipt == {
        "targets": ["flashinfer"],
        "prebuilt_modules": ["fused_moe_103"],
        "build_spec_digest": BUILD_DIGEST,
    }


def test_noop_when_not_active(tmp_path, monkeypatch, fresh):
    environment, source, _consumer, _loaded = _stub_flashinfer(monkeypatch)
    _candidate(tmp_path, monkeypatch)
    stock = source.gen_cutlass_fused_moe_sm103_module
    monkeypatch.setenv("OPTIMA_ACTIVE", "0")
    fov.install(registry=None)
    assert environment.FLASHINFER_CSRC_DIR == Path("/stock/csrc")
    assert source.gen_cutlass_fused_moe_sm103_module is stock


def test_declared_missing_overlay_is_terminal(tmp_path, monkeypatch, fresh):
    _stub_flashinfer(monkeypatch)
    _candidate(tmp_path, monkeypatch, materialize=False)
    with pytest.raises(RuntimeError, match="declared dep overlay.*missing"):
        fov.install(registry=None)


def test_corrupt_module_is_terminal(tmp_path, monkeypatch, fresh):
    _stub_flashinfer(monkeypatch)
    _bundle_path, artifact = _candidate(tmp_path, monkeypatch)
    module = artifact / "dep_modules/flashinfer/fused_moe_103/fused_moe_103.so"
    module.chmod(0o644)
    module.write_bytes(b"tampered")
    module.chmod(0o444)
    with pytest.raises(RuntimeError, match="unsafe shape|hash mismatch|inventory differs"):
        fov.install(registry=None)


def test_writable_publication_and_off_domain_arch_are_terminal(tmp_path, monkeypatch, fresh):
    _stub_flashinfer(monkeypatch)
    _bundle_path, artifact = _candidate(tmp_path, monkeypatch)
    artifact.chmod(0o755)
    with pytest.raises(RuntimeError, match="artifact root is writable|directory is writable|mode 0555"):
        fov.install(registry=None)

    artifact.chmod(0o555)
    monkeypatch.setenv("OPTIMA_TARGET_GPU_ARCH", "sm120")
    with pytest.raises(RuntimeError, match="requires 'sm103', got 'sm120'"):
        fov.install(registry=None)


def test_load_proxy_rechecks_bytes_immediately_before_dlopen(tmp_path, monkeypatch, fresh):
    _environment, source, _consumer, loaded = _stub_flashinfer(monkeypatch)
    _bundle_path, artifact = _candidate(tmp_path, monkeypatch)
    fov.install(registry=None)
    spec = source.gen_cutlass_fused_moe_sm103_module(False)
    spec.path.chmod(0o644)
    spec.path.write_bytes(b"changed-after-install")
    spec.path.chmod(0o444)
    with pytest.raises(RuntimeError, match="changed before load"):
        spec.build_and_load()
    assert loaded == []


def test_install_is_idempotent(tmp_path, monkeypatch, fresh):
    _environment, source, _consumer, _loaded = _stub_flashinfer(monkeypatch)
    _candidate(tmp_path, monkeypatch)
    fov.install(registry=None)
    replacement = source.gen_cutlass_fused_moe_sm103_module
    fov.install(registry=None)
    assert source.gen_cutlass_fused_moe_sm103_module is replacement


def test_seam_table_row_and_canary_form():
    from optima.compat import _chokepoint_present
    from optima.seams import SEAM_ADAPTERS, TARGET_MODULES

    (row,) = [adapter for adapter in SEAM_ADAPTERS if adapter.name == "flashinfer_overlay"]
    assert row.target_module == "flashinfer.jit.core"
    assert row.requires == "flashinfer"
    assert row.slots == ()
    assert "flashinfer.jit.core" in TARGET_MODULES

    stub = types.ModuleType("m")

    class JitSpec:
        pass

    stub.JitSpec = JitSpec
    assert _chokepoint_present(stub, "attr:JitSpec")
    assert not _chokepoint_present(stub, "attr:Missing")
