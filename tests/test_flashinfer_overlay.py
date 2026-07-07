"""Runtime consume side of dep_patches (optima/integrations/flashinfer_overlay.py).

Exercised against STUB flashinfer modules (the real package only exists on engine
boxes): the integration must rebind the policy's env constant at the materialized
overlay, force JIT for exactly the policy's module names, clear the cached module
getter, write the `overlay` receipt — and no-op in every non-candidate situation.
"""

from __future__ import annotations

import functools
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest

import optima.integrations.flashinfer_overlay as fov
from optima import receipts


@pytest.fixture()
def fresh(monkeypatch):
    monkeypatch.setattr(fov, "_installed", False)
    return fov


def _stub_flashinfer(monkeypatch):
    """Install minimal flashinfer.jit.{env,core} + fused_moe.core stubs."""
    env_mod = types.ModuleType("flashinfer.jit.env")
    env_mod.FLASHINFER_CSRC_DIR = Path("/stock/csrc")

    core_mod = types.ModuleType("flashinfer.jit.core")

    class JitSpec:
        def __init__(self, name):
            self.name = name

        @property
        def is_aot(self):
            return True  # stock behavior on this "box": everything is AOT-prebuilt

    core_mod.JitSpec = JitSpec

    fm_mod = types.ModuleType("flashinfer.fused_moe.core")

    @functools.cache
    def get_cutlass_fused_moe_module(backend):
        return f"module-{backend}"

    fm_mod.get_cutlass_fused_moe_module = get_cutlass_fused_moe_module

    for name, mod in [("flashinfer.jit.env", env_mod), ("flashinfer.jit.core", core_mod),
                      ("flashinfer.fused_moe.core", fm_mod)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return env_mod, core_mod, fm_mod


def _mk_candidate(tmp_path, monkeypatch, *, materialize=True):
    """A dep-patched bundle + (optionally) its materialized overlay + candidate env."""
    bundle = tmp_path / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "patches").mkdir()
    (bundle / "kernels" / "k.py").write_text("def entry(x, out):\n    out.copy_(x)\n")
    (bundle / "patches" / "p.patch").write_text(
        "--- a/flashinfer/data/csrc/fused_moe/x.cu\n"
        "+++ b/flashinfer/data/csrc/fused_moe/x.cu\n"
        "@@ -1 +1 @@\n-old\n+new\n")
    (bundle / "manifest.toml").write_text(textwrap.dedent("""\
        bundle_id = "overlay-rt-test"
        abi_version = "optima-op-abi-v0"

        [[ops]]
        slot = "activation.silu_and_mul"
        source = "kernels/k.py"
        entry = "entry"
        dtypes = ["float32"]

        [[dep_patches]]
        target = "flashinfer"
        path = "patches/p.patch"
    """))
    cache = tmp_path / "overlay_cache"
    monkeypatch.setenv("OPTIMA_DEP_OVERLAY_CACHE", str(cache))
    monkeypatch.setenv("OPTIMA_BUNDLE_PATH", str(bundle))
    monkeypatch.setenv("OPTIMA_ACTIVE", "1")
    if materialize:
        ov = cache / "overlay-rt-test" / "flashinfer"
        subtree = ov / "flashinfer" / "data" / "csrc"
        (subtree / "fused_moe").mkdir(parents=True)
        (subtree / "fused_moe" / "x.cu").write_text("new\n")
        (ov / "overlay.json").write_text(json.dumps({
            "bundle_id": "overlay-rt-test", "target": "flashinfer",
            "patch_shas": {"patches/p.patch": "x"},
            "subtree": "flashinfer/data/csrc",
            "files": {"flashinfer/data/csrc/fused_moe/x.cu": "y"},
        }))
    return bundle


def test_installs_rebind_forcejit_receipt(tmp_path, monkeypatch, fresh):
    env_mod, core_mod, fm_mod = _stub_flashinfer(monkeypatch)
    _mk_candidate(tmp_path, monkeypatch)
    rdir = tmp_path / "receipts"
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(rdir))
    # simulate a getter that was already consulted (must be cache_clear'd)
    assert fm_mod.get_cutlass_fused_moe_module("103") == "module-103"

    fov.install(registry=None)

    assert env_mod.FLASHINFER_CSRC_DIR == (tmp_path / "overlay_cache" / "overlay-rt-test"
                                           / "flashinfer" / "flashinfer" / "data" / "csrc")
    assert core_mod.JitSpec("fused_moe_103").is_aot is False  # forced JIT
    assert core_mod.JitSpec("norm").is_aot is True  # everything else untouched
    assert fm_mod.get_cutlass_fused_moe_module.cache_info().currsize == 0  # cleared
    (got,) = receipts.collect(rdir, "overlay")
    assert got == {"targets": ["flashinfer"], "force_jit": ["fused_moe_103"]}


def test_noop_when_not_active(tmp_path, monkeypatch, fresh):
    env_mod, _, _ = _stub_flashinfer(monkeypatch)
    _mk_candidate(tmp_path, monkeypatch)
    monkeypatch.setenv("OPTIMA_ACTIVE", "0")
    fov.install(registry=None)
    assert env_mod.FLASHINFER_CSRC_DIR == Path("/stock/csrc")


def test_noop_without_materialized_overlay(tmp_path, monkeypatch, fresh, caplog):
    env_mod, _, _ = _stub_flashinfer(monkeypatch)
    _mk_candidate(tmp_path, monkeypatch, materialize=False)
    with caplog.at_level("WARNING", logger="optima.flashinfer_overlay"):
        fov.install(registry=None)
    assert env_mod.FLASHINFER_CSRC_DIR == Path("/stock/csrc")
    assert any("no overlay is materialized" in r.message for r in caplog.records)


def test_install_is_idempotent(tmp_path, monkeypatch, fresh):
    env_mod, core_mod, _ = _stub_flashinfer(monkeypatch)
    _mk_candidate(tmp_path, monkeypatch)
    fov.install(registry=None)
    forced_prop = core_mod.JitSpec.is_aot
    fov.install(registry=None)  # second call must not re-wrap the wrapped property
    assert core_mod.JitSpec.is_aot is forced_prop


def test_seam_table_row_and_canary_form():
    from optima.compat import _chokepoint_present
    from optima.seams import SEAM_ADAPTERS, TARGET_MODULES

    (row,) = [a for a in SEAM_ADAPTERS if a.name == "flashinfer_overlay"]
    assert row.target_module == "flashinfer.jit.core"
    assert row.requires == "flashinfer"
    assert row.slots == ()
    assert "flashinfer.jit.core" in TARGET_MODULES

    stub = types.ModuleType("m")

    class JitSpec:  # non-callable-instance attr is fine; attr: form is hasattr
        pass

    stub.JitSpec = JitSpec
    assert _chokepoint_present(stub, "attr:JitSpec")
    assert not _chokepoint_present(stub, "attr:Missing")
