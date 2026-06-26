"""CPU tests for per-model activation profiles (slots.MODEL_PROFILES / slot_for_model) and
the override-point verify path: a swigluoai epilogue PASSES under the MiniMax-M3 profile
(cosine vs a swigluoai reference) and FAILS generic (vs the SiLU reference) — the control
that proves the profile, not the kernel, was the missing piece.
"""

from __future__ import annotations

import textwrap

import pytest

torch = pytest.importorskip("torch")

from optima.slots import Activation, _moe_reference, get_slot, slot_for_model  # noqa: E402
from optima.verify import verify_entry_from_source  # noqa: E402


def test_slot_for_model_generic_unchanged():
    # No model key -> identical object to the generic slot (existing bundles untouched).
    assert slot_for_model("moe.fused_experts", None) is get_slot("moe.fused_experts")
    assert slot_for_model("moe.fused_experts", "UnknownModel") is get_slot("moe.fused_experts")


def test_slot_for_model_m3_swaps_activation_and_correctness():
    generic = get_slot("moe.fused_experts")
    m3 = slot_for_model("moe.fused_experts", "MiniMax-M3")
    assert generic.correctness.mode == "matched_ratio"
    assert m3.correctness.mode == "cosine" and m3.correctness.min_cosine == 0.985
    # alias resolves to the same profile
    assert slot_for_model("moe.fused_experts", "MiniMax-M3-NVFP4").correctness.mode == "cosine"

    # the rebound reference computes swigluoai, not SiLU
    g = torch.Generator().manual_seed(0)
    inp = generic.make_inputs(dtype=torch.float32, device="cpu", seed=0,
                              num_tokens=8, num_experts=4, hidden=32, inter=16, topk=2)
    ref_silu = generic.invoke_reference(inp)[0]
    ref_swig = m3.invoke_reference(inp)[0]
    assert not torch.allclose(ref_silu, ref_swig, atol=1e-3)


def test_moe_reference_swigluoai_differs_from_silu():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 32, generator=g) * 0.1
    w13 = torch.randn(4, 32, 32, generator=g) * 0.05  # 2I=32 -> I=16
    w2 = torch.randn(4, 32, 16, generator=g) * 0.05
    ids = torch.randint(0, 4, (8, 2), generator=g).to(torch.int32)
    sc = torch.rand(8, 2, generator=g)
    w = (sc / sc.sum(1, keepdim=True)).float()
    silu = _moe_reference(x, w13, w2, ids, w)  # default SiLU
    swig = _moe_reference(x, w13, w2, ids, w, Activation("swigluoai", 1.702, 7.0))
    assert not torch.allclose(silu, swig, atol=1e-3)


# A CPU override bundle source: just the torch reference (the device @cute.jit epilogue is
# GPU-only and absent here — the loader returns None for it, the dense path uses this).
_OVERRIDE_SRC = textwrap.dedent("""
    import torch

    def gemm1_epilogue_ref(gate, up, alpha=1.702, limit=7.0):
        g = gate.clamp(max=limit)
        u = up.clamp(min=-limit, max=limit)
        return g * torch.sigmoid(alpha * g) * (u + 1.0)
""")


def _write_src(tmp_path):
    p = tmp_path / "swigluoai.py"
    p.write_text(_OVERRIDE_SRC)
    return str(p)


def test_override_verify_passes_with_m3_profile(tmp_path):
    src = _write_src(tmp_path)
    res = verify_entry_from_source(
        "moe.fused_experts", src, "gemm1_epilogue",
        override_point="gemm1_epilogue", model_key="MiniMax-M3",
        dtype_name="float32", device="cpu",
    )
    assert res.passed, res.shape_results
    # cosine metric on the M3 profile
    assert all(r.metric == "cosine" for r in res.shape_results)


def test_override_verify_fails_generic(tmp_path):
    # Same swigluoai override, but the GENERIC slot reference is SiLU -> mismatch -> FAIL.
    # This is the control: the profile (not the kernel) is what makes it pass.
    src = _write_src(tmp_path)
    res = verify_entry_from_source(
        "moe.fused_experts", src, "gemm1_epilogue",
        override_point="gemm1_epilogue", model_key=None,
        dtype_name="float32", device="cpu",
    )
    assert not res.passed


def test_override_missing_torch_reference_errors(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def gemm1_epilogue(gate, up):\n    return gate\n")  # no _ref
    with pytest.raises(ValueError, match="must ship a torch reference"):
        verify_entry_from_source(
            "moe.fused_experts", str(p), "gemm1_epilogue",
            override_point="gemm1_epilogue", model_key="MiniMax-M3",
            dtype_name="float32", device="cpu",
        )


def test_example_swigluoai_override_bundle_verifies():
    """The shipped example override bundle: policy-clean (no open/sglang/vendored tree),
    is_override, and verifies under its declared M3 profile via the CPU dense path."""
    import json
    from pathlib import Path

    from optima.manifest import load_manifest, resolve_source
    from optima.sandbox import scan_path

    bundle = "examples/miner_m3_swigluoai_override"
    m = load_manifest(bundle)
    op = m.op_for("moe.fused_experts")
    assert op.is_override and op.base_kernel == "nvfp4_moe_megakernel"
    src = resolve_source(bundle, op)
    assert scan_path(str(src)).ok  # no banned builtins / vendored tree
    model_key = json.loads((Path(bundle) / op.metadata).read_text())["model"]
    res = verify_entry_from_source(
        "moe.fused_experts", str(src), op.entry, override_point=op.override_point,
        model_key=model_key, dtype_name="float32", device="cpu",
    )
    assert res.passed, res.shape_results
    assert all(r.metric == "cosine" for r in res.shape_results)
