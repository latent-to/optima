"""CPU tests for optima_kernels: the NVFP4 codec/layout primitives (round-trips) and the
epilogue override-point compose() (dense path == generic MoE with a pluggable activation).

No GPU, no cutlass, no sglang — exercises the portable library spine on the laptop.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from optima_kernels import codec  # noqa: E402
from optima_kernels.override import compose, point_for  # noqa: E402


# ---- codec: layout transforms round-trip EXACTLY ----------------------------

def test_interleave_w13_halves_roundtrips():
    w = torch.randn(4, 256, 8)  # (E, 2I=256, H); I=128, group=64 -> ng=2
    inter = codec.interleave_w13_halves(w, group=64)
    assert inter.shape == w.shape
    assert not torch.equal(inter, w)  # it actually reorders
    back = codec.deinterleave_w13_halves(inter, group=64)
    assert torch.equal(back, w)


def test_interleave_places_up_before_gate():
    # gate = first half, up = second half; after interleave, the first `group` rows are up's.
    E, I, group = 1, 64, 64
    gate = torch.zeros(E, I, 1)
    up = torch.ones(E, I, 1)
    w = torch.cat([gate, up], dim=1)  # [gate(0) | up(1)]
    inter = codec.interleave_w13_halves(w, group=group)
    assert torch.equal(inter[:, :group], up)   # up first
    assert torch.equal(inter[:, group:], gate)  # gate second


def test_swizzle_blockscale_roundtrips():
    sf = torch.randn(2, 128, 8)  # (B, M=128, K=8); M%128==0, K%4==0
    sw = codec.swizzle_blockscale(sf)
    back = codec.unswizzle_blockscale(sw, M=128, K=8)
    assert torch.equal(back, sf)


def test_swizzle_rejects_bad_shape():
    with pytest.raises(ValueError):
        codec.swizzle_blockscale(torch.randn(1, 100, 8))  # M not %128


# ---- codec: NVFP4 quant round-trips within representational error ------------

def test_nvfp4_exact_on_grid_values():
    # Values already on the (scaled) e2m1 grid quantize+dequantize exactly.
    x = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                       -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0]])
    codes, scales = codec.quantize_nvfp4(x, block=16)
    deq = codec.dequantize_nvfp4(codes, scales, block=16)
    assert torch.allclose(deq, x, atol=1e-5)


def test_nvfp4_roundtrip_faithful():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 64, generator=g)
    codes, scales = codec.quantize_nvfp4(x, block=16, global_scale=1.0)
    deq = codec.dequantize_nvfp4(codes, scales, block=16, global_scale=1.0)
    cos = F.cosine_similarity(deq.flatten(), x.flatten(), dim=0)
    assert cos > 0.99  # NVFP4 representational floor on smooth data
    assert codes.abs().max() <= codec.NVFP4_MAX + 1e-6  # values stay on/in the grid


def test_scalarize_and_alpha():
    qs = torch.tensor([0.25, 0.5, 0.125, 1.0])
    assert torch.isclose(codec.scalarize_scale(qs), torch.tensor(0.125))
    a = codec.gemm_alpha(torch.tensor([2.0, 4.0]), torch.tensor(0.5))
    assert torch.allclose(a, torch.tensor([4.0, 8.0]))


# ---- override: compose() dense path == generic MoE with a pluggable activation ----

def _dense_prepared(E=4, I=8, H=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randn(E, 2 * I, H, generator=g) * 0.1
    w2 = torch.randn(E, H, I, generator=g) * 0.1
    return {"fmt": "dense", "w13": w13, "w2": w2, "inter": I}, g


def _routing(M=8, E=4, topk=2, gen=None):
    ids = torch.randint(0, E, (M, topk), generator=gen)
    sc = torch.rand(M, topk, generator=gen)
    w = sc / sc.sum(1, keepdim=True)
    return ids.to(torch.int32), w.to(torch.float32)


def test_compose_silu_matches_slot_reference():
    """A SiLU torch epilogue through compose() reproduces the slot's own fp32 SiLU reference
    -> the dense override path IS the generic MoE, activation injected."""
    from optima.slots import _moe_reference

    prepared, g = _dense_prepared()
    H = prepared["w13"].shape[2]
    x = torch.randn(8, H, generator=g) * 0.1
    ids, weights = _routing(M=8, E=4, topk=2, gen=g)

    silu_epilogue = lambda gate, up: F.silu(gate) * up  # noqa: E731
    fused = compose("moe.fused_experts", "gemm1_epilogue", epilogue_torch=silu_epilogue)
    out = torch.empty(8, H)
    fused(x, ids, weights, prepared, out)

    ref = _moe_reference(x, prepared["w13"], prepared["w2"], ids, weights)
    assert torch.allclose(out.float(), ref.float(), atol=1e-4)


def test_compose_swigluoai_differs_and_is_correct():
    """The swigluoai epilogue produces the clamped-swigluoai math, distinct from SiLU."""
    prepared, g = _dense_prepared(seed=1)
    H = prepared["w13"].shape[2]
    x = torch.randn(8, H, generator=g) * 0.1
    ids, weights = _routing(M=8, E=4, topk=2, gen=g)

    def swigluoai(gate, up, alpha=1.702, limit=7.0):
        gc = gate.clamp(max=limit)
        uc = up.clamp(min=-limit, max=limit)
        return gc * torch.sigmoid(alpha * gc) * (uc + 1.0)

    fused = compose("moe.fused_experts", "gemm1_epilogue", epilogue_torch=swigluoai)
    out = torch.empty(8, H)
    fused(x, ids, weights, prepared, out)
    assert fused.__optima_override__ == "moe.fused_experts/gemm1_epilogue"

    # hand-compute the same dense MoE with swigluoai
    I = prepared["inter"]
    acc = torch.zeros(8, H)
    for k in range(ids.shape[1]):
        e = ids[:, k].long()
        fc1 = torch.einsum("mh,mih->mi", x.float(), prepared["w13"][e].float())
        act = swigluoai(fc1[:, :I], fc1[:, I:])
        acc += weights[:, k:k + 1].float() * torch.einsum("mi,mhi->mh", act, prepared["w2"][e].float())
    assert torch.allclose(out.float(), acc, atol=1e-4)

    # and it is NOT the SiLU answer
    silu = compose("moe.fused_experts", "gemm1_epilogue", epilogue_torch=lambda gate, up: F.silu(gate) * up)
    out_silu = torch.empty(8, H)
    silu(x, ids, weights, prepared, out_silu)
    assert not torch.allclose(out.float(), out_silu.float(), atol=1e-3)


def test_unknown_override_point_raises():
    with pytest.raises(KeyError, match="unknown override-point"):
        point_for("moe.fused_experts", "nonexistent_point")
    with pytest.raises(KeyError):
        compose("moe.fused_experts", "nonexistent_point", epilogue_torch=lambda g, u: g)
