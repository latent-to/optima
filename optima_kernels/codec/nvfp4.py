"""NVFP4 codec + weight-layout primitives — re-homed clean, pure torch, CPU-validatable.

These are the weight-layout math the M3 swigluoai win *borrowed* from sglang
(``interleave_w13_halves`` / ``convert_sf_to_mma_layout`` / the scalar-alpha algebra),
reimplemented as engine-independent primitives we own. They are the spine of the
portable library and are themselves contributable (a faster/more-faithful codec is a
win). Here they are correct **references** — the high-perf CuTe-DSL kernels are a
swappable backend behind this interface, not the interface.

NVFP4 = e2m1 4-bit values {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, a per-16-element block
scale, and a per-tensor/per-expert fp32 global scale. The quant/dequant here round-trip
within NVFP4's representational error (exact when the input already lies on the grid);
the layout transforms (interleave / swizzle) round-trip EXACTLY (pure reshape/permute).
"""

from __future__ import annotations

import torch

# e2m1 positive magnitudes (1 sign + 2 exp + 1 mantissa).
_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_MAX = 6.0
NVFP4_BLOCK = 16


def _nearest_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round each element to the nearest e2m1 magnitude, preserving sign."""
    grid = torch.tensor(_E2M1_POS, device=x.device, dtype=torch.float32)
    idx = (x.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
    return torch.sign(x) * grid[idx]


def quantize_nvfp4(
    x: torch.Tensor, *, block: int = NVFP4_BLOCK, global_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference NVFP4 quantize. Returns ``(codes, block_scales)``:

    * ``codes`` — the e2m1 grid values (fp32, in [-6, 6]) per element (last dim unchanged).
    * ``block_scales`` — one fp32 scale per ``block`` elements (shape ``(..., n//block)``).

    ``dequantize_nvfp4(codes, block_scales, global_scale=...)`` is the inverse. Codes are
    kept in value space (not packed bits) so the round-trip is CPU-checkable; a real
    backend packs ``codes`` to 4-bit + ``block_scales`` to UE4M3.
    """
    *lead, n = x.shape
    if n % block != 0:
        raise ValueError(f"last dim {n} is not a multiple of block {block}")
    xb = (x.float() / global_scale).reshape(*lead, n // block, block)
    block_scale = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / NVFP4_MAX
    codes = _nearest_e2m1(xb / block_scale)  # in [-6, 6] on the e2m1 grid
    return codes.reshape(*lead, n), block_scale.squeeze(-1)


def dequantize_nvfp4(
    codes: torch.Tensor, block_scales: torch.Tensor, *, block: int = NVFP4_BLOCK,
    global_scale: float = 1.0,
) -> torch.Tensor:
    """Inverse of :func:`quantize_nvfp4`: ``x ≈ codes * block_scale * global_scale``."""
    *lead, n = codes.shape
    cb = codes.float().reshape(*lead, n // block, block)
    x = cb * block_scales.unsqueeze(-1) * global_scale
    return x.reshape(*lead, n)


def interleave_w13_halves(w: torch.Tensor, *, group: int = 64) -> torch.Tensor:
    """``w:(E, 2I, ...)`` laid out ``[gate(0:I) | up(I:2I)]`` along dim 1 ->
    ``[up, gate]`` interleaved in ``group``-row chunks (the donor megakernel's subtile
    order; M3 ships chunked ``[gate|up]``). Pure reshape — exactly invertible by
    :func:`deinterleave_w13_halves`. ``I`` must be a multiple of ``group``."""
    E, N = w.shape[0], w.shape[1]
    if N % 2 != 0 or (N // 2) % group != 0:
        raise ValueError(f"w13 dim1={N} must be even with half a multiple of group {group}")
    I = N // 2
    rest = w.shape[2:]
    ng = I // group
    g = w[:, :I].reshape(E, ng, group, *rest)
    u = w[:, I:].reshape(E, ng, group, *rest)
    return torch.stack([u, g], dim=2).reshape(E, N, *rest)


def deinterleave_w13_halves(w: torch.Tensor, *, group: int = 64) -> torch.Tensor:
    """Inverse of :func:`interleave_w13_halves` -> ``[gate | up]``."""
    E, N = w.shape[0], w.shape[1]
    I = N // 2
    rest = w.shape[2:]
    ng = I // group
    inter = w.reshape(E, ng, 2, group, *rest)
    u = inter[:, :, 0].reshape(E, I, *rest)
    g = inter[:, :, 1].reshape(E, I, *rest)
    return torch.cat([g, u], dim=1)


def swizzle_blockscale(sf: torch.Tensor) -> torch.Tensor:
    """Swizzle a ``(B, M, K)`` block-scale tensor into the FP4 swizzled layout
    (lifted from sglang ``utils.py``: ``reshape(B, M//128, 4, 32, K//4, 4).permute(
    0,1,4,3,2,5)``). Requires ``M % 128 == 0`` and ``K % 4 == 0``. The permute is an
    involution on axes 2/4 -> :func:`unswizzle_blockscale` inverts it exactly."""
    B, M, K = sf.shape
    if M % 128 != 0 or K % 4 != 0:
        raise ValueError(f"block-scale (M,K)=({M},{K}) needs M%128==0 and K%4==0")
    return sf.reshape(B, M // 128, 4, 32, K // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous()


def unswizzle_blockscale(s: torch.Tensor, *, M: int, K: int) -> torch.Tensor:
    """Inverse of :func:`swizzle_blockscale` (same involutive permute, then reshape back)."""
    B = s.shape[0]
    return s.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(B, M, K)


def scalarize_scale(quant_scale: torch.Tensor) -> torch.Tensor:
    """Collapse a per-expert/per-token quant scale to a single scalar via the TRTLLM
    convention ``min(quant_scale) = 1 / max(raw_scale)`` (the scalar activation scale the
    CuTe-DSL standard path consumes). Pure math, portable."""
    return quant_scale.reshape(-1).amin()


def gemm_alpha(weight_global_scale: torch.Tensor, used_input_scale: torch.Tensor) -> torch.Tensor:
    """Per-expert GEMM alpha consistent with a *scalar* activation quant:
    ``alpha[e] = weight_global_scale[e] / used_input_scale`` (the alpha re-derivation the
    CuTe-DSL standard path needs). The EP-slicing / checkpoint-format coercion is NOT here
    (that is engine glue — re-derive it against our own loader)."""
    return weight_global_scale.float() / used_input_scale.float()
