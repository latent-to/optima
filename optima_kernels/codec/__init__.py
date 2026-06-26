"""NVFP4 codec + layout primitives (re-homed clean from sglang/flashinfer)."""

from __future__ import annotations

from optima_kernels.codec.nvfp4 import (
    NVFP4_BLOCK,
    NVFP4_MAX,
    deinterleave_w13_halves,
    dequantize_nvfp4,
    gemm_alpha,
    interleave_w13_halves,
    quantize_nvfp4,
    scalarize_scale,
    swizzle_blockscale,
    unswizzle_blockscale,
)

__all__ = [
    "NVFP4_BLOCK",
    "NVFP4_MAX",
    "quantize_nvfp4",
    "dequantize_nvfp4",
    "interleave_w13_halves",
    "deinterleave_w13_halves",
    "swizzle_blockscale",
    "unswizzle_blockscale",
    "scalarize_scale",
    "gemm_alpha",
]
