"""MiniMax-M3 NVFP4 MoE — the swigluoai GEMM1 epilogue (an override submission).

The entire win: replace the fused MoE megakernel's hardcoded ``up * silu(gate)`` with M3's
clamped *swigluoai* ``g*sigmoid(1.702*g)*(u+1)`` (clamp ±7), computed in the GEMM1 epilogue
while the accumulator is still TMEM/RMEM-resident (no HBM round-trip). The ~8.7k-LoC fused
kernel is the validator's base; the miner ships only this epilogue.

Two callables, the EFC pairing (CUTLASS Epilogue Fusion Customization):
  * ``gemm1_epilogue_ref`` — the portable torch reference: the fidelity oracle, and what the
    CPU/dense path runs. Always importable (no cutlass needed).
  * ``gemm1_epilogue``     — the CuTe-DSL device epilogue (GPU). Behind a cutlass guard so this
    module imports on a CPU box; the validator's loader returns None for it there and uses the
    torch reference.
"""

from __future__ import annotations

SWIGLU_ALPHA = 1.702  # the sigmoid gain — DISTINCT from the per-expert NVFP4 dequant scale
SWIGLU_LIMIT = 7.0


def gemm1_epilogue_ref(gate, up, alpha: float = SWIGLU_ALPHA, limit: float = SWIGLU_LIMIT):
    """M3 clamped swigluoai. ``gate``/``up`` are the GEMM1 (gate, up) subtiles in activation
    space (the per-expert NVFP4 dequant scale is folded into the accumulator on GPU)."""
    import torch

    g = gate.clamp(max=limit)
    u = up.clamp(min=-limit, max=limit)
    return g * torch.sigmoid(alpha * g) * (u + 1.0)


try:  # GPU device epilogue — optional (a CPU box has no cutlass; the torch reference runs)
    import cutlass
    import cutlass.cute as cute

    _LOG2E = cutlass.Float32(1.4426950408889634)  # 1/ln2: exp(x)=exp2(x*LOG2E), donor's fast path

    @cute.jit
    def gemm1_epilogue(
        tCompute, acc_vec_gate, acc_vec_up, alpha_val,
        swiglu_alpha: cutlass.Constexpr = SWIGLU_ALPHA,
        swiglu_limit: cutlass.Constexpr = SWIGLU_LIMIT,
    ):
        """Per-element SCALAR epilogue (no packed min/max in the ISA -> the clamp can't pack).
        ``alpha_val`` is the per-expert NVFP4 dequant scale; ``swiglu_alpha`` is the sigmoid
        gain — kept separate (the two-alpha trap)."""
        L = cutlass.Float32(swiglu_limit)
        A = cutlass.Float32(swiglu_alpha)
        a = cutlass.Float32(alpha_val)
        for i in cutlass.range_constexpr(cute.size(acc_vec_up)):
            g = cute.arch.fmin(acc_vec_gate[i] * a, L)
            u = cute.arch.fmin(cute.arch.fmax(acc_vec_up[i] * a, -L), L)
            s = cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp2(-(g * A) * _LOG2E, fastmath=True)
            )
            tCompute[i] = g * s * (u + cutlass.Float32(1.0))
except ImportError:  # pragma: no cover - exercised only on a GPU box with cutlass
    pass
