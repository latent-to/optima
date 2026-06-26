"""nvfp4_moe_megakernel — the validator-owned, epilogue-hooked NVFP4 MoE base kernel.

This is the base a ``moe.fused_experts / gemm1_epilogue`` override submission fills. The
heavy kernel (gather -> GEMM1 -> {EPILOGUE} -> NVFP4 requant -> GEMM2 -> fused finalize,
one persistent CuTe-DSL megakernel, ~8.7k LoC) is NVIDIA/flashinfer's; we vendor it ONCE
into the library and refactor its GEMM1 epilogue into a constexpr-callable hook so a miner
ships only the small epilogue (the swigluoai class). Shipped + tuned once; NOT per-bundle.

STATUS: the GPU path is a stub. The CPU/dense path (a generic dense MoE with the miner's
torch epilogue) lives in ``optima_kernels.override`` and is what runs + is tested on the
laptop. The GPU build is M1.2 in docs/SUBMISSION_MODEL_BUILD_SPEC.md and needs a GPU + the
pinned flashinfer source. Precise vendoring spec (verified against flashinfer 0.6.13):

  base file (vendor it):
    flashinfer/fused_moe/cute_dsl/blackwell/
        blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py
  refactor (~60-90 LoC, low-medium risk — the kernel already threads `epilogue_op:
  cutlass.Constexpr = lambda x: x` end-to-end for requant at :3520/:2697, so constexpr
  callables are a proven, zero-overhead pattern here):
    1. add `activation_fn: cutlass.Constexpr` (+ `act_params` tuple for swiglu_alpha/limit)
       to the ctor (:398-525), the device `wrapper` entry (:3499-3520), and the entrypoint
       (:303-328), threaded like `alpha`/`epilogue_op` already are.
    2. replace the hardcoded `up*silu(gate)` if/else (:2604-2666 — the unique
       `acc_vec_up_alpha * silu_f32(` site) with a single call
       `self.activation_fn(tCompute, acc_vec_gate, acc_vec_up, alpha_val, *act_params)`,
       defaulting to a `swiglu_silu_default` that preserves today's packed+scalar split.
    3. KEEP the requant branches (:2705 SFC, :2758 quantize) OUTSIDE the hook — they are
       format, not activation.
  ABI for the override device fn (matches optima_kernels.override / the EFC pattern, and
  experiments/minimax_m3/kernels/swigluoai_epilogue_cutedsl.py::swigluoai_epilogue_unpacked):
    epilogue(tCompute, acc_vec_gate, acc_vec_up, alpha_val, *act_params)  # per-element SCALAR
  Contract on the SCALAR accumulator (no packed min/max in the CuTe-DSL ISA -> a clamped
  activation must be scalar). Pass `act_params` (swiglu_alpha, swiglu_limit) SEPARATELY from
  the per-expert NVFP4 dequant `alpha_val` (the two-alpha trap).
"""

from __future__ import annotations


def run(x, topk_ids, topk_weights, prepared, out, *, epilogue_device=None, epilogue_torch=None):
    """GPU fused MoE with the miner's GEMM1 epilogue installed. Not yet built."""
    raise NotImplementedError(
        "nvfp4_moe_megakernel GPU path is M1.2 (vendor + epilogue-hook the flashinfer "
        "CuTe-DSL kernel; see this module's docstring). The CPU/dense path runs via "
        "optima_kernels.override._dense_moe with the torch epilogue."
    )
