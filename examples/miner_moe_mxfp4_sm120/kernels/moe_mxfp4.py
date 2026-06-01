"""MXFP4 fused-MoE slot kernel for GPT-OSS on RTX Blackwell (sm_120).

This is a REAL miner submission for the ``moe.fused_experts_mxfp4`` slot — the
FlashInfer CUTLASS MXFP8(act) x MXFP4(weight) fused-MoE path that beats SGLang's
plain-Triton MoE fallback on sm_120 (the experiments/ win, ~1.2-1.3x), expressed
entirely through the (prepare, forward) contract. No SGLang source patch, no engine
reconfigure: the validator routes the model's experts through this via the FusedMoE
block seam, allocates the output, and gates fidelity against a high-precision
reference.

Contract (validator-owned everything else):
  * prepare(w13, w2) -> prepared      runs ONCE on the layer's expert weights.
        w13:(E, 2I, H) in [gate; up] row order, w2:(E, H, I).  Reorders to the
        CUTLASS-Swiglu [up; gate] layout, pads to the kernel's tile, downcasts to
        packed MXFP4, and interleaves the block scales.  The validator holds the
        result.
  * forward(x, topk_ids, topk_weights, prepared, out)    runs per step.
        MXFP8-quantizes the activations and calls flashinfer.cutlass_fused_moe with
        GPT-OSS's clamped gated-SiLU (alpha=1.702, beta=1, limit=7), writing out:(M, H).

The winning recipe (see experiments/sm120_flashinfer_moe): plain packed FP4 weight
bytes, MXFP4 block scales through nvfp4_block_scale_interleave, linear (non-swizzled)
MXFP8 activation scales, enable_pdl=False.
"""

from __future__ import annotations

import contextlib
import os

import torch

import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp

# GPT-OSS expert activation constants (clamped gated SiLU).
_ALPHA, _BETA, _LIMIT = 1.702, 1.0, 7.0
_MXFP4_BLOCK = 32

# Autotune the CUTLASS grouped-GEMM tactic once per problem shape, then reuse the
# cached choice. sglang's startup flashinfer-autotune pass only profiles the MoE
# backend it's configured with (triton here) — it never sees this injected cutlass
# call, so without our own pass the kernel runs an untuned default tactic (~6% slower,
# the whole gap vs the patched/integrated flashinfer_mxfp4 path). The AutoTuner cache is
# a process-global singleton, so subsequent same-shape calls hit it with no profiling.
_AUTOTUNE = os.environ.get("OPTIMA_MXFP4_AUTOTUNE", "1") == "1"
_TUNED_SHAPES: set = set()


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _interleave_scales(scale: torch.Tensor) -> torch.Tensor:
    # Per-expert MXFP4 block-scale interleave into FlashInfer's layout.
    return torch.stack(
        [
            flashinfer.nvfp4_block_scale_interleave(scale[i].view(torch.uint8))
            .reshape_as(scale[i])
            .contiguous()
            for i in range(scale.shape[0])
        ]
    ).contiguous()


def prepare(weights):
    """Lay out the expert weights for the SM120 CUTLASS MXFP4 MoE (runs once).

    ``weights`` is a dict (the validator builds it from the slot's prepare_from_layer
    for the live model, or the verify harness for synthetic checks):
        w13:(E, 2I, H)   gate/up projection (dequantized bf16 on the live gpt-oss path)
        w2:(E, H, I)     down projection
        w13_bias:(E,2I)  optional fc1 bias       w2_bias:(E,H)  optional fc2 bias
        interleaved:bool HF stores gate_up as [gate0, up0, gate1, up1, ...]; True ->
                         de-interleave (even=gate, odd=up).  False -> [gate; up] blocks.
    CUTLASS wants halved [up(linear); gate]; we de-interleave/reorder, pad I->N_pad and
    H->K_pad, downcast to packed MXFP4, and interleave the block scales.
    """
    w13 = weights["w13"]
    w2 = weights["w2"]
    w13_bias = weights.get("w13_bias")
    w2_bias = weights.get("w2_bias")
    interleaved = bool(weights.get("interleaved", False))

    E, two_i, H = w13.shape
    I = two_i // 2
    dev = w13.device

    # Pull out the gate and up (linear) rows in whatever layout they arrived in.
    if interleaved:
        gate_w, up_w = w13[:, 0::2, :], w13[:, 1::2, :]          # even=gate, odd=up
        gate_b = None if w13_bias is None else w13_bias[:, 0::2]
        up_b = None if w13_bias is None else w13_bias[:, 1::2]
    else:
        gate_w, up_w = w13[:, :I, :], w13[:, I:, :]
        gate_b = None if w13_bias is None else w13_bias[:, :I]
        up_b = None if w13_bias is None else w13_bias[:, I:]

    N_pad = _round_up(I, 128)
    K_pad = _round_up(H, 128)

    # CUTLASS [up; gate] padded buffer.
    w13_p = torch.zeros(E, 2 * N_pad, K_pad, device=dev, dtype=torch.bfloat16)
    w13_p[:, :I, :H] = up_w.to(torch.bfloat16)                   # up (linear) -> first half
    w13_p[:, N_pad : N_pad + I, :H] = gate_w.to(torch.bfloat16)  # gate        -> second half
    w2_p = torch.zeros(E, K_pad, N_pad, device=dev, dtype=torch.bfloat16)
    w2_p[:, :H, :I] = w2.to(torch.bfloat16)

    w13_q, w13_s = downcast_to_mxfp(w13_p, torch.uint8, axis=-1)
    del w13_p  # free the padded bf16 scratch before allocating the next one (load-time peak)
    w2_q, w2_s = downcast_to_mxfp(w2_p, torch.uint8, axis=-1)
    del w2_p

    bias1 = torch.zeros(E, 2 * N_pad, device=dev, dtype=torch.bfloat16)
    bias2 = torch.zeros(E, K_pad, device=dev, dtype=torch.bfloat16)
    if up_b is not None:
        bias1[:, :I] = up_b.to(torch.bfloat16)
        bias1[:, N_pad : N_pad + I] = gate_b.to(torch.bfloat16)
    if w2_bias is not None:
        bias2[:, :H] = w2_bias.to(torch.bfloat16)

    w13_s_i, w2_s_i = _interleave_scales(w13_s), _interleave_scales(w2_s)
    del w13_s, w2_s
    torch.cuda.empty_cache()  # release transient scratch so the next layer's prepare has room
    return {
        "w13_q": w13_q.contiguous(),
        "w2_q": w2_q.contiguous(),
        "w13_s": w13_s_i,
        "w2_s": w2_s_i,
        "bias1": bias1,
        "bias2": bias2,
        "alpha": torch.full((E,), _ALPHA, device=dev, dtype=torch.float32),
        "beta": torch.full((E,), _BETA, device=dev, dtype=torch.float32),
        "limit": torch.full((E,), _LIMIT, device=dev, dtype=torch.float32),
        "g1": torch.ones((E,), device=dev, dtype=torch.float32),
        "g2": torch.ones((E,), device=dev, dtype=torch.float32),
        "K_pad": K_pad,
        "H": H,
    }


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fused_experts_mxfp4(x, topk_ids, topk_weights, prepared, out):
    """Per-step fused experts: MXFP8-quantize x, run the CUTLASS MXFP4 MoE, fill out."""
    K_pad = prepared["K_pad"]
    H = prepared["H"]
    M = x.shape[0]

    xb = x.to(torch.bfloat16).contiguous()
    x_q, x_sf = flashinfer.mxfp8_quantize(xb, False, alignment=K_pad)  # linear scales
    x_sf = x_sf.reshape(M, -1).contiguous()

    pad_out = torch.empty(M, K_pad, device=x.device, dtype=torch.bfloat16)
    # First time we see a problem shape, profile the cutlass tactic under autotune(True)
    # (caches the best choice); afterwards the call hits the global AutoTuner cache.
    shape_key = (M, K_pad)
    tune = _AUTOTUNE and shape_key not in _TUNED_SHAPES
    with (autotune(True) if tune else contextlib.nullcontext()):
        y = cutlass_fused_moe(
            input=x_q,
            input_sf=x_sf,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights.to(torch.float32),
            fc1_expert_weights=prepared["w13_q"].view(torch.long),
            fc2_expert_weights=prepared["w2_q"].view(torch.long),
            output_dtype=torch.bfloat16,
            quant_scales=[
                prepared["w13_s"].view(torch.int32),
                prepared["g1"],
                prepared["w2_s"].view(torch.int32),
                prepared["g2"],
            ],
            fc1_expert_biases=prepared["bias1"],
            fc2_expert_biases=prepared["bias2"],
            swiglu_alpha=prepared["alpha"],
            swiglu_beta=prepared["beta"],
            swiglu_limit=prepared["limit"],
            swizzled_input_sf=False,
            use_mxfp8_act_scaling=True,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=_next_pow2(M),
            enable_pdl=False,
            output=pad_out,
        )[0]
    if tune:
        _TUNED_SHAPES.add(shape_key)
    out.copy_(y[:, :H].to(out.dtype))
