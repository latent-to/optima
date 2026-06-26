# miner_m3_swigluoai_override

The MiniMax-M3 NVFP4 MoE win, as the submission model intends it: **an override, not a kernel.**

## What the miner ships

One file — [`kernels/swigluoai.py`](kernels/swigluoai.py) — with two functions:

- `gemm1_epilogue_ref(gate, up)` — the **torch reference** (the fidelity oracle; runs the
  CPU/dense path). ~5 lines.
- `gemm1_epilogue(...)` — the **CuTe-DSL device epilogue** (GPU; cutlass-guarded). ~12 lines.

That's the whole contribution: M3's clamped swigluoai (`g·sigmoid(1.702·g)·(u+1)`, clamp ±7)
in place of the donor megakernel's hardcoded `up·silu(gate)`. No vendored kernel, no `sglang`
import, no `open()`-patch.

## How it runs

The fused NVFP4 MoE megakernel is the **validator-owned base** (`optima_kernels`
`nvfp4_moe_megakernel`, declared via `base_kernel`/`override_point` in the manifest). At load
the validator JIT-composes `base(epilogue=this)` into a standard `fused_experts`, so it flows
through the normal MoE dispatcher and inherits all four invariants. The activation is a
**model fact**: `MODEL_PROFILES["MiniMax-M3"]` gives the validator a swigluoai reference +
a cosine gate, so the kernel passes correctness against the *real* model.

## Verify (CPU)

```bash
python -m optima.cli verify examples/miner_m3_swigluoai_override --device cpu --dtype float32
```

The model key is auto-read from `metadata/moe.json` (`MiniMax-M3-NVFP4`). On CPU the device
epilogue is absent (no cutlass) and the **dense path** runs the torch reference — exercising
the contract end to end. With the generic slot (`--model __none__`) it correctly FAILS
(swigluoai output vs the SiLU reference) — the control that the *profile*, not the kernel, is
what makes it pass.

## GPU (tomorrow)

The GPU path needs `optima_kernels` `nvfp4_moe_megakernel` — the vendored-once,
epilogue-hooked flashinfer kernel (M1.2 in `docs/SUBMISSION_MODEL_BUILD_SPEC.md`). Until then
the GPU `run()` raises with the precise vendoring spec.
