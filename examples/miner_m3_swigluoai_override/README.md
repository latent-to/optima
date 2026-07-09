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
a cosine gate, so the kernel passes correctness against the *real* model. On CPU the dense
path runs the torch reference; the GPU base kernel is not yet built (see **GPU status**
below).

## Verify (CPU)

```bash
python -m optima.cli verify examples/miner_m3_swigluoai_override --device cpu --dtype float32
```

The model key is auto-read from `metadata/moe.json` (`MiniMax-M3-NVFP4`). On CPU the device
epilogue is absent (no cutlass) and the **dense path** runs the torch reference — exercising
the contract end to end. With the generic slot (`--model __none__`) it correctly FAILS
(swigluoai output vs the SiLU reference) — the control that the *profile*, not the kernel, is
what makes it pass.

## GPU status

The GPU fast path is **not yet built**. The `optima_kernels nvfp4_moe_megakernel` base — the
vendored-once, epilogue-hooked flashinfer kernel (M1.2 in
`docs/SUBMISSION_MODEL_BUILD_SPEC.md`) — is still a stub: its `run()` raises
`NotImplementedError` with the vendoring spec, so the override composes on the CPU/dense path
only. The composition mechanism itself (manifest `base_kernel`/`override_point` →
`optima_kernels.override.build_override`) is implemented and CPU-tested; the unbuilt piece is
the GPU base kernel it composes into.

The 1.12×/1.15× numbers in `metadata/moe.json` are the provenance of this epilogue, not a
result of this bundle: they were measured on the earlier non-override shape (a runtime
flashinfer `open()`-patch, since sandbox-banned). Reproducing them through this bundle is what
building M1.2 unblocks. The realized MiniMax-M3 win went a different route — the
`miner_m3_fused_epilogue[_deep]` bundles on the `collective.ar_residual_rmsnorm` slot
(documented in `docs/STATE_OF_RECORD.md`).
