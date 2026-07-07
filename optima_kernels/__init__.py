"""optima_kernels — the portable kernel library (the product / the moat).

This package is the thing the subnet dogfeeds into our own inference engine. Hard rule:
**ZERO sglang imports.** Everything here is device code + its launch + its weight-layout
math, expressed against kernel libraries (CUTLASS / CuTe-DSL / Triton / torch) and
portable to any engine. The harness (``optima/``) depends on this package; never the
reverse. See docs/SUBMISSION_MODEL.md (Axiom 5 — transferability).

Contents:
  * ``codec``    — re-homed-clean NVFP4 codec + layout primitives (pure torch, CPU-validatable).
  * ``override`` — the EFC-style epilogue override-point: a miner ships a small device
                   epilogue + a torch reference; the validator composes it into a base kernel.
  * ``moe``      — base kernels (the vendored-once, epilogue-hooked NVFP4 MoE megakernel; GPU).
  * ``collective`` — fused cross-GPU epilogues; first entry = the M3 fused
                   AR+residual+RMSNorm decode-epilogue win (portable launcher + .cu).
"""

from __future__ import annotations

__all__ = ["codec", "override", "moe", "collective"]
