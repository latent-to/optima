"""Epilogue override-points — the EFC (Epilogue Fusion Customization) submission ABI.

A miner ships only a small **epilogue** (a CuTe-DSL device fn) + its **torch reference**,
not a whole kernel. The validator owns a base kernel that exposes a typed hole and
JIT-composes the override in at load time. This is NVIDIA's own pattern: CUTLASS ships it
as ``examples/python/CuTeDSL/cute/blackwell/efc/`` (a named registry of activations, each a
device method + a built-in torch reference for the correctness check), and flashinfer's
fused-MoE kernel already threads an ``epilogue_op: cutlass.Constexpr`` hook.

Why load-time composition (not a new seam): the composed result has the *standard*
``fused_experts(x, topk_ids, topk_weights, prepared, out)`` signature, so it flows through
the existing MoE dispatcher and inherits validator output-ownership, eligibility, quant
pairing, graph-safety, and fallback — all four invariants — for free.

The override carries TWO callables:
  * the **device** epilogue (``@cute.jit``) — runs on GPU inside the base megakernel;
  * the **torch** epilogue (``(gate, up) -> act``) — the fidelity oracle, and what the
    CPU/dense path runs (so the whole mechanism is laptop-testable without cutlass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class EpiloguePoint:
    """A typed hole in a base kernel that a miner epilogue fills."""

    key: str  # "<slot>/<override_point>", e.g. "moe.fused_experts/gemm1_epilogue"
    base_kernel: str  # the validator-owned base in optima_kernels (e.g. "nvfp4_moe_megakernel")
    summary: str


# THE registry. Add an override-point here (epilogue -> codec -> prologue, per the roadmap).
OVERRIDE_POINTS: dict[str, EpiloguePoint] = {
    "moe.fused_experts/gemm1_epilogue": EpiloguePoint(
        key="moe.fused_experts/gemm1_epilogue",
        base_kernel="nvfp4_moe_megakernel",
        summary=(
            "GEMM1 epilogue of the fused NVFP4 MoE megakernel: a per-element activation "
            "epilogue(tCompute, gate, up, alpha, *act_params) applied to the GEMM1 "
            "accumulator (gate/up subtiles) before the fused NVFP4 requant. The swigluoai win."
        ),
    ),
}


def point_for(slot: str, override_point: str) -> EpiloguePoint:
    """Resolve (slot, override_point) to its EpiloguePoint, or raise a clear error."""
    key = f"{slot}/{override_point}"
    try:
        return OVERRIDE_POINTS[key]
    except KeyError:
        known = ", ".join(sorted(OVERRIDE_POINTS)) or "(none)"
        raise KeyError(f"unknown override-point {key!r}; known: {known}") from None


def _dense_moe(x, topk_ids, topk_weights, prepared, out, *, activation: Callable) -> torch.Tensor:
    """Generic dense SwiGLU-MLP MoE with a pluggable ``activation(gate, up) -> act``.

    The CPU/dense path of every gemm1_epilogue override — identical to the slot's own fp32
    reference except the activation is the miner's torch epilogue. Fills the validator-owned
    ``out``; computes in fp32."""
    w13, w2, I = prepared["w13"], prepared["w2"], prepared["inter"]
    M, H = x.shape
    acc = torch.zeros(M, H, dtype=torch.float32, device=x.device)
    x32 = x.float()
    for k in range(topk_ids.shape[1]):
        e = topk_ids[:, k].long()
        wk = topk_weights[:, k].float()
        fc1 = torch.einsum("mh,mih->mi", x32, w13[e].float())  # (M, 2I)
        gate, up = fc1[:, :I], fc1[:, I:]
        act = activation(gate, up).float()
        acc += wk[:, None] * torch.einsum("mi,mhi->mh", act, w2[e].float())
    out.copy_(acc.to(out.dtype))
    return out


def compose(
    slot: str,
    override_point: str,
    *,
    epilogue_torch: Callable,
    epilogue_device: Optional[Callable] = None,
) -> Callable:
    """Compose a base kernel + a miner epilogue into a standard ``fused_experts`` callable.

    ``epilogue_torch(gate, up) -> act`` is the portable torch reference (required; the
    fidelity oracle + the CPU/dense path). ``epilogue_device`` is the GPU ``@cute.jit``
    epilogue (optional on CPU). The returned ``fused_experts(x, topk_ids, topk_weights,
    prepared, out)`` picks the path off ``prepared["fmt"]``: ``"dense"`` -> the torch
    epilogue via :func:`_dense_moe` (laptop); otherwise the GPU megakernel with the device
    epilogue installed.
    """
    point = point_for(slot, override_point)  # validates the override-point exists

    def fused_experts(x, topk_ids, topk_weights, prepared, out):
        fmt = prepared.get("fmt") if isinstance(prepared, dict) else None
        if fmt == "dense":
            return _dense_moe(x, topk_ids, topk_weights, prepared, out, activation=epilogue_torch)
        from optima_kernels.moe import nvfp4_megakernel  # base kernel by name (point.base_kernel)

        assert point.base_kernel == "nvfp4_moe_megakernel"
        return nvfp4_megakernel.run(
            x, topk_ids, topk_weights, prepared, out,
            epilogue_device=epilogue_device, epilogue_torch=epilogue_torch,
        )

    fused_experts.__optima_override__ = point.key  # provenance (attribution)
    return fused_experts
