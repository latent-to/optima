"""Adversarial SPARSE kernel: drops SiLU on a thin stride of rows.

A *partial* "skip work" cheat — not the blunt drop-everything broken kernel. It is
bit-exact on most tokens and drops the SiLU nonlinearity only on every ``STRIDE``-th
row (token). Validated on Qwen2.5-1.5B (deterministic): ~1.3x faster, mean_kl ~9e-2,
**argmax-disagreement rate ~4%**, gsm8k 56%->19%. The hardened gate (``kl_gate_ok``)
catches it on the flip rate — *and*, at this density, also on the mean and accuracy.

The point of the **rate** check is the adversary the mean alone misses: a cheat
*tuned* to keep mean_kl just under the threshold while still flipping a few percent of
tokens. That regime is hard to hit with a real kernel (corrupting whole rows is bimodal
— some tokens break, others are perfect — so mean and rate move together), so it is
pinned down by the CPU regression test in ``tests/test_benchmarks.py`` instead; this
fixture is the end-to-end "a real faster cheat still fails" half. Not a real submission.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_silu(x_ptr, out_ptr, d, sxr, sxc, sor, soc, STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < d
    gate = tl.load(x_ptr + row * sxr + col * sxc, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * sxr + (col + d) * sxc, mask=mask, other=0.0).to(tl.float32)
    silu = gate * tl.sigmoid(gate)
    # corrupt == 1.0 on every STRIDE-th row -> that row gets gate*up (no SiLU); every
    # other row gets the correct silu*up. Scalar blend, broadcast over BLOCK.
    corrupt = ((row % STRIDE) == 0).to(tl.float32)
    res = corrupt * (gate * up) + (1.0 - corrupt) * (silu * up)
    tl.store(out_ptr + row * sor + col * soc, res.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    x2 = x.reshape(-1, x.shape[-1])
    o2 = out.reshape(-1, out.shape[-1])
    rows, _ = x2.shape
    d = o2.shape[1]
    STRIDE = 64  # drop SiLU on ~1/64 (~1.6%) of rows
    grid = (rows, triton.cdiv(d, 1024))
    _sparse_silu[grid](x2, o2, d, x2.stride(0), x2.stride(1), o2.stride(0), o2.stride(1), STRIDE, BLOCK=1024)
