"""Adversarial CPU dry-run kernel: drops the SiLU nonlinearity.

Computes ``gate * up`` instead of ``silu(gate) * up`` — the canonical "faster by
skipping work" cheat, in pure torch so the rejection is observable on a machine
without a GPU or Triton. ``optima verify`` must FAIL this bundle on
op-correctness. Not a real submission.
"""

from __future__ import annotations

import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    out.copy_(x[..., :d] * x[..., d:])  # missing silu() on the gate — wrong on purpose
