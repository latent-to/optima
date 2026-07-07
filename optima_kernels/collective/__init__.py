"""Collective kernels: fused cross-GPU epilogues (all-reduce + downstream math).

Same hard rule as the rest of ``optima_kernels``: ZERO sglang imports. A collective
here takes a ``torch.distributed`` process group from the caller — whichever engine
owns the call site hands it in.
"""

from __future__ import annotations

__all__ = ["fused_ar_rmsnorm"]
