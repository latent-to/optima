"""Faithful MoE-experts-that-own-the-trailing-reduce kernel (the overlap-slot demo).

Contract (moe.fused_experts_reduce): ``prepare`` lays out the expert weights once;
``fused_experts_reduce(x, topk_ids, topk_weights, prepared, out, group)`` fills the
validator-allocated ``out`` with the SUM-OVER-RANKS of this rank's local expert
output — i.e. it owns BOTH the expert GEMM AND the trailing TP all-reduce.

This pure-torch version is a correctness demo: it computes the local experts, then
calls torch's all-reduce. A real submission is the whole point of this slot — it
**overlaps** the expert GEMM with the reduce (e.g. reduce-scatter the down-proj
output as it is produced), which a plain moe.fused_experts slot can't express
because the validator there replays a separate stock all-reduce after the kernel.

Verified DISTRIBUTED (optima.verify_collective) vs the fp32 cross-rank sum.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def prepare(w13, w2):
    """Runs ONCE at load. Reorder fused gate-up weights [gate; up] -> [up; gate]."""
    I = w13.shape[1] // 2
    w13_up_gate = torch.cat([w13[:, I:], w13[:, :I]], dim=1).contiguous()
    return {"w13": w13_up_gate, "w2": w2.contiguous(), "inter": I}


def fused_experts_reduce(x, topk_ids, topk_weights, prepared, out, group=None):
    """Local experts, THEN the cross-rank reduce — the kernel owns both."""
    w13 = prepared["w13"]   # (E, 2I, H), order [up; gate]
    w2 = prepared["w2"]     # (E, H, I)
    I = prepared["inter"]
    M, H = x.shape
    K = topk_ids.shape[1]
    x32 = x.float()
    acc = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    for k in range(K):
        e = topk_ids[:, k].long()
        wk = topk_weights[:, k].float()
        w13_e = w13[e].float()                          # (M, 2I, H)  [up; gate]
        w2_e = w2[e].float()                            # (M, H, I)
        fc1 = torch.einsum("mh,mih->mi", x32, w13_e)    # (M, 2I)
        up, gate = fc1[:, :I], fc1[:, I:]
        act = F.silu(gate) * up
        acc += wk[:, None] * torch.einsum("mi,mhi->mh", act, w2_e)
    out.copy_(acc.to(out.dtype))
    # The kernel OWNS the trailing reduce (a real submission overlaps this with the GEMM).
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
