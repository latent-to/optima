"""Fused all-reduce + residual-add + RMSNorm — the M3 decode-epilogue win, first-party.

The 2026-07-02 MiniMax-M3 campaign's v6 kernel (``fused_ar_rmsnorm_sm103.cu``): a
one-shot/two-shot Lamport exchange (data-as-signal, triple-buffered sentinel rotation,
PDL) that fuses the TP all-reduce with the residual add and RMSNorm in ONE launch,
CUDA-graph-safe. Beat both the stock unfused chain and flashinfer's own fused kernel
at decode token counts on 4xB300 (receipts:
``experiments/minimax_m3/frontier_2026-07-02/``; regime NVFP4 TP4 graphs-ON decode).

This module is the PORTABLE launch layer: pure torch + torch.distributed, zero
sglang/harness imports (Axiom 5 — a kernel is a kernel). The engine-side wiring lives
elsewhere (optima's ``collective.ar_residual_rmsnorm`` slot / seam); the example miner
bundle ``experiments/minimax_m3/bundle/miner_m3_fused_epilogue`` vendors the same
source to simulate an external submission. THIS copy is the durable first-party asset.

Usage (engine-agnostic):
    from optima_kernels.collective import fused_ar_rmsnorm as far
    far.init(ext, group, device)                # once, EAGER (never under graph capture)
    far.ar_residual_rmsnorm(ext, x, residual, weight, eps, out_norm, out_res, group)

``ext`` is the compiled extension module (built from ``fused_ar_rmsnorm_sm103.cu`` by
whatever build pipeline the host uses — optima's reviewed ``build_cuda_ext`` patcher,
or a plain nvcc call per the campaign's build.sh). Passing it in keeps this module
import-clean on CPU boxes and lets the host own compilation policy.

Constraints carried from the campaign (each cost a real regression once):
* ``TWOSHOT_MIN = 48``: measured one-shot/two-shot crossover; two-shot at/above.
* ``MAX_T = 1024``: above this, callers must route to their stock path (prefill).
* init is FORBIDDEN under CUDA-graph capture (IPC workspace exchange allocates and
  synchronizes) — hosts must init on an eager warmup call.
* Plain caching-allocator tensors only: cudaIpcGetMemHandle breaks under
  ``expandable_segments`` — the host must not set that allocator mode.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

# Measured dispatch constants (bench_tp4, 4xB300, 2026-07-02; champion env of record).
TWOSHOT_MIN = 48
MAX_T = 1024
_LAMPORT_SENTINEL_I16 = -32768  # 0x8000 = bf16 -0.0 = "not arrived"

_state: dict = {"init": False}


def init(ext, group, device, *, pdl: bool = True) -> None:
    """Exchange the IPC workspace across ranks. EAGER ONLY — never under capture."""
    if _state["init"]:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("fused_ar_rmsnorm.init is not allowed under CUDA-graph capture")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    data = torch.empty(ext.NBUFS * ext.MAX_SLOTS * ext.MAX_RANKS * ext.H,
                       dtype=torch.bfloat16, device=device)
    data.view(torch.int16).fill_(_LAMPORT_SENTINEL_I16)
    flags = torch.zeros(ext.MAX_SLOTS * ext.MAX_RANKS, dtype=torch.int32, device=device)
    counter = torch.zeros(ext.MAX_SLOTS, dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    mine = {"data": ext.get_ipc_handle(data.data_ptr()).cpu(),
            "flags": ext.get_ipc_handle(flags.data_ptr()).cpu()}
    handles = [None] * world
    dist.all_gather_object(handles, mine, group=group)
    dptrs, fptrs = [], []
    for r in range(world):
        if r == rank:
            dptrs.append(data.data_ptr())
            fptrs.append(flags.data_ptr())
        else:
            dptrs.append(ext.open_ipc_handle(handles[r]["data"]))
            fptrs.append(ext.open_ipc_handle(handles[r]["flags"]))
    _state.update(
        keep=(data, flags, counter),
        dptrs=torch.tensor(dptrs, dtype=torch.int64),
        fptrs=torch.tensor(fptrs, dtype=torch.int64),
        cptr=counter.data_ptr(), rank=rank, world=world,
        pdl=1 if pdl else 0, init=True,
    )
    dist.barrier(group=group)


def mode_for(num_tokens: int) -> int:
    """Measured dispatch: 1 = one-shot below the crossover, 2 = two-shot at/above.
    Shape-dependent only (no data dependence) — capture-safe per shape."""
    return 2 if num_tokens >= TWOSHOT_MIN else 1


def ar_residual_rmsnorm(ext, x, residual, weight, eps, out_norm, out_residual, group) -> None:
    """Fill ``out_residual = sum_over_ranks(x) + residual`` and
    ``out_norm = rmsnorm(out_residual, weight, eps)`` in one fused launch.

    Callers own eligibility: bf16, hidden == ext.H, num_tokens <= MAX_T, init() done.
    """
    if not _state["init"]:
        raise RuntimeError("fused_ar_rmsnorm.init(ext, group, device) must run first (eager)")
    # ext arg order: (partial, residual, weight, NEW_RES, NORMED, ...) — new_res first.
    ext.ar_add_rmsnorm(x, residual, weight, out_residual, out_norm,
                       _state["dptrs"], _state["fptrs"], _state["cptr"],
                       _state["rank"], _state["world"], float(eps), 0.0,
                       mode_for(x.shape[0]), _state["pdl"])


def reference(x_summed: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
              eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The trusted math this kernel must reproduce, given the already-summed partials.
    fp32 in/out; the distributed reference is ``reference(allreduce(x), ...)``."""
    new_residual = x_summed.float() + residual.float()
    var = new_residual.pow(2).mean(dim=-1, keepdim=True)
    norm_out = new_residual * torch.rsqrt(var + float(eps)) * weight.float()
    return norm_out, new_residual
