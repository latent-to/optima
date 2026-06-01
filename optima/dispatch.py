"""The validator-owned dispatcher.

This is the only place a miner kernel is actually invoked during serving. It is
written so that the *validator* owns everything risky around the call:

  * output allocation (shape/dtype/device/stride) — never the miner,
  * eligibility gating via the registry,
  * a fallback to the trusted baseline on ineligibility or error,
  * a single, auditable call into the miner ``entry(*inputs, out)``.

The miner's ``entry`` therefore only ever sees already-allocated tensors and is
expected to fill ``out``. That is the smallest host surface we can give a
Triton/CuteDSL submission while still letting it own the actual computation.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from optima.registry import REGISTRY, KernelRegistry

logger = logging.getLogger("optima.dispatch")
_MOE_LOGGED_ACTIVE = False
_MOE_LOGGED_FALLBACK = False


def _arch_tag(device_index: int = 0) -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"


def make_silu_and_mul_dispatcher(
    baseline_forward: Callable[[object, torch.Tensor], torch.Tensor],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "activation.silu_and_mul",
) -> Callable[[object, torch.Tensor], torch.Tensor]:
    """Build a replacement for ``SiluAndMul.forward_*``.

    ``baseline_forward`` is the captured original (used for fallback). The
    returned function has the same ``(self, x)`` signature.
    """

    def dispatched(self: object, x: torch.Tensor) -> torch.Tensor:
        last_dim = x.shape[-1]
        impl = registry.lookup(
            slot,
            dtype_name=_dtype_name(x.dtype),
            last_dim=last_dim,
            arch=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        )
        if impl is None:
            return baseline_forward(self, x)

        d = last_dim // 2
        out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
        try:
            impl.entry(x, out)
        except Exception:
            if registry.strict:
                raise
            # Quality/throughput already protect us; a crashing kernel just loses.
            return baseline_forward(self, x)
        return out

    return dispatched


def make_rmsnorm_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "norm.rmsnorm",
) -> Callable[..., object]:
    """Build a replacement for ``RMSNorm.forward_cuda`` / ``forward_native``.

    sglang's RMSNorm has two modes: plain (``residual is None`` -> return normed)
    and fused add+norm (``residual`` given -> return ``(normed, x+residual)``).
    The validator owns the residual add; the miner kernel only ever computes the
    pure rmsnorm: ``entry(x, weight, out, eps)``. Unusual paths fall back to the
    trusted baseline.
    """

    def dispatched(self, x, residual=None, post_residual_addition=None):
        # Rare / fp32 paths -> trusted baseline (keeps the contract simple & safe).
        if post_residual_addition is not None or getattr(self, "fp32_residual", False):
            return baseline_forward(self, x, residual, post_residual_addition)

        impl = registry.lookup(
            slot,
            dtype_name=_dtype_name(x.dtype),
            last_dim=x.shape[-1],
            arch=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        )
        if impl is None:
            return baseline_forward(self, x, residual, post_residual_addition)

        eps = float(self.variance_epsilon)
        weight = self.weight.data
        try:
            if residual is None:
                out = torch.empty_like(x)
                impl.entry(x, weight, out, eps)
                return out
            new_residual = x + residual  # validator owns the add
            out = torch.empty_like(new_residual)
            impl.entry(new_residual, weight, out, eps)
            return out, new_residual
        except Exception:
            if registry.strict:
                raise
            return baseline_forward(self, x, residual, post_residual_addition)

    return dispatched


def make_attention_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "attention.decode",
) -> Callable[..., object]:
    """Build a replacement for ``RadixAttention.forward`` — the single chokepoint
    every attention call funnels through (so it is backend-agnostic).

    Attention is a *block* slot. At **decode** (one query token per request) we
    extract the request's paged KV out of ``forward_batch`` and hand the miner kernel
    a dense ``(q, k, v, seq_lens)`` view; the validator keeps the backend metadata,
    owns the KV-cache **write** (``set_kv_buffer``), and only ever lets the miner
    *read* q/k/v. The kernel output feeds the residual stream -> downstream layers ->
    sampler (all stock), so there is no final output to substitute — the same
    property that makes the op slots safe.

    SCOPE: this routes **decode** attention through the ``attention.decode`` slot, and
    only when ``OPTIMA_ATTENTION_SEAM=1`` (opt-in until paged-direct lands). It is a
    *gather* MVP — it pulls the paged KV into a dense padded tensor so the miner writes
    an ordinary attention kernel, but the gather is variable-shape, hence
    **eager-only** (a per-step ``max_len`` is not CUDA-graph-capturable). The
    production rung is a paged-direct contract (the miner consumes the page table +
    pool buffers, graph-safe). Prefill / MLA / cross-attention / windowed paths fall
    back to the trusted backend. Conservative by construction: when in doubt, trust
    the baseline.
    """

    def dispatched(self, q, k, v, forward_batch, save_kv_cache: bool = True, **kwargs):
        if _attention_seam_active():
            try:
                if forward_batch.forward_mode.is_decode() and _decode_supported(self, k, v, kwargs):
                    impl = registry.lookup(
                        slot,
                        dtype_name=_dtype_name(q.dtype),
                        last_dim=getattr(self, "qk_head_dim", q.shape[-1]),
                        arch=_arch_tag(q.device.index or 0) if q.is_cuda else None,
                    )
                    if impl is not None:
                        return _run_decode_kernel(self, q, k, v, forward_batch, save_kv_cache, impl)
            except Exception:
                if registry.strict:
                    raise
                # any mismatch with this sglang's internals -> trust the baseline
        return baseline_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)

    return dispatched


def _attention_seam_active() -> bool:
    # Opt-in until the paged-direct (graph-safe) contract lands; keeps the attention
    # seam from engaging in production before it is validated end-to-end.
    import os

    return os.environ.get("OPTIMA_ATTENTION_SEAM") == "1"


def _decode_supported(self, k, v, kwargs) -> bool:
    # The gather MVP supports standard MHA decode only: real k/v, uniform head dim,
    # no MLA-rope-split / cross-attention / sliding window. Anything else -> baseline.
    if k is None or v is None or "k_rope" in kwargs:
        return False
    if getattr(self, "is_cross_attention", False):
        return False
    if getattr(self, "sliding_window_size", -1) not in (-1, 0):
        return False
    return getattr(self, "qk_head_dim", None) == getattr(self, "v_head_dim", None)


def _run_decode_kernel(self, q, k, v, forward_batch, save_kv_cache, impl):
    """Extract this decode step's paged KV, gather it dense, run the miner kernel.

    Mirrors what a stock backend does: store the new token's k/v at ``out_cache_loc``
    (validator-owned write), then gather each request's context via
    ``req_to_token[req_pool_idx, :seq_len]`` and let the miner compute attention.
    """
    Hq = self.tp_q_head_num
    Hkv = self.tp_k_head_num
    D = self.qk_head_dim
    pool = forward_batch.token_to_kv_pool

    # Validator owns the KV-cache write (miner only reads). Store BEFORE gathering so
    # the gathered context includes the current token.
    if save_kv_cache and k is not None:
        pool.set_kv_buffer(self, forward_batch.out_cache_loc,
                           k.view(-1, Hkv, D), v.view(-1, Hkv, D))

    seq_lens = forward_batch.seq_lens
    B = seq_lens.shape[0]
    max_len = int(seq_lens.max().item())  # variable shape -> eager only (see docstring)
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    slots = req_to_token[forward_batch.req_pool_indices][:, :max_len].long()  # (B, max_len)

    k_cache = pool.get_key_buffer(self.layer_id)   # (pool_size, Hkv, D)
    v_cache = pool.get_value_buffer(self.layer_id)
    k_pad = k_cache[slots]                          # (B, max_len, Hkv, D)
    v_pad = v_cache[slots]
    q3 = q.view(B, Hq, D)

    out = torch.empty((B, Hq, D), dtype=q.dtype, device=q.device)
    impl.entry(q3, k_pad, v_pad, seq_lens, float(self.scaling), out)  # miner fills out
    return out.reshape(B, Hq * D)


def make_moe_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slots: tuple[str, ...] = ("moe.fused_experts_mxfp4", "moe.fused_experts"),
) -> Callable[..., object]:
    """Build a replacement for ``FusedMoE.forward`` — the single chokepoint every MoE
    layer funnels through (``sglang.srt.layers.moe.fused_moe_triton.layer``), so the
    seam is backend-agnostic (triton / cutlass sm90 / sm100 / sm120 / marlin all sit
    *below* it).

    MoE experts are a *block*, and a (prepare, forward) one: the validator owns routing
    (``topk_output`` is computed upstream and handed in), owns the expert weights, and
    owns the output allocation; the miner only (1) transforms the raw weights once via
    ``prepare`` (the FP4 repack / scale-interleave / padding) and (2) fills the
    validator-allocated ``out`` each step via ``entry``. The combined expert output
    feeds the residual stream -> downstream layers -> sampler (all stock) — so there is
    no final output to substitute, the same property that makes the op slots safe, and
    no source patch / engine reconfigure (unlike the framework/rebuild path).

    ARCHITECTURE-GENERAL by design: nothing here is sm120-specific. A miner kernel
    declares the arch(es) it supports via eligibility; the dispatcher routes to it only
    on matching hardware and otherwise trusts the baseline. (sm120 was merely the first
    box with a real win; the B200/sm100 endgame uses the identical contract.)

    SCOPE (MVP, mirrors the attention seam): routes the **standard** routing format
    through the ``moe.fused_experts`` contract, **eager-only** and **non-expert-parallel**
    (the ``(M,H)->(M,H)`` contract does not model EP token dispatch/combine), and only
    when ``OPTIMA_MOE_SEAM=1`` (opt-in until validated end-to-end). Tensor-parallel
    experts are supported: the kernel fills this rank's partial ``out`` and the
    validator replays ``FusedMoE.forward_impl``'s TP all-reduce. Bypassed / triton-kernel
    routing formats, EP>1, and CUDA-graph capture all fall back to the trusted backend.
    Conservative by construction: when in doubt, trust the baseline.
    """

    def dispatched(self, hidden_states, topk_output):
        _maybe_inspect_moe(self, hidden_states, topk_output)
        if _moe_seam_active():
            try:
                if _moe_supported(self) and hidden_states.dim() == 2 and not _in_cuda_graph():
                    routed = _standard_topk(topk_output)
                    if routed is not None:
                        x = hidden_states
                        arch = _arch_tag(x.device.index or 0) if x.is_cuda else None
                        for slot in slots:
                            impl = registry.lookup(
                                slot, dtype_name=_dtype_name(x.dtype), last_dim=x.shape[-1], arch=arch
                            )
                            # (prepare, forward) slot: a registered kernel MUST carry
                            # prepare, else we can't honor the contract -> skip it.
                            if impl is not None and impl.prepare is not None:
                                out = _run_moe_kernel(self, x, routed, impl, slot)
                                _log_once_active(slot)
                                return out
            except Exception as exc:  # noqa: BLE001
                if registry.strict:
                    raise
                _log_once_fallback(exc)
                # any mismatch with this sglang's internals -> trust the baseline
        return baseline_forward(self, hidden_states, topk_output)

    return dispatched


_MOE_INSPECTED = False


def _maybe_inspect_moe(self, hidden_states, topk_output) -> None:
    """Debug aid (off unless ``OPTIMA_MOE_INSPECT`` is set): dump the live FusedMoE
    layer's tensors + the topk_output structure ONCE, so seam integration on a new
    model/quant format can be written against the real layout. Writes to the path in
    ``OPTIMA_MOE_INSPECT`` (or /tmp/optima_moe_inspect.txt if set to "1"). Never raises.
    """
    import os

    path = os.environ.get("OPTIMA_MOE_INSPECT")
    if not path:
        return
    global _MOE_INSPECTED
    if _MOE_INSPECTED:
        return
    _MOE_INSPECTED = True
    if path == "1":
        path = "/tmp/optima_moe_inspect.txt"
    try:
        lines = ["=== FusedMoE layer tensors (name: shape dtype) ==="]
        for n in sorted(dir(self)):
            if n.startswith("__"):
                continue
            try:
                v = getattr(self, n)
            except Exception:  # noqa: BLE001
                continue
            t = v if torch.is_tensor(v) else getattr(v, "data", None)
            if torch.is_tensor(t):
                lines.append(f"  {n}: {tuple(t.shape)} {t.dtype}")
        for n in ("moe_tp_size", "moe_ep_size", "reduce_results", "hidden_size",
                  "intermediate_size_per_partition", "num_local_experts", "layer_id"):
            lines.append(f"  .{n} = {getattr(self, n, None)}")
        lines.append(f"  hidden_states: {tuple(hidden_states.shape)} {hidden_states.dtype}")
        fields = getattr(topk_output, "_fields", None)
        lines.append(f"  topk_output: {type(topk_output).__name__} fields={fields}")
        for f in fields or []:
            v = getattr(topk_output, f, None)
            if torch.is_tensor(v):
                lines.append(f"    {f}: {tuple(v.shape)} {v.dtype}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _log_once_active(slot: str) -> None:
    global _MOE_LOGGED_ACTIVE
    if not _MOE_LOGGED_ACTIVE:
        _MOE_LOGGED_ACTIVE = True
        logger.warning("optima: MoE seam ACTIVE — experts routed through miner kernel (slot=%s)", slot)


def _log_once_fallback(exc: Exception) -> None:
    global _MOE_LOGGED_FALLBACK
    if not _MOE_LOGGED_FALLBACK:
        _MOE_LOGGED_FALLBACK = True
        logger.warning("optima: MoE seam FELL BACK to baseline after kernel error: %r", exc)


def _moe_seam_active() -> bool:
    # Opt-in until the seam is validated end-to-end on the pod (graph-safe TP path,
    # quant weight view); keeps it inert in production until then.
    import os

    return os.environ.get("OPTIMA_MOE_SEAM") == "1"


def _moe_supported(self) -> bool:
    # The (M,H)->(M,H) expert contract models local experts only. Expert parallelism
    # adds an all-to-all token dispatch/combine the contract doesn't express, so EP>1
    # falls back to the trusted backend. (Pure tensor-parallel IS supported — see the
    # all-reduce in _run_moe_kernel.)
    if getattr(self, "moe_ep_size", 1) != 1:
        return False
    # Need the dense expert-weight params the (w13, w2) contract hands to prepare().
    if not (hasattr(self, "w13_weight") and hasattr(self, "w2_weight")):
        return False
    # Dense (unquantized) experts only for now: a quantized layer exposes packed FP4/FP8
    # bytes plus separate *_scale params, which the dense contract would mis-read. The
    # quantized win (GPT-OSS MXFP4, the B200 endgame) needs the richer weight view — the
    # explicit next rung — so fall back until that lands rather than feed prepare() bytes.
    if hasattr(self, "w13_weight_scale") or hasattr(self, "w2_weight_scale"):
        return False
    return True


def _in_cuda_graph() -> bool:
    # Eager-only MVP: a kernel captured into a piecewise CUDA graph isn't validated yet.
    # If we can't tell (helper missing), assume NOT in a graph (CPU tests / older sglang);
    # the env opt-in + eager eval keep this safe in practice.
    try:
        from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(is_in_piecewise_cuda_graph())
    except Exception:  # noqa: BLE001
        return False


def _standard_topk(topk_output):
    """Return ``(topk_ids, topk_weights)`` iff routing is already materialized (the
    STANDARD format), else None. BypassedTopKOutput / TritonKernelTopKOutput don't carry
    explicit topk tensors -> fall back (conservative; no implicit re-routing here)."""
    topk_ids = getattr(topk_output, "topk_ids", None)
    topk_weights = getattr(topk_output, "topk_weights", None)
    if topk_ids is None or topk_weights is None:
        return None
    return topk_ids, topk_weights


def _moe_prepared(self, impl, slot):
    """Run the miner's ``prepare`` ONCE on this layer's expert weights, memoized on the
    layer (one bundle per process, fresh process per eval). The slot's
    ``prepare_from_layer`` (validator-owned) maps the live sglang layer to the prepare
    call shape — weights + biases + layout flags — so the miner owns only the transform."""
    if not getattr(self, "_optima_moe_prepared_done", False):
        from optima.slots import get_slot

        spec = get_slot(slot)
        if spec.prepare_from_layer is not None:
            args = spec.prepare_from_layer(self)
        else:
            args = (self.w13_weight.data, self.w2_weight.data)
        self._optima_moe_prepared = impl.prepare(*args)
        self._optima_moe_prepared_done = True
    return self._optima_moe_prepared


def _run_moe_kernel(self, x, routed, impl, slot):
    """Allocate the output (validator-owned), run the miner's fused-experts kernel, then
    replay FusedMoE.forward_impl's tensor-parallel all-reduce. EP is excluded upstream."""
    topk_ids, topk_weights = routed
    prepared = _moe_prepared(self, impl, slot)
    M, H = x.shape[0], x.shape[-1]
    out = torch.empty((M, H), dtype=x.dtype, device=x.device)
    impl.entry(x, topk_ids, topk_weights, prepared, out)  # miner fills `out` (local experts)
    if getattr(self, "reduce_results", False) and getattr(self, "moe_tp_size", 1) > 1:
        # Sum this rank's partial expert output across the TP group (raises if the
        # collective is unavailable -> caller falls back to the trusted baseline).
        from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

        out = tensor_model_parallel_all_reduce(out)
    return out


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dtype, str(dtype).replace("torch.", ""))
