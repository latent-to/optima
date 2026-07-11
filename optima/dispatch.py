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

from optima import audit as _audit
from optima import moe_export as _moe_export
from optima import receipts as _receipts
from optima.capabilities import msa_prefill_call_descriptor
from optima.registry import REGISTRY, KernelRegistry
from optima.slots import get_slot
from optima.tensor_spec import validate_output_spec

logger = logging.getLogger("optima.dispatch")
_MOE_LOGGED_ACTIVE = False
_MOE_LOGGED_FALLBACK = False


def _arch_tag(device_index: int = 0) -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"


def _runtime_parallel_sizes() -> tuple[Optional[int], Optional[int]]:
    """Return validator-observed ``(tp_size, world_size)`` when initialized.

    The MSA function itself carries no model-runner object, so these values come
    only from sglang's already-initialized parallel-state authority.  Import or
    initialization failure means unknown (descriptor fields omitted), never a
    guessed value from environment variables.
    """

    tp_size: Optional[int] = None
    world_size: Optional[int] = None
    try:
        from sglang.srt.distributed import parallel_state as ps

        try:
            tp_size = int(ps.get_tensor_model_parallel_world_size())
        except Exception:  # noqa: BLE001 - parallel state may not be initialized
            pass
        try:
            world_size = int(ps.get_world_size())
        except Exception:  # noqa: BLE001 - parallel state may not be initialized
            pass
    except Exception:  # noqa: BLE001 - CPU/unit environments need no sglang import
        pass
    return tp_size, world_size


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
        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return baseline_forward(self, x)
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
        aud = _audit.sampled()
        a_x = x.clone() if aud else None  # pre-call clone: the kernel may scribble on x
        try:
            impl.entry(x, out)
        except Exception as exc:
            if registry.strict:
                raise
            # Quality/throughput already protect us; a crashing kernel just loses.
            stock = baseline_forward(self, x)
            _receipts.fallback(slot, exc)
            return stock
        if aud:
            _audit.run(slot, (out,), lambda: baseline_forward(self, a_x))
        _receipts.completed(slot)
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
        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return baseline_forward(self, x, residual, post_residual_addition)
        # Rare / semantic-override paths -> trusted baseline (keeps the contract simple
        # & safe): fp32 residual, a variance computed over a prefix subset of the hidden
        # dim (variance_size_override), or HF cast-before-multiply semantics
        # (cast_x_before_out_mul) are all NOT the pure rmsnorm the slot contract states.
        if (post_residual_addition is not None or getattr(self, "fp32_residual", False)
                or getattr(self, "variance_size_override", None) is not None
                or getattr(self, "cast_x_before_out_mul", False)):
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
        aud = _audit.sampled()
        try:
            if residual is None:
                a_x = x.clone() if aud else None
                out = torch.empty_like(x)
                impl.entry(x, weight, out, eps)
                if aud:
                    _audit.run(slot, (out,), lambda: baseline_forward(self, a_x, None, None))
                _receipts.completed(slot)
                return out
            a_x, a_res = (x.clone(), residual.clone()) if aud else (None, None)
            new_residual = x + residual  # validator owns the add
            out = torch.empty_like(new_residual)
            impl.entry(new_residual, weight, out, eps)
            if aud:
                _audit.run(slot, (out, new_residual),
                           lambda: baseline_forward(self, a_x, a_res, None))
            _receipts.completed(slot)
            return out, new_residual
        except Exception as exc:
            if registry.strict:
                raise
            stock = baseline_forward(self, x, residual, post_residual_addition)
            _receipts.fallback(slot, exc)
            return stock

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

    NO IN-ENGINE AUDIT here (optima.audit): re-running ``baseline_forward`` would
    re-drive the backend's KV-cache write path (``save_kv_cache``) — stateful, and
    idempotence is backend-specific, so a double-write could corrupt the run being
    scored. Until a save-free audit call exists, attention fidelity rides verify
    (matched_ratio vs fp32 ground truth) + the benchmark gate.
    """

    def dispatched(self, q, k, v, forward_batch, save_kv_cache: bool = True, **kwargs):
        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return baseline_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)
        selected = False
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
                        selected = True
                        out = _run_decode_kernel(
                            self, q, k, v, forward_batch, save_kv_cache, impl
                        )
                        _receipts.completed(slot)
                        return out
            except Exception as exc:
                if registry.strict:
                    raise
                if selected:
                    stock = baseline_forward(
                        self, q, k, v, forward_batch, save_kv_cache, **kwargs
                    )
                    _receipts.fallback(slot, exc)
                    return stock
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
    # no MLA-rope-split / cross-attention / sliding window. ANY extra kwarg means a
    # contract the dense-gather kernel doesn't model (k_rope = MLA rope split,
    # sinks = gpt-oss attention sinks, ...) — running the kernel anyway would
    # silently drop that piece of the computation. Anything unknown -> baseline.
    if k is None or v is None or kwargs:
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
    slots: tuple[str, ...] = ("moe.fused_experts_reduce", "moe.fused_experts"),
) -> Callable[..., object]:
    """Build a replacement for ``FusedMoE.forward_impl`` — the single chokepoint every MoE
    layer funnels through (``sglang.srt.layers.moe.fused_moe_triton.layer``; ``.forward`` is
    a router that bypasses to ``forward_impl`` under piecewise capture, so ``forward_impl`` is
    the waist — see optima/integrations/sglang_moe.py), so the seam is backend-agnostic (the
    triton / cutlass / marlin MoE backends all sit *below* it).

    MoE experts are a *block*, and a (prepare, forward) one: the validator owns routing
    (``topk_output`` is computed upstream and handed in), owns the expert weights, and
    owns the output allocation; the miner only (1) transforms the raw weights once via
    ``prepare`` (the FP4 repack / scale-interleave / padding) and (2) fills the
    validator-allocated ``out`` each step via ``entry``. The combined expert output
    feeds the residual stream -> downstream layers -> sampler (all stock) — so there is
    no final output to substitute, the same property that makes the op slots safe, and
    no source patch / engine reconfigure (unlike the framework/rebuild path).

    ARCHITECTURE-GENERAL by design: nothing here is hardware- or model-specific. A miner
    kernel declares the arch(es)/dtype(s) it supports via eligibility; the dispatcher
    routes to it only on matching hardware and otherwise trusts the baseline.

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
        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return baseline_forward(self, hidden_states, topk_output)
        _maybe_inspect_moe(self, hidden_states, topk_output)
        selected_slot = None
        route_exc = None
        if _moe_seam_active():
            try:
                if not (_moe_supported(self) and hidden_states.dim() == 2):
                    _moe_debug(lambda: f"SKIP not-supported (ep={getattr(self,'moe_ep_size',1)} "
                                       f"dim={hidden_states.dim()})")
                else:
                    in_graph = _in_cuda_graph()
                    quant_fmt = _moe_quant_format(self)
                    routed = _standard_topk(topk_output)
                    if routed is None:
                        _moe_debug(lambda: f"SKIP non-standard topk_output={type(topk_output).__name__}")
                    else:
                        x = hidden_states
                        arch = _arch_tag(x.device.index or 0) if x.is_cuda else None
                        for slot in slots:
                            impl = registry.lookup(
                                slot, dtype_name=_dtype_name(x.dtype), last_dim=x.shape[-1], arch=arch
                            )
                            # (prepare, forward) slot: a registered kernel MUST carry
                            # prepare, else we can't honor the contract -> skip it.
                            if impl is None or impl.prepare is None:
                                _moe_debug(lambda s=slot: f"SKIP {s}: no impl/prepare (lookup miss)")
                                continue
                            # Under CUDA graphs (the scoring config) only run a kernel the
                            # miner DECLARED graph-capturable; otherwise trust the baseline
                            # in-graph so an un-capturable kernel can't wedge graph capture.
                            if in_graph and not impl.eligibility.graph_safe:
                                _moe_debug(lambda s=slot: f"SKIP {s}: in_graph & not graph_safe")
                                continue
                            # The reduce-owning contract only fits a layer that actually
                            # reduces here: under TP>1 with reduce_results=False the model
                            # defers the reduce downstream (e.g. fuses it after a shared-
                            # expert add), so a kernel that reduces anyway would insert an
                            # EXTRA all-reduce and diverge from stock.
                            if (slot.endswith(".fused_experts_reduce")
                                    and getattr(self, "moe_tp_size", 1) > 1
                                    and not getattr(self, "reduce_results", False)):
                                _moe_debug(lambda s=slot: f"SKIP {s}: layer does not reduce here "
                                                          "(reduce_results=False under TP)")
                                continue
                            # Pair kernel<->layer by quant format: a dense kernel never runs
                            # a quantized layer (it would mis-read packed bytes + scales), and
                            # a quant kernel never runs a dense layer. Mismatch -> baseline.
                            if not _quant_ok(impl.eligibility.quant, quant_fmt):
                                _moe_debug(lambda s=slot: f"SKIP {s}: quant mismatch "
                                                          f"(kernel={set(impl.eligibility.quant)} layer={quant_fmt})")
                                continue
                            selected_slot = slot
                            # Audit: baseline forward_impl on a pre-call clone (its TP
                            # reduce is collective — rank-seeded sampling keeps lockstep).
                            # Both sides are post-reduce here (the kernel path replays the
                            # validator reduce for plain fused_experts), so comparable.
                            aud = not in_graph and _audit.sampled()
                            a_x = x.clone() if aud else None
                            out = _run_moe_kernel(self, x, routed, impl, slot)
                            if aud:
                                _audit.run(slot, (out,) if torch.is_tensor(out) else tuple(out),
                                           lambda: baseline_forward(self, a_x, topk_output))
                            _log_once_active(slot)
                            _moe_debug(lambda s=slot, m=x.shape[0]: f"FIRED slot={s} M={m} quant={quant_fmt}")
                            _receipts.completed(slot)
                            return out
            except Exception as exc:  # noqa: BLE001
                if registry.strict:
                    raise
                if selected_slot is not None:
                    route_exc = exc
                _log_once_fallback(exc)
                _moe_debug(lambda e=exc: f"FELL BACK after kernel error: {e!r}")
                # any mismatch with this sglang's internals -> trust the baseline
        stock = baseline_forward(self, hidden_states, topk_output)
        if route_exc is not None:
            _receipts.fallback(selected_slot, route_exc)
        return stock

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


_MOE_DEBUG_SEEN: set = set()


def _moe_debug(msg) -> None:
    """Print a one-time-per-message MoE-seam decision to stderr when OPTIMA_MOE_DEBUG=1.

    A definitive observability hook for the spawned scheduler child, where the optima logger's
    records don't always reach the captured stream. ``msg`` is a thunk so the (cheap) f-string
    only runs when debugging is on. Inert otherwise."""
    import os
    import sys

    if os.environ.get("OPTIMA_MOE_DEBUG") != "1":
        return
    try:
        m = msg() if callable(msg) else str(msg)
    except Exception:  # noqa: BLE001
        return
    if m in _MOE_DEBUG_SEEN:
        return
    _MOE_DEBUG_SEEN.add(m)
    print(f"[optima.moe] {m}", file=sys.stderr, flush=True)


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
    # Need the expert-weight params prepare_from_layer maps to the kernel's weight view.
    if not (hasattr(self, "w13_weight") and hasattr(self, "w2_weight")):
        return False
    # Quantized layers (FP4/FP8 — they expose *_weight_scale) ARE admitted here and
    # paired to a kernel BY FORMAT in the dispatch loop (_quant_ok): a quantized layer
    # only runs a kernel that DECLARES its exact format (and gets the quant-aware
    # prepare_from_layer); a dense layer still runs only a dense kernel. The format gate
    # is per-impl, so it lives in the loop — _moe_supported runs before the impl is known.
    return True


def _moe_quant_format(self) -> Optional[str]:
    """The layer's expert-weight quant format: ``None`` (dense), ``"fp8"``, or ``"nvfp4"``.

    A quantized FusedMoE exposes ``*_weight_scale`` params; the format is read off the
    packed weight dtype (float8 -> ``"fp8"``; sub-byte packed -> ``"nvfp4"``). The
    dispatcher pairs this to a kernel's declared ``Eligibility.quant`` (see ``_quant_ok``).
    The EXACT NVFP4 scale-param layout a kernel's prepare consumes is pinned against the
    live layer via ``OPTIMA_MOE_INSPECT`` (the dump hook above) — this only classifies."""
    if not (hasattr(self, "w13_weight_scale") or hasattr(self, "w2_weight_scale")):
        return None
    w = getattr(getattr(self, "w13_weight", None), "data", None)
    dt = getattr(w, "dtype", None)
    if dt in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        return "fp8"
    return "nvfp4"


def _quant_ok(declared: frozenset[str], fmt: Optional[str]) -> bool:
    """Pair a kernel's declared quant formats to the layer's format. Conservative: a DENSE
    layer (``fmt is None``) runs only a dense kernel (no declared formats); a QUANTIZED
    layer runs only a kernel that declares its exact format. Never cross — a mismatch falls
    back to the trusted baseline rather than mis-read the weights."""
    if fmt is None:
        return not declared
    return fmt in declared


def _dynamo_compiling() -> bool:
    """True while torch.compile (Dynamo) is TRACING the caller.

    Newer sglang compiles some seam call sites piecewise (observed 2026-07-07: the
    prefill fusion path traces ``flashinfer_allreduce_residual_rmsnorm`` — i.e. our
    rebound dispatcher — under Dynamo, which hard-errors on the dispatcher's
    function-body imports/registry machinery). A traced region must bake PURE STOCK:
    Dynamo constant-folds ``torch.compiler.is_compiling()`` to True during trace, so
    an early ``return baseline_fn(...)`` erases the dispatcher from the compiled graph
    entirely. Eager execution and classic CUDA-graph capture — the validated decode
    win regime — see False and route normally. A slot that wants to live INSIDE a
    compiled region needs the custom-op wrapper design (ledger, not built yet).
    """
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001 - older torch without torch.compiler
        return False


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
        _maybe_free_dense_weights(self)
    return self._optima_moe_prepared


def _maybe_free_dense_weights(self) -> None:
    """OPT-IN (``OPTIMA_MOE_FREE_DENSE=1``): release this layer's dense bf16 expert
    weights once the miner kernel holds its own (packed/quantized) copies — the dequantized
    originals are dead weight (~4x the size of the packed copies). This lowers
    steady-state residency so the engine can run at a higher mem_fraction.

    SAFETY: only valid in FULL eager (disable_cuda_graph AND disable_piecewise_cuda_graph)
    where every forward routes to the kernel; a fallback after freeing would hit empty
    weights (it fails loudly, never silently). Left OFF by default — the supported path
    is to give ``prepare`` GPU headroom via mem_fraction (the eval uses ~0.6)."""
    import os

    if os.environ.get("OPTIMA_MOE_FREE_DENSE") != "1":
        return
    for attr in ("w13_weight", "w2_weight", "w13_weight_bias", "w2_weight_bias"):
        p = getattr(self, attr, None)
        data = getattr(p, "data", None)
        if data is not None:
            p.data = torch.empty(0, device=data.device, dtype=data.dtype)
    torch.cuda.empty_cache()


def _run_moe_kernel(self, x, routed, impl, slot):
    """Allocate the output (validator-owned) and run the miner's fused-experts kernel.

    Two contracts share this path:
      * ``moe.fused_experts`` — miner fills the LOCAL expert output; the validator then
        replays FusedMoE.forward_impl's tensor-parallel all-reduce (the reduce is stock).
      * ``moe.fused_experts_reduce`` — the miner kernel OWNS the trailing all-reduce (it is
        handed the TP process group), so it can overlap the expert GEMM with the reduce; the
        validator does NOT replay it. This is the only contract that can express the
        compute-comm overlap win. EP is excluded upstream for both.
    """
    topk_ids, topk_weights = routed
    prepared = _moe_prepared(self, impl, slot)
    M, H = x.shape[0], x.shape[-1]
    out = torch.empty((M, H), dtype=x.dtype, device=x.device)

    if slot.endswith(".fused_experts_reduce"):
        # The kernel does experts AND the cross-rank reduce. Hand it the TP group; do not
        # replay a second reduce. If there's no real TP group (single rank), the kernel's
        # reduce is a no-op and the local output is already the answer.
        group = _tp_device_group()
        impl.entry(x, topk_ids, topk_weights, prepared, out, group)  # miner fills the REDUCED out
        return out

    impl.entry(x, topk_ids, topk_weights, prepared, out)  # miner fills `out` (local experts)
    if getattr(self, "reduce_results", False) and getattr(self, "moe_tp_size", 1) > 1:
        # Sum this rank's partial expert output across the TP group (raises if the
        # collective is unavailable -> caller falls back to the trusted baseline).
        from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

        out = tensor_model_parallel_all_reduce(out)
    return out


def _tp_device_group():
    """The tensor-parallel process group to hand a reduce-owning MoE kernel.

    Returns the live TP ``device_group`` (NCCL/gloo) or None if TP isn't initialized
    (single-rank dev/CPU runs), in which case the miner kernel's reduce is a no-op."""
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        return getattr(get_tp_group(), "device_group", None)
    except Exception:  # noqa: BLE001 - no TP group available -> single-rank semantics
        return None


_COLLECTIVE_LOGGED_ACTIVE = False
_COLLECTIVE_LOGGED_FALLBACK = False


def make_allreduce_dispatcher(
    baseline_all_reduce: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "collective.all_reduce",
) -> Callable[..., object]:
    """Build a replacement for ``GroupCoordinator.all_reduce`` — the single chokepoint
    every tensor-parallel reduce funnels through (``sglang.srt.distributed.parallel_state``).

    The TP all-reduce is the largest single category of decode time at scale (~32–43%),
    and it is latency-bound — so the lever is a lower-latency reduce or a compute-comm
    overlap, both expressible here. The validator owns the output buffer, the process
    group, and the call site; the miner only fills ``out`` with the cross-rank sum. The
    reduce is mid-network (upstream of the sampler) → no output to substitute.

    SCOPE (MVP, mirrors the other seams): only the multi-rank (``world_size > 1``)
    default SUM all-reduce of a 2D tensor, opt-in via ``OPTIMA_COLLECTIVE_SEAM=1``.
    Extra args/kwargs or single-rank → trusted baseline. Under CUDA-graph capture
    (the scoring config) a kernel runs only if it DECLARED ``graph_safe``; an
    undeclared kernel stays eager-only and the stock reduce runs in-graph. The miner gets
    the process group (``self.device_group``) — a wider capability than op/block slots —
    so this slot is verified DISTRIBUTED (optima.verify_collective) and the end-to-end
    gate is mandatory (docs/SLOT_CONTRACT.md).
    """

    def dispatched(self, input_, *args, **kwargs):
        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return baseline_all_reduce(self, input_, *args, **kwargs)
        selected = False
        route_exc = None
        if _collective_seam_active() and not args and not kwargs:
            try:
                if (torch.is_tensor(input_) and input_.dim() == 2
                        and getattr(self, "world_size", 1) > 1):
                    impl = registry.lookup(
                        slot,
                        dtype_name=_dtype_name(input_.dtype),
                        last_dim=input_.shape[-1],
                        arch=_arch_tag(input_.device.index or 0) if input_.is_cuda else None,
                    )
                    # Under CUDA graphs (the scoring config) only run a kernel the miner
                    # DECLARED graph-capturable; else trust the stock reduce in-graph.
                    if impl is not None and not (_in_cuda_graph() and not impl.eligibility.graph_safe):
                        selected = True
                        # Audited baseline is COLLECTIVE (see arfusion note): rank-seeded
                        # sampling + lockstep dispatch make the extra reduce safe.
                        aud = not _in_cuda_graph() and _audit.sampled()
                        a_in = input_.clone() if aud else None
                        out = torch.empty_like(input_)
                        group = getattr(self, "device_group", None)
                        impl.entry(input_, out, group)  # miner fills out with sum-over-ranks
                        if aud:
                            _audit.run(slot, (out,), lambda: baseline_all_reduce(self, a_in))
                        _log_collective_active()
                        _receipts.completed(slot)
                        return out
            except Exception as exc:  # noqa: BLE001
                if registry.strict:
                    raise
                if selected:
                    route_exc = exc
                _log_collective_fallback(exc)
        stock = baseline_all_reduce(self, input_, *args, **kwargs)
        if route_exc is not None:
            _receipts.fallback(slot, route_exc)
        return stock

    return dispatched


def _collective_seam_active() -> bool:
    import os

    return os.environ.get("OPTIMA_COLLECTIVE_SEAM") == "1"


def _log_collective_active() -> None:
    global _COLLECTIVE_LOGGED_ACTIVE
    if not _COLLECTIVE_LOGGED_ACTIVE:
        _COLLECTIVE_LOGGED_ACTIVE = True
        logger.warning("optima: collective.all_reduce seam ACTIVE — TP reduce routed through miner kernel")


def _log_collective_fallback(exc: Exception) -> None:
    global _COLLECTIVE_LOGGED_FALLBACK
    if not _COLLECTIVE_LOGGED_FALLBACK:
        _COLLECTIVE_LOGGED_FALLBACK = True
        logger.warning("optima: collective.all_reduce seam FELL BACK to baseline after kernel error: %r", exc)


# ---------------------------------------------------------------------------
# collective.ar_residual_rmsnorm — the fused AR+residual+RMSNorm epilogue waist
# ---------------------------------------------------------------------------

_ARFUSION_LOGGED_ACTIVE = False
_ARFUSION_LOGGED_FALLBACK = False


def make_arfusion_dispatcher(
    baseline_fn: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "collective.ar_residual_rmsnorm",
) -> Callable[..., object]:
    """Build a replacement for the MODULE-LEVEL function
    ``sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce_residual_rmsnorm``
    — sglang's own fused-epilogue waist. With ``--enable-flashinfer-allreduce-fusion``
    (an arena server flag) every participating layer epilogue funnels through this one
    function: the layer defers its TP all-reduce, and the next norm call performs
    AR + residual-add + RMSNorm fused. The call site resolves the symbol per call via a
    function-local import, so rebinding the module attribute reroutes every caller
    (the mechanism the 2026-07-02 M3 fused-epilogue campaign validated in production).

    The validator owns the call site, BOTH output buffers (norm_out, new_residual), and
    the process group; the miner owns the reduce transport + the fused add/norm math.
    Mid-network, upstream of the sampler — nothing to substitute. Stock signature and
    the ``Tuple[Tensor, Tensor]`` return are preserved exactly; any deviation from the
    plain path (extra semantics via kwargs, missing residual, non-2D input) falls back.

    SCOPE: 2D input with a residual, multi-rank group, opt-in via
    ``OPTIMA_ARFUSION_SEAM=1``. Token-count dispatch windows (a kernel measured to win
    only at decode-sized T) are declared via eligibility ``max_num_tokens`` — oversized
    calls (prefill) route to the trusted baseline rather than trusting the kernel to
    decline. Under CUDA-graph capture a kernel runs only if it declared ``graph_safe``.
    """

    def dispatched(input_tensor, residual, weight, eps=1e-6, max_token_num=2048,
                   use_oneshot=None, trigger_completion_at_end=False, fp32_acc=False,
                   use_attn_tp_group=True):
        # FIRST, before any Python machinery: inside a Dynamo trace this constant-folds
        # to True and the compiled piece bakes pure stock (see _dynamo_compiling — the
        # piecewise-prefill trace of this exact call site hard-errored otherwise).
        if _dynamo_compiling():
            return baseline_fn(input_tensor, residual, weight, eps, max_token_num,
                               use_oneshot, trigger_completion_at_end, fp32_acc,
                               use_attn_tp_group)
        if _arfusion_seam_active():
            # DEEP consume first: if this call's input is a moe output whose in-op
            # finalize was skipped (ptr-keyed pend from the export seam), the tensor
            # is UNFINALIZED — it must never reach the shallow kernel or the stock
            # baseline directly. _deep_consume always returns a finalized result
            # (miner deep kernel, or trusted fp32 reconstruct + stock fusion).
            if _moe_export.has_pends():
                exp = _moe_export.consume(input_tensor)
                if exp is not None:
                    return _deep_consume(
                        exp, input_tensor, residual, weight, eps, max_token_num,
                        use_oneshot, trigger_completion_at_end, fp32_acc,
                        use_attn_tp_group, registry=registry, baseline_fn=baseline_fn)
            selected = False
            route_exc = None
            try:
                # Contiguity guard = STOCK PARITY: the stock function refuses
                # non-contiguous input/residual/weight (real call sites pass views —
                # upstream guards for it, flashinfer_comm_fusion.py). A raw-pointer
                # kernel fed a strided view reads the wrong layout silently; verify
                # can't see it (it always builds contiguous tensors), only the
                # engine's own call mix does.
                if (torch.is_tensor(input_tensor) and input_tensor.dim() == 2
                        and torch.is_tensor(residual)
                        and input_tensor.is_contiguous() and residual.is_contiguous()
                        and (not torch.is_tensor(weight) or weight.is_contiguous())):
                    impl = registry.lookup(
                        slot,
                        dtype_name=_dtype_name(input_tensor.dtype),
                        last_dim=input_tensor.shape[-1],
                        arch=_arch_tag(input_tensor.device.index or 0) if input_tensor.is_cuda else None,
                        num_tokens=input_tensor.shape[0],
                    )
                    # Under CUDA graphs (the scoring config) only run a kernel the miner
                    # DECLARED graph-capturable; else trust the stock path in-graph.
                    if impl is not None and not (_in_cuda_graph() and not impl.eligibility.graph_safe):
                        group = _arfusion_group(use_attn_tp_group)
                        if group is not None and group.size() > 1:
                            selected = True
                            # The audited baseline is COLLECTIVE: safe only because the
                            # sampling RNG is rank-identically seeded (audit.py) and all
                            # ranks reach this dispatcher in lockstep; never under capture.
                            aud = not _in_cuda_graph() and _audit.sampled()
                            if aud:
                                a_x, a_res = input_tensor.clone(), residual.clone()
                            out_norm = torch.empty_like(input_tensor)
                            out_residual = torch.empty_like(residual)
                            impl.entry(input_tensor, residual, weight, float(eps),
                                       out_norm, out_residual, group)
                            if aud:
                                _audit.run(slot, (out_norm, out_residual),
                                           lambda: baseline_fn(a_x, a_res, weight, eps,
                                                               max_token_num, use_oneshot,
                                                               trigger_completion_at_end,
                                                               fp32_acc, use_attn_tp_group))
                            _log_arfusion_active()
                            _receipts.completed(slot)
                            return out_norm, out_residual
            except Exception as exc:  # noqa: BLE001
                if registry.strict:
                    raise
                if selected:
                    route_exc = exc
                _log_arfusion_fallback(exc)
            stock = baseline_fn(
                input_tensor, residual, weight, eps, max_token_num, use_oneshot,
                trigger_completion_at_end, fp32_acc, use_attn_tp_group
            )
            if route_exc is not None:
                _receipts.fallback(slot, route_exc)
            return stock
        return baseline_fn(input_tensor, residual, weight, eps, max_token_num,
                           use_oneshot, trigger_completion_at_end, fp32_acc,
                           use_attn_tp_group)

    return dispatched


def _arfusion_seam_active() -> bool:
    import os

    return os.environ.get("OPTIMA_ARFUSION_SEAM") == "1"


_DEEP_SLOT = "collective.moe_finalize_ar_rmsnorm"


def _deep_consume(exp, input_tensor, residual, weight, eps, max_token_num,
                  use_oneshot, trigger_completion_at_end, fp32_acc,
                  use_attn_tp_group, *, registry, baseline_fn):
    """Serve a skipped-finalize moe export: run the deep kernel when eligible, else
    reconstruct with the validator-trusted fp32 finalize and hand the FINALIZED
    tensor to the stock fusion call. Correctness invariant: every path out of here
    has performed the finalize — the unfinalized input never leaks downstream.

    The consume call may HEAD-TRIM CUDA-graph batch padding (same data_ptr,
    T <= exp["T"]); more rows than were exported means the pointer pairing is
    broken and nothing recoverable exists — that one raises."""
    t = input_tensor.shape[0] if input_tensor.dim() == 2 else -1
    if t < 0 or t > exp["T"]:
        raise RuntimeError(
            f"optima deep seam: consume T={t} exceeds export T={exp['T']} — "
            "pointer pairing broken, refusing to serve an unfinalized output")
    selected = False
    route_exc = None
    try:
        impl = registry.lookup(
            _DEEP_SLOT,
            dtype_name=_dtype_name(input_tensor.dtype),
            last_dim=input_tensor.shape[-1],
            arch=_arch_tag(input_tensor.device.index or 0) if input_tensor.is_cuda else None,
            num_tokens=t,
        )
        if impl is not None and not (_in_cuda_graph() and not impl.eligibility.graph_safe):
            group = _arfusion_group(use_attn_tp_group)
            if group is not None and group.size() > 1:
                selected = True
                gemm_out, row_map, scales = _moe_export.export_views(
                    exp, input_tensor.device)
                # Collective audit: rank-identical sampling (audit.py) keeps the
                # reference all-reduce in lockstep; never under capture. The deep
                # slot has no runnable stock function (the stock finalize was
                # skipped), so 'expected' is the slot's own trusted fp32 math —
                # the SAME reference verify gates against.
                aud = not _in_cuda_graph() and _audit.sampled()
                if aud:
                    a_inputs = {"gemm_out": gemm_out.clone(), "row_map": row_map.clone(),
                                "scales": scales.clone(), "residual": residual.clone(),
                                "weight": weight, "eps": eps}
                out_norm = torch.empty_like(input_tensor)
                out_residual = torch.empty_like(residual)
                impl.entry(gemm_out, row_map, scales, residual, weight, float(eps),
                           out_norm, out_residual, group)
                if aud:
                    def _reference():
                        import torch.distributed as dist

                        from optima.slots import (_ar_norm_reference_from_sum,
                                                  _moe_fin_local_finalize)

                        part = _moe_fin_local_finalize(a_inputs)
                        dist.all_reduce(part, group=group)
                        return _ar_norm_reference_from_sum(a_inputs, part, None)

                    _audit.run(_DEEP_SLOT, (out_norm, out_residual), _reference)
                _log_arfusion_active()
                _receipts.completed(_DEEP_SLOT)
                return out_norm, out_residual
    except Exception as exc:  # noqa: BLE001
        if registry.strict:
            raise
        if selected:
            route_exc = exc
        _log_arfusion_fallback(exc)
    # Trusted recovery: fp32 finalize from the exported views (head-trimmed to this
    # call's T), then the stock fusion path on the now-FINALIZED tensor. Correct but
    # slow — receipted as an orphan so a nonzero count is visible seam-health data.
    finalized = _moe_export.trusted_finalize(exp, input_tensor)
    _moe_export.orphaned(exp)
    stock = baseline_fn(finalized, residual, weight, eps, max_token_num, use_oneshot,
                        trigger_completion_at_end, fp32_acc, use_attn_tp_group)
    if route_exc is not None:
        _receipts.fallback(_DEEP_SLOT, route_exc)
    return stock


# (The 2026-07-07 one-off "stockcheck" diagnostic that lived here was productized
# into optima/audit.py — the in-engine audit is the same mechanism, generic across
# dispatchers, receipted, and gated by the eval driver.)


def make_msa_prefill_dispatcher(
    baseline_fn: Callable[..., object],
    module: object,
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "attention.msa_prefill_block_score",
) -> Callable[..., object]:
    """Build a replacement for the MODULE-LEVEL function
    ``...minimax_sparse_ops.prefill.flash_with_topk_idx.flash_prefill_with_topk_index``
    — the MSA arena's prefill-side indexer waist (the score kernel alone is ~30% of
    long-context serving prefill; the 2026-07-10 M3 campaign's +19.6%/+22.4% e2e lever).

    The cheat-resistant split (docs/SLOT_CONTRACT.md, same as the decode MSA slot): the
    miner fills the per-(row, block) SCORE SHEET only; the validator keeps the stock
    top-k selection kernel AND the attend over the chosen blocks, so the kernel stays
    strictly upstream of the sampler and a wrong score merely mis-selects (the
    topk_overlap op-gate + the e2e gate catch it).

    SCOPE (production MSA config only; anything else -> stock verbatim): opt-in via
    ``OPTIMA_MSA_PREFILL_SEAM=1``; ``disable_index_value`` (the stock kernel's ONLY
    output is then the score slab), ``score_type == "max"``, no sink, ONE index-K head.
    Never under Dynamo tracing or CUDA-graph capture (this path does a host sync for
    the per-request metadata; serving prefill runs eager). Any exception -> announced
    whole-batch stock fallback (partial slab writes are overwrite-safe: both kernels
    share the -inf masking convention on the same pre-filled slab).

    ``module`` is the (M3-arena-pinned) target module itself: the stock top-k tail is
    reproduced from its OWN pieces (``get_cu_seqblocks``, ``_topk_index_kernel``,
    ``robust_allocator``) so selection stays byte-stock. Version-pinned glue by design
    — this module only exists on the MSA arena's pinned sglang build.

    AUDIT: wired (optima/audit.py, topk_overlap mode). On sampled calls the STOCK
    function runs FIRST on the still-pristine inputs (no pre-call clones are possible
    here — the gathered K comes from the multi-GB KV pool), then the miner path runs,
    and the audit compares the CONSUMED product: the selection rows our
    validator-owned selector produced from miner scores vs the rows stock produced —
    per-row set overlap gated at the slot's own min_overlap. Only the untimed quality
    launch arms sampling (OPTIMA_SLOT_AUDIT); timed launches carry zero overhead.
    """
    import os

    # Resolve once at binding installation, not in the serving hot loop.  The
    # logical shape is still resolved per invocation below, but dtype/layout/stride
    # policy now comes from the same validator-owned declaration as offline verify.
    output_slot = get_slot(slot)

    def dispatched(q, k_cache, v_cache, sink, req_to_token, slot_ids, cu_seqlens,
                   seq_lens, prefix_lens, max_seqlen_q, max_seqlen_k, block_size_q,
                   block_size_k, topk, init_blocks=1, local_blocks=2, sm_scale=None,
                   use_tma=False, score_type="max", disable_index_value=False,
                   cu_seqblocks_q=None, max_seqblock_q=None, all_seqblock_q=None):
        def stock():
            return baseline_fn(q, k_cache, v_cache, sink, req_to_token, slot_ids,
                               cu_seqlens, seq_lens, prefix_lens, max_seqlen_q,
                               max_seqlen_k, block_size_q, block_size_k, topk,
                               init_blocks, local_blocks, sm_scale, use_tma,
                               score_type, disable_index_value, cu_seqblocks_q,
                               max_seqblock_q, all_seqblock_q)

        if _dynamo_compiling():  # traced region bakes pure stock (see _dynamo_compiling)
            return stock()
        if os.environ.get("OPTIMA_MSA_PREFILL_SEAM") != "1" or _in_cuda_graph():
            return stock()
        selected = False
        try:
            num_kv_heads = k_cache.shape[1]
            contract_top_k = int(output_slot.correctness.top_k)
            if not (disable_index_value and score_type == "max" and sink is None
                    and num_kv_heads == 1 and q.is_cuda and q.dim() == 3
                    and int(topk) == contract_top_k):
                return stock()
            total_q, num_heads, head_dim = q.shape
            batch_size = cu_seqlens.shape[0] - 1
            scale = float(sm_scale if sm_scale is not None else head_dim ** -0.5) * 1.4426950409

            # ONE host sync for the batch metadata (eager prefill path; mirrors the
            # stock wrapper's own reliance on host-known max_seqlen_q/batch_size).
            cu = cu_seqlens[: batch_size + 1].cpu()
            sls = seq_lens[:batch_size].cpu()
            pls = prefix_lens[:batch_size].cpu()
            sids = slot_ids[:batch_size].cpu()

            # Pre-scan BEFORE writing anything: the slot's causal convention (key n
            # visible to row m iff n <= prefix+m) requires seq == prefix + q_len.
            meta = []
            for b in range(batch_size):
                qs, qe = int(cu[b]), int(cu[b + 1])
                seq_b, pre_b = int(sls[b]), int(pls[b])
                if qe - qs > 0 and seq_b != pre_b + (qe - qs):
                    return stock()
                meta.append((b, qs, qe, seq_b, pre_b, int(sids[b])))

            # Describe and select EVERY call before allocating/writing the shared
            # score slab.  A mixed batch is atomic at this binding: if even one
            # request/head lies outside all variants (or matches ambiguously), the
            # whole batch runs byte-stock and no miner fallback is consulted.
            architecture = _arch_tag(q.device.index or 0)
            tp_size, world_size = _runtime_parallel_sizes()
            planned: dict[int, tuple[object, object]] = {}
            for b, qs, qe, seq_b, _pre_b, _sid in meta:
                q_len_b = qe - qs
                if q_len_b == 0 or seq_b == 0:
                    continue
                # The seam invokes one head at a time, but every head in this
                # request has the same canonical descriptor. Select once and reuse
                # the immutable implementation rather than multiplying Python
                # capability matching by num_heads.
                descriptor = msa_prefill_call_descriptor(
                    dtype=_dtype_name(q.dtype),
                    architecture=architecture,
                    head_dim=head_dim,
                    block_size=int(block_size_k),
                    q_len=q_len_b,
                    kv_len=seq_b,
                    top_k=contract_top_k,
                    num_kv_heads=num_kv_heads,
                    tp_size=tp_size,
                    world_size=world_size,
                )
                decision = registry.select(
                    slot, descriptor, write_fired_receipt=False
                )
                if not decision.use_candidate:
                    return stock()
                planned[b] = (descriptor, decision.impl)
            if not planned:
                return stock()

            # Commit routing only after the complete batch passed preflight.  This
            # writes the slot-level "fired" receipt without reintroducing a partial
            # batch: registry state is immutable after engine initialization, and a
            # changed/ambiguous decision still fails closed to stock.
            first_descriptor, first_impl = next(iter(planned.values()))
            committed = registry.select(slot, first_descriptor)
            if committed.impl is not first_impl:
                return stock()
            selected = True

            # In-engine audit (per-rank independent here — the indexer is not a
            # collective): stock runs FIRST on pristine inputs; the comparison target
            # is the selection the engine would actually consume.
            aud = _audit.sampled()
            expected_idx = None
            if aud:
                exp = stock()
                expected_idx = (exp[1] if isinstance(exp, (tuple, list)) and len(exp) > 1
                                else None)

            max_seqblock_k = (max_seqlen_k + block_size_k - 1) // block_size_k
            probe_contract = output_slot.output_contract({
                "q": q[:1, 0, :],
                "index_k": k_cache[:1, 0, :],
                "prefix_len": 0,
                "scale": scale,
                "block_size": block_size_k,
            })
            if len(probe_contract.outputs) != 1:
                raise RuntimeError(f"{slot} must declare exactly one score output")
            score_tensor_spec = probe_contract.outputs[0]
            score_dtype = score_tensor_spec.dtype or q.dtype
            score_device = score_tensor_spec.device or q.device
            score = torch.full((num_heads, total_q, max_seqblock_k), float("-inf"),
                               dtype=score_dtype, device=score_device)

            contract_validated = False
            candidate_calls = 0
            for b, qs, qe, seq_b, pre_b, sid in meta:
                if qe - qs == 0 or seq_b == 0:
                    continue
                nblk_b = (seq_b + block_size_k - 1) // block_size_k
                # Gather-first: paged index-K -> contiguous (S, D). Paging layout is
                # the validator's, not part of the miner contract.
                slots_b = req_to_token[sid, :seq_b].to(torch.long)
                kg = k_cache[slots_b, 0, :].contiguous()
                for h in range(num_heads):
                    _descriptor, impl = planned[b]
                    q_bh = q[qs:qe, h, :].contiguous()
                    out_view = score[h, qs:qe, :nblk_b]
                    if not contract_validated:
                        # One representative slice is enough: every per-head/request
                        # view comes from this same slab with the same dtype, device,
                        # layout and row pitch; logical shapes are constructed directly
                        # from the validated metadata above.  Avoid per-head hot-path tax.
                        live_inputs = {
                            "q": q_bh,
                            "index_k": kg,
                            "prefix_len": pre_b,
                            "scale": scale,
                            "block_size": block_size_k,
                        }
                        validate_output_spec(
                            output_slot.output_contract(live_inputs),
                            [out_view],
                            fallback_dtype=q.dtype,
                            fallback_device=q.device,
                            inputs=(q_bh, kg),
                        )
                        contract_validated = True
                    impl.entry(q_bh, kg, pre_b, scale, block_size_k, out_view)
                    candidate_calls += 1

            if not candidate_calls:
                raise RuntimeError("committed MSA route executed no candidate calls")

            # The validator-owned top-k tail, byte-stock from the target module's own
            # pieces: the SELECTION never comes from miner code.
            import triton  # noqa: PLC0415 - deferred; only on the GPU path

            triton.set_allocator(module.robust_allocator)
            if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
                cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = (
                    module.get_cu_seqblocks(cu_seqlens, max_seqlen_q, block_size_q,
                                            block_size_k))
            topk_idx = torch.full((num_heads, all_seqblock_q, topk), fill_value=-1,
                                  device=score.device, dtype=torch.int32)
            grid = (max_seqblock_q, batch_size, num_heads)
            module._topk_index_kernel[grid](
                score, topk_idx, block_size_q, block_size_k, cu_seqlens,
                cu_seqblocks_q, prefix_lens, topk, init_blocks, local_blocks,
                score.stride(0), score.stride(1), score.stride(2),
                topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
                MASK_INIT=False, MASK_LOCAL=False,
            )
            if aud:
                if expected_idx is None:
                    _audit.baseline_refused(slot)
                else:
                    _audit.record(slot, (topk_idx,), (expected_idx,))
            _log_msa_prefill_active()
            if candidate_calls:
                _receipts.completed(slot)
            return None, topk_idx  # o is None in the gated (disable_index_value) mode
        except Exception as exc:  # noqa: BLE001
            if registry.strict:
                raise
            _log_msa_prefill_fallback(exc)
            stock_result = stock()
            if selected:
                _receipts.fallback(slot, exc)
            return stock_result

    return dispatched


_MSA_PREFILL_LOGGED_ACTIVE = False
_MSA_PREFILL_LOGGED_FALLBACK = False


def _log_msa_prefill_active() -> None:
    global _MSA_PREFILL_LOGGED_ACTIVE
    if not _MSA_PREFILL_LOGGED_ACTIVE:
        _MSA_PREFILL_LOGGED_ACTIVE = True
        logger.warning(
            "optima: attention.msa_prefill_block_score seam ACTIVE — prefill indexer scores routed through miner kernel")


def _log_msa_prefill_fallback(exc: Exception) -> None:
    global _MSA_PREFILL_LOGGED_FALLBACK
    if not _MSA_PREFILL_LOGGED_FALLBACK:
        _MSA_PREFILL_LOGGED_FALLBACK = True
        logger.warning(
            "optima: attention.msa_prefill_block_score seam FELL BACK to baseline after kernel error: %r", exc)


def _arfusion_group(use_attn_tp_group: bool):
    """The torch ProcessGroup the stock call would reduce over. ``use_attn_tp_group``
    mirrors the stock argument (attention-TP vs full-TP chain); under plain TP the two
    coincide. Resolution failure -> None -> baseline (never guess a group)."""
    from sglang.srt.distributed import parallel_state as ps

    try:
        coord = ps.get_attention_tp_group() if use_attn_tp_group else ps.get_tp_group()
    except (AttributeError, AssertionError):
        coord = ps.get_tp_group()
    return getattr(coord, "device_group", None)


def _log_arfusion_active() -> None:
    global _ARFUSION_LOGGED_ACTIVE
    if not _ARFUSION_LOGGED_ACTIVE:
        _ARFUSION_LOGGED_ACTIVE = True
        logger.warning(
            "optima: collective.ar_residual_rmsnorm seam ACTIVE — fused AR+norm epilogue routed through miner kernel")


def _log_arfusion_fallback(exc: Exception) -> None:
    global _ARFUSION_LOGGED_FALLBACK
    if not _ARFUSION_LOGGED_FALLBACK:
        _ARFUSION_LOGGED_FALLBACK = True
        logger.warning(
            "optima: collective.ar_residual_rmsnorm seam FELL BACK to baseline after kernel error: %r", exc)


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dtype, str(dtype).replace("torch.", ""))
