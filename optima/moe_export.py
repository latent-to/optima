"""Deep-seam export/consume state — the validator side of the fe_export protocol.

The deep fused-epilogue slot (``collective.moe_finalize_ar_rmsnorm``) lets ONE miner
kernel own MoE finalize + all-reduce + residual-add + RMSNorm. That requires the
engine to (a) SKIP the in-op standalone finalize inside flashinfer's fused-moe launcher
and (b) hand the pre-finalize tensors to the deferred-AR call site two chokepoints
later. The flashinfer side ships as a bundle ``dep_patches`` entry (built into an
overlay by the reviewed patcher); THIS module is the sglang-side state machine the two
validator-owned seam integrations share:

  * ``sglang_defer_gate``  — LayerCommunicator wraps: forward scoping (stale-pend drop
    at the first ``prepare_attn`` of a new forward), per-layer decision scoping
    (``prepare_mlp`` entry), and recording of the ``should_fuse_mlp_allreduce_with_
    next_layer`` / ``should_use_reduce_scatter`` decisions. Model-agnostic: every
    fusion-capable upstream model computes these on its LayerCommunicator and marks
    the deferred tensor (``_sglang_needs_allreduce_fusion``) — patching the
    communicator generalizes what the 2026-07-02 campaign did per-model
    (``MiniMaxM3MoE.forward_normal`` + ``MiniMaxM3Model.forward``, fe_patch.py:85-232).
  * ``sglang_moe_export``  — wraps ``modelopt_quant.flashinfer_cutlass_fused_moe``:
    when the ACTIVE bundle has an eligible deep kernel AND this layer's AR is deferred,
    arm skip-finalize around the call and stash the exported pre-finalize pointers,
    keyed by the output tensor's ``data_ptr`` (NOT call order — hybrid layers
    interleave attn-side AR calls between an export and its consume).
  * the arfusion dispatcher (optima/dispatch.py) — CONSUMES: a call whose input
    data_ptr matches a pend reconstructs zero-copy tensor views and runs the deep
    kernel; a pend the kernel can't serve is reconstructed by the TRUSTED fp32
    finalize + the stock fusion call (correct-but-slow, never corrupt).

The export ABI (implemented by the bundle's dep patch; versioned by this docstring):
  fe_set_skip_finalize(bool)   — arm/disarm skipping the standalone finalize kernel
  fe_get_export() -> 9 ints    — (seq, gemm_ptr, row_map_ptr, scales_ptr, tokens,
                                  hidden, padded_hidden, top_k, dtype_bits);
                                  ``seq`` increments per export; unchanged seq after a
                                  call means the launcher ran a FINALIZE-FUSED tactic
                                  and the output is already complete.
Layouts match the slot contract (optima/slots.py): gemm ``[T*K, H]`` per-rank partial,
``row_map`` K-major i32 replicated, ``scales`` ``[T, K]`` f32 replicated; the consume
call may HEAD-TRIM CUDA-graph batch padding (same data_ptr, T_consume <= T_export).

Correctness invariant (the reason for all the scoping): skip-finalize may ONLY be
armed when the deferred fusion call is certain to consume the export. Layers doing an
IMMEDIATE all-reduce (last layer, reduce-scatter layers) need their finalize in-op,
and abandoned capture-warmup forwards leave pends whose addresses a later forward can
recycle — both classes corrupted real runs in the campaign (the 06:59Z crash; the
smoke8 illegal access). Dynamo-compiled regions never record decisions (the wraps
bail under tracing), so the deep seam simply stays stock inside compiled pieces.
"""

from __future__ import annotations

import logging
import os
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from optima.capabilities import CallDescriptor, collective_call_descriptor

logger = logging.getLogger("optima.moe_export")

DEEP_SLOT = "collective.moe_finalize_ar_rmsnorm"


@dataclass(frozen=True)
class GroupTopology:
    """Validator-observed identity of the process group the deep kernel will use.

    Size alone is insufficient: two disjoint TP groups can have the same cardinality.
    Real distributed execution therefore fingerprints the ordered global ranks.  The
    object-identity fallback exists only for lightweight unit doubles in a process
    without an initialized ``torch.distributed`` runtime.
    """

    world_size: int
    fingerprint: tuple[Any, ...]


@dataclass(frozen=True)
class DeepSelection:
    """Exact pre-arm routing decision carried from export to consume."""

    impl: Any
    descriptor: CallDescriptor
    topology: GroupTopology

# OPTIMA_DEEP_DEBUG=1: trace every arm/refusal/export/consume with the stream-
# capturing flag. The 2026-07-07 GSM8K capture crash (pend collision inside the
# first captured shape) was only attributable from this interleaving — keep it.
_DEBUG = os.environ.get("OPTIMA_DEEP_DEBUG", "") == "1"


def _capturing() -> int:
    try:
        return int(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return -1


def _dbg(msg: str) -> None:
    if _DEBUG:
        logger.warning("deep-dbg[cap=%d] %s", _capturing(), msg)

# Backstop only: consumes pop continuously (~1-2 outstanding within a forward);
# growth means exports are never consumed — the seam is broken, refuse to continue.
_MAX_PENDS = 256
_RECEIPT_EVERY = 256

_state: dict[str, Any] = {
    # out data_ptr -> export pointers/layout + exact pre-arm DeepSelection
    "pends": {},
    "fwd": None,          # weakref.ref to the ForwardBatch of the live forward
    "fuse": False,        # last recorded should_fuse_mlp_allreduce_with_next_layer
    "rs": False,          # last recorded should_use_reduce_scatter
    "raw": None,          # loaded export-ABI module (shared .so globals)
    "disabled": False,    # one-shot: ABI missing / arch unsupported
    "ordinal": 0,         # prepare_mlp calls seen in the LIVE forward (layer counter)
    "n_layers": 0,        # decoder-layer count from the model config (0 = unresolved)
    "last_layer_vetoes": 0,
    "seq": 0,             # last fe_get_export sequence number seen
    "exports": 0,
    "consumed": 0,
    "orphans": 0,         # pend hit but the deep kernel couldn't serve it
    "stale_dropped": 0,   # pends dropped at a forward boundary (abandoned forward)
}


def _receipt() -> None:
    from optima import receipts

    receipts.write("export", {
        "slot": DEEP_SLOT,
        "exports": _state["exports"], "consumed": _state["consumed"],
        "orphans": _state["orphans"], "stale_dropped": _state["stale_dropped"],
        "last_layer_vetoes": _state["last_layer_vetoes"],
    }, tag=DEEP_SLOT)


def reset() -> None:
    """Test hook: fresh state."""
    _state.update(pends={}, fwd=None, fuse=False, rs=False, raw=None, disabled=False,
                  ordinal=0, n_layers=0, last_layer_vetoes=0,
                  seq=0, exports=0, consumed=0, orphans=0, stale_dropped=0)


# ---------------------------------------------------------------------------
# Defer-gate side (called by the sglang_defer_gate wraps)
# ---------------------------------------------------------------------------


def on_forward_boundary(forward_batch: Any) -> None:
    """Called at LayerCommunicator.prepare_attn ENTRY — before any fusion call of the
    layer can run. A changed ForwardBatch identity means a NEW forward: any pend still
    alive belongs to an abandoned one (capture warmup abandons forwards) and its
    address can be recycled by this forward's allocations — drop it before it can
    false-match. Weakref identity: a dead ref never equals a live object, so pool
    reuse of the ForwardBatch address itself cannot alias two forwards."""
    ref = _state["fwd"]
    if ref is not None and ref() is forward_batch:
        return
    _dbg(f"fb: NEW forward (pends={len(_state['pends'])})")
    if _state["pends"]:
        _state["stale_dropped"] += len(_state["pends"])
        logger.warning("optima: deep seam dropped %d abandoned export pend(s) at "
                       "forward boundary (total %d)", len(_state["pends"]),
                       _state["stale_dropped"])
        _state["pends"].clear()
        _receipt()
    _state["fuse"] = _state["rs"] = False
    _state["ordinal"] = 0
    try:
        _state["fwd"] = weakref.ref(forward_batch)
    except TypeError:  # un-weakref-able batch object: scope conservatively per call
        _state["fwd"] = None


def on_layer_mlp_boundary() -> None:
    """Called at LayerCommunicator.prepare_mlp ENTRY: decisions are scoped to ONE
    layer's MoE call. A layer whose MLP never reaches the wrapped fused-moe call
    (dense layer, ineligible path) must not leak a stale True into the next layer —
    the generalized equivalent of the campaign's forward_normal try/finally.

    Also advances the per-forward layer ordinal (layer i's prepare_mlp -> ordinal
    i+1), the input to the LAST-LAYER VETO in ``maybe_export``. CUDA-graph capture
    replays the same ForwardBatch for warmup+record passes, so the ordinal keeps
    climbing across passes there — the veto is therefore ``ordinal % n_layers``,
    never an equality against a reset counter."""
    _state["fuse"] = _state["rs"] = False
    _state["ordinal"] += 1


def record_fuse_decision(value: bool) -> None:
    _dbg(f"fuse={bool(value)}")
    _state["fuse"] = bool(value)


def record_rs_decision(value: bool) -> None:
    _dbg(f"rs={bool(value)}")
    _state["rs"] = bool(value)


def _consume_will_defer() -> bool:
    """One-shot read of 'this layer's AR is deferred to the next norm call'.
    Immediate-AR layers (last layer / reduce-scatter) must keep finalize in-op."""
    defer = _state["fuse"] and not _state["rs"]
    _state["fuse"] = _state["rs"] = False
    return defer


# ---------------------------------------------------------------------------
# Export side (called by the sglang_moe_export wrap)
# ---------------------------------------------------------------------------


def _tuning() -> bool:
    # NEVER toggle skip-finalize while flashinfer's autotuner is profiling: the tuner
    # sweeps many M buckets inside one warmup forward; a single deferring forward with
    # skip armed poisons the tactic table for ALL M (campaign, plan §22). Tuning must
    # always see the stock pipeline.
    try:
        from flashinfer.autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:  # noqa: BLE001
        return False


def _raw_module():
    """The loaded fused-moe JIT module carrying the export ABI. dlopen of the same
    overlay-built .so shares C++ globals, so any handle sees the same fe_* state as
    the engine's own calls. Resolution mirrors flashinfer's arch dispatch; a stock
    build (no dep patch) lacks the fe_* symbols -> disabled one-shot."""
    if _state["raw"] is not None:
        return _state["raw"]
    from flashinfer.fused_moe import core as fm_core

    major, minor = torch.cuda.get_device_capability()
    gen = getattr(fm_core, f"gen_cutlass_fused_moe_sm{major}{minor}_module", None)
    if gen is None:
        raise AttributeError(f"no fused-moe module generator for sm{major}{minor}")
    raw = gen(False).build_and_load()
    raw.fe_set_skip_finalize(False)  # AttributeError here = export ABI absent
    _state["raw"] = raw
    return raw


def _disable(reason: str) -> None:
    if not _state["disabled"]:
        _state["disabled"] = True
        logger.warning("optima: deep export seam DISABLED: %s", reason)


def _num_layers() -> Optional[int]:
    """Per-rank decoder-layer count from the model's own config — the static truth
    the communicator can't provide (minimax_m3 constructs LayerCommunicator without
    ``is_last_layer``, so sglang's own last-layer refusal never fires there; measured
    2026-07-07: 57 arms / 56 consumes per forward). Config-read, never learned from
    observed forwards: an aborted first forward would poison a learned count forever.
    None -> undeterminable or a config the ordinal model doesn't hold for (pipeline
    parallel splits layers across ranks; speculative draft loops add extra passes)
    — the caller disables the deep seam, fail-closed."""
    if _state["n_layers"]:
        return _state["n_layers"]
    try:
        from sglang.srt.server_args import get_global_server_args

        srv = get_global_server_args()
        pp = getattr(srv, "pp_size", 1)
        if pp not in (None, 0, 1):
            logger.warning("optima: deep seam needs a per-rank layer count; "
                           "pp_size=%r unsupported", pp)
            return None
        # sglang resolves this to an enum whose NONE member is truthy — compare by
        # name, not truthiness (burned 15:26Z 07-07: every rank silently disabled).
        spec = getattr(srv, "speculative_algorithm", None)
        spec_name = str(getattr(spec, "name", spec)).upper() if spec is not None else "NONE"
        if spec_name not in ("NONE", ""):
            logger.warning("optima: deep seam disabled under speculative decoding "
                           "(%r): draft loops break the layer-ordinal model", spec)
            return None
        # Raw config.json, not AutoConfig: custom-config classes keep nested configs
        # as PLAIN DICTS, so getattr misses them silently (burned 15:45Z 07-07 —
        # M3's count lives at text_config.num_hidden_layers). Arena models are
        # local paths; a hub id has no local config.json -> fail-closed below.
        import json

        raw = json.loads((Path(srv.model_path) / "config.json").read_text())
        holders = (raw, raw.get("text_config"), raw.get("language_config"),
                   raw.get("llm_config"))
        for holder in holders:
            n = holder.get("num_hidden_layers") if isinstance(holder, dict) else None
            if isinstance(n, int) and n > 0:
                logger.warning("optima: deep seam layer count = %d (last-layer "
                               "veto active)", n)
                _state["n_layers"] = n
                return n
        logger.warning("optima: deep seam found no num_hidden_layers in %s "
                       "(top-level keys: %s)", srv.model_path,
                       sorted(raw.keys())[:12])
    except Exception as exc:  # noqa: BLE001
        logger.warning("optima: deep seam could not resolve num_hidden_layers: %s",
                       exc)
    return None


def _input_ok(inp) -> bool:
    return torch.is_tensor(inp) and inp.dim() == 2 and inp.is_cuda


def group_topology(group: Any) -> Optional[GroupTopology]:
    """Resolve the actual candidate communicator, never an env/caller size hint."""
    if group is None:
        return None
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            size = int(dist.get_world_size(group))
            if size <= 1:
                return None
            try:
                ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
            except AttributeError:
                ranks = tuple(int(dist.get_global_rank(group, rank)) for rank in range(size))
            if len(ranks) != size or len(set(ranks)) != size:
                return None
            return GroupTopology(size, ("global_ranks", *ranks))
    except Exception:  # noqa: BLE001 - unresolved topology means stock, never a guess
        return None

    # Unit-test process-group doubles do not initialize torch.distributed.  Binding
    # their object identity is still stronger than accepting a caller-provided size;
    # production ProcessGroups always take the global-rank path above.
    try:
        size = int(group.size())
    except Exception:  # noqa: BLE001
        return None
    if size <= 1:
        return None
    return GroupTopology(size, ("process_object", id(group)))


def _producer_group_and_topology() -> tuple[Any, GroupTopology] | None:
    """Resolve the MoE TP group used by the eventual deferred norm call."""
    try:
        # A deferred MoE output reaches forward_with_allreduce_fusion with
        # use_attn_tp_group=False.  Resolve that exact group before destructively
        # skipping finalize; the consumer must later prove it sees the same group.
        from optima.dispatch import (
            _arfusion_group,
            _arfusion_group_role,
            _moe_data_parallel_world_size,
        )

        if (
            _moe_data_parallel_world_size() != 1
            or _arfusion_group_role(False) != "tp"
        ):
            return None
        group = _arfusion_group(False)
    except Exception:  # noqa: BLE001 - unavailable/upstream-drifted group -> stock
        return None
    topology = group_topology(group)
    return None if topology is None else (group, topology)


def _arg(args: tuple, kwargs: dict, name: str, position: int) -> Any:
    """Read one FlashInfer argument from its keyword or positional ABI."""
    return kwargs.get(name, args[position] if len(args) > position else None)


def maybe_export(orig: Callable[..., Any], args: tuple, kwargs: dict, *,
                 registry) -> Any:
    """The wrapped ``flashinfer_cutlass_fused_moe`` body. Falls through to ``orig``
    unless EVERY gate holds; when armed, skips the in-op finalize and pends the
    exported pointers for the deferred fusion call to consume."""
    defer = _consume_will_defer()  # one-shot: this decision belongs to THIS call
    if _state["disabled"] or not defer:
        _dbg(f"me: pass-through (disabled={_state['disabled']} defer={defer})")
        return orig(*args, **kwargs)

    # LAST-LAYER VETO (root cause of the 2026-07-07 capture crash): minimax_m3 never
    # passes is_last_layer, so sglang lets the FINAL layer defer its AR — harmless in
    # stock (finalize already ran in-op; the late AR lands on a transformed tensor),
    # FATAL with skip-finalize armed (the transform copies UNFINALIZED data and the
    # ptr-keyed consume can never match). A layer may only defer if a successor
    # layer's prepare_attn will consume it: ordinal % n_layers == 0 <=> last layer
    # of a (possibly same-ForwardBatch multi-pass) forward.
    n = _num_layers()
    if n is None:
        _disable("num_hidden_layers unresolvable (pp/spec/config)")
        return orig(*args, **kwargs)
    if _state["ordinal"] % n == 0:
        _state["last_layer_vetoes"] += 1
        _dbg(f"me: pass-through (last-layer veto, ordinal={_state['ordinal']} n={n})")
        return orig(*args, **kwargs)

    # Pinned FlashInfer positional ABI (kwargs remain primary): input=0,
    # selected/scales=1/2, fc1=3, tp/ep=13/15, output=19.
    inp = _arg(args, kwargs, "input", 0)
    if not _input_ok(inp):
        _dbg("me: pass-through (input shape/device ineligible)")
        return orig(*args, **kwargs)
    ep_size = _arg(args, kwargs, "ep_size", 15)
    if type(ep_size) is not int or ep_size != 1:
        # Export layouts are TP-only (no EP reshuffle).
        _dbg("me: pass-through (ep_size != 1)")
        return orig(*args, **kwargs)

    if _tuning():
        _dbg("me: pass-through (autotuner active)")
        return orig(*args, **kwargs)

    # Select from facts the validator can observe BEFORE skipping finalize.  Use the
    # final output buffer for dtype/H: FlashInfer may receive an FP8/FP4-packed input,
    # while the deep ABI and downstream residual stream are BF16.
    out_buf = _arg(args, kwargs, "output", 19)
    selected_experts = _arg(args, kwargs, "token_selected_experts", 1)
    final_scales = _arg(args, kwargs, "token_final_scales", 2)
    fc1_weights = _arg(args, kwargs, "fc1_expert_weights", 3)
    if not (
        torch.is_tensor(out_buf)
        and out_buf.dim() == 2
        # export_views wraps the raw 16-bit payload as BF16. Accepting an FP16/FP32
        # output here would make producer selection describe a different live ABI.
        and out_buf.dtype == torch.bfloat16
        and out_buf.device == inp.device
        and torch.is_tensor(selected_experts)
        and selected_experts.dim() == 2
        and torch.is_tensor(final_scales)
        and final_scales.dim() == 2
        and torch.is_tensor(fc1_weights)
        and fc1_weights.dim() >= 1
        and int(fc1_weights.shape[0]) > 0
    ):
        _dbg("me: pass-through (routing/output ABI unavailable)")
        return orig(*args, **kwargs)
    exp_tokens, hidden = (int(out_buf.shape[0]), int(out_buf.shape[1]))
    if (
        int(inp.shape[0]) != exp_tokens
        or int(selected_experts.shape[0]) != exp_tokens
        or tuple(final_scales.shape) != tuple(selected_experts.shape)
    ):
        _dbg("me: pass-through (routing/output token shape mismatch)")
        return orig(*args, **kwargs)
    top_k = int(selected_experts.shape[1])
    if not 1 <= top_k <= 64:
        _dbg(f"me: pass-through (invalid top_k={top_k})")
        return orig(*args, **kwargs)

    resolved = _producer_group_and_topology()
    if resolved is None:
        _dbg("me: pass-through (MoE TP group unavailable or single-rank)")
        return orig(*args, **kwargs)
    _group, topology = resolved
    tp_hint = _arg(args, kwargs, "tp_size", 13)
    if tp_hint is not None and (
        type(tp_hint) is not int or tp_hint != topology.world_size
    ):
        _dbg(
            "me: pass-through (caller TP size disagrees with actual group: "
            f"hint={tp_hint!r} actual={topology.world_size})"
        )
        return orig(*args, **kwargs)

    from optima.dispatch import _arch_tag, _dtype_name, _in_cuda_graph

    descriptor = collective_call_descriptor(
        dtype=_dtype_name(out_buf.dtype),
        architecture=(
            _arch_tag(out_buf.device.index or 0) if out_buf.is_cuda else None
        ),
        graph_mode="cuda_graph" if _in_cuda_graph() else "eager",
        world_size=topology.world_size,
        tp_size=topology.world_size,
        dimensions={
            "ep_size": 1,
            "num_tokens": exp_tokens,
            "exp_tokens": exp_tokens,
            "top_k": top_k,
            "hidden_dim": hidden,
            "last_dim": hidden,
        },
    )
    decision = registry.select(
        DEEP_SLOT, descriptor, write_fired_receipt=False
    )
    impl = decision.impl
    if impl is None:
        _dbg(
            "me: pass-through (no unique eligible deep variant, "
            f"T_exp={exp_tokens} K={top_k} TP={topology.world_size})"
        )
        return orig(*args, **kwargs)
    selection = DeepSelection(impl=impl, descriptor=descriptor, topology=topology)

    try:
        raw = _raw_module()
    except AttributeError as exc:
        _disable(f"export ABI unavailable ({exc}); stock finalize stays in-op")
        return orig(*args, **kwargs)

    raw.fe_set_skip_finalize(True)
    try:
        ret = orig(*args, **kwargs)
    finally:
        raw.fe_set_skip_finalize(False)

    seq, g2, rmap, scl, rows, hid, phid, k, bits = (int(x) for x in raw.fe_get_export())
    if seq == _state["seq"]:
        _dbg(f"me: finalize-fused tactic (seq unchanged at {seq}, T={exp_tokens})")
        return ret  # no export: a FINALIZE-FUSED tactic ran — output already complete
    _state["seq"] = seq

    out_t = ret[0] if isinstance(ret, (list, tuple)) else ret
    # Export layout the consume side can't recover from must FAIL LOUDLY: the output
    # tensor is unfinalized garbage and there is no way back once we return it.
    if not (
        torch.is_tensor(out_t)
        and out_t.dim() == 2
        and tuple(out_t.shape) == (exp_tokens, hidden)
        and out_t.dtype == out_buf.dtype
        and out_t.device == out_buf.device
        and rows == exp_tokens
        and hid == hidden
        and phid == hidden
        and k == top_k
        and bits == 16
    ):
        raise RuntimeError(
            "optima deep seam: export ABI mismatch "
            f"(rows={rows} T_exp={exp_tokens} hid={hid}/{hidden} phid={phid} "
            f"bits={bits} k={k}/{top_k}) — refusing to serve an unfinalized output"
        )
    optr = out_t.data_ptr()
    # Pends are cleared at every forward boundary, so within one forward an export
    # finding its own ptr still pending means its consume was missed — seam broken.
    if optr in _state["pends"]:
        raise RuntimeError(
            f"optima deep seam: moe output buffer {optr:#x} reused before its export "
            f"was consumed within one forward (exports={_state['exports']} "
            f"consumed={_state['consumed']} orphans={_state['orphans']} "
            f"pends={[hex(p) for p in _state['pends']]} capturing={_capturing()})")
    if len(_state["pends"]) > _MAX_PENDS:
        raise RuntimeError("optima deep seam: export pends leaking (never consumed)")
    _state["pends"][optr] = {
        "g2": g2,
        "idx": rmap,
        "scl": scl,
        "T": rows,
        "K": k,
        "hid": hid,
        "selection": selection,
    }
    _dbg(f"me: EXPORT seq={seq} optr={optr:#x} T={rows} K={k} "
         f"pends={len(_state['pends'])}")
    _state["exports"] += 1
    if _state["exports"] == 1 or _state["exports"] % _RECEIPT_EVERY == 0:
        _receipt()
    return ret


# ---------------------------------------------------------------------------
# Consume side (called by the arfusion dispatcher)
# ---------------------------------------------------------------------------


def has_pends() -> bool:
    return bool(_state["pends"])


def consume(input_tensor: torch.Tensor) -> Optional[dict]:
    """Pop the pend for this fusion call's input, if it is a skipped-finalize moe
    output (ptr-keyed). None -> a plain (already-finalized) fusion call."""
    if not _state["pends"] or not torch.is_tensor(input_tensor):
        return None
    exp = _state["pends"].pop(input_tensor.data_ptr(), None)
    if _DEBUG:
        _dbg(f"co: input={input_tensor.data_ptr():#x} T={input_tensor.shape[0]} "
             f"matched={exp is not None} "
             f"pends_left={[hex(p) for p in _state['pends']]}")
        if exp is None:
            import traceback
            frames = [f"{fr.filename.rsplit('/', 1)[-1]}:{fr.lineno}({fr.name})"
                      for fr in traceback.extract_stack()[-9:-1]]
            _dbg("co-miss stack: " + " <- ".join(frames))
    if exp is not None:
        _state["consumed"] += 1
        if _state["consumed"] == 1:
            logger.warning("optima: deep epilogue consume path LIVE (T_export=%d K=%d)",
                           exp["T"], exp["K"])
            _receipt()
    return exp


def orphaned(exp: dict) -> None:
    """A pend was popped but the deep kernel could not serve it (eligibility drift
    between export and consume). The dispatcher reconstructs via the trusted path;
    count it — a nonzero orphan count in receipts is a seam-health tell."""
    _state["orphans"] += 1
    logger.warning("optima: deep export ORPHANED (T=%d K=%d) — trusted reconstruct "
                   "(correct but slow); orphans=%d", exp["T"], exp["K"],
                   _state["orphans"])
    _receipt()


class _CudaPtrView:
    """Minimal __cuda_array_interface__ carrier so torch.as_tensor can wrap a raw
    exported device pointer zero-copy. bf16 has no CAI typestr — export as u2 and
    .view(torch.bfloat16) after."""

    def __init__(self, ptr: int, shape: tuple, typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "version": 3, "shape": tuple(shape), "typestr": typestr,
            "data": (int(ptr), False), "strides": None,
        }


def _wrap_ptr(ptr: int, shape: tuple, typestr: str, device: torch.device,
              view_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    with torch.cuda.device(device):
        t = torch.as_tensor(_CudaPtrView(ptr, shape, typestr), device=device)
    if t.data_ptr() != int(ptr):  # a copy would silently detach us from the workspace
        raise RuntimeError("optima deep seam: pointer wrap copied instead of aliasing")
    return t.view(view_dtype) if view_dtype is not None else t


def export_views(exp: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor,
                                                           torch.Tensor]:
    """(gemm_out [T*K,H] bf16, row_map [T*K] i32, scales [T,K] f32) — zero-copy views
    over the exported workspace, in the slot-contract layout."""
    rows = exp["T"] * exp["K"]
    gemm_out = _wrap_ptr(exp["g2"], (rows, exp["hid"]), "<u2", device,
                         view_dtype=torch.bfloat16)
    row_map = _wrap_ptr(exp["idx"], (rows,), "<i4", device)
    scales = _wrap_ptr(exp["scl"], (exp["T"], exp["K"]), "<f4", device)
    return gemm_out, row_map, scales


def trusted_finalize(exp: dict, input_tensor: torch.Tensor) -> torch.Tensor:
    """Validator-trusted fp32 local finalize from the exported views, head-trimmed to
    the consume call's T and cast back to the call dtype. The orphan-recovery path:
    its output feeds the STOCK fusion call, so a gate mismatch degrades to
    correct-but-slow instead of corrupt."""
    from optima.slots import _moe_fin_local_finalize

    gemm_out, row_map, scales = export_views(exp, input_tensor.device)
    acc = _moe_fin_local_finalize({
        "gemm_out": gemm_out, "row_map": row_map, "scales": scales,
        "residual": input_tensor,  # supplies only the head-trim T (this call's rows)
    })
    return acc.to(input_tensor.dtype)


def env_enabled() -> bool:
    """The deep machinery rides the arfusion seam's opt-in (consume lives there)."""
    return os.environ.get("OPTIMA_ARFUSION_SEAM") == "1"
