"""Model-agnostic defer-gate for the deep fused-epilogue seam.

The deep slot (``collective.moe_finalize_ar_rmsnorm``) may only skip flashinfer's
in-op finalize when THIS layer's all-reduce is deferred to the next norm call —
layers doing an immediate AR (last layer, reduce-scatter layers) need finalize in-op,
or their output is unfinalized garbage. The 2026-07-02 campaign learned that decision
by patching the M3 model classes (``MiniMaxM3MoE.forward_normal`` +
``MiniMaxM3Model.forward``); the generalization patches ``LayerCommunicator`` instead,
because the upstream idiom is identical across every fusion-capable model (glm4_moe,
qwen3_moe, minimax_m2/m3, bailing, ...):

    prepare_attn(...)                                # (consume site lives in here)
    prepare_mlp(...)
    fuse = comm.should_fuse_mlp_allreduce_with_next_layer(batch)
    rs   = comm.should_use_reduce_scatter(batch)
    out  = self.mlp(hidden, batch, fuse, rs)         # -> the wrapped fused-moe call
    if fuse: out._sglang_needs_allreduce_fusion = True

Four thin wraps feed optima.moe_export's state machine:
  * ``prepare_attn`` ENTRY      -> forward-boundary scoping: the first prepare_attn of
    a NEW forward (ForwardBatch identity via weakref) drops stale pends BEFORE any
    fusion call of that forward can false-match a recycled address.
  * ``prepare_mlp`` ENTRY       -> per-layer decision scoping (a dense/ineligible
    layer must not leak its decision into the next layer's MoE call).
  * ``should_fuse_mlp_allreduce_with_next_layer`` / ``should_use_reduce_scatter``
    -> record the two decisions; will_defer = fuse AND NOT rs, read one-shot by the
    export wrap.

Every wrap bails first under Dynamo tracing: decisions are then never recorded inside
compiled pieces, so the export seam stays stock there by construction (the compiled
region also bakes the stock fused-moe + fusion call — consistent).

Installed only when the arfusion seam is opted in (OPTIMA_ARFUSION_SEAM=1) AND the
registry holds a deep-slot kernel; otherwise install() returns WITHOUT marking the
module patched, so a later activate() pass (bundle now loaded) can retry.
"""

from __future__ import annotations

from optima import moe_export
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_defer_gate_patched"
_MODULE = "sglang.srt.layers.communicator"
_CLASS = "LayerCommunicator"


def _wrap_methods(cls) -> None:
    from optima.dispatch import _dynamo_compiling

    orig_prepare_attn = cls.prepare_attn
    orig_prepare_mlp = cls.prepare_mlp
    orig_fuse = cls.should_fuse_mlp_allreduce_with_next_layer
    orig_rs = cls.should_use_reduce_scatter

    def prepare_attn(self, *args, **kwargs):
        if not _dynamo_compiling():
            fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
            if fb is not None:
                moe_export.on_forward_boundary(fb)
        return orig_prepare_attn(self, *args, **kwargs)

    def prepare_mlp(self, *args, **kwargs):
        if not _dynamo_compiling():
            moe_export.on_layer_mlp_boundary()
        return orig_prepare_mlp(self, *args, **kwargs)

    def should_fuse(self, *args, **kwargs):
        out = orig_fuse(self, *args, **kwargs)
        if not _dynamo_compiling():
            moe_export.record_fuse_decision(out)
        return out

    def should_rs(self, *args, **kwargs):
        out = orig_rs(self, *args, **kwargs)
        if not _dynamo_compiling():
            moe_export.record_rs_decision(out)
        return out

    cls.prepare_attn = prepare_attn
    cls.prepare_mlp = prepare_mlp
    cls.should_fuse_mlp_allreduce_with_next_layer = should_fuse
    cls.should_use_reduce_scatter = should_rs


def install(registry: KernelRegistry = REGISTRY) -> None:
    import sys

    mod = sys.modules.get(_MODULE)
    if mod is None or getattr(mod, _PATCH_FLAG, False):
        return
    if not moe_export.env_enabled():
        return
    # Retryable gate (no flag set): activate() installs integrations before it loads
    # the bundle on its first pass; a later pass sees the registry populated.
    if moe_export.DEEP_SLOT not in registry.slots():
        return
    cls = getattr(mod, _CLASS, None)
    if cls is None or not all(hasattr(cls, m) for m in (
            "prepare_attn", "prepare_mlp", "should_fuse_mlp_allreduce_with_next_layer",
            "should_use_reduce_scatter")):
        return  # upstream drift: compat canary owns the loud failure
    _wrap_methods(cls)
    setattr(mod, _PATCH_FLAG, True)
