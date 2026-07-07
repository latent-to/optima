"""Export producer for the deep fused-epilogue seam — the modelopt fused-moe wrap.

Rebinds the MODULE-LEVEL ``flashinfer_cutlass_fused_moe`` name inside
``sglang.srt.layers.quantization.modelopt_quant`` (the NVFP4/FP8 cutlass MoE path
resolves it as a module global at every call site, so an attribute rebind reroutes
all of them — same mechanism as the arfusion seam, campaign-validated).

The wrap is a thin shell over ``optima.moe_export.maybe_export``: when the ACTIVE
bundle registered an eligible ``collective.moe_finalize_ar_rmsnorm`` kernel AND the
defer-gate says this layer's AR is deferred AND the flashinfer overlay build carries
the export ABI, the call runs with skip-finalize armed and its pre-finalize pointers
are pended for the arfusion dispatcher to consume; in every other case it is exactly
the stock call. Dynamo tracing bakes pure stock (the defer-gate never records
decisions under tracing, and we bail here too for symmetry).

Install gating mirrors sglang_defer_gate: opt-in env + a deep kernel in the registry,
retryable across activate() passes.
"""

from __future__ import annotations

from optima import moe_export
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_moe_export_patched"
_MODULE = "sglang.srt.layers.quantization.modelopt_quant"
_FUNC = "flashinfer_cutlass_fused_moe"
_ORIG_ATTR = "_optima_orig_flashinfer_cutlass_fused_moe"


def install(registry: KernelRegistry = REGISTRY) -> None:
    import sys

    mod = sys.modules.get(_MODULE)
    if mod is None or getattr(mod, _PATCH_FLAG, False):
        return
    if not moe_export.env_enabled():
        return
    if moe_export.DEEP_SLOT not in registry.slots():
        return  # retryable: the bundle may load on a later activate() pass
    orig = getattr(mod, _FUNC, None)
    if orig is None:
        return  # flashinfer absent (CPU box) or upstream drift — canary owns loudness

    from optima.dispatch import _dynamo_compiling

    def wrapped(*args, **kwargs):
        if _dynamo_compiling():
            return orig(*args, **kwargs)
        return moe_export.maybe_export(orig, args, kwargs, registry=registry)

    setattr(mod, _ORIG_ATTR, orig)
    setattr(mod, _FUNC, wrapped)
    setattr(mod, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    mod = sys.modules.get(_MODULE)
    if mod is None or not getattr(mod, _PATCH_FLAG, False):
        return
    setattr(mod, _FUNC, getattr(mod, _ORIG_ATTR))
    delattr(mod, _ORIG_ATTR)
    setattr(mod, _PATCH_FLAG, False)
