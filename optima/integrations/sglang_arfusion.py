"""Wire the Optima dispatcher into SGLang's fused AR+residual+RMSNorm epilogue waist.

With ``--enable-flashinfer-allreduce-fusion`` (an arena server flag), sglang's
LayerCommunicator defers each participating layer's TP all-reduce and routes the whole
epilogue — all-reduce + residual-add + RMSNorm — through ONE module-level function:
``sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce_residual_rmsnorm``.
Decode spends most of its non-GEMM time in exactly this chain, which is why the fused
epilogue is the measured lever (the 2026-07-02 MiniMax-M3 campaign: a two-shot Lamport
AR+add+norm kernel beat both the stock unfused chain AND flashinfer's own fused kernel
at decode T, graphs-on).

Patch mechanism: the call site (``layernorm._forward_with_allreduce_fusion``) resolves
the symbol per call via a function-local import, so rebinding the MODULE ATTRIBUTE
reroutes every call site — no class patch needed. This is the same mechanism the
campaign's fe_patch validated in production, folded into the seam table so bootstrap,
activate() and the compat canary all track it.

This is a COLLECTIVE slot (kind="collective"): the kernel is handed the process group,
so it is verified DISTRIBUTED (optima.verify_collective, fp32 cross-rank sum + trusted
add/norm via the slot's collective_finish) and the end-to-end gate is mandatory. See
docs/SLOT_CONTRACT.md.
"""

from __future__ import annotations

from optima.dispatch import make_arfusion_dispatcher
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_arfusion_patched"
_MODULE = "sglang.srt.layers.flashinfer_comm_fusion"
_FUNC = "flashinfer_allreduce_residual_rmsnorm"
_ORIG_ATTR = "_optima_orig_allreduce_residual_rmsnorm"


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Rebind the module-level fusion function. No-ops until the module is imported."""
    import sys

    mod = sys.modules.get(_MODULE)
    if mod is None:
        return

    if getattr(mod, _PATCH_FLAG, False):
        return

    orig = getattr(mod, _FUNC, None)
    if orig is None:
        return

    setattr(mod, _ORIG_ATTR, orig)
    setattr(mod, _FUNC, make_arfusion_dispatcher(orig, registry=registry))
    setattr(mod, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    mod = sys.modules.get(_MODULE)
    if mod is None or not getattr(mod, _PATCH_FLAG, False):
        return
    setattr(mod, _FUNC, getattr(mod, _ORIG_ATTR))
    delattr(mod, _ORIG_ATTR)
    setattr(mod, _PATCH_FLAG, False)


def is_installed() -> bool:
    import sys

    mod = sys.modules.get(_MODULE)
    if mod is None:
        return False
    return bool(getattr(mod, _PATCH_FLAG, False))
