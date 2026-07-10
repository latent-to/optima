"""Wire the Optima dispatcher into the MSA arena's PREFILL indexer waist
(attention.msa_prefill_block_score).

The chokepoint is the module-level wrapper
``sglang.srt.layers.attention.minimax_sparse_ops.prefill.flash_with_topk_idx
.flash_prefill_with_topk_index``: every chunked-prefill call of every sparse layer
funnels through it, and in the production MSA config (``disable_index_value``) its ONLY
output is the (heads, total_q, blocks) score slab that the stock top-k kernel then
consumes. The dispatcher (``optima.dispatch.make_msa_prefill_dispatcher``) swaps the
score PRODUCTION per (request, head) and keeps the selection + attend byte-stock — the
2026-07-10 M3 campaign measured this exact swap at +19.6%/+22.4% e2e serving prefill at
equal fidelity (needles 5/5, GSM8K paired identical, ITL-neutral on the peak backend).

Unlike its decode sibling (``sglang_msa.py``, still a stub), this seam IS in the
generic table (optima/seams.py): the ``requires`` field points at the M3-only
``minimax_sparse_ops`` package, so the compat canary SKIPS the row on pins that don't
ship the MSA backend instead of false-failing — the reason the decode stub stayed out
of the table no longer applies.

Same module-attribute rebind mechanism as the arfusion seam: the wrapper is resolved
per call site, so rebinding reroutes every caller; ``OPTIMA_MSA_PREFILL_SEAM=1`` gates
dispatch at call time.
"""

from __future__ import annotations

from optima.dispatch import make_msa_prefill_dispatcher
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_msa_prefill_patched"
_MODULE = "sglang.srt.layers.attention.minimax_sparse_ops.prefill.flash_with_topk_idx"
_FUNC = "flash_prefill_with_topk_index"
_ORIG_ATTR = "_optima_orig_flash_prefill_with_topk_index"


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Rebind the module-level wrapper. No-ops until the module is imported."""
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
    setattr(mod, _FUNC, make_msa_prefill_dispatcher(orig, mod, registry=registry))
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
