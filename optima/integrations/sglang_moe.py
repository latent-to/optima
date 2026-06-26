"""Wire the Optima dispatcher into SGLang's fused-MoE layer.

Like attention (and unlike the single-op silu / norm seams), MoE experts are a
*block* — and a (prepare, forward) one. We patch ``FusedMoE.forward_impl`` in
``sglang.srt.layers.moe.fused_moe_triton.layer`` — NOT ``.forward``. ``forward``
is a thin router: under ``is_in_piecewise_cuda_graph()`` it dispatches to two
module-level custom ops (``moe_forward_piecewise_cuda_graph_impl`` /
``fused_moe_bypassed_piecewise_cuda_graph_impl``) that call ``forward_impl``
*directly* (layer.py:1304/1331 on the pin), bypassing a ``.forward`` patch — so a
kernel installed on ``.forward`` silently does NOT run under piecewise capture
(the production graphs-ON regime). ``forward_impl`` is the true waist: the eager
path, the in-piecewise standard path, and both custom-op paths all converge on it
(layer.py:1087/1089/1304/1331), and the trailing TP all-reduce lives inside it
(layer.py:1126). Routing (gate -> top-k) runs upstream and is handed in as
``topk_output``, so this is exactly the "run the experts given the routing"
boundary — backend-agnostic (it sits above the triton / cutlass / marlin MoE
backends alike).

This is the cheat-resistant home for an expert-kernel submission: the validator owns
routing, the expert weights, and the output buffer;
the miner only repacks the weights once (``prepare``) and fills the output each step
(``entry``). The combined output feeds the residual stream + downstream layers +
sampler (all stock) — no final output to substitute, and no sglang source patch or
engine reconfigure (unlike the framework/rebuild path). One pinned, unmodified
sglang per the consensus invariant; the kernel is injected at runtime exactly like
the other seams.

See ``optima/dispatch.make_moe_dispatcher`` for the scope wired today (standard
routing, eager-only, non-EP, behind ``OPTIMA_MOE_SEAM=1``) versus the graph-safe /
quantized-weight-view GPU integration points that come next.
"""

from __future__ import annotations

from optima.dispatch import make_moe_dispatcher
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_moe_patched"
_MODULE = "sglang.srt.layers.moe.fused_moe_triton.layer"


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch ``FusedMoE.forward_impl``. No-ops until the fused-moe layer module is
    imported, or if this sglang lacks ``forward_impl`` (older pin) -> the seam stays
    inert and the compat canary flags the missing chokepoint."""
    import sys

    mod = sys.modules.get(_MODULE)
    FusedMoE = getattr(mod, "FusedMoE", None) if mod is not None else None
    if FusedMoE is None or not hasattr(FusedMoE, "forward_impl"):
        return

    if getattr(FusedMoE, _PATCH_FLAG, False):
        return

    orig_impl = FusedMoE.forward_impl
    FusedMoE.forward_impl = make_moe_dispatcher(orig_impl, registry=registry)
    FusedMoE._optima_orig_forward_impl = orig_impl  # type: ignore[attr-defined]
    setattr(FusedMoE, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    if _MODULE not in sys.modules:
        return
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if not getattr(FusedMoE, _PATCH_FLAG, False):
        return
    FusedMoE.forward_impl = FusedMoE._optima_orig_forward_impl  # type: ignore[attr-defined]
    delattr(FusedMoE, "_optima_orig_forward_impl")
    setattr(FusedMoE, _PATCH_FLAG, False)


def is_installed() -> bool:
    import sys

    mod = sys.modules.get(_MODULE)
    FusedMoE = getattr(mod, "FusedMoE", None) if mod is not None else None
    if FusedMoE is None:
        return False
    return bool(getattr(FusedMoE, _PATCH_FLAG, False))
