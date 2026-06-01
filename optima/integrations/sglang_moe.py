"""Wire the Optima dispatcher into SGLang's fused-MoE layer.

Like attention (and unlike the single-op silu / norm seams), MoE experts are a
*block* — and a (prepare, forward) one. We patch the single chokepoint every MoE
layer funnels through: ``FusedMoE.forward`` in
``sglang.srt.layers.moe.fused_moe_triton.layer``. Routing (gate -> top-k) runs
upstream and is handed in as ``topk_output``, so this is exactly the
"run the experts given the routing" boundary — backend-agnostic (it sits above
triton / cutlass sm90 / sm100 / sm120 / marlin alike).

This is the cheat-resistant home for an expert-kernel win (e.g. the GPT-OSS MXFP4
fused MoE): the validator owns routing, the expert weights, and the output buffer;
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
    """Patch ``FusedMoE.forward``. No-ops until the fused-moe layer module is imported."""
    import sys

    mod = sys.modules.get(_MODULE)
    FusedMoE = getattr(mod, "FusedMoE", None) if mod is not None else None
    if FusedMoE is None:
        return

    if getattr(FusedMoE, _PATCH_FLAG, False):
        return

    orig_forward = FusedMoE.forward
    FusedMoE.forward = make_moe_dispatcher(orig_forward, registry=registry)
    FusedMoE._optima_orig_forward = orig_forward  # type: ignore[attr-defined]
    setattr(FusedMoE, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    if _MODULE not in sys.modules:
        return
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if not getattr(FusedMoE, _PATCH_FLAG, False):
        return
    FusedMoE.forward = FusedMoE._optima_orig_forward  # type: ignore[attr-defined]
    delattr(FusedMoE, "_optima_orig_forward")
    setattr(FusedMoE, _PATCH_FLAG, False)


def is_installed() -> bool:
    import sys

    mod = sys.modules.get(_MODULE)
    FusedMoE = getattr(mod, "FusedMoE", None) if mod is not None else None
    if FusedMoE is None:
        return False
    return bool(getattr(FusedMoE, _PATCH_FLAG, False))
