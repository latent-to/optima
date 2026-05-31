"""Wire the Optima dispatcher into SGLang's RMSNorm seam.

Same approach as the SiluAndMul seam: ``RMSNorm`` is a ``MultiPlatformOp``, so we
replace its class methods ``forward_cuda`` / ``forward_native`` (before the model
is built) with a dispatcher that routes to a miner kernel when one is registered
and eligible, else falls back to the captured baseline.

RMSNorm is universal (every transformer layer), so this is the slot that fires on
models like GPT-OSS whose activation is fused into the MoE kernel and therefore
doesn't go through SiluAndMul.
"""

from __future__ import annotations

from optima.dispatch import make_rmsnorm_dispatcher
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_patched"


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch RMSNorm.forward_cuda/native. No-ops until layernorm is imported."""
    import sys

    mod = sys.modules.get("sglang.srt.layers.layernorm")
    RMSNorm = getattr(mod, "RMSNorm", None) if mod is not None else None
    if RMSNorm is None:
        return

    if getattr(RMSNorm, _PATCH_FLAG, False):
        return

    orig_cuda = RMSNorm.forward_cuda
    orig_native = RMSNorm.forward_native

    RMSNorm.forward_cuda = make_rmsnorm_dispatcher(orig_cuda, registry=registry)
    RMSNorm.forward_native = make_rmsnorm_dispatcher(orig_native, registry=registry)

    RMSNorm._optima_orig_cuda = orig_cuda  # type: ignore[attr-defined]
    RMSNorm._optima_orig_native = orig_native  # type: ignore[attr-defined]
    setattr(RMSNorm, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    if "sglang.srt.layers.layernorm" not in sys.modules:
        return
    from sglang.srt.layers.layernorm import RMSNorm

    if not getattr(RMSNorm, _PATCH_FLAG, False):
        return
    RMSNorm.forward_cuda = RMSNorm._optima_orig_cuda  # type: ignore[attr-defined]
    RMSNorm.forward_native = RMSNorm._optima_orig_native  # type: ignore[attr-defined]
    delattr(RMSNorm, "_optima_orig_cuda")
    delattr(RMSNorm, "_optima_orig_native")
    setattr(RMSNorm, _PATCH_FLAG, False)
