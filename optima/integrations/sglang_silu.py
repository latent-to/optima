"""Wire the Optima dispatcher into SGLang's SiluAndMul seam.

WHY PATCH THE CLASS METHOD (and not register_oot_forward):
``SiluAndMul`` subclasses ``MultiPlatformOp``, whose ``dispatch_forward()`` only
consults the out-of-tree registry when ``current_platform.is_out_of_tree()`` is
True — which is False on a normal CUDA validator. On CUDA it binds
``self.forward_cuda`` at ``__init__``. So the robust, platform-independent seam
is to replace the *class* method ``SiluAndMul.forward_cuda`` (and
``forward_native`` for CPU dry-runs) BEFORE the model is constructed; each
instance then binds our dispatcher when it is created.

This keeps the seam validator-owned and singular: exactly one function, holding
the captured baseline for fallback, is installed at one place.

Call order matters: ``install()`` MUST run before ``sglang.Engine(...)`` builds
the model, otherwise already-constructed instances keep the original bound
method. ``uninstall()`` restores the originals.
"""

from __future__ import annotations

from typing import Optional

from optima.dispatch import make_silu_and_mul_dispatcher
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_patched"


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch SiluAndMul.forward_cuda/native to route through the dispatcher.

    Safe to call before the activation module is imported — it no-ops until then
    (the bootstrap post-import hook calls it again once the module loads).
    """
    import sys

    mod = sys.modules.get("sglang.srt.layers.activation")
    SiluAndMul = getattr(mod, "SiluAndMul", None) if mod is not None else None
    if SiluAndMul is None:
        # not imported yet, or still mid-import (class not yet defined) -> the
        # bootstrap post-import hook calls install() again when the module finishes.
        return

    if getattr(SiluAndMul, _PATCH_FLAG, False):
        return

    orig_cuda = SiluAndMul.forward_cuda
    orig_native = SiluAndMul.forward_native

    SiluAndMul.forward_cuda = make_silu_and_mul_dispatcher(orig_cuda, registry=registry)
    SiluAndMul.forward_native = make_silu_and_mul_dispatcher(orig_native, registry=registry)

    SiluAndMul._optima_orig_cuda = orig_cuda  # type: ignore[attr-defined]
    SiluAndMul._optima_orig_native = orig_native  # type: ignore[attr-defined]
    setattr(SiluAndMul, _PATCH_FLAG, True)


def uninstall() -> None:
    from sglang.srt.layers.activation import SiluAndMul

    if not getattr(SiluAndMul, _PATCH_FLAG, False):
        return
    SiluAndMul.forward_cuda = SiluAndMul._optima_orig_cuda  # type: ignore[attr-defined]
    SiluAndMul.forward_native = SiluAndMul._optima_orig_native  # type: ignore[attr-defined]
    delattr(SiluAndMul, "_optima_orig_cuda")
    delattr(SiluAndMul, "_optima_orig_native")
    setattr(SiluAndMul, _PATCH_FLAG, False)


def is_installed() -> bool:
    try:
        from sglang.srt.layers.activation import SiluAndMul
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(SiluAndMul, _PATCH_FLAG, False))


def rebind_existing(root_module) -> int:
    """Safety net for ordering: re-dispatch already-constructed SiluAndMul ops.

    ``MultiPlatformOp`` binds ``self._forward_method`` at ``__init__``. If the
    model was built *before* ``install()`` ran (e.g. plugin load ordering), those
    instances still point at the original method. Call this with the loaded model
    to re-run dispatch so each SiluAndMul picks up the patched class method.
    Returns the number of instances rebound.
    """
    from sglang.srt.layers.activation import SiluAndMul

    if not is_installed():
        return 0
    count = 0
    for module in root_module.modules():
        if isinstance(module, SiluAndMul):
            module._forward_method = module.dispatch_forward()
            count += 1
    return count
