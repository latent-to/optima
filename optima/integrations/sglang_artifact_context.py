"""Finalize direct artifacts after SGLang establishes the rank CUDA context."""

from __future__ import annotations

import functools
import sys

from optima.registry import REGISTRY, KernelRegistry

_MODULE = "sglang.srt.model_executor.model_runner"
_CLASS = "ModelRunner"
_METHOD = "init_torch_distributed"
_HOOK_FLAG = "_optima_artifact_context"


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Install one AFTER hook at the pinned post-device/pre-model boundary."""

    del registry
    mod = sys.modules.get(_MODULE)
    cls = getattr(mod, _CLASS, None) if mod is not None else None
    fn = getattr(cls, _METHOD, None) if cls is not None else None
    if fn is None or getattr(fn, _HOOK_FLAG, False):
        return

    @functools.wraps(fn)
    def init_torch_distributed(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        if not getattr(self, "is_draft_worker", False):
            from optima import seam

            seam.finalize_pending_candidate_bundle()
        return result

    setattr(init_torch_distributed, _HOOK_FLAG, True)
    init_torch_distributed._optima_orig = fn  # type: ignore[attr-defined]
    setattr(cls, _METHOD, init_torch_distributed)


def uninstall() -> None:
    mod = sys.modules.get(_MODULE)
    cls = getattr(mod, _CLASS, None) if mod is not None else None
    fn = getattr(cls, _METHOD, None) if cls is not None else None
    if fn is None or not getattr(fn, _HOOK_FLAG, False):
        return
    setattr(cls, _METHOD, fn._optima_orig)


def is_installed() -> bool:
    mod = sys.modules.get(_MODULE)
    cls = getattr(mod, _CLASS, None) if mod is not None else None
    fn = getattr(cls, _METHOD, None) if cls is not None else None
    return bool(fn is not None and getattr(fn, _HOOK_FLAG, False))
