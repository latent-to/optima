"""Positive scheduler-role gate: candidate bundles load ONLY here.

sglang spawns the scheduler ranks AND a detokenizer (plus other manager
children) through the same interpreter bootstrap, and ALL of them import
watched seam modules transitively — measured on the B300 worker image
(2026-07-13): the detokenizer child alone pulls in parallel_state,
communicator, flashinfer_comm_fusion, modelopt_quant, radix_attention and
flashinfer.jit.core. An import-time bundle load therefore executes miner
module-level code in output-path processes; the detokenizer turns token ids
into text DOWNSTREAM of the sampler, so miner code there is the
output-substitution surface (e.g. spoofing benchmark-accuracy text). The
engine worker's active-member coverage gate refuses such an engine
(active receipts != tp_size) — this adapter makes the honest case pass by
pinning the load to positively-identified scheduler execution processes.

``run_scheduler_process`` is the module-level function sglang spawns one
scheduler rank with (mp spawn pickles it by qualified name, so the child
resolves THIS wrapper after its own bootstrap re-installs it). The wrapper
loads ordinary bundles at process entry. A direct device-artifact bundle is
only staged there, then bound by the validator-owned post-device hook after
SGLang establishes the rank-local CUDA context and before model load, warmup,
or graph capture. Non-scheduler children never call this gate, never acquire
that pending authority, never execute miner code, and never write an active
receipt.
"""

from __future__ import annotations

import functools
import sys

from optima.registry import REGISTRY, KernelRegistry

_MODULE = "sglang.srt.managers.scheduler"
_GATE_FLAG = "_optima_scheduler_gate"


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Wrap run_scheduler_process so bundle load happens at scheduler entry.

    Safe to call before the scheduler module is imported — it no-ops until then
    (the bootstrap post-import hook calls it again once the module loads).
    ``registry`` is unused (the load path owns registry mutation); the parameter
    keeps the uniform ``install(REGISTRY)`` adapter signature.
    """
    del registry
    mod = sys.modules.get(_MODULE)
    fn = getattr(mod, "run_scheduler_process", None) if mod is not None else None
    if fn is None or getattr(fn, _GATE_FLAG, False):
        return

    @functools.wraps(fn)
    def run_scheduler_process(*args, **kwargs):
        from optima import seam

        try:
            seam.load_candidate_bundle()
            result = fn(*args, **kwargs)
        except BaseException:
            # Preserve the initiating engine failure. Teardown still attempts to
            # fence/release direct artifacts, but a secondary cleanup error must
            # not replace the rank traceback the controller needs to diagnose.
            seam.teardown_candidate_bundle(suppress_errors=True)
            raise
        seam.teardown_candidate_bundle()
        return result

    setattr(run_scheduler_process, _GATE_FLAG, True)
    run_scheduler_process._optima_orig = fn  # type: ignore[attr-defined]
    mod.run_scheduler_process = run_scheduler_process


def uninstall() -> None:
    mod = sys.modules.get(_MODULE)
    fn = getattr(mod, "run_scheduler_process", None) if mod is not None else None
    if fn is None or not getattr(fn, _GATE_FLAG, False):
        return
    mod.run_scheduler_process = fn._optima_orig


def is_installed() -> bool:
    mod = sys.modules.get(_MODULE)
    fn = getattr(mod, "run_scheduler_process", None) if mod is not None else None
    return bool(fn is not None and getattr(fn, _GATE_FLAG, False))
