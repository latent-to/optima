"""Shared seam activation, used by both the .pth bootstrap and the plugin hook.

Installs the validator-owned dispatcher into SiluAndMul and, when the env marks
this process as the candidate, loads + enables the vetted bundle. Driven by env
so the same code serves both runs of the two-launch eval:

    OPTIMA_BUNDLE_PATH   path to the bundle (empty -> baseline)
    OPTIMA_ACTIVE        "1" to load + enable the miner kernel, else baseline
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("optima.seam")


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


# Set TRUE in the validator's driver/timer process (NOT inherited by the spawned
# scheduler, because spawn starts a fresh interpreter and module globals don't
# cross). When true, we install the pass-through seam but NEVER import the miner
# module here. That keeps the process that measures wall-clock free of any miner
# code, so a malicious kernel cannot monkeypatch the timer. The miner kernel is
# loaded only in the scheduler child, which is where it actually runs.
_IS_DRIVER = False


def mark_driver() -> None:
    """Call in the timing/driver process BEFORE importing sglang."""
    global _IS_DRIVER
    _IS_DRIVER = True


# Bundle is loaded once per process even though activate() may run twice (once
# per watched module: activation, then layernorm).
_bundle_loaded = False


def activate() -> None:
    """Install all seams; load + enable the bundle iff this process is active.

    Called by the bootstrap post-import hook after EITHER seamed module loads
    (activation / layernorm). Each install no-ops until its module is present, so
    calling activate twice patches whatever is available each time.
    """
    from optima.integrations import sglang_attention, sglang_moe, sglang_norm, sglang_silu
    from optima.registry import REGISTRY

    for install in (sglang_silu.install, sglang_norm.install, sglang_attention.install, sglang_moe.install):
        try:
            install(REGISTRY)
        except Exception:  # noqa: BLE001 - never break engine startup
            logger.exception("optima: failed to install a seam")

    if _IS_DRIVER:
        # Timing process: seams installed (pass-through) but we never load the
        # miner module here, so the timer is out of the miner's reach.
        REGISTRY.disable()
        return

    global _bundle_loaded
    if _bundle_loaded:
        return

    bundle = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not bundle or not _truthy(os.environ.get("OPTIMA_ACTIVE")):
        REGISTRY.disable()
        return

    try:
        _load_bundle_into_registry(bundle)
        REGISTRY.enable()
        if _truthy(os.environ.get("OPTIMA_STRICT")):
            # Surface kernel errors instead of silently falling back (debug/proof: a
            # failing kernel crashes the engine rather than masquerading as baseline).
            REGISTRY.set_strict(True)
        _bundle_loaded = True
        logger.info("optima: bundle %s active -> slots %s", bundle, REGISTRY.slots())
    except Exception:  # noqa: BLE001 - a bad bundle must not wedge the engine
        logger.exception("optima: bundle load failed for %s; running baseline", bundle)
        REGISTRY.clear()


def _load_bundle_into_registry(bundle: str) -> None:
    from optima.manifest import load_manifest, resolve_source
    from optima.registry import REGISTRY, KernelImpl, eligibility_from_metadata
    from optima.sandbox import load_entry, scan_path
    from optima.slots import SLOTS

    manifest = load_manifest(bundle)
    for op in manifest.ops:
        if op.slot not in SLOTS:
            continue
        src = resolve_source(bundle, op)
        scan = scan_path(src)
        if not scan.ok:  # defense-in-depth: re-scan in the worker before load
            logger.warning("optima: skip %s, failed scan: %s", op.slot, scan.violations)
            continue
        meta = json.loads((Path(bundle) / op.metadata).read_text()) if op.metadata else {}
        entry = load_entry(src, op.entry)
        # (prepare, forward) slots: load the 2nd callable too, so the runtime dispatcher
        # can run the miner's weight-layout transform once and feed `prepared` to forward.
        # (Until now prepare was only exercised by CPU `verify`; the block seam needs it
        # live.) None for forward-only slots.
        prepare = load_entry(src, op.prepare) if getattr(op, "prepare", None) else None
        REGISTRY.register(
            KernelImpl(
                slot=op.slot,
                bundle_id=manifest.bundle_id,
                entry=entry,
                prepare=prepare,
                eligibility=eligibility_from_metadata(meta, op.dtypes),
            )
        )
        # FRAMEWORK MODE: an optional setup() runs ONCE here (candidate scheduler only —
        # the driver/timer process returns before this). It may patch the engine
        # (monkeypatch a backend, register a custom op, set flags) — that is how a
        # surface-opening win like the sm120 flashinfer fixes is expressed. It is
        # untrusted, so correctness is gated by token-match vs the stock baseline
        # (framework_mode) and it MUST run in the no-egress isolation worker before the
        # subnet opens to untrusted miners.
        if getattr(op, "setup", None):
            load_entry(src, op.setup)()
            logger.info("optima: ran setup() for %s", op.slot)
