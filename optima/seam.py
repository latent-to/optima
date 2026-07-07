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

    Called by the bootstrap post-import hook after ANY seamed module loads. Each
    install no-ops until its module is present, so calling activate repeatedly patches
    whatever is available each time. The adapter list comes from the single seam table
    (optima/seams.py) — the same table the bootstrap watch-list and the compat canary
    use, so there is no parallel list to keep in sync.
    """
    import importlib

    from optima.registry import REGISTRY
    from optima.seams import SEAM_ADAPTERS

    for adapter in SEAM_ADAPTERS:
        try:
            mod = importlib.import_module(f"optima.integrations.{adapter.integration}")
            mod.install(REGISTRY)
        except Exception:  # noqa: BLE001 - never break engine startup
            logger.exception("optima: failed to install seam %s", adapter.name)

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
        from optima import receipts

        if REGISTRY.slots():
            # Positive evidence for the eval driver (anti phantom-pass): this rank
            # loaded the bundle and enabled the registry. See optima/receipts.py.
            receipts.write("active", {"bundle": bundle, "slots": REGISTRY.slots()})
        else:
            # Loaded without exception but registered nothing (scan-tree reject, no
            # known slots, every op skipped): for the eval this is exactly as
            # stock-vs-stock as a failed load — say so, don't stay silent.
            receipts.write("load_failed", {"bundle": bundle, "reason": "no slots registered"})
    except Exception:  # noqa: BLE001 - a bad bundle must not wedge the engine
        logger.exception("optima: bundle load failed for %s; running baseline", bundle)
        from optima import receipts

        receipts.write("load_failed", {"bundle": bundle, "reason": "exception during load"})
        REGISTRY.clear()


def _load_bundle_into_registry(bundle: str) -> None:
    from optima.manifest import load_manifest, resolve_source
    from optima.registry import REGISTRY, KernelImpl, eligibility_from_metadata
    from optima.sandbox import callable_from, load_module, scan_path, scan_tree
    from optima.slots import SLOTS

    manifest = load_manifest(bundle)
    # Recursive vendored-tree guard: a bundle can carry a whole vendored library; every .py
    # must clear the policy scan, not just the declared entries. Fail closed (load nothing).
    # Pass the manifest's declared cuda_sources + dep_patches so the runtime load
    # enforces the SAME strict allowlist as the CLI scan (undeclared non-.py files
    # reject here too).
    from optima.manifest import all_declared_cuda_sources, all_declared_dep_patches

    tree = scan_tree(bundle,
                     declared_cuda_sources=all_declared_cuda_sources(bundle, manifest),
                     declared_dep_patches=all_declared_dep_patches(bundle, manifest))
    if not tree.ok:
        logger.warning("optima: skip bundle %s, recursive scan failed: %s", bundle, tree.violations)
        return
    # The reviewed patchers' artifacts (compiled CUDA exts, dep overlays) must exist in
    # THIS process: sglang runs the model in spawned scheduler ranks, and the driver's
    # sys.modules preloads do not survive the spawn. Cache-hit fast after the driver's
    # prepare_candidate_environment built once. (2026-07-07: without this the engine
    # silently scored the shim's reference fallback — a phantom-kernel run.) A patcher
    # failure raises out to activate() -> load_failed receipt -> the eval refuses.
    from optima.rebuild import apply_rebuild_plan

    apply_rebuild_plan(bundle)
    # ONE module instance per SOURCE FILE, shared across ops: two slots declared on
    # the same source (e.g. the shallow + deep fused-epilogue entries sharing one IPC
    # workspace in module globals) must not get two module instances — each would
    # re-init its own comm state and the second barrier could interleave across ranks.
    loaded_by_src: dict[Path, object] = {}
    for op in manifest.ops:
        if op.slot not in SLOTS:
            continue
        src = resolve_source(bundle, op)
        scan = scan_path(src)
        if not scan.ok:  # defense-in-depth: re-scan in the worker before load
            logger.warning("optima: skip %s, failed scan: %s", op.slot, scan.violations)
            continue
        meta = json.loads((Path(bundle) / op.metadata).read_text()) if op.metadata else {}
        # ONE module instance per op source: pulling entry/prepare/setup via separate
        # load_entry calls would re-execute the module body per callable and split
        # them across different module namespaces (module-global state shared between
        # prepare and entry would vanish; sys.modules would point at the last copy
        # while entry closed over an earlier one — torch.compile re-imports by name).
        src_key = Path(src).resolve()
        module = loaded_by_src.get(src_key)
        if module is None:
            module = loaded_by_src[src_key] = load_module(src)
        if getattr(op, "override_point", None):
            # Override submission: compose the miner's epilogue into the validator-owned base
            # kernel -> a standard (entry, prepare) that flows through the normal dispatcher.
            from optima_kernels.override import build_override

            def _loader(name, _mod=module):
                fn = getattr(_mod, name, None)
                return fn if callable(fn) else None  # absent symbol (GPU-only device fn) -> None

            entry, prepare = build_override(op.slot, op.override_point, op.entry, _loader)
        else:
            entry = callable_from(module, op.entry)
            # (prepare, forward) slots: pull the 2nd callable too, so the runtime dispatcher
            # can run the miner's weight-layout transform once and feed `prepared` to forward.
            # (Until now prepare was only exercised by CPU `verify`; the block seam needs it
            # live.) None for forward-only slots.
            prepare = callable_from(module, op.prepare) if getattr(op, "prepare", None) else None
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
        # surface-opening change (e.g. fixing a broken backend on new hardware) is expressed. It is
        # untrusted, so correctness is gated by token-match vs the stock baseline
        # (framework_mode) and it MUST run in the no-egress isolation worker before the
        # subnet opens to untrusted miners.
        if getattr(op, "setup", None):
            callable_from(module, op.setup)()
            logger.info("optima: ran setup() for %s", op.slot)
