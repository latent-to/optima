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
import re
import stat
import sys
from importlib import abc, import_module, machinery
from pathlib import Path

logger = logging.getLogger("optima.seam")
_ENGINE_TREE = "/optima/engine-tree"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_NAMESPACE = re.compile(r"optima_c_[0-9a-f]{64}\Z")


class _MaterializedNamespaceFinder(abc.MetaPathFinder):
    def __init__(self, root: str):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if path is not None or _NAMESPACE.fullmatch(fullname) is None:
            return None
        return machinery.PathFinder.find_spec(fullname, [self.root])


def _install_materialized_namespace() -> bool:
    """Expose only sealed contribution namespaces after child spawn preparation."""

    release_required = _truthy(os.environ.get("OPTIMA_RELEASE_REQUIRED"))
    root = _ENGINE_TREE
    release_digest = os.environ.get("OPTIMA_RELEASE_DESCRIPTOR_DIGEST", "")
    release_verified = os.environ.get("OPTIMA_RELEASE_VERIFIED", "")
    if (
        os.environ.get("OPTIMA_ENGINE_WORKER") != "1"
        or os.environ.get("OPTIMA_BUNDLE_PATH") != root
        or _DIGEST.fullmatch(os.environ.get("OPTIMA_ENGINE_TREE_DIGEST", "")) is None
        or _DIGEST.fullmatch(os.environ.get("OPTIMA_STACK_DIGEST", "")) is None
        or (
            release_required
            and (
                _DIGEST.fullmatch(release_digest) is None
                or release_verified != release_digest
            )
        )
    ):
        return False
    try:
        info = os.lstat(root)
    except OSError:
        return False
    if not stat.S_ISDIR(info.st_mode) or os.path.realpath(root) != root:
        return False
    if not any(
        isinstance(finder, _MaterializedNamespaceFinder) and finder.root == root
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _MaterializedNamespaceFinder(root))
    return True


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _release_abort(message: str) -> None:
    """Terminate signed-release startup past bootstrap's development catch-all."""

    raise SystemExit(f"optima signed release refused: {message}")


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


# Bundle is loaded once per process even though activate() may run many times
# (once per watched module import) and load_candidate_bundle() is re-entrant.
_bundle_loaded = False
# A direct device artifact cannot be admitted until SGLang has selected this
# scheduler rank's CUDA device and created its context.  Only the positive
# scheduler-entry gate may set this marker; output-path processes can import the
# context hook but have no authority to make it load a bundle.
_bundle_pending: tuple[str, bool] | None = None


def activate() -> None:
    """Install all seams; ARM (but never load) the bundle in active processes.

    Called by the bootstrap post-import hook after ANY seamed module loads. Each
    install no-ops until its module is present, so calling activate repeatedly patches
    whatever is available each time. The adapter list comes from the single seam table
    (optima/seams.py) — the same table the bootstrap watch-list and the compat canary
    use, so there is no parallel list to keep in sync.

    Miner code is NEVER imported here. Watched modules are imported by
    non-execution engine children too — sglang's spawned detokenizer transitively
    imports parallel_state and five other watched modules (measured on the
    B300 worker image, 2026-07-13) — and miner module-level code in an
    output-path process is the output-substitution surface (a bundle could
    patch detokenization and spoof benchmark-accuracy text downstream of the
    sampler). The load authority begins only in load_candidate_bundle(), invoked
    by scheduler_gate at run_scheduler_process entry. Ordinary bundles load
    there; direct artifacts remain disabled until the table-tracked post-device
    hook binds them to that scheduler rank's CUDA context. The engine-side
    active-member coverage gate then counts exactly the scheduler ranks
    (tp_size), and any extra armed process is a refusal.
    """
    release_required = _truthy(os.environ.get("OPTIMA_RELEASE_REQUIRED"))
    if not _IS_DRIVER:
        namespace_installed = _install_materialized_namespace()
        if release_required and not namespace_installed:
            _release_abort("materialized namespace did not reopen")

    from optima.registry import REGISTRY

    _install_adapters(release_required)

    if _IS_DRIVER:
        # Timing process: seams installed (pass-through) but we never load the
        # miner module here, so the timer is out of the miner's reach.
        REGISTRY.disable()
        return

    if _bundle_loaded or _bundle_pending is not None:
        # A watched module imported after the scheduler-entry load (lazy model
        # imports): adapters were (re)installed above.  A pending direct bundle
        # remains disabled until the post-device hook finalizes it.
        return

    bundle = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not bundle or not _truthy(os.environ.get("OPTIMA_ACTIVE")):
        if release_required:
            _release_abort("candidate activation was not armed")
    # Armed or baseline: stay pass-through until (unless) this process proves it
    # is a scheduler execution rank via load_candidate_bundle().
    REGISTRY.disable()


def _install_adapters(release_required: bool) -> None:
    from optima.registry import REGISTRY
    from optima.seams import SEAM_ADAPTERS

    for adapter in SEAM_ADAPTERS:
        try:
            mod = import_module(f"optima.integrations.{adapter.integration}")
            mod.install(REGISTRY)
        except Exception:  # noqa: BLE001 - never break engine startup
            logger.exception("optima: failed to install seam %s", adapter.name)
            if release_required:
                _release_abort(f"seam {adapter.name} did not install")


def load_candidate_bundle() -> None:
    """Load + enable the vetted bundle in THIS process; scheduler ranks only.

    The single caller is the scheduler_gate seam (optima/integrations/
    sglang_scheduler_gate.py) wrapping run_scheduler_process entry. Direct
    artifact rows are staged, not bound, until the post-device hook. Idempotent;
    a no-op in the driver, in baseline processes, and once loaded or staged.
    """
    release_required = _truthy(os.environ.get("OPTIMA_RELEASE_REQUIRED"))
    if _IS_DRIVER or _bundle_loaded or _bundle_pending is not None:
        return

    from optima.registry import REGISTRY

    bundle = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not bundle or not _truthy(os.environ.get("OPTIMA_ACTIVE")):
        if release_required:
            _release_abort("candidate activation was not armed")
        REGISTRY.disable()
        return

    _load_candidate_bundle_locked(
        bundle,
        REGISTRY,
        release_required,
        defer_direct_artifacts=True,
    )


def finalize_pending_candidate_bundle() -> None:
    """Bind a scheduler-staged direct bundle after rank-local CUDA setup.

    The sole caller is the validator-owned AFTER hook on
    ``ModelRunner.init_torch_distributed``.  That SGLang method has selected the
    exact rank device and initialized its process groups, but model loading,
    warmup, and CUDA-graph capture have not begun.  Non-scheduler processes
    never acquire ``_bundle_pending`` and therefore remain inert here.
    """

    global _bundle_pending
    if _IS_DRIVER or _bundle_loaded or _bundle_pending is None:
        return

    from optima.registry import REGISTRY

    bundle, release_required = _bundle_pending
    # Consume the authority before any CUDA-bound work.  A failed target-rank
    # finalization is terminal; a later draft/additional runner must not retry it
    # in another context.
    _bundle_pending = None
    if (
        not _truthy(os.environ.get("OPTIMA_ACTIVE"))
        or os.environ.get("OPTIMA_BUNDLE_PATH", "").strip() != bundle
    ):
        _bundle_pending = None
        REGISTRY.clear()
        from optima import receipts

        receipts.write(
            "load_failed",
            {"bundle": bundle, "reason": "pending candidate authority changed"},
        )
        if release_required:
            _release_abort("pending candidate authority changed before device binding")
        return
    _load_candidate_bundle_locked(bundle, REGISTRY, release_required)


def teardown_candidate_bundle(*, suppress_errors: bool = False) -> None:
    """Release sealed-artifact lifecycle state at scheduler-process exit.

    The scheduler gate invokes this after the engine function unwinds, never from
    an import hook or output-path process.  Direct artifact entries synchronize,
    run declared destroy exports, and retain accounting on any failure.
    """

    global _bundle_pending
    _bundle_pending = None
    try:
        from optima.artifact_runtime import shutdown_direct_artifact_runtimes

        shutdown_direct_artifact_runtimes()
    except Exception as exc:  # noqa: BLE001 - teardown evidence must survive
        logger.exception("optima: candidate artifact teardown failed")
        try:
            from optima import receipts

            receipts.write("teardown_failed", {"reason": str(exc)[:4096]})
        except Exception:  # noqa: BLE001 - preserve the initiating teardown error
            logger.exception("optima: failed to write teardown failure receipt")
        if not suppress_errors:
            raise


def _load_candidate_bundle_locked(
    bundle,
    REGISTRY,
    release_required: bool,
    *,
    defer_direct_artifacts: bool = False,
) -> None:
    global _bundle_loaded, _bundle_pending
    try:
        if defer_direct_artifacts:
            from optima.manifest import load_manifest

            manifest = load_manifest(bundle)
            if any(op.aot_exports for op in manifest.ops):
                pending = (bundle, release_required)
                if _bundle_pending not in (None, pending):
                    raise RuntimeError(
                        "scheduler already staged a different direct artifact bundle"
                    )
                _bundle_pending = pending
                REGISTRY.disable()
                logger.info(
                    "optima: staged direct bundle %s until rank-local CUDA setup",
                    bundle,
                )
                return
        _load_bundle_into_registry(bundle)
        REGISTRY.enable()
        if _truthy(os.environ.get("OPTIMA_STRICT")):
            # Surface kernel errors instead of silently falling back (debug/proof: a
            # failing kernel crashes the engine rather than masquerading as baseline).
            REGISTRY.set_strict(True)
        _bundle_loaded = True
        _bundle_pending = None
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
            if release_required:
                _release_abort("bundle registered no slots")
        # Registry-conditional adapters (defer_gate/moe_export install only once
        # the deep slot is registered) saw an empty registry on every pre-load
        # activate() pass. Retry them NOW rather than hoping a later watched
        # module import re-triggers activate() after this load.
        _install_adapters(release_required)
    except Exception:  # noqa: BLE001 - a bad bundle must not wedge the engine
        _bundle_pending = None
        logger.exception("optima: bundle load failed for %s", bundle)
        from optima import receipts

        receipts.write("load_failed", {"bundle": bundle, "reason": "exception during load"})
        REGISTRY.clear()
        if release_required:
            _release_abort("bundle activation failed")


def _load_bundle_into_registry(bundle: str) -> None:
    from optima.manifest import load_manifest, resolve_source
    from optima.registry import REGISTRY, KernelImpl, eligibility_from_metadata
    from optima.sandbox import callable_from, load_module, scan_path, scan_tree
    from optima.slots import SLOTS

    manifest = load_manifest(bundle)
    if any(op.setup for op in manifest.ops) and not _truthy(
        os.environ.get("OPTIMA_FRAMEWORK_MODE")
    ):
        raise RuntimeError(
            "bundle declares setup() but OPTIMA_FRAMEWORK_MODE is not armed; "
            "refusing engine-wide mutation"
        )
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

    # Production scheduler ranks consume only the publication produced by the
    # trusted OCI prebuild worker.  The explicit direct-eval lane has no native
    # publication (it uses the content-addressed development cache), so replay
    # the already-selected development plan there.  Never infer production from
    # one marker alone: a partial marker set must fail rather than downgrade a
    # hardened worker into the build-capable development lane.
    prebuilt = _truthy(os.environ.get("OPTIMA_PREBUILT_ARTIFACTS"))
    engine_worker = _truthy(os.environ.get("OPTIMA_ENGINE_WORKER"))
    if prebuilt != engine_worker:
        raise RuntimeError(
            "candidate scheduler has an incomplete native-artifact authority"
        )
    apply_rebuild_plan(bundle, phase="load" if prebuilt else "all")
    from optima.artifact_runtime import resolve_direct_artifact_entry

    # Resolve every direct-AOT binding before importing ANY candidate module.  A
    # mixed/atomic bundle therefore cannot monkeypatch CuTe, torch, an artifact path,
    # or adapter globals and redirect a later direct row.
    direct_entries = {
        (op.slot, op.variant): resolve_direct_artifact_entry(op)
        for op in manifest.ops
        if op.aot_exports
    }
    # ONE module instance per SOURCE FILE, shared across ops: two slots declared on
    # the same source (e.g. the shallow + deep fused-epilogue entries sharing one IPC
    # workspace in module globals) must not get two module instances — each would
    # re-init its own comm state and the second barrier could interleave across ranks.
    loaded_by_src: dict[Path, object] = {}
    # Variant deduplication only; framework admission was enforced above. This set
    # does not make candidate execution trusted—the later OCI/no-egress boundary does.
    setup_done: set[tuple[Path, str]] = set()
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
        module = None
        if op.aot_exports:
            entry = direct_entries[(op.slot, op.variant)]
            if entry is None:
                raise RuntimeError("direct CuTe AOT row resolved no validator entry")
            # Direct artifacts may carry validator-generated init/prepare/storage
            # lifecycle.  The runtime entry owns this method; no candidate Python
            # callable is imported in the scheduler process.
            prepare = getattr(entry, "prepare", None)
            if prepare is not None and not callable(prepare):
                raise RuntimeError("direct CuTe AOT prepare boundary is not callable")
        else:
            module = loaded_by_src.get(src_key)
            if module is None:
                module = loaded_by_src[src_key] = load_module(src)
        if op.aot_exports:
            pass
        elif getattr(op, "override_point", None):
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
                eligibility=eligibility_from_metadata(
                    meta, op.dtypes, op.architectures
                ),
                variant=op.variant,
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
            assert module is not None  # direct-AOT rows forbid setup at manifest load
            setup_key = (src_key, op.setup)
            if setup_key not in setup_done:
                callable_from(module, op.setup)()
                setup_done.add(setup_key)
                logger.info("optima: ran setup() for %s", op.slot)
