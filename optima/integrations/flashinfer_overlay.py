"""Runtime consume side of the ``dep_patches`` tier — repoint flashinfer at the overlay.

The reviewed patcher (optima/patchers/apply_dep_patch.py) materializes a patched COPY
of the pinned dependency's source subtree under the overlay cache; the shared install
is never mutated. THIS module, installed via the seam table when ``flashinfer.jit.core``
imports in an ACTIVE candidate rank, makes the engine actually compile+load from it:

  1. rebind the policy's late-bound source-root constant (for flashinfer:
     ``jit_env.FLASHINFER_CSRC_DIR`` — every one of its ~280 consumer sites is
     attribute-late-bound BY UPSTREAM DESIGN, env.py:17-19; flashinfer's own
     ``aot.py`` uses this exact rebind) at the overlay's subtree;
  2. force JIT for the policy's module names by overriding ``JitSpec.is_aot`` to
     return False for exactly those names — the AOT prebuilt ``.so`` (e.g.
     ``fused_moe_103``, 176MB in flashinfer_jit_cache) would otherwise shadow the
     patched csrc silently (the one silent-break gap in the overlay design);
  3. clear ``get_cutlass_fused_moe_module``'s functools cache if it was already
     consulted (a cached module object would ignore the is_aot override);
  4. write an ``overlay`` receipt — positive evidence for the eval driver.

Everything here keys off VALIDATOR policy (optima/dep_policy.py) + the applier's
overlay.json stamp; bundle content decides nothing. Baseline ranks (OPTIMA_ACTIVE
unset) and bundles without dep_patches no-op. The candidate-local JIT workspace is
handled separately: ``FLASHINFER_WORKSPACE_BASE`` is a real env var read once at
``flashinfer.jit.env`` import, so the eval driver exports it before the engine spawns
(optima/eval/_launch.py) — it cannot be rebound here.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys

logger = logging.getLogger("optima.flashinfer_overlay")

_installed = False


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _active_overlays() -> list[tuple[str, dict, "object"]]:
    """(target, overlay.json payload, policy) for every overlay the reviewed patcher
    materialized for the ACTIVE bundle. [] whenever this process isn't an active
    candidate or nothing was materialized."""
    bundle = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not bundle or not _truthy(os.environ.get("OPTIMA_ACTIVE")):
        return []
    from optima.dep_policy import PATCHABLE_DEPS, overlay_base
    from optima.manifest import ManifestError, load_manifest

    try:
        manifest = load_manifest(bundle)
    except (ManifestError, OSError):
        return []
    out = []
    for target in sorted({dp.target for dp in manifest.dep_patches}):
        policy = PATCHABLE_DEPS.get(target)
        if policy is None:
            continue  # the applier already hard-rejected; never guess here
        stamp = overlay_base(manifest.bundle_id) / target / "overlay.json"
        if not stamp.is_file():
            logger.warning(
                "optima: bundle declares dep_patches for %r but no overlay is "
                "materialized at %s — the rebuild plan did not run; the engine "
                "will serve the UNPATCHED dependency", target, stamp)
            continue
        try:
            data = json.loads(stamp.read_text())
        except (OSError, ValueError):
            logger.exception("optima: unreadable overlay stamp %s", stamp)
            continue
        out.append((target, data, policy))
    return out


def install(registry) -> None:  # registry unused; signature shared by all integrations
    global _installed
    if _installed:
        return
    overlays = _active_overlays()
    if not overlays:
        return

    from optima import receipts
    from optima.dep_policy import overlay_base

    force_jit: set[str] = set()
    applied: list[str] = []
    for target, data, policy in overlays:
        root = overlay_base(data["bundle_id"]) / target
        if policy.env_rebind is not None:
            mod_name, attr = policy.env_rebind
            env_mod = importlib.import_module(mod_name)
            new_root = root / policy.overlay_subtree
            if not new_root.is_dir():
                raise RuntimeError(f"overlay subtree missing on disk: {new_root}")
            setattr(env_mod, attr, new_root)
            logger.info("optima: %s.%s -> %s", mod_name, attr, new_root)
        force_jit.update(policy.force_jit_modules)
        applied.append(target)

    if force_jit:
        core = importlib.import_module("flashinfer.jit.core")
        orig_fget = core.JitSpec.is_aot.fget
        names = frozenset(force_jit)
        core.JitSpec.is_aot = property(
            lambda self: False if self.name in names else orig_fget(self))
        logger.info("optima: forcing JIT (AOT bypass) for %s", sorted(names))
        # A cached module getter that already ran would hand back the AOT build and
        # silently ignore everything above.
        fm = sys.modules.get("flashinfer.fused_moe.core")
        if fm is not None and hasattr(getattr(fm, "get_cutlass_fused_moe_module", None),
                                      "cache_clear"):
            fm.get_cutlass_fused_moe_module.cache_clear()

    _installed = True
    receipts.write("overlay", {"targets": applied, "force_jit": sorted(force_jit)})
    print(f"[optima] dep overlay ACTIVE: targets={applied} force_jit={sorted(force_jit)}",
          flush=True)
