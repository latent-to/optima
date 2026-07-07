"""Which dependencies a bundle may patch, and where — VALIDATOR policy, never bundle data.

The ``dep_patches`` tier lets a bundle declare unified diffs against a PINNED
dependency tree (see optima/deppatch.py + optima/patchers/apply_dep_patch.py). This
module is the allowlist side of that contract: a patch target must have a row here,
and every file the patch touches must match this row's globs, or the one reviewed
applier hard-rejects the bundle. Nothing a bundle ships can widen this.

Consciously minimal and validator-owned. When the arena registry lands
(feat/arena-registry re-implementation — see the 2026-07-07 ledger), this table moves
onto ``Arena.patchable_deps`` so the allowlist is pinned per arena alongside the dep
versions it is valid against; keep the shape identical so that move is mechanical.

Why csrc-only for flashinfer: a .cu/.cuh/.h source patch is inspectable, fingerprints
like source (copy detection), and takes effect through a JIT rebuild the validator
controls (overlay + force-JIT — the runtime half). Patching PYTHON in a dependency is
NOT offered: dep Python executes in-process with validator privileges and would bypass
the sandbox scan that bundle Python goes through.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def overlay_base(bundle_id: str) -> Path:
    """Where the reviewed applier materializes overlays (and the runtime finds them).
    One definition — the patcher script and the runtime integration both use this."""
    root = os.environ.get("OPTIMA_DEP_OVERLAY_CACHE", "")
    base = Path(root) if root else Path.home() / ".cache" / "optima" / "dep_overlay"
    return base / bundle_id


@dataclass(frozen=True)
class DepPolicy:
    # Importable package whose site-packages install anchors the patch paths: a patch
    # path like "flashinfer/data/csrc/..." is resolved relative to the package's
    # site-root (the parent of the package directory).
    package: str
    # The subtree (site-root-relative, POSIX) that gets COPIED into the candidate-local
    # overlay. Must be broad enough that relative #includes inside it still resolve.
    overlay_subtree: str
    # fnmatch patterns (site-root-relative, POSIX; ``*`` crosses ``/``) for files a
    # patch may modify or create. Everything a patch touches must ALSO live under
    # overlay_subtree (else the patched file couldn't take effect via the overlay).
    allowed_globs: tuple[str, ...]
    # JitSpec names whose prebuilt AOT artifact must be bypassed at runtime so the
    # patched csrc actually compiles + loads (flashinfer prefers the AOT .so per spec
    # name; verified 2026-07-07 — see the ledger's overlay-assumptions report).
    force_jit_modules: tuple[str, ...] = ()
    # Runtime rebind coordinates: (module, attr) of the dependency's source-root
    # constant to repoint at ``<overlay>/<overlay_subtree>``. Must be LATE-BOUND at
    # every consumer site in the pinned dep (flashinfer's is, by upstream's own
    # documented design — env.py:17-19). None = the patched tree takes effect some
    # other way (no rebind step).
    env_rebind: tuple[str, str] | None = None


PATCHABLE_DEPS: dict[str, DepPolicy] = {
    "flashinfer": DepPolicy(
        package="flashinfer",
        overlay_subtree="flashinfer/data/csrc",
        # First occupant: the MoE cutlass backend (the fe_export deep seam lives in
        # fused_moe/cutlass_backend). Widen deliberately, per-arena, as slots demand.
        allowed_globs=("flashinfer/data/csrc/fused_moe/*",),
        force_jit_modules=("fused_moe_103",),
        env_rebind=("flashinfer.jit.env", "FLASHINFER_CSRC_DIR"),
    ),
}
