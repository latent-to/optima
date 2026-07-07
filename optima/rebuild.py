"""Rebuild-plan helper for the framework-mode escape hatch.

Framework mode lets a candidate open a larger backend surface than the narrow
tensor-in/out dispatcher can express (a backend swap, a source recompile). That is
the *fenced escape hatch*, NOT the core slot contract — see docs/SLOT_CONTRACT.md.

The hard rule here: a ``rebuild.json`` step may reference **only a validator-shipped,
reviewed patcher** that lives in this repo's dedicated patcher directory
(``optima/patchers/``), selected via ``repo_python``. It must NOT execute
bundle-supplied code — that would be arbitrary miner RCE in the candidate process,
which no-egress isolation bounds but does not prevent (it can still touch the
filesystem, the shared sglang install, secrets on the box). This mirrors how PyTorch
gates backends: you submit a patch to core to add one; you do not ship arbitrary
code into the dispatcher. A miner who needs a patcher gets it *reviewed and merged*
into ``optima/patchers/`` first; then a bundle's ``rebuild.json`` may select it by name.

CONTAINMENT IS NOT ENOUGH: allowing any ``.py`` under the repo root (the earlier
behavior) let a bundle ``runpy`` an arbitrary repo module as ``__main__`` — e.g. a
CLI whose ``__main__`` has side effects — which is not a "reviewed patcher" in any
meaningful sense. So the resolved script must live under ``optima/patchers/`` AND be a
``.py`` file; anything else is refused. The repo root is derived from THIS package's
location (deterministic), not the process CWD, and is overridable only by the operator
via ``OPTIMA_REPO_ROOT``.

(The earlier ``bundle_python`` step type — run an arbitrary script from the bundle —
is deliberately removed; it is rejected with a clear error.)
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

# Reviewed patchers live ONLY here (repo-relative). A repo_python step may select a file
# under this dir and nowhere else, so "reviewed patcher" is an enforced boundary, not an
# honor system. Empty today (the feature is forward-looking); a plan naming a missing
# patcher fails closed.
_PATCHER_SUBDIR = ("optima", "patchers")


class RebuildError(RuntimeError):
    pass


def _repo_root() -> Path:
    """The repo root. Deterministic (this package's parent), NOT the process CWD — the
    old ``Path.cwd()`` default made the patcher boundary depend on where the validator
    happened to launch from. ``OPTIMA_REPO_ROOT`` overrides for relocated deployments."""
    env = os.environ.get("OPTIMA_REPO_ROOT")
    if env:
        return Path(env).resolve()
    # optima/rebuild.py -> optima/ -> repo root
    return Path(__file__).resolve().parents[1]


def apply_rebuild_plan(bundle_path: str | Path) -> bool:
    """Apply ``rebuild.json`` from ``bundle_path`` if present.

    Returns True when a plan was found and applied. All paths are repo-relative or
    bundle-relative and containment-checked. Network isolation is handled by the
    caller before this function is invoked.
    """
    bundle = Path(bundle_path).resolve()
    plan_path = bundle / "rebuild.json"
    if not plan_path.exists():
        return False
    if not plan_path.is_file():
        raise RebuildError(f"rebuild plan is not a file: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise RebuildError("rebuild.json must be an object")
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise RebuildError("rebuild.json 'steps' must be a list")

    repo_root = _repo_root()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RebuildError(f"rebuild step {i} must be an object")
        typ = step.get("type")
        if typ == "repo_python":
            # ONLY validator-shipped, reviewed patchers under optima/patchers/. Never
            # bundle code, and never an arbitrary repo module.
            script = _safe_patcher_path(repo_root, str(step.get("path", "")))
            _run_python_script(script)
        elif typ == "bundle_python":
            raise RebuildError(
                "rebuild step 'bundle_python' is not allowed: a bundle may not execute its "
                "own code in the candidate process (arbitrary RCE). Use a validator-shipped, "
                "reviewed 'repo_python' patcher instead. See docs/SLOT_CONTRACT.md."
            )
        else:
            raise RebuildError(f"unsupported rebuild step type: {typ!r}")
    return True


def _safe_patcher_path(repo_root: Path, rel: str) -> Path:
    """Resolve ``rel`` to a reviewed patcher, refusing anything outside ``optima/patchers/``.

    ``rel`` may be given relative to the repo root (``optima/patchers/foo.py``) or to the
    patcher dir itself (``foo.py``); either way the RESOLVED path must be contained in the
    patcher dir, be a regular ``.py`` file, and not be a symlink (which could re-point
    outside the reviewed set)."""
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise RebuildError(f"patcher path must be a simple relative path: {rel!r}")
    patcher_dir = repo_root.joinpath(*_PATCHER_SUBDIR).resolve()
    # Accept either a repo-relative or a patcher-dir-relative spelling.
    candidate = (repo_root / rel) if rel.startswith(os.path.join(*_PATCHER_SUBDIR)) else (patcher_dir / rel)
    if candidate.is_symlink():
        raise RebuildError(f"patcher path must not be a symlink: {rel!r}")
    p = candidate.resolve()
    if p != patcher_dir and patcher_dir not in p.parents:
        raise RebuildError(
            f"patcher path escapes the reviewed patcher dir {os.path.join(*_PATCHER_SUBDIR)!r}: {rel!r}"
        )
    if p.suffix != ".py":
        raise RebuildError(f"patcher must be a .py file: {rel!r}")
    if p.is_symlink() or not p.is_file():
        raise RebuildError(f"reviewed patcher not found under {os.path.join(*_PATCHER_SUBDIR)!r}: {rel!r}")
    return p


def _run_python_script(script: Path) -> None:
    old_argv = sys.argv
    sys.argv = [str(script)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
