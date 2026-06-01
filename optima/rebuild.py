"""Rebuild-plan helper for the framework-mode escape hatch.

Framework mode lets a candidate open a larger backend surface than the narrow
tensor-in/out dispatcher can express (a backend swap, a source recompile). That is
the *fenced escape hatch*, NOT the core slot contract — see docs/SLOT_CONTRACT.md.

The hard rule here: a ``rebuild.json`` step may reference **only a validator-shipped,
reviewed patcher** that lives in this repo (``repo_python``). It must NOT execute
bundle-supplied code — that would be arbitrary miner RCE in the candidate process,
which no-egress isolation bounds but does not prevent (it can still touch the
filesystem, the shared sglang install, secrets on the box). This mirrors how PyTorch
gates backends: you submit a patch to core to add one; you do not ship arbitrary
code into the dispatcher. A miner who needs a patcher gets it *reviewed and merged*
into the repo first; then a bundle's ``rebuild.json`` may select it by relative path.

(The earlier ``bundle_python`` step type — run an arbitrary script from the bundle —
is deliberately removed; it is rejected with a clear error.)
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


class RebuildError(RuntimeError):
    pass


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

    repo_root = Path(os.environ.get("OPTIMA_REPO_ROOT", Path.cwd())).resolve()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RebuildError(f"rebuild step {i} must be an object")
        typ = step.get("type")
        if typ == "repo_python":
            # ONLY validator-shipped, reviewed patchers (repo-local). Never bundle code.
            script = _safe_repo_path(repo_root, str(step.get("path", "")))
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


def _safe_repo_path(repo_root: Path, rel: str) -> Path:
    if not rel or rel.startswith("/"):
        raise RebuildError(f"repo script path must be relative: {rel!r}")
    p = (repo_root / rel).resolve()
    if repo_root != p and repo_root not in p.parents:
        raise RebuildError(f"repo script path escapes repo: {rel!r}")
    if not p.is_file():
        raise RebuildError(f"repo script not found: {rel!r}")
    return p


def _run_python_script(script: Path) -> None:
    old_argv = sys.argv
    sys.argv = [str(script)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
