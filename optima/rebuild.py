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

import hashlib
import json
import os
import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Reviewed patchers live ONLY here (repo-relative). A repo_python step may select a
# registered file under this directory and nowhere else.
_PATCHER_SUBDIR = ("optima", "patchers")
_REGISTERED_PATCHERS = {
    "apply_dep_patch.py": ("optima.apply-dep-patch.v1", 0),
    "build_cuda_ext.py": ("optima.build-cuda-ext.v1", 1),
    "build_cute_cubin.py": ("optima.build-cute-cubin.v1", 2),
}


class RebuildError(RuntimeError):
    pass


RebuildPhase = Literal["all", "build", "load"]
_REBUILD_PHASES = frozenset({"all", "build", "load"})


@dataclass(frozen=True)
class RebuildStep:
    """One resolved validator-owned patcher invocation; never bundle code."""

    step_type: str
    patcher_id: str
    path: str
    patcher_sha256: str
    _script_path: Path = field(repr=False, compare=False)

    def snapshot(self) -> dict[str, str]:
        return {
            "type": self.step_type,
            "patcher_id": self.patcher_id,
            "path": self.path,
            "patcher_sha256": self.patcher_sha256,
        }

    def runtime_step(self) -> dict[str, str]:
        return {"type": self.step_type, "path": self.path}


@dataclass(frozen=True)
class RebuildPlan:
    schema_version: int
    steps: tuple[RebuildStep, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise RebuildError("rebuild schema_version must be 1")
        if not isinstance(self.steps, tuple) or not all(
            isinstance(step, RebuildStep) for step in self.steps
        ):
            raise RebuildError("rebuild steps must be a tuple of RebuildStep values")

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "steps": [step.snapshot() for step in self.steps],
        }

    identity_data = snapshot

    def to_dict(self) -> dict[str, object]:
        """Canonical validator-generated ``rebuild.json`` projection."""
        return {"steps": [step.runtime_step() for step in self.steps]}

def _repo_root() -> Path:
    """The repo root. Deterministic (this package's parent), NOT the process CWD — the
    old ``Path.cwd()`` default made the patcher boundary depend on where the validator
    happened to launch from. ``OPTIMA_REPO_ROOT`` overrides for relocated deployments."""
    env = os.environ.get("OPTIMA_REPO_ROOT")
    if env:
        return Path(env).resolve()
    # optima/rebuild.py -> optima/ -> repo root
    return Path(__file__).resolve().parents[1]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RebuildError(f"rebuild.json contains duplicate key {key!r}")
        result[key] = value
    return result


def _canonical_patcher_name(rel: object) -> str:
    if not isinstance(rel, str) or not rel or rel.startswith("/"):
        raise RebuildError(f"patcher path must be a canonical relative string: {rel!r}")
    path = Path(rel)
    if ".." in path.parts:
        raise RebuildError(f"patcher path must not contain traversal: {rel!r}")
    if len(path.parts) == 1:
        name = path.name
    elif path.parts == (*_PATCHER_SUBDIR, path.name):
        name = path.name
    else:
        raise RebuildError(
            f"patcher path must name a registered file under "
            f"{os.path.join(*_PATCHER_SUBDIR)!r}: {rel!r}"
        )
    if name not in _REGISTERED_PATCHERS:
        raise RebuildError(f"unregistered rebuild patcher {name!r}")
    return name


def _resolve_registered_patcher(repo_root: Path, name: str) -> Path:
    patcher_dir = repo_root.joinpath(*_PATCHER_SUBDIR).resolve()
    candidate = repo_root.joinpath(*_PATCHER_SUBDIR, name)
    if candidate.is_symlink():
        raise RebuildError(f"patcher path must not be a symlink: {name!r}")
    resolved = candidate.resolve()
    if patcher_dir not in resolved.parents:
        raise RebuildError(
            f"patcher path escapes reviewed directory {os.path.join(*_PATCHER_SUBDIR)!r}"
        )
    if not resolved.is_file():
        raise RebuildError(f"registered rebuild patcher not found: {name!r}")
    return resolved


def parse_rebuild_plan(bundle_path: str | Path) -> RebuildPlan | None:
    """Strictly parse and resolve rebuild authority without executing a patcher."""
    bundle = Path(bundle_path).resolve()
    plan_path = bundle / "rebuild.json"
    if plan_path.is_symlink():
        raise RebuildError(f"rebuild plan must be a regular non-symlink file: {plan_path}")
    if not plan_path.exists():
        return None
    if not plan_path.is_file():
        raise RebuildError(f"rebuild plan must be a regular non-symlink file: {plan_path}")
    try:
        plan = json.loads(
            plan_path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except RebuildError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RebuildError(f"invalid rebuild.json: {exc}") from exc
    if not isinstance(plan, dict):
        raise RebuildError("rebuild.json must be an object")
    if set(plan) != {"steps"}:
        raise RebuildError("rebuild.json must contain exactly the 'steps' key")
    raw_steps = plan["steps"]
    if not isinstance(raw_steps, list):
        raise RebuildError("rebuild.json 'steps' must be a list")

    repo_root = _repo_root()
    parsed: list[tuple[int, RebuildStep]] = []
    seen: set[str] = set()
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise RebuildError(f"rebuild step {index} must be an object")
        if set(step) != {"type", "path"}:
            raise RebuildError(
                f"rebuild step {index} must contain exactly 'type' and 'path'"
            )
        if step["type"] == "bundle_python":
            raise RebuildError(
                "rebuild step 'bundle_python' is not allowed: a bundle may not "
                "execute its own code"
            )
        if step["type"] != "repo_python":
            raise RebuildError(f"unsupported rebuild step type: {step['type']!r}")
        name = _canonical_patcher_name(step["path"])
        if name in seen:
            raise RebuildError(f"duplicate rebuild patcher {name!r}")
        seen.add(name)
        patcher_id, order = _REGISTERED_PATCHERS[name]
        script = _resolve_registered_patcher(repo_root, name)
        try:
            source_hash = hashlib.sha256(script.read_bytes()).hexdigest()
        except OSError as exc:
            raise RebuildError(
                f"cannot read registered rebuild patcher {name!r}"
            ) from exc
        parsed.append(
            (
                order,
                RebuildStep(
                    step_type="repo_python",
                    patcher_id=patcher_id,
                    path=f"{os.path.join(*_PATCHER_SUBDIR)}/{name}",
                    patcher_sha256=source_hash,
                    _script_path=script,
                ),
            )
        )
    parsed.sort(key=lambda row: row[0])
    return RebuildPlan(schema_version=1, steps=tuple(step for _, step in parsed))


def apply_rebuild_plan(
    bundle_path: str | Path, *, phase: RebuildPhase = "all"
) -> bool:
    """Apply ``rebuild.json`` from ``bundle_path`` if present.

    Returns True when a plan was found and applied. All paths are repo-relative or
    bundle-relative and containment-checked. ``build`` creates native products but
    must not load them, while ``load`` may validate and load an already-published
    product but must not compile or repair it. ``all`` preserves the old direct-eval
    behavior as an explicitly non-authoritative development path. Isolation and the
    phase-specific mounts are owned by the caller.
    """
    if phase not in _REBUILD_PHASES:
        raise RebuildError(f"unsupported rebuild phase: {phase!r}")
    bundle = Path(bundle_path).resolve()
    plan = parse_rebuild_plan(bundle)
    if plan is None:
        return False
    for step in plan.steps:
        try:
            current_hash = hashlib.sha256(step._script_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RebuildError(f"cannot reread rebuild patcher {step.path!r}") from exc
        if current_hash != step.patcher_sha256:
            raise RebuildError(f"rebuild patcher changed after parsing: {step.path!r}")
        _run_python_script(step._script_path, bundle=bundle, phase=phase)
    return True


def _run_python_script(
    script: Path, *, bundle: Path, phase: RebuildPhase
) -> None:
    """Run a reviewed patcher with the triggering bundle's path in the environment.

    ``OPTIMA_BUNDLE_PATH`` is the patcher contract: every caller of
    ``apply_rebuild_plan`` (engine launch, distributed-verify ranks, CLI smoke) hands
    the bundle path as an argument, so the plan must not depend on who set what env —
    the earlier env-only convention silently no-op'd patchers in verify ranks (the
    build skipped, the shim fell back to its reference path, and the "verify"
    validated nothing)."""
    old_argv = sys.argv
    old_bundle = os.environ.get("OPTIMA_BUNDLE_PATH")
    old_phase = os.environ.get("OPTIMA_REBUILD_PHASE")
    sys.argv = [str(script)]
    os.environ["OPTIMA_BUNDLE_PATH"] = str(bundle)
    os.environ["OPTIMA_REBUILD_PHASE"] = phase
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
        if old_bundle is None:
            os.environ.pop("OPTIMA_BUNDLE_PATH", None)
        else:
            os.environ["OPTIMA_BUNDLE_PATH"] = old_bundle
        if old_phase is None:
            os.environ.pop("OPTIMA_REBUILD_PHASE", None)
        else:
            os.environ["OPTIMA_REBUILD_PHASE"] = old_phase


def _main(argv: list[str] | None = None) -> int:
    """Container/development entry point; never a trusted-host rebuild API."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m optima.rebuild")
    parser.add_argument("--phase", choices=sorted(_REBUILD_PHASES), required=True)
    parser.add_argument("bundle")
    args = parser.parse_args(argv)
    if args.phase == "build" and os.environ.get("OPTIMA_REBUILD_CONTAINER") != "1":
        raise RebuildError(
            "build subprocess entry requires the disposable rebuild container"
        )
    if args.phase == "load" and os.environ.get("OPTIMA_ENGINE_WORKER") != "1":
        raise RebuildError("load subprocess entry requires an isolated engine worker")
    if args.phase == "all" and os.environ.get("OPTIMA_REBUILD_DEVELOPMENT") != "1":
        raise RebuildError(
            "combined rebuild entry is development-only; set "
            "OPTIMA_REBUILD_DEVELOPMENT=1 explicitly"
        )
    apply_rebuild_plan(args.bundle, phase=args.phase)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(_main())
