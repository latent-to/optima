"""Loading miner kernel source — defense-in-depth, not the trust boundary.

Read this carefully, because it is the most misunderstood part of the system.

With Triton/CuteDSL the miner's kernel is **Python that executes in-process**:
the ``@triton.jit`` body is traced and the surrounding launch code runs on the
CPU. There is therefore *no artifact* we can statically prove safe, and nothing
in this file makes it safe to import untrusted code into the validator process.

The real isolation must be provided by the host running this harness:

  * a separate process / PID+mount+net namespace with no network egress,
  * a per-evaluation CUDA context (MPS or one process per eval) so an
    out-of-bounds device write cannot corrupt other work,
  * a watchdog that kills the whole context on a hung kernel,
  * resource limits (RLIMIT_AS / RLIMIT_CPU) on the worker.

What this module DOES provide:

  1. ``scan_source`` — a cheap AST policy scan that rejects the obvious
     egress/escape patterns *before* anything is imported. This is a filter to
     cut noise and catch lazy attacks, not a sandbox.
  2. ``load_entry`` — import the module and pull out the slot's ``entry``
     callable, in-process, *after* the scan passes. Intended to be called from
     inside the isolated worker, never from the trusted validator process.
  3. ``probe_in_subprocess`` — best-effort: import the module in a
     resource-limited child (Linux) just to surface import/JIT errors and
     import-time payloads away from the caller.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Import roots a kernel has no business touching. Defense-in-depth only.
_BANNED_IMPORT_ROOTS = frozenset(
    {
        "socket", "ssl", "asyncio", "selectors",
        "subprocess", "multiprocessing", "concurrent",
        "ctypes", "cffi", "_ctypes",
        "requests", "urllib", "urllib2", "urllib3", "http", "httplib",
        "ftplib", "smtplib", "telnetlib", "poplib", "imaplib", "xmlrpc",
        "pickle", "dill", "marshal", "shelve", "cloudpickle",
        "importlib", "imp", "runpy", "pkg_resources", "pkgutil",
        "shutil", "tempfile", "pathlib", "glob", "fileinput",
        "pty", "fcntl", "termios", "resource", "signal", "mmap",
        "webbrowser", "wsgiref", "paramiko", "fabric",
    }
)

# Attribute calls that indicate egress / process control / dlopen. Kept narrow
# and unambiguous on purpose: broad names like ``load``/``run``/``replace`` are
# common in kernels (``tl.load``, ``s.replace``) and would false-positive, so
# deserialization is handled separately via a qualified-base check below.
_BANNED_ATTR_CALLS = frozenset(
    {
        "system", "popen", "fork", "forkpty",
        "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
        "spawnv", "spawnve", "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "kill", "killpg", "putenv", "setuid", "setgid",
        "dlopen", "LoadLibrary", "WinDLL", "CDLL",
        "check_output", "check_call", "Popen", "urlopen",
    }
)

# Deserialization is dangerous only when the base is a known (de)serializer.
# This catches ``pickle.loads``/``torch.load``/``dill.load`` without nuking
# Triton's ``tl.load``/``tl.store``.
_DESERIALIZE_ATTRS = frozenset({"load", "loads"})
_DESERIALIZE_BASES = frozenset(
    {"pickle", "cpickle", "_pickle", "dill", "marshal", "cloudpickle", "joblib", "torch"}
)

# Bare builtins that are almost always a code-execution smell in a kernel.
# Includes namespace-exposers (globals/vars/locals) used to reach a sandbox escape.
_BANNED_BUILTINS = frozenset(
    {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "globals", "vars", "locals"}
)

# Dynamic attribute access — the classic literal-AST-scan bypass
# (``getattr(os, 'sys'+'tem')``). Flagged ONLY when the attribute NAME is not a string
# literal; a literal ``getattr(self, 'forward')`` is fine and common in kernels.
_DYNAMIC_ATTR_FNS = frozenset({"getattr", "setattr", "delattr"})

# Dunder attribute names used in classic sandbox-escape chains. ``__class__`` is the
# entry hop of ``().__class__.__bases__[0].__subclasses__()`` and was previously missed.
_BANNED_DUNDERS = frozenset(
    {"__globals__", "__builtins__", "__subclasses__", "__bases__", "__mro__", "__code__",
     "__loader__", "__dict__", "__class__", "__subclasshook__", "__getattribute__", "__base__"}
)

# Names whose SUBSCRIPT is an escape (``__builtins__['eval']``), not just attribute access.
_BANNED_SUBSCRIPT_NAMES = frozenset({"__builtins__", "__globals__", "__dict__"})


def _is_string_literal(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


@dataclass(frozen=True)
class ScanResult:
    ok: bool
    violations: tuple[str, ...]


def scan_source(source: str, *, filename: str = "<kernel>") -> ScanResult:
    """Cheap AST policy scan. Returns violations; does not raise on findings."""
    out: list[str] = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return ScanResult(ok=False, violations=(f"{filename}: syntax error: {exc}",))

    for node in ast.walk(tree):
        # import socket / import urllib.request
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BANNED_IMPORT_ROOTS:
                    out.append(f"{filename}:{node.lineno}: banned import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _BANNED_IMPORT_ROOTS:
                out.append(f"{filename}:{node.lineno}: banned import-from {node.module!r}")
        # eval(...) / exec(...) / open(...) / globals(...) used as a bare name
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_BUILTINS:
                out.append(f"{filename}:{node.lineno}: banned builtin call {node.func.id!r}")
            # getattr/setattr/delattr with a NON-literal attribute name = dynamic-attr escape.
            elif node.func.id in _DYNAMIC_ATTR_FNS and len(node.args) >= 2 and not _is_string_literal(node.args[1]):
                out.append(f"{filename}:{node.lineno}: dynamic {node.func.id}() with a "
                           "non-literal attribute name (sandbox-escape pattern)")
        # __builtins__['eval'] / __globals__[...] subscript escape
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in _BANNED_SUBSCRIPT_NAMES:
            out.append(f"{filename}:{node.lineno}: banned subscript on {node.value.id!r}")
        # os.system(...) / subprocess.Popen(...) / ctypes.CDLL(...)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in _BANNED_ATTR_CALLS:
                out.append(f"{filename}:{node.lineno}: banned call .{attr}()")
            # qualified deserialization: pickle.loads(...) / torch.load(...)
            elif attr in _DESERIALIZE_ATTRS and isinstance(node.func.value, ast.Name):
                if node.func.value.id in _DESERIALIZE_BASES:
                    out.append(
                        f"{filename}:{node.lineno}: banned deserialization "
                        f"{node.func.value.id}.{attr}()"
                    )
        # attribute access to escape dunders — or to ALIAS a banned callable without
        # an ast.Call at the access site (``f = os.system; f("id")``). Flagging the
        # bare access double-reports a direct ``os.system(...)`` (once via the Call
        # branch, once here); harmless for a reject-on-any-violation scan.
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_DUNDERS:
                out.append(f"{filename}:{node.lineno}: banned attribute {node.attr!r}")
            elif node.attr in _BANNED_ATTR_CALLS:
                out.append(f"{filename}:{node.lineno}: banned attribute {node.attr!r} "
                           "(aliasable escape callable)")
            elif (node.attr in _DESERIALIZE_ATTRS and isinstance(node.value, ast.Name)
                    and node.value.id in _DESERIALIZE_BASES):
                out.append(f"{filename}:{node.lineno}: banned deserialization alias "
                           f"{node.value.id}.{node.attr}")

    return ScanResult(ok=not out, violations=tuple(out))


def scan_path(path: str | Path) -> ScanResult:
    p = Path(path)
    return scan_source(p.read_text(encoding="utf-8"), filename=p.name)


# Suffixes that are never allowed in a bundle tree, declared or not: compiled/binary
# artifacts that copy_fingerprint's import-closure walk cannot see and that scan_source
# cannot inspect. A miner cannot launder one of these past the scan by "declaring" it as
# a cuda_source (manifest.py's suffix check already blocks that at the manifest layer;
# this is the belt to that suspenders — scan_tree must reject them even if some future
# caller passes a bogus/wide-open allowlist). ``.pyc`` is only banned OUTSIDE
# ``__pycache__`` (the normal bytecode cache dir is skipped entirely, see below).
_BANNED_BINARY_SUFFIXES = frozenset({".so", ".o", ".a", ".dylib", ".bin"})

# Benign metadata a bundle may carry with no scan/declaration needed: docs, license
# text, the manifest itself, eligibility JSON, and git plumbing. Matched by exact
# stem+suffix rules below, not just suffix, so this stays narrow.
# ``rebuild.json`` is allowlisted by NAME because it is not free-form content: it is
# strictly validated by ``optima/rebuild.py`` (fail-closed; may only select reviewed
# patchers under ``optima/patchers/``) — the scan's job is done by that validator.
_BENIGN_METADATA_NAMES = frozenset({"manifest.toml", "rebuild.json", ".gitignore"})
_BENIGN_METADATA_STEM_PREFIXES = ("README", "LICENSE")


def _is_benign_metadata(rel: Path) -> bool:
    name = rel.name
    if name in _BENIGN_METADATA_NAMES:
        return True
    if name.startswith(_BENIGN_METADATA_STEM_PREFIXES):
        return True
    # *.json under a top-level (or nested) metadata/ directory.
    if rel.suffix == ".json" and "metadata" in rel.parts[:-1]:
        return True
    return False


def scan_tree(
    root: str | Path,
    *,
    declared_cuda_sources: "frozenset[Path] | set[Path] | None" = None,
    declared_dep_patches: "frozenset[Path] | set[Path] | None" = None,
) -> ScanResult:
    """Recursively scan EVERY ``.py`` under a bundle root — the vendored-tree guard.

    ``scan_path`` only covers the single declared entry module; a bundle can carry a whole
    vendored library, and a vendored module using ``open``/``importlib``/``subprocess`` must
    not slip in unscanned (the hole the single-file scan left). Aggregates violations across
    all files (skips ``__pycache__``). Still defense-in-depth, not a sandbox — but it closes
    the "ship the dangerous code in a file nobody scans" gap.

    ``declared_cuda_sources`` is the resolved-path allowlist from a bundle's manifest
    (``optima.manifest.all_declared_cuda_sources``) — the sanctioned "CUDA source" tier:
    a ``.cu``/``.cuh`` file the manifest declares for an op, compiled only by a
    validator-reviewed patcher (``optima/rebuild.py``), never scanned as Python here.

    Backward-compatible signature: ``declared_cuda_sources=None`` (the default, used by
    every existing call site that scans a tree without a loaded manifest) preserves the
    OLD behavior for anything that isn't a scanned ``.py`` — such a file is silently
    skipped, EXCEPT a file with a banned binary/artifact suffix (``.so``/``.o``/``.a``/
    ``.dylib``/``.bin``, or a stray ``.pyc`` outside ``__pycache__``) is now rejected
    unconditionally, declared or not, manifest or not. This is what closes the gap: a
    ``.cu``/``.so``/``.o`` used to be invisible to this scan yet still entered
    ``bundle_hash.content_hash`` (identity) and evaded ``copy_fingerprint``'s
    import-closure walk. Once a manifest IS available, pass its declared cuda sources and
    this function fails CLOSED on anything that is neither a scanned ``.py``, a declared
    cuda_source, nor the benign-metadata allowlist (README*, LICENSE*, ``manifest.toml``,
    ``*.json`` under ``metadata/``, ``.gitignore``) — so an undeclared ``.cu`` (or
    anything else) is rejected with a clear message instead of silently passing through.
    """
    root = Path(root)
    declared = frozenset(p.resolve() for p in (declared_cuda_sources or ()))
    declared_patches = frozenset(p.resolve() for p in (declared_dep_patches or ()))
    strict = declared_cuda_sources is not None or declared_dep_patches is not None
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root)
        # Fail-closed on ANY symlink: rglob does not follow directory symlinks, so a
        # symlinked dir full of .py would otherwise be silently invisible to this scan
        # (while still perfectly importable at runtime). A bundle has no business
        # containing symlinks at all. This also catches a symlinked "cuda_source" —
        # manifest.py already refuses to declare one, but a bundle can still place a
        # symlink at a path nobody declared, so this stays the backstop.
        if p.is_symlink():
            out.append(f"{rel}: symlink (not allowed in a bundle)")
            continue
        if not p.is_file():
            continue
        # Fail CLOSED on binary/artifact suffixes, unconditionally — these can never be
        # inspected by scan_source or folded into copy_fingerprint's closure walk, so no
        # declaration can sanction them.
        if p.suffix in _BANNED_BINARY_SUFFIXES or (p.suffix == ".pyc"):
            out.append(f"{rel}: binary/artifact file not allowed in a bundle ({p.suffix or 'no suffix'})")
            continue
        if p.suffix == ".py":
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:  # noqa: PERF203
                out.append(f"{rel}: unreadable: {exc}")
                continue
            out.extend(scan_source(text, filename=str(rel)).violations)
            continue
        if p.suffix in (".cu", ".cuh"):
            if p.resolve() in declared:
                continue  # sanctioned CUDA source tier — inspectable, not scanned as Python
            if strict:
                out.append(f"{rel}: .cu/.cuh file not declared in manifest cuda_sources")
            continue  # no manifest context (old call sites) -> preserve prior silent-skip
        if p.suffix in (".patch", ".diff"):
            # Sanctioned dep-patch tier: a DECLARED text unified diff was already
            # structurally validated at manifest load (optima/deppatch.py) and is applied
            # only by the one reviewed patcher against an arena allowlist. An UNDECLARED
            # patch file has no sanctioned reader — reject under a manifest, skip without.
            if p.resolve() in declared_patches:
                continue
            if strict:
                out.append(f"{rel}: .patch/.diff file not declared in manifest dep_patches")
            continue
        if strict:
            if _is_benign_metadata(rel):
                continue
            out.append(f"{rel}: file is neither a scanned .py, a declared cuda_source, "
                        "nor benign metadata")
            continue
        # Old behavior: anything else not covered above is silently skipped when no
        # manifest context was supplied (unchanged from before this hardening).
    return ScanResult(ok=not out, violations=tuple(out))


def load_module(source_path: str | Path):
    """Import ``source_path`` in-process and return the MODULE object.

    SECURITY: this executes the module body. Call it only from inside an isolated
    worker (separate process/namespace, no network, per-eval GPU context), never
    from the trusted validator process. The scan is enforced here as a tripwire,
    but it is not the boundary.

    Every call EXECUTES THE BODY AGAIN in a fresh module instance (and repoints
    ``sys.modules``) — so a caller that needs several callables from one op
    (entry + prepare + setup, or an override's device fns) must call this ONCE and
    ``getattr`` them all, or the callables end up in different module namespaces
    (module-global state shared between prepare and entry would silently vanish).
    """
    p = Path(source_path).resolve()
    scan = scan_path(p)
    if not scan.ok:
        raise PermissionError("kernel source failed policy scan:\n  " + "\n  ".join(scan.violations))

    import importlib.util
    import sys

    mod_name = f"optima_kernel_{p.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load kernel module from {p}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so the module is resolvable by name.
    # sglang 0.5.12+ traces the swapped kernel through torch.compile / piecewise
    # CUDA graph and imports it by module name; without this the scheduler raises
    # ModuleNotFoundError ("No module named 'optima_kernel_<stem>'") during capture.
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # runs the miner module body
    return module


def callable_from(module, name: str) -> Callable:
    """Pull a named callable off an already-loaded kernel module (see ``load_module``)."""
    fn = getattr(module, name, None)
    if not callable(fn):
        raise AttributeError(
            f"kernel module {getattr(module, '__name__', '?')} has no callable {name!r}")
    return fn


def load_entry(source_path: str | Path, entry: str) -> Callable:
    """``load_module`` + pull one callable. For SEVERAL callables from the same op use
    ``load_module`` once + ``callable_from`` — repeated ``load_entry`` calls re-execute
    the module body into separate instances (see ``load_module``)."""
    return callable_from(load_module(source_path), entry)


def probe_in_subprocess(source_path: str | Path, entry: str, *, cpu_seconds: int = 20, mem_mb: int = 4096) -> tuple[bool, str]:
    """Best-effort import probe in a resource-limited child process.

    Surfaces import-time errors / payloads away from the caller. On Linux applies
    RLIMIT_CPU and RLIMIT_AS. Returns ``(ok, message)``. This is a smoke check,
    not isolation — a real deployment runs the whole worker namespaced.
    """
    import subprocess
    import textwrap

    code = textwrap.dedent(
        f"""
        import resource, sys
        try:
            soft = {cpu_seconds}
            resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 2))
            resource.setrlimit(resource.RLIMIT_AS, ({mem_mb} * 1024 * 1024,) * 2)
        except Exception:
            pass
        from optima.sandbox import load_entry
        fn = load_entry({str(Path(source_path).resolve())!r}, {entry!r})
        sys.stdout.write("OK:" + getattr(fn, "__name__", "?"))
        """
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(sys.path),
        # No proxy/network creds; production should also drop the net namespace.
    }
    try:
        proc = subprocess.run(  # noqa: S603 - controlled argv, isolated child
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=cpu_seconds + 30,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "import probe timed out"
    ok = proc.returncode == 0 and proc.stdout.startswith("OK:")
    return ok, (proc.stdout + proc.stderr).strip()
