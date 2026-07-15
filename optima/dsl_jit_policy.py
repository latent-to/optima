"""Validator-owned allowlist for GPU-kernel DSL tracing-JIT compile entrypoints.

Single source of truth for the ONE carve-out in the engine-tree dynamic-import
policy (optima/engine_tree.py): a call like ``cute.compile(<@cute.jit fn>, ...)``.

WHY A CARVE-OUT AT ALL.  The engine-tree policy bans ``compile``/``eval``/
``exec``/``__import__`` (bare and as attribute calls) because each is a
string-to-code path that defeats the static inspectability the whole intake
pipeline rests on.  A DSL *tracing JIT* entrypoint is different in kind:
``cutlass.cute.compile`` consumes a Python *callable object* (the ``@cute.jit``
kernel, whose body is in the already-scanned bundle source) and lowers it to
PTX — there is no ``source_string -> code`` step.  Triton needs no entry here
at all — ``@triton.jit`` compiles lazily on first launch, no explicit call.

WHAT ACTUALLY KEEPS THIS SOUND (read before trusting the admission).  The
carve-out only decides whether ``<recv>.<attr>(...)`` raises during static
intake; it NEVER exempts any file from the full policy scan or from the no-egress
OCI execution fence.  So the load-bearing invariant is *not* "the receiver is
provably the pinned DSL" — a module global can always be rebound (reflection,
a wildcard import, a sibling file mutating ``sys.modules[...].recv``).  The real
invariant is stronger and reflection-proof: **every ``.compile`` target a
candidate can reach is either the trusted pinned DSL or a bundle symbol that was
itself scanned by the same two gates and runs only inside the OCI fence.**  A
substituted receiver therefore routes the call to scanned, sandboxed bundle code
that can do no more than the bundle's own module body already does — never to
unscanned code and never to ``builtins.compile``/``eval``/``exec`` (those stay
banned everywhere).  So substitution is a harmless routing detail, not an escape.

Given that, the admission below is still made as tight and *sound-as-stated* as
cheap same-file analysis allows (fail-closed on the vectors an adversarial review
found — wildcard import, in-file namespace reflection, PEP 695 type-param
shadowing, string-literal compile arguments), so the static picture matches
runtime for the common case and a reviewer is not misled by a false "bound once"
claim.  Cross-file / reflective substitution is the documented residual above,
and is safe for the reason stated (scanned + OCI-fenced), not because it cannot
happen.

Adding a DSL is one row in ``DSL_JIT_ENTRYPOINTS`` — but first verify, against
the pinned version, that the entrypoint truly never accepts a source *string*
(some Triton ``compile`` surfaces accept raw IR text; those must NOT be added).
Import-light on purpose (stdlib only): the engine-tree materializer imports this
without pulling torch/sglang; the AST sandbox (optima/sandbox.py) is an
independent gate with a deliberately different, broader compile stance and does
not consume this table.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DslJitEntrypoint:
    """One allowlisted ``<module>.<attr>(<callable>, ...)`` tracing-JIT entry.

    ``module`` is the fully-qualified, ABSOLUTE import path of the trusted DSL
    module (must be provided by the pinned base engine, never by the bundle).
    ``attr`` is the compile method name on that module.
    """

    module: str
    attr: str
    note: str = ""


# THE table.  One row per trusted DSL tracing-JIT compile entrypoint.
DSL_JIT_ENTRYPOINTS: tuple[DslJitEntrypoint, ...] = (
    DslJitEntrypoint(
        module="cutlass.cute",
        attr="compile",
        note="CuTe DSL: cute.compile(<@cute.jit callable>, *args) -> compiled kernel.",
    ),
)

# attr names that a table entry may admit — keeps the engine-tree ban and this
# carve-out talking about the same set (only 'compile' today).
ADMITTED_ATTRS: frozenset[str] = frozenset(e.attr for e in DSL_JIT_ENTRYPOINTS)

# Module dotted-paths, for quick membership tests during alias analysis.
_ENTRY_BY_MODULE: dict[str, DslJitEntrypoint] = {e.module: e for e in DSL_JIT_ENTRYPOINTS}

# Bare-name calls that manipulate a namespace by reflection — their presence in a
# file poisons the carve-out for that file (an unbounded, statically-unenumerable
# rebinding of the receiver could follow). globals/vars/locals are also banned by
# the AST sandbox; kept here so the carve-out is self-sufficient.
_REFLECTION_CALLS: frozenset[str] = frozenset(
    {"setattr", "delattr", "globals", "vars", "locals"}
)
# Reflective attribute names that reach or mutate a module/global namespace.
_REFLECTION_ATTRS: frozenset[str] = frozenset(
    {"__setattr__", "__delattr__", "__dict__", "__globals__", "__builtins__"}
)


def _all_bound_names(tree: ast.AST) -> dict[str, int]:
    """Count every binding of every name in the module, across all scopes.

    Fail-closed by construction: any construct that could rebind or shadow a
    name is counted (including PEP 695 type parameters), so a receiver admitted
    below is provably bound exactly once *by the import alias we inspected*
    within this file.  In-file reflection and wildcard imports — which can
    rebind names this pass cannot enumerate — are handled separately by
    ``_has_namespace_reflection`` (they poison the whole file's admission).
    """

    counts: dict[str, int] = {}

    def bind(name: str | None) -> None:
        if name:
            counts[name] = counts.get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bind(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node.name)
        elif isinstance(node, ast.arg):
            bind(node.arg)
        elif isinstance(node, ast.ExceptHandler):
            bind(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bind(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bind(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":  # wildcard handled by _has_namespace_reflection
                    bind(alias.asname or alias.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            bind(node.name)
        elif isinstance(node, ast.MatchMapping):
            bind(node.rest)
        else:
            # PEP 695 type parameters (def f[T] / class C[T] / type X[T] = ...):
            # ast.TypeVar / ast.ParamSpec / ast.TypeVarTuple all carry a .name str.
            type_param = getattr(ast, "TypeVar", ())
            param_spec = getattr(ast, "ParamSpec", ())
            type_var_tuple = getattr(ast, "TypeVarTuple", ())
            if isinstance(node, (type_param, param_spec, type_var_tuple)):
                bind(getattr(node, "name", None))
    return counts


def _sys_module_names(tree: ast.AST) -> set[str]:
    """Names bound to the ``sys`` module (``import sys`` / ``import sys as s``)."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sys":
                    names.add(alias.asname or "sys")
    return names


def _has_namespace_reflection(tree: ast.AST) -> bool:
    """Whether the module can rebind an unbounded set of names at runtime.

    Wildcard imports and in-file namespace reflection (setattr/delattr/globals/
    vars/locals, ``sys.modules`` access, ``__dict__``/``__setattr__`` and kin)
    can substitute the admitted receiver in ways ``_all_bound_names`` cannot see.
    Their mere presence in a file poisons that file's DSL-JIT carve-out — a
    fail-closed, whole-file disqualifier.  No kernel bundle in-repo uses any of
    these, so this never withdraws a legitimate admission.
    """

    sys_names = _sys_module_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                return True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _REFLECTION_CALLS:
                return True
        elif isinstance(node, ast.Attribute):
            if node.attr in _REFLECTION_ATTRS:
                return True
            # sys.modules[...] — the reflective handle to every module global.
            if (
                node.attr == "modules"
                and isinstance(node.value, ast.Name)
                and node.value.id in sys_names
            ):
                return True
    return False


def _import_alias_modules(tree: ast.AST) -> dict[str, str]:
    """Map name -> absolute dotted module for plain import aliases of table modules.

    Only two spellings resolve a table module to a single Name receiver:
      * ``import <module> as <name>``           (asname REQUIRED — bare
        ``import a.b`` binds ``a``, not ``a.b``, so it is not a receiver)
      * ``from <pkg> import <leaf> [as <name>]`` where ``pkg.leaf`` is a table
        module and the import is absolute (``level == 0``)
    Relative imports (``from . import x``) never name an external trusted DSL.
    """

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and alias.name in _ENTRY_BY_MODULE:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue
            for alias in node.names:
                dotted = f"{node.module}.{alias.name}"
                if dotted in _ENTRY_BY_MODULE:
                    aliases[alias.asname or alias.name] = dotted
    return aliases


def admitted_receivers(
    tree: ast.AST,
    *,
    module_resolves_locally: Callable[[tuple[str, ...]], bool] | None = None,
) -> dict[str, DslJitEntrypoint]:
    """Names usable as ``<name>.<attr>(...)`` DSL-JIT receivers (fail-closed).

    A name qualifies iff (1) the file performs no namespace reflection or
    wildcard import that could rebind names opaquely, (2) its SINGLE binding in
    the whole module is a plain absolute import alias of a table module, and
    (3) — when a resolver is supplied — that module does not resolve inside the
    contribution tree.  The resolver is optional only so this stdlib-only module
    imports cleanly without tree context; the sole in-tree caller
    (engine_tree.py) always supplies it, and any future caller MUST too or it
    silently reopens the vendored-``cutlass/`` shadow.
    """

    if _has_namespace_reflection(tree):
        return {}  # opaque rebinding possible -> withdraw the whole file
    counts = _all_bound_names(tree)
    aliases = _import_alias_modules(tree)
    admitted: dict[str, DslJitEntrypoint] = {}
    for name, module in aliases.items():
        if counts.get(name, 0) != 1:
            continue  # rebound / shadowed somewhere -> withdraw
        if module_resolves_locally is not None:
            parts = tuple(module.split("."))
            if any(
                module_resolves_locally(parts[:end]) for end in range(1, len(parts) + 1)
            ):
                continue  # a bundle-local module could shadow the trusted DSL
        admitted[name] = _ENTRY_BY_MODULE[module]
    return admitted


def _is_string_literal(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))


def is_admitted_call(node: ast.Call, receivers: dict[str, DslJitEntrypoint]) -> bool:
    """True iff ``node`` is an admitted ``<receiver>.<attr>(<object>, ...)`` call.

    Requires: an attribute call whose base is a bare Name in ``receivers`` and
    whose attribute equals that entry's ``attr``; a first positional argument
    that exists and is statically NOT a string/bytes literal (a tracing JIT is
    handed a callable object, never source text); and no string/bytes literal in
    any keyword argument.  A starred first argument (``*args``) is opaque and so
    rejected — fail closed rather than skip the check.
    """

    func = node.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
        return False
    entry = receivers.get(func.value.id)
    if entry is None or func.attr != entry.attr:
        return False
    if not node.args or isinstance(node.args[0], ast.Starred):
        return False  # no inspectable first positional -> fail closed
    if _is_string_literal(node.args[0]):
        return False
    if any(_is_string_literal(kw.value) for kw in node.keywords):
        return False
    return True
