"""Near-copy fingerprint — catch reformatted/renamed copies the exact hash misses.

``bundle_hash.content_hash`` is an exact SHA-256: any byte change flips it. That is
right for *identity* but blind to the cheapest plagiarism in an open competition,
where every champion's source is public at reveal: copy the leader, reflow the
whitespace, restyle the comments, rewrite the docstring, and the content hash is
new — so exact copy-detection passes it as "original." King-of-the-hill stops a
*byte-identical* tie (it can't clear the margin), but a reformat + a one-constant
tweak can clear a noisy margin on stolen work.

This module adds a second, structural signal that survives cosmetic edits. It
parses each kernel to an AST, strips docstrings, and canonically re-emits it
(``ast.unparse``), so **whitespace, comments, docstrings, and redundant parens all
normalize away** — a reflowed/recommented/redocumented copy fingerprints IDENTICAL
to its source. It deliberately does NOT normalize identifier names or constants, so
two genuinely different kernels keep different fingerprints (near-zero false
positives — we never want to demote independent work). It is the conservative
"obvious reformat" catcher; a fuzzier structural-skeleton score is left for review
tooling, not the automatic demote path.

Pure-Python (no torch), so it runs in the trusted intake path next to the manifest
parse, exactly like ``content_hash``.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from optima.manifest import load_manifest, resolve_source


def _strip_docstrings(tree: ast.AST) -> None:
    """Drop the docstring (a leading bare string Expr) from every module/def/class."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]


def normalized_source(source: str) -> str:
    """Canonical form of Python source: comments/whitespace/docstrings/parens removed.

    Returns the ``ast.unparse`` of the docstring-stripped AST. Raises SyntaxError on
    unparseable source (the caller treats that as "no fingerprint").
    """
    tree = ast.parse(source)
    _strip_docstrings(tree)
    return ast.unparse(ast.fix_missing_locations(tree))


def source_fingerprint(source: str) -> str:
    return hashlib.sha256(normalized_source(source).encode("utf-8")).hexdigest()


class _Skeletonize(ast.NodeTransformer):
    """Blank identifier NAMES and constant VALUES, keeping call/attribute structure.

    So a copy that renames every variable and tweaks a constant (block size 64->128)
    skeletonizes IDENTICALLY to its source — the residue the reformat-invariant
    ``normalized_source`` misses. Attribute names are KEPT (``.silu`` vs ``.relu`` stays
    distinct), so two genuinely different kernels keep different skeletons. Higher
    false-positive risk than the normalized form (two simple kernels can share a
    skeleton), so this is an ADVISORY review signal, never an auto-demote."""

    def visit_Name(self, node):  # noqa: N802
        return ast.copy_location(ast.Name(id="_v", ctx=node.ctx), node)

    def visit_arg(self, node):  # noqa: N802
        node.arg = "_a"
        node.annotation = None
        return node

    def visit_Constant(self, node):  # noqa: N802
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)


def structural_source(source: str) -> str:
    tree = ast.parse(source)
    _strip_docstrings(tree)
    tree = _Skeletonize().visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree))


def structural_fingerprint(source: str) -> str:
    """Advisory near-copy signal robust to variable renames AND constant tweaks."""
    return hashlib.sha256(structural_source(source).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Transitive bundle-local import closure.
#
# A per-op fingerprint that hashes only the DECLARED entry module is blind to any
# ``.py`` that module imports from — a copier can move a champion's kernel body
# verbatim into ``kernels/_impl.py`` and make the entry a one-line re-export, and
# the body never enters the fingerprint at all. The closure folds every imported
# bundle-local file in. NOTE the closure HASH alone still cannot MATCH a relocated
# copy against a differently-laid-out original (equality is all-or-nothing over
# paths+content); the relocation-proof compare is the per-FILE set with ledger
# containment — ``bundle_slot_file_fingerprints`` below. The closure follows
# absolute (``kernels._impl``) and relative (``from ._impl import``) imports that
# resolve to a file UNDER the bundle root; external imports (torch, triton, …) are
# ignored. A visited set guards against import cycles.
# ---------------------------------------------------------------------------


def _resolve_module(root: Path, entry_rel: Path, node: ast.AST) -> list[Path]:
    """Bundle-local file(s) an Import / ImportFrom node could resolve to (may be empty)."""
    targets: list[tuple[int, str]] = []  # (relative-import level, dotted module or "")
    if isinstance(node, ast.Import):
        targets = [(0, alias.name) for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        targets = [(node.level or 0, node.module or "")]
        # ``from .pkg import name`` where ``name`` is itself a submodule file.
        for alias in node.names:
            sub = ((node.module + ".") if node.module else "") + alias.name
            targets.append((node.level or 0, sub))
    out: list[Path] = []
    for level, dotted in targets:
        if level:  # relative: resolve against the entry module's package
            base = entry_rel.parent
            for _ in range(level - 1):
                base = base.parent
            parts = base.parts + tuple(p for p in dotted.split(".") if p)
        else:  # absolute: rooted at the bundle
            parts = tuple(p for p in dotted.split(".") if p)
        if not parts:
            continue
        stem = Path(*parts)
        for cand in (stem.with_suffix(".py"), stem / "__init__.py"):
            abs_cand = (root / cand)
            try:
                abs_cand.resolve().relative_to(root.resolve())
            except (ValueError, OSError):
                continue  # escapes the bundle -> not a local module
            if abs_cand.is_file():
                out.append(root / cand)
    return out


def _closure_files(root: Path, entry: Path) -> list[Path]:
    """Entry module + every bundle-local ``.py`` it transitively imports, sorted by relpath."""
    root = Path(root)
    seen: set[Path] = set()
    order: list[Path] = []
    stack = [Path(entry)]
    while stack:
        f = stack.pop()
        try:
            key = f.resolve()
        except OSError:
            continue
        if key in seen or not f.is_file():
            continue
        seen.add(key)
        order.append(f)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        entry_rel = f.resolve().relative_to(root.resolve())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                stack.extend(_resolve_module(root, entry_rel, node))
    return sorted(order, key=lambda p: str(p.resolve().relative_to(root.resolve())))


def _closure_norm(root: Path, entry: Path, transform) -> str:
    """Canonical join of ``transform(source)`` over the entry's import closure.

    ``transform`` is ``normalized_source`` (reformat-invariant) or ``structural_source``
    (rename/constant-tweak skeleton). Files are joined in sorted-relpath order so the
    result is stable regardless of import discovery order.

    A NON-entry closure file that fails to parse/decode is SKIPPED, not fatal: such a
    file cannot execute at runtime either (importing it would raise), so it carries no
    kernel code — while treating it as fatal would let a copier disable the near-copy
    fingerprint on purpose by "importing" a deliberately-broken module behind
    ``if False:``. Only an unrunnable ENTRY keeps the ""/{} no-fingerprint fallback.
    """
    root = Path(root)
    entry_key = Path(entry).resolve()
    parts: list[str] = []
    for f in _closure_files(root, entry):
        rel = str(f.resolve().relative_to(root.resolve()))
        try:
            parts.append(rel + "\x00" + transform(f.read_text(encoding="utf-8")))
        except (SyntaxError, OSError, UnicodeDecodeError):
            if f.resolve() == entry_key:
                raise
    return "\x1e".join(parts)


def _op_identity(op) -> str:
    """The non-source identity of an op: slot + callable names + override composition.

    ``base_kernel`` / ``override_point`` are INCLUDED — an M1 override submission JIT-composes
    base+override at load, so the same epilogue source composed at a different hole (or into a
    different base) is a genuinely different kernel and must fingerprint distinctly, or honest
    work gets demoted as a self-collision.
    """
    return "\x00".join([
        op.slot, op.entry, op.prepare or "", op.setup or "",
        op.base_kernel or "", op.override_point or "",
    ])


def bundle_slot_fingerprints(bundle_root: str | Path) -> dict[str, str]:
    """Per-slot reformat-invariant fingerprint over each op's transitive import closure.

    An auto-demote key (exact equality; ``bundle_slot_file_fingerprints`` adds the
    relocation-proof containment compare). Keyed by slot so a copier cannot perturb a stolen slot's
    fingerprint by PADDING the bundle with an unrelated extra op (each slot is compared
    independently). Covers the op identity (``_op_identity``, incl. override composition)
    and the NORMALIZED source of the whole bundle-local import closure (so a body hidden
    in an imported module is folded in). ``{}`` if any closure source can't be parsed.
    """
    root = Path(bundle_root)
    manifest = load_manifest(root)
    out: dict[str, str] = {}
    try:
        for op in manifest.ops:
            closure = _closure_norm(root, resolve_source(root, op), normalized_source)
            blob = _op_identity(op) + "\x1e" + closure
            out[op.slot] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    except SyntaxError:
        return {}
    return out


# Normalized-source length below which a closure file is boilerplate (an empty
# ``__init__``, a one-line re-export shim) and is EXCLUDED from the per-file set —
# trivial shims would otherwise collide across honest bundles. A real kernel body
# normalizes far above this.
_SUBSTANTIAL_NORM_LEN = 80


def bundle_slot_file_fingerprints(bundle_root: str | Path) -> dict[str, list[str]]:
    """Per-slot, PATH-INDEPENDENT fingerprints of each substantial closure file.

    The relocation-proof copy signal. The closure hash above is an all-or-nothing
    EQUALITY, so a copier who moves a stolen body verbatim into an imported
    ``kernels/_impl.py`` (or pads the closure with one extra file) perturbs it into
    freshness — but the stolen file itself still fingerprints identically here no
    matter where it lives or what sits next to it. The ledger demotes a reveal when,
    for some slot, every substantial file of a prior reveal appears in this one (or
    vice versa) — "all of their work is inside yours" — a condition a merely-shared
    vendored utility can never trigger on its own (see ``Ledger.reveal``).

    Skips closure files that don't parse/decode (they cannot execute at runtime
    either); ``{}`` only if an op's ENTRY can't be parsed, matching the other maps.
    """
    root = Path(bundle_root)
    manifest = load_manifest(root)
    out: dict[str, list[str]] = {}
    try:
        for op in manifest.ops:
            entry = resolve_source(root, op)
            entry_key = entry.resolve()
            fps: set[str] = set()
            for f in _closure_files(root, entry):
                try:
                    norm = normalized_source(f.read_text(encoding="utf-8"))
                except (SyntaxError, OSError, UnicodeDecodeError):
                    if f.resolve() == entry_key:
                        raise
                    continue
                if len(norm) >= _SUBSTANTIAL_NORM_LEN:
                    fps.add(hashlib.sha256(norm.encode("utf-8")).hexdigest())
            out[op.slot] = sorted(fps)
    except SyntaxError:
        return {}
    return out


def bundle_slot_structural_fingerprints(bundle_root: str | Path) -> dict[str, str]:
    """Advisory per-slot structural (rename/constant-tweak) fingerprint over the closure."""
    root = Path(bundle_root)
    manifest = load_manifest(root)
    out: dict[str, str] = {}
    try:
        for op in manifest.ops:
            closure = _closure_norm(root, resolve_source(root, op), structural_source)
            out[op.slot] = hashlib.sha256((op.slot + "\x1e" + closure).encode("utf-8")).hexdigest()
    except SyntaxError:
        return {}
    return out


def _fold(slot_map: dict[str, str]) -> str:
    """Fold a per-slot map into one canonical hash (for logging / audit; "" if empty)."""
    if not slot_map:
        return ""
    blob = "\x1e".join(f"{slot}\x00{fp}" for slot, fp in sorted(slot_map.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def bundle_structural_fingerprint(bundle_root: str | Path) -> str:
    """Advisory whole-bundle structural fingerprint (fold of the per-slot map). "" if
    any source can't be parsed."""
    return _fold(bundle_slot_structural_fingerprints(bundle_root))


def bundle_fingerprint(bundle_root: str | Path) -> str:
    """A reformat-invariant fingerprint over a bundle's kernels + slot wiring.

    Whole-bundle fold of ``bundle_slot_fingerprints`` (retained for audit/logging); the
    LOAD-BEARING copy-compare is per-slot via ``bundle_slot_fingerprints`` so an extra
    padding op cannot perturb a stolen slot. Covers, per op: the slot id, the
    entry/prepare/setup callable names, the override composition, and the NORMALIZED
    source of the op's transitive bundle-local import closure. Deliberately excludes
    ``bundle_id`` and the manifest's formatting. "" if any source can't be parsed.
    """
    return _fold(bundle_slot_fingerprints(bundle_root))
