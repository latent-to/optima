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


def bundle_structural_fingerprint(bundle_root: str | Path) -> str:
    """Advisory structural fingerprint over a bundle's kernels + slot wiring (see
    ``structural_fingerprint``). "" if any source can't be parsed."""
    root = Path(bundle_root)
    manifest = load_manifest(root)
    parts: list[str] = []
    try:
        for op in sorted(manifest.ops, key=lambda o: o.slot):
            src = resolve_source(root, op)
            parts.append(op.slot + "\x00" + structural_fingerprint(src.read_text(encoding="utf-8")))
    except SyntaxError:
        return ""
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def bundle_fingerprint(bundle_root: str | Path) -> str:
    """A reformat-invariant fingerprint over a bundle's kernels + slot wiring.

    Covers, per op (sorted by slot): the slot id, the entry/prepare/setup callable
    names, and the NORMALIZED source of the op's module. Deliberately excludes
    ``bundle_id`` and the manifest's formatting — a copier reflowing the manifest or
    renaming the bundle does not change this. Returns "" if any source can't be
    parsed (fall back to exact-hash-only copy detection for that bundle).
    """
    root = Path(bundle_root)
    manifest = load_manifest(root)
    parts: list[str] = []
    try:
        for op in sorted(manifest.ops, key=lambda o: o.slot):
            src = resolve_source(root, op)
            norm = normalized_source(src.read_text(encoding="utf-8"))
            parts.append("\x00".join([
                op.slot, op.entry, op.prepare or "", op.setup or "",
                hashlib.sha256(norm.encode("utf-8")).hexdigest(),
            ]))
    except SyntaxError:
        return ""
    blob = "\x1e".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
