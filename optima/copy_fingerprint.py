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

Layers, and what covers what:
  * Python kernels get all three: exact hash, AST-normalized (reformat-invariant),
    and AST-skeleton (rename/constant-tweak, advisory).
  * Declared ``cuda_sources`` (the sanctioned CUDA-source tier, see
    ``optima/manifest.py``) get only TWO of the three: an exact sha256 of the raw
    bytes, plus a regex-based reformat-invariant normalization (strip ``//`` and
    ``/* */`` comments, collapse whitespace) — folded into the SAME per-file set
    ``bundle_slot_file_fingerprints`` returns, so the relocation-proof containment
    compare covers ``.cu``/``.cuh`` bodies too. There is deliberately NO CUDA parser
    here and therefore no structural-skeleton layer for CUDA: a rename-every-identifier
    + tweak-a-constant copy of a ``.cu`` file is NOT caught by anything in this module.
    That gap is accepted for now (CUDA sources are validator-reviewed via
    ``optima/rebuild.py`` before compilation, which is a stronger check than an advisory
    skeleton fingerprint would be); revisit if the CUDA-source tier grows before a real
    C-family parser is worth building.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from optima.manifest import (load_manifest, resolve_cuda_sources, resolve_dep_patches,
                             resolve_source)


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
# CUDA source normalization (the sanctioned cuda_sources tier — see manifest.py).
#
# Deliberately NOT a CUDA/C++ parser: no AST, no identifier/constant skeletonization,
# so rename+constant-tweak evasion is NOT caught for .cu/.cuh (documented at the top
# of this module). This is regex-based reformat-invariance only: strip // line
# comments and /* */ block comments, then collapse all whitespace runs to a single
# space. That is enough to make a reflowed/recommented copy of a .cu file fingerprint
# identically, which is the same bar the .py normalized_source() clears.
# ---------------------------------------------------------------------------

_CUDA_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_CUDA_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def normalized_cuda_source(source: str) -> str:
    """Reformat-invariant normalization of CUDA source: strip comments, collapse whitespace.

    No parsing — a best-effort textual normalization, not a semantic one. String/char
    literals containing ``//`` or ``/*`` are not specially protected (a rare false
    positive here just under-normalizes, it never over-matches unrelated kernels).
    """
    no_comments = _CUDA_BLOCK_COMMENT_RE.sub(" ", source)
    no_comments = _CUDA_LINE_COMMENT_RE.sub("", no_comments)
    return _WHITESPACE_RE.sub(" ", no_comments).strip()


def cuda_source_fingerprint(source: str) -> str:
    return hashlib.sha256(normalized_cuda_source(source).encode("utf-8")).hexdigest()


def normalized_dep_patch(source: str) -> str:
    """Reformat-invariant normalization of a unified diff (the dep_patches tier).

    What a patch DOES is its +/- lines and the files it touches; everything else is
    presentation an evader can regenerate freely — hunk headers move with -U context
    width, context lines multiply with it, git headers come and go. Keep only the
    file headers and the +/- payload, whitespace-collapsed. A patch re-emitted with
    different context width / offsets / comments-in-context fingerprints identically;
    changing what it actually changes does not.
    """
    keep: list[str] = []
    for ln in source.splitlines():
        if ln.startswith(("--- ", "+++ ")):
            keep.append(_WHITESPACE_RE.sub(" ", ln.strip()))
        elif ln.startswith(("+", "-")):
            keep.append(_WHITESPACE_RE.sub(" ", ln[1:].strip()))
    return "\n".join(keep)


def dep_patch_fingerprint(source: str) -> str:
    return hashlib.sha256(normalized_dep_patch(source).encode("utf-8")).hexdigest()


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
    work gets demoted as a self-collision. ``variant`` is deliberately EXCLUDED: it is a routing
    label, not source identity, and renaming it must not evade copy detection. Capability metadata
    is likewise excluded, so relabeling a stolen implementation's domain does not make it fresh.
    """
    return "\x00".join([
        op.slot, op.entry, op.prepare or "", op.setup or "",
        op.base_kernel or "", op.override_point or "",
    ])


def _aggregate_variant_fingerprints(parts: list[str], *, domain: str) -> str:
    """Canonically fold every implementation row for one slot.

    A singleton returns its component digest verbatim so existing ledger fingerprints remain
    stable. Multiple rows are a sorted multiset: manifest order and variant labels are irrelevant,
    while duplicate implementations still contribute twice and adding/removing/changing either
    implementation changes the aggregate.
    """
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    blob = domain + "\x00" + "\x1e".join(sorted(parts))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def bundle_slot_fingerprints(bundle_root: str | Path) -> dict[str, str]:
    """Per-slot reformat-invariant fingerprint over each op's transitive import closure.

    An auto-demote key (exact equality; ``bundle_slot_file_fingerprints`` adds the
    relocation-proof containment compare). Keyed by slot so a copier cannot perturb a stolen slot's
    fingerprint by PADDING the bundle with an unrelated extra op (each slot is compared
    independently). Covers the op identity (``_op_identity``, incl. override composition)
    and the NORMALIZED source of the whole bundle-local import closure (so a body hidden
    in an imported module is folded in). Multiple variants of one slot are folded as a
    canonical, order-independent multiset; the historical singleton digest is unchanged.
    ``{}`` if any closure source can't be parsed.
    """
    root = Path(bundle_root)
    manifest = load_manifest(root)
    components: dict[str, list[str]] = {}
    try:
        for op in manifest.ops:
            closure = _closure_norm(root, resolve_source(root, op), normalized_source)
            blob = _op_identity(op) + "\x1e" + closure
            components.setdefault(op.slot, []).append(
                hashlib.sha256(blob.encode("utf-8")).hexdigest()
            )
    except SyntaxError:
        return {}
    return {
        slot: _aggregate_variant_fingerprints(
            fingerprints,
            domain="optima.slot.normalized-variants.v1",
        )
        for slot, fingerprints in components.items()
    }


# Normalized-source length below which a closure file is boilerplate (an empty
# ``__init__``, a one-line re-export shim) and is EXCLUDED from the per-file set —
# trivial shims would otherwise collide across honest bundles. A real kernel body
# normalizes far above this.
_SUBSTANTIAL_NORM_LEN = 80


def _bound_top_level_names(node: ast.stmt) -> set[str]:
    """Names one module-level statement binds for dependency projection."""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Import):
        return {
            alias.asname or alias.name.split(".", 1)[0]
            for alias in node.names
        }
    if isinstance(node, ast.ImportFrom):
        return {
            alias.asname or alias.name
            for alias in node.names
            if alias.name != "*"
        }

    def _targets(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return set().union(*(_targets(item) for item in target.elts))
        return set()

    if isinstance(node, ast.Assign):
        return set().union(*(_targets(target) for target in node.targets))
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _targets(node.target)
    return set()


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _top_level_definition_fingerprints(source: str) -> set[str]:
    """Path-independent fingerprints for each substantial top-level definition.

    Whole-file hashes catch relocation but are vulnerable to same-file padding: a
    copier can retain a stolen function and append an unrelated sibling.  For each
    function/class, retain both its standalone normalized body and the minimal module
    projection containing the top-level imports/constants/helpers it references.
    Unrelated sibling definitions and imports therefore cannot perturb the copied
    implementation's containment signal.

    The dependency projection also preserves backward compatibility for the common
    one-import/one-entry module: its projection is byte-identical to the historical
    whole-file normalized fingerprint.  The standalone body protects new records
    against padding an existing import statement itself.
    """

    tree = ast.parse(source)
    body = list(tree.body)
    bound_at: dict[str, int] = {}
    future_imports: set[int] = set()
    for index, node in enumerate(body):
        for name in _bound_top_level_names(node):
            bound_at[name] = index
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        ):
            future_imports.add(index)

    definitions = {
        index
        for index, node in enumerate(body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    fingerprints: set[str] = set()

    def _add_projection(indexes: set[int], *, prune_imports: bool = False) -> None:
        used_names = set().union(*(_loaded_names(body[index]) for index in indexes))
        projected_body: list[ast.stmt] = []
        for index in sorted(indexes):
            node = body[index]
            if prune_imports and isinstance(node, ast.Import):
                aliases = [
                    alias
                    for alias in node.names
                    if (alias.asname or alias.name.split(".", 1)[0]) in used_names
                ]
                if not aliases:
                    continue
                node = ast.Import(names=aliases)
            elif (
                prune_imports
                and isinstance(node, ast.ImportFrom)
                and node.module != "__future__"
            ):
                aliases = [
                    alias
                    for alias in node.names
                    if alias.name == "*"
                    or (alias.asname or alias.name) in used_names
                ]
                if not aliases:
                    continue
                node = ast.ImportFrom(
                    module=node.module,
                    names=aliases,
                    level=node.level,
                )
            projected_body.append(node)
        projection = ast.Module(
            body=projected_body,
            type_ignores=[],
        )
        normalized = normalized_source(ast.unparse(projection))
        if len(normalized) >= _SUBSTANTIAL_NORM_LEN:
            fingerprints.add(
                hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            )

    for definition_index in definitions:
        _add_projection({definition_index})

        selected = set(future_imports) | {definition_index}
        pending = [definition_index]
        while pending:
            current = pending.pop()
            for name in _loaded_names(body[current]):
                dependency = bound_at.get(name)
                if dependency is not None and dependency not in selected:
                    selected.add(dependency)
                    pending.append(dependency)
        _add_projection(selected)
        _add_projection(selected, prune_imports=True)

    return fingerprints


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

    Also folds in the op's declared ``cuda_sources`` (see ``optima/manifest.py``): each
    contributes TWO entries to the set — the exact sha256 of its raw bytes (catches a
    byte-identical relocated copy) and ``cuda_source_fingerprint`` (the reformat-invariant
    normalization, catches a copy that's just been reflowed/recommented). This is what
    closes the "ship a stolen binary/.cu behind a trivially different .py shim" gap: a
    declared cuda_source used to be invisible to copy_fingerprint's import-closure walk
    entirely (it isn't Python, isn't imported, and previously wasn't even scanned).
    Sibling variants contribute to one union, so adding a fresh variant cannot hide a
    stolen singleton from the ledger's subset-containment comparison.
    """
    root = Path(bundle_root)
    manifest = load_manifest(root)
    out: dict[str, set[str]] = {}
    # Declared dep patches are BUNDLE-level (they modify the engine's dependency tree
    # for every slot the bundle claims), so their fingerprints fold into EVERY slot's
    # file set: a stolen deep-seam patch re-shipped behind a different kernel shim is
    # still "all of their work inside yours" for the containment compare. Two entries
    # each, mirroring cuda_sources: exact raw sha256 + the reformat-invariant
    # normalized-diff fingerprint (context width / hunk offsets / git headers free).
    patch_fps: set[str] = set()
    for dp in resolve_dep_patches(root, manifest):
        try:
            raw = dp.read_bytes()
        except OSError:
            continue
        patch_fps.add(hashlib.sha256(raw).hexdigest())
        patch_fps.add(dep_patch_fingerprint(raw.decode("utf-8", errors="replace")))
    try:
        for op in manifest.ops:
            entry = resolve_source(root, op)
            entry_key = entry.resolve()
            # All sibling variants contribute to one slot-level union. This keeps a
            # stolen singleton's set contained in a stolen+fresh multi-variant bundle,
            # which is the load-bearing relocation/padding copy signal in Ledger.reveal.
            fps = out.setdefault(op.slot, set(patch_fps))
            for f in _closure_files(root, entry):
                try:
                    source = f.read_text(encoding="utf-8")
                    norm = normalized_source(source)
                except (SyntaxError, OSError, UnicodeDecodeError):
                    if f.resolve() == entry_key:
                        raise
                    continue
                if len(norm) >= _SUBSTANTIAL_NORM_LEN:
                    fps.add(hashlib.sha256(norm.encode("utf-8")).hexdigest())
                fps.update(_top_level_definition_fingerprints(source))
            for cs in resolve_cuda_sources(root, op):
                try:
                    raw = cs.read_bytes()
                except OSError:
                    continue
                fps.add(hashlib.sha256(raw).hexdigest())
                fps.add(cuda_source_fingerprint(raw.decode("utf-8", errors="replace")))
    except SyntaxError:
        return {}
    return {slot: sorted(fingerprints) for slot, fingerprints in out.items()}


def bundle_slot_structural_fingerprints(bundle_root: str | Path) -> dict[str, str]:
    """Advisory per-slot structural fingerprint over every variant's closure.

    The multi-variant fold is order-independent and label-independent; singleton output
    remains byte-for-byte compatible with prior ledgers.
    """
    root = Path(bundle_root)
    manifest = load_manifest(root)
    components: dict[str, list[str]] = {}
    try:
        for op in manifest.ops:
            closure = _closure_norm(root, resolve_source(root, op), structural_source)
            components.setdefault(op.slot, []).append(
                hashlib.sha256((op.slot + "\x1e" + closure).encode("utf-8")).hexdigest()
            )
    except SyntaxError:
        return {}
    return {
        slot: _aggregate_variant_fingerprints(
            fingerprints,
            domain="optima.slot.structural-variants.v1",
        )
        for slot, fingerprints in components.items()
    }


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


@dataclass(frozen=True)
class SubmittedDeltaFingerprint:
    """Copy/provenance identity for submitted bytes only.

    The immutable incumbent, assembled engine, and validator-owned adapters never enter
    this product.  ``containment_fingerprints`` are the only relocation/padding signal
    allowed to auto-demote; shared fragments and structural similarity stay advisory.
    """

    product_kind: str
    target_id: str
    target_spec_digest: str
    members: tuple[str, ...]
    exact_payload_digest: str
    selected_delta_digest: str
    normalized_delta_digest: str
    containment_fingerprints: tuple[str, ...]
    advisory_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.product_kind not in {"component", "discovery"}:
            raise ValueError("submitted delta product kind is unsupported")
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError("submitted delta target is malformed")
        members = tuple(self.members)
        containment = tuple(self.containment_fingerprints)
        advisory = tuple(self.advisory_fingerprints)
        if (
            not members
            or members != tuple(sorted(set(members)))
            or any(not isinstance(row, str) or not row for row in members)
            or containment != tuple(sorted(set(containment)))
            or advisory != tuple(sorted(set(advisory)))
        ):
            raise ValueError("submitted delta fingerprint sets are not canonical")
        digest_fields = (
            "exact_payload_digest",
            "selected_delta_digest",
            "normalized_delta_digest",
        )
        if self.product_kind == "component":
            digest_fields += ("target_spec_digest",)
        elif self.target_spec_digest:
            raise ValueError("discovery fingerprint cannot claim a target spec")
        for field in digest_fields:
            value = getattr(self, field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"submitted delta {field} is malformed")
        for value in containment + advisory:
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("submitted delta member fingerprint is malformed")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "containment_fingerprints", containment)
        object.__setattr__(self, "advisory_fingerprints", advisory)

    @property
    def reward_namespace(self) -> tuple[str, ...]:
        return self.members if self.product_kind == "component" else ("discovery",)

    @property
    def digest(self) -> str:
        from optima.stack_identity import canonical_digest

        return canonical_digest("optima.copy.submitted-delta", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "advisory_fingerprints": list(self.advisory_fingerprints),
            "containment_fingerprints": list(self.containment_fingerprints),
            "exact_payload_digest": self.exact_payload_digest,
            "members": list(self.members),
            "normalized_delta_digest": self.normalized_delta_digest,
            "product_kind": self.product_kind,
            "selected_delta_digest": self.selected_delta_digest,
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SubmittedDeltaFingerprint":
        fields = {
            "advisory_fingerprints", "containment_fingerprints",
            "exact_payload_digest", "members", "normalized_delta_digest",
            "product_kind", "selected_delta_digest", "target_id",
            "target_spec_digest",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("submitted delta fingerprint fields differ")
        return cls(
            product_kind=value["product_kind"],  # type: ignore[arg-type]
            target_id=value["target_id"],  # type: ignore[arg-type]
            target_spec_digest=value["target_spec_digest"],  # type: ignore[arg-type]
            members=tuple(value["members"]),  # type: ignore[arg-type]
            exact_payload_digest=value["exact_payload_digest"],  # type: ignore[arg-type]
            selected_delta_digest=value["selected_delta_digest"],  # type: ignore[arg-type]
            normalized_delta_digest=value["normalized_delta_digest"],  # type: ignore[arg-type]
            containment_fingerprints=tuple(value["containment_fingerprints"]),  # type: ignore[arg-type]
            advisory_fingerprints=tuple(value["advisory_fingerprints"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class DeltaCopyDecision:
    authoritative: bool
    reason: str
    shared_fragments: tuple[str, ...] = ()
    shared_advisory: tuple[str, ...] = ()


def _digest_rows(domain: str, rows: list[str] | tuple[str, ...]) -> str:
    values = tuple(sorted(rows))
    if not values:
        return ""
    return hashlib.sha256(
        (domain + "\x00" + "\x1e".join(values)).encode("utf-8")
    ).hexdigest()


def fingerprint_submitted_delta(
    bundle_root: str | Path,
    *,
    catalog=None,
    discovery: bool = False,
) -> SubmittedDeltaFingerprint:
    """Fingerprint one target-owned proposal without canonical stack bytes.

    Normal component intake reuses :func:`inspect_contribution`, the same trusted
    projection used by stack assembly.  Discovery intake fingerprints only the closed
    discovery manifest and declared patches.  Callers must choose the lane explicitly;
    parser failure never silently reclassifies a component as discovery.
    """

    root = Path(bundle_root)
    if discovery:
        from optima.discovery import inspect_discovery

        inspected = inspect_discovery(root)
        normalized = tuple(
            sorted(dep_patch_fingerprint(text) for _path, text in inspected.patch_texts)
        )
        return SubmittedDeltaFingerprint(
            product_kind="discovery",
            target_id="discovery",
            target_spec_digest="",
            members=("discovery",),
            exact_payload_digest=inspected.proposal_digest,
            selected_delta_digest=inspected.proposal_digest,
            normalized_delta_digest=_digest_rows(
                "optima.discovery.normalized-delta.v1", normalized
            ),
            containment_fingerprints=normalized,
            advisory_fingerprints=(),
        )

    from optima.engine_tree import inspect_contribution
    from optima.target_catalog import default_target_catalog

    active_catalog = catalog or default_target_catalog()
    inspected = inspect_contribution(root, catalog=active_catalog)
    normalized_by_slot = bundle_slot_fingerprints(root)
    files_by_slot = bundle_slot_file_fingerprints(root)
    structural_by_slot = bundle_slot_structural_fingerprints(root)
    members = tuple(sorted({op.slot for op in inspected.manifest.ops}))
    if members != tuple(sorted(active_catalog.require(inspected.target_id).members)):
        raise ValueError("submitted delta members differ from the resolved target")
    normalized_rows = [
        f"{member}\x00{normalized_by_slot[member]}"
        for member in members
        if normalized_by_slot.get(member)
    ]
    containment = tuple(sorted({
        fingerprint
        for member in members
        for fingerprint in files_by_slot.get(member, ())
    }))
    advisory = tuple(sorted({
        fingerprint
        for member in members
        for fingerprint in (structural_by_slot.get(member),)
        if fingerprint
    }))
    # Tiny but otherwise valid implementations may have no substantial
    # definition-level containment fragment. Their exact and normalized
    # whole-delta identities remain authoritative.
    if not normalized_rows:
        raise ValueError("submitted component delta produced incomplete fingerprints")
    return SubmittedDeltaFingerprint(
        product_kind="component",
        target_id=inspected.target_id,
        target_spec_digest=inspected.target_spec_digest,
        members=members,
        exact_payload_digest=inspected.selected_payload_digest,
        selected_delta_digest=inspected.selected_delta_digest,
        normalized_delta_digest=_digest_rows(
            "optima.component.normalized-delta.v1", normalized_rows
        ),
        containment_fingerprints=containment,
        advisory_fingerprints=advisory,
    )


def compare_submitted_deltas(
    earlier: SubmittedDeltaFingerprint,
    later: SubmittedDeltaFingerprint,
) -> DeltaCopyDecision:
    """Compare two chain-ordered deltas without fragment-intersection poisoning."""

    if type(earlier) is not SubmittedDeltaFingerprint or type(later) is not SubmittedDeltaFingerprint:
        raise TypeError("copy comparison requires typed submitted-delta fingerprints")
    if earlier.product_kind != later.product_kind:
        return DeltaCopyDecision(False, "different_product_kind")
    if not set(earlier.reward_namespace) & set(later.reward_namespace):
        return DeltaCopyDecision(False, "different_reward_namespace")
    if earlier.exact_payload_digest == later.exact_payload_digest:
        return DeltaCopyDecision(True, "exact_delta_identity")
    if (
        earlier.normalized_delta_digest
        and earlier.normalized_delta_digest == later.normalized_delta_digest
    ):
        return DeltaCopyDecision(True, "normalized_delta_identity")
    left = set(earlier.containment_fingerprints)
    right = set(later.containment_fingerprints)
    if left and right and (left <= right or right <= left):
        return DeltaCopyDecision(True, "symmetric_delta_containment")
    shared = tuple(sorted(left & right))
    advisory = tuple(sorted(
        set(earlier.advisory_fingerprints) & set(later.advisory_fingerprints)
    ))
    return DeltaCopyDecision(False, "advisory_only", shared, advisory)
