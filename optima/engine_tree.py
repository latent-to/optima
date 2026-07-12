"""Deterministic, data-only assembly of a complete Optima engine bundle.

The materializer never imports contribution Python or loads native code.  It resolves
content-addressed proposal/integrated sources, selects only the registered target payload,
rewrites bundle-local module identities, and emits one validator-owned runtime manifest.
Execution, publication, and crown authority intentionally live elsewhere.
"""

from __future__ import annotations

import ast
import hashlib
import json
import keyword
import os
import posixpath
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Protocol

from optima.bundle_hash import content_hash
from optima.deppatch import parse_patch_text
from optima.manifest import ABI_VERSION, Manifest, OpEntry, load_manifest
from optima.rebuild import RebuildPlan, parse_rebuild_plan
from optima.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)


_MATERIALIZER_VERSION = 1
_FILE_MODE = 0o444
_DIR_MODE = 0o755
_INTERNAL_BUNDLE_ID = "optima-materialized-v1"
_DISCOVERY_METADATA = "metadata/optima_discovery.json"
_REBUILD_ORDER = {
    "optima/patchers/apply_dep_patch.py": 0,
    "optima/patchers/build_cuda_ext.py": 1,
}
_SKIP_DIRS = frozenset({".git", "__pycache__"})
_SKIP_SUFFIXES = frozenset({".pyc", ".pyo"})
_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")
_INCLUDE_RE = re.compile(rb"(?m)^[ \t]*#[ \t]*include\b[ \t]*(.*)$")
_UNSUPPORTED_INCLUDE_RE = re.compile(
    rb"(?m)^[ \t]*(?:#[ \t]*include_next\b|%:[ \t]*include(?:_next)?\b)"
)


class EngineTreeError(ValueError):
    """A contribution cannot be assembled into one deterministic engine tree."""


class ContributionSourceResolver(Protocol):
    def resolve_proposal(self, artifact_digest: str) -> str | Path: ...

    def resolve_integrated(self, source_tree_digest: str) -> str | Path: ...


@dataclass(frozen=True)
class EmittedFile:
    path: str
    mode: int
    size: int
    sha256: str

    def identity_data(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True)
class MaterializedEngineTree:
    root: Path
    stack_digest: str
    tree_digest: str
    files: tuple[EmittedFile, ...]
    runtime_manifest: str | None


@dataclass(frozen=True)
class InspectedContribution:
    root: Path
    manifest: Manifest
    target_id: str
    target_spec_digest: str
    selected_payload_digest: str
    selected_delta_digest: str
    rebuild_plan: RebuildPlan | None
    python_files: tuple[str, ...]
    metadata: tuple[tuple[str, bytes], ...]
    cuda_files: tuple[str, ...]
    patch_files: tuple[str, ...]


def _logical_path(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EngineTreeError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise EngineTreeError(f"{field} is not a canonical relative path: {value!r}")
    if any(part in _SKIP_DIRS for part in path.parts):
        raise EngineTreeError(f"{field} uses a reserved ignored directory: {value!r}")
    if path.suffix in _SKIP_SUFFIXES or path.name.startswith("._"):
        raise EngineTreeError(f"{field} uses a reserved ignored filename: {value!r}")
    if "\\" in value or "\x00" in value or path.as_posix() != value:
        raise EngineTreeError(f"{field} is not canonical POSIX syntax: {value!r}")
    return value


def _stable_read(root: Path, relative: str) -> bytes:
    relative = _logical_path(relative, field="source path")
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        before = path.lstat()
    except OSError as exc:
        raise EngineTreeError(f"cannot stat {relative!r}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise EngineTreeError(f"source must be a regular non-symlink file: {relative!r}")
    try:
        first = path.read_bytes()
        middle = path.lstat()
        second = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise EngineTreeError(f"cannot read stable source {relative!r}: {exc}") from exc
    signature = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_size, row.st_mtime_ns)
    if signature(before) != signature(middle) or signature(middle) != signature(after):
        raise EngineTreeError(f"source changed while being read: {relative!r}")
    if first != second:
        raise EngineTreeError(f"source bytes changed while being read: {relative!r}")
    return first


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, str], ...]:
    if not root.is_dir() or root.is_symlink():
        raise EngineTreeError(f"contribution source is not a regular directory: {root}")
    rows: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_symlink():
            raise EngineTreeError(f"source tree contains symlink: {rel.as_posix()}")
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise EngineTreeError(f"source tree contains nonregular file: {rel.as_posix()}")
        if path.suffix in _SKIP_SUFFIXES or path.name.startswith("._"):
            continue
        logical = _logical_path(rel.as_posix(), field="source tree path")
        rows.append((logical, stat.S_IMODE(path.stat().st_mode), sha256_hex(_stable_read(root, logical))))
    if not rows:
        raise EngineTreeError("contribution source tree is empty")
    return tuple(rows)


def _source_directory(value: str | Path, *, field: str) -> Path:
    unresolved = Path(value)
    if unresolved.is_symlink():
        raise EngineTreeError(f"{field} must not be a symlink: {unresolved}")
    try:
        path = unresolved.resolve(strict=True)
    except OSError as exc:
        raise EngineTreeError(f"cannot resolve {field}: {exc}") from exc
    if not path.is_dir():
        raise EngineTreeError(f"{field} is not a regular directory: {path}")
    return path


@contextmanager
def _staged_source_tree(source: Path) -> Iterator[Path]:
    """Freeze one hostile source tree before identity inspection and emission."""

    original = _tree_snapshot(source)
    with tempfile.TemporaryDirectory(prefix="optima-engine-source-") as temp:
        staged = Path(temp) / "source"
        staged.mkdir()
        for relative, mode, digest in original:
            data = _stable_read(source, relative)
            if sha256_hex(data) != digest:
                raise EngineTreeError(f"source changed while staging {relative!r}")
            output = staged.joinpath(*PurePosixPath(relative).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            os.chmod(output, mode)
        if _tree_snapshot(staged) != original or _tree_snapshot(source) != original:
            raise EngineTreeError("contribution source changed while being staged")
        try:
            yield staged
        finally:
            if _tree_snapshot(source) != original:
                raise EngineTreeError("contribution source changed during materialization")


def _integrated_source_tree_digest(path: Path) -> str:
    rows = [
        {"mode": _FILE_MODE, "path": rel, "sha256": digest}
        for rel, _source_mode, digest in _tree_snapshot(path)
    ]
    return canonical_digest("optima.integrated-source-tree", {"files": rows})


def integrated_source_tree_digest(root: str | Path) -> str:
    """Canonical identity of one reviewed integrated contribution source tree."""

    source = _source_directory(root, field="integrated source tree")
    with _staged_source_tree(source) as staged:
        return _integrated_source_tree_digest(staged)


def _canonical_metadata(raw: bytes, *, relative: str) -> tuple[dict[str, object], bytes]:
    from optima.capabilities import capability_domain_from_metadata, canonical_value

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EngineTreeError(f"metadata {relative!r} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EngineTreeError(f"metadata {relative!r} must be a JSON object")
    projected: dict[str, object] = {}
    for key, field in (
        ("architectures", "architecture"),
        ("dtypes", "dtype"),
        ("quant", "quant"),
    ):
        values = data.get(key, ())
        if isinstance(values, str) or not isinstance(values, (list, tuple)):
            raise EngineTreeError(f"metadata {relative!r} field {key!r} must be a list")
        try:
            canonical_values = sorted({str(canonical_value(field, value)) for value in values})
        except (TypeError, ValueError) as exc:
            raise EngineTreeError(f"metadata {relative!r} field {key!r} is invalid: {exc}") from exc
        if canonical_values:
            projected[key] = canonical_values
    graph_safe = data.get("graph_safe", False)
    if not isinstance(graph_safe, bool):
        raise EngineTreeError(f"metadata {relative!r} field 'graph_safe' must be boolean")
    if graph_safe:
        projected["graph_safe"] = True
    for key in ("max_last_dim", "max_num_tokens", "min_num_tokens"):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EngineTreeError(
                f"metadata {relative!r} field {key!r} must be a non-negative integer"
            )
        projected[key] = value
    try:
        domain = capability_domain_from_metadata(data)
    except (TypeError, ValueError) as exc:
        raise EngineTreeError(f"metadata {relative!r} capabilities are invalid: {exc}") from exc
    capabilities: dict[str, object] = {}
    for predicate in domain.predicates:
        if predicate.allowed:
            values = sorted(set(predicate.allowed), key=lambda value: canonical_json_bytes(value))
            capabilities[predicate.field] = (
                {"exact": values[0]} if len(values) == 1 else {"one_of": values}
            )
        else:
            bounds: dict[str, int] = {}
            if predicate.minimum is not None:
                bounds["min"] = predicate.minimum
            if predicate.maximum is not None:
                bounds["max"] = predicate.maximum
            capabilities[predicate.field] = bounds
    if capabilities:
        projected["capabilities"] = capabilities
    # The runtime consumes exactly this canonical semantic projection.
    try:
        canonical = canonical_json_bytes(projected)
    except ValueError as exc:
        raise EngineTreeError(f"metadata {relative!r} is not canonical data: {exc}") from exc
    emitted = json.dumps(
        json.loads(canonical), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    return projected, emitted


def _module_candidates(parts: tuple[str, ...]) -> tuple[str, ...]:
    if not parts:
        return ()
    stem = PurePosixPath(*parts)
    return ((stem / "__init__.py").as_posix(), stem.with_suffix(".py").as_posix())


def _import_parts(current: str, level: int, module: str | None) -> tuple[str, ...]:
    package = list(PurePosixPath(current).parent.parts)
    if level:
        remove = level - 1
        if remove > len(package):
            raise EngineTreeError(f"relative import escapes contribution package in {current!r}")
        if remove:
            package = package[:-remove]
    else:
        package = []
    return tuple(package + ([part for part in (module or "").split(".") if part]))


def _existing_module(root: Path, parts: tuple[str, ...]) -> str | None:
    for candidate in _module_candidates(parts):
        path = root.joinpath(*PurePosixPath(candidate).parts)
        if path.is_file() or path.is_symlink():
            _stable_read(root, candidate)
            return candidate
    return None


def _local_namespace(root: Path, parts: tuple[str, ...]) -> bool:
    if not parts:
        return False
    path = root.joinpath(*parts)
    if path.is_symlink():
        raise EngineTreeError(f"local namespace must not be a symlink: {path}")
    return path.is_dir()


def _has_local_prefix(root: Path, parts: tuple[str, ...]) -> bool:
    """Return whether a proper import prefix is contribution-local.

    An unresolved dotted import must not fall through to an ambient package after
    materialization merely because only its leading package exists in the selected
    contribution closure.
    """

    for end in range(1, len(parts)):
        prefix = parts[:end]
        if _existing_module(root, prefix) is not None or _local_namespace(root, prefix):
            return True
    return False


def _native_stems(cuda_files: tuple[str, ...]) -> dict[str, str]:
    by_stem: dict[str, str] = {}
    for relative in cuda_files:
        path = PurePosixPath(relative)
        if path.suffix != ".cu":
            continue
        previous = by_stem.get(path.stem)
        if previous is not None and previous != relative:
            raise EngineTreeError(
                f"ambiguous native module stem {path.stem!r}: {previous!r}, {relative!r}"
            )
        by_stem[path.stem] = relative
    return by_stem


def _local_include(current: str, raw: bytes) -> str:
    try:
        include = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EngineTreeError(f"CUDA include in {current!r} is not UTF-8") from exc
    if not include or include.startswith("/") or "\\" in include or "\x00" in include:
        raise EngineTreeError(f"CUDA include in {current!r} is not a safe relative path")
    joined = posixpath.normpath(
        posixpath.join(PurePosixPath(current).parent.as_posix(), include)
    )
    return _logical_path(joined, field=f"CUDA include in {current!r}")


def _include_literal(declaration: bytes, *, relative: str) -> tuple[bool, bytes]:
    if declaration.startswith(b"<"):
        closing = b">"
        system = True
    elif declaration.startswith(b'"'):
        closing = b'"'
        system = False
    else:
        raise EngineTreeError(
            f"CUDA source {relative!r} contains a dynamic include directive"
        )
    end = declaration.find(closing, 1)
    if end < 0:
        raise EngineTreeError(
            f"CUDA source {relative!r} contains a dynamic include directive"
        )
    suffix = declaration[end + 1 :].strip()
    if suffix and not (
        suffix.startswith(b"//")
        or (suffix.startswith(b"/*") and suffix.endswith(b"*/"))
    ):
        raise EngineTreeError(
            f"CUDA source {relative!r} contains a dynamic include directive"
        )
    return system, declaration[1:end]


def _validate_cuda_closure(root: Path, cuda_files: tuple[str, ...]) -> None:
    declared = set(cuda_files)
    for relative in cuda_files:
        raw = _stable_read(root, relative)
        logical = raw.replace(b"\\\r\n", b"").replace(b"\\\n", b"")
        if _UNSUPPORTED_INCLUDE_RE.search(logical):
            raise EngineTreeError(
                f"CUDA source {relative!r} contains an unsupported include directive"
            )
        for match in _INCLUDE_RE.finditer(logical):
            declaration = match.group(1).strip()
            system, include_path = _include_literal(declaration, relative=relative)
            if system:
                try:
                    system_header = include_path.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise EngineTreeError(
                        f"CUDA include in {relative!r} is not UTF-8"
                    ) from exc
                parts = system_header.split("/")
                if (
                    not system_header
                    or system_header.startswith("/")
                    or "\\" in system_header
                    or "\x00" in system_header
                    or any(part in {"", ".", ".."} for part in parts)
                ):
                    raise EngineTreeError(
                        f"CUDA source {relative!r} has an unsafe system include"
                    )
                continue  # toolchain/dependency header bound by base-engine identity
            included = _local_include(relative, include_path)
            path = root.joinpath(*PurePosixPath(included).parts)
            if not path.exists() and not path.is_symlink():
                raw_include = include_path.decode("utf-8")
                if any(part in {"", ".", ".."} for part in raw_include.split("/")):
                    raise EngineTreeError(
                        f"CUDA source {relative!r} has an unsafe dependency include"
                    )
                continue  # dependency/toolchain include, bound by the base engine
            _stable_read(root, included)
            if included not in declared:
                raise EngineTreeError(
                    f"CUDA source {relative!r} includes undeclared local input {included!r}"
                )
            if PurePosixPath(included).suffix == ".cu":
                raise EngineTreeError(
                    f"CUDA source {relative!r} includes a compilation unit {included!r}"
                )


def _parse_python(root: Path, relative: str) -> tuple[bytes, ast.Module]:
    raw = _stable_read(root, relative)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EngineTreeError(f"python source {relative!r} is not UTF-8: {exc}") from exc
    try:
        tree = ast.parse(text, filename=relative, type_comments=True)
    except SyntaxError as exc:
        raise EngineTreeError(f"python source {relative!r} does not parse: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {
            "__import__",
            "compile",
            "eval",
            "exec",
        }:
            raise EngineTreeError(f"dynamic import is unsupported in {relative!r}")
        if isinstance(node, ast.alias) and node.name in {"__import__", "import_module"}:
            raise EngineTreeError(f"dynamic import is unsupported in {relative!r}")
        if isinstance(node, ast.Constant) and node.value in {"__import__", "import_module"}:
            raise EngineTreeError(f"dynamic import is unsupported in {relative!r}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "__import__",
                "compile",
                "eval",
                "exec",
                "import_module",
            }:
                raise EngineTreeError(f"dynamic import is unsupported in {relative!r}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "__import__",
                "compile",
                "eval",
                "exec",
                "import_module",
                "load_module",
                "exec_module",
                "spec_from_file_location",
            }:
                raise EngineTreeError(f"dynamic import is unsupported in {relative!r}")
    return raw, tree


def _python_closure(
    root: Path,
    entries: tuple[str, ...],
    cuda_files: tuple[str, ...],
) -> tuple[str, ...]:
    native = set(_native_stems(cuda_files))
    pending = list(entries)
    seen: set[str] = set()
    while pending:
        current = _logical_path(pending.pop(), field="python source")
        if current in seen:
            continue
        if PurePosixPath(current).suffix != ".py":
            raise EngineTreeError(f"entry source must be .py: {current!r}")
        _raw, tree = _parse_python(root, current)
        seen.add(current)
        source_path = PurePosixPath(current)
        for parent in source_path.parents:
            if parent == PurePosixPath("."):
                break
            initializer = (parent / "__init__.py").as_posix()
            if initializer != current:
                candidate = root.joinpath(*parent.parts, "__init__.py")
                if candidate.is_file() or candidate.is_symlink():
                    _stable_read(root, initializer)
                    pending.append(initializer)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = tuple(alias.name.split("."))
                    if len(parts) > 1 and parts[0] in native:
                        raise EngineTreeError(
                            f"partially local import is unsupported in {current!r}"
                        )
                    local = _existing_module(root, parts)
                    namespace = _local_namespace(root, parts)
                    if local is not None and alias.name in native:
                        raise EngineTreeError(
                            f"import {alias.name!r} is both local Python and declared native "
                            f"in {current!r}"
                        )
                    if local is not None:
                        pending.append(local)
                    elif _has_local_prefix(root, parts):
                        raise EngineTreeError(
                            f"partially local import is unsupported in {current!r}"
                        )
                    elif namespace:
                        raise EngineTreeError(
                            f"bare local namespace import is unsupported in {current!r}"
                        )
            elif isinstance(node, ast.ImportFrom):
                base = _import_parts(current, node.level, node.module)
                if node.level == 0 and len(base) > 1 and base[0] in native:
                    raise EngineTreeError(
                        f"partially local import is unsupported in {current!r}"
                    )
                local = _existing_module(root, base)
                namespace = _local_namespace(root, base)
                if local is not None and len(base) == 1 and base[0] in native:
                    raise EngineTreeError(
                        f"import {base[0]!r} is both local Python and declared native "
                        f"in {current!r}"
                    )
                if local is not None:
                    pending.append(local)
                resolved = 0
                for alias in node.names:
                    sub = _existing_module(root, base + tuple(alias.name.split(".")))
                    if sub is not None:
                        resolved += 1
                        pending.append(sub)
                if local is None and resolved and resolved != len(node.names):
                    raise EngineTreeError(
                        f"partially local import is unsupported in {current!r}"
                    )
                if (
                    node.level == 0
                    and local is None
                    and not namespace
                    and _has_local_prefix(root, base)
                ):
                    raise EngineTreeError(
                        f"partially local import is unsupported in {current!r}"
                    )
                if local is None and not resolved and (node.level or namespace):
                    raise EngineTreeError(
                        f"unresolved relative import is unsupported in {current!r}"
                    )
    return tuple(sorted(seen))


def _generated_name(prefix: str, relative: str, *, suffix: str) -> str:
    stem = PurePosixPath(relative).stem
    slug = _SAFE_NAME_RE.sub("_", stem).strip("_") or "module"
    path_hash = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}__{slug}_{path_hash}{suffix}"


def _module_name(prefix: str, relative: str) -> str:
    path = PurePosixPath(relative)
    parts = path.parent.parts if path.name == "__init__.py" else path.with_suffix("").parts
    if any(not part.isidentifier() or keyword.iskeyword(part) for part in parts):
        raise EngineTreeError(f"python module path has a non-identifier component: {relative!r}")
    return ".".join((prefix, *parts))


def _node_offsets(raw: bytes, node: ast.AST) -> tuple[int, int]:
    if not all(hasattr(node, field) for field in ("lineno", "col_offset", "end_lineno", "end_col_offset")):
        raise EngineTreeError("python import node lacks source locations")
    lines = raw.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
    end = sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
    return start, end


def _node_indent(raw: bytes, node: ast.AST) -> bytes:
    lines = raw.splitlines(keepends=True)
    line = lines[node.lineno - 1]
    return line[: node.col_offset]


def _rewrite_python(
    root: Path,
    relative: str,
    module_names: Mapping[str, str],
    native_names: Mapping[str, str],
) -> bytes:
    raw, tree = _parse_python(root, relative)
    replacements: list[tuple[int, int, bytes]] = []
    for node in ast.walk(tree):
        replacement: ast.stmt | list[ast.stmt] | None = None
        if isinstance(node, ast.Import):
            imports: list[ast.stmt] = []
            changed = False
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                if len(parts) > 1 and parts[0] in native_names:
                    raise EngineTreeError(
                        f"partially local import is unsupported in {relative!r}"
                    )
                local = _existing_module(root, parts)
                namespace = _local_namespace(root, parts)
                if local is not None and alias.name in native_names:
                    raise EngineTreeError(
                        f"import {alias.name!r} is both local Python and declared native "
                        f"in {relative!r}"
                    )
                if local is not None:
                    generated = module_names[local]
                    if "." in alias.name and alias.asname is None:
                        imports.extend(
                            (
                                ast.Import(names=[ast.alias(name=generated)]),
                                ast.Import(
                                    names=[
                                        ast.alias(
                                            name=".".join(generated.split(".")[:2]),
                                            asname=alias.name.split(".")[0],
                                        )
                                    ]
                                ),
                            )
                        )
                    else:
                        imports.append(
                            ast.Import(
                                names=[
                                    ast.alias(
                                        name=generated,
                                        asname=alias.asname or alias.name,
                                    )
                                ]
                            )
                        )
                    changed = True
                elif alias.name in native_names:
                    imports.append(
                        ast.Import(
                            names=[
                                ast.alias(
                                    name=native_names[alias.name],
                                    asname=alias.asname or alias.name,
                                )
                            ]
                        )
                    )
                    changed = True
                elif _has_local_prefix(root, parts):
                    raise EngineTreeError(
                        f"partially local import is unsupported in {relative!r}"
                    )
                elif namespace:
                    raise EngineTreeError(
                        f"bare local namespace import is unsupported in {relative!r}"
                    )
                else:
                    imports.append(ast.Import(names=[alias]))
            if changed:
                replacement = imports
        elif isinstance(node, ast.ImportFrom):
            base = _import_parts(relative, node.level, node.module)
            if node.level == 0 and len(base) > 1 and base[0] in native_names:
                raise EngineTreeError(
                    f"partially local import is unsupported in {relative!r}"
                )
            local = _existing_module(root, base)
            namespace = _local_namespace(root, base)
            if local is not None and len(base) == 1 and base[0] in native_names:
                raise EngineTreeError(
                    f"import {base[0]!r} is both local Python and declared native "
                    f"in {relative!r}"
                )
            if local is not None:
                replacement = ast.ImportFrom(
                    module=module_names[local], names=node.names, level=0
                )
            elif node.level == 0 and len(base) == 1 and base[0] in native_names:
                replacement = ast.ImportFrom(
                    module=native_names[base[0]], names=node.names, level=0
                )
            else:
                imports: list[ast.stmt] = []
                resolved = 0
                for alias in node.names:
                    sub = _existing_module(root, base + tuple(alias.name.split(".")))
                    if sub is None:
                        continue
                    resolved += 1
                    imports.append(
                        ast.Import(
                            names=[
                                ast.alias(
                                    name=module_names[sub],
                                    asname=alias.asname or alias.name,
                                )
                            ]
                        )
                    )
                if resolved and resolved != len(node.names):
                    raise EngineTreeError(
                        f"partially local import is unsupported in {relative!r}"
                    )
                if (
                    node.level == 0
                    and local is None
                    and not namespace
                    and _has_local_prefix(root, base)
                ):
                    raise EngineTreeError(
                        f"partially local import is unsupported in {relative!r}"
                    )
                if resolved:
                    replacement = imports
                elif node.level or namespace:
                    raise EngineTreeError(
                        f"unresolved relative import is unsupported in {relative!r}"
                    )
        if replacement is None:
            continue
        nodes = replacement if isinstance(replacement, list) else [replacement]
        indent = _node_indent(raw, node)
        text = (b"\n" + indent).join(
            ast.unparse(item).encode("utf-8") for item in nodes
        )
        start, end = _node_offsets(raw, node)
        replacements.append((start, end, text))
    for start, end, text in sorted(replacements, reverse=True):
        raw = raw[:start] + text + raw[end:]
    return raw


def _rebuild_rows(plan: RebuildPlan | None) -> list[dict[str, object]]:
    return [] if plan is None else [dict(row) for row in plan.to_dict()["steps"]]


def _rebuild_identity_data(plan: RebuildPlan | None) -> dict[str, object]:
    if plan is None:
        return {"schema_version": 1, "steps": []}
    return plan.identity_data()


def _rebuild_features(plan: RebuildPlan | None) -> tuple[str, ...]:
    from optima.target_catalog import (
        FEATURE_REBUILD_APPLY_DEP_PATCH,
        FEATURE_REBUILD_BUILD_CUDA_EXT,
    )

    features: list[str] = []
    for step in () if plan is None else plan.steps:
        if step.patcher_id == "optima.apply-dep-patch.v1":
            features.append(FEATURE_REBUILD_APPLY_DEP_PATCH)
        elif step.patcher_id == "optima.build-cuda-ext.v1":
            features.append(FEATURE_REBUILD_BUILD_CUDA_EXT)
        else:
            raise EngineTreeError(
                f"unregistered parsed rebuild patcher: {step.patcher_id!r}"
            )
    return tuple(features)


def _op_identity(op: OpEntry) -> dict[str, object]:
    from optima.capabilities import canonical_value

    if op.extra:
        raise EngineTreeError(
            f"target-selected op {op.slot!r} contains unregistered fields "
            f"{tuple(sorted(op.extra))!r}"
        )
    return {
        "architectures": sorted(
            {str(canonical_value("architecture", value)) for value in op.architectures}
        ),
        "base_kernel": op.base_kernel,
        "cuda_sources": sorted(set(op.cuda_sources)),
        "dtypes": sorted({str(canonical_value("dtype", value)) for value in op.dtypes}),
        "entry": op.entry,
        "metadata": op.metadata,
        "override_point": op.override_point,
        "prepare": op.prepare,
        "setup": op.setup,
        "slot": op.slot,
        "source": op.source,
        "variant": op.variant,
    }


def _validate_variant_domains(root: Path, manifest: Manifest) -> None:
    from optima.registry import (
        eligibility_domain_is_empty,
        eligibility_domains_overlap,
        eligibility_from_metadata,
    )

    by_slot: dict[str, list[tuple[str, object]]] = {}
    for op in sorted(manifest.ops, key=lambda row: (row.slot, row.variant)):
        metadata = None
        if op.metadata is not None:
            try:
                metadata = json.loads(_stable_read(root, op.metadata).decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise EngineTreeError(
                    f"variant metadata {op.metadata!r} is invalid: {exc}"
                ) from exc
        try:
            eligibility = eligibility_from_metadata(
                metadata, op.dtypes, op.architectures
            )
        except (TypeError, ValueError) as exc:
            raise EngineTreeError(
                f"variant {op.variant!r} for {op.slot!r} has invalid eligibility: {exc}"
            ) from exc
        if eligibility_domain_is_empty(eligibility) is True:
            raise EngineTreeError(
                f"variant {op.variant!r} for {op.slot!r} has an empty capability domain"
            )
        prior = by_slot.setdefault(op.slot, [])
        for prior_variant, prior_eligibility in prior:
            if eligibility_domains_overlap(prior_eligibility, eligibility) is True:
                raise EngineTreeError(
                    f"overlapping capability domains for slot {op.slot!r}: "
                    f"variants {prior_variant!r} and {op.variant!r}"
                )
        prior.append((op.variant, eligibility))


def _inspect_contribution(
    root: Path,
    *,
    catalog: object,
) -> InspectedContribution:
    """Inspect one registered source without importing it or mutating its tree."""

    from optima.registry import eligibility_from_metadata
    from optima.target_catalog import TargetCatalog

    if not isinstance(catalog, TargetCatalog):
        raise TypeError("catalog must be a TargetCatalog")
    before = _tree_snapshot(root)
    manifest = load_manifest(root)
    plan = parse_rebuild_plan(root)
    resolved = catalog.resolve_intake(
        manifest, observed_features=_rebuild_features(plan)
    )
    assert resolved.target_id is not None

    metadata_by_path: dict[str, bytes] = {}
    for op in manifest.ops:
        if op.metadata is None:
            continue
        relative = _logical_path(op.metadata, field="metadata path")
        raw = _stable_read(root, relative)
        projected, emitted = _canonical_metadata(raw, relative=relative)
        try:
            eligibility_from_metadata(projected, op.dtypes, op.architectures)
        except (TypeError, ValueError) as exc:
            raise EngineTreeError(f"metadata {relative!r} is invalid: {exc}") from exc
        existing = metadata_by_path.setdefault(relative, emitted)
        if existing != emitted:
            raise EngineTreeError(f"metadata path has conflicting projections: {relative!r}")
    _validate_variant_domains(root, manifest)
    metadata_identity = [
        {"path": relative, "sha256": sha256_hex(emitted), "size": len(emitted)}
        for relative, emitted in sorted(metadata_by_path.items())
    ]

    cuda_files = tuple(
        sorted({
            _logical_path(relative, field="CUDA source")
            for op in manifest.ops
            for relative in op.cuda_sources
        })
    )
    _validate_cuda_closure(root, cuda_files)
    patch_files = tuple(
        sorted(_logical_path(row.path, field="dependency patch") for row in manifest.dep_patches)
    )
    python_files = _python_closure(
        root,
        tuple(op.source for op in manifest.ops),
        cuda_files,
    )
    for relative in python_files:
        _module_name("optima_validation", relative)

    file_rows: list[dict[str, object]] = []
    for role, paths in (
        ("python", python_files),
        ("cuda", cuda_files),
        ("dep_patch", patch_files),
    ):
        for relative in paths:
            raw = _stable_read(root, relative)
            file_rows.append(
                {
                    "mode": _FILE_MODE,
                    "path": relative,
                    "role": role,
                    "sha256": sha256_hex(raw),
                    "size": len(raw),
                }
            )
    patch_declarations = [
        {"path": row.path, "target": row.target}
        for row in sorted(manifest.dep_patches, key=lambda item: (item.target, item.path))
    ]
    selected = {
        "abi_version": manifest.abi_version,
        "files": sorted(file_rows, key=lambda row: (str(row["role"]), str(row["path"]))),
        "materializer_policy_version": _MATERIALIZER_VERSION,
        "metadata": sorted(metadata_identity, key=lambda row: str(row["path"])),
        "ops": [
            _op_identity(op)
            for op in sorted(manifest.ops, key=lambda item: (item.slot, item.variant))
        ],
        "patches": patch_declarations,
        "rebuild": _rebuild_identity_data(plan),
    }
    selected_payload_digest = canonical_digest(
        "optima.contribution.selected-payload", selected
    )
    target_spec_digest = catalog.target_spec_digest(resolved.target_id)
    selected_delta_digest = canonical_digest(
        "optima.contribution.selected_delta",
        {
            "selected_payload_digest": selected_payload_digest,
            "target_id": resolved.target_id,
            "target_spec_digest": target_spec_digest,
        },
    )
    after = _tree_snapshot(root)
    if before != after:
        raise EngineTreeError("contribution source tree changed during inspection")
    return InspectedContribution(
        root=root,
        manifest=manifest,
        target_id=resolved.target_id,
        target_spec_digest=target_spec_digest,
        selected_payload_digest=selected_payload_digest,
        selected_delta_digest=selected_delta_digest,
        rebuild_plan=plan,
        python_files=python_files,
        metadata=tuple(sorted(metadata_by_path.items())),
        cuda_files=cuda_files,
        patch_files=patch_files,
    )


def inspect_contribution(
    source_root: str | Path,
    *,
    catalog: object,
) -> InspectedContribution:
    """Freeze and inspect one registered source without executing submission code."""

    source = _source_directory(source_root, field="contribution source")
    with _staged_source_tree(source) as staged:
        return replace(_inspect_contribution(staged, catalog=catalog), root=source)


def _put_file(files: dict[str, bytes], path: str, data: bytes) -> None:
    path = _logical_path(path, field="emitted path")
    if path in files:
        raise EngineTreeError(f"emitted path collision: {path!r}")
    files[path] = data


def _resolve_contribution_source(
    resolver: ContributionSourceResolver | Mapping[tuple[str, str], str | Path],
    ref: object,
) -> Path:
    from optima.stack_manifest import IntegratedContributionRef, ProposalContributionRef

    if isinstance(ref, ProposalContributionRef):
        digest = ref.artifact_digest
        source_type = "proposal"
        method = "resolve_proposal"
    elif isinstance(ref, IntegratedContributionRef):
        digest = ref.integrated_source_tree_digest
        source_type = "integrated"
        method = "resolve_integrated"
    else:
        raise EngineTreeError("contribution ref has no registered source identity")
    try:
        value = (
            resolver[(source_type, digest)]
            if isinstance(resolver, Mapping)
            else getattr(resolver, method)(digest)
        )
    except (AttributeError, KeyError, OSError, TypeError) as exc:
        raise EngineTreeError(f"cannot resolve contribution {digest}: {exc}") from exc
    return _source_directory(value, field=f"resolved contribution {digest}")


def _contribution_files(
    inspection: InspectedContribution,
    *,
    delta_digest: str,
    patch_destinations: set[tuple[str, str]],
) -> tuple[dict[str, bytes], list[dict[str, object]], list[dict[str, str]], list[dict[str, object]]]:
    prefix = f"optima_c_{delta_digest}"
    files: dict[str, bytes] = {}
    module_names = {
        relative: _module_name(prefix, relative)
        for relative in inspection.python_files
    }
    native_paths: dict[str, str] = {}
    native_names: dict[str, str] = {}
    for relative in inspection.cuda_files:
        source_path = PurePosixPath(relative)
        suffix = source_path.suffix
        output_name = (
            _generated_name(prefix, relative, suffix=suffix)
            if suffix == ".cu"
            else source_path.name
        )
        output = (PurePosixPath("cuda") / prefix / source_path.parent / output_name).as_posix()
        native_paths[relative] = output
        if suffix == ".cu":
            original = PurePosixPath(relative).stem
            generated = PurePosixPath(output_name).stem
            previous = native_names.setdefault(original, generated)
            if previous != generated:
                raise EngineTreeError(f"ambiguous native import {original!r}")
        _put_file(files, output, _stable_read(inspection.root, relative))

    if "__init__.py" not in inspection.python_files:
        _put_file(files, f"{prefix}/__init__.py", b"")
    for relative in inspection.python_files:
        output = (PurePosixPath(prefix) / relative).as_posix()
        rewritten = _rewrite_python(
            inspection.root,
            relative,
            module_names,
            native_names,
        )
        try:
            compile(rewritten, output, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
        except SyntaxError as exc:
            raise EngineTreeError(
                f"rewritten python source {relative!r} does not parse: {exc}"
            ) from exc
        _put_file(
            files,
            output,
            rewritten,
        )

    required_entry_names: dict[str, set[str]] = {}
    optional_entry_names: dict[str, set[str]] = {}
    for op in inspection.manifest.ops:
        required = required_entry_names.setdefault(op.source, set())
        optional = optional_entry_names.setdefault(op.source, set())
        if op.is_override:
            required.add(op.entry + "_ref")
            optional.add(op.entry)
        else:
            required.add(op.entry)
        if op.prepare is not None:
            required.add(op.prepare)
        if op.setup is not None:
            required.add(op.setup)
    entry_paths: dict[str, str] = {}
    for relative, required in sorted(required_entry_names.items()):
        output = f"entries/{_generated_name(prefix, relative, suffix='.py')}"
        module_name = module_names[relative]
        lines = [
            f"from {module_name} import {name} as {name}\n"
            for name in sorted(required)
        ]
        for name in sorted(optional_entry_names[relative] - required):
            lines.extend(
                (
                    "try:\n",
                    f"    from {module_name} import {name} as {name}\n",
                    "except ImportError:\n",
                    "    pass\n",
                )
            )
        shim = "".join(lines).encode("utf-8")
        try:
            compile(shim, output, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
        except SyntaxError as exc:  # pragma: no cover - manifest names are validated
            raise EngineTreeError(
                f"generated entry shim for {relative!r} does not parse: {exc}"
            ) from exc
        _put_file(files, output, shim)
        entry_paths[relative] = output

    metadata_paths: dict[str, str] = {}
    for relative, data in inspection.metadata:
        output = f"metadata/{_generated_name(prefix, relative, suffix='.json')}"
        metadata_paths[relative] = output
        _put_file(files, output, data)

    patch_paths: dict[str, str] = {}
    patches_by_path = {row.path: row for row in inspection.manifest.dep_patches}
    patch_rows: list[dict[str, str]] = []
    for relative in inspection.patch_files:
        declaration = patches_by_path[relative]
        raw = _stable_read(inspection.root, relative)
        for file_patch in parse_patch_text(raw.decode("utf-8")):
            key = (declaration.target, file_patch.path)
            if key in patch_destinations:
                raise EngineTreeError(
                    f"dependency patch destination collision: {declaration.target}:{file_patch.path}"
                )
            patch_destinations.add(key)
        output = f"patches/{_generated_name(prefix, relative, suffix=PurePosixPath(relative).suffix)}"
        patch_paths[relative] = output
        _put_file(files, output, raw)
        patch_rows.append({"path": output, "target": declaration.target})

    op_rows: list[dict[str, object]] = []
    for op in sorted(inspection.manifest.ops, key=lambda row: (row.slot, row.variant)):
        row = _op_identity(op)
        row["source"] = entry_paths[op.source]
        row["metadata"] = metadata_paths.get(op.metadata) if op.metadata else None
        row["cuda_sources"] = [native_paths[path] for path in row["cuda_sources"]]
        op_rows.append(row)
    return files, op_rows, patch_rows, _rebuild_rows(inspection.rebuild_plan)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _runtime_manifest(
    ops: list[dict[str, object]], patches: list[dict[str, str]]
) -> bytes:
    lines = [
        f"bundle_id = {_toml_string(_INTERNAL_BUNDLE_ID)}",
        f"abi_version = {_toml_string(ABI_VERSION)}",
        "",
    ]
    for patch in sorted(patches, key=lambda row: (row["target"], row["path"])):
        lines.extend(
            [
                "[[dep_patches]]",
                f"target = {_toml_string(patch['target'])}",
                f"path = {_toml_string(patch['path'])}",
                "",
            ]
        )
    for op in ops:
        lines.append("[[ops]]")
        for key in ("slot", "variant", "source", "entry"):
            value = op[key]
            assert isinstance(value, str)
            lines.append(f"{key} = {_toml_string(value)}")
        for key in ("prepare", "setup", "base_kernel", "override_point", "metadata"):
            value = op[key]
            if value is not None:
                assert isinstance(value, str)
                lines.append(f"{key} = {_toml_string(value)}")
        for key in ("dtypes", "architectures", "cuda_sources"):
            values = op[key]
            assert isinstance(values, list) and all(isinstance(value, str) for value in values)
            if values:
                lines.append(f"{key} = {_toml_array(values)}")
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _runtime_rebuild(rows: list[dict[str, object]]) -> bytes | None:
    if not rows:
        return None
    try:
        ordered = sorted(rows, key=lambda row: _REBUILD_ORDER[str(row["path"])])
    except (KeyError, TypeError) as exc:  # defensive join-time validation
        raise EngineTreeError("runtime rebuild contains an unregistered patcher") from exc
    return json.dumps(
        {"steps": ordered}, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _emitted_rows(files: Mapping[str, bytes]) -> tuple[EmittedFile, ...]:
    return tuple(
        EmittedFile(path, _FILE_MODE, len(data), sha256_hex(data))
        for path, data in sorted(files.items())
    )


def _logical_tree_digest(rows: tuple[EmittedFile, ...]) -> str:
    return canonical_digest(
        "optima.materialized-engine-tree",
        {
            "files": [row.identity_data() for row in rows],
            "materializer_policy_version": _MATERIALIZER_VERSION,
        },
    )


def _validate_contribution_rows(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list):
        raise EngineTreeError("materialized contribution inventory is invalid")
    fields = {
        "contribution_ref_digest",
        "namespace",
        "selected_delta_digest",
        "selected_payload_digest",
        "source_digest",
        "source_kind",
        "target_id",
        "target_spec_digest",
    }
    targets: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise EngineTreeError("materialized contribution row schema mismatch")
        target_id = row["target_id"]
        if not isinstance(target_id, str) or not target_id:
            raise EngineTreeError("materialized contribution target is invalid")
        targets.append(target_id)
        for field in (
            "contribution_ref_digest",
            "selected_delta_digest",
            "selected_payload_digest",
            "source_digest",
            "target_spec_digest",
        ):
            try:
                require_sha256_hex(row[field], field=field)
            except ValueError as exc:
                raise EngineTreeError(str(exc)) from exc
        expected_delta = canonical_digest(
            "optima.contribution.selected_delta",
            {
                "selected_payload_digest": row["selected_payload_digest"],
                "target_id": target_id,
                "target_spec_digest": row["target_spec_digest"],
            },
        )
        if row["selected_delta_digest"] != expected_delta:
            raise EngineTreeError("materialized selected-delta identity mismatch")
        if row["namespace"] != f"optima_c_{expected_delta}":
            raise EngineTreeError("materialized contribution namespace mismatch")
        if row["source_kind"] not in {"proposal_artifact", "integrated_source"}:
            raise EngineTreeError("materialized contribution source kind is invalid")
    if targets != sorted(targets) or len(set(targets)) != len(targets):
        raise EngineTreeError("materialized contribution targets are not canonical")
    return rows


def _write_materialized_tree(
    destination: Path,
    *,
    stack_digest: str,
    files: dict[str, bytes],
    runtime_manifest: str | None,
    contributions: list[dict[str, object]],
) -> MaterializedEngineTree:
    if destination.exists() or destination.is_symlink():
        raise EngineTreeError(f"materialized destination already exists: {destination}")
    pre_metadata = _emitted_rows(files)
    metadata = {
        "contributions": contributions,
        "files": [row.identity_data() for row in pre_metadata],
        "materializer_policy_version": _MATERIALIZER_VERSION,
        "runtime_manifest": runtime_manifest,
        "stack_digest": stack_digest,
    }
    _put_file(
        files,
        "metadata/optima_engine_tree.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    rows = _emitted_rows(files)
    tree_digest = _logical_tree_digest(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for relative, data in sorted(files.items()):
            path = temp.joinpath(*PurePosixPath(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            os.chmod(path, _FILE_MODE)
            os.utime(path, (0, 0), follow_symlinks=False)
        directories = sorted(
            (path for path in temp.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for path in directories:
            os.chmod(path, _DIR_MODE)
            os.utime(path, (0, 0), follow_symlinks=False)
        os.chmod(temp, _DIR_MODE)
        os.utime(temp, (0, 0), follow_symlinks=False)
        reopen_materialized_engine_tree(temp, expected_tree_digest=tree_digest)
        os.replace(temp, destination)
    except BaseException:
        for path in sorted(temp.rglob("*"), reverse=True):
            try:
                if path.is_dir():
                    os.chmod(path, 0o755)
            except OSError:
                pass
        try:
            os.chmod(temp, 0o755)
        except OSError:
            pass
        shutil.rmtree(temp, ignore_errors=True)
        raise
    result = MaterializedEngineTree(
        root=destination,
        stack_digest=stack_digest,
        tree_digest=tree_digest,
        files=rows,
        runtime_manifest=runtime_manifest,
    )
    try:
        reopen_materialized_engine_tree(destination, expected_tree_digest=tree_digest)
    except BaseException:
        try:
            os.chmod(destination, 0o755)
        except OSError:
            pass
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return result


def reopen_materialized_engine_tree(
    root: str | Path,
    *,
    expected_tree_digest: str | None = None,
) -> MaterializedEngineTree:
    """Reopen and independently verify a previously materialized logical tree."""

    path = _source_directory(root, field="materialized tree")
    if stat.S_IMODE(path.stat().st_mode) != _DIR_MODE:
        raise EngineTreeError("materialized root directory mode mismatch")
    rows: list[EmittedFile] = []
    directories: set[str] = set()
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path).as_posix()
        if candidate.is_symlink():
            raise EngineTreeError(f"materialized tree contains symlink: {relative}")
        if candidate.is_dir():
            if stat.S_IMODE(candidate.stat().st_mode) != _DIR_MODE:
                raise EngineTreeError(f"materialized directory mode mismatch: {relative}")
            directories.add(relative)
            continue
        if not candidate.is_file():
            raise EngineTreeError(f"materialized tree contains nonregular file: {relative}")
        mode = stat.S_IMODE(candidate.stat().st_mode)
        if mode != _FILE_MODE:
            raise EngineTreeError(f"materialized file mode mismatch: {relative}")
        data = _stable_read(path, relative)
        rows.append(EmittedFile(relative, mode, len(data), sha256_hex(data)))
    row_tuple = tuple(rows)
    expected_directories: set[str] = set()
    for row in row_tuple:
        parent = PurePosixPath(row.path).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if directories != expected_directories:
        raise EngineTreeError("materialized directory inventory mismatch")
    metadata_path = "metadata/optima_engine_tree.json"
    metadata_row = next((row for row in row_tuple if row.path == metadata_path), None)
    if metadata_row is None:
        raise EngineTreeError("materialized tree lacks metadata/optima_engine_tree.json")
    try:
        metadata = json.loads(_stable_read(path, metadata_path).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EngineTreeError(f"materialized tree metadata is invalid: {exc}") from exc
    if not isinstance(metadata, dict) or set(metadata) != {
        "contributions",
        "files",
        "materializer_policy_version",
        "runtime_manifest",
        "stack_digest",
    }:
        raise EngineTreeError("materialized tree metadata schema mismatch")
    if metadata["materializer_policy_version"] != _MATERIALIZER_VERSION:
        raise EngineTreeError("materialized tree policy version mismatch")
    contributions = _validate_contribution_rows(metadata["contributions"])
    try:
        if json.loads(canonical_json_bytes(contributions)) != contributions:
            raise EngineTreeError("materialized contribution inventory is not canonical")
    except ValueError as exc:
        raise EngineTreeError("materialized contribution inventory is invalid") from exc
    stack_digest = metadata["stack_digest"]
    if not isinstance(stack_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", stack_digest):
        raise EngineTreeError("materialized tree stack digest is malformed")
    runtime_manifest = metadata["runtime_manifest"]
    if runtime_manifest is not None and runtime_manifest != "manifest.toml":
        raise EngineTreeError("materialized tree runtime_manifest is invalid")
    if (path / "manifest.toml").is_file() != (runtime_manifest == "manifest.toml"):
        raise EngineTreeError("materialized tree runtime manifest presence mismatch")
    if runtime_manifest is not None:
        try:
            manifest = load_manifest(path)
        except ValueError as exc:
            raise EngineTreeError(f"materialized runtime manifest is invalid: {exc}") from exc
        if manifest.bundle_id != _INTERNAL_BUNDLE_ID or manifest.competition is not None:
            raise EngineTreeError("materialized runtime manifest authority mismatch")
        _validate_variant_domains(path, manifest)
    expected_files = [
        row.identity_data() for row in row_tuple if row.path != metadata_path
    ]
    if metadata["files"] != expected_files:
        raise EngineTreeError("materialized tree file inventory mismatch")
    tree_digest = _logical_tree_digest(row_tuple)
    if expected_tree_digest is not None and tree_digest != expected_tree_digest:
        raise EngineTreeError("materialized tree digest mismatch")
    return MaterializedEngineTree(
        root=path,
        stack_digest=stack_digest,
        tree_digest=tree_digest,
        files=row_tuple,
        runtime_manifest=runtime_manifest,
    )


def materialize_discovery_engine_tree(
    incumbent_root: str | Path,
    discovery: object,
    *,
    policy: object,
    build_profile: object,
    destination: str | Path,
) -> MaterializedEngineTree:
    """Add one source-intent discovery delta to an exact incumbent tree.

    This step copies proposal bytes only. The image-pinned SGLang overlay is
    constructed later by the hermetic OCI prebuild and published by the common
    native-artifact authority.
    """

    from optima.discovery import (
        DiscoveryBuildProfile,
        DiscoveryPolicy,
        InspectedDiscovery,
        discovery_candidate_stack_digest,
        inspect_discovery,
        require_discovery_build_profile,
    )

    if (
        type(discovery) is not InspectedDiscovery
        or type(policy) is not DiscoveryPolicy
        or type(build_profile) is not DiscoveryBuildProfile
    ):
        raise EngineTreeError("discovery materialization requires exact typed inputs")
    incumbent = reopen_materialized_engine_tree(incumbent_root)
    observed = inspect_discovery(discovery.root)
    if observed != discovery:
        raise EngineTreeError("discovery proposal changed after inspection")
    try:
        require_discovery_build_profile(discovery.manifest, build_profile, policy)
    except (TypeError, ValueError) as exc:
        raise EngineTreeError(f"discovery build profile is invalid: {exc}") from None

    destination_path = Path(destination)
    destination_resolved = destination_path.resolve(strict=False)
    sources = (incumbent.root, discovery.root)
    if any(
        destination_resolved == source
        or source in destination_resolved.parents
        or destination_resolved in source.parents
        for source in sources
    ):
        raise EngineTreeError("discovery destination overlaps an immutable input tree")

    try:
        metadata = json.loads(
            _stable_read(
                incumbent.root, "metadata/optima_engine_tree.json"
            ).decode("utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise EngineTreeError(f"incumbent materialized metadata is invalid: {exc}") from None
    if not isinstance(metadata, dict):
        raise EngineTreeError("incumbent materialized metadata is not an object")
    contributions = _validate_contribution_rows(metadata.get("contributions"))

    files: dict[str, bytes] = {}
    for row in incumbent.files:
        if row.path == "metadata/optima_engine_tree.json":
            continue
        raw = _stable_read(incumbent.root, row.path)
        if row.mode != _FILE_MODE or row.size != len(raw) or row.sha256 != sha256_hex(raw):
            raise EngineTreeError(f"incumbent file changed during discovery copy: {row.path!r}")
        _put_file(files, row.path, raw)
    for row in discovery.files:
        raw = _stable_read(discovery.root, row.path)
        if row.size != len(raw) or row.sha256 != sha256_hex(raw):
            raise EngineTreeError(f"discovery proposal file changed: {row.path!r}")
        _put_file(files, f"discovery/{row.path}", raw)

    candidate_stack_digest = discovery_candidate_stack_digest(
        incumbent_stack_digest=incumbent.stack_digest,
        incumbent_tree_digest=incumbent.tree_digest,
        proposal_digest=discovery.proposal_digest,
        policy_digest=policy.digest,
        build_profile_digest=build_profile.digest,
    )
    discovery_metadata = {
        "build_profile": build_profile.to_dict(),
        "build_profile_digest": build_profile.digest,
        "incumbent_stack_digest": incumbent.stack_digest,
        "incumbent_tree_digest": incumbent.tree_digest,
        "policy": policy.to_dict(),
        "policy_digest": policy.digest,
        "proposal_digest": discovery.proposal_digest,
        "proposal_files": [row.to_dict() for row in discovery.files],
        "schema": "optima.discovery-engine-tree.v1",
    }
    _put_file(
        files,
        _DISCOVERY_METADATA,
        json.dumps(
            discovery_metadata, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8") + b"\n",
    )
    result = _write_materialized_tree(
        destination_path,
        stack_digest=candidate_stack_digest,
        files=files,
        runtime_manifest=incumbent.runtime_manifest,
        contributions=contributions,
    )
    if reopen_materialized_engine_tree(incumbent.root) != incumbent:
        raise EngineTreeError("incumbent tree changed during discovery materialization")
    return result


def materialize_engine_tree(
    stack: object,
    *,
    context: object,
    catalog: object,
    resolver: ContributionSourceResolver | Mapping[tuple[str, str], str | Path],
    destination: str | Path,
) -> MaterializedEngineTree:
    """Assemble one validated evaluation or release stack without executing it."""

    from optima.stack_manifest import (
        EngineReleaseManifest,
        EvaluationStackManifest,
        IntegratedContributionRef,
        ProposalContributionRef,
    )
    from optima.target_catalog import TargetCatalog

    if not isinstance(catalog, TargetCatalog):
        raise TypeError("catalog must be a TargetCatalog")
    if not isinstance(stack, (EvaluationStackManifest, EngineReleaseManifest)):
        raise TypeError("stack must be an EvaluationStackManifest or EngineReleaseManifest")
    stack.validate_against(context)
    if stack.catalog_digest != catalog.digest or stack.catalog_snapshot != catalog.snapshot():
        raise EngineTreeError("stack catalog does not match materializer catalog")
    entries = stack.entries
    destination_path = Path(destination)
    destination_resolved = destination_path.resolve(strict=False)
    ordered_targets = catalog.ordered_active_targets(entries)
    files: dict[str, bytes] = {}
    op_rows: list[dict[str, object]] = []
    patch_rows: list[dict[str, str]] = []
    rebuild_by_path: dict[str, dict[str, object]] = {}
    patch_destinations: set[tuple[str, str]] = set()
    contribution_rows: list[dict[str, object]] = []

    for target_id in ordered_targets:
        ref = entries[target_id]
        source = _resolve_contribution_source(resolver, ref)
        if destination_resolved == source or source in destination_resolved.parents:
            raise EngineTreeError(
                f"materialized destination must be outside contribution source {source}"
            )
        with _staged_source_tree(source) as staged:
            if isinstance(ref, ProposalContributionRef):
                source_kind = "proposal_artifact"
                source_digest = ref.artifact_digest
                if content_hash(staged) != source_digest:
                    raise EngineTreeError(
                        f"proposal artifact digest mismatch for {target_id!r}"
                    )
            elif isinstance(ref, IntegratedContributionRef):
                source_kind = "integrated_source"
                source_digest = ref.integrated_source_tree_digest
                if _integrated_source_tree_digest(staged) != source_digest:
                    raise EngineTreeError(
                        f"integrated source digest mismatch for {target_id!r}"
                    )
            else:  # pragma: no cover - stack manifest already enforces this
                raise EngineTreeError(f"unsupported contribution ref for {target_id!r}")

            inspection = _inspect_contribution(staged, catalog=catalog)
            if inspection.target_id != target_id or ref.target_id != target_id:
                raise EngineTreeError(f"resolved target mismatch for {target_id!r}")
            if inspection.target_spec_digest != ref.target_spec_digest:
                raise EngineTreeError(f"target spec digest mismatch for {target_id!r}")
            if inspection.selected_payload_digest != ref.selected_payload_digest:
                raise EngineTreeError(f"selected payload digest mismatch for {target_id!r}")
            if inspection.selected_delta_digest != ref.selected_delta_digest:
                raise EngineTreeError(f"selected delta digest mismatch for {target_id!r}")

            contributed, contribution_ops, contribution_patches, contribution_rebuild = (
                _contribution_files(
                    inspection,
                    delta_digest=ref.selected_delta_digest,
                    patch_destinations=patch_destinations,
                )
            )
            for relative, data in contributed.items():
                _put_file(files, relative, data)
            op_rows.extend(contribution_ops)
            patch_rows.extend(contribution_patches)
            for row in contribution_rebuild:
                path_value = row.get("path")
                if not isinstance(path_value, str):
                    raise EngineTreeError("canonical rebuild step lacks path")
                previous = rebuild_by_path.setdefault(path_value, row)
                if previous != row:
                    raise EngineTreeError(f"conflicting rebuild step {path_value!r}")
            contribution_rows.append(
                {
                    "contribution_ref_digest": ref.digest,
                    "namespace": f"optima_c_{ref.selected_delta_digest}",
                    "selected_delta_digest": ref.selected_delta_digest,
                    "selected_payload_digest": ref.selected_payload_digest,
                    "source_digest": source_digest,
                    "source_kind": source_kind,
                    "target_id": target_id,
                    "target_spec_digest": ref.target_spec_digest,
                }
            )

    runtime_manifest: str | None = None
    if entries:
        runtime_manifest = "manifest.toml"
        _put_file(files, runtime_manifest, _runtime_manifest(op_rows, patch_rows))
        rebuild = _runtime_rebuild(list(rebuild_by_path.values()))
        if rebuild is not None:
            _put_file(files, "rebuild.json", rebuild)
    return _write_materialized_tree(
        destination_path,
        stack_digest=stack.digest,
        files=files,
        runtime_manifest=runtime_manifest,
        contributions=sorted(contribution_rows, key=lambda row: str(row["target_id"])),
    )
