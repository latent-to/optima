"""Deterministic content hash of a bundle.

The hash is the bundle's identity. It is what a commitment binds to (so a miner
commits to an exact bundle before revealing it) and what copy-detection compares
(two reveals with the same content hash are the same submission). It must be
stable across machines and insensitive to incidental filesystem noise.

We hash the manifest plus every source file under the bundle, in sorted order,
length-prefixed so concatenation is unambiguous. ``__pycache__`` and editor junk
are excluded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_SKIP_DIRS = {"__pycache__", ".git"}
_SKIP_SUFFIXES = {".pyc", ".pyo"}


def _iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        # Skip symlinks: is_file()/read_bytes() would FOLLOW a symlink and fold an
        # out-of-bundle file's bytes into the identity hash, so a bundle's hash could
        # depend on mutable state elsewhere on disk (and rglob doesn't descend
        # symlinked dirs, so their contents would be invisible here yet importable at
        # runtime — the same split scan_tree now rejects). Identity is over the
        # bundle's own regular files only; a symlinked bundle is refused at load.
        if p.is_symlink() or not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.suffix in _SKIP_SUFFIXES or p.name.startswith("._"):
            continue
        yield p


def content_hash(bundle_dir: str | Path) -> str:
    """Return the SHA-256 content hash of a bundle directory (hex)."""
    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"bundle dir not found: {root}")
    h = hashlib.sha256()
    n = 0
    for p in _iter_files(root):
        rel = p.relative_to(root).as_posix().encode("utf-8")
        data = p.read_bytes()
        # length-prefix path and contents so boundaries are unambiguous
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
        n += 1
    if n == 0:
        raise ValueError(f"bundle dir has no files to hash: {root}")
    return h.hexdigest()
