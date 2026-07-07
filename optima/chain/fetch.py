"""Bundle transport: pack a bundle directory into a tarball; fetch + safely extract one.

The tarball is *transport only* — identity is ``optima.bundle_hash.content_hash``
over the extracted DIRECTORY, so the same bundle hashes the same however it was
shipped. Packaging includes exactly the files the identity hash covers (same walk,
same skip rules); fetching re-hashes after extraction and refuses anything that
does not match the hash the miner committed on chain.

Extraction treats the archive as hostile: only regular files and directories are
accepted (no symlinks/hardlinks/devices), member paths must stay inside the
destination, and archive/extracted/member-count budgets are enforced. A rejected
archive leaves nothing behind.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from optima.bundle_hash import _iter_files, content_hash

logger = logging.getLogger("optima.chain.fetch")

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 4096
FETCH_TIMEOUT_S = 60.0


class FetchError(RuntimeError):
    """A submission artifact could not be fetched/extracted/verified. One bad
    submission must never take the validator loop down — callers catch this,
    record the rejection, and move on."""


def package_bundle(bundle_dir: str | Path, out_path: str | Path | None = None) -> tuple[Path, str]:
    """Miner side: tar.gz the bundle and return ``(archive_path, content_hash)``.

    Contains exactly the files the identity hash covers, under a single top-level
    directory named after the bundle dir. The returned hash is what goes on chain.
    """
    root = Path(bundle_dir).resolve()
    ch = content_hash(root)  # raises if not a dir / empty
    out = Path(out_path) if out_path else Path(f"{root.name}.tar.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for p in _iter_files(root):
            rel = p.relative_to(root).as_posix()
            tar.add(p, arcname=f"{root.name}/{rel}", recursive=False)
    return out, ch


def _download(url: str, dest: Path, max_bytes: int) -> None:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        src = Path(urllib.request.url2pathname(parsed.path))
        if not src.is_file():
            raise FetchError(f"file url does not point at a file: {url}")
        if src.stat().st_size > max_bytes:
            raise FetchError(f"archive exceeds {max_bytes} bytes: {url}")
        shutil.copyfile(src, dest)
        return
    if parsed.scheme not in ("http", "https"):
        raise FetchError(f"unsupported url scheme: {url}")
    try:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as resp, open(dest, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise FetchError(f"archive exceeds {max_bytes} bytes: {url}")
                f.write(chunk)
    except FetchError:
        raise
    except Exception as e:  # noqa: BLE001 — network errors are all "couldn't fetch"
        raise FetchError(f"download failed for {url}: {type(e).__name__}: {e}") from e


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract accepting only regular files/dirs with in-tree relative paths."""
    budget = MAX_EXTRACTED_BYTES
    members = 0
    try:
        with tarfile.open(archive, "r:*") as tar:
            for m in tar:
                members += 1
                if members > MAX_MEMBERS:
                    raise FetchError(f"archive has more than {MAX_MEMBERS} members")
                name = Path(m.name)
                if name.is_absolute() or ".." in name.parts or not m.name:
                    raise FetchError(f"archive member escapes destination: {m.name!r}")
                if m.isdir():
                    (dest / name).mkdir(parents=True, exist_ok=True)
                    continue
                if not m.isreg():
                    raise FetchError(f"archive member is not a regular file: {m.name!r}")
                budget -= m.size
                if budget < 0:
                    raise FetchError(f"extracted size exceeds {MAX_EXTRACTED_BYTES} bytes")
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                if src is None:
                    raise FetchError(f"unreadable archive member: {m.name!r}")
                with src, open(target, "wb") as f:
                    shutil.copyfileobj(src, f)
    except tarfile.TarError as e:
        raise FetchError(f"corrupt archive: {e}") from e


def _bundle_root(extract_dir: Path) -> Path:
    """The bundle root is the single top-level dir if there is exactly one, else the
    extraction dir itself (manifest.toml at archive top level)."""
    entries = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir


def fetch_bundle(url: str, expected_hash: str, dest_root: str | Path) -> Path:
    """Validator side: fetch, safely extract, and hash-verify a committed bundle.

    Returns the bundle directory at ``dest_root/<expected_hash>``. Idempotent: an
    existing directory for this hash is re-verified and reused. Raises FetchError
    on any transport, extraction, or hash failure — leaving no partial state.
    """
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    final = dest_root / expected_hash
    if final.exists():
        actual = content_hash(final)
        if actual == expected_hash:
            return final
        # A corrupted/tampered cache entry: refuse to silently reuse it.
        raise FetchError(f"cached bundle at {final} re-hashes to {actual[:16]}…; "
                         "remove it manually to re-fetch")

    with tempfile.TemporaryDirectory(dir=dest_root, prefix=".fetch.") as tmp:
        tmp = Path(tmp)
        archive = tmp / "bundle.tar.gz"
        _download(url, archive, MAX_ARCHIVE_BYTES)
        extract_dir = tmp / "extract"
        extract_dir.mkdir()
        _safe_extract(archive, extract_dir)
        root = _bundle_root(extract_dir)
        try:
            actual = content_hash(root)
        except (ValueError, NotADirectoryError) as e:
            raise FetchError(f"extracted archive is not a bundle: {e}") from e
        if actual != expected_hash:
            raise FetchError(
                f"content hash mismatch: committed {expected_hash[:16]}…, "
                f"fetched {actual[:16]}… — rejecting submission")
        root.rename(final)
    logger.info("fetched bundle %s… from %s", expected_hash[:16], url)
    return final
