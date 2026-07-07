"""Strict unified-diff parsing + exact application — the ``dep_patches`` engine.

The generic ingestion tier for dependency patches (the deep-seam vehicle): a bundle
declares TEXT unified diffs against a PINNED dependency tree (flashinfer / sglang),
and ONE validator-reviewed patcher applies them to an overlay COPY of that tree
(never the shared install). This module is the engine both sides share:

  * ``parse_patch_text``  — structural validation at manifest-load time (intake
    fails early and loudly on anything but a plain text unified diff);
  * ``apply_file_patch``  — EXACT application at patcher time.

Strictness is a security posture, not a convenience trade-off:

  * The dependency is PINNED (arena digest), so every hunk must apply at its stated
    position with byte-exact context — NO fuzz, NO offset search. A context mismatch
    means the bundle was built against something other than the pinned tree; refusing
    is correct.
  * Only modifications and new-file creations. Deletions, renames, copies, binary
    hunks, and ``\\ No newline at end of file`` markers are rejected outright.
  * Paths are relative, component-checked (no ``..``, no absolute, no backslash);
    WHERE a patch may land is the arena's allowlist decision (see the reviewed
    patcher), not this module's — but a path that can't even be safely interpreted
    never gets that far.

Pure stdlib, no torch — importable from manifest validation on any intake box.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class DepPatchError(ValueError):
    """Raised when a dep patch is malformed, unsupported, or fails to apply exactly."""


@dataclass(frozen=True)
class Hunk:
    old_start: int  # 1-based line number in the original file (0 for new-file hunks)
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]  # each ' ' (context), '-' (remove), '+' (add); no newline


@dataclass(frozen=True)
class FilePatch:
    path: str  # the b-side path, relative to the dependency root
    is_new_file: bool
    hunks: tuple[Hunk, ...]


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# Header lines a git-generated diff carries between file sections; harmless, skipped.
_SKIPPABLE = ("diff ", "index ", "new file mode", "old mode ", "new mode ")
# A line STARTING with one of these (outside a hunk body — bodies only ever see
# ' '/'-'/'+' prefixes) means the patch is not a plain text modification set.
# Line-anchored on purpose: an ADDED source line may legitimately CONTAIN these
# strings ("+copy from ...") and must not be rejected for it.
_REJECT_LINE_STARTS = (
    ("GIT binary patch", "binary patch"),
    ("Binary files ", "binary file diff"),
    ("rename from ", "rename"),
    ("rename to ", "rename"),
    ("copy from ", "copy"),
    ("copy to ", "copy"),
    ("\\ No newline at end of file", "missing trailing newline marker"),
)


def _check_rel_path(path: str) -> str:
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        raise DepPatchError(f"illegal patch path: {path!r}")
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise DepPatchError(f"illegal patch path component: {path!r}")
    return path


def _strip_prefix(raw: str, *, side: str) -> str:
    """``a/<path>`` / ``b/<path>`` -> ``<path>``; ``/dev/null`` passes through."""
    if raw == "/dev/null":
        return raw
    prefix = f"{side}/"
    if not raw.startswith(prefix):
        raise DepPatchError(f"{side}-side path must start with '{prefix}': {raw!r}")
    return _check_rel_path(raw[len(prefix):])


def parse_patch_text(text: str) -> tuple[FilePatch, ...]:
    """Parse + structurally validate a unified diff. Raises DepPatchError on anything
    that is not a plain text modification/new-file diff with self-consistent hunks."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # trailing newline of the patch file itself

    files: list[FilePatch] = []
    seen_paths: set[str] = set()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line == "" or line.startswith(_SKIPPABLE):
            i += 1
            continue
        for marker, why in _REJECT_LINE_STARTS:
            if line.startswith(marker):
                raise DepPatchError(f"unsupported patch construct ({why}): {line!r}")
        if not line.startswith("--- "):
            raise DepPatchError(f"unexpected line outside any file section: {line!r}")
        old_raw = line[4:].split("\t")[0].strip()
        i += 1
        if i >= n or not lines[i].startswith("+++ "):
            raise DepPatchError(f"missing +++ line after {old_raw!r}")
        new_raw = lines[i][4:].split("\t")[0].strip()
        i += 1

        if new_raw == "/dev/null":
            raise DepPatchError(f"file deletion not supported: {old_raw!r}")
        old_path = _strip_prefix(old_raw, side="a") if old_raw != "/dev/null" else old_raw
        new_path = _strip_prefix(new_raw, side="b")
        is_new = old_path == "/dev/null"
        if not is_new and old_path != new_path:
            raise DepPatchError(f"rename not supported: {old_path!r} -> {new_path!r}")
        if new_path in seen_paths:
            raise DepPatchError(f"duplicate file in patch: {new_path!r}")
        seen_paths.add(new_path)

        hunks: list[Hunk] = []
        while i < n:
            m = _HUNK_RE.match(lines[i])
            if m is None:
                break
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            i += 1
            body: list[str] = []
            got_old = got_new = 0
            while i < n and (got_old < old_count or got_new < new_count):
                bl = lines[i]
                tag = bl[:1]
                if tag == " " or bl == "":
                    got_old += 1
                    got_new += 1
                    body.append(" " + bl[1:] if bl else " ")
                elif tag == "-":
                    got_old += 1
                    body.append(bl)
                elif tag == "+":
                    got_new += 1
                    body.append(bl)
                else:
                    raise DepPatchError(f"illegal hunk body line: {bl!r}")
                i += 1
            if got_old != old_count or got_new != new_count:
                raise DepPatchError(
                    f"hunk count mismatch in {new_path!r} @@ -{old_start},{old_count} "
                    f"+{new_start},{new_count} @@ (saw {got_old}/{got_new})"
                )
            if is_new and (old_start != 0 or old_count != 0):
                raise DepPatchError(f"new-file hunk must be @@ -0,0 ... in {new_path!r}")
            hunks.append(Hunk(old_start, old_count, new_start, new_count, tuple(body)))
        if not hunks:
            raise DepPatchError(f"file section without hunks: {new_path!r}")
        if is_new and len(hunks) != 1:
            raise DepPatchError(f"new file must have exactly one hunk: {new_path!r}")
        files.append(FilePatch(path=new_path, is_new_file=is_new, hunks=tuple(hunks)))

    if not files:
        raise DepPatchError("patch contains no file sections")
    return tuple(files)


def apply_file_patch(original: str | None, fp: FilePatch) -> str:
    """Apply one FilePatch EXACTLY (no fuzz, no offset search).

    ``original`` is the pinned file's text, or None for a new-file patch. Every
    context/removal line must match byte-for-byte at the hunk's stated position.
    """
    if fp.is_new_file:
        if original is not None:
            raise DepPatchError(f"new-file patch but {fp.path!r} already exists")
        added = [ln[1:] for ln in fp.hunks[0].lines if ln.startswith("+")]
        return "".join(a + "\n" for a in added)

    if original is None:
        raise DepPatchError(f"patch target missing: {fp.path!r}")
    if original and not original.endswith("\n"):
        raise DepPatchError(f"{fp.path!r}: pinned file lacks trailing newline (unsupported)")
    src = original.split("\n")
    src.pop()  # drop the empty tail from the trailing newline

    out: list[str] = []
    pos = 0  # 0-based cursor into src
    for h in fp.hunks:
        start = h.old_start - 1
        if start < pos:
            raise DepPatchError(f"{fp.path!r}: overlapping/unordered hunks")
        if start + h.old_count > len(src):
            raise DepPatchError(f"{fp.path!r}: hunk exceeds file length")
        out.extend(src[pos:start])
        expect = [ln[1:] for ln in h.lines if ln[0] in (" ", "-")]
        actual = src[start:start + h.old_count]
        if expect != actual:
            for k, (e, a) in enumerate(zip(expect, actual)):
                if e != a:
                    raise DepPatchError(
                        f"{fp.path!r}: context mismatch at line {start + k + 1}: "
                        f"expected {e!r}, found {a!r} — bundle was not built against "
                        "the pinned dependency tree"
                    )
            raise DepPatchError(f"{fp.path!r}: context length mismatch at line {start + 1}")
        out.extend(ln[1:] for ln in h.lines if ln[0] in (" ", "+"))
        pos = start + h.old_count
    out.extend(src[pos:])
    return "".join(o + "\n" for o in out)
