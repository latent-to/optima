"""The sanctioned 'CUDA source' tier: scan-hardening + copy-fingerprint coverage.

Closes the gap where a bundle's ``.cu``/``.so``/``.o`` was INVISIBLE to
``scan_tree`` (suffix-gated to ``.py``) yet still entered ``bundle_hash.content_hash``
(identity/commit-reveal) and was EXCLUDED from ``copy_fingerprint``'s import-closure
walk — a miner could ship a byte-identical stolen binary/CUDA source behind a
trivially different ``.py`` shim and evade both scanning and copy detection.

Covers:
  * ``optima.manifest``: the new ``cuda_sources`` field and its validation
    (existence, containment, symlink refusal, suffix).
  * ``optima.sandbox.scan_tree``: fail-CLOSED once a manifest's declared
    cuda_sources are threaded in (an undeclared ``.cu`` or a ``.so`` anywhere is
    rejected); the old suffix-gated behavior is preserved when no allowlist is
    passed, EXCEPT binary/artifact suffixes are now rejected unconditionally.
  * ``optima.copy_fingerprint``: declared cuda_sources fold into the per-slot,
    path-independent file fingerprint set (the relocation-proof containment
    compare), by exact bytes AND a reformat-invariant normalization.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from optima.manifest import (
    ManifestError,
    all_declared_cuda_sources,
    load_manifest,
    resolve_cuda_sources,
)
from optima.copy_fingerprint import (
    bundle_slot_file_fingerprints,
    cuda_source_fingerprint,
    normalized_cuda_source,
)
from optima.sandbox import scan_tree

CUDA_BODY = """\
// a silu-and-mul epilogue
extern "C" __global__ void silu_and_mul(const float* x, float* out, int n) {
    /* block */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = x[i]; }
}
"""

# Same logic, reflowed: different comments/whitespace -> same normalized fingerprint,
# different exact bytes.
CUDA_BODY_REFORMATTED = """\
extern "C" __global__ void silu_and_mul(const float* x, float* out, int n)
{
    // totally different comment
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = x[i]; /* trailing note */
    }
}
"""

PY_ENTRY = "import torch\n\ndef silu_and_mul(x, out):\n    out.copy_(x)\n"


def _write_manifest(root: Path, ops_toml: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.toml").write_text(
        'bundle_id = "t"\nabi_version = "optima-op-abi-v0"\n\n' + ops_toml
    )


def _basic_bundle(root: Path, *, cuda_sources: list[str] | None = None) -> Path:
    (root / "kernels").mkdir(parents=True, exist_ok=True)
    (root / "kernels" / "k.py").write_text(PY_ENTRY)
    cs_line = ""
    if cuda_sources is not None:
        joined = ", ".join(f'"{c}"' for c in cuda_sources)
        cs_line = f"cuda_sources = [{joined}]\n"
    _write_manifest(
        root,
        "[[ops]]\n"
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/k.py"\n'
        'entry = "silu_and_mul"\n' + cs_line,
    )
    return root


# ---- manifest.py: cuda_sources field validation -----------------------------


def test_manifest_accepts_declared_cuda_source(tmp_path):
    root = _basic_bundle(tmp_path, cuda_sources=["kernels/k.cu"])
    (root / "kernels" / "k.cu").write_text(CUDA_BODY)
    m = load_manifest(root)
    op = m.op_for("activation.silu_and_mul")
    assert op.cuda_sources == ("kernels/k.cu",)
    resolved = resolve_cuda_sources(root, op)
    assert resolved == (root.resolve() / "kernels" / "k.cu",)


def test_manifest_accepts_cuh_suffix(tmp_path):
    root = _basic_bundle(tmp_path, cuda_sources=["kernels/k.cuh"])
    (root / "kernels" / "k.cuh").write_text(CUDA_BODY)
    m = load_manifest(root)
    assert m.op_for("activation.silu_and_mul").cuda_sources == ("kernels/k.cuh",)


def test_manifest_rejects_missing_cuda_source(tmp_path):
    root = _basic_bundle(tmp_path, cuda_sources=["kernels/missing.cu"])
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(root)


def test_manifest_rejects_cuda_source_outside_bundle(tmp_path):
    outside = tmp_path.parent / "outside_cuda_src.cu"
    outside.write_text(CUDA_BODY)
    try:
        root = _basic_bundle(tmp_path, cuda_sources=["../outside_cuda_src.cu"])
        with pytest.raises(ManifestError, match="escapes bundle root"):
            load_manifest(root)
    finally:
        outside.unlink(missing_ok=True)


def test_manifest_rejects_wrong_suffix_cuda_source(tmp_path):
    root = _basic_bundle(tmp_path, cuda_sources=["kernels/k.py"])  # a .py, not .cu/.cuh
    with pytest.raises(ManifestError, match=r"\.cu or \.cuh"):
        load_manifest(root)


def test_manifest_rejects_symlinked_cuda_source(tmp_path):
    real = tmp_path / "real.cu"
    real.write_text(CUDA_BODY)
    root = _basic_bundle(tmp_path, cuda_sources=["kernels/link.cu"])
    (root / "kernels" / "link.cu").symlink_to(real)
    with pytest.raises(ManifestError, match="symlink"):
        load_manifest(root)


def test_manifest_cuda_sources_defaults_to_empty(tmp_path):
    root = _basic_bundle(tmp_path)
    m = load_manifest(root)
    assert m.op_for("activation.silu_and_mul").cuda_sources == ()
    assert all_declared_cuda_sources(root, m) == frozenset()


# ---- sandbox.py: scan_tree fail-closed hardening ----------------------------


def test_scan_tree_undeclared_cuda_source_is_rejected(tmp_path):
    root = _basic_bundle(tmp_path)  # no cuda_sources declared
    (root / "kernels" / "k.cu").write_text(CUDA_BODY)  # present on disk, undeclared
    m = load_manifest(root)
    declared = all_declared_cuda_sources(root, m)
    res = scan_tree(root, declared_cuda_sources=declared)
    assert not res.ok
    assert any("k.cu" in v and "not declared" in v for v in res.violations)


def test_scan_tree_declared_cuda_source_passes(tmp_path):
    root = _basic_bundle(tmp_path, cuda_sources=["kernels/k.cu"])
    (root / "kernels" / "k.cu").write_text(CUDA_BODY)
    m = load_manifest(root)
    declared = all_declared_cuda_sources(root, m)
    res = scan_tree(root, declared_cuda_sources=declared)
    assert res.ok, res.violations


@pytest.mark.parametrize("suffix", [".so", ".o", ".a", ".dylib", ".bin"])
def test_scan_tree_rejects_binary_suffix_even_without_manifest(tmp_path, suffix):
    # No manifest/allowlist at all (the old-call-site default) -> still fail closed on
    # binary/artifact suffixes, because no declaration can ever sanction them.
    root = _basic_bundle(tmp_path)
    (root / "kernels" / f"evil{suffix}").write_bytes(b"\x00\x01\x02binary-stuff")
    res = scan_tree(root)
    assert not res.ok
    assert any(f"evil{suffix}" in v for v in res.violations)


def test_scan_tree_rejects_binary_suffix_even_if_declared_as_cuda_source(tmp_path):
    # A .so cannot be declared as a cuda_source at the manifest layer (wrong suffix), but
    # scan_tree must ALSO reject it unconditionally even if a caller passed a bogus
    # allowlist containing it directly (belt-and-suspenders past the manifest layer).
    root = _basic_bundle(tmp_path)
    so_path = root / "kernels" / "evil.so"
    so_path.write_bytes(b"\x7fELF fake shared object")
    res = scan_tree(root, declared_cuda_sources=frozenset({so_path.resolve()}))
    assert not res.ok
    assert any("evil.so" in v for v in res.violations)


def test_scan_tree_rejects_symlinked_cuda_source(tmp_path):
    outside = tmp_path / "outside.cu"
    outside.write_text(CUDA_BODY)
    root = _basic_bundle(tmp_path)
    link = root / "kernels" / "link.cu"
    link.symlink_to(outside)
    # Even if (hypothetically) declared, scan_tree's own symlink refusal fires first.
    res = scan_tree(root, declared_cuda_sources=frozenset({link.resolve()}))
    assert not res.ok
    assert any("symlink" in v for v in res.violations)


def test_scan_tree_no_manifest_context_preserves_old_cuda_skip_behavior(tmp_path):
    # Backward compat: declared_cuda_sources=None (the default; every pre-existing call
    # site) silently skips a .cu the way the old scan_tree silently skipped non-.py files.
    root = _basic_bundle(tmp_path)
    (root / "kernels" / "k.cu").write_text(CUDA_BODY)
    res = scan_tree(root)
    assert res.ok


def test_scan_tree_benign_metadata_allowlisted_in_strict_mode(tmp_path):
    root = _basic_bundle(tmp_path, cuda_sources=["kernels/k.cu"])
    (root / "kernels" / "k.cu").write_text(CUDA_BODY)
    (root / "README.md").write_text("# hi\n")
    (root / "LICENSE").write_text("MIT\n")
    (root / ".gitignore").write_text("*.pyc\n")
    # rebuild.json is load-bearing (not free-form): strictly validated by
    # optima/rebuild.py, so the scan allowlists it by name. A bundle using the
    # CUDA-source tier ALWAYS carries one — rejecting it would break the tier.
    (root / "rebuild.json").write_text('{"steps": []}')
    (root / "metadata").mkdir()
    (root / "metadata" / "silu_and_mul.json").write_text("{}")
    m = load_manifest(root)
    res = scan_tree(root, declared_cuda_sources=all_declared_cuda_sources(root, m))
    assert res.ok, res.violations


def test_scan_tree_strict_mode_rejects_unrecognized_file(tmp_path):
    root = _basic_bundle(tmp_path, cuda_sources=["kernels/k.cu"])
    (root / "kernels" / "k.cu").write_text(CUDA_BODY)
    (root / "kernels" / "notes.txt").write_text("scratch notes\n")  # not benign-metadata
    m = load_manifest(root)
    res = scan_tree(root, declared_cuda_sources=all_declared_cuda_sources(root, m))
    assert not res.ok
    assert any("notes.txt" in v for v in res.violations)


def test_existing_scan_tree_symlink_and_py_behavior_unchanged(tmp_path):
    # Sanity: the pre-existing symlink-refusal and .py scanning semantics survive the
    # signature change untouched (no declared_cuda_sources passed at all).
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.py").write_text("import os\nos.system('id')\n")
    bundle = tmp_path / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "kernels" / "k.py").write_text("import torch\n")
    (bundle / "kernels" / "vendored").symlink_to(outside, target_is_directory=True)
    res = scan_tree(bundle)
    assert not res.ok
    assert any("symlink" in v for v in res.violations)


# ---- copy_fingerprint.py: cuda_sources fold into the per-slot file set ------


def test_cuda_source_normalization_strips_comments_and_whitespace():
    a = normalized_cuda_source(CUDA_BODY)
    b = normalized_cuda_source(CUDA_BODY_REFORMATTED)
    assert a == b
    assert cuda_source_fingerprint(CUDA_BODY) == cuda_source_fingerprint(CUDA_BODY_REFORMATTED)


def test_declared_cuda_source_appears_in_slot_file_fingerprints(tmp_path):
    root = _basic_bundle(tmp_path, cuda_sources=["kernels/k.cu"])
    (root / "kernels" / "k.cu").write_text(CUDA_BODY)
    fps = bundle_slot_file_fingerprints(root)["activation.silu_and_mul"]
    # Both the exact-byte hash and the reformat-invariant normalized hash are present.
    import hashlib

    exact = hashlib.sha256(CUDA_BODY.encode("utf-8")).hexdigest()
    norm = cuda_source_fingerprint(CUDA_BODY)
    assert exact in fps
    assert norm in fps


def test_relocated_cuda_source_still_matches_by_containment(tmp_path):
    # Same .cu bytes declared under DIFFERENT bundle-relative paths in two bundles ->
    # the path-independent per-file fingerprint set still overlaps (the relocation-proof
    # containment compare the task asks for), even though the declared path differs.
    a = _basic_bundle(tmp_path / "a", cuda_sources=["kernels/orig.cu"])
    (a / "kernels" / "orig.cu").write_text(CUDA_BODY)

    b = _basic_bundle(tmp_path / "b", cuda_sources=["kernels/moved/here.cu"])
    (b / "kernels" / "moved").mkdir()
    (b / "kernels" / "moved" / "here.cu").write_text(CUDA_BODY_REFORMATTED)

    fps_a = set(bundle_slot_file_fingerprints(a)["activation.silu_and_mul"])
    fps_b = set(bundle_slot_file_fingerprints(b)["activation.silu_and_mul"])
    # The exact-byte hashes differ (reformatted), but the normalized fingerprint is shared.
    assert fps_a & fps_b
    assert cuda_source_fingerprint(CUDA_BODY) in fps_a
    assert cuda_source_fingerprint(CUDA_BODY) in fps_b


def test_bundle_without_cuda_sources_unaffected(tmp_path):
    root = _basic_bundle(tmp_path)
    fps = bundle_slot_file_fingerprints(root)["activation.silu_and_mul"]
    # Only the .py entry's own normalized-source hash (below the substantial-length floor
    # for this tiny stub is possible; just assert it doesn't crash and returns a list).
    assert isinstance(fps, list)
