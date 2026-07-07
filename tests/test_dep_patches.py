"""The dep_patches tier — strict diff parsing, exact apply, manifest/scan/fingerprint.

Covers the security posture, not just the happy path: the parser must reject every
construct that isn't a plain text modification/new-file diff, the applier must refuse
fuzzy application (pinned dep ⇒ byte-exact context), the scan must fail closed on
undeclared patch files, and the copy fingerprints must be invariant to diff
re-presentation (context width) but sensitive to what the patch actually does.

Also pins the REAL artifact: the generated flashinfer fe_export patch (deep-seam
bundle) must parse under this exact parser and clear the flashinfer arena policy.
"""

from __future__ import annotations

import difflib
import textwrap
from pathlib import Path

import pytest

from optima.deppatch import DepPatchError, apply_file_patch, parse_patch_text

REPO = Path(__file__).resolve().parent.parent
REAL_PATCH = (REPO / "experiments/minimax_m3/bundle/miner_m3_fused_epilogue_deep/"
              "patches/flashinfer_fe_export.patch")

OLD = "line one\nline two\nline three\nline four\nline five\n"
NEW = "line one\nline two CHANGED\nline three\nline four\nline five\nline six\n"


def _diff(a: str, b: str, path: str = "pkg/sub/file.cu", n: int = 3) -> str:
    return "".join(difflib.unified_diff(
        a.splitlines(keepends=True), b.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=n))


def _newfile_diff(b: str, path: str = "pkg/sub/new.h") -> str:
    return "".join(difflib.unified_diff(
        [], b.splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{path}"))


# -- parser ------------------------------------------------------------------

def test_parse_modification_roundtrip():
    (fp,) = parse_patch_text(_diff(OLD, NEW))
    assert fp.path == "pkg/sub/file.cu" and not fp.is_new_file
    assert apply_file_patch(OLD, fp) == NEW


def test_parse_new_file():
    (fp,) = parse_patch_text(_newfile_diff("a\nb\n"))
    assert fp.is_new_file
    assert apply_file_patch(None, fp) == "a\nb\n"
    with pytest.raises(DepPatchError, match="already exists"):
        apply_file_patch("existing\n", fp)


def test_parse_multi_file_and_git_headers():
    text = ("diff --git a/pkg/x.cu b/pkg/x.cu\nindex 111..222 100644\n"
            + _diff(OLD, NEW, path="pkg/x.cu")
            + "diff --git a/pkg/y.h b/pkg/y.h\nnew file mode 100644\n"
            + _newfile_diff("h\n", path="pkg/y.h"))
    fps = parse_patch_text(text)
    assert [f.path for f in fps] == ["pkg/x.cu", "pkg/y.h"]


@pytest.mark.parametrize("bad,why", [
    ("Binary files a/x and b/x differ\n", "binary"),
    ("GIT binary patch\nliteral 5\n", "binary"),
    ("diff --git a/x b/y\nrename from x\nrename to y\n", "rename"),
    (_diff(OLD, NEW).replace("+++ b/pkg/sub/file.cu", "+++ /dev/null"), "deletion"),
])
def test_parse_rejects_unsupported_constructs(bad, why):
    with pytest.raises(DepPatchError):
        parse_patch_text(bad)


def test_parse_rejects_no_newline_marker():
    text = _diff(OLD, NEW) + "\\ No newline at end of file\n"
    with pytest.raises(DepPatchError, match="newline"):
        parse_patch_text(text)


def test_parse_rejects_path_traversal():
    with pytest.raises(DepPatchError, match="component"):
        parse_patch_text(_diff(OLD, NEW, path="pkg/../../etc/passwd"))
    with pytest.raises(DepPatchError):
        parse_patch_text(_diff(OLD, NEW).replace("a/pkg/sub/file.cu", "a//abs")
                         .replace("b/pkg/sub/file.cu", "b//abs"))


def test_parse_rejects_rename_shaped_paths():
    text = _diff(OLD, NEW).replace("+++ b/pkg/sub/file.cu", "+++ b/pkg/sub/other.cu")
    with pytest.raises(DepPatchError, match="rename"):
        parse_patch_text(text)


def test_parse_rejects_corrupt_hunk_counts():
    text = _diff(OLD, NEW)
    # inflate the old-count so the body can't satisfy it
    text = text.replace("@@ -1,5 +1,6 @@", "@@ -1,9 +1,6 @@")
    with pytest.raises(DepPatchError):
        parse_patch_text(text)


def test_added_lines_containing_scary_strings_are_fine():
    # line-anchored rejection: ADDED content mentioning "copy from " must not reject
    new = OLD + "// copy from upstream\n"
    (fp,) = parse_patch_text(_diff(OLD, new))
    assert apply_file_patch(OLD, fp) == new


# -- exact application -------------------------------------------------------

def test_apply_refuses_context_mismatch():
    (fp,) = parse_patch_text(_diff(OLD, NEW))
    drifted = OLD.replace("line three", "line three DRIFTED")
    with pytest.raises(DepPatchError, match="context mismatch"):
        apply_file_patch(drifted, fp)


def test_apply_refuses_offset_application():
    # Same content shifted by one line — a fuzzy patcher would "helpfully" apply it.
    (fp,) = parse_patch_text(_diff(OLD, NEW))
    shifted = "inserted line zero\n" + OLD
    with pytest.raises(DepPatchError):
        apply_file_patch(shifted, fp)


def test_apply_multi_hunk_wide_file():
    a_lines = [f"l{i}\n" for i in range(60)]
    b_lines = list(a_lines)
    b_lines[5] = "l5 CHANGED\n"
    b_lines[50] = "l50 CHANGED\n"
    a, b = "".join(a_lines), "".join(b_lines)
    (fp,) = parse_patch_text(_diff(a, b, n=2))
    assert len(fp.hunks) == 2
    assert apply_file_patch(a, fp) == b


# -- manifest + scan integration ---------------------------------------------

def _mk_bundle(tmp_path: Path, patch_text: str, *, declare: bool = True) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "patches").mkdir()
    (bundle / "kernels" / "k.py").write_text(
        "def entry(x, out):\n    out.copy_(x)\n")
    (bundle / "patches" / "p.patch").write_text(patch_text)
    dep = ('\n[[dep_patches]]\ntarget = "flashinfer"\npath = "patches/p.patch"\n'
           if declare else "")
    (bundle / "manifest.toml").write_text(textwrap.dedent("""\
        bundle_id = "dep-patch-test"
        abi_version = "optima-op-abi-v0"

        [[ops]]
        slot = "activation.silu_and_mul"
        source = "kernels/k.py"
        entry = "entry"
        dtypes = ["float32"]
    """) + dep)
    return bundle


def test_manifest_loads_declared_patch(tmp_path):
    from optima.manifest import all_declared_dep_patches, load_manifest

    bundle = _mk_bundle(tmp_path, _diff(OLD, NEW, path="flashinfer/data/csrc/fused_moe/x.cu"))
    m = load_manifest(bundle)
    assert [(d.target, d.path) for d in m.dep_patches] == [("flashinfer", "patches/p.patch")]
    assert len(all_declared_dep_patches(bundle, m)) == 1


def test_manifest_rejects_binary_patch(tmp_path):
    from optima.manifest import ManifestError, load_manifest

    bundle = _mk_bundle(tmp_path, "Binary files a/x and b/x differ\n")
    with pytest.raises(ManifestError, match="rejected"):
        load_manifest(bundle)


def test_scan_fails_closed_on_undeclared_patch(tmp_path):
    from optima.manifest import (all_declared_cuda_sources, all_declared_dep_patches,
                                 load_manifest)
    from optima.sandbox import scan_tree

    bundle = _mk_bundle(tmp_path, _diff(OLD, NEW), declare=False)
    m = load_manifest(bundle)
    res = scan_tree(bundle, declared_cuda_sources=all_declared_cuda_sources(bundle, m),
                    declared_dep_patches=all_declared_dep_patches(bundle, m))
    assert not res.ok
    assert any("not declared in manifest dep_patches" in v for v in res.violations)


def test_scan_passes_with_declared_patch(tmp_path):
    from optima.manifest import (all_declared_cuda_sources, all_declared_dep_patches,
                                 load_manifest)
    from optima.sandbox import scan_tree

    bundle = _mk_bundle(tmp_path, _diff(OLD, NEW, path="flashinfer/data/csrc/fused_moe/x.cu"))
    m = load_manifest(bundle)
    res = scan_tree(bundle, declared_cuda_sources=all_declared_cuda_sources(bundle, m),
                    declared_dep_patches=all_declared_dep_patches(bundle, m))
    assert res.ok, res.violations


# -- copy fingerprints ---------------------------------------------------------

def test_fingerprint_invariant_to_context_width_sensitive_to_payload(tmp_path):
    from optima.copy_fingerprint import dep_patch_fingerprint

    u1 = _diff(OLD, NEW, n=1)
    u3 = _diff(OLD, NEW, n=3)
    assert u1 != u3  # different presentations...
    assert dep_patch_fingerprint(u1) == dep_patch_fingerprint(u3)  # ...same identity
    other = _diff(OLD, NEW.replace("CHANGED", "DIFFERENTLY"), n=3)
    assert dep_patch_fingerprint(u3) != dep_patch_fingerprint(other)


def test_bundle_slot_fps_include_patch(tmp_path):
    from optima.copy_fingerprint import bundle_slot_file_fingerprints, dep_patch_fingerprint

    text = _diff(OLD, NEW, path="flashinfer/data/csrc/fused_moe/x.cu")
    with_patch = bundle_slot_file_fingerprints(_mk_bundle(tmp_path, text))
    assert dep_patch_fingerprint(text) in with_patch["activation.silu_and_mul"]


# -- policy gate (the reviewed applier's checks, exercised directly) -----------

def test_policy_rejects_unknown_target_and_out_of_tree_paths():
    import importlib.util as ilu
    from pathlib import Path as P

    spec = ilu.spec_from_file_location(
        "apply_dep_patch_mod", P(__file__).parent.parent / "optima/patchers/apply_dep_patch.py")
    # The patcher runs main() at import; neuter it by extracting just _check_policy via
    # a controlled namespace exec (the file is validator-owned).
    src = (P(__file__).parent.parent / "optima/patchers/apply_dep_patch.py").read_text()
    ns: dict = {}
    exec(compile(src.replace("\nmain()\n", "\n"), str(spec.origin), "exec"), ns)
    check = ns["_check_policy"]

    (ok,) = parse_patch_text(_diff(OLD, NEW, path="flashinfer/data/csrc/fused_moe/x.cu"))
    check("flashinfer", [ok])  # allowed

    with pytest.raises(RuntimeError, match="allowlist"):
        check("leftpad", [ok])
    (bad,) = parse_patch_text(_diff(OLD, NEW, path="flashinfer/jit/core.py"))
    with pytest.raises(RuntimeError, match="outside the overlay subtree"):
        check("flashinfer", [bad])
    (bad2,) = parse_patch_text(_diff(OLD, NEW, path="flashinfer/data/csrc/norm.cu"))
    with pytest.raises(RuntimeError, match="outside the overlay subtree|allowed"):
        check("flashinfer", [bad2])


def test_overlay_materialization_roundtrip(tmp_path):
    """The applier's overlay half against a synthetic site-root: subtree copied, hunks
    applied exactly, new file created, untouched files byte-identical."""
    from pathlib import Path as P

    src = (P(__file__).parent.parent / "optima/patchers/apply_dep_patch.py").read_text()
    ns: dict = {}
    exec(compile(src.replace("\nmain()\n", "\n"), "apply_dep_patch", "exec"), ns)

    site = tmp_path / "site-packages"
    (site / "pkg" / "data" / "csrc" / "moe").mkdir(parents=True)
    (site / "pkg" / "data" / "csrc" / "moe" / "kern.cu").write_text(OLD)
    (site / "pkg" / "data" / "csrc" / "moe" / "other.cuh").write_text("untouched\n")

    from optima.dep_policy import DepPolicy

    policy = DepPolicy(package="pkg", overlay_subtree="pkg/data/csrc",
                       allowed_globs=("pkg/data/csrc/moe/*",))
    parsed = parse_patch_text(
        _diff(OLD, NEW, path="pkg/data/csrc/moe/kern.cu")
        + _newfile_diff("new header\n", path="pkg/data/csrc/moe/extra.h"))
    dest = tmp_path / "overlay"
    manifest = ns["_apply_to_overlay"](policy, [("patches/p.patch", parsed)], site, dest)

    assert (dest / "pkg/data/csrc/moe/kern.cu").read_text() == NEW
    assert (dest / "pkg/data/csrc/moe/extra.h").read_text() == "new header\n"
    assert (dest / "pkg/data/csrc/moe/other.cuh").read_text() == "untouched\n"
    assert set(manifest["files"]) == {"pkg/data/csrc/moe/kern.cu",
                                      "pkg/data/csrc/moe/extra.h"}
    # the shared "install" was never mutated
    assert (site / "pkg/data/csrc/moe/kern.cu").read_text() == OLD


# -- the REAL artifact ----------------------------------------------------------

def test_real_fe_export_patch_parses_and_clears_policy():
    assert REAL_PATCH.is_file(), "deep-bundle fe_export patch missing"
    fps = parse_patch_text(REAL_PATCH.read_text())
    paths = sorted(f.path for f in fps)
    assert paths == [
        "flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
        "flashinfer/data/csrc/fused_moe/cutlass_backend/fe_export.h",
        "flashinfer/data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu",
    ]
    new_files = [f.path for f in fps if f.is_new_file]
    assert new_files == ["flashinfer/data/csrc/fused_moe/cutlass_backend/fe_export.h"]
    from optima.dep_policy import PATCHABLE_DEPS
    from fnmatch import fnmatch

    policy = PATCHABLE_DEPS["flashinfer"]
    for f in fps:
        assert any(fnmatch(f.path, g) for g in policy.allowed_globs), f.path
