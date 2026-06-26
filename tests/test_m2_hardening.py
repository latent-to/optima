"""M2: the recursive vendored-tree scan, the blessed-base lockfile, and the optima_kernels
no-sglang boundary. All CPU, no torch/sglang needed.
"""

from __future__ import annotations

import re
from pathlib import Path

from optima.blessed_base import BLESSED_BASE, PinnedDep, check_blessed_base
from optima.sandbox import scan_path, scan_tree


# ---- recursive scan closes the vendored-tree hole ---------------------------

def test_scan_tree_catches_vendored_bad_file(tmp_path):
    """A clean entry + a vendored module that trips the denylist -> scan_path(entry) is clean
    but scan_tree(bundle) catches the vendored file (the hole the single-file scan left)."""
    (tmp_path / "kernels").mkdir()
    entry = tmp_path / "kernels" / "k.py"
    entry.write_text("import torch\n\ndef silu_and_mul(x, out):\n    out.copy_(x)\n")
    vendored = tmp_path / "kernels" / "_vendor"
    vendored.mkdir()
    (vendored / "evil.py").write_text("import socket\n\ndef f():\n    socket.socket()\n")

    assert scan_path(entry).ok  # entry alone is clean
    res = scan_tree(tmp_path)
    assert not res.ok
    assert any("socket" in v for v in res.violations)
    assert any("_vendor/evil.py" in v for v in res.violations)


def test_scan_tree_skips_pycache(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "k.py").write_text("def f():\n    pass\n")
    pc = tmp_path / "kernels" / "__pycache__"
    pc.mkdir()
    (pc / "k.cpython-310.py").write_text("import socket\n")  # must be ignored
    assert scan_tree(tmp_path).ok


def test_override_bundle_scans_clean_recursively():
    assert scan_tree("examples/miner_m3_swigluoai_override").ok


# ---- blessed-base lockfile --------------------------------------------------

def test_blessed_base_record_only_is_ok():
    # The shipped base is record-only (versions None) -> every row ok, just reports installed.
    rows = check_blessed_base()
    assert {d.dist for d in BLESSED_BASE} == {name for name, _, _ in rows}
    assert all(ok for _, ok, _ in rows)
    assert all("record-only" in detail for _, _, detail in rows)


def test_blessed_base_enforces_a_pinned_version():
    # A pinned dep that is installed at a different version FAILS (the consensus break a
    # flashinfer/cutlass skew would silently cause); a not-installed pinned dep FAILS.
    synthetic = (
        PinnedDep("pytest", "0.0.0-not-real", "a pinned dep installed at another version"),
        PinnedDep("definitely-not-installed-xyz", "1.2.3", "pinned but absent"),
    )
    rows = {name: (ok, detail) for name, ok, detail in check_blessed_base(synthetic)}
    assert rows["pytest"][0] is False and "DIFFERS" in rows["pytest"][1]
    assert rows["definitely-not-installed-xyz"][0] is False
    assert "NOT INSTALLED" in rows["definitely-not-installed-xyz"][1]


# ---- the optima_kernels boundary: zero sglang imports (the moat is portable) ----

def test_optima_kernels_has_no_sglang_import():
    pat = re.compile(r"^\s*(import\s+sglang|from\s+sglang)", re.MULTILINE)
    offenders = []
    for p in Path("optima_kernels").rglob("*.py"):
        if pat.search(p.read_text(encoding="utf-8")):
            offenders.append(str(p))
    assert not offenders, f"optima_kernels must not import sglang: {offenders}"
