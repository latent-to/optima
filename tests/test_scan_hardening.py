"""scan_source hardening (#5): close the literal-AST-denylist bypasses.

Still a tripwire, not the boundary (isolation is) — but the trivial one-liners the
report flagged should no longer pass clean.
"""

from pathlib import Path

import pytest

from optima.sandbox import scan_source, scan_tree

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("src", [
    "import os\nx = getattr(os, 'sys' + 'tem')('id')\n",          # dynamic getattr
    "x = __builtins__['eval']('1+1')\n",                          # builtins subscript
    "g = globals()\n",                                            # namespace exposure
    "v = vars()\n",
    "y = ().__class__.__bases__[0]\n",                            # __class__ escape hop
    "import os\nsetattr(os, 'x'+'y', 1)\n",                       # dynamic setattr
    "import os\nf = os.system\nf('id')\n",                        # banned-callable ALIAS (no Call at the access)
    "import os\ncmds = [os.system]\ncmds[0]('id')\n",             # alias via a container
    "import dill\nl = dill.loads\nl(b'')\n",                      # deserializer alias
])
def test_known_bypasses_are_flagged(src):
    assert not scan_source(src).ok, f"should have flagged: {src!r}"


def test_scan_tree_flags_symlinks_fail_closed(tmp_path):
    # rglob does not follow directory symlinks, so a symlinked dir of .py files would
    # be invisible to the scan while staying perfectly importable at runtime. Any
    # symlink in a bundle is now a violation in itself.
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


@pytest.mark.parametrize("src", [
    "import torch\ndef k(x, out):\n    out.copy_(torch.relu(x))\n",
    "class C:\n    pass\nc = C()\nv = getattr(c, 'attr', None)\n",   # LITERAL getattr is fine
    "d = {'a': 1}\nx = d['a']\n",                                    # ordinary subscript fine
])
def test_legitimate_code_not_flagged(src):
    assert scan_source(src).ok, f"false positive on: {src!r}"


def test_all_example_kernels_still_scan_clean():
    # No false positives on the shipped bundles after hardening.
    for kernel in EXAMPLES.glob("*/kernels/*.py"):
        res = scan_source(kernel.read_text(), filename=kernel.name)
        assert res.ok, f"{kernel}: {res.violations}"
