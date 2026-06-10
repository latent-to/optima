"""scan_source hardening (#5): close the literal-AST-denylist bypasses.

Still a tripwire, not the boundary (isolation is) — but the trivial one-liners the
report flagged should no longer pass clean.
"""

from pathlib import Path

import pytest

from optima.sandbox import scan_source

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("src", [
    "import os\nx = getattr(os, 'sys' + 'tem')('id')\n",          # dynamic getattr
    "x = __builtins__['eval']('1+1')\n",                          # builtins subscript
    "g = globals()\n",                                            # namespace exposure
    "v = vars()\n",
    "y = ().__class__.__bases__[0]\n",                            # __class__ escape hop
    "import os\nsetattr(os, 'x'+'y', 1)\n",                       # dynamic setattr
])
def test_known_bypasses_are_flagged(src):
    assert not scan_source(src).ok, f"should have flagged: {src!r}"


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
