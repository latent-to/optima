"""Torch-free regression tests for the validator's static/intake layers.

These run anywhere (no GPU, no torch). They cover the security-critical pieces:
the policy scanner and the manifest path-safety checks, plus eligibility and KL.
"""

from __future__ import annotations

import optima.manifest as M
from optima.registry import eligibility_from_metadata
from optima.sandbox import scan_source

TRITON_BUNDLE = "examples/miner_silu_triton"


# ---- scanner ----------------------------------------------------------------


def test_scanner_allows_triton_load_store():
    src = "import triton.language as tl\nx = tl.load(p)\ntl.store(q, x)\n"
    assert scan_source(src).ok


def test_scanner_catches_egress_and_ace():
    evil = (
        "import socket\n"
        "import pickle\n"
        "import torch\n"
        "def f(x, out):\n"
        "    socket.socket().connect(('h', 1))\n"
        "    pickle.loads(b'')\n"
        "    torch.load('/w')\n"
        "    __import__('os').system('x')\n"
        "    eval('1')\n"
    )
    r = scan_source(evil, filename="evil.py")
    assert not r.ok
    joined = "\n".join(r.violations)
    for needle in ["socket", "pickle.loads", "torch.load", "system", "eval", "__import__"]:
        assert needle in joined, needle


# ---- manifest ---------------------------------------------------------------

_GOOD = {
    "bundle_id": "example-silu-triton-v1",
    "abi_version": M.ABI_VERSION,
    "ops": [
        {
            "slot": "activation.silu_and_mul",
            "source": "kernels/silu_and_mul.py",
            "entry": "silu_and_mul",
            "dtypes": ["bfloat16"],
            "metadata": "metadata/silu_and_mul.json",
        }
    ],
}


def _with_loader(payload, fn):
    orig = M._load_toml
    M._load_toml = lambda p: payload
    try:
        return fn()
    finally:
        M._load_toml = orig


def test_manifest_valid():
    m = _with_loader(_GOOD, lambda: M.load_manifest(TRITON_BUNDLE))
    assert m.bundle_id == "example-silu-triton-v1"
    assert m.op_for("activation.silu_and_mul").entry == "silu_and_mul"


def test_manifest_rejects_path_escape():
    bad = {**_GOOD, "ops": [dict(_GOOD["ops"][0], source="../../../../etc/passwd")]}
    try:
        _with_loader(bad, lambda: M.load_manifest(TRITON_BUNDLE))
        raise AssertionError("expected ManifestError")
    except M.ManifestError as e:
        assert "escapes bundle root" in str(e)


def test_manifest_rejects_absolute_path():
    bad = {**_GOOD, "ops": [dict(_GOOD["ops"][0], source="/etc/passwd")]}
    try:
        _with_loader(bad, lambda: M.load_manifest(TRITON_BUNDLE))
        raise AssertionError("expected ManifestError")
    except M.ManifestError as e:
        assert "must be relative" in str(e)


def test_manifest_rejects_foreign_abi():
    bad = {**_GOOD, "abi_version": "not-ours"}
    try:
        _with_loader(bad, lambda: M.load_manifest(TRITON_BUNDLE))
        raise AssertionError("expected ManifestError")
    except M.ManifestError as e:
        assert "abi_version" in str(e)


def test_manifest_override_point_fields():
    payload = {**_GOOD, "ops": [dict(
        _GOOD["ops"][0], base_kernel="nvfp4_moe_megakernel", override_point="gemm1_epilogue")]}
    m = _with_loader(payload, lambda: M.load_manifest(TRITON_BUNDLE))
    op = m.op_for("activation.silu_and_mul")
    assert op.base_kernel == "nvfp4_moe_megakernel"
    assert op.override_point == "gemm1_epilogue"
    assert op.is_override
    # base_kernel/override_point are first-class, not swept into extra.
    assert "base_kernel" not in op.extra and "override_point" not in op.extra


def test_manifest_override_point_requires_base_kernel():
    bad = {**_GOOD, "ops": [dict(_GOOD["ops"][0], override_point="gemm1_epilogue")]}
    try:
        _with_loader(bad, lambda: M.load_manifest(TRITON_BUNDLE))
        raise AssertionError("expected ManifestError")
    except M.ManifestError as e:
        assert "requires 'base_kernel'" in str(e)


# ---- eligibility ------------------------------------------------------------


def test_eligibility_gates():
    e = eligibility_from_metadata(
        {"dtypes": ["bfloat16"], "architectures": ["sm90"], "max_last_dim": 8192},
        ("bfloat16",),
    )
    assert e.accepts(dtype_name="bfloat16", last_dim=4096, arch="sm90")
    assert not e.accepts(dtype_name="float32", last_dim=4096, arch="sm90")
    assert not e.accepts(dtype_name="bfloat16", last_dim=4096, arch="sm80")
    assert not e.accepts(dtype_name="bfloat16", last_dim=9000, arch="sm90")
