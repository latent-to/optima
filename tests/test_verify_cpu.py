"""Op-correctness test that needs torch but not a GPU.

Runs the full slot -> sandbox-load -> verify_entry path against the pure-torch
example bundle on CPU. Skipped automatically where torch is unavailable (e.g. the
dev laptop); runs on the VM.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from optima.sandbox import load_entry  # noqa: E402
from optima.slots import get_slot  # noqa: E402
from optima.verify import verify_entry  # noqa: E402

from pathlib import Path  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent  # cwd-independent
TORCH_BUNDLE = str(_REPO / "examples/miner_silu_torch/kernels/silu_and_mul.py")
BROKEN_TORCH_BUNDLE = str(_REPO / "examples/miner_silu_broken_torch/kernels/silu_and_mul.py")


def test_torch_silu_passes_correctness_cpu():
    entry = load_entry(TORCH_BUNDLE, "silu_and_mul")
    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, entry, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed, "\n".join(
        f"{r.shape}: max_abs={r.max_abs_err} {r.detail}" for r in result.shape_results
    )


def test_broken_torch_example_bundle_fails_cpu():
    # The committed adversarial bundle the miner guide's no-GPU walkthrough runs
    # (drops the SiLU). If this ever passes verify, the walkthrough demo is broken.
    entry = load_entry(BROKEN_TORCH_BUNDLE, "silu_and_mul")
    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, entry, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed


def test_wrong_kernel_fails_correctness_cpu():
    # A deliberately broken "kernel": forgets the multiply, just copies silu(gate).
    def broken(x, out):
        d = x.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(x[..., :d]))

    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, broken, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed
