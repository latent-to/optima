"""optima_kernels.collective.fused_ar_rmsnorm — the portable library spine, CPU-only.

No GPU, no sglang. Pins three things:
* the library's ``reference`` and the VALIDATOR's slot reference (slots.py
  ``collective_finish``) are independent implementations that must agree — the gate's
  math is deliberately NOT imported from the library that ships the kernel;
* the measured dispatch constants survive refactors (one-shot/two-shot crossover 48,
  prefill fall-through 1024 — each was a real regression once);
* the module stays import-clean of sglang and the harness (Axiom 5).
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

from optima_kernels.collective import fused_ar_rmsnorm as far  # noqa: E402


def test_reference_agrees_with_validator_slot_reference():
    from optima.slots import get_slot

    slot = get_slot("collective.ar_residual_rmsnorm")
    inputs = slot.make_inputs(num_tokens=16, hidden=64, dtype=torch.float32,
                              device="cpu", seed=3)
    summed = torch.randn(16, 64)
    lib_norm, lib_res = far.reference(summed, inputs["residual"], inputs["weight"],
                                      inputs["eps"])
    val_norm, val_res = slot.collective_finish(inputs, summed, None)
    assert torch.allclose(lib_norm, val_norm)
    assert torch.allclose(lib_res, val_res)


def test_measured_dispatch_constants():
    assert far.TWOSHOT_MIN == 48 and far.MAX_T == 1024
    assert far.mode_for(47) == 1
    assert far.mode_for(48) == 2
    assert far.mode_for(1024) == 2


def test_init_requires_eager_and_uninitialized_call_raises():
    with pytest.raises(RuntimeError, match="init"):
        x = torch.zeros(4, 8)
        far.ar_residual_rmsnorm(None, x, x, x[0], 1e-6, x.clone(), x.clone(), None)


def test_no_sglang_or_harness_imports():
    assert "optima_kernels.collective.fused_ar_rmsnorm" in sys.modules
    src = open(far.__file__).read()
    assert "import sglang" not in src and "from sglang" not in src
    assert "from optima." not in src and "import optima." not in src.replace("optima_kernels", "")
