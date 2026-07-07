"""collective.ar_residual_rmsnorm — the fused AR+residual+RMSNorm epilogue slot.

Covers the four layers added for this seam:
* the SlotSpec contract (two outputs; trusted post-reduce math via collective_finish;
  per-rank vs replicated input seeding);
* the seam-table row + the compat canary's module-level-function chokepoint form;
* the integration's module-attribute rebind (install/uninstall on a fake module);
* the dispatcher gates (opt-in env, token-count cap, graph_safe under capture,
  error fallback vs strict) and the distributed verify on gloo/CPU (faithful passes,
  a kernel that skips the residual add fails).
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from optima.compat import _chokepoint_present  # noqa: E402
from optima.dispatch import make_arfusion_dispatcher  # noqa: E402
from optima.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402
from optima.seams import SEAM_ADAPTERS, TARGET_MODULES  # noqa: E402
from optima.slots import get_slot  # noqa: E402
from optima.verify_collective import verify_collective  # noqa: E402

SLOT = "collective.ar_residual_rmsnorm"
_MODULE = "sglang.srt.layers.flashinfer_comm_fusion"


# ---- slot contract ---------------------------------------------------------------


def test_slot_registered_collective_two_outputs():
    slot = get_slot(SLOT)
    assert slot.kind == "collective"
    inputs = slot.make_inputs(num_tokens=8, hidden=32, dtype=torch.float32, device="cpu", seed=0)
    assert slot.out_shapes(inputs) == [(8, 32), (8, 32)]
    assert slot.collective_finish is not None


def test_inputs_split_per_rank_vs_replicated():
    # x is the per-rank partial (differs by rank); residual + weight are replicated
    # model state (identical across ranks) — the seeding split the M3 shared-expert
    # lesson demands.
    slot = get_slot(SLOT)
    r0 = slot.make_inputs(num_tokens=4, hidden=16, dtype=torch.float32, device="cpu",
                          seed=7, rank=0, world_size=2)
    r1 = slot.make_inputs(num_tokens=4, hidden=16, dtype=torch.float32, device="cpu",
                          seed=7, rank=1, world_size=2)
    assert not torch.equal(r0["x"], r1["x"])
    assert torch.equal(r0["residual"], r1["residual"])
    assert torch.equal(r0["weight"], r1["weight"])


def test_collective_finish_matches_reference_math():
    slot = get_slot(SLOT)
    inputs = slot.make_inputs(num_tokens=4, hidden=16, dtype=torch.float32, device="cpu", seed=1)
    summed = torch.randn(4, 16)
    norm_out, new_residual = slot.collective_finish(inputs, summed, None)
    exp_res = summed + inputs["residual"].float()
    var = exp_res.pow(2).mean(dim=-1, keepdim=True)
    exp_norm = exp_res * torch.rsqrt(var + float(inputs["eps"])) * inputs["weight"].float()
    assert torch.allclose(new_residual, exp_res)
    assert torch.allclose(norm_out, exp_norm)


# ---- seam table + canary ----------------------------------------------------------


def test_seam_table_has_arfusion_row():
    row = next(a for a in SEAM_ADAPTERS if a.name == "arfusion")
    assert row.target_module == _MODULE
    assert row.chokepoint == "flashinfer_allreduce_residual_rmsnorm"
    assert SLOT in row.slots
    assert _MODULE in TARGET_MODULES


def test_chokepoint_present_module_function_form():
    mod = ModuleType("fake_mod")
    mod.some_fn = lambda: None

    class Klass:
        def meth(self):
            pass

    mod.Klass = Klass
    assert _chokepoint_present(mod, "some_fn")
    assert not _chokepoint_present(mod, "missing_fn")
    assert _chokepoint_present(mod, "Klass.meth")
    assert not _chokepoint_present(mod, "Klass.missing")


# ---- integration: module-attribute rebind -----------------------------------------


def _fake_fusion_module():
    mod = ModuleType(_MODULE)

    def flashinfer_allreduce_residual_rmsnorm(input_tensor, residual, weight, eps=1e-6,
                                              max_token_num=2048, use_oneshot=None,
                                              trigger_completion_at_end=False, fp32_acc=False,
                                              use_attn_tp_group=True):
        return ("stock", input_tensor)

    mod.flashinfer_allreduce_residual_rmsnorm = flashinfer_allreduce_residual_rmsnorm
    return mod


def test_install_rebinds_module_attribute(monkeypatch):
    from optima.integrations import sglang_arfusion

    mod = _fake_fusion_module()
    orig = mod.flashinfer_allreduce_residual_rmsnorm
    monkeypatch.setitem(sys.modules, _MODULE, mod)

    sglang_arfusion.install(KernelRegistry())
    assert sglang_arfusion.is_installed()
    assert mod.flashinfer_allreduce_residual_rmsnorm is not orig
    assert mod._optima_orig_allreduce_residual_rmsnorm is orig
    # Inactive registry -> dispatcher passes through to the stock function.
    x = torch.zeros(2, 8)
    assert mod.flashinfer_allreduce_residual_rmsnorm(x, x, x[0])[0] == "stock"

    sglang_arfusion.uninstall()
    assert not sglang_arfusion.is_installed()
    assert mod.flashinfer_allreduce_residual_rmsnorm is orig


# ---- dispatcher gates --------------------------------------------------------------


def _entry_fill(x, residual, weight, eps, out_norm, out_residual, group):
    out_norm.fill_(1.0)
    out_residual.fill_(2.0)


def _registry(**elig):
    reg = KernelRegistry()
    reg.register(KernelImpl(slot=SLOT, bundle_id="t", entry=_entry_fill,
                            eligibility=Eligibility(dtypes=frozenset({"float32"}), **elig)))
    reg.enable()
    return reg


def _baseline_recorder(calls):
    def baseline(input_tensor, residual, weight, eps=1e-6, max_token_num=2048,
                 use_oneshot=None, trigger_completion_at_end=False, fp32_acc=False,
                 use_attn_tp_group=True):
        calls.append("baseline")
        return ("baseline", input_tensor)
    return baseline


@pytest.fixture()
def _fake_group(monkeypatch):
    import optima.dispatch as dispatch

    monkeypatch.setattr(dispatch, "_arfusion_group",
                        lambda use_attn: SimpleNamespace(size=lambda: 2))


def test_dispatcher_inactive_without_env(monkeypatch, _fake_group):
    monkeypatch.delenv("OPTIMA_ARFUSION_SEAM", raising=False)
    calls = []
    d = make_arfusion_dispatcher(_baseline_recorder(calls), registry=_registry())
    x = torch.zeros(4, 8)
    assert d(x, x.clone(), x[0])[0] == "baseline"
    assert calls == ["baseline"]


def test_dispatcher_routes_to_kernel(monkeypatch, _fake_group):
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    calls = []
    d = make_arfusion_dispatcher(_baseline_recorder(calls), registry=_registry())
    x = torch.zeros(4, 8)
    out_norm, out_residual = d(x, x.clone(), x[0])
    assert calls == []
    assert torch.equal(out_norm, torch.full((4, 8), 1.0))
    assert torch.equal(out_residual, torch.full((4, 8), 2.0))


def test_dispatcher_token_cap_falls_back(monkeypatch, _fake_group):
    # A kernel with a MEASURED dispatch window (decode-sized T) must not see
    # prefill-sized calls — the seam, not the kernel, enforces the cap.
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    calls = []
    d = make_arfusion_dispatcher(_baseline_recorder(calls),
                                 registry=_registry(max_num_tokens=1024))
    ok = torch.zeros(1024, 8)
    big = torch.zeros(1025, 8)
    assert isinstance(d(ok, ok.clone(), ok[0]), tuple) and calls == []
    assert d(big, big.clone(), big[0])[0] == "baseline"
    assert calls == ["baseline"]


def test_dispatcher_graph_capture_requires_graph_safe(monkeypatch, _fake_group):
    import optima.dispatch as dispatch

    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    monkeypatch.setattr(dispatch, "_in_cuda_graph", lambda: True)
    calls = []
    d = make_arfusion_dispatcher(_baseline_recorder(calls), registry=_registry())
    x = torch.zeros(4, 8)
    assert d(x, x.clone(), x[0])[0] == "baseline"  # undeclared -> stock in-graph

    calls2 = []
    d2 = make_arfusion_dispatcher(_baseline_recorder(calls2),
                                  registry=_registry(graph_safe=True))
    out = d2(x, x.clone(), x[0])
    assert calls2 == [] and torch.equal(out[0], torch.full((4, 8), 1.0))


def test_dispatcher_kernel_error_falls_back_unless_strict(monkeypatch, _fake_group):
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")

    def boom(*a):
        raise RuntimeError("kernel exploded")

    reg = KernelRegistry()
    reg.register(KernelImpl(slot=SLOT, bundle_id="t", entry=boom,
                            eligibility=Eligibility(dtypes=frozenset({"float32"}))))
    reg.enable()
    calls = []
    d = make_arfusion_dispatcher(_baseline_recorder(calls), registry=reg)
    x = torch.zeros(4, 8)
    assert d(x, x.clone(), x[0])[0] == "baseline"
    assert calls == ["baseline"]

    reg.set_strict(True)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        d(x, x.clone(), x[0])


# ---- distributed verify (gloo / CPU, 2 ranks) ---------------------------------------

FAITHFUL = """
import torch
import torch.distributed as dist

def ar_residual_rmsnorm(x, residual, weight, eps, out_norm, out_residual, group):
    s = x.float().clone()
    dist.all_reduce(s, op=dist.ReduceOp.SUM, group=group)
    new_res = s + residual.float()
    var = new_res.pow(2).mean(dim=-1, keepdim=True)
    norm = new_res * torch.rsqrt(var + float(eps)) * weight.float()
    out_residual.copy_(new_res.to(out_residual.dtype))
    out_norm.copy_(norm.to(out_norm.dtype))
"""

# Skips the residual add — the exact wrong-math class the fused epilogue can hide
# (norm of the bare sum still LOOKS like a plausible normed tensor).
NO_RESIDUAL = FAITHFUL.replace("s + residual.float()", "s + 0.0 * residual.float()")


def test_arfusion_faithful_passes_gloo_cpu(tmp_path):
    p = tmp_path / "faithful.py"
    p.write_text(FAITHFUL)
    res = verify_collective(get_slot(SLOT), str(p), "ar_residual_rmsnorm",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert res.passed, "\n".join(f"{r.shape}: {r.detail}" for r in res.shape_results)


def test_arfusion_no_residual_fails_gloo_cpu(tmp_path):
    p = tmp_path / "no_residual.py"
    p.write_text(NO_RESIDUAL)
    res = verify_collective(get_slot(SLOT), str(p), "ar_residual_rmsnorm",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert not res.passed
