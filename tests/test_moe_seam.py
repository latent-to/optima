"""CPU tests for the FusedMoE block seam (optima.dispatch.make_moe_dispatcher).

The seam replaces ``FusedMoE.forward_impl(self, hidden_states, topk_output)`` — the
waist every path (eager, in-piecewise, the two piecewise custom ops) converges on; a
patch on ``.forward`` is bypassed under piecewise capture. These tests drive that
dispatcher directly with a *fake* layer + a fake standard ``topk_output`` — no GPU, no
sglang — so the routing extraction, the (prepare, forward) wiring, the validator-owned
output allocation, and the conservative fallbacks are all exercised on the laptop. The
faithful kernel is the real example bundle; correctness is checked against the slot's
own fp32 reference. The install path (which method gets patched) is covered by
``test_install_patches_forward_impl`` below.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import optima.dispatch as dispatch  # noqa: E402
from optima.dispatch import make_moe_dispatcher  # noqa: E402
from optima.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402
from optima.sandbox import load_entry  # noqa: E402
from optima.slots import get_slot  # noqa: E402

MOE_BUNDLE = "examples/miner_moe_fused_experts_torch/kernels/moe.py"
_BASELINE = object()  # sentinel: the dispatcher returns this iff it fell back


@pytest.fixture(autouse=True)
def _single_moe_dp_rank(monkeypatch):
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)


def _baseline_forward(self, hidden_states, topk_output):
    return _BASELINE


class _Param:  # mimics torch.nn.Parameter's .data access the seam uses
    def __init__(self, t):
        self.data = t


def _fake_layer(inputs, *, moe_tp_size=1, moe_ep_size=1, reduce_results=False, **extra):
    layer = SimpleNamespace(
        w13_weight=_Param(inputs["w13"]),
        w2_weight=_Param(inputs["w2"]),
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        reduce_results=reduce_results,
    )
    for k, v in extra.items():  # e.g. w13_weight_scale to simulate a quantized layer
        setattr(layer, k, v)
    return layer


def _standard_topk_output(inputs):
    # Duck-typed StandardTopKOutput: carries explicit topk tensors.
    return SimpleNamespace(
        topk_ids=inputs["topk_ids"], topk_weights=inputs["topk_weights"], router_logits=None
    )


def _bypassed_topk_output():
    # BypassedTopKOutput has no topk_ids/topk_weights (routing not materialized).
    return SimpleNamespace(hidden_states=None, router_logits=None, topk_config=None)


def _inputs(seed=0):
    slot = get_slot("moe.fused_experts")
    shape = {"num_tokens": 16, "num_experts": 8, "hidden": 256, "inter": 128, "topk": 2}
    return slot.make_inputs(**shape, dtype=torch.float32, device="cpu", seed=seed)


def _registry(entry, prepare, *, quant=frozenset()):
    reg = KernelRegistry()
    reg.register(
        KernelImpl(
            slot="moe.fused_experts",
            bundle_id="test",
            entry=entry,
            prepare=prepare,
            eligibility=Eligibility(dtypes=frozenset({"float32"}), quant=quant),
        )
    )
    reg.enable()
    return reg


def _matched_ratio(actual, expected, atol=1e-4, rtol=1e-4):
    within = (actual.float() - expected.float()).abs() <= (atol + rtol * expected.float().abs())
    return within.float().mean().item()


def test_seam_routes_to_kernel_and_matches_reference(monkeypatch):
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    slot = get_slot("moe.fused_experts")
    reference = slot.invoke_reference(inputs)[0]

    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))

    out = dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs))
    assert out is not _BASELINE, "seam should have routed to the miner kernel"
    assert tuple(out.shape) == (inputs["x"].shape[0], inputs["x"].shape[1])
    assert _matched_ratio(out, reference) >= slot.correctness.min_ratio


def test_disabled_when_env_flag_unset(monkeypatch):
    # Opt-in: with OPTIMA_MOE_SEAM unset the seam is inert (trusts the baseline).
    monkeypatch.delenv("OPTIMA_MOE_SEAM", raising=False)
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))
    assert dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs)) is _BASELINE


def test_prepare_runs_once_and_is_memoized(monkeypatch):
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    base_prepare = load_entry(MOE_BUNDLE, "prepare")
    calls = {"n": 0}

    def counting_prepare(w13, w2):
        calls["n"] += 1
        return base_prepare(w13, w2)

    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, counting_prepare))
    layer = _fake_layer(inputs)
    topk = _standard_topk_output(inputs)
    dispatched(layer, inputs["x"], topk)
    dispatched(layer, inputs["x"], topk)
    assert calls["n"] == 1, "prepare must run ONCE per layer (memoized), not per step"


def test_expert_parallel_falls_back(monkeypatch):
    # EP>1 adds an all-to-all the (M,H)->(M,H) contract doesn't model -> trust baseline.
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))
    layer = _fake_layer(inputs, moe_ep_size=2)
    assert dispatched(layer, inputs["x"], _standard_topk_output(inputs)) is _BASELINE


def test_bypassed_routing_falls_back(monkeypatch):
    # Routing not materialized (no topk tensors) -> conservative fallback (no re-routing).
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))
    assert dispatched(_fake_layer(inputs), inputs["x"], _bypassed_topk_output()) is _BASELINE


def test_missing_prepare_falls_back(monkeypatch):
    # A (prepare, forward) slot with no prepare loaded can't honor the contract -> baseline.
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare=None))
    assert dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs)) is _BASELINE


def test_quantized_layer_falls_back_for_dense_kernel(monkeypatch):
    # A quantized layer (packed bytes + *_scale) must NOT run a DENSE kernel (empty
    # Eligibility.quant) — it would mis-read the weights -> fall back. The pairing gate
    # (_quant_ok) never crosses dense<->quant.
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))  # dense
    layer = _fake_layer(inputs, w13_weight_scale=_Param(torch.ones(1)))
    assert dispatched(layer, inputs["x"], _standard_topk_output(inputs)) is _BASELINE


def test_quantized_layer_routes_when_kernel_declares_format(monkeypatch):
    # The unblock: a quantized layer DOES route to a kernel that DECLARES its format
    # (Eligibility.quant={"nvfp4"}). The gate admits it; the kernel runs.
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    reg = _registry(entry, prepare, quant=frozenset({"nvfp4"}))
    dispatched = make_moe_dispatcher(_baseline_forward, registry=reg)
    layer = _fake_layer(inputs, w13_weight_scale=_Param(torch.ones(1)))
    out = dispatched(layer, inputs["x"], _standard_topk_output(inputs))
    assert out is not _BASELINE, "a format-declaring kernel must be admitted for a quant layer"
    assert tuple(out.shape) == (inputs["x"].shape[0], inputs["x"].shape[1])


def test_dense_layer_skips_quant_only_kernel(monkeypatch):
    # The other direction: a DENSE layer must NOT run a quant-only kernel (it expects
    # scales that aren't there) -> fall back.
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    reg = _registry(entry, prepare, quant=frozenset({"nvfp4"}))
    dispatched = make_moe_dispatcher(_baseline_forward, registry=reg)
    assert dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs)) is _BASELINE


def test_non_2d_hidden_states_falls_back(monkeypatch):
    # The (M,H)->(M,H) contract assumes flattened 2D tokens; anything else -> baseline.
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    entry = load_entry(MOE_BUNDLE, "fused_experts")
    prepare = load_entry(MOE_BUNDLE, "prepare")
    dispatched = make_moe_dispatcher(_baseline_forward, registry=_registry(entry, prepare))
    x3d = inputs["x"].unsqueeze(0)  # (1, M, H)
    assert dispatched(_fake_layer(inputs), x3d, _standard_topk_output(inputs)) is _BASELINE


def test_raising_kernel_falls_back_unless_strict(monkeypatch):
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    inputs = _inputs()
    prepare = load_entry(MOE_BUNDLE, "prepare")

    def raising(x, topk_ids, topk_weights, prepared, out):
        raise RuntimeError("boom")

    reg = _registry(raising, prepare)
    dispatched = make_moe_dispatcher(_baseline_forward, registry=reg)
    # non-strict: a crashing kernel just loses -> baseline
    assert dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs)) is _BASELINE
    # strict: surface the error (used in debugging, not scoring)
    reg.set_strict(True)
    with pytest.raises(RuntimeError, match="boom"):
        dispatched(_fake_layer(inputs), inputs["x"], _standard_topk_output(inputs))


# ---- M0: the install path patches forward_impl (the piecewise waist), not forward ----

def test_install_patches_forward_impl(monkeypatch):
    """``.forward`` is bypassed under piecewise capture (it routes to custom ops that call
    ``forward_impl`` directly), so the seam MUST patch ``forward_impl``."""
    import sys
    from types import ModuleType

    from optima.integrations import sglang_moe

    def forward(self, hidden_states, topk_output):          # the router — must stay untouched
        return ("forward", hidden_states)

    def forward_impl(self, hidden_states, topk_output):     # the waist — must be patched
        return ("impl", hidden_states)

    class FakeFusedMoE:
        pass

    FakeFusedMoE.forward = forward
    FakeFusedMoE.forward_impl = forward_impl
    mod = ModuleType(sglang_moe._MODULE)
    mod.FusedMoE = FakeFusedMoE
    monkeypatch.setitem(sys.modules, sglang_moe._MODULE, mod)

    orig_forward, orig_impl = FakeFusedMoE.forward, FakeFusedMoE.forward_impl
    try:
        sglang_moe.install()
        assert sglang_moe.is_installed()
        assert FakeFusedMoE.forward_impl is not orig_impl          # patched
        assert FakeFusedMoE.forward is orig_forward                # router untouched
        assert FakeFusedMoE._optima_orig_forward_impl is orig_impl  # captured for fallback/uninstall
        sglang_moe.install()  # idempotent
        assert FakeFusedMoE._optima_orig_forward_impl is orig_impl
    finally:
        sglang_moe.uninstall()
    assert FakeFusedMoE.forward_impl is orig_impl
    assert not sglang_moe.is_installed()


def test_install_noop_without_forward_impl(monkeypatch):
    """An older sglang lacking ``forward_impl`` -> the seam stays inert (the compat canary
    flags the missing chokepoint rather than the seam silently patching the wrong method)."""
    import sys
    from types import ModuleType

    from optima.integrations import sglang_moe

    class OldFusedMoE:
        def forward(self, hidden_states, topk_output):
            return ("forward", hidden_states)

    mod = ModuleType(sglang_moe._MODULE)
    mod.FusedMoE = OldFusedMoE
    monkeypatch.setitem(sys.modules, sglang_moe._MODULE, mod)

    sglang_moe.install()
    assert not sglang_moe.is_installed()
    assert not hasattr(OldFusedMoE, "_optima_orig_forward_impl")
