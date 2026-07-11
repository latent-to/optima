"""Dispatcher conservatism gates: paths whose semantics the slot contracts don't
model must fall back to the trusted baseline instead of silently computing the
wrong thing (and framing an honest kernel for the resulting KL failure).

Pins three of them:
* attention decode: ANY unexpected kwarg (gpt-oss ``sinks``, MLA ``k_rope``) -> baseline;
* rmsnorm: ``variance_size_override`` / ``cast_x_before_out_mul`` / ``fp32_residual`` -> baseline;
* moe.fused_experts_reduce: a TP layer with ``reduce_results=False`` defers its reduce
  downstream, so the reduce-OWNING kernel must not run (it would insert an extra reduce).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from optima.dispatch import (  # noqa: E402
    _decode_supported,
    make_moe_dispatcher,
    make_rmsnorm_dispatcher,
)
from optima.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402
from optima.slots import get_slot  # noqa: E402

_BASELINE = object()  # sentinel: the dispatcher fell back


class _Param:
    def __init__(self, t):
        self.data = t


# ---- attention decode: unknown kwargs -------------------------------------------------


def _attn_self():
    return SimpleNamespace(qk_head_dim=64, v_head_dim=64, is_cross_attention=False,
                           sliding_window_size=-1)


def test_decode_supported_plain_mha():
    k = v = torch.zeros(2, 64)
    assert _decode_supported(_attn_self(), k, v, {})


@pytest.mark.parametrize("kwargs", [
    {"sinks": torch.zeros(8)},           # gpt-oss attention sinks
    {"k_rope": torch.zeros(2, 64)},      # MLA rope split
    {"anything_new_sglang_adds": 1},     # future-proof: unknown means unmodeled
])
def test_decode_supported_rejects_any_extra_kwarg(kwargs):
    k = v = torch.zeros(2, 64)
    assert not _decode_supported(_attn_self(), k, v, kwargs)


# ---- rmsnorm: semantic overrides -------------------------------------------------------


def _rms_entry(x, weight, out, eps):
    v = (x.float() * x.float()).mean(-1, keepdim=True)
    out.copy_((x.float() * torch.rsqrt(v + eps)).to(x.dtype) * weight)


def _rms_registry():
    reg = KernelRegistry()
    reg.register(KernelImpl(slot="norm.rmsnorm", bundle_id="t", entry=_rms_entry,
                            eligibility=Eligibility(dtypes=frozenset({"float32"}))))
    reg.enable()
    return reg


def _rms_self(**extra):
    s = SimpleNamespace(variance_epsilon=1e-6, weight=_Param(torch.ones(16)))
    for k, v in extra.items():
        setattr(s, k, v)
    return s


def test_rmsnorm_plain_layer_routes_to_kernel():
    dispatched = make_rmsnorm_dispatcher(lambda *a: _BASELINE, registry=_rms_registry())
    out = dispatched(_rms_self(), torch.randn(4, 16))
    assert out is not _BASELINE and torch.is_tensor(out)


@pytest.mark.parametrize("attrs", [
    {"variance_size_override": 8},     # variance over a prefix of the hidden dim
    {"cast_x_before_out_mul": True},   # HF cast-before-multiply semantics
    {"fp32_residual": True},
])
def test_rmsnorm_semantic_overrides_fall_back(attrs):
    dispatched = make_rmsnorm_dispatcher(lambda *a: _BASELINE, registry=_rms_registry())
    assert dispatched(_rms_self(**attrs), torch.randn(4, 16)) is _BASELINE


# ---- moe.fused_experts_reduce: layers that defer their reduce --------------------------


def _moe_inputs():
    slot = get_slot("moe.fused_experts_reduce")
    shape = {"num_tokens": 8, "num_experts": 4, "hidden": 64, "inter": 32, "topk": 2}
    return slot.make_inputs(**shape, dtype=torch.float32, device="cpu", seed=0)


def _moe_layer(inputs, *, moe_tp_size, reduce_results):
    return SimpleNamespace(
        w13_weight=_Param(inputs["w13"]), w2_weight=_Param(inputs["w2"]),
        moe_tp_size=moe_tp_size, moe_ep_size=1, reduce_results=reduce_results,
    )


def _moe_reduce_registry(calls):
    def entry(x, topk_ids, topk_weights, prepared, out, group=None):
        calls.append("fired")
        out.zero_()

    reg = KernelRegistry()
    reg.register(KernelImpl(
        slot="moe.fused_experts_reduce", bundle_id="t", entry=entry,
        prepare=lambda w13, w2: (w13, w2),
        eligibility=Eligibility(dtypes=frozenset({"float32"})),
    ))
    reg.enable()
    return reg


def test_reduce_owning_kernel_skipped_when_tp_layer_defers_its_reduce(monkeypatch):
    import optima.dispatch as dispatch

    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)
    inputs = _moe_inputs()
    calls: list = []
    dispatched = make_moe_dispatcher(lambda *a: _BASELINE, registry=_moe_reduce_registry(calls))
    topk = SimpleNamespace(topk_ids=inputs["topk_ids"], topk_weights=inputs["topk_weights"])
    layer = _moe_layer(inputs, moe_tp_size=2, reduce_results=False)
    assert dispatched(layer, inputs["x"], topk) is _BASELINE
    assert calls == []  # the kernel never ran — an extra all-reduce would diverge from stock


def test_reduce_owning_kernel_runs_when_layer_reduces(monkeypatch):
    import optima.dispatch as dispatch

    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        dispatch, "_tp_device_group", lambda: SimpleNamespace(size=lambda: 2)
    )
    inputs = _moe_inputs()
    calls: list = []
    dispatched = make_moe_dispatcher(lambda *a: _BASELINE, registry=_moe_reduce_registry(calls))
    topk = SimpleNamespace(topk_ids=inputs["topk_ids"], topk_weights=inputs["topk_weights"])
    layer = _moe_layer(inputs, moe_tp_size=2, reduce_results=True)
    out = dispatched(layer, inputs["x"], topk)
    assert out is not _BASELINE and calls == ["fired"]
