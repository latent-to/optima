"""Control-flow receipts across the non-MSA serving dispatcher families."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import optima.dispatch as dispatch  # noqa: E402
from optima.registry import Eligibility, KernelImpl, KernelRegistry  # noqa: E402


@pytest.fixture()
def events(monkeypatch):
    completed: list[str] = []
    fallbacks: list[tuple[str, str]] = []
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)
    monkeypatch.setattr(
        dispatch._receipts,
        "fallback",
        lambda slot, exc: fallbacks.append((slot, type(exc).__name__)),
    )
    monkeypatch.setattr(dispatch._audit, "sampled", lambda: False)
    monkeypatch.setattr(dispatch, "_moe_data_parallel_world_size", lambda: 1)
    return completed, fallbacks


def _registry(
    slot,
    entry,
    *,
    prepare=None,
    graph_safe=False,
    strict=False,
    dtype="float32",
):
    registry = KernelRegistry()
    registry.register(
        KernelImpl(
            slot=slot,
            bundle_id="test",
            entry=entry,
            prepare=prepare,
            eligibility=Eligibility(
                dtypes=frozenset({dtype}), graph_safe=graph_safe
            ),
        )
    )
    registry.enable()
    registry.set_strict(strict)
    return registry


def _boom(*_args, **_kwargs):
    raise RuntimeError("candidate path failed")


def test_op_dispatchers_receipt_success_and_fallback(events):
    completed, fallbacks = events
    baseline = object()
    silu = dispatch.make_silu_and_mul_dispatcher(
        lambda *_: baseline,
        registry=_registry(
            "activation.silu_and_mul",
            lambda x, out: out.copy_(x[..., : x.shape[-1] // 2]),
        ),
    )
    assert silu(object(), torch.randn(2, 8)) is not baseline

    rms_self = SimpleNamespace(
        variance_epsilon=1e-6,
        weight=SimpleNamespace(data=torch.ones(8)),
    )
    rms = dispatch.make_rmsnorm_dispatcher(
        lambda *_: baseline,
        registry=_registry(
            "norm.rmsnorm", lambda x, _weight, out, _eps: out.copy_(x)
        ),
    )
    assert rms(rms_self, torch.randn(2, 8)) is not baseline
    assert completed == ["activation.silu_and_mul", "norm.rmsnorm"]

    silu_bad = dispatch.make_silu_and_mul_dispatcher(
        lambda *_: baseline,
        registry=_registry("activation.silu_and_mul", _boom),
    )
    rms_bad = dispatch.make_rmsnorm_dispatcher(
        lambda *_: baseline, registry=_registry("norm.rmsnorm", _boom)
    )
    assert silu_bad(object(), torch.randn(2, 8)) is baseline
    assert rms_bad(rms_self, torch.randn(2, 8)) is baseline
    assert fallbacks == [
        ("activation.silu_and_mul", "RuntimeError"),
        ("norm.rmsnorm", "RuntimeError"),
    ]


def test_fallback_requires_successful_stock_and_strict_never_falls_back(events):
    completed, fallbacks = events

    def stock_fails(*_args):
        raise ValueError("stock also failed")

    wrapped = dispatch.make_silu_and_mul_dispatcher(
        stock_fails, registry=_registry("activation.silu_and_mul", _boom)
    )
    with pytest.raises(ValueError, match="stock also failed"):
        wrapped(object(), torch.randn(2, 8))
    assert completed == fallbacks == []

    strict = dispatch.make_silu_and_mul_dispatcher(
        lambda *_: pytest.fail("strict mode called stock"),
        registry=_registry("activation.silu_and_mul", _boom, strict=True),
    )
    with pytest.raises(RuntimeError, match="candidate path failed"):
        strict(object(), torch.randn(2, 8))
    assert completed == fallbacks == []


def _attention_inputs():
    class Pool:
        def __init__(self):
            self.k = torch.zeros(2, 1, 2)
            self.v = torch.zeros(2, 1, 2)

        def set_kv_buffer(self, _layer, locations, k, v):
            self.k[locations] = k
            self.v[locations] = v

        def get_key_buffer(self, _layer_id):
            return self.k

        def get_value_buffer(self, _layer_id):
            return self.v

    pool = Pool()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        token_to_kv_pool=pool,
        out_cache_loc=torch.tensor([1]),
        seq_lens=torch.tensor([2]),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[0, 1]])),
        req_pool_indices=torch.tensor([0]),
    )
    layer = SimpleNamespace(
        qk_head_dim=2,
        v_head_dim=2,
        tp_q_head_num=1,
        tp_k_head_num=1,
        scaling=1.0,
        layer_id=0,
        is_cross_attention=False,
        sliding_window_size=-1,
    )
    return layer, batch


def test_attention_dispatcher_receipts(events, monkeypatch):
    completed, fallbacks = events
    monkeypatch.setenv("OPTIMA_ATTENTION_SEAM", "1")
    baseline = object()
    layer, batch = _attention_inputs()
    args = (
        layer,
        torch.ones(1, 2),
        torch.ones(1, 1, 2),
        torch.ones(1, 1, 2),
        batch,
    )

    def entry(q, _k, _v, _seq_lens, _scale, out):
        out.copy_(q)

    good = dispatch.make_attention_dispatcher(
        lambda *_a, **_k: baseline,
        registry=_registry("attention.decode", entry),
    )
    assert good(*args) is not baseline
    bad = dispatch.make_attention_dispatcher(
        lambda *_a, **_k: baseline,
        registry=_registry("attention.decode", _boom),
    )
    assert bad(*args) is baseline
    assert completed == ["attention.decode"]
    assert fallbacks == [("attention.decode", "RuntimeError")]


def _moe_call(entry, *, slot="moe.fused_experts"):
    x = torch.randn(2, 4)
    layer = SimpleNamespace(
        w13_weight=SimpleNamespace(data=torch.randn(2, 4, 4)),
        w2_weight=SimpleNamespace(data=torch.randn(2, 4, 2)),
        moe_tp_size=1,
        moe_ep_size=1,
        reduce_results=False,
    )
    topk = SimpleNamespace(
        topk_ids=torch.zeros(2, 1, dtype=torch.long),
        topk_weights=torch.ones(2, 1),
    )
    registry = _registry(slot, entry, prepare=lambda *_: object())
    wrapped = dispatch.make_moe_dispatcher(
        lambda *_: "stock", registry=registry, slots=("moe.fused_experts_reduce", slot)
    )
    return wrapped, layer, x, topk


def test_moe_records_actual_selected_slot(events, monkeypatch):
    completed, fallbacks = events
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")

    def good_entry(x, _ids, _weights, _prepared, out):
        out.copy_(x)

    good, layer, x, topk = _moe_call(good_entry)
    assert torch.is_tensor(good(layer, x, topk))
    bad, layer, x, topk = _moe_call(_boom)
    assert bad(layer, x, topk) == "stock"
    assert completed == ["moe.fused_experts"]
    assert fallbacks == [("moe.fused_experts", "RuntimeError")]


def test_moe_selected_audit_prelude_failure_is_fallback(events, monkeypatch):
    completed, fallbacks = events
    monkeypatch.setenv("OPTIMA_MOE_SEAM", "1")
    monkeypatch.setattr(dispatch._audit, "sampled", lambda: True)
    monkeypatch.setattr(
        torch.Tensor,
        "clone",
        lambda _self: (_ for _ in ()).throw(RuntimeError("clone failed")),
    )

    def entry(x, _ids, _weights, _prepared, out):
        out.copy_(x)

    wrapped, layer, x, topk = _moe_call(entry)
    assert wrapped(layer, x, topk) == "stock"
    assert completed == []
    assert fallbacks == [("moe.fused_experts", "RuntimeError")]


def test_allreduce_dispatcher_receipts_and_topology_skip(events, monkeypatch):
    completed, fallbacks = events
    monkeypatch.setenv("OPTIMA_COLLECTIVE_SEAM", "1")
    monkeypatch.setattr(dispatch, "_allreduce_group_role", lambda *_args: "tp")
    x = torch.randn(2, 4)

    def good_entry(inp, out, _group):
        out.copy_(inp)

    good = dispatch.make_allreduce_dispatcher(
        lambda *_a, **_k: "stock",
        registry=_registry("collective.all_reduce", good_entry),
    )
    coordinator = SimpleNamespace(
        world_size=2, device_group=SimpleNamespace(size=lambda: 2)
    )
    assert torch.is_tensor(good(coordinator, x))
    bad = dispatch.make_allreduce_dispatcher(
        lambda *_a, **_k: "stock",
        registry=_registry("collective.all_reduce", _boom),
    )
    with pytest.raises(RuntimeError, match="candidate path failed"):
        bad(coordinator, x)
    assert good(SimpleNamespace(world_size=1, device_group=object()), x) == "stock"
    assert completed == ["collective.all_reduce"]
    assert fallbacks == []


def _fusion_baseline(x, residual, *_args, **_kwargs):
    return "stock", x + residual


def test_shallow_and_deep_fusion_receipts(events, monkeypatch):
    completed, fallbacks = events
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setattr(dispatch, "_arfusion_group", lambda _use_attn: group)
    monkeypatch.setattr(dispatch, "_arfusion_group_role", lambda _use_attn: "tp")
    x = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    weight = torch.ones(4)

    def shallow_entry(x, residual, _weight, _eps, out_norm, out_residual, _group):
        out_norm.copy_(x)
        out_residual.copy_(residual)

    shallow = dispatch.make_arfusion_dispatcher(
        _fusion_baseline,
        registry=_registry("collective.ar_residual_rmsnorm", shallow_entry),
    )
    assert torch.is_tensor(shallow(x, residual, weight)[0])
    shallow_bad = dispatch.make_arfusion_dispatcher(
        _fusion_baseline,
        registry=_registry("collective.ar_residual_rmsnorm", _boom),
    )
    with pytest.raises(RuntimeError, match="candidate path failed"):
        shallow_bad(x, residual, weight)

    deep_x = torch.randn(2, 4, dtype=torch.bfloat16)
    deep_residual = torch.randn(2, 4, dtype=torch.bfloat16)
    deep_weight = torch.ones(4, dtype=torch.bfloat16)
    monkeypatch.setattr(dispatch._moe_export, "has_pends", lambda: True)
    monkeypatch.setattr(
        dispatch._moe_export,
        "export_views",
        lambda _exp, _device: (
            torch.randn(2, 4, dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.int32),
            torch.ones(2, 1, dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(dispatch._moe_export, "trusted_finalize", lambda _exp, inp: inp)
    monkeypatch.setattr(dispatch._moe_export, "orphaned", lambda _exp: None)

    def deep_entry(
        _gemm, _row_map, _scales, residual, _weight, _eps,
        out_norm, out_residual, _group,
    ):
        out_norm.copy_(residual)
        out_residual.copy_(residual)

    deep_registry = _registry(
        "collective.moe_finalize_ar_rmsnorm", deep_entry, dtype="bfloat16"
    )
    deep = dispatch.make_arfusion_dispatcher(
        _fusion_baseline, registry=deep_registry
    )
    deep_impl = deep_registry.variants("collective.moe_finalize_ar_rmsnorm")[0]
    deep_descriptor = dispatch._collective_call_descriptor(
        deep_x, group_size=2, exp_tokens=2, top_k=1, ep_size=1
    )
    exp = {
        "T": 2,
        "K": 1,
        "hid": 4,
        "selection": dispatch._moe_export.DeepSelection(
            deep_impl,
            deep_descriptor,
            dispatch._moe_export.group_topology(group),
        ),
    }
    monkeypatch.setattr(dispatch._moe_export, "consume", lambda _x: exp)
    assert torch.is_tensor(deep(deep_x, deep_residual, deep_weight)[0])

    deep_bad_registry = _registry(
        "collective.moe_finalize_ar_rmsnorm", _boom, dtype="bfloat16"
    )
    deep_bad = dispatch.make_arfusion_dispatcher(
        _fusion_baseline,
        registry=deep_bad_registry,
    )
    bad_impl = deep_bad_registry.variants("collective.moe_finalize_ar_rmsnorm")[0]
    bad_exp = {
        **exp,
        "selection": dispatch._moe_export.DeepSelection(
            bad_impl,
            deep_descriptor,
            dispatch._moe_export.group_topology(group),
        ),
    }
    monkeypatch.setattr(dispatch._moe_export, "consume", lambda _x: bad_exp)
    with pytest.raises(RuntimeError, match="candidate path failed"):
        deep_bad(deep_x, deep_residual, deep_weight)

    assert completed == [
        "collective.ar_residual_rmsnorm",
        "collective.moe_finalize_ar_rmsnorm",
    ]
    assert fallbacks == []


def test_deep_trusted_recovery_is_not_candidate_fallback(events, monkeypatch):
    completed, fallbacks = events
    monkeypatch.setattr(dispatch, "_arfusion_group", lambda _use_attn: None)
    monkeypatch.setattr(dispatch._moe_export, "trusted_finalize", lambda _exp, inp: inp)
    monkeypatch.setattr(dispatch._moe_export, "orphaned", lambda _exp: None)
    registry = KernelRegistry()
    registry.enable()
    result = dispatch._deep_consume(
        {"T": 2, "K": 1, "hid": 4},
        torch.randn(2, 4),
        torch.randn(2, 4),
        torch.ones(4),
        1e-6,
        2048,
        None,
        False,
        False,
        True,
        registry=registry,
        baseline_fn=_fusion_baseline,
    )
    assert result[0] == "stock"
    assert completed == fallbacks == []


def test_deep_failed_recovery_does_not_claim_stock_was_served(events, monkeypatch):
    completed, fallbacks = events
    monkeypatch.setattr(
        dispatch, "_arfusion_group", lambda _use_attn: SimpleNamespace(size=lambda: 2)
    )
    monkeypatch.setattr(
        dispatch._moe_export,
        "export_views",
        lambda _exp, _device: (
            torch.randn(2, 4),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2),
        ),
    )
    monkeypatch.setattr(
        dispatch._moe_export,
        "trusted_finalize",
        lambda *_args: (_ for _ in ()).throw(ValueError("recovery failed")),
    )
    registry = _registry("collective.moe_finalize_ar_rmsnorm", _boom)
    with pytest.raises(ValueError, match="recovery failed"):
        dispatch._deep_consume(
            {"T": 2, "K": 1, "hid": 4},
            torch.randn(2, 4),
            torch.randn(2, 4),
            torch.ones(4),
            1e-6,
            2048,
            None,
            False,
            False,
            True,
            registry=registry,
            baseline_fn=_fusion_baseline,
        )
    assert completed == fallbacks == []
