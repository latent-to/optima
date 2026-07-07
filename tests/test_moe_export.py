"""Deep-seam export/consume runtime (optima/moe_export.py + the dispatcher's deep
consume branch + the defer-gate/moe-export integrations).

The correctness invariant under test everywhere: skip-finalize is only armed when
the deferred fusion call will certainly consume the export, and every exit from the
consume branch has PERFORMED the finalize — an unfinalized moe output never reaches
the shallow kernel, the stock baseline, or the caller.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from optima import moe_export
from optima.dispatch import make_arfusion_dispatcher
from optima.registry import Eligibility, KernelImpl, KernelRegistry

DEEP = moe_export.DEEP_SLOT
SHALLOW = "collective.ar_residual_rmsnorm"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    moe_export.reset()
    monkeypatch.delenv("OPTIMA_ARFUSION_SEAM", raising=False)
    monkeypatch.delenv("OPTIMA_SLOT_AUDIT", raising=False)
    monkeypatch.delenv("OPTIMA_SEAM_RECEIPT_DIR", raising=False)
    yield
    moe_export.reset()


# ---- defer-gate state machine --------------------------------------------------


def test_will_defer_requires_fuse_and_no_reduce_scatter():
    moe_export.record_fuse_decision(True)
    moe_export.record_rs_decision(False)
    assert moe_export._consume_will_defer()

    moe_export.record_fuse_decision(True)
    moe_export.record_rs_decision(True)  # reduce-scatter layer: immediate AR
    assert not moe_export._consume_will_defer()


def test_will_defer_is_one_shot():
    moe_export.record_fuse_decision(True)
    assert moe_export._consume_will_defer()
    assert not moe_export._consume_will_defer()  # consumed; must not leak forward


def test_layer_boundary_clears_decisions():
    # A layer that never reaches the wrapped moe call (dense/ineligible) must not
    # leak its True into the NEXT layer's moe call.
    moe_export.record_fuse_decision(True)
    moe_export.on_layer_mlp_boundary()
    assert not moe_export._consume_will_defer()


class _Batch:  # weakref-able stand-in for ForwardBatch
    pass


def test_forward_boundary_drops_stale_pends():
    b1, b2 = _Batch(), _Batch()
    moe_export.on_forward_boundary(b1)
    moe_export._state["pends"][0xdead] = {"T": 4, "K": 5}
    moe_export.on_forward_boundary(b1)  # same forward: pends live
    assert moe_export._state["pends"]
    moe_export.on_forward_boundary(b2)  # NEW forward: abandoned pend dropped
    assert not moe_export._state["pends"]
    assert moe_export._state["stale_dropped"] == 1


def test_forward_boundary_dead_ref_counts_as_new_forward():
    moe_export.on_forward_boundary(_Batch())  # dies immediately -> dead weakref
    moe_export._state["pends"][0xbeef] = {"T": 4, "K": 5}
    moe_export.on_forward_boundary(_Batch())
    assert not moe_export._state["pends"]


# ---- export wrap (maybe_export) -------------------------------------------------


class _FakeRaw:
    """Export-ABI stand-in: records skip toggles, serves a scripted export."""

    def __init__(self, export=None):
        self.skip = []
        self.export = export

    def fe_set_skip_finalize(self, v):
        self.skip.append(bool(v))

    def fe_get_export(self):
        return self.export


def _deep_registry(**elig):
    reg = KernelRegistry()
    reg.register(KernelImpl(
        slot=DEEP, bundle_id="t", entry=lambda *a: None,
        eligibility=Eligibility(dtypes=frozenset({"float32"}), **elig)))
    reg.enable()
    return reg


def _armed(monkeypatch, raw, *, n_layers=4, ordinal=1):
    monkeypatch.setattr(moe_export, "_input_ok",
                        lambda inp: torch.is_tensor(inp) and inp.dim() == 2)
    moe_export._state["raw"] = raw
    # A non-last layer of a known-depth model (the last-layer veto needs both:
    # ordinal % n_layers == 0 means no successor prepare_attn will consume).
    moe_export._state["n_layers"] = n_layers
    moe_export._state["ordinal"] = ordinal
    moe_export.record_fuse_decision(True)


def _orig_recorder(calls, out):
    def orig(*args, **kwargs):
        calls.append((args, kwargs))
        return out
    return orig


def test_export_arms_skip_and_pends_by_output_ptr(monkeypatch):
    out = torch.zeros(64, 32)
    raw = _FakeRaw(export=(1, 111, 222, 333, 64, 32, 32, 5, 16))
    _armed(monkeypatch, raw)
    calls = []
    ret = moe_export.maybe_export(_orig_recorder(calls, out),
                                  (None, torch.zeros(64, 32)), {},
                                  registry=_deep_registry())
    assert ret is out and len(calls) == 1
    assert raw.skip == [True, False]  # armed strictly around the call
    pend = moe_export._state["pends"][out.data_ptr()]
    assert pend == {"g2": 111, "idx": 222, "scl": 333, "T": 64, "K": 5, "hid": 32}
    assert moe_export._state["exports"] == 1


def test_no_defer_means_stock_and_no_skip(monkeypatch):
    raw = _FakeRaw()
    monkeypatch.setattr(moe_export, "_input_ok", lambda inp: True)
    moe_export._state["raw"] = raw
    calls = []
    moe_export.maybe_export(_orig_recorder(calls, torch.zeros(2)),
                            (None, torch.zeros(64, 32)), {},
                            registry=_deep_registry())
    assert len(calls) == 1 and raw.skip == []


def test_defer_decision_consumed_even_when_gates_fail(monkeypatch):
    # The decision belongs to THIS call: a gate-failed call must still consume it,
    # or a later unrelated moe call would inherit a stale True.
    _armed(monkeypatch, _FakeRaw())
    calls = []
    moe_export.maybe_export(_orig_recorder(calls, torch.zeros(2)),
                            (None, torch.zeros(64, 32)), {"ep_size": 2},
                            registry=_deep_registry())
    assert len(calls) == 1
    assert not moe_export._consume_will_defer()


def test_min_num_tokens_gates_the_export_side(monkeypatch):
    # Below the deep kernel's measured floor the finalize must stay in-op.
    raw = _FakeRaw(export=(1, 1, 2, 3, 8, 32, 32, 5, 16))
    _armed(monkeypatch, raw)
    calls = []
    moe_export.maybe_export(_orig_recorder(calls, torch.zeros(8, 32)),
                            (None, torch.zeros(8, 32)), {},
                            registry=_deep_registry(min_num_tokens=48))
    assert len(calls) == 1 and raw.skip == [] and not moe_export._state["pends"]


def test_unchanged_seq_means_finalize_fused_tactic_no_pend(monkeypatch):
    raw = _FakeRaw(export=(0, 1, 2, 3, 64, 32, 32, 5, 16))  # seq stays 0
    _armed(monkeypatch, raw)
    out = torch.zeros(64, 32)
    moe_export.maybe_export(_orig_recorder([], out), (None, torch.zeros(64, 32)), {},
                            registry=_deep_registry())
    assert raw.skip == [True, False] and not moe_export._state["pends"]


def test_export_abi_mismatch_raises(monkeypatch):
    # rows != T: the output is unfinalized and unrecoverable — refuse loudly.
    raw = _FakeRaw(export=(1, 1, 2, 3, 63, 32, 32, 5, 16))
    _armed(monkeypatch, raw)
    with pytest.raises(RuntimeError, match="export ABI mismatch"):
        moe_export.maybe_export(_orig_recorder([], torch.zeros(64, 32)),
                                (None, torch.zeros(64, 32)), {},
                                registry=_deep_registry())


def test_output_ptr_reuse_within_forward_raises(monkeypatch):
    out = torch.zeros(64, 32)
    raw = _FakeRaw(export=(1, 1, 2, 3, 64, 32, 32, 5, 16))
    _armed(monkeypatch, raw)
    reg = _deep_registry()
    moe_export.maybe_export(_orig_recorder([], out), (None, torch.zeros(64, 32)), {},
                            registry=reg)
    raw.export = (2, 1, 2, 3, 64, 32, 32, 5, 16)
    moe_export.record_fuse_decision(True)
    with pytest.raises(RuntimeError, match="reused before its export was consumed"):
        moe_export.maybe_export(_orig_recorder([], out), (None, torch.zeros(64, 32)),
                                {}, registry=reg)


def test_missing_export_abi_disables_permanently(monkeypatch):
    _armed(monkeypatch, None)
    moe_export._state["raw"] = None

    def no_abi():
        raise AttributeError("fe_set_skip_finalize missing (stock build)")
    monkeypatch.setattr(moe_export, "_raw_module", no_abi)
    calls = []
    moe_export.maybe_export(_orig_recorder(calls, torch.zeros(2)),
                            (None, torch.zeros(64, 32)), {},
                            registry=_deep_registry())
    assert len(calls) == 1 and moe_export._state["disabled"]
    # disabled short-circuits every later call
    moe_export.record_fuse_decision(True)
    moe_export.maybe_export(_orig_recorder(calls, torch.zeros(2)),
                            (None, torch.zeros(64, 32)), {},
                            registry=_deep_registry())
    assert len(calls) == 2


def test_tuning_mode_never_arms(monkeypatch):
    # Toggling skip-finalize during autotuner profiling poisons the tactic table.
    raw = _FakeRaw(export=(1, 1, 2, 3, 64, 32, 32, 5, 16))
    _armed(monkeypatch, raw)
    monkeypatch.setattr(moe_export, "_tuning", lambda: True)
    moe_export.maybe_export(_orig_recorder([], torch.zeros(64, 32)),
                            (None, torch.zeros(64, 32)), {},
                            registry=_deep_registry())
    assert raw.skip == []


def test_last_layer_never_arms(monkeypatch):
    # The 2026-07-07 capture crash: minimax_m3 leaves is_last_layer unset, so
    # sglang lets the FINAL layer defer — but no successor prepare_attn consumes
    # the export (the late AR lands on a transformed tensor). ordinal % n == 0
    # must keep the finalize in-op.
    raw = _FakeRaw(export=(1, 1, 2, 3, 64, 32, 32, 5, 16))
    _armed(monkeypatch, raw, n_layers=4, ordinal=4)
    calls = []
    moe_export.maybe_export(_orig_recorder(calls, torch.zeros(64, 32)),
                            (None, torch.zeros(64, 32)), {},
                            registry=_deep_registry())
    assert len(calls) == 1 and raw.skip == [] and not moe_export._state["pends"]
    assert moe_export._state["last_layer_vetoes"] == 1
    # veto consumed the defer decision — no leak into the next call
    assert not moe_export._consume_will_defer()


def test_last_layer_veto_is_modulo_for_capture_multipass(monkeypatch):
    # CUDA-graph capture replays the SAME ForwardBatch (2 warmups + record), so
    # the ordinal climbs across passes without a forward boundary: layer 4 of a
    # 4-layer model appears as ordinal 4, 8, 12... — every one must veto, while
    # mid-stack ordinals in later passes still arm.
    raw = _FakeRaw(export=(1, 1, 2, 3, 64, 32, 32, 5, 16))
    _armed(monkeypatch, raw, n_layers=4, ordinal=8)  # pass 2's last layer
    moe_export.maybe_export(_orig_recorder([], torch.zeros(64, 32)),
                            (None, torch.zeros(64, 32)), {},
                            registry=_deep_registry())
    assert raw.skip == [] and moe_export._state["last_layer_vetoes"] == 1

    out = torch.zeros(64, 32)
    moe_export._state["ordinal"] = 10  # pass 3, layer 2: mid-stack, safe
    moe_export.record_fuse_decision(True)
    moe_export.maybe_export(_orig_recorder([], out),
                            (None, torch.zeros(64, 32)), {},
                            registry=_deep_registry())
    assert raw.skip == [True, False] and out.data_ptr() in moe_export._state["pends"]


def test_unresolvable_layer_count_disables(monkeypatch):
    # No config to veto against -> the deep seam must refuse to arm AT ALL
    # (fail-closed), not guess.
    raw = _FakeRaw(export=(1, 1, 2, 3, 64, 32, 32, 5, 16))
    _armed(monkeypatch, raw, n_layers=0)  # unresolved
    monkeypatch.setattr(moe_export, "_num_layers", lambda: None)
    calls = []
    moe_export.maybe_export(_orig_recorder(calls, torch.zeros(64, 32)),
                            (None, torch.zeros(64, 32)), {},
                            registry=_deep_registry())
    assert len(calls) == 1 and raw.skip == [] and moe_export._state["disabled"]


def test_forward_boundary_resets_ordinal():
    moe_export._state["ordinal"] = 7
    moe_export.on_forward_boundary(_Batch())
    assert moe_export._state["ordinal"] == 0
    moe_export.on_layer_mlp_boundary()
    assert moe_export._state["ordinal"] == 1


def _stub_server_args(monkeypatch, model_path, spec_name="NONE"):
    sa = types.ModuleType("sglang.srt.server_args")
    spec = SimpleNamespace(name=spec_name) if spec_name else None
    sa.get_global_server_args = lambda: SimpleNamespace(
        pp_size=1, speculative_algorithm=spec,
        model_path=str(model_path), trust_remote_code=True)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", sa)


def test_num_layers_reads_nested_dict_text_config(monkeypatch, tmp_path):
    # The two 07-07 launch-vacuous failure shapes pinned: (a) custom-config
    # classes keep text_config as a PLAIN DICT (getattr misses it), (b) sglang's
    # speculative_algorithm NONE is a truthy enum. Both must resolve, not disable.
    import json
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "vl", "text_config": {"num_hidden_layers": 60}}))
    _stub_server_args(monkeypatch, tmp_path, spec_name="NONE")
    assert moe_export._num_layers() == 60
    assert moe_export._state["n_layers"] == 60


def test_num_layers_disables_under_real_speculative(monkeypatch, tmp_path):
    import json
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 60}))
    _stub_server_args(monkeypatch, tmp_path, spec_name="EAGLE3")
    assert moe_export._num_layers() is None


# ---- dispatcher deep consume branch ---------------------------------------------


def _pend(t=64, k=5, hid=32):
    return {"g2": 1, "idx": 2, "scl": 3, "T": t, "K": k, "hid": hid}


def _views_for(pend, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = pend["T"] * pend["K"]
    gemm = torch.randn(rows, pend["hid"], generator=g) * 0.1
    row_map = torch.randperm(rows, generator=g).to(torch.int32)
    scales = (torch.rand(pend["T"], pend["K"], generator=g) + 0.1) / pend["K"]
    return gemm, row_map, scales


def _stub_views(monkeypatch, views):
    monkeypatch.setattr(moe_export, "export_views", lambda exp, device: views)


def _deep_entry_recorder(record):
    def entry(gemm_out, row_map, scales, residual, weight, eps, out_norm,
              out_residual, group):
        record.append((gemm_out.shape, residual.shape))
        out_norm.fill_(1.0)
        out_residual.fill_(2.0)
    return entry


def _dual_registry(record, **deep_elig):
    reg = KernelRegistry()
    reg.register(KernelImpl(slot=DEEP, bundle_id="t",
                            entry=_deep_entry_recorder(record),
                            eligibility=Eligibility(dtypes=frozenset({"float32"}),
                                                    **deep_elig)))
    reg.enable()
    return reg


def _baseline_recorder(calls):
    def baseline(input_tensor, residual, weight, eps=1e-6, max_token_num=2048,
                 use_oneshot=None, trigger_completion_at_end=False, fp32_acc=False,
                 use_attn_tp_group=True):
        calls.append(input_tensor)
        return ("baseline", input_tensor)
    return baseline


@pytest.fixture()
def _fake_group(monkeypatch):
    import optima.dispatch as dispatch

    monkeypatch.setattr(dispatch, "_arfusion_group",
                        lambda use_attn: SimpleNamespace(size=lambda: 2))


def test_consume_routes_pend_to_deep_kernel(monkeypatch, _fake_group):
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    pend = _pend()
    _stub_views(monkeypatch, _views_for(pend))
    record, base_calls = [], []
    d = make_arfusion_dispatcher(_baseline_recorder(base_calls),
                                 registry=_dual_registry(record))
    x = torch.zeros(64, 32)
    moe_export._state["pends"][x.data_ptr()] = pend
    out_norm, out_residual = d(x, x.clone(), torch.ones(32))
    assert record and base_calls == []
    assert torch.equal(out_norm, torch.full((64, 32), 1.0))
    assert torch.equal(out_residual, torch.full((64, 32), 2.0))
    assert not moe_export._state["pends"]  # popped exactly once


def test_consume_head_trims_graph_padding(monkeypatch, _fake_group):
    # T_consume < T_export (CUDA-graph batch padding): full views, trimmed residual.
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    pend = _pend(t=64)
    _stub_views(monkeypatch, _views_for(pend))
    record = []
    d = make_arfusion_dispatcher(_baseline_recorder([]),
                                 registry=_dual_registry(record))
    x = torch.zeros(48, 32)
    moe_export._state["pends"][x.data_ptr()] = pend
    d(x, x.clone(), torch.ones(32))
    assert record == [((320, 32), (48, 32))]  # 64*5 rows kept, residual trimmed


def test_consume_more_rows_than_export_raises(monkeypatch, _fake_group):
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    d = make_arfusion_dispatcher(_baseline_recorder([]),
                                 registry=_dual_registry([]))
    x = torch.zeros(65, 32)
    moe_export._state["pends"][x.data_ptr()] = _pend(t=64)
    with pytest.raises(RuntimeError, match="pointer pairing broken"):
        d(x, x.clone(), torch.ones(32))


def test_orphan_pend_reconstructs_before_stock(monkeypatch, _fake_group):
    # Deep kernel ineligible at consume time -> the TRUSTED finalize must run and
    # the stock fusion call must receive the FINALIZED tensor, never the raw input.
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    pend = _pend()
    views = _views_for(pend)
    _stub_views(monkeypatch, views)
    base_calls = []
    reg = KernelRegistry()  # no deep kernel registered
    reg.enable()
    d = make_arfusion_dispatcher(_baseline_recorder(base_calls), registry=reg)
    x = torch.zeros(64, 32)
    moe_export._state["pends"][x.data_ptr()] = pend
    out = d(x, x.clone(), torch.ones(32))
    assert out[0] == "baseline"
    gemm, row_map, scales = views
    per_k = gemm.float()[row_map.long().view(pend["K"], pend["T"])]
    expect = (per_k * scales.float().t().unsqueeze(-1)).sum(dim=0)
    assert torch.allclose(base_calls[0], expect, atol=1e-5)
    assert moe_export._state["orphans"] == 1


def test_deep_kernel_error_falls_back_to_reconstruct(monkeypatch, _fake_group):
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    pend = _pend()
    _stub_views(monkeypatch, _views_for(pend))

    def boom(*a):
        raise RuntimeError("deep kernel exploded")

    reg = KernelRegistry()
    reg.register(KernelImpl(slot=DEEP, bundle_id="t", entry=boom,
                            eligibility=Eligibility(dtypes=frozenset({"float32"}))))
    reg.enable()
    base_calls = []
    d = make_arfusion_dispatcher(_baseline_recorder(base_calls), registry=reg)
    x = torch.zeros(64, 32)
    moe_export._state["pends"][x.data_ptr()] = pend
    assert d(x, x.clone(), torch.ones(32))[0] == "baseline"
    assert len(base_calls) == 1  # reconstruct path, not a bare fallthrough

    reg.set_strict(True)
    moe_export._state["pends"][x.data_ptr()] = _pend()
    with pytest.raises(RuntimeError, match="deep kernel exploded"):
        d(x, x.clone(), torch.ones(32))


def test_plain_calls_untouched_when_no_pend(monkeypatch, _fake_group):
    # A fusion call whose input is NOT a pended moe output takes the shallow path.
    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    base_calls = []
    reg = KernelRegistry()
    reg.enable()
    d = make_arfusion_dispatcher(_baseline_recorder(base_calls), registry=reg)
    x = torch.zeros(64, 32)
    moe_export._state["pends"][0x123456] = _pend()  # some OTHER buffer's pend
    assert d(x, x.clone(), torch.ones(32))[0] == "baseline"
    assert moe_export._state["pends"]  # untouched


# ---- registry additions ----------------------------------------------------------


def test_peek_matches_lookup_without_fired_receipt(tmp_path, monkeypatch):
    from optima import registry as registry_mod

    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(tmp_path))
    monkeypatch.setattr(registry_mod, "_FIRED_SLOTS", set())
    reg = _deep_registry(min_num_tokens=48, max_num_tokens=1024)
    kw = dict(dtype_name="float32", last_dim=32, arch=None)
    assert reg.peek(DEEP, num_tokens=64, **kw) is not None
    assert reg.peek(DEEP, num_tokens=4, **kw) is None
    assert reg.peek(DEEP, num_tokens=2048, **kw) is None
    assert not list(tmp_path.glob("fired*")), "peek must not write 'fired'"
    assert reg.lookup(DEEP, num_tokens=64, **kw) is not None
    assert list(tmp_path.glob("fired*")), "lookup still writes 'fired'"


def test_eligibility_min_num_tokens_from_metadata():
    from optima.registry import eligibility_from_metadata

    e = eligibility_from_metadata({"min_num_tokens": 48}, ("bfloat16",))
    assert e.min_num_tokens == 48
    assert not e.accepts(dtype_name="bfloat16", last_dim=64, arch=None, num_tokens=4)
    assert e.accepts(dtype_name="bfloat16", last_dim=64, arch=None, num_tokens=48)


# ---- integrations (stub sglang modules) ------------------------------------------


def _stub_communicator(monkeypatch):
    mod = types.ModuleType("sglang.srt.layers.communicator")

    class LayerCommunicator:
        def prepare_attn(self, hidden_states, residual, forward_batch):
            return "attn"

        def prepare_mlp(self, hidden_states, residual, forward_batch):
            return "mlp"

        def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
            return True

        def should_use_reduce_scatter(self, forward_batch):
            return False

    mod.LayerCommunicator = LayerCommunicator
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.communicator", mod)
    return mod


def test_defer_gate_install_records_decisions(monkeypatch):
    from optima.integrations import sglang_defer_gate

    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    mod = _stub_communicator(monkeypatch)
    reg = _deep_registry()
    sglang_defer_gate.install(reg)
    assert getattr(mod, "_optima_defer_gate_patched")

    comm, batch = mod.LayerCommunicator(), _Batch()
    moe_export._state["pends"][0xabc] = _pend()
    assert comm.prepare_attn(None, None, batch) == "attn"
    assert not moe_export._state["pends"]  # new forward -> stale pend dropped
    comm.prepare_mlp(None, None, batch)
    comm.should_fuse_mlp_allreduce_with_next_layer(batch)
    comm.should_use_reduce_scatter(batch)
    assert moe_export._consume_will_defer()


def test_defer_gate_install_is_retryable_until_bundle_loads(monkeypatch):
    from optima.integrations import sglang_defer_gate

    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    mod = _stub_communicator(monkeypatch)
    empty = KernelRegistry()
    sglang_defer_gate.install(empty)  # activate() pass before the bundle loaded
    assert not getattr(mod, "_optima_defer_gate_patched", False)
    sglang_defer_gate.install(_deep_registry())  # later pass: bundle present
    assert getattr(mod, "_optima_defer_gate_patched")


def test_moe_export_install_rebinds_module_function(monkeypatch):
    from optima.integrations import sglang_moe_export

    monkeypatch.setenv("OPTIMA_ARFUSION_SEAM", "1")
    mod = types.ModuleType("sglang.srt.layers.quantization.modelopt_quant")
    orig_calls = []

    def flashinfer_cutlass_fused_moe(*a, **kw):
        orig_calls.append(1)
        return "stock"

    mod.flashinfer_cutlass_fused_moe = flashinfer_cutlass_fused_moe
    monkeypatch.setitem(sys.modules,
                        "sglang.srt.layers.quantization.modelopt_quant", mod)
    sglang_moe_export.install(_deep_registry())
    assert mod.flashinfer_cutlass_fused_moe is not flashinfer_cutlass_fused_moe
    # no defer decision recorded -> pure stock passthrough
    assert mod.flashinfer_cutlass_fused_moe(None, torch.zeros(4, 8)) == "stock"
    assert orig_calls == [1]
    sglang_moe_export.uninstall()
    assert mod.flashinfer_cutlass_fused_moe is flashinfer_cutlass_fused_moe


# ---- the deep bundle artifact -----------------------------------------------------


def test_deep_bundle_manifest_shape():
    # Pins the deep bundle's load-bearing structure: BOTH epilogue slots declared on
    # ONE source module (they share the IPC workspace in module globals — the seam
    # loader guarantees one module instance per source file), the fe_export dep
    # patch declared, and the deep op carrying the measured min_num_tokens floor.
    import json
    from pathlib import Path

    from optima.manifest import load_manifest

    bundle = Path("experiments/minimax_m3/bundle/miner_m3_fused_epilogue_deep")
    m = load_manifest(bundle)
    assert m.bundle_id == "m3-fused-epilogue-deep"
    slots = {op.slot: op for op in m.ops}
    assert set(slots) == {SHALLOW, DEEP}
    assert slots[SHALLOW].source == slots[DEEP].source  # shared module = shared IPC
    assert [dp.target for dp in m.dep_patches] == ["flashinfer"]
    meta = json.loads((bundle / slots[DEEP].metadata).read_text())
    assert meta["min_num_tokens"] == 48 and meta["max_num_tokens"] == 1024
    assert meta["graph_safe"] is True
