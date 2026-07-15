"""MSA live binding parity for typed outputs and capability-selected variants."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import optima.dispatch as dispatch  # noqa: E402
from optima.registry import (  # noqa: E402
    Eligibility,
    KernelImpl,
    KernelRegistry,
    eligibility_from_metadata,
)


class _CudaLikeQ:
    """CPU-backed q with the CUDA-shaped surface used by the MSA wrapper."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.is_cuda = True

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def device(self):
        return self.tensor.device

    def dim(self):
        return self.tensor.dim()

    def __getitem__(self, item):
        return self.tensor[item]


class _FakeTopKKernel:
    def __getitem__(self, _grid):
        def launch(_score, topk_idx, *_args, **_kwargs):
            topk_idx.zero_()

        return launch


def _msa_batched_args(*, topk=16):
    # Request 0 has fewer blocks than the batch max, so its logical score view
    # has a padded row pitch.
    return (
        _CudaLikeQ(torch.ones(3, 2, 2)),
        torch.ones(5, 1, 2),
        torch.ones(5, 1, 2),
        None,
        torch.tensor([[0, 1, 0], [2, 3, 4]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 2, 3]),
        torch.tensor([2, 3]),
        torch.tensor([0, 2]),
        2,
        3,
        1,
        1,
        topk,
    )


def _install_fake_runtime(monkeypatch):
    monkeypatch.setenv("OPTIMA_MSA_PREFILL_SEAM", "1")
    monkeypatch.setattr(dispatch, "_arch_tag", lambda *_args: "sm103")
    monkeypatch.setattr(dispatch, "_runtime_parallel_sizes", lambda: (4, 8))
    fake_triton = ModuleType("triton")
    fake_triton.set_allocator = lambda _allocator: None
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    return SimpleNamespace(
        robust_allocator=object(), _topk_index_kernel=_FakeTopKKernel()
    )


def _single_registry(entry):
    registry = KernelRegistry()
    registry.register(
        KernelImpl(
            slot="attention.msa_prefill_block_score",
            bundle_id="test",
            entry=entry,
            eligibility=Eligibility(dtypes=frozenset({"float32"})),
        )
    )
    registry.enable()
    return registry


class _RecordingRegistry(KernelRegistry):
    def __init__(self):
        super().__init__()
        self.decisions = []

    def select(self, slot, descriptor, **kwargs):
        self.decisions.append((descriptor, kwargs.get("write_fired_receipt", True)))
        return super().select(slot, descriptor, **kwargs)


def _variant_registry(entries):
    registry = _RecordingRegistry()
    for q_len, entry in entries.items():
        eligibility = eligibility_from_metadata(
            {
                "graph_safe": False,
                "capabilities": {
                    "dtype": "float32",
                    "architecture": "sm103",
                    "head_dim": 2,
                    "block_size": 1,
                    "q_len": q_len,
                    "phase": "prefill",
                    "layout": "row_major",
                    "graph_mode": "eager",
                    "quant": "dense",
                },
            },
            ("float32",),
            ("sm103",),
        )
        registry.register(
            KernelImpl(
                slot="attention.msa_prefill_block_score",
                bundle_id="test",
                variant=f"q{q_len}",
                entry=entry,
                eligibility=eligibility,
            )
        )
    registry.enable()
    return registry


def _invoke(wrapped, args=None, *, device="cpu"):
    return wrapped(
        *(args or _msa_batched_args()),
        disable_index_value=True,
        cu_seqblocks_q=torch.tensor([0, 2, 3], device=device),
        max_seqblock_q=2,
        all_seqblock_q=3,
    )


def test_msa_live_binding_uses_typed_strided_score_view(monkeypatch):
    module = _install_fake_runtime(monkeypatch)
    observed = []

    def entry(_q, _k, _prefix, _scale, _block_size, out):
        observed.append((out.dtype, out.is_contiguous(), tuple(out.shape), out.stride()))
        out.fill_(1.0)

    wrapped = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: "stock",
        module,
        registry=_single_registry(entry),
    )
    result = _invoke(wrapped)

    assert result[0] is None
    assert observed
    dtype, contiguous, shape, stride = observed[0]
    assert dtype == torch.float32
    assert shape == (2, 2)
    assert not contiguous
    assert stride == (3, 1)


def test_msa_preflights_each_request_head_and_routes_variants(monkeypatch):
    module = _install_fake_runtime(monkeypatch)
    calls = {1: 0, 2: 0}
    completed = []
    fallbacks = []
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)
    monkeypatch.setattr(
        dispatch._receipts,
        "fallback",
        lambda slot, exc: fallbacks.append((slot, type(exc).__name__)),
    )

    def entry_for(q_len):
        def entry(q, _k, _prefix, _scale, _block_size, out):
            assert q.shape[0] == q_len
            calls[q_len] += 1
            out.fill_(float(q_len))

        return entry

    registry = _variant_registry({1: entry_for(1), 2: entry_for(2)})
    stock_calls = 0

    def stock(*_args, **_kwargs):
        nonlocal stock_calls
        stock_calls += 1
        return "stock"

    result = _invoke(
        dispatch.make_msa_prefill_dispatcher(stock, module, registry=registry)
    )

    assert result[0] is None
    assert stock_calls == 0
    assert calls == {1: 2, 2: 2}
    assert completed == ["attention.msa_prefill_block_score"]
    assert fallbacks == []
    preflight = [descriptor for descriptor, fired in registry.decisions if not fired]
    assert len(preflight) == 2
    assert {descriptor["q_len"] for descriptor in preflight} == {1, 2}
    assert {descriptor["kv_len"] for descriptor in preflight} == {2, 3}
    for descriptor in preflight:
        assert descriptor.as_dict().items() >= {
            "dtype": "float32",
            "architecture": "sm103",
            "head_dim": 2,
            "block_size": 1,
            "top_k": 8,
            "phase": "prefill",
            "layout": "row_major",
            "graph_mode": "eager",
            "quant": "dense",
            "tp_size": 4,
            "world_size": 8,
            "num_q_heads": 1,
            "num_kv_heads": 1,
        }.items()


def test_msa_mixed_batch_off_domain_is_wholly_stock(monkeypatch):
    module = _install_fake_runtime(monkeypatch)
    candidate_calls = 0
    completed = []
    fallbacks = []
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)
    monkeypatch.setattr(
        dispatch._receipts,
        "fallback",
        lambda slot, exc: fallbacks.append((slot, type(exc).__name__)),
    )

    def q2_only(*_args):
        nonlocal candidate_calls
        candidate_calls += 1

    registry = _variant_registry({2: q2_only})
    stock_result = object()
    stock_calls = 0

    def stock(*_args, **_kwargs):
        nonlocal stock_calls
        stock_calls += 1
        return stock_result

    result = _invoke(
        dispatch.make_msa_prefill_dispatcher(stock, module, registry=registry)
    )

    assert result is stock_result
    assert stock_calls == 1
    assert candidate_calls == 0
    assert completed == fallbacks == []


def test_msa_selected_failure_receipts_only_after_stock_succeeds(monkeypatch):
    module = _install_fake_runtime(monkeypatch)
    completed = []
    fallbacks = []
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)
    monkeypatch.setattr(
        dispatch._receipts,
        "fallback",
        lambda slot, exc: fallbacks.append((slot, type(exc).__name__)),
    )

    def boom(*_args):
        raise RuntimeError("candidate path failed")

    registry = _variant_registry({1: boom, 2: boom})
    stock_result = object()
    wrapped = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: stock_result, module, registry=registry
    )
    assert _invoke(wrapped) is stock_result
    assert completed == []
    assert fallbacks == [("attention.msa_prefill_block_score", "RuntimeError")]

    fallbacks.clear()
    failing_stock = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("stock failed")),
        module,
        registry=registry,
    )
    with pytest.raises(ValueError, match="stock failed"):
        _invoke(failing_stock)
    assert completed == fallbacks == []


def test_msa_validator_topk_tail_failure_is_selected_path_fallback(monkeypatch):
    module = _install_fake_runtime(monkeypatch)
    completed = []
    fallbacks = []
    monkeypatch.setattr(dispatch._receipts, "completed", completed.append)
    monkeypatch.setattr(
        dispatch._receipts,
        "fallback",
        lambda slot, exc: fallbacks.append((slot, type(exc).__name__)),
    )

    class FailingTopK:
        def __getitem__(self, _grid):
            def launch(*_args, **_kwargs):
                raise RuntimeError("validator top-k tail failed")

            return launch

    module._topk_index_kernel = FailingTopK()

    def entry(_q, _k, _prefix, _scale, _block_size, out):
        out.fill_(1.0)

    stock_result = object()
    wrapped = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: stock_result,
        module,
        registry=_variant_registry({1: entry, 2: entry}),
    )
    assert _invoke(wrapped) is stock_result
    assert completed == []
    assert fallbacks == [("attention.msa_prefill_block_score", "RuntimeError")]


def test_msa_runtime_topk_does_not_gate_score_kernel(monkeypatch):
    module = _install_fake_runtime(monkeypatch)
    candidate_calls = 0

    def candidate(_q, _k, _prefix, _scale, _block_size, out):
        nonlocal candidate_calls
        candidate_calls += 1
        out.fill_(1.0)

    stock_result = object()
    stock_calls = 0

    def stock(*_args, **_kwargs):
        nonlocal stock_calls
        stock_calls += 1
        return stock_result

    registry = _variant_registry({1: candidate, 2: candidate})
    wrapped = dispatch.make_msa_prefill_dispatcher(
        stock,
        module,
        registry=registry,
    )
    result = _invoke(wrapped)

    assert result[0] is None
    assert result[1].shape[-1] == 16
    assert stock_calls == 0
    assert candidate_calls == 4
    descriptors = [descriptor for descriptor, _fired in registry.decisions]
    assert descriptors
    assert {descriptor["top_k"] for descriptor in descriptors} == {8}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_msa_live_binding_real_cuda_routes_typed_variants(monkeypatch, tmp_path):
    # Torch's first CUDA initialization probes the real Triton module. Complete
    # that before replacing only the binding's top-k launch surface with our fake.
    torch.empty((), device="cuda")
    module = _install_fake_runtime(monkeypatch)
    cpu_args = list(_msa_batched_args())
    cpu_args[0] = cpu_args[0].tensor.to("cuda")
    cuda_args = [
        value.to("cuda") if torch.is_tensor(value) else value
        for value in cpu_args
    ]
    observed = []
    receipt_dir = tmp_path / "receipts"
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", str(receipt_dir))
    monkeypatch.setattr(dispatch._receipts, "_ONCE", set())

    def entry_for(q_len):
        def entry(q, _k, _prefix, _scale, _block_size, out):
            observed.append(
                (q_len, out.device.type, out.dtype, tuple(out.shape), out.stride())
            )
            out.fill_(float(q_len))

        return entry

    registry = _variant_registry({1: entry_for(1), 2: entry_for(2)})
    wrapped = dispatch.make_msa_prefill_dispatcher(
        lambda *_a, **_k: "stock",
        module,
        registry=registry,
    )
    result = _invoke(wrapped, cuda_args, device="cuda")
    torch.cuda.synchronize()

    assert result[0] is None
    assert len(observed) == 4
    completed = dispatch._receipts.collect(receipt_dir, "completed")
    assert len(completed) == 1
    assert completed[0]["slot"] == "attention.msa_prefill_block_score"
    assert completed[0]["pid"] > 0
    assert dispatch._receipts.collect(receipt_dir, "fallback") == []
    assert {row[0] for row in observed} == {1, 2}
    assert all(row[1] == "cuda" and row[2] == torch.float32 for row in observed)
    assert any(shape == (2, 2) and stride == (3, 1)
               for _q, _device, _dtype, shape, stride in observed)
