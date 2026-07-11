from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from optima import receipts, seam
from optima.eval import _launch


def _active(pid: int, rank: int, slots=("a", "b"), world_size=2):
    return {
        "pid": pid,
        "rank": rank,
        "world_size": world_size,
        "slots": list(slots),
    }


def _write(root, kind, payload, index):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{kind}.{index}.json").write_text(json.dumps(payload))


def _distributed_receipt_worker(rank, world_size, store_path, receipt_dir):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        os.environ["OPTIMA_SEAM_RECEIPT_DIR"] = receipt_dir
        receipts.write("active", {"slots": ["slot.a"]})
        receipts.completed("slot.a")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_active_members_require_exact_count_and_identical_slot_set():
    active = [_active(10, 0), _active(11, 1)]
    assert _launch._active_execution_members(
        active, expected_member_count=2
    ) == ["a", "b"]

    with pytest.raises(RuntimeError, match="1/2"):
        _launch._active_execution_members(active[:1], expected_member_count=2)
    with pytest.raises(RuntimeError, match="3/2"):
        _launch._active_execution_members(
            [*active, _active(12, 2, world_size=3)], expected_member_count=2
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        _launch._active_execution_members(
            [active[0], {**active[1], "pid": 10}], expected_member_count=2
        )
    with pytest.raises(RuntimeError, match="disagree"):
        _launch._active_execution_members(
            [active[0], _active(11, 1, slots=("a",))], expected_member_count=2
        )


def test_fired_without_completed_fails_execution_gate(tmp_path):
    active = [_active(10, 0, slots=("a",), world_size=1)]
    _write(
        tmp_path,
        "fired",
        {"slot": "a", "pid": 10, "rank": 0, "world_size": 1},
        0,
    )
    with pytest.raises(RuntimeError, match="failed execution coverage"):
        _launch._require_execution_completion(
            str(tmp_path),
            active_receipts=active,
            expected_slots=["a"],
            expected_member_count=1,
        )


def test_every_member_must_complete_every_slot(tmp_path):
    active = [_active(10, 0), _active(11, 1)]
    index = 0
    for rank, pid in enumerate((10, 11)):
        for slot in ("a", "b"):
            _write(
                tmp_path,
                "completed",
                {"slot": slot, "pid": pid, "rank": rank, "world_size": 2},
                index,
            )
            index += 1
    detail = _launch._require_execution_completion(
        str(tmp_path),
        active_receipts=active,
        expected_slots=["a", "b"],
        expected_member_count=2,
    )
    assert "4/4" in detail


def test_any_selected_path_fallback_disqualifies(tmp_path):
    active = [_active(10, 0, slots=("a",), world_size=1)]
    complete = {"slot": "a", "pid": 10, "rank": 0, "world_size": 1}
    _write(tmp_path, "completed", complete, 0)
    _write(tmp_path, "fallback", {**complete, "error_type": "RuntimeError"}, 0)
    with pytest.raises(RuntimeError, match="selected-path fallbacks"):
        _launch._require_execution_completion(
            str(tmp_path),
            active_receipts=active,
            expected_slots=["a"],
            expected_member_count=1,
        )


@pytest.mark.parametrize("active", (False, True))
def test_launched_engine_calls_completion_gate_only_for_active_run(monkeypatch, active):
    calls = []
    receipt_dirs = []

    class Engine:
        def __init__(self, **_kwargs):
            receipt_dirs.append(os.environ.get("OPTIMA_SEAM_RECEIPT_DIR"))

        def shutdown(self):
            pass

    cfg = SimpleNamespace(
        model_path="model",
        dtype="float32",
        mem_fraction_static=0.1,
        seed=0,
        log_level="error",
        framework_mode=False,
        tp_size=1,
    )
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    monkeypatch.setenv("OPTIMA_SEAM_RECEIPT_DIR", "ambient")
    monkeypatch.setattr(seam, "mark_driver", lambda: None)
    monkeypatch.setattr(_launch, "prepare_candidate_environment", lambda *_a, **_k: None)
    monkeypatch.setattr(_launch, "_wait_gpu_drain", lambda: None)
    monkeypatch.setattr(
        receipts,
        "require",
        lambda *_a, **_k: [_active(10, 0, slots=("a",), world_size=1)],
    )
    monkeypatch.setattr(
        _launch,
        "_require_execution_completion",
        lambda *_a, **_k: calls.append("completed") or "ok",
    )

    with _launch.launched_engine(
        cfg, bundle_path="bundle" if active else "", active=active
    ):
        pass

    assert calls == (["completed"] if active else [])
    assert bool(receipt_dirs[0]) is active
    assert os.environ["OPTIMA_SEAM_RECEIPT_DIR"] == "ambient"


def test_launcher_uses_final_resolved_candidate_tp_size(monkeypatch):
    observed_counts = []

    class Engine:
        def __init__(self, **_kwargs):
            pass

        def shutdown(self):
            pass

    active = [
        _active(10, 0, slots=("a",)),
        _active(11, 1, slots=("a",)),
    ]
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    monkeypatch.setattr(seam, "mark_driver", lambda: None)
    monkeypatch.setattr(_launch, "prepare_candidate_environment", lambda *_a, **_k: None)
    monkeypatch.setattr(_launch, "_wait_gpu_drain", lambda: None)
    monkeypatch.setattr(_launch, "engine_kwargs", lambda *_a, **_k: {"tp_size": 2})
    monkeypatch.setattr(receipts, "require", lambda *_a, **_k: active)

    def completed_gate(*_args, **kwargs):
        observed_counts.append(kwargs["expected_member_count"])
        return "ok"

    monkeypatch.setattr(_launch, "_require_execution_completion", completed_gate)
    with _launch.launched_engine(
        SimpleNamespace(framework_mode=False), bundle_path="bundle", active=True
    ):
        pass

    assert observed_counts == [2]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs",
)
def test_real_distributed_receipts_cover_every_nccl_member(tmp_path):
    import torch.multiprocessing as mp

    receipt_dir = tmp_path / "receipts"
    store_path = tmp_path / "nccl_store"
    mp.spawn(
        _distributed_receipt_worker,
        args=(2, str(store_path), str(receipt_dir)),
        nprocs=2,
        join=True,
    )
    active = receipts.collect(receipt_dir, "active")
    completed = receipts.collect(receipt_dir, "completed")
    assert {row["rank"] for row in active} == {0, 1}
    assert len({row["pid"] for row in active}) == 2
    ok, detail = receipts.completed_gate(
        completed,
        expected_slots=("slot.a",),
        member_receipts=active,
        expected_member_count=2,
    )
    assert ok, detail
