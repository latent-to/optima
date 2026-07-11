"""Real multi-rank CUDA-graph tests; CPU-only CI skips this module."""

from __future__ import annotations

import os
import time
import tempfile

import pytest

torch = pytest.importorskip("torch")

from optima.slots import get_slot  # noqa: E402
from optima.verify_collective import verify_collective  # noqa: E402


CUDA2 = torch.cuda.is_available() and torch.cuda.device_count() >= 2
pytestmark = pytest.mark.skipif(CUDA2 is False, reason="requires at least two CUDA GPUs")

ALLREDUCE = "examples/miner_allreduce_torch/kernels/all_reduce.py"
SHAPES = [{"num_tokens": 4, "hidden": 64}]


def _live_dispatch_graph_worker(rank, world_size, store_path):
    import torch.distributed as dist
    import traceback

    from optima.dispatch import make_allreduce_dispatcher
    import optima.dispatch as dispatch
    from optima.registry import Eligibility, KernelImpl, KernelRegistry

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        os.environ["OPTIMA_COLLECTIVE_SEAM"] = "1"

        # Use the pinned runtime's real coordinator and get_tp_group authority;
        # monkeypatching Optima's role classifier would make this only an NCCL
        # kernel test, not proof of the live Sglang adapter boundary.
        from sglang.srt.distributed import parallel_state as ps

        coordinator = ps.GroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            torch_distributed_backend="nccl",
            use_pynccl=False,
            use_pymscclpp=False,
            use_custom_allreduce=False,
            use_torch_symm_mem_all_reduce=False,
            use_hpu_communicator=False,
            use_xpu_communicator=False,
            use_npu_communicator=False,
            group_name="optima_live_test_tp",
        )
        ps._TP = coordinator
        group = coordinator.device_group
        assert dispatch._allreduce_group_role(coordinator, group) == "tp"

        def entry(x, out, group):
            out.copy_(x)
            dist.all_reduce(out, group=group)

        registry = KernelRegistry()
        registry.register(
            KernelImpl(
                slot="collective.all_reduce",
                bundle_id="live-cuda-test",
                entry=entry,
                eligibility=Eligibility(
                    dtypes=frozenset({"float32"}), graph_safe=True
                ),
            )
        )
        registry.enable()
        wrapped = make_allreduce_dispatcher(
            lambda _self, x: dist.all_reduce(x) or x,
            registry=registry,
        )
        x = torch.full((4, 64), float(rank + 1), device="cuda")

        # Warm the exact live adapter and NCCL call before capture.
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(2):
                eager = wrapped(coordinator, x)
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        assert torch.equal(eager, torch.full_like(eager, 3.0))

        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = wrapped(coordinator, x)
        # Capture records the collective; only replay is a gradeable execution.
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, torch.full_like(out, 3.0))

        x.fill_(float(rank + 2))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, torch.full_like(out, 5.0))
    except BaseException:  # noqa: BLE001 - durable parent exit code owns the test
        traceback.print_exc()
        os._exit(1)
    # Destroying a process group that owns a captured NCCL collective can hang in
    # PyTorch/NCCL finalization. These are disposable test workers; successful
    # process exit releases the CUDA context and the parent grades the exit code.
    os._exit(0)


def _verify(source=ALLREDUCE, *, slot="collective.all_reduce", entry="all_reduce",
            world_size=2, timeout_s=120.0, shapes=SHAPES):
    return verify_collective(
        get_slot(slot), str(source), entry, world_size=world_size,
        backend="nccl", device="cuda", shapes=shapes, graph_safe=True,
        graph_replays=3, timeout_s=timeout_s,
    )


def test_collective_nccl_cuda_graph_faithful_replays():
    world_size = int(os.environ.get("OPTIMA_COLLECTIVE_TEST_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA GPUs")
    result = _verify(world_size=world_size)

    assert result.passed, result.shape_results[0].detail
    assert result.graph_required
    assert result.graph_verified
    assert result.fully_verified
    assert result.shape_results[0].graph_replays == 3


def test_live_allreduce_binding_captures_and_replays_actual_nccl_group():
    import torch.multiprocessing as mp

    from optima.verify_collective import _terminate_processes

    pytest.importorskip("sglang.srt.distributed.parallel_state")

    with tempfile.NamedTemporaryFile(delete=False) as store:
        store_path = store.name
    os.unlink(store_path)
    try:
        context = mp.spawn(
            _live_dispatch_graph_worker,
            args=(2, store_path),
            nprocs=2,
            join=False,
        )
        deadline = time.monotonic() + 120.0
        joined = False
        while time.monotonic() < deadline:
            if context.join(timeout=1.0):
                joined = True
                break
        if not joined:
            _terminate_processes(list(context.processes), grace_s=2.0)
            pytest.fail("live all-reduce graph workers timed out after 120s")
    finally:
        try:
            os.unlink(store_path)
        except FileNotFoundError:
            pass


@pytest.mark.parametrize("partial_write", [False, True])
def test_collective_nccl_cuda_graph_replay_adversaries_fail(tmp_path, partial_write):
    source = tmp_path / "capture_bad.py"
    if partial_write:
        captured = (
            "        out[:1].copy_(x[:1])\n"
            "        dist.all_reduce(out[:1], op=dist.ReduceOp.SUM, group=group)\n"
        )
    else:
        captured = "        out.zero_()\n"
    source.write_text(
        "import torch\n"
        "import torch.distributed as dist\n\n"
        "def all_reduce(x, out, group=None):\n"
        "    if torch.cuda.is_current_stream_capturing():\n"
        + captured
        + "    else:\n"
        "        out.copy_(x)\n"
        "        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)\n"
    )
    result = _verify(source)

    assert not result.passed
    assert not result.graph_verified
    assert "cuda graph replay[0]" in result.shape_results[0].detail


def test_collective_nccl_cuda_graph_rejects_cached_correct_output(tmp_path):
    source = tmp_path / "cached_output.py"
    source.write_text(
        "import torch\n"
        "import torch.distributed as dist\n\n"
        "cached = None\n"
        "def all_reduce(x, out, group=None):\n"
        "    global cached\n"
        "    if torch.cuda.is_current_stream_capturing():\n"
        "        out.copy_(cached)\n"
        "    else:\n"
        "        out.copy_(x)\n"
        "        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)\n"
        "        cached = out.clone()\n"
    )

    result = _verify(source)

    assert not result.passed
    assert not result.graph_verified
    assert "cuda graph replay[0]" in result.shape_results[0].detail


def test_collective_nccl_cuda_graph_rejects_first_shape_capture_cache(tmp_path):
    source = tmp_path / "first_shape_only.py"
    source.write_text(
        "import torch\n"
        "import torch.distributed as dist\n\n"
        "captured_tokens = None\n"
        "def all_reduce(x, out, group=None):\n"
        "    global captured_tokens\n"
        "    if torch.cuda.is_current_stream_capturing():\n"
        "        if captured_tokens is None:\n"
        "            captured_tokens = x.shape[0]\n"
        "        if x.shape[0] != captured_tokens:\n"
        "            out.zero_()\n"
        "            return\n"
        "    out.copy_(x)\n"
        "    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)\n"
    )

    result = _verify(
        source,
        shapes=[
            {"num_tokens": 2, "hidden": 64},
            {"num_tokens": 7, "hidden": 64},
        ],
    )

    assert not result.passed
    assert not result.graph_verified
    assert all(row.passed for row in result.shape_results[:2])
    graph_rows = [
        row for row in result.shape_results
        if "same-process cuda graphs" in str(row.shape)
    ]
    assert len(graph_rows) == 1
    assert not graph_rows[0].passed


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="requires four CUDA GPUs",
)
def test_collective_nccl_cuda_graph_multi_output_tp4(tmp_path):
    source = tmp_path / "ar_norm.py"
    source.write_text(
        "import torch\n"
        "import torch.distributed as dist\n\n"
        "def ar_residual_rmsnorm(x, residual, weight, eps, out_norm, out_residual, group):\n"
        "    reduced = x.float().clone()\n"
        "    dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)\n"
        "    new_residual = reduced + residual.float()\n"
        "    variance = new_residual.pow(2).mean(dim=-1, keepdim=True)\n"
        "    norm = new_residual * torch.rsqrt(variance + float(eps)) * weight.float()\n"
        "    out_residual.copy_(new_residual.to(out_residual.dtype))\n"
        "    out_norm.copy_(norm.to(out_norm.dtype))\n"
    )
    result = _verify(
        source, slot="collective.ar_residual_rmsnorm",
        entry="ar_residual_rmsnorm", world_size=4, timeout_s=180.0,
    )

    assert result.passed, result.shape_results[0].detail
    assert result.graph_verified
    assert result.shape_results[0].graph_replays == 3


def test_collective_nccl_cuda_graph_divergent_rank_is_bounded(tmp_path):
    source = tmp_path / "divergent.py"
    source.write_text(
        "import torch\n"
        "import torch.distributed as dist\n\n"
        "def all_reduce(x, out, group=None):\n"
        "    if dist.get_rank(group) == 1 and torch.cuda.is_current_stream_capturing():\n"
        "        while True:\n"
        "            pass\n"
        "    out.copy_(x)\n"
        "    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)\n"
    )
    started = time.monotonic()
    result = _verify(source, timeout_s=5.0)

    assert not result.passed
    assert time.monotonic() - started < 30
    assert result.shape_results[0].detail
