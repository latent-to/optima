"""Distributed verification of the collective.all_reduce slot (CPU / gloo, 2 ranks).

Spawns 2 gloo ranks, runs the example all-reduce, and checks each rank's output equals
the trusted fp32 cross-rank sum. No GPU needed; torch-only (skipped where torch absent).
gloo has no bf16, so verify_collective uses fp32 on the CPU path.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from optima.slots import get_slot  # noqa: E402
from optima.registry import Eligibility  # noqa: E402
from optima.verify_collective import (  # noqa: E402
    _MAX_VERDICT_BYTES,
    _RankVerdict,
    _read_rank_verdict,
    _regular_identity,
    _write_rank_verdict,
    CollectiveVerdictError,
    verify_collective,
)

ALLREDUCE_BUNDLE = "examples/miner_allreduce_torch/kernels/all_reduce.py"
SMALL_SHAPES = [{"num_tokens": 2, "hidden": 8}]


def _verify(source=ALLREDUCE_BUNDLE, **kwargs):
    options = dict(
        world_size=2, backend="gloo", device="cpu", shapes=SMALL_SHAPES,
    )
    options.update(kwargs)
    return verify_collective(
        get_slot("collective.all_reduce"), str(source), "all_reduce", **options
    )


def test_collective_kind_discriminator():
    assert get_slot("collective.all_reduce").kind == "collective"


def test_allreduce_faithful_passes_gloo_cpu():
    slot = get_slot("collective.all_reduce")
    res = verify_collective(slot, ALLREDUCE_BUNDLE, "all_reduce",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert res.passed, "\n".join(f"{r.shape}: {r.detail}" for r in res.shape_results)


def test_collective_cpu_verify_does_not_claim_graph_proof():
    res = _verify(graph_safe=True)
    assert res.passed
    assert res.graph_required
    assert not res.graph_verified
    assert not res.fully_verified
    assert all(result.graph_replays == 0 for result in res.shape_results)


def test_non_reducing_kernel_fails_gloo_cpu(tmp_path):
    # A "reduce" that returns the LOCAL partial (forgets to sum across ranks) must fail:
    # out = x_rank != sum_r(x_r). Distributed verify is what catches this — a single-rank
    # check never would.
    broken = tmp_path / "broken_allreduce.py"
    broken.write_text("def all_reduce(x, out, group=None):\n    out.copy_(x)  # BUG: no cross-rank sum\n")
    slot = get_slot("collective.all_reduce")
    res = verify_collective(slot, str(broken), "all_reduce",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert not res.passed


def test_candidate_cannot_poison_its_own_trusted_reference(tmp_path):
    poisoned = tmp_path / "poison_reference.py"
    poisoned.write_text(
        "def all_reduce(x, out, group=None):\n"
        "    x.zero_()\n"
        "    out.zero_()\n"
    )

    result = _verify(poisoned)

    assert not result.passed
    assert result.shape_results[0].detail


def test_candidate_cannot_rebind_collective_input_to_equal_storage(tmp_path):
    rebinding = tmp_path / "rebind_input.py"
    rebinding.write_text(
        "import torch.distributed as dist\n"
        "def all_reduce(x, out, group=None):\n"
        "    replacement = x.detach().clone()\n"
        "    x.set_(replacement)\n"
        "    out.copy_(x)\n"
        "    dist.all_reduce(out, group=group)\n"
    )

    result = _verify(rebinding)

    assert not result.passed
    assert "validator-owned storage/tensor binding" in result.shape_results[0].detail


def test_candidate_cannot_replace_collective_output_storage(tmp_path):
    replacing = tmp_path / "replace_output.py"
    replacing.write_text(
        "import torch\n"
        "def all_reduce(x, out, group=None):\n"
        "    replacement = torch.empty_like(out)\n"
        "    out.set_(replacement)\n"
        "    out.copy_(x)\n"
    )

    result = _verify(replacing)

    assert not result.passed
    assert "validator-owned storage" in result.shape_results[0].detail


def test_candidate_cannot_change_collective_output_strides(tmp_path):
    restriding = tmp_path / "restride_output.py"
    restriding.write_text(
        "import torch.distributed as dist\n"
        "def all_reduce(x, out, group=None):\n"
        "    expected = x.clone()\n"
        "    dist.all_reduce(expected, group=group)\n"
        "    out.as_strided_(out.shape, (1, out.shape[0]))\n"
        "    out.copy_(expected)\n"
    )

    result = _verify(restriding)

    assert not result.passed
    assert "validator-owned storage/tensor binding" in result.shape_results[0].detail


def test_collective_dtype_and_topology_are_truthful():
    result = _verify(dtype_name="float64")
    assert result.passed, result.shape_results[0].detail
    assert result.dtype == "float64"
    assert all(row.dtype == "float64" for row in result.shape_results)

    with pytest.raises(ValueError, match="world_size >= 2"):
        _verify(world_size=1)
    with pytest.raises(ValueError, match="tp_size must equal"):
        _verify(tp_size=4)
    with pytest.raises(ValueError, match="floating torch dtype"):
        _verify(dtype_name="int32")


def test_collective_eligibility_routes_off_context_to_na_before_import(tmp_path):
    marker = tmp_path / "imported"
    source = tmp_path / "must_not_import.py"
    source.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "def all_reduce(x, out, group=None):\n"
        "    raise AssertionError('off-domain entry ran')\n"
    )
    result = _verify(
        source,
        eligibility=Eligibility(
            architectures=frozenset({"sm103"}), min_num_tokens=8,
        ),
        bundle_path=str(tmp_path / "missing_bundle"),
    )

    assert result.context_inapplicable
    assert not result.passed
    assert result.num_applicable == 0
    assert result.num_not_applicable == 1
    assert not marker.exists()


def test_collective_eligibility_runs_only_matching_shapes():
    result = _verify(
        shapes=[
            {"num_tokens": 1, "hidden": 8},
            {"num_tokens": 8, "hidden": 8},
        ],
        eligibility=Eligibility(max_num_tokens=2),
    )

    assert result.passed
    assert result.coverage_sufficient
    assert result.num_applicable == 1
    assert result.num_not_applicable == 1


def test_collective_graph_replay_and_timeout_arguments_fail_closed():
    with pytest.raises(ValueError, match="at least two"):
        _verify(graph_safe=True, graph_replays=1)
    for timeout in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            _verify(timeout_s=timeout)


def test_collective_verify_watchdog_terminates_hung_ranks(tmp_path):
    hanging = tmp_path / "hanging_allreduce.py"
    hanging.write_text(
        "def all_reduce(x, out, group=None):\n"
        "    while True:\n"
        "        pass\n"
    )
    started = time.monotonic()
    result = _verify(hanging, timeout_s=1.0)

    assert not result.passed
    assert "timed out" in result.shape_results[0].detail
    assert time.monotonic() - started < 20


def test_collective_verify_rejects_abrupt_rank_exit(tmp_path):
    abrupt = tmp_path / "abrupt.py"
    abrupt.write_text(
        "import os\n"
        "def all_reduce(x, out, group=None):\n"
        "    os._exit(7)\n"
    )
    result = _verify(abrupt, timeout_s=10.0)

    assert not result.passed
    assert result.shape_results[0].detail


def test_collective_valid_json_cannot_hide_nonzero_worker_exit(tmp_path):
    source = tmp_path / "exit_after_verdict.py"
    source.write_text(
        "import os\n"
        "import torch.distributed as dist\n"
        "import optima.verify_collective as verifier\n"
        "original_write = verifier._write_rank_verdict\n"
        "def exit_after_write(*args, **kwargs):\n"
        "    original_write(*args, **kwargs)\n"
        "    os._exit(7)\n"
        "verifier._write_rank_verdict = exit_after_write\n"
        "def all_reduce(x, out, group=None):\n"
        "    out.copy_(x)\n"
        "    dist.all_reduce(out, group=group)\n"
    )
    result = _verify(source, timeout_s=10.0)

    assert not result.passed
    assert "worker" in result.shape_results[0].detail


def _valid_verdict(*, rank=0, world_size=2):
    return _RankVerdict(
        version=1,
        rank=rank,
        world_size=world_size,
        passed=True,
        score=1.0,
        max_abs=0.0,
        detail="",
        metric="ratio",
        error=None,
        graph_replays=3,
    )


def _precreated(path: Path):
    path.touch(mode=0o600, exist_ok=False)
    return _regular_identity(path)


def test_collective_rank_verdict_round_trip(tmp_path):
    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    expected = _valid_verdict()
    _write_rank_verdict(path, expected, identity)

    assert _read_rank_verdict(
        path,
        expected_rank=0,
        expected_world_size=2,
        expected_identity=identity,
    ) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.replace('"rank":0', '"rank":0,"rank":0'),
        lambda value: value.replace('"score":1.0', '"score":NaN'),
        lambda value: value[:-1] + ',"extra":1}',
        lambda value: value.replace('"rank":0', '"rank":true'),
    ],
)
def test_collective_rank_verdict_rejects_malformed_json(tmp_path, mutation):
    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    payload = json.dumps(
        {
            "version": 1,
            "rank": 0,
            "world_size": 2,
            "passed": True,
            "score": 1.0,
            "max_abs": 0.0,
            "detail": "",
            "metric": "ratio",
            "error": None,
            "graph_replays": 3,
        },
        separators=(",", ":"),
    )
    path.write_text(mutation(payload))

    with pytest.raises(CollectiveVerdictError):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )


def test_collective_rank_verdict_rejects_rank_spoof_and_oversize(tmp_path):
    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    _write_rank_verdict(path, _valid_verdict(rank=1), identity)
    with pytest.raises(CollectiveVerdictError, match="identity mismatch"):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )

    path.write_bytes(b"x" * (_MAX_VERDICT_BYTES + 1))
    with pytest.raises(CollectiveVerdictError, match="size"):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )


def test_collective_rank_verdict_rejects_replaced_inode_and_symlink(tmp_path):
    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    path.unlink()
    path.write_text("{}")
    with pytest.raises(CollectiveVerdictError, match="inode changed"):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )

    path.unlink()
    target = tmp_path / "target"
    target.write_text("{}")
    path.symlink_to(target)
    with pytest.raises(CollectiveVerdictError, match="regular file"):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )


def test_collective_rank_parser_never_executes_pickle_payload(tmp_path):
    marker = tmp_path / "pickle_executed"

    class Payload:
        def __reduce__(self):
            return Path.write_text, (marker, "executed")

    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    path.write_bytes(pickle.dumps(Payload()))
    with pytest.raises(CollectiveVerdictError):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )
    assert not marker.exists()


def test_verify_entry_rejects_collective():
    # Collective slots must be verified distributed, not via the single-process verify_entry.
    from optima.verify import verify_entry

    slot = get_slot("collective.all_reduce")
    with pytest.raises(ValueError, match="collective"):
        verify_entry(slot, lambda *a, **k: None, device="cpu")
