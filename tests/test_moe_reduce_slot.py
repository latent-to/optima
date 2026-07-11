"""Distributed verify of moe.fused_experts_reduce — the block that owns its reduce.

Spawns 2 gloo ranks: each computes its local SwiGLU experts (different weight shard)
AND the cross-rank reduce, and the result must equal the fp32 sum of the per-rank
expert outputs. This is the slot that makes the compute-comm OVERLAP win expressible;
the test proves the distributed contract (experts + owned reduce) end to end on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from optima.slots import get_slot  # noqa: E402
from optima.verify_collective import verify_collective  # noqa: E402

BUNDLE = "examples/miner_moe_fused_experts_reduce_torch/kernels/moe_reduce.py"


def test_slot_is_collective_kind():
    slot = get_slot("moe.fused_experts_reduce")
    assert slot.kind == "collective"
    assert slot.prepare == "prepare"
    assert slot.collective_partial is not None and slot.invoke_collective is not None


def test_faithful_experts_plus_owned_reduce_passes_gloo_cpu():
    slot = get_slot("moe.fused_experts_reduce")
    res = verify_collective(slot, BUNDLE, "fused_experts_reduce", prepare_name="prepare",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert res.passed, "\n".join(f"{r.shape}: {r.detail}" for r in res.shape_results)


def test_kernel_that_skips_the_reduce_fails(tmp_path):
    # A kernel that computes the local experts but FORGETS the cross-rank reduce returns
    # only its own shard's output != sum over ranks -> distributed verify must catch it.
    broken = tmp_path / "broken.py"
    broken.write_text(
        "import torch, torch.nn.functional as F\n"
        "def prepare(w13, w2):\n"
        "    I = w13.shape[1] // 2\n"
        "    return {'w13': torch.cat([w13[:, I:], w13[:, :I]], 1).contiguous(), 'w2': w2.contiguous(), 'inter': I}\n"
        "def fused_experts_reduce(x, topk_ids, topk_weights, prepared, out, group=None):\n"
        "    w13, w2, I = prepared['w13'], prepared['w2'], prepared['inter']\n"
        "    M, H = x.shape; K = topk_ids.shape[1]; x32 = x.float()\n"
        "    acc = torch.zeros(M, H, dtype=torch.float32)\n"
        "    for k in range(K):\n"
        "        e = topk_ids[:, k].long(); wk = topk_weights[:, k].float()\n"
        "        fc1 = torch.einsum('mh,mih->mi', x32, w13[e].float())\n"
        "        up, gate = fc1[:, :I], fc1[:, I:]\n"
        "        acc += wk[:, None] * torch.einsum('mi,mhi->mh', F.silu(gate) * up, w2[e].float())\n"
        "    out.copy_(acc.to(out.dtype))  # BUG: never reduces across ranks\n"
    )
    slot = get_slot("moe.fused_experts_reduce")
    res = verify_collective(slot, str(broken), "fused_experts_reduce", prepare_name="prepare",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert not res.passed


def test_prepare_cannot_mutate_raw_weights_and_grade_against_them(tmp_path):
    poisoned = tmp_path / "mutating_prepare.py"
    poisoned.write_text(
        "def prepare(w13, w2):\n"
        "    w13.zero_(); w2.zero_()\n"
        "    return None\n"
        "def fused_experts_reduce(x, topk_ids, topk_weights, prepared, out, group=None):\n"
        "    out.zero_()\n"
    )

    result = verify_collective(
        get_slot("moe.fused_experts_reduce"),
        str(poisoned),
        "fused_experts_reduce",
        prepare_name="prepare",
        world_size=2,
        backend="gloo",
        device="cpu",
        shapes=[
            {
                "num_tokens": 2,
                "num_experts": 2,
                "hidden": 8,
                "inter": 4,
                "topk": 1,
            }
        ],
    )

    assert not result.passed
    assert "prepare input 'w13' was mutated" in result.shape_results[0].detail


def test_prepare_state_is_reused_across_temporal_calls(tmp_path):
    source = tmp_path / "one_call_prepared.py"
    source.write_text(
        Path(BUNDLE).read_text()
        + "\n_base_prepare = prepare\n"
        + "_base_entry = fused_experts_reduce\n"
        + "def prepare(w13, w2):\n"
        + "    state = _base_prepare(w13, w2)\n"
        + "    state['calls'] = 0\n"
        + "    return state\n"
        + "def fused_experts_reduce(x, topk_ids, topk_weights, prepared, out, group=None):\n"
        + "    prepared['calls'] += 1\n"
        + "    if prepared['calls'] == 1:\n"
        + "        return _base_entry(x, topk_ids, topk_weights, prepared, out, group)\n"
        + "    out.zero_()\n"
        + "    if dist.is_available() and dist.is_initialized():\n"
        + "        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)\n"
    )
    shapes = [
        {"num_tokens": tokens, "num_experts": 4, "hidden": 16,
         "inter": 8, "topk": 2}
        for tokens in (2, 5)
    ]

    result = verify_collective(
        get_slot("moe.fused_experts_reduce"),
        str(source),
        "fused_experts_reduce",
        prepare_name="prepare",
        world_size=2,
        backend="gloo",
        device="cpu",
        shapes=shapes,
    )

    assert not result.passed
    assert any("sequence" in str(row.shape) and not row.passed
               for row in result.shape_results)
