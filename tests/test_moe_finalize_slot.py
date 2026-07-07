"""collective.moe_finalize_ar_rmsnorm — the deep fused-epilogue slot, CPU-only.

Pins the fe_export ABI (K-major row_map, T-major scales, head-trim T<=T_exp), the
replication split (gemm_out per-rank, routing/model state replicated), the trusted
finalize math against an independent naive loop, and the distributed gate: a faithful
torch entry passes gloo world-2; dropping the finalize scales (or the residual) fails.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from optima.slots import get_slot  # noqa: E402
from optima.verify_collective import verify_collective  # noqa: E402

SLOT = "collective.moe_finalize_ar_rmsnorm"


def _inputs(rank=0, **over):
    kw = dict(num_tokens=6, exp_tokens=9, topk=5, hidden=32, dtype=torch.float32,
              device="cpu", seed=11, rank=rank, world_size=2)
    kw.update(over)
    return get_slot(SLOT).make_inputs(**kw)


def test_replication_split():
    a, b = _inputs(rank=0), _inputs(rank=1)
    assert not torch.equal(a["gemm_out"], b["gemm_out"])  # per-rank partial
    for key in ("row_map", "scales", "residual", "weight"):
        assert torch.equal(a[key], b[key]), key  # replicated router/model state


def test_finalize_matches_naive_loop():
    slot = get_slot(SLOT)
    i = _inputs()
    got = slot.collective_partial(i, None)
    t, h = i["residual"].shape
    t_exp, k = i["scales"].shape
    want = torch.zeros(t, h)
    for tok in range(t):
        for ki in range(k):
            row = int(i["row_map"][tok + ki * t_exp])
            want[tok] += float(i["scales"][tok, ki]) * i["gemm_out"][row].float()
    assert torch.allclose(got, want, atol=1e-5)
    assert got.shape == (t, h)  # head-trimmed to T, not T_exp


def test_jitter_clamp_keeps_t_le_texp():
    i = _inputs(num_tokens=40, exp_tokens=9)  # jitter can push T past T_exp
    assert i["scales"].shape[0] == 40  # exp_tokens clamped up
    assert i["gemm_out"].shape[0] == 40 * 5


def test_reference_composes_partial_and_finish():
    slot = get_slot(SLOT)
    i = _inputs()
    norm, res = slot.invoke_reference(i)
    summed = slot.collective_partial(i, None)
    new_res = summed + i["residual"].float()
    var = new_res.pow(2).mean(dim=-1, keepdim=True)
    want = new_res * torch.rsqrt(var + i["eps"]) * i["weight"].float()
    assert torch.allclose(norm, want) and torch.allclose(res, new_res)


FAITHFUL = """\
import torch
import torch.distributed as dist


def moe_finalize_ar_rmsnorm(gemm_out, row_map, scales, residual, weight, eps,
                            out_norm, out_residual, group):
    t = residual.shape[0]
    t_exp, k = scales.shape
    per_k = gemm_out.float()[row_map.long().view(k, t_exp)]
    acc = (per_k * scales.float().t().unsqueeze(-1)).sum(dim=0)[:t]
    dist.all_reduce(acc, op=dist.ReduceOp.SUM, group=group)
    new_res = acc + residual.float()
    var = new_res.pow(2).mean(dim=-1, keepdim=True)
    norm = new_res * torch.rsqrt(var + float(eps)) * weight.float()
    out_residual.copy_(new_res.to(out_residual.dtype))
    out_norm.copy_(norm.to(out_norm.dtype))
"""

# Ignores the router scales (uniform 1/K instead) — plausible-looking output, wrong
# expert weighting: the class of cheat/bug the finalize gate must kill.
NO_SCALES = FAITHFUL.replace(
    "(per_k * scales.float().t().unsqueeze(-1)).sum(dim=0)[:t]",
    "(per_k / float(k)).sum(dim=0)[:t]")


def test_faithful_passes_gloo_cpu(tmp_path):
    p = tmp_path / "faithful.py"
    p.write_text(FAITHFUL)
    res = verify_collective(get_slot(SLOT), str(p), "moe_finalize_ar_rmsnorm",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert res.passed, "\n".join(f"{r.shape}: {r.detail}" for r in res.shape_results)


def test_no_scales_fails_gloo_cpu(tmp_path):
    p = tmp_path / "no_scales.py"
    p.write_text(NO_SCALES)
    res = verify_collective(get_slot(SLOT), str(p), "moe_finalize_ar_rmsnorm",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert not res.passed
