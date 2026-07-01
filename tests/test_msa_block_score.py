"""CPU tests for the attention.msa_block_score slot + the topk_overlap correctness mode.

Proves the subnet can ingest a SELECTION-output win: a kernel emits block scores, and the
gate is whether the top-k block SETS agree (not the score values) — so a value-perturbing but
selection-preserving kernel (an fp8 index-K) passes, while a wrong-selection kernel fails.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from optima.slots import Correctness, get_slot  # noqa: E402
from optima.verify import _compare, verify_entry  # noqa: E402

SLOT = get_slot("attention.msa_block_score")


def _block_scores(q, index_k, seq_lens, block_size):
    B, Hq, D = q.shape
    S = index_k.shape[1]
    nblk = S // block_size
    qs = q.float().sum(1)
    ks = index_k.float()[:, :, 0, :]
    sc = torch.einsum("bd,bsd->bs", qs, ks)
    sidx = torch.arange(S, device=q.device).view(1, S)
    sc = sc.masked_fill(sidx >= seq_lens.view(B, 1), float("-inf"))
    return sc.view(B, nblk, block_size).amax(-1)


def _faithful(q, index_k, seq_lens, block_size, out):
    out.copy_(_block_scores(q, index_k, seq_lens, block_size).to(out.dtype))


def _monotone_perturb(q, index_k, seq_lens, block_size, out):
    # Values shifted/scaled (like fp8 index-K) but MONOTONICALLY -> identical selection.
    s = _block_scores(q, index_k, seq_lens, block_size)
    out.copy_((s * 1.01 + 0.001).to(out.dtype))


def _wrong_selection(q, index_k, seq_lens, block_size, out):
    # Negate the scores -> the top-k becomes the bottom-k -> selection disagrees.
    s = _block_scores(q, index_k, seq_lens, block_size)
    out.copy_((-s).to(out.dtype))


# ---- the slot is in the catalog with the right contract ---------------------

def test_msa_slot_registered():
    assert SLOT.kind == "block"
    assert SLOT.correctness.mode == "topk_overlap"
    assert SLOT.correctness.top_k == 8


# ---- the topk_overlap metric, unit ------------------------------------------

def test_topk_overlap_metric_unit():
    c = Correctness("topk_overlap", top_k=2, min_overlap=0.875)
    a = torch.tensor([[5.0, 1.0, 4.0, 0.0], [0.0, 9.0, 8.0, 1.0]])  # top-2: {0,2}, {1,2}
    same = torch.tensor([[50.0, 2.0, 40.0, 1.0], [1.0, 90.0, 80.0, 2.0]])  # same selection, diff values
    flipped = -a  # bottom becomes top
    ok, *_, score, _, metric = _compare(a, same, atol=0, rtol=0, correctness=c)
    assert ok and score == 1.0 and metric == "overlap"
    ok2, *_, score2, _, _ = _compare(flipped, same, atol=0, rtol=0, correctness=c)
    assert not ok2 and score2 == 0.0


def test_topk_overlap_tolerates_masked_inf():
    # -inf masked positions must NOT trip the finite guard (the metric runs before it).
    c = Correctness("topk_overlap", top_k=2, min_overlap=0.875)
    a = torch.tensor([[5.0, 1.0, 4.0, float("-inf")]])
    e = torch.tensor([[50.0, 2.0, 40.0, float("-inf")]])
    ok, *_rest = _compare(a, e, atol=0, rtol=0, correctness=c)
    assert ok


# ---- end-to-end through verify_entry (jittered shapes) ----------------------

def test_msa_faithful_kernel_verifies():
    res = verify_entry(SLOT, _faithful, dtype=torch.float32, device="cpu", seed=0, jitter_seed=7)
    assert res.passed, res.shape_results
    assert all(r.metric == "overlap" for r in res.shape_results)


def test_msa_monotone_perturbation_verifies():
    # fp8-like: perturb every score, keep the selection -> still passes (the whole point).
    res = verify_entry(SLOT, _monotone_perturb, dtype=torch.float32, device="cpu", seed=0)
    assert res.passed, res.shape_results


def test_msa_wrong_selection_fails():
    res = verify_entry(SLOT, _wrong_selection, dtype=torch.float32, device="cpu", seed=0)
    assert not res.passed


def test_msa_gate_is_never_vacuous():
    # n_blocks must exceed top_k on EVERY verify shape (top-k of exactly k blocks
    # selects everything -> any output scores overlap 1.0), including when count-dim
    # jitter drives ctx down (the make_inputs floor).
    for sh in list(SLOT.shapes) + [dict(SLOT.shapes[0], ctx=1024), dict(SLOT.shapes[0], ctx=256)]:
        i = SLOT.make_inputs(**sh, dtype=torch.float32, device="cpu", seed=0)
        n_blocks = i["index_k"].shape[1] // i["block_size"]
        assert n_blocks > SLOT.correctness.top_k, f"vacuous shape: {sh} -> {n_blocks} blocks"
