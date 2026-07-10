"""CPU tests for the attention.msa_prefill_block_score slot (the prefill-side indexer).

The prefill sibling of attention.msa_block_score: a T-token chunk scores S = prefix+T keys
under the causal rule, emitting a (T, ceil(S/block)) score SHEET gated per row on
topk_overlap. Same selection-not-values philosophy as the decode slot, plus the two failure
classes specific to prefill: a kernel that ignores CAUSALITY (future keys leak into scores)
and a kernel that mis-handles the RAGGED tail block.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from optima.slots import get_slot  # noqa: E402
from optima.verify import verify_entry  # noqa: E402

SLOT = get_slot("attention.msa_prefill_block_score")


def _sheet(q, index_k, prefix_len, scale, block_size, *, causal: bool = True):
    T, D = q.shape
    S = index_k.shape[0]
    s = (q.float() @ index_k.float().t()) * float(scale)
    if causal:
        m = torch.arange(T, device=q.device).view(T, 1)
        n = torch.arange(S, device=q.device).view(1, S)
        s = s.masked_fill(n > int(prefix_len) + m, float("-inf"))
    nblk = (S + block_size - 1) // block_size
    pad = nblk * block_size - S
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
    return s.view(T, nblk, block_size).amax(-1)


def _faithful(q, index_k, prefix_len, scale, block_size, out):
    out.copy_(_sheet(q, index_k, prefix_len, scale, block_size).to(out.dtype))


def _monotone_perturb(q, index_k, prefix_len, scale, block_size, out):
    # fp8-like: every score moved, selection preserved.
    s = _sheet(q, index_k, prefix_len, scale, block_size)
    out.copy_((s * 1.01 + 0.001).to(out.dtype))


def _wrong_selection(q, index_k, prefix_len, scale, block_size, out):
    s = _sheet(q, index_k, prefix_len, scale, block_size)
    out.copy_((-s).to(out.dtype))


def _acausal(q, index_k, prefix_len, scale, block_size, out):
    # Ignores the causal mask: rows see FUTURE keys. With random data, future blocks
    # outscore visible ones often enough that per-row selections diverge -> must fail.
    out.copy_(_sheet(q, index_k, prefix_len, scale, block_size, causal=False).to(out.dtype))


def _tail_garbage(q, index_k, prefix_len, scale, block_size, out):
    # Correct everywhere except the ragged tail block, which reads past S (modeled as a
    # huge score): the tail block jumps into every row's top-k -> selection disagrees.
    s = _sheet(q, index_k, prefix_len, scale, block_size)
    s[:, -1] = s.max() + 100.0
    out.copy_(s.to(out.dtype))


# ---- catalog / contract ------------------------------------------------------

def test_prefill_slot_registered():
    assert SLOT.kind == "block"
    assert SLOT.correctness.mode == "topk_overlap"
    assert SLOT.correctness.top_k == 8
    assert SLOT.kl_threshold == 3e-2


def test_out_shape_covers_ragged_tail():
    i = SLOT.make_inputs(**SLOT.shapes[0], dtype=torch.float32, device="cpu", seed=0)
    S = i["index_k"].shape[0]
    assert S % i["block_size"] != 0, "shape must exercise the ragged tail"
    (shape,) = SLOT.out_shapes(i)
    assert shape == (i["q"].shape[0], (S + i["block_size"] - 1) // i["block_size"])


# ---- verify_entry (jittered shapes) ------------------------------------------

def test_prefill_faithful_kernel_verifies():
    res = verify_entry(SLOT, _faithful, dtype=torch.float32, device="cpu", seed=0, jitter_seed=7)
    assert res.passed, res.shape_results
    assert all(r.metric == "overlap" for r in res.shape_results)


def test_prefill_monotone_perturbation_verifies():
    res = verify_entry(SLOT, _monotone_perturb, dtype=torch.float32, device="cpu", seed=0)
    assert res.passed, res.shape_results


def test_prefill_wrong_selection_fails():
    res = verify_entry(SLOT, _wrong_selection, dtype=torch.float32, device="cpu", seed=0)
    assert not res.passed


def test_prefill_acausal_kernel_fails():
    res = verify_entry(SLOT, _acausal, dtype=torch.float32, device="cpu", seed=0)
    assert not res.passed


def test_prefill_tail_block_garbage_fails():
    res = verify_entry(SLOT, _tail_garbage, dtype=torch.float32, device="cpu", seed=0)
    assert not res.passed


# ---- the gate is never vacuous, per ROW --------------------------------------

def test_prefill_gate_is_never_vacuous():
    # Row 0 is the worst case (it sees only prefix_len+1 keys): even under count-dim
    # jitter driving prefix_blocks down, every row's VISIBLE block count must exceed
    # top_k, else top-k-of-k makes that row's overlap 1.0 for any output.
    for sh in list(SLOT.shapes) + [dict(SLOT.shapes[0], prefix_blocks=1),
                                   dict(SLOT.shapes[0], q_len=1)]:
        i = SLOT.make_inputs(**sh, dtype=torch.float32, device="cpu", seed=0)
        visible_row0 = (int(i["prefix_len"]) + 1 + i["block_size"] - 1) // i["block_size"]
        assert visible_row0 > SLOT.correctness.top_k, f"vacuous row-0: {sh} -> {visible_row0}"
