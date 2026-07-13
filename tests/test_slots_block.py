"""CPU tests for the generalized (op + block) slot abstraction.

Covers: (1) the new ``kind`` discriminator, (2) a multi-input *block* slot
(attention.sdpa) verifies a faithful pure-torch kernel, (3) the ``matched_ratio``
correctness mode FAILS a broken attention kernel, and (4) backward-compat — the
single-op slots (silu) still verify under the generalized spec. torch-only; skipped
where torch is unavailable (e.g. the dev laptop); runs on the pod.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from optima.sandbox import load_entry  # noqa: E402
from optima.slots import get_slot  # noqa: E402
from optima.verify import verify_entry  # noqa: E402

ATTN_BUNDLE = "examples/miner_attention_torch/kernels/attention.py"
ATTN_DECODE_BUNDLE = "examples/miner_attention_decode_torch/kernels/attention_decode.py"
MOE_BUNDLE = "examples/miner_moe_fused_experts_torch/kernels/moe.py"


def test_slot_kind_discriminator():
    assert get_slot("attention.sdpa").kind == "block"
    assert get_slot("attention.decode").kind == "block"
    assert get_slot("activation.silu_and_mul").kind == "op"
    assert get_slot("norm.rmsnorm").kind == "op"


def test_attention_block_passes_correctness_cpu():
    entry = load_entry(ATTN_BUNDLE, "attention")
    slot = get_slot("attention.sdpa")
    result = verify_entry(slot, entry, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed, "\n".join(
        f"{r.shape}: max_abs={r.max_abs_err} ratio={r.pass_ratio} {r.detail}" for r in result.shape_results
    )


def test_broken_attention_fails_matched_ratio_cpu():
    # A broken "kernel": ignores k/v entirely and copies q -> wrong everywhere, so the
    # fraction-of-elements-within-tolerance falls below the slot's min_ratio (0.99).
    def broken(q, k, v, out, sm_scale, causal=True):
        out.copy_(q)

    slot = get_slot("attention.sdpa")
    result = verify_entry(slot, broken, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed


def test_attention_decode_passes_correctness_cpu():
    entry = load_entry(ATTN_DECODE_BUNDLE, "attention_decode")
    slot = get_slot("attention.decode")
    result = verify_entry(slot, entry, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed, "\n".join(
        f"{r.shape}: max_abs={r.max_abs_err} ratio={r.pass_ratio} {r.detail}" for r in result.shape_results
    )


def test_broken_decode_fails_matched_ratio_cpu():
    # Ignores the seq_lens mask -> attends to ALL padded positions (incl. garbage),
    # so requests with seq_len < ctx come out wrong -> ratio below min_ratio.
    def broken(q, k, v, seq_lens, sm_scale, out):
        Hq, Hkv = q.shape[1], k.shape[2]
        g = Hq // Hkv
        k32 = k.float().repeat_interleave(g, dim=2)
        v32 = v.float().repeat_interleave(g, dim=2)
        scores = torch.einsum("bhd,bshd->bhs", q.float(), k32) * sm_scale  # NO mask
        p = torch.softmax(scores, dim=-1)
        out.copy_(torch.einsum("bhs,bshd->bhd", p, v32).to(out.dtype))

    slot = get_slot("attention.decode")
    result = verify_entry(slot, broken, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed


def test_moe_prepare_forward_passes_correctness_cpu():
    # The (prepare, forward) PAIR: load BOTH miner callables; verify runs prepare (the
    # weight layout) then forward, and compares to the fp32 MoE reference.
    fwd = load_entry(MOE_BUNDLE, "fused_experts")
    prep = load_entry(MOE_BUNDLE, "prepare")
    slot = get_slot("moe.fused_experts")
    result = verify_entry(slot, fwd, prepare=prep, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed, "\n".join(
        f"{r.shape}: ratio={r.pass_ratio} {r.detail}" for r in result.shape_results
    )


def test_moe_broken_prepare_fails_cpu():
    # A `prepare` that forgets the [gate;up]->[up;gate] reorder -> forward swaps the
    # halves -> silu(up)*gate != silu(gate)*up -> wrong. The slot is only correct when
    # BOTH callables agree, so a bad prepare must fail just like a bad forward would.
    def broken_prepare(w13, w2):
        return {"w13": w13.contiguous(), "w2": w2.contiguous(), "inter": w13.shape[1] // 2}

    fwd = load_entry(MOE_BUNDLE, "fused_experts")
    slot = get_slot("moe.fused_experts")
    result = verify_entry(slot, fwd, prepare=broken_prepare, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed


def test_moe_is_prepare_forward_slot():
    slot = get_slot("moe.fused_experts")
    assert slot.kind == "block"
    assert slot.prepare == "prepare"          # names the 2nd miner callable
    assert slot.invoke_prepare is not None
    # forward-only slots have no prepare:
    assert get_slot("activation.silu_and_mul").prepare is None
    assert get_slot("activation.silu_and_mul").invoke_prepare is None


def test_silu_op_still_verifies_under_generalized_spec():
    # Backward-compat: the single-op path is unchanged by the multi-output/block work.
    def silu(x, out):
        d = x.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(x[..., :d].float()).to(x.dtype) * x[..., d:])

    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, silu, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed


def test_matched_ratio_is_active_only_for_the_block_slot():
    # The op slots keep all-close; the attention block uses matched_ratio. This guards
    # against accidentally loosening the op gates when the abstraction was generalized.
    assert get_slot("activation.silu_and_mul").correctness.mode == "allclose"
    assert get_slot("norm.rmsnorm").correctness.mode == "allclose"
    assert get_slot("attention.sdpa").correctness.mode == "matched_ratio"
    assert get_slot("attention.decode").correctness.mode == "matched_ratio"
    assert get_slot("moe.fused_experts").correctness.mode == "matched_ratio"
