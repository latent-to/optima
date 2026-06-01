"""Typed op-slot catalog — the submission ABI.

A *slot* is a replaceable, narrowly-typed region of the fixed model graph. The
validator owns this catalog; a miner may only target a slot that exists here, and
provides the small ``entry`` callable described by the slot's contract. Everything
around the slot (tensor allocation, the call site, the rest of the model) stays
validator-owned.

A slot comes in two ``kind``s, and the difference is only the *breadth* of the
typed boundary — the cheat-resistance story is identical for both: the validator
allocates the outputs, the miner only fills them, and the miner never produces the
final tokens/logprobs (so there is nothing to substitute, the attack that bites
whole-model submissions).

* ``"op"`` — a single fused op. ``silu_and_mul`` (``entry(x, out)``), ``rmsnorm``
  (``entry(x, weight, out, eps)``).
* ``"block"`` — a region that fuses several ops behind one tensor-in/tensor-out
  contract, for bigger wins. ``attention.sdpa`` (``entry(q, k, v, out, sm_scale,
  causal)``) is the first: it subsumes QK^T + softmax + (·)V. A block has the *same
  shape* of contract as an op (named tensor inputs -> validator-allocated outputs),
  just wider — which is exactly why the seam / verify / registry machinery is
  unchanged. The breadth is bounded: a slot must stay strictly upstream of the
  logprobs/sampler, or the output-substitution attack reappears.

Each slot carries everything the validator needs to verify a submission without
trusting it: a trusted high-precision ``invoke_reference``, a deterministic input
generator, the standard shapes, per-dtype tolerances, explicit
``invoke_reference`` / ``invoke_entry`` (so non-uniform call shapes work), and a
``Correctness`` policy. The policy matters once a kernel legitimately changes
numerics (flash-style softmax reductions, fp8, MLA weight absorption): such kernels
are NOT bit-exact to the reference, so the gate is a *matched ratio* (>= rho of
elements within tolerance against high-precision ground truth) rather than
all-close — the deterministic-vs-low-precision tiering from FlashInfer-Bench. The
reference is always high-precision ground truth, never the stock kernel.

Some slots are a **(prepare, forward) pair**: a quantized / layout-sensitive kernel
(MoE experts, a quant GEMM) needs the *weights* in a custom layout, and that layout
transform is part of the kernel. Such a slot names a second miner callable via
``prepare`` — it runs ONCE at load on the raw checkpoint weights, the validator holds
the result, and ``entry`` (forward) consumes it each step as ``prepared``. This is how
a win like the GPT-OSS sm120 MoE (repack W13, interleave the FP4 block scales, then a
fused CUTLASS call) fits *one* slot: the repack/interleave is ``prepare``, the kernel
is ``forward``.

Adding a slot is a validator action (a code change here), never a miner action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass(frozen=True)
class Correctness:
    """How ``verify`` compares the miner output to the reference.

    * ``"allclose"`` — every element must satisfy ``|a-e| <= atol + rtol*|e|``.
      Right for kernels meant to be numerically equivalent (a faster silu).
    * ``"matched_ratio"`` — at least ``min_ratio`` of elements must satisfy that
      bound. Right for kernels that legitimately differ from the reference at the
      ULP level (attention reorders the softmax reduction; fp8 / weight-absorbed
      forms shift a few elements). Calibrate ``min_ratio`` to the stock-vs-stock
      noise floor — the same discipline as the KL gate.
    * ``"cosine"`` — cosine similarity of the flattened output vs the HP reference
      must be >= ``min_cosine``, with an optional relative-L2-norm guard
      (``max_rel_norm_err``) to catch a kernel that gets the direction right but the
      scale wrong. This is the correct fidelity metric for **low-bit** kernels
      (MXFP4/MXFP8): element-wise tolerance is meaningless when every element carries
      ~6-12% quantization error, but the *direction* (and energy) of the block output
      is preserved — which is what actually drives the model's logits. Measured on
      sm120 GPT-OSS MoE: ~0.999 vs a dequant reference, ~0.99 vs fp32 ground truth.
    """

    mode: str = "allclose"  # "allclose" | "matched_ratio" | "cosine"
    min_ratio: float = 1.0
    min_cosine: float = 0.0  # cosine mode: min cosine similarity vs the HP reference
    max_rel_norm_err: float = 0.0  # cosine mode: optional |‖a‖-‖e‖|/‖e‖ guard (0 = off)


@dataclass(frozen=True)
class SlotSpec:
    name: str  # dotted slot id, e.g. "activation.silu_and_mul"
    entry: str  # required callable name the miner module must expose
    summary: str  # human-readable contract
    kind: str  # "op" (single fused op) | "block" (a region of several fused ops)

    make_inputs: Callable[..., dict]  # (**shape, dtype, device, seed) -> {name: tensor|scalar}
    out_shapes: Callable[[dict], Sequence[tuple]]  # (inputs) -> one shape per output the validator allocates
    invoke_reference: Callable[[dict], Sequence[torch.Tensor]]  # (inputs) -> expected outputs (HIGH PRECISION)
    invoke_entry: Callable[..., None]  # (entry, inputs, outs, prepared) -> None; writes each tensor in `outs`
    shapes: tuple[dict, ...]
    # Optional 2nd miner callable for (prepare, forward) slots: `prepare` runs ONCE at
    # load on the raw weights (quant/layout transform); the validator holds the result
    # and passes it to `entry` each step as `prepared`. None -> a plain forward-only slot.
    prepare: Optional[str] = None
    invoke_prepare: Optional[Callable] = None  # (prepare_fn, inputs) -> prepared (None for forward-only)
    # Live seam: build the args for the miner's prepare() from the actual sglang layer
    # (validator-owned layer->contract mapping). The dispatcher calls
    # prepare(*prepare_from_layer(layer)); invoke_prepare mirrors the SAME call shape for
    # verify. This is how a slot carries more than two dense tensors (biases, the
    # interleaving flag, quant scales) without widening the generic contract. None ->
    # the dispatcher defaults to (layer.w13_weight.data, layer.w2_weight.data).
    prepare_from_layer: Optional[Callable] = None
    correctness: Correctness = field(default_factory=Correctness)
    tolerances: dict[torch.dtype, Tolerance] = field(default_factory=dict)

    def tolerance_for(self, dtype: torch.dtype) -> Tolerance:
        if dtype in self.tolerances:
            return self.tolerances[dtype]
        if dtype in (torch.float16, torch.bfloat16):
            return Tolerance(atol=2e-2, rtol=2e-2)
        return Tolerance(atol=1e-4, rtol=1e-4)


_BF16_TOL = {
    torch.bfloat16: Tolerance(2e-2, 2e-2),
    torch.float16: Tolerance(1e-2, 1e-2),
    torch.float32: Tolerance(1e-5, 1e-5),
}


# ---------------------------------------------------------------------------
# Slot (op): activation.silu_and_mul   (Qwen/Llama-class MLP)
#   x:(...,2d) -> out:(...,d) = silu(x[...,:d]) * x[...,d:]
#   contract: entry(x, out)
# ---------------------------------------------------------------------------


def _silu_reference(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d].float()).to(x.dtype) * x[..., d:]


def _silu_inputs(*, num_tokens: int, d: int, dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_tokens, 2 * d, generator=g, device=device, dtype=torch.float32).to(dtype)
    return {"x": x}


SILU_AND_MUL = SlotSpec(
    name="activation.silu_and_mul",
    entry="silu_and_mul",
    summary="out = silu(x[...,:d]) * x[...,d:];  x:(...,2d) -> out:(...,d);  entry(x, out)",
    kind="op",
    make_inputs=_silu_inputs,
    out_shapes=lambda i: [(*i["x"].shape[:-1], i["x"].shape[-1] // 2)],
    invoke_reference=lambda i: [_silu_reference(i["x"])],
    invoke_entry=lambda entry, i, outs, prepared: entry(i["x"], outs[0]),
    shapes=(
        {"num_tokens": 1, "d": 1024},
        {"num_tokens": 8, "d": 1024},
        {"num_tokens": 128, "d": 4096},
        {"num_tokens": 4096, "d": 4096},
        {"num_tokens": 333, "d": 2880},
    ),
    correctness=Correctness("allclose"),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Slot (op): norm.rmsnorm   (universal — every transformer, incl. GPT-OSS)
#   out = x / sqrt(mean(x^2, -1) + eps) * weight
#   contract: entry(x, weight, out, eps).  Validator owns the residual add.
# ---------------------------------------------------------------------------


def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    normed = x32 * torch.rsqrt(var + eps)
    return (normed * weight.float()).to(x.dtype)


def _rmsnorm_inputs(*, num_tokens: int, hidden: int, dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_tokens, hidden, generator=g, device=device, dtype=torch.float32).to(dtype)
    w = torch.randn(hidden, generator=g, device=device, dtype=torch.float32).to(dtype)
    return {"x": x, "weight": w, "eps": 1e-6}


RMSNORM = SlotSpec(
    name="norm.rmsnorm",
    entry="rmsnorm",
    summary="out = x*rsqrt(mean(x^2,-1)+eps)*weight;  x:(...,H),weight:(H,) -> out:(...,H);  entry(x, weight, out, eps)",
    kind="op",
    make_inputs=_rmsnorm_inputs,
    out_shapes=lambda i: [tuple(i["x"].shape)],
    invoke_reference=lambda i: [_rmsnorm_reference(i["x"], i["weight"], i["eps"])],
    invoke_entry=lambda entry, i, outs, prepared: entry(i["x"], i["weight"], outs[0], i["eps"]),
    shapes=(
        {"num_tokens": 1, "hidden": 2880},
        {"num_tokens": 8, "hidden": 2880},
        {"num_tokens": 128, "hidden": 2880},
        {"num_tokens": 4096, "hidden": 4096},
        {"num_tokens": 333, "hidden": 1536},
    ),
    correctness=Correctness("allclose"),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Slot (BLOCK): attention.sdpa   (scaled-dot-product attention, GQA/MQA-capable)
#   q:(T,Hq,D)  k,v:(S,Hkv,D) -> o:(T,Hq,D) = softmax(qk^T*scale + causal) v
#   contract: entry(q, k, v, out, sm_scale, causal)
#
# This is the first *block* slot — the attention compute core every backend
# (FlashAttention / FlashInfer / FlashMLA / Triton) implements. It demonstrates the
# generalization: several fused ops behind one typed boundary, multiple tensor
# inputs, and a matched-ratio gate (a real flash/online-softmax/fp8 kernel is not
# bit-exact). The seam (optima/integrations/sglang_attention.py) routes the model's
# attention through this contract at the RadixAttention chokepoint; the paged-decode
# / MLA-latent variants are sibling slots that reuse the same dispatcher with a wider
# input tuple (compressed KV + page table). See dispatch.make_attention_dispatcher.
# ---------------------------------------------------------------------------


def _sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, causal: bool) -> torch.Tensor:
    # q:(T,Hq,D)  k,v:(S,Hkv,Dv) -> o:(T,Hq,Dv).  GQA/MQA via Hq % Hkv == 0.
    T, Hq, D = q.shape
    S, Hkv, Dv = v.shape
    g = Hq // Hkv
    q32 = q.float()
    k32 = k.float().repeat_interleave(g, dim=1)  # (S,Hq,D)
    v32 = v.float().repeat_interleave(g, dim=1)  # (S,Hq,Dv)
    scores = torch.matmul(q32.permute(1, 0, 2), k32.permute(1, 2, 0)) * sm_scale  # (Hq,T,S)
    if causal:
        offset = S - T  # the cached prefix length (0 in the self-contained case)
        ti = torch.arange(T, device=q.device).view(T, 1)
        si = torch.arange(S, device=q.device).view(1, S)
        scores = scores.masked_fill((si > ti + offset).view(1, T, S), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, v32.permute(1, 0, 2)).permute(1, 0, 2)  # (T,Hq,Dv)
    return o.to(q.dtype)


def _sdpa_inputs(*, num_tokens: int, num_q_heads: int, num_kv_heads: int, head_dim: int,
                 dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(dtype)

    return {
        "q": rnd(num_tokens, num_q_heads, head_dim),
        "k": rnd(num_tokens, num_kv_heads, head_dim),
        "v": rnd(num_tokens, num_kv_heads, head_dim),
        "sm_scale": 1.0 / (head_dim ** 0.5),
        "causal": True,
    }


ATTENTION_SDPA = SlotSpec(
    name="attention.sdpa",
    entry="attention",
    summary=(
        "o = softmax(q k^T * scale + causal_mask) v  (GQA/MQA);  "
        "q:(T,Hq,D) k,v:(S,Hkv,D) -> o:(T,Hq,D);  entry(q, k, v, out, sm_scale, causal)"
    ),
    kind="block",
    make_inputs=_sdpa_inputs,
    out_shapes=lambda i: [tuple(i["q"].shape)],
    invoke_reference=lambda i: [_sdpa_reference(i["q"], i["k"], i["v"], i["sm_scale"], i["causal"])],
    invoke_entry=lambda entry, i, outs, prepared: entry(i["q"], i["k"], i["v"], outs[0], i["sm_scale"], i["causal"]),
    shapes=(
        {"num_tokens": 1, "num_q_heads": 8, "num_kv_heads": 8, "head_dim": 64},
        {"num_tokens": 16, "num_q_heads": 8, "num_kv_heads": 2, "head_dim": 128},   # GQA
        {"num_tokens": 64, "num_q_heads": 16, "num_kv_heads": 16, "head_dim": 128},
        {"num_tokens": 128, "num_q_heads": 8, "num_kv_heads": 1, "head_dim": 128},  # MQA
    ),
    # A real attention kernel reorders the softmax reduction (flash / online softmax)
    # and may run in fp8, so it is NOT bit-exact: gate on a matched ratio against the
    # fp32 reference, not all-close. 0.99 tolerates a thin tail of ULP-level diffs.
    correctness=Correctness("matched_ratio", min_ratio=0.99),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Slot (BLOCK): attention.decode   (paged-decode attention — the runtime-wired one)
#   q:(B,Hq,D)  k,v:(B,S,Hkv,D)  seq_lens:(B,) -> o:(B,Hq,D)
#   Each request's single query attends to its first seq_lens[i] cached k/v.
#   contract: entry(q, k, v, seq_lens, sm_scale, out)
#
# This is the slot the attention seam routes *decode* attention to: the validator
# gathers each request's paged KV out of forward_batch into the padded (B,S,Hkv,D)
# view, the miner fills `out`. See dispatch.make_attention_dispatcher / _run_decode_kernel.
# ---------------------------------------------------------------------------


def _decode_attn_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           seq_lens: torch.Tensor, sm_scale: float) -> torch.Tensor:
    # q:(B,Hq,D)  k,v:(B,S,Hkv,D)  seq_lens:(B,) -> o:(B,Hq,D).  GQA/MQA via Hq % Hkv == 0.
    B, Hq, D = q.shape
    S, Hkv = k.shape[1], k.shape[2]
    g = Hq // Hkv
    q32 = q.float()
    k32 = k.float().repeat_interleave(g, dim=2)  # (B,S,Hq,D)
    v32 = v.float().repeat_interleave(g, dim=2)  # (B,S,Hq,D)
    scores = torch.einsum("bhd,bshd->bhs", q32, k32) * sm_scale  # (B,Hq,S)
    sidx = torch.arange(S, device=q.device).view(1, 1, S)
    scores = scores.masked_fill(sidx >= seq_lens.view(B, 1, 1), float("-inf"))  # mask padding/beyond-context
    p = torch.softmax(scores, dim=-1)
    o = torch.einsum("bhs,bshd->bhd", p, v32)  # (B,Hq,D)
    return o.to(q.dtype)


def _decode_attn_inputs(*, batch: int, num_q_heads: int, num_kv_heads: int, head_dim: int, ctx: int,
                        dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(dtype)

    seq_lens = torch.randint(1, ctx + 1, (batch,), generator=g, device=device).to(torch.int32)
    seq_lens[0] = ctx  # ensure one full-length request (exercises the whole window + the mask)
    return {
        "q": rnd(batch, num_q_heads, head_dim),
        "k": rnd(batch, ctx, num_kv_heads, head_dim),
        "v": rnd(batch, ctx, num_kv_heads, head_dim),
        "seq_lens": seq_lens,
        "sm_scale": 1.0 / (head_dim ** 0.5),
    }


ATTENTION_DECODE = SlotSpec(
    name="attention.decode",
    entry="attention_decode",
    summary=(
        "decode attention: each request's query attends to its first seq_lens[i] cached k/v;  "
        "q:(B,Hq,D) k,v:(B,S,Hkv,D) seq_lens:(B,) -> o:(B,Hq,D);  entry(q, k, v, seq_lens, sm_scale, out)"
    ),
    kind="block",
    make_inputs=_decode_attn_inputs,
    out_shapes=lambda i: [tuple(i["q"].shape)],
    invoke_reference=lambda i: [_decode_attn_reference(i["q"], i["k"], i["v"], i["seq_lens"], i["sm_scale"])],
    invoke_entry=lambda entry, i, outs, prepared: entry(i["q"], i["k"], i["v"], i["seq_lens"], i["sm_scale"], outs[0]),
    shapes=(
        {"batch": 4, "num_q_heads": 8, "num_kv_heads": 8, "head_dim": 64, "ctx": 16},
        {"batch": 2, "num_q_heads": 8, "num_kv_heads": 2, "head_dim": 128, "ctx": 32},   # GQA
        {"batch": 8, "num_q_heads": 16, "num_kv_heads": 16, "head_dim": 128, "ctx": 64},
        {"batch": 3, "num_q_heads": 8, "num_kv_heads": 1, "head_dim": 128, "ctx": 48},   # MQA
    ),
    correctness=Correctness("matched_ratio", min_ratio=0.99),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Slot (BLOCK, prepare+forward): moe.fused_experts   (the headroom slot)
#   prepare(w13, w2) -> prepared              (weight layout; runs ONCE at load)
#   forward(x, topk_ids, topk_weights, prepared, out)   (per step)
#   x:(M,H)  w13:(E,2I,H)[gate;up]  w2:(E,H,I)  topk_ids/weights:(M,K) -> out:(M,H)
#   SwiGLU-MLP experts: out = sum_k topk_w * (silu(gate)*up) @ w2.T over each token's
#   top-k experts. This is the slot the GPT-OSS sm120 MoE win fits: the W13 repack +
#   FP4 block-scale interleave are `prepare`; the fused CUTLASS call is `forward`. The
#   pure-torch example reorders [gate;up]->[up;gate] in prepare to prove the contract; a
#   real Blackwell submission carries MXFP4 weights + scales and calls flashinfer
#   cutlass_fused_moe in forward (a sibling moe.fused_experts_mxfp4 slot, sm_100/sm_120).
# ---------------------------------------------------------------------------


def _moe_reference(x, w13, w2, topk_ids, topk_weights):
    # x:(M,H) w13:(E,2I,H)[gate;up] w2:(E,H,I) topk_ids:(M,K) topk_weights:(M,K) -> (M,H)
    M, H = x.shape
    I = w13.shape[1] // 2
    K = topk_ids.shape[1]
    x32 = x.float()
    out = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    for k in range(K):
        e = topk_ids[:, k].long()
        wk = topk_weights[:, k].float()
        w13_e = w13[e].float()                          # (M,2I,H)
        w2_e = w2[e].float()                            # (M,H,I)
        fc1 = torch.einsum("mh,mih->mi", x32, w13_e)    # (M,2I)
        gate, up = fc1[:, :I], fc1[:, I:]
        act = F.silu(gate) * up                         # (M,I)
        out += wk[:, None] * torch.einsum("mi,mhi->mh", act, w2_e)
    return out


def _moe_gptoss_reference(x, w13, w2, topk_ids, topk_weights, *, alpha=1.702, beta=1.0, limit=7.0):
    # GPT-OSS expert MLP: a CLAMPED gated-SiLU (not plain silu(gate)*up). w13:(E,2I,H)
    # rows [gate; up], w2:(E,H,I). act = clamp(gate)*sigmoid(alpha*clamp(gate))*(clamp(up)+beta).
    # This is the high-precision reference the MXFP4 cutlass kernel is gated against.
    M, H = x.shape
    I = w13.shape[1] // 2
    K = topk_ids.shape[1]
    x32 = x.float()
    out = torch.zeros(M, H, device=x.device, dtype=torch.float32)
    lim = torch.tensor(float(limit), device=x.device)
    for k in range(K):
        e = topk_ids[:, k].long()
        wk = topk_weights[:, k].float()
        w13_e = w13[e].float()                         # (M,2I,H)
        w2_e = w2[e].float()                           # (M,H,I)
        fc1 = torch.einsum("mh,mih->mi", x32, w13_e)   # (M,2I)
        gate, up = fc1[:, :I], fc1[:, I:]
        gate_c = torch.minimum(gate, lim)
        up_c = torch.clamp(up, -float(limit), float(limit))
        act = gate_c * torch.sigmoid(gate_c * alpha) * (up_c + beta)  # (M,I)
        out += wk[:, None] * torch.einsum("mi,mhi->mh", act, w2_e)
    return out


def _moe_inputs(*, num_tokens: int, num_experts: int, hidden: int, inter: int, topk: int,
                dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * scale).to(dtype)

    ids = torch.randint(0, num_experts, (num_tokens, topk), generator=g, device=device).to(torch.int32)
    scores = torch.rand(num_tokens, topk, generator=g, device=device)
    weights = (scores / scores.sum(dim=1, keepdim=True)).to(torch.float32)  # normalize per token
    return {
        "x": rnd(num_tokens, hidden, scale=0.1),
        "w13": rnd(num_experts, 2 * inter, hidden, scale=0.05),
        "w2": rnd(num_experts, hidden, inter, scale=0.05),
        "topk_ids": ids,
        "topk_weights": weights,
    }


MOE_FUSED_EXPERTS = SlotSpec(
    name="moe.fused_experts",
    entry="fused_experts",
    prepare="prepare",
    summary=(
        "fused MoE experts — a (prepare, forward) PAIR.  prepare(w13, w2) -> prepared "
        "(weight layout, once at load);  forward(x, topk_ids, topk_weights, prepared, out).  "
        "x:(M,H) w13:(E,2I,H)[gate;up] w2:(E,H,I) -> out:(M,H);  SwiGLU-MLP experts."
    ),
    kind="block",
    make_inputs=_moe_inputs,
    out_shapes=lambda i: [(i["x"].shape[0], i["x"].shape[1])],
    invoke_reference=lambda i: [_moe_reference(i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"])],
    invoke_prepare=lambda prepare_fn, i: prepare_fn(i["w13"], i["w2"]),
    prepare_from_layer=lambda layer: (layer.w13_weight.data, layer.w2_weight.data),
    invoke_entry=lambda entry, i, outs, prepared: entry(i["x"], i["topk_ids"], i["topk_weights"], prepared, outs[0]),
    shapes=(
        {"num_tokens": 4, "num_experts": 8, "hidden": 256, "inter": 128, "topk": 2},
        {"num_tokens": 16, "num_experts": 32, "hidden": 512, "inter": 256, "topk": 4},
        {"num_tokens": 8, "num_experts": 4, "hidden": 384, "inter": 192, "topk": 1},
        {"num_tokens": 33, "num_experts": 16, "hidden": 320, "inter": 160, "topk": 4},
    ),
    # A real fused-MoE kernel runs in fp8/fp4 with reordered reductions -> not bit-exact;
    # gate on a matched ratio vs the fp32 reference, calibrated to the stock noise floor.
    correctness=Correctness("matched_ratio", min_ratio=0.97),
    tolerances=_BF16_TOL,
)


MOE_FUSED_EXPERTS_MXFP4 = SlotSpec(
    name="moe.fused_experts_mxfp4",
    entry="fused_experts_mxfp4",
    prepare="prepare",
    summary=(
        "MXFP4 fused MoE experts — a (prepare, forward) PAIR for GPT-OSS/Blackwell-style "
        "expert kernels.  prepare(w13, w2) owns weight/scale layout (repack [gate;up]->[up;gate], "
        "pack MXFP4, interleave scales) once at load;  forward(x, topk_ids, topk_weights, prepared, "
        "out) MXFP8-quantizes x and runs the fused CUTLASS call.  Gated against the GPT-OSS clamped "
        "gated-SiLU reference; matched_ratio tolerance calibrated to MXFP4/MXFP8 quant error."
    ),
    kind="block",
    make_inputs=_moe_inputs,
    out_shapes=lambda i: [(i["x"].shape[0], i["x"].shape[1])],
    invoke_reference=lambda i: [_moe_gptoss_reference(i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"])],
    # Verify: synthetic dense [gate; up] block weights, no biases (a clean kernel-vs-
    # reference check). The dict shape matches the live prepare_from_layer call.
    invoke_prepare=lambda prepare_fn, i: prepare_fn({"w13": i["w13"], "w2": i["w2"]}),
    # Live seam (gpt-oss): hand prepare the dequantized bf16 experts + biases; rows are
    # HF-interleaved [gate0, up0, ...] so prepare de-interleaves to CUTLASS [up; gate].
    prepare_from_layer=lambda layer: (
        {
            "w13": layer.w13_weight.data,
            "w2": layer.w2_weight.data,
            "w13_bias": layer.w13_weight_bias.data,
            "w2_bias": layer.w2_weight_bias.data,
            "interleaved": True,
        },
    ),
    invoke_entry=lambda entry, i, outs, prepared: entry(i["x"], i["topk_ids"], i["topk_weights"], prepared, outs[0]),
    # GPT-OSS-flavored, CUTLASS-MXFP4-valid dims (hidden 2880; intermediate a 32-block
    # multiple). Small E so verify stays light. The fp4 kernel is NOT bit-exact, so the
    # gate is matched_ratio vs the fp32 reference at a quant-calibrated tolerance.
    shapes=(
        {"num_tokens": 16, "num_experts": 8, "hidden": 2880, "inter": 736, "topk": 4},
        {"num_tokens": 4, "num_experts": 4, "hidden": 2880, "inter": 736, "topk": 2},
    ),
    # Low-bit fidelity: cosine vs the fp32 reference (element-wise tolerance is
    # meaningless at ~6-12% per-element fp4 error). min_cosine calibrated below; the
    # rel-norm guard catches a kernel that gets direction right but energy wrong.
    correctness=Correctness("cosine", min_cosine=0.97, max_rel_norm_err=0.0),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

SLOTS: dict[str, SlotSpec] = {
    SILU_AND_MUL.name: SILU_AND_MUL,
    RMSNORM.name: RMSNORM,
    ATTENTION_SDPA.name: ATTENTION_SDPA,
    ATTENTION_DECODE.name: ATTENTION_DECODE,
    MOE_FUSED_EXPERTS.name: MOE_FUSED_EXPERTS,
    MOE_FUSED_EXPERTS_MXFP4.name: MOE_FUSED_EXPERTS_MXFP4,
}


def get_slot(name: str) -> SlotSpec:
    try:
        return SLOTS[name]
    except KeyError:
        known = ", ".join(sorted(SLOTS)) or "(none)"
        raise KeyError(f"unknown slot {name!r}; known slots: {known}") from None


def list_slots() -> list[str]:
    return sorted(SLOTS)
