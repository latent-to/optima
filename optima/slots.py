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
the result, and ``entry`` (forward) consumes it each step as ``prepared``. A quantized
fused-MoE (repack the expert weights, interleave the FP4 block scales, then a fused
GEMM) fits *one* slot this way: the repack/interleave is ``prepare``, the kernel is
``forward``.

Adding a slot is a validator action (a code change here), never a miner action.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
      (FP4/FP8): element-wise tolerance is meaningless when every element carries
      ~6-12% quantization error, but the *direction* (and energy) of the block output
      is preserved — which is what actually drives the model's logits.
    * ``"topk_overlap"`` — for a kernel whose output is a **selection**, not a tensor value
      (an MSA block-score indexer: scores -> the validator takes top-k blocks -> attends). The
      values don't matter, only which top-``top_k`` they pick: the mean per-row overlap
      ``|topk(actual) ∩ topk(expected)| / top_k`` must be >= ``min_overlap``. Element-wise
      cosine/KL are the wrong metric — a kernel can perturb every score (fp8 index-K) as long
      as the SELECTED set matches.

    DESIGN NOTE — this is the *op-correctness* gate, a cheap **sanity** check ("is this
    even computing the slot's function?"), explicitly necessary-but-not-sufficient
    (verify.py). It is NOT the fidelity authority: the load-bearing anti-cheat gate is
    the end-to-end per-token **KL on the model's logits** (optima.eval), which is exactly
    where a temp-0 distributional metric belongs. The op-gate's only job is to never let
    through a kernel computing the WRONG function (e.g. plain SiLU on a swigluoai model:
    cosine 0.45) while never false-failing a kernel the e2e KL gate accepts (a faithful
    low-bit kernel: cosine 0.996). Hence: same-function reference + a validator-owned
    floor, never a per-element bound that the irreducible quant noise alone would trip.
    """

    mode: str = "allclose"  # "allclose" | "matched_ratio" | "cosine" | "topk_overlap"
    min_ratio: float = 1.0
    min_cosine: float = 0.0  # cosine mode: min cosine similarity vs the HP reference
    max_rel_norm_err: float = 0.0  # cosine mode: optional |‖a‖-‖e‖|/‖e‖ guard (0 = off)
    top_k: int = 0  # topk_overlap mode: the K of the selection (e.g. 16 blocks)
    min_overlap: float = 0.0  # topk_overlap mode: required mean per-row set overlap


@dataclass(frozen=True)
class Activation:
    """The gated-MLP activation a model's MoE/FFN uses — a MODEL fact (read from the
    model's config), NOT a miner choice. ``silu`` is the Qwen/Llama default
    ``silu(gate)*up``; ``swigluoai`` is the clamped GPT-OSS / MiniMax-M3 form
    ``g=min(gate,limit); u=clamp(up,-limit,limit); g*sigmoid(alpha*g)*(u+1)``."""

    kind: str = "silu"  # "silu" | "swigluoai"
    alpha: float = 1.702  # swigluoai sigmoid gain (config swiglu_alpha)
    limit: float = 7.0  # swigluoai clamp (config swiglu_limit)


_SILU = Activation("silu")


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
    # Collective slots (kind="collective") are verified DISTRIBUTED, so the single-process
    # invoke_reference/invoke_entry don't apply. These two hooks let optima.verify_collective
    # drive ANY collective slot (a bare all-reduce, OR a block that OWNS its trailing reduce
    # like moe.fused_experts_reduce) without hard-coding one contract:
    #   * collective_partial(inputs, prepared) -> the fp32 per-rank tensor whose cross-rank
    #     SUM is the trusted reference (x for all-reduce; the local experts' fp32 output for
    #     the MoE-overlap block).
    #   * invoke_collective(entry, inputs, out, group, prepared) -> call the miner kernel,
    #     handing it the process group; it fills `out` with the REDUCED result.
    collective_partial: Optional[Callable] = None
    invoke_collective: Optional[Callable] = None
    # Per-slot end-to-end KL gate, calibrated to THIS slot's intrinsic noise floor (the
    # generic 5e-3 default is tuned for elementwise ops; attention sits ~6e-3 vs flash's
    # reordered softmax, so a flat 5e-3 false-fails a faithful attention kernel — README
    # calibration finding 6). None -> use the eval's generic threshold.
    kl_threshold: Optional[float] = None

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
    # Attention's intrinsic end-to-end KL floor (~6e-3 vs flash) is above the generic
    # 5e-3 gate; calibrate to ~5x the floor so a faithful attention kernel isn't false-failed.
    kl_threshold=3e-2,
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
    kl_threshold=3e-2,  # attention's higher intrinsic floor (see attention.sdpa)
)


# ---------------------------------------------------------------------------
# Slot (BLOCK, prepare+forward): moe.fused_experts
#   prepare(w13, w2) -> prepared              (weight layout; runs ONCE at load)
#   forward(x, topk_ids, topk_weights, prepared, out)   (per step)
#   x:(M,H)  w13:(E,2I,H)[gate;up]  w2:(E,H,I)  topk_ids/weights:(M,K) -> out:(M,H)
#   SwiGLU-MLP experts: out = sum_k topk_w * (silu(gate)*up) @ w2.T over each token's
#   top-k experts. The (prepare, forward) split is what lets a quantized / layout-
#   sensitive expert kernel fit one slot: a weight repack / FP4 block-scale interleave
#   is `prepare`, the fused GEMM is `forward`. The pure-torch example reorders
#   [gate;up]->[up;gate] in prepare to exercise the contract.
# ---------------------------------------------------------------------------


def _gated_activation(gate: torch.Tensor, up: torch.Tensor, act: Activation) -> torch.Tensor:
    """The fc1 -> intermediate activation. ``act`` (a MODEL fact) selects the form so the
    HP reference matches the model the kernel targets — using SiLU as the reference for a
    swigluoai model is the ratio-0.0 false-fail this fixes."""
    if act.kind == "swigluoai":
        g = gate.clamp(max=act.limit)
        u = up.clamp(min=-act.limit, max=act.limit)
        return g * torch.sigmoid(act.alpha * g) * (u + 1.0)
    return F.silu(gate) * up


def _moe_reference(x, w13, w2, topk_ids, topk_weights, act: Activation = _SILU):
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
        act_out = _gated_activation(gate, up, act)      # (M,I)
        out += wk[:, None] * torch.einsum("mi,mhi->mh", act_out, w2_e)
    return out


def _moe_prepare_args_from_layer(layer):
    """Map a LIVE sglang FusedMoE layer to the miner ``prepare()`` call shape — the
    validator-owned layer->contract mapping. Live-eval seam only (``dispatch._moe_prepared``);
    ``optima verify`` uses ``invoke_prepare`` on synthetic weights and never calls this.

    * DENSE layer -> ``(w13_weight.data, w2_weight.data)`` — identical to the old default; the
      miner's prepare reorders/repacks the two tensors.
    * QUANTIZED layer (NVFP4 ``ModelOptNvFp4FusedMoEMethod``, weights are packed uint8) ->
      ``("nvfp4_layer", layer)``: the validator hands the miner's prepare the LIVE layer. A
      quantized kernel's weight layout is *kernel-specific* (the flashinfer CuteDSL v2 path wants
      a ``CuteDslMoEWrapper`` + [Up,Gate]-interleaved weights + MMA-layout block-scales +
      scalarized scales; a cutlass kernel wants something else), so the per-kernel transform
      belongs in the miner's prepare, which builds it ONCE — reusing the model runtime's own
      prepared state (e.g. sglang ``ensure_cutedsl_wrapper``) rather than re-deriving the fragile
      scale algebra. This keeps the generic slot free of any one kernel's layout while still
      giving the miner everything the quantized weights need (the dense 2-tuple omits the scales).

    hasattr-guarded so a dense (or unrecognized) layer always falls back to the dense 2-tuple."""
    w13 = getattr(getattr(layer, "w13_weight", None), "data", None)
    is_quant = getattr(w13, "dtype", None) == torch.uint8 and (
        getattr(layer, "w13_weight_scale", None) is not None
        or getattr(layer, "g1_alphas", None) is not None
    )
    if is_quant:
        return ("nvfp4_layer", layer)
    return (layer.w13_weight.data, layer.w2_weight.data)  # dense — unchanged default


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
    prepare_from_layer=_moe_prepare_args_from_layer,
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


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Slot (COLLECTIVE): collective.all_reduce   (the TP comms waist)
#   x:(M,H) on each of `world_size` ranks -> out:(M,H) = sum over ranks.
#   contract: entry(x, out, group)  — miner owns the reduce algorithm + transport;
#   validator owns `out`, the process group, and the call site.
#
# Unlike op/block slots, a collective spans GPUs: the kernel needs the TP process group
# to move data across ranks, so it is verified DISTRIBUTED (optima.verify_collective,
# NOT verify_entry) against the trusted fp32 cross-rank sum. The reduce is mid-network
# (upstream of the sampler) — no output to substitute. Decode is comms-bound (~32–43%
# of GPU time at TP/EP scale, the largest single category), and it is *latency*-bound,
# so the lever is a lower-latency reduce or compute-comm overlap — both expressible here
# while staying inside the four invariants. WIDER SURFACE: handing the miner the
# communicator is more capability than "fill a tensor"; the invariants still bound it,
# but distributed verify + the end-to-end gate are MANDATORY (docs/SLOT_CONTRACT.md).
# ---------------------------------------------------------------------------


def _all_reduce_inputs(*, num_tokens: int, hidden: int, dtype: torch.dtype, device: str,
                       seed: int, rank: int = 0, world_size: int = 1) -> dict:
    # Each rank gets a DIFFERENT partial (seeded by rank); the all-reduce sums them.
    g = torch.Generator(device=device).manual_seed(seed + 1_000_003 * rank)
    x = (torch.randn(num_tokens, hidden, generator=g, device=device, dtype=torch.float32) * 0.1).to(dtype)
    return {"x": x}


COLLECTIVE_ALL_REDUCE = SlotSpec(
    name="collective.all_reduce",
    entry="all_reduce",
    summary=(
        "TP all-reduce (the comms waist): x:(M,H) per rank -> out:(M,H) = sum over ranks;  "
        "entry(x, out, group).  Validator owns out + the process group; verified DISTRIBUTED "
        "vs the fp32 cross-rank sum (optima.verify_collective)."
    ),
    kind="collective",
    make_inputs=_all_reduce_inputs,
    out_shapes=lambda i: [tuple(i["x"].shape)],
    # Collectives are verified distributed: the real reference is the fp32 sum ACROSS
    # ranks, which a single-process invoke_reference can't compute. These two are unused
    # for kind="collective" (kept non-None to satisfy the dataclass); verify_collective
    # drives the real verification.
    invoke_reference=lambda i: [i["x"]],
    invoke_entry=lambda entry, i, outs, prepared: entry(i["x"], outs[0], i.get("__group__")),
    # Distributed-verify hooks: the reference is the fp32 SUM of each rank's x.
    collective_partial=lambda i, prepared: i["x"].float(),
    invoke_collective=lambda entry, i, out, group, prepared: entry(i["x"], out, group),
    shapes=(
        {"num_tokens": 1, "hidden": 4096},
        {"num_tokens": 8, "hidden": 4096},
        {"num_tokens": 128, "hidden": 7168},
    ),
    # A different reduce algorithm/order (one-shot/NVLS/tree vs ring) is not bit-exact;
    # gate on matched_ratio vs the fp32 sum, with the end-to-end token/KL gate mandatory.
    correctness=Correctness("matched_ratio", min_ratio=0.99),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Slot (COLLECTIVE block — owns its trailing reduce): moe.fused_experts_reduce
#   prepare(w13, w2) -> prepared                                  (once at load)
#   forward(x, topk_ids, topk_weights, prepared, out, group)      (per step)
#   x:(M,H) per rank -> out:(M,H) = SUM_over_ranks( local_experts(x) )
#
# This is the fix for the structural ceiling: the decode win is the OVERLAP of the
# expert GEMM with the trailing TP all-reduce (~75% of decode at TP/EP scale), and a
# plain moe.fused_experts slot can't express it because the validator replays a SEPARATE
# stock all-reduce after the kernel — the two ops are severed. Here ONE kernel owns BOTH
# the experts AND the reduce (it is handed the process group), so it can fuse/overlap them.
# The validator does NOT replay the reduce. Wider capability -> verified DISTRIBUTED vs the
# fp32 cross-rank sum of the per-rank expert outputs, and the end-to-end gate is mandatory.
# Still inside the four invariants: validator owns out + the group + the call site; the
# reduced output feeds the residual stream upstream of the sampler (nothing to substitute).
# ---------------------------------------------------------------------------


def _moe_reduce_inputs(*, num_tokens: int, num_experts: int, hidden: int, inter: int, topk: int,
                       dtype: torch.dtype, device: str, seed: int, rank: int = 0, world_size: int = 1) -> dict:
    # Tokens + routing are REPLICATED across ranks (seeded without rank), so every rank
    # runs the same tokens; the expert WEIGHTS are SHARDED (seeded WITH rank), so each
    # rank computes a different partial and the cross-rank reduce does real work.
    gx = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(num_tokens, hidden, generator=gx, device=device, dtype=torch.float32) * 0.1).to(dtype)
    ids = torch.randint(0, num_experts, (num_tokens, topk), generator=gx, device=device).to(torch.int32)
    scores = torch.rand(num_tokens, topk, generator=gx, device=device)
    weights = (scores / scores.sum(dim=1, keepdim=True)).to(torch.float32)
    gw = torch.Generator(device=device).manual_seed(seed + 1_000_003 * rank)
    w13 = (torch.randn(num_experts, 2 * inter, hidden, generator=gw, device=device, dtype=torch.float32) * 0.05).to(dtype)
    w2 = (torch.randn(num_experts, hidden, inter, generator=gw, device=device, dtype=torch.float32) * 0.05).to(dtype)
    return {"x": x, "w13": w13, "w2": w2, "topk_ids": ids, "topk_weights": weights}


MOE_FUSED_EXPERTS_REDUCE = SlotSpec(
    name="moe.fused_experts_reduce",
    entry="fused_experts_reduce",
    prepare="prepare",
    summary=(
        "fused MoE experts that OWN the trailing TP all-reduce (the compute-comm overlap "
        "lever).  prepare(w13, w2) -> prepared;  "
        "forward(x, topk_ids, topk_weights, prepared, out, group) fills out with the "
        "SUM-over-ranks of the local expert output.  x:(M,H) -> out:(M,H);  verified DISTRIBUTED."
    ),
    kind="collective",
    make_inputs=_moe_reduce_inputs,
    out_shapes=lambda i: [(i["x"].shape[0], i["x"].shape[1])],
    # Single-process invoke_reference/entry are unused for kind="collective"; the real
    # reference is the cross-rank fp32 sum (collective_partial), driven by verify_collective.
    invoke_reference=lambda i: [_moe_reference(i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"])],
    invoke_entry=lambda entry, i, outs, prepared: None,
    invoke_prepare=lambda prepare_fn, i: prepare_fn(i["w13"], i["w2"]),
    prepare_from_layer=_moe_prepare_args_from_layer,
    # Reference partial = this rank's fp32 expert output (HP, from the RAW weights, NOT the
    # miner's `prepared`); the trusted cross-rank SUM is the full MoE output.
    collective_partial=lambda i, prepared: _moe_reference(
        i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"]).float(),
    invoke_collective=lambda entry, i, out, group, prepared: entry(
        i["x"], i["topk_ids"], i["topk_weights"], prepared, out, group),
    shapes=(
        {"num_tokens": 4, "num_experts": 8, "hidden": 256, "inter": 128, "topk": 2},
        {"num_tokens": 16, "num_experts": 32, "hidden": 512, "inter": 256, "topk": 4},
        {"num_tokens": 8, "num_experts": 4, "hidden": 384, "inter": 192, "topk": 1},
    ),
    correctness=Correctness("matched_ratio", min_ratio=0.97),
    tolerances=_BF16_TOL,
)


# ---------------------------------------------------------------------------
# Slot (BLOCK): attention.msa_block_score   (the MSA / sparse-attention indexer)
#   q:(B,Hq,D)  index_k:(B,S,1,D)  seq_lens:(B,)  block_size -> block_scores:(B, S//block_size)
#   contract: entry(q, index_k, seq_lens, block_size, out)
#
# The FINER-than-attention.decode seam for a SELECTION win (the fp8 MSA indexer, M3). The kernel
# computes per-128-token-block SCORES (block-max-pool of the index QK); the validator owns the
# irreducible downstream step — the top-k block SELECTION and the bf16 attend over the chosen
# blocks. So the kernel stays strictly upstream of the sampler (a wrong score just mis-selects,
# caught by the gate + e2e KL). The output is a SELECTION, gated on `topk_overlap` (top-k block
# SETS agree vs the bf16 reference), NOT cosine/KL: an fp8 index-K may perturb every score yet
# pick the same blocks. Reusable pattern: finer seam + set-metric + validator-owns-the-step. The
# live seam (the MSA backend's score kernel) is GPU/M3-specific — see integrations/sglang_msa.py.
# ---------------------------------------------------------------------------

_MSA_TOPK = 8  # the block-selection K this slot's correctness checks (<= every shape's n_blocks)


def _msa_block_score_reference(q, index_k, seq_lens, block_size):
    # q:(B,Hq,D) index_k:(B,S,1,D) seq_lens:(B,) -> (B, S//block_size) fp32 block-max-pool of QK.
    B, Hq, D = q.shape
    S = index_k.shape[1]
    nblk = S // block_size
    q32 = q.float().sum(dim=1)                       # (B,D): sum over index q-heads (1 shared idx-k head)
    k32 = index_k.float()[:, :, 0, :]                # (B,S,D)
    scores = torch.einsum("bd,bsd->bs", q32, k32)    # (B,S) per-token index QK
    sidx = torch.arange(S, device=q.device).view(1, S)
    scores = scores.masked_fill(sidx >= seq_lens.view(B, 1), float("-inf"))  # mask beyond context
    return scores.view(B, nblk, block_size).amax(dim=-1)  # (B, nblk) block-max-pool


def _msa_inputs(*, batch: int, num_q_heads: int, head_dim: int, ctx: int, block_size: int,
                dtype: torch.dtype, device: str, seed: int) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(dtype)

    # Keep ctx a clean multiple of block_size with n_blocks comfortably ABOVE _MSA_TOPK,
    # robust to count-dim jitter. n_blocks == top_k would make the gate vacuous (top-k of
    # k blocks selects everything — any output, even all-zeros, scores overlap 1.0), so
    # the floor keeps at least 4 distractor blocks the selection can get wrong.
    nblk = max(_MSA_TOPK + 4, ctx // block_size)
    ctx = nblk * block_size
    seq_lens = torch.randint(_MSA_TOPK * block_size, ctx + 1, (batch,), generator=g, device=device).to(torch.int32)
    seq_lens[0] = ctx  # one full-length request
    return {
        "q": rnd(batch, num_q_heads, head_dim),
        "index_k": rnd(batch, ctx, 1, head_dim),
        "seq_lens": seq_lens,
        "block_size": block_size,
    }


ATTENTION_MSA_BLOCK_SCORE = SlotSpec(
    name="attention.msa_block_score",
    entry="msa_block_score",
    summary=(
        "MSA indexer block scores: q:(B,Hq,D) index_k:(B,S,1,D) seq_lens:(B,) block_size -> "
        "block_scores:(B,S//block_size) = block-max-pool of the index QK.  "
        "entry(q, index_k, seq_lens, block_size, out).  The validator owns the top-k block "
        "SELECTION + the attend; gated on topk_overlap (the SELECTED set), not score values."
    ),
    kind="block",
    make_inputs=_msa_inputs,
    out_shapes=lambda i: [(i["q"].shape[0], i["index_k"].shape[1] // i["block_size"])],
    invoke_reference=lambda i: [_msa_block_score_reference(i["q"], i["index_k"], i["seq_lens"], i["block_size"])],
    invoke_entry=lambda entry, i, outs, prepared: entry(
        i["q"], i["index_k"], i["seq_lens"], i["block_size"], outs[0]),
    shapes=(
        # Every shape keeps n_blocks > _MSA_TOPK (=8): at n_blocks == top_k the overlap
        # gate is vacuous (any selection of 8-of-8 blocks scores 1.0).
        {"batch": 4, "num_q_heads": 4, "head_dim": 128, "ctx": 1536, "block_size": 128},   # 12 blocks
        {"batch": 2, "num_q_heads": 4, "head_dim": 128, "ctx": 2048, "block_size": 128},   # 16 blocks
        {"batch": 8, "num_q_heads": 8, "head_dim": 64, "ctx": 1536, "block_size": 128},    # 12 blocks
        {"batch": 3, "num_q_heads": 4, "head_dim": 128, "ctx": 4096, "block_size": 128},   # 32 blocks
    ),
    # The output is a SELECTION: gate on the top-k block SETS agreeing, not the score values.
    # 7/8 tolerates a thin selection drift (e.g. fp8 index-K flipping a borderline block).
    correctness=Correctness("topk_overlap", top_k=_MSA_TOPK, min_overlap=0.875),
    tolerances=_BF16_TOL,
    kl_threshold=3e-2,  # attention's higher intrinsic floor (this rides the attention path)
)


SLOTS: dict[str, SlotSpec] = {
    SILU_AND_MUL.name: SILU_AND_MUL,
    RMSNORM.name: RMSNORM,
    ATTENTION_SDPA.name: ATTENTION_SDPA,
    ATTENTION_DECODE.name: ATTENTION_DECODE,
    ATTENTION_MSA_BLOCK_SCORE.name: ATTENTION_MSA_BLOCK_SCORE,
    MOE_FUSED_EXPERTS.name: MOE_FUSED_EXPERTS,
    MOE_FUSED_EXPERTS_REDUCE.name: MOE_FUSED_EXPERTS_REDUCE,
    COLLECTIVE_ALL_REDUCE.name: COLLECTIVE_ALL_REDUCE,
}


def get_slot(name: str) -> SlotSpec:
    try:
        return SLOTS[name]
    except KeyError:
        known = ", ".join(sorted(SLOTS)) or "(none)"
        raise KeyError(f"unknown slot {name!r}; known slots: {known}") from None


def list_slots() -> list[str]:
    return sorted(SLOTS)


# ---------------------------------------------------------------------------
# Per-model slot policy — the VALIDATOR-OWNED specialization.
#
# A slot's default (above) is the generic case (SiLU experts, matched_ratio). But the
# *activation*, *quant format*, and the *correctness floor* for a given (model, slot) are
# MODEL/VALIDATOR facts, never miner choices: the validator controls the model it serves,
# reads swiglu_alpha/limit from its config, and calibrates the floor to the measured noise.
# A miner only NAMES which model it targets; the numbers below are the validator's.
#
# This is the precursor to a full per-model arena registry (docs: arenas). When that lands,
# these profiles move there; today it is the one validator-owned table that makes the MoE
# slot verifiable on a swigluoai/NVFP4 model (e.g. MiniMax-M3) without weakening the generic
# slot or letting a submission set its own gate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotProfile:
    """Validator-owned (model, slot) overrides. ``activation`` retargets the HP reference
    to the model's real activation; ``correctness`` (optional) swaps the op-sanity metric
    (e.g. cosine for a low-bit kernel). None fields keep the generic slot default."""

    activation: Activation = field(default_factory=Activation)
    correctness: Optional[Correctness] = None


_MOE_SLOTS = ("moe.fused_experts", "moe.fused_experts_reduce")


def specialize_slot(slot: SlotSpec, profile: SlotProfile) -> SlotSpec:
    """Return a copy of ``slot`` retargeted by a validator ``profile``. Only rebinds the
    pieces the profile changes (the activation-bearing references + the correctness policy);
    everything else — inputs, shapes, seam wiring — is untouched. The module-level slot
    singletons are never mutated, so ``get_slot`` stays generic."""
    repl: dict = {}
    if slot.name in _MOE_SLOTS:
        act = profile.activation

        def _ref(i, _act=act):
            return [_moe_reference(i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"], _act)]

        repl["invoke_reference"] = _ref
        if slot.collective_partial is not None:  # the reduce block's distributed reference
            def _partial(i, prepared, _act=act):
                return _moe_reference(i["x"], i["w13"], i["w2"], i["topk_ids"], i["topk_weights"], _act).float()

            repl["collective_partial"] = _partial
    if profile.correctness is not None:
        repl["correctness"] = profile.correctness
    return replace(slot, **repl) if repl else slot


_M3_MOE_PROFILE = SlotProfile(
    activation=Activation("swigluoai", alpha=1.702, limit=7.0),
    # Low-bit (NVFP4) experts: gate on cosine vs the same-function fp32 reference.
    # min_cosine = the measured NVFP4 representational floor (0.9958 at M3 shape,
    # m3_swigluoai_gate.py) with headroom; plain-SiLU scores 0.45 and is rejected.
    # No norm guard yet (max_rel_norm_err uncalibrated — TODO measure the floor).
    correctness=Correctness("cosine", min_cosine=0.985),
)

# model key (as a miner may declare it / as the validator keys its served model) -> {slot: profile}
MODEL_PROFILES: dict[str, dict[str, SlotProfile]] = {
    "MiniMax-M3": {
        # BOTH experts slots run the same swigluoai experts on M3 — the reduce-owning
        # block (the overlap target) just also owns the trailing all-reduce, and
        # specialize_slot retargets its distributed reference (collective_partial) too.
        # Registering only the plain slot would verify an M3 reduce kernel against a
        # SiLU reference and false-fail every honest submission.
        "moe.fused_experts": _M3_MOE_PROFILE,
        "moe.fused_experts_reduce": _M3_MOE_PROFILE,
    },
}
# NVFP4 builds carry a "-NVFP4" suffix in their declared model id; alias them.
MODEL_PROFILES["MiniMax-M3-NVFP4"] = MODEL_PROFILES["MiniMax-M3"]


def model_profile(model_key: Optional[str], slot_name: str) -> Optional[SlotProfile]:
    if not model_key:
        return None
    return MODEL_PROFILES.get(model_key, {}).get(slot_name)


def slot_for_model(slot_name: str, model_key: Optional[str] = None) -> SlotSpec:
    """``get_slot`` + the validator's per-model specialization. With no model key (or no
    registered profile) this is exactly ``get_slot`` — the generic slot — so existing
    callers and bundles are unchanged."""
    slot = get_slot(slot_name)
    prof = model_profile(model_key, slot_name)
    return specialize_slot(slot, prof) if prof else slot
