# The slot contract (the narrow waist)

> The one thing that must never change. Everything *around* it — which slots exist, which
> sglang chokepoints we patch, correctness modes, eval knobs — is free to churn. This page is
> the invariant that keeps the subnet cheat-resistant, written as a checklist. **Before adding
> a slot or extending `SlotSpec`, check a submission against the four rules. If it can't satisfy
> all four, it is not a core slot** — it goes in the fenced escape hatch (bottom), or not at all.

> **Miners:** you don't need this page to compete — see [MINER_GUIDE.md](MINER_GUIDE.md).
> The one rule that shapes how you write a kernel: you only ever *fill an output tensor the
> validator allocated*, strictly upstream of the sampler (rules 1–2). That's why you never
> return tensors and never see the final tokens. The rest is the validator's side.

## Why a "waist"

Long-lived extensible systems are hourglasses: a *thin, stable, minimal* waist, with a
proliferation of things above it (slots) and below it (sglang seam adapters) that change
freely. The waist's job is to be **boring and ossified** — that is the feature, not a bug
(cf. the Internet's IP layer, Unix syscalls, the PyTorch dispatcher contract). We keep the
waist thin so the "seemingly impossible" submission becomes "add a slot / add an adapter /
stage it in the escape hatch," never "rewrite everything."

## The four invariants (inviolable)

1. **Validator owns the boundary.** The validator allocates every output tensor
   (shape/dtype/device/stride) and owns the call site. The miner kernel only *fills*
   validator-allocated outputs; it never allocates or returns the result the model consumes.
2. **Strictly upstream of the sampler.** A slot sits above the logprobs/sampler. Its output
   feeds the residual stream → downstream layers → sampler, all stock. There is therefore no
   final output to substitute — this is what defeats the "run gibberish, fetch the real answer
   from an API" attack.
3. **Gated against high-precision ground truth.** Correctness is measured against an fp32 /
   dequant reference, **never against the stock kernel**. A faster, slightly-different kernel
   can pass; a wrong one cannot. The *metric* may vary per slot (`allclose` / `matched_ratio`
   / `cosine`); the *reference* is always HP ground truth.
4. **No trust in miner-reported numbers.** The score never trusts a value the miner's code
   could have produced. Throughput is timed out-of-process (the driver never imports miner
   code); fidelity is judged by the validator from emitted tokens / the HP-reference
   comparison — not the engine's self-reported logprobs when the engine itself was patched.

If a feature preserves all four, it is *bell-work* — safe to add. If it can't, it is not a
core slot.

## Collective slots (`kind="collective"`) — the wider surface

Decode at TP/EP scale is **comms-bound** (the all-reduce / all-to-all is the single largest
category, ~32–43% of GPU time, and it's *latency*-bound). Capturing that lever needs a slot
that spans GPUs — so the kernel is handed the **process group** (a wider capability than the
op/block "fill a tensor" contract). The four invariants still hold (validator owns the output
buffer + the group + the call site; the reduce is mid-network, upstream of the sampler; gated
vs the trusted fp32 cross-rank result; timing is out-of-process). But the larger surface makes
two things **mandatory, not optional**:

- **Distributed verification.** A collective can't be checked on one GPU. `verify_entry`
  *refuses* `kind="collective"`; use `optima.verify_collective.verify_collective`, which spawns
  `world_size` ranks, runs the miner kernel as the real collective, and compares each rank's
  output to a `torch.distributed` reduce of the **fp32** partials (gloo/CPU for a numeric check,
  nccl/GPU for the real path).
- **The end-to-end gate.** Per-collective error compounds across every layer, so the op-level
  cosine/matched_ratio is necessary-but-not-sufficient — the token/KL/accuracy gate is required.

Overlap wins (hide the reduce behind the producer GEMM) are *not* a reduce-only slot — they need
a **block that owns both** (the MoE block owning its trailing reduce; a row-parallel `linear+reduce`
block). Same contract, wider boundary. This is now realized as **`moe.fused_experts_reduce`**: a
(prepare, forward) block handed the process group whose `forward(x, topk_ids, topk_weights,
prepared, out, group)` fills `out` with the *already-reduced* expert output — the validator does
NOT replay a stock all-reduce after it, so the kernel can fuse/overlap the expert GEMM with the
reduce. Verified distributed vs the fp32 cross-rank sum of the per-rank expert outputs
(`optima.verify_collective`, driven by the slot's `collective_partial` / `invoke_collective` hooks).
This lifts the structural ceiling — the plain `moe.fused_experts` slot can't express the overlap
because the reduce is severed onto a separate stock call.

## Graph-safety is part of the contract (you only win with graphs ON)

Scoring runs with **CUDA graphs ON** — graphs-off cripples the baseline ~4.5–6.5×, so a graphs-off
"win" is meaningless and beating sglang/vLLM/TensorRT graphs-on is the entire point. The op seams
(silu/rmsnorm) are graph-captured directly. A **block/collective** kernel must DECLARE
`graph_safe: true` in its metadata to be run inside the graph; otherwise the seam falls back to the
trusted baseline under capture (an un-capturable kernel can't wedge the graph). `graph_safe` means:
static shapes, no host syncs (`.item()`/`.cpu()`), no data-dependent Python control flow, writes only
the validator-allocated buffer. A kernel that lies either errors at capture (→ fallback) or is caught
by the fidelity gate. The attention `decode` gather-MVP is the one seam that is structurally eager (a
per-step `max_len` host-sync); its graph-safe form is a paged-direct contract (the next rung).

## Evolution rules

- **Additive only.** `abi_version` gates the bundle format; new `SlotSpec` fields are
  optional; never break a bundle that parsed before (Linux's "don't break userspace").
- **Seam adapters are version-pinned glue, not the contract.** Each sglang chokepoint
  (`SiluAndMul.forward`, `RadixAttention.forward`, `FusedMoE.forward`, …) is a *lower-bell*
  adapter, re-validated on every `PINNED_SGLANG` bump via `optima compat`. Proliferate them
  freely; they never touch the four invariants.
- **Extraction trigger (do NOT do early).** `SlotSpec` is a dataclass today, and that is
  correct while it's small. When the next slot forces *yet another* optional field/callable —
  or the `lambda` soup stops being readable — extract `Slot` into an interface (one class per
  slot, mirroring sglang's `AttentionBackend` base class). Not before: premature abstraction
  is its own way projects die. The refactor is local — it touches neither the invariants nor
  any miner's bundle.

## The escape hatch (where exotic submissions go FIRST)

Some wins can't be expressed as tensor-in/tensor-out (a backend swap, a source recompile).
Those do **not** get special-cased into the core. They go through the framework / rebuild
tier, which is **experimental and fenced**:

- It runs **only validator-shipped, reviewed patchers** (`rebuild.json` `repo_python` steps) —
  never miner-supplied code (`bundle_python` is rejected). Like PyTorch: you submit a patch to
  core to add a backend; you don't ship arbitrary code into the dispatcher.
- It is gated by the **token / task-accuracy** path plus **no-egress isolation**, because a
  patched engine's self-reported numbers aren't trusted (invariant 4).
- When a pattern recurs there *and* can be made to satisfy the four invariants, **promote it**
  to a clean core slot. The escape hatch is a staging area, not a permanent home.

## What's just churn (don't)

A "robustness PR," a plugin framework, GPU CI, or load-time weight conversion are all
premature until their trigger fires (a real second contributor; a real exotic submission; mem
becoming the bottleneck). Build slots concretely, keep this waist thin, and extract only when
the pattern is real.
