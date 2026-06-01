# The slot contract (the narrow waist)

> The one thing that must never change. Everything *around* it — which slots exist, which
> sglang chokepoints we patch, correctness modes, eval knobs — is free to churn. This page is
> the invariant that keeps the subnet cheat-resistant, written as a checklist. **Before adding
> a slot or extending `SlotSpec`, check a submission against the four rules. If it can't satisfy
> all four, it is not a core slot** — it goes in the fenced escape hatch (bottom), or not at all.

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
