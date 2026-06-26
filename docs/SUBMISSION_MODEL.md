# The submission model — what a miner contributes, and why it becomes *our* library

> Companion to [`SLOT_CONTRACT.md`](SLOT_CONTRACT.md) (the cheat-resistance invariants) and
> [`HOW_OPTIMA_WORKS.md`](HOW_OPTIMA_WORKS.md) (the pipeline). This page answers a *different*
> question: **what may a miner submit, such that every accepted win is portable kernel IP we
> can fold into our own inference library — never an sglang patch we get entrenched on.**
>
> Status: design of record (2026-06-24). The machinery deltas at the bottom are not all built
> yet; where the current code differs, this page is the target.

## Why this page exists

Optima's end goal is **not** "patches to sglang." It is **our own kernel library**
(`optima-kernels`) — portable, inspectable, composable device code — shipped as a product on
top of *any* engine (sglang/vLLM today, our own later). The subnet is the machine that
populates that library: miners are paid to *create* fast kernels, the validator measures
them, and an accepted win drops straight into the library.

That only works if the contribution is **portable IP**. A win expressed as "monkeypatch
sglang's `FusedMoE` to call kernel X" is worthless to the library — it evaporates the moment
we change engines. So the submission model is built around one product axiom and one
judgment test, on top of the four cheat-resistance invariants.

## Axiom 5 — Transferability (sits alongside the four invariants)

> A contribution is **transferable kernel IP**: device code + its launch + its weight-layout
> math, such that it can be lifted into *our own* engine leaving only orchestration behind.
> **A kernel is a kernel** — write it against whatever kernel libraries help it win (CUTLASS /
> CuTe-DSL / Triton / flashinfer / sgl-kernel / DeepGEMM / torch); we don't "avoid" anything.
> What a contribution must NOT *be* is **engine orchestration** — scheduler wiring, the
> MoE-runner registry dance, monkeypatching `forward`. The cut line is concrete:
> **`forward_impl` (the kernel run boundary).** Above it is rented orchestration the validator
> owns; below it is the portable kernel. Engine-specific placement (live-layer→inputs,
> installing at a chokepoint) is the **validator's adapter**, never the miner's bundle.

The four invariants (`SLOT_CONTRACT.md`) keep the subnet honest (the cheat axis). Axiom 5 keeps
the dogfood worth eating (the product axis): it guarantees an accepted kernel is something we can
read, own, and run in our own engine — leaving only orchestration behind. `sglang` is the referee
+ a rented runtime, **not the product foundation.** (Importing `flashinfer`/`sgl-kernel`/cutlass
kernels is fine — those are *kernel libraries*; the thing we exclude is engine glue, enforced by
the contribution *unit* being device-source-at-a-slot, not the import namespace.)

## The Win Test — "did this create what wasn't there?"

A submission scores **only** if it is *new device code that beats the **tuned** blessed base
at equal fidelity* — i.e. it created a capability or a speed the base did not already have.
Three things are explicitly **not** wins, and the design makes them *non-submittable* rather
than leaving them to a judge:

| Not a win | Why | Who owns it instead |
|---|---|---|
| **Engine config / args** (batch, TP/EP, `--*-backend`, mem-fraction, flags) | turning a knob isn't creating a kernel | the validator runs the model at its **best config** — that is the fixed baseline |
| **Integration / rerouting** (wire an existing-but-unwired faster kernel) | calling code that already exists isn't new IP | the validator's adapter; once wired it becomes the new base everyone competes against |
| **Python-level recomposition** (reorder/skip existing launches, no new device code) | no new device code → nothing to fold into the library | the tuned base already does the obvious recompositions |

**Enforced by construction, not adjudication.** The miner surface admits *only device source
bound to a typed slot or override-point*. There is no manifest field for engine config; the
base kernel is validator-owned and shipped **tuned** (its own autotune / constexprs at best
settings); the baseline is that tuned base. So the *only* way to move the score is to write
device code that beats the tuned base — which is exactly "created what wasn't there." A
reroute ties the baseline (no win); a knob isn't expressible; a retile the tuned base already
does ties the baseline. The validator never argues "is this a real win" — **the surface only
admits the kind of thing that can be one.**

This is intentionally *generous at the margin*: a flashinfer-epilogue rewrite like the M3
swigluoai win **is** a win even though it leans on a vendor megakernel, because it is *new
device code*, not an arg tweak. The bar is **"new code that improves,"** not "from-scratch
everything."

## The contribution unit — a portable kernel at a typed boundary

One unit, three granularities. Pick the smallest that expresses the win.

### 1. Override (preferred — and the accessible on-ramp)

The library ships a parametric base kernel with declared, typed **override-points**; the
miner submits **only** the override as portable device source. The M3 swigluoai win is the
archetype: a ~12-line `gemm1_epilogue` against the NVFP4 MoE megakernel — *not* 3.5k LoC of
vendored kernel, *not* an `open()` patch. Override-points are also the **accessibility**
lever: the base does the hard mainloop; the miner writes the small hook, in Triton or
CuTe-DSL, without understanding the grouped-GEMM internals.

**Override-point roadmap** (coverage × ease-of-exposure × *measured* yield):

1. **epilogue / activation** — swiglu/swigluoai/geglu, bias, clamp, scale, requant, fused
   finalize. Highest coverage, cheapest to expose (CUTLASS EVT / Triton parameterization),
   the swigluoai class. **Ship first.**
2. **quant codec** — pack/unpack + scale layout + the scale-factor path the GEMM consumes.
   The format-specialization frontier (beat NVFP4, sub-4-bit) — the *renewable* edge per the
   north star. Requires the served-clean-reference control (below) to be winnable.
3. **prologue / input transform** — gather, dequant-on-load, layout swizzle (the MoE
   expand/gather glue; the paged-KV gather). Medium yield.
4. **tile schedule / mainloop params** — tile shape, swap-AB, occupancy, pipeline depth,
   cluster. **Last, and low-priority on purpose:** across the DSV4 / Qwen3.5 / Nemotron / M3
   campaigns, occupancy/tile "wins" were repeatedly **vendor floors** (nameplate-fiction
   rooflines; 1-CTA/SM *by design*). Expose only for a genuinely occupancy-bound
   launch-window kernel, and gate hard against that false-positive history.

### 2. Whole kernel at a slot (the higher ceiling)

The miner writes the entire kernel for a boundary — a from-scratch attention block, a novel
MoE megakernel, and crucially the **compute↔comm fused block** (`moe.fused_experts_reduce` —
the experts block that owns its trailing all-reduce). This tier is where the *durable*
renewable wins live (fusion across the GEMM/comm seam, ~75% of decode). **Overrides
*augment* a base; whole-kernels *create* one.** Both are first-class — do not let the
override framing imply every win is a small hook.

### 3. New format / codec

A quant codec as a kernel + `prepare`, gated against a **clean higher-precision reference the
validator serves** (so fp4-vs-fp4 lossy-on-lossy is impossible). This is granularity-1 item 2
promoted to a standalone contribution.

## The blessed base, and the languages miners write in

- **The base is an explicit, hashed lockfile** — CUTLASS, CuTe-DSL, Triton, flashinfer
  kernels, torch — pinned with CUDA/arch and asserted by `optima compat`. (Today only sglang
  is pinned; flashinfer/cutlass/triton ride along *implicitly* — which is also a latent
  consensus bug: two validators on different flashinfer pick different kernels → divergent
  throughput **and** numerics → Yuma penalty.) **`sglang` is excluded from the
  miner-importable base.**
- **Triton is the lingua franca**; CuTe-DSL/CUTLASS is the wizard tier (TMA / tcgen05 /
  warp-specialization). The subnet is an incentive to *force* miners to write real kernels —
  the bar is deliberately "can write a Triton/CuTe-DSL kernel or epilogue," **not** "PTX
  wizard." Override-points are the **graduated on-ramp**: start by rewriting an epilogue;
  graduate to whole kernels.
- **Accessibility is kept by the harness, not by lowering the bar:** examples, a local
  CPU/GPU `optima verify` loop, the [miner guide](MINER_GUIDE.md). Hard-banning `sglang` does
  not hurt approachability — *because* an override-point lets a miner contribute
  meaningfully without touching the engine at all.

## The three layers (where `sglang` is allowed to live)

- **`optima-kernels` — the library / the product.** Portable device code + host launchers +
  the clean-reimplemented codec/layout primitives. **Zero `sglang` imports.** Each entry
  carries a regime tag: `(model, ctx, concurrency, GPU, format) → base/override + measured
  speedup + fidelity gate`. This is what we ship and (maybe) open-source.
- **`optima-harness` — the subnet referee.** sglang + the seam + scoring, **confined here**.
  Consumes `optima-kernels` to place and measure submissions. `sglang` lives in this layer
  and nowhere else.
- **`optima-serve` — the managed service (later).** Runs `optima-kernels` on a *rented*
  engine (sglang/vLLM) for customers. Swappable infrastructure; the moat is the library.

**The dogfood pipeline:** subnet measures → because the win is portable inspectable source
*by construction*, it drops into `optima-kernels` with its regime tag → the service ships it.
No separate repo, no re-derivation. "Understand and replicate" is guaranteed by the
inspectability invariant, not left to goodwill.

## Kernel-adjacent code (codecs, layouts) — reimplement clean

The NVFP4/FP8 pack/unpack/scale-interleave/MMA-layout helpers the M3 bundle borrowed from
sglang are **reimplemented clean as first-class `optima-kernels` primitives** — *not*
classified as validator-adapter glue, *not* vendored. They are the spine of the library and
are themselves contributable (a better codec is a granularity-3 win). The Win Test applies:
a clean, faster, more-faithful codec is a win; calling sglang's is not.

## What the n=1 reconstruction pinned down (code-grounded, 2026-06-24)

A grounded read of the real `flashinfer` / `sglang` / `cutlass` clones (vs the literal
`apply_swigluoai_patch.py`) turned the abstract model above into concrete mechanism.

**The override-point ABI = CUTLASS EFC (Epilogue Fusion Customization) — not speculative.**
flashinfer's fused-MoE kernel *already* threads an `epilogue_op: cutlass.Constexpr = lambda x:x`
hook end-to-end (used today for requant, not the activation), and NVIDIA ships the canonical
pattern as an example family (`examples/python/CuTeDSL/cute/blackwell/efc/`): a **named registry**
of activations, each a **phased device method** plus a **built-in torch reference** for the
correctness check. So the submission ABI is: the miner ships a named `@cute.jit` epilogue + its
torch reference; the validator owns the base kernel and JIT-composes via the constexpr param
(monomorphized → zero overhead). **Adopt EFC's shape verbatim.** Consequences:
- The override **generalizes for free across the GEMM family** — MoE GEMM1, MoE GEMM2/finalize,
  and dense GEMM all already carry `epilogue_op`. **Attention does not** (hardcoded epilogue) →
  an attention override-point is a separate, more invasive keystone.
- **Contract on per-element scalar accumulators, not packed pairs.** There is no packed min/max
  in the CuTe-DSL ISA (only fma/mul/add/sub `*_packed_f32x2`), so a clamped activation
  (swigluoai) *must* be scalar — that's why the packed epilogue hung.
- **Pass model constexprs separately from the dequant scale** (the two-alpha trap): per-expert
  NVFP4 dequant `alpha` ≠ swiglu gain `swiglu_alpha=1.702`. Conflating them is the silent-wrong
  0.449-cosine failure; the activation reference + cosine gate catch it.

**The transferability cut line is `forward_impl`, and the current seam is on the wrong method.**
The win runs `FusedMoE.forward → forward_impl → run_moe_core → quant_method.apply →
@register_fused_func("none","flashinfer_cutedsl") → CuteDslMoEWrapper.run → patched device
kernel`. `forward_impl` is the true waist (eager, full-graph decode, AND both piecewise custom-op
paths converge there); the seam patches `FusedMoE.forward`, which the **piecewise-graph path
bypasses** (it calls `forward_impl` directly via a registered custom op). This is the concrete
form of the slot-waist-ceiling "why no kernel beats sglang is structural" bug — and it's a
one-line move (same for attention → the `unified_attention_with_output` custom-op).

**An epilogue win has no clean home in the 5 method-seams → add a registry-level override.**
"Same fused kernel, different epilogue" can't be expressed by wholesale-replacing
`FusedMoE.forward`. The right new seam is a shim at `FusedOpPool.get_fused_func` /
`@register_fused_func`: the miner ships a *runner variant* (the epilogue) keyed by `(a2a,
runner)`, and the validator relaxes the `assert activation == "silu"`. This is the natural home
for epilogue and codec wins. (sglang ships `srt/plugins/hook_registry.py` — BEFORE/AFTER/AROUND/
REPLACE with stale-`from X import Y` repair — which could retire the hand-rolled `.pth` for the
*method* seams, but it doesn't cover the registry/custom-op dispatch, so the shim is still needed.)

**Ingesting a *second, different* win — the taxonomy.** Five campaign win-shapes collapse to
**3 slot families × 2 granularities (whole-kernel vs epilogue-override)**:

| shape | example | slot | status |
|---|---|---|---|
| epilogue/activation override | swigluoai | `moe.fused_experts` + epilogue point | **delivered/scorable** |
| block-internal megablock fusion | the fused MoE megakernel | `moe.fused_experts` | exists (needs the `forward_impl` fix) |
| compute↔comm fused block | MoE+reduce overlap | `moe.fused_experts_reduce` | exists; M3 doesn't exercise it |
| mainloop/tile change | thin-N NVFP4 GEMM | `moe.fused_experts` | **killed (vendor floor)** — value is the kill-gate |
| whole-kernel attention sub-op | fp8 MSA indexer | **`attention.msa_block_score` (MISSING)** | **the second win the catalog can't score** |

- A **second MoE-family win** (another epilogue or codec) needs **zero new slots** — a registry
  override + a per-model activation row. Already shaped for it.
- A **structurally different win** (the fp8 indexer) needs the reusable triple: **(1) a finer
  seam** — the kernel produces *block scores*; the validator owns the top-k selection AND the
  bf16 attend (so the kernel stays upstream of the sampler); **(2) a new correctness mode** —
  `Correctness("topk_overlap", min_overlap≈15/16)`, because the output is a *selection set*, not
  a tensor (cosine/KL are the wrong metric); **(3) HP reference = bf16 scores.** This triple —
  *finer seam + set-metric + validator owns the irreducible downstream step* — is the pattern
  for any win at a sub-op whose output is a selection or intermediate.
- **Admission kill-gate** (from the killed tile-change shape): before the subnet pays to score a
  kernel, require a cheap bound-type + representative-M check — the win must move the **e2e
  serving wall**, not a microbench µs — so the subnet never rewards a vendor-floor rabbit hole.

## Versioning: the pin binds the validator, never the miner

Consensus (Yuma) requires all validators to measure the *same* submission and get the *same*
number, so the engine version must be pinned **for the measurement environment**. That pin is the
validator's, not the miner's. The miner is decoupled from it four ways:

1. **Portable IP** (Axiom 5) — the kernel contains no sglang version.
2. **Targets a frozen slot contract, not a sglang version** — a submission declares `slot` (+ model/
   arch eligibility), never `sglang==X`. The `SlotSpec` is sglang-agnostic and frozen.
3. **Develops anywhere; measured on the arena** — the miner never installs the arena's sglang. They
   build against `optima_kernels` + the slot reference and run `optima verify` (CPU/GPU, no engine);
   only the *validator* runs the full engine for the throughput number.
4. **The score is a within-version ratio** (bracketed B,C,B' vs the *same-version* baseline), so a
   minor version skew perturbs the ratio far less than an absolute number.

**The pin is per-arena, and an arena can be bleeding-edge.** Merge + generalize `optima/arenas.py`
(currently unmerged): an `Arena{model, sglang_ref, blessed_base_lock, seam_adapters, engine_kwargs}`
where `sglang_ref` is a **release, a PR branch, OR a commit** — `PINNED_SGLANG` becomes
`DEFAULT_ARENA.sglang_ref`. A launch-window model (M3 on `pr27944-current`) is competed on *that
branch* — the validator never waits for `main`.

**"5 days later it merges to main — then what?"** Nothing breaks: the arena stays self-consistent on
its commit, the champion is valid, consensus holds. When you *want* the merged version's
improvements, **bump the arena as a new season** (announced; all validators move together; `optima
compat` re-validates the seam adapters; re-baseline the champion; re-measure challengers). **Miners
ride the bump — they don't resubmit**; a kernel that only won on an old quirk loses on re-measure,
which is correct. The bump is a *validator* operation; the miner is a passenger.

**Hard limit (honest):** within an active arena, all validators run byte-identical (the blessed-base
lockfile). A submission cannot be measured on a different sglang than the arena's pin — two
validators on different runtimes break Yuma. The freedom is *per-model + bleeding-edge + portable-IP
+ ride-the-bump*, not "every miner picks their own sglang" (structurally impossible under any
consensus mechanism).

**Two things make this comfortable, not painful:** (a) migrate the seam to sglang's sanctioned
`srt/plugins/hook_registry.py` (AROUND/REPLACE + stale-import repair) so a bump is *cheap* → rotate
arenas aggressively → the pin stays near the frontier instead of a stale release; (b) the `Arena`
runtime is swappable — as `optima_kernels` + the own engine mature, an arena's runtime becomes
**`optima-serve`, not sglang**, and the portable kernels carry over untouched. "Decouple miners from
sglang" and "build our own engine" are the same arc.

## Cheat-resistance is unchanged (a separate axis)

Axiom 5 is the **product / composability** axis. The **cheating** axis is still governed by
the four invariants and still needs the unbuilt host isolation (`HOW_OPTIMA_WORKS.md` §8.4:
PID/mount namespaces, per-eval CUDA context, watchdog, RLIMIT) before opening to *untrusted*
miners. **A portable kernel is not a safe kernel — do not conflate the two axes.**

## What this is NOT

- **Not a container / BYO-stack.** That sacrifices composability, inspectability, and
  attribution — three of the four reasons Optima exists, and all three are the product goal.
- **Not "patch sglang."** Non-portable; evaporates on engine swap.
- **Not a config / integration knob.** The validator owns best-config and integration.

## Machinery deltas (sequenced — target state, not all built)

1. **Scanner: denylist → allowlist** over the blessed *kernel-library* base (CUTLASS/CuTe-DSL/
   Triton/flashinfer/sgl-kernel/DeepGEMM/torch — kernel libs are fine); **scan recursively**
   over all bundle `.py` (today only the declared entry file is scanned — the vendored-tree
   hole). Transferability is enforced by the contribution *unit* (device-source-at-a-slot/
   override-point), not by banning a namespace.
2. **Move the MoE/attention seams from `.forward` → `forward_impl`** (the run boundary every
   path converges on) — one-line fixes that close the piecewise-graph bypass (see below).
3. **Override-point** as a first-class manifest concept (`base_kernel` + `override_point`),
   modeled on **CUTLASS EFC** (named registry + phased device method + torch reference); ship
   the **epilogue** point first, installed via a registry shim at `FusedOpPool` (see below).
4. **Blessed-base lockfile** (CUTLASS/CuTe-DSL/Triton/flashinfer/torch + CUDA/arch) asserted
   by `optima compat`. (the consensus fix)
4. **`optima-kernels`** as a standalone package; harness/seam depend on it, never the
   reverse. Reimplement the codec/layout primitives clean.
5. **Served-reference-precision** control (serve clean bf16/fp8 ground truth while a
   candidate competes in fp4) so a new quant is winnable.
6. Win records stamped with **portable regime tags** → the library's index.

## Forced edits elsewhere (pending)

- `SLOT_CONTRACT.md`: add **Axiom 5** and the **Win Test** alongside the four invariants.
- `HOW_OPTIMA_WORKS.md`: document the library/harness/serve split, and that `sglang` is
  referee + rented runtime, not the product foundation.
