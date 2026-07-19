# Optima, explained end to end

This document describes the end-to-end architecture:

- what Optima is and the three roles (miner / validator / chain),
- **what the validator actually does** and what the "gold standard" should be,
- **what a miner submits** and why it's a kernel-slot, not a whole model,
- the entire lifecycle of a submission, file by file,
- the deep trick that gets an untrusted kernel *into* a running model,
- exactly how scoring (throughput + fidelity) and anti-copy (commit-reveal) work,
- the principal failure modes, their controls, and remaining limitations,
- what is *proven on real hardware* vs what is still a stub.

It points at every file. Paths are relative to this doc (`docs/`), so
`../optima/slots.py` is the harness module `optima/slots.py`.

> **Current scope:** eleven slots across three kinds (op / block / collective) — run
> `python -m optima.cli slots` for the live catalog; `STATE_OF_RECORD.md` carries
> the slot list of record. The mandatory separate eager/untimed audit role, typed
> host-regraded witness, and fail-closed durable transport are implemented and
> CPU/mock-covered, but these new bytes are not GPU-qualified until the exact production
> MiniMax-M3 canary passes;
> [FIDELITY.md](FIDELITY.md) distinguishes historical receipts from current enforcement.
> Validated on real GPUs up to gpt-oss-120b (1×H100) and
> MiniMax-M3-NVFP4 (4×B300).
> `STATE_OF_RECORD.md` is the live state-of-record (results + calibration); prose
> below may lag it — where they disagree, **the state of record wins**. Since
> 2026-07-07, submitted
> kernels **have** measured faster than stock sglang through the referee (the
> fused-epilogue collectives, 1.044–1.074× vs the noise bar — see the README's
> measured record); the bundles in `examples/` remain correctness demos.
> Crownable candidate execution is isolated in validator-owned no-egress OCI
> workers. Production intake and settlement use SQLite and a registered arena
> service; one passing qualification remains `reproduction_pending` until an
> independent second pass. Serving artifacts are produced by a separate, signed,
> chain-independent release path.

> Reading order: Parts 1–3 are the mental model. Part 4 is the pipeline. Part 5
> is the clever bit (how a kernel gets into the model). Parts 6–7 are scoring and
> anti-copy. **Part 8 is the threat model** — the one you asked for. Parts 9–13
> are status, a file-by-file index, how to run it, and a glossary.

---

## Part 1 — The mental model

Optima is a **competition to make LLM inference faster**, run as a Bittensor
subnet. There are three roles:

- **Miners** write optimized GPU **kernels** (Triton / CuteDSL) for individual
  operations inside a fixed model, and submit them.
- **The validator** is the **referee**. It takes each miner's kernel, plugs it
  into a model *it* controls, runs the model, and measures two things:
  1. **throughput** — did the kernel make the model faster?
  2. **fidelity** — did the model's output stay correct? (M3 uses an in-engine
     audit as the primary gate and treats rollout-KL as advisory)
  It turns those into a **score** and tells the chain how to pay miners.
- **The chain** (Bittensor) handles identity (miner hotkeys), the token
  emissions that pay miners, and consensus across validators.

One sentence: **miners submit kernels, the validator swaps each kernel into a
model and checks it's both faster and still correct, and the chain pays the
miners whose kernels genuinely win.**

The whole design exists to make that referee **un-cheatable**: a miner must not
be able to look faster by secretly doing less, by copying someone else, by
faking the measurement, or by escaping into the validator's machine.

---

## Part 2 — What the validator actually is

You asked: *"I think the validator's function is to run the forward pass and
compare it to some gold standard which we could pin to a production API or
another GPU pod."* That's the right shape. Two refinements that matter a lot.

### 2.1 The validator owns everything except one registered target

The validator owns the model weights, model graph, tokenizer, sampler, workload,
timing, reference, arena configuration, and stack assembly. A miner contributes one
registered singleton target or one exact atomic target. Targets may be individual ops,
fused blocks, collectives, or a bounded reviewed overlay product, but remain upstream of
the sampler and have validator-owned input/output and quality contracts.

This is the single most important design decision, and it's a direct answer to
your "arbitrary code execution / API substitution" worry (see Part 8.1). Because
the miner only produces one op's output and never the final tokens or the
logprobs, the classic "route to an API and return the right answer" attack
**doesn't apply** — there's nothing to substitute.

### 2.2 The "gold standard" should be local, not a production API

Your instinct to compare against a gold standard is exactly right. But **what**
the gold standard is matters enormously. There are three candidates:

| Reference | What it is | Verdict |
|---|---|---|
| **Production API** (e.g. DeepSeek's) | hit a remote endpoint, compare | ❌ avoid as the per-round gate |
| **Another GPU pod** | a second trusted instance of the same model | ✅ fine — same thing as below, on separate HW |
| **A local trusted run** (what Optima does) | the *same model* on the *same box* with the *stock* kernels | ✅ best |

Why the production API is the wrong per-round reference:

- **It's non-reproducible.** At temperature > 0 it's random; even at temperature
  0, the served version drifts (different quantization, system prompt, sampler).
- **It's sparse.** APIs return only top-k logprobs (or none), so you can't
  compute a true distribution distance, only a gameable approximation.
- **It breaks consensus.** Two validators hitting the API get *different*
  samples, so they compute different scores and Bittensor consensus falls apart.
- **It's a network dependency** inside your trusted measurement loop.

What Optima does instead: it runs the model **twice on its own GPU** — once with
the **stock** kernels (this is the reference / "gold standard") and once with the
**miner's** kernel — and compares the two. The reference is local, exact, free,
deterministic, and adversary-independent. "Another GPU pod" is the same idea with
the reference on separate hardware; it's compatible, just more plumbing.

Crownable qualification uses the strict causal authority: graph-on/audit-free baseline B,
candidate C, and baseline B′; a mandatory separate eager/untimed candidate audit role whose
exact slot×TP-rank receipts become a typed host-regraded witness; then a separately launched
pristine T worker that teacher-forces the sealed trajectories and supplies hidden quality
evidence. The candidate worker is never the grading oracle.

> Nuance for later: today's reference is "the same model with stock kernels."
> That's perfect for kernels meant to be *numerically equivalent* (a faster
> silu). For kernels that legitimately change precision (a new quantization
> kernel), you'd instead reference a frozen **full-precision** run and widen the
> slot/distribution tolerance. The machinery is the same; only the reference and
> calibrated contract change.

---

## Part 3 — What a miner submits

### 3.1 A bundle

A submission is a **bundle**: a directory containing data + kernel source.

```
bundle/
  manifest.toml          # DATA: which slot(s) this targets, where the source is
  kernels/
    silu_and_mul.py      # the kernel source (Triton/CuteDSL Python)
  metadata/
    silu_and_mul.json    # optional eligibility (dtypes, GPU arch, max sizes)
```

A real example is [../examples/miner_silu_triton/](../examples/miner_silu_triton).
Its manifest:

```toml
bundle_id = "example-silu-triton-v1"
abi_version = "optima-op-abi-v0"

[[ops]]
slot = "activation.silu_and_mul"
source = "kernels/silu_and_mul.py"
entry = "silu_and_mul"
dtypes = ["bfloat16", "float16"]
architectures = ["sm80", "sm86", "sm89", "sm90", "sm100", "sm103"]
metadata = "metadata/silu_and_mul.json"
```

The manifest is **data, not code** — the validator parses it
([../optima/manifest.py](../optima/manifest.py)) to learn *which* slot the bundle
targets and *where* the kernel source lives. It does not run anything yet.

Loading rows and contribution identity are separate. An optional syntax-only
`[competition]` table may request a registered target, while
[../optima/target_catalog.py](../optima/target_catalog.py) resolves the distinct
semantic slots (deduplicating variants) to a validator-owned singleton or exact atomic
target. The catalog supplies canonical member order and explicit displacement/compatible-
overlap policy; miners cannot declare those relationships. Unknown multi-slot work is
classified as unregistered for the fenced reviewed-discovery lane rather than inheriting
`ops[0]` as an accidental reward identity. It cannot auto-earn or reset a registered
family clock. The selected pure policy intends either promotion into a registered target
followed by fresh requalification/CROWN, or one finite bounty. Durable schema 5 currently
retains review-pending wins and can issue `bounty_only`; it rejects promotion until typed
promotion transport, target registration, requalification linkage, and one cross-lane
work identity exist. The “never both” rule is therefore not yet end-to-end enforcement.
Identity-only resolution marks external feature evidence
incomplete; intake resolution requires a trusted projection of exact rebuild capabilities.
Neither duplicates nor executes `rebuild.json`, whose reviewed-patcher policy remains separate.
Legacy CLI/chain score records are intentionally not wired to this catalog yet; stack assembly,
qualification, intake, and settlement must migrate together so two identity authorities never
coexist in one economic path.

### 3.2 The op-slot ABI — the contract

The set of operations a miner is allowed to replace is the **slot catalog**,
owned by the validator in [../optima/slots.py](../optima/slots.py). Today there are
eleven slots across three `kind`s — `op` (`activation.silu_and_mul`, `norm.rmsnorm`),
`block` (`attention.sdpa`, `attention.decode`, `attention.msa_block_score`,
`attention.msa_prefill_block_score`, `moe.fused_experts`), and `collective`
(`collective.all_reduce`,
`moe.fused_experts_reduce` — the experts block that owns its trailing reduce —
and the fused epilogues `collective.ar_residual_rmsnorm` /
`collective.moe_finalize_ar_rmsnorm`). A slot
(`SlotSpec`) declares everything the validator needs to *use* and *verify* a kernel
without trusting it:

```python
SlotSpec(
    name="activation.silu_and_mul",
    entry="silu_and_mul",        # the function the miner module must expose
    make_inputs=_silu_and_mul_inputs,    # deterministic test-input generator
    out_shape=_silu_and_mul_out_shape,   # how to size the output
    invoke_reference=lambda i: ...,      # trusted ground-truth implementation
    invoke_entry=lambda entry, i, out: ...,  # slot-specific call contract
    shapes=( ... ),              # the shapes correctness is checked on
    tolerances={ bf16: (2e-2, 2e-2), ... },
)
```

The **kernel contract** is deliberately tiny. The miner provides one function:

```python
def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    # write  silu(x[..., :d]) * x[..., d:]  into out
```

Note who allocates `out`: **the validator does**, then passes it in. The miner
only *fills* it. The miner never controls the output shape, dtype, or stride, and
never sees anything but the input tensor. This is how we keep the miner's "host
surface" — the amount of non-kernel code they run — as small as possible. (See
the real kernel at
[../examples/miner_silu_triton/kernels/silu_and_mul.py](../examples/miner_silu_triton/kernels/silu_and_mul.py):
a `@triton.jit` device kernel plus a ~10-line launch.)

### 3.3 Why a kernel-slot and not a whole model

You could imagine letting miners submit a whole model implementation that the
validator compiles and runs. We deliberately **don't**, and your API-substitution
worry is exactly why. If the miner controls the entire forward pass, they can:

- drop layers, 1-bit quantize, or otherwise gut the model, then
- make the *output* look correct by fetching real answers from an API.

With a registered target, the validator still produces tokens and quality evidence from
the complete model and a separate pristine T authority. The candidate does not control
the final output, timing driver, workload, or grading reference. Arbitrary host launch
code remains untrusted and is therefore imported, built, and executed only inside the
OCI isolation boundary described in Part 8.

---

## Part 4 — The lifecycle of a submission

Here is the entire pipeline, with the file/function that does each step.

```
   MINER                                  VALIDATOR
   -----                                  ---------
1  native timelock commitment       ───► finalized chain history
2  revealed hash + fetch URL        ───► private hostile-archive fetch
3                                         exact re-hash + copy priority
4                                         immutable worker publication
5                                         registered arena admission
6                                         non-crown static/build/ABI/graph/serve screen
7                                         isolated B/C/B' + audit witness + pristine-T qualification
8                                         independent reproduction of the same delta
9                                         transactional target/stack settlement
10                                        journaled weight reconciliation
11                                        reviewed chain-independent Engine release
```

The production stages are:

1. **Finalized arrival.** Native Bittensor timelock commit-reveal hides the URL until
   reveal. `chain.read_finalized_reveal_history` supplies consensus ordering; SQLite
   records a durable cursor and reservation before fetch.
2. **Private intake and publication.** `chain.fetch.fetch_bundle` rejects unsafe or
   oversized archives and re-derives the committed tree hash. Copy precedence is resolved
   from finalized order. `chain.publication.publish_worker_bundle` copies accepted input
   into an immutable, hash-complete worker tree; the private fetch tree is not mounted into
   the worker.
3. **Arena admission and screening.** A validator-injected `ArenaServiceRegistry` selects
   an exact runtime/model/topology/workload policy. The fixed static, build, ABI, graph, and
   abbreviated-serving screens are non-economic: they can reject, retry, hold, or promote,
   but cannot crown.
4. **Qualification.** Promoted candidates run graph-on/audit-free charged B/C/B′ roles,
   a mandatory separate eager/untimed audit role with a typed host-regraded witness, and
   pristine-T authority. Candidate import and native build occur only in the no-egress OCI
   worker. The controller owns roles, timing, graph proof, raw quality evidence, evidence
   authentication, and teardown. Unauditable attention slots fail closed.
5. **Independent reproduction.** The first PASS is retained as
   `reproduction_pending`. A second PASS must bind the same arena, target, delta, incumbent
   and candidate stack identities while using independent authority and selection evidence.
6. **Settlement and incentives.** `settlement.py` plans the target-level stack transition
   from the paired candidate and uses the lower speedup. `FinalizedIntakeStore` commits it
   transactionally. The current `set-weights` command separately reconciles retained
   legacy-V1 projection state. The selected V2 registered-CROWN and reviewed-discovery
   classes have schema-4/5 pre-activation state APIs, including review-pending retention,
   bounded bounty issuance, terminal pending-review expiry, and registered-family runtime
   invalidation. Promotion is deliberately fail-closed and the invalidity decision remains
   external authority.
   `chain-incentive-shadow` has a live registered-CROWN-only
   synthetic receipt; the signer-free `chain-incentive-composition-shadow` also passed
   live finalized-membership projection over explicitly synthetic states and wrote
   `submitted=false`. Neither receipt is review, settlement, publication, debt-debit,
   or activation authority. `chain-activate-incentives` now performs a wallet-free atomic
   schema-5→6 cutover for exactly one immutable MiniMax-M3 campaign. It binds exact
   core/composition/approval bytes at the exact finalized intake cursor, preflights the
   complete catalog-derived family roster against exactly one retained arena, derives the
   campaign ID from that arena/catalog/roster, checks reserve membership, and receipts the
   arena/stack/catalog/membership identities. `set-debt-weights` implements restart-safe
   gapless projection, finalized readback, only-then debit, and rate-limited catch-up.
   Neither path has a live receipt. Launch still needs exact production family/reserve
   manifests and a fresh shadow, independently graded review/runtime invalidation,
   membership-departure history, reliable pending-review expiry, promotion linkage, the
   production audit GPU canary, and actual activation/mainnet operations.
7. **Integration and release.** An approved `IntegrationReviewRecord` binds the two crown
   attempts, exact source, provenance/license/security/compatibility/test evidence, and
   review commit. Only integrated refs enter `EngineReleaseManifest`. Model provisioning and
   signed release construction are chain-independent and produce serving artifacts rather
   than referee state.

---

## Part 5 — The deep magic: getting a kernel *into* a running model

This is the part that took the most engineering and is the least obvious. The
goal: when the model computes the SiLU op, it should call the **miner's** function
instead of sglang's built-in one — but only in the eval, only for the candidate
run, and without the miner being able to touch the timer.

### 5.1 The obstacle: sglang runs the model in a *separate process*

`sglang.Engine` does `mp.set_start_method("spawn")` and launches the model in a
**separate "scheduler" process**. With *spawn*, that child is a brand-new Python
interpreter — it re-imports sglang from scratch. So if you patch a class in the
parent process, the child never sees it. The model runs in the child; your patch
is in the parent. Naive monkeypatching silently does nothing.

(Correction to an earlier note: the pinned sglang **does** ship a hook/plugin
framework — `srt/plugins/hook_registry.py` (BEFORE/AFTER/AROUND/REPLACE hooks via
`sglang.srt.plugins` entry points), added by PR #21388 and present at the pin
`0.5.12.post1`. So an entry-point hook IS available; we keep the `.pth` path primary
because it is version-independent and known spawn-safe, and track migrating to the
sanctioned hook as future work. `PINNED_SGLANG` is in
[../optima/compat.py](../optima/compat.py).)

### 5.2 The solution: a `.pth` + a post-import hook

We need code to run **inside every interpreter in the venv**, including the
spawned child, regardless of sglang version. Python gives us exactly one such
hook: **`.pth` files**. A line in a `.pth` file under site-packages that starts
with `import` is executed at interpreter startup — in the parent *and* in every
spawned child.

So the install step writes one line into site-packages:

```
echo 'import optima.bootstrap' > $SITE_PACKAGES/optima.pth
```

[../optima/bootstrap.py](../optima/bootstrap.py) then does **not** import sglang
at startup (too heavy/fragile). Instead it registers a **meta-path finder** that
watches for the import of the seam-target modules (`_TARGETS`: `activation`,
`layernorm`, `radix_attention`, the fused-MoE `layer`, and the distributed
`parallel_state`) and, the moment one finishes loading, runs `seam.activate()`
against it. This is the "post-import hook" pattern:

```python
class _SeamFinder(MetaPathFinder):
    def find_spec(self, fullname, ...):
        if fullname not in _TARGETS:           # the seam chokepoint modules
            return None
        spec = <real spec from the other finders>
        spec.loader = _wrap_loader(spec.loader)   # run seam.activate() after exec
        return spec
```

### 5.3 What `seam.activate()` does

[../optima/seam.py](../optima/seam.py), `activate()`:

1. **Installs the dispatcher** into `SiluAndMul`
   ([../optima/integrations/sglang_silu.py](../optima/integrations/sglang_silu.py),
   `install`). It replaces the class method `SiluAndMul.forward_cuda` (and
   `forward_native` for CPU) with a wrapper that consults a registry, keeping the
   original for fallback. Patching at the *class* before the model is built means
   every `SiluAndMul` instance binds the wrapper when it's constructed.
2. **Decides whether to load the miner kernel**, from env:
   - `OPTIMA_ACTIVE=1` and `OPTIMA_BUNDLE_PATH=<dir>` → re-scan, load, and
     register the kernel; enable the registry.
   - otherwise → registry stays empty/disabled → the dispatcher always falls
     back to stock. This is the **baseline** run.

So the *same* process code serves both runs; the validator picks baseline vs
candidate by setting env before each launch (see `_run_launch`).

### 5.4 The dispatcher

[../optima/dispatch.py](../optima/dispatch.py),
`make_silu_and_mul_dispatcher`. This is the **one place** a miner kernel is
called during inference. It:

```python
def dispatched(self, x):
    impl = REGISTRY.lookup("activation.silu_and_mul",
                           dtype_name=..., last_dim=x.shape[-1], arch=...)
    if impl is None:
        return baseline_forward(self, x)          # not eligible / not active
    out = torch.empty(<validator-chosen shape>, dtype=x.dtype, device=x.device)
    try:
        impl.entry(x, out)                         # the ONLY miner call
    except Exception:
        return baseline_forward(self, x)           # crash -> fall back
    return out
```

The validator owns the allocation, the eligibility check
([../optima/registry.py](../optima/registry.py), `Eligibility.accepts` — matches
dtype/arch/size, else falls back), and the fallback. The miner's `entry` only ever
sees pre-allocated tensors.

### 5.5 Tamper-resistant timing (`mark_driver`)

There's a subtlety. The `.pth` runs in **every** process, including the
validator's *driver* process (the one that calls `engine.generate()` and holds
`time.perf_counter`). If the driver also imported the miner module, a malicious
kernel's module-level code could monkeypatch the driver's clock.

Fix: the driver calls `seam.mark_driver()` **before importing sglang**
(see `_run_launch`). `activate()` checks that flag and, in the driver, installs a
*pass-through* seam but **never imports the miner module**. The miner kernel is
loaded only in the spawned scheduler child (a fresh process where the flag isn't
set). So:

- **scheduler child**: runs the miner kernel (this is where it executes).
- **driver process**: holds the timer, never touches miner code.

The two are different OS processes communicating over IPC, so the kernel cannot
reach the timer. This is what "out-of-process timing" means here, and it's why the
broken-kernel test still fails (proving the kernel really runs in the child) while
the timer stays clean.

### 5.6 A full trace of one candidate forward pass

1. Driver: `seam.mark_driver()`; set `OPTIMA_ACTIVE=1`, `OPTIMA_BUNDLE_PATH=…`;
   `import sglang`; `Engine(...)`.
2. Engine spawns the scheduler child. Child boots → `.pth` runs
   `import optima.bootstrap` → meta-path finder installed.
3. Child imports `sglang.srt.layers.activation` → finder fires →
   `seam.activate()` → patches `SiluAndMul`, loads + registers the miner kernel,
   enables the registry.
4. Child builds the model; each `SiluAndMul` binds the dispatcher.
5. Driver: `engine.generate(prompts, return_logprob=True, top_logprobs_num=k)`.
6. Child runs the forward pass; at each MLP, `SiluAndMul.forward` → dispatcher →
   miner `entry(x, out)`. The rest of the model (attention, norms, sampler) is
   stock sglang.
7. Child returns tokens + per-position top-k logprobs over IPC to the driver.
8. Driver stamps wall-clock around the call and reads the logprobs. **Neither the
   timer nor the logprob computation ran any miner code.**

---

## Part 6 — How scoring measures (and why)

The measurement principles below were developed on the original two-launch
developer evaluator (deleted in the post-arc trim) and now live inside the
production qualification authority: robust bracketing in
[../optima/eval/scoring.py](../optima/eval/scoring.py), the graph-on/audit-free
charged B/C/B′ schedule plus mandatory separate eager/untimed audit role and pristine-T in
[../optima/eval/qualification_runner.py](../optima/eval/qualification_runner.py),
the quality record in
[../optima/eval/reference_quality.py](../optima/eval/reference_quality.py), and
threshold provenance in [../optima/eval/calibration.py](../optima/eval/calibration.py).

### 6.1 Baseline and candidate launches

Same weights, same seed, same sampler, same prompts — the **only** difference is
the one kernel, so any delta is attributable to it.

> **Each launch runs in its own fresh process.** sglang + deterministic mode set
> process-global CUDA/torch state (deterministic algorithms, the cuBLAS workspace,
> the sampling backend); in a *shared* driver the baseline's state corrupted the
> candidate launch on big MoE models (observed on gpt-oss-120b: a no-op kernel
> "regressed" to 0%). Isolated processes make the launches independent and free
> all GPU memory between them. Production goes further: candidate execution is
> fenced in a no-egress OCI worker, and the trusted controller never imports miner
> code.

### 6.2 Throughput (robust)

Per launch: a **warmup** generate (so JIT/compile/graph costs aren't timed), then
**K timed** generates, reported as the **median** + spread. Tokens come from the
driver-known token budget (`ignore_eos` keeps budgets identical), never from the
miner. The candidate is **bracketed** by a baseline before AND after (B, C, B′),
paired against the mean of the brackets, with the bar derived from the measured
baseline noise (`1 + max(margin, k·noise)`) and a NO-DECISION verdict when the
bracketing baselines disagree (`optima/eval/scoring.py`). Median-of-K is why a
single noisy sample can't swing the score (we measured ~7% run-to-run sd on a
tiny model — exactly why the noise-derived bar exists).

### 6.3 Fidelity (distribution checks)

With `return_logprob=True, top_logprobs_num=k`, sglang exposes, per generated
position, the top-k `(logprob, token_id, text)` — its actual output distribution.
The pristine T worker teacher-forces the sealed trajectories and the quality
record (`reference_quality.py`) carries `mean_nll` / `worst_nll`, top-k rollout KL
(`topk_kl`), `argmax_rate` (sparse flips), `coverage_dev` (a flattened head-matching
distribution that fools top-k KL), and the arena's `task_score`. The arena policy grades
these facts; M3 treats rollout-KL as advisory while the audit is primary and pristine-T
task/distribution evidence remains the semantic backstop.

The **alignment** subtlety was a real bug fix and still governs how per-position
comparison works: greedy decoding means baseline and candidate can *diverge* in
their token sequence if the kernel changes an argmax. After a divergence, later
positions aren't comparable (different context). So compare position *i*, **then**
stop if token *i* differs — and compare position 0 *before* checking, because
position 0 always has identical context (same prompt), so a kernel that derails
the very first token still gets a huge KL instead of "zero comparable positions."

> Honest limitation: top-k truncation approximates the true full-vocab KL. It's
> useful for catching calibration collapse and dropped work at k≥20. The intended
> in-engine audit (Part 6.5, [FIDELITY.md](FIDELITY.md)) re-runs stock on clones of
> the candidate's real calls. Its separate role, typed host-regraded witness, and durable
> fail-closed transport are implemented and CPU/mock-covered, but a real-GPU canary has
> not yet qualified these new causal bytes.

### 6.4 Gates and verdict

The gate philosophy, unchanged since the first evaluator:

- **fail quality → no crown**, no matter how fast. You cannot trade correctness
  for speed.
- **pass quality but below the noise-derived bar → no improvement**; it can't take
  a target, but it isn't punished.
- **pass both → the settled value is the measured speedup** (production settles
  the *lower* of two independent reproductions).
- **bracketing baselines disagree → NO-DECISION**: re-queue, never crown.

### 6.5 The realistic workload + the quality authority

Throughput must be scored on the regime the arena sells (decode-heavy serving; a
prefill-heavy slot needs a prefill-heavy workload). The causal path uses a mandatory
separate eager/untimed **in-engine audit** with exact slot×TP-rank coverage and a typed
host-regraded witness, plus pristine-T distribution/task evidence. Charged graph-on B/C/B′
roles carry no audit state or receipts. MiniMax M3 uses the audit as its primary fidelity
gate and rollout-KL as advisory. The underlying audit mechanism has B300 evidence; the new
causal transport/report bytes have CPU/mock coverage but still require the production GPU
canary. Unauditable attention slots fail closed. See
[FIDELITY.md](FIDELITY.md) before changing or relying on this boundary.

### 6.6 Calibration (learned on real hardware)

The gates work; the *thresholds* must be calibrated, and we found this empirically
on gpt-oss-120b:

- **KL threshold must equal k× the nondeterminism noise floor**, not a hand-picked
  constant. With `--no-deterministic`, stock-vs-stock KL on gpt-oss-120b was
  **3.9e-4** (the floor). A genuinely-drifting kernel sat at **9.2e-3 (~24× the
  floor)** — correctly flagged. Always measure stock-vs-stock first. On arenas
  where two identical stock launches are NOT logit-identical (MiniMax-M3), the
  floor never reaches ~0 — that measurement is why the audit mode exists.
- **End-to-end distribution checks catch what op-correctness misses.** A drifting
  kernel *passed* per-op correctness (bf16 tolerance) but failed end-to-end — the
  layered check (cheap per-op pre-filter → end-to-end gate) working as designed.
- **Task accuracy needs large n.** At n=12, GSM8K's ~12% std turns a 2-problem
  flip into "−16.7%." Dense per-token statistics are the primary gate; task
  accuracy is a capability *floor* at realistic sample counts. Every merged gate
  threshold must cite a measured stock-vs-stock floor artifact
  (`optima/eval/calibration.py` binds that provenance).

---

## Part 7 — Anti-copy and emissions

[../optima/bundle_hash.py](../optima/bundle_hash.py) and
[../optima/copy_fingerprint.py](../optima/copy_fingerprint.py).

### 7.1 Bundle identity

`content_hash(bundle_dir)` is a deterministic SHA-256 over the manifest + every
source file (sorted, length-prefixed, junk excluded). Two bundles with the same
content have the same hash; any change flips it. This hash is the thing
commitments bind to and the thing copy-detection compares.

### 7.2 Commit-reveal defeats copying

The problem: submissions are evaluated in the open, so a lazy miner could copy the
current leader's bundle and resubmit. Commit-reveal kills this:

- **Commit window:** each miner posts a native timelock commitment containing the
  content hash and encrypted fetch URL. The payload is unreadable until its reveal round.
- **Reveal window:** validators read the revealed payload from finalized chain history,
  fetch the bundle, and independently re-derive the exact committed content hash.

A copier who only sees a rival's bundle at reveal time has **no matching prior
commitment** for it, so they cannot reveal it. And if two miners independently
committed to the same content, the **earliest commit (lowest sequence) is the
original**; later identical ones are marked copies (`original=False`).

### 7.3 Transactional target settlement

Production settlement operates on canonical singleton or atomic targets, not manifest
row order. A candidate is eligible only after two independently authorized passing
qualifications; its settlement speed is the lower measured value. The planner assembles
the exact incumbent stack plus that one delta, checks target displacement and compatibility,
and produces an append-only event plan. SQLite commits the event and resulting evaluation
stack atomically. Copy-demoted and non-passing submissions never enter this path.

Emission projection is separate from target settlement. The live, retained policy generation
is legacy V1: standing relative-improvement credit with reciprocal time decay plus bounded
discovery rewards. Its `set-weights` submission is journaled and fail-closed rather than a
side effect of evaluation.

The selected V2 registered-CROWN rule instead issues finite principal in multiplicative
1%-log units, uses a rational elapsed-family bonus capped at 10%, keeps a 10% reserve, and
expires unpaid principal after 90 days. D-015 makes the model campaign the claim-sizing
unit. Launch accepts exactly one immutable MiniMax M3 campaign at 100% sizing.
Historical two-model 50/50 cells remain arithmetic research; a second campaign,
rotation, and successor activation are unsupported by this generation.
Reward families remain independent frontiers and clocks, and
adding families within a campaign causes no principal dilution. Campaign shares size
claims rather than hard-siloing epoch payout; registered claims share the global pool pro
rata. D-013 composes one separately reviewed discovery
class without weakening that reserve floor:

```text
P_d     = min(50,000, live discovery debt)
P_c     = min(900,000 - P_d, live registered-CROWN debt)
reserve = 1,000,000 - P_d - P_c
```

All 14 preregistered D-015 screens passed. At `k=1`, the normal weekly load was
one full-sized 4.4%/5% claim with one campaign, or—in the historical research
cells—one half-sized claim in each of two campaigns (one full share aggregate),
rotated across families. It paid 100%,
expired zero, and drained to zero; five-day cadence was marginal and four-day
cadence overloaded. Sustained simultaneous per-family wins were not the normal-tape
assumption. Report semantic digest:
`7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590`.

A tracked one-campaign supplement then exercised 64 launch/stress cells across
1/2/5/10 independently winning M3 families, four cadences, and empty/saturated
discovery. Its semantic report digest is
`505fed4d40a6acc6bc92d6330170e8e2260a52e5f3099c22a6c0eb4b2308c672`.

Each class has its own claim-digest largest-remainder allocation. A discovery award is
capped at one 50,000-unit pool epoch and has no campaign share, family clock, time bonus, renewal,
or permanent title. Its 648,000-block lifetime starts at the retained qualified-win block,
not review; delay consumes the window, and review at or after expiry cannot mint. The
landed finalized expiry path records `review_expired`/`discovery_review_expired` for pending
wins.

The pure D-013 disposition type expresses promotion plus fresh registered
qualification/CROWN versus finite bounty as mutually exclusive choices. The durable store
does not implement both branches: it retains `ReviewPendingDiscoveryWin`, issues only
bounded `bounty_only`, and rejects `registered_promotion`. Promotion still needs typed
`DiscoveryWinRecord`/`DiscoveryPromotion` transport, target registration, fresh
requalification/CROWN linkage, and cross-lane same-work identity. Until then, “never both”
is policy intent rather than an enforced work-level fact. The selected D-013 cell is
`8561028c943738da2fe622e5f5c9fd43ebec16fdd59feab3561de25fbfa450d9` and its report
digest is `6bdfce26e4e6090e0dcc8814a636c665f28d1ff20945a09d43a9a90dc94151fc`.

D-014's 288-row review-delay sensitivity replayed byte-identically across
arm64/Python 3.11 and x86_64/Python 3.12. Its preregistered 0/1/7-day SLA passed
all 108 rows with 100% discovery payout, no expiry/unissued debt, at most 55,555 ppm
instantaneous CROWN-capacity dilution, and no CROWN paid-fraction regression; the
90/120-day cases issued no stale debt, while 30/60/89 days remained diagnostic.
This is synthetic accounting evidence only; see [INCENTIVES.md](INCENTIVES.md) for
the report digest and limits.

This composition is not active. The wallet-free `chain-activate-incentives` command
implements one atomic schema-5→6 cutover over exact approved core/composition bytes and the
exact finalized intake cursor. It preflights the complete catalog-derived family roster
against exactly one retained evaluation arena, deterministically derives the campaign from
the arena/catalog/roster, checks reserve membership, and reproduces the arena, stack,
catalog, and membership identities pinned in the independent approval before retaining
them in the activation row/event. `set-debt-weights` implements restart-safe gapless publication,
finalized readback, only-then debit, and rate-limited catch-up. Neither path has a live receipt.
Production must still freeze the exact MiniMax-M3 family/reserve manifests and run a fresh
campaign-policy shadow; grade review and runtime invalidation independently; retain membership
departure history rather than only a current snapshot; schedule pending-review expiry
reliably; complete promotion linkage; pass the production audit GPU canary; and perform
actual activation/mainnet operations. The landed
family-invalidation API only consumes an
external authority digest, while `review_digest` is controller-supplied/content-bound.
The signer-free
`chain-incentive-composition-shadow` command projects explicitly synthetic claims against
exact finalized membership and always records `submitted=false`. Before D-015, its live
testnet-netuid-307 run retained finalized block 7,586,146 and metagraph size 6, allocating 850,000 ppm to
registered-CROWN claims, 50,000 ppm to reviewed-discovery claims, and 100,000 ppm to the
reserve, exactly 1,000,000 ppm. Receipt semantic digest:
`3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`; file SHA-256:
`ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`. The states were
explicitly synthetic, no wallet was constructed, and the receipt supplies no review,
settlement, publication, debt-debit, D-015 policy, or activation authority. See
[INCENTIVES.md](INCENTIVES.md) and
[EMISSIONS_POLICY.md](EMISSIONS_POLICY.md).

### 7.4 Mapping to Bittensor

Commitments live on-chain, `hotkey` is the miner's SS58 address, and weights are the
validator's finalized projection. `optima/chain/` is the production path and persists its
authority in SQLite.

---

## Part 8 — The threat model (every failure mode I can think of)

The tables group attacks by the authority they target. "Mitigated" denotes a control in
the production path; "partial" denotes a residual that still requires calibration,
operational policy, or broader coverage.

### 8.1 Your attack: substitute the output (e.g. via an API call)

> *"The miner runs gibberish kernels while the real results come from an API call,
> so the KL diff looks correct."*

**Verdict: structurally defeated by the registered-target design and enforced by the
OCI boundary.**

- The miner implements one registered singleton or exact atomic target upstream of the
  sampler. It does not own the final tokens, role schedule, timing, or T quality evidence.
  There is no miner-defined final output or grading reference to substitute.
- Candidate Python can mutate its own scheduler process, so that entire process is
  untrusted. It runs inside a no-egress OCI worker with bounded mounts, read-only root,
  dropped privileges/capabilities, seccomp/resource policy, a distinct CUDA context, and
  controller-owned teardown. Static scanning remains a tripwire rather than the boundary.
  The trusted controller and pristine T authority are outside that worker.

### 8.2 Fake the result — pass fidelity without doing the real work

| # | Attack | Verdict | How |
|---|---|---|---|
| 1 | Return wrong `out` cheaply, hope it's "close enough" | **Mitigated** | M3's separate audit role compares sampled real slot outputs with stock under the slot tolerance; pristine-T task/distribution evidence is the semantic backstop. Rollout-KL is advisory there. |
| 2 | Compute only some tokens/positions, leave the rest stale | **Mitigated** | An audited call compares the complete model-facing slot output, exact slot×rank minimum coverage fails closed, and pristine-T catches downstream degradation. |
| 3 | Memoize/replay op outputs across calls | **Mitigated** | Audit baselines use pre-call clones from the real engine distribution; activations and per-epoch prompts vary. |
| 4 | Behave in the eager audit role, garbage in charged graph-on roles | **Partially open** | Charged B/C/B′ are audit-free and pristine-T grades charged candidate evidence, but miner code can distinguish the eager audit role from charged execution. Audit-role fingerprinting remains explicit risk. |
| 5 | Special-case verify/audit inputs or tamper with the in-process audit | **Open/partial** | Verify jitters shapes and the typed witness is host-regraded, but miner host code shares the audit rank process. In-process tampering and role fingerprinting require isolation or an indistinguishable audit design. |

### 8.3 Fake the speed — inflate throughput

| # | Attack | Verdict | How |
|---|---|---|---|
| 6 | Monkeypatch the timer | **Mitigated** | `seam.mark_driver()` — the timing process never imports miner code (Part 5.5). |
| 7 | Fabricate the token count | **Mitigated** | The controller fixes the token budget and validates authenticated, fixed-width worker evidence; candidate output is not trusted as an unconstrained throughput numerator. |
| 8 | Offload work to an untimed stream / return early | **Mitigated** | `torch.cuda.synchronize()` brackets the timer; async work is counted. |
| 9 | Be fast only on charged benchmark shapes | **Partial** | Fresh prompts and pristine-T reduce the attack surface, but stable charged-workload features remain visible. Timed-workload fingerprinting stays open until charged work is drawn from a broader serving distribution with randomized regimes. |
| 10 | Manipulate GPU clocks | **Needs host op** | Lock clocks (`nvidia-smi -lgc`) at the validator host; not enforced in code yet. |

### 8.4 Escape the abstraction — arbitrary code execution

Candidate Python and generated native artifacts are treated as hostile. The production
boundary is the validator-owned OCI session, not Python-level sandboxing.

| # | Attack | Verdict | How (required) |
|---|---|---|---|
| 11 | Patch model components / sampler / measurement | **Mitigated at authority boundary** | Such mutation remains inside the candidate worker. The trusted controller assigns roles and timing; the pristine T worker independently grades sealed trajectories. |
| 12 | Exfiltrate weights or evidence | **Mitigated** | Candidate OCI networking is disabled; mounts and protocol frames are bounded and validator-owned. |
| 13 | Persist on the validator or affect later evaluations | **Mitigated** | Ephemeral OCI lifecycle, read-only root, bounded mounts, dropped capabilities, seccomp/resource policy, teardown and residue checks. |
| 14 | Hang the GPU | **Mitigated operationally** | Controller deadlines terminate the worker and own device teardown; the result is validator-side NO_DECISION, not a candidate pass. |
| 15 | Out-of-bounds device writes | **Contained per evaluation** | Candidate engine construction and CUDA context live inside the disposable worker; no other candidate shares that session. |
| 16 | Malicious import or native compilation | **Mitigated at authority boundary** | Source/native build and loading occur only inside the isolated worker under an attested toolchain and artifact policy. |

**Bottom line for Part 8:** untrusted code may control its own candidate engine, but not
the chain/control plane, arm schedule, trusted evidence authority, settlement record, or
serving release. The mandatory audit witness does not close in-process tampering,
audit-role fingerprinting, or timed-workload fingerprinting. The first is a known isolation
gap; the latter two remain partially open even after the required GPU canary.

### 8.5 Mechanism / economic attacks

| # | Attack | Verdict | How |
|---|---|---|---|
| 17 | Copy the champion's bundle | **Mitigated** | Finalized timelock priority + exact/structural copy disposition; copied content never enters qualification or settlement. |
| 18 | Front-run a rival's reveal | **Mitigated** | Can't reveal what you have no prior commitment to. |
| 19 | Sybil / discovery double-dip | **Partial; promotion open** | The active V1 path uses canonical reward families, improvement over the frontier, copy demotion, and registration cost. V2 log accounting removes the base CROWN split advantage, and schema 5 bounds/uniquifies bounty-only claims. But registered promotion is rejected rather than linked, so cross-lane work identity does not yet enforce the pure policy's “never both” intent. Independent review, promotion transport, and cross-family collusion remain open. |
| 20 | Overfit the eval distribution (great on eval prompts, useless in production) | **Ongoing risk** | Fresh per-epoch prompts from a corpus; rotate/expand toward the real serving distribution. This never fully "closes" — it's a tuning discipline. |
| 21 | Self-dealing validator | **Needs consensus** | Multiple validators + reproducible scoring so Bittensor consensus catches an outlier. Determinism work (Part 10) enables this. |

### 8.6 Reference / measurement validity

| # | Issue | Verdict | How |
|---|---|---|---|
| 22 | Reference itself wrong/drifting (if pinned to an API) | **Avoided** | Local stock-kernel reference, not an API (Part 2.2). |
| 23 | Per-op tolerance ≠ end-to-end quality | **Mitigated** | M3 gates sampled in-engine slot outputs plus pristine-T task/distribution evidence; rollout-KL is advisory there. |
| 24 | Cross-validator score divergence (noise, HW) | **Partial** | Pooled charged arm rates + B/C/B′ noise margin now; locked clocks + determinism + pinned HW still needed. |
| 25 | Numerical nondeterminism fails a faithful kernel | **Mitigated** | Tolerance in verify; `enable_deterministic_inference` available; greedy decoding for alignment. |

---

## Part 9 — What is proven (on a real H100)

Validated on a real H100 (sglang 0.5.12.post1 / CUDA 13, torch 2.11+cu130), Qwen2.5
up to gpt-oss-120b:

- **The seam works on real models, including a 120B MoE.** Confirmed because the
  *broken* kernel changes the output. The catalog later grew to eleven slots across
  op / block / collective kinds; this H100 evidence covers specific seam paths, not
  every current slot. For example, `norm.rmsnorm` fires on gpt-oss (whose activation is fused into the MoE kernel so
  silu is inert), and the `FusedMoE.forward` block seam routes gpt-oss's experts to the
  miner kernel.
- **The anti-cheat gate works**, both ways it's measured:
  - KL gate: faithful silu → mean KL ~0 **PASS**; broken silu → mean KL ~14
    **FAIL**, score **0** (~4 orders of magnitude separation).
  - Benchmark gate: on Qwen2.5-1.5B, broken silu drops GSM8K 62.5%→0% while being
    26% *faster* → **FAIL**, score 0 — a faster kernel with wrong answers earns nothing.
  - On gpt-oss-120b, broken rmsnorm drops GSM8K 75%→0% → **FAIL**.
- **End-to-end KL catches subtle drift op-correctness misses.** A "faithful"
  rmsnorm passed per-op correctness but sat at KL 9.2e-3 vs a measured stock-vs-stock
  noise floor of 3.9e-4 (~24×) — correctly flagged. (The layered gate working.)
- **Robust scoring**: current authority recomputes one pooled charged rate per arm,
  brackets C with B/B′, and derives the bar from their spread; tamper-resistant timing
  (`mark_driver`) remains in place, and a faithful-but-slower kernel earns no title.
- **The current authority is stricter than the original mechanism:** finalized native
  commit-reveal intake, immutable publication, registered non-crown screening, isolated
  graph-on/audit-free B/C/B′, mandatory separate audit witness, pristine-T qualification,
  independent reproduction, and transactional target settlement.
- **gpt-oss-120b fits one 80 GB H100** (~69 GB at its native quantization), so the
  bootstrap doesn't need a B200 cluster.

**What was NOT yet proven at this stage:** a submitted kernel measuring faster than
sglang. That milestone landed later — 2026-07-07, the fused-epilogue collectives on
the MiniMax-M3 arena, 1.044–1.074× vs the noise bar (see the README's measured
record). Every bundle in `examples/` remains a correctness demo — the faithful ones
reproduce the model but are *slower* than sglang's own tuned kernels; the broken
ones are caught by the gate.

---

## Part 10 — Remaining work

1. **More kernels that beat sglang** — proven possible (2026-07-07: the fused-epilogue
   collectives measured 1.044–1.074× through the referee on the M3 arena); now it needs
   breadth. Eleven slots exist; the example bundles are correctness demos. The prizes are MLA/weight-absorbed attention,
   dense FP8/FP4 GEMM, and *comms-overlap* blocks (a block that owns its trailing reduce) —
   plus the multi-GPU surface (TP / PD-disaggregation / EP). Each new slot is a `SlotSpec` +
   a seam patch; the hard part is a kernel that wins, not the wiring.
2. **Cross-validator calibration**: pinned arena/runtime/model/topology identities,
   measured noise floors and workload mixtures reduce divergence; B300-specific
   false-crown, charged-tail, drift, SM103 and NVLink behavior still require B300 proof.
3. **Cross-validator determinism**: locked clocks, pinned HW/driver, deterministic
   mode on, more medians — so independent validators agree (Bittensor consensus).
4. **Full-logit KL** at a reference seam (vs top-k) as a tighter supplemental
   distribution gate where its stock-control premise holds.
5. **Mainnet operation** — chain integration, finalized SQLite intake, registered arena
   screening, settlement, and journaled weight publication exist. Deployment still needs
   an owned subnet, production validator permits/cadence, hosted bundle storage, backups,
   monitoring, and serving-registry rollout.
6. **Bigger models / multi-GPU** (DeepSeek-V4 scale) on the 8×B300 validator.
7. **A leaderboard/dashboard** over transactional intake and release state.

---

## Part 11 — File-by-file reference

The harness package, [../optima/](../optima):

| File | Role |
|---|---|
| [slots.py](../optima/slots.py) | The slot ABI. `SlotSpec` (`invoke_reference`/`invoke_entry` for non-uniform signatures, `kind` = op/block/collective, a `Correctness` mode, `prepare`/`prepare_from_layer` for quant-layout slots), the **11 slots** (silu, rmsnorm ops; attention.sdpa/decode/msa decode+prefill block scores + moe.fused_experts blocks; all_reduce, moe.fused_experts_reduce, ar_residual_rmsnorm, moe_finalize_ar_rmsnorm collectives), references, input generators, tolerances. Adding a slot = editing here. |
| [manifest.py](../optima/manifest.py) | Parse + validate `manifest.toml`. Schema + ABI check + **path-safety** (`_safe_relpath`). Pure-Python. |
| [target_catalog.py](../optima/target_catalog.py) | Pure validator policy for canonical singleton/atomic contribution identity, exact members, displacement/compatible overlap, and allowed features. It contains no crown, chain, or settlement policy; stack manifests bind catalog identity in the later assembly layer. |
| [sandbox.py](../optima/sandbox.py) | `scan_source` (AST policy tripwire), `load_entry` (import the kernel — isolate in prod). |
| [arena_service.py](../optima/arena_service.py) | Validator-owned arena identity, capacity/retry policy, fixed non-crown screen, admission, and qualification planning. |
| [stack_manifest.py](../optima/stack_manifest.py) | Evaluation and release stack identity, exact marginal replacement, integrated contribution refs, and integration-review authority. |
| [settlement.py](../optima/settlement.py) | Two-PASS reproduction candidate, conservative speed, transactional target-level settlement plan and evidence. |
| [registry.py](../optima/registry.py) | `KernelRegistry` (process-global `REGISTRY`), `KernelImpl`, `Eligibility`. The dispatcher's lookup table + active toggle. |
| [dispatch.py](../optima/dispatch.py) | `make_{silu_and_mul,rmsnorm,attention,moe,allreduce}_dispatcher` — the one place a miner kernel is called; validator owns the allocation, the call site, + fallback. |
| [seam.py](../optima/seam.py) | `activate()` (install seam + env-driven load), `mark_driver()` (tamper-resistant timing). Shared by bootstrap + plugin. |
| [bootstrap.py](../optima/bootstrap.py) | The `.pth`-loaded post-import hook that installs the seam in every interpreter, incl. the spawned scheduler. |
| [integrations/sglang_silu.py](../optima/integrations/sglang_silu.py) | Patches `SiluAndMul.forward_cuda/native`. |
| [integrations/sglang_norm.py](../optima/integrations/sglang_norm.py) | Patches `RMSNorm.forward_cuda/native` (fires on gpt-oss and every transformer). |
| [integrations/sglang_attention.py](../optima/integrations/sglang_attention.py) | Patches `RadixAttention.forward` (the attention **block** chokepoint; gathers paged KV for the decode swap). |
| [integrations/sglang_moe.py](../optima/integrations/sglang_moe.py) | Patches `FusedMoE.forward` (the MoE **block** chokepoint; opt-in `OPTIMA_MOE_SEAM=1`). |
| [integrations/sglang_allreduce.py](../optima/integrations/sglang_allreduce.py) | Patches `GroupCoordinator.all_reduce` (the **collective** chokepoint; opt-in `OPTIMA_COLLECTIVE_SEAM=1`). |
| [integrations/sglang_plugin.py](../optima/integrations/sglang_plugin.py) | Entry-point shim for SGLang's plugin framework (present at the default stable pin); `.pth` remains the primary spawn-safe path. |
| [verify.py](../optima/verify.py) | `verify_entry` — op/block correctness vs the slot's HP reference (`allclose` / `matched_ratio` / `cosine`); refuses `kind="collective"`. |
| [verify_collective.py](../optima/verify_collective.py) | `verify_collective` — DISTRIBUTED verify for collective slots: mp-spawns `world_size` ranks, runs the kernel as the real collective, compares to the fp32 cross-rank reduce. |
| [compat.py](../optima/compat.py) | `PINNED_SGLANG` + `run_checks` — the static seam canary (`optima compat`), re-run on every sglang bump. |
| [rebuild.py](../optima/rebuild.py) | The fenced escape hatch: applies only validator-shipped `repo_python` patchers (miner `bundle_python` is rejected). |
| [eval/_launch.py](../optima/eval/_launch.py) | `call_in_subprocess` — the spawn-safe fresh-process helper `cmd_verify` loads candidates through. |
| [eval/engine_worker.py](../optima/eval/engine_worker.py) | In-worker engine session: isolation probes, `engine_kwargs`, active/completed receipt gates. |
| [eval/oci_backend.py](../optima/eval/oci_backend.py) | Validator-owned OCI policy, no-egress/resource fence, leases, native prebuild and authoritative teardown. |
| [eval/oci_outer_session.py](../optima/eval/oci_outer_session.py) | Trusted-controller protocol for isolated candidate engine sessions and bounded evidence frames. |
| [eval/qualification_runner.py](../optima/eval/qualification_runner.py) | Graph-on/audit-free charged B/C/B′, mandatory separate eager/untimed audit role and typed host-regraded witness, pristine-T authority, graph proof, and aggregate verdict. |
| [bundle_hash.py](../optima/bundle_hash.py) | `content_hash` — deterministic bundle identity. |
| [finite_debt.py](../optima/finite_debt.py) | Pure content-addressed finite-claim issuance, expiry/cancellation, pro-rata allocation, and reserve-conserving projection arithmetic. |
| [incentive_shadow.py](../optima/incentive_shadow.py) | Signer-free, explicitly synthetic finite-debt projection against twice-reopened finalized membership; writes only `submitted=false` receipts. |
| [incentive_composition.py](../optima/incentive_composition.py) | Pure reviewed-discovery issuance/lifecycle and deterministic two-class epoch composition. |
| [incentive_composition_shadow.py](../optima/incentive_composition_shadow.py) | Signer-free synthetic composed projection against twice-reopened finalized membership; writes only `submitted=false` receipts. |
| [chain/intake.py](../optima/chain/intake.py) | SQLite production authority for finalized intake, screens, qualifications, reproductions, stacks, settlement and publication journals; exposes schema-5 pre-activation and schema-6 active incentive stores. |
| [chain/finite_debt_store.py](../optima/chain/finite_debt_store.py) | Additive schema-4 V2 authority: seeded clocks, atomic CROWN claim issuance, lifecycle, finalized family invalidation, projection, and confirmed-epoch close. Runtime-invalidity truth remains external. |
| [chain/incentive_composition_store.py](../optima/chain/incentive_composition_store.py) | Schema-5 D-013/D-015 authority plus atomic one-campaign schema-6 activation: review-pending retention/expiry, bounded bounty-only disposition, balances, and gapless composed projection. Registered promotion is intentionally rejected. |
| [chain/incentive_activation.py](../optima/chain/incentive_activation.py) | Wallet-free one-campaign cutover: canonical manifests/approval, finalized cursor, retained arena/catalog/family roster, reserve membership, and schema-6 receipt binding. |
| [chain/debt_publication.py](../optima/chain/debt_publication.py) | Restart-safe gapless V2 publication binding, finalized readback confirmation, rate-limited catch-up, and only-then debt-debit authority. |
| [chain/validator_loop.py](../optima/chain/validator_loop.py) | Finalized reveal → private fetch → immutable publication → registered arena screen/qualification → transactional settlement. |
| [model_provision.py](../optima/model_provision.py) | Exact all-file model-tree hashing and independently reopenable content-addressed receipts. |
| [release.py](../optima/release.py) | Signed chain-independent Engine release descriptor, deterministic source/wheel, SBOM/provenance, and OCI build context. |
| [cli.py](../optima/cli.py) | User/operator commands for verification/evaluation, chain intake, legacy/V2 weight reconciliation, one-campaign activation, signer-free incentive shadows, model provisioning, and release verification/context construction. |

Examples [../examples/](../examples): one (or more) bundle per slot —
`miner_silu_{triton,torch,broken,sparse}`, `miner_rmsnorm_{triton,broken}`,
`miner_attention_torch` / `miner_attention_decode_torch`, `miner_moe_fused_experts_torch`,
`miner_allreduce_torch` (`*_broken` = adversarial, must FAIL). The
[../tests/](../tests) tree covers submission policy, typed ABI and variants, distributed
verification/graph proof, OCI isolation and protocols, causal qualification, finalized
SQLite intake, settlement/economics/weight publication, stack assembly, model provisioning,
and signed release construction.

---

## Part 12 — How to run it

CPU dry-run (no GPU; exercises manifest → scan → load → op-correctness):

```bash
pip install -e ".[cpu,dev]"   # [cpu] pulls torch; a GPU box gets torch from its sglang install
optima slots
optima scan   examples/miner_silu_torch
optima verify examples/miner_silu_torch --device cpu --dtype float32
pytest tests/
```

GPU, on a CUDA box (the validated recipe — see the main README for the env setup
of `CUDA_HOME`, the `.pth`, and `python -m optima.cli` for spawn-safety):

```bash
# op-correctness on device: faithful PASSes, broken FAILs
python -m optima.cli verify examples/miner_silu_triton --device cuda --dtype bfloat16
python -m optima.cli verify examples/miner_silu_broken --device cuda --dtype bfloat16
```

End-to-end throughput + fidelity is validator-side: the chain intake loop runs the
qualification bracket in no-egress workers ([TESTNET.md](TESTNET.md)).

---

## Part 13 — Glossary

- **Bundle** — a miner's submission: manifest + kernel source + optional metadata.
- **Slot** — a typed operation in the model a miner may replace (e.g.
  `activation.silu_and_mul`). Defined by the validator in `slots.py`.
- **Seam** — the mechanism that routes a model op to the miner's kernel
  (`bootstrap.py` + `seam.py` + `dispatch.py`).
- **Controller / worker** — the trusted validator authority vs the isolated OCI process
  that constructs an engine and runs a candidate.
- **B / C / B′ / audit / T** — graph-on/audit-free baseline bookend, candidate, second
  baseline bookend; the separate eager/untimed audited candidate role; and the pristine
  teacher-forced quality authority.
- **KL** — divergence between candidate and baseline output distributions; the fidelity
  gate only where a stock-vs-stock control is approximately zero, and advisory on M3 where
  the in-engine audit is primary.
- **Champion / challenger** — the standing target contribution vs a paired,
  independently reproduced candidate trying to clear the measured bar.
- **Commit-reveal** — commit a hash first, reveal the bundle later; makes copying
  impossible.
- **Hotkey** — a miner's on-chain identity (SS58 address).
- **Eligibility** — declared (dtype, arch, size) a kernel supports; outside it,
  the dispatcher falls back to stock.
