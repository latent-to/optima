# Optima, explained end to end

This is the long-form explainer. By the end you should understand, with no gaps:

- what Optima is and the three roles (miner / validator / chain),
- **what the validator actually does** and what the "gold standard" should be,
- **what a miner submits** and why it's a kernel-slot, not a whole model,
- the entire lifecycle of a submission, file by file,
- the deep trick that gets an untrusted kernel *into* a running model,
- exactly how scoring (throughput + KL) and anti-copy (commit-reveal) work,
- **every failure mode I can think of**, which are defended and which are not,
- what is *proven on real hardware* vs what is still a stub.

It points at every file. Paths are relative to this doc (`docs/`), so
`../optima/slots.py` is the harness module `optima/slots.py`.

> **Current scope (kept in sync):** two slots (`activation.silu_and_mul`,
> `norm.rmsnorm`), two quality gates (per-token **KL** *and* real **benchmark
> accuracy** — Part 6), validated up to **gpt-oss-120b**. `README.md` is the live
> state-of-record (results + calibration); if it disagrees with this doc, README
> wins. We have **not** improved throughput yet — this validates the referee, not
> any optimization.

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
  2. **fidelity** — did the model's output stay correct? (measured as KL
     divergence against a trusted reference run)
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

### 2.1 The validator owns everything except one op

The validator owns the model weights, the model graph, the tokenizer, the
sampler, the benchmark prompts, the timing, and the reference. The **only** thing
the miner contributes is the implementation of **one operation** ("op slot") deep
inside the forward pass — for example the SiLU activation in the MLP. Everything
around that op is trusted validator code.

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

This is implemented as the **two-launch** evaluation in
[../optima/eval/throughput_kl.py](../optima/eval/throughput_kl.py): `evaluate()`
launches the model with the kernel **off** (baseline = the gold standard), then
again with it **on** (candidate), and diffs them.

> Nuance for later: today's reference is "the same model with stock kernels."
> That's perfect for kernels meant to be *numerically equivalent* (a faster
> silu). For kernels that legitimately change precision (a new quantization
> kernel), you'd instead reference a frozen **full-precision** run and widen the
> KL threshold. The machinery is the same; only the reference run and threshold
> change.

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
architectures = ["sm80", "sm86", "sm89", "sm90", "sm100"]
metadata = "metadata/silu_and_mul.json"
```

The manifest is **data, not code** — the validator parses it
([../optima/manifest.py](../optima/manifest.py)) to learn *which* slot the bundle
targets and *where* the kernel source lives. It does not run anything yet.

### 3.2 The op-slot ABI — the contract

The set of operations a miner is allowed to replace is the **op-slot catalog**,
owned by the validator in [../optima/slots.py](../optima/slots.py). Today there
are two slots, `activation.silu_and_mul` and `norm.rmsnorm`. A slot (`SlotSpec`)
declares everything the validator needs to *use* and *verify* a kernel without
trusting it:

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

With a kernel-slot, the miner controls **one cheap op** and the validator
produces the tokens and logprobs from the rest of the trusted model. There's no
final output to substitute, and faking a cheap op (silu) via an API is slower and
pointless. The kernel-slot design **structurally removes** the attack class you
were most worried about. The residual risk (the kernel is still *code* running
in-process) is real and is handled separately by isolation — see Part 8.

---

## Part 4 — The lifecycle of a submission

Here is the entire pipeline, with the file/function that does each step.

```
   MINER                              VALIDATOR
   -----                              ---------
1  commit  H(bundle,hotkey,salt) ───► record commitment      commit_reveal.Ledger.commit
                                      (no bundle seen yet)
                ... commit window closes ...
2  reveal  bundle + salt        ───► verify vs commitment     commit_reveal.Ledger.reveal
                                      detect copies            (earliest commit = original)
3                                     parse manifest           manifest.load_manifest
4                                     static policy scan        sandbox.scan_source
5                                     (build) load kernel       sandbox.load_entry   [isolated]
6                                     op-correctness vs ref     verify.verify_entry
7                                     end-to-end: 2 launches    eval.throughput_kl.evaluate
                                        baseline (kernel off)
                                        candidate (kernel on)
                                        -> throughput + KL
8                                     score + king-of-the-hill  commit_reveal.Ledger.settle
                                      -> weights to the chain
```

Step by step:

1. **Commit** ([../optima/commit_reveal.py](../optima/commit_reveal.py),
   `Ledger.commit`). The miner posts `H(content_hash, hotkey, salt)` — a hash that
   *binds* them to an exact bundle without revealing it. CLI: `optima commit`.

2. **Reveal** (`Ledger.reveal`). After the commit window, the miner posts the
   bundle + salt. The validator recomputes the hash and checks it matches a
   commitment *that hotkey* made earlier. This is what makes copying impossible
   (Part 7). CLI: `optima reveal`.

3. **Parse manifest** ([../optima/manifest.py](../optima/manifest.py),
   `load_manifest`). Validates schema, ABI version, and — importantly —
   **path-safety**: source paths must be relative, inside the bundle, no `..`
   escape, no absolute paths, no symlink escape (`_safe_relpath`).

4. **Static policy scan** ([../optima/sandbox.py](../optima/sandbox.py),
   `scan_source`). An AST pass over the kernel source that rejects obvious
   egress / code-execution patterns (`socket`, `subprocess`, `pickle.loads`,
   `torch.load`, `os.system`, `eval`, `__import__`, dunder-escape attributes…)
   *before* anything is imported. It is careful not to false-positive on Triton's
   `tl.load`/`tl.store`. **This is a tripwire, not the boundary** (Part 8).
   CLI: `optima scan`.

5. **Load the kernel** (`sandbox.load_entry`). Imports the miner module and pulls
   out the `entry` callable. This *runs miner code*, so in production it must
   happen inside the isolated worker (Part 5.4 / Part 8), not the trusted parent.

6. **Op-correctness** ([../optima/verify.py](../optima/verify.py),
   `verify_entry`). Generates deterministic inputs for the slot's standard shapes,
   runs the miner kernel and the slot's `reference`, and compares with an
   allclose-style tolerance. This is the cheap gate: a kernel that's outright
   wrong is rejected here, before the expensive end-to-end run. CLI:
   `optima verify`.

7. **End-to-end evaluate**
   ([../optima/eval/throughput_kl.py](../optima/eval/throughput_kl.py),
   `evaluate`). The two-launch run that produces throughput + KL. This is the
   heart of scoring; Part 6 dissects it. CLI: `optima evaluate`.

8. **Settle** (`Ledger.settle`). Applies king-of-the-hill: a challenger only
   becomes champion if it beats the current best by a margin; copies and
   non-improvers earn nothing. Emits weights for the chain. CLI: `optima settle`.

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

(We also learned sglang's released 0.5.9 has **no** plugin framework — that only
exists on bleeding-edge main — so the "register a plugin entry point" approach is
dead on the version that actually installs from PyPI.)

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
watches for the import of exactly one module —
`sglang.srt.layers.activation` — and, the moment that module finishes loading,
runs `seam.activate()` against it. This is the "post-import hook" pattern:

```python
class _SeamFinder(MetaPathFinder):
    def find_spec(self, fullname, ...):
        if fullname != "sglang.srt.layers.activation":
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

## Part 6 — How scoring works

All in [../optima/eval/throughput_kl.py](../optima/eval/throughput_kl.py),
`evaluate(cfg, bundle_path)`.

### 6.1 Two launches

```python
baseline  = _run_launch(cfg, prompts, bundle_path="",          active=False)  # gold standard
candidate = _run_launch(cfg, prompts, bundle_path=bundle_path, active=True)   # miner kernel
```

Same weights, same seed, same sampler, same prompts — the **only** difference is
the one op. So any delta is attributable to the kernel.

> **Each launch runs in its own fresh process** (`_launch.call_in_subprocess`).
> sglang + deterministic mode set process-global CUDA/torch state (deterministic
> algorithms, the cuBLAS workspace, the sampling backend); in a *shared* driver the
> baseline's state corrupted the candidate launch on big MoE models (observed on
> gpt-oss-120b: a no-op kernel "regressed" to 0%). Isolated processes make the two
> launches independent and free all GPU memory between them. On a 120B MoE the
> KL noise floor is also workload-dependent — score in `enable_deterministic_inference`
> mode (floor ~0) or run KL **advisory** (`--kl-advisory`) and rely on the accuracy
> gate; see README "Calibration findings".

### 6.2 Throughput (robust)

`_measure` does, per launch:

- a **warmup** generate (so JIT/compile/graph costs aren't timed),
- then **K timed** generates (`timed_iters`, default 3),
- records tokens/sec each time, reports the **median** + min/max/stdev (`spread`).

Tokens come from sglang's trusted `meta_info["completion_tokens"]`, not from the
miner. Timing uses `time.perf_counter` in the driver around `engine.generate`,
with `torch.cuda.synchronize()` on both sides so async work is fully counted.
Median-of-K is why a single noisy sample can't swing the score (we measured ~7%
run-to-run sd on a tiny model — exactly why the margin gate exists).

### 6.3 Fidelity (KL)

When `return_logprob=True, top_logprobs_num=k`, sglang returns, per generated
position, the top-k `(logprob, token_id, text)` — its actual output distribution.
We capture this for baseline and candidate.

[../optima/eval/kl.py](../optima/eval/kl.py):

- `kl_position(ref_topk, cand_topk)` turns each top-k list into a distribution and
  computes `KL(ref || cand)` over the union of their supports (with a floor).
- `kl_over_positions` averages across positions and reports `mean_kl`, `max_kl`,
  `p99_kl`, and `argmax_disagreements` (how many positions the top token differs).

The alignment (`_aligned_kl` in `throughput_kl.py`) is subtle and was a real bug
fix: greedy decoding means baseline and candidate can *diverge* in their token
sequence if the kernel changes an argmax. After a divergence, later positions
aren't comparable (different context). So we compare position *i*, **then** stop
if token *i* differs. Crucially we compare position 0 *before* checking — position
0 always has identical context (same prompt), so a kernel that derails the very
first token still gets a huge KL instead of "zero comparable positions."

> Honest limitation: top-k truncation means this approximates the true full-vocab
> KL. It's sharp enough to catch the cheats that matter (calibration collapse,
> dropped work) at k≥20, but production should capture full logits at a reference
> seam for the tightest gate.

### 6.4 Gates and score

```python
passed_quality  = mean_kl <= kl_threshold and num_positions > 0       # default 5e-3
passed_speedup  = speedup >= 1.0 + speedup_margin                     # default 1.02
score = speedup  if (passed_quality and passed_speedup)
        else 0.0 if not passed_quality                               # cheat -> 0
        else speedup                                                 # faithful but slow -> ~1.0, no title
```

Read that carefully — it encodes the whole philosophy:

- **fail quality → score 0**, no matter how fast. You cannot trade correctness for
  speed.
- **pass quality but below the speedup margin → no improvement**, so it can't take
  the title (Part 7), but it isn't punished.
- **pass both → score = the speedup.**

### 6.5 The realistic workload + the capability gate

The `evaluate` path above times throughput on a generic prompt corpus — fine as a
cheap KL/calibration smoke, but a *toy* workload (short generations on filler
prompts). The **scoring** path is `optima bench`
([../optima/eval/capability.py](../optima/eval/capability.py)): it drives both
launches with a few samples from **real benchmarks** at realistic generation
lengths, so the throughput we score reflects the decode-heavy work production
actually does — not a 64-token toy task.

Two benchmarks today
([../optima/eval/benchmarks.py](../optima/eval/benchmarks.py)): **GSM8K** (math,
chain-of-thought, ~256 tokens) and **MMLU** (knowledge, CoT-prompted to ~512). The
same `Benchmark` protocol takes SWE-bench / Tau-bench / KernelBench later (only
`check()` changes — run tests/tools in a sandbox instead of extracting an answer).

On that one realistic run we gate on **both**:

- **per-token KL** vs the baseline — the dense, low-variance *primary* gate, on the
  same prompts (a faithful kernel sits at the noise floor; a cheat blows up).
- **task accuracy** — no regression on any benchmark beyond a tolerance ("did the
  model still solve real problems?"); a capability *floor*, but noisy at small n,
  which is exactly why KL is primary (Part 6.6).

A faithful kernel preserves both; a kernel that secretly degrades the model drops
accuracy and/or blows up KL and scores zero, even if it looked fast. CLI:
`optima bench`.

> Honest scope: this fixes the *regime* (real tasks, decode-heavy lengths). It does
> **not** change that a single small op (silu/rmsnorm) is a tiny fraction of
> end-to-end runtime, so its *throughput* contribution can sit below the
> measurement noise floor — that is a slot-size problem (bigger slots like
> attention/MoE + op-isolated microbenchmarks), tracked separately from the gate.

Why this *simplifies* the design: our objective is **scalar** (throughput), so the
benchmarks are **AND-gates**, not score components — there is nothing to aggregate
with a geometric mean (unlike Affine, whose objective is a capability *vector*).
The broken kernel fails both (KL huge, accuracy → 0); a faithful kernel passes both.

### 6.6 Calibration (learned on real hardware)

The gates work; the *thresholds* must be calibrated, and we found this empirically
on gpt-oss-120b:

- **KL threshold must equal k× the nondeterminism noise floor**, not a hand-picked
  constant. With `--no-deterministic`, stock-vs-stock KL on gpt-oss-120b was
  **3.9e-4** (the floor). A genuinely-drifting kernel sat at **9.2e-3 (~24× the
  floor)** — correctly flagged. Always measure stock-vs-stock first; run with
  `enable_deterministic_inference` so the floor → ~0.
- **End-to-end KL catches what op-correctness misses.** That drifting kernel
  *passed* per-op correctness (bf16 tolerance) but failed end-to-end — the layered
  check (cheap per-op pre-filter → end-to-end gate) working as designed.
- **Benchmark accuracy needs large n.** At n=12, GSM8K's ~12% std turns a 2-problem
  flip into "−16.7%." Use KL as the dense, low-variance *primary* gate and
  benchmark accuracy as a *capability floor* at ~100–200 samples.

---

## Part 7 — Anti-copy and emissions

[../optima/bundle_hash.py](../optima/bundle_hash.py) and
[../optima/commit_reveal.py](../optima/commit_reveal.py).

### 7.1 Bundle identity

`content_hash(bundle_dir)` is a deterministic SHA-256 over the manifest + every
source file (sorted, length-prefixed, junk excluded). Two bundles with the same
content have the same hash; any change flips it. This hash is the thing
commitments bind to and the thing copy-detection compares.

### 7.2 Commit-reveal defeats copying

The problem: submissions are evaluated in the open, so a lazy miner could copy the
current leader's bundle and resubmit. Commit-reveal kills this:

- **Commit window:** each miner posts `H(content_hash, hotkey, salt)`. This hides
  the bundle (you can't tell what they committed to) but **binds** them to it.
- **Reveal window:** the miner posts `(content_hash, salt)`. A reveal is accepted
  only if `H(content_hash, hotkey, salt)` equals a commitment *that hotkey* posted
  earlier (`Ledger.reveal`).

A copier who only sees a rival's bundle at reveal time has **no matching prior
commitment** for it, so they cannot reveal it. And if two miners independently
committed to the same content, the **earliest commit (lowest sequence) is the
original**; later identical ones are marked copies (`original=False`).

### 7.3 King of the hill defeats ties and Sybils

`Ledger.settle(round, margin)`:

- Take all this round's **passing, original** scores (copies are excluded and
  listed in `rejected_copies`).
- The best is the **challenger**. It takes the title only if
  `challenger_score >= champion_score * (1 + margin)`.
- The champion gets the emission weight (winner-take-all baseline:
  `weights = {champion: 1.0}`).

Why this is copy- and Sybil-resistant: a copy *ties* the champion, never clears
the margin, so it earns nothing. The only way to earn is to genuinely beat the
best. (We validated this on GPU: a faithful-but-slower kernel scored 0.935 and
took no title — `weights: {}`.)

> Winner-take-all is the simple baseline. Production likely smooths it (reward the
> champion + a bounty to the most recent *distinct* improver, decay former
> champions) to keep miners engaged. The mechanism file is where that policy
> lives.

### 7.4 Mapping to Bittensor

In a real subnet: commitments live **on-chain** (subtensor commit-reveal),
bundles are fetched from a content-addressed store, `hotkey` is the miner's SS58
address, and `weights` is what the validator sets on-chain each epoch. The
semantics in `commit_reveal.py` are identical; the JSON ledger is a local stand-in
so the whole mechanism is testable without a chain or a GPU
([../tests/test_commit_reveal.py](../tests/test_commit_reveal.py)).

---

## Part 8 — The threat model (every failure mode I can think of)

This is the section you asked for. I group attacks by **what the miner is trying
to fake**, give each a verdict, and am explicit about residual risk. "Mitigated"
means *in the current design*; "needs isolation" means *requires the sandbox layer
that is specified but not yet built*.

### 8.1 Your attack: substitute the output (e.g. via an API call)

> *"The miner runs gibberish kernels while the real results come from an API call,
> so the KL diff looks correct."*

**Verdict: structurally defeated by the kernel-slot design — for the output. The
in-process variant is the real residual risk.**

- The miner only implements **one op** (silu). They never produce the final
  tokens or the logprobs — sglang does, downstream, from the whole forward pass.
  The KL is computed by the validator from sglang's logprobs (Part 5.6, step 8).
  There is **no final output to substitute**, and faking a cheap op via a network
  call is slower and pointless. The attack you described assumes whole-model
  submission, which we specifically don't allow (Part 3.3).
- **BUT** — and this is the honest residual — the kernel is still *arbitrary code
  running in the scheduler process*. The op-slot contract ("just fill `out` from
  `x`") is enforced only by convention. A malicious kernel's code could, in
  principle, reach beyond its op: monkeypatch the sampler, hook the model output,
  patch sglang's logprob computation, or exfiltrate weights. So the attack doesn't
  vanish — it **relocates** from "fake the output" to "escape the op-slot
  abstraction via in-process code." That is handled by **isolation** (8.4), which
  is the most important not-yet-built piece. The static scan (`sandbox.scan_source`)
  is a tripwire for the lazy version of this, not a guarantee.

### 8.2 Fake the result — pass KL without doing the real work

| # | Attack | Verdict | How |
|---|---|---|---|
| 1 | Return wrong `out` cheaply, hope it's "close enough" | **Mitigated** | KL gate. Demonstrated: broken silo → KL 14 → score 0. |
| 2 | Compute only some tokens/positions, leave the rest stale | **Mitigated** | The gate is no longer mean-only: `kl_gate_ok` also caps the **argmax-disagreement rate** and (opt-in) **p99 KL** (`kl.py`). A sparse cheat that keeps `mean_kl` low by corrupting a few positions still trips the flip-rate, which is magnitude-independent. (CPU regression test: a 3%-flip cheat with `mean_kl=1e-3` passes the old mean-only gate, fails the new one.) |
| 3 | Memoize/replay op outputs across calls | **Mitigated** | Activations differ per prompt/token; and prompts are fresh per epoch (`sample_prompts(seed)`). Caching doesn't help at the op level. |
| 4 | Behave correctly when measured for correctness, garbage when measured for speed | **Mitigated** | Quality and speed are measured on the **same** candidate run; there's no separate "correctness mode" to detect. |
| 5 | Special-case the op-correctness inputs (fixed seed/shapes in `verify_entry`) | **Partial** | Re-randomize the verify seed/shapes per epoch (the caller can pass `shapes`/`seed`). End-to-end KL on fresh prompts is the backstop. Residual: a kernel can still branch on shape. |

### 8.3 Fake the speed — inflate throughput

| # | Attack | Verdict | How |
|---|---|---|---|
| 6 | Monkeypatch the timer | **Mitigated** | `seam.mark_driver()` — the timing process never imports miner code (Part 5.5). |
| 7 | Fabricate the token count | **Needs isolation** | `completion_tokens` is computed *in the scheduler process — where the miner kernel also runs* (Part 5.5), so an in-process escape (8.4 #11) can inflate it without touching the driver's clock. **General rule: everything the scheduler sends the driver — token count, logprobs, text — is only as trustworthy as the (unbuilt) isolation; the wall-clock *denominator* is the only genuinely tamper-proof half of throughput.** Partial hardening short of isolation: count from the returned `output_ids` rather than a separate field, and/or fix the budget (`ignore_eos` + a set `max_new_tokens`) so the driver knows the count *a priori*. |
| 8 | Offload work to an untimed stream / return early | **Mitigated** | `torch.cuda.synchronize()` brackets the timer; async work is counted. |
| 9 | Be fast only on the benchmark shapes | **Partial** | The scoring corpus is now **real benchmarks** (GSM8K + MMLU, long CoT generation), sampled per epoch — far more shape-diverse than the old 40-prompt toy set — and a kernel that corrupts only certain shapes trips the argmax-rate gate (#2). Residual: a kernel can still *branch* on shape to be fast-where-measured; rotate/expand toward the serving distribution and consider random sequence lengths. |
| 10 | Manipulate GPU clocks | **Needs host op** | Lock clocks (`nvidia-smi -lgc`) at the validator host; not enforced in code yet. |

### 8.4 Escape the abstraction — arbitrary code execution

This is the real frontier and the biggest not-yet-built piece. The kernel is
Python that runs in the scheduler process.

| # | Attack | Verdict | How (required) |
|---|---|---|---|
| 11 | Patch other model components / the sampler / measurement from inside the kernel | **Needs isolation** | The seam only *invites* the kernel; it doesn't sandbox it. Run the scheduler in a locked-down process. |
| 12 | Exfiltrate weights / IP over the network | **Needs isolation** | Network namespace with **no egress**. Static scan blocks the obvious imports but isn't a boundary. |
| 13 | Attack the validator host / persist / hit other evals | **Needs isolation** | Ephemeral per-eval process + PID/mount/user namespaces, seccomp, no shared FS. |
| 14 | Hang the GPU (infinite-loop kernel) — DoS | **Needs watchdog** | Per-eval CUDA **context** + a watchdog that kills the whole context on timeout. CUDA can't preempt a running kernel cleanly. |
| 15 | Out-of-bounds device writes corrupt other tensors / leak across requests | **Needs isolation** | Device code isn't memory-safe within a context. One CUDA context per eval (MPS/MIG or one process per eval). |
| 16 | Supply-chain: malicious source runs at import/compile (global ctor, etc.) | **Partial** | Static scan + the load happening only inside the isolated worker. Source-only (no prebuilt binaries) + offline build help. |

**Bottom line for Part 8:** the *scoring* attacks (8.2, 8.3) are largely defeated
by the current design and demonstrated. The *isolation* attacks (8.4) are the
ones that still require the sandbox layer described in Part 10 before you can open
the door to genuinely untrusted miners. Today, "the VM is the sandbox" — fine for
a controlled demo, not for adversaries.

### 8.5 Mechanism / economic attacks

| # | Attack | Verdict | How |
|---|---|---|---|
| 17 | Copy the champion's bundle | **Mitigated** | Commit-reveal + copy detection + king-of-the-hill (Part 7). |
| 18 | Front-run a rival's reveal | **Mitigated** | Can't reveal what you have no prior commitment to. |
| 19 | Sybil (many identities split reward) | **Mitigated-ish** | Only improvement-over-best earns; copies earn 0; chain registration has a cost. |
| 20 | Overfit the eval distribution (great on eval prompts, useless in production) | **Ongoing risk** | Fresh per-epoch prompts from a corpus; rotate/expand toward the real serving distribution. This never fully "closes" — it's a tuning discipline. |
| 21 | Self-dealing validator | **Needs consensus** | Multiple validators + reproducible scoring so Bittensor consensus catches an outlier. Determinism work (Part 10) enables this. |

### 8.6 Reference / measurement validity

| # | Issue | Verdict | How |
|---|---|---|---|
| 22 | Reference itself wrong/drifting (if pinned to an API) | **Avoided** | Local stock-kernel reference, not an API (Part 2.2). |
| 23 | Per-op tolerance ≠ end-to-end quality | **Mitigated** | We gate end-to-end KL, not just per-op correctness. |
| 24 | Cross-validator score divergence (noise, HW) | **Partial** | Median-of-K + margin now; locked clocks + determinism + pinned HW still needed. |
| 25 | Numerical nondeterminism fails a faithful kernel | **Mitigated** | Tolerance in verify; `enable_deterministic_inference` available; greedy decoding for alignment. |

---

## Part 9 — What is proven (on a real H100)

Validated on a single H100 (sglang 0.5.9, torch 2.9.1), Qwen2.5 up to gpt-oss-120b:

- **The seam works on real models, including a 120B MoE.** Confirmed because the
  *broken* kernel changes the output. Two slots: `activation.silu_and_mul` (fires
  on Qwen/Llama) and `norm.rmsnorm` (fires on gpt-oss, whose activation is fused
  into the MoE kernel so silu is inert).
- **The anti-cheat gate works**, both ways it's measured:
  - KL gate: faithful silu → mean KL ~0 **PASS**; broken silu → mean KL ~14
    **FAIL**, score **0** (~4 orders of magnitude separation).
  - Benchmark gate: on Qwen2.5-1.5B, broken silu drops GSM8K 62.5%→0% while being
    26% *faster* → **FAIL**, score 0. *Fast-but-dumb = worthless.*
  - On gpt-oss-120b, broken rmsnorm drops GSM8K 75%→0% → **FAIL**.
- **End-to-end KL catches subtle drift op-correctness misses.** A "faithful"
  rmsnorm passed per-op correctness but sat at KL 9.2e-3 vs a measured stock-vs-stock
  noise floor of 3.9e-4 (~24×) — correctly flagged. (The layered gate working.)
- **Robust scoring**: median-of-K with spread; tamper-resistant timing
  (`mark_driver`); a faithful-but-slower kernel correctly earns no title.
- **Commit-reveal + king-of-the-hill**: commit→reveal→evaluate→settle, copy
  detection, margin gate. 21 unit tests green.
- **gpt-oss-120b at MXFP4 fits one 80 GB H100** (~69 GB), so the bootstrap doesn't
  need a B200 cluster.

**NOT proven / honest caveat:** we have **not improved on base sglang throughput
at all** — the example kernels are toy demos, *slower* than sglang's tuned kernels.
This validates the *referee*, not any optimization. See README for the live numbers.

---

## Part 10 — What is NOT done (production gaps)

In rough priority:

1. **Isolation** (8.4): per-eval process + namespaces + no network egress + GPU
   context isolation + watchdog. *Required before untrusted miners.* Today the VM
   is the trust boundary.
2. **More (bigger) slots**: two toy slots (silu, rmsnorm) aren't a competition.
   Real prizes are MLA/attention, MoE runner, GEMM — and the multi-GPU surface
   (TP / PD-disaggregation / EP). The slot system is built for this; each new slot
   is a `SlotSpec` + a seam patch. **Throughput improvement is the actual goal and
   has not started.**
3. **Cross-validator determinism**: locked clocks, pinned HW/driver, deterministic
   mode on, more medians — so independent validators agree (Bittensor consensus).
4. **Full-logit KL** at a reference seam (vs top-k) for the tightest fidelity gate.
5. **Chain integration**: on-chain commitments, content-addressed bundle fetch,
   real weight-setting. Today it's a local JSON ledger.
6. **Bigger models / multi-GPU** (DeepSeek-V4 scale) on the 8×B200 validator.
7. **A leaderboard/dashboard** over the ledger (mostly a fundraising artifact).

---

## Part 11 — File-by-file reference

The harness package, [../optima/](../optima):

| File | Role |
|---|---|
| [slots.py](../optima/slots.py) | The op-slot ABI. `SlotSpec` (with `invoke_reference`/`invoke_entry` so non-uniform signatures work), the `activation.silu_and_mul` and `norm.rmsnorm` slots, references, input generators, tolerances. Adding a slot = editing here. |
| [manifest.py](../optima/manifest.py) | Parse + validate `manifest.toml`. Schema + ABI check + **path-safety** (`_safe_relpath`). Pure-Python. |
| [sandbox.py](../optima/sandbox.py) | `scan_source` (AST policy tripwire), `load_entry` (import the kernel — isolate in prod), `probe_in_subprocess`. |
| [registry.py](../optima/registry.py) | `KernelRegistry` (process-global `REGISTRY`), `KernelImpl`, `Eligibility`. The dispatcher's lookup table + active toggle. |
| [dispatch.py](../optima/dispatch.py) | `make_silu_and_mul_dispatcher` / `make_rmsnorm_dispatcher` — the one place a miner kernel is called; validator owns alloc, the residual add, + fallback. |
| [seam.py](../optima/seam.py) | `activate()` (install seam + env-driven load), `mark_driver()` (tamper-resistant timing). Shared by bootstrap + plugin. |
| [bootstrap.py](../optima/bootstrap.py) | The `.pth`-loaded post-import hook that installs the seam in every interpreter, incl. the spawned scheduler. |
| [integrations/sglang_silu.py](../optima/integrations/sglang_silu.py) | Patches `SiluAndMul.forward_cuda/native`; `rebind_existing` safety net. |
| [integrations/sglang_norm.py](../optima/integrations/sglang_norm.py) | Patches `RMSNorm.forward_cuda/native` (fires on gpt-oss and every transformer). |
| [integrations/sglang_plugin.py](../optima/integrations/sglang_plugin.py) | Entry-point shim for sglang builds that *have* the plugin framework (not 0.5.9). |
| [verify.py](../optima/verify.py) | `verify_entry` — op-correctness vs the slot reference, allclose-style. |
| [eval/throughput_kl.py](../optima/eval/throughput_kl.py) | `evaluate` — the two-launch throughput + KL run, median-of-K, gates, score. |
| [eval/capability.py](../optima/eval/capability.py) | `evaluate_capability` — the real-task scoring path: two-launch throughput + **KL + benchmark accuracy** on real prompts (`optima bench`). |
| [eval/benchmarks.py](../optima/eval/benchmarks.py) | `Benchmark` protocol + `Problem` + **GSM8K & MMLU** (HF datasets, numeric + multiple-choice answer extraction) + registry. |
| [eval/_launch.py](../optima/eval/_launch.py) | Shared spawn-safe, tamper-resistant `launched_engine` + `engine_kwargs` (incl. `tp_size` / `moe_runner_backend`) used by both eval paths. |
| [eval/kl.py](../optima/eval/kl.py) | `kl_over_positions` / `aligned_kl` / `extract_per_prompt` — per-position top-k KL, per-prompt alignment, and sglang-output parsing shared by both eval paths. |
| [eval/prompts.py](../optima/eval/prompts.py) | `CORPUS` + `sample_prompts(n, seed)` — per-epoch prompt sampling. |
| [bundle_hash.py](../optima/bundle_hash.py) | `content_hash` — deterministic bundle identity. |
| [commit_reveal.py](../optima/commit_reveal.py) | `Ledger` — commit/reveal, copy detection, king-of-the-hill `settle`, persistence. |
| [cli.py](../optima/cli.py) | The driver: `slots scan verify evaluate hash commit reveal ledger settle`. |

Examples [../examples/](../examples): `miner_silu_triton` (real Triton, GPU),
`miner_silu_torch` (pure-torch, CPU dry-run), `miner_silu_broken` (adversarial,
must FAIL). Tests [../tests/](../tests): `test_static.py` (scanner, manifest,
eligibility, KL), `test_verify_cpu.py` (op-correctness, needs torch),
`test_commit_reveal.py` (the whole ledger mechanism).

---

## Part 12 — How to run it

CPU dry-run (no GPU; exercises manifest → scan → load → op-correctness):

```bash
pip install -e .
optima slots
optima scan   examples/miner_silu_torch
optima verify examples/miner_silu_torch --device cpu --dtype float32
pytest tests/
```

GPU, on a CUDA box (the validated recipe — see the main README for the env setup
of `CUDA_HOME`, the `.pth`, and `python -m optima.cli` for spawn-safety):

```bash
# op-correctness on device
python -m optima.cli verify   examples/miner_silu_triton --device cuda --dtype bfloat16
# end-to-end gate: faithful PASSes, broken FAILs
python -m optima.cli evaluate examples/miner_silu_triton --model Qwen/Qwen2.5-0.5B-Instruct --no-deterministic
python -m optima.cli evaluate examples/miner_silu_broken  --model Qwen/Qwen2.5-0.5B-Instruct --no-deterministic
# a full anti-copy round
optima commit  examples/miner_silu_triton --hotkey alice --salt s1 --round 0 --ledger l.json
optima reveal  examples/miner_silu_triton --hotkey alice --salt s1 --round 0 --ledger l.json
python -m optima.cli evaluate examples/miner_silu_triton --model Qwen/Qwen2.5-0.5B-Instruct \
    --no-deterministic --ledger l.json --hotkey alice --round 0
optima settle  --round 0 --ledger l.json
```

---

## Part 13 — Glossary

- **Bundle** — a miner's submission: manifest + kernel source + optional metadata.
- **Slot** — a typed operation in the model a miner may replace (e.g.
  `activation.silu_and_mul`). Defined by the validator in `slots.py`.
- **Seam** — the mechanism that routes a model op to the miner's kernel
  (`bootstrap.py` + `seam.py` + `dispatch.py`).
- **Driver / scheduler** — the validator's timing process vs the spawned process
  that runs the model (and the miner kernel).
- **Baseline / candidate** — the two launches: stock kernels (reference) vs miner
  kernel.
- **KL** — divergence between the candidate's and baseline's output distributions;
  the fidelity gate.
- **Champion / challenger** — the standing best kernel vs a new one trying to beat
  it by the margin.
- **Commit-reveal** — commit a hash first, reveal the bundle later; makes copying
  impossible.
- **Hotkey** — a miner's on-chain identity (SS58 address).
- **Eligibility** — declared (dtype, arch, size) a kernel supports; outside it,
  the dispatcher falls back to stock.
