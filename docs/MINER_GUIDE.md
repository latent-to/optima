# Optima miner guide

**Read this if you want to compete.** It takes you from "I've never heard of this"
to "my kernel is being scored," in plain language. It assumes you can write a GPU
kernel (PyTorch / Triton / CuteDSL) but knows nothing else about you.

For the deep design (threat model, exactly how a kernel is injected into a running
model, the full scoring math) see [HOW_OPTIMA_WORKS.md](HOW_OPTIMA_WORKS.md). This
guide is the on-ramp; that's the reference.

---

## 1. What Optima is, and how you earn

Optima is a Bittensor-style subnet that **pays you for making LLM inference faster
without making it worse.**

The validator runs a fixed, public model on its GPUs. You submit a **kernel** — a
small piece of GPU code for one operation in that model (an activation, a norm, an
attention block, the MoE experts, a collective). The validator swaps your kernel in,
runs the model, and measures two things:

1. **Throughput** — tokens/second, versus the model running stock (unmodified sglang).
2. **Fidelity** — does the model still produce the *same outputs*? Measured by
   per-token KL divergence against the stock run, plus real-benchmark accuracy.

**You earn if and only if your kernel is *both* faster *and* faithful.** A kernel
that's 30% faster but changes the model's answers scores **zero**. A kernel that's
perfectly faithful but slower than stock also scores zero (no speedup). The whole
game is: *a genuine speedup at equal quality.*

You don't have to understand Bittensor to develop a kernel. You write and test it
locally; the chain only matters when you submit (§9). To actually be paid you'll
need a Bittensor **hotkey** (wallet), but you can do everything up to submission
without one.

**Reality check (be honest with yourself):** as of this writing, *no submitted
kernel has beaten sglang.* sglang's built-in kernels are already heavily tuned. Most
faithful kernels you write will be *slower* than the baseline. Winning is real work —
§8 is about where the actual openings are. This isn't a faucet; it's a performance-
engineering competition with a tuned opponent.

---

## 2. The scorecard — exactly how you win or lose

Your kernel is put through gates in order. Fail any gate → score 0. Pass all → your
score is your **speedup**, and the highest score holds the slot.

### Gate A — op-correctness (cheap, local, no model)
The validator feeds your kernel deterministic inputs and compares its output to a
high-precision reference. This is a *sanity check* — "is this even the right
function?" — not the real anti-cheat. It catches a kernel that computes the wrong
thing (e.g. SiLU where the model wants SwiGLU). Run it yourself with `optima verify`.

### Gate B — fidelity (the load-bearing anti-cheat, on the real model)
With your kernel swapped into the live model, the validator compares the model's
**per-token output distribution** to the stock run:

| check | what it catches | default threshold |
|---|---|---|
| **mean KL** | the output distribution drifting | `5e-3` (most slots); `3e-2` for attention |
| **argmax-disagree rate** | a *sparse* cheat — bit-exact almost everywhere but a few tokens flipped | `1%` |
| **coverage / tail-mass** | a flattened distribution that fools top-k KL | loose by default |
| **benchmark accuracy** | the model getting *dumber* (GSM8K / MMLU) | no regression beyond ~2 points |

The KL threshold is **per-slot** and is calibrated to the model's own
nondeterminism floor (running the *same* stock kernel twice isn't bitwise identical).
Attention's flash-style softmax reorders arithmetic, so its floor is higher — hence
`3e-2`, not `5e-3`. **Don't claim your kernel "drifts" or panic about a small KL
until you've measured the stock-vs-stock floor for that model.**

### Gate C — throughput (is the speedup real?)
This is **CUDA-graphs-ON** scoring — graphs-off cripples the baseline ~5×, so it's
never used. Block/collective kernels must declare `graph_safe: true` in metadata to
run *inside* the graph; otherwise they fall back to the baseline (you can't win from
outside the graph).

The validator can't lock GPU clocks on rented boxes, so it doesn't trust a single
timing. It **brackets** your candidate between two baseline runs (B, C, B′), pairs
your speedup against the *mean* of the two baselines, and sets the bar from the
*measured* noise:

```
required speedup = 1 + max(2%, k · noise)      # k ≈ 2
```

where `noise` is the spread between the two baseline reads. If the two baselines
disagree by more than ~10%, the box is too noisy → **NO-DECISION** (the round is
discarded, nobody is crowned on a number that can't be reproduced). Token budgets are
held equal (`ignore_eos` on), so you can't win by emitting fewer tokens.

### Holding the slot — king-of-the-hill
The current best kernel for a slot is the **champion**. To take the title you must
beat it by the margin: `your_score ≥ champion_score × (1 + 2%)`. A bit-identical copy
ties, never clears the margin, and earns 0. There's **one champion per slot** and
emission is split across slots that have a champion — so you can win by *specializing*
in one slot, you don't need to beat everyone everywhere.

---

## 3. The slots — pick your target

A **slot** is one typed boundary the validator will swap your kernel into. Run
`python -m optima.cli slots` for the live list; today there are seven:

| slot | kind | what it computes | entry signature |
|---|---|---|---|
| `activation.silu_and_mul` | op | `silu(x[...,:d]) * x[...,d:]` | `entry(x, out)` |
| `norm.rmsnorm` | op | `x · rsqrt(mean(x²)+eps) · weight` | `entry(x, weight, out, eps)` |
| `attention.sdpa` | block | `softmax(qkᵀ·scale + causal) v` (GQA/MQA) | `entry(q, k, v, out, sm_scale, causal)` |
| `attention.decode` | block | decode attention over each request's cached K/V | `entry(q, k, v, seq_lens, sm_scale, out)` |
| `moe.fused_experts` | block | fused MoE experts (SwiGLU-MLP), a **(prepare, forward)** pair | `prepare(w13, w2)` + `forward(x, topk_ids, topk_weights, prepared, out)` |
| `collective.all_reduce` | collective | tensor-parallel all-reduce (sum across GPUs) | `entry(x, out, group)` |
| `moe.fused_experts_reduce` | collective | MoE experts that **own their trailing all-reduce** (the compute-comm overlap lever) | `prepare(...)` + `forward(x, topk_ids, topk_weights, prepared, out, group)` |

**Kinds:**
- **op** — a single elementwise/reduction op. Small, and sglang's versions are very
  tight. Hard to beat; good for learning the pipeline.
- **block** — a fused region behind one tensor boundary (attention, MoE). Bigger
  surface = higher ceiling, because you can *fuse* work the stock path splits.
- **collective** — spans GPUs; the validator hands you the process group. Verified
  *distributed* (you can't fake cross-rank comms). `moe.fused_experts_reduce` is the
  one slot that can express the experts↔all-reduce *overlap*, ~75% of decode at scale.

**Where the ceiling is higher:** op slots are a tuned wall. The block and collective
slots are where real wins live — fusion across the GEMM/comm boundary, format
specialization (FP8/FP4), and kernels the vendor never tuned for your GPU. See §8.

The full invariants every slot guarantees (and why they make cheating impossible)
are in [SLOT_CONTRACT.md](SLOT_CONTRACT.md). The one you must never break: **a slot
stays strictly upstream of the sampler.** You only ever fill a tensor the validator
allocated; you never see or produce the final tokens. That's what kills the
"run garbage, fetch the real answer from an API" attack.

---

## 4. Your first kernel in 20 minutes (CPU, no GPU)

```bash
git clone <repo> && cd optima
pip install -e .            # installs the `optima` CLI + test deps
pytest tests/              # sanity: should be all green
```

Look at the simplest example — a pure-PyTorch `silu_and_mul`:

```bash
cat examples/miner_silu_torch/manifest.toml
cat examples/miner_silu_torch/kernels/silu_and_mul.py
```

Run the two local gates against it (no GPU needed):

```bash
# static policy scan (no code execution)
python -m optima.cli scan    examples/miner_silu_torch
# op-correctness vs the reference, on CPU
python -m optima.cli verify  examples/miner_silu_torch --device cpu --dtype float32
```

Now make it yours: copy the bundle, edit the kernel, re-verify.

```bash
cp -r examples/miner_silu_torch my_silu && $EDITOR my_silu/kernels/silu_and_mul.py
python -m optima.cli verify my_silu --device cpu --dtype float32
```

Compare against the broken examples to see what a *failing* kernel looks like — they
exist on purpose:

```bash
python -m optima.cli verify examples/miner_silu_broken --device cpu --dtype float32   # fails op-correctness
```

That's the whole inner loop: edit → `verify` → repeat. No GPU, no cost.

---

## 5. The submission bundle — anatomy

A submission is a **directory** (a "bundle") with three parts:

```
my_bundle/
  manifest.toml          # what you're submitting (data only — never executed)
  kernels/
    my_kernel.py         # your kernel code
  metadata/
    my_kernel.json       # optional: eligibility (dtypes, GPU arch, graph_safe)
```

### manifest.toml

```toml
bundle_id  = "my-silu-v1"              # unique id, [0-9A-Za-z._-]
abi_version = "optima-op-abi-v0"       # must be exactly this

[[ops]]                                 # one [[ops]] block per slot you target
slot   = "activation.silu_and_mul"     # the slot id (from `optima slots`)
source = "kernels/my_kernel.py"        # relative path to your module
entry  = "silu_and_mul"                # the function name inside it
dtypes = ["bfloat16", "float16"]       # optional eligibility filter
metadata = "metadata/my_kernel.json"   # optional
# prepare = "prepare"                   # for (prepare, forward) slots like MoE
# setup   = "setup"                     # optional one-time engine setup (advanced)
# architectures = ["sm90", "sm100"]     # optional GPU-arch gate (sm90=H100, sm100=B200)
```

All paths are relative and must stay inside the bundle. One `[[ops]]` per slot; a
bundle can target several slots at once.

### The kernel contract

Your entry function **writes into a pre-allocated output tensor and returns `None`.**
The validator owns and allocates `out`; you fill it. You never return tensors and you
never allocate the output — that's the anti-cheat boundary.

```python
import torch
import torch.nn.functional as F

def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d].float()).to(x.dtype) * x[..., d:])
```

**(prepare, forward) slots** (MoE) have two functions. `prepare` runs **once** at
load on the raw checkpoint weights — do your weight relayout / quantization-packing
here; the validator caches the result. `forward` (the `entry`) runs every step and
receives that cached object as `prepared`:

```python
def prepare(w13, w2):                  # once, at load
    return {"w13": relayout(w13), "w2": w2.contiguous()}

def fused_experts(x, topk_ids, topk_weights, prepared, out):   # every step
    out.copy_(run_experts(x, topk_ids, topk_weights, prepared))
```

### metadata/*.json (optional but recommended)

```json
{
  "op": "activation.silu_and_mul",
  "dtypes": ["bfloat16", "float16"],
  "architectures": ["sm90", "sm100"],
  "graph_safe": true,
  "notes": "free text"
}
```

- `dtypes` / `architectures` — eligibility filters; outside them the validator runs
  the baseline instead of your kernel (so claim only what you support).
- `graph_safe: true` — **required for a block/collective kernel to be scored under
  CUDA graphs.** Without it, your block kernel falls back to the baseline in-graph and
  can't win. Only declare it if your kernel truly captures (no host syncs, no
  data-dependent shapes inside the graph).

### What you may NOT write (sandbox)

Submitted kernels are statically scanned before they're loaded. **Banned:** network
(`socket`, `urllib`, `requests`, …), process/FS escape (`subprocess`,
`multiprocessing`, `os.system`, `ctypes`, `shutil`, `tempfile`, …), dynamic code
(`eval`, `exec`, `compile`, `__import__`, `open`, `globals`), deserializers
(`pickle.loads`, `torch.load`, `marshal`, …), introspection escapes (`__globals__`,
`__subclasses__`, `__class__`, …), and dynamic `getattr/setattr` with a non-literal
name. **Allowed:** `torch`, `triton`, `triton.language`, normal Python, math, and
`from sglang... import ...` (you may reuse sglang's own helpers). If `optima scan`
flags you, that's why.

---

## 6. Testing with a GPU (the real scoring)

When local `verify` passes, measure on a GPU. You can use your own or rent one
(see [DEV_ENVIRONMENT.md](DEV_ENVIRONMENT.md) for the pod/toolchain recipe; set
`TORCH_CUDA_ARCH_LIST` to your arch — `9.0`=H100, `10.0`=B200, and have `nvcc`+`ninja`
on PATH for Triton JIT).

```bash
# 1) op-correctness on real shapes/dtypes
python -m optima.cli verify my_bundle --device cuda --dtype bfloat16

# 2) end-to-end throughput + KL (the real gate), on a small model first
python -m optima.cli evaluate my_bundle --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-prompts 64 --max-new-tokens 64

# 3) capability floor on a real task
python -m optima.cli bench my_bundle --model Qwen/Qwen2.5-1.5B-Instruct \
    --benchmarks gsm8k --samples 128
```

Always launch via `python -m optima.cli` — sglang spawns the scheduler with
`mp spawn` and the `__main__` guard matters.

Read the `evaluate` output like the validator does: a **speedup ≥ the noise-derived
bar**, **KL under the per-slot threshold**, and **no accuracy regression**. If the two
bracketing baselines disagree a lot, your box is too noisy — fix that before trusting
any number (lock clocks if you can, warm up, or run two replicas concurrently on
disjoint GPUs so drift cancels).

---

## 7. Why did my kernel fail?

| symptom | likely cause | fix |
|---|---|---|
| `scan` reports a banned construct | network / file / dynamic-code call in your kernel | remove it; reuse sglang helpers via `import`, don't `open()`/`exec()` |
| `verify` op-correctness fails | wrong math, wrong dtype handling, or shape mismatch vs the slot contract | check the exact signature in `optima slots`; compare to the matching example bundle |
| `evaluate` runs but your kernel never fires | dtype/arch eligibility excludes the run, or a block kernel isn't `graph_safe` so it fell back | widen `metadata`, declare `graph_safe`, set `OPTIMA_MOE_DEBUG=1`-style debug to confirm it fired |
| KL gate fails on a *faithful* kernel | you didn't measure the stock-vs-stock noise floor; your "drift" may be the model's own nondeterminism — or your kernel genuinely isn't bit-faithful | measure the floor first; for big MoE use deterministic mode or `--kl-advisory` and let the accuracy gate carry quality |
| speedup gate fails | your kernel is simply slower than sglang's (the common case) | profile; see §8 — a faithful-but-slower kernel is the default outcome, not a bug |
| score is 0 despite a speedup | a fidelity gate failed (fast-but-dumb), or it tied the champion and didn't clear the +2% margin | check the quality line in the report |
| NO-DECISION | the box was too noisy (baselines disagreed >~10%) | quieten the box; re-run |

---

## 8. How to actually find a win

This is the hard part and where most effort should go. Distilled from real
profiling sessions on real hardware:

1. **Measure, don't reason. The GPU is the only judge.** Every "this should be
   faster" intuition has a ~50% chance of being wrong on contact with hardware.
   Build the cheap measurement *before* the kernel.

2. **Size the lever before you build it — and use the *wall* fraction, not
   kernel-time-sum.** If your target op is X% of the decode *wall-clock* and you make
   it `S×` faster, the end-to-end gain is Amdahl: `1/((1−X) + X/S)`. A common trap:
   "this kernel is 59% of summed kernel time" is **not** "59% of the wall" — overlap
   and gaps mean the wall fraction is often far smaller. Confirm the lever's *wall*
   share with an end-to-end measurement before committing.

3. **Let the bound-type pick the tool.** Profile (e.g. `ncu`) to see whether the
   kernel is memory-bound, compute-bound, or co-limited:
   - memory-bound → lower precision (FP8/FP4) halves bytes and wins.
   - compute-bound on a dense GEMM → the vendor (cuBLAS/cutlass) is near-optimal;
     don't fight it.
   - co-limited → lowering precision speeds *only* the part it touches; the other
     limiter becomes the floor (e.g. a reduction the dot's precision can't shrink).
     Lower precision and bigger fusion help less than the byte count suggests.

4. **Concede what the vendor already did.** sglang's MoE megakernel already fuses
   GEMM+activation+requant+finalize; the all-reduce is already overlapped; dense
   GEMMs are cuBLAS. Re-fusing what's fused, or "overlapping" what's overlapped, is a
   dead end. **The open ground is the kernels the vendor did *not* tune for your GPU**
   — e.g. a model's own Triton kernels running un-optimized on a new architecture.

5. **State your regime — a win is (model, context length, concurrency)-scoped.** A
   kernel that helps at long context / high concurrency can be a no-op at short
   context / low concurrency, and vice versa. The decode-throughput regime under CUDA
   graphs is what's scored; optimize there, and report the regime your win holds in.

6. **Distrust a surprising win until it survives an adversarial check.** Big speedups
   are usually artifacts: clock drift, a thin low-batch corner that doesn't
   generalize, an unfair baseline (e.g. comparing kernels doing different amounts of
   work), or a fidelity regression you haven't measured. Bracket your timing, match
   the work both sides do, and check the quality gates before you believe it.

7. **The realistic shape of a win is ~1.1–1.4×, not 2×.** Kernels are a slice of the
   $/token stack; the strategy is to *stack* several regime-specific wins, not to find
   one giant lever. A clean, faithful 1.1× that survives the gates is a real win.

---

## 9. Submitting to the subnet

Submission uses **commit-reveal** so nobody can copy your bundle off-chain and
front-run you:

```bash
# 1) commit a hash of your bundle (during the commit window)
python -m optima.cli commit my_bundle --hotkey <YOUR_HOTKEY> --salt <random> --round N

# 2) reveal it (during the reveal window) — proves you committed it first
python -m optima.cli reveal my_bundle --hotkey <YOUR_HOTKEY> --salt <random> --round N
```

A reveal whose content matches an *earlier* commit by a different hotkey (exact hash
**or** a reformat-invariant fingerprint that ignores whitespace/renames) is flagged a
**copy** and earns 0. Copy detection is cumulative across rounds.

Settlement (run by the validator) scores all original, passing reveals and applies
king-of-the-hill (`settle`, or `settle --per-slot` for per-slot champions + emission
split). You can inspect ledger state with `optima ledger`.

You'll need a Bittensor hotkey for identity/payout; setting one up is standard
Bittensor wallet tooling (outside this guide). Everything *before* commit — writing,
verifying, GPU-evaluating — needs no chain and no hotkey.

---

## 10. Glossary

- **slot** — a typed boundary in the model the validator swaps your kernel into. Three
  kinds: **op** (one operation), **block** (a fused region), **collective** (spans
  GPUs).
- **bundle** — your submission directory: `manifest.toml` + `kernels/` + `metadata/`.
- **entry / prepare / setup** — the functions your kernel exposes. `entry` is the
  per-step op; `prepare` runs once on weights; `setup` (advanced) patches the engine
  once.
- **seam** — the mechanism that injects your kernel into the running model process
  (a `.pth` import hook), so a swap needs no fork of sglang.
- **baseline / candidate** — the model running stock (baseline) vs with your kernel
  (candidate). Your speedup is candidate ÷ baseline.
- **fidelity** — how faithfully your kernel reproduces the model's outputs. Gated by
  KL + benchmark accuracy.
- **KL (divergence)** — distance between the stock and your per-token output
  distributions; the primary fidelity gate.
- **graph-safe** — a kernel that can run inside a CUDA graph capture. Required for
  block/collective kernels to be scored (scoring is graphs-ON).
- **king-of-the-hill** — the best kernel per slot is champion; you take the title only
  by beating it by the margin (+2%).
- **commit-reveal** — submit a hash first, reveal later; defeats off-chain copying.
- **champion / challenger** — the slot's current holder vs a new submission.
- **NO-DECISION** — a round discarded because the box was too noisy to trust.
- **pinned sglang** — the single sglang version all validators score against
  (consensus); your dev box doesn't have to match it to develop.
- **hotkey** — your Bittensor wallet identity, needed to submit and be paid.
- **TP / EP / MoE** — tensor-parallel / expert-parallel / mixture-of-experts.

---

*Questions the deep docs answer:* the full pipeline and injection mechanism
([HOW_OPTIMA_WORKS.md](HOW_OPTIMA_WORKS.md)), the slot invariants
([SLOT_CONTRACT.md](SLOT_CONTRACT.md)), the GPU/toolchain setup
([DEV_ENVIRONMENT.md](DEV_ENVIRONMENT.md)), and how the scored sglang version is
pinned and bumped ([SGLANG_TRACKING.md](SGLANG_TRACKING.md)).
