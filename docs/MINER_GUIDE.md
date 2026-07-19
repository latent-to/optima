# Optima miner guide

**Read this if you want to compete.** It takes you from "I've never heard of this"
to "my kernel is being scored," in plain language. It assumes you can write a GPU
kernel (PyTorch / Triton / CuteDSL) but knows nothing else about you.

For the deep design (threat model, exactly how a kernel is injected into a running
model, the full scoring math) see [HOW_OPTIMA_WORKS.md](HOW_OPTIMA_WORKS.md). This
guide is the on-ramp; that's the reference.

---

## 1. What Optima is, and how you earn

Optima is a Bittensor-style subnet designed to **pay you for making LLM inference
faster without making it worse.** The ordinary competition route is a registered
singleton or atomic kernel target. Cross-cutting work has a separate fenced
reviewed-discovery route; it never silently inherits a registered target or reward.

The validator runs a fixed, public model on its GPUs. You submit a **kernel** — a
small piece of GPU code for one operation in that model (an activation, a norm, an
attention block, the MoE experts, a collective). The validator swaps your kernel in,
runs the model, and measures two things:

1. **Throughput** — tokens/second, versus the model running stock (unmodified sglang).
2. **Fidelity** — does the model still produce the *same outputs*? Measured by
   an in-engine audit of your kernel's real calls against the stock baseline, plus
   per-token distribution checks against a pristine reference transcript.

**On the registered route, you earn if and only if your kernel is *both* faster and
faithful.** A kernel
that's 30% faster but changes the model's answers scores **zero**. A kernel that's
perfectly faithful but slower than stock also scores zero (no speedup). The
requirement is a genuine speedup at equal quality.

You don't have to understand Bittensor to develop a kernel. You write and test it
locally; the chain only matters when you submit (§9). To actually be paid you'll
need a Bittensor **hotkey** (wallet), but you can do everything up to submission
without one.

**Expectation-setting (measured, not vibes):** sglang's built-in kernels are heavily
tuned, and most faithful kernels you write will measure *slower* than the baseline.
It is doable — the first submissions passed every gate on 2026-07-07: two related
fused-epilogue collective kernels (shallow and deep) measured 1.044–1.049× and
1.071–1.074× against the noise-derived bar on the MiniMax-M3 / 4×B300 arena, at
zero fidelity violations across ~12,500 audited calls each, reproduced on
independent prompt seeds. That is the realistic shape of a passing
submission: a few percent, earned from an opening the vendor left (§8), proven
through the gates. Plan for that, not for 2×.

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
Fidelity has two policy regimes selected and sealed by the validator-owned arena
authority; miners cannot select the regime. See [FIDELITY.md](FIDELITY.md):

- **`audit` — the in-engine audit** (used on arenas where two identical stock runs
  are *not* logit-identical, which includes the current MiniMax-M3 arena): an extra
  untimed launch randomly samples your kernel's real dispatcher calls, re-runs the
  stock baseline on clones of the same inputs, and compares under the slot's own
  verify tolerances. **Zero violations** (plus a minimum call coverage) is the gate;
  KL becomes advisory. The 2026-07-07 record submissions passed with ~12,500 audited
  calls / 0 violations each.
- **`kl` — rollout-KL** (valid only on arenas where a stock-vs-stock control
  measures ~0): the model's **per-token output distribution** is compared to the
  stock run:

| check | what it catches | default threshold |
|---|---|---|
| **mean KL** | the output distribution drifting | `5e-3` (most slots); `3e-2` for attention |
| **argmax-disagree rate** | a *sparse* cheat — bit-exact almost everywhere but a few tokens flipped | `1%` |
| **coverage / tail-mass** | a flattened distribution that fools top-k KL | loose by default |

In **both** modes, the arena's **task-accuracy evidence** (the `task_score` in the
pristine reference-quality record) gates the model getting *dumber*, paired against
the same run's baseline.

The mandatory separate eager/untimed audit role, exact slot×TP-rank receipt transport,
typed host-regraded witness, and fail-closed durable reopen are implemented and
CPU/mock-covered. Charged B/C/B′ stay graph-on and audit-free. These new causal bytes are
not GPU-qualified until the exact production M3 canary passes; unauditable attention slots
fail closed. In-process tampering, audit-role fingerprinting, and timed-workload
fingerprinting remain open residuals even after that canary.

The KL threshold is **per-slot** and is calibrated to the model's own
nondeterminism floor (running the *same* stock kernel twice isn't bitwise identical).
Attention's flash-style softmax reorders arithmetic, so its floor is higher — hence
`3e-2`, not `5e-3`. **Don't claim your kernel "drifts" or panic about a small KL
until you've measured the stock-vs-stock floor for that model** — on a noisy arena
a *bit-identical* kernel can read mean KL ~0.9, which is exactly why the audit mode
exists.

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
required speedup = 1 + max(0.5%, k · noise)      # k ≈ 2; margin floor 0.005
```

where `noise` is the spread between the two baseline reads. If the two baselines
disagree by more than ~10%, the box is too noisy → **NO-DECISION** (the round is
discarded, nobody is crowned on a number that can't be reproduced). Token budgets are
held equal (`ignore_eos` on), so you can't win by emitting fewer tokens.

### Moving the frontier — finite reward, not a perpetual throne

The current best kernel for a canonical target is the **champion** used as the next
baseline. A bit-identical copy ties, never clears the measured confidence bar, and
earns 0. You can still win by specializing in one target; you do not need to optimize
the complete model.

Under the selected but inactive V2 rule, two independent passing measurements and
a settled crown create a **finite claim**. Its base size is proportional to
`ln(your relative speedup) / ln(1.01)`, with a mild same-family elapsed-time bonus
that reaches 5% after 90 days and can never reach 10%. With no activation-seeded
prior family clock, the first crown has no age windfall. Claims are capped at their
issued principal and expire after 90 days,
so neither a champion nor a tiny lone win is normalized into permanent full emission.

At launch, MiniMax M3 is the only model campaign, so its claims use 100% of
registered claim sizing. A first 1% CROWN earns `0.9` full-vector days of principal; 4.4%
earns about `3.895`, and 5% about `4.413`. Adding more M3 target families does not
divide those amounts. The launch generation accepts exactly this one immutable campaign;
a second campaign, model rotation, and successor activation are unsupported future work.
Campaign share sizes the finite
claim; it is not a hard daily payout silo, so all open claims share available CROWN
capacity pro rata and can wait or expire under overload.

The tracked 14-day launch study gave 1, 2, and 5 independently winning weekly
families 100% collection even with saturated discovery. Its deliberately harsh
10-family weekly row collected 99.0211%; the year-long version falls much lower
under sustained overload. At 25% success probability, those measured launch rows
put a `$1,000-$1,500` campaign targeting a 4.4% win at roughly
`$1,027-$1,556` break-even value per full Optima vector-day. These are sensitivity
figures, not promised token prices or an assurance that real collection stays near
100% after the launch. A 1% optimization is much harder to justify at that rental
budget unless success odds or emissions are materially higher.

The campaign-sized registered-CROWN rule is implemented but inactive. The earlier
family-share rule's signer-free testnet shadow passed; it is historical evidence and
does not test the current campaign-policy bytes.
The selected discovery rule separately caps one reviewed bounty at 50,000
weight-ppm epoch units and gives it no family clock, time bonus, renewal, or
permanent title. Its 90-day clock starts when the qualified discovery win is
retained, not when review finishes: delay consumes the window and review at or
after expiry cannot mint. Overdue pending reviews have a landed terminal expiry
record, though production still has to schedule it reliably.

The preregistered D-014 sensitivity test supports a seven-day review-service target:
all 108 synthetic 0/1/7-day rows paid discovery principal fully, with no
expiry/unissued debt or CROWN paid-fraction regression and no more than the selected
55,555-ppm instantaneous CROWN-capacity dilution. Longer 30/60/89-day delays were
diagnostic only, and 90/120-day review issued no stale debt. This does not mean the
external review service is authoritative or that V2 has been activated. The publication
path is implemented but has no live receipt.

The selected pure policy intends promotion plus fresh requalification/CROWN or the
finite bounty, never both. The durable implementation currently retains
`review_pending` wins and can issue `bounty_only`, but rejects promotion until typed
promotion transport, target registration, fresh CROWN linkage, and cross-lane work
identity exist. So “never both” is not yet an end-to-end same-work guarantee. The
composed signer-free shadow passed on testnet netuid 307 at finalized
block 7,586,146 (metagraph size 6): explicitly synthetic states projected
850,000 ppm registered-CROWN, 50,000 ppm reviewed-discovery, and 100,000 ppm
reserve, totaling 1,000,000 ppm, with `submitted=false` (semantic digest
`3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`,
file SHA-256
`ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`).
It constructed no wallet and carries no review, settlement, D-015 publication, or
activation authority. Meaningful-emission activation has not been authorized.
The wallet-free one-campaign cutover and restart-safe debt publisher are implemented,
but neither has a live receipt. Launch still needs the exact MiniMax-M3 family/reserve
manifests plus a fresh shadow, independent review/runtime-invalidity authority,
membership-departure history, the promotion linkage above, the production audit GPU
canary, and actual activation/mainnet operations.
The exact formulas, examples, payout rules, and activation status are in
[INCENTIVES.md](INCENTIVES.md).

---

## 3. The slots — pick your target

A **slot** is one typed boundary the validator will swap your kernel into. Run
`python -m optima.cli slots` for the live list; today there are eleven:

| slot | kind | what it computes | entry signature |
|---|---|---|---|
| `activation.silu_and_mul` | op | `silu(x[...,:d]) * x[...,d:]` | `entry(x, out)` |
| `norm.rmsnorm` | op | `x · rsqrt(mean(x²)+eps) · weight` | `entry(x, weight, out, eps)` |
| `attention.sdpa` | block | `softmax(qkᵀ·scale + causal) v` (GQA/MQA) | `entry(q, k, v, out, sm_scale, causal)` |
| `attention.decode` | block | decode attention over each request's cached K/V | `entry(q, k, v, seq_lens, sm_scale, out)` |
| `attention.msa_block_score` | block | per-block scores for sparse-attention selection (the validator owns the top-k selection + the attend) | `entry(q, index_k, seq_lens, block_size, out)` |
| `attention.msa_prefill_block_score` | block | causal per-row block scores for sparse-attention prefill (the validator owns top-k selection + attend) | `entry(q, index_k, prefix_len, scale, block_size, out)` |
| `moe.fused_experts` | block | fused MoE experts (SwiGLU-MLP), a **(prepare, forward)** pair | `prepare(w13, w2)` + `forward(x, topk_ids, topk_weights, prepared, out)` |
| `collective.all_reduce` | collective | tensor-parallel all-reduce (sum across GPUs) | `entry(x, out, group)` |
| `moe.fused_experts_reduce` | collective | MoE experts that **own their trailing all-reduce** (the compute-comm overlap lever) | `prepare(...)` + `forward(x, topk_ids, topk_weights, prepared, out, group)` |
| `collective.ar_residual_rmsnorm` | collective | fused all-reduce + residual add + RMSNorm epilogue (the slot the first gate-passing submission targeted) | `entry(x, residual, weight, eps, out_norm, out_residual, group)` |
| `collective.moe_finalize_ar_rmsnorm` | collective | MoE finalize + all-reduce + residual + RMSNorm as one kernel; requires a `dep_patches` diff declared in `manifest.toml` (policy-allowlisted, applied to an overlay copy — see `optima/dep_policy.py`) | `entry(gemm_out, row_map, scales, residual, weight, eps, out_norm, out_residual, group)` |

**Kinds:**
- **op** — a single elementwise/reduction op. Small, and sglang's versions are very
  tight. Hard to beat; good for learning the pipeline.
- **block** — a fused region behind one tensor boundary (attention, MoE). Bigger
  surface = higher ceiling, because you can *fuse* work the stock path splits.
- **collective** — spans GPUs; the validator hands you the process group. Verified
  *distributed* (you can't fake cross-rank comms). `moe.fused_experts_reduce` is the
  one slot that can express the experts↔all-reduce *overlap*, ~75% of decode at scale.

**Where the ceiling is higher:** op slots are a tuned wall. The block and collective
slots have the larger openings — fusion across the GEMM/comm boundary, format
specialization (FP8/FP4), and kernels the vendor never tuned for your GPU. The
submissions that have passed the gates so far both landed on the fused-epilogue
collectives. See §8.

The full invariants every slot guarantees (and why they make cheating impossible)
are in [SLOT_CONTRACT.md](SLOT_CONTRACT.md). The one you must never break: **a slot
stays strictly upstream of the sampler.** You only ever fill a tensor the validator
allocated; you never see or produce the final tokens. That's what kills the
"run garbage, fetch the real answer from an API" attack.

---

## 4. Your first kernel in 20 minutes (CPU, no GPU)

```bash
git clone https://github.com/latent-to/optima && cd optima
python3 -m venv .venv && source .venv/bin/activate   # python -m venv ships pip; `uv venv` does NOT
pip install -e ".[cpu,dev]"   # the CLI + torch (CPU build) + pytest
pytest tests/                 # sanity: all green (a few skips are normal — they need the maintainers' local data)
```

(`[cpu]` exists because the core deliberately doesn't pin torch — a GPU box gets
torch from its sglang/CUDA install instead; see [GPU_SETUP.md](GPU_SETUP.md).)

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
# adversarial bundle: drops the SiLU, so it's faster but wrong — must FAIL
python -m optima.cli verify examples/miner_silu_broken_torch --device cpu --dtype float32
```

(`miner_silu_broken` is the same cheat as a Triton kernel — use it instead once
you're on a GPU box; Triton has no CPU/macOS wheels.)

That's the whole inner loop: edit → `verify` → repeat. No GPU, no cost. What CPU
`verify` proves is **op-correctness only** — throughput, CUDA-graph capture, and
the fidelity gates are measured on GPU (§6).

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

# Optional for a singleton; useful as an explicit compatibility assertion.
[competition]
target = "activation.silu_and_mul"
mode = "slot"                          # "slot" or a registered "atomic" target

[[ops]]                                 # one [[ops]] block per slot you target
slot   = "activation.silu_and_mul"     # the slot id (from `optima slots`)
variant = "general"                    # optional; required on every row when a slot repeats
source = "kernels/my_kernel.py"        # relative path to your module
entry  = "silu_and_mul"                # the function name inside it
dtypes = ["bfloat16", "float16"]       # optional eligibility filter
metadata = "metadata/my_kernel.json"   # optional
# prepare = "prepare"                   # for (prepare, forward) slots like MoE
# setup   = "setup"                     # engine-wide mutation; fenced framework lane only
# architectures = ["sm90", "sm100", "sm103"] # optional gate (H100, B200, B300)
```

All paths are relative and must stay inside the bundle. It may carry multiple
shape-specialized implementations of one slot: repeat `[[ops]]`, give every row a
unique explicit `variant`, and declare non-overlapping capability domains in each
row's metadata. Manifest order is never routing priority; a live call must match
exactly one variant or Optima runs stock.

`[competition]` requests the validator-owned contribution identity; it does not let
the bundle invent one. A singleton bundle may omit it and resolves to its one slot.
Several slots form one normal target only when their **exact semantic member set** is
registered as an atomic target. The catalog registers the proven deep fused-epilogue
pair under this corrected identity:

```toml
[competition]
target = "collective.moe_epilogue.v1"
mode = "atomic"
```

The catalog—not manifest row order—owns that target's canonical members, overlap,
displacement, compatible targets, and permitted contribution features. Unknown
multi-slot work is classified as unregistered and cannot silently become a
slot/atomic target or auto-earn. A separate fenced discovery ABI/review can retain
that qualified work as review-pending and, under the inactive selected composition,
can durably issue only one bounded bounty. Promotion into a registered boundary is
the selected policy direction but is deliberately rejected today until its typed
transport, target registration, fresh requalification/CROWN linkage, and same-work
identity across lanes are complete. The review decision is validator-owned, not
selected by the miner.
Shape or architecture specialization within a registered slot is still metadata/bundle-
only; it does not require a new target or Optima code change. Donor-era `mode = "system"`
syntax remains parseable only for migration and never creates a registered system title.

This contract layer does not partially rewrite the legacy score ledger: the existing
CLI/chain paths keep their historical per-slot identity until evaluation-stack assembly
and settlement migrate together. `[competition]` is canonical intake data now, not a
claim that old `ops[0]` economics have already been replaced.

`setup` is not an ordinary slot hook. It can mutate the whole engine, so Optima refuses
to import or execute a bundle that declares it unless the validator explicitly arms the
fenced framework lane. That lane requires candidate isolation and externally observed
token fidelity; an in-engine audit cannot grade a framework patch. Declaring `setup`
does not arm the lane on the miner's behalf.

The identity-only resolver classifies features declared in the manifest and marks its
external evidence incomplete. Reviewed patchers selected by `rebuild.json` are outside
that object; the separate intake resolver requires their exact trusted capability
projection before it can report complete feature evidence. The existing rebuild parser
and patcher allowlist remain authoritative—target resolution neither executes a plan nor
turns a miner-provided patcher name into permission.

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
  "capabilities": {
    "head_dim": 128,
    "block_size": [64, 128],
    "q_len": {"min": 1, "max": 4096},
    "phase": "prefill",
    "layout": "row_major"
  },
  "notes": "free text"
}
```

- `dtypes` / `architectures` — eligibility filters; outside them the validator runs
  the baseline instead of your kernel (so claim only what you support).
- `graph_safe: true` — **required for a block/collective kernel to be scored under
  CUDA graphs.** Without it, your block kernel falls back to the baseline in-graph and
  can't win. Only declare it if your kernel truly captures (no host syncs, no
  data-dependent shapes inside the graph).
- `capabilities` — the normative specialization domain. A scalar means exact,
  a list means one-of, and `{ "min": ..., "max": ... }` is an inclusive numeric
  range. Missing live fields do not act as wildcards: the validator reports the
  shape N/A and runs stock. Contradictory or overlapping variant domains are rejected.

`attention.msa_prefill_block_score` is the first binding with a complete rich live
descriptor. It supplies `dtype`, `architecture`, `head_dim`, `block_size`, `q_len`,
`kv_len`, `top_k`, `phase`, `layout`, `graph_mode`, `quant`, `tp_size`, and
`world_size` (plus the one-head call semantics). Other bindings currently expose only
their legacy eligibility facts; a capability field they do not explicitly supply fails
closed rather than being guessed from a similarly named tensor.

Output allocation is also slot-typed. Legacy slots receive inherited-dtype contiguous
outputs. MSA prefill receives an FP32 row-major score view whose row pitch may be padded;
its kernel must use the supplied strides and must not assume contiguous BF16 storage.
`optima verify` deliberately allocates the padded form and reports catalog shapes outside
a variant's selected dtype/architecture/TP context as N/A. Run verification once for each
arena dtype/context (`--tp-size` and `--world-size` when declared); sibling variants for
other contexts are neutral only when at least one row is applicable. If a bounded MSA
shape domain misses the static catalog, Optima synthesizes both random and causal probes
inside it. Probe allocation has validator-owned safety ceilings, so an untestably large
domain fails closed. A few probes do not prove an arbitrary wide range—the registered
arena's workload distribution remains the end-to-end coverage authority.

### Advanced: override submissions and dependency patches

Two tiers beyond the plain kernel bundle, both optional:

- **Override submissions** — for some slots you can submit a small *epilogue*
  composed onto a validator-owned base kernel (`base_kernel` / `override_point`
  in the manifest) instead of a full implementation. See
  [SUBMISSION_MODEL.md](SUBMISSION_MODEL.md);
  `examples/miner_m3_swigluoai_override` is a copyable override bundle.
- **`dep_patches`** — a bundle may declare a unified diff against a pinned
  dependency's sources (e.g. flashinfer) to export data a deep-fusion kernel
  needs. The diff is policy-allowlisted per dependency (`optima/dep_policy.py`)
  and applied by the validator to an overlay copy — the install is never mutated.
  Required by `collective.moe_finalize_ar_rmsnorm`.

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

When local `verify` passes, measure on a GPU. You can use your own or rent one from
any provider — [GPU_SETUP.md](GPU_SETUP.md) is the provider-agnostic checklist
(toolchain, the seam `.pth`, env vars, self-checks). The short version: set
`TORCH_CUDA_ARCH_LIST` to your arch (`9.0`=H100, `10.0`=B200) and have `nvcc`+`ninja`
on PATH for Triton JIT.

```bash
# op-correctness on real shapes/dtypes (count dims are jittered per run)
python -m optima.cli verify my_bundle --device cuda --dtype bfloat16
```

Always launch via `python -m optima.cli` — sglang spawns the scheduler with
`mp spawn` and the `__main__` guard matters.

The authoritative throughput + fidelity measurement is **validator-side**: your
bundle goes through graph-on/audit-free baseline, candidate, and baseline roles; a
separate eager/untimed audit role that produces a typed host-regraded witness; and an
untimed pristine reference launch inside no-egress workers. A PASS must then be
**independently reproduced** before settlement. There is no local command
that reproduces that authority — and that's deliberate: a score you could compute
locally would be a score you could game.

To estimate your win before submitting, A/B the serving throughput yourself with
stock sglang as the control, measured the way the validator measures (§8): warm up
and discard, run the candidate **bracketed by two baseline runs**, compare against
the mean of the brackets, and distrust any delta smaller than the spread between
your two baselines. If the bracketing baselines disagree a lot, your box is too
noisy — fix that before trusting any number (lock clocks if you can, warm up, or
run two replicas concurrently on disjoint GPUs so drift cancels).

---

## 7. Why did my kernel fail?

| symptom | likely cause | fix |
|---|---|---|
| `scan` reports a banned construct | network / file / dynamic-code call in your kernel | remove it; reuse sglang helpers via `import`, don't `open()`/`exec()` |
| `verify` op-correctness fails | wrong math, wrong dtype handling, or shape mismatch vs the slot contract | check the exact signature in `optima slots`; compare to the matching example bundle |
| `verify` fails only on *some* shapes | your kernel branches on shape or hard-codes dims | make it shape-generic; the verify shapes are jittered per run precisely to catch this |
| your module raises at import (`load_failed` receipt) | a GPU-only import on a CPU box, or a syntax error | guard GPU imports; make sure the module imports cleanly everywhere it might load |
| qualification aborts for missing `completed` coverage | the workload never reached an applicable variant on every rank/slot, or a selected path failed before producing the model-facing output | check `active` membership, `fired` routing diagnostics, capability/graph gates, and any `fallback` receipt; never interpret `fired` alone as execution |
| audit gate fails (audit mode) | your kernel's output differs from stock beyond the slot's tolerance on real calls — often dropped work | the audit re-ran stock on clones of your actual inputs; treat every violation as real |
| KL gate fails on a *faithful* kernel (kl mode) | you didn't measure the stock-vs-stock noise floor; your "drift" may be the model's own nondeterminism — or your kernel genuinely isn't bit-faithful | measure the floor first; on nondeterministic arenas the audit mode is the gate and KL is advisory |
| speedup gate fails | your kernel is simply slower than sglang's (the common case) | profile; see §8 — a faithful-but-slower kernel is the default outcome, not a bug |
| score is 0 despite a speedup | a fidelity gate failed, or the incumbent displacement margin wasn't cleared | check the quality evidence in the qualification record |
| NO-DECISION | the box was too noisy (baselines disagreed >~10%) | quieten the box; re-run |

**The phantom-pass (read this one).** A result that looks *too good* — KL exactly
0.0, accuracy delta exactly 0.0, a large speedup — usually means the candidate
engine came up **without your kernel**: missing seam `.pth`, a bad env var, or a
bundle load failure made the dispatcher fall back to stock, and you measured
stock-vs-stock. The driver therefore requires `active` from every expected scheduler
member, `completed` for every registered slot/member pair, and no selected-path
`fallback`. `fired` is deliberately weaker: it proves only that registry routing
selected a candidate, before adapter marshalling, the entry, and validator-owned tail
work finish. Missing completion or any fallback aborts the eval. These receipts catch
accidental phantom paths but remain forgeable inside today's shared candidate process;
external qualification and complete-engine isolation own correctness/crown authority.
Locally without the `.pth` installed, running stock is expected
(see [GPU_SETUP.md](GPU_SETUP.md)).

Common contract mistakes worth checking before anything else: returning a tensor
instead of writing into `out` (the dispatcher ignores your return value);
allocating the output yourself; mutating weights in `forward` (that belongs in
`prepare`); a host sync (`.item()`, `.cpu()`) inside a kernel you declared
`graph_safe` (capture fails → fallback → no scored result); ignoring the `group`
argument in a collective and reducing over a global.

---

## 8. How to find a real improvement

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
   - memory-bound → lower precision (FP8/FP4) halves bytes and usually pays.
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

5. **State your regime — an improvement is (model, context length, concurrency)-scoped.** A
   kernel that helps at long context / high concurrency can be a no-op at short
   context / low concurrency, and vice versa. The decode-throughput regime under CUDA
   graphs is what's scored; optimize there, and report the regime your improvement
   holds in.

6. **Distrust a surprising speedup until it survives an adversarial check.** Big speedups
   are usually artifacts: clock drift, a thin low-batch corner that doesn't
   generalize, an unfair baseline (e.g. comparing kernels doing different amounts of
   work), or a fidelity regression you haven't measured. Bracket your timing, match
   the work both sides do, and check the quality gates before you believe it.

7. **The realistic shape of a passing kernel is a few percent end-to-end, not 2×.**
   Kernels are a slice of the $/token stack; the strategy is to *stack* several
   regime-specific improvements, not to find one giant lever. The submissions that
   have passed the gates so far measured 1.04–1.07× end-to-end against the noise
   bar; a clean, faithful speedup of that shape is the target.

---

## 9. Submitting to the subnet

Everything *before* this section — writing, verifying, GPU-evaluating — needs no
chain and no wallet. Submission itself rides Bittensor's native **timelock
commit-reveal**: you commit `{content hash, fetch URL}` on-chain, encrypted until
the reveal block. Until then nobody (validators included) can read your URL, and
the reveal block is your priority timestamp — a later copy of your bundle is
detected and demoted, by exact hash **or** a reformat-invariant fingerprint that
survives whitespace/rename changes. Copy detection is cumulative across rounds.

**One-time setup — a wallet and a registered hotkey:**

```bash
pip install bittensor-cli
btcli wallet create        # coldkey (funds) + hotkey (identity); back up the mnemonics
# testnet TAO for registration comes from the Bittensor discord faucet
python -m optima.cli chain-register --netuid <NETUID> --network <wss-endpoint>
```

(Bittensor's own wallet documentation: <https://docs.learnbittensor.org/keys/wallets>.
The hotkey signs your submissions and receives emission; the coldkey only pays
the registration fee and never touches this repo's data path.)

**Per submission:**

```bash
# 1) package: tars the bundle exactly as the identity hash sees it, prints the content hash
python -m optima.cli chain-package my_bundle

# 2) host my_bundle.tar.gz anywhere the validator can HTTPS-fetch it (any static host);
#    the validator re-hashes what it downloads, so the archive must be byte-exact

# 3) commit hash + URL on-chain (timelock; ~1 KB rides the chain, not your code)
python -m optima.cli chain-submit my_bundle --url https://<where-you-hosted-it> \
    --netuid <NETUID> --network <wss-endpoint>

# check what the subnet currently sees (block, your uid, revealed submissions)
python -m optima.cli chain-status --netuid <NETUID> --network <wss-endpoint>
```

Submissions are made under the terms in
[SUBMISSION_TERMS.md](SUBMISSION_TERMS.md) (currently a published draft): in
short, you keep copyright, the operator gets a perpetual license to run and
commercialize submitted kernels, and emissions are the sole compensation.

After your reveal block passes, the validator loop fetches the archive, verifies
it re-hashes to the committed hash, runs copy detection, evaluates through the
full gate chain (§2), and settles king-of-the-hill per slot. The whole path —
commit on a public testnet through GPU evaluation to settlement — was run
end-to-end on 2026-07-08 ([TESTNET.md](TESTNET.md) is the operator-side runbook
with the record).

**Ready to submit when:**

- [ ] `optima scan` and `optima verify --device cuda` pass on your bundle
- [ ] your own bracketed sglang A/B (§6) shows a speedup that clears your measured
      baseline noise — the validator's bar is derived the same way
- [ ] block/collective kernels: `graph_safe` declared and the kernel actually captures
- [ ] the hosted tar.gz re-hashes to what `chain-package` printed

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
- **fidelity** — how faithfully your kernel reproduces the model's outputs. M3 gates it
  with the in-engine audit plus pristine-reference task/distribution evidence; deterministic
  arenas may gate rollout-KL after a near-zero stock control.
- **KL (divergence)** — distance between the stock and your per-token output
  distributions; the fidelity gate on deterministic arenas (advisory where the
  audit is the gate).
- **in-engine audit** — the fidelity mode on nondeterministic arenas: an untimed
  launch samples your kernel's real calls, re-runs stock on cloned inputs, and
  compares under the slot's verify tolerances; zero violations required.
- **receipts** — candidate-process control-flow diagnostics: `active` means loaded,
  `fired` means selected, `completed` means the full model-facing path returned, and
  `fallback` means selected work failed and stock was served. They prevent accidental
  phantom scoring but do not prove hostile-code isolation or correctness (§7).
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
([SLOT_CONTRACT.md](SLOT_CONTRACT.md)), the two fidelity modes and their measured
rationale ([FIDELITY.md](FIDELITY.md)), the GPU/toolchain setup
([GPU_SETUP.md](GPU_SETUP.md)), the chain loop from the validator's side
([TESTNET.md](TESTNET.md)), and how the scored sglang version is pinned and
bumped ([SGLANG_TRACKING.md](SGLANG_TRACKING.md)).
