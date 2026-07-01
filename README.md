# Optima — inference-throughput competition harness on SGLang

A validator harness for a Bittensor-style subnet where miners submit **kernels**
(Triton / CuteDSL) that get swapped into a **fixed** model at typed op-slots, and
are scored on **throughput** gated by **output fidelity** — measured two ways:
per-token **KL** against a reference run, and **task accuracy** on real benchmarks.

> **Want to compete? Start with [docs/MINER_GUIDE.md](docs/MINER_GUIDE.md)** — the
> miner on-ramp in plain language: what Optima is and how you earn, exactly how a
> kernel is scored (the gates and thresholds), the slots you can target, a 20-minute
> first-kernel walkthrough, the bundle format, how to test locally then on a GPU,
> how to actually find a real win, and how to submit. No prior subnet knowledge
> assumed.
>
> **New here (design)? Read [docs/HOW_OPTIMA_WORKS.md](docs/HOW_OPTIMA_WORKS.md)** — the
> full end-to-end explainer: what the validator does, what miners submit, the
> exact pipeline, how a kernel gets into the model, and the complete threat model
> (including the "fake the output via an API call" attack and why the op-slot
> design defeats it). *(Some of its prose predates the current slot catalog —
> this README is the current state of record.)*
>
> **Going to production?** [docs/SUBNET_BLUEPRINT.md](docs/SUBNET_BLUEPRINT.md)
> distills how a real Bittensor subnet (Affine) is built — chain plumbing, the
> service decomposition, DB-backed state, copy detection, and the isolation
> security pattern — and maps each onto Optima's production architecture.

## What is and isn't done

**Done & validated on real GPUs (H100, up to gpt-oss-120b; sglang 0.5.12.post1 / CUDA 13):**
the whole *mechanism* — typed op-slots, fused-*block* slots, **and a cross-GPU *collective*
slot** (a slot can be one op, a region behind one typed tensor boundary, or a collective
handed the process group), the seam that swaps an untrusted kernel into a spawned model
process, op-correctness, two-launch throughput measurement, the KL gate, a real-task
capability gate (GSM8K + MMLU), commit-reveal + king-of-the-hill scoring, and
tamper-resistant timing. **Seven slots:** `activation.silu_and_mul`, `norm.rmsnorm` (ops);
`attention.sdpa` / `attention.decode`, `moe.fused_experts` (blocks); `collective.all_reduce`
and `moe.fused_experts_reduce` (collectives, verified distributed). The last is the **block
that owns its trailing TP all-reduce** — the only contract that can express the compute-comm
**overlap** win (~75% of decode at scale), where a plain MoE slot can't (the validator there
replays a separate stock reduce). The **attention-decode swap is proven end-to-end on a live
Qwen** (the validator extracts the running model's paged KV and routes decode to the miner
kernel; a broken kernel is caught ~20×).

**Scoring is CUDA-graphs-ON** (graphs-off cripples the baseline ~4.5–6.5×, so it's never used
to score). The op seams (silu/rmsnorm) are graph-captured directly; the **block/collective
seams now run a kernel the miner DECLARES graph-capturable** (`graph_safe` in metadata) *inside*
the graph — so a real MoE/comms/overlap win is scorable in the only regime that matters. An
undeclared kernel stays eager-only (falls back to the trusted baseline in-graph, so it can't
wedge capture); the attention gather-MVP is still eager (a paged-direct, graph-safe contract is
the next rung). **Noise-robust scoring** (we can't lock clocks on rented pods): each candidate is
**bracketed by a baseline before AND after** (B,C,B'), the speedup is paired against their mean,
the bar is `1 + max(2%, k·measured-noise)` not a hand-picked 2%, and a round whose baselines
disagree past a tolerance is **NO-DECISION** (never crowns). `ignore_eos` is on for scoring so
both sides emit identical token counts.

**No kernel has beaten sglang — the optimization side is unproven.** This validates the
*referee*, not any optimization. Every example kernel is a correctness demo: faithful ones
reproduce the model and are *slower* than sglang's own tuned kernels; broken ones are caught
by the gate. The point so far is that the harness can tell a real kernel from a cheat and
time it tamper-resistantly — not that any submission moves throughput.

**Still open — including the actual goal:** a submitted kernel that genuinely beats sglang at
equal fidelity. Plus isolation for untrusted miners, chain integration, a real DB, bigger
slots (MLA / weight-absorbed attention, GEMM, comms-overlap blocks), and **eval calibration**
(see "Calibration findings").

## Status: validated end-to-end

Two-launch runs (baseline = stock kernels, candidate = miner kernel swapped into
the live model). The **broken** kernels are adversarial — faster-looking but they
degrade the model; the gate must reject them.

**Qwen2.5-1.5B, GSM8K benchmark gate:**

| Bundle | GSM8K base→cand | throughput | gate | score |
|---|---|---|---|---|
| `miner_silu_triton` (faithful) | 62.5% → 62.5% | 0.94× | **PASS** | 1.0 |
| `miner_silu_broken` (drops SiLU) | 62.5% → **0.0%** | 1.26× faster | **FAIL** | **0** |

The cheat is genuinely 26% faster yet scores **zero** because it can't do the
work anymore. *Fast-but-dumb = worthless.*

**gpt-oss-120b (single H100), GSM8K + KL:**

| Bundle | GSM8K base→cand | KL | gate |
|---|---|---|---|
| `miner_rmsnorm_broken` (skips norm) | 75.0% → **0.0%** | huge | **FAIL** (correct) |
| `miner_rmsnorm_triton` (faithful) | 75.0% → 58.3%* | 9.2e-3* | FAIL* |

\* We measured the control — stock-vs-stock KL (the nondeterminism floor) is
**3.9e-4** (1/2041 token flips). The faithful kernel's **9.2e-3 / 24-flips is ~24×
the floor**, so it's *real* drift, not sampling noise: this toy kernel isn't
bit-faithful to sglang's RMSNorm, and the **end-to-end gate correctly caught what
op-correctness (bf16 tolerance) passed**. What this validates: the RMSNorm seam **fires on a
120B MoE model** (gpt-oss fuses its activation into the MoE kernel, so `SiluAndMul`
is inert but `RMSNorm` fires), the cheat is caught hard (75%→0%), and the gate
caught a *subtle* real drift a per-op check missed.

### Calibration findings (from running on real hardware)

1. **The KL threshold must be calibrated to the model's nondeterminism noise
   floor**, not hand-picked. We measured it on gpt-oss-120b: stock-vs-stock KL with
   `--no-deterministic` is **3.9e-4** (1/2041 flips) — the floor. Set ε = k×floor
   (e.g. 5×), and run with `enable_deterministic_inference` so the floor → ~0 and
   kernel drift is cleanly attributable. (The faithful rmsnorm above sat at 24×
   the floor — genuinely above any sane threshold, correctly flagged.)
2. **Benchmark accuracy needs large n.** At n=12, GSM8K has a ~12% std; a 2-problem
   flip reads as "−16.7%." Use **KL as the dense, low-variance primary gate** and
   **benchmark accuracy as a capability floor at ~100–200 samples**.
3. **For a quantized model there's no fp32 ground truth** (gpt-oss is natively quantized), so the
   KL reference is the stock-kernel run; the threshold must tolerate benign
   rounding in either direction.
4. **Big MoE models need per-launch process isolation + deterministic scoring.**
   The two launches must each run in their **own process** (`call_in_subprocess`):
   on gpt-oss-120b in deterministic mode, running baseline then candidate in one
   driver process corrupted the candidate (NaN outputs → a *no-op* kernel "regressed"
   to 0%). With isolation, deterministic mode works and the stock-vs-stock KL floor
   is **~0** (a clean gate — validated: a no-op scores KL `0.0`, PASS). In
   **non-deterministic** mode the floor on the realistic long-generation workload is
   **1.17e-2** — *above* a 5e-3 gate — so a faithful kernel false-fails. Takeaway:
   **score big MoE in deterministic mode**; where that's unavailable, run
   `--kl-advisory` and let the **accuracy gate** carry quality. (KL is also now
   hardened: a genuinely degenerate candidate — all-non-finite logprobs — reads as
   maximal divergence, not 0.)
5. **The KL gate is not mean-only.** `kl_gate_ok` also caps the **argmax-disagreement
   rate** (default 1%) and an opt-in **p99 KL** — so a *sparse* cheat (bit-exact
   almost everywhere, a few tokens flipped) that keeps `mean_kl` under the threshold
   is still caught by the magnitude-independent flip rate. Calibrate the rate to the
   noise floor: in deterministic mode a faithful kernel sits at **0 flips**, so the
   default is safe; in advisory mode (big MoE) all KL checks are off and accuracy
   carries quality.
6. **Attention has a higher intrinsic KL floor than elementwise ops** (measured on
   the decode-attention swap). A faithful decode kernel — *any* reference SDPA — sits
   at **~6e-3 mean KL vs fa3's flash attention** (flash's online-softmax reduction
   rounds differently, and it compounds over layers), stable across kernel precisions
   and backends. So the **default 5e-3 gate (tuned for silu/rmsnorm) is too strict for
   attention** — the slot needs its own calibrated threshold (~k×6e-3). A broken
   decode kernel sits at **0.126 (20× higher)** and is caught either way; the floor is
   real, not a bug (op-correctness is exact). Per-slot KL thresholds are the fix.

## Repo layout

```
optima/
  slots.py                  # the slot ABI: SlotSpec catalog (7 slots; kind = op|block|collective)
  seams.py                  # single source of truth for the seam adapters (bootstrap/activate/compat derive from it)
  eval/scoring.py           # noise-robust speedup verdict (bookended A/B, noise-derived margin, no-decision)
  copy_fingerprint.py       # reformat-invariant near-copy fingerprint (AST-normalized)
  manifest.py               # bundle manifest parse + path-safety
  sandbox.py                # static policy scan + isolated load (defense-in-depth)
  registry.py               # kernel registry + eligibility + active toggle
  dispatch.py               # per-slot dispatchers — silu/rmsnorm/attention/moe/all_reduce
  verify.py                 # op/block correctness vs HP reference (allclose|matched_ratio|cosine)
  verify_collective.py      # DISTRIBUTED verify for collective slots (mp-spawn N ranks)
  rebuild.py                # fenced escape hatch: validator-shipped repo patchers only (no bundle code)
  compat.py                 # PINNED_SGLANG (0.5.13.post1; re-baseline pending) + the seam canary (`optima compat`)
  seam.py / bootstrap.py    # install the seam in every venv interpreter via a .pth
  integrations/
    sglang_silu.py / sglang_norm.py        # ops: SiluAndMul, RMSNorm
    sglang_attention.py / sglang_moe.py    # blocks: RadixAttention.forward, FusedMoE.forward
    sglang_allreduce.py                    # collective: GroupCoordinator.all_reduce
    sglang_plugin.py                       # entry point for sglang builds that have a plugin fw
  eval/
    throughput_kl.py        # two-launch throughput + KL (generic corpus; calibration smoke)
    capability.py           # two-launch throughput + KL + benchmark accuracy (the real-task scoring path)
    benchmarks.py           # Benchmark protocol + GSM8K & MMLU (HF), answer extraction
    kl.py / prompts.py / _launch.py
  bundle_hash.py            # deterministic bundle identity
  commit_reveal.py          # commit-reveal + king-of-the-hill ledger
  cli.py                    # slots|compat|scan|verify|evaluate|bench|hash|commit|reveal|ledger|settle
examples/
  miner_silu_{triton,torch,broken,sparse}/     # silu slot (faithful / CPU dry-run / adversarial / sparse)
  miner_rmsnorm_{triton,broken}/               # rmsnorm slot (faithful / adversarial)
  miner_attention_torch/ miner_attention_decode_torch/   # attention.sdpa / attention.decode (blocks)
  miner_moe_fused_experts_torch/               # moe.fused_experts (block)
  miner_allreduce_torch/                       # collective.all_reduce
tests/                                  # 75 tests (scanner, manifest, KL, verify, block/moe seams, collective, rebuild, commit-reveal)
```

## How a kernel gets into the model (the seam)

`sglang.Engine` forces `mp.set_start_method("spawn")` and runs the model in a
separate scheduler process, so a class-patch in the parent never reaches it. We
install the seam in **every** venv interpreter via a `.pth` file
(`import optima.bootstrap`) + a post-import hook that patches the target chokepoint the
moment its module loads — including in the spawned scheduler. Five chokepoints today:
`SiluAndMul` / `RMSNorm` (ops), `RadixAttention.forward` / `FusedMoE.forward` (blocks),
and `GroupCoordinator.all_reduce` (collective). The pinned sglang (0.5.13.post1, see
`optima/compat.py`) **does** ship a hook/plugin framework (`srt/plugins/hook_registry.py`,
added by PR #21388 — present at the pin), so migrating the seam to a sanctioned
`sglang.srt.plugins` entry-point hook is a tracked option (`integrations/sglang_plugin.py`
is the shim); the `.pth` path is kept primary today because it is version-independent and
known spawn-safe.

The validator does **two launches** of the same model (identical weights/seed):
baseline (`OPTIMA_ACTIVE=0`, stock kernels) and candidate (`OPTIMA_ACTIVE=1`,
miner kernel). Only the one op differs, so the throughput delta and the KL/accuracy
deltas are attributable to the kernel.

**Tamper-resistant timing:** the driver/timer process calls `seam.mark_driver()`
*before* importing sglang, so it never imports miner code; the kernel runs only in
the spawned scheduler, which the driver times over IPC. A malicious kernel can't
reach the clock.

## Run it

### CPU dry-run (no GPU)

```bash
pip install -e .
python -m optima.cli slots
python -m optima.cli verify examples/miner_silu_torch --device cpu --dtype float32
pytest tests/
```

### GPU (the recipe validated on an H100)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python sglang -e . ninja datasets
SP=$(.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')
echo 'import optima.bootstrap' > "$SP/optima.pth"     # install the seam everywhere
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PWD/.venv/bin:$PATH   # sglang JIT needs nvcc+ninja
export TORCH_CUDA_ARCH_LIST=9.0                        # set to your GPU arch (9.0=H100, 10.0=B200)

# op-correctness on device
.venv/bin/python -m optima.cli verify examples/miner_rmsnorm_triton --device cuda

# cheap KL smoke on a generic corpus (calibration / quick check)
.venv/bin/python -m optima.cli evaluate examples/miner_silu_triton \
    --model Qwen/Qwen2.5-0.5B-Instruct --no-deterministic            # KL gate

# the real scoring path: throughput on real benchmark prompts (GSM8K + MMLU, long
# CoT generation), gated on KL *and* task accuracy from the same run
.venv/bin/python -m optima.cli bench examples/miner_silu_triton \
    --model Qwen/Qwen2.5-1.5B-Instruct --benchmarks gsm8k,mmlu --samples 64

# gpt-oss-120b TP=4 (multi-GPU): plain-triton MoE, custom-allreduce off
.venv/bin/python -m optima.cli bench examples/miner_rmsnorm_broken \
    --model openai/gpt-oss-120b --benchmarks gsm8k,mmlu --samples 16 \
    --tp-size 4 --moe-runner-backend triton --disable-custom-all-reduce \
    --mem-fraction 0.85   # must FAIL (accuracy collapse and/or KL blowup)
```

## The submission ABI

A bundle is a directory: `manifest.toml` (data — which slots, where the source is)
+ kernel **source** + optional eligibility `metadata/`. The miner provides only the
slot's `entry` callable; the **validator** allocates outputs, owns the dispatch and
fallback, and does the registration. Adding a slot is a validator action in
`optima/slots.py` (+ a seam patch). A slot's `kind` is `op` (one fused op), `block`
(a region of several ops behind one typed boundary), or `collective` (a cross-GPU reduce —
handed the process group, verified distributed). Correctness is `allclose` for bit-faithful
ops, `matched_ratio` (≥ρ of elements within tol vs high-precision ground truth) for kernels
that legitimately differ (attention/fp8/absorbed), or `cosine` (vs the HP reference) for
low-bit kernels where element-wise tolerance is meaningless (FP4/FP8). A kernel that
targets a block/collective slot also declares `graph_safe` in its metadata to be run
(and scored) under CUDA graphs; undeclared kernels stay eager-only and fall back in-graph.
**Seven slots today:**

- `activation.silu_and_mul` — `entry(x, out)` — Qwen/Llama-class MLP (op).
- `norm.rmsnorm` — `entry(x, weight, out, eps)` — universal; fires on gpt-oss (op).
- `attention.sdpa` — `entry(q, k, v, out, sm_scale, causal)` — scaled-dot-product
  attention (block; the op-correctness demo of the wider boundary).
- `attention.decode` — `entry(q, k, v, seq_lens, sm_scale, out)` — paged-decode
  attention; the seam extracts the running model's paged KV and routes decode through
  it (block; eager-only gather MVP — a paged-direct, CUDA-graph-safe contract is next).
- `moe.fused_experts` — `(prepare, forward)` pair — SwiGLU fused experts; `prepare` owns
  the weight layout once at load, `forward(x, topk_ids, topk_weights, prepared, out)` runs
  per step (block; a quantized kernel carries its FP4/FP8 weight layout in `prepare`).
- `moe.fused_experts_reduce` — `(prepare, forward)`; `forward(x, topk_ids, topk_weights,
  prepared, out, group)` — the experts block that **owns its trailing TP all-reduce** (the
  compute-comm overlap lever). The kernel is handed the process group and fills `out` with
  the reduced output; the validator does NOT replay a reduce. Verified distributed vs the
  fp32 cross-rank sum of the per-rank expert outputs.
- `collective.all_reduce` — `entry(x, out, group)` — TP all-reduce (the comms waist); the
  validator owns the buffer + the process group; verified distributed vs the fp32
  cross-rank sum (`optima.verify_collective`).

## Anti-copy & scoring: commit-reveal + king of the hill

A round is `commit → reveal → evaluate → settle`:

```bash
optima commit  examples/miner_silu_triton --hotkey alice --salt s1 --round 0 --ledger l.json
optima reveal  examples/miner_silu_triton --hotkey alice --salt s1 --round 0 --ledger l.json
optima evaluate examples/miner_silu_triton --model Qwen/Qwen2.5-0.5B-Instruct \
    --ledger l.json --hotkey alice --round 0 --no-deterministic
optima settle  --round 0 --margin 0.02 --ledger l.json
```

- **commit-reveal** binds `H(content_hash, hotkey, salt)`; a reveal must match a
  prior commitment by that hotkey, so you can't reveal a bundle you didn't commit
  to (copying at reveal time is impossible). In production this is Bittensor's
  native commit-reveal (we keep only the off-chain scoring half).
- **copy detection** (`optima/copy_fingerprint.py`): cumulative **across rounds** (a copy
  in a later round is caught, not just same-round), on the exact content hash OR a
  **reformat-invariant fingerprint** (AST-normalized — a reflowed/recommented/renamed-
  whitespace copy with a fresh hash is still demoted). Fingerprints are computed **per slot**
  over each op's transitive bundle-local **import closure**, and the ledger also compares
  per-FILE fingerprint sets by **containment** — so padding the bundle with an extra op, or
  relocating a stolen body into an imported `_impl.py` behind a re-export shim, still demotes
  (while two miners merely vendoring the same public utility never match). A **structural**
  fingerprint (names and constants blanked) additionally flags rename + constant-tweak
  near-copies as an **advisory** at reveal (surfaced for review, never auto-demoted —
  skeletons can collide).
- **king of the hill**: a champion holds the emission; a challenger takes the title only by
  beating it by a margin. A copy ties → earns nothing. `optima settle --per-slot` runs a
  **champion per slot** and splits emission across slots, so a specialist who owns one slot
  is paid (vs winner-take-all giving 100% to the single best end-to-end bundle).

Robust scoring (see `optima/eval/scoring.py`), built for a validator that **can't lock GPU
clocks**: each launch does median-of-K timed passes; the candidate is **bracketed by a
baseline before and after** (B,C,B'); the speedup is paired against the baseline mean; the
bar is **derived from the measured baseline noise** (`1 + max(margin, k·noise)`) not a
hand-picked constant; a round whose bracketing baselines disagree past a tolerance is
**NO-DECISION** and cannot crown. The ledger records a crownable speedup or 0.0. Fidelity
gating beyond mean-KL: a **coverage (tail-mass) guard** catches a flattened distribution
whose visible head matches (top-k KL is blind to it), the argmax-rate catches sparse flips,
**per-slot KL thresholds** calibrate to each slot's floor (attention's ~6e-3 vs silu's), and
`aligned_kl` now counts early-stop as dropped positions. Plus per-epoch seeded prompts
(anti-overfit), **shape jitter** on the per-op verify (count dims vary per run, so a kernel
can't hard-code the verify shapes), `ignore_eos` so both sides emit identical token counts
and the throughput numerator is a driver-known fixed budget (not a scheduler-reported
count), a `max_running_requests` knob to score at a serving-realistic batch, and a
**stale-champion** flag at settle when the `PINNED_SGLANG` differs (re-baseline on a bump).

## Security model

With Triton/CuteDSL the miner's kernel is **Python that runs in the model
process**, so there's no artifact we can prove safe. The boundary must come from
how you run it (and the model is public, so there's no IP to steal):

- the kernel runs on the GPU box, **not** the process that holds chain keys / sets
  weights — those live on a separate CPU control box (Affine's SSH pattern);
- **no network egress** from the GPU box; **ephemeral** per-eval, wiped after;
- a **per-eval CUDA context + watchdog** (DoS / out-of-bounds writes);
- timing is already out-of-process (`mark_driver`); the static scan
  (`sandbox.scan_source`) is a tripwire, not the boundary.

`--framework-mode` and `--isolate` now fail closed if the candidate process
cannot prove no-egress network isolation. Use `--allow-unsafe-no-isolation` only
for local throughput debugging on dev pods that lack `CAP_SYS_ADMIN`; production
scoring should run the eval worker with real namespace support, or inside a
container/VM whose candidate side has `--network=none`.

Worst case for a fully-compromised kernel is one wrong score for itself;
cross-validator consensus catches a rogue validator.

## What's MVP vs. production

| Concern | Now | Production |
|---|---|---|
| Slots | 7: silu/rmsnorm, attention.sdpa/decode, MoE, all-reduce, MoE+reduce (overlap) | + MLA, FP8/FP4 GEMM, graph-safe paged attention |
| Throughput gain | **none — no submitted kernel beats sglang yet** | a kernel that beats sglang at equal fidelity |
| Model | up to gpt-oss-120b (1 GPU) | DSV4-scale (multi-GPU, TP/PD/EP) |
| Quality gate | mean-KL + coverage/argmax/per-slot-threshold + GSM8K/MMLU, det-mode default | full-vocab KL at a reference seam + large-n (100–200) benchmarks |
| Scoring noise | noise-derived margin + bookended A/B + no-decision (no clock-lock needed) | + interleaved per-iter A/B + locked clocks where available |
| Isolation | scan (hardened) + **out-of-process** verify | namespaces + no-egress + per-eval ctx + watchdog (needs Linux/root) |
| Champion | per-round, pin-staleness flagged | head-to-head re-eval vs a content-addressed bundle store |
| Chain | local JSON ledger | on-chain commit-reveal + set_weights |
| State | JSON | a real DB, single-writer weights |

## Adding a slot

1. Define a `SlotSpec` in `optima/slots.py` (`make_inputs`, `invoke_reference`,
   `invoke_entry`, `out_shapes`, a `Correctness` mode, tolerances). It must satisfy the
   four invariants in [docs/SLOT_CONTRACT.md](docs/SLOT_CONTRACT.md); if it can't, it
   belongs in the fenced escape hatch (`rebuild.py`), not the core.
2. If the slot needs a new chokepoint, add a seam patch under `optima/integrations/` (a
   dispatcher built with `make_*_dispatcher`) and a **single `SeamAdapter` entry in
   `optima/seams.py`** — the bootstrap watch-list, `seam.activate()`, and the `optima compat`
   canary all derive from that one table (no parallel list to edit).
3. Miners target the new slot by name in their manifest. (A `collective` slot is verified
   with `optima.verify_collective`, not `verify_entry` — see the contract doc. A
   block/collective kernel declares `graph_safe` in metadata to be scored under CUDA graphs.)
