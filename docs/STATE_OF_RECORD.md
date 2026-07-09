# Optima — state of record

The detailed, numbers-first record of what is built, measured, and proven:
results, gates, calibration findings, run recipes, the submission ABI, and the
security model. **Where any doc and this file disagree, this file wins** (it is
kept current). The [README](../README.md) carries the quickstart and
orientation; this file carries the record.

## What is and isn't done

**Done & validated on real GPUs (H100 up to gpt-oss-120b; 4×B300 MiniMax-M3-NVFP4;
CUDA 13; the scored sglang version is `PINNED_SGLANG` in `optima/compat.py`, which
also records that pin's validation state):**
the whole *mechanism* — typed op-slots, fused-*block* slots, **and cross-GPU *collective*
slots** (a slot can be one op, a region behind one typed tensor boundary, or a collective
handed the process group), the seam that swaps an untrusted kernel into a spawned model
process, op-correctness, bookended throughput measurement, the fidelity gates (in-engine
audit / KL), a real-task capability gate (GSM8K + MMLU), commit-reveal + king-of-the-hill
scoring, and tamper-resistant timing. **Ten slots:** `activation.silu_and_mul`,
`norm.rmsnorm` (ops); `attention.sdpa` / `attention.decode` / `attention.msa_block_score`,
`moe.fused_experts` (blocks); `collective.all_reduce`, `moe.fused_experts_reduce`,
`collective.ar_residual_rmsnorm`, and `collective.moe_finalize_ar_rmsnorm` (collectives,
verified distributed). `moe.fused_experts_reduce` is the **block that owns its trailing TP
all-reduce** — the contract that can express the compute-comm **overlap** win (~75% of
decode at scale), where a plain MoE slot can't (the validator there replays a separate
stock reduce); `collective.moe_finalize_ar_rmsnorm` goes deeper still — MoE finalize +
all-reduce + residual + RMSNorm as ONE kernel, enabled by a bundle-declared **`dep_patches`**
diff against the pinned flashinfer (policy-allowlisted, applied by a reviewed patcher to an
**overlay copy** — the install is never mutated). The **attention-decode swap is proven
end-to-end on a live Qwen** (the validator extracts the running model's paged KV and routes
decode to the miner kernel; a broken kernel is caught ~20×).

**Scoring is CUDA-graphs-ON** (graphs-off cripples the baseline ~4.5–6.5×, so it's never used
to score). The op seams (silu/rmsnorm) are graph-captured directly; the **block/collective
seams now run a kernel the miner DECLARES graph-capturable** (`graph_safe` in metadata) *inside*
the graph — so a real MoE/comms/overlap win is scorable in the only regime that matters. An
undeclared kernel stays eager-only (falls back to the trusted baseline in-graph, so it can't
wedge capture); the attention gather-MVP is still eager (a paged-direct, graph-safe contract is
the next rung). **Noise-robust scoring** (we can't lock clocks on rented pods): each candidate is
**bracketed by a baseline before AND after** (B,C,B'), the speedup is paired against their mean,
the bar is `1 + max(margin, k·measured-noise)` (margin floor 0.5% — real improvements stack at 1-2%;
the noise term is what guards an unstable box), and a round whose baselines disagree past a
tolerance is **NO-DECISION** (never crowns). `ignore_eos` is on for scoring so both sides emit
identical token counts.

**First gate-passing submission (2026-07-07): a submitted kernel measured faster than
stock sglang through the referee at equal fidelity.** The `miner_m3_fused_epilogue` bundle (fused AR+residual+RMSNorm collective, the
July-2 MiniMax-M3 campaign kernel) scored **1.044× against the noise-derived bar 1.038 —
PASS, noise-confident** — on the M3-NVFP4/4×B300 arena, graphs-on, with the full gate chain
green (distributed verify; GSM8K paired no-regression 93.8%/92.2%; in-engine audit 12,456
sampled calls, 0 violations), and **reproduced on an independent prompt seed** (1.049× vs
bar 1.005; audit 12,648 calls, 0 violations).

**Same day, the deep bundle went through the same gate: SCORE 1.074.** The
`miner_m3_fused_epilogue_deep` bundle adds `collective.moe_finalize_ar_rmsnorm` — the MoE
finalize + all-reduce + residual + RMSNorm fused into ONE kernel via a declared `dep_patches`
overlay against pinned flashinfer — and scored **1.074× vs bar 1.010** (audit 12,480 calls /
0 violations), **reproduced at 1.071× vs bar 1.037** (audit 12,636 / 0) on an independent
prompt seed. The deep increment over the shallow win (1.049 × 1.025 ≈ 1.074) matches the
July-2 campaign's +2.7% claim for the fused epilogue. Landing it surfaced a real seam
hazard: upstream `minimax_m3` never wires `is_last_layer`, so sglang lets the final layer
defer its all-reduce — harmless stock, fatal with a skipped finalize; the seam now vetoes
the last-of-forward arm from the model's own layer count (`optima/moe_export.py`). Every
other example bundle remains a correctness demo (faithful but slower). (Full run records
live in the local `experiments/` ledger on the dev machine — gitignored, like `WORKLOG.md`;
the numbers above are the record of record.)

**Chain integration is live (2026-07-08): the deep bundle came back through the chain.**
The full miner→chain→validator loop ran on the public Bittensor testnet (netuid 307):
the deep bundle was committed from a miner hotkey via the chain's native **timelock
commit-reveal** (`set_reveal_commitment` — the bundle URL is drand-encrypted until the
reveal block, which doubles as the anti-copy priority timestamp), and `optima
chain-validate` discovered it, fetched the artifact, **re-hashed it against the
committed content hash**, fingerprinted it for copy detection, drove the full referee
on the GPU box, and crowned it: **1.072× vs bar 1.026, in-engine audit 12,824 calls /
0 violations — SCORE 1.0717**, the third independent reproduction of the deep win
(1.074 / 1.071 / 1.072), this one with no human in the path. `optima chain-package/
chain-submit/chain-status/chain-validate/chain-register` are the operator surface;
`docs/TESTNET.md` is the runbook. Weight policy is read from ONE seam
(`Ledger.current_weights`) so the planned relative-improvement emission scheme swaps
in without touching chain I/O.

**Still open:** isolation for untrusted miners; a tiered eval scheduler; a real DB;
mainnet economics (own subnet, stake/permits, hosted bundle storage); bigger slots
(MLA / weight-absorbed attention, GEMM, comms-overlap blocks); and **eval
calibration** (see "Calibration findings").

## Status: validated end-to-end

Two-launch runs (baseline = stock kernels, candidate = miner kernel swapped into
the live model). The **broken** kernels are adversarial — faster-looking but they
degrade the model; the gate must reject them.

**Qwen2.5-1.5B, GSM8K benchmark gate:**

| Bundle | GSM8K base→cand | throughput | gate | score |
|---|---|---|---|---|
| `miner_silu_triton` (faithful) | 62.5% → 62.5% | 0.94× | **PASS** | 1.0 |
| `miner_silu_broken` (drops SiLU) | 62.5% → **0.0%** | 1.26× faster | **FAIL** | **0** |

The cheat is genuinely 26% faster yet scores **zero** because it no longer does
the work: a faster kernel that changes the model's answers earns nothing.

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
  slots.py                  # the slot ABI: SlotSpec catalog (10 slots; kind = op|block|collective)
  seams.py                  # single source of truth for the seam adapters (bootstrap/activate/compat derive from it)
  eval/scoring.py           # noise-robust speedup verdict (bookended A/B, noise-derived margin, no-decision)
  audit.py                  # the IN-ENGINE AUDIT: sampled per-call stock-baseline comparison inside the scored engine
  receipts.py               # seam-health receipts (bundle loaded / impl fired / audit / export counters) by file
  copy_fingerprint.py       # reformat-invariant near-copy fingerprint (AST-normalized)
  manifest.py               # bundle manifest parse + path-safety (+ dep_patches declarations)
  sandbox.py                # static policy scan + isolated load (defense-in-depth)
  registry.py               # kernel registry + eligibility + active toggle
  dispatch.py               # per-slot dispatchers — silu/rmsnorm/attention/moe/all_reduce/arfusion(+deep consume)
  moe_export.py             # deep-seam export/consume state machine (fe_export ABI, last-layer veto)
  dep_policy.py / deppatch.py  # dep_patches tier: per-dep allowlist policy + unified-diff apply to an OVERLAY copy
  patchers/                 # the reviewed patcher scripts a rebuild plan may run (apply_dep_patch, build_cuda_ext)
  verify.py                 # op/block correctness vs HP reference (allclose|matched_ratio|cosine)
  verify_collective.py      # DISTRIBUTED verify for collective slots (mp-spawn N ranks; count-dim jitter)
  rebuild.py                # fenced escape hatch: validator-shipped repo patchers only (no bundle code)
  compat.py                 # PINNED_SGLANG (0.5.13.post1) + the seam canary (`optima compat`)
  seam.py / bootstrap.py    # install the seam in every venv interpreter via a .pth
  integrations/
    sglang_silu.py / sglang_norm.py        # ops: SiluAndMul, RMSNorm
    sglang_attention.py / sglang_moe.py    # blocks: RadixAttention.forward, FusedMoE.forward
    sglang_allreduce.py                    # collective: GroupCoordinator.all_reduce
    sglang_arfusion.py                     # collective: fused AR+residual+RMSNorm epilogue chokepoint
    sglang_defer_gate.py / sglang_moe_export.py  # deep seam: LayerCommunicator scoping + fused-moe export wrap
    flashinfer_overlay.py                  # routes the engine's flashinfer import to the patched overlay copy
    sglang_plugin.py                       # entry point for sglang builds that have a plugin fw
  eval/
    throughput_kl.py        # bookended throughput + fidelity (audit|kl modes; calibration smoke)
    capability.py           # throughput + fidelity + benchmark accuracy (the real-task scoring path)
    benchmarks.py           # Benchmark protocol + GSM8K & MMLU (HF), answer extraction
    kl.py / prompts.py / _launch.py
  bundle_hash.py            # deterministic bundle identity
  commit_reveal.py          # commit-reveal + king-of-the-hill ledger
  cli.py                    # slots|compat|scan|verify|evaluate|bench|hash|commit|reveal|ledger|settle
optima_kernels/
  collective/               # validator-owned reference lib for the fused AR+norm family (sm103 CUDA + wrapper)
examples/
  miner_silu_{triton,torch,broken,broken_torch,sparse}/   # silu slot (faithful / CPU dry-run / adversarial ×2 / sparse)
  miner_rmsnorm_{triton,broken}/               # rmsnorm slot (faithful / adversarial)
  miner_attention_torch/ miner_attention_decode_torch/   # attention.sdpa / attention.decode (blocks)
  miner_moe_fused_experts_torch/               # moe.fused_experts (block)
  miner_allreduce_torch/                       # collective.all_reduce
  miner_moe_fused_experts_reduce_torch/        # moe.fused_experts_reduce (experts + owned reduce)
  miner_m3_swigluoai_override/                 # the override submission tier (base_kernel + override_point)
  miner_setup_demo/                            # framework-mode demo: a setup() engine patch, gated by token-match
tests/                                  # the test suite (scanner, manifest, fidelity/audit, verify, seams, deep seam, dep_patches, collective, rebuild, commit-reveal, chain)
```

## How a kernel gets into the model (the seam)

`sglang.Engine` forces `mp.set_start_method("spawn")` and runs the model in a
separate scheduler process, so a class-patch in the parent never reaches it. We
install the seam in **every** venv interpreter via a `.pth` file
(`import optima.bootstrap`) + a post-import hook that patches the target chokepoint the
moment its module loads — including in the spawned scheduler. The chokepoints today
(one `SeamAdapter` row each in `optima/seams.py`): `SiluAndMul` / `RMSNorm` (ops),
`RadixAttention.forward` / `FusedMoE.forward` (blocks), `GroupCoordinator.all_reduce`
(collective), the fused AR+residual+RMSNorm epilogue behind
`--enable-flashinfer-allreduce-fusion` (arfusion), and the deep-seam pair
(`LayerCommunicator` defer-gate + the fused-moe export wrap). The pinned sglang (0.5.13.post1, see
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
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu,dev]"     # [cpu] pulls torch; a GPU box gets torch from its sglang install instead
python -m optima.cli slots
python -m optima.cli verify examples/miner_silu_torch        --device cpu --dtype float32
python -m optima.cli verify examples/miner_silu_broken_torch --device cpu --dtype float32   # must FAIL
pytest tests/
```

### GPU (the recipe validated on an H100)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "sglang==<PINNED_SGLANG from optima/compat.py>" -e . ninja datasets
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
**Ten slots today:**

- `activation.silu_and_mul` — `entry(x, out)` — Qwen/Llama-class MLP (op).
- `norm.rmsnorm` — `entry(x, weight, out, eps)` — universal; fires on gpt-oss (op).
- `attention.sdpa` — `entry(q, k, v, out, sm_scale, causal)` — scaled-dot-product
  attention (block; the op-correctness demo of the wider boundary).
- `attention.decode` — `entry(q, k, v, seq_lens, sm_scale, out)` — paged-decode
  attention; the seam extracts the running model's paged KV and routes decode through
  it (block; eager-only gather MVP — a paged-direct, CUDA-graph-safe contract is next).
- `attention.msa_block_score` — the MiniMax sparse-attention block-score stage
  (block; `matched_ratio` vs high-precision ground truth — see the M3 arena work).
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
- `collective.ar_residual_rmsnorm` — `entry(x, residual, weight, eps, out_norm,
  out_residual, group)` — the fused all-reduce + residual-add + RMSNorm epilogue behind
  sglang's `--enable-flashinfer-allreduce-fusion` (the **first slot a submitted kernel
  crowned through**). Verified distributed vs the fp32 sum+add+norm.
- `collective.moe_finalize_ar_rmsnorm` — `entry(gemm_out, row_map, scales, residual,
  weight, eps, out_norm, out_residual, group)` — the DEEP fused epilogue: MoE finalize +
  all-reduce + residual + RMSNorm in one kernel. Requires the bundle to declare a
  **`dep_patches`** unified diff against the pinned flashinfer csrc (policy-allowlisted;
  applied by a reviewed patcher to an **overlay copy**, never the install) that exports
  pre-finalize pointers; validator-owned export/consume seams (`optima/moe_export.py`)
  hand them to the kernel at the deferred fusion call, and a **last-layer veto** keeps
  the finalize in-op for any layer whose deferred call has no consumer.

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
| Slots | 10: silu/rmsnorm, attention ×3, MoE ×2, all-reduce, AR+norm epilogues ×2 (deep via dep_patches) | + MLA, FP8/FP4 GEMM, graph-safe paged attention |
| Throughput gain | **two crowned bundles on M3-NVFP4/4×B300: 1.044×/1.049× (shallow) and 1.074×/1.071× (deep), each double-proven** | keep beating the pinned baseline as it advances |
| Model | gpt-oss-120b (1×H100); MiniMax-M3-NVFP4 (4×B300, TP4) | DSV4-scale (multi-GPU, TP/PD/EP) |
| Quality gate | in-engine audit (nondet arenas) / calibrated KL (det arenas) + coverage/argmax/per-slot-threshold + GSM8K/MMLU | full-vocab KL at a reference seam + large-n (100–200) benchmarks |
| Scoring noise | noise-derived margin + bookended A/B + no-decision (no clock-lock needed) | + interleaved per-iter A/B + locked clocks where available |
| Isolation | scan (hardened) + **out-of-process** verify | namespaces + no-egress + per-eval ctx + watchdog (needs Linux/root) |
| Champion | per-round, pin-staleness flagged | head-to-head re-eval vs a content-addressed bundle store |
| Chain | **native timelock commit-reveal + hash-verified fetch + the validator loop, run live on testnet** (`optima chain-*`, docs/TESTNET.md); weights dry-run (permit-gated) | own subnet, staked permits, real set_weights cadence, hosted bundle store |
| State | JSON | a real DB, single-writer weights |

## Adding a slot

1. Define a `SlotSpec` in `optima/slots.py` (`make_inputs`, `invoke_reference`,
   `invoke_entry`, `out_shapes`, a `Correctness` mode, tolerances). It must satisfy the
   four invariants in [docs/SLOT_CONTRACT.md](SLOT_CONTRACT.md); if it can't, it
   belongs in the fenced escape hatch (`rebuild.py`), not the core.
2. If the slot needs a new chokepoint, add a seam patch under `optima/integrations/` (a
   dispatcher built with `make_*_dispatcher`) and a **single `SeamAdapter` entry in
   `optima/seams.py`** — the bootstrap watch-list, `seam.activate()`, and the `optima compat`
   canary all derive from that one table (no parallel list to edit).
3. Miners target the new slot by name in their manifest. (A `collective` slot is verified
   with `optima.verify_collective`, not `verify_entry` — see the contract doc. A
   block/collective kernel declares `graph_safe` in metadata to be scored under CUDA graphs.)
