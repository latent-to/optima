# Optima — inference-throughput competition harness on SGLang

A validator harness for a Bittensor-style subnet where miners submit **kernels**
(Triton / CuteDSL) that get swapped into a **fixed** model at typed op-slots, and
are scored on **throughput** gated by **output fidelity** — measured two ways:
per-token **KL** against a reference run, and **task accuracy** on real benchmarks.

> **New here? Read [docs/HOW_OPTIMA_WORKS.md](docs/HOW_OPTIMA_WORKS.md)** — the
> full end-to-end explainer: what the validator does, what miners submit, the
> exact pipeline, how a kernel gets into the model, and the complete threat model
> (including the "fake the output via an API call" attack and why the op-slot
> design defeats it). *(Some of its scoring/slot sections predate the benchmark
> gate and the second slot — this README is the current state of record.)*
>
> **Going to production?** [docs/SUBNET_BLUEPRINT.md](docs/SUBNET_BLUEPRINT.md)
> distills how a real Bittensor subnet (Affine) is built — chain plumbing, the
> service decomposition, DB-backed state, copy detection, and the isolation
> security pattern — and maps each onto Optima's production architecture.

## What is and isn't done

**Done & validated on a real GPU (H100, sglang 0.5.9):** the whole *mechanism* —
typed op-slots, the seam that swaps an untrusted kernel into a spawned model
process, op-correctness, two-launch throughput measurement, the KL gate, a
real-task capability gate (GSM8K + MMLU), commit-reveal + king-of-the-hill
scoring, and
tamper-resistant timing. Two slots exist (`activation.silu_and_mul`,
`norm.rmsnorm`), proven on models from Qwen2.5-0.5B up to **gpt-oss-120b**.

**Explicitly NOT done:** we have **not improved on base sglang throughput at all**
— the example kernels are toy demos and are *slower* than sglang's tuned kernels.
The point so far has been the *referee*, not the optimization. Also open:
isolation for untrusted miners, chain integration, a real DB, more (bigger) slots
like attention/MoE, and **eval calibration** (see "Calibration findings").

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

**gpt-oss-120b (MXFP4, single H100), GSM8K + KL:**

| Bundle | GSM8K base→cand | KL | gate |
|---|---|---|---|
| `miner_rmsnorm_broken` (skips norm) | 75.0% → **0.0%** | huge | **FAIL** (correct) |
| `miner_rmsnorm_triton` (faithful) | 75.0% → 58.3%* | 9.2e-3* | FAIL* |

\* We measured the control — stock-vs-stock KL (the nondeterminism floor) is
**3.9e-4** (1/2041 token flips). The faithful kernel's **9.2e-3 / 24-flips is ~24×
the floor**, so it's *real* drift, not sampling noise: this toy kernel isn't
bit-faithful to sglang's RMSNorm, and the **end-to-end gate correctly caught what
op-correctness (bf16 tolerance) passed**. Wins here: the RMSNorm seam **fires on a
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
3. **For a quantized model there's no fp32 ground truth** (gpt-oss is MXFP4), so the
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

## Repo layout

```
optima/
  slots.py                  # the op-slot ABI: SlotSpec catalog (silu, rmsnorm)
  manifest.py               # bundle manifest parse + path-safety
  sandbox.py                # static policy scan + isolated load (defense-in-depth)
  registry.py               # kernel registry + eligibility + active toggle
  dispatch.py               # per-slot dispatchers (the one place a kernel is called)
  verify.py                 # op-level correctness vs reference
  seam.py / bootstrap.py    # install the seam in every venv interpreter via a .pth
  integrations/
    sglang_silu.py          # patch SiluAndMul
    sglang_norm.py          # patch RMSNorm
    sglang_plugin.py        # entry point for sglang builds that have a plugin fw
  eval/
    throughput_kl.py        # two-launch throughput + KL (generic corpus; calibration smoke)
    capability.py           # two-launch throughput + KL + benchmark accuracy (the real-task scoring path)
    benchmarks.py           # Benchmark protocol + GSM8K & MMLU (HF), answer extraction
    kl.py / prompts.py / _launch.py
  bundle_hash.py            # deterministic bundle identity
  commit_reveal.py          # commit-reveal + king-of-the-hill ledger
  cli.py                    # slots|scan|verify|evaluate|bench|hash|commit|reveal|ledger|settle
examples/
  miner_silu_triton|torch|broken/      # silu slot (faithful / CPU dry-run / adversarial)
  miner_rmsnorm_triton|broken/         # rmsnorm slot (faithful / adversarial)
tests/                                  # 21 tests (scanner, manifest, KL, commit-reveal, verify)
```

## How a kernel gets into the model (the seam)

`sglang.Engine` forces `mp.set_start_method("spawn")` and runs the model in a
separate scheduler process, so a class-patch in the parent never reaches it. We
install the seam in **every** venv interpreter via a `.pth` file
(`import optima.bootstrap`) + a post-import hook that patches the target op
(`SiluAndMul` / `RMSNorm`) the moment its module loads — including in the spawned
scheduler. Released sglang 0.5.9 has no plugin framework, so this `.pth` path is
primary; the entry-point plugin is kept for builds that do.

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
export TORCH_CUDA_ARCH_LIST=9.0                        # 9.0=H100, 12.0=RTX Blackwell

# op-correctness on device
.venv/bin/python -m optima.cli verify examples/miner_rmsnorm_triton --device cuda

# cheap KL smoke on a generic corpus (calibration / quick check)
.venv/bin/python -m optima.cli evaluate examples/miner_silu_triton \
    --model Qwen/Qwen2.5-0.5B-Instruct --no-deterministic            # KL gate

# the real scoring path: throughput on real benchmark prompts (GSM8K + MMLU, long
# CoT generation), gated on KL *and* task accuracy from the same run
.venv/bin/python -m optima.cli bench examples/miner_silu_triton \
    --model Qwen/Qwen2.5-1.5B-Instruct --benchmarks gsm8k,mmlu --samples 64

# gpt-oss-120b TP=4 on Blackwell (sm_120a): plain-triton MoE, custom-allreduce off
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
`optima/slots.py` (+ a seam patch). Two slots today:

- `activation.silu_and_mul` — `entry(x, out)` — Qwen/Llama-class MLP.
- `norm.rmsnorm` — `entry(x, weight, out, eps)` — universal; fires on gpt-oss.

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
- **copy detection**: earliest commit of a content hash is original; later ones
  earn 0. *(Next: a behavioral/functional fingerprint to catch reformatted
  near-copies — exact hashes miss those; see SUBNET_BLUEPRINT.)*
- **king of the hill**: a champion holds the emission; a challenger takes the title
  only by beating it by a margin. A copy ties → earns nothing.

Robust scoring: median-of-K timed passes with spread, per-epoch seeded prompts
(anti-overfit), and a speedup margin gate.

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

Worst case for a fully-compromised kernel is one wrong score for itself;
cross-validator consensus catches a rogue validator.

## What's MVP vs. production

| Concern | Now | Production |
|---|---|---|
| Slots | silu, rmsnorm (toy kernels) | + attention/MLA, MoE, GEMM (real wins) |
| Throughput gain | **none yet** | the actual point — real faster kernels |
| Model | up to gpt-oss-120b (1 GPU) | DSV4-scale (multi-GPU, TP/PD/EP) |
| Quality gate | KL + GSM8K/MMLU on real prompts, **uncalibrated** | noise-floor KL + large-n benchmarks + det mode |
| Isolation | scan + in-proc load | namespaces + no-egress + per-eval ctx + watchdog |
| Chain | local JSON ledger | on-chain commit-reveal + set_weights |
| State | JSON | a real DB, single-writer weights |

## Adding a slot

1. Define a `SlotSpec` in `optima/slots.py` (`make_inputs`, `invoke_reference`,
   `invoke_entry`, `out_shape`, tolerances).
2. Add a seam patch under `optima/integrations/` that routes the real sglang op
   through a dispatcher built with `make_*_dispatcher`, and install it from
   `seam.activate()`; add the op's module to `bootstrap._TARGETS`.
3. Miners target the new slot by name in their manifest.
