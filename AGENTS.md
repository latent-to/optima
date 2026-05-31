# Optima — agent onboarding

> This file is the quick orientation for any coding agent working in this repo.
> Read it first; the deep docs are under `docs/`.

## What this is

**Optima** is a Bittensor-style subnet that incentivizes **inference-throughput
optimization**. Miners submit GPU **kernels** (Triton / CuteDSL) for individual
ops in a *fixed* model; a validator swaps each kernel into the model it controls,
runs it, and scores it on **throughput** gated by **output fidelity** (per-token
KL + real-benchmark task accuracy). The endgame is a continuously-improving SOTA
inference stack sold as a managed service; the validator endgame is an 8×B200
fleet evaluating submissions.

This repo is the **validator harness** (the referee), plus example miner bundles.

## Read these (in order)

0. `WORKLOG.md` (**local & gitignored — not in a fresh clone; on the dev machine
   only**). The candid working log: full experiment history, live GPU-pod access,
   the prioritized roadmap, and "how to resume". If it exists, read it first — it's
   the fastest way to reconstruct where we actually are.
1. `docs/HOW_OPTIMA_WORKS.md` — the full explainer: validator function, what
   miners submit, the pipeline, how a kernel gets into the spawned model process,
   and the complete threat model.
2. `README.md` — current state of record (results, gates, run recipes). If it and
   HOW_OPTIMA_WORKS disagree, **README wins** (it's kept current).
3. `docs/SUBNET_BLUEPRINT.md` — how a real subnet (Affine) is built: chain
   plumbing, services, DB, copy detection, isolation. The production roadmap.
4. `docs/DEV_ENVIRONMENT.md` — the GPU pods (lium), the `sn120` toolchain env, and
   how to push code + run evals on them.
5. `docs/SGLANG_TRACKING.md` — how we stay current with sglang (it's both our
   baseline and our runtime): a pinned version for consensus, the bump+re-baseline
   process, and the `optima compat` canary.

## Current state (keep this honest)

- **Mechanism: done & validated on real GPUs** (H100, up to gpt-oss-120b). Typed
  op-slots, the `.pth`+post-import seam, op-correctness, two-launch throughput+KL,
  a GSM8K capability gate, commit-reveal + king-of-the-hill, tamper-resistant
  timing. Two slots: `activation.silu_and_mul`, `norm.rmsnorm`.
- **Not done: any actual throughput improvement.** The example kernels are toy
  demos and are *slower* than sglang's tuned kernels. We built the referee, not
  optimizations.
- **Open:** isolation for untrusted miners, chain integration, a real DB, bigger
  slots (attention/MLA, MoE), and **eval calibration** (KL threshold = k× the
  measured nondeterminism noise floor; run with `enable_deterministic_inference`;
  benchmark accuracy needs large n). RMSNorm was chosen as the second slot because
  it is simple and universal, not because it is expected to be a major speed win.

## How to run

```bash
# CPU dry-run (no GPU): manifest -> scan -> load -> op-correctness, + tests
pip install -e . && pytest tests/
python -m optima.cli verify examples/miner_silu_torch --device cpu --dtype float32

# GPU: see docs/DEV_ENVIRONMENT.md for the env setup, then
python -m optima.cli evaluate examples/miner_silu_triton --model Qwen/Qwen2.5-0.5B-Instruct --no-deterministic
python -m optima.cli bench     examples/miner_silu_triton --model Qwen/Qwen2.5-1.5B-Instruct --benchmarks gsm8k --samples 64
```

Always run GPU evals via `python -m optima.cli` (the spawn-safe `__main__` guard
matters — sglang uses `mp spawn`).

## Conventions / gotchas (learned the hard way)

- The seam is installed in **every** venv interpreter via a `.pth`
  (`echo 'import optima.bootstrap' > $SITE_PACKAGES/optima.pth`) because sglang
  runs the model in a spawned child. Don't expect parent-process patching to work.
- sglang's `jit_kernel` JIT-compiles CUDA at runtime → the box needs `nvcc` +
  `ninja` on PATH (`export CUDA_HOME=/usr/local/cuda`). Set `TORCH_CUDA_ARCH_LIST`
  to the GPU arch (9.0 = H100, 12.0 = RTX PRO 6000 Blackwell, 10.0 = B200).
- gpt-oss-120b fits a single H100 in the validated Hopper path. On the 4× RTX PRO
  6000 Blackwell box, stock pinned sglang still needs the plain Triton MoE fallback
  at TP=4; the current dev-pod experiment gets `flashinfer_mxfp4` working on
  `sm_120a` by padding GPT-OSS shards, using plain packed FP4 weights, interleaving
  only MXFP4 block scales, and disabling PDL for that call. See
  `docs/DEV_ENVIRONMENT.md`.
- Adding a slot = a `SlotSpec` in `optima/slots.py` + a seam patch in
  `optima/integrations/` (installed from `seam.activate()`, module added to
  `bootstrap._TARGETS`).
- Miner submissions are Triton/CuteDSL source, not raw CUDA extensions. That
  narrows the attack surface and keeps submissions inspectable, but the host
  Python launch code still runs in-process, so isolation remains required.
- Don't claim a kernel "drifts" without measuring the **stock-vs-stock KL noise
  floor** first (we got burned on this).
- **sglang is pinned** (`PINNED_SGLANG` in `optima/compat.py`) — all validators
  must run the same version (consensus). After any sglang change run `optima
  compat` (static seam canary) + the broken-bundle smoke; bump deliberately and
  re-baseline the champion. See `docs/SGLANG_TRACKING.md`.

## Persistence note for future agents

This repo's `AGENTS.md`, the `CLAUDE.md` shim, and `docs/` are the canonical
*committed* context (they travel with the repo). The candid working log lives in
`WORKLOG.md` (gitignored, local-only — keep it off GitHub). Auto-memory is a
per-cwd supplement. Keep README + this file current when state changes; keep the
blow-by-blow in `WORKLOG.md`, not in the committed docs.
