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

> Writing or testing a competing kernel (the miner side)? Start with
> `docs/MINER_GUIDE.md` — the plain-language on-ramp (slots, scoring gates, the
> bundle format, local→GPU testing, how to find a real win). The docs below are the
> deeper validator/agent references.

0. `WORKLOG.md` (**local & gitignored — not in a fresh clone; on the dev machine
   only**). The candid working log: full experiment history, live GPU-pod access,
   the prioritized roadmap, and "how to resume". If it exists, read it first — it's
   the fastest way to reconstruct where we actually are.
1. `docs/HOW_OPTIMA_WORKS.md` — the full explainer: validator function, what
   miners submit, the pipeline, how a kernel gets into the spawned model process,
   and the complete threat model.
2. `docs/SLOT_CONTRACT.md` — **the narrow waist: the four invariants a slot must
   never break.** Read before touching `optima/slots.py` or adding a seam. Short.
3. `docs/STATE_OF_RECORD.md` — current state of record (results, gates, run
   recipes). If it and any other doc disagree, **the state of record wins** (it's
   kept current). `README.md` is the thin front door (quickstart + routing).
4. `docs/SUBNET_BLUEPRINT.md` — how a real subnet (Affine) is built: chain
   plumbing, services, DB, copy detection, isolation. The production roadmap.
5. `docs/DEV_ENVIRONMENT.md` — the GPU pods (lium), the `sn120` toolchain env, and
   how to push code + run evals on them.
6. `docs/SGLANG_TRACKING.md` — how we stay current with sglang (it's both our
   baseline and our runtime): a pinned version for consensus, the bump+re-baseline
   process, and the `optima compat` canary.

## Current state (keep this honest)

- **Mechanism: done & validated on real GPUs** (H100, up to gpt-oss-120b). Typed
  op-slots, the `.pth`+post-import seam, op-correctness, two-launch throughput+KL,
  a GSM8K capability gate, commit-reveal + king-of-the-hill, tamper-resistant
  timing. A slot is a single **op**, a fused **block**, *or* a cross-GPU **collective**
  (same cheat-resistant contract — validator allocates outputs, miner fills them, the
  kernel never reaches the sampler — just a wider boundary): `activation.silu_and_mul`,
  `norm.rmsnorm` (ops); `attention.sdpa`/`attention.decode` (blocks via the
  `RadixAttention.forward` seam, `OPTIMA_ATTENTION_SEAM=1`); `moe.fused_experts` (block
  via the `FusedMoE.forward` seam, `OPTIMA_MOE_SEAM=1`); `collective.all_reduce`
  (the TP comms waist, via the `GroupCoordinator.all_reduce` seam,
  `OPTIMA_COLLECTIVE_SEAM=1`); and **`moe.fused_experts_reduce`** — the experts block that
  **owns its trailing all-reduce** (the compute-comm OVERLAP lever, ~75% of decode), so the
  kernel fuses experts + reduce and the validator does NOT replay a stock reduce. Plus
  `collective.ar_residual_rmsnorm` (the fused AR+residual+RMSNorm epilogue waist behind
  sglang's `--enable-flashinfer-allreduce-fusion`, `OPTIMA_ARFUSION_SEAM=1`) and
  **`collective.moe_finalize_ar_rmsnorm`** — the DEEP fused epilogue (MoE finalize + AR +
  add + norm in ONE kernel): a bundle-declared **`dep_patches`** unified diff against the
  pinned flashinfer csrc (policy-allowlisted, applied by a reviewed patcher to an OVERLAY
  copy — the install is never mutated) makes the launcher skip its in-op finalize and
  export pre-finalize pointers; validator-owned export/consume seams
  (`optima/moe_export.py`) hand them to the kernel at the deferred fusion call. All
  collectives are verified distributed by `optima.verify_collective`. ALL seam adapters
  live in ONE table, `optima/seams.py` (the bootstrap watch-list, `seam.activate`, and the
  `compat` canary all derive from it — no parallel list).
- **Graphs-ON is the only regime that counts.** Scoring runs CUDA graphs ON (graphs-off
  cripples the baseline ~4.5–6.5×). Op seams capture directly; a block/collective kernel must
  declare `graph_safe: true` in metadata to run under capture, else it falls back in-graph.
  Beating sglang/vLLM/TensorRT graphs-on is the whole point.
- **Fidelity has two modes** (docs/FIDELITY.md — read it before touching the quality gate):
  `--fidelity-mode kl` (legacy rollout-KL, valid ONLY on arenas where a stock-vs-stock
  control measures ~0) and `--fidelity-mode audit` — the **in-engine audit**
  (`optima/audit.py`): an extra untimed EAGER candidate launch randomly samples dispatcher
  calls, re-runs the captured stock baseline on pre-call clones, and compares under the
  slot's own verify tolerances (receipted; zero violations + minimum coverage required;
  KL becomes advisory). Built 2026-07-07 after measuring that on the M3 arena two identical
  launches are NOT logit-identical (bit-stock candidates scored mean_kl 0.81–0.96;
  deterministic mode refuses fa4) — rollout-KL there punishes ANY timing change, i.e.
  exactly what miners are paid for. Pod-validated: honest kernel 2,996 audited calls /
  0 violations = PASS while advisory KL read 0.89; sabotage kernel 3,120/3,120 = FAIL.
  Known residuals (see the doc's adversarial matrix): in-process tampering (the standing
  isolation gap), timed-workload fingerprinting, attention slot not yet audited.
- **Scoring is noise-robust without clock-locking** (`optima/eval/scoring.py`): the candidate is
  bracketed by a baseline before AND after (B,C,B'), paired against the mean, with the bar
  derived from measured baseline noise (`1 + max(margin, k·noise)`) and a NO-DECISION verdict
  when the bracketing baselines disagree. `ignore_eos` on → identical token budgets AND a
  driver-known throughput numerator (not a scheduler-reported count). Fidelity gating beyond
  mean-KL (kl mode): a coverage/tail-mass guard (catches a flattened head-matching distribution
  top-k KL misses), argmax-rate (sparse flips), early-stop dropped-position accounting, and
  **per-slot KL thresholds** (`SlotSpec.kl_threshold`; attention 3e-2 vs the 5e-3 default).
  Per-op verify **jitters count dims** per run (anti shape-branching; collective verify too;
  plus synced-temporal AND unsynced-burst sequence gates for stateful collectives). Anti-copy
  (`optima/copy_fingerprint.py`): cumulative-across-rounds detection on exact hash OR a
  reformat-invariant fingerprint — computed **per slot over each op's transitive bundle-local
  import closure**, with a per-FILE containment compare so neither padding the bundle with an
  extra op nor relocating a stolen body into an imported module evades auto-demote — plus a
  structural skeleton fingerprint (advisory, flags rename/constant-tweak).
  `optima settle --per-slot` = a champion per slot, emission split (pays specialists); a champion
  on a different `PINNED_SGLANG` is flagged stale (re-baseline). `optima verify` loads + runs the
  kernel **out-of-process** so the CLI never imports miner code (full netns isolation is still TODO).
- **FIRST REAL WIN (2026-07-07): a submitted kernel beat sglang through optima's own
  scorer at equal fidelity.** The `miner_m3_fused_epilogue` bundle (the July-2 campaign's
  v6 Lamport fused AR+residual+RMSNorm, `collective.ar_residual_rmsnorm`, graph_safe) on
  the MiniMax-M3-NVFP4/4×B300 arena: **speedup 1.044× vs the noise-derived bar 1.038 →
  PASS (noise-confident), SCORE 1.044**, with the full gate chain green — distributed
  verify, GSM8K paired no-regression (93.8%/92.2%), in-engine audit 12,456 calls /
  0 violations (graphs-on, NP=256/MNT=256, heat-soaked bookends). **Reproduced on an
  independent prompt seed: 1.049× vs bar 1.005, audit 12,648 calls / 0 violations** —
  the win is workload-robust, not a lucky draw. **Same day, the DEEP bundle
  (`miner_m3_fused_epilogue_deep`, both slots incl. `collective.moe_finalize_ar_rmsnorm`
  via dep_patches) crowned at SCORE 1.074 (vs bar 1.010; audit 12,480/0) and
  reproduced at 1.071 (vs bar 1.037; audit 12,636/0) — the campaign's +2.7% deep-fusion
  claim converted through the referee, stacked on the shallow win (1.049 × 1.025 ≈
  1.074).** Shipping that required the LAST-LAYER VETO in `optima/moe_export.py`
  (upstream minimax_m3 never wires `is_last_layer`, so sglang lets the final layer
  defer its AR — fatal with skip-finalize armed; the seam now reads the layer count
  from the model config and refuses the last-of-forward arm). Full record:
  `experiments/minimax_m3/frontier_2026-07-07/02_FE_BUNDLE_INGEST_LEDGER.md`
  (**local-only, gitignored** — dev machine, like WORKLOG.md; the numbers here are
  the committed record). Every OTHER example bundle remains a correctness demo
  (faithful but slower).
- **Chain integration: DONE and live-validated on testnet (2026-07-08).**
  `optima/chain/` = the full loop: miners commit `{"v":1,"h":<content_hash>,"u":<url>}`
  via the chain's NATIVE timelock commit-reveal (`set_reveal_commitment`, ≤1024 B —
  URL unreadable until the reveal block, which is the anti-copy priority timestamp);
  `optima chain-validate` reads reveals in chain order, fetches (hostile-archive-safe,
  size-capped), **re-hashes the extracted tree against the committed hash**, fingerprints
  + demotes copies, evaluates out-of-process (pluggable `--eval-cmd` → the real GPU gate
  chain), settles per slot, and pushes weights (SDK auto-routes through drand CRv4 when
  the subnet enables commit-reveal weights). Emission policy lives in ONE seam,
  `Ledger.current_weights` — swap it there, nowhere else. Proven on netuid 307: the deep
  FE bundle was chain-committed by a miner hotkey and the loop crowned it at **SCORE
  1.0717 (1.072× vs bar 1.026; audit 12,824/0)** — the third independent deep repro.
  Runbook: `docs/TESTNET.md`; canary: `optima chain-compat` (SDK 10.3.2; note
  `bittensor-drand<2.0.0`). Weights stayed dry-run on 307 (zero-stake = no permit, and
  that subnet can't accept stake) — a real push needs our own subnet.
- **Open — the next goals:** isolation for untrusted miners; the tiered eval scheduler
  (screen cheap, record rarely, amortized B,C1..Ck,B' bookends; resident-engine screener
  design in the 07-07 ledger); a real DB behind the Ledger; mainnet economics (own
  subnet, staked validator permits, hosted bundle store); more slots (MLA/weight-absorbed
  attention, FP8/FP4 GEMM, graph-safe paged attention); upstream: report minimax_m3's
  unwired `is_last_layer` (stock last-layer AR realization is unclear — probe before
  filing). NOTE: emissions will NOT stay winner-take-all (relative-improvement +
  time-decay direction) — don't design around argmax-only scoring.

## How to run

```bash
# CPU dry-run (no GPU): manifest -> scan -> load -> op-correctness, + tests
pip install -e ".[cpu,dev]" && pytest tests/   # [cpu] pulls torch (the core leaves it unpinned)
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
  to the GPU arch (9.0 = H100, 10.0 = B200).
- gpt-oss-120b fits a single H100 in the validated Hopper path. Multi-GPU (TP) runs on
  other boxes select the MoE backend via `--moe-runner-backend`; see
  `docs/DEV_ENVIRONMENT.md`.
- Adding a slot = a `SlotSpec` in `optima/slots.py` (set `kind="op"`/`"block"`/`"collective"`;
  use `Correctness("matched_ratio", ...)` for kernels that legitimately differ from
  the reference — attention / fp8 / MLA weight-absorption — gated against
  high-precision ground truth, never the stock kernel). If it needs a NEW sglang
  chokepoint, add a seam patch in `optima/integrations/` and a **single entry in
  `optima/seams.py`** (the one table the bootstrap watch-list, `seam.activate`, and the
  `compat` canary all derive from — do NOT re-add a parallel list to `bootstrap`/`compat`).
  It **must** satisfy the four invariants in `docs/SLOT_CONTRACT.md` (the waist); a
  block/collective kernel also declares `graph_safe` to be scored under CUDA graphs. If it
  can't satisfy the invariants, it belongs in the fenced escape hatch, not the core.
- **The seam patches a pinned, unmodified sglang at runtime** — we never fork,
  commit, or reconfigure sglang, so the gitignored `sglang/` clone is a dev
  reference only. Runtime injection is how a miner changes a *backend* (e.g.
  attention via the `RadixAttention.forward` chokepoint) while every validator runs
  the same pinned package (consensus via `PINNED_SGLANG` + `optima compat`). This is
  strictly better than the `--attention-backend` flag: it accepts *novel* kernels and
  needs no per-submission reconfigure. **Hard line: a slot must stay upstream of the
  logprobs/sampler**, or the output-substitution attack (run gibberish, fetch the
  real answer from an API) reappears — that line is what keeps op/block slots safe.
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
per-cwd supplement. Keep docs/STATE_OF_RECORD.md + this file current when state changes; keep the
blow-by-blow in `WORKLOG.md`, not in the committed docs.
