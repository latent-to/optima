# Optima — agent onboarding

> This file is the quick orientation for any coding agent working in this repo.
> Read it first; the deep docs are under `docs/`.

## What this is

**Optima** is a Bittensor-style subnet that incentivizes **inference-throughput
optimization**. The normal lane lets miners submit GPU **kernels** (Triton /
CuteDSL) for registered targets in a *fixed* model; a separate fenced review lane
handles cross-cutting discovery proposals without silently turning them into a
registered title. A validator swaps each kernel into the model it controls, runs
it, and scores it on **throughput** gated by **output fidelity**. The intended
gate combines an in-engine audit with pristine-reference distribution and task checks;
the current production-wiring caveat is called out below. The endgame is a continuously-improving SOTA
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

- **The kernel mechanism is validated on real GPUs** (H100, up to gpt-oss-120b).
  The hardened authority adds typed targets, correctness/graph proof, graph-on/audit-free
  charged B/C/B′ throughput, a mandatory separate eager/untimed audit role with a typed
  host-regraded witness, pristine-T quality authority, independent reproduction,
  finalized commit-reveal intake, and transactional target settlement. A slot is a single **op**,
  a fused **block**, *or* a cross-GPU **collective**
  (same cheat-resistant contract — validator allocates outputs, miner fills them, the
  kernel never reaches the sampler — just a wider boundary): `activation.silu_and_mul`,
  `norm.rmsnorm` (ops); `attention.sdpa`/`attention.decode` (blocks via the
  `RadixAttention.forward` seam, `OPTIMA_ATTENTION_SEAM=1`);
  **`attention.msa_prefill_block_score`** (block via the MSA arena's
  `flash_prefill_with_topk_index` seam, `OPTIMA_MSA_PREFILL_SEAM=1`: the miner fills the
  prefill indexer's causal block-score SHEET, the validator keeps the stock top-k
  selection + attend — gated on per-row `topk_overlap`, with a long-chunk verify shape
  that makes causality violations detectable through the set metric; the seam row's
  `requires` makes the compat canary SKIP it on pins without `minimax_sparse_ops`; its
  regime is prefill-heavy serving, so score it on a prefill-heavy workload — a
  short-prompt pure-decode regime cannot see a prefill win);
  `moe.fused_experts` (block
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
- **Fidelity implementation status:** the 2026-07-07 evaluator had two modes
  (docs/FIDELITY.md — read it before touching the quality gate):
  `kl` (legacy rollout-KL, valid ONLY on arenas where a stock-vs-stock control
  measures ~0) and `audit` — the **in-engine audit**
  (`optima/audit.py`): an extra untimed EAGER candidate launch randomly samples dispatcher
  calls, re-runs the captured stock baseline on pre-call clones, and compares under the
  slot's own verify tolerances (receipted; zero violations + minimum coverage required;
  KL becomes advisory). Built 2026-07-07 after measuring that on the M3 arena two identical
  launches are NOT logit-identical (bit-stock candidates scored mean_kl 0.81–0.96;
  deterministic mode refuses fa4) — rollout-KL there punishes ANY timing change, i.e.
  exactly what miners are paid for. Pod-validated: honest kernel 2,996 audited calls /
  0 violations = PASS while advisory KL read 0.89; sabotage kernel 3,120/3,120 = FAIL.
  Known residuals (see the doc's adversarial matrix): in-process tampering, audit-role
  fingerprinting, and timed-workload fingerprinting. Stateful attention slots without a
  save-free stock baseline are unauditable and therefore fail closed rather than crown.
  Crownable candidate execution is fenced in validator-
  owned OCI sessions with no network egress; the trusted controller never loads miner
  Python or native extensions. The separate eager/untimed role, policy-bound transport,
  exact slot×TP-rank coverage, typed witness, and trusted-host regrade are implemented and
  CPU/mock-covered. Charged B/C/B′ roles remain graph-on and reject audit state/evidence.
  These new causal bytes are **not GPU-qualified** until the exact production MiniMax-M3
  canary passes. The July 7 audit receipts validate the underlying mechanism, not the new
  transport/report path.
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
  Production settlement uses explicit singleton or atomic competition targets and
  transactional stack state. One passing qualification is retained as
  `reproduction_pending`; settlement requires a second independent passing authority and
  uses the lower of the two measured speedups. Candidate execution runs out of process in
  a no-egress OCI worker, so the trusted controller never imports miner code.
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
  `optima chain-validate` reads finalized reveals in chain order, fetches
  hostile archives into private storage, **re-hashes the extracted tree against the
  committed hash**, fingerprints copies, and publishes an immutable worker tree. A
  validator-injected `ArenaServiceRegistry` applies the registered static/build/ABI/graph/
  abbreviated-serving screen before expensive qualification. SQLite records intake,
  receipts, reproduction state, stack transitions, and settlement; `optima set-weights`
  reconciles the resulting global reward projection separately. Proven on netuid 307:
  the deep
  FE bundle was chain-committed by a miner hotkey and the loop crowned it at **SCORE
  1.0717 (1.072× vs bar 1.026; audit 12,824/0)** — the third independent deep repro.
  Runbook: `docs/TESTNET.md`; canary: `optima chain-compat` (SDK 10.3.2; note
  `bittensor-drand<2.0.0`). The first pass kept weights dry-run; a later permitted
  netuid-307 pass validated real commit-reveal weight publication and restart safety.
- **Incentive composition selected; V2 remains inactive.** D-012's historical
  deterministic
  224,000-run sweep, byte-identically replayed locally and on the RTX pod, selected
  registered-CROWN finite debt in multiplicative 1%-log units with a rational
  10%-capped family-time bonus (`tau=648,000` blocks), `k=1`, 90-day expiry,
  10% reserve, digest-ordered near-equal integer family shares, and a family-clock
  reset on every accepted CROWN. Its selected cell
  `15623f7679f5c1099ab48ecc88b1fe6aac926f58b309d07b3a788180848477a4`
  paid 98.7203% of principal over the simulation horizon with maximum measured
  split/withhold/sybil distortion 3.0287%. The live pod
  `chain-incentive-shadow` receipt tested this registered-CROWN class only. D-015
  supersedes its target-family claim division with schema-2 model-campaign sizing.
  **The implemented launch path accepts exactly one immutable MiniMax-M3 campaign at
  100% sizing.** The historical two-campaign 50/50 cells remain arithmetic research;
  model rotation, a second campaign, and any successor activation are unsupported live.
  Families remain independent frontiers and clocks, but 1/10/100-family catalogs
  caused zero principal dilution. All 14 preregistered D-015 screens passed. At
  `k=1`, the historical normal weekly load was one full-sized 4.4%/5% claim for one
  campaign, or one half-sized claim in each of two campaigns (one full share aggregate),
  rotated across families. It paid 100%, expired zero, drained to zero, and had
  five-day maximum latency under empty and saturated discovery; sustained
  simultaneous per-family wins were not the normal-tape assumption.
  five-day cadence was marginal, four-day cadence overloaded, and `k=1.25`
  marginal. At `k=1.5` the worst rows overloaded while other rows remained
  marginal; `k=2` was plainly overloaded. Report semantic digest:
  `7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590`.
  A tracked one-campaign supplement then exercised 64 launch/stress cells over
  1/2/5/10 independently winning M3 families, 7/14/30/90-day cadence, and empty or
  saturated discovery. Its semantic digest is
  `505fed4d40a6acc6bc92d6330170e8e2260a52e5f3099c22a6c0eb4b2308c672`.
  The raw D-015 sweep remains a local-only experiment record. Current local
  validation is 2,191 passed/19 skipped repository-wide; the tracked 64-cell
  D-015 supplement replays at the digest above. D-015 has no pod receipt.
  The old testnet shadow does not test D-015 bytes.
  D-013 then selected a separately reviewed discovery bounty capped at 50,000 ppm
  per epoch and at one such epoch of principal per award, with 648,000-block
  expiry and no campaign share, family clock, time bonus, renewal, or permanent title.
  The selected pure policy intends one reviewed win to choose promotion into a
  registered target followed by fresh requalification/CROWN, or the finite bounty,
  never both. Durable schema-5 does not yet implement both branches: it retains
  qualified discovery wins as `review_pending` and can issue bounded `bounty_only`
  debt, but deliberately rejects `registered_promotion` until typed
  `DiscoveryWinRecord`/`DiscoveryPromotion` transport, target registration, fresh
  requalification/CROWN linkage, and cross-lane work identity exist. “Never both”
  is therefore policy intent, not end-to-end same-work enforcement today. A bounty's
  648,000-block life starts at the retained qualified-win block; review delay consumes
  that window, and review at or after expiry cannot mint. The durable terminal
  review-expiry API records `review_expired` plus `discovery_review_expired`, but its
  production scheduling remains part of the unfinished control plane. The selected
  D-013 cell is `8561028c943738da2fe622e5f5c9fd43ebec16fdd59feab3561de25fbfa450d9`;
  report digest
  `6bdfce26e4e6090e0dcc8814a636c665f28d1ff20945a09d43a9a90dc94151fc`.
  Its 3,240-row synthetic sweep replayed byte-identically locally and on the pod;
  non-departed principal paid 273,000,000/273,000,000 units, departed debt
  cancelled 9,000,000 units, and saturated CROWN-capacity dilution was 55,555 ppm.
  D-014 then tested review delay in 288 synthetic rows and replayed byte-identically
  on arm64/Python 3.11 and x86_64/Python 3.12. Its preregistered 0/1/7-day review
  SLA passed all 108 rows: discovery paid 100%, expiry/unissued was zero, maximum
  instantaneous CROWN-capacity dilution was 55,555 ppm, and CROWN paid-fraction
  regression versus zero delay was zero. The 90/120-day cases issued no stale
  debt; 30/60/89 days were diagnostic only. Report digest:
  `f0939d67241dffa49aac95c035c43dd7ea14b51eb2671fe106cb09347511b7ef`.
  This is deterministic accounting sensitivity evidence, not external review,
  publication, activation, durable-state completion, or GPU-performance evidence.
  Pure arithmetic, schema-5 review-pending/bounty-only durable state, and the signer-free
  `chain-incentive-composition-shadow` surface are implemented. A live read-only
  final hardened-source run on testnet netuid 307 at finalized block 7,586,146 (metagraph size 6)
  projected explicitly synthetic states to 850,000 ppm registered-CROWN payout,
  50,000 ppm reviewed-discovery payout, and 100,000 ppm reserve, totaling
  1,000,000 ppm; it wrote `submitted=false`. Receipt semantic digest:
  `3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`;
  receipt-file SHA-256:
  `ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`.
  It constructed no wallet and supplied no review, settlement, publication,
  D-015 policy, or activation authority. A separate multi-pass restart audit now binds exact paired
  qualification/evidence/CROWN speed to principal, derives family clocks and balance
  transitions all-and-only from their journals, validates every retained discovery
  lifecycle before filtering status, and reopens terminal histories before upgrades.
  The reproduced substitution/cardinality cases are regressions; pre-D-015 local
  tests were 2,135 passed/19 skipped and historical pod conformance was 111 passed.
  D-015 itself has no pod receipt. A fresh final-source
  `chain-validate --intake-only` pass at finalized block 7,586,142 saw/reserved 19,
  rejected five malformed payloads, and did no qualification/settlement; its restart
  at block 7,586,144 had all counters zero. Schema-5 migration starts empty and creates
  no retroactive debt; activation rejects any retained legacy discovery row because V1
  has no journal that proves mutable terminal status, and active composition
  disables legacy discovery auto-award. The wallet-free `chain-activate-incentives`
  command atomically binds the exact core manifest, composition manifest, independent
  approval digest, finalized chain block/hash, and equal intake cursor. It derives the
  campaign from exactly one retained arena's complete catalog/family roster, checks the
  reserve in the approved finalized membership, reproduces the arena/stack/catalog/
  membership digests pinned in that approval, and requires audit-control-manifest,
  final-canary-receipt, and residual-risk-acceptance digests. It retains the complete
  approval in the activation row/event,
  and raises the durable schema floor 5→6 in the same transaction. It permits
  exactly one immutable MiniMax-M3 campaign. `set-debt-weights` implements restart-safe
  gapless projection, signing, finalized readback, confirmation, and only-then debt debit.
  Delayed boundaries retain their nominal order and catch up no faster than one policy
  cadence after the prior confirmation. Neither command has a live activation/publication
  receipt.
  A landed finalized `invalidate_finite_debt_family` API can cancel one registered
  family's open debt and reset its next-CROWN clock, but the runtime-invalidity
  decision/digest is still external authority. Meaningful V2 emissions still require:
  exact MiniMax-M3 family and reserve manifests plus a fresh shadow; retained
  membership-departure history rather than only a current snapshot; independently
  graded review and runtime-invalidation authority; the promotion transport/linkage
  above; the production audit GPU canary; and actual activation/mainnet
  operations. Later model rotation, a second campaign, and successor activation are
  explicitly unsupported rather than launch blockers hidden behind configuration.
- **Release path:** evaluation and serving are separate products. Approved integration
  reviews authorize contributions in an `EngineReleaseManifest`; model provisioning
  receipts seal every model file; signed, chain-independent releases carry deterministic
  source/wheel artifacts, SBOM/provenance, a pinned serving specification, and an OCI
  build context. Chain or wallet code is not included in the serving wheel.
- **Open — the next goals:** execute and canary the implemented one-campaign V2 cutover and
  publication path, then mainnet operations (exact MiniMax-M3 family/reserve manifests plus
  a fresh shadow, independently graded discovery review and runtime invalidation,
  membership-departure history, discovery promotion transport/linkage, the production
  audit GPU canary, owned subnet, validator permits, hosted bundle storage);
  more slots
  (MLA/weight-absorbed attention, FP8/FP4
  GEMM, graph-safe paged attention); and B300-only qualification of SM103/CuTe,
  NVLink/custom-collective behavior, topology-specific calibration, and the existing
  MiniMax-M3 campaign kernels. Emissions will not stay winner-take-all; the selected
  direction is finite log-relative CROWN debt plus bounded reviewed discovery debt,
  not perpetual argmax credit.

## How to run

```bash
# CPU dry-run (no GPU): manifest -> scan -> load -> op-correctness, + tests
pip install -e ".[cpu,dev]" && pytest tests/   # [cpu] pulls torch (the core leaves it unpinned)
python -m optima.cli verify examples/miner_silu_torch --device cpu --dtype float32

# GPU: see docs/DEV_ENVIRONMENT.md for the env setup, then
python -m optima.cli verify examples/miner_silu_triton --device cuda --dtype bfloat16
# authoritative scoring is validator-side: chain intake -> qualification
# (graph-on/audit-free B/C/B' + eager/untimed audit witness + pristine-T,
#  all in no-egress workers) -> reproduction -> settlement
# (docs/TESTNET.md; the legacy local evaluate/bench diagnostics were deleted)
```

Always run GPU work via `python -m optima.cli` (the spawn-safe `__main__` guard
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
- **The seam normally patches an exact validator-owned sglang runtime in memory**;
  miner submissions never patch the engine install. The default stable/discovery pin is
  `PINNED_SGLANG`, while the current MiniMax-M3 arena evidence names source build
  `0.0.0.dev1+g56e290315` plus reviewed validator overlays. The exact full-source/image
  consensus relationship is still an open production policy gap. Runtime injection
  is how a miner changes a *backend* (e.g.
  attention via the `RadixAttention.forward` chokepoint) while every validator runs
  the same exact arena package/image identities. `optima compat` fails when the
  installed version differs from its default `PINNED_SGLANG`; it does not by itself
  reconcile the M3 source-build policy. This is
  strictly better than the `--attention-backend` flag: it accepts *novel* kernels and
  needs no per-submission reconfigure. **Hard line: a slot must stay upstream of the
  logprobs/sampler**, or the output-substitution attack (run gibberish, fetch the
  real answer from an API) reappears — that line is what keeps op/block slots safe.
- Miner submissions are Triton/CuteDSL source, not prebuilt CUDA extensions. That
  narrows the artifact surface and keeps submissions inspectable. Import, native build,
  engine construction, and candidate execution occur only inside the no-egress OCI
  worker; the trusted controller validates bounded protocol evidence and owns teardown.
- Don't claim a kernel "drifts" without measuring the **stock-vs-stock KL noise
  floor** first (we got burned on this).
- **sglang identity is consensus-critical.** `PINNED_SGLANG` in `optima/compat.py`
  is the default stable/discovery version; arena/preflight/release facts record the M3
  source build separately, but do not yet close one unified pin policy. Do not mix results
  across those identities or describe the source build
  as validating the default pin. After any SGLang change run the strict `optima compat`
  canary against the intended pin plus the broken-bundle smoke, then re-baseline the
  champion. The still-open governance choice is whether production converges on one global
  pin or adopts a reviewed per-arena pin registry. See `docs/SGLANG_TRACKING.md`.

## Persistence note for future agents

This repo's `AGENTS.md`, the `CLAUDE.md` shim, and `docs/` are the canonical
*committed* context (they travel with the repo). The candid working log lives in
`WORKLOG.md` (gitignored, local-only — keep it off GitHub). Auto-memory is a
per-cwd supplement. Keep docs/STATE_OF_RECORD.md + this file current when state changes; keep the
blow-by-blow in `WORKLOG.md`, not in the committed docs.
