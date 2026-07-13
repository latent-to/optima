# Fidelity: why rollout-KL was retired as a gate, and how the audit stack works

> Status 2026-07-07: designed + built + unit-tested (325 local tests) **and
> pod-validated on the M3 arena (4×B300)**: the honest v6 kernel passed the
> audit (2,996 audited calls, 0 violations, worst_frac 0.9901) while the
> advisory KL still read 0.89, and a sabotage kernel (residual-add dropped)
> failed on all 3,120 audited calls (worst_frac 0.0029). Full record:
> `experiments/minimax_m3/frontier_2026-07-07/02_FE_BUNDLE_INGEST_LEDGER.md`.

## 1. What the fidelity gate is actually for

The referee pays for **speed at equal fidelity**. "Equal fidelity" has to be a
*definition*, not a vibe, and the definition must catch:

1. **Garbage** — kernels computing the wrong function (broken math);
2. **Cheats** — kernels that buy throughput by skipping/faking work, including
   ones that behave only while being watched;
3. while **never false-failing an honest reimplementation** — a kernel that
   computes the same function with different (even better) rounding must pass,
   or no one can ever win and the subnet is dead on arrival.

Arbitrary-code-execution / fooling-the-harness is NOT this gate's job: that is
source-only bundles + policy scan + out-of-process verify + seam receipts +
(roadmap) process isolation, and none of it changed in this redesign.

## 2. The measured failure of rollout-KL (2026-07-07, MiniMax-M3 arena, 4×B300)

The original primary gate compared per-token logprobs between two engine
launches (stock baseline vs candidate) along greedy rollouts. A day of
controls, each one engine pair on the real arena (eager fa4/NVFP4 unless
noted):

| control                                                    | mean_kl | verdict |
|------------------------------------------------------------|---------|---------|
| stock vs stock, identical speed (no seam loaded at all)     | exactly 0.0 | launches CAN be bit-identical |
| v6 miner kernel (graphs-on)                                 | 0.15–0.17 | FAIL |
| v6 miner kernel (eager)                                     | 0.38–0.43 | FAIL |
| **fp32-EXACT reference** at the same slot, same eligibility  | **0.78** | FAIL — *worse than the kernel it was checking* |
| **bit-stock fallback** (entry raises, dispatcher runs stock) at 0.545× speed | **0.96** | FAIL |
| bit-stock fallback, **single prompt** (no batch composition) | **0.81** | FAIL |
| sglang deterministic mode on this arena                     | refuses to launch (fa4 unsupported) | — |

Chain of eliminations: the per-call **stockcheck** (23,000 in-engine calls per
rank) showed the v6 kernel's outputs differ from stock by **single-ULP bf16
rounding on outlier channels** — no corruption, no coverage divergence, no
input mutation. Yet exact math failed harder than the kernel, bit-stock failed
harder still, and one prompt with zero batching still failed. Conclusion:

**Two launches of identical code are not logit-identical on this stack** (no
deterministic-inference support on the arena backend; per-launch autotuner
tactic selection; batch-variant kernels; then autoregressive chaos compounds
per-forward differences into full trajectory divergence). Cross-launch logit
identity is therefore not a measurable property here — a gate keyed on it
fails *every* candidate, honest or not, including stock itself. Worse, the
divergence scales with any *timing* change, so the gate structurally punishes
the one thing the subnet exists to buy: speed.

## 3. The replacement stack

```
scan  →  verify  →  IN-ENGINE AUDIT  →  benchmark no-regression  →  timed bracket
(static) (offline,  (primary fidelity   (semantic backstop,          (throughput,
          fp32 GT)   gate, in the        graphs-on, paired,           B/C/B′,
                     scored engine)      accuracy-graded)             KL advisory)
```

* **verify** (unchanged): out-of-process, fp32 ground truth, per-slot
  `Correctness` mode + `Tolerance`, jittered count dims, temporal + unsynced
  burst sequences, distributed for collectives.
* **in-engine audit** (`optima/audit.py`, new — the KL replacement): during an
  extra **untimed, eager** candidate launch, randomly sampled dispatcher calls
  are re-run through the **captured stock baseline on pre-call clones** and the
  miner's outputs must match within the slot's own verify tolerances. Per-rank
  rolling receipts; the eval driver passes only on **zero violations AND
  minimum audited-call coverage** (unproven ≠ innocent), failing closed on
  compare errors.
* **benchmark gate** (unchanged in role, recalibrated): paired accuracy
  no-regression on real tasks, graphs-on. Fixed 2026-07-07: accuracy
  generations must NOT inherit the throughput eval's `ignore_eos` (forcing a
  model past its answer graded hallucinated self-Q&A: 6.2% absolute on a ~90%
  model), and completion-format benches declare stop cues (`"\nQuestion:"`).
* **KL demoted to advisory**: still computed and printed from the quality
  launch — it is useful calibration/trend data — but it gates nothing.

No layer requires deterministic inference, a specific attention backend, or
cross-launch reproducibility.

### Why the audit is the right primary gate

It tests the invariant the referee actually needs — *"in the scored engine,
the miner kernel computes the slot's declared function"* — **directly, in
place, on the real call distribution**, instead of through a 62-layer chaotic
proxy. Properties:

* **Same numeric contract offline and online**: the audit reuses the slot's
  verify `Correctness`/`Tolerance`, so "equal fidelity" is one definition in
  one place (`optima/slots.py`), calibrated per slot.
* **Unpredictable**: the audit decision is drawn per call from a
  validator-private RNG (`OPTIMA_SLOT_AUDIT_SEED`, rank-identical so collective
  baselines keep lockstep). A kernel cannot know ex-ante whether a call is
  audited, so "behave only when watched" is not a strategy.
* **Timing-clean**: the audit runs only in a separate untimed quality launch;
  the timed launches carry zero audit overhead, so the audit can bias the
  speedup measurement in neither direction.
* **Fail-closed**: no receipts / too few audited calls / comparison errors all
  FAIL. A kernel that arranges to never be compared loses.
* **Baseline-poisoning-proof at the tensor level**: the baseline runs on
  clones taken *before* the kernel executes, so a kernel that mutates its
  inputs cannot steer the expected value.

## 4. The adversarial matrix — honest edition

| # | Attack | Caught by | Status |
|---|--------|-----------|--------|
| 1 | Wrong math (garbage kernel) | verify (fp32 GT), audit, bench | closed, 3 layers |
| 2 | Shape-conditional garbage (right on verify shapes only) | verify jitter; audit samples the REAL engine distribution | closed |
| 3 | Verify-harness detection (behave offline, cheat in engine) | audit runs inside the scored engine | closed — the reason audit exists |
| 4 | Cheat on unaudited calls only | sampling unpredictable ex-ante; per-call i.i.d. | closed (for out-of-process adversaries; see #9) |
| 5 | Mutate inputs so the baseline agrees with garbage | pre-call clones | closed |
| 6 | Temporal cheats (degrade late, cross-call state rot) | audit samples the whole run; temporal/burst verify; long bench | closed |
| 7 | Approximation inside slot tolerance (internal fp8 etc.) | ALLOWED by definition — that's the tolerance contract; bench backstops task quality | closed *by definition* (this is the freedom miners are paid to use) |
| 8 | Output substitution (gibberish + external fetch) | slots upstream of sampler; framework-mode = token-match + no-egress isolation | closed structurally (unchanged) |
| 9 | **In-process tampering**: monkeypatch `optima.audit`/the baseline fn, forge/suppress receipts, introspect the RNG | policy scan raises the bar; **not closed** — miner host code shares the rank process | **OPEN — the known isolation gap** (AGENTS.md), unchanged by this redesign but now with a sharper attack list. Closure = process/namespace isolation + receipts over an authenticated channel. Until then, scan + human review of host code is the control. |
| 10 | **Launch fingerprinting**: behave when `OPTIMA_SLOT_AUDIT` is set (or eager-vs-graphs), cheat otherwise | any env/graphs-keyed cheat also hits the BENCH launch (graphs-on, no audit env, accuracy-graded) → task quality tanks | closed **only because the bench gate is mandatory** — the stack is sound as a whole, not layer-by-layer |
| 11 | **Timed-workload fingerprinting**: detect the timed launch's workload shape (short fixed-budget repeated batches) and run fast-garbage only there — "phantom speed" | numerator is driver-computed a priori (`prompts × max_new_tokens` — can't be inflated); per-epoch prompt reseeding; champions re-run every round, so phantom speed must fool every epoch; production non-reproduction → re-baseline/dethrone | **PARTIALLY OPEN** (economic-grade, not correctness-grade: the outputs of the timed launch are no longer graded). Roadmap closure: draw timed workloads from the bench/serving distribution + randomize regimes so "timed" has no stable fingerprint. |
| 12 | Attention-slot cheats in-engine | verify (matched-ratio vs fp32 GT) + bench only — **audit not wired** (re-running the baseline re-drives the stateful KV-cache write path) | **OPEN for attention specifically**, documented in `dispatch.py`; needs a save-free baseline call. Attention slots are eager-only MVP anyway. The MSA *indexer* slots are NOT in this class: the score pass is stateless (no KV write), so `attention.msa_prefill_block_score` is audited — stock runs first on pristine inputs, and the comparison is the consumed product (selection rows, per-row set overlap at the slot's own `min_overlap`). |
| 13 | Copy a champion's kernel | copy fingerprints (containment, normalized, skeleton) | closed (unchanged) |
| 14 | Tamper with timing | driver-process timing; miner code never loads in the driver | closed (unchanged) |
| 15 | Honest kernel, different rounding (the v6 case) | PASSES audit (tolerance absorbs ULP-class drift) — the false-negative rollout-KL created | **fixed** — this was the point |

The honest bottom line: the audit closes every *output-correctness* cheat an
out-of-process adversary can mount, and turns "watched vs unwatched" into a
losing game. The two structural residuals — in-process tampering (#9) and
timed-workload fingerprinting (#11) — are not new holes introduced by the
redesign; #9 predates it explicitly, and #11 replaces rollout-KL's implicit
(and broken) coverage of the timed launch. Both have named closures on the
roadmap; neither has a working exploit that survives scan + bench + per-epoch
re-evaluation today, but they are where a motivated adversary would dig.

## 5. Knobs & usage

The audit is armed on the validator's untimed quality launch inside the
qualification bracket (the legacy local `optima evaluate --fidelity-mode ...`
diagnostic was deleted in the post-arc trim; audit statistics now flow into the
pristine reference-quality record, `optima/eval/reference_quality.py`).

* `OPTIMA_SLOT_AUDIT` / `OPTIMA_SLOT_AUDIT_SEED` — set by the launcher
  for the quality launch only; never set them on a timed launch.
* Per-slot strictness lives where it always did: `SlotSpec.correctness` +
  `SlotSpec.tolerances` in `optima/slots.py`.
* Calibration technique (reusable): **control bundles** — an fp32-exact
  reference bundle (must PASS; measures the floor) and a sabotage bundle
  (wrong function, e.g. dropped residual-add; must FAIL). Templates:
  `experiments/minimax_m3/bundle/miner_m3_fused_epilogue_{refctl,sabotage}`.
  This caught a real miscalibration on its first outing (2026-07-07): the
  fp32-exact control audited at worst_frac 0.9894 — under the raw 0.99
  matched_ratio bar — because verify compares against FP32 (one side rounds)
  while the audit compares against STOCK (both sides round). Fix:
  `_MATCHED_RATIO_AUDIT_MARGIN = 0.005` in `optima/audit.py` (audit bar =
  verify bar − margin). Honest implementations measure ≥0.989; sabotage 0.003.

## 6. Which fidelity mode for which arena

Rollout-KL remains valid where its premise holds — the premise is now a
*measured requirement*, not an assumption: **stock-vs-stock KL must be ~0 on
that arena config** (deterministic-capable backend, pinned autotune, fixed
batching). Record the control result in the arena row. Where the control
fails (any nondeterministic stack — e.g. M3/fa4/NVFP4), `audit` is the only
sound mode. The arena registry is the home for this choice, together with
bench budgets, stop cues, and template policy.
