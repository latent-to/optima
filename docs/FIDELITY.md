# Fidelity: why rollout-KL was retired as a gate, and how the audit stack works

> Status 2026-07-07: designed + built + unit-tested (325 local tests) **and
> pod-validated on the M3 arena (4×B300)**: the honest v6 kernel passed the
> audit (2,996 audited calls, 0 violations, worst_frac 0.9901) while the
> advisory KL still read 0.89, and a sabotage kernel (residual-add dropped)
> failed on all 3,120 audited calls (worst_frac 0.0029). Full record:
> `experiments/minimax_m3/frontier_2026-07-07/02_FE_BUNDLE_INGEST_LEDGER.md`.

> **Production wiring status (2026-07-19): IMPLEMENTED, NOT YET GPU-QUALIFIED.**
> Causal qualification now requires a validator-owned `SlotAuditPolicy`, launches a
> separate eager/untimed candidate role, transports bounded raw receipts bound to the
> session/request/nonce/launch/policy, requires exact slot×TP-rank coverage, and
> regrades them with `optima.audit.gate` on the trusted host. The typed audit witness
> is mandatory in the aggregate report and durable reopen; missing or malformed
> evidence fails closed. Charged B/C/B′ (and optional repeat) roles reject audit
> receipts and carry neither audit environment variable. This wiring has CPU/mock
> protocol and qualification coverage only. The July 7 B300 results validate the
> underlying audit mechanism, **not these new causal transport/report bytes**. Do not
> enable meaningful emissions until the exact production MiniMax-M3 arena passes the
> GPU canary below.

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

## 3. The production-wired stack (implemented and CPU/mock-covered, not GPU-qualified)

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

## 4. The adversarial matrix — target/historical audit path

The causal transport now runs the audit launch and receipt gate. These statuses still
describe the complete defense-in-depth stack, not a claim that the audit role is
indistinguishable from a charged role.

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
| 10 | **Launch fingerprinting**: behave when `OPTIMA_SLOT_AUDIT` is set (or eager-vs-graphs), cheat otherwise | mandatory graphs-on pristine-T/bench grading exercises the no-audit role and should punish a material fidelity drop | **PARTIALLY OPEN at the audit layer.** The new transport deliberately exposes an audit env marker and forces eager execution; in-process miner code can distinguish it. The quality gate is a mitigation, not proof of role indistinguishability. Closure requires isolating miner code from audit control state or an audit design indistinguishable from charged execution. |
| 11 | **Timed-workload fingerprinting**: detect the timed launch's workload shape (short fixed-budget repeated batches) and run fast-garbage only there — "phantom speed" | driver-known numerator; pristine-T distribution/task grading of selected C evidence; per-epoch prompt reseeding; independent reproduction and champion re-evaluation | **PARTIALLY OPEN.** Stable workload/role features remain visible, and a quality-preserving targeted cheat is not cryptographically excluded. Roadmap closure: draw charged work from the serving distribution and randomize regimes so “timed” has no stable fingerprint. |
| 12 | Attention-slot cheats in-engine | verify + pristine-T/bench; the generic audit transport is wired, but re-running a stateful KV-cache baseline is not | **FAIL-CLOSED / NOT CROWNABLE for unauditable attention slots** under exact slot×rank coverage: absent audit receipts cannot pass qualification. A save-free baseline call is required before activation. The MSA *indexer* slots are not in this class: their score pass is stateless and auditable. |
| 13 | Copy a champion's kernel | copy fingerprints (containment, normalized, skeleton) | closed (unchanged) |
| 14 | Tamper with timing | driver-process timing; miner code never loads in the driver | closed (unchanged) |
| 15 | Honest kernel, different rounding (the v6 case) | PASSES audit (tolerance absorbs ULP-class drift) — the false-negative rollout-KL created | **fixed** — this was the point |

The honest bottom line: the transport now makes sampled tensor comparisons a
mandatory, durable gate, but it does **not** make the eager audit role secret from
miner code in the same rank process. In-process tampering (#9), audit-role
fingerprinting (#10), and timed-workload fingerprinting (#11) remain named residuals.
Hiding an environment variable alone cannot close them: eager-vs-graphs state and
other process observations are also visible. The durable closure is process/namespace
isolation of miner code from audit control state, or a calibrated audit regime that is
indistinguishable from charged execution without contaminating timing.

## 5. Knobs, usage, and the required GPU canary

The causal bracket now implements the separate audit role and transport. Its
`SlotAuditPolicy` (private seed, integer sample-rate ppm, per-slot/per-rank minimum
calls, exact registered slots, and TP member count) is sealed into the qualification
authority before B. The legacy local `optima evaluate --fidelity-mode ...`
diagnostic was deleted in the post-arc trim.

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

Before mainnet emissions, run one final-source canary on the exact production
MiniMax-M3 image/model/topology and retain the complete qualification artifacts:

1. Seal the intended sample-rate and minimum-call policy before B, then run the
   honest champion through full primary qualification and independent reproduction.
   Every registered slot on every TP rank must meet the minimum with zero violations
   and compare errors, and both durable attempts must reopen independently.
2. Run the residual-drop sabotage control through the same path and prove the typed
   audit witness makes the aggregate report a non-retryable FAIL.
3. Inspect the worker/container evidence for every charged B/C/B′ leg: both audit
   environment variables must be empty, no audit receipt may appear, and the charged
   throughput result must remain the graph-on measurement.
4. Delete or alter one slot/rank receipt and alter one policy/request binding in copied
   artifacts; protocol validation or durable reopen must reject each mutation. Confirm
   settlement cannot accept an auditless historical report.

Until all four pass, the code is fail-closed launch machinery, not GPU evidence that
the production arena has sufficient audit call volume or that its controls separate
honest and sabotaged kernels under the newly wired path.

Passing the canary still does not close #9–#11. The one-campaign activation approval
now requires the production audit-control manifest digest, exact canary receipt digest,
and a separate explicit residual-risk-acceptance digest; activation retains all three.
Those residuals are not safely repairable by removing
`OPTIMA_SLOT_AUDIT` in a launch-day patch because eager execution itself remains
observable and sampling collectives inside charged timing would change the measured
workload.

## 6. Which fidelity mode for which arena

As a target policy, rollout-KL remains valid where its premise holds — the premise is a
*measured requirement*, not an assumption: **stock-vs-stock KL must be ~0 on
that arena config** (deterministic-capable backend, pinned autotune, fixed
batching). Record the control result in the arena row. Where the control
fails (any nondeterministic stack — e.g. M3/fa4/NVFP4), the in-engine audit is the
primary control. Its causal transport is now wired but remains activation-blocked on
the production-arena canary above. The arena registry is the intended home for this
choice, together with audit policy, bench budgets, stop cues, and template policy.
