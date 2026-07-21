# Optima — state of record

The detailed, numbers-first record of what is built, measured, and proven:
results, gates, calibration findings, run recipes, the submission ABI, and the
security model. **Where any doc and this file disagree, this file wins** (it is
kept current). The [README](../README.md) carries the quickstart and
orientation; this file carries the record.

The current referee-hardening program is governed by the
[product contract](PRODUCT_CONTRACT.md), the
[focused extraction plan](REFEREE_HARDENING_SPLIT_PLAN.md), and the
[donor disposition map](REFEREE_HARDENING_DONOR_MAP.md). The
[hardening audit ledger](REFEREE_HARDENING_AUDIT.md) records exact extracted commits,
receipts, cumulative size, accepted review findings, and explicit nonclaims. The product
contract and this state of record remain the architectural and factual authorities; the
plan and audit make the ongoing implementation reviewable and resumable.

## What is and isn't done

**Done & validated on real GPUs (H100 up to gpt-oss-120b; 4×B300 MiniMax-M3-NVFP4;
CUDA 13):** the stable/default SGLang pin is `PINNED_SGLANG` in `optima/compat.py`;
the crowned MiniMax-M3 campaign instead used source build
`0.0.0.dev1+g56e290315` at revision `56e290315b8fdb4c8c10f8e31360d9bc3d878633`.
Those are distinct runtime identities and their evidence must not be mixed; making the
source revision/image identity a mandatory authenticated arena pin remains open. The historical
GPU results validate the whole *mechanism* — typed op-slots, fused-*block* slots,
**and cross-GPU *collective*
slots** (a slot can be one op, a region behind one typed tensor boundary, or a collective
handed the process group), the seam that swaps an untrusted kernel into a spawned model
process, op-correctness, bookended throughput measurement, the historical evaluator's
in-engine audit plus pristine-reference distribution/task checks, chain-native commit-reveal
intake with cumulative copy disposition, and tamper-resistant timing. **Eleven slots:**
`activation.silu_and_mul`, `norm.rmsnorm` (ops); `attention.sdpa` / `attention.decode` /
`attention.msa_block_score` / `attention.msa_prefill_block_score`,
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

The calibrated production default remains the historical, byte-compatible v1
`B/C/B′` speed-evidence policy. A complete v2 repeat-read authority is available only
as an explicit opt-in: it runs `B/C/B′/C′/B″`, binds that read shape and estimator into
the causal authority and settlement evidence, teacher-grades both overlapping quality
triplets, and requires primary and reproduction to use the same policy. V2 has not yet
received a current-head GPU stock-null and honest-control calibration, so it is not the
production default and no v2 performance result is claimed here.

**The production referee authority is isolated and fail-closed.** Finalized submissions
move from validator-private fetch storage into immutable, hash-complete worker
publications. Candidate import, hermetic native compilation, engine construction, and
execution occur inside validator-owned OCI sessions with no network egress, read-only
root filesystems, bounded mounts/protocols, and controller-owned teardown. The trusted
controller supplies the incumbent, candidate delta, workload, role schedule, and evidence
schema; it does not load miner Python or native extensions. Qualification uses graph-on,
audit-free charged B/C/B′ roles, a mandatory separate eager/untimed audit role with a typed
host-regraded witness, and a separate pristine teacher-forced T authority.

**Current production fidelity gap:** the separate audit role, policy-bound transport,
exact slot×TP-rank coverage, typed aggregate/durable witness, and trusted-host regrade are
implemented and CPU/mock-covered. Charged B/C/B′ roles reject audit state and receipts.
These new causal transport/report bytes are not GPU-qualified: meaningful-emission launch
remains blocked until the exact production MiniMax-M3 canary proves honest primary plus
reproduction, sabotage rejection, audit-free charged legs, and fail-closed mutation/reopen.
Unauditable attention slots fail closed. In-process tampering, audit-role fingerprinting,
and timed-workload fingerprinting remain open even after that canary; see
[FIDELITY.md](FIDELITY.md).

The bounded 2026-07-19 B300 run did not satisfy that launch gate. The sabotage control was
correctly rejected. The honest primary produced no qualification verdict because two
concurrent legs shared an executor label, so its final quiescence proof saw the other live
leg. The honest reproduction passed graph and pristine-T quality, but its deep slot had
only 4 audited calls per TP rank against the required 32 and its speed gate failed at
1.005507x. It is retained as failure evidence, not an activation canary or a performance
authority. The separate audit role observed zero comparison violations, but insufficient
coverage is still a fail.

**The production intake and arena path uses SQLite and explicit target/stack identity.**
`FinalizedIntakeStore` persists finalized ordering, copy disposition, screen receipts,
qualification attempts, stack transitions, settlement, and weight-publication state. A
trusted `ArenaServiceRegistry` binds runtime/model/topology identity, decode and
long-prefill workload regimes, capacity/retry policy, and the fixed non-crown
static/build/ABI/graph/abbreviated-serving screen. A first passing qualification is retained
as `reproduction_pending`; settlement requires a second independent passing authority for
the same candidate identity and conservatively uses the lower speedup. The retained PASS
durably records a finalized head read after qualification completes and starts a fresh,
bounded reproduction SLA;
pre-PASS work remains bounded from arrival, while legacy retained rows without a known
progress block and schema-v3 migration holds remain fail-closed for explicit operator
disposition (the latter only through the terminal, evidence-preserving
`chain-archive-schema3-hold` command). Hostile bundle intake also rejects malformed gzip/DEFLATE streams during
bounded preflight or extraction before anything can be published into the cache.

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
`docs/TESTNET.md` is the runbook. Weight publication is a separate, journaled control-
plane reconciliation over the transactional global reward projection.

**The one-campaign V2 incentive composition is selected and inactive. Its arithmetic,
bounty-only durable subset, atomic activation boundary, and restart-safe publisher are
implemented, but neither activation nor V2 publication has a live receipt.** D-012's now-superseded
1,600-cell/224,000-run synthetic sweep replayed byte-identically on local
arm64/Python 3.11 and the RTX pod's x86_64/Python 3.12. Its registered-CROWN
selected cell
`15623f7679f5c1099ab48ecc88b1fe6aac926f58b309d07b3a788180848477a4`
uses gross multiplicative 1%-log units, `beta=100,000 ppm`, `tau=648,000`
blocks, `k=1,000,000`, a 648,000-block claim life, a 100,000-ppm reserve,
digest-ordered near-equal integer family shares, and a clock reset on every
accepted CROWN. It paid 98.7203% of issued principal over the synthetic horizon;
maximum measured split/withhold/sybil distortion was 3.0287%. The D-012
hash-gated selection report digest is
`9cdec61232343b663f24291c13af6b283db064370fd89a1f3b9ad6084c47cce1`.

D-015 replaced only the claim-sizing hierarchy: the funded unit is now a model
campaign, while target families remain independent frontiers and elapsed-time
clocks. Pure schema-2 `optima.finite-debt.v2` arithmetic represented 100% sizing for
one campaign and 50% sizing for each of two research campaigns. The implemented
activation approval and durable store accept exactly one immutable MiniMax-M3 campaign;
model rotation, a second campaign, and successor activation are unsupported future work.
Every family maps exactly once.
Adding 1, 10, or 100 target families changed principal by zero. All 14
preregistered screens passed. At selected `k=1`, the normal weekly tape issued one
full-sized 4.4%/5% claim for one campaign, or—in the historical research cells—one
half-sized claim in each of two campaigns (one full share in aggregate), rotating
across families. Those tapes paid
100%, expired zero, drained to zero outstanding, and reached at most five days of
payment latency under zero and saturated discovery. Worst normal
utilization was 77.4136%; five-day cadence was marginal and four-day cadence was
overloaded. `k=1.25` was already marginal; at `k=1.5` the worst rows overloaded
while other rows remained marginal, and `k=2` was plainly overloaded. D-015 semantic
report digest:
`7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590`.
This is deterministic control-plane/ROI sensitivity, not token-value, validator-
influence, GPU-performance, or miner-equilibrium evidence. Its normal tape does not
model sustained simultaneous wins in every active family. The raw D-015 config,
simulator, ledger, and report are local-only experiment records; tracked code binds
the report digest but this PR alone cannot reproduce the sweep. D-015 has no pod receipt.

A tracked one-campaign supplement closes the simultaneous-family evidence gap with
64 launch/stress cells over 1/2/5/10 independently winning M3 families,
7/14/30/90-day cadences, and empty/saturated discovery. Its semantic report digest is
`505fed4d40a6acc6bc92d6330170e8e2260a52e5f3099c22a6c0eb4b2308c672`.
This is deterministic accounting/ROI evidence, not a pod, chain, GPU-performance,
token-value, or miner-equilibrium receipt. After the one-campaign activation,
publication-fence, and audit-transport changes, the exact repository suite is
2,193 passed with 19 skips; the tracked supplement reproduces the digest above.

The earlier live testnet-netuid-307 `chain-incentive-shadow` receipt exercised
**only the historical D-012 registered-CROWN class**: it twice reopened exact
finalized membership, mapped
three explicitly synthetic claims to 900,000 miner ppm plus 100,000 reserve ppm,
and wrote `submitted=false` (receipt digest
`a4006912ec3e34b98fe51031fe25864915e4a2d588209877c41a459a6094dcf3`).
No wallet or chain mutation was used. It does not test D-015 campaign bytes; a fresh
campaign-policy shadow remains activation work.

D-013 selected the separate reviewed-discovery composition. Its cell
`8561028c943738da2fe622e5f5c9fd43ebec16fdd59feab3561de25fbfa450d9`
sets a 50,000-ppm discovery epoch cap, caps each award at one such epoch of
principal (50,000 weight-ppm epoch units), and gives it no campaign share, family
clock, time bonus, renewal, or permanent title. Its 648,000-block lifetime begins
when the qualified discovery win is retained, not at later review; review delay
consumes the window and review at or after expiry cannot mint. The pure policy
intends registered promotion followed by fresh requalification/CROWN, or one finite
bounty, never both. Durable schema 5 currently implements review-pending retention
and `bounty_only`, not the promotion branch, so this is not yet end-to-end
same-work enforcement. The exact composed payout is:

```text
P_d     = min(50,000, live discovery debt)
P_c     = min(900,000 - P_d, live registered-CROWN debt)
reserve = 1,000,000 - P_d - P_c
```

Each class has its own claim-digest largest-remainder allocation. The D-013 report
digest is `6bdfce26e4e6090e0dcc8814a636c665f28d1ff20945a09d43a9a90dc94151fc`;
the retained report-file SHA-256 is
`7369c6890dcc880b5f7295a94d07f915d59241e23d95b2c9328295780c99fb38`.
Its 9-cell × 36-scenario × 10-seed matrix produced 3,240 rows and replayed
byte-identically locally and on the RTX pod. The selected cell paid
273,000,000/273,000,000 units of non-departed principal with zero expiry or
outstanding balance and 100% worst-run payout; 9,000,000 units of departed debt
were forfeited/cancelled. Analytic and measured saturated CROWN-capacity dilution
was 55,555 ppm (5.5555%), while saturated tapes eventually paid 100% CROWN
principal. These are synthetic accounting results, not miner-equilibrium,
token-value, or GPU-performance evidence. Selector checks passed 11/11. After a
multi-pass restart/cardinality audit, the pre-D-015 implementation passed 98
focused tests and the repository passed 2,135 tests with 19 skips; historical pod
conformance passed 111 tests under Python 3.12 with CUDA hidden. Those pod receipts
do not cover D-015.

D-014 held the selected D-013 policy fixed and measured review-delay sensitivity in
8 delays × 3 service modes × 4 scenarios × 3 seeds = 288 rows. The artifacts
replayed byte-identically on local arm64/Python 3.11 and pod x86_64/Python 3.12.
Its preregistered 0/1/7-day SLA covered 108 rows and passed all of them: discovery
paid 100%, expiry/unissued principal was zero, maximum instantaneous CROWN-capacity
dilution was 55,555 ppm, and CROWN paid-fraction regression versus zero delay was
zero percentage points. The 90/120-day cases issued no stale debt. The 30/60/89-day
cases were retained as diagnostics only and do not expand the review SLA. Report
digest: `f0939d67241dffa49aac95c035c43dd7ea14b51eb2671fe106cb09347511b7ef`.
This is deterministic synthetic accounting evidence, not external review authority,
publication, activation, by itself proof of durable-state hardening, or GPU
performance evidence.

Pure arithmetic lives in `optima/incentive_composition.py`; schema-5 pre-activation
durable authority lives in `optima/chain/incentive_composition_store.py`; and the signer-free
`chain-incentive-composition-shadow` surface lives in
`optima/incentive_composition_shadow.py`. Before D-015, its live signer-free
testnet-netuid-307 run passed at finalized block 7,586,146 with metagraph size 6.
Explicitly synthetic
states allocated 850,000 ppm to registered-CROWN claims, 50,000 ppm to
reviewed-discovery claims, and 100,000 ppm to reserve, exactly 1,000,000 ppm total;
the receipt wrote `submitted=false`. Semantic digest:
`3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`;
receipt-file SHA-256:
`ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`.
The command constructed no wallet and the receipt supplies no independent review,
settlement, publication, debt-debit, D-015 policy, or activation authority.
Schema-4→5 migration
creates empty composition tables and no retroactive debt.
Activation fails with any retained legacy discovery row (and requires validated,
clean open-debt state), because V1 has no journal that can prove a mutable terminal
flag. Active composition disables legacy discovery auto-award. The restart audit also
requires exact paired qualification, evidence, CROWN speed, principal, family clocks,
lifecycle events, and all-and-only balance revisions; every reproduced
substitution/cardinality case now fails closed.

`optima/chain/incentive_activation.py` and `chain-activate-incentives` implement the
one-way wallet-free cutover. They bind canonical core/composition/approval files plus
an independently recorded approval digest at the approval's exact finalized intake
cursor. Preflight requires the complete catalog-derived reward-family roster to identify
exactly one retained evaluation arena and derives the campaign ID deterministically from
that arena/catalog/roster. Arena, evaluation-stack, catalog, and finalized-membership
digests are fields of the independently pinned approval itself, not receipt-only facts;
preflight must reproduce all four, and the activation JSON plus reward event retain and
reopen those exact bytes. That approval also requires and retains the production
audit-control manifest, final B300 canary receipt, and explicit audit-residual-risk
acceptance digests. Core and composition activate in one SQLite transaction and
raise schema 5→6, which fences older runtimes. Existing V1 projection/publication state
is deliberately not bridged: launch needs a fresh/quiescent activation database, and an
existing V1 publisher database remains a fail-closed blocker rather than an invitation to
delete history. This includes metadata-only V1 projection/dry-run state. The constrained
first 1–2 week operation assumes stable reserve, validator, and positively weighted miner
registrations; any departure or UID change halts publication rather than remapping or
skipping the gapless boundary. The 90-day claim lifetime prevents ordinary expiry during
that window. No production manifest set, shadow, or live activation receipt exists
yet. The activation command binds the independently approved canary digest but does not
parse or regrade the canary receipt. Non-PASS evidence must be rejected by the external
reviewed evidence-package and approval process; digest equality alone is not a PASS
judgment.

The legacy-V1 `set-weights` command additionally supports the **all-uncrowned
bootstrap burn**: `--burn-hotkey <ss58>` projects the full pool to one designated
registered hotkey (the subnet owner's burn registration) instead of failing closed
before the first crown. `FinalizedIntakeStore.build_burn_weight_projection` refuses
on any active reward claim, any crowned (generation > 0) arena, any activated
incentive composition, or an unregistered hotkey, and digest-binds the empty
settlement state — so a burn vector cannot survive past the first real crown. The
projection flows through the normal journal/CAS/reconcile/confirmation path with
one explicit override: the burn CLI branch passes `require_current_crown=False`
to `reconcile_weight_publication` (the burn vector is crownless by construction;
all other invocations keep the pre-crown submission gate). The store's write-once
emissions-policy binding applies: the policy parameters chosen at burn time are
the parameters every later V1 projection on that database must reuse.
Unit/CLI-tested incl. a mocked-chain non-dry-run publish-to-confirmed flow and a
live testnet-307 dry-run (2026-07-21); no live burn publication receipt yet.

`optima/chain/debt_publication.py` and `set-debt-weights` implement the active V2
publication authority. The command retains the earliest gapless economic boundary,
binds the exact signer-facing vector, resumes in-flight journals after restart, grades
finalized chain readback, and debits only after the intake cursor catches the readback.
Late boundaries remain nominally gapless and catch up no faster than one full policy
cadence after the prior confirmation. There is no live V2 publication receipt yet.

The final hardened-source pod cycle also ran a fresh live
`chain-validate --intake-only --once`: at finalized block 7,586,142 it saw/reserved
19 retained entries, rejected five malformed payloads, and performed no screens,
decisions, settlements, or holds. Reopening the same database at finalized block
7,586,144 produced zero work in every counter. This is live reveal/intake/restart
evidence without a new candidate, qualification, wallet, signature, or publication.

The current durable boundary is narrower than the selected pure policy. Active
discovery settlement can atomically retain a `ReviewPendingDiscoveryWin`; reviewed
issuance can create one bounded `bounty_only` claim whose award/expiry remain anchored
to that win. `registered_promotion` is intentionally rejected until existing typed
`DiscoveryWinRecord`/`DiscoveryPromotion` authority is transported and reopened,
the target is registered, fresh requalification/CROWN is linked, and one work
identity spans discovery and registered lanes. Bounty uniqueness does not substitute
for that cross-lane identity. The finalized
`expire_review_pending_discovery_wins` API terminalizes overdue pending wins as
`review_expired` and appends `discovery_review_expired`; production orchestration
must still call it reliably.

A finalized registered-family invalidation API is also landed: it cancels open debt
and inserts an invalidation clock marker so the next CROWN acts as a first CROWN.
It accepts an external invalidation digest and does not independently decide or
grade runtime invalidity.

Meaningful V2 emissions still require: exact production MiniMax-M3 family and reserve
manifests plus a fresh campaign-policy shadow; retained membership-departure history
rather than only a current snapshot; independently graded review and runtime-
invalidation authority; the promotion transport/linkage above; the production audit GPU
canary; and actual activation/mainnet operations. The current
`review_digest` remains controller-supplied and content-bound. Model rotation, a second
campaign, and successor activation are unsupported future work, not launch settings.

**The serving release is separate from evaluation and chain state.** Approved
`IntegrationReviewRecord`s authorize exact contributions in an
`EngineReleaseManifest`; deterministic model provisioning seals every file in a clean model
tree and rejects transient `.cache` paths rather than silently excluding mounted bytes. The
chain-independent release module emits signed descriptors, deterministic source
and wheel artifacts, SPDX SBOM, provenance, a pinned serve specification, and an OCI build
context. The serving wheel excludes chain, wallet, settlement, and evaluation-control code.

**Still open:** execution and live validation of the implemented V2 activation/publication
authority plus mainnet operations (production family catalog/reserve plus exact shadow,
independently graded discovery review and runtime invalidation, membership-departure
history, discovery-promotion transport/linkage, the production audit GPU canary,
owned subnet,
stake/permits, hosted bundle storage); broader optimization targets (MLA /
weight-absorbed attention, GEMM,
comms-overlap blocks); and B300-only proof for SM103/CuTe, NVLink/custom collectives,
topology-specific calibration, TP4 role swaps, and the existing MiniMax-M3 campaign
kernels. The global-versus-per-arena SGLang pin policy is also open. The earlier measured
calibration findings remain below.

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

### Historical deterministic-arena KL calibration findings

These measurements remain valid for arenas whose stock-vs-stock control is approximately
zero. They are not the MiniMax-M3 launch policy: M3 requires the separate audit role and
typed host-regraded witness, with rollout-KL advisory.

1. **The KL threshold must be calibrated to the model's nondeterminism noise
   floor**, not hand-picked. We measured it on gpt-oss-120b: stock-vs-stock KL with
   `--no-deterministic` is **3.9e-4** (1/2041 flips) — the floor. Set ε = k×floor
   (e.g. 5×), and run with `enable_deterministic_inference` so the floor → ~0 and
   kernel drift is cleanly attributable. (The faithful rmsnorm above sat at 24×
   the floor — genuinely above any sane threshold, correctly flagged.)
2. **Benchmark accuracy needs large n.** At n=12, GSM8K has a ~12% std; a 2-problem
   flip reads as "−16.7%." On a deterministic arena with a near-zero stock control,
   KL can be the dense, low-variance gate while benchmark accuracy remains a capability
   floor at ~100–200 samples. M3 does not satisfy that premise.
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
   **score big MoE in deterministic mode**. The historical diagnostic used
   `--kl-advisory` where that was unavailable; current causal qualification instead
   requires the separate audit witness plus pristine-T evidence. (KL is also hardened:
   a genuinely degenerate candidate — all-non-finite logprobs — reads as maximal
   divergence, not 0.)
5. **The KL gate is not mean-only.** `kl_gate_ok` also caps the **argmax-disagreement
   rate** (default 1%) and an opt-in **p99 KL** — so a *sparse* cheat (bit-exact
   almost everywhere, a few tokens flipped) that keeps `mean_kl` under the threshold
   is still caught by the magnitude-independent flip rate. Calibrate the rate to the
   noise floor: in deterministic mode a faithful kernel sits at **0 flips**, so the
   default is safe. On M3, KL is advisory while the mandatory audit witness and
   pristine-T evidence carry fidelity authority.
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
  slots.py                  # the slot ABI: SlotSpec catalog (11 slots; kind = op|block|collective)
  seams.py                  # single source of truth for the seam adapters (bootstrap/activate/compat derive from it)
  eval/scoring.py           # noise-robust speedup verdict (bookended A/B, noise-derived margin, no-decision)
  audit.py                  # the IN-ENGINE AUDIT: sampled per-call stock-baseline comparison inside the scored engine
  receipts.py               # strict active/routed/completed/fallback member coverage; diagnostic, not crown authority
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
  compat.py                 # default PINNED_SGLANG (0.5.13.post1) + strict version/seam canary
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
    scoring.py              # bookended B/C/B' pairing, noise-derived bar, NO-DECISION
    reference_quality.py    # pristine-T quality record (NLL/top-k KL/argmax/coverage/task)
    oci_backend.py          # validator-owned no-egress worker lifecycle and native prebuild
    qualification_runner.py # charged B/C/B', separate audit witness, pristine-T, aggregate verdict
    engine_worker.py / _launch.py
  arena_service.py          # registered runtime/model/topology/workload + non-crown screen
  stack_manifest.py         # evaluation/release stack identity + integration review
  settlement.py             # paired reproduction and transactional target settlement
  finite_debt.py            # pure finite-claim issuance, lifecycle and epoch projection arithmetic
  incentive_shadow.py       # signer-free synthetic projection against exact finalized membership
  incentive_composition.py  # pure reviewed discovery + two-class projection arithmetic
  incentive_composition_shadow.py # signer-free synthetic composed membership projection
  model_provision.py        # exact model-tree content receipt
  release.py                # signed chain-independent Engine release artifacts
  chain/intake.py           # SQLite production authority + schema-5 preactivation/schema-6 active incentive facades
  chain/finite_debt_store.py # schema-4 activation/clock/claim/invalidation/epoch authority
  chain/incentive_composition_store.py # schema-5 state + atomic one-campaign schema-6 activation
  chain/incentive_activation.py # wallet-free retained-arena/catalog/membership activation preflight
  chain/debt_publication.py # gapless restart-safe V2 projection/readback/debit authority
  bundle_hash.py            # deterministic bundle identity
  cli.py                    # developer, chain intake, legacy/V2 weights, activation, shadows, model/release commands
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
(`LayerCommunicator` defer-gate + the fused-moe export wrap). The default stable SGLang
pin (0.5.13.post1, see `optima/compat.py`) **does** ship a hook/plugin framework (`srt/plugins/hook_registry.py`,
added by PR #21388 — present at the pin), so migrating the seam to a sanctioned
`sglang.srt.plugins` entry-point hook is a tracked option (`integrations/sglang_plugin.py`
is the shim); the `.pth` path is kept primary today because it is version-independent and
known spawn-safe.

The direct developer evaluator retains baseline/candidate launches. Crownable
qualification uses graph-on/audit-free B/C/B′ bookends, a mandatory separate eager/untimed
audit role with a typed host-regraded witness, and a separately launched pristine T quality
authority; it repeats a passing candidate under independent authority before settlement.
The exact evaluation stack differs from its incumbent by one registered singleton or
atomic target.

**Tamper-resistant timing and evidence:** candidate import, build and engine execution
remain inside a disposable no-egress OCI worker. The trusted controller assigns roles,
times requests, authenticates bounded evidence, launches the pristine T worker, and owns
teardown; it never imports candidate code or trusts the candidate as its own grading
oracle.

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

# op-correctness on device: faithful PASSes, broken FAILs
.venv/bin/python -m optima.cli verify examples/miner_rmsnorm_triton --device cuda
.venv/bin/python -m optima.cli verify examples/miner_rmsnorm_broken --device cuda
```

End-to-end throughput + fidelity scoring is validator-side: the chain intake loop runs
graph-on/audit-free charged B/C/B′, the separate eager/untimed audit role and typed
host-regraded witness, and pristine-T in no-egress workers, then
settles only after independent reproduction (TESTNET.md is the runbook; the
legacy local `evaluate`/`bench` diagnostics were deleted in the post-arc trim —
the numbers recorded in this file were measured with them while they existed).

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
**Eleven slots today:**

- `activation.silu_and_mul` — `entry(x, out)` — Qwen/Llama-class MLP (op).
- `norm.rmsnorm` — `entry(x, weight, out, eps)` — universal; fires on gpt-oss (op).
- `attention.sdpa` — `entry(q, k, v, out, sm_scale, causal)` — scaled-dot-product
  attention (block; the op-correctness demo of the wider boundary).
- `attention.decode` — `entry(q, k, v, seq_lens, sm_scale, out)` — paged-decode
  attention; the seam extracts the running model's paged KV and routes decode through
  it (block; eager-only gather MVP — a paged-direct, CUDA-graph-safe contract is next).
- `attention.msa_block_score` — the MiniMax sparse-attention block-score stage
  (block; `matched_ratio` vs high-precision ground truth — see the M3 arena work).
- `attention.msa_prefill_block_score` — the prefill indexer's causal block-score
  sheet; the validator retains stock top-k selection/attention and verifies per-row
  `topk_overlap` on a long-chunk causality-sensitive shape.
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

## Anti-copy & settlement

Production intake is the chain's NATIVE timelock commit-reveal (`optima chain-validate`,
runbook `docs/TESTNET.md`); copy disposition, qualification, and settlement are
transactional inside the SQLite intake authority (`optima/chain/intake.py`). The legacy
local JSON-ledger round simulator (`optima commit/reveal/ledger/legacy-settle`,
`optima/commit_reveal.py`) was deleted after the chain path superseded it.

- **commit-reveal** binds the content hash on chain before the URL is readable; the
  reveal block is the anti-copy priority timestamp, so you can't commit to a bundle
  you saw revealed (copying at reveal time is impossible).
- **copy detection** (`optima/copy_fingerprint.py`): cumulative in **finalized chain
  order** (a copy revealed later is caught and retroactively demotable), on the exact
  content hash OR a **reformat-invariant fingerprint** (AST-normalized — a
  reflowed/recommented/renamed-whitespace copy with a fresh hash is still demoted).
  Fingerprints are computed **per slot** over each op's transitive bundle-local
  **import closure**, and intake also compares per-FILE fingerprint sets by
  **containment** — relocating a stolen body into an imported `_impl.py` behind a
  re-export shim, or padding the stolen slot with extra variants, still demotes
  (while two miners merely vendoring the same public utility never match; a padded
  multi-op submission cannot even resolve a registered competition target). A
  **structural** fingerprint (names and constants blanked) additionally flags
  rename + constant-tweak near-copies as an **advisory** at intake (surfaced for
  review, never auto-demoted — skeletons can collide).
- **settlement**: explicit singleton or atomic competition targets with transactional
  stack state (`optima/settlement.py`); one passing qualification is retained as
  `reproduction_pending`, a second independent passing authority is required, and the
  settled speedup is the **lower** of the two. Emission projection is separate and
  policy-driven (`docs/EMISSIONS_POLICY.md`). The wired authority is legacy V1
  standing/discovery credit; the selected but inactive V2 instead uses finite
  multiplicative-log CROWN principal plus separately bounded reviewed-discovery
  principal, with independent payout classes and expiry. Durable discovery issuance
  is currently bounty-only; selected promotion remains fail-closed pending typed
  cross-lane authority. Neither generation is winner-take-all.

Robust scoring (see `optima/eval/scoring.py`), built for a validator that **can't lock GPU
clocks**: each arm recomputes one charged rate from the pooled conditioning and timed
batch token/interval evidence; the candidate is **bracketed by a
baseline before and after** (B,C,B'); the speedup is paired against the baseline mean; the
bar is **derived from the measured baseline noise** (`1 + max(margin, k·noise)`) not a
hand-picked constant; a round whose bracketing baselines disagree past a tolerance is
**NO-DECISION** and cannot crown. The ledger records a crownable speedup or 0.0. Fidelity
uses the mandatory separate audit witness plus pristine-T evidence; M3 treats rollout-KL
as advisory. Where a deterministic arena gates KL, a **coverage (tail-mass) guard** catches
a flattened distribution whose visible head matches, argmax-rate catches sparse flips,
per-slot thresholds calibrate to each slot's floor, and `aligned_kl` counts early-stop as
dropped positions. Plus per-epoch seeded prompts
(anti-overfit), **shape jitter** on the per-op verify (count dims vary per run, so a kernel
can't hard-code the verify shapes), `ignore_eos` so both sides emit identical token counts
and the throughput numerator is a driver-known fixed budget (not a scheduler-reported
count), a `max_running_requests` knob to score at a serving-realistic batch, and a
**stale-champion** flag at settle when the stack/runtime identity differs (re-baseline on a bump).

## Security model

Triton/CuteDSL submissions contain Python host launch code and generated device code;
static inspection is therefore a tripwire, not the trust boundary. Production candidate
execution runs in an ephemeral, validator-owned OCI worker with no network, read-only root,
bounded mounts, dropped privileges/capabilities, seccomp/resource policy, a separate CUDA
context, and authoritative teardown. The controller owns timing, arm identity, evidence
frames, and the pristine T quality authority, and never imports candidate code. Chain keys
and weight publication remain in the separate control plane.

Unsafe direct execution flags exist only for non-crownable development diagnostics on
pods that cannot provide the required OCI or namespace boundary.

Worst case for a fully-compromised kernel is one wrong score for itself;
cross-validator consensus catches a rogue validator.

## What's MVP vs. production

| Concern | Now | Production |
|---|---|---|
| Slots | 11: silu/rmsnorm, attention ×4, MoE ×2, all-reduce, AR+norm epilogues ×2 (deep via dep_patches) | + MLA, FP8/FP4 GEMM, graph-safe paged attention |
| Throughput gain | **two crowned bundles on M3-NVFP4/4×B300: 1.044×/1.049× (shallow) and 1.074×/1.071× (deep), each double-proven** | keep beating the pinned baseline as it advances |
| Model | gpt-oss-120b (1×H100); MiniMax-M3-NVFP4 (4×B300, TP4) | DSV4-scale (multi-GPU, TP/PD/EP) |
| Quality gate | Mandatory separate eager/untimed audit role + typed host-regraded exact slot×TP-rank witness is implemented and CPU/mock-covered; charged B/C/B′ are graph-on/audit-free; historical B300 receipts validate the underlying mechanism, not the new bytes | pass the exact production M3 GPU canary; retain explicit risk acceptance for in-process tampering, audit-role fingerprinting, and timed-workload fingerprinting; then add full-vocab reference-seam checks + large-n (100–200) benchmarks |
| Scoring noise | noise-derived margin + calibrated v1 B/C/B′ + no-decision (no clock-lock needed); v2 B/C/B′/C′/B″ is explicit and uncalibrated | + calibrate v2 repeat reads; interleaved per-iter A/B + locked clocks where available |
| Isolation | validator-owned no-egress OCI worker; trusted controller never loads candidate code | deploy the same policy on each validator's production runtime |
| Champion | explicit singleton/atomic targets; two independent PASSes; transactional stack settlement | continued whole-stack regression and re-baseline on pin changes |
| Chain | **native timelock commit-reveal + hash-verified finalized intake + SQLite authority + journaled weights**, with chain behavior run on testnet | own subnet, production permits/cadence, hosted bundle store |
| State | SQLite, transactional single-writer authority | operational backup/replication and monitoring |
| Release | signed, chain-independent Engine release with model seal, SBOM/provenance and deterministic OCI context | registry publication and serving-fleet rollout policy |

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
