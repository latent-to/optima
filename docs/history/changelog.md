# Changelog

This history records architectural changes and the evidence available when they landed. It is not the current operator contract. For present capabilities, limitations, and validation status, use [State of record](../reference/state-of-record.md).

Historical benchmark results on this page retain their original meaning. Results before the July referee hardening are not retroactively treated as crowns from the final SQLite, isolated B/C/B′/T, independent-reproduction, or signed-release path.

## 2026-05-31 — Initial mechanism

The initial Optima implementation established the core thesis:

- a validator-owned typed slot ABI;
- SGLang post-import seam adapters;
- throughput plus model-quality evaluation;
- local commit-reveal and hill-climb state;
- compatibility checks around a pinned runtime.

The first seams targeted `SiluAndMul.forward_cuda` and `RMSNorm.forward_cuda`. Even at this stage, candidates filled validator-allocated outputs rather than returning final model responses.

Foundation commit: [`d066d45`](https://github.com/latent-to/cacheon/commit/d066d45).

## 2026-06-01 — Real workloads and stronger quality gates

Early evaluation moved from toy prompts to GSM8K and MMLU workloads with realistic decode lengths. Throughput, KL, and task accuracy were gathered from the same comparison path.

Mean KL alone was found insufficient for sparse failures. The gate added p99 KL and argmax-disagreement rate, making a rare but severe token change visible even when the mean stayed small.

The ABI expanded from individual operations to blocks:

- `attention.sdpa` and `attention.decode` introduced bounded attention regions;
- the RadixAttention seam connected those contracts to live SGLang;
- graphs-on evaluation became the default because graphs-off crippled the baseline and distorted economic results;
- per-slot quality thresholds were introduced for numerically different attention reductions.

The same period added `(prepare, forward)` slots and `moe.fused_experts`, allowing layout or quantization transforms once at model load while retaining a narrow forward boundary.

Representative changes: [PR #1](https://github.com/latent-to/cacheon/pull/1), [PR #3](https://github.com/latent-to/cacheon/pull/3), [PR #4](https://github.com/latent-to/cacheon/pull/4), and [PR #5](https://github.com/latent-to/cacheon/pull/5).

## 2026-06-01 — Isolation, framework experiments, and the narrow waist

A fenced framework mode allowed bounded engine experiments under token-match evaluation. Candidate execution was required to run without network egress; when isolation could not be established, the path changed from silent continuation to fail-closed behavior.

The SGLang seam expanded to `FusedMoE.forward_impl`. Low-bit output comparison added cosine similarity and a norm guard, since elementwise tolerances are not meaningful for every quantized implementation.

The canonical slot contract was then written around four invariants:

1. validator-owned outputs and call sites;
2. strict placement upstream of sampling;
3. correctness against high-precision or dequantized truth;
4. no trust in miner-reported measurements.

Arbitrary rebuild scripts were rejected. Only validator-shipped reviewed patchers could implement the rebuild escape hatch.

Representative changes: [PR #6](https://github.com/latent-to/cacheon/pull/6), [PR #8](https://github.com/latent-to/cacheon/pull/8), [PR #9](https://github.com/latent-to/cacheon/pull/9), and [PR #11](https://github.com/latent-to/cacheon/pull/11).

## 2026-06-02 — Collective slots

Profiling showed that decode at tensor-parallel scale was frequently communication-bound. Optima introduced a collective all-reduce benchmark and then a real `collective.all_reduce` slot at `GroupCoordinator.all_reduce`.

Collectives widened the capability boundary by passing a process group, so single-process verification became invalid. `verify_collective` added multi-rank execution and comparison with trusted fp32 cross-rank partials. The design also established terminal all-rank selection: after ranks select a candidate collective, one rank cannot safely fall back while peers enter candidate communication.

Changes: [PR #12](https://github.com/latent-to/cacheon/pull/12) and [PR #13](https://github.com/latent-to/cacheon/pull/13).

## 2026-06-07 to 2026-06-10 — Chain plumbing and noise-aware scoring

The first chain-facing system added durable evaluation records, block-hash prompt seeding, chain SDK checks, commitment reads, and weight submission. This was an earlier JSON-ledger architecture, not the current SQLite authority.

Scoring moved to a bookended `B, C, B′` comparison. Candidate throughput was paired with the mean of opening and closing baselines, while baseline disagreement produced `NO_DECISION`. The threshold combined a minimum margin with measured noise. Graph-safety declarations, count-dimension jitter, out-of-process verification, and per-slot economic state were added in the same period.

Changes: [PR #14](https://github.com/latent-to/cacheon/pull/14), [PR #15](https://github.com/latent-to/cacheon/pull/15), [PR #16](https://github.com/latent-to/cacheon/pull/16), and [PR #17](https://github.com/latent-to/cacheon/pull/17).

## 2026-06-21 to 2026-06-26 — Quantized MoE and MiniMax-M3

Eligibility began pairing a layer's quantization format with candidate metadata so dense and quantized implementations could not be confused. MiniMax-M3 support added:

- the clamped `swigluoai` activation profile;
- NVFP4-aware MoE preparation and execution;
- per-model slot profiles;
- the `attention.msa_block_score` semantic slot;
- fidelity policies suitable for low-bit outputs.

The pinned SGLang runtime moved to `0.5.13.post1`. Compatibility checks were split so a moved or missing chokepoint fails, while availability of a newer upstream release is informational until a reviewed pin bump and requalification.

The first dedicated miner and submission guides also landed during this period.

Changes: [PR #18](https://github.com/latent-to/cacheon/pull/18), [PR #19](https://github.com/latent-to/cacheon/pull/19), [PR #20](https://github.com/latent-to/cacheon/pull/20), [PR #21](https://github.com/latent-to/cacheon/pull/21), and [PR #22](https://github.com/latent-to/cacheon/pull/22).

## 2026-07-02 to 2026-07-07 — Adversarial hardening and deep epilogue

An adversarial review tightened copy detection, source-tree scanning, symlink handling, dispatcher eligibility, benchmark parsing, collective shape jitter, and attention/MoE fallbacks.

The fused-epilogue work introduced two collective slots:

- `collective.ar_residual_rmsnorm`;
- `collective.moe_finalize_ar_rmsnorm`.

The deep path added reviewed dependency overlays. A bundle could declare a constrained unified diff against admitted FlashInfer source; a validator-owned patcher applied it to a copied overlay, and runtime integration redirected only the admitted late-bound source path. The installed dependency was never mutated in place.

The `moe_export` state machine established scoped export/consume behavior between fused MoE production and the AR+residual+RMSNorm consumer.

Changes: [PR #23](https://github.com/latent-to/cacheon/pull/23), [PR #24](https://github.com/latent-to/cacheon/pull/24), and [PR #25](https://github.com/latent-to/cacheon/pull/25).

## 2026-07-07 — First historical submitted speedups

The shallow fused epilogue measured 1.044× against its noise-derived bar on MiniMax-M3-NVFP4, TP4, 4×B300 with graphs enabled, then reproduced at 1.049× on an independent prompt seed. The deep epilogue measured 1.074× and reproduced at 1.071×. The recorded audit samples had zero violations.

These were the first submitted Optima kernels to beat the pinned SGLang baseline through the then-current referee. They are historical performance receipts. They predate the final hostile evaluation stack, SQLite authority, pristine T, independent two-pass settlement, and signed release pipeline.

## 2026-07-07 — Fidelity authority correction

On the MiniMax-M3-NVFP4 arena, two stock launches were not logit-identical enough for rollout KL to act as a reliable primary gate. Timing changes altered low-level nondeterminism, causing faithful candidates to fail.

The earlier path introduced an untimed in-engine audit that sampled seam calls, replayed trusted baseline behavior on cloned inputs, and graded them under slot tolerances. KL became advisory on that arena. This corrected a false-failure mode, but the in-engine audit remained part of the older referee design. The final architecture later replaced candidate-adjacent quality authority with a separately launched pristine T reference over sealed trajectories.

Change: [`4eef097`](https://github.com/latent-to/cacheon/commit/4eef097).

## 2026-07-08 — Historical native chain loop

Native timelock commit-reveal, hostile archive fetch, re-hashing, copy fingerprinting, evaluation, per-slot settlement, and weight projection were connected on public testnet netuid 307.

The deep fused-epilogue proposal completed that historical loop at 1.072× against a 1.026 bar, with score 1.0717 and 12,824 sampled calls with zero violations. Weight publication remained a dry run because the validator lacked the required permit on the externally owned subnet.

This demonstrated live chain transport and then-current autonomous evaluation. It did not exercise the later SQLite intake, isolated current-schema B/C/B′/T evidence, two-pass reproduction, or release promotion.

Change: [PR #26](https://github.com/latent-to/cacheon/pull/26).

## 2026-07-09 to 2026-07-10 — Submission ergonomics and MSA prefill

Developer onboarding and documentation were simplified, and the MSA prefill scoring region became a real slot and live seam at `flash_prefill_with_topk_index`.

Two direct TP4 B300 lanes measured the MSA prefill candidate at 1.115× and 1.205×, with zero recorded audit violations. No crown or settlement was attempted. These remain historical direct-evaluator performance receipts.

Changes: [PR #27](https://github.com/latent-to/cacheon/pull/27), [PR #28](https://github.com/latent-to/cacheon/pull/28), [PR #29](https://github.com/latent-to/cacheon/pull/29), and [PR #30](https://github.com/latent-to/cacheon/pull/30).

## 2026-07-10 — Product and referee architecture reset

A whole-system optimization prototype exposed a product-model error: complete engine execution had been conflated with whole-engine economic identity. That model would have required later candidates to package incumbent work and would have made the chain-selected evaluation stack a production dependency.

The reset established the current invariants:

- Optima Engine is chain-independent;
- proposals, crowns, integrated contributions, and releases are separate objects;
- complete engines are the isolation unit;
- registered slot or atomic deltas are the reward unit;
- evaluation and release stacks are separate content-addressed manifests;
- crown and ship are independent decisions;
- cross-cutting work uses a bounded discovery lane.

A donor implementation at `1b1c842b` was retained as an architectural reference but not merged wholesale. Current main was built through bounded source extraction from [`203bb559`](https://github.com/latent-to/cacheon/commit/203bb559).

## 2026-07-11 — Product contract and hardened target waist

The first hardening group encoded the product contract and rebuilt the miner-facing boundary:

- capability variants and domains;
- typed output/workspace verification;
- live MSA prefill routing;
- setup gating;
- positive active/routed/completed receipts;
- the validator-owned target catalog;
- distributed collective graph replay;
- governance contracts for the extraction.

The target catalog made economic identity independent of bundle row order. It registered singleton and atomic targets, explicit displacement, compatible composition, allowed features, and frozen slot contract digests.

Merged changes: [#31](https://github.com/latent-to/cacheon/pull/31), [#32](https://github.com/latent-to/cacheon/pull/32), [#33](https://github.com/latent-to/cacheon/pull/33), [#34](https://github.com/latent-to/cacheon/pull/34), [#35](https://github.com/latent-to/cacheon/pull/35), [#36](https://github.com/latent-to/cacheon/pull/36), [#37](https://github.com/latent-to/cacheon/pull/37), [#38](https://github.com/latent-to/cacheon/pull/38), and [#39](https://github.com/latent-to/cacheon/pull/39).

## 2026-07-11 to 2026-07-12 — Stack identity and isolated execution

Vendor provenance was pinned before execution identity work proceeded.

The new stack layer added strict `EvaluationStackManifest` and `EngineReleaseManifest` types, proposal and integrated contribution references, exact marginal arm planning, cohorts, rollback, deterministic engine-tree materialization, source/native namespace rewriting, and complete tree reopening.

The execution layer added:

- host-owned OCI lifecycle and device allocation;
- no-egress/read-only/bounded-mount workers;
- hermetic native prebuild and immutable native publication;
- bounded authenticated streaming sessions;
- host-side timing and teardown;
- exact stack/tree/native/arena/model launch bindings;
- B/C/B′ materialization from one frozen incumbent.

Merged changes: [#40](https://github.com/latent-to/cacheon/pull/40), [#41](https://github.com/latent-to/cacheon/pull/41), [#42](https://github.com/latent-to/cacheon/pull/42), [#43](https://github.com/latent-to/cacheon/pull/43), [#44](https://github.com/latent-to/cacheon/pull/44), [#45](https://github.com/latent-to/cacheon/pull/45), and [#46](https://github.com/latent-to/cacheon/pull/46).

## 2026-07-12 — Pristine reference and causal qualification

Qualification became a retained authority rather than a report emitted by a benchmark command.

The new path introduced:

- versioned calibration and reference profiles;
- content-addressed evidence publications;
- separately launched pristine T quality authority;
- sealed prompt trajectories and hidden tasks;
- host device-state and causal ordering receipts;
- aggregate `PASS`, `FAIL`, and `NO_DECISION` semantics;
- closed seam-binding transport from stack plan to worker;
- graph evidence tied to exact candidate identity;
- correction of tied top-logprob and pristine-publication edge cases.

Merged changes: [#47](https://github.com/latent-to/cacheon/pull/47), [#48](https://github.com/latent-to/cacheon/pull/48), [#49](https://github.com/latent-to/cacheon/pull/49), [#50](https://github.com/latent-to/cacheon/pull/50), and [#51](https://github.com/latent-to/cacheon/pull/51).

## 2026-07-12 — Bounded discovery

Cross-cutting source-patch proposals gained a separate discovery ABI. The validator owns patch policy, build profile, overlay construction, scheduler-only activation, and grade identity. Discovery evidence cannot masquerade as registered graph evidence or mint a permanent whole-engine reward family.

A structural RTX exercise completed the joined discovery B/C/B′/T path. The no-op proposal correctly received an economic failure rather than a synthetic win.

Change: [PR #52](https://github.com/latent-to/cacheon/pull/52).

## 2026-07-13 — Finalized SQLite intake

The production validator loop moved to `FinalizedIntakeStore` in SQLite. It now persists finalized chain order, HTTPS fetch/re-hash, copy disposition, immutable worker publication, screens, qualification reservations and attempts, retained evidence, retries, and restart state.

The old shell/CPU report evaluator and JSON settlement path were removed from production authority. Infrastructure and cohort failures became typed per-reservation `NO_DECISION` outcomes.

A live testnet reveal was fetched, published, and reopened across restart without duplicate work. This intake proof did not claim a current-path performance crown.

Change: [PR #53](https://github.com/latent-to/cacheon/pull/53).

## 2026-07-13 — Transactional settlement and emissions

Settlement was separated from qualification grading. It reopens retained qualification evidence and applies a validator-owned transition to the current incumbent.

The economics layer added:

- relative per-target credit instead of argmax winner-take-all;
- reciprocal age decay;
- atomic-target suppression of overlapping singleton families;
- bounded non-renewable discovery rewards;
- reopen-before-project and stale-projection rejection;
- intent, pending, held, released, and confirmed weight-journal states;
- explicit rejection of “SDK returned” as publication confirmation;
- a requirement for genuine current-schema crown evidence before a real extrinsic.

Numeric policy parameters remained uncalibrated by design.

Change: [PR #54](https://github.com/latent-to/cacheon/pull/54).

## 2026-07-13 — Reproduction, integration, and signed release closure

The closure added the missing product authorities:

- a closed `ArenaServiceRegistry` with fixed non-crown screens;
- first-`PASS` `reproduction_pending` state;
- settlement requiring a second independent `PASS` for the exact identity;
- conservative use of the lower reproduced speedup;
- durable crown reopening and transactional stack state;
- integration review that binds retained evidence, byte-preserved selected source, surrounding packaging, immutable attribution, and a reviewed Git commit;
- integrated-only release manifests;
- sealed model and native identities;
- deterministic source and wheel artifacts;
- SPDX SBOM and in-toto provenance;
- external Ed25519 trust anchors;
- immutable signed release publication;
- reproducible Registry-v2 image identity;
- host digest, label, seccomp, mount, and inspect authorization;
- fail-closed serving under `OPTIMA_RELEASE_REQUIRED`;
- all-rank active/routed/completed release smoke receipts.

The exact final structural proof used a synthetic 100-ppm candidate to exercise two-pass crown, restart, and a live `submitted=false` weight dry-run. It did not claim a real acceleration result. The real joined double-build and TP8 serve proof was not run.

Changes: [PR #55](https://github.com/latent-to/cacheon/pull/55) and [PR #56](https://github.com/latent-to/cacheon/pull/56). The latter only clarified that the value embedded in the container context is an Ed25519 public verification key.

## 2026-07-13 — CuTe admission and scheduler-role loading

The initial B300 intake work added three boundaries that were later included in
PR #59:

- `804147c4` admitted
  CuTe's function-object `cute.compile(...)` idiom under conservative
  whole-module alias analysis, while banning `builtins` import and failing
  closed on alias shadowing, rebinding, or local `cutlass` vendoring.
- `3cff4731` moved
  candidate-bundle loading to a positive scheduler-process gate. Import-hook
  activation now stays pass-through, preventing miner module code from running
  in SGLang's detokenizer/output-path and other non-scheduler children and
  restoring exact active-receipt coverage at `tp_size`.
- `5325468a` extracted the tracing-JIT exception into the validator-owned,
  stdlib-only `DSL_JIT_ENTRYPOINTS` table. The shared call-shape policy also
  refuses a string/bytes literal as the first positional argument; CuTe remains
  the sole admitted entry and Triton requires no explicit `.compile` row.

These changes were prerequisites for the later sealed direct-artifact path; they did not
by themselves make runtime CuTe host objects an acceptable crown surface.

## 2026-07-13 to 2026-07-15 — Sealed direct artifacts

The artifact boundary gained a closed provider registry and the authoritative
`cutlass.cute.cubin.v1` provider. Candidate factory code executes in a GPU-hidden,
no-network prebuild child. The retained product is an exact-architecture CUDA ELF CUBIN
plus validator-reconstructed call, launch, resource, specialization, and lifecycle
authority.

The runtime binds complete physical kernel inventories to logical exports by ordinal
after rank-local CUDA and group initialization. Validator-owned materialization covers
pointer/scalar parameters, packed fields, CUDA 13 TMA descriptors, distinct CUTLASS and
CuTe FastDivmod representations, launch geometry, optional clusters, streams, generated
resources, and teardown. PTX, host objects, shared libraries, candidate launch callbacks,
and runtime JIT inputs are outside the provider contract.

Ledger-attested B300 TP4 blockscore runs reached complete 4/4 load, invocation, and
completion with zero fallback; the pod-local receipts were not retained. The subsequent
TP4 B200 joined qualification retained primary and reproduction PASSes at 1.0561× and
1.0487× on the charged basis. Settlement used 1.0487×, committed a generation-1 crown,
and reopened it after restart. The run proves the blockscore direct-CUBIN lane, not
arbitrary direct-artifact collectives.

Changes: [PR #59](https://github.com/latent-to/cacheon/pull/59), including
`368727f6` through `93dfb182` and their validation follow-ups.

## 2026-07-15 — Chain reconciliation and executable-surface trim

Finalized-height refresh now precedes settlement leasing and commit. Raw reveal history
decoding is bounded and requires event/storage equality. Failed extrinsics report
`submitted=false`, and a retained pending weight intent can be reopened across later
heads without resubmission.

The superseded local JSON commit/reveal simulator and direct evaluator/benchmark surface
were removed. The public local contribution loop is `scan` plus `verify`; complete-engine
speed and quality remain registered-arena authorities.

Changes: `5afdf2e2`, `1796fc24`, `a52712ef`, `e9ac2f30`, `030d619c`, and
`28d8ff29`; the trim reached `main` through PR #58.

## 2026-07-16 — Shared strict validation and core wheel smoke

The `refactor/strict-validation-kernel` branch centralizes canonical digest, integer,
identifier, exact-field, duplicate-key, CUDA-driver-integer, and environment-flag checks
in `optima/_strict.py`. Calling modules retain their own exception types, grammars, and
bounds. Most authority digests now reject the all-zero placeholder; digest computation is
unchanged.

A follow-up adds `_strict.py` to the serving-wheel allowlist and exercises selected core
imports from a clean extracted wheel. The broader manifest/direct-artifact runtime closure
is still absent, so the release gate must expand from that core smoke to every
manifest-reachable serving entrypoint before the branch is release-ready.

## Corrections and retractions

### MXFP4 throughput claim retracted — 2026-06-07

An early sm120/MXFP4 example was described as a submitted throughput improvement before it had passed Optima's scored end-to-end gate. Commit [`c83c5a9`](https://github.com/latent-to/cacheon/commit/c83c5a9) removed the slot/example and the unsupported throughput language while retaining the useful generic MoE and cosine-correctness machinery.

### Weight submission truthfulness — 2026-07

The earlier chain adapter could report `submitted=true` even when the extrinsic failed, which could suppress a necessary retry. Commit [`20b8d674`](https://github.com/latent-to/cacheon/commit/20b8d674) corrected the immediate bug. The final economics design institutionalized the lesson: submission intent, release, and confirmation are separate persisted states, and an SDK return value alone never confirms weights.

### Historical crown language narrowed — 2026-07-13

The July 7 and July 8 fused-epilogue results remain real, reproduced measurements under their contemporary referee. They are no longer described as proof of the final hardened path. Current authority requires finalized SQLite intake, isolated B/C/B′/T evidence, independent reproduction, transactional settlement, and current-schema crown reopening.

### Slot count corrected — 2026-07-13

The live slot catalog contains 11 slots, not 10. `attention.msa_prefill_block_score` is the additional current slot. The executable [`slots.py`](https://github.com/latent-to/cacheon/blob/main/optima/slots.py) catalog is authoritative.

## Hardening merge index

| Capability | Pull requests | Merge commits |
|---|---|---|
| Product contract, typed waist, target catalog, graph proof | #31–#39 | `a80925c8` through `c09a6896` |
| Vendor provenance | #40 | `f0745c99` |
| Stack identity and engine-tree materialization | #41–#42 | `f6c56dd2`, `9c2d4773` |
| OCI host, native publication, streaming executor | #43–#45 | `6e4b744c`, `30649398`, `38ddaed6` |
| Marginal runtime | #46 | `17ecdeb5` |
| Qualification, pristine T, causal evidence, seam transport | #47–#51 | `9a34b68a` through `1b8a3556` |
| Bounded discovery | #52 | `3606a95c` |
| Finalized intake and production wiring | #53 | `b5825913` |
| Transactional settlement and emissions | #54 | `084fb8d7` |
| Arena, independent reproduction, integration, release | #55 | `b218190d` |
| Public-key CodeQL clarification | #56 | `990721bc` |
| Direct CUBIN provider, B300/B200 intake fixes, chain reconciliation | #59 | `93fe837b` |
| Legacy simulator and evaluator removal | #58 (including the stacked trim) | `6d7e92ef` |
| Shared strict validation, runtime/intake closure, adaptive qualification | #61 | `67bbb57f` |
| Finite-debt accounting, launch activation, native/audit runtime closure | #62 | `43d9be23` |
| Resident crossover, calibration continuation, and routing-only screen | #64 | `4e4cf3bb` |
| All-uncrowned burn bootstrap | #65 | `b79ddbe9` |
| Launch-cohort cap, screen swappability, stable-UID catch-up, watch loop | — | `d69ba1b8`, `27806743`, `7d56a236` |
| Public artifact cleanup and minimal ignore policy | — | `115e09ce`, `4c80a286` |

## Current interpretation

The history resolves into four layers:

1. early mechanism work established slots, seams, quality gates, collectives, and live chain transport;
2. historical B300 runs demonstrated that shallow, deep, and MSA-prefill optimizations could produce real speedups;
3. the July hardening rebuilt authority so those ideas can be evaluated marginally,
   reproduced independently, and settled transactionally; and
4. the direct-artifact provider exercised that authority for blockscore while preserving a
   separate integration and release gate.

Only retained current-schema qualification defines crown authority, and a crown still
does not define release authority. See [State of record](../reference/state-of-record.md)
for the remaining validation debt and [Architecture overview](../architecture/overview.md)
for the stable model.
