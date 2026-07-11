# Optima product contract

Status: architectural authority for the referee-hardening split. Changes to these
invariants require an explicit product decision; they are not incidental refactors.

## Product definition

**Optima Engine is a chain-independent, open inference-acceleration distribution.**
It pins an upstream runtime (initially SGLang) and replaces or extends its
performance-critical data plane with a canonical stack of accepted optimizations.

The subnet/referee is a permissionless proposal, measurement, attribution, and reward
market for those optimizations. It is not part of the deployed engine. The managed
service runs signed Optima Engine releases, never live miner bundles or live chain state.

## Invariants

1. Optima Engine builds, tests, releases, and serves without Bittensor.
2. Miner bundles are proposals, never permanent production dependencies.
3. The validator owns and assembles both the hill-climbing stack and the reviewed engine
   release stack; miners never package either incumbent.
4. Rewards attach to the smallest validator-controlled attributable delta: a slot,
   registered atomic target, or explicitly reviewed discovery contribution.
5. Every candidate executes as a complete isolated engine; execution isolation does not
   determine economic identity.
6. A crown establishes measured contribution and attribution. Shipping is a separate
   security, licensing, provenance, maintainability, and integration decision.
7. Normal miner submissions target the inference data plane, not API serving,
   authentication, tokenization, request admission, autoscaling, or service operations.
8. Cross-cutting work that does not fit a registered target uses a fenced discovery lane.
   A discovery win is promoted into a component/atomic boundary or receives a bounded
   reviewed bounty; it does not automatically mint a permanent equal-emission fork title.
9. Accepted contributions become ordinary Optima-owned source, tests, wheels, containers,
   and signed releases, retaining immutable author/contribution attribution.
10. SGLang is a pinned upstream substrate. Runtime neutrality is optional; canonical
    stack composition within the chosen runtime is mandatory.

## Four distinct objects

| Object | Meaning | Authority |
|---|---|---|
| Proposal | Miner-supplied target-scoped delta or discovery prototype | Hostile/untrusted input |
| Crown | Evidence that the proposal improved one registered arena/target | External referee plus retained replayable evidence |
| Integrated contribution | Reviewed, normalized Optima source tied to an immutable contribution ID | Optima source control and release review |
| Engine release | Pinned upstream runtime plus canonical accepted stack | Signed, chain-independent build artifact |

No object may be silently substituted for another. In particular, a winning bundle is
not itself the production engine, and OCI execution does not imply a whole-engine crown.

## Two stacks and one trusted reference

The economic hill climb and the shipped product are deliberately different objects:

- `EvaluationStackManifest` binds the upstream runtime, arena, target catalog, and current
  winning proposal artifact for each active region. It may contain unintegrated hostile
  contributions and is used only inside the isolated referee.
- `EngineReleaseManifest` contains only reviewed, normalized Optima source and is the
  chain-independent product consumed by serving deployments.
- `ReferenceManifest` is a pristine validator-owned semantic reference used only for
  untimed quality grading. It contains no miner proposal and does not compete on speed.

All manifests and materialized engines are content-addressed. A crown may update the
evaluation incumbent after regression; only a separate shipping decision updates the
engine release.

## Marginal evaluation

For an authoritative finalist bracket:

- B and B' are independently launched from the exact incumbent
  `EvaluationStackManifest` digest;
- each C is the same manifest with exactly one registered slot or atomic target replaced;
- the validator supplies every other incumbent, adapter, fallback, workload, and engine
  configuration;
- only the submitted delta enters miner copy/provenance identity;
- after C is destroyed, an untimed pristine T arm teacher-forces the sealed B/C/B'
  trajectories and performs hidden quality work; an untrusted incumbent B' is never the
  grading oracle;
- a win updates the evaluation stack transactionally only after a whole-stack regression.

These are logical evidence arms, not a requirement to cold-load B, C, B', and T for every
chain reveal. The operational evaluator may seal a cohort `C1..Ck` against one frozen
incumbent and amortize bookends/reference work, provided every C still has one exact delta
and the retained evidence preserves its comparison authority.

An atomic target owns an explicit validator-registered set of semantic regions. Target
overlap, displacement, compatibility, rollback, and emission activation must be explicit;
manifest order or miner packaging must never define them.

## Evaluation service contract

The subnet must remain responsive under continuous submissions and model-sized arenas.
Evaluation is therefore staged:

1. Intake, static policy, deduplication, deterministic build, ABI correctness, graph replay,
   and narrow microbench screens reject invalid work before full model launches.
2. A non-crownable serving screen uses a small registered workload and may reuse trusted
   incumbent/reference infrastructure. It advances only clearly promising candidates;
   noisy or marginal results receive more evidence rather than a false terminal loss.
3. At an epoch boundary the validator freezes one incumbent stack digest and a chain-ordered
   cohort. Full candidate engines remain disposable, but B/B' bookends are amortized over
   `C1..Ck`; candidate order is derived from post-commit entropy. Drift beyond the registered
   envelope makes the cohort `NO_DECISION`.
4. Only speed finalists receive full raw qualification. After all relevant C engines are
   destroyed, one pristine T lifetime can batch teacher forcing and hidden tasks for their
   sealed trajectories.
5. Only a finalist that clears the authoritative bracket, quality, whole-stack regression,
   and independent reproduction can update the evaluation incumbent.

Where an arena uses TP4 on an eight-GPU validator, screens and paired measurements should
use both four-GPU halves concurrently when the registered topology permits, followed by a
role/half swap for finalist bias control. This is an arena scheduling policy, not a change
to the semantic evidence contract.

Every arena must publish measured budgets for queue age, candidates per cohort, GPU-hours
per stage, retry limits, and expected crown latency. Finalized chain priority must survive
cohorting; admission/rate limits may bound spam but may not let network speed or evaluator
arrival order rewrite priority.

## Crown versus ship

After a crown, integration must independently establish:

- reproducibility against the crowned evaluation stack and current engine release;
- correctness, security, and maintained fallbacks;
- license and provenance acceptability;
- compatibility with active stack entries and pinned SGLang;
- normalized Optima-owned source and tests;
- release notes and immutable contributor attribution.

Emission may follow the crown policy, but production deployment follows the ship policy.
The service consumes only signed engine releases.

## In-scope data plane

Kernels and quantized GEMMs; attention algorithms; MoE execution; collectives and
compute/communication overlap; KV-cache layouts and operations; graph plans; fused blocks;
model-specific execution strategies; and bounded scheduling-adjacent execution changes
whose performance value requires them.

The upstream/service control plane remains responsible for HTTP/API behavior,
authentication, tokenization, request admission, fleet orchestration, autoscaling,
observability, and operational lifecycle management.

## Architectural acceptance test

The product boundary is correct only if all of the following are true:

1. Removing chain access and miner hosting does not prevent rebuilding or serving the
   latest signed Optima Engine release.
2. A new component can be evaluated as one marginal substitution over the current stack
   without copying or repackaging other miners' work.
3. The trusted controller never imports candidate Python/native code; all three arms are
   complete isolated engines under host-owned timing, while a separate pristine reference
   owns untimed quality authority.
4. A whole-system prototype can demonstrate a novel win without acquiring a duplicate
   permanent whole-engine reward title.
5. Every active production component resolves to reviewed Optima source and an immutable
   contribution record, not a mutable URL, hotkey, or chain row.
6. Updating the untrusted evaluation incumbent never mutates a signed engine release, and
   shipping a reviewed contribution never depends on chain availability.
