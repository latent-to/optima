# Evaluation pipeline

Optima has two evaluation paths with different authority:

- the **developer path** helps a miner build and debug a proposal;
- the **production referee path** can create retained qualification, crown, settlement, and reward state.

The developer commands intentionally reuse parts of the ABI and evaluation machinery, but their output is not production crown evidence.

## Developer path

```text
slots -> scan -> verify -> chain-package -> host -> chain-submit -> chain-status
```

| Step | Purpose | Authority |
|---|---|---|
| `slots` | Inspect the live slot ABI | Informational |
| `scan` | Run static bundle policy checks | Diagnostic |
| `verify` | Check a single op/block against its trusted reference; collective slots use distributed verification | Diagnostic |
| `chain-package` | Produce the deterministic hosted archive and content identity | Submission preparation |
| HTTPS host | Make the immutable archive available for validator fetch | Transport only |
| `chain-submit` | Commit the proposal through timelock commit-reveal | Chain intake |
| `chain-status` | Inspect submission state | Informational |

Local measurements are useful for iteration. They do not select a production arena
authority, reserve a finalized cohort position, retain authenticated resident
B/C/B′ evidence with conditional C′/B″ reads, perform the registered eager audit
A and pristine-reference T stages, or satisfy independent reproduction.

## Production path at a glance

```mermaid
flowchart TD
    A["Finalized timelock reveal"] --> B["Hardened fetch and immutable publication"]
    B --> C["Registered arena and target resolution"]
    C --> D["Non-crown screens and routing"]
    D -->|"reject"| F["Terminal invalid or attributable failure"]
    D -->|"promote or waive"| Q["Resident adaptive speed<br/>audit, then pristine T"]
    D -->|"infrastructure or ambiguous"| N["NO_DECISION / retry"]
    Q -->|"FAIL"| F
    Q -->|"NO_DECISION"| N
    Q -->|"first PASS"| P["reproduction_pending"]
    P --> R["Independent second qualification"]
    R -->|"attributable FAIL"| F
    R -->|"NO_DECISION"| N
    R -->|"matching PASS"| S["Reopen both evidence roots"]
    S --> G["Same-authority cohort planning"]
    G -->|"selected current registered winner"| T["Transactional settlement and stack update"]
    G -->|"current discovery candidate"| J["DISCOVERY_BOUNTY / no stack transition"]
    G -->|"stale or not selected"| H["HOLD / no stack transition"]
    T --> W["Reward projection and journaled weights"]
    J --> W
    T -. "separate decision" .-> I["Integration review and signed release"]
```

## 1. Finalized intake

Submissions enter through native timelock commit-reveal. The validator acts only on finalized chain order. Finalized block position and commitment identity establish priority; evaluator network arrival does not.

`FinalizedIntakeStore` persists production authority in SQLite. It records finalized observations, fetch state, copy disposition, screen receipts, cohort reservations, qualification attempts, evidence roots, reproduction state, stack transitions, settlement, and weight-publication state.

State transitions are typed and transactional. The validator does not reconstruct production authority from console output, mutable directories, or a legacy JSON ledger.

Principal code: [`chain/intake.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/intake.py) and [`chain/validator_loop.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/validator_loop.py).

## 2. Hardened fetch and publication

The submitted URL is transport, not identity. The fetch path:

1. applies the validator's HTTPS and network policy;
2. writes into validator-private bounded storage;
3. recomputes the deterministic bundle content hash;
4. compares it with committed identity;
5. derives the copy/provenance disposition;
6. publishes the complete artifact into an immutable, hash-addressed worker namespace;
7. reopens the publication before any screen or launch consumes it.

Partial downloads, path tricks, changed content, duplicate/malformed archives, and publication mismatches fail closed. Candidate workers receive only immutable publications; they do not fetch miner URLs themselves.

Principal code: [`chain/fetch.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/fetch.py), [`chain/payload.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/payload.py), and [`bundle_hash.py`](https://github.com/latent-to/cacheon/blob/main/optima/bundle_hash.py).

## 3. Arena and target resolution

An `ArenaServiceRegistry` maps a public arena identifier to a closed
`ArenaService`. Its manifest directly binds:

- runtime, base-engine, validator-overlay, worker, model, architecture, GPU,
  and topology identities;
- the decode/long-prefill workload mixture and prompt-seed scheme;
- non-crown screen policy;
- queue depth/age, cohort size, screen/qualification concurrency, and retry policy;
- the qualification-policy digest; and
- the reviewed provider implementation digest.

The target catalog, incumbent and candidate stacks, graph/engine settings,
calibration, reference, evidence, and quality identities are closed later by
the promoted candidate bindings and the provider-created typed qualification
plan. `ArenaService` checks the plan's policy digest and finalized reservation
order; it does not pretend all of that authority is a field of the service
manifest itself.

The proposal is resolved against the exact target catalog snapshot. A registered candidate must match its target members and permitted features. Cross-cutting unregistered work is routed through the discovery lane.

The command-line `chain-validate` loop can perform intake alone. Full production qualification requires the operator to inject a real `ArenaServiceRegistry` and select `--arena-id`; the repository does not manufacture a production arena provider from implicit defaults.

Principal code: [`arena_service.py`](https://github.com/latent-to/cacheon/blob/main/optima/arena_service.py), [`target_catalog.py`](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py), and [`stack_plan.py`](https://github.com/latent-to/cacheon/blob/main/optima/stack_plan.py).

## 4. Non-crown screens and routing

Expensive full-engine qualification is reserved for plausible candidates. The registered
arena applies a fixed sequence of non-crownable screens:

1. static policy and manifest resolution;
2. deterministic source closure and build planning;
3. typed ABI correctness, including distributed verification for collectives;
4. graphs-on capture and dynamic-input replay; and
5. abbreviated serving on a small registered workload, implemented by the
   calibrated resident lane when the contribution is safely hot-swappable.

The resident screen keeps one stock engine alive, swaps candidate bundles into a separate
resident session, recaptures graphs, and evaluates a bounded queue against shared stock
brackets and canaries. Every batch is bound to its swap generation. The screen exists only
to route work: a promising result advances to qualification and a clearly uncompetitive,
stable result may be rejected under the registered screen policy. It cannot create a
qualification PASS, crown, settlement speedup, or reward claim.

Some contribution classes cannot satisfy the hot-swap contract. Direct AOT artifacts,
dependency patches, native rebuilds, and setup hooks receive an explicit screen waiver
and proceed to authoritative qualification. A waiver means “not screenable,” not “screen
passed.” The arena caps a promoted screen cohort by both its registered policy and the
provider's current capacity.

Infrastructure errors and measurements too ambiguous for an attributable rejection remain
retryable rather than being converted into a loss.

Principal code: [`eval/oci_resident_session.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_resident_session.py),
[`eval/resident_queue.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/resident_queue.py),
and [`eval/resident_screen_lane.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/resident_screen_lane.py).

## 5. Cohort authority

At an evaluation boundary, the validator freezes:

- one incumbent `EvaluationStackManifest` digest;
- a finalized chain-ordered set of candidate reservations;
- one registered arena and target-catalog context;
- deterministic candidate order derived from committed authority;
- workload, prompt, seed, role, topology, calibration, and evidence policy.

Each candidate stack is the frozen incumbent with exactly one registered target delta.
Production qualification uses two isolated resident TP lanes while serializing GPU work.
The primary attempt assigns the incumbent and candidate to fixed physical lanes. An
eligible reproduction must bind the exact opposite physical-lane role assignment. The
lane swap is part of independence authority; it is not a scheduler preference.

The routing screen may amortize a stock engine and shared brackets across candidates.
Authoritative qualification does not inherit those measurements. It constructs a fresh,
candidate-specific authority and retains each speed, audit, graph, and T product against
that exact delta.

Principal code: [`eval/qualification_intake.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification_intake.py) and [`stack_plan.py`](https://github.com/latent-to/cacheon/blob/main/optima/stack_plan.py).

## 6. Resident adaptive qualification

The production provider selects qualification policy version 3. Earlier policy encodings
remain readable for historical evidence: version 1 is the legacy fixed B/C/B′ shape and
version 2 adds fixed repeat-read legs. Version 3 binds two resident lanes, stage and total
budgets, adaptive repeat evidence, and the required physical-lane role assignment.

The authoritative work is staged. Speed is decided first; audit and pristine-reference
quality run only after the speed stage remains eligible, apart from an explicitly
registered calibration-observation continuation.

| Arm | Stack | Timed? | Purpose |
|---|---|---:|---|
| B | Exact frozen incumbent, already loaded on the resident baseline lane | Yes | Opening performance read |
| C | Incumbent plus one exact target delta, already loaded on the disjoint resident candidate lane | Yes | Candidate measurement and sealed trajectory |
| B′ | Same loaded incumbent on the resident baseline lane | Yes | Closing bookend and drift detection |
| C′ | Same loaded candidate on the resident candidate lane | Yes | Optional repeat read when the initial result is inside the escalation band |
| B″ | Same loaded incumbent on the resident baseline lane | Yes | Optional repeat bookend paired with C′ |
| A | Candidate in a separate eager, untimed role | No | Registered sampled slot audit and typed host regrade |
| T | Pristine candidate-free reference | No | Teacher-forced semantic quality and hidden tasks |

Native build and timed execution use separate containers. Before a runtime arm
starts, a disposable, no-GPU/no-network prebuild OCI parses the materialized
tree, invokes only registered build patchers, and emits a sealed native
publication. The resident speed lanes mount that reopened
publication read-only. Candidate Python import, engine construction, and
execution occur only in the runtime's positively identified scheduler ranks;
runtime ranks may validate and load native products but may never compile or
repair them. Both stages use read-only roots, bounded mounts and protocols, and
host-owned cleanup; the trusted controller also owns timing.

For a direct-artifact row, prebuild executes the declared compiler factory only
inside a no-egress compiler child and publishes CUBIN rather than a host launcher.
After rank-local CUDA setup, the scheduler worker admits the exact CUBIN, binds its
complete driver-observed ABI to the declarative device plan by ordinal, and
materializes parameters and lifecycle storage in validator code. Qualification
requires per-member `aot_loaded`, `aot_invoked`, and normal `completed` coverage,
with no fallback receipt. See [Sealed direct artifacts](direct-artifacts.md).

The initial B/C/B′ sequence is sufficient when its result is clearly outside the
registered escalation band. Borderline evidence adds C′/B″ under the same frozen
authority. The grader does not add favorable reads after observing an outcome or replace a
failed bookend with a candidate measurement.

When the registered plan requires sampled slot audit, a separate eager, untimed candidate
role emits bounded raw facts. The trusted host grades exact slot × TP-rank/process coverage
through a Torch-free gate and canonicalizes floating-point facts before durable receipt
identity is computed. These facts cannot enter the charged speed roles. A slot or target
without a registered audit requirement cannot acquire audit authority from an incidental
diagnostic receipt.

After the candidate speed and audit lifetimes are destroyed, T grades the sealed
trajectory under a separate pristine lifetime. T never contains the candidate and does
not compete on speed. Hidden reference work, quality policy, and selected prompt identity
are bound into retained evidence.

The host pairs candidate throughput with the B/B′ bookend and applies the registered noise policy. Conceptually:

```text
paired_speedup = candidate_throughput / mean(B, B′)
required_bar   = 1 + max(margin_floor, noise_multiplier × measured_noise)
```

The exact registered policy, not this explanatory formula, is authoritative.

Principal code: [`eval/crossover_runtime.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/crossover_runtime.py),
[`eval/qualification.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification.py),
[`eval/qualification_runner.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification_runner.py),
and [`audit_gate.py`](https://github.com/latent-to/cacheon/blob/main/optima/audit_gate.py).

## 7. Verdict semantics

Qualification has three outcomes:

### `PASS`

The candidate clears the registered speed, quality, graph, evidence, and whole-stack requirements under a stable cohort authority. A first pass is retained as `reproduction_pending`; it does not settle alone.

### `FAIL`

Complete evidence attributes a policy failure to the candidate under a valid authority. Examples include incorrect output, a quality regression, or a stable speed result below the registered bar.

### `NO_DECISION`

The evaluator cannot make a valid attributable decision. Infrastructure failure, missing evidence, baseline drift, broken cohort invariants, or an invalid reference lifetime must not mint either a crown or a loss. The result retains a failure product and retry policy.

The distinction is load-bearing: treating evaluator failure as candidate failure would let infrastructure state rewrite economic truth.

### Worked lifecycle: retry, reproduce, settle

Consider a hypothetical candidate for one registered singleton target:

1. Intake fixes its finalized priority and immutable publication. It clears all five
   non-crown screens.
2. Its first bracket produces valid C work, but B′ drifts outside the frozen noise policy.
   The attempt is `NO_DECISION`. The proposal remains retryable; it has neither lost nor
   passed.
3. A later fresh authority reopens the same candidate identity. B/C/B′ are stable; the
   result falls inside the escalation band, so C′/B″ are collected. The registered audit
   and T products accept the candidate, and it clears the speed bar. The attempt becomes
   the first `PASS` and state becomes `reproduction_pending`.
4. A second independently selected authority repeats the exact reproduction identity,
   swaps the incumbent and candidate physical-lane roles, and also passes. A pass against
   a newer incumbent, different target specification, or the same lane-role assignment
   would not count as this reproduction.
5. Settlement reopens both evidence roots and makes the pair eligible for its
   same-authority cohort. If the candidate is selected as that cohort's current registered
   winner, settlement takes the lower accepted speedup, revalidates the live target
   transition, and atomically updates the evaluation stack; otherwise the pair is held.
6. If crowned, reward projection can now see the active claim. Product integration remains
   a separate review; no proposal bytes have entered a release merely because settlement
   completed.

The values and candidate in this walkthrough are illustrative. The state transitions and
failure semantics are the important part.

## 8. Independent reproduction

Settlement requires a second `PASS` for the exact same core reproduction
identity. `SettlementReproductionIdentity` contains exactly the arena digest,
target ID, selected-delta digest, hotkey, incumbent stack/tree digests, and
candidate stack/tree digests.

The settlement pair applies additional rules around that core identity. The two
qualification rows must match the same contribution, reservation, finalized
priority, manifests, members, and arm, while seven independence fields must all
differ: qualification authority, plan, attempt, report, selection commitment,
selection-secret commitment, and selection evidence. Authority therefore does
**not** belong inside the equal core identity; distinct authority is a separate
pair constraint.

For resident version-3 evidence, the pair must also prove the exact physical-lane role
swap required by the registered plan. Two nominally independent attempts that assign
stock and candidate to the same physical lanes do not satisfy production reproduction.

An attributable second-attempt `FAIL` terminates the proposal. A
`NO_DECISION` follows the registered bounded retry/hold policy. In either case,
the retained first pass cannot update the incumbent by itself.

## 9. Settlement

Settlement reopens both recorded attempt references instead of trusting an in-memory
verdict. The references may live under the same content-addressed store root. It verifies:

- both reports are complete `PASS` results;
- all seven required authority, attempt, report, commitment, and selection digests are
  pairwise distinct across the two passes;
- reproduction identities match exactly;
- the incumbent and target transition are still current;
- target overlap, displacement, and composition remain valid;
- the requested stack update matches the measured candidate.

The planner leases one cohort whose rows share qualification authority and incumbent
state. Stale rows are held, and exactly one current registered winner is selected across
the remaining rows, including rows whose targets do not overlap. Other current pairs are
held as `conflict_lost` or `incumbent_advanced`; a stale pair is held as
`stale_incumbent`.

For the selected winner, the conservative settled speedup is the lower accepted speedup
from the two passes. The stack transition and settlement evidence are committed
transactionally; a partial write cannot expose a half-updated incumbent.

Principal code: [`settlement.py`](https://github.com/latent-to/cacheon/blob/main/optima/settlement.py).

## 10. Incentive state and weight publication

Settlement and weight publication are separate state machines. The repository retains two
explicitly fenced incentive paths:

- **Legacy V1** projects active standing and discovery claims through `set-weights`. During
  an all-uncrowned bootstrap, an operator may provide a registered burn hotkey; the burn
  projection becomes invalid as soon as a crown, claim, or active V2 composition exists.
  `--watch` operates the same journaled reconciler continuously with bounded retry rules.
- **V2 finite debt** converts exact registered-CROWN and reviewed-discovery events into
  fixed-point debt under a content-addressed campaign/composition policy. The implemented
  launch path activates one immutable campaign at 100% sizing. Activation is wallet-free
  and binds finalized chain cursor, retained arena/stack/catalog/families, membership,
  reserve, audit controls, and independent approval. `set-debt-weights` publishes gapless
  policy boundaries and debits only after exact readback confirmation.

Reviewed discovery promotion into a registered target still fails closed because the
durable promotion transport and fresh requalification/CROWN linkage are not implemented.
The current V2 implementation can retain review-pending work and issue bounded
`bounty_only` debt; prose must not infer the missing promotion branch.

Both publishers persist intent and later readback states. An SDK return value does not
prove inclusion, and neither publisher may debit or advance economic authority from an
unconfirmed vector.

Principal code: [`chain/weights.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/weights.py),
[`finite_debt.py`](https://github.com/latent-to/cacheon/blob/main/optima/finite_debt.py),
[`incentive_composition.py`](https://github.com/latent-to/cacheon/blob/main/optima/incentive_composition.py),
[`chain/incentive_activation.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/incentive_activation.py),
and [`chain/debt_publication.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/debt_publication.py).
See the [emissions policy](../reference/emissions-policy.md).

## 11. Integration and release

A settled crown may enter integration review, but serving remains a separate state machine. Reviewed source is promoted to an integrated contribution, assembled into an `EngineReleaseManifest`, rematerialized, signed, published, and verified without chain dependencies.

See [Release architecture](releases.md). The dated
[State of record](../reference/state-of-record.md) tracks implementation and validation
limits.

## Operational handoff checklist

Before treating an attempt as production authority, an operator or reviewer should be
able to reopen, rather than merely observe, each handoff:

- finalized chain position, commitment, fetched content identity, and immutable worker
  publication;
- registered arena, target catalog, incumbent manifest, candidate transition, and screen
  receipt;
- resident lane identities and a `ResidentSpeedWitness` containing B/C/B′ plus
  conditional C′/B″ read rows, plus the retained
  graph/quality/pristine-T references and witnesses; raw B/C/B′ session/device frames are
  validated in-run but are not serialized into the attempt;
- frozen calibration and the exact policy that maps the retained witness/evidence products
  to the verdict;
- the seven digest-distinctness fields across primary and reproduction over one
  reproduction identity;
- transactional settlement events and resulting evaluation-stack digest;
- reward projection plus publication intent/status/chronology records; later readback
  vectors must be re-observed because the journal does not serialize them; and
- if shipping is proposed, the separate integration records and signed release identity.

Console output, a green local benchmark, one `PASS`, or a successful chain SDK return is
not a substitute for the corresponding reopenable product.
