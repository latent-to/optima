# Arena service

An arena service is the trusted bridge between an immutable proposal publication and
crownable qualification. It binds **what is being measured**, **where it may run**, and
**how scarce evaluator capacity is allocated**.

It is validator-owned policy. A submission cannot name a Python provider, shell command,
model path, workload, or qualification factory.

## Manifest identity

`ArenaServiceManifest` contains five content-bound parts:

| Part | Bound facts |
|---|---|
| Runtime | Arena ID; runtime, base engine, validator overlay, worker distribution, model revision/manifest/content, architecture, topology, GPU count, tensor-parallel size |
| Workload | Prompt corpus, seed scheme, weighted regimes, exact serving shapes |
| Capacity | Queue depth/age, concurrent screens and qualifications, cohort size, retry budgets |
| Screens | Exact ordered non-crownable screen policy |
| Provider and qualification policy | Reviewed provider digest and qualification-policy digest |

The manifest digest is the service identity. A provider object is accepted only when its
declared digest matches the manifest and it implements the typed interface.

The digest comparison is a closed-configuration check, not remote attestation. Deployment
review must establish that the in-process object really corresponds to the declared
provider bytes and that those bytes were installed on the control-plane host. A provider
can state any digest; the service prevents candidate-selected substitution, not a
malicious validator operator.

Treat the service manifest as immutable for the life of evidence derived from it. A model
refresh, SGLang update, worker image change, prompt-corpus change, topology change, or
capacity/policy edit creates a new service digest. Do not silently “update” an arena ID
in place and compare results across the old and new digest.

## Workload mixture

A workload regime contains an exact phase, weight in integer parts per million, and one
or more `(input_tokens, output_tokens, batch_size, samples)` shapes. A valid mixture:

- totals exactly 1,000,000 ppm;
- contains both `decode` and `long_prefill` phases;
- binds a prompt-corpus digest; and
- binds the prompt seed scheme.

This prevents a candidate, operator typo, or later configuration drift from quietly
changing the workload represented by a result. It does not prove that the registered
mixture perfectly predicts every production deployment; workload representativeness
remains a governance and measurement responsibility.

## Non-crownable screens

Every service runs these stages in this order:

1. `static`
2. `build`
3. `abi`
4. `graph`
5. `abbreviated_serving`

Each stage has a timeout and emits typed evidence with one of three grades: `pass`,
`fail`, or `no_decision`. The derived promotion result is:

- `promote` only after all five stages pass;
- `reject` after a candidate-caused failure;
- `retry` for a bounded inconclusive attempt; or
- `hold` after capacity or retry policy is exhausted.

Screens are explicitly marked `crownable: false`. A fast abbreviated-serving screen is
an admission signal, not economic evidence, and cannot update the evaluation stack.

The abbreviated-serving stage can use a long-lived resident screen lane. It keeps stock
resident, swaps candidate bundles into a reviewed seam, recaptures graphs, and binds every
batch to the current swap generation. Stock reads bracket candidates and act as canaries:

```text
B0 → swap(candidate-1) → C1 → swap(stock) → B1 → ...
```

The queue may reuse a preceding stock bracket under policy and withdraws the lane when a
stock canary leaves tolerance. Screening remains serialized and routing-only. Its rates
cannot be copied into a qualification witness, and its PASS cannot crown.

Sealed AOT device artifacts, dependency patches, native-source rebuilds, and setup hooks
are deliberately non-swappable. The bridge records an explicit waiver and routes that
work to full qualification. A waiver is not a performance PASS; qualification decides
the candidate. Swap nonces, generations, verb deadlines, canary tolerance, and engine
lifetime are all bounded. Replay or out-of-order control fails closed.

A screen receipt is a canonical prefix. The provider cannot skip `build`, run `graph`
before `abi`, or return a later-stage result after an earlier failure. Each result binds
the stage, grade, evidence digest, and elapsed milliseconds. If the reported elapsed time
exceeds the registered stage timeout, the service converts it to `no_decision` even if
the provider labelled it a pass.

The operator should retain the bytes addressed by every stage evidence digest. The
receipt proves which bytes were named; it does not make missing bytes replayable.

## Admission and capacity

The controller supplies a durable queue snapshot. The service returns:

- `admit` when screen or qualification capacity is available;
- `queue` when work should wait without losing finalized priority; or
- `hold` when queue depth, queue age, or cohort size has crossed an admission bound.

Retry policy is applied after `NO_DECISION` evidence through the separate
`retry_disposition()` promotion decision; it is not an input to admission.

Capacity policy must be chosen from measured operational budgets. It should include
expected queue age, stage cost, cohort size, retries, and crown latency. Network fetch
speed and evaluator arrival order must not replace finalized chain order.

The decisions have deliberately different queue effects:

| Decision | Controller behavior | Priority consequence |
|---|---|---|
| `admit` | Starts one screen or admitted cohort | Finalized order is retained |
| `queue` | Stops selecting further work for this pass | Work stays in its current durable lane |
| `hold` | Moves affected work to an operator-visible hold | Work cannot advance until reviewed |

At screen admission, reaching maximum queue depth or maximum queue age holds the current
reservation; reaching active-screen capacity queues it. At qualification admission, an
oversized cohort or exceeded queue bound holds it, while insufficient active
qualification capacity queues it. These distinctions prevent temporary saturation from
becoming a miner loss while ensuring an indefinitely overloaded service fails visibly.
The controller caps a promoted cohort to the lower of the intake policy and registered
service capacity; it never promotes an oversized cohort and asks the provider to cope.

A hold is durable across wall-clock time, restart, and transient capacity changes. The
store exposes two explicit reservation-level dispositions: `release_hold`, with an
audited reason, and `expire`, which is admitted only after the configured minimum age.
Separately, eligible unresolved rows—including ordinary `held` and `no_decision`
rows—expire automatically when the finalized arrival/progress-block SLA is reached.
Active `fetching`, `screening`, and `qualifying` work is not aged out underneath an
in-flight operation, and the dedicated schema-3 migration hold requires its explicit
archive path. This row-level SLA is not a generic wall-clock TTL or a typed transition
that retires an entire arena. Settlement lease expiry and a discovery claim's configured
lifetime remain different state machines. Operators must monitor held rows and preserve
their evidence; deleting them is not a supported recovery path.

## Provider interface

Reviewed deployment code implements two operations:

```python
class ArenaServiceProvider:
    provider_digest: str

    def run_screen(self, manifest, stage, candidate): ...
    def build_qualification(self, request, state=None): ...
```

`run_screen` must return the requested stage and typed evidence. Abbreviated serving may
delegate to the reviewed resident bridge, but that bridge and its `resident_swap` seam
remain deployment capabilities rather than miner-selected plugins. The service always passes
the controller-supplied `state` object to `build_qualification`; providers that do not need
it must still accept the optional parameter. The returned work must preserve the exact
promoted reservation order and qualification-policy digest while supplying a plan factory,
isolated executor, post-commit entropy provider, hidden judge, and absolute deadline.

The provider runs in the trusted deployment process. It is not discovered from package
entry points or submission metadata.

`build_qualification` is the most security-sensitive join. Its returned
`ArenaQualificationWork` must contain:

| Field | Operational responsibility |
|---|---|
| `factory` | Rebuilds the frozen registered/discovery causal plan and preserves the exact promoted reservation order |
| `executor` | Runs resident crossover, the audit-only role, and pristine reference under reviewed OCI authority |
| `entropy_provider` | Reveals post-commit selection entropy without exposing the private selection secret early |
| `hidden_judge` | Performs the registered hidden task work outside candidate authority |
| `deadline` | Absolute controller deadline, not a candidate-supplied timeout |
| `qualification_policy_digest` | Must equal the service manifest's registered policy digest |

The controller independently checks the factory reservations, initializes or reopens the
exact incumbent evaluation stack, marks every cohort member `qualifying`, runs and reopens
the attempt, and transactionally applies its per-reservation outcomes. A provider cannot
return a different cohort, reorder finalized candidates, or replace the policy after
screening.

## Deployment composition

Keep construction in a reviewed composition root—not in bundle parsing or a general
plugin loader:

```text
deployment configuration
├── reviewed ArenaServiceManifest
├── installed provider implementation + provider digest
├── sealed model/runtime/worker identities
├── private entropy and hidden-judge authorities
├── OCI executor and evidence roots
└── ArenaServiceRegistry (sorted, closed, exact arena IDs)
        └── run_validator(..., arena_registry=registry, arena_id=<exact ID>)
```

Commission the composition in four stages:

1. Reopen every path-free identity from its content-addressed artifact or deployment
   record; do not populate digests from human labels.
2. Run each non-crown screen against known faithful, known broken, timeout, and provider-
   error fixtures, retaining evidence for all grades.
3. Run an isolated qualification control that demonstrates resident B/C/B′ and optional
   C′/B″ ordering, serialized physical lanes, audit/T stage exits, cleanup, evidence
   reopen, and `PASS`/`FAIL`/`NO_DECISION` separation.
4. Inject the registry into a one-pass controller, then test restart during primary
   screen, reproduction screen, and qualification before enabling daemon mode.

An unexpected provider exception is not a candidate `FAIL`. The pass-level controller
contains it, and restart recovery preserves or returns the reservation to an appropriate
retry/hold lane. Providers should translate anticipated infrastructure conditions into
typed `no_decision` evidence; they must not catch attributable candidate violations and
mislabel them as infrastructure.

## Registry and CLI boundary

`ArenaServiceRegistry` is a closed, nonempty, deterministically ordered collection of
services. It rejects duplicate arena IDs and resolves only an exact registered ID.

The CLI parser exposes `--arena-id`, but the standalone console entry point has no way to
construct the registry. Calling:

```bash
optima chain-validate --arena-id example ...
```

without a Python caller injecting `ArenaServiceRegistry` fails closed. Use
`--intake-only` for the public standalone workflow. A production operator must supply a
reviewed provider and registry through deployment code.

The deployment API also accepts `retained_only=True` on `run_pass()` and
`run_validator()`. This mode requires an existing finalized cursor, reads a fresh
finalized head, and processes the already-durable queue without rereading or advancing
reveal history. It conflicts with `intake_only` and still requires the reviewed arena
authority needed by the work. It is intentionally not exposed as a public
`chain-validate --retained-only` flag.

The repository also does not ship a registry serialization format that turns arbitrary
configuration into trusted executable authority. That omission is deliberate: the
provider object, secrets, executor, and hidden judge are deployment capabilities, while
the manifest is their public identity. Storing the manifest does not recreate those
capabilities.

## Operating signals

At minimum, export and alert on:

- queue depth and oldest finalized age;
- active and configured screen/qualification capacity;
- stage latency and grade counts by service digest;
- resident swap generation, stock-canary drift, waiver reason, and engine recycle count;
- primary versus reproduction screen attempts;
- retry budget remaining and hold reasons;
- provider exceptions and controller restarts; and
- evidence publication/reopen failures.

Always label metrics with the full service digest, runtime/model identity, and lane. An
arena ID alone is not enough to compare measurements after a deployment change.

## Nonclaims

- The repository does not ship a production provider for any hardware fleet.
- Registry typing does not attest that an operator chose representative prompts or
  sufficient capacity.
- Screens do not prove a win, grant a crown, or authorize a release.
- A resident-screen waiver does not grade speed; it routes non-swappable work to
  authoritative qualification.
- Screen and qualification receipts do not make optional sampled in-engine audit a
  universal authority; only evidence required by the registered policy contributes to
  the decision.
- Multiple registered arenas do not imply their measurements can be mixed; every result
  remains bound to one arena and stack identity.

Next: [Authoritative qualification](qualification.md).

## Source anchors

- [Arena service types and registry](https://github.com/latent-to/cacheon/blob/main/optima/arena_service.py)
- [Resident screen bridge](https://github.com/latent-to/cacheon/blob/main/optima/eval/resident_screen_lane.py)
- [Resident screen queue](https://github.com/latent-to/cacheon/blob/main/optima/eval/resident_queue.py)
- [Resident OCI session](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_resident_session.py)
- [Arena service tests](https://github.com/latent-to/cacheon/blob/main/tests/test_arena_service.py)
- [Qualification intake projection](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification_intake.py)
