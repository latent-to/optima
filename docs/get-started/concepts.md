# What Optima is

Optima is an open inference-acceleration system built around a pinned SGLang runtime.
It uses a permissionless market to discover GPU optimizations, a hostile-code referee
to measure them, and a separate integration and release process to turn selected work
into a product.

The chain is useful for proposal ordering, attribution, and rewards. It is not a runtime
dependency of Optima Engine.

## A useful mental model

Think of Optima as a compiler contest whose winning patches must pass through a product
release process:

```text
miner source       validator competition                 product work
------------       ---------------------                 ------------
proposal      ->   immutable candidate -> crown     ->   reviewed integration
                         |                  |                    |
                         | measurement      | economic record    | signed release
                         v                  v                    v
                    retained evidence   emissions policy    Optima Engine
```

The arrows are deliberately not automatic. A validator can prove that one exact source
delta is faster without deciding that the source is maintainable or safe to ship. A
maintainer can integrate the useful idea while preserving the crowned selected-payload
bytes and still wait for a later release. A serving operator can verify and run that
release without consulting the chain.

This separation answers three questions that otherwise get mixed together:

- **Who arrived first?** Finalized chain ordering and durable intake answer this.
- **What was actually faster and faithful?** Validator-owned qualification evidence
  answers this.
- **What may run in production?** Review, integration, signing, and release verification
  answer this.

## Two systems, three operational surfaces

At the highest level, Optima separates the chain-independent product from the
market that improves it. Inside the market system, the subnet control plane and
the hostile-code referee remain distinct operational surfaces.

### Optima Engine

Optima Engine is the chain-independent release contract for serving deployments. A
completed release must bind a pinned upstream runtime, reviewed Optima source, exact
model identity, native artifacts, a serving specification, provenance, an SBOM, and a
signature rooted in an external Ed25519 public key. The current revision implements
construction and verification primitives but does not claim a completed production
release.

The release contract excludes miner URLs, wallets, chain state, and the live evaluation
incumbent from the serving dependency graph.

### The subnet and referee

The subnet is a market for proposals. The referee is the measurement system that:

1. preserves finalized proposal priority;
2. fetches and republishes hostile bytes immutably;
3. screens invalid or unpromising work without creating economic authority;
4. evaluates finalists as one exact marginal substitution;
5. retains and reopens the evidence;
6. requires an independent reproduction before settlement; and
7. projects rewards for active, verifiable contributions.

The referee may update the untrusted evaluation incumbent. It cannot update a signed
serving release.

## The four objects

| Object | Trust level | What it establishes |
|---|---|---|
| **Proposal** | Hostile input | A miner asks the validator to evaluate one target-scoped delta or discovery prototype. |
| **Crown** | Retained measurement evidence | Two independent qualifications show that the same delta improved one registered arena and target. |
| **Integrated contribution** | Reviewed Optima source | Maintainers preserve the crowned selected-payload bytes while approving provenance, tests, fallbacks, compatibility, and surrounding packaging. |
| **Engine release** | Signed deployment artifact | A reproducible product assembles reviewed contributions and a sealed model for serving. |

No transition is implicit. In particular, a crown is not permission to run miner source
in production.

## A proposal, worked end to end

Suppose a miner profiles a published arena and finds that
`activation.silu_and_mul` is on the critical path. They build a source-only bundle for
that registered singleton target and declare that it applies to BF16 calls on `sm90`.

1. Locally, `scan` checks the tree without importing the candidate. `verify` then loads
   it in a fresh child process, supplies validator-shaped tensors, and compares every
   applicable result with the trusted slot reference. These are development checks, so
   even a clean GPU result does not establish priority or a win.
2. The miner packages the tree. The proposal identity is a SHA-256 over the sorted
   identity-bearing relative paths and bytes, not over gzip timestamps or archive
   compression. They host that archive on public HTTPS and commit the hash and URL with
   a hotkey-signed timelock submission.
3. After finalized reveal, the validator fetches the archive, safely extracts it,
   recomputes the hash, republishes an immutable tree, and resolves the claimed target
   against the active catalog. It observes the actual proposal features rather than
   trusting the manifest to grant itself permissions.
4. Static, build, ABI, graph, and abbreviated-serving screens decide only whether the
   candidate should consume full qualification capacity.
5. V3 qualification loads the exact incumbent and one-target-transition candidate
   engines once onto disjoint resident TP lanes. The host serializes timed B/C/B′
   reads and adds C′/B″ only when the frozen escalation policy requires them. A
   registered eager, untimed audit role (**A**) then checks the candidate delta;
   after candidate teardown, the pristine reference (**T**) supplies
   candidate-free quality evidence.
6. One complete PASS is retained as `reproduction_pending`. A fresh independent attempt
   must reproduce the result. If it does, settlement retains the lower reproduced
   speedup and may create the target crown.
7. The crown can affect rewards, but production still runs the previous release until
   maintainers separately review, integrate, package, sign, and publish a new Engine
   release.

If the implementation is wrong at step 1, the miner changes the bundle. If its HTTPS
server times out at step 3, the validator may retry the same identity. If the candidate
is slower at step 5, that exact identity fails; a revised kernel is a new proposal. This
is why the last authoritative lifecycle state matters more than the last local command.

## Slots, targets, and engines

These terms describe different layers:

- A **slot** is a typed runtime ABI boundary such as `collective.all_reduce`.
- A **singleton target** assigns economic identity to one slot.
- An **atomic target** assigns one economic identity to a validator-defined group of
  regions that must change together.
- A **discovery proposal** is a bounded cross-cutting prototype that cannot claim a
  normal target. Policy and typed records define promotion into a target, integration
  as a reviewed engine change, or a one-time bounded bounty. The durable settlement
  path currently transports only bounty disposition; registered promotion remains
  unimplemented and fails closed.
- A **materialized engine** is the complete content-addressed tree executed for an arm.

The candidate engine can contain the full incumbent stack, but the validator constructs
it. The miner contributes only the selected delta.

The distinction also explains fallback. A narrowly specialized proposal does not replace
the entire engine. For a live call outside its declared capability domain, the incumbent
implementation remains selected. That is safe, but an implementation which never routes
onto material arena calls cannot contribute measurable speedup.

## Two stacks and a reference

`EvaluationStackManifest`
: The referee's incumbent. It may refer to crowned but unintegrated hostile proposals and
  is executed only inside isolation.

`EngineReleaseManifest`
: The serving-stack contract. Every entry in a valid completed manifest is a reviewed
  integrated contribution.

`ReferenceManifest`
: A pristine validator-owned semantic reference. It contains no proposal and never
  competes for speed.

This separation prevents the fastest current proposal from becoming its own correctness
oracle and prevents chain state from leaking into deployment.

## Authority levels

Optima exposes contributor diagnostics that do not create crowns:

- `scan` checks static policy without loading a bundle;
- `verify` compares a slot implementation with its trusted reference; and
- contributor-controlled matched A/B profiling can investigate full-engine behavior and
  throughput without producing validator evidence.

Production authority begins only after finalized intake, immutable publication, a closed
arena service, isolated qualification, retained evidence, and independent reproduction.
See [Proposal to release](../architecture/pipeline.md) for the complete sequence.

Use this rule when reading the rest of the documentation: a command is not authoritative
merely because it performs similar math. Authority comes from the identities, isolation,
policies, evidence, and independent reproduction bound to the production path.

## Read next

- [Run the CPU quickstart](quickstart.md)
- [Understand the architecture](../architecture/overview.md)
- [Choose a registered target](../miner-guide/slots.md)
- [Inspect the current evidence boundary](../reference/state-of-record.md)
