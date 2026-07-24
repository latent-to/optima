# Product model

Optima Engine is a chain-independent, open inference-acceleration distribution. It pins
the supported SGLang runtime as part of product identity and assembles accepted data-plane
optimizations into a canonical engine stack.

The subnet is a market around that product. It discovers, measures, attributes, and rewards improvements. It is not embedded in the released engine, and the managed service never serves mutable miner bundles or reads live chain state to decide what code to execute.

This page is the normative product contract for the proposal, crown,
integration, and release boundaries.

## A useful mental model

Think of Optima as a **compiler-and-release pipeline fed by a market**, not as a
marketplace that hot-loads winner code. The market proposes and measures deltas. The
product pipeline decides whether a measured delta can become maintained source. The
release pipeline freezes a reviewed product that can be consumed without either of the
first two systems being online.

```text
market authority          product authority          deployment authority
proposal -> crown    ->    integrated source    ->    signed release -> rollout
       evidence                 review                     external key
```

The arrows are evidence-bearing handoffs, not automatic conversions. This is why the
same implementation can simultaneously be a valid crown, an unresolved integration
candidate, and absent from the current production release without contradiction.

## Four objects, four authorities

| Object | Contents | Authority | May enter production? |
|---|---|---|---|
| Proposal | A miner-supplied target delta or discovery prototype | Hostile input identified by finalized intake and content digests | No |
| Crown | Reopened evidence that the proposal improved one arena and attributable target | Referee qualification, independent reproduction, and settlement | Not by itself |
| Integrated contribution | Reviewed source and tests that preserve the crowned selected payload, with immutable contributor identity | Source control and integration review | Yes, as release input |
| Engine release | Pinned upstream runtime plus a canonical reviewed stack and sealed release inputs | Signed release descriptor and publication | Yes |

No step may silently substitute one object for another:

- building a proposal in an isolated engine does not make it a release;
- passing a local benchmark does not make it a crown;
- earning a crown does not waive security, licensing, provenance, or compatibility review;
- integrating source does not change an already signed release;
- a chain row or mutable URL is never a production component identity.

## Crown and ship are different decisions

A crown answers an economic question: **did this exact attributable delta improve the frozen evaluation incumbent under the registered arena policy?**

Shipping answers a product question: **can reviewed source, still bound to the crowned selected payload, be maintained and safely included in a chain-independent engine release?**

The ship decision separately requires:

- reproduction against the crowned evaluation stack and the current release context;
- correctness and maintained fallback behavior;
- security review;
- license and provenance approval;
- compatibility with active contributions and the pinned SGLang revision;
- reviewed Optima source that preserves the crowned selected payload, plus
  maintained surrounding packaging and tests;
- immutable contribution attribution;
- exact release, native, model, and policy identities.

This permits emissions to follow crown policy while production deployment follows release policy. See [Release architecture](releases.md).

## Marginal contribution, complete execution

Optima deliberately separates economic identity from process identity.

The **execution unit** is a complete engine. Production qualification materializes
the exact incumbent and one-target-transition candidate engines, loads each once
onto a disjoint resident TP lane, and serializes their timed B/C/B′ reads. A
policy-defined borderline result adds C′/B″ reads from those same loaded lanes.
The candidate then runs in a separate eager, untimed audit role A, and pristine T
runs candidate-free; candidate code never shares the controller's trust domain.

The **reward unit** is the smallest validator-controlled attributable delta:

- one registered singleton slot;
- one registered atomic target spanning an explicit set of semantic regions; or
- one explicitly reviewed discovery contribution.

The candidate stack is built by the validator. It equals the incumbent stack except for one selected target transition. The miner does not supply the incumbent entries and does not gain attribution for the whole engine simply because the complete engine is the safe execution envelope.

This is the core composability property: later work can be evaluated on top of earlier wins without copying earlier contributors' artifacts and without collapsing attribution into winner-take-all engine ownership.

### Worked example: a later 3% delta on top of an earlier 7% delta

Assume the frozen evaluation stack already contains contribution **A** for one target and
is 7% faster than the original base engine. Miner **B** submits a different target delta
that may add another 3%:

1. The validator materializes the exact incumbent engine containing A and loads it once
   onto the resident baseline lane.
2. It materializes the candidate engine from that same stack, replacing only B's
   declared registered target, and loads it once onto the disjoint candidate lane.
   Timed B/C/B′ reads are serialized across the lanes; a policy-defined borderline
   result adds C′/B″ reads from the same resident engines.
3. B's hosted bundle does not need to contain A. The validator supplies A from the
   incumbent manifest and gives B attribution only for the selected delta introduced by B.
4. The registered eager audit role checks the candidate delta outside the timed resident
   reads. T then grades the sealed candidate trajectory using a pristine candidate-free
   reference. The earlier contribution A is not allowed to grade B merely because it is
   in the incumbent.
5. If two independent qualifications pass, settlement may update the evaluation stack to
   contain both A and B. The release stack is still unchanged.
6. If B later passes integration review, a new integrated reference can be selected for a
   future signed release. That product decision does not alter A's or B's historical crown
   evidence.

This example is conceptual; the percentages are not claims about a recorded Optima run.
Its point is the identity split: the worker executes a complete A+B engine, while the
economic transition and integration record describe only B's registered delta.

## Two stacks and a trusted reference

The product model is reflected directly in manifest types:

| Manifest | May contain hostile proposal code? | Arena-bound? | Used for timing? | Used for serving? |
|---|---:|---:|---:|---:|
| `EvaluationStackManifest` | Yes | Yes | Yes | No |
| `EngineReleaseManifest` | No; integrated contributions only | No | Release validation only | Yes |
| `ReferenceManifest` | No; validator-owned | Quality profile-bound | No | No |

The evaluation stack is an economic hill-climb state. The release manifest is a product state. The reference manifest is semantic authority. A crown can transactionally update the first; only reviewed promotion and signing can create a new instance of the second. See [Stacks and manifests](stacks.md).

## Registered targets

The validator-owned target catalog defines what can receive ordinary attribution. It records:

- target identity and kind;
- singleton members or the explicit members of an atomic target;
- the frozen slot contract digest;
- permitted contribution features;
- overlap, displacement, requirements, and composition precedence.

Miner packaging and manifest row order do not define economic scope. A bundle that explicitly claims a registered target but does not resolve to its exact members and allowed features fails resolution rather than falling through to an unregistered identity.

The registered catalog contains every singleton slot and the atomic `collective.moe_epilogue.v1` target. The atomic target owns both `collective.ar_residual_rmsnorm` and `collective.moe_finalize_ar_rmsnorm`, and explicitly displaces the corresponding singleton targets while active. The live policy is implemented in [`target_catalog.py`](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py).

## Discovery lane

Cross-cutting work may initially be impossible to express as one registered target. Optima admits it through a fenced discovery lane rather than pretending the contribution has narrow attribution.

Discovery is intentionally bounded:

- its execution and evidence remain isolated and content-addressed;
- validator policy defines the permitted source-patch surface;
- it does not silently mutate the registered target catalog;
- it does not create a permanent equal-emission whole-engine family;
- the durable disposition path can retain review and pay a bounded reviewed bounty;
  registered promotion into a reviewed slot or atomic boundary remains fail-closed
  until typed promotion transport, target registration, and fresh
  qualification/CROWN linkage exist.

Rebuild and dependency-patch capabilities are likewise validator-reviewed. A normal target can only use features explicitly admitted by its target specification, and permanent framework mutation is not a miner-selected permission.

## Product scope

### Inference data plane

Ordinary contribution targets may cover:

- fused operators and quantized GEMMs;
- attention algorithms and sparse-attention scoring;
- MoE execution;
- collectives and compute/communication overlap;
- KV-cache layouts and operations;
- graph plans and fused blocks;
- model-specific execution strategies;
- bounded scheduling-adjacent execution changes whose value depends on the data plane.

### Service control plane

The following remain the responsibility of the upstream runtime, service, or orchestrator:

- HTTP and API semantics;
- authentication and authorization;
- tokenization;
- request admission;
- fleet orchestration and autoscaling;
- observability;
- deployment and operational lifecycle management.

Keeping the service plane outside normal submissions limits the blast radius of untrusted proposals and preserves the engine as an embeddable distribution rather than a competing API server.

## Chain independence

A valid engine release has no runtime dependency on Bittensor, wallets, miner endpoints, current weights, or referee databases. The chain determines proposal priority and reward state; it does not dynamically choose production code.

The release build consumes reviewed source and exact, signed inputs. The serving container consumes the signed release publication and a sealed model tree. If chain access and miner hosting disappear, the released engine remains rebuildable, verifiable, and deployable from its retained artifacts.

## Architectural acceptance test

The product model is intact only if all six statements hold:

1. Removing chain access and miner hosting does not prevent rebuilding or serving the latest signed release.
2. A new component is evaluated as one marginal substitution over the current stack.
3. The trusted controller never imports candidate code; candidate runtime execution stays
   inside a complete isolated engine, while a sealed-direct-artifact factory may execute
   only inside its further isolated no-network/no-GPU compiler child.
4. A whole-system prototype cannot acquire a duplicate permanent whole-engine reward title by packaging alone.
5. Every shipped component resolves to reviewed source and immutable attribution.
6. Updating the evaluation incumbent and publishing a release are independent, explicitly authorized state transitions.

## Review questions for a proposed feature

Before extending Optima, locate the feature in this model:

1. **What is the reward unit?** Name the exact registered singleton, atomic target, or
   discovery identity. “The whole engine” is not an acceptable default.
2. **Who supplies surrounding code?** The validator must assemble the incumbent; a miner
   must not be required to redistribute other contributors' bundles.
3. **Where does hostile code execute?** Runtime proposal code belongs only in the complete
   isolated evaluation engine. A direct-artifact factory may run only in the bounded
   compiler child during disposable prebuild. Neither path may import candidate code into
   the controller or carry a miner runtime callback into a serving release.
4. **What creates product authority?** Identify the review record, preserved selected
   payload, integrated source, maintained tests, and release decision.
5. **Can the resulting release stand alone?** Rebuild, verification, and serving must not
   require a wallet, live chain query, miner URL, or referee database.

A design that cannot answer these questions usually crosses the crown/ship boundary or
confuses the execution unit with the reward unit.

## Source map

- This page — normative product invariants
- [`stack_manifest.py`](https://github.com/latent-to/cacheon/blob/main/optima/stack_manifest.py) — proposal, integration, evaluation-stack, and release-stack identities
- [`target_catalog.py`](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py) — reward-unit policy
- [`engine_tree.py`](https://github.com/latent-to/cacheon/blob/main/optima/engine_tree.py) — deterministic proposal and integrated-source materialization
- [`release.py`](https://github.com/latent-to/cacheon/blob/main/optima/release.py) — chain-independent release artifacts and signatures
