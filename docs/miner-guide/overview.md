# Miner guide

Optima accepts source proposals that make a validator-owned inference engine
faster without changing its required behavior. A miner contributes one
registered delta, the validator evaluates that delta inside the exact incumbent
stack, and only independently reproduced wins can become a crown.

This is not a contest for self-reported microbenchmarks. The validator owns the
model, runtime, prompts, reference behavior, build, launch policy, measurements,
and final decision.

## The job in one sentence

Find one validator-registered boundary that matters to the published workload,
implement exactly that boundary, and show that substituting only your delta into the
current incumbent produces a reproducible end-to-end improvement without weakening
behavior.

That sentence contains the whole discipline:

- **registered boundary** rules out inventing an economic target in the manifest;
- **matters to the workload** rules out optimizing a fast but irrelevant microkernel;
- **exactly that boundary** preserves a causal marginal comparison;
- **current incumbent** rules out benchmarking against a convenient stock baseline;
- **end-to-end** accounts for dispatch, synchronization, graph mode, and downstream work;
- **reproducible** requires two independent qualification attempts; and
- **without weakening behavior** keeps pristine quality evidence outside candidate
  control.

## Choose your path before writing code

| Your idea | Correct path | Why |
|---|---|---|
| replaces one published callable seam | singleton component target | the catalog already defines its ABI and reward identity |
| must replace every member of a published coupled seam | registered atomic target | one measured/rewarded delta owns the coupled semantics |
| specializes one slot for a narrow shape or topology | component variant with an explicit capability domain | the incumbent safely handles non-matching calls |
| changes scheduler/model-executor behavior or proposes a new seam | discovery proposal | the change crosses the closed component ABI |
| only changes how reviewed contributions are packaged or served | product integration, not mining | release authority is separate from competition authority |

Do not force a discovery-shaped idea into a component bundle by vendoring SGLang or
using `setup`. No registered target currently permits engine-wide setup, and legacy
`mode = "system"` is not crownable.

## The three identities to keep separate

A **slot** is a callable seam in the engine, such as
`activation.silu_and_mul` or `collective.all_reduce`. Its ABI says which inputs
you receive and which validator-allocated outputs you must fill.

A **registered target** is the economic identity of a contribution. The target
catalog fixes its members, overlap rules, allowed features, and semantic
contract. The current catalog registers a singleton target with the same
identifier for each slot.

An **atomic target** is one registered target whose delta necessarily spans
multiple slots. It is not “two entries in one bundle” by itself. The catalog
must explicitly register the combined semantics and say which singleton
targets it displaces. The current catalog includes one such target,
`collective.moe_epilogue.v1`, over two collective seams.

The manifest requests a target; it does not define one. For a competitive
bundle, declare the request explicitly:

```toml
[competition]
target = "activation.silu_and_mul"
mode = "slot"
```

See [Slots and targets](slots.md) and the authoritative
[target catalog](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py).

## What happens to a submission

Optima keeps four objects distinct:

1. A **proposal** is the source archive you publish and commit on-chain. It is
   untrusted input, not an engine dependency.
2. A **crown** is an independently reproduced marginal win for one registered
   target in one evaluation stack. It can receive standing reward under the
   active emissions policy.
3. An **integrated contribution** is reviewed Optima-owned source whose selected
   payload remains bound byte-for-byte to the crown, with maintained surrounding
   packaging, tests, and attribution. Crowning does not perform this review
   automatically.
4. A conforming **Engine release** is a signed, chain-independent software release
   built from integrated source. A crown is not permission to ship miner code. The
   current revision does not claim a completed production release.

That separation is part of the product contract, not release ceremony. Read
the full [product model](../architecture/product-model.md)
before working on an advanced target.

## How qualification works

For a registered target, the validator constructs an exact marginal comparison:

- **B**: the opening read from the exact incumbent on the resident baseline lane;
- **C**: the first read from the one-target-transition candidate on the disjoint
  resident candidate lane;
- **B′**: the closing read from the same loaded incumbent;
- **C′/B″**: conditional repeat reads from those same loaded lanes when the
  frozen escalation policy requires them;
- **A**: a registered eager, untimed audit role for the candidate delta; and
- **T**: a candidate-free pristine reference used after candidate teardown.

The candidate does not choose the rest of the stack. The validator materializes
the exact incumbent and candidate engines, loads each once onto its physical TP
lane, and serializes the timed reads. Bookending detects drift, A supplies the
registered sampled slot regrade, and T prevents “fast because behavior changed”
from becoming a win. Static, build, ABI, graph, and abbreviated-serving checks
are admission screens only; they cannot crown a proposal.

A promoted proposal must pass two complete, independent qualification attempts.
The crown records the lower reproduced speedup. After the first passing attempt,
the durable intake state is `reproduction_pending`; that is not yet a crown.

The evaluation design and evidence objects live in
[qualification.py](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification.py),
[qualification_runner.py](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification_runner.py),
and [arena_service.py](https://github.com/latent-to/cacheon/blob/main/optima/arena_service.py).

## How rewards work

Rewards attach to canonical registered targets, not arbitrary bundles. The
repository contains two non-interchangeable generations: V1 assigns the active
crown a decaying relative share, while the selected but inactive V2 policy
issues bounded principal and pays it down through confirmed epochs. Atomic
targets suppress only the overlapping singleton families declared by the
catalog.

The validator publishes the active policy bytes and digest. Do not infer income
from a local speedup: reward generation, arena availability, target activity,
competing crowns or debt, discovery capacity, metagraph state, and confirmed
publication all matter. Discovery proposals never receive a standing target
crown. Although policy and typed records define registered discovery promotion,
the durable settlement path does not transport that authority and fails closed;
the implemented durable discovery disposition is bounty-only. Read
[Incentives](incentives.md) for the exact formulas and activation boundary, then
[Discovery lane](discovery-lane.md) for the separate proposal class.

## Your development loop

The local CLI provides static and component diagnostics:

```bash
python -m optima.cli scan my_bundle
python -m optima.cli verify my_bundle --device cuda --dtype bfloat16
```

These commands find manifest, static-policy, ABI, correctness, routing, and graph
problems. Profile and bracket serving performance in an environment matching the
published arena contract. Neither the CLI checks nor a contributor-controlled A/B run
reproduces crown authority: local work does not possess the finalized intake record,
validator stack manifest, hidden inputs, calibrated policies, immutable publications,
isolated service, or second independent qualification attempt.

Use this guide in order:

1. [Slots and targets](slots.md)
2. [Bundle format](bundle-format.md)
3. [Kernel ABI](kernel-abi.md)
4. [Your first kernel](your-first-kernel.md)
5. [Finding a win](finding-a-win.md)
6. [Submitting](submitting.md)
7. [Incentives](incentives.md)
8. [Diagnostics](diagnostics.md)

Read [Override points](override-points.md), [Dependency patches](dep-patches.md),
and [Discovery lane](discovery-lane.md) only when the registered target or
proposal class requires them.

At the end of the sequence you should be able to answer, with concrete identities:

1. Which arena, stack generation, target, model, architecture, topology, phase, and dtype
   does the proposal address?
2. Which exact files and capability domain form the selected delta?
3. Which verifier shapes and graph replays exercised every applicable variant?
4. Why can this slot-level mechanism move end-to-end critical-path time?
5. Which local result is diagnostic, and which operator receipt is the last
   authoritative state?

If any answer is still “whatever the validator chooses,” obtain the published arena
contract before submitting. A content-addressed proposal cannot be repaired in place
after reveal.
