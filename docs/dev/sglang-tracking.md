# SGLang compatibility

Optima competes against and integrates with an exact SGLang runtime. The pin is
part of evaluation and release identity, not a loose minimum version. The
source pin is `0.5.13.post1` in
[`optima/compat.py`](https://github.com/latent-to/cacheon/blob/main/optima/compat.py).

A green static seam canary establishes import and chokepoint compatibility only. A
runtime pin is eligible for evaluation or release authority only after end-to-end GPU
controls reject a deliberately broken bundle, accept a faithful bundle, and rebaseline
the registered champions under the exact new identity. Treat the source pin as the
compatibility target, not as evidence that these empirical gates passed. Completed proof
coverage belongs in [State of record](../reference/state-of-record.md).

## Why the pin matters

SGLang internals define the lower side of Optima's narrow waist: module paths,
call signatures, graph capture, attention/MoE dispatch, collectives, engine
arguments, and native dependencies. An upstream change can alter both whether
a contribution loads and what baseline it must beat.

Therefore a pin bump changes evaluation context. Existing performance evidence
cannot simply be relabeled with the new version.

### Validator consensus and the miner boundary

Validators independently score the same revealed contributions and publish weight
vectors. Bittensor's Yuma consensus compares those vectors with the stake-weighted
validator result. If validators run different SGLang revisions, the same contribution
can encounter different kernels, shapes, graph paths, throughput, or quality behavior;
the resulting weight divergence is a protocol fault rather than ordinary measurement
noise. Every validator for one arena must therefore use the same authenticated runtime
and evaluation identity and coordinate a pin transition.

That requirement binds authoritative measurement, not a miner's workstation. Miners may
develop with another SGLang revision—or without SGLang when a slot reference is
sufficient—and target Optima's typed slot ABI. Qualification always re-runs the submitted
delta in the validator's pinned arena, and an Engine release binds its own exact pin.
A pin bump remeasures existing contributions against a changed baseline and execution
context; it does not automatically make a portable contribution's source invalid.
Contributions that depended on an old runtime quirk can fail or lose their advantage
when remeasured, which is the intended outcome.

## What Optima depends on upstream

The canary is table-driven from the registered seam adapters and adds richer
checks for the most consequential surfaces:

| Upstream surface | Why Optima needs it | What movement can break |
|---|---|---|
| `SiluAndMul` and `RMSNorm` | Narrow component call sites | Argument order, residual semantics, fallback routing |
| `RadixAttention.forward` | Attention block insertion before sampling | Q/K/V and batch-state translation, graph behavior |
| `FusedMoE.forward_impl` | MoE waist that survives piecewise capture | Expert inputs, routing outputs, reduction ownership |
| `GroupCoordinator.all_reduce` | Validator-owned TP collective boundary | Process-group ownership and all-rank behavior |
| `Engine.generate` logprob API | Trusted KL/quality observation | Top-logprob collection and sealed trajectory schema |
| `ServerArgs` fields | Deterministic engine launch policy | Model, graph, memory, seed, backend, and logging controls |
| Blessed native base | FlashInfer/CUTLASS/Triton kernel surface | JIT products, numerics, throughput, validator agreement |

Optional seams may report a skip when their required package is absent on a
development box. That is not evidence for the engine environment; run the
canary again inside the exact pinned image where those packages are required.

## Compatibility canary

```bash
python -m optima.cli compat
```

The canary imports the installed runtime, reports its version, and inspects every
registered seam. It does not load a model or require a GPU. A version mismatch
is printed as `DIFFERS from pin`, fails the version check, and makes
`python -m optima.cli compat` exit with status 2. A green result means the exact
pin and expected symbols and signatures are present; it does not mean behavior
or performance is unchanged.

### Interpret the output

- **Import or signature FAIL**: the version cannot be used for scoring until a
  reviewed adapter restores the same semantic boundary.
- **`DIFFERS from pin`**: the version row fails even when every inspected seam
  remains present. Restore the exact pin or complete a reviewed pin bump; do
  not treat signature compatibility as arena identity.
- **Optional-package SKIP**: that seam was not assessed on this host. It must be
  exercised in the complete engine environment.
- **All seams intact**: imports and signatures match; proceed to behavioral
  negative controls and GPU/model validation.

Capture the complete output with installed package versions and image digest.
Do not retain only the final “ALL SEAMS INTACT” line; which optional rows were
skipped matters.

Run the Bittensor SDK canary separately:

```bash
python -m optima.cli chain-compat
```

This separation prevents an unrelated chain SDK change from being confused
with an Engine runtime change.

## Scheduled release and seam check

The repository ships
[`scripts/check_sglang.py`](https://github.com/latent-to/cacheon/blob/main/scripts/check_sglang.py)
and the
[`sglang-canary` workflow](https://github.com/latent-to/cacheon/blob/main/.github/workflows/sglang-canary.yml).
GitHub Actions runs it every Monday at 09:00 UTC and also permits a manual
`workflow_dispatch`.

The script deliberately separates notification from compatibility failure:

- a newer PyPI release emits a GitHub warning but exits successfully; a bump is a
  reviewed rebaseline decision, not an automatic dependency update;
- an installed SGLang version that differs from `PINNED_SGLANG`, or a failed registered
  seam/API check, exits nonzero;
- when SGLang cannot be imported, the PyPI release check still runs but seam coverage is
  skipped, so the CPU workflow does not claim compatibility evidence it could not
  produce.

The workflow attempts to install the pin on its CPU runner, but treats installation
failure as best effort so the release check still runs. Full seam coverage must run in
an environment where the pinned package and its required optional dependencies are
importable. Behavioral GPU/model proof remains a separate step.

Run the same check locally from an installed checkout:

```bash
python scripts/check_sglang.py
```

An operator without GitHub Actions can schedule the command directly, for example:

```cron
0 9 * * 1 cd /path/to/cacheon && .venv/bin/python scripts/check_sglang.py
```

## Pin-bump procedure

1. **Freeze the candidate upstream build.** Record the exact SGLang version,
   repository revision, base image digest, Torch/CUDA stack, native packages,
   and supported GPU architectures.
2. **Run the static canary.** Update seam adapters only where the registered
   semantic boundary is unchanged. A moved upstream symbol is not permission to
   widen a slot.
3. **Run contract tests.** Exercise manifests, target resolution, dispatch,
   typed outputs, graph replay, collectives, deterministic Engine-tree
   materialization, qualification schemas, and release verification.
4. **Run negative controls.** Confirm missing, ambiguous, wrong, graph-unsafe,
   and sabotage contributions still fail closed or route to the trusted
   baseline as specified.
5. **Run real GPU/model qualification controls.** Establish stock stability,
   B/B′ drift behavior, pristine T grading, worker cleanup, supported topology,
   and known-good/known-bad candidates under the new exact environment.
6. **Recalibrate the arena.** Freeze new workload, noise, quality, resource, and
   timing policy through the normal reviewed arena process. Do not reuse old
   empirical thresholds by assumption.
7. **Create new identities.** Runtime, base engine, arena, evaluation stack,
   native build, and release digests must reflect the bump.
8. **Publish migration state.** Mark prior crowns/releases by their original
   pin; do not imply they reproduce on the new one until they do.

## Behavioral proof after the canary

A pin is ready for authority only after progressively stronger controls:

1. a faithful component still passes its registered reference;
2. a deliberately broken component still fails rather than falling through an
   unobserved upstream path;
3. graph capture/replay observes the registered dynamic inputs and outputs;
4. collective seams activate and complete on every rank at the real world
   size;
5. complete stock B/B′ engines remain stable under the arena workload;
6. candidate C changes only the selected target delta;
7. pristine T regrades sealed outputs without candidate code present; and
8. known champions are rebaselined rather than inheriting old speedups.

The broken-bundle control is especially important. A faithful candidate can
appear green even if the adapter never fired and SGLang silently used its
baseline. The negative control proves that the intended call path is actually
under Optima's verifier and receipt authority.

## Seam design rule

Slot contracts are stable semantic ABIs; SGLang adapters are version-specific
glue. Keep upstream-specific imports, call translation, and fallback behavior in
the adapter layer. If the new runtime cannot preserve a target's four
invariants—validator-owned boundary, upstream-of-sampler position,
high-precision reference, and untrusted-number exclusion—do not carry that
target forward unchanged.

## Evidence migration rule

A new pin creates a new runtime digest and therefore new evaluation-stack,
arena, native-build, and release identities. Historical evidence remains valid
for the old context; it does not become corrupt, but it cannot authorize the
new one. Record whether each result is:

- static canary evidence;
- component/graph/collective evidence;
- complete-engine qualification evidence; or
- release build-and-serve evidence.

This prevents a successful import check from being cited as a serving proof,
or a champion from another runtime identity from being quoted as the active baseline.

## Release implications

An `EngineReleaseDescriptor` binds the SGLang version, upstream revision,
digest-pinned base image, reproducible Optima artifacts, native build, and
serving policy. Consumers verify those exact identities. Upgrading a running
deployment means building and signing a new release; it is never an in-place
package upgrade inside an existing release.

See [State of record](../reference/state-of-record.md) for which empirical GPU
proofs exist for the present architecture and which remain outstanding.
