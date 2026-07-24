# Codebase map

This map groups the Optima source by authority boundary. File names are links
to the current code repository; source remains authoritative when details
change.

## Read by authority, not import depth

The easiest way to get lost in Optima is to follow imports as though every
module had the same trust level. Start from the decision whose authority you
are trying to understand:

```mermaid
flowchart LR
    M["manifest + slots + target catalog"] --> D["local dispatch and verification"]
    M --> N["sealed artifact prebuild + device runtime"]
    M --> I["finalized intake + immutable publication"]
    I --> A["injected arena service"]
    A --> Q["isolated qualification evidence"]
    Q --> S["settlement + evaluation stack"]
    S --> W["emissions + weight journal"]
    Q --> R["integration review"]
    R --> E["Engine tree + signed release"]
    N --> Q
    N --> E
```

The local branch is useful to contributors but cannot crown anything. The
intake, arena, qualification, settlement, and weight branch owns hostile
evaluation and economic state. A sealed direct artifact enters qualification
through the registered prebuild/runtime boundary; after integration review, its
sealed native publication may also be bound into a release. The release branch
reopens evidence but accepts only reviewed integrated source.

## Contribution contract

| Area | Primary source |
|---|---|
| Bundle parsing and path rules | [`manifest.py`](https://github.com/latent-to/cacheon/blob/main/optima/manifest.py) |
| Slot ABI and trusted references | [`slots.py`](https://github.com/latent-to/cacheon/blob/main/optima/slots.py) |
| Target identity and composition | [`target_catalog.py`](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py) |
| Typed tensor/output boundary | [`tensor_spec.py`](https://github.com/latent-to/cacheon/blob/main/optima/tensor_spec.py) |
| Static policy | [`sandbox.py`](https://github.com/latent-to/cacheon/blob/main/optima/sandbox.py) |
| Tracing-JIT admission | [`dsl_jit_policy.py`](https://github.com/latent-to/cacheon/blob/main/optima/dsl_jit_policy.py) |
| Local and distributed verification | [`verify.py`](https://github.com/latent-to/cacheon/blob/main/optima/verify.py), [`verify_collective.py`](https://github.com/latent-to/cacheon/blob/main/optima/verify_collective.py) |
| SGLang dispatch | [`dispatch.py`](https://github.com/latent-to/cacheon/blob/main/optima/dispatch.py), [`seams.py`](https://github.com/latent-to/cacheon/blob/main/optima/seams.py) |
| Scheduler-role candidate load | [`seam.py`](https://github.com/latent-to/cacheon/blob/main/optima/seam.py), [`sglang_scheduler_gate.py`](https://github.com/latent-to/cacheon/blob/main/optima/integrations/sglang_scheduler_gate.py) |

## Sealed direct artifacts

| Area | Primary source |
|---|---|
| Closed provider policy | [`artifact_provider.py`](https://github.com/latent-to/cacheon/blob/main/optima/artifact_provider.py) |
| Slot call ABI, resources, and lifecycle | [`artifact_abi.py`](https://github.com/latent-to/cacheon/blob/main/optima/artifact_abi.py), [`artifact_runtime.py`](https://github.com/latent-to/cacheon/blob/main/optima/artifact_runtime.py) |
| Canonical direct-execution identity | [`artifact_identity.py`](https://github.com/latent-to/cacheon/blob/main/optima/artifact_identity.py), [`artifact_resource_identity.py`](https://github.com/latent-to/cacheon/blob/main/optima/artifact_resource_identity.py) |
| Declarative device launch | [`artifact_device_launch.py`](https://github.com/latent-to/cacheon/blob/main/optima/artifact_device_launch.py) |
| CUBIN ABI and Driver admission | [`cuda_cubin.py`](https://github.com/latent-to/cacheon/blob/main/optima/cuda_cubin.py), [`cuda_launch.py`](https://github.com/latent-to/cacheon/blob/main/optima/cuda_launch.py) |
| Parameter, TMA, and FastDivmod materialization | [`cuda_materialize.py`](https://github.com/latent-to/cacheon/blob/main/optima/cuda_materialize.py) |
| CuTe compiler boundary and sealed index | [`cute_aot.py`](https://github.com/latent-to/cacheon/blob/main/optima/cute_aot.py), [`cute_cubin.py`](https://github.com/latent-to/cacheon/blob/main/optima/cute_cubin.py) |
| Measured compile profile | [`eval/native_compile_profile.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/native_compile_profile.py) |
| Registered build patcher | [`patchers/build_cute_cubin.py`](https://github.com/latent-to/cacheon/blob/main/optima/patchers/build_cute_cubin.py) |
| Rank-local post-CUDA binding | [`integrations/sglang_artifact_context.py`](https://github.com/latent-to/cacheon/blob/main/optima/integrations/sglang_artifact_context.py) |

## Intake and referee

| Area | Primary source |
|---|---|
| Chain-facing commands | [`cli.py`](https://github.com/latent-to/cacheon/blob/main/optima/cli.py) |
| Finalized intake and SQLite state | [`chain/intake.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/intake.py) |
| Hardened archive fetch | [`chain/fetch.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/fetch.py) |
| Validator loop | [`chain/validator_loop.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/validator_loop.py) |
| Injected arena boundary | [`arena_service.py`](https://github.com/latent-to/cacheon/blob/main/optima/arena_service.py) |
| Qualification schema and regrading | [`eval/qualification.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification.py) |
| Resident routing screen | [`eval/oci_resident_session.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_resident_session.py), [`eval/resident_queue.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/resident_queue.py), [`eval/resident_screen_lane.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/resident_screen_lane.py) |
| Adaptive two-lane qualification | [`eval/crossover_runtime.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/crossover_runtime.py), [`eval/qualification_runner.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification_runner.py) |
| Host audit grading | [`audit_gate.py`](https://github.com/latent-to/cacheon/blob/main/optima/audit_gate.py) |
| OCI lifecycle and protocol | [`eval/oci_backend.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_backend.py), [`eval/oci_session_protocol.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_session_protocol.py) |
| Host-owned resident B/C/B′ and conditional C′/B″ session | [`eval/oci_outer_session.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_outer_session.py) |
| Immutable native prebuild | [`eval/oci_prebuild.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_prebuild.py) |
| Device conditioning/cleanup | [`eval/device_state.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/device_state.py) |
| Compile-profile and multi-architecture prebuild | [`eval/native_compile_profile.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/native_compile_profile.py), [`eval/oci_prebuild.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_prebuild.py) |

## State, economics, and weights

| Area | Primary source |
|---|---|
| Evaluation/release stack identities | [`stack_manifest.py`](https://github.com/latent-to/cacheon/blob/main/optima/stack_manifest.py) |
| Transactional settlement state | [`chain/intake.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/intake.py) |
| Pure emissions projection | [`economics.py`](https://github.com/latent-to/cacheon/blob/main/optima/economics.py) |
| Weight publication reconciliation | [`chain/weights.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/weights.py) |
| Finite-debt arithmetic | [`finite_debt.py`](https://github.com/latent-to/cacheon/blob/main/optima/finite_debt.py), [`incentive_composition.py`](https://github.com/latent-to/cacheon/blob/main/optima/incentive_composition.py) |
| Durable V2 activation and state | [`chain/finite_debt_store.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/finite_debt_store.py), [`chain/incentive_composition_store.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/incentive_composition_store.py), [`chain/incentive_activation.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/incentive_activation.py) |
| V2 projection, confirmation, and debit | [`chain/debt_publication.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/debt_publication.py) |
| Copy and attribution evidence | [`copy_fingerprint.py`](https://github.com/latent-to/cacheon/blob/main/optima/copy_fingerprint.py) |

## Engine integration and release

| Area | Primary source |
|---|---|
| Deterministic Engine tree | [`engine_tree.py`](https://github.com/latent-to/cacheon/blob/main/optima/engine_tree.py) |
| Model provisioning | [`model_provision.py`](https://github.com/latent-to/cacheon/blob/main/optima/model_provision.py) |
| Release evidence/signing/publication | [`release.py`](https://github.com/latent-to/cacheon/blob/main/optima/release.py) |
| Release runtime verification | [`release_runtime.py`](https://github.com/latent-to/cacheon/blob/main/optima/release_runtime.py) |
| Container host policy | [`release_host.py`](https://github.com/latent-to/cacheon/blob/main/optima/release_host.py) |

## Compatibility and discovery

| Area | Primary source |
|---|---|
| SGLang pin and canary | [`compat.py`](https://github.com/latent-to/cacheon/blob/main/optima/compat.py) |
| Bittensor SDK canary | [`chain_canary.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain_canary.py) |
| Discovery policy and arm | [`discovery.py`](https://github.com/latent-to/cacheon/blob/main/optima/discovery.py) |
| Discovery overlay | [`discovery_overlay.py`](https://github.com/latent-to/cacheon/blob/main/optima/discovery_overlay.py) |

## Follow a concrete task

### “Why was this bundle rejected?”

Read in this order:

1. `manifest.py` for exact TOML shape and contained-path rules;
2. `sandbox.py` and `dep_policy.py` for observed source/build features;
3. `target_catalog.py` for target resolution and admitted features;
4. `artifact_provider.py` and `artifact_abi.py` when the row declares direct exports;
5. `slots.py` and `tensor_spec.py` for callable/output semantics; and
6. `verify.py` or `verify_collective.py` for executable correctness.

This ordering separates syntax, capability admission, ABI, and numerical
failure. They are different diagnoses even when the CLI reports them in one
run.

### “How did a finalized reveal become a crown?”

Start at `chain/intake.py`, then follow `chain/fetch.py` and
`chain/publication.py` into `chain/validator_loop.py`. The loop resolves the
closed `ArenaServiceRegistry`, receives screening and qualification work,
persists evidence references, and invokes transactional settlement. Read
`eval/qualification_runner.py` alongside `eval/qualification.py`: the runner
orchestrates work; the schema and regrader define what counts as authority.

The first matching PASS leaves reproduction pending. A second distinct,
matching PASS may settle the contribution and advance the evaluation stack.
Console output and evaluator summary text are never the settlement input.

### “Why did weight publication remain pending?”

Read `economics.py` first: it computes the pure global projection from reopened
active contribution state. Then read `chain/weights.py`: it refreshes the live
metagraph, journals intent before submission, records SDK results without
treating them as confirmation, and reconciles later chain observation. The journal stores
status/chronology metadata and a projection digest, not raw pre/post readback vectors.
`--dry-run` creates no journal intent, while reconciliation and submission have
different durable states.

### “What exactly ships?”

Follow `stack_manifest.py` into `engine_tree.py`, `model_provision.py`, and
`release.py`. The release manifest accepts only `IntegratedContributionRef`
objects. The selected payload remains bound to its crowned digest; later
materialization owns deterministic module namespaces and packaging. Finally,
`release_runtime.py` and `release_host.py` enforce consumer and host policy.

## State and evidence locations

| Object | Owner | Identity / persistence rule |
|---|---|---|
| Parsed bundle | Contributor/diagnostic process | Content hash over the admitted bundle tree |
| Finalized publication | Intake controller | Validator-owned immutable worker publication |
| Qualification evidence | External evidence root plus deployment-owned expected plan/provider context | Typed attempt and referenced artifacts; full regrade additionally requires reconstructed `CausalQualificationInput`, while settlement restart performs narrower byte/PASS authentication |
| Evaluation stack | Referee state | Canonical manifest that may reference hostile proposals |
| Settlement and weight state | Chain-scoped SQLite controller | Transactional single-writer state plus projection-linked intent/status journal; live readback vectors are not serialized |
| Integrated source | Reviewed source control | Full reviewed commit plus selected-payload and attribution digests |
| Engine release | Release publication root | Descriptor-addressed signed tree reopened under an expected public key |

Paths are deliberately not identities. A local directory name, URL, database
row number, or registry tag cannot replace the corresponding digest-bound
object.

## Tests as executable maps

Tests mirror the authority boundaries rather than one monolithic integration
fixture:

- `test_static.py` and `test_target_catalog.py` cover hostile
  input and target admission;
- `test_artifact_abi.py`, `test_artifact_device_launch.py`,
  `test_artifact_runtime.py`, `test_cuda_cubin.py`, `test_cuda_materialize.py`, and
  `test_cute_cubin.py` cover the sealed direct-artifact boundary;
- `test_stack_manifest.py`, stack-planning tests, and `test_engine_tree.py`
  cover canonical composition and integration materialization;
- qualification, OCI, audit, and reference-protocol tests cover resident
  B/C/B′, conditional C′/B″, registered eager audit A, then pristine T;
- chain-intake, settlement, economics, and weight-publication tests cover
  durable economic transitions; and
- release, runtime, registry, and host tests cover the signed serving boundary.

When learning a type, search for both its successful construction and its
rejection tests. The negative cases usually reveal which fields are security
inputs rather than descriptive metadata.

## Production authority entry points

Development helpers and test fixtures are not alternate qualification authorities. Audit
production intake from the validator loop and durable intake store, then follow the
registered arena, qualification runner, settlement transition, and weight-publication
journal. A path that does not emit and reopen the typed products for those boundaries
cannot substitute for them.

Tests are organized under `tests/` by the same boundaries. Contract tests are
often the clearest executable examples because they build exact typed objects
and assert fail-closed behavior.
