# Stack manifests

Optima represents evaluation and serving composition with separate canonical
manifest types. They are deliberately not interchangeable.

## The three questions they answer

| Manifest | Question | Proposal source? | Arena identity? | Timed? |
|---|---|---:|---:|---:|
| `EvaluationStackManifest` | What complete incumbent does this arena evaluate? | allowed | yes | as resident B/B′ and conditional B″ |
| `EngineReleaseManifest` | What reviewed set may be materialized and shipped? | rejected | no | serving policy is separate |
| `ReferenceManifest` | What pristine candidate-free engine grades sealed trajectories? | rejected | yes, with reference identities | no |

An evaluation stack and a release stack may name similar semantic
improvements, but equality of intent is not equality of identity. The former
can point at hostile content because it is confined to the referee; the latter
must resolve through integration review records.

## Evaluation stack

`EvaluationStackManifest` identifies the complete incumbent used by a specific
arena. It binds:

- runtime and base-engine digests;
- the arena digest;
- the full target-catalog snapshot and digest, including the registered artifact-provider
  policy; and
- one content-addressed contribution reference per active target.

Its entries may be hostile `ProposalContributionRef` values or reviewed
`IntegratedContributionRef` values. That is why an evaluation stack is never a
serving release. `with_contribution()` creates a new immutable identity; it does
not mutate a signed release.

Conceptual shape:

```json
{
  "type": "evaluation_stack",
  "schema_version": 1,
  "stack_policy_version": "...",
  "runtime_digest": "<sha256>",
  "base_engine_digest": "<sha256>",
  "arena_digest": "<sha256>",
  "catalog_digest": "<sha256>",
  "catalog_snapshot": {},
  "entries": {
    "attention.sdpa": { "type": "proposal", "...": "..." }
  }
}
```

### Read an evaluation identity in layers

- `runtime_digest` fixes the executable runtime context.
- `base_engine_digest` fixes the engine before active contribution deltas.
- `arena_digest` fixes the hardware/workload/policy comparison domain.
- `catalog_snapshot` explains the target policy used at that time, while
  `catalog_digest` authenticates those exact bytes.
- The catalog snapshot carries both `artifact_provider_registry` and its digest. Provider
  kind, ABI, build/load phases, capability vocabulary, and crownability therefore rotate
  catalog and stack authority even when the visible target IDs are unchanged.
- `entries` names the active contribution for each economic target.

Each entry separates the whole artifact digest, selected-payload digest,
target-spec digest, and attribution digest. Two archives are therefore not the
same marginal delta merely because they share a `bundle_id`, and historical
evidence cannot be reinterpreted through a newer catalog.

### A marginal transition

Suppose the incumbent already contains an `attention.sdpa` contribution and a
new proposal wins `norm.rmsnorm`. C is not a two-file bundle: the validator
materializes a complete engine equal to the incumbent everywhere except the
resolved RMSNorm target. After two matching PASS attempts, settlement can
derive a new evaluation stack by adding that reference and applying any
catalog-defined displacement. The old manifest remains a content-addressed
rollback point.

That transition changes the stack digest even when runtime, base engine,
arena, and every unrelated contribution remain fixed.

## Engine release stack

`EngineReleaseManifest` binds runtime, base engine, catalog, and active targets
without an arena. Every entry must be an `IntegratedContributionRef` backed by
a complete `IntegrationReviewRecord`. Proposal references are rejected.

That makes it chain-independent and suitable for deterministic materialization:

```json
{
  "type": "engine_release",
  "schema_version": 1,
  "stack_policy_version": "...",
  "runtime_digest": "<sha256>",
  "base_engine_digest": "<sha256>",
  "catalog_digest": "<sha256>",
  "catalog_snapshot": {},
  "entries": {
    "attention.sdpa": { "type": "integrated", "...": "..." }
  }
}
```

The missing `arena_digest` is intentional. Release identity must not depend on
the validator market that discovered an improvement. Arena measurements remain
provenance in integration records, while the serving stack is reconstructed
from reviewed source, catalog context, and release inputs.

Before materialization, integration validation requires exact one-to-one
coverage between active integrated references and their review records. A
record for the wrong target, selected payload, attribution identity, or source
tree does not count as close enough.

## Pristine reference

`ReferenceManifest` is the separate quality authority used by pristine T. It
binds an empty, candidate-free stack and its materialized tree, launch,
runtime, base engine, arena, catalog, controller/worker distributions, exact
model bytes, logical hardware, workload, tokenizer, hidden corpus commitment,
hidden judge, and selection policy.

T is untimed. It grades trajectories sealed by resident B/C/B′ and conditional
C′/B″ only after any registered eager audit A has completed and candidate
engines have been destroyed. Neither the incumbent evaluation stack nor a
candidate's self-reported scores can substitute for this authority.

Reference identity is intentionally broader than a kernel catalog. It binds
the controller and worker distributions, launch contract, sealed model bytes,
logical hardware, tokenizer, workload, hidden-corpus commitment, judge, and
selection policy. If any of those changes, old quality evidence cannot be
silently attached to the new reference.

## Proposal and integrated references

Both reference kinds preserve the same economic core:

```text
selected_delta_digest = H(
  target_id,
  target_spec_digest,
  selected_payload_digest
)
```

`ProposalContributionRef` additionally names the hostile artifact digest.
`IntegratedContributionRef` instead names the reviewed source-tree digest and
the approving integration-record digest. Attribution remains explicit in both
forms. Promotion changes trust and source ownership without changing the
selected crowned payload under the current contract.

## Canonical identity rules

- Parsing is exact-schema and rejects unknown or mistyped structure.
- Catalog snapshots travel with their digests; installed policy cannot silently
  reinterpret retained evidence.
- Artifact-provider policy is part of that catalog snapshot; a later provider registry
  cannot reinterpret an older direct-artifact contribution.
- Entry keys must equal each reference's target ID.
- Target-spec digests must match the bound catalog context.
- Active-target composition and displacement are revalidated.
- Serialized ordering is canonical before a digest or signature is computed.

## Common rejection cases

| Symptom | Meaning |
|---|---|
| Unknown field during reopen | The object does not match the exact current schema |
| Catalog digest and snapshot disagree | The supplied context is inconsistent or forged |
| Artifact-provider registry or registry digest differs | Native admission/build/load policy belongs to another catalog authority |
| Entry key differs from `ref.target_id` | The mapping tries to relabel a contribution |
| Target-spec digest differs | Qualification used another semantic contract |
| Overlapping active targets | Catalog displacement/composition was not applied canonically |
| Proposal reference in a release manifest | Hostile evaluation content crossed the serving boundary |
| Integration record missing or mismatched | Reviewed-source authority is incomplete |
| Runtime/base/arena context mismatch | The stack is being reopened under a different environment |

Treat these as identity failures, not migration hints. The safe response is to
construct a new typed object through the appropriate transition and retain the
old identity for audit and rollback.

## When to use each API

- Construct an `EvaluationStackContext` from validator-owned arena policy and
  call `validate_against()` before planning resident B/C/B′ or conditional
  C′/B″.
- Use `with_contribution()` only for a catalog-valid evaluation transition; it
  returns a new manifest and never edits the old one.
- Construct a `ReleaseStackContext` from reviewed release inputs and validate
  an `EngineReleaseManifest` before Engine-tree materialization.
- Reopen integration records from retained evidence and require exact coverage
  before accepting integrated entries.
- Construct and reopen `ReferenceManifest` independently of candidate engines;
  never derive T from the current evaluation stack.

Source: [`optima/stack_manifest.py`](https://github.com/latent-to/cacheon/blob/main/optima/stack_manifest.py) and
[`optima/eval/qualification.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification.py).
