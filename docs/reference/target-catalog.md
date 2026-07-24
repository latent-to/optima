# Target catalog

The target catalog answers a different question from the slot registry:
**which smallest validator-registered semantic contribution is being proposed
and rewarded?** A complete isolated engine is the execution unit; one resolved
target is the reward unit.

Target identity is validator-owned. A bundle may request a target ID and mode,
but cannot define its members, overlap, displacement, features, correctness
contract, or composition order.

## What a target specification freezes

A target is more than a name mapped to a slot. Its canonical specification
binds:

- singleton or atomic kind and exact member slots;
- displaced, required, and explicitly compatible targets;
- admitted bundle features such as variants, prepare, override, inspectable
  CUDA source, or reviewed rebuild operations;
- the slot contract projection: entry/prepare ABI, graph-dynamic inputs,
  references, correctness profile, dtype tolerances, KL threshold, and binding
  family; and
- any validator-owned composition rule and precedence.

The resulting target-spec digest travels with proposals, stack entries,
qualification evidence, integration records, and releases. A target ID without
that digest is insufficient to reopen historical authority.

## Registered targets

The `target-catalog.v1` policy contains the 11 singleton slot targets plus one
atomic target:

| Target | Kind | Members / effect |
|---|---|---|
| `activation.silu_and_mul` | slot | Same-named slot |
| `norm.rmsnorm` | slot | Same-named slot |
| `attention.sdpa` | slot | Same-named slot |
| `attention.decode` | slot | Same-named slot |
| `attention.msa_block_score` | slot | Same-named slot |
| `attention.msa_prefill_block_score` | slot | Same-named slot |
| `moe.fused_experts` | slot | Experts without ownership of the trailing reduction |
| `moe.fused_experts_reduce` | slot | Experts plus their trailing reduction |
| `collective.all_reduce` | slot | Same-named slot |
| `collective.ar_residual_rmsnorm` | slot | Same-named slot |
| `collective.moe_finalize_ar_rmsnorm` | slot | Same-named slot |
| `collective.moe_epilogue.v1` | atomic | Owns both fused collective epilogue members below |

The atomic target owns and displaces both
`collective.ar_residual_rmsnorm` and
`collective.moe_finalize_ar_rmsnorm`. Those overlapping identities cannot be
active alongside the atomic target. This prevents one semantic change from
creating duplicate permanent reward titles.

Catalog registration defines identity and admission; it does not by itself
prove that a serving seam is installed. The `attention.msa_block_score`
decode-side integration is non-installing unless the pinned runtime exposes a
stable registered chokepoint. See [State of record](state-of-record.md) for
validated adapter coverage.

## Resolution

A new contribution should declare exactly one competition target explicitly:

```toml
[competition]
target = "attention.msa_prefill_block_score"
mode = "slot"
```

An atomic contribution declares its registered atomic identity:

```toml
[competition]
target = "collective.moe_epilogue.v1"
mode = "atomic"
```

For compatibility with older bundles, the resolver can infer an exact
singleton target or an unambiguous registered atomic member set when the table
is absent. Inference never creates a target, broadens allowed features, or
resolves legacy `system` ownership; explicit declarations remain the clearest
submission contract.

Intake independently observes every feature in the bundle, including CUDA
sources and rebuild operations, and requires the complete set to be admitted by
the registered target. An unknown field or extra capability cannot enlarge the
target. Legacy `mode = "system"` manifests remain parseable for migration but
do not resolve to a current reward title.

### Worked resolution examples

**Ordinary singleton.** A bundle requests `norm.rmsnorm`, contains one RMSNorm
entry, and observes only features admitted by that target. Resolution produces
the singleton target and its frozen target-spec digest. Adding an unrelated
all-reduce row would not create a larger target; it would make the bundle
ineligible for the requested one.

**Atomic epilogue.** A bundle requests `collective.moe_epilogue.v1` and supplies
the exact registered member set. If it becomes active, catalog displacement
removes the two overlapping singleton epilogue titles so one semantic change
does not earn three continuing rewards.

**Legacy inference.** A bundle without `[competition]` may resolve only when
its observed members identify an exact singleton or registered atomic target
unambiguously. A parseable `mode = "system"` row has no current registered
reward identity and therefore cannot resolve by nostalgia.

## Composition

The registered rule `sglang.moe.reduce-first.v1` allows
`moe.fused_experts_reduce` and `moe.fused_experts` to coexist. At the shared
binding family it applies the reduce-owning target first and uses the plain
expert target only where applicable afterward. Manifest order never chooses
precedence.

All other active target pairs must be non-overlapping under their registered
contracts. Catalog validation checks displacement, dependency, compatibility,
and precedence for cycles and contradictions.

Composition and displacement solve different problems. Displacement says two
economic titles cannot remain active together. Composition says two compatible
targets can coexist at a shared runtime binding and fixes which one gets the
first applicable opportunity. Neither order is chosen by manifest row order,
proposal arrival order, or speed.

## Versioning and identity

The complete catalog snapshot is embedded in stack manifests and bound by a
digest. Each target also binds a target-spec digest and a contract digest. A
validator does not reinterpret a historical contribution through whatever
catalog happens to be installed later.

Source: [`optima/target_catalog.py`](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py).
