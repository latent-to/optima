# Emissions policy

Optima projects rewards from independently verified marginal improvements in the
active evaluation stack. Settlement and chain publication use a content-addressed
policy manifest; policy parameters are validator consensus configuration and are not
inferred from submissions or historical ledger state.

## Standing reward families

Each active registered target defines one reward family. A singleton target defines
one family for its slot. An atomic target defines one family for the complete target
and suppresses every overlapping singleton family while it is active. Packaging,
integration, release, and engine-stack records do not create additional reward
families.

A crown contributes credit only while its exact contribution is active in the current
evaluation stack and its retained qualification authority can be reopened. Credit is
derived from the measured marginal speedup above unity, expressed as integer parts per
million. It is never derived from an argmax or from the number of slots in a target.

For a crown with `speedup_ppm > 1_000_000`, age `a` blocks, and policy half-life `h`,
the standing credit is:

```text
improvement = speedup_ppm - 1_000_000
credit      = floor(improvement * h / (h + a))
```

This reciprocal decay is deterministic under integer arithmetic and reaches half of
the original credit at one half-life. A retired or neutralized crown contributes no
credit. A stale, incompatible, missing, or unverifiable crown holds the complete
projection; its share is not redistributed silently.

## Discovery bounties

A discovery qualification does not create a standing family or modify the evaluation
stack. It may create one non-renewable bounty claim, identified by the discovery
proposal and retained evidence. The policy manifest bounds both the bounty lifetime
in blocks and the aggregate discovery share in parts per million. Duplicate packaging,
promotion, integration, or release of the same discovery cannot create another claim.

## Projection and publication

Standing credit is summed by miner hotkey after every active family has reopened.
When live discovery claims exist, their aggregate is normalized inside the bounded
discovery pool and standing credit receives the remainder. Otherwise standing credit
receives the complete vector. The final vector is normalized only after chain scope,
validator identity, current stack generations, retained evidence, and current
metagraph membership have been bound into one projection.

Only the control-plane signer may publish a projection. It refreshes chain authority
immediately before reconciliation and again after submission. Publication is journaled
as `intent`, `pending`, `held`, `released`, or `confirmed`; an SDK return value alone
never confirms weights. A held publication requires an explicit audited release event
before another intent. A real extrinsic requires at least one genuine current-schema
crown. Dry runs may exercise projection and reconciliation but cannot create publication
intent.
