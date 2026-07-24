# Emissions policy

Optima separates economic accounting from chain publication. Settlement creates
content-addressed claims. A policy projects those claims into an exact
1,000,000-part weight vector. A separate signer journals, submits, reads back, and
confirms that vector.

Policy bytes are validator consensus configuration. They are not supplied by a
miner, inferred from a bundle, or changed by an operator after observing a result.

## Policy generations

Two generations coexist so retained evidence remains reopenable:

| Generation | Claim model | Publication command | Status |
|---|---|---|---|
| Legacy V1 | Decaying standing credit plus bounded discovery claims | `optima set-weights` | Retained and operational |
| Finite-debt V2 | Finite registered-CROWN principal plus reviewed discovery bounty | `optima set-debt-weights` | Implemented behind an explicit one-way activation |

Activation does not reinterpret a V1 claim or rewrite its policy digest. V1 and V2
must never publish concurrently from the same economic authority.

## Legacy V1

Every active registered target defines one reward family. A singleton target owns
one slot. An atomic target owns its complete member set and suppresses explicitly
overlapping singleton families while active. Packaging, integration, and release
records do not create additional reward families.

For a retained crown with `speedup_ppm > 1_000_000`, age `a`, and policy
half-life `h`, standing credit is:

```text
improvement = speedup_ppm - 1_000_000
credit      = floor(improvement * h / (h + a))
```

Retirement or neutralization removes standing credit. A stale, incompatible,
missing, or unreopenable active crown holds the complete projection; its share is
not silently redistributed.

A discovery qualification can create one non-renewable bounded claim. It does not
install an evaluation-stack contribution or create a standing family. Duplicate
packaging, promotion, integration, or release cannot renew that claim.

The V1 projector reopens every active stack and claim, binds exact finalized chain
scope and metagraph membership, aggregates credit by hotkey, and normalizes one
positive integer-ppm vector totaling 1,000,000.

### All-uncrowned bootstrap

Normal V1 projection refuses to publish without a real crown. An operator may
explicitly direct the complete vector to a registered burn hotkey:

```bash
optima set-weights \
  --intake-db chain_intake/intake.sqlite3 \
  --netuid <NETUID> \
  --network <NETWORK_OR_WSS_URL> \
  --wallet default \
  --hotkey validator \
  --half-life-blocks <BLOCKS> \
  --discovery-lifetime-blocks <BLOCKS> \
  --discovery-pool-ppm <PPM> \
  --refresh-blocks <BLOCKS> \
  --burn-hotkey <REGISTERED_BURN_HOTKEY> \
  --dry-run
```

The burn path is valid only when all of these are true:

- there is no active standing or discovery claim;
- no evaluation arena has a crowned generation;
- V2 composition has not been activated; and
- the burn hotkey belongs to the exact projection metagraph.

The same command fails closed as soon as real economic authority exists.

### V1 publication loop

`set-weights` supports one reconciliation or a continuous operator loop:

```bash
optima set-weights <POLICY_AND_SIGNER_ARGUMENTS> \
  --watch \
  --interval <SECONDS>
```

Watch mode reruns the complete authority refresh and reconciliation. It uses bounded
retry for retryable transport or chain faults and does not retry a nonretryable
publication fault. It cannot be combined with `--dry-run`, `--reconcile-only`, or
`--release-hold`.

Before signing, the reconciler refreshes finalized authority. A later finalized
head is acceptable only when the validator UID and every weighted recipient UID
remain unchanged. UID reassignment before signing aborts publication; reassignment
after submission prevents confirmation and retains a hold.

The publication journal distinguishes `intent`, `pending`, `confirmed`, `held`, and
`released`. An SDK success response is not confirmation. Confirmation requires an
exact finalized readback of the intended recipient set and values within the fixed
verifier tolerance.

Signer-free modes reopen journal state without submitting:

- `--reconcile-only` grades retained publication state against chain authority.
- `--release-hold "reason"` appends an audited release; it does not approve or
  submit the old vector.

## Finite-debt V2

V2 accounting uses integers and fixed-point arithmetic. One epoch contains
1,000,000 weight-ppm units. The selected composition reserves at least 100,000
units and allocates the rest to finite registered and discovery debt.

### Selected policy evidence identity

The implemented one-campaign composition recognizes two exact selection-report
identities:

| Selection authority | Report digest |
|---|---|
| Registered-CROWN core policy | `7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590` |
| Reviewed-discovery composition policy | `6bdfce26e4e6090e0dcc8814a636c665f28d1ff20945a09d43a9a90dc94151fc` |

These values identify the reviewed selection reports; they are not substitutes
for the content digests of the policy manifests, activation approval, claims,
or shadow receipts. The selected-policy checks in
[`incentive_composition_store.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/incentive_composition_store.py)
require both report identities together with the selected numeric policy. The
tracked D-015 replay configuration records the same pair in
[`d015_launch_load_config.json`](https://github.com/latent-to/cacheon/blob/main/tests/fixtures/incentives/d015_launch_load_config.json).
Activation still reopens and binds the exact policy and approval bytes described
below.

### Campaign policy

A finite-debt policy is content-addressed and accepts:

- one campaign with a 1,000,000-ppm claim-sizing share; or
- two campaigns with 500,000 ppm each.

Every registered family maps to exactly one campaign, and every campaign has at
least one family. Campaign share sizes new claims; it is not a permanent epoch
silo and is not divided by the number of families.

The current activation path accepts exactly one retained campaign. Supporting a
second live campaign, model rotation, or successor activation requires a new
reviewed migration contract.

### Registered-CROWN principal

Only an independently reproduced registered-lane `CROWN` issues registered
principal. For reproduced multiplicative speedup `s`:

```text
G = floor(1_000_000 * ln(s) / ln(1.01))
```

If the family has no retained or activation-seeded prior clock, its multiplier is
`1_000_000`. Otherwise, with accepted finalized-crown gap `D`:

```text
M = 1_000_000 + floor(100_000 * D / (D + 648_000))
```

For post-reserve capacity `900_000`, campaign share `B_c`, and scaling parameter
`k`, principal uses this flooring order:

```text
F_c = floor(900_000 * B_c / 1_000_000)
Q   = floor(F_c * k * G * M / 1_000_000^3)
```

Only an accepted crown advances the family clock. A failed, held, copied,
discovery-only, retirement, or neutralization disposition does not.

### Reviewed discovery

Reviewed discovery has two policy-level decisions:

1. `bounty_only` issues one finite, non-renewable discovery claim.
2. `registered_promotion` issues no discovery debt and would require fresh
   registered qualification, reproduction, and a CROWN.

The durable implementation currently supports only `bounty_only`. Registered
promotion fails closed until typed `DiscoveryWinRecord` and `DiscoveryPromotion`
authority can be transported, reopened, and linked to fresh registered-lane
authority.

A bounty is capped at 50,000 weight-ppm epoch units and has a 648,000-block
lifetime measured from the retained qualified-win block. Review delay consumes
that lifetime; review at or after expiry cannot mint a claim. The review digest is
content-bound but supplied by the controller. The store does not independently
grade an external governance system.

### Epoch composition

For live discovery debt `D_d` and registered debt `D_c`, the selected composition
allocates:

```text
P_d     = min(50_000, D_d)
P_c     = min(900_000 - P_d, D_c)
reserve = 1_000_000 - P_d - P_c
```

Each class distributes its integer allocation pro rata by remaining principal
using claim-digest largest remainder, then aggregates by hotkey. Expired or
cancelled principal is forfeited accounting debt, not a one-time reserve payment.
Projection never debits a claim.

The denomination is confirmed weight-ppm epochs. The policy does not promise a
quantity of TAO, alpha, or fiat value.

## Activation

These commands inspect candidate activation authority without a wallet:

```bash
optima chain-incentive-shadow \
  --netuid <NETUID> \
  --network <NETWORK_OR_WSS_URL> \
  --policy <CORE_POLICY.json> \
  --claims-fixture <SYNTHETIC_CORE_CLAIMS.json> \
  --expected-policy-digest <SHA256> \
  --expected-claims-digest <SHA256> \
  --output <NEW_RECEIPT.json>

optima chain-incentive-composition-shadow \
  --netuid <NETUID> \
  --network <NETWORK_OR_WSS_URL> \
  --core-policy <CORE_POLICY.json> \
  --core-claims-fixture <SYNTHETIC_CORE_CLAIMS.json> \
  --discovery-policy <COMPOSITION_POLICY.json> \
  --discovery-claims-fixture <SYNTHETIC_DISCOVERY_CLAIMS.json> \
  --expected-core-policy-digest <SHA256> \
  --expected-core-claims-digest <SHA256> \
  --expected-discovery-policy-digest <SHA256> \
  --expected-discovery-claims-digest <SHA256> \
  --output <NEW_RECEIPT.json>
```

Activation is a separate one-way local transaction:

```bash
optima chain-activate-incentives \
  --intake-db chain_intake/intake.sqlite3 \
  --netuid <NETUID> \
  --network <NETWORK_OR_WSS_URL> \
  --core-policy <CORE_POLICY.json> \
  --composition-policy <COMPOSITION_POLICY.json> \
  --approval <APPROVAL.json> \
  --expected-approval-digest <SHA256>
```

Preflight reopens and binds:

- the exact core and composition policy bytes;
- an independent approval and activation bundle;
- chain genesis, netuid, finalized block/hash, and equal intake cursor;
- one retained arena, stack, target catalog, campaign, and complete family roster;
- approved finalized membership and reserve hotkey;
- audit-control, production-canary, and residual-risk acceptance digests; and
- quiescent intake with no incompatible V1 publication or retained legacy debt.

Activation constructs no wallet and signs no chain transaction. The atomic schema
cutover is also a rollback fence: an older runtime cannot reopen the activated
database as legacy authority.

## V2 publication

`set-debt-weights` publishes the earliest due boundary:

```bash
optima set-debt-weights \
  --intake-db chain_intake/intake.sqlite3 \
  --netuid <NETUID> \
  --network <NETWORK_OR_WSS_URL> \
  --wallet default \
  --hotkey validator \
  --refresh-blocks <BLOCKS>
```

For each boundary it:

1. reopens the active policy and exact finalized economic state;
2. creates or resumes the earliest gapless projection;
3. binds that projection to the signer-facing hotkey/UID vector;
4. journals signing and submission;
5. grades exact finalized chain readback; and
6. debits claims only after confirmation and intake-cursor catch-up.

A restart, dry run, SDK success response, or projection build never reduces debt.
A missed boundary remains ahead of later boundaries, and confirmed catch-up is
rate-limited to at least one full policy cadence after the prior confirmation.

## Operational invariants

- Preserve one writer for a validator/database authority.
- Back up SQLite with WAL-aware tooling and retain every referenced evidence root.
- Treat policy, campaign, reserve, membership, and activation digests as immutable.
- Halt on unexplained membership departure or UID reassignment; do not rewrite a
  historical boundary from a later metagraph snapshot.
- Never repair a hold by deleting journal rows, editing debt, or replacing evidence
  with a summary.
- Keep evaluator containers separate from wallet and signer authority.

## Current evidence limits

- The V2 implementation has no retained live activation or live debt-publication
  receipt.
- Registered discovery promotion remains unsupported.
- Signer-free shadows exercise arithmetic and state binding; they authorize no
  activation or chain mutation.
- Synthetic load sweeps establish accounting behavior, not miner equilibrium,
  token value, GPU performance, or production readiness.
- Production still requires exact campaign/reserve manifests, retained historical
  membership authority, independently graded review and invalidation authority,
  and accepted production audit-canary evidence.

See [Current status](state-of-record.md) for the maintained evidence ledger and
[Settlement and weights](../validator-guide/settlement-and-weights.md) for the
operator flow.

## Source anchors

- [Legacy economics](https://github.com/latent-to/cacheon/blob/main/optima/economics.py)
- [Finite-debt arithmetic](https://github.com/latent-to/cacheon/blob/main/optima/finite_debt.py)
- [Incentive composition](https://github.com/latent-to/cacheon/blob/main/optima/incentive_composition.py)
- [Activation authority](https://github.com/latent-to/cacheon/blob/main/optima/chain/incentive_activation.py)
- [V2 publication](https://github.com/latent-to/cacheon/blob/main/optima/chain/debt_publication.py)
- [V1 publication](https://github.com/latent-to/cacheon/blob/main/optima/chain/weights.py)
- [CLI](https://github.com/latent-to/cacheon/blob/main/optima/cli.py)
