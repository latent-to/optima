# Incentives

Optima converts independently reproduced marginal improvements into weight
projections. A local benchmark, a clean bundle, a qualification attempt, and a
crown are different authorities. Only transactional settlement can create a
reward claim, and only a confirmed weight publication can realize the
corresponding projection.

## Check the active policy before estimating a reward

The repository contains two deliberately separate reward generations:

- **V1 standing rewards** assign the active crown for each registered target a
  decaying share of the current weight vector. This is the only publisher with
  retained live exercise.
- **V2 finite debt** issues a bounded claim when a registered-target crown
  settles, then pays that principal down over later confirmed epochs. The
  one-campaign MiniMax-M3 policy, activation transaction, and publisher are
  implemented but inactive. There is no live V2 activation or publication
  receipt.

An operator announcement must identify the active policy digest, chain scope,
arena, target catalog, reserve, and publication cadence. Do not combine V1 and
V2 formulas or treat repository support for V2 as evidence that a deployment
has activated it. Activation disables legacy V1 projection for that database;
the two publishers must not run concurrently.

## Conditions shared by both generations

A registered-target proposal can affect incentives only after all of these
conditions hold:

1. finalized intake binds the proposal's arrival and content identity;
2. immutable publication and target resolution bind the selected delta;
3. every non-crown screen passes;
4. a complete v3 qualification passes: resident B/C/B′, conditional C′/B″,
   registered eager audit A, then pristine T;
5. a fresh independent attempt reproduces the same proposal and authority;
6. settlement reopens the retained evidence and resolves priority and overlap;
7. the candidate settles as the crown for one canonical reward family; and
8. the miner hotkey remains eligible under the active metagraph policy.

The lower of the two reproduced speedups is the settled measurement. Failed,
held, neutralized, copied, discovery-only, or unreproduced proposals create no
registered-target reward authority.

Atomic targets and singleton targets are separate reward families. A registered
atomic crown suppresses only the overlapping singleton families declared by the
target catalog. A bundle cannot create a new family or split one improvement
across invented target identifiers.

## V1 standing rewards

V1 retains exactly one active standing claim for every active target in every
registered arena. Its policy converts conservative settled speedup and claim
age into decaying standing credit. The exact conversion, flooring, and
normalization rules are defined only in
[Legacy V1](../reference/emissions-policy.md#legacy-v1).

The policy normalizes all live standing credit into the standing pool. If live
legacy discovery claims exist, they are normalized separately into the configured
discovery pool; otherwise that capacity remains in the standing pool. A V1 crown
therefore represents a decaying relative share, not fixed principal and not a
fixed token amount.

The complete V1 projection fails closed when an active target lacks exactly one
compatible crown, a live claimant has left the metagraph, active targets overlap,
the stack and catalog disagree, or all standing credit has decayed to zero.

Before the first crown, an operator may use the explicit all-uncrowned bootstrap
path to project the full vector to one registered burn hotkey. The implementation
refuses that path as soon as any crown, active reward claim, or V2 activation
exists. This is an operator bootstrap mechanism, not miner income.

The pure projection is implemented in
[`optima/economics.py`](https://github.com/latent-to/cacheon/blob/main/optima/economics.py).
Journaled chain publication is implemented separately from settlement.

## V2 finite-debt issuance

V2 pays for the independently reproduced improvement over the prior frontier.
A settling crown issues finite principal; later crowns do not erase that debt,
and an old crown does not receive a perpetual royalty.

### Improvement units

V2 represents conservative settled speedup in log-relative fixed-point units.
This makes base accounting path-independent apart from the policy's defined
flooring and elapsed-time multiplier. The unit is an accounting denomination,
not an acceptance threshold: an improvement must still clear the validator's
calibrated marginal bar and reproduce. See
[Registered-CROWN principal](../reference/emissions-policy.md#registered-crown-principal)
for the normative conversion and flooring order.

### Family clock

For a crown after an earlier accepted crown in the same reward family, V2
applies the policy-defined multiplier for the finalized block gap. A family
without retained or activation-seeded clock authority uses the policy's
first-crown treatment. An activation may seed a clock from a retained
pre-activation crown, but the seed creates no retroactive principal. The exact
clock function and bounds live in
[Registered-CROWN principal](../reference/emissions-policy.md#registered-crown-principal).

Only an accepted registered-target crown advances the family clock. Waiting for
the bonus is not risk-free: it delays payment and lets another qualifying
proposal take priority, while the multiplier remains bounded.

### Campaign-sized principal

Every reward family maps once to a validator-owned model campaign. Campaign
share sizes claim principal; it is not an epoch payout silo. The selected
activation roster is a single immutable MiniMax-M3 campaign. Although the pure
arithmetic supports another research shape, the current activation and durable
store reject it. Model rotation, another active campaign, and successor
activation require a new protocol. The normative roster constraints and
principal arithmetic are in
[Campaign policy](../reference/emissions-policy.md#campaign-policy) and
[Registered-CROWN principal](../reference/emissions-policy.md#registered-crown-principal).

Adding unused target families to the sole campaign changes existing principal
by zero.
Each family still has an independent frontier and clock, while actual wins in
more families create more total debt.

The issuance arithmetic is implemented in
[`optima/finite_debt.py`](https://github.com/latent-to/cacheon/blob/main/optima/finite_debt.py).

## V2 epoch composition

At each confirmed composed epoch, reviewed-discovery and registered-CROWN
claims draw from separate policy-defined class allocations. Each class splits
its allocation pro rata by remaining principal using claim-digest
largest-remainder ordering before hotkey aggregation, so one hotkey owning
several claims cannot alter the class's rounding order. The allocation formula
and constants are defined in
[Epoch composition](../reference/emissions-policy.md#epoch-composition).

Unused discovery capacity returns to registered-claim capacity. Capacity left
after both classes' actual payouts goes to the reserve.

A registered claim expires under the active lifecycle policy. Deregistration
or a finalized runtime invalidation can forfeit the unpaid balance. Principal
always conserves as paid, forfeited, and remaining units; expiry does not mint
a separate reserve transfer. The normative lifetime is part of
[Registered-CROWN principal](../reference/emissions-policy.md#registered-crown-principal).

The projection is pure and cannot debit debt. `set-debt-weights`:

1. reopens or constructs the earliest gapless due boundary;
2. binds the economic projection to the signer-facing sparse vector;
3. journals submission intent and outcome;
4. obtains an exact finalized chain readback;
5. confirms the publication only when that readback matches; and
6. debits principal only after retained finalized intake reaches the readback.

A dry run, projection build, SDK success, restart, or unconfirmed submission
does not pay a claim. A delayed boundary remains first in sequence, and later
boundaries cannot skip it.

The composed arithmetic is implemented in
[`optima/incentive_composition.py`](https://github.com/latent-to/cacheon/blob/main/optima/incentive_composition.py);
the restart-safe publication authority is in
[`optima/chain/debt_publication.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/debt_publication.py).

## Discovery rewards

Discovery proposals do not own a registered target or family clock.

Under legacy V1, a retained discovery claim is non-renewable, expires under the
policy lifetime, and shares only the configured discovery pool.

The selected V2 composition intends one qualified discovery win to take one of
two mutually exclusive paths:

- `registered_promotion`: issue no discovery debt, register a reviewed target,
  then require a fresh component submission, qualification, reproduction, and
  crown; or
- `bounty_only`: issue one policy-bounded, non-renewable discovery claim.

The durable implementation currently supports review-pending retention and
`bounty_only`; it rejects `registered_promotion`. The promotion path still
needs typed transport, target registration, fresh-crown linkage, and one
cross-lane work identity. “Never both for the same work” is therefore policy
intent, not an end-to-end property of the inactive implementation.

The bounty lifetime begins at the retained qualified-win block, not at later
review. Review delay consumes the policy-defined window, and review at or after
expiry cannot mint principal. The bounty has no campaign share, elapsed-time
bonus, renewal, or standing title. Exact bounty capacity and lifetime are
defined in
[Reviewed discovery](../reference/emissions-policy.md#reviewed-discovery).

See [Discovery lane](discovery-lane.md) for the proposal ABI.

## Activation boundary

`chain-activate-incentives` is a wallet-free local cutover. It requires canonical
core and composition policies plus an independently pinned approval. Before one
transaction activates the policy and raises the durable schema floor, preflight
must reproduce:

- one complete MiniMax-M3 reward-family roster from exactly one retained arena;
- the arena, evaluation-stack, catalog, and finalized-membership digests;
- reserve membership at the approved finalized metagraph;
- equal approved and retained finalized intake cursors; and
- the independently approved audit-control, canary, and residual-risk bindings.

Activation also requires quiescent pre-cutover intake, no open debt, no retained
legacy discovery state, and no legacy publication state in the activation
database. Existing V1 publisher databases are not silently rewritten or erased
to force a cutover.

The implementation boundary is in
[`optima/chain/incentive_activation.py`](https://github.com/latent-to/cacheon/blob/main/optima/chain/incentive_activation.py).
The [state of record](../reference/state-of-record.md) distinguishes implemented
authority from receipts that actually exist.

## What miners should optimize for

- Optimize one canonical registered target and publish the narrowest honest
  capability domain.
- Submit a reproducible improvement when it is ready. A bounded time bonus does
  not compensate for pre-emption or delayed payment.
- Do not split one gain to manufacture credit. Log units remove the base split
  advantage, and each accepted piece resets its own family clock.
- Do not budget from a microbenchmark or a first qualification PASS. Only the
  lower independently reproduced speedup reaches settlement.
- Do not assume that a discovery label earns a bounty. Review and the active
  reward generation determine the disposition.
- Keep the claimant hotkey registered while it owns a standing reward or unpaid
  finite principal.
- Model collection under plausible concurrent wins. Finite expiry bounds
  liability; it does not guarantee that every issued unit is paid before expiry.

The deterministic one-campaign load study is reported separately in
[Incentive load validation](../results/incentive-load-validation.md).

## Denomination and non-promises

Optima accounts in confirmed validator weight-ppm epochs. It does not promise a
fixed amount of TAO, alpha, fiat value, or realized validator influence. Token
emission depends on Bittensor consensus, subnet state, chain mechanics, and the
validator's realized position.

Rental-cost and collection calculations are sensitivity analyses, not price
forecasts or payout promises. Always bind an estimate to the exact active
policy digest and retained publication evidence.
