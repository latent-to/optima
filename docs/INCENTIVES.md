# Optima incentives

> **Activation status:** this is the selected but inactive V2 composition. Launch is
> exactly one immutable MiniMax-M3 campaign at 100% claim sizing. Target families keep
> independent frontiers and clocks without diluting one another's issued principal;
> open claims still share payout capacity. Historical two-campaign D-015 cells remain
> arithmetic research only: a second campaign, model rotation, and successor activation
> are unsupported by the live generation.
>
> The schema-2 `optima.finite-debt.v2` arithmetic, D-013's separately reviewed
> discovery-bounty arithmetic, and schema-5 pre-activation durable state are implemented.
> `chain-activate-incentives` provides one wallet-free atomic cutover: exact core,
> composition, and independently approved bytes are bound at the exact finalized intake
> cursor. It matches the complete approved family roster to exactly one retained arena,
> derives the campaign from that arena/catalog/roster, verifies reserve membership at the
> approved finalized metagraph, requires the independently pinned approval's arena/stack/
> catalog/membership digests to match, retains them in the activation row/event, and raises
> the durable schema floor from 5 to 6 in the same transaction.
> `set-debt-weights` provides restart-safe gapless projection, signing, finalized
> readback, confirmation, and only-then debt debit, including rate-limited catch-up.
> Neither command has a live receipt, and V2 has not been activated.
>
> D-012 and its registered-CROWN testnet shadow remain immutable historical evidence
> for the superseded family-share policy; the composed shadow likewise used historical
> core bytes and explicitly synthetic claims with `submitted=false`. They supplied no
> wallet, review, settlement, D-015 publication, or activation authority. Before launch
> operators still need exact MiniMax-M3 family and reserve manifests plus a fresh shadow,
> retained membership-departure history, independently graded review and runtime-
> invalidation authority, durable discovery-promotion linkage, the production audit GPU
> canary plus explicit acceptance of its in-process/audit-role/timed-workload residuals,
> and the actual activation/mainnet operations.

For registered-family rewards, this curve pays for an independently reproduced
improvement over the current validator-controlled frontier. A submission earns
nothing merely for being fast, novel, or the newest upload. It must pass
correctness and fidelity, beat the exact incumbent under the noise-aware scorer,
pass a second independent reproduction, and settle as a `CROWN` for one canonical
reward family. The separately reviewed discovery boundary is described below.

## The short version

Once this rule is activated, a settling crown creates a finite claim whose size is
determined by:

1. the multiplicative throughput improvement over the prior frontier;
2. a small bonus for how long that reward family had gone without an accepted
   crown; and
3. the active model campaign's claim-sizing share: the sole launch campaign uses
   100% sizing.

The claim is paid down from later confirmed weight epochs. It cannot receive more
than its issued principal and expires after 90 days. A tiny lone claim is therefore
paid only what it earned; it is never normalized into the entire emission vector.
Adding target families within MiniMax M3 changes neither an existing family's
principal nor its clock. More actual wins still create more total debt, so target
count does not make the system economically free.

The selected pure policy intends a separately reviewed discovery win to take exactly
one economic path: promotion into a registered target followed by fresh
requalification/CROWN, or one bounded finite bounty. Durable schema-5 currently
implements review-pending retention plus the `bounty_only` branch and deliberately
rejects `registered_promotion`. Until typed promotion transport and cross-lane work
identity exist, “never both” is policy intent, not end-to-end same-work enforcement.

## Exact launch curve

The conservative settled speedup `s` is the slower of the two independently
passing measurements. Production represents its improvement in parts per
million of a 1%-log unit:

```text
G_ppm = floor(1,000,000 * ln(s) / ln(1.01))
```

`G_ppm = 1,000,000` is one multiplicative 1% improvement. Log units make
compounding path-independent: two successive 1% improvements create the same
base credit as one 2.01% improvement, apart from the documented fixed-point
flooring and any elapsed-time bonus.

For every crown after the first one in its family, let `D` be the number of blocks
since the previous accepted crown in that same family. The time multiplier also
uses fixed-point integer arithmetic:

```text
M_ppm = 1,000,000 + floor(100,000 * D / (D + 648,000))
```

With no retained prior-family clock, the first crown has `M_ppm = 1,000,000`;
chain age alone does not create a windfall. An activation may explicitly seed a
family clock from a retained pre-activation accepted crown, so the first
post-activation claim can legitimately have a bonus; every seed is part of the
activation authority and creates no retroactive principal. Every accepted crown
then resets the family clock. Failed,
rejected, held, discovery-only, copied, or arbitrary submissions do not.

The curve is deliberately mild:

| Time since prior crown | Multiplier |
|---:|---:|
| 0 days | 1.000x |
| 30 days | 1.025x |
| 90 days | 1.050x |
| 180 days | 1.0667x |
| 365 days | 1.0802x |
| Infinite limit | 1.100x |

## Claim size

One confirmed payout epoch is 7,200 blocks (approximately one day) and contains
`E = 1,000,000` weight-parts. The minimum reserve is 10%, so the reference miner
claim pool is `C = 900,000` parts per epoch.

Every validator-owned reward family `f` maps exactly once to model campaign `c`.
If that campaign has claim-sizing share `B_c_ppm`, the exact issuance order is:

```text
F_c = floor(C * B_c_ppm / 1,000,000)
Q   = floor(F_c * k_ppm * G_ppm * M_ppm / 1,000,000^3)
```

Here `k_ppm = 1,000,000`: one campaign-share of a claim-pool epoch per 1%-log
unit. The pure schema-2 arithmetic can represent these research rosters:

- launch: claims in one campaign use `1,000,000 ppm` sizing (MiniMax M3);
- historical expansion research: claims in either of two campaigns use
  `500,000 ppm` sizing; and
- never three or more under this policy version.

The selected activation approval and durable store accept only the first roster:
one immutable MiniMax-M3 campaign. Two-campaign arithmetic is not an operator
configuration and cannot be activated by this generation.

Campaign and family mappings are explicit, content-addressed validator policy.
Every family maps once, every campaign has at least one family, and submissions
cannot choose or infer their own campaign. Adding 1, 10, or 100 unused target
families to a campaign changes claim principal by exactly zero.

For launch intuition with one 100% campaign:

- a first 1% CROWN issues `900,000` units, or `0.9` full-vector days;
- the same 1% CROWN after a 90-day family drought issues `945,000` units;
- the measured 4.4% example issues exactly `3,894,697` units, about `3.895`
  vector-days; and
- a first 5% CROWN issues `4,413,033` units, about `4.413` vector-days.

For historical two-campaign sensitivity only, each number is approximately half:
a first 1% CROWN is `450,000`, 4.4% is `1,947,348`, and 5% is `2,206,516` units.
Those figures are not an activatable launch promise. Families inside the same
campaign do not divide these numbers.

The campaign share sizes principal; it is not a hard per-epoch payout silo. All
open registered-CROWN claims share the global post-discovery pool pro rata. In the
historical equal-arrival two-campaign research, the claim sizes gave both campaigns
equal economic weight. Under excess load,
claims wait and can expire rather than manufacturing additional emissions.

The launch roster is immutable. The current generation deliberately fences every
later activation, so rotating from MiniMax M3 or expanding to two simultaneous
models is unsupported future work. It requires a separately designed, reviewed,
and implemented successor protocol after open debt is resolved; it is not a launch
configuration or prerequisite for activating the one-campaign generation.

An improvement smaller than 1% can still earn proportionally if it clears the
validator's measured confidence bar and reproduces. The 1% figure is an accounting
unit, not a minimum accepted speedup.

### D-015 feasibility and rental economics

D-015 preregistered the campaign hierarchy before opening its results. All 14
screens passed. One, ten, and one hundred target-family catalogs had zero principal
difference; sibling family clocks had zero multiplier or principal cross-talk. The
original normal tape paid 100%, but it rotated one aggregate campaign CROWN among
families. It therefore did not model several families independently winning on the
same cadence. Its report semantic digest remains
`7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590`.

A tracked supplement now closes that evidence gap with 64 one-campaign cells: 1,
2, 5, or 10 independently winning MiniMax-M3 families, each producing a 4.4% CROWN
every 7, 14, 30, or 90 days, under empty and saturated discovery. It separately
models the announced 14-day launch and a 365-day sustained-pressure control, then
drains every issued claim through its full 90-day life. In the launch window, all
rows paid 100% except the deliberately harsh ten-family weekly row under saturated
discovery, which paid 99.0211%. Up through five independently winning weekly
families paid fully. A same-day burst of nineteen claims also paid fully under
saturated discovery; twenty paid 98.2104%.

The sustained control is intentionally less flattering and prevents the launch
result from becoming a permanent promise. With saturated discovery, weekly 1/2/5/10
family streams collected 100%, 85.6341%, 37.1198%, and 18.5635% respectively.
Finite expiry prevents unbounded liability, but miners cannot assume full collection
if many families keep producing large wins indefinitely. The tracked config,
simulator, report, and CI replay are in
[`evidence/incentives/D015_LAUNCH_LOAD_REPORT.md`](../evidence/incentives/D015_LAUNCH_LOAD_REPORT.md);
the supplemental semantic report digest is
`505fed4d40a6acc6bc92d6330170e8e2260a52e5f3099c22a6c0eb4b2308c672`.

For a `$1,000-$1,500` optimization campaign and a hypothetical 25% success
probability, a one-campaign 4.4% win at the measured 100% launch collection rows
breaks even when a full Optima vector-day is worth `$1,027-$1,541`. The harshest
launch row's measured 99.0211% collection moves that range only to
`$1,037-$1,556`. A 5% win at full collection needs about `$906-$1,360`; a 1% win
needs about `$4,444-$6,667`. These are break-even sensitivities, not a token-price
forecast or payout promise. The report records ROI against every measured load
cell rather than reusing D-012's superseded 98.7203% collection factor. A second
model campaign is deliberately deferred and receives no launch-ROI claim here.

## How claims are paid

At each confirmed composed epoch, let `D_live` be open discovery principal and
`C_live` be open registered-CROWN principal, both measured in weight-ppm epoch
units. The two classes receive independent capacities:

```text
P_d     = min(50,000, D_live)
P_c     = min(900,000 - P_d, C_live)
reserve = 1,000,000 - P_d - P_c
```

- discovery claims share `P_d` pro rata by remaining principal, using
  claim-digest largest-remainder ordering;
- registered-CROWN claims independently share `P_c` by the same rule;
- claims are allocated before their hotkeys are aggregated, so one hotkey owning
  several claims cannot change a class's rounding order;
- unused discovery quota returns to registered-CROWN capacity; only capacity
  left after both classes' actual payouts goes to the explicit reserve.
  Discovery can consume at most 50,000 ppm, so saturated discovery reduces the
  otherwise 900,000-ppm CROWN capacity to 850,000 ppm;
- `set-debt-weights` debits a claim only after the exact projected vector has an
  exact finalized chain readback and the retained intake cursor has reached that
  readback. Rebuilding, dry-running, an SDK success, or a restart does not pay it.
  Boundaries remain a gapless nominal sequence; a missed or slow boundary is retained
  and caught up before later ones, with at least one full policy cadence between
  confirmations. This path is implemented but has no live receipt yet;
- registered-CROWN principal expires 648,000 blocks after its settlement. A discovery
  bounty's same-length window starts at the retained qualified-win block, not review:
  delayed review consumes the window and review at or after expiry cannot mint;
- deregistration
  forfeits the remaining balance; the resulting unused capacity in later epochs
  flows to the reserve. A finalized durable API can cancel one registered family's
  open debt and reset its next-CROWN clock for runtime invalidation, but the authority
  that decides and signs that invalidation remains external to the API and is not
  independently graded or wired.

The active kernel title and the payment claim are separate. Being superseded
does not erase already earned finite principal, but it also does not create a
perpetual royalty. Expiry places a hard bound on historical liability.

## Discovery-lane boundary

The D-015 campaign curve covers registered singleton or atomic reward-family CROWNs. A
cross-cutting discovery prototype does not automatically mint log-relative
principal or reset a family clock. D-013 selected the separate reviewed rule:

- discovery payout capacity is capped at 50,000 ppm per epoch;
- one award can issue at most one discovery-pool epoch of principal, exactly
  50,000 weight-ppm epoch units under the selected policy;
- the 648,000-block lifetime (about 90 days) starts at the retained qualified-win
  block, so delayed review consumes the available payout window and review at or
  after expiry cannot mint;
- it has no campaign share, family clock, elapsed-time bonus, renewal, or permanent
  title; and
- the pure disposition type expresses `registered_promotion` versus `bounty_only`
  as mutually exclusive policy choices.

Pure-policy promotion issues no discovery debt and intends a route to a registered
target followed by fresh qualification, reproduction, and CROWN. The durable store
does **not** execute that route today. It atomically retains a qualified discovery as
`ReviewPendingDiscoveryWin`, can later issue one unique bounded `bounty_only` claim,
and rejects `registered_promotion` until existing typed `DiscoveryWinRecord` and
`DiscoveryPromotion` authority are transported/reopened, the target is registered,
fresh requalification/CROWN is linked, and the same work has one identity across
discovery and registered lanes. Consequently the bounty ledger prevents duplicate
bounties for its retained identities, but it cannot yet prove that repackaged work
did not later earn through the registered lane.

The durable `expire_review_pending_discovery_wins` path terminalizes an unreviewed
win at its deadline as `review_expired` and appends `discovery_review_expired`.
Production still needs to schedule that finalized expiry reconciliation reliably.

The selected D-013 cell is
`8561028c943738da2fe622e5f5c9fd43ebec16fdd59feab3561de25fbfa450d9`;
the report digest is
`6bdfce26e4e6090e0dcc8814a636c665f28d1ff20945a09d43a9a90dc94151fc`.
The deterministic matrix contained 9 cells × 36 scenarios × 10 seeds = 3,240
rows and replayed byte-identically locally and on the RTX pod. In the selected
cell, non-departed principal paid 273,000,000/273,000,000 units, no such principal
expired or remained outstanding, and the worst run still paid 100%. Departed debt
forfeited/cancelled 9,000,000 units. Analytic and measured saturated CROWN-capacity
dilution was 55,555 ppm (5.5555%), while saturated tapes eventually paid 100% of
CROWN principal. These are synthetic accounting results, not evidence about miner
equilibrium, token value, or GPU performance.

### Review-delay sensitivity

D-014 held the selected policy fixed and varied only review delay and review-service
mode. Its deterministic matrix contained 8 delays × 3 modes × 4 scenarios × 3 seeds
= 288 rows and replayed byte-identically on arm64/Python 3.11 and
x86_64/Python 3.12. The preregistered review-SLA screen covered delays of 0, 1, and
7 days across every mode, scenario, and seed: all 108 rows passed. Within that
screen discovery paid 100%, expiry/unissued principal was zero, maximum
instantaneous CROWN-capacity dilution was 55,555 ppm, and CROWN paid-fraction
regression versus the zero-delay case was zero percentage points.

The 90- and 120-day cases issued no stale discovery debt, as required by the
win-anchored 90-day lifetime. Delays of 30, 60, and 89 days were diagnostic only;
they did not widen the preregistered review SLA. The report digest is
`f0939d67241dffa49aac95c035c43dd7ea14b51eb2671fe106cb09347511b7ef`.
This establishes deterministic synthetic accounting behavior under the tested
review delays. It does not provide an external review authority, activate V2,
publish weights, by itself prove durable-state hardening, or measure GPU
performance.

Before D-015, the signer-free composed shadow passed on testnet netuid 307 at
finalized block 7,586,146 with metagraph size 6. Its explicitly synthetic states
projected 850,000 ppm of registered-CROWN payout, 50,000 ppm of reviewed-discovery
payout, and 100,000 ppm of reserve, exactly 1,000,000 ppm total. It wrote
`submitted=false`; receipt semantic digest
`3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`,
receipt-file SHA-256
`ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`.
This remains read-only projection/membership feasibility evidence for the historical
D-012 family-share core. It constructed no wallet and provides no review,
settlement, publication, debt-debit, D-015 policy, or activation authority. A fresh
campaign-policy shadow belongs in the production activation change.

A separate multi-pass restart audit then exercised claim/event/cardinality
substitution cases. Reopen now derives exact paired qualification/evidence/CROWN
speed, principal, family clocks, discovery lifecycle, and all balance transitions
from their immutable journals before filtering status or allowing an upgrade. The
reproduced cases are retained regressions; before D-015, final results were 98/98
focused, 2,135 passed with 19 skips repository-wide, and 111/111 on the pod. Those
pod receipts are historical D-012/D-013/D-014 conformance, not D-015 validation.
This hardens the inactive state implementation; it does not close the production
authorities listed below.

The implementation retains a controller-supplied, content-bound `review_digest`,
but does not independently reopen or grade an external review system. That review
authority is therefore still an activation blocker, not an enforced governance fact.

## What miners should optimize for

- Submit a real, reproducible frontier improvement as soon as it is ready. The
  time bonus is capped at 10%, while waiting risks being pre-empted and delays all
  payout.
- Optimize one canonical target well. Packaging the same work into more bundles,
  hotkeys, singleton targets, or overlapping atomic targets cannot create extra
  reward families.
- Do not split a gain merely to manufacture credit. Log units remove the base
  split advantage; every accepted piece also resets the family clock.
- Do not count on noisy borderline measurements. A crown requires two independent
  passes and uses the lower measured speedup.
- Do not assume that labeling work “discovery” earns a bounty. The validator-owned
  review chooses the disposition, and the current implementation is inactive.
- The intended rule forbids bounty-plus-promotion for the same work, but do not treat
  that as fully enforced yet: promotion transport and cross-lane work identity are
  still missing.
- Keep the hotkey registered while a balance is open.

## What the numbers do—and do not—promise

Claims are denominated in confirmed validator **weight-part epochs**, not in a
fixed amount of TAO or alpha. Actual token emission also depends on Bittensor
consensus, the validator's realized influence, subnet state, and chain mechanics.
Optima can state exactly what weight share its accounting owes; it cannot promise
a token conversion rate it does not control.

Each activation policy is versioned and content-addressed. Schema-5 migration creates
empty composition tables and no retroactive debt for legacy crowns or discovery
awards. `chain-activate-incentives` accepts exactly one independently approved,
immutable MiniMax-M3 roster and atomically binds its core and composition manifests,
chain scope, finalized block/hash, equal intake cursor, family roster, and reserve. It
requires that complete roster to identify exactly one retained evaluation arena, derives
the campaign identity from the retained arena/catalog/roster, checks reserve membership
at the approved finalized metagraph, and reproduces the approval-pinned arena/stack/
catalog/membership digests before retaining them in the activation row and reward event.
The same transaction raises schema 5→6, creating a rollback fence that prevents an
older schema-5 runtime from reopening active state. Activation fails if any legacy
discovery row or legacy publication state is retained, if pre-cutover intake is not
quiescent, or if open debt exists. Existing V1 publisher databases are not bridged;
launch requires a fresh/quiescent activation database and must not erase V1 history to
force a cutover. Active composition disables the legacy automatic discovery award path.

The cutover and V2 publication machinery are implemented but have no live activation
or publication receipt. Production still must freeze the exact MiniMax-M3 family and
reserve manifests and run a fresh shadow, retain membership-departure history rather
than applying only a latest snapshot, bind independently graded review and runtime-
invalidation authority, complete promotion linkage, pass the production audit GPU canary,
explicitly accept its in-process/audit-role/timed-workload residuals, and perform the actual
activation/mainnet operations. Model rotation,
a second campaign, and successor activation are unsupported future work, not hidden
launch configuration. Parameter changes cannot silently rewrite existing claims.

For the retained technical authority and migration boundary, see
[EMISSIONS_POLICY.md](EMISSIONS_POLICY.md). For the evaluation gates that must be
passed before any claim exists, see [HOW_OPTIMA_WORKS.md](HOW_OPTIMA_WORKS.md) and
[MINER_GUIDE.md](MINER_GUIDE.md).
