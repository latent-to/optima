# Running Optima on the Bittensor testnet

How the chain layer (`optima/chain/`) runs against a real subnet, and the exact
loop validated on testnet netuid 307 (2026-07-08). See `docs/SUBNET_BLUEPRINT.md`
for the architecture this implements; this doc is the operator's runbook.

## What rides the chain

The chain is the trust anchor ONLY (commitments + weights). A miner's whole
on-chain footprint is one commitment carrying
`{"v":1,"h":"<sha256 content hash>","u":"<fetch url>"}` posted via the chain's
NATIVE timelock commit-reveal (`set_reveal_commitment`):

- the payload is drand-encrypted until the reveal round — nobody (validators
  included) can read the bundle URL before the reveal, so there is no
  pre-evaluation copy window;
- the reveal block is the consensus anti-copy priority timestamp — the validator
  replays finalized reveals in chain order, so "earliest committer wins"
  is decided by the chain, not by any off-chain clock;
- chain-side caps: 1024 bytes per timelock payload, ~3100 payload-bytes per
  hotkey per epoch (each commit costs `max(100, bytes)` of that budget).

Everything else happens on the validator. `FinalizedIntakeStore` records finalized
priority, private fetch, immutable worker publication, copy disposition, arena-screen
receipts, qualifications, reproduction state, settlement, stack state, and weight-
publication journal entries in SQLite.

## Prerequisites

- `bittensor` SDK (validated: 10.3.2; note it pins `bittensor-drand<2.0.0` — a
  drand 2.x wheel breaks the import). Run `optima chain-compat` after ANY SDK
  bump — it introspects the installed SDK for every method we call.
- A wallet (`btcli wallet create`). Key roles, verified against SDK source:
  - registration (`burned_register`): coldkey + hotkey
  - staking: coldkey
  - **commitments and set_weights: hotkey only** — the recurring loop never
    touches the coldkey. Keep the coldkey off the validator box.
- Testnet TAO (the faucet is Discord-manual these days).

## The loop, step by step

```bash
NET="wss://test.chain.opentensor.ai:443"   # pass the URL explicitly: the SDK's
                                            # 'test' alias is a DIFFERENT host
                                            # (test.finney.opentensor.ai)

# 0. one-time: join the subnet (validator hotkey, and each miner hotkey)
optima chain-register --netuid 307 --network "$NET" --wallet default --hotkey default

# 1. miner: package a bundle -> tar.gz + the content hash that will be committed
optima chain-package examples/miner_silu_torch --out hosted/bundle.tar.gz

# 2. miner: host the tarball anywhere fetchable (https://...; file:// for a
#    same-machine dev loop), then commit hash+URL on-chain (timelock)
optima chain-submit examples/miner_silu_torch --url https://example.com/bundle.tar.gz \
    --netuid 307 --network "$NET" --wallet default --hotkey miner
#    (--dry-run prints the exact payload without signing)

# 3. inspect the subnet: neurons, permits, revealed submissions
optima chain-status --netuid 307 --network "$NET" --wallet default --hotkey default

# 4. validator: finalized intake only (single pass with --once; daemon without)
optima chain-validate --netuid 307 --network "$NET" \
    --intake-db chain_intake/intake.sqlite3 \
    --private-root chain_intake/private --publication-root chain_intake/worker \
    --intake-only --once
```

One intake pass reads finalized reveals in consensus order, fetches each new artifact
with archive and extraction limits, **re-hashes the extracted tree against the committed
hash**, records copy priority, and creates an immutable worker publication. The SQLite
cursor and per-stage state make the pass restart-safe; miner failures, transient transport
faults, and validator holds remain distinct dispositions.

### Qualification and settlement

Production validation is not selected by a miner-supplied command or module path.
Deployment code calls `cmd_chain_validate(..., arena_registry=...)` or
`run_validator(...)` with a trusted `ArenaServiceRegistry` and an `arena_id`. The
registered service binds the runtime, model, topology, workload mixture, capacity and
retry policy, screen policy, and qualification-plan factory. Its non-crown screen runs
the fixed static/build/ABI/graph/abbreviated-serving stages before promotion.

Full qualification uses isolated graph-on/audit-free charged B/C/B′ roles, a mandatory
separate eager/untimed audit role whose exact slot×TP-rank receipts are host-regraded into
a typed witness, and pristine-T authority. This wiring is implemented and CPU/mock-covered
but not yet GPU-qualified; unauditable attention slots fail closed. One passing report is
stored as `reproduction_pending`; a second pass must use independent authority and selection
evidence while matching the same arena, target, delta, incumbent and candidate stack
identities. Settlement receives only the paired candidate and uses the lower speedup. The
currently wired weight command below is the retained **legacy-V1**
standing-credit/discovery policy. It does not exercise or activate finite debt:

```bash
optima set-weights --intake-db chain_intake/intake.sqlite3 \
  --netuid 307 --network "$NET" --wallet default --hotkey default \
  --half-life-blocks <N> --discovery-lifetime-blocks <N> \
  --discovery-pool-ppm <PPM> --refresh-blocks <N> --dry-run
```

**All-uncrowned bootstrap (burn-to-owner).** Before the first crown exists the
normal projection fails closed (a crown is a payment claim; stock cannot hold
one). The explicit operator policy for that world is `--burn-hotkey <ss58>`:
it projects the full pool to one designated registered hotkey (the subnet
owner's own burn registration, e.g. uid 0) so the miner emission share is not
left to squatters. The burn projection digest-binds the empty settlement
state and is refused the moment any crown, active reward claim, or activated
incentive composition exists — the transition to real payouts is forced, not
optional. The flow reuses the normal journal/CAS/reconcile/confirmation path,
with one explicit difference: the burn branch disables reconcile's pre-crown
submission gate (`require_current_crown=False`), because the burn vector is
crownless by construction; every other invocation keeps the gate. Two
operational notes: use `--dry-run` first, and pick the launch emissions-policy
parameters (`--half-life-blocks` etc.) deliberately — the store binds the
policy digest write-once, so the same parameters must be used for every later
V1 projection on that intake database.

The signer-free synthetic shadow surface constructs no wallet, accepts no intake
database, and cannot submit
weights:

```bash
optima chain-incentive-shadow \
  --network "$NET" --netuid 307 \
  --policy <canonical-synthetic-policy.json> \
  --claims-fixture <canonical-explicitly-synthetic-claims.json> \
  --expected-policy-digest <sha256-semantic-policy-digest> \
  --expected-claims-digest <sha256-semantic-fixture-digest> \
  --output <new-shadow-receipt.json>
```

The command reopens the exact finalized height/hash and historical metagraph
twice, requires every positive miner plus the reserve anchor to be registered,
and writes `submitted=false`. Its claim fixture is deliberately non-authoritative:
this tests projection and membership mapping, not settlement, a production family
catalog, a reserve-governance decision, or emission publication.

The 2026-07-18 feasibility pass ran this command with the now-historical D-012
family-share policy from the RTX pod against netuid
307: three synthetic claims mapped to three registered miners, the reserve mapped
separately, the vector conserved 900,000/100,000 ppm, and the retained receipt
recorded `submitted=false` (semantic digest
`a4006912ec3e34b98fe51031fe25864915e4a2d588209877c41a459a6094dcf3`).
That receipt exercised the historical registered-CROWN class only; it did **not**
exercise D-013 reviewed-discovery composition or D-015 campaign-policy bytes.

The selected but inactive D-013 composition has a distinct signer-free command for
explicitly synthetic registered-CROWN and discovery fixtures:

```bash
optima chain-incentive-composition-shadow \
  --network "$NET" --netuid 307 \
  --core-policy <canonical-synthetic-core-policy.json> \
  --core-claims-fixture <canonical-synthetic-core-claims.json> \
  --discovery-policy <canonical-synthetic-composition-policy.json> \
  --discovery-claims-fixture <canonical-synthetic-discovery-claims.json> \
  --expected-core-policy-digest <semantic-core-policy-digest> \
  --expected-core-claims-digest <semantic-core-fixture-digest> \
  --expected-discovery-policy-digest <semantic-composition-policy-digest> \
  --expected-discovery-claims-digest <semantic-discovery-fixture-digest> \
  --output <new-composed-shadow-receipt.json>
```

It also constructs no signer or wallet, accepts no intake database, reopens exact
finalized membership twice, maps each payout class separately, verifies the
1,000,000-ppm vector, and can only write `submitted=false` to a new output path.
The selected allocation is
`P_d=min(50,000, live discovery debt)`,
`P_c=min(900,000-P_d, live registered-CROWN debt)`, with the remainder to the
reserve.

Before D-015, the live RTX-pod feasibility run passed against testnet netuid 307 at finalized
block 7,586,146 with metagraph size 6. Its explicitly synthetic states produced
850,000 ppm registered-CROWN payout, 50,000 ppm reviewed-discovery payout, and
100,000 ppm reserve, conserving 1,000,000 ppm; the receipt recorded
`submitted=false`. Receipt semantic digest:
`3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`;
receipt-file SHA-256:
`ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`.
The command constructed no wallet. This is projection/membership feasibility
evidence for the historical D-012 core only—not review, settlement, publication,
debt-debit, D-015 policy, or activation authority. Do not reuse those fixture
digests as a campaign-policy activation shadow.

Opening an existing intake database migrates schema 4→5 by creating empty immutable
composition tables; it imports no historical CROWN or discovery claims and creates
no retroactive debt. Activation is now one wallet-free atomic command, but **do not
run it with placeholder or historical-shadow bytes**:

```bash
optima chain-activate-incentives \
  --intake-db chain_intake/intake.sqlite3 \
  --network "$NET" --netuid 307 \
  --core-policy <canonical-minimax-m3-core-policy.json> \
  --composition-policy <canonical-composition-policy.json> \
  --approval <independently-reviewed-activation-approval.json> \
  --expected-approval-digest <independently-recorded-approval-digest>
```

The command constructs no wallet and signs nothing. It accepts exactly one immutable
MiniMax-M3 campaign, reopens chain genesis and finalized ancestry, requires the intake
cursor to equal the approval's exact finalized block/hash, and atomically binds the
core policy, composition policy, approval, family roster, and reserve. The independently
pinned approval itself contains the arena, evaluation-stack, catalog, and finalized
membership digests. It also requires the production audit-control manifest digest, the
exact final B300 canary receipt digest, and a separate digest of the operator's explicit
acceptance of the three residual risks named in `FIDELITY.md`; these are retained in the
atomic activation row and public result. Preflight must reproduce the four retained
arena/chain facts. The complete approved roster
must match that retained arena/catalog, and the campaign ID is derived from the
arena/catalog/roster rather than chosen independently.

`chain-activate-incentives` binds the digest of an independently approved canary; it does
not parse or regrade the canary receipt. Rejecting a non-PASS or incomplete canary belongs
to the external reviewed evidence package and approval process. A matching digest proves
which evidence was approved, not that the evidence passed.

Activation also requires quiescent pre-cutover intake, no open V2 debt, no retained
legacy discovery row, and **a database that has never retained V1 projection/publication
state**. There is intentionally no automatic V1-history bridge in this command: launch
must use a fresh/quiescent activation database, while an existing V1 publisher database
remains an explicit fail-closed blocker. Do not delete or rewrite V1 history to force the
cutover. “Fresh” is literal: even a metadata-only `emissions_policy_digest` left by
building or dry-running a V1 projection makes that database ineligible. Success raises
the database schema floor 5→6 in the same transaction; an older
schema-5 runtime then fails closed. There is no live activation receipt yet.

The selected pure policy describes promotion-or-bounty, but the schema-5 settlement
surface currently retains qualified discovery wins as `review_pending` and can issue
only bounded `bounty_only`. It intentionally rejects `registered_promotion` until
typed `DiscoveryWinRecord`/`DiscoveryPromotion` transport, target registration,
fresh requalification/CROWN linkage, and cross-lane work identity exist. Do not
interpret “never both” as end-to-end same-work enforcement yet. The bounty lifetime
starts at the retained qualified-win block, so review delay consumes its 648,000
blocks and review at or after expiry cannot mint. A finalized durable API can mark
overdue pending wins `review_expired` and append `discovery_review_expired`, but no
operator scheduler for that reconciliation is established here.

D-015 tested the one/two-campaign arithmetic hierarchy offline. All 14 preregistered
screens passed: target-family counts caused zero principal dilution; the normal weekly
tape issued one full-sized 4.4%/5% claim for one campaign, or one half-sized claim in
each of two research campaigns (one full share aggregate), and paid fully with zero
expiry/outstanding; and the required marginal/overload controls were detected. The
operator activation path nevertheless accepts only one MiniMax-M3 campaign; two-campaign
cells, rotation, and successor activation are unsupported historical research.
Sustained simultaneous per-family wins were not the normal-tape assumption. Report
semantic digest:
`7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590`.
The raw sweep is a local-only experiment record. Current validation is 2,193
passed/19 skipped repository-wide. It is
deterministic control-plane and ROI evidence, not a new GPU or testnet run.

A tracked one-campaign supplement covers 64 launch/stress cells over 1/2/5/10
independently winning M3 families, 7/14/30/90-day cadences, and empty/saturated
discovery. Its semantic report digest is
`505fed4d40a6acc6bc92d6330170e8e2260a52e5f3099c22a6c0eb4b2308c672`.
This is deterministic accounting evidence and has no pod or live-chain receipt.

D-014 tested that win-anchored lifetime offline rather than through a chain command.
The 288-row matrix replayed byte-identically on arm64/Python 3.11 and
x86_64/Python 3.12. Every one of the 108 preregistered 0/1/7-day SLA rows paid
100% of discovery principal, left zero expiry/unissued principal, imposed at most
55,555 ppm instantaneous CROWN-capacity dilution, and had zero CROWN paid-fraction
regression versus zero delay. The 90/120-day cases issued no stale debt; 30/60/89
days were diagnostic only. Report digest:
`f0939d67241dffa49aac95c035c43dd7ea14b51eb2671fe106cb09347511b7ef`.
This is synthetic accounting sensitivity evidence, not testnet publication,
external review, activation, durable-state completion, or GPU-performance evidence.

The final hardened-source feasibility pass separately exercised the real intake
loop without a new winner. A fresh GPU-disabled `chain-validate --intake-only --once`
at finalized block 7,586,142 saw/reserved 19 retained entries, rejected five malformed
payloads, and ran no screens, decisions, settlements, or holds. Reopening the same
database at finalized block 7,586,144 produced zero work in every counter. Final pod
conformance passed 111 tests. No wallet, signature, candidate qualification, weight
publication, or GPU kernel was involved.

Likewise, `invalidate_finite_debt_family` can durably cancel a registered family's
open debt and reset its next-CROWN clock, but it accepts an external invalidation
digest and is not an independent runtime-invalidity authority.

Legacy V1 remains the only publisher exercised live. After a successful schema-6
activation, the implemented V2 publisher is:

```bash
# First inspect the exact next retained projection without signing.
optima set-debt-weights \
  --intake-db chain_intake/intake.sqlite3 \
  --network "$NET" --netuid 307 \
  --wallet default --hotkey default \
  --refresh-blocks <FINALITY_PLUS_REVEAL_MARGIN> --dry-run

# Remove --dry-run only after reviewing the retained projection.
optima set-debt-weights \
  --intake-db chain_intake/intake.sqlite3 \
  --network "$NET" --netuid 307 \
  --wallet default --hotkey default \
  --refresh-blocks <FINALITY_PLUS_REVEAL_MARGIN>
```

`set-debt-weights` always targets the earliest unclosed nominal boundary. It reuses
an in-flight journal after restart, binds the economic projection to the exact signer-
facing vector, confirms only from finalized chain readback, and debits claims only when
the intake cursor has reached that readback. If a boundary lands late, later boundaries
remain gapless and catch up no faster than one full policy cadence after the preceding
confirmation. The dry run prints a canonical JSON object containing the complete
economic projection and exact hotkey/UID/PPM signer binding; review and retain those
bytes before enabling the wallet-backed invocation. Exit 3 means a submission is still
pending, readback/intake work remains, or a confirmed row is refresh-due; it is not
permission to skip the boundary. `--reconcile-only --validator-hotkey <SS58>` reopens an in-flight
publication without constructing a wallet. The existing `--release-hold <REASON>` flow
is also available for an audited V2 hold. This command has no live receipt yet.

For the constrained first 1–2 week launch, keep the reserve and validator registrations
stable and monitor every positively weighted miner at each boundary. Claims live for 90
days, so ordinary expiry cannot arise in that window. If the reserve, validator, or a
positive recipient departs or changes UID, stop the publisher. There is deliberately no
launch-day remap or abandon protocol: do not skip the gapless boundary, rebuild an
in-flight vector against different UIDs, release the hold as if it fixed membership, or
rewrite the journal. General retained membership/departure reconciliation remains
post-launch work.

Before activation/mainnet operation, operators still need:

- exact MiniMax-M3 family and reserve manifests, followed by a fresh campaign-policy
  shadow and independently reviewed activation approval;
- retained membership-departure history rather than only the current metagraph snapshot;
- independently graded review and runtime-invalidation authority;
- the discovery promotion transport/linkage above;
- the production audit GPU canary plus explicit acceptance of in-process tampering,
  audit-role fingerprinting, and timed-workload fingerprinting; and
- owned-subnet operational authority, stake/permits, storage, and the actual activation.

Model rotation, a second campaign, and successor activation are unsupported future work,
not options in this launch runbook.

### Legacy-V1 publication reconciliation

**Commit-reveal timing:** `--refresh-blocks` is both the normal refresh cadence
and the bounded deadline for authoritative readback of an in-flight publication.
On a commit-reveal weights subnet it must exceed one full reveal tempo plus a
finality/readback margin; the chain's `weights_rate_limit` is not a safe substitute.
Netuid 307 has tempo 360, so use `--refresh-blocks 400`, not 100.

If a real publication remains `pending`, reconcile it with the same policy arguments
plus `--reconcile-only --validator-hotkey <VALIDATOR_SS58>`. This mode never constructs
a wallet or accepts a signer: it reopens the retained projection and records
`confirmed` only when the exact finalized sparse row and update chronology match. If
that historical landing is already older than the refresh cadence, the command prints
`refresh_due=True` and exits 3 after recording the valid confirmation. Run the ordinary
wallet-backed command once afterward; it builds a fresh current projection and submits
exactly one refresh. Other mismatch/submission-required cases fail without mutation.

If an undersized or interrupted deadline already placed the journal in `held`, first
verify the intended vector against finalized chain state, then run `set-weights` with
`--release-hold "verified late commit-reveal readback" --validator-hotkey
<VALIDATOR_SS58>`. Release reopens only the digest-chained retained head: it does not
rebuild off-pod qualification evidence, construct a signer, read weights, or submit.
Next use `--reconcile-only`. A refresh-due released row fails without mutation;
after resolving that result, run the ordinary wallet-backed command exactly once
to refresh. Otherwise do not emit a duplicate.

### Archiving legacy schema-v3 holds

An intake database migrated from schema v1/v2 may contain the exact fail-closed
reason `schema3_reproduction_required`: that historical single-PASS evidence is
never eligible to crown. After reviewing and backing up the retained row, remove
only its permanent queue/priority veto with the signer-free command below:

```bash
optima chain-archive-schema3-hold \
  --intake-db chain_intake/intake.sqlite3 \
  --network <NETWORK> --netuid <NETUID> \
  --reservation-id <RESERVATION_DIGEST> \
  --reason "reviewed legacy evidence; retained for audit only"
```

The command requires the exact migration hold plus consistent non-settled
candidate state at a finalized head. It terminalizes only the reservation;
qualification/candidate bytes remain retained and held, and cannot be released,
leased, or crowned. Ordinary holds and already-settled authority fail closed.

## Historical testnet receipts

The following receipts established the chain SDK, timelock, fetch, copy, restart, and
weight-publication behavior before the SQLite authority replacement. They do not claim a
joined production-arena run for the current implementation.

### Netuid 307 (2026-07-08) — legacy-V1 outcome

- Registered validator (uid 3) + miner (uid 4) hotkeys; burn ≈ 0.0005 tTAO each.
- Committed `miner_silu_torch` (178-byte payload, 5-block timelock) from the
  miner hotkey; the reveal appeared at block 7509374.
- One validator pass picked it up, fetched via `file://`, hash-verified,
  CPU-verified, crowned it, and produced `{miner_hotkey: 1.0}` targeting uid 4;
  a second pass did zero new work (idempotence).
- Subnet-307 caveats (someone else's unstarted subnet): its alpha pool rejects
  stake (`SubtokenDisabled`), and validator permits are the gate for actually
  landing `set_weights` — check `chain-status` for your permit before expecting
  weights to apply. Hyperparams there: tempo 360, commit-reveal weights ENABLED,
  weights_rate_limit 100 (skipped under commit-reveal). The control-plane
  readback deadline must therefore use 400 blocks, as described above.

### Netuid 307, round 2 (2026-07-10) — real legacy-V1 weights

Everything the 07-08 pass left dry-run, landed for real (the subnet owner
start-called 307 around 07-08, so `SubtokenEnabled` and permits now work there):

- Validator hotkey holds a validator permit (stake-weight via alpha stake; the
  permit is top-`max_validators` by stake-weight, recalculated per epoch).
- `chain-validate --once --margin 0` WITHOUT `--dry-run-weights`:
  `set_weights` SUBMITTED (the SDK auto-routes through drand commit-reveal on
  this subnet; the weights become visible in the metagraph after the reveal at
  the next epoch boundary — check `Subtensor.weights(netuid)`).
- Multi-miner emission split: a second miner hotkey committed a bundle for a
  second slot → per-slot settle → weights `{miner: 2/3, miner2: 1/3}` pushed.
- Copy demotion through the chain: the same bundle committed later by another
  hotkey was demoted (`copies=1`, never evaluated), and the loop skipped the
  redundant weight push (weights unchanged, not stale).
- Broken bundle through the chain: failed the gate chain (`passed=False`,
  score 0), crown unchanged.
- Daemon mode + mid-epoch restart: kill and restart → next pass `new=0`
  (EvalRecords suppress replay), weights stable.

Operational gotchas (learned here):

- **One active submission per hotkey.** The chain keeps a reveal history (last
  10) but the protocol takes each hotkey's LATEST reveal as its current
  submission. Two commits from one hotkey between validator passes = the
  earlier one is superseded unseen. Stagger commits across passes.
- **bittensor's import reconfigures global logging** — it sets pre-existing
  loggers to CRITICAL, which silenced daemon mode entirely (the ledger advanced
  while the log stayed empty). `chain-validate` now takes ownership of the
  `optima.chain` logger subtree after connecting. A silent validator is an
  unoperable validator; keep this in mind for any new entry point.
- **dTAO staking slippage is real.** 307's pool was alpha-drained (constant
  product: `alpha_in ~0.01`); 100 tTAO bought ~0.005 alpha — the measured
  numbers match `x·y=k` exactly. Check the pool before assuming stake buys
  stake-weight at par on mainnet.

## Threat-model notes

- The trusted controller never imports miner code or loads candidate native artifacts.
  Candidate import, hermetic native compilation, engine construction, and execution occur
  in a validator-owned OCI worker with no network egress and bounded mounts/protocols.
- Chain keys remain on the control plane. Arena workers receive only immutable,
  hash-complete publications and validator-owned plans; they cannot set weights.
- A miner lying about the hash (committing X, hosting Y) is rejected at the
  re-hash step and recorded, so the lie is not retried every pass.
