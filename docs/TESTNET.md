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

Full qualification uses the isolated B/C/B′/pristine-T authority. One passing report is
stored as `reproduction_pending`; a second pass must use independent authority and
selection evidence while matching the same arena, target, delta, incumbent and candidate
stack identities. Settlement receives only the paired candidate and uses the lower
speedup. Weight submission is a separate control-plane reconciliation:

```bash
optima set-weights --intake-db chain_intake/intake.sqlite3 \
  --netuid 307 --network "$NET" --wallet default --hotkey default \
  --half-life-blocks <N> --discovery-lifetime-blocks <N> \
  --discovery-pool-ppm <PPM> --refresh-blocks <N> --dry-run
```

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

### Netuid 307 (2026-07-08)

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

### Netuid 307, round 2 (2026-07-10) — real weights

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
